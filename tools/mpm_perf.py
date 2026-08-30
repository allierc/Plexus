#!/usr/bin/env python
"""MPM WALL-CLOCK BENCHMARK: one spec, one implementation, one capture setting, one row.

    python tools/mpm_perf.py --sweep                       # write the config/si_material/si_perf_* sweep
    python tools/mpm_perf.py --run si_perf_wf_warp_cap --device cuda:0
    python tools/mpm_perf.py --table                       # collate every row written so far

WHY A SEPARATE TOOL AND NOT `tools/mpm_bench.py`. That one sweeps PARTICLE COUNT on a synthetic
spec and reports effective memory bandwidth, to answer "how far is this from the hardware". This
one holds the scene FIXED -- si_waterfall, 570,760 particles, n_grid 96, 13 substeps/frame -- and
sweeps the IMPLEMENTATION, to answer "which of the paths we ship is fastest, and by how much".
Different question, so a different tool rather than a fifth axis bolted onto the first.

WHAT IS TIMED, AND WHAT IS DELIBERATELY NOT. Frames `warmup+1 .. warmup+timed`, with a
`cuda.synchronize()` at each end so the number is GPU work and not a queue depth. Excluded:

  * process start (~11 s of torch + warp + pyvista imports),
  * hierarchy build,
  * the CUDA-graph capture itself, which happens at tick 1 and costs ~4 s at 945k particles --
    folding it into a 30-frame average once reported a 2.4x speedup as 1.3x,
  * rendering (`out_path=None`, no LiveMovie), and data recording (`save_data: false`).

so the row is the steady-state cost of the substep cycle and nothing else. `warmup` must exceed the
capture tick (1) and the allocator's growth phase; 40 is generous at this size.

WHY PEAK MEMORY IS IN THE TABLE. The torch gather materialises `(N, 27, D, D)` -- 554 MB per
intermediate at 570,760 particles -- and the loop-27 variant exists precisely to not. A speed
column alone would make the two look like a wash on an 80 GB A100 and hide the reason one of them
cannot run at 5M particles at all.

A ROW IS WRITTEN PER (spec, host, gpu), so the same sweep run on the devcontainer's L40S and on a
cluster A100 collates into one table instead of overwriting.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import socket
import sys
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG = os.path.join(ROOT, "config", "si_material")
ROWS = os.path.join(ROOT, "graphs_data", "si_material", "_perf")

# THE SWEEP. `impl` is what goes on mpm_strain / mpm_scatter / mpm_gather; None means the default
# torch operator. `gather` overrides the gather alone, which is the whole point of the
# torch_loop27 rows: strain and scatter stay on the torch path so the column isolates ONE kernel.
#
# THE `torch_` / `warp_` PREFIX IS THE PARTITION THE TABLE IS READ BY: every `torch_*` row is pure
# PyTorch and every `warp_*` row lowers at least one operator through NVIDIA Warp. `torch_loop27`
# carries the prefix for that reason -- it is a PyTorch gather, not a third backend.
VARIANTS = [
    # name suffix          strain/scatter  gather          grid_update  capture
    ("warp_cap",           "warp",         "warp",         None,        True),
    ("warp_eager",         "warp",         "warp",         None,        False),
    # THE GRID SOLVE IS THE FOURTH AXIS, and it is orthogonal to the other three: it is O(cells)
    # while they are O(particles), so it has to be swept separately or its share is invisible.
    # `warpgrid` is `warp` plus the fused two-kernel grid update -- one row against one row.
    ("warpgrid_cap",       "warp",         "warp",         "warp",      True),
    ("warpgrid_eager",     "warp",         "warp",         "warp",      False),
    ("torch_cap",          None,           None,           None,        True),
    ("torch_eager",        None,           None,           None,        False),
    ("torch_loop27_cap",   None,           "torch_loop27", None,        True),
    ("torch_loop27_eager", None,           "torch_loop27", None,        False),
]
# TWO SCENES, AND THE SECOND ONE IS NOT A BIGGER COPY OF THE FIRST. si_waterfall_5m is 8.8x the
# particles AND 8x the grid cells AND 2x the substeps, so it is where a per-particle intermediate
# stops fitting: the batched gather's `[N, 27, D, D]` temporary is 554 MB at 570k and 4.9 GB at 5M.
# Fewer timed frames there, because a torch row costs ~17 s each.
#                  base spec           prefix          n_frames  warmup
SCENES = {
    "wf":  ("si_waterfall",     "si_perf_wf_",  240, 40),
    "w5m": ("si_waterfall_5m",  "si_perf_w5m_",  50, 15),
}
N_FRAMES = 240
WARMUP = 40


class _Enough(Exception):
    """Raised from the per-frame hook once `timed` frames are done: the rest of the run is cost
    with no information in it."""


def _spec_path(name):
    return os.path.join(CONFIG, name + ".yaml")


def write_sweep(scene):
    base_name, prefix, n_frames, _ = SCENES[scene]
    base = yaml.safe_load(open(_spec_path(base_name)))
    made = []
    for suffix, impl, gather, grid, cap in VARIANTS:
        s = json.loads(json.dumps(base))          # deep copy through plain types
        name = f"{prefix}{suffix}"
        s["general"]["name"] = name
        s["general"]["n_frames"] = n_frames
        s["general"]["save_data"] = False
        for op in s["operators"]:
            if op["op"] in ("mpm_strain", "mpm_scatter"):
                op.pop("implementation", None)
                if impl:
                    op["implementation"] = impl
            elif op["op"] == "mpm_gather":
                op.pop("implementation", None)
                if gather:
                    op["implementation"] = gather
            elif op["op"] == "mpm_grid_update":
                op.pop("implementation", None)
                if grid:
                    op["implementation"] = grid
        for blk in s["schedule"]:
            if isinstance(blk, dict) and "steps" in blk:
                blk["capture"] = bool(cap)
        with open(_spec_path(name), "w") as f:
            yaml.safe_dump(s, f, sort_keys=False, default_flow_style=False)
        made.append(name)
    print(f"  {scene}: wrote {len(made)} specs from {base_name} into config/si_material/")
    for n in made:
        print(f"    {n}")
    return made


def run_one(name, device, timed, warmup, tag=None):
    import torch

    import plexus.operators                                          # noqa: F401
    import plexus.operators.mpm_warp                                 # noqa: F401
    try:
        import plexus.operators.mpm_loop                             # noqa: F401
    except ImportError:
        pass                                                    # torch_loop27 not on this checkout
    from plexus.engine import run
    from plexus.schema import load

    sim = load(_spec_path(name))
    n_part = None
    times, state = [], {}

    def hook(H, tick):
        nonlocal n_part
        if n_part is None:
            n_part = int(H.level("mpm_particle").n)
        if tick == warmup:
            torch.cuda.synchronize(device)
            state["t0"] = time.perf_counter()
            # PEAK IS READ TWICE, and the difference is not cosmetic. `max_memory_allocated` since
            # PROCESS START is what the run actually needs; the same statistic reset here is what
            # the STEADY STATE needs. They disagree wildly for a captured graph: capture happens at
            # tick 1, its private pool is allocated then, and a replay allocates nothing -- so the
            # window figure reported torch+capture at 0.38 GB against torch-eager's 1.81 GB and made
            # the 554 MB intermediates look like they had gone away. They had not; they were
            # allocated before the counter was reset. `peak_mem_GB` is therefore the since-start
            # number, and the window number rides along as `peak_window_GB` for the eager rows where
            # it means something.
            state["peak_pre"] = torch.cuda.max_memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
        elif tick > warmup:
            times.append(tick)
            if tick >= warmup + timed:
                torch.cuda.synchronize(device)
                state["t1"] = time.perf_counter()
                state["peak_win"] = torch.cuda.max_memory_allocated(device)
                state["peak"] = max(state["peak_pre"], state["peak_win"])
                raise _Enough()

    buf = io.StringIO()
    t_start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            run(sim, out_path=None, device=device, progress=False, on_frame=hook)
    except _Enough:
        pass
    log = buf.getvalue()

    if "t1" not in state:
        print(log[-2000:])
        raise SystemExit(f"{name}: run ended before frame {warmup + timed}")

    # substeps per frame, from the schedule the engine actually ran
    blk = [b for b in sim.schedule if isinstance(b, dict) and "steps" in b][0]
    n_sub = max(1, round(sim.dt / float(blk["substep_dt"])))
    wall = state["t1"] - state["t0"]
    ms_frame = 1e3 * wall / timed
    captured = "substep captured as a CUDA graph" in log
    refusal = ""
    if not captured:
        for line in log.splitlines():
            if "not captured" in line:
                refusal = line.split("(", 1)[-1].rstrip(")")
                break

    # A SPEC OUTSIDE THE SWEEP IS ITS OWN SCENE. The scaling benchmarks (si_bench_200m/500m/1b) are
    # one configuration each, not a sweep of implementations, so they group under their own name
    # instead of falling into an unlabelled bucket with the waterfall rows.
    scene = next((k for k, v in SCENES.items() if name.startswith(v[1])), name)
    known = scene in SCENES
    row = {
        "spec": name,
        "scene": scene,
        "base": SCENES[scene][0] if known else name,
        "variant": name[len(SCENES[scene][1]):] if known else "as written",
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(device),
        "device": device,
        "particles": n_part,
        "substeps_per_frame": n_sub,
        "frames_timed": timed,
        "warmup": warmup,
        "ms_per_frame": round(ms_frame, 3),
        "ms_per_substep": round(ms_frame / n_sub, 4),
        "ns_per_particle_substep": round(1e6 * ms_frame / (n_part * n_sub), 3),
        "peak_mem_GB": round(state["peak"] / 2**30, 3),
        "peak_window_GB": round(state["peak_win"] / 2**30, 3),
        "captured": captured,
        "capture_refusal": refusal,
        "startup_s": round(state["t0"] - t_start, 1),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "tag": tag or "",
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(ROWS, exist_ok=True)
    gpu_slug = row["gpu"].replace(" ", "_").replace("/", "_")
    with open(os.path.join(ROWS, f"{name}__{gpu_slug}.json"), "w") as f:
        json.dump(row, f, indent=2)

    print(f"\n  {name}  on {row['gpu']}")
    print(f"    {row['ms_per_frame']:.1f} ms/frame   {row['ms_per_substep']:.2f} ms/substep   "
          f"{row['ns_per_particle_substep']:.2f} ns/particle-substep")
    print(f"    peak {row['peak_mem_GB']:.2f} GB   capture {'YES' if captured else 'no'}"
          f"{('  -- ' + refusal) if refusal else ''}   startup {row['startup_s']:.0f} s\n")
    return row


def profile(name, device, frames, warmup):
    """Per-operator share of the substep, measured with CUDA events and no host syncs.

    WHY THIS EXISTS ALONGSIDE THE VARIANT TABLE. The table says which path is fastest; it cannot say
    whether the difference could ever have been large. If `mpm_grid_update` is most of the substep
    then swapping the gather is rearranging deck chairs, and that is a fact about where to spend the
    next week, not about this sweep.

    EVENTS, NOT `synchronize()` AROUND EACH CALL. Syncing per operator stops the CPU running ahead,
    which is itself most of what eager MPM costs -- the profile would then measure the profiler.
    Event pairs are queued on the stream and read once at the end, so the run proceeds normally.

    CAPTURE IS FORCED OFF: a captured graph is one launch and there is nothing inside it to
    attribute. The shares are therefore EAGER shares, and the note under the table says so.

    `set_device` IS LOAD-BEARING AND WAS THE FIRST BUG IN THIS FUNCTION. `Event.record()` records
    on `torch.cuda.current_stream()`, which is a stream on the CURRENT device -- cuda:0 unless told
    otherwise. Profiling a run on cuda:1 without setting the device therefore timed the gaps between
    events on an idle card, and returned a breakdown that summed to roughly the right total while
    being wrong in every row: it reported `mpm_grid_update` at 75% of the substep and the aggregate
    at 1%, when CUPTI showed the aggregate was 84% and the grid update a fraction of it.
    """
    import torch

    import plexus.operators                                          # noqa: F401
    import plexus.operators.mpm_warp                                 # noqa: F401
    try:
        import plexus.operators.mpm_loop                             # noqa: F401
    except ImportError:
        pass
    from plexus.engine import run
    from plexus.models.registry import get_operator
    from plexus.schema import load

    torch.cuda.set_device(device)
    sim = load(_spec_path(name))
    sim.n_frames = warmup + frames
    for blk in sim.schedule:
        if isinstance(blk, dict) and "steps" in blk:
            blk["capture"] = False

    ev = {}                       # op label -> list of (start, end)
    live = {"on": False}
    seen = set()

    def wrap(cls):
        if cls in seen:
            return
        seen.add(cls)
        inner = cls.forward
        label = cls.__name__

        def timed(self, H, mask=None):
            if not live["on"]:
                return inner(self, H, mask)
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            out = inner(self, H, mask)
            e.record()
            ev.setdefault(label, []).append((s, e))
            return out
        cls.forward = timed

    for op in sim.operators:
        wrap(get_operator(op.op, implementation=op.impl))

    def hook(H, tick):
        if tick == warmup:
            live["on"] = True

    t0 = time.perf_counter()
    run(sim, out_path=None, device=device, progress=False, on_frame=hook)
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0

    blk = [b for b in sim.schedule if isinstance(b, dict) and "steps" in b][0]
    n_sub = max(1, round(sim.dt / float(blk["substep_dt"])))
    tot = {k: sum(s.elapsed_time(e) for s, e in v) for k, v in ev.items()}
    calls = {k: len(v) for k, v in ev.items()}
    grand = sum(tot.values())
    print(f"\n  {name}  eager, {frames} frames after {warmup} warmup, "
          f"{n_sub} substeps/frame\n")
    print(f"  {'operator':<26}{'calls':>8}{'ms total':>11}{'ms/call':>10}"
          f"{'ms/frame':>10}{'share':>8}")
    print("  " + "-" * 73)
    for k in sorted(tot, key=lambda k: -tot[k]):
        print(f"  {k:<26}{calls[k]:>8}{tot[k]:>11.1f}{tot[k] / calls[k]:>10.3f}"
              f"{tot[k] / frames:>10.2f}{100 * tot[k] / grand:>7.1f}%")
    print("  " + "-" * 73)
    print(f"  {'sum of operators':<26}{'':>8}{grand:>11.1f}{'':>10}{grand / frames:>10.2f}"
          f"{100.0:>7.1f}%")
    print(f"\n  GPU time inside operators {grand / frames:.1f} ms/frame; "
          f"wall {1e3 * wall / (warmup + frames):.1f} ms/frame including warmup and startup.")
    print("  Shares are EAGER shares -- a captured graph is one launch with nothing to "
          "attribute.\n")


def table():
    if not os.path.isdir(ROWS):
        raise SystemExit(f"no rows yet in {ROWS}")
    rows = [json.load(open(os.path.join(ROWS, f)))
            for f in sorted(os.listdir(ROWS)) if f.endswith(".json")]
    if not rows:
        raise SystemExit(f"no rows yet in {ROWS}")
    seen = [s for s in SCENES if any(r.get("scene") == s for r in rows)]
    seen += sorted({r.get("scene", "") for r in rows} - set(seen) - {""})
    for scene in seen:
        srows = [r for r in rows if r.get("scene") == scene]
        r0 = srows[0]
        print(f"\n  {r0['base']}   {r0['particles']:,} particles   "
              f"{r0['substeps_per_frame']} substeps/frame   "
              f"{r0['frames_timed']} timed frames after {r0['warmup']} warmup\n")
        for gpu in sorted({r["gpu"] for r in srows}):
            mine = [r for r in srows if r["gpu"] == gpu]
            # BASELINE = the slowest row on this card, so "x" reads as speedup and needs no key.
            base = max(r["ms_per_frame"] for r in mine)
            print(f"  {gpu}")
            print(f"  {'variant':<16}{'ms/frame':>10}{'ms/substep':>12}{'ns/p-substep':>14}"
                  f"{'peak GB':>9}{'capture':>9}{'x':>7}")
            print("  " + "-" * 77)
            for r in sorted(mine, key=lambda r: r["ms_per_frame"]):
                print(f"  {r.get('variant', r['spec']):<16}"
                      f"{r['ms_per_frame']:>10.1f}{r['ms_per_substep']:>12.2f}"
                      f"{r['ns_per_particle_substep']:>14.2f}{r['peak_mem_GB']:>9.2f}"
                      f"{('YES' if r['captured'] else 'no'):>9}"
                      f"{base / r['ms_per_frame']:>6.2f}x")
            print()


def latex(out=None):
    """Emit `paper/mpm_perf_tables.tex` from the measured rows, for `paper/mpm_warp.tex`.

    GENERATED, NOT TYPED, for the same reason `tools/mpm_warp_note.py` generates its tables: a
    number retyped into a paper is a number that stops tracking the code the moment either moves.
    Two macros, because they answer two different questions:

        \\tblImplSweep   which implementation, at a fixed scene    (8 rows x 2 scenes x 2 cards)
        \\tblScaling     how far one GPU goes                      (200 M / 800 M / 1 B)
    """
    out = out or os.path.join(ROOT, "paper", "mpm_perf_tables.tex")
    rows = [json.load(open(os.path.join(ROWS, f)))
            for f in sorted(os.listdir(ROWS)) if f.endswith(".json")]
    esc = lambda s: str(s).replace("_", r"\_")
    L = ["% GENERATED by tools/mpm_perf.py --latex -- do not edit"]

    L.append(r"\newcommand{\tblImplSweep}{%")
    L.append(r"\begin{tabular}{@{}llrrrrc@{}}")
    L.append(r"\toprule")
    L.append(r"scene & implementation & \multicolumn{1}{c}{ms/frame} & "
             r"\multicolumn{1}{c}{ms/substep} & \multicolumn{1}{c}{ns/p-substep} & "
             r"\multicolumn{1}{c}{peak GiB} & capture \\")
    for scene in ("wf", "w5m"):
        srows = [r for r in rows if r.get("scene") == scene]
        if not srows:
            continue
        for gpu in sorted({r["gpu"] for r in srows}):
            mine = sorted([r for r in srows if r["gpu"] == gpu],
                          key=lambda r: r["ms_per_frame"])
            L.append(r"\midrule")
            L.append(r"\multicolumn{7}{@{}l}{\emph{%s, %s particles, %d substeps/frame --- %s}} \\"
                     % (esc(mine[0]["base"]), f"{mine[0]['particles']:,}",
                        mine[0]["substeps_per_frame"], esc(gpu)))
            for r in mine:
                L.append(r"& \code{%s} & %.1f & %.2f & %.2f & %.2f & %s \\" % (
                    esc(r["variant"]), r["ms_per_frame"], r["ms_per_substep"],
                    r["ns_per_particle_substep"], r["peak_mem_GB"],
                    "yes" if r["captured"] else "--"))
    L += [r"\bottomrule", r"\end{tabular}}", ""]

    big = [r for r in rows if r["spec"].startswith("si_bench_")]
    L.append(r"\newcommand{\tblScaling}{%")
    L.append(r"\begin{tabular}{@{}lrrrrrr@{}}")
    L.append(r"\toprule")
    L.append(r"GPU & \multicolumn{1}{c}{particles} & \multicolumn{1}{c}{$n_{\mathrm{grid}}$} & "
             r"\multicolumn{1}{c}{substeps} & \multicolumn{1}{c}{ms/frame} & "
             r"\multicolumn{1}{c}{ns/p-substep} & \multicolumn{1}{c}{peak GiB} \\")
    L.append(r"\midrule")
    for r in sorted(big, key=lambda r: r["particles"]):
        # `n_grid` is not in the row -- it is a property of the spec, read back here so the table
        # states the discretisation each row was measured at rather than implying one grid for all.
        try:
            ng = list(yaml.safe_load(open(_spec_path(r["spec"])))["fields"].values())[0]["n_grid"]
        except Exception:
            ng = "--"
        L.append(r"%s & %s & %s & %d & %.0f & %.2f & %.1f \\" % (
            esc(r["gpu"].replace("NVIDIA ", "")), f"{r['particles'] / 1e6:.0f}\\,M",
            ng, r["substeps_per_frame"], r["ms_per_frame"],
            r["ns_per_particle_substep"], r["peak_mem_GB"]))
    L += [r"\bottomrule", r"\end{tabular}}", ""]
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {out}  ({len(rows)} rows, {len(big)} scaling rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="?", const="all", default=None,
                    help=f"write the sweep specs for a scene ({'/'.join(SCENES)}/all)")
    ap.add_argument("--run", default=None, help="spec name to time, e.g. si_perf_wf_warp_cap")
    ap.add_argument("--table", action="store_true", help="collate every row written so far")
    ap.add_argument("--latex", action="store_true",
                    help="write paper/mpm_perf_tables.tex from the rows")
    ap.add_argument("--profile", default=None,
                    help="spec name to break down per operator (eager, capture forced off)")
    ap.add_argument("--profile-frames", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--timed", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    if a.sweep:
        for s in (SCENES if a.sweep == "all" else [a.sweep]):
            write_sweep(s)
    if a.run:
        # FRAME COUNTS FOLLOW THE SCENE unless overridden: 200 timed frames of the 5M spec is 57
        # minutes of torch, and a benchmark nobody waits for is a benchmark nobody runs.
        sc = next((v for v in SCENES.values() if a.run.startswith(v[1])), None)
        warm = a.warmup if a.warmup is not None else (sc[3] if sc else WARMUP)
        timed = a.timed if a.timed is not None else ((sc[2] - sc[3]) if sc else N_FRAMES - WARMUP)
        run_one(a.run, a.device, timed, warm, a.tag)
    if a.profile:
        profile(a.profile, a.device, a.profile_frames, a.warmup or WARMUP)
    if a.table:
        table()
    if a.latex:
        latex()
    if not (a.sweep or a.run or a.table or a.profile or a.latex):
        ap.print_help()


if __name__ == "__main__":
    main()

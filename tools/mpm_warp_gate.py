#!/usr/bin/env python
"""GATE: run a selection of `config/material` specs under `default` and under `warp`, and decide
whether `warp` could become what a spec gets when it does not ask.

WHAT A GATE HAS TO ANSWER, and it is two questions not one:

  1. WHERE WARP APPLIES, does it track `default`? It cannot be byte-identical (the 27-tap sums are
     reassociated and the scatter's atomics commit in hardware order), so the test is a tolerance
     one -- and the tolerance has to be chosen for the right quantity.
  2. WHERE WARP DOES NOT APPLY, what happens? `mpm_scatter[warp]` and `mpm_gather[warp]` are 3D and
     CUDA only. `general.dim` defaults to 2 (schema.py:121), so most of `config/material` is 2D.
     A gate that only ran the 3D specs would answer question 1 and silently report a pass on a
     library three quarters of which cannot run at all.

POINTWISE DIVERGENCE IS THE WRONG PASS CRITERION ON ITS OWN. These are turbulent flows: two runs
differing in the last bit of one particle's position separate exponentially, so `max|dpos|` after a
few hundred substeps measures the Lyapunov exponent, not the implementation. What must agree is the
AGGREGATES -- total mass, centre of mass, kinetic energy, bounding box -- because those are set by
the conservation laws both implementations claim to obey, and a real bug (a dropped stencil tap, a
sign error, a missed boundary condition) moves them immediately. So both are reported and the
verdict keys on the aggregates, with the pointwise column shown for what it is.

    python tools/mpm_warp_gate.py --frames 25 --device cuda:1 --json /tmp/gate.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG = os.path.join(ROOT, "config", "material")

# THE SELECTION, written down rather than globbed: a glob changes under the gate as specs are added
# and the table stops being comparable between runs.
#
# ALL TWENTY 3D SPECS ARE HERE, including the ones where `default` runs out of memory. Those are not
# omitted, because "default cannot run this at all" is a RESULT and dropping the row would turn the
# most important finding into an absence. The runner records the failure and moves on.
SPECS_3D = [
    "material_3d_ball_drop", "material_3d_balls_bouncy", "material_3d_cube_drop",
    "material_3d_multimaterial", "material_3d_obstacle_pillars", "material_3d_obstacle_slab",
    "material_3d_obstacle_sphere", "material_3d_snow_block", "material_3d_water_drop",
    "material_3d_water_lots", "material_3d_water_lots_x4", "material_3d_water_lots_x10",
    "material_3d_water_bench", "material_3d_water_bench_500k", "material_3d_water_bench_1m",
    "material_3d_water_bench_5m", "material_3d_water_bench_10m", "material_3d_water_bench_20m",
    "material_3d_water_bench_100m", "material_3d_water_bench_200m",
]
SPECS_2D = [
    "material_dam_break", "material_dam_viscous", "material_slosh", "material_funnel",
    "material_hydrostatic", "material_crown_splash", "material_snow_pile", "material_snow_funnel",
    "material_bowl_1", "material_steps_1", "material_vessel_1", "material_zigzag",
    "material_coalesce", "material_two_drops_st", "material_active_swirl",
]


def _final(H):
    """The state both implementations must agree on, plus the aggregates the verdict keys on."""
    import torch
    out = {}
    for name, lvl in H.levels.items():
        if not hasattr(lvl, "F"):
            continue
        pos = lvl.get("pos").detach()
        vel = lvl.get("vel").detach()
        m = getattr(lvl, "mass", None)
        m = m.detach() if m is not None else torch.ones(pos.shape[0], device=pos.device)
        out[name] = {
            "pos": pos.clone(), "vel": vel.clone(),
            "F": lvl.F.detach().clone(), "C": lvl.C.detach().clone(),
            # AGGREGATES. Set by conservation, not by summation order.
            "mass": float(m.sum()),
            "com": (pos * m[:, None]).sum(0).div(m.sum().clamp(min=1e-12)).tolist(),
            "ke": float(0.5 * (m[:, None] * vel * vel).sum()),
            "bbox_lo": pos.min(0).values.tolist(), "bbox_hi": pos.max(0).values.tolist(),
        }
    return out


def run(spec_name, impl, frames, warmup, device, seed=0):
    import torch
    import yaml

    import plexus.operators  # noqa: F401
    from plexus import engine as E
    from plexus.schema import load
    if impl == "warp":
        import plexus.operators.mpm_warp  # noqa: F401

    spec = yaml.safe_load(open(os.path.join(CFG, spec_name + ".yaml")))
    for o in spec.get("operators", []):
        if isinstance(o, dict) and o.get("op") in ("mpm_scatter", "mpm_gather"):
            o.pop("implementation", None)
            if impl != "default":
                o["implementation"] = impl
            if o["op"] == "mpm_scatter":
                # SAME POLAR ON BOTH SIDES. The warp kernel implements Higham's iteration only, so
                # leaving default on SVD would compare two different algorithms and blame the
                # difference on the port.
                o["polar"] = "higham"
    for st in spec.get("schedule", []):
        if isinstance(st, dict) and "substep_dt" in st:
            st["capture"] = False                     # capture is orthogonal; keep the gate on the ops
            st.pop("compile", None)
    spec["general"]["n_frames"] = frames + warmup
    spec["general"]["record_cap"] = 2
    spec["general"]["seed"] = seed

    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(spec, f); f.close()
    sim = load(f.name); os.unlink(f.name)

    marks = {}

    def on_frame(H, tick):
        if tick == warmup:
            torch.cuda.synchronize(device); marks["a"] = time.perf_counter()
        elif tick == warmup + frames:
            torch.cuda.synchronize(device); marks["b"] = time.perf_counter()

    torch.cuda.reset_peak_memory_stats(device)
    H, _ = E.run(sim, out_path=None, device=device, on_frame=on_frame, progress=False)
    ms = (marks["b"] - marks["a"]) / frames * 1000 if "b" in marks else float("nan")
    n = sum(int(l.n) for l in H.levels.values() if hasattr(l, "F"))
    return {"ms": ms, "n": n, "final": _final(H),
            "gib": torch.cuda.max_memory_allocated(device) / 2 ** 30}


def compare(a, b):
    """Pointwise divergence AND aggregate agreement, per particle set, reduced to the worst."""
    import torch
    worst = {"dpos": 0.0, "rpos": 0.0, "dF": 0.0, "d_mass": 0.0, "d_com": 0.0, "d_ke": 0.0,
             "d_bbox": 0.0}
    for name, A in a.items():
        B = b.get(name)
        if B is None:
            continue
        for k, key in (("pos", "dpos"), ("F", "dF")):
            d = (A[k] - B[k]).abs().max().item()
            worst[key] = max(worst[key], d)
        scale = A["pos"].abs().max().clamp(min=1e-9)
        worst["rpos"] = max(worst["rpos"], ((A["pos"] - B["pos"]).abs().max() / scale).item())
        # RELATIVE, because "the centre of mass moved by 1e-5" means nothing without the box size.
        box = max(1e-9, max(hi - lo for hi, lo in zip(A["bbox_hi"], A["bbox_lo"])))
        worst["d_mass"] = max(worst["d_mass"],
                              abs(A["mass"] - B["mass"]) / max(abs(A["mass"]), 1e-12))
        worst["d_com"] = max(worst["d_com"],
                             max(abs(x - y) for x, y in zip(A["com"], B["com"])) / box)
        worst["d_ke"] = max(worst["d_ke"], abs(A["ke"] - B["ke"]) / max(abs(A["ke"]), 1e-12))
        worst["d_bbox"] = max(worst["d_bbox"],
                              max(max(abs(x - y) for x, y in zip(A[s], B[s]))
                                  for s in ("bbox_lo", "bbox_hi")) / box)
    return worst


# THRESHOLDS DECLARED BEFORE THE RUN, per the paper's rule that a threshold chosen after seeing the
# number is not a threshold. All are RELATIVE and all are on aggregates, for the reason in the
# module docstring. 1e-4 is ~1000x float32 epsilon: loose enough that reassociation over a few
# hundred substeps cannot trip it, tight enough that a dropped stencil tap (which would move mass by
# ~1/27 = 4%) or a missed boundary condition cannot hide under it.
TOL = {"d_mass": 1e-6, "d_com": 1e-4, "d_ke": 1e-2, "d_bbox": 1e-3}


def verdict(w):
    bad = [k for k, t in TOL.items() if w[k] > t]
    return ("PASS" if not bad else "FAIL:" + ",".join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--json", default="/tmp/mpm_warp_gate.json")
    ap.add_argument("--only", default=None, help="comma-separated spec names, for a re-run")
    a = ap.parse_args()

    import torch
    torch.cuda.init(); torch.zeros(1, device=a.device)
    names = a.only.split(",") if a.only else (SPECS_3D + SPECS_2D)

    print(f"\n  {torch.cuda.get_device_properties(a.device).name}   "
          f"{a.frames} timed frames after {a.warmup} warm-up, capture off, polar=higham both sides")
    print(f"\n  {'spec':<32}{'dim':>4}{'particles':>10}{'default':>10}{'warp':>10}{'x':>6}"
          f"{'max|dpos|':>11}{'d_mass':>9}{'d_com':>9}{'d_KE':>9}  verdict")
    print("  " + "-" * 128)
    rows = []
    for nm in names:
        row = {"spec": nm}
        try:
            import yaml
            row["dim"] = int((yaml.safe_load(open(os.path.join(CFG, nm + ".yaml")))
                              .get("general") or {}).get("dim", 2))
        except Exception:
            row["dim"] = 0

        # WARP FIRST, DEFAULT SECOND, and the order is load-bearing. `default` is the memory hog --
        # at 100 M it asks for 60 GiB in ONE allocation -- and a run that dies part way through
        # leaves the caching allocator holding what it had already taken. Running it first made the
        # gate report "BOTH FAIL" at 100 M on a card where warp alone runs that spec in 30.89 GiB:
        # the second run was OOMing on the first run's corpse, not on its own footprint. Clearing
        # the exception's traceback frames (which own the tensors through the frame locals) helps
        # and is still done, but it is not sufficient. Running the cheap side first is.
        got = {}
        for impl in ("warp", "default"):
            try:
                got[impl] = run(nm, impl, a.frames, a.warmup, a.device)
                row[f"ms_{impl}"] = got[impl]["ms"]
                row[f"gib_{impl}"] = got[impl]["gib"]
                row["n"] = got[impl]["n"]
            except Exception as e:
                row[f"error_{impl}"] = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
                traceback.clear_frames(e.__traceback__)
                e = None
            torch.cuda.empty_cache()

        if "warp" in got and "default" in got:
            row.update(compare(got["default"]["final"], got["warp"]["final"]))
            row["verdict"] = verdict(row)
            print(f"  {nm:<32}{row['dim']:>4}{row['n']:>10,}{row['ms_default']:>10.1f}"
                  f"{row['ms_warp']:>10.1f}{row['ms_default']/row['ms_warp']:>6.1f}"
                  f"{row['dpos']:>11.2e}{row['d_mass']:>9.1e}{row['d_com']:>9.1e}"
                  f"{row['d_ke']:>9.1e}  {row['verdict']}", flush=True)
        elif "warp" in got:
            row["verdict"] = "WARP ONLY"
            print(f"  {nm:<32}{row['dim']:>4}{row['n']:>10,}{'OOM':>10}"
                  f"{row['ms_warp']:>10.1f}{'--':>6}{'':>11}{'':>9}{'':>9}{'':>9}  "
                  f"WARP ONLY  ({row['error_default'][:36]})", flush=True)
        elif "default" in got:
            row["verdict"] = "N/A"
            print(f"  {nm:<32}{row['dim']:>4}{row['n']:>10,}{row['ms_default']:>10.1f}"
                  f"{'--':>10}{'--':>6}{'':>11}{'':>9}{'':>9}{'':>9}  "
                  f"N/A  {row['error_warp'][:48]}", flush=True)
        else:
            row["verdict"] = "BOTH FAIL"
            print(f"  {nm:<32}{row['dim']:>4}{'':>10}  BOTH FAIL  "
                  f"warp={row.get('error_warp','?')[:28]}  "
                  f"default={row.get('error_default','?')[:28]}", flush=True)
        got.clear()
        torch.cuda.empty_cache()
        rows.append(row)

    ok = [r for r in rows if r.get("verdict") == "PASS"]
    fail = [r for r in rows if str(r.get("verdict", "")).startswith("FAIL")]
    na = [r for r in rows if r.get("verdict") == "N/A"]
    wonly = [r for r in rows if r.get("verdict") == "WARP ONLY"]
    print("\n  " + "-" * 128)
    print(f"  {len(ok)} PASS   {len(fail)} FAIL   {len(na)} N/A (warp cannot run: 2D/CPU)   "
          f"{len(wonly)} WARP ONLY (default OOM)   of {len(rows)} specs")
    if ok:
        sp = [r["ms_default"] / r["ms_warp"] for r in ok]
        print(f"  speedup where warp runs: min {min(sp):.1f}x  median "
              f"{sorted(sp)[len(sp)//2]:.1f}x  max {max(sp):.1f}x")
    if na:
        print(f"  N/A reasons: " + "; ".join(sorted({r.get("error_warp", "?")[:60] for r in na})))
    json.dump(rows, open(a.json, "w"), indent=1)
    print(f"\n  rows -> {a.json}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""phase1 -- the preliminary battery, on instruments that have been checked.

NOT a re-run of the overnight study. That is quarantined, not inherited: every one of those runs
was measured with at least one instrument since proved wrong, and re-running them "to see if the
conclusion survives" would import the old design along with the old data.

WHAT THIS ASKS
------------------------------------------------------------------------------------------------
Two blocks, both cheap, run locally across the two GPUs.

  A  SOLO KNOCKOUTS.  Remove one operator at a time from a physiological base and measure what
     changes. The map currently characterises the solo effect of 0 of 8 operators, so this is the
     emptiest and cheapest cell to fill -- and every later ablation is read against it.

  B  PHENOMENON PROBES.  One spec each for the behaviours we care about: does a pattern form at
     all, does the tissue grow, does it divide, does it undulate, does it tube. These are not
     searching for anything; they establish what the substrate does before anyone searches it.

WHAT CHANGED FROM THE OLD RECIPE, and why
------------------------------------------------------------------------------------------------
  * BASELINE GROWTH IS ON (rho > 0). The old recipe ran rho = 0, so growth was purely
    activator-gated and the body could not grow at all -- a protrusion could only be drawn from
    existing material. That is not a setting, it is unphysiological: cells grow by taking up
    nutrients. The floor is now above zero and the activator adds growth on top.
  * SEVERAL SEED SPOTS. One protrusion per run is a single sample; you cannot separate "this
    mechanism makes tubes" from "this run made a tube". Several spots give within-run replication
    for free, and it is what Okuda shows.
  * AN RD INITIAL CONDITION. `okuda_route` carries cell_react and cell_diffuse but no
    cell_rd_seed. The type checker passes it -- cell_react's `adjacency` precondition is met -- but
    Gray-Scott is autocatalytic, so from a uniform zero it STAYS at zero and no pattern ever forms.
    A type system cannot see that; it is exactly the kind of thing a biological premise would
    catch. Probe B1 tests it directly rather than assuming either way.

Every spec is timed, and the times are the point: nobody has ever measured what a spec costs.

    python phase1.py --frames 300            # the whole battery
    python phase1.py --only p1_ph_undulation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

OUT = os.path.join(HERE, "_phase1")
# knockouts that cannot be expressed: filled by specs(), and reported as necessity results
REQUIRED: list = []
GPUS = ["cuda:0", "cuda:1"]
PER_GPU = 3                    # concurrent runs per device; the meshes are small


# ---------------------------------------------------------------- the physiological base
def base_graph():
    """okuda_route + an RD seed, with a baseline growth floor. No forcing operator."""
    from composition_space import reference_recipes
    g = reference_recipes()["okuda_route"]
    g, _ = g.apply(("add_op", "cell_rd_seed", "scatter"))
    # GRAY-SCOTT, not Gierer-Meinhardt. okuda_route ships with GM, and the emitter then writes GM
    # parameters -- so the F/kk below would have been SILENTLY DROPPED and the run would have used
    # defaults while the spec appeared to set them. Caught by printing the emitted spec instead of
    # trusting the graph. The minisite validated Gray-Scott; use the kinetics the numbers belong to.
    _rid = next(o["id"] for o in g.ops if o["op"] == "cell_react")
    g, _ = g.apply(("set_impl", _rid, "gray_scott"))
    nid = {o["op"]: o["id"] for o in g.ops}
    p = dict(g.params)
    p[f'{nid["morphogen_growth_3d"]}.rho'] = 0.30     # BASELINE GROWTH ON (was 0.0 = no body growth)
    p[f'{nid["morphogen_growth_3d"]}.a_sw'] = 0.30    # switch inside the activator's actual range
    p[f'{nid["morphogen_growth_3d"]}.rate'] = 0.010
    # THE VALIDATED GRAY-SCOTT, taken from the minisite coral spec rather than invented. Those
    # are the only settings measured to give a live pattern on this substrate (act_max 0.43);
    # the search-space defaults sit 35x off the validated diffusion ratio (0.7/0.02 against
    # 0.16/0.08) with chi 4.0 against 1.3. Scattered seeding also gives many spots at once, which
    # is the within-run replication we want.
    p[f'{nid["cell_rd_seed"]}.seed_frac'] = 0.06
    p[f'{nid["cell_diffuse"]}.d_a'] = 0.08
    p[f'{nid["cell_diffuse"]}.d_h'] = 0.16
    p[f'{nid["cell_diffuse"]}.chi'] = 1.3
    p[f'{nid["cell_react"]}.F'] = 0.055
    p[f'{nid["cell_react"]}.kk'] = 0.062
    p[f'{nid["cell_react"]}.rd_rate'] = 1.0
    return g.with_params(p), nid


def specs():
    """[(name, graph, question)] -- what each spec is for, in one line."""
    from composition_space import reference_recipes
    g0, nid = base_graph()
    REQUIRED.clear()
    out = [("p1_control", g0, "the base: RD + baseline growth + division + mechanics, no forcing")]

    # ---- A. solo knockouts: one operator removed, everything else identical
    for op, why in (("cell_rd_seed", "no initial pattern -- does RD start at all?"),
                    ("cell_react", "no reaction -- diffusion alone"),
                    ("cell_diffuse", "no spreading -- reaction alone"),
                    ("cell_adjacency", "no neighbour graph -- RD has nothing to run on"),
                    ("morphogen_growth_3d", "no growth -- mechanics and division only"),
                    ("divide_3d", "no division -- growth without proliferation"),
                    ("reconnect_t1_3d", "no neighbour exchange -- can the tissue still flow?"),
                    ("cell_geometry_3d", "no per-cell geometry readout")):
        if op not in nid:
            continue
        try:
            gk, _ = g0.apply(("remove_op", nid[op]))
        except Exception as e:
            REQUIRED.append((op, f"the edit itself is illegal: {e}"))
            continue
        ok, reason = gk.is_runnable()
        if not ok:
            # A KNOCKOUT THAT CANNOT EVEN BE EXPRESSED IS A RESULT, not an error. If removing an
            # operator leaves another one's input fed by nothing, that operator is STRUCTURALLY
            # REQUIRED by this composition -- a necessity claim obtained for free, without
            # spending a GPU-second. Recording it is the point; crashing on it throws it away.
            REQUIRED.append((op, reason))
            continue
        # NAMED "without", not "ko". `p1_ko_divide_3d` reads as "divide" at a glance -- Cedric
        # browsed the folder and asked why the division run showed no green cells, in the one run
        # where division is REMOVED. A directory name is an instrument too.
        out.append((f"p1_without_{op}", gk, f"WITHOUT {op}: {why}"))

    # ---- B. phenomenon probes
    p = dict(g0.params)

    q = dict(p); q[f'{nid["morphogen_growth_3d"]}.rate'] = 0.0
    out.append(("p1_ph_rd_only", g0.with_params(q),
                "PATTERN: growth off. Does a Turing pattern form on a static shell?"))

    r = dict(p); r[f'{nid["morphogen_growth_3d"]}.rho'] = 1.0
    r[f'{nid["morphogen_growth_3d"]}.a_sw'] = 50.0
    out.append(("p1_ph_growth_only", g0.with_params(r),
                "GROWTH: uniform, activator switch off. Does the ball actually get bigger?"))

    u = dict(p)
    u[f'{nid["cell_diffuse"]}.chi'] = 8.0             # long-range inhibition -> few, broad domains
    u[f'{nid["cell_react"]}.rd_rate'] = 3.0
    u[f'{nid["cell_rd_seed"]}.seed_frac'] = 0.12
    out.append(("p1_ph_undulation", g0.with_params(u),
                "UNDULATION (Okuda Fig 7): many shallow bumps, not one deep protrusion"))

    t = dict(p)
    t[f'{nid["cell_rd_seed"]}.seed_frac'] = 0.02
    t[f'{nid["morphogen_growth_3d"]}.rate'] = 0.02
    t[f'{nid["cell_diffuse"]}.chi'] = 1.0
    out.append(("p1_ph_tube", g0.with_params(t),
                "TUBE: one spot, stronger activator-driven growth"))

    gd = reference_recipes()["uniform_inflation"]
    out.append(("p1_ph_divide_only", gd,
                "DIVISION: uniform inflation + division, no chemistry at all"))

    # ---- C. the CORAL reference: reaction-diffusion on a FIXED ball.
    # No growth, no division, nothing moving -- just the chemistry, on a shell held still. This is
    # the minisite's front-page movie and it is the cleanest possible read on whether our Turing
    # pattern is alive: any structure you see is the chemistry, because nothing else is running.
    # It is also the control the whole battery was missing. Every other spec confounds pattern
    # formation with mechanics; this one cannot. 2000 cells rather than 500 because the Turing
    # wavelength is set in CELL widths -- four times the cells on the same sphere means four times
    # as many pattern periods across it, which is what makes the coral legible instead of a
    # handful of blobs.
    gc = g0
    for op in ("morphogen_growth_3d", "divide_3d"):
        if op in nid:
            gc, _ = gc.apply(("remove_op", nid[op]))
    c = dict(gc.params)
    c[f'{nid["seed_mesh_3d"]}.n_cells'] = 2000
    out.append(("p1_ph_coral_fixed_ball", gc.with_params(c),
                "CORAL: Turing chemistry alone on a rigid 2000-cell ball. Is the pattern alive?"))
    return out


# ---------------------------------------------------------------- run
def write_configs(sp, frames):
    import translate as T
    for name, g, _ in sp:
        T.write_config(g, name, frames=frames)
    return [name for name, _, _ in sp]


def run_all(sp, frames, jobs_per_gpu=PER_GPU):
    os.makedirs(OUT, exist_ok=True)
    queue = list(sp)
    running, done = {}, {}
    slots = [(g, i) for g in GPUS for i in range(jobs_per_gpu)]
    free = list(slots)
    t_batch = time.time()
    while queue or running:
        while queue and free:
            name, _, why = queue.pop(0)
            dev, _ = free.pop(0)
            log = open(os.path.join(OUT, f"{name}.log"), "w")
            cmd = [sys.executable, "-u", os.path.join(HERE, "run_one.py"), name,
                   "--frames", str(frames), "--device", dev, "--campaign", "phase1"]
            running[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                              cwd=HERE,
                                              env={**os.environ,
                                                   "PYTHONPATH": os.path.join(ROOT, "src")}),
                             log, time.time(), dev, why)
            print(f"  [start {dev}] {name}", flush=True)
        time.sleep(3)
        for name, (proc, log, t0, dev, why) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            free.append((dev, 0))
            secs = time.time() - t0
            d = os.path.join(ROOT, "log", "okuda", name, "diag.json")
            summ = json.load(open(d)).get("summary", {}) if os.path.exists(d) else {}
            done[name] = {"secs": round(secs, 1), "device": dev, "question": why,
                          "exit": proc.returncode, **{k: summ.get(k) for k in
                          ("protr_peak", "protr_final", "Q_protr_after_relax", "n_cells_final",
                           "act_max_final", "ta_n_tubes_final", "shape_idx_mean", "broken_n",
                           "horizon_frame", "valid_frac", "inert_operators", "saturated")}}
            print(f"  [done  {dev}] {name:26} {secs:6.1f}s  exit {proc.returncode}", flush=True)
    return done, time.time() - t_batch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--only", default=None)
    ap.add_argument("--per-gpu", type=int, default=PER_GPU)
    a = ap.parse_args()

    sp = specs()
    if a.only:
        sp = [s for s in sp if s[0] == a.only]
    print(f"[phase1] {len(sp)} specs, {a.frames} frames, {len(GPUS)} GPUs x {a.per_gpu}\n")
    for n, _, why in sp:
        print(f"    {n:26} {why}")
    print()
    if REQUIRED:
        print("\nSTRUCTURALLY REQUIRED -- these cannot be removed at all, which is a result:")
        for op, why in REQUIRED:
            print(f"    {op:26} {why}")
        print()
    write_configs(sp, a.frames)
    done, wall = run_all(sp, a.frames, a.per_gpu)

    os.makedirs(OUT, exist_ok=True)
    json.dump({"frames": a.frames, "wall_s": round(wall, 1), "runs": done,
               "structurally_required": [{"operator": o, "why": w} for o, w in REQUIRED]},
              open(os.path.join(OUT, "battery.json"), "w"), indent=1)
    print(f"\n[phase1] {len(done)} specs in {wall:.0f}s wall "
          f"(sum of run times {sum(d['secs'] for d in done.values()):.0f}s)")
    print(f"[phase1] -> {OUT}/battery.json")

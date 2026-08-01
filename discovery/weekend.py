#!/usr/bin/env python
"""weekend -- the first experiment this campaign has run on instruments that were checked first.

WHAT IT ASKS
================================================================================================
Does the tissue's SHAPE feeding back into its CHEMISTRY change the morphology, and does it matter
WHICH shape feature the chemistry listens to?

That is a mechanism question, not a parameter question, which is why the four features are
IMPLEMENTATIONS of one contract and not four values of a knob (see tyssue_shape_to_chem). Swapping
curvature for tension is scored as a new mechanism; changing beta is not.

THE DESIGN, and why it is shaped this way
------------------------------------------------------------------------------------------------
  THE NULL IS MANDATORY.  beta = 0 is the identical composition with the feedback silent. Without
  it "shape feeds back" is asserted, not tested -- and it is the same operator, same schedule,
  same seed, so nothing but the feedback differs.

  BOTH SIGNS.  Whether deformed cells signal MORE or LESS is a real hypothesis. A one-sided sweep
  would answer it silently.

  THREE SEEDS EVERYWHERE.  One protrusion is one sample. The campaign has twice drawn a conclusion
  from a single run and twice retracted it (F001, F004). Three seeds is the minimum that can
  distinguish "this mechanism does X" from "this run did X".

  ABLATION IS PRESENT BY CONSTRUCTION.  The null is the subtractive direction, so a causal claim
  about shape_to_chem has both directions available and passes critic.check_batch. That is not a
  coincidence -- the batch was built to satisfy the rule Phase 3 added.

WHAT IS DIFFERENT FROM EVERY PREVIOUS BATCH
------------------------------------------------------------------------------------------------
Every instrument it will be read with exists and has been certified against a case whose answer
was already known:

    biologist      12 premises, 15/15 certified, gating every run
    morphology         sphere/undulation/tube/branched/invalid, certified on built shapes
    pattern_scale      n_spots exact at 3/5/12, spacing within 13% of R sqrt(4pi/k)
    reduced_volume     0.996 on a sphere, 0.55 on a crumple
    ray_single_frac    1.00 on a clean shell, 0.00 on one folded through itself
    chemistry          calibrated to three stable spots, reproducible across seeds (F012)

The three batteries launched earlier today were each discarded because a ruler turned out to be
broken. This one is the first that will not be.

    python weekend.py            # 27 runs, ~6 concurrent across two GPUs
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CFG = os.path.join(ROOT, "config", "okuda")
OUT = os.path.join(HERE, "_weekend")
GPUS = ["cuda:0", "cuda:1"]
PER_GPU = 3

FEATURES = ["curvature", "tension", "apical_area", "pressure"]
BETAS = [1.5, -1.5]
SEEDS = [0, 1, 2]


def base_config():
    """A growing, dividing, patterned vesicle -- the shape must be able to CHANGE, or a feedback
    from shape to chemistry has nothing to feed back."""
    c = yaml.safe_load(open(os.path.join(CFG, "mini_grow_divide_bigger.yaml")))
    c["general"]["n_frames"] = 400
    c["general"]["record_cap"] = 402
    ops = {o["op"]: o for o in c["operators"]}
    # chemistry, at the calibrated values (F012)
    rd = [{"op": "cell_adjacency", "at": "cell"},
          {"op": "cell_rd_seed", "at": "cell", "seed": 0, "before_frame": 3,
           "mode": "scatter", "seed_frac": 0.06},
          {"op": "cell_diffuse", "at": "cell", "d_a": 0.08, "d_h": 0.16, "chi": 1.3},
          {"op": "cell_react", "at": "cell", "implementation": "gray_scott",
           "F": 0.046, "kk": 0.062, "rate": 1.0}]
    order = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "cell_rd_seed", "cell_diffuse",
             "cell_react", "shape_to_chem", "morphogen_growth_3d", "shape_energy_3d",
             "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    c["operators"] = [ops["seed_mesh_3d"], ops["cell_geometry_3d"]] + rd + \
                     [ops["morphogen_growth_3d"], ops["shape_energy_3d"],
                      ops["reconnect_t1_3d"], ops["divide_3d"], ops["topo_snapshot_3d"]]
    # the growth switch must be INSIDE the activator's range or the chemistry cannot shape anything
    for o in c["operators"]:
        if o["op"] == "morphogen_growth_3d":
            o["a_sw"] = 0.25; o["rho"] = 0.35; o["hill"] = 4.0; o["vth_frac"] = 2.5
            # DILUTION OFF, and it is a declared choice with a measured reason (finding F007).
            # Growth dilutes what is inside a cell -- correct physics -- but Gray-Scott's activator
            # is sustained by a QUADRATIC term, so any steady multiplicative loss beats it. The
            # calibrated pattern (F = 0.046) is closer to its existence boundary than the old one
            # and dies even faster: a smoke run with dilution on ended at act_max 0.0009, and the
            # P4 premise check caught it before a single result was read. Leaving it on would make
            # every run in this batch a measurement of a dead field.
            o["conserve_amount"] = False
    c["_order"] = order
    c.pop("_premises", None)          # P3 no longer needs waiving: vth_frac 2.5 > factor 2.0
    # relax_iters is a constant while the mesh grows, and P5b flagged the residual force climbing
    # on the smoke run. Raising it is cheap next to discovering afterwards that the mechanics was
    # lagging the forces for the whole batch.
    for o in c["operators"]:
        if o["op"] == "shape_energy_3d":
            o["relax_iters"] = 60
    return c


def specs():
    out = []
    b = base_config()
    for sd in SEEDS:
        # THE NULL. Same composition, feedback silent. This is also the subtractive direction that
        # makes a causal claim about shape_to_chem legal under critic.check_batch.
        c = copy.deepcopy(b)
        c["operators"].insert(6, {"op": "shape_to_chem", "at": "cell", "implementation": "curvature",
                                  "vertex_set": "vertex", "beta": 0.0, "F0": 0.046, "rate": 1.0})
        out.append((f"wk_null_s{sd}", c, sd, "NULL: shape feedback silent (beta = 0)"))
        for feat in FEATURES:
            for beta in BETAS:
                c = copy.deepcopy(b)
                c["operators"].insert(6, {"op": "shape_to_chem", "at": "cell",
                                          "implementation": feat, "vertex_set": "vertex",
                                          "beta": beta, "F0": 0.046, "rate": 1.0})
                sgn = "pos" if beta > 0 else "neg"
                out.append((f"wk_{feat}_{sgn}_s{sd}", c, sd,
                            f"{feat}, beta {beta:+.1f}: deformed cells feed "
                            f"{'faster' if beta > 0 else 'slower'}"))
    return out


def write(name, cfg, seed):
    c = copy.deepcopy(cfg)
    order = c.pop("_order")
    c["general"]["name"] = name
    c["general"]["seed"] = seed
    for o in c["operators"]:
        if "seed" in o:
            o["seed"] = seed
    c["schedule"] = [x for x in order if any(o["op"] == x for o in c["operators"])]
    c["_discovery"] = {"comp_hash": f"WK_{name}", "region": "shape->chemistry feedback"}
    yaml.safe_dump(c, open(os.path.join(CFG, f"{name}.yaml"), "w"), sort_keys=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--per-gpu", type=int, default=PER_GPU)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sp = specs()
    print(f"[weekend] {len(sp)} runs: {len(FEATURES)} features x {len(BETAS)} signs x "
          f"{len(SEEDS)} seeds, plus {len(SEEDS)} nulls\n")
    for n, c, sd, why in sp:
        write(n, c, sd)
    queue, running, done = list(sp), {}, {}
    free = [(g, i) for g in GPUS for i in range(a.per_gpu)]
    t0 = time.time()
    while queue or running:
        while queue and free:
            n, c, sd, why = queue.pop(0)
            dev, _ = free.pop(0)
            log = open(os.path.join(OUT, f"{n}.log"), "w")
            running[n] = (subprocess.Popen(
                [sys.executable, "-u", os.path.join(HERE, "run_one.py"), n,
                 "--frames", str(a.frames), "--device", dev, "--campaign", "weekend"],
                stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}),
                log, time.time(), dev, why)
            print(f"  [start {dev}] {n}", flush=True)
        time.sleep(5)
        for n, (p, log, ts, dev, why) in list(running.items()):
            if p.poll() is None:
                continue
            log.close(); del running[n]; free.append((dev, 0))
            d = os.path.join(ROOT, "log", "okuda", n, "diag.json")
            rec = {}
            if os.path.exists(d):
                j = json.load(open(d))
                rec = {"summary": j.get("summary", {}), "premises_broken": j.get("premises_broken", [])}
            done[n] = {"secs": round(time.time() - ts, 1), "exit": p.returncode,
                       "question": why, **rec}
            print(f"  [done  {dev}] {n:26} {time.time()-ts:6.0f}s  exit {p.returncode}", flush=True)
            json.dump(done, open(os.path.join(OUT, "weekend.json"), "w"), indent=1)
    print(f"\n[weekend] {len(done)} runs in {(time.time()-t0)/60:.0f} min -> {OUT}/weekend.json")

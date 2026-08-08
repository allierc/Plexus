#!/usr/bin/env python
"""Now that recording is ALIGNED (topo every=1, record every frame -> posf and hist same length), how
far does vesicle_divide stay clean? Run 500 and 1000 frames (big buffer), report hollow over time in
100-frame blocks -- does a REAL defect ever appear, or does it stay a smooth dividing sphere?"""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tyssue_diag import hollow_flags


def build(frames):
    verts, es, et, ef, nF = build_sphere_mesh(150, 5.0, 0.18, 0); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": 5.0, "jitter": 0.18,
            "p0": 3.72, "seed": 0, "before_frame": 1},
           {"op": "grow_3d", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 64, "conserve_amount": False},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72, "every": 2,
            "max_div": 12, "max_div_frac": 0.02},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]                       # ALIGNED recording
    sched = ["seed_mesh_3d", "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "stress", "seed": 0, "n_frames": frames, "dt": 1.0, "record_cap": frames + 2,
                       "boundary": "free", "dim": 3, "world": [40, 40, 40]},
           "sets": {"vertex": {"n": int(Nv * 30)}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim


for frames in (500, 1000):
    Hf, out = engine_run(build(frames), device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    hs, cc = [], []
    for t in range(T):
        mt = hist[min(t, len(hist) - 1)]; pt = posf[t][:mt["Nv"]].astype(np.float64)
        hs.append(hollow_flags(pt, mt)[2]["frac"]); cc.append(int(mt["nF"]))
    hs = np.array(hs)
    print(f"\n=== {frames} frames  (len posf={T}, hist={len(hist)})  cells 150->{cc[-1]}  "
          f"hollow max={hs.max():.3f} mean={hs.mean():.3f} ===")
    for b in range(0, T, 100):
        blk = hs[b:b + 100]
        print(f"  [{b:4d}-{min(b+100,T):4d})  cells~{cc[min(b+50,T-1)]:4d}  hollow mean={blk.mean():.3f} max={blk.max():.3f}")

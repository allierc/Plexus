#!/usr/bin/env python
"""Are the runs DETERMINISTIC in their overlap? Run the IDENTICAL vesicle_divide config (same seed,
buffer, jitter) at 220 and 300 frames, recording EVERY frame, and print the hollow fraction at the same
sim-frames. If they match through frame 220, the runs are deterministic -> the 300f buckle appears only
AFTER 220 (the movie just LOOKS early because it is compressed to a fixed 7.5s). If they diverge early,
that is a real length-dependent bug."""
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

RADIUS, JITTER, SEED = 5.0, 0.18, 0     # EXACTLY the vesicle driver's constants


def build(frames):
    verts, es, et, ef, nF = build_sphere_mesh(150, RADIUS, JITTER, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.72, "seed": SEED, "before_frame": 1},
           {"op": "grow_3d", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 1e9, "conserve_amount": False},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 20},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72, "every": 2, "max_div": 10},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]     # record EVERY frame -> exact alignment
    sched = ["seed_mesh_3d", "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "align", "seed": SEED, "n_frames": frames, "dt": 1.0, "record_cap": frames + 2,
                       "boundary": "free", "dim": 3, "world": [6 * RADIUS] * 3},
           "sets": {"vertex": {"n": int(Nv * 8)}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim


def series(frames):
    Hf, out = engine_run(build(frames), device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    d = {}
    for t in range(T):
        mt = hist[min(t, len(hist) - 1)]; pt = posf[t][:mt["Nv"]].astype(np.float64)
        d[t] = (hollow_flags(pt, mt)[2]["frac"], int(mt["nF"]))
    return d


d220 = series(220); d300 = series(300)
print("\nsim_frame   220f: hollow (cells)     300f: hollow (cells)")
for t in range(0, 301, 20):
    a = d220.get(t); b = d300.get(t)
    sa = f"{a[0]:.3f} ({a[1]})" if a else "   --      "
    sb = f"{b[0]:.3f} ({b[1]})" if b else "   --"
    print(f"  {t:4d}      {sa:20s}  {sb}")
# do they match through 220?
common = [t for t in d220 if t in d300]
maxdiff = max(abs(d220[t][0] - d300[t][0]) for t in common)
print(f"\nmax |hollow_220 - hollow_300| over the {len(common)} common frames [0..220] = {maxdiff:.4f}")
print("=> IDENTICAL in overlap" if maxdiff < 1e-6 else "=> DIVERGE in overlap (length-dependent!)")

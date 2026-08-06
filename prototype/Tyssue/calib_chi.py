#!/usr/bin/env python
"""Fast spot-count calibration: 2000-cell vesicle, Brusselator RD only (no growth/render), sweep the
diffusion multiplier chi (and d_h) to find ~5 big Turing spots as in Okuda Fig 5. Prints spot count."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
import run_tyssue_fig5 as F

N, FRAMES = 2000, 120


def build(chi, d_h):
    verts, es, et, ef, nF = build_sphere_mesh(N, 5.0, 0.16, 0); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": N, "radius": 5.0, "jitter": 0.16, "p0": 3.9, "seed": 0, "before_frame": 1},
           {"op": "cell_geometry_3d", "at": "cell"}, {"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": 0, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
           {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": d_h, "chi": chi},
           {"op": "cell_react", "at": "cell", "model": "brusselator", "gamma": 2.0, "A": 1.0, "B": 3.0},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.9, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05, "Lambda": 0.2,
            "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 0.02, "relax_iters": 8, "eta": 0.08, "cap_frac": 0.12}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react", "shape_energy_3d"]
    cfg = {"general": {"name": "calibchi", "seed": 0, "n_frames": FRAMES, "dt": 0.02, "record_cap": 8,
                       "boundary": "free", "dim": 3, "world": [80, 80, 80]},
           "sets": {"vertex": {"n": int(Nv * 1.3)},
                    "cell": {"n": int(nF * 1.3), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


for chi, d_h in [(30, 0.7), (100, 0.7), (300, 0.7), (300, 2.0), (900, 2.0)]:
    sim, mesh = build(chi, d_h)
    Hf, out = engine_run(sim, device="cpu")
    a = out["sets"]["cell"]["state"]["chem"][-1][:mesh["nF"], 0]
    thr = float(np.percentile(a, 60))
    spots = F.count_spots(a, mesh, thr)
    print(f"chi={chi:4d} d_h={d_h}  spots={spots}  a_std={a.std():.3f} a_range=[{a.min():.2f},{a.max():.2f}]", flush=True)

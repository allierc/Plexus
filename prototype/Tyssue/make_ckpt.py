#!/usr/bin/env python
"""Generate the smoke_hom homogenised-vesicle CHECKPOINT (archive/smoke_hom/ckpt.npz): seed 900c and run
pure homogenising growth (rho=0.15 baseline, vth=1.35 tight cap, cv=0.4, K_V=3) with NO activation -> a
uniform ~2400-cell vesicle. Saved via ckpt.save_state so round_XX tubing runs can genuinely initialise
from it (load_mesh_3d) + seed big RD spots. dt=1.0 (no RD -> no CFL). Run once."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d, ckpt  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh

R, J, SEED, CELLS, FR = 5.0, 0.16, 0, 900, 110   # gentle homogenisation -> ~2400 clean cells (not over-proliferated)
OUT = os.path.join(HERE, "archive", "smoke_hom"); os.makedirs(OUT, exist_ok=True)


def build():
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": CELLS, "radius": R, "jitter": J, "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"},
           {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03, "a_sw": 0.5, "hill": 4.0, "rho": 0.15, "vth_frac": 1.35},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05, "Lambda": 0.2, "K_V": 3.0, "K_R": 0.05, "mu": 1.0, "dt": 1.0, "relax_iters": 45, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 120},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.3, "p0": 3.90, "every": 2, "max_div": 40, "max_div_frac": 0.02, "cell_set": "cell", "min_cycle": 4, "max_cycle": 16},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 50}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "smoke_hom_ckpt", "seed": SEED, "n_frames": FR, "dt": 1.0, "record_cap": 4, "boundary": "free", "dim": 3, "world": [16 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 10)}, "cell": {"n": int(nF * 10), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim


sim = build()
Hf, out = engine_run(sim, device="cpu")
m = Hf.level("vertex")._mesh
print(f"[make_ckpt] homogenised {CELLS} -> {int(m['nF'])} cells", flush=True)
ckpt.save_state(Hf, os.path.join(OUT, "ckpt.npz"))

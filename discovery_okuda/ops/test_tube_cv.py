#!/usr/bin/env python
"""Tube cell-size uniformity study (user: tube_1 tip cells are too big). Runs the single-tube config
(patch tip) and measures per-cell WEDGE VOLUME split by z: TOP third (tube tip) vs BOTTOM third (base),
mean+/-SD, plus overall CV. Sweeps the Okuda homogenisation knobs: rho (baseline growth so all cells
cycle), vth (v_eq cap = vth*v_ref -> cells cycle in [~2/3,vth] not bulge), K_V (volume stiffness), and
the division DURATION bounds (min/max cell-cycle length). Goal: lower CV, tip volume ~ base volume."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import mesh_ops, chem_ops, t1_ops  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from mesh_ops import build_sphere_mesh, face_geometry_3d
import torch

RADIUS, JITTER, SEED = 5.0, 0.16, 0
NCELLS, FRAMES = 400, 400


def build(grow, rho, vth, K_V, min_cycle, max_cycle):
    verts, es, et, ef, nF = build_sphere_mesh(NCELLS, RADIUS, JITTER, SEED); Nv = verts.shape[0]
    buf = int(Nv * 6.0); cbuf = int(nF * 6.0)
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": NCELLS, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.4},
           {"op": "cell_geometry", "at": "cell"},
           {"op": "cell_chem_seed", "at": "cell", "mode": "patch", "patch_z": 0.90},
           {"op": "cell_grow", "at": "vertex", "cell_set": "cell", "rate": grow,
            "a_sw": 0.5, "hill": 4.0, "rho": rho, "vth_frac": vth},
           {"op": "cell_mechanics", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05,
            "Lambda": 0.2, "K_V": K_V, "K_R": 0.02, "mu": 1.0, "dt": 1.0, "relax_iters": 30, "eta": 0.08, "cap_frac": 0.12},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 60},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.4,
            "p0": 3.90, "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell",
            "min_cycle": min_cycle, "max_cycle": max_cycle},
           {"op": "topo_record", "at": "vertex", "every": 1}]
    sched = ["mesh_seed", "cell_geometry", "cell_chem_seed", "cell_grow",
             "cell_mechanics", "edge_flip", "cell_divide", "topo_record"]
    cfg = {"general": {"name": "tubecv", "seed": SEED, "n_frames": FRAMES, "dt": 1.0, "record_cap": FRAMES + 2,
                       "boundary": "free", "dim": 3, "world": [16 * RADIUS] * 3},
           "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                        "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def vols(mesh, pos):
    _, _, cen, vf = face_geometry_3d(torch.as_tensor(pos), torch.as_tensor(np.asarray(mesh["E_srce"])),
                                     torch.as_tensor(np.asarray(mesh["E_trgt"])), torch.as_tensor(np.asarray(mesh["E_face"])), mesh["nF"])
    return vf.numpy(), cen.numpy()


def run(label, grow, rho, vth, K_V, min_cycle, max_cycle):
    sim, mesh0 = build(grow, rho, vth, K_V, min_cycle, max_cycle)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    mt = hist[-1] if hist else mesh0; pt = posf[T - 1][:mt["Nv"]].astype(np.float64)
    vf, cen = vols(mt, pt); z = cen[:, 2]
    zt, zb = np.percentile(z, 66), np.percentile(z, 33)
    top = vf[z > zt]; bot = vf[z < zb]
    ext = pt.max(0) - pt.min(0); aspect = ext.max() / max(ext.min(), 1e-6)
    print(f"{label:16s} cells={mt['nF']:4d} aspect={aspect:.2f}  CV_all={vf.std()/vf.mean():.3f}  "
          f"TOP v={top.mean():.3f}+/-{top.std():.3f}  BOT v={bot.mean():.3f}+/-{bot.std():.3f}  "
          f"top/bot={top.mean()/max(bot.mean(),1e-9):.2f}x", flush=True)


# baseline (legacy bulge, no duration) vs homogenised variants
run("baseline",     0.008, 0.0,  1.35, 1.0, 0,  10**9)   # legacy: activator-only bulge (like tube_1)
run("rho+cap",      0.008, 0.15, 1.35, 3.0, 0,  10**9)   # Okuda: baseline growth + v_eq cap + stiff K_V
run("rho+cap+dur",  0.008, 0.15, 1.35, 3.0, 3,  14)      # + bounded cell-cycle duration
run("strong",       0.010, 0.20, 1.30, 4.0, 4,  12)      # stronger homogenisation

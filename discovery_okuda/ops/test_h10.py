#!/usr/bin/env python
"""H10 (perfecting: the cells are triangular vs the rounder Voronoi cells of fig4_coral). Hypothesis:
more SURFACE TENSION (line tension Lambda + cortical contractility Gamma) rounds AVM cells and removes
division/T1 slivers. New authored metrics: mean shape index P/sqrt(A) (round hexagon ~3.72, sliver higher)
and sliver_frac = fraction of cells with <=4 sides. Keep uniformity (vol_cv) low. Short 150c/200f."""
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
from topology_ops import rings_from_flat_3d
import torch

R, J, SEED, FR = 5.0, 0.18, 0, 200


def build(Lam, Gam, K_V, p0):
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J, "p0": p0, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry", "at": "cell"},
           {"op": "cell_grow", "at": "vertex", "cell_set": "cell", "rate": 0.03, "a_sw": 50.0, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
           {"op": "cell_mechanics", "at": "vertex", "p0": p0, "K_A": 1.0, "K_P": 1.0, "Gamma": Gam, "Lambda": Lam, "K_V": K_V, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 30, "eta": 0.08, "cap_frac": 0.12},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": p0, "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": 4, "max_cycle": 12},
           {"op": "topo_record", "at": "vertex", "every": 1}]
    sched = ["mesh_seed", "cell_geometry", "cell_grow", "cell_mechanics", "edge_flip", "cell_divide", "topo_record"]
    cfg = {"general": {"name": "h10", "seed": SEED, "n_frames": FR, "dt": 1.0, "record_cap": FR + 2, "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 8)}, "cell": {"n": int(nF * 8), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def roundness(mt, pt):
    es, et, ef, nF = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"]
    area, perim, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(es), torch.as_tensor(et), torch.as_tensor(ef), nF)
    area, perim, vf = area.numpy(), perim.numpy(), vf.numpy()
    ok = area > 1e-9
    shape = perim[ok] / np.sqrt(area[ok] + 1e-12)
    rings = rings_from_flat_3d(es, et, ef, nF); nsides = np.array([len(r) if r is not None else 0 for r in rings])
    sliver = float((nsides[ok & (nsides > 0)] <= 4).mean())
    vc = float(vf[np.abs(vf) > 1e-9].std() / (np.abs(vf[np.abs(vf) > 1e-9].mean()) + 1e-9))
    return float(np.median(shape)), sliver, vc


for name, Lam, Gam, K_V, p0 in [("baseline", 0.5, 0.1, 4.0, 3.72), ("more_tension", 2.0, 0.3, 2.0, 3.72),
                                 ("high_tension", 4.0, 0.5, 1.5, 3.60), ("lowp0_tension", 3.0, 0.4, 2.0, 3.50)]:
    sim, mesh0 = build(Lam, Gam, K_V, p0)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    mt = hist[-1] if hist else mesh0; pt = posf[T - 1][:mt["Nv"]].astype(np.float64)
    sh, sl, vc = roundness(mt, pt)
    print(f"{name:14s} Lam={Lam} Gam={Gam} K_V={K_V} p0={p0}  cells->{mt['nF']:4d}  shape_idx={sh:.3f}  sliver_frac={sl:.3f}  vol_cv={vc:.3f}", flush=True)

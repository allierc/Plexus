#!/usr/bin/env python
"""H1: vesicle cell non-uniformity comes from growing by INFLATION (grow_3d ramps V0f -> big
cells) rather than PROLIFERATION. Test: short A/B, same seed/length, measure cell-area CV.
  A inflation    = grow_3d (current vesicle_grow_divide)
  B proliferation= morphogen_growth rho=1 (all cells) + v_eq capped at vth*v_ref (Okuda)
H1 validated if area_cv(B) << area_cv(A). Short (150c/180f), metrics only."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tyssue_diag import hollow_metric, hollow_flags

R, J, SEED, FR = 5.0, 0.18, 0, 180


def build(mode):
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    buf, cbuf = int(Nv * 8), int(nF * 8)
    common_seed = {"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J,
                   "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": 0.4}
    if mode == "inflation":
        grow = {"op": "grow_3d", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 15.625, "conserve_amount": False}
        gname = "grow_3d"; pre = ["seed_mesh_3d", "grow_3d"]; extra = [common_seed, grow]
    else:  # proliferation (Okuda uniform): rho=1 + v_eq cap; needs cell set + geometry
        extra = [common_seed, {"op": "cell_geometry_3d", "at": "cell"},
                 {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03,
                  "a_sw": 0.5, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4}]
        pre = ["seed_mesh_3d", "cell_geometry_3d", "grow_3d"]
    ops = extra + [
        {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1,
         "Lambda": 0.5, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
        {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
        {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.4, "p0": 3.72,
         "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell"},
        {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = pre + ["shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": f"h1_{mode}", "seed": SEED, "n_frames": FR, "dt": 1.0, "record_cap": FR + 2,
                       "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                        "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


for mode in ("inflation", "proliferation"):
    sim, mesh0 = build(mode)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    cvs = []
    for tt in np.linspace(0, T - 1, 24).astype(int):
        mt, pt = frame(int(tt)); _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]
        cvs.append(float(ar.std() / (ar.mean() + 1e-9)) if ar.size else 0)
    mt, pt = frame(T - 1); _, arT, _ = hollow_metric(pt, mt); arT = arT[arT > 0]
    hollow = float(hollow_flags(pt, mt)[2]["frac"])
    print(f"{mode:14s} cells 150->{mt['nF']:4d}  area_cv final={arT.std()/arT.mean():.3f} "
          f"mean={np.mean(cvs):.3f} max={np.max(cvs):.3f}  hollow={hollow:.3f}", flush=True)

#!/usr/bin/env python
"""H2/H3 (small incremental steps on the H1 winner = proliferation growth). Each condition adds ONE
factor to isolate its effect on cell-area CV:
  base    proliferation, cv=0.40, no duration bound, K_V=1
  +tight  H2: cv=0.15 + bounded cell-cycle duration (min4,max12)   -- tighter division timing
  +stiff  H3: +K_V=4                                                -- cells track v_eq tightly
Short (150c/180f), metrics only. Validated if area_cv drops step by step."""
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


def build(cv, K_V, mn, mx):
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J,
            "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": cv},
           {"op": "cell_geometry_3d", "at": "cell"},
           {"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03,
            "a_sw": 0.5, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1,
            "Lambda": 0.5, "K_V": K_V, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv, "p0": 3.72,
            "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": mn, "max_cycle": mx},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "morphogen_growth_3d", "shape_energy_3d",
             "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "h2", "seed": SEED, "n_frames": FR, "dt": 1.0, "record_cap": FR + 2,
                       "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 8)}, "cell": {"n": int(nF * 8), "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                                       "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def score(sim, mesh0):
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    fr = lambda t: (hist[min(t, len(hist) - 1)] if hist else mesh0, posf[t][:(hist[min(t, len(hist) - 1)] if hist else mesh0)["Nv"]].astype(np.float64))
    cvs = []
    for tt in np.linspace(0, T - 1, 24).astype(int):
        mt, pt = fr(int(tt)); _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]
        cvs.append(float(ar.std() / (ar.mean() + 1e-9)) if ar.size else 0)
    mt, pt = fr(T - 1); _, arT, _ = hollow_metric(pt, mt); arT = arT[arT > 0]
    return mt["nF"], arT.std() / arT.mean(), np.mean(cvs), float(hollow_flags(pt, mt)[2]["frac"])


for label, cv, K_V, mn, mx in [("base", 0.40, 1.0, 0, 10 ** 9), ("+tight", 0.15, 1.0, 4, 12), ("+stiff", 0.15, 4.0, 4, 12)]:
    sim, mesh0 = build(cv, K_V, mn, mx)
    n, cvf, cvm, ho = score(sim, mesh0)
    print(f"{label:8s} cv={cv} K_V={K_V} dur=({mn},{mx})  cells->{n:4d}  area_cv final={cvf:.3f} mean={cvm:.3f}  hollow={ho:.3f}", flush=True)

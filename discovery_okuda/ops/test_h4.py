#!/usr/bin/env python
"""H4 (tube stage): can UNIFORM cells still form protruding tubes? Apply the vesicle homogenisation
recipe (v_eq cap + tight division cv=0.15 + duration bounds + K_V=4) to the multi-cone tube case, and
test whether a proliferation-rate CONTRAST (low baseline rho vs cone boost) drives protrusion while
keeping cells uniform. Short (800c/250f). Metrics: protr (95pct/median radius, want>1.4), area_cv
(want<0.3), hollow, spots. Conditions probe the contrast/concentration knobs:
  A concentrated  rho=0.05 cone=11 rate=0.05 vth=1.5   (strong contrast, narrow, uniform)
  B moderate      rho=0.10 cone=14 rate=0.04 vth=1.5
  C bulge-control rho=0.05 cone=11 rate=0.05 vth=3.0   (allow some v_eq bulge -> protrusion drive?)"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import plexus.operators  # noqa
import mesh_ops, chem_ops, t1_ops  # noqa
from plexus.engine import run as engine_run
from mesh_ops import build_sphere_mesh
from diag_tools import hollow_metric, hollow_flags
from topology_ops import rings_from_flat_3d
import run_tyssue_fig5 as F

CELLS, FR = 800, 250


def score(name, cone, rho, rate, vth):
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, F.RADIUS, F.JITTER, F.SEED); Nv = verts.shape[0]
    buf, cbuf = int(Nv * 4), int(nF * 4)
    # fig5 make_spec(name, n_cells, frames, n_spots, cone, grow, cv, rho, vth, K_V, min_cyc, max_cyc, buf, cbuf)
    sim, _ = F.make_spec(name, CELLS, FR, 5, cone, rate, 0.15, rho, vth, 4.0, 4, 12, buf, cbuf)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    mt = hist[-1] if hist else dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
    pt = posf[T - 1][:mt["Nv"]].astype(np.float64); aT = chemf[T - 1][:mt["nF"], 0]
    _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]; area_cv = ar.std() / ar.mean()
    rings = rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
    rad = np.array([np.linalg.norm(pt[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings]); rad = rad[rad > 0]
    protr = np.percentile(rad, 95) / (np.median(rad) + 1e-9)
    hollow = float(hollow_flags(pt, mt)[2]["frac"])
    spots = F.count_spots(aT, mt, float(np.percentile(aT, 70)))
    print(f"{name:14s} cells->{mt['nF']:4d}  protr={protr:.3f}  area_cv={area_cv:.3f}  hollow={hollow:.3f}  spots={spots}", flush=True)


score("A_concentr", 11.0, 0.05, 0.05, 1.5)
score("B_moderate", 14.0, 0.10, 0.04, 1.5)
score("C_bulge",    11.0, 0.05, 0.05, 3.0)

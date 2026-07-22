#!/usr/bin/env python
"""H5 (Loop-III, literature-grounded): tubes protrude because the non-activated body is VOLUME-LOCKED
(Okuda: k_v strong, only activated cells grow) -> it pins the body radius so activated growth protrudes.
H4 failed because rho>0 grew the whole body (no incompressible surround). Test rho=0 strict, varying the
bulge cap and cone width. Short 800c/250f. protr>1.3 would validate."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tyssue_diag import hollow_metric, hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d
import run_tyssue_fig5 as F

CELLS, FR = 2000, 200   # Fig5-scale start: seed cells already ~v_ref -> body/tube disparity should shrink


def score(name, cone, rate, vth):
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, F.RADIUS, F.JITTER, F.SEED); Nv = verts.shape[0]
    # rho=0 + max_cycle=inf -> body is truly VOLUME-LOCKED (no growth, no forced division); only cones grow
    sim, _ = F.make_spec(name, CELLS, FR, 5, cone, rate, 0.15, 0.0, vth, 4.0, 0, 10 ** 9, int(Nv * 4), int(nF * 4))
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
    print(f"{name:16s} cells->{mt['nF']:4d}  protr={protr:.3f}  area_cv={area_cv:.3f}  hollow={hollow:.3f}  spots={spots}", flush=True)


# fast conditions first: does a volume-locked body force protrusion with UNIFORM cells (vth=1.5)?
score("H5_prolif12", 12.0, 0.05, 1.5)   # rho=0, uniform, cone 12
score("H5_prolif8",   8.0, 0.06, 1.5)   # rho=0, uniform, narrow cone 8 (more concentrated)
score("H5_bulge18",  18.0, 0.05, 2.0)   # rho=0, some bulge, wider -- contrast test

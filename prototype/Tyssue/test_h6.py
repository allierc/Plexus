#!/usr/bin/env python
"""H6 (literature-grounded, user hypothesis): tubes protrude at the MOVING activator FRONT of a real RD
spot -- Okuda p7 "the activator stayed around the tip, from which the tubes continuously grew; the spot
size is maintained -> constant diameter". Static cones activate a broad fixed region -> flat. So: add the
Brusselator RD (self-maintaining spots) to the homogenised vesicle recipe (v_eq cap + cv=0.15 + dur +
K_V=4), couple activator->growth (low body rho + activator boost), and test whether the RD spots
protrude while cells stay uniform behind the front. Short (400c, 300f, dt=0.02). protr>1.3 validates."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh, face_geometry_3d
from tyssue_diag import hollow_metric, hollow_flags
from tyssue_topology_ops3d import rings_from_flat_3d
import run_tyssue_fig5 as F
import torch


def vol_cv(mt, pt):   # cleaner uniformity metric than area_cv when tubes exist (area confounds with geometry)
    _, _, _, vf = face_geometry_3d(torch.as_tensor(pt), torch.as_tensor(np.asarray(mt["E_srce"])),
                                   torch.as_tensor(np.asarray(mt["E_trgt"])), torch.as_tensor(np.asarray(mt["E_face"])), mt["nF"])
    vf = vf.numpy(); vf = vf[np.abs(vf) > 1e-9]; return float(vf.std() / (np.abs(vf.mean()) + 1e-9))

R, J, SEED = 5.0, 0.16, 0
CELLS, FR = 400, 300


def build(chi, rho, vth, rate, a_sw, gamma=2.0):
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": CELLS, "radius": R, "jitter": J,
            "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}, {"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
           {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": chi},
           {"op": "cell_react", "at": "cell", "model": "brusselator", "gamma": gamma, "A": 1.0, "B": 3.0},
           # activator->growth: low body rho + activator boost; v_eq capped (uniform behind the front)
           {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": rate,
            "a_sw": a_sw, "hill": 4.0, "rho": rho, "vth_frac": vth},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.05,
            "Lambda": 0.2, "K_V": 4.0, "K_R": 0.02, "mu": 1.0, "dt": 0.02, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 60},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": 3.90,
            "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": 4, "max_cycle": 30},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react",
             "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "h6", "seed": SEED, "n_frames": FR, "dt": 0.02, "record_cap": FR + 2,
                       "boundary": "free", "dim": 3, "world": [16 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 4)}, "cell": {"n": int(nF * 4), "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                                       "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def run(name, chi, rho, vth, rate, a_sw, gamma=2.0):
    sim, mesh0 = build(chi, rho, vth, rate, a_sw, gamma)   # thread gamma through (H9 spot-size sweep)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    mt = hist[-1] if hist else mesh0; pt = posf[T - 1][:mt["Nv"]].astype(np.float64); aT = chemf[T - 1][:mt["nF"], 0]
    _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]; area_cv = ar.std() / ar.mean()
    rings = rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
    rad = np.array([np.linalg.norm(pt[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings]); rad = rad[rad > 0]
    protr = np.percentile(rad, 95) / (np.median(rad) + 1e-9)
    hollow = float(hollow_flags(pt, mt)[2]["frac"]); spots = F.count_spots(aT, mt, float(np.percentile(aT, 70)))
    vc = vol_cv(mt, pt)
    print(f"{name:10s} chi={chi} rho={rho} vth={vth}  cells->{mt['nF']:4d}  protr={protr:.3f}  vol_cv={vc:.3f}  "
          f"area_cv={area_cv:.3f}  hollow={hollow:.3f}  spots={spots}  a_range=[{aT.min():.2f},{aT.max():.2f}]", flush=True)


# H6 RD-front coupling: low rho (near-locked body -> protrusion, per H5) + activator boost at the RD spots.
# a_sw~2.5 matches the Brusselator activator (peaks ~4). Judge uniformity by vol_cv, not area_cv.
# H9: lower gamma (slower reaction) -> longer Turing wavelength -> FEWER, BIGGER spots (no CFL hit) that
# may be large enough to protrude, unlike the 112 fine spots at gamma=2. rho=0.05 (near-locked body).
run("h9_g20", 4.0, 0.05, 1.5, 0.05, 2.5, 2.0)   # baseline fine spots
run("h9_g06", 4.0, 0.05, 1.5, 0.05, 1.5, 0.6)   # bigger spots
run("h9_g03", 4.0, 0.05, 1.5, 0.05, 1.2, 0.3)   # biggest spots

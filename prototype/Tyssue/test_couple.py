#!/usr/bin/env python
"""SMALL-STEP INSIGHT (h6_2k blew up: too many large changes at once). Isolate the ONE thing we don't
understand -- how activator->growth coupling turns a clean RD pattern into protrusion -- on a SMALL 150-cell
clean vesicle, gentle everything else (slow rate, dt small), sweeping only the baseline-growth knob rho:
  rho=1.0 -> every cell grows == homogenised sphere, NO localisation (known-good, our validated vh_rd)
  rho=0.0 -> only activated (red) cells grow == fully-locked body (what blew up at 2000c)
Ladder between them and read protr / hollow_n_peak / area_cv / spots at 150c/150f. Where does gentle
protrusion first appear, and where does it start to hollow?  Local + fast (~30s each)."""
from __future__ import annotations
import os, sys, tempfile, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_rd_ops, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tissue_analysis import frame_metrics
import run_tyssue_fig5 as F

R, J, SEED, CELLS, FR = 5.0, 0.16, 0, 150, 150


def build(rho, rate=0.02, gamma=2.0, vth=1.3, mdf=0.02, dev=2):
    verts, es, et, ef, nF = build_sphere_mesh(CELLS, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": CELLS, "radius": R, "jitter": J, "p0": 3.90, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}, {"op": "cell_adjacency", "at": "cell"},
           {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
           {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": 4.0},
           {"op": "cell_react", "at": "cell", "model": "brusselator", "gamma": gamma, "A": 1.0, "B": 3.0},
           # locked body (rho=0) so growth is fully localised to the red cells; only rate (coupling strength) varies
           {"op": "grow_3d", "at": "vertex", "cell_set": "cell", "rate": rate, "a_sw": 2.5, "hill": 4.0, "rho": rho, "vth_frac": vth},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.90, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.2, "Lambda": 0.6, "K_V": 4.0, "K_R": 0.02, "mu": 1.0, "dt": 0.02, "relax_iters": 30, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.28, "every": 1, "max_flips": 40},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": 3.90, "every": dev, "max_div": 40, "max_div_frac": mdf, "cell_set": "cell", "min_cycle": 4, "max_cycle": 30},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched = ["seed_mesh_3d", "cell_geometry_3d", "cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react",
             "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "couple", "seed": SEED, "n_frames": FR, "dt": 0.02, "record_cap": FR + 2, "boundary": "free", "dim": 3, "world": [16 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 16)}, "cell": {"n": int(nF * 16), "state": {"chem": {"width": 2, "integration": "first_order"}, "cen": {"width": 3}, "area": {"width": 1}}}},  # big buffer: proliferation must not hit the cell-slot ceiling (890 confound)
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


print(f"# 150c DIVISION ladder at rate=0.06 (protrusion onset): can faster division convert ballooning-hollow into clean proliferation-tubes?", flush=True)
#            label            vth   mdf  every
LAD = [("baseline",          1.30, 0.02, 2),
       ("more_throughput",   1.30, 0.15, 1),
       ("divide_sooner",     1.15, 0.15, 1),
       ("max_prolif",        1.10, 0.30, 1)]
for label, vth, mdf, dev in LAD:
    sim, mesh0 = build(0.0, 0.06, 2.0, vth, mdf, dev); rho, rate = 0.0, 0.06
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]
    chemf = out["sets"]["cell"]["state"]["chem"]; T = posf.shape[0]
    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64), chemf[t][:mt["nF"], 0]
    hn, acv = 0, 0.0
    for t in np.unique(np.linspace(0, T - 1, 20).astype(int)):
        mt, pt, _ = frame(int(t)); m = frame_metrics(pt, mt); hn = max(hn, m["hollow_n"]); acv = max(acv, m["area_cv"])
    mtT, ptT, aT = frame(T - 1); mE = frame_metrics(ptT, mtT)
    spots = int(F.count_spots(aT, mtT, float(np.percentile(aT, 70))))
    print(f"{label:16s} vth={vth} mdf={mdf} ev={dev}  cells->{mE['cells']:4d}  protr={mE['protr']:.2f}  "
          f"hollow_peak={hn:3d}  area_cv_peak={acv:.2f}  tube_diam={mE['tube_diam']:.2f} n_tubes={mE['n_tubes']}", flush=True)

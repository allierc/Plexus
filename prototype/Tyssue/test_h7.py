#!/usr/bin/env python
"""H7 sanity: is the Brusselator RD PURE COLOURING on the homogenised vesicle recipe? Add RD but do NOT
couple activator->growth (morphogen a_sw huge -> drive=rho=1 uniform), compare vertex trajectory to the
SAME run WITHOUT RD. If byte-identical -> RD is coloring (validated), and the vesicle still homogenises
with the RD ops present. Both dt=0.02, short 150c/250f."""
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

R, J, SEED, FR = 5.0, 0.18, 0, 250


def build(with_rd):
    verts, es, et, ef, nF = build_sphere_mesh(150, R, J, SEED); Nv = verts.shape[0]
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": 150, "radius": R, "jitter": J,
            "p0": 3.72, "seed": SEED, "before_frame": 1, "vseed_cv": 0.15},
           {"op": "cell_geometry_3d", "at": "cell"}]
    sched = ["seed_mesh_3d", "cell_geometry_3d"]
    if with_rd:                                             # RD ops (colour only): adjacency + seed + diffuse + react
        ops += [{"op": "cell_adjacency", "at": "cell"},
                {"op": "seed_cell_rd", "at": "cell", "seed": SEED, "before_frame": 3, "mode": "noise", "A": 1.0, "B": 3.0, "noise": 0.04},
                {"op": "cell_diffuse", "at": "cell", "d_a": 0.05, "d_h": 0.7, "chi": 4.0},
                {"op": "cell_react", "at": "cell", "model": "brusselator", "gamma": 2.0, "A": 1.0, "B": 3.0}]
        sched += ["cell_adjacency", "seed_cell_rd", "cell_diffuse", "cell_react"]
    # UNMODULATED growth: a_sw=50 so the activator never fires the Hill -> drive=rho=1 uniform (homogenised recipe)
    ops += [{"op": "morphogen_growth_3d", "at": "vertex", "cell_set": "cell", "rate": 0.03,
             "a_sw": 50.0, "hill": 4.0, "rho": 1.0, "vth_frac": 1.4},
            {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Gamma": 0.1,
             "Lambda": 0.5, "K_V": 4.0, "K_R": 0.4, "mu": 1.0, "dt": 0.02, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
            {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 30},
            {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": 0.15, "p0": 3.72,
             "every": 2, "max_div": 12, "max_div_frac": 0.03, "cell_set": "cell", "min_cycle": 4, "max_cycle": 12},
            {"op": "topo_snapshot_3d", "at": "vertex", "every": 1}]
    sched += ["morphogen_growth_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "h7", "seed": SEED, "n_frames": FR, "dt": 0.02, "record_cap": FR + 2,
                       "boundary": "free", "dim": 3, "world": [10 * R] * 3},
           "sets": {"vertex": {"n": int(Nv * 12)}, "cell": {"n": int(nF * 12), "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                                         "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


res = {}
for tag in ("no_rd", "with_rd"):
    sim, mesh0 = build(tag == "with_rd")
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]
    mt = hist[-1] if hist else mesh0; pt = posf[T - 1][:mt["Nv"]].astype(np.float64)
    _, ar, _ = hollow_metric(pt, mt); ar = ar[ar > 0]
    a = out["sets"]["cell"]["state"]["chem"][T - 1][:mt["nF"], 0]
    res[tag] = dict(posf=posf, cells=mt["nF"], area_cv=float(ar.std() / ar.mean()),
                    hollow=float(hollow_flags(pt, mt)[2]["frac"]), a_range=(float(a.min()), float(a.max())))
    print(f"{tag:8s} cells->{res[tag]['cells']:4d}  area_cv={res[tag]['area_cv']:.3f}  hollow={res[tag]['hollow']:.3f}  "
          f"activator_range={res[tag]['a_range']}", flush=True)

# trajectory comparison: RD is pure colouring iff vertex positions are identical
T = min(res['no_rd']['posf'].shape[0], res['with_rd']['posf'].shape[0])
N = min(res['no_rd']['posf'].shape[1], res['with_rd']['posf'].shape[1])
d = np.abs(res['no_rd']['posf'][:T, :N] - res['with_rd']['posf'][:T, :N]).max()
print(f"\nmax |pos(no_rd) - pos(with_rd)| over all frames = {d:.2e}")
print("=> RD is PURE COLOURING (trajectories identical)" if d < 1e-4 else "=> RD PERTURBS the mechanics (not pure colouring!)")

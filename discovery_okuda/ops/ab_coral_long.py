#!/usr/bin/env python
"""Investigation: why does a LONG rd_coral_grow run develop hollow cells while the short one is clean?

rd_coral_grow (OK)   = 150c / 220f, cv=0,   max_div=10 (fixed)
"rd_coral_grow_long" = same params, many more frames -> runs through the full proliferation phase.

Two hypotheses about the extra frames (both dropped by rd_coral_grow_big, which is clean):
  H1  cv=0 -> SYNCHRONISED division waves inject correlated strain the relaxation can't dissipate.
  H2  max_div is a FIXED absolute cap (10) set from the INITIAL count, so as the live count grows the
      ready cells BACKLOG, keep ramping v_eq while queued, then divide oversized -> tip strain.

Run the long spec under: baseline / +cv / +live-cap / +both, count hollow cells over the rollout.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml, tempfile
import plexus.operators  # noqa
import mesh_ops, chem_ops, t1_ops  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from diag_tools import hollow_flags
import run_tyssue_rd as R

FRAMES = 600           # LONG: well past the growth plateau (~scale 2.5 reached ~frame 306 at rate 0.003)
N_CELLS = 150


def build(cv, max_div_frac, grow=0.003):
    """rd_coral_grow spec (GS coral + uniform growth + division), long, with the two knobs exposed."""
    rd = R.GS; dt = rd["dt"]; rec_cap = 300
    sstride = max(1, (FRAMES + rec_cap) // rec_cap)
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": N_CELLS, "radius": R.RADIUS,
            "jitter": R.JITTER, "p0": 3.72, "seed": R.SEED, "before_frame": 1, "vseed_cv": cv},
           {"op": "cell_grow", "at": "vertex", "rate": grow, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 15.625, "conserve_amount": False},
           {"op": "cell_mechanics", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0,
            "Gamma": 0.1, "Lambda": 0.5, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": dt,
            "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": max(20, N_CELLS // 15)},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "cycle_cv": cv,
            "p0": 3.72, "every": 2, "max_div": 10, "max_div_frac": max_div_frac, "cell_set": "cell"},
           {"op": "cell_geometry", "at": "cell"}, {"op": "cell_neighbours", "at": "cell"},
           {"op": "cell_chem_seed", "at": "cell", "seed": R.SEED, "before_frame": 3, **rd["seed"]},
           {"op": "cell_chem_diffuse", "at": "cell", **rd["diffuse"]}, {"op": "cell_chem_react", "at": "cell", **rd["react"]},
           {"op": "topo_record", "at": "vertex", "every": sstride}]
    sched = ["mesh_seed", "cell_grow", "cell_mechanics", "edge_flip", "cell_divide",
             "cell_geometry", "cell_neighbours", "cell_chem_seed", "cell_chem_diffuse", "cell_chem_react", "topo_record"]
    mesh0, nF = R._mesh(N_CELLS); Nv = mesh0["Nv"]; buf = int(Nv * 6.0); cbuf = int(nF * 6.0)
    cfg = {"general": {"name": "coral_long_ab", "seed": R.SEED, "n_frames": FRAMES, "dt": dt, "record_cap": rec_cap,
                       "boundary": "free", "dim": 3, "world": [8 * R.RADIUS] * 3},
           "sets": {"vertex": {"n": buf}, "cell": {"n": cbuf, "state": {"chem": {"width": 2, "integration": "first_order"},
                                                                        "cen": {"width": 3}, "area": {"width": 1}}}},
           "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, mesh0


def run(label, cv, max_div_frac):
    sim, mesh0 = build(cv, max_div_frac)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]; Tn = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    hs, cellcount = [], []
    for tt in np.linspace(0, Tn - 1, 30).astype(int):
        mt, pt = frame(int(tt)); hs.append(hollow_flags(pt, mt)[2]["frac"]); cellcount.append(int(mt["nF"]))
    mtT, _ = frame(Tn - 1)
    print(f"{label:16s} cv={cv} frac={max_div_frac}  hollow max={max(hs):.3f} mean={np.mean(hs):.3f} "
          f"final={hs[-1]:.3f}  cells->{cellcount[-1]} (+{int(emesh.get('n_div',0))} div, "
          f"{int(emesh.get('n_t1',0))} T1)", flush=True)
    return dict(label=label, hollow_max=max(hs), hollow_mean=float(np.mean(hs)), final=hs[-1])


for label, cv, frac in [("baseline_long", 0.0, 0.0),   # reproduce the hollow problem
                        ("+cv",          0.4, 0.0),   # H1: desync cell cycle
                        ("+livecap",     0.0, 0.05),  # H2: live-scaled division cap
                        ("+both",        0.4, 0.05)]:  # = the rd_coral_grow_big recipe (+ live cap)
    run(label, cv, frac)

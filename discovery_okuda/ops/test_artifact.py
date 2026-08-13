#!/usr/bin/env python
"""CONFIRM the hollow is a recording-misalignment artifact. Replicate the twin's recording (topo_snapshot
every=2, record_cap=300) and check: does the position array (posf, engine stride) have the SAME length as
the mesh-history (hist, topo stride)? If not, hollow_flags(posf[t], hist[t]) pairs positions from one
sim-frame with topology from another -> spurious 'hollow'. Compare NAIVE pairing (what the twin did) vs a
correct same-length pairing."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import mesh_ops, t1_ops  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from mesh_ops import build_sphere_mesh
from diag_tools import hollow_flags


def build(frames, every, rec_cap):
    verts, es, et, ef, nF = build_sphere_mesh(150, 5.0, 0.18, 0); Nv = verts.shape[0]
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": 150, "radius": 5.0, "jitter": 0.18,
            "p0": 3.72, "seed": 0, "before_frame": 1},
           {"op": "cell_grow", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 1e9, "conserve_amount": False},
           {"op": "cell_mechanics", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 20},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72, "every": 2, "max_div": 10},
           {"op": "topo_record", "at": "vertex", "every": every}]
    sched = ["mesh_seed", "cell_grow", "cell_mechanics", "edge_flip", "cell_divide", "topo_record"]
    cfg = {"general": {"name": "artifact", "seed": 0, "n_frames": frames, "dt": 1.0, "record_cap": rec_cap,
                       "boundary": "free", "dim": 3, "world": [30, 30, 30]},
           "sets": {"vertex": {"n": int(Nv * 8)}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path); return sim


for tag, every, rec_cap in [("twin (every=2, cap=300)", 2, 300), ("aligned (every=1, cap=big)", 1, 400)]:
    Hf, out = engine_run(build(300, every, rec_cap), device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]
    T = posf.shape[0]; H = len(hist)
    hs = []
    for tt in np.linspace(0, T - 1, 40).astype(int):     # EXACT twin sampling: mt=hist[min(t,H-1)], pt=posf[t]
        mt = hist[min(int(tt), H - 1)]; pt = posf[int(tt)][:mt["Nv"]].astype(np.float64)
        hs.append(hollow_flags(pt, mt)[2]["frac"])
    print(f"{tag:30s}  len(posf)={T}  len(hist)={H}  "
          f"{'MISALIGNED' if T != H else 'aligned':11s}  hollow max={max(hs):.3f} mean={np.mean(hs):.3f}")

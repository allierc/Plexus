#!/usr/bin/env python
"""A/B the two anti-inversion fixes on the case that explodes (mechanics-only vesicle_divide, 400 f,
large buffer): (1) anti-inversion FILTERED STEP in cell_mechanics (block moves that push a face toward
signed-volume inversion), (2) LOCAL DAUGHTER RELAX in cell_divide (heal the fresh caps at birth). Hollows
are division-only inverted caps, so a working fix must drop the inverted-cap fraction hard."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import mesh_ops, t1_ops  # noqa (no RD)
import plexus.schema as S
from plexus.engine import run as engine_run
from mesh_ops import build_sphere_mesh
from diag_tools import hollow_metric

RADIUS, JITTER, SEED = 5.0, 0.16, 0
FRAMES, N_CELLS = 400, 150


def split(pos, mesh):
    dev, area, ndeg = hollow_metric(pos, mesh)
    devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    tiny = area < 0.15 * med; inv = (devd > 50.0) & (~tiny); under = ndeg < 3
    return dict(total=float((tiny | inv | under).mean()), inverted=float(inv.mean()))


def build(antiinv=0.0, local_relax=0, K_bend=0.0):
    verts, es, et, ef, nF = build_sphere_mesh(N_CELLS, RADIUS, JITTER, SEED)
    Nv = verts.shape[0]; buf = int(Nv * 30.0)
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": N_CELLS, "radius": RADIUS,
            "jitter": JITTER, "p0": 3.72, "seed": SEED, "before_frame": 1},
           {"op": "cell_grow", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 1e9, "conserve_amount": False},
           {"op": "cell_mechanics", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26,
            "eta": 0.08, "cap_frac": 0.12, "antiinv": antiinv, "K_bend": K_bend},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": max(20, N_CELLS // 15)},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72,
            "every": 2, "max_div": 10, "local_relax": local_relax},
           {"op": "topo_record", "at": "vertex", "every": max(1, (FRAMES + 300) // 300)}]
    sched = ["mesh_seed", "cell_grow", "cell_mechanics", "edge_flip", "cell_divide", "topo_record"]
    cfg = {"general": {"name": "fix_ab", "seed": SEED, "n_frames": FRAMES, "dt": 1.0, "record_cap": 300,
                       "boundary": "free", "dim": 3, "world": [6 * RADIUS] * 3},
           "sets": {"vertex": {"n": buf}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def run(label, antiinv=0.0, local_relax=0, K_bend=0.0):
    sim, mesh0 = build(antiinv, local_relax, K_bend)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    ser = []
    for tt in np.linspace(0, T - 1, 40).astype(int):
        mt, pt = frame(int(tt)); ser.append(split(pt, mt))
    print(f"{label:28s} cells->{int(frame(T-1)[0]['nF'])} (+{int(emesh.get('n_div',0))} div)  "
          f"TOTAL max={max(s['total'] for s in ser):.3f} mean={np.mean([s['total'] for s in ser]):.3f}  "
          f"INVERTED mean={np.mean([s['inverted'] for s in ser]):.3f}", flush=True)


run("baseline")
run("K_bend=15",   K_bend=15.0)
run("K_bend=40",   K_bend=40.0)
run("K_bend=100",  K_bend=100.0)

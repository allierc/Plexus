#!/usr/bin/env python
"""The hollow is emergent GLOBAL BUCKLING (single divisions are clean), so test the growth/relaxation
REGIME levers on the exploding case (vesicle_divide, big buffer): growth rate (quasi-static?), relax
iters per frame, and the radial K_R term (per-vertex radial spring to a ramping R0 -- a candidate
buckling driver). Report hollow + the radius spread (std/mean = crumpling) which is the direct buckling
signal. Matched to the SAME final linear scale (~2.5x) so cell counts are comparable across rates."""
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
from diag_tools import hollow_metric
from topology_ops import rings_from_flat_3d

RADIUS, JITTER, SEED = 5.0, 0.16, 0
N_CELLS = 150


def split(pos, mesh):
    dev, area, ndeg = hollow_metric(pos, mesh); devd = np.degrees(dev)
    med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    tiny = area < 0.15 * med; inv = (devd > 50.0) & (~tiny)
    rings = rings_from_flat_3d(np.asarray(mesh["E_srce"]), np.asarray(mesh["E_trgt"]), np.asarray(mesh["E_face"]), mesh["nF"])
    rad = np.array([np.linalg.norm(pos[r].mean(0)) if (r is not None and len(r)) else 0 for r in rings])
    rad = rad[rad > 0]
    return dict(total=float((tiny | inv | (ndeg < 3)).mean()), crumple=float(rad.std() / (rad.mean() + 1e-9)))


def build(rate, relax, K_R, frames, max_scale):
    verts, es, et, ef, nF = build_sphere_mesh(N_CELLS, RADIUS, JITTER, SEED); Nv = verts.shape[0]
    ops = [{"op": "mesh_seed", "at": "vertex", "n_cells": N_CELLS, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.72, "seed": SEED, "before_frame": 1},
           {"op": "cell_grow", "at": "vertex", "rate": rate, "every": 1, "max_scale": max_scale, "rho": 1.0, "a_sw": 0.0, "vth_frac": 1e9, "conserve_amount": False},
           {"op": "cell_mechanics", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": K_R, "mu": 1.0, "dt": 1.0, "relax_iters": relax, "eta": 0.08, "cap_frac": 0.12},
           {"op": "edge_flip", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 20},
           {"op": "cell_divide", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72, "every": 2, "max_div": 10},
           {"op": "topo_record", "at": "vertex", "every": max(1, (frames + 300) // 300)}]
    sched = ["mesh_seed", "cell_grow", "cell_mechanics", "edge_flip", "cell_divide", "topo_record"]
    cfg = {"general": {"name": "regime", "seed": SEED, "n_frames": frames, "dt": 1.0, "record_cap": 300,
                       "boundary": "free", "dim": 3, "world": [6 * RADIUS] * 3},
           "sets": {"vertex": {"n": int(Nv * 20)}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


def run(label, rate, relax, K_R):
    # match final scale 2.5x: frames so (1+rate)^frames ~ 2.5  -> frames = ln2.5/ln(1+rate)
    frames = int(np.log(2.5) / np.log(1 + rate))
    sim, mesh0 = build(rate, relax, K_R, frames, 2.5)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    ser = [split(*(lambda mp: (mp[1], mp[0]))(frame(int(tt)))) for tt in np.linspace(0, T - 1, 40).astype(int)]
    print(f"{label:22s} rate={rate} relax={relax} K_R={K_R} frames={frames}  cells->{int(frame(T-1)[0]['nF'])}  "
          f"HOLLOW max={max(s['total'] for s in ser):.3f} mean={np.mean([s['total'] for s in ser]):.3f}  "
          f"CRUMPLE(rad std/mean) max={max(s['crumple'] for s in ser):.3f}", flush=True)


run("baseline",       0.003,  26, 0.4)
run("slow_growth",    0.001,  26, 0.4)
run("more_relax",     0.003,  80, 0.4)
run("no_radial",      0.003,  26, 0.0)
run("slow+relax",     0.001,  80, 0.4)

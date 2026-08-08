#!/usr/bin/env python
"""STOP guessing -- look. Run the exploding case (400f vesicle_divide) and, at the peak-hollow frame,
dump exactly WHAT hollow_flags is flagging: the breakdown (dev>50 vs tiny-area vs under-connected deg<3),
the deviation distribution, and whether flagged cells correlate with area / degree / radius / ring
validity. This decides the real mechanism instead of another guessed fix."""
from __future__ import annotations
import os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np, yaml
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa
import plexus.schema as S
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tyssue_diag import hollow_metric
from tyssue_topology_ops3d import rings_from_flat_3d

RADIUS, JITTER, SEED = 5.0, 0.16, 0
FRAMES, N_CELLS = 400, 150


def build():
    verts, es, et, ef, nF = build_sphere_mesh(N_CELLS, RADIUS, JITTER, SEED)
    Nv = verts.shape[0]; buf = int(Nv * 30.0)
    ops = [{"op": "seed_mesh_3d", "at": "vertex", "n_cells": N_CELLS, "radius": RADIUS, "jitter": JITTER,
            "p0": 3.72, "seed": SEED, "before_frame": 1},
           {"op": "grow_3d", "at": "vertex", "rate": 0.003, "every": 1, "rho": 1.0, "a_sw": 0.0, "vth_frac": 1e9, "conserve_amount": False},
           {"op": "shape_energy_3d", "at": "vertex", "p0": 3.72, "K_A": 1.0, "K_P": 1.0, "Lambda": 0.5,
            "Gamma": 0.1, "K_V": 1.0, "K_R": 0.4, "mu": 1.0, "dt": 1.0, "relax_iters": 26, "eta": 0.08, "cap_frac": 0.12},
           {"op": "reconnect_t1_3d", "at": "vertex", "l_th_frac": 0.35, "every": 2, "max_flips": 20},
           {"op": "divide_3d", "at": "vertex", "factor": 2.0, "reset_noise": 0.12, "p0": 3.72, "every": 2, "max_div": 10},
           {"op": "topo_snapshot_3d", "at": "vertex", "every": max(1, (FRAMES + 300) // 300)}]
    sched = ["seed_mesh_3d", "grow_3d", "shape_energy_3d", "reconnect_t1_3d", "divide_3d", "topo_snapshot_3d"]
    cfg = {"general": {"name": "inspect", "seed": SEED, "n_frames": FRAMES, "dt": 1.0, "record_cap": 300,
                       "boundary": "free", "dim": 3, "world": [6 * RADIUS] * 3},
           "sets": {"vertex": {"n": buf}}, "fields": {}, "operators": ops, "schedule": sched}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name
    sim = S.load(path); os.unlink(path)
    return sim, dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)


sim, mesh0 = build()
Hf, out = engine_run(sim, device="cpu")
emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist"); posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]


def frame(t):
    mt = hist[min(t, len(hist) - 1)] if hist else mesh0
    return mt, posf[t][:mt["Nv"]].astype(np.float64)


# find peak-hollow frame
peak_t, peak_frac = 0, -1
for tt in np.linspace(0, T - 1, 40).astype(int):
    mt, pt = frame(int(tt)); dev, area, ndeg = hollow_metric(pt, mt)
    devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    frac = float(((devd > 50) | (area < 0.15 * med) | (ndeg < 3)).mean())
    if frac > peak_frac:
        peak_frac, peak_t = frac, int(tt)

mt, pt = frame(peak_t); dev, area, ndeg = hollow_metric(pt, mt)
devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
tiny = area < 0.15 * med; big_dev = devd > 50; under = ndeg < 3; hollow = tiny | big_dev | under
rings = rings_from_flat_3d(np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"]), mt["nF"])
ringlen = np.array([len(r) if r is not None else 0 for r in rings])
rad = np.linalg.norm(np.array([pt[r].mean(0) if (r is not None and len(r) > 0) else [0, 0, 0] for r in rings]), axis=1)

print(f"\n=== PEAK frame t={peak_t}  cells={mt['nF']}  hollow_frac={peak_frac:.3f} ===")
print(f"breakdown:  dev>50 = {big_dev.mean():.3f}   tiny-area = {tiny.mean():.3f}   deg<3 = {under.mean():.3f}"
      f"   (overlap dev&tiny = {(big_dev & tiny).mean():.3f})")
print(f"deviation deg percentiles [50,75,90,99,max]: {np.percentile(devd,[50,75,90,99]).round(1)} {devd.max():.1f}")
print(f"ring sizes (n-gon) counts: " + ", ".join(f"{k}:{int((ringlen==k).sum())}" for k in range(2, 9)) +
      f"  (>=9: {int((ringlen>=9).sum())})")
print(f"cos(dev) sign: negative (truly inverted, >90deg) = {(devd>90).mean():.3f}   50-90deg (tilted) = {((devd>50)&(devd<=90)).mean():.3f}")
for name, sel in [("HOLLOW", hollow), ("clean", ~hollow)]:
    if sel.any():
        print(f"  {name:6s} n={int(sel.sum()):4d}  area/med={np.median(area[sel])/med:.2f}  deg={ndeg[sel].mean():.1f}"
              f"  ring={ringlen[sel].mean():.1f}  rad={rad[sel].mean():.2f}")
print(f"radius spread all cells: mean={rad[rad>0].mean():.2f} std={rad[rad>0].std():.2f} (R0={float(emesh.get('R0','nan')) if not isinstance(emesh.get('R0'),(int,float)) else emesh.get('R0')})")
# correlation: are hollow cells the SMALL (fresh) ones or not?
print(f"area percentile of hollow cells (median): {(area[hollow][:,None] > area[None,:]).mean()*100:.0f}%ile" if hollow.any() else "")

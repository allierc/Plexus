#!/usr/bin/env python
"""Step-by-step: is vesicle_grow_divide (MECHANICS ONLY, no RD) actually hollow-free, and does LENGTH
break it? rd_coral_grow == vesicle_divide + pure colouring (byte-identical vertex trajectories, per the
report), so if rd_coral_grow_long is hollow, the mechanics-only vesicle_divide must be too -- the coral
cannot be the cause. Run the exact vesicle_divide preset at increasing lengths and count hollow cells."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import plexus.operators  # noqa
import tyssue_ops3d, tyssue_t1_ops3d  # noqa  (NO tyssue_rd_ops -> no reaction-diffusion at all)
from plexus.engine import run as engine_run
from tyssue_ops3d import build_sphere_mesh
from tyssue_diag import hollow_metric
import run_tyssue_vesicle as V


def split(pos, mesh):
    dev, area, ndeg = hollow_metric(pos, mesh)
    devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    tiny = area < 0.15 * med; inv = (devd > 50.0) & (~tiny); under = ndeg < 3
    return dict(total=float((tiny | inv | under).mean()), tiny=float(tiny.mean()), inverted=float(inv.mean()))


def run(frames):
    n_cells = 150
    verts, es, et, ef, nF = build_sphere_mesh(n_cells, V.RADIUS, V.JITTER, V.SEED)
    Nv = verts.shape[0]; mesh0 = dict(E_srce=es, E_trgt=et, E_face=ef, nF=nF, Nv=Nv)
    buf = int(Nv * 30.0)   # LARGE buffer: rule out the 890-cell vertex-buffer cap as an artifact
    sim, _ = V.make_spec("vlen", 3.72, buf, 0.003, True, n_cells, frames)   # divide=True, NO RD ops
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    ser = []
    for tt in np.linspace(0, T - 1, 40).astype(int):
        mt, pt = frame(int(tt)); ser.append(split(pt, mt))
    cells_end = int(frame(T - 1)[0]["nF"])
    print(f"vesicle_divide  frames={frames:4d}  cells 150->{cells_end} (+{int(emesh.get('n_div',0))} div)  "
          f"TOTAL max={max(s['total'] for s in ser):.3f} mean={np.mean([s['total'] for s in ser]):.3f}  "
          f"INVERTED mean={np.mean([s['inverted'] for s in ser]):.3f}  "
          f"final={ser[-1]['total']:.3f}", flush=True)


for frames in (220, 400):          # native 220 (the 'ok' one) -> extended, with a LARGE vertex buffer
    run(frames)

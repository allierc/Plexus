#!/usr/bin/env python
"""Does hollowing come from DIVISION or from GROWTH? Controlled A/B on the SAME coral spec:
  growth-only  (divide=False): cells inflate but never split
  growth+div   (divide=True) : the baseline_long case
Same growth rate, length, seed. Report hollow split (tiny sliver / inverted cap / total) over the
rollout for each. If growth-only stays ~flat and only growth+division hollows, the defect is created
AT DIVISION -> the fix lives in the Divide operator (septum quality + fresh-daughter handling), not
in the mechanics/growth. Also renders the growth-only shell coloured by hollow flag for the report."""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import plexus.operators  # noqa
import mesh_ops, chem_ops, t1_ops  # noqa
from plexus.engine import run as engine_run
from diag_tools import hollow_metric
from run_tyssue_vesicle import _draw, _draw_cross
import run_tyssue_rd as R

FRAMES, N_CELLS, GROW = 400, 150, 0.003


def split(pos, mesh):
    dev, area, ndeg = hollow_metric(pos, mesh)
    devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    tiny = area < 0.15 * med; inv = (devd > 50.0) & (~tiny); under = ndeg < 3
    return dict(total=float((tiny | inv | under).mean()), tiny=float(tiny.mean()), inverted=float(inv.mean()))


def run(divide):
    mesh0, nF = R._mesh(N_CELLS); Nv = mesh0["Nv"]
    buf = int(Nv * (4.0 if divide else 1.2)); cbuf = int(nF * (4.0 if divide else 1.2))
    sim, cfg = R.make_spec("hsrc", R.GS, N_CELLS, FRAMES, GROW, divide, 0.0, buf, cbuf)
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    ser = []
    for tt in np.linspace(0, T - 1, 40).astype(int):
        mt, pt = frame(int(tt)); s = split(pt, mt); s["t"] = int(tt); s["cells"] = int(mt["nF"]); ser.append(s)
    tag = "growth+div " if divide else "growth-only"
    print(f"{tag}  cells {N_CELLS}->{ser[-1]['cells']} (+{int(emesh.get('n_div',0))} div)  "
          f"TOTAL max={max(s['total'] for s in ser):.3f} mean={np.mean([s['total'] for s in ser]):.3f}  "
          f"TINY mean={np.mean([s['tiny'] for s in ser]):.3f}  "
          f"INVERTED mean={np.mean([s['inverted'] for s in ser]):.3f}", flush=True)
    return frame, T, tag


for divide in (False, True):
    frame, T, tag = run(divide)
    if not divide:                                        # render growth-only, coloured by hollow -> should be clean
        Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 12).astype(int))
        fig = plt.figure(figsize=(17.6, 4.6)); fig.patch.set_facecolor("black")
        for i, fr in enumerate((0.0, 0.5, 1.0)):
            t = int(round(fr * (T - 1))); mt, pt = frame(t)
            dev, area, ndeg = hollow_metric(pt, mt); devd = np.degrees(dev)
            med = np.median(area[area > 0]) if (area > 0).any() else 1.0
            sc = np.clip(devd / 70.0, 0, 1); sc[(area < 0.15 * med) | (ndeg < 3)] = 1.0
            ax = fig.add_subplot(1, 3, i + 1, projection="3d"); _draw(ax, pt, mt, 3.72, azim=30, act=sc[:mt["nF"]], Lbox=Rmax * 1.06)
        fig.subplots_adjust(0.004, 0.005, 0.996, 0.996, wspace=0.02)
        fig.savefig(os.path.join(HERE, "archive", "rd_coral_grow_long", "growth_only_strip.png"),
                    dpi=120, facecolor="black"); plt.close(fig)

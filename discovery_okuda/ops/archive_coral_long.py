#!/usr/bin/env python
"""Archive the rd_coral_grow_long hollow-cell study: render the shell coloured by the HOLLOW flag
(so the defect is visible) and split the count into TINY slivers (degenerate just-divided daughters,
edge-midpoint septum) vs INVERTED caps (normal deviation > 50 deg) over the rollout. Writes
archive/rd_coral_grow_long/{strip.png, movie.mp4, diag.json} + the 4-condition A/B table."""
from __future__ import annotations
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src")); sys.path.insert(0, HERE)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import plexus.operators  # noqa
import mesh_ops, chem_ops, t1_ops  # noqa
from plexus.engine import run as engine_run
from diag_tools import hollow_metric
from run_tyssue_vesicle import _draw, _draw_cross
from ab_coral_long import build

OUT = os.path.join(HERE, "archive", "rd_coral_grow_long"); os.makedirs(OUT, exist_ok=True)

# the 4-condition A/B already run (hollow fraction = tiny slivers OR inverted caps OR under-connected)
AB = [{"cond": "baseline_long", "cv": 0.0, "max_div_frac": 0.0, "hollow_max": 0.993, "hollow_mean": 0.457, "final": 0.000},
      {"cond": "+cv",           "cv": 0.4, "max_div_frac": 0.0,  "hollow_max": 0.966, "hollow_mean": 0.415, "final": 0.044},
      {"cond": "+livecap",      "cv": 0.0, "max_div_frac": 0.05, "hollow_max": 0.996, "hollow_mean": 0.333, "final": 0.000},
      {"cond": "+both",         "cv": 0.4, "max_div_frac": 0.05, "hollow_max": 0.987, "hollow_mean": 0.274, "final": 0.057}]


def split(pos, mesh):
    """total / tiny-sliver / inverted-cap hollow fractions + a [0,1] colour score (red = hollow)."""
    dev, area, ndeg = hollow_metric(pos, mesh)
    devd = np.degrees(dev); med = np.median(area[area > 0]) if (area > 0).any() else 1.0
    tiny = area < 0.15 * med; inv = (devd > 50.0) & (~tiny); under = ndeg < 3
    hollow = tiny | inv | under
    score = np.clip(devd / 70.0, 0.0, 1.0); score[tiny | under] = 1.0
    nF = mesh["nF"]
    return (dict(total=float(hollow.mean()), tiny=float(tiny.mean()), inverted=float(inv.mean())),
            score[:nF])


def main():
    sim, mesh0 = build(cv=0.0, max_div_frac=0.0)          # baseline_long = the problem
    Hf, out = engine_run(sim, device="cpu")
    emesh = Hf.level("vertex")._mesh; hist = emesh.get("hist")
    posf = out["sets"]["vertex"]["pos"]; T = posf.shape[0]

    def frame(t):
        mt = hist[min(t, len(hist) - 1)] if hist else mesh0
        return mt, posf[t][:mt["Nv"]].astype(np.float64)

    series = []
    for tt in np.linspace(0, T - 1, 40).astype(int):
        mt, pt = frame(int(tt)); s, _ = split(pt, mt); s["t"] = int(tt); s["cells"] = int(mt["nF"]); series.append(s)
    peak = max(series, key=lambda s: s["total"])
    Rmax = max(float(np.linalg.norm(frame(t)[1], axis=1).max()) for t in np.linspace(0, T - 1, 20).astype(int))
    L3, L2 = Rmax * 1.06, Rmax * 2.23

    # strip: 4 timepoints, shell + cross-section, coloured RED by hollow flag
    fig = plt.figure(figsize=(17.6, 9.0)); fig.patch.set_facecolor("black")
    picks = [int(round(fr * (T - 1))) for fr in (0.0, 0.33, 0.66, 1.0)]
    for i, t in enumerate(picks):
        mt, pt = frame(t); _, sc = split(pt, mt)
        ax3 = fig.add_subplot(2, 4, i + 1, projection="3d"); _draw(ax3, pt, mt, 3.72, azim=30, act=sc, Lbox=L3)
        ax2 = fig.add_subplot(2, 4, 4 + i + 1); _draw_cross(ax2, pt, mt, 3.72, act=sc, Lbox=L2)
    fig.subplots_adjust(0.006, 0.005, 0.996, 0.996, wspace=0.02, hspace=0.02)
    fig.savefig(os.path.join(OUT, "strip.png"), dpi=120, facecolor="black"); plt.close(fig)

    figm = plt.figure(figsize=(5.0, 5.2)); figm.patch.set_facecolor("black")
    axm = figm.add_subplot(111, projection="3d"); figm.subplots_adjust(0, 0, 1, 1)
    keep = np.linspace(0, T - 1, min(T, 60)).astype(int)
    wri = FFMpegWriter(fps=max(1, round(len(keep) / 8.0)), metadata={"title": "rd_coral_grow_long"})
    with wri.saving(figm, os.path.join(OUT, "movie.mp4"), dpi=110):
        for j, t in enumerate(keep):
            mt, pt = frame(t); _, sc = split(pt, mt)
            _draw(axm, pt, mt, 3.72, azim=(2 * j) % 360, act=sc, Lbox=L3); wri.grab_frame()
    plt.close(figm)

    diag = {"name": "rd_coral_grow_long", "frames": T, "cells_end": int(frame(T - 1)[0]["nF"]),
            "n_div": int(emesh.get("n_div", 0)), "n_t1": int(emesh.get("n_t1", 0)),
            "hollow_peak": {k: round(peak[k], 3) for k in ("total", "tiny", "inverted", "t", "cells")},
            "hollow_final": {k: round(series[-1][k], 3) for k in ("total", "tiny", "inverted")},
            "hollow_mean_total": round(float(np.mean([s["total"] for s in series])), 3),
            "hollow_mean_tiny": round(float(np.mean([s["tiny"] for s in series])), 3),
            "hollow_mean_inverted": round(float(np.mean([s["inverted"] for s in series])), 3),
            "ab_conditions": AB}
    json.dump(diag, open(os.path.join(OUT, "diag.json"), "w"), indent=1)
    print(f"[archive] peak total={peak['total']:.3f} (tiny={peak['tiny']:.3f} inv={peak['inverted']:.3f}) "
          f"@t={peak['t']} cells={peak['cells']};  mean tiny={diag['hollow_mean_tiny']} "
          f"inv={diag['hollow_mean_inverted']}", flush=True)


if __name__ == "__main__":
    main()

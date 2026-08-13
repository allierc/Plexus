#!/usr/bin/env python
"""figure_geometries -- the seven test fixtures, rendered, with what the metrics say about each.

CEDRIC, 5 AUGUST: *"add figure with test geometry and metrics outputs"*, *"in pdf note"*.

WHY THE FIGURE AND NOT JUST THE TABLE. `test_geometries.py` asserts bounds, and a bound cannot catch a
metric that is wrong in a way nobody thought to bound. Reading the shapes beside their numbers can:
`n_tubes` = 3 on a rugby ball was obvious the moment the ellipsoid sat next to the column, and no
assertion I would have written beforehand was looking for it.

IT CALLS THE REFERENCE RENDERER. `run_tyssue_vesicle._draw` is what draws every movie and strip this
campaign has produced -- cells as prisms, apical faces coloured by activator, magenta for a genuinely
broken ring. Drawing these fixtures any other way would put a second renderer in the project and make
the figure a picture of something the campaign never looks at.

RUN: python figure_geometries.py   ->  figs/test_geometries.pdf (+ .png)
"""
from __future__ import annotations

import io
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TYSSUE = os.path.join(HERE, "ops")
for _p in (HERE, os.path.join(HERE, "agents"), TYSSUE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

import test_geometries as G                                      # noqa: E402

OUT = os.path.join(HERE, "figs")

# The columns worth reading beside a shape. `_peak` and the rest are temporal and meaningless on a
# single frame, so these are the per-frame quantities.
COLS = [("protr", "protr"), ("protr_p99", "p99"), ("r_cv", "r_cv"),
        ("gyr_prolate", "prolate"), ("gyr_oblate", "oblate"), ("reduced_volume", "red.vol"),
        ("n_tubes", "tubes"), ("n_tips", "tips"), ("protrusion_aspect_max", "aspect"),
        ("ray_single_frac", "ray1"), ("shape_idx_p95", "shape95"), ("genus", "genus")]

ORDER = ["sphere", "prolate", "oblate", "undulated", "tubed", "branched", "multi_tube",
         "self_intersecting"]
LABEL = {"sphere": "sphere", "prolate": "prolate (2:1)", "oblate": "oblate (2.5:1)",
         "undulated": "undulated", "tubed": "tube", "branched": "branched",
         "multi_tube": "5 tubes", "self_intersecting": "self-intersecting"}


def render():
    from run_tyssue_vesicle import _draw
    fig = plt.figure(figsize=(17.0, 5.4), facecolor="black")
    gs = fig.add_gridspec(2, 8, height_ratios=[1.15, 1.0], hspace=-0.06, wspace=0.0,
                          left=0.004, right=0.996, top=0.985, bottom=0.02)

    rows = {}
    for i, name in enumerate(ORDER):
        v, mt = G.GEOMETRIES[name]()
        with contextlib.redirect_stdout(io.StringIO()):
            m = G.measure(name)
        rows[name] = m

        ax = fig.add_subplot(gs[0, i], projection="3d", facecolor="black")
        # THE BOX IS THE SAME FOR EVERY PANEL. A per-panel limit would rescale each shape to fill its
        # frame, so a tube and a sphere would look equally elongated -- the figure would hide exactly
        # the difference it exists to show.
        with contextlib.redirect_stdout(io.StringIO()):
            _draw(ax, np.asarray(v, float), mt, p0=3.5, azim=32, act=None, Lbox=2.35)
        ax.text2D(0.02, 0.94, f"{chr(97 + i)}", transform=ax.transAxes, color="white",
                  fontsize=13, fontweight="bold", va="top")
        ax.text2D(0.02, 0.84, LABEL[name], transform=ax.transAxes, color="white",
                  fontsize=10, va="top")

    # ---- the numbers, as one table spanning the width
    axt = fig.add_subplot(gs[1, :], facecolor="black")
    axt.axis("off")
    axt.text(0.004, 0.97, "i", color="white", fontsize=13, fontweight="bold",   # not "h": the last panel is h
             va="top", transform=axt.transAxes)

    x0, dx = 0.155, (1.0 - 0.165) / len(COLS)
    y0, dy = 0.84, 0.108
    for j, (_k, hdr) in enumerate(COLS):
        axt.text(x0 + j * dx, y0 + 0.085, hdr, color="white", fontsize=9.5, ha="center",
                 transform=axt.transAxes, fontweight="bold")
    for i, name in enumerate(ORDER):
        y = y0 - i * dy
        axt.text(0.004, y, f"{chr(97 + i)}", color="white", fontsize=9.5, fontweight="bold",
                 va="center", transform=axt.transAxes)
        axt.text(0.022, y, LABEL[name], color="white", fontsize=9.5, va="center",
                 transform=axt.transAxes)
        for j, (k, _h) in enumerate(COLS):
            val = rows[name].get(k)
            if val is None:
                txt, col = "--", "0.45"
            else:
                fv = float(val)
                txt = f"{int(fv)}" if k in ("n_tubes", "n_tips", "genus") else f"{fv:.3f}"
                # THE ONE THING THE FIGURE ARGUES: which numbers answer the campaign's question.
                # gold where a metric fires on the shape it is supposed to fire on.
                fires = ((k == "n_tubes" and fv > 0) or (k == "n_tips" and fv > 0)
                         or (k == "protrusion_aspect_max" and fv > 1.0)
                         or (k == "ray_single_frac" and fv < 1.0)
                         or (k == "gyr_prolate" and fv > 1.5) or (k == "gyr_oblate" and fv > 0.15)
                         or (k == "protr" and fv > 1.3))
                col = "#f2c14e" if fires else "white"
            axt.text(x0 + j * dx, y, txt, color=col, fontsize=9.5, ha="center",
                     va="center", transform=axt.transAxes)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(OUT, f"test_geometries.{ext}")
        fig.savefig(p, facecolor="black", dpi=170)
        print(f"  wrote {p}")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    render()

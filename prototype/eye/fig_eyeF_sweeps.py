"""fig_eyeF_sweeps -- what was swept on eye F's lateral rectus, and what it bought.

    python fig_eyeF_sweeps.py
    -> connectome-gnn-cx/figures/zebrafish/fig_eyeF_material_sweeps.png

Panel (a) is the plant itself with LR -- the muscle every sweep drove -- picked out
against the other five, so the reader can see which strap the numbers refer to.
Panels (b-d) are the three sweeps, each against the 25 deg the tracking task needs.

The figure exists to make one point quickly: three separate families of parameter
were swept, none of them moved the span, and the reason is in panel (e) -- the muscle
shortens by a third of its length and delivers a sixth of that to the globe.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

import fish_anatomy as FA
import view3d_fish as V3

ARCHIVE = os.path.join(HERE, "archive")
OUT = "/workspace/connectome-gnn-cx/figures/zebrafish/fig_eyeF_material_sweeps.png"
BG, FG = "black", "white"
TARGET = 25.0                      # deg horizontal, from the tracking task
LR = "#4da3ff"


def plant_panel(path="/tmp/_fig_eyeF_plant.png"):
    """Eye F down the optic axis, LR at full brightness and the other five dimmed."""
    import pyvista as pv
    C = V3.plant_config("larva")
    B = V3.build("larva", n_pts=14000)
    # the globe is drawn nearly clear HERE, unlike the movie template: this panel exists
    # to show which strap the sweeps drove, and an opaque globe hides it
    V3.SCLERA_ALPHA, V3.CORNEA_ALPHA, V3.LENS_ALPHA = 0.13, 0.20, 0.16
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(1100, 1000), border=False)
    p.set_background(BG)
    V3._add_globe(p, B)
    for k, cloud in B["muscles"].items():
        hot = (k == "LR")
        p.add_mesh(cloud, color=FA.COLOR[k] if hot else "#5a5a5a",
                   point_size=5.0 if hot else 3.0, render_points_as_spheres=True,
                   opacity=1.0 if hot else 0.42, show_scalar_bar=False)
    ins = B["ins"] / np.linalg.norm(B["ins"], axis=1, keepdims=True)
    for j, k in enumerate(FA.MUSCLE_KEYS):
        p.add_point_labels(np.array([ins[j] * 1.30]), [k],
                           text_color=FA.COLOR[k] if k == "LR" else "#888888",
                           font_size=26 if k == "LR" else 16, bold=(k == "LR"),
                           shape=None, show_points=False, always_visible=True)
    # three-quarter from below and in front: LR runs caudo-ventrally, so this is the
    # direction that shows its whole length rather than its end
    p.camera_position = [(-2.6, -2.2, 3.4), (0, -0.1, -0.1), (0, 1, 0)]
    p.camera.parallel_projection = True
    p.camera.parallel_scale = 1.62
    p.render()
    img = p.screenshot(None, return_img=True)
    p.close()
    import imageio.v2 as iio
    iio.imwrite(path, img)
    return path


def _rows(pattern, key):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if key in r:
            out.append(r)
    return out


def main():
    gap = sorted(_rows(os.path.join(ARCHIVE, "gap_sweep", "*.json"), "gap"),
                 key=lambda r: r["gap"])
    susp = _rows(os.path.join(ARCHIVE, "suspension", "*.json"), "tonic")
    ae = json.load(open(os.path.join(ARCHIVE, "eye_A", "sweep_LR_span.json"))) \
        if os.path.exists(os.path.join(ARCHIVE, "eye_A", "sweep_LR_span.json")) else {"cells": []}

    fig = plt.figure(figsize=(15.2, 7.4), dpi=170, facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], height_ratios=[1, 1],
                          wspace=0.28, hspace=0.42, left=0.045, right=0.985,
                          top=0.90, bottom=0.10)

    ax = fig.add_subplot(gs[:, 0], facecolor=BG)
    ax.imshow(imread(plant_panel()))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("a   eye F, three-quarter view\nevery sweep drove LR (blue)", color=FG,
                 fontsize=12, loc="left")

    def style(ax, xlabel):
        ax.set_facecolor(BG)
        ax.axhline(TARGET, color="#ff5c5c", lw=1.4, ls="--")
        ax.text(0.98, TARGET, " task needs 25", color="#ff5c5c", fontsize=9,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel(xlabel, color=FG, fontsize=11)
        ax.set_ylabel("abduction, deg", color=FG, fontsize=11)
        ax.tick_params(colors=FG, labelsize=9)
        for k, s in ax.spines.items():
            s.set_color("#555555" if k in ("left", "bottom") else "none")
        ax.set_ylim(0, 27)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot([r["gap"] for r in gap], [r["pose_deg"][0] for r in gap], "o-", color=LR, lw=1.8)
    ax.axvline(0.0161, color="#7ee081", lw=1.2, ls=":")
    ax.text(0.0161, 24, " F ships here", color="#7ee081", fontsize=9, rotation=90, va="top")
    style(ax, "sclera stand-off  gap")
    ax.set_title("b   geometry: the strap's clearance", color=FG, fontsize=11, loc="left")

    ax = fig.add_subplot(gs[0, 2])
    cells = [c for c in ae.get("cells", []) if "abduction_deg" in c
             and abs(c.get("radius_drift_pct", 0)) < 2.0]
    for E in sorted({c["youngs"] for c in cells}):
        sub = sorted([c for c in cells if c["youngs"] == E], key=lambda c: c["A_over_E"])
        ax.plot([c["A_over_E"] for c in sub], [c["abduction_deg"] for c in sub],
                "o-", lw=1.6, label=f"E = {E:g}")
    ax.legend(fontsize=8, frameon=False, labelcolor=FG, loc="upper left")
    style(ax, "active stress / stiffness   A/E")
    ax.set_title("c   material: A and E together\n(cells that crushed the globe removed)",
                 color=FG, fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 1])
    tags = ["base", "tonic07", "tonic02", "fat1000", "fat250", "socket1500", "socket500",
            "all_loose"]
    lab = ["baseline", "tonic\n0.07", "tonic\n0.02", "fat\n1000", "fat\n250", "socket\n1500",
           "socket\n500", "all\nloose"]
    vals = {r["tag"]: r["pose_deg"][0] for r in susp}
    y = [vals.get(t, np.nan) for t in tags]
    ax.bar(range(len(tags)), y, color=[LR if t != "base" else "#888888" for t in tags])
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(lab, fontsize=7.5, color=FG)
    style(ax, "")
    ax.set_title("d   suspension and the five antagonists", color=FG, fontsize=11, loc="left")

    ax = fig.add_subplot(gs[1, 2], facecolor=BG)
    ax.axis("off")
    ax.set_title("e   where the contraction goes", color=FG, fontsize=11, loc="left")
    txt = ("LR shortens        31 % of its rest length\n"
           "moment arm         107 um  (larger than the\n"
           "                   mammalian plant's 69)\n\n"
           "if all of that shortening pulled the\n"
           "insertion round the arm:      34.6 deg\n"
           "measured:                      6.1 deg\n\n"
           "so 82 % of the contraction is absorbed\n"
           "INSIDE the muscle -- the slender strap\n"
           "buckles instead of transmitting.")
    ax.text(0.0, 0.92, txt, color=FG, fontsize=9.6, va="top", family="monospace",
            transform=ax.transAxes, linespacing=1.45)

    fig.suptitle("Eye F: three families of parameter swept on the lateral rectus, "
                 "and why none of them bought span", color=FG, fontsize=13)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, facecolor=BG, dpi=170)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""fig_plant_comparison -- one figure: the guessed plant, the measured plant, and what
changes when you stop guessing.

    python fig_plant_comparison.py   ->  archive/eye_F/fig_plant_comparison.png

The point of the figure is the last panel. Nowhere in this prototype is a muscle's
action written down: each one's rotation axis is  r x u  -- insertion radius crossed
with line of action -- computed from where the tissue is. So swapping the anatomy
swaps the actions, and the panel shows by how much. On the guessed (mammalian) plant
the textbook actions come out, as they should. On the measured fish plant two of them
INVERT: the superior oblique elevates and the inferior oblique depresses, because both
zebrafish obliques pull from the ROSTRAL orbit onto the dorsal and ventral faces, with
no trochlea to reverse the superior one. That is a prediction of the anatomy, not a
choice, and it is the sort of thing the guess could never have produced.
"""
from __future__ import annotations

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

OUT = os.path.join(HERE, "archive", "eye_F")
FG = "white"
BG = "black"


def torque_axes(plant):
    """[6,3] unit rotation axis per muscle, as the running model measures it.

    axis = r_hat x u_hat: the insertion's radius vector crossed with the direction it
    is pulled. Components read as  +y abduction, -x elevation, +z intorsion.
    """
    C = V3.plant_config(plant)
    ratio = C["ratio"]
    out = []
    for n, o in zip(C["ins"], C["org"]):
        n = np.asarray(n, float) / np.linalg.norm(n)
        r = 1.0 / np.sqrt(n[0] ** 2 + n[1] ** 2 + (n[2] / ratio) ** 2)
        P = r * n                                        # insertion point on the globe
        u = np.asarray(o, float) - P
        u /= np.linalg.norm(u)
        a = np.cross(P, u)
        out.append(a / np.linalg.norm(a))
    return np.asarray(out)


def _panel_image(ax, path, crop=None, label=None):
    im = imread(path)
    if crop:
        h, w = im.shape[:2]
        y0, y1, x0, x1 = [int(v * s) for v, s in zip(crop, (h, h, w, w))]
        im = im[y0:y1, x0:x1]
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if label:
        ax.text(0.012, 0.975, label, transform=ax.transAxes, color=FG, fontsize=11,
                va="top", ha="left")


def main():
    for plant in ("mammal", "larva"):
        p = os.path.join(OUT, f"fig_plant3d_{plant}.png")
        if not os.path.exists(p):
            V3.render(plant)

    M = FA.measurements()
    fig = plt.figure(figsize=(15.5, 12.6), dpi=170, facecolor=BG)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.22, 1.0, 0.86],
                          hspace=0.10, wspace=0.045,
                          left=0.006, right=0.994, top=0.985, bottom=0.045)

    # (a) the source, digitized
    ax = fig.add_subplot(gs[0, :2])
    _panel_image(ax, os.path.join(OUT, "fig_source_vs_model.png"))
    ax.text(0.008, 0.985, "a   the drawing, digitized, and the plant built from it",
            transform=ax.transAxes, color=FG, fontsize=11, va="top", ha="left")

    # (b) the numbers that came out of it
    ax = fig.add_subplot(gs[0, 2], facecolor=BG)
    ax.axis("off")
    G, L = M["globe"], M["lens"]
    rows = [("globe, axial : equatorial", f"{G['axial_over_equatorial']:.3f}", "0.82 guessed"),
            ("globe, um", f"{G['axial_diameter_um']:.0f} x {G['equatorial_diameter_um']:.0f}", ""),
            ("outline, ellipse residual", f"{G['outline_fit_rms_px']:.1f} px", "of a 279 px axis"),
            ("fore-aft asymmetry", f"{G['fore_aft_asymmetry']:+.3f}", "i.e. none"),
            ("lens radius / a_eq", f"{L['radius_over_equatorial_semi_axis']:.3f}", ""),
            ("lens reach / a_axial", f"{L['lateral_pole_reach']:.3f}", "1.0 = the cornea"),
            ("", "", "")]
    for k in FA.MUSCLE_KEYS:
        m = M["muscles"][k]
        rows.append((f"{k}  {FA.LONG_NAME[k]}", f"{m['length_um']:.0f} x {m['mean_width_um']:.1f} um",
                     f"CN {FA.CRANIAL_NERVE[k]}"))
    y = 0.955
    ax.text(0.0, y, "b   what the drawing measures", color=FG, fontsize=11, va="top")
    y -= 0.075
    for name, val, note in rows:
        if name:
            col = FA.COLOR.get(name[:2], FG) if name[:2] in FA.COLOR else FG
            ax.text(0.0, y, name, color=col, fontsize=8.6, va="top", family="monospace")
            ax.text(0.62, y, val, color=col, fontsize=8.6, va="top", family="monospace")
            ax.text(0.985, y, note, color="#888888", fontsize=7.6, va="top", ha="right",
                    family="monospace")
        y -= 0.062

    # (c, d) the two plants, same camera
    for j, (plant, lab) in enumerate([("mammal", "c   the guess: mammalian plant, globe 0.82"),
                                      ("larva", "d   measured: zebrafish larva, globe 0.676")]):
        ax = fig.add_subplot(gs[1, j], facecolor=BG)
        _panel_image(ax, os.path.join(OUT, f"fig_plant3d_{plant}.png"),
                     crop=(0.02, 0.46, 0.50, 1.0), label=lab)
    ax = fig.add_subplot(gs[1, 2], facecolor=BG)
    _panel_image(ax, os.path.join(OUT, "fig_plant3d_larva.png"), crop=(0.02, 0.46, 0.0, 0.50),
                 label="e   measured, ventral -- compare with a")

    # (f) the actions, which nothing in the model tabulates
    ax = fig.add_subplot(gs[2, :], facecolor=BG)
    A_m, A_f = torque_axes("mammal"), torque_axes("larva")
    comp = ["abduction", "elevation", "intorsion"]
    x = np.arange(FA.N_MUSCLE)
    w = 0.13
    for c in range(3):
        vm = [A_m[i][1] if c == 0 else (-A_m[i][0] if c == 1 else A_m[i][2]) for i in range(6)]
        vf = [A_f[i][1] if c == 0 else (-A_f[i][0] if c == 1 else A_f[i][2]) for i in range(6)]
        ax.bar(x + (c - 1) * 2 * w - w / 2, vm, w, color="#777777",
               label="guessed" if c == 0 else None)
        ax.bar(x + (c - 1) * 2 * w + w / 2, vf, w,
               color=[FA.COLOR[k] for k in FA.MUSCLE_KEYS],
               label="measured" if c == 0 else None)
        for i in range(6):
            ax.text(x[i] + (c - 1) * 2 * w, -1.42, comp[c][:6], color="#999999",
                    fontsize=6.6, ha="center")
    ax.axhline(0, color="#555555", lw=0.8)
    for i, k in enumerate(FA.MUSCLE_KEYS):
        ax.text(x[i], -1.24, k, color=FA.COLOR[k], fontsize=13, ha="center",
                fontweight="bold")
    # the two that invert
    for i, k in enumerate(FA.MUSCLE_KEYS):
        if k in ("SO", "IO"):
            ax.annotate("", xy=(x[i] + w / 2, A_f[i][0] * -1), xytext=(x[i] - w / 2, A_m[i][0] * -1),
                        arrowprops=dict(arrowstyle="->", color="#ff5c5c", lw=1.4))
            ax.text(x[i], -1.80, "elevation INVERTS", color="#ff5c5c", fontsize=8.5,
                    ha="center", fontweight="bold")
    ax.set_ylim(-1.95, 1.10)
    ax.set_xlim(-0.55, 5.55)
    ax.set_xticks([])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.text(0.004, 1.035, "f   rotation axis of each muscle, r x u, measured from the geometry "
                          "-- grey: guessed plant, colour: measured plant",
            transform=ax.transAxes, color=FG, fontsize=11, va="bottom")
    lg = ax.legend(loc="upper right", fontsize=8.5, frameon=False, ncol=2)
    for t in lg.get_texts():
        t.set_color(FG)

    out = os.path.join(OUT, "fig_plant_comparison.png")
    fig.savefig(out, facecolor=BG, dpi=170)
    print("wrote", out)


if __name__ == "__main__":
    main()

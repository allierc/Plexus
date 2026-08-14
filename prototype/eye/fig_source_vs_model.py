"""fig_source_vs_model -- the drawing and the model of it, side by side, same scale.

    python fig_source_vs_model.py  ->  archive/eye_F/fig_source_vs_model.png

Left: Fig. 12.1A of Tulenko & Currie (2020), redrawn from Easter & Nicola's camera
lucida of a 96 hpf larva, with everything measured off it drawn back on -- the globe
outline and its best ellipse, the lens, and the six traced bands with their
attachments.

Right: the plant the spec builds from those measurements, rendered from the same
direction and at the SAME SCALE, so the two globes are the same size on the page and
any muscle can be followed from one panel to the other.

The right panel is mirrored, because the drawing is of the RIGHT eye and the model --
which uses (caudal, dorsal, lateral) as a right-handed frame -- is the left one. A
mirror is exactly the map between an animal's two eyes; it changes no length and no
angle, only the sign convention for torsion.
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

import digitize_fig121 as D
import fish_anatomy as FA
import view3d_fish as V3

OUT = os.path.join(HERE, "archive", "eye_F")
BG = "black"
FG = "white"

# the frame, in equatorial semi-axes above and below the globe centre. Both panels
# use it: the drawing is cropped to it, the camera is set to it.
HALF_HEIGHT = 1.30
ASPECT = 0.77                                   # width / height of each panel


def source_panel():
    """Fig. 12.1A with the fit drawn on, cropped to +-HALF_HEIGHT about the globe centre."""
    A = D.extract_panel()
    M = D.masks(A)
    (cx, cy), coef, r_of, rms, _ = D.outline_profile(M["globe"], M["outside"])
    R = {"medial": float(r_of(0.0)[0]), "caudal": float(r_of(np.pi / 2)[0]),
         "lateral": float(r_of(np.pi)[0]), "rostral": float(r_of(-np.pi / 2)[0])}
    a_ax = 0.5 * (R["lateral"] + R["medial"])
    a_eq = 0.5 * (R["rostral"] + R["caudal"])
    bands = D.band_masks(A)
    tr = D.trace_bands(A, bands, (cx, cy), a_ax, a_eq)
    ly, lx = np.nonzero(M["lens"])
    lens_c, lens_r = (lx.mean(), ly.mean()), float(np.sqrt(M["lens"].sum() / np.pi))

    half_y = HALF_HEIGHT * a_eq
    half_x = half_y * ASPECT
    fig, ax = plt.subplots(figsize=(6.4, 6.4 / ASPECT), dpi=200)
    ax.imshow(A.astype(np.uint8))
    p = np.linspace(-np.pi, np.pi, 721)
    rr = r_of(p)
    ax.plot(cx + rr * np.cos(p), cy + rr * np.sin(p), "-", color="#00e5ff", lw=2.2)
    ax.plot(cx + a_ax * np.cos(p), cy + a_eq * np.sin(p), "--", color="#ffd24d", lw=1.3)
    ax.add_patch(plt.Circle(lens_c, lens_r, fill=False, ec="#7ee081", lw=2.0))
    ax.plot(cx, cy, "+", color="#00e5ff", ms=15, mew=2.4)
    ax.annotate("", xy=(cx - a_ax * 1.22, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=1.6))
    for key, T in tr.items():
        col = FA.COLOR[key]
        ax.contour(bands[key].astype(float), levels=[0.5], colors=[col], linewidths=1.7)
        ax.plot(*T["insertion_px"], "o", mfc="none", mec="white", ms=11, mew=2.0)
        ax.plot(*T["origin_px"], "s", mfc=col, mec="white", ms=7, mew=0.8)
        ax.annotate(key, T["insertion_px"], textcoords="offset points", xytext=(-7, -12),
                    color=col, fontsize=13, fontweight="bold", ha="right")
    # keep the window inside the panel: the globe sits left of centre in the drawing,
    # so a window centred on it would otherwise hang off the paper edge.
    x0 = min(max(cx - half_x, 0.0), A.shape[1] - 2 * half_x)
    y0 = min(max(cy - half_y, 0.0), A.shape[0] - 2 * half_y)
    ax.set_xlim(x0, x0 + 2 * half_x)
    ax.set_ylim(y0 + 2 * half_y, y0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    out = "/tmp/_source_cropped.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main():
    src = source_panel()
    h = 1250
    # labels off: the panel is mirrored, and mirrored text reads backwards. They go
    # back on in matplotlib below, at the same projected positions.
    mdl = V3.render_single("larva", view=0, out="/tmp/_model_ventral.png",
                           size=(int(h * ASPECT), h), parallel_scale=HALF_HEIGHT,
                           mirror=True, labels=False)

    a, b = imread(src), imread(mdl)
    fig = plt.figure(figsize=(13.2, 8.9), dpi=175, facecolor=BG)
    gs = fig.add_gridspec(1, 2, wspace=0.012, left=0.004, right=0.996,
                          top=0.955, bottom=0.045)
    for j, (im, lab) in enumerate([
            (a, "a   Fig. 12.1A (Tulenko & Currie 2020, after Easter & Nicola 1996), digitized"),
            (b, "b   the plant the spec builds from it -- same direction, same scale")]):
        ax = fig.add_subplot(gs[0, j], facecolor=BG)
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.014, 0.982, lab, transform=ax.transAxes, color=FG if j else "black",
                fontsize=10.5, va="top", ha="left",
                bbox=dict(fc=BG if j else "white", ec="none", alpha=0.65, pad=1.6))
        if j == 1:
            # Project each insertion the way the camera did: parallel projection, view
            # direction ventral, up = rostral, then the left-right mirror.
            ins = FA.insertion_dirs("larva")
            ratio = FA.axial_ratio()
            for i, k in enumerate(FA.MUSCLE_KEYS):
                n = ins[i]
                r = 1.0 / np.sqrt(n[0] ** 2 + n[1] ** 2 + (n[2] / ratio) ** 2)
                p3 = 1.28 * r * n
                fx = 0.5 + p3[2] / (2 * HALF_HEIGHT * ASPECT)      # image right = lateral
                fy = 0.5 + p3[0] / (2 * HALF_HEIGHT)               # image up = rostral (-x)
                ax.text(1.0 - fx, 1.0 - fy, k, transform=ax.transAxes, color=FA.COLOR[k],
                        fontsize=13, fontweight="bold", ha="center", va="center")
    fig.text(0.5, 0.017,
             "ventral view, rostral up  |  cyan: measured globe outline, yellow: best ellipse "
             "(axial:equatorial = %.3f), green: lens  |  o insertion   [] origin"
             % FA.axial_ratio(), color="#aaaaaa", fontsize=9, ha="center")
    out = os.path.join(OUT, "fig_source_vs_model.png")
    fig.savefig(out, facecolor=BG, dpi=175)
    print("wrote", out)


if __name__ == "__main__":
    main()

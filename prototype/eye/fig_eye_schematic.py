"""Two-panel schematic: the eye and its muscles, and the variables.

Left: the globe seen from the front (down the corneal axis) with the six
extraocular muscles at their true insertions, taken from
``eye_anatomy.MUSCLES`` — the same table the MPM uses to shape the straps, so
this is the model's own geometry rather than a textbook redraw. The two the
circuit drives, LR and MR, are solid; the other four are drawn faint because
they exist in the plant but are not innervated by this circuit.

Right: every symbol in the model equations, in the order the signal passes
through them, so the note's algebra can be read against a picture.

Usage::

    python fig_eye_schematic.py [--out fig_eye_schematic.png] [--bg white]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eye_anatomy as EA                                  # noqa: E402

# All six are commanded now. The readout emits one non-negative drive per muscle
# and the static map is a function of all six, so there is no faint half of the
# picture any more -- which is the whole difference between this model and the
# push-pull one it replaced.
DRIVEN = ("LR", "SR", "MR", "IR", "SO", "IO")
# the renderer's own strap colours, so the labels match the picture
MUS_COLOR = {m["key"]: m["color"] for m in EA.MUSCLES}


def insertion(m):
    """Unit-sphere insertion point. theta from +z (anterior, the cornea),
    phi from +x toward +y, both in degrees — eye_anatomy's convention."""
    th, ph = np.radians(m["theta"]), np.radians(m["phi"])
    return np.array([np.sin(th) * np.cos(ph),
                     np.sin(th) * np.sin(ph),
                     np.cos(th)])


def panel_eye(ax, bg):
    """Eye G itself, drawn by the surface renderer, not a schematic of it.

    This panel used to be a circle with six straps placed from the analytic
    ``eye_anatomy`` table. Eye G's geometry is scanned rather than generated and
    its muscles do not sit where that table puts them --- the obliques run from the
    rostral orbit with no trochlea, which is why SO elevates here and IO depresses,
    the reverse of the mammalian arrangement a schematic would imply. So the panel
    now shows the render, and ``fig_eyeG_anterior.py`` supplies both the image and
    the projected label positions so they cannot drift from the geometry.
    """
    import json
    import matplotlib.image as mpimg
    fg = "white" if bg == "black" else "#111111"
    png = os.path.join(HERE, "fig_eyeG_anterior.png")
    meta = png.replace(".png", ".json")
    ax.axis("off")
    if not (os.path.isfile(png) and os.path.isfile(meta)):
        ax.text(0.5, 0.5, "run fig_eyeG_anterior.py first", ha="center",
                va="center", color=fg, fontsize=11)
        return
    img = mpimg.imread(png)
    ax.imshow(img)
    d = json.load(open(meta))
    # A legend rather than labels on the straps. Six labels placed at their own
    # projected positions collide in the rostral corner, where four of the six
    # muscles converge, and no amount of nudging separates them without lying
    # about where they are. The colours are the renderer's own, so the mapping
    # from swatch to strap is exact.
    from matplotlib.lines import Line2D
    order = ["LR", "MR", "SR", "IR", "SO", "IO"]
    handles = [Line2D([], [], color=MUS_COLOR[m], lw=7, solid_capstyle="round",
                      label=f"{m}   {d['action'][m]}")
               for m in order if m in d["action"]]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(-0.01, 1.005), frameon=False, fontsize=11,
                    labelspacing=0.75, handlelength=1.5, handletextpad=0.7,
                    borderpad=0.2)
    # the legend sits on the render, which is black whatever the figure ground is
    for t in leg.get_texts():
        t.set_color("white")
    ax.text(0.5, -0.02, "eye G at rest, near-anterior view --- scanned geometry, "
            "not a schematic", transform=ax.transAxes, ha="center", va="top",
            color=fg, fontsize=9)


def panel_vars(ax, bg):
    """Every symbol of section 4, in the order the signal meets them.

    Stacked top to bottom because the signal path is a straight line. Six stages
    now rather than four: the static map and the mechanics are separate blocks,
    since one is measured from holds and the other from transients, and the output
    is three angles rather than two.
    """
    fg = "white" if bg == "black" else "#111111"
    dim = "#555555" if bg == "white" else "#aaaaaa"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    X, W, H = 0.045, 0.91, 0.108
    CX = X + W / 2

    def box(y, fill, title, sub, fs=10.5):
        ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle="round,pad=0.010",
                                    linewidth=1.0, edgecolor="0.45",
                                    facecolor=fill, zorder=2))
        ax.text(CX, y + H * 0.64, title, ha="center", va="center",
                fontsize=fs, color="#111111", zorder=3)
        ax.text(CX, y + H * 0.24, sub, ha="center", va="center",
                fontsize=7.8, color="#333333", zorder=3)

    def down(y0, y1):
        ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), color="0.35",
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.3, zorder=4))

    ax.text(CX, 0.972, r"$(\dot x,\dot y)$", ha="center", va="center",
            fontsize=13, color=fg)
    ax.text(CX, 0.936, "target velocity  (the only input)", ha="center",
            va="center", fontsize=7.8, color=dim)
    down(0.918, 0.895)

    box(0.787, "#c7d8e8",
        r"$\mathbf{I}=\hat W^{\rm in}\,(\dot x,\dot y)^{\!\top}$",
        r"AF5 afferents,  $\hat W^{\rm in}\!\in\mathbb{R}^{N\times 2}$")
    down(0.787, 0.764)

    box(0.656, "#cfe8d2",
        r"$\tau_i\dot v_i=-v_i+\sum_j \hat W_{ij}\,{\rm ReLU}(v_j)+I_i$",
        r"INTG recurrent core,  "
        r"$\hat W_{ij}=|\hat S_{ij}|\,{\rm sign}(W^{\rm con}_{ij})$", fs=11.0)
    down(0.656, 0.633)

    box(0.525, "#ddd0f0",
        r"$\mathbf{m}=[\hat W^{\rm out}\mathbf{r}]_+\ \geq 0$",
        r"six muscle drives,  $\hat W^{\rm out}\!\in\mathbb{R}^{6\times N}$"
        "   (LR SR MR IR SO IO)")
    down(0.525, 0.502)

    box(0.394, "#f6e6ce",
        r"$x^{k}_{\infty}=\sum_i a^{k}_i m_i+\sum_{i\leq j} b^{k}_{ij} m_i m_j$",
        r"static map, measured;  $k\in\{\theta,\varphi,\psi\}$,  27 coefficients each")
    down(0.394, 0.371)

    box(0.263, "#f4d6e6",
        r"$\ddot{\mathbf{x}}+C\dot{\mathbf{x}}+K\mathbf{x}=K\mathbf{x}_\infty$",
        r"mechanics, fitted;  $C,K\in\mathbb{R}^{3\times3}$, both frozen", fs=11.0)
    down(0.263, 0.240)

    ax.text(CX, 0.203, r"$(\theta,\varphi,\psi)$", ha="center", va="center",
            fontsize=13, color=fg)
    ax.text(CX, 0.167, "gaze angles: horizontal, vertical, torsion", ha="center",
            va="center", fontsize=7.8, color=dim)
    ax.text(CX, 0.088,
            r"loss  $\mathcal{L}=\frac{1}{T}\sum_t[(\theta-\theta^\star)^2"
            r"+(\varphi-\varphi^\star)^2]"
            r"+\lambda_\psi\frac{1}{T}\sum_t\psi^2$",
            ha="center", va="center", fontsize=10.5, color=fg)
    ax.text(CX, 0.036,
            "two angles matched to the target, the third penalised toward zero",
            ha="center", va="center", fontsize=8.2, color=dim)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "fig_eye_schematic.png"))
    p.add_argument("--bg", default="white", choices=["white", "black"])
    a = p.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 7.6), facecolor=a.bg,
                           gridspec_kw=dict(width_ratios=[1.0, 1.30],
                                            wspace=0.05))
    for x in ax:
        x.set_facecolor(a.bg)
    panel_eye(ax[0], a.bg)
    panel_vars(ax[1], a.bg)
    fg = "white" if a.bg == "black" else "black"
    for x, L in zip(ax, "ab"):
        x.text(0.0, 1.0, L, transform=x.transAxes, fontsize=14,
               fontweight="bold", va="top", ha="left", color=fg)
    fig.savefig(a.out, dpi=170, facecolor=a.bg, bbox_inches="tight")
    print("wrote", a.out)
    print("driven:", ", ".join(DRIVEN),
          "| all six:", ", ".join(m["key"] for m in EA.MUSCLES))


if __name__ == "__main__":
    main()

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
# left edge of panel (b)'s blocks, in its axes coordinates; the panel label
# is lined up with it rather than with the axes box
X_BOX = 0.045


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
    now shows the render itself, produced by ``fig_eyeG_anterior.py``, with no
    labels or legend over it: the muscles are named and their measured actions
    tabulated in section 4.2, and a key laid on top of a 1100-px render costs
    more of the picture than it explains.
    """
    import matplotlib.image as mpimg
    fg = "white" if bg == "black" else "#111111"
    png = os.path.join(HERE, "fig_eyeG_anterior.png")
    ax.axis("off")
    if not os.path.isfile(png):
        ax.text(0.5, 0.5, "run fig_eyeG_anterior.py first", ha="center",
                va="center", color=fg, fontsize=11)
        return
    img = mpimg.imread(png)
    ax.imshow(img)
    # No legend and no strap labels. Six labels placed at their own projected
    # positions collide in the rostral corner where four of the six muscles
    # converge, and a legend covers the render it explains -- which muscle is
    # which is the subject of section 4.2, not of this panel. The panel shows
    # the geometry; the caption says what it is.


def panel_vars(ax, bg):
    """Every symbol of section 4, in the order the signal meets them.

    Stacked top to bottom because the signal path is a straight line. Six stages
    now rather than four: the static map and the mechanics are separate blocks,
    since one is measured from holds and the other from transients, and the output
    is three angles rather than two.
    """
    fg = "white" if bg == "black" else "#111111"
    dim = "#555555" if bg == "white" else "#aaaaaa"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.055); ax.axis("off")

    X, W, H = X_BOX, 0.91, 0.108
    CX = X + W / 2

    def box(y, fill, title, sub, fs=9.7):
        ax.add_patch(FancyBboxPatch((X, y), W, H, boxstyle="round,pad=0.010",
                                    linewidth=1.0, edgecolor="0.45",
                                    facecolor=fill, zorder=2))
        ax.text(CX, y + H * 0.64, title, ha="center", va="center",
                fontsize=fs, color="#111111", zorder=3)
        ax.text(CX, y + H * 0.24, sub, ha="center", va="center",
                fontsize=8.0, color="#333333", zorder=3)

    def down(y0, y1):
        ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), color="0.35",
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.3, zorder=4))

    ax.text(CX, 0.972, r"$(\dot x,\dot y)$", ha="center", va="center",
            fontsize=13, color=fg)
    ax.text(CX, 0.936, "target velocity  (the only input)", ha="center",
            va="center", fontsize=8.0, color=dim)
    down(0.918, 0.895)

    box(0.787, "#c7d8e8",
        r"$\mathbf{I}=\hat W^{\rm in}\,(\dot x,\dot y)^{\!\top}$",
        r"AF5 afferents,  $\hat W^{\rm in}\!\in\mathbb{R}^{N\times 2}$")
    down(0.787, 0.764)

    box(0.656, "#cfe8d2",
        r"$\tau_i\dot v_i=-v_i+\sum_j \hat W_{ij}\,{\rm ReLU}(v_j)+I_i$",
        r"INTG recurrent core,  "
        r"$\hat W_{ij}=|\hat S_{ij}|\,{\rm sign}(W^{\rm con}_{ij})$", fs=10.1)
    down(0.656, 0.633)

    box(0.525, "#ddd0f0",
        r"$\mathbf{m}=[\hat W^{\rm out}\mathbf{r}]_+\ \geq 0$",
        r"six muscle drives,  $\hat W^{\rm out}\!\in\mathbb{R}^{6\times N}$"
        "   (LR SR MR IR SO IO)")
    down(0.525, 0.502)

    box(0.394, "#f6e6ce",
        r"$u^{k}_{\infty}=\sum_i a^{k}_i m_i+\sum_{i\leq j} b^{k}_{ij} m_i m_j$",
        r"static map, measured;  $k\in\{\theta,\varphi,\psi\}$,  27 coefficients each")
    down(0.394, 0.371)

    box(0.263, "#f4d6e6",
        r"$\ddot{\mathbf{u}}+C\dot{\mathbf{u}}+K\mathbf{u}=K\mathbf{u}_\infty$",
        r"mechanics, fitted;  $C,K\in\mathbb{R}^{3\times3}$, both frozen", fs=10.1)
    down(0.263, 0.240)

    ax.text(CX, 0.203, r"$\mathbf{u}(t)=(\theta,\varphi,\psi)^{\!\top}$",
            ha="center", va="center", fontsize=13, color=fg)
    ax.text(CX, 0.167, "gaze angles: horizontal, vertical, torsion", ha="center",
            va="center", fontsize=8.0, color=dim)
    ax.text(CX, 0.088,
            r"loss  $\mathcal{L}=\frac{1}{T}\sum_t[(\theta-\theta^\star)^2"
            r"+(\varphi-\varphi^\star)^2]"
            r"+\lambda_\psi\frac{1}{T}\sum_t\psi^2$",
            ha="center", va="center", fontsize=9.7, color=fg)
    ax.text(CX, 0.036,
            "two angles matched to the target, the third penalised toward zero",
            ha="center", va="center", fontsize=8.0, color=dim)


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
    # Above and left of each panel, not inside it: panel (a) is a black render, so
    # a label at (0, 1) in axes coordinates was black on black and invisible.
    # Both labels are placed in FIGURE coordinates on one shared baseline. In
    # axes coordinates they came out at different heights, because imshow forces
    # aspect="equal" and matplotlib shrinks panel (a)'s axes box to fit the
    # render, so its y=1.035 sits well below panel (b)'s. The shrink only happens
    # at draw time, which is why the canvas is drawn first and the real extents
    # read back off it.
    # 13 pt here, 16 in fig_1_oculomotor_overview.py and 26 in
    # plot_oculomotor_connectome.py: all three figures go in at \textwidth, so
    # equal PRINTED height means equal fontsize/(72 * figure width in inches),
    # and those widths are 11.0, 13.7 and 22.2. The common target is
    # fig_eyeG_charac.py's 17 pt on a 14.5 in canvas.
    #
    # Both labels are placed in FIGURE coordinates against the CONTENT of their
    # panel, not against its axes box. Two things make the axes box the wrong
    # anchor here: imshow forces aspect="equal", so panel (a)'s box is shrunk to
    # the square render at draw time and its nominal top sits far above the black
    # edge; and panel (b)'s blocks are inset at x=0.045 with the whole column
    # centred, so its box starts well left of the blue box. The canvas is drawn
    # first because both corrections are only knowable after layout.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    img = inv.transform(ax[0].images[0].get_window_extent().get_points())
    x_b = inv.transform(ax[1].transAxes.transform([(X_BOX, 0.0)]))[0][0]
    y = img[1][1] + 0.033                       # clear of the render's top edge
    for x, L in ((img[0][0], "a"), (x_b, "b")):
        fig.text(x, y, L, fontsize=13, fontweight="bold", va="bottom",
                 ha="left", color=fg)
    fig.savefig(a.out, dpi=170, facecolor=a.bg, bbox_inches="tight")
    print("wrote", a.out)
    print("driven:", ", ".join(DRIVEN),
          "| all six:", ", ".join(m["key"] for m in EA.MUSCLES))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""fig_spheroid_bm_ecm -- the three entities, their real proportions, and the operators between them.

    python fig_spheroid_bm_ecm.py        ->  fig_spheroid_bm_ecm.pdf

Two panels, and they answer different questions. (a) is anatomy at the proportions the literature
measures -- a basement membrane ~100 nm thick, hemidesmosome plaques ~300-400 nm across on ~600-800 nm
centres, the linkage across the lamina lucida ~30 nm -- because those ratios decide what a model can
resolve, and every earlier figure in this note drew the adhesion as a slender thread, which it is not.
(b) is the operator graph: one arrow per mechanism, in the direction it acts, with the ones that carry
a force drawn solid (they must have a reaction) and the ones that carry information dashed.

Drawn in the same visual language as `fig_bm_schematic.pdf`: white ground, tan cells, a green sheet,
orange adhesions, the matrix as a scatter of fibre segments.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

CELL = "#e8dcc0"
CELL_E = "#8a7f63"
BM = "#1f8a5c"
ADH = "#e08a2e"
ECM = "#5b5b9c"
TXT = "#222222"


def panel_anatomy(ax):
    """A section through the three entities, at the measured proportions."""
    T = 1.0                                   # the basement membrane's thickness sets the scale
    L = 0.3 * T                               # integrin a6b4 linkage across the lamina lucida, ~30 nm
    D = 3.5 * T                               # hemidesmosome plaque, ~300-400 nm across
    S = 7.0 * T                               # plaque spacing, ~600-800 nm
    z_cell = 0.0                              # basal plasma membrane
    x0, x1 = -1.0, 3 * S + 1.0

    # --- epithelium: cell bodies above the basal surface
    for k in range(-1, 4):
        xa, xb = k * S - 0.5 * S, (k + 1) * S - 0.5 * S
        ax.add_patch(Polygon([[xa, z_cell], [xb, z_cell], [xb - 0.3, z_cell + 5 * T],
                              [xa + 0.3, z_cell + 5 * T]],
                             facecolor=CELL, edgecolor=CELL_E, lw=0.8, zorder=2))  # noqa: E501
    ax.plot([x0, x1], [z_cell, z_cell], color=CELL_E, lw=1.4, zorder=3)

    # --- hemidesmosome plaques: squat, wider than the sheet is thick
    for k in range(4):
        cx = k * S
        ax.add_patch(Rectangle((cx - D / 2, z_cell - L), D, L, facecolor=ADH, edgecolor="none",
                               alpha=0.85, zorder=4))
        ax.add_patch(Rectangle((cx - D / 2, z_cell - 0.12 * T), D, 0.12 * T, facecolor=ADH,
                               edgecolor="none", zorder=5))

    # --- the basement membrane: a continuous sheet, not a row of nodes
    z_bm = z_cell - L - T
    ax.add_patch(Rectangle((x0, z_bm), x1 - x0, T, facecolor=BM, edgecolor="none", alpha=0.30,
                           zorder=3))
    ax.add_patch(Rectangle((x0, z_bm + 0.62 * T), x1 - x0, 0.38 * T, facecolor=BM, edgecolor="none",
                           alpha=0.55, zorder=4))
    ax.text(x1 - 0.2, z_bm + 0.80 * T, "lamina lucida", ha="right", va="center", fontsize=6.5,
            color="#0d5c3c")
    ax.text(x1 - 0.2, z_bm + 0.30 * T, "lamina densa: collagen IV + laminin", ha="right", va="center",
            fontsize=6.5, color="#0d5c3c")

    # --- interstitial matrix: fibre segments, anchored to the sheet by a few fibrils
    rng = np.random.default_rng(3)
    for _ in range(150):
        cx = rng.uniform(x0, x1); cz = z_bm - rng.uniform(0.4, 7.5) * T
        th = rng.uniform(0, np.pi)
        ln = rng.uniform(0.5, 1.6) * T
        ax.plot([cx - ln * np.cos(th), cx + ln * np.cos(th)],
                [cz - ln * np.sin(th) * 0.5, cz + ln * np.sin(th) * 0.5],
                color=ECM, lw=0.7, alpha=0.55, zorder=1)
    for k in range(4):
        cx = k * S + 0.5 * S
        ax.plot([cx, cx + 0.4 * T], [z_bm, z_bm - 1.6 * T], color=ECM, lw=1.3, alpha=0.9, zorder=2)

    # --- the four lengths, as measured
    def dim(x, z0, z1, label, dx=0.0):
        ax.annotate("", xy=(x + dx, z0), xytext=(x + dx, z1),
                    arrowprops=dict(arrowstyle="<->", color=TXT, lw=0.7, shrinkA=0, shrinkB=0))
        ax.text(x + dx - 0.35, 0.5 * (z0 + z1), label, fontsize=6.5, va="center", ha="right",
                color=TXT)

    dim(-1.6, z_bm, z_bm + T, "BM  $T\\approx100$ nm")
    dim(-1.6, z_cell - L, z_cell, "linkage  $0.3\\,T\\approx30$ nm")
    ax.annotate("", xy=(0 - D / 2, z_cell + 0.9 * T), xytext=(0 + D / 2, z_cell + 0.9 * T),
                arrowprops=dict(arrowstyle="<->", color=TXT, lw=0.7))
    ax.text(0, z_cell + 1.3 * T, "plaque  $3.5\\,T\\approx350$ nm", ha="center", fontsize=6.5,
            color=TXT)
    ax.annotate("", xy=(0, z_cell + 3.4 * T), xytext=(S, z_cell + 3.4 * T),
                arrowprops=dict(arrowstyle="<->", color=TXT, lw=0.7))
    ax.text(S / 2, z_cell + 3.8 * T, "spacing  $7\\,T\\approx700$ nm", ha="center", fontsize=6.5,
            color=TXT)

    ax.text(x1 - 0.2, z_cell + 3.9 * T, "epithelium", ha="right", fontsize=8, color=CELL_E)
    ax.text(x1 - 0.2, z_bm - 5.5 * T, "interstitial matrix", ha="right", fontsize=8, color=ECM)
    ax.text(x0, z_bm - 7.4 * T, "the adhesion is SQUAT -- wider than the sheet is thick, a third as "
                                "long, on 7-thickness centres", ha="left", fontsize=6.8, color=ADH)
    ax.set_xlim(x0 - 7.0, x1 + 0.2); ax.set_ylim(z_bm - 8.2 * T, z_cell + 5.2 * T)
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def panel_operators(ax):
    """One arrow per mechanism. Solid = carries force (and therefore a reaction); dashed = signal."""
    boxes = {"epi": (0.5, 0.86, "epithelium\ngrows, divides, contracts"),
             "bm": (0.5, 0.50, "basement membrane\nassembled, cross-linked, replaced"),
             "ecm": (0.5, 0.17, "interstitial matrix\naligned, cross-linked, degraded")}
    for k, (x, y, lab) in boxes.items():
        ax.add_patch(Rectangle((x - 0.26, y - 0.055), 0.52, 0.11, facecolor="#f4f4f4",
                               edgecolor="#666666", lw=0.8, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=7.6, zorder=4, color=TXT)

    def pair(y_top, y_bot, down, up):
        """One arrow each way, with the operators listed beside it -- not one arrow per operator.

        The first version drew seven parallel arrows between two boxes and put a label on each; at this
        width every label landed on its neighbours and the panel said nothing. A direction is one
        arrow; what travels along it is a list.
        """
        ax.add_patch(FancyArrowPatch((0.38, y_top), (0.38, y_bot), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.4, color="#555", zorder=2))
        ax.add_patch(FancyArrowPatch((0.62, y_bot), (0.62, y_top), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.4, color="#555", zorder=2))
        for i, (txt, force, col) in enumerate(down):
            ax.text(0.355, y_top - 0.028 - i * 0.026, ("\u25cf " if force else "\u25cb ") + txt,
                    fontsize=6.6, ha="right", va="top", color=col)
        for i, (txt, force, col) in enumerate(up):
            ax.text(0.645, y_top - 0.028 - i * 0.026, ("\u25cf " if force else "\u25cb ") + txt,
                    fontsize=6.6, ha="left", va="top", color=col)

    pair(0.800, 0.562,
         [("secretion of laminin, collagen IV,", False, "#444"),
          ("   nidogen, perlecan", False, "#444"),
          ("nucleated assembly (integrin,", False, "#444"),
          ("   dystroglycan)", False, "#444"),
          ("mechanical loading", True, ADH),
          ("proteolysis (MMP, invadopodia)", False, "#a33")],
         [("adhesion (plaques)", True, ADH),
          ("shape constraint", True, BM),
          ("stiffness sensing", False, BM),
          ("polarity cue", False, "#444"),
          ("anoikis suppression", False, "#444"),
          ("growth-factor reservoir", False, "#444")])
    pair(0.440, 0.232,
         [("load transmission via", True, BM),
          ("   anchoring fibrils", True, BM),
          ("barrier: pores 10-130 nm", False, "#444")],
         [("confinement reaction", True, ECM),
          ("stromal supply", False, ECM),
          ("stromal MMPs", False, "#a33")])

    ax.text(0.5, 0.072, "\u25cf carries force, so it has a reaction        "
                        "\u25cb signal or material flux", ha="center", fontsize=6.8, color="#555")
    ax.text(0.5, 0.028, "no arrow joins the epithelium to the matrix: while the sheet is intact,\n"
                        "the basement membrane IS the interface", ha="center", fontsize=6.8,
            color="#777")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 1.0, "b", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def main():
    fig = plt.figure(figsize=(9.8, 4.6), facecolor="white")
    ax1 = fig.add_axes([0.005, 0.02, 0.56, 0.96])
    ax2 = fig.add_axes([0.575, 0.02, 0.425, 0.96])
    panel_anatomy(ax1)
    panel_operators(ax2)
    out = os.path.join(HERE, "fig_spheroid_bm_ecm.pdf")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()

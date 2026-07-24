#!/usr/bin/env python
"""Small before/after schematic of a T1 transition (neighbour exchange) for monolayer_tube_note.tex.
Left: cells alpha,beta (orange) share a short edge; gamma,delta (blue) are kept apart. The edge shrinks to a
four-fold vertex and reopens rotated 90 deg. Right: now gamma,delta share the edge, alpha,beta are separated."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, FancyArrowPatch

ORANGE, BLUE, EDGE = "#e8b06a", "#7fa8d0", "#c0392b"
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6)); fig.patch.set_facecolor("white")


def cell(a, pts, fc, lbl=None, lc="black"):
    a.add_patch(MplPoly(np.array(pts), closed=True, facecolor=fc, edgecolor="k", lw=1.3))
    if lbl:
        c = np.array(pts).mean(0)
        a.text(c[0], c[1], lbl, ha="center", va="center", fontsize=12, color=lc, style="italic")


# ---------- before: vertical shared edge (alpha|beta) ----------
Pt, Pb = (0, 0.42), (0, -0.42)
cell(ax[0], [(-1.7, -1.15), Pb, Pt, (-1.7, 1.15)], ORANGE, r"$\alpha$")
cell(ax[0], [(1.7, -1.15), Pb, Pt, (1.7, 1.15)], ORANGE, r"$\beta$")
cell(ax[0], [(-1.7, 1.15), Pt, (1.7, 1.15), (1.7, 1.95), (-1.7, 1.95)], BLUE, r"$\gamma$")
cell(ax[0], [(-1.7, -1.15), Pb, (1.7, -1.15), (1.7, -1.95), (-1.7, -1.95)], BLUE, r"$\delta$")
ax[0].plot([Pt[0], Pb[0]], [Pt[1], Pb[1]], color=EDGE, lw=3.2, solid_capstyle="round")
ax[0].text(0.28, 0.0, "shared\nedge", color=EDGE, fontsize=8.5, va="center")
ax[0].set_title(r"before:  $\alpha,\beta$ neighbours", fontsize=10)

# ---------- after: horizontal shared edge (gamma|delta) ----------
Ql, Qr = (-0.42, 0), (0.42, 0)
cell(ax[1], [(-1.7, 1.15), Ql, Qr, (1.7, 1.15), (1.7, 1.95), (-1.7, 1.95)], BLUE, r"$\gamma$")
cell(ax[1], [(-1.7, -1.15), Ql, Qr, (1.7, -1.15), (1.7, -1.95), (-1.7, -1.95)], BLUE, r"$\delta$")
cell(ax[1], [(-1.7, -1.15), Ql, (-1.7, 1.15)], ORANGE, r"$\alpha$")
cell(ax[1], [(1.7, -1.15), Qr, (1.7, 1.15)], ORANGE, r"$\beta$")
ax[1].plot([Ql[0], Qr[0]], [Ql[1], Qr[1]], color=EDGE, lw=3.2, solid_capstyle="round")
ax[1].text(0.0, 0.26, "new edge", color=EDGE, fontsize=8.5, ha="center")
ax[1].set_title(r"after:  $\gamma,\delta$ neighbours", fontsize=10)

for a in ax:
    a.set_xlim(-2.0, 2.0); a.set_ylim(-2.05, 2.05); a.set_aspect("equal"); a.axis("off")
# arrow between panels
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.18)
fig.text(0.5, 0.5, r"$\longrightarrow$", ha="center", va="center", fontsize=22)
fig.text(0.5, 0.60, "edge $\\to 0$\n(4-fold vertex)", ha="center", va="center", fontsize=8, color="#555")
fig.savefig("t1_schematic.png", dpi=150, facecolor="white")
print("wrote t1_schematic.png")

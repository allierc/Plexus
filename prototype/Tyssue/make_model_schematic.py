#!/usr/bin/env python
"""Three-panel schematic: SPV | mid-surface AVM | monolayer AVM -- for monolayer_tube_note.tex."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPoly
from scipy.spatial import Voronoi, voronoi_plot_2d

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6)); fig.patch.set_facecolor("white")
BLUE, RED, GRAY, ORANGE = "#2b6cb0", "#c0392b", "#555555", "#d97706"

# ---------------- Panel A: Self-Propelled Voronoi ----------------
a = ax[0]; g = np.random.default_rng(3)
pts = np.array([[x + 0.18 * g.standard_normal(), y + 0.18 * g.standard_normal()]
                for x in range(5) for y in range(5)], float)
vor = Voronoi(pts)
voronoi_plot_2d(vor, ax=a, show_points=False, show_vertices=False, line_colors=GRAY, line_width=1.1)
inside = (pts[:, 0] > 0.5) & (pts[:, 0] < 3.5) & (pts[:, 1] > 0.5) & (pts[:, 1] < 3.5)
a.plot(pts[inside, 0], pts[inside, 1], "o", color="black", ms=5)
c = pts[np.argmin(np.linalg.norm(pts - [2, 2], axis=1))]
a.add_patch(FancyArrowPatch(c, c + [0.55, 0.35], arrowstyle="-|>", mutation_scale=15, color=RED, lw=2))
a.text(c[0] + 0.58, c[1] + 0.42, r"$v_0\,\hat n$", color=RED, fontsize=12)
a.annotate("cell centre $r_i$", (c[0], c[1]), (c[0] - 1.3, c[1] - 1.15),
           fontsize=10, arrowprops=dict(arrowstyle="->", color="black"))
a.text(2.0, -0.35, "Voronoi polygon (derived)\nimplicit T1 (retessellate each step)",
       ha="center", fontsize=9, color=GRAY)
a.set_xlim(0.3, 3.7); a.set_ylim(0.1, 3.7)
a.set_title(r"A.  Self-Propelled Voronoi" "\n" r"$U=\sum_i K_A(A_i-A_0)^2+K_P(P_i-P_0)^2$", fontsize=11)

# ---------------- Panel B: mid-surface AVM ----------------
b = ax[1]
# a small honeycomb of shared-vertex hexagons
def hexc(cx, cy, r=0.6):
    th = np.pi/6 + np.arange(6)*np.pi/3
    return np.c_[cx + r*np.cos(th), cy + r*np.sin(th)]
centres = [(0, 0), (1.04, 0.6), (1.04, -0.6), (-1.04, 0.6), (-1.04, -0.6), (0, 1.2), (0, -1.2)]
allv = []
for cx, cy in centres:
    h = hexc(cx, cy); b.add_patch(MplPoly(h, closed=True, fill=False, edgecolor=GRAY, lw=1.3)); allv.append(h)
V = np.vstack(allv); Vu = np.unique(np.round(V, 3), axis=0)
b.plot(Vu[:, 0], Vu[:, 1], "o", color=RED, ms=4)
# wedge volume: dashed lines from one cell's ring to a centre O below (side hint)
h0 = hexc(0, 0); O = np.array([0.0, -2.55])
for v in h0:
    b.plot([v[0], O[0]], [v[1], O[1]], "--", color=BLUE, lw=0.7, alpha=0.7)
b.plot(*O, "s", color=BLUE, ms=6); b.text(O[0] + 0.12, O[1], "$O$ (shell centre)", color=BLUE, fontsize=9)
b.text(0.0, 0.0, r"$v_f=\frac{1}{3} c_f\!\cdot\!N_f$", ha="center", fontsize=9, color=BLUE)
b.annotate("vertex $x_i$ = DOF\n(shared junction)", (Vu[np.argmax(Vu[:, 1])][0], Vu[np.argmax(Vu[:, 1])][1]),
           (1.35, 1.7), fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
b.text(0.0, -3.15, "explicit T1 flip", ha="center", fontsize=9, color=GRAY)
b.set_xlim(-2.0, 2.4); b.set_ylim(-3.35, 2.2); b.set_aspect("equal")
b.set_title(r"B.  Mid-surface vertex model" "\n" r"$U=\sum_f K_A(A{-}A^0)^2{+}K_P(P{-}P^0)^2{+}K_V(v{-}v^{eq})^2$", fontsize=10)

# ---------------- Panel C: monolayer AVM (cross-section) ----------------
c2 = ax[2]
s = np.linspace(-1.5, 1.5, 7)
mid = np.c_[s, 0.5 * np.cos(s * 0.9)]                      # curved mid-surface
# outward normals
d = np.gradient(mid, axis=0); n = np.c_[-d[:, 1], d[:, 0]]; n /= np.linalg.norm(n, axis=1, keepdims=True)
h = 0.42
ap = mid + (h/2) * n; ba = mid - (h/2) * n                # apical (outer) / basal (inner)
c2.plot(ap[:, 0], ap[:, 1], "-", color=RED, lw=2, label="apical (outer)")
c2.plot(ba[:, 0], ba[:, 1], "-", color=BLUE, lw=2, label="basal (inner)")
for i in range(len(s)):                                    # lateral walls = cell boundaries
    c2.plot([ap[i, 0], ba[i, 0]], [ap[i, 1], ba[i, 1]], "-", color=GRAY, lw=1.1)
for i in range(len(s) - 1):                                # shade cells
    quad = np.array([ap[i], ap[i+1], ba[i+1], ba[i]])
    c2.add_patch(MplPoly(quad, closed=True, facecolor="#fdf0e6", edgecolor="none", zorder=0))
i = 1
c2.annotate("", xy=ap[i], xytext=ba[i], arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.6))
c2.text(ap[i, 0] - 0.42, (ap[i, 1] + ba[i, 1]) / 2, "$h$", color=ORANGE, fontsize=12)
c2.text(0.0, 1.15, "apical arc $>$ basal arc on a curve\n$\\Rightarrow$ emergent bending", ha="center", fontsize=9, color=RED)
c2.text(0.0, -1.15, r"$v=A^{mid}h,\ \ s=A^{ap}{+}A^{ba}{+}A^{lat}$", ha="center", fontsize=9.5, color="black")
c2.text(1.62, ap[-1, 1], r"$a_i=x_i+\frac{h}{2}n_i$", color=RED, fontsize=9)
c2.text(1.62, ba[-1, 1], r"$b_i=x_i-\frac{h}{2}n_i$", color=BLUE, fontsize=9)
c2.legend(loc="upper right", fontsize=8, framealpha=0.9)
c2.set_xlim(-1.9, 2.9); c2.set_ylim(-1.5, 1.6); c2.set_aspect("equal")
c2.set_title(r"C.  Monolayer vertex model (cross-section)" "\n" r"$U=\sum_j \frac{1}{2} k_v(v{-}v^{eq})^2+\kappa_s s+\frac{1}{2}\gamma P^2$", fontsize=10)

for a_ in ax:
    a_.set_xticks([]); a_.set_yticks([])
    for sp in a_.spines.values():
        sp.set_visible(False)
fig.tight_layout()
fig.savefig("model_schematic.png", dpi=150, facecolor="white", bbox_inches="tight")
print("wrote model_schematic.png")

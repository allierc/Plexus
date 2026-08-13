#!/usr/bin/env python
"""Three-panel schematic for monolayer_tube_note.tex:
  a  self-propelled Voronoi (2D)          -- cell = point, polygon derived, implicit T1
  b  mid-surface 3D vertex model          -- cells are 3D (shared vertices, wedge volume); the round_* tubing model
  c  monolayer vertex model (cross-section)-- prism cells with apical/basal offset + real thickness h
Panel b is drawn as a genuine 3D cell slab (Okuda 2018 Fig 1c style: Vertex / Boundary / Cell) because the
Tyssue prototype's vertices live in 3D -- a flat 2D honeycomb misrepresents it.  Bold lowercase panel labels
sit top-left (no titles); model name + energy sit as captions below each panel."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPoly
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Voronoi, voronoi_plot_2d

BLUE, RED, GRAY, ORANGE, BEIGE = "#2b6cb0", "#c0392b", "#555555", "#d97706", "#fbe3c2"

fig = plt.figure(figsize=(14.0, 5.2)); fig.patch.set_facecolor("white")
axA = fig.add_subplot(1, 3, 1)
axB = fig.add_subplot(1, 3, 2, projection="3d")
axC = fig.add_subplot(1, 3, 3)


def panel_label(ax, s, three_d=False):
    (ax.text2D if three_d else ax.text)(0.02, 0.97, s, transform=ax.transAxes,
                                        fontsize=18, fontweight="bold", va="top", ha="left", color="black")


# ============================ a: Self-Propelled Voronoi (2D) ============================
g = np.random.default_rng(3)
pts = np.array([[x + 0.18 * g.standard_normal(), y + 0.18 * g.standard_normal()]
                for x in range(5) for y in range(5)], float)
vor = Voronoi(pts)
voronoi_plot_2d(vor, ax=axA, show_points=False, show_vertices=False, line_colors=GRAY, line_width=1.1)
inside = (pts[:, 0] > 0.5) & (pts[:, 0] < 3.5) & (pts[:, 1] > 0.5) & (pts[:, 1] < 3.5)
axA.plot(pts[inside, 0], pts[inside, 1], "o", color="black", ms=5)
c = pts[np.argmin(np.linalg.norm(pts - [2, 2], axis=1))]
axA.add_patch(FancyArrowPatch(c, c + [0.55, 0.35], arrowstyle="-|>", mutation_scale=15, color=RED, lw=2))
axA.text(c[0] + 0.58, c[1] + 0.42, r"$v_0\,\hat n$", color=RED, fontsize=12)
axA.annotate("cell centre $r_i$", (c[0], c[1]), (c[0] - 1.35, c[1] - 1.15),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="black"))
axA.set_xlim(0.3, 3.7); axA.set_ylim(0.1, 3.7); axA.set_aspect("equal")
panel_label(axA, "a")

# ============================ b: mid-surface 3D VERTEX MODEL (Okuda Fig 1c) ============================
def hexv(cx, cy, r):
    th = np.radians(np.arange(0, 360, 60))                     # flat-top hexagon
    return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]

r, dep = 0.62, 0.74
cens = []
for col in range(-1, 2):                                       # 3x3 honeycomb that tiles
    for row in range(-1, 2):
        cx = 1.5 * r * col
        cy = np.sqrt(3) * r * row + (col % 2) * np.sqrt(3) * r / 2
        cens.append((cx, cy))
zt, zb = dep / 2, -dep / 2
HI = 4                                                          # centre cell -> highlighted (orange)
faces, fcol, fedge = [], [], []
grey_face = None; vtx_label = None
for ci, (cx, cy) in enumerate(cens):
    hv = hexv(cx, cy, r)
    top = np.c_[hv, np.full(6, zt)]; bot = np.c_[hv, np.full(6, zb)]
    base = ORANGE if ci == HI else BEIGE
    faces += [top, bot]; fcol += [base, base]; fedge += ["k", "k"]
    for k in range(6):
        quad = np.array([top[k], top[(k + 1) % 6], bot[(k + 1) % 6], bot[k]])
        if ci == HI and k == 0:                                # shared wall toward the NE neighbour -> "Boundary"
            grey_face = quad
        else:
            faces.append(quad); fcol.append(base); fedge.append("k")
    if ci == HI:
        vtx_label = top[1]                                     # a top corner -> "Vertex"
pc = Poly3DCollection(faces, facecolors=fcol, edgecolors=fedge, linewidths=0.5, alpha=0.97)
axB.add_collection3d(pc)
if grey_face is not None:                                      # the highlighted boundary wall, drawn on top
    axB.add_collection3d(Poly3DCollection([grey_face], facecolors="#8a7a6a", edgecolors="k",
                                          linewidths=0.9, alpha=0.98))
# vertices as dots
allv = np.vstack([np.c_[hexv(cx, cy, r), np.full(6, zt)] for cx, cy in cens])
axB.scatter(allv[:, 0], allv[:, 1], allv[:, 2], color="k", s=9, depthshade=False)
# labels (Okuda Fig 1c): Vertex / Boundary / Cell -- text2D overlays (always on top of the opaque mesh)
lblbb = dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85)
for xf, yf, txt, col in [(0.51, 0.92, "Vertex", BLUE), (0.51, 0.66, "Cell", RED),
                         (0.51, 0.30, "Boundary", "#6b5b4a")]:
    t = axB.text2D(xf, yf, txt, transform=axB.transAxes, color=col, fontsize=10.5, ha="center", bbox=lblbb)
    t.set_zorder(1e6)                                          # force above the opaque 3D faces
axB.set_xlim(-1.4, 1.4); axB.set_ylim(-1.4, 1.4); axB.set_zlim(zb - 0.25, zt + 0.35)
axB.set_box_aspect((1, 1, 0.6)); axB.view_init(elev=20, azim=-62); axB.set_axis_off()
panel_label(axB, "b", three_d=True)
axB.text2D(0.5, 0.02, "cell deformation $\\cdot$ rearrangement (T1) $\\cdot$ division $\\cdot$ apoptosis",
           transform=axB.transAxes, ha="center", fontsize=8, color=GRAY)

# ============================ c: monolayer vertex model (cross-section) ============================
s = np.linspace(-1.5, 1.5, 7)
mid = np.c_[s, 0.5 * np.cos(s * 0.9)]                           # curved mid-surface
d = np.gradient(mid, axis=0); n = np.c_[-d[:, 1], d[:, 0]]; n /= np.linalg.norm(n, axis=1, keepdims=True)
h = 0.42
ap = mid + (h / 2) * n; ba = mid - (h / 2) * n                 # apical (outer) / basal (inner)
for i in range(len(s) - 1):
    axC.add_patch(MplPoly(np.array([ap[i], ap[i + 1], ba[i + 1], ba[i]]), closed=True,
                          facecolor=BEIGE, edgecolor="none", zorder=0))
axC.plot(ap[:, 0], ap[:, 1], "-", color=RED, lw=2, label="apical (outer)")
axC.plot(ba[:, 0], ba[:, 1], "-", color=BLUE, lw=2, label="basal (inner)")
for i in range(len(s)):
    axC.plot([ap[i, 0], ba[i, 0]], [ap[i, 1], ba[i, 1]], "-", color=GRAY, lw=1.1)
i = 1
axC.annotate("", xy=ap[i], xytext=ba[i], arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
axC.text(ap[i, 0], ap[i, 1] + 0.17, "$h$", color="black", fontsize=13, ha="center")
axC.text(0.0, 1.16, "apical arc $>$ basal arc on a curve\n$\\Rightarrow$ emergent bending", ha="center",
         fontsize=9, color=RED)
axC.text(1.62, ap[-1, 1], r"$a_i=x_i+\frac{h}{2}n_i$", color=RED, fontsize=9)
axC.text(1.62, ba[-1, 1], r"$b_i=x_i-\frac{h}{2}n_i$", color=BLUE, fontsize=9)
axC.legend(loc="lower center", fontsize=8, framealpha=0.9, ncol=2)
axC.set_xlim(-1.9, 2.9); axC.set_ylim(-1.5, 1.6); axC.set_aspect("equal")
panel_label(axC, "c")

for a_ in (axA, axC):
    a_.set_xticks([]); a_.set_yticks([])
    for sp in a_.spines.values():
        sp.set_visible(False)

# captions: model name (bold) + energy, below each column
fig.text(0.185, 0.115, "self-propelled Voronoi (2D)", ha="center", fontsize=10)
fig.text(0.185, 0.055, r"$U=\sum_i K_A(A_i-A_0)^2+K_P(P_i-P_0)^2$", ha="center", fontsize=8.5)
fig.text(0.515, 0.115, "mid-surface 3D vertex model", ha="center", fontsize=10)
fig.text(0.515, 0.055, r"$U=\sum_f \frac{1}{2}K_V(v-v^{eq})^2+K_A(A-A^0)^2+K_P(P-P^0)^2$", ha="center", fontsize=8)
fig.text(0.845, 0.115, "monolayer vertex model (apical/basal)", ha="center", fontsize=10)
fig.text(0.845, 0.055, r"$U=\sum_j \frac{1}{2}k_v(v-v^{eq})^2+\kappa_s s+\frac{1}{2}\gamma P^2$", ha="center", fontsize=8.5)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.16, wspace=0.05)
fig.savefig("model_schematic.png", dpi=150, facecolor="white")
print("wrote model_schematic.png")

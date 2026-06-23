"""Generate the per-kind operator glyphs used as logos in the operator library.

One small square icon per operator KIND (and one for sets and one for fields),
drawn in the exact visual language of Figure 1 (paper/fig_ops.tex): blue nodes,
red operator arrows, green fields, gray dashed neighbourhoods. Run:

    python scripts/make_op_icons.py

writes figures/icons/<kind>.png (transparent, square) + a contact sheet.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Arc

# --- palette lifted verbatim from paper/fig_ops.tex ----------------------- #
CSET = (31 / 255, 119 / 255, 180 / 255)     # nodes (sets)
CSET_FILL = (31 / 255, 119 / 255, 180 / 255, 0.10)
CFIELD = (44 / 255, 160 / 255, 44 / 255)    # fields
CFIELD_FILL = (44 / 255, 160 / 255, 44 / 255, 0.12)
COP = (214 / 255, 39 / 255, 40 / 255)       # operator arrows
CGRAY = (120 / 255, 120 / 255, 120 / 255)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "icons")
NR = 0.115                                   # node radius


def node(ax, x, y, r=NR, dashed=False, ghost=False):
    ax.add_patch(Circle((x, y), r, facecolor=(1, 1, 1, 0.04) if ghost else CSET_FILL,
                        edgecolor=CGRAY if ghost else CSET, lw=1.6,
                        linestyle=(0, (3, 2)) if dashed else "-", zorder=3))


def field(ax, x, y, w=0.20, h=0.20, grid=True):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=CFIELD_FILL,
                          edgecolor=CFIELD, lw=1.6, zorder=2))
    if grid:
        ax.plot([x - w / 2, x + w / 2], [y, y], color=CFIELD, lw=0.8, zorder=2.5)
        ax.plot([x, x], [y - h / 2, y + h / 2], color=CFIELD, lw=0.8, zorder=2.5)


def box(ax, x, y, w=0.46, h=0.20):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.0,rounding_size=0.03",
                               facecolor=CSET_FILL, edgecolor=CSET, lw=1.6, zorder=3))


def arrow(ax, p0, p1, color=COP, shrink=NR * 100 * 0.62, lw=2.2, dashed=False):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11,
                                color=color, lw=lw, shrinkA=shrink, shrinkB=shrink,
                                linestyle=(0, (3, 2)) if dashed else "-", zorder=4))


def _ax():
    fig, ax = plt.subplots(figsize=(1.7, 1.7), dpi=140)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def draw_lateral(ax):
    a, b, c = (0.30, 0.70), (0.72, 0.70), (0.50, 0.30)
    for p in (a, b, c): node(ax, *p)
    arrow(ax, a, b); arrow(ax, a, c); arrow(ax, b, c)


def draw_aggregate(ax):
    box(ax, 0.50, 0.74)
    c1, c2 = (0.32, 0.28), (0.68, 0.28)
    node(ax, *c1); node(ax, *c2)
    arrow(ax, c1, (0.42, 0.66)); arrow(ax, c2, (0.58, 0.66))


def draw_broadcast(ax):
    box(ax, 0.50, 0.74)
    c1, c2 = (0.32, 0.28), (0.68, 0.28)
    node(ax, *c1); node(ax, *c2)
    arrow(ax, (0.42, 0.66), c1); arrow(ax, (0.58, 0.66), c2)


def draw_exchange(ax):
    o1, o2 = (0.26, 0.70), (0.26, 0.30)
    node(ax, *o1); node(ax, *o2)
    field(ax, 0.70, 0.62, grid=False); field(ax, 0.70, 0.40, grid=False)
    arrow(ax, o1, (0.60, 0.62))                       # scatter S (red)
    arrow(ax, (0.60, 0.40), o2, color=CFIELD)         # gather  G (green)


def draw_field(ax):
    field(ax, 0.50, 0.50, w=0.46, h=0.46)
    ax.add_patch(Arc((0.50, 0.50), 0.62, 0.62, theta1=20, theta2=300,
                    color=COP, lw=2.0, zorder=5))
    arrow(ax, (0.78, 0.34), (0.80, 0.46), color=COP, shrink=0, lw=2.0)


def draw_rewire(ax):
    rc = (0.34, 0.50)
    ax.add_patch(Circle(rc, 0.34, fill=False, edgecolor=CGRAY, lw=1.4,
                        linestyle=(0, (3, 2)), zorder=1))
    n1, n2 = (0.60, 0.66), (0.58, 0.34)
    node(ax, *rc); node(ax, *n1); node(ax, *n2); node(ax, 0.88, 0.52, ghost=True)
    arrow(ax, rc, n1); arrow(ax, rc, n2)


def draw_structural(ax):
    m = (0.50, 0.72)
    node(ax, *m)
    ax.plot([0.50, 0.50], [0.72 - NR, 0.72 + NR], color=COP, lw=1.6,
            linestyle=(0, (2, 1.5)), zorder=5)            # the division plane
    c1, c2 = (0.32, 0.28), (0.68, 0.28)
    node(ax, *c1); node(ax, *c2)
    arrow(ax, m, c1); arrow(ax, m, c2)


def draw_set(ax):
    ax.add_patch(FancyBboxPatch((0.16, 0.20), 0.68, 0.60,
                               boxstyle="round,pad=0.0,rounding_size=0.06",
                               facecolor=(CSET[0], CSET[1], CSET[2], 0.05),
                               edgecolor=CSET, lw=1.4, linestyle=(0, (4, 2)), zorder=1))
    for p in [(0.36, 0.58), (0.64, 0.60), (0.50, 0.36)]:
        node(ax, *p, r=0.10)


ICONS = {
    "lateral": draw_lateral, "aggregate": draw_aggregate, "broadcast": draw_broadcast,
    "exchange": draw_exchange, "field": draw_field, "rewire": draw_rewire,
    "structural": draw_structural, "set": draw_set,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in ICONS.items():
        fig, ax = _ax(); fn(ax)
        fig.savefig(os.path.join(OUT, f"{name}.png"), transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
    # contact sheet for a quick visual check
    fig, axes = plt.subplots(2, 4, figsize=(8, 4), dpi=110)
    for ax, (name, fn) in zip(axes.ravel(), ICONS.items()):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        fn(ax); ax.set_title(name, fontsize=11)
    fig.savefig(os.path.join(OUT, "_contact_sheet.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", len(ICONS), "icons to", os.path.normpath(OUT))


if __name__ == "__main__":
    main()

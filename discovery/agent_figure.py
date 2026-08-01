#!/usr/bin/env python
"""agent_figure -- the round as a picture: who tells whom, and which links do not exist.

The static scan in `agent_graph.py` reads the source, and it cannot see everything: the Proposer
writes analysis.md with its TOOLS, not through Python `open()`, so no amount of parsing finds that
edge. This figure is therefore authored, and every edge in it was checked by hand against the code
during the Phase 5 diagnosis. It is a claim about the loop, and it is meant to be argued with.

WHY A RING. The agents are drawn in the order the round runs them, and the forward hand-offs go all
the way round the rim. What the ring makes visible is the inside: the arrows that carry a finding
BACK to the agent that chooses the next edits. There should be a bundle of them. There is one, and
it was added this week. The Supervisor's steer, the Interpreter's causal descriptions and
Evolution's refinement have no path back, so a round cannot be changed by what the round found ---
which is why eight simulations moved coverage by nothing.

    grey   A's output reaches B, and B uses it        red dashed  the link that should exist
    green  return path wired in Phase 5                           and does not: work evaporates
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage        # noqa: E402
from PIL import Image                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SHEET = os.path.join(HERE, "icons.png")

# The sheet as it actually reads, row by row. Getting this wrong is silent -- the figure simply
# shows the wrong face over the right name -- so it is written out in full rather than derived.
TILES = [["supervisor", "proposer", "peer_review", "critic"],
         ["duplicate", "analyst", "eye_check", "judge"],
         ["interpreter", "meta_review", "metrologist", "biologist"],
         ["cedric", "grounder", "evolution", "referee"]]

SPRITE_MIN_H = 100           # a character is ~170-250 px tall, a printed caption ~30. Nothing between.

# The ring, in the order the round runs. Two lines each: who it is, what it is for.
RING = [("grounder",    "Grounder",     "reads Okuda"),
        ("proposer",    "Proposer",     "chooses the edits"),
        ("peer_review", "Peer-review",  "checks the batch"),
        ("critic",      "Critic",       "refuses the illegal"),
        ("biologist",   "Biologist",    "is it a tissue?"),
        ("metrologist", "Metrologist",  "are metrics trustworthy?"),
        ("analyst",     "Analysts x3",  "read each run"),
        ("eye_check",   "Eye-check",    "watches the movie"),
        ("judge",       "Judge",        "settles disagreement"),
        ("referee",     "Referee",      "ranks the batch"),
        ("interpreter", "Interpreter",  "why it happened"),
        ("meta_review", "Meta-review",  "distils the round"),
        ("evolution",   "Evolution",    "refines the winner"),
        ("supervisor",  "Supervisor",   "holds the objective")]

# how: "flow" works | "wired" connected in Phase 5 | "missing" output goes nowhere
EDGES = [("grounder", "proposer", "flow", ""),
         ("proposer", "peer_review", "flow", ""),
         ("peer_review", "critic", "flow", ""),
         ("critic", "biologist", "flow", ""),
         ("biologist", "metrologist", "flow", ""),
         ("metrologist", "analyst", "flow", ""),
         ("analyst", "eye_check", "flow", ""),
         ("eye_check", "judge", "flow", ""),
         ("judge", "referee", "flow", ""),
         ("referee", "interpreter", "flow", ""),
         ("interpreter", "meta_review", "flow", ""),
         ("meta_review", "evolution", "flow", ""),
         ("evolution", "supervisor", "flow", ""),
         # the inside of the ring: everything that should come back to the Proposer
         ("critic", "proposer", "wired", "refusals, and why"),
         ("supervisor", "proposer", "missing", "the steer"),
         ("interpreter", "proposer", "missing", "causal descriptions"),
         ("evolution", "proposer", "missing", "the refinement"),
         ("eye_check", "supervisor", "missing", "what the movie showed")]

STYLE = {"flow": ("#5A5A5A", "solid", 1.7), "wired": ("#1B7F3B", "solid", 2.6),
         "missing": ("#B3261E", (0, (5, 3)), 2.4)}


def _runs(profile, thr=2, gap=1):
    """Contiguous stretches of ink along one axis, merged across gaps of `gap` px or less."""
    idx = np.where(profile > thr)[0]
    if idx.size == 0:
        return []
    cut = np.where(np.diff(idx) > gap)[0]
    return [(int(r[0]), int(r[-1]) + 1) for r in np.split(idx, cut + 1)]


def tiles():
    """Crop the 4x4 sheet into named sprites.

    NOT on equal quarters. The artwork is not laid out on them: a quarter cut both chops the
    bottom of a character and hands one cell its neighbour's emblem, and both mistakes are silent
    -- the figure simply shows a headless agent or somebody else's ruler. Two things are measured
    from the sheet instead.

    ROWS, PER COLUMN. Down any one column the ink separates into eight clean stretches, four
    characters (170-250 px) alternating with four printed names (~30 px). Nothing in the sheet
    falls between those heights, so the split needs no tuning. Measured per column and not
    globally, because globally a name in row 3 and a head in row 4 overlap and merge.

    COLUMNS, BY NEAREST CENTRE. Each character carries a small emblem to its left, and some of
    them cross the gutter -- the Metrologist's ruler sits inside the Meta-reviewer's quarter. So
    every blob of ink is assigned to the column whose centre it is nearest, which puts the ruler
    with the Metrologist because that is who it is nearer to. Geometry, not a boundary.
    """
    a = np.asarray(Image.open(SHEET).convert("RGB")).astype(np.int16)
    ink = ~((a.min(2) > 226) & ((a.max(2) - a.min(2)) < 16))
    H, W = ink.shape

    xs = np.where(ink.any(0))[0]
    lo, hi = int(xs.min()), int(xs.max()) + 1
    pitch = (hi - lo) / 4.0
    centres = [lo + pitch * (c + 0.5) for c in range(4)]

    out = {}
    for c in range(4):
        slab = ink[:, int(centres[c] - pitch * 0.35):int(centres[c] + pitch * 0.35)]
        bands = [(y0, y1) for y0, y1 in _runs(slab.sum(1)) if y1 - y0 >= SPRITE_MIN_H]
        if len(bands) != 4:
            raise SystemExit(f"column {c}: found {len(bands)} character bands, expected 4")
        for r, (y0, y1) in enumerate(bands):
            y0 = max(y0 - 12, 0)     # air ABOVE only: the emblem can sit higher than the head,
            #                          and anything added below picks up the caption underneath
            mine = [(bx0, bx1) for bx0, bx1 in _runs(ink[y0:y1].sum(0), gap=6)
                    if int(np.argmin([abs((bx0 + bx1) / 2 - m) for m in centres])) == c]
            if not mine:
                continue
            x0, x1 = max(min(b[0] for b in mine) - 2, 0), min(max(b[1] for b in mine) + 2, W)
            cell, cink = a[y0:y1, x0:x1].copy(), ink[y0:y1, x0:x1]
            ys = np.where(cink.any(1))[0]                        # trim the air back off the top
            cell, cink = cell[ys.min():ys.max() + 1], cink[ys.min():ys.max() + 1]
            cell[~cink] = 255                                    # knock the background to white
            out[TILES[r][c]] = Image.fromarray(cell.astype(np.uint8))
    return out


def _on_arc(p0, p2, rad, t):
    """A point on the quadratic Bezier matplotlib actually draws for connectionstyle=arc3."""
    (x0, y0), (x2, y2) = p0, p2
    cx = (x0 + x2) / 2 - rad * (y2 - y0)                          # control point: midpoint,
    cy = (y0 + y2) / 2 + rad * (x2 - x0)                          # displaced perpendicular
    u = 1 - t
    return (u * u * x0 + 2 * u * t * cx + t * t * x2,
            u * u * y0 + 2 * u * t * cy + t * t * y2)


def build(path=None):
    sprites = tiles()
    R, n = 4.0, len(RING)
    pos = {}
    for i, (key, _, _) in enumerate(RING):                       # clockwise from the top
        th = np.pi / 2 - 2 * np.pi * i / n
        pos[key] = (R * np.cos(th) * 1.28, R * np.sin(th))       # widened: labels need the room

    fig, ax = plt.subplots(figsize=(14.0, 10.2))
    fig.patch.set_facecolor("white")

    for a, b, how, lbl in EDGES:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        col, ls, lw = STYLE[how]
        ia, ib = [i for i, t in enumerate(RING) if t[0] == a][0], \
                 [i for i, t in enumerate(RING) if t[0] == b][0]
        rim = (ib - ia) % n == 1                                 # neighbours: hug the rim
        rad = -0.16 if rim else (0.24 if how == "wired" else 0.30)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.7", color=col,
                                    lw=lw, linestyle=ls, shrinkA=40, shrinkB=42,
                                    connectionstyle=f"arc3,rad={rad}"))
        if lbl:
            # ON THE ARC, not on the chord. Placing a label along the straight line between two
            # agents puts it where the arrow is not -- and on a ring, that is on top of whichever
            # sprite the arc bows away from. Every one of these labels landed on a face.
            mx, my = _on_arc((x0, y0), (x1, y1), rad, 0.42)
            mx, my = mx * 0.80, my * 0.80                        # ease inward, off the rim
            ax.text(mx, my, lbl, fontsize=11.5, color=col, ha="center", va="center", zorder=5,
                    style="italic" if how == "missing" else "normal",
                    bbox=dict(fc="white", ec="none", pad=1.4, alpha=0.92))

    for key, name, role in RING:
        x, y = pos[key]
        ax.add_artist(AnnotationBbox(OffsetImage(sprites[key], zoom=0.235), (x, y),
                                     frameon=False, zorder=3))
        # Always directly beneath its own sprite. Radial placement reads well on an empty ring and
        # badly on a full one -- at fourteen nodes a label ends up nearer its neighbour than itself.
        ax.text(x, y - 0.72, f"{name}\n{role}", fontsize=11.5, ha="center", va="top", zorder=4,
                linespacing=1.3, bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.9))

    miss = sum(1 for e in EDGES if e[2] == "missing")
    ax.text(0, -1.75, f"THE RETURN PATH\none arrow of five carries a finding back\n"
                     f"{miss} agents write into the void",
            fontsize=13, ha="center", va="center", color="#B3261E", linespacing=1.7,
            bbox=dict(fc="white", ec="#B3261E", lw=1.2, pad=7, alpha=0.96))

    for how, lab in (("flow", "output reaches the next agent, and it is used"),
                     ("wired", "return path wired in Phase 5"),
                     ("missing", "the link that should exist and does not — the finding evaporates")):
        col, ls, lw = STYLE[how]
        ax.plot([], [], color=col, ls=ls, lw=lw, label=lab)
    ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.012))

    ax.set_xlim(-6.55, 6.55)
    ax.set_ylim(-5.55, 4.95)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    path = path or os.path.join(HERE, "figures", "agent_figure.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    inbound = sum(1 for a, b, h, _ in EDGES if b == "proposer")
    print(f"  {n} agents, {len(EDGES)} edges, {miss} missing, {inbound} arrows into the Proposer")
    print(f"  -> {os.path.relpath(path, os.path.dirname(HERE))}")


if __name__ == "__main__":
    build()

#!/usr/bin/env python
"""The stratified tissue of exp 14 Phase 2, drawn off a run's trajectory: the basal shell seen from
outside, and an equatorial section through it -- the basal layer as a ring, its suprabasal cells
inside -- at chosen weeks on the progenitor clock, every cell in its clone's colour.

    PYTHONPATH=src:tools python tools/exp14_strata_figure.py tissue/exp14_p2_hazard2_s1 [--weeks 2 13 52]

WHY A FIGURE AND NOT THE MOVIE. The movie's renderer draws one set as its subject; the suprabasal
set is empty at frame 0 and its dot colours are fixed then, so it cannot show which clone a
suprabasal cell came from. The section is the view Clayton et al. 2007 (Fig. 3a) and Colom et al. 2020
(Fig. 3a) reason in: a clone is a patch of basal cells with a column of its differentiated daughters
above it -- here inside, towards the lumen of the organoid.

Colours are `plexus.measures.label_rgb` of the clone, the same rule the movie's faces and its Muller
panel use. In the section, progenitors (A, `fate` 0) are ringed in black; committed basal cells (B)
are not; suprabasal cells are the smaller dots. Written to the run folder as `strata.png`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def snapshot(T, r, supra_set="supra"):
    """(basal centroids, clone, fate, supra positions, supra clone, shell centre, mean radius) at row r."""
    from exp_measures.common import cells
    c = cells(T, r)
    X = np.asarray(c.block("centroid"), float)
    cl = np.rint(c.block("clone")[:, 0]).astype(np.int64)
    fa = c.block("fate")[:, 0]
    z = T.z
    occ = np.asarray(z[f"{supra_set}__occ"][r], bool)
    P = np.asarray(z[f"{supra_set}__pos"][r], float)[occ]
    pc = np.rint(np.asarray(z[f"{supra_set}__clone"][r], float)[occ][:, 0]).astype(np.int64)
    ctr = X.mean(0)
    R = float(np.linalg.norm(X - ctr, axis=1).mean())
    return X, cl, fa, P, pc, ctr, R


def draw(run, weeks=(2, 13, 52), out=None, slab=0.12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.measures import label_rgb
    from exp_measures.common import open_run, run_dir
    from exp_measures import exp14
    T = open_run(run)
    t, _nA, _nB = exp14.week_clock(T)
    weeks = [w for w in weeks if w <= t[-1] + 1e-9]
    fig = plt.figure(figsize=(4.2 * len(weeks), 8.4), facecolor="white")
    for j, w in enumerate(weeks):
        r = int(np.argmin(np.abs(t - w)))
        X, cl, fa, P, pc, ctr, R = snapshot(T, r)
        col = np.array([label_rgb(k) for k in cl])
        pcol = np.array([label_rgb(k) for k in pc]) if len(pc) else np.zeros((0, 3))
        # THE SHELL FROM OUTSIDE: the front hemisphere (+x), projected on (y, z), far cells smaller
        ax = fig.add_subplot(2, len(weeks), j + 1)
        d = X - ctr
        front = d[:, 0] > 0
        order = np.argsort(d[front, 0])
        ax.scatter(d[front, 1][order], d[front, 2][order], s=18 + 40 * d[front, 0][order] / R,
                   c=col[front][order], linewidths=0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"week {t[r]:.0f}: {len(np.unique(cl))} clones, {len(X)} basal cells", fontsize=11)
        # THE EQUATORIAL SECTION: every cell within `slab` x R of the plane x = centre
        ax = fig.add_subplot(2, len(weeks), len(weeks) + j + 1)
        sb = np.abs(d[:, 0]) < slab * R
        dp = P - ctr
        ss = np.abs(dp[:, 0]) < slab * R
        ax.scatter(dp[ss, 1], dp[ss, 2], s=14, c=pcol[ss], linewidths=0, alpha=0.9)
        a = sb & (fa < 0.5)
        ax.scatter(d[sb & ~a, 1], d[sb & ~a, 2], s=46, c=col[sb & ~a], marker="s", linewidths=0)
        ax.scatter(d[a, 1], d[a, 2], s=46, c=col[a], marker="s", edgecolors="black", linewidths=1.2)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"section: {int(sb.sum())} basal ({int(a.sum())} A, ringed), {int(ss.sum())} suprabasal",
                     fontsize=10)
    fig.suptitle(f"{run.split('/')[-1]}   clone colours; weeks on the progenitor clock", fontsize=11)
    fig.tight_layout()
    out = out or os.path.join(run_dir(run), "strata.png")
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weeks", nargs="+", type=float, default=[2, 13, 52])
    a = ap.parse_args()
    for r in a.runs:
        print(draw(r, a.weeks))


if __name__ == "__main__":
    main()

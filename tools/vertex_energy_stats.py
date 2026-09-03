#!/usr/bin/env python
"""Per-frame uniformity statistics for the `vertex_energy` sweep.

Answers, explicitly, "CV of WHAT":

  CV(l)     -- coefficient of variation (std/mean) of JUNCTION LENGTH, over the unordered
               vertex pairs of the live mesh. Each junction is counted ONCE: a closed mesh
               carries two half-edges per junction, so the raw half-edge list is folded on
               `(min(vi,vj), max(vi,vj))` first. This is the quantity the Marinari line terms
               `Lambda*l + (Gamma/2)*l^2` act on directly.

  CV(A)     -- std/mean of the CELL FACE AREA, over live cells, in the same length units as
               the world box. This is the quantity `(K/2)(A - A0)^2` acts on, and A0 is a
               SINGLE SCALAR shared by every cell (`mesh_seed` sets `A0 = mean(area)` at
               t = 0), so a nonzero CV(A) at the end is residual frustration, not a spread of
               per-cell targets.

  radius    -- mean distance of a cell centroid from the centroid of all cells, i.e. the
               spheroid's own radius, in world length units. It shrinks when the contractile
               line terms win against the volume constraint `K_V`.

  A/A0      -- mean cell area divided by that shared scalar target. 1.0 means the sheet sits
               at its preferred area; below 1.0 means the line tension has squeezed it.

Usage:  vertex_energy_stats.py <run-dir> [<run-dir> ...] [--frames 0,-1] [--csv out.csv]
"""
import argparse
import os
import sys

import numpy as np


def frame_stats(z, t):
    """The four numbers above, for one recorded frame of one run."""
    o0, o1 = z["vertex__mesh_offsets"][t], z["vertex__mesh_offsets"][t + 1]
    es = z["vertex__mesh_E_srce"][o0:o1]
    et = z["vertex__mesh_E_trgt"][o0:o1]
    Nv = int(z["vertex__mesh_Nv"][t])
    pos = z["vertex__pos"][t][:Nv].astype(np.float64)

    # FOLD THE HALF-EDGES ONTO JUNCTIONS. Without this every junction appears twice and the CV is
    # unchanged but the count is 2x -- harmless for a ratio, misleading in a table of counts.
    lo = np.minimum(es, et)
    hi = np.maximum(es, et)
    _, first = np.unique(lo.astype(np.int64) * Nv + hi.astype(np.int64), return_index=True)
    l = np.linalg.norm(pos[et[first]] - pos[es[first]], axis=1)

    occ = z["cell__occ"][t]
    area = z["cell__area"][t][occ, 0].astype(np.float64)
    cen = z["cell__cen"][t][occ].astype(np.float64)
    A0 = float(z["vertex__mesh_A0"][0])

    return dict(
        frame=t,
        cells=int(occ.sum()),
        junctions=int(l.size),
        radius=float(np.linalg.norm(cen - cen.mean(0), axis=1).mean()),
        mean_l=float(l.mean()),
        cv_l=float(l.std() / l.mean()),
        mean_A=float(area.mean()),
        cv_A=float(area.std() / area.mean()),
        A_over_A0=float(area.mean() / A0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--frames", default="0,-1",
                    help="comma-separated frame indices, negatives count from the end")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rows = []
    for run in a.runs:
        p = os.path.join(run, "trajectory.npz")
        if not os.path.isfile(p):
            print(f"  MISSING {p}", file=sys.stderr)
            continue
        z = np.load(p)
        nT = int(z["vertex__mesh_Nv"].shape[0])
        for f in a.frames.split(","):
            t = int(f)
            t = nT + t if t < 0 else t
            if not 0 <= t < nT:
                continue
            rows.append(dict(run=os.path.basename(run.rstrip("/")), **frame_stats(z, t)))

    hdr = ["run", "frame", "cells", "junctions", "radius", "mean_l", "cv_l", "mean_A", "cv_A",
           "A_over_A0"]
    w = max(len(r["run"]) for r in rows) if rows else 3
    print(f"{'run':<{w}} {'frame':>5} {'cells':>5} {'junc':>6} {'radius':>7} {'mean_l':>7} "
          f"{'CV(l)':>6} {'mean_A':>7} {'CV(A)':>6} {'A/A0':>6}")
    for r in rows:
        print(f"{r['run']:<{w}} {r['frame']:>5} {r['cells']:>5} {r['junctions']:>6} "
              f"{r['radius']:>7.3f} {r['mean_l']:>7.3f} {r['cv_l']:>6.3f} {r['mean_A']:>7.3f} "
              f"{r['cv_A']:>6.3f} {r['A_over_A0']:>6.3f}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, hdr)
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n  wrote {a.csv}")


if __name__ == "__main__":
    main()

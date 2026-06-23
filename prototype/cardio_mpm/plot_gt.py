#!/usr/bin/env python
"""plot_gt.py -- superpose the cardio_mpm fit target onto the canonical GT trajectory plot.

Reuses the SAME node selection + amplification as `cardio/cardio_real_render.py` (which made
`cardio/gt_trajectories.png`): a 10x10, margin-10 subset of the 137x137 control grid, displacement
amplified about rest. Draws:

  * GREEN  -- the FULL real sequence (reproduces gt_trajectories.png) from cardio_real.npz.
  * ORANGE -- the ONE BEAT the trainer fits (cardio_mpm_data: beat onset + period), overlaid, so you
              can see exactly which beat-loop the inverse model is matching, in the same coordinates.

Run:
  PYTHONPATH=../../src python plot_gt.py            # default amp 10, fit_beat -2 (same as the trainer)
  python plot_gt.py --amp 12 --fit_beat 2
Output -> archive/gt_compare.png
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cardio"))
import numpy as np

import cardio_mpm_data as D
from cardio_real_render import select_grid_nodes          # the exact selection used for gt_trajectories.png

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=10.0, help="displacement amplification (gt_trajectories uses 10)")
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--fit_beat", type=int, default=-2, help="which beat the trainer fits (onset index)")
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy.signal import find_peaks

    P = np.load(D.NPZ)["pos"].astype(np.float32)          # [F, 137^2, 2] in ~[0,1]  (cardio_mpm source)
    sel = select_grid_nodes(args.grid, args.margin)       # SAME 10x10 margin-10 nodes as gt_trajectories.png
    track = P[:, sel]                                     # [F, g*g, 2]
    rest = track[0]
    A = rest[None] + args.amp * (track - rest[None])      # amplified about rest (canonical recipe)

    # the beat the trainer fits (cardio_mpm_data idiom)
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    pk, _ = find_peaks(spd, height=spd.mean(), distance=20)
    period = int(round(float(np.diff(pk).mean())))
    fb = args.fit_beat % len(pk)                          # the FULL beat = onset -> NEXT onset (closes the loop)
    onset = int(pk[fb])
    end = int(pk[fb + 1]) + 1 if fb + 1 < len(pk) else min(onset + period, A.shape[0])
    beat = A[onset:end]                                   # full inter-beat interval -> closed loop

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black"); ax.set_facecolor("black"); ax.axis("off")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(1, 0)   # Y-down (image convention, matches gt render)

    def add(seq, color, lw):
        segs = np.stack([seq[:-1], seq[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        ax.add_collection(LineCollection(list(segs), colors=color, linewidths=lw))

    add(A, (0.2, 1.0, 0.2, 0.55), 1.2)                    # GREEN: full sequence (= gt_trajectories.png)
    add(beat, (1.0, 0.55, 0.1, 0.95), 1.6)               # ORANGE: the fit beat, on top
    ax.scatter(A[0, :, 0], A[0, :, 1], s=14, c="lime", edgecolors="black", linewidths=0.4, zorder=3)
    ax.set_title(f"GT (green, full {A.shape[0]}f) + fit beat (orange, onset {onset}, period {period})  amp x{args.amp:.0f}",
                 color="#ccc")
    out = os.path.join(HERE, "archive", "gt_compare.png"); os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}  (beats@{[int(p) for p in pk]} period={period} fit onset={onset})")


if __name__ == "__main__":
    main()

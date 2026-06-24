#!/usr/bin/env python
"""test_gt_align.py -- sanity check that an ATLAS sim run aligns with the real GT.

The atlas dashboard only plots the SIM loops (red) + a rest-point scatter (lime); it never
loads the real data. This test loads the real cardiomyocyte GT the SAME way the trainer does
(cardio_mpm_data.load_real -> mapped into the sheet domain, indexed per MPM particle) and
overlays it (GREEN) on the sim loops (RED), in the same coordinates and the same x10
amplification as archive/gt_compare.png, so you can confirm the GT is in the right frame /
scale before running the loop again.

  PYTHONPATH=../../src python test_gt_align.py                       # default run = mpm_b11_s0_base
  PYTHONPATH=../../src python test_gt_align.py mpm_b11_s2_amplitude25 --amp 10
Output -> archive/<run>/checkpoints/gt_align_test.png  (+ printed size/range diagnostics)
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import cardio_mpm_atlas as A
import cardio_mpm_data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="mpm_b11_s0_base", help="atlas archive dir name")
    ap.add_argument("--amp", type=float, default=10.0, help="displacement amplification (gt_compare uses 10)")
    ap.add_argument("--bwidth", type=float, default=0.06)
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    name = "material_atlas_" + args.run
    outdir = os.path.join(A.HERE, "archive", args.run)

    # --- SIM: the atlas forward run, one pacemaker beat, same node selection as morphology() ---
    m, rest, sim_disp, resid, idx = A.morphology(name)                       # sim_disp [Tb, N, 2] (domain coords)

    # --- REAL GT: mapped into the SAME sheet domain + indexed to the SAME MPM particles ---
    real_disp, bnd, onsets, period = D.load_real(rest, args.bwidth)          # real_disp [F, N, 2] (domain coords)
    fb = -2 % len(onsets)                                                     # the trainer's default fit beat (-2)
    onset = int(onsets[fb])
    end = int(onsets[fb + 1]) + 1 if fb + 1 < len(onsets) else min(onset + period, real_disp.shape[0])  # next onset closes the loop
    real_beat = real_disp[onset:end] - real_disp[onset]                      # one closed real beat [P, N, 2]

    # --- overlay: red = sim, green = real, both x amp about each node's rest position ---
    Rs = rest[idx][None] + args.amp * sim_disp[:, idx]                       # [Tb, n, 2]
    Rg = rest[idx][None] + args.amp * real_beat[:, idx]                      # [P,  n, 2]

    def segs(seq):
        return list(np.stack([seq[:-1], seq[1:]], 2).transpose(1, 0, 2, 3).reshape(-1, 2, 2))

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black"); ax.set_facecolor("black")
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_collection(LineCollection(segs(Rg), colors=(0.2, 1.0, 0.2, 0.55), linewidths=1.2))   # GREEN real
    ax.scatter(rest[idx, 0], rest[idx, 1], s=10, c="lime", edgecolors="black", linewidths=0.3, zorder=3)
    ax.set_xlim(0.1, 0.9); ax.set_ylim(0.9, 0.1)                              # Y-down (image convention, match gt_compare.png)

    # diagnostics: per-node max excursion (loop size) + spatial range, sim vs real
    sim_size = np.abs(sim_disp[:, idx]).max(axis=(0, 2)).mean()
    real_size = np.abs(real_beat[:, idx]).max(axis=(0, 2)).mean()
    ax.set_title(f"{args.run}: real GT GREEN  amp x{args.amp:.0f}  beat onset {onset} period {period}", color="#ccc", fontsize=9)

    ck = os.path.join(outdir, "checkpoints"); os.makedirs(ck, exist_ok=True)
    out = os.path.join(ck, "gt_align_test.png")
    fig.savefig(out, dpi=130, facecolor="black", bbox_inches="tight"); plt.close(fig)

    print(f"saved {out}")
    print(f"  real beats @ {onsets} period={period}  fit onset={onset}")
    print(f"  rest range      x[{rest[:,0].min():.3f},{rest[:,0].max():.3f}] y[{rest[:,1].min():.3f},{rest[:,1].max():.3f}]")
    print(f"  sim  disp range x[{sim_disp[...,0].min():.4f},{sim_disp[...,0].max():.4f}] y[{sim_disp[...,1].min():.4f},{sim_disp[...,1].max():.4f}]")
    print(f"  real disp range x[{real_beat[...,0].min():.4f},{real_beat[...,0].max():.4f}] y[{real_beat[...,1].min():.4f},{real_beat[...,1].max():.4f}]")
    print(f"  mean loop size  sim={sim_size:.4f}  real={real_size:.4f}  ratio={sim_size/max(real_size,1e-9):.2f}")


if __name__ == "__main__":
    main()

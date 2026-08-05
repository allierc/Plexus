#!/usr/bin/env python
"""montage -- the loops, ranked best to worst, with the verdict in the frame.

WHY RANKED, AND WHY THE BORDER CARRIES THE SCORE
================================================================================================
The inherited dashboard lays the loops out by POSITION on the tissue. That is the right picture
for asking *where* the model fails, and the wrong one for asking *how often* -- a reader has to
scan a hundred panels and hold a distribution in their head.

Ranked, the distribution is the picture. Sorted best to worst with the frame coloured green
through red, you see at a glance how much of the tissue is matched, where the cliff is, and --
the reason this exists -- **that the perfect panels are the ones tied to the recording.** On the
inherited grid the top 36 panels of the ranking are exactly the 36 pinned to the answer. They
appear as a green block at the top of an otherwise red picture, and the mean of the panel grid is
+0.17 higher for it.

Both grids are drawn so the difference is visible rather than argued: the inherited one with its
pinned panels marked, and a corrected one clear of the band.

    python montage.py --dump _replay/fs2.npz --out _replay/ranked.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def panel_scores(sim, real, idx):
    """Per-node loopscore, in the order `idx` is given."""
    import torch
    import harmonic_inherited as H
    s = torch.tensor(np.ascontiguousarray(sim[:, idx]), dtype=torch.float32)
    r = torch.tensor(np.ascontiguousarray(real[:, idx]), dtype=torch.float32)
    return H._pernode_score(s, r, None).numpy()


def grid_indices(rest, margin):
    """The 10x10 selection, mapped to the nearest particles."""
    from scipy.spatial import cKDTree
    import data as D
    import metrics as M
    P = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)["pos"].astype(np.float64)
    Pm = D.DOM_LO + D.DOM * P
    nodes = M.select_grid_nodes(margin=margin)
    return cKDTree(rest).query(Pm[0][nodes])[1]


def draw(dump, out, cols=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    import metrics as M

    z = np.load(dump)
    sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
    rest, bnd = z["rest"].astype(np.float64), z["bnd"].astype(bool)

    panels = []
    for label, margin in (("as the campaign read it (margin 10)", M.MARGIN_INHERITED),
                          ("clear of the pinned band (margin 20)", M.MARGIN_SAFE)):
        idx = grid_indices(rest, margin)
        sc = panel_scores(sim, real, idx)
        panels.append((label, idx, sc, bnd[idx]))

    rows_per = int(np.ceil(len(panels[0][1]) / cols))
    fig = plt.figure(figsize=(cols * 1.32, len(panels) * (rows_per * 1.32 + 1.0)), facecolor="black")
    # green at the top of the scale, red at the bottom; the scale is FIXED across both grids so
    # the two are comparable by eye
    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["#B3261E", "#B26B00", "#1B7F3B"])
    norm = mcolors.Normalize(vmin=-0.3, vmax=1.0)

    for b, (label, idx, sc, pinned) in enumerate(panels):
        order = np.argsort(-sc)
        base = b * (rows_per + 1)
        ax0 = fig.add_subplot(len(panels) * (rows_per + 1), 1, base + 1)
        ax0.axis("off")
        n_pin = int(pinned.sum())
        ax0.text(0.0, 0.25,
                 f"{label}     mean {sc.mean():+.3f}"
                 + (f"     {n_pin} of {len(sc)} panels are PINNED to the recording "
                    f"(mean without them {sc[~pinned].mean():+.3f})" if n_pin else
                    "     no panel is pinned"),
                 color="white", fontsize=11, fontweight="bold", transform=ax0.transAxes)
        for k, j in enumerate(order):
            ax = fig.add_subplot(len(panels) * (rows_per + 1), cols,
                                 (base + 1) * cols + k + 1)
            p = real[:, idx[j]]; q = sim[:, idx[j]]
            c = p.mean(0)
            ax.plot(p[:, 0] - c[0], p[:, 1] - c[1], color="#22DD22", lw=1.1)
            ax.plot(q[:, 0] - c[0], q[:, 1] - c[1], color="#FF3B30", lw=1.1)
            r = max(np.abs(np.concatenate([p - c, q - c])).max(), 1e-12) * 1.15
            ax.set_xlim(-r, r); ax.set_ylim(-r, r)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_color(cmap(norm(sc[j]))); sp.set_linewidth(3.0)
            ax.text(0.03, 0.86, f"{sc[j]:+.2f}", color="white", fontsize=6.5,
                    fontweight="bold", transform=ax.transAxes)
            if pinned[j]:
                ax.text(0.62, 0.86, "PINNED", color="#66CCFF", fontsize=5.5,
                        fontweight="bold", transform=ax.transAxes)

    fig.suptitle("Every loop of the best fit in the archive, ranked best to worst.  "
                 "green = the recording, red = the model, frame = its score.\n"
                 "The block of near-perfect panels at the top of the upper grid is the ring tied "
                 "to the recording -- it is scored, and it is not a result.",
                 color="white", fontsize=11, y=0.995)
    fig.subplots_adjust(hspace=0.35, wspace=0.12, top=0.955, bottom=0.01, left=0.01, right=0.99)
    fig.savefig(out, dpi=125, facecolor="black")
    plt.close(fig)
    return out, panels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "_replay", "fs2.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "_replay", "ranked.png"))
    a = ap.parse_args(argv)
    out, panels = draw(a.dump, a.out)
    print(f"[montage] {out}")
    for label, idx, sc, pinned in panels:
        q = np.percentile(sc, [10, 50, 90])
        print(f"  {label:<38s} mean {sc.mean():+.3f}  p10 {q[0]:+.3f}  median {q[1]:+.3f}  "
              f"p90 {q[2]:+.3f}  pinned {int(pinned.sum())}  above +0.9: {(sc > 0.9).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

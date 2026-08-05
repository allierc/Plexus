#!/usr/bin/env python
"""grid_plot -- the loops where they actually sit on the tissue, ten by ten.

TWO WAYS TO LAY OUT THE SAME HUNDRED PANELS, AND THEY ANSWER DIFFERENT QUESTIONS
================================================================================================
`montage.py` sorts them best to worst, which shows the DISTRIBUTION -- how much of the sheet is
matched and where the cliff is. This one puts each panel at the position of the node it came from,
which is the prototype's dashboard layout and shows the MAP -- whether the failures are scattered
or concentrated, and whether they line up with anything (an edge, a fibre direction, a stiff patch).
A sorted plot cannot answer that and a mapped plot cannot answer the other, so both exist.

Green is the recording, red is the model, and every panel is centred on its own loop and scaled to
its own extent -- so a panel says whether the SHAPE matches, never whether the size does. Size is
`peak_excursion`'s job and it is in the header.

The margin is 20, not the inherited 10: the outer ring of the margin-10 grid sits on particles tied
to the recording, which score a perfect 1.000 and are not a result.

    python grid_plot.py --dump _replay/fs2.npz --out figures/grid_fs2.png
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

import metrics as M                                                 # noqa: E402


def grid_indices(rest, margin=None):
    """The 10x10 selection mapped to particles, plus each node's (row, col) on the sheet."""
    from scipy.spatial import cKDTree
    import data as D
    margin = M.MARGIN_SAFE if margin is None else margin
    P = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)["pos"].astype(np.float64)
    Pm = D.DOM_LO + D.DOM * P
    nodes = M.select_grid_nodes(margin=margin)
    idx = cKDTree(rest).query(Pm[0][nodes])[1]
    n = int(np.sqrt(len(idx)))
    rc = [(k // n, k % n) for k in range(len(idx))]
    return idx, rc, n


def per_node_score(sim, real, idx):
    import torch
    import harmonic_inherited as H
    t = lambda a: torch.tensor(np.ascontiguousarray(a[:, idx]), dtype=torch.float32)
    return H._pernode_score(t(sim), t(real), None).numpy()


def draw(dump, out, margin=None, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    z = np.load(dump)
    sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
    rest, bnd = z["rest"].astype(np.float64), z["bnd"].astype(bool)
    idx, rc, n = grid_indices(rest, margin)
    sc = per_node_score(sim, real, idx)
    pinned = bnd[idx]

    mask = np.zeros(rest.shape[0], bool); mask[idx] = True
    head = []
    for name in ("openness", "peak_excursion", "path_length", "chirality_match",
                 "orientation_error", "coordination"):
        try:
            head.append(f"{name} {M.REGISTRY[name](sim, real, mask):.4f}")
        except Exception as e:
            head.append(f"{name} {type(e).__name__}")

    cmap = mcolors.LinearSegmentedColormap.from_list("gr", ["#B3261E", "#B26B00", "#1B7F3B"])
    norm = mcolors.Normalize(vmin=-0.3, vmax=1.0)

    fig, axes = plt.subplots(n, n, figsize=(n * 1.28, n * 1.28 + 1.1), facecolor="black")
    for k, (r, c) in enumerate(rc):
        ax = axes[r, c]
        p, q = real[:, idx[k]], sim[:, idx[k]]
        ctr = p.mean(0)
        ax.plot(p[:, 0] - ctr[0], p[:, 1] - ctr[1], color="#22DD22", lw=1.1)
        ax.plot(q[:, 0] - ctr[0], q[:, 1] - ctr[1], color="#FF3B30", lw=1.1)
        rad = max(np.abs(np.concatenate([p - ctr, q - ctr])).max(), 1e-12) * 1.15
        ax.set_xlim(-rad, rad); ax.set_ylim(-rad, rad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("black")
        for sp in ax.spines.values():
            sp.set_color(cmap(norm(sc[k]))); sp.set_linewidth(2.6)
        ax.text(0.03, 0.85, f"{sc[k]:+.2f}", color="white", fontsize=6, fontweight="bold",
                transform=ax.transAxes)
        if pinned[k]:
            ax.text(0.60, 0.85, "PINNED", color="#66CCFF", fontsize=5, fontweight="bold",
                    transform=ax.transAxes)

    m = M.MARGIN_SAFE if margin is None else margin
    fig.suptitle(
        (title or os.path.basename(dump)) +
        f"    the loops where they sit on the tissue, margin {m}"
        f"{'  (PINNED ring included -- not a result)' if pinned.any() else ''}\n"
        f"green = the recording, red = the model, frame = its loopscore    "
        f"mean {sc.mean():+.3f}   median {np.median(sc):+.3f}\n" + "    ".join(head),
        color="white", fontsize=10, y=0.995)
    fig.subplots_adjust(hspace=0.06, wspace=0.06, top=0.90, bottom=0.01, left=0.01, right=0.99)
    fig.savefig(out, dpi=115, facecolor="black")
    plt.close(fig)
    return out, sc, pinned


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "_replay", "fs2.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "grid_loops.png"))
    ap.add_argument("--margin", type=int, default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out, sc, pinned = draw(a.dump, a.out, a.margin, a.title)
    print(f"[grid_plot] {out}")
    print(f"  mean {sc.mean():+.3f}  median {np.median(sc):+.3f}  "
          f"p10 {np.percentile(sc, 10):+.3f}  p90 {np.percentile(sc, 90):+.3f}  "
          f"pinned {int(pinned.sum())}/{len(sc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

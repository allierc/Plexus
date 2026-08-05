#!/usr/bin/env python
"""figure_metrics -- what each measurement responds to, and how finely, drawn as loops.

The figure Phase 2 reports into. Two panels:

  TOP     a gallery of distortions. One reference loop in green, the distorted one in red. This is
          the battery, drawn: each column is a single, named change to a loop.
  BOTTOM  the response matrix. One row per measurement, one column per distortion, and the cell
          says whether it MOVED or HELD and by how much -- with the colour saying whether that was
          correct. A measurement is only useful if it moves on its own axis and ignores the others,
          and this is that claim as a picture rather than a paragraph.

  RIGHT   the precision. What the same measurement reads comparing one real beat with another --
          the tissue's own variation -- and what the best fit in the archive reads. A change
          smaller than the first column is not a change.

    python figure_metrics.py            # -> figures/metrics_figure.png, used by the note
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import metrics as M                                                 # noqa: E402

# the ones a person can read directly; the residual/* family is a decomposition and is reported
# in the run record rather than here
SHOW = ["loopscore", "interior_r2", "openness", "peak_excursion", "path_length",
        "chirality_match", "orientation_error", "coordination"]
SHORT = {"loopscore": "loopscore", "interior_r2": "interior $R^2$", "openness": "openness",
         "peak_excursion": "size (peak)", "path_length": "path length",
         "chirality_match": "chirality", "orientation_error": "orientation err",
         "coordination": "coordination"}


def one_loop(G=64, a=1.0, b=0.45, th=0.6):
    t = np.linspace(0, 2 * np.pi, G, endpoint=False)
    u, v = np.cos(t) * a, np.sin(t) * b
    return np.stack([u * np.cos(th) - v * np.sin(th), u * np.sin(th) + v * np.cos(th)], -1)


def gallery():
    """One cell per distortion, in the order the battery reports them.

    A cell is (label, items, axis, curve_unchanged) where items is a list of
    (offset, reference, distorted) -- more than one when the distortion is only visible across
    nodes. `curve_unchanged` says the two curves lie on top of each other, so the red is dashed and
    the only visible difference is the start dot: that is how the two TIMING distortions, which no
    static picture of a shape can show, become visible at all.
    """
    p = one_loop()
    G = p.shape[0]
    rot = np.pi / 5
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    mir = p.copy(); mir[:, 1] *= -1
    n = np.array([np.cos(0.6), np.sin(0.6)])
    z = np.array([0.0, 0.0])
    L, Rt = np.array([-1.05, 0.0]), np.array([1.05, 0.0])
    s = 0.52
    return [
        ("nothing changed",         [(z, p, p.copy())],                  "identity",      True),
        ("twice the size",          [(z, p, p * 1.9)],                   "size",          False),
        ("turned by 36 degrees",    [(z, p, p @ R.T)],                   "orientation",   False),
        ("mirrored",                [(z, p, mir)],                       "chirality",     False),
        ("flattened to a line",     [(z, p, np.outer(p @ n, n))],        "openness",      False),
        ("whole beat starts later", [(z, p, np.roll(p, G // 6, 0))],     "phase",         True),
        ("moved elsewhere",         [(z, p, p + np.array([1.2, -0.95]))], "placement",    False),
        ("half the tissue doubled", [(L, p * s, p * s),
                                     (Rt, p * s, p * s * 1.9)],          "heterogeneity", False),
        ("each node's own timing",  [(L, p * s, np.roll(p * s, G // 3, 0)),
                                     (Rt, p * s, np.roll(p * s, -G // 4, 0))],
                                                                         "coordination",  True),
    ]


def draw(out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    cert = json.load(open(os.path.join(HERE, "_metrology", "metrics_certify.json")))
    beats = json.load(open(os.path.join(HERE, "_metrology", "noise_beats.json")))
    rows = cert["rows"]
    dists = [r["distortion"] for r in rows]
    gal = gallery()
    labels = [g[0] for g in gal]
    axes = [g[2] for g in gal]
    # the drawing is a claim about the battery, so it may not drift from it
    if len(gal) != len(rows):
        raise SystemExit(f"gallery has {len(gal)} cells, the battery ran {len(rows)} distortions")

    # the best fit in the archive, read on the corrected grid, for the right-hand column
    model = {}
    try:
        z = np.load(os.path.join(HERE, "_replay", "fs2.npz"))
        sim, real = z["sim_d"].astype(np.float64), z["real_d"].astype(np.float64)
        rest = z["rest"].astype(np.float64)
        from scipy.spatial import cKDTree
        import data as D
        P = D.open_npz(expect_sha256=D.HEALTHY_POS_SHA256)["pos"].astype(np.float64)
        Pm = D.DOM_LO + D.DOM * P
        idx = cKDTree(rest).query(Pm[0][M.select_grid_nodes(margin=M.MARGIN_SAFE)])[1]
        msk = np.zeros(rest.shape[0], bool); msk[idx] = True
        for n in SHOW:
            model[n] = M.REGISTRY[n](sim, real, msk)
    except Exception:
        pass

    nc, nr = len(dists), len(SHOW)
    W_RIGHT = 3.1                       # the precision block, in units of one matrix column
    fig = plt.figure(figsize=(1.30 * (nc + W_RIGHT) + 1.7, 1.55 + 0.46 * nr), facecolor="white")
    # one gridspec for everything, so the gallery cell and the matrix column below it are the same
    # column by construction rather than by fiddling
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 0.46 * nr], width_ratios=[nc, W_RIGHT],
                          hspace=0.44, wspace=0.035,
                          left=0.098, right=0.992, top=0.835, bottom=0.018)

    # ---- TOP: the gallery -------------------------------------------------------------------
    top = gs[0, 0].subgridspec(1, nc, wspace=0.10)
    for k, (lab, items, axis, unchanged) in enumerate(gal):
        ax = fig.add_subplot(top[0, k])
        for off, ref, dis in items:
            a, b = ref + off, dis + off
            ax.plot(a[:, 0], a[:, 1], color="#1B7F3B", lw=1.6)
            ax.plot(b[:, 0], b[:, 1], color="#B3261E", lw=1.6,
                    ls=(0, (2.2, 1.7)) if unchanged else "-")
            ax.plot(a[0, 0], a[0, 1], "o", color="#1B7F3B", ms=4.0, mec="white", mew=0.7, zorder=5)
            ax.plot(b[0, 0], b[0, 1], "o", color="#B3261E", ms=4.0, mec="white", mew=0.7, zorder=5)
        r = 2.1
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#CCCCCC")
        ax.set_title("\n".join(textwrap.wrap(lab, 16)), fontsize=7.6, pad=4, linespacing=1.3)
        ax.text(0.5, 0.025, axis, ha="center", fontsize=6.8, color="#2B4C7E", style="italic",
                transform=ax.transAxes)
    fig.text(0.098, 0.917,
             "green = the reference loop      red = one thing changed      dot = the first frame "
             "of the beat, which is the only way the two changes of timing are visible at all",
             fontsize=7.9, color="#444444")

    # ---- BOTTOM: the response matrix --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.invert_yaxis()
    # no column labels: the gallery cell sits directly above its own column and names it
    ax.set_xticks([])
    ax.set_yticks(np.arange(nr) + 0.5)
    ax.set_yticklabels([SHORT[n] for n in SHOW], fontsize=8.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    C_MOVE, C_HOLD, C_BAD, C_NA = "#2B4C7E", "#EDEDED", "#B3261E", "#FAFAFA"
    for i, n in enumerate(SHOW):
        for j, r in enumerate(rows):
            a = r["metrics"].get(n)
            if a is None or "error" in a or a.get("skipped"):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=C_NA, edgecolor="white"))
                ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=6.5,
                        color="#999999")
                continue
            ok, moved = a["ok"], a["moved"]
            fc = (C_MOVE if moved else C_HOLD) if ok else C_BAD
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white"))
            g = a["rel_change"] * 100
            txt = ("--" if not moved else f"{g:.0f}%" if g < 300 else ">300%")
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7.0,
                    color="white" if (moved and ok) else ("white" if not ok else "#666666"),
                    fontweight="bold" if moved else "normal")
    ax.set_title("what it responds to:   blue = moved, as it should   grey = ignored, as it "
                 "should   red = wrong   (per cent = how far it moved)",
                 fontsize=8.2, pad=25, loc="left", color="#222222")

    # ---- RIGHT: the precision ---------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_xlim(0, 3); ax2.set_ylim(0, nr); ax2.invert_yaxis()
    ax2.set_yticks([]); ax2.set_xticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    heads = ["one real beat\nvs another", "its spread\n= the precision", "the best fit\nin the "
             "archive"]
    for j, h in enumerate(heads):
        ax2.text(j + 0.5, -0.06, h, ha="center", va="bottom", fontsize=7.1, color="#1B3A6B",
                 linespacing=1.25)
    for i, n in enumerate(SHOW):
        b = beats["metrics"].get(n, {})
        mval, sd, fit = b.get("median"), b.get("sd"), model.get(n)
        # is the model's reading outside the tissue's own variation? that is the only comparison
        # that licenses the word "worse", and it is what the third column is for
        far = (mval is not None and sd is not None and fit is not None
               and abs(fit - mval) > 3.0 * max(sd, 1e-12))
        for j, val in enumerate([mval, sd, fit]):
            fc = "#FCEBEA" if (j == 2 and far) else "#F6F8FB"
            ax2.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="white"))
            if val is None:
                ax2.text(j + 0.5, i + 0.5, "not read", ha="center", va="center", fontsize=6.5,
                         color="#AAAAAA")
                continue
            ax2.text(j + 0.5, i + 0.5, f"{val:.4f}" if abs(val) < 10 else f"{val:.2e}",
                     ha="center", va="center", fontsize=7.3,
                     color="#8C1D18" if (j == 2 and far) else "#1B3A6B",
                     fontweight="bold" if j == 1 else "normal")
    ax2.set_title("how finely it reads   (pink = the fit is further out than three spreads)",
                  fontsize=8.2, pad=25, loc="left", color="#222222")

    fig.suptitle("What each measurement responds to, and how finely it reads", fontsize=12.5,
                 fontweight="bold", y=0.975)
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "metrics_figure.png"))
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[figure] {draw(a.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

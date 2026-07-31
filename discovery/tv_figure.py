#!/usr/bin/env python
"""tv_figure -- the Turing x vertex finding as one figure.

Top two rows: the SAME Turing pattern on the SAME recipe, capped vs nearly uncapped. Bottom:
the lever curve that says why there is no clean window -- coupling and mesh integrity rise and
fail together.

House style (Plexus/SMG2 figures): black background, no panel titles, white labels top-left.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "_turing_vertex")
FS = 11


def _strip_row(ax, png, label, rows=None):
    """Draw the FIRST row of a saved strip.png (the side-on 3D view).

    Do not assume the strip has two rows. It had two (3D + cross-section) when this was written;
    the renderer now emits three (side / top-down / cross-section), so a hard-coded `//2` takes
    one and a half rows and silently produces a wrong figure -- no exception, just a clipped
    near-polar row colliding with the next label.

    A verifier raised this before the renderer changed, I checked it, and it was a FALSE ALARM at
    the time -- the study strips really were still two rows. It became true when the renderer was
    fixed hours later. The lesson is not "the verifier was right"; it is that a layout constant
    copied from a file you do not own is a time bomb with someone else's finger on it. So the row
    count is now DERIVED from the image's aspect ratio rather than assumed, and stated in the
    docstring rather than a trailing comment.
    """
    im = plt.imread(png)
    h, w = im.shape[0], im.shape[1]
    if rows is None:
        # panels are ~square and laid out n_cols wide; rows = round(h / (w / n_cols)) is not
        # knowable without n_cols, so use the recorded aspect: 2-row strips are ~1.96:1 wide,
        # 3-row ~1.30:1. Bucket on that rather than guess.
        rows = 2 if (w / max(h, 1)) > 1.6 else 3
    ax.imshow(im[: h // rows])                       # the FIRST row, whatever the layout
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.006, 0.97, label, transform=ax.transAxes, color="white", fontsize=FS,
            va="top", ha="left", fontweight="bold")


def main():
    rho, corr, hollow, protr = [], [], [], []
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f, "diag.json")
        if not f.startswith("waveE_") or not os.path.exists(p):
            continue
        d = json.load(open(p))
        if not d.get("ok"):
            continue
        rho.append(d["knobs"]["rho"]); corr.append(d.get("corr_act_rad") or np.nan)
        hollow.append(d["hollow_frac"]); protr.append(d["protr"])
    o = np.argsort(rho)[::-1]
    rho = np.array(rho)[o]; corr = np.array(corr)[o]
    hollow = np.array(hollow)[o]; protr = np.array(protr)[o]

    fig = plt.figure(figsize=(13.0, 9.2))
    fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.06,
                          left=0.055, right=0.985, top=0.985, bottom=0.075)

    _strip_row(fig.add_subplot(gs[0]), os.path.join(OUT, "waveE_0_1", "strip.png"),
               "a   rho = 1.0  (capped)   corr +0.29   the pattern is painted on")
    _strip_row(fig.add_subplot(gs[1]), os.path.join(OUT, "waveE_3_0p05", "strip.png"),
               "b   rho = 0.05 (near-uncapped)   corr +0.63   the shell clefts along the bands")

    ax = fig.add_subplot(gs[2])
    ax.set_facecolor("black")
    x = np.arange(len(rho))
    # red / blue: two distinct-source traces (house convention; green/black is reserved for
    # ground-truth vs predicted, which this is not)
    ax.plot(x, corr, "o-", color="#e03030", lw=2.2, ms=8, label="corr(activator, radius)")
    ax.plot(x, hollow, "s--", color="#4a90d9", lw=2.2, ms=8, label="hollow_frac (folded faces)")
    # Shade the REGIME (an x-span over rho), not a y-band. A horizontal band at y>0.5 reads as
    # "coupling above 0.5 means the mesh is destroyed", which is not what was measured: at
    # rho=0.05 corr is 0.63 with the mesh merely straining. The failure is a property of the
    # setting, not of the correlation value.
    for lo, hi, c, al, txt in ((2.5, 3.5, "#d9a441", 0.13, "straining"),
                               (3.5, 4.4, "#e03030", 0.16, "mesh destroyed")):
        ax.axvspan(lo, hi, color=c, alpha=al)
        ax.text((lo + hi) / 2, 1.0, txt, color="#ffb0b0" if "destroy" in txt else "#f0cf90",
                fontsize=FS - 1, ha="center", va="top")
    ax.set_xticks(x); ax.set_xticklabels([f"{r:g}" for r in rho], color="white", fontsize=FS)
    ax.set_xlabel("rho   (uniform growth floor;  0 = activator-only, cap removed)",
                  color="white", fontsize=FS)
    ax.set_ylim(-0.03, 1.05)
    ax.tick_params(colors="white", labelsize=FS - 1)
    for s in ax.spines.values():
        s.set_color("white")
    for i, p in enumerate(protr):
        ax.annotate(f"protr {p:.2f}" if p < 10 else f"protr {p:.0f}", (x[i], max(corr[i], hollow[i])),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    color="white", fontsize=FS - 2)
    lg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), facecolor="black",
                   edgecolor="white", fontsize=FS - 1, framealpha=0.0)
    for t in lg.get_texts():
        t.set_color("white")
    ax.text(0.006, 0.97, "c   coupling and mesh integrity fail together — no clean window",
            transform=ax.transAxes, color="white", fontsize=FS, va="top", fontweight="bold")

    p = os.path.join(OUT, "turing_vertex_lever.png")
    fig.savefig(p, dpi=135, facecolor="black")
    plt.close(fig)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

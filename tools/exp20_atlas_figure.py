"""exp20: GLUCOSE FISH 1 IN THE Z-BRAIN ATLAS (BigWarp landmarks, tools/exp20_bigwarp.py apply): the atlas reference from
above (maximum over depth, head up) with fish 1's cells carried into it -- every cell faint, the gut-responsive cells
yellow, the cells in the area postrema and the vagal ganglia red -- and beside it the gut-responsive cells per region.

    PYTHONPATH=src:tools python tools/exp20_atlas_figure.py
Writes presentation/figs/atlas_f1.png (and a copy in png/).
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    z = np.load(os.path.join(DATA, "atlas", "fish1_regions.npz"), allow_pickle=True)
    A, reg, names = z["atlas_um"], z["regions"], [str(n) for n in z["names"]]
    resp = np.load(os.path.join(DATA, "baselines_gutbrain_glucose_f1_cells.npz"))["responsive"]
    F = tifffile.imread(os.path.join(DATA, "atlas", "bigwarp", "zbrain_reference.tif"))     # (z, y, x), head up
    S = json.load(open(os.path.join(DATA, "atlas", "fish1_regions_summary.json")))
    entry = reg[:, [i for i, n in enumerate(names) if "Area Postrema" in n or "Ganglia - Vagal Ganglia" in n]].any(1)
    fig = plt.figure(figsize=(14, 8.4), facecolor="black")
    for j, (title, view) in enumerate((("from above (max over depth)", 0), ("from the side (max over left-right)", 2))):
        ax = fig.add_axes([0.0 + 0.25 * j, 0.04, 0.25 if j == 0 else 0.17, 0.88])
        ax.set_facecolor("black"); ax.axis("off")
        if view == 0:
            img, ext, xy = F.max(0), [0, F.shape[2] * 0.798, F.shape[1] * 0.798, 0], (A[:, 0], A[:, 1])
        else:
            img, ext, xy = F.max(2).T, [0, F.shape[0] * 2.0, F.shape[1] * 0.798, 0], (A[:, 2], A[:, 1])
        ax.imshow(img, cmap="gray", extent=ext, vmax=2.2 * np.percentile(img, 99.5))     # dimmed: the cells show on it
        ax.scatter(xy[0][::8], xy[1][::8], s=0.5, c="#29b6f6", alpha=0.6, lw=0)
        ax.scatter(xy[0][resp], xy[1][resp], s=0.8, c="#ffeb3b", lw=0)
        ax.scatter(xy[0][entry], xy[1][entry], s=3.0, c="#ff4040", lw=0)
        if j == 0:                                          # the legend (Cedric, 2026-10-04: "what are yellow vs red")
            from matplotlib.lines import Line2D
            mk = lambda c: Line2D([], [], marker="o", ls="", color=c, markersize=7)   # noqa: E731
            from matplotlib.patches import Patch
            ax.legend([mk("#ffeb3b"), mk("#ff4040"), mk("#29b6f6"), Patch(color="0.45")],
                      [f"gut-responsive ({int(resp.sum()):,}, the paper's rule)",
                       f"area postrema + vagal ganglia ({int(entry.sum()):,}, atlas)",
                       "fish 1's other cells (1 in 8)", "grey: the Z-Brain reference brain"],
                      loc="lower left", bbox_to_anchor=(0.0, -0.02), frameon=True, facecolor="black", edgecolor="0.5",
                      labelcolor="white", fontsize=8)
        ax.set_title(f"Z-Brain {title}", color="white", fontsize=10)
        ax.set_aspect("equal")
    ax = fig.add_axes([0.63, 0.08, 0.35, 0.84])
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("0.6")
    R = S["regions"]
    keys = list(R.keys())
    v = [R[k]["frac_responsive"] * 100 for k in keys]
    ax.barh(range(len(keys)), v, color=["#ff4040" if k.startswith(("area postrema", "vagal")) else "0.6" for k in keys])
    for i, k in enumerate(keys):
        ax.text(v[i] + 1, i, f"{R[k]['gut_responsive']:,} of {R[k]['cells']:,} cells", color="white", fontsize=9, va="center")
    ax.set_yticks(range(len(keys)), keys, color="0.9", fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(colors="0.8")
    ax.set_xlabel("% of the region's cells that are gut-responsive", color="0.9")
    ax.set_xlim(0, 115)
    ax.set_title(f"glucose fish 1 in Z-Brain: 30 BigWarp landmarks,\nleave-one-out error {S['loo_um_median']:.0f} um "
                 f"(max {S['loo_um_max']:.0f})", color="white", fontsize=10)
    fig.text(0.63, 0.005, "bars: red the gut's entry regions (area postrema, vagal ganglia), grey the others", color="0.8",
             fontsize=9)
    out = os.path.join(EXP, "presentation", "figs", "atlas_f1.png")
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    print("[atlas]", out)


if __name__ == "__main__":
    main()

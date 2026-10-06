"""exp20: GLUCOSE FISH 1'S INPUT CELLS IN THE Z-BRAIN ATLAS (BigWarp landmarks, tools/exp20_bigwarp.py apply), on the
atlas reference (grey, maximum over depth or over left-right). Cedric, 2026-10-05: no bar column; left the gut-input
cells, right the visual input cells, each from above and from the side at one scale, drawn from bottom to top

    gut      fish 1's other cells; the area postrema + vagal ganglia (atlas); + the dorsal vagal complex (atlas: the X
             vagus motor nucleus and the noradrenergic vagal area, the cells beyond AP + vagal ganglia, so the layer
             under stays visible); the gut-responsive cells (the paper's rule, mode + 3 SD)
    visual   fish 1's other cells; the paper's rule on the grating (mode + 3 SD, tools/exp20_input_cells.py); the 10 %
             rule (the top 10 % by coherence with the grating, every arm's grating cells)

with the number of cells in the legends.

    PYTHONPATH=src:tools python tools/exp20_atlas_figure.py
Writes presentation/figs/atlas_f1.png (and a copy in png/).
"""
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
ZVOX = (0.798, 0.798, 2.0)
REC = "gutbrain_glucose_f1"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import tifffile
    from plexus.paths import graphs_data_path
    A = np.load(os.path.join(DATA, "atlas", "fish1_regions.npz"), allow_pickle=True)["atlas_um"]
    ld = lambda t: np.load(graphs_data_path("zebrafish", f"input_mask_{REC}{t}.npz"))["mask_by_input"]   # noqa: E731
    apvg, dvc = ld("_anat_apvg")[:, 0] > 0, ld("_anat_dvc")[:, 0] > 0
    gut3 = np.load(os.path.join(DATA, f"baselines_{REC}_cells.npz"))["responsive"]
    vis10 = ld("_bio")[:, 3] > 0
    vis3 = np.load(os.path.join(DATA, f"visual_paper3sd_{REC}.npz"))["mask"]
    F = tifffile.imread(os.path.join(DATA, "atlas", "bigwarp", "zbrain_reference.tif"))     # (z, y, x), head up
    ex, ey, ez = F.shape[2] * ZVOX[0], F.shape[1] * ZVOX[1], F.shape[0] * ZVOX[2]
    gut_layers = [(~(apvg | dvc | gut3), "0.55", 0.35, f"fish 1's other cells ({int((~(apvg | dvc | gut3)).sum()):,}, 1 in 6)", 6),
                  (apvg, "#ff4040", 2.6, f"area postrema + vagal ganglia ({int(apvg.sum()):,}, atlas)", 1),
                  (dvc & ~apvg, "#e040fb", 1.8, f"dorsal vagal complex ({int((dvc & ~apvg).sum()):,}, atlas)", 1),
                  (gut3, "#ffeb3b", 1.4, f"gut-responsive, the paper's rule 3 SD ({int(gut3.sum()):,})", 1)]
    vis_layers = [(~(vis3 | vis10), "0.55", 0.35, f"fish 1's other cells ({int((~(vis3 | vis10)).sum()):,}, 1 in 6)", 6),
                  (vis3, "#1e88e5", 0.6, f"the paper's rule on the grating, 3 SD ({int(vis3.sum()):,})", 1),
                  (vis10, "#66bb6a", 0.6, f"10 % rule, coherence with the grating ({int(vis10.sum()):,})", 1)]
    # one scale, no blank inside a panel (exp17's rule): each view cropped to the cells' own extent + 25 um, every panel
    # as wide as its view (x from above, depth from the side), all the same head-to-tail height; a gap between groups
    lo, hi = np.percentile(A, 0.05, 0) - 25, np.percentile(A, 99.95, 0) + 25
    wx, wz, hy = hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]
    fig = plt.figure(figsize=(14, 14 * hy / (2 * wx + 2 * wz + 0.25 * wx) + 1.6), facecolor="black")
    gs = fig.add_gridspec(1, 5, width_ratios=[wx, wz, 0.25 * wx, wx, wz], wspace=0.03, left=0.0, right=1.0, top=0.93,
                          bottom=0.16)
    for g, (layers, ttl) in enumerate(((gut_layers, "gut-input cells"), (vis_layers, "visual input cells"))):
        axs = []
        for j, view in enumerate((0, 2)):
            ax = fig.add_subplot(gs[0, 3 * g + j])
            ax.set_facecolor("black"); ax.axis("off")
            if view == 0:
                img, ext, xy, xl = F.max(0), [0, ex, ey, 0], (A[:, 0], A[:, 1]), (lo[0], hi[0])
            else:
                img, ext, xy, xl = F.max(2).T, [0, ez, ey, 0], (A[:, 2], A[:, 1]), (lo[2], hi[2])
            ax.imshow(img, cmap="gray", extent=ext, vmax=2.6 * np.percentile(img, 99.5))      # dimmed: the cells show on it
            for m, col, s_, _, every in layers:
                idx = np.where(m)[0][::every]
                ax.scatter(xy[0][idx], xy[1][idx], s=s_, c=col, lw=0)
            ax.set_xlim(*xl); ax.set_ylim(hi[1], lo[1]); ax.set_aspect("equal")
            ax.set_title(("from above" if view == 0 else "from the side"), color="0.85", fontsize=10)
            axs.append(ax)
        x0 = axs[0].get_position().x0                                     # the group's own left edge
        fig.text(x0, 0.975, ttl, color="white", fontsize=13, weight="bold", va="bottom")
        mk = lambda c: Line2D([], [], marker="o", ls="", color=c, markersize=7)     # noqa: E731
        fig.legend([mk(c) for _, c, _, _, _ in layers[::-1]], [lab for _, _, _, lab, _ in layers[::-1]],
                   loc="upper left", bbox_to_anchor=(x0, 0.15), frameon=False, labelcolor="white", fontsize=9.5)
    fig.text(0.5, 0.0, "glucose fish 1 in Z-Brain (30 BigWarp landmarks), head up; grey the Z-Brain reference brain; each "
             "layer drawn over the one listed below it", color="0.7", fontsize=9, ha="center", va="bottom")
    out = os.path.join(EXP, "presentation", "figs", "atlas_f1.png")
    fig.savefig(out, dpi=150, facecolor="black", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    print("[atlas]", out)


if __name__ == "__main__":
    main()

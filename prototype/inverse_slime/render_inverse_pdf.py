"""render_inverse_pdf.py -- GT vs inverse-problem stages, as a PDF.

Two pages: (1) CELL evolution, (2) CHEMICAL FIELD evolution. Rows = the ground-truth
spec and the successive inverse-problem recoveries (UCB black-box, gradient-descent
4D field+move, gradient-descent 8D incl. sense), columns = time. Lets you eyeball how
each stage's morphology tracks GT -- the visual companion to the recovery numbers.
"""
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/slime"
CFG = "/workspace/Plexus/config/slime"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inverse_gt_vs_stages.pdf")

# (folder, label) in recovery order
STAGES = [
    ("slime_gt", "GROUND TRUTH"),
    ("slime_recovered", "UCB (black-box)"),
    ("slime_gd", "grad-descent 4D (field+move)"),
    ("slime_gd8d", "grad-descent 8D (+sense)"),
]
NCOL = 5


def load(folder):
    d = np.load(os.path.join(DR, folder, "trajectory.npz"))
    return np.asarray(d["cell__pos"]), np.asarray(d["chemical__grid"])


def params(folder):
    try:
        y = yaml.safe_load(open(os.path.join(CFG, folder + ".yaml")))
        ops = {o["op"]: o for o in y["operators"]}
        t = next(iter(y["sets"]["cell"]["types"].values()))
        return (f"diffuse={ops['diffuse']['rate']:.3f} decay={ops['decay']['rate']:.4f} "
                f"sensA={t['sensor_angle']:.1f} sensD={t['sensor_dist']:.3f}")
    except Exception:
        return ""


def main():
    data = [(lbl, *load(f), params(f)) for f, lbl in STAGES if os.path.exists(os.path.join(DR, f, "trajectory.npz"))]
    nrow = len(data)
    with PdfPages(OUT) as pdf:
        # ---- page 1: cells ----
        fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 2.2, nrow * 2.3), facecolor="white")
        fig.suptitle("Cell evolution -- GT vs inverse-problem stages", fontsize=14, y=0.995)
        for r, (lbl, pos, grid, pstr) in enumerate(data):
            fr = np.linspace(0, pos.shape[0] - 1, NCOL).astype(int)
            for c, t in enumerate(fr):
                ax = axes[r, c]
                ax.set_facecolor("black")
                ax.scatter(pos[t, :, 0], pos[t, :, 1], s=0.3, c="#33f0a6", linewidths=0)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"t={t}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(lbl, fontsize=9, rotation=90)
            axes[r, 0].text(0.02, -0.16, pstr, transform=axes[r, 0].transAxes, fontsize=6.5, color="0.3")
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

        # ---- page 2: chemical field ----
        fig, axes = plt.subplots(nrow, NCOL, figsize=(NCOL * 2.2, nrow * 2.3), facecolor="white")
        fig.suptitle("Chemical field evolution -- GT vs inverse-problem stages", fontsize=14, y=0.995)
        for r, (lbl, pos, grid, pstr) in enumerate(data):
            fr = np.linspace(0, grid.shape[0] - 1, NCOL).astype(int)
            for c, t in enumerate(fr):
                ax = axes[r, c]
                g = grid[t, 0].T ** 0.7                      # gamma like the spec; .T to image orient
                ax.imshow(g, origin="lower", cmap="magma", vmin=0, vmax=max(1e-6, grid.max() ** 0.7))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"t={t}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(lbl, fontsize=9, rotation=90)
        fig.tight_layout(rect=[0, 0, 1, 0.98]); pdf.savefig(fig); plt.close(fig)

    print("wrote", OUT, "(", nrow, "stages )")


if __name__ == "__main__":
    main()

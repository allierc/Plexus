"""exp17: batch 17's graphs in one figure (Cedric, 2026-10-03: "the brain rollout mean dF/F and the rollout R2 in one
slide"). Local. From each run's free rollout (results/<run>_movie.npz, the whole 2 h):
    a  the brain-mean dF/F, recorded (green) and learned by each graph
    b  the per-frame R2 (denoised, over the neurons), smoothed over 50 frames (46 s)
and data/graph_curves.json with each run's long MSE (h 16-32, 1e-3 dF/F^2), brain-mean R2 and RMSE, per-neuron R2.

    PYTHONPATH=src:tools python tools/exp17_graph_curves.py
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
G = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "training", "zapbench")
from exp17_param_compare import RUNS  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import exp17_slides as S
    cols = [c for c in plt.get_cmap("tab10").colors if c != plt.get_cmap("tab10").colors[2]]     # green is the record
    plt.style.use("dark_background")                     # Cedric, 2026-10-03: black, as the deck
    fig, axs = plt.subplots(2, 1, figsize=(16, 8.6), sharex=True, gridspec_kw={"hspace": 0.25}, facecolor="black")
    doc = {}
    for i, (lab, n) in enumerate(RUNS):
        d = os.path.join(G, n, "results")
        z = np.load(os.path.join(d, f"{n}_movie.npz"))
        t = z["r2_t"] * 0.914 / 60
        if i == 0:
            axs[0].plot(t, z["mean_obs_all"], color="#2ca02c", lw=1.6, label="recorded")
        axs[0].plot(t, z["mean_pred_all"], color=cols[i % len(cols)], lw=0.7, label=lab)
        r = np.convolve(np.nan_to_num(z["r2_denoised_all"]), np.ones(50) / 50, mode="same")
        axs[1].plot(t, r, color=cols[i % len(cols)], lw=0.9, label=lab)
        tj = json.load(open(os.path.join(d, f"{n}_test.json")))
        ms = S.mse_summary(tj)
        bm = S.bm_metrics(os.path.join(d, f"{n}_movie.npz"))
        doc[n] = {"label": lab, "mse_short": float(ms["model_s"]), "mse_long": float(ms["model_l"]),
                  "mean_long": float(ms["mean_l"]), "bm_r2": bm["r2"], "bm_rmse": bm["rmse"],
                  "per_neuron_r2": float(np.nanmean(z["r2_denoised_all"]))}
    axs[0].set_ylabel("brain-mean dF/F")
    axs[0].legend(fontsize=8, ncol=5, frameon=False, loc="upper right")
    axs[1].set_ylabel("R$^2$ per frame (denoised)\nsmoothed over 50 frames")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel("time, min")
    for ax, ttl in zip(axs, ("a   the brain-mean dF/F of the 2 h free rollout: recorded and learned on each graph",
                             "b   the per-frame R2 over the neurons")):
        ax.set_title(ttl, loc="left", fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    path = os.path.join(EXP, "presentation", "figs", "graph_curves.png")
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    json.dump(doc, open(os.path.join(EXP, "data", "graph_curves.json"), "w"), indent=1)
    for n, v in doc.items():
        print(f"{n:18s} long {v['mse_long']:.3f}  brain-mean R2 {v['bm_r2']:+.3f} RMSE {v['bm_rmse']:.4f}  per-neuron {v['per_neuron_r2']:+.3f}")


if __name__ == "__main__":
    main()

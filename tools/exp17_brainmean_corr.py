"""exp17: HOW MUCH OF EACH NEURON IS THE BRAIN MEAN? (Cedric, 2026-10-05: "measure how the neurons correlate with the
brain mean; this can give an idea" -- of whether a shared signal could stand in for the graph's coupling). Local, on the
recording itself (no model).

Per neuron i, over the 7,879 recorded frames:
  r_global   corr(x_i, the brain mean)                                  (the mean field: one shared signal)
  r_local    corr(x_i, the mean of its 6 nearest neurons)               (the graph's finest coupling)
  r_partial  corr of the two after the brain mean is regressed out of both (local signal BEYOND the shared one)
and over the whole brain: the variance fraction the brain mean explains (sum_i var_i r_global_i^2 / sum_i var_i) and the
first principal component's fraction (the best one-dimensional signal).
Writes data/brainmean_corr.json and presentation/figs/brainmean_corr.png (histograms and the maps from above).

    PYTHONPATH=src:tools python tools/exp17_brainmean_corr.py
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def main(device="cuda:0", k=6):
    from scipy.spatial import cKDTree
    from plexus.tasks import trace_recording as TR
    rec = TR.load("zapbench_destripe")
    X = torch.as_tensor(rec["dff"], dtype=torch.float32, device=device)            # [T, N]
    T_, N = X.shape
    X = X - X.mean(0)
    sd = X.std(0).clamp(min=1e-9)
    Z = X / sd                                                                      # z-scored per neuron
    B = X.mean(1)
    Bz = (B - B.mean()) / B.std()
    r_g = (Z * Bz[:, None]).mean(0)                                                 # corr with the brain mean
    pos = np.asarray(rec["pos_um"], np.float64)
    _, nn = cKDTree(pos).query(pos, k=k + 1)
    nn = torch.as_tensor(nn[:, 1:], device=device)
    L = torch.zeros_like(X)
    for j in range(k):
        L += X[:, nn[:, j]]
    L /= k
    Lz = (L - L.mean(0)) / L.std(0).clamp(min=1e-9)
    r_l = (Z * Lz).mean(0)                                                          # corr with the 6 nearest's mean
    # the brain mean regressed out of both (per neuron, least squares on Bz), then their correlation
    Zr = Z - r_g[None] * Bz[:, None]
    r_lg = (Lz * Bz[:, None]).mean(0)
    Lr = Lz - r_lg[None] * Bz[:, None]
    r_p = (Zr * Lr).mean(0) / (Zr.std(0) * Lr.std(0)).clamp(min=1e-9)
    var = X.var(0)
    frac_g = float((var * r_g ** 2).sum() / var.sum())
    U, S, V = torch.svd_lowrank(X, q=8, niter=4)
    frac_pc = (S ** 2 / (X ** 2).sum()).cpu().numpy()
    act = (X.std(0) > 1e-6).cpu().numpy()                 # 2026-10-05: ~6 % of the neurons are flat (the destriping floor)
    rg, rl, rp = (t.cpu().numpy()[act] for t in (r_g, r_l, r_p))
    pos_a = pos[act]
    doc = {"neurons": int(N), "neurons_flat_left_out": int((~act).sum()), "frames": int(T_), "k_nearest": k,
           "r_global": {"median": float(np.median(rg)), "p10": float(np.percentile(rg, 10)), "p90": float(np.percentile(rg, 90)),
                        "frac_above_0.5": float((rg > 0.5).mean()), "frac_negative": float((rg < 0).mean())},
           "r_local": {"median": float(np.median(rl)), "p10": float(np.percentile(rl, 10)), "p90": float(np.percentile(rl, 90))},
           "r_partial_local_beyond_global": {"median": float(np.median(rp)), "p10": float(np.percentile(rp, 10)),
                                             "p90": float(np.percentile(rp, 90))},
           "variance_explained_by_brain_mean": frac_g,
           "variance_explained_by_pcs_1_to_8": frac_pc.tolist()}
    json.dump(doc, open(os.path.join(EXP, "data", "brainmean_corr.json"), "w"), indent=1)
    print(json.dumps(doc, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_ablation import _brain_view
    P = _brain_view(pos_a)
    o = np.argsort(P[:, 2])
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 8.4), facecolor="black")
    ax = fig.add_axes([0.05, 0.58, 0.27, 0.34])
    bins = np.linspace(-0.4, 1.0, 71)
    for v, c, lab in ((rg, "white", "with the brain mean"), (rl, "#ff9f1c", f"with its {k} nearest neurons' mean"),
                      (rp, "#4aa8ff", "local, the brain mean removed")):
        ax.hist(v, bins, histtype="step", color=c, lw=1.6, label=f"{lab} (median {np.median(v):+.2f})")
    ax.set_xlabel("correlation over the 2 h, per neuron")
    ax.set_ylabel("neurons")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax2 = fig.add_axes([0.05, 0.10, 0.27, 0.34])
    ax2.bar(np.arange(1, 9), 100 * frac_pc, color="0.7")
    ax2.axhline(100 * frac_g, color="white", ls="--", lw=1)
    ax2.text(8.4, 100 * frac_g, f"brain mean {100 * frac_g:.1f} %", color="white", fontsize=9, va="bottom", ha="right")
    ax2.set_xlabel("principal component")
    ax2.set_ylabel("variance explained, %")
    for s_ in ("top", "right"):
        ax2.spines[s_].set_visible(False)
    for j, (v, ttl, cm, lo, hi) in enumerate(((rg, "correlation with the brain mean", "viridis", -0.2, 0.9),
                                              (rp, "local correlation beyond the brain mean", "magma", -0.1, 0.8))):
        a3 = fig.add_axes([0.37, 0.52 - 0.47 * j, 0.55, 0.40])
        a3.set_facecolor("black")
        sc = a3.scatter(P[o, 0], P[o, 1], c=v[o], s=0.25, cmap=cm, vmin=lo, vmax=hi, linewidths=0)
        a3.set_aspect("equal")
        a3.axis("off")
        fig.text(0.37, 0.93 - 0.47 * j, ttl + " (from above, head left)", color="white", fontsize=11)
        cax = fig.add_axes([0.93, 0.56 - 0.47 * j, 0.008, 0.30])
        fig.colorbar(sc, cax=cax).ax.tick_params(colors="0.8", labelsize=8)
    path = os.path.join(EXP, "presentation", "figs", "brainmean_corr.png")
    fig.savefig(path, dpi=120, facecolor="black")
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    print("[corr]", path)


if __name__ == "__main__":
    main()

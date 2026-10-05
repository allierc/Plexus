"""exp17: A FEW NEURONS, RECORDED AGAINST LEARNED (Cedric, 2026-10-05: "a few traces true vs inferred, with the figure
template of connectome_gnn.pdf Supp. Fig. 14; first choose neurons representative of different parts of the fish; a map
to point the probed neurons"). Local.

THE CHOICE, from the RECORDING ONLY (blind to the model, so not picked for a good fit): the neurons that move (SD > 0)
split by position into K regions (k-means on x, y, z, um); in each region the neuron whose recorded trace correlates
most with its region's mean trace -- the most typical neuron of that part of the brain. Numbered head to tail.
THE FIGURE (black): left, the brain from above and from the side with the probed neurons numbered; right, each neuron's
free-rollout trace over the 2 h at the rollout's 800 saved frames (one every ~9 s), recorded green, learned white, on the
neuron's own scale, its Pearson r over the 2 h at the right; the stimulus conditions as blocks above.

    PYTHONPATH=src:tools python tools/exp17_traces.py zap_e15_cur_siren_mesh3 [K]
Writes presentation/figs/traces_<run>.png and data/traces_<run>.json (+ png/).
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


def pick(X, pos, K, seed=0):
    """K representative neurons: k-means regions on position, in each the neuron most correlated with its region mean."""
    from sklearn.cluster import KMeans
    act = np.where(X.std(0) > 1e-6)[0]
    lab = KMeans(K, n_init=4, random_state=seed).fit_predict(pos[act])
    Z = (X[:, act] - X[:, act].mean(0)) / X[:, act].std(0)
    out = []
    for k in range(K):
        m = np.where(lab == k)[0]
        mean = Z[:, m].mean(1)
        mean = (mean - mean.mean()) / mean.std()
        r = (Z[:, m] * mean[:, None]).mean(0)
        out.append((int(act[m[np.argmax(r)]]), float(r.max()), int(len(m)), pos[act[m]].mean(0)))
    return out


def main(run, K=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import _brain_view
    spec = T.load(run)
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    X = np.asarray(rec["dff"], np.float64)
    P = _brain_view(np.asarray(rec["pos_um"], np.float64))
    sel = sorted(pick(X, P, K), key=lambda t: t[3][0])                           # head (left) to tail
    z = np.load(os.path.join(G, run, "results", f"{run}_movie.npz"))
    fr = z["frames"]
    pred = z["pred"]
    tm = fr * 0.914 / 60
    cond = rec["condition"][fr]
    names = [str(n) for n in rec["names"]]
    cut = np.flatnonzero(np.diff(cond)) + 1
    st, en = np.r_[0, cut], np.r_[cut, len(cond)] - 1
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[::2][:K] if K <= 10 else plt.get_cmap("tab20")(np.arange(K) % 20)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9.0), facecolor="black")
    o = np.argsort(P[:, 2])
    for j, (ys, h, view) in enumerate(((0.50, 0.42, "top"), (0.08, 0.30, "side"))):
        ax = fig.add_axes([0.01, ys, 0.33, h])
        ax.set_facecolor("black")
        ax.axis("off")
        Y = P[:, 1] if view == "top" else P[:, 2]
        ax.scatter(P[o, 0], Y[o], s=0.08, c="0.32", linewidths=0)
        for i, (n, *_r) in enumerate(sel):
            ax.scatter(P[n, 0], Y[n], s=70, facecolors="none", edgecolors=[cols[i]], linewidths=1.6)
            ax.text(P[n, 0], Y[n] + (14 if view == "top" else 9), str(i + 1), color=cols[i], fontsize=9, ha="center",
                    va="bottom", weight="bold")
        ax.set_aspect("equal")
        fig.text(0.015, ys + h + 0.01, "from above (head left)" if view == "top" else "from the side", color="white",
                 fontsize=11)
    tr = fig.add_axes([0.40, 0.06, 0.55, 0.86])
    tr.set_facecolor("black")
    rs = []
    for i, (n, rrep, size, _c) in enumerate(sel):
        x_, p_ = X[fr, n], pred[:, n].astype(np.float64)
        lo, hi = np.percentile(x_, [1, 99.5])
        sc = max(hi - lo, 1e-6)
        off = (K - 1 - i) * 1.25
        tr.plot(tm, (x_ - lo) / sc + off, color="#2ca02c", lw=0.8)
        tr.plot(tm, (p_ - lo) / sc + off, color="white", lw=0.7)
        r = float(np.corrcoef(x_, p_)[0, 1])
        rs.append(r)
        tr.text(-1.5, off + 0.4, str(i + 1), color=cols[i], fontsize=10, ha="right", va="center", weight="bold")
        tr.text(tm[-1] + 1.0, off + 0.4, f"r {r:+.2f}", color="0.85", fontsize=8.5, va="center")
    for i_, (a_, b_) in enumerate(zip(st, en)):
        tr.axvspan(tm[a_], tm[b_], color=("0.16" if i_ % 2 else "0.08"), lw=0, zorder=0)
        tr.text((tm[a_] + tm[b_]) / 2, K * 1.25 + 0.05, names[int(cond[a_])], color="0.75", fontsize=7.5, ha="center")
    tr.set_xlim(tm[0], tm[-1])
    tr.set_ylim(-0.4, K * 1.25 + 0.5)
    tr.set_yticks([])
    for s_ in ("top", "right", "left"):
        tr.spines[s_].set_visible(False)
    tr.set_xlabel("time, min (the free rollout of the 2 h, 800 frames)", fontsize=9)
    fig.text(0.40, 0.955, "recorded (green), learned (white); each neuron on its own scale (its 1st-99.5th percentile)",
             color="0.85", fontsize=10)
    path = os.path.join(EXP, "presentation", "figs", f"traces_{run}.png")
    fig.savefig(path, dpi=120, facecolor="black", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    doc = {"run": run, "K": K, "neurons": [{"index": n, "region_size": size, "r_with_region_mean": rrep,
                                           "pos_view_um": c.tolist(), "r_learned_vs_recorded": r}
                                          for (n, rrep, size, c), r in zip(sel, rs)],
           "r_median": float(np.median(rs)), "r_min": float(np.min(rs)), "r_max": float(np.max(rs))}
    json.dump(doc, open(os.path.join(EXP, "data", f"traces_{run}.json"), "w"), indent=1)
    print("[traces]", path, f"r median {doc['r_median']:+.2f} ({doc['r_min']:+.2f} .. {doc['r_max']:+.2f})")


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))

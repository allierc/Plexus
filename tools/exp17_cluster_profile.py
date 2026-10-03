"""exp17: WHAT THE CLUSTERS ARE, AND HOW FAR TO TRUST THEM (Cedric, 2026-10-03, on batch 15's 8 clusters: "how
trustful are the 8 clusters, and what are these clusters about: different tau, different V rest, different W?").

Local analysis. The clusters are exp17_clusters.labels_of's (as the montage slide): KMeans(k) on each neuron's own
learned constants -- softplus(tau) (its rate per frame), its rest V, its stimulus weights B -- each block z-scored and
weighted to unit total variance; numbered by size, largest first, as the montage. W is NOT a clustering feature; it
is profiled here beside the others.

TRUST
  restarts     ARI of KMeans with 10 other random starts against the shown one (1 = the same partition)
  subsamples   ARI of KMeans fitted on 5 random 80 % subsets, every neuron then assigned, against the shown one
  silhouette   on 20,000 neurons (-1..1; > 0.5 well separated, < 0.25 weak structure)
  other runs   ARI against the same clustering of an independently trained run on the same neurons (`--others`),
               on the features both share (rate, rest, the first 22 stimulus columns): a cluster that is the data's,
               not the seed's, comes back
  mask         how much of the partition is the input mask (B is 0 by construction outside it: those neurons' B
               gets no gradient): ARI against the mask, and each cluster's masked fraction

WHAT THEY ARE (per cluster): tau = 0.914 s / rate (the leak's time constant), V in dF/F (V sd + mu, the reference's
normalisation), |B| (masked neurons only), the summed signed W into and out of each neuron (all three edge sets), the
recorded dF/F's mean and SD, the position along the body axis (0 head .. 1 tail); eta^2 = the share of each
quantity's variance BETWEEN the clusters (1: the clusters are that quantity; 0: they ignore it).

    PYTHONPATH=src:tools python tools/exp17_cluster_profile.py zap_e15_cur_siren --k 8 --others zap_e15_cur zap_v14_cur_siren

Writes experiments/exp17_zapbench_graphcast/data/cluster_profile_<run>_k<k>.json and
presentation/figs/cluster_profile_<run>_k<k>.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914


def features(name, ncols_b=None):
    """(z-scored, block-weighted features [N, F] as labels_of, the raw blocks, spec, fitted)."""
    from plexus import trainer as T
    spec = T.load(name)
    fit = torch.load(os.path.join(T.out_dir(spec, None), "models", "best.pt"), weights_only=False,
                     map_location="cpu")["fitted"]
    have = [b for b in ("tau", "rest", "gain", "input") if f"neuron.{b}" in fit]
    raw = {b: (torch.nn.functional.softplus(fit[f"neuron.{b}"].float()) if b == "tau" else fit[f"neuron.{b}"].float())
           for b in have}
    if ncols_b is not None and "input" in raw:
        raw["input"] = raw["input"][:, :ncols_b]
    emb = torch.cat([((b - b.mean(0)) / b.std(0).clamp(min=1e-9)) / b.shape[1] ** 0.5 for b in raw.values()], 1).numpy()
    return emb, {k: v.numpy() for k, v in raw.items()}, spec, fit


def kmeans(X, k, seed=0):
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=4, random_state=seed).fit(X)


def ranked(lab, k):
    """Relabel 0..k-1 by size, largest first (the montage's numbering)."""
    order = np.argsort(-np.bincount(lab, minlength=k))
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    return rank[lab]


def eta2(x, lab, k):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    x, l_ = x[ok], lab[ok]
    tot = ((x - x.mean()) ** 2).sum()
    btw = sum(((x[l_ == c].mean() - x.mean()) ** 2) * (l_ == c).sum() for c in range(k) if (l_ == c).any())
    return float(btw / tot) if tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--others", nargs="*", default=[])
    a = ap.parse_args()
    from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
    from plexus import trainer as T
    from plexus.tasks import trace_recording as TR
    from exp17_ablation import neuron_graph_op
    k = a.k
    X, raw, spec, fit = features(a.run)
    km = kmeans(X, k, 0)
    lab = ranked(km.labels_, k)
    N = len(lab)
    rng = np.random.default_rng(0)
    trust = {"restarts_ari": [float(ari(lab, kmeans(X, k, s).labels_)) for s in range(1, 11)]}
    sub = []
    for s in range(5):
        idx = rng.choice(N, int(0.8 * N), replace=False)
        sub.append(float(ari(lab, kmeans(X[idx], k, 100 + s).predict(X))))
    trust["subsample_ari"] = sub
    si = rng.choice(N, 20000, replace=False)
    trust["silhouette"] = float(silhouette_score(X[si], lab[si]))
    for o in a.others:                                  # the same clustering of another run, on the shared features
        nb = min(raw["input"].shape[1], features(o)[1]["input"].shape[1]) if "input" in raw else None
        Xa = features(a.run, nb)[0]
        Xo = features(o, nb)[0]
        trust[f"ari_vs_{o}"] = float(ari(ranked(kmeans(Xa, k, 0).labels_, k), ranked(kmeans(Xo, k, 0).labels_, k)))
    # the mask, the edges' weights, the recording
    op = neuron_graph_op(spec, "cpu")
    mask = op.input_mask.cpu().numpy().reshape(-1).astype(bool) if op.input_mask is not None else np.ones(N, bool)
    trust["ari_vs_mask"] = float(ari(lab, mask.astype(int)))
    w_in, w_out, a_in = np.zeros(N), np.zeros(N), np.zeros(N)
    for e in spec["learnable"]:
        p = str(e.get("param", ""))
        if p.startswith("W_"):
            snd, rcv = (t.cpu().numpy() for t in op._E[p[2:]])
            w = fit[T.Learnables.key(e)].float().numpy().reshape(-1)
            np.add.at(w_in, rcv, w)
            np.add.at(w_out, snd, w)
            np.add.at(a_in, rcv, np.abs(w))
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    Xd = rec["dff"]
    mu, sd = float(Xd.mean()), float(Xd.std())            # the reference's normalisation (split: all)
    act_mean, act_sd = Xd.mean(0), Xd.std(0)
    pos = rec["pos_um"]
    ax_ = int(np.argmax(np.ptp(pos, 0)))
    body = (pos[:, ax_] - pos[:, ax_].min()) / np.ptp(pos[:, ax_])   # 0 .. 1 along the long axis (head = small x)
    tau_s = FRAME_S / raw["tau"].reshape(-1)
    V = raw["rest"].reshape(-1) * sd + mu
    Bn = np.linalg.norm(raw["input"], axis=1) if "input" in raw else np.zeros(N)
    q = {"log10 tau": np.log10(tau_s), "V rest": V, "|B| (masked)": np.where(mask, Bn, np.nan),
         "masked": mask.astype(float), "W in": w_in, "W out": w_out, "|W| in": a_in, "dF/F mean": act_mean,
         "dF/F SD": act_sd, "body axis": body}
    prof = []
    names = [str(s) for s in rec.get("stimulus_names", [f"f{i}" for i in range(raw["input"].shape[1])])]
    for c in range(k):
        m = lab == c
        Bm = raw["input"][m & mask].mean(0) if (m & mask).any() else np.zeros(raw["input"].shape[1])
        top = np.argsort(-np.abs(Bm))[:3]
        prof.append({"cluster": c + 1, "n": int(m.sum()), "masked_frac": float(mask[m].mean()),
                     "masked_n": int((m & mask).sum()), "share_of_all_masked": float((m & mask).sum() / max(mask.sum(), 1)),
                     "tau_s_median": float(np.median(tau_s[m])), "tau_s_iqr": np.percentile(tau_s[m], [25, 75]).tolist(),
                     "V_median": float(np.median(V[m])), "B_norm_median_masked": float(np.median(Bn[m & mask])) if (m & mask).any() else None,
                     "W_in_median": float(np.median(w_in[m])), "W_out_median": float(np.median(w_out[m])),
                     "absW_in_median": float(np.median(a_in[m])), "dff_mean": float(act_mean[m].mean()),
                     "dff_sd": float(np.median(act_sd[m])), "body_median": float(np.median(body[m])),
                     "top_B": [(names[i], float(Bm[i])) for i in top]})
    e2 = {kq: eta2(v, lab, k) for kq, v in q.items()}
    doc = {"run": a.run, "k": k, "features": list(raw), "trust": trust, "eta2": e2, "clusters": prof,
           "n_masked": int(mask.sum()), "n": int(N),
           "norm": {"mu": mu, "sd": sd}, "frame_s": FRAME_S}
    os.makedirs(os.path.join(EXP, "data"), exist_ok=True)
    jp = os.path.join(EXP, "data", f"cluster_profile_{a.run}_k{k}.json")
    json.dump(doc, open(jp, "w"), indent=1)
    figure(doc, q, lab, mask, raw["input"], names, a.run, k)
    print(json.dumps({"trust": {kk: (np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3)) for kk, v in trust.items()},
                      "eta2": {kk: round(v, 3) for kk, v in e2.items()}}, indent=1))
    for p in prof:
        print(p)


def figure(doc, q, lab, mask, B, names, run, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    show = [("log10 tau", "leak time constant $\\tau$, log10 s"), ("V rest", "rest $V$, dF/F"),
            ("|B| (masked)", "stimulus weight $|B|$ (input neurons)"), ("W in", "summed W into the neuron"),
            ("W out", "summed W out of the neuron"), ("dF/F SD", "recorded dF/F, SD over time"),
            ("body axis", "position, head 0 .. tail 1")]
    fig = plt.figure(figsize=(16, 9.6), facecolor="white")
    for i, (key, lab_) in enumerate(show):
        ax = fig.add_axes([0.05 + (i % 4) * 0.245, 0.67 - (i // 4) * 0.33, 0.20, 0.19])
        data = [q[key][(lab == c) & np.isfinite(q[key])] for c in range(k)]
        ax.boxplot(data, showfliers=False, widths=0.6, medianprops={"color": "black"})
        ax.set_xticks(range(1, k + 1))
        ax.set_xlabel("cluster (by size)", fontsize=8)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.text(0.05 + (i % 4) * 0.245, 0.87 - (i // 4) * 0.33, f"{lab_}\n$\\eta^2$ = {doc['eta2'][key]:.2f}",
                 fontsize=9, va="bottom")
    ax = fig.add_axes([0.05 + 3 * 0.245, 0.34, 0.20, 0.19])
    ax.bar(range(1, k + 1), [p["masked_frac"] for p in doc["clusters"]], color="0.4")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, k + 1))
    ax.set_xlabel("cluster (by size)", fontsize=8)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.05 + 3 * 0.245, 0.54, f"fraction of input neurons (mask)\nARI with the mask = {doc['trust']['ari_vs_mask']:.2f}",
             fontsize=9, va="bottom")
    ax = fig.add_axes([0.05, 0.05, 0.92, 0.15])
    M = np.stack([B[(lab == c) & mask].mean(0) if ((lab == c) & mask).any() else np.zeros(B.shape[1]) for c in range(k)])
    v = np.abs(M).max()
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_yticks(range(k))
    ax.set_yticklabels([str(c + 1) for c in range(k)], fontsize=8)
    ax.set_xticks(range(B.shape[1]))
    ax.set_xticklabels([n.split(" (")[0].replace("ephys ", "") for n in names], fontsize=7)
    ax.set_ylabel("cluster", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).ax.tick_params(labelsize=7)
    fig.text(0.05, 0.215, "mean stimulus weight B per cluster (input neurons only), per stimulus feature", fontsize=9)
    path = os.path.join(EXP, "presentation", "figs", f"cluster_profile_{run}_k{k}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("[profile]", path)


if __name__ == "__main__":
    main()

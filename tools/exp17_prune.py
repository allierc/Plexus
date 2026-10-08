"""exp17: WHICH EDGES CAN GO? (Cedric, 2026-10-08: "look at the distribution of W_ij, define a threshold where W_ij is low
enough to consider the edge removable"). Local, a landed neuron-graph run's models/best.pt, no training.

THE DISTRIBUTION, PER LEVEL (Cedric, 2026-10-08: "partition the threshold per level"): log10 |W| of each set is bimodal -- a DEAD mode near 1e-3 (the L1 prior's floor,
the edges training never used) and a LIVE mode near 0.05. A two-Gaussian mixture is fitted to log10 |W| (EM); the
threshold t* is where the two components' weighted densities cross between their means: below it an edge is more likely
dead than alive.
THE CHECK, behavioural, per set: its edges with |W| < t zeroed (the other sets intact), for a geometric ladder from
t*/9 to 3 t*, and the 2 h free rollout rerun
(trainer._trace_free, the test's own); brain-mean r and per-neuron r (exp17_slides.bm_metrics / local_r) against the
unpruned law. A set's edges below t are removable when both move by less than the nominal's seed spread (19.25 vs
19.26: 0.009 brain-mean r, 0.005 per-neuron r); its threshold is the largest such t. Then the three sets cut at once at
their own thresholds, checked again (the losses add). On 19.25 (2026-10-08) one global cut at the mixture's crossing,
0.025, already cost 0.020 per-neuron r: the dead mode's upper tail still carries signal.

    PYTHONPATH=src:tools python tools/exp17_prune.py zap_n19_nom
-> data/prune_<run>.json, presentation/figs/prune_<run>.png (the histogram, t*, and r against t), data/prune/<run>/ (rollouts)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
SETS = ("short", "mid", "long")
SEED_SPREAD = {"brain_mean_r": 0.009, "per_neuron_r": 0.005}     # 19.25 vs 19.26 (tools/exp17_meanfield_stats.py --n19)


def gmm2(x, it=200):
    """A two-Gaussian mixture on 1-D x by EM; returns (weights, means, sds), the lower mean first."""
    m = np.percentile(x, [25, 75]).astype(np.float64)
    s = np.array([x.std(), x.std()]) / 2
    w = np.array([0.5, 0.5])
    for _ in range(it):
        p = w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x[:, None] - m) / s) ** 2)
        g = p / p.sum(1, keepdims=True)
        n = g.sum(0)
        w, m = n / len(x), (g * x[:, None]).sum(0) / n
        s = np.sqrt((g * (x[:, None] - m) ** 2).sum(0) / n)
    o = np.argsort(m)
    return w[o], m[o], s[o]


def crossing(w, m, s):
    """Where w0 N(m0, s0) = w1 N(m1, s1) between the two means."""
    xs = np.linspace(m[0], m[1], 20001)
    d = [w[k] / s[k] * np.exp(-0.5 * ((xs - m[k]) / s[k]) ** 2) for k in (0, 1)]
    return float(xs[np.argmin(np.abs(d[0] - d[1]))])


def main(run):
    import torch
    from plexus import trainer as T
    import exp17_slides as S
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    spec = T.load(run)
    out = T.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=dev)
    fit = ck["fitted"]
    W = {s: fit[f"state_diffuse.W_{s}"].float().cpu().numpy() for s in SETS if f"state_diffuse.W_{s}" in fit}
    n_all = sum(len(w) for w in W.values())
    box = T._trace_setup(spec, dev)
    pdir = os.path.join(EXP, "data", "prune", run)
    os.makedirs(os.path.join(pdir, "results"), exist_ok=True)

    def rollout(th, tag):
        """The 2 h free rollout with every edge of set s below th[s] zeroed; its two r and the edges kept."""
        f2, kept = dict(fit), {}
        for s in W:
            k = f"state_diffuse.W_{s}"
            w = fit[k]
            t = th.get(s, 0.0)
            f2[k] = torch.where(w.abs() < t, torch.zeros_like(w), w)
            kept[s] = int((w.abs() >= t).sum())
        learn = T.Learnables(spec["learnable"], dev)
        learn.restore(f2)
        with torch.no_grad():
            T._trace_free(spec, learn, box, dev, pdir, f"{run}_{tag}")
        npz = os.path.join(pdir, "results", f"{run}_{tag}_movie.npz")
        bm, lr = S.bm_metrics(npz), S.local_r(npz, "zapbench_destripe")
        r = {"thresholds": dict(th), "kept": kept, "kept_share": sum(kept.values()) / n_all,
             "brain_mean_r": bm["r"], "per_neuron_r": lr["mean"]}
        print(f"[prune] {tag}: {sum(kept.values()):,} of {n_all:,} edges kept ({100 * r['kept_share']:.1f} %), "
              f"brain-mean r {bm['r']:.4f}, per-neuron r {lr['mean']:.4f}", flush=True)
        return r
    base = rollout({}, "none")

    def judge(r):
        r["d_brain_mean_r"] = r["brain_mean_r"] - base["brain_mean_r"]
        r["d_per_neuron_r"] = r["per_neuron_r"] - base["per_neuron_r"]
        r["removable"] = bool(abs(r["d_brain_mean_r"]) < SEED_SPREAD["brain_mean_r"]
                              and abs(r["d_per_neuron_r"]) < SEED_SPREAD["per_neuron_r"])
        return r
    # PER LEVEL (Cedric, 2026-10-08: "partition the threshold per level"): each set's own mixture and ladder, the other
    # sets intact; its threshold the largest of its ladder within the seed spread
    doc = {"run": run, "seed_spread": SEED_SPREAD, "unpruned": base, "per_set": {}}
    best = {}
    for s in W:
        lw = np.log10(np.abs(W[s]) + 1e-12)
        w_, m_, s_ = gmm2(lw)
        ts = 10 ** crossing(w_, m_, s_)
        print(f"[gmm] {s}: dead mode 10^{m_[0]:.2f} ({w_[0]:.2f}), live 10^{m_[1]:.2f} ({w_[1]:.2f}); crossing {ts:.4g}")
        ladder = [judge(rollout({s: float(t)}, f"{s}_t{t:.3g}")) for t in np.geomspace(ts / 9, 3 * ts, 7)]
        ok_ = [r["thresholds"][s] for r in ladder if r["removable"]]
        best[s] = max(ok_) if ok_ else 0.0
        doc["per_set"][s] = {"n": int(len(W[s])), "gmm": {"weights": w_.tolist(), "log10_means": m_.tolist(),
                                                          "log10_sds": s_.tolist()},
                             "crossing": ts, "ladder": ladder, "threshold": best[s],
                             "removed_share": float((np.abs(W[s]) < best[s]).mean())}
        print(f"[prune] {s}: removable below |W| = {best[s]:.4g}, {100 * doc['per_set'][s]['removed_share']:.1f} % of its edges")
    # the three together at their own thresholds: errors add, so the joint cut is checked on its own
    doc["joint"] = judge(rollout(best, "joint"))
    doc["thresholds"] = best
    json.dump(doc, open(os.path.join(EXP, "data", f"prune_{run}.json"), "w"), indent=1)
    draw(doc, W)


def draw(doc, W):
    """Per set: its log10 |W| histogram, the mixture, the crossing and the kept threshold (top); the change of both r
    against the threshold, the seed spread shaded (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    sets = list(doc["per_set"])
    fig, axs = plt.subplots(2, len(sets), figsize=(5.2 * len(sets), 8.0), facecolor="black", squeeze=False)
    names = {"short": "short: level 0, every neuron", "mid": "mid: level 1, 32-um cubes", "long": "long: level 2, 64-um cubes"}
    for c, s in enumerate(sets):
        P = doc["per_set"][s]
        lw = np.log10(np.abs(W[s]) + 1e-12)
        a1, a2 = axs[0, c], axs[1, c]
        b = np.arange(-7, 1.01, 0.1)
        a1.hist(lw, bins=b, color="0.7")
        w_, m_, s_ = (np.array(P["gmm"][k]) for k in ("weights", "log10_means", "log10_sds"))
        xs = np.linspace(-7, 1, 600)
        for k, col in ((0, "#4a7bff"), (1, "#ff4a4a")):
            a1.plot(xs, len(lw) * 0.1 * w_[k] / (s_[k] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - m_[k]) / s_[k]) ** 2),
                    color=col, lw=1.4, label=("dead" if k == 0 else "live") + f" mode, 10^{m_[k]:.2f}")
        a1.axvline(np.log10(P["crossing"]), color="0.6", ls=":", lw=1.2, label=f"modes cross, {P['crossing']:.3g}")
        if P["threshold"] > 0:
            a1.axvline(np.log10(P["threshold"]), color="yellow", ls="--", lw=1.4,
                       label=f"removable below {P['threshold']:.3g} ({100 * P['removed_share']:.0f} %)")
        a1.set_title(f"{names.get(s, s)}: {P['n']:,} edges", fontsize=10, loc="left")
        a1.set_xlabel("log10 |W|", fontsize=9)
        a1.legend(fontsize=7.5, frameon=False, loc="upper left")
        L = P["ladder"]
        t = [r["thresholds"][s] for r in L]
        a2.axhspan(-doc["seed_spread"]["per_neuron_r"], doc["seed_spread"]["per_neuron_r"], color="0.3", alpha=0.5, lw=0,
                   label="seed spread, per-neuron r")
        a2.plot(t, [r["d_per_neuron_r"] for r in L], "o-", color="#ff9e1a", label="per-neuron r")
        a2.plot(t, [r["d_brain_mean_r"] for r in L], "s-", color="#4dd94d", label="brain-mean r")
        if P["threshold"] > 0:
            a2.axvline(P["threshold"], color="yellow", ls="--", lw=1.2)
        a2.set_xscale("log")
        a2.axhline(0, color="0.6", lw=0.6)
        a2.set_xlabel(f"threshold: every {s} edge with |W| below it zeroed", fontsize=9)
        if c == 0:
            a2.set_ylabel("change against the unpruned law, 2 h free rollout", fontsize=9)
        a2.legend(fontsize=7.5, frameon=False, loc="lower left")
    J = doc["joint"]
    fig.suptitle(f"all three cut at once: {100 * J['kept_share']:.0f} % of the edges kept, brain-mean r "
                 f"{J['d_brain_mean_r']:+.4f}, per-neuron r {J['d_per_neuron_r']:+.4f}"
                 + (" (within the seed spread)" if J["removable"] else " (beyond the seed spread)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"prune_{doc['run']}.png"), dpi=130, facecolor="black")
    plt.close(fig)

if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)

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

THE LATTICE GRID (state_diffuse[neuron_grid], 20.3; Cedric, 2026-10-08: "a twin of the pruned-mesh slides for 20.3"): its
weights W_grid sit on the grid's corner-to-corner edges, one per direction, plus a self edge per corner; an edge's set is its
length (16 / 32 / 64 um: level 0 / 1 / 2, 0: the self edges), its strength the coupling it applies, W_grid^2 under
sign: neuron (|W_grid| under sign: grid). Two differences from the mesh, measured on 20.3 (2026-10-08):
  INERT EDGES  an edge whose sending corner no neuron encodes into, or whose receiving corner no neuron decodes from, carries
               nothing: its gradient is 0 and it keeps its starting W_grid (1.0) -- 27,853 of the 133,643 (21 %), exactly the
               edges at W_grid = 1.0. Removed first, in every rollout below, checked on their own (the change must be 0).
  NO DEAD MODE the live edges' W_grid^2 has no mode near 0 (the grid had no L1 prior; the median is 1.0): no mixture, the
               ladder is the live edges' own 5 / 10 / 20 / 30 / 40 / 50 % quantiles, judged as above.

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


GRID_SETS = ("self", "fine", "middle", "coarse")                  # W_grid by edge length: 0, 16, 32, 64 um (L0 x 2^k)


def grid_edges(spec):
    """The lattice grid of a neuron_grid run, as its operator holds it: the op, its mesh (centre_um), every W_grid entry's
    sender / receiver corner, its set (GRID_SETS, by the edge's length) and whether it is inert (its sending corner encodes
    no neuron, or its receiving corner decodes to none)."""
    import torch
    from exp17_ablation import neuron_graph_op
    from plexus.operators.cell_ops import StateDiffuseGraphCast
    op = neuron_graph_op(spec, "cpu")
    pos = np.load(op._pos_file[0])[op._pos_file[1]]
    G = StateDiffuseGraphCast.mesh(op, torch.as_tensor(np.asarray(pos, np.float64)))    # the op's own (cached) mesh
    ms, mr = (t.numpy() for t in op._grid["mm"])
    nm = op._grid["n_mesh"]
    fed, read = np.zeros(nm, bool), np.zeros(nm, bool)
    fed[op._grid["g2m"][1].numpy()] = True
    read[op._grid["m2g"][0].numpy()] = True
    ln = np.linalg.norm(G["centre_um"][ms] - G["centre_um"][mr], axis=1)
    k = np.where(ln < 1e-6, 0, np.round(np.log2(np.maximum(ln, 1e-9) / op.L0)) + 1).astype(int)   # 0 self, 1 + level
    return {"op": op, "G": G, "ms": ms, "mr": mr, "set": np.array(GRID_SETS)[k], "inert": ~fed[ms] | ~read[mr],
            "pos": np.asarray(pos, np.float64)}


def grid_w_in(E, fit, th=None):
    """Each neuron's SUMMED EFFECTIVE INCOMING WEIGHT through the lattice grid (Cedric, 2026-10-08: "panel c computed after
    edge removal"): the law's message into neuron i is g_i (1/8) sum_{l in cube(i)} sum_{k -> l} w_kl (1/n_k) sum_{j -> k}
    a_j tanh z_j, so its weight on neuron j is W_ij = g_i (1/8) sum_l sum_k w_kl a_j / n_k over the paths j -> k -> l -> i,
    and sum_j W_ij = g_i (1/8) sum_l sum_k w_kl abar_k, abar_k the mean a_j of corner k's neurons -- the grid's twin of the
    mesh's summed W_ij. With `th` (data/prune_<run>.json's thresholds) the inert edges and those below their set's
    threshold are zeroed first. a = A_send, w = W_grid^2, g = G_recv^2 under sign: neuron (a = 1, w = W_grid: sign grid)."""
    op = E["op"]
    gs, gr = (t.numpy() for t in op._grid["g2m"])
    cs, cr = (t.numpy() for t in op._grid["m2g"])
    nm, N = op._grid["n_mesh"], op.n_elements
    wg = fit["state_diffuse.W_grid"].float().numpy().astype(np.float64)
    neu = op.sign == "neuron"
    a = fit["state_diffuse.A_send"].float().numpy().astype(np.float64) if neu else np.ones(N)
    w = wg ** 2 if neu else wg
    if th is not None:
        eff = np.abs(w)
        thv = np.array([th.get(s, 0.0) for s in E["set"]])           # each entry its set's threshold
        w = np.where(E["inert"] | (eff < thv), 0.0, w)
    cnt = np.maximum(np.bincount(gr, minlength=nm), 1)
    abar = np.bincount(gr, weights=a[gs], minlength=nm) / cnt
    h = np.bincount(E["mr"], weights=w * abar[E["ms"]], minlength=nm)
    g = fit["state_diffuse.G_recv"].float().numpy().astype(np.float64) ** 2
    return g * np.bincount(cr, weights=h[cs], minlength=N) / 8.0


def main(run):
    import torch
    from plexus import trainer as T
    import exp17_slides as S
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    spec = T.load(run)
    out = T.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=dev)
    fit = ck["fitted"]
    grid = "state_diffuse.W_grid" in fit
    if grid:                                   # the lattice grid: W_grid's live entries per set, its strength the coupling
        E = grid_edges(spec)
        wg = fit["state_diffuse.W_grid"].float().cpu().numpy()
        eff = wg ** 2 if E["op"].sign == "neuron" else np.abs(wg)
        IDX = {s: np.flatnonzero((E["set"] == s) & ~E["inert"]) for s in GRID_SETS}
        W = {s: eff[IDX[s]] for s in GRID_SETS}
        inert_idx = np.flatnonzero(E["inert"])
        n_all = len(wg)
    else:
        W = {s: fit[f"state_diffuse.W_{s}"].float().cpu().numpy() for s in SETS if f"state_diffuse.W_{s}" in fit}
        n_all = sum(len(w) for w in W.values())
    box = T._trace_setup(spec, dev)
    pdir = os.path.join(EXP, "data", "prune", run)
    os.makedirs(os.path.join(pdir, "results"), exist_ok=True)

    def rollout(th, tag, inert=True):
        """The 2 h free rollout with every edge of set s below th[s] zeroed (the grid: its inert edges too, unless
        inert=False); its two r and the edges kept."""
        f2, kept = dict(fit), {}
        if grid:
            w = fit["state_diffuse.W_grid"].clone()
            if inert:
                w[torch.as_tensor(inert_idx)] = 0.0
            for s in W:
                t = th.get(s, 0.0)
                w[torch.as_tensor(IDX[s][W[s] < t])] = 0.0
                kept[s] = int((W[s] >= t).sum())
            f2["state_diffuse.W_grid"] = w
            kept["inert"] = 0 if inert else int(len(inert_idx))          # kept_share over every W_grid entry
        for s in ([] if grid else W):
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
    base = rollout({}, "none", inert=False)

    def judge(r):
        r["d_brain_mean_r"] = r["brain_mean_r"] - base["brain_mean_r"]
        r["d_per_neuron_r"] = r["per_neuron_r"] - base["per_neuron_r"]
        r["removable"] = bool(abs(r["d_brain_mean_r"]) < SEED_SPREAD["brain_mean_r"]
                              and abs(r["d_per_neuron_r"]) < SEED_SPREAD["per_neuron_r"])
        return r
    # PER LEVEL (Cedric, 2026-10-08: "partition the threshold per level"): each set's own mixture and ladder, the other
    # sets intact; its threshold the largest of its ladder within the seed spread
    doc = {"run": run, "seed_spread": SEED_SPREAD, "unpruned": base, "per_set": {}}
    if grid:
        doc["grid"] = {"edges": n_all, "inert": int(len(inert_idx)), "strength": "W_grid^2" if E["op"].sign == "neuron"
                       else "|W_grid|", "inert_per_set": {s: int(((E["set"] == s) & E["inert"]).sum()) for s in GRID_SETS},
                       "inert_only": judge(rollout({}, "inert"))}
    best = {}
    for s in W:
        lw = np.log10(np.abs(W[s]) + 1e-12)
        if grid:                               # no dead mode: the live edges' own quantiles
            ts, gm = None, None
            ts_ = sorted({max(float(q), 1e-6) for q in np.percentile(np.abs(W[s]), [5, 10, 20, 30, 40, 50])})
        else:
            w_, m_, s_ = gmm2(lw)
            ts = 10 ** crossing(w_, m_, s_)
            gm = {"weights": w_.tolist(), "log10_means": m_.tolist(), "log10_sds": s_.tolist()}
            print(f"[gmm] {s}: dead mode 10^{m_[0]:.2f} ({w_[0]:.2f}), live 10^{m_[1]:.2f} ({w_[1]:.2f}); crossing {ts:.4g}")
            ts_ = np.geomspace(ts / 9, 3 * ts, 7)
        ladder = [judge(rollout({s: float(t)}, f"{s}_t{t:.3g}")) for t in ts_]
        ok_ = [r["thresholds"][s] for r in ladder if r["removable"]]
        best[s] = max(ok_) if ok_ else 0.0
        doc["per_set"][s] = {"n": int(len(W[s])), "gmm": gm, "crossing": ts, "ladder": ladder, "threshold": best[s],
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
    names = {"short": "short: level 0, every neuron", "mid": "mid: level 1, 32-um cubes", "long": "long: level 2, 64-um cubes",
             "self": "self: each corner to itself", "fine": "fine: level 0, 16-um edges", "middle": "middle: level 1, 32-um edges",
             "coarse": "coarse: level 2, 64-um edges"}
    gd = doc.get("grid")
    wl_ = "log10 " + (gd["strength"] if gd else "|W|") + (" (the live edges)" if gd else "")
    for c, s in enumerate(sets):
        P = doc["per_set"][s]
        lw = np.log10(np.abs(W[s]) + 1e-12)
        a1, a2 = axs[0, c], axs[1, c]
        b = np.arange(-7, 1.01, 0.1)
        a1.hist(lw, bins=b, color="0.7")
        if P["gmm"]:
            w_, m_, s_ = (np.array(P["gmm"][k]) for k in ("weights", "log10_means", "log10_sds"))
            xs = np.linspace(-7, 1, 600)
            for k, col in ((0, "#4a7bff"), (1, "#ff4a4a")):
                a1.plot(xs, len(lw) * 0.1 * w_[k] / (s_[k] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - m_[k]) / s_[k]) ** 2),
                        color=col, lw=1.4, label=("dead" if k == 0 else "live") + f" mode, 10^{m_[k]:.2f}")
            a1.axvline(np.log10(P["crossing"]), color="0.6", ls=":", lw=1.2, label=f"modes cross, {P['crossing']:.3g}")
        if P["threshold"] > 0:
            a1.axvline(np.log10(P["threshold"]), color="yellow", ls="--", lw=1.4,
                       label=f"removable below {P['threshold']:.3g} ({100 * P['removed_share']:.0f} %)")
        a1.set_title(f"{names.get(s, s)}: {P['n']:,} edges" + (f" (+ {gd['inert_per_set'][s]:,} inert)" if gd else ""),
                     fontsize=10, loc="left")
        a1.set_xlabel(wl_, fontsize=9)
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
        a2.set_xlabel(f"threshold: every {s} edge with {gd['strength'] if gd else '|W|'} below it zeroed", fontsize=9)
        if c == 0:
            a2.set_ylabel("change against the unpruned law, 2 h free rollout", fontsize=9)
        a2.legend(fontsize=7.5, frameon=False, loc="lower left")
    J = doc["joint"]
    fig.suptitle(("the inert edges and every set cut at once: " if gd else "all three cut at once: ")
                 + f"{100 * J['kept_share']:.0f} % of the edges kept, brain-mean r "
                 f"{J['d_brain_mean_r']:+.4f}, per-neuron r {J['d_per_neuron_r']:+.4f}"
                 + (" (within the seed spread)" if J["removable"] else " (beyond the seed spread)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"prune_{doc['run']}.png"), dpi=130, facecolor="black")
    plt.close(fig)

if __name__ == "__main__":
    for n_ in sys.argv[1:]:
        main(n_)

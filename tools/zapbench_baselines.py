#!/usr/bin/env python
"""exp17 Stage 0: every reference number a GraphCast law on ZAPBench is read against, measured BEFORE training.

    PYTHONPATH=src python tools/zapbench_baselines.py [--device cuda:0]

Writes experiments/exp17_zapbench_graphcast/data/baselines.json, data/mesh_stats.json and data/figs/*.png.
Runs in the devcontainer (INSTRUCTION.md, the cluster rule). STANDARD MSE EVERYWHERE (Cedric, 2026-09-30).

1. THE SCORER'S IDENTITY CHECK. ZAPBench's mean baseline (eq. 10, W = 4 at context 4) on ZAPBench's OWN test
   split (per condition: the last 20 % of the condition's frames, padding 1, windows of context + 32 at stride 1;
   TAXIS on its `test_holdout` range) must reproduce the published MSE (`combined.json`, method `mean`,
   context 4): per condition and in the grand average over the 8 non-held-out conditions. A scorer that does not
   is wrong, and nothing below is trusted until it does.
2. THE MEAN BASELINE ON exp17's PROTOCOL. Every frame o with 6 frames behind it (o >= 5) and 32 ahead is an
   origin; the forecast of step h is the neuron's mean over its last W frames (W = 1 is persistence). MSE per
   origin (over the 71,721 neurons) per h and W; per condition (the origin's) and grand average (mean over the 9
   conditions). THE MEAN BASELINE OF THE GATES is, per h, the best W in 1..6 on the grand average -- the
   strongest baseline the 6-frame context allows.
3. A STIMULUS-EVOKED LOOKUP (in place of the release's `stimulus_evoked_response`, not on disk -- asked of Cedric):
   each neuron's mean dF/F over all OTHER frames with the same condition, the same stimulus vector and the same
   number of frames since the stimulus last changed (capped at 64) -- ZAPBench's eq. 11 read as "the response
   phase-locked to the stimulus", leave-one-out so a frame never predicts itself.
4. THE NOISE CEILING. The structure function D(h) = MSE between frames h apart; its intercept at h = 0 (a line
   through h = 1..3) is 2 sigma^2, twice the white-noise variance; no forecast of a noisy target beats MSE =
   sigma^2, so the ceiling skill at h is 1 - sigma^2 / MSE_mean(h).
5. THE kNN SPATIAL POOL (prototype/graphcast/PLAN.md, G17): the change dF/F(t+1) - dF/F(t) of each neuron
   predicted by the mean change of its 8 nearest neighbours at the same t; R^2 over neurons and frames.
6. FREE-ROLLOUT REFERENCES, frames 6..T-1 from frames 0..5: the context mean held, and (reading only, it uses the
   future) each neuron's time-mean; R^2 per frame over neurons (spatial), against the raw recording and the
   DENOISED one -- each neuron's 3-frame mean (t-1, t, t+1), fixed here, before any model.
7. MESH STATISTICS (GraphCast's Table 4 for this brain): finest spacing 8, 16, 32 um x 5 levels.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
from plexus.paths import graphs_data_path                              # noqa: E402

EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
DATA = os.path.join(EXP, "data")
H, CTX, WMAX = 32, 6, 6                        # horizon, exp17's context, largest W of the mean baseline
PAD, VAL_F, TEST_F, MAXCTX = 1, 0.1, 0.2, 256  # ZAPBench constants.py
HOLDOUT = 3                                    # TAXIS


def load():
    z = np.load(graphs_data_path("zebrafish", "zapbench_recording.npz"))
    return {k: z[k] for k in z.files}


def mean_pred(X, o, W):
    """[len(o), N]: each neuron's mean over frames o-W+1..o."""
    return torch.stack([X[o - w] for w in range(W)], 0).mean(0)


# ------------------------------------------------------------------------------ 1. the scorer check
def zapbench_test_windows(off, c, C):
    lo, hi = off[c] + PAD, off[c + 1] - PAD
    tot = hi - lo
    if c == HOLDOUT:
        lo += MAXCTX - C
    else:
        lo += tot - int(tot * TEST_F) - C          # train + val = tot - test
        # (ZAPBench: inclusive_min += num_train + num_val - context, num_test = int(tot * 0.2))
    n_win = hi - lo - H - C + 1
    return lo + C - 1 + np.arange(n_win)           # the origin = the last context frame


def scorer_check(X, off, names, dev):
    import io
    import pandas as pd
    pub = pd.read_json(io.StringIO(json.load(open(os.path.join(
        EXP, "papers", "Lueckmann_2025_published_results_combined.json")))))
    pm = pub[(pub.method == "mean") & (pub.context == 4)]
    out = {}
    for c, name in enumerate(names):
        o = torch.as_tensor(zapbench_test_windows(off, c, 4), device=dev)
        m = mean_pred(X, o, 4)
        ours = [float(((X[o + h] - m) ** 2).mean()) for h in range(1, H + 1)]
        p = pm[pm.condition == name].groupby("steps_ahead").MSE.mean()
        out[name] = {"n_windows": int(len(o)), "ours": ours, "published": [float(p[h]) for h in range(1, H + 1)]}
    cs = [n for i, n in enumerate(names) if i != HOLDOUT]
    ga_o = np.mean([out[n]["ours"] for n in cs], 0)
    ga_p = np.mean([out[n]["published"] for n in cs], 0)
    rel = np.abs(ga_o / ga_p - 1)
    out["grand_average"] = {"ours": ga_o.tolist(), "published": ga_p.tolist(), "max_rel_err": float(rel.max())}
    print(f"[scorer] grand average, mean W=4 ctx 4: ours h1 {ga_o[0]:.6f} / published {ga_p[0]:.6f}; "
          f"h32 {ga_o[-1]:.6f} / {ga_p[-1]:.6f}; max relative error over h {rel.max():.2e}")
    return out


# ------------------------------------------------------------------------------ 2. mean baselines, exp17 protocol
def protocol_origins(T):
    return np.arange(CTX - 1, T - H)


def mean_baselines(X, cond, dev):
    T = X.shape[0]
    o = torch.as_tensor(protocol_origins(T), device=dev)
    per = np.zeros((WMAX, H, len(o)))               # [W, h, origin] MSE over neurons
    for W in range(1, WMAX + 1):
        m = mean_pred(X, o, W)
        for h in range(1, H + 1):
            per[W - 1, h - 1] = ((X[o + h] - m) ** 2).mean(1).cpu().numpy()
    oc = cond[o.cpu().numpy()]
    by_c = np.stack([per[:, :, oc == c].mean(2) for c in range(int(cond.max()) + 1)], 0)   # [C, W, h]
    ga = by_c.mean(0)                                                                      # [W, h]
    best_w = ga.argmin(0) + 1
    return {"origins": int(len(o)), "per_condition": by_c, "grand": ga, "best_W": best_w,
            "per_origin": per, "origin_condition": oc}


# ------------------------------------------------------------------------------ 3. stimulus-evoked lookup
def stimulus_lookup(X, S, cond, dev, cap=64):
    T = X.shape[0]
    keys, since = [], 0
    for t in range(T):
        if t > 0 and (cond[t] != cond[t - 1] or not np.array_equal(S[t], S[t - 1])):
            since = 0
        elif t > 0:
            since = min(since + 1, cap)
        keys.append((int(cond[t]),) + tuple(np.round(S[t], 4)) + (since,))
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    kid = torch.as_tensor([uniq[k] for k in keys], device=dev)
    nk = len(uniq)
    sums = torch.zeros(nk, X.shape[1], device=dev).index_add_(0, kid, X)
    cnt = torch.zeros(nk, device=dev).index_add_(0, kid, torch.ones(T, device=dev))
    loo_n = (cnt[kid] - 1).clamp(min=1)[:, None]
    pred = (sums[kid] - X) / loo_n                                   # leave-one-out
    single = cnt[kid] < 2
    pred[single] = X.mean(0)                                         # a key seen once: the neuron's mean
    o = torch.as_tensor(protocol_origins(T), device=dev)
    per = np.stack([((X[o + h] - pred[o + h]) ** 2).mean(1).cpu().numpy() for h in range(1, H + 1)], 0)
    return {"n_keys": nk, "frames_single_key": int(single.sum()), "per_origin": per}   # [h, origin]


# ------------------------------------------------------------------------------ 4. noise ceiling
def noise_ceiling(X):
    D = [float(((X[h:] - X[:-h]) ** 2).mean()) for h in range(1, 7)]
    hs = np.arange(1, 4)
    slope, icpt = np.polyfit(hs, D[:3], 1)
    return {"D": D, "intercept_2sigma2": float(icpt), "sigma2": float(max(icpt, 0) / 2), "slope": float(slope)}


# ------------------------------------------------------------------------------ 5. kNN pool
def knn_pool(X, pos, dev, k=8):
    from scipy.spatial import cKDTree
    nb = cKDTree(pos).query(pos, k=k + 1)[1][:, 1:]
    dX = X[1:] - X[:-1]
    nbt = torch.as_tensor(nb, device=dev)
    ss_res = ss_tot = 0.0
    for t0 in range(0, dX.shape[0], 512):
        d = dX[t0:t0 + 512]
        p = d[:, nbt].mean(2)
        ss_res += float(((d - p) ** 2).sum())
        ss_tot += float(((d - d.mean()) ** 2).sum())
    return {"k": k, "r2_dx": 1 - ss_res / ss_tot}


# ------------------------------------------------------------------------------ 6. free-rollout references
def r2_frames(pred, X):
    """R^2 per frame over neurons: 1 - sum_i (pred - x)^2 / sum_i (x - mean_i x)^2."""
    num = ((pred - X) ** 2).sum(1)
    den = ((X - X.mean(1, keepdim=True)) ** 2).sum(1)
    return (1 - num / den).cpu().numpy()


def free_refs(X):
    Xd = X.clone()
    Xd[1:-1] = (X[:-2] + X[1:-1] + X[2:]) / 3                     # the denoised recording, fixed here
    ctx = X[:CTX].mean(0, keepdim=True).expand(X.shape[0] - CTX, -1)
    tm = X.mean(0, keepdim=True).expand(X.shape[0] - CTX, -1)
    out = {}
    for lab, p in (("context_mean_held", ctx), ("time_mean_map", tm)):
        for ref, R in (("raw", X[CTX:]), ("denoised", Xd[CTX:])):
            r = r2_frames(p, R)
            out[f"{lab}_{ref}"] = {"mean": float(r.mean()), "sd": float(r.std())}
    # the ceiling of the raw R^2: the denoised recording itself scored against the raw one
    r = r2_frames(Xd[CTX:], X[CTX:])
    out["denoised_vs_raw"] = {"mean": float(r.mean()), "sd": float(r.std())}
    return out


def published_curves(names):
    """ZAPBench's published MSE skill over its own mean baseline, per step ahead, grand average over its 8
    non-held-out conditions (combined.json, the MSE column of the runs behind Fig. 4): the best model at each
    context and the stimulus baseline. Written beside the frozen recording (graphs_data/zebrafish/
    zapbench_published.json) so every run's curves draw them (plexus.tasks.trace_recording.render_curves)."""
    import io
    import pandas as pd
    df = pd.read_json(io.StringIO(json.load(open(os.path.join(
        EXP, "papers", "Lueckmann_2025_published_results_combined.json")))))
    tr = df[df.condition != names[HOLDOUT]]
    g = (tr.groupby(["method", "context", "steps_ahead", "xid"]).MSE.mean().reset_index()
         .groupby(["method", "context", "steps_ahead"]).MSE.mean())
    out = {"source": "papers/Lueckmann_2025_published_results_combined.json (ZAPBench, ICLR 2025, Fig. 4, MSE)",
           "split": "ZAPBench's: per condition the last 20 % of frames, TAXIS held out -- not exp17's all-frames origins"}
    for ctx in (4, 256):
        mean = np.array([g[("mean", ctx, h)] for h in range(1, H + 1)])
        best = None
        for m in ("linear", "tide", "tsmixer", "time-mix", "unet"):
            sk = 1 - np.array([g[(m, ctx, h)] for h in range(1, H + 1)]) / mean
            if best is None or sk[15:].mean() > best[1][15:].mean():
                best = (m, sk)
        out[f"best_ctx{ctx}"] = {"method": best[0], "skill": best[1].tolist()}
        out[f"stimulus_ctx{ctx}"] = {"skill": (1 - np.array([g[("stimulus", ctx, h)] for h in range(1, H + 1)]) / mean).tolist()}
    json.dump(out, open(graphs_data_path("zebrafish", "zapbench_published.json"), "w"), indent=1)
    return out


# ------------------------------------------------------------------------------ figures
def figures(mb, stim, ceil, names, check, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figs, exist_ok=True)
    h = np.arange(1, H + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for W in range(1, WMAX + 1):
        a.plot(h, mb["grand"][W - 1], lw=1.2, label=f"mean, W = {W}" + (" (persistence)" if W == 1 else ""))
    a.plot(h, stim["grand"], lw=1.6, color="#c44e52", label="stimulus-evoked lookup")
    a.axhline(ceil["sigma2"], color="0.5", ls=":", lw=1, label="noise floor sigma$^2$")
    a.set_xlabel("steps ahead h (0.914 s)")
    a.set_ylabel("MSE, dF/F$^2$ (grand average, 9 conditions)")
    a.legend(frameon=False, fontsize=7)
    a.text(0.0, 1.03, "a", transform=a.transAxes, fontsize=12)
    best = mb["grand"][mb["best_W"] - 1, np.arange(H)]
    b.plot(h, 1 - stim["grand"] / best, color="#c44e52", lw=1.6, label="stimulus-evoked lookup")
    b.plot(h, 1 - ceil["sigma2"] / best, color="0.5", ls=":", lw=1.2, label="noise ceiling")
    b.axhline(0, color="k", lw=0.8)
    b.set_xlabel("steps ahead h")
    b.set_ylabel("MSE skill over the best mean baseline")
    b.legend(frameon=False, fontsize=7)
    b.text(0.0, 1.03, "b", transform=b.transAxes, fontsize=12)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "baselines.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ga = check["grand_average"]
    ax.plot(h, ga["published"], "o", ms=4, color="0.6", label="published (combined.json)")
    ax.plot(h, ga["ours"], "-", color="k", lw=1.2, label="this scorer")
    ax.set_xlabel("steps ahead h")
    ax.set_ylabel("MSE, mean baseline W = 4, ZAPBench test split")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "scorer_check.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    t_all = time.time()
    R = load()
    off, names, cond = R["offsets"], [str(n) for n in R["names"]], R["condition"]
    X = torch.as_tensor(R["dff"], device=a.device)
    print(f"[data] {tuple(X.shape)} on {a.device}")
    check = scorer_check(X, off, names, a.device)
    published_curves(names)
    mb = mean_baselines(X, cond, a.device)
    print(f"[mean] best W per h: {mb['best_W'].tolist()}")
    st = stimulus_lookup(X, R["stimulus"], cond, a.device)
    oc = mb["origin_condition"]
    st_c = np.stack([st["per_origin"][:, oc == c].mean(1) for c in range(len(names))], 0)   # [C, h]
    st["per_condition"], st["grand"] = st_c, st_c.mean(0)
    ceil = noise_ceiling(X)
    knn = knn_pool(X, R["pos_um"], a.device)
    free = free_refs(X)
    best = mb["grand"][mb["best_W"] - 1, np.arange(H)]
    skill_st = 1 - st["grand"] / best
    ceil_skill = 1 - ceil["sigma2"] / best
    out = {
        "written": time.strftime("%Y-%m-%d %H:%M"), "script": "tools/zapbench_baselines.py",
        "protocol": {"context": CTX, "horizon": H, "origins": mb["origins"], "W_max": WMAX,
                     "grand_average": "mean over the 9 conditions of the per-condition mean over origins",
                     "origin_condition": "the condition of the origin frame"},
        "scorer_check": check,
        "mean_baseline": {
            "best_W": mb["best_W"].tolist(),
            "mse_best": best.tolist(),
            "mse_by_W": {int(W): mb["grand"][W - 1].tolist() for W in range(1, WMAX + 1)},
            "per_condition_best": {names[c]: mb["per_condition"][c][mb["best_W"] - 1, np.arange(H)].tolist()
                                   for c in range(len(names))},
            "short_mse": float(best[:3].mean()), "long_mse": float(best[15:].mean()),
        },
        "stimulus_lookup": {"n_keys": st["n_keys"], "frames_single_key": st["frames_single_key"],
                            "mse": st["grand"].tolist(), "skill": skill_st.tolist(),
                            "skill_short": float(skill_st[:3].mean()), "skill_long": float(skill_st[15:].mean()),
                            "per_condition_skill_long": {
                                names[c]: float((1 - st_c[c] / mb["per_condition"][c][mb["best_W"] - 1, np.arange(H)])[15:].mean())
                                for c in range(len(names))}},
        "noise": {**ceil, "ceiling_skill": ceil_skill.tolist(), "ceiling_skill_short": float(ceil_skill[:3].mean()),
                  "ceiling_skill_long": float(ceil_skill[15:].mean())},
        "knn_pool": knn,
        "free_rollout": free,
        "seconds": None,
    }
    import exp17_slides as SL
    ms = []
    for L0 in (8.0, 16.0, 32.0):
        M = SL.op_mesh(R["pos_um"], L0, 5)                          # the law's own mesh (vertices, nested, mirrored)
        ms.append({"L0_um": L0, **M["stats"], "neurons_per_node": R["pos_um"].shape[0] / M["stats"]["mesh_nodes"]})
        print(f"[mesh] {L0:.0f} um: {M['stats']['mesh_nodes']} nodes, nodes per level {M['stats']['nodes_per_level']}")
    os.makedirs(DATA, exist_ok=True)
    json.dump(ms, open(os.path.join(DATA, "mesh_stats.json"), "w"), indent=1)
    out["seconds"] = round(time.time() - t_all, 1)
    json.dump(out, open(os.path.join(DATA, "baselines.json"), "w"), indent=1)
    figures(mb, st, ceil, names, check, os.path.join(DATA, "figs"))
    s = out
    print(json.dumps({"best_W": s["mean_baseline"]["best_W"], "short_mse": s["mean_baseline"]["short_mse"],
                      "long_mse": s["mean_baseline"]["long_mse"], "stim_skill_short": s["stimulus_lookup"]["skill_short"],
                      "stim_skill_long": s["stimulus_lookup"]["skill_long"], "noise": {k: s["noise"][k] for k in
                      ("sigma2", "ceiling_skill_short", "ceiling_skill_long")}, "knn": knn, "free": free,
                      "scorer_max_rel_err": check["grand_average"]["max_rel_err"]}, indent=1))


if __name__ == "__main__":
    main()

"""exp17: THE INPUT NEURONS BY A SIGMA RULE AGAINST THE DARK BLOCK (Cedric, 2026-10-06: "replace the 5 / 10 / 20 % rule by
a method closer to exp20's 3-sigma rule; we have no visual stimulus outside the fish, but we have the dark control").
Local, model-free.

For each of the 13 visual features that change within their condition (exp17_stim_coherence --varying-only's rule) and
each direction, the EVENTS are the frames where the feature switches to a positive value (or to a negative one). Neuron
i's RESPONSE to an event type is
    R = mean over the events of [ mean dF/F over the W frames after the switch  -  mean over the W frames before ]
and its NULL (`--null block`, the default) is the same difference at the frames of the SAME block at least W from any
switch -- the block's own fluctuation; `--null dark` takes every frame of the DARK block instead (too narrow: the
brain is calmer in the dark, 78 % of the neurons passed 3 sigma). Mean mu_i and SD s_i over those frames, so a mean over E pseudo-events has mean mu_i and SD s_i / sqrt(E). Neuron i is an input neuron for feature k
when |R - mu_i| > n s_i / sqrt(E) for one of k's event types (n = 2 or 3; both signs: excited or suppressed). The
per-feature mask: neuron i reads exactly the features it passes. Per block, the union of its features' neurons.
Left / right: the head-up view of the movies, split at the neurons' median.

    PYTHONPATH=src:tools python tools/exp17_sigma_mask.py [--window 10] [--sigma 2 3] [--write]
Prints the counts per block (total / left / right); with --write, graphs_data/zebrafish/input_mask_destripe_sig<n>.npz
(mask, mask_by_input, z [N, event types], the events) and data/sigma_mask_counts.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")


def main(W=10, sigmas=(2.0, 3.0), write=False, null="block", remove_mean=False, tag="", spread="null"):
    from plexus.paths import graphs_data_path
    from exp17_ablation import _brain_view
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    U, off, names = np.asarray(z["stimulus"], np.float32), z["offsets"], [str(x) for x in z["names"]]
    T, N = X.shape
    if remove_mean:     # each neuron's share of the brain-wide signal removed: the response BEYOND the brain mean
        b = X.mean(1, dtype=np.float64)
        bc = (b - b.mean()).astype(np.float32)
        beta = (bc[:, None] * (X - X.mean(0))).sum(0) / float((bc ** 2).sum())
        X = X - bc[:, None] * beta[None]
    cond = np.searchsorted(off, np.arange(T), side="right") - 1
    # the 13 changing features and their block (exp17_stim_coherence --varying-only)
    live = [j for j in range(U.shape[1]) if np.abs(U[:, j]).max() > 0]
    fblock = {j: next(k for k in range(len(off) - 1) if np.abs(U[off[k]:off[k + 1], j]).max() > 0) for j in live}
    feats = [j for j in live if U[np.isin(cond, np.unique(cond[np.abs(U[:, j]) > 0])), j].std() > 1e-6]
    P = _brain_view(z["pos_um"].astype(np.float64))
    left = P[:, 1] < np.median(P[:, 1])
    # the null: the after-minus-before difference at every frame of the dark block
    kd = names.index("dark")
    ts = np.arange(off[kd] + W, off[kd + 1] - W)
    cs = np.concatenate([np.zeros((1, N), np.float64), np.cumsum(X, 0, dtype=np.float64)])
    diff = lambda t: ((cs[t + 1 + W] - cs[t + 1]) - (cs[t] - cs[t - W])) / W          # noqa: E731
    D0 = np.stack([diff(t) for t in ts])
    mu0, s0 = D0.mean(0), D0.std(0) + 1e-9
    etypes, Z = [], []
    for j in feats:
        u = U[:, j]
        for sgn in (+1, -1):
            ev = np.where((np.sign(u[1:]) == sgn) & (np.sign(u[:-1]) != sgn))[0] + 1
            ev = ev[(ev >= W) & (ev + W + 1 <= T) & (cond[ev] == fblock[j])]
            if len(ev) < 2:
                continue
            Re = np.stack([diff(t) for t in ev])
            R = Re.mean(0)
            if null == "block":
                # the null within the SAME block: every frame of the block at least W from any switch of any feature
                k_ = fblock[j]
                sw = np.where(np.any(U[1:] != U[:-1], 1))[0] + 1
                fr = np.arange(off[k_] + W, off[k_ + 1] - W - 1)
                far = np.ones(len(fr), bool)
                for t_ in sw:
                    far &= np.abs(fr - t_) > W
                fr = fr[far]
                if len(fr) < 20:
                    continue
                Db = np.stack([diff(t) for t in fr])
                m_, s_ = Db.mean(0), Db.std(0) + 1e-9
            else:
                m_, s_ = mu0, s0
            if spread == "events":     # the response's own switch-to-switch variability (a one-sample t)
                s_ = Re.std(0, ddof=1) + 1e-9
            Z.append((R - m_) / (s_ / np.sqrt(len(ev))))
            etypes.append({"feature": int(j), "sign": sgn, "events": int(len(ev)), "block": names[fblock[j]],
                           "null_frames": int(len(fr)) if null == "block" else int(len(ts))})
    Z = np.stack(Z, 1)                                                                     # [N, event types]
    out = {"window": W, "dark_frames": int(len(ts)), "event_types": etypes, "by_sigma": {}}
    for n in sigmas:
        hit = np.abs(Z) > n
        mbi = np.zeros((N, U.shape[1]), np.float32)
        for c, e in enumerate(etypes):
            mbi[hit[:, c], e["feature"]] = 1.0
        mask = mbi.any(1)
        rows = {}
        for k, b in enumerate(names):
            fb = [j for j in feats if fblock[j] == k]
            a = mbi[:, fb].any(1) if fb else np.zeros(N, bool)
            rows[b] = {"total": int(a.sum()), "left": int((a & left).sum()), "right": int((a & ~left).sum())}
        out["by_sigma"][str(n)] = {"input_neurons": int(mask.sum()), "left": int((mask & left).sum()),
                                   "right": int((mask & ~left).sum()), "features_per_input": float(mbi[mask].sum(1).mean()),
                                   "blocks": rows}
        print(f"\n{n:g} sigma: {int(mask.sum()):,} input neurons ({100 * mask.mean():.1f} %), left {int((mask & left).sum()):,}"
              f" / right {int((mask & ~left).sum()):,}; features per input neuron {mbi[mask].sum(1).mean():.2f}")
        for b, r in rows.items():
            print(f"   {b:10s} total {r['total']:7,d}  left {r['left']:7,d}  right {r['right']:7,d}")
        if write:
            np.savez(graphs_data_path("zebrafish", f"input_mask_destripe_sig{n:g}{tag}.npz"), mask=mask.astype(np.float32),
                     mask_by_input=mbi, z=Z.astype(np.float32), event_types=json.dumps(etypes), window=W, sigma=n)
    print("\nevent types:", [(e["feature"], e["sign"], e["events"]) for e in etypes])
    json.dump(out, open(os.path.join(EXP, "data", f"sigma_mask_counts{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--sigma", type=float, nargs="+", default=[2.0, 3.0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--null", default="block", choices=["block", "dark"])
    ap.add_argument("--remove-mean", action="store_true", help="test the response beyond the brain mean")
    ap.add_argument("--tag", default="")
    ap.add_argument("--spread", default="null", choices=["null", "events"])
    a = ap.parse_args()
    main(a.window, tuple(a.sigma), a.write, a.null, a.remove_mean, a.tag, a.spread)

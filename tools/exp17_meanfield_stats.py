"""exp17: IS THE GRAPH BETTER THAN A MEAN FIELD, SIGNIFICANTLY? (Cedric, 2026-10-05: "a significance test, a barplot with
p values *, **, ***"). Local, from the free rollouts' movie npz (results/<run>[_W0]_movie.npz).

THE NULL: two laws follow the recording equally well -- the same brain-mean r, the same local r -- over the same 2 h.
THE METHOD: a paired block bootstrap over TIME. The 2 h are cut into N_BLOCKS consecutive equal blocks (~5 min); a
resample draws as many blocks with replacement, the SAME blocks for every law (paired: every law is scored on the same
recorded frames), and both metrics are recomputed on it. Blocks, not frames: neighbouring frames are correlated, and a
frame-wise resample would understate the spread. Neurons are not resampled one by one either: with ~95,000 correlated
neurons any difference would read as p ~ 0.
    p = (1 + #{ |d* - d| >= |d| }) / (1 + B),   d the difference of the metric between two laws, d* its resampled value
(two-sided, the resampled differences shifted to the null). Stars: *** p < 0.001, ** < 0.01, * < 0.05, n.s. otherwise.
What it does NOT cover: a law's training seed (one seed per law). For scale, the one seed pair with rollouts on disk
(15.1 and 17.8, the same spec, seeds 0 and 1) gives a seed-to-seed difference, reported beside the tests.

The metrics, as tools/exp17_slides.py but on the STEADY part of the rollout, the first block (~5 min, the transient
from the recorded initial state) left out: brain-mean r = Pearson r of the learned and recorded brain-mean dF/F over
every free frame (mean_obs_all / mean_pred_all); local r = per neuron, the learned and recorded traces over the saved
frames each regressed on their own brain mean, then correlated; the mean over ONE neuron set for every law (finite in
every law, recorded residual moving). A neuron whose learned residual is FLAT scores r = 0: a flat prediction predicts
nothing. Why (2026-10-05): with W = 0 at inference 58 % of the neurons relax to rest within the first block and stay
flat (the no-W twin: 40 %), so the deck's local r of those rows comes from the opening transient alone, and on a resample
without that block their r is 0 / 0. On a resample both come from
weighted sums per block (local r: 9 sums per neuron per block), so a resample costs a matrix product.

    PYTHONPATH=src:tools python tools/exp17_meanfield_stats.py [--raw] [--n19]
Another experiment (exp20, 2026-10-05) calls main(laws, tests, seeds, exp_dir, raw) with its own runs: laws = ((label,
run name, rollout suffix "" or "_W0"), ...), tests = ((i, j), ...) indices into laws, seeds = a (label, run, "") pair of
one spec trained with two seeds, or None; every run's free rollout must share its frames.
Writes data/meanfield_stats.json and presentation/figs/meanfield_stats.png (+ png/). With --raw (Cedric, 2026-10-05:
"a twin without the mean subtraction"): the per-neuron r of the raw traces, the brain mean NOT regressed out, written
as meanfield_stats_raw.*; the brain-mean r and the test are unchanged.
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914
RAW = "--raw" in sys.argv       # the per-neuron r without the brain-mean regression
SUF = "_raw" if RAW else ""
N_BLOCKS = 24                   # equal blocks of the free frames: ~5 min each over the 2 h
B = 10000                       # resamples: the smallest p is 1 / (B + 1)
LAWS = (("15.4 graph", "zap_e15_cur_siren", ""), ("15.4, W = 0 at inference", "zap_e15_cur_siren", "_W0"),
        ("15.17 mean field", "zap_e15_cur_siren_mf", ""), ("15.9 no W", "zap_e15_now", ""))
TESTS = ((0, 2), (2, 3), (0, 1))            # graph vs mean field, mean field vs no W, graph vs its own W = 0
SEEDS = (("15.1, seed 0", "zap_e15_cur", ""), ("17.8, seed 1", "zap_g17_s1", ""))
# `--n19` (Cedric, 2026-10-08: "make all the slides with 19.25"): the same test on the new nominal and its controls,
# written as meanfield_stats_n19*; its seed pair is the nominal itself, seeds 0 and 1 (19.25, 19.26)
LAWS_N19 = (("19.25 graph", "zap_n19_nom", ""), ("19.25, W = 0 at inference", "zap_n19_nom", "_W0"),
            ("19.41 mean field", "zap_n19_mf", ""), ("19.40 no W", "zap_n19_now", ""))
SEEDS_N19 = (("19.25, seed 0", "zap_n19_nom", ""), ("19.26, seed 1", "zap_n19_nom_s1", ""))


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def brain_mean_sums(z, nblk_t):
    """Per block: n, sum o, p, o^2, p^2, o p of the brain-mean traces over every free frame."""
    o, p = np.asarray(z["mean_obs_all"], np.float64), np.asarray(z["mean_pred_all"], np.float64)
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o - o[ok].mean(), p - p[ok].mean()
    blk = nblk_t(np.arange(len(o)) + int(z["frames"][0]))     # the free frames, as frame numbers of the recording
    S = np.zeros((blk.max() + 1, 6))
    for k, v in enumerate((np.ones_like(o), o, p, o * o, p * p, o * p)):
        np.add.at(S[:, k], blk[ok], v[ok])
    return S


def r_from(S):
    """Pearson r from summed (n, o, p, oo, pp, op) [..., 6]."""
    n, so, sp, soo, spp, sop = (S[..., k] for k in range(6))
    c = sop / n - so * sp / n ** 2
    return c / np.sqrt((soo / n - (so / n) ** 2) * (spp / n - (sp / n) ** 2))


def local_sums(z, X_all, blk, dev, keep):
    """Per block and neuron, the 9 sums local r needs: P, X, PP, XX, PX, P ap, P ax, X ap, X ax (ap / ax the learned /
    recorded brain mean per frame); plus per block n, ap, ax, ap ap, ax ax, ap ax. The neurons as tools/exp17_slides.local_r:
    the neurons `keep` (the common set), the brain means over all the finite ones."""
    P = z["pred"].astype(np.float64)
    fin = np.isfinite(P).all(0)
    ap = torch.as_tensor(np.nan_to_num(P[:, fin] - P[:, fin].mean(0)).mean(1), device=dev)
    ax = torch.as_tensor((X_all[:, fin] - X_all[:, fin].mean(0)).mean(1), device=dev)
    P = torch.as_tensor(P[:, keep], device=dev)
    X = torch.as_tensor(X_all[:, keep], device=dev)
    P, X = P - P.mean(0), X - X.mean(0)
    nb = int(blk.max()) + 1
    oh = torch.zeros(nb, len(blk), dtype=torch.float64, device=dev)
    oh[torch.as_tensor(blk, device=dev), torch.arange(len(blk), device=dev)] = 1.0
    per = torch.stack([oh @ q for q in (P, X, P * P, X * X, P * X, P * ap[:, None], P * ax[:, None], X * ap[:, None],
                                        X * ax[:, None])], 1)             # [nb, 9, N]
    sc = torch.stack([oh @ q for q in (torch.ones_like(ap), ap, ax, ap * ap, ax * ax, ap * ax)], 1)   # [nb, 6]
    return per, sc


def local_from(per, sc, vfull=None, per_neuron=False):
    """Local r per neuron from summed sums: per [..., 9, N], sc [..., 6]; the mean over the neurons, a FLAT learned
    residual (variance <= 1e-6 of its full-data value `vfull`, or 1e-12) scored 0, |r| <= 1. Returns (mean, the learned
    residual's variance) -- or the per-neuron r with per_neuron."""
    n = sc[..., 0:1]
    m = lambda k: per[..., k, :] / n                                    # noqa: E731
    s = lambda k: sc[..., k:k + 1] / n                                  # noqa: E731
    C = lambda a, ma, mb: a - ma * mb                                   # noqa: E731   a covariance from its sums
    cPX, cPP, cXX = C(m(4), m(0), m(1)), C(m(2), m(0), m(0)), C(m(3), m(1), m(1))
    cPap, cPax, cXap, cXax = C(m(5), m(0), s(1)), C(m(6), m(0), s(2)), C(m(7), m(1), s(1)), C(m(8), m(1), s(2))
    vap, vax, capax = s(3) - s(1) ** 2, s(4) - s(2) ** 2, s(5) - s(1) * s(2)
    bp, bx = (0.0, 0.0) if RAW else (cPap / vap, cXax / vax)          # RAW: the brain mean kept
    cov = cPX - bx * cPax - bp * cXap + bp * bx * capax
    vp, vx = cPP - bp * cPap, cXX - bx * cXax
    floor = 1e-12 if vfull is None else torch.clamp_min(1e-6 * vfull, 1e-12)
    live = (vp > floor) & (vx > 1e-12)
    r = torch.where(live, cov / torch.sqrt(vp.clamp_min(1e-30) * vx.clamp_min(1e-30)), torch.zeros_like(cov)).clamp(-1, 1)
    return r if per_neuron else (r.mean(-1), vp)


def main(laws=LAWS, tests=TESTS, seeds=SEEDS, exp_dir=EXP, raw=None, tag=""):
    global RAW, SUF
    if raw is not None:
        RAW, SUF = raw, ("_raw" if raw else "")
    from plexus.tasks import trace_recording as TR
    from plexus import trainer as T
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    rec = TR.load(T.load(laws[0][1])["task"]["reference"]["trace_recording"])
    LAWS_ = laws
    laws = tuple(laws) + tuple(seeds or ())
    Z = {k: np.load(os.path.join(T.out_dir(T.load(n), None), "results", f"{n}{v}_movie.npz")) for k, n, v in laws}
    fr = Z[LAWS_[0][0]]["frames"]
    # the frame interval of the recording, from the first law whose test json exists (<run><suffix>_test.json for a
    # checkpoint-scored law, else <run>_test.json; exp20, 2026-10-06), else the recording's own frame times
    frame_s = None
    for _, n_, v_ in laws:
        for f_ in (f"{n_}{v_}_test.json", f"{n_}_test.json"):
            tj = os.path.join(T.out_dir(T.load(n_), None), "results", f_)
            if frame_s is None and os.path.exists(tj) and "frame_s" in json.load(open(tj)):
                frame_s = float(json.load(open(tj))["frame_s"])
    if frame_s is None:
        t_ = np.asarray(rec.get("t_s", [])) if hasattr(rec, "get") else np.asarray([])
        frame_s = float(np.median(np.diff(t_))) if t_.size > 1 else FRAME_S
    f0, f1 = int(fr[0]), int(fr[-1])                                    # the free rollout's first and last frame
    nblk_t = lambda f: np.minimum((np.asarray(f) - f0) * N_BLOCKS // (f1 - f0 + 1), N_BLOCKS - 1)   # noqa: E731
    for k in Z:
        assert np.array_equal(Z[k]["frames"], fr), f"{k}: other frames"
    blk_bm = {k: brain_mean_sums(Z[k], nblk_t)[1:] for k in Z}            # block 0, the opening transient, left out
    st = nblk_t(fr) > 0
    blk = nblk_t(fr)[st] - 1
    nb = N_BLOCKS - 1
    assert all(len(v) == nb and (v[:, 0] > 0).all() for v in blk_bm.values()), "a brain-mean block is empty"
    X_all = np.asarray(rec["dff"], np.float64)[fr[st]]
    Xr = X_all - X_all.mean(0)
    Xr = Xr - np.outer(Xr.mean(1), (Xr * Xr.mean(1)[:, None]).sum(0) / (Xr.mean(1) ** 2).sum())
    keep = (Xr.std(0) > 1e-9) & np.all([np.isfinite(Z[k]["pred"]).all(0) for k in Z], 0)    # ONE neuron set for every law
    print(f"[neurons] {int(keep.sum())} of {len(keep)}: finite in every law, recorded residual moving")
    loc, est, sd_n = {}, {}, {}
    for k in Z:
        zk = {"pred": Z[k]["pred"][st]}
        per, sc = local_sums(zk, X_all, blk, dev, keep)
        r0, vfull = local_from(per.sum(0), sc.sum(0))
        rn = local_from(per.sum(0), sc.sum(0), per_neuron=True)
        loc[k] = (per, sc, vfull)
        sd_n[k] = float(rn.std())
        print(f"[local] {k}: {float(r0):+.4f} +- {sd_n[k]:.3f}; flat learned residual (scored 0): {int((rn == 0).sum())}")
    bm_blocks = {k: torch.as_tensor(v, device=dev) for k, v in blk_bm.items()}             # [nb, 6]
    g = torch.Generator(device="cpu").manual_seed(0)
    est = {k: {"brain_mean_r": float(r_from(bm_blocks[k].sum(0).cpu().numpy())),
               "local_r": float(local_from(loc[k][0].sum(0), loc[k][1].sum(0), loc[k][2])[0])} for k in Z}
    boot = {k: {"brain_mean_r": [], "local_r": []} for k in Z}
    for a in range(0, B, 250):
        b_ = min(250, B - a)
        idx = torch.randint(nb, (b_, nb), generator=g)
        W = torch.zeros(b_, nb, dtype=torch.float64)
        W.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
        W = W.to(dev)
        for k in Z:
            boot[k]["brain_mean_r"].append(r_from((W @ bm_blocks[k]).cpu().numpy()))
            per, sc, vfull = loc[k]
            boot[k]["local_r"].append(local_from(torch.einsum("bk,kmn->bmn", W, per), W @ sc, vfull)[0].cpu().numpy())
    boot = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in boot.items()}
    dur_min = (f1 - f0 + 1) * frame_s / 60              # the free rollout's length (exp20, 2026-10-06: not always 2 h)
    doc = {"null": f"two laws follow the recording equally well over the same {dur_min:.0f} min", "rollout_min": dur_min,
           "method":
           f"paired block bootstrap over time, {nb} blocks of {(f1 - f0 + 1) * frame_s / N_BLOCKS / 60:.1f} min (the "
           f"first, the opening transient, left out), {B} resamples, two-sided, the resampled differences shifted to the null",
           "block_min": (f1 - f0 + 1) * frame_s / N_BLOCKS / 60, "blocks": nb, "resamples": B,
           "neurons": int(keep.sum()),
           "laws": {}, "tests": [], "seed_pair": {}}
    for k, _, _ in laws:
        doc["laws"][k] = {m: {"estimate": est[k][m], "ci95": [float(np.percentile(boot[k][m], 2.5)),
                                                               float(np.percentile(boot[k][m], 97.5))]}
                          for m in ("brain_mean_r", "local_r")}
        doc["laws"][k]["local_r"]["sd_over_neurons"] = sd_n[k]
    for i, j in tests:
        a, b = LAWS_[i][0], LAWS_[j][0]
        t = {"a": a, "b": b}
        for m in ("brain_mean_r", "local_r"):
            d = est[a][m] - est[b][m]
            ds = boot[a][m] - boot[b][m]
            p = (1 + int((np.abs(ds - d) >= abs(d)).sum())) / (1 + B)
            t[m] = {"difference": d, "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p": p,
                    "stars": stars(p), "sd": float(ds.std()), "sigmas": float(d / ds.std())}   # d in units of its SD
        doc["tests"].append(t)
        print(f"[test] {a} - {b}: " + "; ".join(f"{m} {t[m]['difference']:+.3f} +- {t[m]['sd']:.4f} = {t[m]['sigmas']:.1f} sigma, "
                                               f"p {t[m]['p']:.1e} {t[m]['stars']}"
                                               for m in ("brain_mean_r", "local_r")))
    if seeds:
        s0, s1 = seeds[0][0], seeds[1][0]
        doc["seed_pair"] = {"a": s0, "b": s1, **{m: abs(est[s0][m] - est[s1][m]) for m in ("brain_mean_r", "local_r")}}
        print("[seed]", doc["seed_pair"])
    doc["raw"] = RAW
    doc["law_order"] = [k for k, _, _ in LAWS_]
    doc["test_pairs"] = [list(t) for t in tests]
    doc["tag"] = tag
    json.dump(doc, open(os.path.join(exp_dir, "data", f"meanfield_stats{tag}{SUF}.json"), "w"), indent=1)
    draw(doc, exp_dir)


def draw(doc, exp_dir=EXP):
    """Two panels, brain-mean r and local r: one bar per law (the deck's colour code on its value: green > 0.8, orange
    > 0.4, red), its 95 % bootstrap interval, and the three tests as brackets with their stars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    names = doc.get("law_order") or [k for k, _, _ in LAWS]
    pairs = [tuple(t) for t in doc.get("test_pairs") or TESTS]
    col = lambda v: "#4dd94d" if v > 0.8 else ("#ff9e1a" if v > 0.4 else "#ff4d40")   # noqa: E731
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 5.6), facecolor="black")
    for ax, m, ttl in zip(axs, ("brain_mean_r", "local_r"), ("a  brain-mean r", "b  per-neuron r, brain mean kept"
                                                              if doc.get("raw") else "b  per-neuron r, brain mean removed")):
        v = [doc["laws"][k][m]["estimate"] for k in names]
        lo = [v[i] - doc["laws"][k][m]["ci95"][0] for i, k in enumerate(names)]
        hi = [doc["laws"][k][m]["ci95"][1] - v[i] for i, k in enumerate(names)]
        x = np.arange(len(names))
        ax.bar(x, v, color=[col(q) for q in v], width=0.66)
        ax.errorbar(x, v, yerr=[lo, hi], fmt="none", ecolor="white", capsize=4, lw=1.2)
        for i, q in enumerate(v):
            ax.text(i, 0.02, f"{q:+.3f}", ha="center", va="bottom", color="black", fontsize=10, weight="bold")
        top = max(np.array(v) + np.array(hi))
        step = 0.085 * top
        for h, (i, j) in enumerate(sorted(pairs, key=lambda t: abs(t[1] - t[0]))):
            t = next(t for t in doc["tests"] if t["a"] == names[i] and t["b"] == names[j])
            y = top + step * (0.7 + h)
            ax.plot([i, i, j, j], [y - 0.02 * top, y, y, y - 0.02 * top], color="white", lw=1.0)
            ax.text((i + j) / 2, y + 0.005 * top, t[m]["stars"], ha="center", va="bottom", color="white", fontsize=13)
        ax.set_ylim(0, top + step * (len(pairs) + 1.0))
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(", ", ",\n").replace(" graph", "\ngraph").replace(" mean", "\nmean")
                            .replace(" no W", "\nno W") for n in names], fontsize=9)
        ax.set_title(ttl, loc="left", fontsize=12)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout()
    path = os.path.join(exp_dir, "presentation", "figs", f"meanfield_stats{doc.get('tag', '')}{'_raw' if doc.get('raw') else ''}.png")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    if os.path.isdir(os.path.join(exp_dir, "png")):
        shutil.copy(path, os.path.join(exp_dir, "png", os.path.basename(path)))
    print("[stats]", path)


if __name__ == "__main__":
    tag_ = "_n19" if "--n19" in sys.argv else ""
    if "--draw" in sys.argv:
        draw(json.load(open(os.path.join(EXP, "data", f"meanfield_stats{tag_}{SUF}.json"))))
    elif tag_:
        main(LAWS_N19, TESTS, SEEDS_N19, tag=tag_)
    else:
        main()

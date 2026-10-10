"""exp17: DOES THE GRAPH W RELAY THE STIMULUS FROM THE INPUT NEURONS TO THE REST? (Cedric, 2026-10-10: "the clamp
rollout ... launch them on 22 to 26"). The input-neuron clamp test; local.

In every law the varying stimulus enters only a set of INPUT neurons -- the `mask` array of the run's input-mask file
(batch 26: of its `video_mask` file) -- and every other neuron gets it only through W, besides a per-block offset (the 9
condition markers, or rest_block), which is not a relay. Two rollout variants of each trained law, declared in its
training spec (config/training/zapbench/<run>.yaml, task.rollouts; the trainer's `clamp: {mask:, array:}`):
    clamp_in     the input neurons given their RECORDED traces every tick, every other neuron free, the drive on
    clamp_in_W0  the same with the law's W = 0: its `zero:` list (W_short, W_mid, W_long; 23.3: A_send, every message 0)
The gap clamp_in - clamp_in_W0 on the free neurons is what W relays from the inputs. Beside them, on the SAME neurons,
the plain free rollouts: nominal and W = 0 (rollout Wall0; 23.3: W0), so the four bars compare.

THE SCORES, on the free neurons only (not input; finite in all four rollouts; recorded residual moving), over the 800
movie frames of each rollout (results/<run>[_<rollout>]_movie.npz) after block 1 of 24 (~5 min, the transient from the
recorded start, left out), as tools/exp17_meanfield_stats.py: brain-mean r = Pearson r of the learned and recorded mean
dF/F over the free neurons; per-neuron r = per free neuron, the learned and recorded traces each regressed on their own
free-neuron mean, then correlated (a flat learned residual scored 0), the mean over the free neurons. The tests: the
paired block bootstrap over time of exp17_meanfield_stats (23 blocks, 10,000 resamples, two-sided), for the clamp gap
(clamp_in - clamp_in_W0) and the plain gap (nominal - W = 0).

    PYTHONPATH=src:tools python tools/exp17_clamp.py --roll RUN [RUN ...] [--device cuda:0]   # the rollouts (trace_rollouts)
    PYTHONPATH=src:tools python tools/exp17_clamp.py RUN [RUN ...] [--device cuda:0]          # the scores and the figure
Writes data/clamp_<run>.json and presentation/figs/clamp_<run>.png. --roll makes, from the run's best model, the
rollouts of the four a run's results lack (trainer.trace_rollouts: results/<run>_<name>_free.npz / _movie.npz,
results/movie_<name>.mp4, the test json's `rollouts`); ~1-2 min per rollout.
The runs still training on 2026-10-10 (26.3 and 26.4), once results/<run>_test.json exists:
    PYTHONPATH=src:tools python tools/exp17_clamp.py --roll zap_n26_ph_vid zap_n26_ph_vid_bio && PYTHONPATH=src:tools python tools/exp17_clamp.py zap_n26_ph_vid zap_n26_ph_vid_bio
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
# the run's number in the deck and its plain W = 0 rollout (the one with the clamp_in_W0 `zero:` list)
RUNS = {"zap_n22_markall": "22.3", "zap_n23_markall": "23.3", "zap_n24_ph_edge_blk": "24.10",
        "zap_n25_bio": "25.1", "zap_n25_bio_eff": "25.2", "zap_n25_bio_eff_perm": "25.3",
        "zap_n25_ph_bio": "25.4", "zap_n25_ph_bio_eff": "25.5", "zap_n25_ph_bio_eff_perm": "25.6",
        "zap_n26_vid": "26.1", "zap_n26_vid_bio": "26.2", "zap_n26_ph_vid": "26.3", "zap_n26_ph_vid_bio": "26.4"}
ROLL = ("", "W0", "clamp_in", "clamp_in_W0")       # "": the nominal; "W0": the plain W = 0, named per run below
LABEL = {"": "full", "W0": "W = 0", "clamp_in": "clamp_in", "clamp_in_W0": "clamp_in, W = 0"}


def w0_name(spec):
    """The plain free rollout with clamp_in_W0's `zero:` list: Wall0 or W0, whichever zeroes the same learnables."""
    ro = {r["name"]: r for r in spec["task"]["rollouts"]}
    z = set(ro["clamp_in_W0"].get("zero") or [])
    return next(n for n in ("Wall0", "W0") if n in ro and set(ro[n].get("zero") or []) == z)


def roll(runs, device):
    from plexus import trainer as T
    for run in runs:
        spec = T.load(run)
        res = os.path.join(T.out_dir(spec, None), "results")
        want = [n for n in (w0_name(spec), "clamp_in", "clamp_in_W0")
                if not os.path.exists(os.path.join(res, f"{run}_{n}_movie.npz"))]
        print(f"[roll] {run}: {want or 'nothing to make'}", flush=True)
        if want:
            T.trace_rollouts(run, device=device, names=want)


def score(run, device):
    import exp17_meanfield_stats as MF
    from plexus import trainer as T
    from plexus.paths import graphs_data_path
    from plexus.tasks import trace_recording as TR
    spec = T.load(run)
    res = os.path.join(T.out_dir(spec, None), "results")
    w0 = w0_name(spec)
    names = {"": "", "W0": "_" + w0, "clamp_in": "_clamp_in", "clamp_in_W0": "_clamp_in_W0"}
    Z = {k: np.load(os.path.join(res, f"{run}{names[k]}_movie.npz")) for k in ROLL}
    fr = Z[""]["frames"]
    for k in ROLL:
        assert np.array_equal(Z[k]["frames"], fr), f"{run}{names[k]}: other frames"
    cl = Z["clamp_in"]["clamped"]
    assert np.array_equal(cl, Z["clamp_in_W0"]["clamped"]) and cl.any()
    ro = next(r for r in spec["task"]["rollouts"] if r["name"] == "clamp_in")
    m_file = np.asarray(np.load(graphs_data_path(ro["clamp"]["mask"]))[ro["clamp"].get("array", "mask")]) != 0
    assert np.array_equal(cl, m_file), "the clamped neurons are not the mask file's"
    rec = TR.load(spec["task"]["reference"]["trace_recording"])
    f0, f1 = int(fr[0]), int(fr[-1])
    nblk = lambda f: np.minimum((np.asarray(f) - f0) * MF.N_BLOCKS // (f1 - f0 + 1), MF.N_BLOCKS - 1)   # noqa: E731
    st = nblk(fr) > 0                                  # block 1, the opening transient, left out
    blk = nblk(fr)[st] - 1
    nb = MF.N_BLOCKS - 1
    X_all = np.asarray(rec["dff"], np.float64)[fr[st]]
    Xr = X_all - X_all.mean(0)
    Xr = Xr - np.outer(Xr.mean(1), (Xr * Xr.mean(1)[:, None]).sum(0) / (Xr.mean(1) ** 2).sum())
    keep = (~cl) & (Xr.std(0) > 1e-9) & np.all([np.isfinite(Z[k]["pred"]).all(0) for k in ROLL], 0)
    print(f"[{run}] free neurons scored: {int(keep.sum()):,} of {len(keep):,} ({int(cl.sum()):,} clamped)", flush=True)
    Xk = X_all[:, keep]
    o_mean = Xk.mean(1)
    dev = device if torch.cuda.is_available() else "cpu"
    bm, loc, est, sd_n = {}, {}, {}, {}
    for k in ROLL:
        P = Z[k]["pred"][st][:, keep].astype(np.float64)
        o, p = o_mean - o_mean.mean(), P.mean(1) - P.mean()
        S_ = np.zeros((nb, 6))
        for c, v in enumerate((np.ones_like(o), o, p, o * o, p * p, o * p)):
            np.add.at(S_[:, c], blk, v)
        bm[k] = torch.as_tensor(S_, device=dev)
        per, sc = MF.local_sums({"pred": P}, Xk, blk, dev, np.ones(P.shape[1], bool))   # brain means over the free neurons
        r0, vfull = MF.local_from(per.sum(0), sc.sum(0))
        sd_n[k] = float(MF.local_from(per.sum(0), sc.sum(0), per_neuron=True).std())
        loc[k] = (per, sc, vfull)
        est[k] = {"brain_mean_r": float(MF.r_from(S_.sum(0))),
                  "local_r": float(MF.local_from(per.sum(0), sc.sum(0), vfull)[0])}
        print(f"[{run}] {LABEL[k]:16s} brain-mean r {est[k]['brain_mean_r']:+.4f}  per-neuron r {est[k]['local_r']:+.4f}",
              flush=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    boot = {k: {"brain_mean_r": [], "local_r": []} for k in ROLL}
    for a in range(0, MF.B, 250):
        b_ = min(250, MF.B - a)
        idx = torch.randint(nb, (b_, nb), generator=g)
        W = torch.zeros(b_, nb, dtype=torch.float64)
        W.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))
        W = W.to(dev)
        for k in ROLL:
            boot[k]["brain_mean_r"].append(MF.r_from((W @ bm[k]).cpu().numpy()))
            per, sc, vfull = loc[k]
            boot[k]["local_r"].append(MF.local_from(torch.einsum("bk,kmn->bmn", W, per), W @ sc, vfull)[0].cpu().numpy())
    boot = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in boot.items()}
    frame_s = float(json.load(open(os.path.join(res, f"{run}_test.json")))["frame_s"])
    doc = {"run": run, "num": RUNS.get(run, run), "w0_rollout": w0, "zero": list(
               next(r for r in spec["task"]["rollouts"] if r["name"] == "clamp_in_W0").get("zero") or []),
           "mask": ro["clamp"]["mask"], "array": ro["clamp"].get("array", "mask"),
           "n_neurons": int(len(cl)), "n_clamped": int(cl.sum()), "n_free_scored": int(keep.sum()),
           "frames": {"movie": int(len(fr)), "movie_steady": int(st.sum())},
           "blocks": nb, "block_min": (f1 - f0 + 1) * frame_s / MF.N_BLOCKS / 60, "resamples": MF.B,
           "rollouts": {LABEL[k]: names[k].lstrip("_") or "nominal" for k in ROLL},
           # with every message 0 a free neuron reads nothing of the clamped ones: the two W = 0 rollouts, on the free
           # neurons, the same frames (dF/F, the movie's float16)
           "max_abs_w0_minus_clamp_in_w0": float(np.abs(Z["W0"]["pred"][:, keep].astype(np.float32)
                                                        - Z["clamp_in_W0"]["pred"][:, keep].astype(np.float32)).max()),
           "laws": {}, "tests": []}
    for k in ROLL:
        doc["laws"][LABEL[k]] = {m: {"estimate": est[k][m], "ci95": [float(np.percentile(boot[k][m], 2.5)),
                                                                        float(np.percentile(boot[k][m], 97.5))]}
                                 for m in ("brain_mean_r", "local_r")}
        doc["laws"][LABEL[k]]["local_r"]["sd_over_neurons"] = sd_n[k]
    for a, b in (("clamp_in", "clamp_in_W0"), ("", "W0")):
        t = {"a": LABEL[a], "b": LABEL[b]}
        for m in ("brain_mean_r", "local_r"):
            d = est[a][m] - est[b][m]
            ds = boot[a][m] - boot[b][m]
            p = (1 + int((np.abs(ds - d) >= abs(d)).sum())) / (1 + MF.B)
            t[m] = {"difference": d, "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "p": p,
                    "stars": MF.stars(p), "sd": float(ds.std())}
        doc["tests"].append(t)
        print(f"[{run}] {t['a']} - {t['b']}: " + "; ".join(
            f"{m} {t[m]['difference']:+.4f} p {t[m]['p']:.1e} {t[m]['stars']}" for m in ("brain_mean_r", "local_r")), flush=True)
    json.dump(doc, open(os.path.join(EXP, "data", f"clamp_{run}.json"), "w"), indent=1)
    draw(doc)
    return doc


def draw(doc):
    """Two panels, brain-mean r and per-neuron r over the free neurons: the four rollouts as bars (the deck's colour
    code on the value: green > 0.8, orange > 0.4, red), their 95 % bootstrap intervals, the two gaps as brackets with
    their stars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    names = [LABEL[k] for k in ROLL]
    col = lambda v: "#4dd94d" if v > 0.8 else ("#ff9e1a" if v > 0.4 else "#ff4d40")   # noqa: E731
    fig, axs = plt.subplots(1, 2, figsize=(12.0, 5.6), facecolor="black")
    for ax, m, ttl in zip(axs, ("brain_mean_r", "local_r"),
                          ("a  brain-mean r, free neurons", "b  per-neuron r, free neurons")):
        v = np.array([doc["laws"][k][m]["estimate"] for k in names])
        lo = np.array([doc["laws"][k][m]["ci95"][0] for k in names])
        hi = np.array([doc["laws"][k][m]["ci95"][1] for k in names])
        x = np.arange(len(names))
        ax.bar(x, v, color=[col(q) for q in v], width=0.66)
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="white", capsize=5, lw=1.4)
        base = min(0.0, float(lo.min()))
        for i, q in enumerate(v):
            ax.text(i, (0.02 if q >= 0 else q - 0.02), f"{q:+.3f}", ha="center", va="bottom" if q >= 0 else "top",
                    color="black" if q >= 0 else "white", fontsize=13, weight="bold")
        top = max(float(hi.max()), 0.05)
        step = 0.10 * (top - base)
        for h, t in enumerate(doc["tests"]):
            i, j = names.index(t["a"]), names.index(t["b"])
            y = top + step * (0.6 + 0.0 * h)
            ax.plot([i, i, j, j], [y - 0.25 * step, y, y, y - 0.25 * step], color="white", lw=1.2)
            ax.text((i + j) / 2, y + 0.05 * step, t[m]["stars"], ha="center", va="bottom", color="white", fontsize=15)
        ax.set_ylim(base - (0.05 * top if base < 0 else 0), top + step * 1.8)
        ax.axhline(0, color="white", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["full", "$W = 0$", "clamp_in", "clamp_in,\n$W = 0$"], fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(ttl, loc="left", fontsize=16)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.tight_layout()
    path = os.path.join(EXP, "presentation", "figs", f"clamp_{doc['run']}.png")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    if os.path.isdir(os.path.join(EXP, "png")):
        shutil.copy(path, os.path.join(EXP, "png", os.path.basename(path)))
    print("[clamp]", path)


if __name__ == "__main__":
    dev_ = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    runs_ = [a for a in sys.argv[1:] if a.startswith("zap_")]
    if "--roll" in sys.argv:
        roll(runs_, dev_)
    elif "--draw" in sys.argv:
        for r_ in runs_:
            draw(json.load(open(os.path.join(EXP, "data", f"clamp_{r_}.json"))))
    else:
        for r_ in runs_:
            score(r_, dev_)

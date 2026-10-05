"""exp20 BATCH 6 COMPILED (glucose fish 1): the graph (neuron graph; exp17's multi-level mesh at 3, 4, 5 levels) and the
gut-input cells (six rules) against batch 4's fish-1 reference (gb_sx_f1_mask_siren: the neuron graph, the batch 1-5
mask), each arm the same SIREN law with one thing changed.

    PYTHONPATH=src:tools python tools/exp20_batch6.py

Per run, from its own result files, exp17's results print (Cedric, 2026-10-05: "R2 changed to r, the local activity
added"; tools/exp17_slides.py bm_metrics, local_r, imported): the brain-mean r of the free rollout of the whole session
(Pearson, learned against recorded brain-mean dF/F) and the local r (per cell, learned against recorded after each is
regressed on its own brain mean; mean over the cells), the law and the same law with W = 0 at inference
(<run>_movie.npz, <run>_W0_movie.npz; the R2 kept in the json for the md's earlier findings); the gut response the free
rollout keeps -- the gut-responsive cells' evoked change, 0-20 s after a stimulated pulse minus the 10 s before, mean
over every gut pulse, the law's over the recorded (and W = 0's), from <run>_freetrial.json.
Writes data/batch6.json, presentation/figs/batch6_bars.png and batch6_traces.png.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
RUNS = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "log", "training", "gutbrain")
MASKS = [("batches 1-5", "", "#9e9e9e"), ("10 % gut-not-control", "bio", "#ff8a65"), ("paper, 2 SD", "paper2", "#ffd54f"),
         ("paper, 3 SD", "paper3", "#fff176"), ("atlas: AP + vagal ganglia", "anat_apvg", "#ef5350"),
         ("atlas: DVC", "anat_dvc", "#ab47bc")]
GRAPHS = [("neuron graph", ""), ("mesh 3", "mesh3"), ("mesh 4", "mesh4"), ("mesh 5", "mesh5")]
EV_S = 20.0


def name(graph, mask):
    if not graph and not mask:
        return "gb_sx_f1_mask_siren"                    # batch 4's fish-1 reference
    return "gb_b6_glucose_f1_" + "_".join(x for x in (graph, mask) if x)


def bm(n, stem=None, key="r2"):
    """The brain-mean metric of a run's free rollout (exp17_slides.bm_metrics): key "r2", "r" or "rmse"."""
    import exp17_slides as E17
    m = E17.bm_metrics(os.path.join(RUNS, n, "results", f"{stem or n}_movie.npz"))
    return None if m is None else float(m[key])


def local(n, stem=None):
    """The local r, the brain mean removed (exp17_slides.local_r): the mean over the cells, None without the movie npz."""
    import exp17_slides as E17
    rec = json.load(open(os.path.join(RUNS, n, "results", f"{n}_test.json"))).get("trace_recording")
    if rec not in E17._REC:
        E17._REC.clear()                       # one recording held at a time
    m = E17.local_r(os.path.join(RUNS, n, "results", f"{stem or n}_movie.npz"), rec)
    return None if m is None else m["mean"]


def gut(n):
    p = os.path.join(RUNS, n, "results", f"{n}_freetrial.json")
    if not os.path.exists(p):
        return None
    ft = json.load(open(p))
    pre = ft["window"][0]
    ev = int(round(EV_S / 1.117))
    out = {}
    for arm in ("full", "W0"):
        P = [q for q in ft["arms"].get(arm, {}).get("pulses", []) if q["site"] != 1]
        if not P:
            continue
        e = lambda key: float(np.mean([np.mean(q[key][pre:pre + ev]) - np.mean(q[key][:pre]) for q in P]))  # noqa: E731
        out[arm] = {"law": e("trace_free"), "rec": e("trace_rec"), "trace_law": np.mean([q["trace_free"] for q in P], 0).tolist(),
                    "trace_rec": np.mean([q["trace_rec"] for q in P], 0).tolist()}
    out["pre"] = pre
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for gl, g in GRAPHS:
        for ml, m, col in MASKS:
            n = name(g, m)
            if not os.path.exists(os.path.join(RUNS, n, "results", f"{n}_test.json")):
                continue
            G = gut(n)
            r = {"run": n, "graph": gl, "mask": ml, "colour": col, "brain_r2": bm(n), "brain_r2_W0": bm(n, f"{n}_W0"),
                 "brain_r": bm(n, key="r"), "brain_r_W0": bm(n, f"{n}_W0", key="r"),
                 "local_r": local(n), "local_r_W0": local(n, f"{n}_W0")}
            if G and "full" in G:
                r.update({"gut_kept": G["full"]["law"] / G["full"]["rec"], "gut_rec": G["full"]["rec"],
                          "gut_kept_W0": (G["W0"]["law"] / G["full"]["rec"]) if "W0" in G else None,
                          "trace_law": G["full"]["trace_law"], "trace_rec": G["full"]["trace_rec"], "pre": G["pre"]})
            rows.append(r)
    json.dump(rows, open(os.path.join(EXP, "data", "batch6.json"), "w"), indent=1, default=float)
    lab = [f"{r['graph']}\n{r['mask']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(3, 1, figsize=(16, 8.4), facecolor="black", sharex=True)
    for a, (k, k0, ttl) in zip(ax, (("brain_r", "brain_r_W0", "brain-mean dF/F r, free rollout of the session"),
                                    ("local_r", "local_r_W0", "local activity r, the brain mean removed (mean over the cells)"),
                                    ("gut_kept", "gut_kept_W0", "gut response kept (law / recorded, every gut pulse)"))):
        a.set_facecolor("black")
        for sp in a.spines.values():
            sp.set_color("0.6")
        a.tick_params(colors="0.85", labelsize=7)
        v = [r.get(k) if r.get(k) is not None else np.nan for r in rows]
        v0 = [r.get(k0) if r.get(k0) is not None else np.nan for r in rows]
        a.bar(x - 0.18, v, 0.36, color=[r["colour"] for r in rows], label="the law")
        a.bar(x + 0.18, v0, 0.36, color="#1e88e5", alpha=0.8, label="the same law, W = 0")
        for i, t in enumerate(v):
            if np.isfinite(t):
                a.text(i - 0.18, t + 0.02 if t >= 0 else t - 0.08, f"{t:.2f}", color="white", fontsize=7, ha="center")
        a.axhline(0, color="0.5", lw=0.6)
        a.set_title(ttl, color="white", fontsize=10, loc="left")
        a.legend(frameon=False, labelcolor="white", fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    ax[-1].set_xticks(x, lab, fontsize=7, color="0.9")
    for i in range(1, len(rows)):
        if rows[i]["graph"] != rows[i - 1]["graph"]:
            for a in ax:
                a.axvline(i - 0.5, color="0.4", lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "batch6_bars.png"), dpi=160, facecolor="black", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    # the gut response, one panel per mask rule: recorded, the neuron graph (solid), the 4-level mesh (dashed)
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.4), facecolor="black", sharey=True)
    for a, (ml, m, col) in zip(ax.ravel(), MASKS):
        a.set_facecolor("black")
        for sp in a.spines.values():
            sp.set_color("0.6")
        a.tick_params(colors="0.8", labelsize=8)
        rec_done = False
        for r in rows:
            if r["mask"] != ml or "trace_law" not in r:
                continue
            tt = (np.arange(len(r["trace_rec"])) - r["pre"]) * 1.117
            if not rec_done:
                a.plot(tt, r["trace_rec"], color="#4caf50", lw=2.0, label="recorded")
                rec_done = True
            a.plot(tt, r["trace_law"], color=col, lw=1.6, ls="-" if r["graph"] == "neuron graph" else "--",
                   label=f"{r['graph']}: {r['gut_kept']:.2f} kept")
        a.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        a.set_title(ml, color="white", fontsize=10)
        a.legend(frameon=False, labelcolor="white", fontsize=7)
        a.set_xlabel("s from a gut pulse", color="0.9", fontsize=8)
    ax[0, 0].set_ylabel("mean dF/F, gut-responsive cells", color="0.9", fontsize=8)
    ax[1, 0].set_ylabel("mean dF/F, gut-responsive cells", color="0.9", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "batch6_traces.png"), dpi=160, facecolor="black", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    for r in rows:
        f = lambda v, fm="{:+.2f}": fm.format(v) if v is not None and np.isfinite(v) else "--"   # noqa: E731
        print(f"{r['run']:38s} brain r {f(r['brain_r'])} (W0 {f(r['brain_r_W0'])})  local r {f(r['local_r'])} "
              f"(W0 {f(r['local_r_W0'])})  gut kept {f(r.get('gut_kept'), '{:.2f}')} "
              f"(W0 {f(r.get('gut_kept_W0'), '{:.2f}')})")


if __name__ == "__main__":
    main()

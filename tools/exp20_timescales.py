#!/usr/bin/env python
"""exp20: THE PAPER'S FIG. 4d-g ON THE RECORDINGS AND THE LAWS -- the time courses of the gut-responsive cells' responses,
recorded against the law's forecast from each stimulated pulse.

    PYTHONPATH=src:tools python tools/exp20_timescales.py <run> [<run> ...] [--device cuda:0]
    PYTHONPATH=src:tools python tools/exp20_timescales.py summary            # every run side by side

Chen 2026: "the responses of most neurons showed an onset time of 5 s or less, their decay times varied from ~10 s to
over a minute"; k-means on the average responses gives five clusters segregated by decay time (Fig. 4d); every region
holds cells tiling the temporal spectrum (Fig. 4e); the nodose answers shorter than most regions (Supp. Fig. 6b).

Per run: every stimulated pulse (every site but 1, off the fish) with a full window; the law rolled out from the
recorded volume at the onset over the window, the inputs only after it (exp_measures/exp20.py's trial rollout, here on
every stimulated pulse, not only the held-out ones); per gut-responsive cell, the mean over the pulses of its change
from the 10 s before, recorded and learned; then per cell
    peak     the largest change, 0 to the window's end (50 volumes)
    t_peak   the time of that maximum after the onset, s
    decay    the time from the peak to the first volume under half the peak, s; censored (the window ends first) when
             the cell stays above half to the end -- the share censored is reported, the paper's "over a minute"
and five k-means clusters of the recorded responses scaled by their own peak (the paper's Fig. 4d), sorted by decay,
with the law's mean response over the same cells; by station (tools/exp20_landmarks.py, the cells within 40 um), the
median t_peak and decay. Cells whose recorded peak is under 0.05 dF/F are left out (no response to time).
Writes experiments/exp20_gutbrain_graphcast/data/timescales_<run>.json and presentation/figs/timescales_<run>.png.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
DATA = os.path.join(EXP, "data")
CTRL_SITE = 1
MIN_PEAK = 0.05          # dF/F: a cell whose mean recorded change never reaches this has no time course to measure
STATION_UM = 40.0
K = 5


def responses(name, device):
    """[cells, post] recorded and learned mean changes over the stimulated pulses, the responsive cells' indices, dt."""
    import torch
    from plexus import engine, trainer
    spec = trainer.load(name)
    engine.quiet(True)
    out = trainer.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = trainer.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = trainer._trace_setup(spec, device)
    X, rec = box["X"], box["rec"]
    rname = spec["task"]["reference"]["trace_recording"]
    base = json.load(open(os.path.join(DATA, f"baselines_{rname}.json")))
    resp = np.where(np.load(os.path.join(DATA, f"baselines_{rname}_cells.npz"))["responsive"])[0]
    rt = torch.as_tensor(resp, device=X.device)
    pre, post = base["window"]
    sim = trainer._trace_sim(spec, False, post)
    T = X.shape[0]
    R, L, n = 0.0, 0.0, 0
    with torch.no_grad():
        for f, site in zip(rec["trials"][:, 0].astype(int), rec["trials"][:, 2].astype(int)):
            if site == CTRL_SITE or f - pre < 0 or f + post + 1 > T:
                continue
            pred = trainer._trace_rollout(sim, learn, spec, box, f, post, device, False)       # [post, N]
            b = X[f - pre:f, rt].mean(0)
            R = R + (X[f + 1:f + post + 1, rt] - b)
            L = L + (pred[:, rt] - b)
            n += 1
    dt = float(np.median(np.diff(rec["t_s"])))
    return (R / n).T.cpu().numpy(), (L / n).T.cpu().numpy(), resp, dt, n, rname


def shape(Y, dt):
    """Per row: peak, t_peak (s after the onset), decay (s from the peak to under half of it), censored flag."""
    k = Y.argmax(1)
    pk = Y[np.arange(len(Y)), k]
    dec, cen = np.zeros(len(Y)), np.zeros(len(Y), bool)
    for i in range(len(Y)):
        below = np.flatnonzero(Y[i, k[i]:] < pk[i] / 2)
        if len(below):
            dec[i] = below[0] * dt
        else:
            dec[i], cen[i] = (Y.shape[1] - k[i]) * dt, True
    return pk, (k + 1) * dt, dec, cen


def run(name, device="cuda:0"):
    from sklearn.cluster import KMeans
    from plexus.tasks import trace_recording as TR
    from exp20_landmarks import landmarks
    Rm, Lm, resp, dt, n, rname = responses(name, device)
    pk_r, tp_r, dc_r, cn_r = shape(Rm, dt)
    pk_l, tp_l, dc_l, cn_l = shape(Lm, dt)
    ok = pk_r >= MIN_PEAK
    c = lambda a, b: float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None
    res = {"run": name, "recording": rname, "pulses": n, "cells": int(len(resp)), "cells_used": int(ok.sum()), "dt": dt,
           "window_s": Rm.shape[1] * dt,
           "rec": {"t_peak_median": float(np.median(tp_r[ok])), "decay_median": float(np.median(dc_r[ok])),
                   "decay_p10": float(np.percentile(dc_r[ok], 10)), "decay_p90": float(np.percentile(dc_r[ok], 90)),
                   "censored": float(cn_r[ok].mean()), "onset_under_5s": float((tp_r[ok] <= 5.0).mean())},
           "law": {"t_peak_median": float(np.median(tp_l[ok])), "decay_median": float(np.median(dc_l[ok])),
                   "decay_p10": float(np.percentile(dc_l[ok], 10)), "decay_p90": float(np.percentile(dc_l[ok], 90)),
                   "censored": float(cn_l[ok].mean()), "peak_over_rec": float(np.median(pk_l[ok] / pk_r[ok]))},
           "r_cells": {"t_peak": c(tp_r, tp_l), "decay": c(dc_r, dc_l), "peak": c(pk_r, pk_l)}}
    # the paper's Fig. 4d: five clusters of the recorded responses, scaled by their peak, sorted by decay
    Z = Rm[ok] / pk_r[ok, None]
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(Z)
    order = np.argsort([np.median(dc_r[ok][lab == j]) for j in range(K)])
    cl = []
    for rank, j in enumerate(order):
        m = lab == j
        cl.append({"rank": rank, "cells": int(m.sum()), "decay_rec": float(np.median(dc_r[ok][m])),
                   "decay_law": float(np.median(dc_l[ok][m])), "rec": Rm[ok][m].mean(0).tolist(), "law": Lm[ok][m].mean(0).tolist()})
    res["clusters"] = cl
    # by station
    P = np.asarray(TR.load(rname.replace("_nosw", ""))["pos_view"], np.float64)[resp][:, :2]
    st = {}
    for s_, pts in landmarks(rname).items():
        if s_ == "forebrain":
            m = ok & (P[:, 0] < P[:, 0].min() + 0.25 * np.ptp(P[:, 0]))        # the head quarter of the gut cells
        else:
            d = np.min([np.linalg.norm(P - np.asarray(p_), axis=1) for p_ in pts], 0)
            m = ok & (d < STATION_UM)
        if m.sum() >= 10:
            st[s_] = {"cells": int(m.sum()), "t_peak_rec": float(np.median(tp_r[m])), "t_peak_law": float(np.median(tp_l[m])),
                      "decay_rec": float(np.median(dc_r[m])), "decay_law": float(np.median(dc_l[m]))}
    res["stations"] = st
    figure(name, res, tp_r[ok], tp_l[ok], dc_r[ok], dc_l[ok], dt)
    json.dump(res, open(os.path.join(DATA, f"timescales_{name}.json"), "w"), indent=1)
    print(f"[timescales] {name}: {ok.sum()} cells, {n} pulses; t_peak {res['rec']['t_peak_median']:.1f} s rec / "
          f"{res['law']['t_peak_median']:.1f} s law; decay {res['rec']['decay_median']:.1f} / {res['law']['decay_median']:.1f} s "
          f"(censored {res['rec']['censored']:.2f} / {res['law']['censored']:.2f}); r cells: t_peak {res['r_cells']['t_peak']:+.2f}, "
          f"decay {res['r_cells']['decay']:+.2f}, peak {res['r_cells']['peak']:+.2f}", flush=True)
    return res


def figure(name, res, tp_r, tp_l, dc_r, dc_l, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), facecolor="black")
    for a_ in ax:
        a_.set_facecolor("black")
        for sp in a_.spines.values():
            sp.set_color("0.6")
        a_.tick_params(colors="0.8", labelsize=8)
    cols = matplotlib.colormaps["plasma"](np.linspace(0.15, 0.9, len(res["clusters"])))
    for c_, col in zip(res["clusters"], cols):
        tt = (np.arange(len(c_["rec"])) + 1) * dt
        ax[0].plot(tt, c_["rec"], color=col, lw=1.8, label=f"{c_['cells']} cells, decay {c_['decay_rec']:.0f} s")
        ax[0].plot(tt, c_["law"], color=col, lw=1.0, ls="--")
    ax[0].set_title("five clusters by decay (Fig. 4d): recorded solid, the law dashed", color="white", fontsize=9)
    ax[0].set_xlabel("s after the pulse", color="0.9", fontsize=8)
    ax[0].legend(frameon=False, labelcolor="white", fontsize=7)
    hi = res["window_s"]
    ax[1].scatter(dc_r, dc_l, s=2, color="white", alpha=0.3, linewidths=0)
    ax[1].plot([0, hi], [0, hi], color="#ffd54f", lw=0.8, ls="--")
    ax[1].set_xlabel("decay, recorded (s)", color="0.9", fontsize=8)
    ax[1].set_ylabel("decay, the law (s)", color="0.9", fontsize=8)
    ax[1].set_title(f"per cell: r {res['r_cells']['decay']:+.2f}; censored at {hi:.0f} s: "
                    f"{100 * res['rec']['censored']:.0f} % rec, {100 * res['law']['censored']:.0f} % law", color="white", fontsize=9)
    b = np.arange(0, hi + dt, max(dt, 1.0))
    ax[2].hist(tp_r, b, color="#4caf50", alpha=0.6, label="recorded")
    ax[2].hist(tp_l, b, color="white", alpha=0.5, label="the law")
    ax[2].set_xlabel("time to peak after the pulse (s)", color="0.9", fontsize=8)
    ax[2].set_title(f"time to peak: median {res['rec']['t_peak_median']:.1f} s rec, {res['law']['t_peak_median']:.1f} s law",
                    color="white", fontsize=9)
    ax[2].legend(frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"timescales_{name}.png"), dpi=160, facecolor="black")
    plt.close(fig)


COND = [("glucose", "D-glucose", "#ff8a65"), ("glutamate", "glutamate", "#ffd54f"), ("Lglucose", "L-glucose", "#9e9e9e"),
        ("fish_water", "fish water", "#64b5f6"), ("blood_glucose", "blood glucose", "#ba68c8")]


def summary():
    """Every timescales_<run>.json side by side: the medians recorded against the law, the per-cell correlations.
    Writes data/timescales_all.json and presentation/figs/timescales_all.png."""
    import glob
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "timescales_*.json"))):
        if f.endswith("timescales_all.json"):
            continue
        r = json.load(open(f))
        m = re.match(r"gutbrain_(\w+?)_f(\d+)", r["recording"])
        rows.append({"condition": m.group(1), "fish": int(m.group(2)), "run": r["run"], "cells": r["cells_used"],
                     "t_peak_rec": r["rec"]["t_peak_median"], "t_peak_law": r["law"]["t_peak_median"],
                     "decay_rec": r["rec"]["decay_median"], "decay_law": r["law"]["decay_median"],
                     "decay_p90_rec": r["rec"]["decay_p90"], "censored_rec": r["rec"]["censored"],
                     "r_peak": r["r_cells"]["peak"], "r_decay": r["r_cells"]["decay"], "r_t_peak": r["r_cells"]["t_peak"]})
    json.dump(rows, open(os.path.join(DATA, "timescales_all.json"), "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="black")
    for a_ in ax:
        a_.set_facecolor("black")
        for sp in a_.spines.values():
            sp.set_color("0.6")
        a_.tick_params(colors="0.8", labelsize=8)
    for c, lab, col in COND:
        R_ = [r for r in rows if r["condition"] == c]
        if not R_:
            continue
        ax[0].scatter([r["decay_rec"] for r in R_], [r["decay_law"] for r in R_], color=col, s=26, label=lab)
        ax[1].scatter([r["t_peak_rec"] for r in R_], [r["t_peak_law"] for r in R_], color=col, s=26, label=lab)
        i = [x[0] for x in COND].index(c)
        for j, k in enumerate(("r_peak", "r_decay", "r_t_peak")):
            ax[2].scatter(np.full(len(R_), j) + (i - 2) * 0.09, [r[k] for r in R_], color=col, s=18)
    for a_, hi in ((ax[0], 16), (ax[1], 50)):
        a_.plot([0, hi], [0, hi], color="0.6", lw=0.8, ls="--")
    ax[0].set_xlabel("decay, recorded (s, median over the cells)", color="0.9", fontsize=8)
    ax[0].set_ylabel("decay, the law (s)", color="0.9", fontsize=8)
    ax[0].set_title("how long the response lasts", color="white", fontsize=10)
    ax[0].legend(frameon=False, labelcolor="white", fontsize=7)
    ax[1].set_xlabel("time to peak, recorded (s, median)", color="0.9", fontsize=8)
    ax[1].set_ylabel("time to peak, the law (s)", color="0.9", fontsize=8)
    ax[1].set_title("when it peaks", color="white", fontsize=10)
    ax[2].set_xticks([0, 1, 2], ["peak", "decay", "time to peak"], color="0.9")
    ax[2].axhline(0, color="0.5", lw=0.6)
    ax[2].set_title("per cell, recorded against the law: r", color="white", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(EXP, "presentation", "figs", "timescales_all.png"), dpi=160, facecolor="black")
    plt.close(fig)
    print(f"[timescales] summary of {len(rows)} runs -> data/timescales_all.json, figs/timescales_all.png")
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["summary"]:
        summary()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    for r_ in a.runs:
        try:
            run(r_, a.device)
        except Exception as e:                       # one run's failure must not stop the others
            print(f"[timescales] {r_} FAILED: {e!r}", flush=True)

"""exp17: EXCITATORY AND INHIBITORY NEURONS UNDER THE DALE PRIOR (Cedric, 2026-10-08: "a slide on inhibitory / excitatory:
the proportion, globally and per region"). Local, the landed runs' models/best.pt, no training.

THE SIGN of a neuron j, from the law's learned outgoing weights over every edge set (tools/exp17_dale.sender_mass):
P_j = the summed positive W out of it, N_j = the summed negative; EXCITATORY when P_j > N_j, INHIBITORY when N_j > P_j,
SILENT when P_j + N_j < SILENT (it sends nothing worth a sign). The Dale prior (min(P_j, N_j) penalised) pushes every
sender to one sign; the five folds are the same spec from five seeds (19.29-19.33, on the new nominal 19.25).
  global      per fold: the shares excitatory / inhibitory / silent; the minority mass sum min(P, N) / sum (P + N)
  agreement   of the neurons signed (not silent) in every fold: the share with the same sign in all five, against
              chance (each fold's own excitatory share, independent)
  per region  per atlas table region: the inhibitory share of its signed neurons, mean and SD over the folds
  the atlas   the same share from the Z-Brain transmitter labels (Randlett et al. 2015): of a region's neurons in a
              Gad1b or Glyt2 mask (inhibitory) or a Vglut2 mask (excitatory), the inhibitory share -- an independent
              anatomical reference, correlated with the learned one across regions

    PYTHONPATH=src:tools python tools/exp17_ei.py
-> presentation/figs/ei_dale_n19.png, data/ei_dale_n19.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FOLDS = [f"zap_n19_dale_s{k}" for k in range(5)]
SILENT = 1e-3


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from exp17_dale import sender_mass
    from exp17_tau_regions import labels
    import exp17_slides as S
    from plexus import trainer as T
    za = np.load(os.path.join(EXP, "data", "atlas_destripe.npz"))
    A, reg, names, ins = za["atlas_um"].astype(np.float64), za["regions"], [str(x) for x in za["names"]], za["inside"]
    lab, short = labels(reg, names)
    lab[~ins] = -1
    sign, minority, score = [], [], []
    for run in FOLDS:
        P, N, _ = sender_mass(run)
        s = np.where(P + N < SILENT, 0, np.where(P > N, 1, -1))
        sign.append(s)
        minority.append(float(np.minimum(P, N).sum() / max((P + N).sum(), 1e-12)))
        npz = os.path.join(T.out_dir(T.load(run), None), "results", f"{run}_movie.npz")
        score.append({"brain_mean_r": S.bm_metrics(npz)["r"], "per_neuron_r": S.local_r(npz, "zapbench_destripe")["mean"]})
        print(f"[ei] {run}: E {100 * (s == 1).mean():.1f} %, I {100 * (s == -1).mean():.1f} %, silent "
              f"{100 * (s == 0).mean():.1f} %, minority mass {100 * minority[-1]:.2f} %, brain-mean r "
              f"{score[-1]['brain_mean_r']:.3f}, per-neuron r {score[-1]['per_neuron_r']:.3f}", flush=True)
    sg = np.stack(sign)                                               # [folds, N]
    signed_all = (sg != 0).all(0)
    unanimous = (np.abs(sg[:, signed_all].sum(0)) == len(FOLDS)).mean()
    pe = [(s[s != 0] == 1).mean() for s in sg]
    chance = float(np.prod(pe) + np.prod([1 - p for p in pe]))      # five independent folds, each its own E share
    # the atlas's transmitter labels: inhibitory = Gad1b or Glyt2 masks, excitatory = Vglut2 masks
    inh_m = reg[:, [k for k, n in enumerate(names) if "Gad1b" in n or "Glyt2" in n]].any(1)
    exc_m = reg[:, [k for k, n in enumerate(names) if "Vglut2" in n or "vglut2" in n]].any(1)
    per = {}
    for k, r_ in enumerate(short):
        m = lab == k
        if m.sum() < 100:
            continue
        sh = [float((s[m & (s != 0)] == -1).mean()) if (m & (s != 0)).sum() else np.nan for s in sg]
        nl = int((m & (inh_m | exc_m)).sum())
        per[r_] = {"neurons": int(m.sum()), "inhibitory_share_mean": float(np.nanmean(sh)), "inhibitory_share_sd": float(np.nanstd(sh)),
                   "silent_share": float((sg[:, m] == 0).mean()),
                   "atlas_labelled": nl, "atlas_inhibitory_share": (float((m & inh_m & ~exc_m).sum() / max(int((m & (inh_m ^ exc_m)).sum()), 1))
                                                                   if nl >= 30 else None)}
    pairs = [(v["inhibitory_share_mean"], v["atlas_inhibitory_share"]) for v in per.values() if v["atlas_inhibitory_share"] is not None]
    r_atlas = float(np.corrcoef(*zip(*pairs))[0, 1]) if len(pairs) > 3 else None
    doc = {"folds": FOLDS, "silent_threshold": SILENT,
           "global": [{"run": r, "excitatory": float((s == 1).mean()), "inhibitory": float((s == -1).mean()),
                       "silent": float((s == 0).mean()), "minority_mass": mm, **sc}
                      for r, s, mm, sc in zip(FOLDS, sg, minority, score)],
           "signed_in_all_folds": int(signed_all.sum()), "unanimous_share": float(unanimous), "unanimous_chance": chance,
           "atlas_global_inhibitory_share": float((inh_m & ~exc_m & ins).sum() / max(int(((inh_m ^ exc_m) & ins).sum()), 1)),
           "r_learned_vs_atlas_across_regions": r_atlas, "per_region": per}
    json.dump(doc, open(os.path.join(EXP, "data", "ei_dale_n19.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("per_region",)}, indent=1))

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.4), facecolor="black")
    RED, BLUE, GREY = "#ff4a3a", "#4a9bff", "0.45"
    # a: per fold, the shares
    a = fig.add_axes([0.05, 0.60, 0.25, 0.32])
    for i, g in enumerate(doc["global"]):
        a.bar(i, 100 * g["excitatory"], color=RED, width=0.7)
        a.bar(i, 100 * g["inhibitory"], bottom=100 * g["excitatory"], color=BLUE, width=0.7)
        a.bar(i, 100 * g["silent"], bottom=100 * (g["excitatory"] + g["inhibitory"]), color=GREY, width=0.7)
    a.set_xticks(range(len(FOLDS)))
    a.set_xticklabels([f"fold {k}" for k in range(len(FOLDS))], fontsize=9)
    a.set_ylabel("neurons, %", fontsize=10)
    a.set_ylim(0, 100)
    a.set_title("a  per fold: excitatory (red), inhibitory (blue), silent (grey)", fontsize=10, loc="left")
    # b: the fold agreement and the sign map (consensus) from above
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    cons = np.where(signed_all & (sg.sum(0) == len(FOLDS)), 1, np.where(signed_all & (sg.sum(0) == -len(FOLDS)), -1, 0))
    b = fig.add_axes([0.02, 0.06, 0.32, 0.44])
    b.scatter(xd[ins & (cons == 0)], yd[ins & (cons == 0)], s=0.1, color="0.25", lw=0, rasterized=True)
    b.scatter(xd[cons == 1], yd[cons == 1], s=0.25, color=RED, lw=0, rasterized=True)
    b.scatter(xd[cons == -1], yd[cons == -1], s=0.25, color=BLUE, lw=0, rasterized=True)
    b.set_aspect("equal")
    b.axis("off")
    b.set_title(f"b  the sign all five folds agree on, from above: {100 * doc['unanimous_share']:.0f} % of the "
                f"{doc['signed_in_all_folds']:,} signed\n    in every fold (chance {100 * chance:.0f} %); grey: disagree or silent",
                fontsize=9.5, loc="left")
    # c: per region, the learned inhibitory share (mean +- SD over folds) and the atlas's
    rs = list(per)
    c = fig.add_axes([0.50, 0.08, 0.22, 0.86])
    y = np.arange(len(rs))
    c.barh(y, [100 * per[r]["inhibitory_share_mean"] for r in rs], xerr=[100 * per[r]["inhibitory_share_sd"] for r in rs],
           color=BLUE, height=0.65, error_kw={"ecolor": "white", "lw": 0.8})
    at = [(i, 100 * per[r]["atlas_inhibitory_share"]) for i, r in enumerate(rs) if per[r]["atlas_inhibitory_share"] is not None]
    if at:
        c.scatter([v for _, v in at], [i for i, _ in at], marker="D", s=26, color="#ffd24a", zorder=3,
                  label="atlas: Gad1b / Glyt2 share of labelled")
    c.set_yticks(y)
    c.set_yticklabels([f"{r} ({per[r]['neurons']:,})" for r in rs], fontsize=8.5)
    c.set_ylim(len(rs) - 0.5, -0.5)
    c.set_xlim(0, 100)
    c.set_xlabel("inhibitory, % of the signed neurons", fontsize=9.5)
    c.axvline(100 * np.mean([g["inhibitory"] / max(g["inhibitory"] + g["excitatory"], 1e-9) for g in doc["global"]]),
              color="0.7", ls="--", lw=0.8)
    c.legend(fontsize=8, frameon=False, loc="upper right")
    c.set_title("c  per region: learned (blue, mean $\\pm$ SD over the folds)", fontsize=10, loc="left")
    # d: learned vs atlas across regions
    d = fig.add_axes([0.78, 0.55, 0.20, 0.37])
    if at:
        xs_ = [per[rs[i]]["inhibitory_share_mean"] * 100 for i, _ in at]
        d.scatter([v for _, v in at], xs_, s=22, color="white")
        for (i, v), x_ in zip(at, xs_):
            d.text(v, x_, " " + rs[i].split(" (")[0][:12], fontsize=6.5, color="0.75", va="center")
        d.plot([0, 100], [0, 100], color="0.4", lw=0.7, ls=":")
    d.set_xlim(0, 100)
    d.set_ylim(0, 100)
    d.set_xlabel("atlas inhibitory share, %", fontsize=9)
    d.set_ylabel("learned inhibitory share, %", fontsize=9)
    d.set_title(f"d  across regions, r = {r_atlas:+.2f}" if r_atlas is not None else "d  across regions", fontsize=10, loc="left")
    fig.savefig(os.path.join(EXP, "presentation", "figs", "ei_dale_n19.png"), dpi=130, facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    main()

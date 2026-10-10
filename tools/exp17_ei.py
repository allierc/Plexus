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
(The Z-Brain transmitter masks are no reference for it -- Cedric, 2026-10-08: they mark a specific cell type, not every
cell of a region -- so the share they give is still written to the json, but not drawn.)

    PYTHONPATH=src:tools python tools/exp17_ei.py
-> presentation/figs/ei_dale_n<19|22>.png, data/ei_dale_n<19|22>.json  ([--n22] for batch 22's folds)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
BATCH = "22" if "--n22" in sys.argv else "19"       # Cedric, 2026-10-09: batch 22's Dale twins (22.5-22.9) with --n22
FOLDS = [f"zap_n{BATCH}_dale_s{k}" for k in range(5)]
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
           "sure_excitatory_share": float((sg[:, signed_all].sum(0) == len(FOLDS)).sum()
                                          / max(int((np.abs(sg[:, signed_all].sum(0)) == len(FOLDS)).sum()), 1)),
           "atlas_global_inhibitory_share": float((inh_m & ~exc_m & ins).sum() / max(int(((inh_m ^ exc_m) & ins).sum()), 1)),
           "r_learned_vs_atlas_across_regions": r_atlas, "per_region": per}
    json.dump(doc, open(os.path.join(EXP, "data", f"ei_dale_n{BATCH}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in doc.items() if k not in ("per_region",)}, indent=1))

    plt.style.use("dark_background")
    FW, FH = 15.0, 8.4
    fig = plt.figure(figsize=(FW, FH), facecolor="black")
    RED, BLUE = "#ff4a3a", "#4a9bff"
    # LEFT, as the tau map (Cedric, 2026-10-08: "the top and side zebrafish as in the tau map, one above the other, the
    # same size"): every neuron by the sign all five folds agree on -- red excitatory, blue inhibitory, grey otherwise
    xd, yd = A[:, 1], (621 - 1) * 0.798 - A[:, 0]
    cons = np.where(signed_all & (sg.sum(0) == len(FOLDS)), 1, np.where(signed_all & (sg.sum(0) == -len(FOLDS)), -1, 0))
    ins_ = lab >= 0
    lo_ = lambda v: float(np.percentile(v[ins_], 0.1))                   # noqa: E731
    hi_ = lambda v: float(np.percentile(v[ins_], 99.9))                  # noqa: E731
    ex, ey, ez = hi_(xd) - lo_(xd), hi_(yd) - lo_(yd), hi_(A[:, 2]) - lo_(A[:, 2]) + 40.0
    TT = 0.26
    scl = min(0.56 * FW / ex, (0.86 * FH - 2 * TT) / (ey + ez))
    wf = ex * scl / FW
    gh = (2 * TT + (ey + ez) * scl) / FH
    ycur = 0.5 + gh / 2 + 0.02
    for view, ttl in (("top", "from above, head left"), ("side", "from the side")):
        hv = (ey if view == "top" else ez) * scl / FH
        ycur -= TT / FH + hv
        a = fig.add_axes([0.01, ycur, wf, hv])
        a.axis("off")
        a.set_title(ttl + ": the cells all five folds sign alike", fontsize=14, loc="left", x=0.0, pad=2)   # panel titles at the deck's one size on the page, ~4 pt (Cedric, 2026-10-09: the modulation movie's titles)
        Y = yd if view == "top" else A[:, 2]
        a.scatter(xd[ins & (cons == 0)], Y[ins & (cons == 0)], s=0.08, color="0.13", lw=0, rasterized=True)   # the outline only
        a.scatter(xd[cons == 1], Y[cons == 1], s=0.35, color=RED, lw=0, rasterized=True)
        a.scatter(xd[cons == -1], Y[cons == -1], s=0.35, color=BLUE, lw=0, rasterized=True)
        a.set_xlim(lo_(xd), hi_(xd))
        a.set_ylim(lo_(Y) - (40.0 if view == "side" else 0.0), hi_(Y))
    nE, nI = int((cons == 1).sum()), int((cons == -1).sum())
    fig.text(0.01, ycur - 0.045, f"only the {nE + nI:,} cells all five folds sign alike ({100 * doc['unanimous_share']:.0f} %, chance "
             f"{100 * chance:.0f} %): {nE:,} excitatory (red, {100 * nE / max(nE + nI, 1):.0f} %), {nI:,} inhibitory (blue)",
             fontsize=9.5, color="0.85")
    # RIGHT: per region, the inhibitory share of its signed neurons, mean +- SD over the folds; the global share dashed
    rs = list(per)
    bx0 = 0.01 + wf + 0.17
    c = fig.add_axes([bx0, 0.10, 0.985 - bx0, 0.85])
    y = np.arange(len(rs))
    c.barh(y, [100 * per[r]["inhibitory_share_mean"] for r in rs], xerr=[100 * per[r]["inhibitory_share_sd"] for r in rs],
           color=BLUE, height=0.65, error_kw={"ecolor": "white", "lw": 0.8})
    gI = 100 * np.mean([g["inhibitory"] / max(g["inhibitory"] + g["excitatory"], 1e-9) for g in doc["global"]])
    c.axvline(gI, color="0.7", ls="--", lw=0.8)
    c.text(gI, -0.9, f"whole brain {gI:.0f} %", fontsize=8.5, ha="center", va="bottom", color="0.8")
    c.set_yticks(y)
    c.set_yticklabels([f"{r} ({per[r]['neurons']:,})" for r in rs], fontsize=8.5)
    c.set_ylim(len(rs) - 0.5, -0.5)
    c.set_xlim(0, 100)
    c.set_xlabel("inhibitory, % of the signed neurons\n(mean $\\pm$ SD over the five folds)", fontsize=9)
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"ei_dale_n{BATCH}.png"), dpi=130, facecolor="black",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

if __name__ == "__main__":
    main()

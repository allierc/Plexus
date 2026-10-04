"""exp20: THE FOUR RULES FOR THE UV INPUT CELLS of glucose fish 1 (Cedric, 2026-10-04, batch 6), drawn for the deck:

  batches 1-5     top 10 % by |t| of the change after EVERY training UV pulse, the off-fish control included (so cells
                  that answer the light alone are kept); every input enters every masked cell
  batch 6, 10 %   top 10 % by t after the GUT pulses, minus the cells the CONTROL pulses also excite (t >= 2): the
                  light-only cells dropped, but still a fixed quota; each input only its own cells (grating and gut
                  cells dissociated)
  batch 6, 2 SD   the paper's own gut-responsive selection (tools/gutbrain_baselines.py) at mode + 2 sd: no quota
  batch 6, 3 SD   the same at the paper's mode + 3 sd (the 2,538 gut-responsive cells)

  brains   figs/mask_rules_f1.png: each rule's UV cells from above (head left), the gut-responsive cells in yellow
  traces   figs/mask_rules_traces_f1.png: each rule's UV cells, their mean dF/F change around the training GUT pulses
           (red) and around the CONTROL pulses (grey), mean +- SD over the pulses -- does the rule pick cells that
           answer the gut, or the light?

    PYTHONPATH=src:tools python tools/exp20_mask_rules.py
Writes the two figures and data/mask_rules_f1.json (per rule: cells, gut-responsive inside, mean evoked change after
gut and control pulses).
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
ZF = os.path.join(os.environ["GNN_OUTPUT_ROOT"], "graphs_data", "zebrafish")
RULES = [("batches 1-5: top 10 %, |t| after every pulse (control included)", ""),
         ("batch 6: top 10 %, excited after gut pulses, not after control", "_bio"),
         ("batch 6: the paper's rule, mode + 2 SD, no quota", "_paper2sd"),
         ("batch 6: the paper's rule, mode + 3 SD = the gut-responsive cells", "_paper3sd"),
         ("batch 6: atlas, area postrema + vagal ganglia", "_anat_apvg"),
         ("batch 6: atlas, + X vagus motor + noradrenergic vagal area (DVC)", "_anat_dvc")]
PRE_S, POST_S, EV_S = 10.0, 40.0, 20.0


def main(rec="gutbrain_glucose_f1"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    R = TR.load(rec)
    P = np.asarray(R["pos_view"], np.float64)
    o = np.argsort(P[:, 2])
    X, tr, sp = R["dff"], R["trials"], R["split"]
    dt = float(np.median(np.diff(R["t_s"])))
    pre, post, ev = int(round(PRE_S / dt)), int(round(POST_S / dt)), int(round(EV_S / dt))
    resp = np.load(os.path.join(EXP, "data", f"baselines_{rec}_cells.npz"))["responsive"]
    on = [(int(f), int(s)) for f, s, h in zip(tr[:, 0], tr[:, 2], tr[:, 6]) if not h and f - pre >= 0 and f + post < len(X)]
    sets = {t: np.load(os.path.join(ZF, f"input_mask_{rec}{t}.npz"))["mask_by_input"][:, 0] > 0 for _, t in RULES}
    stats = {}
    fig, ax = plt.subplots(2, 3, figsize=(19, 6.8), facecolor="black")       # 2 x 3: the brains large (Cedric)
    ax = ax.ravel()
    for a, (lab, t) in zip(ax, RULES):
        m = sets[t]
        a.set_facecolor("black"); a.axis("off"); a.set_aspect("equal")
        a.scatter(P[o[::6], 0], P[o[::6], 1], s=0.12, c="0.28", lw=0)
        a.scatter(P[o][m[o], 0], P[o][m[o], 1], s=0.8 if m.sum() > 8000 else (1.6 if m.sum() > 1500 else 4.0), c="#ff4040", lw=0)
        a.scatter(P[resp, 0], P[resp, 1], s=1.2, c="#ffeb3b", lw=0, alpha=0.8)
        inside = int((m & resp).sum())
        a.set_title(f"{lab}\n{int(m.sum()):,} cells; {100 * inside / resp.sum():.0f} % of the {int(resp.sum()):,} "
                    "gut-responsive (yellow) inside", color="white", fontsize=10)
        stats[t or "_batch1_5"] = {"rule": lab, "cells": int(m.sum()), "gut_responsive_inside": inside}
    x0, y0 = P[:, 0].min(), P[:, 1].min() - 25
    ax[3].plot([x0, x0 + 100], [y0, y0], color="white", lw=2)
    ax[3].text(x0 + 50, y0 - 8, "100 µm", color="white", fontsize=9, ha="center", va="top")
    fig.text(0.5, 0.01, "red: the gut-input cells (the law's UV-pulse and beam inputs enter them); yellow: the gut-responsive cells; batch 6: "
             "each input only its own cells (the grating 19,035 coherent cells, the swim none); glucose fish 1 from above, "
             "head left", color="0.75", fontsize=10, ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out1 = os.path.join(EXP, "presentation", "figs", "mask_rules_f1.png")
    fig.savefig(out1, dpi=150, facecolor="black")
    plt.close(fig)
    # the traces: each rule's UV cells around the training gut and control pulses
    tt = (np.arange(-pre, post) + 0.5) * dt
    fig, ax = plt.subplots(2, 3, figsize=(18, 6.6), facecolor="black", sharey=True)
    ax = ax.ravel()
    for a, (lab, t) in zip(ax, RULES):
        m = np.where(sets[t])[0]
        a.set_facecolor("black")
        for sp_ in a.spines.values():
            sp_.set_color("0.6")
        a.tick_params(colors="0.8", labelsize=8)
        res = {}
        for kind, col, pick in (("gut", "#ff4040", lambda s: s != 1), ("control", "0.7", lambda s: s == 1)):
            Y = np.array([X[f - pre:f + post][:, m].mean(1) - X[f - pre:f][:, m].mean() for f, s in on if pick(s)])
            if not len(Y):
                continue
            mu, sd = Y.mean(0), Y.std(0)
            a.fill_between(tt, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
            a.plot(tt, mu, color=col, lw=1.8, label=f"{kind} pulses ({len(Y)})")
            res[kind] = float(np.mean(Y[:, pre:pre + ev]))
        a.axvline(0, color="#ffd54f", lw=0.8, ls=":")
        a.axhline(0, color="0.5", lw=0.5)
        a.set_title(f"{lab}\nevoked 0-20 s: gut {res.get('gut', np.nan):+.3f}, control {res.get('control', np.nan):+.3f} dF/F",
                    color="white", fontsize=8.5)
        a.set_xlabel("s from the UV pulse (training pulses)", color="0.9", fontsize=8)
        stats[t or "_batch1_5"].update({"evoked_gut": res.get("gut"), "evoked_control": res.get("control")})
    for k_ in (0, 3):
        ax[k_].set_ylabel("mean dF/F change of the rule's\ngut-input cells (mean +- SD)", color="0.9", fontsize=8)
    ax[0].legend(frameon=False, labelcolor="white", fontsize=8)
    fig.tight_layout()
    out2 = os.path.join(EXP, "presentation", "figs", "mask_rules_traces_f1.png")
    fig.savefig(out2, dpi=150, facecolor="black")
    plt.close(fig)
    for f_ in (out1, out2):
        shutil.copy(f_, os.path.join(EXP, "png", os.path.basename(f_)))
    json.dump(stats, open(os.path.join(EXP, "data", "mask_rules_f1.json"), "w"), indent=1)
    print("[mask_rules]", json.dumps(stats))


if __name__ == "__main__":
    main()

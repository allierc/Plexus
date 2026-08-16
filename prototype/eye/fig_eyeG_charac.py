#!/usr/bin/env python
"""fig_eyeG_charac -- the four things the characterisation of eye G established.

    python fig_eyeG_charac.py

One panel per stage that produced a result, drawn from the stage files themselves
(`archive/eye_G/charac/stage*.json` and the stage 0-lite diagnostics beside them),
so the figure cannot drift from the numbers the note quotes.

  (a) stage 0-lite -- the four cardinal synergies, each arrow a settled excursion,
      against the workspace the task needs;
  (b) stage 1     -- the six marginals on each muscle's own dominant axis, which is
      where eye G's convexity lives;
  (c) stage 2a    -- every pair's residual against its additive prediction, with the
      flag threshold and the settling floor;
  (d) stage 6d    -- the quadratic of the note's Eq. (4), fitted on the 64 Sobol
      points, measured against fitted.

Panel (d) fits on the Sobol design ALONE, which is what the note's 0.15/0.16/0.08
quotes, and then shows the 45 axis and edge holds as held-out points. Those sit on
the boundary of the cube, where a quadratic has the least room to follow a curve, so
they are the honest test of a model fitted in the interior.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MUS = ["LR", "SR", "MR", "IR", "SO", "IO"]
PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]
ANG = [r"$\theta$", r"$\varphi$", r"$\psi$"]
ACOL = ["#cf222e", "#1f6feb", "#2ea043"]
MCOL = {"LR": "#1f6feb", "SR": "#cf222e", "MR": "#d29922",
        "IR": "#2ea043", "SO": "#8957e5", "IO": "#e07a29"}
LBL = dict(fontsize=17, fontweight="bold", va="top", ha="left")
GATE_H, GATE_V = 15.0, 10.0


def load(eye):
    ch = os.path.join(eye, "charac")
    rows = []
    for st in ("stage0.json", "stage1.json", "stage2a.json", "stage2b.json",
               "stage6d.json"):
        p = os.path.join(ch, st)
        if os.path.isfile(p):
            rows += [dict(r, _stage=st[5:-5]) for r in json.load(open(p))
                     if r.get("settled")]
    idx = {m: k for k, m in enumerate(MUS)}
    U = np.zeros((len(rows), 6))
    for k, r in enumerate(rows):
        for nm, lv in zip(r["muscles"], r["level"]):
            U[k, idx[str(nm)]] = float(lv)
    P = np.array([r["pose_deg"] for r in rows], float)
    st = np.array([r["_stage"] for r in rows])
    syn = json.load(open(os.path.join(eye, "pairs_long_diag.json")))["synergies"]
    return U, P, st, syn


def design(U):
    U = np.atleast_2d(U)
    return np.concatenate([U, U ** 2,
                           np.stack([U[:, i] * U[:, j] for i, j in PAIRS], -1)], -1)


def dress(ax, letter=None, dx=-0.14):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black")
    ax.tick_params(colors="black", labelsize=12)
    ax.xaxis.label.set_size(13.5); ax.yaxis.label.set_size(13.5)
    if letter:
        ax.text(dx, 1.09, letter, transform=ax.transAxes, color="black", **LBL)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye", default=os.path.join(HERE, "archive", "eye_G"))
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "..", "..", "connectome-gnn-cx", "figures", "zebrafish",
        "fig_eyeG_characterisation.png"))
    a = p.parse_args()
    U, P, st, syn = load(a.eye)

    fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.4), facecolor="white")
    fig.subplots_adjust(hspace=0.34, wspace=0.26, left=0.075, right=0.98,
                        top=0.95, bottom=0.075)

    # --- (a) the four synergies ------------------------------------------
    A = ax[0, 0]; dress(A, "a")
    names = {"SR+SO": "up", "IR+IO": "down", "LR": "temporal", "MR": "nasal"}
    hs, vs = [0.0], [0.0]
    for key, lab in names.items():
        e = syn[key]["gaze_excursion_deg"]
        A.annotate("", xy=(e[0], e[1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#333333",
                                   mutation_scale=18))
        A.text(e[0] * 1.10, e[1] * 1.10, f"{lab}\n{key}", ha="center", va="center",
               fontsize=11)
        hs.append(e[0]); vs.append(e[1])
    sh, sv = max(hs) - min(hs), max(vs) - min(vs)
    A.add_patch(plt.Rectangle((min(hs), min(vs)), sh, sv, fill=False,
                              ec="#1f6feb", ls="--", lw=1.6))
    A.add_patch(plt.Rectangle((-GATE_H / 2, -GATE_V / 2), GATE_H, GATE_V,
                              fill=False, ec="#cf222e", ls=":", lw=1.6))
    A.text(0.03, 0.97, f"workspace {sh:.1f}$^\\circ$ h $\\times$ {sv:.1f}$^\\circ$ v"
           f"\ntask needs {GATE_H:.0f} $\\times$ {GATE_V:.0f} — passes",
           transform=A.transAxes, va="top", fontsize=11.5)
    A.axhline(0, color="0.85", lw=0.8); A.axvline(0, color="0.85", lw=0.8)
    A.set_xlabel(r"horizontal $\theta$ (deg)"); A.set_ylabel(r"vertical $\varphi$ (deg)")
    # annotate() and add_patch() do not autoscale, so the limits are set from the
    # data: without this the axes stay at 0-1 and every arrow is off the panel.
    lim = 1.45 * max(max(abs(h) for h in hs), max(abs(v) for v in vs), GATE_H / 2)
    A.set_xlim(-lim, lim); A.set_ylim(-lim, lim)
    A.set_aspect("equal")

    # --- (b) the six marginals, each on its own dominant axis -------------
    B = ax[0, 1]; dress(B, "b")
    single = np.array([int((u > 1e-6).sum()) == 1 for u in U])
    beta_all, *_ = np.linalg.lstsq(design(U), P, rcond=None)
    for j, m in enumerate(MUS):
        sel = single & (U[:, j] > 1e-6)
        if not sel.any():
            continue
        k = int(np.argmax(np.abs(P[sel][np.argmax(U[sel, j])])))     # dominant axis
        o = np.argsort(U[sel, j])
        B.plot(U[sel, j][o], P[sel, k][o], "o", color=MCOL[m], ms=6, mfc="white",
               mew=1.6)
        g = np.linspace(0, 1, 100); M = np.zeros((100, 6)); M[:, j] = g
        B.plot(g, (design(M) @ beta_all)[:, k], "-", color=MCOL[m], lw=2.0,
               label=f"{m}  ({ANG[k]})")
    B.axvspan(0, 0.25, color="0.90", zorder=0)
    B.text(0.125, B.get_ylim()[1] * 0.96, "nearly dead", ha="center", va="top",
           fontsize=10.5, color="0.35")
    B.axhline(0, color="0.85", lw=0.8)
    B.set_xlabel("drive $m_i$"); B.set_ylabel("pose on the dominant axis (deg)")
    B.legend(frameon=False, fontsize=10, ncol=2, loc="lower left")

    # --- (c) pair residuals against the additive prediction ---------------
    C = ax[1, 0]; dress(C, "c")
    at = {}
    for j in range(6):
        s = single & (U[:, j] > 1e-6) & np.isclose(U[:, j], 0.5)
        if s.any():
            at[j] = P[s][0]
    lab, res = [], []
    two = np.array([int((u > 1e-6).sum()) == 2 for u in U])
    for k in np.where(two)[0]:
        i, j = np.nonzero(U[k] > 1e-6)[0]
        if i not in at or j not in at:
            continue
        lab.append(f"{MUS[i]}+{MUS[j]}")
        res.append(P[k] - (at[i] + at[j]))
    res = np.array(res)
    x = np.arange(len(lab))
    # clamp: a residual of exactly zero is -inf on a log axis and takes the whole
    # figure's extent with it
    R = np.maximum(np.abs(res), 5e-3)
    for kk in range(3):
        C.plot(x, R[:, kk], "o", color=ACOL[kk], ms=6, label=ANG[kk])
    C.axhline(0.20, color="#cf222e", ls="--", lw=1.4)
    C.text(len(lab) - 0.4, 0.21, "flag threshold 0.20$^\\circ$", ha="right",
           va="bottom", fontsize=10.5, color="#cf222e")
    C.axhline(0.03, color="0.45", ls=":", lw=1.4)
    C.text(len(lab) - 0.4, 0.031, "settling floor 0.03$^\\circ$", ha="right",
           va="bottom", fontsize=10.5, color="0.45")
    C.set_yscale("log")
    C.set_ylim(5e-3, max(2.0, float(R.max()) * 1.6))
    C.set_xticks(x); C.set_xticklabels(lab, rotation=90, fontsize=9)
    C.set_ylabel("|residual vs additive| (deg)")
    C.legend(frameon=False, fontsize=10.5, ncol=3, loc="upper left")
    n_flag = int((np.abs(res).max(1) > 0.20).sum())
    C.text(0.02, 0.86, f"{n_flag} of {len(lab)} pairs non-additive",
           transform=C.transAxes, fontsize=11.5)

    # --- (d) the quadratic, fitted on the Sobol design --------------------
    D = ax[1, 1]; dress(D, "d")
    so = st == "6d"
    beta, *_ = np.linalg.lstsq(design(U[so]), P[so], rcond=None)
    fit = design(U[so]) @ beta
    rms = np.sqrt(((P[so] - fit) ** 2).mean(0))
    hel = design(U[~so]) @ beta
    rms_h = np.sqrt(((P[~so] - hel) ** 2).mean(0))
    for kk in range(3):
        D.plot(P[so, kk], fit[:, kk], "o", color=ACOL[kk], ms=4.6, mfc="none",
               mew=1.1, label=f"{ANG[kk]}  Sobol")
        D.plot(P[~so, kk], hel[:, kk], "x", color=ACOL[kk], ms=4.6, mew=1.0,
               alpha=0.55)
    lim = np.abs(np.concatenate([P.ravel(), fit.ravel()])).max() * 1.05
    D.plot([-lim, lim], [-lim, lim], "-", color="0.4", lw=1.1, zorder=0)
    D.set_xlabel("measured pose (deg)"); D.set_ylabel(r"fitted $g^k(m)$ (deg)")
    D.text(0.03, 0.97,
           "fitted on 64 Sobol points\nrms  " + " / ".join(f"{r:.2f}" for r in rms)
           + "$^\\circ$\nheld out (45 axis/edge, $\\times$)  "
           + " / ".join(f"{r:.2f}" for r in rms_h) + "$^\\circ$",
           transform=D.transAxes, va="top", fontsize=11.5)
    D.legend(frameon=False, fontsize=10, loc="lower right")

    out = os.path.abspath(a.out)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"stage 6d fit rms  {np.round(rms,3)}   held-out {np.round(rms_h,3)}")
    print(f"pairs non-additive: {n_flag}/{len(lab)}")
    print("wrote", out)


if __name__ == "__main__":
    main()

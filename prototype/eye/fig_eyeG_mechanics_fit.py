#!/usr/bin/env python
"""fig_eyeG_mechanics_fit -- true vs modelled gaze, every recorded frame.

    python fig_eyeG_mechanics_fit.py

Figure 5 (fig_eyeG_charac.py) checks the static map g against the PLATEAUS of the
holds -- one point per settled pose. It never asks the question section 4.3 of the
note is about: rolled through the fitted (C, K) as well as g, does the model track
the eye MOVING, not just where it stops?

This reads every `archive/eye_G/charac/runs/*_curves.npz` that has both `act` and
`gaze` -- currently 111 traces, all six-muscle holds recorded at full rate -- and
for each one:

  1. maps the recorded drive through g to get the commanded equilibrium at every
     frame, xi(t) = g(m(t));
  2. rolls xi(t) through the fitted (C, K) with `train_eyeG.rollout`, the same
     semi-implicit integrator the controller is trained through;
  3. compares the rolled-out gaze against the gaze the MPM eye actually reported.

Eight of the 111 traces (the six single-cardinal stage-0 holds plus the first two
of stage 1, in `train_eyeG.fit_deep`'s own sort order) are what (C, K) were FITTED
on. The other 103 are held out from the mechanics fit entirely and are the honest
test of it, the same fit/held-out split fig_eyeG_charac.py's panel (d) uses for g.

One panel per angle -- (a) theta, (b) phi, (c) psi -- true on x, modelled on y,
fontsize template matched to fig_eyeG_charac.py (Figure 5 of the note).

The (beta, C, K) fit itself is cached to `archive/eye_G/charac/plant_fit_deep.npz`
because `fit_mechanics` is 3000 Adam steps over 8 trajectories and costs several
minutes; delete the cache to refit (e.g. after the characterisation gains runs).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOT_TRACKING = os.path.join(HERE, "..", "..", "..", "connectome-gnn-cx",
                            "prototype", "dot_tracking")
sys.path.insert(0, os.path.abspath(DOT_TRACKING))
import train_eyeG as TG                                       # noqa: E402

ANG = [r"$\theta$  horizontal", r"$\varphi$  vertical", r"$\psi$  torsion"]
ACOL = ["#cf222e", "#1f6feb", "#2ea043"]
LBL = dict(fontsize=17, fontweight="bold", va="top", ha="left")
N_FIT = 8            # train_eyeG.fit_deep's own traces[:8]


def dress(ax, letter=None, dx=-0.14):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black")
    ax.tick_params(colors="black", labelsize=12)
    ax.xaxis.label.set_size(13.5); ax.yaxis.label.set_size(13.5)
    if letter:
        ax.text(dx, 1.09, letter, transform=ax.transAxes, color="black", **LBL)


def fit_cached(eye_dir, cache):
    if os.path.isfile(cache):
        z = np.load(cache)
        return z["beta"], z["C"], z["K"]
    r = TG.fit_deep(eye_dir, verbose=True)
    np.savez(cache, beta=r["beta"], C=r["C"], K=r["K"],
             static_rms_deg=r["static_rms_deg"], fit_rms_deg=r["fit_rms_deg"],
             n_holds=r["n_holds"])
    return r["beta"], r["C"], r["K"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye", default=os.path.join(HERE, "archive", "eye_G"))
    p.add_argument("--out", default=os.path.join(
        HERE, "..", "..", "..", "connectome-gnn-cx", "figures", "zebrafish",
        "fig_eyeG_mechanics_fit.png"))
    p.add_argument("--refit", action="store_true", help="ignore the cache")
    a = p.parse_args()

    cache = os.path.join(a.eye, "charac", "plant_fit_deep.npz")
    if a.refit and os.path.isfile(cache):
        os.remove(cache)
    beta, C, K = fit_cached(a.eye, cache)

    files = sorted(glob.glob(os.path.join(a.eye, "charac", "runs", "*_curves.npz")))
    files = [f for f in files
             if set(("act", "gaze")) <= set(np.load(f).files)]
    true_fit, pred_fit = [[] for _ in range(3)], [[] for _ in range(3)]
    true_ho, pred_ho = [[] for _ in range(3)], [[] for _ in range(3)]
    for i, f in enumerate(files):
        z = np.load(f)
        t, act, gaze = z["t"], np.asarray(z["act"], np.float64), \
            np.asarray(z["gaze"], np.float64)
        dt = float(np.median(np.diff(t)))
        xi = TG.quad_design(act) @ beta
        pred = TG.rollout(xi, K, C, dt, backend=np)
        dst_t, dst_p = (true_fit, pred_fit) if i < N_FIT else (true_ho, pred_ho)
        for k in range(3):
            dst_t[k].append(gaze[:, k]); dst_p[k].append(pred[:, k])

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor="white")
    fig.subplots_adjust(wspace=0.30, left=0.06, right=0.99, top=0.86, bottom=0.13)
    for k, L in enumerate("abc"):
        A = ax[k]; dress(A, L)
        tf, pf = np.concatenate(true_fit[k]), np.concatenate(pred_fit[k])
        th, ph = np.concatenate(true_ho[k]), np.concatenate(pred_ho[k])
        A.plot(th, ph, ".", color=ACOL[k], ms=2.2, alpha=0.10, mew=0,
               label=f"held out  ({len(files) - N_FIT} runs)")
        A.plot(tf, pf, ".", color="0.15", ms=2.6, alpha=0.35, mew=0,
               label=f"fitted on  ({N_FIT} runs)")
        lim = np.abs(np.concatenate([tf, pf, th, ph])).max() * 1.05
        A.plot([-lim, lim], [-lim, lim], "-", color="0.4", lw=1.1, zorder=0)
        A.set_xlim(-lim, lim); A.set_ylim(-lim, lim); A.set_aspect("equal")
        rms_f = float(np.sqrt(((tf - pf) ** 2).mean()))
        rms_h = float(np.sqrt(((th - ph) ** 2).mean()))
        A.set_title(ANG[k], fontsize=13.5, color="black", pad=8)
        A.set_xlabel("true gaze (deg)"); A.set_ylabel("modelled gaze (deg)")
        A.text(0.03, 0.97, f"rms  fit {rms_f:.2f}$^\\circ$\n"
               f"     held out {rms_h:.2f}$^\\circ$", transform=A.transAxes,
               va="top", fontsize=11.5)
        if k == 0:
            A.legend(frameon=False, fontsize=10, loc="lower right", markerscale=4)

    out = os.path.abspath(a.out)
    fig.savefig(out, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"{len(files)} traces, {N_FIT} fitted / {len(files) - N_FIT} held out")
    print("wrote", out)


if __name__ == "__main__":
    main()

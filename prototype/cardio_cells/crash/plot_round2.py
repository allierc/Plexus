"""plot_round2.py -- three panels of round 2. Reads round2_summary.json and the shard JSONs only."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RED, BLUE = "#e8503a", "#4da6ff"

S = json.load(open(os.path.join(HERE, "round2_summary.json")))
rows = S["rows"]
sh0 = json.load(open(os.path.join(HERE, "crash_round2_s0.json")))
sh1 = json.load(open(os.path.join(HERE, "crash_round2_s1.json")))
ro = {**sh1["rollouts"], **sh0["rollouts"]}
a_ref = np.array(sh0["a_ref_percell"], dtype=float)
keep = np.array(sh0["keep_percell"], dtype=bool)
ar = a_ref[keep] / a_ref[keep].mean()

plt.rcParams.update({"font.size": 9, "text.color": "w", "axes.labelcolor": "w",
                     "xtick.color": "w", "ytick.color": "w", "axes.edgecolor": "w"})
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1), facecolor="k")
for a in ax:
    a.set_facecolor("k")


def lab(a, s):
    a.text(0.02, 0.97, s, transform=a.transAxes, color="w", fontsize=11, va="top", ha="left")


er_r = np.array([r["raw_E_ratio"] for r in rows])
er_g = np.array([r["gauged_E_ratio"] for r in rows])
ls_r = np.array([r["raw_loopscore"] for r in rows])
ls_g = np.array([r["gauged_loopscore"] for r in rows])
me = np.array([r["med_E"] for r in rows])

ax[0].scatter(np.log10(er_r), ls_r, c=RED, s=34, label="raw")
ax[0].scatter(np.log10(er_g), ls_g, c=BLUE, s=34, marker="s", label="gauge-fixed")
x = np.linspace(np.log10(er_r).min(), np.log10(er_r).max(), 20)
p = np.polyfit(np.log10(er_r), ls_r, 1)
ax[0].plot(x, np.polyval(p, x), color=RED, lw=1, ls="--")
ax[0].set_xlabel("log10 interior motion-energy ratio")
ax[0].set_ylabel("loopscore (margin 20)")
ax[0].legend(facecolor="k", edgecolor="w", labelcolor="w", loc="lower left", fontsize=8)
lab(ax[0], f"a   amplitude axis: corr {S['regressions']['raw']['corr_loopscore_vs_log_E_ratio']:+.3f}"
    f" -> {S['regressions']['gauged']['corr_loopscore_vs_log_E_ratio']:+.3f}")

ax[1].scatter(me, ls_r, c=RED, s=34, label="raw")
ax[1].scatter(me, ls_g, c=BLUE, s=34, marker="s", label="gauge-fixed")
for r in rows:
    if r["name"] in ("true_gain_x1.8", "blind_E40_g1", "theta_hat_frame_ridge0",
                     "blind_E130_g0.95", "frame_DISP_oracle_rescale"):
        ax[1].annotate(r["name"].replace("theta_", ""), (r["med_E"], r["gauged_loopscore"]),
                       color="w", fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax[1].axhline(0.2917, color="w", lw=0.8, ls=":")
ax[1].text(0.55, 0.30, "replay bar (synthetic)", color="w", fontsize=7)
ax[1].set_xlabel("median |dE/E| (per-cell parameter error)")
ax[1].set_ylabel("loopscore (margin 20)")
ax[1].legend(facecolor="k", edgecolor="w", labelcolor="w", loc="lower left", fontsize=8)
lab(ax[1], f"b   per-cell axis: corr {S['regressions']['raw']['corr_loopscore_vs_med_E']:+.3f}"
    f" -> {S['regressions']['gauged']['corr_loopscore_vs_med_E']:+.3f}")

for nm, col, mk in (("blind_E130_g0.95", RED, "o"), ("theta_hat_frame_ridge0", "#f2c14e", "^"),
                    ("frame_DISP", BLUE, "s")):
    a = np.array(ro[nm]["gauged"]["a_percell"], dtype=float)[keep]
    a = a / a.mean()
    sk = [r for r in rows if r["name"] == nm][0]["gauged_percell_skill_vs_blind"]
    ax[2].scatter(ar, a, s=18, c=col, marker=mk, alpha=0.8, label=f"{nm}  skill {sk:+.2f}")
ax[2].plot([0, ar.max() * 1.05], [0, ar.max() * 1.05], color="w", lw=0.8)
ax[2].set_xlabel("reference per-cell peak amplitude / mean")
ax[2].set_ylabel("model per-cell peak amplitude / mean")
ax[2].legend(facecolor="k", edgecolor="w", labelcolor="w", loc="lower right", fontsize=7)
lab(ax[2], "c   per-cell amplitude field, all gauge-fixed")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "crash_round2.png"), dpi=150, facecolor="k")
print("wrote crash_round2.png")

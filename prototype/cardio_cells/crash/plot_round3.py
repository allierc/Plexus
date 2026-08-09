"""plot_round3.py -- three panels from the round-3 artefacts. No new numbers are computed."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{H}/round3_summary.json"))
FI = json.load(open(f"{H}/finject.json"))
TH = json.load(open(f"{H}/finject_thresh.json"))
NO = json.load(open(f"{H}/finject_noise.json"))
rows, bank = S["rows"], S["null_band"]["members"]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))

# (a) the null band and where the candidates fall
conv = [n for n in bank if rows[n]["gauge_status"] == "converged" or rows[n]["n_extra"] == 0]
lo, hi = min(rows[n]["gau_loop"] for n in conv), max(rows[n]["gau_loop"] for n in conv)
names = sorted(rows, key=lambda k: rows[k]["gau_loop"])
y = np.arange(len(names))
col = ["0.55" if n in bank else ("tab:green" if rows[n]["gau_loop"] >= hi else "tab:red")
       for n in names]
ax[0].axhspan(lo, hi, color="0.85", zorder=0)
ax[0].bar(y, [rows[n]["gau_loop"] for n in names], color=col)
ax[0].set_xticks(y)
ax[0].set_xticklabels([("* " if n in bank else "") + n for n in names], rotation=90, fontsize=6)
ax[0].set_ylabel("gauged loopscore (2-D gauge, margin 20)")
ax[0].text(0.02, 0.96, "a  grey = zero-information NULL BANK; band = its converged span",
           transform=ax[0].transAxes, va="top", fontsize=9, fontweight="bold")

# (b) the F-injection ladder
V = ["none", "F_hold", "F_lerp", "F_true", "C_hold", "C_lerp", "C_true",
     "FC_hold", "FC_lerp", "FC_true"]
me = [FI["variants"][v]["scores"]["ridge0"]["med_E"] for v in V]
ls = [FI["rollouts"][v]["raw"]["margin20"]["loopscore"] for v in V]
b = ax[1].bar(np.arange(len(V)), me, color=["0.4"] + ["tab:blue"] * 9)
ax[1].set_yscale("log"); ax[1].set_xticks(np.arange(len(V)))
ax[1].set_xticklabels(V, rotation=90, fontsize=8)
ax[1].set_ylabel("med |dE/E| after least squares")
for i, (v, l) in enumerate(zip(V, ls)):
    ax[1].text(i, me[i] * 1.15, f"{l:.3f}", ha="center", fontsize=6.5, rotation=90)
ax[1].text(0.02, 0.96, "b  frame-cadence recovery with state injection\n"
                       "    (number above bar = raw loopscore of its rollout)",
           transform=ax[1].transAxes, va="top", fontsize=9, fontweight="bold")

# (c) how accurate must F be
for mode, c in (("indep", "tab:blue"), ("cell", "tab:orange")):
    r = [v for v in TH["rows"].values() if v["mode"] == mode and v["sigma_F"] > 0]
    r.sort(key=lambda v: v["sigma_F"])
    ax[2].loglog([v["sigma_F"] for v in r], [v["med_E"] for v in r], "o-", color=c, label=mode)
ax[2].axvline(NO["real_error_bars"]["sigma_F_temporal"], color="k", ls="--")
ax[2].axvline(NO["real_error_bars"]["sigma_F_systematic"], color="k", ls=":")
ax[2].text(NO["real_error_bars"]["sigma_F_temporal"], 3e-3, " recording: temporal noise",
           rotation=90, fontsize=7, va="bottom")
ax[2].text(NO["real_error_bars"]["sigma_F_systematic"], 3e-3, " recording: two estimates disagree",
           rotation=90, fontsize=7, va="bottom")
ax[2].axhline(0.05, color="0.6", lw=0.8)
ax[2].set_xlabel("sigma_F  (Frobenius, per node)")
ax[2].set_ylabel("med |dE/E|")
ax[2].legend(fontsize=8)
ax[2].text(0.02, 0.96, "c  how accurate the measured F has to be",
           transform=ax[2].transAxes, va="top", fontsize=9, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{H}/crash_round3.png", dpi=150)
print("wrote crash_round3.png")

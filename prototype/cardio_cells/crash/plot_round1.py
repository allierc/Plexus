"""plot_round1.py -- the four pictures of round 1, read off crash_round1.json / theta_round1.npz.

Nothing is computed here that is not already in those files.
  PYTHONPATH=/workspace/Plexus/src python plot_round1.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "crash_round1.json")))
Z = np.load(os.path.join(HERE, "theta_round1.npz"))
C = R["C"]
th = Z["cand::theta_true"]

fig, ax = plt.subplots(1, 4, figsize=(19.5, 4.6), facecolor="black")
for a in ax:
    a.set_facecolor("black")
    for sp in a.spines.values():
        sp.set_color("white")
    a.tick_params(colors="white", labelsize=9)
    a.xaxis.label.set_color("white")
    a.yaxis.label.set_color("white")


def lab(a, s):
    a.set_title(s, color="white", fontsize=11, fontweight="bold", loc="left", pad=8)


def leg(a):
    a.legend(fontsize=8, facecolor="black", edgecolor="white", labelcolor="white")


# a -- what the two cadences recover
ax[0].plot([0, 230], [0, 230], "-", color="0.5", lw=1)
ax[0].plot(th[:C], Z["cand::theta_hat_substep"][:C], ".", color="#4da6ff", ms=6,
           label="substep cadence (dt_sub)")
ax[0].plot(th[:C], Z["cand::theta_hat_frame_ridge0"][:C], ".", color="#ff5555", ms=6,
           label="frame cadence (dt = 10 substeps)")
ax[0].axhline(0, color="0.35", lw=0.8, ls=":")
ax[0].set_xlabel("planted E"); ax[0].set_ylabel("recovered E")
ax[0].set_xlim(0, 230); ax[0].set_ylim(-20, 240)
ax[0].legend(fontsize=8, facecolor="black", edgecolor="white", labelcolor="white",
             loc="lower right")
lab(ax[0], "a  exact per substep, biased low per frame")

# b -- equation error -> output error
names = [n for n in R["candidates"]]
xs = [max(R["candidates"][n]["med_E"], 3e-7) for n in names]
ls = [R["rollouts"][n + "|free"]["margin20"]["loopscore"] for n in names]
r2 = [R["rollouts"][n + "|free"]["coarse"]["R2_displacement_interior"] for n in names]
jit = [n for n in names if "jitter" in n]
col = ["#ffd24d" if "jitter" in n else ("#ff5555" if "frame" in n else "#4da6ff") for n in names]
ax[1].semilogx(xs, ls, "o", ms=7, mfc="none", mew=1.6)
for x, y, n, c in zip(xs, ls, names, col):
    ax[1].plot([x], [y], "o", color=c, ms=7)
ax[1].axhline(R["nulls"]["do_nothing"]["loopscore"], color="#aaaaaa", lw=1, ls="--")
ax[1].text(4e-7, R["nulls"]["do_nothing"]["loopscore"] + 0.02, "do-nothing null", color="#aaaaaa",
           fontsize=8)
ax[1].axhline(R["nulls"]["replay_previous_beat"]["loopscore"], color="#88ff88", lw=1, ls=":")
ax[1].text(4e-7, R["nulls"]["replay_previous_beat"]["loopscore"] - 0.07, "replay previous beat",
           color="#88ff88", fontsize=8)
KEEP = {"theta_hat_substep": "hat_substep", "theta_hat_frame_ridge0": "hat_frame",
        "theta_hat_frame_ridge1e-2": "hat_frame+ridge", "theta_const_E130_g1": "const E=130",
        "theta_shuffled_true": "shuffled", "theta_true_perturbed_0.3": "theta_true +-30%",
        "theta_true_x0jitter_0.1dx": "x0 jitter 0.1 dx"}
for n, x, y in zip(names, xs, ls):
    if n in KEEP:
        ax[1].annotate(KEEP[n], (x, y), color="white", fontsize=7.5, xytext=(5, -11),
                       textcoords="offset points")
ax[1].set_xlabel("parameter error, median |dE/E|")
ax[1].set_ylabel("loopscore of the 150-frame rollout (margin 20)")
lab(ax[1], "b  equation error -> output error")

# c -- how the trajectory error grows
show = [("theta_hat_substep", "#00ff9f"), ("theta_true_perturbed_0.03", "#4da6ff"),
        ("theta_true_perturbed_0.3", "#9ecbff"), ("theta_const_E130_g1", "#ffffff"),
        ("theta_shuffled_true", "#cc99ff"), ("theta_hat_frame_ridge0", "#ff5555"),
        ("theta_true_x0jitter_0.1dx", "#ffd24d")]
for n, c in show:
    y = R["rollouts"][n + "|free"]["coarse"]["rms_pos_err_dx_per_frame"]
    ax[2].semilogy(np.arange(1, len(y) + 1), np.maximum(y, 1e-16), "-", color=c, lw=1.4,
                   label=n.replace("theta_", ""))
ax[2].axvspan(0, 15, color="0.25", alpha=0.5, lw=0)
ax[2].text(2, 2e-4, "pulse on", color="0.75", fontsize=8, rotation=90)
ax[2].set_xlabel("frame of the rollout"); ax[2].set_ylabel("rms position error / grid cell")
ax[2].set_ylim(1e-4, 3)
leg(ax[2])
lab(ax[2], "c  the error saturates; it does not blow up")

# d -- what the anchor buys
bx = [R["rollouts"][n + "|free"]["coarse"]["rms_pos_err_dx_BAND_mean"] for n in names]
d20 = [R["rollouts"][n + "|anchored"]["margin20"]["loopscore"]
       - R["rollouts"][n + "|free"]["margin20"]["loopscore"] for n in names]
d10 = [R["margin10"][n + "|anchored"]["loopscore"] - R["margin10"][n + "|free"]["loopscore"]
       for n in names]
ax[3].plot(np.maximum(bx, 1e-6), d10, "o", color="#ff5555", ms=7, label="margin 10 (36/100 pinned)")
ax[3].plot(np.maximum(bx, 1e-6), d20, "o", color="#4da6ff", ms=7, label="margin 20 (0/100 pinned)")
ax[3].set_xscale("log")
ax[3].axhline(0, color="0.5", lw=1)
ax[3].set_xlabel("free-run error inside the band it pins (grid cells)")
ax[3].set_ylabel("loopscore gained by anchoring")
leg(ax[3])
lab(ax[3], "d  the anchor mends only what it touches")

fig.tight_layout()
p = os.path.join(HERE, "crash_round1.png")
fig.savefig(p, dpi=130, facecolor="black")
print("wrote", p)

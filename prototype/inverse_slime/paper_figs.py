"""paper_figs.py -- figures for inverse_slime.tex.

  figures/fig_param_recovery.pdf : true vs recovered parameters (explicit), single-type
                                   (cross held) and 4-type (cross recovered).
  figures/fig_evolution.pdf      : GT vs position-loss recovery, cells + field, 5 frames.
Convention: GT/true = black, recovered/predicted = green (the repo plot convention).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
NAMES = ["deposit\namount", "diffuse\nrate", "decay\nrate", "sense\ncross",
         "move\nspeed", "turn\nspeed", "sensor\nangle", "sensor\ndist"]

# measured recoveries (one-step position loss; u-space)
SINGLE_TRUE = [0.851, 0.991, 0.466, 0.597, 0.257, 0.167, 0.877, 0.182]
SINGLE_REC = [0.851, 0.985, 0.466, 0.597, 0.257, 0.121, 0.891, 0.214]   # cross held = true
FOUR_TRUE = [0.625, 0.897, 0.776, 0.225, 0.300, 0.874, 0.005, 0.821]
FOUR_REC = [0.625, 0.897, 0.776, 0.233, 0.300, 0.780, 0.078, 0.860]


def fig_param_recovery():
    import json
    pj = os.path.join(HERE, "runs", "posonly_full.json")
    panels = [("(a) full recovery -- field loss + position loss\n(cross HELD at GT, not learned: single-type)",
               SINGLE_TRUE, SINGLE_REC, "cross_held"),
              ("(b) four-type slime\n(cross now LEARNED)", FOUR_TRUE, FOUR_REC, "cross_rec")]
    if os.path.exists(pj):
        d = json.load(open(pj))
        panels.append(("(c) POSITION-ONLY loss, no field loss\n(field unobserved: field params fail)",
                       d["u_true"], d["u_rec"], "field_fail"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.7))
    for ax, (title, tru, rec, mode) in zip(np.atleast_1d(axes), panels):
        x = np.arange(8); w = 0.38
        rec = list(rec)
        ax.bar(x - w / 2, tru, w, color="black", label="true")
        # recovered bars green, EXCEPT the held cross bar (panel a) drawn gray/hatched
        learned = [i for i in range(8) if not (mode == "cross_held" and i == 3)]
        ax.bar(x[learned] + w / 2, [rec[i] for i in learned], w, color="#1aa64b", label="recovered")
        if mode == "cross_held":
            ax.bar(3 + w / 2, rec[3], w, color="0.6", hatch="////", edgecolor="white",
                   label="held at GT (not learned)")
            ax.axvspan(2.5, 3.5, color="0.85", alpha=0.5)
        elif mode == "cross_rec":
            ax.axvspan(2.5, 3.5, color="gold", alpha=0.14)
            ax.annotate("cross\nLEARNED", (3, 1.02), ha="center", va="top", fontsize=7, color="#7a5a00")
        else:                                                        # field params fail
            ax.axvspan(-0.5, 2.5, color="red", alpha=0.08)
            ax.annotate("field\n(not identifiable\nfrom positions)", (1, 1.02), ha="center",
                        va="top", fontsize=7, color="#a00")
        ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=7.5)
        ax.set_ylim(0, 1.05); ax.set_ylabel("parameter value (unit-cube u)", fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_param_recovery.pdf")); plt.close(fig)


def fig_evolution():
    rows = [("slime_gt", "GT  cells"), ("slime_gd8d_v2", "recovered  cells"),
            ("slime_gt", "GT  field"), ("slime_gd8d_v2", "recovered  field")]
    chans = ["cell", "cell", "field", "field"]
    fig, axes = plt.subplots(4, build_pdf.NCOL, figsize=(build_pdf.NCOL * 1.9, 4 * 1.9),
                             facecolor="white", squeeze=False)
    for r, ((folder, label), ch) in enumerate(zip(rows, chans)):
        build_pdf.draw_row(axes[r], folder, label, ch)
    for c in range(build_pdf.NCOL):
        axes[0][c].set_title(f"t={build_pdf.FRAMES[c]}", fontsize=8)
    fig.suptitle("ground truth vs recovered (position loss) -- topologically matched, not pixel-identical",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(FIG, "fig_evolution.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig_param_recovery(); fig_evolution()
    print("wrote figures/fig_param_recovery.pdf and figures/fig_evolution.pdf")

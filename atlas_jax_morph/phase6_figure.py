"""phase6_figure -- does averaging seeds fix the straight-through gradient?

The Phase 6 sweep asks one question with a number attached: the division draw is discrete, so a
straight-through gradient through one sampled rollout is a gradient through that sample's luck.
Fit the same target K times over, averaging K seeds per Adam step, across 8 independent
replicates, and plot what the answer does.

Two things have to be read separately, and conflating them is how this would be over-sold:

  the SPREAD   sigma vs K on log-log, against the 1/sqrt(K) line a plain Monte-Carlo average
               would give. This is about precision.
  the CENTRE   where the fits actually land at each K. This is about correctness, and it is the
               one that turned out to matter.

    python phase6_figure.py        # -> _state/phase6_variance.png
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "_state", "phase6_variance.json")
OUT = os.path.join(HERE, "_state", "phase6_variance.png")

BG = "black"
C_OBS = "#4FA3FF"      # what we measured
C_REF = "#FF6B6B"      # the 1/sqrt(K) reference the measurement is judged against


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.load(open(SRC))
    S = {r["K"]: r for r in d["summary"]}
    ks = sorted(S)
    sig = [S[k]["stdev"] for k in ks]
    mean = [S[k]["mean"] for k in ks]
    sem = [S[k]["stdev"] / math.sqrt(S[k]["n"]) for k in ks]

    xs = [math.log(k) for k in ks]
    ys = [math.log(s) for s in sig]
    xb, yb = st.fmean(xs), st.fmean(ys)
    a = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=BG)

    # ---- left: precision -------------------------------------------------------------------
    axL.set_facecolor(BG)
    axL.plot(ks, sig, "o-", color=C_OBS, lw=2.2, ms=8, label=f"measured  ($\\sigma \\propto K^{{{a:.2f}}}$)")
    axL.plot(ks, [sig[0] / math.sqrt(k) for k in ks], "--", color=C_REF, lw=1.8,
             label="$1/\\sqrt{K}$  (plain Monte-Carlo)")
    axL.set_xscale("log", base=2), axL.set_yscale("log")
    axL.set_xticks(ks), axL.set_xticklabels([str(k) for k in ks])
    axL.set_xlabel("K  (seeds averaged per Adam step)", color="white", fontsize=10)
    axL.text(0.03, 0.06, "spread of the fitted value across 8 replicates",
             transform=axL.transAxes, color="white", fontsize=11, va="bottom", ha="left")

    # ---- right: correctness ----------------------------------------------------------------
    axR.set_facecolor(BG)
    for i, k in enumerate(ks):
        vals = S[k]["values"]
        axR.scatter([k] * len(vals), vals, s=26, color=C_OBS, alpha=0.55, linewidths=0,
                    label="individual fits" if i == 0 else None)
    axR.errorbar(ks, mean, yerr=sem, fmt="o-", color="white", lw=2.0, ms=7, capsize=4,
                 label="mean $\\pm$ s.e.m.")
    axR.axhline(mean[-1], color=C_REF, ls="--", lw=1.4, label="K=8 mean")
    axR.set_xscale("log", base=2)
    axR.set_xticks(ks), axR.set_xticklabels([str(k) for k in ks])
    axR.set_xlabel("K  (seeds averaged per Adam step)", color="white", fontsize=10)
    axR.text(0.03, 0.06, "where the fits land  —  the centre moves too",
             transform=axR.transAxes, color="white", fontsize=11, va="bottom", ha="left")

    for ax in (axL, axR):
        ax.tick_params(colors="white", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#444444")
        leg = ax.legend(loc="upper right", fontsize=9, facecolor=BG, edgecolor="#444444")
        for t in leg.get_texts():
            t.set_color("white")

    fig.text(0.008, 0.985,
             "Phase 6 — fitting grow_radius.max_radius through 24 frames of real physics, "
             "32 runs on gpu_l4 (K ∈ {1,2,4,8} × 8 replicates)",
             color="white", fontsize=11, va="top", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"sigma ~ K^{a:.3f}   -> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""phase6_verdict -- was the K=1 drift a biased estimator, or just a truncated budget?

The 20-step sweep found two things: the spread of the fitted value falls like 1/sqrt(K) (good --
the straight-through gradient behaves like a noisy estimator, not a broken one), and the CENTRE
moves, K=1 landing 6% below K=8 at t = 2.90. The second is the one that matters, and it has two
possible causes that the first sweep could not tell apart:

  BIASED ESTIMATOR    the straight-through gradient through a discrete division systematically
                      points somewhere other than downhill, and no step budget fixes it. The
                      reference's trace/replay/score contract is then REQUIRED.
  TRUNCATED BUDGET    the estimator is fine, but a noisier gradient converges more slowly, so at
                      a fixed 20 steps the K=1 runs simply had not arrived yet. Then Figure 5 is
                      an engineering problem and the contract is an optimisation.

The discriminator is to run the same grid four times longer and Polyak-average the tail. If the
drift collapses it was the budget; if it survives at 80 steps it is the estimator.

    python phase6_verdict.py --tail 30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "_state")
OUT = os.path.join(STATE, "phase6_verdict.png")


def load(path, tail):
    """Per-K mean/stdev, Polyak-averaging the last `tail` steps of each replicate."""
    d = json.load(open(path))
    by_k = {}
    for r in d["runs"]:
        key = list(r["final"])[0]
        hist = [h["params"][key] for h in r["history"]]
        val = st.fmean(hist[-tail:]) if (tail and hist) else r["final"][key]
        by_k.setdefault(r["K"], []).append(val)
    return {k: {"n": len(v), "mean": st.fmean(v),
                "sd": st.stdev(v) if len(v) > 1 else float("nan"), "values": v}
            for k, v in sorted(by_k.items())}


def drift(S):
    """K=1 vs K=8: the difference, and how many standard errors it is."""
    a, b = S.get(1), S.get(max(S))
    if not a or not b or a is b:
        return None
    se = math.sqrt(a["sd"] ** 2 / a["n"] + b["sd"] ** 2 / b["n"])
    return {"lo_K": 1, "hi_K": max(S), "diff": b["mean"] - a["mean"], "se": se,
            "t": (b["mean"] - a["mean"]) / se if se else float("nan")}


def slope(S):
    xs = [math.log(k) for k in S]
    ys = [math.log(S[k]["sd"]) for k in S]
    xb, yb = st.fmean(xs), st.fmean(ys)
    return sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=30)
    a = ap.parse_args()

    runs = {"20 steps (final value)": (os.path.join(STATE, "phase6_variance.json"), 0),
            f"80 steps (tail-{a.tail} mean)": (os.path.join(STATE, "phase6_variance_conv.json"),
                                               a.tail)}
    loaded = {}
    for label, (path, tail) in runs.items():
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        loaded[label] = load(path, tail)

    for label, S in loaded.items():
        print(f"\n=== {label} ===")
        print(f"{'K':>3} {'n':>3} {'mean':>9} {'sem':>8} {'sd':>8}")
        for k, r in S.items():
            print(f"{k:>3} {r['n']:>3} {r['mean']:>9.4f} {r['sd']/math.sqrt(r['n']):>8.4f} "
                  f"{r['sd']:>8.4f}")
        print(f"  sigma ~ K^{slope(S):.3f}   (Monte-Carlo: -0.500)")
        d = drift(S)
        if d:
            print(f"  drift K=1 -> K={d['hi_K']}: {d['diff']:+.4f} +/- {d['se']:.4f}  "
                  f"t = {d['t']:.2f}")

    if len(loaded) == 2:
        (l0, S0), (l1, S1) = list(loaded.items())
        d0, d1 = drift(S0), drift(S1)
        print("\n" + "=" * 68)
        print(f"VERDICT   drift at 20 steps: t = {d0['t']:.2f}   "
              f"at 80 steps: t = {d1['t']:.2f}")
        if abs(d1["t"]) < 2.0:
            print("  the drift COLLAPSED with a longer budget -> it was TRUNCATION, not bias.")
            print("  The straight-through estimator is sound; Figure 5 is an engineering problem")
            print("  and the trace/replay/score contract is an optimisation, not a prerequisite.")
        else:
            print("  the drift SURVIVED a 4x longer budget -> the estimator is BIASED.")
            print("  More seeds buy precision but not correctness; the reference's")
            print("  trace/replay/score contract is required for Figure 5.")
        print("=" * 68)

    # ---- figure ---------------------------------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG = "black"
    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=BG)
    ax.set_facecolor(BG)
    for (label, S), colour in zip(loaded.items(), ("#FF6B6B", "#4FA3FF")):
        ks = list(S)
        m = [S[k]["mean"] for k in ks]
        e = [S[k]["sd"] / math.sqrt(S[k]["n"]) for k in ks]
        ax.errorbar(ks, m, yerr=e, fmt="o-", lw=2.0, ms=7, capsize=4, color=colour, label=label)
    ax.set_xscale("log", base=2)
    ks = list(next(iter(loaded.values())))
    ax.set_xticks(ks), ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("K  (seeds averaged per Adam step)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for s in ax.spines.values():
        s.set_color("#444444")
    leg = ax.legend(loc="lower right", fontsize=9, facecolor=BG, edgecolor="#444444")
    for t in leg.get_texts():
        t.set_color("white")
    ax.text(0.03, 0.95, "does the centre still move with a 4x longer budget?",
            transform=ax.transAxes, color="white", fontsize=11, va="top", ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=135, facecolor=BG)
    plt.close(fig)
    print(f"\n-> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

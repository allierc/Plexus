#!/usr/bin/env python
"""S4's evidence: the four cell-cycle models before and after the cycle became a rate.

WHAT S4 CHANGED. `cell_cycle` used to hold four integers and a per-phase counter, and advance them
with a predicate: `if the cell is big enough, phase <- phase + 1`. That is not a rate law, has no
delta form, and is why the operator was `kind: structural` and had to declare that it writes
integrated state. It now integrates ONE continuous coordinate -- `cycle_progress`, the cell's
position through its cycle, 0 at birth and 1 at the end of M -- whose RATE each model sets, and
reads `phase` off it by the phase boundaries.

WHY BYTE-IDENTITY IS NOT THE TEST HERE, unlike every other rung of this campaign. This is a model
change and the plan says so: a predicate that fires on a frame boundary and a rate that arrives at
the same boundary continuously do not produce the same trajectory, and a tolerance that accepted
one would accept anything. What has to hold instead is that each model still MEANS what it meant --
its cells still leave G1 for its own reason, at its own spread -- and that is a population
statement, so this reports populations.

THE THREE COLUMNS, and why each is there rather than a summary number:

  phase fractions   the % of live cells in G1/S/G2/M against time, BEFORE dashed and AFTER solid.
                    This is the panel the plan named. Two runs of the same model should sit on top
                    of each other in the mean and differ in the wobble; a model whose G1 fraction
                    moves by tens of percent has had its rule changed, not its discretisation.
  cell count        the population against time. It is the integral of everything above and the
                    thing a reader of the movie actually sees, so a divergence that the fractions
                    hide -- a cycle that runs uniformly faster, say -- shows up here.
  progress density  where the population sits in `cycle_progress`, against time. AFTER ONLY, and
                    that is the point: the quantity did not exist before. Under the predicate a
                    cell in G1 was just a cell in G1; there was no reading of how far through,
                    because "far" meant elapsed frames and three of the four models do not consult
                    elapsed frames at all. Here each model's own currency defines it -- volume
                    accumulated toward the checkpoint (sizer), inhibitor diluted away (dilution),
                    elapsed time (timer), fraction of the cell's own drawn waiting time (hazard) --
                    so the four are on one axis and comparable for the first time.

    PYTHONPATH=src python tools/cycle_report.py \
        --before /groups/.../GraphData/s4_before --after /groups/.../GraphData/s4_after
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four arms, in the order the family is argued: the size checkpoint, the clock that ignores
# size, the memoryless hazard, and the checkpoint with a molecule under it.
ARMS = [
    ("cyc4_sizer",    "sizer -- absolute size checkpoint"),
    ("cyc4_timer",    "timer -- G1 on the clock"),
    ("cyc4_hazard",   "transition probability -- constant hazard"),
    ("cyc4_dilution", "inhibitor dilution -- Whi5 / Rb"),
]
PHASE_NAMES = ["G1", "S", "G2", "M"]
PHASE_COLORS = ["#3b57c0", "#2e9e4f", "#e8b024", "#e03b2f"]


def series(root, name):
    """Per-frame phase fractions, live cell count, and the progress values of the live cells."""
    p = os.path.join(root, "graphs_data", "tissue", name, "trajectory.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    nF = np.asarray(z["vertex__mesh_nF"])
    T = len(nF)
    ph = np.asarray(z["cell__phase"])[:, :, 0]
    pr = (np.asarray(z["cell__cycle_progress"])[:, :, 0]
          if "cell__cycle_progress" in z.files else None)
    frac = np.zeros((T, 4))
    prog = []
    for t in range(T):
        n = int(nF[t])
        if n <= 0:
            prog.append(np.zeros(0)); continue
        a = np.rint(ph[t, :n]).astype(int).clip(0, 3)
        frac[t] = np.bincount(a, minlength=4) / n
        prog.append(pr[t, :n] if pr is not None else np.zeros(0))
    return dict(T=T, nF=nF, frac=frac, prog=prog)


def density(prog, bins=48):
    """A [bins, T] map of where the population sits in `cycle_progress`, column-normalised.

    NORMALISED PER FRAME, not globally, because the population grows by a third over the run and a
    globally-normalised map would read as "more cells later" -- which the cell-count panel already
    says -- instead of "the population is distributed like this".
    """
    T = len(prog)
    d = np.zeros((bins, T))
    for t, v in enumerate(prog):
        if v.size == 0:
            continue
        h, _ = np.histogram(np.clip(v, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
        s = h.sum()
        if s > 0:
            d[:, t] = h / s
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="output root of the pre-S4 runs")
    ap.add_argument("--after", default=ROOT,
                    help="output root of the post-S4 runs; default this repo, whose `graphs_data` "
                         "is where the four runs live once the rung has landed")
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "s4_cell_cycle.png"))
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for name, label in ARMS:
        b, n = series(a.before, name), series(a.after, name)
        if b is None or n is None:
            print(f"  {name:<16} missing: before={b is not None} after={n is not None}")
            continue
        rows.append((name, label, b, n))
    if not rows:
        print("  nothing to plot"); return 1

    plt.rcParams.update({"font.size": 11, "text.color": "white",
                         "axes.labelcolor": "white", "xtick.color": "white",
                         "ytick.color": "white"})
    fig, ax = plt.subplots(len(rows), 3, figsize=(15.0, 3.1 * len(rows)),
                           facecolor="black", squeeze=False)
    letters = "abcdefghijkl"
    k = 0
    for i, (name, label, b, n) in enumerate(rows):
        for j in range(3):
            ax[i][j].set_facecolor("black")
            for sp in ax[i][j].spines.values():
                sp.set_color("#777777")

        # --- phase fractions: before dashed, after solid, one colour per phase
        t_b, t_n = np.arange(b["T"]), np.arange(n["T"])
        for q in range(4):
            ax[i][0].plot(t_b, 100.0 * b["frac"][:, q], color=PHASE_COLORS[q],
                          lw=1.0, ls="--", alpha=0.55)
            ax[i][0].plot(t_n, 100.0 * n["frac"][:, q], color=PHASE_COLORS[q], lw=1.6,
                          label=PHASE_NAMES[q])
        ax[i][0].set_ylabel("% of live cells")
        ax[i][0].set_ylim(0, 100)
        if i == 0:
            lg = ax[i][0].legend(loc="lower right", frameon=False, ncol=4,
                                 fontsize=9, labelcolor="white")
            for h in lg.legend_handles:
                h.set_linewidth(2.0)

        # --- cell count. RED AND BLUE, not green/black: these are two sources of the same
        # quantity, not a prediction against a ground truth.
        ax[i][1].plot(t_b, b["nF"], color="#e03b2f", lw=1.6, ls="--", label="before S4")
        ax[i][1].plot(t_n, n["nF"], color="#4a8fe0", lw=1.6, label="after S4")
        ax[i][1].set_ylabel("live cells")
        if i == 0:
            ax[i][1].legend(loc="lower right", frameon=False, fontsize=9, labelcolor="white")

        # --- the progress density, after only
        d = density(n["prog"])
        ax[i][2].imshow(d, origin="lower", aspect="auto", cmap="magma",
                        extent=[0, n["T"], 0.0, 1.0],
                        vmax=max(float(d.max()), 1e-9))
        ax[i][2].set_ylabel("cycle progress")
        # the phase boundaries, so the density is readable against the phases it replaces
        for c in np.cumsum([110.0, 80.0, 40.0]) / 240.0:
            ax[i][2].axhline(c, color="white", lw=0.6, alpha=0.4)

        # THE ROW'S MODEL NAME RIDES WITH THE FIRST PANEL'S LETTER rather than sitting over the
        # axes as a title: a title on a panel is the thing this project's figures do not do, and
        # the name still has to be next to the curves it describes.
        for j in range(3):
            ax[i][j].set_xlabel("frame")
            ax[i][j].text(0.015, 1.035, f"{letters[k]}   {label}" if j == 0 else letters[k],
                          transform=ax[i][j].transAxes, color="white", fontsize=12,
                          fontweight="bold", ha="left", va="bottom")
            k += 1

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, facecolor="black")
    print(f"  wrote {a.out}")

    # THE NUMBERS THE FIGURE IS MAKING A CLAIM ABOUT, printed so the claim can be checked without
    # reading pixels. The mean phase fractions over the last quarter of the run -- the part that is
    # steady state rather than the seed relaxing -- and the final cell count.
    print(f"\n{'arm':<16} {'':<7} {'G1':>6} {'S':>6} {'G2':>6} {'M':>6}   {'cells':>6}")
    for name, label, b, n in rows:
        for tag, r in (("before", b), ("after", n)):
            q = r["frac"][3 * r["T"] // 4:].mean(axis=0) * 100.0
            print(f"{name if tag=='before' else '':<16} {tag:<7} {q[0]:6.1f} {q[1]:6.1f} "
                  f"{q[2]:6.1f} {q[3]:6.1f}   {int(r['nF'][-1]):6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

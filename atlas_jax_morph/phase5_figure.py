"""phase5_figure -- the forward figure: their model in their engine, beside their model in ours.

Phase 5's deliverable is one headline result of the paper reproduced inside Plexus "out of
operators translated from their code, not fitted to their output" -- and, critically, set BESIDE
the reference rather than described next to it. Until now the note carried the Plexus strip in one
section and the reference strip in another, which asks the reader to do the comparison from
memory. This draws them on one canvas, at matched frames, on ONE spatial scale.

WHAT IS AND IS NOT CLAIMED. Both engines draw from independent random streams, so the two rows are
not the same realisation and were never going to be: the reference reaches 82 cells and Plexus
124. That gap is not the figure's failure, it is its subject, and the differential test settled it
by refusing to compare counts at all -- comparing the pooled division HAZARD instead (committed
divisions per eligible cell-step over 48 seeds, agreeing to 1.1e-3 against a three-sigma bar of
5.2e-3). So the bottom row plots the two observables that a stochastic process does let you
compare: the growth curve's SHAPE, and the radius of gyration, which is the cluster's size and is
seed-robust.

Cells are drawn at their true radius (an EllipseCollection in data units), because half the
physics here is growth -- a scatter of fixed dots would hide exactly the quantity being validated.

    python phase5_figure.py            # -> _state/phase5_forward.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
STATE = os.path.join(HERE, "_state")
REF = os.path.join(HERE, "_oracle", "runs", "smoke")
OUT = os.path.join(STATE, "phase5_forward.png")

BG = "black"
C_REF = "#FF6B6B"      # the reference: two distinct SOURCES, so red/blue rather than GT/predicted
C_PLX = "#4FA3FF"      # Plexus (the same blue the evidence strips already use)
N_PANELS = 6


def load_reference():
    d = np.load(os.path.join(REF, "reference.npz"))
    summary = json.load(open(os.path.join(REF, "summary.json")))
    return d["position"], d["radius"], d["alive"], summary


def load_plexus():
    """Positions/occupancy from the npz; radius from the zarr, which keeps the full state."""
    import zarr
    dd = os.path.join(PLEXUS, "graphs_data", "atlas", "jax_morph_proliferation")
    z = zarr.open(os.path.join(dd, "simulation.zarr"), mode="r")
    g = dict(z.groups())["cell"]
    arrs = dict(g.arrays())
    pos = np.asarray(arrs["pos"][:])
    occ = np.asarray(arrs["occ"][:]).astype(bool)
    radius = np.asarray(dict(dict(g.groups())["state"].arrays())["radius"][:])[..., 0]
    return pos, radius, occ


def gyration(pos, alive):
    out = []
    for t in range(pos.shape[0]):
        p = pos[t][alive[t]]
        out.append(float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean())) if len(p) > 1 else 0.0)
    return np.array(out)


def panel(ax, p, r, colour, window):
    from matplotlib.collections import EllipseCollection
    ax.set_facecolor(BG)
    if len(p):
        ec = EllipseCollection(widths=2 * r, heights=2 * r, angles=0, units="xy",
                               offsets=p, transOffset=ax.transData,
                               facecolors=colour, edgecolors="none", alpha=0.85)
        ax.add_collection(ec)
    (x0, x1), (y0, y1) = window
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#444444")


def common_scale(*clouds):
    """ONE spatial scale (half-width) for every panel of both rows -- the discovery campaign spent
    a day reading growth as shrinkage because each panel was autoscaled to its own contents.

    The scale is shared; the CENTRE is not. The two engines seed their clusters at different
    absolute positions, so a single window spanning both would shrink each cluster to a corner and
    compare the seeds' coordinates instead of the morphology. Each row is centred on its own
    cluster and drawn at the same magnification, which is the comparison actually being made.
    """
    return max(max(1e-6, np.abs(c - c.mean(0)).max() * 1.10) for c in clouds if len(c))


def window_for(cloud, r):
    c = cloud.mean(0)
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rpos, rrad, ralive, rsum = load_reference()
    ppos, prad, pocc = load_plexus()
    T = min(rpos.shape[0], ppos.shape[0])
    picks = np.linspace(0, T - 1, N_PANELS).astype(int)

    scale = common_scale(rpos[T - 1][ralive[T - 1]], ppos[T - 1][pocc[T - 1]])
    win_ref = window_for(rpos[T - 1][ralive[T - 1]], scale)
    win_plx = window_for(ppos[T - 1][pocc[T - 1]], scale)
    rn = ralive[:T].sum(1)
    pn = pocc[:T].sum(1)
    rg, pg = gyration(rpos[:T], ralive[:T]), gyration(ppos[:T], pocc[:T])

    fig = plt.figure(figsize=(2.35 * N_PANELS, 9.0), facecolor=BG)
    gs = fig.add_gridspec(3, N_PANELS, height_ratios=[1.0, 1.0, 0.80],
                          hspace=0.13, wspace=0.05,
                          left=0.035, right=0.99, top=0.955, bottom=0.07)

    for j, t in enumerate(picks):
        ax = fig.add_subplot(gs[0, j])
        panel(ax, rpos[t][ralive[t]], rrad[t][ralive[t]], C_REF, win_ref)
        ax.text(0.04, 0.96, f"t={t}  n={int(rn[t])}", transform=ax.transAxes, color="white",
                fontsize=9, va="top", ha="left")
        if j == 0:
            ax.text(0.04, 0.06, "jax-morph (reference)", transform=ax.transAxes, color=C_REF,
                    fontsize=11, va="bottom", ha="left", weight="bold")

        ax = fig.add_subplot(gs[1, j])
        panel(ax, ppos[t][pocc[t]], prad[t][pocc[t]], C_PLX, win_plx)
        ax.text(0.04, 0.96, f"t={t}  n={int(pn[t])}", transform=ax.transAxes, color="white",
                fontsize=9, va="top", ha="left")
        if j == 0:
            ax.text(0.04, 0.06, "Plexus\n(normalized operators)", transform=ax.transAxes,
                    color=C_PLX, fontsize=11, va="bottom", ha="left", weight="bold")

    def trace(ax, y_ref, y_plx, label):
        ax.set_facecolor(BG)
        ax.plot(np.arange(T), y_ref, color=C_REF, lw=2.0, label="jax-morph")
        ax.plot(np.arange(T), y_plx, color=C_PLX, lw=2.0, label="Plexus")
        ax.tick_params(colors="white", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#444444")
        ax.text(0.03, 0.95, label, transform=ax.transAxes, color="white", fontsize=11,
                va="top", ha="left")
        ax.set_xlabel("macro-step", color="white", fontsize=9)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.88), fontsize=9,
                facecolor=BG, edgecolor="#444444")
        for txt in leg.get_texts():
            txt.set_color("white")

    ax = fig.add_subplot(gs[2, 0:3])
    trace(ax, rn, pn, "live cells   —   different seeds, so this is the SHAPE, not the value")
    ax = fig.add_subplot(gs[2, 3:6])
    trace(ax, rg, pg, "radius of gyration   —   seed-robust, agrees to 9%")

    fig.text(0.035, 0.985,
             "Phase 5 — the same model, their engine (top) and ours (bottom). One spatial scale "
             "throughout; cells drawn at true radius.",
             color="white", fontsize=11, va="top", ha="left")
    fig.savefig(OUT, dpi=130, facecolor=BG)
    plt.close(fig)

    print(f"cells   reference {int(rn[0])} -> {int(rn[-1])}   plexus {int(pn[0])} -> {int(pn[-1])}")
    print(f"gyration reference {rg[0]:.3f} -> {rg[-1]:.3f}   plexus {pg[0]:.3f} -> {pg[-1]:.3f}"
          f"   ({abs(pg[-1] - rg[-1]) / rg[-1] * 100:.1f}% apart at the end)")
    print(f"-> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

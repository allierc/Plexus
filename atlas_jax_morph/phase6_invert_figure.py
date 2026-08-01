"""phase6_invert_figure -- a target morphology in, a gene circuit out.

Three runs of one composition, differing only in two numbers of the intracellular circuit:

    hand-written activator  W_in = +0.35   the interior grows fastest
    FITTED inhibitor        W_in = -0.2526 the rim grows fastest      <- asked for, not written
    ablated control         W_in =  0      nothing distinguishes any cell

The middle one is the result. Nobody told the optimiser the sign had to change; it was given a
target response and 24 frames of real physics -- growth, adhesion, the morphogen solve, the
intracellular ODE, the discrete division draw -- and gradient descent crossed zero on its own.

    python phase6_invert_figure.py     # -> _state/phase6_invert.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "_state", "phase6_invert.png")
BG = "black"

RUNS = [("regulated_growth", "hand-written activator", "$W_{in}=+0.35$", "#4FA3FF"),
        ("regulated_growth_fitted", "FITTED inhibitor", "$W_{in}=-0.253$", "#FFD166"),
        ("regulated_growth_ablate", "ablated control", "$W_{in}=0$", "#FF6B6B")]


def load(name):
    import zarr
    z = zarr.open(os.path.join(PLEXUS, "graphs_data", "atlas", name, "simulation.zarr"), mode="r")
    g = dict(z.groups())["cell"]
    a = dict(g.arrays())
    st = dict(dict(g.groups())["state"].arrays())
    return {"occ": np.asarray(a["occ"][:]).astype(bool), "pos": np.asarray(a["pos"][:]),
            "chem": np.asarray(st["chemical"][:])[..., 0],
            "k": np.asarray(st["growth_rate"][:])[..., 0],
            "r": np.asarray(st["radius"][:])[..., 0]}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import EllipseCollection

    D = {n: load(n) for n, *_ in RUNS}
    hist = json.load(open(os.path.join(HERE, "_state", "phase6_invert.json")))["history"]
    T = D[RUNS[0][0]]["occ"].shape[0] - 1

    fig = plt.figure(figsize=(14.2, 8.4), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.24, wspace=0.20,
                          left=0.045, right=0.985, top=0.90, bottom=0.08)

    # --- top: the three morphologies, each on its own colour scale --------------------------- #
    for col, (name, label, wtxt, colour) in enumerate(RUNS):
        ax = fig.add_subplot(gs[0, col])
        d = D[name]
        m = d["occ"][T]
        p, rr, kk = d["pos"][T][m], d["r"][T][m], d["k"][T][m]
        dist = np.linalg.norm(p - p.mean(0), axis=1)
        cc = np.corrcoef(dist, kk)[0, 1] if kk.std() > 1e-12 else float("nan")
        lo, hi = float(kk.min()), float(kk.max())
        if hi - lo < 1e-9:
            lo, hi = lo - 5e-4, hi + 5e-4
        ax.set_facecolor(BG)
        ec = EllipseCollection(widths=2 * rr, heights=2 * rr, angles=0, units="xy", offsets=p,
                               transOffset=ax.transData, array=kk, cmap="viridis",
                               norm=plt.Normalize(lo, hi), linewidths=0)
        ax.add_collection(ec)
        c = p.mean(0)
        s = np.abs(p - c).max() * 1.12
        ax.set_xlim(c[0] - s, c[0] + s), ax.set_ylim(c[1] - s, c[1] + s)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#444444")
        ax.text(0.03, 0.975, label, transform=ax.transAxes, color=colour, fontsize=11,
                va="top", ha="left", weight="bold")
        ax.text(0.03, 0.915, wtxt, transform=ax.transAxes, color="white", fontsize=10,
                va="top", ha="left")
        tag = ("rim fastest" if cc > 0.3 else "centre fastest" if cc < -0.3 else "uniform")
        ax.text(0.03, 0.04, f"growth {lo:.3f}–{hi:.3f}\ncorr(dist, growth) = {cc:+.3f}  ({tag})"
                if kk.std() > 1e-12 else f"growth {kk[0]:.4f} everywhere\n(no variance)",
                transform=ax.transAxes, color="white", fontsize=9, va="bottom", ha="left")
        cb = fig.colorbar(ec, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=8)
        cb.outline.set_edgecolor("#444444")

    # --- bottom left: the optimisation crossing zero ----------------------------------------- #
    steps = [h["step"] for h in hist]
    ax = fig.add_subplot(gs[1, 0:2])
    ax.set_facecolor(BG)
    ax.plot(steps, [h["W_in"] for h in hist], color="#FFD166", lw=2.4, label="$W_{in}$")
    ax.plot(steps, [h["b"] for h in hist], color="#9C6ADE", lw=1.8, label="$b$")
    ax.axhline(0, color="white", ls=":", lw=1.3)
    ax.text(steps[-1] * 0.02, 0.02, "  zero: activator $\\rightarrow$ inhibitor", color="white",
            fontsize=9, va="bottom")
    ax.set_xlabel("Adam step", color="white", fontsize=10)
    ax.text(0.02, 0.96, "the circuit, fit through 24 frames of real physics",
            transform=ax.transAxes, color="white", fontsize=11, va="top", ha="left")

    # --- bottom right: the response it was actually asked for -------------------------------- #
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.set_facecolor(BG)
    ax2.plot(steps, [h["slope"] for h in hist], color="#4FA3FF", lw=2.2)
    ax2.axhline(-0.15, color="#FF6B6B", ls="--", lw=1.6)
    ax2.text(steps[-1] * 0.03, -0.15, "  target slope", color="#FF6B6B", fontsize=9, va="bottom")
    ax2.axhline(0, color="white", ls=":", lw=1.0)
    ax2.set_xlabel("Adam step", color="white", fontsize=10)
    ax2.text(0.04, 0.96, "d(growth) / d(sensed signal)", transform=ax2.transAxes,
             color="white", fontsize=11, va="top", ha="left")

    for a in (ax, ax2):
        a.tick_params(colors="white", labelsize=9)
        for sp in a.spines.values():
            sp.set_color("#444444")
    leg = ax.legend(loc="center right", fontsize=9, facecolor=BG, edgecolor="#444444")
    for t in leg.get_texts():
        t.set_color("white")

    fig.text(0.045, 0.965,
             "Phase 6 — a target morphology in, a gene circuit out.  "
             "One composition, three circuits; only the middle one was fitted.",
             color="white", fontsize=11.5, va="top", ha="left")
    fig.savefig(OUT, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"-> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

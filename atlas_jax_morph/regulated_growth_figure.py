"""regulated_growth_figure -- the sense -> regulate -> grow loop, and the control that proves it.

Six operators, all previously validated against the authors' code, none new: cells secrete, each
reads its own local concentration, an intracellular ODE turns that reading into a growth rate,
growth and division move the geometry, and the geometry changes the field.

THE CONTROL IS THE POINT. Every operator "acting" is not evidence of a loop -- in the ablated run
(W_in = 0, the single weight that lets a cell read the field) morphogen still computes a field,
regulate still integrates, cells still grow and divide, and the acted ledger looks the same. What
changes is that `growth_rate` collapses to a spatially UNIFORM value, because nothing distinguishes
one cell from another any more. That is the difference between running and coupling, and only the
control can tell them apart.

    python regulated_growth_figure.py     # -> _state/regulated_growth.png
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLEXUS = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "_state", "regulated_growth.png")

BG = "black"
C_LOOP = "#4FA3FF"     # loop closed
C_ABL = "#FF6B6B"      # ablated control


def load(name):
    import zarr
    z = zarr.open(os.path.join(PLEXUS, "graphs_data", "atlas_jax", name, "simulation.zarr"), mode="r")
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

    L, A = load("regulated_growth"), load("regulated_growth_ablate")
    T = L["occ"].shape[0] - 1

    fig = plt.figure(figsize=(13.6, 8.2), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.22, wspace=0.24,
                          left=0.05, right=0.98, top=0.92, bottom=0.08)

    # --- top row: the cluster at t=T, coloured by growth rate ------------------------------- #
    # PER-PANEL colour scale. A shared one is the honest default for two comparable fields, but
    # here the ablated panel has literally zero range, so a shared scale paints the loop panel a
    # single flat colour and hides the spatial structure that is the entire result. Each panel is
    # normalised to its own range and the range is printed on it, so nothing is implied by hue.
    for col, (D, title, colour) in enumerate((
            (L, "loop closed  ($W_{in}=0.35$)", C_LOOP),
            (A, "ablated control  ($W_{in}=0$)", C_ABL))):
        ax = fig.add_subplot(gs[0, col])
        m = D["occ"][T]
        p, rr, kk = D["pos"][T][m], D["r"][T][m], D["k"][T][m]
        ax.set_facecolor(BG)
        lo, hi = float(kk.min()), float(kk.max())
        if hi - lo < 1e-9:                       # the ablated panel: uniform by construction
            lo, hi = lo - 5e-4, hi + 5e-4
        ec = EllipseCollection(widths=2 * rr, heights=2 * rr, angles=0, units="xy",
                               offsets=p, transOffset=ax.transData, array=kk,
                               cmap="viridis", norm=plt.Normalize(lo, hi), linewidths=0)
        ax.add_collection(ec)
        c = p.mean(0)
        s = np.abs(p - c).max() * 1.12
        ax.set_xlim(c[0] - s, c[0] + s), ax.set_ylim(c[1] - s, c[1] + s)
        ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#444444")
        ax.text(0.03, 0.97, title, transform=ax.transAxes, color=colour, fontsize=11,
                va="top", ha="left", weight="bold")
        ax.text(0.03, 0.04, f"growth rate  {kk.min():.4f} … {kk.max():.4f}",
                transform=ax.transAxes, color="white", fontsize=9, va="bottom", ha="left")
        cb = fig.colorbar(ec, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(colors="white", labelsize=8)
        cb.outline.set_edgecolor("#444444")

    # --- top-right: what the cell sensed vs what it decided --------------------------------- #
    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(BG)
    for D, lab, colour in ((L, "loop closed", C_LOOP), (A, "ablated", C_ABL)):
        m = D["occ"][T]
        kk = D["k"][T][m]
        r = np.corrcoef(D["chem"][T][m], kk)[0, 1] if kk.std() > 1e-12 else float("nan")
        ax.scatter(D["chem"][T][m], kk - kk.mean(), s=22, color=colour, alpha=0.7,
                   linewidths=0, label=f"{lab}  (r = {r:.3f})" if kk.std() > 1e-12
                   else f"{lab}  (no variance)")
    ax.set_xlabel("chemical sensed", color="white", fontsize=10)
    ax.set_ylabel("growth rate, deviation from mean", color="white", fontsize=10)
    ax.text(0.03, 0.97, "sense $\\rightarrow$ decide, per cell", transform=ax.transAxes,
            color="white", fontsize=11, va="top", ha="left")

    # --- bottom: the two time series -------------------------------------------------------- #
    ts = np.arange(T + 1)
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax2.set_facecolor(BG)
    for D, lab, colour in ((L, "loop closed", C_LOOP), (A, "ablated", C_ABL)):
        mean = [D["k"][t][D["occ"][t]].mean() for t in ts]
        sd = [D["k"][t][D["occ"][t]].std() for t in ts]
        ax2.plot(ts, mean, color=colour, lw=2.2, label=f"{lab}: mean")
        ax2.fill_between(ts, np.array(mean) - np.array(sd), np.array(mean) + np.array(sd),
                         color=colour, alpha=0.25, lw=0)
    B, GAMMA = -0.6, 0.8            # the spec's bias and decay; keep in step with the yaml
    fp = (0.5 + 0.5 * B / math.sqrt(1 + B * B)) / GAMMA
    ax2.axhline(fp, color="white", ls=":", lw=1.4)
    ax2.text(T * 0.52, fp, f"  autonomous fixed point $\\sigma(b)/\\gamma$ = {fp:.4f}",
             color="white", fontsize=9, va="bottom")
    ax2.set_xlabel("macro-step", color="white", fontsize=10)
    ax2.text(0.02, 0.96, "growth rate across the population (band = $\\pm$1 s.d.)",
             transform=ax2.transAxes, color="white", fontsize=11, va="top", ha="left")

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(BG)
    for D, lab, colour in ((L, "loop closed", C_LOOP), (A, "ablated", C_ABL)):
        ax3.plot(ts, [D["k"][t][D["occ"][t]].std() for t in ts], color=colour, lw=2.2, label=lab)
    ax3.set_yscale("symlog", linthresh=1e-6)
    ax3.set_xlabel("macro-step", color="white", fontsize=10)
    ax3.text(0.04, 0.96, "spread of growth rate\nacross cells", transform=ax3.transAxes,
             color="white", fontsize=11, va="top", ha="left")

    for ax in (fig.axes[2], ax2, ax3):
        ax.tick_params(colors="white", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#444444")
        leg = ax.legend(loc="center right", fontsize=9, facecolor=BG, edgecolor="#444444")
        for t in leg.get_texts():
            t.set_color("white")

    fig.text(0.05, 0.975,
             "morphogen → chemical → regulate → growth_rate → grow_radius → geometry → morphogen"
             "     ·     six validated operators, none new",
             color="white", fontsize=11, va="top", ha="left")
    fig.savefig(OUT, dpi=130, facecolor=BG)
    plt.close(fig)
    print(f"-> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""10 visualizations of the DEPOSITED CHEMOATTRACTANT (scent field) from one
winner run -- each with a different colormap (LUT) and contrast, on black.

    python viz_scent.py
"""

from __future__ import annotations

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from viz_components import run_winner2, FOOD, HOME

FPS = 20
# (name, cmap, vmax_fraction, gamma)  -- gamma<1 boosts faint trails (more glow)
VARIANTS = [
    ("inferno",  "inferno",       0.50, 1.0),
    ("magma_glow", "magma",       0.30, 0.6),
    ("plasma",   "plasma",        0.50, 1.0),
    ("viridis",  "viridis",       0.60, 1.0),
    ("hot",      "hot",           0.40, 0.8),
    ("turbo",    "turbo",         0.50, 1.0),
    ("gist_heat","gist_heat",     0.45, 0.7),
    ("cool",     "cool",          0.55, 1.0),
    ("cubehelix","cubehelix",     0.55, 1.0),
    ("spectral", "nipy_spectral", 0.50, 0.9),
]


def main():
    a = run_winner2()
    scent, cp, obs = a["field"], a["cell_pos"], a["obstacles"]
    T = scent.shape[0]
    smax = max(scent.max(), 1e-6)

    for i, (name, cmap, vfrac, gamma) in enumerate(VARIANTS, 1):
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        fig.patch.set_facecolor("black"); ax.set_facecolor("black")
        norm = PowerNorm(gamma, vmin=0, vmax=smax * vfrac)
        im = ax.imshow(scent[0].T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, norm=norm, zorder=0)
        for r in obs:                                   # dark grey walls on black
            ax.add_patch(Rectangle((r[0], r[1]), r[2]-r[0], r[3]-r[1], facecolor="#555555", edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((HOME[0], HOME[1]), HOME[2]-HOME[0], HOME[3]-HOME[1], fill=False, edgecolor="#4f4", lw=1.0, zorder=2))
        ax.add_patch(Rectangle((FOOD[0], FOOD[1]), FOOD[2]-FOOD[0], FOOD[3]-FOOD[1], fill=False, edgecolor="#fa4", lw=1.0, zorder=2))
        s = ax.scatter(cp[0][:, 0], cp[0][:, 1], s=2.5, c="white", lw=0, alpha=0.8, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
        tt = ax.set_title(f"scent_{i} · {name}  (vmax {vfrac}, gamma {gamma})", color="#ccc", fontsize=7, loc="left")

        def upd(f, im=im, s=s):
            im.set_data(scent[f].T); s.set_offsets(cp[f]); return im, s

        FuncAnimation(fig, upd, frames=T, blit=False).save(f"scent_{i}.gif", writer=PillowWriter(fps=FPS))
        plt.close(fig); print(f"scent_{i}.gif = {name}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

"""Render a forage-style scenario to a gif (loaded cells green over the scent field).

Generic over any forage_*-style spec (scent field first, home/food boxes, walls).
Reuses the graphic design of forage_loop.render_winner.

    python render_forage.py forage_wander            # -> forage_wander.gif
    python render_forage.py forage_maze fps=30        # optional fps override
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
os.chdir(_HERE)

from scenario_schema import load
import engine2

DEVICE = os.environ.get("DEVICE", "cuda")
PAL = np.array([[1.0, 0.30, 0.30, 1.0], [0.30, 0.65, 1.0, 1.0]])   # type A red, B blue
GREEN = np.array([0.15, 0.95, 0.25, 1.0])


def render(name, fps=30):
    sc = load(f"scenarios/{name}.yaml")
    t0 = time.time()
    _, a = engine2.run(sc, None, device=DEVICE)
    dt = time.time() - t0
    pp, par = a["particle_pos"], a["parent"]
    pt = a["cell_type"][par]; ld = a["loaded"]
    walls, scent, T = a["walls"], a["field"], a["particle_pos"].shape[0]
    base = PAL[pt]
    smax = max(scent.max() * 0.5, 1e-6)
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    imS = ax.imshow(scent[0].T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=0, vmax=smax)
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, 1, 0, 1], cmap="gray", vmin=0, vmax=2.2)
    ax.add_patch(plt.Rectangle((0, 0), 0.18, 0.18, fill=False, edgecolor="#5599ff", lw=1.5))
    ax.add_patch(plt.Rectangle((0.82, 0.82), 0.18, 0.18, fill=False, edgecolor="#ffaa33", lw=1.5))
    sc_pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.3, c=base)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", fontsize=8, color="white")

    def upd(fr):
        sc_pts.set_offsets(pp[fr])
        col = base.copy(); col[ld[fr][par]] = GREEN
        sc_pts.set_color(col)
        imS.set_data(scent[fr].T)
        tt.set_text(f"{sc.name}  frame {fr}/{T-1} | food delivered {a['delivered_t'][fr]}")
        return [sc_pts, imS, tt]

    out = f"{name}.gif"
    FuncAnimation(fig, upd, frames=T, blit=False).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"[done] {name}: food_delivered={a['food_delivered']} over {T} rec-frames, "
          f"{dt:.0f}s -> {out}", flush=True)
    return a


if __name__ == "__main__":
    name = next((x for x in sys.argv[1:] if "=" not in x), "forage_wander")
    fps = next((int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("fps=")), 30)
    render(name, fps=fps)

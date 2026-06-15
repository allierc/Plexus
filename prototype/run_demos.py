"""run_demos.py -- generate a gif + intent check for each new demo/scenario.

This is the "does the simulator actually work" driver: each entry loads a spec,
runs the generic engine, renders a gif in the SAME graphic style as the previous
gallery (black bg, inferno field, red/blue particles per type, gray walls, region
boxes, white title), and prints an INTENT metric (per plexus.tex: "it ran" is not
"it worked"). Intent metrics are computed here from the output arrays -- the
engine stays generic and never branches on scenario name.

    PYTHONPATH=../src python run_demos.py                 # all demos
    PYTHONPATH=../src python run_demos.py demo_sort race_pillars   # a subset
    DEVICE=cpu python run_demos.py demo_sort              # cpu fallback
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scenario_schema import load
import engine2

DEVICE = os.environ.get("DEVICE", "cuda")
PAL = np.array([[1.0, 0.30, 0.30, 1.0], [0.30, 0.65, 1.0, 1.0]])   # type A red, B blue (RGBA)
GREEN = np.array([0.15, 0.95, 0.25, 1.0])


# --------------------------------------------------------------------------- #
#  intent metrics (computed from the run output -- the "it worked" check)
# --------------------------------------------------------------------------- #
def _knn_cross_fraction(pos, typ, r=0.10):
    """mean fraction of within-r neighbours that are a DIFFERENT type (0 = sorted)."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    near = (d < r) & (d > 0)
    cnt = near.sum(1).clip(min=1)
    cross = (near & (typ[:, None] != typ[None, :])).sum(1)
    return float((cross / cnt).mean())


def _mean_nn_dist(pos):
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(d.min(1).mean())


def intent_sort(a):
    typ = a["cell_type"]
    f0, f1 = a["cell_pos"][0], a["cell_pos"][-1]
    c0, c1 = _knn_cross_fraction(f0, typ), _knn_cross_fraction(f1, typ)
    ok = c1 < c0 - 0.02
    return ok, f"cross-type neighbour fraction {c0:.3f} -> {c1:.3f} (lower = sorted)"


def intent_aggregate(a):
    soft = a["cell_type"] == 0
    d0 = _mean_nn_dist(a["cell_pos"][0][soft]); d1 = _mean_nn_dist(a["cell_pos"][-1][soft])
    ok = d1 < d0 - 1e-3
    return ok, f"soft-cell mean NN distance {d0:.4f} -> {d1:.4f} (lower = aggregated)"


def intent_graze(a):
    fld = a["field"]
    dep = 1.0 - fld[-1].mean() / max(fld[0].mean(), 1e-9)
    h = a["harvested"]
    ok = h > 0 and dep > 0.05
    return ok, f"harvested={h:.1f}, field depleted {dep*100:.0f}% (start {fld[0].mean():.2f} -> end {fld[-1].mean():.2f})"


def intent_race(x_finish):
    def f(a):
        x = a["cell_pos"][..., 0]                       # [T, Ncell]
        crossed = x > x_finish                           # [T, Ncell]
        ever = crossed.any(0)
        arrived = int(ever.sum()); n = x.shape[1]
        T = x.shape[0]
        first = np.where(ever, crossed.argmax(0), T)      # first crossing frame (T if never)
        speed_bonus = float(np.clip(1 - first[ever] / T, 0, 1).sum()) if arrived else 0.0
        score = arrived + speed_bonus
        ok = arrived > 0
        return ok, f"{arrived}/{n} crossed finish (x>{x_finish}); race score={score:.1f}"
    return f


# --------------------------------------------------------------------------- #
#  render (shared graphic design, matching the previous gallery gifs)
# --------------------------------------------------------------------------- #
def render(a, sc, path, fps=20, x_finish=None, start_box=None):
    pp, par = a["particle_pos"], a["parent"]
    pt = a["cell_type"][par]; ld = a["loaded"]
    walls, fld, T = a["walls"], a["field"], a["particle_pos"].shape[0]
    base = PAL[pt]                                              # [Np,4]
    vmax = max(float(fld.max()) * 0.5, 1e-6)
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    imF = ax.imshow(fld[0].T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=0, vmax=vmax)
    if walls is not None:
        ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, 1, 0, 1],
                  cmap="gray", vmin=0, vmax=2.2)
    if start_box is not None:
        x0, y0, x1, y1 = start_box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#5599ff", lw=1.2))
    if x_finish is not None:
        ax.axvline(x_finish, color="#33dd55", ls="--", lw=1.5)
    sct = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.3, c=base)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", fontsize=8, color="white")

    def upd(fr):
        sct.set_offsets(pp[fr])
        col = base.copy()
        if ld is not None and ld.shape[0] > fr:
            col[ld[fr][par]] = GREEN
        sct.set_color(col)
        imF.set_data(fld[fr].T)
        tt.set_text(f"{sc.name}  frame {fr}/{T-1}")
        return [sct, imF, tt]

    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  demo table
# --------------------------------------------------------------------------- #
DEMOS = {
    "demo_sort":      dict(gif="demo1_sort.gif",       intent=intent_sort),
    "demo_graze":     dict(gif="demo2_graze.gif",      intent=intent_graze),
    "demo_aggregate": dict(gif="demo3_aggregate.gif",  intent=intent_aggregate),
    "race_pillars":   dict(gif="race1_pillars.gif",    intent=intent_race(0.86),
                           x_finish=0.86, start_box=[0.0, 0.0, 0.12, 1.0]),
    "race_maze_easy": dict(gif="race2_maze_easy.gif",  intent=intent_race(0.88),
                           x_finish=0.88, start_box=[0.0, 0.0, 0.10, 1.0]),
    "race_maze_hard": dict(gif="race3_maze_hard.gif",  intent=intent_race(0.88),
                           x_finish=0.88, start_box=[0.0, 0.0, 0.10, 1.0]),
}


def main():
    names = [a for a in sys.argv[1:] if a in DEMOS] or list(DEMOS)
    print(f"[device] {DEVICE}\n", flush=True)
    results = []
    for nm in names:
        cfg = DEMOS[nm]
        t = time.time()
        sc = load(f"scenarios/{nm}.yaml")
        _, a = engine2.run(sc, None, device=DEVICE)
        render(a, sc, cfg["gif"], x_finish=cfg.get("x_finish"), start_box=cfg.get("start_box"))
        ok, msg = cfg["intent"](a)
        dt = time.time() - t
        flag = "OK " if ok else "!! "
        print(f"{flag}{nm:16s} -> {cfg['gif']:22s} [{dt:5.1f}s]  {msg}", flush=True)
        results.append((nm, ok, msg))
    print("\n[summary]")
    for nm, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {nm:16s} {msg}")


if __name__ == "__main__":
    main()

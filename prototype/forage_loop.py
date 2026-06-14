"""Hour-long UCB optimization of maze foraging; render a winner gif on each
significant improvement.

- Searches the behavioural parameters (motility speed/rot, sense gain, mpm drag)
  with a kNN-surrogate UCB acquisition, maximizing food delivered.
- When a new best beats the previous by >= max(5, 12%), saves winner_<k>.gif at
  DOUBLE fps (40), with LOADED cells drawn green (whole cell, not a ring).
- Periodically rewrites opt_curve.png and prints a flushed log.
- Optimizes 5 levers: motility speed & rotation, sense gain, MPM drag, and the
  MATERIAL stiffness (cell Young's modulus).

    python forage_loop.py            # no time limit -> runs until you kill it
    python forage_loop.py 3600       # optional: stop after N seconds
"""

from __future__ import annotations

import sys
import time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scenario_schema import load
import engine2

BOUNDS = np.array([[20., 120.], [80., 400.], [0.5, 3.0], [0.10, 0.60],
                   [20., 600.], [400., 2000.], [0.008, 0.018], [30., 120.]])
PARAMS = ["motility.speed", "sense.gain", "mpm.drag", "motility.rot",
          "cell.youngs", "mpm.a_max", "particle.radius", "cell.n"]   # 8 levers
SEARCH_FRAMES = 3000
GIF_FRAMES = 3000
DEVICE = "cuda"
PAL = np.array([[1.0, 0.30, 0.30, 1.0], [0.30, 0.65, 1.0, 1.0]])   # type A red, B blue (RGBA)
GREEN = np.array([0.15, 0.95, 0.25, 1.0])


def _apply(sc, p, n_frames):
    sc.n_frames = n_frames
    for o in sc.operators:
        if o.op == "motility":
            o.params["speed"] = float(p[0]); o.params["rot"] = float(p[3])
        elif o.op == "sense":
            o.params["gain"] = float(p[1])
        elif o.op == "mpm":
            o.params["drag"] = float(p[2]); o.params["a_max"] = float(p[5])
    for t in sc.sets["cell"]["types"].values():     # material lever: cell stiffness
        t["youngs"] = float(p[4])
    sc.sets["particle"]["radius"] = float(p[6])      # cell size
    sc.sets["cell"]["n"] = int(round(p[7]))          # fleet size
    return sc


def evaluate(u):
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load("scenarios/forage_maze.yaml"), p, SEARCH_FRAMES)
    _, a = engine2.run(sc, None, device=DEVICE)
    return float(a["food_delivered"])


def render_winner(u, path, fps=40):
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load("scenarios/forage_maze.yaml"), p, GIF_FRAMES); sc.record_every = 1
    _, a = engine2.run(sc, None, device=DEVICE)
    pp, par = a["particle_pos"], a["parent"]
    pt = a["cell_type"][par]; cp = a["cell_pos"]; ld = a["loaded"]
    walls, scent, T = a["walls"], a["field"], a["particle_pos"].shape[0]
    base = PAL[pt]                                       # [Np,4] per-particle base colour
    smax = max(scent.max() * 0.5, 1e-6)
    fig, ax = plt.subplots(figsize=(5.5, 5.5)); fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    imS = ax.imshow(scent[0].T, origin="lower", extent=[0, 1, 0, 1], cmap="inferno", vmin=0, vmax=smax)
    ax.imshow(np.where(walls.T, 1.0, np.nan), origin="lower", extent=[0, 1, 0, 1], cmap="gray", vmin=0, vmax=2.2)
    ax.add_patch(plt.Rectangle((0, 0), 0.18, 0.18, fill=False, edgecolor="#5599ff", lw=1.5))
    ax.add_patch(plt.Rectangle((0.82, 0.82), 0.18, 0.18, fill=False, edgecolor="#ffaa33", lw=1.5))
    sc_pts = ax.scatter(pp[0][:, 0], pp[0][:, 1], s=1.3, c=base)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    tt = ax.set_title("", fontsize=8, color="white")

    def upd(fr):
        sc_pts.set_offsets(pp[fr])
        col = base.copy()
        col[ld[fr][par]] = GREEN                         # loaded cells -> green
        sc_pts.set_color(col)
        imS.set_data(scent[fr].T)
        tt.set_text(f"winner  frame {fr} | food delivered {a['delivered_t'][fr]}")
        return [sc_pts, imS, tt]

    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return int(a["food_delivered"])


def write_index(winners):
    """Index the winner gifs so they are discoverable (food score + params + path)."""
    head = "| # | food delivered | " + " | ".join(PARAMS) + " | gif |"
    sep = "|" + "---|" * (len(PARAMS) + 3)
    lines = ["# Foraging optimization winners (auto-indexed by forage_loop.py)\n",
             "Each row is a significantly-better design found by UCB search; the gif "
             "shows it at 40 fps with loaded cells in green.\n", head, sep]
    for w in winners:
        vals = " | ".join(f"{v:.2f}" for v in w["params"])
        lines.append(f"| {w['k']} | {w['food']} | {vals} | [{w['gif']}]({w['gif']}) |")
    open("WINNERS.md", "w").write("\n".join(lines) + "\n")


def save_curve(Y, best_hist):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(Y) + 1), Y, "o", ms=3, color="#999", label="evaluation")
    plt.plot(range(1, len(Y) + 1), np.maximum.accumulate(Y), "-", color="#d62728", lw=2, label="best so far")
    plt.xlabel("simulation #"); plt.ylabel("food delivered"); plt.legend(); plt.tight_layout()
    plt.savefig("opt_curve.png", dpi=110); plt.close()


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf")   # no limit -> run until killed
    rng = np.random.default_rng(0)
    D = len(PARAMS)
    t0 = time.time()
    print(f"[start] {len(PARAMS)} levers: {PARAMS}", flush=True)
    print("[start] rendering initial reference gif (~90s, no output until done)...", flush=True)
    # initial reference: render the default (starting) configuration before optimizing
    init_u = (np.array([40., 180., 2.0, 0.30, 300., 700., 0.012, 60.])
              - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    init_food = render_winner(init_u, "initial.gif", fps=40)
    winners_list = [{"k": 0, "food": init_food,
                     "params": BOUNDS[:, 0] + init_u * (BOUNDS[:, 1] - BOUNDS[:, 0]),
                     "gif": "initial.gif"}]
    write_index(winners_list)
    print(f"[initial] food={init_food} -> initial.gif", flush=True)

    X, Y = [], []
    for _ in range(5):                                   # random seed
        u = rng.random(D); X.append(u); Y.append(evaluate(u))
    best = max(Y); winners = 0
    print(f"[seed] best={best:.0f}  ({len(Y)} evals, {time.time()-t0:.0f}s)", flush=True)

    while time.time() - t0 < budget:
        Xa, Ya = np.array(X), np.array(Y)
        best_u = Xa[Ya.argmax()]
        # candidate pool = global exploration + LOCAL refinement around the best
        glob = rng.random((2500, D))
        loc = np.clip(best_u + rng.normal(0, 0.07, (2500, D)), 0, 1)
        cand = np.vstack([glob, loc])
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)
        w = np.exp(-(d / 0.22) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)
        unc = d.min(1)
        scale = (Ya.max() - Ya.min()) or 1.0
        u = cand[(mean + 0.6 * scale * unc).argmax()]      # lower exploration -> faster convergence
        f = evaluate(u); X.append(u); Y.append(f)

        if f >= best + max(5.0, 0.12 * best):           # significant new winner
            best = f; winners += 1
            gif = f"winner_{winners}.gif"
            full = render_winner(u, gif, fps=40)
            p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
            winners_list.append({"k": winners, "food": full, "params": p, "gif": gif})
            write_index(winners_list)
            print(f"[WINNER {winners}] search_food={f:.0f} full_food={full} -> {gif}  "
                  + ", ".join(f"{k}={v:.2f}" for k, v in zip(PARAMS, p)), flush=True)
        if len(Y) % 10 == 0:
            save_curve(Y, best)
            print(f"  {len(Y)} evals, best={best:.0f}, {time.time()-t0:.0f}s", flush=True)

    save_curve(Y, best)
    bi = int(np.argmax(Y))
    print(f"\n[DONE] {len(Y)} evals in {time.time()-t0:.0f}s, {winners} winners, best food={Y[bi]:.0f}", flush=True)


if __name__ == "__main__":
    main()

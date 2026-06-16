"""race_opt.py -- overnight UCB optimization of a RACE spec (pillars / maze).

Mirrors forage_loop.py (the food-gathering optimizer) for the three new race specs.
The objective is the number of cells that ESCAPE the exit (H.finished, via the
`death` operator). A kNN-surrogate UCB search maximizes it; whenever the best
escaped count improves by >= 10 it renders a winner gif (same look as run_demos)
and rewrites WINNERS_<scenario>.md. No time limit -> runs until killed.

Levers = the food-gathering set + the boids params:
    motility.speed/rot, sense.gain, mpm.drag, mpm.a_max,
    cell.youngs, particle.radius, cell.n,
    boids.cohesion, boids.align, boids.separate, boids.radius   (12 total)

    # local
    PYTHONPATH=../src python race_opt.py race_maze_hard
    # cluster (relative paths; ~100 evals overnight, no limit)
    bsub ... "python race_opt.py race_pillars"
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

# --- self-bootstrap: run from anywhere (no PYTHONPATH / cwd needed) ---
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../prototype
sys.path.insert(0, _HERE)                                   # prototype modules
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))  # the `plexus` package
os.chdir(_HERE)                                             # so 'scenarios/...' resolves

from scenario_schema import load
import engine2
from run_demos import render          # reuse the exact race render (wide, death, counter)

DEVICE = os.environ.get("DEVICE", "cuda")

#                 speed  rot   gain   drag  a_max  youngs  radius   n     cohes  align  separ  b_rad
PARAMS = ["motility.speed", "motility.rot", "sense.gain", "mpm.drag", "mpm.a_max",
          "cell.youngs", "particle.radius", "cell.n",
          "boids.cohesion", "boids.align", "boids.separate", "boids.radius"]
BOUNDS = np.array([
    [20.,  80.],     # motility.speed
    [0.10, 0.60],    # motility.rot
    [80., 400.],     # sense.gain
    [0.5,  3.0],     # mpm.drag
    [400., 1500.],   # mpm.a_max
    [20.,  600.],    # cell.youngs   (material)
    [0.006, 0.016],  # particle.radius (cell size)
    [40.,  160.],    # cell.n        (fleet size)
    [0.0,  0.20],    # boids.cohesion
    [0.0,  0.30],    # boids.align
    [0.0,  1.00],    # boids.separate
    [0.03, 0.10],    # boids.radius
])
D = len(PARAMS)


def _apply(sc, p):
    for o in sc.operators:
        if o.op == "motility":
            o.params["speed"] = float(p[0]); o.params["rot"] = float(p[1])
        elif o.op == "sense":
            o.params["gain"] = float(p[2])
        elif o.op == "mpm":
            o.params["drag"] = float(p[3]); o.params["a_max"] = float(p[4])
        elif o.op == "boids":
            o.params["cohesion"] = float(p[8]); o.params["align"] = float(p[9])
            o.params["separate"] = float(p[10]); o.params["radius"] = float(p[11])
    for t in sc.sets["cell"]["types"].values():
        t["youngs"] = float(p[5])
    sc.sets["particle"]["radius"] = float(p[6])
    sc.sets["cell"]["n"] = int(round(p[7]))
    return sc


def _finish_x(sc):
    return next((float(o.params.get("x", 3.92)) for o in sc.operators if o.op == "death"), 3.92)


def evaluate(u, scenario):
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load(f"scenarios/{scenario}.yaml"), p)
    _, a = engine2.run(sc, None, device=DEVICE)
    return float(a["finished"])


def render_winner(u, scenario, path):
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load(f"scenarios/{scenario}.yaml"), p)
    _, a = engine2.run(sc, None, device=DEVICE)
    render(a, sc, path, x_finish=_finish_x(sc))
    return int(a["finished"])


def write_index(scenario, winners):
    head = "| # | escaped | " + " | ".join(PARAMS) + " | gif |"
    sep = "|" + "---|" * (len(PARAMS) + 3)
    lines = [f"# {scenario} optimization winners (auto-indexed by race_opt.py)\n",
             "Each row is a design whose escaped count beat the previous best by >= 10.\n", head, sep]
    for w in winners:
        vals = " | ".join(f"{v:.3f}" for v in w["params"])
        lines.append(f"| {w['k']} | {w['escaped']} | {vals} | [{w['gif']}]({w['gif']}) |")
    open(f"WINNERS_{scenario}.md", "w").write("\n".join(lines) + "\n")


def main(scenario=None, budget=None):
    scenario = scenario or (sys.argv[1] if len(sys.argv) > 1 else "race_maze_hard")
    if budget is None:
        budget = float(sys.argv[2]) if len(sys.argv) > 2 else float("inf")   # seconds; default: no limit
    rng = np.random.default_rng(0)
    t0 = time.time()
    print(f"[start] {scenario}: {D} levers {PARAMS}", flush=True)

    X, Y, winners_list, winners = [], [], [], 0
    last_winner = -1e9

    def maybe_winner(u, f):
        nonlocal winners, last_winner
        if f < last_winner + 10:
            return
        winners += 1
        gif = f"{scenario}_winner_{winners}.gif"
        full = render_winner(u, scenario, gif)
        p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        winners_list.append({"k": winners, "escaped": full, "params": p, "gif": gif})
        write_index(scenario, winners_list)
        last_winner = f
        print(f"[WINNER {winners}] escaped={f:.0f} -> {gif}  "
              + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAMS, p)), flush=True)

    for _ in range(5):                                       # random exploration seed
        u = rng.random(D); f = evaluate(u, scenario); X.append(u); Y.append(f)
        print(f"  eval {len(Y):3d} (seed): escaped={f:.0f}  best={max(Y):.0f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, f)
    print(f"[seed] best={max(Y):.0f} ({len(Y)} evals, {time.time()-t0:.0f}s)", flush=True)

    while time.time() - t0 < budget:
        Xa, Ya = np.array(X), np.array(Y)
        best_u = Xa[Ya.argmax()]
        glob = rng.random((2500, D))                          # global exploration
        loc = np.clip(best_u + rng.normal(0, 0.07, (2500, D)), 0, 1)   # local refinement
        cand = np.vstack([glob, loc])
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)
        w = np.exp(-(d / 0.22) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)
        unc = d.min(1)
        scale = (Ya.max() - Ya.min()) or 1.0
        u = cand[(mean + 0.6 * scale * unc).argmax()]         # UCB acquisition
        f = evaluate(u, scenario); X.append(u); Y.append(f)
        print(f"  eval {len(Y):3d}: escaped={f:.0f}  best={max(Y):.0f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, f)

    print(f"[DONE] {len(Y)} evals, {winners} winners, best escaped={max(Y):.0f}", flush=True)


if __name__ == "__main__":
    main()

"""opt_ant.py -- UCB optimization of the ant colony's MATERIAL + DYNAMICS parameters.

Same kNN-surrogate UCB search as the race optimizer (prototype/opt_maze.py /
race_opt.py), retargeted at the ant colony. The objective is the food delivered
(H.food_delivered, from the `colony` operator). Because first-discovery of food is
stochastic (the single-seed counts vary wildly -- see ant/results.md), each
candidate is evaluated as the MEAN delivered over several seeds, so the search
finds parameters that forage well *robustly*, not ones that got one lucky seed.

Levers (13) -- the material + dynamics knobs, exactly the kind the framework says
the LLM/optimizer should search (operator params + per-type material props):

    dynamics : motility.speed, motility.rot, trail.turn, trail.sensor_dist,
               secrete.rate, secrete.runout, field.decay, field.diffusion,
               colony.perception, mpm.drag
    material : cell.youngs (stiffness), particle.radius (body size)
    fleet    : cell.n

Whenever the best mean-delivered improves by >= IMPROVE, it writes a winner spec
(specs/ant_opt_winner_<k>.yaml) and renders its gif+montage (render_ant), and
rewrites WINNERS_ant.md. No time limit -> runs until killed.

    PYTHONPATH=../../src python opt_ant.py                # run until killed
    PYTHONPATH=../../src python opt_ant.py 7200           # stop after N seconds
    PYTHONPATH=../../src python opt_ant.py --test         # 3-eval smoke test, no winners
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

# --- self-bootstrap: run from anywhere (no PYTHONPATH / cwd needed) ---
_HERE = os.path.dirname(os.path.abspath(__file__))               # .../prototype/ant
_PROTO = os.path.dirname(_HERE)                                  # .../prototype
sys.path.insert(0, _HERE)                                        # ant/ modules (render_ant)
sys.path.insert(0, _PROTO)                                       # prototype modules (engine2, schema)
sys.path.insert(0, os.path.join(os.path.dirname(_PROTO), "src")) # the `plexus` package
os.chdir(_HERE)                                                  # so 'specs/...' resolves

from scenario_schema import load
import engine2
import render_ant

DEVICE = os.environ.get("DEVICE", "cuda:1")
BASE = "ant_colony"                 # base spec; only the swept params are overridden
EVAL_FRAMES = 1200                  # short rollout for evaluation (winners render full-length)
EVAL_SEEDS = (0, 1, 2)             # mean over seeds -> robust to discovery luck
IMPROVE = 8.0                       # save a winner gif when mean delivered beats best by this

#         dynamics .......................................................  material .......  fleet
PARAMS = ["motility.speed", "motility.rot", "trail.turn", "trail.sensor_dist",
          "secrete.rate", "secrete.runout", "field.decay", "field.diffusion",
          "colony.perception", "mpm.drag", "cell.youngs", "particle.radius", "cell.n"]
BOUNDS = np.array([
    [120., 360.],    # motility.speed      (dynamics)
    [0.05, 0.30],    # motility.rot
    [0.20, 0.80],    # trail.turn
    [0.03, 0.09],    # trail.sensor_dist
    [2.0,  8.0],     # secrete.rate
    [200., 800.],    # secrete.runout
    [0.008, 0.060],  # field.decay
    [0.005, 0.050],  # field.diffusion
    [0.04, 0.14],    # colony.perception
    [0.30, 2.00],    # mpm.drag
    [40.,  400.],    # cell.youngs         (material: stiffness)
    [0.006, 0.014],  # particle.radius     (material: body size)
    [100., 240.],    # cell.n              (fleet size)
])
D = len(PARAMS)


def _apply(sc, p):
    """Write a parameter vector p (physical units) onto a loaded scenario. The two
    `trail` and two `secrete` ops (loaded=0/1) and both pheromone fields are all set."""
    for o in sc.operators:
        if o.op == "motility":
            o.params["speed"] = float(p[0]); o.params["rot"] = float(p[1])
        elif o.op == "trail":
            o.params["turn"] = float(p[2]); o.params["sensor_dist"] = float(p[3])
        elif o.op == "secrete":
            o.params["rate"] = float(p[4]); o.params["runout"] = float(p[5])
        elif o.op == "colony":
            o.params["perception"] = float(p[8])
        elif o.op == "mpm":
            o.params["drag"] = float(p[9])
    for f in sc.fields.values():
        f["decay"] = float(p[6]); f["diffusion"] = float(p[7])
    for t in sc.sets["cell"]["types"].values():
        t["youngs"] = float(p[10])
    sc.sets["particle"]["radius"] = float(p[11])
    sc.sets["cell"]["n"] = int(round(p[12]))
    return sc


def evaluate(u, frames=EVAL_FRAMES, seeds=EVAL_SEEDS):
    """Mean food delivered over `seeds` for the candidate u in [0,1]^D."""
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    vals = []
    for s in seeds:
        sc = _apply(load(f"specs/{BASE}.yaml"), p)
        sc.seed = int(s); sc.n_frames = frames
        _, a = engine2.run(sc, None, device=DEVICE)
        vals.append(float(a["food_delivered"]))
    return float(np.mean(vals)), vals


def render_winner(u, k):
    """Write a winner spec (full length) and render its gif+montage via render_ant."""
    p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
    sc = _apply(load(f"specs/{BASE}.yaml"), p)
    name = f"ant_opt_winner_{k}"
    # re-emit as a standalone spec file so the winner is reproducible + inspectable
    import yaml
    raw = yaml.safe_load(open(f"specs/{BASE}.yaml"))
    for o in raw["operators"]:
        if o["op"] == "motility": o["speed"], o["rot"] = float(p[0]), float(p[1])
        elif o["op"] == "trail": o["turn"], o["sensor_dist"] = float(p[2]), float(p[3])
        elif o["op"] == "secrete": o["rate"], o["runout"] = float(p[4]), float(p[5])
        elif o["op"] == "colony": o["perception"] = float(p[8])
        elif o["op"] == "mpm": o["drag"] = float(p[9])
    for f in raw["fields"].values():
        f["decay"], f["diffusion"] = float(p[6]), float(p[7])
    for t in raw["sets"]["cell"]["types"].values():
        t["youngs"] = float(p[10])
    raw["sets"]["particle"]["radius"] = float(p[11])
    raw["sets"]["cell"]["n"] = int(round(p[12]))
    raw["name"] = name
    open(f"specs/{name}.yaml", "w").write(yaml.safe_dump(raw, sort_keys=False))
    m = render_ant.render(name, device=DEVICE)
    return m["delivered"]


def write_index(winners):
    head = "| # | mean_delivered | " + " | ".join(PARAMS) + " | gif |"
    sep = "|" + "---|" * (len(PARAMS) + 3)
    lines = ["# ant colony optimization winners (auto-indexed by opt_ant.py)\n",
             "Objective = mean food delivered over seeds; each row beat the previous best "
             f"by >= {IMPROVE:.0f}.\n", head, sep]
    for w in winners:
        vals = " | ".join(f"{v:.3f}" for v in w["params"])
        lines.append(f"| {w['k']} | {w['mean']:.1f} | {vals} | [{w['gif']}]({w['gif']}) |")
    open("WINNERS_ant.md", "w").write("\n".join(lines) + "\n")


def main(budget=float("inf")):
    rng = np.random.default_rng(0)
    t0 = time.time()
    print(f"[start] ant colony opt: {D} levers (material+dynamics), "
          f"mean over seeds {EVAL_SEEDS}, {EVAL_FRAMES} frames/eval", flush=True)
    X, Y, winners_list, winners = [], [], [], 0
    best_so_far = -1e9

    def maybe_winner(u, f):
        nonlocal winners, best_so_far
        if f < best_so_far + IMPROVE:
            return
        winners += 1
        gif = f"ant_opt_winner_{winners}.gif"
        full = render_winner(u, winners)
        p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
        winners_list.append({"k": winners, "mean": f, "params": p, "gif": gif})
        write_index(winners_list)
        best_so_far = f
        print(f"[WINNER {winners}] mean_delivered={f:.1f} full_run={full} -> {gif}  "
              + ", ".join(f"{k}={v:.3f}" for k, v in zip(PARAMS, p)), flush=True)

    for _ in range(6):                                          # random exploration seed
        u = rng.random(D); f, vals = evaluate(u); X.append(u); Y.append(f)
        print(f"  eval {len(Y):3d} (seed): mean={f:.1f} {vals}  best={max(Y):.1f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, f)

    while time.time() - t0 < budget:
        Xa, Ya = np.array(X), np.array(Y)
        best_u = Xa[Ya.argmax()]
        glob = rng.random((2500, D))                            # global exploration
        loc = np.clip(best_u + rng.normal(0, 0.07, (2500, D)), 0, 1)   # local refinement
        cand = np.vstack([glob, loc])
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)
        w = np.exp(-(d / 0.22) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)
        unc = d.min(1)
        scale = (Ya.max() - Ya.min()) or 1.0
        u = cand[(mean + 0.6 * scale * unc).argmax()]           # UCB acquisition
        f, _ = evaluate(u); X.append(u); Y.append(f)
        print(f"  eval {len(Y):3d}: mean={f:.1f}  best={max(Y):.1f}  ({time.time()-t0:.0f}s)", flush=True)
        maybe_winner(u, f)

    print(f"[DONE] {len(Y)} evals, {winners} winners, best mean_delivered={max(Y):.1f}", flush=True)


if __name__ == "__main__":
    if "--test" in sys.argv:
        # smoke test: a few evaluations only, no winners/gifs (verifies the loop runs)
        import numpy as _np
        _rng = _np.random.default_rng(1)
        print("[TEST] running 3 evaluations (short, no winners) ...", flush=True)
        t0 = time.time()
        for i in range(3):
            u = _rng.random(D)
            f, vals = evaluate(u, frames=400, seeds=(0, 1))
            p = BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])
            print(f"  test eval {i+1}: mean_delivered={f:.1f} per-seed={vals}  ({time.time()-t0:.0f}s)", flush=True)
        print("[TEST OK] optimizer loop runs end-to-end.", flush=True)
    else:
        budget = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].replace('.', '').isdigit() else float("inf")
        main(budget)

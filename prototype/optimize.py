"""Optimize a simulation: run variants in the same maze to maximize food collected.

This prototypes "how do we optimize a differentiable system". The objective here
-- units of food delivered -- comes from a DISCRETE pick-up/drop-off state machine,
so it is non-differentiable: gradient descent through the rollout (the
Physical-Design-with-differentiable-simulators recipe) does not directly apply.
We therefore optimize the continuous behavioural parameters with a black-box
**UCB** search (a kNN-surrogate upper-confidence acquisition): exploit where food
is high, explore where the space is unsampled.

See METHODOLOGY discussion in docs/tissue_graph for when gradient descent applies
(smooth surrogate objective) and why UCB + gradient together is the general answer.

    python optimize.py            # ~20 evaluations, prints best, saves opt_curve.png
"""

from __future__ import annotations

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_schema import load
import engine2

# design space: behavioural parameters that affect foraging speed
PARAMS = ["motility.speed", "sense.gain", "mpm.drag", "motility.rot"]
BOUNDS = np.array([[20., 120.], [80., 400.], [0.5, 3.0], [0.10, 0.60]])
N_FRAMES = 1000
N_SEED = 4
N_UCB = 16
DEVICE = "cuda"


def _denorm(u):
    return BOUNDS[:, 0] + u * (BOUNDS[:, 1] - BOUNDS[:, 0])


def evaluate(u):
    p = _denorm(u)
    sc = load("scenarios/forage_maze.yaml")
    sc.n_frames = N_FRAMES
    for o in sc.operators:
        if o.op == "motility":
            o.params["speed"] = float(p[0]); o.params["rot"] = float(p[3])
        elif o.op == "sense":
            o.params["gain"] = float(p[1])
        elif o.op == "mpm":
            o.params["drag"] = float(p[2])
    _, a = engine2.run(sc, None, device=DEVICE)
    return float(a["food_delivered"])


def main():
    rng = np.random.default_rng(0)
    D = len(PARAMS)

    base = evaluate(np.full(D, 0.5))            # mid-space reference
    X, Y = [], []
    for _ in range(N_SEED):                     # random exploration seed
        u = rng.random(D); X.append(u); Y.append(evaluate(u))

    for it in range(N_UCB):                     # UCB acquisition loop
        Xa, Ya = np.array(X), np.array(Y)
        cand = rng.random((3000, D))
        d = np.linalg.norm(cand[:, None, :] - Xa[None, :, :], axis=2)   # [C,N]
        w = np.exp(-(d / 0.25) ** 2) + 1e-9
        mean = (w * Ya[None, :]).sum(1) / w.sum(1)                      # surrogate mean
        unc = d.min(1)                                                  # distance to nearest = uncertainty
        scale = (Ya.max() - Ya.min()) or 1.0
        acq = mean + 2.0 * scale * unc                                  # UCB
        u = cand[acq.argmax()]
        X.append(u); Y.append(evaluate(u))
        print(f"iter {it:2d}  food={Y[-1]:5.0f}  best={max(Y):5.0f}")

    Y = np.array(Y)
    best = int(Y.argmax())
    bp = _denorm(X[best])
    print(f"\nbaseline (mid-space) food = {base:.0f}")
    print(f"BEST food = {Y[best]:.0f}  at  "
          + ", ".join(f"{k}={v:.2f}" for k, v in zip(PARAMS, bp)))

    running = np.maximum.accumulate(Y)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(Y) + 1), Y, "o", color="#888", label="evaluation")
    plt.plot(range(1, len(Y) + 1), running, "-", color="#d62728", lw=2, label="best so far")
    plt.axvline(N_SEED + 0.5, ls="--", color="gray"); plt.text(N_SEED + 0.7, Y.min(), " UCB starts", fontsize=8)
    plt.axhline(base, ls=":", color="blue", label=f"mid-space baseline ({base:.0f})")
    plt.xlabel("simulation #"); plt.ylabel("food delivered"); plt.legend(); plt.tight_layout()
    plt.savefig("opt_curve.png", dpi=110)
    print("[saved] opt_curve.png")
    return X[best]


if __name__ == "__main__":
    main()

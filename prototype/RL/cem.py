"""cem.py -- the black-box inverse baseline (the bar RL must beat).

Cross-entropy method over a mechanism's lever box to match a TARGET trajectory
under the constellation reward. No learned network -- this is the amortization-free
baseline. It de-risks the program: if CEM cannot invert a held-out spec under this
metric, no RL will. Returns (best_u, best_dist, history).
"""
from __future__ import annotations

import numpy as np
from rollout import rollout
from metric import constellation_dist


def cem_match(mech, target_traj, rng, iters=40, batch=24, elite=0.25,
              std_floor=0.05, seed=0, device="cpu", on_iter=None):
    D = mech.D
    mean = np.full(D, 0.5)
    std = np.full(D, 0.25)
    n_elite = max(2, int(round(batch * elite)))
    best_d, best_u = float("inf"), None
    history = []
    for it in range(iters):
        U = np.clip(mean[None] + std[None] * rng.standard_normal((batch, D)), 0, 1)
        scored = []
        for u in U:
            tc = rollout(mech, u, seed=seed, device=device)
            d = constellation_dist(tc, target_traj) if tc is not None else float("inf")
            scored.append((d, u))
        scored.sort(key=lambda t: t[0])
        elites = np.array([u for d, u in scored[:n_elite] if np.isfinite(d)])
        if len(elites) >= 2:
            mean = elites.mean(0)
            std = np.maximum(elites.std(0), std_floor)
        if scored[0][0] < best_d:
            best_d, best_u = scored[0]
        history.append(best_d)
        if on_iter is not None:
            on_iter(it, best_d, best_u, std)
    return best_u, best_d, history

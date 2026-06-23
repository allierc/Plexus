"""overnight_multi.py -- UCB-tree inverse-matching across slime / boids / ar.

The sound preliminary experiment, run unattended over three 2-D mechanisms. For
each, repeatedly: draw a held-out ground-truth spec, simulate the TARGET, and run
the UCB tree (HOO) to recover its parameters under the particle-id-invariant
constellation metric (position + explicit velocity channel). Known-seed regime:
the candidate shares the target's seed, so the metric only has to separate specs
(floor 0) -- which is what gives boids/ar usable signal.

Per-mechanism device + budget; time-boxed by --hours, split evenly; resumable
(one JSON line per solved target under runs/multi/<mech>/results.jsonl).

    PYTHONPATH=../../src nohup python overnight_multi.py --hours 9 > runs/multi/log 2>&1 &
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_space import MECHANISMS
from rollout import rollout
from metric import constellation_dist
from ucb_tree import minimize

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:
    _CUDA = False

# slime has many small field ops -> faster on CPU; boids/ar are O(N^2)-ish -> GPU
PLAN = {
    "slime": dict(device="cpu",                      evals=250, mkw={}),
    "boids": dict(device="cuda" if _CUDA else "cpu", evals=300, mkw=dict(n_frames=24, L=16)),
    "ar":    dict(device="cuda" if _CUDA else "cpu", evals=300, mkw={}),
}


def solve_one(mech, k, device, evals, deadline, mkw=None):
    mkw = mkw or {}
    rng = np.random.default_rng(3000 + k)
    u_true = mech.sample_u(rng)
    target = rollout(mech, u_true, seed=0, device=device)   # target seed 0
    if target is None:
        return None
    n_eval = {"v": 0}

    def f(u):
        n_eval["v"] += 1
        tc = rollout(mech, u, seed=0, device=device)        # SHARED seed (known IC)
        return constellation_dist(tc, target, **mkw) if tc is not None else 10.0

    t0 = time.time()
    # honor the deadline: stop early if we run out of time mid-target
    def capped_f(u):
        if time.time() > deadline:
            raise TimeoutError
        return f(u)
    try:
        best_u, best_d, hist = minimize(capped_f, mech.D, evals)
    except TimeoutError:
        return None
    return dict(mech=mech.name, target=k, evals=n_eval["v"],
                start_dist=hist[0] if hist else None, best_dist=best_d,
                u_err=float(np.linalg.norm(best_u - u_true)),
                per_lever_err=np.abs(best_u - u_true).tolist(),
                levers=[l[:-2] for l in mech.levers],
                u_true=u_true.tolist(), best_u=best_u.tolist(),
                curve=hist, seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=9.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "multi"))
    ap.add_argument("--mechs", nargs="+", default=["slime", "boids", "ar"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    per = args.hours * 3600 / len(args.mechs)
    print(f"[multi] cuda={_CUDA}; {args.hours}h total, {per/3600:.1f}h/mechanism", flush=True)

    for mi, name in enumerate(args.mechs):
        mech = MECHANISMS[name]
        cfg = PLAN[name]
        deadline = time.time() + per
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        log = os.path.join(d, "results.jsonl")
        k = sum(1 for _ in open(log)) if os.path.exists(log) else 0
        print(f"[multi] === {name} (D={mech.D}, device={cfg['device']}, "
              f"evals={cfg['evals']}, resume@{k}) ===", flush=True)
        while time.time() < deadline:
            rec = solve_one(mech, k, cfg["device"], cfg["evals"], deadline, cfg.get("mkw"))
            if rec is None:
                if time.time() >= deadline:
                    break
                k += 1
                continue
            with open(log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"[multi] {name} target {k}: {rec['start_dist']:.3f} -> "
                  f"{rec['best_dist']:.3f}  u_err={rec['u_err']:.2f}  "
                  f"({rec['seconds']}s, {rec['evals']} evals)", flush=True)
            k += 1
    print("[multi] complete (or time-boxed).", flush=True)


if __name__ == "__main__":
    main()

"""test_inverse_baseline.py -- the CEM inversion baseline over many held-out targets.

For each of M random ground-truth specs, draw a target trajectory and run CEM to
invert it. Records the achievable behavioural distance and the parameter error
(u_err) per target. This is the BAR an RL policy must beat -- and the per-lever
u_err tells you which parameters are identifiable from behaviour and which are
degenerate (RL should be scored on behaviour for the latter). Resumable: appends
one JSON line per solved target. Runs until --n targets or killed.

    python test_inverse_baseline.py --n 40 --iters 30 --out runs/baseline
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
from cem import cem_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="slime")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "baseline"))
    ap.add_argument("--n", type=int, default=40)            # number of targets
    ap.add_argument("--iters", type=int, default=30)        # CEM iters per target
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mech = MECHANISMS[args.mech]
    log = os.path.join(args.out, "results.jsonl")
    done = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    print(f"[baseline] resuming: {done} targets already solved", flush=True)

    for k in range(done, args.n):
        rng = np.random.default_rng(1000 + k)               # per-target stream
        u_true = mech.sample_u(rng)
        target = rollout(mech, u_true, seed=0, device=args.device)
        if target is None:
            continue
        t0 = time.time()
        best_u, best_d, hist = cem_match(mech, target, rng, iters=args.iters,
                                         batch=args.batch, device=args.device)
        u_err = float(np.linalg.norm(best_u - u_true)) if best_u is not None else None
        per_lever = (np.abs(best_u - u_true)).tolist() if best_u is not None else None
        rec = dict(target=k, best_dist=best_d, u_err=u_err, per_lever_err=per_lever,
                   u_true=u_true.tolist(), best_u=best_u.tolist() if best_u is not None else None,
                   curve=hist, seconds=round(time.time() - t0, 1))
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[baseline] target {k}: dist={best_d:.4f} u_err={u_err:.3f} "
              f"({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

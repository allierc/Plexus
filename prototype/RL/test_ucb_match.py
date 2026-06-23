"""test_ucb_match.py -- preliminary inverse result via a UCB tree (HOO).

The sound, model-free baseline in the KNOWN-SEED regime: the candidate shares the
target's seed (same initial positions/velocities), so the reward is the DIRECT
per-cell, per-frame L2 (metric.cellwise_l2) -- deterministic, floor exactly 0 at the
true spec. HOO (ucb_tree.minimize) searches the 8-lever cube. Reports the
convergence curve, final distance, and parameter recovery (u_err + per-lever),
the numbers to compare later against gradient descent and the LLM loop (dicty-style).

    python test_ucb_match.py --evals 300 --targets 1 --out runs/ucb
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
from metric import cellwise_l2, constellation_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="slime")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "ucb"))
    ap.add_argument("--evals", type=int, default=300)
    ap.add_argument("--targets", type=int, default=1)
    ap.add_argument("--metric", default="constellation",
                    choices=["constellation", "cellwise"],
                    help="constellation = particle-id-INVARIANT OT (robust under chaos); "
                         "cellwise = id-dependent L2 (only valid at short --horizon w/ shared seed)")
    ap.add_argument("--horizon", type=int, default=15)   # cellwise only: pre-chaos window
    ap.add_argument("--seed", type=int, default=0)        # TARGET seed
    ap.add_argument("--cand-seed", type=int, default=7)   # candidate seed (!= target -> no id exploit)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mech = MECHANISMS[args.mech]
    from ucb_tree import minimize

    log = os.path.join(args.out, "results.jsonl")
    done = sum(1 for _ in open(log)) if os.path.exists(log) else 0

    for k in range(done, args.targets):
        rng = np.random.default_rng(2000 + k)
        u_true = mech.sample_u(rng)
        target = rollout(mech, u_true, seed=args.seed, device=args.device)
        if target is None:
            print(f"[ucb] target {k}: unstable truth, skipping", flush=True)
            continue
        # cellwise needs the SAME seed (particle correspondence); constellation does not
        cseed = args.seed if args.metric == "cellwise" else args.cand_seed

        def f(u):
            tc = rollout(mech, u, seed=cseed, device=args.device)
            if args.metric == "cellwise":
                return cellwise_l2(tc, target, frames=args.horizon)
            return constellation_dist(tc, target) if tc is not None else 10.0

        t0 = time.time()
        last = {"t": 0}

        def on_eval(t, best, bu):
            if t - last["t"] >= 25 or t == 0:
                print(f"[ucb] target {k} eval {t:4d}: best_dist={best:.5f} "
                      f"u_err={np.linalg.norm(bu - u_true):.3f}", flush=True)
                last["t"] = t

        best_u, best_d, hist = minimize(f, mech.D, args.evals, on_eval=on_eval)
        rec = dict(target=k, evals=args.evals, start_dist=hist[0], best_dist=best_d,
                   u_err=float(np.linalg.norm(best_u - u_true)),
                   per_lever_err=np.abs(best_u - u_true).tolist(),
                   levers=[l[:3] for l in mech.levers],
                   u_true=u_true.tolist(), best_u=best_u.tolist(),
                   curve=hist, seconds=round(time.time() - t0, 1))
        with open(log, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[ucb] target {k} DONE: {hist[0]:.4f} -> {best_d:.5f}  "
              f"u_err={rec['u_err']:.3f}  ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

"""test_characterize_metric.py -- establish the reward's signal-to-noise.

For RL you must know the reward's *noise floor* (how far same-spec/different-seed
runs sit apart) and its *dynamic range* (how far different specs sit apart). It
also builds a pairwise distance matrix over random specs to quantify
identifiability (degeneracy): if many distinct specs are mutually close, the
inverse problem is many-to-one and RL should target BEHAVIOUR, not parameters.

Writes <out>/characterize.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="slime")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "characterize"))
    ap.add_argument("--n-specs", type=int, default=24)     # for the pairwise matrix
    ap.add_argument("--n-seeds", type=int, default=4)      # seed-floor replicates
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mech = MECHANISMS[args.mech]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # --- seed floor: one spec, many seeds, all pairwise distances ---
    u0 = mech.sample_u(rng)
    seed_trajs = [rollout(mech, u0, seed=s, device=args.device) for s in range(args.n_seeds)]
    seed_trajs = [t for t in seed_trajs if t is not None]
    seed_d = [constellation_dist(seed_trajs[i], seed_trajs[j])
              for i in range(len(seed_trajs)) for j in range(i + 1, len(seed_trajs))]

    # --- dynamic range + identifiability: N random specs, full pairwise matrix ---
    U = np.array([mech.sample_u(rng) for _ in range(args.n_specs)])
    trajs = [rollout(mech, u, seed=0, device=args.device) for u in U]
    ok = [i for i, t in enumerate(trajs) if t is not None]
    M = len(ok)
    Dmat = np.zeros((M, M))
    for a in range(M):
        for b in range(a + 1, M):
            d = constellation_dist(trajs[ok[a]], trajs[ok[b]])
            Dmat[a, b] = Dmat[b, a] = d
    off = Dmat[np.triu_indices(M, 1)]
    floor = float(np.mean(seed_d)) if seed_d else float("nan")
    # degeneracy: fraction of distinct-spec pairs within 2x the seed floor
    degenerate = float(np.mean(off < 2 * floor)) if seed_d else float("nan")

    report = dict(
        mechanism=args.mech, D=mech.D, n_specs=M, n_seeds=len(seed_trajs),
        seed_floor_mean=floor, seed_floor_max=float(np.max(seed_d)) if seed_d else None,
        random_pair_min=float(off.min()), random_pair_mean=float(off.mean()),
        random_pair_max=float(off.max()),
        separation_ratio=float(off.mean() / floor) if floor else None,
        degenerate_fraction=degenerate,
        seconds=round(time.time() - t0, 1),
    )
    with open(os.path.join(args.out, "characterize.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(args.out, "pairwise.npy"), Dmat)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

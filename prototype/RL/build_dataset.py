"""build_dataset.py -- the training data for amortized RL / the recognition net.

Generate many (u, observation) pairs from the random spec generator, running the
codebase engine. The recognition network q(spec | observation) is pretrained by
supervised regression on these pairs (the cheap warm-start), then RL fine-tunes on
the constellation reward. We store a COMPACT observation, not full trajectories:
positions+velocities at K frames, subsampled to M cells per type -> (K, M, 4),
plus the label u and per-cell type. Runs until --n reached or killed; shards every
--shard samples so an overnight run is resumable and never loses work.

    python build_dataset.py --n 20000 --out runs/dataset
"""
from __future__ import annotations

import os
import sys
import time
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec_space import MECHANISMS
from rollout import rollout


def observation(traj, n_frames=24, n_cells=256, rng=None):
    """(K, M, 4) downsampled position+velocity cloud, per the recorded trajectory."""
    pos, nt = traj
    F = pos.shape[0] - 1
    fr = np.linspace(0, F - 1, n_frames).astype(int)
    vel = pos[1:] - pos[:-1]
    idx = np.arange(pos.shape[1])
    if pos.shape[1] > n_cells:
        idx = (rng or np.random).choice(idx, n_cells, replace=False)
    obs = np.stack([np.concatenate([pos[t][idx], vel[t][idx]], 1) for t in fr], 0)
    return obs.astype(np.float32), nt[idx].astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="slime")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs", "dataset"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--n-cells", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    mech = MECHANISMS[args.mech]
    rng = np.random.default_rng(args.seed)

    done = sum(int(np.load(p)["u"].shape[0]) for p in glob.glob(os.path.join(args.out, "shard_*.npz")))
    shard_id = len(glob.glob(os.path.join(args.out, "shard_*.npz")))
    print(f"[dataset] resuming: {done} samples already on disk", flush=True)

    buf_u, buf_obs, buf_nt = [], [], []
    t0 = time.time(); made = 0
    while done + made < args.n:
        u = mech.sample_u(rng)
        traj = rollout(mech, u, seed=int(rng.integers(0, 1 << 30)), device=args.device)
        if traj is None:
            continue
        obs, nt = observation(traj, args.n_frames, args.n_cells, rng)
        buf_u.append(u.astype(np.float32)); buf_obs.append(obs); buf_nt.append(nt)
        made += 1
        if len(buf_u) >= args.shard:
            path = os.path.join(args.out, f"shard_{shard_id:04d}.npz")
            np.savez_compressed(path, u=np.array(buf_u), obs=np.array(buf_obs),
                                node_type=np.array(buf_nt), levers=[l[:3] for l in mech.levers])
            rate = made / (time.time() - t0)
            print(f"[dataset] wrote {path}  total={done+made}  {rate:.2f} sims/s", flush=True)
            shard_id += 1; buf_u, buf_obs, buf_nt = [], [], []
    print(f"[dataset] done: {done+made} samples", flush=True)


if __name__ == "__main__":
    main()

"""metric.py -- the RL reward: constellation-to-constellation trajectory distance.

"Constellation to constellation" the only permutation-correct way: cell i in one
run is NOT cell i in another (different seeds -> different individual paths), so we
compare the point *distributions* with sliced-Wasserstein optimal transport --
within each type, on the joint (position, velocity) cloud, at each of several
frames, then averaged. velocity = pos[t+1]-pos[t]. Types may be exchangeable, so a
Hungarian assignment matches candidate types to GT types per frame.

Validated on slime: self (same seed)=0, same-spec/diff-seed ~0.06 (the irreducible
seed floor), random specs ~0.31 -> ~5x separation. The reward for RL is
-constellation_dist(candidate, target).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment   # Hungarian (type matching)
except Exception:
    linear_sum_assignment = None

_Q = np.linspace(0.0, 1.0, 64)        # quantile grid for 1-D OT (handles unequal N)


def _sliced_w(A, B, dirs):
    tot = 0.0
    for w in dirs:
        qa = np.quantile(np.sort(A @ w), _Q)
        qb = np.quantile(np.sort(B @ w), _Q)
        tot += np.abs(qa - qb).mean()
    return tot / len(dirs)


def constellation_dist(traj_c, traj_g, n_frames=40, L=24, hungarian=True, seed=0,
                       w_pos=1.0, w_vel=1.0):
    """Distance between two trajectories (pos:(F,N,2), node_type:(N,)).

    POSITION and VELOCITY are each compared as a SEPARATE id-invariant channel:
    a 2-D sliced-Wasserstein OT on the (x,y) cloud and another on the (vx,vy)
    cloud, per type per frame, combined as w_pos*d_pos + w_vel*d_vel. Splitting
    them (vs one 4-D joint OT) keeps velocity from being drowned out by position
    and lets the velocity field be matched on its own terms. Both channels are
    z-scored by the GT per-channel std so the weights are comparable. velocity =
    pos[t+1]-pos[t]; types matched by a per-frame Hungarian assignment."""
    pc, ntc = traj_c
    pg, ntg = traj_g
    pc = np.asarray(pc); pg = np.asarray(pg)
    F = min(pc.shape[0], pg.shape[0]) - 1
    fr = np.linspace(0, F - 1, min(n_frames, F)).astype(int)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((L, 2)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tc, tg = np.unique(ntc), np.unique(ntg)
    vg = pg[1:] - pg[:-1]
    sp = pg[:-1].reshape(-1, 2).std(0) + 1e-9        # GT position scale
    sv = vg.reshape(-1, 2).std(0) + 1e-9             # GT velocity scale

    def chan(a, b, dirs):                            # OT on a 2-D channel
        return _sliced_w(a, b, dirs)

    acc = []
    for t in fr:
        vc, vgt = pc[t + 1] - pc[t], pg[t + 1] - pg[t]
        C = np.zeros((len(tc), len(tg)))
        for i, kc in enumerate(tc):
            for j, kg in enumerate(tg):
                dp = chan(pc[t][ntc == kc] / sp, pg[t][ntg == kg] / sp, dirs)
                dv = chan(vc[ntc == kc] / sv, vgt[ntg == kg] / sv, dirs)
                C[i, j] = w_pos * dp + w_vel * dv
        if hungarian and linear_sum_assignment is not None and C.shape[0] == C.shape[1]:
            ri, ci = linear_sum_assignment(C)
            acc.append(C[ri, ci].mean())
        else:
            acc.append(np.mean([C[i, i] for i in range(min(C.shape))]))
    return float(np.mean(acc))


def reward(traj_c, traj_g, **kw):
    """RL reward = negative constellation distance (inf-safe)."""
    if traj_c is None:
        return -10.0
    return -constellation_dist(traj_c, traj_g, **kw)


def cellwise_l2(traj_c, traj_g, frames=None):
    """DIRECT per-cell, per-frame L2 -- valid ONLY when candidate and target share
    the seed (same initial positions -> cell i corresponds to cell i). Deterministic,
    dense, floor exactly 0 at the true spec.

    `frames` caps the horizon: chaotic dynamics decorrelate, so the FULL-trajectory
    L2 saturates to a flat ~constant (unsearchable). A SHORT horizon (first ~10-20
    frames, before trajectories diverge) is smooth and monotone in parameter
    distance -- the sound objective for parameter ID of chaotic systems."""
    if traj_c is None:
        return 10.0
    pc, _ = traj_c
    pg, _ = traj_g
    F = min(pc.shape[0], pg.shape[0])
    if frames is not None:
        F = min(F, int(frames))
    return float(np.linalg.norm(pc[:F] - pg[:F], axis=-1).mean())

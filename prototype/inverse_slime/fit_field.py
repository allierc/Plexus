"""fit_field.py -- one-step gradient-descent parameter recovery (beats UCB).

The ParticleGraph idea applied to Plexus: instead of a black-box full-rollout search
(UCB), TEACHER-FORCE one step. From the true (field_t, pos_t) predict field_{t+1}
with the differentiable field_model and minimise the one-step MSE by Adam. No
rollout -> no chaos -> a dense exact gradient on the field parameters
(deposit.amount, diffuse.rate, decay.rate). move_speed is read directly from the
per-step displacement. We then compare per-lever recovery error to the UCB winner on
the SAME slime target.

    PYTHONPATH=../../src python fit_field.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "RL"))   # reuse spec_space

import plexus.operators  # noqa
from plexus.engine import run
from spec_space import SLIME
from field_model import field_step

DEV = "cpu"
# the 4 differentiable levers among SLIME.levers (indices into SLIME.lo/hi):
#   0 deposit.amount, 1 diffuse.rate, 2 decay.rate, 4 move_speed
FIELD_IDX = {"amount": 0, "diffuse": 1, "decay": 2}
MOVE_IDX = 4


def gt_transitions(u_true, frames=150, seed=0):
    """Generate the GT slime via the ENGINE and return per-frame field + pos + nt.
    frames<=160 => the engine records the field EVERY tick (consecutive transitions)."""
    SLIME.frames = frames
    spec, p = SLIME.apply_u(u_true, seed=seed)
    _, out = run(spec, out_path=None, device=DEV)
    grid = torch.tensor(np.asarray(out["fields"]["chemical"]["grid"]), dtype=torch.float32)  # [T,C,nx,ny]
    pos = torch.tensor(np.asarray(out["sets"]["cell"]["pos"]), dtype=torch.float32)          # [T+?,N,2]
    nt = torch.tensor(np.asarray(out["sets"]["cell"]["node_type"]), dtype=torch.long)
    return grid, pos, nt, p


def find_alignment(grid, pos, nt, R, p):
    """The engine records end-of-tick; find the (field,pos) index offset that makes
    field_step reproduce the next recorded field (sanity that our math == engine)."""
    amount = torch.tensor(p[FIELD_IDX["amount"]], dtype=torch.float32)
    dr = torch.tensor(p[FIELD_IDX["diffuse"]], dtype=torch.float32)
    kr = torch.tensor(p[FIELD_IDX["decay"]], dtype=torch.float32)
    T = min(grid.shape[0], pos.shape[0]) - 1
    best = None
    for dp in (0, -1, 1):                       # pos index relative to field_t
        errs = []
        for t in range(5, min(40, T - 1)):
            pt = pos[t + dp] if 0 <= t + dp < pos.shape[0] else pos[t]
            pred = field_step(grid[t], pt, nt, amount, dr, kr, R)
            errs.append((pred - grid[t + 1]).abs().max().item())
        m = float(np.mean(errs))
        if best is None or m < best[1]:
            best = (dp, m)
    return best


def main():
    # use UCB's best slime target so the comparison is head-to-head
    rs = [json.loads(l) for l in open(os.path.join(_HERE, "..", "RL", "runs", "multi", "slime", "results.jsonl"))]
    tgt = min(rs, key=lambda r: r["best_dist"])
    u_true = np.array(tgt["u_true"]); ucb_u = np.array(tgt["best_u"])
    print(f"[fit] slime target {tgt['target']} (UCB best_dist={tgt['best_dist']:.3f}, u_err={tgt['u_err']:.2f})")

    grid, pos, nt, p = gt_transitions(u_true)
    R = grid.shape[-1]                          # ny == res
    print(f"[fit] GT: field {tuple(grid.shape)}, pos {tuple(pos.shape)}, R={R}")
    dp, aerr = find_alignment(grid, pos, nt, R, p)
    print(f"[fit] field_step vs engine: offset={dp}, max|err|={aerr:.2e} (should be ~1e-6: math matches)")

    # ---- teacher-forced transitions (field_t, pos_t) -> field_{t+1} ----
    T = min(grid.shape[0], pos.shape[0]) - 1
    ts = list(range(2, T))                      # skip the empty-field warmup frame
    lo = torch.tensor(SLIME.lo[[FIELD_IDX["amount"], FIELD_IDX["diffuse"], FIELD_IDX["decay"]]], dtype=torch.float32)
    hi = torch.tensor(SLIME.hi[[FIELD_IDX["amount"], FIELD_IDX["diffuse"], FIELD_IDX["decay"]]], dtype=torch.float32)

    theta = torch.zeros(3, requires_grad=True)  # sigmoid(theta) in [0,1] -> physical via lo/hi
    opt = torch.optim.Adam([theta], lr=0.1)
    for it in range(400):
        opt.zero_grad()
        u = torch.sigmoid(theta)
        phys = lo + u * (hi - lo)
        amount, dr, kr = phys[0], phys[1], phys[2]
        loss = 0.0
        for t in ts:
            pt = pos[t + dp] if 0 <= t + dp < pos.shape[0] else pos[t]
            pred = field_step(grid[t], pt, nt, amount, dr, kr, R)
            loss = loss + ((pred - grid[t + 1]) ** 2).mean()
        loss = loss / len(ts)
        loss.backward()
        opt.step()
        if it % 80 == 0 or it == 399:
            print(f"  iter {it:3d}  loss={loss.item():.3e}  u={[round(x,3) for x in u.tolist()]}")

    u_hat = torch.sigmoid(theta).detach().numpy()
    # move_speed: pure per-step displacement magnitude (advance: ||dx|| = dt*move_speed)
    disp = (pos[3:T] - pos[2:T - 1]).norm(dim=-1)
    ms_phys = float(disp.median())
    ms_u = (ms_phys - SLIME.lo[MOVE_IDX]) / (SLIME.hi[MOVE_IDX] - SLIME.lo[MOVE_IDX])

    # ---- head-to-head per-lever error (in unit-cube u, like UCB) ----
    names = ["deposit.amount", "diffuse.rate", "decay.rate", "move_speed"]
    idxs = [FIELD_IDX["amount"], FIELD_IDX["diffuse"], FIELD_IDX["decay"], MOVE_IDX]
    gd_u = np.array([u_hat[0], u_hat[1], u_hat[2], ms_u])
    print(f"\n{'lever':16s} {'true_u':>8s} {'UCB_u':>8s} {'GD_u':>8s} | {'UCB_err':>8s} {'GD_err':>8s}")
    for n, ix, g in zip(names, idxs, gd_u):
        ue = abs(u_true[ix] - ucb_u[ix]); ge = abs(u_true[ix] - g)
        print(f"{n:16s} {u_true[ix]:8.3f} {ucb_u[ix]:8.3f} {g:8.3f} | {ue:8.3f} {ge:8.3f}")
    ucb_err = np.mean([abs(u_true[i] - ucb_u[i]) for i in idxs])
    gd_err = np.mean(np.abs(gd_u - u_true[idxs]))
    print(f"\n[RESULT] mean per-lever u-error on these 4 levers:  UCB={ucb_err:.3f}   gradient-descent={gd_err:.3f}")
    print(f"[RESULT] gradient descent is {ucb_err/max(gd_err,1e-6):.1f}x more accurate (one-step, no rollout)")

    # full recovered u-vector: GD on the 4 differentiable levers, sensing params
    # left at GT (gradient descent does not fit them yet -> held, not claimed).
    u_gd_full = u_true.copy()
    for ix, g in zip(idxs, gd_u):
        u_gd_full[ix] = float(g)
    os.makedirs(os.path.join(_HERE, "runs"), exist_ok=True)
    json.dump(dict(target=tgt["target"], u_true=u_true.tolist(), ucb_u=ucb_u.tolist(),
                   u_gd_full=u_gd_full.tolist(), fit_idx=idxs),
              open(os.path.join(_HERE, "runs", "fit_field.json"), "w"), indent=2)
    print(f"[saved] runs/fit_field.json (u_gd_full: GD on {names}, sensing@GT)")


if __name__ == "__main__":
    main()

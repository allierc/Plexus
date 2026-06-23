"""exp_multitype.py -- expand the inverse training to TWO-type and FOUR-type slime.

Same one-step position-loss training, but the chemical field now has C channels and
`sense` couples types via `cross` -- which is UNIDENTIFIABLE for single-type and becomes
recoverable here. Field ops are already multi-channel; sense uses mu_vec_mc. Reports
per-parameter recovery (incl. cross) and the operator self-check, on slime2 + slime4.
"""
import os, sys, datetime, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL"))
import plexus.operators  # noqa
from plexus.engine import run
from spec_space import SLIME2, SLIME4
from operators import LearnableDeposit, LearnableDiffuse, LearnableDecay
from sense_trainer import fit_sense_pos_mc, DEFAULT

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")
DEV = "cpu"
NAMES = ["amount", "diffuse", "decay", "cross", "move", "turn", "sensA", "sensD"]


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def u_of(v, M, i):
    return (v - M.lo[i]) / (M.hi[i] - M.lo[i])


def gen_gt(M, u_true, frames=150):
    M.frames = frames
    spec, p = M.apply_u(u_true, seed=0)
    _, out = run(spec, out_path=None, device=DEV)
    grid = torch.tensor(np.asarray(out["fields"]["chemical"]["grid"]), dtype=torch.float32)  # [T,C,nx,ny]
    pos = torch.tensor(np.asarray(out["sets"]["cell"]["pos"]), dtype=torch.float32)
    nt = torch.tensor(np.asarray(out["sets"]["cell"]["node_type"]), dtype=torch.long)
    return grid, pos, nt, p


def recover_field_mc(grid, pos, nt, R):
    """Multi-channel exact field fit (deposit writes per-type channel via real nt)."""
    M_lo = SLIME2.lo  # same field bounds for slime2/slime4
    dep = LearnableDeposit(0.5, M_lo[0], SLIME2.hi[0])
    dif = LearnableDiffuse(0.5, M_lo[1], SLIME2.hi[1])
    dec = LearnableDecay(0.5, M_lo[2], SLIME2.hi[2])
    opt = torch.optim.Adam(list(dep.parameters()) + list(dif.parameters()) + list(dec.parameters()), lr=0.1)
    ts = list(range(2, min(grid.shape[0], pos.shape[0]) - 1))
    for _ in range(300):
        opt.zero_grad(); loss = 0.0
        for t in ts:
            loss = loss + ((dec(dif(dep(grid[t], pos[t], nt, R))) - grid[t + 1]) ** 2).mean()
        (loss / len(ts)).backward(); opt.step()
    move = float((pos[3:120] - pos[2:119]).norm(dim=-1).median())
    return float(dep.amount()), float(dif.rate()), float(dec.rate()), move


def main():
    logw(f"\n## MULTI-TYPE (2 & 4 types) [{datetime.datetime.now().strftime('%H:%M:%S')}]: "
         f"same one-step position-loss training; `cross` now recoverable (multi-channel field).")
    for M in (SLIME2, SLIME4):
        rng = np.random.default_rng(7)
        u_true = M.sample_u(rng)
        grid, pos, nt, p = gen_gt(M, u_true)
        R = grid.shape[-1]; C = grid.shape[1]
        a, d, k, move = recover_field_mc(grid, pos, nt, R)
        sense_lo = np.array([M.lo[5], M.lo[6], M.lo[7]]); sense_hi = np.array([M.hi[5], M.hi[6], M.hi[7]])
        r = fit_sense_pos_mc(grid, pos, nt, p, R, {**DEFAULT}, sense_lo, sense_hi,
                             float(M.lo[3]), float(M.hi[3]), seed=0, schedule=(1,), iters_per=800)
        rec = {0: a, 1: d, 2: k, 3: r["cross"], 4: move, 5: r["turn_speed"], 6: r["sensor_angle"], 7: r["sensor_dist"]}
        logw(f"\n  [{M.name}, C={C} channels] per-parameter recovery (u-space):")
        logw(f"  {'param':9s} {'true_u':>7s} {'rec_u':>7s} {'err':>6s}")
        for i, nm in enumerate(NAMES):
            tu = float(u_true[i]); ru = float(u_of(rec[i], M, i))
            mark = "  <- now identifiable" if i == 3 else ""
            logw(f"  {nm:9s} {tu:7.3f} {ru:7.3f} {abs(tu-ru):6.3f}{mark}")


if __name__ == "__main__":
    main()

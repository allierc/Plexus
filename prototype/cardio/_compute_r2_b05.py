#!/usr/bin/env python
"""Offline R2 (honest render_fit) for batch 5 slots, which ran on the pre-metric script.
Replicates cardio_train08_09.main()'s final render_fit block exactly, using each slot's
saved fit_pos.npy + the GT real render + the interior&moving mask."""
import os, sys
os.environ.setdefault("CARDIO_BWIDTH", "11")
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import cardio_real_fit as R  # noqa: E402

_G = 137
GRID = (_G, _G)
N_CYCLES_RENDER, PERIOD, REC = 6, int(R.T_FIT), 2
dev = torch.device("cpu")


def openness(traj):
    c = traj - traj.mean(0, keepdim=True)
    xx = (c[..., 0] ** 2).mean(0); yy = (c[..., 1] ** 2).mean(0); xy = (c[..., 0] * c[..., 1]).mean(0)
    tr = xx + yy; det = (xx * yy - xy ** 2)
    disc = (tr * tr / 4 - det).clamp(min=1e-12).sqrt()
    lam_min = (tr / 2 - disc).clamp(min=0)
    return (lam_min + 1e-12).sqrt()


def loop_r2(pred, gt, mask):
    p = pred[:, mask]; g = gt[:, mask]
    res = (p - g).pow(2).sum()
    tot = (g - g.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-20)
    return 1 - res / tot


geo, _ = R.setup_fit(GRID)
geo = tuple(g.to(dev) for g in geo)
N = GRID[0] * GRID[1]
beats_cpu = R.real_all_beats(GRID); nb = beats_cpu.shape[0]
beats_up = torch.stack([R.upsample_beat(beats_cpu[b], R.T_FIT) for b in range(nb)]).to(dev)
open_real = [openness(beats_up[b]) for b in range(nb)]
open_mask = [(o > 0.10 * o.max()) for o in open_real]
bnd = geo[4].bool() if len(geo) > 4 else torch.zeros(N, dtype=torch.bool, device=dev)
interior = ~bnd
mov = interior & torch.stack(open_mask).any(0)
real_rec = R.real_full(GRID, N_CYCLES_RENDER * PERIOD)[::REC]

slots = [
    "loop_b05_s0_parent_gfloor055",
    "loop_b05_s1_gfloor065",
    "loop_b05_s2_gfloor075",
    "loop_b05_s3_ablation_gfloor035",
    "loop_b05_s4_lamopen4",
    "loop_b05_s5_trans006",
]
print(f"N_moving_interior={int(mov.sum())}  N_interior={int(interior.sum())}  nb={nb}")
for s in slots:
    fp = os.path.join(HERE, "archive", s, "fit_pos.npy")
    if not os.path.exists(fp):
        print(f"{s:38s} MISSING fit_pos.npy"); continue
    fit_pos = np.load(fp)
    ip = torch.tensor(fit_pos, device=dev)
    Fr = min(ip.shape[0], real_rec.shape[0])
    rr = torch.tensor(np.asarray(real_rec)[:Fr], device=dev); ip = ip[:Fr]
    if torch.isfinite(ip).all():
        r2 = loop_r2(ip, rr, mov).item(); nrmse = (max(0.0, 1.0 - r2)) ** 0.5
    else:
        r2 = float("nan"); nrmse = float("nan")
    # amplitude diagnostic: ratio of pred motion RMS to GT motion RMS over moving nodes
    g = rr[:, mov]; p = ip[:, mov]
    gm = (g - g.mean(0, keepdim=True)).pow(2).mean().sqrt().item()
    pm_ = (p - p.mean(0, keepdim=True)).pow(2).mean().sqrt().item()
    print(f"{s:38s} R2={r2:+.3f}  NRMSE={nrmse:.3f}  predRMS/gtRMS={pm_/gm:.2f}  frames={Fr}")

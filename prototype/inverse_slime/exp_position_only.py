"""exp_position_only.py -- can the FIELD params be recovered from PARTICLE POSITIONS
ALONE, with NO field loss and the field never observed? (The cardio question.)

Closed-loop free-run: start from the GT field + positions at t0, then evolve the field
with LEARNABLE deposit/diffuse/decay (its OWN field, teacher-forced only at t0) and move
particles by sense+advance reading that self-generated field. Loss is ONLY on particle
positions. Profile a field param (diffuse.rate) under this position-only loss: a well at
truth => the latent field is identifiable from motion alone.
"""
import os, sys, datetime
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import _wrap, DEFAULT
from operators import LearnableDeposit, LearnableDiffuse, LearnableDecay, LearnableSense
from spec_space import SLIME

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def closed_loop_pos_loss(T, dep, dif, dec, sense, K, starts, move):
    """Free-run the FULL model (field + particles) from GT IC; loss on positions only."""
    grid, pos, R = T["grid"], T["pos"], T["R"]
    d = pos[1:] - pos[:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    nt = torch.zeros(pos.shape[1], dtype=torch.long, device=pos.device)
    margin = move + float(T["p"][7]) + 0.02
    tot, wsum = 0.0, 0.0
    for t0 in starts:
        t0 = int(t0)
        fld = grid[t0].clone()                                # [1,nx,ny] initial field (teacher-forced ONCE)
        p = pos[t0].clone()
        h = theta[t0 - 1]
        cx, sx = torch.cos(h), torch.sin(h)
        interior0 = ((pos[t0][:, 0] > margin) & (pos[t0][:, 0] < 1 - margin) &
                     (pos[t0][:, 1] > margin) & (pos[t0][:, 1] < 1 - margin))
        for k in range(K):
            fld = dec(dif(dep(fld, p, nt, R)))                # learnable field step (own field)
            mu = sense.mu_vec(fld, torch.zeros(p.shape[0], dtype=torch.long, device=p.device),
                              p, cx, sx, R)
            cm, sm = torch.cos(mu), torch.sin(mu)
            cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
            p = p + move * torch.stack([cx, sx], -1)
            tgt = pos[t0 + k + 1]
            valid = (interior0 & (dth[t0 + k - 1].abs() < 1.0)).float()
            tot = tot + (((p - tgt) ** 2).sum(-1) * valid).sum(); wsum = wsum + valid.sum()
    return float(tot / (wsum + 1e-6))


def main():
    T = A.load_targets()[0]
    lo, hi = SLIME.lo, SLIME.hi
    dev = T["pos"].device
    K = 10
    g = np.random.default_rng(0)
    starts = g.integers(2, min(T["pos"].shape[0] - 1, 120) - K, size=4)
    move = float(T["p"][4])
    # sense + deposit + decay fixed at TRUTH; profile diffuse.rate under position-only loss
    def mk(cls, idx, u):
        m = cls(u, lo[idx], hi[idx]).to(dev)
        for q in m.parameters():
            q.requires_grad_(False)
        return m
    u_dep = A.u_of(float(T["p"][0]), 0); u_dec = A.u_of(float(T["p"][2]), 2)
    dep = mk(LearnableDeposit, 0, u_dep); dec = mk(LearnableDecay, 2, u_dec)
    sense = LearnableSense(A.u_of(float(T["p"][5]), 5), A.u_of(float(T["p"][6]), 6),
                           A.u_of(float(T["p"][7]), 7), A.SENSE_LO, A.SENSE_HI).to(dev)
    for q in sense.parameters():
        q.requires_grad_(False)
    logw(f"\n## POSITION-ONLY field test [{datetime.datetime.now().strftime('%H:%M:%S')}]: profile "
         f"diffuse.rate from PARTICLE POSITIONS ALONE (field free-run, never observed). K={K}.")
    u_true = A.u_of(float(T["p"][1]), 1)
    us = np.linspace(0.05, 0.95, 19); losses = []
    for u in us:
        dif = mk(LearnableDiffuse, 1, float(u))
        with torch.no_grad():
            losses.append(closed_loop_pos_loss(T, dep, dif, dec, sense, K, starts, move))
    losses = np.array(losses); amin = float(us[int(losses.argmin())])
    depth = float((losses.max() - losses.min()) / (losses.min() + 1e-9))
    verdict = ("IDENTIFIABLE from positions (dip at truth)" if abs(amin - u_true) < 0.12
               else "weak/biased" if depth > 0.1 else "UNIDENTIFIABLE from positions (flat)")
    logw(f"- diffuse.rate: true_u={u_true:.2f} argmin={amin:.2f} depth={depth:.3f} -> {verdict}")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(us, losses, "-o", ms=3); ax.axvline(u_true, color="g", ls="--", label="true")
    ax.axvline(amin, color="r", ls=":", label="argmin"); ax.legend(fontsize=7)
    ax.set_xlabel("diffuse.rate (u)"); ax.set_ylabel("position-only loss")
    ax.set_title(f"field from positions alone: depth={depth:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "archive", "profile_field_from_pos.png"), dpi=110)
    logw("- saved archive/profile_field_from_pos.png")


if __name__ == "__main__":
    main()

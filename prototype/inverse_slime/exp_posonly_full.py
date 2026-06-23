"""exp_posonly_full.py -- PURE position-only recovery (no field loss): fit ALL 8 params
(field + motion + sense) from particle POSITIONS alone, field free-run & never observed.

Closed loop: field starts at GT field[t0] then self-evolves with LEARNABLE
deposit/diffuse/decay; particles move by sense+advance reading that self-made field;
loss only on positions. Shows which params positions can pin -- expected: motion+sense
recover, FIELD params do not (cardio's latent-field lesson). Saves the recovered vector
for the figure.
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RL"))
import autorun as A
from sense_trainer import _wrap, DEFAULT
from operators import LearnableDeposit, LearnableDiffuse, LearnableDecay, LearnableSense
from spec_space import SLIME

NAMES = ["amount", "diffuse", "decay", "cross", "move", "turn", "sensA", "sensD"]


def main():
    T = A.load_targets()[0]
    grid, pos, R = T["grid"], T["pos"], T["R"]
    dev = grid.device
    nt = torch.zeros(pos.shape[1], dtype=torch.long, device=dev)
    d = pos[1:] - pos[:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    move = float(T["p"][4]); margin = move + float(T["p"][7]) + 0.02
    lo, hi = SLIME.lo, SLIME.hi
    dep = LearnableDeposit(0.5, lo[0], hi[0]).to(dev); dif = LearnableDiffuse(0.5, lo[1], hi[1]).to(dev)
    dec = LearnableDecay(0.5, lo[2], hi[2]).to(dev)
    sense = LearnableSense(0.5, 0.5, 0.5, np.array([lo[5], lo[6], lo[7]]), np.array([hi[5], hi[6], hi[7]])).to(dev)
    params = list(dep.parameters()) + list(dif.parameters()) + list(dec.parameters()) + list(sense.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    g = np.random.default_rng(0); K = 10; Tmax = min(theta.shape[0] - 1, 120)
    for it in range(250):
        if it % 25 == 0:
            starts = g.integers(2, Tmax - K, size=4)
        loss, wsum = 0.0, 0.0
        for t0 in starts:
            t0 = int(t0); fld = grid[t0].clone(); p = pos[t0].clone()
            cx, sx = torch.cos(theta[t0 - 1]), torch.sin(theta[t0 - 1])
            interior = ((pos[t0][:, 0] > margin) & (pos[t0][:, 0] < 1 - margin) &
                        (pos[t0][:, 1] > margin) & (pos[t0][:, 1] < 1 - margin)).float()
            for k in range(K):
                fld = dec(dif(dep(fld, p, nt, R)))                 # learnable field, self-evolved
                mu = sense.mu_vec(fld[0:1], torch.zeros(p.shape[0], dtype=torch.long, device=dev), p, cx, sx, R)
                cm, sm = torch.cos(mu), torch.sin(mu); cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
                p = p + move * torch.stack([cx, sx], -1)
                valid = interior * (dth[t0 + k - 1].abs() < 1.0).float()
                loss = loss + (((p - pos[t0 + k + 1]) ** 2).sum(-1) * valid).sum(); wsum = wsum + valid.sum()
        loss = loss / (wsum + 1e-6)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 2.5); opt.step()
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in sense.params())
        rec = [float(dep.amount()), float(dif.rate()), float(dec.rate()), float(T["p"][3]),
               move, ts, ang * 180 / np.pi, sd]
    u_rec = [(rec[i] - SLIME.lo[i]) / (SLIME.hi[i] - SLIME.lo[i]) for i in range(8)]
    u_true = [float(x) for x in T["u_true"]]
    print("param    true_u  rec_u   err", flush=True)
    for i, n in enumerate(NAMES):
        print(f"{n:8s} {u_true[i]:6.3f} {u_rec[i]:6.3f} {abs(u_true[i]-u_rec[i]):6.3f}"
              + ("  <- field (latent: should fail)" if i < 3 else ""), flush=True)
    json.dump(dict(u_true=u_true, u_rec=u_rec),
              open(os.path.join(os.path.dirname(__file__), "runs", "posonly_full.json"), "w"))


if __name__ == "__main__":
    main()

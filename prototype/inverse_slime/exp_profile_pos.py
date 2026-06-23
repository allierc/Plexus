"""exp_profile_pos.py -- loss ONLY ON PARTICLES (positions), and its sensor_angle profile.

Instead of matching the reconstructed heading (cos,sin), FREE-RUN the full particle
dynamics (sense -> rotate heading -> advance position) and match the PREDICTED
POSITIONS to GT positions per frame. This removes the atan2(displacement) heading
reconstruction from the loss -- a candidate source of the sensor_angle bias. Then
profile sensor_angle (turn,dist,move at truth) under the position loss and compare the
argmin to the heading-loss profile. Logs + saves archive/profile_pos.png.
"""
import os, sys, math, datetime
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autorun as A
from sense_trainer import _wrap, DEFAULT
from operators import LearnableSense

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "logbook2.md")


def logw(s):
    open(LOG, "a").write(s + "\n"); print(s, flush=True)


def build_pos_windows(T, cfg, K, B=24, cells=256):
    pos = T["pos"]; grid = T["grid"]
    d = pos[1:] - pos[:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    move, sdist = float(T["p"][4]), float(T["p"][7]); margin = move + sdist + 0.02
    off = int(cfg["field_offset"]); dev = pos.device; g = np.random.default_rng(0)
    Tmax = min(theta.shape[0] - 1, 140)
    starts = g.integers(1, max(2, Tmax - K - 1), size=B)
    wid, init, p0 = [], [], []
    fld, ptgt, bnc = [[] for _ in range(K)], [[] for _ in range(K)], [[] for _ in range(K)]
    nwin = 0
    for t0 in starts:
        t0 = int(t0); pt = pos[t0]
        interior = ((pt[:, 0] > margin) & (pt[:, 0] < 1 - margin) &
                    (pt[:, 1] > margin) & (pt[:, 1] < 1 - margin))
        ids = torch.nonzero(interior).squeeze(-1)
        if ids.numel() < 10:
            continue
        if ids.numel() > cells:
            ids = ids[torch.from_numpy(g.permutation(ids.numel())[:cells]).to(dev)]
        wid.append(torch.full((ids.numel(),), nwin, device=dev))
        init.append(theta[t0 - 1][ids]); p0.append(pos[t0][ids])
        for k in range(K):
            fld[k].append(grid[min(t0 + k + off, grid.shape[0] - 1), 0])
            ptgt[k].append(pos[t0 + k + 1][ids])
            bnc[k].append(dth[t0 + k - 1][ids].abs() > cfg["max_turn"])
        nwin += 1
    if not init:
        return None
    return (torch.cat(wid), torch.cat(init), torch.cat(p0), move,
            [torch.stack(f) for f in fld], [torch.cat(p) for p in ptgt], [torch.cat(b) for b in bnc])


def pos_loss(s, bat, K, R):
    wid, init, p0, move, fld_k, ptgt_k, bnc_k = bat
    cx, sx = torch.cos(init), torch.sin(init)
    pos = p0.clone(); loss, wsum = 0.0, 0.0
    for k in range(K):
        mu = s.mu_vec(fld_k[k], wid, pos, cx, sx, R)
        cm, sm = torch.cos(mu), torch.sin(mu)
        cx, sx = cx * cm - sx * sm, cx * sm + sx * cm           # rotate heading
        pos = pos + move * torch.stack([cx, sx], -1)            # advance position
        tgt = ptgt_k[k]; valid = (~bnc_k[k]).float()
        loss = loss + (((pos - tgt) ** 2).sum(-1) * valid).sum(); wsum = wsum + valid.sum()
        rs = bnc_k[k].float()[:, None]
        pos = pos * (1 - rs) + tgt * rs                         # resync position at bounce
        # also resync heading at bounce (to GT direction at that step)
        gh = torch.atan2((ptgt_k[k] - (ptgt_k[k - 1] if k > 0 else p0))[:, 1],
                         (ptgt_k[k] - (ptgt_k[k - 1] if k > 0 else p0))[:, 0])
        rb = bnc_k[k].float()
        cx = cx * (1 - rb) + torch.cos(gh) * rb; sx = sx * (1 - rb) + torch.sin(gh) * rb
    return float(loss / (wsum + 1e-6))


def main():
    targets = A.load_targets(); cfg = {**DEFAULT}; K = 20; n = 25
    lo, hi = A.SENSE_LO, A.SENSE_HI; angs = np.linspace(0.02, 0.98, n)
    logw(f"\n## PROFILE-POS [{datetime.datetime.now().strftime('%H:%M:%S')}]: loss ONLY ON PARTICLES "
         f"(positions, free-run sense->advance). sensor_angle profile, K={K}.")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 3.0))
    for ti, T in enumerate(targets):
        bat = build_pos_windows(T, cfg, K)
        u_ts, u_sd, u_true = float(T["u_true"][5]), float(T["u_true"][7]), float(T["u_true"][6])
        losses = []
        for ua in angs:
            s = LearnableSense(u_ts, ua, u_sd, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(T["pos"].device)
            for p in s.parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                losses.append(pos_loss(s, bat, K, T["R"]))
        losses = np.array(losses); amin = float(angs[int(losses.argmin())])
        depth = float((losses.max() - losses.min()) / (losses.min() + 1e-9))
        verdict = ("UNIDENTIFIABLE" if depth < 0.15 else "IDENTIFIABLE (dip at truth)"
                   if abs(amin - u_true) < 0.12 else "BIASED")
        logw(f"- target{ti}: true={u_true:.2f} argmin={amin:.2f} depth={depth:.2f} -> {verdict}")
        ax = axes[ti]; ax.plot(angs, losses, "-"); ax.axvline(u_true, color="g", ls="--")
        ax.axvline(amin, color="r", ls=":"); ax.set_title(f"t{ti} d={depth:.2f}", fontsize=8)
        ax.set_xlabel("angle (u)", fontsize=7)
    os.makedirs(os.path.join(os.path.dirname(__file__), "archive"), exist_ok=True)
    fig.suptitle("sensor_angle profile -- LOSS ON PARTICLES (positions)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "archive", "profile_pos.png"), dpi=110)
    logw("- saved archive/profile_pos.png")


if __name__ == "__main__":
    main()

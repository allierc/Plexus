"""sense_trainer.py -- configurable one-step trainer for the `sense` surrogate.

Reconstructs per-step heading turns from a GT trajectory (interior, non-bounced,
moved cells), then fits LearnableSense by Gaussian NLL. Every knob the autorun
search tunes lives in `cfg`: lr, iters, restarts, beta0/temp0 init, beta annealing,
weight decay, field-index alignment, turn cutoff, sample size, lr schedule.

Pure/reproducible: given the same (transitions, cfg, seed) it returns the same fit.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from operators import LearnableSense, LearnableSenseMargin

DEFAULT = dict(lr=0.05, iters=400, restarts=2, beta0=12.0, temp0=0.05,
               anneal=False, wd=0.0, field_offset=1, max_turn=1.0,
               n_per_frame=300, sched="cos", max_frames=40)


def reconstruct(grid, pos, p_true, R, cfg):
    """One BATCHED sample set across strided frames: fields:[F,nx,ny] + flat per-sample
    (frame_idx, pos, pre-heading, observed turn). Single tensor -> one training call."""
    d = pos[1:] - pos[:-1]
    theta = torch.atan2(d[..., 1], d[..., 0])                       # post-sense heading per tick
    T = theta.shape[0]
    move, sdist = float(p_true[4]), float(p_true[7])
    margin = move + sdist + 0.02
    off = int(cfg["field_offset"])
    g = np.random.default_rng(0)
    hi_t = min(T - 1, 140)
    frame_ids = np.unique(np.linspace(1, hi_t - 1, int(cfg.get("max_frames", 40))).astype(int))
    fields, fidx, poss, preh, turns = [], [], [], [], []
    for j, t in enumerate(frame_ids):
        t = int(t); pt = pos[t]
        interior = ((pt[:, 0] > margin) & (pt[:, 0] < 1 - margin) &
                    (pt[:, 1] > margin) & (pt[:, 1] < 1 - margin))
        turn = (theta[t] - theta[t - 1] + math.pi) % (2 * math.pi) - math.pi
        moved = d[t].norm(dim=-1) > 1e-5
        ids = torch.nonzero(interior & moved & (turn.abs() < cfg["max_turn"])).squeeze(-1)
        if ids.numel() > cfg["n_per_frame"]:
            ids = ids[torch.from_numpy(g.permutation(ids.numel())[:cfg["n_per_frame"]]).to(ids.device)]
        if ids.numel() > 10:
            fields.append(grid[min(max(t + off, 0), grid.shape[0] - 1), 0])
            poss.append(pt[ids]); preh.append(theta[t - 1][ids]); turns.append(turn[ids])
            fidx.append(torch.full((ids.numel(),), len(fields) - 1, device=ids.device))
    if not fields:
        return None
    return (torch.stack(fields), torch.cat(fidx), torch.cat(poss), torch.cat(preh), torch.cat(turns))


def _fit_once(batch, lo, hi, R, cfg, seed):
    fields, fidx, poss, preh, turns = batch
    dev = fields.device
    rng = np.random.default_rng(seed)
    init = rng.uniform(0.2, 0.8, 3) if seed > 0 else np.full(3, 0.5)
    s = LearnableSense(init[0], init[1], init[2], lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
    opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"])
             if cfg["sched"] == "cos" else None)
    npos = cfg.get("noise", 0.0)                                    # input position jitter (world units)
    nh = cfg.get("noise_h", 0.0)                                    # pre-heading jitter (radians)
    for it in range(cfg["iters"]):
        bs = (0.25 + 3.75 * it / cfg["iters"]) if cfg["anneal"] else 1.0   # beta anneal
        # add a little noise in training: jitter the inputs each step (regularises +
        # smooths the rugged sensor-geometry landscape). Decays to 0 over training.
        if npos > 0 or nh > 0:
            decay = 1.0 - it / cfg["iters"]
            pj = poss + npos * decay * torch.randn_like(poss)
            hj = preh + nh * decay * torch.randn_like(preh)
        else:
            pj, hj = poss, preh
        opt.zero_grad()
        s.nll_b(fields, fidx, pj, hj, turns, R, beta_scale=bs).backward()
        opt.step()
        if sched: sched.step()
    with torch.no_grad():
        final = s.nll_b(fields, fidx, poss, preh, turns, R).item()
    return s, final


def fit_sense(grid, pos, p_true, R, cfg, lo, hi, seed=0):
    """Returns (ts, sensor_angle_deg, sensor_dist, diagnostics) -- best of N restarts."""
    batch = reconstruct(grid, pos, p_true, R, cfg)
    if batch is None:
        return None
    best, best_nll = None, float("inf")
    for r in range(cfg["restarts"]):
        s, nll = _fit_once(batch, lo, hi, R, cfg, seed=seed * 100 + r)
        if nll < best_nll:
            best, best_nll = s, nll
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in best.params())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi,
                sensor_dist=sd, nll=best_nll, n_samples=int(batch[1].numel()))


def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _pos_windows(grid, pos, p_true, R, cfg, K, B, cells, g):
    """Windows for the POSITION loss: per step k -> field, GT-next-position, bounce."""
    d = pos[1:] - pos[:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    move, sdist = float(p_true[4]), float(p_true[7]); margin = move + sdist + 0.02
    off = int(cfg["field_offset"]); dev = pos.device
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
            ptgt[k].append(pos[t0 + k + 1][ids]); bnc[k].append(dth[t0 + k - 1][ids].abs() > cfg["max_turn"])
        nwin += 1
    if not init:
        return None
    return (torch.cat(wid), torch.cat(init), torch.cat(p0), move,
            [torch.stack(f) for f in fld], [torch.cat(p) for p in ptgt], [torch.cat(b) for b in bnc])


def fit_sense_curriculum_pos(grid, pos, p_true, R, cfg, lo, hi, seed=0,
                             schedule=(10, 20, 30), iters_per=400, B=16, cells=256):
    """LOSS ON PARTICLES: free-run sense->advance and match PREDICTED POSITIONS to GT
    (no heading reconstruction in the loss -> removes the sensor_angle bias). move_speed
    fixed at its known value; curriculum over horizon; resync position+heading at bounces."""
    dev = pos.device
    s = LearnableSense(0.5, 0.5, 0.5, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
    opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
    g = np.random.default_rng(seed)
    resample = int(cfg.get("resample", 25))
    for stage, Kreq in enumerate(schedule):
        K = int(min(Kreq, min(pos.shape[0], grid.shape[0]) - 3))
        for pg in opt.param_groups:
            pg["lr"] = cfg["lr"] * (0.6 ** stage)
        bat = None
        for it in range(iters_per):
            if it % resample == 0:
                bat = _pos_windows(grid, pos, p_true, R, cfg, K, B, cells, g)
            if bat is None:
                continue
            wid, init, p0, move, fld_k, ptgt_k, bnc_k = bat
            rec_noise = float(cfg.get("rec_noise", 0.0))            # connectome-style recurrent state noise
            cx, sx = torch.cos(init), torch.sin(init)
            pp = p0; loss, wsum = 0.0, 0.0
            for k in range(K):
                mu = s.mu_vec(fld_k[k], wid, pp, cx, sx, R)
                cm, sm = torch.cos(mu), torch.sin(mu)
                cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
                if rec_noise > 0:                                  # perturb the heading state each step
                    an = rec_noise * torch.randn_like(cx); cn, sn = torch.cos(an), torch.sin(an)
                    cx, sx = cx * cn - sx * sn, cx * sn + sx * cn
                pp = pp + move * torch.stack([cx, sx], -1)
                tgt = ptgt_k[k]; valid = (~bnc_k[k]).float()
                loss = loss + (((pp - tgt) ** 2).sum(-1) * valid).sum(); wsum = wsum + valid.sum()
                rs = bnc_k[k].float()[:, None]; pp = pp * (1 - rs) + tgt * rs
            loss = loss / (wsum + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s.parameters(), cfg.get("grad_clip", 2.5)); opt.step()
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in s.params())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi, sensor_dist=sd)


def _windows(grid, pos, p_true, R, cfg, K):
    """K-step windows: cells interior + unbounced throughout. Per step k a stacked
    field [nwin,nx,ny] + per-sample pos; plus init heading and GT net K-step turn."""
    d = pos[1:] - pos[:-1]
    theta = torch.atan2(d[..., 1], d[..., 0])
    T = theta.shape[0]
    move, sdist = float(p_true[4]), float(p_true[7]); margin = move + sdist + 0.02
    off = int(cfg["field_offset"])
    g = np.random.default_rng(0)
    starts = np.unique(np.linspace(2, min(T - 1, 140) - K - 1, cfg.get("n_windows", 24)).astype(int))
    wid, init_h, gt_net = [], [], []
    fld_k = [[] for _ in range(K)]; pos_k = [[] for _ in range(K)]
    nwin = 0
    for t0 in starts:
        t0 = int(t0)
        keep = torch.ones(pos.shape[1], dtype=torch.bool, device=pos.device)
        for k in range(K):
            pk = pos[t0 + k]
            keep &= (pk[:, 0] > margin) & (pk[:, 0] < 1 - margin) & (pk[:, 1] > margin) & (pk[:, 1] < 1 - margin)
            keep &= _wrap(theta[t0 + k] - theta[t0 + k - 1]).abs() < cfg["max_turn"]
        ids = torch.nonzero(keep).squeeze(-1)
        if ids.numel() > cfg["n_per_frame"]:
            ids = ids[torch.from_numpy(g.permutation(ids.numel())[:cfg["n_per_frame"]]).to(ids.device)]
        if ids.numel() < 10:
            continue
        wid.append(torch.full((ids.numel(),), nwin, device=ids.device))
        init_h.append(theta[t0 - 1][ids]); gt_net.append(_wrap(theta[t0 + K - 1][ids] - theta[t0 - 1][ids]))
        for k in range(K):
            fld_k[k].append(grid[min(t0 + k + off, grid.shape[0] - 1), 0]); pos_k[k].append(pos[t0 + k][ids])
        nwin += 1
    if nwin == 0:
        return None
    return (torch.cat(wid), torch.cat(init_h), torch.cat(gt_net),
            [torch.stack(f) for f in fld_k], [torch.cat(p) for p in pos_k])


def fit_sense_recurrent(grid, pos, p_true, R, cfg, lo, hi, seed=0, K=6):
    """RECURRENT fit: unroll the heading K steps (model turns along GT positions) and
    match the NET K-step heading change. The accumulation amplifies sensor_angle, which
    one-step data underdetermines."""
    W = _windows(grid, pos, p_true, R, cfg, K)
    if W is None:
        return None
    wid, init_h, gt_net, fld_k, pos_k = W
    dev = init_h.device
    best, best_loss = None, float("inf")
    for r in range(cfg["restarts"]):
        rng = np.random.default_rng(seed * 100 + r)
        init = rng.uniform(0.2, 0.8, 3) if r > 0 else np.full(3, 0.5)
        s = LearnableSense(init[0], init[1], init[2], lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
        opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"]) if cfg["sched"] == "cos" else None)
        for it in range(cfg["iters"]):
            h = init_h.clone()
            for k in range(K):
                h = h + s.dist_b(fld_k[k], wid, pos_k[k], h, R)[0]      # expected turn each step
            loss = (_wrap((h - init_h) - gt_net) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if sched: sched.step()
        with torch.no_grad():
            h = init_h.clone()
            for k in range(K):
                h = h + s.dist_b(fld_k[k], wid, pos_k[k], h, R)[0]
            fl = (_wrap((h - init_h) - gt_net) ** 2).mean().item()
        if fl < best_loss:
            best, best_loss = s, fl
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in best.params())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi, sensor_dist=sd, loss=best_loss)


def _curr_batch(grid, pos, theta, dth, starts, K, off, margin, cells, g, dev, max_turn):
    wid, init = [], []
    fld = [[] for _ in range(K)]; pp = [[] for _ in range(K)]
    tg = [[] for _ in range(K)]; bn = [[] for _ in range(K)]
    w = 0
    for t0 in starts:
        t0 = int(t0); pt = pos[t0]
        interior = ((pt[:, 0] > margin) & (pt[:, 0] < 1 - margin) &
                    (pt[:, 1] > margin) & (pt[:, 1] < 1 - margin))
        ids = torch.nonzero(interior).squeeze(-1)
        if ids.numel() < 10:
            continue
        if ids.numel() > cells:
            ids = ids[torch.from_numpy(g.permutation(ids.numel())[:cells]).to(dev)]
        wid.append(torch.full((ids.numel(),), w, device=dev)); init.append(theta[t0 - 1][ids])
        for k in range(K):
            fld[k].append(grid[min(t0 + k + off, grid.shape[0] - 1), 0])
            pp[k].append(pos[t0 + k][ids]); tg[k].append(theta[t0 + k][ids])
            bn[k].append(dth[t0 + k - 1][ids].abs() > max_turn)
        w += 1
    if not init:
        return None
    return (torch.cat(wid), torch.cat(init), [torch.stack(f) for f in fld],
            [torch.cat(p) for p in pp], [torch.cat(t) for t in tg], [torch.cat(b) for b in bn])


def fit_sense_curriculum(grid, pos, p_true, R, cfg, lo, hi, seed=0,
                         schedule=(10, 30, 60, 100, 140), iters_per=300, B=16, cells=256, on_stage=None):
    """The connectome-cx CURRICULUM scheme applied to slime heading integration:
    FREE-RUN the heading (cos,sin) along GT positions, score the PER-FRAME (cos,sin)
    error over a GROWING horizon (schedule), from RANDOM start frames, with a soft tail
    weight, LR decay per stage, grad clipping, and bounce-aware resync (teacher-force
    the heading back to GT at wall bounces, which inject un-predictable random reheads)."""
    d = pos[1:] - pos[:-1]
    theta = torch.atan2(d[..., 1], d[..., 0]); T = theta.shape[0]
    dth = _wrap(theta[1:] - theta[:-1])
    move, sdist = float(p_true[4]), float(p_true[7]); margin = move + sdist + 0.02
    off = int(cfg["field_offset"]); dev = pos.device
    Tmax = min(T - 1, grid.shape[0] - 1)
    tail = cfg.get("tail", 0.035)
    s = LearnableSense(0.5, 0.5, 0.5, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
    opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
    g = np.random.default_rng(seed)
    K0 = schedule[0]
    for stage, Kreq in enumerate(schedule):
        K = int(min(Kreq, Tmax - 2))
        for pg in opt.param_groups:
            pg["lr"] = cfg["lr"] * (0.6 ** stage)                  # per-stage lr decay
        resample = int(cfg.get("resample", 25))                    # fresh random starts every N iters
        cos_init = None
        for it in range(iters_per):
            if it % resample == 0:                                 # rebuild the window pool (diversity)
                starts = g.integers(1, max(2, Tmax - K), size=B)
                bat = _curr_batch(grid, pos, theta, dth, starts, K, off, margin, cells, g, dev, cfg["max_turn"])
                if bat is None:
                    continue
                wid, init, fld_k, pos_k, tgt_k, bnc_k = bat
                cos_init, sin_init = torch.cos(init), torch.sin(init)
            if cos_init is None:
                continue
            cx, sx = cos_init, sin_init
            loss = 0.0; wsum = 0.0
            for k in range(K):
                mu = s.mu_vec(fld_k[k], wid, pos_k[k], cx, sx, R)
                cm, sm = torch.cos(mu), torch.sin(mu)
                cx, sx = cx * cm - sx * sm, cx * sm + sx * cm        # rotate by predicted turn
                tc, ts_ = torch.cos(tgt_k[k]), torch.sin(tgt_k[k])
                valid = (~bnc_k[k]).float()
                fw = 1.0 if k < K0 else tail                        # soft-tail curriculum weight
                err = ((cx - tc) ** 2 + (sx - ts_) ** 2) * valid * fw
                loss = loss + err.sum(); wsum = wsum + valid.sum() * fw
                rs = bnc_k[k].float()                               # resync heading at bounces
                cx = cx * (1 - rs) + tc * rs; sx = sx * (1 - rs) + ts_ * rs
            loss = loss / (wsum + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s.parameters(), cfg.get("grad_clip", 2.5))
            opt.step()
        if on_stage is not None:
            with torch.no_grad():
                ts, ang, sd = (float(x) for x in s.params())
            on_stage(stage, K, ts, ang * 180 / math.pi, sd)
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in s.params())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi, sensor_dist=sd)


def _pos_windows_mc(grid, pos, nt, p_true, R, cfg, K, B, cells, g):
    """Multi-channel position windows: per step k -> flat per-channel fields, total
    field, per-sample type, GT-next-position, bounce. grid:[T,C,nx,ny], nt:[N]."""
    C = grid.shape[1]
    d = pos[1:] - pos[:-1]; theta = torch.atan2(d[..., 1], d[..., 0])
    dth = _wrap(theta[1:] - theta[:-1])
    move, sdist = float(p_true[4]), float(p_true[7]); margin = move + sdist + 0.02
    off = int(cfg["field_offset"]); dev = pos.device
    Tmax = min(theta.shape[0] - 1, 140)
    starts = g.integers(1, max(2, Tmax - K - 1), size=B)
    wid, init, p0, tid = [], [], [], []
    fflat, total, ptgt, bnc = [[] for _ in range(K)], [[] for _ in range(K)], [[] for _ in range(K)], [[] for _ in range(K)]
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
        wid.append(torch.full((ids.numel(),), nwin, device=dev)); init.append(theta[t0 - 1][ids])
        p0.append(pos[t0][ids]); tid.append(nt[ids].long())
        for k in range(K):
            fi = min(t0 + k + off, grid.shape[0] - 1)
            gC = grid[fi]                                          # [C,nx,ny]
            fflat[k].append(gC); total[k].append(gC.sum(0))       # per-channel + summed
            ptgt[k].append(pos[t0 + k + 1][ids]); bnc[k].append(dth[t0 + k - 1][ids].abs() > cfg["max_turn"])
        nwin += 1
    if not init:
        return None
    fflat_k = [torch.cat(f, 0) for f in fflat]                    # [nwin*C, nx, ny] each step
    total_k = [torch.stack(t) for t in total]                    # [nwin, nx, ny]
    return (torch.cat(wid), torch.cat(init), torch.cat(p0), torch.cat(tid), move, C,
            fflat_k, total_k, [torch.cat(p) for p in ptgt], [torch.cat(b) for b in bnc])


def fit_sense_pos_mc(grid, pos, nt, p_true, R, cfg, lo, hi, cross_lo, cross_hi,
                     seed=0, schedule=(1,), iters_per=800, B=16, cells=256):
    """MULTI-TYPE one-step position-loss fit -- recovers turn/angle/dist AND cross
    (inter-type sensing weight), which only multi-type data identifies."""
    dev = pos.device
    s = LearnableSense(0.5, 0.5, 0.5, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"],
                       u_cross=0.5, cross_lo=cross_lo, cross_hi=cross_hi).to(dev)
    opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
    g = np.random.default_rng(seed); resample = int(cfg.get("resample", 25))
    for stage, Kreq in enumerate(schedule):
        K = int(min(Kreq, min(pos.shape[0], grid.shape[0]) - 3))
        for pg in opt.param_groups:
            pg["lr"] = cfg["lr"] * (0.6 ** stage)
        bat = None
        for it in range(iters_per):
            if it % resample == 0:
                bat = _pos_windows_mc(grid, pos, nt, p_true, R, cfg, K, B, cells, g)
            if bat is None:
                continue
            wid, init, p0, tid, move, C, ff_k, tot_k, ptgt_k, bnc_k = bat
            cx, sx = torch.cos(init), torch.sin(init); pp = p0; loss, wsum = 0.0, 0.0
            for k in range(K):
                mu = s.mu_vec_mc(ff_k[k], tot_k[k], wid, tid, pp, cx, sx, R, C)
                cm, sm = torch.cos(mu), torch.sin(mu)
                cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
                pp = pp + move * torch.stack([cx, sx], -1)
                tgt = ptgt_k[k]; valid = (~bnc_k[k]).float()
                loss = loss + (((pp - tgt) ** 2).sum(-1) * valid).sum(); wsum = wsum + valid.sum()
                rs = bnc_k[k].float()[:, None]; pp = pp * (1 - rs) + tgt * rs
            loss = loss / (wsum + 1e-6)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s.parameters(), cfg.get("grad_clip", 2.5)); opt.step()
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in s.params()); cross = float(s.cross_val())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi, sensor_dist=sd, cross=cross)


def fit_sense_recurrent_cossin(grid, pos, p_true, R, cfg, lo, hi, seed=0, K=6):
    """Recurrent unroll with heading as a UNIT VECTOR (cos,sin): each turn is a ROTATION
    (no wrap), loss = MSE between final and GT heading VECTORS (the connectome-cx recipe)."""
    W = _windows(grid, pos, p_true, R, cfg, K)
    if W is None:
        return None
    wid, init_h, gt_net, fld_k, pos_k = W
    dev = init_h.device
    gt_final = init_h + gt_net                                          # GT final heading (angle)
    gtc, gts = torch.cos(gt_final), torch.sin(gt_final)
    c0, s0 = torch.cos(init_h), torch.sin(init_h)
    best, best_loss = None, float("inf")
    for r in range(cfg["restarts"]):
        rng = np.random.default_rng(seed * 100 + r)
        init = rng.uniform(0.2, 0.8, 3) if r > 0 else np.full(3, 0.5)
        s = LearnableSense(init[0], init[1], init[2], lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
        opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"]) if cfg["sched"] == "cos" else None)
        def unroll():
            cx, sx = c0.clone(), s0.clone()
            for k in range(K):
                mu = s.mu_vec(fld_k[k], wid, pos_k[k], cx, sx, R)
                cm, sm = torch.cos(mu), torch.sin(mu)                   # rotate by the turn
                cx, sx = cx * cm - sx * sm, cx * sm + sx * cm
            return cx, sx
        for it in range(cfg["iters"]):
            cx, sx = unroll()
            loss = ((cx - gtc) ** 2 + (sx - gts) ** 2).mean()          # MSE on (cos,sin)
            opt.zero_grad(); loss.backward(); opt.step()
            if sched: sched.step()
        with torch.no_grad():
            cx, sx = unroll(); fl = ((cx - gtc) ** 2 + (sx - gts) ** 2).mean().item()
        if fl < best_loss:
            best, best_loss = s, fl
    with torch.no_grad():
        ts, ang, sd = (float(x) for x in best.params())
    return dict(turn_speed=ts, sensor_angle=ang * 180 / math.pi, sensor_dist=sd, loss=best_loss)


def fit_sense_hybrid(grid, pos, p_true, R, cfg, lo, hi, seed=0, ngrid=11, inner=120):
    """UCB/GD HYBRID ('use both'): black-box 2-D grid over (sensor_angle, sensor_dist)
    -- the non-smooth params -- with GRADIENT DESCENT on (turn_speed, beta, temp) at each
    grid point, scored by the one-step mixture NLL. Returns the min-NLL geometry."""
    batch = reconstruct(grid, pos, p_true, R, cfg)
    if batch is None:
        return None
    fields, fidx, poss, preh, turns = batch
    dev = fields.device
    import math as _m
    def logit(u): return _m.log(min(max(u, 1e-3), 1 - 1e-3) / (1 - min(max(u, 1e-3), 1 - 1e-3)))
    best = (float("inf"), 0.5, 0.5, 0.5)
    axis = np.linspace(0.05, 0.95, ngrid)
    for ua in axis:
        for ud in axis:
            s = LearnableSense(0.5, ua, ud, lo, hi, beta=cfg["beta0"], temp=cfg["temp0"]).to(dev)
            s.t_ang.data = torch.tensor(logit(ua)); s.t_ang.requires_grad_(False)
            s.t_sd.data = torch.tensor(logit(ud)); s.t_sd.requires_grad_(False)
            opt = torch.optim.Adam([s.t_ts, s.log_beta, s.log_temp], lr=cfg["lr"])
            for _ in range(inner):
                opt.zero_grad()
                s.nll_b(fields, fidx, poss, preh, turns, R).backward()
                opt.step()
            with torch.no_grad():
                nll = s.nll_b(fields, fidx, poss, preh, turns, R).item()
                ts = float(s.params()[0])
            if nll < best[0]:
                best = (nll, ua, ud, ts)
    nll, ua, ud, ts = best
    ang = float(lo[1] + ua * (hi[1] - lo[1])); sd = float(lo[2] + ud * (hi[2] - lo[2]))
    return dict(turn_speed=ts, sensor_angle=ang, sensor_dist=sd, nll=nll, n_samples=int(fidx.numel()))


def fit_sense_margin(grid, pos, p_true, R, cfg, lo, hi, seed=0):
    """No-Gumbel fit: ts closed-form from |turn|; sensor_angle/dist by margin loss."""
    batch = reconstruct(grid, pos, p_true, R, cfg)
    if batch is None:
        return None
    fields, fidx, poss, preh, turns = batch
    dev = fields.device
    ts_est = float((2.0 * turns.abs().mean()).clamp(lo[0], hi[0]))     # E|turn| = ts/2
    best, best_loss = None, float("inf")
    for r in range(cfg["restarts"]):
        rng = np.random.default_rng(seed * 100 + r)
        init = rng.uniform(0.2, 0.8, 2) if r > 0 else np.full(2, 0.5)
        s = LearnableSenseMargin(init[0], init[1], lo, hi).to(dev)
        opt = torch.optim.Adam(s.parameters(), lr=cfg["lr"])
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["iters"])
                 if cfg["sched"] == "cos" else None)
        for it in range(cfg["iters"]):
            opt.zero_grad()
            s.margin_loss(fields, fidx, poss, preh, turns, R).backward()
            opt.step()
            if sched: sched.step()
        with torch.no_grad():
            fl = s.margin_loss(fields, fidx, poss, preh, turns, R).item()
        if fl < best_loss:
            best, best_loss = s, fl
    with torch.no_grad():
        ang, sd = (float(x) for x in best.geom())
    return dict(turn_speed=ts_est, sensor_angle=ang * 180 / math.pi,
                sensor_dist=sd, loss=best_loss, n_samples=int(fidx.numel()))

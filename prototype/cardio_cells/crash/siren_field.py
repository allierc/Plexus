"""siren_field.py -- ROUND 7, TASK 1.  Does analytic differentiation of a FITTED CONTINUOUS FIELD
escape the boxcar attenuation, or does the network merely hide the same smoothing?

THE CLAIM UNDER TEST
====================================================================================================
`freal_derivedF.py` established that deriving F the way a recording must -- bin the displacement onto
a control grid of spacing h and central-difference it -- collapses the per-cell (E, gain) solve from
med|dE/E| 0.0078 (simulator's own F) to 0.999, with ZERO measurement noise.  A central difference
over 2h is the boxcar average of the derivative; the attenuation eats the signal the fit lives on.

The proposal: do not finite-difference at all.  Fit u_hat(x, t) with a SIREN to the SAME observed
(binned, frame-cadence) displacement and differentiate it ANALYTICALLY:

    F = I + grad_X u_hat        v = d u_hat / dt        C = (grad_X v) F^-1

All three derived quantities come from one field.  The risk is that a SIREN is itself a smoothness
prior: over-smooth and the attenuation is reproduced inside the network instead of in a stencil.
omega is the knob and is SWEPT, never chosen.

WHAT THIS SCRIPT MEASURES (task 1 only -- no solve)
----------------------------------------------------------------------------------------------------
  stage c  collect.  Plant the same system (C=100, per_parent=100, float64, seed 2026), warm to
           t_lo, walk to t_hi, and cache per frame: the measured positions x0, the simulator's F0,
           v0, C0, plus the tick-0 reference configuration X (the recording's frame-0 grid).
  stage f  fit + measure.  For each control spacing h and each omega:
             1. FIT QUALITY of the field itself -- rel err of u_hat (at the nodes it was trained on
                AND at the particles), of F, of v, of C, against the simulator's own values;
                and the ATTENUATION RATIO for F (per-channel least-squares slope of F_hat - I on
                F_true - I, plus the pooled slope).
             2. PER-CELL CONTRAST -- between-cell variance of F over within-cell variance, for the
                true field, the central difference and the SIREN.
           The central difference on the SAME grid and the true field are rows of the same table.

TWO REFERENCE LAGS, because they answer different questions
  m = 165  the displacement is measured from tick 0, F_ref = I: FULLY REALIZABLE, no oracle.
  m = 2    the RELATIVE gradient between t-2 and t, composed with the true F(t-2): needs an oracle
           for F_ref, but it isolates the pure smoothing question from the accumulated-reference
           problem.  freal_derivedF measured both for the central difference (rel_fro 0.135 and
           0.0124), and its per-particle least-squares floor (0.147 / 0.0139) says the m=165 gap is
           NOT a grid artefact -- so if the SIREN is only fighting smoothing it can only win at m=2.

CONTROLS REPRODUCED FROM freal_derivedF.json (t0 = 165, same seed, interior mask)
  central difference, m165, h=15px : ls_scale [0.688 0.458 0.423 0.614]  corr [0.35 0.65 0.60 0.30]
                                     rel_fro 0.1350  med|Fd-Ft| 0.01936  (|F-I| median 0.01543)
  central difference, m165, h=34px : ls_scale [0.773 0.298 0.226 0.835]  rel_fro 0.1666
  central difference, m2,   h=15px : ls_scale [1.003 0.980 0.977 1.001]  rel_fro 0.0124
  NOTE the 0.003 quoted for "the central difference's attenuation" in the brief is mean(E_hat)/mean(E)
  from the SOLVE (fits.F_der_m165_15px.pstats.mean_ratio_E = 0.00332), not the F attenuation ratio;
  the F attenuation ratio of the same configuration is 0.42-0.69.  Both are reported here.

usage:
  PYTHONPATH=/workspace/Plexus/src python siren_field.py --device cuda:0 --stages cf
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

from assemble import System, SUBSTEP_TOKENS                          # noqa: E402
from recover import install_E                                        # noqa: E402
import metrics as MET                                                # noqa: E402
from refute_round3 import advance                                    # noqa: E402
from round5_fit import SNAP                                          # noqa: E402
from freal_derivedF import ControlGrid, derive_F, attenuation, rel, PX, GRID_PX   # noqa: E402
from train import Siren                                              # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  stage c -- collect.  freal_derivedF.plant_and_warm_x0 + state_derive's v0/C0 capture.
# --------------------------------------------------------------------------------------------- #
def plant_and_warm_x0(args, log, seed=2026):
    sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode=args.mode)
    C = sy.C
    g = torch.Generator().manual_seed(seed)
    E = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    gn = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    sy.E_true[1:], sy.gain_true[1:] = E, gn
    sy.theta_true = torch.cat([sy.E_true[1:], sy.gain_true[1:]])
    install_E(sy, sy.E_true)
    X0 = sy.p.get("pos").clone()                       # tick 0 = the MPM reference configuration
    F00 = sy.p.F.clone()
    for tick in range(args.warmup):
        sy._outer(tick, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
    sy.warmup_frames = args.warmup
    sy._snapshot(args.warmup)
    I2 = torch.eye(2, device=sy.device, dtype=sy.dtype)
    log(f"[planted] C={sy.C} Np={sy.Np} grid {sy.g.nx}^2 dtype={sy.dtype} dt={sy.dt} "
        f"n_sub={sy.n_sub_per_frame}  warm-up {args.warmup} frames")
    log(f"          |F(tick 0) - I|max = {float((F00 - I2).abs().max()):.2e}  "
        f"(0 => X is the configuration p.F is measured against)")
    return sy, X0


def collect(args, t_lo, t_hi, log):
    a2 = SimpleNamespace(**{**vars(args), "warmup": t_lo})
    sy, X0 = plant_and_warm_x0(a2, log)
    B, cur = {}, t_lo
    for k in range(t_hi - t_lo + 1):
        if k > 0:
            sy.restore()
            advance(sy, cur, cur + 1)
            sy._snapshot(cur + 1)
            cur += 1
        B[cur] = {"x0": sy.x0.clone(), "F0": sy.F0.clone(),
                  "v0": sy.v0.clone(), "C0": sy.C0.clone()}
    return sy, X0, B


# --------------------------------------------------------------------------------------------- #
#  the observation.  EXACTLY what the recording hands over: a boxcar mean of the displacement on a
#  regular control grid, once per frame.  Nodes with no particle are NOT used as training data (the
#  central difference has to dilate them to have a stencil; a fitted field does not, and inventing
#  data there would be worse than leaving it out).
# --------------------------------------------------------------------------------------------- #
def observe(cg, X0, B, ticks):
    """-> node coords [M,2], per-frame node displacement [T,M,2], valid mask."""
    node_i = torch.arange(cg.nx, device=X0.device)
    node_j = torch.arange(cg.ny, device=X0.device)
    gi, gj = torch.meshgrid(node_i, node_j, indexing="ij")
    P = torch.stack([cg.lo[0] + cg.h * gi.to(X0.dtype),
                     cg.lo[1] + cg.h * gj.to(X0.dtype)], -1).reshape(-1, 2)
    U = []
    for t in ticks:
        Ub = cg.bin_(B[t]["x0"] - X0)                       # [nx, ny, 2] (dilated copy)
        U.append(Ub.reshape(-1, 2))
    U = torch.stack(U)                                      # [T, nnodes, 2]
    v = cg.valid
    return P[v], U[:, v], v


# --------------------------------------------------------------------------------------------- #
#  the field
# --------------------------------------------------------------------------------------------- #
class Field(torch.nn.Module):
    """u_hat(X, t): a SIREN on normalised (x, y, t), with the normalisation constants attached so
    that every derivative comes back in WORLD units."""

    def __init__(self, Xc, Xs, tc, ts, uc, us, omega, hidden, layers, tw=1.0, dtype=torch.float32):
        super().__init__()
        self.net = Siren(3, hidden, layers, 2, outermost_linear=True,
                         first_omega_0=omega, hidden_omega_0=omega).to(dtype)
        self.register_buffer("Xc", Xc.to(dtype))
        self.register_buffer("Xs", torch.as_tensor(float(Xs), dtype=dtype))
        self.register_buffer("tc", torch.as_tensor(float(tc), dtype=dtype))
        self.register_buffer("ts", torch.as_tensor(float(ts), dtype=dtype))
        self.register_buffer("uc", uc.to(dtype))
        self.register_buffer("us", torch.as_tensor(float(us), dtype=dtype))
        self.tw = float(tw)

    def coords(self, X, t):
        xn = (X - self.Xc) / self.Xs
        tn = ((t - self.tc) / self.ts) * self.tw
        return torch.cat([xn, tn], -1)

    def u(self, X, t):
        return self.net(self.coords(X, t)) * self.us + self.uc

    def derivs(self, X, t):
        """-> u, F, v, C in world units.  Two first-order and two second-order backward passes."""
        c = self.coords(X, t).detach().requires_grad_(True)
        y = self.net(c)
        J = []
        for i in range(2):
            J.append(torch.autograd.grad(y[:, i].sum(), c, create_graph=True)[0])
        J = torch.stack(J, 1)                                   # [N,2,3] d y_i / d c_j
        H = []
        for i in range(2):
            H.append(torch.autograd.grad(J[:, i, 2].sum(), c, retain_graph=True)[0][:, :2])
        H = torch.stack(H, 1)                                   # [N,2,2] d(dy_i/dtn)/d xn_j
        us, Xs, ts = self.us, self.Xs, self.ts * (1.0 / self.tw)
        u = y * us + self.uc
        gradu = J[:, :, :2] * (us / Xs)                         # d u_i / d X_j
        I2 = torch.eye(2, device=X.device, dtype=X.dtype)[None]
        F = I2 + gradu
        v = J[:, :, 2] * (us / ts)
        gradXv = H * (us / (Xs * ts))
        C = gradXv @ torch.linalg.inv(F)
        return (u.detach(), F.detach(), v.detach(), C.detach())


def fit_field(P, T, U, omega, hidden, layers, iters, lr, tw, dtype, device, seed=0, log=None,
              batch=0):
    """Least-squares fit of u_hat to the observed node displacements.  P [M,2], T [nT,1], U [nT,M,2]."""
    torch.manual_seed(seed)
    nT, M, _ = U.shape
    Xc = P.mean(0)
    Xs = float((P - Xc).abs().max())
    tc = float(T.mean())
    ts = float((T - tc).abs().max()) if nT > 1 else 1.0
    uc = U.reshape(-1, 2).mean(0)
    us = float(U.reshape(-1, 2).std())
    fld = Field(Xc, Xs, tc, ts, uc, us, omega, hidden, layers, tw, dtype).to(device)

    Pg = P.repeat(nT, 1).to(dtype)                                   # [nT*M, 2]
    Tg = T.repeat_interleave(M, 0).to(dtype)                         # [nT*M, 1]
    Yg = ((U.reshape(-1, 2) - uc) / us).to(dtype)
    C_in = fld.coords(Pg, Tg).detach()

    opt = torch.optim.Adam(fld.net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 1e-2)
    N = C_in.shape[0]
    for it in range(iters):
        if batch and batch < N:
            idx = torch.randint(0, N, (batch,), device=device)
            loss = (fld.net(C_in[idx]) - Yg[idx]).pow(2).mean()
        else:
            loss = (fld.net(C_in) - Yg).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if log is not None and (it % max(1, iters // 4) == 0 or it == iters - 1):
            log(f"      it {it:6d}  mse {float(loss):.3e}  rms {math.sqrt(float(loss))*us/PX:.4f} px")
    with torch.no_grad():
        res = (fld.net(C_in) - Yg)
        node_rel = float(res.norm() / (Yg.norm() + 1e-30))
        node_rms_world = float(res.pow(2).mean().sqrt() * us)
    return fld, {"node_rel_centred": node_rel, "node_rms_world": node_rms_world,
                 "u_scale": us, "X_scale": Xs, "t_scale": ts, "n_train": int(N)}


# --------------------------------------------------------------------------------------------- #
#  measurement 2 -- per-cell contrast
# --------------------------------------------------------------------------------------------- #
def cell_contrast(F, cid, mask, C, min_per_cell=10):
    """between-cell variance / within-cell variance of F - I, per channel and pooled."""
    I2 = torch.eye(2, device=F.device, dtype=F.dtype)[None]
    Y = (F - I2).reshape(-1, 4)[mask]
    cc = cid[mask]
    nb = torch.zeros(C + 1, device=F.device, dtype=F.dtype).index_add_(
        0, cc, torch.ones_like(Y[:, 0]))
    sm = torch.zeros(C + 1, 4, device=F.device, dtype=F.dtype).index_add_(0, cc, Y)
    keep = nb >= min_per_cell
    mu = sm / nb.clamp(min=1)[:, None]
    dev = Y - mu[cc]
    sq = torch.zeros(C + 1, 4, device=F.device, dtype=F.dtype).index_add_(0, cc, dev.pow(2))
    within = (sq[keep].sum(0) / (nb[keep].sum() - int(keep.sum())))          # pooled within var
    mk = mu[keep]
    between = ((mk - mk.mean(0)).pow(2).sum(0) / (int(keep.sum()) - 1))
    return {"between": [float(x) for x in between], "within": [float(x) for x in within],
            "ratio": [float(b / (w + 1e-300)) for b, w in zip(between, within)],
            "ratio_pooled": float(between.sum() / (within.sum() + 1e-300)),
            "n_cells": int(keep.sum())}


def pooled_slope(Fd, Ft, mask):
    I2 = torch.eye(2, device=Fd.device, dtype=Fd.dtype)[None]
    a = (Fd - I2).reshape(-1, 4)[mask].reshape(-1)
    b = (Ft - I2).reshape(-1, 4)[mask].reshape(-1)
    return float((a @ b) / (b @ b))


def report_row(name, F, Ft, mask, cid, C, extra=None):
    at = attenuation(F, Ft, mask)
    ct = cell_contrast(F, cid, mask, C)
    row = {"name": name, "ls_scale": at["ls_scale"], "corr": at["corr"],
           "slope_pooled": pooled_slope(F, Ft, mask),
           "rel_fro": rel(F[mask], Ft[mask]),
           "med_abs_disagree": at["med_abs_disagree"],
           "med_abs_F_minus_I": at["med_abs_F_minus_I"],
           "disagree_over_FmI": at["disagree_over_FmI"],
           "contrast_ratio": ct["ratio_pooled"], "contrast_per_channel": ct["ratio"],
           "between": ct["between"], "within": ct["within"]}
    if extra:
        row.update(extra)
    return row


def fmt(r):
    ls = " ".join(f"{v:6.3f}" for v in r["ls_scale"])
    co = " ".join(f"{v:6.3f}" for v in r["corr"])
    return (f"  {r['name']:<26s} {r['slope_pooled']:7.3f} [{ls}] [{co}] "
            f"{r['rel_fro']:8.4f} {r['med_abs_disagree']:9.5f} {r['disagree_over_FmI']:7.3f} "
            f"{r['contrast_ratio']:9.3f}")


HDR = (f"  {'row':<26s} {'slopeP':>7s} [{'ls_scale (4 ch)':^29s}] [{'corr (4 ch)':^29s}] "
       f"{'rel_fro':>8s} {'med|dF|':>9s} {'/|F-I|':>7s} {'btw/wth':>9s}")


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="siren_field")
    ap.add_argument("--stages", default="cf")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--eval-ticks", default="165,172,180")
    ap.add_argument("--t-lo", type=int, default=161)
    ap.add_argument("--t-hi", type=int, default=184)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--hpx", default="15,34")
    ap.add_argument("--omegas", default="1,2,3,5,10,20,30,60")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tw", type=float, default=1.0, help="extra scale on the normalised time axis")
    ap.add_argument("--batch", type=int, default=0, help="0 = full batch")
    ap.add_argument("--fit-dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--mlag", type=int, default=2, help="short reference lag for the relative-F row")
    ap.add_argument("--cache", default="")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    def dump():
        json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=float)
        open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")

    R = {"config": vars(args), "args": vars(a), "px_world": PX, "grid_px": GRID_PX}
    t_start = time.time()
    dev = torch.device(a.device)
    ev_ticks = [int(x) for x in a.eval_ticks.split(",")]
    hpxs = [float(x) for x in a.hpx.split(",")]
    omegas = [float(x) for x in a.omegas.split(",")]
    cache = a.cache or os.path.join(HERE, f"{a.tag}_cache_t{a.t_lo}_{a.t_hi}.pt")

    # ------------------------------------------------------------------ stage c ------------ #
    if "c" in a.stages or not os.path.exists(cache):
        with torch.no_grad():
            sy, X0, B = collect(args, a.t_lo, a.t_hi, log)
        torch.save({"X0": X0.cpu(), "cid": sy.cid.cpu(), "C": sy.C, "Np": sy.Np,
                    "dt": sy.dt, "n_sub": sy.n_sub_per_frame,
                    "B": {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in B.items()}},
                   cache)
        log(f"[cache] {cache}  ticks {a.t_lo}..{a.t_hi}  [{time.time()-t_start:.0f}s]")
        del sy, B, X0
        torch.cuda.empty_cache()

    D = torch.load(cache, map_location=dev)
    X0, cid, Cn, dt = D["X0"].to(dev), D["cid"].to(dev), D["C"], D["dt"]
    B = {int(k): {kk: vv.to(dev) for kk, vv in v.items()} for k, v in D["B"].items()}
    ticks = sorted(B.keys())
    Np = X0.shape[0]
    I2 = torch.eye(2, device=dev, dtype=X0.dtype)[None]
    log(f"[loaded] Np={Np} C={Cn} dt={dt} ticks {ticks[0]}..{ticks[-1]} ({len(ticks)} frames)")

    span = (X0.max(0).values - X0.min(0).values)
    area = float(span[0] * span[1])
    p_sp = math.sqrt(area / Np)
    cell_side = math.sqrt(area / Cn)
    R["geometry"] = {"span": [float(v) for v in span], "particle_spacing_world": p_sp,
                     "particle_spacing_px": p_sp / PX, "cell_side_world": cell_side,
                     "cell_side_px": cell_side / PX}
    log(f"[geometry] particle spacing {p_sp:.5f} ({p_sp/PX:.1f} px), cell side {cell_side:.4f} "
        f"({cell_side/PX:.1f} px)")

    band = 0.06 / MET.SHEET_SPAN
    xW = B[a.t0]["x0"]
    interior = ~((xW[:, 0] < band) | (xW[:, 0] > 1 - band)
                 | (xW[:, 1] < band) | (xW[:, 1] > 1 - band))
    R["n_interior"] = int(interior.sum())
    log(f"           interior (outside the {band:.4f} wall band) {int(interior.sum())}/{Np}")

    if "f" not in a.stages:
        dump()
        return

    # ------------------------------------------------------------------ stage f ------------ #
    fdt = torch.float32 if a.fit_dtype == "float32" else torch.float64
    Tall = torch.tensor([[t * dt] for t in ticks], device=dev, dtype=X0.dtype)
    R["tables"] = {}

    for hpx in hpxs:
        h = hpx * PX
        cg = ControlGrid(X0, h)
        P, U, vmask = observe(cg, X0, B, ticks)
        log(f"\n{'='*118}\n[grid] h = {hpx:g} px = {h:.5f} world | {cg.nx}x{cg.ny} nodes, "
            f"{cg.n_valid} valid ({100*cg.n_valid/cg.n_nodes:.0f}%), "
            f"{cg.pts_per_valid:.2f} particles per valid node, "
            f"{cell_side/h:.1f} control points per cell side")
        log(f"       training set = {cg.n_valid} valid nodes x {len(ticks)} frames = "
            f"{cg.n_valid*len(ticks)} samples; |u| rms over the window "
            f"{float(U.pow(2).mean().sqrt()):.5f} world "
            f"({float(U.pow(2).mean().sqrt())/PX:.1f} px)")
        key = f"h{hpx:g}px"
        R["tables"][key] = {"h_px": hpx, "h_world": h, "nodes": [cg.nx, cg.ny],
                            "n_valid": cg.n_valid, "pts_per_valid": cg.pts_per_valid,
                            "cd_honest_nodes": cg.n_cd_honest,
                            "pts_per_cell_side": cell_side / h, "ticks": {}}

        # ---- reference rows: TRUE field, and the CENTRAL DIFFERENCE on the same grid --------- #
        for k in ev_ticks:
            rows_abs, rows_rel = [], []
            Ft = B[k]["F0"]
            rows_abs.append(report_row("true F (simulator)", Ft, Ft, interior, cid, Cn))
            Fcd = derive_F(cg, X0, B[k]["x0"])
            rows_abs.append(report_row("central diff m=165", Fcd, Ft, interior, cid, Cn))
            # relative, short lag m: dF between t-m and t, composed with the TRUE F(t-m) (oracle)
            m = a.mlag
            if (k - m) in B:
                # the grid must live on the configuration the displacement is measured FROM
                cgm = ControlGrid(B[k - m]["x0"], h)
                Frel = derive_F(cgm, B[k - m]["x0"], B[k]["x0"], F_ref=B[k - m]["F0"])
                rows_rel.append(report_row(f"central diff m={m}", Frel, Ft, interior, cid, Cn))
            R["tables"][key]["ticks"][str(k)] = {"abs": rows_abs, "rel": rows_rel}

        # ---- v and C by centred difference (the controls 0.0227 / 0.0101) -------------------- #
        for k in ev_ticks:
            vcd = (B[k + 1]["x0"] - B[k - 1]["x0"]) / (2 * dt)
            Ccd_sim = ((B[k + 1]["F0"] - B[k - 1]["F0"]) / (2 * dt)) @ torch.linalg.inv(B[k]["F0"])
            Fm, Fp, Fk = (derive_F(cg, X0, B[t]["x0"]) for t in (k - 1, k + 1, k))
            Ccd_der = ((Fp - Fm) / (2 * dt)) @ torch.linalg.inv(Fk)
            R["tables"][key]["ticks"][str(k)]["cd_vC"] = {
                "v_centred_rel": rel(vcd[interior], B[k]["v0"][interior]),
                "C_centred_simF_rel": rel(Ccd_sim[interior], B[k]["C0"][interior]),
                "C_centred_derF_rel": rel(Ccd_der[interior], B[k]["C0"][interior]),
                "v_slope": float((vcd[interior].reshape(-1) @ B[k]["v0"][interior].reshape(-1))
                                 / (B[k]["v0"][interior].reshape(-1) @ B[k]["v0"][interior].reshape(-1))),
                "C_slope_simF": float((Ccd_sim[interior].reshape(-1) @ B[k]["C0"][interior].reshape(-1))
                                      / (B[k]["C0"][interior].reshape(-1) @ B[k]["C0"][interior].reshape(-1))),
            }

        # ---- the sweep ----------------------------------------------------------------------- #
        for om in omegas:
            t_om = time.time()
            log(f"\n[fit] h={hpx:g}px omega={om:g} hidden={a.hidden} layers={a.layers} "
                f"iters={a.iters} dtype={a.fit_dtype}")
            fld, fi = fit_field(P, Tall, U, om, a.hidden, a.layers, a.iters, a.lr, a.tw,
                                fdt, dev, log=log, batch=a.batch)
            fi["seconds"] = time.time() - t_om
            log(f"      node fit: rel(centred) {fi['node_rel_centred']:.3e}  rms "
                f"{fi['node_rms_world']:.3e} world ({fi['node_rms_world']/PX:.3f} px)")

            Xf = X0.to(fdt)
            for k in ev_ticks:
                tk = torch.full((Np, 1), float(k * dt), device=dev, dtype=fdt)
                u_h, F_h, v_h, C_h = fld.derivs(Xf, tk)
                u_h, F_h, v_h, C_h = (q.to(X0.dtype) for q in (u_h, F_h, v_h, C_h))
                Ft, vt, Ct = B[k]["F0"], B[k]["v0"], B[k]["C0"]
                u_t = B[k]["x0"] - X0
                slot = R["tables"][key]["ticks"][str(k)]
                row = report_row(f"SIREN omega={om:g}", F_h, Ft, interior, cid, Cn,
                                 extra={"omega": om, **fi})
                row["u_rel_particles"] = rel(u_h[interior], u_t[interior])
                row["u_rms_world"] = float((u_h - u_t)[interior].pow(2).mean().sqrt())
                row["v_rel"] = rel(v_h[interior], vt[interior])
                row["v_slope"] = float((v_h[interior].reshape(-1) @ vt[interior].reshape(-1))
                                       / (vt[interior].reshape(-1) @ vt[interior].reshape(-1)))
                row["C_rel"] = rel(C_h[interior], Ct[interior])
                row["C_slope"] = float((C_h[interior].reshape(-1) @ Ct[interior].reshape(-1))
                                       / (Ct[interior].reshape(-1) @ Ct[interior].reshape(-1)))
                slot["abs"].append(row)
                # relative short lag from the SAME field: F(t) F(t-m)^-1 composed with true F(t-m)
                m = a.mlag
                if (k - m) in B:
                    tkm = torch.full((Np, 1), float((k - m) * dt), device=dev, dtype=fdt)
                    _, F_hm, _, _ = fld.derivs(Xf, tkm)
                    F_hm = F_hm.to(X0.dtype)
                    Frel = (F_h @ torch.linalg.inv(F_hm)) @ B[k - m]["F0"]
                    slot["rel"].append(report_row(f"SIREN omega={om:g}", Frel, Ft, interior,
                                                  cid, Cn, extra={"omega": om}))
            del fld
            torch.cuda.empty_cache()
            dump()

        # ---- print the tables ----------------------------------------------------------------- #
        for k in ev_ticks:
            slot = R["tables"][key]["ticks"][str(k)]
            log(f"\n---- h = {hpx:g} px, tick {k} --------------------------------------------"
                f"------------------------------------")
            log(f"  ABSOLUTE F, displacement referenced to tick 0 (F_ref = I, FULLY REALIZABLE)")
            log(HDR)
            for r in slot["abs"]:
                log(fmt(r))
            log(f"  RELATIVE F, displacement referenced to tick {k-a.mlag} then composed with the "
                f"TRUE F there (ORACLE F_ref)")
            log(HDR)
            for r in slot["rel"]:
                log(fmt(r))
            cv = slot["cd_vC"]
            log(f"  v, C: centred difference  v_rel {cv['v_centred_rel']:.4f} "
                f"(slope {cv['v_slope']:.4f})  C_rel[simF] {cv['C_centred_simF_rel']:.4f} "
                f"(slope {cv['C_slope_simF']:.4f})  C_rel[derF] {cv['C_centred_derF_rel']:.4f}")
            log(f"        SIREN")
            log(f"  {'omega':>8s} {'u_rel(part)':>12s} {'u_rms_px':>9s} {'v_rel':>8s} "
                f"{'v_slope':>8s} {'C_rel':>9s} {'C_slope':>8s}")
            for r in slot["abs"]:
                if "omega" not in r:
                    continue
                log(f"  {r['omega']:8g} {r['u_rel_particles']:12.4e} "
                    f"{r['u_rms_world']/PX:9.4f} {r['v_rel']:8.4f} {r['v_slope']:8.4f} "
                    f"{r['C_rel']:9.4f} {r['C_slope']:8.4f}")

    R["wall_seconds"] = time.time() - t_start
    dump()
    log(f"\n[done] {R['wall_seconds']:.0f}s -> {a.tag}.json / {a.tag}.log")


if __name__ == "__main__":
    main()

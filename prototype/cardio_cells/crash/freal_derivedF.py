"""freal_derivedF.py -- EXPERIMENT 1.  Make the synthetic F wrong the way the REAL one is wrong.

WHAT IS BEING ATTACKED
====================================================================================================
`finject.py:319-320` takes the injected deformation gradient straight out of the simulator

    F0, C0 = sy.F0.clone(), sy.C0.clone()
    F1, C1 = Fs[-1].clone(), Cs[-1].clone()      # what a measurement at frame t+1 would give

and the celebrated med|dE/E| 0.257 -> 0.0078 is therefore an upper bound obtained with a PERFECT F.
It is called "realizable" because a real counterpart exists, not because any measurement process was
applied.  Round 5 attacked the gap by adding synthetic noise to that perfect F.  That is the wrong
SHAPE of error: on the recording, F is wrong for a structural reason -- its derivative channels and
a central difference of its own displacement field disagree by 0.0327 = 97% of |F - I|, all four
channels differ from the central difference by the SAME least-squares scale (0.59, 0.61, 0.61, 0.59)
at correlation 0.74-0.76, and 79.7% of the disagreement is a STATIC BIAS FIELD.  That is a
resolution / attenuation mismatch, not noise, and no amount of averaging touches it.

WHAT THIS SCRIPT DOES INSTEAD
----------------------------------------------------------------------------------------------------
It derives F on the synthetic system the way the recording derives it, and injects THAT:

  1. reference positions X = the particle positions at simulator tick 0 (the MPM reference config,
     the configuration p.F is measured against), displacement u(X) = x(t) - X  -- Lagrangian from
     frame 0, exactly the recording's convention (ch 0,1 are the grid X,Y and u = D[t]-D[0]);
  2. BIN u onto a regular control grid of spacing h by a square boxcar of width h -- the PIV
     interrogation window.  h = 15 px is the recording's spacing; 1 px = 4.88e-4 world;
  3. CENTRAL DIFFERENCE on that grid, (U[i+1]-U[i-1])/(2h), to get grad u -- so F = I + grad u;
  4. bilinearly interpolate F back to the particles' REFERENCE positions and inject it through
     finject's own `lerp` + `assemble_inj` + `step_inj`.

No noise is added anywhere.  The only difference from the 0.0078 control is that F is MEASURED off
positions with a real instrument's spatial support instead of being read out of the solver.

MEASURED, with a control for every number
----------------------------------------------------------------------------------------------------
  stage d   the derivation alone.  For each h: rel err of the derived F against the simulator's F,
            the per-channel least-squares attenuation scale and correlation (the recording's
            0.59-0.61 / 0.74-0.76 signature), the median disagreement as a fraction of |F - I| (the
            recording's 97%), the STATIC-BIAS fraction over the fit window (the recording's 79.7%),
            and whether matching the smoothing closes part of the gap (the recording: 0.0326 ->
            0.0249, 24%).  Two floors are separated here: the grid's contribution, and the
            irreducible gap between ANY kinematic gradient of x(X) and the MPM solver's own
            integrated p.F (measured with a per-particle weighted least-squares gradient).
  stage f   the fit.  Single frame at t0, ridge0, the finject ladder plus one rung per h; plus the
            HELD-OUT one-frame residual at a later tick, with the derived F there too, because the
            acceptance statistic must not be an oracle either.
  stage r   free 150-frame rollout at margin 20, deterministic 2-D gauge (round5_score.gauge_grid,
            not gauge_fix2), gauged loopscore + the band it is uncertain by.

CONTROLS REPRODUCED (finject.log, same t0 = 165, same seed)
  none 0.2572 | F_hold 0.1401 | F_lerp 0.0078 | F_true 0.0091 | rollout gauged 0.8277 / 0.9997

usage:
  PYTHONPATH=/workspace/Plexus/src python freal_derivedF.py --device cuda:1 --stages dfr
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

from assemble import System, SUBSTEP_TOKENS                        # noqa: E402
from recover import Solver, install_E, score, theta_scale          # noqa: E402
import metrics as MET                                              # noqa: E402
import crash_test as CT                                            # noqa: E402
from crash_round2 import percell_amplitude, r2_percell             # noqa: E402
from crash_round3 import scale2, t2_of                             # noqa: E402
from finject import lerp, hold, assemble_inj, y_of, record_substeps  # noqa: E402
from refute_round3 import advance                                  # noqa: E402
from round5_fit import SNAP                                        # noqa: E402
from round5_solve import pstats                                    # noqa: E402
from round5_score import gauge_grid                                # noqa: E402

PX = 4.88e-4                      # 1 recording pixel in world units (finject.py:400)
GRID_PX = 15.0                    # the recording's control-point spacing, in pixels


def rel(a, b):
    return float((a - b).norm() / b.norm())


# --------------------------------------------------------------------------------------------- #
#  crash_test.plant_and_warm, plus the ONE thing it does not return: the tick-0 particle positions.
#  Those are the MPM reference configuration -- the configuration p.F is the gradient against -- and
#  the recording's frame-0 grid is its counterpart.  Nothing else differs; the warm-up loop, the
#  seeds and the snapshot are copied verbatim so the planted system is bit-identical.
# --------------------------------------------------------------------------------------------- #
def plant_and_warm_x0(args, log, seed=2026, keep_ticks=()):
    sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                n_grid=args.n_grid, warmup=0, dtype=args.dtype, mode=args.mode)
    C = sy.C
    g = torch.Generator().manual_seed(seed)
    E = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    gn = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=g)).to(sy.device, sy.dtype)
    sy.E_true[1:], sy.gain_true[1:] = E, gn
    sy.theta_true = torch.cat([sy.E_true[1:], sy.gain_true[1:]])
    install_E(sy, sy.E_true)

    keep = set(int(t) for t in keep_ticks)
    REF = {0: {"x": sy.p.get("pos").clone(), "F": sy.p.F.clone()}}   # tick 0 = MPM reference config

    W = args.warmup
    for tick in range(W):
        sy._outer(tick, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
        if (tick + 1) in keep:
            REF[tick + 1] = {"x": sy.p.get("pos").clone(), "F": sy.p.F.clone()}
    sy.warmup_frames = W
    sy._snapshot(W)
    log(f"[planted] C={sy.C} Np={sy.Np} grid {sy.g.nx}^2 dtype={sy.dtype} dx={sy.g.dx:.6g}  "
        f"warm-up {W} frames")
    log(f"          reference config at tick 0: |F(0) - I| max "
        f"{float((REF[0]['F'] - torch.eye(2, device=sy.device, dtype=sy.dtype)).abs().max()):.2e}  "
        f"(must be 0 for X to be the configuration p.F is measured against)")
    return sy, REF


def collect(args, t_lo, t_hi, log, keep_ticks=()):
    """Frame-boundary series for ticks t_lo..t_hi: measured x0, oracle F0, end-of-frame x_next."""
    a2 = SimpleNamespace(**{**vars(args), "warmup": t_lo})
    sy, REF = plant_and_warm_x0(a2, log, keep_ticks=keep_ticks)
    n = sy.n_sub_per_frame
    B, cur = {}, t_lo
    for k in range(t_hi - t_lo + 1):
        if k > 0:
            sy.restore()
            advance(sy, cur, cur + 1)
            sy._snapshot(cur + 1)
            cur += 1
        Fs, _, Xs = record_substeps(sy, n)
        B[cur] = {"tick": cur, "x0": sy.x0.clone(), "F0": sy.F0.clone(),
                  "F1": Fs[-1].clone(), "x_next": Xs[-1].clone(),
                  "snap": {kk: getattr(sy, kk).clone() for kk in SNAP}}
    return sy, REF, B


# --------------------------------------------------------------------------------------------- #
#  THE MEASUREMENT.  Bin -> central difference -> interpolate.  Nothing here knows about theta.
# --------------------------------------------------------------------------------------------- #
class ControlGrid:
    """A regular control grid of spacing h over the reference configuration.

    `bin_` is a square boxcar of width h (each particle falls in exactly one cell, the one whose
    node is nearest) -- the PIV interrogation window.  Nodes with no particle are INVALID; they are
    filled from their valid neighbours by iterated dilation, which is what a masked PIV field forces
    on anyone who wants a central difference at the tissue rim.
    """

    def __init__(self, X, h, pad=3):
        self.h = float(h)
        lo = X.min(0).values - pad * h
        hi = X.max(0).values + pad * h
        self.lo = lo
        self.nx = int(torch.ceil((hi[0] - lo[0]) / h).item()) + 1
        self.ny = int(torch.ceil((hi[1] - lo[1]) / h).item()) + 1
        gx = ((X[:, 0] - lo[0]) / h).round().long().clamp(0, self.nx - 1)
        gy = ((X[:, 1] - lo[1]) / h).round().long().clamp(0, self.ny - 1)
        self.flat = gx * self.ny + gy
        self.cnt = torch.zeros(self.nx * self.ny, device=X.device, dtype=X.dtype)
        self.cnt.index_add_(0, self.flat, torch.ones_like(X[:, 0]))
        self.valid = self.cnt > 0
        self.n_nodes = self.nx * self.ny
        self.n_valid = int(self.valid.sum())
        # nodes whose 4-neighbourhood in BOTH directions is valid: where a central difference is
        # honest rather than dilated
        v = self.valid.view(self.nx, self.ny)
        self.n_cd_honest = int((v[2:, 1:-1] & v[:-2, 1:-1] & v[1:-1, 2:] & v[1:-1, :-2]
                                & v[1:-1, 1:-1]).sum())
        self.pts_per_valid = float(self.cnt[self.valid].mean())

    def bin_(self, q):
        """q [Np, K] -> [nx, ny, K] boxcar mean, invalid nodes filled by dilation."""
        K = q.shape[1]
        s = torch.zeros(self.n_nodes, K, device=q.device, dtype=q.dtype)
        s.index_add_(0, self.flat, q)
        V = torch.where(self.valid[:, None], s / self.cnt.clamp(min=1)[:, None],
                        torch.zeros_like(s)).view(self.nx, self.ny, K)
        m = self.valid.view(self.nx, self.ny).clone()
        for _ in range(2 * max(self.nx, self.ny)):
            if bool(m.all()):
                break
            acc = torch.zeros_like(V)
            wgt = torch.zeros(self.nx, self.ny, device=q.device, dtype=q.dtype)
            for dx_, dy_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                Vs = torch.roll(V, (dx_, dy_), (0, 1))
                ms = torch.roll(m, (dx_, dy_), (0, 1)).to(V.dtype)
                acc = acc + Vs * ms[:, :, None]
                wgt = wgt + ms
            fill = (~m) & (wgt > 0)
            V = torch.where(fill[:, :, None], acc / wgt.clamp(min=1)[:, :, None], V)
            m = m | fill
        return V

    def central_diff(self, V):
        """d/dx and d/dy of a [nx, ny, K] field, central inside, one-sided on the outermost ring."""
        h = self.h
        dX = torch.zeros_like(V)
        dY = torch.zeros_like(V)
        dX[1:-1] = (V[2:] - V[:-2]) / (2 * h)
        dX[0] = (V[1] - V[0]) / h
        dX[-1] = (V[-1] - V[-2]) / h
        dY[:, 1:-1] = (V[:, 2:] - V[:, :-2]) / (2 * h)
        dY[:, 0] = (V[:, 1] - V[:, 0]) / h
        dY[:, -1] = (V[:, -1] - V[:, -2]) / h
        return dX, dY

    def sample(self, V, X, mode="bilinear"):
        gx = (X[:, 0] - self.lo[0]) / self.h
        gy = (X[:, 1] - self.lo[1]) / self.h
        if mode == "nearest":
            i = gx.round().long().clamp(0, self.nx - 1)
            j = gy.round().long().clamp(0, self.ny - 1)
            return V[i, j]
        i0 = gx.floor().long().clamp(0, self.nx - 2)
        j0 = gy.floor().long().clamp(0, self.ny - 2)
        fx = (gx - i0).clamp(0, 1)[:, None]
        fy = (gy - j0).clamp(0, 1)[:, None]
        return ((1 - fx) * (1 - fy) * V[i0, j0] + fx * (1 - fy) * V[i0 + 1, j0]
                + (1 - fx) * fy * V[i0, j0 + 1] + fx * fy * V[i0 + 1, j0 + 1])

    def smooth3(self, V, axis):
        """[1/4, 1/2, 1/4] along `axis`: the trapezoid sampling of a width-2h boxcar on this grid.

        The central difference of a boxcar-binned field is (B_h * B_2h) * u', so the operator that
        the derived F applies to the TRUE derivative is: bin at h, then average over +-h along the
        differencing direction.  This is that second factor, and applying it to the true F is the
        "matched smoothing" control.
        """
        Vp = torch.roll(V, 1, axis)
        Vm = torch.roll(V, -1, axis)
        idx = [slice(None)] * V.dim()
        out = 0.25 * Vp + 0.5 * V + 0.25 * Vm
        idx[axis] = 0
        out[tuple(idx)] = V[tuple(idx)]
        idx[axis] = V.shape[axis] - 1
        out[tuple(idx)] = V[tuple(idx)]
        return out


def derive_F(cg, X, x, mode="bilinear", F_ref=None):
    """F = I + grad u, u = x - X, by boxcar binning + central difference on the control grid.

    `X` is the REFERENCE configuration the displacement is measured from -- tick 0 of the recording.
    If the reference is not the material's own reference (i.e. F(X-tick) != I) then what a
    displacement field can give is the RELATIVE gradient, and the absolute F is the composition
    dF @ F_ref.  Passing F_ref makes that composition explicit; F_ref = I is the realizable case.
    """
    U = cg.bin_(x - X)                          # [nx, ny, 2]
    dX, dY = cg.central_diff(U)                 # du/dx, du/dy  (columns are u and v)
    G = torch.stack([dX, dY], dim=-1)           # [nx, ny, 2(comp), 2(deriv)]
    Gp = cg.sample(G.reshape(cg.nx, cg.ny, 4), X, mode).reshape(-1, 2, 2)
    dF = Gp + torch.eye(2, device=X.device, dtype=X.dtype)[None]
    return dF if F_ref is None else dF @ F_ref


def matched_smooth_F(cg, X, F, mode="bilinear"):
    """Apply the derived-F operator's OWN smoothing to a true F: bin at h, then [1/4,1/2,1/4] along
    the differencing axis of each column, then read back.  Column j of F is differentiated in
    direction j, so column j gets smoothed along axis j on top of the isotropic bin."""
    Vb = cg.bin_(F.reshape(-1, 4))              # [nx, ny, 4] = (F00, F01, F10, F11)
    Vb = Vb.reshape(cg.nx, cg.ny, 2, 2)
    out = torch.stack([cg.smooth3(Vb[:, :, :, 0], 0), cg.smooth3(Vb[:, :, :, 1], 1)], dim=-1)
    return cg.sample(out.reshape(cg.nx, cg.ny, 4), X, mode).reshape(-1, 2, 2)


def box_smooth_F(cg2, X, F, mode="bilinear"):
    """The blunt version of the same control: an isotropic square boxcar of width 2h (cg2 built at
    spacing 2h), which is how the recording's 0.0326 -> 0.0249 test was framed."""
    Vb = cg2.bin_(F.reshape(-1, 4))
    return cg2.sample(Vb, X, mode).reshape(-1, 2, 2)


def particle_lsq_F(X, x, radius, chunk=1000, F_ref=None):
    """The finest kinematic gradient available: per-particle weighted least squares of the
    displacement on its neighbours inside `radius`.  This is NOT the grid estimator -- it is the
    floor that separates "the grid is coarse" from "no kinematic gradient equals the solver's F".
    """
    N = X.shape[0]
    u = x - X
    out = torch.zeros(N, 2, 2, device=X.device, dtype=X.dtype)
    nnb = torch.zeros(N, device=X.device, dtype=X.dtype)
    I2 = torch.eye(2, device=X.device, dtype=X.dtype)
    for a0 in range(0, N, chunk):
        a1 = min(a0 + chunk, N)
        d = X[None, a0:a1, :] - X[:, None, :]                      # [N, m, 2]
        r2 = d.pow(2).sum(-1)
        w = torch.exp(-r2 / (2 * (radius / 2.0) ** 2)) * (r2 <= radius ** 2)
        du = u[:, None, :] - u[None, a0:a1, :]                     # [N, m, 2]
        # solve  min_G sum_k w_k | du_k - G d_k |^2   ->  G = (sum w du d^T)(sum w d d^T)^-1
        M = torch.einsum("nm,nma,nmb->mab", w, d, d)
        Y = torch.einsum("nm,nma,nmb->mab", w, -du, d)
        M = M + 1e-14 * I2[None]
        out[a0:a1] = Y @ torch.linalg.inv(M)
        nnb[a0:a1] = (w > 0).sum(0).to(X.dtype)
    dF = out + I2[None]
    return (dF if F_ref is None else dF @ F_ref), float(nnb.mean())


def attenuation(Fd, Ft, mask=None):
    """The recording's signature, channel by channel: least-squares scale of the derived deviation
    on the true deviation, and their correlation.

    The recording quotes MEDIANS (0.0327 disagreement, 97% of |F - I|) because the field has heavy
    tails, and so does this one: over all 10 000 particles the Frobenius norm is set by a handful
    of wall particles.  Both are reported, and every one is restricted to `mask` when given.
    """
    if mask is not None:
        Fd, Ft = Fd[mask], Ft[mask]
    I2 = torch.eye(2, device=Fd.device, dtype=Fd.dtype)[None]
    a = (Fd - I2).reshape(-1, 4)
    b = (Ft - I2).reshape(-1, 4)
    sc, co, sct = [], [], []
    for c in range(4):
        x_, y_ = b[:, c], a[:, c]
        sc.append(float((x_ @ y_) / (x_ @ x_)) if float(x_ @ x_) > 0 else float("nan"))
        xm, ym = x_ - x_.mean(), y_ - y_.mean()
        co.append(float((xm @ ym) / (xm.norm() * ym.norm() + 1e-300)))
        # trimmed: drop the 1% of particles with the largest |derived| deviation, so one wall
        # particle cannot set the slope
        thr = torch.quantile(y_.abs(), 0.99)
        k = y_.abs() <= thr
        xk, yk = x_[k], y_[k]
        sct.append(float((xk @ yk) / (xk @ xk)) if float(xk @ xk) > 0 else float("nan"))
    dev = (Fd - Ft).reshape(-1, 4).abs()
    ref = (Ft - I2).reshape(-1, 4).abs()
    return {"ls_scale": sc, "ls_scale_trim99": sct, "corr": co,
            "med_abs_disagree": float(dev.median()),
            "p90_abs_disagree": float(torch.quantile(dev.reshape(-1), 0.90)),
            "med_abs_F_minus_I": float(ref.median()),
            "med_abs_Fd_minus_I": float((Fd - I2).reshape(-1, 4).abs().median()),
            "disagree_over_FmI": float(dev.median() / (ref.median() + 1e-300)),
            "rel_fro": rel(Fd, Ft)}


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="freal_derivedF")
    ap.add_argument("--stages", default="dfr")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8, help="frames used for the static-bias statistic")
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--hpx", default="15,22.5,34,45,68,102",
                    help="control-grid spacings, in recording pixels (1 px = 4.88e-4 world)")
    ap.add_argument("--fit-hpx", default="15,34,68")
    ap.add_argument("--roll-hpx", default="34")
    ap.add_argument("--mrefs", default="2,10,40,165",
                    help="lags, in frames, of the displacement's reference configuration; the "
                         "largest one must be t0 so that F_ref = I and the measurement is "
                         "fully realizable")
    ap.add_argument("--gn", type=int, default=5)
    ap.add_argument("--refine", type=int, default=2)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "args": vars(a), "px_world": PX, "grid_px": GRID_PX}
    t_start = time.time()
    torch.manual_seed(0)
    hs_px = [float(v) for v in a.hpx.split(",")]
    fit_px = [float(v) for v in a.fit_hpx.split(",")]
    roll_px = [float(v) for v in a.roll_hpx.split(",")]

    m_list = [int(v) for v in a.mrefs.split(",")]

    with torch.no_grad():
        t_lo, t_hi = a.t0, max(a.t0 + a.T - 1, a.holdout_tick)
        keep = set(a.t0 - m for m in m_list if 0 < a.t0 - m)
        sy, REF, B = collect(args, t_lo, t_hi, log, keep_ticks=keep)
        C, n, dt, dx = sy.C, sy.n_sub_per_frame, sy.dt, sy.g.dx
        th = sy.theta_true.double()
        dev, f64 = th.device, torch.float64
        I2 = torch.eye(2, device=dev, dtype=f64)
        X0 = REF[0]["x"]
        k0, hk = a.t0, a.holdout_tick
        log(f"[collect] boundaries {t_lo}..{t_hi} ({len(B)}); reference configs at ticks "
            f"{sorted(REF)} [{time.time()-t_start:.0f}s]")

        # ---- geometry: what "6 control points per cell" means here ------------------------------
        span = (X0.max(0).values - X0.min(0).values)
        area = float(span[0] * span[1])
        p_sp = math.sqrt(area / sy.Np)
        cell_side = math.sqrt(area / C)
        R["geometry"] = {"sheet_span_world": [float(v) for v in span], "area": area,
                         "particle_spacing_world": p_sp, "particle_spacing_px": p_sp / PX,
                         "cell_side_world": cell_side, "cell_side_px": cell_side / PX,
                         "dx_world": dx, "px_world": PX, "h_15px_world": GRID_PX * PX,
                         "points_per_cell_at_15px": cell_side / (GRID_PX * PX),
                         "recording_points_per_cell": 6.0,
                         "h_matching_6_per_cell_px": cell_side / 6.0 / PX}
        gg = R["geometry"]
        log(f"[geometry] sheet {span[0]:.4f}x{span[1]:.4f}, particle spacing {p_sp:.5f} "
            f"({p_sp/PX:.1f} px), cell side {cell_side:.4f} ({cell_side/PX:.1f} px), dx {dx:.5f}")
        log(f"           h = 15 px = {GRID_PX*PX:.5f} world -> {gg['points_per_cell_at_15px']:.1f} "
            f"control points per cell HERE, and it is FINER than the particle spacing, so it "
            f"resolves nothing the sheet contains.  The recording gets 6 points per cell; the "
            f"matching spacing here is h = {gg['h_matching_6_per_cell_px']:.0f} px.")

        band = 0.06 / MET.SHEET_SPAN
        xW = B[k0]["x0"]
        interior = ~((xW[:, 0] < band) | (xW[:, 0] > 1 - band)
                     | (xW[:, 1] < band) | (xW[:, 1] > 1 - band))
        R["n_interior"] = int(interior.sum())
        log(f"           interior (outside the {band:.4f} wall band) {int(interior.sum())}/{sy.Np}")

        # grids, one per (reference lag m, spacing h)
        def refidx(m):
            return max(a.t0 - m, 0)

        allh = sorted(set(hs_px + fit_px + roll_px))
        CG = {(m, hp): ControlGrid(REF[refidx(m)]["x"], hp * PX) for m in m_list for hp in allh}
        CG2 = {(m, hp): ControlGrid(REF[refidx(m)]["x"], 2 * hp * PX) for m in m_list for hp in allh}

        def derF(m, hp, x, mode="bilinear"):
            r = refidx(m)
            return derive_F(CG[(m, hp)], REF[r]["x"], x, mode, F_ref=REF[r]["F"])

        Ft0 = B[k0]["F0"]

        # ------------------------------------------------------------------ stage d ------------ #
        if "d" in a.stages:
            log(f"\n[d] THE DERIVATION ALONE, tick {k0}, zero noise, INTERIOR particles.  Reference "
                f"= the simulator's own p.F.")
            log(f"    m = how many frames back the displacement's reference configuration sits.  "
                f"m = {a.t0} is tick 0 = the MPM reference, F_ref = I, FULLY REALIZABLE; smaller m "
                f"composes with the oracle F at the reference tick and so isolates the grid.")

            # -- d1: the kinematic floor.  grad_X x at particle resolution, no grid at all --------
            log(f"\n    [d1] kinematic floor: grad_X x by weighted least squares, no control grid")
            log(f"    {'m':>5s} {'radius':>8s} {'nbrs':>5s} {'med|dis|':>9s} {'/|F-I|':>7s} "
                f"{'relFro':>7s} {'scale(4ch)':>30s} {'corr(4ch)':>30s}")
            fl = {}
            for m in m_list:
                for rm in (2.5,):
                    r = refidx(m)
                    Fk, nb = particle_lsq_F(REF[r]["x"], B[k0]["x0"], rm * p_sp,
                                            F_ref=REF[r]["F"])
                    at = attenuation(Fk, Ft0, interior)
                    fl[f"m{m}_r{rm:g}ps"] = {"m": m, "radius_ps": rm, "mean_neighbours": nb, **at}
                    log(f"    {m:>5d} {rm:>8.1f} {nb:>5.0f} {at['med_abs_disagree']:>9.2e} "
                        f"{at['disagree_over_FmI']:>7.3f} {at['rel_fro']:>7.4f} "
                        f"{str([round(v,3) for v in at['ls_scale_trim99']]):>30s} "
                        f"{str([round(v,3) for v in at['corr']]):>30s}")
            R["kinematic_floor"] = fl

            # -- d2: the grid, at every (m, h) ----------------------------------------------------
            log(f"\n    [d2] the full measurement: bin at h, central difference, bilinear back")
            log(f"    {'m':>5s} {'h_px':>6s} {'h/psp':>6s} {'pts/cell':>8s} {'pts/node':>8s} "
                f"{'cd_ok%':>7s} {'med|dis|':>9s} {'/|F-I|':>7s} {'relFro':>7s} "
                f"{'scale(4ch)':>30s} {'corr(4ch)':>30s} {'matched':>9s} {'box2h':>9s} "
                f"{'nearest':>8s}")
            rows = []
            for m in m_list:
                for hp in hs_px:
                    cg, cg2 = CG[(m, hp)], CG2[(m, hp)]
                    r = refidx(m)
                    Fd = derF(m, hp, B[k0]["x0"])
                    at = attenuation(Fd, Ft0, interior)
                    # matched-smoothing controls: apply the estimator's own support to the TRUE F,
                    # in the same reference frame (so F_true is pulled back by F_ref first)
                    Ft_rel = Ft0 @ torch.linalg.inv(REF[r]["F"])
                    Fm = matched_smooth_F(cg, REF[r]["x"], Ft_rel) @ REF[r]["F"]
                    Fb = box_smooth_F(cg2, REF[r]["x"], Ft_rel) @ REF[r]["F"]
                    Fn = derF(m, hp, B[k0]["x0"], "nearest")
                    row = {"m": m, "h_px": hp, "h_world": hp * PX,
                           "h_over_particle_spacing": hp * PX / p_sp,
                           "points_per_cell": cell_side / (hp * PX),
                           "pts_per_valid_node": cg.pts_per_valid,
                           "n_valid_nodes": cg.n_valid, "n_nodes": cg.n_nodes,
                           "frac_central_diff_honest": cg.n_cd_honest / cg.n_nodes, **at,
                           "med_abs_vs_matched": float((Fd - Fm)[interior].abs().median()),
                           "med_abs_vs_box2h": float((Fd - Fb)[interior].abs().median()),
                           "gap_closed_by_matched": 1.0 - float((Fd - Fm)[interior].abs().median())
                           / max(at["med_abs_disagree"], 1e-300),
                           "med_abs_nearest": attenuation(Fn, Ft0, interior)["med_abs_disagree"]}
                    rows.append(row)
                    log(f"    {m:>5d} {hp:>6.1f} {row['h_over_particle_spacing']:>6.2f} "
                        f"{row['points_per_cell']:>8.1f} {cg.pts_per_valid:>8.2f} "
                        f"{100*row['frac_central_diff_honest']:>6.1f}% "
                        f"{at['med_abs_disagree']:>9.2e} {at['disagree_over_FmI']:>7.3f} "
                        f"{at['rel_fro']:>7.4f} "
                        f"{str([round(v,3) for v in at['ls_scale_trim99']]):>30s} "
                        f"{str([round(v,3) for v in at['corr']]):>30s} "
                        f"{row['med_abs_vs_matched']:>9.2e} {row['med_abs_vs_box2h']:>9.2e} "
                        f"{row['med_abs_nearest']:>8.2e}")
            R["derivation"] = rows

            # -- d3: static bias.  Can more frames average it away? -------------------------------
            log(f"\n    [d3] static-bias fraction over {a.T} frames (the recording: 79.7%)")
            log(f"    {'m':>5s} {'h_px':>6s} {'static%':>8s} {'fluct_lag1':>11s} "
                f"{'med|dis|':>9s} {'med|static|':>11s}")
            sb = []
            ticks = list(range(a.t0, a.t0 + a.T))
            for m in m_list:
                for hp in hs_px:
                    E = torch.stack([derF(m, hp, B[k]["x0"])[interior] - B[k]["F0"][interior]
                                     for k in ticks])
                    mean_e = E.mean(0)
                    ms = float(E.pow(2).mean())
                    sf = float(mean_e.pow(2).mean() / (ms + 1e-300))
                    Fl = E - mean_e[None]
                    v0 = float((Fl[:-1] * Fl[1:]).mean())
                    vv = float(Fl.pow(2).mean())
                    sb.append({"m": m, "h_px": hp, "static_fraction": sf,
                               "fluct_lag1_autocorr": v0 / (vv + 1e-300),
                               "med_abs_err": float(E.abs().median()),
                               "med_abs_static": float(mean_e.abs().median())})
                    log(f"    {m:>5d} {hp:>6.1f} {100*sf:>7.1f}% {v0/(vv+1e-300):>11.3f} "
                        f"{float(E.abs().median()):>9.2e} {float(mean_e.abs().median()):>11.2e}")
            R["static_bias"] = sb

        # ------------------------------------------------------------------ stage f ------------ #
        thetas, R["fits"] = {}, {}
        if "f" in a.stages or "r" in a.stages:
            y_obs_h = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
            injh_true = lerp(B[hk]["F0"], B[hk]["F1"], n)

            def holdout(theta, injh):
                for kk, vv in B[hk]["snap"].items():
                    setattr(sy, kk, vv.clone())
                y = y_of(sy, theta, n, injh, None)
                return float((y - y_obs_h).norm() / y_obs_h.norm())

            def pair(fn):
                """(injected F over the fit frame, injected F over the held-out frame)."""
                return (lerp(fn(B[k0]["x0"]), fn(B[k0]["x_next"]), n),
                        lerp(fn(B[hk]["x0"]), fn(B[hk]["x_next"]), n))

            INJ = [("none", None, None),
                   ("F_hold_simF", hold(B[k0]["F0"], n), hold(B[hk]["F0"], n)),
                   ("F_lerp_simF", lerp(B[k0]["F0"], B[k0]["F1"], n), injh_true)]
            m_real = max(m_list)
            m_grid = min(m_list)
            h_match = min(fit_px, key=lambda v: abs(v - gg["h_matching_6_per_cell_px"]))
            seen = set()
            for m, hp in ([(m_real, h) for h in fit_px] + [(m_grid, h) for h in fit_px]
                          + [(m, h_match) for m in m_list]):
                nm = f"F_der_m{m}_{hp:g}px"
                if nm in seen:
                    continue
                seen.add(nm)
                i0, i1 = pair(lambda x, m=m, hp=hp: derF(m, hp, x))
                INJ.append((nm, i0, i1))
            for m in (m_real, m_grid):
                r = refidx(m)
                i0, i1 = pair(lambda x, r=r: particle_lsq_F(REF[r]["x"], x, 2.5 * p_sp,
                                                            F_ref=REF[r]["F"])[0])
                INJ.append((f"F_kin_lsq_m{m}", i0, i1))

            log(f"\n[f] FIT, single frame at tick {k0}, ridge0, displacement read-out; held-out "
                f"one-frame residual at tick {hk} with the SAME measurement there.")
            log(f"    {'variant':<20s} {'medE':>8s} {'p90E':>8s} {'relL2':>8s} {'negE':>5s} "
                f"{'corrE':>7s} {'meanrat':>8s} {'fit_res':>9s} {'hold_ownF':>10s} "
                f"{'hold_simF':>10s} {'ctrl':>8s}")
            tgt = {"none": 0.2572, "F_hold_simF": 0.1401, "F_lerp_simF": 0.0078}
            for name, iF, iFh in INJ:
                tc = time.time()
                for kk, vv in B[k0]["snap"].items():
                    setattr(sy, kk, vv.clone())
                A, y0, _ = assemble_inj(sy, n, iF, None)
                b = (B[k0]["x_next"] - B[k0]["x0"]).reshape(-1) - y0
                S = Solver(A, C)
                t_hat = S(b)["ridge0"]
                fit_res = float((A @ t_hat - b).norm() / b.norm())
                cond = S.cond
                S.free()
                del A, S
                torch.cuda.empty_cache()
                ps = pstats(t_hat.cpu().numpy(), th.cpu().numpy(), C)
                ho_own = holdout(t_hat, iFh if iFh is not None else injh_true)
                ho_sim = holdout(t_hat, injh_true)
                thetas[name] = t_hat.clone()
                R["fits"][name] = {"pstats": ps, "fit_residual": fit_res, "cond": cond,
                                   "holdout_ownF": ho_own, "holdout_simF": ho_sim,
                                   "seconds": time.time() - tc}
                dd = f"{ps['med_E']-tgt[name]:+.4f}" if name in tgt else ""
                log(f"    {name:<20s} {ps['med_E']:>8.4f} {ps['p90_E']:>8.4f} "
                    f"{ps['rel_l2']:>8.4f} {ps['n_negE']:>5d} {ps['corr_E']:>7.4f} "
                    f"{ps['mean_ratio_E']:>8.4f} {fit_res:>9.3e} {ho_own:>10.5f} "
                    f"{ho_sim:>10.5f} {dd:>8s}")

            R["holdout_floor"] = {"theta_true_simF": holdout(th, injh_true)}
            for name, iF, iFh in INJ:
                if iFh is not None and name.startswith(("F_der", "F_kin")):
                    R["holdout_floor"][f"theta_true_{name}"] = holdout(th, iFh)
            log(f"    held-out floor at theta_true (the measurement's OWN irreducible residual):")
            for kk, vv in R["holdout_floor"].items():
                log(f"        {kk:<32s} {vv:.5f}")
            np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                     theta_true=th.cpu().numpy(),
                     **{k: v.cpu().numpy() for k, v in thetas.items()})

        # ------------------------------------------------------------------ stage r ------------ #
        if "r" in a.stages:
            W, G = a.t0, a.window
            for kk, vv in B[a.t0]["snap"].items():
                setattr(sy, kk, vv.clone())
            snap0 = {kk: getattr(sy, kk).clone() for kk in SNAP}
            x0, cid = sy.x0.clone(), sy.cid
            tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                       for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
            anchor = ~interior
            ref_full = torch.zeros(G, sy.Np, 2, device=dev, dtype=sy.dtype)
            sy.restore()
            install_E(sy, sy.E_true)
            for k in range(G):
                sy._outer(W + k, gain_cell=sy.gain_true)
                sy.H.sub_dt = sy.dt_sub
                for _ in range(n):
                    for tok in SUBSTEP_TOKENS:
                        sy._tok(tok)
                sy.H.sub_dt = None
                ref_full[k] = sy.p.get("pos")
            d_ref = ref_full - x0[None]
            dm = d_ref[:, interior].mean(0, keepdim=True)
            ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
            real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
            a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
            keepc = np.isfinite(a_ref) & (a_ref > 0)
            log(f"\n[r] reference window built, {G} frames [{time.time()-t_start:.0f}s]")

            def scored(theta, full_out=True):
                for kk, vv in snap0.items():
                    setattr(sy, kk, vv.clone())
                tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                              anchor=None, interior=interior, ss_tot=ss_tot,
                                              keep_full=full_out, band_mask=anchor)
                m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                out = {"loop": m20["loopscore"], "t1": coarse["motion_energy_ratio_interior"],
                       "t2": t2_of(m20), "R2": coarse["R2_displacement_interior"],
                       "rms_dx_mean": coarse["rms_pos_err_dx_mean"]}
                if full_out:
                    ah, _ = percell_amplitude(full, x0, cid, C, interior)
                    out.update({"margin20": m20, "coarse": coarse,
                                "percell": r2_percell(ah, a_ref, keepc)})
                    del full
                return out

            names = ["theta_true", "none", "F_lerp_simF"]
            names += [f"F_der_m{max(m_list)}_{h:g}px" for h in roll_px]
            names += [f"F_der_m{min(m_list)}_{h:g}px" for h in roll_px]
            names += [f"F_kin_lsq_m{max(m_list)}"]
            names = [nm for nm in names if nm == "theta_true" or nm in thetas]
            R["rollouts"] = {}
            log(f"    {'candidate':<22s} {'medE':>7s} {'raw':>8s} {'kE':>6s} {'kg':>6s} "
                f"{'gauged':>8s} {'band_lo':>8s} {'band_hi':>8s} {'R2':>8s} {'r2cell':>7s} "
                f"{'conv':>5s}")
            for name in names:
                tc = time.time()
                theta = th if name == "theta_true" else thetas[name]
                raw = scored(theta)

                def probe(lE, lg, theta=theta):
                    return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

                gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"],
                                gn=a.gn, refine=a.refine)
                kE, kg = gf["k_E"], gf["k_g"]
                gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                    else scored(scale2(theta, kE, kg, C))
                bl, bh = gf["loop_spread_within_10pct"] or (float("nan"), float("nan"))
                R["rollouts"][name] = {
                    "param": pstats(theta.cpu().numpy(), th.cpu().numpy(), C),
                    "holdout": R["fits"].get(name, {}).get("holdout_ownF"),
                    "raw": raw, "gauged": gau,
                    "gauge": {k: v for k, v in gf.items() if k != "cells"},
                    "seconds": time.time() - tc}
                log(f"    {name:<22s} "
                    f"{R['rollouts'][name]['param']['med_E']:>7.4f} {CT.fmt(raw['loop'],8)} "
                    f"{kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['loop'],8)} {CT.fmt(bl,8)} "
                    f"{CT.fmt(bh,8)} {CT.fmt(gau['R2'],8)} "
                    f"{gau.get('percell',{}).get('r2',float('nan')):>7.4f} "
                    f"{str(gf['converged']):>5s} [{time.time()-tc:.0f}s]")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

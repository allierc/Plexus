"""bestvc_C.py -- ROUND 6, TASK B.  THE BEST AFFINE VELOCITY MATRIX C FROM MEASURED QUANTITIES.

WHAT C IS, AND THE TWO ROUTES TO IT
====================================================================================================
`mpm_gather` builds C as the MLS estimate of the VELOCITY GRADIENT of the background-grid velocity
field, sampled with the same quadratic B-spline stencil that gathered the particle velocity:

    v_p = sum_i w_ip v_i                      C_p[a,b] = (4/dx^2) sum_i w_ip v_i[a] (x_i - x_p)[b]

and `mpm_strain` integrates F <- (I + dt_sub C) F, so C = Fdot F^-1 as well.  Those are the two
routes.  They are the SAME object in the continuum and have completely different discrete error:

  TEMPORAL  C = Fdot F^-1  needs a time derivative of the measured F at FRAME cadence.
  SPATIAL   C = grad v     needs no time derivative of F at all -- only the velocity field at the
                           frame boundary, differentiated in SPACE on the solver's own grid.

Round 3 used the temporal route with a forward difference; the sibling run used a centred one.  This
script reproduces both as controls and adds the spatial route, which had never been tried.

WHAT EACH ESTIMATOR CONSUMES (stated for every row, and enforced -- nothing reads sy.C0 except the
`oracle`/`ctrl_*` rows, which exist only to bound the scale)
----------------------------------------------------------------------------------------------------
  t_*   measured F at ticks k-2..k+2                             (F only)
  s_*   measured x at ticks k-2..k+2 (v by a centred/5-pt stencil) + the modelling constants
        dx, n_grid, dt_sub and the particle masses (uniform)     (x only -- no F at all)
  s_*_ov  the same with the SIMULATOR's v_p: a control that separates the error the spatial route
        inherits from the derived velocity from the error of the spatial operator itself.

MEASURED, for every estimator
----------------------------------------------------------------------------------------------------
  relC        ||C_est - C_true||_F / ||C_true||_F at tick t0 and averaged over the 8 fit frames
  medE1       end-to-end med|dE/E| of round 3's single-frame fit at tick 165, clean F, ridge0,
              with v LEFT ORACLE so that C is the only thing that moves
  dy0         ||y0(C_est) - y0(C_true)|| / ||y_obs||: the perturbation C actually injects
  medE8       (selected rows) the T=8 stacked naive solve of round 5
  holdout     (selected rows) the held-out one-frame residual at tick 180 -- the acceptance statistic

plus stage `m`: the term-by-term magnitude of C inside the scatter, which is the explanation of the
insensitivity; stage `n`: the best rows again under refute5_fit's realizable grid-48 F noise.

usage:
  PYTHONPATH=/workspace/Plexus/src python bestvc_C.py --device cuda:0 --stages abcmdn
"""
from __future__ import annotations

import argparse
import json
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

from plexus.operators.mpm_grid import stencil_offsets, bspline    # noqa: E402
from plexus.models.entities import _lame                          # noqa: E402
from recover import theta_scale                                   # noqa: E402
import crash_test as CT                                           # noqa: E402
from finject import lerp, assemble_inj, y_of                      # noqa: E402
from refute_round3 import fit                                     # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X                           # noqa: E402
from round5_solve import pstats                                   # noqa: E402
from refute5_fit import NoiseF                                    # noqa: E402
from state_derive import collect, derived_v, install_state, rel   # noqa: E402


# =============================================================================================== #
#  the transfer primitives, taken verbatim from mpm_scatter / mpm_gather so that the estimator is
#  differentiating the same field with the same kernel the solver uses
# =============================================================================================== #
def bs(X, g):
    off = stencil_offsets(2, X.device).to(X.dtype)
    fx, w, flat = bspline(X, g.inv_dx, off, g.shape, False)
    return off, fx, w.to(X.dtype), flat


def gather_C(off, fx, w, flat, vgrid, g):
    """mpm_gather's new_C, verbatim: 4 inv_dx sum_i w_i v_i (x_i - x_p)/dx."""
    N, S = w.shape
    gvn = vgrid[flat].view(N, S, 2)
    dpos = off[None] - fx[:, None, :]
    return 4 * g.inv_dx * (w[..., None, None] * (gvn[..., :, None] @ dpos[..., None, :])).sum(1)


def gather_v(off, fx, w, flat, vgrid):
    N, S = w.shape
    return (w[..., None] * vgrid[flat].view(N, S, 2)).sum(1)


def scatter_v(off, fx, w, flat, vp, Cp, mass, g):
    """mpm_scatter's mass/momentum scatter with ONLY the kinematic part (no stress): the APIC
    transfer of a particle velocity field to the grid.  Cp=None gives the plain PIC scatter."""
    N, S = w.shape
    dpos_phys = (off[None] - fx[:, None, :]) * g.dx
    mom = mass[:, None, None] * vp[:, None, :]
    if Cp is not None:
        mom = mom + mass[:, None, None] * (Cp[:, None] @ dpos_phys[..., None]).squeeze(-1)
    gm = torch.zeros(g.n_cells, device=vp.device, dtype=vp.dtype)
    gmv = torch.zeros(g.n_cells, 2, device=vp.device, dtype=vp.dtype)
    gm.index_add_(0, flat, (w * mass[:, None]).reshape(-1))
    gmv.index_add_(0, flat, (w[..., None] * mom).reshape(-1, 2))
    return gmv / gm.clamp(min=1e-10)[:, None], gm


def cg(Aop, b, x0, iters=400, tol=1e-11):
    x = x0.clone()
    r = b - Aop(x)
    p = r.clone()
    rs = (r * r).sum()
    nb = b.norm()
    it = 0
    for it in range(iters):
        Ap = Aop(p)
        al = rs / (p * Ap).sum().clamp_min(1e-300)
        x = x + al * p
        r = r - al * Ap
        rs2 = (r * r).sum()
        if rs2.sqrt() < tol * nb:
            break
        p = r + (rs2 / rs) * p
        rs = rs2
    return x, it + 1, float(rs.sqrt() / nb.clamp_min(1e-300))


def deconv_grid_v(off, fx, w, flat, vp, mass, g, lam, iters=400):
    """THE INVERSE OF THE GATHER.  v_p = W v_grid exactly (mpm_gather), so the grid velocity field
    is recovered by least squares, regularised toward the PIC scatter with a mass weight:

        min_v  ||W v - v_p||^2 + lam sum_i (m_i/max m) |v_i - v_i^PIC|^2

    Consumes only the measured positions (weights W, masses) and the particle velocities handed in.
    """
    N, S = w.shape
    prior, gm = scatter_v(off, fx, w, flat, vp, None, mass, g)
    R = (gm / gm.max().clamp_min(1e-300) + 1e-6)[:, None]

    def Wm(vg):
        return (w[..., None] * vg[flat].view(N, S, 2)).sum(1)

    def WT(rp):
        o = torch.zeros(g.n_cells, 2, device=vp.device, dtype=vp.dtype)
        o.index_add_(0, flat, (w[..., None] * rp[:, None, :]).reshape(-1, 2))
        return o

    rhs = WT(vp) + lam * R * prior
    vg, nit, res = cg(lambda z: WT(Wm(z)) + lam * R * z, rhs, prior, iters=iters)
    dres = float((Wm(vg) - vp).norm() / vp.norm())
    return vg, {"cg_iters": nit, "cg_res": res, "data_resid": dres}


def gatherC_T(off, fx, w, flat, Q, g):
    """The adjoint of gather_C: (G^T Q)_i = 4 inv_dx sum_p w_ip Q_p (x_i - x_p)/dx."""
    N, S = w.shape
    dpos = off[None] - fx[:, None, :]                                # [N,S,2]
    contrib = 4 * g.inv_dx * w[..., None] * (Q[:, None] @ dpos[..., None]).squeeze(-1)
    o = torch.zeros(g.n_cells, 2, device=Q.device, dtype=Q.dtype)
    o.index_add_(0, flat, contrib.reshape(-1, 2))
    return o


def deconv_grid_vC(off, fx, w, flat, vp, Ct, mass, g, lam, beta, iters=600):
    """THE HYBRID.  The grid velocity field that simultaneously (i) reproduces the measured particle
    velocities through the gather and (ii) is closest to the TEMPORAL estimate of C:

        min_v  ||W v - v_p||^2/||v_p||^2 + beta ||G v - C_t||^2/||C_t||^2 + lam*mass-ridge

    Consumes measured x (through v_p and the weights) AND measured F (through C_t)."""
    N, S = w.shape
    prior, gm = scatter_v(off, fx, w, flat, vp, None, mass, g)
    R = (gm / gm.max().clamp_min(1e-300) + 1e-6)[:, None]
    sv, sc = float(vp.norm()) ** 2, float(Ct.norm()) ** 2

    def Wm(z):
        return (w[..., None] * z[flat].view(N, S, 2)).sum(1)

    def WT(rp):
        o = torch.zeros(g.n_cells, 2, device=vp.device, dtype=vp.dtype)
        o.index_add_(0, flat, (w[..., None] * rp[:, None, :]).reshape(-1, 2))
        return o

    rhs = WT(vp) / sv + (beta / sc) * gatherC_T(off, fx, w, flat, Ct, g) + lam * R * prior
    vg, nit, res = cg(lambda z: (WT(Wm(z)) / sv
                                 + (beta / sc) * gatherC_T(off, fx, w, flat,
                                                           gather_C(off, fx, w, flat, z, g), g)
                                 + lam * R * z), rhs, prior, iters=iters)
    return vg, {"cg_iters": nit, "cg_res": res,
                "data_resid": float((Wm(vg) - vp).norm() / vp.norm())}


def mls_node_grad(off, fx, w, flat, vp, mass, g, ridge=0.5):
    """Weighted LINEAR least-squares fit of the particle velocities in the neighbourhood of every
    grid node,  v(x) ~ a_i + G_i (x - x_i),  weights = the same B-spline w_ip times the mass.
    G_i is the velocity gradient at the node; interpolate it back to the particles with w.
    A mesh-free-style estimator that does NOT go through the grid velocity field."""
    N, S = w.shape
    d = -(off[None] - fx[:, None, :]) * g.dx                       # x_p - x_node, [N,S,2]
    wm = (w * mass[:, None])                                       # [N,S]
    dev, dt = vp.device, vp.dtype
    A0 = torch.zeros(g.n_cells, device=dev, dtype=dt)
    A1 = torch.zeros(g.n_cells, 2, device=dev, dtype=dt)
    A2 = torch.zeros(g.n_cells, 2, 2, device=dev, dtype=dt)
    b0 = torch.zeros(g.n_cells, 2, device=dev, dtype=dt)
    b1 = torch.zeros(g.n_cells, 2, 2, device=dev, dtype=dt)
    A0.index_add_(0, flat, wm.reshape(-1))
    A1.index_add_(0, flat, (wm[..., None] * d).reshape(-1, 2))
    A2.index_add_(0, flat, (wm[..., None, None] * (d[..., :, None] @ d[..., None, :])).reshape(-1, 2, 2))
    b0.index_add_(0, flat, (wm[..., None] * vp[:, None, :]).reshape(-1, 2))
    b1.index_add_(0, flat, (wm[..., None, None]
                            * (vp[:, None, :, None] * d[..., None, :])).reshape(-1, 2, 2))
    M = torch.zeros(g.n_cells, 3, 3, device=dev, dtype=dt)
    M[:, 0, 0] = A0
    M[:, 0, 1:] = A1
    M[:, 1:, 0] = A1
    M[:, 1:, 1:] = A2
    # 0.65 particles per occupied node: the 3x3 node fit is RANK DEFICIENT almost everywhere, so the
    # gradient block carries a Tikhonov term with the units of the fit (A0 * (ridge*dx)^2), which
    # shrinks G toward 0 exactly where the neighbourhood cannot support a slope.
    tik = torch.zeros(3, device=dev, dtype=dt)
    tik[1:] = (ridge * g.dx) ** 2
    eye3 = torch.eye(3, device=dev, dtype=dt)[None]
    M = M + A0[:, None, None] * torch.diag(tik)[None] + (A0.max() * 1e-9) * eye3
    rhs = torch.cat([b0[:, :, None], b1.transpose(1, 2)], dim=2)   # [n,2,3]: [b0_a, b1_a0, b1_a1]
    z = torch.linalg.solve(M[:, None].expand(-1, 2, -1, -1).reshape(-1, 3, 3),
                           rhs.reshape(-1, 3, 1)).reshape(g.n_cells, 2, 3)
    G = z[:, :, 1:]                                                # [n_cells,2,2] node velocity grad
    return (w[..., None, None] * G[flat].view(N, S, 2, 2)).sum(1)


# =============================================================================================== #
#  the estimator table
# =============================================================================================== #
def Fm(B, k, eF):
    return B[k]["F0"] + (eF[k] if eF is not None else 0.0)


_POLY = {}


def poly_w(m, deg, dt):
    """Least-squares local polynomial of degree `deg` on the 2m+1 offsets -m..m; the weights that
    return the DERIVATIVE at the centre.  deg = 2m recovers the classical high-order central
    stencils (m=1 -> c2, m=2 -> c4, m=3 -> c6); deg < 2m is the Savitzky-Golay family."""
    key = (m, deg, dt)
    if key not in _POLY:
        j = np.arange(-m, m + 1, dtype=float)
        V = np.vander(j, deg + 1, increasing=True)                # [2m+1, deg+1]
        P = np.linalg.pinv(V)                                     # coeffs = P @ y
        _POLY[key] = P[1] / dt                                    # d/dt at centre = c1 / dt
    return _POLY[key]


def Fdot(B, k, dt, eF, stencil):
    if isinstance(stencil, tuple):                                # ("poly", m, deg)
        _, m, deg = stencil
        cw = poly_w(m, deg, dt)
        out = None
        for i, j in enumerate(range(-m, m + 1)):
            if cw[i] == 0.0:
                continue
            t = float(cw[i]) * Fm(B, k + j, eF)
            out = t if out is None else out + t
        return out
    if stencil == "fwd":
        return (Fm(B, k + 1, eF) - Fm(B, k, eF)) / dt
    if stencil == "bwd":
        return (Fm(B, k, eF) - Fm(B, k - 1, eF)) / dt
    if stencil == "c2":
        return (Fm(B, k + 1, eF) - Fm(B, k - 1, eF)) / (2 * dt)
    if stencil == "c4":
        return (-Fm(B, k + 2, eF) + 8 * Fm(B, k + 1, eF)
                - 8 * Fm(B, k - 1, eF) + Fm(B, k - 2, eF)) / (12 * dt)
    if stencil == "sg5":
        return (2 * Fm(B, k + 2, eF) + Fm(B, k + 1, eF)
                - Fm(B, k - 1, eF) - 2 * Fm(B, k - 2, eF)) / (10 * dt)
    raise ValueError(stencil)


class CEst:
    """Every estimator of C, all reading the same measured series."""

    def __init__(self, sy, B, dt, lam=1e-3):
        self.sy, self.B, self.dt, self.lam = sy, B, dt, lam
        self.g = sy.g
        self.mass = sy.p.mass.to(sy.dtype)
        self.diag = {}

    # ---- temporal: Fdot F^-1 -------------------------------------------------------------- #
    def temporal(self, k, stencil, eF=None):
        return Fdot(self.B, k, self.dt, eF, stencil) @ torch.linalg.inv(Fm(self.B, k, eF))

    # ---- spatial: grad of the velocity field ---------------------------------------------- #
    def vfield(self, k, vsten="c2", ex=None, oracle_v=False):
        if oracle_v:
            return self.B[k]["v0"]
        return derived_v(self.B, k, self.dt, ex, vsten)

    def spatial(self, k, how="dec", vsten="c2", ex=None, oracle_v=False, shift=False, npass=2):
        B, g = self.B, self.g
        v = self.vfield(k, vsten, ex, oracle_v)
        X = B[k]["x0"] + (ex[k] if ex is not None else 0.0)
        if shift:                       # the gather that DEFINED C ran at the pre-advection position
            X = X - self.sy.dt_sub * v
        off, fx, w, flat = bs(X, g)
        if how == "mls":
            return mls_node_grad(off, fx, w, flat, v, self.mass, g)
        if how == "pic":
            vg, _ = scatter_v(off, fx, w, flat, v, None, self.mass, g)
            return gather_C(off, fx, w, flat, vg, g)
        if how == "apic":
            Cp = None
            for _ in range(npass):
                vg, _ = scatter_v(off, fx, w, flat, v, Cp, self.mass, g)
                Cp = gather_C(off, fx, w, flat, vg, g)
            return Cp
        if how == "dec":
            vg, d = deconv_grid_v(off, fx, w, flat, v, self.mass, g, self.lam)
            self.diag[(k, how, vsten, bool(oracle_v), bool(shift))] = d
            return gather_C(off, fx, w, flat, vg, g)
        raise ValueError(how)

    # ---- hybrid: both routes at once -------------------------------------------------------- #
    def hybrid(self, k, beta, tsten="c2", vsten="c2", eF=None, ex=None, oracle_v=False):
        Ct = self.temporal(k, tsten, eF)
        v = self.vfield(k, vsten, ex, oracle_v)
        X = self.B[k]["x0"] + (ex[k] if ex is not None else 0.0)
        off, fx, w, flat = bs(X, self.g)
        vg, d = deconv_grid_vC(off, fx, w, flat, v, Ct, self.mass, self.g, self.lam, beta)
        self.diag[(k, "hyb", beta, tsten)] = d
        return gather_C(off, fx, w, flat, vg, self.g)

    def blend(self, k, alpha, tsten="c2", eF=None, ex=None):
        return (alpha * self.spatial(k, "dec", "c2", ex)
                + (1 - alpha) * self.temporal(k, tsten, eF))


def build_table(E, k, eF=None, ex=None, extras=True):
    """name -> (C, consumes).  eF/ex are the measurement-noise fields (None = clean)."""
    T = {
        "t_fwd (round3)":      (lambda: E.temporal(k, "fwd", eF), "F(k),F(k+1)"),
        "t_bwd":               (lambda: E.temporal(k, "bwd", eF), "F(k-1),F(k)"),
        "t_c2 (sibling)":      (lambda: E.temporal(k, "c2", eF), "F(k+-1)"),
        "t_c4":                (lambda: E.temporal(k, "c4", eF), "F(k+-1),F(k+-2)"),
        "t_c6":                (lambda: E.temporal(k, ("poly", 3, 6), eF), "F(k+-1..3)"),
        "t_c8":                (lambda: E.temporal(k, ("poly", 4, 8), eF), "F(k+-1..4)"),
        "t_sg5":               (lambda: E.temporal(k, "sg5", eF), "F(k+-1),F(k+-2)"),
        "t_p9d4":              (lambda: E.temporal(k, ("poly", 4, 4), eF), "F(k+-1..4)"),
        "t_p11d6":             (lambda: E.temporal(k, ("poly", 5, 6), eF), "F(k+-1..5)"),
        "s_pic":               (lambda: E.spatial(k, "pic", "c2", ex), "x(k+-1),x(k)"),
        "s_apic2":             (lambda: E.spatial(k, "apic", "c2", ex, npass=2), "x(k+-1),x(k)"),
        "s_apic4":             (lambda: E.spatial(k, "apic", "c2", ex, npass=4), "x(k+-1),x(k)"),
        "s_mls":               (lambda: E.spatial(k, "mls", "c2", ex), "x(k+-1),x(k)"),
        "s_dec":               (lambda: E.spatial(k, "dec", "c2", ex), "x(k+-1),x(k)"),
        "s_dec_c4v":           (lambda: E.spatial(k, "dec", "c4", ex), "x(k+-1),x(k+-2)"),
        "s_dec_shift":         (lambda: E.spatial(k, "dec", "c2", ex, shift=True), "x(k+-1),x(k)"),
        "h_dec+c2 b=1":        (lambda: E.hybrid(k, 1.0, "c2", "c2", eF, ex), "x + F"),
        "h_dec+c4 b=1":        (lambda: E.hybrid(k, 1.0, "c4", "c2", eF, ex), "x + F"),
        "h_dec+c4 b=10":       (lambda: E.hybrid(k, 10.0, "c4", "c2", eF, ex), "x + F"),
        "h_blend0.3+c4":       (lambda: E.blend(k, 0.3, "c4", eF, ex), "x + F"),
    }
    if extras:
        T.update({
            "s_dec_ov  [ctrl]":  (lambda: E.spatial(k, "dec", "c2", ex, oracle_v=True), "TRUE v_p"),
            "s_pic_ov  [ctrl]":  (lambda: E.spatial(k, "pic", "c2", ex, oracle_v=True), "TRUE v_p"),
            "s_mls_ov  [ctrl]":  (lambda: E.spatial(k, "mls", "c2", ex, oracle_v=True), "TRUE v_p"),
        })
    return T


def controls(E, k, seed=4242):
    B = E.B
    C0 = B[k]["C0"]
    gen = torch.Generator(device=C0.device).manual_seed(seed + k)
    perm = torch.randperm(C0.shape[0], generator=gen, device=C0.device)
    nz = torch.randn(C0.shape, generator=gen, device=C0.device, dtype=C0.dtype)
    return {
        "ctrl_oracle":     (lambda: C0.clone(), "TRUE C (oracle)"),
        "ctrl_zero":       (lambda: torch.zeros_like(C0), "nothing (C=0)"),
        "ctrl_scale1.28":  (lambda: 1.28 * C0, "TRUE C, 28% too big"),
        "ctrl_noise28":    (lambda: C0 + 0.28 * C0.norm() / nz.norm() * nz, "TRUE C + 28% noise"),
        "ctrl_perm":       (lambda: C0[perm], "TRUE C, particles shuffled"),
    }


# =============================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="bestvc_C")
    ap.add_argument("--stages", default="abcmdn")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--nodes", type=int, default=48)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "stages": a.stages, "t0": a.t0, "T": a.T,
         "holdout_tick": a.holdout_tick, "lam": a.lam, "sigma_F": SIGMA_F, "sigma_x": SIGMA_X}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 5, a.holdout_tick + 5      # +-5: the widest polynomial time stencil
        sy, B = collect(args, t_lo, t_hi, log)
        nC, n, dt, g = sy.C, sy.n_sub_per_frame, sy.dt, sy.g
        s = theta_scale(nC, sy.device)
        th = sy.theta_true.double()
        ticks = list(range(a.t0, a.t0 + a.T))
        E = CEst(sy, B, dt, lam=a.lam)
        log(f"[collect] {t_lo}..{t_hi}, dt={dt}, n_sub={n}, dx={g.dx:.6g}, "
            f"grid {g.nx}x{g.ny} [{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage a --------- #
        if "a" in a.stages:
            X = B[a.t0]["x0"]
            off, fx, w, flat = bs(X, g)
            gm = torch.zeros(g.n_cells, device=sy.device, dtype=sy.dtype)
            gm.index_add_(0, flat, (w * sy.p.mass.to(sy.dtype)[:, None]).reshape(-1))
            occ = int((gm > 1e-12).sum())
            mass = sy.p.mass.to(sy.dtype)
            cellsz = ((X.max(0).values - X.min(0).values) / g.dx)
            # is C temporally resolved at frame cadence?  (per-particle correlation tick to tick)
            def corrC(x, y):
                x, y = x.reshape(-1), y.reshape(-1)
                return float(((x - x.mean()) @ (y - y.mean())) / (x.std() * y.std() * x.numel()))
            ac = [{"lag": L,
                   "corr": float(np.mean([corrC(B[k]["C0"], B[k + L]["C0"]) for k in ticks])),
                   "rel": float(np.mean([rel(B[k + L]["C0"], B[k]["C0"]) for k in ticks]))}
                  for L in (1, 2, 3)]
            acv = [{"lag": L,
                    "corr": float(np.mean([corrC(B[k]["v0"], B[k + L]["v0"]) for k in ticks])),
                    "rel": float(np.mean([rel(B[k + L]["v0"], B[k]["v0"]) for k in ticks]))}
                   for L in (1, 2, 3)]
            R["stage_a"] = {"Np": int(X.shape[0]), "occupied_grid_nodes": occ,
                            "particles_per_occupied_node": float(X.shape[0] / max(occ, 1)),
                            "sheet_extent_cells": [float(v) for v in cellsz],
                            "mass_uniform": bool(float(mass.std() / mass.mean()) < 1e-12),
                            "mass_cv": float(mass.std() / mass.mean()),
                            "C_autocorr": ac, "v_autocorr": acv,
                            "dt": dt, "dt_sub": sy.dt_sub, "dx": g.dx}
            log(f"\n[a] Np={X.shape[0]}  occupied nodes={occ}  "
                f"particles/node={X.shape[0]/max(occ,1):.2f}  mass uniform="
                f"{R['stage_a']['mass_uniform']}")
            log("    frame-to-frame persistence (is the quantity temporally resolved?)")
            for q, arr in (("C", ac), ("v", acv)):
                log(f"      {q}: " + "  ".join(f"lag{d['lag']} corr {d['corr']:+.4f} "
                                               f"rel {d['rel']:.4f}" for d in arr))

        # ------------------------------------------------------------------ stage b --------- #
        #  accuracy of every estimator against the simulator's own C -- no fits, cheap
        if "b" in a.stages:
            names, acc = None, {}
            for k in ticks:
                T = {**build_table(E, k), **controls(E, k)}
                names = list(T)
                for nm, (f, cons) in T.items():
                    Cst = f()
                    acc.setdefault(nm, {"consumes": cons, "rel": [], "rel_eff": []})
                    acc[nm]["rel"].append(rel(Cst, B[k]["C0"]))
                    acc[nm]["rel_eff"].append(rel(Cst, B[k]["C_eff"]))
            log(f"\n[b] accuracy of C against the simulator's C0, ticks {ticks[0]}..{ticks[-1]}")
            log(f"    {'estimator':<20s} {'consumes':<22s} {'relC@t0':>8s} {'relC_mean':>10s} "
                f"{'vs C_eff':>9s}")
            for nm in names:
                d = acc[nm]
                d["rel_t0"], d["rel_mean"] = d["rel"][0], float(np.mean(d["rel"]))
                d["rel_eff_mean"] = float(np.mean(d["rel_eff"]))
                log(f"    {nm:<20s} {d['consumes']:<22s} {d['rel_t0']:>8.4f} "
                    f"{d['rel_mean']:>10.4f} {d['rel_eff_mean']:>9.4f}")
            R["stage_b"] = acc
            # lambda scan for the deconvolution (chosen from the DATA residual, not from truth)
            sc = []
            for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0):
                E2 = CEst(sy, B, dt, lam=lam)
                r_ = [rel(E2.spatial(k, "dec", "c2"), B[k]["C0"]) for k in ticks]
                dr = float(np.mean([v["data_resid"] for v in E2.diag.values()]))
                sc.append({"lam": lam, "relC_mean": float(np.mean(r_)), "data_resid": dr})
                log(f"    lam {lam:<8g} relC {np.mean(r_):.4f}  data resid {dr:.3e}")
            R["stage_b_lam_scan"] = sc
            # the polynomial time-stencil family: is the temporal route truncation-limited (higher
            # order keeps paying) or ALIASED (it stops paying and then hurts)?
            ps_ = []
            for m in (1, 2, 3, 4, 5):
                for deg in sorted({2, min(4, 2 * m), 2 * m}):
                    if deg > 2 * m:
                        continue
                    r_ = [rel(E.temporal(k, ("poly", m, deg)), B[k]["C0"]) for k in ticks]
                    ps_.append({"m": m, "deg": deg, "relC_mean": float(np.mean(r_))})
                    log(f"    poly m={m} deg={deg} ({2*m+1}-point) relC {np.mean(r_):.4f}")
            R["stage_b_poly_scan"] = ps_
            # the mesh-free node fit: how much shrinkage does 0.65 particles/node need?
            ms_ = []
            for rg in (0.1, 0.3, 1.0, 3.0):
                r_ = [rel(mls_node_grad(*bs(B[k]["x0"], g), derived_v(B, k, dt, None, "c2"),
                                        E.mass, g, ridge=rg), B[k]["C0"]) for k in ticks]
                ms_.append({"ridge": rg, "relC_mean": float(np.mean(r_))})
                log(f"    mls ridge {rg:<5g} relC {np.mean(r_):.4f}")
            R["stage_b_mls_scan"] = ms_
            # blend and hybrid scans
            bl_ = []
            for al in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0):
                r_ = [rel(E.blend(k, al, "c4"), B[k]["C0"]) for k in ticks]
                bl_.append({"alpha": al, "relC_mean": float(np.mean(r_))})
                log(f"    blend a={al:<5g} (a*s_dec + (1-a)*t_c4) relC {np.mean(r_):.4f}")
            R["stage_b_blend_scan"] = bl_
            hy_ = []
            for be in (0.1, 1.0, 10.0, 100.0):
                for ts in ("c2", "c4"):
                    r_ = [rel(E.hybrid(k, be, ts), B[k]["C0"]) for k in ticks]
                    hy_.append({"beta": be, "tsten": ts, "relC_mean": float(np.mean(r_))})
                    log(f"    hybrid beta={be:<6g} t={ts} relC {np.mean(r_):.4f}")
            R["stage_b_hybrid_scan"] = hy_

        # ------------------------------------------------------------------ stage m --------- #
        #  WHERE C ENTERS AND HOW BIG IT IS
        if "m" in a.stages:
            k = a.t0
            injF = lerp(B[k]["F0"], B[k]["F1"], n)
            X, V, C0, F = B[k]["x0"], B[k]["v0"], B[k]["C0"], injF[0]
            mass, p_vol = sy.p.mass.to(sy.dtype), sy.p.p_vol.to(sy.dtype)
            mu, la = _lame(sy.E_true[sy.cid])
            aa, bb, cc, dd = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
            J = aa * dd - bb * cc
            cs, sn = (F[:, 0, 0] + F[:, 1, 1]), (F[:, 1, 0] - F[:, 0, 1])
            rr = torch.sqrt(cs * cs + sn * sn) + 1e-9
            cs, sn = cs / rr, sn / rr
            Rot = torch.stack([torch.stack([cs, -sn], -1), torch.stack([sn, cs], -1)], -2)
            eye = torch.eye(2, device=sy.device, dtype=sy.dtype).expand(F.shape[0], 2, 2)
            tau = (2 * mu[:, None, None] * ((F - Rot) @ F.transpose(-2, -1))
                   + eye * (la * J * (J - 1))[:, None, None])
            stress = (-sy.dt_sub * 4 * g.inv_dx * g.inv_dx) * p_vol[:, None, None] * tau
            affC = mass[:, None, None] * C0
            a_ext = sy.pass0 + sy.gain_true[sy.cid][:, None] * sy.act0
            a_gain = sy.gain_true[sy.cid][:, None] * sy.act0
            off, fx, w, flat = bs(X, g)
            dpos = (off[None] - fx[:, None, :]) * g.dx
            mom_v = mass[:, None, None] * (V + sy.dt_sub * a_ext)[:, None, :]
            mom_g = mass[:, None, None] * (sy.dt_sub * a_gain)[:, None, :]
            mom_C = (affC[:, None] @ dpos[..., None]).squeeze(-1)
            mom_s = (stress[:, None] @ dpos[..., None]).squeeze(-1)

            def wn(t):
                return float((w[..., None] * t).norm())
            mm = {"affine_stress_over_affine_C": float(stress.norm() / affC.norm()),
                  "mom_v": wn(mom_v), "mom_gain_part_of_v": wn(mom_g),
                  "mom_C": wn(mom_C), "mom_stress": wn(mom_s),
                  "mom_C_over_mom_v": wn(mom_C) / wn(mom_v),
                  "mom_C_over_mom_stress": wn(mom_C) / wn(mom_s),
                  "mom_C_over_mom_gain": (wn(mom_C) / wn(mom_g)) if wn(mom_g) > 0 else float("inf"),
                  "act0_norm": float(sy.act0.norm()), "pass0_norm": float(sy.pass0.norm()),
                  "n_substeps_C_acts_in": 1, "n_substeps_per_frame": n}
            R["stage_m_terms"] = mm
            log(f"\n[m] where C enters: affine = stress_dt + mass*C, scattered as affine @ (x_i-x_p)")
            log(f"    ||stress_dt|| / ||mass*C||          {mm['affine_stress_over_affine_C']:.4f}")
            log(f"    momentum: ||m v|| {mm['mom_v']:.4e}  ||affC.d|| {mm['mom_C']:.4e}  "
                f"||stress.d|| {mm['mom_stress']:.4e}  ||m dt a_gain|| {mm['mom_gain_part_of_v']:.4e}")
            log(f"    C-term / v-term {mm['mom_C_over_mom_v']:.4f}   C-term / stress-term "
                f"{mm['mom_C_over_mom_stress']:.4f}   C-term / gain-term "
                f"{mm['mom_C_over_mom_gain']:.4f}")
            log(f"    C is overwritten by mpm_gather every substep and mpm_strain's use of it is "
                f"annihilated by the F injection, so C0 acts in 1 of {n} substeps, in b only.")

            # does a wrong C move the COLUMNS of A, or only the offset y0?
            probe = [0, 37, nC, nC + 37]
            cols = {}
            for nm in ("ctrl_oracle", "t_fwd (round3)", "ctrl_perm"):
                T = {**build_table(E, k, extras=False), **controls(E, k)}
                Cst = T[nm][0]()
                install_state(sy, B[k]["snap"], None, Cst, Jp_one=True)
                z = torch.zeros(2 * nC, device=sy.device, dtype=sy.dtype)
                y0 = y_of(sy, z, n, injF, None)
                cl = []
                for j in probe:
                    e = z.clone()
                    e[j] = 100.0 if j < nC else 1.0
                    cl.append((y_of(sy, e, n, injF, None) - y0) / (100.0 if j < nC else 1.0))
                cols[nm] = {"y0": y0.clone(), "A": torch.stack(cl)}
            yo = (B[k]["x_next"] - B[k]["x0"]).reshape(-1)
            R["stage_m_Asens"] = {}
            for nm in ("t_fwd (round3)", "ctrl_perm"):
                dA = rel(cols[nm]["A"], cols["ctrl_oracle"]["A"])
                dy = float((cols[nm]["y0"] - cols["ctrl_oracle"]["y0"]).norm() / yo.norm())
                R["stage_m_Asens"][nm] = {"rel_dA_probe_cols": dA, "dy0_over_yobs": dy}
                log(f"    {nm:<18s} -> relative change in 4 probe columns of A {dA:.3e}; "
                    f"||dy0||/||y_obs|| {dy:.4f}")

        # ------------------------------------------------------------------ stage c --------- #
        #  END-TO-END, single frame at t0, clean F, ridge0, v LEFT ORACLE
        if "c" in a.stages:
            k = a.t0
            injF = lerp(B[k]["F0"], B[k]["F1"], n)
            yo = (B[k]["x_next"] - B[k]["x0"]).reshape(-1)
            T = {**build_table(E, k), **controls(E, k)}
            tgt = {"ctrl_oracle": 0.007777332098339839, "t_fwd (round3)": 0.01251113155512942,
                   "t_c2 (sibling)": 0.010075}
            log(f"\n[c] END-TO-END med|dE/E|, single frame tick {k}, clean F, ridge0, v ORACLE")
            log(f"    {'estimator':<20s} {'relC':>7s} {'medE':>8s} {'p90E':>7s} {'relL2':>7s} "
                f"{'dy0/|y|':>8s} {'negE':>5s} {'target':>8s}")
            R["stage_c"] = {}
            z = torch.zeros(2 * nC, device=sy.device, dtype=sy.dtype)
            install_state(sy, B[k]["snap"], None, B[k]["C0"].clone(), Jp_one=True)
            y0_ref = y_of(sy, z, n, injF, None)
            for nm, (f, cons) in T.items():
                Cst = f()
                install_state(sy, B[k]["snap"], None, Cst, Jp_one=True)
                y0 = y_of(sy, z, n, injF, None)
                dy = float((y0 - y0_ref).norm() / yo.norm())
                install_state(sy, B[k]["snap"], None, Cst, Jp_one=True)
                sc, _ = fit(sy, n, injF, B[k]["x_next"], B[k]["x0"], th, nC)
                sc.update({"relC": rel(Cst, B[k]["C0"]), "dy0_over_yobs": dy, "consumes": cons})
                R["stage_c"][nm] = sc
                tt = tgt.get(nm)
                log(f"    {nm:<20s} {sc['relC']:>7.4f} {sc['med_E']:>8.5f} {sc['p90_E']:>7.4f} "
                    f"{sc['rel_l2']:>7.4f} {dy:>8.4f} {sc['n_negE']:>5d} "
                    f"{(f'{tt:.5f}' if tt else ''):>8s} [{time.time()-t_start:.0f}s]")
            R["stage_c_targets"] = tgt

        # ------------------------------------------------------------------ stage d --------- #
        #  T = 8 STACKED, the round-5 configuration, plus the held-out acceptance statistic
        if "d" in a.stages:
            sel = ["ctrl_oracle", "t_fwd (round3)", "t_c2 (sibling)", "t_c4",
                   "s_dec", "s_mls", "ctrl_zero", "ctrl_perm"]
            log(f"\n[d] T={a.T} stacked, clean F, naive solve, v ORACLE  "
                f"(round-5 oracle-state target med|dE/E| 0.008562)")
            R["stage_d"] = {}
            thetas = {}
            for nm in sel:
                G0 = torch.zeros(2 * nC, 2 * nC, device=sy.device, dtype=sy.dtype)
                r0 = torch.zeros(2 * nC, device=sy.device, dtype=sy.dtype)
                for k in ticks:
                    T = {**build_table(E, k, extras=False), **controls(E, k)}
                    Cst = T[nm][0]()
                    install_state(sy, B[k]["snap"], None, Cst, Jp_one=True)
                    A, y0, _ = assemble_inj(sy, n, lerp(B[k]["F0"], B[k]["F1"], n), None)
                    Az = A * s[None, :]
                    b = (B[k]["x_next"] - B[k]["x0"]).reshape(-1) - y0
                    G0 += Az.T @ Az
                    r0 += Az.T @ b
                    del A, Az
                    torch.cuda.empty_cache()
                t_hat = torch.linalg.solve(G0, r0) * s
                thetas[nm] = t_hat
                ps = pstats(t_hat.cpu().numpy(), th.cpu().numpy(), nC)
                R["stage_d"][nm] = {"pstats": ps}
                log(f"    {nm:<20s} medE {ps['med_E']:>8.5f} p90 {ps['p90_E']:>7.4f} "
                    f"relL2 {ps['rel_l2']:>7.4f} negE {ps['n_negE']:>3d} corr {ps['corr_E']:>6.4f} "
                    f"meanratio {ps['mean_ratio_E']:>6.4f} [{time.time()-t_start:.0f}s]")

            hk = a.holdout_tick
            injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
            y_obs = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
            Th = {**build_table(E, hk, extras=False), **controls(E, hk)}
            log(f"\n    HELD-OUT one-frame residual at tick {hk} (the acceptance statistic), "
                f"|y_obs| {float(y_obs.norm()):.4e}; v ORACLE at the held-out frame too")
            log(f"    {'theta':<20s} " + " ".join(f"{c:>11s}" for c in
                                                  ("oracleC", "t_fwd", "t_c2", "s_dec")))
            for nm in ["theta_true"] + sel:
                t_h = th if nm == "theta_true" else thetas[nm]
                row = {}
                for cn in ("ctrl_oracle", "t_fwd (round3)", "t_c2 (sibling)", "s_dec"):
                    install_state(sy, B[hk]["snap"], None, Th[cn][0](), Jp_one=True)
                    y = y_of(sy, t_h, n, injh, None)
                    row[cn] = float((y - y_obs).norm() / y_obs.norm())
                R["stage_d"].setdefault(nm, {})["holdout"] = row
                log(f"    {nm:<20s} " + " ".join(f"{row[c]:>11.5f}" for c in
                    ("ctrl_oracle", "t_fwd (round3)", "t_c2 (sibling)", "s_dec")))
            np.savez(os.path.join(HERE, "bestvc_C_theta.npz"),
                     **{k.replace(" ", "_"): v.cpu().numpy() for k, v in thetas.items()},
                     theta_true=th.cpu().numpy())

        # ------------------------------------------------------------------ stage n --------- #
        #  REALIZABLE F NOISE (refute5_fit --noise grid --nodes 48).  C is derived FROM F.
        if "n" in a.stages:
            NF = NoiseF("grid", sy.x0, a.nodes, sy.device, sy.dtype)
            log(f"\n[n] realizable grid-{a.nodes} F noise, sigma_F={SIGMA_F} "
                f"(same draw in the derivation and in the injected F)")
            sel = ["ctrl_oracle", "t_fwd (round3)", "t_c2 (sibling)", "t_c4", "t_sg5",
                   "s_dec", "s_mls", "ctrl_zero"]
            R["stage_n"] = {}
            k = a.t0
            for cond, seeds in (("F only", (90210, 555, 777)), ("F + x", (90210, 555, 777))):
                for seed in seeds:
                    gn = torch.Generator(device=sy.device).manual_seed(seed + 7)
                    eF = {t: (SIGMA_F / 2.0) * NF(gn) for t in B}
                    ex = ({t: SIGMA_X * torch.randn(B[t]["x0"].shape, generator=gn,
                                                    device=sy.device, dtype=sy.dtype) for t in B}
                          if cond == "F + x" else None)
                    injF = lerp(B[k]["F0"] + eF[k], B[k]["F1"] + eF[k + 1], n)
                    xn = B[k]["x_next"] + (ex[k + 1] if ex is not None else 0.0)
                    x0m = B[k]["x0"] + (ex[k] if ex is not None else 0.0)
                    T = {**build_table(E, k, eF, ex, extras=False), **controls(E, k)}
                    for nm in sel:
                        Cst = T[nm][0]()
                        install_state(sy, B[k]["snap"], None, Cst, Jp_one=True)
                        sc, _ = fit(sy, n, injF, xn, x0m, th, nC)
                        key = f"{cond}|{nm}"
                        R["stage_n"].setdefault(key, []).append(
                            {"seed": seed, "relC": rel(Cst, B[k]["C0"]), "med_E": sc["med_E"],
                             "p90_E": sc["p90_E"], "rel_l2": sc["rel_l2"], "n_negE": sc["n_negE"]})
                    log(f"    {cond} seed {seed} done [{time.time()-t_start:.0f}s]")
            log(f"    {'condition':<10s} {'estimator':<20s} {'relC':>7s} {'medE':>8s} "
                f"{'p90E':>7s} {'relL2':>7s}")
            for key, rows in R["stage_n"].items():
                cond, nm = key.split("|")
                log(f"    {cond:<10s} {nm:<20s} "
                    f"{np.mean([r['relC'] for r in rows]):>7.4f} "
                    f"{np.mean([r['med_E'] for r in rows]):>8.5f} "
                    f"{np.mean([r['p90_E'] for r in rows]):>7.4f} "
                    f"{np.mean([r['rel_l2'] for r in rows]):>7.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

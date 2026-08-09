"""finject.py -- ROUND 3, part 2. THE ROUTE TO REAL DATA: inject the measured deformation gradient
instead of integrating it.

THE PROBLEM THIS ATTACKS
====================================================================================================
The formulation is exact per SUBSTEP (dt_sub = 2e-4) and broken per FRAME (dt = 2e-3 = 10 substeps):
over a frame, F and C become theta-dependent from substep 2, superposition fails, the frame-cadence
defect is 0.515 on the displacement read-out and the recovered med|dE/E| is 0.257. Real data offers
frames, not substeps -- so the frame defect IS the method's real-data error floor.

The real recording measures the deformation gradient directly: channels 2..5 of
`...derivatives.npy` are du/dx, du/dy, dv/dx, dv/dy, so F = I + grad u is OBSERVED at every frame
and does not have to be integrated. If F-drift is what breaks superposition, then holding /
interpolating the MEASURED F across the substeps of a frame should repair it.

WHAT IS MEASURED HERE, on the synthetic system where the answer is known
----------------------------------------------------------------------------------------------------
A ladder of injections, all at FRAME cadence with the displacement read-out y(theta) = x_end - x_0:

  none        the baseline (round 1's frame_DISP)
  F_hold      p.F <- F(t) at every substep                        REALIZABLE (one measured frame)
  F_lerp      p.F <- F(t) + (s+1)/n (F(t+1) - F(t))               REALIZABLE (two measured frames)
  F_true      p.F <- the reference F at that substep              ORACLE upper bound
  C_*         the same three for the affine velocity matrix C
  FC_*        both

For each: (i) the AFFINITY defect rel(A theta_true - b_self, b_self) with b_self the injected
model's own response -- does injection make the frame map linear in theta? and (ii) the RECOVERY
error from fitting the TRUE observed displacement -- does the injected model also predict the right
thing? These are different questions and injection can win the first and lose the second.

Then the winner is put through the crash test (free 150-frame rollout, registry, round 3's 2-D
gauge) so that "the equation is satisfied" and "the trajectory is right" stay separate.

usage: PYTHONPATH=/workspace/Plexus/src python finject.py --device cuda:0
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

from plexus.models.entities import _lame                        # noqa: E402
from assemble import SUBSTEP_TOKENS, rel                        # noqa: E402
from recover import Solver, install_E, score, theta_scale       # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  one frame, with optional per-substep state injection
# --------------------------------------------------------------------------------------------- #
def step_inj(sy, E_cell, gain_cell, n_sub, injF=None, injC=None):
    """`refute_round1.step_disp`, plus: overwrite p.F after mpm_strain and/or p.C after mpm_gather.

    injF/injC : [n_sub, Np, 2, 2] tensors or None. Injection is a no-op when the injected values
    are exactly the ones the free run would have produced, which is the control below.
    """
    H, p = sy.H, sy.p
    sy.restore()
    mu, la = _lame(E_cell[sy.cid])
    p.mu, p.la = mu, la
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + gain_cell[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    for s in range(n_sub):
        sy._tok("mpm_strain")
        if injF is not None:
            p.F = injF[s].clone()
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        if injC is not None:
            p.C = injC[s].clone()
    H.sub_dt = None
    return (p.get("pos") - sy.x0).reshape(-1).clone()


def y_of(sy, theta, n_sub, injF=None, injC=None):
    E = torch.zeros(sy.C + 1, device=sy.device, dtype=sy.dtype)
    gn = torch.zeros_like(E)
    E[1:], gn[1:] = theta[:sy.C], theta[sy.C:]
    return step_inj(sy, E, gn, n_sub, injF, injC)


def assemble_inj(sy, n_sub, injF=None, injC=None, sE=100.0, sg=1.0):
    t0 = time.time()
    z = torch.zeros(2 * sy.C, device=sy.device, dtype=sy.dtype)
    y0 = y_of(sy, z, n_sub, injF, injC)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = z.clone()
        e[j] = s
        A[:, j] = (y_of(sy, e, n_sub, injF, injC) - y0) / s
    torch.cuda.synchronize()
    return A, y0, time.time() - t0


def record_substeps(sy, n_sub):
    """The reference substep trajectory of (F after strain, C after gather, x) under theta_true."""
    H, p = sy.H, sy.p
    sy.restore()
    install_E(sy, sy.E_true)
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + sy.gain_true[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    Fs, Cs, Xs = [], [], []
    for _ in range(n_sub):
        sy._tok("mpm_strain")
        Fs.append(p.F.clone())
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        Cs.append(p.C.clone())
        Xs.append(p.get("pos").clone())
    H.sub_dt = None
    return torch.stack(Fs), torch.stack(Cs), torch.stack(Xs)


def hold(v0, n):
    return v0[None].expand(n, *v0.shape).contiguous()


# --------------------------------------------------------------------------------------------- #
#  ROUND 4, THE ONE CHANGE.  The measured F sits in A, not in b, so ordinary least squares in it
#  is ERRORS-IN-VARIABLES and its leading defect is a BIAS (attenuation), not variance:
#  refute3_simex measured mean(E_hat)/mean(E) = 0.391 and slope 0.344 at the recording's own
#  sigma_F = 3.9e-3, falling monotonically as more known noise is added.  Ridge, more frames and
#  better conditioning all attack variance and none of them touch that.
#
#  Noise-corrected normal equations, with the step refute3_debias.py was missing -- a TRUNCATED
#  subspace solve of the CORRECTED Gram, because subtracting Sigma makes G indefinite (min eig
#  -3.2e-8 against a naive +7e-15) and torch.linalg.solve then returns |mean ratio| = 5.8.
#
#  sigma_F is read off the recording (real_F_check.py's quiet-stretch second difference, verified
#  temporally white, lag-1 0.0006).  Nothing about theta_true is used.
# --------------------------------------------------------------------------------------------- #
def _noise_F(F0h, F1h, sigma_F, n, gen):
    """One COHERENT error per frame boundary, then lerp -- never a fresh draw per substep.

    The convention (sigma_F/2 on each of the two frame boundaries) is refute_round3.py's phase-D
    `coherent_per_frame`, kept identical so the numbers are comparable.
    """
    def d(x):
        return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return lerp(F0h + (sigma_F / 2.0) * d(F0h), F1h + (sigma_F / 2.0) * d(F1h), n)


def solve_eiv(sy, n, F0h, F1h, xn, x0, sigma_F, K=8, taus=(1.0,), gen=None, keep_A=False):
    """Least squares in a MEASURED regressor, corrected for the regressor's own noise.

    1. assemble A_hat, y0_hat from the measured F_hat (lerp across the frame); column-scale with
       recover.theta_scale; G0 = A^T A, r0 = A^T b.
    2. K times, draw a fresh COHERENT error at the measured sigma_F, re-noise the ALREADY NOISY
       F_hat, re-assemble, and average -> Gbar, rbar.
    3. Sigma = Gbar - G0, c = rbar - r0;  Gc = G0 - Sigma,  rc = r0 - c.
    4. eigendecompose Gc, keep only lambda_i > tau*||Sigma||_2, solve there (truncated pinv of the
       CORRECTED Gram).  Report the retained rank and lambda_min.

    Returns (out, extra) where out maps a solver name to a theta.  The names are:
       naive            plain ridge0 on G0                     (the control this has to beat)
       naive_trunc<tau> the SAME truncation applied to G0 only  (truncation without correction)
       eiv_trunc<tau>   the change under test
       eiv_full         the corrected Gram solved without truncation (refute3_debias's version)
    """
    C = sy.C
    s = theta_scale(C, sy.device)
    A0, y00, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
    b0 = (xn - x0).reshape(-1) - y00
    Az = A0 * s[None, :]
    G0 = Az.T @ Az
    r0 = Az.T @ b0
    if not keep_A:
        del A0, Az
        torch.cuda.empty_cache()

    Gb = torch.zeros_like(G0)
    rb = torch.zeros_like(r0)
    for _ in range(K):
        Ak, y0k, _ = assemble_inj(sy, n, _noise_F(F0h, F1h, sigma_F, n, gen), None)
        Azk = Ak * s[None, :]
        Gb += Azk.T @ Azk
        rb += Azk.T @ ((xn - x0).reshape(-1) - y0k)
        del Ak, Azk
        torch.cuda.empty_cache()
    Gb /= K
    rb /= K
    Sig, cvec = Gb - G0, rb - r0
    Gc, rc = G0 - Sig, r0 - cvec

    def trunc(G, rhs, thr):
        ev, U = torch.linalg.eigh(G)
        keep = ev > thr
        if int(keep.sum()) == 0:
            return torch.zeros_like(rhs), 0, float(ev.max())
        Uk = U[:, keep]
        z = Uk @ ((Uk.T @ rhs) / ev[keep])
        return z * s, int(keep.sum()), float(ev[keep].min())

    # ---- the Sigma-metric ("SNR") truncation ------------------------------------------------
    # The prescribed threshold lambda_i > tau*||Sigma||_2 is ISOTROPIC, and Sigma is not: at
    # t0 = 165 the E columns of the scaled A are 2.8x the gain columns in norm but Sigma's mass
    # sits almost entirely on the E block (its gain diagonal is exactly 0, because the gain
    # multiplies the active-force delta and not F).  One scalar threshold therefore throws away
    # the whole weaker block -- measured below, it keeps 66/200 directions and returns theta ~ 0.
    # The direction-wise version of the same idea is the generalised problem Gc v = lambda Sigma v,
    # whose eigenvalue IS the signal-to-noise ratio along v.  Sigma is a K-sample estimate and can
    # be indefinite, so it is floored at delta = 1e-2 ||Sigma||_2 before whitening.
    w, U = torch.linalg.eigh(Sig)
    sig2 = float(w.abs().max())
    wf = w.clamp(min=1e-2 * sig2) if sig2 > 0 else w + 1.0
    Lh = U @ torch.diag(wf.rsqrt()) @ U.T
    Mw = Lh @ Gc @ Lh
    lam, V = torch.linalg.eigh((Mw + Mw.T) / 2)
    Vw = Lh @ V

    def gtrunc(thr):
        keep = lam > thr
        if int(keep.sum()) == 0:
            return torch.zeros_like(rc), 0, float(lam.max())
        Vk = Vw[:, keep]
        return (Vk @ ((Vk.T @ rc) / lam[keep])) * s, int(keep.sum()), float(lam[keep].min())

    ev0 = torch.linalg.eigvalsh(Gc)
    out, extra = {}, {"sigma_spec_norm": sig2, "G0_spec_norm": float(torch.linalg.eigvalsh(G0).max()),
                      "sigma_fro_over_G_fro": float(Sig.norm() / G0.norm()),
                      "min_eig_corrected": float(ev0.min()),
                      "min_eig_naive": float(torch.linalg.eigvalsh(G0).min()), "K": K,
                      "rank": {}, "lam_min_retained": {}}
    try:
        out["naive"] = torch.linalg.solve(G0, r0) * s
    except Exception:
        out["naive"] = torch.linalg.lstsq(G0, r0.unsqueeze(1)).solution.squeeze(1) * s
    try:
        out["eiv_full"] = torch.linalg.solve(Gc, rc) * s
    except Exception:
        out["eiv_full"] = torch.linalg.lstsq(Gc, rc.unsqueeze(1)).solution.squeeze(1) * s
    for tau in taus:
        thr = tau * sig2
        t_e, rk, lm = trunc(Gc, rc, thr)
        out[f"eiv_trunc{tau:g}"] = t_e
        extra["rank"][f"eiv_trunc{tau:g}"] = rk
        extra["lam_min_retained"][f"eiv_trunc{tau:g}"] = lm
        t_n, rk_n, lm_n = trunc(G0, r0, thr)
        out[f"naive_trunc{tau:g}"] = t_n
        extra["rank"][f"naive_trunc{tau:g}"] = rk_n
        extra["lam_min_retained"][f"naive_trunc{tau:g}"] = lm_n
    for tau in (0.0,) + tuple(taus):
        t_g, rk, lm = gtrunc(tau)
        out[f"eiv_snr{tau:g}"] = t_g
        extra["rank"][f"eiv_snr{tau:g}"] = rk
        extra["lam_min_retained"][f"eiv_snr{tau:g}"] = lm
    extra["snr_spectrum_quantiles"] = [float(v) for v in
                                       torch.quantile(lam, torch.tensor(
                                           [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                                           device=lam.device, dtype=lam.dtype))]
    extra["n_snr_gt1"] = int((lam > 1).sum())
    # the generalised basis, so the caller can ask WHERE the information is without the estimator
    # ever seeing theta_true
    extra["_lam"], extra["_V"] = lam, Vw
    if keep_A:
        extra["A"] = A0
    return out, extra


def lerp(v0, v1, n):
    w = torch.arange(1, n + 1, device=v0.device, dtype=v0.dtype).view(n, *([1] * v0.dim())) / n
    return v0[None] + w * (v1 - v0)[None]


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="finject")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--rollout", action="store_true", help="also crash-test the recovered thetas")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.warmup, window=a.window, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G, n = sy.C, args.warmup, args.window, sy.n_sub_per_frame
        th = sy.theta_true.double()
        x0, dx, cid = sy.x0.clone(), sy.g.dx, sy.cid

        Fs, Cs, Xs = record_substeps(sy, n)
        x_next = Xs[-1].clone()
        y_obs = (x_next - x0).reshape(-1)
        F0, C0 = sy.F0.clone(), sy.C0.clone()
        F1, C1 = Fs[-1].clone(), Cs[-1].clone()      # what a measurement at frame t+1 would give
        R["within_frame_drift"] = {
            "F_relchange_over_frame": float((F1 - F0).norm() / F0.norm()),
            "C_relchange_over_frame": float((C1 - C0).norm() / (C0.norm() + 1e-30)),
            "x_displacement_over_frame_dx": float((x_next - x0).norm(dim=1).max() / dx),
            "F_hold_error": float((Fs - hold(F0, n)).norm() / Fs.norm()),
            "F_lerp_error": float((Fs - lerp(F0, F1, n)).norm() / Fs.norm()),
            "C_hold_error": float((Cs - hold(C0, n)).norm() / Cs.norm()),
            "C_lerp_error": float((Cs - lerp(C0, C1, n)).norm() / Cs.norm())}
        log(f"[frame] {n} substeps; |dF|/|F| over the frame "
            f"{R['within_frame_drift']['F_relchange_over_frame']:.3e}; max displacement "
            f"{R['within_frame_drift']['x_displacement_over_frame_dx']:.4f} dx")
        log(f"        approximation error of the REALIZABLE injections: F_hold "
            f"{R['within_frame_drift']['F_hold_error']:.3e}  F_lerp "
            f"{R['within_frame_drift']['F_lerp_error']:.3e}  C_hold "
            f"{R['within_frame_drift']['C_hold_error']:.3e}  C_lerp "
            f"{R['within_frame_drift']['C_lerp_error']:.3e}")

        # ---- control: injecting the reference values is a no-op ---------------------------------
        y_ctrl = y_of(sy, th, n, Fs, Cs)
        y_free = y_of(sy, th, n, None, None)
        R["control_injection_is_noop"] = {
            "max_abs_diff_world": float((y_ctrl - y_obs).abs().max()),
            "free_run_matches_reference": float((y_free - y_obs).abs().max())}
        log(f"[control] injecting the reference (F,C) reproduces the reference displacement to "
            f"{R['control_injection_is_noop']['max_abs_diff_world']:.3e} world "
            f"({R['control_injection_is_noop']['max_abs_diff_world']/dx:.2e} dx); free run "
            f"{R['control_injection_is_noop']['free_run_matches_reference']:.3e}")

        VAR = [("none", None, None),
               ("F_hold", hold(F0, n), None),
               ("F_lerp", lerp(F0, F1, n), None),
               ("F_true", Fs, None),
               ("C_hold", None, hold(C0, n)),
               ("C_lerp", None, lerp(C0, C1, n)),
               ("C_true", None, Cs),
               ("FC_hold", hold(F0, n), hold(C0, n)),
               ("FC_lerp", lerp(F0, F1, n), lerp(C0, C1, n)),
               ("FC_true", Fs, Cs)]

        R["variants"] = {}
        thetas = {}
        log(f"\n[ladder] frame cadence (n_sub={n}), displacement read-out")
        log(f"    {'variant':<10s} {'affinity':>10s} {'model_bias':>10s} {'fit_res':>9s} "
            f"{'medE':>8s} {'medg':>8s} {'l2':>8s} {'negE':>5s} {'cond':>10s}")
        for name, iF, iC in VAR:
            t_v = time.time()
            A, y0, t_as = assemble_inj(sy, n, iF, iC)
            y_self = y_of(sy, th, n, iF, iC)            # the injected model's OWN response
            b_self = y_self - y0
            aff = rel(A @ sy.theta_true - b_self, b_self)
            bias = float((y_self - y_obs).norm() / y_obs.norm())
            b = y_obs - y0                              # the OBSERVED displacement
            S = Solver(A, C)
            sol = S(b)
            t_hat = sol["ridge0"]
            sc = {k: score(v, th, C) for k, v in sol.items()}
            fit = rel(A @ t_hat - b, b)
            R["variants"][name] = {"affinity_defect": aff, "model_bias_vs_truth": bias,
                                   "fit_residual": fit, "cond": S.cond, "assembly_s": t_as,
                                   "residual_true": rel(A @ sy.theta_true - b, b),
                                   "scores": sc, "seconds": time.time() - t_v}
            thetas[name] = t_hat.clone()
            S.free(); del A, S
            torch.cuda.empty_cache()
            s0 = sc["ridge0"]
            log(f"    {name:<10s} {aff:>10.3e} {bias:>10.3e} {fit:>9.3e} {s0['med_E']:>8.4f} "
                f"{s0['med_gain']:>8.4f} {s0['rel_l2']:>8.4f} "
                f"{int((t_hat[:C] < 0).sum()):>5d} {R['variants'][name]['cond']:>10.2e}")

        # ---- noise: the only thing real data adds that this does not have -----------------------
        best = min(("F_hold", "F_lerp", "FC_hold", "FC_lerp", "none"),
                   key=lambda k: R["variants"][k]["scores"]["ridge0"]["med_E"])
        R["best_realizable"] = best
        iF, iC = dict((v[0], (v[1], v[2])) for v in VAR)[best]
        A, y0, _ = assemble_inj(sy, n, iF, iC)
        S = Solver(A, C)
        R["noise_sweep"] = {}
        gen = torch.Generator(device=sy.device).manual_seed(31)
        log(f"\n[noise] best realizable variant = {best}; Gaussian noise on the observed "
            f"positions (1 px = 4.88e-4 world, dx = {dx:.3e})")
        for sig in (0.0, 1e-6, 1e-5, 1e-4):
            errs = []
            for _ in range(3 if sig > 0 else 1):
                xn = x_next + sig * torch.randn(x_next.shape, generator=gen, device=sy.device,
                                                dtype=sy.dtype)
                bb = (xn - x0).reshape(-1) - y0
                errs.append(score(S(bb)["ridge0"], th, C))
            R["noise_sweep"][f"{sig:g}"] = {
                "med_E": float(np.mean([e["med_E"] for e in errs])),
                "med_gain": float(np.mean([e["med_gain"] for e in errs])),
                "rel_l2": float(np.mean([e["rel_l2"] for e in errs]))}
            v = R["noise_sweep"][f"{sig:g}"]
            log(f"    sigma {sig:<8g} med|dE/E| {v['med_E']:.4f}  med|dg/g| {v['med_gain']:.4f}  "
                f"l2 {v['rel_l2']:.4f}")
        S.free(); del A, S
        torch.cuda.empty_cache()

        np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                 **{f"inj::{k}": v.cpu().numpy() for k, v in thetas.items()})

        # ---- the crash test on the recovered thetas ---------------------------------------------
        if a.rollout:
            import crash_round3 as R3
            tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                       for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
            band = 0.06 / MET.SHEET_SPAN
            anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                      (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
            interior = ~anchor
            ref_full = torch.zeros(G, sy.Np, 2, device=sy.device, dtype=sy.dtype)
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
            a_ref, _ = R3.percell_amplitude(ref_full, x0, cid, C, interior)
            keep = np.isfinite(a_ref) & (a_ref > 0)
            R["a_ref_percell"] = a_ref.tolist()
            R["keep_percell"] = keep.tolist()

            def scored(theta, full_out=True):
                tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                              anchor=None, interior=interior, ss_tot=ss_tot,
                                              keep_full=full_out, band_mask=anchor)
                m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                out = {"margin20": m20, "coarse": coarse,
                       "t1": coarse["motion_energy_ratio_interior"], "t2": R3.t2_of(m20)}
                if full_out:
                    ah, _ = R3.percell_amplitude(full, x0, cid, C, interior)
                    out["percell"] = R3.r2_percell(ah, a_ref, keep)
                    out["a_percell"] = ah.tolist()
                    del full
                return out

            R["rollouts"] = {}
            log(f"\n[crash test] recovered thetas, {G}-frame free rollout, margin-20, "
                f"raw and 2-D gauged")
            for name in ["theta_true"] + [v[0] for v in VAR]:
                theta = th if name == "theta_true" else thetas[name]
                raw = scored(theta)

                def probe(lE, lg, theta=theta):
                    d = scored(R3.scale2(theta, float(np.exp(lE)), float(np.exp(lg)), C),
                               full_out=False)
                    return (d["t1"], d["t2"])
                gf = R3.gauge_fix2(probe, (raw["t1"], raw["t2"]))
                kE, kg = gf["k_E"], gf["k_g"]
                gau = raw if (kE == 1.0 and kg == 1.0) else scored(R3.scale2(theta, kE, kg, C))
                R["rollouts"][name] = {"theta_error": score(theta, th, C), "raw": raw,
                                       "gauged": gau, "gauge": gf}
                log(f"    {name:<10s} medE {score(theta, th, C)['med_E']:>7.4f} | raw loop "
                    f"{CT.fmt(raw['margin20']['loopscore'],8)} t1 {raw['t1']:>6.3f} | kE {kE:>6.3f} "
                    f"kg {kg:>6.3f} gauged loop {CT.fmt(gau['margin20']['loopscore'],8)} R2 "
                    f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)} r2cell "
                    f"{gau['percell']['r2']:>7.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

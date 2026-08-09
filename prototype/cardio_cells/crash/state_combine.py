"""state_combine.py -- ROUND 6, TASK 2.  THE FACTOR-20 MEASUREMENT.

THE QUESTION
====================================================================================================
Two error sources have each been measured ALONE and never together:

    (i)  removing the STATE ORACLE (System.restore(), algebraic/assemble.py:225, hands the estimator
         the true v, C, Jp at every one of the 401 assemblies of every frame).  Zero noise, T=1:
         med|dE/E| 0.0078 -> 0.0404 (round 3, forward C).  Zero noise, T=8, centred C (task 1):
         0.00856 -> 0.04880.
    (ii) the recording's REALIZABLE, spatially-correlated F noise (refute5_fit.py --noise grid
         --nodes 48): med|dE/E| ~0.15 with a full state oracle.

    quadrature      sqrt(0.0404^2 + 0.15^2) = 0.1553          -> the oracle is free
    multiplicative  (0.0404/0.0086) * 0.15  = 0.7047          -> no per-cell content at all

Both errors enter A, not b, so both are errors-in-variables and both produce ATTENUATION; two
attenuations compose multiplicatively, which is why the pessimistic branch is physically plausible.
A factor of 20 in the answer.  This script runs the missing cell.

THE 2x2 (T=8, t0=165, frame cadence, displacement read-out, F injected by lerp)
----------------------------------------------------------------------------------------------------
                      | true v,C (oracle)                | derived v,C
    clean F           | state_derive.py stage 8          | state_derive.py stage 8
    realizable F noise| refute5_norm_grid48_s{seed}      | THIS SCRIPT

The derived state is task 1's recipe, applied to the MEASURED series:
    v_k  <- (x_meas(k+1) - x_meas(k-1)) / (2 dt)
    C_k  <- ((F_meas(k+1) - F_meas(k-1)) / (2 dt)) @ inv(F_meas(k))
    Jp_k <- 1

MEASUREMENT MODEL -- identical to refute5_fit.py, extended coherently in time
----------------------------------------------------------------------------------------------------
refute5_fit draws ONE spatially-correlated F error per frame BOUNDARY (shared by the two frames that
use it) and one white position error per frame END.  Here the same fields are indexed by TICK, so
the same realization also feeds the state derivation:
    F_meas(t) = F0(t) + eF[t],   eF[t] = (sigma_F/2) * NoiseF(grid,48)
    x_meas(t) = x0(t) + ex[t],   ex[t]  = sigma_x * randn
The draw ORDER is chosen so that eF[t0+j] and ex[t0+1+k] are bit-identical to refute5_fit's `eb[j]`
and `xs[k]` for the same seed -- stage `c` asserts this by re-assembling frame 0 with an oracle
state and comparing G0/r0 against refute5_norm_grid48_s{seed}.npz.  Ticks outside refute5's range
(t0-1, t0) are drawn afterwards from a separate generator, so nothing shared is disturbed.

Two observation conventions are assembled in the same pass (they differ only in b, so the cost is
one extra mat-vec):
    r0   b = x_meas(k+1) - x0_true(k)     <- refute5's convention (clean start), for comparability
    r0c  b = x_meas(k+1) - x_meas(k)      <- fully measured, sqrt(2) noise, the honest one

The EIV Monte-Carlo (K re-noisings) re-derives v and C from the doubly-perturbed measurements, so
the correction sees the noise-induced part of the state error too.  It cannot see the ZERO-NOISE
truncation floor (v 1.0%, C 13.8%), which is a deterministic bias -- that is the point of the cell.

STAGES (--stage)
----------------------------------------------------------------------------------------------------
  c  control: frame 0, oracle state, refute5's convention -> must equal refute5_norm_grid48_s{seed}
  f  fit: T frames, derived state, realizable noise -> state_norm_grid48_der_s{seed}.npz
  s  solve: naive / naive_box / eiv_snr0 / eiv_box on the new fits (round5_solve, unmodified)
  r  score: free 150-frame rollout, margin 20, 2-D gauge; held-out one-frame residual with an
     ORACLE and with a DERIVED holdout state; metrics.py imported unmodified

usage:
  PYTHONPATH=/workspace/Plexus/src python state_combine.py --stage f --seed 90210 --device cuda:1
  PYTHONPATH=/workspace/Plexus/src python state_combine.py --stage s
  PYTHONPATH=/workspace/Plexus/src python state_combine.py --stage r --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import glob
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

from assemble import SUBSTEP_TOKENS                                    # noqa: E402
from recover import theta_scale, install_E                             # noqa: E402
import metrics as MET                                                  # noqa: E402
import crash_test as CT                                                # noqa: E402
from crash_round2 import percell_amplitude, r2_percell                 # noqa: E402
from crash_round3 import scale2, t2_of                                 # noqa: E402
from finject import record_substeps, lerp, assemble_inj, y_of          # noqa: E402
from refute_round3 import advance                                      # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                          # noqa: E402
from round5_solve import solve_box, snr_trunc, pstats                  # noqa: E402
from round5_score import gauge_grid                                    # noqa: E402
from refute5_fit import NoiseF                                         # noqa: E402
import state_derive as SD                                              # noqa: E402

PLANTED_MAX_E = 216.3


# --------------------------------------------------------------------------------------------- #
def draw_noise(sy, NF, seed, t0, T, sigma_F, sigma_x, extra_ticks=()):
    """eF/ex keyed by TICK, drawn in refute5_fit.py's exact order for the shared ticks."""
    gm = torch.Generator(device=sy.device).manual_seed(seed)
    eF, ex = {}, {}
    for j in range(T + 1):                                   # refute5_fit: eb[0..T]
        eF[t0 + j] = (sigma_F / 2.0) * NF(gm)
    for k in range(T):                                       # refute5_fit: xs[0..T-1]
        eF_shape = sy.x0.shape
        ex[t0 + 1 + k] = sigma_x * torch.randn(eF_shape, generator=gm, device=sy.device,
                                               dtype=sy.dtype)
    ge = torch.Generator(device=sy.device).manual_seed(seed + 1013)
    for t in extra_ticks:                                    # ticks refute5 never needed
        if t not in eF:
            eF[t] = (sigma_F / 2.0) * NF(ge)
        if t not in ex:
            ex[t] = sigma_x * torch.randn(sy.x0.shape, generator=ge, device=sy.device,
                                          dtype=sy.dtype)
    return eF, ex


def fit_stage(a, args, log, R):
    """T frames of normal equations with a DERIVED state and realizable F noise."""
    t_lo, t_hi = a.t0 - 2, a.t0 + a.T + 1
    sy, B = SD.collect(args, t_lo, t_hi, log)
    C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
    s = theta_scale(C, sy.device)
    NF = NoiseF(a.noise, sy.x0, a.nodes, sy.device, sy.dtype)
    R["noise_eff_samples"] = None
    gchk = torch.Generator(device=sy.device).manual_seed(7)
    cs = []
    for _ in range(24):
        e = NF(gchk).reshape(-1, 4)
        m = torch.zeros(C + 1, 4, device=sy.device, dtype=sy.dtype)
        m.index_add_(0, sy.cid, e)
        cnt = torch.zeros(C + 1, device=sy.device, dtype=sy.dtype)
        cnt.index_add_(0, sy.cid, torch.ones_like(e[:, 0]))
        cs.append(float((m[1:] / cnt[1:, None]).std()))
    R["noise_eff_samples"] = float(1.0 / np.mean(cs) ** 2)
    log(f"[noise {a.noise}{a.nodes}] effective independent F samples per cell "
        f"{R['noise_eff_samples']:.1f}  (round-5 indep = 100, recording = 22.8)")

    ticks = list(range(a.t0, a.t0 + a.T))
    need = sorted(set([t - 1 for t in ticks] + ticks + [t + 1 for t in ticks]))
    eF, ex = draw_noise(sy, NF, a.seed, a.t0, a.T, a.sigma_F, a.sigma_x, extra_ticks=need)

    # ---- stage c: does the draw reproduce refute5_fit's, bit for bit? ------------------------- #
    ref = os.path.join(HERE, f"refute5_norm_{a.noise}{a.nodes}_s{a.seed}_sF{a.sigma_F:g}.npz")
    if a.control and os.path.exists(ref):
        k = a.t0
        SD.install_state(sy, B[k]["snap"])                      # ORACLE state, refute5's setting
        F0h, F1h = B[k]["F0"] + eF[k], B[k]["F1"] + eF[k + 1]
        A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
        Az = A * s[None, :]
        b = ((B[k]["x_next"] + ex[k + 1]) - B[k]["x0"]).reshape(-1) - y0
        G0, r0 = Az.T @ Az, Az.T @ b
        Z = np.load(ref)
        gr = torch.as_tensor(Z["G0"], device=sy.device, dtype=sy.dtype)
        rr = torch.as_tensor(Z["r0"], device=sy.device, dtype=sy.dtype)
        R["control_vs_refute5"] = {
            "file": os.path.basename(ref),
            "rel_G0": float((G0 - gr).norm() / gr.norm()),
            "rel_r0": float((r0 - rr).norm() / rr.norm()),
            "max_abs_G0": float((G0 - gr).abs().max())}
        log(f"[control] frame 0 oracle-state assembly vs {os.path.basename(ref)}: "
            f"rel G0 {R['control_vs_refute5']['rel_G0']:.3e}  "
            f"rel r0 {R['control_vs_refute5']['rel_r0']:.3e}")
        del A, Az
        torch.cuda.empty_cache()
        if not a.fit:
            return sy, B, None

    # ---- the derivation error at this noise level, for the record ----------------------------- #
    derr = []
    for k in ticks:
        v, Cc, _ = SD.derived_state(B, k, dt, eF, ex)
        derr.append({"tick": k, "v_rel": SD.rel(v, B[k]["v0"]), "C_rel": SD.rel(Cc, B[k]["C0"])})
    R["derivation_error_at_noise"] = {
        "v_mean": float(np.mean([d["v_rel"] for d in derr])),
        "C_mean": float(np.mean([d["C_rel"] for d in derr])), "per_tick": derr}
    log(f"[derivation] at this noise draw: v {R['derivation_error_at_noise']['v_mean']:.4f}  "
        f"C {R['derivation_error_at_noise']['C_mean']:.4f}")

    gk = torch.Generator(device=sy.device).manual_seed(31337 + a.seed)
    out, R["frames"] = {}, []
    t_start = time.time()
    for i, k in enumerate(ticks):
        v, Cc, _ = SD.derived_state(B, k, dt, eF, ex)
        SD.install_state(sy, B[k]["snap"], v, Cc, Jp_one=True)
        F0h, F1h = B[k]["F0"] + eF[k], B[k]["F1"] + eF[k + 1]
        A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
        Az = A * s[None, :]
        xm1 = B[k]["x_next"] + ex[k + 1]
        b = (xm1 - B[k]["x0"]).reshape(-1) - y0                     # refute5 convention
        bc = (xm1 - (B[k]["x0"] + ex[k])).reshape(-1) - y0          # fully measured
        Gk, rk, rkc = Az.T @ Az, Az.T @ b, Az.T @ bc
        del A, Az
        torch.cuda.empty_cache()
        Gs = torch.zeros_like(Gk)
        rs, rsc = torch.zeros_like(rk), torch.zeros_like(rk)
        for _ in range(a.K):
            # re-noise the MEASUREMENTS (F at k-1,k,k+1 and x at k-1,k,k+1) and re-derive the state
            eFj = {t: eF[t] + (a.sigma_F / 2.0) * NF(gk) for t in (k - 1, k, k + 1)}
            exj = {t: ex[t] + a.sigma_x * torch.randn(sy.x0.shape, generator=gk,
                                                      device=sy.device, dtype=sy.dtype)
                   for t in (k - 1, k, k + 1)}
            vj, Ccj, _ = SD.derived_state(B, k, dt, eFj, exj)
            SD.install_state(sy, B[k]["snap"], vj, Ccj, Jp_one=True)
            Aj, y0j, _ = assemble_inj(sy, n, lerp(B[k]["F0"] + eFj[k], B[k]["F1"] + eFj[k + 1], n),
                                      None)
            Azj = Aj * s[None, :]
            Gs += Azj.T @ Azj
            rs += Azj.T @ ((xm1 - B[k]["x0"]).reshape(-1) - y0j)
            rsc += Azj.T @ ((xm1 - (B[k]["x0"] + ex[k])).reshape(-1) - y0j)
            del Aj, Azj
            torch.cuda.empty_cache()
        if a.K > 0:
            Gs, rs, rsc = Gs / a.K, rs / a.K, rsc / a.K
        out[f"G{i}"], out[f"r{i}"], out[f"rc{i}"] = (Gk.cpu().numpy(), rk.cpu().numpy(),
                                                    rkc.cpu().numpy())
        out[f"Gm{i}"], out[f"rm{i}"], out[f"rmc{i}"] = (Gs.cpu().numpy(), rs.cpu().numpy(),
                                                       rsc.cpu().numpy())
        R["frames"].append({"tick": k, "y_obs_norm": float((xm1 - B[k]["x0"]).norm())})
        log(f"    frame {i} (tick {k}) derived-state assembly + {a.K} re-noisings "
            f"[{time.time()-t_start:.0f}s]")

    out["theta_true"] = sy.theta_true.double().cpu().numpy()
    out["s"] = s.cpu().numpy()
    np.savez(os.path.join(HERE, f"{a.fittag}.npz"), **out)
    log(f"[fit] wrote {a.fittag}.npz")
    return sy, B, out


# --------------------------------------------------------------------------------------------- #
def solve_stage(a, log, R):
    """round5_solve's stack, unmodified, on the derived-state fits (+ the reference cells)."""
    thetas, rows = {}, []
    files = sorted(glob.glob(os.path.join(HERE, "state_norm_*_der_s*.npz")))
    log(f"[solve] {len(files)} derived-state normal-equation files")
    log(f"    {'fit':<34s} {'conv':<5s} {'solver':<11s} {'medE':>7s} {'p90':>7s} {'neg':>4s} "
        f"{'relL2':>7s} {'medE_re':>8s} {'corr':>6s} {'meanratio':>9s} {'boxtopE':>8s}")
    for fp in files:
        z = np.load(fp)
        name = os.path.basename(fp)[:-4].replace("state_norm_", "")
        th = torch.as_tensor(z["theta_true"], dtype=torch.float64)
        s = torch.as_tensor(z["s"], dtype=torch.float64)
        C = th.numel() // 2
        nfr = sum(1 for k in z.files if k.startswith("G") and not k.startswith("Gm"))
        for conv, rk, rmk in (("b1", "r", "rm"), ("b2", "rc", "rmc")):
            T = min(a.T, nfr)
            G0 = sum(torch.as_tensor(z[f"G{k}"], dtype=torch.float64) for k in range(T))
            r0 = sum(torch.as_tensor(z[f"{rk}{k}"], dtype=torch.float64) for k in range(T))
            Gb = sum(torch.as_tensor(z[f"Gm{k}"], dtype=torch.float64) for k in range(T))
            rb = sum(torch.as_tensor(z[f"{rmk}{k}"], dtype=torch.float64) for k in range(T))
            has_mc = float(Gb.abs().max()) > 0
            Sig = (Gb - G0) if has_mc else torch.zeros_like(G0)
            Gc, rc = G0 - Sig, r0 - (rb - r0 if has_mc else torch.zeros_like(r0))
            out = {"naive": torch.linalg.solve(G0, r0) * s}
            if has_mc:
                out["eiv_snr0"], _ = snr_trunc(G0, Sig, Gc, rc, s, tau=0.0)
            nv = out["naive"]
            mE = float(nv[:C][nv[:C] > 0].median())
            mg = float(nv[C:][nv[C:] > 0].median())
            lo = torch.cat([torch.full((C,), a.lo_f * mE, dtype=torch.float64),
                            torch.full((C,), a.lo_f * mg, dtype=torch.float64)])
            hi = torch.cat([torch.full((C,), a.hi_f * mE, dtype=torch.float64),
                            torch.full((C,), a.hi_f * mg, dtype=torch.float64)])
            out["naive_box"], _ = solve_box(G0, r0, s, lo, hi,
                                            z0=torch.clamp(nv, lo, hi) / s, iters=a.box_iters)
            if has_mc:
                out["eiv_box"], _ = solve_box(Gc, rc, s, lo, hi,
                                              z0=torch.clamp(out["eiv_snr0"], lo, hi) / s,
                                              iters=a.box_iters)
            boxtop = a.hi_f * mE
            n_excl = int((th[:C].numpy() > boxtop).sum())
            for kk, t in out.items():
                p = pstats(t.numpy(), th.numpy(), C)
                p.update({"box_top_E": boxtop, "n_planted_E_above_box_top": n_excl,
                          "planted_max_E": PLANTED_MAX_E, "fit": name, "conv": conv, "solver": kk})
                rows.append(p)
                thetas[f"{name}|{conv}|{kk}"] = t.numpy()
                log(f"    {name:<34s} {conv:<5s} {kk:<11s} {p['med_E']:>7.4f} {p['p90_E']:>7.3f} "
                    f"{p['n_negE']:>4d} {p['rel_l2']:>7.3f} {p['med_E_after_rescale']:>8.4f} "
                    f"{p['corr_E']:>6.3f} {p['mean_ratio_E']:>9.3f} {boxtop:>8.1f}")
        thetas["theta_true"] = th.numpy()

    # ---- the three reference cells of the 2x2, copied in so one npz holds the whole design ----- #
    ZD = np.load(os.path.join(HERE, "state_theta_derive.npz"))
    thetas["theta_true"] = ZD["theta_true"]
    thetas["cleanF_oracleState"] = ZD["oracle_state"]
    thetas["cleanF_derivedState"] = ZD["derived_vC"]
    ZR = np.load(os.path.join(HERE, "theta_refute5.npz"))
    for k in ZR.files:
        if k.startswith("grid48_") and "|T8|" in k and k.split("|")[-1] in ("naive", "eiv_box"):
            thetas["noiseF_oracleState_" + k.replace("_sF0.0039|T8|", "|")] = ZR[k]
    th = torch.as_tensor(ZD["theta_true"], dtype=torch.float64)
    C = th.numel() // 2
    for k, v in thetas.items():
        if k == "theta_true":
            continue
        p = pstats(np.asarray(v, float), th.numpy(), C)
        R.setdefault("pstats", {})[k] = p
    np.savez(os.path.join(HERE, a.out), **thetas)
    R["solve_rows"] = rows
    log(f"[solve] wrote {a.out} ({len(thetas)} vectors)")
    return thetas


# --------------------------------------------------------------------------------------------- #
def score_stage(a, args, log, R):
    """refute5_score.py's scoring, verbatim in structure, plus a DERIVED holdout state."""
    t_start = time.time()
    sy, _ = CT.plant_and_warm(args, log)
    C, W, G, n = sy.C, a.t0, a.window, sy.n_sub_per_frame
    th = sy.theta_true.double()
    dev, f64, dx = th.device, torch.float64, sy.g.dx
    x0, cid = sy.x0.clone(), sy.cid

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
    a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
    keep = np.isfinite(a_ref) & (a_ref > 0)
    snap0 = {k: getattr(sy, k).clone() for k in SNAP}
    R["reference"] = {"max_disp_dx": float(d_ref.norm(dim=-1).max() / dx)}
    log(f"[reference] built [{time.time()-t_start:.0f}s]")

    # ---- the held-out frame, with its two neighbours so the state can be DERIVED there --------- #
    hk = a.holdout_tick
    sy.restore()
    advance(sy, W, hk - 1)
    sy._snapshot(hk - 1)
    HB = {}
    for t in (hk - 1, hk, hk + 1):
        if t > hk - 1:
            sy.restore()
            advance(sy, t - 1, t)
            sy._snapshot(t)
        Fs, _, Xs = record_substeps(sy, n)
        HB[t] = {"x0": sy.x0.clone(), "F0": sy.F0.clone(), "F1": Fs[-1].clone(),
                 "x_next": Xs[-1].clone(),
                 "snap": {k: getattr(sy, k).clone() for k in SNAP}}
    hx0, hF0, hF1, hxn = HB[hk]["x0"], HB[hk]["F0"], HB[hk]["F1"], HB[hk]["x_next"]
    y_obs_h = (hxn - hx0).reshape(-1)
    injh = lerp(hF0, hF1, n)
    dt = sy.dt

    NF = NoiseF(a.noise, x0, a.nodes, sy.device, sy.dtype)
    gh = torch.Generator(device=sy.device).manual_seed(555001)
    eFh = {t: (SIGMA_F / 2.0) * NF(gh) for t in (hk - 1, hk, hk + 1)}
    exh = {t: SIGMA_X * torch.randn(x0.shape, generator=gh, device=sy.device, dtype=sy.dtype)
           for t in (hk - 1, hk, hk + 1)}
    injh_noisy = lerp(hF0 + eFh[hk], hF1 + eFh[hk + 1], n)
    y_obs_h_noisy = ((hxn + exh[hk + 1]) - hx0).reshape(-1)
    v_der_c, C_der_c, _ = SD.derived_state(HB, hk, dt)
    v_der_n, C_der_n, _ = SD.derived_state(HB, hk, dt, eFh, exh)
    R["holdout_state_error"] = {
        "clean": {"v": SD.rel(v_der_c, HB[hk]["snap"]["v0"]),
                  "C": SD.rel(C_der_c, HB[hk]["snap"]["C0"])},
        "noisy": {"v": SD.rel(v_der_n, HB[hk]["snap"]["v0"]),
                  "C": SD.rel(C_der_n, HB[hk]["snap"]["C0"])}}

    def holdout(theta):
        o = {}
        for nm, (vv, CC, inj, yo) in {
                "cleanF_oracleState": (None, None, injh, y_obs_h),
                "cleanF_derivedState": (v_der_c, C_der_c, injh, y_obs_h),
                "noisyF_oracleState": (None, None, injh_noisy, y_obs_h_noisy),
                "noisyF_derivedState": (v_der_n, C_der_n, injh_noisy, y_obs_h_noisy)}.items():
            SD.install_state(sy, HB[hk]["snap"], vv, CC, Jp_one=(vv is not None))
            y = y_of(sy, theta, n, inj, None)
            o[nm] = float((y - yo).norm() / yo.norm())
        return o

    def scored(theta, full_out=True):
        for k, v in snap0.items():
            setattr(sy, k, v.clone())
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
                        "percell": r2_percell(ah, a_ref, keep)})
            del full
        return out

    Z = np.load(os.path.join(HERE, a.out))
    cands = [("theta_true", th)]
    for k in Z.files:
        if k == "theta_true":
            continue
        cands.append((k, torch.as_tensor(Z[k], device=dev, dtype=f64)))
    if a.only:
        want = set(a.only.split(","))
        cands = [c for c in cands if c[0] in want or c[0] == "theta_true"]
    mine = [c for i, c in enumerate(cands)
            if a.nshards == 1 or i % a.nshards == a.shard or c[0] == "theta_true"]
    log(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(cands)}: " + ", ".join(nm for nm, _ in mine))

    R["candidates"] = {}
    log(f"    {'candidate':<40s} {'medE':>7s} {'neg':>4s} {'mnratio':>7s} {'hold_oc':>7s} "
        f"{'hold_dv':>7s} | {'raw':>8s} {'kE':>6s} {'gauged':>8s} {'R2':>8s}")
    for name, theta in mine:
        tc = time.time()
        ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
        hd = holdout(theta)
        raw = scored(theta)

        def probe(lE, lg, theta=theta):
            return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

        gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"])
        kE, kg = gf["k_E"], gf["k_g"]
        gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
            else scored(scale2(theta, kE, kg, C))
        R["candidates"][name] = {
            "param": ps, "holdout": hd, "raw": raw, "gauged": gau,
            "gauge": {k: v for k, v in gf.items() if k != "cells"}, "seconds": time.time() - tc}
        log(f"    {name:<40s} {ps['med_E']:>7.4f} {ps['n_negE']:>4d} "
            f"{ps['mean_ratio_E']:>7.3f} {hd['cleanF_oracleState']:>7.4f} "
            f"{hd['cleanF_derivedState']:>7.4f} | {CT.fmt(raw['loop'],8)} {kE:>6.3f} "
            f"{CT.fmt(gau['loop'],8)} {CT.fmt(gau['R2'],8)}  [{time.time()-tc:.0f}s]")


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="f", choices=("c", "f", "s", "r"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--noise", default="grid", choices=("indep", "grid", "gridsm"))
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--sigma-x", type=float, default=SIGMA_X)
    ap.add_argument("--seed", type=int, default=90210)
    ap.add_argument("--lo-f", type=float, default=0.2)
    ap.add_argument("--hi-f", type=float, default=5.0)
    ap.add_argument("--box-iters", type=int, default=4000)
    ap.add_argument("--out", default="state_theta_combine.npz")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    a.control = a.stage in ("c", "f")
    a.fit = a.stage == "f"
    a.fittag = f"state_norm_{a.noise}{a.nodes}_der_s{a.seed}"
    tag = a.tag or (f"state_combine_{a.stage}"
                    + (f"_s{a.seed}" if a.stage in ("c", "f") else "")
                    + (f"_sh{a.shard}" if a.stage == "r" else ""))

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "args": {k: v for k, v in vars(a).items()}, "tag": tag}
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        if a.stage in ("c", "f"):
            fit_stage(a, args, log, R)
        elif a.stage == "s":
            solve_stage(a, log, R)
        else:
            score_stage(a, args, log, R)
    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

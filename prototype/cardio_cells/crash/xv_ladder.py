"""xv_ladder.py -- ROUND 6, TASK 1.  THE EXTENDED LADDER: which state variables must be injected
per substep, and which of them are REALIZABLE from a microscope?

WHAT IS NEW HERE
====================================================================================================
finject.py's ladder only ever injected F and C.  It never injected x or v.  That is the gap this
script closes, and the reason it matters is an asymmetry in the existing harness:

    F is re-injected at EVERY one of the 10 substeps of a frame.
    v and C are set ONCE, at the frame start, by `System.restore()` / `install_state`.

The whole repair that took med|dE/E| from 0.257 to 0.0078 was "delete the theta-dependence of the
state at every substep so the frame map is linear again".  v never got that treatment, and deriving
v costs 0.0078 -> 0.0227 AT ZERO NOISE, i.e. pure sub-frame truncation.  x never got it either, and
x is the ONE channel that is exactly observed at both frame boundaries.

THE READ-OUT CONFLICT, HANDLED EXPLICITLY
----------------------------------------------------------------------------------------------------
The read-out is y(theta) = x_end - x_0, with x_end the particle position after the LAST substep.
If x is injected at every substep INCLUDING the last, y is a constant: A == 0, b == 0, the fit is
vacuous and every score looks perfect.  Rung `x_all` below is exactly that, run on purpose, so the
trap is measured rather than asserted.

The rule this script adopts instead:

    POST-GATHER variables (x, v, C) are injected at substeps 0 .. n-2 and NOT at substep n-1.
    PRE-SCATTER variables (F) are injected at all n substeps.

This is not a fudge to save the read-out, it is the only consistent rule.  Injection exists to
delete the theta-dependence of a state that is CONSUMED by a later substep.  After the last gather
nothing consumes C or v either -- injecting them there is a provable no-op (asserted below).  x
after the last gather is consumed by exactly one thing, the read-out, so overwriting it there is
not injection, it is deleting the measurement.

Under this rule the theta-dependence of y survives through the last substep only:

    y = (x_inj[n-2] - x_0)  +  dt_sub * v_n(theta)

so ||A|| shrinks by roughly a factor n but is NOT zero.  Column (iv) of the table proves that with
||A||, sigma_min, the theta-carried fraction ||A theta_true|| / ||y_obs||, and a wrong-theta control
(a permuted theta must give a different y).

TWO FLAVOURS
----------------------------------------------------------------------------------------------------
  ORACLE      the reference substep values (finject.record_substeps), plus the true v0, C0 at the
              frame start.  An upper bound, not achievable from a microscope.
  REALIZABLE  built only from observed frame-boundary quantities:
                x  measured at both boundaries          -> x_lerp is EXACT at the boundaries
                F  measured at both boundaries          -> F_lerp
                v  derived, centred difference of x     -> v_lerp
                C  derived, centred (Fdot F^-1)         -> C_lerp
              and the frame-start v0, C0 are the DERIVED ones, because a microscope has no others.
              This is the honest baseline and it is worse than the oracle one before any rung runs.

FOUR MEASUREMENTS PER RUNG
----------------------------------------------------------------------------------------------------
  (i)   affinity defect   rel(A theta_true - b_self, b_self), b_self the injected model's OWN
        response -- is the frame map linear in theta?
  (ii)  recovery          med|dE/E| from fitting the TRUE observed displacement.
  (iii) held-out residual one frame at tick 180 under a FIXED reference protocol (F_lerp, no other
        injection), identical for every rung so the numbers are comparable; reported with an oracle
        state and with a derived state, plus the zero-information band from permuted / mean theta.
  (iv)  read-out carries theta: ||A_scaled||_F, sigma_min, ||b_self||/||y_obs||, wrong-theta delta.

A rung can win (i) and lose (ii).  finject.py's docstring says so and F_true vs F_lerp shows it.

usage:
  PYTHONPATH=/workspace/Plexus/src python xv_ladder.py --device cuda:1
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

from plexus.models.entities import _lame                            # noqa: E402
from assemble import rel as relres                                  # noqa: E402  ||x|| / ||ref||
from recover import Solver, install_E, score, theta_scale           # noqa: E402
import crash_test as CT                                             # noqa: E402
from finject import lerp, hold                                      # noqa: E402
from state_derive import collect, derived_state, install_state      # noqa: E402
from round5_fit import SNAP                                         # noqa: E402


def rel(a, b):
    return float((a - b).norm() / b.norm())


# --------------------------------------------------------------------------------------------- #
#  the reference substep trajectory, now with v as well as F, C, x
# --------------------------------------------------------------------------------------------- #
def record_substeps_xv(sy, n):
    """finject.record_substeps + the post-gather VELOCITY.  Token order identical."""
    H, p = sy.H, sy.p
    sy.restore()
    install_E(sy, sy.E_true)
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + sy.gain_true[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    Fs, Cs, Xs, Vs = [], [], [], []
    for _ in range(n):
        sy._tok("mpm_strain")
        Fs.append(p.F.clone())
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        Cs.append(p.C.clone())
        Xs.append(p.get("pos").clone())
        Vs.append(p.get("vel").clone())
    H.sub_dt = None
    return torch.stack(Fs), torch.stack(Cs), torch.stack(Xs), torch.stack(Vs)


# --------------------------------------------------------------------------------------------- #
#  finject.step_inj, EXTENDED: x and v as well as F and C, and the last-substep rule
# --------------------------------------------------------------------------------------------- #
def step_inj_xv(sy, E_cell, gain_cell, n_sub, inj, inject_last_post=False, readout="disp"):
    """One frame with per-substep state injection.

    inj : dict, any subset of {"F","C","x","v"} -> [n_sub, ...] tensors.
          F is written after mpm_strain (it is consumed by the SAME substep's scatter).
          C, x, v are written after mpm_gather (they are consumed by the NEXT substep) and, unless
          `inject_last_post`, NOT at the final substep -- see the module docstring.
    readout : "disp" -> x_end - x_0 (the observed one)     "vel" -> v_end (an oracle diagnostic)
    """
    H, p = sy.H, sy.p
    sy.restore()
    mu, la = _lame(E_cell[sy.cid])
    p.mu, p.la = mu, la
    H.zero_delta()
    H._delta["mpm_particle"] = sy.pass0 + gain_cell[sy.cid][:, None] * sy.act0
    H.sub_dt = sy.dt_sub
    pa, pb = p.state_schema["pos"]
    va, vb = p.state_schema["vel"]
    iF, iC, ix, iv = (inj.get(k) for k in ("F", "C", "x", "v"))
    last = n_sub - 1
    for s in range(n_sub):
        sy._tok("mpm_strain")
        if iF is not None:
            p.F = iF[s].clone()
        sy._tok("mpm_scatter")
        sy._tok("mpm_grid_update")
        sy._tok("mpm_gather")
        if s < last or inject_last_post:
            if iC is not None:
                p.C = iC[s].clone()
            if ix is not None or iv is not None:
                st = p.state.clone()
                if ix is not None:
                    st[:, pa:pb] = ix[s]
                if iv is not None:
                    st[:, va:vb] = iv[s]
                p.state = st
    H.sub_dt = None
    if readout == "vel":
        return p.get("vel").reshape(-1).clone()
    return (p.get("pos") - sy.x0).reshape(-1).clone()


def y_of_xv(sy, theta, n_sub, inj, **kw):
    E = torch.zeros(sy.C + 1, device=sy.device, dtype=sy.dtype)
    gn = torch.zeros_like(E)
    E[1:], gn[1:] = theta[:sy.C], theta[sy.C:]
    return step_inj_xv(sy, E, gn, n_sub, inj, **kw)


def assemble_xv(sy, n_sub, inj, sE=100.0, sg=1.0, **kw):
    z = torch.zeros(2 * sy.C, device=sy.device, dtype=sy.dtype)
    y0 = y_of_xv(sy, z, n_sub, inj, **kw)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = z.clone()
        e[j] = s
        A[:, j] = (y_of_xv(sy, e, n_sub, inj, **kw) - y0) / s
    torch.cuda.synchronize()
    return A, y0


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="xv_ladder")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--stages", default="pRHl")     # p=preflight R=round3 repro H=holdout l=ladder
    ap.add_argument("--rungs", default="none,F,C,FC,x,v,Fv,FCv,Fx,FCx,FCvx")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=a.n_grid,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "t0": a.t0, "holdout_tick": a.holdout_tick, "stages": a.stages}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 2, a.holdout_tick + 2
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        th = sy.theta_true.double()
        sc_col = theta_scale(C, sy.device)
        k = a.t0
        log(f"[collect] boundaries {t_lo}..{t_hi}, C={C} Np={sy.Np} n_sub={n} dt={dt} "
            f"[{time.time()-t_start:.0f}s]")

        # ---- the frame under test, and its reference substep trajectory --------------------- #
        install_state(sy, B[k]["snap"])                    # true state, exactly as recorded
        Fs, Cs, Xs, Vs = record_substeps_xv(sy, n)
        x0 = B[k]["x0"]
        y_obs = (B[k]["x_next"] - x0).reshape(-1)
        assert float((Xs[-1] - B[k]["x_next"]).abs().max()) == 0.0

        # ---- REALIZABLE series: only measured x, measured F, and things derived from them ---- #
        v_k, C_k, _ = derived_state(B, k, dt)
        v_k1, C_k1, _ = derived_state(B, k + 1, dt)
        F0m, F1m = B[k]["F0"], B[k]["F1"]
        RS = {"F": lerp(F0m, F1m, n), "C": lerp(C_k, C_k1, n),
              "x": lerp(x0, B[k]["x_next"], n), "v": lerp(v_k, v_k1, n)}
        OS = {"F": Fs, "C": Cs, "x": Xs, "v": Vs}
        HS = {"F": hold(F0m, n), "C": hold(C_k, n), "x": hold(x0, n), "v": hold(v_k, n)}
        base_real = {"v": v_k, "C": C_k}                   # frame-start state, realizable flavour

        R["series_error_vs_reference"] = {
            f"{fl}_{ch}": rel(S[ch], OS[ch]) for fl, S in (("lerp", RS), ("hold", HS))
            for ch in ("F", "C", "x", "v")}
        R["boundary_state_error"] = {"v0_derived": rel(v_k, B[k]["v0"]),
                                     "C0_derived": rel(C_k, B[k]["C0"]),
                                     "x0_measured": 0.0, "F0_measured": 0.0}
        log("\n[series] rel error of the injected series against the reference substep values")
        log(f"    lerp:  F {R['series_error_vs_reference']['lerp_F']:.3e}  "
            f"C {R['series_error_vs_reference']['lerp_C']:.3e}  "
            f"x {R['series_error_vs_reference']['lerp_x']:.3e}  "
            f"v {R['series_error_vs_reference']['lerp_v']:.3e}")
        log(f"    hold:  F {R['series_error_vs_reference']['hold_F']:.3e}  "
            f"C {R['series_error_vs_reference']['hold_C']:.3e}  "
            f"x {R['series_error_vs_reference']['hold_x']:.3e}  "
            f"v {R['series_error_vs_reference']['hold_v']:.3e}")
        log(f"    frame-start derived state: v0 {R['boundary_state_error']['v0_derived']:.4f}  "
            f"C0 {R['boundary_state_error']['C0_derived']:.4f}   (x0, F0 are measured exactly)")

        # ---- wrong-theta probes (read-out discrimination) ------------------------------------ #
        gperm = torch.Generator(device=sy.device).manual_seed(1234)
        pE = torch.randperm(C, generator=gperm, device=sy.device)
        pg = torch.randperm(C, generator=gperm, device=sy.device)
        th_perm = torch.cat([th[:C][pE], th[C:][pg]])
        th_mean = torch.cat([th[:C].mean().expand(C), th[C:].mean().expand(C)])
        R["wrong_theta"] = {"perm_rel_l2": rel(th_perm, th), "mean_rel_l2": rel(th_mean, th)}

        # ================================================================== preflight ======== #
        if "p" in a.stages:
            log("\n[p] PREFLIGHT: the two claims the last-substep rule rests on")
            install_state(sy, B[k]["snap"])
            y_ref = y_of_xv(sy, th, n, {})
            y_ctrl = y_of_xv(sy, th, n, {"F": Fs, "C": Cs, "x": Xs, "v": Vs})
            y_ctrl_all = y_of_xv(sy, th, n, {"C": Cs, "v": Vs}, inject_last_post=True)
            y_ctrl_no = y_of_xv(sy, th, n, {"C": Cs, "v": Vs}, inject_last_post=False)
            R["preflight"] = {
                "free_run_matches_reference": float((y_ref - y_obs).abs().max()),
                "oracle_injection_is_noop": float((y_ctrl - y_obs).abs().max()),
                "Cv_at_last_substep_is_noop": float((y_ctrl_all - y_ctrl_no).abs().max())}
            log(f"    free run reproduces the reference displacement to "
                f"{R['preflight']['free_run_matches_reference']:.3e} world")
            log(f"    injecting the ORACLE (F,C,x,v) at 0..n-2 (+F at n-1) is a no-op to "
                f"{R['preflight']['oracle_injection_is_noop']:.3e} world")
            log(f"    injecting C and v at the LAST substep changes nothing: "
                f"{R['preflight']['Cv_at_last_substep_is_noop']:.3e} world  <- the rule is not a "
                f"fudge, post-gather C and v at substep n-1 are provably unused")

        # ================================================== round-3 control reproduction ===== #
        if "R" in a.stages:
            log(f"\n[R] REPRODUCE THE STATED CONTROLS (single frame, tick {k}, ridge0)")
            tgt = {"none (no injection, frame cadence)": 0.2572,
                   "F_lerp, oracle v0 C0": 0.0078,
                   "F_lerp, v derived (centred)": 0.0227,
                   "F_lerp, C derived (centred)": 0.0101,
                   "F_lerp, C derived (forward)": 0.0252}
            _, _, Cf_k = derived_state(B, k, dt)
            ctl = [("none (no injection, frame cadence)", {}, None, None),
                   ("F_lerp, oracle v0 C0", {"F": RS["F"]}, None, None),
                   ("F_lerp, v derived (centred)", {"F": RS["F"]}, v_k, None),
                   ("F_lerp, C derived (centred)", {"F": RS["F"]}, None, C_k),
                   ("F_lerp, C derived (forward)", {"F": RS["F"]}, None, Cf_k),
                   ("F_lerp, v+C derived (centred)", {"F": RS["F"]}, v_k, C_k)]
            R["controls"] = {}
            log(f"    {'control':<34s} {'medE':>8s} {'target':>8s} {'delta':>10s}")
            for nm, inj, vv, CC in ctl:
                install_state(sy, B[k]["snap"], vv, CC, Jp_one=True)
                A, y0 = assemble_xv(sy, n, inj)
                S = Solver(A, C)
                t_hat = S(y_obs - y0)["ridge0"]
                s0 = score(t_hat, th, C)
                R["controls"][nm] = s0
                d = f"{s0['med_E'] - tgt[nm]:+.1e}" if nm in tgt else ""
                log(f"    {nm:<34s} {s0['med_E']:>8.4f} {tgt.get(nm, float('nan')):>8.4f} {d:>10s}")
                S.free(); del A, S
                torch.cuda.empty_cache()

        # ============================================== the held-out acceptance protocol ===== #
        hk = a.holdout_tick
        injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
        yh_obs = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
        vh, Ch, _ = derived_state(B, hk, dt)

        def holdout(theta):
            """FIXED protocol, identical for every rung: F_lerp only, no other injection.
            Returns (oracle-state residual, derived-state residual)."""
            install_state(sy, B[hk]["snap"], None, None, Jp_one=True)
            yo = y_of_xv(sy, theta, n, {"F": injh})
            install_state(sy, B[hk]["snap"], vh, Ch, Jp_one=True)
            yd = y_of_xv(sy, theta, n, {"F": injh})
            return (float((yo - yh_obs).norm() / yh_obs.norm()),
                    float((yd - yh_obs).norm() / yh_obs.norm()))

        if "H" in a.stages:
            log(f"\n[H] HELD-OUT PROTOCOL at tick {hk}, |y_obs| {float(yh_obs.norm()):.4e}: floor "
                f"and zero-information band")
            R["holdout_band"] = {}
            for nm, tt in (("theta_true (floor)", th), ("theta permuted (null)", th_perm),
                           ("theta = block mean (null)", th_mean),
                           ("theta = 0 (null)", torch.zeros_like(th))):
                o, d = holdout(tt)
                R["holdout_band"][nm] = {"oracle_state": o, "derived_state": d}
                log(f"    {nm:<26s} oracleState {o:.5f}   derivedState {d:.5f}")

        # =========================================================== THE EXTENDED LADDER ===== #
        R["ladder"] = {}
        thetas = {}
        HEAD = (f"    {'flavour':<10s} {'rung':<6s} {'affinity':>9s} {'bias':>9s} {'medE':>8s} "
                f"{'medg':>8s} {'ho_orc':>7s} {'ho_der':>7s} {'|A|F':>9s} {'smin':>9s} "
                f"{'carry':>7s} {'wrongth':>8s} {'cond':>9s}")

        def run(flav, rung, series, inject_last_post=False, readout="disp", extra="",
                base=None):
            """base : (v0, C0) to install at the frame start; default = the flavour's own."""
            t_v = time.time()
            with torch.no_grad():
                inj = {ch: series[ch] for ch in rung} if rung != "none" else {}
                if base is not None:
                    install_state(sy, B[k]["snap"], base[0], base[1], Jp_one=True)
                elif flav.startswith("realizable"):
                    install_state(sy, B[k]["snap"], base_real["v"], base_real["C"], Jp_one=True)
                else:
                    install_state(sy, B[k]["snap"], None, None, Jp_one=True)
                st = dict(inject_last_post=inject_last_post, readout=readout)
                A, y0 = assemble_xv(sy, n, inj, **st)
                y_self = y_of_xv(sy, th, n, inj, **st)
                y_wrong = y_of_xv(sy, th_perm, n, inj, **st)
                tgt_y = y_obs if readout == "disp" else None
                b_self = y_self - y0
                denom = float(tgt_y.norm()) if tgt_y is not None else float(y_self.norm())
                Az = A * sc_col[None, :]
                svd = torch.linalg.svdvals(Az)
                dead = float(Az.norm()) <= 1e-13 * float(y_obs.norm())
                d = {"affinity": relres(A @ th - b_self, b_self) if float(b_self.norm()) > 0 else
                     float("nan"),
                     "A_fro_scaled": float(Az.norm()), "sigma_max": float(svd[0]),
                     "sigma_min": float(svd[-1]), "A_is_numerically_zero": dead,
                     "carried_fraction": float(b_self.norm()) / denom,
                     "wrong_theta_delta": float((y_wrong - y_self).norm()) / denom,
                     "y0_over_yobs": float(y0.norm()) / denom}
                if readout == "disp" and not dead:
                    d["model_bias"] = float((y_self - y_obs).norm() / y_obs.norm())
                    b = y_obs - y0
                    S = Solver(A, C)
                    t_hat = S(b)["ridge0"]
                    d["cond"] = S.cond
                    d["fit_residual"] = relres(A @ t_hat - b, b)
                    d["scores"] = score(t_hat, th, C)
                    ho, hd = holdout(t_hat)
                    d["holdout"] = {"oracle_state": ho, "derived_state": hd}
                    thetas[f"{flav}|{rung}{extra}"] = t_hat.clone()
                    S.free(); del S
                elif readout == "disp" and dead:
                    d["model_bias"] = float((y_self - y_obs).norm() / y_obs.norm())
                    d["cond"] = float("inf")
                    d["scores"] = {"med_E": float("nan"), "med_gain": float("nan")}
                    d["holdout"] = {"oracle_state": float("nan"), "derived_state": float("nan")}
                    d["note"] = "A is numerically zero: the read-out no longer carries theta"
                else:
                    d["model_bias"] = float("nan")
                    d["cond"] = float("nan")
                    d["scores"] = {"med_E": float("nan"), "med_gain": float("nan")}
                    d["holdout"] = {"oracle_state": float("nan"), "derived_state": float("nan")}
                d["seconds"] = time.time() - t_v
                R["ladder"][f"{flav}|{rung}{extra}"] = d
                log(f"    {flav:<10s} {rung + extra:<6s} {d['affinity']:>9.2e} "
                    f"{d['model_bias']:>9.2e} {d['scores']['med_E']:>8.4f} "
                    f"{d['scores']['med_gain']:>8.4f} {d['holdout']['oracle_state']:>7.4f} "
                    f"{d['holdout']['derived_state']:>7.4f} {d['A_fro_scaled']:>9.2e} "
                    f"{d['sigma_min']:>9.2e} {d['carried_fraction']:>7.4f} "
                    f"{d['wrong_theta_delta']:>8.2e} {d['cond']:>9.2e}")
                del A, Az
                torch.cuda.empty_cache()
            return d

        if "l" in a.stages:
            RUNGS = [r for r in a.rungs.split(",") if r]
            log(f"\n[l] THE EXTENDED LADDER, frame cadence (n_sub={n}), displacement read-out, "
                f"ZERO measurement noise")
            log(f"    injection at substeps 0..{n-2} for x,v,C (post-gather) and 0..{n-1} for F")
            log(HEAD)
            for rung in RUNGS:
                run("realizable", rung, RS)
            log("")
            for rung in RUNGS:
                run("oracle", rung, OS)

            # ---- THE TRAP, run on purpose ---------------------------------------------------- #
            log(f"\n[l] THE READ-OUT CONFLICT, measured rather than asserted")
            for flav, series in (("realizable", RS), ("oracle", OS)):
                run(flav, "x", series, inject_last_post=True, extra="!")
            run("oracle", "x", OS, inject_last_post=True, readout="vel", extra="!v")
            log(f"    rung 'x!'  = x injected at ALL {n} substeps, displacement read-out.  This is "
                f"the vacuous fit: y is a constant, ||A|| collapses, medE is meaningless.")
            log(f"    rung 'x!v' = the same injection read out on the VELOCITY: ||A|| is back, so "
                f"theta IS still there -- but v is not observed, so this read-out is not realizable.")

            # ---- hold vs lerp for the informative rungs ------------------------------------- #
            log(f"\n[l] hold instead of lerp (one measured frame instead of two)")
            HH = {ch: HS[ch] for ch in HS}
            for rung in ("F", "Fv", "FCv"):
                run("realizable", rung, HH, extra="h")

        # ================================================================ THE v GAP ========== #
        #  The ladder's realizable flavour derives BOTH v0 and C0, so its rungs cannot answer the
        #  one question the brief asks: does injecting v at every substep close 0.0227 -> 0.0078?
        #  Here everything except v is held at the reference, exactly as in the stated control.
        if "V" in a.stages:
            log(f"\n[V] THE v GAP, isolated: F_lerp always injected, C0 ORACLE, only v varies")
            log(f"    target = the substep oracle 0.0078;  control to beat = v derived, set once, "
                f"0.0227")
            log(HEAD)
            v4_k, _, _ = derived_state(B, k, dt, stencil="c4")
            v4_k1, _, _ = derived_state(B, k + 1, dt, stencil="c4")
            RS4 = {**RS, "v": lerp(v4_k, v4_k1, n)}
            RSh = {**RS, "v": HS["v"]}                       # F_lerp with v HELD, not lerped
            OSv = {**RS, "v": OS["v"]}                       # F_lerp with the ORACLE substep v
            OC = B[k]["C0"]
            V = [("v0=true,       no v injection  (= the 0.0078 target)", "F", RS, (None, None)),
                 ("v0=true,       v_oracle injected (upper bound)", "Fv", OSv, (None, None)),
                 ("v0=derived c2, no v injection  (= the 0.0227 control)", "F", RS, (v_k, OC)),
                 ("v0=derived c2, v_lerp c2 injected  <- THE TEST", "Fv", RS, (v_k, OC)),
                 ("v0=derived c2, v_hold c2 injected", "Fv", RSh, (v_k, OC)),
                 ("v0=derived c4, no v injection", "F", RS4, (v4_k, OC)),
                 ("v0=derived c4, v_lerp c4 injected", "Fv", RS4, (v4_k, OC)),
                 ("v0=true,       v_lerp c2 injected", "Fv", RS, (None, OC))]
            R["v_gap"] = {}
            for i, (nm, rung, series, base) in enumerate(V):
                d = run("vgap", rung, series, extra=f"{i}", base=base)
                R["v_gap"][nm] = d
                log(f"      ^ {nm}")

        # ============================================ WHAT THE LOST SIGNAL COSTS UNDER NOISE = #
        #  Injecting x deletes 9/10 of the frame's motion from the read-out but does NOT delete
        #  the measurement noise on x_next.  At zero noise that only costs bias amplification;
        #  the claim that it is far worse with a real microscope is measured here rather than
        #  asserted.  sigma_x is the recording's own (round5_fit.SIGMA_X).
        if "N" in a.stages:
            from round5_fit import SIGMA_X
            log(f"\n[N] the SAME rungs with noise on the observed x_next only, "
                f"sigma_x = {SIGMA_X:.3e} world ({SIGMA_X/sy.g.dx:.4f} dx), 3 seeds")
            log(f"    {'flavour':<10s} {'rung':<5s} {'carry':>7s} {'medE(0)':>8s} {'medE(sx)':>9s} "
                f"{'ratio':>7s}")
            R["noise_probe"] = {}
            for flav, rung, series, base in (("realizable", "F", RS, None),
                                             ("realizable", "Fx", RS, None),
                                             ("realizable", "FCvx", RS, None),
                                             ("oracle", "F", OS, None),
                                             ("oracle", "Fx", OS, None)):
                inj = {ch: series[ch] for ch in rung}
                if flav == "realizable":
                    install_state(sy, B[k]["snap"], base_real["v"], base_real["C"], Jp_one=True)
                else:
                    install_state(sy, B[k]["snap"], None, None, Jp_one=True)
                A, y0 = assemble_xv(sy, n, inj)
                S = Solver(A, C)
                carry = float((y_of_xv(sy, th, n, inj) - y0).norm() / y_obs.norm())
                e0 = score(S(y_obs - y0)["ridge0"], th, C)["med_E"]
                gen = torch.Generator(device=sy.device).manual_seed(31)
                es = []
                for _ in range(3):
                    xn = B[k]["x_next"] + SIGMA_X * torch.randn(
                        B[k]["x_next"].shape, generator=gen, device=sy.device, dtype=sy.dtype)
                    es.append(score(S((xn - x0).reshape(-1) - y0)["ridge0"], th, C)["med_E"])
                es = float(np.mean(es))
                R["noise_probe"][f"{flav}|{rung}"] = {"carry": carry, "med_E_clean": e0,
                                                      "med_E_sigma_x": es, "sigma_x": SIGMA_X}
                log(f"    {flav:<10s} {rung:<5s} {carry:>7.4f} {e0:>8.4f} {es:>9.4f} "
                    f"{es/max(e0,1e-12):>7.2f}")
                S.free(); del A, S
                torch.cuda.empty_cache()

        if "l" in a.stages or "V" in a.stages:
            np.savez(os.path.join(HERE, f"xv_theta_{a.tag}.npz"),
                     theta_true=th.cpu().numpy(),
                     **{kk.replace("|", "__").replace(" ", "_").replace(",", ""): v.cpu().numpy()
                        for kk, v in thetas.items()})

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""mcv_v.py -- ROUND 6, TASK B.  MODEL-CORRECTED v:  use the SCHEME, not a better stencil.

THE IDEA
====================================================================================================
The assembly needs the particle velocity v at the START of a frame.  Only x is observed, at frame
cadence.  A centred difference costs med|dE/E| 0.0078 (oracle v) -> 0.0227 at ZERO measurement
noise, and that gap is pure SUB-FRAME TRUNCATION: one frame is n = 10 substeps and

    x_{k+1} = x_k + dt_sub * v_{k+1}                    (mpm_gather, unconditional)

so over one frame

    X1 - X0 = dt_sub * sum_{j=1..n} v_j
            = dt_frame * v_0  +  dt_sub * sum_{i=1..n} (n - i + 1) (v_i - v_{i-1})       (*)

The second term is NOT noise -- the v_i after the first are produced BY THE MODEL from v_0 and
theta, so it is computable.  Writing

    pred(v_0, theta) := the model's one-frame displacement from (v_0, theta)
    corr(v_0, theta) := pred(v_0, theta) / dt_frame  -  v_0            [ = the second term of (*) ]

route (a) is one application of

    v_0  <-  (X1 - X0)/dt_frame  -  corr(v_0, theta)                                     (A)
          =  v_0 + (obs_disp - pred(v_0, theta)) / dt_frame                              (A')

((A) and (A') are the same line of algebra; (A') is how it is coded, one model evaluation, no
per-substep bookkeeping.)  Route (b) alternates (A') with the theta least-squares.

Note what (A') is: a quasi-Newton shooting step with J ~ dt_frame * I.  That is the honest name for
it, and it exposes the structural risk this script is built to measure -- v_0 has 2*Np degrees of
freedom and one frame of displacement has 2*Np observations, so v_0 alone can explain the ENTIRE
observation.  If the shooting is iterated to convergence the theta least-squares has nothing left to
explain and returns the theta it started from.  Whether route (a)'s single step falls into that trap
is an empirical question about the contraction factor ||I - J/dt_frame||, and it is measured here.

THE DRAG LEAK (found here, reported whether or not it is convenient)
----------------------------------------------------------------------------------------------------
`material_cardio_cells.yaml` has `{op: drag, k: 30}` in the OUTER schedule, so the frozen body-force
delta `sy.pass0` is  -30 * v_TRUE  evaluated at the frame boundary.  `state_derive.install_state`
overwrites the state velocity but NOT pass0, so every "derived v" number published so far still
reads the simulator's true v through the drag body force.  This script measures that leak and runs
every route in a `honest_drag` mode that recomputes the outer schedule from the INSTALLED v, which
is the only mode in which the corruption proof (stage x) can pass.

STAGES
    p  probe: size of the drag leak; the honest recomputation reproduces pass0 at v = v_true
    c  controls: oracle v 0.007777 / centred-difference v 0.022707 / centred-difference C 0.010084
    a  route (a), one-step correction
    b  route (b), block alternation, per-iteration trajectory
    s  shooting DIAGNOSTIC at oracle theta -- is the sub-frame structure model-predictable at all?
    n  noise check on the winner at the recording's sigma_x
    x  corruption proof: destroy the stored true v; the answer must not move

usage:
  PYTHONPATH=/workspace/Plexus/src python mcv_v.py --device cuda:1 --stages pcabsnx
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

from finject import lerp, y_of                                        # noqa: E402
from refute_round3 import fit                                         # noqa: E402
from round5_fit import SIGMA_X                                        # noqa: E402
from state_derive import collect, install_state, rel, derived_v       # noqa: E402

DRAG_K = 30.0          # material_cardio_cells.yaml: {op: drag, k: 30.0, emit: mpm_acceleration}


# --------------------------------------------------------------------------------------------- #
def install2(sy, snap, tick, v=None, Cm=None, honest_drag=False):
    """state_derive.install_state, plus the fix for the drag leak.

    honest_drag: after the derived v is written into the state, RE-RUN the outer schedule so the
    velocity-dependent body force (drag) is recomputed from the v we actually installed.  Nothing
    from the true snapshot's pass0/act0 survives.  With v=None (oracle) this is a no-op by
    construction, which is checked in stage p.
    """
    install_state(sy, snap, v, Cm, Jp_one=True)
    if honest_drag and v is not None:
        sy.restore()                                   # p.state <- state0 (holding the derived v)
        act, pas = sy._outer(int(tick), gain_cell=None)
        sy.act0, sy.pass0 = act.clone(), pas.clone()


def predict(sy, snap, tick, v, Cm, theta, n, injF, honest):
    """One-frame displacement [2Np] the MODEL produces from (v, C, theta) with the measured F."""
    install2(sy, snap, tick, v, Cm, honest)
    return y_of(sy, theta, n, injF, None)


def shoot(sy, snap, tick, v, Cm, theta, n, injF, y_obs, dt, honest, steps=1, v_true=None):
    """Quasi-Newton shooting on v, J ~ dt_frame * I.  Equation (A')."""
    v = v.clone()
    hist = []
    for _ in range(int(steps)):
        pred = predict(sy, snap, tick, v, Cm, theta, n, injF, honest)
        r = y_obs - pred
        hist.append({"shoot_resid": float(r.norm() / y_obs.norm()),
                     "relv_in": (None if v_true is None else rel(v, v_true))})
        v = v + r.reshape(v.shape) / dt
    if v_true is not None:
        hist.append({"shoot_resid": None, "relv_in": rel(v, v_true)})
    return v, hist


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="mcv_v")
    ap.add_argument("--stages", default="pcabsnx")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    a = ap.parse_args()
    if any(st in a.stages for st in "ab") and "c" not in a.stages:
        raise SystemExit("stages a/b need the control fits of stage c; add 'c'")

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "stages": a.stages, "t0": a.t0, "holdout_tick": a.holdout_tick,
         "sigma_x": SIGMA_X, "drag_k": DRAG_K}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - 2, a.holdout_tick + 2
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        th = sy.theta_true.double()
        k0, hk = a.t0, a.holdout_tick
        log(f"[collect] {t_lo}..{t_hi}  C={C} Np={sy.Np} n_sub={n} dt={dt} "
            f"[{time.time()-t_start:.0f}s]")

        # ---- the frame under fit, and the held-out frame ------------------------------------- #
        inj0 = lerp(B[k0]["F0"], B[k0]["F1"], n)
        injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
        y_obs0 = (B[k0]["x_next"] - B[k0]["x0"]).reshape(-1)
        y_obsh = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
        v_true0, v_trueh = B[k0]["v0"], B[hk]["v0"]

        def C_cd(k, ex=None, eF=None):
            def F(t):
                return B[t]["F0"] + (0.0 if eF is None else eF[t])
            return ((F(k + 1) - F(k - 1)) / (2 * dt)) @ torch.linalg.inv(F(k))

        v_cd0, v_cdh = derived_v(B, k0, dt), derived_v(B, hk, dt)
        v_fwd0 = (B[k0]["x_next"] - B[k0]["x0"]) / dt
        C_cd0, C_cdh = C_cd(k0), C_cd(hk)
        R["derivation"] = {"relv_c2_t0": rel(v_cd0, v_true0), "relv_fwd_t0": rel(v_fwd0, v_true0),
                           "relC_c2_t0": rel(C_cd0, B[k0]["C0"]),
                           "relv_c2_holdout": rel(v_cdh, v_trueh)}
        log(f"[derivation] tick {k0}: rel v c2 {R['derivation']['relv_c2_t0']:.5f}  "
            f"fwd {R['derivation']['relv_fwd_t0']:.5f}  rel C c2 {R['derivation']['relC_c2_t0']:.5f}")

        # ------------------------------------------------------------------ stage p ---------- #
        if "p" in a.stages:
            log("\n[p] THE DRAG LEAK -- is pass0 a hidden v_true oracle?")
            snap = B[k0]["snap"]
            install2(sy, snap, k0, v_true0, None, honest_drag=True)
            d_true = float((sy.pass0 - snap["pass0"]).abs().max())
            a_true = float((sy.act0 - snap["act0"]).abs().max())
            install2(sy, snap, k0, v_cd0, None, honest_drag=True)
            d_cd = float((sy.pass0 - snap["pass0"]).abs().max())
            pred_leak = float((DRAG_K * (v_true0 - v_cd0)).abs().max())
            drag_form = float((snap["pass0"] + DRAG_K * v_true0).abs().max())
            R["drag_probe"] = {
                "max_abs_pass0_recomputed_minus_snapshot_at_v_true": d_true,
                "max_abs_act0_recomputed_minus_snapshot_at_v_true": a_true,
                "max_abs_pass0_change_at_v_c2": d_cd,
                "max_abs_(-k v_true) - pass0": drag_form,
                "max_abs_k*(v_true-v_c2)": pred_leak,
                "norm_pass0": float(snap["pass0"].norm()),
                "rel_pass0_change_at_v_c2": float((sy.pass0 - snap["pass0"]).norm()
                                                  / snap["pass0"].norm())}
            log(f"    pass0 == -k*v_true to {drag_form:.2e}  (|pass0| {snap['pass0'].norm():.4e})")
            log(f"    recomputing the outer schedule at v = v_TRUE reproduces pass0 to {d_true:.2e}"
                f" and act0 to {a_true:.2e}   <- the recomputation is a no-op on the oracle")
            log(f"    at v = v_c2 the body force moves by {d_cd:.3e} "
                f"(rel {R['drag_probe']['rel_pass0_change_at_v_c2']:.4f}); "
                f"k*(v_true-v_c2) = {pred_leak:.3e}")
            log("    => every published 'derived v' number leaks v_true through the drag force.")

        # ---- the held-out one-frame residual ------------------------------------------------- #
        def holdout(theta, mode, honest=False, theta_shoot=None):
            """mode: 'oracle' (true state), 'c2' (centred-difference state), 'shot' (route's own
            shooting AT the held-out frame -- circular, reported only with that label)."""
            if mode == "oracle":
                v, Cm = None, None
            elif mode == "c2":
                v, Cm = v_cdh, C_cdh
            elif mode == "shot":
                v, _ = shoot(sy, B[hk]["snap"], hk, v_cdh, C_cdh,
                             theta if theta_shoot is None else theta_shoot, n, injh, y_obsh, dt,
                             honest, steps=1)
                Cm = C_cdh
            install2(sy, B[hk]["snap"], hk, v, Cm, honest)
            y = y_of(sy, theta, n, injh, None)
            return float((y - y_obsh).norm() / y_obsh.norm())

        R["holdout_floor"] = {"theta_true_oracle_state": holdout(th, "oracle"),
                              "theta_true_c2_state": holdout(th, "c2")}
        log(f"\n[holdout] tick {hk}, |y_obs| {float(y_obsh.norm()):.4e};  floor at theta_true: "
            f"oracle state {R['holdout_floor']['theta_true_oracle_state']:.5f}, "
            f"c2 state {R['holdout_floor']['theta_true_c2_state']:.5f}")

        # ------------------------------------------------------------------ stage c ---------- #
        # the ladder, both drag conventions.  `snap`  = pass0 frozen from the true-v snapshot (the
        # convention of round 3 / state_derive, kept so 0.007777 / 0.022707 reproduce exactly);
        # `honest` = pass0 recomputed from the installed v (no v_true anywhere).
        CTRL = {"oracle_v_oracle_C": 0.007777332098339839,
                "v_c2_oracle_C": 0.022707019926523922,
                "oracle_v_C_c2": 0.010084}
        fits, R["controls"] = {}, {}

        def do_fit(name, v, Cm, honest, tick=k0, injF=None, y_obs_pos=None, store=None):
            install2(sy, B[tick]["snap"], tick, v, Cm, honest)
            sc, t_hat = fit(sy, n, inj0 if injF is None else injF,
                            B[tick]["x_next"] if y_obs_pos is None else y_obs_pos,
                            B[tick]["x0"], th, C)
            (store if store is not None else fits)[name] = t_hat
            return sc, t_hat

        if "c" in a.stages:
            log(f"\n[c] CONTROLS, single frame tick {k0}, clean F, ridge0, Jp=1")
            log(f"    {'variant':<28s} {'drag':<7s} {'med|dE/E|':>10s} {'p90':>8s} {'relL2':>8s} "
                f"{'published':>10s} {'d':>10s}")
            ladder = [("oracle_v_oracle_C", None, None, False),
                      ("v_c2_oracle_C", v_cd0, None, False),
                      ("v_c2_oracle_C", v_cd0, None, True),
                      ("oracle_v_C_c2", None, C_cd0, False),
                      ("v_c2_C_c2", v_cd0, C_cd0, False),
                      ("v_c2_C_c2", v_cd0, C_cd0, True)]
            for nm, v, Cm, honest in ladder:
                key = f"{nm}|{'honest' if honest else 'snap'}"
                sc, t_h = do_fit(key, v, Cm, honest)
                R["controls"][key] = sc
                pub = CTRL.get(nm, float("nan"))
                d = "" if nm not in CTRL or honest else f"{sc['med_E']-pub:+.2e}"
                log(f"    {nm:<28s} {'honest' if honest else 'snap':<7s} {sc['med_E']:>10.5f} "
                    f"{sc['p90_E']:>8.4f} {sc['rel_l2']:>8.4f} {pub:>10.6f} {d:>10s} "
                    f"[{time.time()-t_start:.0f}s]")
                if "H" in a.stages:      # the acceptance statistic for the controls too
                    ho, hc = holdout(t_h, "oracle"), holdout(t_h, "c2")
                    sc["holdout_oracle_state"], sc["holdout_c2_state"] = ho, hc
                    log(f"        held-out one-frame residual: oracle state {ho:.5f}  "
                        f"c2 state {hc:.5f}")
            R["controls_reproduced"] = bool(
                abs(R["controls"]["oracle_v_oracle_C|snap"]["med_E"] - CTRL["oracle_v_oracle_C"]) < 1e-9
                and abs(R["controls"]["v_c2_oracle_C|snap"]["med_E"] - CTRL["v_c2_oracle_C"]) < 1e-9)
            log(f"    round-3 controls reproduced to <1e-9: {R['controls_reproduced']}")

        def report(name, t_hat, sc, honest, extra=None):
            row = {"pstats": sc,
                   "holdout_oracle_state": holdout(t_hat, "oracle"),
                   "holdout_c2_state": holdout(t_hat, "c2"),
                   "holdout_shot_state_CIRCULAR": holdout(t_hat, "shot", honest)}
            if extra:
                row.update(extra)
            log(f"    {name:<30s} medE {sc['med_E']:>9.5f}  p90 {sc['p90_E']:>7.4f}  "
                f"relL2 {sc['rel_l2']:>7.4f}  ho(oracle) {row['holdout_oracle_state']:>7.5f}  "
                f"ho(c2) {row['holdout_c2_state']:>7.5f}  ho(shot) "
                f"{row['holdout_shot_state_CIRCULAR']:>7.5f} [{time.time()-t_start:.0f}s]")
            return row

        # ------------------------------------------------------------------ stage a ---------- #
        if "a" in a.stages:
            log("\n[a] ROUTE (a): ONE-STEP MODEL CORRECTION,  v <- (X1-X0)/dt - corr(v_c2, theta_0)")
            R["route_a"] = {}
            for honest in (False, True):
                tag = "honest" if honest else "snap"
                th0 = fits[f"v_c2_oracle_C|{tag}"]
                v1, hist = shoot(sy, B[k0]["snap"], k0, v_cd0, None, th0, n, inj0, y_obs0, dt,
                                 honest, steps=1, v_true=v_true0)
                corr = (v1 - v_fwd0)
                sc, t1 = do_fit(f"route_a|{tag}", v1, None, honest)
                extra = {"shoot_hist": hist, "relv_after": rel(v1, v_true0),
                         "relv_before": rel(v_cd0, v_true0),
                         "corr_norm_over_v": float(corr.norm() / v_true0.norm()),
                         "true_trunc_norm_over_v": float((v_fwd0 - v_true0).norm()
                                                         / v_true0.norm()),
                         "theta_move_rel": float((t1 - th0).norm() / th0.norm())}
                R["route_a"][tag] = report(f"route_a ({tag}, C oracle)", t1, sc, honest, extra)
                log(f"        rel v {extra['relv_before']:.5f} -> {extra['relv_after']:.5f} ; "
                    f"|corr|/|v| {extra['corr_norm_over_v']:.5f} vs true truncation "
                    f"{extra['true_trunc_norm_over_v']:.5f} ; theta moved "
                    f"{extra['theta_move_rel']:.2e}")
            # the fully derived version (C also from the centred difference)
            th0 = fits["v_c2_C_c2|honest"]
            v1, hist = shoot(sy, B[k0]["snap"], k0, v_cd0, C_cd0, th0, n, inj0, y_obs0, dt,
                             True, steps=1, v_true=v_true0)
            sc, t1 = do_fit("route_a_Cc2|honest", v1, C_cd0, True)
            R["route_a"]["honest_C_c2"] = report("route_a (honest, C c2)", t1, sc, True,
                                                 {"shoot_hist": hist,
                                                  "relv_after": rel(v1, v_true0)})

        # ------------------------------------------------------------------ stage b ---------- #
        if "b" in a.stages:
            log(f"\n[b] ROUTE (b): BLOCK ALTERNATION, {a.iters} iterations "
                f"(theta <- LS | v <- shooting), C oracle, honest drag")
            R["route_b"] = {}
            for nsteps in (1, 2):
                key = f"newton{nsteps}"
                th_c = fits["v_c2_oracle_C|honest"]
                v_c = v_cd0.clone()
                traj = [{"iter": 0, "relv": rel(v_c, v_true0),
                         "med_E": R["controls"]["v_c2_oracle_C|honest"]["med_E"],
                         "shoot_resid": None, "theta_move": None,
                         "holdout_oracle_state": holdout(th_c, "oracle")}]
                log(f"    --- {nsteps} Newton step(s) per iteration ---")
                log(f"    {'iter':>4s} {'shootResid':>11s} {'relv':>9s} {'med|dE/E|':>10s} "
                    f"{'thetaMove':>10s} {'ho(oracle)':>11s} {'ho(c2)':>8s}")
                log(f"    {0:>4d} {'-':>11s} {traj[0]['relv']:>9.5f} "
                    f"{traj[0]['med_E']:>10.5f} {'-':>10s} "
                    f"{traj[0]['holdout_oracle_state']:>11.5f} "
                    f"{holdout(th_c, 'c2'):>8.5f}")
                for it in range(1, a.iters + 1):
                    v_new, hist = shoot(sy, B[k0]["snap"], k0, v_c, None, th_c, n, inj0, y_obs0,
                                        dt, True, steps=nsteps, v_true=v_true0)
                    sc, t_new = do_fit(f"route_b|{key}|it{it}", v_new, None, True)
                    row = {"iter": it, "shoot_resid": hist[0]["shoot_resid"],
                           "shoot_resid_last": hist[-2]["shoot_resid"] if nsteps > 1
                           else hist[0]["shoot_resid"],
                           "relv": rel(v_new, v_true0), "med_E": sc["med_E"],
                           "p90_E": sc["p90_E"], "rel_l2": sc["rel_l2"],
                           "theta_move": float((t_new - th_c).norm() / th_c.norm()),
                           "holdout_oracle_state": holdout(t_new, "oracle"),
                           "holdout_c2_state": holdout(t_new, "c2")}
                    traj.append(row)
                    log(f"    {it:>4d} {row['shoot_resid']:>11.5f} {row['relv']:>9.5f} "
                        f"{row['med_E']:>10.5f} {row['theta_move']:>10.2e} "
                        f"{row['holdout_oracle_state']:>11.5f} {row['holdout_c2_state']:>8.5f} "
                        f"[{time.time()-t_start:.0f}s]")
                    v_c, th_c = v_new, t_new
                R["route_b"][key] = traj
                fits[f"route_b|{key}|final"] = th_c

        # ------------------------------------------------------------------ stage s ---------- #
        # Is the sub-frame velocity structure model-predictable AT ALL?  Shoot with theta_TRUE and
        # the oracle C: this is the ceiling of routes (a)/(b), an oracle DIAGNOSTIC, not a result.
        if "s" in a.stages:
            log("\n[s] DIAGNOSTIC (oracle theta, oracle C): how far can shooting move v?")
            log(f"    {'step':>4s} {'shootResid':>11s} {'relv_in':>9s}")
            v_s, hist = shoot(sy, B[k0]["snap"], k0, v_cd0, None, th, n, inj0, y_obs0, dt, True,
                              steps=6, v_true=v_true0)
            for i, h in enumerate(hist):
                sr = "-" if h["shoot_resid"] is None else f"{h['shoot_resid']:.5f}"
                log(f"    {i:>4d} {sr:>11s} {h['relv_in']:>9.5f}")
            sc_s, t_s = do_fit("shoot_oracle_theta", v_s, None, True)
            R["stage_s"] = {"hist": hist, "converged_relv": rel(v_s, v_true0),
                            "fit_at_converged_v": sc_s,
                            "model_error_at_true_state": None}
            R["stage_s"]["fit_row"] = report("fit at shot-v (oracle theta)", t_s, sc_s, True)
            # how exact is the model at the TRUE state?  this bounds what shooting can recover.
            pred_t = predict(sy, B[k0]["snap"], k0, None, None, th, n, inj0, False)
            R["stage_s"]["model_error_at_true_state"] = float((pred_t - y_obs0).norm()
                                                              / y_obs0.norm())
            pred_tc = predict(sy, B[k0]["snap"], k0, v_true0, C_cd0, th, n, inj0, True)
            R["stage_s"]["model_error_true_v_C_c2"] = float((pred_tc - y_obs0).norm()
                                                            / y_obs0.norm())
            log(f"    model error at the TRUE state (injF lerp is the only approximation): "
                f"{R['stage_s']['model_error_at_true_state']:.5f}  <- the shooting floor")
            log(f"    model error at true v + centred-difference C: "
                f"{R['stage_s']['model_error_true_v_C_c2']:.5f}")

        # ------------------------------------------------------------------ stage n ---------- #
        if "n" in a.stages:
            log(f"\n[n] NOISE, sigma_x = {SIGMA_X:.3e} on every measured position used by the "
                f"derivation AND on the observation x_next (the state's own x0 stays clean, the "
                f"convention of rounds 3-6)")
            log(f"    {'variant':<24s} {'seed':>7s} {'relv':>8s} {'med|dE/E|':>10s} "
                f"{'ho(oracle)':>11s}")
            R["noise"] = []
            for seed in (90210, 555):
                g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                ex = {t: SIGMA_X * torch.randn(B[t]["x0"].shape, generator=g, device=sy.device,
                                               dtype=sy.dtype) for t in B}
                v_cd_n = derived_v(B, k0, dt, ex)
                xn_n = B[k0]["x_next"] + ex[k0 + 1]
                y_obs_n = (xn_n - B[k0]["x0"]).reshape(-1)
                # baseline: centred difference
                sc_b, t_b = do_fit(f"noise_c2_{seed}", v_cd_n, None, True, y_obs_pos=xn_n)
                rows = [("c2 (baseline)", rel(v_cd_n, v_true0), sc_b, t_b)]
                # route (a): one model-informed step off the SAME noisy centred difference
                v_a, _ = shoot(sy, B[k0]["snap"], k0, v_cd_n, None, t_b, n, inj0, y_obs_n, dt,
                               True, steps=1, v_true=v_true0)
                sc_a, t_a = do_fit(f"noise_routea_{seed}", v_a, None, True, y_obs_pos=xn_n)
                rows.append(("route (a)", rel(v_a, v_true0), sc_a, t_a))
                # route (b): 2 alternations
                v_c, th_c = v_a, t_a
                for _ in range(2):
                    v_c, _ = shoot(sy, B[k0]["snap"], k0, v_c, None, th_c, n, inj0, y_obs_n, dt,
                                   True, steps=1)
                    sc_c, th_c = do_fit(f"noise_routeb_{seed}", v_c, None, True, y_obs_pos=xn_n)
                rows.append(("route (b) x3", rel(v_c, v_true0), sc_c, th_c))
                for nm, rv, sc, t_h in rows:
                    ho = holdout(t_h, "oracle")
                    R["noise"].append({"variant": nm, "seed": seed, "relv": rv,
                                       "med_E": sc["med_E"], "p90_E": sc["p90_E"],
                                       "rel_l2": sc["rel_l2"], "holdout_oracle_state": ho})
                    log(f"    {nm:<24s} {seed:>7d} {rv:>8.5f} {sc['med_E']:>10.5f} {ho:>11.5f} "
                        f"[{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage x ---------- #
        if "x" in a.stages:
            log("\n[x] CORRUPTION PROOF: destroy every stored copy of the simulator's true v, "
                "then re-run route (a).  honest drag must be immune; snap drag must not be.")
            ref_h = R.get("route_a", {}).get("honest", {}).get("pstats", {}).get("med_E")
            ref_s = R.get("route_a", {}).get("snap", {}).get("pstats", {}).get("med_E")
            gcor = torch.Generator(device=sy.device).manual_seed(31337)
            va, vb = sy.p.state_schema["vel"]
            for t in B:
                junk = 1e3 * torch.randn(B[t]["v0"].shape, generator=gcor, device=sy.device,
                                         dtype=sy.dtype)
                # v is stored in THREE places: B[t]['v0'], the vel slice of the snapshot state, and
                # -- because drag is in the outer schedule -- the frozen body force pass0.  A
                # corruption that misses the third one is not a corruption of v, so pass0 is moved
                # to the value it would have had if the true v had been `junk` (stage p verified
                # pass0 == -k v_true to fp).
                B[t]["snap"]["pass0"] = B[t]["snap"]["pass0"] + DRAG_K * (B[t]["v0"] - junk)
                B[t]["v0"] = junk
                st = B[t]["snap"]["state0"].clone()
                st[:, va:vb] = junk
                B[t]["snap"]["state0"] = st
                B[t]["snap"]["v0"] = junk.clone()
            v_cd_x = derived_v(B, k0, dt)
            log(f"    positions untouched: max|v_c2(corrupted store) - v_c2| "
                f"{float((v_cd_x - v_cd0).abs().max()):.2e}")
            out = {}
            for honest in (True, False):
                tag = "honest" if honest else "snap"
                # exactly the stage-a recipe, recomputed on the corrupted store
                install2(sy, B[k0]["snap"], k0, v_cd_x, None, honest)
                sc0, th0 = fit(sy, n, inj0, B[k0]["x_next"], B[k0]["x0"], th, C)
                v1, _ = shoot(sy, B[k0]["snap"], k0, v_cd_x, None, th0, n, inj0, y_obs0, dt,
                              honest, steps=1)
                install2(sy, B[k0]["snap"], k0, v1, None, honest)
                sc1, _ = fit(sy, n, inj0, B[k0]["x_next"], B[k0]["x0"], th, C)
                ref = ref_h if honest else ref_s
                out[tag] = {"med_E_corrupted": sc1["med_E"], "med_E_clean": ref,
                            "baseline_med_E_corrupted": sc0["med_E"],
                            "delta": None if ref is None else sc1["med_E"] - ref}
                log(f"    route_a {tag:<7s} med|dE/E| corrupted {sc1['med_E']:.6f}  clean "
                    f"{('%.6f' % ref) if ref is not None else 'n/a':>8s}  delta "
                    f"{('%+.2e' % out[tag]['delta']) if out[tag]['delta'] is not None else 'n/a'}")
            R["corruption"] = out

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

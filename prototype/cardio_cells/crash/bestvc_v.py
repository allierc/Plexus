"""bestvc_v.py -- ROUND 6, TASK A.  THE BEST v FROM POSITIONS ALONE.

WHY
====================================================================================================
state_derive.py removed the last state oracle and measured what deriving (v, C) costs at zero noise:

    derived nothing (oracle)                med|dE/E| 0.0078
    C only, centred difference                        0.0101      <- nearly free
    v only, centred difference                        0.0227      <- THE BOTTLENECK
    both, centred                                     0.0218
    both, round 3's FORWARD C                         0.0404

So v is the expensive one.  v_k = (x_{k+1} - x_{k-1})/(2 dt) is O(dt^2) truncation on a signal that
is smooth at this cadence, and state_derive's stage h already showed the truncation itself is easy
to beat (rel err vs the simulator's v: c2 0.01291 -> c4 0.00736) -- and that beating it MADE THE
ANSWER WORSE (stage 1 med|dE/E| 0.02271 -> 0.02697).  That contradiction is the subject here.

WHAT THIS SCRIPT CONSUMES
----------------------------------------------------------------------------------------------------
EVERY estimator below is a linear functional of the MEASURED POSITION TRACK ONLY,

    v_k = ( sum_t  w_t  x_t ) / dt ,          w supported on the collected tick range,

so `W` is an [n_targets, T_all] matrix and every estimator -- finite-difference stencil,
Savitzky-Golay, interpolating spline, Fourier fit, smoother-then-difference -- is one of these.
Nothing else is read.  C is left ORACLE everywhere in this file so that v is isolated (task A);
Jp is set to 1 (state_derive measured max|Jp-1| = 0 exactly).

WHAT IS MEASURED, per estimator
----------------------------------------------------------------------------------------------------
  r : (i)  rel err of v against the simulator's true v0, clean, at every fit tick and the holdout;
      plus the ANALYTIC noise gain ||w||_2 / dt, which is the whole noisy story for a linear rule;
      plus the empirical noisy rel err over a sigma_x grid and 3 seeds.
  a : (ii) end-to-end med|dE/E| -- single frame at t0, ridge0, clean F, C ORACLE, v DERIVED, i.e.
      exactly round 3's ladder rung `v_*_only`, whose c2 value 0.022707 is reproduced as a control.
      (iii) the held-out one-frame residual at tick 180 (round 5's acceptance statistic, floor
      0.00474 at theta_true with an oracle state), with the holdout frame's own v ALSO derived.
  e : the same end-to-end number in round 5's real configuration: T=8 stacked frames, naive solve.
  n : the noise sweep -- top estimators re-run with sigma_x on the positions the derivative is taken
      from, over a grid bracketing the recording's 0.0409 px = 2.00e-5 world units.

usage:
  PYTHONPATH=/workspace/Plexus/src python bestvc_v.py --device cuda:0 --stages rae
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

from recover import theta_scale                                   # noqa: E402
from finject import lerp, assemble_inj, y_of                      # noqa: E402
from refute_round3 import fit                                     # noqa: E402
from round5_fit import SIGMA_X                                    # noqa: E402
from round5_solve import pstats                                   # noqa: E402
from state_derive import collect, install_state, rel, derived_v   # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  every estimator is a weight matrix over the collected position track
# --------------------------------------------------------------------------------------------- #
def sg_W(T_all, idx, half, order, shift=0.0):
    """Savitzky-Golay / local polynomial: least-squares fit of degree `order` to the 2*half+1
    centred samples, derivative read at offset `shift` frames from the centre.
    order = 2*half recovers the exact interpolating stencil (c2, c4, c6, c8...)."""
    j = np.arange(-half, half + 1, dtype=float)
    V = j[:, None] ** np.arange(order + 1)[None, :]
    q = np.arange(order + 1, dtype=float)
    dc = np.where(q >= 1, q * np.power(float(shift), np.maximum(q - 1, 0)), 0.0)
    c = dc @ np.linalg.pinv(V)
    W = np.zeros((len(idx), T_all))
    for r, i in enumerate(idx):
        W[r, i - half:i + half + 1] = c
    return W


def onesided_W(T_all, idx, which):
    W = np.zeros((len(idx), T_all))
    for r, i in enumerate(idx):
        if which == "back1":
            W[r, i], W[r, i - 1] = 1.0, -1.0
        elif which == "fwd1":
            W[r, i + 1], W[r, i] = 1.0, -1.0
        elif which == "back2":                       # 2nd-order backward, causal
            W[r, i], W[r, i - 1], W[r, i - 2] = 1.5, -2.0, 0.5
        elif which == "back3":
            W[r, i], W[r, i - 1], W[r, i - 2], W[r, i - 3] = 11 / 6, -3.0, 1.5, -1 / 3
    return W


def fourier_W(T_all, idx, tt, period, M, trend=0, ridge=0.0):
    """Least-squares trigonometric fit with the pacemaker period, differentiated exactly.
    `trend` extra polynomial columns absorb the fact that 165 frames of warm-up is 1.1 beats and
    the sheet is not yet exactly on its limit cycle."""
    tau = tt - tt.mean()
    cols, dcols = [np.ones_like(tau)], [np.zeros_like(tau)]
    for d in range(1, trend + 1):
        cols.append(tau ** d)
        dcols.append(d * tau ** (d - 1))
    w0 = 2 * np.pi / period
    for m in range(1, M + 1):
        cols += [np.cos(w0 * m * tau), np.sin(w0 * m * tau)]
        dcols += [-w0 * m * np.sin(w0 * m * tau), w0 * m * np.cos(w0 * m * tau)]
    D, dD = np.stack(cols, 1), np.stack(dcols, 1)
    G = D.T @ D + ridge * np.eye(D.shape[1])
    return dD[idx] @ np.linalg.solve(G, D.T)


def spline_W(T_all, idx, tt, k=3, bc="not-a-knot"):
    from scipy.interpolate import CubicSpline, make_interp_spline
    I = np.eye(T_all)
    if k == 3:
        sp = CubicSpline(tt, I, axis=0, bc_type=bc)
    else:
        sp = make_interp_spline(tt, I, k=k, axis=0)
    return sp.derivative()(tt[idx])


def smooth_then_W(T_all, base_W, lam, order=2):
    """Whittaker-Henderson smoother S = (I + lam D^T D)^-1 applied to the track first, then any
    linear derivative rule.  A tunable denoiser for the noisy regime; lam = 0 is a no-op."""
    if lam <= 0:
        return base_W
    D = np.zeros((T_all - order, T_all))
    stencil = np.array([1.0, -2.0, 1.0]) if order == 2 else np.array([-1.0, 1.0])
    for i in range(T_all - order):
        D[i, i:i + order + 1] = stencil
    S = np.linalg.solve(np.eye(T_all) + lam * (D.T @ D), np.eye(T_all))
    return base_W @ S


def build_estimators(T_all, idx, tt, period):
    """name -> (W, one-line statement of what it consumes)."""
    E = {}
    E["back1"] = (onesided_W(T_all, idx, "back1"), "x_{k-1},x_k  (round 3 control)")
    E["fwd1"] = (onesided_W(T_all, idx, "fwd1"), "x_k,x_{k+1}")
    E["back2"] = (onesided_W(T_all, idx, "back2"), "3 past frames, causal O(dt^2)")
    E["c2"] = (sg_W(T_all, idx, 1, 2), "x_{k+-1}          BASELINE")
    E["c4"] = (sg_W(T_all, idx, 2, 4), "x_{k+-2}   O(dt^4)")
    E["c6"] = (sg_W(T_all, idx, 3, 6), "x_{k+-3}   O(dt^6)")
    E["c8"] = (sg_W(T_all, idx, 4, 8), "x_{k+-4}   O(dt^8)")
    E["c10"] = (sg_W(T_all, idx, 5, 10), "x_{k+-5}   O(dt^10)")
    for (h, o) in [(2, 1), (3, 1), (3, 3), (4, 1), (4, 3), (4, 5), (5, 3), (5, 5),
                   (6, 3), (6, 5), (7, 3), (8, 5), (10, 5)]:
        E[f"sg{2*h+1}p{o}"] = (sg_W(T_all, idx, h, o), f"{2*h+1} frames, poly deg {o}")
    E["spline3"] = (spline_W(T_all, idx, tt, 3), "whole track, not-a-knot cubic spline")
    E["spline5"] = (spline_W(T_all, idx, tt, 5), "whole track, quintic spline")
    for M in (3, 5, 8, 12, 16):
        E[f"fourier{M}"] = (fourier_W(T_all, idx, tt, period, M),
                            f"whole track, {M} harmonics of the {period}-frame beat")
    for M in (5, 8, 12):
        E[f"fourier{M}+t2"] = (fourier_W(T_all, idx, tt, period, M, trend=2),
                               f"whole track, {M} harmonics + quadratic trend")
    for lam in (1.0, 10.0, 100.0):
        E[f"wh{lam:g}+c4"] = (smooth_then_W(T_all, E["c4"][0], lam),
                              f"whole track, 2nd-diff smoother lam={lam:g}, then c4")
    for d in (-0.5, -0.25, 0.25, 0.5):
        E[f"c4@{d:+g}"] = (sg_W(T_all, idx, 2, 4, shift=d),
                           f"x_{{k+-2}}, derivative read {d:+g} frames off the boundary")
    return E


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="bestvc_v")
    ap.add_argument("--stages", default="rae")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--pad", type=int, default=45)
    ap.add_argument("--period", type=float, default=150.0)
    ap.add_argument("--ntop", type=int, default=4)
    ap.add_argument("--epick", default="", help="comma list overriding stage e's automatic pick")
    ap.add_argument("--npick", default="", help="comma list overriding stage n's automatic pick")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "cli": vars(a), "sigma_x": SIGMA_X}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        t_lo, t_hi = a.t0 - a.pad, a.holdout_tick + a.pad
        sy, B = collect(args, t_lo, t_hi, log)
        C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
        s = theta_scale(C, sy.device)
        th = sy.theta_true.double()
        tt = np.arange(t_lo, t_hi + 1, dtype=float)
        T_all = len(tt)
        Xs = torch.stack([B[int(t)]["x0"] for t in tt])            # [T_all, Np, 2] MEASURED
        V0 = torch.stack([B[int(t)]["v0"] for t in tt])            # the simulator's true v
        fit_ticks = list(range(a.t0, a.t0 + a.T))
        targets = fit_ticks + [a.holdout_tick]
        idx = [int(t - t_lo) for t in targets]
        log(f"[collect] ticks {t_lo}..{t_hi} ({T_all}); targets {targets}; dt={dt} "
            f"n_sub={n} [{time.time()-t_start:.0f}s]")

        EST = build_estimators(T_all, idx, tt, a.period)
        Wt = {k: torch.as_tensor(v[0], device=sy.device, dtype=sy.dtype) for k, v in EST.items()}

        def vhat(name, Xuse):
            return torch.einsum("rt,tnd->rnd", Wt[name], Xuse) / dt

        # ---- CONTROL: the weight formulation reproduces state_derive.derived_v exactly --------- #
        ctrl = {}
        for nm, st in (("c2", "c2"), ("c4", "c4"), ("sg5p1", "sg5")):
            vv = vhat(nm, Xs)[0]
            ctrl[nm] = float((vv - derived_v(B, a.t0, dt, None, st)).abs().max())
        R["control_matches_state_derive"] = ctrl
        log(f"[control] max|W-form - state_derive.derived_v| at tick {a.t0}: "
            + ", ".join(f"{k} {v:.2e}" for k, v in ctrl.items()))

        # ------------------------------------------------------------------ stage r ----------- #
        rel_clean, rows = {}, []
        if "r" in a.stages:
            log("\n[r] v accuracy from positions alone.  rel err = |v_hat - v_true| / |v_true|")
            log(f"    {'estimator':<14s} {'span':>5s} {'|w|/dt':>9s} {'clean165':>9s} "
                f"{'cleanFIT':>9s} {'clean180':>9s} {'0.5sx':>8s} {'1sx':>8s} {'2sx':>8s} "
                f"{'4sx':>8s} {'8sx':>8s}  consumes")
            sig_grid = [0.5, 1.0, 2.0, 4.0, 8.0]
            noise = {}
            for f in sig_grid:
                for seed in (90210, 555, 777):
                    g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                    noise[(f, seed)] = Xs + (f * SIGMA_X) * torch.randn(
                        Xs.shape, generator=g, device=sy.device, dtype=sy.dtype)
            for nm, (Wnp, doc) in EST.items():
                vv = vhat(nm, Xs)
                ce = [rel(vv[r], V0[i]) for r, i in enumerate(idx)]
                row = {"estimator": nm, "consumes": doc,
                       "span_frames": int(np.count_nonzero(np.abs(Wnp).sum(0) > 1e-14)),
                       "noise_gain_per_dt": float(np.linalg.norm(Wnp[0]) / dt),
                       "clean_t0": ce[0], "clean_fit_mean": float(np.mean(ce[:a.T])),
                       "clean_holdout": ce[-1],
                       "clean_per_tick": {int(t): c for t, c in zip(targets, ce)}}
                for f in sig_grid:
                    ee = []
                    for seed in (90210, 555, 777):
                        vn = vhat(nm, noise[(f, seed)])
                        ee += [rel(vn[r], V0[i]) for r, i in enumerate(idx[:a.T])]
                    row[f"noisy_{f:g}sx"] = float(np.mean(ee))
                rows.append(row)
                rel_clean[nm] = row["clean_fit_mean"]
                log(f"    {nm:<14s} {row['span_frames']:>5d} {row['noise_gain_per_dt']:>9.1f} "
                    f"{row['clean_t0']:>9.5f} {row['clean_fit_mean']:>9.5f} "
                    f"{row['clean_holdout']:>9.5f} "
                    + " ".join(f"{row[f'noisy_{f:g}sx']:>8.5f}" for f in sig_grid)
                    + f"  {doc}")
            R["stage_r"] = rows

        # ------------------------------------------------------------------ stage a ----------- #
        #  single frame at t0, clean F, ridge0, C ORACLE, v DERIVED == round 3's `v_only` rung
        injh = lerp(B[a.holdout_tick]["F0"], B[a.holdout_tick]["F1"], n)
        y_obs_h = (B[a.holdout_tick]["x_next"] - B[a.holdout_tick]["x0"]).reshape(-1)

        def holdout(theta, vh):
            install_state(sy, B[a.holdout_tick]["snap"], vh, None, Jp_one=True)
            y = y_of(sy, theta, n, injh, None)
            return float((y - y_obs_h).norm() / y_obs_h.norm())

        def one_frame(vk, k=None, x_next=None):
            k = a.t0 if k is None else k
            install_state(sy, B[k]["snap"], vk, None, Jp_one=True)
            return fit(sy, n, lerp(B[k]["F0"], B[k]["F1"], n),
                       B[k]["x_next"] if x_next is None else x_next, B[k]["x0"], th, C)

        if "a" in a.stages:
            log(f"\n[a] end-to-end, SINGLE frame tick {a.t0}, clean F, ridge0, C ORACLE, v derived")
            log(f"    (round 3 control: oracle v 0.007777 / c2 0.022707;  held-out floor 0.00474)")
            log(f"    {'estimator':<14s} {'med|dE/E|':>10s} {'p90':>8s} {'relL2':>8s} "
                f"{'holdout':>8s} {'ho_ostate':>10s} {'relv':>8s}")
            R["stage_a"] = {}
            order = ["ORACLE_v"] + list(EST)
            for nm in order:
                vk = None if nm == "ORACLE_v" else vhat(nm, Xs)[0]
                sc, t_hat = one_frame(vk)
                vh = None if nm == "ORACLE_v" else vhat(nm, Xs)[-1]
                ho = holdout(t_hat, vh)
                hoo = holdout(t_hat, None)
                R["stage_a"][nm] = {**sc, "holdout_derived_v": ho, "holdout_oracle_state": hoo,
                                    "rel_v": rel_clean.get(nm, 0.0)}
                log(f"    {nm:<14s} {sc['med_E']:>10.5f} {sc['p90_E']:>8.4f} "
                    f"{sc['rel_l2']:>8.4f} {ho:>8.5f} {hoo:>10.5f} "
                    f"{rel_clean.get(nm, 0.0):>8.5f}")
            R["stage_a"]["theta_true_holdout_oracle_state"] = holdout(th, None)
            log(f"    theta_true, oracle state, held-out residual: "
                f"{R['stage_a']['theta_true_holdout_oracle_state']:.5f}  "
                f"[{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage e ----------- #
        def stacked(name, Xuse=None, ticks=None):
            """T stacked frames, naive normal-equation solve, C oracle, v from `name`."""
            Xuse = Xs if Xuse is None else Xuse
            ticks = fit_ticks if ticks is None else ticks
            vv = None if name == "ORACLE_v" else vhat(name, Xuse)
            G0 = torch.zeros(2 * C, 2 * C, device=sy.device, dtype=sy.dtype)
            r0 = torch.zeros(2 * C, device=sy.device, dtype=sy.dtype)
            for r, k in enumerate(ticks):
                install_state(sy, B[k]["snap"], None if vv is None else vv[r], None, Jp_one=True)
                A, y0, _ = assemble_inj(sy, n, lerp(B[k]["F0"], B[k]["F1"], n), None)
                Az = A * s[None, :]
                b = (B[k]["x_next"] - B[k]["x0"]).reshape(-1) - y0
                G0 += Az.T @ Az
                r0 += Az.T @ b
                del A, Az
                torch.cuda.empty_cache()
            t_hat = torch.linalg.solve(G0, r0) * s
            return t_hat, pstats(t_hat.cpu().numpy(), th.cpu().numpy(), C)

        if "e" in a.stages:
            if "a" in a.stages:
                cand = sorted((k for k in EST), key=lambda k: R["stage_a"][k]["med_E"])
                pick = ["ORACLE_v", "c2"] + [k for k in cand[:a.ntop] if k != "c2"]
            else:
                pick = ["ORACLE_v", "c2", "c4", "spline3", "fourier8"]
            if a.epick:
                pick = a.epick.split(",")
            log(f"\n[e] end-to-end, T={a.T} STACKED frames, clean F, naive solve, C ORACLE")
            log(f"    (round 5 control: oracle state 0.008562)")
            log(f"    {'estimator':<14s} {'med|dE/E|':>10s} {'p90':>8s} {'relL2':>8s} "
                f"{'corr':>7s} {'meanrat':>8s} {'holdout':>8s}")
            R["stage_e"] = {}
            for nm in pick:
                t_hat, ps = stacked(nm)
                vh = None if nm == "ORACLE_v" else vhat(nm, Xs)[-1]
                ho = holdout(t_hat, vh)
                R["stage_e"][nm] = {**ps, "holdout_derived_v": ho}
                log(f"    {nm:<14s} {ps['med_E']:>10.5f} {ps['p90_E']:>8.4f} "
                    f"{ps['rel_l2']:>8.4f} {ps['corr_E']:>7.4f} {ps['mean_ratio_E']:>8.4f} "
                    f"{ho:>8.5f} [{time.time()-t_start:.0f}s]")
            R["stage_e_picked"] = pick

        # ------------------------------------------------------------------ stage m ----------- #
        #  the noise sweep in round 5's REAL configuration: T=8 stacked, naive solve.
        if "m" in a.stages:
            mp = (a.epick or "c2,c4,sg7p3,sg9p3").split(",")
            log(f"\n[m] T={a.T} STACKED, C oracle, sigma_x on the differentiated track only")
            log(f"    {'estimator':<14s} {'sx':>6s} {'seed':>7s} {'med|dE/E|':>10s} {'p90':>8s} "
                f"{'relL2':>8s} {'holdout':>8s}")
            R["stage_m"] = []
            for f, seed in [(1.0, 90210), (4.0, 90210)]:
                g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                Xn = Xs + (f * SIGMA_X) * torch.randn(Xs.shape, generator=g,
                                                      device=sy.device, dtype=sy.dtype)
                for nm in mp:
                    t_hat, ps = stacked(nm, Xn)
                    ho = holdout(t_hat, vhat(nm, Xn)[-1])
                    R["stage_m"].append({"estimator": nm, "sigma_x_units": f, "seed": seed,
                                         **ps, "holdout_derived_v": ho})
                    log(f"    {nm:<14s} {f:>6g} {seed:>7d} {ps['med_E']:>10.5f} "
                        f"{ps['p90_E']:>8.4f} {ps['rel_l2']:>8.4f} {ho:>8.5f} "
                        f"[{time.time()-t_start:.0f}s]")

        # ------------------------------------------------------------------ stage n ----------- #
        if "n" in a.stages:
            # the clean winner AND the estimators the stage-r table says take over under noise
            noisy_cands = ["sg7p3", "sg9p3", "sg11p3", "sg5p1"]
            if "a" in a.stages:
                cand = sorted((k for k in EST), key=lambda k: R["stage_a"][k]["med_E"])
                pick = ["c2"] + [k for k in cand[:a.ntop] if k != "c2"]
            else:
                pick = ["c2", "c4", "spline3"]
            pick += [k for k in noisy_cands if k not in pick]
            if a.npick:
                pick = a.npick.split(",")
            log(f"\n[n] NOISE on the positions the derivative is taken from "
                f"(sigma_x = 1 unit = {SIGMA_X:.3e} world = 0.0409 px).  Single frame, "
                f"tick {a.t0}, C oracle.")
            log(f"    x0 / x_next entering the fit stay clean, so this isolates differentiation "
                f"noise amplification.")
            log(f"    {'estimator':<14s} {'sx':>6s} {'seed':>7s} {'relv':>8s} "
                f"{'med|dE/E|':>10s} {'holdout':>8s}")
            R["stage_n"] = []
            for f in (0.0, 0.5, 1.0, 2.0, 4.0):
                for seed in (90210, 555):
                    if f == 0.0 and seed != 90210:
                        continue
                    g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                    Xn = Xs + (f * SIGMA_X) * torch.randn(Xs.shape, generator=g,
                                                          device=sy.device, dtype=sy.dtype)
                    for nm in pick:
                        vv = vhat(nm, Xn)
                        sc, t_hat = one_frame(vv[0])
                        ho = holdout(t_hat, vv[-1])
                        rv = float(np.mean([rel(vv[r], V0[i]) for r, i in enumerate(idx[:a.T])]))
                        R["stage_n"].append({"estimator": nm, "sigma_x_units": f, "seed": seed,
                                             "rel_v": rv, "med_E": sc["med_E"],
                                             "p90_E": sc["p90_E"], "rel_l2": sc["rel_l2"],
                                             "holdout_derived_v": ho})
                        log(f"    {nm:<14s} {f:>6g} {seed:>7d} {rv:>8.5f} "
                            f"{sc['med_E']:>10.5f} {ho:>8.5f} [{time.time()-t_start:.0f}s]")
            R["stage_n_picked"] = pick

        # ------------------------------------------------------------------ stage c ----------- #
        #  CONTROL for stage n: there the observation x_next feeding b stayed clean so that only
        #  differentiation noise was in play.  Here the SAME draw also corrupts x_next, which is
        #  what a recording actually does, so the two effects can be separated.
        if "c" in a.stages:
            log(f"\n[c] control: the same sigma_x ALSO on x_next (the observation in b)")
            log(f"    {'estimator':<14s} {'sx':>6s} {'seed':>7s} {'med|dE/E|':>10s} "
                f"{'vs stage n':>11s}")
            R["stage_c"] = []
            base = {(d["estimator"], d["sigma_x_units"], d["seed"]): d["med_E"]
                    for d in R.get("stage_n", [])}
            for f in (1.0, 4.0):
                for seed in (90210, 555):
                    g = torch.Generator(device=sy.device).manual_seed(seed + 7)
                    Xn = Xs + (f * SIGMA_X) * torch.randn(Xs.shape, generator=g,
                                                          device=sy.device, dtype=sy.dtype)
                    xn_next = B[a.t0]["x_next"] + (Xn - Xs)[int(a.t0 + 1 - t_lo)]
                    for nm in R.get("stage_n_picked", ["c2"])[:3]:
                        sc, _ = one_frame(vhat(nm, Xn)[0], x_next=xn_next)
                        d = base.get((nm, f, seed))
                        R["stage_c"].append({"estimator": nm, "sigma_x_units": f, "seed": seed,
                                             "med_E": sc["med_E"], "stage_n_med_E": d})
                        log(f"    {nm:<14s} {f:>6g} {seed:>7d} {sc['med_E']:>10.5f} "
                            f"{(sc['med_E'] - d) if d else float('nan'):>+11.5f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

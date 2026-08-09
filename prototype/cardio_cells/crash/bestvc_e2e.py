"""bestvc_e2e.py -- ROUND 6, TASK C.  THE WINNERS TOGETHER, AGAINST REALIZABLE F NOISE.

WHAT THIS IS
====================================================================================================
Task A found the best v from the measured position track (c4 clean, sg7p3/sg9p5 at the recording's
sigma_x).  Task B found the best C from measured quantities (t_c2 = Fdot F^-1 with a centred Fdot).
This runs the 2x2 that the project turns on:

                      | oracle state | best DERIVED state (nothing from the answer key)
    clean F           |   0.008562   |     ?
    realizable F noise|    ~0.15     |     ?

T = 8 stacked frames, ticks 165..172, the round-5 configuration throughout.  "realizable F noise" is
refute5_fit's `--noise grid --nodes 48` draw (a cell of the recording carries ~23 independent F
measurements, not 100) at sigma_F = 3.9e-3, plus sigma_x = 0.0409 px = 2.00e-5 world on positions.

WHAT "DERIVED" CONSUMES -- and what is left in the assembly
----------------------------------------------------------------------------------------------------
    v_k  = (sum_t w_t x_t) / dt        a linear rule on the MEASURED position track (task A)
    C_k  = ((F_{k+1} - F_{k-1})/2dt) @ inv(F_k)      on the MEASURED F  (task B's t_c2)
    Jp_k = 1                           (state_derive measured max|Jp - 1| = 0 exactly)

Under noise BOTH are derived from the NOISY measurements: the same per-frame-boundary F error field
that is injected across the substeps also enters the C stencil, and the same position noise that
corrupts the observation also enters the v stencil.  There is no state oracle anywhere in the
assembly.  What remains simulator-side is (i) the particle positions x0 used as the assembler's
geometry -- round 5's convention, treated as exact, whose cost is measured in stage `x` -- and (ii)
the pacemaker drive act0/pass0, which is the known stimulus, not state.

REPORTED PER CELL
----------------------------------------------------------------------------------------------------
  med|dE/E| ; held-out one-frame residual at tick 180 (the ACCEPTANCE statistic, floor 0.00474) ;
  gauged loopscore from a free 150-frame rollout, margin 20, anchor=None, discovery_cardio_mpm/
  metrics.py imported unmodified ; number of negative moduli ; mean(E_hat)/mean(E).
The naive solve is reported beside the box/EIV solve, and the zero-information null bank is scored
in the same protocol (round 5 measured its gauged loopscore band as 0.26-0.68).

STAGES
----------------------------------------------------------------------------------------------------
  f  fit every cell of the 2x2 (+ the decomposition cells v-only / C-only) -> bestvc_e2e_theta.npz
  x  control: the cost of the ONE remaining idealization, exact particle positions in the assembler
  s  score: full crash test with the deterministic 5x5 grid gauge + 2 Broyden steps
  p  the figure, bestvc_e2e.png

usage:
  PYTHONPATH=/workspace/Plexus/src python bestvc_e2e.py --device cuda:0 --stages f
  PYTHONPATH=/workspace/Plexus/src python bestvc_e2e.py --device cuda:0 --stages s
  PYTHONPATH=/workspace/Plexus/src python bestvc_e2e.py --stages p --device cpu
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

from recover import theta_scale, install_E                        # noqa: E402
from finject import lerp, assemble_inj, y_of                      # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                     # noqa: E402
from round5_solve import pstats, solve_box, snr_trunc             # noqa: E402
from refute5_fit import NoiseF                                    # noqa: E402
from state_derive import collect, install_state, rel              # noqa: E402
from bestvc_v import sg_W                                         # noqa: E402
from bestvc_C import Fdot, Fm                                     # noqa: E402

# task A's rules, as (half-width, polynomial degree).  half=deg/2 -> the exact centred stencil.
V_RULES = {"c2": (1, 2), "c4": (2, 4), "sg7p3": (3, 3), "sg9p5": (4, 5)}


# --------------------------------------------------------------------------------------------- #
def install_state_x(sy, snap, v, C, dx=None, Jp_one=True):
    """install_state, plus the option to perturb the assembler's own particle positions by `dx`.

    Round 5 (and every round before it) treats x0 as exact: the noise goes on the OBSERVATION
    x_next only.  Stage `x` uses dx to measure what that idealization is worth.
    """
    install_state(sy, snap, v, C, Jp_one=Jp_one)
    if dx is None:
        return
    pa, pb = sy.p.state_schema["pos"]
    st = sy.state0.clone()
    st[:, pa:pb] = st[:, pa:pb] + dx
    sy.state0 = st
    sy.x0 = sy.x0 + dx


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="bestvc_e2e")
    ap.add_argument("--stages", default="f")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--pad", type=int, default=6)
    ap.add_argument("--K", type=int, default=6, help="EIV Monte-Carlo re-noisings per frame")
    ap.add_argument("--nodes", type=int, default=48, help="refute5 grid noise nodes per unit side")
    ap.add_argument("--seeds", default="90210,555")
    ap.add_argument("--vclean", default="c4", help="task A's clean winner")
    ap.add_argument("--vnoisy", default="sg7p3", help="task A's winner at the recording's sigma_x")
    ap.add_argument("--valt", default="c4", help="second v rule to run in the noisy cell")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--theta-in", default="bestvc_e2e_theta.npz")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.T, a.K, a.pad = 2, 1, 6
        a.seeds = "90210"

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "cli": vars(a), "sigma_F": SIGMA_F, "sigma_x": SIGMA_X}
    t_start = time.time()
    torch.manual_seed(0)
    seeds = [int(x) for x in a.seeds.split(",") if x]

    # =========================================================================== FIT =========== #
    if "f" in a.stages or "x" in a.stages:
        with torch.no_grad():
            t_lo, t_hi = a.t0 - a.pad, a.holdout_tick + a.pad
            sy, B = collect(args, t_lo, t_hi, log)
            C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
            s = theta_scale(C, sy.device)
            th = sy.theta_true.double()
            tt = np.arange(t_lo, t_hi + 1, dtype=float)
            T_all = len(tt)
            Xs = torch.stack([B[int(t)]["x0"] for t in tt])
            V0 = torch.stack([B[int(t)]["v0"] for t in tt])
            fit_ticks = list(range(a.t0, a.t0 + a.T))
            targets = fit_ticks + [a.holdout_tick]
            idx = [int(t - t_lo) for t in targets]
            hk = a.holdout_tick
            log(f"[collect] ticks {t_lo}..{t_hi} ({T_all}); fit {fit_ticks}; holdout {hk}; "
                f"dt={dt} n_sub={n} [{time.time()-t_start:.0f}s]")

            Wt = {k: torch.as_tensor(sg_W(T_all, idx, h, d), device=sy.device, dtype=sy.dtype)
                  for k, (h, d) in V_RULES.items()}

            def vhat(name, Xuse):
                """v at every target tick, from the measured position track alone."""
                return torch.einsum("rt,tnd->rnd", Wt[name], Xuse) / dt

            # ---- the noise draws.  ONE F error per frame boundary, shared by every consumer of
            #      that boundary (the injection AND the C stencil).  ONE position error per frame,
            #      shared by the v stencil AND the observation x_next(k) = x0(k+1).
            def draws(noisy, seed):
                if not noisy:
                    z = {t: torch.zeros_like(B[t]["F0"]) for t in B}
                    zx = {t: torch.zeros_like(B[t]["x0"]) for t in B}
                    return z, zx, None
                NF = NoiseF("grid", sy.x0, a.nodes, sy.device, sy.dtype)
                gm = torch.Generator(device=sy.device).manual_seed(seed)
                eb = {t: (SIGMA_F / 2.0) * NF(gm) for t in sorted(B)}
                ex = {t: SIGMA_X * torch.randn(B[t]["x0"].shape, generator=gm,
                                               device=sy.device, dtype=sy.dtype) for t in sorted(B)}
                return eb, ex, NF

            def Xnoisy(ex):
                return Xs + torch.stack([ex[int(t)] for t in tt])

            def state_of(kind, vname, k, eb, ex):
                """(v, C) at tick k.  kind 'oracle' -> (None, None) i.e. the simulator's own."""
                if kind == "oracle":
                    return None, None
                vv = None
                if kind in ("derived", "v_only"):
                    r = targets.index(k)
                    vv = vhat(vname, Xnoisy(ex))[r]
                CC = None
                if kind in ("derived", "C_only"):
                    CC = Fdot(B, k, dt, eb, "c2") @ torch.linalg.inv(Fm(B, k, eb))
                return vv, CC

            # ------------------------------------------------------------------------------------
            def fit_cell(kind, vname, noisy, seed, K, dxnoise=False):
                """T stacked frames -> every solver of round 5.  Returns (thetas, extras)."""
                eb, ex, NF = draws(noisy, seed)
                gk = torch.Generator(device=sy.device).manual_seed(31337 + seed)
                G0 = torch.zeros(2 * C, 2 * C, device=sy.device, dtype=sy.dtype)
                r0 = torch.zeros(2 * C, device=sy.device, dtype=sy.dtype)
                Gs, rs = torch.zeros_like(G0), torch.zeros_like(r0)
                relv, relC = [], []
                for k in fit_ticks:
                    vv, CC = state_of(kind, vname, k, eb, ex)
                    if vv is not None:
                        relv.append(rel(vv, B[k]["v0"]))
                    if CC is not None:
                        relC.append(rel(CC, B[k]["C0"]))
                    F0h, F1h = B[k]["F0"] + eb[k], B[k]["F1"] + eb[k + 1]
                    yk = (B[k]["x_next"] + ex[k + 1] - B[k]["x0"] - ex[k]).reshape(-1) \
                        if dxnoise else (B[k]["x_next"] + ex[k + 1] - B[k]["x0"]).reshape(-1)
                    dxk = ex[k] if dxnoise else None
                    install_state_x(sy, B[k]["snap"], vv, CC, dxk)
                    A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
                    Az = A * s[None, :]
                    G0 += Az.T @ Az
                    r0 += Az.T @ (yk - y0)
                    del A, Az
                    torch.cuda.empty_cache()
                    for _ in range(K):
                        e0, e1 = (SIGMA_F / 2.0) * NF(gk), (SIGMA_F / 2.0) * NF(gk)
                        install_state_x(sy, B[k]["snap"], vv, CC, dxk)
                        Aj, y0j, _ = assemble_inj(sy, n, lerp(F0h + e0, F1h + e1, n), None)
                        Azj = Aj * s[None, :]
                        Gs += Azj.T @ Azj
                        rs += Azj.T @ (yk - y0j)
                        del Aj, Azj
                        torch.cuda.empty_cache()
                G0c, r0c = G0.cpu().double(), r0.cpu().double()
                sc_ = s.cpu().double()
                out = {"naive": torch.linalg.solve(G0c, r0c) * sc_}
                if K > 0:
                    Gbc, rbc = (Gs / K).cpu().double(), (rs / K).cpu().double()
                    Sig = Gbc - G0c
                    Gc, rc = G0c - Sig, r0c - (rbc - r0c)
                    out["eiv_snr0"], _ = snr_trunc(G0c, Sig, Gc, rc, sc_, tau=0.0)
                nv = out["naive"]
                mE = float(nv[:C][nv[:C] > 0].median()) if int((nv[:C] > 0).sum()) \
                    else float(nv[:C].abs().median())
                mg = float(nv[C:][nv[C:] > 0].median()) if int((nv[C:] > 0).sum()) \
                    else float(nv[C:].abs().median())
                lo = torch.cat([torch.full((C,), 0.2 * mE, dtype=torch.float64),
                                torch.full((C,), 0.2 * mg, dtype=torch.float64)])
                hi = torch.cat([torch.full((C,), 5.0 * mE, dtype=torch.float64),
                                torch.full((C,), 5.0 * mg, dtype=torch.float64)])
                out["naive_box"], _ = solve_box(G0c, r0c, sc_, lo, hi,
                                                z0=torch.clamp(nv, lo, hi) / sc_, iters=4000)
                if K > 0:
                    out["eiv_box"], _ = solve_box(Gc, rc, sc_, lo, hi,
                                                  z0=torch.clamp(out["eiv_snr0"], lo, hi) / sc_,
                                                  iters=4000)
                ex_ = {"rel_v_mean": float(np.mean(relv)) if relv else 0.0,
                       "rel_C_mean": float(np.mean(relC)) if relC else 0.0,
                       "box": [0.2 * mE, 5.0 * mE]}
                return out, ex_, (eb, ex)

            # ---- the held-out one-frame residual at tick 180 -------------------------------- #
            def holdout(theta, kind, vname, eb, ex, use_noisy_F):
                vv, CC = state_of(kind, vname, hk, eb, ex)
                install_state(sy, B[hk]["snap"], vv, CC, Jp_one=True)
                if use_noisy_F:
                    inj = lerp(B[hk]["F0"] + eb[hk], B[hk]["F1"] + eb[hk + 1], n)
                    yo = (B[hk]["x_next"] + ex[hk + 1] - B[hk]["x0"]).reshape(-1)
                else:
                    inj = lerp(B[hk]["F0"], B[hk]["F1"], n)
                    yo = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)
                y = y_of(sy, theta.to(sy.device, sy.dtype), n, inj, None)
                return float((y - yo).norm() / yo.norm())

            zeroF = {t: torch.zeros_like(B[t]["F0"]) for t in B}
            zerox = {t: torch.zeros_like(B[t]["x0"]) for t in B}

            # ---- the cell list ----------------------------------------------------------------- #
            CELLS = []
            if "f" in a.stages:
                CELLS += [("cleanF|oracle", "oracle", None, False, 0, 0),
                          ("cleanF|derived", "derived", a.vclean, False, 0, 0),
                          ("cleanF|v_only", "v_only", a.vclean, False, 0, 0),
                          ("cleanF|C_only", "C_only", a.vclean, False, 0, 0)]
                for j, sd in enumerate(seeds):
                    CELLS += [(f"noisyF_s{sd}|oracle", "oracle", None, True, sd, a.K),
                              (f"noisyF_s{sd}|derived", "derived", a.vnoisy, True, sd, a.K)]
                    if j:                       # the decomposition runs on the first seed only
                        continue
                    if a.valt and a.valt != a.vnoisy:
                        CELLS += [(f"noisyF_s{sd}|derived_{a.valt}", "derived", a.valt,
                                   True, sd, a.K)]
                    CELLS += [(f"noisyF_s{sd}|v_only", "v_only", a.vnoisy, True, sd, a.K),
                              (f"noisyF_s{sd}|C_only", "C_only", a.vnoisy, True, sd, a.K)]

            thetas, R["cells"] = {}, {}
            log(f"\n[f] T={a.T} stacked, ticks {fit_ticks[0]}..{fit_ticks[-1]}; "
                f"noise = refute5 grid-{a.nodes}, sigma_F={SIGMA_F}, sigma_x={SIGMA_X:.3e}, "
                f"K={a.K}\n    controls: cleanF|oracle|naive must be 0.008562 (round 5); "
                f"noisyF|oracle|naive ~0.73 and eiv_box ~0.15 (refute5 grid48)")
            log(f"    {'cell':<28s} {'solver':<10s} {'medE':>7s} {'p90':>7s} {'relL2':>7s} "
                f"{'neg':>4s} {'mr':>6s} {'hoCLEAN':>8s} {'hoMEAS':>8s} {'relv':>7s} {'relC':>7s}")
            for nm, kind, vname, noisy, sd, K in CELLS:
                out, ex_, (eb, ex) = fit_cell(kind, vname, noisy, sd, K)
                R["cells"][nm] = {"kind": kind, "v_rule": vname, "noisy": noisy, "seed": sd,
                                  "K": K, **ex_, "solvers": {}}
                for kk, t in out.items():
                    ps = pstats(t.numpy(), th.cpu().numpy(), C)
                    hoc = holdout(t, kind, vname, zeroF, zerox, False)
                    hom = holdout(t, kind, vname, eb, ex, noisy) if noisy else hoc
                    R["cells"][nm]["solvers"][kk] = {**ps, "holdout_cleanF": hoc,
                                                     "holdout_measured": hom}
                    thetas[f"{nm}|{kk}"] = t.numpy()
                    log(f"    {nm:<28s} {kk:<10s} {ps['med_E']:>7.4f} {ps['p90_E']:>7.4f} "
                        f"{ps['rel_l2']:>7.4f} {ps['n_negE']:>4d} {ps['mean_ratio_E']:>6.3f} "
                        f"{hoc:>8.5f} {hom:>8.5f} {ex_['rel_v_mean']:>7.4f} "
                        f"{ex_['rel_C_mean']:>7.4f} [{time.time()-t_start:.0f}s]")

            # theta_true's own held-out floors
            R["holdout_floor"] = {
                "theta_true_cleanF_oracleState": holdout(th, "oracle", None, zeroF, zerox, False),
                "theta_true_cleanF_derivedState_c4": holdout(th, "derived", a.vclean, zeroF,
                                                             zerox, False)}
            log(f"    floors: theta_true clean-F oracle-state "
                f"{R['holdout_floor']['theta_true_cleanF_oracleState']:.5f}; derived-state "
                f"{R['holdout_floor']['theta_true_cleanF_derivedState_c4']:.5f}")

            # ---------------------------------------------------------------- stage x --------- #
            #  the ONE remaining idealization: the assembler's particle positions are exact.
            if "x" in a.stages:
                log(f"\n[x] control: sigma_x ALSO on the assembler's own particle positions "
                    f"(round 5 keeps x0 exact and noises the observation only)")
                for kind, vname, noisy, sd in [("oracle", None, True, seeds[0]),
                                               ("derived", a.vnoisy, True, seeds[0])]:
                    out, ex_, _ = fit_cell(kind, vname, noisy, sd, 0, dxnoise=True)
                    nm = f"xpos|noisyF_s{sd}|{kind}"
                    R["cells"][nm] = {"kind": kind, "v_rule": vname, "noisy": noisy, "seed": sd,
                                      "K": 0, "x0_also_noisy": True, **ex_, "solvers": {}}
                    for kk, t in out.items():
                        ps = pstats(t.numpy(), th.cpu().numpy(), C)
                        R["cells"][nm]["solvers"][kk] = ps
                        thetas[f"{nm}|{kk}"] = t.numpy()
                        log(f"    {nm:<28s} {kk:<10s} {ps['med_E']:>7.4f} "
                            f"relL2 {ps['rel_l2']:>7.4f} neg {ps['n_negE']:>3d} "
                            f"mr {ps['mean_ratio_E']:>6.3f} [{time.time()-t_start:.0f}s]")

            thetas["theta_true"] = th.cpu().numpy()
            np.savez(os.path.join(HERE, f"{a.tag}_theta.npz"), **thetas)
            log(f"\n    wrote {a.tag}_theta.npz ({len(thetas)} vectors)")

        json.dump(R, open(os.path.join(HERE, f"{a.tag}_fit.json"), "w"), indent=1, default=str)
        open(os.path.join(HERE, f"{a.tag}_fit.log"), "w").write("\n".join(lines) + "\n")

    # ========================================================================= SCORE =========== #
    if "s" in a.stages:
        import metrics as MET
        import crash_test as CT
        from assemble import SUBSTEP_TOKENS
        from crash_round2 import percell_amplitude, r2_percell
        from crash_round3 import scale2, t2_of
        from finject import record_substeps
        from refute_round3 import advance
        from round5_score import gauge_grid

        lines2 = []

        def log2(s):
            print(s, flush=True)
            lines2.append(str(s))

        S = {"config": vars(args), "cli": vars(a)}
        t2 = time.time()
        with torch.no_grad():
            sy, recA = CT.plant_and_warm(args, log2)
            C, W, G, n = sy.C, a.t0, a.window, sy.n_sub_per_frame
            th = sy.theta_true.double()
            dev, f64 = th.device, torch.float64
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
            log2(f"[reference] {G}-frame window from tick {W}; cells kept {int(keep.sum())} "
                 f"[{time.time()-t2:.0f}s]")

            # held-out frame, clean F (the acceptance statistic's own configuration)
            sy.restore()
            advance(sy, W, a.holdout_tick)
            sy._snapshot(a.holdout_tick)
            Fh, _, Xh = record_substeps(sy, n)
            hx0, hF0, hF1, hxn = sy.x0.clone(), sy.F0.clone(), Fh[-1].clone(), Xh[-1].clone()
            y_obs_h = (hxn - hx0).reshape(-1)
            injh = lerp(hF0, hF1, n)
            hsnap = {k: getattr(sy, k).clone() for k in SNAP}

            def holdout_oracle(theta):
                for k, v in hsnap.items():
                    setattr(sy, k, v.clone())
                y = y_of(sy, theta, n, injh, None)
                return float((y - y_obs_h).norm() / y_obs_h.norm())

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
                    out["percell_r2"] = r2_percell(ah, a_ref, keep)["r2"]
                    del full
                return out

            # ---- candidates ------------------------------------------------------------------ #
            Z = np.load(os.path.join(HERE, a.theta_in))
            sd0 = seeds[0]
            want = ["theta_true",
                    "cleanF|oracle|naive", "cleanF|derived|naive"]
            for sd in seeds:
                want += [f"noisyF_s{sd}|oracle|eiv_box", f"noisyF_s{sd}|derived|eiv_box"]
            want += [f"noisyF_s{sd0}|oracle|naive", f"noisyF_s{sd0}|derived|naive",
                     f"noisyF_s{sd0}|oracle|naive_box", f"noisyF_s{sd0}|derived|naive_box"]
            if a.valt and a.valt != a.vnoisy:
                want += [f"noisyF_s{sd0}|derived_{a.valt}|eiv_box"]
            cands = [(k, torch.as_tensor(Z[k], device=dev, dtype=f64)) for k in want if k in Z.files]
            missing = [k for k in want if k not in Z.files]

            # ---- the zero-information null bank, round 5's definition, same protocol --------- #
            def const(E, g):
                return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                                  torch.full((C,), float(g), device=dev, dtype=f64)])
            BANK = [("bank_blind_E130_g0.95", const(130.0, 0.95))]
            gg = torch.Generator().manual_seed(101)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            BANK.append(("bank_prior_draw_101", torch.cat([Ed, gd])))
            gnull = torch.Generator().manual_seed(4242)
            ii = torch.randperm(C, generator=gnull)[:45].to(dev)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gnull,
                                                                   dtype=f64)).to(dev)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gnull,
                                                                   dtype=f64)).to(dev)
            nm0 = th.clone()
            nm0[ii] = Ed[ii]
            nm0[C + ii] = gd[ii]
            BANK.append(("null_med0_rand45", nm0))
            cands = cands + BANK
            if a.only:
                keepn = set(a.only.split(","))
                cands = [c for c in cands if c[0] in keepn]
            mine = [c for i, c in enumerate(cands)
                    if a.nshards == 1 or i % a.nshards == a.shard]
            S["missing_from_theta_npz"] = missing
            log2(f"[score] {len(mine)}/{len(cands)} candidates (missing {missing}); "
                 f"free {G}-frame rollout, margin {MET.MARGIN_SAFE}, anchor=None, "
                 f"{a.grid}x{a.grid} grid gauge + {a.refine} Broyden")
            log2(f"    {'candidate':<34s} {'medE':>7s} {'neg':>4s} {'mr':>6s} {'hoCLEAN':>8s} "
                 f"{'raw':>8s} {'kE':>6s} {'kg':>6s} {'gauged':>8s} {'+-':>5s} {'R2':>8s} "
                 f"{'r2cell':>7s}")
            S["candidates"] = {}
            for name, theta in mine:
                tc = time.time()
                ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
                hc = holdout_oracle(theta)
                raw = scored(theta)

                def probe(lE, lg, theta=theta):
                    return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

                gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"],
                                gn=a.grid, refine=a.refine)
                kE, kg = gf["k_E"], gf["k_g"]
                gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                    else scored(scale2(theta, kE, kg, C))
                S["candidates"][name] = {
                    "param": ps, "holdout_1frame_cleanF_oracleState": hc,
                    "raw": raw, "gauged": gau,
                    "gauge": {k: v for k, v in gf.items() if k != "cells"},
                    "seconds": time.time() - tc}
                log2(f"    {name:<34s} {ps['med_E']:>7.4f} {ps['n_negE']:>4d} "
                     f"{ps['mean_ratio_E']:>6.3f} {hc:>8.5f} {CT.fmt(raw['loop'],8)} "
                     f"{kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['loop'],8)} "
                     f"{gf['gauge_uncertainty']:>5.3f} {CT.fmt(gau['R2'],8)} "
                     f"{(gau['percell_r2'] if gau['percell_r2'] is not None else float('nan')):>7.4f}"
                     f"  [{time.time()-tc:.0f}s]")

        S["wall_seconds"] = time.time() - t2
        json.dump(S, open(os.path.join(HERE, f"{a.tag}_score_s{a.shard}.json"), "w"),
                  indent=1, default=str)
        open(os.path.join(HERE, f"{a.tag}_score_s{a.shard}.log"), "w").write("\n".join(lines2) + "\n")
        log2(f"\nwrote {a.tag}_score_s{a.shard}.json [{S['wall_seconds']:.0f} s]")

    # ========================================================================== PLOT =========== #
    if "p" in a.stages:
        make_figure(a, HERE)

    if "f" in a.stages or "x" in a.stages:
        log(f"\ndone [{time.time()-t_start:.0f} s]")


# --------------------------------------------------------------------------------------------- #
def make_figure(a, HERE):
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z = np.load(os.path.join(HERE, f"{a.tag}_theta.npz"))
    th = Z["theta_true"]
    C = th.size // 2
    Etrue = th[:C]
    F = json.load(open(os.path.join(HERE, f"{a.tag}_fit.json")))
    SC = {}
    for fp in sorted(glob.glob(os.path.join(HERE, f"{a.tag}_score_s*.json"))):
        SC.update(json.load(open(fp)).get("candidates", {}))
    sd0 = a.seeds.split(",")[0]

    panels = [("clean F", [("cleanF|oracle|naive", "oracle state"),
                           ("cleanF|derived|naive", "derived state")]),
              (f"realizable F noise ($\\sigma_F$ = {SIGMA_F:g}, grid-{a.nodes})",
               [(f"noisyF_s{sd0}|oracle|eiv_box", "oracle state"),
                (f"noisyF_s{sd0}|derived|eiv_box", "derived state")])]
    COL = {"oracle state": "#e04a4a", "derived state": "#4a9ce0"}

    plt.rcParams.update({"font.size": 10, "text.color": "w", "axes.labelcolor": "w",
                         "xtick.color": "w", "ytick.color": "w"})
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 5.0), facecolor="k")
    for ax, (ttl, ser) in zip(axs, panels):
        ax.set_facecolor("k")
        for sp in ax.spines.values():
            sp.set_color("0.5")
        lim = [0, 260]
        ax.plot(lim, lim, "-", color="0.55", lw=1.0, zorder=1)
        for key, lab in ser:
            if key not in Z.files:
                continue
            Eh = Z[key][:C]
            cell = key.rsplit("|", 1)[0]
            solver = key.rsplit("|", 1)[1]
            st = F["cells"][cell]["solvers"][solver]
            g = SC.get(key, {}).get("gauged", {}).get("loop", float("nan"))
            ax.plot(Etrue, Eh, "o", ms=4.2, mfc=COL[lab], mec="none", alpha=0.85, zorder=3,
                    label=f"{lab}:  med|dE/E| {st['med_E']:.3f}   held-out {st['holdout_cleanF']:.3f}"
                          f"   loop {g:.3f}")
        ax.set_xlim(lim)
        ax.set_ylim(-40, 340)
        ax.set_xlabel("planted $E$")
        ax.text(0.02, 0.975, ttl, transform=ax.transAxes, ha="left", va="top", color="w",
                fontsize=11)
        lg = ax.legend(loc="lower right", frameon=False, fontsize=8.2, handletextpad=0.4)
        for t in lg.get_texts():
            t.set_color("w")
        ax.grid(alpha=0.12, color="w")
    axs[0].set_ylabel("recovered $\\hat E$")
    axs[0].text(0.02, 0.90, "a", transform=axs[0].transAxes, color="w", fontsize=13,
                fontweight="bold")
    axs[1].text(0.02, 0.90, "b", transform=axs[1].transAxes, color="w", fontsize=13,
                fontweight="bold")
    fig.tight_layout()
    out = os.path.join(HERE, f"{a.tag}.png")
    fig.savefig(out, dpi=160, facecolor="k")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

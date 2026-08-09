#!/usr/bin/env python
"""p1d_gainfit.py -- PROBE D.  What does a GAIN-ONLY fit actually recover?

THE QUESTION
====================================================================================================
P0 measured that a uniform Young's modulus moves the certified amplitude instrument by 12% over a
40-fold sweep, NON-MONOTONICALLY, while a uniform contraction gain moves it monotonically with
exponent +0.886 over a 16-fold sweep -- 22x the dynamic range.  That says E is not the channel the
data carries and gain is.  The pivot from "solve for (E, gain)" to "solve for gain, declare E" is
only worth making if the second problem is actually SOLVABLE on data a recording can supply, so
this file fits it, end to end, and scores the rollout with the P0 acceptance statistic.

Three fits on IDENTICAL data, at two cadences, under three deformation-gradient conditions:

    joint          E and gain free, 2C = 200 unknowns          the campaign's current formulation
    gain@E_true    E pinned to the truth, C = 100 unknowns     the best case, an upper bound
    gain@E_const   E pinned to a WRONG uniform constant        the realistic case: on a recording E
                   (132 = the true median, 60 and 240)         is unknown and, per P0, unrecoverable

THE KEY QUESTION is the last row: how much does getting E wrong cost the gain estimate?  If gain
recovery is robust to a badly wrong E the pivot is clean; if the two trade off, it is not.

WHAT IS AND IS NOT THIS FILE'S OWN
----------------------------------------------------------------------------------------------------
Nothing about the estimator is new here.  The system is `assemble.System`, the assembly is
`finject.assemble_inj`, the solve is `recover.Solver` at ridge0, the derived F is
`freal_derivedF.ControlGrid` + `derive_F`, the noise draw is `refute5_fit.NoiseF`, the rollout is
`crash_test.rollout` and the score is `accept.accept`.  All are imported unmodified.  The ONE new
line of algebra is the restriction of the design matrix to its gain block,

    A theta = A_E E + A_g g      ->      A_g g = b - A_E E_fixed

which is the same normal-equations solve on 100 columns instead of 200.  `recover.Solver` reads its
column scale from the module-level `recover.theta_scale`, which returns the JOINT [130]*C + [1]*C
vector; for a 100-column design that vector is the wrong length, so the function is redirected for
the duration of the construction (`solve_block`) and put back.  The solver -- normal equations,
ridge ladder, eigendecomposition, `cond` -- is the campaign's, untouched.

THE TICK IS 165, NOT 180, AND THAT IS NOT A CHOICE
----------------------------------------------------------------------------------------------------
The task specifies warmup=180.  At tick 180 the pacemaker clock is `(180 % 150) < 30` = False, the
active-force delta is identically zero, and therefore EVERY GAIN COLUMN OF A IS EXACTLY ZERO
(CRASH_TEST.md section 4.4: 100/100 zero columns at 8 of 9 ticks tested).  A gain-only fit there is
not hard, it is undefined -- 0 x = b.  Every F-injection number in the record sits at tick 165,
where the clock is exactly 1.0, so the fit is at 165 and the controls below reproduce
`freal_derivedF.log` digit for digit.

WHAT IS HELD FIXED SO THE COMPARISON MEANS SOMETHING
----------------------------------------------------------------------------------------------------
* the observation `b` is the same exact simulated displacement in all conditions -- ONLY F, which
  sits in A, is corrupted.  That is the errors-in-variables structure round 4 established, and it
  is why the failure mode is attenuation and not variance.
* the fit uses ONE frame (or one substep) at tick 165.  No stacking, no box, no EIV correction:
  this probe is about which BLOCK is identifiable, not about how well the campaign's best estimator
  does.  The joint column is therefore the campaign's *unstacked* baseline, and it reproduces
  `freal_derivedF.log`.
* the rollout installs the SAME E the fit assumed.  A gain-only fit that pins E = 60 must be rolled
  out at E = 60; using the true E at rollout time would be scoring a parameter vector nobody has.
* the acceptance statistic is read at THREE snapshot ticks (165, 180, 195), because `accept` refuses
  fewer and because tick 165 is the easiest frame in the window.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1d_gainfit.py --device cuda:1 --stages f
  PYTHONPATH=/workspace/Plexus/src python p1d_gainfit.py --device cuda:1 --stages fr
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

import recover as RC                                                    # noqa: E402
from recover import Solver, install_E                                   # noqa: E402
import metrics as MET                                                   # noqa: E402
import crash_test as CT                                                 # noqa: E402
import accept as ACC                                                    # noqa: E402
from finject import lerp, assemble_inj, record_substeps                 # noqa: E402
from freal_derivedF import ControlGrid, derive_F, collect, PX, GRID_PX  # noqa: E402
from round5_fit import SIGMA_F, SNAP                                    # noqa: E402
from round5_solve import pstats                                         # noqa: E402
from refute5_fit import NoiseF                                          # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  the one new line of algebra: solve a BLOCK of the campaign's design matrix
# --------------------------------------------------------------------------------------------- #
def solve_block(A, scale):
    """`recover.Solver` on a design matrix with `scale.numel()` columns.

    Solver column-scales by `recover.theta_scale(C, device)`, which is the joint 2C vector.  The
    gain block has C columns, so the function is redirected to return the block's own scale
    (g_ref = 1 for every column) and restored immediately.  Nothing inside Solver changes.
    """
    orig = RC.theta_scale
    RC.theta_scale = lambda C, device, **kw: scale.to(device)
    try:
        return RC.Solver(A, A.shape[1])
    finally:
        RC.theta_scale = orig


def gstats(g_hat, g_true):
    """What a gain estimate is worth: error, placement, and the attenuation of its spread.

    The attenuation is the OLS slope of recovered on true WITH AN INTERCEPT.  A gain has a large
    non-zero mean (1.0) and a small spread (+/-0.5), so a through-the-origin slope would report the
    mean ratio and hide a completely flattened per-cell pattern -- which is exactly the failure
    this probe is looking for.
    """
    g_hat = np.asarray(g_hat, float)
    g_true = np.asarray(g_true, float)
    r = np.abs(g_hat - g_true) / g_true
    ok = np.isfinite(g_hat)
    slope = float(np.polyfit(g_true[ok], g_hat[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    return {"med_rel": float(np.median(r)), "p90_rel": float(np.percentile(r, 90)),
            "max_rel": float(r.max()),
            "corr": float(np.corrcoef(g_hat, g_true)[0, 1]),
            "attenuation_slope": slope,
            "mean_ratio": float(g_hat.mean() / g_true.mean()),
            "std_ratio": float(g_hat.std() / g_true.std()),
            "n_negative": int((g_hat < 0).sum()),
            "rel_l2": float(np.linalg.norm(g_hat - g_true) / np.linalg.norm(g_true))}


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1d")
    ap.add_argument("--stages", default="fr")
    ap.add_argument("--t0", type=int, default=165, help="fit tick; MUST be on the pacemaker pulse")
    ap.add_argument("--score-ticks", default="165,180,195")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--hpx", type=float, default=GRID_PX, help="control-grid spacing, in px")
    ap.add_argument("--e-fix", default="132,60,240")
    ap.add_argument("--eps-e", default="0.03,0.1,0.3", help="E-error dose-response, fit-stage only")
    ap.add_argument("--box-g", default="0.2,5.0", help="DECLARED box on gain, for the rollout")
    ap.add_argument("--box-e", default="26.4,660.0", help="DECLARED box on E (132 x /5, x5)")
    ap.add_argument("--noise-nodes", type=int, default=48)
    ap.add_argument("--seed-noise", type=int, default=90210)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "args": vars(a), "sigma_F": SIGMA_F}
    t_start = time.time()
    torch.manual_seed(0)
    sticks = [int(v) for v in a.score_ticks.split(",")]
    e_fix_vals = [float(v) for v in a.e_fix.split(",")]

    with torch.no_grad():
        sy, REF, B = collect(args, a.t0, max(sticks), log)
        C, n_frame, dx = sy.C, sy.n_sub_per_frame, sy.g.dx
        th = sy.theta_true.double()
        E_true, g_true = th[:C], th[C:]
        dev, f64 = th.device, torch.float64
        X0, F0ref = REF[0]["x"], REF[0]["F"]
        k0 = a.t0
        log(f"[collect] ticks {a.t0}..{max(sticks)}; C={C} Np={sy.Np} dx={dx:.6g} "
            f"[{time.time()-t_start:.0f}s]")

        # the pacemaker must be ON, or the gain block does not exist
        clock = float(np.sin(np.pi * (k0 % 150) / 30.0)) if (k0 % 150) < 30 else 0.0
        for kk, vv in B[k0]["snap"].items():
            setattr(sy, kk, vv.clone())
        R["pacemaker"] = {"tick": k0, "clock": clock, "act_norm": float(sy.act0.norm()),
                          "pass_norm": float(sy.pass0.norm())}
        log(f"[pacemaker] tick {k0}: clock {clock:.4f}, ||active delta|| "
            f"{float(sy.act0.norm()):.4g}, ||passive delta|| {float(sy.pass0.norm()):.4g}")
        if float(sy.act0.norm()) == 0.0:
            raise SystemExit("the active-force delta is zero at this tick: the gain block of A is "
                             "identically zero and a gain-only fit is undefined")

        band = 0.06 / MET.SHEET_SPAN
        xW = B[k0]["x0"]
        interior = ~((xW[:, 0] < band) | (xW[:, 0] > 1 - band)
                     | (xW[:, 1] < band) | (xW[:, 1] > 1 - band))

        cg = ControlGrid(X0, a.hpx * PX)

        def derF(x):
            """F = I + grad u on a control grid of spacing h, from the tick-0 reference config.

            This is `freal_derivedF`'s m = t0 case: the displacement's reference configuration is
            the recording's own frame 0, F_ref = I, so nothing in the measurement is an oracle.
            """
            return derive_F(cg, X0, x, "bilinear", F_ref=F0ref)

        R["control_grid"] = {"h_px": a.hpx, "h_world": a.hpx * PX, "nodes": cg.n_nodes,
                             "valid": cg.n_valid, "pts_per_valid_node": cg.pts_per_valid}

        # ---------------------------------------------------------------- the six conditions ----
        NF = NoiseF("grid", B[k0]["x0"], a.noise_nodes, dev, f64)

        conds, obs = {}, {}
        for n in (1, n_frame):
            # the generator is reset per cadence so BOTH cadences see the SAME realisation of the
            # error field -- otherwise the two columns differ by the draw as well as by the cadence,
            # and at n = 10 this reproduces freal_derivedF's F_noise_grid48 exactly
            gnoise = torch.Generator(device=dev).manual_seed(a.seed_noise)
            for kk, vv in B[k0]["snap"].items():
                setattr(sy, kk, vv.clone())
            Fs, _, Xs = record_substeps(sy, n)
            F0, F1, xN = sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()
            obs[n] = {"x0": sy.x0.clone(), "x_next": xN}
            e0 = (SIGMA_F / 2.0) * NF(gnoise)
            e1 = (SIGMA_F / 2.0) * NF(gnoise)
            # one COHERENT error per frame boundary, then lerp across the substeps -- finject's
            # `_noise_F` convention, kept identical so the number is comparable
            conds[(n, "clean")] = lerp(F0, F1, n)
            conds[(n, f"noiseF_grid{a.noise_nodes}")] = lerp(F0 + e0, F1 + e1, n)
            conds[(n, f"derivedF_{a.hpx:g}px")] = lerp(derF(sy.x0), derF(xN), n)
            # the truth the injected F is compared against, per substep
            conds[(n, "_true")] = Fs.clone()

        # ---------------------------------------------------------------- stage f: the fits -----
        # A DOSE-RESPONSE, not a verdict: E pinned to the truth times (1 + eps u), one FIXED unit
        # vector u shared by every condition, so "how wrong may E be before the gain fit dies" is a
        # curve rather than an adjective.  Free -- it reuses the same assembly.
        eps_E = [float(v) for v in a.eps_e.split(",") if float(v) > 0]
        gu = torch.Generator(device=dev).manual_seed(77)
        uE = torch.randn(C, generator=gu, device=dev, dtype=f64)
        uE = uE / uE.abs().max()
        R["eps_E_unit_vector"] = {"seed": 77, "max_abs": 1.0, "std": float(uE.std())}

        thetas, R["fits"] = {}, {}
        log(f"\n[f] FITS at tick {k0}, ridge0, displacement read-out.  b is the EXACT simulated "
            f"displacement in every row; only F (which lives in A) is corrupted.")
        log(f"    {'cadence':<8s} {'F':<18s} {'fit':<14s} {'medg':>8s} {'p90g':>8s} "
            f"{'corr_g':>7s} {'atten':>7s} {'meanrat':>8s} {'negg':>5s} {'medE':>8s} "
            f"{'cond':>9s} {'resid':>9s}")
        for n in (1, n_frame):
            Ftrue = conds[(n, "_true")]
            for fname in ("clean", f"noiseF_grid{a.noise_nodes}", f"derivedF_{a.hpx:g}px"):
                iF = conds[(n, fname)]
                de = (iF - Ftrue)[:, interior].abs()
                errF = {"med": float(de.median()), "rms": float(de.pow(2).mean().sqrt())}
                for kk, vv in B[k0]["snap"].items():
                    setattr(sy, kk, vv.clone())
                A, y0, t_as = assemble_inj(sy, n, iF, None)
                b = (obs[n]["x_next"] - obs[n]["x0"]).reshape(-1) - y0
                A_E, A_g = A[:, :C].contiguous(), A[:, C:].contiguous()
                col = {"colnorm_E_med": float(A_E.norm(dim=0).median()),
                       "colnorm_gain_med": float(A_g.norm(dim=0).median()),
                       "n_zero_gain_cols": int((A_g.norm(dim=0) == 0).sum())}

                def best_ridge(sol, gt):
                    """The rung of the ridge ladder an ORACLE would pick, and what it would buy.

                    Reported, never used: choosing it needs the truth.  It is here so that "the
                    gain-only fit failed" cannot be answered with "you did not regularise it" --
                    round 1 already measured that the best l2 sits at ridge 0, and this checks the
                    same thing per condition on the gain block.
                    """
                    rows = [(k, float(np.median(np.abs(v[C:].cpu().numpy() - gt) / gt)))
                            for k, v in sol.items() if k.startswith("ridge")]
                    k, m = min(rows, key=lambda r: r[1])
                    return {"ridge": k, "med_rel_gain": m}

                gt_np = g_true.cpu().numpy()
                fits, oracle = {}, {}
                S = Solver(A, C)
                sol = S(b)
                fits["joint"] = (sol["ridge0"], S.cond)
                oracle["joint"] = best_ridge(sol, gt_np)
                S.free()
                del S, sol
                for lab, Efx in ([("gain@E_true", E_true)]
                                 + [(f"gain@E{v:g}", torch.full((C,), v, device=dev, dtype=f64))
                                    for v in e_fix_vals]
                                 + [(f"gain@E_true_err{e:g}", E_true * (1.0 + e * uE))
                                    for e in eps_E]):
                    Sg = solve_block(A_g, torch.ones(C, device=dev, dtype=f64))
                    solg = Sg(b - A_E @ Efx)
                    fits[lab] = (torch.cat([Efx, solg["ridge0"]]), Sg.cond)
                    oracle[lab] = best_ridge({k: torch.cat([Efx, v]) for k, v in solg.items()},
                                             gt_np)
                    Sg.free()
                    del Sg, solg
                torch.cuda.empty_cache()

                for lab, (t_hat, cond) in fits.items():
                    key = f"n{n}|{fname}|{lab}"
                    thetas[key] = t_hat.clone()
                    gs = gstats(t_hat[C:].cpu().numpy(), g_true.cpu().numpy())
                    ps = pstats(t_hat.cpu().numpy(), th.cpu().numpy(), C)
                    resid = float((A @ t_hat - b).norm() / b.norm())
                    R["fits"][key] = {"n_sub": n, "F": fname, "fit": lab, "gain": gs,
                                      "pstats": ps, "cond": cond, "fit_residual": resid,
                                      "errF": errF, "cols": col, "assemble_s": t_as,
                                      "oracle_ridge": oracle[lab]}
                    log(f"    {('sub' if n == 1 else 'frame'):<8s} {fname:<18s} {lab:<14s} "
                        f"{gs['med_rel']:>8.4f} {gs['p90_rel']:>8.4f} {gs['corr']:>7.3f} "
                        f"{gs['attenuation_slope']:>7.3f} {gs['mean_ratio']:>8.4f} "
                        f"{gs['n_negative']:>5d} {ps['med_E']:>8.4f} {cond:>9.2e} "
                        f"{resid:>9.3e}  oracle-ridge {oracle[lab]['ridge']:<10s} "
                        f"{oracle[lab]['med_rel_gain']:.4f}")
                del A, A_E, A_g
                torch.cuda.empty_cache()

        R["controls_expected"] = {
            "freal_derivedF.log, joint, frame cadence": {
                "clean(F_lerp_simF) med_E": 0.0078, "derivedF_15px med_E": 0.9986,
                "noiseF_grid48 med_E": 0.8416}}
        np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"), theta_true=th.cpu().numpy(),
                 **{k.replace("|", "__"): v.cpu().numpy() for k, v in thetas.items()})
        json.dump(R, open(os.path.join(HERE, f"{a.tag}_fit.json"), "w"), indent=1, default=str)
        log(f"\n[f] {len(thetas)} fits in {time.time()-t_start:.0f}s -> {a.tag}_fit.json")

        # ---------------------------------------------------------------- the nulls -------------
        # Honest nulls, in the same column, rolled out through the same code path.
        gp = torch.Generator(device=dev).manual_seed(4242)
        nulls = {"theta_true": th.clone(),
                 "null_g1_at_E_true": torch.cat([E_true, torch.ones(C, device=dev, dtype=f64)]),
                 "null_g1_at_E132": torch.cat([torch.full((C,), 132.0, device=dev, dtype=f64),
                                               torch.ones(C, device=dev, dtype=f64)])}
        for v in e_fix_vals:
            nulls[f"null_gain_true_at_E{v:g}"] = torch.cat(
                [torch.full((C,), v, device=dev, dtype=f64), g_true])
        # theta_true plus a CORRECTLY-SIZED but MISPLACED gain error: same l2, same tail, wrong cell
        for src in (f"n{n_frame}|derivedF_{a.hpx:g}px|gain@E132",
                    f"n{n_frame}|clean|gain@E_true"):
            if src not in thetas:
                continue
            err = thetas[src][C:] - g_true
            perm = torch.randperm(C, generator=gp, device=dev)
            nulls[f"null_permerr[{src}]"] = torch.cat([thetas[src][:C], g_true + err[perm]])
        R["nulls_gain"] = {k: gstats(v[C:].cpu().numpy(), g_true.cpu().numpy())
                           for k, v in nulls.items()}

        if "r" not in a.stages:
            open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
            return

        # ---------------------------------------------------------------- stage r: rollouts -----
        # WHAT IS ROLLED OUT, and why it is the CHARITABLE version.
        # An unconstrained ridge0 gain-only fit with a wrong E returns gains of -900 and +300.
        # Nobody would submit that vector, and CRASH_TEST.md already records that the box is
        # load-bearing rather than a mild prior (widen it 4x and rel l2 goes 0.254 -> 0.719).  So
        # every candidate is first projected onto a DECLARED box -- gain in [0.2, 5], E in
        # 132 x [1/5, 5] -- declared from the same physiology the campaign declares it from, never
        # anchored on the data (P0 showed the amplitude anchor is not invertible).  The truth lies
        # well inside both.  Clipping can only help the wrong-E fits, so a failure after clipping
        # is a failure of the formulation and not of the arithmetic.  Two unclipped candidates are
        # rolled out beside their clipped twins to price the projection.
        box_g = [float(v) for v in a.box_g.split(",")]
        box_e = [float(v) for v in a.box_e.split(",")]

        def boxed(t):
            return torch.cat([t[:C].clamp(box_e[0], box_e[1]), t[C:].clamp(box_g[0], box_g[1])])

        R["declared_box"] = {"E": box_e, "gain": box_g,
                             "planted_E": [float(E_true.min()), float(E_true.max())],
                             "planted_gain": [float(g_true.min()), float(g_true.max())]}
        keep_roll = ("joint", "gain@E_true", "gain@E132", "gain@E60", "gain@E240")
        cands, R["clipped"] = {}, {}
        for k, v in thetas.items():
            if k.rsplit("|", 1)[-1] not in keep_roll:
                continue
            tb = boxed(v)
            R["clipped"][k] = {"n_E_on_bound": int((tb[:C] != v[:C]).sum()),
                               "n_gain_on_bound": int((tb[C:] != v[C:]).sum())}
            cands[k] = tb
        for k in (f"n{n_frame}|derivedF_{a.hpx:g}px|gain@E132", f"n{n_frame}|clean|gain@E132"):
            if k in thetas:
                cands[f"RAW(unboxed) {k}"] = thetas[k].clone()
        cands.update(nulls)
        G = a.window
        log(f"\n[r] {len(cands)} candidates x {len(sticks)} ticks x {G}-frame free rollouts "
            f"(margin {MET.MARGIN_SAFE}); the reference at each tick is theta_true through the "
            f"SAME code path, so theta_true must read 0.00 steps")
        floors = ACC.working_floors()
        R["floors"] = {k: v for k, v in floors.items()}
        R["null_steps"] = ACC.null_steps(floors)
        loops, coarse_all = {k: [] for k in cands}, {k: [] for k in cands}
        for T in sticks:
            for kk, vv in B[T]["snap"].items():
                setattr(sy, kk, vv.clone())
            snapT = {kk: getattr(sy, kk).clone() for kk in SNAP}
            x0T = sy.x0.clone()
            trc = {MET.MARGIN_SAFE: CT.tracer_indices(x0T, CT.probe_points(MET.MARGIN_SAFE))}
            intT = ~((x0T[:, 0] < band) | (x0T[:, 0] > 1 - band)
                     | (x0T[:, 1] < band) | (x0T[:, 1] > 1 - band))
            t_ref = time.time()
            _, ref_full, _ = CT.rollout(sy, th, T, G, trc, keep_full=True)
            d_ref = ref_full - x0T[None]
            ss_tot = (d_ref[:, intT] - d_ref[:, intT].mean(0, keepdim=True)).pow(2).sum()
            real = ref_full[:, trc[MET.MARGIN_SAFE]].cpu().numpy()
            log(f"    tick {T}: reference built in {time.time()-t_ref:.0f}s; max displacement "
                f"{float(d_ref.norm(dim=-1).max()/dx):.2f} dx")
            for name, theta in cands.items():
                tc = time.time()
                for kk, vv in snapT.items():
                    setattr(sy, kk, vv.clone())
                tr, _, co = CT.rollout(sy, theta, T, G, trc, ref_full=ref_full,
                                       interior=intT, ss_tot=ss_tot, band_mask=~intT)
                loops[name].append(tr[MET.MARGIN_SAFE].cpu().numpy())
                coarse_all[name].append({"R2": co["R2_displacement_interior"],
                                         "Eratio": co["motion_energy_ratio_interior"],
                                         "rms_dx": co["rms_pos_err_dx_mean"]})
                log(f"      {name:<44s} R2 {co['R2_displacement_interior']:>9.4f} "
                    f"rms/dx {co['rms_pos_err_dx_mean']:>7.4f} [{time.time()-tc:.0f}s]")
            del ref_full
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------- the statistic ---------
        R["accept"] = {}
        for name in cands:
            if not all(np.isfinite(L).all() for L in loops[name]):
                # a rollout that left the number line is not scored and not excused: it is recorded
                R["accept"][name] = {"statistic": float("inf"), "limiting_instrument": "DIVERGED",
                                     "pattern_channel": float("nan"),
                                     "amplitude_channel": float("nan"),
                                     "beats_null": {}, "informative": False,
                                     "coarse": coarse_all[name],
                                     "gain": gstats(cands[name][C:].cpu().numpy(),
                                                    g_true.cpu().numpy()),
                                     "med_E": pstats(cands[name].cpu().numpy(),
                                                     th.cpu().numpy(), C)["med_E"]}
                continue
            pairs = [(loops[name][i], loops["theta_true"][i]) for i in range(len(sticks))]
            v = ACC.accept(pairs, floors)
            v["coarse"] = coarse_all[name]
            v["gain"] = gstats(cands[name][C:].cpu().numpy(), g_true.cpu().numpy())
            v["med_E"] = pstats(cands[name].cpu().numpy(), th.cpu().numpy(), C)["med_E"]
            R["accept"][name] = v

        nul_min = min(R["null_steps"].values())
        log(f"\n[r] THE ACCEPTANCE STATISTIC, worst certified instrument over {len(sticks)} ticks, "
            f"in distinguishable steps (lower is better).  Knowing nothing = {nul_min:.2f}.")
        log(f"    {'candidate':<44s} {'STAT':>8s} {'pattern':>8s} {'amplit':>8s} "
            f"{'limiting':<18s} {'medg':>7s} {'atten':>6s} {'beats null?':>12s}")
        for name, v in sorted(R["accept"].items(), key=lambda kv: kv[1]["statistic"]):
            pc = v["pattern_channel"] if v["pattern_channel"] is not None else float("nan")
            log(f"    {name:<44s} {v['statistic']:>8.2f} {pc:>8.2f} "
                f"{v['amplitude_channel']:>8.2f} {v['limiting_instrument']:<18s} "
                f"{v['gain']['med_rel']:>7.3f} {v['gain']['attenuation_slope']:>6.2f} "
                f"{('YES' if v['statistic'] < nul_min else 'no'):>12s}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

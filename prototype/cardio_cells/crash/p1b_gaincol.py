"""p1b_gaincol.py -- PROBE B: do the GAIN columns of A depend on the deformation gradient F at all?

THE QUESTION
====================================================================================================
The F blocker that killed the algebraic route for per-cell E is a MEASUREMENT limit: a deformation
gradient derived the way a real recording must (boxcar bin + central difference on a ~15 px control
grid, zero added noise) reproduces the solver's own F with a least-squares scale of ~0.003 on the
E half, and the recovered med|dE/E| is 0.999 -- i.e. nothing.  If the GAIN columns of A do not
contain F, that limit simply does not apply to the gain, and the algebraic route survives for gain
even though it is dead for E.

THE ANALYTIC CLAIM UNDER TEST (the numbers below are what decides it)
----------------------------------------------------------------------------------------------------
  E    enters through  _lame(E) -> mu, la  ->  stress = 2mu (F-R) F^T + la J (J-1) I
       (mpm_scatter.py:112-113), manifestly a function of F.
  gain enters through  H._delta["mpm_particle"] = pass0 + gain[cid] * act0   (assemble.py:238),
       consumed at mpm_scatter.py:75 `a_ext += H.delta(p.name)`, :76 `V = V + dt*a_ext`, and
       :142 `mom = mass*V + affine @ dpos_phys` -- the mass*V term.  act0 is produced by
       active_force.py:69-70 `grad = fld.grad_at(pos); acc = sign*amplitude*grad`, which reads the
       activation field at the particle POSITION and nothing else: no F, no C, no Jp.
  => for ONE substep, dA_gain/dF == 0 in the bulk, and can be nonzero only through the ACTIVE SET of
     the wall kinks (mpm_grid_update.py:118-122, mpm_gather.py:75), whose branch condition is a
     function of the total grid velocity and therefore of the stress and therefore of F.
  => over MULTIPLE substeps the injected F moves the particles, the b-spline weights are read off
     the moved positions (mpm_grid.py:74-96) and p.C is re-gathered, so the ACCUMULATED gain column
     acquires an F dependence at second order in dt.  (a) and (b) are measured separately.

WHAT IS RUN
----------------------------------------------------------------------------------------------------
The planted C=100 system of crash_test.py at tick 165, float64, cuda.  A is assembled with
finject.assemble_inj, which overwrites p.F after every mpm_strain with an injected value, so the F
that A sees is an INPUT and can be perturbed exactly.

  assembly base theta = 0        (the campaign's; what the recovery actually uses)
  assembly base theta = theta_true, forward difference   (the local Jacobian AT the operating
                                                          point, where the stress and hence the
                                                          kink active set are the true ones)

  F variants, each [n_sub, Np, 2, 2]:
     true            the reference per-substep F                              (control)
     lerp_true       lerp(F(t), F(t+1)) -- two exact frame measurements       (realizable, no noise)
     sF_indep        true + coherent N(0, 3.9e-3) per component, i.i.d. in space
     sF_grid48       true + coherent N(0, 3.9e-3) smoothed on a 48-node grid  (the recording's
                                                                               spatial structure)
     lerp_sF_indep   the campaign's convention: noisy F(t), noisy F(t+1), lerped
     derived_15px    F = I + grad u from a boxcar/central-difference control grid at 15 px,
     derived_34px    the SAME derivation at 34 px                             (attenuation 0.003)
     shatter_I       F := I everywhere      -- an infinite F error; the E columns must go to zero
     shatter_1.5x    F := 1.5 * F_true      -- a 50% F error

  Reported per variant, SEPARATELY for the E block and the gain block:
     block ||dA||_F / ||A||_F, per-column ||dA_col||/||A_col|| (median/p90/max),
     relative change of the column norm, and the interior/wall split of dA
     -> the attenuation of the RECOVERED parameter: slope of theta_hat on theta_true, through the
        origin and by OLS, plus mean(hat)/mean(true) and the Pearson r, for the E half and the gain
        half; from the joint solve, and from a gain-only solve (E held at truth, and E held at a
        constant 130 = no per-cell knowledge of E).

  at n_sub = 1 (one substep) and n_sub = 10 (one FRAME).

usage:
  PYTHONPATH=/workspace/Plexus/src python p1b_gaincol.py --device cuda:1
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

from recover import Solver, theta_scale                              # noqa: E402
from finject import lerp, assemble_inj, y_of, record_substeps        # noqa: E402
from freal_derivedF import (ControlGrid, derive_F, plant_and_warm_x0,  # noqa: E402
                            PX, attenuation as F_attenuation)
from refute5_fit import NoiseF                                       # noqa: E402
from round5_fit import SIGMA_F                                       # noqa: E402
import metrics as MET                                                # noqa: E402


# --------------------------------------------------------------------------------------------- #
#  assembly at an arbitrary base theta (the campaign's assemble_inj is the base-0 special case)
# --------------------------------------------------------------------------------------------- #
def assemble_at(sy, n_sub, base, injF=None, sE=100.0, sg=1.0):
    """A [2Np, 2C], y_base [2Np]:  column j = (y(base + s e_j) - y(base)) / s.

    For an exactly affine map this is base-independent; near the wall kinks it is not, which is the
    whole reason for offering base = theta_true as well as base = 0.
    """
    t0 = time.time()
    y0 = y_of(sy, base, n_sub, injF, None)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = base.clone()
        e[j] = e[j] + s
        A[:, j] = (y_of(sy, e, n_sub, injF, None) - y0) / s
    torch.cuda.synchronize()
    return A, y0, time.time() - t0


# --------------------------------------------------------------------------------------------- #
#  the two read-outs
# --------------------------------------------------------------------------------------------- #
def col_compare(A, Aref, C, int_mask_flat):
    """Block- and column-wise change of A relative to Aref, split E / gain."""
    out = {}
    dA = A - Aref
    for tag, sl in (("E", slice(0, C)), ("gain", slice(C, 2 * C))):
        a, r, d = A[:, sl], Aref[:, sl], dA[:, sl]
        na, nr, nd = a.norm(dim=0), r.norm(dim=0), d.norm(dim=0)
        good = nr > 0
        relcol = torch.where(good, nd / nr.clamp(min=1e-300), torch.full_like(nd, float("nan")))
        dn = torch.where(good, (na - nr).abs() / nr.clamp(min=1e-300),
                         torch.full_like(nd, float("nan")))
        q = lambda t, p: float(torch.nanquantile(t, p))
        rec = {"block_rel_fro": float(d.norm() / max(float(r.norm()), 1e-300)),
               "colnorm_ref_med": float(nr.median()), "colnorm_new_med": float(na.median()),
               "colnorm_relchange_med": q(dn, 0.5), "colnorm_relchange_p90": q(dn, 0.9),
               "colnorm_relchange_max": float(torch.nan_to_num(dn, nan=-1).max()),
               "dcol_over_col_med": q(relcol, 0.5), "dcol_over_col_p90": q(relcol, 0.9),
               "dcol_over_col_max": float(torch.nan_to_num(relcol, nan=-1).max()),
               "n_zero_ref_cols": int((~good).sum())}
        if float(d.norm()) > 0:
            rec["frac_of_dA_sq_on_wall_particles"] = float(
                d[~int_mask_flat].pow(2).sum() / d.pow(2).sum())
            rec["block_rel_fro_interior"] = float(
                d[int_mask_flat].norm() / max(float(r[int_mask_flat].norm()), 1e-300))
        else:
            rec["frac_of_dA_sq_on_wall_particles"] = float("nan")
            rec["block_rel_fro_interior"] = 0.0
        out[tag] = rec
    return out


def atten(hat, true):
    """The attenuation of a recovered half: slope of hat on true, several ways."""
    hat, true = hat.double(), true.double()
    hm, tm = hat - hat.mean(), true - true.mean()
    sl = float((hm @ tm) / (tm @ tm))
    return {"slope_origin": float((hat @ true) / (true @ true)),
            "slope_ols": sl, "intercept_ols": float(hat.mean() - sl * true.mean()),
            "mean_ratio": float(hat.mean() / true.mean()),
            "corr": float((hm @ tm) / (hm.norm() * tm.norm() + 1e-300)),
            "med_abs_rel_err": float(((hat - true) / true).abs().median()),
            "n_negative": int((hat < 0).sum())}


def solve_and_score(A, b, C, th, key="ridge0"):
    """Joint solve + two gain-only solves, all scored by attenuation."""
    out = {}
    try:
        S = Solver(A, C)
        sol = S(b)
        out["cond"] = S.cond
        for k in (key, "ridge1e-08", "ridge0.0001"):
            if k in sol:
                out[k] = {"E": atten(sol[k][:C], th[:C]), "gain": atten(sol[k][C:], th[C:])}
        out["fit_residual"] = float((A.double() @ sol[key].double() - b.double()).norm()
                                    / max(float(b.double().norm()), 1e-300))
        S.free()
        del S
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:180]}"
    torch.cuda.empty_cache()

    # gain-only: the E half is not being asked for at all.  The elastic part of the observation
    # still has to be removed, and it is removed with THIS variant's own (perturbed) E columns --
    # which is what an experimenter who only wants gain would actually be forced to do.
    Ag = A[:, C:].double()
    Ae = A[:, :C].double()
    Gg = Ag.T @ Ag
    for tag, Ek in (("gain_only_E_oracle", th[:C].double()),
                    ("gain_only_E_const130", torch.full_like(th[:C].double(), 130.0))):
        try:
            bg = b.double() - Ae @ Ek
            gh = torch.linalg.solve(Gg, Ag.T @ bg)
            out[tag] = atten(gh, th[C:])
            out[tag]["fit_residual"] = float((Ag @ gh - bg).norm() / max(float(bg.norm()), 1e-300))
        except Exception as e:
            out[tag] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    # THE PIVOT'S OWN MODEL: one GLOBAL E (P0: uniform E moves peak_excursion 12% over a 40x range,
    # non-monotonically) plus C per-cell gains.  The E half is deliberately misspecified, so this is
    # biased even at the true F -- the number that matters is how much WORSE it gets when F is wrong.
    try:
        Ap = torch.cat([Ae.sum(1, keepdim=True), Ag], 1)
        ph = torch.linalg.solve(Ap.T @ Ap, Ap.T @ b.double())
        out["pivot_uniformE"] = atten(ph[1:], th[C:])
        out["pivot_uniformE"]["E_global_hat"] = float(ph[0])
        out["pivot_uniformE"]["E_true_mean"] = float(th[:C].mean())
        out["pivot_uniformE"]["fit_residual"] = float(
            (Ap @ ph - b.double()).norm() / max(float(b.double().norm()), 1e-300))
        del Ap
    except Exception as e:
        out["pivot_uniformE"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    out["cond_gain_block"] = float(torch.linalg.cond(Gg))
    del Ag, Ae, Gg
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1b_gaincol")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--nsub", default="1,10")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base-true", action="store_true", default=True,
                    help="also assemble at base = theta_true (the local Jacobian)")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.t0, window=150, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "args": vars(a), "sigma_F": SIGMA_F, "px_world": PX}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, REF = plant_and_warm_x0(args, log)
        C, n, dx = sy.C, sy.n_sub_per_frame, sy.g.dx
        th = sy.theta_true.double()
        x0 = sy.x0.clone()
        I2 = torch.eye(2, device=sy.device, dtype=sy.dtype)

        Fs, Cs, Xs = record_substeps(sy, n)         # the TRUE per-substep F, C, x under theta_true
        F0, F1 = sy.F0.clone(), Fs[-1].clone()
        x_next = Xs[-1].clone()

        band = 0.06 / MET.SHEET_SPAN
        interior = ~((x0[:, 0] < band) | (x0[:, 0] > 1 - band)
                     | (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        int_flat = interior[:, None].expand(-1, 2).reshape(-1)
        R["n_interior"] = int(interior.sum())
        R["Np"] = sy.Np
        R["C"] = C
        log(f"[state] tick {a.t0}: |F0 - I| med {float((F0-I2).abs().median()):.4e}, "
            f"|dF|/|F| over the frame {float((F1-F0).norm()/F0.norm()):.3e}; interior "
            f"{int(interior.sum())}/{sy.Np}; ||act0|| {float(sy.act0.norm()):.4g}, "
            f"||pass0|| {float(sy.pass0.norm()):.4g}")

        # -------- ANALYTIC CROSS-CHECK, cheap and exact: is act0 a function of F? ---------------
        # act0 is recomputed by sy._outer from the CURRENT particle positions.  Change p.F wildly,
        # leave positions alone, recompute -> if the active force moves at all, the analytic claim
        # is wrong.  (pass0 = drag = -k*vel is likewise position/velocity only.)
        F_save = sy.p.F.clone()
        sy.restore()
        a_ref, p_ref = sy._outer(a.t0, gain_cell=sy.gain_true)
        sy.p.F = (2.5 * F_save).clone()
        a_wild, p_wild = sy._outer(a.t0, gain_cell=sy.gain_true)
        sy.p.F = F_save
        R["act0_depends_on_F"] = {
            "max_abs_diff_act0_when_F_scaled_2.5x": float((a_wild - a_ref).abs().max()),
            "max_abs_diff_passive": float((p_wild - p_ref).abs().max()),
            "norm_act0": float(a_ref.norm())}
        log(f"[analytic] scaling p.F by 2.5x changes the active-force delta by "
            f"{R['act0_depends_on_F']['max_abs_diff_act0_when_F_scaled_2.5x']:.3e} "
            f"(||act0|| = {R['act0_depends_on_F']['norm_act0']:.4g}) and the drag delta by "
            f"{R['act0_depends_on_F']['max_abs_diff_passive']:.3e}")
        sy.restore()

        # -------- the F variants ----------------------------------------------------------------
        gen = torch.Generator(device=sy.device).manual_seed(a.seed)
        nz_indep = NoiseF("indep", x0, 48, sy.device, sy.dtype)
        nz_grid = NoiseF("gridsm", x0, 48, sy.device, sy.dtype)
        e_ind = SIGMA_F * nz_indep(gen)                      # [Np,2,2], coherent across the frame
        e_grd = SIGMA_F * nz_grid(gen)
        e_ind0 = (SIGMA_F / 2.0) * nz_indep(gen)
        e_ind1 = (SIGMA_F / 2.0) * nz_indep(gen)

        X0ref = REF[0]["x"]
        cg15 = ControlGrid(X0ref, 15.0 * PX)
        cg34 = ControlGrid(X0ref, 34.0 * PX)
        R["control_grids"] = {
            "h15px": {"nodes": cg15.n_nodes, "valid": cg15.n_valid,
                      "pts_per_valid": cg15.pts_per_valid, "cd_honest": cg15.n_cd_honest},
            "h34px": {"nodes": cg34.n_nodes, "valid": cg34.n_valid,
                      "pts_per_valid": cg34.pts_per_valid, "cd_honest": cg34.n_cd_honest}}
        Fref0 = REF[0]["F"]

        def der(cg, x):
            return derive_F(cg, X0ref, x, "bilinear", F_ref=Fref0)

        d15_0, d15_1 = der(cg15, x0), der(cg15, x_next)
        d34_0, d34_1 = der(cg34, x0), der(cg34, x_next)

        VAR = [
            ("true",           Fs),
            ("lerp_true",      lerp(F0, F1, n)),
            ("sF_indep",       Fs + e_ind[None]),
            ("sF_grid48",      Fs + e_grd[None]),
            ("lerp_sF_indep",  lerp(F0 + e_ind0, F1 + e_ind1, n)),
            ("derived_15px",   lerp(d15_0, d15_1, n)),
            ("derived_34px",   lerp(d34_0, d34_1, n)),
            ("shatter_I",      I2[None, None].expand(n, sy.Np, 2, 2).contiguous()),
            ("shatter_1.5x",   1.5 * Fs),
        ]

        # how big is each F error, in the recording's own currency
        R["F_error"] = {}
        log(f"\n[F error] how far each injected F is from the true per-substep F "
            f"(sigma_F = {SIGMA_F:g}; median |F_true - I| = "
            f"{float((Fs - I2).abs().median()):.4e})")
        log(f"    {'variant':<15s} {'med|dF|':>10s} {'rms|dF|':>10s} {'relFro':>9s} "
            f"{'ls_scale(4ch) on F-I':>34s}")
        for name, Fv in VAR:
            d = (Fv - Fs).abs()
            at = F_attenuation(Fv[0], Fs[0], interior)
            R["F_error"][name] = {"med_abs": float(d.median()), "rms_abs": float(d.pow(2).mean().sqrt()),
                                  "rel_fro": float((Fv - Fs).norm() / Fs.norm()),
                                  "substep0_vs_true": at}
            log(f"    {name:<15s} {float(d.median()):>10.3e} "
                f"{float(d.pow(2).mean().sqrt()):>10.3e} "
                f"{float((Fv-Fs).norm()/Fs.norm()):>9.3e} "
                f"{str([round(v,4) for v in at['ls_scale_trim99']]):>34s}")

        # -------- the sweep ----------------------------------------------------------------------
        R["sweep"] = {}
        for ns in [int(v) for v in a.nsub.split(",")]:
            y_obs = (Xs[ns - 1] - x0).reshape(-1)
            bases = [("base0", torch.zeros(2 * C, device=sy.device, dtype=sy.dtype))]
            if a.base_true:
                bases.append(("baseTrue", sy.theta_true.clone()))
            for bname, base in bases:
                sE, sg = (100.0, 1.0) if bname == "base0" else (1.0, 0.01)
                key = f"nsub{ns}_{bname}"
                R["sweep"][key] = {"n_sub": ns, "base": bname, "step_E": sE, "step_gain": sg,
                                   "variants": {}}
                Aref = yref = None
                log(f"\n[nsub={ns}, base={bname}] assembling {len(VAR)} variants "
                    f"(step_E={sE:g}, step_gain={sg:g})")
                log(f"    {'variant':<15s} | {'E: dA/A':>9s} {'dcol med':>9s} {'|col| ch':>9s} | "
                    f"{'g: dA/A':>9s} {'dcol med':>9s} {'|col| ch':>9s} | {'wall frac(g)':>12s} "
                    f"| {'E/g ratio':>14s}")
                for name, Fv in VAR:
                    A, y0, t_as = assemble_at(sy, ns, base, Fv[:ns], sE=sE, sg=sg)
                    # dimensionless column norms: the response to a 1% change in ONE cell's
                    # parameter, relative to the whole theta-dependent signal ||b||
                    Adim = A * sy.theta_true[None, :]
                    nb = max(float((y_obs - y0).norm()), 1e-300)
                    rec = {"assembly_s": t_as,
                           "colnorm_E_med": float(A[:, :C].norm(dim=0).median()),
                           "colnorm_gain_med": float(A[:, C:].norm(dim=0).median()),
                           "signal_1pct_over_b_E_med": float(
                               0.01 * Adim[:, :C].norm(dim=0).median() / nb),
                           "signal_1pct_over_b_gain_med": float(
                               0.01 * Adim[:, C:].norm(dim=0).median() / nb)}
                    del Adim
                    if name == "true":
                        Aref, yref = A.clone(), y0.clone()
                    cc = col_compare(A, Aref, C, int_flat)
                    rec["cols"] = cc
                    # model bias of this injected model at theta_true
                    y_self = y_of(sy, sy.theta_true, ns, Fv[:ns], None)
                    rec["model_bias_vs_obs"] = float((y_self - y_obs).norm() / y_obs.norm())
                    # ---- recovery, base-0 assembly only (base-true recovery starts at the truth)
                    if bname == "base0":
                        b = y_obs - y0
                        rec["recovery"] = solve_and_score(A, b, C, th)
                    R["sweep"][key]["variants"][name] = rec
                    e, g = cc["E"], cc["gain"]
                    ratio = (e["block_rel_fro"] / g["block_rel_fro"]
                             if g["block_rel_fro"] > 0 else float("inf"))
                    rec["E_over_gain_sensitivity_ratio"] = ratio
                    log(f"    {name:<15s} | {e['block_rel_fro']:>9.2e} "
                        f"{e['dcol_over_col_med']:>9.2e} {e['colnorm_relchange_med']:>9.2e} | "
                        f"{g['block_rel_fro']:>9.2e} {g['dcol_over_col_med']:>9.2e} "
                        f"{g['colnorm_relchange_med']:>9.2e} | "
                        f"{g['frac_of_dA_sq_on_wall_particles']:>12.4f} | "
                        f"E/g {ratio:>10.3e}")
                    del A
                    torch.cuda.empty_cache()
                del Aref, yref
                torch.cuda.empty_cache()

                # ---- the headline table -----------------------------------------------------
                if bname == "base0":
                    log(f"\n    [attenuation, nsub={ns}] slope of theta_hat on theta_true "
                        f"(1.000 = perfect; joint solve, ridge0)")
                    log(f"    {'variant':<15s} | {'E slope':>9s} {'E mean_r':>9s} {'E corr':>8s} "
                        f"{'E medErr':>9s} | {'g slope':>9s} {'g mean_r':>9s} {'g corr':>8s} "
                        f"{'g medErr':>9s} | {'g-only(E*)':>10s} {'g-only(130)':>11s}")
                    for name, _ in VAR:
                        rv = R["sweep"][key]["variants"][name].get("recovery", {})
                        if "ridge0" not in rv:
                            log(f"    {name:<15s} | {rv.get('error','(no solve)')}")
                            continue
                        E_, g_ = rv["ridge0"]["E"], rv["ridge0"]["gain"]
                        go = rv.get("gain_only_E_oracle", {})
                        gc = rv.get("gain_only_E_const130", {})
                        pv = rv.get("pivot_uniformE", {})
                        log(f"    {name:<15s} | {E_['slope_ols']:>9.4f} {E_['mean_ratio']:>9.4f} "
                            f"{E_['corr']:>8.4f} {E_['med_abs_rel_err']:>9.4f} | "
                            f"{g_['slope_ols']:>9.4f} {g_['mean_ratio']:>9.4f} "
                            f"{g_['corr']:>8.4f} {g_['med_abs_rel_err']:>9.4f} | "
                            f"{go.get('slope_ols', float('nan')):>10.4f} "
                            f"{gc.get('slope_ols', float('nan')):>11.4f} | "
                            f"pivot g slope {pv.get('slope_ols', float('nan')):>8.4f} "
                            f"corr {pv.get('corr', float('nan')):>7.4f} "
                            f"medErr {pv.get('med_abs_rel_err', float('nan')):>7.4f}")

        # ---------------------------------------------------------------- THE HEADLINE ----------
        log("\n" + "=" * 100)
        log("HEADLINE -- attenuation of the GAIN half next to the E half, under REALIZABLE F error")
        log("=" * 100)
        H = {}
        for key, blk in R["sweep"].items():
            if blk["base"] != "base0":
                continue
            for name in ("true", "lerp_true", "sF_indep", "sF_grid48", "derived_15px",
                         "derived_34px"):
                rv = blk["variants"].get(name, {}).get("recovery", {})
                if "ridge0" not in rv:
                    continue
                H[f"{key}::{name}"] = {
                    "E_slope_ols": rv["ridge0"]["E"]["slope_ols"],
                    "E_mean_ratio": rv["ridge0"]["E"]["mean_ratio"],
                    "gain_slope_ols": rv["ridge0"]["gain"]["slope_ols"],
                    "gain_mean_ratio": rv["ridge0"]["gain"]["mean_ratio"],
                    "gain_corr": rv["ridge0"]["gain"]["corr"],
                    "pivot_gain_slope_ols": rv.get("pivot_uniformE", {}).get("slope_ols"),
                    "colchange_E": blk["variants"][name]["cols"]["E"]["block_rel_fro"],
                    "colchange_gain": blk["variants"][name]["cols"]["gain"]["block_rel_fro"]}
                v = H[f"{key}::{name}"]
                log(f"  {key:<20s} {name:<14s}  E slope {v['E_slope_ols']:>9.4f}   "
                    f"GAIN slope {v['gain_slope_ols']:>8.4f}  (gain corr {v['gain_corr']:.4f}, "
                    f"pivot gain slope "
                    f"{(v['pivot_gain_slope_ols'] if v['pivot_gain_slope_ols'] is not None else float('nan')):.4f})")
        R["HEADLINE"] = H

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

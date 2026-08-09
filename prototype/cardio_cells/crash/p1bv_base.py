"""p1bv_base.py -- ADVERSARIAL VERIFICATION of probe B ("the gain columns of A are exactly F-free").

THE SUSPICION
====================================================================================================
Probe B's decisive numbers (max|dA_gain| = 0.000e+00 bitwise, max|dy0| = 0.000e+00, "100% of the
gain damage comes from the E columns") are all taken at ONE linearization point: base theta = 0.
At theta = 0, _lame(0) = (0, 0) (entities.py:36-40), so `stress = 2 mu (F-R) F^T + I la J (J-1)`
(mpm_scatter.py:112-113) is IDENTICALLY ZERO at every substep and the injected F is multiplied by
nothing.  The bitwise zero is therefore a property of the BASE POINT, not of the gain column, and
"the only channel F has is A_E" is true by construction of that base point rather than measured.

Probe B's own base = theta_true table already shows the gain block moving 9.28e-02 (derived F,
15 px, one frame) once the elastic path is switched on -- but no RECOVERY was ever run at a
non-degenerate base, so the attribution was never re-tested where F can reach the gain columns.

WHAT THIS RUNS
----------------------------------------------------------------------------------------------------
The identical system (crash_test / freal_derivedF plant, seed 2026, C=100, float64, one FRAME =
10 substeps), the identical F variants, but the linearization base is swept:

  base0      theta = 0                              (probe B's; degenerate: mu = la = 0)
  baseMid    E = 130 uniform, gain = 1.0 uniform    (a Gauss-Newton step from a sane guess; the
                                                     elastic path is ON, so F reaches everything)
  baseTrue   theta = theta_true                     (the local Jacobian at the operating point)

At each base and each F variant it measures
  (i)   max|dA_gain| and ||dA_gain||/||A_gain||, and the SAME for y(base) -- probe B's two
        "exactly zero" quantities, re-measured off the degenerate base;
  (ii)  the recovery of the FULL theta (base + solved increment), scored exactly as probe B scores
        it (atten: OLS slope, Pearson r, med|rel err|), column-scaled ridge0;
  (iii) HYBRID assemblies that put the derived F in ONE place at a time -- E columns only, gain
        columns only, offset y(base) only -- which is the fair version of probe B's attribution
        claim.  Probe B could not run this: at its base all three are bitwise identical.
  (iv)  the drop_E ceiling (E block refused) at each base and each F, to check probe B's
        "with a PERFECT F ... gain r = 0.157" is a real F comparison and not an F-free solve.

and the whole thing at TWO ticks (probe B ran only tick 165), to test the phase claim.

usage: PYTHONPATH=/workspace/Plexus/src python p1bv_base.py --device cuda:1 --ticks 165,100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

from recover import theta_scale                                       # noqa: E402
from finject import lerp, y_of, record_substeps                       # noqa: E402
from freal_derivedF import ControlGrid, derive_F, plant_and_warm_x0, PX   # noqa: E402
from refute5_fit import NoiseF                                        # noqa: E402
from round5_fit import SIGMA_F                                        # noqa: E402
from p1b_gaincol import atten, assemble_at                            # noqa: E402


def solve_scaled(A, b, C, drop_E=False):
    """Column-scaled ridge0 normal equations -- byte-for-byte probe B's block_ridge(lamE=0)."""
    s = theta_scale(C, A.device)
    Az = A.double() * s[None, :]
    if drop_E:
        Ag = Az[:, C:]
        z = torch.linalg.solve(Ag.T @ Ag, Ag.T @ b.double())
        out = torch.zeros(2 * C, device=A.device, dtype=torch.float64)
        out[C:] = z * s[C:]
        res = float((Ag @ z - b.double()).norm() / max(float(b.double().norm()), 1e-300))
        return out, res
    G = Az.T @ Az
    z = torch.linalg.solve(G, Az.T @ b.double())
    res = float((Az @ z - b.double()).norm() / max(float(b.double().norm()), 1e-300))
    return z * s, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1bv_base")
    ap.add_argument("--ticks", default="165,100")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--nsub", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"args": vars(a), "sigma_F": SIGMA_F, "ticks": {}}
    t_start = time.time()
    torch.manual_seed(0)

    for t0 in [int(v) for v in a.ticks.split(",")]:
        args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                               n_grid=a.n_grid, warmup=t0, window=150, dtype="float64",
                               mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
        with torch.no_grad():
            sy, REF = plant_and_warm_x0(args, log)
            C, n = sy.C, a.nsub
            th = sy.theta_true.double()
            x0 = sy.x0.clone()
            Fs, Cs, Xs = record_substeps(sy, sy.n_sub_per_frame)
            x_next = Xs[-1].clone()
            y_obs = (Xs[n - 1] - x0).reshape(-1)

            gen = torch.Generator(device=sy.device).manual_seed(a.seed)
            e_ind = SIGMA_F * NoiseF("indep", x0, 48, sy.device, sy.dtype)(gen)
            cg15 = ControlGrid(REF[0]["x"], 15.0 * PX)
            d0 = derive_F(cg15, REF[0]["x"], x0, "bilinear", F_ref=REF[0]["F"])
            d1 = derive_F(cg15, REF[0]["x"], x_next, "bilinear", F_ref=REF[0]["F"])
            VAR = [("true", Fs), ("sF_indep", Fs + e_ind[None]),
                   ("derived_15px", lerp(d0, d1, sy.n_sub_per_frame))]

            zero = torch.zeros(2 * C, device=sy.device, dtype=sy.dtype)
            mid = zero.clone()
            mid[:C] = 130.0
            mid[C:] = 1.0
            BASES = [("base0", zero, 100.0, 1.0),
                     ("baseMid", mid, 1.0, 0.01),
                     ("baseMid_secant", mid, 100.0, 1.0),
                     ("baseTrue", sy.theta_true.clone(), 1.0, 0.01)]

            TR = {}
            for bname, base, sE, sg in BASES:
                store = {}
                for name, Fv in VAR:
                    A, y0, t_as = assemble_at(sy, n, base, Fv[:n], sE=sE, sg=sg)
                    store[name] = (A, y0)
                At, y0t = store["true"]
                rec = {"step_E": sE, "step_gain": sg, "norm_A_gain": float(At[:, C:].norm()),
                       "norm_A_E": float(At[:, :C].norm()), "norm_y0": float(y0t.norm()),
                       "variants": {}}
                log(f"\n[tick {t0}] base={bname} (step_E={sE:g}, step_gain={sg:g})  "
                    f"||A_gain|| {float(At[:,C:].norm()):.6g}  ||y(base)|| {float(y0t.norm()):.6g}")
                log(f"    {'variant':<14s} | {'max|dA_gain|':>12s} {'rel dA_gain':>12s} | "
                    f"{'rel dA_E':>10s} | {'max|dy0|':>11s} {'rel dy0':>10s}")
                for name, _ in VAR:
                    A, y0 = store[name]
                    dg = A[:, C:] - At[:, C:]
                    dE = A[:, :C] - At[:, :C]
                    dy = y0 - y0t
                    v = {"max_abs_dA_gain": float(dg.abs().max()),
                         "rel_dA_gain": float(dg.norm() / At[:, C:].norm()),
                         "max_abs_dA_E": float(dE.abs().max()),
                         "rel_dA_E": float(dE.norm() / At[:, :C].norm()),
                         "max_abs_dy0": float(dy.abs().max()),
                         "rel_dy0": float(dy.norm() / max(float(y0t.norm()), 1e-300))}
                    rec["variants"][name] = v
                    log(f"    {name:<14s} | {v['max_abs_dA_gain']:>12.3e} "
                        f"{v['rel_dA_gain']:>12.3e} | {v['rel_dA_E']:>10.3e} | "
                        f"{v['max_abs_dy0']:>11.3e} {v['rel_dy0']:>10.3e}")

                # ---------- recovery of the FULL theta, base + increment ------------------------
                log(f"    [recovery, base={bname}] theta_hat = base + ridge0 increment")
                log(f"    {'assembly':<26s} | {'E slope':>9s} {'E corr':>8s} {'E medErr':>9s} | "
                    f"{'g slope':>9s} {'g corr':>8s} {'g medErr':>9s} | {'fit res':>8s}")
                combos = [("all_true", "true", "true", "true"),
                          ("all_sF_indep", "sF_indep", "sF_indep", "sF_indep"),
                          ("all_derived15", "derived_15px", "derived_15px", "derived_15px"),
                          ("Eonly_derived15", "derived_15px", "true", "true"),
                          ("gainonly_derived15", "true", "derived_15px", "true"),
                          ("y0only_derived15", "true", "true", "derived_15px")]
                rec["recovery"] = {}
                for cname, kE, kg, ky in combos:
                    A = torch.cat([store[kE][0][:, :C], store[kg][0][:, C:]], 1)
                    b = y_obs - store[ky][1]
                    try:
                        d, res = solve_scaled(A, b, C)
                        hat = base.double() + d
                        rr = {"E": atten(hat[:C], th[:C]), "gain": atten(hat[C:], th[C:]),
                              "fit_residual": res}
                    except Exception as e:
                        rr = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                    try:
                        dd, resd = solve_scaled(A, b, C, drop_E=True)
                        hatd = base.double() + dd
                        rr["drop_E"] = atten(hatd[C:], th[C:])
                        rr["drop_E"]["fit_residual"] = resd
                    except Exception as e:
                        rr["drop_E"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                    rec["recovery"][cname] = rr
                    if "error" in rr:
                        log(f"    {cname:<26s} | {rr['error']}")
                    else:
                        E_, g_ = rr["E"], rr["gain"]
                        de = rr["drop_E"]
                        log(f"    {cname:<26s} | {E_['slope_ols']:>9.4f} {E_['corr']:>8.4f} "
                            f"{E_['med_abs_rel_err']:>9.4f} | {g_['slope_ols']:>9.4f} "
                            f"{g_['corr']:>8.4f} {g_['med_abs_rel_err']:>9.4f} | "
                            f"{rr['fit_residual']:>8.4f}   dropE: g slope "
                            f"{de.get('slope_ols', float('nan')):>8.4f} corr "
                            f"{de.get('corr', float('nan')):>7.4f} medErr "
                            f"{de.get('med_abs_rel_err', float('nan')):>7.4f}")
                    del A
                    torch.cuda.empty_cache()
                TR[bname] = rec
                store.clear()
                del At, y0t
                torch.cuda.empty_cache()

            R["ticks"][str(t0)] = TR
            del sy, Fs, Cs, Xs
            torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

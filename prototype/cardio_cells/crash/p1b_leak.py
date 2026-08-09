"""p1b_leak.py -- PROBE B, part 2: if the gain columns are clean, why is the recovered gain not?

p1b_gaincol.py established, at the campaign's own base point (theta = 0, finject.assemble_inj):

    the GAIN columns of A are BIT-IDENTICAL under every F perturbation tried, including F := I and
    F := 1.5 F_true, at 1 substep AND at 10 substeps (one frame);
    the E columns change by ||dA_E||/||A_E|| = 10.2 (derived F at 15 px, 1 substep);

and yet the recovered per-cell gain collapses anyway (Pearson r on the gain half falls 0.9995 with
the true F to 0.183 with the derived F).  Two things could do that, and they have different
consequences for the pivot:

  (1) the right-hand side b = y_obs - y(theta = 0) moved.  It cannot: at theta = 0 the Lame
      coefficients are zero, the fixed-corotated stress is identically zero at every substep, and
      the injected F is multiplied by nothing -- so y(0) should be byte-identical across variants.
      MEASURED here rather than argued.
  (2) the corrupted E columns leak into the gain estimate through the cross-Gram A_E^T A_g.
      If so, the leak is a property of the JOINT SOLVE, not of the gain column, and the question
      becomes whether any estimator that stops asking for per-cell E lets the gain through.

WHAT IS RUN
----------------------------------------------------------------------------------------------------
  * exact equality checks on y(0) and on the gain block of A across F variants;
  * a BLOCK RIDGE sweep: lambda applied to the E block only, from 0 to 1e6, plus the limit
    "E block dropped entirely" (solve the gain columns alone against the full b).  The gain block
    is never penalised.  Reported: slope of gain_hat on gain_true, Pearson r, med|dg/g|, and the
    same for E, at every lambda_E, for the true F and for two realizable F errors;
  * the control: the identical sweep with the TRUE F, so that "ridge helps" and "ridge hides the
    question" can be told apart.

usage: PYTHONPATH=/workspace/Plexus/src python p1b_leak.py --device cuda:1
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
from finject import lerp, assemble_inj, y_of, record_substeps         # noqa: E402
from freal_derivedF import ControlGrid, derive_F, plant_and_warm_x0, PX   # noqa: E402
from refute5_fit import NoiseF                                        # noqa: E402
from round5_fit import SIGMA_F                                        # noqa: E402
from p1b_gaincol import atten                                         # noqa: E402


def block_ridge(A, b, C, lamE, lamg=0.0, drop_E=False):
    """Column-scaled normal equations with a separate ridge on each block.

    drop_E: solve the gain columns alone against the full b (the lamE -> infinity limit).
    """
    s = theta_scale(C, A.device)
    Az = A.double() * s[None, :]
    if drop_E:
        Ag = Az[:, C:]
        z = torch.linalg.solve(Ag.T @ Ag, Ag.T @ b.double())
        th = torch.zeros(2 * C, device=A.device, dtype=torch.float64)
        th[C:] = z * s[C:]
        res = float((Ag @ z - b.double()).norm() / max(float(b.double().norm()), 1e-300))
        return th, res
    G = Az.T @ Az
    tr = float(torch.diagonal(G).sum() / G.shape[0])
    d = torch.zeros(2 * C, device=A.device, dtype=torch.float64)
    d[:C] = lamE * tr
    d[C:] = lamg * tr
    z = torch.linalg.solve(G + torch.diag(d), Az.T @ b.double())
    res = float((Az @ z - b.double()).norm() / max(float(b.double().norm()), 1e-300))
    return z * s, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1b_leak")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--nsub", default="1,10")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.t0, window=150, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "args": vars(a), "sigma_F": SIGMA_F}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, REF = plant_and_warm_x0(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        x0 = sy.x0.clone()
        I2 = torch.eye(2, device=sy.device, dtype=sy.dtype)
        Fs, Cs, Xs = record_substeps(sy, n)
        F0, F1 = sy.F0.clone(), Fs[-1].clone()
        x_next = Xs[-1].clone()

        gen = torch.Generator(device=sy.device).manual_seed(a.seed)
        e_ind = SIGMA_F * NoiseF("indep", x0, 48, sy.device, sy.dtype)(gen)
        X0ref, Fref0 = REF[0]["x"], REF[0]["F"]
        cg15 = ControlGrid(X0ref, 15.0 * PX)
        d15_0 = derive_F(cg15, X0ref, x0, "bilinear", F_ref=Fref0)
        d15_1 = derive_F(cg15, X0ref, x_next, "bilinear", F_ref=Fref0)

        VAR = [("true", Fs),
               ("sF_indep", Fs + e_ind[None]),
               ("derived_15px", lerp(d15_0, d15_1, n))]

        R["sweep"] = {}
        for ns in [int(v) for v in a.nsub.split(",")]:
            y_obs = (Xs[ns - 1] - x0).reshape(-1)
            key = f"nsub{ns}"
            R["sweep"][key] = {}
            store = {}
            for name, Fv in VAR:
                A, y0, t_as = assemble_inj(sy, ns, Fv[:ns], None)
                store[name] = (A, y0)

            # ---- (1) is b the same across variants? ------------------------------------------
            y0t = store["true"][1]
            Agt = store["true"][0][:, C:]
            eq = {}
            for name, _ in VAR:
                A, y0 = store[name]
                eq[name] = {
                    "max_abs_dy0": float((y0 - y0t).abs().max()),
                    "max_abs_dA_gain": float((A[:, C:] - Agt).abs().max()),
                    "max_abs_dA_E": float((A[:, :C] - store["true"][0][:, :C]).abs().max()),
                    "rel_dA_E": float((A[:, :C] - store["true"][0][:, :C]).norm()
                                      / store["true"][0][:, :C].norm()),
                    "norm_y0": float(y0.norm()), "norm_A_gain": float(Agt.norm())}
            R["sweep"][key]["exact_equality"] = eq
            log(f"\n[nsub={ns}] EXACT EQUALITY of the base-0 assembly across F variants "
                f"(||y0|| = {float(y0t.norm()):.6g}, ||A_gain|| = {float(Agt.norm()):.6g})")
            for name, v in eq.items():
                log(f"    {name:<14s} max|dy0| {v['max_abs_dy0']:.3e}   "
                    f"max|dA_gain| {v['max_abs_dA_gain']:.3e}   "
                    f"max|dA_E| {v['max_abs_dA_E']:.3e}  (rel {v['rel_dA_E']:.3e})")

            # ---- (2) the block-ridge sweep ---------------------------------------------------
            LAMS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6]
            R["sweep"][key]["block_ridge"] = {}
            for name, _ in VAR:
                A, y0 = store[name]
                b = y_obs - y0
                rows = {}
                log(f"\n    [block ridge, nsub={ns}, F = {name}]  penalty on the E block only")
                log(f"    {'lambda_E':>10s} | {'E slope':>9s} {'E corr':>8s} {'E medErr':>9s} | "
                    f"{'g slope':>9s} {'g corr':>8s} {'g medErr':>9s} {'g meanrat':>9s} | "
                    f"{'fit res':>9s}")
                for lam in LAMS + ["drop_E"]:
                    try:
                        if lam == "drop_E":
                            t_hat, res = block_ridge(A, b, C, 0.0, drop_E=True)
                        else:
                            t_hat, res = block_ridge(A, b, C, lam)
                        aE, ag = atten(t_hat[:C], th[:C]), atten(t_hat[C:], th[C:])
                        rows[str(lam)] = {"E": aE, "gain": ag, "fit_residual": res}
                        log(f"    {str(lam):>10s} | {aE['slope_ols']:>9.4f} {aE['corr']:>8.4f} "
                            f"{aE['med_abs_rel_err']:>9.4f} | {ag['slope_ols']:>9.4f} "
                            f"{ag['corr']:>8.4f} {ag['med_abs_rel_err']:>9.4f} "
                            f"{ag['mean_ratio']:>9.4f} | {res:>9.4f}")
                    except Exception as e:
                        rows[str(lam)] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                        log(f"    {str(lam):>10s} | {rows[str(lam)]['error']}")
                R["sweep"][key]["block_ridge"][name] = rows
                del A
            store.clear()
            torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

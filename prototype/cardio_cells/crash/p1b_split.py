"""p1b_split.py -- PROBE B, part 3: how much of the observation is elastic, and does the gain
columns' F-immunity survive into the estimate once the elastic part is accounted for?

p1b_leak.py showed that with the TRUE F and the E block dropped, the recovered gain is exactly as
bad (r = 0.196 at one substep) as with the DERIVED F and the E block solved (r = 0.183).  That says
the damage is the un-modelled elastic response, not F.  Two numbers finish the argument:

  1. the block split of the observation:  ||A_E theta_E|| and ||A_g theta_g|| against ||b||.
     If the elastic term dominates, no estimator that refuses to model per-cell E can work,
     whatever F it is given.
  2. the CONDITIONAL gain estimate: remove the elastic contribution using the TRUE-F E columns,
     then solve for gain with the columns from each F variant.  Because the gain columns are
     bit-identical across variants (p1b_leak), this must return the same answer for every F --
     which is the immunity, stated in the units of the recovered parameter rather than of A.

usage: PYTHONPATH=/workspace/Plexus/src python p1b_split.py --device cuda:1
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

from finject import lerp, assemble_inj, record_substeps                # noqa: E402
from freal_derivedF import ControlGrid, derive_F, plant_and_warm_x0, PX   # noqa: E402
from refute5_fit import NoiseF                                         # noqa: E402
from round5_fit import SIGMA_F                                         # noqa: E402
from p1b_gaincol import atten                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1b_split")
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

    R = {"config": vars(args), "args": vars(a)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, REF = plant_and_warm_x0(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        x0 = sy.x0.clone()
        Fs, Cs, Xs = record_substeps(sy, n)
        x_next = Xs[-1].clone()

        gen = torch.Generator(device=sy.device).manual_seed(a.seed)
        e_ind = SIGMA_F * NoiseF("indep", x0, 48, sy.device, sy.dtype)(gen)
        cg15 = ControlGrid(REF[0]["x"], 15.0 * PX)
        d0 = derive_F(cg15, REF[0]["x"], x0, "bilinear", F_ref=REF[0]["F"])
        d1 = derive_F(cg15, REF[0]["x"], x_next, "bilinear", F_ref=REF[0]["F"])
        VAR = [("true", Fs), ("sF_indep", Fs + e_ind[None]), ("derived_15px", lerp(d0, d1, n))]

        R["sweep"] = {}
        for ns in [int(v) for v in a.nsub.split(",")]:
            y_obs = (Xs[ns - 1] - x0).reshape(-1)
            out = {}
            store = {nm: assemble_inj(sy, ns, Fv[:ns], None)[:2] for nm, Fv in VAR}
            At, y0t = store["true"]
            b = (y_obs - y0t).double()
            cE = (At[:, :C].double() @ th[:C])
            cg = (At[:, C:].double() @ th[C:])
            out["block_split"] = {
                "norm_b": float(b.norm()), "norm_A_E_thetaE": float(cE.norm()),
                "norm_A_g_thetag": float(cg.norm()),
                "elastic_over_b": float(cE.norm() / b.norm()),
                "gain_over_b": float(cg.norm() / b.norm()),
                "cos_between_blocks": float((cE @ cg) / (cE.norm() * cg.norm())),
                "superposition_residual": float((cE + cg - b).norm() / b.norm()),
                "norm_y0_offset": float(y0t.norm()),
                "y0_over_yobs": float(y0t.norm() / y_obs.norm())}
            s = out["block_split"]
            log(f"\n[nsub={ns}] block split of the theta-dependent signal b = y_obs - y(theta=0)")
            log(f"    ||b|| {s['norm_b']:.6g};  elastic ||A_E theta_E||/||b|| = "
                f"{s['elastic_over_b']:.4f};  active ||A_g theta_g||/||b|| = "
                f"{s['gain_over_b']:.4f};  cos(elastic, active) = "
                f"{s['cos_between_blocks']:+.4f};  superposition residual "
                f"{s['superposition_residual']:.3e}")
            log(f"    the theta-free offset y(theta=0) is {s['y0_over_yobs']:.4f} of the observed "
                f"displacement")

            # ---- the conditional gain estimate ------------------------------------------------
            b_cond = b - cE                       # elastic part removed with the TRUE-F E columns
            out["gain_conditional_on_true_elastic"] = {}
            log(f"    [conditional] gain solved with the elastic part removed using the TRUE-F "
                f"E columns; the gain columns come from each F variant")
            log(f"    {'F variant':<14s} | {'g slope':>9s} {'g corr':>9s} {'g medErr':>9s} "
                f"{'g meanrat':>9s} | {'fit res':>9s}")
            for nm, _ in VAR:
                Ag = store[nm][0][:, C:].double()
                gh = torch.linalg.solve(Ag.T @ Ag, Ag.T @ b_cond)
                rec = atten(gh, th[C:])
                rec["fit_residual"] = float((Ag @ gh - b_cond).norm() / b_cond.norm())
                out["gain_conditional_on_true_elastic"][nm] = rec
                log(f"    {nm:<14s} | {rec['slope_ols']:>9.4f} {rec['corr']:>9.6f} "
                    f"{rec['med_abs_rel_err']:>9.2e} {rec['mean_ratio']:>9.4f} | "
                    f"{rec['fit_residual']:>9.2e}")

            # ---- and the honest version: elastic removed with THAT variant's own E columns -----
            out["gain_conditional_on_own_elastic"] = {}
            log(f"    [honest]      the same, but the elastic part is removed with the variant's "
                f"OWN (measured-F) E columns -- what an experimenter could actually do")
            for nm, _ in VAR:
                A = store[nm][0]
                bb = (y_obs - store[nm][1]).double() - A[:, :C].double() @ th[:C]
                Ag = A[:, C:].double()
                gh = torch.linalg.solve(Ag.T @ Ag, Ag.T @ bb)
                rec = atten(gh, th[C:])
                rec["fit_residual"] = float((Ag @ gh - bb).norm() / bb.norm())
                out["gain_conditional_on_own_elastic"][nm] = rec
                log(f"    {nm:<14s} | {rec['slope_ols']:>9.4f} {rec['corr']:>9.6f} "
                    f"{rec['med_abs_rel_err']:>9.2e} {rec['mean_ratio']:>9.4f} | "
                    f"{rec['fit_residual']:>9.2e}")

            R["sweep"][f"nsub{ns}"] = out
            store.clear()
            torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

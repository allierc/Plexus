#!/usr/bin/env python
"""p1d_balance.py -- PROBE D, the mechanism.  WHY does pinning E wrong destroy the gain estimate?

`p1d_gainfit.py` measures that it does.  This measures why, and it is three numbers, all read off
the campaign's own design matrix at tick 165 with no fit and no rollout:

  1. SIGNAL BALANCE.  ||A_E E_true|| against ||A_g g_true||: how much of the observed displacement
     is the passive elastic response and how much is the contraction.  If the elastic term is much
     the larger, a modest fractional error in E is a large ABSOLUTE error in the prediction, and it
     has to go somewhere.

  2. CROSS-TALK.  Where it goes.  With E pinned at E_fix the gain-only normal equations return,
     exactly and with no noise anywhere,

         g_hat - g_true  =  (A_g^T A_g)^-1 A_g^T A_E (E_true - E_fix)

     so the induced gain error can be computed in closed form and compared with the fit.  The
     fraction of the pinning error that lands INSIDE the gain block's column space,
     ||A_g dg|| / ||A_E dE||, is the trade-off coefficient: 0 means the two blocks are orthogonal
     and a wrong E is simply ignored by the gain fit; 1 means the gain block can imitate the E
     block completely and the two parameters are not separately identifiable from this data.

  3. PRINCIPAL ANGLES between span(A_E) and span(A_g), which is the same statement without a
     particular error vector in it.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1d_balance.py --device cuda:1
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

from finject import lerp, assemble_inj, record_substeps                 # noqa: E402
from freal_derivedF import ControlGrid, derive_F, collect, PX, GRID_PX  # noqa: E402
from round5_fit import SNAP                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1d_balance")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--e-fix", default="132,60,240")
    ap.add_argument("--hpx", type=float, default=GRID_PX)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R, t_start = {"args": vars(a)}, time.time()
    with torch.no_grad():
        sy, REF, B = collect(args, a.t0, a.t0, log)
        C, n_frame = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        E_true, g_true = th[:C], th[C:]
        dev, f64 = th.device, torch.float64
        k0 = a.t0
        cg = ControlGrid(REF[0]["x"], a.hpx * PX)

        for n in (1, n_frame):
            for kk, vv in B[k0]["snap"].items():
                setattr(sy, kk, vv.clone())
            Fs, _, Xs = record_substeps(sy, n)
            F0, F1, xN = sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()
            injs = {"clean": lerp(F0, F1, n),
                    f"derivedF_{a.hpx:g}px": lerp(
                        derive_F(cg, REF[0]["x"], sy.x0, "bilinear", F_ref=REF[0]["F"]),
                        derive_F(cg, REF[0]["x"], xN, "bilinear", F_ref=REF[0]["F"]), n)}
            for fname, iF in injs.items():
                for kk, vv in B[k0]["snap"].items():
                    setattr(sy, kk, vv.clone())
                A, y0, _ = assemble_inj(sy, n, iF, None)
                b = (xN - sy.x0).reshape(-1) - y0
                A_E, A_g = A[:, :C].contiguous(), A[:, C:].contiguous()
                yE, yg = A_E @ E_true, A_g @ g_true
                Gg = A_g.T @ A_g
                key = f"n{n}|{fname}"
                row = {"norm_b": float(b.norm()), "norm_y0": float(y0.norm()),
                       "norm_A_E_Etrue": float(yE.norm()), "norm_A_g_gtrue": float(yg.norm()),
                       "elastic_over_active": float(yE.norm() / yg.norm()),
                       "colnorm_E_med": float(A_E.norm(dim=0).median()),
                       "colnorm_gain_med": float(A_g.norm(dim=0).median()),
                       "percell_E_med": float((A_E.norm(dim=0) * E_true).median()),
                       "percell_gain_med": float((A_g.norm(dim=0) * g_true).median()),
                       "cond_Gg": float(torch.linalg.cond(Gg)), "e_fix": {}}
                # principal angles between the two blocks' column spaces
                QE = torch.linalg.qr(A_E).Q
                Qg = torch.linalg.qr(A_g).Q
                sv = torch.linalg.svdvals(QE.T @ Qg).clamp(-1, 1)
                ang = torch.rad2deg(torch.arccos(sv))
                row["principal_angles_deg"] = {"min": float(ang.min()), "median":
                                               float(ang.median()), "max": float(ang.max())}
                for v in [float(x) for x in a.e_fix.split(",")]:
                    dE = E_true - v
                    rhs = A_E @ dE
                    dg = torch.linalg.solve(Gg, A_g.T @ rhs)
                    frac = float((A_g @ dg).norm() / rhs.norm())
                    rel = (dg.abs() / g_true).cpu().numpy()
                    row["e_fix"][f"E{v:g}"] = {
                        "med_rel_E_error": float((dE.abs() / E_true).median()),
                        "norm_A_E_dE_over_A_g_gtrue": float(rhs.norm() / yg.norm()),
                        "tradeoff_fraction": frac,
                        "predicted_med_rel_gain_error": float(np.median(rel)),
                        "predicted_p90_rel_gain_error": float(np.percentile(rel, 90))}
                R[key] = row
                log(f"\n[{key}]  ||b|| {row['norm_b']:.4g}   elastic ||A_E E|| {yE.norm():.4g} "
                    f"vs active ||A_g g|| {yg.norm():.4g}  ->  ELASTIC IS "
                    f"{row['elastic_over_active']:.1f}x THE ACTIVE TERM")
                log(f"    principal angles between span(A_E) and span(A_g): min "
                    f"{row['principal_angles_deg']['min']:.2f} deg, median "
                    f"{row['principal_angles_deg']['median']:.2f} deg  "
                    f"(0 = the blocks can imitate each other, 90 = independent)")
                log(f"    {'E pinned at':<14s} {'med|dE/E|':>10s} {'||A_E dE||/||A_g g||':>21s} "
                    f"{'trade-off':>10s} {'-> med|dg/g|':>13s} {'p90':>9s}")
                for kk, vv in row["e_fix"].items():
                    log(f"    {kk:<14s} {vv['med_rel_E_error']:>10.4f} "
                        f"{vv['norm_A_E_dE_over_A_g_gtrue']:>21.3f} "
                        f"{vv['tradeoff_fraction']:>10.4f} "
                        f"{vv['predicted_med_rel_gain_error']:>13.4f} "
                        f"{vv['predicted_p90_rel_gain_error']:>9.4f}")
                del A, A_E, A_g, QE, Qg, Gg
                torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

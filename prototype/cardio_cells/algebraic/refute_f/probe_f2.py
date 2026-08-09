"""TASK F -- probe 6: is the FRAME-CADENCE failure a property of the proposal, or of the
linearisation recover.py chose?

recover.py builds the frame-cadence design matrix as a SECANT FROM theta = 0:
      A[:,j] = ( a(s e_j) - a(0) ) / s,   s = 100 for E, 1 for the gain
For the 1-substep map that is exact (the map is affine).  For a 10-substep FRAME the map is NOT
affine, so a chord anchored at E = 0 (a tissue with zero stiffness) is a wild extrapolation, and
the reported 'model residual 1.395' measures the failure of THAT chord, not the information
content of the frame constraint.

This probe compares, on the same frozen state, at ZERO noise:
   (1) chord@0        -- what recover.py does
   (2) chord@nominal  -- same construction, anchored at (E=130, gain=1)
   (3) local Jacobian at theta_true (forward differences, 0.1% steps)
   (4) LEVENBERG-MARQUARDT on the actual proposal: inject x_k, predict x_{k+1} over one frame,
       minimise || x_pred(theta) - x_obs ||, started from the nominal theta.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, "/workspace/Plexus/prototype/cardio_cells/algebraic")

from assemble import SUBSTEP_TOKENS, System, rel            # noqa: E402
from recover import Solver, fd_accel, install_E, score      # noqa: E402
from probe_f import build, draw                             # noqa: E402

OUT = "/workspace/Plexus/prototype/cardio_cells/algebraic/refute_f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--tag", default="C100")
    args = ap.parse_args()
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"argv": vars(args)}
    t0 = time.time()
    with torch.no_grad():
        dev, dt = args.device, torch.float64
        C0 = 472 if args.real else args.cells
        gg = torch.Generator().manual_seed(2026)
        E_p = (40.0 + 180.0 * torch.rand(C0, generator=gg)).to(dev, dt)
        g_p = (0.5 + 1.0 * torch.rand(C0, generator=gg)).to(dev, dt)
        sy = build(args, args.warmup, E_p, g_p, E_p, g_p, log)
        C, ns = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        R["C"], R["Np"], R["n_sub_per_frame"] = C, sy.Np, ns
        nom = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt),
                         torch.ones(C, device=dev, dtype=dt)])
        sc_nom = score(nom, th, C)
        R["null_nominal"] = sc_nom
        log(f"   null (E=130,g=1): med E {sc_nom['med_E']:.3f} med g {sc_nom['med_gain']:.3f} "
            f"l2 {sc_nom['rel_l2']:.3f}")

        # -------- the observation the proposal asks for: x at the next FRAME ---------------- #
        a_obs = sy.step(sy.E_true, sy.gain_true, n_sub=ns)
        x_obs = sy.p.get("pos").clone()
        x_k = sy.x0.clone()
        R["frame_disp_rms"] = float((x_obs - x_k).norm() / np.sqrt(x_obs.numel()))
        log(f"   one-frame displacement rms = {R['frame_disp_rms']:.3e} "
            f"({R['frame_disp_rms']/sy.g.dx:.3f} grid cells)")

        def resid(theta):
            """r(theta) = x_pred(theta) - x_obs, one FRAME from the injected state."""
            E = torch.zeros(C + 1, device=dev, dtype=sy.dtype)
            gn = torch.zeros_like(E)
            E[1:] = theta[:C]
            gn[1:] = theta[C:]
            sy.step(E, gn, n_sub=ns)
            return (sy.p.get("pos") - x_obs).reshape(-1).double()

        def jac(theta, hE=0.13, hg=1e-3):
            r0 = resid(theta)
            J = torch.zeros(r0.numel(), 2 * C, device=dev, dtype=torch.float64)
            for j in range(2 * C):
                h = hE if j < C else hg
                tp = theta.clone()
                tp[j] += h
                J[:, j] = (resid(tp) - r0) / h
            return J, r0

        # ---- (1) chord@0 and (2) chord@nominal, both as ACCELERATION constraints ----------- #
        def chord(anchor, sE=100.0, sg=1.0):
            a_anchor = sy.a_of_theta(anchor, n_sub=ns)
            A = torch.zeros(a_anchor.numel(), 2 * C, device=dev, dtype=sy.dtype)
            for j in range(2 * C):
                s = sE if j < C else sg
                tp = anchor.clone()
                tp[j] += s
                A[:, j] = (sy.a_of_theta(tp, n_sub=ns) - a_anchor) / s
            return A, a_anchor

        zero = torch.zeros(2 * C, device=dev, dtype=sy.dtype)
        for name, anchor in (("chord@0", zero), ("chord@nominal", nom.to(sy.dtype))):
            A, a_anchor = chord(anchor)
            b = a_obs - a_anchor + A @ anchor           # A theta = b  form
            res = rel(A @ sy.theta_true - b, b)
            S = Solver(A, C)
            sols = S(b)
            sc = {k: score(v, th, C) for k, v in sols.items()}
            best = min(sc, key=lambda k: sc[k]["rel_l2"])
            R[name] = {"model_residual_rel_b": res, "cond": S.cond,
                       "ridge0": sc["ridge0"], "best": best, "best_score": sc[best]}
            log(f"   [{name:14s}] superposition residual {res:.3e}  | ridge0 med E "
                f"{sc['ridge0']['med_E']:.3e} med g {sc['ridge0']['med_gain']:.3e} "
                f"l2 {sc['ridge0']['rel_l2']:.3e} | best {best} l2 {sc[best]['rel_l2']:.3e}")
            S.free()
            del A, S
            torch.cuda.empty_cache()

        # ---- (3) local Jacobian at theta_true: how linear IS the frame map, locally? ------- #
        J, r_true = jac(sy.theta_true.double())
        gj = torch.Generator(device=dev).manual_seed(9)
        out3 = {}
        for frac in (0.01, 0.1, 0.5):
            d = frac * th * torch.randn(2 * C, generator=gj, device=dev, dtype=dt)
            r_pred = r_true + J @ d
            r_act = resid((th + d).to(sy.dtype))
            out3[f"{frac:g}"] = float((r_pred - r_act).norm() / max(r_act.norm(), 1e-300))
        R["local_linearity_of_frame_map"] = out3
        log(f"   local Jacobian at theta_true: ||J d - (r(th+d)-r(th))||/||r(th+d)|| for a "
            f"random d of relative size: {out3}")

        # ---- (4) LM on the proposal itself -------------------------------------------------- #
        log("\n   [LM] inject x_k, predict x_{k+1} over ONE FRAME, minimise ||x_pred - x_obs||, "
            "start at the nominal theta")
        s_col = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt),
                           torch.ones(C, device=dev, dtype=dt)])
        theta = nom.clone()
        lam = 1e-6
        hist = []
        r = resid(theta.to(sy.dtype))
        f0 = float(r.norm())
        for it in range(args.iters):
            J, r = jac(theta.to(sy.dtype))
            Js = J * s_col[None, :]
            G = Js.T @ Js
            rhs = -(Js.T @ r)
            trd = float(torch.diagonal(G).mean())
            ok = False
            for _ in range(12):
                M = G + lam * trd * torch.eye(2 * C, device=dev, dtype=dt)
                try:
                    d = torch.linalg.solve(M, rhs) * s_col
                except Exception:
                    lam *= 10
                    continue
                cand = theta + d
                rc = resid(cand.to(sy.dtype))
                if float(rc.norm()) < float(r.norm()):
                    theta = cand
                    lam = max(lam * 0.3, 1e-14)
                    ok = True
                    break
                lam *= 10
            sc = score(theta, th, C)
            hist.append({"iter": it, "obj_over_start": float(rc.norm() / f0), "lam": lam,
                         "accepted": ok, **sc})
            log(f"      it {it}: ||r||/||r_0|| {float(rc.norm()/f0):.3e}  med E {sc['med_E']:.3e} "
                f"med g {sc['med_gain']:.3e}  l2 {sc['rel_l2']:.3e}  lam {lam:.1e} ok={ok}")
            del J, Js, G
            torch.cuda.empty_cache()
            if not ok:
                break
        R["LM"] = hist
        R["LM_final"] = hist[-1] if hist else None

    R["wall_seconds"] = time.time() - t0
    p = os.path.join(OUT, f"probe_f2_{args.tag}.json")
    json.dump(R, open(p, "w"), indent=1, default=str)
    open(os.path.join(OUT, f"probe_f2_{args.tag}.log"), "w").write("\n".join(lines))
    log(f"\nwrote {p}  [{R['wall_seconds']:.1f}s]")


if __name__ == "__main__":
    main()

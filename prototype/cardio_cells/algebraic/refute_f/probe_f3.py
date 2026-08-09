"""TASK F -- probe 7: noise tolerance of the FRAME-cadence proposal when it is solved properly.

recover.py's headline noise number ('useful recovery needs sigma <= 1e-7 grid cells') comes from a
SUBSTEP-cadence second difference (divisor dt_sub^2 = 4e-8) and its frame-cadence noise sweep was
run through the chord@0 design matrix, which probe_f2 shows is broken at zero noise already.

Here: the proposal as stated -- inject x_k, predict x_{k+1} one FRAME ahead, minimise
||x_pred(theta) - x_obs||_2 -- with Gaussian noise of std sigma on the observed target positions
(the same optimistic 'N1' convention: the state A/J are built from stays clean).  Levenberg-
Marquardt from the nominal theta.  Reports the sigma at which the median |dE/E| crosses 10%.
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

from assemble import System, rel                            # noqa: E402
from recover import score                                   # noqa: E402
from probe_f import build                                   # noqa: E402

OUT = "/workspace/Plexus/prototype/cardio_cells/algebraic/refute_f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--sigmas", default="0,1e-6,1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--state-noise", action="store_true")
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
        C, ns, dx = sy.C, sy.n_sub_per_frame, sy.g.dx
        th = sy.theta_true.double()
        nom = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt),
                         torch.ones(C, device=dev, dtype=dt)])
        R["null_nominal"] = score(nom, th, C)
        sy.step(sy.E_true, sy.gain_true, n_sub=ns)
        x_clean = sy.p.get("pos").clone()
        R["frame_disp_over_dx"] = float((x_clean - sy.x0).norm()
                                        / np.sqrt(x_clean.numel()) / dx)
        log(f"   frame displacement rms = {R['frame_disp_over_dx']:.4f} dx;  "
            f"null med E {R['null_nominal']['med_E']:.3f}")
        gen = torch.Generator(device=dev).manual_seed(4242)
        s_col = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt),
                           torch.ones(C, device=dev, dtype=dt)])
        R["sweep"] = {}
        for sg in [float(x) for x in args.sigmas.split(",")]:
            sigma = sg * dx
            x_obs = x_clean + sigma * torch.randn(x_clean.shape, generator=gen,
                                                  device=dev, dtype=sy.dtype)

            def resid(theta):
                E = torch.zeros(C + 1, device=dev, dtype=sy.dtype)
                gn = torch.zeros_like(E)
                E[1:] = theta[:C]
                gn[1:] = theta[C:]
                sy.step(E, gn, n_sub=ns)
                return (sy.p.get("pos") - x_obs).reshape(-1).double()

            theta = nom.clone()
            lam, hist = 1e-6, []
            r = resid(theta.to(sy.dtype))
            f0 = float(r.norm())
            rc = r
            for it in range(args.iters):
                r0 = resid(theta.to(sy.dtype))
                J = torch.zeros(r0.numel(), 2 * C, device=dev, dtype=dt)
                for j in range(2 * C):
                    h = 0.13 if j < C else 1e-3
                    tp = theta.clone()
                    tp[j] += h
                    J[:, j] = (resid(tp.to(sy.dtype)) - r0) / h
                Js = J * s_col[None, :]
                G = Js.T @ Js
                rhs = -(Js.T @ r0)
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
                    rcand = resid(cand.to(sy.dtype))
                    if float(rcand.norm()) < float(r0.norm()):
                        theta, rc, ok = cand, rcand, True
                        lam = max(lam * 0.3, 1e-14)
                        break
                    lam *= 10
                sc = score(theta, th, C)
                hist.append({"iter": it, "obj_over_start": float(rc.norm() / f0),
                             "accepted": ok, **sc})
                del J, Js, G
                torch.cuda.empty_cache()
                if not ok:
                    break
            best = min(hist, key=lambda h: h["rel_l2"])
            R["sweep"][f"{sg:g}"] = {"sigma_world": sigma, "hist": hist, "final": hist[-1],
                                     "best_iter": best}
            log(f"   sigma={sg:<8g} dx : final med E {hist[-1]['med_E']:.3e}  med g "
                f"{hist[-1]['med_gain']:.3e}  p90 E {hist[-1]['p90_E']:.3e}  l2 "
                f"{hist[-1]['rel_l2']:.3e}  ({len(hist)} accepted iters, "
                f"obj/obj0 {hist[-1]['obj_over_start']:.2e})")

    R["wall_seconds"] = time.time() - t0
    p = os.path.join(OUT, f"probe_f3_{args.tag}.json")
    json.dump(R, open(p, "w"), indent=1, default=str)
    open(os.path.join(OUT, f"probe_f3_{args.tag}.log"), "w").write("\n".join(lines))
    log(f"\nwrote {p}  [{R['wall_seconds']:.1f}s]")


if __name__ == "__main__":
    main()

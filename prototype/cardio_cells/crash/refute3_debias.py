"""refute3_debias.py -- the F noise sits in A, not in b.  Correct the normal equations for it.

refute3_simex.py measured the defect exactly: with a per-frame measurement error sigma_F = 0.0039
on the injected deformation gradient, the recovered stiffness is ATTENUATED --
mean(E_hat)/mean(E_true) = 0.391, regression slope of E_hat on E_true = 0.344 -- and adding more
known noise attenuates it further, monotonically.  That is errors-in-variables, whose leading term
is a BIAS: it does not average down over frames and it is not cured by a better conditioned solve.

The textbook remedy, when the noise covariance is known (and sigma_F IS measurable on the
recording: quiet-stretch second difference, verified temporally white), is the noise-corrected
normal equations.  With A_hat = A(F_hat) = A(F) + Delta,

    E[A_hat^T A_hat] = A^T A + E[Delta^T Delta],     E[A_hat^T b_hat] = A^T b + E[Delta^T b_delta]

so estimate both correction terms by Monte Carlo -- re-noise the ALREADY NOISY measurement K times
with the same sigma and the same within-frame coherence, assemble, and average -- and subtract.
K extra assemblies at ~10 s each.  Nothing about theta_true is used.

usage: PYTHONPATH=/workspace/Plexus/src python refute3_debias.py --device cuda:0 --K 8
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

from recover import score, theta_scale                          # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import assemble_inj, record_substeps, lerp         # noqa: E402
from refute_round3 import advance, fit                          # noqa: E402


def solve_G(G, rhs, s, lam=0.0):
    n = G.shape[0]
    tr = float(torch.diagonal(G).abs().sum() / n)
    M = G + lam * tr * torch.eye(n, device=G.device, dtype=G.dtype)
    try:
        z = torch.linalg.solve(M, rhs)
    except Exception:
        z = torch.linalg.lstsq(M, rhs.unsqueeze(1)).solution.squeeze(1)
    return z * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="refute3_debias")
    ap.add_argument("--sigma-F", type=float, default=3.9e-3)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    START = 40
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=START,
                           window=START, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)
    sx = 0.0409 * 4.88e-4
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        sy.restore()
        advance(sy, START, a.t0)
        sy._snapshot(a.t0)
        Fs, Cs, Xs = record_substeps(sy, n)
        x0, F0, F1, x_next = sy.x0.clone(), sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()
        gen = torch.Generator(device=sy.device).manual_seed(5150)
        sc_scale = theta_scale(C, sy.device)

        def draw(shape):
            return torch.randn(shape, generator=gen, device=sy.device, dtype=sy.dtype)

        s_clean, _ = fit(sy, n, lerp(F0, F1, n), x_next, x0, th, C)
        log(f"[control] noise-free F_lerp at t0={a.t0}: med|dE/E| {s_clean['med_E']:.4f}")
        R["clean"] = s_clean
        R["runs"] = {}

        for sF in (0.0, a.sigma_F):
            for rep in range(a.reps if sF > 0 else 1):
                if sF > 0:
                    F0h = F0 + (sF / 2.0) * draw(F0.shape)
                    F1h = F1 + (sF / 2.0) * draw(F0.shape)
                    xn = x_next + sx * draw(x_next.shape)
                else:
                    F0h, F1h, xn = F0.clone(), F1.clone(), x_next.clone()
                Ah, y0h, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
                bh = (xn - x0).reshape(-1) - y0h
                Az = Ah * sc_scale[None, :]
                G0 = Az.T @ Az
                r0 = Az.T @ bh
                th_naive = solve_G(G0, r0, sc_scale)

                # Monte-Carlo estimate of the two correction terms
                Gk = torch.zeros_like(G0)
                rk = torch.zeros_like(r0)
                for k in range(a.K):
                    kk = (sF / 2.0) if sF > 0 else 0.0
                    Fa = F0h + kk * draw(F0.shape)
                    Fb = F1h + kk * draw(F0.shape)
                    Ak, y0k, _ = assemble_inj(sy, n, lerp(Fa, Fb, n), None)
                    Azk = Ak * sc_scale[None, :]
                    bk = (xn - x0).reshape(-1) - y0k
                    Gk += Azk.T @ Azk
                    rk += Azk.T @ bk
                    del Ak, Azk
                    torch.cuda.empty_cache()
                Gk /= a.K
                rk /= a.K
                Gc = G0 - (Gk - G0)                 # A^T A - E[Delta^T Delta]
                rc = r0 - (rk - r0)
                ev = torch.linalg.eigvalsh(Gc)
                out = {"sigma_F": sF, "rep": rep,
                       "naive": score(th_naive, th, C),
                       "corr_norm_over_G": float((Gk - G0).norm() / G0.norm()),
                       "min_eig_corrected": float(ev.min()),
                       "min_eig_naive": float(torch.linalg.eigvalsh(G0).min())}
                for lam in (0.0, 1e-8, 1e-6, 1e-4):
                    tc = solve_G(Gc, rc, sc_scale, lam)
                    out[f"corrected_ridge{lam:g}"] = score(tc, th, C)
                    out[f"corrected_ridge{lam:g}"]["mean_ratio"] = float(
                        tc[:C].mean() / th[:C].mean())
                out["naive"]["mean_ratio"] = float(th_naive[:C].mean() / th[:C].mean())
                R["runs"][f"sF{sF:g}_rep{rep}"] = out
                log(f"[sF {sF:g} rep {rep}] naive medE {out['naive']['med_E']:.4f} "
                    f"(mean ratio {out['naive']['mean_ratio']:.3f})  |corr|/|G| "
                    f"{out['corr_norm_over_G']:.3e}  min eig {out['min_eig_corrected']:.2e} "
                    f"(naive {out['min_eig_naive']:.2e})")
                for lam in (0.0, 1e-8, 1e-6, 1e-4):
                    v = out[f"corrected_ridge{lam:g}"]
                    log(f"      corrected ridge {lam:<6g} medE {v['med_E']:.4f}  p90 "
                        f"{v['p90_E']:.4f}  mean ratio {v['mean_ratio']:.3f}  l2 {v['rel_l2']:.3f}")
                del Ah, Az
                torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""round5_sensitivity.py -- ROUND 5, stage 2b.  Is each estimator a FUNCTION of its inputs?

WHY THIS EXISTS
====================================================================================================
Round 5's `naive` theta at T=8, seed 90210 reproduces round 4's stored theta to 1.8e-15 RELATIVE --
the two runs are on different GPUs, from different scripts, so G0 and r0 are the same matrices.  The
`eiv_snr0` theta from the SAME normal equations differs from round 4's by 85 % relative.  An
estimator that moves 85 % when its input moves 1e-15 is not an estimator.  This script measures that
directly, on the saved normal equations, with no GPU and no simulator:

  A. the box-constrained solve, at 4000 vs 40000 iterations (is the reported theta the optimum?),
     with the KKT residual normalised by ||r||;
  B. a symmetric relative perturbation of size eps applied to G0, Gbar and r, five draws each, for
     every estimator -- the amplification factor d(theta)/d(input);
  C. where the information is: the generalised (Gc, Sigma) spectrum, and the share of ||theta_hat||^2
     that the SNR < 0.3 directions carry.

usage: PYTHONPATH=/workspace/Plexus/src python round5_sensitivity.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/workspace/Plexus/prototype/cardio_cells/algebraic")

from round5_solve import solve_box, snr_trunc                     # noqa: E402


def load(tag, T=8):
    z = np.load(os.path.join(HERE, f"{tag}.npz"))
    G0 = sum(torch.as_tensor(z[f"G{k}"], dtype=torch.float64) for k in range(T))
    r0 = sum(torch.as_tensor(z[f"r{k}"], dtype=torch.float64) for k in range(T))
    Gb = sum(torch.as_tensor(z[f"Gm{k}"], dtype=torch.float64) for k in range(T))
    rb = sum(torch.as_tensor(z[f"rm{k}"], dtype=torch.float64) for k in range(T))
    s = torch.as_tensor(z["s"], dtype=torch.float64)
    th = torch.as_tensor(z["theta_true"], dtype=torch.float64)
    return G0, r0, Gb, rb, s, th


def solvers(G0, r0, Gb, rb, s, lo, hi, box_iters=4000):
    Sig = Gb - G0
    Gc, rc = G0 - Sig, r0 - (rb - r0)
    out = {"naive": torch.linalg.solve(G0, r0) * s}
    out["eiv_snr0"], ex = snr_trunc(G0, Sig, Gc, rc, s, tau=0.0)
    out["naive_box"], i1 = solve_box(G0, r0, s, lo, hi,
                                     z0=torch.clamp(out["naive"], lo, hi) / s, iters=box_iters)
    out["eiv_box"], i2 = solve_box(Gc, rc, s, lo, hi,
                                   z0=torch.clamp(out["eiv_snr0"], lo, hi) / s, iters=box_iters)
    return out, {"snr": ex, "naive_box": i1, "eiv_box": i2, "Sig": Sig, "Gc": Gc, "rc": rc}


def main():
    R = {}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    tag = "round5_norm_s90210_sF0.0039"
    G0, r0, Gb, rb, s, th = load(tag)
    C = th.numel() // 2
    nv = torch.linalg.solve(G0, r0) * s
    mE = float(nv[:C][nv[:C] > 0].median())
    mg = float(nv[C:][nv[C:] > 0].median())
    lo = torch.cat([torch.full((C,), 0.2 * mE, dtype=torch.float64),
                    torch.full((C,), 0.2 * mg, dtype=torch.float64)])
    hi = torch.cat([torch.full((C,), 5.0 * mE, dtype=torch.float64),
                    torch.full((C,), 5.0 * mg, dtype=torch.float64)])

    base, ex = solvers(G0, r0, Gb, rb, s, lo, hi, box_iters=4000)
    long, exl = solvers(G0, r0, Gb, rb, s, lo, hi, box_iters=40000)

    # ---- A: is the box solve at its optimum? -------------------------------------------------
    R["A_box_convergence"] = {}
    log(f"[A] box solve, 4000 vs 40000 iterations (T=8, {tag})")
    for k in ("naive_box", "eiv_box"):
        d = float((base[k] - long[k]).norm() / long[k].norm())
        R["A_box_convergence"][k] = {
            "rel_change_4k_to_40k": d,
            "kkt_over_r_4k": ex[k]["kkt_resid"] / float(r0.norm()),
            "kkt_over_r_40k": exl[k]["kkt_resid"] / float(r0.norm()),
            "n_active_bounds": ex[k]["n_active_bounds"],
            "objective_4k": ex[k]["objective"], "objective_40k": exl[k]["objective"]}
        q = R["A_box_convergence"][k]
        log(f"    {k:<10s} |dtheta|/|theta| {d:.3e}  KKT/|r| {q['kkt_over_r_4k']:.2e} -> "
            f"{q['kkt_over_r_40k']:.2e}  active bounds {q['n_active_bounds']}  "
            f"objective {q['objective_4k']:.8e} -> {q['objective_40k']:.8e}")

    # ---- B: amplification of a relative input perturbation ------------------------------------
    R["B_amplification"] = {}
    g = torch.Generator().manual_seed(7)
    log("\n[B] symmetric relative perturbation of (G0, Gbar, r0, rbar); median over 5 draws of "
        "||dtheta||/||theta||")
    log(f"    {'eps':>8s} " + " ".join(f"{k:>12s}" for k in base))
    for eps in (1e-15, 1e-12, 1e-9, 1e-6):
        acc = {k: [] for k in base}
        for _ in range(5):
            def pert(M):
                N = torch.randn(M.shape, generator=g, dtype=torch.float64)
                if M.dim() == 2:
                    N = (N + N.T) / 2
                return M + eps * M.abs().max() * N
            o, _ = solvers(pert(G0), pert(r0), pert(Gb), pert(rb), s, lo, hi)
            for k in base:
                acc[k].append(float((o[k] - base[k]).norm() / base[k].norm()))
        R["B_amplification"][f"{eps:g}"] = {k: float(np.median(v)) for k, v in acc.items()}
        log(f"    {eps:>8g} " + " ".join(
            f"{R['B_amplification'][f'{eps:g}'][k]:>12.3e}" for k in base))

    # ---- C: where the information is -----------------------------------------------------------
    Sig, Gc, rc = ex["Sig"], ex["Gc"], ex["rc"]
    w, U = torch.linalg.eigh(Sig)
    sig2 = float(w.abs().max())
    wf = w.clamp(min=1e-2 * sig2)
    Lh = U @ torch.diag(wf.rsqrt()) @ U.T
    M = Lh @ Gc @ Lh
    lam, V = torch.linalg.eigh((M + M.T) / 2)
    Vw = Lh @ V
    zt = (th / s)
    coef_true = (torch.linalg.inv(Vw) @ zt) if True else None
    share = (coef_true ** 2) / (coef_true ** 2).sum()
    R["C_spectrum"] = {
        "sigma_spec_norm": sig2, "G0_spec_norm": float(torch.linalg.eigvalsh(G0).max()),
        "sigma_fro_over_G_fro": float(Sig.norm() / G0.norm()),
        "min_eig_G0": float(torch.linalg.eigvalsh(G0).min()),
        "min_eig_Gc": float(torch.linalg.eigvalsh(Gc).min()),
        "cond_G0": float(torch.linalg.eigvalsh(G0).max() / torch.linalg.eigvalsh(G0).min()),
        "n_snr_gt1": int((lam > 1).sum()), "n_snr_gt0.3": int((lam > 0.3).sum()),
        "n_snr_gt0": int((lam > 0).sum()),
        "lam_quantiles": [float(v) for v in torch.quantile(
            lam, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64))],
        "share_of_theta_true_in_snr_lt_0.3": float(share[lam < 0.3].sum()),
        "share_of_theta_true_in_snr_lt_1": float(share[lam < 1].sum())}
    q = R["C_spectrum"]
    log(f"\n[C] cond(G0) {q['cond_G0']:.2e}; ||Sigma||_F/||G0||_F {q['sigma_fro_over_G_fro']:.2e}; "
        f"min eig G0 {q['min_eig_G0']:.2e} -> Gc {q['min_eig_Gc']:.2e}")
    log(f"    SNR>1 in {q['n_snr_gt1']}/200 directions, SNR>0.3 in {q['n_snr_gt0.3']}, "
        f"SNR>0 in {q['n_snr_gt0']}; theta_true has "
        f"{100*q['share_of_theta_true_in_snr_lt_0.3']:.1f}% of its energy where SNR<0.3")

    json.dump(R, open(os.path.join(HERE, "round5_sensitivity.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, "round5_sensitivity.log"), "w").write("\n".join(lines) + "\n")
    log("\nwrote round5_sensitivity.json")


if __name__ == "__main__":
    main()

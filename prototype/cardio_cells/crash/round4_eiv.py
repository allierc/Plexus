"""round4_eiv.py -- ROUND 4, THE ONE CHANGE: the F-injected fit is errors-in-variables, so
correct the normal equations for the regressor's own noise and solve in a TRUNCATED subspace.

WHY (round 3's diagnosis, in one paragraph)
====================================================================================================
Injecting a linearly interpolated MEASURED deformation gradient across the substeps of a frame took
the frame-cadence recovery from med|dE/E| 0.257 to 0.0078 (finject.json), and that survived a t0
sweep, six wrong-F nulls and a state-oracle ablation.  What did not survive was round 3's reading of
the noise: F enters **A**, not b, so at the recording's own sigma_F = 3.9e-3 the estimator is
BIASED, not merely noisy -- mean(E_hat)/mean(E_true) = 0.391, regression slope 0.344, both driven
monotonically down by adding further known noise (refute3_simex.json).  Ridge, better data and
frame stacking all attack variance; refute3_E2.json measured that stacking 8 frames moves white-F
noise only 0.539 -> 0.468 while it halves position noise, which is the signature of a bias.

THE ONE CHANGE (finject.solve_eiv):
  1. assemble A_hat from the measured F_hat, column-scale, form G0 = A^T A and r0 = A^T b;
  2. K = 8 Monte-Carlo re-noisings of the ALREADY NOISY F_hat at the MEASURED sigma_F, with ONE
     COHERENT draw per frame boundary (never per substep -- refute_round3 phase D measured that the
     per-substep model is optimistic by 2x), giving Sigma = E[Delta^T Delta] and c = E[Delta^T b];
  3. Gc = G0 - Sigma, rc = r0 - c;
  4. eigendecompose Gc and solve in the subspace lambda_i > tau*||Sigma||_2 -- the step
     refute3_debias.py was missing, and the reason its draw 1 returned mean ratio -5.80.

sigma_F is read off the recording (real_F_check.json quiet-stretch second difference, verified
temporally white).  Nothing about theta_true is used by the estimator.

CONTROLS AND NULLS RUN HERE
  clean            noise-free F_lerp must reproduce finject.json's 0.0077773
  sigma_F = 0      the correction must be a numerical no-op: ||Sigma||/||G0|| <= 1e-12
  naive            plain solve on G0 -- the control the change has to beat
  naive_trunc      the SAME truncation with NO correction -- separates truncation from de-biasing
  eiv_full         the corrected Gram solved WITHOUT truncation (refute3_debias's version)
  wrong sigma      the correction run at 0.5x and 2x the true sigma_F: how much does it need to know
  n_draws >= 5     spread across independent measurement draws, not one lucky one

usage: PYTHONPATH=/workspace/Plexus/src python round4_eiv.py --device cuda:1
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

from recover import score                                       # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import record_substeps, lerp, solve_eiv, _noise_F  # noqa: E402
from refute_round3 import fit                                   # noqa: E402

SIGMA_X = 0.0409 * 4.88e-4          # real_F_check.json: temporal noise on the recorded displacement
SIGMA_F = 3.9e-3                    # real_F_check.json: quiet-stretch second difference on dF


def stats(theta, th, C):
    s = score(theta, th, C)
    s["mean_ratio_E"] = float(theta[:C].mean() / th[:C].mean())
    s["mean_ratio_g"] = float(theta[C:].mean() / th[C:].mean())
    # regression slope of E_hat on E_true, the quantity refute3_simex used for the attenuation
    x = th[:C] - th[:C].mean()
    y = theta[:C] - theta[:C].mean()
    s["slope_E"] = float((x * y).sum() / (x * x).sum())
    s["n_negE"] = int((theta[:C] < 0).sum())
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round4_eiv")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--taus", default="0.1,0.3,1.0,3.0")
    ap.add_argument("--wrong-sigma", action="store_true")
    a = ap.parse_args()
    taus = tuple(float(x) for x in a.taus.split(","))

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "sigma_F": a.sigma_F, "sigma_x": SIGMA_X, "K": a.K,
         "taus": list(taus),
         "ONE_CHANGE": "finject.solve_eiv -- Monte-Carlo noise-corrected normal equations in the "
                       "MEASURED deformation gradient, solved in the subspace lambda > tau*||Sigma||"}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        Fs, Cs, Xs = record_substeps(sy, n)
        x0, F0, F1, x_next = sy.x0.clone(), sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()

        # ---- control 1: the clean fit reproduces finject.json --------------------------------
        s_clean, th_clean = fit(sy, n, lerp(F0, F1, n), x_next, x0, th, C)
        R["control_clean"] = {**stats(th_clean, th, C),
                              "finject_reported_med_E": 0.0077773}
        log(f"[control] clean F_lerp at t0={a.t0}: med|dE/E| {s_clean['med_E']:.6f} "
            f"(finject.json 0.0077773; delta {abs(s_clean['med_E']-0.0077773):.2e})")

        gm = torch.Generator(device=sy.device).manual_seed(90210)   # measurement draws
        gk = torch.Generator(device=sy.device).manual_seed(31337)   # Monte-Carlo re-noising

        # ---- control 2: at sigma_F = 0 the correction must be a no-op -------------------------
        out0, ex0 = solve_eiv(sy, n, F0, F1, x_next, x0, 0.0, K=a.K, taus=taus, gen=gk)
        R["control_sigma0"] = {"extra": {k: v for k, v in ex0.items() if not k.startswith(("A", "_"))},
                               "scores": {k: stats(v, th, C) for k, v in out0.items()}}
        log(f"[control] sigma_F = 0: ||Sigma||_F/||G0||_F {ex0['sigma_fro_over_G_fro']:.2e}  "
            f"||Sigma||_2/||G0||_2 {ex0['sigma_spec_norm']/ex0['G0_spec_norm']:.2e}  "
            f"naive medE {R['control_sigma0']['scores']['naive']['med_E']:.6f}  "
            f"eiv_trunc1 medE {R['control_sigma0']['scores']['eiv_trunc1']['med_E']:.6f} "
            f"(rank {ex0['rank']['eiv_trunc1']}/{2*C})")

        # ---- the measurement draws -------------------------------------------------------------
        R["draws"] = {}
        thetas = {}
        log(f"\n[eiv] sigma_F = {a.sigma_F:g} COHERENT per frame boundary, sigma_x = {SIGMA_X:.3g}; "
            f"K = {a.K}; {a.draws} independent measurement draws")
        hdr = ["naive", "eiv_full"] + [f"naive_trunc{t:g}" for t in taus] \
            + [f"eiv_trunc{t:g}" for t in taus] \
            + [f"eiv_snr{t:g}" for t in (0.0,) + taus]
        log(f"    {'draw':>4s} " + " ".join(f"{h[:14]:>15s}" for h in hdr))
        for d in range(a.draws):
            def dr(x):
                return torch.randn(x.shape, generator=gm, device=x.device, dtype=x.dtype)
            F0h = F0 + (a.sigma_F / 2.0) * dr(F0)
            F1h = F1 + (a.sigma_F / 2.0) * dr(F1)
            xn = x_next + SIGMA_X * dr(x_next)
            out, ex = solve_eiv(sy, n, F0h, F1h, xn, x0, a.sigma_F, K=a.K, taus=taus, gen=gk)
            sc = {k: stats(v, th, C) for k, v in out.items()}
            # WHERE IS THE INFORMATION?  Decompose theta_true in the generalised (SNR) basis.
            # The estimator never sees this -- it is read afterwards, from the basis it produced.
            lam, V = ex["_lam"], ex["_V"]
            s_col = torch.ones(2 * C, device=th.device, dtype=th.dtype)
            s_col[:C] = 130.0
            tz = th / s_col                              # theta in the column-scaled space
            # coefficients of tz in the (Sigma-orthonormal) generalised basis
            co = torch.linalg.solve(V, tz)
            p = (co ** 2) * torch.tensor([1.0], device=th.device, dtype=th.dtype)
            wgt = p / p.sum()
            R["draws"][f"d{d}"] = {
                "extra": {k: v for k, v in ex.items() if not k.startswith(("A", "_"))},
                "scores": sc,
                "F_meas_err": float((lerp(F0h, F1h, n) - Fs).norm() / Fs.norm()),
                "theta_true_energy_in_SNR_gt1": float(wgt[lam > 1].sum()),
                "theta_true_energy_in_SNR_gt3": float(wgt[lam > 3].sum()),
                "theta_true_energy_in_SNR_lt0.3": float(wgt[lam < 0.3].sum())}
            for k, v in out.items():
                thetas[f"d{d}::{k}"] = v.cpu().numpy()
            log(f"    {d:>4d} " + " ".join(f"{sc[h]['med_E']:>7.4f}/{sc[h]['mean_ratio_E']:>6.3f}"
                                           for h in hdr))
            if d == 0:
                log(f"         min eig  corrected {ex['min_eig_corrected']:.3e}  naive "
                    f"{ex['min_eig_naive']:.3e};  ||Sigma||_2 {ex['sigma_spec_norm']:.3e} "
                    f"= {ex['sigma_spec_norm']/ex['G0_spec_norm']:.2e} of ||G0||_2;  ranks "
                    + ", ".join(f"{k.replace('eiv_trunc','tau=')}:{v}"
                                for k, v in ex["rank"].items() if k.startswith("eiv")))

        # ---- summary across draws ---------------------------------------------------------------
        R["summary"] = {}
        log(f"\n[summary] over {a.draws} draws")
        log(f"    {'solver':<18s} {'medE med':>9s} {'medE sd':>9s} {'medE max':>9s} "
            f"{'mratio med':>11s} {'mratio min':>11s} {'mratio max':>11s} {'slope':>7s} "
            f"{'negE':>5s} {'rank':>5s} {'lam_min':>10s}")
        for h in hdr:
            v = [R["draws"][f"d{d}"]["scores"][h] for d in range(a.draws)]
            mr = [x["mean_ratio_E"] for x in v]
            me = [x["med_E"] for x in v]
            rk = [R["draws"][f"d{d}"]["extra"]["rank"].get(h) for d in range(a.draws)]
            lm = [R["draws"][f"d{d}"]["extra"]["lam_min_retained"].get(h) for d in range(a.draws)]
            row = {"med_E_median": float(np.median(me)), "med_E_sd": float(np.std(me)),
                   "med_E_max": float(np.max(me)), "med_E_spread": float(np.ptp(me)),
                   "mean_ratio_median": float(np.median(mr)), "mean_ratio_min": float(np.min(mr)),
                   "mean_ratio_max": float(np.max(mr)),
                   "slope_median": float(np.median([x["slope_E"] for x in v])),
                   "negE_median": float(np.median([x["n_negE"] for x in v])),
                   "rank": rk[0] if rk[0] is not None else None,
                   "lam_min_min": (float(np.min([x for x in lm if x is not None]))
                                   if any(x is not None for x in lm) else None),
                   "any_mean_ratio_abs_gt2": bool(np.max(np.abs(mr)) > 2.0)}
            R["summary"][h] = row
            log(f"    {h:<18s} {row['med_E_median']:>9.4f} {row['med_E_sd']:>9.4f} "
                f"{row['med_E_max']:>9.4f} {row['mean_ratio_median']:>11.3f} "
                f"{row['mean_ratio_min']:>11.3f} {row['mean_ratio_max']:>11.3f} "
                f"{row['slope_median']:>7.3f} {row['negE_median']:>5.0f} "
                f"{str(row['rank']):>5s} "
                + (f"{row['lam_min_min']:>10.2e}" if row['lam_min_min'] is not None else
                   f"{'n/a':>10s}"))

        # ---- how much does the correction need to know sigma_F? --------------------------------
        if a.wrong_sigma:
            R["wrong_sigma"] = {}
            log(f"\n[wrong sigma] the correction is run at a MIS-STATED sigma (draw 0's data)")
            def dr0(x, g):
                return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
            g2 = torch.Generator(device=sy.device).manual_seed(90210)
            F0h = F0 + (a.sigma_F / 2.0) * dr0(F0, g2)
            F1h = F1 + (a.sigma_F / 2.0) * dr0(F1, g2)
            xn = x_next + SIGMA_X * dr0(x_next, g2)
            for f in (0.5, 0.75, 1.0, 1.5, 2.0):
                gk2 = torch.Generator(device=sy.device).manual_seed(31337)
                out, ex = solve_eiv(sy, n, F0h, F1h, xn, x0, f * a.sigma_F, K=a.K,
                                    taus=(1.0,), gen=gk2)
                sc = {k: stats(v, th, C) for k, v in out.items()}
                R["wrong_sigma"][f"{f:g}"] = {"scores": sc,
                                              "rank": ex["rank"]["eiv_trunc1"]}
                log(f"    sigma used = {f:>4g} x true: eiv_trunc1 medE "
                    f"{sc['eiv_trunc1']['med_E']:.4f}  mean ratio "
                    f"{sc['eiv_trunc1']['mean_ratio_E']:.3f}  rank {ex['rank']['eiv_trunc1']}")

        np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                 **{f"cand::{k}": v for k, v in thetas.items()},
                 **{"cand::clean_F_lerp": th_clean.cpu().numpy(),
                    "cand::theta_true": th.cpu().numpy()})

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

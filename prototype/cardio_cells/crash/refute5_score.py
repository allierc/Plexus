"""refute5_score.py -- ROUND 5 REFUTATION, part 3: the crash test on the realizable-noise fits and
on two nulls round 5 did not run.

Everything about the scoring is round5_score.py's, imported or copied verbatim: same t0 = 165, same
150-frame free rollout, anchor=None, margin 20, metrics.py unmodified, the same deterministic 5x5
grid gauge + 2 Broyden steps, the same held-out one-frame residual at tick 180.

WHAT IS NEW -- two things.

(1) THE FITS.  refute5_fit.py repeats round 5's fit with the F measurement error drawn on a spatial
    NODE GRID instead of independently per particle, because refute5_spatial.py measured that the
    recording's F noise gives a cell ~23 independent samples, not 100.

(2) TWO NULLS ROUND 5 DID NOT RUN, both built from round 5's own winning estimate
    theta_round5.npz["round5_norm_s90210_sF0.0039|T8|eiv_box"]:
      * `null_permerr`  -- theta_true + a random PERMUTATION of that estimate's own error vector
        across cells (per block).  Identical error l2, identical marginal error distribution,
        identical heavy tail; only the pairing "which cell has which error" is destroyed.
      * `null_l2match`  -- theta_true + a Gaussian error of the same per-block l2.
    If these score far below the estimate, the estimate's 25 % l2 error sits in directions the
    rollout cannot see, and a 0.99 loopscore is NOT evidence that the per-cell moduli are recovered.

usage:
  PYTHONPATH=/workspace/Plexus/src python refute5_score.py --device cuda:1 --shard 0 --nshards 2
"""
from __future__ import annotations

import argparse
import glob
import json
import math
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

from assemble import SUBSTEP_TOKENS                              # noqa: E402
from recover import install_E                                    # noqa: E402
import metrics as MET                                            # noqa: E402
import crash_test as CT                                          # noqa: E402
from crash_round2 import percell_amplitude, r2_percell           # noqa: E402
from crash_round3 import scale2, t2_of                           # noqa: E402
from finject import record_substeps, lerp, y_of                  # noqa: E402
from refute_round3 import advance                                # noqa: E402
from round5_solve import pstats                                  # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                    # noqa: E402
from round5_score import gauge_grid                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="refute5_score")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "shard": [a.shard, a.nshards]}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G, n = sy.C, a.t0, a.window, sy.n_sub_per_frame
        th = sy.theta_true.double()
        dev, f64, dx = th.device, torch.float64, sy.g.dx
        x0, cid = sy.x0.clone(), sy.cid

        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ~anchor
        R["probes_in_anchor_band"] = {str(m): int(anchor[t].sum()) for m, t in tracers.items()}

        # ---- the reference window B (identical to round5_score.py) -------------------------------
        ref_full = torch.zeros(G, sy.Np, 2, device=sy.device, dtype=sy.dtype)
        sy.restore()
        install_E(sy, sy.E_true)
        for k in range(G):
            sy._outer(W + k, gain_cell=sy.gain_true)
            sy.H.sub_dt = sy.dt_sub
            for _ in range(n):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            ref_full[k] = sy.p.get("pos")
        d_ref = ref_full - x0[None]
        dm = d_ref[:, interior].mean(0, keepdim=True)
        ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
        real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
        a_ref, nk = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        snap0 = {k: getattr(sy, k).clone() for k in SNAP}
        R["reference"] = {"max_disp_dx": float(d_ref.norm(dim=-1).max() / dx),
                          "n_cells_kept": int(keep.sum())}

        # ---- the HELD-OUT frame (identical) ------------------------------------------------------
        sy.restore()
        advance(sy, W, a.holdout_tick)
        sy._snapshot(a.holdout_tick)
        Fh, Ch, Xh = record_substeps(sy, n)
        hx0, hF0, hF1, hxn = sy.x0.clone(), sy.F0.clone(), Fh[-1].clone(), Xh[-1].clone()
        y_obs_h = (hxn - hx0).reshape(-1)
        injh = lerp(hF0, hF1, n)
        gh = torch.Generator(device=sy.device).manual_seed(555001)
        injh_noisy = lerp(hF0 + (SIGMA_F / 2) * torch.randn(hF0.shape, generator=gh,
                                                            device=hF0.device, dtype=hF0.dtype),
                          hF1 + (SIGMA_F / 2) * torch.randn(hF0.shape, generator=gh,
                                                            device=hF0.device, dtype=hF0.dtype), n)
        y_obs_h_noisy = y_obs_h + SIGMA_X * torch.randn(y_obs_h.shape, generator=gh,
                                                        device=hF0.device, dtype=hF0.dtype)
        hsnap = {k: getattr(sy, k).clone() for k in SNAP}

        def holdout(theta):
            for k, v in hsnap.items():
                setattr(sy, k, v.clone())
            yc = y_of(sy, theta, n, injh, None)
            for k, v in hsnap.items():
                setattr(sy, k, v.clone())
            yn = y_of(sy, theta, n, injh_noisy, None)
            return (float((yc - y_obs_h).norm() / y_obs_h.norm()),
                    float((yn - y_obs_h_noisy).norm() / y_obs_h_noisy.norm()))

        def scored(theta, full_out=True):
            for k, v in snap0.items():
                setattr(sy, k, v.clone())
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                          anchor=None, interior=interior, ss_tot=ss_tot,
                                          keep_full=full_out, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            out = {"loop": m20["loopscore"], "t1": coarse["motion_energy_ratio_interior"],
                   "t2": t2_of(m20), "R2": coarse["R2_displacement_interior"],
                   "rms_dx_mean": coarse["rms_pos_err_dx_mean"]}
            if full_out:
                ah, _ = percell_amplitude(full, x0, cid, C, interior)
                out.update({"margin20": m20, "coarse": coarse,
                            "percell": r2_percell(ah, a_ref, keep)})
                del full
            return out

        # ---- candidates -------------------------------------------------------------------------
        cands = [("theta_true", th)]
        Z5 = np.load(os.path.join(HERE, "theta_round5.npz"))
        Zr = np.load(os.path.join(HERE, "theta_refute5.npz"))
        for k in Zr.files:
            if k == "theta_true":
                continue
            cands.append((k, torch.as_tensor(Zr[k], device=dev, dtype=f64)))

        # ---- the two nulls built from round 5's OWN winner ---------------------------------------
        win = torch.as_tensor(Z5["round5_norm_s90210_sF0.0039|T8|eiv_box"], device=dev, dtype=f64)
        err = (win - th).cpu().numpy()
        thn = th.cpu().numpy()
        rng = np.random.default_rng(20260809)
        for j in (0, 1):
            pe = np.empty_like(err)
            pe[:C] = rng.permutation(err[:C])
            pe[C:] = rng.permutation(err[C:])
            v = thn + pe
            cands.append((f"null_permerr_{j}", torch.as_tensor(v, device=dev, dtype=f64)))
        for j in (0,):
            u = rng.standard_normal(2 * C)
            gnoise = np.empty_like(err)
            for sl in (slice(0, C), slice(C, 2 * C)):
                gnoise[sl] = u[sl] / np.linalg.norm(u[sl]) * np.linalg.norm(err[sl])
            v = np.maximum(thn + gnoise, 1e-3 * thn)
            cands.append((f"null_l2match_{j}", torch.as_tensor(v, device=dev, dtype=f64)))
        R["null_construction"] = {
            "source": "theta_round5.npz round5_norm_s90210_sF0.0039|T8|eiv_box",
            "err_l2_E": float(np.linalg.norm(err[:C])), "err_l2_g": float(np.linalg.norm(err[C:])),
            "rel_l2_total": float(np.linalg.norm(err) / np.linalg.norm(thn))}

        R["all_candidates"] = [nm for nm, _ in cands]
        if a.only:
            want = set(a.only.split(","))
            cands = [c for c in cands if c[0] in want]
        mine = [c for i, c in enumerate(cands)
                if a.nshards == 1 or i % a.nshards == a.shard or c[0] == "theta_true"]
        log(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(cands)}: "
            + ", ".join(nm for nm, _ in mine))

        R["candidates"] = {}
        log(f"    {'candidate':<34s} {'medE':>7s} {'p90':>6s} {'relL2':>6s} {'corr':>6s} "
            f"{'neg':>4s} {'hold1f':>7s} | {'raw':>8s} {'kE':>6s} {'kg':>6s} {'gauged':>8s} "
            f"{'+-':>5s} {'R2':>8s} {'rms/dx':>7s}")
        for name, theta in mine:
            tc = time.time()
            ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
            hc, hn = holdout(theta)
            raw = scored(theta)

            def probe(lE, lg, theta=theta):
                return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

            gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"])
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                else scored(scale2(theta, kE, kg, C))
            R["candidates"][name] = {
                "param": ps, "holdout_1frame_cleanF": hc, "holdout_1frame_noisyF": hn,
                "raw": raw, "gauged": gau,
                "gauge": {k: v for k, v in gf.items() if k != "cells"},
                "gauge_cells": gf["cells"], "seconds": time.time() - tc}
            log(f"    {name:<34s} {ps['med_E']:>7.4f} {ps['p90_E']:>6.3f} {ps['rel_l2']:>6.3f} "
                f"{ps['corr_E']:>6.3f} {ps['n_negE']:>4d} {hc:>7.4f} | "
                f"{CT.fmt(raw['loop'],8)} {kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['loop'],8)} "
                f"{gf['gauge_uncertainty']:>5.3f} {CT.fmt(gau['R2'],8)} "
                f"{gau['rms_dx_mean']:>7.4f}  [{time.time()-tc:.0f}s]")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}_s{a.shard}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}_s{a.shard}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}_s{a.shard}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""state_refscore.py -- TASK 3.  Two things state_combine.py's stage `r` did not do:

  --mode bank   score the ZERO-INFORMATION NULL BANK and a null_permerr / null_l2match built from
                THIS candidate's own (much larger) error vector, on exactly the same scorer, so the
                headline can be compared with a floor measured at the same error magnitude.
  --mode gauge  re-gauge the two headline candidates with 3 different iteration budgets
                (5x5+2 refine, 7x7+2, 9x9+4) and report how far the score moves.

Everything else -- reference, rollout, margin 20, metrics.py -- is state_combine.score_stage's code
path, imported, not re-implemented.
"""
from __future__ import annotations

import argparse
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

from assemble import SUBSTEP_TOKENS                                    # noqa: E402
from recover import install_E                                         # noqa: E402
import metrics as MET                                                  # noqa: E402
import crash_test as CT                                                # noqa: E402
from crash_round2 import percell_amplitude, r2_percell                 # noqa: E402
from crash_round3 import scale2, t2_of                                 # noqa: E402
from finject import record_substeps, lerp, y_of                        # noqa: E402
from refute_round3 import advance                                      # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                          # noqa: E402
from round5_solve import pstats                                        # noqa: E402
from round5_score import gauge_grid                                    # noqa: E402
from refute5_fit import NoiseF                                         # noqa: E402
import state_derive as SD                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bank", choices=("bank", "gauge"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--noise", default="grid")
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--npz", default="state_theta_combine.npz")
    ap.add_argument("--ref", default="grid48_der_s90210|b1|eiv_box")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    tag = a.tag or f"state_refscore_{a.mode}"

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"args": vars(a)}
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
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
        a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        snap0 = {k: getattr(sy, k).clone() for k in SNAP}
        log(f"[reference] built [{time.time()-t_start:.0f}s]")

        # ---- held-out frame, four observability conventions (state_combine's holdout()) --------- #
        hk = a.holdout_tick
        sy.restore()
        advance(sy, W, hk - 1)
        sy._snapshot(hk - 1)
        HB = {}
        for t in (hk - 1, hk, hk + 1):
            if t > hk - 1:
                sy.restore()
                advance(sy, t - 1, t)
                sy._snapshot(t)
            Fs, _, Xs = record_substeps(sy, n)
            HB[t] = {"x0": sy.x0.clone(), "F0": sy.F0.clone(), "F1": Fs[-1].clone(),
                     "x_next": Xs[-1].clone(),
                     "snap": {k: getattr(sy, k).clone() for k in SNAP}}
        hx0, hF0, hF1, hxn = HB[hk]["x0"], HB[hk]["F0"], HB[hk]["F1"], HB[hk]["x_next"]
        y_obs_h = (hxn - hx0).reshape(-1)
        injh = lerp(hF0, hF1, n)
        dt = sy.dt
        NF = NoiseF(a.noise, x0, a.nodes, sy.device, sy.dtype)
        gh = torch.Generator(device=sy.device).manual_seed(555001)
        eFh = {t: (SIGMA_F / 2.0) * NF(gh) for t in (hk - 1, hk, hk + 1)}
        exh = {t: SIGMA_X * torch.randn(x0.shape, generator=gh, device=sy.device, dtype=sy.dtype)
               for t in (hk - 1, hk, hk + 1)}
        injh_noisy = lerp(hF0 + eFh[hk], hF1 + eFh[hk + 1], n)
        y_obs_h_noisy = ((hxn + exh[hk + 1]) - hx0).reshape(-1)
        v_der_c, C_der_c, _ = SD.derived_state(HB, hk, dt)
        v_der_n, C_der_n, _ = SD.derived_state(HB, hk, dt, eFh, exh)

        def holdout(theta):
            o = {}
            for nm, (vv, CC, inj, yo) in {
                    "cleanF_oracleState": (None, None, injh, y_obs_h),
                    "cleanF_derivedState": (v_der_c, C_der_c, injh, y_obs_h),
                    "noisyF_oracleState": (None, None, injh_noisy, y_obs_h_noisy),
                    "noisyF_derivedState": (v_der_n, C_der_n, injh_noisy, y_obs_h_noisy)}.items():
                SD.install_state(sy, HB[hk]["snap"], vv, CC, Jp_one=(vv is not None))
                y = y_of(sy, theta, n, inj, None)
                o[nm] = float((y - yo).norm() / yo.norm())
            return o

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

        Z = np.load(os.path.join(HERE, a.npz))
        ref_theta = torch.as_tensor(Z[a.ref], device=dev, dtype=f64)

        # ------------------------------------------------------------------ candidates --------- #
        cands = []
        if a.mode == "bank":
            def const(E, g):
                return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                                  torch.full((C,), float(g), device=dev, dtype=f64)])
            for E in (40, 90, 320):
                cands.append((f"bank_blind_E{E}_g1", const(E, 1.0)))
            cands.append(("bank_blind_E130_g0.95", const(130.0, 0.95)))
            for sd in (101, 303):
                gg = torch.Generator().manual_seed(sd)
                Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
                gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)
                cands.append((f"bank_prior_draw_{sd}", torch.cat([Ed, gd])))
            gnull = torch.Generator().manual_seed(4242)
            idx = torch.randperm(C, generator=gnull)[:45].to(dev)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gnull,
                                                                   dtype=f64)).to(dev)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gnull,
                                                                   dtype=f64)).to(dev)
            nm0 = th.clone()
            nm0[idx] = Ed[idx]
            nm0[C + idx] = gd[idx]
            cands.append(("null_med0_rand45", nm0))
            # nulls MATCHED to the derived-state candidate's own error vector
            err = (ref_theta - th).cpu().numpy()
            thn = th.cpu().numpy()
            rng = np.random.default_rng(20260809)
            for j in (0, 1):
                pe = np.empty_like(err)
                pe[:C] = rng.permutation(err[:C])
                pe[C:] = rng.permutation(err[C:])
                cands.append((f"null_permerr_der_{j}",
                              torch.as_tensor(thn + pe, device=dev, dtype=f64)))
            u = rng.standard_normal(2 * C)
            gn = np.empty_like(err)
            for sl in (slice(0, C), slice(C, 2 * C)):
                gn[sl] = u[sl] / np.linalg.norm(u[sl]) * np.linalg.norm(err[sl])
            cands.append(("null_l2match_der_0",
                          torch.as_tensor(np.maximum(thn + gn, 1e-3 * thn), device=dev, dtype=f64)))
            R["null_construction"] = {"source": a.ref,
                                      "rel_l2_total": float(np.linalg.norm(err)
                                                            / np.linalg.norm(thn))}
        else:
            for k in ("grid48_der_s90210|b1|eiv_box", "grid48_der_s555|b1|eiv_box"):
                cands.append((k, torch.as_tensor(Z[k], device=dev, dtype=f64)))

        if a.only:
            want = set(a.only.split(","))
            cands = [c for c in cands if c[0] in want]
        log(f"[{a.mode}] {len(cands)} candidates: " + ", ".join(nm for nm, _ in cands))
        R["candidates"] = {}

        if a.mode == "bank":
            log(f"    {'candidate':<26s} {'medE':>7s} {'relL2':>6s} {'hold_cO':>7s} {'hold_cD':>7s} "
                f"{'hold_nD':>7s} | {'raw':>8s} {'gauged':>8s} {'band_lo':>8s} {'band_hi':>8s}")
            for name, theta in cands:
                tc = time.time()
                ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
                hd = holdout(theta)
                raw = scored(theta)

                def probe(lE, lg, theta=theta):
                    return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

                gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"])
                kE, kg = gf["k_E"], gf["k_g"]
                gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                    else scored(scale2(theta, kE, kg, C))
                R["candidates"][name] = {"param": ps, "holdout": hd, "raw": raw, "gauged": gau,
                                         "gauge": {k: v for k, v in gf.items() if k != "cells"},
                                         "seconds": time.time() - tc}
                bl, bh = gf["loop_spread_within_10pct"] or (float("nan"), float("nan"))
                log(f"    {name:<26s} {ps['med_E']:>7.4f} {ps['rel_l2']:>6.3f} "
                    f"{hd['cleanF_oracleState']:>7.4f} {hd['cleanF_derivedState']:>7.4f} "
                    f"{hd['noisyF_derivedState']:>7.4f} | {CT.fmt(raw['loop'],8)} "
                    f"{CT.fmt(gau['loop'],8)} {CT.fmt(bl,8)} {CT.fmt(bh,8)} [{time.time()-tc:.0f}s]")
        else:
            budgets = [("g5s0.8r2", dict(gspan=0.8, gn=5, refine=2)),
                       ("g7s0.8r2", dict(gspan=0.8, gn=7, refine=2)),
                       ("g9s1.0r4", dict(gspan=1.0, gn=9, refine=4))]
            log(f"    {'candidate':<30s} {'budget':<10s} {'kE':>6s} {'kg':>6s} {'resid':>8s} "
                f"{'conv':>5s} {'gauged':>8s} {'band_lo':>8s} {'band_hi':>8s} {'orbitmax':>8s}")
            for name, theta in cands:
                raw = scored(theta)
                R["candidates"][name] = {"raw": raw, "budgets": {}}

                def probe(lE, lg, theta=theta):
                    return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

                for bn, kw in budgets:
                    tc = time.time()
                    gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"], **kw)
                    kE, kg = gf["k_E"], gf["k_g"]
                    gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                        else scored(scale2(theta, kE, kg, C))
                    orbit = [c["loop"] for c in gf["cells"] if isinstance(c["loop"], float)]
                    bl, bh = gf["loop_spread_within_10pct"]
                    R["candidates"][name]["budgets"][bn] = {
                        "gauge": {k: v for k, v in gf.items() if k != "cells"},
                        "gauged_loop": gau["loop"], "gauged_R2": gau["R2"],
                        "orbit_max_loop": float(np.max(orbit)), "seconds": time.time() - tc}
                    log(f"    {name:<30s} {bn:<10s} {kE:>6.3f} {kg:>6.3f} "
                        f"{gf['resid_logmax']:>8.4f} {str(gf['converged']):>5s} "
                        f"{CT.fmt(gau['loop'],8)} {CT.fmt(bl,8)} {CT.fmt(bh,8)} "
                        f"{np.max(orbit):>8.4f} [{time.time()-tc:.0f}s]")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"wrote {tag}.json [{R['wall_seconds']:.0f}s]")


if __name__ == "__main__":
    main()

"""round5_score.py -- ROUND 5, stage 3.  THE CRASH TEST, run cleanly on every candidate.

WHAT IS SCORED AND HOW
====================================================================================================
Every candidate theta is reinstalled at the tick-165 snapshot and FREE-RUN 150 frames (one beat),
anchor=None, and read on the registry's 10x10 grid at MARGIN_SAFE = 20 with
discovery_cardio_mpm/metrics.py imported unmodified.  Reported, never cited (cite() refuses all
five: four PROVISIONAL instruments and loopscore as the OBJECTIVE).

Three changes of hygiene that rounds 2-4 asked for, all in the SCORING, none in the estimator:

  1. THE GAUGE IS DETERMINISTIC.  round 4's diagnosis measured that `gauge_fix2` returns the iterate
     with the smallest GAUGE residual and can hand back a point far worse than raw (T8/eiv_snr0:
     0.0037 from Broyden vs 0.389 from a grid).  Here the gauge is a FIXED BUDGET: a 5x5 log-grid
     over (log k_E, log k_g) in [-0.8, 0.8]^2 (the centre cell IS the raw rollout, already paid
     for), then 2 Broyden refinement steps seeded at the best grid cell.  28 rollouts for every
     candidate, no early exit, no optimiser-dependent answer -- and the loopscore of every grid cell
     is recorded, so THE GAUGE'S OWN UNCERTAINTY (the spread of loopscore over the cells that
     satisfy the gauge) is reported next to the score.
  2. THE HELD-OUT ONE-FRAME RESIDUAL is reported beside every candidate and is round 5's acceptance
     statistic.  It needs b, not theta_true, so it exists on the recording.  Both flavours: with the
     clean F and with a measured (noisy) F at the recording's sigma_F.
  3. THE ZERO-INFORMATION NULL BANK is scored identically, and `null_med0_rand45` (theta_true with
     45 of 100 cells replaced by prior draws -- med|dE/E| = 0.0000 by construction) is in it.

usage:
  PYTHONPATH=/workspace/Plexus/src python round5_score.py --device cuda:1 --shard 0 --nshards 2
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


# --------------------------------------------------------------------------------------------- #
def gauge_grid(probe, raw_t, raw_loop, gspan=0.8, gn=5, refine=2, tol=0.01):
    """Deterministic fixed-budget 2-D gauge: 5x5 log-grid (centre = the raw point) + 2 Broyden steps.

    probe(lE, lg) -> dict with t1, t2, loop.  Returns the chosen (k_E, k_g), the residual, the whole
    grid, and the spread of loopscore among the cells that come within `near` of the targets.
    """
    ls = np.linspace(-gspan, gspan, gn)
    cells = []

    def rec(le, lg, t1, t2, loop):
        r = float(max(abs(math.log(max(t1, 1e-12))), abs(math.log(max(t2, 1e-12)))))
        cells.append({"kE": math.exp(le), "kg": math.exp(lg), "t1": t1, "t2": t2,
                      "loop": loop, "resid": r})
        return r

    rec(0.0, 0.0, raw_t[0], raw_t[1], raw_loop)
    for le in ls:
        for lg in ls:
            if abs(le) < 1e-12 and abs(lg) < 1e-12:
                continue
            d = probe(le, lg)
            rec(le, lg, d["t1"], d["t2"], d["loop"])
    best = min(cells, key=lambda c: c["resid"])
    x = np.array([math.log(best["kE"]), math.log(best["kg"])])
    r = np.array([math.log(max(best["t1"], 1e-12)), math.log(max(best["t2"], 1e-12))])
    # Jacobian from the grid: central differences about the best cell where available
    h = ls[1] - ls[0]
    J = np.zeros((2, 2))
    for j in range(2):
        xp = x.copy()
        xp[j] += h
        d = probe(*xp)
        rec(xp[0], xp[1], d["t1"], d["t2"], d["loop"])
        rj = np.array([math.log(max(d["t1"], 1e-12)), math.log(max(d["t2"], 1e-12))])
        J[:, j] = (rj - r) / h
    hist = [(float(np.exp(x[0])), float(np.exp(x[1])), float(np.abs(r).max()))]
    for _ in range(refine):
        if abs(np.linalg.det(J)) < 1e-10:
            break
        step = -np.linalg.solve(J, r)
        nrm = np.abs(step).max()
        if nrm > 0.8:
            step = step * (0.8 / nrm)
        xn = np.clip(x + step, -2.0, 2.0)
        d = probe(*xn)
        rn = np.array([math.log(max(d["t1"], 1e-12)), math.log(max(d["t2"], 1e-12))])
        rec(xn[0], xn[1], d["t1"], d["t2"], d["loop"])
        hist.append((float(np.exp(xn[0])), float(np.exp(xn[1])), float(np.abs(rn).max())))
        dx_, dr_ = xn - x, rn - r
        if float(dx_ @ dx_) > 1e-14:
            J = J + np.outer(dr_ - J @ dx_, dx_) / float(dx_ @ dx_)
        x, r = xn, rn
    allc = cells
    bestc = min(allc, key=lambda c: c["resid"])
    ltol = math.log(1.0 + 10 * tol)                      # "satisfies the gauge" = within 10 %
    near = [c for c in allc if c["resid"] <= ltol and isinstance(c["loop"], float)]
    loops = [c["loop"] for c in near]
    return {"k_E": bestc["kE"], "k_g": bestc["kg"], "resid_logmax": bestc["resid"],
            "converged": bool(bestc["resid"] <= math.log(1.0 + tol)),
            "n_probe": len(allc) - 1, "history": hist,
            "n_cells_within_10pct": len(near),
            "loop_spread_within_10pct": ([float(np.min(loops)), float(np.max(loops))]
                                         if loops else None),
            "gauge_uncertainty": (float(np.max(loops) - np.min(loops)) if len(loops) > 1 else 0.0),
            "cells": allc}


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round5_score")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=2)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--refine", type=int, default=2)
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
        R["anchor_fraction"] = float(anchor.double().mean())

        dummy, _ = MET.population(G=24, M=20)
        R["cite_status"] = {}
        for nm in CT.INSTRUMENTS + (CT.OBJECTIVE,):
            try:
                MET.REGISTRY[nm].cite(dummy, dummy)
                R["cite_status"][nm] = "cite() permitted"
            except Exception as e:
                R["cite_status"][nm] = f"{type(e).__name__}: {str(e)[:80]}"

        # ---- the reference window B --------------------------------------------------------------
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
        real10 = ref_full[:, tracers[MET.MARGIN_INHERITED]].cpu().numpy()
        a_ref, nk = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        snap0 = {k: getattr(sy, k).clone() for k in SNAP}
        R["a_ref_percell"] = a_ref.tolist()
        R["keep_percell"] = keep.tolist()
        R["reference"] = {"max_disp_dx": float(d_ref.norm(dim=-1).max() / dx),
                          "n_cells_kept": int(keep.sum())}

        # cell centroids, for the figure
        cx = np.zeros(C)
        cy = np.zeros(C)
        for c in range(1, C + 1):
            m = cid == c
            cx[c - 1], cy[c - 1] = float(x0[m, 0].mean()), float(x0[m, 1].mean())
        R["cell_centroid_x"], R["cell_centroid_y"] = cx.tolist(), cy.tolist()

        # ---- the HELD-OUT frame ------------------------------------------------------------------
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
        log(f"[holdout] tick {a.holdout_tick}; |y_obs| {float(y_obs_h.norm()):.4e}")

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
                   "rms_dx_mean": coarse["rms_pos_err_dx_mean"],
                   "rms_dx_final": coarse["rms_pos_err_dx_final"]}
            if full_out:
                ah, _ = percell_amplitude(full, x0, cid, C, interior)
                pc = r2_percell(ah, a_ref, keep)
                out.update({"margin20": m20, "coarse": coarse, "percell": pc,
                            "a_percell": ah.tolist(),
                            "margin10_loop": CT.read_metrics(
                                tr[MET.MARGIN_INHERITED].cpu().numpy(), real10)["loopscore"],
                            "tracks20": tr[MET.MARGIN_SAFE].cpu().numpy()})
                del full
            return out

        # ---- candidates ---------------------------------------------------------------------- #
        Z = np.load(os.path.join(HERE, "theta_round5.npz"))
        p40 = os.path.join(HERE, "theta_round5_box40k.npz")
        Z40 = np.load(p40) if os.path.exists(p40) else None

        def get(k, src=None):
            src = Z if src is None else src
            return torch.as_tensor(src[k], device=dev, dtype=f64)

        SF = f"{SIGMA_F:g}"
        cands = [("theta_true", th)]
        for k in ("naive", "naive_box"):
            key = f"round5_norm_clean|T8|{k}"
            if key in Z.files:
                cands.append((f"clean/T8/{k}", get(key)))
        for sd in (90210, 555, 777):
            for k in ("naive", "eiv_snr0", "naive_box", "eiv_box"):
                key = f"round5_norm_s{sd}_sF{SF}|T8|{k}"
                if key in Z.files:
                    cands.append((f"s{sd}/T8/{k}", get(key)))
        for k in ("naive", "eiv_box"):                      # does the stack matter? T=1 vs T=8
            key = f"round5_norm_s90210_sF{SF}|T1|{k}"
            if key in Z.files:
                cands.append((f"s90210/T1/{k}", get(key)))
        for k in ("naive", "eiv_box"):                      # the two-estimate disagreement level
            key = f"round5_norm_s90210_sF0.0327|T8|{k}"
            if key in Z.files:
                cands.append((f"hiF/T8/{k}", get(key)))
        # the SAME box QP solved to convergence (40000 FISTA iterations instead of 4000): the
        # eiv_box iterate moves 0.139 in relative l2 between the two budgets, so both are scored
        if Z40 is not None:
            for sd in (90210, 555, 777):
                for k in ("naive_box", "eiv_box"):
                    key = f"round5_norm_s{sd}_sF{SF}|T8|{k}"
                    if key in Z40.files:
                        cands.append((f"s{sd}/T8/{k}40k", get(key, Z40)))

        # extra candidates whose npz keys ARE the candidate names (box-width sensitivity etc.)
        pex = os.path.join(HERE, "theta_round5_extra.npz")
        if os.path.exists(pex):
            Zex = np.load(pex)
            for k in Zex.files:
                cands.append((k, get(k, Zex)))

        # ---- the zero-information NULL BANK + the median-matched null ---------------------------
        def const(E, g):
            return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                              torch.full((C,), float(g), device=dev, dtype=f64)])

        BANK = [(f"bank_blind_E{E}_g1", const(E, 1.0)) for E in (40, 90, 320)]
        BANK.append(("bank_blind_E130_g0.95", const(130.0, 0.95)))
        for sd in (101, 303):
            gg = torch.Generator().manual_seed(sd)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            BANK.append((f"bank_prior_draw_{sd}", torch.cat([Ed, gd])))
        gnull = torch.Generator().manual_seed(4242)
        idx = torch.randperm(C, generator=gnull)[:45].to(dev)
        Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gnull,
                                                               dtype=f64)).to(dev)
        gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gnull,
                                                               dtype=f64)).to(dev)
        nm0 = th.clone()
        nm0[idx] = Ed[idx]
        nm0[C + idx] = gd[idx]
        BANK.append(("null_med0_rand45", nm0))
        R["bank_names"] = [nm for nm, _ in BANK]
        cands = cands + BANK
        R["all_candidates"] = [nm for nm, _ in cands]
        if a.only:
            want = set(a.only.split(","))
            cands = [c for c in cands if c[0] in want]
        mine = [c for i, c in enumerate(cands)
                if a.nshards == 1 or i % a.nshards == a.shard or c[0] == "theta_true"]
        log(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(cands)} candidates: "
            + ", ".join(nm for nm, _ in mine))

        # ---- the array-only nulls (no rollout) ---------------------------------------------------
        if a.shard == 0:
            R["nulls"] = {}
            frozen = np.repeat(x0[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None], G, axis=0)
            R["nulls"]["do_nothing"] = CT.read_metrics(frozen, real20)
            R["nulls"]["replay_previous_beat"] = CT.read_metrics(
                recA[:, tracers[MET.MARGIN_SAFE]].cpu().numpy(), real20)
            R["nulls"]["identity"] = CT.read_metrics(real20, real20)
            d0 = d_ref[:, interior]
            dA = (recA - x0[None])[:, interior]
            R["nulls"]["do_nothing"]["R2"] = float(1.0 - d0.pow(2).sum() / ss_tot)
            R["nulls"]["replay_previous_beat"]["R2"] = float(1.0 - (dA - d0).pow(2).sum() / ss_tot)
            a_rep, _ = percell_amplitude(recA, x0, cid, C, interior)
            R["nulls"]["replay_previous_beat"]["percell"] = r2_percell(a_rep, a_ref, keep)
            R["nulls"]["replay_previous_beat"]["a_percell"] = a_rep.tolist()
            R["campaign_nulls"] = {"loopscore_predict_nothing": 0.070,
                                   "loopscore_replay_fit_beat": 0.851,
                                   "loopscore_replay_heldout": 0.62,
                                   "note": "measured on the REAL recording; NOT commensurate with "
                                           "the synthetic rows -- the paired metrics carry the "
                                           "recording's units"}
            for k, v in R["nulls"].items():
                log(f"    null {k:<24s} loopscore {CT.fmt(v['loopscore'],8)}  "
                    f"coord {CT.fmt(v['coordination'],8)}  orient {CT.fmt(v['orientation_error'],8)}")

        # ---- the crash test ----------------------------------------------------------------------
        R["candidates"] = {}
        log(f"\n[crash test] {G}-frame free rollouts from tick {W}, margin-{MET.MARGIN_SAFE}, "
            f"anchor=None; deterministic {a.grid}x{a.grid} grid gauge + {a.refine} refine")
        log(f"    {'candidate':<24s} {'medE':>7s} {'neg':>4s} {'>5x':>4s} {'hold1f':>7s} "
            f"{'holdN':>7s} | {'raw':>8s} {'t1':>6s} {'kE':>6s} {'kg':>6s} {'gauged':>8s} "
            f"{'+-':>5s} {'R2':>8s} {'r2cell':>7s} {'rms/dx':>7s}")
        TRACKS = {}
        for name, theta in mine:
            tc = time.time()
            ps = pstats(theta.cpu().numpy(), th.cpu().numpy(), C)
            hc, hn = holdout(theta)
            raw = scored(theta)

            def probe(lE, lg, theta=theta):
                return scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)

            gf = gauge_grid(probe, (raw["t1"], raw["t2"]), raw["loop"],
                            gn=a.grid, refine=a.refine)
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (abs(kE - 1) < 1e-12 and abs(kg - 1) < 1e-12) \
                else scored(scale2(theta, kE, kg, C))
            gps = pstats(scale2(theta, kE, kg, C).cpu().numpy(), th.cpu().numpy(), C)
            rec = {"param": ps, "param_after_gauge": gps,
                   "holdout_1frame_cleanF": hc, "holdout_1frame_noisyF": hn,
                   "raw": {k: v for k, v in raw.items() if k != "tracks20"},
                   "gauged": {k: v for k, v in gau.items() if k != "tracks20"},
                   "gauge": {k: v for k, v in gf.items() if k != "cells"},
                   "gauge_cells": gf["cells"], "seconds": time.time() - tc}
            R["candidates"][name] = rec
            TRACKS[f"{name}|raw"] = raw["tracks20"]
            TRACKS[f"{name}|gauged"] = gau["tracks20"]
            log(f"    {name:<24s} {ps['med_E']:>7.4f} {ps['n_negE']:>4d} "
                f"{ps['n_cells_relE_gt5']:>4d} {hc:>7.4f} {hn:>7.4f} | "
                f"{CT.fmt(raw['loop'],8)} {raw['t1']:>6.3f} {kE:>6.3f} {kg:>6.3f} "
                f"{CT.fmt(gau['loop'],8)} {gf['gauge_uncertainty']:>5.3f} "
                f"{CT.fmt(gau['R2'],8)} "
                f"{(gau['percell']['r2'] if gau['percell']['r2'] is not None else float('nan')):>7.4f}"
                f" {gau['rms_dx_mean']:>7.4f}  [{time.time()-tc:.0f}s]")

        np.savez(os.path.join(HERE, f"tracks_{a.tag}_s{a.shard}.npz"),
                 real20=real20, a_ref=a_ref, keep=keep, cx=cx, cy=cy,
                 theta_true=th.cpu().numpy(), **TRACKS)

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}_s{a.shard}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}_s{a.shard}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}_s{a.shard}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

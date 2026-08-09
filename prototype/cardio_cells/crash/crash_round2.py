"""crash_round2.py -- ROUND 2. The crash test, with ONE thing changed: THE SCORE.

WHAT WAS THE BIGGEST PROBLEM, AND WHAT IS CHANGED
====================================================================================================
Round 1 ranked twelve parameter vectors by rolling them out and reading the registry. Round 1's
diagnosis then showed that the ranking was a readout of ONE GLOBAL SCALAR -- the amplitude of the
sheet's motion. The measurement that settled it: theta_true with the whole gain block multiplied by
1.8 (per-cell structure EXACTLY right, med|dE/E| = 0) scored loopscore -0.176, statistically the
same as a per-cell-blind constant with no information at all (-0.180) and as the frame estimate
(-0.164). Across 42 candidates loopscore = 0.821 - 1.315*log(E_ratio) with R^2 0.853 for
over-shooting candidates. A score that cannot separate "perfect per-cell parameters, wrong global
amplitude" from "no per-cell parameters at all" cannot say anything about a per-cell estimator, and
that is the only question this investigation is asking.

THE ONE CHANGE, and it is in the scoring stage only:

    AMPLITUDE IS NOW A CALIBRATED NUISANCE, NOT PART OF THE SCORE.

    (a) `gauge_fix` -- before scoring, one global scalar k multiplying the GAIN BLOCK of theta is
        fitted by secant iteration on log k until the rollout's interior motion-energy ratio is
        1.000 +/- `tol`. Every candidate is then scored TWICE, raw and gauge-fixed. This is not
        oracle knowledge: the total motion energy of the recording is observable on real data (it
        is a sum over the PIV field), so one global scalar is genuinely calibratable there.
    (b) `percell_amplitude` / `r2_percell` -- the per-cell readout that was missing. Per cell c,
        a_c = median over that cell's interior particles of the peak |x(t) - x(t0)| over the
        window; r2_percell is the R^2 between a_hat_c/mean(a_hat) and a_c/mean(a) across cells.
        Normalising by the mean makes it amplitude-INVARIANT by construction, so it reads the
        per-cell pattern and nothing else.

NOTHING ELSE MOVES. Same system, same seeds, same t0 = tick 165, same 150-frame window, same
reading surface, same registry (imported, not modified), same rollout function (imported from
crash_test.py unmodified), same estimators (theta_hat_substep and theta_hat_frame are re-solved by
the same code; frame_DISP is read from round 1's archived npz, not re-derived). The candidate list
is round 1's, plus the five vectors the diagnosis named as the discriminating pair and triad.

usage:
  PYTHONPATH=/workspace/Plexus/src python crash_round2.py --device cuda:1 --shard 0 --nshards 2
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
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, DISC)
sys.path.insert(0, HERE)

from assemble import SUBSTEP_TOKENS, rel                        # noqa: E402
from recover import Solver, fd_accel, install_E, score          # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402


# =============================================================================================
#  THE ONE CHANGE, part (a): amplitude as a calibrated nuisance
# =============================================================================================
def scale_gain(theta, k, C):
    return torch.cat([theta[:C], theta[C:] * k])


def energy_ratio(sy, theta, jit, ctx):
    """One rollout, reading only the interior motion-energy ratio. No tracers, no full trajectory."""
    _, _, coarse = CT.rollout(sy, theta, ctx["W"], ctx["G"], {}, ref_full=ctx["ref_full"],
                              anchor=None, interior=ctx["interior"], ss_tot=ctx["ss_tot"],
                              jitter=jit)
    return coarse["motion_energy_ratio_interior"]


def gauge_fix(sy, theta, jit, ctx, r_raw, tol=0.005, iters=6):
    """Secant on log k (k multiplies the gain block) driving the motion-energy ratio to 1.

    r_raw : the ratio at k = 1, which the raw scored rollout has already paid for -- so the fit
            costs `iters` extra rollouts at most and typically two.
    Returns (k, ratio_at_k, history, n_extra_rollouts, status).
    """
    C = sy.C
    hist = [(1.0, r_raw)]
    if not np.isfinite(r_raw) or r_raw <= 0:
        return 1.0, r_raw, hist, 0, "no motion to calibrate"
    if abs(r_raw - 1.0) <= tol:
        return 1.0, r_raw, hist, 0, "already at 1"
    l0, y0 = 0.0, math.log(r_raw)
    # first step: energy ~ amplitude^2 and amplitude ~ gain, so log k = -log(ratio)/2
    l1 = -0.5 * y0
    best = (1.0, r_raw, abs(y0))
    n = 0
    status = "max iters"
    for _ in range(iters):
        l1 = float(np.clip(l1, -4.0, 4.0))
        r1 = energy_ratio(sy, scale_gain(theta, math.exp(l1), C), jit, ctx)
        n += 1
        hist.append((math.exp(l1), r1))
        if not np.isfinite(r1) or r1 <= 0:
            status = "diverged"
            break
        y1 = math.log(r1)
        if abs(y1) < best[2]:
            best = (math.exp(l1), r1, abs(y1))
        if abs(r1 - 1.0) <= tol:
            status = "converged"
            break
        if abs(y1 - y0) < 1e-12:
            status = "flat"
            break
        l2 = l1 - y1 * (l1 - l0) / (y1 - y0)
        l0, y0, l1 = l1, y1, l2
    return best[0], best[1], hist, n, status


# =============================================================================================
#  THE ONE CHANGE, part (b): a per-cell readout
# =============================================================================================
def percell_amplitude(full, x0, cid, C, keep_particle, min_n=5):
    """a_c = median over cell c's kept particles of the peak displacement from x0 over the window.

    full : [G, Np, 2] trajectory. Returns (a [C] with nan where a cell has < min_n kept particles,
    n_kept [C]).
    """
    peak = (full - x0[None]).norm(dim=-1).max(0).values            # [Np]
    a = np.full(C, np.nan)
    nk = np.zeros(C, dtype=int)
    for c in range(1, C + 1):
        m = keep_particle & (cid == c)
        nk[c - 1] = int(m.sum())
        if nk[c - 1] >= min_n:
            a[c - 1] = float(peak[m].median())
    return a, nk


def r2_percell(a_hat, a_ref, keep):
    """R^2 between the two per-cell amplitude fields, each divided by its own mean.

    Amplitude-invariant by construction: scaling a_hat by any constant leaves this unchanged.
    Its zero is "predict the mean", i.e. a flat field; 1 is exact.
    """
    ah, ar = a_hat[keep], a_ref[keep]
    if not (np.isfinite(ah).all() and np.isfinite(ar).all()) or ah.mean() <= 0 or ar.mean() <= 0:
        return {"r2": None, "corr": None, "sse": None}
    ah = ah / ah.mean()
    ar = ar / ar.mean()
    sse = float(((ah - ar) ** 2).sum())
    sst = float(((ar - ar.mean()) ** 2).sum())
    return {"r2": float(1.0 - sse / sst), "corr": float(np.corrcoef(ah, ar)[0, 1]),
            "sse": sse, "sst": sst}


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.cells, a.per_parent, a.n_grid, a.warmup, a.window = 24, 40, 64, 24, 30

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.warmup, window=a.window, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "shard": [a.shard, a.nshards], "gauge_tol": a.tol,
         "ONE_CHANGE": "scoring only: (a) global amplitude gauge-fixed before scoring, "
                       "(b) per-cell amplitude R^2 reported. Estimators, system, reading surface "
                       "and registry unchanged."}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        # ---------------------------------------------------------------- system + reference ----
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G = sy.C, args.warmup, args.window
        th = sy.theta_true.double()
        dx, x0 = sy.g.dx, sy.x0.clone()
        cid = sy.cid

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
            for _ in range(sy.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            ref_full[k] = sy.p.get("pos")
        d_ref = ref_full - x0[None]
        dm = d_ref[:, interior].mean(0, keepdim=True)
        ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
        ref_tr = {m: ref_full[:, t] for m, t in tracers.items()}
        ctx = {"W": W, "G": G, "ref_full": ref_full, "interior": interior, "ss_tot": ss_tot}

        a_ref, n_kept = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        R["percell"] = {"n_cells": C, "n_cells_kept": int(keep.sum()),
                        "min_interior_particles_per_kept_cell": int(n_kept[keep].min()),
                        "a_ref_world": [float(np.nanmin(a_ref)), float(np.nanmedian(a_ref)),
                                        float(np.nanmax(a_ref))],
                        "a_ref_over_dx": [float(np.nanmin(a_ref) / dx),
                                          float(np.nanmedian(a_ref) / dx),
                                          float(np.nanmax(a_ref) / dx)]}
        log(f"[percell] {int(keep.sum())}/{C} cells have >=5 interior particles; reference peak "
            f"displacement per cell: {np.nanmin(a_ref)/dx:.3f} .. {np.nanmedian(a_ref)/dx:.3f} .. "
            f"{np.nanmax(a_ref)/dx:.3f} dx")

        # is the registry citable?  (recorded, not assumed -- unchanged from round 1)
        dummy, _ = MET.population(G=24, M=20)
        R["cite_status"] = {}
        for n in CT.INSTRUMENTS + (CT.OBJECTIVE,):
            try:
                MET.REGISTRY[n].cite(dummy, dummy)
                R["cite_status"][n] = "permitted"
            except Exception as e:
                R["cite_status"][n] = f"{type(e).__name__}: {str(e)[:80]}"

        # ---------------------------------------------------------------- the estimators --------
        # unchanged code paths; re-solved so this run is self-contained
        R["solves"] = {}
        sy.restore()
        a_solver = sy.step(sy.E_true, sy.gain_true, n_sub=1)
        x_next = sy.p.get("pos").clone()
        a_fd = fd_accel(sy.x_prev, x0, x_next, sy.dt_sub)
        A1, a01, t1 = sy.assemble(n_sub=1)
        S1 = Solver(A1, C)
        theta_sub = S1(a_fd - a01)["ridge0"]
        R["solves"]["substep"] = {"residual": rel(A1 @ sy.theta_true - (a_solver - a01),
                                                  a_solver - a01), "cond": S1.cond,
                                  "score": score(theta_sub, th, C), "assembly_s": t1}
        S1.free(); del A1, S1
        torch.cuda.empty_cache()

        nsf = sy.n_sub_per_frame
        x_fprev, x_fnext = recA[-2].clone(), ref_full[0].clone()
        sy.restore()
        a_solver_f = sy.step(sy.E_true, sy.gain_true, n_sub=nsf)
        a_fdf = fd_accel(x_fprev, x0, x_fnext, sy.dt)
        Af, a0f, tf = sy.assemble(n_sub=nsf)
        Sf = Solver(Af, C)
        solf = Sf(a_fdf - a0f)
        theta_frm, theta_frm_r = solf["ridge0"], solf["ridge0.01"]
        R["solves"]["frame"] = {"residual": rel(Af @ sy.theta_true - (a_solver_f - a0f),
                                                a_solver_f - a0f), "cond": Sf.cond,
                                "score": score(theta_frm, th, C), "assembly_s": tf}
        Sf.free(); del Af, Sf
        torch.cuda.empty_cache()
        log(f"[solve] substep med|dE/E| {R['solves']['substep']['score']['med_E']:.3e}; "
            f"frame med|dE/E| {R['solves']['frame']['score']['med_E']:.4f}")

        # ---------------------------------------------------------------- candidates ------------
        dev, f64 = th.device, torch.float64

        def const(E, g):
            return torch.cat([torch.full((C,), E, device=dev, dtype=f64),
                              torch.full((C,), g, device=dev, dtype=f64)])

        gp = torch.Generator(device=str(dev)).manual_seed(77)
        u = torch.randn(2 * C, generator=gp, device=dev, dtype=f64)
        u = u / u.abs().max()
        pm = torch.randperm(C, device=dev)

        cands = [("theta_true", th, 0.0),
                 ("theta_hat_substep", theta_sub, 0.0),
                 ("theta_hat_frame_ridge0", theta_frm, 0.0),
                 ("theta_hat_frame_ridge1e-2", theta_frm_r, 0.0),
                 ("theta_const_E130_g1", const(130.0, 1.0), 0.0),
                 ("theta_shuffled_true", torch.cat([th[:C][pm], th[C:][pm]]), 0.0)]
        for e in (0.01, 0.03, 0.1, 0.3):
            cands.append((f"theta_true_perturbed_{e:g}", th * (1.0 + e * u), 0.0))
        for j in (0.01, 0.1):
            cands.append((f"theta_true_x0jitter_{j:g}dx", th, j * dx))
        # ---- the five the diagnosis named (no new estimator; frame_DISP is read from round 1) ---
        npz = os.path.join(HERE, "theta_refute1.npz")
        R["frame_DISP_source"] = npz
        if os.path.exists(npz) and not a.smoke:
            z2 = np.load(npz)
            t_disp = torch.as_tensor(z2["cand::frame_DISP"], device=dev, dtype=f64)
            kE = float((t_disp[:C] * th[:C]).sum() / (t_disp[:C] ** 2).sum())
            kg = float((t_disp[C:] * th[C:]).sum() / (t_disp[C:] ** 2).sum())
            cands.append(("frame_DISP", t_disp, 0.0))
            cands.append(("frame_DISP_oracle_rescale", torch.cat([t_disp[:C] * kE,
                                                                  t_disp[C:] * kg]), 0.0))
            R["oracle_rescale_k_frame_DISP"] = [kE, kg]
        cands.append(("blind_E130_g0.95", const(130.0, 0.95), 0.0))
        cands.append(("blind_E40_g1", const(40.0, 1.0), 0.0))
        cands.append(("true_gain_x1.8", torch.cat([th[:C], th[C:] * 1.8]), 0.0))

        R["candidate_theta_error"] = {n: score(t, th, C) for n, t, _ in cands}
        for n, t, _ in cands:
            R["candidate_theta_error"][n].update(
                {"mean_E": float(t[:C].mean()), "mean_gain": float(t[C:].mean()),
                 "n_negative_E": int((t[:C] < 0).sum())})
        if a.shard == 0:
            np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                     **{f"cand::{n}": t.cpu().numpy() for n, t, _ in cands})

        mine = [c for i, c in enumerate(cands) if a.nshards == 1 or
                i % a.nshards == a.shard or c[0] == "theta_true"]
        log(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(cands)} candidates: "
            + ", ".join(n for n, _, _ in mine))

        # ---------------------------------------------------------------- the crash test --------
        def scored(theta, jit):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full, anchor=None,
                                          interior=interior, ss_tot=ss_tot, jitter=jit,
                                          keep_full=True, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(),
                                  ref_tr[MET.MARGIN_SAFE].cpu().numpy())
            m10 = CT.read_metrics(tr[MET.MARGIN_INHERITED].cpu().numpy(),
                                  ref_tr[MET.MARGIN_INHERITED].cpu().numpy())
            a_hat, _ = percell_amplitude(full, x0, cid, C, interior)
            pc = r2_percell(a_hat, a_ref, keep)
            del full
            return {"margin20": m20, "margin10": m10, "coarse": coarse, "percell": pc,
                    "a_percell": a_hat.tolist()}

        R["rollouts"] = {}
        log(f"\n[crash test] {G}-frame FREE rollouts from tick {W}, margin-{MET.MARGIN_SAFE} grid; "
            f"each candidate scored RAW and GAUGE-FIXED (|E_ratio-1| <= {a.tol})")
        log(f"    {'candidate':<28s} {'medE':>7s} | {'gauge':<28s} | {'k':>6s} "
            f"{'Erat':>6s} {'loop':>8s} {'R2':>8s} {'r2cell':>7s} {'coord':>6s} {'orient':>6s} "
            f"{'rms/dx':>7s}")
        for name, theta, jit in mine:
            t_c = time.time()
            raw = scored(theta, jit)
            r_raw = raw["coarse"]["motion_energy_ratio_interior"]
            k, r_k, hist, n_extra, status = gauge_fix(sy, theta, jit, ctx, r_raw,
                                                      tol=a.tol, iters=a.iters)
            gau = raw if k == 1.0 else scored(scale_gain(theta, k, C), jit)
            rec = {"theta_error": R["candidate_theta_error"][name], "x0_jitter_world": jit,
                   "raw": raw, "gauged": gau,
                   "gauge": {"k": k, "ratio_at_k": r_k, "history": hist, "n_extra_rollouts": n_extra,
                             "status": status,
                             "theta_error_after_k": score(scale_gain(theta, k, C), th, C)},
                   "seconds": time.time() - t_c}
            R["rollouts"][name] = rec
            for tag, d in (("raw", raw), ("gauged", gau)):
                pr = d["percell"]["r2"]
                pcs = "n/a" if pr is None else f"{pr:.4f}"
                log(f"    {name:<28s} {rec['theta_error']['med_E']:>7.4f} | {tag:<28s} | "
                    f"{(k if tag=='gauged' else 1.0):>6.3f} "
                    f"{d['coarse']['motion_energy_ratio_interior']:>6.3f} "
                    f"{CT.fmt(d['margin20']['loopscore'],8)} "
                    f"{CT.fmt(d['coarse']['R2_displacement_interior'],8)} "
                    f"{pcs:>7s} "
                    f"{CT.fmt(d['margin20']['coordination'],6)} "
                    f"{CT.fmt(d['margin20']['orientation_error'],6)} "
                    f"{CT.fmt(d['coarse']['rms_pos_err_dx_mean'],7)}")

        # ---------------------------------------------------------------- nulls -----------------
        if a.shard == 0:
            log("\n[nulls]")
            R["nulls"] = {}
            real20 = ref_tr[MET.MARGIN_SAFE].cpu().numpy()
            frozen = np.repeat(x0[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None], G, axis=0)
            R["nulls"]["do_nothing"] = CT.read_metrics(frozen, real20)
            d0 = d_ref[:, interior]
            R["nulls"]["do_nothing"]["coarse"] = {
                "R2_displacement_interior": float(1.0 - d0.pow(2).sum() / ss_tot),
                "motion_energy_ratio_interior": 0.0}
            R["nulls"]["do_nothing"]["percell"] = {"r2": None,
                                                   "note": "no motion: a_c == 0 for every cell"}
            replay = recA[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
            R["nulls"]["replay_previous_beat"] = CT.read_metrics(replay, real20)
            dA = (recA - x0[None])[:, interior]
            a_rep, _ = percell_amplitude(recA, x0, cid, C, interior)
            R["nulls"]["replay_previous_beat"]["coarse"] = {
                "R2_displacement_interior": float(1.0 - (dA - d0).pow(2).sum() / ss_tot),
                "motion_energy_ratio_interior": float(dA.pow(2).sum() / d0.pow(2).sum())}
            R["nulls"]["replay_previous_beat"]["percell"] = r2_percell(a_rep, a_ref, keep)
            R["nulls"]["identity"] = CT.read_metrics(real20, real20)
            R["nulls"]["identity"]["percell"] = r2_percell(a_ref, a_ref, keep)
            # per-cell nulls that cost nothing to run
            rs = np.random.RandomState(11)
            sh = a_ref.copy()
            idx = np.where(keep)[0]
            sh[idx] = a_ref[idx][rs.permutation(idx.size)]
            R["nulls"]["percell_shuffled_reference"] = {"percell": r2_percell(sh, a_ref, keep)}
            flat = np.where(keep, float(np.nanmean(a_ref[keep])), np.nan)
            R["nulls"]["percell_flat_mean"] = {"percell": r2_percell(flat, a_ref, keep)}
            R["campaign_nulls"] = {"loopscore_predict_nothing": 0.0700,
                                   "loopscore_replay_fit_beat": 0.851,
                                   "loopscore_replay_heldout": 0.62,
                                   "note": "measured on the REAL recording; not commensurate with "
                                           "the synthetic rows above"}
            for k_, v in R["nulls"].items():
                pc = v.get("percell", {})
                pcs = f"{pc['r2']:.4f}" if isinstance(pc, dict) and pc.get("r2") is not None \
                    else "n/a"
                log(f"    {k_:<28s} loopscore {CT.fmt(v.get('loopscore'),8)}  "
                    f"coord {CT.fmt(v.get('coordination'),8)}  r2_percell {pcs:>8s}")

        R["a_ref_percell"] = a_ref.tolist()
        R["keep_percell"] = keep.tolist()

    R["wall_seconds"] = time.time() - t_start
    sfx = "" if a.nshards == 1 else f"_s{a.shard}"
    json.dump(R, open(os.path.join(HERE, f"crash_{a.tag}{sfx}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"crash_{a.tag}{sfx}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote crash_{a.tag}{sfx}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

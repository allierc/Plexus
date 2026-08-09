"""round4_diverge.py -- ROUND 4, parts 2 and 3.

PART A -- HOW does a rollout go wrong?  The four instruments were chosen to tell apart failures
that loopscore alone cannot, and so far this campaign has used them only as scores.  Here they are
used as a DIAGNOSIS.  Five canonical failure modes are built BY CONSTRUCTION out of the reference
trajectory itself, so the answer is known:

    blow_up      x0 + s*(ref - x0), s > 1          the loops are the right shape, too big
    collapse     s < 1 (and s = 0, do-nothing)     the right shape, too small
    phase_drift  the whole beat rolled by k frames one global timing offset, shape intact
    decoherence  every node rolled by its OWN k    the tissue stops beating together
    shape_loss   each loop rotated / flattened     right size, right timing, wrong path

Each is read through the same registry, on the same margin-20 tracers, giving a FINGERPRINT.  Real
rollouts are then read against those fingerprints, so "it diverged" becomes a statement with a
named mode and a number.  The per-frame rms error is fitted over the second half of the window to
say whether the error GROWS or SATURATES -- round 1 claimed saturation for the eps ladder and never
tested it for a candidate that is actually wrong.

PART B -- the border anchoring question, settled.  `CT.rollout(anchor=...)` overwrites the position
and velocity of the pinned band with the reference's at the end of every frame.  Round 1 measured
that at margin 20 it buys <= +0.0019 loopscore for every candidate but one (+0.0893) and every
round since has passed `anchor=None`.  That is a measurement on one candidate list with the OLD
(un-gauged) score.  Here every candidate is run BOTH ways, read at BOTH margins, on all five
numbers, with the anchor's own cost at theta_true, and the two rankings compared -- so the question
is either closed or it is not.

usage: PYTHONPATH=/workspace/Plexus/src python round4_diverge.py --device cuda:1
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
from recover import Solver, fd_accel, install_E, score           # noqa: E402
import metrics as MET                                            # noqa: E402
import crash_test as CT                                          # noqa: E402
from crash_round2 import percell_amplitude, r2_percell           # noqa: E402
from crash_round3 import gauge_fix2, scale2, t2_of               # noqa: E402


def lag_stats(sim, real):
    """The registry's own timing_lag, summarised two ways: the GLOBAL offset and the SPREAD.

    coordination is |mean exp(4i pi lag/G)| -- it is deliberately invariant to a global shift, so
    it cannot tell phase drift from a perfect match.  The median lag can, and is read here from
    exactly the same function the metric uses, so the two cannot drift apart.
    """
    lag, G = MET.timing_lag(np.asarray(sim, float), np.asarray(real, float))
    z = np.exp(4j * np.pi * lag / G)
    ang = np.angle(z.mean())
    return {"lag_median_frames": float(np.median(np.where(lag > G / 2, lag - G, lag))),
            "lag_global_halfperiod_frames": float(ang * G / (4 * np.pi)),
            "lag_iqr_frames": float(np.subtract(*np.percentile(
                np.where(lag > G / 2, lag - G, lag), [75, 25]))),
            "lag_concentration": float(np.abs(z.mean()))}


def fingerprint(sim, real):
    m = CT.read_metrics(sim, real)
    f = {k: m[k] for k in list(CT.INSTRUMENTS) + [CT.OBJECTIVE]}
    f["t2_path_over_peak"] = t2_of(m)
    pr, ps = m["peak_excursion_reading_real"], m["peak_excursion_reading_sim"]
    f["peak_ratio"] = ps / pr if pr > 0 else float("nan")
    f["path_ratio"] = (m["path_length_reading_sim"] / m["path_length_reading_real"]
                       if m["path_length_reading_real"] > 0 else float("nan"))
    f.update(lag_stats(sim, real))
    f["openness"] = float(MET.REGISTRY["openness"](sim, real))
    f["chirality_match"] = float(MET.REGISTRY["chirality_match"](sim, real))
    return f


def classify(f):
    """A rule, written down BEFORE the candidates were scored, not fitted to them.

    Thresholds come from the canonical modes in part A: peak_ratio is the amplitude axis,
    lag_global the phase axis, coordination the together-or-not axis, orientation_error and
    t2 the shape axes.  Order matters -- amplitude first, because loopscore's response to it
    swamps everything else (round 2).
    """
    pr = f.get("peak_ratio", float("nan"))
    tags = []
    if not np.isfinite(pr):
        return ["undefined"]
    if pr > 1.25:
        tags.append("blow_up")
    elif pr < 0.75:
        tags.append("collapse")
    if abs(f.get("lag_global_halfperiod_frames", 0.0)) > 1.0:
        tags.append("phase_drift")
    c = f.get("coordination")
    if isinstance(c, float) and c < 0.9:
        tags.append("decoherence")
    if (f.get("orientation_error", 0.0) > 0.15 or abs(f.get("t2_path_over_peak", 1.0) - 1) > 0.15
            or f.get("openness", 0.0) > 0.10):
        tags.append("shape_loss")
    return tags or ["intact"]


def growth(rms):
    """Does the per-frame error GROW or SATURATE?  Slope of the second half, relative to its mean."""
    r = np.asarray(rms, float)
    h = r[len(r) // 2:]
    t = np.arange(h.size, dtype=float)
    sl = float(np.polyfit(t, h, 1)[0])
    return {"rms_dx_final": float(r[-1]), "rms_dx_max": float(r.max()),
            "slope_2nd_half_per_frame": sl,
            "slope_rel_per_frame": float(sl / (h.mean() + 1e-300)),
            "final_over_max": float(r[-1] / (r.max() + 1e-300))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round4_diverge")
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--eiv-npz", default="theta_round4_eiv.npz")
    ap.add_argument("--no-anchor", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.warmup, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G = sy.C, args.warmup, args.window
        th = sy.theta_true.double()
        dx, x0, cid = sy.g.dx, sy.x0.clone(), sy.cid
        dev, f64 = th.device, torch.float64

        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ~anchor
        R["anchor_fraction"] = float(anchor.double().mean())
        R["probes_in_band"] = {str(m): int(anchor[t].sum()) for m, t in tracers.items()}

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
        real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
        real10 = ref_full[:, tracers[MET.MARGIN_INHERITED]].cpu().numpy()
        a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)

        # =================================================================== PART A: taxonomy ====
        log("\n[A] canonical failure modes, built out of the reference so the answer is known")
        p0 = real20 - real20.mean(0, keepdims=True)
        cen = real20.mean(0, keepdims=True)
        base = x0[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None]
        rs = np.random.RandomState(4)
        Gf, M = real20.shape[0], real20.shape[1]

        def roll_each(p, ks):
            return np.stack([np.roll(p[:, j], int(ks[j]), 0) for j in range(p.shape[1])], 1)

        def rot(p, ang):
            Rm = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            return cen + (p - cen) @ Rm.T

        ax = MET.major_axis_angle(real20)
        proj = (real20 - cen)[..., 0] * np.cos(ax)[None] + (real20 - cen)[..., 1] * np.sin(ax)[None]
        flat = cen + np.stack([proj * np.cos(ax)[None], proj * np.sin(ax)[None]], -1)

        MODES = [("identity", real20),
                 ("blow_up_x1.5", base + 1.5 * (real20 - base)),
                 ("blow_up_x3", base + 3.0 * (real20 - base)),
                 ("collapse_x0.5", base + 0.5 * (real20 - base)),
                 ("collapse_x0.2", base + 0.2 * (real20 - base)),
                 ("do_nothing", np.repeat(base, Gf, axis=0)),
                 ("phase_drift_8", np.roll(real20, 8, 0)),
                 ("phase_drift_25", np.roll(real20, 25, 0)),
                 ("decoherence", roll_each(real20, rs.randint(0, Gf, M))),
                 ("shape_rot_pi/5", rot(real20, math.pi / 5)),
                 ("shape_flatten", flat),
                 ("replay_previous_beat", recA[:, tracers[MET.MARGIN_SAFE]].cpu().numpy())]
        R["A_modes"] = {}
        log(f"    {'mode':<22s} {'coord':>7s} {'path':>9s} {'peak':>9s} {'orient':>7s} "
            f"{'loop':>8s} | {'peakR':>6s} {'pathR':>6s} {'t2':>6s} {'open':>6s} {'chir':>5s} "
            f"{'lagG':>6s} {'lagIQR':>7s} | tags")
        for nm, p in MODES:
            f = fingerprint(p, real20)
            f["tags"] = classify(f)
            R["A_modes"][nm] = f
            cd = f["coordination"]
            log(f"    {nm:<22s} {(f'{cd:.4f}' if isinstance(cd,float) else 'Undef'):>7s} "
                f"{CT.fmt(f['path_length'],9)} {CT.fmt(f['peak_excursion'],9)} "
                f"{CT.fmt(f['orientation_error'],7)} {CT.fmt(f['loopscore'],8)} | "
                f"{f['peak_ratio']:>6.3f} {f['path_ratio']:>6.3f} {f['t2_path_over_peak']:>6.3f} "
                f"{f['openness']:>6.3f} {f['chirality_match']:>5.2f} "
                f"{f['lag_global_halfperiod_frames']:>6.2f} {f['lag_iqr_frames']:>7.1f} | "
                f"{','.join(f['tags'])}")

        # =================================================================== candidates ==========
        def const(E, g):
            return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                              torch.full((C,), float(g), device=dev, dtype=f64)])

        sy.restore()
        nsf = sy.n_sub_per_frame
        x_fprev, x_fnext = recA[-2].clone(), ref_full[0].clone()
        sy.step(sy.E_true, sy.gain_true, n_sub=nsf)
        a_fdf = fd_accel(x_fprev, x0, x_fnext, sy.dt)
        Af, a0f, _ = sy.assemble(n_sub=nsf)
        Sf = Solver(Af, C)
        theta_frm = Sf(a_fdf - a0f)["ridge0"]
        Sf.free(); del Af, Sf
        torch.cuda.empty_cache()

        pm = torch.randperm(C, device=dev)
        gg = torch.Generator().manual_seed(303)
        Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
        gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)

        cands = [("theta_true", th, 0.0),
                 ("theta_hat_frame_ridge0", theta_frm, 0.0),
                 ("theta_shuffled_true", torch.cat([th[:C][pm], th[C:][pm]]), 0.0),
                 ("bank_blind_E130_g0.95", const(130.0, 0.95), 0.0),
                 ("bank_prior_draw_303", torch.cat([Ed, gd]), 0.0),
                 ("true_gain_x0.3", torch.cat([th[:C], th[C:] * 0.3]), 0.0),
                 ("theta_true_x0jitter_0.1dx", th, 0.1 * dx)]
        npz1 = os.path.join(HERE, "theta_refute1.npz")
        if os.path.exists(npz1):
            z = np.load(npz1)
            cands.append(("frame_DISP", torch.as_tensor(z["cand::frame_DISP"], device=dev,
                                                        dtype=f64), 0.0))
        WANT = ("d0::naive", "d0::eiv_full", "d0::eiv_trunc1", "d0::eiv_snr0", "d0::eiv_snr0.3",
                "clean_F_lerp")
        npz4 = os.path.join(HERE, a.eiv_npz)
        if os.path.exists(npz4):
            z = np.load(npz4)
            for k in z.files:
                nm = k.split("::", 1)[1]
                if nm in WANT:
                    cands.append((f"T1/{nm.replace('d0::','')}",
                                  torch.as_tensor(z[k], device=dev, dtype=f64), 0.0))
        for T in (8,):
            pz = os.path.join(HERE, f"theta_round4_stack_T{T}.npz")
            if os.path.exists(pz):
                z = np.load(pz)
                for k in ("naive", "eiv_snr0", "eiv_snr0.3", "eiv_snr1"):
                    if k in z.files:
                        cands.append((f"T{T}/{k}",
                                      torch.as_tensor(z[k], device=dev, dtype=f64), 0.0))
        if a.only:
            want = set(a.only.split(","))
            cands = [c for c in cands if c[0] in want]
        R["candidate_theta_error"] = {n: score(t, th, C) for n, t, _ in cands}

        def scored(theta, jit, anch, full_out=True):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                          anchor=anch, interior=interior, ss_tot=ss_tot,
                                          jitter=jit, keep_full=full_out, band_mask=anchor)
            s20 = tr[MET.MARGIN_SAFE].cpu().numpy()
            s10 = tr[MET.MARGIN_INHERITED].cpu().numpy()
            out = {"margin20": CT.read_metrics(s20, real20),
                   "margin10": CT.read_metrics(s10, real10), "coarse": coarse}
            out["t1"] = coarse["motion_energy_ratio_interior"]
            out["t2"] = t2_of(out["margin20"])
            out["fingerprint20"] = fingerprint(s20, real20)
            out["growth"] = growth(coarse["rms_pos_err_dx_per_frame"])
            if full_out:
                ah, _ = percell_amplitude(full, x0, cid, C, interior)
                out["percell"] = r2_percell(ah, a_ref, keep)
                out["a_percell"] = ah.tolist()
                del full
            return out

        # =================================================================== PART B: anchor ======
        log(f"\n[B] the anchoring question: every candidate FREE and ANCHORED, both margins "
            f"(band pins {100*R['anchor_fraction']:.1f}% of particles; probes inside it: "
            f"m20 {R['probes_in_band'][str(MET.MARGIN_SAFE)]}/100, "
            f"m10 {R['probes_in_band'][str(MET.MARGIN_INHERITED)]}/100)")
        R["B_anchor"] = {}
        log(f"    {'candidate':<28s} {'medE':>7s} | {'loop20 free':>11s} {'anch':>8s} "
            f"{'delta':>8s} | {'loop10 free':>11s} {'anch':>8s} {'delta':>8s} | "
            f"{'R2 d':>8s} {'bandrms':>8s}")
        for name, theta, jit in cands:
            row = {}
            for mode in ("free", "anchored"):
                row[mode] = scored(theta, jit, anchor if mode == "anchored" else None,
                                   full_out=(mode == "free"))
            d20 = {k: (row["anchored"]["margin20"][k] - row["free"]["margin20"][k])
                   for k in list(CT.INSTRUMENTS) + [CT.OBJECTIVE]
                   if isinstance(row["anchored"]["margin20"][k], float)
                   and isinstance(row["free"]["margin20"][k], float)}
            d10 = {k: (row["anchored"]["margin10"][k] - row["free"]["margin10"][k])
                   for k in list(CT.INSTRUMENTS) + [CT.OBJECTIVE]
                   if isinstance(row["anchored"]["margin10"][k], float)
                   and isinstance(row["free"]["margin10"][k], float)}
            row["delta_margin20"], row["delta_margin10"] = d20, d10
            R["B_anchor"][name] = row
            l2f, l2a = row["free"]["margin20"]["loopscore"], row["anchored"]["margin20"]["loopscore"]
            l1f, l1a = row["free"]["margin10"]["loopscore"], row["anchored"]["margin10"]["loopscore"]
            log(f"    {name:<28s} {R['candidate_theta_error'][name]['med_E']:>7.4f} | "
                f"{CT.fmt(l2f,11)} {CT.fmt(l2a,8)} {CT.fmt(l2a-l2f if isinstance(l2f,float) else None,8)} | "
                f"{CT.fmt(l1f,11)} {CT.fmt(l1a,8)} {CT.fmt(l1a-l1f if isinstance(l1f,float) else None,8)} | "
                f"{CT.fmt(row['anchored']['coarse']['R2_displacement_interior'] - row['free']['coarse']['R2_displacement_interior'],8)} "
                f"{row['free']['coarse']['rms_pos_err_dx_BAND_mean']:>8.4f}")

        # ranking agreement, the thing that actually decides whether it matters
        nms = [n for n in R["B_anchor"]
               if isinstance(R["B_anchor"][n]["free"]["margin20"]["loopscore"], float)]
        vf = np.array([R["B_anchor"][n]["free"]["margin20"]["loopscore"] for n in nms])
        va = np.array([R["B_anchor"][n]["anchored"]["margin20"]["loopscore"] for n in nms])
        from scipy.stats import spearmanr
        R["B_summary"] = {
            "n_candidates": len(nms),
            "max_abs_delta_loopscore_m20": float(np.abs(va - vf).max()),
            "max_abs_delta_loopscore_m20_who": nms[int(np.abs(va - vf).argmax())],
            "median_abs_delta_loopscore_m20": float(np.median(np.abs(va - vf))),
            "spearman_free_vs_anchored_m20": float(spearmanr(vf, va).statistic),
            "kendall_inversions": int(sum(1 for i in range(len(nms)) for j in range(i + 1, len(nms))
                                          if (vf[i] - vf[j]) * (va[i] - va[j]) < 0)),
            "anchor_cost_at_theta_true_rms_dx":
                R["B_anchor"]["theta_true"]["anchored"]["coarse"]["rms_pos_err_dx_mean"],
            "free_cost_at_theta_true_rms_dx":
                R["B_anchor"]["theta_true"]["free"]["coarse"]["rms_pos_err_dx_mean"]}
        for k, v in R["B_summary"].items():
            log(f"    {k:<44s} {v}")

        # =================================================================== PART A': diagnose ===
        log(f"\n[A'] the same rollouts, DIAGNOSED: which failure mode, and does the error grow?")
        R["A_diagnosis"] = {}
        log(f"    {'candidate':<28s} {'peakR':>6s} {'t1':>6s} {'lagG':>6s} {'lagIQR':>7s} "
            f"{'coord':>7s} {'orient':>7s} {'t2':>6s} {'open':>6s} {'rmsend':>7s} {'slope/f':>9s} "
            f"{'fin/max':>7s} | tags")
        for name in R["B_anchor"]:
            fr = R["B_anchor"][name]["free"]
            f = fr["fingerprint20"]
            tags = classify(f)
            gw = fr["growth"]
            R["A_diagnosis"][name] = {"fingerprint": f, "tags": tags, "growth": gw,
                                      "t1": fr["t1"], "t2": fr["t2"]}
            cd = f["coordination"]
            log(f"    {name:<28s} {f['peak_ratio']:>6.3f} {fr['t1']:>6.3f} "
                f"{f['lag_global_halfperiod_frames']:>6.2f} {f['lag_iqr_frames']:>7.1f} "
                f"{(f'{cd:.4f}' if isinstance(cd,float) else 'Undef'):>7s} "
                f"{CT.fmt(f['orientation_error'],7)} {f['t2_path_over_peak']:>6.3f} "
                f"{f['openness']:>6.3f} "
                f"{gw['rms_dx_final']:>7.4f} {gw['slope_rel_per_frame']:>9.2e} "
                f"{gw['final_over_max']:>7.3f} | {','.join(tags)}")

        # =================================================================== PART C: the gauge ===
        # Round 3's score, unchanged: the 2-D gauge quotients out the two global scales (k_E, k_g).
        # An errors-in-variables ATTENUATION is exactly a global scale error, so the estimator can
        # fail the parameter criterion and still pass the crash test -- or not.  Measured, not
        # assumed.  The bar is round 3's converged null-band top, bank_prior_draw_303 at 0.6488,
        # which is re-run here as a control.
        log(f"\n[C] the same candidates under round 3's 2-D gauge (k_E, k_g -> t1 = t2 = 1)")
        R["C_gauged"] = {}
        log(f"    {'candidate':<28s} {'medE':>7s} {'raw':>8s} {'t1':>6s} {'t2':>6s} "
            f"{'kE':>6s} {'kg':>6s} {'gauged':>8s} {'R2':>8s} {'r2cell':>7s} {'nx':>3s} status")
        for name, theta, jit in cands:
            raw = R["B_anchor"][name]["free"]

            def probe(lE, lg, theta=theta, jit=jit):
                d = scored(scale2(theta, math.exp(lE), math.exp(lg), C), jit, None,
                           full_out=False)
                return (d["t1"], d["t2"])

            gf = gauge_fix2(probe, (raw["t1"], raw["t2"]), tol=0.01, iters=20)
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (kE == 1.0 and kg == 1.0) else scored(scale2(theta, kE, kg, C), jit, None)
            gf["theta_error_after_k"] = score(scale2(theta, kE, kg, C), th, C)
            R["C_gauged"][name] = {"gauge": {k: v for k, v in gf.items() if k != "history"},
                                   "history": gf.get("history"),
                                   "gauged": {kk: gau[kk] for kk in
                                              ("margin20", "margin10", "coarse", "t1", "t2",
                                               "percell", "a_percell", "fingerprint20", "growth")
                                              if kk in gau}}
            pc = gau.get("percell", {}).get("r2")
            log(f"    {name:<28s} {R['candidate_theta_error'][name]['med_E']:>7.4f} "
                f"{CT.fmt(raw['margin20']['loopscore'],8)} {raw['t1']:>6.3f} {raw['t2']:>6.3f} "
                f"{kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['margin20']['loopscore'],8)} "
                f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)} "
                f"{(f'{pc:.4f}' if pc is not None else 'n/a'):>7s} {gf['n_extra']:>3d} "
                f"{gf['status']}")

        R["cite_status"] = {}
        dummy, _ = MET.population(G=24, M=20)
        for n in CT.INSTRUMENTS + (CT.OBJECTIVE,):
            try:
                MET.REGISTRY[n].cite(dummy, dummy)
                R["cite_status"][n] = "permitted"
            except Exception as e:
                R["cite_status"][n] = f"{type(e).__name__}: {str(e)[:70]}"

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

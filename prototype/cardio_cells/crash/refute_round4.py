"""refute_round4.py -- attack round 4's headline: "the estimator with the BEST PARAMETERS produces
the WORST trajectory" (T8/eiv_snr0 med|dE/E| 0.0546, gauged loopscore 0.0037, against T8/naive
med|dE/E| 0.4674, gauged loopscore 0.8044).

THREE ATTACKS
====================================================================================================
A. "best parameters" is a claim about med|dE/E| ONLY.  Round 4's own stack artefacts already carry
   rel_l2 = ||dtheta||/||theta|| and never quote it.  Here every candidate is scored on FIVE
   parameter statistics, including the one matched to what the gauged score can see: the relative
   error AFTER the optimal per-block rescale (the 2-parameter gauge orbit).  Cheap, no GPU.

B. THE NULL ROUND 4 DID NOT RUN: a theta whose med|dE/E| is 0.000 or 0.055 BY CONSTRUCTION and
   which contains a random 45 % of cells drawn from the prior.  If it scores badly, then
   "med|dE/E| <= 0.10" is not a statement about parameter quality at all and round 4's acceptance
   criterion is empty.

C. IS 0.0037 A SCORE OR A GAUGE FAILURE?  T8/eiv_snr0's gauge status is "max iters"; gauge_fix2
   returns the iterate with the smallest GAUGE residual, not the best score, so a non-converged
   gauge can hand back a point far worse than raw.  Here the gauge is replaced by a DETERMINISTIC
   5x5 log-grid + the same targets, so the gauged score stops depending on an optimiser.

D. THE REPAIR the round did not try: the tail, not the bias.  eiv_snr0 has 7/100 negative moduli
   and max |dE/E| = 16.1.  Clipping to a data-driven box (no knowledge of theta_true) is tested on
   BOTH estimators.

usage: PYTHONPATH=/workspace/Plexus/src python refute_round4.py --device cuda:1
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
from recover import install_E, score                             # noqa: E402
import metrics as MET                                            # noqa: E402
import crash_test as CT                                          # noqa: E402
from crash_round2 import percell_amplitude, r2_percell           # noqa: E402
from crash_round3 import gauge_fix2, scale2, t2_of               # noqa: E402


def param_stats(t, th, C):
    """Five parameter statistics, the last two matched to the 2-parameter gauge the score uses."""
    t = np.asarray(t, float)
    th = np.asarray(th, float)
    E, g = th[:C], th[C:]
    Eh, gh = t[:C], t[C:]
    kE = float(Eh @ E / (Eh @ Eh)) if float(Eh @ Eh) > 0 else float("nan")
    kg = float(gh @ g / (gh @ gh)) if float(gh @ gh) > 0 else float("nan")
    tg = np.concatenate([kE * Eh, kg * gh])
    return {"med_E": float(np.median(np.abs(Eh - E) / E)),
            "p90_E": float(np.percentile(np.abs(Eh - E) / E, 90)),
            "max_E": float(np.max(np.abs(Eh - E) / E)),
            "rel_l2": float(np.linalg.norm(t - th) / np.linalg.norm(th)),
            "corr_E": float(np.corrcoef(Eh, E)[0, 1]),
            "k_E_opt": kE, "k_g_opt": kg,
            "rel_l2_gauge_opt": float(np.linalg.norm(tg - th) / np.linalg.norm(th)),
            "med_E_after_rescale": float(np.median(np.abs(kE * Eh - E) / E)),
            "n_negE": int((Eh < 0).sum()), "mean_ratio_E": float(Eh.mean() / E.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="refute4")
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--skip-grid", action="store_true")
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
        dev, f64, dx = th.device, torch.float64, sy.g.dx
        x0, cid = sy.x0.clone(), sy.cid

        # ---- cross-script control: is theta_true the same object round 4 fitted against? --------
        z4 = np.load(os.path.join(HERE, "theta_round4_eiv.npz"))
        th_r4 = z4["cand::theta_true"]
        R["control_theta_true_matches_round4"] = float(np.abs(th.cpu().numpy() - th_r4).max())
        log(f"[control] theta_true here vs theta_round4_eiv.npz: max abs diff "
            f"{R['control_theta_true_matches_round4']:.3e}")

        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ~anchor
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
        a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)

        def scored(theta, full_out=True):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                          anchor=None, interior=interior, ss_tot=ss_tot,
                                          keep_full=full_out, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            out = {"margin20": m20, "coarse": coarse,
                   "t1": coarse["motion_energy_ratio_interior"], "t2": t2_of(m20)}
            if full_out:
                ah, _ = percell_amplitude(full, x0, cid, C, interior)
                out["percell"] = r2_percell(ah, a_ref, keep)
                del full
            return out

        # ---- candidates -------------------------------------------------------------------------
        z8 = np.load(os.path.join(HERE, "theta_round4_stack_T8.npz"))
        T8n = torch.as_tensor(z8["naive"], device=dev, dtype=f64)
        T8e = torch.as_tensor(z8["eiv_snr0"], device=dev, dtype=f64)

        def box(t, lo_f=0.2, hi_f=5.0):
            """Data-driven box: each block clipped to [lo_f, hi_f] x the block's own MEDIAN.

            No knowledge of theta_true -- the median of the estimate itself, plus positivity.
            """
            o = t.clone()
            for sl in (slice(0, C), slice(C, 2 * C)):
                v = o[sl]
                m = v[v > 0].median() if int((v > 0).sum()) > 0 else v.abs().median()
                o[sl] = v.clamp(min=lo_f * m, max=hi_f * m)
            return o

        # the null round 4 did not run: a MEDIAN-MATCHED theta that is 45 % random
        gnull = torch.Generator().manual_seed(4242)
        idx = torch.randperm(C, generator=gnull)[:45].to(dev)
        Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gnull)).to(dev, f64)
        gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gnull)).to(dev, f64)
        n_med0 = th.clone()
        n_med0[idx] = Ed[idx]
        n_med0[C + idx] = gd[idx]
        # same, then a 5.46 % multiplicative error on the EXACT cells so the MEDIAN matches
        # T8/eiv_snr0's 0.0546 exactly instead of being 0
        sgn = torch.where(torch.rand(2 * C, generator=gnull).to(dev) > 0.5, 1.0, -1.0)
        n_medm = n_med0.clone()
        mask = torch.ones(C, dtype=torch.bool, device=dev)
        mask[idx] = False
        n_medm[:C][mask] = th[:C][mask] * (1 + 0.0546 * sgn[:C][mask])
        n_medm[C:][mask] = th[C:][mask] * (1 + 0.0546 * sgn[C:][mask])

        cands = [("theta_true", th),
                 ("T8/naive", T8n),
                 ("T8/eiv_snr0", T8e),
                 ("T8/naive_box", box(T8n)),
                 ("T8/eiv_snr0_box", box(T8e)),
                 ("null_med0_rand45", n_med0),
                 ("null_medmatch_0.055", n_medm)]

        R["candidates"] = {}
        log(f"\n[A/B/D] raw and 2-D-gauged rollouts, margin 20, anchor=None, {G} frames")
        log(f"    {'candidate':<22s} {'medE':>7s} {'relL2':>8s} {'relL2_g':>8s} {'medE_re':>8s} "
            f"{'corr':>6s} {'neg':>4s} | {'raw':>8s} {'t1':>6s} {'t2':>6s} {'kE':>6s} {'kg':>6s} "
            f"{'gauged':>8s} {'R2':>8s} {'r2cell':>7s} status")
        for name, theta in cands:
            ps = param_stats(theta.cpu().numpy(), th.cpu().numpy(), C)
            raw = scored(theta)

            def probe(lE, lg, theta=theta):
                d = scored(scale2(theta, math.exp(lE), math.exp(lg), C), full_out=False)
                return (d["t1"], d["t2"])

            gf = gauge_fix2(probe, (raw["t1"], raw["t2"]), tol=0.01, iters=20)
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (kE == 1.0 and kg == 1.0) else scored(scale2(theta, kE, kg, C))
            R["candidates"][name] = {
                "param": ps,
                "raw": {"loop": raw["margin20"]["loopscore"], "t1": raw["t1"], "t2": raw["t2"],
                        "R2": raw["coarse"]["R2_displacement_interior"],
                        "r2cell": raw["percell"]["r2"],
                        "instruments": {k: raw["margin20"][k] for k in CT.INSTRUMENTS}},
                "gauged": {"loop": gau["margin20"]["loopscore"], "t1": gau["t1"], "t2": gau["t2"],
                           "R2": gau["coarse"]["R2_displacement_interior"],
                           "r2cell": gau["percell"]["r2"],
                           "instruments": {k: gau["margin20"][k] for k in CT.INSTRUMENTS}},
                "gauge": {k: v for k, v in gf.items() if k != "history"},
                "gauge_history": gf.get("history")}
            log(f"    {name:<22s} {ps['med_E']:>7.4f} {ps['rel_l2']:>8.3f} "
                f"{ps['rel_l2_gauge_opt']:>8.3f} {ps['med_E_after_rescale']:>8.4f} "
                f"{ps['corr_E']:>6.3f} {ps['n_negE']:>4d} | "
                f"{CT.fmt(raw['margin20']['loopscore'],8)} {raw['t1']:>6.3f} {raw['t2']:>6.3f} "
                f"{kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['margin20']['loopscore'],8)} "
                f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)} "
                f"{gau['percell']['r2']:>7.4f} {gf['status']}")

        # ---- C: the gauge as a DETERMINISTIC grid ------------------------------------------------
        if not a.skip_grid:
            R["C_grid"] = {}
            n = a.grid
            ls = np.linspace(-0.8, 0.8, n)
            log(f"\n[C] deterministic {n}x{n} log-grid gauge (k in [{math.exp(-0.8):.2f},"
                f"{math.exp(0.8):.2f}]): is the non-converged 'gauged' number a score or an "
                f"optimiser artefact?")
            for name in ("T8/eiv_snr0", "T8/naive"):
                theta = dict(cands)[name]
                cells = []
                for le in ls:
                    for lg in ls:
                        d = scored(scale2(theta, math.exp(le), math.exp(lg), C), full_out=False)
                        cells.append({"kE": float(math.exp(le)), "kg": float(math.exp(lg)),
                                      "t1": d["t1"], "t2": d["t2"],
                                      "loop": d["margin20"]["loopscore"],
                                      "resid": float(max(abs(math.log(max(d["t1"], 1e-12))),
                                                         abs(math.log(max(d["t2"], 1e-12)))))})
                best = min(cells, key=lambda c: c["resid"])
                loops = [c["loop"] for c in cells if isinstance(c["loop"], float)]
                near = [c for c in cells if c["resid"] < 0.15]
                R["C_grid"][name] = {
                    "cells": cells, "best_gauge_cell": best,
                    "loop_at_best_gauge_cell": best["loop"],
                    "loop_range_over_grid": [float(np.min(loops)), float(np.max(loops))],
                    "n_cells_resid_lt_0.15": len(near),
                    "loop_range_among_near": ([float(np.min([c["loop"] for c in near])),
                                               float(np.max([c["loop"] for c in near]))]
                                              if near else None),
                    "broyden_loop": R["candidates"][name]["gauged"]["loop"],
                    "broyden_k": [R["candidates"][name]["gauge"]["k_E"],
                                  R["candidates"][name]["gauge"]["k_g"]]}
                q = R["C_grid"][name]
                log(f"    {name:<16s} grid-best gauge cell kE {best['kE']:.3f} kg {best['kg']:.3f} "
                    f"resid {best['resid']:.3f} -> loop {CT.fmt(best['loop'],8)} | Broyden "
                    f"{CT.fmt(q['broyden_loop'],8)} at kE {q['broyden_k'][0]:.3f} "
                    f"kg {q['broyden_k'][1]:.3f} | loops over the whole grid "
                    f"[{q['loop_range_over_grid'][0]:.3f}, {q['loop_range_over_grid'][1]:.3f}]; "
                    f"{q['n_cells_resid_lt_0.15']} cells within 15% of the targets"
                    + (f", their loops span [{q['loop_range_among_near'][0]:.3f}, "
                       f"{q['loop_range_among_near'][1]:.3f}]" if q["loop_range_among_near"]
                       else ""))

        # ---- the rank correlations, on every theta on disk plus the new ones --------------------
        from scipy.stats import spearmanr
        rep = json.load(open(os.path.join(HERE, "round4_report.json")))["crash"]
        disk = {}
        for k in z4.files:
            nm = k.split("::", 1)[1]
            if nm.startswith("d0::"):
                disk["T1/" + nm[4:]] = z4[k]
        disk["T1/clean_F_lerp"] = z4["cand::clean_F_lerp"]
        z1 = np.load(os.path.join(HERE, "theta_refute1.npz"))
        disk["frame_DISP"] = z1["cand::frame_DISP"]
        disk["T8/naive"] = z8["naive"]
        disk["T8/eiv_snr0"] = z8["eiv_snr0"]
        disk["theta_true"] = th_r4
        rows = []
        for nm, t in disk.items():
            if nm not in rep:
                continue
            ps = param_stats(t, th_r4, C)
            rows.append((nm, ps, rep[nm]["raw_loop"], rep[nm]["gau_loop"], rep[nm]["skill"]))
        keys = ["med_E", "rel_l2", "corr_E", "rel_l2_gauge_opt", "med_E_after_rescale"]
        R["A_rank"] = {"n": len(rows), "candidates": [r[0] for r in rows], "spearman": {}}
        log(f"\n[A] which parameter statistic predicts the rollout?  Spearman over the {len(rows)} "
            f"candidates whose theta is on disk (round 4's own set)")
        for k in keys:
            v = np.array([r[1][k] for r in rows], float)
            R["A_rank"]["spearman"][k] = {
                "vs_raw_loop": float(spearmanr(v, [r[2] for r in rows]).statistic),
                "vs_gauged_loop": float(spearmanr(v, [r[3] for r in rows]).statistic),
                "vs_skill": float(spearmanr(v, [r[4] for r in rows]).statistic)}
            q = R["A_rank"]["spearman"][k]
            log(f"    {k:<20s} raw {q['vs_raw_loop']:+.3f}  gauged {q['vs_gauged_loop']:+.3f}  "
                f"skill {q['vs_skill']:+.3f}")
        R["A_param_table"] = {r[0]: r[1] for r in rows}

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

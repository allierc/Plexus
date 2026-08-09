"""refute_round2.py -- attack round 2's headline: "amplitude is now a calibrated nuisance,
the confound is gone, and the per-cell axis has appeared".

Four attacks, each with a control:

 A. THE GAUGE ORBIT.  `true_gain_x1.8` is theta_true perturbed EXACTLY along the direction the
    gauge searches (a global multiple of the gain block). gauge_fix returned k = 0.558 = 1/1.8 to
    0.4%, i.e. it reconstructed theta_true. The triad's +1.176 recovery may therefore be a
    tautology. Control: the SAME kind of error placed off the gauge orbit -- a global multiple of
    the E block, whose per-cell pattern is equally exact -- and see whether the gauge repairs it.

 B. THE BLIND FAMILY.  Every per-cell-blind constant carries ZERO per-cell information. If the
    gauged loopscore / per-cell skill varies a lot across that family, then what "appeared" is a
    second GLOBAL scalar (mean stiffness), not a per-cell axis. Sweep E and read the range.

 C. A NULL ROUND 2 DID NOT RUN: independent draws from the planting prior. Per-cell variation with
    zero information about WHICH cell is which. An honest per-cell instrument must score these at
    or below the blind constant.

 D. THE NULLS WERE NOT GAUGED.  Round 2 compares GAUGE-FIXED candidates (0.55..0.89) against RAW
    nulls (do-nothing +0.260, replay +0.292). Under an amplitude gauge the commensurate null is an
    AMPLITUDE-MATCHED replay: x0 + s*(recA - x0) with s set so the interior motion energy matches.

usage: PYTHONPATH=/workspace/Plexus/src python refute_round2.py --device cuda:0
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
for p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, p)

from recover import install_E, score                            # noqa: E402
from assemble import SUBSTEP_TOKENS                             # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402
import crash_round2 as R2                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="refute2")
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--group", default="ABCD")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=165, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "tol": a.tol}
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
        real20 = ref_tr[MET.MARGIN_SAFE].cpu().numpy()

        a_ref, n_kept = R2.percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        ar = a_ref[keep] / a_ref[keep].mean()

        # ---- the skill denominator: round 2's OWN gauge-fixed blind field, so the numbers below
        #      are directly comparable with its table (and its value is re-derived here as a check)
        S2 = [json.load(open(os.path.join(HERE, f"crash_round2_s{i}.json"))) for i in (0, 1)]
        ro2 = {}
        for s in S2:
            for k_, v in s["rollouts"].items():
                ro2.setdefault(k_, v)
        keep2 = np.array(S2[0]["keep_percell"], dtype=bool)
        a_ref2 = np.array(S2[0]["a_ref_percell"], dtype=float)
        R["control_setup_matches_round2"] = {
            "a_ref_max_abs_diff": float(np.nanmax(np.abs(a_ref - a_ref2))),
            "keep_same": bool((keep == keep2).all())}
        bf = np.array(ro2["blind_E130_g0.95"]["gauged"]["a_percell"], dtype=float)[keep]
        bf = bf / bf.mean()
        sse_blind = float(((bf - ar) ** 2).sum())
        sst = float(((ar - ar.mean()) ** 2).sum())
        R["skill_denominator"] = {"sse_blind": sse_blind, "sst": sst,
                                  "r2_of_blind": 1 - sse_blind / sst}
        log(f"[setup] a_ref matches round 2 to {R['control_setup_matches_round2']['a_ref_max_abs_diff']:.2e}"
            f"; sse_blind {sse_blind:.4f} (round2 r2_blind {1-sse_blind/sst:.4f})")

        def skill_of(a_hat):
            f = np.array(a_hat, dtype=float)[keep]
            f = f / f.mean()
            return 1.0 - float(((f - ar) ** 2).sum()) / sse_blind

        def scored(theta, jit=0.0):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full, anchor=None,
                                          interior=interior, ss_tot=ss_tot, jitter=jit,
                                          keep_full=True, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            a_hat, _ = R2.percell_amplitude(full, x0, cid, C, interior)
            pc = R2.r2_percell(a_hat, a_ref, keep)
            del full
            return {"margin20": m20, "coarse": coarse, "percell": pc,
                    "skill": skill_of(a_hat), "a_percell": a_hat.tolist()}

        def run(name, theta, jit=0.0):
            t_c = time.time()
            raw = scored(theta, jit)
            r_raw = raw["coarse"]["motion_energy_ratio_interior"]
            k, r_k, hist, nx, st = R2.gauge_fix(sy, theta, jit, ctx, r_raw, tol=a.tol,
                                                iters=a.iters)
            gau = raw if k == 1.0 else scored(R2.scale_gain(theta, k, C), jit)
            rec = {"theta_error": score(theta, th, C), "raw": raw, "gauged": gau,
                   "gauge": {"k": k, "ratio_at_k": r_k, "history": hist, "n_extra": nx,
                             "status": st,
                             "theta_error_after_k": score(R2.scale_gain(theta, k, C), th, C)},
                   "seconds": time.time() - t_c}
            R["rollouts"][name] = rec
            log(f"  {name:<26s} medE {rec['theta_error']['med_E']:>6.3f} | raw loop "
                f"{CT.fmt(raw['margin20']['loopscore'],8)} Erat "
                f"{raw['coarse']['motion_energy_ratio_interior']:>6.3f} | k {k:>6.3f} gauged loop "
                f"{CT.fmt(gau['margin20']['loopscore'],8)} R2 "
                f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)} r2cell "
                f"{gau['percell']['r2']:>7.4f} skill {gau['skill']:>+7.3f}")
            return rec

        R["rollouts"] = {}

        def const(E, g):
            return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                              torch.full((C,), float(g), device=dev, dtype=f64)])

        # ---- control: reproduce two of round 2's rows exactly ---------------------------------
        log("\n[control] reproducing round 2 rows on a different GPU")
        run("theta_true", th)
        run("blind_E130_g0.95", const(130.0, 0.95))

        # ---- A. the gauge orbit ----------------------------------------------------------------
        if "A" in a.group:
            log("\n[A] is the triad's recovery a tautology of the gauge orbit?")
            run("true_gain_x1.8", torch.cat([th[:C], th[C:] * 1.8]))
            for s in (1.8, 0.5):
                run(f"true_E_x{s:g}", torch.cat([th[:C] * s, th[C:]]))
            run("true_both_x1.8", th * 1.8)

        # ---- B. the blind family ---------------------------------------------------------------
        if "B" in a.group:
            log("\n[B] the per-cell-BLIND family: zero per-cell information by construction")
            for E in (40, 60, 90, 130, 180, 240, 320):
                run(f"blind_E{E}_g1", const(E, 1.0))

        # ---- C. prior draws ---------------------------------------------------------------------
        if "C" in a.group:
            log("\n[C] independent draws from the planting prior (per-cell variation, zero info)")
            for sd in (101, 202, 303):
                gg = torch.Generator().manual_seed(sd)
                E = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
                gn = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)
                run(f"prior_draw_{sd}", torch.cat([E, gn]))

        # ---- D. the nulls, gauged ----------------------------------------------------------------
        if "D" in a.group:
            log("\n[D] the nulls round 2 left ungauged")
            R["nulls"] = {}
            dA = (recA - x0[None])[:, interior]
            d0 = d_ref[:, interior]
            e_A, e_R = float(dA.pow(2).sum()), float(d0.pow(2).sum())
            s_rep = float(np.sqrt(e_R / e_A))
            for tag, s in (("replay_raw", 1.0), ("replay_energy_matched", s_rep)):
                rep = x0[None] + s * (recA - x0[None])
                m = CT.read_metrics(rep[:, tracers[MET.MARGIN_SAFE]].cpu().numpy(), real20)
                dR = (rep - x0[None])[:, interior]
                a_rep, _ = R2.percell_amplitude(rep, x0, cid, C, interior)
                R["nulls"][tag] = {
                    "scale": s, "margin20": m,
                    "coarse": {"R2_displacement_interior": float(1 - (dR - d0).pow(2).sum()/ss_tot),
                               "motion_energy_ratio_interior": float(dR.pow(2).sum()/d0.pow(2).sum())},
                    "percell": R2.r2_percell(a_rep, a_ref, keep), "skill": skill_of(a_rep)}
                v = R["nulls"][tag]
                log(f"  {tag:<26s} scale {s:.4f} | loop {CT.fmt(m['loopscore'],8)} Erat "
                    f"{v['coarse']['motion_energy_ratio_interior']:>6.3f} R2 "
                    f"{v['coarse']['R2_displacement_interior']:>8.3f} r2cell "
                    f"{v['percell']['r2']:>7.4f} skill {v['skill']:>+7.3f}")
            # a scale that maximises the replay's loopscore -- the strongest form of the bar
            best = None
            for s in np.linspace(0.4, 2.0, 17):
                rep = x0[None] + float(s) * (recA - x0[None])
                ls = MET.REGISTRY["loopscore"](rep[:, tracers[MET.MARGIN_SAFE]].cpu().numpy(),
                                               real20)
                if best is None or ls > best[1]:
                    best = (float(s), float(ls))
            R["nulls"]["replay_best_scale"] = {"scale": best[0], "loopscore": best[1]}
            log(f"  {'replay_best_scale':<26s} scale {best[0]:.3f} | loop {best[1]:8.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

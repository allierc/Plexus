"""crash_round3.py -- ROUND 3. ONE CHANGE, again in the score: quotient BOTH global scalars,
and measure the zero-information floor as a BAND rather than assuming it.

WHY (round 2's diagnosis, in one paragraph)
====================================================================================================
Round 2 made amplitude a calibrated nuisance with a 1-D gauge on the GAIN block. The refutation
showed that this quotients only ONE of the model's two global scalars: the same error placed off
the gauge's own direction (`true_E_x1.8`, per-cell pattern EXACTLY right) scored 0.7291 / skill
+0.215 while `true_gain_x1.8` -- the same information, on the orbit -- scored 0.9995 / +1.000. And
the zero-information floor was never measured: a 12-member blind bank spans gauged loopscore
0.5079..0.7005 and skill -2.324..+0.356, with skill fit to R^2 0.9985 by a quadratic in log(mean E)
alone, i.e. it was reading a SECOND global scalar (mean stiffness).

THE ONE CHANGE, scoring stage only:

 (a) `gauge_fix2` replaces `gauge_fix`. A 2-D Newton/Broyden solve on (log k_E, log k_g), k_E
     multiplying theta[:C] and k_g multiplying theta[C:], driving TWO observable global scalars of
     the recording to 1:
         t1 = coarse["motion_energy_ratio_interior"]                          (round 2's)
         t2 = (PathLength.reading(sim)/PeakExcursion.reading(sim))
              / (PathLength.reading(real)/PeakExcursion.reading(real))        at margin 20
     Both are readable from a recording; both are already computed by `crash_test.read_metrics`.
     Every candidate is scored RAW and 2-D GAUGED.

 (b) `NULL_BANK` -- 13 vectors carrying ZERO per-cell information (7 blind constants, one blind
     constant with g=0.95, 4 independent draws from the planting prior, and one geometry-only
     predictor that knows a cell's radius but nothing about its parameters), scored under the SAME
     2-D gauge. The floor is reported as the BAND [min, max] over that bank, on both instruments,
     and the per-cell skill denominator becomes the bank's argmax rather than an arbitrary member.

NOTHING ELSE MOVES: same system, seeds, t0 = 165, 150-frame window, reading surface, `CT.rollout`,
`metrics.py`. Estimators are re-solved by the same unmodified code; `frame_DISP` is read from
round 1's archived npz.

usage: PYTHONPATH=/workspace/Plexus/src python crash_round3.py --device cuda:1 --shard 0 --nshards 2
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

from assemble import SUBSTEP_TOKENS, rel                        # noqa: E402
from recover import Solver, fd_accel, install_E, score          # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402
from crash_round2 import percell_amplitude, r2_percell          # noqa: E402


# =============================================================================================
#  THE ONE CHANGE (a): a 2-D gauge
# =============================================================================================
def scale2(theta, kE, kg, C):
    return torch.cat([theta[:C] * kE, theta[C:] * kg])


def t2_of(m20):
    """PathLength/PeakExcursion of the model over the same ratio of the recording. 1 is matched.

    Both readings are `PairedProperty.reading`, already computed in crash_test.read_metrics; the
    ratio of the two is a shape-of-the-loop scalar that is nearly blind to the gain scale and
    strongly monotone in the E scale, which is what makes the 2x2 Jacobian invertible.
    """
    ps, pr = m20["path_length_reading_sim"], m20["path_length_reading_real"]
    xs, xr = m20["peak_excursion_reading_sim"], m20["peak_excursion_reading_real"]
    if min(xs, xr, pr) <= 0:
        return float("nan")
    return (ps / xs) / (pr / xr)


def gauge_fix2(probe, r_raw, tol=0.01, iters=8, h=0.15, lim=2.5):
    """2-D Newton (FD Jacobian, then Broyden) on (log k_E, log k_g) driving (t1, t2) to (1, 1).

    probe(lE, lg) -> (t1, t2), one rollout.
    r_raw         : (t1, t2) at (0, 0), already paid for by the raw scored rollout.
    Returns dict with k_E, k_g, the residuals, the history and the number of EXTRA rollouts.
    """
    hist = [(1.0, 1.0) + tuple(r_raw)]
    n = 0

    def res(t):
        if not np.all(np.isfinite(t)) or min(t) <= 0:
            return None
        return np.array([math.log(t[0]), math.log(t[1])])

    r0 = res(r_raw)
    if r0 is None:
        return {"k_E": 1.0, "k_g": 1.0, "t": list(r_raw), "status": "no motion to calibrate",
                "history": hist, "n_extra": 0}
    ltol = math.log(1.0 + tol)
    if np.abs(r0).max() <= ltol:
        return {"k_E": 1.0, "k_g": 1.0, "t": list(r_raw), "status": "already at (1,1)",
                "history": hist, "n_extra": 0}

    x = np.zeros(2)
    best = (x.copy(), tuple(r_raw), float(np.abs(r0).max()))
    J = np.zeros((2, 2))
    for j in range(2):
        xp = x.copy()
        xp[j] += h
        t = probe(*xp)
        n += 1
        hist.append((math.exp(xp[0]), math.exp(xp[1])) + tuple(t))
        rj = res(t)
        if rj is None:
            return {"k_E": float(np.exp(best[0][0])), "k_g": float(np.exp(best[0][1])),
                    "t": list(best[1]), "status": "diverged in FD", "history": hist, "n_extra": n}
        J[:, j] = (rj - r0) / h
        if np.abs(rj).max() < best[2]:
            best = (xp.copy(), tuple(t), float(np.abs(rj).max()))
    status = "max iters"
    r = r0
    for _ in range(iters):
        if abs(np.linalg.det(J)) < 1e-8:
            status = "singular jacobian"
            break
        step = -np.linalg.solve(J, r)
        nrm = np.abs(step).max()
        if nrm > 1.0:
            step = step / nrm                       # trust region in log space
        xn = np.clip(x + step, -lim, lim)
        t = probe(*xn)
        n += 1
        hist.append((math.exp(xn[0]), math.exp(xn[1])) + tuple(t))
        rn = res(t)
        if rn is None:
            status = "diverged"
            break
        if np.abs(rn).max() < best[2]:
            best = (xn.copy(), tuple(t), float(np.abs(rn).max()))
        if np.abs(rn).max() <= ltol:
            status = "converged"
            x, r = xn, rn
            break
        dx, dr = xn - x, rn - r
        if np.dot(dx, dx) > 1e-14:                  # Broyden rank-1 update
            J = J + np.outer(dr - J @ dx, dx) / np.dot(dx, dx)
        x, r = xn, rn
    return {"k_E": float(np.exp(best[0][0])), "k_g": float(np.exp(best[0][1])),
            "t": list(best[1]), "status": status, "history": hist, "n_extra": n,
            "resid_logmax": best[2]}


# =============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round3")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--only", default="")
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
         "ONE_CHANGE": "scoring only: (a) 2-D gauge on (log k_E, log k_g) driving BOTH "
                       "motion-energy ratio and path_length/peak_excursion ratio to 1; "
                       "(b) a 13-member zero-information NULL BANK scored the same way, the floor "
                       "reported as a band and used as the per-cell skill denominator."}
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
        real20 = ref_tr[MET.MARGIN_SAFE].cpu().numpy()

        a_ref, n_kept = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        R["a_ref_percell"] = a_ref.tolist()
        R["keep_percell"] = keep.tolist()
        log(f"[percell] {int(keep.sum())}/{C} cells kept; a_ref/dx "
            f"{np.nanmin(a_ref)/dx:.3f} .. {np.nanmedian(a_ref)/dx:.3f} .. {np.nanmax(a_ref)/dx:.3f}")

        dummy, _ = MET.population(G=24, M=20)
        R["cite_status"] = {}
        for n in CT.INSTRUMENTS + (CT.OBJECTIVE,):
            try:
                MET.REGISTRY[n].cite(dummy, dummy)
                R["cite_status"][n] = "permitted"
            except Exception as e:
                R["cite_status"][n] = f"{type(e).__name__}: {str(e)[:70]}"

        # ------------------------------------------------------------------ estimators ---------
        R["solves"] = {}
        sy.restore()
        a_solver = sy.step(sy.E_true, sy.gain_true, n_sub=1)
        x_next = sy.p.get("pos").clone()
        a_fd = fd_accel(sy.x_prev, x0, x_next, sy.dt_sub)
        A1, a01, t1s = sy.assemble(n_sub=1)
        S1 = Solver(A1, C)
        theta_sub = S1(a_fd - a01)["ridge0"]
        R["solves"]["substep"] = {"residual": rel(A1 @ sy.theta_true - (a_solver - a01),
                                                  a_solver - a01), "cond": S1.cond,
                                  "score": score(theta_sub, th, C)}
        S1.free(); del A1, S1
        torch.cuda.empty_cache()
        nsf = sy.n_sub_per_frame
        x_fprev, x_fnext = recA[-2].clone(), ref_full[0].clone()
        sy.restore()
        a_solver_f = sy.step(sy.E_true, sy.gain_true, n_sub=nsf)
        a_fdf = fd_accel(x_fprev, x0, x_fnext, sy.dt)
        Af, a0f, tfs = sy.assemble(n_sub=nsf)
        Sf = Solver(Af, C)
        solf = Sf(a_fdf - a0f)
        theta_frm, theta_frm_r = solf["ridge0"], solf["ridge0.01"]
        R["solves"]["frame"] = {"residual": rel(Af @ sy.theta_true - (a_solver_f - a0f),
                                                a_solver_f - a0f), "cond": Sf.cond,
                                "score": score(theta_frm, th, C)}
        Sf.free(); del Af, Sf
        torch.cuda.empty_cache()
        log(f"[solve] substep med|dE/E| {R['solves']['substep']['score']['med_E']:.3e}; "
            f"frame med|dE/E| {R['solves']['frame']['score']['med_E']:.4f}")

        # ------------------------------------------------------------------ candidates ---------
        def const(E, g):
            return torch.cat([torch.full((C,), float(E), device=dev, dtype=f64),
                              torch.full((C,), float(g), device=dev, dtype=f64)])

        gp = torch.Generator(device=str(dev)).manual_seed(77)
        u = torch.randn(2 * C, generator=gp, device=dev, dtype=f64)
        u = u / u.abs().max()
        pm = torch.randperm(C, device=dev)

        MAIN = [("theta_true", th, 0.0),
                ("theta_hat_substep", theta_sub, 0.0),
                ("theta_hat_frame_ridge0", theta_frm, 0.0),
                ("theta_hat_frame_ridge1e-2", theta_frm_r, 0.0),
                ("theta_shuffled_true", torch.cat([th[:C][pm], th[C:][pm]]), 0.0)]
        for e in (0.01, 0.03, 0.1, 0.3):
            MAIN.append((f"theta_true_perturbed_{e:g}", th * (1.0 + e * u), 0.0))
        for j in (0.01, 0.1):
            MAIN.append((f"theta_true_x0jitter_{j:g}dx", th, j * dx))
        npz = os.path.join(HERE, "theta_refute1.npz")
        if os.path.exists(npz) and not a.smoke:
            z2 = np.load(npz)
            t_disp = torch.as_tensor(z2["cand::frame_DISP"], device=dev, dtype=f64)
            kE = float((t_disp[:C] * th[:C]).sum() / (t_disp[:C] ** 2).sum())
            kg = float((t_disp[C:] * th[C:]).sum() / (t_disp[C:] ** 2).sum())
            MAIN.append(("frame_DISP", t_disp, 0.0))
            MAIN.append(("frame_DISP_oracle_rescale",
                         torch.cat([t_disp[:C] * kE, t_disp[C:] * kg]), 0.0))
            R["oracle_rescale_k_frame_DISP"] = [kE, kg]
        # the pair the diagnosis says must now converge: same information, different gauge direction
        MAIN += [("true_gain_x1.8", torch.cat([th[:C], th[C:] * 1.8]), 0.0),
                 ("true_E_x1.8", torch.cat([th[:C] * 1.8, th[C:]]), 0.0),
                 ("true_E_x0.5", torch.cat([th[:C] * 0.5, th[C:]]), 0.0),
                 ("true_both_x1.8", th * 1.8, 0.0)]

        # ------------------------------------------------------------------ THE NULL BANK ------
        BANK = [(f"bank_blind_E{E}_g1", const(E, 1.0), 0.0)
                for E in (40, 60, 90, 130, 180, 240, 320)]
        BANK.append(("bank_blind_E130_g0.95", const(130.0, 0.95), 0.0))
        for sd in (101, 202, 303, 404):
            gg = torch.Generator().manual_seed(sd)
            Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gg)).to(dev, f64)
            BANK.append((f"bank_prior_draw_{sd}", torch.cat([Ed, gd]), 0.0))
        # geometry-only: knows where a cell sits, nothing about its parameters
        cx = torch.zeros(C, device=dev, dtype=f64)
        cy = torch.zeros(C, device=dev, dtype=f64)
        for c in range(1, C + 1):
            m = cid == c
            cx[c - 1], cy[c - 1] = x0[m, 0].mean(), x0[m, 1].mean()
        rad = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2).sqrt()
        z = (rad - rad.mean()) / rad.std()
        BANK.append(("bank_geom_radius", torch.cat([130.0 * (1 + 0.5 * z),
                                                    torch.ones(C, device=dev, dtype=f64)]), 0.0))
        R["bank_names"] = [n for n, _, _ in BANK]

        cands = MAIN + BANK
        if a.only:
            want = set(a.only.split(","))
            cands = [c for c in cands if c[0] in want]
        R["candidate_theta_error"] = {n: score(t, th, C) for n, t, _ in cands}
        for n, t, _ in cands:
            R["candidate_theta_error"][n].update({"mean_E": float(t[:C].mean()),
                                                  "mean_gain": float(t[C:].mean())})
        if a.shard == 0 and not a.only:
            np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                     **{f"cand::{n}": t.cpu().numpy() for n, t, _ in cands})

        mine = [c for i, c in enumerate(cands)
                if a.nshards == 1 or i % a.nshards == a.shard or c[0] == "theta_true"]
        log(f"[shard {a.shard}/{a.nshards}] {len(mine)}/{len(cands)} candidates")

        # ------------------------------------------------------------------ scoring ------------
        def scored(theta, jit, full_out=True):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full, anchor=None,
                                          interior=interior, ss_tot=ss_tot, jitter=jit,
                                          keep_full=full_out, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            out = {"margin20": m20, "coarse": coarse,
                   "t1": coarse["motion_energy_ratio_interior"], "t2": t2_of(m20)}
            if full_out:
                a_hat, _ = percell_amplitude(full, x0, cid, C, interior)
                out["percell"] = r2_percell(a_hat, a_ref, keep)
                out["a_percell"] = a_hat.tolist()
                del full
            return out

        R["rollouts"] = {}
        log(f"\n[crash test] {G}-frame free rollouts from tick {W}, margin-{MET.MARGIN_SAFE}; "
            f"RAW and 2-D GAUGED (|t1-1|,|t2-1| <= {a.tol})")
        log(f"    {'candidate':<30s} {'medE':>7s} {'rawloop':>8s} {'t1':>6s} {'t2':>6s} "
            f"{'kE':>6s} {'kg':>6s} {'gauloop':>8s} {'gauR2':>8s} {'r2cell':>7s} {'nx':>3s}")
        for name, theta, jit in mine:
            t_c = time.time()
            raw = scored(theta, jit)

            def probe(lE, lg, theta=theta, jit=jit):
                d = scored(scale2(theta, math.exp(lE), math.exp(lg), C), jit, full_out=False)
                return (d["t1"], d["t2"])

            gf = gauge_fix2(probe, (raw["t1"], raw["t2"]), tol=a.tol, iters=a.iters)
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (kE == 1.0 and kg == 1.0) else scored(scale2(theta, kE, kg, C), jit)
            gf["theta_error_after_k"] = score(scale2(theta, kE, kg, C), th, C)
            rec = {"theta_error": R["candidate_theta_error"][name], "x0_jitter_world": jit,
                   "raw": raw, "gauged": gau, "gauge": gf, "seconds": time.time() - t_c}
            R["rollouts"][name] = rec
            pc = gau["percell"]["r2"]
            log(f"    {name:<30s} {rec['theta_error']['med_E']:>7.4f} "
                f"{CT.fmt(raw['margin20']['loopscore'],8)} {raw['t1']:>6.3f} {raw['t2']:>6.3f} "
                f"{kE:>6.3f} {kg:>6.3f} {CT.fmt(gau['margin20']['loopscore'],8)} "
                f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)} "
                f"{(f'{pc:.4f}' if pc is not None else 'n/a'):>7s} {gf['n_extra']:>3d}  "
                f"[{gf['status']}, {rec['seconds']:.0f}s]")

        # ------------------------------------------------------------------ nulls --------------
        if a.shard == 0 and not a.only:
            R["nulls"] = {}
            frozen = np.repeat(x0[tracers[MET.MARGIN_SAFE]].cpu().numpy()[None], G, axis=0)
            R["nulls"]["do_nothing"] = CT.read_metrics(frozen, real20)
            d0 = d_ref[:, interior]
            R["nulls"]["do_nothing"]["coarse"] = {
                "R2_displacement_interior": float(1.0 - d0.pow(2).sum() / ss_tot),
                "motion_energy_ratio_interior": 0.0}
            dA = (recA - x0[None])[:, interior]
            e_A, e_R = float(dA.pow(2).sum()), float(d0.pow(2).sum())
            # the replay bar, at three scales: raw, energy-matched, and best-of-sweep
            for tag, s in (("replay_raw", 1.0), ("replay_energy_matched", math.sqrt(e_R / e_A))):
                rep = x0[None] + s * (recA - x0[None])
                m = CT.read_metrics(rep[:, tracers[MET.MARGIN_SAFE]].cpu().numpy(), real20)
                dR = (rep - x0[None])[:, interior]
                a_rep, _ = percell_amplitude(rep, x0, cid, C, interior)
                R["nulls"][tag] = {"scale": s, "margin20": m, "t2": t2_of(m),
                                   "coarse": {"R2_displacement_interior":
                                              float(1 - (dR - d0).pow(2).sum() / ss_tot),
                                              "motion_energy_ratio_interior":
                                              float(dR.pow(2).sum() / d0.pow(2).sum())},
                                   "percell": r2_percell(a_rep, a_ref, keep),
                                   "a_percell": a_rep.tolist()}
            R["nulls"]["identity"] = CT.read_metrics(real20, real20)
            R["nulls"]["identity"]["percell"] = r2_percell(a_ref, a_ref, keep)
            rs = np.random.RandomState(11)
            sh = a_ref.copy()
            idx = np.where(keep)[0]
            sh[idx] = a_ref[idx][rs.permutation(idx.size)]
            R["nulls"]["percell_shuffled_reference"] = {"percell": r2_percell(sh, a_ref, keep),
                                                        "a_percell": sh.tolist()}
            flat = np.where(keep, float(np.nanmean(a_ref[keep])), np.nan)
            R["nulls"]["percell_flat_mean"] = {"percell": r2_percell(flat, a_ref, keep),
                                               "a_percell": flat.tolist()}
            R["campaign_nulls"] = {"loopscore_predict_nothing": 0.0700,
                                   "loopscore_replay_fit_beat": 0.851,
                                   "loopscore_replay_heldout": 0.62,
                                   "note": "REAL recording; not commensurate with these rows"}
            log("\n[nulls]")
            for k_, v in R["nulls"].items():
                pc = v.get("percell", {})
                pcs = f"{pc['r2']:.4f}" if isinstance(pc, dict) and pc.get("r2") is not None else "n/a"
                log(f"    {k_:<30s} loop {CT.fmt(v.get('loopscore'),8)}  r2cell {pcs:>8s}")

    R["wall_seconds"] = time.time() - t_start
    sfx = "" if a.nshards == 1 else f"_s{a.shard}"
    if a.only:
        sfx += "_only"
    json.dump(R, open(os.path.join(HERE, f"crash_{a.tag}{sfx}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"crash_{a.tag}{sfx}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote crash_{a.tag}{sfx}.json  [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

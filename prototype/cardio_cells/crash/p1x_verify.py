#!/usr/bin/env python
"""p1x_verify.py -- ADVERSARIAL re-run of probe D-gainonly's decisive numbers.

WHAT IS BEING CHECKED, and how this differs from re-running p1d_gainfit.py

  1. REPRODUCIBILITY of the three numbers the verdict rests on:
        joint frame-cadence med|dE/E|  clean / derived-F(15px) / noise-grid48
        the statistic for derived-F gain@E132              (D reports 90.20)
        the statistic for a PERFECT gain pinned at E=132   (D reports  5.32)

  2. THE NULL BAR, measured ON THIS SHEET instead of imported. `accept.null_steps` divides the
     REGISTRY's null (floors.py N0 = what predicting nothing costs ON THE RECORDING) by this
     sheet's floors. If the synthetic sheet beats with a different amplitude from the recording,
     that bar is not the do-nothing model's score here. So the do-nothing model is ROLLED OUT --
     literally a frozen sheet, sim = frame 0 repeated -- and scored through the same `accept`.

  3. TICK DEPENDENCE: two DISJOINT score-tick triples, (165,180,195) and (170,185,200).

  4. THE SOLVE: every fit is done twice, once through recover.Solver's normal equations (D's
     route) and once through torch.linalg.lstsq/gelsd on the same column-scaled design, so a
     normal-equations artefact shows as a disagreement.

  5. TWO CHARITABLE GAIN-ONLY ESTIMATORS D did not run, both still inside the pivot
     ("declare E's shape, fit the gain"):
        alphaE+gain     one extra unknown, a global stiffness scale: [A_E E_fix | A_g][alpha; g]=b
        boxFISTA        the DECLARED box enforced INSIDE the solve, not projected on afterwards
     If either gets under the null under derived F, D's verdict is wrong.

usage: PYTHONPATH=/workspace/Plexus/src python p1x_verify.py --device cuda:1
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

import recover as RC                                                    # noqa: E402
from recover import Solver                                              # noqa: E402
import metrics as MET                                                   # noqa: E402
import crash_test as CT                                                 # noqa: E402
import accept as ACC                                                    # noqa: E402
from finject import lerp, assemble_inj, record_substeps                 # noqa: E402
from freal_derivedF import ControlGrid, derive_F, collect, PX, GRID_PX  # noqa: E402
from round5_fit import SIGMA_F, SNAP                                    # noqa: E402
from round5_solve import pstats                                         # noqa: E402
from refute5_fit import NoiseF                                          # noqa: E402


def solve_block(A, scale):
    orig = RC.theta_scale
    RC.theta_scale = lambda C, device, **kw: scale.to(device)
    try:
        return RC.Solver(A, A.shape[1])
    finally:
        RC.theta_scale = orig


def lstsq_scaled(A, b, s):
    Az = A.double() * s[None, :]
    z = torch.linalg.lstsq(Az.cpu(), b.double().cpu().unsqueeze(1), driver="gelsd").solution
    return z.squeeze(1).to(A.device) * s


def fista_box(A, b, lo, hi, iters=4000):
    G = A.T @ A
    r = A.T @ b
    L = float(torch.linalg.eigvalsh(G).max())
    x = torch.ones(A.shape[1], device=A.device, dtype=A.dtype).clamp(lo, hi)
    y, t = x.clone(), 1.0
    for _ in range(iters):
        xn = (y - (G @ y - r) / L).clamp(lo, hi)
        tn = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    return x


def gstats(g_hat, g_true):
    g_hat, g_true = np.asarray(g_hat, float), np.asarray(g_true, float)
    r = np.abs(g_hat - g_true) / g_true
    return {"med_rel": float(np.median(r)), "p90_rel": float(np.percentile(r, 90)),
            "corr": float(np.corrcoef(g_hat, g_true)[0, 1]),
            "atten": float(np.polyfit(g_true, g_hat, 1)[0]),
            "n_negative": int((g_hat < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="p1x")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--ticks-a", default="165,180,195")
    ap.add_argument("--ticks-b", default="170,185,200")
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--hpx", type=float, default=GRID_PX)
    ap.add_argument("--seed-noise", type=int, default=90210)
    ap.add_argument("--noise-nodes", type=int, default=48)
    ap.add_argument("--no-roll", action="store_true")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R, t_start = {"args": vars(a)}, time.time()
    tA = [int(v) for v in a.ticks_a.split(",")]
    tB = [int(v) for v in a.ticks_b.split(",")]
    torch.manual_seed(0)

    with torch.no_grad():
        sy, REF, B = collect(args, a.t0, max(tA + tB), log)
        C, n_frame, dx = sy.C, sy.n_sub_per_frame, sy.g.dx
        th = sy.theta_true.double()
        E_true, g_true = th[:C], th[C:]
        dev, f64 = th.device, torch.float64
        X0, F0ref = REF[0]["x"], REF[0]["F"]
        k0 = a.t0
        R["F0ref_max_abs_dev_from_I"] = float(
            (F0ref - torch.eye(2, device=dev, dtype=f64)).abs().max())
        log(f"[collect] {time.time()-t_start:.0f}s; |F(tick0)-I|max = "
            f"{R['F0ref_max_abs_dev_from_I']:.1e}  (0 => the derived-F reference is not an oracle)")

        for kk, vv in B[k0]["snap"].items():
            setattr(sy, kk, vv.clone())
        cg = ControlGrid(X0, a.hpx * PX)

        def derF(x):
            return derive_F(cg, X0, x, "bilinear", F_ref=F0ref)

        band = 0.06 / MET.SHEET_SPAN
        xW = B[k0]["x0"]
        interior = ~((xW[:, 0] < band) | (xW[:, 0] > 1 - band)
                     | (xW[:, 1] < band) | (xW[:, 1] > 1 - band))

        NF = NoiseF("grid", B[k0]["x0"], a.noise_nodes, dev, f64)
        gnoise = torch.Generator(device=dev).manual_seed(a.seed_noise)
        n = n_frame
        for kk, vv in B[k0]["snap"].items():
            setattr(sy, kk, vv.clone())
        Fs, _, Xs = record_substeps(sy, n)
        F0, F1, xN = sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()
        x0o = sy.x0.clone()
        e0 = (SIGMA_F / 2.0) * NF(gnoise)
        e1 = (SIGMA_F / 2.0) * NF(gnoise)
        conds = {"clean": lerp(F0, F1, n),
                 f"noiseF_grid{a.noise_nodes}": lerp(F0 + e0, F1 + e1, n),
                 f"derivedF_{a.hpx:g}px": lerp(derF(x0o), derF(xN), n)}
        Ftrue = Fs.clone()

        E132 = torch.full((C,), 132.0, device=dev, dtype=f64)
        thetas, R["fits"], R["balance"], R["alpha"] = {}, {}, {}, {}
        log("")
        log(f"    {'F':<18s} {'fit':<16s} {'medg':>8s} {'p90g':>9s} {'corr':>7s} {'atten':>10s} "
            f"{'negg':>5s} {'medE':>8s} {'resid':>9s} {'lstsq-vs-norm':>14s}")
        for fname, iF in conds.items():
            de = (iF - Ftrue)[:, interior].abs()
            for kk, vv in B[k0]["snap"].items():
                setattr(sy, kk, vv.clone())
            A, y0, _ = assemble_inj(sy, n, iF, None)
            b = (xN - x0o).reshape(-1) - y0
            A_E, A_g = A[:, :C].contiguous(), A[:, C:].contiguous()
            yE, yg = A_E @ E_true, A_g @ g_true
            R["balance"][fname] = {
                "errF_med_interior": float(de.median()),
                "norm_b": float(b.norm()), "norm_A_E_Etrue": float(yE.norm()),
                "norm_A_g_gtrue": float(yg.norm()),
                "elastic_over_active": float(yE.norm() / yg.norm()),
                "norm_A_E_Etrue_over_b": float(yE.norm() / b.norm()),
                "norm_A_E_E132_over_b": float((A_E @ E132).norm() / b.norm())}
            log(f"  [{fname}] ||b|| {float(b.norm()):.4g}  ||A_E E_true|| {float(yE.norm()):.4g} "
                f" ||A_g g_true|| {float(yg.norm()):.4g}  elastic/active "
                f"{R['balance'][fname]['elastic_over_active']:.2f}x  "
                f"||A_E E_true||/||b|| {R['balance'][fname]['norm_A_E_Etrue_over_b']:.2f}x")

            fits, alt = {}, {}
            S = Solver(A, C)
            sol = S(b)
            fits["joint"] = sol["ridge0"]
            alt["joint"] = lstsq_scaled(A, b, RC.theta_scale(C, dev))
            S.free()
            del S, sol
            for lab, Efx in (("gain@E_true", E_true), ("gain@E132", E132)):
                rhs = b - A_E @ Efx
                Sg = solve_block(A_g, torch.ones(C, device=dev, dtype=f64))
                sg = Sg(rhs)
                fits[lab] = torch.cat([Efx, sg["ridge0"]])
                alt[lab] = torch.cat([Efx, lstsq_scaled(A_g, rhs,
                                                        torch.ones(C, device=dev, dtype=f64))])
                Sg.free()
                del Sg, sg
            col = (A_E @ E132).unsqueeze(1)
            A2 = torch.cat([col, A_g], 1)
            z2 = lstsq_scaled(A2, b, torch.ones(C + 1, device=dev, dtype=f64))
            alpha = float(z2[0])
            R["alpha"][fname] = alpha
            fits["alphaE+gain"] = torch.cat([alpha * E132, z2[1:]])
            alt["alphaE+gain"] = fits["alphaE+gain"]
            fits["boxFISTA@E132"] = torch.cat([E132, fista_box(A_g, b - A_E @ E132, 0.2, 5.0)])
            alt["boxFISTA@E132"] = fits["boxFISTA@E132"]
            fits["boxFISTA@alphaE"] = torch.cat(
                [alpha * E132, fista_box(A_g, b - alpha * (A_E @ E132), 0.2, 5.0)])
            alt["boxFISTA@alphaE"] = fits["boxFISTA@alphaE"]

            for lab, t_hat in fits.items():
                key = f"{fname}|{lab}"
                thetas[key] = t_hat.clone()
                gs = gstats(t_hat[C:].cpu().numpy(), g_true.cpu().numpy())
                ps = pstats(t_hat.cpu().numpy(), th.cpu().numpy(), C)
                resid = float((A @ t_hat - b).norm() / b.norm())
                d_alt = float((t_hat - alt[lab]).norm() / t_hat.norm())
                R["fits"][key] = {"gain": gs, "med_E": ps["med_E"], "fit_residual": resid,
                                  "lstsq_vs_normal_rel": d_alt, "alpha": alpha}
                log(f"    {fname:<18s} {lab:<16s} {gs['med_rel']:>8.4f} {gs['p90_rel']:>9.4f} "
                    f"{gs['corr']:>7.3f} {gs['atten']:>10.3f} {gs['n_negative']:>5d} "
                    f"{ps['med_E']:>8.4f} {resid:>9.3e} {d_alt:>14.2e}")
            del A, A_E, A_g, A2, col
            torch.cuda.empty_cache()

        R["controls"] = {"target": {"clean": 0.0078, "derivedF_15px": 0.9986,
                                    "noiseF_grid48": 0.8416},
                         "got": {k.split("|")[0]: v["med_E"] for k, v in R["fits"].items()
                                 if k.endswith("|joint")}}
        log(f"\n[controls] joint med|dE/E|, frame cadence: got "
            f"{ {k: round(v, 4) for k, v in R['controls']['got'].items()} }")
        log(f"           alpha (global stiffness scale the data wants on E=132): "
            f"{ {k: round(v, 5) for k, v in R['alpha'].items()} }")

        if a.no_roll:
            json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
            open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
            return

        def boxed(t):
            return torch.cat([t[:C].clamp(26.4, 660.0), t[C:].clamp(0.2, 5.0)])

        dF = f"derivedF_{a.hpx:g}px"
        cands = {"theta_true": th.clone(),
                 "null_gain_true_at_E132": torch.cat([E132, g_true]),
                 f"{dF}|gain@E132": boxed(thetas[f"{dF}|gain@E132"]),
                 f"{dF}|alphaE+gain": boxed(thetas[f"{dF}|alphaE+gain"]),
                 f"{dF}|boxFISTA@E132": boxed(thetas[f"{dF}|boxFISTA@E132"]),
                 f"{dF}|boxFISTA@alphaE": boxed(thetas[f"{dF}|boxFISTA@alphaE"]),
                 f"{dF}|joint": boxed(thetas[f"{dF}|joint"]),
                 "clean|gain@E132": boxed(thetas["clean|gain@E132"]),
                 "clean|alphaE+gain": boxed(thetas["clean|alphaE+gain"])}
        floors = ACC.working_floors()
        R["null_steps_imported"] = ACC.null_steps(floors)
        G = a.window
        loops = {k: {} for k in cands}
        loops["ON-SHEET NULL (frozen sheet)"] = {}
        R["reference_amplitude"] = {}
        for T in sorted(set(tA + tB)):
            for kk, vv in B[T]["snap"].items():
                setattr(sy, kk, vv.clone())
            snapT = {kk: getattr(sy, kk).clone() for kk in SNAP}
            x0T = sy.x0.clone()
            trc = {MET.MARGIN_SAFE: CT.tracer_indices(x0T, CT.probe_points(MET.MARGIN_SAFE))}
            tc = time.time()
            _, ref_full, _ = CT.rollout(sy, th, T, G, trc, keep_full=True)
            real = ref_full[:, trc[MET.MARGIN_SAFE]].cpu().numpy()
            R["reference_amplitude"][T] = {
                "raw_peak_excursion": float(MET.REGISTRY["peak_excursion"].reading(real)),
                "raw_path_length": float(MET.REGISTRY["path_length"].reading(real)),
                "max_disp_dx": float((ref_full - x0T[None]).norm(dim=-1).max() / dx)}
            # THE DO-NOTHING MODEL, on this sheet, on the same reading surface
            loops["ON-SHEET NULL (frozen sheet)"][T] = np.repeat(real[:1], real.shape[0], axis=0)
            log(f"    tick {T}: ref in {time.time()-tc:.0f}s; RAW peak_excursion "
                f"{R['reference_amplitude'][T]['raw_peak_excursion']:.6f} (registry null 0.0011); "
                f"RAW path_length {R['reference_amplitude'][T]['raw_path_length']:.6f} "
                f"(registry null 0.0042)")
            for name, theta in cands.items():
                tc = time.time()
                for kk, vv in snapT.items():
                    setattr(sy, kk, vv.clone())
                tr, _, _ = CT.rollout(sy, theta, T, G, trc)
                loops[name][T] = tr[MET.MARGIN_SAFE].cpu().numpy()
                log(f"      {name:<36s} [{time.time()-tc:.0f}s]")
            del ref_full
            torch.cuda.empty_cache()

        R["accept"] = {}
        for label, ticks in (("A(165,180,195)", tA), ("B(170,185,200)", tB)):
            R["accept"][label] = {}
            for name in loops:
                pairs = [(loops[name][t], loops["theta_true"][t]) for t in ticks]
                if not all(np.isfinite(p[0]).all() for p in pairs):
                    R["accept"][label][name] = {"statistic": float("inf"),
                                                "limiting_instrument": "DIVERGED",
                                                "informative": False}
                    continue
                v = ACC.accept(pairs, floors)
                v.pop("per_tick", None)
                R["accept"][label][name] = v
        nul_min = min(R["null_steps_imported"].values())
        onsheet = R["accept"]["A(165,180,195)"]["ON-SHEET NULL (frozen sheet)"]["statistic"]
        log(f"\n[r] the statistic on two DISJOINT tick triples.")
        log(f"    imported null bar (registry N0 / this sheet's floors): min {nul_min:.2f}, "
            f"peak_excursion {R['null_steps_imported']['peak_excursion']:.2f}")
        log(f"    ON-SHEET null bar (a frozen sheet, scored the same way):  {onsheet:.2f}")
        log(f"    {'candidate':<36s} {'A':>9s} {'B':>9s} {'limiting(A)':<16s} {'info':>5s}")
        for name in loops:
            va = R["accept"]["A(165,180,195)"][name]
            vb = R["accept"]["B(170,185,200)"][name]
            log(f"    {name:<36s} {va['statistic']:>9.2f} {vb['statistic']:>9.2f} "
                f"{va.get('limiting_instrument',''):<16s} {str(va.get('informative')):>5s}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f}s]")


if __name__ == "__main__":
    main()

"""refute_spectrum.py -- TASK G: adversarial re-check of task E's identifiability claim.

Four attacks, each with a number and a control:

  U  UNITS.  Is cond(A)=2.78e7 an artefact of mixing E~1e2 with gain~1e0?  Compute cond under
     five column scalings including a BLOCK-ONLY scaling (one scalar per block: the pure unit
     fix, contains no per-cell information) and the two single-block submatrices A_E, A_gain.
     If the units story were right, block-only scaling should collapse cond.

  P  IS sigma_min REAL?  The bottom of the spectrum is set by ~20 tiny columns.  A is built by
     finite differences with probe amplitude sE=100 / sg=1.  For a truly affine map A is
     s-INDEPENDENT.  Re-assemble at 10x and 0.1x probe and measure the per-column relative
     change, resolved by column norm.  Junk columns would move; real ones would not.

  F  A DIFFERENT FRAME.  Task E only ever analysed the near-kernel at tick 12.  Advance the true
     dynamics and redo cond / PR / block weight / worst-cell identity.

  T  A DIFFERENT PLANTED THETA.  gain_true is a fixed seed-101 draw and E_true is the measured
     segmentation.  Re-warm the system with a different gain draw and a permuted E field and
     redo the same statistics.  (A depends on theta_true only through the frozen state and
     through the dimensionless scaling -- but that is enough to change everything.)

usage
  PYTHONPATH=/workspace/Plexus/src python refute_spectrum.py --device cuda:0 --real --attack U,P
  PYTHONPATH=/workspace/Plexus/src python refute_spectrum.py --device cuda:0 --real --attack F,T
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assemble import System, SUBSTEP_TOKENS          # noqa: E402
from spectrum import r_factor, advance              # noqa: E402

HERE = "/workspace/Plexus/prototype/cardio_cells/algebraic"


def svals(A):
    return torch.linalg.svdvals(r_factor(A))


def cond_of(A, d=None):
    s = svals(A if d is None else A * d[None, :])
    return float(s[0] / s[-1]), float(s[0]), float(s[-1])


def kernel_stats(A, th, C, n_worst=6):
    """Everything task E reports about the near-kernel, on the dimensionless scaling."""
    Ad = A * th.abs().clamp(min=1e-300)[None, :]
    R = r_factor(Ad)
    U, s, Vh = torch.linalg.svd(R)
    V = Vh.mH
    eps = (s / s[0]) ** 2
    prs = (1.0 / (V ** 4).sum(dim=0)).cpu().numpy()          # index 0 = largest sigma
    nb = max(1, int(0.10 * 2 * C))
    m5 = max(1, int(0.05 * 2 * C))
    Vb = V[:, -m5:]
    worst = []
    for k in range(n_worst):
        v = V[:, -1 - k]
        wc = v[:C] ** 2 + v[C:] ** 2
        worst.append({"k": k, "eps": float(eps[-1 - k]),
                      "PR": float(prs[-1 - k]),
                      "wE": float((v[:C] ** 2).sum()), "wg": float((v[C:] ** 2).sum()),
                      "dominant_cell": int(torch.argmax(wc)) + 1})
    # per-cell se and the identity of the worst cells
    inv2 = 1.0 / (s ** 2)
    se = ((V ** 2) * inv2[None, :]).sum(dim=1).sqrt()
    cn = Ad.norm(dim=0)
    vif = (cn ** 2) * (se ** 2)
    Ginv = (V * inv2[None, :]) @ V.T
    dg = Ginv.diagonal().clamp(min=1e-300).sqrt()
    rho = (Ginv / dg[:, None] / dg[None, :]).diagonal(offset=C)
    del Ginv
    order_E = torch.argsort(se[:C], descending=True)[:25]
    order_g = torch.argsort(se[C:], descending=True)[:25]
    out = {
        "cond_A_dimensionless": float(s[0] / s[-1]),
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
        "rank_eps>1e-2": int((eps > 1e-2).sum()), "rank_eps>1e-4": int((eps > 1e-4).sum()),
        "rank_eps>1e-8": int((eps > 1e-8).sum()),
        "PR_median_bottom10pct": float(np.median(prs[-nb:])),
        "PR_median_top10pct": float(np.median(prs[:nb])),
        "n_bottom10pct_PR_below_2": int((prs[-nb:] < 2).sum()),
        "PR_null_random_unit_vector": (2 * C + 2) / 3.0,
        "bottom5pct_mean_weight_gain": float((Vb[C:] ** 2).sum() / m5),
        "bottom5pct_mean_weight_E": float((Vb[:C] ** 2).sum() / m5),
        "worst_directions": worst,
        "median_abs_rho_Ec_gainc": float(rho.abs().median()),
        "max_abs_rho_Ec_gainc": float(rho.abs().max()),
        "n_cells_abs_rho_gt_0.9": int((rho.abs() > 0.9).sum()),
        "median_VIF": float(vif.median()), "max_VIF": float(vif.max()),
        "n_VIF_gt_100": int((vif > 100).sum()),
        "colnorm_spread": float(cn.max() / cn.min()),
        "se_E_median": float(se[:C].median()), "se_E_max": float(se[:C].max()),
        "se_gain_median": float(se[C:].median()), "se_gain_max": float(se[C:].max()),
        "worst25_cells_E": [int(i) + 1 for i in order_E],
        "worst25_cells_gain": [int(i) + 1 for i in order_g],
        "colnorm_E": cn[:C].cpu().numpy(), "colnorm_gain": cn[C:].cpu().numpy(),
        "se_E": se[:C].cpu().numpy(), "se_gain": se[C:].cpu().numpy(),
    }
    del Ad, R, U, Vh
    torch.cuda.empty_cache()
    return out, V, s


# ------------------------------------------------------------------------------------------- #
def attack_U(sy, A, log):
    """UNITS.  Does a scaling that carries NO per-cell information fix the conditioning?"""
    C = sy.C
    th = sy.theta_true.double()
    out = {}
    scalings = {
        "raw": torch.ones_like(th),
        "block_only_theta_median": torch.cat([
            torch.full((C,), float(th[:C].median()), device=th.device, dtype=th.dtype),
            torch.full((C,), float(th[C:].median()), device=th.device, dtype=th.dtype)]),
        "block_only_colnorm_median": None,      # filled below
        "dimensionless_percell": th.abs(),
        "equilibrated_percell": None,
    }
    cn_raw = A.norm(dim=0).clamp(min=1e-300)
    scalings["block_only_colnorm_median"] = torch.cat([
        torch.full((C,), 1.0 / float(cn_raw[:C].median()), device=th.device, dtype=th.dtype),
        torch.full((C,), 1.0 / float(cn_raw[C:].median()), device=th.device, dtype=th.dtype)])
    scalings["equilibrated_percell"] = 1.0 / cn_raw
    for nm, d in scalings.items():
        c, smax, smin = cond_of(A, d)
        out[nm] = {"cond_A": c, "sigma_max": smax, "sigma_min": smin, "cond_G": c ** 2}
        log(f"   [U] cond_2(A) [{nm:28s}] = {c:.4e}   (cond_2(G) = {c**2:.3e})")
    # the two blocks on their own -- inside a block every column has IDENTICAL units
    for nm, sl in (("A_E_block_only", slice(0, C)), ("A_gain_block_only", slice(C, 2 * C))):
        sub = A[:, sl]
        c, smax, smin = cond_of(sub)
        cd, _, _ = cond_of(sub, th[sl].abs())
        cnb = sub.norm(dim=0)
        out[nm] = {"cond_A_raw": c, "cond_A_dimensionless": cd,
                   "colnorm_spread": float(cnb.max() / cnb.min()),
                   "theta_true_spread": float(th[sl].abs().max() / th[sl].abs().min())}
        log(f"   [U] {nm}: cond(raw)={c:.4e} cond(dimensionless)={cd:.4e} "
            f"colnorm spread={float(cnb.max()/cnb.min()):.3e} "
            f"(theta spread only {float(th[sl].abs().max()/th[sl].abs().min()):.2f}x)")
    # how much of the column-norm spread does theta_true explain?
    cnd = (A * th.abs()[None, :]).norm(dim=0)
    out["explained_by_units"] = {
        "colnorm_spread_raw": float(cn_raw.max() / cn_raw.min()),
        "colnorm_spread_dimensionless": float(cnd.max() / cnd.min()),
        "spread_of_theta_true": float(th.abs().max() / th.abs().min()),
        "note": "if units drove the spread, dividing by theta_true would collapse it"}
    log(f"   [U] colnorm spread raw {float(cn_raw.max()/cn_raw.min()):.3e} -> dimensionless "
        f"{float(cnd.max()/cnd.min()):.3e};  theta_true itself only spans "
        f"{float(th.abs().max()/th.abs().min()):.1f}x")
    return out


def attack_P(sy, A, log, factors=(10.0, 0.1)):
    """PROBE AMPLITUDE.  For an affine map the finite-difference columns are s-independent."""
    out = {}
    cn0 = A.norm(dim=0)
    small = torch.argsort(cn0)[:20]
    for f in factors:
        A2, _, t = sy.assemble(n_sub=1, sE=100.0 * f, sg=1.0 * f)
        A2 = A2.double()
        dcol = (A2 - A).norm(dim=0) / cn0.clamp(min=1e-300)
        c2, smax2, smin2 = cond_of(A2, sy.theta_true.double().abs())
        out[f"probe_x{f:g}"] = {
            "assembly_s": t,
            "global_rel_change": float((A2 - A).norm() / A.norm()),
            "percol_rel_change_median": float(dcol.median()),
            "percol_rel_change_max": float(dcol.max()),
            "percol_rel_change_on_20_smallest_columns_max": float(dcol[small].max()),
            "percol_rel_change_on_20_smallest_columns_median": float(dcol[small].median()),
            "n_columns_rel_change_gt_1e-6": int((dcol > 1e-6).sum()),
            "n_columns_rel_change_gt_1e-2": int((dcol > 1e-2).sum()),
            "n_columns_rel_change_gt_1e-1": int((dcol > 1e-1).sum()),
            "colnorm_percentile_of_moved_columns": [
                float((cn0 < cn0[i]).float().mean()) for i in
                torch.argsort(dcol, descending=True)[:10]],
            "colnorm_of_moved_columns": [float(cn0[i]) for i in
                                         torch.argsort(dcol, descending=True)[:10]],
            "colnorm_median": float(cn0.median()),
            "cond_A_dimensionless": c2, "sigma_min_dimensionless": smin2,
            "cond_ratio_vs_reference": c2 / float(
                cond_of(A, sy.theta_true.double().abs())[0]),
        }
        log(f"   [P] probe x{f:g}: global ||dA||/||A|| = {float((A2-A).norm()/A.norm()):.3e}; "
            f"per-column rel change median {float(dcol.median()):.3e} max {float(dcol.max()):.3e}; "
            f"on the 20 SMALLEST columns max {float(dcol[small].max()):.3e}")
        log(f"        cond(A,dimensionless) = {c2:.4e}  sigma_min = {smin2:.4e}")
        del A2
        torch.cuda.empty_cache()
    return out


def restat(sy, log, tag):
    A, a0, t = sy.assemble(n_sub=1)
    A = A.double()
    b = sy.a_of_theta(sy.theta_true, n_sub=1) - a0
    resid = float((A @ sy.theta_true - b).norm() / b.norm())
    ks, V, s = kernel_stats(A, sy.theta_true.double(), sy.C)
    ks["residual_rel_b"] = resid
    ks["norm_b"] = float(b.norm())
    log(f"   [{tag}] cond={ks['cond_A_dimensionless']:.4e}  sigma_min={ks['sigma_min']:.4e}  "
        f"resid={resid:.3e}  PR(bottom10%%)={ks['PR_median_bottom10pct']:.2f} "
        f"(null {ks['PR_null_random_unit_vector']:.0f})  bottom5%% gain weight="
        f"{ks['bottom5pct_mean_weight_gain']:.3f}  maxVIF={ks['max_VIF']:.1f}  "
        f"max|rho(E,g)|={ks['max_abs_rho_Ec_gainc']:.3f}  "
        f"colnorm spread={ks['colnorm_spread']:.3e}")
    log(f"        worst direction: eps={ks['worst_directions'][0]['eps']:.3e} "
        f"PR={ks['worst_directions'][0]['PR']:.2f} wg={ks['worst_directions'][0]['wg']:.3f} "
        f"cell {ks['worst_directions'][0]['dominant_cell']}")
    del A, V
    torch.cuda.empty_cache()
    return ks


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))


# ------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=500)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--attack", default="U,P,F,T")
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or ("real" if args.real else f"C{args.cells}")
    at = set(args.attack.split(","))
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    RES = {"argv": vars(args)}
    t00 = time.time()
    with torch.no_grad():
        kw = dict(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                  n_grid=args.n_grid, dtype="float64", mode="full", real=args.real)
        sy = System(warmup=args.warmup, **kw)
        log(f"[{tag}] C={sy.C} Np={sy.Np} grid {sy.g.nx}^2 warmup {sy.warmup_frames}")
        A, a0, t = sy.assemble(n_sub=1)
        A = A.double()
        b = sy.a_of_theta(sy.theta_true, n_sub=1) - a0
        resid = float((A @ sy.theta_true - b).norm() / b.norm())
        RES["reference"] = {"A_shape": list(A.shape), "assembly_s": t,
                            "residual_rel_b": resid, "norm_b": float(b.norm())}
        log(f"   reference A {tuple(A.shape)} in {t:.1f}s, residual {resid:.3e}")

        if "U" in at:
            log("\n=== ATTACK U: is the conditioning a UNITS artefact? ===")
            RES["U_units"] = attack_U(sy, A, log)
        if "P" in at:
            log("\n=== ATTACK P: is sigma_min real, or finite-difference junk? ===")
            RES["P_probe"] = attack_P(sy, A, log)
        del A
        torch.cuda.empty_cache()

        if "F" in at or "T" in at:
            log("\n=== BASELINE (tick 12, planted theta 1) restated with the same estimator ===")
            base = restat(sy, log, "base")
            RES["baseline"] = {k: v for k, v in base.items() if not isinstance(v, np.ndarray)}

        if "F" in at:
            log(f"\n=== ATTACK F: a DIFFERENT frame (+{args.stride} frames) ===")
            advance(sy, args.stride)
            f1 = restat(sy, log, f"tick{sy.warmup_frames}")
            advance(sy, args.stride)
            f2 = restat(sy, log, f"tick{sy.warmup_frames}")
            RES["F_frames"] = {}
            for nm, ks in (("frame_b", f1), ("frame_c", f2)):
                RES["F_frames"][nm] = {k: v for k, v in ks.items()
                                       if not isinstance(v, np.ndarray)}
                RES["F_frames"][nm]["jaccard_worst25_E_vs_base"] = jaccard(
                    ks["worst25_cells_E"], base["worst25_cells_E"])
                RES["F_frames"][nm]["jaccard_worst25_gain_vs_base"] = jaccard(
                    ks["worst25_cells_gain"], base["worst25_cells_gain"])
                RES["F_frames"][nm]["spearman_se_E_vs_base"] = float(np.corrcoef(
                    np.argsort(np.argsort(ks["se_E"])),
                    np.argsort(np.argsort(base["se_E"])))[0, 1])
                RES["F_frames"][nm]["spearman_se_gain_vs_base"] = float(np.corrcoef(
                    np.argsort(np.argsort(ks["se_gain"])),
                    np.argsort(np.argsort(base["se_gain"])))[0, 1])
                log(f"   [F/{nm}] worst-25 cell overlap with tick 12: "
                    f"E {RES['F_frames'][nm]['jaccard_worst25_E_vs_base']:.2f}  "
                    f"gain {RES['F_frames'][nm]['jaccard_worst25_gain_vs_base']:.2f};  "
                    f"rank corr of se over all cells: "
                    f"E {RES['F_frames'][nm]['spearman_se_E_vs_base']:+.3f} "
                    f"gain {RES['F_frames'][nm]['spearman_se_gain_vs_base']:+.3f}")

        if "T" in at:
            log("\n=== ATTACK T: a DIFFERENT planted theta (new gain draw + permuted E) ===")
            del sy
            torch.cuda.empty_cache()
            sy2 = System(warmup=0, **kw)               # build, do NOT warm up yet
            g = torch.Generator().manual_seed(777)
            newgain = torch.zeros_like(sy2.gain_true)
            newgain[1:] = (0.5 + 1.0 * torch.rand(sy2.C, generator=g)).to(
                sy2.device, sy2.dtype)
            perm = torch.randperm(sy2.C, generator=g).to(sy2.device)
            newE = sy2.E_true.clone()
            newE[1:] = sy2.E_true[1:][perm]
            sy2.gain_true, sy2.E_true = newgain, newE
            sy2.theta_true = torch.cat([newE[1:], newgain[1:]])
            sy2.warmup_frames = args.warmup
            sy2._warmup(args.warmup)
            sy2._snapshot(args.warmup)
            log(f"   re-warmed with a NEW theta: E spread "
                f"{float(newE[1:].max()/newE[1:].min()):.2f}x, gain in "
                f"[{float(newgain[1:].min()):.3f},{float(newgain[1:].max()):.3f}]")
            t2 = restat(sy2, log, "theta2")
            RES["T_theta"] = {k: v for k, v in t2.items() if not isinstance(v, np.ndarray)}
            if "baseline" in RES:
                RES["T_theta"]["jaccard_worst25_E_vs_base"] = jaccard(
                    t2["worst25_cells_E"], base["worst25_cells_E"])
                RES["T_theta"]["jaccard_worst25_gain_vs_base"] = jaccard(
                    t2["worst25_cells_gain"], base["worst25_cells_gain"])
                RES["T_theta"]["spearman_se_E_vs_base"] = float(np.corrcoef(
                    np.argsort(np.argsort(t2["se_E"])),
                    np.argsort(np.argsort(base["se_E"])))[0, 1])
                log(f"   [T] worst-25 cell overlap with planted theta 1: "
                    f"E {RES['T_theta']['jaccard_worst25_E_vs_base']:.2f}  "
                    f"gain {RES['T_theta']['jaccard_worst25_gain_vs_base']:.2f};  "
                    f"se_E rank corr {RES['T_theta']['spearman_se_E_vs_base']:+.3f}")

    log(f"\ntotal {time.time()-t00:.1f} s")
    p = os.path.join(HERE, f"refute_{tag}.json")
    json.dump(RES, open(p, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"refute_{tag}.log"), "w").write("\n".join(lines) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()

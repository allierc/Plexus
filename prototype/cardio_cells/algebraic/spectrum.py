"""spectrum.py -- TASK E: what the Gram matrix says about identifiability of the per-cell
parameters theta = (E_1..E_C, gain_1..gain_C) from ONE algebraic (no-integration) constraint.

Formalism and names follow /workspace/connectome-gnn-cx/neurips_review/linalg_note.tex:

    A theta = b                                       (eq:ls)      the algebraic constraint
    G := A^T A                                        (eq:normal)  the Gram matrix = the Hessian
    A = U Sigma V^T,  lambda_k(G) = sigma_k(A)^2      (eq:key)
    eps_k := lambda_k(G)/lambda_max(G) = (sigma_k/sigma_1)^2       (eq:relspec)   THE SPECTRUM
    cond_2(A) = sigma_1/sigma_n,  cond_2(G) = cond_2(A)^2          (eq:cond)
    ker A = ker G                                     (eq:kereq)   degenerate directions
    theta_hat^(eps) = sum_{eps_k>eps} (u_k^T b / sigma_k) v_k      (eq:tsvd)      TSVD
    ||d theta|| <~ ||d b|| / sigma_min^kept           (eq:amp)     error amplification

ONE METHODOLOGICAL POINT, TAKEN STRAIGHT FROM THE NOTE (sec 3.3).  Task C computed the spectrum as
`eigvalsh(A^T A).sqrt()`.  Forming G numerically SQUARES the condition number, so any sigma_k below
sqrt(eps_mach)*sigma_1 ~ 1.5e-8 * sigma_1 is destroyed.  Task C's real-system sigma_min = 8.9e-5 sits
only 3x above that floor -- i.e. it may be a numerical artefact, not a measurement.  Here every
spectrum is taken from a backward-stable QR of A (A = QR, sigma(A) = sigma(R), V(A) = V(R)), which
does not square anything.  The two routes are compared head to head, and they DISAGREE at real size.

WHAT IS COMPUTED
  1. the spectrum eps_k of G, its rank at stated tolerances, cond(A), cond(G)   [+ Gaussian null]
  2. the near-kernel: the smallest right singular vectors v_k in CELL space -- localisation
     (participation ratio, against the random-vector null n/3), E-block vs gain-block weight,
     and the within-direction Pearson correlation between the E-block and the gain-block
  3. per-cell identifiability: column norm, well-determined weight w_c(eps) = sum_{eps_k>eps} V_ck^2,
     the standard error sqrt((G^-1)_cc), the variance-inflation factor -- mapped over the cells and
     regressed (Spearman + label-permutation null) on cell covariates: distance to the wall, cell
     size, activity, speed, E_true
  4. cond vs number of FRAMES: stack A_f from several frozen states.  Memory-cheap: each A_f is
     reduced to its R factor (2C x 2C) and the R's are stacked, which has the singular values of the
     stacked A exactly.

SCALINGS.  cond of the raw A is meaningless (E ~ 1e2, gain ~ 1e0 mix units), so everything is
reported for three column scalings: raw, DIMENSIONLESS A diag(theta_true) (response to a fractional
parameter change -- the default, and what task C used), and EQUILIBRATED A D with unit column norms
(van der Sluis; note sec 5.3, tr(G) = n).

usage
  PYTHONPATH=/workspace/Plexus/src python spectrum.py --real  --device cuda:1
  PYTHONPATH=/workspace/Plexus/src python spectrum.py --cells 100 --per-parent 100 --n-grid 128
  ... --frames 8 --stride 1     (multi-frame conditioning)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assemble import System, SUBSTEP_TOKENS          # noqa: E402  (upstream, not modified)

HERE = "/workspace/Plexus/prototype/cardio_cells/algebraic"
FG, BG = "white", "black"


# ============================================================================================== #
#  linear algebra                                                                                #
# ============================================================================================== #
def r_factor(A):
    """R of a reduced QR.  sigma(R) = sigma(A) and V(R) = V(A), without ever forming G."""
    return torch.linalg.qr(A, mode="r").R


def spectrum_of(A=None, R=None, want_V=True):
    """Backward-stable spectrum.  Returns sigma, V, eps (= sigma^2/sigma_1^2 = lambda_k(G)/lmax)."""
    if R is None:
        R = r_factor(A)
    if want_V:
        U, s, Vh = torch.linalg.svd(R)
        V = Vh.mH
    else:
        s, V = torch.linalg.svdvals(R), None
    eps = (s / s[0]) ** 2
    return s, V, eps


def spec_summary(s, tag):
    eps = (s / s[0]) ** 2
    out = {"tag": tag,
           "n": int(s.numel()),
           "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
           "cond_A": float(s[0] / s[-1]), "cond_G": float((s[0] / s[-1]) ** 2),
           "lambda_max_G": float(s[0] ** 2), "lambda_min_G": float(s[-1] ** 2),
           "sigma_top5": [float(x) for x in s[:5]],
           "sigma_bot5": [float(x) for x in s[-5:]],
           "eps_bot5": [float(x) for x in eps[-5:]]}
    # rank at a stated tolerance, on eps (the note's convention) and on sigma
    for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-16, 1e-22):
        out[f"rank_eps>{t:g}"] = int((eps > t).sum().item())
    out["numpy_default_rank"] = int((s > s[0] * s.numel() * 2.22e-16).sum().item())
    # the sloppiness curve Phi(eps) = fraction of directions below eps
    out["Phi_frac_below"] = {f"{t:g}": float((eps < t).float().mean().item())
                             for t in (1e-2, 1e-4, 1e-6, 1e-8, 1e-12)}
    return out


def gaussian_null(shape, device, seed=0):
    """Same shape, i.i.d. entries: the Marchenko-Pastur reference (note sec 5.2).
    A well-posed design of this aspect ratio has cond ~ (1+sqrt(c))/(1-sqrt(c))."""
    g = torch.Generator(device=device).manual_seed(seed)
    B = torch.randn(shape, generator=g, device=device, dtype=torch.float64)
    s = torch.linalg.svdvals(r_factor(B))
    c = shape[1] / shape[0]
    return {"cond_A": float(s[0] / s[-1]),
            "cond_A_MP_prediction": float((1 + c ** 0.5) / (1 - c ** 0.5)),
            "eps_min": float((s[-1] / s[0]) ** 2)}


# ============================================================================================== #
#  per-cell covariates                                                                           #
# ============================================================================================== #
def cell_covariates(sy):
    cid, C = sy.cid, sy.C
    X, V0 = sy.x0.double(), sy.v0.double()
    nP = torch.zeros(C + 1, device=X.device, dtype=torch.float64).index_add_(
        0, cid, torch.ones_like(X[:, 0]))
    def cmean(v):
        out = torch.zeros(C + 1, device=X.device, dtype=torch.float64).index_add_(0, cid, v)
        return (out / nP.clamp(min=1))[1:]
    cx, cy = cmean(X[:, 0]), cmean(X[:, 1])
    spread = (cmean(X[:, 0] ** 2) - cx ** 2 + cmean(X[:, 1] ** 2) - cy ** 2).clamp(min=0).sqrt()
    dwall = torch.minimum(torch.minimum(cx, 1 - cx), torch.minimum(cy, 1 - cy))
    act = cmean(sy.act0.double().norm(dim=1))
    spd = cmean(V0.norm(dim=1))
    F = sy.F0.double()
    eye = torch.eye(2, device=F.device, dtype=torch.float64)
    defo = cmean((F - eye).reshape(F.shape[0], -1).norm(dim=1))
    return {"cx": cx, "cy": cy, "n_particles": nP[1:], "spread": spread,
            "dist_to_wall": dwall, "activity": act, "speed": spd, "deformation": defo,
            "E_true": sy.E_true[1:].double(), "gain_true": sy.gain_true[1:].double(),
            "r_from_centre": ((cx - 0.5) ** 2 + (cy - 0.5) ** 2).sqrt()}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def spearman_perm(a, b, n=2000, seed=0):
    """rho plus a label-permutation null (the p-value is two-sided)."""
    rho = spearman(a, b)
    rng = np.random.default_rng(seed)
    bb = np.asarray(b, float)
    null = np.array([spearman(a, rng.permutation(bb)) for _ in range(n)])
    return rho, float((np.abs(null) >= abs(rho)).mean()), float(null.std())


# ============================================================================================== #
#  the analysis of one design matrix                                                             #
# ============================================================================================== #
def analyse(A, sy, tag, log, want_kernel=True, n_worst=6):
    """A is [2Np, 2C] in float64.  Everything below is on the DIMENSIONLESS scaling unless said."""
    C = sy.C
    th = sy.theta_true.double()
    out = {}

    scalings = {
        "raw": torch.ones_like(th),
        "dimensionless": th.abs().clamp(min=1e-300),           # A diag(theta_true)
    }
    colnorm = A.norm(dim=0).clamp(min=1e-300)
    scalings["equilibrated"] = 1.0 / colnorm                   # unit column norms (van der Sluis)

    Vdim = None
    for name, d in scalings.items():
        s, V, eps = spectrum_of(A * d[None, :], want_V=(name == "dimensionless"))
        out[name] = spec_summary(s, name)
        if name == "dimensionless":
            Vdim, sdim, epsdim = V, s, eps
            out[name]["trace_G"] = float((s ** 2).sum())
        log(f"   [{tag}/{name}] cond(A)={out[name]['cond_A']:.4e} cond(G)={out[name]['cond_G']:.4e} "
            f"rank(eps>1e-8)={out[name]['rank_eps>1e-08']}/{2*C}  "
            f"eps_min={out[name]['eps_bot5'][0]:.3e}")

    # ---- CONTROL 1: the Gram route task C used, on the same matrix ---------------------------- #
    Ad = A * scalings["dimensionless"][None, :]
    G = Ad.T @ Ad
    s_gram = torch.linalg.eigvalsh(G).clamp(min=0).flip(0).sqrt()
    out["control_gram_route"] = {
        "cond_A_via_eigvalsh_of_G": float(s_gram[0] / s_gram[-1].clamp(min=1e-300)),
        "cond_A_via_QR_of_A": out["dimensionless"]["cond_A"],
        "gram_floor_sqrt_epsmach_times_sigma1": float(1.49e-8 * sdim[0]),
        "sigma_min_QR": float(sdim[-1]), "sigma_min_gram": float(s_gram[-1]),
        "ratio_QR_over_gram": float(sdim[-1] / s_gram[-1].clamp(min=1e-300))}
    log(f"   [{tag}] CONTROL Gram-route sigma_min={float(s_gram[-1]):.4e} vs QR-route "
        f"{float(sdim[-1]):.4e}  (Gram floor {1.49e-8*float(sdim[0]):.3e})")

    # ---- CONTROL 2: Gaussian design of the same shape ----------------------------------------- #
    if A.shape[0] * A.shape[1] < 8e8:
        out["control_gaussian_null"] = gaussian_null(tuple(A.shape), A.device)
        log(f"   [{tag}] NULL gaussian design same shape: cond(A)="
            f"{out['control_gaussian_null']['cond_A']:.3f} "
            f"(MP prediction {out['control_gaussian_null']['cond_A_MP_prediction']:.3f})")

    # ---- 2. the near-kernel -------------------------------------------------------------------- #
    if want_kernel:
        K = []
        rand_pr = []
        gg = torch.Generator(device=Vdim.device).manual_seed(5)
        for _ in range(200):
            r = torch.randn(2 * C, generator=gg, device=Vdim.device, dtype=torch.float64)
            r = r / r.norm()
            rand_pr.append(float(1.0 / (r ** 4).sum()))
        for k in range(n_worst):
            v = Vdim[:, -1 - k]
            vE, vg = v[:C], v[C:]
            w = v ** 2
            pr = float(1.0 / (w ** 2).sum())                       # participation ratio, 1..2C
            # per-CELL weight of this direction (E and gain pooled)
            wc = (vE ** 2 + vg ** 2)
            prc = float(wc.sum() ** 2 / (wc ** 2).sum())
            a, bb = vE.cpu().numpy(), vg.cpu().numpy()
            pear = float(np.corrcoef(a, bb)[0, 1]) if a.std() > 0 and bb.std() > 0 else 0.0
            top = torch.argsort(wc, descending=True)[:5]
            dom = int(top[0])
            K.append({"k_from_bottom": k, "sigma": float(sdim[-1 - k]),
                      "eps": float(epsdim[-1 - k]),
                      "participation_ratio": pr, "participation_ratio_cells": prc,
                      "PR_null_random_unit_vector": float(np.mean(rand_pr)),
                      "PR_over_null": pr / float(np.mean(rand_pr)),
                      "weight_on_E_block": float((vE ** 2).sum()),
                      "weight_on_gain_block": float((vg ** 2).sum()),
                      "pearson_E_vs_gain_within_direction": pear,
                      # what the direction says PHYSICALLY: at the dominant cell, the fractional
                      # change in E that the data cannot tell from a fractional change in gain
                      "dominant_cell": dom + 1,
                      "dominant_cell_vE": float(vE[dom]), "dominant_cell_vgain": float(vg[dom]),
                      "dominant_cell_dlnE_over_dlngain": float(
                          vE[dom] / vg[dom]) if abs(float(vg[dom])) > 1e-14 else float("inf"),
                      "top5_cells": [int(i) + 1 for i in top],
                      "top5_weight": [float(wc[i]) for i in top],
                      "weight_in_top5_cells": float(wc[top].sum())})
        out["near_kernel"] = K
        # how many of the bottom directions are essentially SINGLE-parameter (localised)?
        prs = []
        for k in range(2 * C):
            w = Vdim[:, k] ** 2
            prs.append(float(1.0 / (w ** 2).sum()))
        prs = np.array(prs)                       # index 0 = LARGEST sigma (torch.linalg.svd order)
        nb = max(1, int(0.10 * 2 * C))
        out["localisation"] = {
            "PR_null_random_unit_vector": float(np.mean(rand_pr)),
            "PR_median_bottom10pct": float(np.median(prs[-nb:])),
            "PR_median_top10pct": float(np.median(prs[:nb])),
            "n_bottom10pct_with_PR_below_2": int((prs[-nb:] < 2).sum()),
            "n_bottom10pct_with_PR_below_4": int((prs[-nb:] < 4).sum()),
            "n_directions_in_bottom10pct": nb}
        # correlation between blocks, null: random unit vectors
        nullp = []
        for _ in range(500):
            r = torch.randn(2 * C, generator=gg, device=Vdim.device, dtype=torch.float64)
            nullp.append(float(np.corrcoef(r[:C].cpu().numpy(), r[C:].cpu().numpy())[0, 1]))
        out["pearson_E_gain_null_std"] = float(np.std(nullp))
        # E/gain trade-off measured over the whole near-null SUBSPACE, not one vector
        for frac, nm in ((0.05, "bottom5pct"), (0.25, "bottom25pct")):
            m = max(1, int(frac * 2 * C))
            Vb = Vdim[:, -m:]
            wE = float((Vb[:C] ** 2).sum() / m)
            wg = float((Vb[C:] ** 2).sum() / m)
            out[f"subspace_{nm}"] = {"n_directions": m, "mean_weight_E": wE,
                                     "mean_weight_gain": wg}
        log(f"   [{tag}] worst direction: eps={K[0]['eps']:.3e} PR={K[0]['participation_ratio']:.1f} "
            f"(null {K[0]['PR_null_random_unit_vector']:.1f}) wE={K[0]['weight_on_E_block']:.3f} "
            f"wg={K[0]['weight_on_gain_block']:.3f} r(E,gain)={K[0]['pearson_E_vs_gain_within_direction']:+.3f}")

    # ---- 3. per-cell identifiability ----------------------------------------------------------- #
    W = {}
    for t in (1e-2, 1e-4, 1e-6, 1e-8):
        keep = (epsdim > t)
        W[t] = (Vdim[:, keep] ** 2).sum(dim=1)                 # in [0,1]
    inv2 = 1.0 / (sdim ** 2)
    se = ((Vdim ** 2) * inv2[None, :]).sum(dim=1).sqrt()       # sqrt(diag(G^-1)), dimensionless
    se_tsvd = {}
    for t in (1e-8, 1e-6, 1e-4):
        keep = (epsdim > t)
        se_tsvd[t] = ((Vdim[:, keep] ** 2) * inv2[keep][None, :]).sum(dim=1).sqrt()
    cn = Ad.norm(dim=0)                                        # ||A diag(theta) e_j||
    # THE DECOMPOSITION.  se_j = sqrt(VIF_j)/||a_j||, with VIF_j = ||a_j||^2 (G^-1)_jj = 1/(1-R_j^2)
    # the variance-inflation factor -- INVARIANT under column scaling, so it isolates COLLINEARITY
    # from weak signal.  VIF ~ 1 means "this cell is simply quiet"; VIF >> 1 means "this cell's
    # parameter is aliased onto the others".
    vif = (cn ** 2) * (se ** 2)

    # ---- the E <-> gain trade-off, done properly ----------------------------------------------- #
    # The within-vector Pearson r(vE, vg) is an artefact when the direction is localised (PR ~ 1):
    # 471 points sit at the origin and one does not.  The artefact-free object is the CORRELATION
    # MATRIX of the estimator covariance, P = D^{-1/2} G^{-1} D^{-1/2},  D = diag(G^{-1}).
    # P[c, C+c] is exactly "how much E_c and gain_c are traded off against each other".
    Ginv = (Vdim * inv2[None, :]) @ Vdim.T
    dg = Ginv.diagonal().clamp(min=1e-300).sqrt()
    P = Ginv / dg[:, None] / dg[None, :]
    rho_Eg = P.diagonal(offset=C)                              # [C]
    Pa = P.abs().clone()
    Pa.fill_diagonal_(0.0)
    worst_partner_val, worst_partner_idx = Pa.max(dim=1)
    same_cell_partner = ((worst_partner_idx % C) == torch.arange(2 * C, device=P.device) % C)
    out["E_gain_tradeoff"] = {
        "definition": "P = D^-1/2 G^-1 D^-1/2 ; rho_c = P[E_c, gain_c]",
        "median_abs_rho_Ec_gainc": float(rho_Eg.abs().median()),
        "max_abs_rho_Ec_gainc": float(rho_Eg.abs().max()),
        "n_cells_abs_rho_gt_0.5": int((rho_Eg.abs() > 0.5).sum()),
        "n_cells_abs_rho_gt_0.9": int((rho_Eg.abs() > 0.9).sum()),
        "mean_signed_rho": float(rho_Eg.mean()),
        "median_worst_offdiag_abs_corr_any_partner": float(worst_partner_val.median()),
        "max_worst_offdiag_abs_corr_any_partner": float(worst_partner_val.max()),
        "n_params_with_worst_partner_being_own_cell": int(same_cell_partner.sum()),
        "n_params": int(2 * C)}
    log(f"   [{tag}] E<->gain trade-off: median |rho(E_c,gain_c)| = "
        f"{float(rho_Eg.abs().median()):.3f}, max {float(rho_Eg.abs().max()):.3f}, "
        f"{int((rho_Eg.abs()>0.9).sum())}/{C} cells above 0.9;  worst off-diagonal partner "
        f"correlation median {float(worst_partner_val.median()):.3f}")
    del Ginv, P, Pa

    cov = cell_covariates(sy)
    bnorm = float(getattr(sy, "_bnorm", 1.0))
    percell = {
        "col_norm_E": cn[:C], "col_norm_gain": cn[C:],
        "w1e-4_E": W[1e-4][:C], "w1e-4_gain": W[1e-4][C:],
        "w1e-8_E": W[1e-8][:C], "w1e-8_gain": W[1e-8][C:],
        "se_E": se[:C], "se_gain": se[C:],
        "se_tsvd1e-8_E": se_tsvd[1e-8][:C], "se_tsvd1e-8_gain": se_tsvd[1e-8][C:],
        "vif_E": vif[:C], "vif_gain": vif[C:],
        "rho_E_gain": rho_Eg,
    }
    # WHY the gain block is blind: active_force only exists inside the activation pulse
    # (radius 0.12 about the centre in this spec), so a cell outside it has an exactly zero
    # gain column -- an exact kernel direction, not a sloppy one.
    actc = cov["activity"]
    out["quiet_cells"] = {
        "activity_max": float(actc.max()), "activity_median": float(actc.median()),
        "n_cells_activity_below_1e-6_of_max": int((actc < 1e-6 * actc.max()).sum()),
        "n_cells_activity_below_1e-3_of_max": int((actc < 1e-3 * actc.max()).sum()),
        "n_cells_activity_exactly_zero": int((actc == 0).sum()),
        "C": int(C)}
    log(f"   [{tag}] quiet cells: {int((actc==0).sum())}/{C} have EXACTLY zero active force "
        f"(exact gain kernel); {int((actc < 1e-3*actc.max()).sum())}/{C} below 1e-3 of the max")
    # tolerable relative noise on b for 10% fractional accuracy on that one parameter (eq:amp)
    if bnorm > 0:
        percell["max_rel_noise_for_10pct_E"] = 0.10 / (bnorm * se[:C]).clamp(min=1e-300)
        percell["max_rel_noise_for_10pct_gain"] = 0.10 / (bnorm * se[C:]).clamp(min=1e-300)
    out["per_cell_stats"] = {k: {"min": float(v.min()), "median": float(v.median()),
                                 "max": float(v.max())} for k, v in percell.items()}
    # which failure mode dominates, cell by cell: weak signal or collinearity?
    lo_sig = (cn < cn.median() / 10)
    hi_vif = (vif > 100)
    out["failure_mode"] = {
        "n_params": int(2 * C),
        "median_VIF": float(vif.median()), "max_VIF": float(vif.max()),
        "n_VIF_gt_10": int((vif > 10).sum()), "n_VIF_gt_100": int(hi_vif.sum()),
        "n_colnorm_below_median_over_10": int(lo_sig.sum()),
        "n_both": int((lo_sig & hi_vif).sum()),
        "spread_of_column_norms_max_over_min": float(cn.max() / cn.min().clamp(min=1e-300)),
        "cond_of_equilibrated_A_is_the_collinearity_only_number":
            out["equilibrated"]["cond_A"]}
    log(f"   [{tag}] failure mode: median VIF={float(vif.median()):.2f}  max VIF={float(vif.max()):.3e}"
        f"  #VIF>100: {int(hi_vif.sum())}/{2*C};  column-norm spread "
        f"{float(cn.max()/cn.min().clamp(min=1e-300)):.3e};  collinearity-only cond "
        f"(equilibrated) = {out['equilibrated']['cond_A']:.4e}")

    # regress each per-cell score on the covariates
    reg = {}
    for score in ("se_E", "se_gain", "w1e-4_E", "w1e-4_gain", "col_norm_E", "col_norm_gain"):
        y = np.log10(np.maximum(percell[score].cpu().numpy(), 1e-300)) \
            if score.startswith(("se", "col")) else percell[score].cpu().numpy()
        reg[score] = {}
        for cn_, cv in cov.items():
            if cn_ in ("cx", "cy"):
                continue
            rho, p, sd = spearman_perm(y, cv.cpu().numpy(), n=1000, seed=1)
            reg[score][cn_] = {"spearman": rho, "perm_p": p, "null_sd": sd}
    out["covariate_regression"] = reg
    log(f"   [{tag}] per-cell se_E   median {float(se[:C].median()):.3e}  "
        f"worst {float(se[:C].max()):.3e}  (x{float(se[:C].max()/se[:C].median()):.1f})")
    log(f"   [{tag}] per-cell se_gain median {float(se[C:].median()):.3e}  "
        f"worst {float(se[C:].max()):.3e}")
    best = sorted(reg["se_E"].items(), key=lambda kv: -abs(kv[1]["spearman"]))[:3]
    log(f"   [{tag}] log10 se_E vs covariates: " +
        "  ".join(f"{k}: rho={v['spearman']:+.3f} (p={v['perm_p']:.3f})" for k, v in best))

    return out, {k: v.cpu().numpy() for k, v in percell.items()}, \
        {k: v.cpu().numpy() for k, v in cov.items()}, Vdim, sdim, epsdim


# ============================================================================================== #
#  frames                                                                                        #
# ============================================================================================== #
def advance(sy, n):
    """Advance the frozen state by n FRAMES of the true dynamics, keeping the tick clock right."""
    sy.restore()
    t0 = sy.warmup_frames
    for tick in range(t0, t0 + n):
        sy._outer(tick, gain_cell=sy.gain_true)
        sy.H.sub_dt = sy.dt_sub
        for _ in range(sy.n_sub_per_frame):
            for tok in SUBSTEP_TOKENS:
                sy._tok(tok)
        sy.H.sub_dt = None
    sy.warmup_frames = t0 + n
    sy._snapshot(sy.warmup_frames)


def frame_study(sy, n_frames, stride, log, tag=""):
    """cond vs number of stacked frames.  Each A_f -> R_f (2C x 2C); stacking R's preserves sigma."""
    C = sy.C
    th = sy.theta_true.double()
    Rs, out = [], []
    for f in range(n_frames):
        if f > 0:
            advance(sy, stride)
        t0 = time.time()
        A, a0, _ = sy.assemble(n_sub=1)
        Rf = r_factor(A.double() * th.abs()[None, :])
        del A
        torch.cuda.empty_cache()
        Rs.append(Rf)
        Rc = torch.cat(Rs, 0)
        U, s, Vh = torch.linalg.svd(Rc)
        V = Vh.mH
        se = ((V ** 2) / (s ** 2)[None, :]).sum(dim=1).sqrt()      # sqrt(diag(G_K^-1))
        K = f + 1
        # THE REDUNDANCY NULL.  If frame K carries no information the earlier frames did not,
        # stacking K copies of the SAME constraint multiplies every sigma by sqrt(K) exactly and
        # leaves cond unchanged.  So the informative statistic is sigma_k(K)/(sqrt(K) sigma_k(1)):
        # 1.000 = a redundant repeat, > 1 = genuinely new geometry.
        rec = {"n_frames": K, "tick": sy.warmup_frames,
               "cond_A": float(s[0] / s[-1]), "cond_G": float((s[0] / s[-1]) ** 2),
               "sigma_min": float(s[-1]), "sigma_max": float(s[0]),
               "sigma_min_over_sqrtK_times_sigma_min_1frame":
                   float(s[-1]) / (K ** 0.5 * out[0]["sigma_min"]) if out else 1.0,
               "sigma_max_over_sqrtK_times_sigma_max_1frame":
                   float(s[0]) / (K ** 0.5 * out[0]["sigma_max"]) if out else 1.0,
               "se_median": float(se.median()), "se_max": float(se.max()),
               "se_median_x_sqrtK_over_1frame":
                   float(se.median()) * K ** 0.5 / out[0]["se_median"] if out else 1.0,
               "se_max_x_sqrtK_over_1frame":
                   float(se.max()) * K ** 0.5 / out[0]["se_max"] if out else 1.0,
               "rank_eps>1e-8": int((((s / s[0]) ** 2) > 1e-8).sum().item()),
               "rank_eps>1e-4": int((((s / s[0]) ** 2) > 1e-4).sum().item()),
               "seconds": time.time() - t0}
        out.append(rec)
        log(f"   [{tag}frames] {K:2d} frame(s) (tick {rec['tick']:3d}): cond(A)={rec['cond_A']:.4e}"
            f"  sigma_min={rec['sigma_min']:.4e} (x{rec['sigma_min_over_sqrtK_times_sigma_min_1frame']:.4f}"
            f" vs the redundant-repeat null)  rank(eps>1e-8)={rec['rank_eps>1e-8']}/{2*C}"
            f"  se_max x sqrtK / 1frame = {rec['se_max_x_sqrtK_over_1frame']:.4f}"
            f"  [{rec['seconds']:.1f} s]")
    # NULL: the same stacking with independent Gaussian blocks -- cond should fall like 1/sqrt(nf)
    #       only through the aspect ratio, never below the MP floor.
    g = torch.Generator(device=Rs[0].device).manual_seed(3)
    nulls = []
    for f in range(n_frames):
        Bs = torch.randn((2 * C * (f + 1), 2 * C), generator=g, device=Rs[0].device,
                         dtype=torch.float64)
        s = torch.linalg.svdvals(Bs)
        nulls.append(float(s[0] / s[-1]))
    return {"per_frame": out, "gaussian_stack_null_cond": nulls}


# ============================================================================================== #
#  plots  (Plexus convention: black background, no titles, white top-left labels)                #
# ============================================================================================== #
def _ax(ax, label):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(FG)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_fontsize(10); ax.yaxis.label.set_fontsize(10)
    ax.text(0.0, 1.012, label, transform=ax.transAxes, color=FG, fontsize=11,
            fontweight="bold", va="bottom", ha="left")


def plot_spectrum(res, sdim, epsdim, path, C, eps_model_floor=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=BG)
    e = epsdim.cpu().numpy()
    n = e.size
    ax = axes[0]
    ax.semilogy(np.arange(1, n + 1), e, ".", ms=3, color="#4da6ff")
    for t, c in ((1e-2, "#ff8c42"), (1e-4, "#ffd166")):
        ax.axhline(t, color=c, lw=0.8, ls="--")
        ax.text(n * 0.02, t * 1.6, f"$\\varepsilon$={t:g}", color=c, fontsize=8)
    if eps_model_floor is not None:
        ax.axhline(eps_model_floor, color="#ef476f", lw=1.4, ls="-")
        ax.text(n * 0.02, eps_model_floor * 1.8,
                f"model-error floor  $\\varepsilon$={eps_model_floor:.1e}", color="#ef476f",
                fontsize=8)
    ax.axhline(2.2e-16, color="#888888", lw=0.8, ls=":")
    ax.text(n * 0.02, 3e-16, "$\\varepsilon_{mach}$", color="#888888", fontsize=8)
    ax.set_xlabel("index $k$"); ax.set_ylabel(r"$\varepsilon_k=\lambda_k(G)/\lambda_{max}(G)$")
    _ax(ax, f"a   spectrum of $G=A^TA$, $n={n}$ ($C={C}$)")

    ax = axes[1]
    ax.loglog(np.sort(e), np.arange(1, n + 1) / n, "-", color="#4da6ff", lw=1.6)
    ax.set_xlabel(r"tolerance $\varepsilon$")
    ax.set_ylabel(r"$\Phi(\varepsilon)$ = fraction of directions below")
    _ax(ax, "b   sloppiness curve")

    ax = axes[2]
    bars = [("raw", "#8888ff"), ("dimensionless", "#4da6ff"), ("equilibrated", "#66dd88")]
    for nm, c in bars:
        if nm in res:
            ax.bar(nm, np.log10(res[nm]["cond_A"]), color=c)
            ax.text(nm, np.log10(res[nm]["cond_A"]) + 0.06, f"{res[nm]['cond_A']:.1e}",
                    color=FG, fontsize=8, ha="center", va="bottom")
    if "control_gaussian_null" in res:
        ax.bar("gauss null", np.log10(res["control_gaussian_null"]["cond_A"]), color="#aaaaaa")
        ax.text("gauss null", np.log10(res["control_gaussian_null"]["cond_A"]) + 0.06,
                f"{res['control_gaussian_null']['cond_A']:.2f}", color=FG, fontsize=8,
                ha="center", va="bottom")
    ax.set_ylim(0, max(np.log10(res["raw"]["cond_A"]), np.log10(res["dimensionless"]["cond_A"]))
                * 1.18)
    ax.set_ylabel(r"$\log_{10}\,\mathrm{cond}_2(A)$")
    _ax(ax, "c   conditioning by column scaling  (equilibrated = collinearity only)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def cellmap(ax, cx, cy, val, label, cmap="magma", log=False, size=None):
    v = np.log10(np.maximum(val, 1e-300)) if log else val
    s = 14 if size is None else 6 + 40 * (size / max(size.max(), 1e-9))
    sc = ax.scatter(cx, cy, c=v, s=s, cmap=cmap, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor(FG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    _ax(ax, label)


def plot_kernel(Vdim, epsdim, cov, path, C, n=3):
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.4 * n), facecolor=BG)
    cx, cy = cov["cx"], cov["cy"]
    rng = np.random.default_rng(0)
    for k in range(n):
        v = Vdim[:, -1 - k].cpu().numpy()
        vE, vg = v[:C], v[C:]
        wc = vE ** 2 + vg ** 2
        pr = 1.0 / (v ** 4).sum()
        # 1: WHERE the direction lives, log weight per cell (linear colour hides a PR=1 spike)
        cellmap(axes[k, 0], cx, cy, np.maximum(wc, 1e-20),
                f"{'abc'[k]}   $v_{{n-{k}}}$, $\\varepsilon$={float(epsdim[-1-k]):.1e}, "
                f"PR={pr:.2f}: $\\log_{{10}}$ weight per cell", cmap="inferno", log=True)
        # 2: WHAT it is made of, at the cells that carry it
        ax = axes[k, 1]
        top = np.argsort(-wc)[:12]
        xx = np.arange(len(top))
        ax.bar(xx - 0.2, vE[top], width=0.4, color="#ff6b6b", label="E")
        ax.bar(xx + 0.2, vg[top], width=0.4, color="#4da6ff", label="gain")
        ax.set_xticks(xx)
        ax.set_xticklabels([str(int(t) + 1) for t in top], rotation=60, fontsize=7)
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
        ax.set_xlabel("cell id (12 largest)"); ax.set_ylabel("component of $v$")
        _ax(ax, f"{'abc'[k]}2   composition: "
                f"$\\|v_E\\|^2$={float((vE**2).sum()):.3f}, "
                f"$\\|v_g\\|^2$={float((vg**2).sum()):.3f}")
        # 3: localisation against the random-unit-vector null
        ax = axes[k, 2]
        ax.semilogy(np.sort(np.abs(v))[::-1], "-", color="#4da6ff", lw=1.6, label="$|v_k|$ sorted")
        r = rng.standard_normal(2 * C); r /= np.linalg.norm(r)
        ax.semilogy(np.sort(np.abs(r))[::-1], "--", color="#aaaaaa", lw=1.2,
                    label="random unit vector")
        ax.set_ylim(1e-12, 2)
        ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
        ax.set_xlabel("rank"); ax.set_ylabel("$|$component$|$")
        _ax(ax, f"{'abc'[k]}3   localisation (PR {pr:.2f} vs null {2*C/3.0:.0f})")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)


def plot_percell(pc, cov, path):
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.8), facecolor=BG)
    cx, cy, npart = cov["cx"], cov["cy"], cov["n_particles"]
    cellmap(axes[0, 0], cx, cy, pc["se_E"], "a   $\\log_{10}\\sqrt{(G^{-1})_{cc}}$, E "
            "(higher = worse)", log=True, size=npart)
    cellmap(axes[0, 1], cx, cy, pc["se_gain"], "b   $\\log_{10}\\sqrt{(G^{-1})_{cc}}$, gain",
            log=True, size=npart)
    cellmap(axes[0, 2], cx, cy, pc["rho_E_gain"], r"c   $\rho(E_c,\mathrm{gain}_c)$ from $G^{-1}$",
            cmap="coolwarm", size=npart)
    axes[0, 2].collections[0].set_clim(-1, 1)
    cellmap(axes[0, 3], cx, cy, pc["w1e-4_E"], r"d   $w_c(10^{-4})$, E   (1 = well conditioned)",
            cmap="viridis", size=npart)
    ax = axes[1, 0]
    ax.loglog(cov["dist_to_wall"], pc["se_E"], ".", color="#ff6b6b", ms=6, label="E")
    ax.loglog(cov["dist_to_wall"], pc["se_gain"], ".", color="#4da6ff", ms=6, label="gain")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel("distance of cell centroid to wall")
    ax.set_ylabel(r"$\sqrt{(G^{-1})_{cc}}$")
    _ax(ax, "e   vs distance to wall")
    ax = axes[1, 1]
    ax.loglog(np.maximum(cov["activity"], 1e-30), pc["se_E"], ".", color="#ff6b6b", ms=6)
    ax.loglog(np.maximum(cov["activity"], 1e-30), pc["se_gain"], ".", color="#4da6ff", ms=6)
    ax.set_xlabel("mean |active force| in the cell")
    ax.set_ylabel(r"$\sqrt{(G^{-1})_{cc}}$")
    _ax(ax, "f   vs activity")
    ax = axes[1, 2]
    ax.semilogy(cov["r_from_centre"], pc["se_E"], ".", color="#ff6b6b", ms=6, label="E")
    ax.semilogy(cov["r_from_centre"], pc["se_gain"], ".", color="#4da6ff", ms=6, label="gain")
    ax.axvline(0.12, color="#ffd166", lw=1.0, ls="--")
    ax.text(0.125, pc["se_gain"].max(), "pulse radius", color="#ffd166", fontsize=8, va="top")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel("distance of cell centroid from the activation centre")
    ax.set_ylabel(r"$\sqrt{(G^{-1})_{cc}}$")
    _ax(ax, "g   vs distance from the pulse")
    ax = axes[1, 3]
    ax.hist(np.log10(np.maximum(pc["se_E"], 1e-300)), bins=40, color="#ff6b6b", alpha=0.65,
            label="E")
    ax.hist(np.log10(np.maximum(pc["se_gain"], 1e-300)), bins=40, color="#4da6ff", alpha=0.65,
            label="gain")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel(r"$\log_{10}\sqrt{(G^{-1})_{cc}}$"); ax.set_ylabel("cells")
    _ax(ax, "h   distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BG)
    plt.close(fig)


def plot_frames(fs, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
    nf = [r["n_frames"] for r in fs["per_frame"]]
    ax = axes[0]
    ax.semilogy(nf, [r["cond_A"] for r in fs["per_frame"]], "o-", color="#4da6ff", label="cond(A)")
    ax.semilogy(nf, fs["gaussian_stack_null_cond"], "s--", color="#aaaaaa",
                label="Gaussian stack null")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel("frames stacked"); ax.set_ylabel(r"$\mathrm{cond}_2(A)$")
    _ax(ax, "a   conditioning vs frames")
    ax = axes[1]
    ax.plot(nf, [r["sigma_min_over_sqrtK_times_sigma_min_1frame"] for r in fs["per_frame"]],
            "o-", color="#66dd88", label=r"$\sigma_{min}$")
    ax.plot(nf, [r["sigma_max_over_sqrtK_times_sigma_max_1frame"] for r in fs["per_frame"]],
            "s-", color="#ff8c42", label=r"$\sigma_{max}$")
    ax.axhline(1.0, color="#aaaaaa", ls="--", lw=1.0)
    ax.text(nf[len(nf) // 2], 1.02, "redundant-repeat null", color="#aaaaaa", fontsize=8)
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel("frames stacked")
    ax.set_ylabel(r"$\sigma_k(K)\,/\,[\sqrt{K}\,\sigma_k(1)]$")
    _ax(ax, "b   information beyond a redundant repeat")
    ax = axes[2]
    ax.plot(nf, [r["rank_eps>1e-8"] for r in fs["per_frame"]], "o-", color="#ffd166",
            label=r"$\varepsilon>10^{-8}$")
    ax.plot(nf, [r["rank_eps>1e-4"] for r in fs["per_frame"]], "s-", color="#ef476f",
            label=r"$\varepsilon>10^{-4}$")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=FG, labelcolor=FG)
    ax.set_xlabel("frames stacked"); ax.set_ylabel("effective rank")
    _ax(ax, "c   effective rank")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


# ============================================================================================== #
#  post-hoc, off the saved npz: no solver call                                                   #
# ============================================================================================== #
def partial_spearman(y, x, z):
    """Spearman(y, x) after linearly removing the rank of the control z from both."""
    r = lambda a: (np.argsort(np.argsort(np.asarray(a, float))).astype(float))
    ry, rx, rz = r(y), r(x), r(z)
    rz = np.c_[np.ones_like(rz), rz]
    ey = ry - rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
    ex = rx - rz @ np.linalg.lstsq(rz, rx, rcond=None)[0]
    d = np.sqrt((ey ** 2).sum() * (ex ** 2).sum())
    return float((ey * ex).sum() / d) if d > 0 else 0.0


def post(tag, log):
    """Two things the one-shot report cannot answer:
    (i)  'edge cell' and 'quiet cell' are confounded (a stimulus disc on a square domain), so the
         marginal Spearman cannot separate them -- partial correlations can;
    (ii) the spectrum has a MODEL-ERROR floor, not just a machine floor: A theta = b only holds to
         ||A theta_true - b||/||b|| = 7.8e-5, so a direction whose response is below that is
         indistinguishable from the affine model's own error.  That, not eps_mach, is the cut."""
    z = np.load(os.path.join(HERE, f"spectrum_{tag}_percell.npz"))
    j = json.load(open(os.path.join(HERE, f"spectrum_{tag}.json")))
    cov = {k[4:]: z[k] for k in z.files if k.startswith("cov_")}
    pc = {k[3:]: z[k] for k in z.files if k.startswith("pc_")}
    sig, eps = z["sigma"], z["eps"]
    out = {}

    log(f"\n[{tag}] PARTIAL correlations (rank-linear control removed)")
    par = {}
    for score in ("se_E", "se_gain"):
        y = np.log10(np.maximum(pc[score], 1e-300))
        par[score] = {}
        for x, ctrl in (("dist_to_wall", "r_from_centre"), ("r_from_centre", "dist_to_wall"),
                        ("dist_to_wall", "activity"), ("activity", "dist_to_wall"),
                        ("activity", "r_from_centre"), ("r_from_centre", "activity"),
                        ("spread", "activity"), ("n_particles", "activity")):
            rho = partial_spearman(y, cov[x], cov[ctrl])
            par[score][f"{x} | {ctrl}"] = {"partial_spearman": rho,
                                           "marginal_spearman": spearman(y, cov[x])}
            log(f"   {score}: rho({x} | {ctrl}) = {rho:+.3f}   "
                f"(marginal {spearman(y, cov[x]):+.3f})")
    out["partial_correlations"] = par

    # the model-error floor
    nb = j["assembly"]["norm_b"]
    mrel = j["assembly"]["residual_rel_b"]
    floor = mrel * nb                        # absolute acceleration units the affine model is wrong by
    out["model_error_floor"] = {
        "residual_rel_b": mrel, "norm_b": nb, "absolute_model_error": floor,
        "sigma_max": float(sig[0]),
        "n_directions_detectable_at_100pct_param_change": int((sig > floor).sum()),
        "n_directions_detectable_at_10pct_param_change": int((sig > 10 * floor).sum()),
        "n_directions_detectable_at_1pct_param_change": int((sig > 100 * floor).sum()),
        "n_params": int(sig.size),
        "eps_at_the_model_floor": float((floor / sig[0]) ** 2)}
    log(f"\n[{tag}] MODEL-ERROR FLOOR: the affine identity itself is only good to {mrel:.2e} of "
        f"||b||={nb:.4g}, i.e. {floor:.4g} in acceleration units.")
    log(f"   directions a 100% / 10% / 1% per-cell parameter change can move above that floor: "
        f"{int((sig>floor).sum())} / {int((sig>10*floor).sum())} / {int((sig>100*floor).sum())} "
        f"of {sig.size}")
    log(f"   equivalently the spectrum is only meaningful down to "
        f"eps = {float((floor/sig[0])**2):.3e}, not 1e-16.")

    # the bottom line in cells, not in singular values: how many cells survive a given noise level?
    C = pc["se_E"].size
    yields = {}
    for eta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        okE = int((pc["max_rel_noise_for_10pct_E"] > eta).sum())
        okg = int((pc["max_rel_noise_for_10pct_gain"] > eta).sum())
        both = int(((pc["max_rel_noise_for_10pct_E"] > eta) &
                    (pc["max_rel_noise_for_10pct_gain"] > eta)).sum())
        yields[f"{eta:g}"] = {"cells_E_to_10pct": okE, "cells_gain_to_10pct": okg,
                              "cells_both_to_10pct": both, "C": C}
        log(f"   at {eta:g} relative noise on b: {okE}/{C} cells give E to 10%, "
            f"{okg}/{C} give gain to 10%, {both}/{C} give both")
    out["cell_yield_vs_noise"] = yields
    out["caveat"] = ("this ignores the affine model's own 7.8e-5 error, which acts like a noise "
                     "floor eta >= 7.8e-5 that cannot be averaged away by taking more frames of "
                     "the same state")
    return out


# ============================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--mode", default="full", choices=["full", "inset"])
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--interior", action="store_true",
                    help="also analyse A restricted to interior particles (where A.theta=b is exact)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--post", default="", help="post-hoc pass over an existing tag's npz/json")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    if args.post:
        out = post(args.post, log)
        p = os.path.join(HERE, f"spectrum_{args.post}_post.json")
        json.dump(out, open(p, "w"), indent=1, default=str)
        print(f"wrote {p}")
        return

    tag = args.tag or ("real" if args.real else f"C{args.cells}_{args.mode}")
    RES = {"argv": vars(args)}
    torch.manual_seed(0)

    with torch.no_grad():
        sy = System(device=args.device, n_cells=args.cells, per_parent=args.per_parent,
                    n_grid=args.n_grid, warmup=args.warmup, dtype="float64", mode=args.mode,
                    real=args.real)
        log(f"[{tag}] C={sy.C} cells, Np={sy.Np} particles, grid {sy.g.nx}^2, "
            f"warmup {sy.warmup_frames} frames")

        t0 = time.time()
        A, a0, t_asm = sy.assemble(n_sub=1)
        b = sy.a_of_theta(sy.theta_true, n_sub=1) - a0
        sy._bnorm = float(b.norm())
        resid = float((A @ sy.theta_true - b).norm() / b.norm())
        RES["assembly"] = {"A_shape": list(A.shape), "seconds": t_asm,
                           "norm_b": sy._bnorm, "residual_rel_b": resid}
        log(f"   A {tuple(A.shape)} assembled in {t_asm:.1f} s; ||A theta_true - b||/||b|| "
            f"= {resid:.3e}  (task C's number, reproduced)")

        Ad = A.double()
        out, pc, cov, Vdim, sdim, epsdim = analyse(Ad, sy, tag, log)
        RES["full"] = out
        np.savez(os.path.join(HERE, f"spectrum_{tag}_percell.npz"),
                 **{f"pc_{k}": v for k, v in pc.items()},
                 **{f"cov_{k}": v for k, v in cov.items()},
                 sigma=sdim.cpu().numpy(), eps=epsdim.cpu().numpy(),
                 V_bottom32=Vdim[:, -32:].cpu().numpy())
        eps_floor = (resid * sy._bnorm / float(sdim[0])) ** 2
        RES["eps_model_error_floor"] = eps_floor
        plot_spectrum(out, sdim, epsdim, os.path.join(HERE, f"spectrum_{tag}.png"), sy.C,
                      eps_model_floor=eps_floor)
        plot_kernel(Vdim, epsdim, cov, os.path.join(HERE, f"kernel_{tag}.png"), sy.C)
        plot_percell(pc, cov, os.path.join(HERE, f"percell_{tag}.png"))
        log(f"   wrote spectrum_{tag}.png kernel_{tag}.png percell_{tag}.png")

        # ---- the same spectrum on the rows where the affine model is EXACT -------------------- #
        if args.interior:
            pmask = sy.interior_particle_mask(4.0)
            fm = sy.flat_mask(pmask)
            # a cell with NO interior particle has two exactly-zero columns -> an EXACT kernel.
            nP = torch.zeros(sy.C + 1, device=pmask.device).index_add_(
                0, sy.cid, torch.ones(sy.Np, device=pmask.device))
            nI = torch.zeros(sy.C + 1, device=pmask.device).index_add_(
                0, sy.cid, pmask.double())
            frac = (nI / nP.clamp(min=1))[1:]
            RES["interior_trivial_kernel"] = {
                "cells_with_zero_interior_particles": int((frac == 0).sum()),
                "cells_below_1pct_interior": int((frac < 0.01).sum()),
                "cells_below_10pct_interior": int((frac < 0.10).sum()),
                "C": sy.C}
            log(f"   [interior] cells with NO interior particle (exact kernel, 2 dims each): "
                f"{int((frac==0).sum())}/{sy.C};  below 10% interior: {int((frac<0.10).sum())}")
            out_i, pc_i, cov_i, Vi, si, ei = analyse(Ad[fm], sy, tag + "/interior", log,
                                                     want_kernel=True)
            RES["interior_rows"] = out_i
            plot_spectrum(out_i, si, ei, os.path.join(HERE, f"spectrum_{tag}_interior.png"), sy.C)

        del A, Ad
        torch.cuda.empty_cache()

        # ---- 4. frames -------------------------------------------------------------------------- #
        if args.frames > 1:
            RES["frame_study"] = frame_study(sy, args.frames, args.stride, log, tag=tag + " ")
            plot_frames(RES["frame_study"], os.path.join(HERE, f"frames_{tag}.png"))
            log(f"   wrote frames_{tag}.png")
        log(f"   total {time.time()-t0:.1f} s")

    out_path = os.path.join(HERE, f"spectrum_{tag}.json")
    json.dump(RES, open(out_path, "w"), indent=1, default=str)
    open(os.path.join(HERE, f"spectrum_{tag}.log"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

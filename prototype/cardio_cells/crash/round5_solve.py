"""round5_solve.py -- ROUND 5, stage 2.  Every solver variant, from the saved normal equations.

THE ONE CHANGE ROUND 5 CARRIES (round 4's diagnosis)
====================================================================================================
The solve is unconstrained, so a handful of cells come back negative or 16x too large, and that
TAIL -- invisible to med|dE/E| -- is what destroys the rollout (n_negE vs R^2, Spearman -0.893).
Replace the unconstrained / eigen-truncated solve with a BOX-CONSTRAINED solve of the SAME normal
equations:

    min_z  0.5 z' G z - r' z      s.t.   lo <= z*s <= hi

with the box read off the estimate itself, no truth: per block, m = median(naive[naive > 0]) and
(lo, hi) = (0.2 m, 5 m).  One prior only -- per-cell moduli within a factor 25 of each other; the
planted spread is 5.4x, so the box is loose by 4.6x.  Solved by monotone FISTA (accelerated
projected gradient with a restart), step 1/lambda_max(G), warm start = clip(naive).

The Gram is either G0 (naive) or Gc = G0 - Sigma (the errors-in-variables correction of round 4),
so the constraint and the correction are separated: four cells of a 2x2 design.

Nothing here touches theta_true; `score` is applied afterwards for reporting only.

usage: PYTHONPATH=/workspace/Plexus/src python round5_solve.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, HERE):
    sys.path.insert(0, _p)


# --------------------------------------------------------------------------------------------- #
def solve_box(G, r, s, lo, hi, z0=None, iters=4000, tol=1e-13):
    """Monotone FISTA on the box-constrained QP.  Returns (theta, info).

    G may be indefinite (the EIV correction subtracts Sigma), in which case the problem is not
    convex; the box is compact so projected gradient still converges to a stationary point, and the
    monotone restart keeps the objective non-increasing.  The objective, the projected-gradient
    (KKT) residual and the number of active bounds are returned so a failure is visible.
    """
    Gs = (G + G.T) / 2
    L = float(torch.linalg.eigvalsh(Gs).max())
    step = 1.0 / max(L, 1e-300)
    zl, zh = lo / s, hi / s
    z = torch.zeros_like(r) if z0 is None else z0.clone()
    z = torch.clamp(z, zl, zh)

    def obj(v):
        return float(0.5 * v @ (Gs @ v) - r @ v)

    y, t, f = z.clone(), 1.0, obj(z)
    n_it = iters
    for k in range(iters):
        g = Gs @ y - r
        zn = torch.clamp(y - step * g, zl, zh)
        fn = obj(zn)
        if fn > f:                                  # monotone restart
            zn = torch.clamp(z - step * (Gs @ z - r), zl, zh)
            fn = obj(zn)
            t = 1.0
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = zn + ((t - 1.0) / tn) * (zn - z)
        d = float((zn - z).norm())
        z, t, f = zn, tn, fn
        if d <= tol * max(1.0, float(z.norm())):
            n_it = k + 1
            break
    gfin = Gs @ z - r
    kkt = torch.where((z <= zl + 1e-14) & (gfin > 0), torch.zeros_like(gfin),
                      torch.where((z >= zh - 1e-14) & (gfin < 0), torch.zeros_like(gfin), gfin))
    n_act = int(((z <= zl * (1 + 1e-9)) | (z >= zh * (1 - 1e-9))).sum())
    return z * s, {"iters": n_it, "objective": f, "kkt_resid": float(kkt.norm()),
                   "grad_norm": float(gfin.norm()), "n_active_bounds": n_act,
                   "lam_max": L, "converged": bool(n_it < iters)}


def snr_trunc(G0, Sig, Gc, rc, s, tau=0.0):
    """Round 4's repair: the generalised problem Gc v = lambda Sigma v; lambda IS the SNR."""
    w, U = torch.linalg.eigh(Sig)
    sig2 = float(w.abs().max())
    if sig2 <= 0:
        return torch.linalg.solve(Gc, rc) * s, {"rank": Gc.shape[0], "n_snr_gt1": Gc.shape[0]}
    wf = w.clamp(min=1e-2 * sig2)
    Lh = U @ torch.diag(wf.rsqrt()) @ U.T
    M = Lh @ Gc @ Lh
    lam, V = torch.linalg.eigh((M + M.T) / 2)
    Vw = Lh @ V
    keep = lam > tau
    if int(keep.sum()) == 0:
        return torch.zeros_like(rc), {"rank": 0, "n_snr_gt1": 0}
    Vk = Vw[:, keep]
    return (Vk @ ((Vk.T @ rc) / lam[keep])) * s, {
        "rank": int(keep.sum()), "n_snr_gt1": int((lam > 1).sum()),
        "sigma_spec": sig2, "snr_max": float(lam.max())}


def clipbox(t, C, lo, hi):
    o = t.clone()
    o[:C] = o[:C].clamp(lo[0], hi[0])
    o[C:] = o[C:].clamp(lo[1], hi[1])
    return o


def pstats(t, th, C):
    t = np.asarray(t, float)
    th = np.asarray(th, float)
    E, g, Eh, gh = th[:C], th[C:], t[:C], t[C:]
    rE = np.abs(Eh - E) / E
    rg = np.abs(gh - g) / g
    kE = float(Eh @ E / (Eh @ Eh)) if float(Eh @ Eh) > 0 else float("nan")
    kg = float(gh @ g / (gh @ gh)) if float(gh @ gh) > 0 else float("nan")
    tg = np.concatenate([kE * Eh, kg * gh])
    return {"med_E": float(np.median(rE)), "p90_E": float(np.percentile(rE, 90)),
            "max_E": float(rE.max()), "med_gain": float(np.median(rg)),
            "n_cells_relE_gt5": int((rE > 5).sum()), "n_negE": int((Eh < 0).sum()),
            "n_neg_gain": int((gh < 0).sum()),
            "rel_l2": float(np.linalg.norm(t - th) / np.linalg.norm(th)),
            "corr_E": float(np.corrcoef(Eh, E)[0, 1]),
            "k_E_opt": kE, "k_g_opt": kg,
            "rel_l2_gauge_opt": float(np.linalg.norm(tg - th) / np.linalg.norm(th)),
            "med_E_after_rescale": float(np.median(np.abs(kE * Eh - E) / E)),
            "mean_ratio_E": float(Eh.mean() / E.mean()),
            "mean_E": float(Eh.mean()), "mean_gain": float(gh.mean())}


# --------------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="round5_solve")
    ap.add_argument("--lo-f", type=float, default=0.2)
    ap.add_argument("--hi-f", type=float, default=5.0)
    ap.add_argument("--box-iters", type=int, default=4000)
    ap.add_argument("--out", default="theta_round5")
    a = ap.parse_args()
    dev = torch.device(a.device)
    R = {"box": [a.lo_f, a.hi_f], "fits": {}}
    thetas = {}
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    files = sorted(glob.glob(os.path.join(HERE, "round5_norm_*.npz")))
    log(f"[solve] {len(files)} normal-equation files")
    log(f"    {'fit':<26s} {'T':>2s} {'solver':<16s} {'medE':>7s} {'p90':>7s} {'maxE':>8s} "
        f"{'>5x':>4s} {'neg':>4s} {'relL2':>7s} {'relL2g':>7s} {'medE_re':>8s} {'corr':>6s} "
        f"{'mr':>7s}")
    for fp in files:
        z = np.load(fp)
        name = os.path.basename(fp)[:-4]
        th = torch.as_tensor(z["theta_true"], device=dev, dtype=torch.float64)
        s = torch.as_tensor(z["s"], device=dev, dtype=torch.float64)
        C = th.numel() // 2
        nfr = sum(1 for k in z.files if k.startswith("G") and not k.startswith("Gm"))
        R["fits"][name] = {"n_frames": nfr, "T": {}}
        for T in (1, 2, 4, 8):
            if T > nfr:
                continue
            G0 = sum(torch.as_tensor(z[f"G{k}"], device=dev, dtype=torch.float64)
                     for k in range(T))
            r0 = sum(torch.as_tensor(z[f"r{k}"], device=dev, dtype=torch.float64)
                     for k in range(T))
            Gb = sum(torch.as_tensor(z[f"Gm{k}"], device=dev, dtype=torch.float64)
                     for k in range(T))
            rb = sum(torch.as_tensor(z[f"rm{k}"], device=dev, dtype=torch.float64)
                     for k in range(T))
            has_mc = float(Gb.abs().max()) > 0
            Sig = (Gb - G0) if has_mc else torch.zeros_like(G0)
            Gc, rc = G0 - Sig, r0 - (rb - r0 if has_mc else torch.zeros_like(r0))

            out = {}
            try:
                out["naive"] = torch.linalg.solve(G0, r0) * s
            except Exception:
                out["naive"] = torch.linalg.lstsq(G0, r0.unsqueeze(1)).solution.squeeze(1) * s
            if has_mc:
                out["eiv_snr0"], ex_snr = snr_trunc(G0, Sig, Gc, rc, s, tau=0.0)
            else:
                ex_snr = {}

            # --- the box, read off the NAIVE estimate; identical for every constrained variant ---
            nv = out["naive"]
            mE = float(nv[:C][nv[:C] > 0].median()) if int((nv[:C] > 0).sum()) else \
                float(nv[:C].abs().median())
            mg = float(nv[C:][nv[C:] > 0].median()) if int((nv[C:] > 0).sum()) else \
                float(nv[C:].abs().median())
            lo = torch.cat([torch.full((C,), a.lo_f * mE, device=dev, dtype=torch.float64),
                            torch.full((C,), a.lo_f * mg, device=dev, dtype=torch.float64)])
            hi = torch.cat([torch.full((C,), a.hi_f * mE, device=dev, dtype=torch.float64),
                            torch.full((C,), a.hi_f * mg, device=dev, dtype=torch.float64)])
            info = {}
            out["naive_box"], info["naive_box"] = solve_box(
                G0, r0, s, lo, hi, z0=torch.clamp(nv, lo, hi) / s, iters=a.box_iters)
            out["naive_clip"] = torch.clamp(nv, lo, hi)
            if has_mc:
                out["eiv_box"], info["eiv_box"] = solve_box(
                    Gc, rc, s, lo, hi, z0=torch.clamp(out["eiv_snr0"], lo, hi) / s,
                    iters=a.box_iters)
                out["eiv_clip"] = torch.clamp(out["eiv_snr0"], lo, hi)

            row = {"box_bounds": {"E": [a.lo_f * mE, a.hi_f * mE],
                                  "gain": [a.lo_f * mg, a.hi_f * mg]},
                   "snr": ex_snr, "box_info": info,
                   "cond_G0": float(torch.linalg.eigvalsh(G0).max()
                                    / torch.linalg.eigvalsh(G0).clamp(min=1e-300).min()),
                   "min_eig_G0": float(torch.linalg.eigvalsh(G0).min()),
                   "min_eig_Gc": float(torch.linalg.eigvalsh(Gc).min()),
                   "sigma_fro_over_G_fro": float(Sig.norm() / G0.norm()),
                   "solvers": {}}
            for k, t in out.items():
                row["solvers"][k] = pstats(t.cpu().numpy(), th.cpu().numpy(), C)
                thetas[f"{name}|T{T}|{k}"] = t.cpu().numpy()
                p = row["solvers"][k]
                log(f"    {name:<26s} {T:>2d} {k:<16s} {p['med_E']:>7.4f} {p['p90_E']:>7.3f} "
                    f"{p['max_E']:>8.3f} {p['n_cells_relE_gt5']:>4d} {p['n_negE']:>4d} "
                    f"{p['rel_l2']:>7.3f} {p['rel_l2_gauge_opt']:>7.3f} "
                    f"{p['med_E_after_rescale']:>8.4f} {p['corr_E']:>6.3f} "
                    f"{p['mean_ratio_E']:>7.3f}")
            R["fits"][name]["T"][f"T{T}"] = row
    thetas["theta_true"] = th.cpu().numpy()
    np.savez(os.path.join(HERE, f"{a.out}.npz"), **thetas)
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    R["box_iters"] = a.box_iters
    log(f"\nwrote {a.tag}.json and {a.out}.npz ({len(thetas)} vectors, box iters {a.box_iters})")


if __name__ == "__main__":
    main()

"""p1e_alias.py -- THE CONFOUND TEST nobody in P1 ran.

Question: if a per-cell GAIN is recovered while per-cell E is refused (or wrong), is the recovered
gain actually gain, or is it a relabelled copy of the per-cell STIFFNESS pattern?

Three read-outs, all on the honest local Jacobian at base = theta_true, clean (true) F, tick 165
(the global argmax of ||act0||, i.e. the most favourable phase there is):

  1. ALIAS OPERATOR.  dE = E_true - median(E_true)  is the per-cell stiffness pattern with the
     global level removed.  g_alias = argmin_g || A_g g - A_E dE ||.  If ||A_g g_alias|| is a large
     fraction of ||A_E dE||, the gain block can imitate the stiffness pattern, and any gain map
     fitted on a sheet whose stiffness is heterogeneous is partly a stiffness map.
     Reported in relative units: g_alias / gain_true against dE / E_true.

  2. PER-CELL COLUMN GEOMETRY.  cos( A[:,c], A[:,C+c] ) -- how parallel one cell's E column and
     its own gain column are.  Parallel columns mean only ONE number per cell is readable.

  3. THE IDENTIFIABLE COMBINATION.  With b-noise sigma (iid), cov of the RELATIVE parameter error
     is sigma^2 (D^T D)^-1 with D = A diag(theta_true).  For each cell take the 2x2 sub-block at
     (c, C+c): its eigenvectors are the best- and worst-determined per-cell directions in the
     (dE/E, dg/g) plane, and its eigenvalues give the detectable size of each.  This asks whether
     a reparameterisation exists that IS identifiable, and what it would be called.

Writes p1e_alias.json / p1e_alias.log.  Modifies nothing.
usage: PYTHONPATH=/workspace/Plexus/src python p1e_alias.py --device cuda:0
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

from finject import y_of, record_substeps                             # noqa: E402
from freal_derivedF import plant_and_warm_x0, PX                      # noqa: E402
from round5_fit import SIGMA_X                                        # noqa: E402
import metrics as MET                                                 # noqa: E402


def assemble_at(sy, n_sub, base, sE=100.0, sg=1.0):
    t0 = time.time()
    y0 = y_of(sy, base, n_sub, None, None)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = base.clone()
        e[j] = e[j] + s
        A[:, j] = (y_of(sy, e, n_sub, None, None) - y0) / s
    torch.cuda.synchronize()
    return A, y0, time.time() - t0


def pear(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))


def spear(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pear(ra, rb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="p1e_alias")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--nsub", default="1,10")
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                           n_grid=a.n_grid, warmup=a.t0, window=150, dtype="float64",
                           mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"args": vars(a), "config": vars(args), "sigma_x_world": SIGMA_X, "px_world": PX}
    t_start = time.time()

    with torch.no_grad():
        sy, _REF = plant_and_warm_x0(args, log)
        C = sy.C
        th = sy.theta_true.double()
        E_t, g_t = th[:C], th[C:]
        x0 = sy.x0.clone()
        band = 0.06 / MET.SHEET_SPAN
        interior = ~((x0[:, 0] < band) | (x0[:, 0] > 1 - band)
                     | (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        int_flat = interior[:, None].expand(-1, 2).reshape(-1)
        log(f"[state] C={C} Np={sy.Np} interior={int(interior.sum())} "
            f"||act0||={float(sy.act0.norm()):.5g}  E in [{float(E_t.min()):.2f},"
            f"{float(E_t.max()):.2f}] med {float(E_t.median()):.2f}; gain in "
            f"[{float(g_t.min()):.4f},{float(g_t.max()):.4f}] mean {float(g_t.mean()):.4f}")
        R["n_interior"] = int(interior.sum())
        R["Np"] = sy.Np
        R["C"] = C
        R["act0_norm"] = float(sy.act0.norm())

        for ns in [int(s) for s in a.nsub.split(",")]:
            A, y0, dt_as = assemble_at(sy, ns, th.to(sy.dtype))
            log(f"\n===== n_sub = {ns} (assembly {dt_as:.1f} s) =====")
            Ai = A[int_flat].double()                    # interior rows only
            Ae, Ag = Ai[:, :C], Ai[:, C:]
            ne0, ng0 = Ae.norm(dim=0), Ag.norm(dim=0)
            dead = (ne0 == 0) | (ng0 == 0)               # a cell with no interior footprint at all
            alive = ~dead
            rec = {"assembly_s": dt_as,
                   "norm_A_E": float(Ae.norm()), "norm_A_gain": float(Ag.norm()),
                   "n_dead_cells_interior": int(dead.sum()),
                   "n_zero_E_cols": int((ne0 == 0).sum()),
                   "n_zero_gain_cols": int((ng0 == 0).sum())}
            log(f"  [rank] cells with an all-zero interior column: {int(dead.sum())}/{C} "
                f"(E {int((ne0==0).sum())}, gain {int((ng0==0).sum())})")

            # ---------------- 1. the alias operator ------------------------------------------- #
            dE = E_t - E_t.median()
            sig_E = Ae @ dE                                   # the stiffness pattern's signal
            Gg = Ag.T @ Ag
            g_alias = torch.linalg.pinv(Gg, rtol=1e-12) @ (Ag.T @ sig_E)
            fitted = Ag @ g_alias
            absorbed = float(fitted.norm() / sig_E.norm())
            resid = float((sig_E - fitted).norm() / sig_E.norm())
            rel_alias = g_alias / g_t                          # fractional gain error it induces
            rel_dE = dE / E_t
            al = {"absorbed_fraction_of_stiffness_signal": absorbed,
                  "residual_fraction": resid,
                  "corr_g_alias_vs_dE": pear(g_alias, dE),
                  "spearman_g_alias_vs_dE": spear(g_alias, dE),
                  "corr_rel": pear(rel_alias, rel_dE),
                  "med_abs_rel_gain_error_induced": float(rel_alias.abs().median()),
                  "p90_abs_rel_gain_error_induced": float(rel_alias.abs().quantile(0.9)),
                  "std_g_alias_over_std_gain_true": float(g_alias.std() / g_t.std()),
                  "std_rel_alias": float(rel_alias.std()),
                  "std_rel_gain_true": float((g_t / g_t.mean() - 1).std()),
                  "planted_rel_E_spread_std": float(rel_dE.std())}
            rec["alias_E_into_gain"] = al
            log(f"[alias E->gain] the gain block absorbs {absorbed*100:.1f}% (in norm) of the "
                f"per-cell STIFFNESS signal A_E dE; residual {resid*100:.1f}%")
            log(f"                corr(g_alias, dE) = {al['corr_g_alias_vs_dE']:+.4f}  "
                f"spearman {al['spearman_g_alias_vs_dE']:+.4f}")
            log(f"                induced spurious gain: med |dg/g| "
                f"{al['med_abs_rel_gain_error_induced']:.4f}, p90 "
                f"{al['p90_abs_rel_gain_error_induced']:.4f}; std(g_alias)/std(gain_true) "
                f"{al['std_g_alias_over_std_gain_true']:.3f}")

            # reverse: can the E block imitate a per-cell gain pattern?
            dg = g_t - g_t.mean()
            sig_g = Ag @ dg
            Ge = Ae.T @ Ae
            E_alias = torch.linalg.pinv(Ge, rtol=1e-12) @ (Ae.T @ sig_g)
            fit2 = Ae @ E_alias
            rec["alias_gain_into_E"] = {
                "absorbed_fraction_of_gain_signal": float(fit2.norm() / sig_g.norm()),
                "residual_fraction": float((sig_g - fit2).norm() / sig_g.norm()),
                "corr_E_alias_vs_dg": pear(E_alias, dg),
                "med_abs_rel_E_error_induced": float((E_alias / E_t).abs().median())}
            log(f"[alias gain->E] the E block absorbs "
                f"{rec['alias_gain_into_E']['absorbed_fraction_of_gain_signal']*100:.1f}% of the "
                f"per-cell GAIN signal; corr {rec['alias_gain_into_E']['corr_E_alias_vs_dg']:+.4f}")

            # ---------------- 2. per-cell column geometry -------------------------------------- #
            ne, ng = Ae.norm(dim=0), Ag.norm(dim=0)
            cos_c = torch.where((ne > 0) & (ng > 0),
                                (Ae * Ag).sum(0) / (ne * ng).clamp(min=1e-300),
                                torch.full_like(ne, float("nan")))
            cc = cos_c[~torch.isnan(cos_c)]
            rec["percell_cos_E_gain"] = {
                "n_valid": int(cc.numel()), "n_zero_gain_cols": int((ng == 0).sum()),
                "median_abs": float(cc.abs().median()), "p90_abs": float(cc.abs().quantile(0.9)),
                "max_abs": float(cc.abs().max()), "min_abs": float(cc.abs().min()),
                "frac_abs_gt_0.9": float((cc.abs() > 0.9).double().mean()),
                "frac_abs_gt_0.99": float((cc.abs() > 0.99).double().mean())}
            log(f"[geometry] |cos(A_E[:,c], A_g[:,c])| per cell: med "
                f"{rec['percell_cos_E_gain']['median_abs']:.4f}, p90 "
                f"{rec['percell_cos_E_gain']['p90_abs']:.4f}, max "
                f"{rec['percell_cos_E_gain']['max_abs']:.4f}; "
                f"{rec['percell_cos_E_gain']['frac_abs_gt_0.9']*100:.0f}% above 0.90")

            # ---------------- 3. the identifiable per-cell combination -------------------------- #
            keep = torch.nonzero(alive).flatten()
            idx = torch.cat([keep, keep + C])
            D = Ai[:, idx] * th[idx][None, :]          # dimensionless design (relative parameters)
            G = (D.T @ D).cpu()
            sv = torch.linalg.svdvals(D.cpu())
            rec["svd"] = {"n_cols": int(D.shape[1]),
                          "cond": float(sv[0] / sv[-1]), "s_max": float(sv[0]),
                          "s_min": float(sv[-1]),
                          "s": [float(v) for v in sv[::10]]}
            try:
                Ginv = torch.linalg.inv(G)
            except Exception as e:
                Ginv = torch.linalg.pinv(G)
                log(f"  [warn] inv failed ({e}); used pinv")
            Ck = int(keep.numel())
            sig_b = math.sqrt(2.0) * SIGMA_X          # position noise at both frame boundaries
            best_dir, best_sd, worst_sd, ratio = [], [], [], []
            for c in range(Ck):
                S = Ginv[[c, Ck + c]][:, [c, Ck + c]]
                ev, V = torch.linalg.eigh(S)          # ascending: ev[0] = best determined
                sd = sig_b * torch.sqrt(ev.clamp(min=0))
                v = V[:, 0]
                if float(v[0]) < 0:
                    v = -v
                best_dir.append([float(v[0]), float(v[1])])
                best_sd.append(float(sd[0]))
                worst_sd.append(float(sd[1]))
                ratio.append(float(sd[1] / max(sd[0], 1e-300)))
            bd = torch.tensor(best_dir)
            ang = torch.atan2(bd[:, 1], bd[:, 0]) * 180.0 / math.pi
            bs, ws, rt = (torch.tensor(best_sd), torch.tensor(worst_sd), torch.tensor(ratio))
            rec["percell_conditional"] = {
                "sigma_b_world": sig_b,
                "note": ("std of the RELATIVE per-cell error at iid position noise sigma_x="
                         f"{SIGMA_X:.3e} world = 0.0409 px, ALL {int(interior.sum())} interior "
                         "particles independent -- optimistic; the recording has ~22.8 effective "
                         "samples per cell against 100 particles here"),
                "best_direction_angle_deg_med": float(ang.median()),
                "best_direction_angle_deg_p10": float(ang.quantile(0.1)),
                "best_direction_angle_deg_p90": float(ang.quantile(0.9)),
                "best_dir_median_[dE/E, dg/g]": [float(bd[:, 0].median()),
                                                 float(bd[:, 1].median())],
                "sd_best_med": float(bs.median()), "sd_best_min": float(bs.min()),
                "sd_worst_med": float(ws.median()), "sd_worst_min": float(ws.min()),
                "anisotropy_ratio_med": float(rt.median()),
                "anisotropy_ratio_max": float(rt.max()),
                "n_cells_scored": Ck, "n_cells_excluded_dead": int(dead.sum()),
                "n_cells_sd_best_under_0.10": int((bs < 0.10).sum()),
                "n_cells_sd_worst_under_0.10": int((ws < 0.10).sum()),
                "n_cells_sd_best_under_0.30": int((bs < 0.30).sum()),
                "n_cells_sd_worst_under_0.30": int((ws < 0.30).sum())}
            # marginal per-parameter error bars, same noise
            d = torch.diagonal(Ginv)
            sd_all = sig_b * torch.sqrt(d.clamp(min=0))
            rec["marginal_rel_sd"] = {
                "E_med": float(sd_all[:Ck].median()), "E_min": float(sd_all[:Ck].min()),
                "gain_med": float(sd_all[Ck:].median()), "gain_min": float(sd_all[Ck:].min()),
                "n_E_under_0.10": int((sd_all[:Ck] < 0.10).sum()),
                "n_gain_under_0.10": int((sd_all[Ck:] < 0.10).sum()),
                "n_gain_under_0.30": int((sd_all[Ck:] < 0.30).sum())}
            log(f"[conditional] best per-cell direction (dE/E, dg/g), median "
                f"[{rec['percell_conditional']['best_dir_median_[dE/E, dg/g]'][0]:+.3f}, "
                f"{rec['percell_conditional']['best_dir_median_[dE/E, dg/g]'][1]:+.3f}], angle "
                f"{float(ang.median()):+.1f} deg (0 = pure E, 90 = pure gain)")
            log(f"              rel sd along it: med {float(bs.median()):.4g}; along the worst "
                f"direction med {float(ws.median()):.4g}; anisotropy med "
                f"{float(rt.median()):.3g}")
            log(f"              cells with a <=10% readable combination: "
                f"{rec['percell_conditional']['n_cells_sd_best_under_0.10']}/{Ck}; both "
                f"directions <=10%: "
                f"{rec['percell_conditional']['n_cells_sd_worst_under_0.10']}/{Ck}")
            log(f"[marginal]    rel sd per cell: E med {float(sd_all[:C].median()):.4g}, gain med "
                f"{float(sd_all[C:].median()):.4g}; gains under 10%: "
                f"{rec['marginal_rel_sd']['n_gain_under_0.10']}/{Ck}, under 30%: "
                f"{rec['marginal_rel_sd']['n_gain_under_0.30']}/{Ck}")

            np.savez(os.path.join(HERE, f"{a.tag}_n{ns}.npz"),
                     cos_c=cos_c.cpu().numpy(), g_alias=g_alias.cpu().numpy(),
                     dE=dE.cpu().numpy(), E_true=E_t.cpu().numpy(), gain_true=g_t.cpu().numpy(),
                     best_dir=bd.numpy(), sd_best=bs.numpy(), sd_worst=ws.numpy(),
                     sd_marginal=sd_all.numpy(), svals=sv.numpy(), alive=alive.cpu().numpy())
            R[f"nsub{ns}"] = rec
            del A, Ai, Ae, Ag, D, G, Ginv
            torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    with open(os.path.join(HERE, f"{a.tag}.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(os.path.join(HERE, f"{a.tag}.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\n[done] {R['wall_seconds']:.1f} s -> {a.tag}.json / .log")


if __name__ == "__main__":
    main()

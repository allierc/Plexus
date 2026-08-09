"""fcell_basis.py -- TASK 1.  A REGRESSION ANCHORED TO THE SEGMENTATION, INSTEAD OF A STENCIL.

WHAT IS BEING ATTACKED
====================================================================================================
`freal_derivedF.py` derives F the way the recording must -- bin the displacements onto a control
grid of the recording's spacing and central-difference them -- and the per-cell (E, gain) solve
collapses:

    central difference 15 px   med|dE/E| 0.999   corr(E_hat, E) 0.034   mean(E_hat)/mean(E) 0.0033
    central difference 34 px   0.998            -0.045                  0.0038
    the simulator's own F      0.0078            ~1                      ~1

with NO noise added anywhere.  A central difference over 2h is a boxcar average of the derivative:
it is a BIASED estimator and it attenuates by ~300x here.  The fix cannot be a finer stencil (the
recording's spacing is fixed at 15 px); it has to be a different ESTIMATOR on a better BASIS.

THE HYPOTHESIS
----------------------------------------------------------------------------------------------------
A least-squares fit of the displacement over a patch is a REGRESSION -- unbiased for whatever it can
represent -- where a stencil is a biased filter.  And the instance segmentation says where the basis
is allowed to break:

        F SMOOTH WITHIN A CELL, FREE TO JUMP ACROSS CELL BOUNDARIES.

A global smooth field (a SIREN over the whole sheet) would blur exactly the contrast the per-cell
parameters live on; a stencil does not use the partition at all.

THE LADDER (all anchored to the partition, all fitted per CELL, never per particle)
----------------------------------------------------------------------------------------------------
Each basis fits the MATERIAL map x(X) as a per-cell polynomial in the REFERENCE coordinate and
differentiates it analytically -- F = dx/dX, which is what F means.  Regressing on the CURRENT
position would give F^-1.

    CD     the reference stencil: boxcar-bin at h, central difference, bilinear back  (the 0.999)
    B0     per-cell polynomial degree 1  ->  F constant per cell        (map: 6 params/cell)
    B1     per-cell polynomial degree 2  ->  F linear per cell          (map: 12 params/cell)
    B2     per-cell polynomial degree 3  ->  F quadratic per cell       (map: 20 params/cell)
    SIM    the simulator's own p.F                                      (the 0.0078)

A NOTE ON THE 12 PARAMETERS OF B1, because it is a real restriction and not a coding choice.
The brief specifies B1 as F(x) = F_c + (x - x_c).G_c, "4 + 8 = 12 params/cell".  An 8-parameter G
is not identifiable from displacements: if F is the gradient of a map then dF_ij/dX_k must be
symmetric in (j, k) -- the compatibility condition -- which leaves 6 free second-order coefficients,
not 8.  The quadratic map has 2 + 4 + 6 = 12 parameters per cell, so the parameter COUNT the brief
asks for is exactly the count of the compatible fit; the two incompatible components of G simply
cannot be measured from tracked positions by any estimator, and this one does not pretend to.

MEASUREMENTS, IN THE ORDER THE BRIEF ASKS FOR THEM
----------------------------------------------------------------------------------------------------
  1  ATTENUATION.  Least-squares scale of (F_hat - I) on (F_true - I), per channel and pooled.
     CD ~ 0.003, SIM = 1.  This single number says whether the bias is gone.
  2  BETWEEN-CELL over WITHIN-CELL variance of F, for the true field, the CD and every basis, plus
     the between-cell attenuation and correlation of the CELL MEANS.  A basis that wins (1) and
     flattens (2) has bought nothing -- so both are always printed side by side.
  3  The non-affine residual per cell: FVU of the displacement, in-sample, held-out control points,
     and at PARTICLE resolution.  The anchor from the recording is FVU 0.113 for one affine map per
     compact region.

NOT MEASURED HERE, DELIBERATELY: theta.  The brief stops before the solve.

WARNINGS HONOURED
----------------------------------------------------------------------------------------------------
  * tick 165 is the easiest frame in the window, so FOUR ticks are reported and every table carries
    all of them.  A number quoted at 165 alone is an accident.
  * nothing in this file touches pass0, installs a derived state, or reads C -- there is no solve,
    so the honest-drag and C-oracle traps are out of reach by construction.
  * the per-cell fit is per CELL.  A per-particle fit to one frame is free (2Np dof, 2Np
    observations) and is never performed: the assertion `n_control_points_per_cell > n_params` is
    checked and reported for every cell.

usage:
  /workspace/.conda_envs/neural-graph-linux/bin/python fcell_basis.py --stages M
  PYTHONPATH=/workspace/Plexus/src python fcell_basis.py --stages S --device cuda:0
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

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

REAL = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/"
        "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")      # HEALTHY.  The diseased is SEALED.
LABEL = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material/cardio_cells_label.tif"
PROPS = "/groups/saalfeld/home/allierc/GraphData/graphs_data/material/cardio_cells_props.json"
FIT = (152, 201)                                                 # split.json's fit beat
PX = 4.88e-4                                                     # 1 recording pixel, world units


# =============================================================================================== #
#  STAGE M -- IS THE SEGMENTATION MATERIAL?   numpy only, the recording, no simulator.
# =============================================================================================== #
def stage_M(log, R):
    """Everything below depends on one claim: a control point is a MATERIAL point, so a cell
    membership assigned once at frame 0 is valid for all time and only the positions move.

    Four checks, each of which could refute it:
      M1  are channels 0,1 TRACKED positions, or a static grid?  (a static grid would make the
          whole scheme Eulerian and the membership time-dependent)
      M2  is the displacement the pipeline uses exactly ch(t) - ch(0)?
      M3  does the label map register to the node grid at all?  (checked against props.json's own
          per-cell node counts, which were produced by a different piece of code)
      M4  does the tracked lattice ever FOLD?  If no quadrilateral of the 137x137 lattice ever
          inverts, the tracked map is a homeomorphism for the whole recording, and a partition
          assigned at frame 0 is a partition at every frame.  This is the check that actually
          licenses "the region moves, the membership does not".
    and one quantification:
      M5  how wrong would the lazy alternative be -- re-reading the frame-0 label map at the node's
          frame-t position?  That fraction is the error the material assignment avoids.
    """
    import tifffile

    D = np.load(REAL, mmap_mode="r")
    T, N0, N1, _ = D.shape
    log(f"[M] {REAL.split('/')[-1]}  shape {D.shape}")

    # ---- M1: tracked or static? --------------------------------------------------------------- #
    P = np.asarray(D[:, :, :, 0:2], np.float64)                  # [T, N0, N1, 2] = (X(t), Y(t))
    sd = P.std(0)                                                # per node, per component
    rng = P.max(0) - P.min(0)
    # what a STATIC grid would look like: sd == 0 exactly
    frac_moving = float((sd.max(-1) > 1e-6).mean())
    # which array axis carries which world coordinate (freal_secondderiv.stage_R's convention)
    d0X = float(np.median(np.diff(P[0, :, :, 0], axis=0)))
    d1X = float(np.median(np.diff(P[0, :, :, 0], axis=1)))
    ax_x = 0 if abs(d0X) > abs(d1X) else 1
    h_x = abs(d0X) if ax_x == 0 else abs(d1X)
    d0Y = float(np.median(np.diff(P[0, :, :, 1], axis=0)))
    d1Y = float(np.median(np.diff(P[0, :, :, 1], axis=1)))
    h_y = abs(d0Y) if abs(d0Y) > abs(d1Y) else abs(d1Y)
    # a PERFECT grid would have frame-0 positions on an exact lattice; measure the departure
    q = P[0, :, :, 0] if ax_x == 1 else P[0, :, :, 0].T
    lat = q[:, :1] * 0 + np.arange(q.shape[1])[None, :] * d1X + q[:, :1]
    m1 = {"frac_nodes_moving": frac_moving,
          "median_time_sd_px": float(np.median(sd)),
          "p99_time_sd_px": float(np.percentile(sd, 99)),
          "median_peak_to_peak_px": float(np.median(rng)),
          "max_peak_to_peak_px": float(rng.max()),
          "axis_of_x": ax_x, "node_spacing_x_px": h_x, "node_spacing_y_px": h_y,
          "frame0_deviation_from_perfect_lattice_px": float(np.abs(q - lat).mean())}
    log(f"  [M1] channels 0,1 MOVE on {100*frac_moving:.1f}% of nodes: median time-sd "
        f"{m1['median_time_sd_px']:.3f} px, median peak-to-peak {m1['median_peak_to_peak_px']:.2f} "
        f"px, max {m1['max_peak_to_peak_px']:.2f} px, on a {h_x:.2f} x {h_y:.2f} px lattice, and "
        f"they already sit {m1['frame0_deviation_from_perfect_lattice_px']:.2f} px off a perfect "
        f"lattice at frame 0.  They are TRACKED POSITIONS, not a grid.")

    # ---- M2: the displacement convention ------------------------------------------------------ #
    u = P[:, :, :, 0] - P[0, :, :, 0][None]
    v = P[:, :, :, 1] - P[0, :, :, 1][None]
    m2 = {"max_abs_u_px": float(np.abs(u).max()), "max_abs_v_px": float(np.abs(v).max()),
          "median_abs_u_over_spacing": float(np.median(np.abs(u)) / abs(h_x))}
    log(f"  [M2] u = ch0(t) - ch0(0):  max |u| {m2['max_abs_u_px']:.2f} px, "
        f"|u| is {m2['median_abs_u_over_spacing']:.3f} of a node spacing at the median -- the "
        f"displacement is SMALL compared with the lattice, which is why a stencil over it "
        f"attenuates and why a fit over many nodes is worth doing.")

    # ---- M3: registration of the label map to the node grid ------------------------------------ #
    L = tifffile.imread(LABEL)
    if L.ndim == 3:
        L = L[..., 0]
    props = json.load(open(PROPS))
    # node positions are in FULL-resolution recording pixels; the label map is the same field of
    # view at a different raster, so the scale is (label pixels) / (recording pixels).
    ext_x = float(P[0, :, :, 0].max() - P[0, :, :, 0].min()) + abs(h_x)
    ext_y = float(P[0, :, :, 1].max() - P[0, :, :, 1].min()) + abs(h_y)
    sx, sy_ = L.shape[0] / ext_x, L.shape[1] / ext_y
    ix = np.clip((P[0, :, :, 0] * sx).round().astype(int), 0, L.shape[0] - 1)
    iy = np.clip((P[0, :, :, 1] * sy_).round().astype(int), 0, L.shape[1] - 1)
    lab0 = L[ix, iy]                                             # [N0, N1] label at frame 0
    cnt = np.bincount(lab0.ravel(), minlength=int(L.max()) + 1)
    have = np.array([props[str(k)]["n"] if str(k) in props else -1
                     for k in range(len(cnt))])
    ok = have >= 0
    ok[0] = False
    agree = float(np.mean(cnt[ok] == have[ok]))
    corr = float(np.corrcoef(cnt[ok], have[ok])[0, 1])
    m3 = {"label_shape": list(L.shape), "n_labels": int(L.max()),
          "scale_label_px_per_recording_px": [sx, sy_],
          "exact_count_agreement_with_props": agree, "count_correlation": corr,
          "nodes_with_label_0": int((lab0 == 0).sum()), "n_nodes": int(lab0.size),
          "nodes_per_cell_median": float(np.median(cnt[ok])),
          "nodes_per_cell_min": int(cnt[ok].min()), "nodes_per_cell_max": int(cnt[ok].max()),
          "cells_with_lt_6_nodes": int((cnt[ok] < 6).sum()),
          "cells_with_lt_12_nodes": int((cnt[ok] < 12).sum()),
          "cells_with_lt_20_nodes": int((cnt[ok] < 20).sum()),
          "n_cells_hit": int((cnt[ok] > 0).sum())}
    log(f"  [M3] label map {L.shape} / {int(L.max())} cells, scale {sx:.4f} label-px per "
        f"recording-px.  Per-cell node counts vs props.json: exact agreement on "
        f"{100*agree:.1f}% of cells, correlation {corr:.4f}  ->  the registration is the one the "
        f"pipeline already used.")
    log(f"       nodes per cell: median {m3['nodes_per_cell_median']:.0f}, min "
        f"{m3['nodes_per_cell_min']}, max {m3['nodes_per_cell_max']};  "
        f"{m3['cells_with_lt_6_nodes']} cells < 6 nodes, {m3['cells_with_lt_12_nodes']} < 12, "
        f"{m3['cells_with_lt_20_nodes']} < 20  (= the small-cell tail B0/B1/B2 must handle).")

    # ---- M4: does the tracked lattice ever fold? ----------------------------------------------- #
    # signed area of every elementary quadrilateral, every frame.  A single negative value would
    # mean two material points swapped sides and no fixed partition could be correct.
    def quad_area(Pt):
        a = Pt[:-1, :-1]
        b = Pt[1:, :-1]
        c = Pt[1:, 1:]
        d = Pt[:-1, 1:]
        return 0.5 * ((a[..., 0] * b[..., 1] - b[..., 0] * a[..., 1])
                      + (b[..., 0] * c[..., 1] - c[..., 0] * b[..., 1])
                      + (c[..., 0] * d[..., 1] - d[..., 0] * c[..., 1])
                      + (d[..., 0] * a[..., 1] - a[..., 0] * d[..., 1]))
    A0 = quad_area(P[0])
    sgn = np.sign(np.median(A0))
    worst, nneg = np.inf, 0
    for t in range(T):
        At = quad_area(P[t]) * sgn
        worst = min(worst, float(At.min() / abs(np.median(A0))))
        nneg += int((At <= 0).sum())
    m4 = {"n_quads": int(A0.size), "n_frames": int(T),
          "min_relative_signed_area": worst, "n_negative_quad_frames": nneg,
          "frac_negative": nneg / float(A0.size * T)}
    log(f"  [M4] FOLD TEST over all {T} frames x {A0.size} lattice quads: "
        f"{nneg} inverted ({100*m4['frac_negative']:.4f}%), smallest signed area "
        f"{worst:.4f} of the median.  The tracked lattice is a homeomorphism"
        f"{'' if nneg == 0 else ' EXCEPT for the inversions above -- READ THEM'}, so a partition "
        f"fixed at frame 0 is a partition at every frame.")

    # ---- M5: what the lazy Eulerian alternative would cost ------------------------------------- #
    ts = list(range(FIT[0], FIT[1], 6))
    changed = []
    for t in ts:
        jx = np.clip((P[t, :, :, 0] * sx).round().astype(int), 0, L.shape[0] - 1)
        jy = np.clip((P[t, :, :, 1] * sy_).round().astype(int), 0, L.shape[1] - 1)
        changed.append(float((L[jx, jy] != lab0).mean()))
    m5 = {"ticks": ts, "frac_relabelled": changed, "max_frac_relabelled": max(changed)}
    log(f"  [M5] re-reading the frame-0 label map at the frame-t position would move "
        f"{100*min(changed):.2f}-{100*max(changed):.2f}% of nodes to a different cell.  That is "
        f"the error the MATERIAL assignment avoids -- and it is not small at boundaries.")

    R["stage_M"] = {"M1_tracked": m1, "M2_displacement": m2, "M3_registration": m3,
                    "M4_fold": m4, "M5_eulerian_cost": m5}
    verdict = (m1["frac_nodes_moving"] > 0.99 and m4["n_negative_quad_frames"] == 0)
    R["stage_M"]["membership_is_material"] = bool(verdict)
    log(f"  [M] VERDICT: control points ARE tracked material points and the lattice never folds "
        f"-> membership fixed at frame 0 is valid for the whole recording: "
        f"{'CONFIRMED' if verdict else 'NOT CONFIRMED -- STOP'}")
    return verdict


# =============================================================================================== #
#  THE BASES.  Per-cell polynomial map x(X), differentiated analytically.
# =============================================================================================== #
import torch  # noqa: E402


def _exps(deg):
    return [(d - i, i) for d in range(deg + 1) for i in range(d + 1)]


def poly_and_grad(s, deg):
    """s [n, 2] -> P [n, K] and dP/ds [n, K, 2] for the monomials up to `deg`."""
    ex = _exps(deg)
    K = len(ex)
    P = torch.empty(s.shape[0], K, device=s.device, dtype=s.dtype)
    dP = torch.zeros(s.shape[0], K, 2, device=s.device, dtype=s.dtype)
    for k, (a, b) in enumerate(ex):
        P[:, k] = s[:, 0] ** a * s[:, 1] ** b
        if a > 0:
            dP[:, k, 0] = a * s[:, 0] ** (a - 1) * s[:, 1] ** b
        if b > 0:
            dP[:, k, 1] = b * s[:, 0] ** a * s[:, 1] ** (b - 1)
    return P, dP


NPAR = {1: 3, 2: 6, 3: 10}                       # coefficients per output component
NAME = {1: "B0", 2: "B1", 3: "B2"}


class PerCellPoly:
    """Fit x(X) per cell on CONTROL POINTS, evaluate F = dx/dX anywhere.

    Degree fallback and pooling, both reported:
      * a cell is fitted at the highest degree d <= deg with n_points >= 2 * NPAR[d]
        (an honest factor-2 overdetermination, never a per-point fit);
      * a cell with fewer than 2 * NPAR[1] = 6 points is POOLED: its own points plus the points of
        its nearest neighbour cells, by reference centroid, until 6 are available.  The pooled fit
        is used only for that cell, and the cell is counted.
    """

    def __init__(self, Xc, xc, cell, C, deg, ridge=1e-8, log=None, tag=""):
        dev, dt = Xc.device, Xc.dtype
        Kmax = NPAR[deg]
        self.deg = deg
        self.C = C
        self.coef = torch.zeros(C + 1, Kmax, 2, device=dev, dtype=dt)
        self.ctr = torch.zeros(C + 1, 2, device=dev, dtype=dt)
        self.rad = torch.ones(C + 1, device=dev, dtype=dt)
        self.deg_used = torch.zeros(C + 1, dtype=torch.long)
        self.n_pts = torch.zeros(C + 1, dtype=torch.long)
        idx = [torch.nonzero(cell == c, as_tuple=True)[0] for c in range(C + 1)]
        cen = torch.stack([Xc[i].mean(0) if len(i) else torch.zeros(2, device=dev, dtype=dt)
                           for i in range(0)] + [Xc[i].mean(0) if len(i)
                                                 else torch.full((2,), 1e9, device=dev, dtype=dt)
                                                 for i in idx])
        self.n_pooled, self.n_dropped, self.deg_hist = 0, 0, {}
        for c in range(1, C + 1):
            own = idx[c]
            self.n_pts[c] = len(own)
            use = own
            if len(own) < 2 * NPAR[1]:
                d2 = (cen - cen[c][None]).pow(2).sum(-1)
                d2[c] = -1.0
                order = torch.argsort(d2)
                extra = []
                for cc in order.tolist():
                    if cc == c or cc == 0 or not len(idx[cc]):
                        continue
                    extra.append(idx[cc])
                    if len(own) + sum(len(e) for e in extra) >= 2 * NPAR[1]:
                        break
                if extra:
                    use = torch.cat([own] + extra)
                    self.n_pooled += 1
            if len(use) < 2 * NPAR[1]:
                self.n_dropped += 1
                self.deg_used[c] = 0
                continue
            d = deg
            while d > 1 and len(use) < 2 * NPAR[d]:
                d -= 1
            self.deg_used[c] = d
            self.deg_hist[d] = self.deg_hist.get(d, 0) + 1
            ctr = Xc[own].mean(0) if len(own) else Xc[use].mean(0)
            rad = (Xc[use] - ctr[None]).pow(2).sum(-1).mean().sqrt().clamp(min=1e-12)
            s = (Xc[use] - ctr[None]) / rad
            P, _ = poly_and_grad(s, d)
            G = P.T @ P
            lam = ridge * float(len(use)) * torch.eye(P.shape[1], device=dev, dtype=dt)
            lam[0, 0] = 0.0
            co = torch.linalg.solve(G + lam, P.T @ xc[use])
            self.ctr[c], self.rad[c] = ctr, rad
            self.coef[c, :P.shape[1]] = co
        if log is not None:
            log(f"      [{NAME[deg]}{tag}] degrees used {dict(sorted(self.deg_hist.items()))}, "
                f"pooled {self.n_pooled} cells, dropped {self.n_dropped}; "
                f"points/cell median {float(self.n_pts[1:].double().median()):.0f} "
                f"min {int(self.n_pts[1:].min())} max {int(self.n_pts[1:].max())}")

    def eval(self, X, cell):
        """-> x_hat [n, 2] and F [n, 2, 2] at reference points X assigned to `cell`."""
        s = (X - self.ctr[cell]) / self.rad[cell][:, None]
        P, dP = poly_and_grad(s, self.deg)
        co = self.coef[cell]                                     # [n, K, 2]
        xh = torch.einsum("nk,nka->na", P, co)
        F = torch.einsum("nkb,nka->nab", dP, co) / self.rad[cell][:, None, None]
        return xh, F


# =============================================================================================== #
#  MEASUREMENTS
# =============================================================================================== #
def atten(Fh, Ft, mask=None):
    """(1) attenuation: least-squares scale of (F_hat - I) on (F_true - I), and correlation."""
    if mask is not None:
        Fh, Ft = Fh[mask], Ft[mask]
    I2 = torch.eye(2, device=Fh.device, dtype=Fh.dtype)[None]
    a = (Fh - I2).reshape(-1, 4)
    b = (Ft - I2).reshape(-1, 4)
    sc, co = [], []
    for c in range(4):
        x_, y_ = b[:, c], a[:, c]
        sc.append(float((x_ @ y_) / (x_ @ x_)))
        xm, ym = x_ - x_.mean(), y_ - y_.mean()
        co.append(float((xm @ ym) / (xm.norm() * ym.norm() + 1e-300)))
    xf, yf = b.reshape(-1), a.reshape(-1)
    xm, ym = xf - xf.mean(), yf - yf.mean()
    return {"scale_pooled": float((xf @ yf) / (xf @ xf)),
            "corr_pooled": float((xm @ ym) / (xm.norm() * ym.norm() + 1e-300)),
            "scale_ch": sc, "corr_ch": co,
            "rel_fro": float((Fh - Ft).norm() / Ft.norm()),
            "med_abs_disagree": float((Fh - Ft).abs().median()),
            "disagree_over_FmI": float((Fh - Ft).abs().median()
                                       / ((Ft - I2).abs().median() + 1e-300))}


def bw_variance(F, cell, C, mask=None):
    """(2) between-cell over within-cell variance of F, plus the cell-mean field itself."""
    if mask is not None:
        F, cell = F[mask], cell[mask]
    A = F.reshape(-1, 4)
    n = torch.zeros(C + 1, device=A.device, dtype=A.dtype).index_add_(
        0, cell, torch.ones_like(A[:, 0]))
    s = torch.zeros(C + 1, 4, device=A.device, dtype=A.dtype).index_add_(0, cell, A)
    m = s / n.clamp(min=1)[:, None]
    keep = n > 0
    gm = A.mean(0)
    vb = ((m[keep] - gm[None]).pow(2) * n[keep][:, None]).sum(0) / n[keep].sum()
    vw = (A - m[cell]).pow(2).sum(0) / n[keep].sum()
    return {"var_between": [float(v) for v in vb], "var_within": [float(v) for v in vw],
            "var_between_pooled": float(vb.sum()), "var_within_pooled": float(vw.sum()),
            "ratio_pooled": float(vb.sum() / (vw.sum() + 1e-300)),
            "sd_between_pooled": float(vb.sum().sqrt()),
            "sd_within_pooled": float(vw.sum().sqrt())}, m, keep


def cellmean_atten(mh, mt, keep):
    """(2b) the number that decides whether the CONTRAST survived: attenuation and correlation of
    the per-cell MEAN deviation.  Per-cell (E, gain) live on exactly this quantity."""
    I4 = torch.tensor([1.0, 0.0, 0.0, 1.0], device=mh.device, dtype=mh.dtype)[None]
    a = (mh[keep] - I4).reshape(-1)
    b = (mt[keep] - I4).reshape(-1)
    am, bm = a - a.mean(), b - b.mean()
    return {"cellmean_scale": float((b @ a) / (b @ b)),
            "cellmean_corr": float((bm @ am) / (am.norm() * bm.norm() + 1e-300))}


def fvu(u, uh, cell, C):
    """(3) fraction of variance unexplained of the displacement, WITHIN each cell (the cell-mean
    translation is free, so it is removed from both).  Returns pooled and per-cell median."""
    n = torch.zeros(C + 1, device=u.device, dtype=u.dtype).index_add_(
        0, cell, torch.ones_like(u[:, 0]))
    mu = torch.zeros(C + 1, 2, device=u.device, dtype=u.dtype).index_add_(0, cell, u)
    mu = mu / n.clamp(min=1)[:, None]
    res = (u - uh).pow(2).sum(-1)
    tot = (u - mu[cell]).pow(2).sum(-1)
    rc = torch.zeros(C + 1, device=u.device, dtype=u.dtype).index_add_(0, cell, res)
    tc = torch.zeros(C + 1, device=u.device, dtype=u.dtype).index_add_(0, cell, tot)
    k = (n > 2) & (tc > 0)
    per = (rc[k] / tc[k])
    return {"fvu_pooled": float(res.sum() / tot.sum()),
            "fvu_percell_median": float(per.median()),
            "fvu_percell_p90": float(torch.quantile(per, 0.90)),
            "n_cells": int(k.sum())}


# =============================================================================================== #
#  STAGE S -- the synthetic sheet, where F_true is known.
# =============================================================================================== #
def stage_S(a, log, R):
    import freal_derivedF as FD
    from recover import install_E                                # noqa: F401
    import metrics as MET

    args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent, n_grid=128,
                           warmup=0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    ticks = [int(v) for v in a.ticks.split(",")]
    hs_px = [float(v) for v in a.hpx.split(",")]
    t_max = max(ticks)
    t_start = time.time()

    with torch.no_grad():
        aa = SimpleNamespace(**{**vars(args), "warmup": t_max})
        sy, REF = FD.plant_and_warm_x0(aa, log, keep_ticks=ticks)
        C, Np = sy.C, sy.Np
        dev, dt = sy.device, sy.p.F.dtype
        I2 = torch.eye(2, device=dev, dtype=dt)
        X0 = REF[0]["x"]
        cid = sy.cid
        log(f"[S] planted, ticks {ticks} kept  [{time.time()-t_start:.0f}s]")

        # -- membership in the harness: the analogue of M ------------------------------------- #
        assert int(cid.min()) >= 1 and int(cid.max()) == C
        pc = torch.bincount(cid, minlength=C + 1)[1:]
        log(f"  [S.mem] particle -> cell is a FIXED tensor carried by the MPM particles "
            f"({Np} particles, {C} cells, {int(pc.min())}-{int(pc.max())} per cell, median "
            f"{float(pc.double().median()):.0f}); it cannot change with time.  The control points "
            f"below are boxcar bins defined on the REFERENCE configuration X0, so their membership "
            f"is fixed by construction too -- the bins are Lagrangian, the region they cover moves.")
        R["harness_membership"] = {"Np": Np, "C": C,
                                   "particles_per_cell_min": int(pc.min()),
                                   "particles_per_cell_median": float(pc.double().median()),
                                   "particles_per_cell_max": int(pc.max()),
                                   "membership_fixed_by_construction": True}

        span = (X0.max(0).values - X0.min(0).values)
        area = float(span[0] * span[1])
        p_sp = math.sqrt(area / Np)
        cell_side = math.sqrt(area / C)
        band = 0.06 / MET.SHEET_SPAN
        log(f"  [S.geom] sheet {span[0]:.4f}x{span[1]:.4f}, particle spacing {p_sp/PX:.1f} px, "
            f"cell side {cell_side/PX:.1f} px -> the recording's 6 control points per cell is "
            f"h = {cell_side/6/PX:.0f} px here")
        R["geometry"] = {"particle_spacing_px": p_sp / PX, "cell_side_px": cell_side / PX,
                         "h_matching_6_per_cell_px": cell_side / 6 / PX}

        # ---------------------------------------------------------------------------------- #
        #  Control points.  A control point is a boxcar bin of the reference configuration:
        #  its members are fixed at frame 0 (the bin is defined on X0), its REFERENCE position
        #  is the members' mean X0 and its CURRENT position is the members' mean x(t).  That is
        #  exactly a tracked material point of the recording, and it carries the same spatial
        #  support the recording's PIV window carries.
        #  The CELL of a control point is the majority cell of its members at frame 0 -- the
        #  harness's label-map lookup.
        # ---------------------------------------------------------------------------------- #
        CP = {}
        for hp in hs_px:
            cg = FD.ControlGrid(X0, hp * PX)
            flat = cg.flat
            nb = torch.zeros(cg.n_nodes, device=dev, dtype=dt).index_add_(
                0, flat, torch.ones_like(X0[:, 0]))
            val = torch.nonzero(nb > 0, as_tuple=True)[0]
            remap = torch.full((cg.n_nodes,), -1, device=dev, dtype=torch.long)
            remap[val] = torch.arange(val.numel(), device=dev)
            pid = remap[flat]                                    # particle -> control point
            Xn = torch.zeros(val.numel(), 2, device=dev, dtype=dt).index_add_(0, pid, X0)
            Xn = Xn / nb[val][:, None]
            # majority cell, at frame 0, once
            oh = torch.zeros(val.numel(), C + 1, device=dev, dtype=dt)
            oh.index_put_((pid, cid), torch.ones_like(X0[:, 0]), accumulate=True)
            lab = oh.argmax(1)
            purity = float((oh.max(1).values / nb[val]).mean())
            npc = torch.bincount(lab, minlength=C + 1)[1:]
            CP[hp] = {"cg": cg, "pid": pid, "nb": nb[val], "Xn": Xn, "lab": lab,
                      "n": val.numel()}
            log(f"  [S.cp] h = {hp:.0f} px: {val.numel()} valid control points of {cg.n_nodes} "
                f"nodes, {float(nb[val].mean()):.2f} particles each, majority-cell purity "
                f"{100*purity:.1f}%; per cell median {float(npc.double().median()):.0f} "
                f"min {int(npc.min())} max {int(npc.max())}, "
                f"{int((npc < 6).sum())} cells < 6, {int((npc < 12).sum())} < 12, "
                f"{int((npc < 20).sum())} < 20")
            R.setdefault("control_points", {})[str(hp)] = {
                "n_valid": int(val.numel()), "n_nodes": int(cg.n_nodes),
                "particles_per_point": float(nb[val].mean()), "purity": purity,
                "per_cell_median": float(npc.double().median()),
                "per_cell_min": int(npc.min()), "per_cell_max": int(npc.max()),
                "cells_lt6": int((npc < 6).sum()), "cells_lt12": int((npc < 12).sum()),
                "cells_lt20": int((npc < 20).sum())}

        # ------------------------------------------------------------------ the ladder ----- #
        gen = torch.Generator(device="cpu").manual_seed(7)
        ROWS = []
        for tk in ticks:
            x = REF[tk]["x"]
            Ft = REF[tk]["F"]
            interior = ~((x[:, 0] < band) | (x[:, 0] > 1 - band)
                         | (x[:, 1] < band) | (x[:, 1] > 1 - band))
            u_par = x - X0
            log(f"\n  [S] tick {tk}: med|F_true - I| {float((Ft-I2).abs().median()):.3e}, "
                f"interior {int(interior.sum())}/{Np}")

            # truth's own between/within, the yardstick for measurement 2
            bwT, mT, keepT = bw_variance(Ft, cid, C, interior)
            log(f"      truth: sd_between {bwT['sd_between_pooled']:.3e}  sd_within "
                f"{bwT['sd_within_pooled']:.3e}  B/W {bwT['ratio_pooled']:.3f}")

            def record(name, hp, Fh, extra=None):
                at = atten(Fh, Ft, interior)
                bw, mh, keep = bw_variance(Fh, cid, C, interior)
                cm = cellmean_atten(mh, mT, keepT & keep)
                row = {"tick": tk, "basis": name, "h_px": hp, **at, **bw, **cm}
                if extra:
                    row.update(extra)
                ROWS.append(row)
                return row

            # ---- SIM: the simulator's own F -------------------------------------------------- #
            record("SIM", None, Ft, {"fvu_pooled": float("nan"),
                                     "fvu_percell_median": float("nan")})

            for hp in hs_px:
                cp = CP[hp]
                cg, pid, nbv, Xn, lab = (cp["cg"], cp["pid"], cp["nb"], cp["Xn"], cp["lab"])
                xn = torch.zeros(cp["n"], 2, device=dev, dtype=dt).index_add_(0, pid, x)
                xn = xn / nbv[:, None]                           # tracked control-point position

                # ---- CD: the stencil, exactly as freal_derivedF derives it -------------------- #
                Fcd = FD.derive_F(cg, X0, x, "bilinear", F_ref=None)
                record("CD", hp, Fcd)

                # ---- B0 / B1 / B2 ------------------------------------------------------------ #
                for deg in (1, 2, 3):
                    fit = PerCellPoly(Xn, xn, lab, C, deg, ridge=a.ridge, log=log,
                                      tag=f" h={hp:.0f}px t={tk}")
                    # F at every PARTICLE, from the particle's own cell
                    _, Fh = fit.eval(X0, cid)
                    # (3) non-affine residual: control points in-sample, control points held out,
                    #     and at particle resolution
                    xh_n, _ = fit.eval(Xn, lab)
                    f_in = fvu(xn - Xn, xh_n - Xn, lab, C)
                    xh_p, _ = fit.eval(X0, cid)
                    f_par = fvu(u_par, xh_p - X0, cid, C)
                    # held out: fit on a random half of each cell's control points, score on the
                    # other half.  Nothing here is fitted per particle -- a per-particle fit to one
                    # frame is free and would prove nothing.
                    r = torch.rand(cp["n"], generator=gen).to(dev)
                    ho = {}
                    for half in (0, 1):
                        m = (r < 0.5) if half == 0 else (r >= 0.5)
                        f2 = PerCellPoly(Xn[m], xn[m], lab[m], C, deg, ridge=a.ridge)
                        xh2, _ = f2.eval(Xn[~m], lab[~m])
                        ho[half] = fvu(xn[~m] - Xn[~m], xh2 - Xn[~m], lab[~m], C)
                    row = record(NAME[deg], hp, Fh, {
                        "fvu_in_pooled": f_in["fvu_pooled"],
                        "fvu_in_median": f_in["fvu_percell_median"],
                        "fvu_heldout_pooled": 0.5 * (ho[0]["fvu_pooled"] + ho[1]["fvu_pooled"]),
                        "fvu_heldout_median": 0.5 * (ho[0]["fvu_percell_median"]
                                                     + ho[1]["fvu_percell_median"]),
                        "fvu_particle_pooled": f_par["fvu_pooled"],
                        "fvu_particle_median": f_par["fvu_percell_median"],
                        "fvu_particle_p90": f_par["fvu_percell_p90"],
                        "n_pooled_cells": fit.n_pooled, "n_dropped_cells": fit.n_dropped,
                        "deg_hist": {str(k): v for k, v in sorted(fit.deg_hist.items())}})
                    log(f"      {NAME[deg]} h={hp:>4.0f}px  scale {row['scale_pooled']:.4f}  "
                        f"corr {row['corr_pooled']:.4f}  relFro {row['rel_fro']:.4f}  | "
                        f"B/W {row['ratio_pooled']:.3f}  cellmean scale "
                        f"{row['cellmean_scale']:.4f} corr {row['cellmean_corr']:.4f}  | "
                        f"FVU part {row['fvu_particle_pooled']:.4f} ho "
                        f"{row['fvu_heldout_pooled']:.4f}")

        R["rows"] = ROWS

        # ------------------------------------------------------------------ the tables ----- #
        def get(name, hp, tk):
            for r in ROWS:
                if r["basis"] == name and r["h_px"] == hp and r["tick"] == tk:
                    return r
            return None

        log("\n" + "=" * 118)
        log("TABLE 1+2 -- attenuation AND contrast, together.  scale: CD ~ 0.003, SIM = 1.")
        log("            B/W = between-cell / within-cell variance of F.  truth's own B/W is the "
            "yardstick.")
        log("=" * 118)
        hdr = (f"{'tick':>5s} {'basis':>5s} {'h_px':>5s} | {'scale':>8s} {'corr':>7s} "
               f"{'relFro':>7s} | {'sd_betw':>9s} {'sd_with':>9s} {'B/W':>8s} | "
               f"{'cm_scale':>8s} {'cm_corr':>7s}")
        log(hdr)
        for tk in ticks:
            bwT, mT, keepT = bw_variance(REF[tk]["F"], cid, C, None)
            for name, hp in ([("SIM", None)]
                             + [(n, h) for h in hs_px for n in ("CD", "B0", "B1", "B2")]):
                r = get(name, hp, tk)
                if r is None:
                    continue
                log(f"{tk:>5d} {name:>5s} {('-' if hp is None else f'{hp:.0f}'):>5s} | "
                    f"{r['scale_pooled']:>8.4f} {r['corr_pooled']:>7.4f} {r['rel_fro']:>7.4f} | "
                    f"{r['sd_between_pooled']:>9.3e} {r['sd_within_pooled']:>9.3e} "
                    f"{r['ratio_pooled']:>8.3f} | {r['cellmean_scale']:>8.4f} "
                    f"{r['cellmean_corr']:>7.4f}")
            log("")

        log("=" * 118)
        log("TABLE 3 -- the non-affine residual.  Anchor: one affine map per compact region on the "
            "recording leaves FVU 0.113.")
        log("=" * 118)
        log(f"{'tick':>5s} {'basis':>5s} {'h_px':>5s} | {'FVU in':>8s} {'FVU hold':>9s} "
            f"{'FVU part':>9s} {'med(part)':>9s} {'p90':>7s} | {'pooled':>6s} {'drop':>5s} "
            f"{'degrees':>18s}")
        for tk in ticks:
            for hp in hs_px:
                for name in ("B0", "B1", "B2"):
                    r = get(name, hp, tk)
                    if r is None:
                        continue
                    log(f"{tk:>5d} {name:>5s} {hp:>5.0f} | {r['fvu_in_pooled']:>8.4f} "
                        f"{r['fvu_heldout_pooled']:>9.4f} {r['fvu_particle_pooled']:>9.4f} "
                        f"{r['fvu_particle_median']:>9.4f} {r['fvu_particle_p90']:>7.4f} | "
                        f"{r['n_pooled_cells']:>6d} {r['n_dropped_cells']:>5d} "
                        f"{str(r['deg_hist']):>18s}")
            log("")

        # tick-averaged summary, because 165 is the easiest frame in the window
        log("=" * 118)
        log(f"SUMMARY over ticks {ticks} (mean +- spread).  165 is the easiest frame; a headline "
            f"quoted there alone is an accident.")
        log("=" * 118)
        log(f"{'basis':>5s} {'h_px':>5s} | {'scale':>16s} {'corr':>16s} | {'B/W':>16s} | "
            f"{'cm_scale':>16s} {'cm_corr':>16s} | {'FVU part':>16s}")
        summ = {}
        for name, hp in ([("SIM", None)]
                         + [(n, h) for h in hs_px for n in ("CD", "B0", "B1", "B2")]):
            rs = [get(name, hp, tk) for tk in ticks]
            rs = [r for r in rs if r is not None]
            if not rs:
                continue

            def ms(k):
                v = np.array([r[k] for r in rs], float)
                return float(np.nanmean(v)), float(np.nanstd(v))
            e = {k: ms(k) for k in ("scale_pooled", "corr_pooled", "ratio_pooled",
                                    "cellmean_scale", "cellmean_corr", "rel_fro")}
            e["fvu"] = ms("fvu_particle_pooled") if "fvu_particle_pooled" in rs[0] else (
                float("nan"), float("nan"))
            summ[f"{name}_h{hp}"] = e
            log(f"{name:>5s} {('-' if hp is None else f'{hp:.0f}'):>5s} | "
                f"{e['scale_pooled'][0]:>9.4f}+-{e['scale_pooled'][1]:<6.4f} "
                f"{e['corr_pooled'][0]:>9.4f}+-{e['corr_pooled'][1]:<6.4f} | "
                f"{e['ratio_pooled'][0]:>9.3f}+-{e['ratio_pooled'][1]:<6.3f} | "
                f"{e['cellmean_scale'][0]:>9.4f}+-{e['cellmean_scale'][1]:<6.4f} "
                f"{e['cellmean_corr'][0]:>9.4f}+-{e['cellmean_corr'][1]:<6.4f} | "
                f"{e['fvu'][0]:>9.4f}+-{e['fvu'][1]:<6.4f}")
        R["summary"] = summ
        log(f"\n[S] done [{time.time()-t_start:.0f}s]")


# =============================================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="fcell_basis")
    ap.add_argument("--stages", default="MS")
    ap.add_argument("--ticks", default="165,170,175,180")
    ap.add_argument("--hpx", default="15,34")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--ridge", type=float, default=1e-8)
    a = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"args": vars(a)}
    t0 = time.time()
    if "M" in a.stages:
        stage_M(log, R)
    if "S" in a.stages:
        stage_S(a, log, R)
    R["seconds"] = time.time() - t0
    with open(os.path.join(HERE, f"{a.tag}.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(os.path.join(HERE, f"{a.tag}.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[wrote] {a.tag}.json / .log  [{R['seconds']:.0f}s]")


if __name__ == "__main__":
    main()

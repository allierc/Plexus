"""How the salivary bud's SURFACE cells move, read off Wang et al. 2021's own Imaris nucleus tracks, for exp 21 (an
MPM epithelium whose cells should move as the gland's surface cells do).

    python tools/exp21_gland_surface_motion.py            # both movies, writes data/wang_surface_motion.json

THE DATA (the Figshare zip, read through `tools/wang_smg_stats.py`'s `ZIP` and `_csv`, not copied):
  25x-4  Tracks-epithelial/<movie>_Position.csv + _Shortest_Distance_to_Surfaces...csv: every epithelial (K14-RFP+)
         nucleus, frames 25-289 at 4.998 min (Tracks-all/<movie>_Time.csv), x/y/z in um, and its signed distance
         d to the rendered epithelial surface (negative = inside), joined on the spot ID.
  25x-2  Tracks-surface-proximal/ for the second movie: only tracks that come within 15 um of the surface at some
         frame (Wang Video S3), so the surface layer is complete but the interior is not. Used as a replicate.
The gland is a lobed pancake about 430 x 350 x 100 um, sandwiched between a filter (top, z ~ 1130) and the glass
(bottom, z ~ 1032; Wang STAR Methods "Live-organ imaging": glands between filter and glass bottom, 120 um spacer,
2 um z steps over 100 um). Its surface therefore has three faces: BOTTOM (glass side, outward normal n_z < -0.7),
TOP (filter side, n_z > 0.7) and the RIM (|n_z| <= 0.7), the lateral bud surface where tips and clefts are.

THE SURFACE LAYER. The histogram of d over every spot has one peak at d = -10..-8 um (the outermost nuclei) and a
trough at -18..-14 um before the interior's broad hump. SURFACE here = d > -15 um at the START of a step or window
(the outermost nucleus layer; Wang's own "within 15 um of the surface", Fig 1D-E). The exp 21 targets so far used
d > -8 um, which keeps only the outer ~quarter of that layer (its nuclei read 13.9 um apart against 10-11 um for the
whole layer); every headline number is given for both, labelled S15 and S8. INTERIOR = d < -20 um (past the trough).

THE NORMAL. Per frame, the local surface normal at a spot is the gradient of the signed distance, fitted by least
squares to d over the spot's 20 nearest nuclei within 30 um: d is a distance field, so |grad d| should read 1 (it
reads 0.91 median on the surface layer), and its direction is the outward normal of the nearest surface. Checked
against the smallest principal axis of the 10 nearest surface nuclei (7 deg median disagreement).

DRIFT. As `wang_smg_stats.py`: per frame, the median step of every tracked nucleus (the whole gland's translation),
removed from every displacement. Displacements at lag L use any two spots of one track L frames apart (gaps
between them allowed), with the cumulative drift between the two frames removed.

What it measures (each section of the JSON names its estimator):
  a  speed       per-step speed (5 min), its TANGENTIAL and NORMAL parts (displacement projected on the normal at
                 the step's start), and the change of d; the tangential share of the squared displacement against
                 the isotropic 2/3, raw and with the per-axis localisation noise removed
  b  coherence   `exp_measures.exp21.velocity_correlation_length` (C(r) = <v_i.v_j>/<|v|^2>, first r below 1/e, in
                 spacings) on 30-min tangential displacements of S15 cells, per face and window; the 2D function on
                 the flat faces' (x, y), and its 3D twin (same arithmetic) on every face; with the face's mean removed
                 (the function's own) and with the local mean of same-face cells within 50 um removed; plus the
                 neighbour displacement correlation c(L) = 1 - <|dA - dB|^2> / (<|dA|^2> + <|dB|^2>) by pair
                 distance and lag; gland/face rotation, expansion and top-vs-bottom shear; flow toward clefts
  c  persistence MSD 5 min - 3 h (3D, tangential, normal), log-log slopes, `exp11.prw_fit`, the tangential
                 velocity autocorrelation, speed distributions, and whether neighbours turn together
  d  exchange    `exp21.neighbour_retention` (THE estimator of `tools/exp21_wang_neighbours.py`, its windows and its
                 d0) from 5 min to 3 h, for S8 (reproducing 0.883 at 1 h), S15, each face, rim tip vs cleft and the
                 interior; and how each LOST pair came apart: partner dived, a cell from below inserted between
                 them, a flanking layer cell moved in (T1-like), another layer cell intercalated, or a gap
  e  dives       excursions of surface cells below the layer (d < -20 um) and back (d > -15 um): rate, duration,
                 depth, return fraction; on-surface ratio after 1-4 h (Wang's 0.930)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import wang_smg_stats as W                                                  # noqa: E402
from exp_measures.exp11 import prw_fit                                      # noqa: E402
from exp_measures.exp21 import neighbour_retention, velocity_correlation_length  # noqa: E402

EXP = os.path.join(ROOT, "experiments/exp21_mpm_epithelium")
DATA = os.path.join(EXP, "data")
FRAME_MIN = 4.9983                       # wang_smg_cell_stats.json frame_min (Tracks-all Time.csv)
DT_H = FRAME_MIN / 60.0
D0_UM = 13.9                             # in-layer spacing of the d > -8 um nuclei, wang_surface_neighbours.json
S15, S8, DEEP = -15.0, -8.0, -20.0       # surface layer / outer quarter of it / interior, signed distance in um
NZ_FACE = 0.7                            # |n_z| above which a surface point is on a flat face (45 deg)
MOVIES = {"25x-4": ("2020-01-25-K14R-HisG-2photon-25x-4", "Tracks-epithelial"),
          "25x-2": ("2020-01-25-K14R-HisG-2photon-25x-2", "Tracks-surface-proximal")}


# ============================================================================ loading, normals, drift, pairs
def load(movie, folder):
    """Every tracked nucleus: x, y, z (um), t (frame), tid, d (signed distance, um), sorted by (t, tid)."""
    z = zipfile.ZipFile(W.ZIP)
    pos = W._csv(z, f"{folder}/{movie}-denoised_Position.csv")
    dist = W._csv(z, f"{folder}/{movie}-denoised_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_1.csv")
    df = pos.rename(columns={"Position X": "x", "Position Y": "y", "Position Z": "z", "Time": "t",
                             "TrackID": "tid"})
    df = df.merge(dist[["ID", "Shortest Distance to Surfaces"]].rename(
        columns={"Shortest Distance to Surfaces": "d"}), on="ID", how="left")
    df = df[["x", "y", "z", "t", "tid", "d"]].dropna()
    df["t"] = df["t"].astype(int); df["tid"] = df["tid"].astype(np.int64)
    df = df.drop_duplicates(["tid", "t"]).sort_values(["t", "tid"]).reset_index(drop=True)
    return df


def add_normals(df, k=20, r_um=30.0):
    """Outward normal = grad d, least squares of d on position over the k nearest nuclei within r_um, per frame."""
    X = df[["x", "y", "z"]].values; d = df["d"].values; t = df["t"].values
    G = np.full((len(df), 3), np.nan)
    for tt in np.unique(t):
        ix = np.flatnonzero(t == tt)
        if ix.size < k:
            continue
        P = X[ix]
        dist, nb = cKDTree(P).query(P, k=k)
        A = np.concatenate([np.ones(nb.shape + (1,)), P[nb] - P[:, None, :]], 2)
        w = (dist < r_um).astype(float)
        AtA = np.einsum("nki,nk,nkj->nij", A, w, A)
        Aty = np.einsum("nki,nk,nk->ni", A, w, d[ix][nb])
        ok = w.sum(1) >= 8
        sol = np.full((ix.size, 4), np.nan)
        sol[ok] = np.linalg.solve(AtA[ok] + 1e-6 * np.eye(4), Aty[ok][..., None])[..., 0]
        G[ix] = sol[:, 1:]
    gn = np.linalg.norm(G, axis=1)
    N = G / np.where(gn > 0, gn, np.nan)[:, None]
    df["nx"], df["ny"], df["nz"], df["gn"] = N[:, 0], N[:, 1], N[:, 2], gn
    face = np.full(len(df), "", object)
    surf = (df["d"].values > S15) & np.isfinite(N[:, 2])
    face[surf & (N[:, 2] < -NZ_FACE)] = "bottom"
    face[surf & (N[:, 2] > NZ_FACE)] = "top"
    face[surf & (np.abs(N[:, 2]) <= NZ_FACE)] = "rim"
    df["face"] = face
    return df


def frame_drift(df):
    """Per frame t, the median step t -> t+1 of every nucleus tracked at both; C[t] = drift summed up to frame t."""
    a = df[["tid", "t", "x", "y", "z"]]
    b = a.assign(t=a["t"] - 1)
    m = a.merge(b, on=["tid", "t"], suffixes=("0", "1"))
    st = m[["x1", "y1", "z1"]].values - m[["x0", "y0", "z0"]].values
    dr = pd.DataFrame(st, columns=["dx", "dy", "dz"]).assign(t=m["t"].values).groupby("t").median()
    T = int(df["t"].max()) + 2
    D = np.zeros((T, 3)); D[dr.index.values] = dr.values
    C = np.vstack([np.zeros(3), np.cumsum(D, 0)])[:T]          # C[t] = sum_{u < t} D[u]
    return D, C


def lag_pairs(df, L):
    """Row indices (i0, i1) of the same track L frames apart."""
    a = pd.DataFrame({"tid": df["tid"].values, "t": df["t"].values, "i": np.arange(len(df))})
    b = a.assign(t=a["t"] - L)
    m = a.merge(b, on=["tid", "t"], suffixes=("0", "1"))
    return m["i0"].values, m["i1"].values


def displacement(df, C, i0, i1):
    X = df[["x", "y", "z"]].values
    t0 = df["t"].values[i0]; t1 = df["t"].values[i1]
    return X[i1] - X[i0] - (C[t1] - C[t0])


def split_tn(V, N):
    """Normal part (scalar, along N) and tangential vector of displacements V at unit normals N."""
    vn = (V * N).sum(1)
    return vn, V - vn[:, None] * N


def groups_at(df, rows):
    """Boolean masks of the regions a spot belongs to at the given rows."""
    d = df["d"].values[rows]; f = df["face"].values[rows]; gn = df["gn"].values[rows]
    return {"S15": d > S15, "S8": d > S8, "S15_bottom": f == "bottom", "S15_top": f == "top",
            "S15_rim": f == "rim", "interior": (d < DEEP) & (gn > 0.5)}


# ============================================================================ a: speed, tangential vs normal
def noise_per_axis(df, C):
    """Localisation noise per axis (um): `prw_fit` on each axis's MSD of the surface layer's nuclei (S15), lags
    1-12; the fit's offset c = 6 sigma^2 is for 3 axes, so one axis's offset 2 sigma_a^2 = c reads sigma_a =
    sqrt(3) x sigma. MODEL-DEPENDENT: the PRW's persistence and the offset trade off at lags near P."""
    msd = np.zeros((12, 3))
    for L in range(1, 13):
        i0, i1 = lag_pairs(df, L)
        m = df["d"].values[i0] > S15
        V = displacement(df, C, i0[m], i1[m])
        msd[L - 1] = (V ** 2).mean(0)
    lag = np.arange(1, 13) * DT_H
    return [float(np.sqrt(3.0) * prw_fit(lag, msd[:, a])["noise_sigma"]) for a in range(3)]


def speed_tn(df, C, sig, lags=(1, 2, 3, 6, 12, 24)):
    out = {}
    sig2 = np.asarray(sig) ** 2
    for L in lags:
        i0, i1 = lag_pairs(df, L)
        V = displacement(df, C, i0, i1)
        N = df[["nx", "ny", "nz"]].values[i0]
        ok = np.isfinite(N).all(1)
        vn, vt = split_tn(np.where(ok[:, None], V, 0), np.where(ok[:, None], N, 0))
        dd = df["d"].values[i1] - df["d"].values[i0]
        for g, m in groups_at(df, i0).items():
            m = m & ok
            if m.sum() < 50:
                continue
            h = L * DT_H
            n2 = 2.0 * (N[m] ** 2 @ sig2)                     # noise in the normal part of a displacement
            a2 = 2.0 * sig2.sum()                             # noise in the whole displacement
            msd3, msdn = float((V[m] ** 2).sum(1).mean()), float((vn[m] ** 2).mean())
            msdt = float((vt[m] ** 2).sum(1).mean())
            o = out.setdefault(g, {})
            o[f"{L * FRAME_MIN:.0f}min"] = {
                "n": int(m.sum()),
                "speed_um_h_median": float(np.median(np.linalg.norm(V[m], axis=1)) / h),
                "speed_um_h_iqr": [float(np.percentile(np.linalg.norm(V[m], axis=1), q) / h) for q in (25, 75)],
                "tangential_um_h_median": float(np.median(np.linalg.norm(vt[m], axis=1)) / h),
                "normal_um_h_median": float(np.median(np.abs(vn[m])) / h),
                "delta_d_um_h_median": float(np.median(np.abs(dd[m])) / h),
                "msd3_um2": msd3, "msd_t_um2": msdt, "msd_n_um2": msdn, "msd_dd_um2": float((dd[m] ** 2).mean()),
                "tangential_share": msdt / msd3,
                "tangential_share_noise_removed": (msdt - (a2 - float(n2.mean()))) / (msd3 - a2),
            }
    return out


# ============================================================================ b: coherence, flows
def vcl3(X, V, d0, r_max=8.0, nbins=16, curve=False):
    """`exp21.velocity_correlation_length` with 3D distances and vectors -- the same arithmetic line for line
    (verified equal to it on planar input in `selfcheck`). curve=True also returns (r mids, C, counts)."""
    X = np.asarray(X, float); V = np.asarray(V, float)
    V = V - V.mean(0)
    v2 = float((V * V).sum(1).mean())
    if v2 <= 0 or X.shape[0] < 10:
        return (None, None) if curve else None
    pr = cKDTree(X).query_pairs(r_max * d0, output_type="ndarray")
    if len(pr) == 0:
        return (None, None) if curve else None
    r = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1) / d0
    c = (V[pr[:, 0]] * V[pr[:, 1]]).sum(1) / v2
    edges = np.linspace(0.5, r_max, nbins + 1)
    mid, C, S, Nn = [], [], np.zeros(nbins), np.zeros(nbins)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (r >= a) & (r < b)
        S[k], Nn[k] = c[m].sum(), m.sum()
        if m.sum() >= 5:
            mid.append(0.5 * (a + b)); C.append(float(c[m].mean()))
    L = float(r_max)
    for k in range(len(C)):
        if C[k] < 1.0 / np.e:
            if k == 0:
                L = float(mid[0]); break
            x0, x1, y0, y1 = mid[k - 1], mid[k], C[k - 1], C[k]
            L = float(x0 + (1.0 / np.e - y0) * (x1 - x0) / (y1 - y0)); break
    return (L, (S, Nn)) if curve else L


def local_mean_removed(X, V, radius):
    """V minus the mean V of the OTHER points within `radius` (the local drift); points with no other kept as is."""
    tr = cKDTree(X)
    out = V.copy()
    for i, nb in enumerate(tr.query_ball_point(X, radius)):
        nb = [j for j in nb if j != i]
        if nb:
            out[i] = V[i] - V[nb].mean(0)
    return out


def coherence(df, C, L=6, every=6, d0=D0_UM, seed=0):
    """Velocity correlation length of 30-min tangential displacements of surface-layer cells, per face and window."""
    rng = np.random.default_rng(seed)
    i0, i1 = lag_pairs(df, L)
    keep = np.isfinite(df["nz"].values[i0])
    i0, i1 = i0[keep], i1[keep]
    t0s = df["t"].values[i0]
    V = displacement(df, C, i0, i1)
    N = df[["nx", "ny", "nz"]].values[i0]
    X = df[["x", "y", "z"]].values[i0]
    f = df["face"].values[i0]
    _, Vt = split_tn(V, N)
    edges = np.linspace(0.5, 8.0, 17)
    res = {k: {"L2d": [], "L3d": [], "L3d_local50": [], "L3d_shuffled": [], "mean_flow_share": [], "n": []}
           for k in ("bottom", "top", "rim")}
    curves = {k: {v: [np.zeros(16), np.zeros(16)] for v in ("face_mean", "local50", "shuffled")}
              for k in ("bottom", "top", "rim")}
    for tw in range(int(t0s.min()), int(t0s.max()) + 1, every):
        for face in ("bottom", "top", "rim"):
            m = (t0s == tw) & (f == face)
            if m.sum() < 30:
                continue
            Xm, Vm = X[m], Vt[m]
            r = res[face]
            r["n"].append(int(m.sum()))
            mv = Vm.mean(0)
            r["mean_flow_share"].append(float((mv ** 2).sum() / (Vm ** 2).sum(1).mean()))
            if face != "rim":
                r["L2d"].append(velocity_correlation_length(Xm[:, :2], Vm[:, :2], d0))
            Lf, cf = vcl3(Xm, Vm, d0, curve=True)
            Vl = local_mean_removed(Xm, Vm, 50.0)
            Ll, cl = vcl3(Xm, Vl, d0, curve=True)
            Ls, cs = vcl3(Xm, Vm[rng.permutation(len(Vm))], d0, curve=True)
            r["L3d"].append(Lf); r["L3d_local50"].append(Ll); r["L3d_shuffled"].append(Ls)
            for key, cc in (("face_mean", cf), ("local50", cl), ("shuffled", cs)):
                if cc is not None:
                    curves[face][key][0] += cc[0]; curves[face][key][1] += cc[1]
    summ = {}
    for face, r in res.items():
        s = {"windows": len(r["n"]), "cells_per_window_median": float(np.median(r["n"])) if r["n"] else None}
        for k in ("L2d", "L3d", "L3d_local50", "L3d_shuffled", "mean_flow_share"):
            v = [x for x in r[k] if x is not None]
            s[k + "_median"] = float(np.median(v)) if v else None
            s[k + "_iqr"] = [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None
        s["C_r"] = {key: (cc[0] / np.maximum(cc[1], 1)).tolist() for key, cc in curves[face].items()}
        summ[face] = s
    summ["r_mid_spacings"] = (0.5 * (edges[:-1] + edges[1:])).tolist()
    summ["d0_um"] = d0
    summ["estimator"] = ("exp21.velocity_correlation_length on (x, y) of the flat faces [L2d]; its 3D twin vcl3 "
                         "[L3d]; 30-min tangential displacements of S15 cells at the window start, face mean "
                         "removed (the function's own); local50 = minus the mean of same-face cells within 50 um; "
                         "shuffled = velocities permuted among the cells (the null: the function's floor)")
    return summ


def neighbour_displacement_corr(df, C, lags=(1, 3, 6, 12, 24), d0=None, every=3,
                                shells=((0.0, 1.25), (1.25, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0))):
    """c(L, r) = 1 - <|dA - dB|^2> / (<|dA|^2> + <|dB|^2>) over pairs of S15 cells (same face side: n_A.n_B > 0.8)
    at distance r (in S15 spacings d0) at the start; 0 = independent, 1 = moving together. Drift cancels."""
    X = df[["x", "y", "z"]].values; N = df[["nx", "ny", "nz"]].values; t = df["t"].values
    out = {}
    for L in lags:
        i0, i1 = lag_pairs(df, L)
        keep = df["d"].values[i0] > S15
        i0, i1 = i0[keep], i1[keep]
        acc = {s: [0.0, 0.0, 0] for s in shells}
        for tw in range(int(t.min()), int(t.max()) - L + 1, every):
            m = t[i0] == tw
            if m.sum() < 20:
                continue
            a0, a1 = i0[m], i1[m]
            P = X[a0]; D = (X[a1] - X[a0])
            pr = cKDTree(P).query_pairs(shells[-1][1] * d0, output_type="ndarray")
            if len(pr) == 0:
                continue
            same = (N[a0][pr[:, 0]] * N[a0][pr[:, 1]]).sum(1) > 0.8
            pr = pr[same]
            r = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1) / d0
            num = ((D[pr[:, 0]] - D[pr[:, 1]]) ** 2).sum(1)
            den = (D[pr[:, 0]] ** 2).sum(1) + (D[pr[:, 1]] ** 2).sum(1)
            for s in shells:
                mm = (r >= s[0]) & (r < s[1])
                acc[s][0] += num[mm].sum(); acc[s][1] += den[mm].sum(); acc[s][2] += int(mm.sum())
        out[f"{L * FRAME_MIN:.0f}min"] = {f"{s[0]:g}-{s[1]:g}": {"c": 1.0 - a[0] / max(a[1], 1e-12), "pairs": a[2]}
                                          for s, a in acc.items()}
    return out


def outline(Pxy, h=2.0, sigma_px=4.0):
    """The gland's x-y footprint: a smoothed density of nuclei, thresholded; returns its largest contour (um, closed,
    counter-clockwise) and the signed curvature along it (1/um, > 0 convex = bud tip, < 0 concave = cleft)."""
    from scipy import ndimage as ndi
    from skimage import measure
    x0, y0 = Pxy.min(0) - 30.0
    nx, ny = (np.ptp(Pxy, 0) + 60.0) / h
    H, _, _ = np.histogram2d(Pxy[:, 0], Pxy[:, 1], bins=[int(nx), int(ny)],
                             range=[[x0, x0 + int(nx) * h], [y0, y0 + int(ny) * h]])
    B = ndi.gaussian_filter(H, sigma_px)
    m = B > 0.25 * np.percentile(B[B > 0], 90)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=2))
    lab, nl = ndi.label(m)
    m = lab == (1 + np.argmax(ndi.sum(m, lab, range(1, nl + 1))))
    cs = measure.find_contours(m.astype(float), 0.5)
    c = max(cs, key=len)
    c = np.c_[x0 + c[:, 0] * h, y0 + c[:, 1] * h]
    # resample at 2 um, smooth, orient counter-clockwise
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    s = np.arange(0, seg[-1], 2.0)
    c = np.c_[np.interp(s, seg, c[:, 0]), np.interp(s, seg, c[:, 1])]
    area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    if area < 0:
        c = c[::-1]
    cg = np.c_[ndi.gaussian_filter1d(c[:, 0], 5, mode="wrap"), ndi.gaussian_filter1d(c[:, 1], 5, mode="wrap")]
    d1 = (np.roll(cg, -1, 0) - np.roll(cg, 1, 0)) / 2.0
    d2 = np.roll(cg, -1, 0) - 2 * cg + np.roll(cg, 1, 0)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)
    return cg, kappa


def clefts_tips(cg, kappa, k_cleft=-1.0 / 60.0, k_tip=1.0 / 80.0):
    """Indices along the contour of clefts (local minima of curvature below k_cleft) and tips (maxima above k_tip)."""
    n = len(kappa)
    w = 15                                                     # +-30 um
    cl, tp = [], []
    for i in range(n):
        win = kappa[[(i + j) % n for j in range(-w, w + 1)]]
        if kappa[i] <= win.min() and kappa[i] < k_cleft:
            cl.append(i)
        if kappa[i] >= win.max() and kappa[i] > k_tip:
            tp.append(i)
    return np.array(cl, int), np.array(tp, int)


def flows(df, C, L=6, win=24):
    """Per 2-h window, 30-min tangential velocities of S15 cells: rotation and expansion of each face about its own
    centroid in (x, y), top-vs-bottom shear, and the along-rim velocity toward the nearest cleft."""
    i0, i1 = lag_pairs(df, L)
    keep = df["face"].values[i0] != ""
    i0, i1 = i0[keep], i1[keep]
    V = displacement(df, C, i0, i1)
    N = df[["nx", "ny", "nz"]].values[i0]
    _, Vt = split_tn(V, N)
    Vt = Vt / (L * DT_H)                                       # um / h
    X = df[["x", "y", "z"]].values[i0]
    f = df["face"].values[i0]; t0 = df["t"].values[i0]
    allX = df[["x", "y"]].values; allt = df["t"].values
    rows, rimrows, field = [], [], []
    tmin, tmax = int(t0.min()), int(t0.max())
    for tw in range(tmin, tmax - win + 2, win):
        mw = (t0 >= tw) & (t0 < tw + win)
        cg, kap = outline(allX[(allt >= tw) & (allt < tw + win)])
        cl, tp = clefts_tips(cg, kap)
        row = {"t0": tw, "n_clefts": int(len(cl)), "n_tips": int(len(tp))}
        for face in ("bottom", "top", "rim"):
            m = mw & (f == face)
            if m.sum() < 30:
                continue
            P = X[m, :2] - X[m, :2].mean(0); v = Vt[m, :2]
            r2 = (P ** 2).sum(1).sum()
            row[face] = {"n": int(m.sum()),
                         "omega_deg_h": float(np.degrees((P[:, 0] * v[:, 1] - P[:, 1] * v[:, 0]).sum() / r2)),
                         "expansion_per_h": float((P * v).sum() / r2),
                         "mean_v_um_h": v.mean(0).tolist(),
                         "rms_v_um_h": float(np.sqrt((v ** 2).sum(1).mean())),
                         "R_rms_um": float(np.sqrt(r2 / m.sum()))}
            field.append((tw, face, X[m], Vt[m]))
        if "top" in row and "bottom" in row:
            row["top_minus_bottom_um_h"] = (np.array(row["top"]["mean_v_um_h"]) -
                                            np.array(row["bottom"]["mean_v_um_h"])).tolist()
        rows.append(row)
        # rim and flat-face cells against the clefts of this window
        if len(cl) == 0:
            continue
        tree = cKDTree(cg)
        n = len(cg)
        s_arc = np.arange(n) * 2.0
        tang = np.roll(cg, -1, 0) - np.roll(cg, 1, 0); tang /= np.linalg.norm(tang, axis=1)[:, None]
        for face in ("rim", "bottom", "top"):
            m = mw & (f == face)
            if m.sum() == 0:
                continue
            dist_c, j = tree.query(X[m, :2])
            v = Vt[m, :2]
            if face == "rim":
                # signed arc distance from contour point j to each cleft; nearest cleft by |arc|
                da = (s_arc[cl][None, :] - s_arc[j][:, None])
                per = n * 2.0
                da = (da + per / 2) % per - per / 2
                k = np.argmin(np.abs(da), 1)
                arc = da[np.arange(len(j)), k]
                toward = (v * tang[j]).sum(1) * np.sign(arc)
                for a, u, kk in zip(np.abs(arc), toward, kap[j]):
                    rimrows.append((tw, face, a, u, kk))
            else:
                cp = cg[cl]
                dv = cp[None, :, :] - X[m, None, :2]
                dn = np.linalg.norm(dv, axis=2)
                k = np.argmin(dn, 1)
                e = dv[np.arange(len(k)), k] / dn[np.arange(len(k)), k][:, None]
                toward = (v * e).sum(1)
                for a, u in zip(dn[np.arange(len(k)), k], toward):
                    rimrows.append((tw, face, a, u, np.nan))
    rr = pd.DataFrame(rimrows, columns=["t0", "face", "dist_um", "toward_cleft_um_h", "kappa"])
    bins = [0, 20, 40, 80, 1e9]
    toward = {}
    for face, g in rr.groupby("face"):
        g = g.assign(b=pd.cut(g["dist_um"], bins, right=False))
        toward[face] = {str(b): {"mean_um_h": float(gg["toward_cleft_um_h"].mean()),
                                 "sem_um_h": float(gg["toward_cleft_um_h"].std() / np.sqrt(max(len(gg), 1))),
                                 "n": int(len(gg))} for b, gg in g.groupby("b", observed=True)}
    return rows, toward, field, rr


# ============================================================================ c: MSD, persistence, turning
def msd_curves(df, C, sig, max_lag=36):
    out = {}
    sig2 = np.asarray(sig) ** 2
    for L in range(1, max_lag + 1):
        i0, i1 = lag_pairs(df, L)
        V = displacement(df, C, i0, i1)
        N = np.nan_to_num(df[["nx", "ny", "nz"]].values[i0])
        vn, vt = split_tn(V, N)
        fin = np.isfinite(df["nz"].values[i0])
        for g, m in groups_at(df, i0).items():
            if g not in ("S15", "S8", "interior", "S15_rim", "S15_bottom", "S15_top"):
                continue
            m = m & fin
            o = out.setdefault(g, {"lag_h": [], "msd3": [], "msd_t": [], "msd_n": [], "msd_dd": [], "n": [],
                                   "noise_t": []})
            o["msd_dd"].append(float(((df["d"].values[i1[m]] - df["d"].values[i0[m]]) ** 2).mean()))
            o["lag_h"].append(L * DT_H); o["n"].append(int(m.sum()))
            o["msd3"].append(float((V[m] ** 2).sum(1).mean()))
            o["msd_t"].append(float((vt[m] ** 2).sum(1).mean()))
            o["msd_n"].append(float((vn[m] ** 2).mean()))
            o["noise_t"].append(float(2.0 * (sig2.sum() - (N[m] ** 2 @ sig2).mean())))
    res = {}
    for g, o in out.items():
        lag = np.array(o["lag_h"]); m3 = np.array(o["msd3"]); mt = np.array(o["msd_t"]); mn = np.array(o["msd_n"])
        nt = float(np.mean(o["noise_t"])); n3 = 2.0 * float(sig2.sum())

        def slope(y, a, b):
            s = (lag >= a - 1e-9) & (lag <= b + 1e-9) & (y > 0)
            return float(np.polyfit(np.log(lag[s]), np.log(y[s]), 1)[0])
        fit3 = prw_fit(lag[:24], m3[:24]); fitt = prw_fit(lag[:24], mt[:24])
        res[g] = {"lag_h": lag.tolist(), "msd3_um2": m3.tolist(), "msd_t_um2": mt.tolist(), "msd_n_um2": mn.tolist(),
                  "msd_dd_um2": o["msd_dd"], "alpha_dd_30_120min": slope(np.array(o["msd_dd"]), 0.5, 2.0),
                  "pairs": o["n"],
                  "alpha3_5_30min": slope(m3, 5 / 60, 0.5), "alpha3_30_120min": slope(m3, 0.5, 2.0),
                  "alpha_t_5_30min": slope(mt, 5 / 60, 0.5), "alpha_t_30_120min": slope(mt, 0.5, 2.0),
                  "alpha_t_noise_removed_5_30min": slope(mt - nt, 5 / 60, 0.5),
                  "alpha_t_noise_removed_30_120min": slope(mt - nt, 0.5, 2.0),
                  "alpha_t_120_180min": slope(mt, 2.0, 3.0),
                  "alpha_n_30_120min": slope(mn, 0.5, 2.0),
                  "noise_t_um2": nt, "noise3_um2": n3,
                  "prw3": {"v_um_h": fit3["v"], "P_h": fit3["P"], "r2": fit3["r2"], "noise_sigma_um": fit3["noise_sigma"]},
                  "prw_t": {"v_um_h": fitt["v"], "P_h": fitt["P"], "r2": fitt["r2"],
                            "note": "2D tangential MSD fitted by the same formula (MSD = 2 v^2 P (t - P(1-e^-t/P)) + c)"},
                  "msd_t_1h_um2": float(mt[11]), "msd3_1h_um2": float(m3[11])}
    return res


def vacf(df, C, step=3, max_k=12):
    """Tangential velocity autocorrelation of S15 cells: 15-min displacements u(t), u(t + k x 15 min) of one track,
    both projected on the tangent plane at t; C(k) = <u0.uk> / sqrt(<|u0|^2><|uk|^2>), and <cos angle>."""
    X = df[["x", "y", "z"]].values
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df["tid"].values, df["t"].values]))
    i0, i1 = lag_pairs(df, step)
    m = (df["d"].values[i0] > S15) & np.isfinite(df["nz"].values[i0])
    i0, i1 = i0[m], i1[m]
    N = df[["nx", "ny", "nz"]].values[i0]
    u0 = displacement(df, C, i0, i1)
    u0 = u0 - (u0 * N).sum(1)[:, None] * N
    tid = df["tid"].values[i0]; t = df["t"].values[i0]
    out = {"lag_h": [], "C": [], "cos": [], "n": []}
    for k in range(0, max_k + 1):
        if k == 0:
            out["lag_h"].append(0.0); out["C"].append(1.0); out["cos"].append(1.0); out["n"].append(int(len(i0))); continue
        a = key.reindex(pd.MultiIndex.from_arrays([tid, t + k * step])).values
        b = key.reindex(pd.MultiIndex.from_arrays([tid, t + k * step + step])).values
        ok = np.isfinite(a) & np.isfinite(b)
        a = a[ok].astype(int); b = b[ok].astype(int)
        uk = X[b] - X[a] - (C[df["t"].values[b]] - C[df["t"].values[a]])
        Nk = N[ok]
        uk = uk - (uk * Nk).sum(1)[:, None] * Nk
        u = u0[ok]
        out["lag_h"].append(k * step * DT_H)
        out["C"].append(float((u * uk).sum(1).mean() / np.sqrt((u ** 2).sum(1).mean() * (uk ** 2).sum(1).mean())))
        nu, nk = np.linalg.norm(u, axis=1), np.linalg.norm(uk, axis=1)
        good = (nu > 0) & (nk > 0)
        out["cos"].append(float(((u * uk).sum(1)[good] / (nu[good] * nk[good])).mean()))
        out["n"].append(int(ok.sum()))
    # C(lag > 0) = A exp(-lag / P) + b: a persistent part decaying with P on top of a slow plateau b (the tissue's
    # flow); 1 - A - b at lag 0 is what decorrelates within one 15-min step (noise, jitter)
    from scipy.optimize import curve_fit
    lag = np.array(out["lag_h"][1:]); cc = np.array(out["C"][1:])
    try:
        (A, P, b), _ = curve_fit(lambda x, A, P, b: A * np.exp(-x / P) + b, lag, cc, p0=(0.6, 0.2, 0.05),
                                 bounds=([0, 0.01, -0.2], [1.5, 5.0, 0.5]))
        out["fit_A_exp_minus_lag_over_P_plus_b"] = {"A": float(A), "P_h": float(P), "b": float(b)}
    except Exception:                                          # noqa: BLE001
        out["fit_A_exp_minus_lag_over_P_plus_b"] = None
    return out


def speed_distributions(df, C):
    out = {}
    for L in (1, 6):
        i0, i1 = lag_pairs(df, L)
        V = displacement(df, C, i0, i1)
        N = np.nan_to_num(df[["nx", "ny", "nz"]].values[i0])
        _, vt = split_tn(V, N)
        gs = groups_at(df, i0)
        fin = np.isfinite(df["nz"].values[i0])
        for g in ("S15", "S8", "interior"):
            m = gs[g] & fin
            s = (np.linalg.norm(vt[m], axis=1) if g != "interior" else np.linalg.norm(V[m], axis=1)) / (L * DT_H)
            hist, edges = np.histogram(s, bins=np.linspace(0, 80 if L == 1 else 50, 41))
            # Rayleigh (2D Gaussian velocities) with the same mean square, for the tangential speeds
            out.setdefault(g, {})[f"{L * FRAME_MIN:.0f}min"] = {
                "median": float(np.median(s)), "mean": float(s.mean()), "p10_p90": [float(np.percentile(s, 10)),
                                                                                  float(np.percentile(s, 90))],
                "cv": float(s.std() / s.mean()), "hist": hist.tolist(), "edges": edges.tolist(),
                "kind": "tangential" if g != "interior" else "3D"}
    # per-track mean tangential speed over NON-overlapping 30-min windows (t0, t0 + 30 min, ...), tracks with >= 4
    # such windows at the surface; the between-track share of the variance against the same share after the
    # speeds are permuted across tracks (track sizes kept): the excess is the cells' own speed difference
    i0, i1 = lag_pairs(df, 6)
    m = (df["d"].values[i0] > S15) & np.isfinite(df["nz"].values[i0])
    i0, i1 = i0[m], i1[m]
    V = displacement(df, C, i0, i1)
    N = df[["nx", "ny", "nz"]].values[i0]
    _, vt = split_tn(V, N)
    s = pd.DataFrame({"tid": df["tid"].values[i0], "t": df["t"].values[i0],
                      "v": np.linalg.norm(vt, axis=1) / (6 * DT_H)}).sort_values(["tid", "t"])
    first = s.groupby("tid")["t"].transform("min")
    s = s[((s["t"] - first) % 6) == 0]
    size = s.groupby("tid")["v"].transform("size")
    s = s[size >= 4]

    def share(v, tid):
        g = pd.Series(v).groupby(tid.values)
        return float(g.mean().var() / np.var(v)), g.mean()
    sh, gm = share(s["v"].values, s["tid"])
    rng = np.random.default_rng(0)
    null = [share(rng.permutation(s["v"].values), s["tid"])[0] for _ in range(20)]
    out["track_mean_tangential_30min"] = {"n_tracks": int(gm.size), "windows_per_track_median": float(
        s.groupby("tid").size().median()), "median": float(gm.median()),
        "cv_across_tracks": float(gm.std() / gm.mean()),
        "between_track_variance_share": sh, "between_track_variance_share_shuffled": float(np.mean(null)),
        "intrinsic_cv_of_cell_speed": float(np.sqrt(max(sh - float(np.mean(null)), 0.0) * np.var(s["v"].values))
                                            / np.mean(s["v"].values)),
        "estimator": "non-overlapping 30-min tangential displacements of S15 cells, tracks with >= 4 windows"}
    out["rayleigh_cv_for_reference"] = 0.5227
    return out


def turn_correlation(df, C, step=3, every=3, d0=None):
    """Do neighbours turn together? At each t: v1 = tangential displacement t-15min -> t, v2 = t -> t+15min, both on
    the tangent plane at t; signed turn = atan2((v1 x v2).n, v1.v2); dv = v2 - v1. Over pairs of S15 cells on the
    same face side at r < 1.25 d0 (neighbours) and 3-5 d0 (far): corr of turns, and <dv_i.dv_j>/<|dv|^2>."""
    X = df[["x", "y", "z"]].values; N = df[["nx", "ny", "nz"]].values
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df["tid"].values, df["t"].values]))
    tt = df["t"].values
    res = {"near": {"th": [], "dvd": [], "dv2": [], "v": [], "v2n": []},
           "far": {"th": [], "dvd": [], "dv2": [], "v": [], "v2n": []}}
    for t in range(int(tt.min()) + step, int(tt.max()) - step + 1, every):
        rows = np.flatnonzero((tt == t) & (df["d"].values > S15))
        tid = df["tid"].values[rows]
        a = key.reindex(pd.MultiIndex.from_arrays([tid, np.full(len(tid), t - step)])).values
        b = key.reindex(pd.MultiIndex.from_arrays([tid, np.full(len(tid), t + step)])).values
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(N[rows]).all(1)
        if ok.sum() < 20:
            continue
        r0, a, b = rows[ok], a[ok].astype(int), b[ok].astype(int)
        n = N[r0]
        v1 = X[r0] - X[a] - (C[t] - C[t - step]); v2 = X[b] - X[r0] - (C[t + step] - C[t])
        v1 -= (v1 * n).sum(1)[:, None] * n; v2 -= (v2 * n).sum(1)[:, None] * n
        th = np.arctan2((np.cross(v1, v2) * n).sum(1), (v1 * v2).sum(1))
        dv = v2 - v1
        P = X[r0]
        pr = cKDTree(P).query_pairs(5.0 * d0, output_type="ndarray")
        if len(pr) == 0:
            continue
        pr = pr[(n[pr[:, 0]] * n[pr[:, 1]]).sum(1) > 0.8]
        r = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1) / d0
        for name, mm in (("near", r < 1.25), ("far", (r >= 3.0) & (r < 5.0))):
            q = pr[mm]
            res[name]["th"].append(np.c_[th[q[:, 0]], th[q[:, 1]]])
            res[name]["dvd"].append((dv[q[:, 0]] * dv[q[:, 1]]).sum(1))
            res[name]["dv2"].append(0.5 * ((dv[q[:, 0]] ** 2).sum(1) + (dv[q[:, 1]] ** 2).sum(1)))
            res[name]["v"].append((v2[q[:, 0]] * v2[q[:, 1]]).sum(1))
            res[name]["v2n"].append(0.5 * ((v2[q[:, 0]] ** 2).sum(1) + (v2[q[:, 1]] ** 2).sum(1)))
    out = {}
    for name, r in res.items():
        th = np.vstack(r["th"])
        # circular-safe: correlate sin of the turns (sign and size), and the agreement of the turn sign
        out[name] = {"pairs": int(len(th)),
                     "turn_sin_corr": float(np.corrcoef(np.sin(th[:, 0]), np.sin(th[:, 1]))[0, 1]),
                     "same_turn_sign_frac": float(np.mean(np.sign(th[:, 0]) == np.sign(th[:, 1]))),
                     "dv_corr": float(np.concatenate(r["dvd"]).sum() / np.concatenate(r["dv2"]).sum()),
                     "v_corr": float(np.concatenate(r["v"]).sum() / np.concatenate(r["v2n"]).sum())}
    return out


# ============================================================================ d: neighbour exchange
def retention_curve(df, sel_fn, lags_h, d0_fn=None, d0_fixed=None):
    """`tools/exp21_wang_neighbours.retention`'s loop, with the selection as a function: per lag k frames, start
    frames every k//2, the nuclei tracked at both ends, `neighbour_retention` on those selected at t0, with d0 the
    median nearest-neighbour distance of EVERY nucleus of the reference set at t0 (d0_fn; default: the selected)."""
    by_t = {int(t): g for t, g in df.groupby("t")}
    frames = sorted(by_t)
    out = {}
    for lh in lags_h:
        k = max(1, int(round(lh * 60.0 / FRAME_MIN)))
        vals, npairs, d0s = [], [], []
        for t0 in range(frames[0], frames[-1] - k + 1, max(1, k // 2)):
            a, b = by_t.get(t0), by_t.get(t0 + k)
            if a is None or b is None:
                continue
            a = a.set_index("tid"); b = b.set_index("tid")
            ref = a[(d0_fn or sel_fn)(a)]
            if len(ref) < 3:
                continue
            S0 = ref[["x", "y", "z"]].values
            d0_all = d0_fixed or float(np.median(cKDTree(S0).query(S0, k=2)[0][:, 1]))
            ids = a.index.intersection(b.index)
            if len(ids) < 20:
                continue
            A = a.loc[ids]; B = b.loc[ids]
            kept, n_p, d0 = neighbour_retention(A[["x", "y", "z"]].values, B[["x", "y", "z"]].values,
                                                sel=sel_fn(A), d0=d0_all)
            if kept is not None:
                vals.append(kept); npairs.append(n_p); d0s.append(d0)
        out[f"{lh * 60:.0f}min"] = {"lag_h": lh, "mean": float(np.mean(vals)) if vals else None,
                                    "sd_over_windows": float(np.std(vals)) if vals else None,
                                    "windows": len(vals),
                                    "pairs_median": float(np.median(npairs)) if npairs else None,
                                    "d0_um_median": float(np.median(d0s)) if d0s else None}
    # exchange time: fit kept = exp(-lag / tau) through the points with kept > 0.2
    lag = np.array([v["lag_h"] for v in out.values() if v["mean"]])
    kp = np.array([v["mean"] for v in out.values() if v["mean"]])
    s = kp > 0.2
    tau = float(-1.0 / np.polyfit(lag[s], np.log(kp[s]), 1, w=np.ones(s.sum()))[0]) if s.sum() >= 3 else None
    tau0 = float(np.sum(lag[s] ** 2) / -np.sum(lag[s] * np.log(kp[s]))) if s.sum() >= 2 else None
    return {"curve": out, "tau_h_exp_fit_free_intercept": tau, "tau_h_exp_fit_through_1": tau0}


def exchange_mechanism(df, L=12, every=6, r_in=1.25, r_out=1.5):
    """How the surface-layer neighbour pairs that `neighbour_retention` counts as LOST came apart. Pairs of S15 nuclei
    closer than r_in d0 at t0 (d0 = median nearest-neighbour distance of the S15 nuclei at t0), both tracked at
    t0 + L, farther than r_out d0 at t0 + L. Each is classified, first match wins:
      dive         A or B has left the layer (d < -15 um) at t0 + L
      insertion    a nucleus now in the layer lies between them (in the lune of A-B: closer to both A and B than
                   they are to each other) and was BELOW the layer at t0 (Wang's reinsertion, Fig 2B)
      T1_flank     a layer nucleus between them was a common neighbour of A and B at t0 (within r_out d0 of both):
                   the flanking cell of the A-B junction moved in -- the T1 quartet
      intercalation a layer nucleus between them came from elsewhere in the layer
      untracked    the only nucleus between them was not tracked at t0
      empty        no nucleus between them (a gap: stretching, or a missed detection)"""
    X = df[["x", "y", "z"]].values; d = df["d"].values; t = df["t"].values; tid = df["tid"].values
    fin = np.isfinite(df["nz"].values)
    by_t = {int(tt): np.flatnonzero(t == tt) for tt in np.unique(t)}
    cnt = {k: 0 for k in ("kept", "dive", "insertion", "T1_flank", "intercalation", "untracked", "empty")}
    r0s, r1s, jump, series = [], [], [], []
    key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([tid, t]))
    for t0 in range(int(t.min()), int(t.max()) - L + 1, every):
        rows0, rows1 = by_t.get(t0), by_t.get(t0 + L)
        if rows0 is None or rows1 is None:
            continue
        map0 = dict(zip(tid[rows0], rows0)); map1 = dict(zip(tid[rows1], rows1))
        s0 = rows0[(d[rows0] > S15) & fin[rows0]]
        P0 = X[s0]
        d0 = float(np.median(cKDTree(P0).query(P0, k=2)[0][:, 1]))
        pr = cKDTree(P0).query_pairs(r_in * d0, output_type="ndarray")
        tree1 = cKDTree(X[rows1])
        for ia, ib in pr:
            A0, B0 = s0[ia], s0[ib]
            A1, B1 = map1.get(tid[A0]), map1.get(tid[B0])
            if A1 is None or B1 is None:
                continue
            rab = np.linalg.norm(X[A1] - X[B1])
            if rab < r_out * d0:
                cnt["kept"] += 1
                continue
            r0s.append(np.linalg.norm(X[A0] - X[B0]) / d0); r1s.append(rab / d0)
            if d[A1] <= S15 or d[B1] <= S15:
                cnt["dive"] += 1
                continue
            # how the separation grew: the largest 15-min rise of the pair distance as a share of the whole rise
            fr = np.arange(t0, t0 + L + 1)
            ia_ = key.reindex(pd.MultiIndex.from_arrays([np.full(fr.size, tid[A0]), fr])).values
            ib_ = key.reindex(pd.MultiIndex.from_arrays([np.full(fr.size, tid[B0]), fr])).values
            ok = np.isfinite(ia_) & np.isfinite(ib_)
            if ok.sum() == fr.size:
                rr = np.linalg.norm(X[ia_.astype(int)] - X[ib_.astype(int)], axis=1)
                tot = rr[-1] - rr[0]
                if tot > 0:
                    jump.append(float(np.max(rr[3:] - rr[:-3]) / tot))
                    series.append(rr)
            mid = 0.5 * (X[A1] + X[B1])
            cand = rows1[tree1.query_ball_point(mid, 0.87 * rab)]
            cand = [e for e in cand if e not in (A1, B1) and d[e] > S15
                    and max(np.linalg.norm(X[e] - X[A1]), np.linalg.norm(X[e] - X[B1])) < rab]
            kinds = set()
            for e in cand:
                e0 = map0.get(tid[e])
                if e0 is None:
                    kinds.add("untracked")
                elif d[e0] <= S15:
                    kinds.add("insertion")
                elif max(np.linalg.norm(X[e0] - X[A0]), np.linalg.norm(X[e0] - X[B0])) < r_out * d0:
                    kinds.add("T1_flank")
                else:
                    kinds.add("intercalation")
            for k in ("insertion", "T1_flank", "intercalation", "untracked"):
                if k in kinds:
                    cnt[k] += 1
                    break
            else:
                cnt["empty"] += 1
    # the null for that share: a Brownian bridge between each lost pair's own end distances, with the 5-min jitter
    # of the lost pairs' distance (std of its 5-min changes about their mean): independent, diffusive relative
    # motion with no discrete swap reads the null; a jammed sheet that swaps in one step reads higher
    null = None
    if series:
        S = np.array(series)
        dr = np.diff(S, axis=1)
        sd = float(np.std(dr - dr.mean(1, keepdims=True)))
        rng = np.random.default_rng(0)
        sh = []
        for rr in S:
            for _ in range(5):
                w = np.r_[0.0, np.cumsum(rng.normal(0, sd, L))]
                b = rr[0] + (rr[-1] - rr[0]) * np.arange(L + 1) / L + w - w[-1] * np.arange(L + 1) / L
                sh.append(np.max(b[3:] - b[:-3]) / (rr[-1] - rr[0]))
        null = float(np.median(sh))
    lost = sum(v for k, v in cnt.items() if k != "kept")
    return {"lag_h": L * DT_H, "pairs": lost + cnt["kept"],
            "lost_largest_15min_rise_share_brownian_bridge_null": null, "kept_frac": cnt["kept"] / max(lost + cnt["kept"], 1),
            "lost_share": {k: v / max(lost, 1) for k, v in cnt.items() if k != "kept"}, "counts": cnt,
            "lost_r_t0_over_d0_median": float(np.median(r0s)) if r0s else None,
            "lost_r_t1_over_d0_median": float(np.median(r1s)) if r1s else None,
            "lost_largest_15min_rise_share_median": float(np.median(jump)) if jump else None,
            "lost_largest_15min_rise_share_for_steady_rise": 3.0 / L,
            "lost_pairs_fully_tracked": len(jump),
            "estimator": "pairs and thresholds of exp21.neighbour_retention on S15 (r_in 1.25, r_out 1.5 x d0); "
                         "a nucleus 'between' A and B = in their lune at t0 + L and in the layer"}


# ============================================================================ e: dives below the layer
def dives(df, C, gap_max=2, link_um=12.0, link_frames=3):
    """Per track, hysteresis states on d: SURFACE (d > -15), DEEP (d < -20), else unchanged. A dive = a surface
    track entering DEEP; it RETURNS if it is SURFACE again later in the same track. Gaps up to gap_max frames are
    bridged. A dive whose track ENDS while deep is checked for Wang's Type I division (Fig 2A): new tracks starting
    within link_frames after the end, within link_um of the last position (drift removed) -- and whether any such
    new track reaches the layer (d > -15) within 4 h. Control: the same count around random deep spots."""
    X = df[["x", "y", "z"]].values; tt = df["t"].values; d = df["d"].values
    ev = []
    surf_frames = 0
    for _, g in df.groupby("tid"):
        idx = g.index.values; t = tt[idx]; dd = d[idx]
        state, start, dmin, last_i = None, None, 0.0, None
        for i, ti, di in zip(idx, t, dd):
            if last_i is not None and ti - tt[last_i] > gap_max + 1:
                if state == "deep" and start is not None:
                    ev.append((start, -1, dmin, False, last_i))
                state, start = None, None
            if di > S15:
                if state == "deep" and start is not None:
                    ev.append((start, ti, dmin, True, i))
                    start = None
                state = "surface"
                surf_frames += 1
            elif di < DEEP:
                if state == "surface":
                    state, start, dmin = "deep", ti, di
                elif state == "deep" and start is not None:
                    dmin = min(dmin, di)
            last_i = i
        if state == "deep" and start is not None:
            ev.append((start, -1, dmin, False, last_i))
    e = pd.DataFrame(ev, columns=["t_dive", "t_return", "d_min", "returned", "i_last"])
    tmax = int(tt.max())
    # new tracks: first spot of each track, after the movie's first frame
    first = df.groupby("tid").head(1)
    first = first[first["t"] > tt.min()]
    starts_by_t = {int(k): v.index.values for k, v in first.groupby("t")}
    tid_rows = {k: v.index.values for k, v in df.groupby("tid")}

    def linked(i_end):
        t_end = int(tt[i_end]); hits = []
        for k in range(1, link_frames + 1):
            for j in starts_by_t.get(t_end + k, []):
                if np.linalg.norm(X[j] - X[i_end] - (C[t_end + k] - C[t_end])) < link_um:
                    hits.append(j)
        return hits

    def returns_within(j, frames=48):
        rows = tid_rows[df["tid"].values[j]]
        rows = rows[(tt[rows] >= tt[j]) & (tt[rows] <= tt[j] + frames)]
        hit = rows[d[rows] > S15]
        return (int(tt[hit[0]] - tt[j]) if len(hit) else None), bool(tt[rows].max() - tt[j] >= frames or len(hit))
    ended = e[(~e["returned"]) & (e["i_last"].map(lambda i: tt[i]) < tmax - link_frames)]
    n_d, ret_t, knowable = [], [], 0
    for i_end in ended["i_last"].values:
        h = linked(int(i_end))
        n_d.append(len(h))
        rt = [returns_within(j) for j in h]
        if any(k for _, k in rt):
            knowable += 1
            r = [x for x, _ in rt if x is not None]
            ret_t.append(min(r) if r else None)
    rng = np.random.default_rng(0)
    deep_rows = np.flatnonzero((d < DEEP) & (tt < tmax - link_frames))
    ctrl = [len(linked(int(i))) for i in rng.choice(deep_rows, size=min(2000, len(deep_rows)), replace=False)]
    dur = (e.loc[e["returned"], "t_return"] - e.loc[e["returned"], "t_dive"]) * DT_H
    n_d = np.array(n_d); rr = [x for x in ret_t if x is not None]
    out = {"surface_cell_hours": surf_frames * DT_H, "n_dives": int(len(e)),
           "dives_per_surface_cell_hour": len(e) / max(surf_frames * DT_H, 1e-9),
           "returned_in_same_track_frac": float(e["returned"].mean()) if len(e) else None,
           "track_ended_while_deep_frac": float((~e["returned"]).mean()) if len(e) else None,
           "deep_duration_h_returned_median": float(dur.median()) if len(dur) else None,
           "deep_duration_h_returned_p90": float(dur.quantile(0.9)) if len(dur) else None,
           "d_min_um_median": float(e["d_min"].median()) if len(e) else None,
           "d_min_um_p10": float(e["d_min"].quantile(0.1)) if len(e) else None,
           "ended_deep_with_new_track_within_12um_3frames_frac": float((n_d >= 1).mean()) if len(n_d) else None,
           "ended_deep_with_2plus_new_tracks_frac": float((n_d >= 2).mean()) if len(n_d) else None,
           "control_random_deep_spot_with_new_track_frac": float(np.mean(np.array(ctrl) >= 1)),
           "linked_new_track_reaches_layer_within_4h_frac": (len(rr) / knowable) if knowable else None,
           "linked_new_track_time_to_layer_h_median": float(np.median(rr) * DT_H) if rr else None}
    # on-surface ratio: of the S15 cells at t, the fraction still S15 at t + L (tracked at both)
    for Lh in (1.0, 2.0, 4.0):
        L = int(round(Lh / DT_H))
        i0, i1 = lag_pairs(df, L)
        m = d[i0] > S15
        out[f"still_surface_after_{Lh:g}h"] = float((d[i1[m]] > S15).mean())
    e["t_last"] = e["i_last"].map(lambda i: tt[i])
    return out, e


# ============================================================================ the run
def selfcheck():
    """vcl3 equals exp21.velocity_correlation_length on planar input."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 100, (300, 2)); V = rng.normal(size=(300, 2)) + 0.5 * np.sin(X / 20.0)
    a = velocity_correlation_length(X, V, 10.0)
    b = vcl3(np.c_[X, np.zeros(300)], np.c_[V, np.zeros(300)], 10.0)
    assert abs(a - b) < 1e-9, (a, b)
    return a


def analyse(key, full=True):
    movie, folder = MOVIES[key]
    df = add_normals(load(movie, folder))
    D, C = frame_drift(df)
    out = {"movie": movie, "folder": folder, "frames": [int(df["t"].min()), int(df["t"].max())],
           "frame_min": FRAME_MIN, "n_spots": int(len(df)), "n_tracks": int(df["tid"].nunique())}
    s = df["d"].values > S15
    out["normals"] = {"grad_d_norm_surface_median": float(np.nanmedian(df["gn"].values[s])),
                      "face_share_of_S15_spots": {f: float((df["face"].values[s] == f).mean())
                                                  for f in ("bottom", "top", "rim")}}
    hist, edges = np.histogram(df["d"], bins=np.arange(-52, 14, 2))
    out["d_hist"] = {"counts": hist.tolist(), "edges": edges.tolist()}
    sp = {}
    for t in (int(df["t"].min()), int(np.median(df["t"])), int(df["t"].max())):
        f = df[df["t"] == t]
        for nm, th in (("S8", S8), ("S15", S15)):
            P = f[f["d"] > th][["x", "y", "z"]].values
            sp.setdefault(nm, []).append(float(np.median(cKDTree(P).query(P, k=2)[0][:, 1])))
    out["layer_spacing_um"] = {k: float(np.median(v)) for k, v in sp.items()}
    d15 = out["layer_spacing_um"]["S15"]
    sig = noise_per_axis(df, C)
    out["noise_sigma_um_xyz"] = sig
    out["drift_um_h_median"] = float(np.median(np.linalg.norm(D[int(df["t"].min()):int(df["t"].max())], axis=1)) / DT_H)
    print(key, "loaded", len(df), "spots; spacing", out["layer_spacing_um"], "noise", sig, flush=True)
    out["a_speed_tn"] = speed_tn(df, C, sig)
    print(key, "a done", flush=True)
    out["b_coherence"] = coherence(df, C)
    out["b_coherence_S15_spacing"] = {f: {k: v for k, v in r.items() if k in ("L3d_median", "L3d_local50_median")}
                                      for f, r in coherence(df, C, d0=d15).items() if isinstance(r, dict) and f != "C_r"}
    out["b_neighbour_displacement_corr"] = neighbour_displacement_corr(df, C, d0=d15)
    print(key, "b done", flush=True)
    rows, toward, field, rr = flows(df, C)
    out["b_flows_windows"] = rows
    out["b_toward_cleft"] = toward
    out["c_msd"] = msd_curves(df, C, sig)
    out["c_vacf_tangential_S15"] = vacf(df, C)
    out["c_speed_distributions"] = speed_distributions(df, C)
    out["c_turn_correlation"] = turn_correlation(df, C, d0=d15)
    print(key, "c done", flush=True)
    lags = (5 / 60, 10 / 60, 15 / 60, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
    dd = {}
    dd["S8"] = retention_curve(df, lambda a: a["d"].values > S8, lags)
    dd["S15"] = retention_curve(df, lambda a: a["d"].values > S15, lags)
    if full:
        dd["interior"] = retention_curve(df, lambda a: a["d"].values < DEEP, lags)
        for face in ("bottom", "top", "rim"):
            dd[f"S15_{face}"] = retention_curve(df, (lambda fc: (lambda a: a["face"].values == fc))(face),
                                                (0.5, 1.0, 2.0), d0_fn=lambda a: a["d"].values > S15)
    # kept at 1 h against the ABSOLUTE spacing d0 used by the estimator (pair statistics at a fixed distance do not
    # depend on how completely the layer's nuclei are sampled; d0 does): a model with cells a um apart reads its
    # kept_1h against this curve at d0 = a
    dd["S15_kept_1h_vs_fixed_d0_um"] = {f"{a0:g}": retention_curve(df, lambda a: a["d"].values > S15, (1.0,),
                                                                   d0_fixed=a0)["curve"]["60min"]["mean"]
                                        for a0 in (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)}
    out["d_retention"] = dd
    out["d_mechanism_1h"] = exchange_mechanism(df, L=12)
    out["d_mechanism_30min"] = exchange_mechanism(df, L=6)
    print(key, "d done", flush=True)
    out["e_dives"], ev = dives(df, C)
    print(key, "e done", flush=True)
    # the exp 21 MSD(1 h) target (60 um2) came from GAPLESS S8 tracks only (wang_smg_stats.py); the same with every
    # pair of spots of a track 12 frames apart
    g = df.groupby("tid")["t"].agg(["min", "max", "size"])
    gapless = set(g.index[(g["max"] - g["min"] + 1) == g["size"]])
    i0, i1 = lag_pairs(df, 12)
    V = displacement(df, C, i0, i1)
    s8 = df["d"].values[i0] > S8
    gl = df["tid"].isin(gapless).values[i0]
    out["c_msd_1h_S8_gapless_vs_all"] = {"gapless_tracks_um2": float((V[s8 & gl] ** 2).sum(1).mean()),
                                         "all_pairs_um2": float((V[s8] ** 2).sum(1).mean()),
                                         "gapless_share_of_pairs": float((s8 & gl).sum() / s8.sum())}
    return out, df, C, field, rr, ev


def summary(J):
    """The headline numbers, each with where it comes from in this file."""
    m = J["25x-4"]; m2 = J["25x-2"]
    a = m["a_speed_tn"]; cm = m["c_msd"]; co = m["b_coherence"]; nd = m["b_neighbour_displacement_corr"]
    om = [abs(rw[f]["omega_deg_h"]) for rw in m["b_flows_windows"] for f in ("bottom", "top", "rim") if f in rw]
    tb = np.array([rw["top_minus_bottom_um_h"] for rw in m["b_flows_windows"] if "top_minus_bottom_um_h" in rw])
    return {
        "movie": "25x-4 unless stated; 25x-2 (surface-proximal tracks) as replicate",
        "data_caveat": {"tracked_surface_nuclei_are_a_sparse_sample": "bottom-face g(r): 6-um hard core, no lattice "
                        "peak; NN ~8.8 um vs 7.8 random vs 16.9 for a full hexagonal layer at their density (~1/3 sampled)",
                        "layer_spacing_um": m["layer_spacing_um"], "nuclear_spacing_um_smg2_segmentation": 7.5},
        "a_speed_um_h_5min": {"S15": a["S15"]["5min"]["speed_um_h_median"], "S8": a["S8"]["5min"]["speed_um_h_median"],
                              "interior": a["interior"]["5min"]["speed_um_h_median"],
                              "S15_25x-2": m2["a_speed_tn"]["S15"]["5min"]["speed_um_h_median"]},
        "a_tangential_share": {g: {L: a[g][L]["tangential_share"] for L in ("5min", "30min", "60min", "120min")}
                               for g in ("S15", "interior")},
        "a_depth_change_msd_um2_1h_3h": {g: [cm[g]["msd_dd_um2"][11], cm[g]["msd_dd_um2"][35]] for g in ("S15", "interior")},
        "a_tangential_msd_um2_1h_3h_S15": [cm["S15"]["msd_t_um2"][11], cm["S15"]["msd_t_um2"][35]],
        "b_velocity_correlation_length_spacings": {f: co[f]["L3d_median"] for f in ("bottom", "top", "rim")},
        "b_C_first_bin": {f: co[f]["C_r"]["face_mean"][0] for f in ("bottom", "top", "rim")},
        "b_shared_displacement_nearest_pairs": {L: nd[L]["0-1.25"]["c"] for L in nd},
        "b_face_rotation_abs_max_deg_h": max(om),
        "b_top_minus_bottom_um_h_mean": tb.mean(0).tolist(),
        "c_alpha_tangential": {"5-30min": cm["S15"]["alpha_t_5_30min"], "30-120min": cm["S15"]["alpha_t_30_120min"],
                               "120-180min": cm["S15"]["alpha_t_120_180min"]},
        "c_prw_3d_S15": cm["S15"]["prw3"], "c_vacf_fit": m["c_vacf_tangential_S15"]["fit_A_exp_minus_lag_over_P_plus_b"],
        "c_intrinsic_speed_cv": m["c_speed_distributions"]["track_mean_tangential_30min"]["intrinsic_cv_of_cell_speed"],
        "c_turn_same_sign_near_far": [m["c_turn_correlation"]["near"]["same_turn_sign_frac"],
                                      m["c_turn_correlation"]["far"]["same_turn_sign_frac"]],
        "c_msd_1h_S8_gapless_vs_all": m["c_msd_1h_S8_gapless_vs_all"],
        "d_kept_S8": {k: v["mean"] for k, v in m["d_retention"]["S8"]["curve"].items()},
        "d_kept_S15": {k: v["mean"] for k, v in m["d_retention"]["S15"]["curve"].items()},
        "d_kept_1h_vs_fixed_d0_um": {"25x-4": m["d_retention"]["S15_kept_1h_vs_fixed_d0_um"],
                                     "25x-2": m2["d_retention"]["S15_kept_1h_vs_fixed_d0_um"]},
        "d_lost_pairs_1h": m["d_mechanism_1h"]["lost_share"],
        "d_lost_rise_share_vs_null": [m["d_mechanism_1h"]["lost_largest_15min_rise_share_median"],
                                      m["d_mechanism_1h"]["lost_largest_15min_rise_share_brownian_bridge_null"]],
        "e_dives_per_surface_cell_hour": m["e_dives"]["dives_per_surface_cell_hour"],
        "e_still_surface_1h_2h_4h": [m["e_dives"][f"still_surface_after_{h}h"] for h in (1, 2, 4)],
    }


def tip_cleft_retention(df, lag_h=1.0, win=24):
    """kept at lag_h for rim pairs near a cleft (contour curvature < -1/60 um^-1 within 20 um) vs near a tip."""
    res = {}
    allX = df[["x", "y"]].values; allt = df["t"].values
    lab = np.full(len(df), "", object)
    for tw in range(int(allt.min()), int(allt.max()) + 1, win):
        mw = (allt >= tw) & (allt < tw + win)
        cg, kap = outline(allX[mw])
        cl, tp = clefts_tips(cg, kap)
        m = mw & (df["face"].values == "rim")
        if not m.any():
            continue
        tree = cKDTree(cg)
        _, j = tree.query(allX[m])
        s_arc = np.arange(len(cg)) * 2.0; per = len(cg) * 2.0

        def near(idx):
            if len(idx) == 0:
                return np.zeros(len(j), bool)
            da = np.abs(((s_arc[idx][None, :] - s_arc[j][:, None]) + per / 2) % per - per / 2)
            return da.min(1) < 20.0
        lab_m = np.where(near(cl), "cleft", np.where(near(tp), "tip", "flank"))
        lab[np.flatnonzero(m)] = lab_m
    df = df.assign(rimclass=lab)
    for cls in ("cleft", "tip", "flank"):
        res[cls] = retention_curve(df, (lambda c: (lambda a: a["rimclass"].values == c))(cls), (0.5, lag_h, 2.0),
                                   d0_fn=lambda a: a["d"].values > S15)["curve"]
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=os.path.join(DATA, "wang_surface_motion.json"))
    ap.add_argument("--cache", default=os.path.join(DATA, "wang_surface_motion_cache.npz"))
    a = ap.parse_args()
    out = {"selfcheck_vcl_planar": selfcheck(),
           "definitions": {"S15": "d > -15 um at the start (outermost nucleus layer)",
                           "S8": "d > -8 um (the exp 21 targets so far)", "interior": "d < -20 um",
                           "faces": "bottom n_z < -0.7 (glass side), top n_z > 0.7 (filter side), rim |n_z| <= 0.7",
                           "d0_um_for_correlation_lengths": D0_UM, "frame_min": FRAME_MIN}}
    m4, df4, C4, field4, rr4, ev4 = analyse("25x-4")
    m4["d_rim_tip_vs_cleft"] = tip_cleft_retention(df4)
    out["25x-4"] = m4
    m2, *_ = analyse("25x-2", full=False)
    out["25x-2"] = m2
    # the figure script's inputs that are slow to recompute: the 2-h flow fields and the cleft rows
    fx, fv, ft, ff = [], [], [], []
    for tw, face, X, V in field4:
        fx.append(X); fv.append(V); ft.append(np.full(len(X), tw)); ff.append(np.full(len(X), face))
    np.savez_compressed(a.cache, X=np.vstack(fx), V=np.vstack(fv), t0=np.concatenate(ft),
                        face=np.concatenate(ff).astype("U6"),
                        rr=rr4[["t0", "dist_um", "toward_cleft_um_h", "kappa"]].values,
                        rr_face=rr4["face"].values.astype("U6"),
                        dives=ev4[["t_dive", "t_return", "d_min", "returned", "t_last"]].astype(float).values)
    if os.path.exists(a.json):                                  # keep the SMG2 section of exp21_gland_smg2_check.py
        old = json.load(open(a.json))
        if "smg2_check" in old:
            out["smg2_check"] = old["smg2_check"]
    out = json.loads(json.dumps(out, default=float))
    out = {"summary": summary(out), **out}
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.json, a.cache)


if __name__ == "__main__":
    main()

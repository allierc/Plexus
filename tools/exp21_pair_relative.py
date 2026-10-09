"""Why the model's cells lose more neighbours than the gland's at the same mobility (exp 21, Finding 28: corr 0.33
with kept_1h 0.82 in the model, corr 0.107 with kept_1h 0.883 in the gland).

Over 1-h windows, for the cells of each side (the gland's surface nuclei, within 8 um of the surface; the model's
interior cells), reads:

  msd_cell     the single cell's 1-h squared displacement: the gland's split into the part ALONG the local
               surface (tangent plane: normal = least-variance axis of the surface nuclei within 3 spacings) and
               the part along the normal; the model's is in the plane
  msd_pair     the 1-h squared change of the separation vector of neighbour pairs (closer than 1.25 spacings at
               the start), tangent part for the gland -- what parts neighbours. For independent cells it is 2 x
               msd_cell, for cells carried together 0
  kurt         the kurtosis of the cells' 1-h tangent displacement (one axis, 3 for a Gaussian): a few cells moving
               far and most little part fewer pairs than everyone moving alike, at the same MSD
  kept         `neighbour_retention` (the gate's), for reference
  *_aff        msd_cell and msd_pair with the window's best AFFINE flow of the set removed (`affine_residual`, the
               correlation gate's): the bud grows and turns, so part of a surface nucleus's displacement is the
               tissue's own stretch and rotation, not the cell moving among its neighbours; the corral has none

    PYTHONPATH=src:tools python tools/exp21_pair_relative.py [--runs tissue/exp21_g_c20_m1_s1 ...]

Writes experiments/exp21_mpm_epithelium/data/pair_relative.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from exp_measures.exp21 import neighbour_retention, interior_mask, affine_residual  # noqa: E402

LAG = 12                                            # 1 h at the gland's 4.998-min and the model's 5-min frames


def _normals(S, d0):
    from scipy.spatial import cKDTree
    tr = cKDTree(S)
    N = np.zeros_like(S)
    for i, x in enumerate(S):
        nb = tr.query_ball_point(x, 3.0 * d0)
        if len(nb) >= 6:
            Q = S[nb] - S[nb].mean(0)
            N[i] = np.linalg.svd(Q, full_matrices=False)[2][-1]
    return N


def _kurt(D):
    x = np.concatenate([D[:, 0], D[:, 1]])
    x = x - x.mean()
    return float((x ** 4).mean() / max((x ** 2).mean() ** 2, 1e-12))


def _pairs(X, d0):
    from scipy.spatial import cKDTree
    return cKDTree(X).query_pairs(1.25 * d0, output_type="ndarray")


def gland(movie, surface_um=8.0):
    import wang_smg_stats as W
    from scipy.spatial import cKDTree
    z = zipfile.ZipFile(W.ZIP)
    pos = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Position.csv")
    dist = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_1.csv")
    df = pos.rename(columns={"Position X": "x", "Position Y": "y", "Position Z": "z", "Time": "t"})
    df = df.merge(dist[["ID", "Shortest Distance to Surfaces"]].rename(
        columns={"Shortest Distance to Surfaces": "d"}), on="ID", how="left")
    df = df[["x", "y", "z", "t", "TrackID", "d"]].dropna(subset=["TrackID", "d"])
    frames = np.sort(df["t"].unique())
    by_t = {int(t): g.drop_duplicates("TrackID").set_index("TrackID") for t, g in df.groupby("t")}
    acc = {k: [] for k in ("msd_tan", "msd_nrm", "msd_pair_tan", "msd_pair_3d", "kurt", "kept", "msd_tan_aff")}
    for t0 in range(int(frames[0]), int(frames[-1]) - LAG + 1, LAG // 2):
        a, b = by_t.get(t0), by_t.get(t0 + LAG)
        if a is None or b is None:
            continue
        S0 = a[a["d"].values > -surface_um][["x", "y", "z"]].values
        if len(S0) < 30:
            continue
        d0 = float(np.median(cKDTree(S0).query(S0, k=2)[0][:, 1]))
        ids = a.index.intersection(b.index)
        A, B = a.loc[ids], b.loc[ids]
        srf = A["d"].values > -surface_um
        X0, X1 = A[["x", "y", "z"]].values[srf], B[["x", "y", "z"]].values[srf]
        if len(X0) < 30:
            continue
        D = X1 - X0
        D = D - np.median(D, axis=0)                                        # the bud's drift over the window
        N = _normals(X0, d0)
        dn = (D * N).sum(1)
        Dt = D - dn[:, None] * N
        acc["msd_tan"].append(float((Dt ** 2).sum(1).mean())); acc["msd_nrm"].append(float((dn ** 2).mean()))
        Da = affine_residual(X0, X1)
        Da = Da - (Da * N).sum(1)[:, None] * N
        acc["msd_tan_aff"].append(float((Da ** 2).sum(1).mean()))
        pr = _pairs(X0, d0)
        if len(pr):
            dR = D[pr[:, 1]] - D[pr[:, 0]]
            Nm = N[pr[:, 0]]
            dRt = dR - (dR * Nm).sum(1)[:, None] * Nm
            acc["msd_pair_tan"].append(float((dRt ** 2).sum(1).mean())); acc["msd_pair_3d"].append(float((dR ** 2).sum(1).mean()))
        # tangent displacement in a local 2D frame per cell (two axes orthogonal to its normal) for the kurtosis
        e1 = np.cross(N, np.array([0.0, 0.0, 1.0])); bad = np.linalg.norm(e1, axis=1) < 1e-6
        e1[bad] = np.cross(N[bad], np.array([1.0, 0.0, 0.0])); e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
        e2 = np.cross(N, e1)
        acc["kurt"].append(_kurt(np.c_[(Dt * e1).sum(1), (Dt * e2).sum(1)]))
        k, _, _ = neighbour_retention(X0, X1, d0=d0)
        if k is not None:
            acc["kept"].append(k)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def model(spec, rim=1.5):
    from exp_measures.common import open_run
    from scipy.spatial import cKDTree
    T = open_run(spec)
    n = T.n_rows()
    z = T.z
    um = float(((T.spec.get("general") or {}).get("units") or {}).get("length_um", 1.0))
    C = np.array([np.asarray(z["cell__pos"][t], float) for t in range(n)])
    alive = np.isfinite(C).all(axis=(0, 2))
    if "cell__occ" in z.files:
        alive &= np.array([(np.asarray(z["cell__occ"][t], float).reshape(-1) > 0.5) for t in range(n)]).all(0)
    X = C[:, alive, :2] * um
    d0 = float(np.median(cKDTree(X[0]).query(X[0], k=2)[0][:, 1]))
    inner = interior_mask(X[0], rim, d0)
    acc = {k: [] for k in ("msd_tan", "msd_pair_tan", "kurt", "kept", "msd_tan_aff")}
    for t0 in range(0, n - LAG, LAG // 2):
        X0, X1 = X[t0][inner], X[t0 + LAG][inner]
        D = X1 - X0
        D = D - np.median(D, axis=0)
        acc["msd_tan"].append(float((D ** 2).sum(1).mean()))
        acc["msd_tan_aff"].append(float((affine_residual(X0, X1) ** 2).sum(1).mean()))
        pr = _pairs(X0, d0)
        dR = D[pr[:, 1]] - D[pr[:, 0]]
        acc["msd_pair_tan"].append(float((dR ** 2).sum(1).mean()))
        acc["kurt"].append(_kurt(D))
        k, _, _ = neighbour_retention(X0, X1, d0=d0)
        if k is not None:
            acc["kept"].append(k)
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["d0_um"] = d0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--movie", default="2020-01-25-K14R-HisG-2photon-25x-4")
    ap.add_argument("--runs", nargs="*", default=["tissue/exp21_g_base_s1", "tissue/exp21_g_c20_m1_s1",
                                                  "tissue/exp21_g_c20_d2_s1", "tissue/exp21_g_a00005_d4f105_s1"])
    a = ap.parse_args()
    res = {"gland": gland(a.movie)}
    for s in a.runs:
        res[s.split("/")[-1]] = model(s)
    with open(os.path.join(ROOT, "experiments/exp21_mpm_epithelium/data", "pair_relative.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, r in res.items():
        ratio = r["msd_pair_tan"] / max(2 * r["msd_tan"], 1e-12)
        print(f"{k:28s} msd_cell {r['msd_tan']:6.1f}" + (f" (+normal {r['msd_nrm']:.1f})" if "msd_nrm" in r else "")
              + f"  affine-removed {r['msd_tan_aff']:6.1f}  msd_pair {r['msd_pair_tan']:6.1f}  pair/(2 cell) {ratio:.2f}  kurt {r['kurt']:.2f}  kept {r.get('kept', float('nan')):.3f}")


if __name__ == "__main__":
    main()

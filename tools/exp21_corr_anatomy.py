"""The ANATOMY of exp 21's neighbour correlation (`corr_1sp`): the gland reads 0.107, every intact model sheet
0.3-0.5. Before another mechanism is tried in the model, three readings of the SAME correlation say what kind of
co-motion it is, on the gland's tracks and on model runs alike (`exp_measures.exp21.neighbour_correlation` and
`affine_residual`, the gate's own functions):

  window   corr_1sp of displacements over 15 min, 30 min (the gate's), 1 h and 2 h. A part of each cell's motion
           that is BOUNDED (a nucleus jiggling inside its cell, localisation noise) dilutes the correlation of short
           windows only, so a gland whose cells really move together would read higher at 2 h than at 15 min;
           cells that move independently read low at every window.
  L / T    the correlation of the displacement components ALONG the line joining the two neighbours (L) and across
           it (T). A push handed on through a contact moves the neighbour ahead the same way: L > T. Two cells
           sliding past each other in opposite directions (the human's "two people passing") read T < 0.
  tangent  (gland only) the correlation of the displacements projected on the local surface plane (normal = the
           least-variance axis of the surface nuclei within 3 spacings): the gland's nuclei also move along the
           surface normal (22 % of their 1-h squared displacement), which the model's flat sheet has no room for.

    PYTHONPATH=src:tools python tools/exp21_corr_anatomy.py [--runs tissue/exp21_g_base_s1 ...]

Writes experiments/exp21_mpm_epithelium/data/corr_anatomy.json and figs/corr_anatomy.png (+ .txt, the watcher's
`exp21-gland-data` record).
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
from exp_measures.exp21 import (neighbour_correlation, neighbour_correlation_lt as pair_lt,  # noqa: E402
                                affine_residual, interior_mask)

WINDOWS_MIN = (15, 30, 60, 120)
FRAME_MIN = 5.0                                     # the gland's 4.998-min frames and the model's 300-s frames


def tangent_project(X, V, S, d0):
    """V with its component along the local surface normal removed; the normal at each X is the least-variance
    axis of the surface nuclei S within 3 spacings (at least 6 of them)."""
    from scipy.spatial import cKDTree
    tr = cKDTree(S)
    out = V.copy()
    for i, x in enumerate(X):
        nb = tr.query_ball_point(x, 3.0 * d0)
        if len(nb) < 6:
            continue
        Q = S[nb] - S[nb].mean(0)
        n = np.linalg.svd(Q, full_matrices=False)[2][-1]
        out[i] = V[i] - (V[i] @ n) * n
    return out


def _summ(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def gland(movie, surface_um=8.0):
    import wang_smg_stats as W
    z = zipfile.ZipFile(W.ZIP)
    pos = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Position.csv")
    dist = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_1.csv")
    df = pos.rename(columns={"Position X": "x", "Position Y": "y", "Position Z": "z", "Time": "t"})
    df = df.merge(dist[["ID", "Shortest Distance to Surfaces"]].rename(
        columns={"Shortest Distance to Surfaces": "d"}), on="ID", how="left")
    df = df[["x", "y", "z", "t", "TrackID", "d"]].dropna(subset=["TrackID", "d"])
    frames = np.sort(df["t"].unique())
    by_t = {int(t): g.drop_duplicates("TrackID").set_index("TrackID") for t, g in df.groupby("t")}
    from scipy.spatial import cKDTree
    out = {"source": f"Wang 2021 {movie}, surface nuclei (within {surface_um} um of the surface)", "window": {}}
    for wmin in WINDOWS_MIN:
        k = int(round(wmin / FRAME_MIN))
        c3, ctan, cl, ct, n = [], [], [], [], 0
        for t0 in range(int(frames[0]), int(frames[-1]) - k + 1, k):
            a, b = by_t.get(t0), by_t.get(t0 + k)
            if a is None or b is None:
                continue
            ids = a.index.intersection(b.index)
            A, B = a.loc[ids], b.loc[ids]
            srf = A["d"].values > -surface_um
            if srf.sum() < 30:
                continue
            S0 = a[a["d"].values > -surface_um][["x", "y", "z"]].values
            d0 = float(np.median(cKDTree(S0).query(S0, k=2)[0][:, 1]))
            X0 = A[["x", "y", "z"]].values[srf]
            D = affine_residual(X0, B[["x", "y", "z"]].values[srf])
            c3.append(neighbour_correlation(X0, D, d0))
            Dt = tangent_project(X0, D, S0, d0)
            ctan.append(neighbour_correlation(X0, Dt, d0))
            l_, t_ = pair_lt(X0, Dt, d0)
            cl.append(l_); ct.append(t_); n += 1
        out["window"][str(wmin)] = {"corr_1sp": _summ(c3), "corr_tangent": _summ(ctan), "C_L": _summ(cl),
                                    "C_T": _summ(ct), "windows": n}
    return out


def model(spec, rim=1.5):
    from exp_measures.common import open_run
    T = open_run(spec)
    n = T.n_rows()
    z = T.z
    C = np.array([np.asarray(z["cell__pos"][t], float) for t in range(n)])
    co = np.array([(np.asarray(z["cell__occ"][t], float).reshape(-1) > 0.5) if "cell__occ" in z.files
                   else np.ones(C.shape[1], bool) for t in range(n)])
    alive = co.all(0) & np.isfinite(C).all(axis=(0, 2))
    from scipy.spatial import cKDTree
    X = C[:, alive, :2]
    d0 = float(np.median(cKDTree(X[0]).query(X[0], k=2)[0][:, 1]))
    Xi = X[:, interior_mask(X[0], rim, d0)]
    out = {"source": spec, "n_interior": int(Xi.shape[1]), "window": {}}
    for wmin in WINDOWS_MIN:
        k = int(round(wmin / FRAME_MIN))
        c2, cl, ct = [], [], []
        for t0 in range(0, n - k, k):
            D = affine_residual(Xi[t0], Xi[t0 + k])
            c2.append(neighbour_correlation(Xi[t0], D, d0))
            l_, t_ = pair_lt(Xi[t0], D, d0)
            cl.append(l_); ct.append(t_)
        out["window"][str(wmin)] = {"corr_1sp": _summ(c2), "C_L": _summ(cl), "C_T": _summ(ct), "windows": len(c2)}
    return out


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, r in res.items():
        w = [int(k) for k in r["window"]]
        key = "corr_tangent" if name == "gland" else "corr_1sp"
        ax[0].plot(w, [r["window"][str(k)][key] for k in w], "o-", color="black" if name == "gland" else None,
                   lw=2.5 if name == "gland" else 1.2, label=name)
        c = r["window"]["30"]
        ax[1].scatter([c["C_L"]], [c["C_T"]], s=40, color="black" if name == "gland" else None, label=name)
    ax[0].set_xscale("log"); ax[0].set_xticks(WINDOWS_MIN); ax[0].set_xticklabels([str(k) for k in WINDOWS_MIN])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("displacement window (min)"); ax[0].set_ylabel("neighbour correlation (affine flow removed)")
    ax[0].axhline(0, color="grey", lw=0.5)
    lim = [-0.2, 0.8]
    ax[1].plot(lim, lim, color="grey", lw=0.5)
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel("C_L, along the pair (30 min)"); ax[1].set_ylabel("C_T, across the pair (30 min)")
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--movie", default="2020-01-25-K14R-HisG-2photon-25x-4")
    ap.add_argument("--runs", nargs="*", default=["tissue/exp21_g_base_s1", "tissue/exp21_g_sig036_s1",
                                                  "tissue/exp21_g_d2f085_s1", "tissue/exp21_g_d4f12_s1"])
    a = ap.parse_args()
    res = {"gland": gland(a.movie)}
    for s in a.runs:
        try:
            res[s.split("/")[-1].replace("exp21_g_", "").replace("_s1", "")] = model(s)
        except Exception as e:                                                  # noqa: BLE001
            print("skip", s, e)
    dd = os.path.join(ROOT, "experiments/exp21_mpm_epithelium")
    with open(os.path.join(dd, "data", "corr_anatomy.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    figure(res, os.path.join(dd, "figs", "corr_anatomy.png"))
    for name, r in res.items():
        row = "  ".join(f"{k}m {v.get('corr_tangent', v['corr_1sp']):+.3f}" if v["corr_1sp"] is not None else f"{k}m --"
                        for k, v in r["window"].items())
        c = r["window"]["30"]
        print(f"{name:14s} {row}   30m: L {c['C_L']:+.3f} T {c['C_T']:+.3f}"
              + (f"  3D {c['corr_1sp']:+.3f}" if name == "gland" else ""))


if __name__ == "__main__":
    main()

"""How fast the salivary bud's SURFACE cells change neighbours, read off Wang et al. 2021's own nucleus tracks, for
exp 21 (an epithelium of MPM cells whose cells must move among their neighbours).

    python tools/exp21_wang_neighbours.py [--movie 2020-01-25-K14R-HisG-2photon-25x-4] [--surface_um 8]

It reads the same Imaris export as `tools/wang_smg_stats.py` (exp 11's reader, imported, not copied): every
epithelial nucleus position per frame and its signed distance to the epithelial surface (negative = inside). A
nucleus is a SURFACE nucleus when it is within `surface_um` of the surface at the start of the window.

For each lag L (1 h and 2 h, at the movie's ~5-min frames) and each start frame t0 (every L/2): the nuclei tracked at
both t0 and t0 + L, the surface ones at t0 selected, and `exp_measures.exp21.neighbour_retention` -- THE SAME function
exp 21's ruler runs on the model -- returns the fraction of their neighbour pairs (closer than 1.25 x the median
nearest-neighbour distance of ALL surface nuclei at t0) still within 1.5 x that distance after the lag. The windows are averaged. The
distance threshold is the population's own, so the gland's 9.9 um nuclei and the model's 1-unit cells compare
directly. Positions are 3D: the gland's surface is a curved sheet, and pair distances need no projection.

Writes experiments/exp21_mpm_epithelium/data/wang_surface_neighbours.json.
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
import wang_smg_stats as W                                          # noqa: E402
from exp_measures.exp21 import neighbour_retention, neighbour_correlation, affine_residual  # noqa: E402


def retention(movie, surface_um=8.0, lags_h=(0.5, 1.0, 2.0)):
    z = zipfile.ZipFile(W.ZIP)
    pos = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Position.csv")
    dist = W._csv(z, f"Tracks-epithelial/{movie}-denoised_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_1.csv")
    df = pos.rename(columns={"Position X": "x", "Position Y": "y", "Position Z": "z", "Time": "t"})
    df = df.merge(dist[["ID", "Shortest Distance to Surfaces"]].rename(
        columns={"Shortest Distance to Surfaces": "d"}), on="ID", how="left")
    df = df[["x", "y", "z", "t", "TrackID", "d"]].dropna(subset=["TrackID", "d"])
    frame_min = 4.9983                                               # wang_smg_cell_stats.json frame_min
    frames = np.sort(df["t"].unique())
    by_t = {int(t): g.set_index("TrackID") for t, g in df.groupby("t")}
    out = {"movie": movie, "surface_um": surface_um, "frame_min": frame_min, "frames": [int(frames[0]), int(frames[-1])],
           "estimator": "exp_measures.exp21.neighbour_retention (r_in 1.25, r_out 1.5 x median nn at t0)"}
    for lh in lags_h:
        k = int(round(lh * 60.0 / frame_min))
        vals, npairs, d0s = [], [], []
        for t0 in range(int(frames[0]), int(frames[-1]) - k + 1, max(1, k // 2)):
            a, b = by_t.get(t0), by_t.get(t0 + k)
            if a is None or b is None:
                continue
            ids = a.index.intersection(b.index)
            ids = ids[~ids.duplicated()]
            if len(ids) < 20:
                continue
            A = a.loc[ids]; B = b.loc[ids]
            A = A[~A.index.duplicated()]; B = B[~B.index.duplicated()]
            surf = A["d"].values > -surface_um
            # THE SPACING IS EVERY SURFACE NUCLEUS'S at t0, not the tracked subset's: tracks have gaps, and the
            # nuclei tracked at both ends are a sparser sample (their own nearest-neighbour distance read 16-22 um
            # against the tissue's 9.9), which would make second neighbours count as neighbours
            S0 = a[a["d"].values > -surface_um][["x", "y", "z"]].values
            from scipy.spatial import cKDTree
            d0_all = float(np.median(cKDTree(S0).query(S0, k=2)[0][:, 1]))
            kept, n_p, d0 = neighbour_retention(A[["x", "y", "z"]].values, B[["x", "y", "z"]].values, sel=surf,
                                                d0=d0_all)
            if kept is not None:
                vals.append(kept); npairs.append(n_p); d0s.append(d0)
        out[f"kept_{lh:g}h"] = {"mean": float(np.mean(vals)) if vals else None,
                                "sd_over_windows": float(np.std(vals)) if vals else None,
                                "windows": len(vals), "pairs_median": float(np.median(npairs)) if npairs else None,
                                "d0_um_median": float(np.median(d0s)) if d0s else None}
    return out


def motion_shape(movie, surface_um=8.0, lag_corr=6, lags_msd=(12, 36)):
    """The gland's numbers for exp 21's newer rulers, through THE SAME functions as the model's:
    corr_1sp      neighbour_correlation of 30-min (6-frame) displacements of surface nuclei, the affine flow of the
                  surface set removed (affine_residual), pairs 0.75-1.25 of the surface spacing, median over windows
    msd_ratio     MSD/lag at 3 h over MSD/lag at 1 h for surface tracks (drift removed per frame, as wang_smg_stats)
    prw_P_h_1h    the persistent-random-walk fit (exp11.prw_fit) over the first hour of the surface MSD"""
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
    cors = []
    for t0 in range(int(frames[0]), int(frames[-1]) - lag_corr + 1, lag_corr):
        a, b = by_t.get(t0), by_t.get(t0 + lag_corr)
        if a is None or b is None:
            continue
        ids = a.index.intersection(b.index)
        A = a.loc[ids]; B = b.loc[ids]
        srf = A["d"].values > -surface_um
        if srf.sum() < 30:
            continue
        S0 = a[a["d"].values > -surface_um][["x", "y", "z"]].values
        d0 = float(np.median(cKDTree(S0).query(S0, k=2)[0][:, 1]))
        X0 = A[["x", "y", "z"]].values[srf]; X1 = B[["x", "y", "z"]].values[srf]
        c = neighbour_correlation(X0, affine_residual(X0, X1), d0)
        if c is not None:
            cors.append(c)
    # MSD of surface tracks, drift removed per frame (the median step of every nucleus), lags up to 3 h
    df = df.sort_values(["TrackID", "t"])
    step = df.groupby("TrackID")[["x", "y", "z"]].diff()
    cont = df.groupby("TrackID")["t"].diff() == 1
    drift = step[cont].assign(t=df.loc[cont, "t"]).groupby("t").median()
    cum = drift.cumsum()
    Lmax = max(lags_msd)
    msd = np.zeros(Lmax + 1); cnt = np.zeros(Lmax + 1)
    for tid, blk in df.groupby("TrackID"):
        t = blk["t"].values.astype(int)
        if len(t) < 3 or np.any(np.diff(t) != 1):
            continue
        P = blk[["x", "y", "z"]].values - cum.reindex(t).ffill().fillna(0.0).values
        dd = blk["d"].values
        for L in range(1, min(Lmax, len(t) - 1) + 1):
            m = dd[:-L] > -surface_um
            r2 = ((P[L:] - P[:-L]) ** 2).sum(1)[m]
            msd[L] += r2.sum(); cnt[L] += m.sum()
    m = msd[1:] / np.maximum(cnt[1:], 1)
    dt_h = 4.9983 / 60.0
    lag = np.arange(1, Lmax + 1) * dt_h
    ratio = (m[lags_msd[1] - 1] / lag[lags_msd[1] - 1]) / (m[lags_msd[0] - 1] / lag[lags_msd[0] - 1])
    from exp_measures.exp11 import prw_fit
    f1 = prw_fit(lag[:lags_msd[0]], m[:lags_msd[0]])
    return {"corr_1sp": float(np.median(cors)) if cors else None, "corr_windows": len(cors),
            "msd_ratio_3h_1h": float(ratio), "prw_P_h_1h": float(f1["P"]), "prw_v_um_h_1h": float(f1["v"]),
            "msd_tracks_at_3h": int(cnt[lags_msd[1]]),
            "estimators": "exp_measures.exp21.neighbour_correlation + affine_residual; exp11.prw_fit over lags <= 1 h"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--movie", default="2020-01-25-K14R-HisG-2photon-25x-4")
    ap.add_argument("--surface_um", type=float, default=8.0)
    ap.add_argument("--json", default=os.path.join(ROOT, "experiments/exp21_mpm_epithelium/data",
                                                   "wang_surface_neighbours.json"))
    a = ap.parse_args()
    out = retention(a.movie, a.surface_um)
    # THE GLAND'S OWN MOTION NUMBERS beside it, read (not recomputed) from exp 11's reader of the same tracks, so a
    # gate can name one file: the surface nuclei's MSD at a 1 h lag (12 lags of the movie's ~5-min frames)
    st = json.load(open(os.path.join(ROOT, "experiments/exp11_bm_hole_budding/data/wang_smg_cell_stats.json")))
    srf = st["surface"]
    out["surface_motion_shape"] = motion_shape(a.movie, a.surface_um)
    out["surface_motion"] = {"from": "experiments/exp11_bm_hole_budding/data/wang_smg_cell_stats.json (tools/wang_smg_stats.py)",
                             "speed_um_h_median": srf["speed_um_h_median"], "prw_P_h": srf["prw_P_h"],
                             "msd_1h_um2": srf["msd_um2"][11], "nn_um_median_3d": srf["nn_um_median"]}
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", a.json)


if __name__ == "__main__":
    main()

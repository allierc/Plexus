"""Cell size, density and motion of the E13 salivary gland epithelium, read off Wang et al. 2021's own
Imaris tracks (Figshare source data, `imaris-overall-tracking-data`), for exp 11's interior cells.

    python tools/wang_smg_stats.py [--movie 2020-01-25-K14R-HisG-2photon-25x-4] [--json out.json]

What it reads, all from the zip in experiments/exp11_bm_hole_budding/data/Wang_2021_figshare_source_data:
  Tracks-epithelial/<movie>_Position.csv                 every epithelial (K14-RFP+) nucleus, per frame, um
  Tracks-epithelial/<movie>_Shortest_Distance_to_...csv  its signed distance to the epithelial surface
                                                         (Imaris: negative = inside the epithelium), um
  Surface-statistics/<movie>_Volume.csv / _Area.csv      that surface's volume (um^3) and area (um^2)

What it reports, for SURFACE cells (a nucleus within `--surface_um` of the surface) and INTERIOR cells
(deeper than `--interior_um`), both in um / h and in exp 11's model units (1 length unit = 10 um,
1 frame = 10 min, `general.units` of config/tissue/exp11_w2_base.yaml):
  density     epithelial nuclei per volume of the surface = 1 / (volume per cell), its equivalent
              sphere diameter, and each region's nearest-neighbour nucleus distance
  motion      after removing the whole gland's drift (per frame, the median step of every tracked
              nucleus): instantaneous speed, the spread of per-track mean speeds across cells (CV),
              the turning angle between successive steps, and the mean squared displacement against
              lag, fitted by the persistent random walk MSD = 2 v^2 P (t - P (1 - exp(-t / P)))
              (Furth 1920; the 3D form with <v(0).v(t)> = v^2 exp(-t / P)), giving the speed v and
              the persistence time P.
A nucleus's region is read at the START of each step or lag window, because cells move between the
two (Wang's Type I divisions dive and return, Fig 2A-B).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.path.join(ROOT, "experiments/exp11_bm_hole_budding/data/Wang_2021_figshare_source_data",
                   "Data-for-plots-Wang-2020.zip")
BASE = "data/imaris-overall-tracking-data/"
UM_PER_UNIT, MIN_PER_FRAME = 10.0, 10.0          # exp 11: length_um 10, time_s 600


def _csv(z, name):
    """An Imaris statistics export: three header lines, then a table ending in an empty column."""
    raw = z.read(BASE + name).decode("latin1")
    df = pd.read_csv(io.StringIO(raw), skiprows=3)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _prw_fit(lag_h, msd):
    """`exp_measures.exp11.prw_fit` -- ONE estimator for the gland's tracks and the model's cells."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from exp_measures.exp11 import prw_fit
    f = prw_fit(lag_h, msd)
    return {"v_um_h": f["v"], "P_h": f["P"], "r2": f["r2"], "noise_sigma_um": f["noise_sigma"]}


def stats(movie, surface_um=8.0, interior_um=15.0, max_lag=24, min_steps=12):
    z = zipfile.ZipFile(ZIP)
    pos = _csv(z, f"Tracks-epithelial/{movie}-denoised_Position.csv")
    dist = _csv(z, f"Tracks-epithelial/{movie}-denoised_Shortest_Distance_to_Surfaces_Surfaces=Surfaces_1.csv")
    vol = _csv(z, f"Surface-statistics/{movie}-denoised_Volume.csv")
    area = _csv(z, f"Surface-statistics/{movie}-denoised_Area.csv")
    tim = _csv(z, f"Tracks-all/{movie}-denoised_Time.csv") if movie.endswith("25x-4") else None
    df = pos.rename(columns={"Position X": "x", "Position Y": "y", "Position Z": "z", "Time": "t"})
    df = df.merge(dist[["ID", "Shortest Distance to Surfaces"]].rename(
        columns={"Shortest Distance to Surfaces": "d"}), on="ID", how="left")
    df = df[["x", "y", "z", "t", "TrackID", "ID", "d"]].dropna(subset=["TrackID"])
    # THE FRAME INTERVAL, from the time stamps where they exist (Time.csv: hours per spot), else 5 min
    if tim is not None:
        tt = tim.groupby("Time.1")["Time"].first()
        dt_h = float(np.median(np.diff(tt.values) / np.diff(tt.index.values)))
    else:
        dt_h = 5.0 / 60.0
    frames = np.sort(df["t"].unique())

    # ---- density: the surface's volume over the nuclei inside it, per frame
    vol = vol.rename(columns={"Volume": "V", "Time": "t"}).groupby("t")["V"].sum()
    area = area.rename(columns={"Area": "A", "Time": "t"}).groupby("t")["A"].sum()
    n_t = df.groupby("t").size()
    common = n_t.index.intersection(vol.index)
    v_cell = (vol.loc[common] / n_t.loc[common])
    mid = int(common[len(common) // 2])
    from scipy.spatial import cKDTree
    f0 = df[df["t"] == mid]
    X0 = f0[["x", "y", "z"]].values
    nn = cKDTree(X0).query(X0, k=2)[0][:, 1]
    surf0, int0 = (f0["d"].values > -surface_um), (f0["d"].values < -interior_um)

    # ---- motion: steps between consecutive frames of one track, drift removed per frame
    df = df.sort_values(["TrackID", "t"])
    g = df.groupby("TrackID")
    nx = df[["x", "y", "z", "t"]].values
    same = (df["TrackID"].values[1:] == df["TrackID"].values[:-1]) & (np.diff(df["t"].values) == 1)
    step = (nx[1:, :3] - nx[:-1, :3])[same]
    t_of = nx[:-1, 3][same]
    d_of = df["d"].values[:-1][same]
    tid_of = df["TrackID"].values[:-1][same]
    drift = pd.DataFrame(step, columns=["dx", "dy", "dz"]).assign(t=t_of).groupby("t").median()
    step_c = step - drift.loc[t_of].values
    spd = np.linalg.norm(step_c, axis=1) / dt_h                       # um / h
    reg = {"surface": d_of > -surface_um, "interior": d_of < -interior_um}
    # turning angle between successive corrected steps of one track
    s_df = pd.DataFrame(step_c, columns=["dx", "dy", "dz"]).assign(tid=tid_of, t=t_of, d=d_of)
    s_df = s_df.sort_values(["tid", "t"])
    a = s_df[["dx", "dy", "dz"]].values
    ok = (s_df["tid"].values[1:] == s_df["tid"].values[:-1]) & (np.diff(s_df["t"].values) == 1)
    na = np.linalg.norm(a, axis=1)
    cosang = (a[1:] * a[:-1]).sum(1) / np.maximum(na[1:] * na[:-1], 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))[ok]
    dang = s_df["d"].values[:-1][ok]
    # per-track mean speed, for tracks of at least `min_steps` steps
    tr = pd.DataFrame({"tid": tid_of, "v": spd, "d": d_of}).groupby("tid").agg(v=("v", "mean"), n=("v", "size"),
                                                                                d=("d", "median"))
    tr = tr[tr["n"] >= min_steps]
    # MSD by lag, drift removed: cumulative corrected displacement along each track
    cum_drift = drift.cumsum()
    out = {"movie": movie, "frame_min": dt_h * 60.0, "frames": [int(frames[0]), int(frames[-1])],
           "surface_um": surface_um, "interior_um": interior_um}
    msd = {k: np.zeros(max_lag + 1) for k in reg}; cnt = {k: np.zeros(max_lag + 1) for k in reg}
    for tid, blk in g:
        if len(blk) < 3:
            continue
        t = blk["t"].values.astype(int)
        if np.any(np.diff(t) != 1):
            continue
        P = blk[["x", "y", "z"]].values
        cd = cum_drift.reindex(t).ffill().fillna(0.0).values
        cd = cd - cd[0]
        # position with the drift of every earlier step removed
        Pc = P - np.vstack([np.zeros(3), cd[:-1]])
        dd = blk["d"].values
        for L in range(1, min(max_lag, len(t) - 1) + 1):
            r2 = ((Pc[L:] - Pc[:-L]) ** 2).sum(1)
            for k, lo in (("surface", dd[:-L] > -surface_um), ("interior", dd[:-L] < -interior_um)):
                msd[k][L] += r2[lo].sum(); cnt[k][L] += lo.sum()
    lag_h = np.arange(max_lag + 1) * dt_h
    for k, m in reg.items():
        mk = msd[k][1:] / np.maximum(cnt[k][1:], 1)
        fit = _prw_fit(lag_h[1:], mk)
        tk = tr[(tr["d"] > -surface_um) if k == "surface" else (tr["d"] < -interior_um)]
        sel = (surf0 if k == "surface" else int0)
        out[k] = {
            "n_steps": int(m.sum()),
            "speed_um_h_median": float(np.median(spd[m])),
            "speed_um_h_iqr": [float(np.percentile(spd[m], 25)), float(np.percentile(spd[m], 75))],
            "track_mean_speed_um_h_median": float(tk["v"].median()) if len(tk) else None,
            "track_mean_speed_cv": float(tk["v"].std() / tk["v"].mean()) if len(tk) > 2 else None,
            "n_tracks": int(len(tk)),
            "turn_deg_median": float(np.median(ang[(dang > -surface_um) if k == "surface" else (dang < -interior_um)])),
            "msd_um2": [float(x) for x in mk[:12]],
            "prw_v_um_h": fit["v_um_h"], "prw_P_h": fit["P_h"], "prw_r2": fit["r2"],
            "prw_noise_sigma_um": fit["noise_sigma_um"],
            "nn_um_median": float(np.median(nn[sel])) if sel.any() else None,
            "n_nuclei_mid_frame": int(sel.sum()),
        }
    vc = float(v_cell.median())
    out["density"] = {"cell_volume_um3_median": vc,
                      "cell_eq_diameter_um": float((6.0 * vc / np.pi) ** (1.0 / 3.0)),
                      "nuclei_per_1000um3": 1000.0 / vc,
                      "gland_volume_um3_first_last": [float(vol.loc[common[0]]), float(vol.loc[common[-1]])],
                      "gland_area_um2_first_last": [float(area.loc[common[0]]), float(area.loc[common[-1]])],
                      "nuclei_first_last": [int(n_t.loc[common[0]]), int(n_t.loc[common[-1]])],
                      "nn_um_median_all": float(np.median(nn)), "mid_frame": mid}
    # the same numbers in exp 11's units: 1 unit = 10 um, 1 frame = 10 min
    mu = {}
    for k in reg:
        o = out[k]
        mu[k] = {"speed_units_per_frame": o["speed_um_h_median"] / UM_PER_UNIT * MIN_PER_FRAME / 60.0,
                 "prw_v_units_per_frame": o["prw_v_um_h"] / UM_PER_UNIT * MIN_PER_FRAME / 60.0,
                 "prw_P_frames": o["prw_P_h"] * 60.0 / MIN_PER_FRAME,
                 "nn_units": (o["nn_um_median"] or np.nan) / UM_PER_UNIT}
    mu["cell_volume_units3"] = vc / UM_PER_UNIT ** 3
    out["model_units"] = mu
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--movie", default="2020-01-25-K14R-HisG-2photon-25x-4")
    ap.add_argument("--surface_um", type=float, default=8.0)
    ap.add_argument("--interior_um", type=float, default=15.0)
    ap.add_argument("--json", default=os.path.join(ROOT, "experiments/exp11_bm_hole_budding/data",
                                                   "wang_smg_cell_stats.json"))
    a = ap.parse_args()
    out = stats(a.movie, a.surface_um, a.interior_um)
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("surface", "interior")}, indent=1))
    for k in ("surface", "interior"):
        print(k, json.dumps({kk: vv for kk, vv in out[k].items() if kk != "msd_um2"}))
    print("wrote", a.json)


if __name__ == "__main__":
    main()

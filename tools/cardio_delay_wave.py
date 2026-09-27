"""Is a fitted per-cell excitation-delay map a travelling wave? (exp06 Stage 0, 2026-09-26)

Reads the per-cell delay `delay` (frames of 1/24 s) of a cardio_mpm strain fit and the cells'
centroids, keeps the interior cells the fit scored (`interior`), and asks three things:

  plane wave     delay = a + b.x          R^2 of the fit, and the arrival spread it implies across
                                          the field (|b| x field width), in ms
  point source   delay = a + |x - x0| / v R^2 of the best of 20 starts
  coherence      corr(delay_i, mean delay of cell i's 6 nearest neighbours), against the same
                 number for 200 shuffles of the delays over the cells

A front slower than one field width per frame would show as a plane-wave or point-source R^2 well
above 0 and an arrival spread of several frames. Speeds are in field widths per second: the
recording carries no micrometre scale.

    python tools/cardio_delay_wave.py [--out experiments/exp06_excitation_wave/data/delay_wave.json]
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

STRAIN = "/workspace/Plexus/prototype/cardio_mpm/strain"
SHEETS = {"healthy": ("out/fits/healthy_allbeats/params.npz", "data/cell_centroids_world.npy"),
          "hcm": ("out/fits/hcm_allbeats/params.npz", "data_hcm/cell_centroids_world.npy")}
FRAME_S = 1.0 / 24.0          # the recordings are 24 frames a second (strain/SUMMARY.md)


def check(params, centroids, seed=0):
    p = np.load(params)
    m = p["interior"]
    d = p["delay"][m] * FRAME_S
    x = np.load(centroids)[:, :2][m]
    width = float((x.max(0) - x.min(0)).max())
    A = np.c_[np.ones(len(d)), x]
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    r2_plane = 1.0 - (d - A @ coef).var() / d.var()
    spread_s = float(np.hypot(*coef[1:]) * width)

    def res(q):
        return q[0] + np.hypot(x[:, 0] - q[1], x[:, 1] - q[2]) * q[3] - d
    starts = x[np.random.default_rng(seed).choice(len(x), 20)]
    best = min((least_squares(res, [d.min(), *s_, 1.0]) for s_ in starts), key=lambda o: o.cost)
    r2_point = 1.0 - (best.fun ** 2).mean() / d.var()

    nb = cKDTree(x).query(x, k=7)[1][:, 1:]
    rho = float(np.corrcoef(d, d[nb].mean(1))[0, 1])
    rng = np.random.default_rng(seed + 1)
    null = []
    for _ in range(200):
        s = rng.permutation(d)
        null.append(np.corrcoef(s, s[nb].mean(1))[0, 1])
    return {"cells": int(m.sum()), "field_width_world": width, "frame_ms": FRAME_S * 1e3,
            "delay_sd_ms": float(d.std() * 1e3), "delay_range_ms": float(np.ptp(d) * 1e3),
            "plane_wave_r2": float(r2_plane), "plane_wave_spread_across_field_ms": spread_s * 1e3,
            "point_source_r2": float(r2_point),
            "neighbour_corr": rho, "neighbour_corr_shuffled_mean": float(np.mean(null)),
            "neighbour_corr_shuffled_sd": float(np.std(null))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/Plexus/experiments/exp06_excitation_wave/data/delay_wave.json")
    a = ap.parse_args()
    out = {k: check(os.path.join(STRAIN, f), os.path.join(STRAIN, c)) for k, (f, c) in SHEETS.items()}
    for k, v in out.items():
        print(f"{k:8s} cells {v['cells']}  delay sd {v['delay_sd_ms']:.0f} ms  plane-wave R2 {v['plane_wave_r2']:.3f} "
              f"(spread {v['plane_wave_spread_across_field_ms']:.0f} ms over the field, frame {v['frame_ms']:.1f} ms)  "
              f"point-source R2 {v['point_source_r2']:.3f}  neighbour corr {v['neighbour_corr']:.2f} "
              f"(shuffled {v['neighbour_corr_shuffled_mean']:.2f} +- {v['neighbour_corr_shuffled_sd']:.2f})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

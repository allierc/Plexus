"""geom_control.py -- how much of the per-cell amplitude field is GEOMETRY rather than theta?

r2_percell reads 0.963 for a per-cell-blind constant. This asks why: regress the reference's
per-cell peak-amplitude field on quantities that contain no per-cell parameter at all -- the cell's
distance from the centre of the sheet, and its size. Nothing here is fitted to theta.

usage: PYTHONPATH=/workspace/Plexus/src python geom_control.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, HERE)

from assemble import System                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    torch.manual_seed(0)
    with torch.no_grad():
        sy = System(device=a.device, n_cells=100, per_parent=100, n_grid=128, warmup=0,
                    dtype="float64", mode="full")
        cid = sy.cid
        x = sy.p.get("pos")
        C = sy.C
        cen = np.zeros((C, 2))
        siz = np.zeros(C)
        n = np.zeros(C, dtype=int)
        for c in range(1, C + 1):
            m = cid == c
            n[c - 1] = int(m.sum())
            cen[c - 1] = x[m].mean(0).cpu().numpy()
            siz[c - 1] = float((x[m] - x[m].mean(0)).norm(dim=1).mean())
        E_true = sy.E_true[1:].cpu().numpy()
        gain_true = sy.gain_true[1:].cpu().numpy()

    S = json.load(open(os.path.join(HERE, "crash_round2_s0.json")))
    a_ref = np.array(S["a_ref_percell"], dtype=float)
    keep = np.array(S["keep_percell"], dtype=bool)
    r = np.linalg.norm(cen - 0.5, axis=1)
    y = a_ref[keep] / a_ref[keep].mean()

    def r2_from(X):
        X = np.column_stack([np.ones(X.shape[0]), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return float(1.0 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())

    out = {
        "n_cells_kept": int(keep.sum()),
        "corr_a_vs_radius": float(np.corrcoef(y, r[keep])[0, 1]),
        "corr_a_vs_cellsize": float(np.corrcoef(y, siz[keep])[0, 1]),
        "corr_a_vs_E_true": float(np.corrcoef(y, E_true[keep])[0, 1]),
        "corr_a_vs_gain_true": float(np.corrcoef(y, gain_true[keep])[0, 1]),
        "R2_of_radius_alone": r2_from(r[keep][:, None]),
        "R2_of_radius_quadratic": r2_from(np.column_stack([r[keep], r[keep] ** 2])),
        "R2_of_xy_quadratic": r2_from(np.column_stack([
            cen[keep, 0], cen[keep, 1], cen[keep, 0] ** 2, cen[keep, 1] ** 2,
            cen[keep, 0] * cen[keep, 1]])),
        "R2_of_theta_true_alone": r2_from(np.column_stack([E_true[keep], gain_true[keep]])),
        "R2_of_geometry_plus_theta": r2_from(np.column_stack([
            cen[keep, 0], cen[keep, 1], cen[keep, 0] ** 2, cen[keep, 1] ** 2,
            cen[keep, 0] * cen[keep, 1], E_true[keep], gain_true[keep]])),
        "blind_constant_r2_percell": 0.9633,
    }
    json.dump(out, open(os.path.join(HERE, "geom_control.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:<32s} {v}")


if __name__ == "__main__":
    main()

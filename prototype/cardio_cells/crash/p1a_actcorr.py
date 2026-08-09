#!/usr/bin/env python
"""p1a_actcorr.py -- PROBE A, follow-up. WHY the per-cell gain column is small where it is small.

p1a_percell.py measured the per-cell sensitivity of every cell but stored the cell covariates only
for the six named ones. This rebuilds the pulse-tick system (nothing else changes: same seed, same
planted theta) and records, per cell, the covariate that the Gaussian stimulus makes decisive --
how much active force the cell actually receives -- then ranks it against the measured per-cell
sensitivities.

The point is not a correlation coefficient. It is that `activation_pulse` writes
exp(-r^2 / 2 sigma^2) with sigma = 0.12 about (0.5, 0.5), so a cell's GAIN multiplies a force that
is four orders of magnitude smaller at the corner than at the centre. A per-cell gain is not one
parameter repeated 100 times; it is 100 parameters with 100 different, and mostly negligible,
levers on the data.

usage:
  PYTHONPATH=/workspace/Plexus/src python p1a_actcorr.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("/workspace/Plexus/src", ALG, HERE, DISC):
    sys.path.insert(0, p)

import crash_test as CT                                               # noqa: E402

PX = 1024.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--e-lo", type=float, default=40.0)
    ap.add_argument("--e-hi", type=float, default=220.0)
    ap.add_argument("--g-lo", type=float, default=0.5)
    ap.add_argument("--g-hi", type=float, default=1.5)
    ap.add_argument("--ladder", default=os.path.join(HERE, "p1a_percell.json"))
    ap.add_argument("--tag", default="p1a")
    args = ap.parse_args()

    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, print)
        C, cid, x0 = sy.C, sy.cid, sy.x0
        cnt = torch.bincount(cid, minlength=C + 1).clamp(min=1).double()
        cxy = torch.stack([torch.bincount(cid, weights=x0[:, k].double(), minlength=C + 1) / cnt
                           for k in (0, 1)], 1)
        d_wall = torch.minimum(torch.minimum(cxy[:, 0], 1 - cxy[:, 0]),
                               torch.minimum(cxy[:, 1], 1 - cxy[:, 1]))
        d_ctr = (cxy - 0.5).norm(dim=1)
        actn = sy.act0.norm(dim=1).double()
        act_cell = torch.bincount(cid, weights=actn, minlength=C + 1) / cnt
        cov = {"cell": list(range(1, C + 1)),
               "cx": cxy[1:, 0].cpu().numpy(), "cy": cxy[1:, 1].cpu().numpy(),
               "d_wall": d_wall[1:].cpu().numpy(), "d_center": d_ctr[1:].cpu().numpy(),
               "act": act_cell[1:].cpu().numpy(),
               "E_true": sy.E_true[1:].cpu().numpy(),
               "gain_true": sy.gain_true[1:].cpu().numpy(),
               "n_particles": torch.bincount(cid, minlength=C + 1)[1:].cpu().numpy()}

    d = json.load(open(args.ladder))
    per = d["ladder"]["pulse"]["per_cell"]
    sens = {}
    for cad in ("substep", "frame"):
        for p in ("E", "gain"):
            sens[f"{cad}_{p}"] = np.array([per[str(c)][p][cad]["max_px"] for c in cov["cell"]])

    a = cov["act"]
    print(f"\n  active-force delta per cell at the pulse peak (tick {args.warmup}): "
          f"min {a.min():.4g}  median {np.median(a):.4g}  max {a.max():.4g}  "
          f"max/min {a.max()/max(a.min(),1e-300):.4g}")
    for q in (0.001, 0.01, 0.1):
        print(f"    cells receiving < {q:.3g} of the strongest cell's active force: "
              f"{int((a < q * a.max()).sum())}/{len(a)}")
    print(f"\n  {'sensitivity':<16s} {'rho vs act':>11s} {'rho vs d_center':>16s} "
          f"{'rho vs d_wall':>14s} {'rho vs own param':>17s}")
    out = {}
    for k, v in sens.items():
        own = cov["E_true"] if k.endswith("_E") else cov["gain_true"]
        row = {"rho_act": spearman(a, v), "rho_d_center": spearman(cov["d_center"], v),
               "rho_d_wall": spearman(cov["d_wall"], v), "rho_own_param": spearman(own, v),
               "median_px": float(np.median(v)), "max_px": float(v.max())}
        out[k] = row
        print(f"  {k:<16s} {row['rho_act']:>11.3f} {row['rho_d_center']:>16.3f} "
              f"{row['rho_d_wall']:>14.3f} {row['rho_own_param']:>17.3f}")

    np.savez(os.path.join(HERE, f"{args.tag}_covariates.npz"),
             **{k: np.asarray(v) for k, v in cov.items()},
             **{f"sens::{k}": v for k, v in sens.items()})
    json.dump({"tick": args.warmup, "act_stats": {"min": float(a.min()),
                                                  "median": float(np.median(a)),
                                                  "max": float(a.max())},
               "n_below_1e-3_of_max": int((a < 1e-3 * a.max()).sum()),
               "n_below_1e-2_of_max": int((a < 1e-2 * a.max()).sum()),
               "n_below_1e-1_of_max": int((a < 1e-1 * a.max()).sum()),
               "spearman": out},
              open(os.path.join(HERE, f"{args.tag}_actcorr.json"), "w"), indent=1)
    print(f"\nwrote {HERE}/{args.tag}_actcorr.json")


if __name__ == "__main__":
    main()

"""refute3_interp.py -- the t0 sweep says F_lerp only works where F is momentarily stationary.
Is the culprit the INTERPOLATION (fixable with more measured frames) or the injection itself?

refute_round3.py phase B, frame cadence, displacement read-out, med|dE/E|:

    t0      |dF|/frame   none    F_hold  F_lerp   F_true(oracle)
    45      1.9e-3      0.3312   0.1048  0.0035   0.0049
    75      1.8e-2      0.2050   0.6219  0.3191   0.0029
    105     2.1e-2      0.2141   0.6202  0.2796   0.0029
    120     2.2e-2      0.2158   0.6197  0.2822   0.0030
    135     2.2e-2      0.2197   0.6196  0.2833   0.0030
    150     2.2e-2      0.2228   0.6196  0.2838   0.0030
    165     2.3e-3      0.2572   0.1401  0.0078   0.0091

The SUBSTEP-RESOLVED F works everywhere (0.003).  The two-frame LINEAR interpolation works only at
the two ticks where F barely moves.  A recording gives F at every frame, so the reconstruction of
F inside a frame does not have to be linear in two samples: it can be quadratic in three or a
Catmull-Rom cubic in four.  This measures whether that closes the gap.

usage: PYTHONPATH=/workspace/Plexus/src python refute3_interp.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
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

import crash_test as CT                                         # noqa: E402
from finject import record_substeps, hold, lerp                 # noqa: E402
from refute_round3 import advance, fit                          # noqa: E402


def tau(n, device, dtype):
    """the within-frame time of substep s, in units of a frame: F is read AFTER strain s."""
    return (torch.arange(1, n + 1, device=device, dtype=dtype) / n).view(n, 1, 1, 1)


def quad3(Pm1, P0, P1, n):
    """Lagrange through F(t-1), F(t), F(t+1) at nodes -1, 0, 1, evaluated at tau in (0, 1]."""
    t = tau(n, P0.device, P0.dtype)
    return 0.5 * t * (t - 1) * Pm1[None] + (1 - t * t) * P0[None] + 0.5 * t * (t + 1) * P1[None]


def catmull(Pm1, P0, P1, P2, n):
    t = tau(n, P0.device, P0.dtype)
    return 0.5 * ((2 * P0[None])
                  + (-Pm1[None] + P1[None]) * t
                  + (2 * Pm1[None] - 5 * P0[None] + 4 * P1[None] - P2[None]) * t * t
                  + (-Pm1[None] + 3 * P0[None] - 3 * P1[None] + P2[None]) * t * t * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="refute3_interp")
    ap.add_argument("--sigma-F", type=float, default=0.0)
    a = ap.parse_args()
    T0S = (75, 105, 120, 135, 150, 165, 180)
    START = 40
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=START,
                           window=START, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "t0s": list(T0S), "sigma_F": a.sigma_F}
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        cur = START
        SNAP = ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")

        def goto(tick):
            nonlocal cur
            sy.restore()
            advance(sy, cur, tick)
            sy._snapshot(tick)
            cur = tick

        log("\n[interp] reconstructing F INSIDE the frame from 2, 3 or 4 measured frame samples")
        log(f"    {'t0':>5s} {'|dF|/frm':>9s} | {'err_lin':>9s} {'err_quad':>9s} {'err_cubic':>9s}"
            f" | {'none':>8s} {'linear':>8s} {'quad':>8s} {'cubic':>8s} {'oracle':>8s}")
        R["rows"] = {}
        for t0 in T0S:
            # frame-boundary samples F(t0-1), F(t0), F(t0+1), F(t0+2) and the substep truth
            goto(t0 - 1)
            Fm1 = sy.F0.clone()
            goto(t0)
            F_0 = sy.F0.clone()
            Fs, Cs, Xs = record_substeps(sy, n)
            x0, x_next = sy.x0.clone(), Xs[-1].clone()
            snap = {k: getattr(sy, k).clone() for k in SNAP}
            F_1 = Fs[-1].clone()
            goto(t0 + 1)
            Fs2, _, _ = record_substeps(sy, n)
            F_2 = Fs2[-1].clone()
            snap_next = {k: getattr(sy, k).clone() for k in SNAP}
            for k, v in snap.items():                       # back to the fit frame
                setattr(sy, k, v)

            cand = {"linear": lerp(F_0, F_1, n),
                    "quad": quad3(Fm1, F_0, F_1, n),
                    "cubic": catmull(Fm1, F_0, F_1, F_2, n)}
            if a.sigma_F > 0:
                g = torch.Generator(device=sy.device).manual_seed(11 + t0)
                e = {}
                for nm, P in (("m1", Fm1), ("0", F_0), ("1", F_1), ("2", F_2)):
                    e[nm] = (a.sigma_F / 2.0) * torch.randn(P.shape, generator=g,
                                                            device=P.device, dtype=P.dtype)
                cand = {"linear": lerp(F_0 + e["0"], F_1 + e["1"], n),
                        "quad": quad3(Fm1 + e["m1"], F_0 + e["0"], F_1 + e["1"], n),
                        "cubic": catmull(Fm1 + e["m1"], F_0 + e["0"], F_1 + e["1"],
                                         F_2 + e["2"], n)}
            row = {"dF_frame_abs": float((F_1 - F_0).norm(dim=(-2, -1)).median()),
                   "err_of_reconstruction": {k: float((v - Fs).norm() / Fs.norm())
                                             for k, v in cand.items()}}
            s, _ = fit(sy, n, None, x_next, x0, th, C)
            row["none"] = s
            for k, v in cand.items():
                s, _ = fit(sy, n, v, x_next, x0, th, C)
                row[k] = s
            s, _ = fit(sy, n, Fs, x_next, x0, th, C)
            row["oracle"] = s
            R["rows"][str(t0)] = row
            for k, v in snap_next.items():                  # cur is t0+1: leave the walk there
                setattr(sy, k, v)
            er = row["err_of_reconstruction"]
            log(f"    {t0:>5d} {row['dF_frame_abs']:>9.2e} | {er['linear']:>9.2e} "
                f"{er['quad']:>9.2e} {er['cubic']:>9.2e} | {row['none']['med_E']:>8.4f} "
                f"{row['linear']['med_E']:>8.4f} {row['quad']['med_E']:>8.4f} "
                f"{row['cubic']['med_E']:>8.4f} {row['oracle']['med_E']:>8.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

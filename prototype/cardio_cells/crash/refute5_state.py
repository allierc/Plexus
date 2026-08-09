"""refute5_state.py -- how big is the LAST state oracle in the round-5 harness?

`System.restore()` (algebraic/assemble.py:225) resets p.state (positions AND velocities) and
p.F, p.C, p.Jp to the TRUE simulator snapshot at the frame start.  round5_fit.py calls it (through
assemble_inj -> y_of -> step_inj, finject.py:72) for every one of the 401 assemblies per frame, so
every column of A and the base term y0 are built from the true v0, C0 and Jp0.  Only F is treated
as measured.

Round 3's diagnosis measured the cost of removing that oracle AT ZERO NOISE (0.0078 -> 0.0125 for
C, -> 0.0227 for v, -> 0.0404 for both) and it was never re-measured with a noisy F.  This script
measures the input side of the problem: how wrong is C <- Fdot F^-1 and v <- a centred difference
when Fdot and the positions carry the recording's own noise?

usage: PYTHONPATH=/workspace/Plexus/src python refute5_state.py --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

import crash_test as CT                                          # noqa: E402
from finject import record_substeps                              # noqa: E402
from refute_round3 import advance                                # noqa: E402
from round5_fit import SIGMA_F, SIGMA_X, SNAP                    # noqa: E402


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--t0", type=int, default=165)
    a = ap.parse_args()
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    torch.manual_seed(0)
    R = {"t0": a.t0, "sigma_F": SIGMA_F, "sigma_x": SIGMA_X}
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, print)
        n, dt = sy.n_sub_per_frame, sy.dt

        # the true state at the frame start, and the true F at the two frame boundaries
        x_prev = sy.x0.clone()
        sy.restore()
        advance(sy, a.t0, a.t0 + 1)                # one frame forward, to get a centred difference
        x_next_prev = sy.p.get("pos").clone()
        sy.restore()
        Fs, Cs, Xs = record_substeps(sy, n)
        F0, F1 = sy.F0.clone(), Fs[-1].clone()
        C0, v0, x0 = sy.C0.clone(), sy.v0.clone(), sy.x0.clone()

        # the frame BEFORE t0, so a centred difference exists
        sy2, _ = CT.plant_and_warm(SimpleNamespace(**{**vars(args), "warmup": a.t0 - 1}), print)
        x_before = sy2.x0.clone()
        Fb, Cb, Xb = record_substeps(sy2, n)
        F_before = sy2.F0.clone()

        g = torch.Generator(device=sy.device).manual_seed(11)

        def dF(shape):
            return (SIGMA_F / 2.0) * torch.randn(shape, generator=g, device=sy.device,
                                                 dtype=sy.dtype)

        def dx(shape):
            return SIGMA_X * torch.randn(shape, generator=g, device=sy.device, dtype=sy.dtype)

        # ---- C from Fdot F^-1 ------------------------------------------------------------------
        Fi = torch.linalg.inv(F0)
        C_clean = ((F1 - F0) / dt) @ Fi
        F0h, F1h, Fbh = F0 + dF(F0.shape), F1 + dF(F0.shape), F_before + dF(F0.shape)
        C_noisy = ((F1h - F0h) / dt) @ torch.linalg.inv(F0h)
        C_noisy_central = ((F1h - Fbh) / (2 * dt)) @ torch.linalg.inv(F0h)
        R["C_oracle"] = {
            "norm_C0_true": float(C0.norm()),
            "rel_err_C_from_Fdot_clean": rel(C_clean, C0),
            "rel_err_C_from_Fdot_noisyF": rel(C_noisy, C0),
            "rel_err_C_from_Fdot_noisyF_central": rel(C_noisy_central, C0),
            "note": "C <- Fdot F^-1; the noisy version differences two measured F at the "
                    "recording's own sigma_F, so its error is sigma_F*sqrt(2)/dt"}

        # ---- v from a centred difference of the measured positions ------------------------------
        v_central_clean = (x_next_prev - x_before) / (2 * dt)
        v_central_noisy = ((x_next_prev + dx(x0.shape)) - (x_before + dx(x0.shape))) / (2 * dt)
        v_back_clean = (x0 - x_before) / dt
        R["v_oracle"] = {
            "norm_v0_true": float(v0.norm()),
            "rel_err_v_central_clean": rel(v_central_clean, v0),
            "rel_err_v_central_noisy": rel(v_central_noisy, v0),
            "rel_err_v_backward_clean": rel(v_back_clean, v0)}

        R["signal"] = {"F_relchange_over_frame": rel(F1, F0),
                       "sigma_F_percomponent_over_dF": float(
                           (SIGMA_F / 2.0) * np.sqrt(F0.numel()) / float((F1 - F0).norm()))}

    json.dump(R, open(os.path.join(HERE, "refute5_state.json"), "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()

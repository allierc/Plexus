"""freal_probe.py -- diagnostics for freal_derivedF's stage d, before anything expensive is run.

Three questions the smoke run raised and cannot answer:
  1. Is the 0.16 "kinematic floor" a TRANSPOSE / convention error?  (compare Fd, Fd^T, and both
     against p.F and p.F^T)
  2. Is it accumulated DRIFT of the solver's integrated F away from grad_X x?  (measure it as a
     function of how many frames have elapsed since the reference configuration)
  3. Is it a WALL artefact?  (interior mask vs everything)
"""
from __future__ import annotations

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

import torch  # noqa: E402
from assemble import SUBSTEP_TOKENS  # noqa: E402
from recover import install_E  # noqa: E402
import freal_derivedF as FD  # noqa: E402

PX = FD.PX


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    args = SimpleNamespace(device=dev, cells=100, per_parent=100, n_grid=128, warmup=0,
                           window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)

    def log(s):
        print(s, flush=True)

    t0 = time.time()
    with torch.no_grad():
        sy, X, F0 = FD.plant_and_warm_x0(args, log)
        I2 = torch.eye(2, device=X.device, dtype=X.dtype)[None]
        band = 0.06 / 0.70
        interior = ~((X[:, 0] < band) | (X[:, 0] > 1 - band)
                     | (X[:, 1] < band) | (X[:, 1] > 1 - band))
        log(f"[mask] interior {int(interior.sum())}/{X.shape[0]}")
        cg = FD.ControlGrid(X, 30.0 * PX)
        p_sp = 0.00996

        install_E(sy, sy.E_true)
        log(f"\n{'tick':>5s} {'med|F-I|':>9s} {'med|Fkin-I|':>11s} "
            f"{'rel(Fkin,F)':>11s} {'rel(FkinT,F)':>12s} {'rel(Fkin,FT)':>12s} "
            f"{'scale':>7s} {'corr':>7s} | {'INTERIOR rel':>12s} {'scale':>7s} {'corr':>7s} "
            f"| {'grid rel':>9s} {'gr int':>8s}")
        for tick in range(200):
            sy._outer(tick, gain_cell=sy.gain_true)
            sy.H.sub_dt = sy.dt_sub
            for _ in range(sy.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            if (tick + 1) not in (1, 2, 5, 10, 20, 40, 80, 120, 165, 200):
                continue
            x = sy.p.get("pos")
            F = sy.p.F.clone()
            Fk, _ = FD.particle_lsq_F(X, x, 4.0 * p_sp)
            Fg = FD.derive_F(cg, X, x)

            def st(A, Bm, m=None):
                if m is not None:
                    A, Bm = A[m], Bm[m]
                a = (A - I2).reshape(-1, 4)
                b = (Bm - I2).reshape(-1, 4)
                sc = float((b.reshape(-1) @ a.reshape(-1)) / (b.reshape(-1) @ b.reshape(-1)))
                am, bm = a.reshape(-1) - a.mean(), b.reshape(-1) - b.mean()
                co = float((am @ bm) / (am.norm() * bm.norm()))
                return float((A - Bm).norm() / Bm.norm()), sc, co

            r, sc, co = st(Fk, F)
            ri, sci, coi = st(Fk, F, interior)
            rg, _, _ = st(Fg, F)
            rgi, _, _ = st(Fg, F, interior)
            rT, _, _ = st(Fk.transpose(1, 2), F)
            rTT, _, _ = st(Fk, F.transpose(1, 2))
            log(f"{tick+1:>5d} {float((F-I2).abs().median()):>9.2e} "
                f"{float((Fk-I2).abs().median()):>11.2e} {r:>11.4f} {rT:>12.4f} {rTT:>12.4f} "
                f"{sc:>7.3f} {co:>7.3f} | {ri:>12.4f} {sci:>7.3f} {coi:>7.3f} "
                f"| {rg:>9.4f} {rgi:>8.4f}")
    log(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

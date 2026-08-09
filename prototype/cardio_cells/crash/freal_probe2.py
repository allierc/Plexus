"""freal_probe2.py -- why does injecting the DERIVED F blow the one-frame prediction up by 500x
even at theta_true, when the derived F agrees with the simulator's to 10% of |F - I| on the
interior?

Three suspects, and each has a control:
  A  the WALL BAND.  Stage d masked it; the injection does not.  Control: inject the derived F on
     the interior and the simulator's F on the band.
  B  the TAIL.  A handful of particles with an absurd derived F.  Control: winsorise |F - I|.
  C  genuine SENSITIVITY.  Injecting F every substep may simply be that fragile.  Control: the
     one-parameter family F_true + eps (F_der - F_true), eps in [0, 1] -- if the residual is
     already large at eps = 0.05 the measurement was never going to be good enough, and if it only
     blows up near eps = 1 then A or B is the cause.
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

import metrics as MET                                            # noqa: E402
from finject import lerp, y_of                                   # noqa: E402
from round5_fit import SNAP                                      # noqa: E402
import freal_derivedF as FD                                      # noqa: E402

PX = FD.PX


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    t0, hk = 165, 180
    args = SimpleNamespace(device=dev, cells=100, per_parent=100, n_grid=128, warmup=t0,
                           window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)

    def log(s):
        print(s, flush=True)

    ts = time.time()
    with torch.no_grad():
        sy, REF, B = FD.collect(args, t0, hk, log, keep_ticks=(t0 - 2,))
        n, th = sy.n_sub_per_frame, sy.theta_true.double()
        I2 = torch.eye(2, device=th.device, dtype=torch.float64)
        band = 0.06 / MET.SHEET_SPAN
        xW = B[t0]["x0"]
        interior = ~((xW[:, 0] < band) | (xW[:, 0] > 1 - band)
                     | (xW[:, 1] < band) | (xW[:, 1] > 1 - band))
        p_sp = 0.00996
        y_obs = (B[t0]["x_next"] - B[t0]["x0"]).reshape(-1)
        Ft0, Ft1 = B[t0]["F0"], B[t0]["F1"]
        log(f"[setup] |y_obs| {float(y_obs.norm()):.4e}  interior {int(interior.sum())}/{sy.Np} "
            f"[{time.time()-ts:.0f}s]")

        def hold_res(injF, theta=th, per_particle=False):
            for kk, vv in B[t0]["snap"].items():
                setattr(sy, kk, vv.clone())
            y = y_of(sy, theta, n, injF, None)
            if per_particle:
                return (y - y_obs).reshape(-1, 2)
            return float((y - y_obs).norm() / y_obs.norm())

        base = hold_res(lerp(Ft0, Ft1, n))
        log(f"[control] simulator F, theta_true: {base:.5f}   (finject's 0.0047-0.0067 regime)")

        for m, hp in ((2, 15.0), (2, 34.0), (165, 34.0)):
            r = t0 - m if t0 - m > 0 else 0
            cg = FD.ControlGrid(REF[r]["x"], hp * PX)
            F0 = FD.derive_F(cg, REF[r]["x"], B[t0]["x0"], F_ref=REF[r]["F"])
            F1 = FD.derive_F(cg, REF[r]["x"], B[t0]["x_next"], F_ref=REF[r]["F"])
            d = (F0 - Ft0).abs()
            log(f"\n=== m={m} h={hp:g}px  |Fd - Ft| med {float(d.median()):.2e} "
                f"p99 {float(torch.quantile(d.reshape(-1),0.99)):.2e} "
                f"max {float(d.max()):.2e} | interior med "
                f"{float(d[interior].median()):.2e} p99 "
                f"{float(torch.quantile(d[interior].reshape(-1),0.99)):.2e} max "
                f"{float(d[interior].max()):.2e} | band max {float(d[~interior].max()):.2e}")
            log(f"    det Fd: min {float(torch.linalg.det(F0).min()):.4f} "
                f"max {float(torch.linalg.det(F0).max()):.4f}; det Ft: min "
                f"{float(torch.linalg.det(Ft0).min()):.4f} max "
                f"{float(torch.linalg.det(Ft0).max()):.4f}")

            # ---- A: wall band --------------------------------------------------------------- #
            def blend(mask, A_, Bm):
                out = Bm.clone()
                out[mask] = A_[mask]
                return out

            rA = hold_res(lerp(blend(interior, F0, Ft0), blend(interior, F1, Ft1), n))
            rAb = hold_res(lerp(blend(~interior, F0, Ft0), blend(~interior, F1, Ft1), n))
            rfull = hold_res(lerp(F0, F1, n))
            log(f"    [A] derived everywhere {rfull:.4f} | derived on INTERIOR only {rA:.4f} | "
                f"derived on BAND only {rAb:.4f}")

            # ---- B: winsorise the tail ------------------------------------------------------ #
            for q in (0.999, 0.99):
                thr = float(torch.quantile((F0 - I2).abs().reshape(-1), q))
                W0 = I2 + (F0 - I2).clamp(-thr, thr)
                W1 = I2 + (F1 - I2).clamp(-thr, thr)
                log(f"    [B] winsorised at q={q} (|dF| <= {thr:.3e}): "
                    f"{hold_res(lerp(W0, W1, n)):.4f}")

            # ---- C: the sensitivity curve --------------------------------------------------- #
            row = []
            for eps in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
                row.append((eps, hold_res(lerp(Ft0 + eps * (F0 - Ft0),
                                               Ft1 + eps * (F1 - Ft1), n))))
            log("    [C] F_true + eps (F_der - F_true): " +
                "  ".join(f"eps {e:g} -> {v:.4f}" for e, v in row))

            # where does the residual live?
            pp = hold_res(lerp(F0, F1, n), per_particle=True).pow(2).sum(-1)
            tot = float(pp.sum())
            log(f"    [where] band {100*float(pp[~interior].sum())/tot:.1f}% of the squared "
                f"residual on {100*float((~interior).sum())/sy.Np:.1f}% of particles; "
                f"top 1% of particles carry "
                f"{100*float(pp.topk(sy.Np//100).values.sum())/tot:.1f}%")
    log(f"[{time.time()-ts:.0f}s]")


if __name__ == "__main__":
    main()

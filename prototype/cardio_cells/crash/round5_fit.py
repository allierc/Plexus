"""round5_fit.py -- ROUND 5, stage 1.  Assemble the T-frame stacked normal equations ONCE and
save them, so that every solver variant can be tried afterwards for free.

THE CONFIGURATION BEING CONSOLIDATED (the best found in rounds 1-4)
====================================================================================================
  * frame cadence (dt = 2e-3 = 10 substeps) -- the only cadence a recording offers;
  * the DISPLACEMENT read-out y(theta) = x_end - x_0 (round 1's diagnosis);
  * the MEASURED deformation gradient injected across the substeps of a frame by LINEAR
    INTERPOLATION between the two frame-boundary values (round 3 part 2: med|dE/E| 0.257 -> 0.008,
    equal to the substep oracle);
  * T consecutive frames STACKED (round 4 part 3);
  * the recording's own error bars: sigma_F = 3.9e-3 on the derivative channels, applied ONCE PER
    FRAME BOUNDARY and shared by the two frames that use it (coherent, not per substep -- round 3's
    diagnosis measured that the per-substep draw averages out and understates the damage), and
    sigma_x = 0.0409 px = 2.00e-5 world on the positions.

WHAT IS SAVED
----------------------------------------------------------------------------------------------------
For each frame k: G_k = A_k^T A_k, r_k = A_k^T b_k (column-scaled by recover.theta_scale), and the
K-sample Monte-Carlo re-noised averages Gm_k, rm_k from which Sigma = Gm - G is formed.  A stack of
T frames is the partial sum, so T in {1,2,4,8} all come out of one file.

usage:
  PYTHONPATH=/workspace/Plexus/src python round5_fit.py --device cuda:1 --seed 90210
  PYTHONPATH=/workspace/Plexus/src python round5_fit.py --device cuda:0 --sigma-F 0 --K 0 \
      --tag round5_norm_clean
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

from recover import theta_scale                                  # noqa: E402
import crash_test as CT                                          # noqa: E402
from finject import record_substeps, lerp, assemble_inj          # noqa: E402
from refute_round3 import advance                                # noqa: E402

SIGMA_X = 0.0409 * 4.88e-4      # real_F_check.json: temporal noise on the recorded displacement
SIGMA_F = 3.9e-3                # real_F_check.json: quiet-stretch second difference on dF (white)
SNAP = ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--sigma-x", type=float, default=SIGMA_X)
    ap.add_argument("--seed", type=int, default=90210)
    a = ap.parse_args()
    tag = a.tag or f"round5_norm_s{a.seed}_sF{a.sigma_F:g}"

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "sigma_F": a.sigma_F, "sigma_x": a.sigma_x,
         "K": a.K, "T": a.T, "seed": a.seed, "tag": tag}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        s = theta_scale(C, sy.device)

        frames = []
        cur = a.t0
        for k in range(a.T):
            if k > 0:
                sy.restore()
                advance(sy, cur, cur + 1)
                sy._snapshot(cur + 1)
                cur += 1
            Fs, Cs, Xs = record_substeps(sy, n)
            frames.append({"tick": cur, "x0": sy.x0.clone(), "F0": sy.F0.clone(),
                           "F1": Fs[-1].clone(), "x_next": Xs[-1].clone(),
                           "snap": {kk: getattr(sy, kk).clone() for kk in SNAP}})
        log(f"[frames] {a.T} consecutive frames from tick {a.t0} (ticks "
            f"{frames[0]['tick']}..{frames[-1]['tick']})")

        gm = torch.Generator(device=sy.device).manual_seed(a.seed)
        gk = torch.Generator(device=sy.device).manual_seed(31337 + a.seed)

        def dr(x, g):
            return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)

        # ONE measurement error per frame boundary, shared by the two frames that use it
        eb = [(a.sigma_F / 2.0) * dr(frames[0]["F0"], gm) for _ in range(a.T + 1)]
        xs = [f["x_next"] + a.sigma_x * dr(f["x_next"], gm) for f in frames]

        out = {}
        R["frames"] = []
        for k, f in enumerate(frames):
            for kk in SNAP:
                setattr(sy, kk, f["snap"][kk].clone())
            F0h, F1h = f["F0"] + eb[k], f["F1"] + eb[k + 1]
            A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
            Az = A * s[None, :]
            b = (xs[k] - f["x0"]).reshape(-1) - y0
            Gk, rk = Az.T @ Az, Az.T @ b
            del A, Az
            torch.cuda.empty_cache()
            Gs, rs = torch.zeros_like(Gk), torch.zeros_like(rk)
            for _ in range(a.K):
                e0 = (a.sigma_F / 2.0) * dr(f["F0"], gk)
                e1 = (a.sigma_F / 2.0) * dr(f["F0"], gk)
                Aj, y0j, _ = assemble_inj(sy, n, lerp(F0h + e0, F1h + e1, n), None)
                Azj = Aj * s[None, :]
                Gs += Azj.T @ Azj
                rs += Azj.T @ ((xs[k] - f["x0"]).reshape(-1) - y0j)
                del Aj, Azj
                torch.cuda.empty_cache()
            if a.K > 0:
                Gs, rs = Gs / a.K, rs / a.K
            out[f"G{k}"] = Gk.cpu().numpy()
            out[f"r{k}"] = rk.cpu().numpy()
            out[f"Gm{k}"] = Gs.cpu().numpy()
            out[f"rm{k}"] = rs.cpu().numpy()
            R["frames"].append({"tick": f["tick"],
                                "F_relchange_over_frame": float((f["F1"] - f["F0"]).norm()
                                                                / f["F0"].norm()),
                                "y_obs_norm": float((xs[k] - f["x0"]).norm())})
            log(f"    frame {k} (tick {f['tick']}) assembled + {a.K} re-noisings "
                f"[{time.time()-t_start:.0f}s]")

        out["theta_true"] = sy.theta_true.double().cpu().numpy()
        out["s"] = s.cpu().numpy()
        np.savez(os.path.join(HERE, f"{tag}.npz"), **out)

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {tag}.npz [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

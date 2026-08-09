"""round5_repro.py -- ROUND 5 control: what is the crash test's own reproducibility floor?

Round 5's `s90210/T8/naive` theta agrees with round 4's stored theta to 1.8e-15 RELATIVE, and yet
its raw margin-20 loopscore reads 0.7821 here against round 4's 0.7827.  Either the score is not a
function of theta, or something in the rollout is not deterministic (MLS-MPM scatters with atomic
adds, and the run already warns that `bincount_cuda` has no deterministic implementation).  Before
any difference of 1e-3 is interpreted, that floor has to be measured.

Same theta, REPEATED raw rollouts on one GPU, then the same on the other.  No gauge, no per-cell
field beyond what one rollout gives.  ~8 s per rollout.

usage: PYTHONPATH=/workspace/Plexus/src python round5_repro.py --device cuda:1 --reps 4
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

from assemble import SUBSTEP_TOKENS                              # noqa: E402
from recover import install_E                                    # noqa: E402
import metrics as MET                                            # noqa: E402
import crash_test as CT                                          # noqa: E402
from crash_round2 import percell_amplitude, r2_percell           # noqa: E402
from crash_round3 import t2_of                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="round5_repro")
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=165, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "reps": a.reps}
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, W, G, n = sy.C, 165, 150, sy.n_sub_per_frame
        th = sy.theta_true.double()
        dev, f64 = th.device, torch.float64
        x0, cid = sy.x0.clone(), sy.cid
        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE,)}
        band = 0.06 / MET.SHEET_SPAN
        anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                  (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
        interior = ~anchor
        ref_full = torch.zeros(G, sy.Np, 2, device=sy.device, dtype=sy.dtype)
        sy.restore()
        install_E(sy, sy.E_true)
        for k in range(G):
            sy._outer(W + k, gain_cell=sy.gain_true)
            sy.H.sub_dt = sy.dt_sub
            for _ in range(n):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            ref_full[k] = sy.p.get("pos")
        d_ref = ref_full - x0[None]
        dm = d_ref[:, interior].mean(0, keepdim=True)
        ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
        real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()
        a_ref, _ = percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)

        Z = np.load(os.path.join(HERE, "theta_round5.npz"))
        cands = {"theta_true": th,
                 "s90210/T8/naive": torch.as_tensor(
                     Z["round5_norm_s90210_sF0.0039|T8|naive"], device=dev, dtype=f64),
                 "s90210/T8/eiv_box": torch.as_tensor(
                     Z["round5_norm_s90210_sF0.0039|T8|eiv_box"], device=dev, dtype=f64),
                 "bank_blind_E130_g0.95": torch.cat([
                     torch.full((C,), 130.0, device=dev, dtype=f64),
                     torch.full((C,), 0.95, device=dev, dtype=f64)])}
        R["runs"] = {}
        log(f"\n[repro] {a.reps} identical raw rollouts per candidate on {a.device}")
        log(f"    {'candidate':<24s} {'loop mean':>10s} {'loop spread':>12s} {'R2 spread':>11s} "
            f"{'r2cell spread':>14s} {'rms spread':>11s}")
        for nm, t in cands.items():
            vals = []
            for _ in range(a.reps):
                tr, full, coarse = CT.rollout(sy, t, W, G, tracers, ref_full=ref_full,
                                              anchor=None, interior=interior, ss_tot=ss_tot,
                                              keep_full=True, band_mask=anchor)
                m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                ah, _ = percell_amplitude(full, x0, cid, C, interior)
                vals.append({"loop": m20["loopscore"], "t1": coarse["motion_energy_ratio_interior"],
                             "t2": t2_of(m20), "R2": coarse["R2_displacement_interior"],
                             "r2cell": r2_percell(ah, a_ref, keep)["r2"],
                             "rms": coarse["rms_pos_err_dx_mean"],
                             "orientation_error": m20["orientation_error"],
                             "coordination": m20["coordination"]})
                del full

            def sp(k):
                v = [x[k] for x in vals if isinstance(x[k], float)]
                return float(max(v) - min(v)) if v else None
            R["runs"][nm] = {"values": vals,
                             "spread": {k: sp(k) for k in
                                        ("loop", "R2", "r2cell", "rms", "t1", "t2",
                                         "orientation_error", "coordination")},
                             "mean_loop": float(np.mean([x["loop"] for x in vals]))}
            q = R["runs"][nm]
            log(f"    {nm:<24s} {q['mean_loop']:>10.5f} {q['spread']['loop']:>12.2e} "
                f"{q['spread']['R2']:>11.2e} "
                f"{(q['spread']['r2cell'] if q['spread']['r2cell'] is not None else float('nan')):>14.2e}"
                f" {q['spread']['rms']:>11.2e}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}_{a.device.replace(':','')}.json"), "w"),
              indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}_{a.device.replace(':','')}.log"),
         "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}_{a.device.replace(':','')}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""gauge_scan.py -- is the 2-D gauge's solution UNIQUE, and is the (t1, t2) map invertible?

Round 3's gauge drives (t1, t2) = (motion-energy ratio, path_length/peak_excursion ratio) to (1, 1)
by moving (k_E, k_g). It converged for 26 of 30 candidates but returned wildly different k_E for
zero-information vectors that differ only by noise (bank_prior_draw_101 -> 2.639, _404 -> 0.968),
and diverged outright on bank_blind_E320_g1. Either the map is not invertible where those live, or
the solver is at fault. This scans the map on a grid so the answer is measured, not assumed, and
also reports the loopscore over the same grid so the COST of landing on the wrong root is visible.

usage: PYTHONPATH=/workspace/Plexus/src python gauge_scan.py --device cuda:1
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

from assemble import SUBSTEP_TOKENS                             # noqa: E402
from recover import install_E                                   # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402
import crash_round3 as R3                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="gauge_scan")
    ap.add_argument("--cands", default="bank_prior_draw_101,theta_true")
    a = ap.parse_args()
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=165,
                           window=150, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t0 = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G = sy.C, args.warmup, args.window
        th = sy.theta_true.double()
        x0 = sy.x0.clone()
        tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                   for m in (MET.MARGIN_SAFE, MET.MARGIN_INHERITED)}
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
            for _ in range(sy.n_sub_per_frame):
                for tok in SUBSTEP_TOKENS:
                    sy._tok(tok)
            sy.H.sub_dt = None
            ref_full[k] = sy.p.get("pos")
        d_ref = ref_full - x0[None]
        dm = d_ref[:, interior].mean(0, keepdim=True)
        ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
        real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()

        z = np.load(os.path.join(HERE, "theta_round3.npz"))
        KE = [0.5, 0.8, 1.2, 2.0, 3.2]
        KG = [0.6, 0.85, 1.1, 1.4, 1.8]
        R["grid"] = {"k_E": KE, "k_g": KG}
        R["scan"] = {}
        for nm in a.cands.split(","):
            theta = th if nm == "theta_true" else torch.as_tensor(
                z[f"cand::{nm}"], device=th.device, dtype=torch.float64)
            tab = []
            log(f"\n[{nm}]  rows k_E, cols k_g;  cell = t1 / t2 / loopscore")
            log("        " + "".join(f"{g:>22.2f}" for g in KG))
            for ke in KE:
                row = []
                for kg in KG:
                    tr, _, coarse = CT.rollout(sy, R3.scale2(theta, ke, kg, C), W, G, tracers,
                                               ref_full=ref_full, anchor=None, interior=interior,
                                               ss_tot=ss_tot)
                    m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                    row.append({"k_E": ke, "k_g": kg,
                                "t1": coarse["motion_energy_ratio_interior"],
                                "t2": R3.t2_of(m20),
                                "loopscore": m20["loopscore"],
                                "R2": coarse["R2_displacement_interior"]})
                tab.append(row)
                log(f"  {ke:>5.2f} " + "".join(
                    f"{c['t1']:>7.3f}/{c['t2']:>6.3f}/{(c['loopscore'] if isinstance(c['loopscore'], float) else float('nan')):>6.3f}"
                    for c in row))
            R["scan"][nm] = tab
            # how many grid cells sit within 5% of BOTH targets, and how much loopscore they span
            near = [c for row in tab for c in row
                    if abs(c["t1"] - 1) < 0.15 and abs(c["t2"] - 1) < 0.05]
            R["scan"][nm + "::near_targets"] = near
            if near:
                ls = [c["loopscore"] for c in near if isinstance(c["loopscore"], float)]
                log(f"  grid cells with |t1-1|<0.15 and |t2-1|<0.05: {len(near)}; "
                    f"loopscore among them {min(ls):.4f} .. {max(ls):.4f}")

    R["wall_seconds"] = time.time() - t0
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

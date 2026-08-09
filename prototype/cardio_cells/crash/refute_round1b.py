"""refute_round1b.py -- ROUND 2 part 2. Is the crash-test score reading AMPLITUDE, not per-cell theta?

refute_round1.py found that per-cell-BLIND constants whose motion-energy ratio happens to be ~2
(blind_E40_g1: 1.95 -> loopscore -0.180) score exactly as badly as theta_hat_frame (2.03 -> -0.164),
and that an independent draw from the prior (E_ratio 0.78) scores +0.676. That is a correlation.
This script turns it into a CONTROL and a CEILING:

  1. AMPLITUDE-ONLY: theta = theta_true with the whole gain block multiplied by k. Per-cell structure
     is PERFECT; only the global amplitude is wrong. If loopscore collapses to the frame estimate's
     value at the k that reproduces its motion-energy ratio, amplitude explains the whole failure.
  2. THE BEST BLIND CONSTANT: sweep a per-cell-blind (E, g). Round 1 used E=130, g=1 (ratio 0.908,
     loopscore 0.737) and called it "the floor". A TUNED constant is the honest floor, and every
     per-cell estimate has to clear it.
  3. GAUGE-FIXED frame estimates: does the frame-cadence fit carry per-cell information that
     survives once its global scale is repaired?

usage: PYTHONPATH=/workspace/Plexus/src python refute_round1b.py --device cuda:0
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
sys.path.insert(0, "/workspace/Plexus/src")
sys.path.insert(0, ALG)
sys.path.insert(0, DISC)
sys.path.insert(0, HERE)

from assemble import SUBSTEP_TOKENS                            # noqa: E402
from recover import install_E, score                           # noqa: E402
import metrics as MET                                          # noqa: E402
import crash_test as CT                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="refute1b")
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
        ref_tr = {m: ref_full[:, t] for m, t in tracers.items()}

        z1 = np.load(os.path.join(HERE, "theta_round1.npz"))
        z2 = np.load(os.path.join(HERE, "theta_refute1.npz"))
        dev, dt64 = th.device, torch.float64
        t_disp = torch.as_tensor(z2["cand::frame_DISP"], device=dev, dtype=dt64)
        t_fd = torch.as_tensor(z1["cand::theta_hat_frame_ridge0"], device=dev, dtype=dt64)

        cand = {}
        # 1. amplitude-only: perfect per-cell structure, wrong global gain
        for k in (0.6, 0.8, 0.9, 1.1, 1.25, 1.5, 1.8):
            cand[f"true_gain_x{k:g}"] = torch.cat([th[:C], th[C:] * k])
        # 1b. amplitude-only through E instead
        for k in (0.5, 0.7, 1.5, 2.0):
            cand[f"true_E_x{k:g}"] = torch.cat([th[:C] * k, th[C:]])
        # 2. the best BLIND constant
        for E in (80.0, 100.0, 115.0, 130.0, 160.0, 200.0):
            cand[f"blind_E{E:g}_g1"] = torch.cat([torch.full((C,), E, device=dev, dtype=dt64),
                                                  torch.ones(C, device=dev, dtype=dt64)])
        for g in (0.85, 0.95, 1.05):
            cand[f"blind_E130_g{g:g}"] = torch.cat([torch.full((C,), 130.0, device=dev, dtype=dt64),
                                                    torch.full((C,), g, device=dev, dtype=dt64)])
        # 3. gauge-fixed frame estimates (oracle per-block scalar -- a diagnostic, not a method)
        for nm, t in (("frame_DISP", t_disp), ("frame_vel_fd", t_fd)):
            kE = float((t[:C] * th[:C]).sum() / (t[:C] ** 2).sum())
            kg = float((t[C:] * th[C:]).sum() / (t[C:] ** 2).sum())
            cand[f"{nm}_gaugefix"] = torch.cat([t[:C] * kE, t[C:] * kg])
            R.setdefault("gauge_k", {})[nm] = [kE, kg]
        # 4. the same per-cell error as frame_DISP but SHUFFLED across cells (destroys the
        #    per-cell assignment, keeps the error magnitude and the global scale)
        pm = torch.randperm(C, generator=torch.Generator().manual_seed(5), device="cpu").to(dev)
        e = t_disp - th
        cand["frame_DISP_error_shuffled"] = th + torch.cat([e[:C][pm], e[C:][pm]])

        R["rollouts"] = {}
        log(f"\n[amplitude control] {G}-frame FREE rollouts, margin-{MET.MARGIN_SAFE}")
        log(f"    {'candidate':<30s} {'medE':>7s} {'medg':>7s} | {'coord':>7s} {'orient':>7s} | "
            f"{'loopsc':>8s} | {'R2':>9s} {'Eratio':>7s} {'rms/dx':>7s}")
        for name, theta in cand.items():
            tr, _, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full, anchor=None,
                                       interior=interior, ss_tot=ss_tot, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(),
                                  ref_tr[MET.MARGIN_SAFE].cpu().numpy())
            sc = score(theta, th, C)
            R["rollouts"][name] = {"theta_error": sc, "margin20": m20, "coarse": coarse}
            log(f"    {name:<30s} {sc['med_E']:>7.4f} {sc['med_gain']:>7.4f} | "
                f"{CT.fmt(m20['coordination'],7)} {CT.fmt(m20['orientation_error'],7)} | "
                f"{CT.fmt(m20['loopscore'],8)} | {CT.fmt(coarse['R2_displacement_interior'],9)} "
                f"{CT.fmt(coarse['motion_energy_ratio_interior'],7)} "
                f"{CT.fmt(coarse['rms_pos_err_dx_mean'],7)}")

    R["wall_seconds"] = time.time() - t0
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f}s]")


if __name__ == "__main__":
    main()

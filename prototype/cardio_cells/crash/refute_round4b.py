"""refute_round4b.py -- does the box repair hold across the OTHER two stack seeds and at T1?

refute_round4.py measured, on seed 90210, that clipping each block of theta to [0.2, 5] x its own
MEDIAN (no knowledge of theta_true) takes T8/eiv_snr0 from raw loopscore 0.2375 to 0.9241 and
T8/naive from 0.7827 to 0.8793.  A repair measured on one draw is a coincidence.  Here every stack
theta on disk (3 seeds x T in {1,2,4,8} x {naive, eiv_snr0, eiv_snr0.3, eiv_snr1}) is scored RAW,
boxed and unboxed, plus a HELD-OUT one-frame prediction residual -- the only acceptance statistic
that is available on the real recording, since it needs b and not theta_true.

usage: PYTHONPATH=/workspace/Plexus/src python refute_round4b.py --device cuda:1
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
from finject import record_substeps, lerp, y_of                  # noqa: E402
from refute_round3 import advance                                # noqa: E402
from refute_round4 import param_stats                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="refute4b")
    ap.add_argument("--warmup", type=int, default=165)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--holdout-tick", type=int, default=180)
    a = ap.parse_args()

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.warmup, window=a.window, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G, n = sy.C, args.warmup, args.window, sy.n_sub_per_frame
        th = sy.theta_true.double()
        dev, f64 = th.device, torch.float64
        x0, cid = sy.x0.clone(), sy.cid

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
        snap0 = {k: getattr(sy, k).clone() for k in
                 ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")}

        # ---- the HELD-OUT frame: tick 180, eight frames past the last frame in the T=8 stack ----
        sy.restore()
        advance(sy, W, a.holdout_tick)
        sy._snapshot(a.holdout_tick)
        Fh, Ch, Xh = record_substeps(sy, n)
        hx0, hF0, hF1, hxn = sy.x0.clone(), sy.F0.clone(), Fh[-1].clone(), Xh[-1].clone()
        y_obs_h = (hxn - hx0).reshape(-1)
        injh = lerp(hF0, hF1, n)
        hsnap = {k: getattr(sy, k).clone() for k in
                 ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")}
        log(f"[holdout] tick {a.holdout_tick}: |y_obs| {float(y_obs_h.norm()):.4e}")

        def holdout_resid(theta):
            for k, v in hsnap.items():
                setattr(sy, k, v.clone())
            y = y_of(sy, theta, n, injh, None)
            return float((y - y_obs_h).norm() / y_obs_h.norm())

        def scored(theta):
            for k, v in snap0.items():
                setattr(sy, k, v.clone())
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full,
                                          anchor=None, interior=interior, ss_tot=ss_tot,
                                          keep_full=True, band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            ah, _ = percell_amplitude(full, x0, cid, C, interior)
            out = {"loop": m20["loopscore"], "t1": coarse["motion_energy_ratio_interior"],
                   "t2": t2_of(m20), "R2": coarse["R2_displacement_interior"],
                   "r2cell": r2_percell(ah, a_ref, keep)["r2"]}
            del full
            return out

        def box(t, lo_f=0.2, hi_f=5.0):
            o = t.clone()
            for sl in (slice(0, C), slice(C, 2 * C)):
                v = o[sl]
                m = v[v > 0].median() if int((v > 0).sum()) > 0 else v.abs().median()
                o[sl] = v.clamp(min=lo_f * m, max=hi_f * m)
            return o

        R["rows"] = {}
        log(f"\n[box across seeds] raw margin-20 loopscore, anchor=None; box = [0.2,5] x the "
            f"BLOCK'S OWN MEDIAN (no theta_true).  Null band top (round 3/4) = 0.6488")
        log(f"    {'theta':<34s} {'medE':>7s} {'relL2g':>7s} {'medE_re':>8s} {'neg':>4s} "
            f"{'hold1f':>7s} {'raw loop':>9s} {'R2':>8s} {'r2cell':>7s}")
        specs = []
        for tag, seedname in (("", "s90210"), ("_s555", "s555"), ("_s777", "s777")):
            for T in (1, 8):
                p = os.path.join(HERE, f"theta_round4_stack{tag}_T{T}.npz")
                if not os.path.exists(p):
                    continue
                z = np.load(p)
                for k in ("naive", "eiv_snr0"):
                    if k in z.files:
                        specs.append((f"{seedname}/T{T}/{k}",
                                      torch.as_tensor(z[k], device=dev, dtype=f64)))
        for name, t in specs:
            for variant, tt in (("raw", t), ("box", box(t))):
                ps = param_stats(tt.cpu().numpy(), th.cpu().numpy(), C)
                hr = holdout_resid(tt)
                sc = scored(tt)
                key = f"{name}|{variant}"
                R["rows"][key] = {"param": ps, "holdout_1frame_rel": hr, **sc}
                log(f"    {key:<34s} {ps['med_E']:>7.4f} {ps['rel_l2_gauge_opt']:>7.3f} "
                    f"{ps['med_E_after_rescale']:>8.4f} {ps['n_negE']:>4d} {hr:>7.4f} "
                    f"{CT.fmt(sc['loop'],9)} {CT.fmt(sc['R2'],8)} {sc['r2cell']:>7.4f}")

        # ---- the same held-out residual for the reference points, so it has a scale -------------
        R["holdout_reference"] = {}
        z4 = np.load(os.path.join(HERE, "theta_round4_eiv.npz"))
        refs = {"theta_true": th,
                "clean_F_lerp": torch.as_tensor(z4["cand::clean_F_lerp"], device=dev, dtype=f64)}
        gnull = torch.Generator().manual_seed(4242)
        idx = torch.randperm(C, generator=gnull)[:45].to(dev)
        Ed = (args.e_lo + (args.e_hi - args.e_lo) * torch.rand(C, generator=gnull,
                                                               dtype=torch.float64)).to(dev)
        gd = (args.g_lo + (args.g_hi - args.g_lo) * torch.rand(C, generator=gnull,
                                                               dtype=torch.float64)).to(dev)
        nm0 = th.clone()
        nm0[idx] = Ed[idx]
        nm0[C + idx] = gd[idx]
        refs["null_med0_rand45"] = nm0
        refs["bank_prior_draw_303"] = torch.cat([Ed, gd])
        refs["bank_blind_E130_g0.95"] = torch.cat([torch.full((C,), 130.0, device=dev, dtype=f64),
                                                   torch.full((C,), 0.95, device=dev, dtype=f64)])
        log(f"\n[holdout scale] the same one-frame held-out residual for the reference points")
        for nm, t in refs.items():
            ps = param_stats(t.cpu().numpy(), th.cpu().numpy(), C)
            hr = holdout_resid(t)
            R["holdout_reference"][nm] = {"holdout_1frame_rel": hr, "param": ps}
            log(f"    {nm:<26s} medE {ps['med_E']:>7.4f}  medE_re {ps['med_E_after_rescale']:>7.4f}"
                f"  hold1f {hr:>7.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

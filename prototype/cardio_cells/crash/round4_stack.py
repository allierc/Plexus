"""round4_stack.py -- does the de-biased estimator improve with MORE FRAMES, where the naive one
does not?

WHAT THIS SEPARATES
====================================================================================================
refute3_E2.json measured that stacking frames does NOT rescue the naive F-injected fit: white F
noise T=1 -> T=8 moves med|dE/E| only 0.539 -> 0.468, while position noise (which is variance)
halves properly over the same stack, 0.0308 -> 0.0163.  That is the signature of a BIAS.

Two different things could be true and they call for opposite conclusions:
  (i)  the bias is all that is wrong -- remove it and T frames then buy the usual 1/sqrt(T);
  (ii) the information is not there -- at sigma_F = 3.9e-3, round 4's SNR spectrum says 65 of the
       200 parameter directions have signal below noise at T = 1, and stacking raises the SNR of a
       direction roughly linearly in T, so the subspace should GROW.

Stacked normal equations, with the same Monte-Carlo correction and the same Sigma-metric
truncation as finject.solve_eiv, with ONE coherent measurement error per FRAME BOUNDARY shared by
the two frames that use it -- which is what a recording gives.

usage: PYTHONPATH=/workspace/Plexus/src python round4_stack.py --device cuda:0 --T 8 --K 6
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

from recover import score, theta_scale                          # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import record_substeps, lerp, assemble_inj         # noqa: E402
from refute_round3 import advance                               # noqa: E402

SIGMA_X = 0.0409 * 4.88e-4
SIGMA_F = 3.9e-3


def solve_stack(G0, r0, Gb, rb, s, taus):
    Sig, cv = Gb - G0, rb - r0
    Gc, rc = G0 - Sig, r0 - cv
    w, U = torch.linalg.eigh(Sig)
    sig2 = float(w.abs().max())
    wf = w.clamp(min=1e-2 * sig2)
    Lh = U @ torch.diag(wf.rsqrt()) @ U.T
    M = Lh @ Gc @ Lh
    lam, V = torch.linalg.eigh((M + M.T) / 2)
    Vw = Lh @ V
    out = {}
    try:
        out["naive"] = torch.linalg.solve(G0, r0) * s
    except Exception:
        out["naive"] = torch.linalg.lstsq(G0, r0.unsqueeze(1)).solution.squeeze(1) * s
    ranks = {}
    for tau in taus:
        keep = lam > tau
        if int(keep.sum()) == 0:
            out[f"eiv_snr{tau:g}"] = torch.zeros_like(r0)
        else:
            Vk = Vw[:, keep]
            out[f"eiv_snr{tau:g}"] = (Vk @ ((Vk.T @ rc) / lam[keep])) * s
        ranks[f"eiv_snr{tau:g}"] = int(keep.sum())
    return out, {"ranks": ranks, "n_snr_gt1": int((lam > 1).sum()),
                 "n_snr_gt3": int((lam > 3).sum()),
                 "snr_max": float(lam.max()), "sigma_spec": sig2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="round4_stack")
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--sigma-F", type=float, default=SIGMA_F)
    ap.add_argument("--seed", type=int, default=90210)
    a = ap.parse_args()
    taus = (0.0, 0.3, 1.0)

    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128,
                           warmup=a.t0, window=150, dtype="float64", mode="full",
                           e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "sigma_F": a.sigma_F, "K": a.K, "T": a.T}
    t_start = time.time()
    torch.manual_seed(0)

    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        s = theta_scale(C, sy.device)
        SNAP = ("state0", "F0", "C0", "Jp0", "v0", "x0", "act0", "pass0")

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
        log(f"[frames] {a.T} consecutive frames from tick {a.t0}; "
            f"|F1-F0| per frame {[f'{float((f['F1']-f['F0']).norm()/f['F0'].norm()):.2e}' for f in frames[:4]]}")

        gm = torch.Generator(device=sy.device).manual_seed(a.seed)
        gk = torch.Generator(device=sy.device).manual_seed(31337 + a.seed)

        def dr(x, g):
            return torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)

        # one MEASUREMENT error per frame boundary, shared by the two frames that use it
        eb = [(a.sigma_F / 2.0) * dr(frames[0]["F0"], gm) for _ in range(a.T + 1)]
        xs = [f["x_next"] + SIGMA_X * dr(f["x_next"], gm) for f in frames]

        Gk = [None] * a.T
        rk = [None] * a.T
        Gm = [None] * a.T
        rm = [None] * a.T
        for k, f in enumerate(frames):
            for kk in SNAP:
                setattr(sy, kk, f["snap"][kk].clone())
            F0h, F1h = f["F0"] + eb[k], f["F1"] + eb[k + 1]
            A, y0, _ = assemble_inj(sy, n, lerp(F0h, F1h, n), None)
            Az = A * s[None, :]
            b = (xs[k] - f["x0"]).reshape(-1) - y0
            Gk[k], rk[k] = Az.T @ Az, Az.T @ b
            del A, Az
            torch.cuda.empty_cache()
            Gs = torch.zeros_like(Gk[k])
            rs = torch.zeros_like(rk[k])
            for j in range(a.K):
                e0 = (a.sigma_F / 2.0) * dr(f["F0"], gk)
                e1 = (a.sigma_F / 2.0) * dr(f["F0"], gk)
                Aj, y0j, _ = assemble_inj(sy, n, lerp(F0h + e0, F1h + e1, n), None)
                Azj = Aj * s[None, :]
                Gs += Azj.T @ Azj
                rs += Azj.T @ ((xs[k] - f["x0"]).reshape(-1) - y0j)
                del Aj, Azj
                torch.cuda.empty_cache()
            Gm[k], rm[k] = Gs / a.K, rs / a.K
            log(f"    frame {k} (tick {f['tick']}) assembled + {a.K} re-noisings "
                f"[{time.time()-t_start:.0f}s]")

        R["runs"] = {}
        log(f"\n[stack] sigma_F {a.sigma_F:g} coherent per boundary, K={a.K}")
        log(f"    {'T':>3s} {'naive medE':>11s} {'naive mr':>9s} {'snr0 medE':>10s} "
            f"{'snr0 mr':>8s} {'snr.3 medE':>11s} {'snr.3 mr':>9s} {'snr1 medE':>10s} "
            f"{'snr1 mr':>8s} {'rank1':>6s} {'n>1':>5s}")
        for T in (1, 2, 4, 8):
            if T > a.T:
                continue
            G0 = sum(Gk[:T])
            r0 = sum(rk[:T])
            Gb = sum(Gm[:T])
            rb = sum(rm[:T])
            out, ex = solve_stack(G0, r0, Gb, rb, s, taus)
            row = {"extra": ex, "scores": {}}
            for nm, t in out.items():
                sc = score(t, th, C)
                sc["mean_ratio_E"] = float(t[:C].mean() / th[:C].mean())
                x = th[:C] - th[:C].mean()
                y = t[:C] - t[:C].mean()
                sc["slope_E"] = float((x * y).sum() / (x * x).sum())
                sc["n_negE"] = int((t[:C] < 0).sum())
                row["scores"][nm] = sc
            R["runs"][f"T{T}"] = row
            q = row["scores"]
            log(f"    {T:>3d} {q['naive']['med_E']:>11.4f} {q['naive']['mean_ratio_E']:>9.3f} "
                f"{q['eiv_snr0']['med_E']:>10.4f} {q['eiv_snr0']['mean_ratio_E']:>8.3f} "
                f"{q['eiv_snr0.3']['med_E']:>11.4f} {q['eiv_snr0.3']['mean_ratio_E']:>9.3f} "
                f"{q['eiv_snr1']['med_E']:>10.4f} {q['eiv_snr1']['mean_ratio_E']:>8.3f} "
                f"{ex['ranks']['eiv_snr1']:>6d} {ex['n_snr_gt1']:>5d}")
            np.savez(os.path.join(HERE, f"theta_{a.tag}_T{T}.npz"),
                     **{k: v.cpu().numpy() for k, v in out.items()})

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""refute3_simex.py -- is the damage done by a noisy measured F a VARIANCE or a BIAS?

Round 3 concluded the recording's F is unusable because at sigma_F = 0.0039 the recovery returns
to med|dE/E| ~ 0.26 (0.56 once the noise is made coherent within a frame, as a real per-frame
measurement is -- refute_round3.py phase D).  But WHERE the noise enters decides the remedy:

  * noise in b (the observed positions) is VARIANCE: more equations average it down as 1/sqrt(T);
  * noise in A (the injected F: every column of A is assembled WITH the noisy F) is
    ERRORS-IN-VARIABLES, whose leading effect is an ATTENUATION BIAS that does NOT average down.

This measures which one it is -- the regression slope of E_hat on E_true, and mean(E_hat)/mean(E) --
and then tries the textbook remedy, SIMEX: refit with extra known noise of variance lambda*sigma^2
for a ladder of lambda >= 0, fit theta_hat(lambda) per parameter, extrapolate to lambda = -1.
Only sigma_F is needed, and sigma_F is measurable on the recording itself (quiet-stretch second
difference; refute3_real.py verifies it is temporally white, lag-1 autocorrelation 0.0006).

usage: PYTHONPATH=/workspace/Plexus/src python refute3_simex.py --device cuda:0
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

from recover import score                                       # noqa: E402
import crash_test as CT                                         # noqa: E402
from finject import record_substeps, lerp                       # noqa: E402
from refute_round3 import advance, fit                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="refute3_simex")
    ap.add_argument("--sigma-F", type=float, default=3.9e-3)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--boot", type=int, default=3)
    a = ap.parse_args()
    START = 40
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=START,
                           window=START, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args)}
    t_start = time.time()
    torch.manual_seed(0)
    sx = 0.0409 * 4.88e-4
    with torch.no_grad():
        sy, _ = CT.plant_and_warm(args, log)
        C, n = sy.C, sy.n_sub_per_frame
        th = sy.theta_true.double()
        sy.restore()
        advance(sy, START, a.t0)
        sy._snapshot(a.t0)
        Fs, Cs, Xs = record_substeps(sy, n)
        x0, F0, F1, x_next = sy.x0.clone(), sy.F0.clone(), Fs[-1].clone(), Xs[-1].clone()
        gen = torch.Generator(device=sy.device).manual_seed(20260808)
        sF = a.sigma_F

        def draw(shape):
            return torch.randn(shape, generator=gen, device=sy.device, dtype=sy.dtype)

        # clean control
        s0, t0h = fit(sy, n, lerp(F0, F1, n), x_next, x0, th, C)
        log(f"[control] noise-free F_lerp at t0={a.t0}: med|dE/E| {s0['med_E']:.4f}")
        R["clean"] = s0

        # ONE measured realisation: F_hat = F + e, coherent within the frame
        e0, e1 = (sF / 2.0) * draw(F0.shape), (sF / 2.0) * draw(F0.shape)
        F0h, F1h = F0 + e0, F1 + e1
        xn = x_next + sx * draw(x_next.shape)

        def refit(lam, rep):
            """extra noise of variance lam*sigma^2 on top of the already-noisy measurement."""
            if lam <= 0:
                return fit(sy, n, lerp(F0h, F1h, n), xn, x0, th, C)
            k = (sF / 2.0) * float(np.sqrt(lam))
            return fit(sy, n, lerp(F0h + k * draw(F0.shape), F1h + k * draw(F0.shape), n),
                       xn, x0, th, C)

        LAM = (0.0, 0.5, 1.0, 1.5, 2.0)
        curve = {}
        log(f"\n[simex] sigma_F = {sF:g} coherent per frame, {a.boot} draws per lambda")
        log(f"    {'lambda':>7s} {'medE':>8s} {'mean(Ehat)/mean(E)':>20s} {'slope Ehat~E':>13s}")
        Etrue = th[:C].cpu().numpy()
        for lam in LAM:
            ths = []
            for r in range(a.boot if lam > 0 else 1):
                s, t_hat = refit(lam, r)
                ths.append(t_hat.cpu().numpy())
            tm = np.mean(ths, 0)
            Eh = tm[:C]
            sl = float(np.polyfit(Etrue, Eh, 1)[0])
            sc = score(torch.tensor(tm, device=sy.device, dtype=sy.dtype), th, C)
            curve[f"{lam:g}"] = {"theta": tm.tolist(), "med_E": sc["med_E"],
                                 "med_gain": sc["med_gain"], "rel_l2": sc["rel_l2"],
                                 "mean_ratio": float(Eh.mean() / Etrue.mean()), "slope": sl,
                                 "n_negE": int((Eh < 0).sum())}
            log(f"    {lam:>7.2f} {sc['med_E']:>8.4f} {curve[f'{lam:g}']['mean_ratio']:>20.4f} "
                f"{sl:>13.4f}")
        R["simex_curve"] = curve

        # quadratic extrapolation in lambda to lambda = -1, per parameter
        L = np.array(LAM)
        TH = np.stack([np.array(curve[f"{l:g}"]["theta"]) for l in LAM])          # [nlam, 2C]
        V = np.stack([np.ones_like(L), L, L ** 2], 1)
        coef, *_ = np.linalg.lstsq(V, TH, rcond=None)
        ext = coef[0] - coef[1] + coef[2]
        sc_ext = score(torch.tensor(ext, device=sy.device, dtype=sy.dtype), th, C)
        Vl = np.stack([np.ones_like(L), L], 1)
        coefl, *_ = np.linalg.lstsq(Vl, TH, rcond=None)
        extl = coefl[0] - coefl[1]
        sc_extl = score(torch.tensor(extl, device=sy.device, dtype=sy.dtype), th, C)
        R["simex_extrapolated"] = {"quadratic": sc_ext, "linear": sc_extl,
                                   "theta_quadratic": ext.tolist()}
        log(f"\n[simex] naive (lambda=0) med|dE/E| {curve['0']['med_E']:.4f}  ->  "
            f"linear extrapolation {sc_extl['med_E']:.4f}, quadratic {sc_ext['med_E']:.4f}")
        log(f"        mean(Ehat)/mean(E): naive {curve['0']['mean_ratio']:.4f} -> "
            f"quadratic {float(np.mean(ext[:C]) / Etrue.mean()):.4f}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

"""finject_noise.py -- ROUND 3, part 3. Put the RECORDING'S measured error bars into the synthetic
F-injection estimator, and see what is left of it.

finject.py: injecting a linearly interpolated F across the substeps of a frame takes the
frame-cadence recovery from med|dE/E| 0.2572 to 0.0078, and the resulting rollout scores loopscore
0.9997 with no gauge at all. That was with an EXACT F.

real_F_check.py, on the healthy specimen only:
  sigma_u  = 0.0409 px temporal noise on the PIV displacement  ( = 2.00e-5 world )
  sigma_F  = 0.0039 (Frobenius, per node) temporal noise on the derivative channels
  the recording's own derivative channels and a central difference of its own displacement field
  disagree by 0.0327 (median) -- 97% of |F - I| itself, and 8.5x more than the temporal noise
  explains, so 0.033 is the honest systematic uncertainty on F at a single node.

This sweeps both, together, on the synthetic system where the truth is known. Noise on F is applied
in two forms: INDEPENDENT per particle (averages out over the ~100 particles of a cell) and
CORRELATED within a cell (does not). The real error, being spatially structured, sits between them.

usage: PYTHONPATH=/workspace/Plexus/src python finject_noise.py --device cuda:0
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

from assemble import SUBSTEP_TOKENS, rel                        # noqa: E402
from recover import Solver, install_E, score                    # noqa: E402
import metrics as MET                                           # noqa: E402
import crash_test as CT                                         # noqa: E402
import crash_round3 as R3                                       # noqa: E402
from finject import assemble_inj, y_of, record_substeps, lerp   # noqa: E402

SIG_U_PX = 0.0409
PX_WORLD = 4.88e-4
SIG_X = SIG_U_PX * PX_WORLD                    # 2.00e-5 world
SIG_F_TEMPORAL = 0.0039
SIG_F_SYSTEMATIC = 0.0327


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="finject_noise")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    args = SimpleNamespace(device=a.device, cells=100, per_parent=100, n_grid=128, warmup=165,
                           window=150, dtype="float64", mode="full", e_lo=40.0, e_hi=220.0,
                           g_lo=0.5, g_hi=1.5)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"config": vars(args), "real_error_bars": {"sigma_u_px": SIG_U_PX, "sigma_x_world": SIG_X,
                                                   "sigma_F_temporal": SIG_F_TEMPORAL,
                                                   "sigma_F_systematic": SIG_F_SYSTEMATIC}}
    t_start = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        sy, recA = CT.plant_and_warm(args, log)
        C, W, G, n = sy.C, args.warmup, args.window, sy.n_sub_per_frame
        th = sy.theta_true.double()
        x0, dx, cid = sy.x0.clone(), sy.g.dx, sy.cid
        Fs, Cs, Xs = record_substeps(sy, n)
        x_next = Xs[-1].clone()
        F0, F1 = sy.F0.clone(), Fs[-1].clone()
        Fl = lerp(F0, F1, n)
        R["reference"] = {"F_lerp_error_vs_true": float((Fs - Fl).norm() / Fs.norm()),
                          "F_norm": float(Fs.norm() / np.sqrt(Fs.numel())),
                          "sigma_F_over_typical_F_minus_I":
                          SIG_F_TEMPORAL / float((Fs - torch.eye(2, device=Fs.device,
                                                                 dtype=Fs.dtype)).norm(dim=(-2, -1)).median())}
        log(f"[ref] |F_lerp - F_true|/|F| = {R['reference']['F_lerp_error_vs_true']:.3e}; "
            f"median |F - I| = "
            f"{float((Fs - torch.eye(2, device=Fs.device, dtype=Fs.dtype)).norm(dim=(-2,-1)).median()):.4e}")

        gen = torch.Generator(device=sy.device).manual_seed(4242)

        def noisy_F(sig, mode):
            if sig <= 0:
                return Fl
            if mode == "indep":
                e = torch.randn(Fl.shape, generator=gen, device=Fl.device, dtype=Fl.dtype)
            else:                                   # one draw per cell per substep, shared
                ec = torch.randn((Fl.shape[0], C + 1, 2, 2), generator=gen, device=Fl.device,
                                 dtype=Fl.dtype)
                e = ec[:, cid]
            return Fl + (sig / 2.0) * e             # /2: sig is the Frobenius norm over 4 entries

        cases = []
        for sF, mode in ((0.0, "indep"), (SIG_F_TEMPORAL, "indep"), (SIG_F_TEMPORAL, "cell"),
                         (SIG_F_SYSTEMATIC, "indep"), (SIG_F_SYSTEMATIC, "cell")):
            for sX in (0.0, SIG_X):
                cases.append((sF, mode, sX))

        R["cases"] = {}
        best = None
        log(f"\n[sweep] F_lerp injection, frame cadence, displacement read-out; "
            f"{a.reps} repeats per case")
        log(f"    {'sigma_F':>9s} {'mode':>6s} {'sigma_x':>9s} {'medE':>8s} {'medg':>8s} "
            f"{'p90E':>8s} {'l2':>9s} {'negE':>5s}")
        for sF, mode, sX in cases:
            reps = []
            for r in range(a.reps if (sF > 0 or sX > 0) else 1):
                iF = noisy_F(sF, mode)
                A, y0, _ = assemble_inj(sy, n, iF, None)
                xn = x_next if sX <= 0 else x_next + sX * torch.randn(
                    x_next.shape, generator=gen, device=sy.device, dtype=sy.dtype)
                b = (xn - x0).reshape(-1) - y0
                S = Solver(A, C)
                t_hat = S(b)["ridge0"]
                reps.append((score(t_hat, th, C), t_hat.clone(), int((t_hat[:C] < 0).sum())))
                S.free(); del A, S
                torch.cuda.empty_cache()
            key = f"sF{sF:g}_{mode}_sx{sX:g}"
            R["cases"][key] = {
                "sigma_F": sF, "mode": mode, "sigma_x": sX,
                "med_E": float(np.mean([r[0]["med_E"] for r in reps])),
                "med_gain": float(np.mean([r[0]["med_gain"] for r in reps])),
                "p90_E": float(np.mean([r[0]["p90_E"] for r in reps])),
                "rel_l2": float(np.mean([r[0]["rel_l2"] for r in reps])),
                "n_negative_E": float(np.mean([r[2] for r in reps])),
                "per_rep_med_E": [r[0]["med_E"] for r in reps]}
            v = R["cases"][key]
            log(f"    {sF:>9.4f} {mode:>6s} {sX:>9.2e} {v['med_E']:>8.4f} {v['med_gain']:>8.4f} "
                f"{v['p90_E']:>8.4f} {v['rel_l2']:>9.3f} {v['n_negative_E']:>5.1f}")
            R["cases"][key]["_theta"] = reps[0][1]

        # ---- crash-test the realistic cases -------------------------------------------------------
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
        a_ref, _ = R3.percell_amplitude(ref_full, x0, cid, C, interior)
        keep = np.isfinite(a_ref) & (a_ref > 0)
        R["a_ref_percell"], R["keep_percell"] = a_ref.tolist(), keep.tolist()

        def scored(theta, full_out=True):
            tr, full, coarse = CT.rollout(sy, theta, W, G, tracers, ref_full=ref_full, anchor=None,
                                          interior=interior, ss_tot=ss_tot, keep_full=full_out,
                                          band_mask=anchor)
            m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
            out = {"margin20": m20, "coarse": coarse,
                   "t1": coarse["motion_energy_ratio_interior"], "t2": R3.t2_of(m20)}
            if full_out:
                ah, _ = R3.percell_amplitude(full, x0, cid, C, interior)
                out["percell"] = R3.r2_percell(ah, a_ref, keep)
                out["a_percell"] = ah.tolist()
                del full
            return out

        R["rollouts"] = {}
        log("\n[crash test] the noisy recoveries, 150-frame free rollout, margin-20")
        for key in [f"sF0_indep_sx{SIG_X:g}", f"sF{SIG_F_TEMPORAL:g}_indep_sx{SIG_X:g}",
                    f"sF{SIG_F_TEMPORAL:g}_cell_sx{SIG_X:g}",
                    f"sF{SIG_F_SYSTEMATIC:g}_indep_sx{SIG_X:g}",
                    f"sF{SIG_F_SYSTEMATIC:g}_cell_sx{SIG_X:g}"]:
            theta = R["cases"][key].pop("_theta")
            raw = scored(theta)

            def probe(lE, lg, theta=theta):
                d = scored(R3.scale2(theta, float(np.exp(lE)), float(np.exp(lg)), C),
                           full_out=False)
                return (d["t1"], d["t2"])
            gf = R3.gauge_fix2(probe, (raw["t1"], raw["t2"]))
            kE, kg = gf["k_E"], gf["k_g"]
            gau = raw if (kE == 1.0 and kg == 1.0) else scored(R3.scale2(theta, kE, kg, C))
            R["rollouts"][key] = {"theta_error": score(theta, th, C), "raw": raw, "gauged": gau,
                                  "gauge": gf}
            log(f"    {key:<26s} medE {score(theta, th, C)['med_E']:>7.4f} | raw loop "
                f"{CT.fmt(raw['margin20']['loopscore'],8)} t1 {raw['t1']:>6.3f} R2 "
                f"{CT.fmt(raw['coarse']['R2_displacement_interior'],8)} | kE {kE:>6.3f} kg "
                f"{kg:>6.3f} gauged loop {CT.fmt(gau['margin20']['loopscore'],8)} r2cell "
                f"{gau['percell']['r2']:>7.4f}")
        for k in list(R["cases"]):
            R["cases"][k].pop("_theta", None)

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

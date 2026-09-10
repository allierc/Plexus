"""s4_score -- score a fitted parameter set on beats the fit never saw, next to the baselines.

For each held-out window (beats 0, 1, 2; the fit beat is 3) the fitted (g, phi, log E, clock) are
rolled out from that window's own rest configuration with that window's own edge band prescribed
from the recording, and compared with the recording's per-cell affine maps on the SAME observable
the fit used. Only the clock's onset t0 may be re-aligned per held-out beat (a 1-D search over
+-3 frames), because the split's onsets are speed peaks and windows open a fixed 8 frames before
them; nothing else is touched.

Baselines on the same observable:
    replay      the FIT beat's recorded A_j(t), u_j(t), placed on the held-out window (the trivial
                model the old campaign never beat: +0.62 on its loop score)
    do nothing  A = I, u = 0 throughout

Scores per window: R^2 of A - I over (frames, cells, 4 components) and of u over (frames, cells,
2); per-cell shortening correlation and axis agreement at the peak; the fit's own loss.

    python s4_score.py --params out/fits/<tag>/params.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402


def r2(pred, ref):
    return float(1.0 - ((pred - ref) ** 2).sum() / ((ref - ref.mean()) ** 2).sum())


def peak_metrics(A, A_ref):
    eye = torch.eye(2, device=A.device)
    sm, sr = R.shortening(A), R.shortening(A_ref)
    pm, pr = int(sm.mean(1).argmax()), int(sr.mean(1).argmax())
    Em = 0.5 * ((A[pm] - eye) + (A[pm] - eye).transpose(-1, -2))
    Er = 0.5 * ((A_ref[pr] - eye) + (A_ref[pr] - eye).transpose(-1, -2))
    wm, vm = torch.linalg.eigh(Em); wr, vr = torch.linalg.eigh(Er)
    am = torch.atan2(vm[:, 1, 0], vm[:, 0, 0]); ar = torch.atan2(vr[:, 1, 0], vr[:, 0, 0])
    w = (-wr[:, 0]).clamp(min=0)
    return dict(shortening_corr=float(torch.corrcoef(torch.stack([sm[pm], sr[pr]]))[0, 1]),
                axis_agreement=float((w * torch.cos(2 * (am - ar))).sum() / w.sum()),
                peak_frame=pm, peak_frame_ref=pr,
                median_shortening_ratio=float(sm[pm].median() / sr[pr].median()))


def score(A, u, A_ref, u_ref, cells=None):
    """All metrics on `cells` (the interior when a band mask was used), else on every cell."""
    if cells is not None:
        A, u, A_ref, u_ref = A[:, cells], u[:, cells], A_ref[:, cells], u_ref[:, cells]
    eye = torch.eye(2, device=A.device)
    out = dict(r2_A=r2(A - eye, A_ref - eye), r2_u=r2(u, u_ref),
               loss=float(M.affine_loss(A, u, A_ref, u_ref)), n_cells=int(A.shape[1]))
    out.update(peak_metrics(A, A_ref))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-parent", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--anchor", type=float, default=1e4)
    ap.add_argument("--drag", type=float, default=30.0, help="Stokes drag k; 150 critically damps the substrate mode (step_test.py)")
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--fit-beat", type=int, default=3)
    ap.add_argument("--beats", default="0,1,2,3")
    ap.add_argument("--no-realign", action="store_true")
    ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True

    rec = R.load(device=dev, specimen=args.specimen)
    C = rec["n_cells"]
    LTIF = os.path.join(HERE, "data_hcm" if rec.get("specimen") == "hcm" else "data", "cells_2560.tif")
    z = np.load(args.params)
    interior = torch.as_tensor(z["interior"], device=dev) if "interior" in z.files else torch.ones(C, dtype=torch.bool, device=dev)
    mode = str(z["clock_mode"]) if "clock_mode" in z.files else "sigmoid"
    P = M.Params(C, dev, nu=args.nu, clock_mode=mode, n_frames=int(z["clock"].shape[0]) if mode == "free" else 0)
    P.load({k: z[k] for k in ("g", "phi", "logE", "clock", "delay", "g2") if k in z.files})
    fit_win = R.beat_window(rec, args.fit_beat)
    A_fit, u_fit = R.window_affine(rec, fit_win)

    results = {}
    for k in [int(b) for b in args.beats.split(",")]:
        win = R.beat_window(rec, k)
        T = len(win["frames"])
        A_ref, u_ref = R.window_affine(rec, win)
        kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, n_cells=C,
                  anchor_k=args.anchor, drag_k=args.drag)
        with torch.no_grad():
            r0 = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="s4_rest",
                                                   **dict(kw, n_frames=0))), P, dev, C, grad=False)
        X0, cid = r0["X0"], r0["cid"]
        band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
                | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
        prescribe = (band, M.band_prescription(A_ref, u_ref, X0, cid, band))
        sim = M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="s4", **kw))
        t0_fit = 0.0
        # a window opens PRE frames before its onset unless the recording starts later (beat 0 opens
        # at frame 0, 2 frames before its onset): that known truncation shifts the clock and the
        # replay alike; the +-3 search around it is the only per-beat freedom
        trunc = (win["onset"] - win["span"][0]) - R.PRE                 # <= 0
        shifts = [float(trunc)] if (args.no_realign or k == args.fit_beat) else \
            [trunc + d for d in (-3, -2, -1, 0, 1, 2, 3)]
        best = None
        for sh in shifts:
            with torch.no_grad():
                P.shift = -sh                      # gamma(t) evaluated at t + sh
                out = M.rollout(sim, P, dev, C, grad=False, prescribe=prescribe)
            sc = score(out["A"], out["u"], A_ref, u_ref, interior)
            if best is None or sc["loss"] < best[1]["loss"]:
                best = (sh, sc)
        P.shift = 0.0
        # baselines on the same window
        off = -trunc                                                   # replay aligned on the onset
        idx = torch.clamp(torch.arange(T, device=dev) + off, max=A_fit.shape[0] - 1)
        A_rep, u_rep = A_fit[idx], u_fit[idx]
        eye = torch.eye(2, device=dev)
        results[k] = dict(window=win["span"], held_out=(k != args.fit_beat), t0_shift=best[0],
                          model=best[1], replay=score(A_rep, u_rep, A_ref, u_ref, interior),
                          nothing=score(eye.expand_as(A_ref).clone(), torch.zeros_like(u_ref), A_ref, u_ref, interior))
        m, rp = results[k]["model"], results[k]["replay"]
        print(f"  beat {k} {'held-out' if k != args.fit_beat else 'FIT     '} window {win['span']}  "
              f"t0 shift {best[0]:+.0f}: model R2(A) {m['r2_A']:.3f} R2(u) {m['r2_u']:.3f} "
              f"short-corr {m['shortening_corr']:.3f} axis {m['axis_agreement']:.3f} | replay R2(A) "
              f"{rp['r2_A']:.3f} R2(u) {rp['r2_u']:.3f} short-corr {rp['shortening_corr']:.3f} axis "
              f"{rp['axis_agreement']:.3f}", flush=True)
    out_path = args.out or os.path.join(os.path.dirname(args.params), "s4_score.json")
    json.dump(dict(params=args.params, config=vars(args), results=results), open(out_path, "w"), indent=1)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

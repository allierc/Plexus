"""s1_forward -- S1: does the active-strain sheet rest -> peak -> rest within one beat, and what
does a fit-sized window cost?

Full size (472 cells x `--per-parent` points, grid `--n-grid`), the fit beat's window as the
physics defines it (recording.beat_window: rest -> contraction -> rest, 57 frames), one clock
fitted to the recording's mean shortening curve, g_j and phi_j read off each cell's peak.

  1. KINEMATICS. Mean shortening per frame, model against recording, on the same axis; the
     efficiency at peak (realised / commanded shortening); the last-frame strain and displacement
     as a fraction of their peak (rest recovery); J range.
  2. A FIRST LOOK AT THE RECORDING. Per-cell strain tensor at the two peaks: correlation per
     component, shortening correlation, axis agreement, centroid-displacement correlation, and
     the median model/recording shortening ratio. Not a fit: says whether "measured amplitude +
     measured axis + one clock" lands in the right place, and what a fit has to add.
  3. COST (unless --no-grad-cost): s per forward without a tape, and s + GB for forward+backward.

    python s1_forward.py --device cuda:0 --anchor 3e4 --nu 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402


def compare(A, u, A_rec, u_rec):
    """Model vs recording at each one's shortening peak. Returns a dict of scalars + frames."""
    eye = torch.eye(2, device=A.device)
    sm, sr = R.shortening(A), R.shortening(A_rec)                    # [T,C]
    pm, pr = int(sm.mean(1).argmax()), int(sr.mean(1).argmax())
    Em = 0.5 * ((A[pm] - eye) + (A[pm] - eye).transpose(-1, -2))
    Er = 0.5 * ((A_rec[pr] - eye) + (A_rec[pr] - eye).transpose(-1, -2))
    comp = {}
    for nm, (i, j) in dict(xx=(0, 0), xy=(0, 1), yy=(1, 1)).items():
        comp[nm] = float(torch.corrcoef(torch.stack([Em[:, i, j], Er[:, i, j]]))[0, 1])
    wm, vm = torch.linalg.eigh(Em); wr, vr = torch.linalg.eigh(Er)
    am = torch.atan2(vm[:, 1, 0], vm[:, 0, 0]); ar = torch.atan2(vr[:, 1, 0], vr[:, 0, 0])
    w = (-wr[:, 0]).clamp(min=0)
    return dict(peak_frame_model=pm, peak_frame_recording=pr, strain_corr=comp,
                shortening_corr=float(torch.corrcoef(torch.stack([sm[pm], sr[pr]]))[0, 1]),
                axis_agreement=float((w * torch.cos(2 * (am - ar))).sum() / w.sum().clamp(min=1e-12)),
                displacement_corr=float(torch.corrcoef(
                    torch.stack([u[pm].reshape(-1), u_rec[pr].reshape(-1)]))[0, 1]),
                median_shortening_ratio=float(sm[pm].median() / sr[pr].median().clamp(min=1e-9)),
                expansion_over_shortening=dict(
                    model=float(wm[:, 1].mean() / (-wm[:, 0]).mean().clamp(min=1e-9)),
                    recording=float(wr[:, 1].mean() / (-wr[:, 0]).mean().clamp(min=1e-9))),
                rotation_rms=dict(
                    model=float((0.5 * (A[pm] - A[pm].transpose(-1, -2)))[:, 0, 1].pow(2).mean().sqrt()),
                    recording=float((0.5 * (A_rec[pr] - A_rec[pr].transpose(-1, -2)))[:, 0, 1].pow(2).mean().sqrt())),
                curve_model=sm.mean(1).tolist(), curve_recording=sr.mean(1).tolist())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--sub", type=float, default=2e-4)
    ap.add_argument("--drag", type=float, default=30.0)
    ap.add_argument("--anchor", type=float, default=0.0, help="substrate spring kappa (1/time^2)")
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--youngs", type=float, default=80.0)
    ap.add_argument("--beat", type=int, default=3, help="which beat window (3 = the fit beat)")
    ap.add_argument("--clock", default=None, help="t0,tau_r,dur,tau_d in frames; default: fitted")
    ap.add_argument("--gain", type=float, default=1.0, help="multiply the measured g by this")
    ap.add_argument("--band", type=float, default=0.0,
                    help="prescribe the recording's motion within this world distance of the edge")
    ap.add_argument("--fast", action="store_true", help="engine-chosen (warp) bodies for the "
                    "kinematics rollout; no tape either way")
    ap.add_argument("--no-grad-cost", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rec = R.load(device=dev)
    C = rec["n_cells"]
    win = R.beat_window(rec, args.beat)
    T = len(win["frames"])
    A_rec, u_rec = R.window_affine(rec, win)
    phi0, amp0, tpk = R.fibre_init(A_rec)
    if args.clock:
        t0, tr, du, td = [float(v) for v in args.clock.split(",")]
        ck = dict(t0=t0, tau_r=tr, dur=du, tau_d=td)
    else:
        ck = R.clock_init(A_rec)
    P = M.Params(C, dev, g0=0.0, phi0=phi0, E0=args.youngs, t0=ck["t0"], tau_r=ck["tau_r"],
                 dur=ck["dur"], tau_d=ck["tau_d"], nu=args.nu)
    with torch.no_grad():
        P.g.copy_(amp0 * args.gain)
    gam = torch.stack([P.gamma(t) for t in range(T)]).detach()
    print(f"  {C} cells, beat {args.beat} window {win['span']} = {T} frames; g median "
          f"{P.g.median():.4f} max {P.g.max():.4f}; clock t0 {ck['t0']:.2f} tau_r {ck['tau_r']:.2f} "
          f"dur {ck['dur']:.2f} tau_d {ck['tau_d']:.2f}; gamma peak {gam.max():.3f} at frame "
          f"{int(gam.argmax())}, gamma(T-1) {gam[-1]:.4f}", flush=True)

    kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, dt=args.dt,
              sub=args.sub, n_cells=C, youngs=args.youngs, drag_k=args.drag, anchor_k=args.anchor)
    sim_fast = M.load_sim(M.build_spec(differentiable=not args.fast, name="s1_forward", **kw))
    eye = torch.eye(2, device=dev)

    prescribe = None
    if args.band > 0:
        raw0 = M.build_spec(differentiable=False, name="s1_rest", **dict(kw, n_frames=0))
        with torch.no_grad():
            r0 = M.rollout(M.load_sim(raw0), P, dev, C, grad=False)
        X0 = r0["X0"]
        band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
                | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
        prescribe = (band, M.band_prescription(A_rec, u_rec, X0, r0["cid"], band))
        print(f"  band {args.band}: {int(band.sum())} of {band.numel()} particles prescribed")

    # ---- 1. kinematics ------------------------------------------------------------------
    t_start = time.time()
    with torch.no_grad():
        out = M.rollout(sim_fast, P, dev, C, grad=False, prescribe=prescribe, keep_pos=True)
    s_fwd = time.time() - t_start
    A, u = out["A"], out["u"]
    sm = R.shortening(A)                                              # [T,C]
    cmd = gam[:, None] * P.g.detach()[None, :]
    pk = int(sm.mean(1).argmax())
    eff = float(sm.mean(1)[pk] / cmd.mean(1)[int(gam.argmax())].clamp(min=1e-9))
    fro = torch.linalg.norm((A - eye).reshape(T, C, 4), dim=-1).mean(1)
    un = u.norm(dim=-1).mean(1)
    rest_strain, rest_u = float(fro[-1] / fro.max()), float(un[-1] / un.max())
    J = out["J"]
    print(f"  forward ({'warp' if args.fast else 'differentiable'} bodies, no tape): {s_fwd:.1f} s "
          f"for {T} frames at {out['X0'].shape[0]:,} particles", flush=True)
    print(f"  model shortening peak at frame {pk} (clock peak {int(gam.argmax())}); efficiency "
          f"{eff:.3f}; last frame strain {rest_strain:.3f} / displacement {rest_u:.3f} of peak; "
          f"J {J.min():.4f}-{J.max():.4f}", flush=True)

    # ---- 2. against the recording -------------------------------------------------------
    cmpd = compare(A, u, A_rec, u_rec)
    print(f"  vs recording (peaks: model {cmpd['peak_frame_model']}, recording "
          f"{cmpd['peak_frame_recording']}): strain corr xx {cmpd['strain_corr']['xx']:+.3f} "
          f"xy {cmpd['strain_corr']['xy']:+.3f} yy {cmpd['strain_corr']['yy']:+.3f}; shortening corr "
          f"{cmpd['shortening_corr']:+.3f}; axis agreement {cmpd['axis_agreement']:+.3f}; "
          f"displacement corr {cmpd['displacement_corr']:+.3f}; median shortening ratio "
          f"{cmpd['median_shortening_ratio']:.3f}; expansion/shortening model "
          f"{cmpd['expansion_over_shortening']['model']:.2f} vs rec "
          f"{cmpd['expansion_over_shortening']['recording']:.2f}; rotation rms model "
          f"{cmpd['rotation_rms']['model']:.4f} vs rec {cmpd['rotation_rms']['recording']:.4f}",
          flush=True)
    print("  mean shortening  model: " + " ".join(f"{v:.4f}" for v in cmpd["curve_model"][:32]))
    print("                 record: " + " ".join(f"{v:.4f}" for v in cmpd["curve_recording"][:32]))

    # ---- 3. cost --------------------------------------------------------------------------
    cost = dict(seconds_forward=s_fwd, fast=args.fast, particles=int(out["X0"].shape[0]), frames=T)
    if not args.no_grad_cost:
        sim = M.load_sim(M.build_spec(differentiable=True, name="s1_grad", **kw))
        torch.cuda.reset_peak_memory_stats(dev)
        t_start = time.time()
        out_g = M.rollout(sim, P, dev, C, grad=True, prescribe=prescribe)
        L = M.affine_loss(out_g["A"], out_g["u"], A_rec, u_rec)
        L.backward()
        cost["seconds_forward_backward"] = time.time() - t_start
        cost["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        cost["loss_at_init"] = float(L)
        gn = {k: float(v.grad.norm()) for k, v in P.leaves().items()}
        print(f"  forward+backward: {cost['seconds_forward_backward']:.1f} s, "
              f"{cost['peak_mem_gb']:.1f} GB; loss at init {float(L):.4f}; grad norms "
              + " ".join(f"{k} {v:.2e}" for k, v in gn.items()), flush=True)

    tag = args.tag or f"p{args.per_parent}_g{args.n_grid}_k{args.anchor:g}_nu{args.nu:g}_b{args.band:g}"
    out_path = os.path.join(HERE, "out", f"s1_{tag}.json")
    json.dump(dict(config=vars(args), n_cells=C, window=win["span"], clock=ck,
                   kinematics=dict(peak_frame=pk, gamma_peak_frame=int(gam.argmax()),
                                   efficiency=eff, last_over_peak_strain=rest_strain,
                                   last_over_peak_displacement=rest_u,
                                   J_min=float(J.min()), J_max=float(J.max()),
                                   gamma=gam.tolist()),
                   recording=cmpd, cost=cost), open(out_path, "w"), indent=1)
    np.savez_compressed(out_path.replace(".json", ".npz"), pos=out["pos"],
                        X0=out["X0"].cpu().numpy(), cid=out["cid"].cpu().numpy(),
                        A=A.cpu().numpy(), u=u.cpu().numpy(), A_rec=A_rec.cpu().numpy(),
                        u_rec=u_rec.cpu().numpy(), gamma=gam.cpu().numpy())
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()

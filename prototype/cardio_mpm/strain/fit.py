"""fit -- gradient descent on the per-cell active-strain sheet, against a planted truth (S2/S3)
or the recording (S4, only after the gate).

    python fit.py --target planted --iters 150 --free g,phi,logE,clock --seed 0
    python fit.py --target recording --iters 150 ...            # S4 only

The loss is the per-cell affine mismatch over the beat window (recording.cell_affine on both
subjects; model.affine_loss), the edge band is prescribed from the recording in both cases (the
band is data either way), the substrate anchor is fixed at `--anchor`.

Planted truth: g and phi as measured on the target beat (g scaled by --plant-gain so the truth is
not the init), log E ~ N(log 80, --plant-sigma-logE) per cell, the fitted clock. The fit starts
FLAT: g = the median, E uniform, phi = the axis read off the TARGET's own peak (exactly what the
recording offers), clock from the target's mean curve. `--noise tracker` adds the measured
tracker-vs-tracker per-cell disagreement (S3); `--noise none` is S2.

Recovery is reported per family as  median |estimate - truth| / spread(truth)  over cells (the
S3 rule: a family ships only if this is < 0.5 over 3 seeds), plus the Pearson correlation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M      # noqa: E402
import recording as R  # noqa: E402


def ang_diff(a, b):
    """Smallest difference between two axes (mod pi), in radians."""
    d = (a - b) % math.pi
    return torch.minimum(d, math.pi - d)


def recovery(P, truth):
    out = {}
    if truth is None:
        return out
    with torch.no_grad():
        for k in ("g", "logE", "g2"):
            if k == "g2" and float(truth["g2"].std()) < 1e-9:
                continue
            e, t = getattr(P, k).detach(), truth[k]
            sp = t.std().clamp(min=1e-12)
            out[k] = dict(rel_err=float((e - t).abs().median() / sp),
                          corr=float(torch.corrcoef(torch.stack([e, t]))[0, 1]),
                          bias=float((e - t).mean() / sp))
        d = ang_diff(P.phi.detach(), truth["phi"])
        w = truth["g"].clamp(min=0)
        out["phi"] = dict(median_deg=float(d.median() * 180 / math.pi),
                          weighted_median_deg=float((d * w).sum() / w.sum() * 180 / math.pi),
                          axis_agreement=float((w * torch.cos(2 * d)).sum() / w.sum()))
        if P.clock_mode == "free":
                out["clock"] = dict(t0=float("nan"), truth_t0=float(truth["clock"][0]), dur=float("nan"),
                                truth_dur=float(truth["clock"][2].exp()))
        else:
            out["clock"] = dict(t0=float(P.clock[0]), truth_t0=float(truth["clock"][0]),
                                dur=float(P.clock[2].exp()), truth_dur=float(truth["clock"][2].exp()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", default="planted", choices=["planted", "recording"])
    ap.add_argument("--beat", type=int, default=3)
    ap.add_argument("--per-parent", type=int, default=50)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--anchor", type=float, default=1e4)
    ap.add_argument("--drag", type=float, default=30.0, help="Stokes drag k; 150 critically damps the substrate mode (step_test.py)")
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--nu", type=float, default=0.3)
    ap.add_argument("--youngs", type=float, default=80.0)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--free", default="g,phi,logE,clock")
    ap.add_argument("--lr", default="g=2e-3,phi=0.03,logE=0.03,clock=0.05,delay=0.1,g2=2e-3,logtau=0.05")
    ap.add_argument("--delay-shrink", type=float, default=0.0,
                    help="weight on mean(delay^2) (frames^2): keeps per-cell timing near the shared clock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plant-gain", type=float, default=1.5)
    ap.add_argument("--plant-sigma-logE", type=float, default=0.3)
    ap.add_argument("--plant-g2", type=float, default=0.0,
                    help="planted transverse strain as a fraction of g (with 30%% per-cell scatter); 0 = rank-1 truth")
    ap.add_argument("--noise", default="none", choices=["none", "tracker", "rest"])
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--clock-mode", default="sigmoid", choices=["sigmoid", "free"])
    ap.add_argument("--clock-smooth", type=float, default=1e-2,
                    help="weight on the free clock's summed squared frame-to-frame steps")
    ap.add_argument("--E-shrink", type=float, default=0.0,
                    help="weight on mean((log E - mean log E)^2): shrinks the per-cell stiffness "
                         "toward uniform. The first live fit let log E spread to sd 2.0 (3 to 900) "
                         "against a gate tested at sd 0.3; the prior says how much of that the "
                         "data insists on")
    ap.add_argument("--mask-band", action="store_true",
                    help="leave cells with any particle in the prescribed band out of the loss")
    ap.add_argument("--specimen", default="healthy", choices=["healthy", "hcm"])
    ap.add_argument("--init-truth", default="", help="planted only: families started AT the truth "
                    "(diagnostic: isolates the identifiability of the free ones)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free = [f for f in args.free.split(",") if f]
    lrs = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.lr.split(",")}

    rec = R.load(device=dev, specimen=args.specimen)
    C = rec["n_cells"]
    LTIF = os.path.join(HERE, "data_hcm" if rec.get("specimen") == "hcm" else "data", "cells_2560.tif")
    win = R.beat_window(rec, args.beat)
    T = len(win["frames"])
    A_rec, u_rec = R.window_affine(rec, win)
    kw = dict(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=T - 1, n_cells=C,
              youngs=args.youngs, anchor_k=args.anchor, drag_k=args.drag)

    # rest positions + band from one 0-frame rollout (the seed decides where particles sit)
    P0 = M.Params(C, dev, nu=args.nu)
    with torch.no_grad():
        r0 = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="fit_rest",
                                               **dict(kw, n_frames=0))), P0, dev, C, grad=False)
    X0, cid = r0["X0"], r0["cid"]
    band = ((X0[:, 0] < M.DOM_LO + args.band) | (X0[:, 0] > M.DOM_HI - args.band)
            | (X0[:, 1] < M.DOM_LO + args.band) | (X0[:, 1] > M.DOM_HI - args.band))
    prescribe = (band, M.band_prescription(A_rec, u_rec, X0, cid, band))
    interior = M.interior_cells(cid, band, C) if args.mask_band else None
    n_int = int(interior.sum()) if interior is not None else C

    # ---- the target ----------------------------------------------------------------------
    phi_m, amp_m, _ = R.fibre_init(A_rec)
    ck = R.clock_init(A_rec)
    truth = None
    if args.target == "planted":
        gen = torch.Generator().manual_seed(1000 + args.seed)
        Pt = M.Params(C, dev, g0=0.0, phi0=phi_m, E0=args.youngs, t0=ck["t0"], tau_r=ck["tau_r"],
                      dur=ck["dur"], tau_d=ck["tau_d"], nu=args.nu)
        with torch.no_grad():
            Pt.g.copy_(amp_m * args.plant_gain)
            Pt.logE.copy_(math.log(args.youngs)
                          + args.plant_sigma_logE * torch.randn(C, generator=gen).to(dev))
            if args.plant_g2 != 0:
                Pt.g2.copy_((args.plant_g2 * Pt.g * (1 + 0.3 * torch.randn(C, generator=gen).to(dev))).clamp(-0.2, 0.2))
        with torch.no_grad():
            tgt = M.rollout(M.load_sim(M.build_spec(label_tif=LTIF, differentiable=False, name="fit_plant", **kw)),
                            Pt, dev, C, grad=False, prescribe=prescribe)
        A_t, u_t = tgt["A"].detach(), tgt["u"].detach()
        truth = {k: v.detach().clone() for k, v in Pt.leaves().items()}
        if args.noise != "none":
            nz = np.load(os.path.join(HERE, "data", f"noise_{args.noise}.npz"))
            dA = torch.as_tensor(nz["dA"], device=dev)[:T] * args.noise_scale
            du = torch.as_tensor(nz["du"], device=dev)[:T] * args.noise_scale
            A_t, u_t = A_t + dA, u_t + du
        # what the recording would offer as an init: the axis and clock read off the TARGET
        phi_i, amp_i, _ = R.fibre_init(A_t)
        ck_i = R.clock_init(A_t)
    else:
        A_t, u_t = A_rec, u_rec
        phi_i, amp_i, ck_i = phi_m, amp_m, ck

    P = M.Params(C, dev, g0=float(amp_i.median()), phi0=phi_i, E0=args.youngs, t0=ck_i["t0"],
                 tau_r=ck_i["tau_r"], dur=ck_i["dur"], tau_d=ck_i["tau_d"], nu=args.nu,
                 clock_mode=args.clock_mode, n_frames=T)
    if truth is not None and args.init_truth:
        with torch.no_grad():
            for k in args.init_truth.split(","):
                getattr(P, k).copy_(truth[k])
    eye = torch.eye(2, device=dev)
    w_A = 1.0 / ((A_t - eye) ** 2).mean()
    w_u = 1.0 / (u_t ** 2).mean()
    sim = M.load_sim(M.build_spec(label_tif=LTIF, differentiable=True, name="fit", **kw))
    groups = [dict(params=[getattr(P, k)], lr=lrs[k]) for k in free]
    opt = torch.optim.Adam(groups)
    # cosine from 1x to 0.05x of each group's own rate
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(i, args.iters) / max(args.iters, 1))))

    tag = args.tag or (f"{args.target}_b{args.beat}_p{args.per_parent}_k{args.anchor:g}"
                       f"_{'-'.join(free)}_n{args.noise}{args.noise_scale:g}_s{args.seed}")
    od = os.path.join(HERE, "out", "fits", tag)
    os.makedirs(od, exist_ok=True)
    print(f"  fit {tag}: {C} cells x {args.per_parent} pts ({X0.shape[0]:,}), window {win['span']} "
          f"({T} frames), free {free}, band {int(band.sum())} particles, {n_int} cells in the loss", flush=True)
    print(f"  init recovery: {json.dumps(recovery(P, truth))}", flush=True)

    log = []
    last_ok = {k: v.detach().clone() for k, v in P.leaves().items()}
    t_fit = time.time()
    for it in range(args.iters):
        t0 = time.time()
        opt.zero_grad()
        out = M.rollout(sim, P, dev, C, grad=True, prescribe=prescribe)
        loss = M.affine_loss(out["A"], out["u"], A_t, u_t, w_u=w_u, w_A=w_A, cells=interior)
        if args.clock_mode == "free":
            loss = loss + args.clock_smooth * P.smoothness()
        if args.E_shrink > 0:
            loss = loss + args.E_shrink * (P.logE - P.logE.mean()).pow(2).mean()
        if args.delay_shrink > 0:
            loss = loss + args.delay_shrink * P.delay.pow(2).mean()
        if not torch.isfinite(loss):
            # a diverged rollout (measured once: lambda 0.03, iteration 90, loss 0.67 -> 2.8 -> nan):
            # restore the last finite parameters, halve every learning rate, and go on
            with torch.no_grad():
                for k, v in last_ok.items():
                    getattr(P, k).copy_(v) if k != "clock" or P.clock_mode != "free" else P.gfree.copy_(v)
            for g_ in opt.param_groups:
                g_["lr"] *= 0.5
            print(f"  it {it:4d} loss not finite -- restored the last finite parameters, learning rates halved",
                  flush=True)
            sched.step()
            continue
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            P.g.clamp_(min=0.0)
            # HARD BOUNDS, not priors: a stiffness a hundred times off, or a delay past the clock's
            # flat tail, is a rollout the MPM cannot integrate (J -> 0, nan), not a hypothesis.
            P.logE.clamp_(min=math.log(args.youngs / 10), max=math.log(args.youngs * 10))
            P.delay.clamp_(min=-8.0, max=8.0)
            P.g2.clamp_(min=-0.2, max=0.2)
            P.logtau.clamp_(min=-1.5, max=1.5)
            last_ok = {k: v.detach().clone() for k, v in P.leaves().items()}
        rowd = dict(it=it, loss=float(loss), seconds=time.time() - t0, recovery=recovery(P, truth),
                    lr=[g_["lr"] for g_ in opt.param_groups])
        log.append(rowd)
        if it % 5 == 0 or it == args.iters - 1:
            rc = rowd["recovery"]
            extra = (f"  g rel {rc['g']['rel_err']:.3f} r {rc['g']['corr']:.3f} | logE rel "
                     f"{rc['logE']['rel_err']:.3f} r {rc['logE']['corr']:.3f} | phi "
                     f"{rc['phi']['weighted_median_deg']:.1f} deg | t0 {rc['clock']['t0']:.2f}/"
                     f"{rc['clock']['truth_t0']:.2f}") if truth is not None else ""
            print(f"  it {it:4d} loss {float(loss):.5f}  {rowd['seconds']:.1f} s{extra}", flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            np.savez(os.path.join(od, "params.npz"), **P.state_dict(),
                     interior=(interior.cpu().numpy() if interior is not None else np.ones(C, bool)),
                     **({f"true_{k}": v.cpu().numpy() for k, v in truth.items()} if truth else {}))
            json.dump(dict(config=vars(args), tag=tag, window=win["span"], n_cells=C, n_cells_in_loss=n_int,
                           particles=int(X0.shape[0]), log=log,
                           seconds_total=time.time() - t_fit,
                           peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 2 ** 30),
                      open(os.path.join(od, "fit.json"), "w"), indent=1)
    print(f"  done in {(time.time() - t_fit) / 60:.1f} min; final loss {log[-1]['loss']:.5f}; "
          f"peak mem {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GB  -> {od}")


if __name__ == "__main__":
    main()

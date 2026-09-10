"""fd_check -- S0: prove the gradient is the gradient, on this geometry, before anything is fitted.

The old trainer descended a surrogate 30% off its own loss for sixty batches, and the only test
that caught it was this one (discovery_cardio_mpm, commit e3b09e8c): a central difference through
the model's OWN forward at three step sizes. A ratio near 1 at one step size proves nothing; a
correct gradient CONVERGES to the finite difference as the step shrinks, and a plateau away from 1
is a wrong gradient no step size will rescue.

    d loss / d p  ~=  ( L(p + h) - L(p - h) ) / 2h        at h, h/4, h/16

Four scalars, one per family of learnable: one cell's g (shortening), one cell's phi (axis), one
cell's log E (stiffness), and the clock's log duration (global). The loss is the per-cell affine
mismatch against a PLANTED rollout, evaluated at flat parameters, so no gradient is zero by
symmetry.

Also measured, because morph.py found repeat runs of one configuration moving the loss 18-44%:
the forward's repeatability (same params twice) and the backward's (same gradient twice). Every
finite difference is judged against the forward floor: a step whose |L+ - L-| does not clear it
says nothing and is reported as such rather than averaged in.

    python fd_check.py --device cuda:0                     # the real 472-cell map (needs data/)
    python fd_check.py --device cuda:0 --smoke             # synthetic 472-cell Voronoi, wiring only
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
import model as M  # noqa: E402

SMOKE_TIF = os.path.abspath(os.path.join(HERE, "..", "..", "cardio_cells", "algebraic",
                                         "small_labels_full_472.tif"))


def planted(C, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = M.Params(C, device, g0=0.04, E0=80.0, t0=3.0, tau_r=1.0, dur=5.0, tau_d=1.5)
    with torch.no_grad():
        P.g.copy_((0.04 + 0.015 * torch.randn(C, generator=g)).clamp(0.005, 0.12).to(device))
        P.phi.copy_((torch.rand(C, generator=g) * np.pi).to(device))
        P.logE.copy_((np.log(80.0) + 0.3 * torch.randn(C, generator=g)).to(device))
    return P


def flat(C, device, phi_true, seed=1):
    g = torch.Generator().manual_seed(seed)
    P = M.Params(C, device, g0=0.04, E0=80.0, t0=3.0, tau_r=1.0, dur=5.0, tau_d=1.5)
    with torch.no_grad():
        P.phi.copy_(phi_true + 0.3 * torch.randn(C, generator=g).to(device))
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-parent", type=int, default=20)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--cell", type=int, default=100, help="the probed cell (1-based label)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                    help="float64 is the referee: a float32 forward through ~120 substeps carries "
                         "~1e-6 relative rounding noise, which at h/16 is a few per cent of the "
                         "finite difference and looks like a drifting ratio")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "fd_check.json"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False      # no TF32 in a referee
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(getattr(torch, args.dtype))

    tif = SMOKE_TIF if args.smoke else M.LABEL_TIF
    C = 472
    raw = M.build_spec(n_grid=args.n_grid, per_parent=args.per_parent, n_frames=args.frames,
                       n_cells=C, label_tif=tif, compile=not args.no_compile, name="fd_check")
    sim = M.load_sim(raw)
    dev = args.device

    print(f"  planted rollout: {C} cells x {args.per_parent} pts, grid {args.n_grid}, "
          f"{args.frames} frames", flush=True)
    t0 = time.time()
    P_true = planted(C, dev)
    with torch.no_grad():
        ref = M.rollout(sim, P_true, dev, C, grad=False)
    A_ref, u_ref = ref["A"].detach(), ref["u"].detach()
    eye = torch.eye(2, device=dev)
    print(f"    {time.time()-t0:.1f} s; planted |A-I| rms {((A_ref-eye)**2).mean().sqrt():.2e}, "
          f"|u| rms {(u_ref**2).mean().sqrt():.2e}, J range "
          f"{ref['J'].min():.4f}-{ref['J'].max():.4f}", flush=True)
    if float(((A_ref - eye) ** 2).mean().sqrt()) < 1e-5:
        print("  !! the planted rollout did not move: the strain injection is not reaching F")

    P = flat(C, dev, P_true.phi.detach())
    w_A = 1.0 / ((A_ref - eye) ** 2).mean()
    w_u = 1.0 / (u_ref ** 2).mean()

    def loss_at(grad):
        out = M.rollout(sim, P, dev, C, grad=grad)
        return M.affine_loss(out["A"], out["u"], A_ref, u_ref, w_u=w_u, w_A=w_A)

    # ---- repeatability floors -------------------------------------------------------------
    t0 = time.time()
    with torch.no_grad():
        L1 = float(loss_at(False)); L2 = float(loss_at(False))
    fwd_floor = abs(L1 - L2)
    s_fwd = (time.time() - t0) / 2
    print(f"  forward twice: {L1!r} / {L2!r}  ->  floor {fwd_floor:.3e} "
          f"({'bit-identical' if fwd_floor == 0 else 'NOT identical'}), "
          f"{(time.time()-t0)/2:.1f} s a forward", flush=True)

    grads = []
    for rep in range(2):
        for v in P.leaves().values():
            v.grad = None
        t0 = time.time()
        L = loss_at(True)
        L.backward()
        grads.append({k: v.grad.detach().clone() for k, v in P.leaves().items()})
        s_bwd = time.time() - t0
        print(f"  forward+backward {rep}: loss {float(L)!r}  ({s_bwd:.1f} s, "
              f"peak mem {torch.cuda.max_memory_allocated(dev)/2**30:.2f} GB)", flush=True)
    bwd = {k: float((grads[0][k] - grads[1][k]).abs().max() / (grads[0][k].abs().max() + 1e-30))
           for k in grads[0]}
    print(f"  backward repeat, max rel diff per family: "
          + "  ".join(f"{k} {v:.2e}" for k, v in bwd.items()), flush=True)
    for k, gk in grads[0].items():
        print(f"    grad {k}: |max| {gk.abs().max():.3e}  nonzero {(gk != 0).float().mean():.3f}")

    # ---- the finite differences ------------------------------------------------------------
    c = args.cell - 1
    probes = [("g", P.g, c, 0.01), ("phi", P.phi, c, 0.05), ("logE", P.logE, c, 0.1),
              ("clock_logdur", P.clock, 2, 0.1)]
    rows = []
    for name, ten, i, h0 in probes:
        an = float(grads[0][name if name != "clock_logdur" else "clock"][i])
        row = dict(param=name, index=int(i), analytic=an, steps=[])
        base = float(ten[i].detach())
        for k, h in enumerate([h0, h0 / 4, h0 / 16]):
            with torch.no_grad():
                ten[i] = base + h; Lp = float(loss_at(False))
                ten[i] = base - h; Lm = float(loss_at(False))
                ten[i] = base
            fd = (Lp - Lm) / (2 * h)
            signal = abs(Lp - Lm)
            # a step is trusted when its signal is far above the arithmetic: 1e-4 of the loss in
            # float32 (~1e3 ulps; the perturbation re-routes rounding through every substep, so
            # the repeat floor understates it), 1e-9 in float64
            ok = signal > (1e-4 if args.dtype == "float32" else 1e-9) * abs(L1)
            row["steps"].append(dict(h=h, fd=fd, ratio=(an / fd if fd != 0 else float("nan")),
                                     signal=signal, clears_floor=bool(ok)))
        rows.append(row)
        s = "  ".join(f"h={st['h']:.3g}: {st['ratio']:.4f}{'' if st['clears_floor'] else '*'}"
                      for st in row["steps"])
        print(f"  {name:<13s} analytic {an:+.5e}   {s}", flush=True)

    def fam_ok(r):
        good = [st for st in r["steps"] if st["clears_floor"]]
        return bool(good) and abs(good[-1]["ratio"] - 1) < 0.02
    verdict = all(fam_ok(r) for r in rows)
    print(f"\n  * = the step did not clear the forward floor; that ratio says nothing")
    print(f"  VERDICT: {'gradient certified' if verdict else 'NOT certified'} "
          f"(every family within 2% of its finite difference at the finest step that clears the floor)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(config=dict(per_parent=args.per_parent, n_grid=args.n_grid, frames=args.frames,
                               smoke=args.smoke, cell=args.cell, compile=not args.no_compile,
                               dtype=args.dtype),
                   loss=L1, forward_floor=fwd_floor, backward_repeat_rel=bwd, probes=rows,
                   seconds_forward=s_fwd, seconds_forward_backward=s_bwd,
                   peak_mem_gb=torch.cuda.max_memory_allocated(dev) / 2**30,
                   certified=verdict), open(args.out, "w"), indent=1)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()

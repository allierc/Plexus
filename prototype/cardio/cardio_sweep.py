#!/usr/bin/env python
"""4-hour autonomous LOSS-TERM SWEEP for the real cardio fit.

Defines four normalized trajectory loss terms and sweeps weighted recipes:
  mse    : per-node position MSE                       (overall fidelity)
  amp    : per-node RMS displacement-magnitude match   (forbids tiny-motion collapse)
  length : per-node total path length match            (how much the node travels)
  area   : per-node enclosed loop AREA match           (OPEN-LOOP-ness, not a degenerate line)
  shape  : per-frame increment (velocity) match        (loop traversal / direction)

Each recipe: UNet(image)->maps + scalars (k_anchor=0.06 spring restored to kill drift),
fit to the single strongest real beat, full 6-cycle render with boundary anchored to real.
Archives archive/train_<idx>_<name>/ (trajectories.png + metrics), writes a running
sweep_summary.md, and a montage comparing all recipes.

Run:  PYTHONPATH=src .../python prototype/cardio/cardio_sweep.py
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import cardio_stage2 as C            # noqa: E402
import cardio_real_fit as R          # noqa: E402
from cardio_unet import UNet, maps_from_unet, load_image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "archive", "sweep")
GRID = (48, 48)         # faster grid for the relative sweep
N_ITER = 260
K_ANCHOR = 0.06         # substrate spring restored (absolute restoring -> no drift)


# --------------------------------------------------------------------------- #
def traj_stats(d):
    """d [L,N,2] displacement -> (RMS magnitude, path length, |loop area|) per node."""
    A = d.pow(2).sum(-1).mean(0).clamp(min=1e-14).sqrt()
    Lp = (d[1:] - d[:-1]).norm(dim=-1).sum(0)
    area = 0.5 * (d[:-1, :, 0] * d[1:, :, 1] - d[:-1, :, 1] * d[1:, :, 0]).sum(0).abs()
    return A, Lp, area


def struct_tensor(d):
    """Per-node 2x2 structure tensor T = <d d^T> -> (Txx, Tyy, Txy), each [N]."""
    return (d[..., 0] ** 2).mean(0), (d[..., 1] ** 2).mean(0), (d[..., 0] * d[..., 1]).mean(0)


def loss_terms(pred, real, rstats):
    """Normalized loss terms vs real (lower=better). GEOMETRIC orientation/coherence via the
    structure tensor (mse/shape are geometry-agnostic; orient/coher are not)."""
    Ar, Lr, ar = rstats
    Ap, Lp, ap = traj_stats(pred)
    pxx, pyy, pxy = struct_tensor(pred); rxx, ryy, rxy = struct_tensor(real)
    th_p = 0.5 * torch.atan2(2 * pxy, pxx - pyy); th_r = 0.5 * torch.atan2(2 * rxy, rxx - ryy)
    amp_r = (rxx + ryy).clamp(min=1e-14).sqrt()
    coh_p = ((pxx - pyy) ** 2 + 4 * pxy ** 2).clamp(min=0).sqrt() / (pxx + pyy + 1e-12)
    coh_r = ((rxx - ryy) ** 2 + 4 * rxy ** 2).clamp(min=0).sqrt() / (ryy + rxx + 1e-12)
    return {
        "mse": (pred - real).pow(2).mean() / (real.pow(2).mean() + 1e-12),
        "amp": (Ap - Ar).pow(2).mean() / ((Ar ** 2).mean() + 1e-12),
        "length": (Lp - Lr).pow(2).mean() / ((Lr ** 2).mean() + 1e-12),
        "area": (ap - ar).pow(2).mean() / ((ar ** 2).mean() + 1e-12),
        "shape": (torch.diff(pred, dim=0) - torch.diff(real, dim=0)).pow(2).mean() /
                 (torch.diff(real, dim=0).pow(2).mean() + 1e-12),
        # GEOMETRIC: amplitude-weighted major-axis director distance (mod pi) -> orientation
        "orient": (amp_r * (1 - torch.cos(2 * (th_p - th_r)))).mean() / (amp_r.mean() + 1e-9),
        # eccentricity (line vs round-loop) match
        "coher": (coh_p - coh_r).pow(2).mean() / ((coh_r ** 2).mean() + 1e-9),
    }


CONFIGS = [
    ("mse",            {"mse": 1}),
    ("amp",            {"mse": 1, "amp": 5}),
    ("length",         {"mse": 1, "length": 5}),
    ("area",           {"mse": 1, "area": 5}),
    ("shape",          {"mse": 1, "shape": 5}),
    ("amp_area",       {"mse": 1, "amp": 3, "area": 3}),
    ("amp_length",     {"mse": 1, "amp": 3, "length": 3}),
    ("amp_shape",      {"mse": 1, "amp": 3, "shape": 3}),
    ("area_length",    {"mse": 1, "area": 3, "length": 3}),
    ("all",            {"mse": 1, "amp": 2, "length": 2, "area": 2, "shape": 2}),
    ("area_heavy",     {"mse": 1, "amp": 2, "area": 6}),
    ("length_heavy",   {"mse": 1, "amp": 2, "length": 6}),
    ("shape_heavy",    {"mse": 1, "amp": 2, "shape": 6}),
    ("amp_area_length",{"mse": 1, "amp": 2, "area": 3, "length": 3}),
]


def run_config(idx, name, w, img, geo, base_act, real, rstats, real_rec):
    net = UNet(); sc = R.Scalars()
    sc.k_anchor.requires_grad_(False)
    with torch.no_grad():
        sc.k_anchor.fill_(K_ANCHOR)
    params = [p for p in list(net.parameters()) + list(sc.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=4e-3)
    x = img[None, None]
    for it in range(N_ITER):
        opt.zero_grad()
        sk, gn, fb = maps_from_unet(net(x))
        pred = R.resample(R.forward_beat(sk, gn, fb, sc, geo, base_act))
        e = loss_terms(pred, real, rstats)
        loss = sum(w.get(k, 0.0) * e[k] for k in e)
        loss.backward(); opt.step()
        with torch.no_grad():
            sc.beta.clamp_(0.05, 2.0); sc.gamma.clamp_(0.05, 1.0); sc.aniso.clamp_(0.0, 1.2)
    with torch.no_grad():
        sk, gn, fb = maps_from_unet(net(x))
        ef = {k: float(v) for k, v in loss_terms(R.resample(R.forward_beat(sk, gn, fb, sc, geo, base_act)),
                                                 real, rstats).items()}
    fit_pos, lvl = R.full_forward(sk, gn, fb, sc, GRID)
    d = os.path.join(SWEEP, f"{idx:02d}_{name}"); os.makedirs(d, exist_ok=True)
    C.render_traj_png(fit_pos, os.path.join(d, "trajectories.png"), GRID, amp=R.AMP, real=real_rec)
    rec_sc = {k: round(getattr(sc, k).item(), 3) for k in ("beta", "k_anchor", "gamma", "aniso")}
    with open(os.path.join(d, "metrics.txt"), "w") as f:
        f.write(f"weights={w}\nfinal_rel_errors={ {k: round(v,4) for k,v in ef.items()} }\nscalars={rec_sc}\n")
    return ef, rec_sc


def montage(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    n = len(results); cols = 5; rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.1), facecolor="white")
    axs = np.atleast_1d(axs).ravel()
    for ax, (idx, name, ef, _) in zip(axs, results):
        p = os.path.join(SWEEP, f"{idx:02d}_{name}", "trajectories.png")
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_title(f"{idx:02d} {name}\nmse{ef['mse']:.2f} area{ef['area']:.2f} len{ef['length']:.2f}",
                     fontsize=8)
        ax.axis("off")
    for ax in axs[len(results):]:
        ax.axis("off")
    fig.suptitle("Loss-term sweep: real(blue) vs fit(green) trajectories  (lower rel-error = closer)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(SWEEP, "montage.png"), dpi=110); plt.close(fig)


def main():
    os.makedirs(SWEEP, exist_ok=True)
    img = load_image(GRID)
    geo, base_act = R.setup_fit(GRID)
    real = R.real_single_beat(GRID)
    rstats = traj_stats(real)
    real_rec = R.real_full(GRID, R.N_CYCLES * R.PERIOD)[::R.REC]
    print(f"sweep over {len(CONFIGS)} recipes, grid {GRID}, {N_ITER} iters each")

    results = []
    for k, (name, w) in enumerate(CONFIGS):
        idx = 10 + k
        try:
            ef, rec_sc = run_config(idx, name, w, img, geo, base_act, real, rstats, real_rec)
            results.append((idx, name, ef, rec_sc))
            print(f"  [{idx:02d} {name:16s}] rel-err " +
                  " ".join(f"{kk}={ef[kk]:.3f}" for kk in ("mse", "amp", "length", "area", "shape")) +
                  f"  beta={rec_sc['beta']}")
        except Exception:
            print(f"  [{idx:02d} {name}] FAILED\n" + traceback.format_exc())
        # incremental summary so partial progress survives
        with open(os.path.join(SWEEP, "sweep_summary.md"), "w") as f:
            f.write("# Loss-term sweep (relative errors vs real; lower=better)\n\n")
            f.write("| idx | recipe | mse | amp | length | area | shape | beta |\n|---|---|---|---|---|---|---|---|\n")
            for i, nm, ef, sc in results:
                f.write(f"| {i:02d} | {nm} | {ef['mse']:.3f} | {ef['amp']:.3f} | {ef['length']:.3f} "
                        f"| {ef['area']:.3f} | {ef['shape']:.3f} | {sc['beta']} |\n")
        if results:
            montage(results)
    print(f"done. summary -> {os.path.join(SWEEP, 'sweep_summary.md')}")


if __name__ == "__main__":
    main()

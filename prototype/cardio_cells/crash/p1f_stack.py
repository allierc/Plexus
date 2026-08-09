"""p1f_stack.py -- does STACKING frames across the active window rescue per-cell gain?

Probe A's declared gap (2) and probe B's declared gap (a): every P1 number sits at ONE frame
boundary. This measures the information ceiling of a multi-frame estimator directly, without
running an estimator: at each tick t, assemble the honest local Jacobian A_t at base = theta_true
with the simulator's own (true) F, form the dimensionless design D_t = A_t diag(theta_true) on
interior particles, and accumulate the Fisher/Gram  G_T = sum_t D_t^T D_t.

With b-noise iid of std sigma (position noise at both frame boundaries), the covariance of the
RELATIVE parameter error is sigma^2 G_T^-1. Per cell, the 2x2 sub-block at (c, C+c) gives the
best- and worst-determined directions in the (dE/E, dg/g) plane and their detectable sizes.

This is the CLEAN-F CEILING: F error (the campaign's actual blocker) enters A, not b, and is
ignored here. Every number is therefore optimistic.

Writes p1f_stack.json / .log. Modifies nothing.
usage: PYTHONPATH=/workspace/Plexus/src python p1f_stack.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
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

from finject import y_of                                              # noqa: E402
from freal_derivedF import plant_and_warm_x0, PX                      # noqa: E402
from round5_fit import SIGMA_X                                        # noqa: E402
import metrics as MET                                                 # noqa: E402


def assemble_at(sy, n_sub, base, sE=100.0, sg=1.0):
    y0 = y_of(sy, base, n_sub, None, None)
    A = torch.zeros(y0.numel(), 2 * sy.C, device=sy.device, dtype=sy.dtype)
    for j in range(2 * sy.C):
        s = sE if j < sy.C else sg
        e = base.clone()
        e[j] = e[j] + s
        A[:, j] = (y_of(sy, e, n_sub, None, None) - y0) / s
    torch.cuda.synchronize()
    return A


def percell(Ginv, C, sig_b):
    best_sd, worst_sd, ang = [], [], []
    for c in range(C):
        S = Ginv[[c, C + c]][:, [c, C + c]]
        ev, V = torch.linalg.eigh(S)
        sd = sig_b * torch.sqrt(ev.clamp(min=0))
        v = V[:, 0]
        if float(v[0]) < 0:
            v = -v
        best_sd.append(float(sd[0]))
        worst_sd.append(float(sd[1]))
        ang.append(math.degrees(math.atan2(float(v[1]), float(v[0]))))
    d = torch.diagonal(Ginv)
    sd_all = sig_b * torch.sqrt(d.clamp(min=0))
    bs, ws, an = torch.tensor(best_sd), torch.tensor(worst_sd), torch.tensor(ang)
    return {
        "sd_best_med": float(bs.median()), "sd_worst_med": float(ws.median()),
        "best_dir_angle_deg_med": float(an.median()),
        "best_dir_angle_deg_p10": float(an.quantile(0.1)),
        "best_dir_angle_deg_p90": float(an.quantile(0.9)),
        "n_best_under_0.10": int((bs < 0.10).sum()), "n_worst_under_0.10": int((ws < 0.10).sum()),
        "marg_E_med": float(sd_all[:C].median()), "marg_gain_med": float(sd_all[C:].median()),
        "n_E_under_0.10": int((sd_all[:C] < 0.10).sum()),
        "n_E_under_0.30": int((sd_all[:C] < 0.30).sum()),
        "n_gain_under_0.10": int((sd_all[C:] < 0.10).sum()),
        "n_gain_under_0.30": int((sd_all[C:] < 0.30).sum()),
        "marg_E_p90": float(sd_all[:C].quantile(0.9)),
        "marg_gain_p90": float(sd_all[C:].quantile(0.9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="p1f_stack")
    ap.add_argument("--ticks", default="155,160,165,170,175,179,100,120",
                    help="pulse ticks first (151-179 is the active window), then off-pulse ticks")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--nsub", type=int, default=10)
    a = ap.parse_args()

    ticks = [int(s) for s in a.ticks.split(",")]
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"args": vars(a), "sigma_x_world": SIGMA_X, "px_world": PX, "ticks": ticks}
    sig_b = math.sqrt(2.0) * SIGMA_X
    t_start = time.time()
    Gacc = None
    order = []

    with torch.no_grad():
        for t in ticks:
            args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                                   n_grid=a.n_grid, warmup=t, window=150, dtype="float64",
                                   mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
            sy, _ = plant_and_warm_x0(args, lambda s: None)
            C = sy.C
            th = sy.theta_true.double()
            x0 = sy.x0
            band = 0.06 / MET.SHEET_SPAN
            interior = ~((x0[:, 0] < band) | (x0[:, 0] > 1 - band)
                         | (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
            int_flat = interior[:, None].expand(-1, 2).reshape(-1)
            act = float(sy.act0.norm())
            A = assemble_at(sy, a.nsub, th.to(sy.dtype))
            D = A[int_flat].double() * th[None, :]
            G = (D.T @ D).cpu()
            ngain0 = int((D[:, C:].norm(dim=0) == 0).sum())
            log(f"[tick {t:4d}] ||act0|| {act:10.4g}  interior {int(interior.sum())}  "
                f"||D_E|| {float(D[:, :C].norm()):.4g}  ||D_gain|| {float(D[:, C:].norm()):.4g}  "
                f"zero gain cols {ngain0}/{C}")
            R.setdefault("per_tick", {})[str(t)] = {
                "act0_norm": act, "n_interior": int(interior.sum()),
                "norm_D_E": float(D[:, :C].norm()), "norm_D_gain": float(D[:, C:].norm()),
                "n_zero_gain_cols": ngain0}
            Gacc = G if Gacc is None else Gacc + G
            order.append(t)
            try:
                Ginv = torch.linalg.inv(Gacc)
                st = percell(Ginv, C, sig_b)
            except Exception as e:
                st = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            st["ticks_stacked"] = list(order)
            st["cond_G"] = float(torch.linalg.cond(Gacc))
            R.setdefault("cumulative", []).append(st)
            log(f"           stacked {len(order)}: cond(G) {st['cond_G']:.3e}  "
                f"marg rel sd  E med {st.get('marg_E_med', float('nan')):.4g} "
                f"gain med {st.get('marg_gain_med', float('nan')):.4g}  |  "
                f"E<=10% {st.get('n_E_under_0.10')}/{C}  gain<=10% "
                f"{st.get('n_gain_under_0.10')}/{C}  gain<=30% {st.get('n_gain_under_0.30')}/{C}  "
                f"| best-dir angle med {st.get('best_dir_angle_deg_med', float('nan')):+.1f} deg")
            del A, D, G, sy
            torch.cuda.empty_cache()

    R["wall_seconds"] = time.time() - t_start
    with open(os.path.join(HERE, f"{a.tag}.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    with open(os.path.join(HERE, f"{a.tag}.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"[done] {R['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()

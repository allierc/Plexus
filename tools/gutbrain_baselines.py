#!/usr/bin/env python
"""exp20's references, measured on a recording BEFORE any law is trained: experiments/exp20_gutbrain_graphcast/data/
baselines_<recording>.json (+ _cells.npz: the gut-responsive cells and the per-cell numbers the rulers reuse).

    PYTHONPATH=src python tools/gutbrain_baselines.py --recording gutbrain_glucose_f1 [--device cuda:0]

Every fit reads TRAINING frames only (the recording's `split` == 0); every score is on the HELD-OUT trials (the
exporter's `trials` column 6), from the origin o = onset (the pulse volume, which the exporter replaced by the volume
before it: the last frame without the pulse's effect; from onset - 1 step 1 would be that copy, exact for every
forecaster) through POST_S = 55 s:

  1. GUT-RESPONSIVE CELLS, the paper's selection (Methods, "Gut-responsive cell selection"; gutBrainPipeline
     `fusedLoocv` / `fitLeftGetThresh`), on the training frames: the FULL model (kernels on the gut-pulse train and on
     the all-UV-pulse train) against the PART model (all-UV only), smoothness-regularised distributed-lag regression
     in closed form (eq. 2: beta = (X^T X + lambda_R I + lambda_F D^T D)^-1 X^T Y, lambda_R 2, lambda_F 20, D the
     second difference, broken between regressors), each cell's correlation of prediction and trace; a cell is
     gut-responsive when its full-model correlation is above mode + 3 sigma of the left half-Gaussian AND above its
     part-model correlation AND its mean evoked change on the training gut pulses exceeds that on the control pulses.
     (The paper cross-validates per pulse and tests with Wilcoxon / Mann-Whitney; this is the same model in sample on
     the training frames, the tests replaced by the two inequalities: exp20 Decisions.)
  2. THE STIMULUS-TRIGGERED MEAN (STA), the target's comparison: x_hat(o + h) = x(o) + mean_k [x(o_k + h) - x(o_k)],
     k over the TRAINING pulses of the same kind (gut: sites != 1; control: site 1), per cell.
  3. THE PAPER'S REGRESSION as a forecaster (the full Fig. 3 model, GMV): kernels on gut UV (55 s), control UV
     (55 s), visual (5 s) and motor (10 s), fitted on the training frames; x_hat(o + h) = x(o) + y_hat(o + h) - y_hat(o).
  4. PERSISTENCE and the MEAN BASELINE (each cell's mean over its last W = 1..6 frames, the best W per h).
  5b. THE REPLICABLE CELLS: the top 1 % by positive t of the training gut pulses' evoked change, and their mean evoked
     change on the held-out gut pulses (G-Lglucose: an L-glucose fish's over a glucose fish's, recording and law alike).
  5. THE RECORDING'S EVOKED RATIO: the held-out control pulse's evoked change over the held-out gut pulses', in the
     gut-responsive cells (mean over 0-20 s after the pulse minus the PRE_S before).
  6. THE NOISE: sigma^2 from the structure function's intercept (exp17), the short-skill ceiling.

MSE is over cells per frame, then over the h of the window (and over the held-out trials of a kind), in float64.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
OUT = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")
LAMBDA_R, LAMBDA_F = 2.0, 20.0                  # Methods, "Regression analysis"
KERNEL_S = {"gut": 55.0, "ctrl": 55.0, "visual": 5.0, "motor": 10.0}
POST_S, PRE_S, EVOKED_S = 55.0, 10.0, 20.0
STD_THRESH = 3.0                                # Methods: 3 sigma_G + m
TOP1 = 0.01                                     # the replicable cells: this fraction, by positive t on training gut pulses
CTRL_SITE = 1


def lagged(x, n):
    """[T, n + 1]: column i is x delayed by i frames (zero-padded), the paper's lag matrix."""
    T = len(x)
    M = np.zeros((T, n + 1))
    for i in range(n + 1):
        M[i:, i] = x[:T - i]
    return M


def design(regs, orders):
    X = np.concatenate([lagged(r, n) for r, n in zip(regs, orders)], 1)
    p = X.shape[1]
    D = 2 * np.eye(p) - np.eye(p, k=1) - np.eye(p, k=-1)
    b = 0
    for n in orders[:-1]:                        # broken at every boundary between two regressors
        b += n + 1
        D[b - 1, b] = D[b, b - 1] = 0
    return X, LAMBDA_R * np.eye(p) + LAMBDA_F * D.T @ D


def fused_ridge(Y, regs, orders, fit, device):
    """Closed form on the `fit` frames; returns the prediction on EVERY frame [T, N] and the coefficients."""
    X, psi = design(regs, orders)
    Xt = torch.as_tensor(X, device=device, dtype=torch.float64)
    f = torch.as_tensor(fit, device=device)
    A = Xt[f].T @ Xt[f] + torch.as_tensor(psi, device=device)
    B = Xt[f].T.float() @ Y[f]
    beta = torch.linalg.solve(A.float(), B)
    return Xt.float() @ beta, beta


def corr_cols(a, b, rows):
    a, b = a[rows] - a[rows].mean(0), b[rows] - b[rows].mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-12)


def left_gauss_thresh(c, k=STD_THRESH):
    """The paper's null: the mode m of the correlations and the sd of their LEFT half about it; threshold m + k sd."""
    h, e = np.histogram(c, bins=200)
    m = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    left = c[c < m] - m
    sd = float(np.sqrt((left ** 2).mean())) if left.size else 0.0
    return m + k * sd, m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="gutbrain_glucose_f1")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    t0 = time.time()
    z = np.load(graphs_data_path("zebrafish", f"{a.recording}_recording.npz"))
    X = torch.as_tensor(z["dff"], device=a.device)
    S, tr, split = z["stimulus"], z["trials"], z["split"]
    T, N = X.shape
    dt = float(np.median(np.diff(z["t_s"])))
    pre, post, ev = int(round(PRE_S / dt)), int(np.ceil(POST_S / dt)), int(round(EVOKED_S / dt))
    fit = split == 0
    onset = tr[:, 0].astype(int)
    gut, held, full = tr[:, 2] != CTRL_SITE, tr[:, 6] > 0, tr[:, 5] > 0
    g_train = np.zeros(T); c_train = np.zeros(T); all_train = np.zeros(T)
    for f, g in zip(onset, gut):
        (g_train if g else c_train)[f] = 1.0
        all_train[f] = 1.0
    K = {k: int(round(v / dt)) for k, v in KERNEL_S.items()}

    # 1. gut-responsive cells
    full_p, _ = fused_ridge(X, [g_train, all_train], [K["gut"], K["ctrl"]], fit, a.device)
    part_p, _ = fused_ridge(X, [all_train], [K["ctrl"]], fit, a.device)
    rows = torch.as_tensor(np.where(fit)[0], device=a.device)
    r_full = corr_cols(full_p, X, rows).cpu().numpy()
    r_part = corr_cols(part_p, X, rows).cpu().numpy()
    del full_p, part_p
    thr, mode, sd = left_gauss_thresh(r_full)

    def evoked(f):                                 # [N]: mean over 0..EVOKED_S after onset minus the PRE_S before
        return (X[f:f + ev].mean(0) - X[max(0, f - pre):f].mean(0)).cpu().numpy()
    tr_g = [f for f, g, h, fu in zip(onset, gut, held, full) if g and not h and fu]
    tr_c = [f for f, g, h, fu in zip(onset, gut, held, full) if not g and not h and fu]
    ev_g, ev_c = np.mean([evoked(f) for f in tr_g], 0), np.mean([evoked(f) for f in tr_c], 0)
    resp = (r_full > thr) & (r_full > r_part) & (ev_g > ev_c)
    # THE REPLICABLE CELLS (G-Lglucose): the top TOP1 of cells by the POSITIVE t-score of their evoked change over the
    # training gut pulses -- the same count rule in every fish, so an L-glucose fish's set is noise that does not
    # replicate on its held-out pulses, a glucose fish's a response that does
    dg = np.stack([evoked(f) for f in tr_g])
    t_g = dg.mean(0) / (dg.std(0) / np.sqrt(len(tr_g)) + 1e-9)
    top1 = np.zeros(N, bool)
    top1[np.argsort(t_g)[::-1][:int(TOP1 * N)]] = True
    cells = torch.as_tensor(np.where(resp)[0], device=a.device)
    print(f"[cells] threshold {thr:.3f} (mode {mode:.3f}, left sd {sd:.3f}): {int(resp.sum()):,} of {N:,} gut-responsive "
          f"({resp.mean() * 100:.2f} %); training pulses gut {len(tr_g)}, control {len(tr_c)}")

    # 3. the paper's full regression (GMV) as a forecaster
    ctrl_tr = c_train
    gmv_p, _ = fused_ridge(X, [g_train, ctrl_tr, S[:, 3], S[:, 4]], [K["gut"], K["ctrl"], K["visual"], K["motor"]],
                           fit, a.device)

    # 2-4 on every held-out full-window trial
    out_trials = []
    H = post
    for f, g, h, fu in zip(onset, gut, held, full):
        if not (h and fu):
            continue
        o = f                                    # the pulse volume (replaced by the one before by the exporter): the
        hs = torch.arange(1, H + 1, device=a.device)   # last frame without the pulse's effect, step 1 a real frame
        tgt = X[o + hs]
        same = tr_g if g else tr_c
        sta = X[o] + torch.stack([X[k + hs] - X[k] for k in same]).mean(0)
        reg = X[o] + gmv_p[o + hs] - gmv_p[o]
        pers = X[o].expand_as(tgt)
        means = [X[o - w + 1:o + 1].mean(0).expand_as(tgt) for w in range(1, 7)]

        def mse(p, sel=None):
            d = (p - tgt) if sel is None else (p - tgt)[:, sel]
            return (d.double() ** 2).mean(1).cpu().numpy()
        row = {"onset": int(f), "site": int(tr[list(onset).index(f), 2]), "kind": "gut" if g else "control"}
        for nm, sel in (("all", None), ("resp", cells)):
            row[f"mse_sta_{nm}"] = mse(sta, sel).tolist()
            row[f"mse_reg_{nm}"] = mse(reg, sel).tolist()
            row[f"mse_pers_{nm}"] = mse(pers, sel).tolist()
            row[f"mse_mean_{nm}"] = np.min([mse(m, sel) for m in means], 0).tolist()
        row["evoked_resp"] = float(evoked(f)[resp].mean())
        row["evoked_top1"] = float(evoked(f)[top1].mean())
        out_trials.append(row)
        print(f"[trial] {row['kind']:7s} onset {f}: window MSE on responsive cells (1e-3) -- STA "
              f"{np.mean(row['mse_sta_resp']) * 1e3:.3f}, paper regression {np.mean(row['mse_reg_resp']) * 1e3:.3f}, "
              f"persistence {np.mean(row['mse_pers_resp']) * 1e3:.3f}, best mean {np.mean(row['mse_mean_resp']) * 1e3:.3f}; "
              f"evoked {row['evoked_resp']:+.4f}")
    gut_rows = [r for r in out_trials if r["kind"] == "gut"]
    ctrl_rows = [r for r in out_trials if r["kind"] == "control"]
    w = lambda rs, k: float(np.mean([np.mean(r[k]) for r in rs])) if rs else None
    reg_skill = 1 - w(gut_rows, "mse_reg_resp") / w(gut_rows, "mse_sta_resp")
    ratio = (np.mean([r["evoked_resp"] for r in ctrl_rows]) / np.mean([r["evoked_resp"] for r in gut_rows])
             if ctrl_rows and gut_rows else None)

    # 6. noise
    Dh = [float(((X[h:] - X[:-h]) ** 2).mean()) for h in (1, 2, 3)]
    sigma2 = max(float(np.polyfit([1, 2, 3], Dh, 1)[1]), 0.0) / 2
    res = {"recording": a.recording, "written": time.strftime("%Y-%m-%d %H:%M"), "tool": "tools/gutbrain_baselines.py",
           "frame_s": dt, "window": [pre, post], "cells": N, "gut_responsive": int(resp.sum()),
           "threshold": {"r": thr, "mode": mode, "left_sd": sd, "k": STD_THRESH},
           "training_pulses": {"gut": len(tr_g), "control": len(tr_c)},
           "trials": out_trials,
           "gut_window_mse_resp": {k: w(gut_rows, f"mse_{k}_resp") for k in ("sta", "reg", "pers", "mean")},
           "gut_window_mse_all": {k: w(gut_rows, f"mse_{k}_all") for k in ("sta", "reg", "pers", "mean")},
           "regression_skill_over_sta_gut_resp": reg_skill,
           "evoked_ratio_ctrl_over_gut": ratio, "top1_cells": int(top1.sum()),
           "evoked_top1_heldout_gut": float(np.mean([r["evoked_top1"] for r in gut_rows])) if gut_rows else None, "noise_sigma2": sigma2, "one_step_msd": Dh[0],
           "kernels_frames": K, "lambda": [LAMBDA_R, LAMBDA_F]}
    p = os.path.join(OUT, f"baselines_{a.recording}.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    np.savez(os.path.join(OUT, f"baselines_{a.recording}_cells.npz"), responsive=resp, r_full=r_full, r_part=r_part,
             evoked_gut=ev_g, evoked_ctrl=ev_c, top1=top1, t_gut=t_g)
    print(f"[done] paper regression skill over the STA, held-out gut trials, responsive cells: {reg_skill:+.3f}; "
          f"evoked ratio control / gut {ratio}; sigma^2 {sigma2:.2e}; -> {p} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

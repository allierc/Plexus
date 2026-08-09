"""freal_secondderiv.py -- EXPERIMENT 2.  DO CHANNELS 6..11 BUY A BETTER F INSIDE A CELL?

THE QUESTION
====================================================================================================
The recording's control points are 15 px apart and a cell spans only ~6 of them, so the injected F
is piecewise-constant at a spacing comparable to the object whose stiffness is being estimated.
Channels 6..11 of `...derivatives.npy` are the SECOND spatial derivatives, and the pipeline throws
them away.  Would a second-order local reconstruction

        F(x) ~ F(x_c) + (x - x_c) . grad F                      grad F from ch 6..11

beat nearest / bilinear interpolation of F onto particles?

TWO STAGES, THE CHEAP ONE FIRST
----------------------------------------------------------------------------------------------------
R (real, CPU, seconds).  `output_cedric.py` says ch 6..11 were produced by convolving the
    FIRST-derivative images with a [1,0,-1] kernel -- at FULL resolution, before the ::15
    subsampling.  So on the 15-px grid that survives they are NOT algebraically recoverable from
    ch 2..5; the open question is empirical: are they CONSISTENT with a coarse central difference
    of ch 2..5 (redundant), pure pixel noise (useless), or genuine sub-15-px structure (valuable)?
    Two measurements decide it:
      R1  correlation / least-squares scale of each of ch 6..11 against the 30-px central
          difference of the matching first-derivative channel, on the frozen eval mask.
      R2  a HELD-OUT NODE test that needs no model at all: decimate the grid by 2 (30-px spacing),
          predict the omitted 15-px node's ch 2..5 from its two neighbours by (a) nearest,
          (b) linear midpoint, (c) Taylor with ch 6..11, (d) Hermite with ch 6..11.  If (c)/(d) do
          not beat (b), the second derivatives buy nothing where they would have to buy it.

S (synthetic).  The same comparison where the truth is known, so the answer is not contaminated by
    the recording's own F error.  The true per-particle F is BINNED onto a control grid whose
    spacing is the recording-equivalent (cell span / 6), then reconstructed at the particles by
    nearest / bilinear / Taylor-with-node-difference-gradient / Taylor-with-IDEAL-gradient (a local
    weighted least-squares fit of the reference F field -- an UPPER BOUND on any measured grad F).
    Scored two ways: the F reconstruction error itself, and then the thing that matters -- inject
    it, fit theta, and report med|dE/E|, the fit residual and the HELD-OUT one-frame residual at
    tick 180 under xv_ladder's fixed protocol (floor 0.00474, null band 0.50-0.59).

usage:
  /workspace/.conda_envs/neural-graph-linux/bin/python freal_secondderiv.py --stages R
  PYTHONPATH=/workspace/Plexus/src python freal_secondderiv.py --stages SI --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np

ALG = "/workspace/Plexus/prototype/cardio_cells/algebraic"
DISC = "/workspace/Plexus/discovery_cardio_mpm"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("/workspace/Plexus/src", ALG, DISC, HERE):
    sys.path.insert(0, _p)

REAL = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1/"
        "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy")     # HEALTHY.  The diseased is SEALED.
FIT = (152, 201)                                                # split.json's fit beat

# channel names as written by output_cedric.py
CH1 = {2: "du/dx", 3: "du/dy", 4: "dv/dx", 5: "dv/dy"}
CH2 = {6: "d2u/dx2", 7: "d2u/dy2", 8: "d2u/dxdy", 9: "d2v/dx2", 10: "d2v/dy2", 11: "d2v/dxdy"}


# =============================================================================================== #
#  STAGE R -- the real recording.  numpy only.
# =============================================================================================== #
def stage_R(log, R, stride=3):
    D = np.load(REAL, mmap_mode="r")
    T, N0, N1, _ = D.shape
    X = np.asarray(D[0, :, :, 0], np.float64)
    Y = np.asarray(D[0, :, :, 1], np.float64)
    h0_X = float(np.median(np.diff(X[:, 0])))
    h1_X = float(np.median(np.diff(X[0, :])))
    h0_Y = float(np.median(np.diff(Y[:, 0])))
    h1_Y = float(np.median(np.diff(Y[0, :])))
    # which array axis carries which world coordinate
    ax_x = 0 if abs(h0_X) > abs(h1_X) else 1
    ax_y = 0 if abs(h0_Y) > abs(h1_Y) else 1
    hx = abs(h0_X) if ax_x == 0 else abs(h1_X)
    hy = abs(h0_Y) if ax_y == 0 else abs(h1_Y)
    R["geometry"] = {"shape": list(D.shape), "axis_of_x": ax_x, "axis_of_y": ax_y,
                     "h_x_px": hx, "h_y_px": hy,
                     "dX_axis0": h0_X, "dX_axis1": h1_X, "dY_axis0": h0_Y, "dY_axis1": h1_Y}
    log(f"[R.geom] {list(D.shape)}   X varies along axis {ax_x} (h = {hx:.3f} px), "
        f"Y along axis {ax_y} (h = {hy:.3f} px)")

    # frozen eval mask (split.json's rule, recomputed here so nothing is imported)
    u = np.asarray(D[:, :, :, 0], np.float64) - X[None]
    v = np.asarray(D[:, :, :, 1], np.float64) - Y[None]
    amp = np.sqrt(u ** 2 + v ** 2).max(0)
    M = amp > 0.2 * np.percentile(amp, 99)
    R["eval_mask"] = {"n_nodes": int(M.sum()), "n_total": int(M.size)}
    del u, v, amp
    log(f"[R.mask] {int(M.sum())} / {M.size} nodes")

    ts = list(range(FIT[0], FIT[1], stride))
    G = np.asarray(D[ts][:, :, :, 2:12], np.float64)            # [nt,N0,N1,10]: ch2..11
    first = {c: G[..., c - 2] for c in CH1}
    second = {c: G[..., c - 2] for c in CH2}
    R["frames_used"] = ts

    # ---------- helper: central difference along an ARRAY AXIS, per-pixel units --------------- #
    def cd(a, axis, h):
        """centred difference along array axis (1 = axis0 of the node grid, 2 = axis1), per px."""
        out = np.full_like(a, np.nan)
        sl = [slice(None)] * a.ndim
        sp, sm, sc = list(sl), list(sl), list(sl)
        sp[axis] = slice(2, None)
        sm[axis] = slice(0, -2)
        sc[axis] = slice(1, -1)
        out[tuple(sc)] = (a[tuple(sp)] - a[tuple(sm)]) / (2.0 * h)
        return out

    def pair(a, b, mask):
        m = mask & np.isfinite(a) & np.isfinite(b)
        aa, bb = a[m], b[m]
        if aa.size < 10:
            return {"n": int(aa.size)}
        r = float(np.corrcoef(aa, bb)[0, 1])
        s = float((aa * bb).sum() / max((bb * bb).sum(), 1e-300))      # a ~ s * b
        return {"n": int(aa.size), "pearson_r": r, "ls_scale_a_over_b": s,
                "rms_a": float(np.sqrt((aa ** 2).mean())),
                "rms_b": float(np.sqrt((bb ** 2).mean())),
                "rms_diff": float(np.sqrt(((aa - bb) ** 2).mean())),
                "rms_diff_over_rms_b": float(np.sqrt(((aa - bb) ** 2).mean())
                                             / max(np.sqrt((bb ** 2).mean()), 1e-300))}

    Mb = np.broadcast_to(M[None], (len(ts),) + M.shape)

    # ---------- R1: every second-derivative channel against every coarse cd ------------------- #
    axis_of = {"x": 1 + ax_x, "y": 1 + ax_y}                     # +1 for the leading time axis
    h_of = {"x": hx, "y": hy}
    coarse = {}
    for c in CH1:
        for d in ("x", "y"):
            coarse[(c, d)] = cd(first[c], axis_of[d], h_of[d])
    tab = {}
    for c2 in CH2:
        for (c1, d) in coarse:
            tab[f"ch{c2}({CH2[c2]}) ~ d/d{d} ch{c1}({CH1[c1]})"] = pair(second[c2],
                                                                        coarse[(c1, d)], Mb)
    R["R1_all_pairings"] = tab
    log("\n[R1] every measured second derivative vs every 30-px central difference of ch2..5")
    log(f"    {'measured':<18s} {'best coarse cd':<26s} {'r':>8s} {'scale':>8s} "
        f"{'rms_meas':>10s} {'rms_cd':>10s} {'rms_dif/rms_cd':>15s}")
    best = {}
    for c2 in CH2:
        cands = {k: v for k, v in tab.items() if k.startswith(f"ch{c2}(")}
        bk = max(cands, key=lambda k: abs(cands[k].get("pearson_r", 0.0)))
        best[c2] = (bk, cands[bk])
        w = cands[bk]
        log(f"    ch{c2:<2d} {CH2[c2]:<13s} {bk.split('~')[1].strip():<26s} "
            f"{w['pearson_r']:>8.4f} {w['ls_scale_a_over_b']:>8.4f} {w['rms_meas'] if False else w['rms_a']:>10.3e} "
            f"{w['rms_b']:>10.3e} {w['rms_diff_over_rms_b']:>15.3f}")
    R["R1_best_pairing"] = {f"ch{c}": {"pairing": bk, **st} for c, (bk, st) in best.items()}

    # ---------- R2: the held-out node test ----------------------------------------------------- #
    # decimate the grid by 2 along one axis; predict the omitted node from its two neighbours.
    # ground truth = the recording's OWN ch2..5 at that node.
    # second-derivative channel needed to Taylor-expand ch c1 along direction d:
    need = {(2, "x"): 6, (2, "y"): 8, (3, "x"): 8, (3, "y"): 7,
            (4, "x"): 9, (4, "y"): 11, (5, "x"): 11, (5, "y"): 10}
    log("\n[R2] HELD-OUT NODE: predict the omitted 15-px node's ch2..5 from a 30-px grid")
    log("     baselines use ch2..5 ONLY.  `cubic4` and `hermite_cd` spend the SAME four numbers as")
    log("     `hermite` but take the two slopes from ch2..5 instead of ch6..11 -- that is the")
    log("     control that decides whether ch6..11 carry anything of their own.")
    log(f"    {'channel':<12s} {'dir':>4s} {'rms_true':>9s} {'nearest':>9s} {'linear':>9s} "
        f"{'cubic4':>9s} {'hermite_cd':>10s} {'taylorSym':>10s} {'hermite':>9s} {'shuf_null':>10s} "
        f"{'/linear':>8s} {'/cubic4':>8s}")
    R["R2_heldout_node"] = {}
    R["R2_verdict"] = {}
    for st in (1, 2, 4):                            # coarse grid = 2*st*15 px; midpoint held out
        r2 = _r2_pass(first, second, need, axis_of, h_of, Mb, st, log,
                      verbose=(st == 1))
        R["R2_heldout_node"][f"step{st}"] = r2
        g = [v["gain_best2nd_vs_linear"] for v in r2.values()]
        gc = [v["gain_best2nd_vs_cubic4"] for v in r2.values()]
        R["R2_verdict"][f"step{st}"] = {
            "coarse_spacing_px": 2 * st * 15.0, "predicted_offset_px": st * 15.0,
            "median_gain_vs_linear": float(np.median(g)),
            "median_gain_vs_cubic4": float(np.median(gc)),
            "n_beating_cubic4": int(sum(x < 1.0 for x in gc)), "n_cases": len(g)}
        v = R["R2_verdict"][f"step{st}"]
        log(f"    [step {st}] coarse grid {v['coarse_spacing_px']:.0f} px, predicting "
            f"{v['predicted_offset_px']:.0f} px away:  best2nd/linear "
            f"{v['median_gain_vs_linear']:.3f}   best2nd/cubic4 {v['median_gain_vs_cubic4']:.3f}  "
            f"({v['n_beating_cubic4']}/{v['n_cases']} beat the cubic control)")

    # ---------- R3: how much of the 2nd-derivative channel is noise? --------------------------- #
    # the same quiet-stretch trick real_F_check used on F: in diastole the field is nearly static,
    # so the second TIME difference is 6 sigma^2 in variance.
    Q = (30, 49)
    Gq = np.asarray(D[Q[0] - 1:Q[1] + 2][:, :, :, 2:12], np.float64)
    d2t = Gq[2:] - 2 * Gq[1:-1] + Gq[:-2]
    Mq = np.broadcast_to(M[None], d2t.shape[:3])
    sig = {}
    for c in list(CH1) + list(CH2):
        j = c - 2
        s = float(np.sqrt((d2t[..., j][Mq] ** 2).mean() / 6.0))
        rms = float(np.sqrt((np.asarray(D[ts][:, :, :, c], np.float64)[Mb] ** 2).mean()))
        sig[f"ch{c}"] = {"sigma_temporal": s, "rms_signal": rms, "snr": rms / max(s, 1e-300)}
    R["R3_noise"] = sig
    log("\n[R3] temporal noise from the quiet stretch (frames 30-49), per channel")
    for c in list(CH1) + list(CH2):
        w = sig[f"ch{c}"]
        nm = CH1.get(c) or CH2.get(c)
        log(f"    ch{c:<2d} {nm:<9s} rms {w['rms_signal']:.4e}  sigma {w['sigma_temporal']:.4e}  "
            f"SNR {w['snr']:.2f}")
    return R


def _r2_pass(first, second, need, axis_of, h_of, Mb, st, log, verbose):
    r2 = {}
    rng = np.random.default_rng(0)
    for c1 in CH1:
        for d in ("x", "y"):
            ax = axis_of[d]
            h = h_of[d] * st
            c2 = need[(c1, d)]
            A = first[c1]
            Dv = second[c2]
            n = A.shape[ax]
            idx_o = np.arange(3 * st, n - 3 * st, 2 * st)
            sl = lambda i: tuple([slice(None)] * ax + [i] + [slice(None)] * (A.ndim - ax - 1))
            g = A[sl(idx_o)]
            gLL, gL, gR, gRR = (A[sl(idx_o - 3 * st)], A[sl(idx_o - st)],
                                A[sl(idx_o + st)], A[sl(idx_o + 3 * st)])
            dL, dR = Dv[sl(idx_o - st)], Dv[sl(idx_o + st)]
            # slopes available from ch2..5 alone, on the COARSE (2h) grid
            sL, sR = (gR - gLL) / (4 * h), (gRR - gL) / (4 * h)
            mk = (Mb[sl(idx_o)] & Mb[sl(idx_o - st)] & Mb[sl(idx_o + st)]
                  & Mb[sl(idx_o - 3 * st)] & Mb[sl(idx_o + 3 * st)])
            # shuffled null: same marginal distribution, local information destroyed
            perm = rng.permutation(dL.size)
            dLs = dL.reshape(-1)[perm].reshape(dL.shape)
            dRs = dR.reshape(-1)[perm].reshape(dR.shape)
            pred = {"nearest": gL,
                    "linear": 0.5 * (gL + gR),
                    "cubic4": (-gLL + 9 * gL + 9 * gR - gRR) / 16.0,
                    "hermite_cd": 0.5 * (gL + gR) + (2.0 * h / 8.0) * (sL - sR),
                    "taylorL": gL + h * dL,
                    "taylorSym": 0.5 * ((gL + h * dL) + (gR - h * dR)),
                    "hermite": 0.5 * (gL + gR) + (2.0 * h / 8.0) * (dL - dR),
                    "shuf_null": 0.5 * (gL + gR) + (2.0 * h / 8.0) * (dLs - dRs)}
            row = {"rms_true": float(np.sqrt((g[mk] ** 2).mean())), "n": int(mk.sum())}
            for kk, p in pred.items():
                row[kk] = float(np.sqrt(((p - g)[mk] ** 2).mean()))
            b2 = min(row["taylorL"], row["taylorSym"], row["hermite"])
            row["gain_best2nd_vs_linear"] = b2 / row["linear"]
            row["gain_best2nd_vs_cubic4"] = b2 / row["cubic4"]
            row["gain_best2nd_vs_hermite_cd"] = b2 / row["hermite_cd"]
            r2[f"ch{c1}({CH1[c1]}) along {d}"] = row
            if verbose:
                log(f"    ch{c1} {CH1[c1]:<7s} {d:>4s} {row['rms_true']:>9.3e} "
                    f"{row['nearest']:>9.3e} {row['linear']:>9.3e} {row['cubic4']:>9.3e} "
                    f"{row['hermite_cd']:>10.3e} {row['taylorSym']:>10.3e} {row['hermite']:>9.3e} "
                    f"{row['shuf_null']:>10.3e} {row['gain_best2nd_vs_linear']:>8.3f} "
                    f"{row['gain_best2nd_vs_cubic4']:>8.3f}")
    return r2



# =============================================================================================== #
#  STAGE S / I -- the synthetic system
# =============================================================================================== #
def bin_nodes(x, val, K, h, dev, dtype):
    """tent-weighted average of `val` [Np,D] onto a K x K node grid of spacing h.  Returns
    (node values [K,K,D], node weight [K,K])."""
    import torch
    D = val.shape[1]
    g = (x / h)
    i0 = torch.floor(g).long().clamp(0, K - 2)
    f = g - i0.to(dtype)
    acc = torch.zeros(K * K, D, device=dev, dtype=dtype)
    wsum = torch.zeros(K * K, device=dev, dtype=dtype)
    for a in (0, 1):
        for b in (0, 1):
            w = ((1 - f[:, 0]) if a == 0 else f[:, 0]) * ((1 - f[:, 1]) if b == 0 else f[:, 1])
            idx = (i0[:, 0] + a) * K + (i0[:, 1] + b)
            acc.index_add_(0, idx, w[:, None] * val)
            wsum.index_add_(0, idx, w)
    node = acc / wsum.clamp(min=1e-300)[:, None]                 # <- the AVERAGE, not the sum
    return node.view(K, K, D), wsum.view(K, K)


def fill_empty(node, w, thr, default):
    """nodes with too little support are replaced by the mean of their supported neighbours,
    iterated; anything still unreachable falls back to `default` (identity, for F)."""
    import torch
    ok = w > thr
    out = node.clone()
    out[~ok] = default
    n_empty = int((~ok).sum())
    cur = ok.clone()
    for _ in range(8):
        if bool(cur.all()):
            break
        s = torch.zeros_like(out)
        c = torch.zeros_like(w)
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = torch.roll(out, shifts=(da, db), dims=(0, 1))
            shc = torch.roll(cur.to(w.dtype), shifts=(da, db), dims=(0, 1))
            s = s + sh * shc[..., None]
            c = c + shc
        new = (~cur) & (c > 0)
        out[new] = (s[new] / c[new][..., None])
        cur = cur | new
    return out, n_empty


def ls_gradient(x, val, K, h, dev, dtype, node_val):
    """per-node weighted least-squares plane fit of the REFERENCE particle field `val`.

    Solves  min_over (a, gx, gy)  sum_p w_p | a + gx dx + gy dy - val_p |^2  with the same tent
    weight, giving the IDEAL local gradient of the true field at each node -- an upper bound on
    what any measured ch 6..11 could supply.  Returns (grad [K,K,2,D], intercept [K,K,D], rank_ok).
    """
    import torch
    D = val.shape[1]
    g = x / h
    i0 = torch.floor(g).long().clamp(0, K - 2)
    f = g - i0.to(dtype)
    Mm = torch.zeros(K * K, 3, 3, device=dev, dtype=dtype)
    rr = torch.zeros(K * K, 3, D, device=dev, dtype=dtype)
    for a in (0, 1):
        for b in (0, 1):
            w = ((1 - f[:, 0]) if a == 0 else f[:, 0]) * ((1 - f[:, 1]) if b == 0 else f[:, 1])
            idx = (i0[:, 0] + a) * K + (i0[:, 1] + b)
            dx = x[:, 0] - (i0[:, 0] + a).to(dtype) * h
            dy = x[:, 1] - (i0[:, 1] + b).to(dtype) * h
            P = torch.stack([torch.ones_like(dx), dx, dy], 1)            # [Np,3]
            Mm.index_add_(0, idx, w[:, None, None] * (P[:, :, None] * P[:, None, :]))
            rr.index_add_(0, idx, w[:, None, None] * (P[:, :, None] * val[:, None, :]))
    # conditioning without a batched eigensolve (cusolver's batched syevj fails above ~4e4 items):
    # scale the moment matrix to O(1) by its natural units (1, h, h) and test its determinant.
    S = torch.diag(torch.tensor([1.0, 1.0 / h, 1.0 / h], device=dev, dtype=dtype))
    Ms = S @ Mm @ S
    w0 = Mm[:, 0, 0]
    Ms = Ms / w0.clamp(min=1e-300)[:, None, None]
    ok = (w0 > 1e-10) & (torch.linalg.det(Ms) > 1e-4)
    sol = torch.zeros(K * K, 3, D, device=dev, dtype=dtype)
    if bool(ok.any()):
        sol[ok] = torch.linalg.solve(Mm[ok], rr[ok])
    grad = sol[:, 1:, :].view(K, K, 2, D)
    inter = sol[:, 0, :].view(K, K, D)
    bad = ~ok.view(K, K)
    grad[bad] = 0.0
    inter[bad] = node_val[bad]
    return grad, inter, int((~ok).sum())


def recon(x, node, grad_cd, grad_ls, K, h, dtype, inter=None):
    """the reconstructions of the node field at the particles.

      nearest        piecewise constant -- WHAT THE PIPELINE DOES TODAY
      bilinear       ch 2..5 only, four nodes
      taylor_cd      2nd order with the gradient taken from NODE DIFFERENCES (ch 2..5 only)
      taylor_ls      2nd order with the IDEAL local gradient (the ch 6..11 idea, upper bound)
      taylor_lsfull  the same, plus the local least-squares node value instead of the bin mean
    """
    import torch
    g = x / h
    i0 = torch.floor(g).long().clamp(0, K - 2)
    f = g - i0.to(dtype)
    inr = torch.round(g).long().clamp(0, K - 1)
    out = {}
    out["nearest"] = node[inr[:, 0], inr[:, 1]]
    bl = 0
    for a in (0, 1):
        for b in (0, 1):
            w = ((1 - f[:, 0]) if a == 0 else f[:, 0]) * ((1 - f[:, 1]) if b == 0 else f[:, 1])
            bl = bl + w[:, None] * node[i0[:, 0] + a, i0[:, 1] + b]
    out["bilinear"] = bl
    d = x - inr.to(dtype) * h
    for nm, gr in (("taylor_cd", grad_cd), ("taylor_ls", grad_ls)):
        if gr is None:
            continue
        G = gr[inr[:, 0], inr[:, 1]]                              # [Np,2,D]
        out[nm] = out["nearest"] + (d[:, :, None] * G).sum(1)
    if inter is not None:
        # ls_const is the CONTROL for taylor_lsfull: the same local least-squares node value,
        # WITHOUT the gradient term.  Their ratio is the gradient's contribution alone, free of
        # the node-estimation noise that separates `nearest` from `taylor_ls`.
        out["ls_const"] = inter[inr[:, 0], inr[:, 1]]
        if grad_ls is not None:
            G = grad_ls[inr[:, 0], inr[:, 1]]
            out["taylor_lsfull"] = out["ls_const"] + (d[:, :, None] * G).sum(1)
    return out


def anova_subcell(x, val, K, h):
    """fraction of the field's variance that lives INSIDE a control cell (nearest-node grouping).

    Interpolation-scheme free: it is the ceiling on what any node-value-only scheme can miss and
    any within-cell model could in principle recover.
    """
    import torch
    inr = torch.round(x / h).long().clamp(0, K - 1)
    idx = inr[:, 0] * K + inr[:, 1]
    D = val.shape[1]
    cnt = torch.zeros(K * K, device=val.device, dtype=val.dtype)
    s = torch.zeros(K * K, D, device=val.device, dtype=val.dtype)
    cnt.index_add_(0, idx, torch.ones_like(idx, dtype=val.dtype))
    s.index_add_(0, idx, val)
    mean = s / cnt.clamp(min=1.0)[:, None]
    within = (val - mean[idx]).pow(2).sum()
    total = (val - val.mean(0, keepdim=True)).pow(2).sum()
    return float(within / total)


def node_cd_grad(node, h):
    """central difference of the node field -- the gradient available from ch 2..5 ALONE."""
    import torch
    gx = torch.zeros(node.shape[0], node.shape[1], node.shape[2], device=node.device,
                     dtype=node.dtype)
    gy = torch.zeros_like(gx)
    gx[1:-1] = (node[2:] - node[:-2]) / (2 * h)
    gx[0] = (node[1] - node[0]) / h
    gx[-1] = (node[-1] - node[-2]) / h
    gy[:, 1:-1] = (node[:, 2:] - node[:, :-2]) / (2 * h)
    gy[:, 0] = (node[:, 1] - node[:, 0]) / h
    gy[:, -1] = (node[:, -1] - node[:, -2]) / h
    return torch.stack([gx, gy], 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="RSI")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default="freal_secondderiv")
    ap.add_argument("--cells", type=int, default=100)
    ap.add_argument("--per-parent", type=int, default=100)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--t0", type=int, default=165)
    ap.add_argument("--holdout-tick", type=int, default=180)
    ap.add_argument("--window", type=int, default=150)
    ap.add_argument("--pts-per-cell", type=float, default=6.0,
                    help="control points across a cell; 6 is the recording's value")
    ap.add_argument("--sweep", default="2,3,4,6,8,12")
    ap.add_argument("--rollout", action="store_true")
    a = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(str(s))

    R = {"argv": sys.argv, "config": vars(a)}
    t_start = time.time()

    if "R" in a.stages:
        log("=" * 100)
        log("STAGE R -- the recording.  Are ch 6..11 independent of ch 2..5?")
        log("=" * 100)
        stage_R(log, R)

    if "S" in a.stages or "I" in a.stages:
        import torch
        from plexus.models.entities import _lame                              # noqa: F401
        from assemble import rel as relres
        from recover import Solver, install_E, score
        import crash_test as CT
        from finject import lerp, record_substeps, assemble_inj, y_of
        from state_derive import collect, install_state

        log("\n" + "=" * 100)
        log("STAGE S -- the synthetic system.  Does SUB-CELL F structure matter?")
        log("=" * 100)
        args = SimpleNamespace(device=a.device, cells=a.cells, per_parent=a.per_parent,
                               n_grid=a.n_grid, warmup=a.t0, window=a.window, dtype="float64",
                               mode="full", e_lo=40.0, e_hi=220.0, g_lo=0.5, g_hi=1.5)
        torch.manual_seed(0)
        with torch.no_grad():
            sy, B = collect(args, a.t0 - 2, a.holdout_tick + 2, log)
            C, n, dt = sy.C, sy.n_sub_per_frame, sy.dt
            th = sy.theta_true.double()
            dev, dtype = sy.device, sy.dtype
            k = a.t0
            install_state(sy, B[k]["snap"])
            Fs, Cs, Xs = record_substeps(sy, n)
            x0 = B[k]["x0"]
            y_obs = (B[k]["x_next"] - x0).reshape(-1)
            F0, F1 = B[k]["F0"].clone(), B[k]["F1"].clone()
            assert float((Fs[-1] - F1).abs().max()) == 0.0
            log(f"[collect] tick {k}, C={C} Np={sy.Np} n_sub={n}  [{time.time()-t_start:.0f}s]")

            # ---- geometry: how big is a cell, in world units and in control points ----------- #
            cid = sy.cid
            span = []
            for c in range(1, C + 1):
                m = cid == c
                if int(m.sum()) < 3:
                    continue
                p = x0[m]
                span.append(float(0.5 * ((p[:, 0].max() - p[:, 0].min())
                                         + (p[:, 1].max() - p[:, 1].min()))))
            d_cell = float(np.median(span))
            h_star = d_cell / a.pts_per_cell
            K = int(round(1.0 / h_star)) + 1
            h_star = 1.0 / (K - 1)
            R["synth_geometry"] = {
                "cell_span_world_median": d_cell, "cell_span_dx": d_cell / sy.g.dx,
                "pts_per_cell_target": a.pts_per_cell, "K_nodes": K, "h_world": h_star,
                "h_over_particle_spacing": h_star / (1.0 / np.sqrt(sy.Np)),
                "particles_per_control_cell": sy.Np * h_star ** 2,
                "mpm_dx": sy.g.dx, "h_over_mpm_dx": h_star / sy.g.dx}
            log(f"\n[S.geom] median cell span {d_cell:.4f} world = {d_cell/sy.g.dx:.2f} mpm-dx; "
                f"a control grid with {a.pts_per_cell:g} points per cell is K={K} nodes, "
                f"h={h_star:.5f} ({sy.Np*h_star**2:.2f} particles per control cell)")
            log(f"          RECORDING equivalent: 137 nodes, 15 px, cell ~6 points")

            # ---- how much of F lives BELOW the control spacing? ------------------------------ #
            Fp = F0.reshape(sy.Np, 4)
            dev_from_I = Fp - torch.tensor([1.0, 0.0, 0.0, 1.0], device=dev,
                                           dtype=dtype)[None, :]
            R["F_field"] = {"rms_F_minus_I": float(dev_from_I.pow(2).mean().sqrt()),
                            "rms_F": float(Fp.pow(2).mean().sqrt())}

            Fid = torch.tensor([1.0, 0.0, 0.0, 1.0], device=dev, dtype=dtype)

            def measure(Fpart, KK, hh):
                nodeF, w = bin_nodes(x0, Fpart, KK, hh, dev, dtype)
                nodeF, n_empty = fill_empty(nodeF, w, 1e-8, Fid)
                gcd = node_cd_grad(nodeF, hh)
                gls, inter, n_bad = ls_gradient(x0, Fpart, KK, hh, dev, dtype, nodeF)
                return (recon(x0, nodeF, gcd, gls, KK, hh, dtype, inter), n_empty, n_bad)

            sweep = [float(s) for s in a.sweep.split(",")]
            log(f"\n[S.recon] F reconstruction error at the particles, "
                f"|F_rec - F_true| / |F_true - I|.  `nearest` is what the pipeline does today; "
                f"`bilinear` and `taylor_cd` use ch2..5 only; `taylor_ls*` is the ch6..11 idea "
                f"with an IDEAL gradient (upper bound).")
            log(f"    {'pts/cell':>9s} {'K':>5s} {'h/dx':>7s} {'p/cell':>7s} {'empty':>6s} "
                f"{'rankdef':>8s} {'subcellVar':>11s} {'nearest':>9s} {'bilinear':>9s} "
                f"{'taylor_cd':>10s} {'taylor_ls':>10s} {'ls_const':>9s} {'tay_lsfull':>11s} "
                f"{'lsfull/const':>13s}")
            R["S_sweep"] = {}
            store = {}
            den = float(dev_from_I.norm())
            for ppc in sweep:
                hh = d_cell / ppc
                KK = max(3, int(round(1.0 / hh)) + 1)
                hh = 1.0 / (KK - 1)
                rec, n_empty, n_bad = measure(Fp, KK, hh)
                row = {"K": KK, "h": hh, "particles_per_cell": sy.Np * hh ** 2,
                       "n_empty_nodes": n_empty, "n_rank_deficient": n_bad,
                       "subcell_var_frac": anova_subcell(x0, dev_from_I, KK, hh)}
                for nm, Fr in rec.items():
                    row[nm] = float((Fr - Fp).norm() / den)
                    # is the bulk error or a few blown-up nodes doing the work?
                    e = (Fr - Fp).norm(dim=1) / (dev_from_I.norm(dim=1) + 1e-300)
                    q = torch.quantile(e, torch.tensor([0.5, 0.9, 0.99], device=dev, dtype=dtype))
                    row[f"{nm}_percell_q50_q90_q99"] = [float(t) for t in q]
                    row[f"{nm}_frac_err2_in_top1pct"] = float(
                        (Fr - Fp).norm(dim=1).pow(2).topk(max(1, sy.Np // 100)).values.sum()
                        / (Fr - Fp).norm().pow(2))
                R["S_sweep"][f"{ppc:g}"] = row
                store[ppc] = (KK, hh)
                log(f"    {ppc:>9g} {KK:>5d} {hh/sy.g.dx:>7.2f} {sy.Np*hh**2:>7.2f} "
                    f"{n_empty:>6d} {n_bad:>8d} {row['subcell_var_frac']:>11.4f} "
                    f"{row['nearest']:>9.4f} {row['bilinear']:>9.4f} "
                    f"{row['taylor_cd']:>10.4f} {row['taylor_ls']:>10.4f} "
                    f"{row['ls_const']:>9.4f} {row['taylor_lsfull']:>11.4f} "
                    f"{row['taylor_lsfull']/row['ls_const']:>13.4f}")

            # ============================ STAGE I: does it change the recovered theta? ======= #
            if "I" in a.stages:
                log("\n" + "=" * 100)
                log("STAGE I -- inject each reconstruction and fit theta.  "
                    "acceptance statistic = the HELD-OUT one-frame residual")
                log("=" * 100)
                hk = a.holdout_tick
                injh = lerp(B[hk]["F0"], B[hk]["F1"], n)
                yh_obs = (B[hk]["x_next"] - B[hk]["x0"]).reshape(-1)

                def holdout(theta):
                    install_state(sy, B[hk]["snap"], None, None, Jp_one=True)
                    yo = y_of(sy, theta, n, injh, None)
                    return float((yo - yh_obs).norm() / yh_obs.norm())

                gperm = torch.Generator(device=dev).manual_seed(1234)
                th_perm = torch.cat([th[:C][torch.randperm(C, generator=gperm, device=dev)],
                                     th[C:][torch.randperm(C, generator=gperm, device=dev)]])
                R["holdout_band"] = {"theta_true_floor": holdout(th),
                                     "theta_permuted_null": holdout(th_perm),
                                     "theta_zero_null": holdout(torch.zeros_like(th))}
                log(f"[I.band] held-out floor {R['holdout_band']['theta_true_floor']:.5f}  "
                    f"permuted null {R['holdout_band']['theta_permuted_null']:.5f}  "
                    f"zero null {R['holdout_band']['theta_zero_null']:.5f}  "
                    f"(xv_ladder: 0.00474 / 0.59239 / 0.16934)")

                KK, hh = store[a.pts_per_cell] if a.pts_per_cell in store else (K, h_star)
                # the two frame boundaries, measured on the SAME control grid (Lagrangian labels
                # frozen at the frame start -- the frame displacement is 0.32 mpm-dx, 4% of h)
                variants = {tag: measure(Fb.reshape(sy.Np, 4), KK, hh)[0]
                            for tag, Fb in (("0", F0), ("1", F1))}

                LAD = [("none", None),
                       ("F_lerp_TRUE", lerp(F0, F1, n)),
                       ("F_true_substep", Fs)]
                for nm in ("nearest", "bilinear", "taylor_cd", "ls_const", "taylor_lsfull"):
                    LAD.append((f"F_{nm}", lerp(variants["0"][nm].reshape(sy.Np, 2, 2),
                                                variants["1"][nm].reshape(sy.Np, 2, 2), n)))
                # ---- the TOLERANCE curve.  Every reconstruction above may die for the same
                #      reason, in which case the ranking between them is beside the point and the
                #      quantity to report is HOW ACCURATE an F the fit needs.  Blend the winning
                #      reconstruction's error towards zero and read the requirement off directly.
                base0 = variants["0"]["taylor_lsfull"].reshape(sy.Np, 2, 2)
                base1 = variants["1"]["taylor_lsfull"].reshape(sy.Np, 2, 2)
                for al in (0.003, 0.01, 0.03, 0.1, 0.3):
                    LAD.append((f"blend{al:g}",
                                lerp(F0 + al * (base0 - F0), F1 + al * (base1 - F1), n)))
                R["I_ladder"] = {}
                thetas = {}
                log(f"\n[I] control grid K={KK}, h={hh:.5f} ({a.pts_per_cell:g} points per cell); "
                    f"frame cadence n_sub={n}, displacement read-out")
                log(f"    {'variant':<16s} {'F_err':>8s} {'affinity':>10s} {'bias':>10s} "
                    f"{'fit_res':>9s} {'medE':>8s} {'medg':>8s} {'HELDOUT':>9s}")
                for nm, iF in LAD:
                    install_state(sy, B[k]["snap"], None, None, Jp_one=True)
                    Ferr = (float((iF[-1] - Fs[-1]).norm() / dev_from_I.norm())
                            if iF is not None else float("nan"))
                    A, y0, _ = assemble_inj(sy, n, iF, None)
                    y_self = y_of(sy, th, n, iF, None)
                    b_self = y_self - y0
                    aff = relres(A @ th - b_self, b_self)
                    bias = float((y_self - y_obs).norm() / y_obs.norm())
                    b = y_obs - y0
                    S = Solver(A, C)
                    t_hat = S(b)["ridge0"]
                    s0 = score(t_hat, th, C)
                    ho = holdout(t_hat)
                    R["I_ladder"][nm] = {"F_series_err": Ferr, "affinity": aff, "bias": bias,
                                         "fit_residual": relres(A @ t_hat - b, b),
                                         "cond": S.cond, "scores": s0, "holdout": ho}
                    thetas[nm] = t_hat.clone()
                    log(f"    {nm:<16s} {Ferr:>8.4f} {aff:>10.3e} {bias:>10.3e} "
                        f"{R['I_ladder'][nm]['fit_residual']:>9.3e} {s0['med_E']:>8.4f} "
                        f"{s0['med_gain']:>8.4f} {ho:>9.5f}")
                    S.free(); del A, S
                    torch.cuda.empty_cache()
                np.savez(os.path.join(HERE, f"theta_{a.tag}.npz"),
                         **{f"inj::{kk}": v.cpu().numpy() for kk, v in thetas.items()})

                # ---- the free rollout, margin 20, 2-D gauge --------------------------------- #
                if a.rollout:
                    import crash_round3 as R3
                    import metrics as MET
                    install_state(sy, B[k]["snap"], None, None, Jp_one=True)
                    G = a.window
                    tracers = {m: CT.tracer_indices(x0, CT.probe_points(m))
                               for m in (MET.MARGIN_SAFE,)}
                    band = 0.06 / MET.SHEET_SPAN
                    anchor = ((x0[:, 0] < band) | (x0[:, 0] > 1 - band) |
                              (x0[:, 1] < band) | (x0[:, 1] > 1 - band))
                    interior = ~anchor
                    ref_full = torch.zeros(G, sy.Np, 2, device=dev, dtype=dtype)
                    from assemble import SUBSTEP_TOKENS
                    sy.restore()
                    install_E(sy, sy.E_true)
                    for kk in range(G):
                        sy._outer(k + kk, gain_cell=sy.gain_true)
                        sy.H.sub_dt = sy.dt_sub
                        for _ in range(n):
                            for tok in SUBSTEP_TOKENS:
                                sy._tok(tok)
                        sy.H.sub_dt = None
                        ref_full[kk] = sy.p.get("pos")
                    d_ref = ref_full - x0[None]
                    dm = d_ref[:, interior].mean(0, keepdim=True)
                    ss_tot = (d_ref[:, interior] - dm).pow(2).sum()
                    real20 = ref_full[:, tracers[MET.MARGIN_SAFE]].cpu().numpy()

                    def scored(theta):
                        tr, _, coarse = CT.rollout(sy, theta, k, G, tracers, ref_full=ref_full,
                                                   anchor=None, interior=interior, ss_tot=ss_tot,
                                                   keep_full=False, band_mask=anchor)
                        m20 = CT.read_metrics(tr[MET.MARGIN_SAFE].cpu().numpy(), real20)
                        return {"margin20": m20, "coarse": coarse,
                                "t1": coarse["motion_energy_ratio_interior"],
                                "t2": R3.t2_of(m20)}

                    R["I_rollouts"] = {}
                    ROLL = ["theta_true", "none", "F_lerp_TRUE", "F_nearest", "F_bilinear",
                            "F_taylor_lsfull", "blend0.01", "blend0.1"]
                    log(f"\n[I.rollout] free {G}-frame rollout, margin 20, 2-D gauge")
                    for nm in ROLL:
                        theta = th if nm == "theta_true" else thetas[nm]
                        raw = scored(theta)

                        def probe(lE, lg, theta=theta):
                            d = scored(R3.scale2(theta, float(np.exp(lE)), float(np.exp(lg)), C))
                            return (d["t1"], d["t2"])
                        gf = R3.gauge_fix2(probe, (raw["t1"], raw["t2"]))
                        kE, kg = gf["k_E"], gf["k_g"]
                        gau = raw if (kE == 1.0 and kg == 1.0) else scored(
                            R3.scale2(theta, kE, kg, C))
                        R["I_rollouts"][nm] = {"raw": raw, "gauged": gau, "gauge": gf}
                        log(f"    {nm:<16s} raw loop {CT.fmt(raw['margin20']['loopscore'],8)} | "
                            f"kE {kE:>6.3f} kg {kg:>6.3f} gauged loop "
                            f"{CT.fmt(gau['margin20']['loopscore'],8)} R2 "
                            f"{CT.fmt(gau['coarse']['R2_displacement_interior'],8)}")

    R["wall_seconds"] = time.time() - t_start
    json.dump(R, open(os.path.join(HERE, f"{a.tag}.json"), "w"), indent=1, default=str)
    open(os.path.join(HERE, f"{a.tag}.log"), "w").write("\n".join(lines) + "\n")
    log(f"\nwrote {a.tag}.json / .log [{R['wall_seconds']:.0f} s]")


if __name__ == "__main__":
    main()

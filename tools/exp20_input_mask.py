#!/usr/bin/env python
"""exp20: which cells each KNOWN INPUT enters directly -- the input mask, model-free, from training frames only (exp17's
`input_mask`, batches 10-14; Cedric 2026-10-02: "define the input neuron mask as in exp17").

    PYTHONPATH=src python tools/exp20_input_mask.py --recording gutbrain_glucose_f1 [--top 0.10] [--device cuda:0]

No law, no learned quantity, no held-out frame (the recording's `split` == 0 only), no pulse-site label:

  uv      the TRIAL-LOCKED t-score of each cell's evoked change after the training UV pulses of EVERY site (onset to
          EVOKED_S after, minus the PRE_S before; mean over pulses / its standard error). Coherence is poor for 14
          sparse pulses; the site is not used, so the mask is not told which pulses were on the gut. Signed: a cell
          that is reliably SUPPRESSED scores as high as one excited (|t|).
  visual  exp17's magnitude-squared coherence (Welch, NSEG-frame Hann segments, half overlap) between the cell and
          the grating column, averaged over the column's N_FREQ strongest frequencies, on the training frames
          (the held-out windows cut out and the rest joined)
  swim    the same with the swim power (the live channel(s); the strongest of them per cell)

Each input keeps its own top `--top` fraction. Written to graphs_data/zebrafish/input_mask_<recording>.npz:
    mask           [N]     1 if ANY input enters the cell (the union; the array exp17's `input_mask` reads)
    mask_by_input  [N, 6]  per forcing column (uv, uv_x, uv_y share the uv mask; swim_l, swim_r the swim mask)
    score_uv, score_visual, score_swim [N], thresholds {input: value}, top
and a montage experiments/exp20_gutbrain_graphcast/png/input_mask_<recording>.png (each input's cells, top view).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
EXP = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast")
NSEG, N_FREQ = 256, 3
PRE_S, EVOKED_S = 10.0, 20.0


def segments(x, nseg=NSEG):
    T = x.shape[0]
    w = torch.hann_window(nseg, periodic=False, device=x.device, dtype=x.dtype).reshape(nseg, *([1] * (x.dim() - 1)))
    return torch.stack([(x[s:s + nseg] - x[s:s + nseg].mean(0)) * w for s in range(0, T - nseg + 1, nseg // 2)], 0)


def coherence(X, u, n_freq=N_FREQ, chunk=20000):
    """[N]: magnitude-squared coherence between every column of X [T, N] and u [T], averaged over u's n_freq
    strongest frequencies (DC excluded)."""
    U = torch.fft.rfft(segments(u[:, None]), dim=1)[..., 0]                        # [S, F]
    Puu = (U.abs() ** 2).mean(0)
    band = torch.argsort(Puu[1:], descending=True)[:n_freq] + 1
    out = []
    for s in range(0, X.shape[1], chunk):
        Xf = torch.fft.rfft(segments(X[:, s:s + chunk]), dim=1)                   # [S, F, n]
        Pxx = (Xf.abs() ** 2).mean(0)
        Pxu = (Xf * U.conj()[..., None]).mean(0)
        c = (Pxu.abs() ** 2) / (Pxx * Puu[:, None]).clamp(min=1e-20)
        out.append(c[band].mean(0))
    return torch.cat(out).cpu().numpy(), band.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="gutbrain_glucose_f1")
    ap.add_argument("--top", type=float, default=0.10)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{a.recording}_recording.npz"))
    prov = json.load(open(graphs_data_path("zebrafish", f"{a.recording}_recording.json")))
    X = torch.as_tensor(z["dff"], device=a.device)
    S, tr, split, P = z["stimulus"], z["trials"], z["split"], z["pos_um"]
    T, N = X.shape
    dt = float(np.median(np.diff(z["t_s"])))
    pre, ev = int(round(PRE_S / dt)), int(round(EVOKED_S / dt))

    # uv: trial-locked t-score over the training pulses, every site
    on = [int(f) for f, h in zip(tr[:, 0], tr[:, 6]) if not h and f - pre >= 0 and f + ev <= T
          and (split[int(f) - pre:int(f) + ev] == 0).all()]
    d = torch.stack([X[f:f + ev].mean(0) - X[f - pre:f].mean(0) for f in on])     # [P, N]
    t_uv = (d.mean(0) / (d.std(0) / np.sqrt(len(on))).clamp(min=1e-9)).abs().cpu().numpy()

    # visual, swim: coherence on the training frames
    keep = torch.as_tensor(np.where(split == 0)[0], device=a.device)
    Xt = X[keep]
    St = torch.as_tensor(S, device=a.device)[keep]
    c_vis, band_vis = coherence(Xt, St[:, 3])
    live = [c for c, ok in zip((4, 5), prov.get("swim_live", [True, True])) if ok]
    sw = [coherence(Xt, St[:, c]) for c in live]
    c_swim = np.max([s[0] for s in sw], 0) if sw else np.zeros(N)
    scores = {"uv": t_uv, "visual": c_vis, "swim": c_swim}
    thr = {k: float(np.quantile(v, 1 - a.top)) for k, v in scores.items()}
    m = {k: (v > thr[k]) for k, v in scores.items()}
    by = np.stack([m["uv"], m["uv"], m["uv"], m["visual"], m["swim"], m["swim"]], 1).astype(np.float32)
    union = by.max(1)
    out = graphs_data_path("zebrafish", f"input_mask_{a.recording}.npz")
    np.savez(out, mask=union, mask_by_input=by, score_uv=t_uv, score_visual=c_vis, score_swim=c_swim,
             thresholds=json.dumps(thr), top=a.top, training_pulses=np.array(on))
    ov = {f"{i}&{j}": int((m[i] & m[j]).sum()) for i, j in (("uv", "visual"), ("uv", "swim"), ("visual", "swim"))}
    print(f"[mask] {a.recording}: {N:,} cells, top {a.top:.0%} per input -- thresholds "
          + ", ".join(f"{k} {v:.3f}" for k, v in thr.items())
          + f"; union {int(union.sum()):,} cells ({union.mean() * 100:.1f} %); overlaps {ov}; uv from {len(on)} pulses; "
          f"visual band periods {[round(NSEG * dt / b, 1) for b in band_vis]} s -> {out}")
    resp_p = os.path.join(EXP, "data", f"baselines_{a.recording}_cells.npz")
    if os.path.exists(resp_p):
        r = np.load(resp_p)["responsive"]
        print(f"[mask] the {int(r.sum()):,} gut-responsive cells (baselines): {np.mean(m['uv'][r]) * 100:.0f} % in the "
              f"uv mask, {np.mean(m['visual'][r]) * 100:.0f} % visual, {np.mean(m['swim'][r]) * 100:.0f} % swim")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), facecolor="black")
    for a_, (k, col) in zip(ax, (("uv", "#ff4040"), ("visual", "#4fc3f7"), ("swim", "#81c784"))):
        a_.set_facecolor("black")
        a_.scatter(P[::20, 0], P[::20, 1], s=0.05, c="0.3", lw=0)
        a_.scatter(P[m[k], 0], P[m[k], 1], s=0.15, c=col, lw=0)
        a_.set_aspect("equal"); a_.axis("off")
        a_.set_title(f"{k}: top {a.top:.0%} ({int(m[k].sum()):,} cells)", color="white", fontsize=10)
    os.makedirs(os.path.join(EXP, "png"), exist_ok=True)
    png = os.path.join(EXP, "png", f"input_mask_{a.recording}.png")
    fig.savefig(png, dpi=130, facecolor="black", bbox_inches="tight")
    print(f"[mask] montage -> {png}")


if __name__ == "__main__":
    main()

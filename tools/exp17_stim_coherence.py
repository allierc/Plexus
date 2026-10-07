"""exp17: which neurons FOLLOW THE STIMULUS, model-free (Cedric, 2026-10-01; the B_i-based version is
tools/exp17_input_coherence.py, which needs a trained law and favours the neurons the law was fitted on).

Local analysis. For every neuron of a recording, the magnitude-squared coherence between its trace and EACH of the
stimulus features (Welch: 256-frame Hann segments, half overlap, 0.914 s frames), on the GPU; per neuron the BEST
feature's coherence averaged over the stimulus's 6 strongest frequencies (the band), and which feature that is.
No learned quantity enters: the stimulus and the recording only.

  --recording zapbench | zapbench_destripe | zapbench_destripe_ephys   (graphs_data/zebrafish/<name>_recording.npz)
  --band-cols 22      the band (the stimulus's 6 strongest frequencies) from the live features among the first 22
                      stimulus columns only, i.e. the visual ones: the 5 ephys columns of zapbench_destripe_ephys are
                      scored in the SAME band as before, and the band does not depend on how they were scaled
  --mask-out NAME     also write graphs_data/zebrafish/NAME: mask [N] 1/0 (the top --top fraction of neurons by
                      best-feature coherence), coherence, threshold, best_feature (an index into ALL stimulus columns)

Writes experiments/exp17_zapbench_graphcast/data/stim_coherence_<recording>.npz (band coherence and best feature per
neuron) and the threshold montage png/stim_coherence_thresholds_<recording>.png (top 50 / 25 / 10 / 5 / 2 %).
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FS, NSEG = 1 / 0.914, 256


def segments(x, nseg=NSEG, keep=None):
    """[S, nseg, ...] half-overlapping Hann-windowed, detrended (mean) segments along axis 0; with `keep` ([T] bool) only
    the segments whose every frame is kept (--train-only: wholly inside ZAPBench's training frames)."""
    T = x.shape[0]
    starts = [s for s in range(0, T - nseg + 1, nseg // 2) if keep is None or bool(keep[s:s + nseg].all())]
    w = torch.hann_window(nseg, periodic=False, device=x.device, dtype=x.dtype).reshape(nseg, *([1] * (x.dim() - 1)))
    return torch.stack([(x[s:s + nseg] - x[s:s + nseg].mean(0)) * w for s in starts], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="zapbench")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--varying-only", action="store_true",
                    help="only the features that CHANGE inside their own condition (Cedric, 2026-10-02): the 9 block "
                         "markers, 1 for their whole condition, pick neurons that shift between blocks, not "
                         "stimulus-locked ones")
    ap.add_argument("--band-cols", type=int, default=0,
                    help="pick the band from the live features among the first BAND_COLS stimulus columns (0 = all)")
    ap.add_argument("--mask-out", default="", help="write the top-`--top` mask to graphs_data/zebrafish/<this>")
    ap.add_argument("--top", type=float, default=0.10, help="the fraction of neurons the mask keeps")
    ap.add_argument("--train-only", action="store_true",
                    help="only the Welch windows wholly inside ZAPBench's TRAINING frames (Cedric, 2026-10-06: the mask of "
                         "a held-out run must not see its test frames)")
    a = ap.parse_args()
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{a.recording}_recording.npz"))
    X = z["dff"]
    U = z["stimulus"].astype(np.float32)
    live = np.where(np.abs(U).max(0) > 0)[0]                         # the features that ever move (22 of 26)
    if a.varying_only:
        # ... and CHANGE within the condition(s) that use them (Cedric, 2026-10-02): a block marker is 1 over its
        # whole condition (dropped); an on/off feature (f9 turning, f13-f15 position) is 0 / 1 inside its condition
        # (kept -- the first version tested only the frames where the feature is non-zero and dropped these four);
        # an ephys column spans every condition and changes (kept)
        cond = np.asarray(z["condition"])
        def changes(j):
            frames = np.isin(cond, np.unique(cond[np.abs(U[:, j]) > 0]))
            return U[frames, j].std() > 1e-6
        live = np.array([j for j in live if changes(j)])
    print(f"[stim coherence] features used: {live.tolist()}")
    U_all_cols = U.shape[1]
    U = U[:, live]
    dev = a.device
    keep = None
    if a.train_only:
        from plexus.tasks.trace_recording import zapbench_split
        keep = zapbench_split(np.asarray(z["offsets"]), U.shape[0]) == 0
        print(f"[stim coherence] training frames only: {int(keep.sum()):,} of {len(keep):,}")
    Uf = torch.fft.rfft(segments(torch.as_tensor(U, device=dev), keep=keep), dim=1)        # [S, F, K]
    Suu = (Uf.abs() ** 2).mean(0)                                                # [F, K]
    f = np.fft.rfftfreq(NSEG, d=1 / FS)
    bc = np.where(live < a.band_cols)[0] if a.band_cols else np.arange(len(live))   # the features that set the band
    band = np.argsort(Suu[:, bc].sum(1).cpu().numpy()[1:])[::-1][:6] + 1        # the stimulus's 6 strongest
    T, N = X.shape
    coh = np.zeros(N, np.float32)
    best = np.zeros(N, np.int16)
    coh_all = np.zeros((N, len(live)), np.float32)                     # every neuron's band coherence per feature
    for c0 in range(0, N, a.chunk):
        c1 = min(N, c0 + a.chunk)
        x = torch.as_tensor(np.asarray(X[:, c0:c1], np.float32), device=dev)
        Xf = torch.fft.rfft(segments(x, keep=keep), dim=1)                       # [S, F, n]
        Sxx = (Xf.abs() ** 2).mean(0)                                            # [F, n]
        Sxu = torch.einsum("sfn,sfk->fnk", Xf, Uf.conj()) / Xf.shape[0]          # [F, n, K]
        C = (Sxu.abs() ** 2) / (Sxx[:, :, None] * Suu[:, None, :]).clamp(min=1e-20)
        cb = C[band].mean(0)                                                     # [n, K] band coherence per feature
        v, k = cb.max(1)
        coh[c0:c1], best[c0:c1] = v.cpu().numpy(), k.cpu().numpy()
        coh_all[c0:c1] = cb.cpu().numpy()
    tag = a.recording + ("_varying" if a.varying_only else "") + ("_train" if a.train_only else "")
    out = os.path.join(EXP, "data", f"stim_coherence_{tag}.npz")
    np.savez(out, coherence=coh, best_feature=live[best], band_hz=f[band], features=live)
    print(f"[stim coherence] {len(live)} features {live.tolist()}; band from {live[bc].tolist()}")
    if a.mask_out:
        thr = np.float32(np.quantile(coh, 1 - a.top))
        m = (coh >= thr).astype(np.float32)
        # THE PER-FEATURE MASK (Cedric, 2026-10-06: "restrict neurons to just their selecting feature(s)"): neuron i
        # receives feature k only if it is in the mask AND its band coherence with k clears the same cut -- its best
        # feature always, any other feature that also selects it; [N, all stimulus columns], read by the law as
        # `input_mask_array: mask_by_input` (each input enters only its own neurons)
        mbi = np.zeros((N, U_all_cols), np.float32)
        sel_k = (coh_all >= thr) & (m[:, None] > 0)
        mbi[:, live] = sel_k
        mbi[np.arange(N), live[best]] = np.maximum(mbi[np.arange(N), live[best]], m)
        np.savez(graphs_data_path("zebrafish", a.mask_out), mask=m, coherence=coh, threshold=thr,
                 best_feature=live[best].astype(np.int64), features=live, band_hz=f[band], mask_by_input=mbi,
                 coherence_by_feature=coh_all)
        nf = mbi[m > 0].sum(1)
        print(f"[stim coherence] mask {a.mask_out}: {int(m.sum()):,} neurons, coherence >= {thr:.4f}; features per input "
              f"neuron: mean {nf.mean():.2f}, max {int(nf.max())}; per feature {mbi.sum(0).astype(int).tolist()}")
    print(f"[stim coherence] {a.recording}: {N:,} neurons; band {np.round(f[band], 4)} Hz; median {np.median(coh):.3f}, "
          f"p90 {np.percentile(coh, 90):.3f}")
    pos = z["pos_um"]
    if a.recording != "zapbench":                                   # the zap-inr anatomy frame drawn head-up
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    montage(coh, pos, tag)


def montage(coh, P, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(coh, [1, 99])
    y = P[:, 1]
    top_cut = np.percentile(y, 100 * 2 / 3)
    order = np.argsort(P[:, 2])
    fr = [0.5, 0.25, 0.10, 0.05, 0.02]
    plt.style.use("dark_background")
    n = len(fr) + 1
    fig = plt.figure(figsize=(3.1 * n, 5.4), facecolor="black")
    for j in range(n):
        ax = fig.add_axes([j / n + 0.005, 0.10, 1 / n - 0.01, 0.74])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_aspect("equal")
        if j == 0:
            sc = ax.scatter(P[order, 0], P[order, 1], c=coh[order], s=0.2, cmap="magma", vmin=vmin, vmax=vmax,
                            linewidths=0)
            t = f"{name}\nbest-feature coherence, {vmin:.2f} .. {vmax:.2f} (1st-99th pct)"
        else:
            q = fr[j - 1]
            thr = np.quantile(coh, 1 - q)
            m = coh >= thr
            mo = order[m[order]]
            ax.scatter(P[order, 0], P[order, 1], c="0.22", s=0.12, linewidths=0)
            ax.scatter(P[mo, 0], P[mo, 1], c=coh[mo], s=0.35, cmap="magma", vmin=vmin, vmax=vmax, linewidths=0)
            t = f"top {100 * q:g} %: coherence >= {thr:.3f}\n{m.sum():,} neurons, {100 * (y[m] >= top_cut).mean():.0f} % in the top third"
        fig.text(j / n + 0.5 / n, 0.95, t, ha="center", va="top", fontsize=8, color="white")
    cax = fig.add_axes([0.02, 0.05, 0.12, 0.018])
    fig.colorbar(sc, cax=cax, orientation="horizontal").ax.tick_params(labelsize=7, colors="0.8")
    out = os.path.join(EXP, "data", "figs", f"stim_coherence_thresholds_{name}.png")
    fig.savefig(out, dpi=130, facecolor="black")
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    plt.close(fig)
    print("[stim coherence] montage", os.path.join(EXP, "png", os.path.basename(out)))


if __name__ == "__main__":
    main()

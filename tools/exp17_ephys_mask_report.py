"""exp17: what the swimming (ephys) features change in the input mask (Cedric, 2026-10-02).

Local analysis, after tools/export_ephys_features.py and

    python tools/exp17_stim_coherence.py --recording zapbench_destripe_ephys --varying-only --band-cols 22 \
        --mask-out input_mask_destripe_ephys_coh10.npz

Compares graphs_data/zebrafish/input_mask_destripe_ephys_coh10.npz (9 visual features that change inside their
condition + 5 ephys features) with input_mask_destripe_coh10v.npz (the 9 visual features only): neurons selected, the
overlap, how many selected neurons have an ephys feature as their best, and where they are.

WHERE, roughly: no atlas is registered to this fish, so the brain is cut along its head-to-tail axis (the montage's
head-up frame, rostral at the top) into fractions of its length, the 0.5th to 99.5th percentile of all somata:
forebrain < 0.20 <= midbrain (tectum, tegmentum) < 0.50 <= hindbrain (cerebellum included). Left / right from the
midline, the median of the left-right coordinate.

THE BAND'S SENSITIVITY. The same per-feature coherences are recomputed in a second band, the 6 strongest frequencies
of ALL 14 features (visual + ephys, as scaled in the recording), and the top 10 % of that score compared.

Writes experiments/exp17_zapbench_graphcast/data/ephys_mask_report.json and data/figs/ephys_mask_map.png (+ png/).
"""
import json
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from exp17_stim_coherence import FS, NSEG, segments          # noqa: E402

EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
REGIONS = [("forebrain", 0.0, 0.20), ("midbrain", 0.20, 0.50), ("hindbrain", 0.50, 1.01)]


def per_feature_coherence(X, U, dev, chunk=4000):
    """The stimulus side of the coherence: the features' Welch spectra Uf [S, F, K] and power Suu [F, K]."""
    Uf = torch.fft.rfft(segments(torch.as_tensor(U, device=dev)), dim=1)
    Suu = (Uf.abs() ** 2).mean(0)
    return Uf, Suu


def main():
    from plexus.paths import graphs_data_path
    dev = "cuda:0"
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_ephys_recording.npz"))
    new = np.load(graphs_data_path("zebrafish", "input_mask_destripe_ephys_coh10.npz"))
    old = np.load(graphs_data_path("zebrafish", "input_mask_destripe_coh10v.npz"))
    names = [str(s) for s in z["stimulus_names"]]
    live = new["features"]
    n_vis = int((live < 22).sum())
    X, U = z["dff"], z["stimulus"][:, live].astype(np.float32)
    Uf, Suu = per_feature_coherence(X, U, dev)
    s = Suu.cpu().numpy()
    bands = {"visual": np.argsort(s[:, :n_vis].sum(1)[1:])[::-1][:6] + 1,
             "all": np.argsort(s.sum(1)[1:])[::-1][:6] + 1}
    f = np.fft.rfftfreq(NSEG, d=1 / FS)
    T, N = X.shape
    cb = {k: np.zeros((N, len(live)), np.float32) for k in bands}
    for c0 in range(0, N, 4000):
        c1 = min(N, c0 + 4000)
        x = torch.as_tensor(np.asarray(X[:, c0:c1], np.float32), device=dev)
        Xf = torch.fft.rfft(segments(x), dim=1)
        Sxx = (Xf.abs() ** 2).mean(0)
        Sxu = torch.einsum("sfn,sfk->fnk", Xf, Uf.conj()) / Xf.shape[0]
        C = (Sxu.abs() ** 2) / (Sxx[:, :, None] * Suu[:, None, :]).clamp(min=1e-20)
        for k, b in bands.items():
            cb[k][c0:c1] = C[b].mean(0).cpu().numpy()
    coh = cb["visual"].max(1)
    if not np.allclose(coh, new["coherence"], atol=1e-4):
        raise ValueError("the visual-band coherence does not reproduce the mask's")
    m_new, m_old = new["mask"] > 0, old["mask"] > 0
    best = new["best_feature"]
    is_eph = best >= 22
    # where: the head-up frame of exp17_stim_coherence's montage (x' = y, y' = -x), rostral at the top
    P = z["pos_um"]
    ap_ = -P[:, 0]
    lo, hi = np.percentile(ap_, [0.5, 99.5])
    rc = (hi - ap_) / (hi - lo)                                   # 0 at the head end, 1 at the tail end
    lr = P[:, 1] - np.median(P[:, 1])

    def where(m):
        out = {r: int((m & (rc >= a) & (rc < b)).sum()) for r, a, b in REGIONS}
        out["left_right"] = [int((m & (lr < 0)).sum()), int((m & (lr >= 0)).sum())]
        return out
    sel_e = m_new & is_eph
    rep = {
        "mask": "graphs_data/zebrafish/input_mask_destripe_ephys_coh10.npz",
        "features": {int(j): names[j] for j in live},
        "band_hz_visual": np.round(f[bands["visual"]], 5).tolist(),
        "band_hz_all": np.round(f[bands["all"]], 5).tolist(),
        "neurons": int(N), "selected": int(m_new.sum()), "threshold": float(new["threshold"]),
        "old_threshold": float(old["threshold"]),
        "overlap_with_coh10v": int((m_new & m_old).sum()),
        "only_new": int((m_new & ~m_old).sum()), "only_old": int((m_old & ~m_new).sum()),
        "selected_best_is_ephys": int(sel_e.sum()),
        "selected_best_by_feature": {names[j]: int((m_new & (best == j)).sum()) for j in live},
        "all_best_by_feature": {names[j]: int((best == j).sum()) for j in live},
        "where_all": where(np.ones(N, bool)), "where_selected": where(m_new),
        "where_selected_best_ephys": where(sel_e), "where_selected_best_visual": where(m_new & ~is_eph),
        "where_old_mask": where(m_old), "where_new_only": where(m_new & ~m_old),
        "region_rule": "fraction of the head-to-tail length (0.5-99.5th pct of somata): forebrain < 0.20 <= "
                       "midbrain < 0.50 <= hindbrain; left/right of the median left-right coordinate",
    }
    # the band's sensitivity: the top 10 % in the all-feature band
    ca = cb["all"].max(1)
    ma = ca >= np.quantile(ca, 0.9)
    rep["all_band"] = {"threshold": float(np.quantile(ca, 0.9)), "overlap_with_new_mask": int((ma & m_new).sum()),
                       "overlap_with_coh10v": int((ma & m_old).sum()),
                       "selected_best_is_ephys": int((ma & (live[cb["all"].argmax(1)] >= 22)).sum())}
    # how much the ephys features win by: among selected neurons with an ephys best, the best visual coherence
    vis_best = cb["visual"][:, :n_vis].max(1)
    rep["selected_best_ephys_visual_coherence_median"] = float(np.median(vis_best[sel_e]))
    rep["selected_best_ephys_above_old_threshold_on_visual_alone"] = int((sel_e & (vis_best >= old["threshold"])).sum())
    json.dump(rep, open(os.path.join(EXP, "data", "ephys_mask_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    np.savez(os.path.join(EXP, "data", "ephys_mask_feature_coherence.npz"), features=live,
             coherence_visual_band=cb["visual"].astype(np.float16), coherence_all_band=cb["all"].astype(np.float16))
    figure(P, m_old, m_new, is_eph, rep, rc)


def figure(P, m_old, m_new, is_eph, rep, rc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    Q = np.stack([P[:, 1], -P[:, 0]], 1)
    order = np.argsort(P[:, 2])
    fig = plt.figure(figsize=(13, 6.2), facecolor="black")
    panels = [("the current mask (9 visual features)\n"
               f"{int(m_old.sum()):,} neurons, coherence >= {rep['old_threshold']:.3f}", [(m_old, "#4cc970")]),
              ("the new mask (9 visual + 5 ephys features)\n"
               f"{int(m_new.sum()):,} neurons, coherence >= {rep['threshold']:.3f}",
               [(m_new & ~is_eph, "#4cc970"), (m_new & is_eph, "#e8834c")]),
              (f"in both: {rep['overlap_with_coh10v']:,}\nonly new: {rep['only_new']:,} (orange), "
               f"only current: {rep['only_old']:,} (blue)",
               [(m_new & m_old, "0.85"), (m_new & ~m_old, "#e8834c"), (m_old & ~m_new, "#4c9be8")])]
    for i, (t, layers) in enumerate(panels):
        ax = fig.add_axes([i / 3 + 0.005, 0.06, 1 / 3 - 0.01, 0.76])
        ax.axis("off")
        ax.set_aspect("equal")
        ax.scatter(Q[order, 0], Q[order, 1], c="0.2", s=0.1, linewidths=0)
        for m, col in layers:
            o = order[m[order]]
            ax.scatter(Q[o, 0], Q[o, 1], c=col, s=0.4, linewidths=0)
        lo, hi = np.percentile(-P[:, 0], [0.5, 99.5])
        for cut, nm in [(0.20, "forebrain | midbrain"), (0.50, "midbrain | hindbrain")]:
            yy = hi - cut * (hi - lo)
            ax.axhline(yy, color="0.45", lw=0.5, ls="--")
            if i == 0:
                ax.text(Q[:, 0].min(), yy + 4, nm, fontsize=7, color="0.6")
        fig.text(i / 3 + 1 / 6, 0.95, t, ha="center", va="top", fontsize=9, color="white")
    fig.text(1.5 / 3, 0.02, "middle panel: green = the neuron's best feature is visual, orange = an ephys feature",
             ha="center", fontsize=8, color="0.8")
    out = os.path.join(EXP, "data", "figs", "ephys_mask_map.png")
    fig.savefig(out, dpi=130, facecolor="black")
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    plt.close(fig)
    print("[figure]", out)


if __name__ == "__main__":
    main()

"""exp17: HOW THE 10 % INPUT NEURONS ARE CHOSEN, as figures and a movie for the slides (Cedric, 2026-10-02).

The selection itself is tools/exp17_stim_coherence.py (model-free): for every neuron, the magnitude-squared coherence
between its dF/F trace and EACH of the 22 stimulus features (Welch, 256-frame = 234 s Hann segments, half overlap),
averaged over the stimulus's 6 strongest frequencies (the band); per neuron the best feature's band coherence; the
top 10 % of neurons by it are the input neurons (graphs_data/zebrafish/input_mask_<zapbench|destripe>_coh10.npz, the
stimulus mask of batches 11, 14 and 15). This script only EXPLAINS that choice; it recomputes the spectra it draws
from the recording and checks them against the saved band coherence.

Per recording (zapbench, zapbench_destripe) it writes, into experiments/exp17_zapbench_graphcast/presentation/:
  figs/input_neurons_<rec>_method.png   the pipeline: one neuron + its feature, the stimulus spectrum, coherence
                                        spectra (selected vs unselected neuron), the band-coherence histogram + cut
  figs/input_neurons_<rec>_method.txt   5-7 plain-English lines for the slide's right column
  figs/input_neurons_<rec>_map.png      the brain (selected coloured by best feature) + per-feature |Pearson r| and
                                        band coherence, selected vs 100 random unselected neurons
  Movies/input_neurons_<rec>_kymo.mp4   sliding kymograph over the whole recording: 22 stimulus features (top),
                                        100 most coherent selected neurons' z-scored dF/F (bottom)  (+ .png still)
and copies every PNG into experiments/exp17_zapbench_graphcast/png/.

  python tools/exp17_input_neurons.py [--recording zapbench zapbench_destripe] [--no-movie]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FIGS = os.path.join(EXP, "presentation", "figs")
MOVIES = os.path.join(EXP, "presentation", "Movies")
PNG = os.path.join(EXP, "png")
FRAME_S = 0.914                      # seconds per recording frame
NSEG = 256                           # Welch segment length, frames (234 s), as exp17_stim_coherence.py
MASK = {"zapbench": "input_mask_zapbench_coh10.npz", "zapbench_destripe": "input_mask_destripe_coh10.npz"}
MASK_MIX = {"zapbench_destripe_ephys": "input_mask_destripe_mix_coh20.npz"}
# batch 19's mask (2026-10-06): the 20 % most coherent with one of the 13 changing visual features, no ephys
MASK_VIS20 = {"zapbench_destripe": "input_mask_destripe_vis_coh20.npz"}
MASK_VARYING = {"zapbench_destripe": "input_mask_destripe_coh10v.npz",
                "zapbench_destripe_ephys": "input_mask_destripe_ephys_coh10.npz"}
LABEL = {"zapbench": "ZAPBench release traces", "zapbench_destripe": "destriped zap-inr traces",
         "zapbench_destripe_ephys": "destriped zap-inr traces, stimulus + ephys"}
N_VISUAL = 22             # the visual features come first; a recording may append ephys columns after them
ORANGE, GREY = "#ff9b2b", "0.55"


def brain_view(pos):
    """The brain HORIZONTAL, HEAD LEFT (as tools/exp17_slides.brain_view): the zap-inr anatomy frame (long axis along
    x) is first turned head-up as (y, -x); then (x, y) -> (-y, x) lays the head-up brain on its side."""
    pos = np.asarray(pos, dtype=np.float64)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    return np.stack([-pos[:, 1], pos[:, 0], pos[:, 2]], 1)


def welch_np(x):
    """[S, NSEG, ...] spectra (rfft) of the half-overlapping, mean-removed, Hann-windowed segments along axis 0."""
    T = x.shape[0]
    w = np.hanning(NSEG).reshape(NSEG, *([1] * (x.ndim - 1)))
    seg = np.stack([(x[s:s + NSEG] - x[s:s + NSEG].mean(0)) * w for s in range(0, T - NSEG + 1, NSEG // 2)], 0)
    return np.fft.rfft(seg, axis=1)


def coherence_spectra(x, Uf, Suu):
    """Coherence [F, n, K] of the traces x [T, n] with each feature, frequency by frequency (Welch, as the selection)."""
    Xf = welch_np(np.asarray(x, np.float64))
    Sxx = (np.abs(Xf) ** 2).mean(0)
    Sxu = np.einsum("sfn,sfk->fnk", Xf, Uf.conj()) / Xf.shape[0]
    return np.abs(Sxu) ** 2 / np.maximum(Sxx[:, :, None] * Suu[:, None, :], 1e-30)


def band_coherence_all(X, Uf, Suu, band, device):
    """Band coherence [n, K] of the traces X [T, n] with every feature (GPU when there is one)."""
    import torch
    from exp17_stim_coherence import segments
    dev = device if torch.cuda.is_available() else "cpu"
    Ufd = torch.as_tensor(Uf.astype(np.complex64), device=dev)
    Suud = torch.as_tensor(Suu.astype(np.float32), device=dev)
    out = []
    for c0 in range(0, X.shape[1], 4000):
        x = torch.as_tensor(np.asarray(X[:, c0:c0 + 4000], np.float32), device=dev)
        Xf = torch.fft.rfft(segments(x), dim=1)
        Sxx = (Xf.abs() ** 2).mean(0)
        Sxu = torch.einsum("sfn,sfk->fnk", Xf, Ufd.conj()) / Xf.shape[0]
        C = (Sxu.abs() ** 2) / (Sxx[:, :, None] * Suud[:, None, :]).clamp(min=1e-20)
        out.append(C[band].mean(0).cpu().numpy())
    return np.concatenate(out, 0)


def zscore(a):
    a = np.asarray(a, np.float64)
    return (a - a.mean(0)) / np.maximum(a.std(0), 1e-12)


def peak_lag(x, u, L=100):
    """Lag (frames, positive = the neuron AFTER the stimulus) of the largest |cross-correlation| of z-scored x and u,
    within +-L frames (+-91 s)."""
    n = len(x)
    F = np.fft.rfft(zscore(x), 2 * n) * np.conj(np.fft.rfft(zscore(u), 2 * n))
    cc = np.fft.irfft(F, 2 * n) / n
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
    vals = np.r_[cc[:L + 1], cc[-L:]]
    return int(lags[np.argmax(np.abs(vals))])


def style_ax(ax):
    ax.set_facecolor("black")
    for k, sp in ax.spines.items():
        sp.set_visible(k in ("left", "bottom"))
        sp.set_color("0.5")
    ax.tick_params(colors="0.75", labelsize=8, length=2)
    ax.xaxis.label.set_color("0.8")
    ax.yaxis.label.set_color("0.8")


def above(fig, ax, s, dy=0.012, fs=10):
    """The panel's label, written ABOVE its axes (never on it)."""
    b = ax.get_position()
    fig.text(b.x0, b.y1 + dy, s, color="white", fontsize=fs, ha="left", va="bottom")


def save(fig, path):
    fig.savefig(path, dpi=130, facecolor="black")
    shutil.copy(path, os.path.join(PNG, os.path.basename(path)))
    print("[input neurons] wrote", path)


# ------------------------------------------------------------------------------------------------------------------
def analyse(rec, device, variant="all"):
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{rec}_recording.npz"))
    X = z["dff"]
    U = z["stimulus"].astype(np.float64)
    names = [str(s) for s in z["names"]]
    off = z["offsets"].astype(int)
    T, N = X.shape
    K = U.shape[1]
    live = np.where(np.abs(U).max(0) > 0)[0]
    assert len(live) == K, "every feature moves somewhere (the saved best_feature indexes the 22 directly)"
    feat_cond = np.array([next(k for k in range(len(names)) if np.abs(U[off[k]:off[k + 1], j]).max() > 0)
                          if j < N_VISUAL else -1 for j in range(K)])   # the ONE condition that uses each feature
    sn = [str(x) for x in z["stimulus_names"]] if "stimulus_names" in z.files else []
    flabels = [f"f{j} {names[feat_cond[j]]}" if j < N_VISUAL else
               (sn[j].replace("ephys ", "ephys: ") if j < len(sn) else f"ephys {j - N_VISUAL}") for j in range(K)]
    # `variant` "varying" (Cedric, 2026-10-02): the mask made from the 13 features that CHANGE inside their own
    # condition only (tools/exp17_stim_coherence.py --varying-only), the 9 condition markers left out
    cond_ = np.asarray(z["condition"])                              # the rule of exp17_stim_coherence --varying-only:
    feats = np.arange(K) if variant == "all" else np.array(         # changes within the condition(s) that use it
        [j for j in range(K) if U[np.isin(cond_, np.unique(cond_[np.abs(U[:, j]) > 0])), j].std() > 1e-6])
    if variant == "mix":
        # THE HALF-AND-HALF MASK of batch 15 (tools/exp17_mask_mix.py): a neuron picked by its visual coherence keeps
        # its visual best feature and coherence, one picked by its ephys coherence its ephys ones
        mk = np.load(graphs_data_path("zebrafish", MASK_MIX[rec]))
        pick = mk["picked_by"]
        sel = mk["mask"] > 0
        coh = np.where(pick == 1, mk["coh_ephys"], mk["coh_visual"]).astype(np.float32)
        best = np.where(pick == 1, mk["best_ephys"], mk["best_visual"]).astype(int)
        band_hz = mk["band_hz"]
        thr = float(coh[sel].min())
    else:
        sc = np.load(os.path.join(EXP, "data", f"stim_coherence_{rec}{'' if variant == 'all' else '_varying'}.npz"))
        coh, best, band_hz = sc["coherence"], sc["best_feature"].astype(int), sc["band_hz"]
        mk = np.load(graphs_data_path("zebrafish", MASK[rec] if variant == "all" else
                                      MASK_VIS20[rec] if variant == "vis20" else MASK_VARYING[rec]))
        sel = mk["mask"] > 0
        thr = float(mk["threshold"])
        assert np.allclose(mk["coherence"], coh), "mask and coherence file disagree"
        pick = None

    Uf = welch_np(U)                                                 # [S, F, K]
    Suu = (np.abs(Uf) ** 2).mean(0)                                  # [F, K] the stimulus power spectrum per feature
    f = np.fft.rfftfreq(NSEG, d=FRAME_S)
    fb = feats[feats < N_VISUAL] if K > N_VISUAL else feats            # the band: the visual stimulus's
    band = np.argsort(Suu[:, fb].sum(1)[1:])[::-1][:6] + 1
    assert np.allclose(f[band], band_hz)

    # the example neurons: the most coherent selected neuron whose best feature is the commonest best feature among
    # the selected (a time-varying one), and an unselected neuron at the median band coherence
    common = int(np.bincount(best[sel], minlength=K).argmax())
    cand = np.where(sel & (best == common))[0]
    ex_sel = int(cand[np.argmax(coh[cand])])
    ex_un = int(np.argsort(coh)[N // 2])
    Cex = coherence_spectra(X[:, [ex_sel, ex_un]], Uf, Suu)          # [F, 2, K]
    for i, n_ in enumerate([ex_sel, ex_un]):
        assert abs(Cex[band, i, best[n_]].mean() - coh[n_]) < 2e-3, (n_, Cex[band, i, best[n_]].mean(), coh[n_])

    rng = np.random.default_rng(0)
    rnd = np.sort(rng.choice(np.where(~sel)[0], 100, replace=False))
    idx_sel = np.where(sel)[0]
    Xs = np.asarray(X[:, idx_sel], np.float32)
    Xr = np.asarray(X[:, rnd], np.float32)
    Uz = zscore(U)
    r_sel = zscore(Xs).T @ Uz / T                                    # [n_sel, K] Pearson r, zero lag
    r_rnd = zscore(Xr).T @ Uz / T
    cb_sel = band_coherence_all(Xs, Uf, Suu, band, device)          # [n_sel, K] band coherence per feature
    cb_rnd = band_coherence_all(Xr, Uf, Suu, band, device)
    if variant != "mix":
        assert np.allclose(cb_sel[:, feats].max(1), coh[idx_sel], atol=2e-3)

    # the 100 shown neurons: the most coherent selected, ordered by best feature then by the lag of their peak
    # cross-correlation with it
    if variant == "mix":                       # 50 picked by the visual stimulus, 50 by the ephys
        top = np.r_[[i for i in idx_sel[np.argsort(-coh[idx_sel])] if pick[i] == 0][:50],
                    [i for i in idx_sel[np.argsort(-coh[idx_sel])] if pick[i] == 1][:50]].astype(int)
    else:
        top = idx_sel[np.argsort(coh[idx_sel])[::-1][:100]]
    Xt = np.asarray(X[:, top], np.float64)
    lag = np.array([peak_lag(Xt[:, i], U[:, best[n_]]) for i, n_ in enumerate(top)])
    # the half-and-half mask: the visually picked block first, the motor (ephys) picked below (Cedric, 2026-10-06)
    o = np.lexsort((lag, best[top], pick[top])) if variant == "mix" else np.lexsort((lag, best[top]))
    top, Xt, lag = top[o], Xt[:, o], lag[o]
    return dict(rec=rec, variant=variant, flabels=flabels, pick=pick, X=X, U=U, names=names, off=off, T=T, N=N, K=K, feat_cond=feat_cond, coh=coh, best=best,
                sel=sel, thr=thr, Suu=Suu, f=f, band=band, ex_sel=ex_sel, ex_un=ex_un, Cex=Cex, common=common,
                rnd=rnd, idx_sel=idx_sel, r_sel=r_sel, r_rnd=r_rnd, cb_sel=cb_sel, cb_rnd=cb_rnd, top=top,
                Zt=zscore(Xt).T, lag=lag, pos=brain_view(z["pos_um"]))


def flab(d, j):
    return d["flabels"][j]


# ------------------------------------------------------------------------------------------------------------------
def fig_method(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec, X, U, f, band, Cex, coh = d["rec"], d["X"], d["U"], d["f"], d["band"], d["Cex"], d["coh"]
    es, eu = d["ex_sel"], d["ex_un"]
    bs, bu = d["best"][es], d["best"][eu]
    fig = plt.figure(figsize=(15, 8.4), facecolor="black")
    fig.text(0.015, 0.975, f"How the input neurons are chosen ({LABEL[rec]}, {d['N']:,} neurons):\nthe coherence of each neuron with each stimulus "
             f"feature in the stimulus's 6 strongest frequencies; the top 10 % of neurons are the input neurons", color="white",
             fontsize=13, va="top")
    mHz = 1e3 * f
    # a: the example neuron and its best feature, 400 frames inside that feature's condition
    c = d["feat_cond"][bs]
    a0, a1 = d["off"][c], d["off"][c + 1]
    t0 = (a0 + a1) // 2 - 200
    tt = np.arange(t0, t0 + 400)
    tm = tt * FRAME_S / 60
    axu = fig.add_axes([0.05, 0.75, 0.40, 0.09])
    axx = fig.add_axes([0.05, 0.56, 0.40, 0.18], sharex=axu)
    for ax in (axu, axx):
        style_ax(ax)
    axu.step(tm, U[tt, bs], where="post", color=ORANGE, lw=1.2)
    axu.set_ylabel("feature\nvalue", fontsize=8)
    plt.setp(axu.get_xticklabels(), visible=False)
    axx.plot(tm, X[tt, es], color="white", lw=0.9)
    axx.set_ylabel("dF/F", fontsize=9)
    axx.set_xlabel(f"time since the recording's start, min ({d['names'][c]} condition, 400 frames = 6.1 min)",
                   fontsize=9)
    above(fig, axu, f"a   selected neuron #{es:,} (dF/F, white) and its best stimulus feature, {flab(d, bs)} "
                    f"(orange)")
    # b: the stimulus spectrum and the band
    axb = fig.add_axes([0.55, 0.56, 0.42, 0.28])
    style_ax(axb)
    S = d["Suu"].sum(1)
    axb.semilogy(mHz[1:], S[1:], color="0.85", lw=1.0)
    axb.semilogy(mHz[band], S[band], "o", color=ORANGE, ms=6)
    for b_ in band:
        axb.text(mHz[b_], S[b_] * 1.6, f"{1 / f[b_]:.0f} s", color=ORANGE, fontsize=8, va="bottom", ha="center")
    axb.set_xscale("log")
    axb.set_xlabel("frequency, mHz (the 256-frame Welch segments: 4.3 mHz steps up to 547 mHz)", fontsize=9)
    axb.set_ylabel("stimulus power,\n22 features summed", fontsize=9)
    above(fig, axb, "b   the stimulus's power spectrum; its 6 strongest frequencies (orange,\n     labelled by "
                    "period) are the band")
    # c: coherence spectra, selected vs unselected
    axc = fig.add_axes([0.05, 0.08, 0.40, 0.33])
    style_ax(axc)
    dfr = mHz[1] - mHz[0]
    for b_ in band:
        axc.axvspan(mHz[b_] - dfr / 2, mHz[b_] + dfr / 2, color=ORANGE, alpha=0.12, lw=0)
    axc.plot(mHz[1:], Cex[1:, 0, bs], color="white", lw=1.0,
             label=f"selected #{es:,} with {flab(d, bs)}: band mean {coh[es]:.2f}")
    axc.plot(mHz[1:], Cex[1:, 1, bu], color="#4fa3ff", lw=1.0,
             label=f"unselected #{eu:,} (median neuron) with {flab(d, bu)}: band mean {coh[eu]:.2f}")
    axc.axhline(coh[es], color="white", ls="--", lw=0.7)
    axc.axhline(coh[eu], color="#4fa3ff", ls="--", lw=0.7)
    axc.set_xscale("log")
    axc.set_ylim(0, 1)
    axc.set_xlabel("frequency, mHz", fontsize=9)
    axc.set_ylabel("coherence with the feature\n(0 = unrelated, 1 = linear at any lag)", fontsize=9)
    leg = axc.legend(loc="upper right", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color("0.85")
    above(fig, axc, "c   coherence of a neuron with its best feature, frequency by frequency;\n     band "
                    "shaded, band mean (the neuron's score) dashed")
    # d: the histogram and the 10 % cut
    axd = fig.add_axes([0.55, 0.08, 0.42, 0.33])
    style_ax(axd)
    bins = np.linspace(0, 1, 101)
    h, _ = np.histogram(coh, bins)
    cols = np.where(bins[:-1] >= d["thr"] - 1e-9, ORANGE, "0.6")
    axd.bar(bins[:-1], h, width=bins[1] - bins[0], align="edge", color=cols, lw=0)
    axd.axvline(d["thr"], color="white", lw=1.0)
    axd.set_yscale("log")
    axd.text(d["thr"] + 0.01, h.max() * 0.6, f"90th percentile = {d['thr']:.3f}\n{int(d['sel'].sum()):,} of "
             f"{d['N']:,} neurons above it\n= the input neurons", color="white", fontsize=9, va="top")
    axd.set_xlabel("band coherence of each neuron with its best feature", fontsize=9)
    axd.set_ylabel("neurons", fontsize=9)
    above(fig, axd, "d   every neuron's score (band coherence with its best feature);\n     the top 10 % "
                    "(orange) are selected")
    out = os.path.join(FIGS, f"input_neurons_{rec}_method.png")
    save(fig, out)
    plt.close(fig)
    per = np.sort(1 / f[band])
    lines = [
        f"Goal: the {100 * d['sel'].mean():.0f} % of the {d['N']:,} neurons whose activity follows the visual stimulus "
        f"most closely; no model, the stimulus and the recording only.",
        f"For each neuron and each of the 22 stimulus features: coherence, frequency by frequency (0 = unrelated, "
        f"1 = a perfect linear relation at any lag; Welch, 256-frame = 234 s windows, half overlap).",
        f"The band: the stimulus's 6 strongest frequencies, periods " + ", ".join(f"{p:.0f}" for p in per) + " s; "
        f"the coherence is averaged over them.",
        "Each neuron keeps its best feature's band coherence (and which feature that is).",
        f"Cut at the 90th percentile, band coherence {d['thr']:.3f}: {int(d['sel'].sum()):,} input neurons "
        f"(the median neuron: {np.median(d['coh']):.2f}).",
        "Batches 11, 14, 15: the stimulus term B_i.u enters only these neurons; the others get it through the "
        "network.",
    ]
    with open(out.replace(".png", ".txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("[input neurons] wrote", out.replace(".png", ".txt"))


# ------------------------------------------------------------------------------------------------------------------
def summary(d):
    """Per feature: median |r| and band coherence over the selected neurons whose best feature it is, and over all 100
    random unselected neurons; plus the own-best-feature medians."""
    K, best, idx_sel, rnd = d["K"], d["best"], d["idx_sel"], d["rnd"]
    bsel, brnd = best[idx_sel], best[rnd]
    s = dict(n=np.bincount(bsel, minlength=K), r_sel=np.full(K, np.nan), c_sel=np.full(K, np.nan),
             r_rnd=np.median(np.abs(d["r_rnd"]), 0), c_rnd=np.median(d["cb_rnd"], 0))
    for j in range(K):
        m = bsel == j
        if m.any():
            s["r_sel"][j] = np.median(np.abs(d["r_sel"][m, j]))
            s["c_sel"][j] = np.median(d["cb_sel"][m, j])
    ar = np.arange
    s["own_r_sel"] = float(np.median(np.abs(d["r_sel"][ar(len(idx_sel)), bsel])))
    s["own_r_rnd"] = float(np.median(np.abs(d["r_rnd"][ar(len(rnd)), brnd])))
    s["own_c_sel"] = float(np.median(d["cb_sel"][ar(len(idx_sel)), bsel]))
    s["own_c_rnd"] = float(np.median(d["cb_rnd"][ar(len(rnd)), brnd]))
    return s


def fig_map(d, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec, P, sel, best, K = d["rec"], d["pos"], d["sel"], d["best"], d["K"]
    n = s["n"]
    major = [j for j in np.argsort(n)[::-1] if n[j] >= 0.01 * sel.sum()]      # >= 1 % of the selected: own colour
    pal = ["#ff9b2b", "#4fa3ff", "#e8e84a", "#ff5ad2", "#5ae0a0", "#c49cff", "#ff5a5a", "#9ad0ff", "#b8860b"]
    col = {j: pal[i % len(pal)] for i, j in enumerate(major)}
    fig = plt.figure(figsize=(16, 8.4), facecolor="black")
    fig.text(0.015, 0.975, f"The input neurons ({LABEL[rec]}): {int(sel.sum()):,} of {d['N']:,} (10 %), where they are "
             f"and how strongly they follow each stimulus feature", color="white", fontsize=13, va="top")
    ax = fig.add_axes([0.01, 0.14, 0.47, 0.70])
    ax.set_facecolor("black")
    ax.axis("off")
    o = np.argsort(P[:, 2])
    un = o[~sel[o]]
    ax.scatter(P[un, 0], P[un, 1], c="0.28", s=0.12, linewidths=0, rasterized=True)
    so = o[sel[o]]
    cs = [col.get(int(best[i]), "0.85") for i in so]
    ax.scatter(P[so, 0], P[so, 1], c=cs, s=2.0, linewidths=0, rasterized=True)
    lo, hi = np.percentile(P[:, :2], [0.2, 99.8], 0)
    pad = 0.03 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.plot([hi[0] - 100, hi[0]], [lo[1] - 0.5 * pad] * 2, color="white", lw=1.5)
    ax.text(hi[0] - 50, lo[1] - 0.2 * pad, "100 µm", color="0.7", fontsize=8, ha="center", va="bottom")
    ax.text(lo[0], lo[1] - 0.6 * pad, "head left; seen from above", color="0.6", fontsize=8, va="top")
    hs = [plt.Line2D([], [], ls="", marker="o", ms=6, color=col[j], label=f"{flab(d, j)}: {n[j]:,}") for j in major]
    rest = int(sel.sum() - sum(n[j] for j in major))
    hs.append(plt.Line2D([], [], ls="", marker="o", ms=6, color="0.85",
                         label=f"other {sum(1 for j in range(K) if n[j] and j not in major)} features: {rest:,}"))
    hs.append(plt.Line2D([], [], ls="", marker="o", ms=4, color="0.28", label=f"unselected: {int((~sel).sum()):,}"))
    leg = ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.0, -0.0), ncol=3, fontsize=8, frameon=False,
                    title="best feature (feature index, the condition that uses it): selected neurons", title_fontsize=8)
    leg.get_title().set_color("0.8")
    for t in leg.get_texts():
        t.set_color("0.85")
    above(fig, ax, "a   selected neurons coloured by their best feature; unselected grey")
    # b, c: per-feature summaries
    x = np.arange(K)
    cond = d["feat_cond"]
    for i, (key, ylab, txt) in enumerate([
            ("r", "|Pearson r| of dF/F with the feature,\nzero lag, whole recording",
             f"b   |r| with each feature. Coloured (as in a): median over the selected neurons whose best feature it "
             f"is (n above);\n     grey: median over 100 random unselected neurons. With its own best feature: "
             f"{s['own_r_sel']:.2f} selected vs {s['own_r_rnd']:.2f} random"),
            ("c", "band coherence with the feature",
             f"c   band coherence with each feature, same groups; dashed: the 10 % cut, {d['thr']:.3f}.\n     With "
             f"its own best feature: {s['own_c_sel']:.2f} selected vs {s['own_c_rnd']:.2f} random")]):
        a = fig.add_axes([0.53, 0.55 - 0.43 * i, 0.455, 0.26])
        style_ax(a)
        for k in range(len(d["names"])):                              # the conditions as alternating blocks
            js = np.where(cond == k)[0]
            a.axvspan(js[0] - 0.5, js[-1] + 0.5, color=("0.22" if k % 2 else "0.12"), lw=0, zorder=0)
            a.text((js[0] + js[-1]) / 2, -0.17, d["names"][k], transform=a.get_xaxis_transform(), ha="center",
                   va="top", fontsize=7.5, color="0.75")
        vs, vr = s[f"{key}_sel"], s[f"{key}_rnd"]
        a.bar(x - 0.2, np.nan_to_num(vs), 0.4, color=[col.get(j, "0.85") if n[j] else "none" for j in x], zorder=2)
        a.bar(x + 0.2, vr, 0.4, color="0.5", zorder=2)
        top_ = np.nanmax(np.r_[vs, vr])
        for j in x:
            a.text(j - 0.2, (0 if np.isnan(vs[j]) else vs[j]) + 0.02 * top_, f"{n[j]}", ha="center", va="bottom",
                   fontsize=6, color="0.75", rotation=90)
        if key == "c":
            a.axhline(d["thr"], color="white", ls="--", lw=0.7)
        a.set_ylim(0, top_ * 1.25)
        a.set_xlim(-0.6, K - 0.4)
        a.set_xticks(x)
        a.set_xticklabels([f"f{j}" for j in x], fontsize=7)
        a.set_ylabel(ylab, fontsize=8.5)
        above(fig, a, txt, fs=8.5)
    out = os.path.join(FIGS, f"input_neurons_{rec}_map.png")
    save(fig, out)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------------
_KY: dict = {}         # the movie's arrays, set before the frame workers fork (they read it, never copy it in)
STIM_CMAP = ["#2b8cff", "black", ORANGE]


def _kymo_frames(ks):
    """One worker: the figure built ONCE (both kymographs over the whole recording, the condition blocks), then per
    frame only the x-limits of the sliding axes, the condition names, the window box and the time text change."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    d = _KY
    T, W, starts, off, names = d["T"], d["W"], d["starts"], d["off"], d["names"]
    K, cond = d["K"], d["feat_cond"]
    tmin = T * FRAME_S / 60
    m = lambda fr: fr * FRAME_S / 60                                   # frame -> minutes since the recording's start
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="black")
    fig.text(0.015, 0.975, f"The input neurons follow the stimulus ({LABEL[d['rec']]}): a {W}-frame "
             f"({W * FRAME_S / 60:.1f} min) window sliding over the whole recording", color="white", fontsize=12,
             va="top")
    t_txt = fig.text(0.015, 0.935, "", color="0.8", fontsize=10.5, va="top", ha="left")
    L, R = 0.13, 0.90
    axn = fig.add_axes([L, 0.855, R - L, 0.030])                      # the condition names in the window
    axs = fig.add_axes([L, 0.61, R - L, 0.24])                        # 22 stimulus features
    axk = fig.add_axes([L, 0.17, R - L, 0.39])                        # 100 neurons
    axo = fig.add_axes([L, 0.045, R - L, 0.045])                      # the whole recording, window box
    ext = (0, tmin)
    stim_cm = LinearSegmentedColormap.from_list("stim", STIM_CMAP)
    ims = axs.imshow(d["U"].T, aspect="auto", interpolation="nearest", cmap=stim_cm, vmin=-1, vmax=1,
                     extent=(ext[0], ext[1], K - 0.5, -0.5))
    imk = axk.imshow(d["Z"], aspect="auto", interpolation="nearest", cmap="inferno", vmin=-1, vmax=3,
                     extent=(ext[0], ext[1], d["Z"].shape[0] - 0.5, -0.5))
    for a in (axs, axk):
        a.set_facecolor("black")
        a.tick_params(colors="0.75", labelsize=7, length=2)
        for sp in a.spines.values():
            sp.set_color("0.4")
        for b in off[1:-1]:
            a.axvline(m(b), color="white", lw=1.0, ls="--")
    axs.set_yticks(np.arange(K))
    axs.set_yticklabels([f"f{j}  {names[cond[j]]}" for j in range(K)], fontsize=6.3)
    for j in range(1, K):
        if cond[j] != cond[j - 1]:
            axs.axhline(j - 0.5, color="0.45", lw=0.6)
    plt.setp(axs.get_xticklabels(), visible=False)
    bt = d["best_top"]
    axk.set_yticks([])
    cuts = np.r_[0, np.flatnonzero(np.diff(bt)) + 1, len(bt)]
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        if a_:
            axk.axhline(a_ - 0.5, color="0.6", lw=0.7)
        axk.text(-0.008, (a_ + b_ - 1) / 2, f"best f{bt[a_]} {names[cond[bt[a_]]]} ({b_ - a_})",
                 transform=axk.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color="0.85")
    axk.set_xlabel("time since the recording's start, min", fontsize=9, color="0.8")
    fig.text(L, 0.89, "stimulus features (rows, grouped by the condition that uses each): value -1 (blue) .. 0 "
             "(black) .. +1 (orange)", color="white", fontsize=9.5, va="bottom")
    fig.text(L, 0.565, "the 100 most coherent input neurons: dF/F, each row z-scored over the whole recording; grouped "
             "by best feature (n), then by lag of peak cross-correlation",
             color="white", fontsize=9.5, va="bottom")
    cb1 = fig.add_axes([0.915, 0.63, 0.010, 0.21])
    fig.colorbar(ims, cax=cb1).ax.tick_params(colors="0.75", labelsize=7)
    cb1.set_title("value", fontsize=7, color="0.75")
    cb2 = fig.add_axes([0.915, 0.19, 0.010, 0.35])
    fig.colorbar(imk, cax=cb2).ax.tick_params(colors="0.75", labelsize=7)
    cb2.set_title("z", fontsize=7, color="0.75")
    axn.axis("off")
    axn.set_ylim(0, 1)
    axo.set_facecolor("black")
    axo.set_yticks([])
    axo.tick_params(colors="0.75", labelsize=7, length=2)
    for k_, sp in axo.spines.items():
        sp.set_visible(k_ == "bottom")
        sp.set_color("0.4")
    for k in range(len(names)):
        a_, b_ = m(off[k]), m(off[k + 1])
        for a in (axn, axo):
            a.axvspan(a_, b_, color=("0.30" if k % 2 else "0.16"), lw=0)
        axo.text((a_ + b_) / 2, 0.5, names[k], ha="center", va="center", fontsize=6.5, color="0.85")
    axo.set_xlim(0, tmin)
    fig.text(L, 0.095, "the whole recording (2 h, nine conditions); the box is the window shown above", color="0.75",
             fontsize=8, va="bottom")
    box = plt.Rectangle((0, 0), m(W), 1, fill=False, ec=ORANGE, lw=1.5, transform=axo.get_xaxis_transform())
    axo.add_patch(box)
    ctext = [axn.text(0, 0.5, names[k], ha="center", va="center", fontsize=9, color="white") for k in
             range(len(names))]
    for k in ks:
        s0 = int(starts[k])
        x0, x1 = m(s0), m(s0 + W)
        for a in (axn, axs, axk):
            a.set_xlim(x0, x1)
        for c_, tx in enumerate(ctext):
            lo_, hi_ = max(s0, off[c_]), min(s0 + W, off[c_ + 1])
            tx.set_visible(hi_ - lo_ > 25)
            tx.set_x(m((lo_ + hi_) / 2))
        box.set_x(x0)
        mid = s0 + W // 2
        c_mid = int(np.searchsorted(off, mid, side="right") - 1)
        t_txt.set_text(f"t = {x0:5.1f} - {x1:5.1f} min of {tmin:.0f} min   (frames {s0:,} - {s0 + W - 1:,}; "
                       f"centre: {names[c_mid]})")
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=d["dpi"], facecolor="black")
    plt.close(fig)


def fig_kymo_full(d, bar_movie=False, n_frames=300, fps=25):
    """THE WHOLE RECORDING AS ONE KYMOGRAPH (Cedric, 2026-10-02: a still, not a movie, with the block partition):
    top the 22 stimulus features (rows grouped by the condition that uses each), bottom the 100 most coherent input
    neurons (each row z-scored over the recording, grouped by best feature then lag), the nine conditions as named,
    alternating bands over both and dashed lines at their borders. x: minutes since the recording's start."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    T, K, off, names, cond = d["T"], d["K"], d["off"], d["names"], d["feat_cond"]
    tmin = T * FRAME_S / 60
    m = lambda fr: fr * FRAME_S / 60
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="black")
    tag = ", 20 % mask: 10 % visual + 10 % ephys" if d.get("variant") == "mix" else (
        ", the balanced 20 % mask: 13 changing visual features, >= 50 per block, left = right" if d.get("variant") == "vis20" else "") if d.get("variant") in ("mix", "vis20") or d.get("variant", "all") == "all" else (
        ", mask: 13 visual + 5 ephys features" if d["K"] > N_VISUAL else ", mask: 13 changing visual features")
    if not bar_movie:                 # the movie has no title (Cedric, 2026-10-06)
        fig.text(0.015, 0.975, f"The input neurons and the stimulus over the whole recording ({LABEL[d['rec']]}{tag})",
                 color="white", fontsize=11, va="top")
    L, R = 0.13, 0.69                         # the kymographs; the brain on the right (Cedric, 2026-10-02)
    if bar_movie:                             # Cedric, 2026-10-07: no fish in the movie (slides 9-10 show them): wide
        R = 0.93
    axn = fig.add_axes([L, 0.885, R - L, 0.035])
    axs = fig.add_axes([L, 0.60, R - L, 0.25])
    axk = fig.add_axes([L, 0.08, R - L, 0.44] if not bar_movie else [L, 0.185, R - L, 0.335])
    if bar_movie:                     # Cedric, 2026-10-07: the brain mean in green under the kymograph, the bar on it too
        axbm = fig.add_axes([L, 0.065, R - L, 0.095])
        tb_ = np.arange(d["T"]) * FRAME_S / 60
        axbm.plot(tb_, np.asarray(d["X"]).mean(1), color="#2ca02c", lw=0.6)
        axbm.set_xlim(0, tmin)
        axbm.set_facecolor("black")
        axbm.tick_params(colors="0.75", labelsize=7, length=2)
        for sp in axbm.spines.values():
            sp.set_color("0.4")
        for b in off[1:-1]:
            axbm.axvline(m(b), color="white", lw=0.9, ls="--")
        axbm.set_ylabel("brain\nmean", fontsize=8, color="#2ca02c")
        axbm.set_xlabel("time since the recording's start, min", fontsize=9, color="0.8")
        fig._bm_ax = axbm
    ims = axs.imshow(d["U"].T, aspect="auto", interpolation="nearest",
                     cmap=LinearSegmentedColormap.from_list("stim", STIM_CMAP), vmin=-1, vmax=1,
                     extent=(0, tmin, K - 0.5, -0.5))
    Zk = d["Zt"]
    if bar_movie:
        # Cedric, 2026-10-06: per block, the 20 input neurons that respond most in it -- the highest mean of their dF/F,
        # z-scored over the whole recording, over the block's frames; 9 blocks x 20 rows
        from plexus.paths import graphs_data_path as gdp_
        si_ = np.where(np.load(gdp_("zebrafish", "input_mask_destripe_bal20.npz"))["mask"] > 0)[0]   # the balanced mask
        Xs = np.asarray(d["X"][:, si_], np.float32)
        Xs = (Xs - Xs.mean(0)) / np.maximum(Xs.std(0), 1e-6)
        rows_, blk_ = [], []
        for k in range(len(names)):
            mz = Xs[off[k]:off[k + 1]].mean(0)
            top_ = np.argsort(mz)[::-1][:20]
            rows_.append(Xs[:, top_].T)
            blk_ += [k] * len(top_)
        Zk, blk_ = np.concatenate(rows_), np.asarray(blk_)
        del Xs
    imk = axk.imshow(Zk, aspect="auto", interpolation="nearest", cmap="inferno", vmin=-1, vmax=3,
                     extent=(0, tmin, Zk.shape[0] - 0.5, -0.5))
    axn.axis("off")
    axn.set_xlim(0, tmin)
    axn.set_ylim(0, 1)
    for k in range(len(names)):
        a_, b_ = m(off[k]), m(off[k + 1])
        axn.axvspan(a_, b_, color=("0.30" if k % 2 else "0.16"), lw=0)
        axn.text((a_ + b_) / 2, 0.5, names[k], ha="center", va="center", fontsize=8.5, color="white")
    for a in (axs, axk):
        a.set_facecolor("black")
        a.tick_params(colors="0.75", labelsize=7, length=2)
        for sp in a.spines.values():
            sp.set_color("0.4")
        for b in off[1:-1]:
            a.axvline(m(b), color="white", lw=0.9, ls="--")
    axs.set_yticks(np.arange(K))
    axs.set_yticklabels(d["flabels"], fontsize=6.3 if K <= 22 else 5.6)
    for j in range(1, K):
        if cond[j] != cond[j - 1]:
            axs.axhline(j - 0.5, color="0.45", lw=0.6)
    plt.setp(axs.get_xticklabels(), visible=False)
    bt = blk_ if bar_movie else d["best"][d["top"]]
    axk.set_yticks([])
    cuts = np.r_[0, np.flatnonzero(np.diff(bt)) + 1, len(bt)]
    for a_, b_ in zip(cuts[:-1], cuts[1:]):
        if a_:
            axk.axhline(a_ - 0.5, color="0.6", lw=0.7)
        axk.text(-0.008, (a_ + b_ - 1) / 2, (f"{names[bt[a_]]} ({b_ - a_})" if bar_movie else
                                              f"best {d['flabels'][bt[a_]]} ({b_ - a_})"),
                 transform=axk.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, color="0.85")
    if d.get("pick") is not None and not bar_movie:               # Cedric, 2026-10-06: which block is visual, which motor (ephys)
        pk = d["pick"][d["top"]]
        for code, col, lab_ in ((0, "#56b4e9", "visual"), (1, "#ff9f1c", "motor (ephys)")):
            rr = np.flatnonzero(pk == code)
            if not len(rr):
                continue
            a_, b_ = rr.min() - 0.5, rr.max() + 0.5
            axk.plot([-0.19, -0.19], [a_ + 0.6, b_ - 0.6], transform=axk.get_yaxis_transform(), color=col, lw=3,
                     clip_on=False, solid_capstyle="butt")
            axk.text(-0.20, (a_ + b_) / 2, f"{lab_} ({len(rr)})", transform=axk.get_yaxis_transform(), rotation=90,
                     ha="right", va="center", fontsize=9, color=col, weight="bold")
            if code == 1:
                axk.axhline(a_, color="white", lw=1.2)
    if bar_movie:
        plt.setp(axk.get_xticklabels(), visible=False)
    else:
        axk.set_xlabel("time since the recording's start, min", fontsize=9, color="0.8")
    fig.text(L, 0.855, "stimulus features, grouped by condition: -1 (blue) .. 0 (black) .. +1 (orange)", color="white",
             fontsize=9.5, va="bottom")
    fig.text(L, 0.525, ("per block, the 20 input neurons that respond most in it: dF/F, rows z-scored" if bar_movie else
                        "the 100 most coherent input neurons: dF/F, rows z-scored, grouped by best feature (n), then by lag"),
             color="white", fontsize=9.5, va="bottom")
    cb1 = fig.add_axes([R + 0.01, 0.62, 0.008, 0.21])
    fig.colorbar(ims, cax=cb1).ax.tick_params(colors="0.75", labelsize=7)
    cb1.set_title("value", fontsize=7, color="0.75")
    cb2 = fig.add_axes([R + 0.01, 0.10, 0.008, 0.40] if not bar_movie else [R + 0.01, 0.20, 0.008, 0.30])
    fig.colorbar(imk, cax=cb2).ax.tick_params(colors="0.75", labelsize=7)
    cb2.set_title("z", fontsize=7, color="0.75")
    # THE INPUT NEURONS ON THE BRAIN, vertical and head up, beside the kymographs: the mask's neurons coloured (the
    # half-and-half mask: blue those picked by visual coherence, orange those picked by ephys), the others grey
    P = d["pos"]
    V = np.stack([P[:, 1], -P[:, 0]], 1)        # brain_view is horizontal, head left: turned to head up
    if bar_movie:                               # Cedric, 2026-10-06: two fish -- the input neurons, and the activity now
        return _kymo_bar_movie(d, fig, axs, axk, V, m, tmin, n_frames, fps)
    axb = fig.add_axes([0.745, 0.06, 0.24, 0.84])
    axb.set_facecolor("black")
    axb.axis("off")
    axb.set_aspect("equal")
    sel = d["sel"]
    axb.scatter(V[~sel, 0], V[~sel, 1], s=0.15, c="0.25", linewidths=0)
    if d.get("pick") is not None:
        for code, col, lab_ in ((0, "#56b4e9", "picked by visual coherence"), (1, "#ff9f1c", "picked by ephys coherence")):
            m_ = d["pick"] == code
            axb.scatter(V[m_, 0], V[m_, 1], s=0.5, c=col, linewidths=0)
            fig.text(0.745, 0.035 - 0.022 * code, f"{lab_}: {int(m_.sum()):,}", color=col, fontsize=8.5)
    else:
        axb.scatter(V[sel, 0], V[sel, 1], s=0.5, c="#ff9f1c", linewidths=0)
    fig.text(0.865, 0.92, f"the input neurons: {int(sel.sum()):,} of {len(sel):,}", color="white", fontsize=10,
             ha="center", va="bottom")
    v_ = d.get("variant", "all")
    stem = f"input_neurons_{d['rec']}{'' if v_ == 'all' else '_' + v_}_kymo_full.png"
    out = os.path.join(FIGS, stem)
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.copy(out, os.path.join(PNG, stem))
    print(f"[input neurons] wrote {out}")


def _kymo_bar_movie(d, fig, axs, axk, V, m, tmin, n_frames, fps):
    """THE KYMOGRAPH AS A MOVIE (Cedric, 2026-10-06: "two zebrafishes, a vertical bar moving on the two kymographs"):
    fig_kymo_full's kymographs with a white bar at the current frame, and beside them two brains, head up -- left the
    input neurons (the mask, fixed), right every neuron's dF/F at that frame (inferno, 0 .. the 97th percentile). One
    frame of the movie every T / n_frames recorded frames, the whole recording; the poster is its first frame.
    -> Movies/input_neurons_<rec>_<variant>_kymo_bar.mp4 (+ .png)."""
    import tempfile
    import matplotlib.pyplot as plt
    from plexus.tasks.trace_recording import _ffmpeg
    X, sel, T = d["X"], d["sel"], d["T"]
    vmax = float(np.percentile(np.asarray(X[::50]), 97))
    # Cedric, 2026-10-06: 2 x 3 fish -- rows the 5 %, 10 % and 20 % visual masks (nested), left each mask's input
    # neurons, right the others, each neuron with its own activity, the excluded set faint grey
    import matplotlib.pyplot as plt_
    from plexus.paths import graphs_data_path
    cm_ = plt_.get_cmap("inferno")
    U_ = np.asarray(d["U"])
    # Cedric, 2026-10-07: 2 x 2 -- the balanced 20 % mask read per feature (an input neuron lit only while one of its
    # features is on) against the same mask with every input neuron reading all the stimulus (lit always)
    z_ = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))
    masks = [("per feature", z_["mask"] > 0, z_["mask_by_input"] > 0), ("all features", z_["mask"] > 0, None)]
    masks = []                                # Cedric, 2026-10-07: the fish removed, redundant with slides 9-10
    # the fish drawn with VTK, as the deck's other brains (Cedric, 2026-10-06): one off-screen plotter per panel, head
    # up, seen from above, parallel projection; each frame's image pasted into its axes
    import pyvista as pv
    pv.OFF_SCREEN = True
    P3 = np.column_stack([V[:, 0], V[:, 1], np.zeros(len(V))]).astype(np.float32)
    lo2, hi2 = V.min(0), V.max(0)
    ctr = (lo2 + hi2) / 2

    def plotter(grey, col):
        pl = pv.Plotter(off_screen=True, window_size=(300, 380))
        pl.set_background("black")
        pl.add_mesh(pv.PolyData(P3[grey]), color="#262626", point_size=2.0, render_points_as_spheres=True)
        mesh = pv.PolyData(P3[col])
        mesh.point_data["rgb"] = np.zeros((len(col), 3), np.uint8)
        pl.add_mesh(mesh, scalars="rgb", rgb=True, point_size=4.0, render_points_as_spheres=True)
        pl.enable_parallel_projection()
        pl.camera.focal_point = (float(ctr[0]), float(ctr[1]), 0.0)
        pl.camera.position = (float(ctr[0]), float(ctr[1]), 5000.0)
        pl.camera.up = (0.0, 1.0, 0.0)
        pl.camera.parallel_scale = float(hi2[1] - lo2[1]) / 2 * 1.03
        return pl, mesh
    scs = []
    for r, (lab, mk, mbi) in enumerate(masks):
        y0 = 0.07 + (len(masks) - 1 - r) * 0.44
        for c_, (part, name) in enumerate(((mk, "input"), (~mk, "others"))):
            a_ = fig.add_axes([0.745 + 0.123 * c_, y0, 0.12, 0.38])
            a_.axis("off")
            a_.set_title(f"{lab}: {name} {int(part.sum()):,}" if c_ == 0 else f"{name} {int(part.sum()):,}",
                         color="white", fontsize=8.5)
            ix = np.where(part)[0]
            pl_, mesh_ = plotter(np.where(~part)[0], ix)
            im_ = a_.imshow(np.zeros((380, 300, 3), np.uint8), aspect="equal")
            scs.append((pl_, mesh_, im_, ix, mbi[ix][:, :U_.shape[1]] if (c_ == 0 and mbi is not None) else None))
    ls = axs.axvline(0.0, color="white", lw=1.6)
    lk = axk.axvline(0.0, color="white", lw=1.6)
    lb = fig._bm_ax.axvline(0.0, color="white", lw=1.6) if hasattr(fig, "_bm_ax") else None
    tt = fig.text(0.93, 0.855, "", color="white", fontsize=11, ha="right", va="bottom", weight="bold")
    tmp = tempfile.mkdtemp(prefix="kymo_bar_")
    CROP_TOP = 0.065                 # the band the title held (0.92 .. 1 of the height), cut: no title in the movie
    frames = np.linspace(0, T - 1, n_frames).astype(int)
    for i, t in enumerate(frames):
        ls.set_xdata([m(t), m(t)])
        lk.set_xdata([m(t), m(t)])
        if lb is not None:
            lb.set_xdata([m(t), m(t)])
        xt = np.asarray(X[t])
        on_ = np.abs(U_[t]) > 0                      # the stimulus features on at this frame
        for pl_, mesh_, im_, ix, fm_ in scs:
            rgba = cm_(np.clip(xt[ix] / vmax, 0.0, 1.0))
            if fm_ is not None:                      # an input neuron: its colour only while one of its features is on
                rgba[~fm_[:, on_].any(1)] = (0.18, 0.18, 0.18, 1.0)
            mesh_.point_data["rgb"] = (rgba[:, :3] * 255).astype(np.uint8)
            pl_.render()
            im_.set_data(pl_.screenshot(return_img=True))
        tt.set_text(f"t = {m(t):5.1f} of {tmin:.0f} min")
        fig.savefig(os.path.join(tmp, f"{i:05d}.png"), dpi=110, facecolor="black")
    plt.close(fig)
    for pl_, *_r in scs:
        pl_.close()
    v_ = d.get("variant", "all")
    stem = os.path.join(MOVIES, f"input_neurons_{d['rec']}{'' if v_ == 'all' else '_' + v_}_kymo_bar")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(3 * fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-r", str(fps),                    # 3x, frames dropped (Cedric, 2026-10-07)
                    "-vf", f"crop=iw:ih*{1 - CROP_TOP}:0:ih*{CROP_TOP},scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
                    "-c:v", "libx264", stem + ".mp4"], check=True)
    from PIL import Image
    im0 = Image.open(os.path.join(tmp, "00000.png"))
    im0.crop((0, int(im0.height * CROP_TOP), im0.width, im0.height)).save(stem + ".png")
    shutil.rmtree(tmp)
    print(f"[input neurons] wrote {stem}.mp4")
    return stem


def movie_kymo(d, W=300, step=10, fps=25, dpi=120, workers=16):
    import multiprocessing as mp
    from plexus.tasks.trace_recording import _ffmpeg
    T = d["T"]
    starts = np.unique(np.r_[np.arange(0, T - W + 1, step), T - W])
    tmp = tempfile.mkdtemp(prefix="inputkymo_")
    _KY.clear()
    _KY.update(rec=d["rec"], T=T, W=W, starts=starts, off=d["off"], names=d["names"], K=d["K"],
               feat_cond=d["feat_cond"], U=d["U"].astype(np.float32), Z=d["Zt"].astype(np.float32),
               best_top=d["best"][d["top"]], tmp=tmp, dpi=dpi)
    ks = np.arange(len(starts))
    workers = min(workers, len(os.sched_getaffinity(0)))
    chunks = [c for c in np.array_split(ks, max(1, min(workers, len(ks)))) if len(c)]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        pool.map(_kymo_frames, chunks)
    out = os.path.join(MOVIES, f"input_neurons_{d['rec']}_kymo.mp4")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", out],
                   check=True)
    # the still: a window in the turning condition (the commonest best feature's), else the middle
    c = d["feat_cond"][d["common"]]
    k_still = int(np.argmin(np.abs(starts - (d["off"][c] + d["off"][c + 1] - W) // 2)))
    still = out.replace(".mp4", ".png")
    shutil.copy(os.path.join(tmp, f"{k_still:05d}.png"), still)
    shutil.copy(still, os.path.join(PNG, os.path.basename(still)))
    shutil.rmtree(tmp)
    _KY.clear()
    print(f"[input neurons] wrote {out} ({len(starts)} frames, {len(starts) / fps:.1f} s) and {still}")


# ------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", nargs="+", default=["zapbench", "zapbench_destripe"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--bar-movie", action="store_true", help="also the kymograph movie: a moving time bar and two fish")
    ap.add_argument("--bar-frames", type=int, default=300, help="frames of the kymograph movie, over the whole recording")
    ap.add_argument("--mask", default="all", choices=["all", "varying", "mix", "vis20"],
                    help="varying: the mask from the 13 features that change within their condition")
    a = ap.parse_args()
    for p in (FIGS, MOVIES, PNG):
        os.makedirs(p, exist_ok=True)
    for rec in a.recording:
        d = analyse(rec, a.device, a.mask)
        fig_kymo_full(d)
        if a.bar_movie:
            fig_kymo_full(d, bar_movie=True, n_frames=a.bar_frames)
        s = summary(d)
        fig_method(d)
        fig_map(d, s)
        bt = d["best"][d["top"]]
        info = {"recording": rec, "neurons": d["N"], "selected": int(d["sel"].sum()), "threshold": d["thr"],
                "band_periods_s": [round(float(1 / d["f"][b]), 1) for b in d["band"]],
                "example_selected": d["ex_sel"], "example_unselected": d["ex_un"],
                "selected_per_best_feature": {flab(d, j): int(s["n"][j]) for j in range(d["K"]) if s["n"][j]},
                "own_best_abs_r": {"selected": s["own_r_sel"], "random_unselected_100": s["own_r_rnd"]},
                "own_best_band_coherence": {"selected": s["own_c_sel"], "random_unselected_100": s["own_c_rnd"]},
                "kymo_100_per_best_feature": {flab(d, j): int((bt == j).sum()) for j in np.unique(bt)},
                "kymo_100_lag_frames_range": [int(d["lag"].min()), int(d["lag"].max())]}
        with open(os.path.join(FIGS, f"input_neurons_{rec}_numbers.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        print(json.dumps(info, indent=1))
        if not a.no_movie:
            movie_kymo(d)


if __name__ == "__main__":
    main()

"""exp17: HOW FAR IS EACH NEURON FROM THE BRAIN MEAN? (Cedric, 2026-10-06: "a plot to visualise how close, how different the
100K neurons are from the mean, in amplitude and frequency domain"). Local, model-free, the recording only.

b(t) the brain mean, the mean dF/F over every neuron at frame t; x_i(t) neuron i's dF/F. Over the whole recording:
  AMPLITUDE   r_i       the correlation of x_i with b -- the shape: +1 a neuron that only follows the brain mean
              s_i / s_b neuron i's SD over the brain mean's SD -- the size of its swings against the mean's
              R2_i      r_i^2, the share of neuron i's variance the brain mean carries (the regression x_i = a_i + beta_i b
                        + e_i, beta_i = r_i s_i / s_b, what tools/exp17_slides.local_r removes)
  FREQUENCY   P_i(f)    neuron i's power spectrum, each normalised to unit total power (the shape of the spectrum, not its
                        size), against the brain mean's, normalised the same
              C_i(f)    the coherence of x_i with b, |<X_i B*>|^2 / (<|X_i|^2> <|B|^2>): 1 at a frequency where the neuron
                        moves in a fixed ratio with the brain mean, 0 where it is unrelated
  P and C by Welch: 256-frame Hann windows (234 s), half overlapping, the window means removed (exp17_stim_coherence's).
Only the neurons that move (SD > 0). Percentile bands over the neurons: 10-90 % and 25-75 %, the median a line.

    PYTHONPATH=src:tools python tools/exp17_brain_mean_spread.py [--recording zapbench_destripe]
-> presentation/figs/brain_mean_spread_<rec>.png, data/brain_mean_spread_<rec>.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914


def welch_vs(X, b, nper=256, chunk=8000):
    """Per column of X [T, N]: its power spectrum and its cross-spectrum with b [T], Welch (Hann, half overlap, each
    window's mean removed). Returns f [F] in Hz, Pxx [F, N], Pbb [F], Pxb [F, N] complex."""
    T = len(b)
    st = np.arange(0, T - nper + 1, nper // 2)
    w = np.hanning(nper).astype(np.float32)

    def seg(A):                                                         # [S, nper, ...] windowed, demeaned
        S = np.stack([A[s:s + nper] for s in st]).astype(np.float32)
        S = S - S.mean(1, keepdims=True)
        return S * (w[:, None] if S.ndim == 3 else w)
    Bf = np.fft.rfft(seg(b), axis=1)                                     # [S, F]
    Pbb = (np.abs(Bf) ** 2).mean(0)
    F = Bf.shape[1]
    Pxx = np.empty((F, X.shape[1]), np.float32)
    Pxb = np.empty((F, X.shape[1]), np.complex64)
    for a in range(0, X.shape[1], chunk):
        Xf = np.fft.rfft(seg(X[:, a:a + chunk]), axis=1)                 # [S, F, n]
        Pxx[:, a:a + chunk] = (np.abs(Xf) ** 2).mean(0)
        Pxb[:, a:a + chunk] = (Xf * np.conj(Bf)[:, :, None]).mean(0)
    return np.fft.rfftfreq(nper, FRAME_S), Pxx, Pbb, Pxb


def umap_of_traces(X, b, rec_name, n_pc=50):
    """The UMAP of the z-scored traces and the brain mean's place on it; cached in data/brain_mean_umap_<rec>.npz."""
    cache = os.path.join(EXP, "data", f"brain_mean_umap_{rec_name}.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        if c["U"].shape[0] == X.shape[1]:
            return c["U"], c["ub"]
    import umap
    from sklearn.decomposition import PCA
    Z = (X - X.mean(0)) / X.std(0)                                       # [T, n], each neuron's shape
    pca = PCA(n_pc, svd_solver="randomized", random_state=0).fit(Z.T)
    Y = pca.transform(Z.T)
    zb = ((b - b.mean()) / b.std()).astype(np.float32)[None]
    um = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit(Y)
    U, ub = um.embedding_, um.transform(pca.transform(zb))[0]
    np.savez(cache, U=U, ub=ub, pca_var=pca.explained_variance_ratio_)
    return U, ub


def main(rec_name="zapbench_destripe"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{rec_name}_recording.npz"))
    X = np.asarray(z["dff"], np.float32)
    T, N = X.shape
    b = X.mean(1, dtype=np.float64)
    sd = X.std(0, dtype=np.float64)
    mv = sd > 1e-9
    X = X[:, mv]
    sd = sd[mv]
    sb = float(b.std())
    Xc = X - X.mean(0, dtype=np.float64).astype(np.float32)
    bc = (b - b.mean()).astype(np.float32)
    r = (bc @ Xc) / (T * sb * sd)                                        # [n] correlation with the brain mean
    amp = sd / sb
    f, Pxx, Pbb, Pxb = welch_vs(X, b)
    C = np.abs(Pxb) ** 2 / (Pxx * Pbb[:, None] + 1e-30)
    keep = f > 0                                                         # the 0 frequency is the removed window mean
    f, Pxx, Pbb, C = f[keep], Pxx[keep], Pbb[keep], C[keep]
    Pn = Pxx / Pxx.sum(0, keepdims=True)                                 # each spectrum's shape: unit total power
    Pbn = Pbb / Pbb.sum()
    q = lambda A: np.percentile(A, [10, 25, 50, 75, 90], axis=1)        # noqa: E731
    qP, qC = q(Pn), q(C)
    bands = {"slow, periods > 100 s": f < 0.01, "middle, 10-100 s": (f >= 0.01) & (f < 0.1), "fast, periods < 10 s": f >= 0.1}
    doc = {"recording": rec_name, "frames": int(T), "neurons": int(N), "moving": int(mv.sum()),
           "brain_mean_sd": sb, "r_median": float(np.median(r)), "r_q10_q90": np.percentile(r, [10, 90]).tolist(),
           "r_negative_share": float((r < 0).mean()), "R2_median": float(np.median(r ** 2)),
           "amp_median": float(np.median(amp)), "amp_q10_q90": np.percentile(amp, [10, 90]).tolist(),
           "welch_window_frames": 256, "freq_hz": f.tolist(),
           "coherence_median_by_band": {k: float(np.median(C[m_])) for k, m_ in bands.items()},
           "power_share_by_band_brain_mean": {k: float(Pbn[m_].sum()) for k, m_ in bands.items()},
           "power_share_by_band_neuron_median": {k: float(np.median(Pn[m_].sum(0))) for k, m_ in bands.items()}}
    json.dump(doc, open(os.path.join(EXP, "data", f"brain_mean_spread_{rec_name}.json"), "w"), indent=1)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 9.4), facecolor="black")
    # Cedric, 2026-10-07 ("too many colours"): grey for the neurons, green for the brain mean as on the deck's other
    # slides, light green the neuron closest to it, white the others
    GY, GB, GL = "0.62", "#2ca02c", "#a6e3a1"
    # the neurons drawn at the right, also marked on the scatter: the closest to b, one at r = 0, one with large swings
    i_close, i_far = int(np.argmax(r)), int(np.argmin(np.abs(r)))
    cand_ = np.where(np.abs(r - 0.25) < 0.02)[0]
    i_big = int(cand_[np.argmin(np.abs(amp[cand_] - 10.0))])
    cand_ = np.where(np.abs(r - 0.32) < 0.005)[0]          # the typical neuron: r at the median, swings at the median
    i_med = int(cand_[np.argmin(np.abs(amp[cand_] - np.median(amp)))])
    # a: r_i
    a = fig.add_axes([0.06, 0.58, 0.25, 0.36])
    a.hist(r, bins=np.linspace(-1, 1, 101), color=GY, lw=0)
    a.axvline(np.median(r), color="white", lw=1.2, ls="--")
    a.set_xlabel("correlation with brain mean", fontsize=10)
    a.set_ylabel("neurons", fontsize=10)
    a.text(0.03, 0.95, f"median {np.median(r):+.2f}\n{100 * (r < 0).mean():.0f} % below 0", transform=a.transAxes,
           va="top", fontsize=9.5)
    # b: r_i against s_i / s_b
    a2 = fig.add_axes([0.38, 0.58, 0.27, 0.36])
    # a linear amplitude axis (Cedric, 2026-10-06), up to the 99.5th percentile of s_i / s_b
    h = a2.hist2d(r, amp, bins=[np.linspace(-1, 1, 120), np.linspace(0, np.percentile(amp, 99.5), 120)],
                  cmap="gray", norm=LogNorm(), cmin=1)          # a grey LUT (Cedric, 2026-10-07)
    a2.axhline(1.0, color="white", lw=0.8, ls=":")
    a2.set_xlabel("correlation with brain mean", fontsize=10)
    a2.set_ylabel("swings, $s_i / s_b$", fontsize=10)
    for i_, lab_, off_ in ((i_close, "closest", (-7, -1)), (i_far, "r = 0", (5, 4)), (i_big, "large swings", (5, 4)),
                           (i_med, "r = 0.32", (5, 4))):
        a2.scatter([r[i_]], [amp[i_]], s=28, color="white", edgecolors="black", lw=0.6, zorder=5)
        if lab_ == "large swings":                     # its dot kept, its label removed (Cedric, 2026-10-07)
            continue
        a2.annotate(lab_, (r[i_], amp[i_]), xytext=off_, textcoords="offset points", color="white", fontsize=8.5,
                    ha="right" if lab_ == "closest" else "left", va="center" if lab_ == "closest" else "baseline")
    a2.scatter([1.0], [1.0], s=28, color=GB, edgecolors="black", lw=0.6, zorder=5)      # b itself: r 1, s/s_b 1
    a2.annotate("mean", (1.0, 1.0), xytext=(1, 7), textcoords="offset points", color=GB, fontsize=8.5, ha="right")
    cb = fig.colorbar(h[3], ax=a2, pad=0.01, fraction=0.05)
    cb.set_label("neurons per bin", fontsize=9)
    a2.text(0.03, 0.95, f"$s_i/s_b$ median {np.median(amp):.1f}\n(10-90 %: {np.percentile(amp, 10):.1f} .. "
            f"{np.percentile(amp, 90):.1f})", transform=a2.transAxes, va="top", fontsize=9.5)
    # c: the spectra's shapes
    per = f                                        # Cedric, 2026-10-07: the conventional axis, frequency in Hz
    a3 = fig.add_axes([0.06, 0.09, 0.27, 0.36])
    a3.fill_between(per, qP[0], qP[4], color=GY, alpha=0.25, lw=0, label="neurons, 10-90 %")
    a3.fill_between(per, qP[1], qP[3], color=GY, alpha=0.45, lw=0, label="neurons, 25-75 %")
    a3.plot(per, qP[2], color="0.85", lw=1.4, label="neurons, median")
    a3.plot(per, Pbn, color=GB, lw=1.8, label="the brain mean")
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xlabel("frequency, Hz", fontsize=10)
    a3.set_ylabel("share of the trace's power", fontsize=10)
    a3.legend(fontsize=8.5, frameon=False, loc="lower left")
    # d: THE RASTER (Cedric, 2026-10-07: "replace the UMAP with a raster of all the moving neurons, sorted by their
    # correlation with the brain mean, the brain-mean trace on top on the same time axis"): each neuron z-scored over
    # the recording, the neurons sorted by r_i (highest at the top), averaged in 600 bins of consecutive ranks (~160
    # neurons each, so every neuron counts and the image fits the slide); grey, the brain mean above it in green
    NB = 600
    rk = np.argsort(-r)
    edges = np.linspace(0, len(rk), NB + 1).astype(int)
    mu_, sd_ = X.mean(0, dtype=np.float64).astype(np.float32), sd.astype(np.float32)
    img = np.empty((NB, T), np.float32)
    for q in range(NB):
        ids_ = rk[edges[q]:edges[q + 1]]
        img[q] = ((X[:, ids_] - mu_[ids_]) / sd_[ids_]).mean(1)
    tmin_ = T * FRAME_S / 60
    a5 = fig.add_axes([0.415, 0.385, 0.225, 0.065])
    a5.plot(np.arange(T) * FRAME_S / 60, b, color=GB, lw=0.5)
    a5.set_xlim(0, tmin_)
    a5.set_xticklabels([])
    a5.set_ylabel("brain\nmean", fontsize=8.5, color=GB)
    a4 = fig.add_axes([0.415, 0.09, 0.225, 0.285])
    vm_ = float(np.percentile(np.abs(img), 99))
    a4.imshow(img, aspect="auto", cmap="gray", vmin=-0.3 * vm_, vmax=vm_, extent=(0, tmin_, len(rk), 0),
              interpolation="nearest")
    a4.set_xlabel("time, min", fontsize=10)
    a4.set_ylabel("sorted by correlation", fontsize=9.5, labelpad=2)
    yt_ = [0, len(rk) // 4, len(rk) // 2, 3 * len(rk) // 4, len(rk) - 1]
    a4.set_yticks(yt_)
    a4.set_yticklabels([f"r {r[rk[y_]]:+.2f}" for y_ in yt_], fontsize=8)
    a5.set_facecolor("black")
    a5.tick_params(labelsize=7.5)
    for ax_ in (a, a2, a3, a4):                 # no panel letters: the slide names each panel by its place (Cedric, 2026-10-06)
        ax_.set_facecolor("black")
        ax_.tick_params(labelsize=8.5)
    # right: the brain mean, the same with the band 95 % of the neurons fall in, then four neurons -- the closest, a
    # typical one, one at r = 0, one with large swings (Cedric, 2026-10-07: each title inside its box, the boxes closer)
    tm = np.arange(T) * FRAME_S / 60
    q95 = np.concatenate([np.percentile(X[a_:a_ + 500], [2.5, 97.5], axis=1) for a_ in range(0, T, 500)], axis=1)
    rows_ = ((None, "the brain mean", GB), ("band", "the brain mean, the band 95 % of the neurons", GB),
             (i_close, f"the closest neuron, r {r[i_close]:+.2f}", "white"),
             (i_med, f"a typical neuron, r {r[i_med]:+.2f}, $s_i/s_b$ {amp[i_med]:.1f}", "white"),
             (i_far, f"a neuron at r {r[i_far]:+.2f}", "white"),
             (i_big, f"large swings, $s_i/s_b$ {amp[i_big]:.1f}, r {r[i_big]:+.2f}", "white"))
    for k_, (i_, lab_, c_) in enumerate(rows_):
        ax_ = fig.add_axes([0.745, 0.835 - 0.153 * k_, 0.235, 0.118])
        if i_ is None:
            ax_.plot(tm, b, color=GB, lw=0.5)
        elif i_ == "band":
            ax_.fill_between(tm, q95[0], q95[1], color="white", alpha=0.25, lw=0)
            ax_.plot(tm, b, color=GB, lw=0.6)
        else:
            ax_.plot(tm, X[:, i_], color=c_, lw=0.4)
            if i_ in (i_close, i_med, i_far, i_big):   # the brain mean beside each, for comparison (Cedric, 2026-10-07)
                ax_.plot(tm, b, color=GB, lw=0.6)
        ax_.text(0.01, 0.97, lab_, transform=ax_.transAxes, fontsize=8.5, color=c_, va="top", ha="left",
                 bbox=dict(facecolor="black", edgecolor="none", alpha=0.6, pad=1.0))
        ax_.set_xlim(tm[0], tm[-1])
        ax_.set_facecolor("black")
        ax_.tick_params(labelsize=7.5, pad=1.5)
        ax_.set_ylabel("dF/F", fontsize=8.5, labelpad=1)
        ax_.set_xlabel("time, min", fontsize=8, labelpad=0)               # on every trace (Cedric, 2026-10-06)
    # each panel also on its own (Cedric, 2026-10-07: three of them reused next to the atlas raster): the
    # correlation-sorted raster with its brain mean, the spectra, the correlation-against-swings density
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    inch = fig.dpi_scale_trans.inverted()
    for tag_, axs_ in (("raster", (a5, a4)), ("spectrum", (a3,)), ("swings", (a2, cb.ax))):
        bb = Bbox.union([x_.get_tightbbox(rnd) for x_ in axs_]).transformed(inch).expanded(1.02, 1.03)
        fig.savefig(os.path.join(EXP, "presentation", "figs", f"brain_mean_spread_{rec_name}_{tag_}.png"), dpi=200,
                    facecolor="black", bbox_inches=bb)
    out = os.path.join(EXP, "presentation", "figs", f"brain_mean_spread_{rec_name}.png")
    fig.savefig(out, dpi=130, facecolor="black")
    plt.close(fig)
    print(f"[brain mean spread] {out}\n" + json.dumps({k: v for k, v in doc.items() if k != "freq_hz"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="zapbench_destripe")
    main(ap.parse_args().recording)

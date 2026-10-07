"""exp17: WHICH NEURONS LEAD THE BRAIN MEAN, WHICH LAG IT (Cedric, 2026-10-07: "which neurons are in advance of the brain
mean, which are late? with the z-slice time correction"). Local, model-free, the recording only.

THE LAG. b(t) the brain mean (every neuron's dF/F at frame t, averaged). Per neuron, the cross-correlation of its trace
with b at shifts of -K .. K frames (K = 10, +-9 s), r_i(k) = corr(x_i(t + k), b(t)); its peak k*, refined below a frame
by a parabola through the peak and its two neighbours; the raw lag l_i = k* x 0.914 s: > 0 the neuron moves AFTER the
brain mean, < 0 before it.

THE SLICE TIME. The 72 planes of a volume are imaged one after another, ventral to dorsal (Lueckmann et al. 2025, A.2:
"after each plane was acquired, the objectives were moved dorsally by 4 um"), over the 0.914-s frame: plane p is sampled
s_p = (p / 72) x 0.914 s into the frame. A neuron identical to b would show a raw lag -(s_i - s_mean) (its frame-t sample
taken s_i - s_mean later than b's). The corrected lag
    tau_i = l_i + (s_i - s_mean),        s_mean the mean of s_i over the neurons (b's own sampling time)
CHECK, before any correction: over the neurons that follow b, the raw lag against depth should fall with slope
-0.914 s / 288 um (-3.2 ms per um) if z grows dorsally -- the fitted slope is printed and saved; its sign fixes which
end of z is ventral (the correction uses it).

Only the neurons whose peak correlation with b exceeds R_MIN (0.3) are given a lag: for the others it is not defined.

    PYTHONPATH=src:tools python tools/exp17_brain_mean_lag.py
-> presentation/figs/brain_mean_lag_<rec>.png, data/brain_mean_lag_<rec>.json (+ .npz, per neuron)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S, PLANES, PLANE_UM, K, R_MIN = 0.914, 72, 4.0, 10, 0.3
R_IN = 0.6         # the r threshold against the input neurons' mean (slide 5)
SIG = 1.0          # s: a lead within +-SIG counts as neither leading nor trailing (Cedric, 2026-10-07: 1 s)


def xcorr_peak(X, b, K, chunk=4000):
    """Per column of X [T, n]: the correlation with b [T] at shifts -K..K (x(t + k) against b(t)), its peak value and
    the peak shift refined by a parabola (frames). Returns (r_peak [n], k_peak [n], r_at [n, 2K + 1])."""
    T = len(b)
    bc = (b - b.mean()) / b.std()
    nf = 1 << int(np.ceil(np.log2(2 * T)))
    Bf = np.conj(np.fft.rfft(bc, nf))
    out_r, out_k, out_all = [], [], []
    for a in range(0, X.shape[1], chunk):
        Xc = np.asarray(X[:, a:a + chunk], np.float64)
        Xc = (Xc - Xc.mean(0)) / np.maximum(Xc.std(0), 1e-12)
        cc = np.fft.irfft(np.fft.rfft(Xc, nf, axis=0) * Bf[:, None], nf, axis=0)      # cc[k] = sum_t x(t + k) b(t)
        lags = np.concatenate([cc[nf - K:], cc[:K + 1]], 0) / T                        # k = -K .. K
        i0 = np.argmax(lags, 0)
        r0 = lags[i0, np.arange(lags.shape[1])]
        im, ip = np.clip(i0 - 1, 0, 2 * K), np.clip(i0 + 1, 0, 2 * K)
        ym, yp = lags[im, np.arange(lags.shape[1])], lags[ip, np.arange(lags.shape[1])]
        den = ym - 2 * r0 + yp
        dk = np.where((i0 > 0) & (i0 < 2 * K) & (np.abs(den) > 1e-12), 0.5 * (ym - yp) / np.where(den == 0, 1, den), 0.0)
        out_r.append(r0)
        out_k.append(i0 - K + np.clip(dk, -0.5, 0.5))
        out_all.append(lags.T.astype(np.float32))
    return np.concatenate(out_r), np.concatenate(out_k), np.concatenate(out_all)


def phase_lag(X, b, f_lo=0.004, f_hi=0.05):
    """Per column of X: its lag to b from the PHASE of their cross-spectrum (Cedric, 2026-10-07: unbiased below a
    frame, where the parabola through the correlation peak pulls small shifts to whole frames). Welch cross-spectrum
    S(f) = <X(f) B*(f)> (exp17_brain_mean_spread.welch_vs: 256-frame Hann windows, half overlap); x(t) = b(t - tau)
    gives arg S(f) = -2 pi f tau, so tau = -sum w phi omega / sum w omega^2 over the bins f_lo..f_hi (periods 20-250 s),
    w the coherence |S|^2 / (Pxx Pbb) at each bin. Returns (tau [n] s, mean coherence over the bins [n])."""
    from exp17_brain_mean_spread import welch_vs
    f, Pxx, Pbb, Pxb = welch_vs(X, b)
    sel = (f >= f_lo) & (f <= f_hi)
    om = 2 * np.pi * f[sel][:, None]
    phi = np.angle(Pxb[sel])
    w = np.abs(Pxb[sel]) ** 2 / (Pxx[sel] * Pbb[sel][:, None] + 1e-30)
    tau = -(w * phi * om).sum(0) / np.maximum((w * om ** 2).sum(0), 1e-30)
    return tau, w.mean(0)


def _maps(fig, Bv, lead, sel, rect_top, rect_side, title_top="from above, head left", dot=1.6):
    """The brain from above (horizontal, head left) and from the side below it: the neurons of `sel` whose lead is
    beyond +-SIG s coloured by it (red ahead, blue behind), larger dots; no background of the other neurons
    (Cedric, 2026-10-07)."""
    # Cedric, 2026-10-07: a three-level LUT -- red a lead of more than SIG, blue a trail of more than SIG, black between
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#4a7bff", "black", "#ff4a4a"])
    nrm = BoundaryNorm([-1e3, -SIG, SIG, 1e3], 3)
    sig = sel & (np.abs(np.nan_to_num(lead)) >= SIG)
    o = np.argsort(np.abs(np.nan_to_num(lead)))
    oo = o[sig[o]]
    out = None
    for rect, (i0, i1), ttl in ((rect_top, (0, 1), title_top), (rect_side, (0, 2), "from the side")):
        a = fig.add_axes(rect)
        out = a.scatter(Bv[oo, i0], Bv[oo, i1], c=lead[oo], cmap=cmap, norm=nrm, s=dot, lw=0, rasterized=True)
        a.set_xlim(Bv[:, i0].min(), Bv[:, i0].max())
        a.set_ylim(Bv[:, i1].min(), Bv[:, i1].max())
        a.set_aspect("equal")
        a.axis("off")
        a.set_title(ttl, fontsize=10, loc="left")
    return out


def lag_figure(plt, Bv, tau, ok, zc_, md_, exp_slope, r_min, who, ref="the brain mean"):
    """Drawn as the LEAD = -lag (Cedric, 2026-10-07: "use lead"): > 0 the neuron moves before `ref`."""
    lead, mdl = -tau, -md_
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    GY, GB = "0.62", "#2ca02c"
    a2 = fig.add_axes([0.06, 0.30, 0.27, 0.45])       # the histogram alone on the left (no depth panel: Cedric)
    a2.hist(np.clip(lead[ok], -6, 6), bins=np.linspace(-6, 6, 121), color=GY, lw=0)
    a2.axvspan(-SIG, SIG, color="0.25", lw=0, zorder=0)
    a2.axvline(0, color=GB, lw=1.2)
    a2.set_xlabel(f"lead over {ref}, s  (> 0 moves first)", fontsize=10)
    a2.set_ylabel("neurons", fontsize=10)
    a2.text(0.03, 0.95, f"{int(ok.sum()):,} neurons with r > {r_min}\nmedian {np.median(lead[ok]):+.2f} s\n"
            f"{100 * (lead[ok] > SIG).mean():.0f} % lead by > {SIG:g} s\n{100 * (lead[ok] < -SIG).mean():.0f} % trail by > {SIG:g} s"
            f"\n(grey band: within +-{SIG:g} s, not drawn)", transform=a2.transAxes, va="top", fontsize=9)
    a2.set_facecolor("black")
    a2.tick_params(labelsize=8.5)
    _maps(fig, Bv, lead, ok, [0.37, 0.36, 0.60, 0.58], [0.37, 0.06, 0.60, 0.26],
          dot=1.6 if ok.sum() > 10000 else 7.0)           # fewer neurons (slide 5, r > 0.6): larger dots
    from matplotlib.lines import Line2D                # one legend, bottom right of the side view (no colour bar)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#ff4a4a", ms=6, label=f"leads by > {SIG:g} s"),
                        Line2D([], [], marker="o", ls="", color="#4a7bff", ms=6, label=f"trails by > {SIG:g} s")],
               loc="lower right", bbox_to_anchor=(0.98, 0.05), frameon=False, fontsize=9.5)
    return fig


def input_figure(plt, Bv, tau, ok, inp, blk, names, r_min):
    """Slide 6: the input neurons' lags against the others', per block, and on the brain."""
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    GY, OR = "0.62", "#ff9f1c"
    a1 = fig.add_axes([0.06, 0.58, 0.26, 0.35])
    bins = np.linspace(-6, 6, 97)
    a1.hist(np.clip(tau[ok & ~inp], -6, 6), bins=bins, density=True, color=GY, lw=0, alpha=0.8, label="the other neurons")
    a1.hist(np.clip(tau[ok & inp], -6, 6), bins=bins, density=True, histtype="step", color=OR, lw=1.8,
            label="the input neurons")
    a1.axvline(0, color="#2ca02c", lw=1.0)
    a1.set_xlabel("lag to the brain mean, s  (< 0 leads)", fontsize=10)
    a1.set_ylabel("share of the neurons", fontsize=10)
    a1.legend(fontsize=8.5, frameon=False, loc="center right")
    li, lo_ = tau[ok & inp], tau[ok & ~inp]
    a1.text(0.03, 0.95, f"input: median {np.median(li):+.2f} s, {100 * (li < -0.5).mean():.0f} % lead\n"
            f"others: median {np.median(lo_):+.2f} s, {100 * (lo_ < -0.5).mean():.0f} % lead", transform=a1.transAxes,
            va="top", fontsize=9)
    a2 = fig.add_axes([0.08, 0.08, 0.24, 0.36])
    ks = [k for k in range(len(names)) if ((blk == k) & ok).sum() > 5]
    data = [tau[(blk == k) & ok] for k in ks]
    bp = a2.boxplot(data, vert=False, showfliers=False, widths=0.6, patch_artist=True,
                    medianprops=dict(color="white", lw=1.6), boxprops=dict(facecolor="#5a3a10", color=OR),
                    whiskerprops=dict(color=OR), capprops=dict(color=OR))
    a2.set_yticks(range(1, len(ks) + 1))
    a2.set_yticklabels([f"{names[k]} ({int(((blk == k) & ok).sum()):,})" for k in ks], fontsize=8.5)
    a2.axvline(0, color="#2ca02c", lw=1.0)
    a2.set_xlabel("lag of the block's input neurons, s", fontsize=10)
    a2.invert_yaxis()
    sc = _maps(fig, Bv, tau, ok & inp, [0.37, 0.36, 0.60, 0.58], [0.37, 0.06, 0.60, 0.26],
               title_top="the input neurons from above, head left")
    cax = fig.add_axes([0.80, 0.33, 0.16, 0.012])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("lag, s: red leads, blue lags", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    for ax_ in (a1, a2):
        ax_.set_facecolor("black")
        ax_.tick_params(labelsize=8.5)
    return fig


def main(rec_name="zapbench_destripe"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", f"{rec_name}_recording.npz"))
    X = z["dff"]
    P = z["pos_um"].astype(np.float64)
    T, N = X.shape
    b = np.asarray(X).mean(1, dtype=np.float64)
    mv = np.asarray(X).std(0) > 1e-9
    rp, kp, _ = xcorr_peak(X, b, K)
    rp[~mv] = np.nan
    ok = mv & (rp > R_MIN)
    lag_raw, coh = phase_lag(np.asarray(X, np.float32), b)                       # s, > 0 after the brain mean
    zz = P[:, 2]
    # THE CHECK: the raw lag against depth, over the neurons that follow b
    zb_ = np.linspace(zz[ok].min(), zz[ok].max(), 13)                           # the trend of the depth-band MEDIANS
    ib_ = np.digitize(zz[ok], zb_)
    zc_ = np.array([(zb_[q - 1] + zb_[q]) / 2 for q in range(1, len(zb_)) if (ib_ == q).sum() > 100])
    md_ = np.array([np.median(lag_raw[ok][ib_ == q]) for q in range(1, len(zb_)) if (ib_ == q).sum() > 100])
    slope, icpt = np.polyfit(zc_, md_, 1)
    expect = FRAME_S / (PLANES * PLANE_UM)                                       # s per um of z
    dorsal_up = slope < 0                    # a neuron later in the scan looks EARLY against b: raw lag falls along the scan
    plane = np.clip(np.round((zz - zz.min()) / PLANE_UM), 0, PLANES - 1)
    if not dorsal_up:
        plane = PLANES - 1 - plane                                               # z grows ventrally: the scan runs down z
    s = plane / PLANES * FRAME_S
    # the correction only when the data show the scan: the depth trend of the raw medians within half of the expected
    # slope (Cedric, 2026-10-07: on the destriped traces it does not -- the zap-inr fit puts each volume on one time)
    exp_slope = -expect if dorsal_up else expect
    apply_ = abs(slope - exp_slope) < 0.5 * expect
    tau = lag_raw + (s - s[mv].mean()) if apply_ else lag_raw.copy()
    mdc_ = np.array([np.median((lag_raw + (s - s[mv].mean()))[ok][ib_ == q]) for q in range(1, len(zb_))
                     if (ib_ == q).sum() > 100])
    slope_c = np.polyfit(zc_, mdc_, 1)[0]
    doc = {"recording": rec_name, "neurons": int(N), "moving": int(mv.sum()), "r_min": R_MIN, "with_lag": int(ok.sum()),
           "max_shift_frames": K, "method": "cross-spectrum phase, periods 20-250 s, coherence-weighted",
           "depth_band_medians_raw_s": md_.tolist(), "depth_band_medians_corrected_s": mdc_.tolist(),
           "depth_band_centres_um": zc_.tolist(),
           "raw_slope_s_per_um": float(slope), "expected_slope_s_per_um": float(-expect if dorsal_up else expect),
           "z_grows_dorsally": bool(dorsal_up), "corrected_slope_s_per_um": float(slope_c),
           "slice_time_correction_applied": bool(apply_),
           "lag_raw_median_s": float(np.median(lag_raw[ok])), "lag_median_s": float(np.median(tau[ok])),
           "lag_q10_q90_s": np.percentile(tau[ok], [10, 90]).tolist(),
           "lead_share": float((tau[ok] < -SIG).mean()), "lag_share": float((tau[ok] > SIG).mean()),
           "at_shift_limit": 0}
    json.dump(doc, open(os.path.join(EXP, "data", f"brain_mean_lag_{rec_name}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(EXP, "data", f"brain_mean_lag_{rec_name}.npz"), r_peak=rp.astype(np.float32),
                        lag_raw_s=lag_raw.astype(np.float32), lag_s=tau.astype(np.float32), slice_s=s.astype(np.float32),
                        with_lag=ok)

    plt.style.use("dark_background")
    from exp17_ablation import _brain_view
    Bv = _brain_view(P)                                   # the deck's horizontal view: head left
    sel_all = ok
    # SLIDE 5 (Cedric, 2026-10-07): left the histogram over the depth profile; right the brain from above, horizontal,
    # and from the side below it; the lag-against-correlation panel dropped
    fig = lag_figure(plt, Bv, tau, sel_all, zc_, md_, exp_slope, R_MIN, "every neuron that follows the brain mean")
    out = os.path.join(EXP, "presentation", "figs", f"brain_mean_lag_{rec_name}.png")
    fig.savefig(out, dpi=130, facecolor="black")
    plt.close(fig)
    # SLIDE 6 (Cedric, 2026-10-07: "the same analysis on the input neurons: are they true leaders? maybe a better
    # population of input neurons this way"): the balanced 20 % mask's input neurons against the others, per block
    mbi = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20.npz"))["mask_by_input"] > 0
    inp = mbi.any(1)
    U = z["stimulus"]
    off, names = z["offsets"], [str(x) for x in z["names"]]
    fblk = np.array([int(np.searchsorted(off, np.flatnonzero(np.abs(U[:, j]) > 0)[0], side="right") - 1)
                     for j in range(U.shape[1])])
    blk = np.where(inp, fblk[np.argmax(mbi[:, :U.shape[1]], 1)], -1)        # each input neuron's (first) block
    li, lo_ = tau[ok & inp], tau[ok & ~inp]
    per = {names[k]: {"n_input": int((blk == k).sum()), "n_with_lag": int(((blk == k) & ok).sum()),
                      "median_s": float(np.median(tau[(blk == k) & ok])) if ((blk == k) & ok).sum() > 5 else None,
                      "lead_share": float((tau[(blk == k) & ok] < -SIG).mean()) if ((blk == k) & ok).sum() > 5 else None}
           for k in range(len(names))}
    doc["input"] = {"n_input": int(inp.sum()), "n_input_with_lag": int((ok & inp).sum()),
                    "median_input_s": float(np.median(li)), "median_other_s": float(np.median(lo_)),
                    "lead_share_input": float((li < -SIG).mean()), "lead_share_other": float((lo_ < -SIG).mean()),
                    "lag_share_input": float((li > SIG).mean()), "lag_share_other": float((lo_ > SIG).mean()),
                    "per_block": per,
                    "leaders_not_input": int((ok & ~inp & (tau < -1.0)).sum())}
    json.dump(doc, open(os.path.join(EXP, "data", f"brain_mean_lag_{rec_name}.json"), "w"), indent=1)
    # SLIDE 5 as Cedric meant it (2026-10-07: "all the other neurons against the input neurons"): the reference is the
    # input neurons' mean trace, b_in; every OTHER neuron's lag to it -- > 0 the neuron moves after the input neurons
    b_in = np.asarray(X)[:, inp].mean(1, dtype=np.float64)
    rp_in, _, _ = xcorr_peak(X, b_in, K)
    ok_in = mv & ~inp & (rp_in > R_IN)                 # Cedric, 2026-10-07: r > 0.6 for slide 5
    tau_in, _ = phase_lag(np.asarray(X, np.float32), b_in)
    ib2 = np.digitize(zz[ok_in], zb_)
    zc2 = np.array([(zb_[q - 1] + zb_[q]) / 2 for q in range(1, len(zb_)) if (ib2 == q).sum() > 100])
    md2 = np.array([np.median(tau_in[ok_in][ib2 == q]) for q in range(1, len(zb_)) if (ib2 == q).sum() > 100])
    doc["vs_input_mean"] = {"n_other_with_lag": int(ok_in.sum()), "median_s": float(np.median(tau_in[ok_in])),
                            "after_share": float((tau_in[ok_in] > SIG).mean()),
                            "before_share": float((tau_in[ok_in] < -SIG).mean()),
                            "q10_q90_s": np.percentile(tau_in[ok_in], [10, 90]).tolist(),
                            "r_input_mean_brain_mean": float(np.corrcoef(b_in, b)[0, 1]), "r_min": R_IN}
    json.dump(doc, open(os.path.join(EXP, "data", f"brain_mean_lag_{rec_name}.json"), "w"), indent=1)
    fig = lag_figure(plt, Bv, tau_in, ok_in, zc2, md2, exp_slope, R_IN, "the other neurons",
                     ref="the input neurons' mean")
    out2 = os.path.join(EXP, "presentation", "figs", f"brain_mean_lag_input_{rec_name}.png")
    fig.savefig(out2, dpi=130, facecolor="black")
    plt.close(fig)
    # THE MERGED SLIDE (Cedric, 2026-10-07: "merge slides 4 and 5"): left against the brain mean, right against the input
    # neurons -- each its histogram over the brain from above and from the side
    fig = plt.figure(figsize=(14, 8.6), facecolor="black")
    for k_, (tt_, ok_, ref_, rmin_) in enumerate(((tau, ok, "the brain mean", R_MIN),
                                                  (tau_in, ok_in, "the input neurons' mean", R_IN))):
        x0 = 0.03 + 0.50 * k_
        ld_ = -tt_
        ah = fig.add_axes([x0 + 0.05, 0.75, 0.36, 0.20])
        ah.hist(np.clip(ld_[ok_], -6, 6), bins=np.linspace(-6, 6, 97), color="0.62", lw=0)
        ah.axvspan(-SIG, SIG, color="0.25", lw=0, zorder=0)
        ah.axvline(0, color="#2ca02c", lw=1.0)
        ah.set_xlabel(f"lead over {ref_}, s  (> 0 moves first)", fontsize=9.5)
        ah.set_ylabel("neurons", fontsize=9)
        ah.text(0.02, 0.95, f"{int(ok_.sum()):,} neurons, r > {rmin_}\nmedian {np.median(ld_[ok_]):+.2f} s\n"
                f"{100 * (ld_[ok_] > SIG).mean():.0f} % lead > {SIG:g} s\n{100 * (ld_[ok_] < -SIG).mean():.0f} % trail > {SIG:g} s",
                transform=ah.transAxes, va="top", fontsize=8.5)
        ah.set_facecolor("black")
        ah.tick_params(labelsize=8)
        _maps(fig, Bv, ld_, ok_, [x0, 0.27, 0.46, 0.36], [x0, 0.04, 0.46, 0.20],
              title_top=f"against {ref_}, from above", dot=1.2 if ok_.sum() > 10000 else 5.0)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#ff4a4a", ms=6, label=f"leads by > {SIG:g} s"),
                        Line2D([], [], marker="o", ls="", color="#4a7bff", ms=6, label=f"trails by > {SIG:g} s")],
               loc="lower right", bbox_to_anchor=(0.99, 0.01), frameon=False, fontsize=9.5, ncol=2)
    fig.savefig(os.path.join(EXP, "presentation", "figs", f"brain_mean_lag_merged_{rec_name}.png"), dpi=130,
                facecolor="black")
    plt.close(fig)
    print(f"[brain mean lag] {out}\n" + json.dumps(doc, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", default="zapbench_destripe")
    main(ap.parse_args().recording)

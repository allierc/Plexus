"""exp17: WHERE does a trained law miss? The residual of its forecasts, from every frame (Cedric, 2026-10-01: "the
known ODE does not learn the high frequency and the large signal changes").

Local analysis (no cluster). For one landed run: the law forecasts 16 steps from EVERY origin of the recording; the
forecasts at h = 1 and h = 16 are kept as two [T, N] series, P_h(t) = the forecast of frame t made h frames earlier.
With x the recording and r_h = x - P_h:

  a  AMPLITUDE CAPTURE  -- the recorded change d = x(t) - x(t-h) against the forecast change p = P_h(t) - x(t-h),
     binned by |d|: captured = mean(p sign d) / mean(|d|) per bin. 1 = the law moves as far as the brain did, 0 = not at
     all. Large changes under-predicted show as captured falling with |d|.
  b  WHERE THE ERROR SITS -- the share of the total squared residual carried by the top 0.1 / 1 / 10 % of the entries
     (neuron x frame), of the frames, of the neurons.
  c  FREQUENCY -- Welch spectra over a sample of neurons of the recording, the forecast and the residual, and the
     captured gain |S_xp| / S_xx per frequency (1: the law reproduces that frequency's amplitude, 0: none of it).
  d  MAPS -- per neuron, the long residual's MSE over its own variance (the unexplained fraction), and the recording's
     kurtosis (how bursty the neuron is); their correlation across neurons.
  e  TIME -- the residual MSE per frame over the recording, with the brain-mean |change|, the conditions as bands.

Writes experiments/exp17_zapbench_graphcast/data/residuals_<run>.json and data/figs/residuals_<run>.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FRAME_S = 0.914
HS = (1, 16)


def forecasts(name, device):
    """x [T, N] (float32, device) and {h: P_h [T, N] float16 on cpu, NaN where no origin}, the condition per frame."""
    import plexus.trainer as TRN
    from plexus import engine
    spec = TRN.load(name)
    engine.quiet(True)
    out = TRN.out_dir(spec, None)
    ck = torch.load(os.path.join(out, "models", "best.pt"), weights_only=False, map_location=device)
    learn = TRN.Learnables(spec["learnable"], device)
    learn.restore(ck["fitted"])
    box = TRN._trace_setup(spec, device)
    X, T, n_in = box["X"], box["T"], box["n_in"]
    K = max(HS)
    sim = TRN._model(spec, train=False, n_frames=K - 1)
    P = {h: np.full((T, X.shape[1]), np.nan, np.float16) for h in HS}
    origins = range(max(n_in - 1, 5), T - K - 1)
    t0 = time.time()
    with torch.no_grad():
        for i, o in enumerate(origins):
            p = TRN._trace_rollout(sim, learn, spec, box, int(o), K, device, False)
            for h in HS:
                P[h][o + h] = p[h - 1].half().cpu().numpy()
            if i % 1000 == 0:
                print(f"[residuals] origin {i}/{len(origins)}  {time.time() - t0:.0f} s", flush=True)
    return X, P, box["rec"]


def capture(X, Ph, h, rng, n=20_000_000):
    """Captured fraction of the recorded change, binned by its size (quantiles of |d|)."""
    T, N = X.shape
    t = rng.integers(h, T, n)
    i = rng.integers(0, N, n)
    ph = torch.as_tensor(Ph[t, i].astype(np.float32))
    ok = torch.isfinite(ph)
    tt, ii = torch.as_tensor(t)[ok], torch.as_tensor(i)[ok]
    Xc = X.cpu()
    d = Xc[tt, ii] - Xc[tt - h, ii]
    p = ph[ok] - Xc[tt - h, ii]
    a = d.abs()
    qs = [0, 50, 75, 90, 95, 99, 99.9, 100]
    edges = np.percentile(a.numpy(), qs)
    out = []
    for k in range(len(qs) - 1):
        m = (a >= edges[k]) & (a <= edges[k + 1])
        out.append({"pct": f"{qs[k]}-{qs[k + 1]}", "abs_change_mean": float(a[m].mean()),
                    "captured": float((p[m] * torch.sign(d[m])).mean() / a[m].mean())})
    return out


def shares(R2):
    """Shares of the total squared residual in the top 0.1 / 1 / 10 % of entries, frames, neurons."""
    flat = np.sort(R2[np.isfinite(R2)].ravel())[::-1]
    tot = flat.sum()
    out = {f"entries_top_{q}pct": float(flat[:max(1, int(len(flat) * q / 100))].sum() / tot) for q in (0.1, 1, 10)}
    fr = np.nansum(R2, 1)
    fr = np.sort(fr[fr > 0])[::-1]
    out.update({f"frames_top_{q}pct": float(fr[:max(1, int(len(fr) * q / 100))].sum() / fr.sum()) for q in (1, 10)})
    nu = np.sort(np.nansum(R2, 0))[::-1]
    out.update({f"neurons_top_{q}pct": float(nu[:max(1, int(len(nu) * q / 100))].sum() / nu.sum()) for q in (1, 10)})
    return out


def spectra(X, Ph, rng, n_neurons=4000):
    from scipy.signal import csd, welch
    valid = np.where(np.isfinite(Ph[:, 0].astype(np.float32)))[0]
    t0, t1 = valid.min(), valid.max() + 1
    idx = rng.choice(X.shape[1], n_neurons, replace=False)
    x = X[t0:t1][:, torch.as_tensor(idx, device=X.device)].cpu().numpy().astype(np.float64)
    p = Ph[t0:t1][:, idx].astype(np.float64)
    x = x - x.mean(0)
    p = p - p.mean(0)
    f, sxx = welch(x, fs=1 / FRAME_S, nperseg=256, axis=0)
    _, spp = welch(p, fs=1 / FRAME_S, nperseg=256, axis=0)
    _, srr = welch(x - p, fs=1 / FRAME_S, nperseg=256, axis=0)
    _, sxp = csd(x, p, fs=1 / FRAME_S, nperseg=256, axis=0)
    return {"f_hz": f.tolist(), "psd_rec": sxx.mean(1).tolist(), "psd_pred": spp.mean(1).tolist(),
            "psd_res": srr.mean(1).tolist(), "gain": (np.abs(sxp).mean(1) / sxx.mean(1)).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="zap_ng_wide")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    X, P, rec = forecasts(a.run, a.device)
    res = {"run": a.run, "horizons": list(HS)}
    Xc = X.cpu().numpy()
    var = Xc.var(0)
    for h in HS:
        R2 = (Xc - P[h].astype(np.float32)) ** 2
        r = {"capture": capture(X, P[h], h, rng), "shares": shares(R2)}
        mse_n = np.nanmean(R2, 0)
        r["unexplained_median"] = float(np.median(mse_n / var))
        # PEAKS (Cedric, 2026-10-01: can the law reach the large signals?): per neuron, the forecast's 99th-percentile
        # excursion above the neuron's recorded mean over the recording's own; and the forecast's sd over the
        # recording's. 1 = the forecasts swing as far as the brain does, <1 = shrunk toward the mean
        Pf = P[h].astype(np.float32)
        ok = ~np.isnan(Pf).any(1)
        mu = Xc[ok].mean(0)
        pk = (np.percentile(Pf[ok] - mu, 99, axis=0) / np.maximum(np.percentile(Xc[ok] - mu, 99, axis=0), 1e-6))
        sd = Pf[ok].std(0) / np.maximum(Xc[ok].std(0), 1e-6)
        r["peak_ratio_p99"] = {"median": float(np.median(pk)), "p10": float(np.percentile(pk, 10)),
                               "p90": float(np.percentile(pk, 90))}
        r["sd_ratio"] = {"median": float(np.median(sd)), "p10": float(np.percentile(sd, 10)),
                         "p90": float(np.percentile(sd, 90))}
        del Pf
        if h == max(HS):
            from scipy.stats import kurtosis, spearmanr
            kur = kurtosis(Xc, axis=0)
            res["map_unexplained"] = (mse_n / var).astype(np.float32)
            res["map_kurtosis"] = kur.astype(np.float32)
            r["spearman_unexplained_vs_kurtosis"] = float(spearmanr(mse_n / var, kur).correlation)
            res["per_frame_mse"] = np.nanmean(R2, 1)
            res["spectra"] = spectra(X, P[h], rng)
        res[f"h{h}"] = r
        del R2
        print(f"[residuals] h {h}: captured by size {[(c['pct'], round(c['captured'], 2)) for c in r['capture']]}; "
              f"shares {r['shares']}", flush=True)
    res["brain_mean_abs_change"] = np.abs(np.diff(Xc, axis=0)).mean(1)
    figure(res, rec, os.path.join(EXP, "data", "figs", f"residuals_{a.run}.png"))
    js = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    json.dump(js, open(os.path.join(EXP, "data", f"residuals_{a.run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in js.items() if k != "spectra"}, indent=1))


def figure(res, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = rec["pos_um"]
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.4), facecolor="black")

    def tidy(ax):
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
    a = fig.add_axes([0.05, 0.57, 0.27, 0.35])
    for h, col in zip(HS, ("#e07b39", "#7aa6ff")):
        c = res[f"h{h}"]["capture"]
        a.plot([x["abs_change_mean"] for x in c], [x["captured"] for x in c], "o-", color=col, label=f"h = {h}")
    a.set_xscale("log")
    a.axhline(1, color="0.5", ls=":", lw=1)
    a.axhline(0, color="0.5", lw=0.7)
    a.set_xlabel("size of the recorded change |x(t) - x(t-h)|, dF/F (bins: percentiles 0-50 ... 99.9-100)")
    a.set_ylabel("captured: forecast change / recorded change")
    a.legend(frameon=False, fontsize=8)
    tidy(a)
    fig.text(0.05, 0.965, "a   how much of a change the law reproduces, by the change's size", fontsize=11)
    b = fig.add_axes([0.40, 0.57, 0.25, 0.35])
    sp = res["spectra"]
    f = np.asarray(sp["f_hz"])[1:]
    b.loglog(f, np.asarray(sp["psd_rec"])[1:], color="#2ca02c", lw=1.6, label="recording")
    b.loglog(f, np.asarray(sp["psd_pred"])[1:], color="white", lw=1.4, label="forecast, h 16")
    b.loglog(f, np.asarray(sp["psd_res"])[1:], color="#d62728", lw=1.2, label="residual")
    b.set_xlabel("frequency, Hz (Nyquist 0.55)")
    b.set_ylabel("power, (dF/F)$^2$/Hz (4,000 neurons)")
    b.legend(frameon=False, fontsize=8)
    tidy(b)
    fig.text(0.40, 0.965, "b   spectra: which frequencies the forecast carries", fontsize=11)
    order = np.argsort(P[:, 2])
    for j, (key, lab, cm, vm) in enumerate((("map_unexplained", "c   unexplained fraction, h 16", "magma", (0, 1.2)),
                                            ("map_kurtosis", "d   kurtosis of the recording", "viridis", None))):
        ax = fig.add_axes([0.70 + 0.15 * j, 0.50, 0.14, 0.42])
        ax.set_facecolor("black")
        ax.axis("off")
        v = np.asarray(res[key])[order]
        vmin, vmax = vm if vm else (np.percentile(v, 2), np.percentile(v, 98))
        ax.scatter(P[order, 0], P[order, 1], c=v, s=0.15, cmap=cm, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        fig.text(0.70 + 0.15 * j, 0.965, lab, fontsize=10)
    fig.text(0.70, 0.47, f"Spearman(unexplained, kurtosis) {res['h16']['spearman_unexplained_vs_kurtosis']:+.2f}",
             fontsize=9, color="0.8")
    e = fig.add_axes([0.05, 0.08, 0.92, 0.3])
    t = np.arange(len(res["per_frame_mse"])) * FRAME_S / 60
    off = rec["offsets"]
    cmap = plt.get_cmap("tab10")
    for ci in range(len(off) - 1):
        e.axvspan(off[ci] * FRAME_S / 60, off[ci + 1] * FRAME_S / 60, color=cmap(ci), alpha=0.13, lw=0)
        e.text((off[ci] + off[ci + 1]) / 2 * FRAME_S / 60, 1.02, rec["names"][ci], fontsize=7, ha="center",
               transform=e.get_xaxis_transform())
    e.plot(t, 1e3 * np.asarray(res["per_frame_mse"]), color="#d62728", lw=0.6, label="residual MSE per frame, h 16")
    e2 = e.twinx()
    e2.plot(t[1:], np.asarray(res["brain_mean_abs_change"]), color="#2ca02c", lw=0.5, alpha=0.8,
            label="brain-mean |frame-to-frame change|")
    e.set_xlabel("time, min")
    e.set_ylabel("residual MSE, $10^{-3}$ dF/F$^2$", color="#d62728")
    e2.set_ylabel("mean |change|, dF/F", color="#2ca02c")
    tidy(e)
    fig.text(0.05, 0.42, "e   when the error happens, against how much the brain changes", fontsize=11)
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    plt.style.use("default")


if __name__ == "__main__":
    main()

"""exp17: do the INPUT NEURONS follow the stimulus at its own frequencies? (Cedric, 2026-10-01)

Local analysis. For every neuron, the magnitude-squared coherence (Welch, 256-frame segments, 0.914 s frames) between
its RECORDED dF/F and its own stimulus drive d_i(t) = B_i . u(t) -- B_i the 22 stimulus weights the law learned
(`--run`, default zap_zs_ng_base, the run the input mask was cut from), u(t) the release's 22 stimulus features.
Coherence is 0..1 per frequency: 1 when the two move together with a fixed phase at that frequency.

  a  the stimulus's power spectrum (the 22 features, summed): which frequencies the visual world holds
  b  the mean coherence per frequency, the input neurons (graphs_data/zebrafish/input_mask_top10.npz) against the rest
  c  per neuron, the coherence averaged over the stimulus's strongest band (its top-power frequencies), on the brain
  d  its distribution, input neurons against the rest

Writes experiments/exp17_zapbench_graphcast/data/input_coherence_<run>.json and data/figs/input_coherence_<run>.png.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
FS = 1 / 0.914


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="zap_zs_ng_base")
    ap.add_argument("--chunk", type=int, default=6000)
    a = ap.parse_args()
    from scipy.signal import coherence, welch
    from plexus.paths import graphs_data_path
    from plexus.tasks import trace_recording as TR
    rec = TR.load("zapbench")
    X = rec["dff"].astype(np.float32)                                   # [T, N]
    U = rec["stimulus"].astype(np.float32)                              # [T, 22]
    f_ = torch.load(os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"),
                                 "log", "training", "zapbench", a.run, "models", "best.pt"),
                    weights_only=False, map_location="cpu")["fitted"]
    B = f_["neuron.input"].float().numpy()                              # [N, 22]
    mask = np.load(graphs_data_path("zebrafish", "input_mask_top10.npz"))["mask"] > 0
    T, N = X.shape
    fu, pu = welch(U - U.mean(0), fs=FS, nperseg=256, axis=0)
    pu = pu.sum(1)
    band = np.argsort(pu[1:])[::-1][:6] + 1                             # the stimulus's 6 strongest frequencies
    coh_mean = {"input": np.zeros(len(fu)), "rest": np.zeros(len(fu))}
    cnt = {"input": 0, "rest": 0}
    coh_band = np.zeros(N, np.float32)
    for c0 in range(0, N, a.chunk):
        c1 = min(N, c0 + a.chunk)
        x = X[:, c0:c1] - X[:, c0:c1].mean(0)
        d = U @ B[c0:c1].T                                              # [T, n] each neuron's own drive
        d = d - d.mean(0)
        f, C = coherence(x, d, fs=FS, nperseg=256, axis=0)              # [F, n]
        C = np.nan_to_num(C)
        coh_band[c0:c1] = C[band].mean(0)
        for key, m in (("input", mask[c0:c1]), ("rest", ~mask[c0:c1])):
            coh_mean[key] += C[:, m].sum(1)
            cnt[key] += int(m.sum())
        print(f"[coherence] neurons {c1}/{N}", flush=True)
    for k in coh_mean:
        coh_mean[k] /= max(cnt[k], 1)
    out = {"run": a.run, "neurons": int(N), "input_neurons": int(mask.sum()),
           "stimulus_band_hz": fu[band].round(4).tolist(),
           "stimulus_band_periods_s": (1 / fu[band]).round(1).tolist(),
           "coherence_in_band": {"input_median": float(np.median(coh_band[mask])),
                                 "rest_median": float(np.median(coh_band[~mask])),
                                 "input_p90": float(np.percentile(coh_band[mask], 90)),
                                 "rest_p90": float(np.percentile(coh_band[~mask], 90))},
           "f_hz": fu.tolist(), "stim_psd": pu.tolist(),
           "coh_input": coh_mean["input"].tolist(), "coh_rest": coh_mean["rest"].tolist()}
    json.dump(out, open(os.path.join(EXP, "data", f"input_coherence_{a.run}.json"), "w"), indent=1)
    np.save(os.path.join(EXP, "data", f"input_coherence_{a.run}_band.npy"), coh_band)   # per neuron, for masks
    figure(out, coh_band, mask, rec["pos_um"], os.path.join(EXP, "data", "figs", f"input_coherence_{a.run}.png"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("f_hz", "stim_psd", "coh_input", "coh_rest")}, indent=1))


def figure(out, coh_band, mask, P, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 6.0), facecolor="black")
    f = np.asarray(out["f_hz"])
    a = fig.add_axes([0.05, 0.12, 0.25, 0.70])
    a.semilogy(f[1:], np.asarray(out["stim_psd"])[1:], color="#7aa6ff", lw=1.4)
    for fb in out["stimulus_band_hz"]:
        a.axvline(fb, color="0.5", lw=0.6, ls=":")
    a.set_xlabel("frequency, Hz (Nyquist 0.55)")
    a.set_ylabel("stimulus power (22 features, summed)")
    fig.text(0.05, 0.95, "a   the stimulus's spectrum (dotted: its 6 strongest frequencies)", fontsize=10, va="top")
    b = fig.add_axes([0.37, 0.12, 0.25, 0.70])
    b.plot(f[1:], np.asarray(out["coh_input"])[1:], color="#ff7f0e", lw=1.6, label=f"input neurons ({out['input_neurons']:,})")
    b.plot(f[1:], np.asarray(out["coh_rest"])[1:], color="0.7", lw=1.2, label="the other neurons")
    for fb in out["stimulus_band_hz"]:
        b.axvline(fb, color="0.5", lw=0.6, ls=":")
    b.set_xlabel("frequency, Hz")
    b.set_ylabel("coherence, recorded dF/F vs own drive B_i . u")
    b.legend(frameon=False, fontsize=8)
    fig.text(0.37, 0.95, "b   mean coherence per frequency", fontsize=10, va="top")
    c = fig.add_axes([0.66, 0.08, 0.15, 0.80])
    c.set_facecolor("black")
    c.axis("off")
    o = np.argsort(P[:, 2])
    sc = c.scatter(P[o, 0], P[o, 1], c=coh_band[o], s=0.15, cmap="magma", vmin=0,
                   vmax=float(np.percentile(coh_band, 99)), linewidths=0)
    c.set_aspect("equal")
    fig.text(0.66, 0.95, "c   coherence in the stimulus band", fontsize=10, va="top")
    cb = fig.add_axes([0.67, 0.06, 0.12, 0.015])
    fig.colorbar(sc, cax=cb, orientation="horizontal").ax.tick_params(labelsize=7)
    d = fig.add_axes([0.85, 0.12, 0.13, 0.70])
    bins = np.linspace(0, float(np.percentile(coh_band, 99.5)), 50)
    d.hist(coh_band[~mask], bins, color="0.6", density=True, histtype="step", lw=1.2, label="others")
    d.hist(coh_band[mask], bins, color="#ff7f0e", density=True, histtype="step", lw=1.6, label="input")
    d.set_xlabel("coherence in the stimulus band")
    d.legend(frameon=False, fontsize=8)
    fig.text(0.85, 0.95, "d   distribution", fontsize=10, va="top")
    for ax in (a, b, d):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    plt.style.use("default")


if __name__ == "__main__":
    main()

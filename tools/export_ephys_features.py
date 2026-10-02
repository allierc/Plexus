#!/usr/bin/env python
"""exp17: the fish's OWN SWIMMING (fictive, tail-nerve ephys) as extra stimulus features (Cedric, 2026-10-02).

    cd /workspace/Plexus; export PYTHONPATH=src:tools GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData
    python tools/export_ephys_features.py            # local only (the cluster rule)

THE SOURCE. ZAPBench's raw ephys file stimuli_and_ephys.10chFlt (release volumes/20240930/stimuli_raw): 10 float32
channels interleaved at 6 kHz over ~7,201 s. Channels 0 and 1 are the left and right motor-nerve recordings (fictive
swimming), 2 the imaging TTL, 3 / 6 the stimulus parameters stimParam4 / stimParam3, 4 the trial (condition) id, 8 the
grating velocity. Swim and turn power are computed by fishFuncEM's FishEphys.compute_behavior (Bennett's swim detector:
40 ms windowed variance, bout detection, drift removal, 10-min sliding normalisation; swim power weights each bout by
a rising exponential, turn power by a decaying one). The 6 kHz behaviour is cached beside the zarr.

THE ALIGNMENT, imaging frame <-> ephys samples. zapbench.zarr's plane_time_ephys [7,872 volumes, 74] holds the ephys
sample of every imaging plane of each volume, read from the TTL channel. The hypothesis is release frame t == marker
volume t for t < 7,872, the release's 7 last frames having no ephys (the fishFuncEM notebook says so; the ephys file
ends 0.68 s after volume 7,871 starts, at its 57th of 72 planes). Checked here, three ways:
  (a) the condition onsets from the ephys trial-id channel, mapped to volumes, against the release's CONDITION_OFFSETS;
  (b) the volume TTL peaks found here (FishEphys) against plane_time_ephys's first plane;
  (c) exact: each of the release's 10 stimulus features that change inside their condition, read as a function of
      the raw stimulus channels sampled at the volume's TTL peak, at lags -3..3 frames: 1 mismatched frame of 9,515
      at lag 0 (a rotation reversal), ~1 per stimulus change at lag +-1 (206 frames on average); softer, the
      channels averaged over each frame correlated with the features at lags -8..8 (peak at 0 or -1, r 0.98-0.99).

THE FEATURES, one value per imaging frame: the mean of the 6 kHz signal over the frame's 0.914 s, from the first plane
of volume t to the first plane of volume t+1 (5,484 samples; the file ends 4,086 samples = 0.68 s into the last volume, which is averaged over those).
  swim_L, swim_R   swim power of the left / right motor channel (FishEphys swim_power0 / swim_power1)
  turn_L, turn_R   turn power of the left / right motor channel (turn_power0 / turn_power1)
  bout             the FRACTION OF THE FRAME'S 0.914 s spent inside a detected swim bout on either channel (0..1)
Each power feature is divided by its own 99th percentile over the 7,872 frames with ephys and clipped to [0, 1.5]
(the 22 release features live in [-1, 1]); bout is already a fraction in [0, 1] and is not rescaled. The release's 7
last frames carry no ephys: their ephys features are 0 and `ephys_valid` marks them False.

Writes graphs_data/zebrafish/zapbench_destripe_ephys_recording.npz (+ .json provenance): zapbench_destripe_recording.npz
with `stimulus` [7879, 22 + 5] and the new arrays stimulus_names [27] and ephys_valid [7879]; and the figure
experiments/exp17_zapbench_graphcast/data/figs/ephys_features.png (copied to png/).
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
from plexus.paths import graphs_data_path                              # noqa: E402

FISHFUNCEM = "/workspace/connectome-gnn-cx/papers/fishFuncEM"
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
ZB = graphs_data_path("zebrafish", "zapbench")
RAW = os.path.join(ZB, "release_20240930", "volumes", "20240930", "stimuli_raw", "stimuli_and_ephys.10chFlt")
ZARR = os.path.join(ZB, "zapbench.zarr")
CACHE = os.path.join(ZB, "ephys_behavior_6khz.npz")
SRC_REC = graphs_data_path("zebrafish", "zapbench_destripe_recording.npz")
OUT_REC = graphs_data_path("zebrafish", "zapbench_destripe_ephys_recording.npz")
FS, PERIOD = 6000, 5484                       # ephys samples per second; per imaging frame (0.914 s)
EPHYS_NAMES = ["swim_L", "swim_R", "turn_L", "turn_R", "bout"]
STIM_CH = {3: "stimParam4", 6: "stimParam3", 8: "grating velocity", 9: "ch9 (gain)"}
LAGS = np.arange(-8, 9)
LAGS_EXACT = np.arange(-3, 4)


def behaviour():
    """6 kHz swim / turn power per motor channel, the in-bout mask, and the channels used for alignment."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    sys.path.insert(0, FISHFUNCEM)
    from fishfuncem.functional.FishEphys import FishEphys
    t0 = time.time()
    e = FishEphys.from_file(RAW, compute_behavior=True)
    print(f"[ephys] {e!r} in {time.time() - t0:.0f} s")
    d = dict(swim_power0=e.swim_power0.astype(np.float32), swim_power1=e.swim_power1.astype(np.float32),
             turn_power0=e.turn_power0.astype(np.float32), turn_power1=e.turn_power1.astype(np.float32),
             imaging_peaks=np.asarray(e.imaging_peaks), trial_onsets_ephys=np.asarray(e.trial_onsets_ephys),
             trial_onsets_img=np.asarray(e.trial_onsets_img), n_samples=np.int64(e.n_samples))
    for c in STIM_CH:
        d[f"ch{c}"] = np.asarray(e.raw[c], np.float32)
    d["trial_id"] = np.asarray(e.raw[4], np.float32)
    np.savez(CACHE, **d)
    return d


def bin_mean(x, starts, stops):
    """[V]: the mean of x over samples starts[v] .. stops[v] - 1."""
    cs = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64))])
    return (cs[stops] - cs[starts]) / (stops - starts)


def change_frames(v):
    return np.where(np.abs(np.diff(v)) > 1e-6)[0] + 1


def main():
    import zarr
    B = behaviour()
    z = zarr.open(ZARR, mode="r")
    pte = z["plane_time_ephys"][:]
    on_ephys, on_img = z["onsets_ephys"][:], z["onsets_img"][:]
    V = pte.shape[0]
    starts = pte[:, 0].astype(np.int64)
    stops = np.append(starts[1:], starts[-1] + PERIOD)
    n = int(B["n_samples"])
    stops[-1] = min(stops[-1], n)             # the file stops 0.68 s into the last volume (57 of its 72 planes)
    rec = np.load(SRC_REC)
    S, cond, offsets = rec["stimulus"], rec["condition"], rec["offsets"]
    T = S.shape[0]

    # ---------------------------------------------------------------- alignment evidence
    ev = {}
    hp = B["imaging_peaks"]
    ev["volume_ttl_peaks_found"] = int(len(hp))
    ev["marker_volumes"] = int(V)
    ev["release_frames"] = int(T)
    k = np.searchsorted(hp, starts, side="right") - 1               # each volume's nearest TTL volume peak before it
    d_tt = starts - hp[np.clip(k, 0, len(hp) - 1)]
    ev["first_plane_minus_its_volume_ttl_peak_samples"] = {"median": float(np.median(d_tt)), "min": int(d_tt.min()),
                                                           "max": int(d_tt.max())}
    ev["volume_ttl_peak_index_minus_volume_index"] = np.unique(k - np.arange(V)).tolist()
    tr = B["trial_id"]
    trial_on = np.where(np.diff(tr) != 0)[0] + 1
    trial_on = np.concatenate([[0], trial_on]) if trial_on[0] > PERIOD else trial_on
    ev["trial_id_changes_ephys_sample"] = trial_on.tolist()
    ev["zarr_onsets_ephys"] = on_ephys.tolist()
    ev["zarr_onsets_img"] = on_img.tolist()
    ev["release_offsets"] = offsets.tolist()
    # a condition onset falls in the volume whose window holds it
    vol_of = np.clip(np.searchsorted(starts, on_ephys, side="right") - 1, 0, V - 1)
    ev["zarr_onsets_ephys_to_volume"] = vol_of.tolist()
    ev["onsets_minus_release_offsets_frames"] = (vol_of - offsets[:9]).tolist()

    # (c1) THE EXACT TEST. Each release feature is read as a function of the raw stimulus channels sampled at ONE
    # instant of the volume, the volume's TTL peak: for every distinct sampled value (or pair of values) the release
    # feature takes its most common value, and every frame that disagrees is a mismatch. Done at lags L = -3..3
    # (release frame t against TTL peak t + L): a correct alignment gives 0 mismatched frames at L = 0 and about
    # one mismatch per stimulus change at L = +-1.
    varying = [j for j in range(S.shape[1]) if S[np.abs(S[:, j]) > 0, j].std() > 1e-6]
    keys = {0: [6], 2: [3], 4: [6], 6: [6], 7: [3], 9: [3, 6], 10: [3, 6], 11: [3, 6], 16: [3], 19: ["rot"]}
    exact = []
    for j, chs in keys.items():
        c = int(np.bincount(cond[np.abs(S[:, j]) > 0]).argmax())
        fr = np.arange(offsets[c] + 3, min(offsets[c + 1], V) - 3)
        if chs == ["rot"]:                       # rotation: the SIGN of the heading's change (stimParam3, wraps at 90)
            a0, a1 = hp[fr[0]] - 60000, hp[fr[-1]] + 60000
            hd = np.unwrap(B["ch6"][a0:a1].astype(np.float64)[::60], period=90.0)     # 100 Hz
        mm = []
        for L in LAGS_EXACT:
            idx = hp[fr + L]
            if chs == ["rot"]:
                i = (idx - a0) // 60
                key = np.sign(hd[i + 5] - hd[i - 5])[:, None]
            else:
                key = np.stack([np.round(B[f"ch{ch}"][idx], 2) for ch in chs], 1)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv, y, m = inv.ravel(), S[fr, j], 0
            for u in np.unique(inv):
                _, cnt = np.unique(y[inv == u], return_counts=True)
                m += int(cnt.sum() - cnt.max())
            mm.append(m)
        src = "sign of d(stimParam3)/dt" if chs == ["rot"] else " + ".join(STIM_CH[ch] for ch in chs)
        exact.append(dict(feature=j, condition=str(rec["names"][c]), channels=src, frames=int(len(fr)),
                          mismatched_by_lag=dict(zip([int(L) for L in LAGS_EXACT], mm))))
        print(f"[align] f{j:2d} ({exact[-1]['condition']:9s}) from {src:26s}: mismatched frames at lag "
              f"{dict(zip(LAGS_EXACT.tolist(), mm))} of {len(fr)}")
    ev["exact_test"] = exact
    ev["exact_test_rule"] = ("release feature at frame t == a function of the stimulus channels sampled at volume t's "
                             "TTL peak (75 samples = 12.5 ms before its first plane); mismatches = frames disagreeing "
                             "with the majority value per sampled key")
    # (c2) THE SOFT TEST. The stimulus channels averaged over each volume's 0.914 s, correlated with the release
    # feature inside its condition at lags -8..8 (a frame-mean smears a change over two frames, so the peak is broad).
    Bin = {c: bin_mean(B[f"ch{c}"], starts, stops) for c in STIM_CH}
    xc = []
    for j in varying:
        on = np.where(np.abs(S[:, j]) > 0)[0]
        c = int(np.bincount(cond[on]).argmax())
        fr = np.arange(offsets[c], min(offsets[c + 1], V))
        best = None
        for ch, b in Bin.items():
            r = []
            for L in LAGS:
                t = fr[(fr + L >= offsets[c]) & (fr + L < min(offsets[c + 1], V))]
                x, y = S[t, j].astype(np.float64), b[t + L]
                r.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0)
            r = np.nan_to_num(np.array(r))
            i = int(np.abs(r).argmax())
            if best is None or abs(r[i]) > abs(best["r_best"]):
                best = dict(feature=j, condition=str(rec["names"][c]), channel=STIM_CH[ch], ch=ch,
                            best_lag_frames=int(LAGS[i]), r_best=float(r[i]), r_lag0=float(r[LAGS == 0][0]),
                            r_by_lag=np.round(r, 4).tolist())
        xc.append(best)
        print(f"[align] f{j:2d} ({best['condition']:9s}) ~ {best['channel']:16s} frame mean: r(lag 0) "
              f"{best['r_lag0']:+.3f}, best lag {best['best_lag_frames']:+d} (r {best['r_best']:+.3f})")
    ev["stimulus_xcorr_frame_mean"] = xc
    # the release frames hold their condition: does the per-volume trial id agree frame by frame?
    tid = np.round(bin_mean(tr, starts, stops), 3)
    vol_tid_mode = {int(c): np.unique(tid[offsets[c]:min(offsets[c + 1], V)], return_counts=True) for c in range(9)}
    ev["trial_id_per_condition"] = {str(rec["names"][c]): {str(u): int(m) for u, m in zip(*vol_tid_mode[c])}
                                    for c in range(9)}

    # ---------------------------------------------------------------- ephys features per frame
    inbout = ((B["swim_power0"] != 0) | (B["swim_power1"] != 0)).astype(np.float32)
    raw = np.stack([bin_mean(B["swim_power0"], starts, stops), bin_mean(B["swim_power1"], starts, stops),
                    bin_mean(B["turn_power0"], starts, stops), bin_mean(B["turn_power1"], starts, stops),
                    bin_mean(inbout, starts, stops)], 1)                 # [V, 5]
    scale = {}
    E = np.zeros((T, len(EPHYS_NAMES)), np.float32)
    for i, nm in enumerate(EPHYS_NAMES):
        x = raw[:, i]
        if nm == "bout":
            scale[nm] = 1.0
            E[:V, i] = x
        else:
            p99 = float(np.percentile(x, 99))
            scale[nm] = p99
            E[:V, i] = np.clip(x / p99, 0.0, 1.5)
    valid = np.zeros(T, bool)
    valid[:V] = True
    swim_frac = {}
    for c in range(9):
        fr = np.arange(offsets[c], min(offsets[c + 1], V))
        swim_frac[str(rec["names"][c])] = {
            "frames_with_a_bout_pct": round(100 * float((E[fr, 4] > 0).mean()), 1),
            "time_in_bouts_pct": round(100 * float(E[fr, 4].mean()), 2),
            "bouts_per_min": None}
    # bouts per minute: bout onsets (rising edges of the 6 kHz in-bout mask) per condition
    rise = np.where(np.diff(inbout) > 0)[0] + 1
    rv = np.clip(np.searchsorted(starts, rise, side="right") - 1, 0, V - 1)
    for c in range(9):
        lo, hi = offsets[c], min(offsets[c + 1], V)
        swim_frac[str(rec["names"][c])]["bouts_per_min"] = round(float(((rv >= lo) & (rv < hi)).sum())
                                                                 / ((hi - lo) * 0.914 / 60), 2)
    corr = np.corrcoef(E[:V].T)
    print("[ephys] swims per condition:")
    for k_, v in swim_frac.items():
        print(f"   {k_:9s} {v}")
    print("[ephys] scale (99th pct of the per-frame mean, raw FishEphys units):", scale)
    print("[ephys] correlation between the 5 features over the 7,872 frames:\n", np.round(corr, 2))

    # ---------------------------------------------------------------- the new recording
    names = []
    for j in range(S.shape[1]):
        on = np.abs(S[:, j]) > 0
        c = int(np.bincount(cond[on]).argmax())
        names.append(f"f{j} ({rec['names'][c]}{', varies' if j in varying else ', constant when on'})")
    names += [f"ephys {nm}" for nm in EPHYS_NAMES]
    out = {k: rec[k] for k in rec.files}
    out["stimulus"] = np.concatenate([S, E], 1).astype(np.float32)
    out["stimulus_names"] = np.array(names)
    out["ephys_valid"] = valid
    tmp = OUT_REC + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, OUT_REC)
    chk = np.load(OUT_REC)
    same = {k: bool(np.array_equal(chk[k], rec[k])) for k in rec.files if k != "stimulus"}
    same["stimulus[:, :22]"] = bool(np.array_equal(chk["stimulus"][:, :S.shape[1]], S))
    print("[recording]", OUT_REC, chk["stimulus"].shape, "bit-identical:", same)
    prov = {
        "written": time.strftime("%Y-%m-%d %H:%M"), "script": "tools/export_ephys_features.py",
        "base_recording": SRC_REC, "ephys_file": RAW,
        "ephys_sha256_first_64MB": hashlib.sha256(open(RAW, "rb").read(1 << 26)).hexdigest(),
        "timing": ZARR + " plane_time_ephys", "behaviour": FISHFUNCEM + "/fishfuncem/functional/FishEphys.py "
        "FishEphys.from_file(compute_behavior=True), defaults", "behaviour_cache": CACHE,
        "stimulus_shape": list(out["stimulus"].shape), "stimulus_names": names,
        "frame_window": "ephys samples plane_time_ephys[t, 0] .. plane_time_ephys[t + 1, 0] (5,484 = 0.914 s); "
                        "the last volume (7,871) only to the file's end, 4,086 samples = 0.68 s",
        "frame_to_volume": "release frame t == marker volume t for t < 7,872; frames 7,872..7,878 (the last 7 of "
                           "'dark') have no ephys: features 0, ephys_valid False",
        "scaling": {nm: (f"divided by {scale[nm]:.6g} (its 99th percentile over the 7,872 frames with ephys), "
                         "clipped to [0, 1.5]" if nm != "bout" else
                         "fraction of the frame's 0.914 s inside a swim bout on either motor channel, 0..1, unscaled")
                    for nm in EPHYS_NAMES},
        "feature_correlation": np.round(corr, 3).tolist(), "swims_per_condition": swim_frac,
        "bit_identical_to_base": same, "alignment": ev}
    json.dump(prov, open(OUT_REC.replace(".npz", ".json"), "w"), indent=1)
    Ttl = {c: B[f"ch{c}"][hp] for c in STIM_CH}
    figure(E, valid, S, cond, offsets, rec["names"], Bin, Ttl, ev, swim_frac, V)


def figure(E, valid, S, cond, offsets, cnames, Bin, Ttl, ev, swim_frac, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    T = len(E)
    t_min = np.arange(T) * 0.914 / 60
    cols = ["#4c9be8", "#e8834c", "#4c9be8", "#e8834c", "white"]
    fig = plt.figure(figsize=(16, 13.5), facecolor="black")
    gs = fig.add_gridspec(9, 3, hspace=1.25, wspace=0.25, left=0.05, right=0.98, top=0.95, bottom=0.04)
    blk = ["#202020", "#333333"]

    def blocks(ax, label=False):
        for c in range(9):
            a, b = t_min[offsets[c]], offsets[c + 1] * 0.914 / 60
            ax.axvspan(a, b, color=blk[c % 2], lw=0)
            if label:
                ax.text((a + b) / 2, 1.02, str(cnames[c]), transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8, color="0.85")
        ax.set_xlim(0, T * 0.914 / 60)
    titles = ["swim power of the LEFT motor channel: its mean over each 0.914 s frame / its 99th percentile",
              "swim power of the RIGHT motor channel: same scaling",
              "turn power of the LEFT motor channel: same scaling",
              "turn power of the RIGHT motor channel: same scaling",
              "bout: the fraction of each frame's 0.914 s spent inside a swim bout (either channel)"]
    for i in range(5):
        ax = fig.add_subplot(gs[i, :])
        blocks(ax, label=(i == 0))
        ax.plot(t_min[valid], E[valid, i], color=cols[i], lw=0.4)
        ax.set_ylim(0, 1.55 if i < 4 else 1.02)
        ax.tick_params(labelsize=7)
        ax.text(0.0, 1.30 if i == 0 else 1.04, titles[i], transform=ax.transAxes, fontsize=9, color="white",
                va="bottom")
    ax.set_xlabel("time since the first imaging frame (min)", fontsize=8)
    # swim time per condition
    ax = fig.add_subplot(gs[5:7, 0])
    names = list(swim_frac)
    ax.bar(range(9), [swim_frac[k]["time_in_bouts_pct"] for k in names], color="white")
    for i, k in enumerate(names):
        ax.text(i, swim_frac[k]["time_in_bouts_pct"] + 0.5, f"{swim_frac[k]['bouts_per_min']:.0f}/min",
                ha="center", fontsize=6.5, color="0.8")
    ax.set_xticks(range(9), names, rotation=40, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.text(0, 1.04, "% of each condition's time inside a swim bout\n(label: bouts per minute)",
            transform=ax.transAxes, fontsize=9, va="bottom")
    # the exact test: mismatched frames vs lag
    ax = fig.add_subplot(gs[5:7, 1])
    for x in ev["exact_test"]:
        lag = np.array(list(x["mismatched_by_lag"].keys()), int)
        mm = np.array(list(x["mismatched_by_lag"].values()))
        ax.plot(lag, mm, marker="o", ms=3, lw=1, label=f"f{x['feature']} {x['condition']}")
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("lag L (frames): release frame t vs the stimulus channels at TTL peak t + L", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center")
    ax.text(0, 1.04, "frames where the release feature DISAGREES with the raw\nstimulus channels sampled at the "
            "volume's TTL peak", transform=ax.transAxes, fontsize=9, va="bottom")
    # condition onsets
    ax = fig.add_subplot(gs[5:7, 2])
    d = np.array(ev["onsets_minus_release_offsets_frames"])
    ax.bar(range(9), d, color="white", width=0.5)
    ax.scatter(range(9), d, color="white", s=12)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_ylim(-3, 3)
    ax.set_xticks(range(9), [str(c) for c in cnames], rotation=40, fontsize=7)
    ax.tick_params(labelsize=7)
    ax.text(0, 1.04, "condition onset from the ephys trial-id channel, as a volume,\nMINUS the release's "
            "CONDITION_OFFSETS (frames): all 0", transform=ax.transAxes, fontsize=9, va="bottom")
    # two zoomed overlays: release feature vs the channel at the TTL peak and the frame mean
    for k_, (j, ch) in enumerate([(4, 6), (7, 3)]):
        ax = fig.add_subplot(gs[7:9, k_])
        c = int(np.bincount(cond[np.abs(S[:, j]) > 0]).argmax())
        fr = np.arange(offsets[c], offsets[c] + 100)

        def unit(v):
            return (v - v.min()) / max(np.ptp(v), 1e-9)
        ax.step(fr, unit(S[fr, j]) + 0.02, where="post", color="#4cc970", lw=1.6, label=f"release feature f{j}")
        ax.step(fr, unit(Ttl[ch][fr]), where="post", color="white", lw=0.9, label=f"{STIM_CH[ch]} at the TTL peak")
        ax.step(fr, unit(Bin[ch][fr]) - 0.02, where="post", color="0.55", lw=0.8, ls="--",
                label=f"{STIM_CH[ch]}, mean over the frame")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("release frame = marker volume", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="center right")
        ax.text(0, 1.04, f"first 100 frames of '{cnames[c]}' (each rescaled to 0..1)", transform=ax.transAxes,
                fontsize=9, va="bottom")
    ax = fig.add_subplot(gs[7:9, 2])
    ax.axis("off")
    tot0 = sum(x["mismatched_by_lag"][0] for x in ev["exact_test"])
    tot1 = sum(x["mismatched_by_lag"][1] + x["mismatched_by_lag"][-1] for x in ev["exact_test"]) / 2
    nfr = sum(x["frames"] for x in ev["exact_test"])
    lines = [f"marker volumes {ev['marker_volumes']:,}; release frames {ev['release_frames']:,};",
             f"TTL volume peaks found in the ephys file {ev['volume_ttl_peaks_found']:,}",
             "",
             "release frame t == marker volume t (lag 0):",
             f"  {tot0} of {nfr:,} frames mismatched at lag 0,",
             f"  {tot1:.0f} on average at lag +-1 (10 features, 8 conditions)",
             "the release samples the stimulus at the volume's",
             "TTL peak, 12.5 ms before its first plane",
             "",
             "the release's 7 last frames (end of 'dark') have",
             "no ephys: the 5 ephys features are 0 there"]
    ax.text(0, 1.0, "\n".join(lines), transform=ax.transAxes, fontsize=9, va="top", color="white")
    out = os.path.join(EXP, "data", "figs", "ephys_features.png")
    fig.savefig(out, dpi=120, facecolor="black")
    shutil.copy(out, os.path.join(EXP, "png", os.path.basename(out)))
    plt.close(fig)
    print("[figure]", out)


if __name__ == "__main__":
    main()

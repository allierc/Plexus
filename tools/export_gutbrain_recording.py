#!/usr/bin/env python
"""Freeze one fish of Chen, James, Ruetten et al. 2026 (Nat Commun, gut / vascular interoception) as a training
REFERENCE: graphs_data/zebrafish/gutbrain_<condition>_f<k>_recording.npz (+ .json provenance), exp17's trace schema.

    PYTHONPATH=src python tools/export_gutbrain_recording.py --condition glucose --fish 1
    PYTHONPATH=src python tools/export_gutbrain_recording.py --condition glucose --fish 1 --inspect   # print, write nothing

Reads the Figshare deposit (10.25378/janelia.31967640) as it was downloaded into
experiments/exp20_gutbrain_graphcast/data/ (fetch_figshare.py, MD5-checked), unbinned (Cedric, 2026-10-02):

  <cond>_fish<k>_cells0_clean.hdf5   Voluseg: cell_timeseries (detrended), cell_baseline, background, cell_x/y/z +
                                     cell_weights (the footprint's voxels and weights, zero-padded)
  <cond>_fish<k>_volume0.hdf5        volume_mean / volume_mask (anatomy, not exported)
  <cond>_fish<k>*.26chFlt-v10        float32, 26 interleaved channels at 6 kHz (eclass.py, software version 10)

THE FRAME CLOCK IS THE LIGHT-SHEET TRIGGER (ch_LS0): one run of samples above 3.8 V per volume (the pipeline's
`pullTimes`); its count must equal the traces' frame count, or exceed it by at most MAX_EXTRA_TRIGGERS trailing
volumes that were not saved (the first T are kept). The deposit has no parameters.pickle.

dF/F IS THE PAPER'S (Methods): (F - F_baseline) / (F_baseline - F_background), F = cell_timeseries,
F_baseline = cell_baseline (Voluseg's rolling low percentile), F_background the file's scalar `background`. A cell
whose baseline comes within MIN_SIGNAL counts of the background at any frame has no defined dF/F and is dropped
(2.8 % of glucose fish 1 at 2 counts). The volume(s) a UV pulse overlaps are replaced by the volume before (the
Methods say the pulse's frames are replaced; the deposit's traces are not -- checked on glucose fish 1). dF/F is then
clipped to [CLIP_LO, CLIP_HI] (ZAPBench's range, App. B.3), the fraction clipped recorded.

THE FORCINGS (the known inputs, one row per volume; nothing about a pulse's TARGET is given but the beam's position):
    uv        fraction of the volume's interval the UV laser was on (ch_UV > 3.8 V): a 200 ms pulse at 1.117 s
              volumes is ~0.18 on one volume
    uv_x, uv_y   the galvo voltages (ch_galvoX, ch_galvoY) times the uv column: where the beam was, while it was on
    visual    the grating's speed multiplier (ch_velmul) / VEL_MAX, the volume's mean (open loop: ch_gain is 0)
    swim_l, swim_r   fictive swim power, the pipeline's windowed variance (40 ms Gaussian kernels) of ch_ep0 / ch_ep1,
              MEAN over the volume's interval (the pipeline samples it at the trigger instant and misses most of a
              ~0.2 s bout), less its median (the amplifier's noise floor), over its 99.5th percentile less the median,
              clipped to [0, 1.5]; a channel with no bouts (99.5th percentile under SWIM_LIVE x the median: glucose fish
              1's ch_ep1, 1.9x, against ch_ep0's 22x) is written as zeros and flagged in the provenance

THE TRIALS: every UV pulse, with its onset volume, duration, `ch_gpos` site label (1 = control, off the fish; 2, 3 =
the gut, the pipeline's `uvOnGut = gpos != ctrlPos`), galvo voltages, and whether its 55-s window (the paper's kernel)
fits in the recording. The labels are for the RULERS only; the law reads the forcings.

THE SPLIT (`split`, per frame: 0 train, 2 held out): the held-out windows are PRE_S before to POST_S after each held-out
pulse -- by default the LAST full-window pulse of every site (exp20 Decisions).

    dff        [T, N] float32   dF/F, clipped; column i = hdf5 cell `cell_index[i]`
    pos_um     [N, 3] float64   footprint centroid x, y, z in um (VOXEL_UM, inferred: exp20 Decisions)
    stimulus   [T, 6] float32   uv, uv_x, uv_y, visual, swim_l, swim_r
    stim_names [6] str
    condition  [T] int8         0 (one session, one condition)
    offsets    [2] int64        0, T
    names      [1] str          the condition
    t_s        [T] float64      the volume onsets, s from the first
    split      [T] int8         0 train, 2 held out
    trials     [P, 7] float64   onset frame, duration ms, site, galvo x, galvo y, full window (0/1), held out (0/1)
    cell_index [N] int64        the kept cells' rows in cells0_clean.hdf5
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
DATA = os.path.join(ROOT, "experiments", "exp20_gutbrain_graphcast", "data")

FS = 6000                                       # ephys samples per second (eclass.create_meta)
CHANNELS_V10 = ['ch_ep0', 'ch_ep1', 'ch_LS0', 'ch_LS1', 'ch_LED0', 'ch_LED1', 'ch_NIR0', 'ch_NIR1', 'ch_galvoX',
                'ch_galvoY', 'ch_UV', 'ch_UVlaserfeedback', 'ch_473', 'ch_velmul', 'ch_velobs', 'ch_gain', 'ch_trial',
                'ch_uvtrial', 'ch_pulsenum', 'ch_pwidth', 'ch_pcycle', 'ch_stimDuration', 'ch_gpos', 'ch_femtoOn',
                'ch_femtoDuration', 'ch_femtoPressure']
CH = {n: i for i, n in enumerate(CHANNELS_V10)}
TRIG_V = 3.8                                    # the pipeline's threshold on the light-sheet and UV channels, volts
UV_GAP = 600                                    # samples (0.1 s): UV-on samples closer than this are one pulse
VOXEL_UM = (0.8125, 0.8125, 2.0)                # x, y, z um per voxel -- INFERRED (exp20 Decisions)
MIN_SIGNAL = 2.0                                # counts: baseline must stay this far above the background
CLIP_LO, CLIP_HI = -0.25, 1.5                   # dF/F range kept (ZAPBench App. B.3)
VEL_MAX = 3.0                                   # the grating's largest speed multiplier in the deposit
SWIM_PCT, SWIM_CLIP = 99.5, 1.5
SWIM_LIVE = 5.0                                 # a swim channel with bouts: its 99.5th percentile power > 5x its median
PRE_S, POST_S = 10.0, 55.0                      # a trial window: before / after the pulse onset, s (55 s: the paper's kernel)
CTRL_SITE = 1
MAX_EXTRA_TRIGGERS = 2                          # light-sheet triggers past the traces' last frame (glucose fish 4-6,
                                                # L-glucose fish 2 and 4: 1-2) -- the acquisition's last volumes unsaved
STIM_NAMES = ["uv", "uv_x", "uv_y", "visual", "swim_l", "swim_r"]


def files(cond, fish):
    pre = os.path.join(DATA, f"{cond}_fish{fish}")
    eph = sorted(glob.glob(pre + "*26chFlt-v10"))
    if len(eph) != 1:
        raise FileNotFoundError(f"{pre}*26chFlt-v10: {len(eph)} files")
    return pre + "_cells0_clean.hdf5", pre + "_volume0.hdf5", eph[0]


def read_ephys(path):
    d = np.memmap(path, dtype=np.float32, mode="r")
    n = d.size // len(CHANNELS_V10)
    return d[:n * len(CHANNELS_V10)].reshape(n, len(CHANNELS_V10))


def runs_above(x, thr, gap=1):
    """(starts, ends) of the runs of samples above thr; samples closer than `gap` join one run."""
    a = np.where(np.asarray(x) > thr)[0]
    if a.size == 0:
        return a, a
    br = np.where(np.diff(a) > gap)[0]
    return a[np.r_[0, br + 1]], a[np.r_[br, a.size - 1]]


def frame_clock(E):
    st, _ = runs_above(E[:, CH["ch_LS0"]], TRIG_V)
    return st


def per_volume_mean(x, st):
    """Mean of x over each volume's interval [st[k], st[k+1]) (the last one as long as the median interval)."""
    edges = np.r_[st, st[-1] + int(np.median(np.diff(st)))]
    edges = np.clip(edges, 0, len(x))
    cs = np.r_[0.0, np.cumsum(np.asarray(x, np.float64))]
    return (cs[edges[1:]] - cs[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)


def swim_power(sig):
    """The pipeline's `windowed_variance`: 40 ms Gaussian kernels (sigma 4 ms) for the mean and the variance."""
    from scipy.signal import fftconvolve
    from scipy.signal.windows import gaussian
    kw = int(0.04 * FS)
    k = gaussian(kw, kw // 10)
    k /= k.sum()
    s = np.asarray(sig, np.float64)
    var = (s - fftconvolve(s, k, "same")) ** 2
    return fftconvolve(var, k, "same")


def forcings_and_trials(E, st):
    T = len(st)
    uv_on = (np.asarray(E[:, CH["ch_UV"]]) > TRIG_V).astype(np.float64)
    uv = per_volume_mean(uv_on, st)
    gx = per_volume_mean(np.asarray(E[:, CH["ch_galvoX"]]) * uv_on, st)
    gy = per_volume_mean(np.asarray(E[:, CH["ch_galvoY"]]) * uv_on, st)
    vis = per_volume_mean(E[:, CH["ch_velmul"]], st) / VEL_MAX
    sw, live = [], []
    for ch in ("ch_ep0", "ch_ep1"):
        p = per_volume_mean(swim_power(E[:, CH[ch]]), st)
        floor, top = np.median(p), np.percentile(p, SWIM_PCT)
        live.append(bool(top > SWIM_LIVE * floor))
        sw.append(np.clip((p - floor) / max(top - floor, 1e-12), 0, SWIM_CLIP) if live[-1] else np.zeros_like(p))
    S = np.stack([uv, gx, gy, vis, sw[0], sw[1]], 1).astype(np.float32)
    on, off = runs_above(E[:, CH["ch_UV"]], TRIG_V, gap=UV_GAP)
    period = float(np.median(np.diff(st))) / FS
    pre, post = int(round(PRE_S / period)), int(np.ceil(POST_S / period))
    trials = []
    for a, b in zip(on, off):
        f = int(np.searchsorted(st, a, side="right") - 1)
        full = f - pre >= 0 and f + post < T
        trials.append([f, (b - a + 1) / FS * 1e3, float(E[a, CH["ch_gpos"]]), float(E[a, CH["ch_galvoX"]]),
                       float(E[a, CH["ch_galvoY"]]), float(full), 0.0])
    return S, np.array(trials, np.float64).reshape(-1, 7), period, pre, post, live


def hold_out(trials, pre, post, T, how="last_per_site"):
    """Mark the held-out trials (column 6) and return the per-frame split: the LAST full-window pulse of every site."""
    split = np.zeros(T, np.int8)
    if how != "last_per_site":
        raise ValueError(how)
    for site in np.unique(trials[:, 2]):
        k = np.where((trials[:, 2] == site) & (trials[:, 5] > 0))[0]
        if len(k):
            trials[k[-1], 6] = 1.0
    for f in trials[trials[:, 6] > 0, 0].astype(int):
        split[max(0, f - pre):min(T, f + post + 1)] = 2
    return split


def centroids(h5, keep, chunk=20000):
    out = np.zeros((len(keep), 3))
    for s in range(0, len(keep), chunk):
        ix = keep[s:s + chunk]
        # the footprint arrays are padded: weight NaN, coordinate -1 (Voluseg); only the valid voxels count
        w = h5["cell_weights"][ix[0]:ix[-1] + 1][ix - ix[0]].astype(np.float64)
        w = np.where(np.isfinite(w) & (h5["cell_x"][ix[0]:ix[-1] + 1][ix - ix[0]] >= 0), w, 0.0)
        for j, key in enumerate(("cell_x", "cell_y", "cell_z")):
            c = h5[key][ix[0]:ix[-1] + 1][ix - ix[0]].astype(np.float64)
            out[s:s + len(ix), j] = (c * w).sum(1) / np.maximum(w.sum(1), 1e-12) * VOXEL_UM[j]
    return out


def dff_kept(h5, uv_frames, chunk=20000):
    bg = float(h5["background"][()])
    N, T = int(h5["n"][()]), int(h5["t"][()])
    keep, cols, n_clip, n_tot = [], [], 0, 0
    for s in range(0, N, chunk):
        B = h5["cell_baseline"][s:s + chunk]
        F = h5["cell_timeseries"][s:s + chunk]
        ok = (B - bg).min(1) >= MIN_SIGNAL
        d = (F[ok] - B[ok]) / (B[ok] - bg)
        for f in uv_frames:                         # the pulse's volume(s) <- the volume before
            if f > 0:
                d[:, f] = d[:, f - 1]
        n_clip += int(((d < CLIP_LO) | (d > CLIP_HI)).sum())
        n_tot += d.size
        cols.append(np.clip(d, CLIP_LO, CLIP_HI).astype(np.float32))
        keep.append(np.where(ok)[0] + s)
    return np.concatenate(cols, 0).T.copy(), np.concatenate(keep), bg, n_clip / max(n_tot, 1), N, T


def sha256_head(p, n=64 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="glucose")
    ap.add_argument("--fish", type=int, default=1)
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    import h5py
    t0 = time.time()
    fc, fv, fe = files(a.condition, a.fish)
    E = read_ephys(fe)
    st = frame_clock(E)
    h5 = h5py.File(fc, "r")
    T = int(h5["t"][()])
    extra = len(st) - T
    if not 0 <= extra <= MAX_EXTRA_TRIGGERS:
        raise ValueError(f"{fe}: {len(st)} light-sheet volumes, the traces {T} frames")
    st = st[:T]                                 # trailing triggers whose volume was not saved (the pipeline indexes
                                                # stackTimes from the first trigger too); recorded in the provenance
    S, trials, period, pre, post, live = forcings_and_trials(E, st)
    split = hold_out(trials, pre, post, T)
    print(f"{a.condition} fish {a.fish}: {T} volumes, {period:.4f} s apart ({T * period / 60:.1f} min); "
          f"{len(trials)} UV pulses; window {pre} + {post} volumes")
    for r in trials:
        print(f"  frame {int(r[0]):5d}  {r[1]:5.0f} ms  site {int(r[2])}  galvo ({r[3]:+.2f}, {r[4]:+.2f})  "
              f"{'full' if r[5] else 'cut '}  {'HELD OUT' if r[6] else ''}")
    print(f"  forcings: visual on {np.mean(S[:, 3] > 0) * 100:.0f} % of volumes; swim power > 0.1 on "
          f"{np.mean(S[:, 4] > 0.1) * 100:.1f} % (L), {np.mean(S[:, 5] > 0.1) * 100:.1f} % (R); swim channels live "
          f"{live}; held out "
          f"{int((split == 2).sum())} volumes")
    if a.inspect:
        return
    uv_frames = sorted({int(f) for f in np.where(S[:, 0] > 0)[0]})
    X, keep, bg, frac_clip, N0, _ = dff_kept(h5, uv_frames)
    P = centroids(h5, keep)
    from plexus.paths import graphs_data_path
    name = f"gutbrain_{a.condition}_f{a.fish}"
    out = graphs_data_path("zebrafish", f"{name}_recording.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, dff=X, pos_um=P, stimulus=S, stim_names=np.array(STIM_NAMES), condition=np.zeros(T, np.int8),
             offsets=np.array([0, T], np.int64), names=np.array([a.condition]), t_s=(st - st[0]) / FS,
             split=split, trials=trials, cell_index=keep)
    prov = {"name": name, "written": time.strftime("%Y-%m-%d %H:%M"), "tool": "tools/export_gutbrain_recording.py",
            "source": {"cells": os.path.basename(fc), "ephys": os.path.basename(fe),
                       "cells_sha256_first64MB": sha256_head(fc), "ephys_sha256_first64MB": sha256_head(fe)},
            "frames": T, "extra_triggers_dropped": int(extra), "volume_s": period, "cells_in_file": N0, "cells_kept": int(len(keep)),
            "min_signal_counts": MIN_SIGNAL, "background": bg, "clip": [CLIP_LO, CLIP_HI], "frac_clipped": frac_clip,
            "uv_frames_replaced": uv_frames, "voxel_um": VOXEL_UM, "window_volumes": [pre, post],
            "trial_columns": ["onset_frame", "duration_ms", "site", "galvo_x", "galvo_y", "full_window", "held_out"],
            "trials": trials.tolist(), "stim_names": STIM_NAMES, "swim_live": live,
            "dff_mean_sd": [float(X.mean()), float(X.std())], "extent_um": np.ptp(P, 0).tolist()}
    json.dump(prov, open(out.replace(".npz", ".json"), "w"), indent=1)
    print(f"-> {out}: {X.shape[1]:,} of {N0:,} cells, clipped {frac_clip * 100:.3f} %, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

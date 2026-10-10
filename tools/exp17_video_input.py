"""exp17 batch 26: THE STIMULUS VIDEO AS INPUT -- its alignment with the recording, and a movie of what the model reads
(Cedric, 2026-10-09: "next batch 26 with a video input data/260910_fish2_stimulus_video.mp4"; "make sure that input
stimuli block from the video align with the block observed data in terms of starting frame and frequency"). Local.

THE VIDEO. experiments/exp17_zapbench_graphcast/data/260910_fish2_stimulus_video.mp4: 360 x 360 px, 30 frames/s,
216,000 frames = 7,200.0 s, h264, the stimulus in the red channel only. The recording: 7,879 volumes, ~0.914 s each.

THE CLOCK. The raw ephys file (stimuli_and_ephys.10chFlt, 10 float32 channels at 6 kHz; tools/export_ephys_features.py)
holds the stimulus parameters (ch3 / ch6 stimParam4 / stimParam3, ch4 the block id) and, through zapbench.zarr's
plane_time_ephys, the ephys sample at which each imaging volume starts (volumes 0 .. 7,871). The video's brightness
steps (|mean red, frame to frame| > 20 of 255) are paired with the nearest raw stimulus change within 0.5 s of
t_video + 6.2 s; a least-squares line t_ephys = a + s * t_video through the pairs maps the video clock to the ephys
clock; each volume's start then maps to video time, t_k = (plane_time_ephys[k, 0] / 6000 - a) / s. Volumes 7,872 ..
7,878 have no ephys (the file ends): their starts are extrapolated at the mean period. Video time 0 is frame 0;
video frame i covers [i / 30, (i + 1) / 30) s.

CHECKS (the alignment is refused if one fails):
  fit residuals          every pair within 33 ms (one video frame)
  block edges            the video steps at the dots and dark onsets fall in recording frames 649 and 7,279, the
                         release's CONDITION_OFFSETS
NOT A CHECK, MEASURED: in taxis the video's brightness steps fall on the ephys file's 10-s counter (ch5) and the raw
stimulus parameters (ch3 / ch6) change ~1.0 s LATER for most steps (`taxis_raw_lag_s`); in flash and at the block
edges they coincide within 33 ms. Which one the fish saw is not decided by these files.
  flash                  each flash step falls in the frame before the release's f4 changes (the release samples a
                         feature at the volume's start, so a mid-frame change appears one frame later)

THE MODEL INPUT (the exporter of batch 26 uses `subbins`): recording frame k = the video frames with time in
[t_k, t_{k+1}), split into SUB = 8 equal sub-bins (~0.114 s, ~3.4 video frames each) and averaged in each; the red
channel averaged over 4 x 4 pixels, 360 -> 90. 8 sub-bins per frame because the gratings' brightness cycles at up to
1.5 Hz (open loop, position), above the recording's 1 / (2 x 0.914 s) = 0.55 Hz.

    PYTHONPATH=src:tools python tools/exp17_video_input.py align      # -> data/video_alignment.json, data/video_frame_start_s.npy
    PYTHONPATH=src:tools python tools/exp17_video_input.py movie      # -> presentation/Movies/video_input.mp4 (+ .png)
    PYTHONPATH=src:tools python tools/exp17_video_input.py export     # -> graphs_data/zebrafish/zapbench_destripe_video_recording.npz
    PYTHONPATH=src:tools python tools/exp17_video_input.py masks      # -> input_mask_destripe_markers.npz, _nomarkers.npz

THE RECORDING OF BATCH 26 (`export`): zapbench_destripe_recording.npz with its `stimulus` [7,879, 22] extended to
[7,879, 22 + SUB x 90 x 90 = 64,822]: the 22 release features, then frame k's model input (`subbins`, sub-bin major,
row-major pixels), the mean red / 255 in [0, 1]. Every other array is copied unchanged. The trainer's drive writes all
64,822 values into the `stimulus` set each frame (one element per value, window [0, 0]), as it writes the 22 features
today: the video enters through the drive, and `drive: off` (the no-stimulus rollout) blanks it with the features.
Built frame by frame from the decoded video (each video frame's 4 x 4 pixel means added to the sub-bin its time falls
in, then divided by the count; the same numbers as `subbins`); frames 7,877-7,878 run past the video's end: their
empty sub-bins stay 0 (dark).

THE MASKS of the 22 feature columns (`masks`), for the batch-26 laws whose varying features the video replaces:
input_mask_destripe_markers.npz (mask_by_input [N, 22]: the 9 marker columns on every neuron, the 13 varying ones on
none: 22.3's per-block offset kept) and input_mask_destripe_nomarkers.npz (all 0: 24.10's law, rest_block the offset).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
VIDEO = os.path.join(EXP, "data", "260910_fish2_stimulus_video.mp4")
FPS = 30.0
SUB = 8                  # sub-bins per recording frame
DOWN = 4                 # pixels averaged per side: 360 -> 90
STEP = 20.0              # a brightness step: |mean red frame to frame| above this, of 255
EPHYS_HZ = 6000.0


def _zb():
    from plexus.paths import graphs_data_path
    return graphs_data_path("zebrafish", "zapbench")


def video_stats():
    """Per video frame: the mean red (0-255) and the mean |change| from the previous frame. Cached in data/."""
    out = os.path.join(EXP, "data", "video_stats.npz")
    if os.path.exists(out) and os.path.getmtime(out) > os.path.getmtime(VIDEO):
        z = np.load(out)
        return z["lum"], z["motion"]
    import cv2
    c = cv2.VideoCapture(VIDEO)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    lum, motion, prev, i = np.zeros(n, np.float32), np.zeros(n, np.float32), None, 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        r = f[::2, ::2, 2].astype(np.float32)
        lum[i] = r.mean()
        motion[i] = np.abs(r - prev).mean() if prev is not None else 0.0
        prev, i = r, i + 1
    np.savez(out, lum=lum[:i], motion=motion[:i])
    return lum[:i], motion[:i]


def align():
    import zarr
    from plexus.tasks import trace_recording as TR
    lum, _ = video_stats()
    zb = _zb()
    E = np.memmap(os.path.join(zb, "release_20240930", "volumes", "20240930", "stimuli_raw", "stimuli_and_ephys.10chFlt"),
                  np.float32, "r").reshape(-1, 10)
    v0 = zarr.open(os.path.join(zb, "zapbench.zarr"), mode="r")["plane_time_ephys"][:][:, 0].astype(np.float64)
    tv = (np.where(np.abs(np.diff(lum)) > STEP)[0] + 1) / FPS
    ev = np.unique(np.concatenate([np.where(np.abs(np.diff(np.asarray(E[:, c]))) > 1e-3)[0] + 1 for c in (3, 4, 6)])) / EPHYS_HZ
    pairs = [(t, ev[j]) for t in tv for j in [int(np.argmin(np.abs(ev - (t + 6.2))))] if abs(ev[j] - (t + 6.2)) < 0.5]
    P = np.array(pairs)
    (a, s), *_ = np.linalg.lstsq(np.c_[np.ones(len(P)), P[:, 0]], P[:, 1], rcond=None)
    res = P[:, 1] - (a + s * P[:, 0])
    rec = TR.load("zapbench_destripe")
    S, off, T = rec["stimulus"], np.asarray(rec["offsets"]), len(rec["stimulus"])
    tk = (v0 / EPHYS_HZ - a) / s
    per = float(np.diff(tk).mean())
    tk = np.r_[tk, tk[-1] + per * np.arange(1, T - len(tk) + 1)]
    edges = np.r_[tk, tk[-1] + per]                                     # T + 1: frame k = [edges[k], edges[k + 1])
    kv = np.searchsorted(edges, tv, side="right") - 1
    f4 = np.where(np.diff(S[:, 4]) != 0)[0] + 1
    fl = (tv > edges[off[2]]) & (tv < edges[off[3]])
    flash_lag = [int(f4[np.argmin(np.abs(f4 - k))] - k) for k in kv[fl]]
    tx = tv[(tv > edges[off[3]]) & (tv < edges[off[4]])]
    ev36 = np.unique(np.concatenate([np.where(np.abs(np.diff(np.asarray(E[:, c]))) > 1e-3)[0] + 1 for c in (3, 6)])) / EPHYS_HZ
    lag = [float(ev36[np.searchsorted(ev36, a + s * t - 0.1)] - (a + s * t)) for t in tx]
    checks = {"fit_residual_ms_max": float(1000 * np.abs(res).max()),
              "dots_onset_frame": int(kv[0]), "dark_onset_frame": int(kv[-1]),
              "condition_offsets": [int(off[1]), int(off[8])],
              "flash_steps": int(fl.sum()), "flash_lag_frames": sorted(set(flash_lag))}
    ok = (checks["fit_residual_ms_max"] < 1000 / FPS and checks["dots_onset_frame"] == off[1]
          and checks["dark_onset_frame"] == off[8] and checks["flash_lag_frames"] == [1])
    J = {"video": VIDEO, "video_frames": int(len(lum)), "video_s": len(lum) / FPS,
         "pairs": int(len(P)), "video_steps": int(len(tv)),
         "t_ephys_s = a + s * t_video_s": {"a": float(a), "s": float(s), "rate_minus_1_ppm": float(1e6 * (s - 1))},
         "frame0_start_video_s": float(edges[0]), "frame_period_video_s": {"mean": per, "min": float(np.diff(tk[:len(v0)]).min()),
                                                                        "max": float(np.diff(tk[:len(v0)]).max())},
         "drift_vs_k_x_0.914_s_at_end": float(edges[T - 1] - (T - 1) * 0.914),
         "frames_without_ephys": [len(v0), T - 1], "frames_past_video_end": [int(k) for k in range(T) if edges[k + 1] > len(lum) / FPS],
         "checks": checks, "checks_pass": bool(ok),
         "taxis_raw_lag_s": {"steps": len(lag), "lag_quantiles_10_50_90": [float(np.quantile(lag, q)) for q in (0.1, 0.5, 0.9)],
                             "within_33_ms": int(np.sum(np.abs(lag) < 1 / FPS))}}
    json.dump(J, open(os.path.join(EXP, "data", "video_alignment.json"), "w"), indent=1)
    np.save(os.path.join(EXP, "data", "video_frame_start_s.npy"), edges)
    print(json.dumps(J, indent=1))
    if not ok:
        raise SystemExit("[video] alignment checks FAILED -- not used")
    return edges


def subbins(frames, t_first, t0, t1):
    """[SUB, 360 / DOWN, 360 / DOWN] float32: the model input of one recording frame [t0, t1) s of video time, from
    `frames` (consecutive decoded red frames, uint8 [n, 360, 360], the first at video time t_first)."""
    out = np.zeros((SUB, 360 // DOWN, 360 // DOWN), np.float32)
    ti = t_first + np.arange(len(frames)) / FPS
    for j in range(SUB):
        a, b = t0 + j * (t1 - t0) / SUB, t0 + (j + 1) * (t1 - t0) / SUB
        m = (ti >= a) & (ti < b)
        if m.any():
            x = frames[m].astype(np.float32).mean(0)
            out[j] = x.reshape(360 // DOWN, DOWN, 360 // DOWN, DOWN).mean((1, 3))
    return out


def movie(sec=6.0):
    """Movies/video_input.mp4: per block a `sec`-s excerpt at 30 frames/s (the window of the block with the most
    frame-to-frame change, see below), LEFT the video as shown, RIGHT the model input at that instant (the current sub-bin of the
    current recording frame, 90 x 90), under them the 9 blocks with the current one under a grey bar. The excerpt is
    centred on the block's largest 1-s change of the video (frame-to-frame |change| plus |brightness step|)."""
    import shutil
    import subprocess
    import tempfile
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plexus.tasks import trace_recording as TR
    from plexus.tasks.trace_recording import _ffmpeg
    edges = np.load(os.path.join(EXP, "data", "video_frame_start_s.npy"))
    J = json.load(open(os.path.join(EXP, "data", "video_alignment.json")))
    if not J["checks_pass"]:
        raise SystemExit("[video] the alignment failed its checks; run `align` first")
    lum, motion = video_stats()
    rec = TR.load("zapbench_destripe")
    names, off = list(rec["names"]), np.asarray(rec["offsets"])
    act = motion + np.abs(np.r_[0.0, np.diff(lum)])
    cs = np.r_[0.0, np.cumsum(act)]
    n_ex = int(sec * FPS)
    c = cv2.VideoCapture(VIDEO)
    tmp = tempfile.mkdtemp(prefix="vidin_")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12.8, 7.6), facecolor="black")
    axl, axr = fig.add_axes([0.03, 0.22, 0.45, 0.66]), fig.add_axes([0.52, 0.22, 0.45, 0.66])
    sx = fig.add_axes([0.03, 0.06, 0.94, 0.07])
    n_out = 0
    for b, nm in enumerate(names):
        lo, hi = int((edges[off[b]] + 2) * FPS), int((edges[off[b + 1]] - sec - 2) * FPS)
        hi = min(hi, len(lum) - n_ex - 1)
        # the excerpt CENTRED on the block's largest 1-s change (a flash step in the middle, not at the edge)
        sm = cs[lo + int(FPS):hi + n_ex] - cs[lo:hi + n_ex - int(FPS)] if hi > lo else np.zeros(1)
        i0 = int(np.clip(lo + int(np.argmax(sm)) + int(FPS) // 2 - n_ex // 2, lo, max(hi, lo)))
        t_first = i0 / FPS - 1.0                                          # one recording frame of margin each side
        c.set(cv2.CAP_PROP_POS_FRAMES, max(int(round(t_first * FPS)), 0))
        t_first = max(int(round(t_first * FPS)), 0) / FPS
        buf = []
        for _ in range(n_ex + int(2.2 * FPS)):
            ok, f = c.read()
            if not ok:
                break
            buf.append(f[:, :, 2])
        buf = np.array(buf)
        cache = {}
        for i in range(i0, i0 + n_ex):
            t = i / FPS
            k = int(np.searchsorted(edges, t, side="right") - 1)
            j = min(int(SUB * (t - edges[k]) / (edges[k + 1] - edges[k])), SUB - 1)
            if k not in cache:
                cache[k] = subbins(buf, t_first, edges[k], edges[k + 1])
            for ax in (axl, axr):
                ax.clear()
                ax.axis("off")
            axl.imshow(buf[i - int(round(t_first * FPS))], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axr.imshow(cache[k][j], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axl.set_title("shown to the fish (red channel): 360 x 360 px, 30 frames/s", fontsize=12, color="white")
            axr.set_title(f"model input: 90 x 90, recording frame {k}, sub-bin {j + 1} of {SUB}", fontsize=12, color="white")
            fig.texts.clear()
            fig.text(0.03, 0.965, f"{nm}   video t = {t:7.1f} s   recording frame {k} starts at {edges[k]:7.2f} s",
                     fontsize=14, color="white", va="top", weight="bold")
            sx.clear()
            for q in range(len(names)):
                sx.axvline(edges[off[q]], color="0.35", lw=0.6)
                sx.text((edges[off[q]] + edges[off[q + 1]]) / 2, 1.08, names[q].replace("open loop", "open"),
                        transform=sx.get_xaxis_transform(), ha="center", va="bottom", fontsize=10,
                        color="white" if q == b else "0.5", weight="bold" if q == b else "normal")
            sx.axvspan(edges[off[b]], edges[off[b + 1]], color="white", alpha=0.28, lw=0)
            sx.axvline(t, color="#ff9e1a", lw=1.5)
            sx.set_xlim(edges[0], edges[-1])
            sx.set_ylim(0, 1)
            sx.axis("off")
            fig.savefig(os.path.join(tmp, f"{n_out:05d}.png"), dpi=100, facecolor="black")
            n_out += 1
        print(f"[video] {nm}: video {i0 / FPS:.1f} .. {(i0 + n_ex) / FPS:.1f} s")
    plt.close(fig)
    stem = os.path.join(EXP, "presentation", "Movies", "video_input")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-framerate", str(int(FPS)), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", stem + ".mp4"],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{int(4.5 * n_ex):05d}.png"), stem + ".png")       # the poster: mid-movie (a grating)
    shutil.rmtree(tmp)
    print("[video] ->", stem + ".mp4", f"({n_out} frames, {n_out / FPS:.0f} s)")


def export():
    import cv2
    from plexus.paths import graphs_data_path
    edges = np.load(os.path.join(EXP, "data", "video_frame_start_s.npy"))
    J = json.load(open(os.path.join(EXP, "data", "video_alignment.json")))
    if not J["checks_pass"]:
        raise SystemExit("[video] the alignment failed its checks; run `align` first")
    T = len(edges) - 1
    P_ = 360 // DOWN
    acc = np.zeros((T, SUB, P_, P_), np.float32)
    cnt = np.zeros((T, SUB), np.int32)
    c = cv2.VideoCapture(VIDEO)
    i = 0
    while True:
        ok, f = c.read()
        if not ok:
            break
        t = i / FPS
        k = int(np.searchsorted(edges, t, side="right") - 1)
        if 0 <= k < T:
            j = min(int(SUB * (t - edges[k]) / (edges[k + 1] - edges[k])), SUB - 1)
            acc[k, j] += f[:, :, 2].astype(np.float32).reshape(P_, DOWN, P_, DOWN).mean((1, 3))
            cnt[k, j] += 1
        i += 1
    acc /= np.maximum(cnt, 1)[:, :, None, None]
    acc /= 255.0
    empty = int((cnt == 0).sum())
    src = graphs_data_path("zebrafish", "zapbench_destripe_recording.npz")
    z = np.load(src)
    out = {k: z[k] for k in z.files}
    S0 = np.asarray(out["stimulus"], np.float32)
    out["stimulus"] = np.concatenate([S0, acc.reshape(T, -1)], 1).astype(np.float32)
    out["video_layout"] = np.array([S0.shape[1], SUB, P_, P_])        # features, then sub-bins x rows x columns
    dst = graphs_data_path("zebrafish", "zapbench_destripe_video_recording.npz")
    np.savez(dst, **out)
    prov = json.load(open(graphs_data_path("zebrafish", "zapbench_destripe_recording.json")))
    prov.update({"base_recording": src, "script": "tools/exp17_video_input.py export", "video": VIDEO,
                 "stimulus": f"[{T}, {out['stimulus'].shape[1]}]: the {S0.shape[1]} release features, then {SUB} sub-bins x "
                             f"{P_} x {P_} px of the video (mean red / 255), sub-bin major, row-major",
                 "video_frames_used": int(cnt.sum()), "video_frames_per_subbin": [int(cnt[cnt > 0].min()), int(cnt.max())],
                 "empty_subbins": empty, "alignment": J})
    json.dump(prov, open(dst.replace(".npz", ".json"), "w"), indent=1)
    print(f"[video] -> {dst}: stimulus {out['stimulus'].shape}, {int(cnt.sum()):,} video frames used, "
          f"{cnt[cnt > 0].min()}-{cnt.max()} per sub-bin, {empty} empty sub-bins (past the video's end)")


def masks():
    from plexus.paths import graphs_data_path
    mk = np.load(graphs_data_path("zebrafish", "input_mask_destripe_bal20_markall.npz"))["mask_by_input"]
    marker = mk.min(0) > 0
    m = np.zeros_like(mk)
    m[:, marker] = 1.0
    np.savez(graphs_data_path("zebrafish", "input_mask_destripe_markers.npz"), mask=np.ones(len(m), np.float32),
             mask_by_input=m)
    np.savez(graphs_data_path("zebrafish", "input_mask_destripe_nomarkers.npz"), mask=np.zeros(len(m), np.float32),
             mask_by_input=np.zeros_like(mk))
    print(f"[video] markers {[int(j) for j in np.where(marker)[0]]} -> input_mask_destripe_markers.npz (every neuron); "
          f"input_mask_destripe_nomarkers.npz (all 0)")


if __name__ == "__main__":
    {"align": align, "movie": movie, "export": export, "masks": masks}[sys.argv[1] if len(sys.argv) > 1 else "align"]()

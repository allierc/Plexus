"""Build the exp17 deck (GraphCast on ZAPBench): figures, movies and slide bodies, no number typed by hand.

    PYTHONPATH=src /workspace/.conda_envs/neural-graph-linux/bin/python tools/exp17_slides.py
    python presentation/fit_deck.py --dir experiments/exp17_zapbench_graphcast/presentation --deck exp17.tex

Every slide follows the deck template of `presentation/` (plexus_oct_26.tex): the picture or movie on the
LEFT, the specifics -- equations, training scheme, numbers -- on the RIGHT, in a `\\fitcol`. Every number on a
slide is computed here from the file it describes (the release traces, the stimulus features, the
segmentation centroids, the published results table), so the deck cannot drift from the data (INSTRUCTION.md,
"The one rule"). Runs in the devcontainer only (INSTRUCTION.md, the cluster rule).

Slides written so far -- the three that explain GraphCast on ZAPBench:
    01  the data: 71,721 neurons, their dF/F, the stimulus conditions            (movie)
    02  GraphCast's graph on this brain: grid = neurons, multi-level mesh          (figure)
    03  training and the two-stage metric: curriculum, MSE against the mean       (figure)
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast")
PRES = os.path.join(EXP, "presentation")
GD = os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")
REL = os.path.join(GD, "graphs_data", "zebrafish", "zapbench", "release_20240930")
VOL = os.path.join(REL, "volumes", "20240930")
FFMPEG = "/workspace/.conda_envs/MPM-pytorch/bin/ffmpeg"      # no ffmpeg on PATH in this container

T_FRAMES, N_NEURONS, CHUNK = 7879, 71721, 512                  # the release's traces zarr: [t, neuron], 512 x 512
FRAME_S = 0.914                                                # seconds per volume (zapbench.zarr .zattrs)
NAMES = ("gain", "dots", "flash", "taxis", "turning", "position", "open loop", "rotation", "dark")

# the base mesh of the plan (exp17 md, "The mesh"): finest spacing, number of levels
MESH_L0_UM, MESH_LEVELS = 16.0, 4             # Cedric 2026-09-30: level 4 (256 um) is too coarse, removed
HISTORY, N_STIM, FORCE_WINDOW = 6, 22, (-5, 1)                 # frames read; stimulus dims kept; forcing frames
CURRICULUM = list(range(1, 21))                                # one stage per horizon, N1 / K updates each


# ============================================================================== data
def condition_offsets() -> tuple[int, ...]:
    """CONDITION_OFFSETS read from ZAPBench's own constants file, never retyped."""
    src = open(os.path.join(EXP, "papers", "zapbench_code", "zapbench", "constants.py")).read()
    return tuple(int(x) for x in re.search(r"CONDITION_OFFSETS = \(([^)]*)\)", src).group(1).split(","))


def centroids() -> np.ndarray:
    """[N, 3] soma centroids in um (x, y, z); row i is segment label i + 1 (checked)."""
    d = json.load(open(os.path.join(VOL, "segmentation", "dataframe_centroids.json")))
    lab = np.array([d["label"][str(i)] for i in range(N_NEURONS)])
    if not (lab == np.arange(1, N_NEURONS + 1)).all():
        raise ValueError("centroid rows are not labels 1..N in order")
    return np.array([[d[k][str(i)] for k in ("centroid_x", "centroid_y", "centroid_z")]
                     for i in range(N_NEURONS)], dtype=np.float64)


def traces(t0: int, t1: int) -> np.ndarray:
    """dF/F rows t0..t1-1 of the release traces (zarr v3, raw little-endian float32 chunks)."""
    root = os.path.join(VOL, "traces_unzipped", "traces", "c")
    rows = []
    for i in range(t0 // CHUNK, (t1 - 1) // CHUNK + 1):
        blk = np.concatenate([np.fromfile(os.path.join(root, str(i), str(j)), "<f4").reshape(CHUNK, CHUNK)
                              for j in range(math.ceil(N_NEURONS / CHUNK))], 1)[:, :N_NEURONS]
        rows.append(blk)
    a = np.concatenate(rows, 0)
    base = (t0 // CHUNK) * CHUNK
    return a[t0 - base:t1 - base]


def stimulus() -> np.ndarray:
    import zarr
    return zarr.open(os.path.join(VOL, "stimuli_features"), mode="r")[:]


def published() -> "pandas.DataFrame":
    import io
    import pandas as pd
    return pd.read_json(io.StringIO(json.load(open(os.path.join(
        EXP, "papers", "Lueckmann_2025_published_results_combined.json")))))


# ============================================================================== the mesh (GraphCast on points)
# ============================================================================== figures
def _black(ax):
    ax.set_facecolor("black")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


MOVIE_FRAMES = 800     # the data movies sample the WHOLE recording, ~800 frames evenly (Cedric, 2026-10-02: as the run movies)


def movie_frames(n=MOVIE_FRAMES) -> np.ndarray:
    """The recording frames a data movie shows: n of the 7,879, evenly spaced from the first to the last."""
    return np.unique(np.linspace(0, T_FRAMES - 1, n).round().astype(int))


def brain_view(pos: np.ndarray) -> np.ndarray:
    """Positions (um) as the run movies draw them (trace_recording.render_movie): the brain HORIZONTAL, HEAD LEFT.
    The zap-inr anatomy frame has its long axis along x and is first turned head-up as (y, -x); ZAPBench's
    segmentation frame is already head-up; then (x, y) -> (-y, x) lays the head-up brain on its side, head left."""
    pos = np.asarray(pos, dtype=np.float64)
    if np.ptp(pos[:, 0]) > np.ptp(pos[:, 1]):
        pos = np.stack([pos[:, 1], -pos[:, 0], pos[:, 2]], 1)
    return np.stack([-pos[:, 1], pos[:, 0], pos[:, 2]], 1)


_BM: dict = {}         # the data movie's arrays, set before the frame workers fork (they read it, never copy it in)


def movie_brains(panels, out_mp4, frames, fps=25, fig_w=None, dpi=100, workers=16):
    """THE DATA MOVIES' ONE LOOK (slides 2 and 12, the --pair movie), that of the run movies: one panel per recording,
    side by side, every neuron a point (head left, the deeper drawn first) coloured by its dF/F on ONE scale (the 97th
    percentile of every panel's shown frames); each panel fitted to ITS OWN brain's extent (the two frames' brains
    differ in size), with its own 100-um bar; under each a strip of the brain-mean dF/F over the whole recording, the
    nine stimulus conditions as alternating grey blocks named above, an orange cursor at the shown frame.
    `panels`: [{P: [N, 3] brain_view positions, X: [len(frames), N] dF/F at `frames`, mean: [T_FRAMES] brain-mean
    dF/F of every frame, label: str}]. The frames render in a fork pool, the figure built once per worker."""
    import multiprocessing as mp
    import shutil
    import tempfile
    off = condition_offsets()
    cond_all = np.clip(np.searchsorted(off, np.arange(T_FRAMES), side="right") - 1, 0, len(NAMES) - 1)
    vmax = float(np.percentile(np.concatenate([p["X"].ravel()[::13] for p in panels]), 97.0))
    tmp = tempfile.mkdtemp(prefix="datamovie_")
    _BM.clear()
    _BM.update(panels=[dict(p, order=np.argsort(p["P"][:, 2])) for p in panels], frames=np.asarray(frames),
               cond_all=cond_all, vmax=vmax, tmp=tmp, fig_w=fig_w or (7.6 if len(panels) == 1 else 9.0 * len(panels)), dpi=dpi)
    ks = np.arange(len(frames))
    workers = min(workers, len(os.sched_getaffinity(0)))          # this machine's (or job's) usable cores
    chunks = [c for c in np.array_split(ks, max(1, min(workers, len(ks)))) if len(c)]
    if len(chunks) > 1:
        with mp.get_context("fork").Pool(len(chunks)) as pool:
            pool.map(_brain_frames, chunks)
    else:
        _brain_frames(chunks[0])
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", out_mp4],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{len(frames) // 2:05d}.png"), out_mp4.replace(".mp4", ".png"))
    shutil.rmtree(tmp)
    _BM.clear()
    return {"frames": int(len(frames)), "first": int(frames[0]), "last": int(frames[-1]), "vmax": vmax,
            "seconds": len(frames) / fps}


def _brain_frames(ks):
    """One worker of movie_brains: the figure built ONCE (brains, strips, condition blocks), then per frame only the
    points' colours, the cursors and the time text change before each save."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _BM
    panels, frames, cond_all, vmax = d["panels"], d["frames"], d["cond_all"], d["vmax"]
    n = len(panels)
    tall_ = any(p.get("band") is not None for p in panels)        # Cedric, 2026-10-07: slide 2's fish larger
    fig = plt.figure(figsize=(d["fig_w"], 9.0 if tall_ else 6.6), facecolor="black")
    tm = np.arange(T_FRAMES) * FRAME_S / 60                          # minutes since the recording's first frame
    cut = np.flatnonzero(np.diff(cond_all)) + 1
    blocks = list(zip(np.r_[0, cut], np.r_[cut, T_FRAMES] - 1))
    sc, cur, side_sc = [], [], []
    for j, p in enumerate(panels):
        x0, w = j / n, 1.0 / n
        two_ = p.get("band") is not None                               # Cedric, 2026-10-07: the brain mean alone above
        P = p["P"]
        lo, hi = np.percentile(P[:, :2], [0.2, 99.8], 0)               # stray points do not set the frame
        pad = 0.03 * (hi - lo).max()
        if two_:
            # TOP AND SIDE AT ONE SCALE (Cedric, 2026-10-07: "the top and side views are not to scale"): um per inch the
            # same in both, the side view right under the top one, both centred in the fish's half
            zz_ = P[:, 2] * p.get("side_flip", 1.0)
            zl_, zh_ = np.percentile(zz_, [0.2, 99.8])
            dz_ = zh_ - zl_
            yb_ = zl_ - 0.16 * dz_                                     # the 100-um bar's height, under the side view
            ex_, ey_, ez_ = hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, 1.27 * dz_
            FH_ = 9.0 if tall_ else 6.6
            sc_ = min(d["fig_w"] * (w - 0.02) / ex_, (0.60 * FH_) / (ey_ + ez_))
            wt_, ht_, hs_ = ex_ * sc_ / d["fig_w"], ey_ * sc_ / FH_, ez_ * sc_ / FH_
            xc_ = x0 + 0.01 + (w - 0.02 - wt_) / 2
            ax = fig.add_axes([xc_, 0.925 - ht_, wt_, ht_])
        else:
            ax = fig.add_axes([x0 + 0.01, 0.31, w - 0.02, 0.56])   # the band strip
        ax.set_facecolor("black")
        ax.axis("off")
        o = p["order"]
        sc.append(ax.scatter(P[o, 0], P[o, 1], c=np.zeros(len(o)), s=0.5 if len(o) < 90000 else 0.35, cmap="inferno",
                             vmin=0, vmax=vmax, linewidths=0))
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        if not two_:
            ax.set_aspect("equal")                                     # the box shrinks to the brain, centred
        if not two_:                      # with a side view the bar goes under it (Cedric, 2026-10-07)
            ax.plot([hi[0] - 100, hi[0]], [lo[1] - 0.4 * pad] * 2, color="white", lw=1.5)
            ax.text(hi[0] - 50, lo[1] + 0.4 * pad, "100 µm", color="0.7", fontsize=8, ha="center", va="bottom")
        fig.text(x0 + 0.02, 0.975, p.get("short", p["label"]), color="white", fontsize=16, va="top")   # Cedric, 2026-10-07
        if two_:                          # Cedric, 2026-10-07: the side view under the top view, tail up and head down
            axs_ = fig.add_axes([xc_, 0.925 - ht_ - 0.01 - hs_, wt_, hs_])   # the top view's scale
            axs_.set_facecolor("black")
            axs_.axis("off")
            os_ = np.argsort(P[:, 1])
            side_sc.append((axs_.scatter(P[os_, 0], zz_[os_], c=np.zeros(len(os_)), s=0.35, cmap="inferno", vmin=0,
                                         vmax=vmax, linewidths=0), os_))
            axs_.set_xlim(lo[0] - pad, hi[0] + pad)
            axs_.plot([hi[0] - 100, hi[0]], [yb_, yb_], color="white", lw=1.5)   # the 100-um bar below the side view
            axs_.text(hi[0] - 104, yb_, "100 µm", color="0.7", fontsize=9, ha="right", va="center")
            axs_.set_ylim(yb_ - 0.06 * dz_, zh_ + 0.05 * dz_)  # its extent ez_ exactly: the same um per inch as above
        else:
            side_sc.append(None)
        strips = ([("mean", [x0 + 0.05 * w, 0.175, 0.90 * w, 0.085]), ("band", [x0 + 0.05 * w, 0.045, 0.90 * w, 0.095])]
                  if two_ else [("one", [x0 + 0.05 * w, 0.075, 0.90 * w, 0.14])])
        for k2_, (kind_, rect_) in enumerate(strips):
            m = fig.add_axes(rect_)
            m.set_facecolor("black")
            for k_, sp in m.spines.items():
                sp.set_visible(k_ in ("left", "bottom"))
                sp.set_color("0.5")
            for i_, (a_, b_) in enumerate(blocks):
                m.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0, zorder=0)
                if k2_ == 0:
                    m.text((tm[a_] + tm[b_]) / 2, 1.02, NAMES[cond_all[a_]], color="0.75", fontsize=7, ha="center",
                           va="bottom", transform=m.get_xaxis_transform())
            if kind_ == "band":           # the range of the neurons' activity (Cedric, 2026-10-06): per frame, the
                m.fill_between(tm, p["band"][0], p["band"][1], color="white", alpha=0.22, lw=0, zorder=1)  # 2.5-97.5th pct
            m.plot(tm, p["mean"], color="#2ca02c", lw=0.6, zorder=2)
            lo_, hi_ = np.nanpercentile(p["band"] if kind_ == "band" else p["mean"], [0.5, 99.5])
            m.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
            m.set_xlim(tm[0], tm[-1])
            m.tick_params(colors="0.6", labelsize=7, length=2)
            if kind_ == "mean":
                m.set_xticklabels([])
            cur.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=4))
        if two_:
            fig.text(x0 + 0.05 * w, 0.282, "brain-mean dF/F over the whole recording", color="0.7", fontsize=9, va="bottom")
            fig.text(x0 + 0.05 * w, 0.143, "the same with the band 95 % of the neurons; time, min", color="0.7", fontsize=9,
                     va="bottom")
        else:
            fig.text(x0 + 0.05 * w, 0.255, "brain-mean dF/F over the whole recording; time, min", color="0.7",
                     fontsize=8, va="bottom")
    t_txt = fig.text(0.02, 0.935, "", color="0.7", fontsize=11, va="top")
    for k in ks:
        f = int(frames[k])
        for s_, ss_, p in zip(sc, side_sc, panels):
            s_.set_array(np.asarray(p["X"][k], np.float32)[p["order"]])
            if ss_ is not None:
                ss_[0].set_array(np.asarray(p["X"][k], np.float32)[ss_[1]])
        t_txt.set_text(f"condition: {NAMES[cond_all[f]]}   t = {tm[f]:5.1f} min (frame {f:,} of {T_FRAMES:,})")
        for c_ in cur:
            c_.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=d["dpi"], facecolor="black")
    plt.close(fig)


def band95(X, chunk=500):
    """Per frame, the 2.5th and 97.5th percentiles of every neuron's dF/F: the band 95 % of the neurons fall in. [2, T]"""
    return np.concatenate([np.percentile(np.asarray(X[a:a + chunk], np.float32), [2.5, 97.5], axis=1)
                           for a in range(0, X.shape[0], chunk)], axis=1)


def zapbench_panel(frames, P=None):
    """ZAPBench's release traces as a movie_brains panel: every frame read once for the brain mean, `frames` kept."""
    X = traces(0, T_FRAMES)
    P = centroids() if P is None else P
    return {"P": brain_view(P), "X": np.ascontiguousarray(X[frames]), "mean": X.mean(1), "band": band95(X),
            "label": f"ZAPBench release, {P.shape[0]:,} neurons, dF/F", "short": "ZAPBench"}


def destripe_panel(frames):
    """The destriped zap-inr traces (graphs_data/zebrafish/zapbench_destripe_recording.npz, mapped to ZAPBench's dF/F
    scale) as a movie_brains panel."""
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = z["dff"]
    return {"P": brain_view(z["pos_um"]), "X": np.ascontiguousarray(X[frames]), "mean": X.mean(1), "band": band95(X),
            "label": f"destriped (zap-inr), {X.shape[1]:,} neurons, mapped to ZAPBench's dF/F scale",
            "short": "destriped (zap-inr)"}


def movie_data(out_stem, frames=None):
    """Slide 2's movie: ZAPBench's release traces over the whole recording (movie_brains' look)."""
    frames = movie_frames() if frames is None else frames
    return movie_brains([zapbench_panel(frames)], out_stem + ".mp4", frames)


def movie_data_pair(out_mp4, frames=None):
    """THE TWO DATASETS SIDE BY SIDE (Cedric, 2026-10-01; redone 2026-10-02 over the whole recording): ZAPBench's
    release traces (left) and the destriped zap-inr traces mapped to ZAPBench's dF/F scale (right), the same ~800
    frames, one colour scale, each panel fitted to its own brain (the zap-inr anatomy frame is stretched against
    ZAPBench's segmentation frame, so one um scale would draw the destriped brain larger; each bar is in its own
    panel's um). Run: `python tools/exp17_slides.py --pair`."""
    frames = movie_frames() if frames is None else frames
    movie_brains([zapbench_panel(frames), destripe_panel(frames)], out_mp4, frames)
    return out_mp4


def movie_destripe_3d(out_mp4, t0=None, n=200, fps=25, opacity=0.4, tilt=40.0):
    """THE DESTRIPED TRACES IN 3-D (Cedric, 2026-10-01): every neuron a translucent point in its 3-D position (the
    zap-inr anatomy frame, um), coloured by its activity on the dF/F scale -- one oblique camera, the flash condition's
    frames as slide 2. VTK off-screen. Run: `python tools/exp17_slides.py --3d`."""
    import shutil
    import tempfile
    import pyvista as pv
    from plexus.paths import graphs_data_path
    pv.OFF_SCREEN = True
    off = condition_offsets()
    c = NAMES.index("flash")
    t0 = off[c] + 20 if t0 is None else t0
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    p = z["pos_um"].astype(np.float32)
    X = np.asarray(z["dff"][t0:t0 + n])
    Q = p - np.percentile(p, 50, 0)
    vmax = float(np.percentile(X, 95.0))
    cloud = pv.PolyData(Q)
    cloud.point_data["a"] = X[0]
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("black")
    pl.add_mesh(cloud, scalars="a", cmap="inferno", clim=(0, vmax), point_size=2.5, opacity=opacity,
                render_points_as_spheres=False, show_scalar_bar=False)
    ext = float(np.ptp(np.percentile(Q, [1, 99], 0), 0).max())
    # OBLIQUE FROM ABOVE: in the zap-inr anatomy frame the head points to -x and the dorsal side to -z (the -z view
    # matches the dorsal 2-D movies); the camera sits over the brain, tilted `tilt` degrees back toward the tail
    r, tl = 2.6 * ext, np.deg2rad(tilt)              # the whole brain inside VTK's 30-degree view angle
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (-1.0, 0.0, 0.0)
    pl.camera.position = (r * np.sin(tl), 0.0, -r * np.cos(tl))
    txt = pl.add_text("", position="upper_left", font_size=11, color="white")
    pl.add_text(f"destriped (zap-inr), {len(Q):,} neurons, 3-D, mapped to dF/F", position="upper_right", font_size=10,
                color="lightgray")
    tmp = tempfile.mkdtemp(prefix="ds3d_")
    for k in range(n):
        cloud.point_data["a"] = X[k]
        txt.SetText(2, f"flash   t = {(t0 + k) * FRAME_S:7.1f} s (frame {t0 + k})")
        pl.render()
        pl.screenshot(os.path.join(tmp, f"{k:05d}.png"))
    pl.close()
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", out_mp4],
                   check=True)
    shutil.copy(os.path.join(tmp, f"{n // 2:05d}.png"), out_mp4.replace(".mp4", ".png"))
    shutil.rmtree(tmp)
    return out_mp4


def movie_note(frames=None) -> list:
    """The data movies' sampling as slide rows, from movie_frames() (no movie needed to write it)."""
    fr = movie_frames() if frames is None else frames
    step = (fr[-1] - fr[0]) / max(len(fr) - 1, 1)
    return [("movie", f"the whole recording: {len(fr)} of its {T_FRAMES:,} frames,"),
            ("", f"one every {step:.1f} frames ({step * FRAME_S:.1f} s); below it the"),
            ("", f"brain-mean dF/F, the {len(NAMES)} conditions as grey blocks")]


def slide_destripe_data():
    """The twin of slide 2 on the DESTRIPED traces (zap-inr, graphs_data/zebrafish/zapbench_destripe_recording.npz,
    tools/export_destripe_recording.py): the same frames of the whole recording, the same movie look."""
    from plexus.paths import graphs_data_path
    f = graphs_data_path("zebrafish", "zapbench_destripe_recording.npz")
    if not os.path.exists(f):
        return ("10_destripe_data", "")
    prov = json.load(open(f.replace(".npz", ".json")))
    stem = os.path.join(PRES, "Movies", "10_destripe_data")
    if "--movie" in sys.argv or not os.path.exists(stem + ".mp4"):
        fr = movie_frames()
        movie_brains([destripe_panel(fr)], stem + ".mp4", fr)
    pos = np.load(f)["pos_um"]
    ext = np.ptp(pos, 0)
    right = (head("the destriped recording") + rows([
        ("neurons", f"{prov['neurons']:,} (zap-inr fit; {prov['failed_fits']['neurons']:,} failed fits clamped)"),
        ("frames", f"{prov['frames']:,}, the same session as ZAPBench"),
        ("brain", f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} \\textmu m (anatomy frame)"),
        ("state", f"$x = {prov['alpha']:.4f}\\,a {prov['beta']:+.4f}$, one value per neuron"),
        ("", "$a$: the zap-inr amplitude (photo-electrons)"),
        ("", "one linear map, matched to ZAPBench's dF/F: mean, sd")])
        + head("against ZAPBench's dF/F") + rows([
            ("mean, sd", f"{prov['mean_sd']['destripe'][0]:.3f}, {prov['mean_sd']['destripe'][1]:.3f} vs "
                         f"{prov['mean_sd']['zapbench'][0]:.3f}, {prov['mean_sd']['zapbench'][1]:.3f}"),
            ("median", f"{prov['quantiles']['destripe'][3]:.3f} vs {prov['quantiles']['zapbench'][3]:.3f}"),
            ("99th pct", f"{prov['quantiles']['destripe'][6]:.3f} vs {prov['quantiles']['zapbench'][6]:.3f}"),
            ("low end", f"the 0.1 amplitude floor $\\to$ {prov['quantiles']['destripe'][0]:.3f}")])
        + head("stimulus, task") + rows([("", "the release's stimulus and conditions, unchanged"),
                                          ] + movie_note()))
    return ("10_destripe_data", frame("The destriped traces: the same brain, the same stimuli",
                                      "\\playmovie{Movies/10_destripe_data}", right, "zapbench_destripe_recording.npz",
                                      deck_title="destriped ZAPBench traces (zap-inr)"))


def op_mesh(P, L0=MESH_L0_UM, levels=MESH_LEVELS, mirror=0):
    """The mesh exactly as the law builds it (`state_diffuse[model: graphcast].mesh`, lattice vertices, nested levels,
    mirror-symmetric across the midline): node positions, each multi-mesh edge's level (its length is 2^k L0) and
    each level's node and edge counts."""
    import torch
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    op = get_contract("state_diffuse").implementations["graphcast"](
        {"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": L0, "mesh_levels": levels,
         "mesh_mirror_axis": mirror})
    G = op.mesh(torch.as_tensor(P, dtype=torch.float32))
    mm_s, mm_r, f = (x.numpy() for x in G["mm"])
    lev = np.clip(np.round(np.log2(np.maximum(f[:, 3], 1e-9))), 0, levels - 1).astype(int)
    return {"G": G, "C": G["centre_um"], "mm": (mm_s, mm_r, lev), "level_nodes": G["level_nodes"],
            "nodes_per_level": G["stats"]["nodes_per_level"], "edges_per_level": [int((lev == k).sum()) for k in range(levels)],
            "stats": G["stats"], "L0": L0, "mid": G["lattice"]["mid_um"]}


def render_multimesh_3d(P, M, png, mp4=None, n_frames=200, fps=25, labels=True, window=None):
    """The multi-mesh in 3-D, to scale in um (VTK, off-screen): the neurons a faint cloud, each level's edges in its
    own colour, wider as the level coarsens, so the long edges that cross the brain in one message stand out over
    the fine lattice. A still at an oblique view (`png`) and, when `mp4` is given, a full turn about the vertical."""
    import shutil
    import tempfile
    import pyvista as pv
    pv.OFF_SCREEN = True
    C = M["C"]
    mm_s, mm_r, lev = M["mm"]
    if window is not None:          # a cube (centre, half-width um): the slide-3 window turning (Cedric, 2026-10-04)
        c0_, h_ = window
        inw = np.all(np.abs(C - c0_) <= h_ + 1e-6, axis=1)
        k_ = inw[mm_s] & inw[mm_r]
        mm_s, mm_r, lev = mm_s[k_], mm_r[k_], lev[k_]
        P = P[np.all(np.abs(P - c0_) <= h_, axis=1)]
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d", "#ffffff"][:len(M["nodes_per_level"])]
    width = [1.0, 1.8, 3.0, 4.5, 7.0]
    alpha = M.get("alpha") or [0.18, 0.45, 0.85, 1.0, 1.0]       # M["alpha"]: lighter fine levels behind a focus star
    mid = (P.max(0) + P.min(0)) / 2
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pl.set_background("black")
    pl.add_mesh(pv.PolyData((P[::3] - mid).astype(np.float32)), color="#777777", point_size=1.0, opacity=0.12,
                render_points_as_spheres=False)
    for k in range(len(cols)):
        sel = np.where(lev == k)[0]
        if not len(sel):
            continue
        a, b = C[mm_s[sel]] - mid, C[mm_r[sel]] - mid
        pts = np.concatenate([a, b]).astype(np.float32)
        n = len(sel)
        lines = np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()
        pl.add_mesh(pv.PolyData(pts, lines=lines), color=cols[k], line_width=width[k], opacity=alpha[k])
        if k >= 2:
            used = M["level_nodes"][k]
            if window is not None:
                used = used[inw[used]]
            pl.add_mesh(pv.PolyData((C[used] - mid).astype(np.float32)), color=cols[k], point_size=4 + 3 * k,
                        render_points_as_spheres=True)
    if M.get("focus") is not None:  # (node, [(senders, colour)]): one node's incoming edges at every level, heavy --
        c_, fl_ = M["focus"]        # the multi-mesh message passing into one node (GraphCast Fig. 1e)
        for s_, col_ in fl_:
            if len(s_):
                n = len(s_)
                pts = np.concatenate([C[s_] - mid, np.repeat(C[c_:c_ + 1] - mid, n, 0)]).astype(np.float32)
                pl.add_mesh(pv.PolyData(pts, lines=np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()),
                            color=col_, line_width=6.0, opacity=1.0)
        pl.add_mesh(pv.PolyData((C[c_:c_ + 1] - mid).astype(np.float32)), color="white", point_size=22,
                    render_points_as_spheres=True)
    if labels:                      # M["labels"]: [(text, colour)] of a mesh with its own levels (the neuron mesh)
        for k, (t_, c_) in enumerate(M.get("labels") or [
                (f"level {k}: {M['L0'] * 2 ** k:.0f} um edges, {M['nodes_per_level'][k]:,} nodes, "
                 f"{M['edges_per_level'][k]:,} edges", cols[k]) for k in range(len(cols))]):
            pl.add_text(t_, position=(20, 960 - 28 * k), font_size=11, color=c_)
    ext = np.ptp(P, 0)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (0.0, 0.0, 1.0)
    r = 1.9 / 0.9 * float(ext.max()) / M.get("zoom", 1.0)   # dezoomed by 0.9 (Cedric): the whole mesh, uncropped
    if window is not None:
        pl.camera.focal_point = tuple((np.asarray(window[0]) - mid).tolist())

    f_ = np.asarray(pl.camera.focal_point)

    def cam(az):
        el = np.deg2rad(32.0)
        pl.camera.position = tuple(f_ + np.array([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]))
        pl.render()                 # off-screen VTK keeps the first view without it: the static turntables, 2026-10-04
    cam(np.deg2rad(-60.0))
    pl.screenshot(png)
    if mp4 and not os.path.exists(mp4):            # a turntable is rendered once (VTK, 200 frames) and reused
        tmp = tempfile.mkdtemp(prefix="multimesh_")
        for i in range(n_frames):
            cam(np.deg2rad(-60.0) + 2 * np.pi * i / n_frames)
            pl.screenshot(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4],
                       check=True)
        if os.path.abspath(png) != os.path.abspath(mp4.replace(".mp4", ".png")):
            shutil.copy(png, mp4.replace(".mp4", ".png"))
        shutil.rmtree(tmp)
    pl.close()


STEP_TEXT = {
    "graphcast": {"t1": "1  encode: neurons -> cube corners", "t2": "2  process: the multi-mesh GNN",
                  "t3": "3  decode: cube corners -> neurons",
                  "c2": "8 layers of message passing on ONE graph\nedges 16 / 32 / 64 / 128 um (coarse ones as arcs)",
                  "c3": "the 8 corners of its cube (4 hidden behind)\n-> delta_i;  x_i(t+1) = x_i(t) + delta_i",
                  "parts": [("dF/F at t-5 .. t (6)", 6, "#e07b39"), ("position (3)", 3, "0.6"),
                            ("embedding a_i (8): learned, one per neuron", 8, "#f7c200"),
                            ("stimulus u (22 x 7): the same for every neuron", 22 * 7, "#6a51a3")]},
    "connectome": {"t1": "1  neurons -> mesh: the mean in reach", "t2": "2  the mesh: M substeps, one W per edge",
                   "t3": "3  mesh -> neurons: the 8 corners' mean",
                   "c2": "v_k += (1/M) f_theta(v_k, a_k, sum_l W_lk g_phi(v_l, a_l)^2)\nM = 8 substeps per frame; edge colour = W",
                   "c3": "m_i = mean of the 8 corners (4 hidden behind)\ndx_i = f_n(x_i history, e_i, m_i, u)",
                   "parts": [("dF/F at t-5 .. t (6)", 6, "#e07b39"), ("embedding e_i (2): learned, one per neuron", 2, "#f7c200"),
                             ("mesh input m_i (1)", 1, "#fd8d3c"), ("stimulus u (22 x 7)", 22 * 7, "#6a51a3")]},
    "known_ode": {"t1": "1  neurons -> mesh: the mean in reach", "t2": "2  the mesh: a leaky ODE, one W per edge",
                  "t3": "3  mesh -> neurons: the 8 corners' mean",
                  "c2": "v_k += (1/M) (-v_k + V_k + msg_k) / tau_k;  current: W_lk tanh(v_l)\n"
                        "conductance: W_lk^2 relu(v_l) (E_l - v_k); edge colour = W",
                  "c3": "m_i = mean of the 8 corners (4 hidden behind)\ndz_i = (-z_i + V_i + g_i m_i + B_i . u) / tau_i",
                  "parts": [("dF/F now (1)", 1, "#e07b39"), ("tau_i, V_i, g_i (3): learned, per neuron", 3, "#f7c200"),
                            ("mesh input m_i (1)", 1, "#fd8d3c"), ("B_i . u(t) (22): stimulus weights, per neuron", 22, "#6a51a3")]},
}


def step_centre(P, M) -> np.ndarray:
    """The mesh node the one-step pictures centre on: a node of the coarsest level nearest the brain's median, so
    every level's edges lie in its one-cube-deep slab."""
    C, top_ = M["C"], M["level_nodes"][-1]
    return C[int(top_[np.argmin(np.linalg.norm(C[top_] - np.median(P, 0), axis=1))])]


def draw_process(b, P, M, c0, w2, kind="graphcast", scale=1.0, reach=None, nsize=None, upto=None, cols=None, arcs=True):
    """The '2 process' panel of figure_step (and, alone and large, of figure_multimesh_gnn): a window 2 w2 um wide,
    one fine cube deep, seen from above -- the neurons in it as grey dots, the finest mesh's edges as straight blue
    lines, the coarser levels' edges as arcs in their own colour (they run along the fine lattice lines, so straight
    they would hide under it) and their nodes as dots. `scale` multiplies the line widths and dot sizes; `reach` (um,
    default w2) the half-width within which mesh nodes are drawn, the view widened to it; `upto` the last level drawn
    (-1: the neurons alone; None: every level) -- the build-up movie of slide 4 (Cedric, 2026-10-05); `cols` the levels'
    colours (default LEVEL_COLS), `arcs` False draws the coarse levels straight (exp17's second deck, 2026-10-06)."""
    from matplotlib.collections import LineCollection
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    C, L = M["C"], M["L0"]
    mm_s, mm_r, lev = M["mm"]
    r_ = w2 if reach is None else reach
    v_ = r_ if reach is None else r_ + 0.6 * L          # a widened view keeps the arcs on its edge nodes whole
    b.set_xlim(c0[0] - v_, c0[0] + v_)
    b.set_ylim(c0[1] - v_, c0[1] + v_)
    slab = lambda Q: np.abs(Q[:, 2] - c0[2]) < 0.55 * L                       # noqa: E731
    in2 = (np.abs(C[:, 0] - c0[0]) < r_) & (np.abs(C[:, 1] - c0[1]) < r_) & slab(C)
    if nsize is None:
        b.scatter(P[::2][slab(P[::2]), 0], P[::2][slab(P[::2]), 1], s=1.5 * scale, c="0.3", linewidths=0, zorder=1)
    else:                                           # every neuron of the slab, at a readable size (Cedric, 2026-10-02)
        b.scatter(P[slab(P), 0], P[slab(P), 1], s=nsize, c="0.6", linewidths=0, zorder=1)
    cols = cols or LEVEL_COLS
    n_lv = min(4, len(M["nodes_per_level"])) if upto is None else min(4, len(M["nodes_per_level"]), upto + 1)
    for k in range(n_lv):
        sel = (lev == k) & in2[mm_s] & in2[mm_r] & (mm_s < mm_r)
        if k == 0:
            if kind == "graphcast":
                b.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1), colors=cols[0],
                                                linewidths=0.7 * scale, alpha=0.7))
            else:                                   # one weight per edge (its random start): red > 0, blue < 0
                wv = np.random.default_rng(0).standard_normal(int(sel.sum()))
                b.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1),
                                                colors=np.where(wv[:, None] > 0, [[0.9, 0.3, 0.3, 0.8]], [[0.3, 0.5, 1.0, 0.8]]),
                                                linewidths=0.3 + 1.2 * np.minimum(np.abs(wv), 2)))
            continue
        if not arcs:                                # straight, over the finer lines they run along
            b.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1), colors=cols[k],
                                            linewidths=(0.8 + 0.7 * k) * scale, alpha=0.95, zorder=2 + k))
            continue
        for a_, b_ in zip(C[mm_s[sel], :2], C[mm_r[sel], :2]):
            d = b_ - a_
            nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            ctrl = (a_ + b_) / 2 + nrm * 0.18 * np.linalg.norm(d)
            b.add_patch(PathPatch(Path([a_, ctrl, b_], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), facecolor="none",
                                  edgecolor=cols[k], lw=(0.8 + 0.7 * k) * scale, alpha=0.95))
    lvl_nodes = M["level_nodes"]
    for k in range(1, min(4, len(lvl_nodes), n_lv)):
        nk = lvl_nodes[k][in2[lvl_nodes[k]]]
        b.scatter(C[nk, 0], C[nk, 1], s=(8 + 8 * k) * scale ** 2, c=cols[k], linewidths=0, zorder=4)


LEVEL_COLS = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d"]   # the multi-mesh levels 0-3 (16, 32, 64, 128 um edges)


def scale_bar(ax, um, fontsize=10, lw=3.0):
    """A white scale bar `um` micrometres long in the lower left of a data-in-um axes (Cedric, 2026-10-05: "add a
    scale bar"), its length written above it."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xa, ya = x0 + 0.03 * (x1 - x0), y0 + 0.025 * (y1 - y0)
    ax.plot([xa, xa + um], [ya, ya], color="white", lw=lw, solid_capstyle="butt", zorder=20)
    ax.text(xa + um / 2, ya + 0.012 * (y1 - y0), f"{um:g} µm", color="white", fontsize=fontsize, ha="center",
            va="bottom", zorder=20, bbox=dict(facecolor="black", edgecolor="none", alpha=0.75, pad=1.5))


def figure_transfer(P, M, path, t=2600, upto=None, stage=None):
    """NEURONS TO THE MESH AND BACK (Cedric, 2026-10-02: "explain better neuron to grid, grid to neuron"): two panels
    on the same window of real neurons (a slab one fine cube deep, from above), larger than the one-step figure's --
    a: grid2mesh, every neuron's edges to the cube corners within sqrt(3)/2 L0 (one neuron's drawn thick); b: mesh2grid,
    every neuron's edges from the 8 corners of its cube (the same neuron's thick). Dots: neurons by dF/F at frame t.
    `upto` the build-up movie's stage (Cedric, 2026-10-06, as slide 4): 0 the neurons, 1 + the fine grid and its nodes,
    2 + every neuron's edges, 3 (None) + the circled neuron's own; `stage` a line naming it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    G, C, L = M["G"], M["C"], M["L0"]
    g_s, g_r, _ = (x.numpy() for x in G["g2m"])
    m_s, m_r, _ = (x.numpy() for x in G["m2g"])
    X = traces(t, t + 1)[0]
    c0 = step_centre(P, M)
    w = 1.6 * L
    inw = lambda Q: (np.abs(Q[:, 0] - c0[0]) < w) & (np.abs(Q[:, 1] - c0[1]) < w) & (np.abs(Q[:, 2] - c0[2]) < 0.55 * L)  # noqa: E731
    sp, sm = inw(P), inw(C)
    ids = np.where(sp)[0]
    one = int(ids[np.argmin(np.linalg.norm(P[ids] - (c0 + [0.35 * L, 0.3 * L, 0]), axis=1))])
    vmax = float(np.percentile(X, 97))
    fig = plt.figure(figsize=(12.0, 6.6), facecolor="black")
    n_a, n_b = int((g_s == one).sum()), int((m_r == one).sum())    # Cedric, 2026-10-06: "3 left, 4 right?" -- by design
    for j, (title, sub) in enumerate((
            ("a   neurons -> mesh (grid2mesh)", f"each neuron sends to the cube corners within {math.sqrt(3) / 2 * L:.1f} µm\n"
             f"(the circled one: {n_a})"),
            ("b   mesh -> neurons (mesh2grid)", f"each neuron receives from the 8 corners of its cube\n(the circled one: {n_b}; "
             "from above they stack in pairs: 4 points)"))):
        ax = fig.add_axes([0.02 + 0.49 * j, 0.10, 0.46, 0.74])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_xlim(c0[0] - w, c0[0] + w)
        ax.set_ylim(c0[1] - w, c0[1] + w)
        ax.set_aspect("equal")
        fig.text(0.02 + 0.49 * j, 0.965, title, color="white", fontsize=13, va="top")
        fig.text(0.02 + 0.49 * j, 0.915, sub, color="0.75", fontsize=10, va="top")
        if j == 0:
            sel, k1, col = sp[g_s] & sm[g_r], g_s == one, "#3182bd"
            seg_all = np.stack([P[g_s[sel], :2], C[g_r[sel], :2]], 1)
            seg_one = np.stack([P[g_s[k1], :2], C[g_r[k1], :2]], 1)
        else:
            sel, k1, col = sp[m_r] & sm[m_s], m_r == one, "#e0445c"
            seg_all = np.stack([C[m_s[sel], :2], P[m_r[sel], :2]], 1)
            seg_one = np.stack([C[m_s[k1], :2], P[m_r[k1], :2]], 1)
        # the fine grid, level 0 of the multi-mesh, drawn as on slide 4 (Cedric, 2026-10-06: "coherent with slide 4")
        st_ = 3 if upto is None else upto
        if st_ >= 1:
            mm_s, mm_r, lev = M["mm"]
            g0 = (lev == 0) & sm[mm_s] & sm[mm_r] & (mm_s < mm_r)
            ax.add_collection(LineCollection(np.stack([C[mm_s[g0], :2], C[mm_r[g0], :2]], 1), colors=LEVEL_COLS[0],
                                             linewidths=1.4, alpha=0.7, zorder=0))
            ax.scatter(C[sm, 0], C[sm, 1], s=140, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
        if st_ >= 2:
            ax.add_collection(LineCollection(seg_all, colors=col, linewidths=0.6, alpha=0.45))
        if st_ >= 3:
            ax.add_collection(LineCollection(seg_one, colors=col, linewidths=3.0))
            ax.scatter([P[one, 0]], [P[one, 1]], s=320, facecolors="none", edgecolors="white", linewidths=2, zorder=5)
        ax.scatter(P[sp, 0], P[sp, 1], s=60, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
        scale_bar(ax, 10, fontsize=11)                  # Cedric, 2026-10-05
    if stage:
        fig.text(0.5, 0.045, stage, color="white", fontsize=12, ha="center", va="bottom")
    fig.text(0.99, 0.01, f"a {2 * w:.0f}-µm window, one cube ({L:.0f} µm) deep, from above; dots: ZAPBench neurons by dF/F; orange: "
             "level-0 nodes (cube corners); light blue: the level-0 grid; circled: one neuron and its own edges",
             color="0.55", fontsize=8.5, ha="right")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"window_um": 2 * w, "radius_um": math.sqrt(3) / 2 * L,
            "g2m_one": int((g_s == one).sum()), "m2g_one": int((m_r == one).sum())}


def figure_multimesh_gnn(P, M, path, half_cubes=8.0, upto=None, stage=None, cols=None, arcs=True):
    """THE '2 PROCESS' PANEL ALONE (Cedric, 2026-10-02: isolated from the one-step figure, large): the multi-mesh GNN
    in a window 2 x `half_cubes` fine cubes wide, one cube deep, seen from above, with a legend of the levels. `upto`
    the last level drawn (-1 the neurons alone), its legend lines dimmed past it; `stage` a line naming the step (the
    build-up movie, Cedric 2026-10-05)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = M["L0"]
    c0 = step_centre(P, M)
    w2 = half_cubes * L
    fig = plt.figure(figsize=(8.0, 8.4), facecolor="black")
    ax = fig.add_axes([0.02, 0.12, 0.96, 0.78])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_aspect("equal")
    # the nodes ON the window's edge drawn too (reach a quarter cube past it): the 128-um edges from the centre node
    # end exactly there, and a strict cut hid every one of them
    draw_process(ax, P, M, c0, w2, "graphcast", scale=2.0, reach=w2 + 0.25 * L, nsize=22.0, upto=upto, cols=cols, arcs=arcs)
    scale_bar(ax, 50, fontsize=11)
    fig.text(0.03, 0.975, f"the multi-mesh GNN: a {2 * w2:.0f}-µm window, one {L:.0f}-µm cube deep, from above",
             color="white", fontsize=12, va="top")
    fig.text(0.03, 0.94, stage or "grey dots: the ZAPBench neurons in the slab (their soma centres); all edges are ONE "
             "graph, every level used in every layer", color="white" if stage else "0.7", fontsize=9.5 + 1.5 * bool(stage),
             va="top")
    n_lev = min(4, len(M["nodes_per_level"]))
    for k in range(n_lev):
        x, y = 0.03 + 0.48 * (k % 2), 0.085 - 0.035 * (k // 2)
        c_ = (cols or LEVEL_COLS)[k] if upto is None or k <= upto else "0.25"
        fig.lines.append(matplotlib.lines.Line2D([x, x + 0.05], [y, y], transform=fig.transFigure,
                                                 color=c_, lw=1.5 + 1.4 * k))
        fig.text(x + 0.065, y, f"level {k}: {L * 2 ** k:.0f}-µm edges, {M['edges_per_level'][k]:,} in the brain",
                 color=c_, fontsize=10, va="center")
    fig.text(0.03, 0.015, ("levels 1-3 drawn as arcs (they run along the fine lattice lines), their nodes as dots" if arcs else
                           "the coarser levels drawn straight and wider, over the finer lines they run along; their nodes as dots"),
             color="0.55", fontsize=8.5, va="bottom")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"window_um": 2 * w2, "L0": L}


def _concat_movie(frames, stem, hold_s, last_s, poster="last"):
    """Hold each PNG `hold_s` s (the last `last_s` s) into <stem>.mp4; the last PNG becomes the poster <stem>.png, or
    the first with poster="first" (the still shown before the movie starts: exp17's second deck, 2026-10-06)."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        lst = "".join(f"file '{f}'\nduration {last_s if i == len(frames) - 1 else hold_s}\n" for i, f in enumerate(frames))
        open(os.path.join(td, "list.txt"), "w").write(lst + f"file '{frames[-1]}'\n")    # concat: the last file twice
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(td, "list.txt"),
                        "-vf", "fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        stem + ".mp4"], check=True)
    shutil.copy(frames[0] if poster == "first" else frames[-1], stem + ".png")


def movie_transfer_build(P, M, stem, hold_s=1.5, last_s=3.0):
    """SLIDE 5 BUILT UP, as slide 4 (Cedric, 2026-10-06): figure_transfer at its 4 stages -- the neurons, + the fine grid
    and its nodes, + every neuron's edges to / from the corners, + one neuron's own; <stem>.mp4 and its poster."""
    import tempfile
    if not os.path.exists(FFMPEG):
        return False
    L = M["L0"]
    stages = ["1  the neurons: every ZAPBench neuron of the slab, coloured by its dF/F",
              f"2  + the fine grid: level 0, {L:.0f}-µm cubes, their corners the mesh nodes",
              "3  + the edges: a, each neuron to the corners within reach; b, each neuron from its cube's 8 corners",
              "4  + one neuron (circled) and its own edges"]
    with tempfile.TemporaryDirectory() as td:
        fs = []
        for i, txt in enumerate(stages):
            fs.append(os.path.join(td, f"s{i}.png"))
            figure_transfer(P, M, fs[-1], upto=i, stage=txt)
        _concat_movie(fs, stem, hold_s, last_s)
    return True


def movie_multimesh_build(P, M, stem, hold_s=1.5, last_s=3.0, cols=None, arcs=True, poster="last"):
    """SLIDE 4 BUILT UP (Cedric, 2026-10-05: "a small animation starting with neurons, fine mesh, middle mesh and large
    mesh"): figure_multimesh_gnn drawn 1 + levels times -- the neurons alone, then each level added in turn -- each
    held `hold_s` s, the whole graph `last_s` s; <stem>.mp4 and its poster <stem>.png (the whole graph). False without
    ffmpeg."""
    import tempfile
    if not os.path.exists(FFMPEG):
        return False
    L = M["L0"]
    n_lev = min(4, len(M["nodes_per_level"]))
    size = lambda k: "fine" if k == 0 else ("large" if k == n_lev - 1 else "middle")      # noqa: E731
    stages = [(-1, "1  the neurons: every ZAPBench neuron of the slab, at its soma centre")] + [
        (k, f"{k + 2}  + the {size(k)} mesh: level {k}, {L * 2 ** k:.0f}-µm edges") for k in range(n_lev)]
    with tempfile.TemporaryDirectory() as td:
        lst = []
        for i, (k, txt) in enumerate(stages):
            f = os.path.join(td, f"s{i}.png")
            figure_multimesh_gnn(P, M, f, upto=k, stage=txt, cols=cols, arcs=arcs)
            lst.append(f"file '{f}'\nduration {last_s if i == len(stages) - 1 else hold_s}\n")
        open(os.path.join(td, "list.txt"), "w").write("".join(lst) + f"file '{f}'\n")    # concat: the last file twice
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(td, "list.txt"),
                        "-vf", "fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        stem + ".mp4"], check=True)
        shutil.copy(os.path.join(td, "s0.png") if poster == "first" else f, stem + ".png")
    return True


def figure_step(P, M, path, t=2600, kind="graphcast"):
    """Slide 02a: ONE STEP of the law on real neurons, in a window of the brain (a slab one cell deep, seen from
    above), left to right -- 1 ENCODE: each neuron's input vector (its dF/F at t-5..t, its position, its learned
    EMBEDDING a_i, the stimulus u) becomes a latent sent to the cube corners within reach; 2 PROCESS: message
    passing on the multi-mesh, fine edges and the long edges of the coarser levels; 3 DECODE: each neuron receives
    from the 8 corners of its cube and becomes x(t+1) = x(t) + delta."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch, Rectangle
    G, C, L = M["G"], M["C"], M["L0"]
    g_s, g_r, _ = (x.numpy() for x in G["g2m"])
    m_s, m_r, _ = (x.numpy() for x in G["m2g"])
    X = traces(t, t + 1)[0]
    c0 = step_centre(P, M)                                               # every level's edges lie in its depth slab
    w = 2.2 * L
    inw = lambda Q, dz=0.55 * L: (np.abs(Q[:, 0] - c0[0]) < w) & (np.abs(Q[:, 1] - c0[1]) < w) & (np.abs(Q[:, 2] - c0[2]) < dz)  # noqa: E731
    sp, sm = inw(P), inw(C)
    ids = np.where(sp)[0]
    one = int(ids[np.argmin(np.linalg.norm(P[ids] - (c0 + [0.35 * L, 0.3 * L, 0]), axis=1))])
    vmax = float(np.percentile(X, 97))
    fig = plt.figure(figsize=(13.0, 7.0), facecolor="black")

    def panel(x0, title):
        ax = fig.add_axes([x0, 0.20, 0.30, 0.66])
        ax.set_facecolor("black")
        ax.axis("off")
        ax.set_xlim(c0[0] - w, c0[0] + w)
        ax.set_ylim(c0[1] - w, c0[1] + w)
        ax.set_aspect("equal")
        fig.text(x0 + 0.15, 0.95, title, color="white", fontsize=13, ha="center", va="top")
        return ax

    def arrow(x0, x1):
        fig.patches.append(FancyArrowPatch((x0, 0.53), (x1, 0.53), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=18, color="0.85", lw=1.5))
    # 1 ENCODE
    a = panel(0.01, STEP_TEXT[kind]["t1"])
    sel = sp[g_s] & sm[g_r]
    a.add_collection(LineCollection(np.stack([P[g_s[sel], :2], C[g_r[sel], :2]], 1), colors="#3182bd", linewidths=0.5,
                                    alpha=0.6))
    k1 = g_s == one
    a.add_collection(LineCollection(np.stack([P[g_s[k1], :2], C[g_r[k1], :2]], 1), colors="#6baed6", linewidths=2.5))
    a.scatter(P[sp, 0], P[sp, 1], s=16, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
    a.scatter(C[sm, 0], C[sm, 1], s=40, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
    a.scatter([P[one, 0]], [P[one, 1]], s=120, facecolors="none", edgecolors="white", linewidths=1.5, zorder=5)
    # the input vector of the highlighted neuron: blocks to scale, a legend line per block
    T_ = STEP_TEXT[kind]
    parts = T_["parts"]
    tot = sum(n for _, n, _ in parts)
    fig.text(0.02, 0.185, "what the circled neuron's update reads (widths to scale):", color="0.8", fontsize=9)
    x = 0.02
    for lab, n, col in parts:
        wd = 0.28 * n / tot
        fig.patches.append(Rectangle((x, 0.13), wd, 0.04, transform=fig.transFigure, facecolor=col, edgecolor="black"))
        x += wd
    for r_, (lab, n, col) in enumerate(parts):
        fig.patches.append(Rectangle((0.02, 0.095 - 0.027 * r_), 0.01, 0.018, transform=fig.transFigure, facecolor=col))
        fig.text(0.035, 0.104 - 0.027 * r_, lab, color=col, fontsize=8.5, va="center")
    arrow(0.315, 0.345)
    # 2 PROCESS -- a wider window, every level's edges: the coarse ones as arcs (they lie on the fine lattice lines)
    b = panel(0.35, STEP_TEXT[kind]["t2"])
    w2 = 8.0 * L
    draw_process(b, P, M, c0, w2, kind)
    fig.text(0.50, 0.17, STEP_TEXT[kind]["c2"],
             color="0.8", fontsize=9, ha="center", va="top")
    fig.text(0.50, 0.215, f"a {2 * w2:.0f}-um window", color="0.5", fontsize=8, ha="center")
    arrow(0.655, 0.685)
    # 3 DECODE
    c = panel(0.69, STEP_TEXT[kind]["t3"])
    k3 = m_r == one
    c.add_collection(LineCollection(np.stack([C[m_s[k3], :2], P[m_r[k3], :2]], 1), colors="#e0445c", linewidths=2.5))
    c.scatter(C[sm, 0], C[sm, 1], s=40, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
    c.scatter(P[sp, 0], P[sp, 1], s=16, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
    c.scatter([P[one, 0]], [P[one, 1]], s=120, facecolors="none", edgecolors="white", linewidths=1.5, zorder=5)
    fig.text(0.84, 0.17, STEP_TEXT[kind]["c3"],
             color="0.8", fontsize=9, ha="center", va="top")
    fig.text(0.99, 0.01, f"a {2 * w:.0f}-um window, one cube ({L:.0f} um) deep; dots: neurons coloured by dF/F; squares: mesh nodes",
             color="0.5", fontsize=8, ha="right")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


NG_COLS = {"short": "#9ecae1", "mid": "#fd8d3c", "long": "#ff4040"}


def neuron_graph_op(mid_um=32.0, long_um=128.0, positions_file="zebrafish/zapbench_recording.npz", **kw):
    """The operator itself (state_diffuse[neuron_graph]) builds the graph the slide draws: the picture IS the graph trained."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    C = get_contract("state_diffuse").implementations["neuron_graph"]
    return C({"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": positions_file,
              "inputs": 1, "short_k": 6, "mid_um": mid_um, "long_um": long_um, **kw})


def figure_neuron_graph(P, path, mp4=None, n_frames=200, fps=25, n_show=80, seed=0, op=None):
    """The neuron graph in 3-D, to scale in um (VTK, off-screen), black. a: the whole brain, the neurons a faint cloud
    and the middle and long edges INTO `n_show` random neurons (every edge would be a solid block); b: a 60-um box
    around one neuron, every short edge between the neurons inside it and that neuron's own 18 edges drawn thick.
    `mp4`: a full turn of panel a about the vertical."""
    import shutil
    import tempfile
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    op = op or neuron_graph_op()                         # P: the positions to DRAW (the graph's, maybe rotated)
    E = {k: (op._E[k][0].cpu().numpy(), op._E[k][1].cpu().numpy()) for k in op.EDGE_SETS}
    mid = (P.max(0) + P.min(0)) / 2
    Q = (P - mid).astype(np.float32)
    rng = np.random.default_rng(seed)
    if getattr(op, "graph_kind", "spatial") == "mesh":        # a mesh: the coarse levels' own neurons (exp17 batch 17)
        coarse = np.unique(np.concatenate([E["mid"][1], E["long"][1]]))
        show = set(rng.choice(coarse, min(n_show, len(coarse)), replace=False).tolist())
    else:
        show = set(rng.choice(len(P), n_show, replace=False).tolist())

    def lines(pl, s, r, col, w, a):
        if not len(s):
            return
        pts = np.concatenate([Q[s], Q[r]])
        n = len(s)
        pl.add_mesh(pv.PolyData(pts, lines=np.column_stack([np.full(n, 2), np.arange(n), np.arange(n) + n]).ravel()),
                    color=col, line_width=w, opacity=a)

    ext = np.ptp(P, 0)
    tmp = tempfile.mkdtemp(prefix="ngraph_")
    # a: the whole brain
    pa = pv.Plotter(off_screen=True, window_size=(1400, 1000))
    pa.set_background("black")
    pa.add_mesh(pv.PolyData(Q[::2]), color="#8a8a8a", point_size=1.5, opacity=0.3)
    for k, w, a in (("mid", 2.5, 0.95), ("long", 2.0, 0.85)):
        s, r = E[k]
        sel = np.fromiter((i in show for i in r), bool, len(r))
        lines(pa, s[sel], r[sel], NG_COLS[k], w, a)
    rec = np.array(sorted(show))
    pa.add_mesh(pv.PolyData(Q[rec]), color="white", point_size=7, render_points_as_spheres=True)
    pa.camera.focal_point = (0.0, 0.0, 0.0)
    pa.camera.up = (0.0, 0.0, 1.0)
    rr = 1.3 * float(ext.max())

    def cam(pl, az, r_, f=(0.0, 0.0, 0.0)):
        el = np.deg2rad(32.0)
        pl.camera.position = (f[0] + r_ * np.cos(el) * np.cos(az), f[1] + r_ * np.cos(el) * np.sin(az),
                              f[2] + r_ * np.sin(el))
        pl.render()                 # off-screen VTK keeps the first view without it: the static turntables, 2026-10-04
    cam(pa, np.deg2rad(-60.0), rr)
    fa = os.path.join(tmp, "a.png")
    pa.screenshot(fa)
    if mp4 and not os.path.exists(mp4):            # a turntable is rendered once and reused
        for i in range(n_frames):
            cam(pa, np.deg2rad(-60.0) + 2 * np.pi * i / n_frames, rr)
            pa.screenshot(os.path.join(tmp, f"f{i:05d}.png"))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4],
                       check=True)
        shutil.copy(fa, mp4.replace(".mp4", ".png"))
    pa.close()
    # b: one neuron's neighbourhood, a 60-um box (the long edges leave it: drawn to their far end)
    c = int(np.argmin(np.linalg.norm(Q - np.median(Q, 0), axis=1)))
    inbox = np.all(np.abs(Q - Q[c]) < 30.0, axis=1)
    pb = pv.Plotter(off_screen=True, window_size=(1000, 1000))
    pb.set_background("black")
    pb.add_mesh(pv.PolyData(Q[inbox]), color="#aaaaaa", point_size=5, render_points_as_spheres=True)
    s, r = E["short"]
    sel = inbox[s] & inbox[r]
    lines(pb, s[sel], r[sel], NG_COLS["short"], 1.0, 0.35)
    for k, w in (("short", 4.0), ("mid", 4.0), ("long", 4.0)):
        s, r = E[k]
        sel = r == c
        if not sel.any():                           # a mesh: the centre neuron may have no coarse edge
            continue
        lines(pb, s[sel], r[sel], NG_COLS[k], w, 1.0)
        pb.add_mesh(pv.PolyData(Q[s[sel]]), color=NG_COLS[k], point_size=12, render_points_as_spheres=True)
    pb.add_mesh(pv.PolyData(Q[c:c + 1]), color="white", point_size=18, render_points_as_spheres=True)
    pb.camera.focal_point = tuple(Q[c].tolist())
    pb.camera.up = (0.0, 0.0, 1.0)
    cam(pb, np.deg2rad(-60.0), 420.0, tuple(Q[c].tolist()))
    fb = os.path.join(tmp, "b.png")
    pb.screenshot(fb)
    pb.close()
    fig = plt.figure(figsize=(12, 5.4), facecolor="black")
    for j, (f, lab) in enumerate(((fa, f"a   the whole brain: middle and long edges into {n_show} of the neurons"),
                                  (fb, "b   one neuron (white), its 18 senders, the short edges around it"))):
        ax = fig.add_axes([0.0 if j == 0 else 0.58, 0.08, 0.58 if j == 0 else 0.42, 0.84])
        ax.imshow(plt.imread(f))
        ax.axis("off")
        fig.text(0.01 if j == 0 else 0.59, 0.955, lab, color="white", fontsize=10, va="top")
    st = op.graph_stats
    fig.text(0.01, 0.02, "   ".join(f"{k}: {st[k]['edges']:,} edges, {st[k]['per_element']:.1f} per neuron, "
                                    f"mean {st[k]['mean_um']:.1f} µm" for k in op.EDGE_SETS),
             color="0.75", fontsize=8)
    for k, x in zip(op.EDGE_SETS, (0.60, 0.72, 0.84)):
        fig.text(x, 0.085, k, color=NG_COLS[k], fontsize=11, weight="bold")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    shutil.rmtree(tmp)
    return op.graph_stats


def figure_latent_calcium(path, tau_s=2.0):
    """The forward model of the recording, black. a: the chain -- stimulus and neighbours drive a latent activity v,
    the indicator reads it out as the recorded dF/F c. b: one real neuron over 40 frames of the flash condition --
    its recorded c, the 6-frame context a rollout starts from, and the latent the kernel's exact one-step inverse
    implies (noisier: what the learned taps must temper). c: the indicator's response to a unit step of v."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    off = condition_offsets()
    c0 = off[NAMES.index("flash")] + 10
    X = traces(c0, c0 + 40)                                           # [40, N]
    sd = X.std(0)
    i = int(np.argsort(sd)[int(0.99 * len(sd))])                      # a responsive neuron, not the outlier at the top
    c = X[:, i].astype(np.float64)
    k = 1 - np.exp(-FRAME_S / tau_s)
    v = np.full_like(c, np.nan)
    v[:-1] = c[:-1] + (c[1:] - c[:-1]) / k                            # v(t) from c(t), c(t+1): the exact inverse
    fig = plt.figure(figsize=(12, 5.6), facecolor="black")
    # a: the chain
    ax = fig.add_axes([0.0, 0.52, 1.0, 0.40])
    ax.set_facecolor("black")
    ax.axis("off")
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 2)
    boxes = [(0.2, "stimulus u(t)\nand neighbours'\nlatent v_j", "0.75"),
             (3.1, "latent activity v_i\nknown ODE on the\nneuron graph", "#fd8d3c"),
             (6.0, "indicator\nH2B-GCaMP7f\nc += k (v - c)", "#9ecae1"),
             (8.4, "recorded\ndF/F  c_i\n(the loss)", "#2ca02c")]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.6, 1.3, boxstyle="round,pad=0.05", fc="black", ec=col, lw=1.8))
        ax.text(x + 0.8, 1.0, txt, color=col, ha="center", va="center", fontsize=10)
    for x0, x1 in ((1.85, 3.05), (4.75, 5.95), (7.65, 8.35)):
        ax.annotate("", (x1, 1.0), (x0, 1.0), arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
    ax.text(5.2, 0.05, "rollout start: v(t0) = sum_j a_j c(t0 - j), from the 6 recorded frames (learned taps)",
            color="0.7", ha="center", fontsize=9)
    fig.text(0.01, 0.955, "a   the forward model: the recording is the indicator's read-out of a latent activity",
             color="white", fontsize=11, va="top")
    # b: one neuron
    bx = fig.add_axes([0.06, 0.08, 0.58, 0.34])
    bx.set_facecolor("black")
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        bx.spines[sp].set_color("0.6")
    bx.tick_params(colors="0.8", labelsize=8)
    t = np.arange(40)
    bx.axvspan(14.5, 20.5, color="0.3", alpha=0.6, lw=0)
    bx.text(17.5, np.nanmax(v) * 0.98, "context\n6 frames", color="0.8", ha="center", va="top", fontsize=8)
    bx.plot(t, v, color="#fd8d3c", lw=1.2, label=f"latent v, the kernel's exact inverse (tau {tau_s:g} s)")
    bx.plot(t, c, color="#2ca02c", lw=1.8, label="recorded dF/F c")
    bx.set_xlabel(f"frame (0.914 s), flash condition, neuron {i + 1:,}", color="0.8", fontsize=9)
    bx.set_ylabel("dF/F", color="0.8", fontsize=9)
    bx.legend(frameon=False, labelcolor="white", fontsize=8, loc="upper right")
    fig.text(0.01, 0.46, "b   one neuron: the latent the recording implies is sharper, and noisier", color="white",
             fontsize=11, va="top")
    # c: the kernel
    cx = fig.add_axes([0.74, 0.08, 0.24, 0.34])
    cx.set_facecolor("black")
    for sp in ("top", "right"):
        cx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        cx.spines[sp].set_color("0.6")
    cx.tick_params(colors="0.8", labelsize=8)
    n = np.arange(12)
    cx.step(n, (n >= 1).astype(float), color="#fd8d3c", where="post", lw=1.2, label="v: a unit step")
    cx.plot(n, np.where(n >= 1, 1 - (1 - k) ** (n - 1 + 1) , 0.0), "o-", color="#9ecae1", ms=3, lw=1.2,
            label=f"c: k = {k:.2f} per frame")
    cx.set_xlabel("frames", color="0.8", fontsize=9)
    cx.legend(frameon=False, labelcolor="white", fontsize=8, loc="lower right")
    fig.text(0.72, 0.46, "c   the indicator's step response", color="white", fontsize=11, va="top")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"neuron": i + 1, "k": float(k), "tau_s": tau_s}


def figure_mesh(P, M, path):
    """Slide 02: the multi-mesh the law uses. Top, in 3-D and to scale (the VTK still): every level's edges in ONE
    graph over the faint neurons. Below, the levels one by one (dorsal views, head at the top): each level's nodes
    -- lattice vertices, the midline through a column of them -- and its edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    still = os.path.join(PRES, "figs", "02_multimesh_3d.png")
    render_multimesh_3d(P, M, still, labels=False)
    C, (mm_s, mm_r, lev), L = M["C"], M["mm"], M["L0"]
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d", "#ffffff"][:MESH_LEVELS]
    fig = plt.figure(figsize=(10.0, 9.0), facecolor="black")
    ax = fig.add_axes([0.0, 0.34, 1.0, 0.64])
    _black(ax)
    ax.imshow(plt.imread(still))
    fig.text(0.02, 0.985, "the multi-mesh in 3-D: all levels' edges are ONE graph", color="white", fontsize=12, va="top")
    lo, hi = P.min(0) - 20, P.max(0) + 20
    n_p = len(cols) + 1                              # the levels one by one, then all of them on top of each other
    dx, w = 0.98 / n_p, 0.98 / n_p - 0.02
    for k in range(n_p):
        a = fig.add_axes([0.01 + dx * k, 0.005, w, 0.27])
        _black(a)
        a.scatter(P[::5, 0], P[::5, 1], s=0.03, c="0.3", linewidths=0)
        for kk in (range(len(cols)) if k == len(cols) else (k,)):   # the overlay: finest first, coarse on top
            sel = lev == kk
            a.add_collection(LineCollection(np.stack([C[mm_s[sel], :2], C[mm_r[sel], :2]], 1), colors=cols[kk],
                                            linewidths=0.2 if kk == 0 else 0.4 + 0.25 * kk))
            nk = M["level_nodes"][kk]
            if kk >= 1:
                a.scatter(C[nk, 0], C[nk, 1], s=1 + 3 * kk, c=cols[kk], linewidths=0)
        a.axvline(M["mid"][0], color="0.35", lw=0.5, ls=":")
        a.set_aspect("equal")
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        if k < len(cols):
            fig.text(0.01 + dx * k + w / 2, 0.325, f"level {k}: {L * 2 ** k:.0f} \u00b5m", color=cols[k], fontsize=10,
                     ha="center", va="top")
            fig.text(0.01 + dx * k + w / 2, 0.298, f"{M['nodes_per_level'][k]:,} nodes, {M['edges_per_level'][k]:,} edges",
                     color="0.75", fontsize=7, ha="center", va="top")
        else:
            fig.text(0.01 + dx * k + w / 2, 0.325, "all levels", color="white", fontsize=10, ha="center", va="top")
            fig.text(0.01 + dx * k + w / 2, 0.298, f"{sum(M['nodes_per_level'][:1]):,} nodes, "
                     f"{sum(M['edges_per_level']):,} edges", color="0.75", fontsize=7, ha="center", va="top")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)


DECK_TITLE = "multi-level GNN on fish 2"  # Cedric, 2026-09-30: every slide carries the deck's one title (was "GraphCast on ZAPBench")


def slides_input_neurons(rec):
    """THE INPUT NEURONS (Cedric, 2026-10-02): how the 10 % are chosen, where they are and how they follow the
    stimulus, from tools/exp17_input_neurons.py's figures, numbers and kymograph movie; they open batch 11 (the batch
    that first used the mask). {} when the figures are not made yet."""
    fd = os.path.join(PRES, "figs")
    if not os.path.exists(os.path.join(fd, f"input_neurons_{rec}_numbers.json")):
        return {}
    j = json.load(open(os.path.join(fd, f"input_neurons_{rec}_numbers.json")))
    steps = [l.strip() for l in open(os.path.join(fd, f"input_neurons_{rec}_method.txt")) if l.strip()]
    dt = "the input neurons (destriped)" if "destripe" in rec else "the input neurons (ZAPBench)"
    s1 = frame("how the 10 % input neurons are chosen", f"\\panel{{figs/input_neurons_{rec}_method.png}}",
               head("model-free: the stimulus and the recording only") + "{\\scriptsize " + "\\\\[3pt]\n".join(
                   f"{i + 1}. {_tex(t)}" for i, t in enumerate(steps)) + "\\par}\n",
               "tools/exp17_input_neurons.py", left_gap=True, deck_title=f"{DECK_TITLE} $\\cdot$ {dt}: the selection")
    top = sorted(j["selected_per_best_feature"].items(), key=lambda kv: -kv[1])[:6]
    s2 = frame("where the input neurons are, and how they follow the stimulus", f"\\panel{{figs/input_neurons_{rec}_map.png}}",
               head("the selected neurons") + rows([
                   ("neurons", f"{j['selected']:,} of {j['neurons']:,} (10 \\%)"),
                   ("cut", f"band coherence $\\geq$ {j['threshold']:.3f}")])
               + head("their best stimulus feature (count)") + rows([(_tex(k), f"{v:,}") for k, v in top])
               + head("own best feature: selected vs 100 random others") + rows([
                   ("|r|, zero lag", f"{j['own_best_abs_r']['selected']:.2f} vs {j['own_best_abs_r']['random_unselected_100']:.2f}"),
                   ("band coherence", f"{j['own_best_band_coherence']['selected']:.2f} vs {j['own_best_band_coherence']['random_unselected_100']:.2f}")])
               + "{\\scriptsize caveat: f1, f3, f5, f8, f12, f17, f18, f20, f21 are condition markers (1 for their whole block), "
                 "so coherence with them picks neurons that shift between blocks, not stimulus-locked ones\\par}\n",
               "tools/exp17_input_neurons.py", left_gap=True, deck_title=f"{DECK_TITLE} $\\cdot$ {dt}: the map")
    k100 = ", ".join(f"{_tex(k)} {v}" for k, v in sorted(j["kymo_100_per_best_feature"].items(), key=lambda kv: -kv[1]))
    def kymo_slide(r_, what, v_="varying"):
        """The whole-recording kymograph still of recording r_ over the slide's width, below the title band."""
        j_ = json.load(open(os.path.join(fd, f"input_neurons_{r_}_numbers.json")))
        k_ = ", ".join(f"{_tex(k)} {v}" for k, v in sorted(j_["kymo_100_per_best_feature"].items(), key=lambda kv: -kv[1]))
        return frame_wide("the stimulus and the input neurons over the whole recording",
                          "\\vspace*{0.4\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.66\\textheight,"
                          f"keepaspectratio]{{figs/input_neurons_{r_}_{v_}_kymo_full.png}}\\par\\vspace{{2pt}}"
                          "{\\tiny\\color{gray} the mask: " + what
                          + "; the 100 shown by best feature: " + k_ + "\\par}\n",
                          "tools/exp17_input_neurons.py", deck_title=f"{DECK_TITLE} $\\cdot$ the input neurons: 22 features-stimuli"
                          + (" + ephys" if "ephys" in r_ else ""))   # Cedric, 2026-10-02
    out = {}
    if os.path.exists(os.path.join(fd, f"input_neurons_{rec}_varying_kymo_full.png")):
        s3 = kymo_slide(rec, "the 10 \\% of neurons most coherent with one of the 13 stimulus features that change "
                                 "within their condition")
    else:
        s3 = frame("the stimulus and the input neurons over the whole recording",
                   f"\\playmovie{{Movies/input_neurons_{rec}_kymo}}",
                   head("a 300-frame window sliding over all 7,879 frames") + "{\\scriptsize " + k100 + "\\par}\n",
                   "tools/exp17_input_neurons.py", left_gap=True, deck_title=f"{DECK_TITLE} $\\cdot$ {dt}: kymograph")
    out = {"12_input_method": s1, "12_input_map": s2, "12_input_kymo": s3}
    re_ = rec + "_ephys"                       # its twin with the ephys features (Cedric, 2026-10-02)
    if os.path.exists(os.path.join(fd, f"input_neurons_{re_}_mix_kymo_full.png")):   # batch 15's 20 % mask
        # Cedric, 2026-10-06: three columns -- the kymograph, the big picture with its one equation, the specifics
        j_ = json.load(open(os.path.join(fd, f"input_neurons_{re_}_numbers.json")))
        k_ = ", ".join(f"{_tex(k)} {v}" for k, v in sorted(j_["kymo_100_per_best_feature"].items(), key=lambda kv: -kv[1]))
        bp_ = j_["band_periods_s"]
        big_ = (head("which neurons the stimulus enters")
                + "{\\scriptsize\\raggedright Those whose recorded dF/F follows a stimulus feature. For neuron $i$ and "
                  "feature $k$, the coherence at frequency $f$:\\par}\\vspace{4pt}\n"
                + "{\\normalsize $C_{ik}(f) = \\dfrac{|\\langle X_i(f)\\,U_k^{*}(f)\\rangle|^2}"
                  "{\\langle |X_i(f)|^2\\rangle\\,\\langle |U_k(f)|^2\\rangle}$\\par}\\vspace{4pt}\n"
                + "{\\scriptsize\\raggedright $X_i$, $U_k$: the Fourier transforms of the neuron's trace and of the feature "
                  "over one window; $\\langle\\cdot\\rangle$ the average over the windows (Welch). 0: unrelated, 1: "
                  "locked at that frequency. Each neuron keeps its best feature's coherence at the stimulus's strongest "
                  "frequencies; the most coherent neurons get stimulus weights $B_i$, all the others $B_i = 0$. No learned "
                  "quantity enters.\\par}")
        sel_ = j_["own_best_band_coherence"]; r_ = j_["own_best_abs_r"]
        spec_ = (head("the specifics") + "{\\scriptsize\\raggedright "
                 "\\textbf{Windows:} 256 frames (234 s), Hann, half overlap.\\\\[2pt]"
                 f"\\textbf{{Band:}} the stimulus's 6 strongest periods, {min(bp_):.1f} to {max(bp_):.0f} s.\\\\[2pt]"
                 f"\\textbf{{Mask:}} {j_['selected']:,} of {j_['neurons']:,} neurons ({100 * j_['selected'] / j_['neurons']:.0f} \\%): "
                 "10 \\% by one of the 13 changing visual features, 10 \\% by one of the 5 ephys features.\\\\[2pt]"
                 f"\\textbf{{Selected vs others:}} coherence {sel_['selected']:.2f} vs {sel_['random_unselected_100']:.2f}; "
                 f"$|r|$ at zero lag {r_['selected']:.2f} vs {r_['random_unselected_100']:.2f}.\\\\[2pt]"
                 "\\textbf{Frames:} all, for the in-sample batches; training frames only for a held-out run "
                 "(2026-10-06), and visual features only once ephys left the input.\\\\[2pt]"
                 "\\textbf{The 100 shown}, by best feature: " + k_ + "\\par}")
        out["12_input_kymo_ephys"] = frame_wide(
            "the stimulus and the input neurons over the whole recording",
            "\\vspace*{0.8\\baselineskip}\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.54\\textwidth}\n\\centering"
            f"\\includegraphics[width=\\linewidth,height=0.78\\textheight,keepaspectratio]{{figs/input_neurons_{re_}_mix_kymo_full.png}}\n"
            "\\end{column}\n\\begin{column}{0.22\\textwidth}\n\\fitcol{%\n" + big_ + "}\n\\end{column}\n"
            "\\begin{column}{0.20\\textwidth}\n\\fitcol{%\n" + spec_ + "}\n\\end{column}\n\\end{columns}",
            "tools/exp17_input_neurons.py, exp17_stim_coherence.py",
            deck_title=f"{DECK_TITLE} $\\cdot$ the input neurons: 22 features-stimuli + ephys")
    elif os.path.exists(os.path.join(fd, f"input_neurons_{re_}_varying_kymo_full.png")):
        out["12_input_kymo_ephys"] = kymo_slide(re_, "the 10 \\% of neurons most coherent with one of the 13 changing visual "
                                                     "features or the 5 ephys features")
    return out


def frame(title, left, right, src, left_gap=False, deck_title=None, widths=(0.58, 0.4)):
    """`widths` the left and right columns' shares of the text width (a wider right column wraps its text longer)."""
    return (f"% generated by tools/exp17_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n"
            f"\\begin{{columns}}[T,onlytextwidth]\n\\begin{{column}}{{{widths[0]}\\textwidth}}\n"
            f"{chr(92) + 'vspace*{2' + chr(92) + 'baselineskip}' + chr(10) if left_gap else ''}{left}\n\\end{{column}}\n\\begin{{column}}{{{widths[1]}\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n\\fitcol{{%\n{right}}}\n"
            "\\end{column}\n\\end{columns}\n\\end{frame}\n")


def head(s):
    return f"{{\\normalsize\\textbf{{{s}}}}}\\\\[4pt]\n"


def rows(pairs):
    body = "".join(f"{k} & {v} \\\\\n" for k, v in pairs)
    return ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}l@{}}\n" + body + "\\end{tabular}\\par}\n"
            "\\vspace{8pt}")


# ============================================================================== results slides
def results_rows() -> list[dict]:
    """The exp17 markdown's results rows whose run has landed (its test json exists), in table order."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from exp_record import table
    from exp import load
    _, body = load(os.path.join(ROOT, "experiments", "exp17_zapbench_graphcast.md"))
    out = []
    for r in table(body):
        if r["_group"] != "training":
            continue
        name = os.path.basename(r["_run"])
        d = os.path.join(GD, "log", "training", r["_run"])
        tj = os.path.join(d, "results", f"{name}_test.json")
        if os.path.exists(tj):
            out.append({"v": r["_v"], "name": name, "dir": d, "row": r, "test": json.load(open(tj)),
                        "report": json.load(open(os.path.join(d, "results", "report.json")))
                        if os.path.exists(os.path.join(d, "results", "report.json")) else {}})
    return out


def _tex(t: str) -> str:
    """A markdown cell as LaTeX TEXT: every character LaTeX would read as markup escaped (a `^2` in a row's
    "what changed" stopped Kile's pdflatex, 2026-09-30; fit_deck's nonstop run had hidden it)."""
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$",
           "^": r"\^{}", "~": r"\~{}", "{": r"\{", "}": r"\}", "\u00b7": r"$\cdot$"}
    return "".join(rep.get(c, c) for c in t)


_ZB_SKILL = None


def mse_summary(t):
    """A run's MSE in 1e-3 dF/F^2, short (h 1-3) and long (h 16-32), grand average: the learned law, the mean
    baseline, the stimulus lookup, and ZAPBench's best model -- its published skill over its own mean baseline applied
    to this run's mean baseline (as panel a of the run's figure). Lower is better. (Cedric, 2026-10-01: no skill.)"""
    global _ZB_SKILL
    if _ZB_SKILL is None:
        _ZB_SKILL = np.array(json.load(open(os.path.join(GD, "graphs_data", "zebrafish",
                                                         "zapbench_published.json")))["best_ctx4"]["skill"])
    m, b, lk = (np.asarray(t[k]) * 1e3 for k in ("mse_model", "mse_mean", "mse_lookup"))
    zb = (1 - _ZB_SKILL) * b
    if t.get("trace_recording", "zapbench") != "zapbench" or str(t.get("name", "")).startswith("zap_ds_"):
        zb = np.full_like(b, np.nan)               # ZAPBench's numbers belong to ZAPBench's own neurons
    elif t.get("split") == "zapbench":               # the same windows as ZAPBench's: its U-Net's own MSE, exactly
        zb = np.asarray(json.load(open(os.path.join(GD, "graphs_data", "zebrafish",
                                                    "zapbench_published.json")))["best_ctx4"]["mse_test"]) * 1e3
    S, L = slice(0, 3), slice(15, 32)
    return {"model_s": m[S].mean(), "model_l": m[L].mean(), "mean_s": b[S].mean(), "mean_l": b[L].mean(),
            "look_s": lk[S].mean(), "look_l": lk[L].mean(), "zb_s": zb[S].mean(), "zb_l": zb[L].mean()}


def bm_metrics(npz):
    """THE BRAIN-MEAN METRICS (Cedric, 2026-10-03: "better than the R2 on the volume, it shows the learning of the
    network dynamics"), from a free rollout's movie npz: R2 = 1 - sum_t (pred_t - obs_t)^2 / sum_t (obs_t - mean obs)^2
    and the RMSE (dF/F) of the learned brain-mean dF/F against the recorded one over every free frame (2 h), as
    trainer._brain_mean_metrics. None when the npz has no brain-mean traces."""
    if not os.path.exists(npz):
        return None
    z = np.load(npz)
    if "mean_obs_all" not in z:
        return None
    o, p_ = np.asarray(z["mean_obs_all"], float), np.asarray(z["mean_pred_all"], float)
    ok = np.isfinite(o) & np.isfinite(p_)
    o, p_ = o[ok], p_[ok]
    return {"r2": 1 - ((p_ - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(), "rmse": float(np.sqrt(((p_ - o) ** 2).mean())),
            "r": float(np.corrcoef(o, p_)[0, 1])}      # Pearson r: the tables' metric, as the local one (Cedric, 2026-10-05)


def _per_neuron_r2(npz):
    """The free rollout's per-neuron R2 (denoised; per frame over the neurons, then the mean over the frames)."""
    if not os.path.exists(npz):
        return None
    z = np.load(npz)
    return float(np.nanmean(z["r2_denoised_all"])) if "r2_denoised_all" in z else None


def network_table(r, landed):
    """THE NETWORK TABLE of a results slide (Cedric, 2026-10-03, after exp20's gut-brain table): the brain-mean R2
    and RMSE for the full model, W = 0 at inference, no stimulus, and the twin TRAINED with no network (the batch's
    `_now` arm). The brain-mean R2 is the network test and the deck's one metric (Cedric, 2026-10-04: the per-neuron
    R2 and the MSE no longer printed)."""
    n, res = r["name"], os.path.join(r["dir"], "results")
    b0 = str(r["row"].get("batch", "")).split(".")[0]
    arms = next((a for t, _, a, _ in BATCHES if t.split(":")[0] == f"batch {b0}"), ())
    nows = ([] if n.endswith("_now") else       # a no-W run is its own twin: no row for it (2026-10-05)
            [m for _, m in arms if m.endswith("_now") and m != n and ("noeph" in m) == ("noeph" in n)])   # 15.12 <-> 15.13
    # the batch's no-network twin: of its `_now` arms, the one sharing the longest name prefix (12: leaky or not)
    twin = max(nows, key=lambda m: len(os.path.commonprefix([m, n]))) if nows else None
    lines = [("full model", os.path.join(res, f"{n}_movie.npz")),
             ("W = 0 at inference", os.path.join(res, f"{n}_W0_movie.npz")),
             ("no stimulus", os.path.join(res, f"{n}_no_stimulus_movie.npz")),
             ("no W", os.path.join(landed[twin]["dir"], "results", f"{twin}_movie.npz")
              if twin in landed else None)]
    body = ""
    for lab, f in lines:
        m = bm_metrics(f) if f else None
        if m is None:
            if lab == "full model" or (lab == "no W" and twin):
                st = ("--" if lab == "full model" else "running" if twin and os.path.exists(os.path.join(
                    ROOT, "config", "training", "zapbench", f"{twin}.yaml")) else "not trained")
                body += f"{lab} & \\multicolumn{{2}}{{l}}{{{st}}} \\\\\n"
            continue
        big_ = lab == "full model"
        body += (f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{lab}}}}}" if big_ else lab) + f" & {qv(m['r'], big=big_)} & {m['rmse']:.4f} \\\\\n"
    return (head("the network test: brain-mean dF/F, 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
            "& r & RMSE \\\\\n\\hline\n" + body
            + "\\end{tabular}\\par}" + SEC_GAP
            + local_table([(lab, f) for lab, f in lines if f], r["test"].get("trace_recording", "zapbench")))


# THE LAW ON A RESULTS SLIDE (Cedric, 2026-10-04: "slide 14 rewrite the known-ODE equation + Omega"), from
# cell_ops.StateDiffuseNeuronGraph.step: per substep z += (1/M) rate (-z + V + Omega m + B.u), rate = 1/tau
LAW_ON = {"zap_dm_ng_rl1lo"}
LAW_KNOWN_ODE = ("{\\normalsize\\textbf{the law: the known ODE, + $\\Omega$}}\\\\[4pt]\n"
                 "{\\scriptsize $\\tau_i\\,\\dot z_i = -z_i + V_i + \\Omega_i(t)\\,m_i + B_i\\cdot u(t)$\\\\[3pt]\n"
                 "$m_i = \\sum_{s}\\sum_{j\\in\\mathcal N_s(i)} W^{s}_{ji}\\tanh z_j$\\\\[3pt]\n"
                 "$\\Omega_i(t) = 1 + f_\\theta(p_i, t)$\\par}\\vspace{3pt}\n"
                 "{\\tiny $z_i$: neuron $i$'s dF/F, normalised; $\\tau_i$, $V_i$, $B_i$: its learned time constant, baseline "
                 "and weights on the stimulus features $u$ ($B_i = 0$ outside the input mask); $W^{s}_{ji}$: one learned "
                 "weight per edge, $s$ = short, mid, long. $\\Omega_i$ scales the neuron's summed messages: $f_\\theta$ a "
                 "SIREN of its position $p_i$ and the time $t$, its last layer at 0 ($\\Omega_i = 1$ untrained); learned "
                 "from batch 14 on, $\\Omega_i = 1$ in this run. 4 Euler substeps per frame (0.914 s).\\par}\\vspace{6pt}\n")
CURVES_TODO = {}                    # {slide name: run} curves figures to redraw once the deck knows the slide is shown
PANEL_A_ONLY = {"zap_b8_lin"}       # curves slides cut to panel a, the forecast error per step (Cedric, 2026-10-04)


def curves_bm(r, path):
    """A run's curves figure (trace_recording.render_curves, as the trainer draws it) redrawn from its saved results
    with panel c the brain-mean dF/F, recorded (green) and learned (white): the curves the brain-mean R2 printed
    beside it is computed on (Cedric, 2026-10-04). Redrawn only when older than its sources; False when the run has
    no brain-mean traces (its own figure is copied instead)."""
    from plexus.tasks import trace_recording as TR
    from plexus.paths import graphs_data_path
    n, res = r["name"], os.path.join(r["dir"], "results")
    zp, hp, tp = (os.path.join(res, f) for f in (f"{n}_movie.npz", "history.jsonl", f"{n}_test.json"))
    if not os.path.exists(zp) or "mean_obs_all" not in np.load(zp):
        return False
    src = [f for f in (zp, hp, tp) if os.path.exists(f)]
    if (os.path.exists(path) and os.path.getmtime(path) > max(os.path.getmtime(f) for f in src)
            and r["name"] not in PANEL_A_ONLY):        # a cut figure is redrawn whole first
        return True
    z, t = np.load(zp), r["test"]
    ref = yaml.safe_load(open(os.path.join(r["dir"], "config.yaml")))["task"]["reference"]["trace_recording"]
    pub = graphs_data_path("zebrafish", f"{ref}_published.json")
    gates = {"short_full": 0.206, "long_full": 0.438, "published": json.load(open(pub)) if os.path.exists(pub) else None}
    hist = [json.loads(l) for l in open(hp)] if os.path.exists(hp) else None
    TR.render_curves(t, path, t["names"], gates, history=hist,
                     brain_mean=(np.asarray(z["r2_t"]) * t["frame_s"] / 60, z["mean_obs_all"], z["mean_pred_all"]))
    return True


SEC_GAP = "\\vspace{\\baselineskip}\n"           # one blank line between a results column's sections (Cedric, 2026-10-05)


def qv(v, fmt="{:+.3f}", big=False):
    """A value coloured by quality (Cedric, 2026-10-05): green > 0.8, orange > 0.4, red otherwise; `big`: the full
    model's row, larger and bold."""
    col = "{rgb}{0.30,0.85,0.30}" if v > 0.8 else ("{rgb}{1.0,0.62,0.10}" if v > 0.4 else "{rgb}{1.0,0.30,0.25}")
    t = f"\\textcolor[{col.split('}{')[0][1:]}]{{{col.split('}{')[1][:-1]}}}{{{fmt.format(v)}}}"
    return f"{{\\normalsize\\textbf{{{t}}}}}" if big else t


def local_r(npz, rec_name):
    """THE LOCAL METRIC (Cedric, 2026-10-05: "computed per neuron, print mean +- SD"): per neuron, the correlation over the
    free rollout's movie frames between its learned and its recorded trace, each first regressed on its OWN brain mean
    (the learned trace on the learned brain mean, the recorded on the recorded) -- the part of a neuron's activity the
    shared brain-wide signal does not carry. {mean, sd, n} over the neurons whose recorded residual moves; cached in
    <npz>.local_r.json. None without the npz."""
    if not os.path.exists(npz):
        return None
    cj = npz + ".local_r.json"
    if os.path.exists(cj) and os.path.getmtime(cj) > os.path.getmtime(npz):
        c_ = json.load(open(cj))
        if c_.get("v") == 2:
            return c_
    from plexus.tasks import trace_recording as TR
    if rec_name not in _REC:
        _REC[rec_name] = TR.load(rec_name)
    z = np.load(npz)
    fr = z["frames"]
    P = z["pred"].astype(np.float64)
    X = _REC[rec_name]["dff"][fr].astype(np.float64)
    fin = np.isfinite(P).all(0)                 # a silenced (exploding) neuron's trace is NaN: left out of both brain
    P, X = P[:, fin], X[:, fin]                 # means, or it poisons every neuron's regression (2026-10-05)

    def resid(A):
        A = A - A.mean(0)
        b = A.mean(1)
        b = b - b.mean()
        beta = (A * b[:, None]).sum(0) / max(float((b * b).sum()), 1e-30)
        return A - b[:, None] * beta[None]
    Rp, Rx = resid(P), resid(X)
    sp, sx = Rp.std(0), Rx.std(0)
    ok = (sx > 1e-9) & (sp > 1e-9) & np.isfinite(sp)
    r = (Rp[:, ok] * Rx[:, ok]).mean(0) / (sp[ok] * sx[ok])
    out = {"mean": float(r.mean()), "sd": float(r.std()), "n": int(ok.sum()), "n_left_out_nonfinite": int((~fin).sum()),
           "v": 2}
    json.dump(out, open(cj, "w"))
    return out


def local_table(lines, rec_name):
    """The local metric's section under a network-test table: one row per (label, npz)."""
    body = ""
    for lab, f in lines:
        m = local_r(f, rec_name) if f else None
        if m is not None:
            big_ = lab == "full model"
            body += ((f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{lab}}}}}" if big_ else lab) + f" & {qv(m['mean'], big=big_)} $\\pm$ "
                     f"{m['sd']:.3f} \\\\\n")
    if not body:
        return ""
    return (head("per-neuron r, brain mean removed") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{}}\n"
            "& mean $\\pm$ SD over the neurons \\\\\n\\hline\n" + body + "\\end{tabular}\\par}\\vspace{2pt}\n"
            "{\\tiny\\color{gray} per neuron: the correlation over the 2 h of its learned and recorded traces, each first "
            "regressed on its own brain mean -- what the shared brain-wide signal does not carry\\par}" + SEC_GAP)


def bm_rows(m, label=""):
    """The two brain-mean rows of a slide's right column."""
    if m is None:
        return ""
    return head("brain-mean dF/F over the 2 h" + (f": {label}" if label else "")) + rows([
        ("R$^2$", f"{m['r2']:+.3f}"), ("RMSE", f"{m['rmse']:.4f} dF/F")])


def slides_ablation(r, tag):
    """TWO SLIDES PER BATCH (Cedric, 2026-10-02: "test full rollout with W = 0 to see if the network is doing a
    thing, and with the left neurons' W = 0 to see if the lateral network does a thing"): the batch's shown run's free
    rollout of the whole 2 h with every edge weight zero, and with the weights INTO the left half zero, each its movie
    and its R2 against the full model (tools/exp17_ablation.py; data/ablation_<run>.json). [] when not made."""
    n = r["name"]
    jp = os.path.join(EXP, "data", f"ablation_{n}.json")
    if not os.path.exists(jp):
        return []
    j = json.load(open(jp))
    a, w0 = j["full"], j["W0"]
    kl = "Sleft0" if "Sleft0" in j else "Wleft0"
    wl = j[kl]
    b0 = tag.split()[-1]
    dt = f"batch {b0} ({_tex(BATCH_LAW.get(b0, ''))}) $\\cdot$ {_tex(n)}"
    lr = j["left_right"]

    def r2(d, k="r2_denoised"):
        v = d.get(k)
        return f"{v:+.3f}" if v is not None and np.isfinite(v) and v > -100 else "diverged"
    out = []
    for key, what, kind in (("W0", "every edge weight W = 0: no network, each neuron its own leak, baseline and stimulus", "W0"),
                            (kl, "no stimulus into the left half (its stimulus weights B = 0), every W kept: "
                                 "stimulus-locked activity left there came through the network from the right"
                                 if kl == "Sleft0" else "W = 0 on every edge INTO the left half", kl)):
        mv = os.path.join(r["dir"], "results", f"movie_{kind}_cmp.mp4")    # the full model beside the ablation
        if not os.path.exists(mv):
            mv = os.path.join(r["dir"], "results", f"movie_{kind}.mp4")
        if not os.path.exists(mv):
            continue
        shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}_{kind}.mp4"))
        shutil.copy(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{n}_{kind}.png"))
        d = w0 if kind == "W0" else wl
        right = (head(f"{tag}: {_tex(n)}") + "{\\scriptsize " + what + "\\par}\\vspace{6pt}\n"
                 + head("free rollout R$^2$ (denoised), 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\n"
                 "& all & left & right \\\\\n\\hline\n"
                 f"full model & {r2(a)} & {r2(a, 'r2_denoised_left')} & {r2(a, 'r2_denoised_right')} \\\\\n"
                 f"{'W = 0' if kind == 'W0' else 'no stim. left'} & {r2(d)} & {r2(d, 'r2_denoised_left')} & "
                 f"{r2(d, 'r2_denoised_right')} \\\\\n\\end{{tabular}}\\par}}\\vspace{{6pt}}\n"
                 + _bm_table(r, n, kind)
                 + rows([("the network adds", f"{a['r2_denoised'] - w0['r2_denoised']:+.3f} R$^2$ (full minus W = 0)"),
                         ("left loses", f"{a['r2_denoised_left'] - wl['r2_denoised_left']:+.3f} with no stimulus left"),
                         ("right loses", f"{a['r2_denoised_right'] - wl['r2_denoised_right']:+.3f} with no stimulus left"),
                         ("halves", f"{lr['n_left']:,} / {lr['n_right']:,} neurons, split at the midline"),
                         ("", f"along {lr['left_right_axis']} ('left' a sign convention)"),
                         ("silenced", f"{d.get('silenced', 0)} exploding neurons (full model: {a.get('silenced', 0)})")]))
        out.append((f"{n}_{kind}", frame(f"{tag}: the free rollout with {'W = 0' if kind == 'W0' else 'no stimulus left'}",
                                         f"\\playmovie{{Movies/{n}_{kind}}}", right, "tools/exp17_ablation.py",
                                         left_gap=True, deck_title=dt + (" $\\cdot$ W = 0" if kind == "W0" else
                                                                         " $\\cdot$ no stimulus left"))))
    return out


def _bm_table(r, n, kind):
    """Brain-mean R2 and RMSE, the full model against one ablation arm (its <run>_<arm>_movie.npz)."""
    mf = bm_metrics(os.path.join(r["dir"], "results", f"{n}_movie.npz"))
    ma = bm_metrics(os.path.join(r["dir"], "results", f"{n}_{kind}_movie.npz"))
    if mf is None or ma is None:
        return ""
    lab = {"W0": "W = 0", "Sleft0": "no stim. left"}.get(kind, kind)
    return (head("brain-mean dF/F over the 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\n"
            "& R$^2$ & RMSE, dF/F \\\\\n\\hline\n"
            f"full model & {mf['r2']:+.3f} & {mf['rmse']:.4f} \\\\\n{lab} & {ma['r2']:+.3f} & {ma['rmse']:.4f} \\\\\n"
            "\\end{tabular}\\par}\\vspace{6pt}\n")


VARIANT_SLIDES = {"lead_left_quarter": ("the left quarter of the brain GIVEN (its recorded activity every frame), the "
                                       "rest run free: do measured leaders carry the rest of the brain?"),
                  # Cedric, 2026-10-03: the edge sets one at a time, at inference (the trained model, its W of the
                  # named sets set to 0)
                  "no_short": "without the streets: every short edge's W = 0 (the 6 nearest neurons), mid and long kept",
                  "no_mid": "without the roads: every middle edge's W = 0 (partners at 32 um), short and long kept",
                  "no_long": "without the highways: every long edge's W = 0 (partners at 128 um), short and mid kept",
                  "short_only": "the streets alone: the middle and long edges' W = 0, only the 6 nearest neurons kept"}
VARIANT_TITLE = {"lead_left_quarter": "leaders given", "no_short": "no streets", "no_mid": "no roads",
                 "no_long": "no highways", "short_only": "streets only"}


VARIANT_SLIDES_MESH = {"no_short": "without the finest level (M2, every neuron): every short edge's W = 0, the 16 / 32 um levels kept",
                       "no_mid": "without the 16 and 32 um levels (M0 and M1): every mid edge's W = 0, the finest level kept",
                       "no_M0": "without the coarsest level (M0, 32 um cubes): its edges' W = 0, M1 and M2 kept",
                       "no_M1": "without the middle level (M1, 16 um cubes): its edges' W = 0, M0 and M2 kept"}
VARIANT_TITLE.update({"no_M0": "no M0 (32 um)", "no_M1": "no M1 (16 um)"})
VARIANT_TITLE_MESH = {**VARIANT_TITLE, "no_short": "no M2 (every neuron)", "no_mid": "M2 only"}


def _cut_note(res_, n, v):
    """The conclusion of a mesh run's finest-level cut (Cedric, 2026-10-05), its numbers read from the rollouts."""
    if not ("_mesh" in n and v == "no_short"):
        return ""
    f_, c_, z_ = (bm_metrics(os.path.join(res_, f"{n}{x}_movie.npz")) for x in ("", "_no_short", "_W0"))
    if None in (f_, c_, z_):
        return ""
    return ("{\\scriptsize The finest level carries most of it: without M2 the model is barely better than $W = 0$ "
            f"(brain-mean r {c_['r']:+.3f} against {z_['r']:+.3f}; the full model {f_['r']:+.3f}).\\par}}\\vspace{{6pt}}\n")


def slides_variant(r, tag):
    """One slide per rollout variant of `task.rollouts` with a comparison movie (Cedric, 2026-10-03: "regions where the
    neuron activity is given ... measured neurons as leaders or hypothesised input neurons"): the full model LEFT,
    the variant RIGHT; the numbers over the FREE neurons only, the given ones left out of every score."""
    n, out = r["name"], []
    b0 = tag.split()[-1]
    dt = f"batch {b0} ({_tex(BATCH_LAW.get(b0, ''))}) $\\cdot$ {_tex(n)}"
    vs_ = VARIANT_SLIDES_MESH if "_mesh" in n else VARIANT_SLIDES          # a mesh: its levels, not streets / roads
    VT_ = VARIANT_TITLE_MESH if "_mesh" in n else VARIANT_TITLE
    for v, what in vs_.items():
        mv = os.path.join(r["dir"], "results", f"movie_{v}_cmp.mp4")
        zp = os.path.join(r["dir"], "results", f"{n}_{v}_movie.npz")
        if not (os.path.exists(mv) and os.path.exists(zp)):
            continue
        shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}_{v}.mp4"))
        shutil.copy(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{n}_{v}.png"))
        za, zf = np.load(zp), np.load(os.path.join(r["dir"], "results", f"{n}_movie.npz"))
        t = r["test"].get("rollouts", {}).get(v, {})
        if "clamp" not in t.get("spec", {"clamp": 1}):              # an ablation: the network test's columns
            res_ = os.path.join(r["dir"], "results")
            lines_ = [("full model", f"{n}_movie.npz"), (VT_[v], f"{n}_{v}_movie.npz"),
                      ("W = 0 (every edge)", f"{n}_W0_movie.npz")]
            body_ = ""
            for lab_, f_ in lines_:
                m_ = bm_metrics(os.path.join(res_, f_))
                if m_ is not None:
                    big_ = lab_ == "full model"
                    body_ += ((f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{lab_}}}}}" if big_ else lab_) + f" & {qv(m_['r'], big=big_)} & "
                              f"{m_['rmse']:.4f} \\\\\n")
            ne_ = {s_: int(r["test"].get("edges", {}).get(s_, 0)) for s_ in ("short", "mid", "long")}
            right = (head(f"{tag}: {_tex(n)}") + "{\\scriptsize " + what + "\\par}\\vspace{6pt}\n"
                     + rows([("zeroed", ", ".join(_tex(z_) for z_ in t["spec"]["zero"])), ("stimulus", "on, as in the full model")])
                     + SEC_GAP + head("the network test: brain-mean dF/F, 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                       "& r & RMSE \\\\\n\\hline\n" + body_
                     + "\\end{tabular}\\par}" + SEC_GAP
                     + _cut_note(res_, n, v)
                     + local_table([(lab_, os.path.join(res_, f_)) for lab_, f_ in lines_],
                                   r["test"].get("trace_recording", "zapbench"))
                     + "{\\tiny\\color{gray} the trained model, its named edge weights set to 0 at inference only (not "
                       "retrained); brain mean over every neuron and every free frame\\par}\n")
            out.append((f"{n}_{v}", frame(f"{tag}: {VT_[v]}", f"\\playmovie{{Movies/{n}_{v}}}", right,
                                          "trainer task.rollouts, tools/exp17_ablation.py", left_gap=True,
                                          deck_title=f"batch {b0} $\\cdot$ {_tex(n)} $\\cdot$ {VT_[v]}")))
            continue
        free_ = ~za["clamped"].astype(bool)
        rec = REC_CACHE(r)
        X = rec["dff"][za["frames"]][:, free_].astype(np.float64)
        o = X.mean(1)

        def bm_(pred):                      # brain mean over the free neurons, on the movie's 800 frames
            p_ = pred[:, free_].astype(np.float64).mean(1)
            return 1 - ((p_ - o) ** 2).sum() / ((o - o.mean()) ** 2).sum(), float(np.sqrt(((p_ - o) ** 2).mean()))
        rf, ra = bm_(zf["pred"]), bm_(za["pred"])
        right = (head(f"{tag}: {_tex(n)}") + "{\\scriptsize " + what + "\\par}\\vspace{6pt}\n"
                 + rows([("given", f"{int((~free_).sum()):,}: first 25 \\% of the length (head)"),
                         ("free", f"{int(free_.sum()):,} neurons, scored"),
                         ("stimulus", "on, into the same input neurons as the full model" if t.get("spec", {}).get("drive", "on") == "on" else "off")])
                 + head("brain-mean dF/F of the free neurons, 2 h") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\n"
                 "& R$^2$ & RMSE, dF/F \\\\\n\\hline\n"
                 f"full model & {rf[0]:+.3f} & {rf[1]:.4f} \\\\\nleft quarter given & {ra[0]:+.3f} & {ra[1]:.4f} \\\\\n"
                 "\\end{tabular}\\par}\\vspace{6pt}\n"
                 + (rows([("R$^2$ per frame, free", f"{t['r2_denoised']:+.3f} (denoised)")]) if t else "")
                 + "{\\tiny\\color{gray} brain means on the movie's 800 frames over the 2 h; the full model's over the same "
                   "free neurons\\par}\n")
        out.append((f"{n}_{v}", frame(f"{tag}: leaders given",
                                      f"\\playmovie{{Movies/{n}_{v}}}", right, "trainer task.rollouts, tools/exp17_ablation.py",
                                      left_gap=True, deck_title=f"batch {b0} $\\cdot$ {_tex(n)} $\\cdot$ leaders given")))
    return out


_REC = {}


def REC_CACHE(r):
    """The run's recording, loaded once."""
    from plexus.tasks import trace_recording as TR
    k = r["test"].get("trace_recording", "zapbench")
    if k not in _REC:
        _REC[k] = TR.load(k)
    return _REC[k]


def g17_comment(n):
    """One sentence for a batch-17 arm's results slide (Cedric, 2026-10-03), every number from the result files."""
    jc, jw, jp = (os.path.join(EXP, "data", f) for f in ("graph_curves.json", "wind_consensus_g17sel.json", "param_compare.json"))
    if not (os.path.exists(jc) and os.path.exists(jp)):
        return ""
    C, Pc = json.load(open(jc)), json.load(open(jp))["corr_with_base"]
    b, r = C["zap_e15_cur"], C.get(n)
    if n.startswith("zap_g17_mesh"):          # the meshes (Cedric, 2026-10-04: "use template slide 25")
        gd_ = os.path.join(GD, "log", "training", "zapbench")
        m_ = bm_metrics(os.path.join(gd_, n, "results", f"{n}_movie.npz"))
        jf_ = os.path.join(EXP, "data", "flow_meshes.json")
        fl_ = json.load(open(jf_)).get(n) if os.path.exists(jf_) else None
        if m_ is None:
            return ""
        s_ = (f"The multi-level mesh, {n[-1]} levels (GraphCast's nested construction on the neurons): brain-mean "
              f"R$^2$ {m_['r2']:+.3f} against the base's {b['bm_r2']:+.3f}"
              + (f"; its time-averaged flow {fl_['ex']:.2f} / {fl_['inh']:.2f} like the base's (excitatory / inhibitory, "
                 "cosine)" if fl_ else "") + ".")
        return "{\\scriptsize " + s_ + "\\par}\\vspace{6pt}\n"
    if r is None:
        return ""
    sim = ""
    if os.path.exists(jw):
        W_ = json.load(open(jw))
        mr = W_.get("matrix_runs", [])
        if n in mr and "zap_e15_cur" in mr:
            i, j = mr.index("zap_e15_cur"), mr.index(n)
            sim = (f"; its flow keeps {W_['similarity']['ex']['movie'][i][j]:.2f} / "
                   f"{W_['similarity']['in']['movie'][i][j]:.2f} of the base's (excitatory / inhibitory)")
    num = f"brain-mean R$^2$ {r['bm_r2']:+.3f} against the base's {b['bm_r2']:+.3f}"
    s_ = {"zap_g17_rot45": f"Turning the axes by 45 deg changes nothing measurable: {num}{sim}.",
          "zap_g17_randdir": f"Random directions per neuron forecast as the base does: {num}{sim}.",
          "zap_g17_knn18": f"Local edges only (the 18 nearest neurons): {num} -- the batch's highest brain-mean R$^2$: "
                           "the long-range edges are not needed.",
          "zap_g17_nolong": f"Without the highways: {num} -- the 128 um edges add little.",
          "zap_g17_r16_64": f"Shorter reaches: {num}{sim}.",
          "zap_g17_r64_256": f"Longer reaches: {num}{sim} -- no gain, and the flow least like the base's.",
          "zap_g17_random": f"The null, 18 senders drawn anywhere in the brain, does as well as every spatial graph: {num}: "
                            "each neuron needs to hear 18 others, not particular ones.",
          "zap_g17_s1": f"Seed 1 on the base graph: {num}; its learned tau, V, W, B correlate "
                        + ", ".join(f"{Pc[k]['zap_g17_s1']:+.2f}" for k in ("tau", "V", "W", "B"))
                        + f" with seed 0{sim}: the training is reproducible, the graph moves the flow."}.get(n, "")
    return ("{\\scriptsize " + s_.replace("um ", "\\textmu m ") + "\\par}\\vspace{6pt}\n") if s_ else ""


def sing_block():
    """The flow's sinks against the graph's own geometry (tools/exp17_flow_singularities.py), for the flow slide."""
    jf = os.path.join(EXP, "data", "flow_singularities.json")
    if not os.path.exists(jf):
        return ""
    F = json.load(open(jf))
    own = [F[n][s]["r_own_graph"] for n in F for s in ("ex", "inh")]
    den = [F[n][s]["r_density"] for n in F for s in ("ex", "inh")]
    b = F["zap_e15_cur"]
    return (head("do the sinks sit on the graph?")
            + "{\\scriptsize the flow's convergence ($-\\nabla\\cdot w$, where the particles pile up) against the graph's "
              "own convergence (every edge, weight 1, same grid and smoothing): r "
            + f"{min(own):+.2f} .. {max(own):+.2f} over the 8 graphs (median {np.median(own):+.2f}); the base "
            + f"{b['ex']['r_own_graph']:+.2f} / {b['inh']['r_own_graph']:+.2f} with its own graph, "
            + f"{b['ex']['r_other_graph']:+.2f} / {b['inh']['r_other_graph']:+.2f} with the random graph's; neuron "
            + f"density {min(den):+.2f} .. {max(den):+.2f}. The scaffold explains a small part (median r$^2$ "
            + f"{100 * np.median(own) ** 2:.0f} \\%) of where "
              "the flow converges: its sinks are compact blobs, not the graph's broad midline ring.\\par}\\vspace{6pt}\n")


def slides_run(r, landed=None, inset=None):
    """Two slides per landed run: its movie (recorded left, learned right) and its curves against the baselines."""
    import shutil
    n, t, rep = r["name"], r["test"], r["report"]
    bt = r["row"].get("batch", "")
    tag = f"batch {bt}" if bt else str(r["v"])   # "batch 15.4", not "batch 15, arm 4" (Cedric, 2026-10-05)
    b0 = bt.split(".")[0] if "." in bt else ""
    law_ = BATCH_LAW.get(b0, "")
    if "noeph" in n:                             # Cedric, 2026-10-05: the no-ephys arms say so in their title
        law_ = law_.replace("destriped + ephys", "destriped, no ephys")
    if bt == "17.9":                             # Cedric, 2026-10-05: 17.9 is the mesh, not a neuron-graph topology
        law_ = law_.replace("neuron graph topologies, ", "")
    dt = (f"batch {bt} ({_tex(law_)}) $\\cdot$ {_tex(n)}" if "." in bt
          else f"{DECK_TITLE} $\\cdot$ {_tex(bt)}" if bt else DECK_TITLE)    # Cedric: the batch and its law in the title
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    out = []
    stages = rep.get("stages") or []
    tr = [("updates", f"{rep.get('iters', 0):,} (horizons {stages[0][0]}..{stages[-1][0]})" if stages else "0"),
          ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"), ("weights", f"{rep.get('n_params', 0):,}")]
    ms = mse_summary(t)
    g17g = n.startswith("zap_g17_")           # a batch-17 graph slide (meshes too): inset + comment, as slide 25
    if "_mesh" in n and os.path.exists(os.path.join(PRES, "figs", f"gcmesh_{n[-1]}_top.png")):   # batch 17 and 15.14-16
        inset = f"figs/gcmesh_{n[-1]}_top.png"         # the mesh from above, head left, as the movie's two brains (Cedric)
    what = _tex(r["row"].get("what changed", ""))
    if n.endswith("_now"):                       # Cedric, 2026-10-05: a twin trained with no W says only that
        what = "no W"
    elif n.startswith("zap_e15_cur_siren_mesh"):   # one line, so the column keeps the deck's type size (2026-10-05)
        what = f"15.4 (SIREN $\\Omega$, ephys, 20 \\% mask) on the multi-level mesh, {n[-1]} levels"
    iw_ = "0.74\\linewidth" if inset and "gcmesh_" in inset else "\\linewidth"   # the mesh: the movie's brains' width
    num = (("\\includegraphics[width=" + iw_ + "]{" + inset + "}\\par\\vspace{4pt}\n" if inset else "")
           + (g17_comment(n) if g17g else "")
           + ("" if g17g else head(f"{tag}: {n.replace('_', chr(92) + '_')}")) +
           ("" if g17g else "{\\scriptsize " + what + "\\par}" + SEC_GAP)   # batch 17: the inset and the comment say it (Cedric, 2026-10-03)
           + network_table(r, landed or {})           # first: the network test (Cedric, 2026-10-03)
           + (LAW_KNOWN_ODE if n in LAW_ON else "")   # the MSE and the free rollout's R2 no longer printed (Cedric, 2026-10-04)
           + head("training") + rows(tr + [("exploding neurons", _diverges(t))]))
    if os.path.exists(mv):
        shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}.mp4"))
        shutil.copy(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{n}.png"))
        out.append((f"{n}_movie", frame(f"{tag}: the free rollout of the whole 2 h, recorded left, learned right",
                                        f"\\playmovie{{Movies/{n}}}", num, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
    fig = os.path.join(r["dir"], "results", f"{n}_test.png")
    if os.path.exists(fig):
        # Cedric, 2026-10-04: panel c the brain-mean dF/F the R2 beside it is computed on, not the per-neuron R2; no
        # MSE printed. The figure is redrawn from the run's saved results once the deck knows the slide is shown
        CURVES_TODO[f"{n}_curves"] = r
        held_ = r["test"].get("split") == "zapbench"
        right = (head(f"{tag}: {n.replace('_', chr(92) + '_')}")
                 + bm_rows(bm_metrics(os.path.join(r["dir"], "results", f"{n}_movie.npz")), "free rollout")
                 + head("panel a: how each curve is computed")       # Cedric, 2026-10-04: the legend, simply
                 + "{\\scriptsize Each curve is a forecast error: from many start frames, predict $h$ frames ahead "
                   "(0.914 s each), then the mean squared difference with the recording, over the neurons and the start "
                   "frames (" + ("ZAPBench's held-out test windows" if held_ else "every frame") + ").\\par}\\vspace{3pt}\n"
                 + "{\\scriptsize\\begin{tabular}{@{}p{0.27\\linewidth}@{\\hspace{4pt}}p{0.69\\linewidth}@{}}\n"
                   "\\textbf{white}, thick & the learned law: run from the last frames and the known stimulus, $h$ steps "
                   "on its own \\\\\\noalign{\\vskip 9pt}\n"
                   "\\textbf{grey dashed} & mean baseline: each neuron's average over its last $W$ frames, held flat; "
                   "$W$ (1 to 6) the best for each $h$ \\\\\\noalign{\\vskip 9pt}\n"
                   "\\textbf{grey dotted} & persistence: the last frame held flat ($W = 1$) \\\\\\noalign{\\vskip 9pt}\n"
                   "\\textbf{red} & stimulus lookup: each neuron's average at the same stimulus moment elsewhere in the "
                   "recording (same condition, same stimulus, same time since it changed); it ignores the past, so it "
                   "does not grow with $h$ \\\\\\noalign{\\vskip 9pt}\n"
                   "\\textbf{blue dash-dot} & ZAPBench's best published model (U-Net, 4 frames of context)"
                 + (": its published error on these test windows" if held_ else
                    ": its published skill applied to this run's mean baseline") + " \\\\\\noalign{\\vskip 9pt}\n"
                   "\\textbf{dark dotted}, flat & noise floor: the recording's own noise, the frame-to-frame change "
                   "$\\langle (x_{t+h} - x_t)^2 \\rangle$ at $h$ = 1, 2, 3 extrapolated to $h = 0$, halved \\\\\n"
                   "\\end{tabular}\\par}\n")
        out.append((f"{n}_curves", frame(f"{tag}: the prediction against the mean baseline, step by step",
                                         f"\\panel{{figs/{n}_curves_bm.png}}", right, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
    return out


# one lever slide per batch, then its base run's movie and curves: two slides per run made ~50 slides by batch 3
BATCHES = (
    ("batch 1: the GraphCast law", ("02c_multimesh_gnn", "02d_transfer", "02b_models", "02a_one_step"),
     (("base", "zap_gc_base"), ("seed 1", "zap_gc_base_s1"), ("no stimulus", "zap_gc_nostim"),
      ("history 1 frame", "zap_gc_hist1"), ("no embedding", "zap_gc_emb0"), ("increment loss", "zap_gc_incr"),
      ("horizon 1 only", "zap_gc_onestep"), ("1 mesh level", "zap_gc_mesh1")), ("zap_gc_base", "zap_gc_base_s1")),
    ("batch 2: an MLP law on the mesh", "05_one_step_current",
     (("base", "zap_cn_base"), ("seed 1", "zap_cn_base_s1"), ("no regularisers", "zap_cn_noreg"),
      ("conductance, lasso 25", "zap_cn_cond_l25"), ("conductance, no lasso", "zap_cn_cond_l0"),
      ("2 substeps", "zap_cn_sub2"), ("16 substeps", "zap_cn_sub16"), ("1 mesh level", "zap_cn_mesh1")), ("zap_cn_base", "zap_cn_base_s1")),
    ("batch 3: the known ODE on the mesh", "06_one_step_known_ode",
     (("current, base", "zap_ko_cur_base"), ("current, seed 1", "zap_ko_cur_s1"), ("current, relu", "zap_ko_cur_relu"),
      ("current, 2 substeps", "zap_ko_cur_sub2"), ("current, 1 level", "zap_ko_cur_mesh1"),
      ("conductance", "zap_ko_cond_base"), ("conductance, seed 1", "zap_ko_cond_s1"),
      ("conductance, 1 level", "zap_ko_cond_mesh1")), ("zap_ko_cur_base", "zap_ko_cur_s1")),
    ("batch 4: the GraphCast law, second levers", None,
     (("seed 2", "zap_gc_base_s2"), ("history 12 frames", "zap_gc_hist12"), ("history 24 frames", "zap_gc_hist24"),
      ("embedding 2", "zap_gc_emb2"), ("embedding 16", "zap_gc_emb16"), ("stimulus t, t+1 only", "zap_gc_stim01"),
      ("curriculum 1..5", "zap_gc_cur5"), ("curriculum 1..40", "zap_gc_cur40")), ("zap_gc_base", "zap_gc_base_s1", "zap_gc_base_s2")),
    ("batch 5: the known ODE on a graph between the neurons", "07_neuron_graph",
     (("mesh known ODE, long", "zap_ko_long"), ("graph, W prior", "zap_ng_base"), ("graph, no W prior", "zap_ng_nol1"),
      ("+ row lasso 1e-4", "zap_ng_rl1lo"), ("+ row lasso 1e-3", "zap_ng_rl1hi"), ("short edges only", "zap_ng_short"),
      ("reach 64 / 256 um", "zap_ng_wide"), ("W = 0 (no graph)", "zap_ng_now")), ()),
    ("batch 6: the latent activity and the calcium indicator (twin of batch 5)", "08_latent_calcium",
     (("mesh known ODE, long", "zap_ca_ko_long"), ("graph, W prior", "zap_ca_ng_base"), ("graph, no W prior", "zap_ca_ng_nol1"),
      ("+ row lasso 1e-4", "zap_ca_ng_rl1lo"), ("+ row lasso 1e-3", "zap_ca_ng_rl1hi"), ("short edges only", "zap_ca_ng_short"),
      ("reach 64 / 256 um", "zap_ca_ng_wide"), ("W = 0 (no graph)", "zap_ca_ng_now"), ("tau_ca from 0.5 s", "zap_ca_ng_tau05"),
      ("tau_ca from 2 s", "zap_ca_ng_tau2"), ("exact-inverse start", "zap_ca_ng_inv"), ("k fixed (1 s)", "zap_ca_ng_kfix")), ()),
    ("batch 7: the neuron graph on ZAPBench's split (twin of batch 5)", None,
     (("mesh known ODE, long", "zap_zs_ko_long"), ("graph, W prior", "zap_zs_ng_base"), ("graph, no W prior", "zap_zs_ng_nol1"),
      ("+ row lasso 1e-4", "zap_zs_ng_rl1lo"), ("+ row lasso 1e-3", "zap_zs_ng_rl1hi"), ("short edges only", "zap_zs_ng_short"),
      ("reach 64 / 256 um", "zap_zs_ng_wide"), ("W = 0 (no graph)", "zap_zs_ng_now")), ()),
    ("batch 8: amplitude and transients on the split", None,
     (("seed 1", "zap_b8_s1"), ("linear messages", "zap_b8_lin"), ("W prior x 0.1", "zap_b8_wl01"),
      ("adaptation 10 s", "zap_b8_ad10"), ("adaptation 30 s", "zap_b8_ad30"), ("adaptation 10 s + linear", "zap_b8_ad10lin"),
      ("mesh + adaptation 10 s", "zap_b8_ko_ad10"), ("seed 2", "zap_b8_s2")), ("zap_zs_ng_base", "zap_b8_s1", "zap_b8_s2")),
    ("batch 9: the destriped traces (zap-inr), twin of batch 7", ("10_destripe_data", "11_destripe_graph"),
     (("mesh known ODE, long", "zap_ds_ko_long"), ("graph, W prior", "zap_ds_ng_base"), ("graph, no W prior", "zap_ds_ng_nol1"),
      ("+ row lasso 1e-4", "zap_ds_ng_rl1lo"), ("+ row lasso 1e-3", "zap_ds_ng_rl1hi"), ("short edges only", "zap_ds_ng_short"),
      ("reach 64 / 256 um", "zap_ds_ng_wide"), ("W = 0 (no graph)", "zap_ds_ng_now")), ()),
    ("batch 10: an input mask, the stimulus into 10 % of the neurons (twin of batch 7)",
     ("12_input_method", "12_input_map", "12_input_kymo"),   # before the first 10 % mask (Cedric)
     (("mesh known ODE, long", "zap_mk_ko_long"), ("graph, W prior", "zap_mk_ng_base"), ("graph, no W prior", "zap_mk_ng_nol1"),
      ("+ row lasso 1e-4", "zap_mk_ng_rl1lo"), ("+ row lasso 1e-3", "zap_mk_ng_rl1hi"), ("short edges only", "zap_mk_ng_short"),
      ("reach 64 / 256 um", "zap_mk_ng_wide"), ("W = 0 (no graph)", "zap_mk_ng_now")), ()),
    ("batch 11: destriped, stimulus into the 10 % most stimulus-coherent neurons (twin of batch 9)", None,
     (("mesh known ODE, long", "zap_dm_ko_long"), ("graph, W prior", "zap_dm_ng_base"), ("graph, no W prior", "zap_dm_ng_nol1"),
      ("+ row lasso 1e-4", "zap_dm_ng_rl1lo"), ("+ row lasso 1e-3", "zap_dm_ng_rl1hi"), ("short edges only", "zap_dm_ng_short"),
      ("reach 64 / 256 um", "zap_dm_ng_wide"), ("W = 0 (no graph)", "zap_dm_ng_now")), ()),
    ("batch 12: the GNN-MLP on the neuron graph (destriped, coherence mask)", None,
     (("per sender g(z_j, a_j)", "zap_gm12_snd"), ("per edge g(z_i, z_j, a_i, a_j)", "zap_gm12_pair"), ("per sender, no W prior", "zap_gm12_snd_nol1"), ("per edge, no W prior", "zap_gm12_pair_nol1"), ("per sender, width 16", "zap_gm12_snd_h16"), ("per edge, seed 1", "zap_gm12_pair_s1"), ("per sender, no MLP priors", "zap_gm12_snd_noreg"), ("per sender, reach 64 / 256 um", "zap_gm12_snd_wide"), ("leak + per sender", "zap_gm12_lk_snd"), ("leak + per edge", "zap_gm12_lk_pair"), ("leak + per sender, seed 1", "zap_gm12_lk_snd_s1"), ("leak + per edge, seed 1", "zap_gm12_lk_pair_s1"), ("trained with no network", "zap_gm12_now"), ("leak, trained with no network", "zap_gm12_lk_now")), ()),
    ("batch 13: the new rig on the destriped traces, coherence mask", None,
     (("Euler (new rig)", "zap_r13_eu"), ("exponential", "zap_r13_ex"), ("Euler, rate cap 7.5", "zap_r13_cap"), ("exponential, seed 1", "zap_r13_ex_s1"), ("exp., no W prior", "zap_r13_ex_nol1"), ("exp., linear messages", "zap_r13_ex_lin"), ("exp., adaptation 10 s", "zap_r13_ex_ad"), ("exp., reach 64 / 256 um", "zap_r13_ex_wide"), ("trained with no network", "zap_r13_now")), ("zap_r13_ex", "zap_r13_ex_s1")),
    ("batch 14: modulation and conductance on the destriped traces, coherence mask", None,
     (("current", "zap_v14_cur"), ("conductance", "zap_v14_cond"), ("current + hash Omega", "zap_v14_cur_hash"), ("current + SIREN Omega", "zap_v14_cur_siren"), ("conductance + hash", "zap_v14_cond_hash"), ("conductance + SIREN", "zap_v14_cond_siren"), ("current + coarse hash", "zap_v14_cur_hashlo"), ("current + hash, seed 1", "zap_v14_cur_hash_s1"), ("trained with no network", "zap_v14_now")), ("zap_v14_cur_hash", "zap_v14_cur_hash_s1")),
    ("batch 15: batch 14 on the ephys stimulus (22 visual + 5 swim / turn), ephys-aware mask", "12_input_kymo_ephys",
     (("current", "zap_e15_cur"), ("conductance", "zap_e15_cond"), ("current + hash Omega", "zap_e15_cur_hash"), ("current + SIREN Omega", "zap_e15_cur_siren"), ("conductance + hash", "zap_e15_cond_hash"), ("conductance + SIREN", "zap_e15_cond_siren"), ("current + coarse hash", "zap_e15_cur_hashlo"), ("current + hash, seed 1", "zap_e15_cur_hash_s1"), ("trained with no network", "zap_e15_now"), ("leak + MLP message, per sender", "zap_e15_lk_snd"), ("leak + MLP message, per edge", "zap_e15_lk_pair"), ("current + SIREN, no ephys", "zap_e15_cur_siren_noeph"), ("no ephys, trained with no network", "zap_e15_noeph_now"), ("current + SIREN, mesh 3 levels", "zap_e15_cur_siren_mesh3"), ("current + SIREN, mesh 4 levels", "zap_e15_cur_siren_mesh4"), ("current + SIREN, mesh 5 levels", "zap_e15_cur_siren_mesh5"), ("mean-field control (no graph)", "zap_e15_cur_siren_mf"), ("tau bounded to [1, 100] s", "zap_e15_cur_siren_tau")), ("zap_e15_cur_hash", "zap_e15_cur_hash_s1")),
    ("batch 16: batch 15 with the calcium indicator, tau_ca fixed, latent substeps", None,
     (("tau_ca learned", "zap_c16_learn"), ("tau_ca 1 s", "zap_c16_t1"), ("tau_ca 2 s", "zap_c16_t2"), ("tau_ca 3 s", "zap_c16_t3"), ("2 s, 5 substeps", "zap_c16_t2_s5"), ("2 s, 10 substeps", "zap_c16_t2_s10"), ("2 s + SIREN Omega", "zap_c16_t2_siren"), ("trained with no network", "zap_c16_t2_now")), ()),
    ("batch 17: batch 15.1 over different graphs", None,
     (("axes turned 45 deg", "zap_g17_rot45"), ("random directions", "zap_g17_randdir"), ("18 nearest only", "zap_g17_knn18"), ("no highways", "zap_g17_nolong"), ("reaches 16 / 64 um", "zap_g17_r16_64"), ("reaches 64 / 256 um", "zap_g17_r64_256"), ("base, seed 1", "zap_g17_s1"), ("random graph (null)", "zap_g17_random"), ("mesh, 3 levels", "zap_g17_mesh3"), ("mesh, 4 levels", "zap_g17_mesh4"), ("mesh, 5 levels", "zap_g17_mesh5")), ()),
    ("batch 18: longer training of the three best laws", None,
     tuple((f"{lab} h{H} x{xf}", f"zap_b18_{k}_h{H}_x{xs}") for k, lab in (("si", "SIREN"), ("ca", "calcium + SIREN"), ("ml", "leaky MLP + SIREN"))
           for H in (50, 100) for xs, xf in (("25", "2.5"), ("5", "5"))), ()),     # horizon 200 killed (Cedric, 2026-10-04: too long)
)
# the law of each batch, in the title of every one of its slides (Cedric: which slide belongs to which batch)
BATCH_LAW = {"1": "GraphCast law", "2": "MLP, mesh", "3": "known ODE, mesh", "4": "GraphCast law",
             "5": "known ODE, neuron graph", "6": "known ODE, neuron graph, calcium",
             "7": "known ODE, neuron graph, ZAPBench split", "8": "known ODE, neuron graph, split, adaptation",
             "9": "known ODE, neuron graph, destriped traces", "10": "known ODE, neuron graph, input mask 10 %",
             "11": "known ODE, neuron graph, destriped, coherence mask 10 %",
             "12": "GNN-MLP, neuron graph, destriped", "13": "neuron graph, new rig, destriped",
             "14": "neuron graph, modulation / conductance, destriped",
             "15": "neuron graph, modulation / conductance, destriped + ephys",
             "16": "neuron graph + calcium indicator, destriped + ephys",
             "17": "neuron graph topologies, destriped + ephys",
             "18": "the three best laws, longer training"}            # Cedric's names, 2026-10-01
BATCH_VARIES = {"1": "stimulus, history, embedding, loss, curriculum, mesh levels", "2": "regularisers, synapse, substeps, levels",
                "3": "activation, substeps, levels, synapse", "4": "seed, history 12/24, embedding 2/16, stimulus window, curriculum 5/40",
                "5": "W prior, row lasso, edge reach, no graph", "6": "batch 5 + indicator; indicator start, k fixed",
                "7": "batch 5 on held-out test frames", "8": "seeds, linear messages, W prior, adaptation 10/30 s",
                "9": "batch 7 on the destriped traces (zap-inr)", "10": "batch 7, stimulus into the top 10 % |B_i| only",
                "11": "batch 9, stimulus into the top 10 % by stimulus coherence only",
                "12": "message per sender or per edge, W prior, MLP priors, width, reach",
                "13": "integrator (Euler, exponential, rate cap), seed, W prior, linear, adaptation, reach",
                "14": "conductance, Omega by hash grid or SIREN, in sample",
                "15": "batch 14 with 5 ephys features in the stimulus and the mask",
                "16": "calcium indicator: tau_ca learned or fixed 1 / 2 / 3 s, latent substeps, SIREN, no network",
                "17": "the graph: rotated or random directions, kNN only, no highways, reaches, random graph, seed, mesh",
                "18": "updates x2.5 / x5, horizons to 50 / 100"}


SHOW_RUN = {"batch 4": "zap_gc_cur40", "batch 5": "zap_ng_wide", "batch 6": "zap_ca_ng_nol1", "batch 7": "zap_zs_ng_base", "batch 8": "zap_b8_lin", "batch 9": "zap_ds_ng_base", "batch 10": "zap_mk_ng_base", "batch 11": "zap_dm_ng_rl1lo", "batch 12": "zap_gm12_snd_nol1", "batch 13": "zap_r13_ex_lin", "batch 14": "zap_v14_cur_siren", "batch 15": "zap_e15_cur_siren", "batch 16": "zap_c16_t2"}   # the run whose movie and curves a batch shows, when not its first arm (the card's)
EXTRA_RUNS = {"batch 15": ("zap_e15_cur_siren_noeph", "zap_e15_noeph_now", "zap_e15_cur_siren_mesh3",   # Cedric, 2026-10-05:
                          "zap_e15_cur_siren_mesh4", "zap_e15_cur_siren_mesh5",            # 15.12-15.17, each its results slide
                          "zap_e15_cur_siren_mf", "zap_e15_cur_siren_tau"), "batch 18": ("zap_b18_ca_h50_x25",)}
HIDE_BATCHES_UPTO = 7     # batches whose own slides are commented out of all.tex (Cedric, 2026-10-02)
# Cedric, 2026-10-06: the one-to-one comparisons from batch 15 (99b_one_to_one) -- (change, from arm, to arm, reading)
ARM_RUN = {"15.1": "zap_e15_cur", "15.4": "zap_e15_cur_siren", "15.9": "zap_e15_now", "15.12": "zap_e15_cur_siren_noeph",
           "15.13": "zap_e15_noeph_now", "15.14": "zap_e15_cur_siren_mesh3", "15.16": "zap_e15_cur_siren_mesh5",
           "15.17": "zap_e15_cur_siren_mf", "15.18": "zap_e15_cur_siren_tau", "16.7": "zap_c16_t2_siren", "17.7": "zap_g17_random",
           "17.8": "zap_g17_s1", "17.9": "zap_g17_mesh3", "17.12": "zap_g17_mesh3_tau", "17.13": "zap_g17_mf",
           "17.14": "zap_g17_mesh3_dale_d1em3", "17.16": "zap_g17_mesh3_dale_d1em2", "18.1": "zap_b18_si_h50_x25",
           "18.7": "zap_b18_ca_h50_x25"}
ONE_TO_ONE = (("SIREN $\\Omega$, neuron graph", "15.1", "15.4", "helps: the largest lever"),
              ("SIREN $\\Omega$, 3-level mesh", "17.9", "15.14", "helps: the largest lever"),
              ("no W (trained without a network)", "15.1", "15.9", "the network is needed"),
              ("mean field instead of the graph, SIREN", "15.4", "15.17", "not a mean field (slide 17)"),
              ("mean field instead of the graph, mesh", "17.9", "17.13", "mean field = no network"),
              ("random graph instead of the neuron graph", "15.1", "17.7", "layout matters little"),
              ("3-level mesh instead of the neuron graph", "15.1", "17.9", "equal"),
              ("3-level mesh instead of the neuron graph, SIREN", "15.4", "15.14", "equal"),
              ("5-level mesh instead of 3, SIREN", "15.14", "15.16", "equal"),
              ("no ephys, SIREN", "15.4", "15.12", "a small cost"),
              ("no ephys, no W", "15.9", "15.13", "a small cost"),
              ("$\\tau$ in [1, 100] s, neuron graph + SIREN", "15.4", "15.18", "helps per-neuron r"),
              ("$\\tau$ in [1, 100] s, mesh, no SIREN", "17.9", "17.12", "no effect"),
              ("Dale prior $10^{-3}$, mesh", "17.9", "17.14", "holds, small cost"),
              ("Dale prior $10^{-2}$, mesh", "17.9", "17.16", "holds, larger cost"),
              ("calcium indicator, SIREN", "15.4", "16.7", "long better, per-neuron worse"),
              ("calcium indicator, x2.5", "18.1", "18.7", "long better, per-neuron worse"),
              ("x2.5 updates, SIREN", "15.4", "18.1", "helps the long range"),
              ("x2.5 updates, calcium", "16.7", "18.7", "helps the long range"),
              ("training seed 1 (the spread)", "15.1", "17.8", "the seed spread"))
HIDE_BATCHES = {"10", "12", "13"}     # single batches hidden (Cedric: 10 on 2026-10-02; 12 and 13 on 2026-10-03)
HIDDEN_BATCHES: set = set()   # Cedric hid batch 4 while one arm had landed (2026-09-30); back with all 8 (2026-10-01)


MSE_AXIS = (0.0, 3.0)    # the batch slides' MSE axis, 1e-3 dF/F^2, the same on every batch (Cedric, 2026-10-01)


def figure_batch(arms, landed, path, band=()):
    """Short and long MSE per arm (bars, 1e-3 dF/F^2, lower is better), the base's seeds as a band; the mean baseline
    (solid grey), ZAPBench's best (dotted white, its skill applied to this mean baseline) and the stimulus lookup
    (dashed orange) as lines. Black, labels above. (Cedric, 2026-10-01: MSE, no skill.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), facecolor="black", sharey=True)
    y = np.arange(len(arms))[::-1]
    got = {n: mse_summary(landed[n]["test"]) for _, n in arms if n in landed}
    ref = next(iter(got.values()), None)
    for ax, sfx, lab in ((axs[0], "_s", "a   short, h 1-3"), (axs[1], "_l", "b   long, h 16-32")):
        ax.set_facecolor("black")
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
        for sd in ("left", "bottom"):
            ax.spines[sd].set_color("0.6")
        ax.tick_params(colors="0.8", labelsize=9)
        seeds = [mse_summary(landed[n]["test"])["model" + sfx] for n in band if n in landed]
        if len(seeds) >= 2:
            ax.axvspan(min(seeds), max(seeds), color="0.35", alpha=0.6, lw=0)
        for yy, (a, n) in zip(y, arms):
            if n in got:
                v = got[n]["model" + sfx]
                ax.barh(yy, min(v, MSE_AXIS[1]) if np.isfinite(v) else MSE_AXIS[1], color="#1f77b4", height=0.6)
                if np.isfinite(v) and v <= MSE_AXIS[1] * 0.9:
                    ax.text(v + 0.03, yy, f"{v:.3f}", color="white", va="center", fontsize=8)
                else:                                   # off the common axis: the value written inside the bar's end
                    ax.text(MSE_AXIS[1] * 0.99, yy, f"{v:.3g} \u2192", color="white", va="center", ha="right", fontsize=8)
            else:
                ax.text(0.01, yy, "running", color="0.5", va="center", fontsize=8, transform=ax.get_yaxis_transform())
        lo_x, hi = MSE_AXIS               # ONE axis on every batch slide (Cedric, 2026-10-01: a fair comparison)
        if ref:
            # the mean baseline (4.65 far ahead) is drawn when inside the axis, written at its edge when not
            if np.isfinite(ref["zb" + sfx]):
                ax.axvline(ref["zb" + sfx], color="white", ls=":", lw=1.2)
            ax.axvline(ref["look" + sfx], color="#ff7f0e", ls="--", lw=1.0)
            if ref["mean" + sfx] <= hi:
                ax.axvline(ref["mean" + sfx], color="0.6", lw=1.2)
            else:
                ax.text(0.99, 1.0, f"mean baseline {ref['mean' + sfx]:.2f} \u2192", color="0.7", fontsize=8,
                        ha="right", va="bottom", transform=ax.transAxes)
        ax.set_xlim(lo_x, hi)
        ax.set_xlabel("MSE, $10^{-3}$ dF/F$^2$ (lower is better)", color="0.8", fontsize=9)
        ax.set_title(lab, color="white", loc="left", fontsize=11)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels([f"{a}   {i + 1}" for i, (a, _) in enumerate(arms)], color="white", fontsize=9)
    fig.text(0.99, 0.01, "grey: mean baseline   dotted: ZAPBench best (U-Net, ctx 4)   "
             "dashed: stimulus lookup   band: the base's seeds", color="0.7", fontsize=7, ha="right")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor="black")
    plt.close(fig)


LAWS = (("zap_gc_", "GraphCast law (MLP, mesh)", "#9ecae1"), ("zap_cn_", "MLP, mesh", "#6baed6"),
        ("zap_ko_", "known ODE, mesh", "#fdae6b"), ("zap_ng_", "known ODE, neuron graph", "#e6550d"),
        ("zap_ca_ko", "known ODE, mesh + calcium", "#c7a0e8"), ("zap_ca_ng", "known ODE, neuron graph + calcium", "#9467bd"))


def _law(n):
    for pre, lab, col in sorted(LAWS, key=lambda x: -len(x[0])):        # longest prefix first (zap_ca_ko before zap_)
        if n.startswith(pre):
            return lab, col
    return None, None


def figure_pool(landed, path):
    """Every landed run: long MSE against short MSE (1e-3 dF/F^2, lower is better), coloured by law, batch.arm beside
    each point; a hollow marker where the 2 h free rollout left the finite numbers. ZAPBench's best (its skill applied to
    the mean baseline) dotted, the stimulus lookup dashed. (Cedric, 2026-10-01: MSE, no skill.)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 6.2), facecolor="black")
    ax.set_facecolor("black")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color("0.6")
    ax.tick_params(colors="0.8", labelsize=9)
    seen, refs = set(), []
    for n, r in landed.items():
        lab, col = _law(n)
        if lab is None or n in ("zap_gc_smoke", "zap_gc_persist", "zap_zs_persist"):
            continue
        t = r["test"]
        ms = mse_summary(t)
        refs.append(ms)
        fin = bool(t["free"]["finite"])
        b = r["row"].get("batch", "").split(".")[0]          # Cedric: the batch in the legend, one colour per batch
        col = plt.get_cmap("tab10")((int(b) - 1) % 10) if b.isdigit() else "0.6"
        lab = f"batch {b}: {BATCH_LAW.get(b, lab)}"
        ax.scatter(ms["model_s"], ms["model_l"], s=46, facecolors=col if fin else "none", edgecolors=col, lw=1.4,
                   label=lab if lab not in seen else None, zorder=3)
        ax.annotate(r["row"].get("batch", ""), (ms["model_s"], ms["model_l"]), xytext=(4, 3),
                    textcoords="offset points", color="0.75", fontsize=6.5)          # batch.arm beside every point
        seen.add(lab)
    if refs:
        zb_l, lu_l = np.median([m["zb_l"] for m in refs]), np.median([m["look_l"] for m in refs])
        zb_s = np.median([m["zb_s"] for m in refs])
        ax.axhline(zb_l, color="white", ls=":", lw=1.1)
        ax.axvline(zb_s, color="white", ls=":", lw=1.1)
        ax.axhline(lu_l, color="#ff7f0e", ls="--", lw=1.0)
        ax.text(ax.get_xlim()[0], zb_l, " ZAPBench best (U-Net, ctx 4)", color="white", fontsize=8, va="bottom")
        ax.text(ax.get_xlim()[0], lu_l, " stimulus lookup", color="#ff7f0e", fontsize=8, va="top")
    ax.set_xlabel("short MSE, h 1-3, $10^{-3}$ dF/F$^2$ (lower is better)", color="0.85", fontsize=10)
    ax.set_ylabel("long MSE, h 16-32, $10^{-3}$ dF/F$^2$", color="0.85", fontsize=10)
    hd, lb = ax.get_legend_handles_labels()
    order = sorted(range(len(lb)), key=lambda q: int(lb[q].split()[1].rstrip(":")) if lb[q].split()[1].rstrip(":").isdigit() else 99)
    ax.legend([hd[q] for q in order], [lb[q] for q in order], frameon=False, labelcolor="white", fontsize=8,
              loc="upper right")
    fig.text(0.01, 0.985, "every landed run, all batches (hollow: the 2 h free rollout is not finite)", color="white",
             fontsize=10, va="top")
    fig.savefig(path, dpi=160, facecolor="black", bbox_inches="tight")
    plt.close(fig)


LADDER = (("per-neuron floor: stimulus + leak, no coupling", "zap_ng_now"),
          ("+ mesh pooling (known ODE on the mesh)", "zap_ko_long"),
          ("+ neuron graph, W prior", "zap_ng_base"),
          ("+ neuron graph, no W prior (diverges)", "zap_ng_nol1"),
          ("  same + calcium indicator (finite)", "zap_ca_ng_nol1"),
          ("GraphCast law, curriculum 1..20", "zap_gc_base"),
          ("GraphCast law, curriculum 1..40", "zap_gc_cur40"))


def slides_pool(landed):
    figure_pool(landed, os.path.join(PRES, "figs", "pool_all_runs.png"))
    best = []
    for pre, lab, _ in LAWS:
        runs = [(n, r) for n, r in landed.items() if _law(n)[0] == lab and n not in ("zap_gc_smoke", "zap_gc_persist")]
        if runs:
            n, r = min(runs, key=lambda x: mse_summary(x[1]["test"])["model_l"])
            t, ms = r["test"], mse_summary(r["test"])
            best.append((lab, r["row"].get("batch", ""), ms["model_s"], ms["model_l"],
                         t["free"]["r2_denoised"] if t["free"]["finite"] else None))
    tab = "".join(f"{_tex(lab)} & {_tex(b)} & {ss:.3f} & {sl:.3f} & {'%+.2f' % fr if fr is not None else 'diverges'} \\\\\n"
                  for lab, b, ss, sl, fr in best)
    n_runs = sum(1 for n in landed if _law(n)[0] and n not in ("zap_gc_smoke", "zap_gc_persist"))
    right = (head("the best run of each law (lowest long MSE)") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
             "law & run & short & long & free R$^2$ \\\\\n\\hline\n" + tab + "\\end{tabular}\\par}\\vspace{6pt}\n"
             + "{\\scriptsize " + f"{n_runs} runs. " + "MSE in $10^{-3}$ dF/F$^2$, h 1-3 and 16-32, grand average; lower is better."
             " In sample for batches 1-6; held out (ZAPBench's unseen test windows) from batch 7 on. Free R$^2$: the whole 2 h rollout against the"
             " denoised recording.\\par}\n")
    s1 = ("pool_all_runs", frame("all runs pooled", "\\panel{figs/pool_all_runs.png}", right, "every landed _test.json",
                                 left_gap=True, deck_title="all batches pooled"))
    rows_ = []
    for lab, n in LADDER:
        if n in landed:
            ms = mse_summary(landed[n]["test"])
            rows_.append((lab, landed[n]["row"].get("batch", ""), ms["model_l"], ms["model_s"], ms["zb_l"]))
    tab2 = "".join(f"{_tex(l)} & {_tex(b)} & {sl:.3f} & {ss:.3f} \\\\\n" for l, b, sl, ss, _ in rows_)
    floor = next((r_[2] for r_ in rows_ if r_[0].startswith("per-neuron")), None)
    best_l = min((r_[2] for r_ in rows_), default=None)
    read = (f"the stimulus and each neuron's own leak alone: long MSE {floor:.3f}; the best coupling between real neurons "
            f"brings it to {best_l:.3f} ({100 * (1 - best_l / floor):.0f}\\,\\% lower); the mesh's W added nothing; the "
            "indicator is learned away (k$\\to$1) but keeps the long rollout finite; for the GraphCast law the curriculum's "
            "reach is the lever" if floor and best_l else "")
    right2 = (head("what lowers the long MSE") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
              "step & run & long & short \\\\\n\\hline\n" + tab2 + "\\end{tabular}\\par}\\vspace{6pt}\n"
              + head("read") + "{\\scriptsize " + read + "\\par}\n")
    figure_ladder(rows_, os.path.join(PRES, "figs", "pool_ladder.png"))
    s2 = ("pool_ladder", frame("what lowers the long MSE", "\\panel{figs/pool_ladder.png}", right2,
                               "every landed _test.json", left_gap=True, deck_title="all batches pooled"))
    return [s1, s2]


def figure_ladder(rows_, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.6, 5.4), facecolor="black")
    ax.set_facecolor("black")
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"):
        ax.spines[sd].set_color("0.6")
    ax.tick_params(colors="0.8", labelsize=9)
    y = np.arange(len(rows_))[::-1]
    for yy, (lab, b, sl, ss, zb) in zip(y, rows_):
        col = "#9ecae1" if lab.startswith("GraphCast") else ("#9467bd" if "calcium" in lab else "#e6550d")
        ax.barh(yy, sl, color=col, height=0.6)
        ax.text(sl * 1.005, yy, f"{sl:.3f}", color="white", va="center", fontsize=9)
    if rows_:
        zb = float(np.median([r_[4] for r_ in rows_]))
        ax.axvline(zb, color="white", ls=":", lw=1.1)
        ax.text(zb, len(rows_) - 0.35, " ZAPBench best", color="white", fontsize=8)
        lo = min(min(r_[2] for r_ in rows_), zb)
        ax.set_xlim(lo * 0.9, max(max(r_[2] for r_ in rows_), zb) * 1.06)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}  [{b}]" for lab, b, *_ in rows_], color="white", fontsize=9)
    ax.set_xlabel("long MSE, h 16-32, $10^{-3}$ dF/F$^2$ (lower is better)", color="0.85", fontsize=10)
    fig.savefig(path, dpi=160, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def slide_overview(landed, pages):
    """The batches at a glance (Cedric: which slide belongs to which batch): per batch its law, what it varies, the split,
    the arms landed, its best arm (lowest long MSE) with its long and short MSE, and the pages of its slides."""
    rows_ = []
    for title, _, arms, _ in BATCHES:
        b = title.split(":")[0].split()[-1]
        got = [(n, landed[n]["test"]) for _, n in arms if n in landed]
        best = min(got, key=lambda x: mse_summary(x[1])["model_l"]) if got else None
        arm = next(i + 1 for i, (_, n) in enumerate(arms) if best and n == best[0]) if best else None
        split = "held out" if 7 <= int(b) <= 13 else "in sample"        # batch 7 on: ZAPBench's split, unseen test windows
        rows_.append(f"{b} & \\raggedright {_tex(BATCH_LAW[b])} & \\raggedright {_tex(BATCH_VARIES[b])} & {split} & {len(got)}/{len(arms)} & "
                     + (f"{b}.{arm} & {mse_summary(best[1])['model_l']:.3f} & {mse_summary(best[1])['model_s']:.3f}" if best else "-- & -- & --")
                     + f" & {pages.get(b, 'hidden' if int(b) <= HIDE_BATCHES_UPTO else '--')} \\\\\n")
    tab = ("{\\tiny\\renewcommand{\\arraystretch}{0.72}\\begin{tabular}{@{}r@{\\hspace{5pt}}p{3.4cm}@{\\hspace{5pt}}p{3.9cm}@{\\hspace{5pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}l"
           "@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}l@{}}\n"
           "batch & law & what it varies & frames scored & landed & best & long & short & pages \\\\\n\\hline\n"
           + "".join(rows_) + "\\end{tabular}\\par}\n")
    body = (f"% generated by tools/exp17_slides.py (the batches at a glance)\n\\begin{{frame}}[t]{{the batches at a glance}}\n"
            "\\vspace*{\\bandgap}\\vspace*{1\\baselineskip}\n" + tab +
            "\\vspace{8pt}{\\scriptsize MSE in $10^{-3}$ dF/F$^2$, grand average, h 1-3 (short) and 16-32 (long); lower is better. Each batch's slides: "
            "its law's one-step slide, its lever slide (every arm), then one arm's movie and curves; every title names the "
            "batch and its law.\\par}\n\\end{frame}\n")
    return ("00_batches", body)


def _diverges(t) -> str:
    """When the free rollout leaves for good: the first frame with R2 denoised below -1 (a few neurons running away
    take the R2 over all neurons with them), as minutes into the recording; 'never' if it stays."""
    r = np.asarray(t.get("free_r2_denoised", []), float)
    bad = ~np.isfinite(r) | (r < -1)
    sil = np.asarray(t.get("free_silenced") or [0])
    if sil[-1]:                       # finding 21: runaway neurons frozen, left out of R2
        return f"{int(sil[-1]):,} neurons silenced, first at t = {np.asarray(t['free_t_s'])[int(np.argmax(sil > 0))] / 60:.0f} min"
    return f"at t = {np.asarray(t['free_t_s'])[int(np.argmax(bad))] / 60:.0f} min" if bad.any() else "never"


def _f3(v) -> str:
    """An MSE for a table, '--' where there is none (ZAPBench's best has no score on the destriped neurons)."""
    return f"{v:.3f}" if np.isfinite(v) else "--"


def _r2pm(v, sd) -> str:
    """Free-rollout R2 mean +- SD, 'diverged' when the rollout blew up."""
    return f"{v:+.3f} $\\pm$ {sd:.3f}" if np.isfinite(v) and v > -100 else "diverged"


def _r2(v) -> str:
    """A free-rollout R2 for a table: 'diverged' when the rollout blew up (-inf, or below -100: the destriped
    neuron-graph runs reach -2.6e15)."""
    return f"{v:+.2f}" if np.isfinite(v) and v > -100 else "diverged"


def slide_batch(title, arms, landed, band=()):
    stem = title.split(":")[0].replace(" ", "_")
    figure_batch(arms, landed, os.path.join(PRES, "figs", f"{stem}_levers.png"), band)
    tab = "".join(
        (f"{i + 1} {_tex(a)} & {mse_summary(landed[n]['test'])['model_s']:.3f} & {mse_summary(landed[n]['test'])['model_l']:.3f} & "
         f"{_r2(landed[n]['test']['free']['r2_denoised'])} \\\\\n") if n in landed else f"{i + 1} {_tex(a)} & \\multicolumn{{3}}{{l}}{{running}} \\\\\n"
        for i, (a, n) in enumerate(arms))
    right = (head(_tex(title.split(": ")[1])) + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
             "arm & short & long & free R$^2$ \\\\\n\\hline\n" + tab + "\\end{tabular}\\par}\\vspace{6pt}\n"
             + "{\\scriptsize one change per arm from the base; MSE in $10^{-3}$ dF/F$^2$, h 1-3 and 16-32 (lower is better), free R$^2$ of the "
             "whole 2 h rollout against the denoised recording. "
             + ("Held out: trained on ZAPBench's training frames, scored on its unseen test windows."
                if int(stem.split("_")[-1]) >= 7 else "In sample: trained and scored on all frames.") + "\\par}\n")
    return (f"{stem}_levers", frame(f"{title}: every lever against the base", f"\\panel{{figs/{stem}_levers.png}}",
                                    right, "the landed runs' _test.json", left_gap=True,
                                    deck_title=f"{title.split(':')[0]} ({_tex(BATCH_LAW.get(title.split(':')[0].split()[-1], ''))}) $\\cdot$ every arm"))


# ============================================================================== the models, side by side
def frame_narrow(title, img, right, src, deck_title=None, left=0.74, height=0.78, img_top="0pt", caption=""):
    """Cedric, 2026-10-05: a figure the slide's full height in a wide left column and a narrow right column -- the right
    column through \\fitcol like every results slide, so its print has the deck's one size (slide 10's)."""
    return (f"% generated by tools/exp17_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n"
            f"\\begin{{columns}}[T,onlytextwidth]\n\\begin{{column}}{{{left}\\textwidth}}\n\\centering\\vspace*{{{img_top}}}"
            f"\\includegraphics[width=\\linewidth,height={height}\\textheight,keepaspectratio]{{{img}}}\n"
            + (f"\\par\\vspace{{6pt}}{{{CAPF}\\raggedright {caption}\\par}}\n" if caption else "")   # under the plot, CAPF
            + f"\\end{{column}}\n\\begin{{column}}{{{0.98 - left:.2f}\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n"
            f"\\fitcol{{%\n{right}}}\n\\end{{column}}\n\\end{{columns}}\n\\end{{frame}}\n")

CAPF = "\\fontsize{4.6}{5.5}\\selectfont"     # the caption under a plot, one size on every slide (Cedric, 2026-10-05)


def frame_wide(title, body, src, deck_title=None):
    """A slide with no columns: `body` over the whole text width (the comparison table). It never goes through
    \\fitcol, so its size does not set the deck's one column type size. Its captions (\\tiny or 5.5 pt) are set at
    CAPF, one size on every slide (Cedric, 2026-10-05)."""
    body = body.replace("\\fontsize{5.5}{6.5}\\selectfont", CAPF).replace("\\tiny", CAPF)
    return (f"% generated by tools/exp17_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n{body}\n\\end{{frame}}\n")


def _spec(run: str) -> dict:
    """A training run's spec read back: its law's state_diffuse parameters, its stages, split, warm-up, stimulus
    window, loss reduction and priors (config/training/zapbench/<run>.yaml and the model yaml it names)."""
    import yaml
    t = yaml.safe_load(open(os.path.join(ROOT, "config", "training", "zapbench", f"{run}.yaml")))
    m = yaml.safe_load(open(os.path.join(ROOT, t["model"])))
    op = next(o for o in m["operators"] if o["op"] == "state_diffuse")
    ca = next((o for o in m["operators"] if o["op"] == "calcium_indicator"), None)
    tk = t["task"]
    loss = tk["loss"]
    red = loss[0].get("reduction", "mean") if isinstance(loss, list) else "mean"
    priors = {lr.get("param") or lr.get("block"): lr["prior"] for lr in t["learnable"] if lr.get("prior")}
    return {"op": op, "ca": ca, "hs": [s["horizon"] for s in t["training"]["stages"]],
            "iters": sum(s["iters"] for s in t["training"]["stages"]), "split": tk["reference"].get("split", "all"),
            "warmup": int(tk.get("warmup") or 0), "window": tk["drive"]["window"], "reduction": red, "priors": priors}


def _horizons(hs) -> str:
    """A curriculum's horizons, short: runs of a constant step as 'a..b' (step 1) or 'a..b by d'."""
    out, i = [], 0
    while i < len(hs):
        j = i + 1
        while j < len(hs) and hs[j] - hs[j - 1] == (hs[i + 1] - hs[i] if i + 1 < len(hs) else 1):
            j += 1
        if j - i >= 3:
            d = hs[i + 1] - hs[i]
            out.append(f"{hs[i]}..{hs[j - 1]}" + (f" by {d}" if d != 1 else ""))
        else:
            out += [str(h) for h in hs[i:j]]
        i = j
    return ", ".join(out)


# ONE COLUMN PER MODEL / TRAINING VARIANT USED IN EXP17 (Cedric, 2026-10-02: slides 4-6 merged, exhaustive): (letter,
# header, batches, the run whose spec the column reads). A new law or rig is one more entry here and one more cell
# per row in slide_models; a cell `= X' repeats column X's cell.
MODEL_COLUMNS = (
    ("A", "GraphCast law, multi-mesh", "1, 4", "zap_gc_base"),
    ("B", "MLP law (connectome), multi-mesh", "2", "zap_cn_base"),
    ("C", "known ODE, multi-mesh", "3; the `mesh' arm of 5-11", "zap_ko_cur_base"),
    ("D", "known ODE, neuron graph", "5", "zap_ng_base"),
    ("E", "latent activity + calcium indicator", "6", "zap_ca_ng_base"),
    ("F", "D on ZAPBench's split", "7", "zap_zs_ng_base"),
    ("G", "F + adaptation or linear messages", "8", "zap_b8_ad10"),
    ("H", "F on the destriped traces", "9", "zap_ds_ng_base"),
    ("I", "F / H + an input mask", "10, 11", "zap_mk_ng_base"),
    ("J", "GNN-MLP on the neuron graph, embedding $a_i$", "12", "zap_gm12_snd"),
    ("K", "new rig: warm-up, horizons to 50, integrator", "13", "zap_r13_ex"),
    ("L", "modulation $\\Omega$ + conductance", "14; 15 + ephys", "zap_v14_cond_hash"),
    ("M", "known-ODE leak + GNN-MLP message", "12.9-12.12; 15.10-11", "zap_gm12_lk_snd"),
    ("N", "latent + calcium, $\\tau_{ca}$ fixed", "16", "zap_c16_t2"),
    ("O", "other graphs", "17", "zap_g17_rot45"),
)


def slide_models(landed, MO, st, st_ds, n_ds_neurons=None):
    """The models of exp17 side by side, one column each (MODEL_COLUMNS), one row per property: the recording, the
    state, the update and its messages, the graph, what is learned and how many numbers, the stimulus, the training
    (horizons, updates, warm-up), the split, the loss. The numbers are read from each column's specs, the graphs' own
    counts and the landed runs' reports (`--' when the run has not landed). The table is scaled to the slide in
    both directions, not through \\fitcol, so the deck's one column size is not set by it."""
    S = {run: _spec(run) for *_, run in MODEL_COLUMNS}
    for extra in ("zap_gc_cur5", "zap_gc_cur40", "zap_r13_cap", "zap_gm12_pair"):
        S[extra] = _spec(extra)
    gc, cn, ko, ng = (S[r]["op"] for r in ("zap_gc_base", "zap_cn_base", "zap_ko_cur_base", "zap_ng_base"))
    ca = S["zap_ca_ng_base"]["ca"]
    lv = int(gc["mesh_levels"])
    mesh = (f"multi-mesh, {lv} levels, {gc['mesh_spacing']:.0f}-{gc['mesh_spacing'] * 2 ** (lv - 1):.0f} \\textmu m "
            f"edges (2 $\\times$ {MO['stats']['mm_edges'] // 2:,})")   # both ways (Cedric, 2026-10-05)
    n_ng = sum(v["edges"] for v in st.values())
    n_ds = sum(v["edges"] for v in st_ds.values()) if st_ds else None
    w = S["zap_ng_base"]["priors"].get("W_short", {})
    win = S["zap_gc_base"]["window"]
    nwin = win[1] - win[0] + 1

    def nparams(run):
        n = ((landed.get(run) or {}).get("report") or {}).get("n_params")
        return f"; {n:,} numbers" if n else "; not landed"

    def train(run):
        s = S[run]
        return (f"h {_horizons(s['hs'])}; {s['iters']:,} updates; "
                + (f"warm-up {s['warmup']} frames" if s["warmup"] else "no warm-up"))
    split = {"all": "none (in sample)", "zapbench": "ZAPBench's"}
    D = "= D"
    zb, ds = "ZAPBench", "destriped"
    cols = [
        # A: the GraphCast law
        [zb, "dF/F $x_i$; latents on the mesh",
         f"$x_i(t{{+}}1) = x_i(t) + \\delta_i$: encode, {gc['layers']} processor layers, decode",
         f"edge and node MLPs, latent {gc['latent']}",
         mesh + "; neuron $\\leftrightarrow$ its cube's 8 corners (batch 1: 5 levels)",
         f"MLP weights; embedding $a_i$ ({gc['embedding_dim']})" + nparams("zap_gc_base"),
         f"$u$ $t{win[0]}..t{win[1]:+d}$ (22 x {nwin}), every neuron; dF/F $t-{gc['inputs'] - 1}..t$, position",
         train("zap_gc_base") + f" (batch 4: h {_horizons(S['zap_gc_cur5']['hs'])}, "
                                f"{_horizons(S['zap_gc_cur40']['hs'])})",
         split[S["zap_gc_base"]["split"]], "MSE"],
        # B: the MLP (connectome) law
        [zb, "dF/F $x_i$; $v_k$ per mesh node, rebuilt each frame",
         f"$v_k \\mathrel{{+}}= \\frac{{1}}{{M}} f_\\theta(v_k, a_k, m_k)$, M = {cn['substeps']}; "
         "$\\Delta x_i = f_n(x_i^{t-5..t}, e_i,$ $\\bar m_i, u)$",
         r"$m_k =$ $\sum_l W_{lk}\, g_\phi(v_l, a_l)^2$",
         mesh + "; $W$ on mesh edges",
         f"$W$; $a_k$, $e_i$ ({cn['embedding_dim']}); MLPs $g_\\phi$, $f_\\theta$, $f_n$" + nparams("zap_cn_base"),
         f"$u$ (22 x {nwin}) into $f_n$",
         train("zap_cn_base"), split[S["zap_cn_base"]["split"]], "MSE (norm2) + exp02 regularisers"],
        # C: the known ODE on the mesh
        [zb, "dF/F $z_i$ now; $v_k$ per mesh node",
         f"$\\tau_k \\dot v_k = -v_k + V_k + m_k$, M = {ko['substeps']}; "
         r"$\tau_i \dot z_i = -z_i + V_i$ $+\, g_i \bar m_i + B_i\!\cdot\! u$",
         r"$W_{lk}\tanh(v_l)$ or $W_{lk}^2\,\mathrm{relu}(v_l)$ $(E_l - v_k)$",
         mesh,
         r"$W$; $\tau_k, V_k$; $\tau_i, V_i, g_i, B_i$" + nparams("zap_ko_cur_base"),
         r"$B_i\!\cdot\! u(t)$", train("zap_ko_cur_base"), split[S["zap_ko_cur_base"]["split"]],
         "MSE (norm2) + W prior"],
        # D: the known ODE on the neuron graph
        [zb, "dF/F $z_i$ now",
         r"$\tau_i \dot z_i = -z_i + V_i$ $+\, m_i + B_i\!\cdot\! u$" + f", M = {ng['substeps']}, Euler",
         r"$m_i = \sum_{s,j} W^s_{ji} \tanh(z_j)$",
         f"neurons: {ng['short_k']} nearest, 6 at $\\pm${ng['mid_um']:.0f}, 6 at $\\pm${ng['long_um']:.0f} "
         f"\\textmu m ({n_ng:,})",
         r"$W^s_{ji}$; $\tau_i, V_i, B_i$" + nparams("zap_ng_base"),
         r"$B_i\!\cdot\! u(t)$, 22 weights per neuron", train("zap_ng_base"), split[S["zap_ng_base"]["split"]],
         "MSE (norm2) + W prior"],
        # E: latent activity and the indicator
        [zb, "latent $v_i$; dF/F $c_i$ its read-out",
         r"$v_i$ as D; $c_i \mathrel{+}= k(v_i - c_i)$",
         "= D, on $v$", D,
         f"= D + $k$ ($\\tau_{{ca}}$ {ca['tau_s']:g} s), {ca['width']} start taps $a_j$" + nparams("zap_ca_ng_base"),
         D, train("zap_ca_ng_base"), split[S["zap_ca_ng_base"]["split"]], "= D, on the recorded dF/F"],
        # F: ZAPBench's split
        [zb, D, D, D, D, D + nparams("zap_zs_ng_base"), D, train("zap_zs_ng_base"), split[S["zap_zs_ng_base"]["split"]],
         D],
        # G: adaptation / linear messages
        [zb, "+ a slow $a_i$", r"= D, $-\,g_i a_i$; $\dot a_i = k_i (z_i - a_i)$",
         r"= D, or linear $W^s_{ji} z_j$", D, r"= D + $g_i, k_i$" + nparams("zap_b8_ad10"), D, train("zap_b8_ad10"),
         split[S["zap_b8_ad10"]["split"]], D],
        # H: the destriped traces
        [ds + (f" ({n_ds_neurons:,} neurons)" if n_ds_neurons else ""), D, D, D,
         f"D's rule on its positions ({n_ds:,})" if n_ds else "D's rule on its positions",
         D + nparams("zap_ds_ng_base"), D, train("zap_ds_ng_base"), split[S["zap_ds_ng_base"]["split"]], D],
        # I: the input mask
        [f"{zb} (10), {ds} (11)", D, D, D, D, D + nparams("zap_mk_ng_base"),
         r"into 10 \% of neurons: top $|B_i|$ (10), top coherence (11)", train("zap_mk_ng_base"),
         split[S["zap_mk_ng_base"]["split"]], D],
        # J: the GNN-MLP on the neuron graph (batch 12)
        [ds, "dF/F $z_i$ now; embedding $a_i$ (2)",
         f"$z_i \\mathrel{{+}}= \\frac{{1}}{{M}} f_\\theta(z_i, a_i, m_i, B_i\\!\\cdot\\! u)$, M = {S['zap_gm12_snd']['op']['substeps']}",
         r"$m_i = \sum_{s,j} W^s_{ji}\, g_\phi(z_j, a_j)^2$ or $g_\phi(z_i, z_j, a_i, a_j)^2$",
         D, f"$W^s$; $a_i$; $g_\\phi$, $f_\\theta$ (width {S['zap_gm12_snd']['op']['hidden']} / {S['zap_gm12_pair']['op']['hidden']}); $B_i$"
            + nparams("zap_gm12_snd"),
         r"$B_i\!\cdot\! u$ into $f_\theta$, coherence mask 10 \%", train("zap_gm12_snd"),
         split[S["zap_gm12_snd"]["split"]], "MSE (norm2) + exp02 regularisers"],
        # K: the new rig (batch 13)
        [ds, D,
         f"= D; Euler, exponential, or rate cap {S['zap_r13_cap']['op']['rate_max']:g} per frame",
         D, D, D + nparams("zap_r13_ex"), r"= D, coherence mask 10 \%", train("zap_r13_ex"), split[S["zap_r13_ex"]["split"]], D],
        # L: modulation + conductance (batch 14)
        [ds, D, "= D, exponential",
         r"$\Omega_i(t)$ $\sum_j W_{ji}^2$ $\mathrm{relu}(z_j)(E_j - z_i)$, or current",
         D, r"= D + $E_j$ + $\Omega$: hash grid or SIREN" + nparams("zap_v14_cond_hash"),
         r"= D, coherence mask 10 \%", train("zap_v14_cond_hash"), split[S["zap_v14_cond_hash"]["split"]], D],
        # M: the leaky GNN-MLP (batch 12, arms 9-12; batch 15, arms 10-11)
        [ds, "dF/F $z_i$ now; embedding $a_i$ (2)",
         f"= D, exponential, M = {S['zap_gm12_lk_snd']['op']['substeps']}; no update MLP",
         r"$m_i = \sum_{s,j} W^s_{ji}\, g_\phi(z_j, a_j)^2$ or $g_\phi(z_i, z_j, a_i, a_j)^2$",
         D, r"$W^s$; $a_i$; $g_\phi$; $\tau_i, V_i, B_i$" + nparams("zap_gm12_lk_snd"),
         r"= D, coherence mask 10 \% (12); ephys, 20 \% (15)", train("zap_gm12_lk_snd"),
         split[S["zap_gm12_lk_snd"]["split"]], "MSE (norm2) + exp02 regularisers"],
        # N: latent activity + the indicator, tau_ca fixed (batch 16)
        [ds + " + ephys", "latent $v_i$; dF/F $c_i$ its read-out",
         f"$v_i$ as D, M = 4 / 5 / 10; $c_i \\mathrel{{+}}= k(v_i - c_i)$",
         "= D, on $v$", D,
         f"= D + {S['zap_c16_t2']['ca']['width']} start taps; $\\tau_{{ca}}$ fixed 1 / 2 / 3 s, or learned" + nparams("zap_c16_t2"),
         r"ephys stimulus, 20 \% mask", train("zap_c16_t2"), split[S["zap_c16_t2"]["split"]], "= D, on the recorded dF/F"],
        # O: other graphs (batch 17)
        [ds + " + ephys", D, "= D, exponential", D,
         "axes turned 45 deg, random directions, 18 nearest only, no long, reaches 16 / 64 or 64 / 256 \\textmu m, "
         "random graph",
         D + nparams("zap_g17_rot45"), r"ephys stimulus, 20 \% mask", train("zap_g17_rot45"), split[S["zap_g17_rot45"]["split"]], D],
    ]
    rows_ = ["recording", "state", "update", "messages", "graph", "learned", "stimulus", "training", "split", "loss"]
    widths = (2.6, 2.6, 2.7, 2.5, 2.0, 1.75, 1.95, 1.85, 2.0, 2.6, 2.05, 2.4, 2.5, 2.3, 2.4)   # cm per column, before the scaling
    spec = "@{}>{\\raggedright\\arraybackslash}p{1.45cm}" + "".join(
        f"@{{\\hspace{{4pt}}}}>{{\\raggedright\\arraybackslash}}p{{{wd}cm}}" for wd in widths) + "@{}"
    hdr = "& " + " & ".join(f"\\textbf{{{c} {h}}}" for c, h, _, _ in MODEL_COLUMNS) + " \\\\\n"
    hdr += "\\textbf{batches} & " + " & ".join(b for _, _, b, _ in MODEL_COLUMNS) + " \\\\\n\\midrule\n"
    body = "".join(f"\\textbf{{{r}}} & " + " & ".join(c[i] for c in cols) + " \\\\[1.5pt]\n"
                   for i, r in enumerate(rows_))
    note = ("$= X$: as column X. $u$: the 22 known stimulus features; h: the curriculum's horizons, frames. "
            f"norm2: MSE in units of the one-step sd. W prior: L1 {w.get('l1', 0):g}, L2 {w.get('l2', 0):g} on every "
            "W. ZAPBench's split: per condition 70 / 10 / 20 \\% train / val / test, taxis held out, scored on its "
            "test windows. Euler: $z \\mathrel{+}= \\frac{r}{M}(\\mathrm{target} - z)$; exponential: "
            "$z \\mathrel{+}= (1 - e^{-r/M})(\\mathrm{target} - z)$. "
            "$\\Omega_i(t) = 1 + f(x_i, y_i, z_i, t)$. Calcium: $k = 1 - e^{-\\Delta t / \\tau_{ca}}$, "
            "$v_i(t_0) = \\sum_j a_j c_i(t_0 - j)$. Numbers: the base run's trained values")
    # scaled to the text width AND the slide's free height, whichever binds (\fitbox and xfp from the preamble)
    tab = ("\\sbox{\\fitbox}{\\scriptsize\\renewcommand{\\arraystretch}{1.0}%\n"
           f"\\begin{{tabular}}[t]{{{spec}}}\n" + hdr + body + "\\midrule\n"
           f"\\multicolumn{{{len(widths) + 1}}}{{@{{}}p{{{1.45 + sum(widths) + 0.14 * len(widths):.2f}cm}}@{{}}}}"
           f"{{\\color{{gray}}{note}}} \\\\\n\\end{{tabular}}}}%\n"
           "\\setlength{\\fitwd}{\\wd\\fitbox}\\setlength{\\fitht}{\\dimexpr\\ht\\fitbox+\\dp\\fitbox\\relax}%\n"
           "\\vspace*{0.3\\baselineskip}\\par\\noindent\\scalebox{\\fpeval{min(\\textwidth / \\fitwd, 1.04\\colheight / \\fitht)}}"
           "{\\usebox{\\fitbox}}\n")
    return ("02b_models", frame_wide("Every model of exp17, side by side", tab,
                                     "config/zapbench, config/training/zapbench, the landed runs' report.json",
                                     deck_title=f"{DECK_TITLE} $\\cdot$ every model, side by side"))


# ANALYSIS SLIDES after the batches, before the pooled slides: each a callable (landed runs by name) -> [(name, tex)],
# shown unless its name is put in HIDDEN_SLIDES. Empty until an analysis is added.
EXTRA_SLIDES: list = []


def main():
    os.makedirs(os.path.join(PRES, "slides"), exist_ok=True)
    os.makedirs(os.path.join(PRES, "Movies"), exist_ok=True)
    os.makedirs(os.path.join(PRES, "figs"), exist_ok=True)
    off = condition_offsets()
    P = centroids()
    S = stimulus()
    from scipy.spatial import cKDTree
    nn = cKDTree(P).query(P, k=2)[0][:, 1]
    ext = np.ptp(P, 0)
    n_cond = [off[i + 1] - off[i] for i in range(len(NAMES))]
    dead = [d for d in range(S.shape[1]) if np.abs(S[:, d]).max() == 0]

    # ---- slide 01: the data (movie over the whole recording, the run movies' look; Cedric 2026-10-02)
    stem = os.path.join(PRES, "Movies", "01_zapbench_data")
    if "--movie" in sys.argv or not os.path.exists(stem + ".mp4"):
        mv = movie_data(stem)
    else:                                                          # the movie is slow; reuse it unless asked
        mv = json.load(open(os.path.join(PRES, "slides", "numbers.json")))["movie"]
    right1 = (head("the recording") + rows([
        ("neurons", f"{N_NEURONS:,} segmented somata (H2B-GCaMP7f)"),
        ("frames", f"{T_FRAMES:,} volumes, {FRAME_S} s apart, {T_FRAMES * FRAME_S / 3600:.1f} h"),
        ("brain", f"{ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} \\textmu m"),
        ("spacing", f"nearest neighbour {np.median(nn):.1f} \\textmu m (median)"),
        ("state", "dF/F = (F - F0) / F0, one value per neuron")])
        + head("the stimulus, known") + rows([
        ("conditions", f"{len(NAMES)}, played in sequence"),
        ("", ", ".join(f"{NAMES[i]} {n_cond[i] * FRAME_S / 60:.0f}" for i in range(0, 5))),
        ("", ", ".join(f"{NAMES[i]} {n_cond[i] * FRAME_S / 60:.0f}" for i in range(5, len(NAMES))) + " (min)"),
        ("features", f"{S.shape[1] - len(dead)} numbers per frame, global"),
        ] + movie_note())
        + head("the task") + "{\\scriptsize forecast every neuron's dF/F\\\\ from its last "
        f"{HISTORY} frames and the stimulus\\par}}\n")
    s1 = frame("ZAPBench: a whole zebrafish brain under nine visual stimuli",
               "\\playmovie{Movies/01_zapbench_data}", right1, "the release traces, stimuli_features, centroids")

    # ---- slide 02: GraphCast's multi-mesh on this brain (the law's own mesh)
    MO = op_mesh(P)
    figure_mesh(P, MO, os.path.join(PRES, "figs", "02_graphcast_graph.png"))
    st = MO["stats"]
    right2 = (head("GraphCast on the neurons") + rows([
        ("grid nodes", f"the {N_NEURONS:,} neurons"),
        ("mesh nodes", f"{st['mesh_nodes']:,} lattice vertices, {MO['L0']:.0f} \\textmu m apart"),
        ("multi-mesh", f"{MESH_LEVELS} nested levels, edges " + ", ".join(f"{MO['L0'] * 2 ** k:.0f}" for k in range(MESH_LEVELS))
         + " \\textmu m"),
        ("", f"2 $\\times$ {st['mm_edges'] // 2:,} edges (both ways), ONE graph"),   # Cedric, 2026-10-05
        ("", "mirror-symmetric about the midline"),
        ("grid2mesh", f"each neuron to the corners within {math.sqrt(3) / 2 * MO['L0']:.1f} \\textmu m ({st['g2m_edges']:,})"),
        ("mesh2grid", f"each neuron from its cube's 8 corners ({st['m2g_edges']:,})")]))
    # Cedric, 2026-10-04: slide 3 turning about the vertical -- the whole multi-mesh, every level labelled
    render_multimesh_3d(P, MO, os.path.join(PRES, "Movies", "02_graphcast_turn.png"),
                        mp4=os.path.join(PRES, "Movies", "02_graphcast_turn.mp4"), labels=True)
    s2 = frame("The multi-mesh: a nested lattice over the brain, the neurons as the grid",
               "\\playmovie{Movies/02_graphcast_turn}", right2, "the law's own mesh (state_diffuse[graphcast].mesh)")
    # ---- the '2 process' panel of the one-step figure alone and large (Cedric, 2026-10-02)
    mg = figure_multimesh_gnn(P, MO, os.path.join(PRES, "figs", "02c_multimesh_gnn.png"))
    # Cedric, 2026-10-05: slide 4 as a small build-up movie -- the neurons, then the fine, middle and large levels
    mv_build = movie_multimesh_build(P, MO, os.path.join(PRES, "Movies", "02c_multimesh_build"))
    right2c = (head("the multi-mesh GNN") + rows([
        ("window", f"{mg['window_um']:.0f} \\textmu m square, one {mg['L0']:.0f}-\\textmu m cube deep, from above"),
        ("nodes", f"{st['mesh_nodes']:,} lattice vertices, {mg['L0']:.0f} \\textmu m apart"),
        ("levels", f"{MESH_LEVELS}, nested: a coarse node is a node of every finer level")]
        + [("", f"level {k}: {MO['L0'] * 2 ** k:.0f}-\\textmu m edges, {MO['nodes_per_level'][k]:,} nodes, "
                f"{MO['edges_per_level'][k]:,} edges") for k in range(MESH_LEVELS)])
        + head("one graph, all levels at once") + "{\\scriptsize every layer passes messages along the edges of all "
        f"levels together: a {MO['L0'] * 2 ** (MESH_LEVELS - 1):.0f}-\\textmu m edge carries a message across the "
        f"brain in one layer, the {MO['L0']:.0f}-\\textmu m edges the local detail. The neurons never message each "
        "other here: they are encoded onto the cube corners, processed on this graph, decoded back\\par}\\vspace{6pt}\n"
        + head("which laws run on it") + rows([
            ("GraphCast law", "8 layers of edge and node MLPs (batches 1, 4)"),
            ("MLP law", "one state per node, one weight per edge (batch 2)"),
            ("known ODE", "a leaky ODE per node, one weight per edge (batch 3)")]))
    s2c = frame("The multi-mesh GNN: the fine lattice and the long edges of the coarser levels",
                "\\playmovie{Movies/02c_multimesh_build}" if mv_build else "\\panel{figs/02c_multimesh_gnn.png}", right2c,
                "the law's own mesh (state_diffuse[graphcast].mesh)")
    tf = figure_transfer(P, MO, os.path.join(PRES, "figs", "02d_transfer.png"))
    gc_ = yaml.safe_load(open(os.path.join(ROOT, "config", "zapbench", "zap_gc_base.yaml")))
    gco = next(o for o in gc_["operators"] if o.get("op") == "state_diffuse")
    right2d = (head("graph sizes") + rows([
        ("grid2mesh", f"{st['g2m_edges']:,} edges (each neuron to the corners within {tf['radius_um']:.1f} \\textmu m)"),
        ("mesh2grid", f"{st['m2g_edges']:,} edges (each neuron from its cube's 8 corners)")])
        + head("GraphCast law (batches 1, 4): the transfer is computed, not stored") + "{\\scriptsize "
        r"latents: neuron $g_i = \phi_g(v_i)$, corner $m_k = \phi_m(q_k)$ ($q_k$ its position), width "
        f"{gco.get('latent', 32)}\\\\[2pt]"
        r"\textbf{neuron $\to$ corner}, edge $i\to k$: geometry $e_{ik} = \phi_e(d_{ik}, |d_{ik}|)$, $d_{ik}$ = corner "
        r"minus neuron in units of the cube; message $e_{ik} \mathrel{+}= \psi(e_{ik}, g_i, m_k)$; "
        r"corner $m_k \mathrel{+}= \chi(m_k, \sum_{i \to k} e_{ik})$\\[2pt]"
        r"\textbf{corner $\to$ neuron}, its cube's 8 corners: $e_{ki} = \phi'_e(d_{ki}, |d_{ki}|)$; "
        r"$e_{ki} \mathrel{+}= \psi'(e_{ki}, m_k, g_i)$; $g_i \mathrel{+}= \chi'(g_i, \sum_{k} e_{ki})$; "
        r"$x_i(t{+}1) = x_i(t) + \sigma_\Delta\,\delta(g_i)$\\[2pt]"
        r"so the weight of an edge $(k, i)$ is not a free number $m_{ki}$: it is $\psi$ / $\psi'$ evaluated on that "
        r"neuron, that corner and their offset, with ONE set of MLP weights for all "
        f"{st['g2m_edges']:,} + {st['m2g_edges']:,} edges -- a learned kernel of the geometry and the two states\\\\[2pt]"
        f"$v_i$: dF/F $t{{-}}5..t$, position, embedding $a_i$ ({gco.get('embedding_dim', 8)}), stimulus. "
        r"Learned: the MLPs $\phi_g, \phi_m, \phi_e, \psi, \chi, \phi'_e, \psi', \chi', \delta$ and $a_i$\par}"
        "\n")                                   # the mesh MLP law and mesh known ODE: no longer used (Cedric)
    mv_tf = movie_transfer_build(P, MO, os.path.join(PRES, "Movies", "02d_transfer_build"))     # Cedric, 2026-10-06
    s2d = frame("Neurons to the mesh and back: the encoder's and the decoder's edges",
                "\\playmovie{Movies/02d_transfer_build}" if mv_tf else "\\panel{figs/02d_transfer.png}",
                right2d, "the law's own mesh (state_diffuse[graphcast].mesh)", left_gap=True,
                deck_title=f"{DECK_TITLE} - GraphCast-GNN on multi-grid")
    figure_step(P, MO, os.path.join(PRES, "figs", "02a_one_step.png"))
    right2a = (head("one step, t to t+1") + rows([
        ("1 encode", "each neuron's input vector $v_i$ -> MLP -> a latent,"),
        ("", f"sent to the cube corners within {math.sqrt(3) / 2 * MO['L0']:.1f} \\textmu m"),
        ("2 process", "8 layers of message passing on the multi-mesh"),
        ("3 decode", "each neuron reads its cube's 8 corners -> $\\delta_i$")])
        + head("the input vector $v_i$") + rows([
        ("dF/F", "the neuron's last 6 frames, t-5 .. t"),
        ("position", "its soma centroid, scaled"),
        ("embedding $a_i$", "8 numbers, LEARNED, one per neuron: the only"),
        ("", "thing that tells two neurons with the same"),
        ("", "history and position apart (connectome-gnn's $a_i$)"),
        ("stimulus $u$", f"{N_STIM} features x {FORCE_WINDOW[1] - FORCE_WINDOW[0] + 1} frames (t-5 .. t+1),"),
        ("", "the same for every neuron: GraphCast's forcings")])
        + "{\\scriptsize $x_i(t+1)=x_i(t)+\\delta_i$; $\\delta$ starts at 0: untrained = persistence\\par}\n")
    s2a = frame("One step: neurons to cube corners, the multi-mesh GNN, back to the neurons",
                "\\panel{figs/02a_one_step.png}", right2a, "the law's own mesh; frame 2600 of the recording", left_gap=True)
    figure_step(P, MO, os.path.join(PRES, "figs", "05_one_step_current.png"), kind="connectome")
    figure_step(P, MO, os.path.join(PRES, "figs", "06_one_step_known_ode.png"), kind="known_ode")
    right_cn = (head("the connectome law on the mesh") + rows([
        ("state", "ONE value per mesh node, rebuilt each frame"),
        ("1", r"$v_k$ = mean of the neurons within 13.9 \textmu m"),
        ("2", r"8 substeps: $v_k \mathrel{+}= \frac{1}{M} f_\theta(v_k, a_k, m_k)$"),
        ("", r"$m_k = \sum_l W_{lk}\, g_\phi(v_l, a_l)^2$"),
        ("3", r"$m_i$ = mean of its cube's 8 corners"),
        ("", r"$\Delta x_i = f_n(x_i^{t-5..t}, e_i, m_i, u)$")])
        + head("learned") + rows([
        (r"$W$", "one weight per mesh edge (97,810)"),
        (r"$a_k$, $e_i$", "2 numbers per mesh node / per neuron"),
        (r"$g_\phi, f_\theta, f_n$", "three small MLPs (width 64)")])
        + head("regularisers (connectome-gnn exp02)") + rows([
        (r"$W$", "L1 7.5e-5, L2 7.5e-7"),
        (r"$g_\phi$", "L1 0.14, monotone 375, pin 0.45"),
        (r"$f_\theta, f_n$", "L1 0.025, L2 5e-4")]))
    right_ko = (head("a known ODE on the mesh (no MLP)") + rows([
        ("mesh", r"$v_k \mathrel{+}= \frac{1}{M}(-v_k + V_k + m_k)/\tau_k$"),
        ("current", r"$m_k = \sum_l W_{lk} \tanh(v_l)$"),
        ("conductance", r"$m_k = \sum_l W_{lk}^2\, \mathrm{relu}(v_l)(E_l - v_k)$"),
        ("neurons", r"$\Delta z_i = (-z_i + V_i + g_i m_i + B_i \cdot u)/\tau_i$")])
        + head("learned") + rows([
        (r"$W$", "one weight per mesh edge (97,810)"),
        ("mesh", r"$\tau_k$, $V_k$ (and $E_l$) per node"),
        ("neurons", r"$\tau_i$, $V_i$, $g_i$, $B_i$ (22) per neuron"),
        ("state", "dF/F now only: a first-order ODE")])
        + head("why tanh") + "{\\scriptsize the normalised activity is signed (mean 0);\\\\ relu would drop every below-mean value\\par}\n")
    s_cn = frame("One step of the connectome law", "\\panel{figs/05_one_step_current.png}", right_cn,
                 "state_diffuse[model: connectome]", left_gap=True, deck_title="multi-level GNN\\_current")
    s_ko = frame("One step of the known ODE", "\\panel{figs/06_one_step_known_ode.png}", right_ko,
                 "state_diffuse[model: known_ode]", left_gap=True, deck_title="multi-level GNN-known\\_ODE")
    st = figure_neuron_graph(P, os.path.join(PRES, "figs", "07_neuron_graph.png"),
                             mp4=os.path.join(PRES, "Movies", "07_neuron_graph.mp4"))      # turning (Cedric, 2026-10-04)
    # Cedric, 2026-10-05: no "(no mesh), M substeps per frame" in the head, no 1/M in the equations
    right_ng = (head("the laws on the neuron graph") + rows([
        # Cedric, 2026-10-05: the known ODE larger, green, a blank line before and after it
        ("", ""),
        (r"{\normalsize\textbf{known ODE}}", r"{\large\textcolor[rgb]{0.30,0.85,0.30}{$z_i \mathrel{+}= (-z_i + V_i + m_i "
         r"+ B_i \cdot u)/\tau_i$}} \\[8pt]"),
        ("current", r"$m_i = \sum_{s} \sum_{j \in \mathcal{N}_s(i)} W^{s}_{ji} \tanh(z_j)$"),
        ("conductance", r"$m_i = \sum_{s,j} (W^{s}_{ji})^2\, \mathrm{relu}(z_j)\,(E_j - z_i)$"),
        ("GNN-MLP", r"$z_i \mathrel{+}= f_\theta(z_i, a_i, m_i, B_i \cdot u)$"),
        ("sender", r"$m_i = \sum_{s,j} W^{s}_{ji}\, g_\phi(a_j, z_j)^2$"),
        ("pair", r"$m_i = \sum_{s,j} W^{s}_{ji}\, g_\phi(a_i, a_j, z_i, z_j)^2$")])
        + "{\\scriptsize $E_j$: a learned reversal per sending neuron; $a_i$: a learned 2-number embedding per "
          "neuron; $g_\\phi$, $f_\\theta$: MLPs; $z$: the normalised dF/F; $u$: the stimulus features\\par}\\vspace{4pt}\n"
        + head("three edge sets, from positions") + rows([
            (k, f"{v['edges']:,} edges, {v['per_element']:.1f} per neuron, {v['mean_um']:.0f} $\\mu$m")
            for k, v in st.items()])
        + "{\\scriptsize short: the 6 nearest neurons; mid / long: the neuron nearest each of the 6 points"
          " $\\pm$32 / $\\pm$128 $\\mu$m along x, y, z (none if farther than half the reach)\\par}\\vspace{6pt}\n"
        + head("learned") + rows([
            (r"$W^s_{ji}$", "one weight per edge, every law"),
            ("known ODE", r"$\tau_i$, $V_i$, $B_i$ (22) per neuron; $E_j$ (conductance)"),
            ("GNN-MLP", r"$a_i$ (2) and $B_i$ (22) per neuron; the MLPs $g_\phi$, $f_\theta$")]))
    s_ng = frame("The neuron graph: one weight per edge between two neurons", "\\playmovie{Movies/07_neuron_graph}",
                 right_ng, "state_diffuse[model: neuron_graph]", left_gap=True, deck_title=f"{DECK_TITLE} - known-ODE-GNN on distance graphs")
    # ---- the twin of the neuron-graph slide on the DESTRIPED traces (Cedric, 2026-10-01): the graph the batch-9 law
    # builds from the destriped positions, drawn head-up as the destriped movie (x_plot = y, y_plot = -x)
    s_ng_ds = ""
    from plexus.paths import graphs_data_path
    fds = graphs_data_path("zebrafish", "zapbench_destripe_recording.npz")
    if os.path.exists(fds):
        pds = np.load(fds)["pos_um"]
        op_ds = neuron_graph_op(positions_file="zebrafish/zapbench_destripe_recording.npz")
        st_ds = figure_neuron_graph(np.stack([pds[:, 1], -pds[:, 0], pds[:, 2]], 1),
                                    os.path.join(PRES, "figs", "11_destripe_graph.png"), op=op_ds)
        right_ds = (head("the same law, the destriped neurons") + rows([
            ("neurons", f"{op_ds.n_elements:,} (zap-inr), positions in the anatomy frame"),
            ("one tick", r"$z_i \mathrel{+}= \frac{1}{M}(-z_i + V_i + m_i + B_i \cdot u)/\tau_i$"),
            ("messages", r"$m_i = \sum_{s} \sum_{j \in \mathcal{N}_s(i)} W^{s}_{ji} \tanh(z_j)$")])
            + head("three edge sets, rebuilt from these positions") + rows([
                (k, f"{v['edges']:,} edges, {v['per_element']:.1f} per neuron, {v['mean_um']:.0f} $\\mu$m")
                for k, v in st_ds.items()])
            + head("against the ZAPBench graph") + rows([
                (k, f"{st[k]['edges']:,} edges, {st[k]['mean_um']:.0f} $\\mu$m") for k in st]))
        s_ng_ds = frame("The neuron graph on the destriped neurons", "\\panel{figs/11_destripe_graph.png}", right_ds,
                        "state_diffuse[model: neuron_graph] on zapbench_destripe_recording.npz", left_gap=True,
                        deck_title="Known\\_ODE-neuron\\_graph, destriped traces")
    lc = figure_latent_calcium(os.path.join(PRES, "figs", "08_latent_calcium.png"))
    right_lc = (head("the recording as an indicator's read-out") + rows([
        ("latent", r"$v_i \mathrel{+}= \frac{1}{M}(-v_i + V_i + m_i + B_i \cdot u)/\tau_i$"),
        ("indicator", r"$c_i(t{+}1) = c_i(t) + k\,(v_i(t) - c_i(t))$"),
        ("", r"$k = 1 - e^{-\Delta t/\tau_{ca}}$, unit gain"),
        ("rollout start", r"$v_i(t_0) = \sum_{j=0}^{5} a_j\, c_i(t_0 - j)$")])
        + head("learned") + rows([
            (r"$k$", f"one, starts at $\\tau_{{ca}}$ = {lc['tau_s']:g} s ($k$ = {lc['k']:.2f})"),
            (r"$a_j$", "6 taps, shared: the start is a learned inverse"),
            ("law", r"$W^s_{ji}$, $\tau_i$, $V_i$, $B_i$ as the neuron graph")])
        + head("why not deconvolve first") + "{\\scriptsize connectome-gnn (flyvis): Wiener deconvolution recovers W"
          " at R$^2$ 0.97 without noise, 0.18 at 40 dB, none at 30 dB -- the voltage's derivative is lost. ZAPBench"
          " sits near 8--10 dB. Here nothing is inverted: the loss is on the recorded dF/F, and the taps learn how"
          " much to invert against the noise it amplifies\\par}\\vspace{6pt}\n"
        + head("the kernel") + "{\\scriptsize H2B-GCaMP7f, nuclear (ZAPBench): its time constant is not in the paper;"
          " learned from 2 s. Unit gain: no scale for W to trade against\\par}\n")
    s_lc = frame("The latent activity and the indicator: a forward model of the recording",
                 "\\panel{figs/08_latent_calcium.png}", right_lc, "calcium_indicator + state_diffuse[neuron_graph]",
                 left_gap=True, deck_title="batch 16 $\\cdot$ Known\\_ODE-latent\\_calcium")   # opens batch 16 (Cedric, 2026-10-03)
    deck = [("01_zapbench_data", s1), ("02_graphcast_graph", s2)]
    landed = {r["name"]: r for r in results_rows()}
    in_batch = {n for _, _, arms, _ in BATCHES for _, n in arms}
    for r in landed.values():                        # the smoke and the identity run, before any batch
        if r["name"] not in in_batch:
            deck += slides_run(r, landed)
    s_models = slide_models(landed, MO, st, st_ds if os.path.exists(fds) else None,
                            op_ds.n_elements if os.path.exists(fds) else None)[1]
    step = {"02c_multimesh_gnn": s2c, "02d_transfer": s2d, "02b_models": s_models, "02a_one_step": s2a, "05_one_step_current": s_cn, "06_one_step_known_ode": s_ko, "07_neuron_graph": s_ng,
            "08_latent_calcium": s_lc,
            "10_destripe_data": slide_destripe_data()[1],   # opens batch 9, as each law's one-step slide
            "11_destripe_graph": s_ng_ds, **slides_input_neurons("zapbench_destripe")}
    for title, one_step, arms, band in BATCHES:            # each law's one-step slide, then its batch
        if title.split(":")[0] in HIDDEN_BATCHES:
            continue
        for one in ((one_step,) if isinstance(one_step, str) else (one_step or ())):   # a batch may open with several
            if step.get(one):
                deck.append((one, step[one]))
        if any(n in landed for _, n in arms):
            deck.append(slide_batch(title, arms, landed, band))
            pick = SHOW_RUN.get(title.split(":")[0])
            base = pick if pick in landed else next(n for _, n in arms if n in landed)   # the base, or the first landed
            if title.startswith("batch 17"):                          # every graph's results slide, its graph inset
                ix_ = {"zap_g17_s1": 0, "zap_g17_rot45": 1, "zap_g17_randdir": 2, "zap_g17_knn18": 3, "zap_g17_nolong": 4,
                       "zap_g17_r16_64": 5, "zap_g17_r64_256": 6, "zap_g17_random": 7, "zap_g17_mesh3": 8,
                       "zap_g17_mesh4": 9, "zap_g17_mesh5": 10}
                for _, n17 in arms:
                    if n17 in landed:
                        deck += slides_run(landed[n17], landed, inset=f"figs/graph_example_{ix_[n17]}.png")
                continue
            deck += slides_run(landed[base], landed)
            for xr_ in EXTRA_RUNS.get(title.split(":")[0], ()):               # further arms' results slides (Cedric)
                if xr_ in landed:
                    deck += slides_run(landed[xr_], landed)
                    if "_mesh" in xr_:                                         # its edge-set / level cuts (Cedric)
                        deck += slides_variant(landed[xr_], title.split(":")[0])
            deck += slides_ablation(landed[base], title.split(":")[0])        # W = 0 and left W = 0 (Cedric)
            deck += slides_variant(landed[base], title.split(":")[0])         # task.rollouts variants (Cedric, 2026-10-03)
    for make in EXTRA_SLIDES:                           # analysis slides (e.g. ablation, stimulus vs network)
        deck += make(landed)
    # Cedric, 2026-10-02: the pooled slides and the overview deleted from the deck (slides_pool, slide_overview kept)
    # Cedric, 2026-10-02: a shown run's neuron clusters and its edge weights in 3-D (exp17_clusters.py, exp17_edges.py
    # --amplitude), right after its movie; batch 11's (now commented out) and, 2026-10-03, batch 15's best run's
    for run_, suf_, b_ in (("zap_dm_ng_rl1lo", "dm", "11"), ("zap_e15_cur_siren", "e15", "15"),
                           ("zap_e15_cur_siren_mesh3", "e15m3", "15"),        # Cedric, 2026-10-05: 15.14's twin of 16
                           ("zap_g17_mesh3", "g17m3", "17")):                 # Cedric, 2026-10-05: 15.14's twin on 17.9
        rt_ = _tex(run_)
        for stem_, title_ in (("clusters_3d", "neuron clusters"), ("clusters_k4_montage", "4 clusters, one panel each"),
                              ("clusters_k8_montage", "8 clusters, one panel each"),
                              ("clusters_k16_montage", "16 clusters, one panel each"),
                              ("clusters_k32_montage", "32 clusters, one panel each")):
            if os.path.exists(os.path.join(PRES, "figs", f"{stem_}_{run_}.png")):
                deck.append((f"13_{stem_}_{suf_}", frame_wide(
                    title_, ("\\vspace*{\\fill}" if "montage" in stem_ else "")     # a wide montage: centred (Cedric)
                    + "\\vspace*{0.3\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height="
                    + ("0.84" if "montage" in stem_ else "0.70") + "\\textheight,"
                    f"keepaspectratio]{{figs/{stem_}_{run_}.png}}\\par\\vspace{{3pt}}"
                    + ("\\vspace*{\\fill}" if "montage" in stem_ else ""), "tools/exp17_clusters.py",
                    deck_title=f"batch {b_} $\\cdot$ {rt_} $\\cdot$ " + title_)))
        # the edge weights on one scale, the counts from the figure's own json (exp17_edges.py --amplitude)
        ja_ = os.path.join(PRES, "figs", f"edges_amp_{run_}.json")
        if os.path.exists(ja_):
            A_ = json.load(open(ja_))
            c_ = A_["counts"]
            sk_ = [k for k in ("short", "mid", "long") if k + "+" in c_]
            share = {k: c_[k + "+"] / max(c_[k + "+"] + c_[k + "-"], 1) for k in sk_}
            lbl_ = ({"short": "short (finest)", "mid": "mid (16, 32 \\textmu m)",
                     "long": "long ($\\geq$ 64 \\textmu m)"} if "_mesh" in run_ else
                    {"short": "short (6 nearest)", "mid": "mid ($\\pm$32 \\textmu m)", "long": "long ($\\pm$128 \\textmu m)"})
            right_amp = (head("strongest edges: one |W| cut for all three sets") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                         "edge set & W $>$ 0 & W $<$ 0 & W $>$ 0 share \\\\\n\\hline\n"
                         + "".join(f"{lbl_[k]} & {c_[k + '+']:,} & {c_[k + '-']:,} & {100 * share[k]:.0f} \\% \\\\\n" for k in sk_)
                         + "\\end{tabular}\\par}\\vspace{6pt}\n"
                         + ("{\\scriptsize Short edges hold most of the strong excitatory weights. The negative weights "
                            "sit mostly on the mid and long edges, so with distance the strongest edges shift from "
                            "mostly excitatory toward a more even mix.\\par}\\vspace{6pt}\n"
                            if len(sk_) == 3 and share["short"] > share["mid"] > share["long"] else "")
                         + f"{{\\tiny\\color{{gray}} the {A_['pooled_top']:,} largest |W| of all "
                           + (f"2 $\\times$ {A_['n_edges'] // 2:,}" if "_mesh" in run_ else f"{A_['n_edges']:,}")   # both ways
                           + " edges, "
                           f"$|W| \\geq$ {A_['cut']:.3g}; one colour range for every panel; rows: W $>$ 0 from above and "
                           "from the side, W $<$ 0 from above and from the side\\par}\n")
            deck.append((f"13_edges_amp_{suf_}", frame_narrow("edge weights on one scale", f"figs/edges_amp_{run_}.png",
                                                             right_amp, "tools/exp17_edges.py --amplitude", left=0.69,
                                                             deck_title=f"batch {b_} $\\cdot$ {rt_} $\\cdot$ edge weights")))
    # Cedric, 2026-10-03: the learned constants on the brain, batch 15's best run (tools/exp17_param_maps.py); 2026-10-06:
    # 15.18 (15.4 with tau bounded to [1, 100] s), the tau map's colour bar the bound
    PM_RUN = "zap_e15_cur_siren_tau"
    jm_ = os.path.join(EXP, "data", f"param_maps_{PM_RUN}.json")
    if os.path.exists(jm_) and os.path.exists(os.path.join(PRES, "figs", f"param_maps_{PM_RUN}.png")):
        S_ = json.load(open(jm_))

        def r3(k_, f_="{:.3g}"):
            return f"{f_.format(S_[k_]['median'])} ({f_.format(S_[k_]['p2'])} .. {f_.format(S_[k_]['p98'])})"
        tb_ = S_.get("tau_bounds")
        right_pm = (head("the learned constants")
                    + "{\\scriptsize each neuron coloured by its own learned value, from above and from the side; median "
                      "(2nd .. 98th percentile)\\par}\\vspace{4pt}\n"
                    + rows([("$\\tau$, s", r3("tau_s"))]
                           + ([(f"$\\tau$ at {tb_[0]:.0f} s", f"{100 * S_['frac_tau_at_floor']:.1f} \\%"),
                               (f"$\\tau$ at {tb_[1]:.0f} s", f"{100 * S_['frac_tau_at_ceiling']:.1f} \\%")] if tb_ else [])
                           + [("baseline $V$", r3("V")), ("W in", r3("W_in")), ("$|B|$", r3("B_norm")),
                              ("W in $<$ 0", f"{100 * S_['frac_W_in_negative']:.1f} \\%"),
                              ("inputs", f"{S_['n_masked']:,} of {S_['n']:,}")])
                    + "{\\tiny\\color{gray} " + (f"$\\tau$ bounded to [{tb_[0]:.0f}, {tb_[1]:.0f}] s (a tanh squashing of "
                                                    "softplus($\\tau_{raw}$)); the colour bar of a spans exactly the bound"
                                                    if tb_ else "$\\tau$ = 0.914 s / softplus($\\tau_{raw}$)")
                    + "; $V$ in dF/F; W the signed sum over the three edge sets into the neuron (the messages then scaled by "
                      "$\\Omega$); $B$ used only inside the input mask\\par}\n")
        deck.append(("13_param_maps_e15", frame_narrow("the learned constants on the brain",
                                                      f"figs/param_maps_{PM_RUN}.png", right_pm,
                                                      f"tools/exp17_param_maps.py {PM_RUN}", left=0.76,
                                                      deck_title=f"batch 15 $\\cdot$ {_tex(PM_RUN)} $\\cdot$ tau, V, W, B")))
    # Cedric, 2026-10-03: what batch 15's 8 clusters are, and how far to trust them (tools/exp17_cluster_profile.py)
    KP_ = 4                                           # Cedric, 2026-10-03: the profile for 4 clusters (was 8)
    jp_ = os.path.join(EXP, "data", f"cluster_profile_zap_e15_cur_siren_k{KP_}.json")
    if os.path.exists(jp_):
        D_ = json.load(open(jp_))
        tr_, e_, C_ = D_["trust"], D_["eta2"], D_["clusters"]
        oth_ = [(k_[len("ari_vs_"):], v_) for k_, v_ in tr_.items() if k_.startswith("ari_vs_zap")]
        lab_e = {"V rest": "baseline $V$", "dF/F mean": "mean dF/F", "log10 tau": "$\\tau$", "|B| (masked)": "$|B|$",
                 "dF/F SD": "dF/F SD", "|W| in": "$|W|$ in", "masked": "in the mask", "W in": "W in", "W out": "W out",
                 "body axis": "head-tail position"}
        fast_ = [c["cluster"] for c in C_ if c["tau_s_median"] < D_["frame_s"]]
        R_ = json.load(open(os.path.join(EXP, "data", f"cluster_reality_zap_e15_cur_siren_k{KP_}.json")))
        sil_ = R_["silhouette"]
        c5_ = [c["cluster"] for c in C_ if c["masked_frac"] > 0.99]
        right_cp = (head("are they real clusters? mostly not")
                    + "{\\scriptsize $\\tau$ and $V$ each form ONE continuous hump (a, b): KMeans cuts them into slices "
                      f"-- clusters {', '.join(map(str, fast_))} tile the fast tail of $\\tau$, the others are steps along "
                      f"$V$. The silhouette, {sil_['data']:.2f}, beats data with no clusters ({sil_['gaussian_null']:.2f} "
                      f"one Gaussian, {sil_['permuted_null']:.2f} permuted, c), but {100 * R_['frac_B_zero']:.0f} \\% of "
                      "the neurons have $B = 0$ (outside the input mask), a point mass that alone separates. "
                    + (f"Cluster {c5_[0]}, input neurons only with strong turning and turn-ephys weights, is the one "
                       "group of its own." if c5_ else "")
                    + "\\par}\\vspace{4pt}\n"
                    + rows([("same run (ARI)", f"restarts {np.median(tr_['restarts_ari']):.2f}, subsamples "
                                               f"{np.median(tr_['subsample_ari']):.2f}"),
                            ("other trainings (ARI)", " / ".join(f"{v_:.2f}" for _, v_ in oth_))])
                    + head("the input neurons (mask) in each cluster")
                    + "{\\scriptsize\\begin{tabular}{@{}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                      "cl. & neurons & inputs & \\% of cl. & \\% of inputs & $\\tau$, s & $V$ \\\\\n\\hline\n"
                    + "".join(f"{c['cluster']} & {c['n']:,} & {c['masked_n']:,} & {100 * c['masked_frac']:.1f} & "
                              f"{100 * c['share_of_all_masked']:.1f} & {c['tau_s_median']:.2g} & {c['V_median']:.3f} \\\\\n" for c in C_)
                    + f"all & {D_['n']:,} & {D_['n_masked']:,} & {100 * D_['n_masked'] / D_['n']:.1f} & 100 & & \\\\\n"
                    + "\\end{tabular}\\par}\\vspace{4pt}\n"
                    + "{\\scriptsize $\\eta^2$ (variance between clusters): " + ", ".join(
                        f"{lab_e.get(k_, k_)} {v_:.2f}" for k_, v_ in sorted(e_.items(), key=lambda kv: -kv[1])[:5])
                    + f"; W out {e_['W out']:.2f}, head-tail position {e_['body axis']:.2f}\\par}}\n")
        deck.append(("13_cluster_profile_e15", frame(f"what the {KP_} clusters are",
                                                    "\\centering\\includegraphics[width=\\linewidth,height=0.26\\textheight,keepaspectratio]"
                                                    f"{{figs/cluster_reality_zap_e15_cur_siren_k{KP_}.png}}\\par\\vspace{{6pt}}"
                                                    "\\includegraphics[width=\\linewidth,height=0.50\\textheight,keepaspectratio]"
                                                    f"{{figs/cluster_profile_zap_e15_cur_siren_k{KP_}.png}}",
                                                    right_cp, "tools/exp17_cluster_profile.py",
                                                    deck_title=f"batch 15 $\\cdot$ zap\\_e15\\_cur\\_siren $\\cdot$ the {KP_} clusters")))
    # Cedric, 2026-10-03: the learned modulation Omega_i(t) of batch 15's best run beside the recorded activity
    # (tools/exp17_modulation.py), right after its movie
    om_ = os.path.join(GD, "log", "training", "zapbench", "zap_e15_cur_siren", "results")
    if os.path.exists(os.path.join(om_, "movie_omega.mp4")):
        # Cedric, 2026-10-04: x4 faster -- every 4th frame kept, the same 25 fps (the 2 h in a quarter of the time)
        from plexus.tasks import trace_recording as TR_
        subprocess.run([TR_._ffmpeg(), "-y", "-loglevel", "error", "-i", os.path.join(om_, "movie_omega.mp4"), "-vf",
                        "setpts=PTS/4", "-r", "25", "-an", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        os.path.join(PRES, "Movies", "zap_e15_cur_siren_omega.mp4")], check=True)
        shutil.copy(os.path.join(om_, "movie_omega.png"), os.path.join(PRES, "Movies", "zap_e15_cur_siren_omega.png"))
        O_ = np.load(os.path.join(om_, "zap_e15_cur_siren_omega.npz"))["omega"].astype(np.float32)
        mt_ = O_.mean(1)
        right_om = (head("the learned modulation $\\Omega_i(t)$")
                    + "{\\scriptsize each neuron's message sum is multiplied by $\\Omega_i(t) = 1 + f(x_i, y_i, z_i, t)$, "
                      "$f$ a SIREN of the position and the absolute time; 1 = no modulation, 0 = no input from the "
                      "network at that frame\\par}\\vspace{6pt}\n"
                    + rows([("mean over all", f"{O_.mean():.2f}"),
                            ("5th-95th pct", f"{np.percentile(O_, 5):.2f} .. {np.percentile(O_, 95):.2f}"),
                            ("brain mean over time", f"{mt_.min():.2f} .. {mt_.max():.2f}"),
                            ("below 1", f"{100 * (O_ < 1).mean():.1f} \\% of neuron-frames")])
                    + "{\\tiny\\color{gray} the movie's 800 frames over the 2 h; left the recorded dF/F, right "
                      "$\\Omega$ per neuron, one colour scale centred on 1\\par}\n")
        deck.append(("zap_e15_cur_siren_omega", frame("the learned modulation of the messages",
                                                      "\\playmovie{Movies/zap_e15_cur_siren_omega}", right_om,
                                                      "tools/exp17_modulation.py", left_gap=True,
                                                      deck_title="batch 15 $\\cdot$ zap\\_e15\\_cur\\_siren $\\cdot$ modulation")))
    # Cedric, 2026-10-02: run 11.8 (no graph from the start) beside the full model 11.4 -- the network, not the
    # stimulus alone, makes the brain-wide swings; the numbers from both runs' movie.npz (800 frames over the 2 h)
    if "zap_dm_ng_now" in landed and "zap_dm_ng_rl1lo" in landed:
        def mv_(n):
            return np.load(os.path.join(landed[n]["dir"], "results", f"{n}_movie.npz"))
        zf, zn = mv_("zap_dm_ng_rl1lo"), mv_("zap_dm_ng_now")
        so, sf, sn = zf["mean_obs_all"].std(), zf["mean_pred_all"].std(), zn["mean_pred_all"].std()
        rf, rn = zf["r2_denoised_all"], zn["r2_denoised_all"]
        q = len(rf) // 10
        src = os.path.join(landed["zap_dm_ng_now"]["dir"], "results", "movie.mp4")
        shutil.copy(src, os.path.join(PRES, "Movies", "zap_dm_ng_now.mp4"))
        shutil.copy(src.replace(".mp4", ".png"), os.path.join(PRES, "Movies", "zap_dm_ng_now.png"))
        right_now = (head("run 11.8: no network, trained from the start")
                     + "{\\scriptsize the same known ODE per neuron, with no graph between the neurons; only the 10 \\% "
                       "input neurons see the stimulus\\par}\\vspace{6pt}\n"
                     + head("brain-mean dF/F over the 2 h: SD over time") + rows([
                         ("recorded", f"{so:.4f}"),
                         ("full model (11.4)", f"{sf:.4f} ({100 * sf / so:.0f} \\% of recorded)"),
                         ("no network (11.8)", f"{sn:.4f} ({100 * sn / so:.0f} \\% of recorded)")])
                     + head("free rollout R$^2$ denoised, per frame") + rows([
                         ("full model (11.4)", f"{np.nanmean(rf):.2f}: {np.nanmean(rf[:q]):.2f} first 10 \\% $\\to$ "
                                               f"{np.nanmean(rf[-q:]):.2f} last 10 \\%"),
                         ("no network (11.8)", f"{np.nanmean(rn):.2f}: {np.nanmean(rn[:q]):.2f} $\\to$ {np.nanmean(rn[-q:]):.2f}")])
                     + head("brain-mean dF/F over the 2 h: R$^2$ / RMSE") + rows([
                         ("full model (11.4)", "{:+.3f} / {:.4f} dF/F".format(*bm_metrics(os.path.join(
                             landed["zap_dm_ng_rl1lo"]["dir"], "results", "zap_dm_ng_rl1lo_movie.npz")).values())),
                         ("no network (11.8)", "{:+.3f} / {:.4f} dF/F".format(*bm_metrics(os.path.join(
                             landed["zap_dm_ng_now"]["dir"], "results", "zap_dm_ng_now_movie.npz")).values()))])
                     + "{\\scriptsize Without the network the brain-mean trace stays nearly flat: the stimulus alone "
                       "does not make the brain-wide swings. R$^2$ per frame is taken across neurons, so holding each "
                       "neuron at its own level already scores well; the brain-mean trace is the test.\\par}\n")
        deck.append(("zap_dm_ng_now_movie", frame("no network: the free rollout of the whole 2 h, recorded left, learned right",
                                                  "\\playmovie{Movies/zap_dm_ng_now}", right_now,
                                                  "log/training/zapbench/zap_dm_ng_now", left_gap=True,
                                                  deck_title="batch 11 $\\cdot$ zap\\_dm\\_ng\\_now $\\cdot$ no network: "
                                                             "network dynamics, not the stimulus only")))
    # Cedric, 2026-10-03: does the calcium batch (6) work? Its shown run beside its twin with no indicator (batch 5),
    # the learned indicator from the checkpoint: k = sigmoid(rate), tau_ca = -dt / ln(1 - k) (neuron_ops.CalciumIndicator)
    if "zap_ca_ng_nol1" in landed and "zap_ng_nol1" in landed:
        import torch
        rc_ = landed["zap_ca_ng_nol1"]
        fit_ = torch.load(os.path.join(rc_["dir"], "models", "best.pt"), weights_only=False, map_location="cpu")["fitted"]
        raw_ = float(fit_["calcium_indicator.rate"].reshape(-1)[0])
        taps_ = fit_["calcium_indicator.taps"].float().numpy()
        tau_ = 0.914 / -np.log(float(torch.sigmoid(torch.tensor(-raw_)))) if raw_ < 30 else 0.0   # -ln(1 - k)
        k_ = float(torch.sigmoid(torch.tensor(raw_)))
        ca0_ = next(o for o in yaml.safe_load(open(os.path.join(rc_["dir"], "model.yaml")))["operators"]
                    if o.get("op") == "calcium_indicator")
        t0_ = float(ca0_["tau_s"])
        k0_ = 1 - math.exp(-float(ca0_.get("frame_s", 0.914)) / t0_)

        def row_(n):                       # the standard numbers: MSE (1e-3 dF/F^2), brain-mean and per-neuron R2
            ms_ = mse_summary(landed[n]["test"])
            f_ = os.path.join(landed[n]["dir"], "results", f"{n}_movie.npz")
            m_ = bm_metrics(f_)
            return f"{_f3(ms_['model_s'])} & {_f3(ms_['model_l'])} & {m_['r2']:+.3f} & {_per_neuron_r2(f_):+.3f}"
        src_ = os.path.join(rc_["dir"], "results", "movie.mp4")
        shutil.copy(src_, os.path.join(PRES, "Movies", "zap_ca_ng_nol1.mp4"))
        shutil.copy(src_.replace(".mp4", ".png"), os.path.join(PRES, "Movies", "zap_ca_ng_nol1.png"))
        t6_ = rc_["test"]
        ms6_ = mse_summary(t6_)
        right_ca = (head("batch 6.3: zap\\_ca\\_ng\\_nol1")
                    + "{\\scriptsize the neuron-graph known ODE on a LATENT activity $v$, read through a learned indicator: "
                      "$c \\mathrel{+}= k\\,(v - c)$; ZAPBench traces, all frames (in sample)\\par}\\vspace{6pt}\n"
                    + network_table(rc_, landed)
                    + head("MSE, $10^{-3}$ dF/F$^2$ (lower is better)") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\n"
                    + "& h 1-3 & h 16-32 \\\\\n" + "".join(f"{lab} & {_f3(ms6_[k + '_s'])} & {_f3(ms6_[k + '_l'])} \\\\\n" for lab, k in (
                        ("learned law", "model"), ("mean baseline", "mean"), ("stimulus lookup", "look"), ("ZAPBench best", "zb")))
                    + "\\end{tabular}\\par}\\vspace{6pt}\n"
                    + head("against the twins with no indicator (batch 5)") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                    "& \\multicolumn{2}{c}{MSE, $10^{-3}$} & \\multicolumn{2}{c}{free R$^2$} \\\\\n"
                    "& h 1-3 & h 16-32 & brain & neuron \\\\\n\\hline\n"
                    f"calcium (6.3) & {row_('zap_ca_ng_nol1')} \\\\\nno indicator (5.3) & {row_('zap_ng_nol1')} \\\\\n"
                    + ("calcium, no network (6.8) & " + row_("zap_ca_ng_now") + " \\\\\n" if "zap_ca_ng_now" in landed else "")
                    + ("no indicator, no network (5.8) & " + row_("zap_ng_now") + " \\\\\n" if "zap_ng_now" in landed else "")
                    + "\\end{tabular}\\par}\\vspace{6pt}\n"
                    + head("the learned indicator") + rows([
                        ("$k$", f"{k_:.7f} (start {k0_:.2f})"),
                        ("$\\tau_{ca}$", f"{tau_:.3f} s (start {t0_:g} s; one frame 0.914 s)"),
                        ("start taps $a_j$", ", ".join(f"{v:.2f}" for v in taps_) + f" (sum {taps_.sum():.2f})")])
                    + "{\\scriptsize It runs, but the indicator learns itself away: $\\tau_{ca}$ falls from its start to "
                      "well under one frame, so $c$ follows $v$ and the latent IS the recording; the start taps are a "
                      "smoothing over the last frames, not an inverse.\\par}\n")
        deck.append(("08b_calcium_result", frame("the calcium batch: does it work?", "\\playmovie{Movies/zap_ca_ng_nol1}",
                                                right_ca, "log/training/zapbench/zap_ca_ng_nol1", left_gap=True,
                                                deck_title="Known\\_ODE-latent\\_calcium $\\cdot$ batch 6 result")))
    # Cedric, 2026-10-03: the graphs themselves, as examples, after the neuron-graph method slide (tools/exp17_graph_examples.py)
    jg_ = os.path.join(EXP, "data", "graph_examples.json")
    if os.path.exists(jg_) and os.path.exists(os.path.join(PRES, "figs", "graph_examples.png")):
        deck.append(("11b_graph_examples", frame_wide(
            "the graphs: 20 example neurons and every edge into them",
            # the flow slide's geometry exactly (Cedric, 2026-10-03: the transition must not jump)
            "\\vspace*{1.6\\baselineskip}\\centering\\vspace{0pt}\\topgfx{\\fitgfx[0.70\\textwidth]{figs/graph_examples_grid.png}}\\par"
            "{\\tiny\\color{gray} each cell: from above (the flow maps' place and scale on the next slide) and an oblique "
            "3-D view; streets gold, roads cyan, highways magenta\\par}", "tools/exp17_graph_examples.py",
            deck_title="batch 17 $\\cdot$ the graphs")))
    jfm_ = os.path.join(EXP, "data", "flow_meshes.json")
    if os.path.exists(os.path.join(PRES, "Movies", "flow_meshes.mp4")) and os.path.exists(jfm_):
        FM_ = json.load(open(jfm_))                     # Cedric, 2026-10-04: the 3 meshes' flows against a previous one
        deck.append(("11c_flow_meshes", frame_wide(
            "the flow on the three multi-level meshes against the base graph's",
            "\\vspace*{1.6\\baselineskip}\\centering\\playmovie[0.86\\textwidth]{Movies/flow_meshes}\\par\\vspace{6pt}"
            "{\\tiny\\color{gray} each cell: excitatory flow above, inhibitory below, smoothed over 25 \\textmu m "
            "(tools/exp17\\_wind.py, exp17\\_flow\\_montage.py meshes)\\par}", "tools/exp17_flow_montage.py meshes",
            deck_title="batch 17 $\\cdot$ the flow on the meshes")))
    if os.path.exists(os.path.join(PRES, "Movies", "flow_mesh4_rec.mp4")):     # Cedric, 2026-10-04
        deck.append(("11c_flow_mesh4", frame_wide(
            "the flow of the 4-level mesh over the recorded dF/F",
            "\\vspace*{1.6\\baselineskip}\\centering\\playmovie[0.86\\textwidth]{Movies/flow_mesh4_rec}\\par\\vspace{4pt}"
            "{\\tiny\\color{gray} the learned messages of 17.10 (the 4-level mesh) as wind, excitatory left, inhibitory right, "
            "over the RECORDED dF/F (gridded, smoothed over 25 \\textmu m); tools/exp17\\_wind\\_consensus.py --median-on-recorded "
            "(one run: its own flow)\\par}", "tools/exp17_flow_montage.py mesh4_rec",
            deck_title="batch 17 $\\cdot$ the flow of the 4-level mesh")))
    jdr_ = os.path.join(EXP, "data", "dots_residual_zap_e15_cur_siren.json")        # Cedric, 2026-10-04: slide 14
    if os.path.exists(jdr_):
        DR_ = json.load(open(jdr_))
        bi_, bo_ = DR_["brain_mean_in_block"], DR_["brain_mean_other"]
        deck.append(("13_dots_residual_e15", frame_wide(
            "what the law misses during the dots",
            "\\vspace*{1.0\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.64\\textheight,keepaspectratio]"
            "{figs/dots_residual_zap_e15_cur_siren.png}\\par\\vspace{2pt}"
            "{\\fontsize{5.5}{6.5}\\selectfont The model misses the dots response where it happens. The two maps are "
            f"anti-correlated across neurons (r = {DR_['r_residual_vs_recorded_response']:.2f}): where neurons rise during dots, "
            "the model stays too low. The response is spread over the whole brain, front to back and strongest dorsally, not "
            "confined to the midbrain visual lobes (tectum). That points more to a brain-wide state change than to missing "
            "visual input.\\par}",   # Cedric, 2026-10-04: his wording, smaller
            "tools/exp17_dots_residual.py zap_e15_cur_siren", deck_title="batch 15 $\\cdot$ zap\\_e15\\_cur\\_siren $\\cdot$ the dots")))
    # Cedric, 2026-10-05: 15.14's traces and messages, then their twins on 17.9 (the 3-level mesh, no SIREN, so no
    # Omega_i(t): the senders' message enters the update as it is)
    # Cedric, 2026-10-05: 17.9's three slides again at 5 other sets of locations ("I want to see more examples";
    # tools/exp17_traces.py / exp17_terms.py --sets 5, exp17_messages.py terms_s<k>)
    for run_, suf_, b_, what_, om_, set_ in (
            (("zap_e15_cur_siren_mesh3", "e15m3", "15", "15.14 (the 3-level mesh + SIREN)", True, ""),
             ("zap_g17_mesh3", "g17m3", "17", "17.9 (the 3-level mesh, no SIREN)", False, ""))
            + tuple(("zap_g17_mesh3", f"g17m3_s{k_}", "17", "17.9 (the 3-level mesh, no SIREN)", False, f"_s{k_}")
                    for k_ in range(1, 6))):
        rt_, ab_ = _tex(run_), what_.split()[0]
        loc_ = f", locations set {set_[2:]} of 5" if set_ else ""
        jtr_ = os.path.join(EXP, "data", f"traces_{run_}{set_}.json")
        if os.path.exists(jtr_):
            TQ_ = json.load(open(jtr_))
            jte_ = os.path.join(EXP, "data", f"terms_{run_}.json")
            TE_ = json.load(open(jte_))["medians"] if os.path.exists(jte_) else None
            side_ = ""
            if TE_:
                side_ = (head("the law, per neuron")
                         + "{\\scriptsize $z_i \\leftarrow z_i + f_i\\,\\big((V_i - z_i) + " + ("\\Omega_i(t)\\," if om_ else "")
                         + "m_i(t) + B_i\\cdot u(t)\\big)$"
                           "\\par}\\vspace{4pt}\n"
                         + "{\\scriptsize $z$ the neuron's dF/F normalised by the recording's mean and SD; $f_i$ the step fraction "
                           "its time constant $\\tau_i$ sets; $m_i = \\sum_j W_{ji}\\tanh z_j$ its senders' message; $B_i\\cdot u$ "
                           "the stimulus (0 outside the input mask).\\par}\\vspace{4pt}\n"
                         + "{\\scriptsize \\textbf{leak pull} $V_i - z_i$: how far the neuron is from its own baseline $V_i$; it pulls "
                           "it back. $V_i$ is constant, so it moves exactly as much as the neuron itself.\\par}" + SEC_GAP
                         + head("how much each term moves")
                         + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{}}\n"
                           "neurons & $" + ("\\Omega " if om_ else "") + "m$ & $B\\cdot u$ & $V - z$ \\\\\n\\hline\n"
                         + "".join(f"{lab_} ({TE_[g_]['n']:,}) & {TE_[g_]['network']:.3f} & {TE_[g_]['stimulus']:.3f} & "
                                   f"{TE_[g_]['leak']:.3f} \\\\\n" for lab_, g_ in (("all", "all"), ("input", "input"),
                                                                                   ("non-input", "non_input")))
                         + "\\end{tabular}\\par}\\vspace{2pt}\n"
                         + "{\\tiny\\color{gray} per neuron, the SD over the 2 h of each term (normalised units), the median over "
                           "the neurons; " + ab_ + "'s free rollout\\par}\n")
                side_ = "\\vspace*{2\\baselineskip}\\fitcol{%\n" + side_ + "}"
            body_tr = ("\\vspace*{0.8\\baselineskip}\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.73\\textwidth}\n"
                       "\\centering\\includegraphics[width=\\linewidth,height=0.72\\textheight,keepaspectratio]"
                       "{figs/traces_" + run_ + set_ + ".png}\\par\\vspace{2pt}"
                       "{\\fontsize{4.6}{5.5}\\selectfont " + (f"{TQ_['K']} neurons chosen from the recording alone, blind to the "
                       f"model: the neurons that move split into {TQ_['K']} regions by position, in each the neuron most typical of "
                       "its region (the most correlated with the region's mean trace), numbered head to tail. Their free-rollout "
                       if not set_ else
                       f"Set {TQ_['set']} of {TQ_['n_sets']}: {TQ_['K']} more neurons, at other locations, from the recording alone. "
                       f"The neurons that move split into {TQ_['K'] * TQ_['n_sets']} regions by position, ranked head to tail and "
                       f"dealt in turn to {TQ_['n_sets']} sets, so each set's {TQ_['K']} regions span the brain; in each the most "
                       "typical neuron (the most correlated with its region's mean trace), numbered head to tail. Their free-rollout ")
                       + f"traces over the 2 h, {what_}: Pearson r median {TQ_['r_median']:+.2f}, from "
                       f"{TQ_['r_min']:+.2f} to {TQ_['r_max']:+.2f}."
                       + ("" if suf_ == "e15m3" or set_ else " The same 12 neurons as 15.14's traces slide (picked from the recording).")
                       + "\\par}\n\\end{column}\n\\begin{column}{0.24\\textwidth}\n"
                       + side_ + "\n\\end{column}\n\\end{columns}")
            deck.append((f"13_traces_{suf_}", frame_wide("a few neurons, recorded against learned", body_tr,
                                                       f"tools/exp17_traces.py, exp17_terms.py {run_}" + (" --sets 5" if set_ else ""),
                                                       deck_title=f"batch {b_} $\\cdot$ {rt_} $\\cdot$ traces" + loc_)))
        jms_ = os.path.join(EXP, "data", f"messages_{run_}{set_}.json")         # Cedric, 2026-10-05
        if os.path.exists(jms_):
            MS_ = json.load(open(jms_))["neurons"]
            desc_ = "; ".join(f"{q_['number']} (r {q_['r']:+.2f}): "
                              f"mean |term| leak {q_['mean_abs']['leak']:.3f}, network {q_['mean_abs']['network']:.3f}, stimulus "
                              f"{q_['mean_abs']['stimulus']:.3f}" for q_ in MS_)
            for nm_, f_, ttl_, cap_ in (
                    (f"13_messages_abc_{suf_}", "abc", "three typical neurons: rollout, the update's terms, the total message",
                     "three neurons typical of the table on the traces slide (tools/exp17\\_terms.py: in each group the neuron "
                     "whose three term SDs are the closest to the group's medians; two non-input, front and back, one input). "
                     "a recorded (green) and learned (white) dF/F; b the three terms of the neuron's update, in normalised "
                     "units: the leak pull V - z, the network " + ("$\\Omega$" if om_ else "") + "m, the stimulus B$\\cdot$u; c the total incoming "
                     "message. " + desc_ + "."
                     # Cedric, 2026-10-05: a twin's three come from its own term SDs, so not 15.14's neurons
                     + ("" if suf_ == "e15m3" else f" {ab_}'s own three, from its own term SDs: not the three of 15.14." if not set_
                        else f" Set {set_[2:]}'s three: the same rule inside set {set_[2:]}'s regions only, the first messages "
                             "slide's three left out.")),
                    (f"13_messages_d_{suf_}", "d", "three typical neurons: each sender's message", "")):   # no caption (Cedric)
                deck.append((nm_, frame_wide(
                    ttl_, "\\vspace*{1.0\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.70\\textheight,"
                    f"keepaspectratio]{{figs/messages_{run_}{set_}_{f_}.png}}\\par\\vspace{{2pt}}"
                    + ("{\\fontsize{4.6}{5.5}\\selectfont " + cap_ + "\\par}" if cap_ else ""),
                    f"tools/exp17_messages.py {run_} terms{set_}",
                    deck_title=f"batch {b_} $\\cdot$ {rt_} $\\cdot$ messages" + loc_)))
    jw0_ = os.path.join(GD, "log", "training", "zapbench", "zap_e15_cur", "results", "zap_e15_cur_wind_fields_sigma0.npz")
    h0_ = float(np.load(jw0_)["grid"][2]) if os.path.exists(jw0_) else float("nan")      # the wind grid's cell, um
    for mv_, nm_, ttl_, cap_ in (("flow_meshes_sigma10", "11c_flow_meshes_s10", "the flow on the three meshes and the base, smoothed over 10 um",
                                  "each cell: excitatory flow above, inhibitory below, smoothed over 10 \\textmu m (between "
                                  "the 25-\\textmu m slides and the unsmoothed ones)"),
                                 ("flow_mesh4_rec_sigma10", "11c_flow_mesh4_s10", "the flow of the 4-level mesh, smoothed over 10 um",
                                  "17.10's learned messages as wind, excitatory left, inhibitory right, smoothed over 10 \\textmu m, "
                                  "over the RECORDED dF/F (smoothed over 25 \\textmu m)"),
                                 ("flow_meshes_sigma0", "11c_flow_meshes_s0", "the flow on the three meshes and the base, NOT smoothed",
                                  "each cell: excitatory flow above, inhibitory below, NOT smoothed (the twin of the slide before: "
                                  f"the same messages on {h0_:.1f}-\\textmu m cells, no 25-\\textmu m Gaussian)"),
                                 ("flow_mesh4_rec_sigma0", "11c_flow_mesh4_s0", "the flow of the 4-level mesh, NOT smoothed",
                                  "17.10's learned messages as wind, excitatory left, inhibitory right, NOT smoothed (the twin of "
                                  "the slide before), over the RECORDED dF/F (smoothed over 25 \\textmu m)")):
        if os.path.exists(os.path.join(PRES, "Movies", mv_ + ".mp4")):            # Cedric, 2026-10-04: twins, no smoothing
            deck.append((nm_, frame_wide(ttl_, "\\vspace*{1.6\\baselineskip}\\centering\\playmovie[0.86\\textwidth]{Movies/" + mv_
                                         + "}\\par\\vspace{4pt}{\\tiny\\color{gray} " + cap_ + "; tools/exp17\\_wind.py --sigma " + mv_.split("sigma")[1] + "\\par}",
                                         "tools/exp17_flow_montage.py sigma0", deck_title="batch 17 $\\cdot$ " + ttl_)))
    if os.path.exists(os.path.join(PRES, "Movies", "flow_views_zap_g17_mesh4.mp4")):      # Cedric, 2026-10-04: slide 41
        deck.append(("11c_flow_views", frame_wide(
            "the flow of the 4-level mesh from above and from the side",
            "\\vspace*{1.2\\baselineskip}\\centering\\playmovie[0.80\\textwidth]{Movies/flow_views_zap_g17_mesh4}\\par\\vspace{2pt}"
            "{\\tiny\\color{gray} 17.10's learned messages as wind (smoothed over 25 \\textmu m), excitatory red, inhibitory "
            "blue, over the RECORDED dF/F in grey; from above (head left) and from the side (head left: the same "
            "arrows projected on the sagittal plane); tools/exp17\\_wind\\_views.py\\par}", "tools/exp17_wind_views.py zap_g17_mesh4",
            deck_title="batch 17 $\\cdot$ the flow of the 4-level mesh, two views")))
    # Cedric, 2026-10-06: the flow slides redone on 17.14 (Dale's law as a prior, lambda 1e-3: one sign per sender, so
    # the excitatory and the inhibitory flows are those of distinct senders) and moved to batch 15's section
    FR_ = "zap_g17_mesh3_dale_d1em3" if os.path.exists(os.path.join(PRES, "Movies", "flow_views_zap_g17_mesh3_dale_d1em3_combined.mp4")) else "zap_g17_mesh4"
    fl_ = "17.14 (Dale's law, 3-level mesh)" if FR_ != "zap_g17_mesh4" else "17.10 (the 4-level mesh)"
    fa_ = "17.14" if FR_ != "zap_g17_mesh4" else "17"
    if os.path.exists(os.path.join(PRES, "Movies", f"flow_views_{FR_}_combined.mp4")):   # Cedric, 2026-10-04: slide 42
        deck.append(("11c_flow_views_combined", frame_wide(
            f"the flow of {fl_}, excitatory and inhibitory together",
            f"\\vspace*{{1.2\\baselineskip}}\\centering\\playmovie[0.84\\textwidth]{{Movies/flow_views_{FR_}_combined}}\\par\\vspace{{2pt}}"
            "{\\tiny\\color{gray} one map per view -- from above, oblique from 45 deg above, from the side: excitatory "
            "particles red, inhibitory cyan (tone-mapped, mixed by weight: never white), on the RECORDED dF/F in grey; "
            f"{fl_}'s learned messages as wind; tools/exp17\\_wind\\_views.py\\par}}", f"tools/exp17_wind_views.py {FR_}",
            deck_title=f"batch {fa_} $\\cdot$ the flow, red and blue together")))
    for mv_, nm_, ttl_ in ((f"flow_views_{FR_}_sigma10", "11c_flow_views_s10",
                            f"the flow of {fl_} from above and from the side, smoothed over 10 um"),
                           (f"flow_views_{FR_}_combined_sigma10", "11c_flow_views_combined_s10",
                            f"the flow of {fl_}, red and blue together, smoothed over 10 um")):
        if os.path.exists(os.path.join(PRES, "Movies", mv_ + ".mp4")):              # Cedric, 2026-10-04: twins of 41, 42
            deck.append((nm_, frame_wide(ttl_, "\\vspace*{1.2\\baselineskip}\\centering\\playmovie[" + ("0.84" if "combined" in mv_ else "0.80")
                                         + "\\textwidth]{Movies/" + mv_ + "}\\par\\vspace{2pt}{\\tiny\\color{gray} as the 25-\\textmu m slide, the wind "
                                         "smoothed over 10 \\textmu m; excitatory red, inhibitory blue, the RECORDED dF/F in grey; "
                                         "tools/exp17\\_wind\\_views.py --sigma 10\\par}", "tools/exp17_wind_views.py --sigma 10",
                                         deck_title=f"batch {fa_} $\\cdot$ " + ttl_.split(", ", 1)[-1])))
    for S_, nm_ in ((25, "11c_field_s25"), (10, "11c_field_s10")):          # Cedric, 2026-10-04: the field, not the flows
        mv_ = f"Movies/field_{FR_}_sigma{S_}"                                   # ... as a movie, the flow slides' colours
        f_ = f"figs/field_{FR_}_sigma{S_}.png"
        body_ = ("\\playmovie[0.72\\textwidth]{" + mv_ + "}" if os.path.exists(os.path.join(PRES, mv_ + ".mp4")) else
                 "\\includegraphics[width=\\textwidth,height=0.74\\textheight,keepaspectratio]{" + f_ + "}")
        if os.path.exists(os.path.join(PRES, f_)) or os.path.exists(os.path.join(PRES, mv_ + ".mp4")):
            deck.append((nm_, frame_wide(
                f"the field of {fl_}, smoothed over {S_} um",
                "\\vspace*{1.4\\baselineskip}\\centering" + body_ + "\\par\\vspace{2pt}{\\tiny\\color{gray} the field the "
                "particles ride, frame by frame: the messages $m_{ji} = W_{ji}\\tanh z_j\\,\\Omega_i$ as arrows sender $\\to$ "
                f"receiver, gridded, smoothed over {S_} \\textmu m; excitatory red, inhibitory cyan, length by the square root "
                "of the strength (one scale over the movie), over the RECORDED dF/F in grey; tools/exp17\\_wind\\_fieldmovie.py\\par}",
                f"tools/exp17_wind_fieldmovie.py {FR_}", deck_title=f"batch {fa_} $\\cdot$ the field, {S_} \\textmu m")))
    if os.path.exists(os.path.join(PRES, "Movies", "flow_graphs.mp4")):
        deck.append(("11c_flow_graphs", frame_wide(
            "the flow (the learned messages as wind) on each graph",
            "\\vspace*{1.6\\baselineskip}\\centering\\playmovie[0.70\\textwidth]{Movies/flow_graphs}\\par"
            "{\\tiny\\color{gray} each cell: excitatory flow above, inhibitory below, smoothed over 25 \\textmu m "
            "(tools/exp17\\_wind.py); base = 15.1, the others batch 17\\par}", "tools/exp17_flow_montage.py",
            deck_title="batch 17 $\\cdot$ the flow on each graph")))
    jw_ = os.path.join(EXP, "data", "wind_consensus_g17sel.json")
    if os.path.exists(jw_) and os.path.exists(os.path.join(PRES, "Movies", "flow_summary.mp4")):
        Wd_ = json.load(open(jw_))
        su_, ag_ = Wd_["similarity_summary"], Wd_["agreement_with_the_others"]
        # Cedric, 2026-10-04: the explanation, the sinks block and the footnote cut; then his explanation of the movies
        right_fs = (head("the flow movies")
                    + "{\\scriptsize The flow movie draws the trained model's own messages between neurons as a weather-style "
                      "wind map. At each moment it shows where the network is sending signal, in which direction, and how "
                      "strongly. Each message is an arrow pointing from the sender toward the receiver; the arrows are summed "
                      "and smoothed into a vector field, one for the positive (excitatory) messages, one for the negative "
                      "(inhibitory), so they cannot cancel each other. Particles released in it drift with the field and "
                      "leave fading trails, so the eye reads streamlines.\\par}\\vspace{6pt}\n"
                    + head("similarity of the flow movies") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                      "& excitatory & inhibitory \\\\\n\\hline\n"
                    + f"same graph, another seed & {su_['ex']['seed_ref_movie']:.2f} & {su_['in']['seed_ref_movie']:.2f} \\\\\n"
                    + f"two different graphs (mean) & {su_['ex']['movie_mean_offdiag']:.2f} & {su_['in']['movie_mean_offdiag']:.2f} \\\\\n"
                    + "\\end{tabular}\\par}\\vspace{6pt}\n"
                    + head("each graph against the others' mean") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{}}\n"
                      "& excitatory & inhibitory \\\\\n\\hline\n"
                    + "".join(f"{_tex(r_.replace('zap_', ''))} & {ag_[r_]['ex']:.2f} & {ag_[r_]['in']:.2f} \\\\\n" for r_ in Wd_["runs"])
                    + "\\end{tabular}\\par}\\vspace{6pt}\n")
        deck.append(("11d_flow_summary", frame("the flow: mean and median over graphs", "\\playmovie[0.80\\linewidth]{Movies/flow_summary}",
                                               right_fs, "tools/exp17_wind_consensus.py, exp17_flow_montage.py",
                                               left_gap=True, deck_title="batch 17 $\\cdot$ the flow over graphs")))
    # Cedric, 2026-10-03: the learned constants across the graphs (exp20's heatmap comparison) and the graphs' curves
    jc_ = os.path.join(EXP, "data", "param_compare.json")
    if os.path.exists(jc_):
        PC_ = json.load(open(jc_))
        # Cedric, 2026-10-03: the four constants on one slide (param_compare_all.png)
        cr_ = PC_["corr_with_base"]
        rng_ = {k_: [v_ for n_, v_ in cr_[k_].items() if n_ not in ("zap_e15_cur", "zap_g17_s1", "zap_g17_random")]
                for k_ in cr_}
        deck.append(("11e_param_all", frame_wide(
            "the learned constants on every graph",
            "\\vspace*{0.6\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.76\\textheight,"
            "keepaspectratio]{figs/param_compare_all.png}\\par\\vspace{2pt}"
            "{\\tiny\\color{gray} each row on the base run's colour scale (2nd-98th percentiles); r: per-neuron correlation "
            "with the base 15.1 -- the 6 other spatial graphs: " + "; ".join(
                f"{k_} {min(v_):+.2f} .. {max(v_):+.2f}" for k_, v_ in rng_.items())
            + f"; the random graph: " + ", ".join(f"{k_} {cr_[k_]['zap_g17_random']:+.2f}" for k_ in cr_) + "\\par}",
            "tools/exp17_param_compare.py", deck_title="batch 17 $\\cdot$ the learned constants on every graph")))
    jgc_ = os.path.join(EXP, "data", "graph_curves.json")
    if os.path.exists(jgc_):
        GC_ = json.load(open(jgc_))
        def bmr_(n_):                                   # the brain-mean Pearson r of a graph run
            return bm_metrics(os.path.join(GD, "log", "training", "zapbench", n_, "results", f"{n_}_movie.npz"))["r"]

        def lr_(n_):                                    # the local metric of a graph run (Cedric, 2026-10-05)
            m_ = local_r(os.path.join(GD, "log", "training", "zapbench", n_, "results", f"{n_}_movie.npz"),
                         "zapbench_destripe_ephys")
            return f"{qv(m_['mean'])} $\\pm$ {m_['sd']:.2f}" if m_ else "--"
        right_gc = (head("the graphs, side by side") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{\\hspace{4pt}}r@{}}\n"
                    "& \\multicolumn{2}{c}{brain mean} & per-neuron \\\\\n& r & RMSE & r \\\\\n\\hline\n"
                    + "".join(f"{_tex(v_['label'].split(':')[0])} & {qv(bmr_(k_))} & {v_['bm_rmse']:.4f} & {lr_(k_)} \\\\\n"
                              for k_, v_ in GC_.items())
                    + (("no W (15.9) & " + "{} & {:.4f}".format(
                        *[(qv(v_) if k_ == "r" else v_) for k_, v_ in bm_metrics(os.path.join(
                            landed["zap_e15_now"]["dir"], "results", "zap_e15_now_movie.npz")).items() if k_ in ("r", "rmse")][::-1])
                        + f" & {lr_('zap_e15_now')} \\\\\n") if "zap_e15_now" in landed else "")
                    + "\\end{tabular}\\par}\\vspace{6pt}\n"
                    + "{\\scriptsize Every graph, the random one included, runs free about as well as the base; with no "
                      "network the brain-mean R$^2$ is clearly worse: the network matters, not where its edges go.\\par}\n")
        deck.append(("11f_graph_curves", frame("the graphs: brain-mean dF/F over the 2 h",
                                               "\\panel{figs/graph_curves.png}", right_gc, "tools/exp17_graph_curves.py",
                                               left_gap=True, deck_title="batch 17 $\\cdot$ every graph's free rollout")))
    # Cedric, 2026-10-04: the experiment's summary, exp17 beside exp19, and the outlook -- the deck's last slide
    try:
        def bm_(n, v=""):
            return bm_metrics(os.path.join(landed[n]["dir"], "results", f"{n}{v}_movie.npz"))["r2"]
        b13 = [m_["r2"] for n in landed if n.startswith("zap_r13_") and not n.endswith("_now")
               for m_ in [bm_metrics(os.path.join(landed[n]["dir"], "results", f"{n}_movie.npz"))] if m_]
        GCs = json.load(open(os.path.join(EXP, "data", "graph_curves.json")))
        sims = json.load(open(os.path.join(EXP, "data", "wind_consensus_g17sel.json")))["similarity_summary"]
        PCs = json.load(open(os.path.join(EXP, "data", "param_compare.json")))["corr_with_base"]
        sp_r = [PCs[k][n] for k in ("tau", "V", "B") for n in PCs[k] if n not in ("zap_e15_cur", "zap_g17_s1", "zap_g17_random")]
        gb = [v["bm_r2"] for v in GCs.values()]
        left_s = (head("what exp17 found (17 batches, 183 runs)")
                  + "{\\scriptsize\\begin{itemize}\\setlength\\itemsep{1pt}\n"
                  f"\\item \\textbf{{Forecast.}} The best law (15.4, ephys + SIREN $\\Omega$) follows the brain-mean dF/F "
                  f"of the 2 h free rollout with R$^2$ {bm_('zap_e15_cur_siren'):+.2f} (in sample), "
                  f"{bm_('zap_c16_t2_siren'):+.2f} with the calcium indicator; trained on ZAPBench's training frames "
                  f"(batch 13): {min(b13):+.2f}--{max(b13):+.2f}.\n"
                  "\\item \\textbf{Free rollout.} One recorded frame + the stimulus, nothing else (leak test): the rig of batch 13 "
                  "stays on the recording for the whole 2 h.\n"
                  f"\\item \\textbf{{Network test.}} Brain-mean R$^2$ of 15.4: {bm_('zap_e15_cur_siren'):+.2f} with the network, "
                  f"{bm_('zap_e15_cur_siren', '_W0'):+.2f} with W = 0, {bm_('zap_e15_cur_siren', '_no_stimulus'):+.2f} with no stimulus, "
                  f"{bm_('zap_e15_now'):+.2f} trained with no network.\n"
                  f"\\item \\textbf{{Not a particular network.}} 8 graphs, a random one included: "
                  f"brain-mean R$^2$ {min(gb):+.2f}--{max(gb):+.2f}. Each neuron's constants are the data's (r {min(sp_r):.2f}--{max(sp_r):.2f} "
                  "across graphs); the coupling is needed, its wiring is not identified.\n"
                  f"\\item \\textbf{{The flow is the graph's.}} Flow movies {sims['ex']['seed_ref_movie']:.2f} alike between seeds, "
                  f"{sims['ex']['movie_mean_offdiag']:.2f} / {sims['in']['movie_mean_offdiag']:.2f} between graphs (excitatory / inhibitory).\n"
                  "\\item \\textbf{No help:} the GNN-MLP (drifts; leaky, still runs away), conductance, a learned indicator, "
                  "more substeps; the learned clusters are cuts of continua.\n"
                  "\\end{itemize}\\par}\n")
        right_s = (head("exp17 and exp20 (gut-brain, Chen et al. 2026)")
                   + "{\\scriptsize\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{1.45cm}@{\\hspace{4pt}}"
                     ">{\\raggedright\\arraybackslash}p{2.25cm}@{\\hspace{4pt}}>{\\raggedright\\arraybackslash}p{2.25cm}@{}}\n"
                     "& \\textbf{exp17} & \\textbf{exp20} \\\\\n\\hline\n"
                     "law & known ODE on the neuron graph & the same law, adopted \\\\\n"
                     "neurons & 100,759, 0.914 s, 2 h & 190,346, 1.117 s, 41 min, far noisier \\\\\n"
                     "drive & visual stimulus, always on (+ ephys) & sparse gut-glucose UV pulses, grating, swim \\\\\n"
                     "what the network carries & the brain-wide swings & the evoked gut response (its finding 10) \\\\\n"
                     "W = 0 / no network & swings flatten & response gone: 0.04 / 0.07 of recorded vs 0.84 \\\\\n"
                     "\\end{tabular}\\par}\\vspace{3pt}\n"
                     "{\\scriptsize Shared: the law, the controls (leak test, W = 0, a twin with no network, the brain-mean R$^2$), "
                     "the Euler runaway (exponential step since). In both, a neuron's own leak forgets its input fast and the "
                     "RECURRENCE holds it. Open for exp20: is it a particular wiring, or any "
                     "graph (exp17's random graph does as well)?\\par}\\vspace{5pt}\n"
                   + head("outlook")
                   + "{\\scriptsize\\begin{itemize}\\setlength\\itemsep{1pt}\n"
                     "\\item Identify the wiring: the forecast cannot; a measured connectome (the fish's EM), perturbations, "
                     "or a strong sparsity prior.\n"
                     "\\item exp20: the random-graph null on the gut response; then where it travels (DVC, PBN, idMO).\n"
                     "\\item A held-out SIREN: $\\Omega$ of position and stimulus, not of absolute time.\n"
                     "\\item The leaky GNN-MLP with a bounded message.\n"
                     "\\end{itemize}\\par}\n")
        deck.append(("99_summary", frame("summary and outlook", "\\vspace*{2\\baselineskip}\\fitcol{%\n" + left_s + "}", right_s, "exp17_zapbench_graphcast.md, ## Summary",
                                          deck_title="multi-level GNN on fish 2 $\\cdot$ summary and outlook")))
    except (KeyError, FileNotFoundError) as e_:
        print(f"[summary] not made: {e_}")
    # Cedric, 2026-10-05: the big picture, coarse and specific -- his text, every number read from the results
    try:
        def B_(n, v=""):
            return bm_metrics(os.path.join(landed[n]["dir"], "results", f"{n}{v}_movie.npz"))["r2"]
        GC2_ = json.load(open(os.path.join(EXP, "data", "graph_curves.json")))
        SIM_ = json.load(open(os.path.join(EXP, "data", "wind_consensus_g17sel.json")))["similarity_summary"]["ex"]
        DOT_ = json.load(open(os.path.join(EXP, "data", "dots_residual_zap_e15_cur_siren.json")))
        g17_ = [v["bm_r2"] for k, v in GC2_.items() if "mesh" not in k]
        msh_ = [B_(f"zap_e15_cur_siren_mesh{L_}") for L_ in (3, 4, 5)]
        e154, e159, e151 = B_("zap_e15_cur_siren"), B_("zap_e15_now"), B_("zap_e15_cur")
        e1512, e1513 = B_("zap_e15_cur_siren_noeph"), B_("zap_e15_noeph_now")
        best_ = max(msh_ + [e154])
        from plexus.paths import graphs_data_path as gdp_
        N_NEURONS_DS = int(np.load(gdp_("zebrafish/zapbench_destripe_recording.npz"))["pos_um"].shape[0])
        f_ = "{\\fontsize{5.8}{6.9}\\selectfont "
        h_ = lambda t: "{\\fontsize{7.5}{9}\\selectfont\\textbf{" + t + "}}\\par\\vspace{2pt}"     # noqa: E731
        it_ = lambda xs: ("\\begin{itemize}\\setlength\\itemsep{0.5pt}\\setlength\\parskip{0pt}\n"
                          + "".join(f"\\item {x}\n" for x in xs) + "\\end{itemize}")                    # noqa: E731
        left_b = (h_("Big picture") + f_ + "A simple, interpretable law (each neuron a leaky unit, coupled to others through "
                  "learned weights on a graph) learned from the ZAPBench zebrafish recording can run free for the whole 2 h. "
                  "Starting from one recorded frame and given only the stimulus, it follows the brain-wide activity, with "
                  f"brain-mean R$^2$ up to {best_:+.2f}. The coupling between neurons is necessary, but which wiring it uses is "
                  "not pinned down.\\par}\\vspace{5pt}"
                  + h_("Coarse") + f_ + it_([
                      "\\textbf{The law:} the GraphCast-style mesh model of the early batches gave way to a known ODE "
                      f"directly on the {N_NEURONS_DS:,} neurons. Each neuron has its own time constant $\\tau$, baseline $V$ and "
                      "stimulus weights $B$, and is coupled by weights $W$ on short, mid and long edges.",
                      "\\textbf{Training:} destriped traces, all frames, curriculum to 50 steps ahead.",
                      "\\textbf{The test:} the brain-mean dF/F R$^2$ of a 2 h free rollout, checked against twins trained "
                      "with no network and against $W = 0$.",
                      "\\textbf{What works:} coupling between neurons, and a SIREN modulation $\\Omega$ that rescales each "
                      "neuron's input over time.",
                      "\\textbf{What makes little difference:} the graph's layout (a random graph does as well), the ephys "
                      "channels, and the calcium indicator, MLP messages or conductance synapses.",
                      "\\textbf{What it misses:} the slow brain-wide rise during the dots block."]) + "\\par}")
        right_b = (h_("Specific") + f_ + it_([
                      f"\\textbf{{Network test:}} 15.4 reaches {e154:+.3f} against {e159:+.3f} for its twin trained with no "
                      f"network. $W = 0$ at inference drops it to {B_('zap_e15_cur_siren', '_W0'):+.2f}.",
                      f"\\textbf{{$\\Omega$:}} it adds about {e154 - e151:+.2f} (15.1 {e151:+.3f} $\\to$ 15.4 {e154:+.3f}). "
                      "Part of that may be fitting time, because $\\Omega$ reads absolute time.",
                      f"\\textbf{{Graphs without $\\Omega$ (batch 17):}} all score {min(g17_):+.2f} to {max(g17_):+.2f}, the "
                      f"random graph {GC2_['zap_g17_random']['bm_r2']:+.3f}, the multi-level meshes "
                      f"{max(GC2_[k]['bm_r2'] for k in GC2_ if 'mesh' in k):+.2f}. Retraining the same graph with another seed "
                      f"gives the same flow (cosine {SIM_['seed_ref_time_mean']:.2f}); a different graph gives a different one "
                      f"(about {SIM_['movie_mean_offdiag']:.2f}).",
                      f"\\textbf{{Meshes with $\\Omega$:}} 3 levels {msh_[0]:+.3f}, the best of exp17; 4 levels {msh_[1]:+.3f}; "
                      f"5 levels {msh_[2]:+.3f}.",
                      f"\\textbf{{Ephys:}} with the network, removing it costs only {e154 - e1512:.3f} ({e154:+.3f} $\\to$ "
                      f"{e1512:+.3f}). Without the network it is worth {e159 - e1513:+.2f}. It is fictive motor output fed back "
                      "as input.",
                      "\\textbf{Dots:} the law's error during dots anti-correlates with the recorded dots response "
                      f"(r = {DOT_['r_residual_vs_recorded_response']:.2f}). The response is spread over the whole brain, so a "
                      "brain-wide internal state is more likely than missing visual input."]) + "\\par}")
        deck.append(("99a_big_picture", frame_wide(
            "the big picture",
            "\\vspace*{0.6\\baselineskip}\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.48\\textwidth}\n" + left_b
            + "\n\\end{column}\n\\begin{column}{0.48\\textwidth}\n" + right_b + "\n\\end{column}\n\\end{columns}",
            "the landed runs' results (Cedric, 2026-10-05)",
            deck_title="multi-level GNN on fish 2 $\\cdot$ the big picture")))
    except (KeyError, FileNotFoundError) as e_:
        print(f"[big picture] not made: {e_}")
    # Cedric, 2026-10-05: THE MEAN-FIELD CONTROL, one table, one sentence and the significance tests -- 15.4 (the graph),
    # its W = 0 at inference, 15.17 (15.4's law with the graph's message replaced by each neuron's gain on the brain mean
    # of tanh z) and 15.9 (trained with no W); the numbers and p values from tools/exp17_meanfield_stats.py
    try:
        import torch
        fit_ = {n: torch.load(os.path.join(landed[n]["dir"], "models", "best.pt"), map_location="cpu",
                              weights_only=False)["fitted"] for n in ("zap_e15_cur_siren", "zap_e15_cur_siren_mf")}
        n_edges = sum(v.numel() for k, v in fit_["zap_e15_cur_siren"].items() if k.startswith("state_diffuse.W_"))
        n_gains = fit_["zap_e15_cur_siren_mf"]["state_diffuse.W_mean"].numel()
        g_, w0_, mf_, nw_ = "15.4 graph", "15.4, W = 0 at inference", "15.17 mean field", "15.9 no W"
        TEXT_ = ("\\textbf{Textbook methods.} The 95~\\% intervals are percentile bootstrap confidence intervals (Efron "
                 "1979; Efron \\& Tibshirani 1993, \\emph{An Introduction to the Bootstrap}). The p values shift the "
                 "resampled differences to the null and add 1 to the count, the standard bootstrap test (Hall \\& Wilson "
                 "1991; Davison \\& Hinkley 1997). Resampling time blocks rather than single frames is the moving block "
                 "bootstrap for correlated time series (K\\\"unsch 1989). Pairing, the same resamples for both laws, is "
                 "how machine-learning papers compare two models on one test set (Koehn 2004; Berg-Kirkpatrick et al.\\ "
                 "2012).")
        # Cedric, 2026-10-05: and its twin WITHOUT the brain-mean subtraction (r per neuron on the raw traces)
        # Cedric, 2026-10-05: "per-neuron r", brain mean removed / kept, not "local r"
        for suf_, ml_, nm_, ttl_ in (("", "per-neuron r", "13_meanfield_e15", ""),
                                     ("_raw", "per-neuron r", "13_meanfield_e15_raw", ", brain mean kept")):
            ST_ = json.load(open(os.path.join(EXP, "data", f"meanfield_stats{suf_}.json")))
            L_ = ST_["laws"]
            T_ = {(t["a"], t["b"]): t for t in ST_["tests"]}

            def pq(p):
                return f"p {p:.3f}" if p >= 1e-3 else f"p $<$ {1 / (ST_['resamples'] + 1) * 1.0001:.0e}".replace("e-0", "e-")
            rows_ = ""
            for k in (g_, w0_, mf_, nw_):
                b_, l_ = L_[k]["brain_mean_r"], L_[k]["local_r"]
                big_ = k == g_
                rows_ += ((f"\\rule{{0pt}}{{2.7ex}}{{\\normalsize\\textbf{{{k}}}}}" if big_ else k)
                          + f" & {qv(b_['estimate'], big=big_)} & {qv(l_['estimate'], big=big_)} $\\pm$ {l_['sd_over_neurons']:.2f} \\\\\n")
            t1, t2, t3 = T_[(g_, mf_)], T_[(mf_, nw_)], T_[(g_, w0_)]
            right_mf = (head("the mean-field control, 2 h free rollout") + "{\\scriptsize\\raggedright "
                        + "\\begin{tabular}{@{}l@{\\hspace{6pt}}r@{\\hspace{6pt}}l@{}}\n& brain-mean r & " + ml_ + (", kept" if suf_ else ", removed") + ", mean $\\pm$ SD \\\\\n"
                        "\\hline\n" + rows_ + "\\end{tabular}\\par}\\vspace{3pt}\n"
                        + "{\\scriptsize\\raggedright graph: $m_i = \\sum_j W_{ji}\\tanh z_j$, " + f"{n_edges:,}" + " edge weights; mean "
                        "field: $m_i = a_i\\,\\langle\\tanh z\\rangle$, " + f"{n_gains:,}" + " gains, one per neuron; no W: $m_i = 0$, "
                        "so $\\Omega$ (which scales $m_i$) has nothing to act on. The rest is 15.4's known ODE.\\par}" + SEC_GAP
                        + head("conclusion")
                        + "{\\scriptsize\\raggedright One brain-wide signal does not replace the graph: the mean field's " + ml_ + " is "
                        f"{L_[mf_]['local_r']['estimate']:+.2f} against the graph's {L_[g_]['local_r']['estimate']:+.2f} "
                        f"({pq(t1['local_r']['p'])}), so the learned coupling carries neuron-to-neuron network dynamics.\\par}}"
                        + SEC_GAP + head("the test")
                        + "{\\scriptsize\\raggedright \\textbf{Null (no difference):} two laws follow the recording equally well. \\textbf{Method:} "
                        f"a paired block bootstrap over time: the 2 h cut into {ST_['blocks'] + 1} blocks of {ST_['block_min']:.1f} min, "
                        f"the first (the transient from the recorded start) left out; {ST_['resamples']:,} resamples of the "
                        f"{ST_['blocks']} others, the same blocks for every law; p two-sided. "
                        "Blocks, not frames or neurons: both are correlated, and neuron-wise tests would give p $\\approx$ 0 for any "
                        f"difference. Per-neuron r over the same {ST_['neurons']:,} neurons for every law; a neuron whose learned trace is "
                        "flat scores 0 (with $W = 0$ most neurons relax to their baseline within 5 min). Bars: each law's 95~\\% interval; the "
                        "tests are on the paired differences, so laws whose intervals overlap can still differ.\\par}\\vspace{3pt}\n"
                        + "{\\scriptsize\\raggedright graph $-$ mean field: brain-mean r " + f"{t1['brain_mean_r']['difference']:+.3f} "
                        f"({pq(t1['brain_mean_r']['p'])}), {ml_} {t1['local_r']['difference']:+.3f} ({pq(t1['local_r']['p'])})\\\\\n"
                        f"mean field $-$ no W: {t2['brain_mean_r']['difference']:+.3f} ({pq(t2['brain_mean_r']['p'])}), "
                        f"{t2['local_r']['difference']:+.3f} ({pq(t2['local_r']['p'])})\\\\\n"
                        f"graph $-$ its $W = 0$: {t3['brain_mean_r']['difference']:+.3f} ({pq(t3['brain_mean_r']['p'])}), "
                        f"{t3['local_r']['difference']:+.3f} ({pq(t3['local_r']['p'])})\\\\[2pt]\n"
                        "Retraining with another seed (15.1 against 17.8, the same spec) moves brain-mean r by "
                        f"{ST_['seed_pair']['brain_mean_r']:.3f} and {ml_} by {ST_['seed_pair']['local_r']:.4f}.\\par}}")
            deck.append((nm_, frame_narrow(
                "the mean-field control" + ttl_, f"figs/meanfield_stats{suf_}.png", right_mf,
                f"tools/exp17_meanfield_stats.py{' --raw' if suf_ else ''} (data/meanfield_stats{suf_}.json); models/best.pt",
                deck_title="batch 15.17 $\\cdot$ the mean-field control" + (ttl_ or ": is the coupling network dynamics?"), left=0.50, height=0.70,
                img_top="0.12\\textheight",            # Cedric, 2026-10-05: the bar plot down, in the middle of the slide
                # Cedric, 2026-10-05: where the local metric and the test come from
                caption=("\\fontsize{3.8}{4.6}\\selectfont "      # Cedric, 2026-10-05: smaller than the deck's captions
                         + ("\\textbf{Brain mean kept.} The twin of the previous slide: each neuron's r on its raw traces, the "
                            "brain mean NOT regressed out, so a law also earns credit for the shared brain-wide signal.\\\\[3pt]"
                            if suf_ else "\\textbf{Per-neuron r and global signal regression.} Regressing each trace on the brain "
                            "mean before correlating is global signal regression in fMRI, reviewed with its debate by Murphy \\& "
                            "Fox (2017): it can create artificial negative correlations. In population recordings, neuron-to-neuron "
                            "correlations are routinely computed after removing the population-wide fluctuation, e.g.\\ Okun et "
                            "al.\\ (2015) on ``population coupling''.\\\\[3pt]")
                         + TEXT_))))
    except (KeyError, FileNotFoundError, TypeError) as e_:
        print(f"[mean field] not made: {e_}")
    # Cedric, 2026-10-06: DALE'S LAW AS A PRIOR, on 17.9 (tools/exp17_dale.py --figure -> data/dale_g17.json)
    try:
        DL_ = json.load(open(os.path.join(EXP, "data", "dale_g17.json")))
        lam_ = lambda d: "--" if not d["lambda"] else f"{d['lambda']:.0e}".replace("e-0", "e-")    # noqa: E731
        rows_d = "".join(f"{d['label'].split(',')[0]} & {lam_(d)} & "
                         f"{100 * d['consistent_senders']:.1f} \\% & {d['minority_mass']:.3f} & {100 * d['excitatory_senders']:.0f} \\% & "
                         f"{d['long_mse']:.3f} & {qv(d['brain_mean_r'])} & {qv(d['per_neuron_r'])} \\\\\n" for d in DL_)
        d0, d1 = DL_[0], DL_[1]
        right_d = ("\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.56\\textwidth}\n\\fitcol{%\n"
                   + head("the three priors against 17.9") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r"
                   "@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                   "run & $\\lambda$ & one-sign & minority & excit. & long MSE & brain-mean r & per-neuron r \\\\\n"
                   "& & senders & mass & senders & ($10^{-3}$) & & \\\\\n\\hline\n" + rows_d + "\\end{tabular}\\par}}\n"
                   "\\end{column}\n\\begin{column}{0.40\\textwidth}\n\\fitcol{%\n"
                   + head("the prior") + "{\\scriptsize\\raggedright $\\lambda\\sum_j \\min(P_j, N_j)$, with $P_j$ / $N_j$ the "
                   "positive / negative outgoing weight of sender $j$ summed over every edge set: the weight on each sender's "
                   "minority sign. 17.9's law (3-level mesh, no SIREN) trained from scratch with it.\\par}" + SEC_GAP
                   + head("conclusion") + "{\\scriptsize\\raggedright Dale's law holds as a prior: at $\\lambda = 10^{-3}$, "
                   f"{100 * d1['consistent_senders']:.0f} \\% of the senders are one sign (17.9: {100 * d0['consistent_senders']:.0f} \\%), "
                   f"the brain-mean r unchanged ({d1['brain_mean_r']:+.3f} against {d0['brain_mean_r']:+.3f}), the per-neuron r "
                   f"{d1['per_neuron_r'] - d0['per_neuron_r']:+.3f} and the long MSE {100 * (d1['long_mse'] / d0['long_mse'] - 1):+.1f} \\%. "
                   "The weakest prior gets nearly all of the sign consistency for the least cost.\\par}}\n"
                   "\\end{column}\n\\end{columns}")
        deck.append(("13_dale_g17", frame_wide(
            "Dale's law as a prior", "\\vspace*{0.5\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.42"
            "\\textheight,keepaspectratio]{figs/dale_g17.png}\\par\\vspace{6pt}\\raggedright\n" + right_d,
            "tools/exp17_dale.py --figure (data/dale_g17.json)",
            deck_title="batch 17.14--17.16 $\\cdot$ Dale's law as a prior")))
    except (KeyError, FileNotFoundError) as e_:
        print(f"[dale] not made: {e_}")
    # Cedric, 2026-10-06: where the excitatory and the inhibitory senders are, 17.14 (tools/exp17_dale.py --map)
    jdm_ = os.path.join(EXP, "data", "dale_map_zap_g17_mesh3_dale_d1em3.json")
    if os.path.exists(jdm_):
        DM_ = json.load(open(jdm_))
        right_dm = (head("excitatory and inhibitory senders")
                    + rows([("senders", f"{DM_['senders']:,} (outgoing $|W| > 10^{{-3}}$)"),
                            ("excitatory", f"{DM_['excitatory']:,} ({100 * DM_['excitatory'] / DM_['senders']:.0f} \\%)"),
                            ("inhibitory", f"{DM_['inhibitory']:,} ({100 * DM_['inhibitory'] / DM_['senders']:.0f} \\%)"),
                            ("excitatory weight", f"{100 * DM_['excitatory_mass']:.0f} \\% of the outgoing $|W|$")])
                    + "{\\scriptsize\\raggedright 17.14, Dale's law as a prior ($\\lambda = 10^{-3}$): a sender is excitatory "
                      "when its positive outgoing weight $P_j$ exceeds its negative $N_j$, inhibitory otherwise; its colour "
                      "strength is its net outgoing weight $|P_j - N_j|$ over the 98th percentile. Every neuron at its "
                      "position, head left; the others dark grey.\\par}")
        deck.append(("13_dale_map_g17", frame_wide(
            "excitatory and inhibitory senders", "\\vspace*{0.4\\baselineskip}\\centering\\includegraphics[width=\\textwidth,"
            "height=0.48\\textheight,keepaspectratio]{figs/dale_map_zap_g17_mesh3_dale_d1em3.png}\\par\\vspace{4pt}"
            "\\raggedright\\begin{columns}[T,onlytextwidth]\\begin{column}{0.46\\textwidth}\\centering"
            "\\includegraphics[width=\\linewidth,height=0.30\\textheight,keepaspectratio]{figs/dale_ei_bars.png}"
            "\\end{column}\\begin{column}{0.50\\textwidth}\\fitcol{%\n" + right_dm
            + "}\\end{column}\\end{columns}",     # Cedric, 2026-10-06: the exc / inh split against lambda beside it
            "tools/exp17_dale.py --map zap_g17_mesh3_dale_d1em3 / --ei",
            deck_title="batch 17.14 $\\cdot$ excitatory and inhibitory senders")))
    # Cedric, 2026-10-06: THE ONE-TO-ONE COMPARISONS FROM BATCH 15 -- each row one change between two landed runs
    try:
        def m3_(a_):
            n_ = ARM_RUN[a_]
            res_ = os.path.join(landed[n_]["dir"], "results")
            t_ = json.load(open(os.path.join(res_, f"{n_}_test.json")))
            mm_ = np.asarray(t_["mse_model"], float)
            return (1e3 * mm_[15:32].mean(), bm_metrics(os.path.join(res_, f"{n_}_movie.npz"))["r"],
                    local_r(os.path.join(res_, f"{n_}_movie.npz"), t_.get("trace_recording", "zapbench"))["mean"])
        M_ = {a_: m3_(a_) for a_ in {x for _, a_, b_, _ in ONE_TO_ONE for x in (a_, b_)}}
        rows_o = "".join(f"{lab_} & {a_} $\\to$ {b_} & {100 * (M_[b_][0] / M_[a_][0] - 1):+.1f} \\% & {M_[b_][1] - M_[a_][1]:+.3f} & "
                         f"{M_[b_][2] - M_[a_][2]:+.3f} & {rd_} \\\\\n" for lab_, a_, b_, rd_ in ONE_TO_ONE)
        body_o = ("\\vspace*{0.6\\baselineskip}\\fitcol{%\n" + head("one change at a time, batches 15--18 (in sample, 2 h free rollout)")
                  + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{6pt}}l@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{6pt}}r@{\\hspace{8pt}}l@{}}\n"
                  "change & from $\\to$ to & long MSE & brain-mean r & per-neuron r & reading \\\\\n\\hline\n" + rows_o
                  + "\\end{tabular}\\par}\\vspace{4pt}\n{\\scriptsize\\raggedright Long MSE (h 16--32) in \\% of the ``from'' run's; "
                  "brain-mean r and per-neuron r (brain mean removed) as differences. What counts: another training seed moves "
                  "them by 0.1 \\%, 0.002 and 0.001 (15.1 against 17.8); the recording's own sampling (slide 17's paired "
                  "block bootstrap) by about 0.035 brain-mean r and 0.009 per-neuron r, so differences under $\\sim$0.07 and "
                  "$\\sim$0.02 are not significant. Batch 19 fills the gaps on the new nominal, with the first held-out pair since "
                  "batch 13.\\par}}")
        deck.append(("99b_one_to_one", frame_wide("one change at a time", body_o, "the landed runs' results",
                                                 deck_title="batches 15--18 $\\cdot$ one change at a time")))
    except (KeyError, FileNotFoundError, TypeError) as e_:
        print(f"[one to one] not made: {e_}")
    # Cedric, 2026-10-04: the MULTI-LEVEL MESHES of batch 17 (graph: mesh) -- GraphCast's construction on the neurons, no
    # encoder / decoder (cell_ops.neuron_mesh_levels); tools/exp17_mesh_figures.py makes the turntables (slide 3's
    # renderer), the level panels (GraphCast Fig. 1e, g) and data/gcmesh_<L>.json, read here
    GM_ = {L_: json.load(open(os.path.join(EXP, "data", f"gcmesh_{L_}.json"))) for L_ in (3, 4, 5)
           if os.path.exists(os.path.join(EXP, "data", f"gcmesh_{L_}.json"))}

    def _um(v):
        return f"{v:.0f}" if v >= 10 else f"{v:.1f}"
    if 5 in GM_:
        g5 = GM_[5]["per_level"]
        cubes = " / ".join(f"{l_['cube_um']:g}" for l_ in g5[:-1])
        body_l = ("\\vspace*{1.4\\baselineskip}\\centering\\includegraphics[width=\\textwidth,height=0.56\\textheight,"
                  "keepaspectratio]"      # off the header (Cedric, 2026-10-04)
                  "{figs/gcmesh_5_levels.png}\\par\\vspace{6pt}\n"
                  "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.49\\textwidth}\n{\\tiny "
                  "\\textbf{GraphCast} (Lam et al.\\ 2023, Fig.~1g; icosahedral\\_mesh.py): $M^0$ the icosahedron, each "
                  "level splitting every triangle in 4; the nodes NESTED, a coarse vertex a vertex of every finer level; "
                  "the multi-mesh = the finest level's nodes and EVERY level's edges, the messages along all of them at "
                  "once (Fig.~1e).\\par}\n\\end{column}\n\\begin{column}{0.49\\textwidth}\n{\\tiny "
                  f"\\textbf{{On the neurons}}: $M^0$..$M^{{{len(g5) - 2}}}$ cubes of {cubes} \\textmu m on one origin "
                  "(each cube splits into 8 of the next); a cube's node is the coarser level's node in it, else the "
                  f"neuron nearest its centroid; $M^{{{len(g5) - 1}}}$ every neuron -- the neurons ARE the finest nodes, "
                  "so no encoder / decoder; each level the Delaunay tetrahedralisation of its nodes (3-D triangles), "
                  "the edges longer than 2 of its cubes dropped, every node keeping its shortest. White arrows: the "
                  "edges INTO one node, a node of every level; each level a slab one of its cubes thick, from above."
                  "\\par}\n\\end{column}\n\\end{columns}\n")
        deck.append(("11h_mesh_levels", frame_wide("the multi-level mesh, level by level", body_l,
                                                   "data/gcmesh_5.json, figs/gcmesh_5_levels.png",
                                                   deck_title="batch 17 $\\cdot$ the multi-level mesh, as GraphCast's")))
    for L_, g_ in GM_.items():
        pl_, sd_, mg_ = g_["per_level"], g_["sets_directed"], g_["merged"]
        tab_ = ("{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{5pt}}l@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{\\hspace{5pt}}r@{}}\n"
                "level & nodes of & nodes & edges & median \\\\\n\\hline\n"
                + "".join(f"$M^{i_}$ & {l_['label'].replace(' um', ' \\textmu m')} & {l_['nodes']:,} & {l_['edges']:,} & "
                          f"{_um(l_['median_um'])} \\textmu m \\\\\n" for i_, l_ in enumerate(pl_))
                + "\\end{tabular}\\par}\\vspace{6pt}\n")
        long_lv_ = [f"$M^{i_}$" for i_, l_ in enumerate(pl_) if l_["cube_um"] >= 4 * g_["bin_um"]]
        mid_lv_ = [f"$M^{i_}$" for i_, l_ in enumerate(pl_) if g_["bin_um"] <= l_["cube_um"] <= 2 * g_["bin_um"]]
        sets_ = [("short", f"$M^{len(pl_) - 1}$"), ("mid", ", ".join(mid_lv_)), ("long", ", ".join(long_lv_) or "none")]
        right_m = (head(f"the multi-level mesh, {L_} levels") + tab_
                   + "{\\scriptsize the law's three W sets, every edge both ways:\\par}\\vspace{2pt}\n"
                   # Cedric, 2026-10-05: "2 x" the neighbour pairs, so the count reads against the level table's
                   + rows([(k_, f"{lv_}: 2 $\\times$ {sd_[k_]['edges'] // 2:,} edges, {sd_[k_]['per_neuron']:.2f} per neuron, "
                                    f"mean {_um(sd_[k_]['mean_um'])} \\textmu m" if sd_[k_]["edges"] else
                                 "none: no level of 64 \\textmu m or coarser") for k_, lv_ in sets_])
                   + "{\\scriptsize the merged multi-mesh is ONE graph over "
                   + (f"all {g_['neurons']:,} neurons" if mg_["components"] == 1 else
                      f"{mg_['largest']:,} of the {g_['neurons']:,} neurons ({g_['neurons'] - mg_['largest']} stray "
                      f"neurons in {mg_['components'] - 1} islands of their own)")
                   + "; white: the edges into one node at every level (GraphCast Fig.~1e); the finest level, every "
                     "neuron, is the cloud\\par}\n")
        deck.append((f"11h_mesh{L_}", frame(f"the multi-level mesh, {L_} levels", f"\\playmovie{{Movies/gcmesh_{L_}}}",
                                           right_m, f"data/gcmesh_{L_}.json (tools/exp17_mesh_figures.py)", left_gap=True,
                                           deck_title=f"batch 17 $\\cdot$ the multi-level mesh, {L_} levels")))
    MOVE_TO_END = ("08b_calcium_result", "02b_models", "99a_big_picture", "99_summary")      # Cedric, 2026-10-03: the models table closes the deck       # Cedric, 2026-10-03: the latent-calcium slide now opens batch 16   # Cedric, 2026-10-02 / 10-03: calcium closes the deck
    deck = [x for x in deck if x[0] not in MOVE_TO_END] + [x for x in deck if x[0] in MOVE_TO_END]
    # Cedric, 2026-10-02: one slide of GraphCast results right after the GraphCast slides -- the best GraphCast run's
    # movie (batch 4's curriculum to 40), shown although batch 4's own slides are hidden
    MOVE_AFTER = {"08_latent_calcium": "batch_16_levers", "11b_graph_examples": "batch_17_levers",
                  "11c_flow_graphs": "11b_graph_examples", "11d_flow_summary": "11c_flow_graphs",
                  "11f_graph_curves": "11d_flow_summary", "11e_param_all": "11f_graph_curves",
                  "11h_mesh_levels": "zap_g17_random_movie", "11h_mesh3": "11h_mesh_levels", "11h_mesh4": "11h_mesh3", "11h_mesh5": "11h_mesh4", "zap_g17_mesh3_movie": "11h_mesh5", "zap_g17_mesh4_movie": "zap_g17_mesh3_movie", "zap_g17_mesh5_movie": "zap_g17_mesh4_movie", "11c_flow_meshes": "11b_graph_examples", "zap_gc_cur40_movie": "02d_transfer",   # Cedric: the graphs open batch 17 "13_clusters_3d_dm": "zap_dm_ng_rl1lo_movie",
                  "13_clusters_k4_montage_dm": "13_clusters_3d_dm", "13_clusters_k8_montage_dm": "13_clusters_k4_montage_dm",
                  "13_clusters_k16_montage_dm": "13_clusters_k8_montage_dm", "13_clusters_k32_montage_dm": "13_clusters_k16_montage_dm",
                  "13_edges_amp_dm": "13_clusters_k32_montage_dm",
                  # batch 15's best run, in order (each entry placed after the one before it, so the chain is ordered)
                  "zap_e15_cur_siren_curves": "zap_e15_cur_siren_movie",          # Cedric, 2026-10-03: curves after the movie
                  "zap_e15_cur_siren_omega": "zap_e15_cur_siren_curves", "13_param_maps_e15": "zap_e15_cur_siren_omega",
                  "13_clusters_3d_e15": "13_param_maps_e15", "13_clusters_k4_montage_e15": "13_clusters_3d_e15",
                  "13_cluster_profile_e15": "13_clusters_k4_montage_e15", "13_clusters_k8_montage_e15": "13_cluster_profile_e15",
                  "13_clusters_k16_montage_e15": "13_clusters_k8_montage_e15", "13_clusters_k32_montage_e15": "13_clusters_k16_montage_e15",
                  "13_edges_amp_e15": "13_clusters_k32_montage_e15", "zap_dm_ng_now_movie": "zap_dm_ng_rl1lo_W0"}   # beside the W = 0 slide: the same argument
    # Cedric, 2026-10-04: the meshes' flow slide and the graphs' free-rollout slide after the meshes' results; the
    # learned-constants slide stays after the flow summary
    for k_ in ("11c_flow_graphs", "11d_flow_summary", "11e_param_all", "11f_graph_curves", "11c_flow_meshes"):
        MOVE_AFTER.pop(k_, None)
    MOVE_AFTER.update({"11d_flow_summary": "11b_graph_examples", "11c_flow_graphs": "11d_flow_summary",   # 21 <-> 22
                       "11e_param_all": "11c_flow_graphs", "11c_flow_meshes": "zap_g17_mesh5_movie",
                       "11c_flow_mesh4": "11c_flow_meshes", "11c_flow_views": "11c_flow_mesh4",
                       "11c_flow_views_combined": "11c_flow_views", "11c_field_s25": "11c_flow_views_combined",
                       "11c_flow_meshes_s10": "11c_field_s25",
                       "11c_flow_mesh4_s10": "11c_flow_meshes_s10", "11c_flow_views_s10": "11c_flow_mesh4_s10",
                       "11c_flow_views_combined_s10": "11c_flow_views_s10", "11c_field_s10": "11c_flow_views_combined_s10",
                       "11c_flow_meshes_s0": "11c_field_s10", "zap_e15_cur_siren_noeph_movie": "zap_e15_cur_siren_short_only", "zap_e15_noeph_now_movie": "zap_e15_cur_siren_noeph_movie",
                       "zap_e15_cur_siren_mesh3_movie": "zap_e15_noeph_now_movie",
                       "13_traces_e15m3": "zap_e15_cur_siren_mesh3_movie",
                       "13_messages_abc_e15m3": "13_traces_e15m3", "13_messages_d_e15m3": "13_messages_abc_e15m3",
                       "13_edges_amp_e15m3": "13_messages_d_e15m3",
                       # Cedric, 2026-10-05: 15.14's traces, messages and edges twinned on 17.9, after its movie
                       "13_traces_g17m3": "zap_g17_mesh3_movie", "13_messages_abc_g17m3": "13_traces_g17m3",
                       "13_messages_d_g17m3": "13_messages_abc_g17m3", "13_edges_amp_g17m3": "13_messages_d_g17m3",
                       "zap_e15_cur_siren_mesh3_no_short": "13_edges_amp_e15m3",
                       "zap_e15_cur_siren_mesh3_no_mid": "zap_e15_cur_siren_mesh3_no_short",
                       "zap_e15_cur_siren_mesh3_no_M0": "zap_e15_cur_siren_mesh3_no_mid",
                       "zap_e15_cur_siren_mesh3_no_M1": "zap_e15_cur_siren_mesh3_no_M0",
                       "zap_e15_cur_siren_mesh4_movie": "zap_e15_cur_siren_mesh3_no_M1",
                       "zap_e15_cur_siren_mesh5_movie": "zap_e15_cur_siren_mesh4_movie",
                       "zap_e15_cur_siren_mf_movie": "zap_e15_cur_siren_mesh5_movie",
                       "zap_e15_cur_siren_tau_movie": "zap_e15_cur_siren_mf_movie",
                       "13_dots_residual_e15": "zap_e15_cur_siren_omega", "13_meanfield_e15": "13_edges_amp_e15", "13_meanfield_e15_raw": "13_meanfield_e15", "13_dale_g17": "13_meanfield_e15_raw", "13_dale_map_g17": "13_dale_g17", "99b_one_to_one": "99a_big_picture",   # Cedric, 2026-10-05: 15.4's network test closes its block
                        "11c_flow_mesh4_s0": "11c_flow_meshes_s0",
                       "11f_graph_curves": "11c_flow_mesh4_s0"})
    prev_ = "13_messages_d_g17m3"                         # Cedric, 2026-10-05: the 5 location sets right after it
    for k_ in range(1, 6):
        for p_ in ("13_traces", "13_messages_abc", "13_messages_d"):
            MOVE_AFTER[f"{p_}_g17m3_s{k_}"] = prev_
            prev_ = f"{p_}_g17m3_s{k_}"
    # Cedric, 2026-10-06: the four flow slides (now 17.14's, Dale's law) after the Dale slides, in batch 15's section
    for k_ in ("11c_flow_views_combined", "11c_field_s25", "11c_flow_views_combined_s10", "11c_field_s10"):
        MOVE_AFTER.pop(k_, None)
    MOVE_AFTER.update({"11c_flow_views_combined": "13_dale_map_g17", "11c_field_s25": "11c_flow_views_combined",
                       "11c_flow_views_combined_s10": "11c_field_s25", "11c_field_s10": "11c_flow_views_combined_s10"})
    for nm, after in MOVE_AFTER.items():
        item = next((x for x in deck if x[0] == nm), None)
        if item is not None and any(x[0] == after for x in deck):
            deck = [x for x in deck if x[0] != nm]
            i = next(k for k, x in enumerate(deck) if x[0] == after)
            deck.insert(i + 1, item)
    # Cedric, 2026-10-04: the calcium block (slides 24-26) at the very end, after an appendix slide
    APPENDIX = ("08_latent_calcium", "zap_b18_ca_h50_x25_movie", "zap_b18_ca_h50_x25_curves")   # Cedric, 2026-10-06: 18.7, not 16.3
    apx = [x for n_ in APPENDIX for x in deck if x[0] == n_]
    if apx:
        deck = ([x for x in deck if x[0] not in APPENDIX]
                + [("99z_appendix", frame_wide("appendix", "\\vspace*{0.30\\textheight}\\centering{\\Huge appendix}\\par",
                                               "Cedric, 2026-10-04", deck_title="multi-level GNN on fish 2 $\\cdot$ appendix"))]
                + apx)
    # Cedric, 2026-10-01: the smoke and identity slides stay in the deck file but commented out (not shown); the
    # overview slide closes the deck. A page is the position among the SHOWN slides + 2 (the title page first).
    HIDDEN_SLIDES = {"zap_gc_smoke_movie", "zap_gc_smoke_curves", "zap_gc_persist_movie", "zap_gc_persist_curves", "12_input_method", "12_input_map"}
    # Cedric, 2026-10-02: the three one-step slides merged into the models table (02b) and the multi-mesh GNN panel
    # (02c); their files stay, commented out of all.tex
    HIDDEN_SLIDES |= {"02a_one_step", "05_one_step_current", "06_one_step_known_ode"}
    # Cedric, 2026-10-02: every batch's lever bar-plot slide (slide_batch) commented out, kept in the deck file
    HIDDEN_SLIDES |= {name for name, _ in deck if name.endswith("_levers")}

    def batch_of(name):
        return next((t.split(":")[0].split()[-1] for t, _, arms, _ in BATCHES
                     if name.startswith(t.split(":")[0].replace(" ", "_") + "_") or any(name.startswith(n + "_") for _, n in arms)), None)
    # Cedric, 2026-10-02: batches 1-7's own slides (levers, movie, curves) commented out, kept in the deck file; the
    # method slides (data, mesh, one-step, neuron graph, calcium) and the pooled slides stay shown
    HIDDEN_SLIDES |= {name for name, _ in deck if (batch_of(name) or "99").isdigit() and int(batch_of(name) or 99) <= HIDE_BATCHES_UPTO}
    HIDDEN_SLIDES |= {name for name, _ in deck if batch_of(name) in HIDE_BATCHES}   # single batches hidden (Cedric)
    # (2026-10-02: the one GraphCast results slide was shown; 2026-10-03: commented out again)
    HIDDEN_SLIDES |= {name for name, _ in deck if name.endswith(("_Sleft0", "_Wleft0"))}   # the ablation test, for now
    HIDDEN_SLIDES |= {"08b_calcium_result"}           # Cedric, 2026-10-03: the batch-6 calcium result commented out
    HIDDEN_SLIDES |= {nm for nm, _ in deck if nm.startswith("zap_g17_") and nm.endswith("_curves")}
    HIDDEN_SLIDES |= {"13_clusters_3d_e15", "13_clusters_k16_montage_e15", "13_clusters_k32_montage_e15",
                      "zap_gc_cur40_movie", "zap_b8_lin_W0", "zap_ds_ng_base_W0", "zap_e15_cur_siren_curves",
                      "13_clusters_k4_montage_e15", "13_cluster_profile_e15", "13_clusters_k8_montage_e15",
                      "zap_e15_cur_siren_W0", "zap_e15_cur_siren_lead_left_quarter",
                      "zap_dm_ng_rl1lo_curves", "zap_dm_ng_rl1lo_W0", "zap_dm_ng_now_movie",
                      "zap_v14_cur_siren_movie", "zap_v14_cur_siren_curves", "zap_v14_cur_siren_W0",
                      "zap_v14_cur_siren_lead_left_quarter"}           # Cedric, 2026-10-03
    # Cedric, 2026-10-03: batch 11's cluster and edge slides commented out; batch 15's best run carries them now
    HIDDEN_SLIDES |= {name for name, _ in deck if name.startswith("13_") and name.endswith("_dm")}
    HIDDEN_SLIDES |= {"11_destripe_graph", "zap_ds_ng_base_curves"}    # Cedric, 2026-10-04: slides 10 and 12
    HIDDEN_SLIDES |= {"12_input_kymo", "zap_dm_ng_rl1lo_movie"}        # Cedric, 2026-10-04, "for now": then slides 11 and 12
    HIDDEN_SLIDES |= {f"{n_}_curves" for n_ in EXTRA_RUNS["batch 15"]}  # their movie slides only, as 15.4
    HIDDEN_SLIDES |= {"11d_flow_summary", "11c_flow_graphs", "11h_mesh5", "11c_flow_meshes", "11c_flow_views",
                      "11c_flow_meshes_s0", "11c_flow_mesh4_s0"}        # Cedric, 2026-10-04: slides 21 22 35 39 41 45 46
    HIDDEN_SLIDES |= {"11c_flow_mesh4", "11c_flow_meshes_s10", "11c_flow_mesh4_s10", "11c_flow_views_s10",
                      "11h_mesh4", "zap_g17_mesh5_movie"}              # Cedric, 2026-10-04: slides 36 39 40 41, then 32 35
    HIDDEN_SLIDES |= {"13_edges_amp_e15m3", "zap_e15_cur_siren_mesh3_no_short", "zap_e15_cur_siren_mesh3_no_mid",
                      "zap_e15_cur_siren_mesh3_no_M0", "zap_e15_cur_siren_mesh3_no_M1",
                      "zap_e15_cur_siren_mesh4_movie", "zap_e15_cur_siren_mesh5_movie"}   # Cedric, 2026-10-05: slides 27-33
    HIDDEN_SLIDES |= {"zap_e15_cur_siren_no_short", "zap_e15_cur_siren_no_mid", "zap_e15_cur_siren_no_long",
                      "zap_e15_cur_siren_short_only", "zap_e15_cur_siren_noeph_movie", "zap_e15_noeph_now_movie",
                      "zap_e15_cur_siren_mesh3_movie", "13_traces_e15m3", "13_messages_abc_e15m3",
                      "13_messages_d_e15m3"}             # Cedric, 2026-10-05: then slides 17-26
    HIDDEN_SLIDES |= {"zap_g17_mesh3_tau_movie", "zap_g17_mesh3_dale_d1em3_movie", "zap_g17_mesh3_dale_d3em3_movie",
                      "zap_g17_mesh3_dale_d1em2_movie", "zap_c16_t2_movie", "zap_c16_t2_curves"}   # Cedric, 2026-10-06:
    # 17.12-17.16 as the Dale slide and the one-to-one table; 16.3 replaced in the appendix by 18.7
    HIDDEN_SLIDES |= {"zap_e15_cur_siren_mf_movie", "zap_g17_mf_movie", "zap_g17_mf_curves"}   # Cedric, 2026-10-05: 15.17 shown as
                                                                                       # one slide (13_meanfield_e15)
    for nm, r_ in CURVES_TODO.items():                  # the shown curves slides' figures, panel c the brain mean
        if nm not in HIDDEN_SLIDES:
            fp = os.path.join(PRES, "figs", f"{r_['name']}_curves_bm.png")
            if not curves_bm(r_, fp):
                shutil.copy(os.path.join(r_["dir"], "results", f"{r_['name']}_test.png"), fp)
            if r_["name"] in PANEL_A_ONLY:              # Cedric, 2026-10-04: slide 8 limited to panel a
                from PIL import Image
                im_ = Image.open(fp)
                W_, H_ = im_.size                       # render_curves: panel a's axes at [0.06, 0.55, 0.40, 0.38]
                im_.crop((0, 0, int(0.48 * W_), int(0.52 * H_))).save(fp)
    pages, shown = {}, 0
    for name, _ in deck:
        if name in HIDDEN_SLIDES:
            continue
        shown += 1
        b = batch_of(name)
        if b:
            pages.setdefault(b, []).append(shown + 1)
    pages = {b: (f"{min(v)}-{max(v)}" if len(v) > 1 else str(v[0])) for b, v in pages.items()}
    pass                                                # the overview: deleted from the deck (Cedric, 2026-10-02)
    # Cedric, 2026-10-05: every batch slide's title carries its batch.arm (the md's numbering), the run name dropped
    arm_of = {n_: str(r_["row"].get("batch", "")) for n_, r_ in landed.items() if "." in str(r_["row"].get("batch", ""))}
    tex_of = {_tex(n_): a_ for n_, a_ in arm_of.items()}

    def arms_range(b_, only=None):
        ks = sorted((a_ for n_, a_ in arm_of.items() if a_.split(".")[0] == b_ and (only is None or only in n_)),
                    key=lambda a_: int(a_.split(".")[1]))
        return f"{ks[0]}--{ks[-1].split('.')[0]}.{ks[-1].split('.')[1]}" if len(ks) > 1 else (ks[0] if ks else b_)

    def retitle(body):
        m_ = re.search(r"\\begin\{frame\}\[t\]\{(.*)\}\n", body)
        if not m_:
            return body
        t_ = m_.group(1)
        m1 = re.match(r"batch (\d+) \$\\cdot\$ (zap\S+) \$\\cdot\$ (.*)$", t_)          # an analysis slide of one run
        m2 = re.match(r"(batch \d+\.\d+ \(.*\)) \$\\cdot\$ zap\S+$", t_)                 # a run's results slide
        m3 = re.match(r"batch (\d+) \$\\cdot\$ (.*)$", t_)                                # a batch-wide slide
        if m1 and m1.group(2) in tex_of:
            new = f"batch {tex_of[m1.group(2)]} $\\cdot$ {m1.group(3)}"
        elif m2:
            new = m2.group(1)
        elif m3:
            new = f"batch {arms_range(m3.group(1), 'mesh' if 'mesh' in m3.group(2) else None)} $\\cdot$ {m3.group(2)}"
        else:
            return body
        return body.replace("\\begin{frame}[t]{" + t_ + "}", "\\begin{frame}[t]{" + new + "}", 1)
    for name, body in deck:
        open(os.path.join(PRES, "slides", name + ".tex"), "w").write(retitle(body))
    open(os.path.join(PRES, "slides", "all.tex"), "w").write(
        "".join(("% " if n in HIDDEN_SLIDES else "") + f"\\input{{slides/{n}}}\n" for n, _ in deck))
    stats = {"mesh": MO["stats"], "nn_median_um": float(np.median(nn)), "extent_um": ext.tolist(), "movie": mv}
    json.dump(stats, open(os.path.join(PRES, "slides", "numbers.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    if "--pair" in sys.argv:                        # only the two-dataset movie (Cedric, 2026-10-01)
        print(movie_data_pair(os.path.join(EXP, "mp4", "pair_zapbench_destripe.mp4")))
    elif "--3d" in sys.argv:                        # only the destriped traces in 3-D (Cedric, 2026-10-01)
        print(movie_destripe_3d(os.path.join(EXP, "mp4", "destripe_3d_oblique.mp4")))
    else:
        main()

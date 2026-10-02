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
    fig = plt.figure(figsize=(d["fig_w"], 6.6), facecolor="black")
    tm = np.arange(T_FRAMES) * FRAME_S / 60                          # minutes since the recording's first frame
    cut = np.flatnonzero(np.diff(cond_all)) + 1
    blocks = list(zip(np.r_[0, cut], np.r_[cut, T_FRAMES] - 1))
    sc, cur = [], []
    for j, p in enumerate(panels):
        x0, w = j / n, 1.0 / n
        ax = fig.add_axes([x0 + 0.01, 0.31, w - 0.02, 0.56])           # the labels sit above, never on the brain
        ax.set_facecolor("black")
        ax.axis("off")
        P, o = p["P"], p["order"]
        sc.append(ax.scatter(P[o, 0], P[o, 1], c=np.zeros(len(o)), s=0.5 if len(o) < 90000 else 0.35, cmap="inferno",
                             vmin=0, vmax=vmax, linewidths=0))
        lo, hi = np.percentile(P[:, :2], [0.2, 99.8], 0)               # stray points do not set the frame
        pad = 0.03 * (hi - lo).max()
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_aspect("equal")                                         # the box shrinks to the brain, centred
        ax.plot([hi[0] - 100, hi[0]], [lo[1] - 0.4 * pad] * 2, color="white", lw=1.5)
        ax.text(hi[0] - 50, lo[1] + 0.4 * pad, "100 µm", color="0.7", fontsize=8, ha="center", va="bottom")
        fig.text(x0 + 0.02, 0.975, p["label"], color="white", fontsize=11, va="top")
        m = fig.add_axes([x0 + 0.05 * w, 0.075, 0.90 * w, 0.14])
        m.set_facecolor("black")
        for k_, sp in m.spines.items():
            sp.set_visible(k_ in ("left", "bottom"))
            sp.set_color("0.5")
        for i_, (a_, b_) in enumerate(blocks):
            m.axvspan(tm[a_], tm[b_], color=("0.30" if i_ % 2 else "0.18"), alpha=0.6, lw=0, zorder=0)
            m.text((tm[a_] + tm[b_]) / 2, 1.02, NAMES[cond_all[a_]], color="0.75", fontsize=7, ha="center", va="bottom",
                   transform=m.get_xaxis_transform())
        m.plot(tm, p["mean"], color="#2ca02c", lw=0.6, zorder=2)
        lo_, hi_ = np.nanpercentile(p["mean"], [0.5, 99.5])
        m.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        m.set_xlim(tm[0], tm[-1])
        m.tick_params(colors="0.6", labelsize=7, length=2)
        cur.append(m.axvline(tm[0], color="#ff7f0e", lw=0.9, zorder=4))
        fig.text(x0 + 0.05 * w, 0.255, "brain-mean dF/F over the whole recording; time, min", color="0.7", fontsize=8,
                 va="bottom")
    t_txt = fig.text(0.02, 0.925, "", color="0.7", fontsize=10, va="top")
    for k in ks:
        f = int(frames[k])
        for s_, p in zip(sc, panels):
            s_.set_array(np.asarray(p["X"][k], np.float32)[p["order"]])
        t_txt.set_text(f"condition: {NAMES[cond_all[f]]}   t = {tm[f]:5.1f} min (frame {f:,} of {T_FRAMES:,})")
        for c_ in cur:
            c_.set_xdata([tm[f]] * 2)
        fig.savefig(os.path.join(d["tmp"], f"{k:05d}.png"), dpi=d["dpi"], facecolor="black")
    plt.close(fig)


def zapbench_panel(frames, P=None):
    """ZAPBench's release traces as a movie_brains panel: every frame read once for the brain mean, `frames` kept."""
    X = traces(0, T_FRAMES)
    P = centroids() if P is None else P
    return {"P": brain_view(P), "X": np.ascontiguousarray(X[frames]), "mean": X.mean(1),
            "label": f"ZAPBench release, {P.shape[0]:,} neurons, dF/F"}


def destripe_panel(frames):
    """The destriped zap-inr traces (graphs_data/zebrafish/zapbench_destripe_recording.npz, mapped to ZAPBench's dF/F
    scale) as a movie_brains panel."""
    from plexus.paths import graphs_data_path
    z = np.load(graphs_data_path("zebrafish", "zapbench_destripe_recording.npz"))
    X = z["dff"]
    return {"P": brain_view(z["pos_um"]), "X": np.ascontiguousarray(X[frames]), "mean": X.mean(1),
            "label": f"destriped (zap-inr), {X.shape[1]:,} neurons, mapped to ZAPBench's dF/F scale"}


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


def render_multimesh_3d(P, M, png, mp4=None, n_frames=200, fps=25, labels=True):
    """The multi-mesh in 3-D, to scale in um (VTK, off-screen): the neurons a faint cloud, each level's edges in its
    own colour, wider as the level coarsens, so the long edges that cross the brain in one message stand out over
    the fine lattice. A still at an oblique view (`png`) and, when `mp4` is given, a full turn about the vertical."""
    import shutil
    import tempfile
    import pyvista as pv
    pv.OFF_SCREEN = True
    C = M["C"]
    mm_s, mm_r, lev = M["mm"]
    cols = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d", "#ffffff"][:len(M["nodes_per_level"])]
    width = [1.0, 1.8, 3.0, 4.5, 7.0]
    alpha = [0.18, 0.45, 0.85, 1.0, 1.0]
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
            pl.add_mesh(pv.PolyData((C[used] - mid).astype(np.float32)), color=cols[k], point_size=4 + 3 * k,
                        render_points_as_spheres=True)
    if labels:
        for k in range(len(cols)):
            pl.add_text(f"level {k}: {M['L0'] * 2 ** k:.0f} um edges, {M['nodes_per_level'][k]:,} nodes, "
                        f"{M['edges_per_level'][k]:,} edges", position=(20, 960 - 28 * k), font_size=11, color=cols[k])
    ext = np.ptp(P, 0)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (0.0, 0.0, 1.0)
    r = 1.9 / 0.9 * float(ext.max())             # dezoomed by 0.9 (Cedric): the whole mesh, uncropped

    def cam(az):
        el = np.deg2rad(32.0)
        pl.camera.position = (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el))
    cam(np.deg2rad(-60.0))
    pl.screenshot(png)
    if mp4:
        tmp = tempfile.mkdtemp(prefix="multimesh_")
        for i in range(n_frames):
            cam(np.deg2rad(-60.0) + 2 * np.pi * i / n_frames)
            pl.screenshot(os.path.join(tmp, f"{i:05d}.png"))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264", mp4],
                       check=True)
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


def draw_process(b, P, M, c0, w2, kind="graphcast", scale=1.0, reach=None, nsize=None):
    """The '2 process' panel of figure_step (and, alone and large, of figure_multimesh_gnn): a window 2 w2 um wide,
    one fine cube deep, seen from above -- the neurons in it as grey dots, the finest mesh's edges as straight blue
    lines, the coarser levels' edges as arcs in their own colour (they run along the fine lattice lines, so straight
    they would hide under it) and their nodes as dots. `scale` multiplies the line widths and dot sizes; `reach` (um,
    default w2) the half-width within which mesh nodes are drawn, the view widened to it."""
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
    cols = LEVEL_COLS
    for k in range(min(4, len(M["nodes_per_level"]))):
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
        for a_, b_ in zip(C[mm_s[sel], :2], C[mm_r[sel], :2]):
            d = b_ - a_
            nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            ctrl = (a_ + b_) / 2 + nrm * 0.18 * np.linalg.norm(d)
            b.add_patch(PathPatch(Path([a_, ctrl, b_], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), facecolor="none",
                                  edgecolor=cols[k], lw=(0.8 + 0.7 * k) * scale, alpha=0.95))
    lvl_nodes = M["level_nodes"]
    for k in range(1, min(4, len(lvl_nodes))):
        nk = lvl_nodes[k][in2[lvl_nodes[k]]]
        b.scatter(C[nk, 0], C[nk, 1], s=(8 + 8 * k) * scale ** 2, c=cols[k], linewidths=0, zorder=4)


LEVEL_COLS = ["#9ecae1", "#6baed6", "#fd8d3c", "#e6550d"]   # the multi-mesh levels 0-3 (16, 32, 64, 128 um edges)


def figure_transfer(P, M, path, t=2600):
    """NEURONS TO THE MESH AND BACK (Cedric, 2026-10-02: "explain better neuron to grid, grid to neuron"): two panels
    on the same window of real neurons (a slab one fine cube deep, from above), larger than the one-step figure's --
    a: grid2mesh, every neuron's edges to the cube corners within sqrt(3)/2 L0 (one neuron's drawn thick); b: mesh2grid,
    every neuron's edges from the 8 corners of its cube (the same neuron's thick). Dots: neurons by dF/F at frame t."""
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
    for j, (title, sub) in enumerate((
            ("a   neurons -> mesh (grid2mesh)", f"each neuron sends to the cube corners within {math.sqrt(3) / 2 * L:.1f} µm"),
            ("b   mesh -> neurons (mesh2grid)", "each neuron receives from the 8 corners of its cube"))):
        ax = fig.add_axes([0.02 + 0.49 * j, 0.06, 0.46, 0.78])
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
        ax.add_collection(LineCollection(seg_all, colors=col, linewidths=0.6, alpha=0.45))
        ax.add_collection(LineCollection(seg_one, colors=col, linewidths=3.0))
        ax.scatter(P[sp, 0], P[sp, 1], s=60, c=X[sp], cmap="inferno", vmin=0, vmax=vmax, linewidths=0, zorder=3)
        ax.scatter(C[sm, 0], C[sm, 1], s=140, c="#fd8d3c", marker="s", linewidths=0, zorder=4)
        ax.scatter([P[one, 0]], [P[one, 1]], s=320, facecolors="none", edgecolors="white", linewidths=2, zorder=5)
    fig.text(0.99, 0.01, f"a {2 * w:.0f}-µm window, one cube ({L:.0f} µm) deep, from above; dots: neurons coloured by "
             "dF/F; orange squares: level-0 mesh nodes (cube corners); circled: one neuron and its own edges",
             color="0.55", fontsize=8.5, ha="right")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"window_um": 2 * w, "radius_um": math.sqrt(3) / 2 * L,
            "g2m_one": int((g_s == one).sum()), "m2g_one": int((m_r == one).sum())}


def figure_multimesh_gnn(P, M, path, half_cubes=8.0):
    """THE '2 PROCESS' PANEL ALONE (Cedric, 2026-10-02: isolated from the one-step figure, large): the multi-mesh GNN
    in a window 2 x `half_cubes` fine cubes wide, one cube deep, seen from above, with a legend of the levels."""
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
    draw_process(ax, P, M, c0, w2, "graphcast", scale=2.0, reach=w2 + 0.25 * L, nsize=22.0)
    fig.text(0.03, 0.975, f"the multi-mesh GNN: a {2 * w2:.0f}-µm window, one {L:.0f}-µm cube deep, from above",
             color="white", fontsize=12, va="top")
    fig.text(0.03, 0.94, "grey dots: the neurons in the slab; all edges are ONE graph, every level used in every layer",
             color="0.7", fontsize=9.5, va="top")
    n_lev = min(4, len(M["nodes_per_level"]))
    for k in range(n_lev):
        x, y = 0.03 + 0.48 * (k % 2), 0.085 - 0.035 * (k // 2)
        fig.lines.append(matplotlib.lines.Line2D([x, x + 0.05], [y, y], transform=fig.transFigure,
                                                 color=LEVEL_COLS[k], lw=1.5 + 1.4 * k))
        fig.text(x + 0.065, y, f"level {k}: {L * 2 ** k:.0f}-µm edges, {M['edges_per_level'][k]:,} in the brain",
                 color=LEVEL_COLS[k], fontsize=10, va="center")
    fig.text(0.03, 0.015, "levels 1-3 drawn as arcs (they run along the fine lattice lines), their nodes as dots",
             color="0.55", fontsize=8.5, va="bottom")
    fig.savefig(path, dpi=150, facecolor="black")
    plt.close(fig)
    return {"window_um": 2 * w2, "L0": L}


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


def neuron_graph_op(mid_um=32.0, long_um=128.0, positions_file="zebrafish/zapbench_recording.npz"):
    """The operator itself (state_diffuse[neuron_graph]) builds the graph the slide draws: the picture IS the graph trained."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_contract
    C = get_contract("state_diffuse").implementations["neuron_graph"]
    return C({"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": positions_file,
              "inputs": 1, "short_k": 6, "mid_um": mid_um, "long_um": long_um})


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
    cam(pa, np.deg2rad(-60.0), rr)
    fa = os.path.join(tmp, "a.png")
    pa.screenshot(fa)
    if mp4:
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
                          "tools/exp17_input_neurons.py", deck_title=f"{DECK_TITLE} $\\cdot$ {dt}: kymograph"
                          + (" + ephys" if "ephys" in r_ else ""))
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
        out["12_input_kymo_ephys"] = kymo_slide(re_, "batch 15's 20 \\%: the 10 \\% most coherent with one of the 13 "
                                                     "changing visual features and the 10 \\% most coherent with one of "
                                                     "the 5 ephys features (swim / turn power left and right, bout "
                                                     "fraction); 50 of each shown", "mix")
    elif os.path.exists(os.path.join(fd, f"input_neurons_{re_}_varying_kymo_full.png")):
        out["12_input_kymo_ephys"] = kymo_slide(re_, "the 10 \\% of neurons most coherent with one of the 13 changing visual "
                                                     "features or the 5 ephys features")
    return out


def frame(title, left, right, src, left_gap=False, deck_title=None):
    return (f"% generated by tools/exp17_slides.py from {src} ({title})\n"
            f"\\begin{{frame}}[t]{{{deck_title or DECK_TITLE}}}\n\\vspace*{{\\bandgap}}\n"
            "\\begin{columns}[T,onlytextwidth]\n\\begin{column}{0.58\\textwidth}\n"
            f"{chr(92) + 'vspace*{2' + chr(92) + 'baselineskip}' + chr(10) if left_gap else ''}{left}\n\\end{{column}}\n\\begin{{column}}{{0.4\\textwidth}}\n\\vspace*{{2\\baselineskip}}\n\\fitcol{{%\n{right}}}\n"
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
    for key, what, kind in (("W0", "every edge weight W = 0: no network, each neuron its own leak, rest and stimulus", "W0"),
                            (kl, "no stimulus into the left half (its stimulus weights B = 0), every W kept: "
                                 "stimulus-locked activity left there came through the network from the right"
                                 if kl == "Sleft0" else "W = 0 on every edge INTO the left half", kl)):
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


def slides_run(r):
    """Two slides per landed run: its movie (recorded left, learned right) and its curves against the baselines."""
    import shutil
    n, t, rep = r["name"], r["test"], r["report"]
    bt = r["row"].get("batch", "")
    tag = (f"batch {bt.split('.')[0]}, arm {bt.split('.')[1]}" if "." in bt else f"batch {bt}" if bt else str(r["v"]))
    b0 = bt.split(".")[0] if "." in bt else ""
    dt = (f"batch {b0} ({_tex(BATCH_LAW.get(b0, ''))}) $\\cdot$ arm {bt.split('.')[1]} $\\cdot$ {_tex(n)}" if "." in bt
          else f"{DECK_TITLE} $\\cdot$ {_tex(bt)}" if bt else DECK_TITLE)    # Cedric: the batch and its law in the title
    mv = os.path.join(r["dir"], "results", "movie.mp4")
    out = []
    stages = rep.get("stages") or []
    tr = [("updates", f"{rep.get('iters', 0):,} (horizons {stages[0][0]}..{stages[-1][0]})" if stages else "0"),
          ("time", f"{rep.get('seconds', 0) / 3600:.1f} h"), ("weights", f"{rep.get('n_params', 0):,}")]
    ms = mse_summary(t)
    num = (head(f"{tag}: {n.replace('_', chr(92) + '_')}") +
           "{\\scriptsize " + _tex(r["row"].get("what changed", "")) + "\\par}\\vspace{6pt}\n"
           + head("MSE, $10^{-3}$ dF/F$^2$ (lower is better)") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{\\hspace{7pt}}r@{}}\n"
           + "& h 1-3 & h 16-32 \\\\\n" + "".join(f"{lab} & {_f3(ms[k + '_s'])} & {_f3(ms[k + '_l'])} \\\\\n" for lab, k in (
               ("learned law", "model"), ("mean baseline", "mean"), ("stimulus lookup", "look"),
               ("ZAPBench best", "zb"))) + "\\end{tabular}\\par}\\vspace{2pt}\n"
           + "\\vspace{6pt}\n"
           + head("free rollout, 2 h") + rows([
               ("R$^2$ raw", _r2pm(t['free']['r2_raw'], t['free']['r2_raw_sd'])),
               ("R$^2$ denoised", _r2pm(t['free']['r2_denoised'], t['free']['r2_denoised_sd'])),
               ("exploding", _diverges(t))])
           + head("training") + rows(tr))
    if os.path.exists(mv):
        shutil.copy(mv, os.path.join(PRES, "Movies", f"{n}.mp4"))
        shutil.copy(mv.replace(".mp4", ".png"), os.path.join(PRES, "Movies", f"{n}.png"))
        out.append((f"{n}_movie", frame(f"{tag}: the free rollout of the whole 2 h, recorded left, learned right",
                                        f"\\playmovie{{Movies/{n}}}", num, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
    fig = os.path.join(r["dir"], "results", f"{n}_test.png")
    if os.path.exists(fig):
        shutil.copy(fig, os.path.join(PRES, "figs", f"{n}_test.png"))
        mlc = np.asarray(t["mse_model_by_condition"])[:, 15:32].mean(1) * 1e3
        cond = "".join(f"{nm} & {v:.3f} \\\\\n" for nm, v in zip(t["names"], mlc))
        right = (head(f"{tag}: {n.replace('_', chr(92) + '_')}") + head("long MSE per condition, $10^{-3}$") + "{\\scriptsize\\begin{tabular}{@{}l@{\\hspace{7pt}}r@{}}\n" + cond
                 + "\\end{tabular}\\par}\\vspace{6pt}\n" + head("the scale") + rows([
                     ("persistence, h 1", f"{t['mse_persistence'][0]:.5f}"), ("mean, h 1", f"{t['mse_mean'][0]:.5f}"),
                     ("model, h 1", f"{t['mse_model'][0]:.5f}"), ("mean, h 32", f"{t['mse_mean'][-1]:.5f}"),
                     ("model, h 32", f"{t['mse_model'][-1]:.5f}")]))
        out.append((f"{n}_curves", frame(f"{tag}: the prediction against the mean baseline, step by step",
                                         f"\\panel{{figs/{n}_test.png}}", right, f"log/training/zapbench/{n}", left_gap=True, deck_title=dt)))
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
     (("per sender g(z_j, a_j)", "zap_gm12_snd"), ("per edge g(z_i, z_j, a_i, a_j)", "zap_gm12_pair"), ("per sender, no W prior", "zap_gm12_snd_nol1"), ("per edge, no W prior", "zap_gm12_pair_nol1"), ("per sender, width 16", "zap_gm12_snd_h16"), ("per edge, seed 1", "zap_gm12_pair_s1"), ("per sender, no MLP priors", "zap_gm12_snd_noreg"), ("per sender, reach 64 / 256 um", "zap_gm12_snd_wide")), ()),
    ("batch 13: the new rig on the destriped traces, coherence mask", None,
     (("Euler (new rig)", "zap_r13_eu"), ("exponential", "zap_r13_ex"), ("Euler, rate cap 7.5", "zap_r13_cap"), ("exponential, seed 1", "zap_r13_ex_s1"), ("exp., no W prior", "zap_r13_ex_nol1"), ("exp., linear messages", "zap_r13_ex_lin"), ("exp., adaptation 10 s", "zap_r13_ex_ad"), ("exp., reach 64 / 256 um", "zap_r13_ex_wide")), ("zap_r13_ex", "zap_r13_ex_s1")),
    ("batch 14: modulation and conductance on the destriped traces, coherence mask", None,
     (("current", "zap_v14_cur"), ("conductance", "zap_v14_cond"), ("current + hash Omega", "zap_v14_cur_hash"), ("current + SIREN Omega", "zap_v14_cur_siren"), ("conductance + hash", "zap_v14_cond_hash"), ("conductance + SIREN", "zap_v14_cond_siren"), ("current + coarse hash", "zap_v14_cur_hashlo"), ("current + hash, seed 1", "zap_v14_cur_hash_s1")), ("zap_v14_cur_hash", "zap_v14_cur_hash_s1")),
    ("batch 15: batch 14 on the ephys stimulus (22 visual + 5 swim / turn), ephys-aware mask", "12_input_kymo_ephys",
     (("current", "zap_e15_cur"), ("conductance", "zap_e15_cond"), ("current + hash Omega", "zap_e15_cur_hash"), ("current + SIREN Omega", "zap_e15_cur_siren"), ("conductance + hash", "zap_e15_cond_hash"), ("conductance + SIREN", "zap_e15_cond_siren"), ("current + coarse hash", "zap_e15_cur_hashlo"), ("current + hash, seed 1", "zap_e15_cur_hash_s1")), ("zap_e15_cur_hash", "zap_e15_cur_hash_s1")),
)
# the law of each batch, in the title of every one of its slides (Cedric: which slide belongs to which batch)
BATCH_LAW = {"1": "GraphCast law", "2": "MLP, mesh", "3": "known ODE, mesh", "4": "GraphCast law",
             "5": "known ODE, neuron graph", "6": "known ODE, neuron graph, calcium",
             "7": "known ODE, neuron graph, ZAPBench split", "8": "known ODE, neuron graph, split, adaptation",
             "9": "known ODE, neuron graph, destriped traces", "10": "known ODE, neuron graph, input mask 10 %",
             "11": "known ODE, neuron graph, destriped, coherence mask 10 %",
             "12": "GNN-MLP, neuron graph, destriped", "13": "neuron graph, new rig, destriped",
             "14": "neuron graph, modulation / conductance, destriped",
             "15": "neuron graph, modulation / conductance, destriped + ephys"}            # Cedric's names, 2026-10-01
BATCH_VARIES = {"1": "stimulus, history, embedding, loss, curriculum, mesh levels", "2": "regularisers, synapse, substeps, levels",
                "3": "activation, substeps, levels, synapse", "4": "seed, history 12/24, embedding 2/16, stimulus window, curriculum 5/40",
                "5": "W prior, row lasso, edge reach, no graph", "6": "batch 5 + indicator; indicator start, k fixed",
                "7": "batch 5 on held-out test frames", "8": "seeds, linear messages, W prior, adaptation 10/30 s",
                "9": "batch 7 on the destriped traces (zap-inr)", "10": "batch 7, stimulus into the top 10 % |B_i| only",
                "11": "batch 9, stimulus into the top 10 % by stimulus coherence only",
                "12": "message per sender or per edge, W prior, MLP priors, width, reach",
                "13": "integrator (Euler, exponential, rate cap), seed, W prior, linear, adaptation, reach",
                "14": "conductance, Omega by hash grid or SIREN, in sample",
                "15": "batch 14 with 5 ephys features in the stimulus and the mask"}


SHOW_RUN = {"batch 4": "zap_gc_cur40", "batch 5": "zap_ng_wide", "batch 6": "zap_ca_ng_nol1", "batch 7": "zap_zs_ng_base", "batch 8": "zap_b8_lin", "batch 9": "zap_ds_ng_base", "batch 10": "zap_mk_ng_base", "batch 11": "zap_dm_ng_rl1lo"}   # the run whose movie and curves a batch shows, when not its first arm (the card's)
HIDE_BATCHES_UPTO = 7     # batches whose own slides are commented out of all.tex (Cedric, 2026-10-02)
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
def frame_wide(title, body, src, deck_title=None):
    """A slide with no columns: `body` over the whole text width (the comparison table). It never goes through
    \\fitcol, so its size does not set the deck's one column type size."""
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
            f"edges ({MO['stats']['mm_edges']:,})")
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
    ]
    rows_ = ["recording", "state", "update", "messages", "graph", "learned", "stimulus", "training", "split", "loss"]
    widths = (2.6, 2.6, 2.7, 2.5, 2.0, 1.75, 1.95, 1.85, 2.0, 2.6, 2.05, 2.4)       # cm per column, before the scaling
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
        ("", f"{st['mm_edges']:,} directed edges, ONE graph"),
        ("", "mirror-symmetric about the midline"),
        ("grid2mesh", f"each neuron to the corners within {math.sqrt(3) / 2 * MO['L0']:.1f} \\textmu m ({st['g2m_edges']:,})"),
        ("mesh2grid", f"each neuron from its cube's 8 corners ({st['m2g_edges']:,})")]))
    s2 = frame("The multi-mesh: a nested lattice over the brain, the neurons as the grid",
               "\\panel{figs/02_graphcast_graph.png}", right2, "the law's own mesh (state_diffuse[graphcast].mesh)")
    # ---- the '2 process' panel of the one-step figure alone and large (Cedric, 2026-10-02)
    mg = figure_multimesh_gnn(P, MO, os.path.join(PRES, "figs", "02c_multimesh_gnn.png"))
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
                "\\panel{figs/02c_multimesh_gnn.png}", right2c, "the law's own mesh (state_diffuse[graphcast].mesh)")
    tf = figure_transfer(P, MO, os.path.join(PRES, "figs", "02d_transfer.png"))
    gc_ = yaml.safe_load(open(os.path.join(ROOT, "config", "zapbench", "zap_gc_base.yaml")))
    gco = next(o for o in gc_["operators"] if o.get("op") == "state_diffuse")
    right2d = (head("graph sizes") + rows([
        ("grid2mesh", f"{st['g2m_edges']:,} edges (each neuron to the corners within {tf['radius_um']:.1f} \\textmu m)"),
        ("mesh2grid", f"{st['m2g_edges']:,} edges (each neuron from its cube's 8 corners)")])
        + head("GraphCast law (batches 1, 4): learned transfers") + "{\\scriptsize "
        r"encode $g_i = \phi_g(v_i)$, $e_{ik} = \phi_e(d_{ik})$; "
        r"$e_{ik} \mathrel{+}= \psi(e_{ik}, g_i, m_k)$; $m_k \mathrel{+}= \chi(m_k, \sum_i e_{ik})$\\[2pt]"
        r"decode $e_{ki} \mathrel{+}= \psi'(e_{ki}, m_k, g_i)$; $g_i \mathrel{+}= \chi'(g_i, \sum_k e_{ki})$; "
        r"$x_i(t{+}1) = x_i(t) + \sigma_\Delta\,\delta(g_i)$\\[2pt]"
        f"$v_i$: dF/F $t{{-}}5..t$, position, embedding $a_i$ ({gco.get('embedding_dim', 8)}), stimulus; "
        r"$d_{ik}$: the vector neuron $\to$ corner and its length. Learned: the MLPs $\phi_g, \phi_e, \psi, \chi, \psi', \chi', \delta$ "
        f"(latent {gco.get('latent', 32)}) and $a_i$\\par}}\\vspace{{5pt}}\n"
        + head("MLP law and known ODE on the mesh (batches 2, 3): fixed transfers") + "{\\scriptsize "
        r"to the mesh: $v_k = $ mean of the neurons' $z_i$ within the radius (no weights)\\[2pt]"
        r"back: $\bar m_i = \frac{1}{8}\sum_{k \in \mathrm{cube}(i)} m_k$, then $\Delta z_i$ uses $g_i \bar m_i$ "
        r"(known ODE, $g_i$ learned per neuron) or $f_n(\cdot, \bar m_i)$ (MLP law)\\[2pt]"
        "the learning sits on the mesh ($W$ per edge, node constants) and per neuron, not in the transfers\\par}\n")
    s2d = frame("Neurons to the mesh and back: the encoder's and the decoder's edges", "\\panel{figs/02d_transfer.png}",
                right2d, "the law's own mesh (state_diffuse[graphcast].mesh)", left_gap=True,
                deck_title=f"{DECK_TITLE} $\\cdot$ neurons to the mesh and back")
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
    st = figure_neuron_graph(P, os.path.join(PRES, "figs", "07_neuron_graph.png"))
    right_ng = (head("a known ODE between the neurons (no mesh)") + rows([
        ("one tick", r"$z_i \mathrel{+}= \frac{1}{M}(-z_i + V_i + m_i + B_i \cdot u)/\tau_i$"),
        ("messages", r"$m_i = \sum_{s} \sum_{j \in \mathcal{N}_s(i)} W^{s}_{ji} \tanh(z_j)$"),
        ("substeps", "M = 4 per frame")])
        + head("three edge sets, from positions") + rows([
            (k, f"{v['edges']:,} edges, {v['per_element']:.1f} per neuron, {v['mean_um']:.0f} $\\mu$m")
            for k, v in st.items()])
        + "{\\scriptsize short: the 6 nearest neurons; mid / long: the neuron nearest each of the 6 points"
          " $\\pm$32 / $\\pm$128 $\\mu$m along x, y, z (none if farther than half the reach)\\par}\\vspace{6pt}\n"
        + head("learned") + rows([
            (r"$W^s_{ji}$", "one weight per edge (start 0)"),
            ("neurons", r"$\tau_i$, $V_i$, $B_i$ (22) per neuron")])
        + head("why") + "{\\scriptsize the mesh's pooling carried the whole margin over the lookup and its W nothing;"
          " here every message runs between two real neurons, and a row lasso on $B_i$ can make the stimulus"
          " travel along the edges\\par}\n")
    s_ng = frame("The neuron graph: one weight per edge between two neurons", "\\panel{figs/07_neuron_graph.png}",
                 right_ng, "state_diffuse[model: neuron_graph]", left_gap=True, deck_title="Known\\_ODE-neuron\\_graph")
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
                 left_gap=True, deck_title="Known\\_ODE-latent\\_calcium")
    deck = [("01_zapbench_data", s1), ("02_graphcast_graph", s2)]
    landed = {r["name"]: r for r in results_rows()}
    in_batch = {n for _, _, arms, _ in BATCHES for _, n in arms}
    for r in landed.values():                        # the smoke and the identity run, before any batch
        if r["name"] not in in_batch:
            deck += slides_run(r)
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
            deck += slides_run(landed[base])
            deck += slides_ablation(landed[base], title.split(":")[0])        # W = 0 and left W = 0 (Cedric)
    for make in EXTRA_SLIDES:                           # analysis slides (e.g. ablation, stimulus vs network)
        deck += make(landed)
    # Cedric, 2026-10-02: the pooled slides and the overview deleted from the deck (slides_pool, slide_overview kept)
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
    for name, body in deck:
        open(os.path.join(PRES, "slides", name + ".tex"), "w").write(body)
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

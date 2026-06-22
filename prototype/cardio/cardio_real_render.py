#!/usr/bin/env python
"""Render the three real-cardiomyocyte mp4s from the Utrecht data.

Source (already-computed; we only render, no re-analysis):
  .../graphs_data/cardiomyocytes_real_data/Cardio_1/
    0_B_15kPa_1_MMStack_Pos0.ome.tif            raw microscope movie [240, 2048, 2048] uint16
    0_B_..._.ome.tif.derivatives.npy            [T, 137, 137, 12] float32

Per the dataset description.txt, the derivatives array is D[t, y, x, c] with
  c=0 -> X(t,x,y)   c=1 -> Y(t,x,y)   (absolute grid-point coords in image pixels)
The grid control points (every 15 px -> 137x137 = 18769) ARE the tracked cell
centres of gravity. X(t),Y(t) is a Lagrangian track (displacement from frame 0).

Outputs (into this folder):
  microscope.mp4         the raw movie
  microscope_cog.mp4     raw movie + green COG dots
  cog_trajectories.mp4   the COG tracks (rolling tails)

Run:
  /workspace/.conda_envs/neural-graph-linux/bin/python cardio_real_render.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/groups/saalfeld/home/allierc/GraphData/graphs_data/cardiomyocytes_real_data/Cardio_1"
TIF = os.path.join(DATA, "0_B_15kPa_1_MMStack_Pos0.ome.tif")
DERIV = TIF + ".derivatives.npy"
FFMPEG = "/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg"

DOWN = 2          # display downsample of the 2048 image (-> 1024)
FPS = 24


def _writer(fig, out, fps=FPS):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    if os.path.exists(FFMPEG):
        plt.rcParams["animation.ffmpeg_path"] = FFMPEG
    return FFMpegWriter(fps=fps, bitrate=6000, metadata={"title": "cardio"})


def load_cog():
    """COG tracks [T, N, 2] in image pixel coords (x, y)."""
    d = np.load(DERIV, mmap_mode="r")
    X = np.asarray(d[:, :, :, 0]).reshape(d.shape[0], -1)   # [T, N]
    Y = np.asarray(d[:, :, :, 1]).reshape(d.shape[0], -1)
    return np.stack([X, Y], axis=-1).astype(np.float32)     # [T, N, 2]


def _frames():
    import tifffile
    tf = tifffile.TiffFile(TIF)
    return tf, len(tf.pages)


def _contrast(tf, npages):
    """Global display range from a few sample frames (consistent across movie)."""
    samp = [np.asarray(tf.pages[i].asarray())[::4, ::4]
            for i in (0, npages // 2, npages - 1)]
    s = np.concatenate([x.ravel() for x in samp])
    return float(np.percentile(s, 1.0)), float(np.percentile(s, 99.5))


# Shared dim background for the COG-dots and trajectory movies (so they match).
DIM = 0.42          # 0..1 brightness ceiling for the tissue behind the green overlay


def _dim_frame(img, vmin, vmax, dim=DIM):
    """Normalised, darkened grayscale of a raw frame (data-level dimming)."""
    out = (img.astype(np.float32) - vmin) / max(vmax - vmin, 1.0)
    return np.clip(out, 0.0, 1.0)[::DOWN, ::DOWN] * dim


# --------------------------------------------------------------------------- #
def render_microscope(out=None, cog=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or os.path.join(HERE, "microscope_cog.mp4" if cog else "microscope.mp4")
    tf, npages = _frames()
    vmin, vmax = _contrast(tf, npages)
    track = load_cog() if cog else None
    T = min(npages, track.shape[0]) if cog else npages

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_position([0, 0, 1, 1]); ax.axis("off")
    ax.set_facecolor("black")
    f0 = np.asarray(tf.pages[0].asarray())
    if cog:                                              # dim tissue (matches trajectory movie)
        im = ax.imshow(_dim_frame(f0, vmin, vmax), cmap="gray", vmin=0, vmax=1,
                       extent=[0, 2048, 2048, 0], interpolation="nearest")
    else:                                               # plain microscope: full brightness
        im = ax.imshow(f0[::DOWN, ::DOWN], cmap="gray", vmin=vmin, vmax=vmax,
                       extent=[0, 2048, 2048, 0], interpolation="nearest")
    sc = None
    if cog:
        p = track[0]
        sc = ax.scatter(p[:, 0], p[:, 1], s=1.6, c="lime", linewidths=0, alpha=0.95)
    ax.set_xlim(0, 2048); ax.set_ylim(2048, 0)

    writer = _writer(fig, out)
    with writer.saving(fig, out, dpi=128):
        for t in range(T):
            frame = np.asarray(tf.pages[t].asarray())
            im.set_data(_dim_frame(frame, vmin, vmax) if cog else frame[::DOWN, ::DOWN])
            if cog:
                sc.set_offsets(track[t])
            writer.grab_frame()
    plt.close(fig); tf.close()
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {T} frames)")


def select_grid_nodes(g=10, margin=10):
    """Indices of a g x g evenly-spaced subset of the 137x137 control grid."""
    G = 137
    idx = np.linspace(margin, G - 1 - margin, g).round().astype(int)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")
    return (rows * G + cols).ravel()            # flat indices into the 18769 nodes


def render_trajectories(out=None, grid=10, amp=10.0, background=True, tail=None):
    """A sparse g x g set of grid nodes, trajectories amplified so beats are visible.

    No cell segmentation exists; the tracked control-grid nodes stand in for cell
    centres of gravity. We plot a 10x10 selection and amplify their (Lagrangian)
    displacement about the rest position so the small contraction beats are legible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    out = out or os.path.join(HERE, "cog_trajectories.mp4")
    sel = select_grid_nodes(grid)
    track = load_cog()[:, sel]                  # [T, g*g, 2]
    T = track.shape[0]
    rest = track[0]                             # rest position (frame 0)
    pts = rest[None] + amp * (track - rest[None])   # amplified about rest

    tf = None; vmin = vmax = None
    if background:
        import tifffile
        tf = tifffile.TiffFile(TIF)
        vmin, vmax = _contrast(tf, len(tf.pages))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_position([0, 0, 1, 1]); ax.axis("off")
    ax.set_facecolor("black")
    if background:
        # dim + darken the tissue so the green trajectories read clearly
        bg = ax.imshow(_dim_frame(np.asarray(tf.pages[0].asarray()), vmin, vmax),
                       cmap="gray", vmin=0, vmax=1, extent=[0, 2048, 2048, 0])
    ax.set_xlim(0, 2048); ax.set_ylim(2048, 0)

    lc = LineCollection([], linewidths=1.4)            # all-green, alpha fades along the tail
    ax.add_collection(lc)
    head = ax.scatter(pts[0, :, 0], pts[0, :, 1], s=16, c="lime",
                      edgecolors="black", linewidths=0.4, zorder=3)

    writer = _writer(fig, out)
    with writer.saving(fig, out, dpi=128):
        for t in range(T):
            t0 = 0 if tail is None else max(0, t - tail)         # default: full path
            win = pts[t0:t + 1]                                  # [w, M, 2]
            if win.shape[0] >= 2:
                segs = np.stack([win[:-1], win[1:]], axis=2)     # [w-1, M, 2, 2]
                segs = segs.transpose(1, 0, 2, 3).reshape(-1, 2, 2)
                age = np.tile(np.linspace(0, 1, win.shape[0] - 1), win.shape[1])
                col = np.zeros((age.size, 4), np.float32)
                col[:, 1] = 1.0                                  # green
                col[:, 3] = 0.25 + 0.75 * age                    # older = fainter, recent = bright
                lc.set_segments(list(segs)); lc.set_color(col)
            if background:
                bg.set_data(_dim_frame(np.asarray(tf.pages[min(t, len(tf.pages) - 1)].asarray()), vmin, vmax))
            head.set_offsets(pts[t])
            writer.grab_frame()
    plt.close(fig)
    if tf is not None:
        tf.close()
    print(f"saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {T} frames, "
          f"{grid}x{grid} nodes, amp={amp})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default="all",
                    choices=["all", "microscope", "cog", "traj"])
    ap.add_argument("--amp", type=float, default=10.0, help="trajectory displacement gain")
    ap.add_argument("--grid", type=int, default=10, help="g x g nodes for trajectories")
    args = ap.parse_args()
    if args.which in ("all", "microscope"):
        render_microscope(cog=False)
    if args.which in ("all", "cog"):
        render_microscope(cog=True)
    if args.which in ("all", "traj"):
        render_trajectories(amp=args.amp, grid=args.grid)


if __name__ == "__main__":
    main()

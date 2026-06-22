#!/usr/bin/env python
"""Real cardiomyocyte data: save, load, plot, and render an mp4.

The source is the ParticleGraph cell-cardio dataset, derived from a real
Micro-Manager time-lapse of a cultured cardiomyocyte monolayer on a 15 kPa gel
(``Cardio_1/0_B_15kPa_1_MMStack_Pos0.ome.tif``, optical-flow derivatives). The
raw ``x_list_{run}.npy`` is ``[F, N, 16]`` per frame:

    col 0       particle id        (0 .. N-1)
    cols 1:3    position (x, y)     normalised to [0, 1]
    cols 3:5    velocity (vx, vy)   = displacement / dt  (zero on frame 0)
    cols 5:16   higher-order features used by the GNN (unused here)

This script extracts (pos, vel) into a compact ``cardio_real.npz`` living next to
it in ``prototype/cardio``, so the Plexus side has the ground-truth velocity field
to fit against without reaching back into the ParticleGraph tree.

Usage
-----
    python cardio_data.py save              # write cardio_real.npz
    python cardio_data.py plot --frame 120  # write a single-frame PNG
    python cardio_data.py mp4               # write cardio_real.mp4
    python cardio_data.py all               # save + plot + mp4

Run with the project python:
    /workspace/.conda_envs/neural-graph-linux/bin/python cardio_data.py all
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Source dataset: the ORIGINAL optical-flow derivatives of the Utrecht movie, living in
# GraphData (NOT the ParticleGraph tree). This is the exact same array that
# cardio_real_render.py renders as cog_trajectories.mp4, so the fit GT and the cog video
# share one source of truth. The derivatives are [F, 137, 137, 12]; channels 0,1 are the
# Lagrangian (x, y) positions of the tracked control grid in image pixels.
DEFAULT_SRC = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/"
               "cardiomyocytes_real_data/Cardio_1")
DERIV_NAME = "0_B_15kPa_1_MMStack_Pos0.ome.tif.derivatives.npy"
IMG_PX = 2048.0               # frame size; positions are normalised to [0, 1] by this
DT = 0.04166  # seconds per frame (~24 fps)
NPZ = os.path.join(HERE, "cardio_real.npz")

# Channel layout of the derivatives array.
CH_X, CH_Y = 0, 1


# --------------------------------------------------------------------------- #
#  Save / load
# --------------------------------------------------------------------------- #
def load_raw(src: str = DEFAULT_SRC, run: int = 0) -> dict:
    """Read the GraphData optical-flow derivatives and split out (pos, vel).

    Image-convention coordinates (y increases downward) normalised to [0, 1] by IMG_PX,
    flattened row-major over the 137x137 control grid -> node index = row*137 + col.
    """
    path = os.path.join(src, DERIV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"raw cardio derivatives not found: {path}\n"
            f"expected the GraphData Cardio_1 optical-flow array at {src}"
        )
    d = np.load(path, mmap_mode="r")            # [F, 137, 137, 12]
    F, Hy, Wx, _ = d.shape
    pos = (np.stack([np.asarray(d[:, :, :, CH_X]),
                     np.asarray(d[:, :, :, CH_Y])], axis=-1)
           .reshape(F, Hy * Wx, 2).astype(np.float32) / IMG_PX)       # [F, N, 2] in [0,1]
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / DT                               # displacement / dt
    ids = np.arange(Hy * Wx, dtype=np.int64)                          # [N]
    return {"pos": pos, "vel": vel, "ids": ids, "dt": DT,
            "source": path, "n_frames": F, "n_cells": Hy * Wx}


def save_npz(out: str = NPZ, src: str = DEFAULT_SRC, run: int = 0) -> dict:
    """Extract the compact ground-truth field and write it next to this script."""
    d = load_raw(src, run)
    np.savez_compressed(
        out, pos=d["pos"], vel=d["vel"], ids=d["ids"],
        dt=np.float32(d["dt"]), source=np.array(d["source"]),
    )
    mb = os.path.getsize(out) / 1e6
    print(f"saved {out}  ({mb:.1f} MB)  "
          f"frames={d['n_frames']} cells={d['n_cells']} dt={d['dt']}")
    return d


def load(path: str = NPZ) -> dict:
    """Load the compact cardio_real.npz (run `save` first if missing)."""
    if not os.path.exists(path):
        print(f"{path} missing -- extracting from source first")
        return save_npz(path)
    z = np.load(path, allow_pickle=True)
    return {"pos": z["pos"], "vel": z["vel"], "ids": z["ids"],
            "dt": float(z["dt"]), "source": str(z["source"]),
            "n_frames": z["pos"].shape[0], "n_cells": z["pos"].shape[1]}


# --------------------------------------------------------------------------- #
#  Plot
# --------------------------------------------------------------------------- #
def _speed(vel_f: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vel_f, axis=-1)


def plot_frame(data: dict, frame: int, ax=None, vmax: float | None = None,
               quiver_stride: int = 60):
    """Scatter cells coloured by speed, with a coarse velocity quiver overlay."""
    import matplotlib.pyplot as plt

    pos, vel = data["pos"], data["vel"]
    frame = int(np.clip(frame, 0, pos.shape[0] - 1))
    p, v = pos[frame], vel[frame]
    spd = _speed(v)
    if vmax is None:
        # contraction is episodic; scale to active beats, not the global 99th pct
        nz = _speed(vel.reshape(-1, 2))
        vmax = float(np.percentile(nz[nz > 0], 90))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("black")
    # gamma<1 brightens the low-speed bulk so quiet cells stay visible
    norm = plt.matplotlib.colors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    sc = ax.scatter(p[:, 0], p[:, 1], c=spd, s=3, cmap="magma",
                    norm=norm, linewidths=0)
    s = quiver_stride
    ax.quiver(p[::s, 0], p[::s, 1], v[::s, 0], v[::s, 1],
              color="cyan", alpha=0.6, scale=vmax * 25, width=0.0025)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"cardiomyocyte velocity field  -  frame {frame}/{pos.shape[0]-1}"
                 f"  (t={frame*data['dt']:.2f}s)", color="white", fontsize=10)
    return ax, sc, vmax


def save_plot(data: dict, frame: int, out: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or os.path.join(HERE, f"cardio_frame_{frame:03d}.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_frame(data, frame, ax=ax)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"saved {out}")
    return out


# --------------------------------------------------------------------------- #
#  MP4
# --------------------------------------------------------------------------- #
def render_mp4(data: dict, out: str | None = None, fps: int = 24,
               stride: int = 1, quiver_stride: int = 60):
    """Render the velocity field over time to an mp4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    # Use the conda-env ffmpeg (no system binary in the devcontainer).
    for cand in ("/workspace/.conda_envs/neural-graph-linux/bin/ffmpeg",):
        if os.path.exists(cand):
            plt.rcParams["animation.ffmpeg_path"] = cand
            break
    else:
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    out = out or os.path.join(HERE, "cardio_real.mp4")
    pos, vel = data["pos"], data["vel"]
    frames = range(0, pos.shape[0], stride)
    vmax = float(np.percentile(_speed(vel.reshape(-1, 2)), 99))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax, sc, _ = plot_frame(data, 0, ax=ax, vmax=vmax, quiver_stride=quiver_stride)
    qv = ax.collections[-1]  # the quiver
    s = quiver_stride

    writer = FFMpegWriter(fps=fps, bitrate=4000, metadata={"title": "cardio"})
    with writer.saving(fig, out, dpi=130):
        for f in frames:
            p, v = pos[f], vel[f]
            sc.set_offsets(p)
            sc.set_array(_speed(v))
            qv.set_offsets(p[::s])
            qv.set_UVC(v[::s, 0], v[::s, 1])
            ax.set_title(
                f"cardiomyocyte velocity field  -  frame {f}/{pos.shape[0]-1}"
                f"  (t={f*data['dt']:.2f}s)", color="white", fontsize=10)
            writer.grab_frame()
    plt.close(fig)
    mb = os.path.getsize(out) / 1e6
    print(f"saved {out}  ({mb:.1f} MB, {len(list(frames))} frames @ {fps}fps)")
    return out


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["save", "plot", "mp4", "all"], default="all",
                    nargs="?")
    ap.add_argument("--src", default=DEFAULT_SRC, help="source ParticleGraph dataset dir")
    ap.add_argument("--run", type=int, default=0, help="which x_list_{run}.npy")
    ap.add_argument("--frame", type=int, default=120, help="frame for `plot`")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="frame stride for mp4")
    args = ap.parse_args()

    if args.cmd in ("save", "all"):
        save_npz(NPZ, args.src, args.run)
    data = load(NPZ)
    if args.cmd in ("plot", "all"):
        save_plot(data, args.frame)
    if args.cmd in ("mp4", "all"):
        render_mp4(data, fps=args.fps, stride=args.stride)


if __name__ == "__main__":
    main()

"""Render an RPS *mix* run (field + optional particles) to <name>.gif + montage."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rps_render


def render_mix(name, out, fps=25, out_dir="."):
    frames = out["frames"]; parts = out["parts"]; n = frames.shape[-1]
    T = frames.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(rps_render.to_rgb(frames[0]), origin="lower", interpolation="nearest",
                   extent=[0, n, 0, n])
    has_p = parts[0] is not None
    sc = ax.scatter([], [], s=8, c="white", edgecolors="black", linewidths=0.4, alpha=0.95) if has_p else None
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(rps_render.to_rgb(frames[fr]))
        if has_p:
            sc.set_offsets(parts[fr])
        return [im]

    path = os.path.join(out_dir, name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    _montage(name, out, out_dir)
    return path


def _montage(name, out, out_dir):
    frames, parts = out["frames"], out["parts"]; n = frames.shape[-1]
    T = frames.shape[0]; idx = [0, int(T * 0.08), int(T * 0.2), int(T * 0.4), int(T * 0.7), T - 1]
    tiles = []
    for i in idx:
        fig, ax = plt.subplots(figsize=(2, 2)); fig.subplots_adjust(0, 0, 1, 1)
        ax.imshow(rps_render.to_rgb(frames[i]), origin="lower", extent=[0, n, 0, n], interpolation="nearest")
        if parts[i] is not None:
            ax.scatter(parts[i][:, 0], parts[i][:, 1], s=4, c="white", edgecolors="black", linewidths=0.2, alpha=0.9)
        ax.set_xlim(0, n); ax.set_ylim(0, n); ax.axis("off")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        tiles.append(Image.fromarray(buf[..., :3].copy())); plt.close(fig)
    w, h = tiles[0].size; m = Image.new("RGB", (w * len(tiles), h), "black")
    for k, t in enumerate(tiles):
        m.paste(t, (k * w, 0))
    m.save(os.path.join(out_dir, name + "_montage.png"))

"""Render an RPS reaction-diffusion run to <name>.gif + <name>_montage.png.

    PYTHONPATH unneeded -- self-contained.
    python rps_render.py <name> [<name> ...]      # loads scenarios/<name>.yaml

For S=3 the three species map to R,G,B (the canonical RPS view: spiral arms are
coloured by who is winning locally). For S != 3 the dominant species is coloured
from a cyclic colormap, with brightness set by its local dominance.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rps_engine


def to_rgb(frame):
    """frame [S,n,n] -> image [n,n,3] in [0,1] (x horizontal, y vertical)."""
    S = frame.shape[0]
    if S == 3:
        img = np.stack([frame[0], frame[1], frame[2]], -1)
        img = img / (img.max() + 1e-6)
    else:
        dom = frame.argmax(0)
        val = frame.max(0)
        colors = matplotlib.colormaps["hsv"](np.linspace(0, 1, S, endpoint=False))[:, :3]
        img = colors[dom] * np.clip(val / (val.max() + 1e-6), 0, 1)[..., None]
    return np.clip(img, 0, 1).transpose(1, 0, 2)       # [ix,iy,c] -> [iy,ix,c]


def render_frames(name, frames, fps=25, out_dir="."):
    T = frames.shape[0]
    fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(to_rgb(frames[0]), origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    def upd(fr):
        im.set_data(to_rgb(frames[fr]))
        return [im]

    path = os.path.join(out_dir, name + ".gif")
    FuncAnimation(fig, upd, frames=T, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    _montage(name, frames, out_dir)
    return path


def _montage(name, frames, out_dir):
    T = frames.shape[0]
    idx = [0, int(T * 0.08), int(T * 0.2), int(T * 0.4), int(T * 0.7), T - 1]
    tiles = [(to_rgb(frames[i]) * 255).astype(np.uint8) for i in idx]
    h, w, _ = tiles[0].shape
    m = Image.new("RGB", (w * len(tiles), h), "black")
    for k, t in enumerate(tiles):
        m.paste(Image.fromarray(t), (k * w, 0))
    m.save(os.path.join(out_dir, name + "_montage.png"))


def render(name, device="cuda", fps=25, out_dir="."):
    spec = rps_engine.load(os.path.join(out_dir, "scenarios", name + ".yaml"))
    out = rps_engine.run(spec, device=device)
    p = render_frames(name, out["frames"], fps=fps, out_dir=out_dir)
    print(f"[done] {name}  T={out['frames'].shape[0]}  S={out['frames'].shape[1]}", flush=True)
    return p


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for nm in (sys.argv[1:] or ["rps_random"]):
        render(nm, out_dir=here)

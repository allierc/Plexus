"""embryo_render -- the "cells in a blob of water" view for the embryogenesis prototypes.

The active-matter agents are CELLS: plotted as filled DOTS coloured by type (not bird
triangles). The MPM material is the BLOB: a soft translucent density field (blue for water,
warm for an elastic body) the cells swim/are-carried inside. Black background throughout.

Outputs (2D): `blob_evolution.png` (5-timepoint montage) and, if movie=True, `blob.mp4`.
Reads only positions from trajectory.npz, so it is cheap and runs after every tune.
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile


def _ffmpeg():
    """Locate ffmpeg: next to the interpreter, on PATH, or the imageio_ffmpeg binary."""
    cand = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if os.path.exists(cand):
        return cand
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def _type_colors(sim):
    # cells must POP on the blue material: red / gold / green / white by default
    types = list(sim.sets["agent"]["types"].keys())
    pcol = (sim.plotting or {}).get("colors", {})
    default = [[1.0, 0.28, 0.28], [1.0, 0.85, 0.2], [0.4, 1.0, 0.5], [1.0, 1.0, 1.0]]
    return [pcol.get(t, default[i % len(default)]) for i, t in enumerate(types)]


def _blob_cmap(sim):
    # the MPM material is ALWAYS blue (water or elastic tissue alike)
    return np.array([0.25, 0.5, 0.95])


def _blob_layer(ax, X, W, rgb, nbin, sigma, amax=0.85, z=1):
    if X.shape[0] == 0:
        return
    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=[int(nbin * W), nbin], range=[[0, W], [0, 1]])
    H = gaussian_filter(H.T, sigma)
    if H.max() <= 0:
        return
    dens = np.clip(H / (0.6 * H.max()), 0, 1)
    rgba = np.zeros(dens.shape + (4,))
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = rgb
    rgba[..., 3] = dens * amax
    ax.imshow(rgba, extent=[0, W, 0, 1], origin="lower", interpolation="bilinear", zorder=z)


def _draw(ax, mpos, apos, atype, colors, blob_rgb, W, nbin=220, sigma=2.2, mem_mask=None):
    ax.set_facecolor("black")
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    # --- the material BLOB: one blue, or two blues (membrane vs inner core) ---
    if mem_mask is not None:
        _blob_layer(ax, mpos[~mem_mask], W, [0.35, 0.65, 1.0], nbin, sigma, amax=0.75, z=1)  # inner core: light blue
        _blob_layer(ax, mpos[mem_mask], W, [0.10, 0.25, 0.70], nbin, sigma, amax=0.95, z=2)  # membrane: deep blue
    else:
        _blob_layer(ax, mpos, W, blob_rgb, nbin, sigma, amax=0.85, z=1)
    # --- the CELLS: filled dots coloured by type (size adapts to cell count) ---
    N = apos.shape[0]
    s = float(max(1.5, 7.0 * (400.0 / max(N, 1)) ** 0.5))
    for ti, col in enumerate(colors):
        m = atype == ti
        ax.scatter(apos[m, 0], apos[m, 1], s=s, c=[col], edgecolors="none", alpha=0.95, zorder=3)


def render_blob(sim, data_dir, movie=True, n_panels=5):
    tr = np.load(os.path.join(data_dir, "trajectory.npz"))
    ap = tr["agent__pos"]; at = tr["agent__node_type"]; mp = tr["mpm_particle__pos"]
    occ = tr["agent__occ"] if "agent__occ" in tr.files else np.ones(ap.shape[:2], bool)
    W = float(tr["world_size"][0]) if "world_size" in tr.files else 1.0
    colors = _type_colors(sim); blob = _blob_cmap(sim)
    T = ap.shape[0]

    def live(t):                                              # only occupied cells (division-aware)
        m = occ[t] > 0
        return ap[t][m], at[m]

    # two-blue material (membrane vs inner core) if the disc has a liquid layer -> split by INITIAL radius
    two_blue = any("liquid" in str(t.get("layers", "")) for t in sim.sets.get("cell", {}).get("types", {}).values())
    mem = None
    if two_blue:
        cc = np.array([0.5, 0.5]); r0 = np.linalg.norm(mp[0] - cc, axis=1)
        mem = r0 > 0.80 * np.quantile(r0, 0.99)               # outer shell = membrane

    # 5-timepoint evolution montage
    ts = np.linspace(0, T - 1, n_panels).astype(int)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels, 3.2))
    fig.patch.set_facecolor("black")
    for k, t in enumerate(ts):
        aL, atL = live(t)
        _draw(axes[k], mp[t], aL, atL, colors, blob, W, mem_mask=mem)
        axes[k].set_title(f"t={t}  n={aL.shape[0]}", color="white", fontsize=9)
    fig.tight_layout()
    out_png = os.path.join(data_dir, "blob_evolution.png")
    fig.savefig(out_png, dpi=120, facecolor="black"); plt.close(fig)

    if movie:
        tmp = tempfile.mkdtemp()
        for t in range(T):
            fig, ax = plt.subplots(figsize=(5, 5)); fig.patch.set_facecolor("black")
            aL, atL = live(t)
            _draw(ax, mp[t], aL, atL, colors, blob, W, mem_mask=mem)
            fig.savefig(os.path.join(tmp, f"f{t:05d}.png"), dpi=110, facecolor="black")
            plt.close(fig)
        out_mp4 = os.path.join(data_dir, "blob.mp4")
        ff = _ffmpeg()
        if ff:
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "25",
                            "-i", os.path.join(tmp, "f%05d.png"),
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", out_mp4],
                           check=False)
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    return out_png

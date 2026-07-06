#!/usr/bin/env python
"""sheet_3d -- cell_polarity in 3D: apical constriction -> a 3D INVAGINATION PIT, with Voronoi cells.

The 3D counterpart of run_sheet: a disc-shaped epithelial sheet (cells on a hex lattice, each a
3D Voronoi-cell prism between a basal and an apical surface). A central patch is given apico-basal
polarity -- its cells apically CONSTRICT (shrink in-plane), ELONGATE, and SINK -- so the sheet
buckles into a rounded 3D pit (gastrulation / neural-tube in 3D). Rendered as transparent 3D
Voronoi prisms (patch highlighted), rotating camera, on black.

    python sheet_3d.py
"""
from __future__ import annotations
import os, math

import numpy as np
from scipy.spatial import Voronoi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FFMpegWriter
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "archive_sheet_3d")
R_DISC = 6.0                                             # sheet radius
R_PATCH = 2.2                                            # apical-constriction domain radius
H0 = 1.2                                                 # resting cell height


def hex_disc(R, a=1.0):
    """Cell centres on a hex lattice inside a disc of radius R."""
    pts = []
    ny = int(R / (a * math.sqrt(3) / 2)) + 2
    for j in range(-ny, ny + 1):
        y = j * a * math.sqrt(3) / 2
        off = (a / 2) if (j % 2) else 0.0
        nx = int(R / a) + 2
        for i in range(-nx, nx + 1):
            x = i * a + off
            if x * x + y * y <= R * R:
                pts.append((x, y))
    return np.array(pts)


def voronoi_polys(xy, R):
    """2D Voronoi polygon per cell (bounded by a ghost ring), ordered CCW."""
    ang = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    ghost = np.stack([(R + 2.5) * np.cos(ang), (R + 2.5) * np.sin(ang)], 1)
    vor = Voronoi(np.vstack([xy, ghost]))
    polys = []
    for i in range(len(xy)):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            polys.append(None); continue
        v = vor.vertices[reg]
        c = v.mean(0)
        polys.append(v[np.argsort(np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0]))])
    return polys


def invaginate(xy0, r, t, depth=4.5, constrict=0.5, elong=1.4):
    """Deterministic apico-basal program: within the patch, cells constrict in-plane, elongate,
    and sink; a smooth radial profile makes a rounded pit. Returns (xy, z_center, height)."""
    w = np.clip(1.0 - (r / R_PATCH) ** 2, 0, 1) ** 1.5      # smooth patch profile (1 centre -> 0 edge)
    s = t                                                  # ramp 0->1
    scale = 1.0 - constrict * w * s                        # in-plane constriction toward centre
    xy = xy0 * scale[:, None]
    z = -depth * w * s                                     # sink into the pit
    h = H0 * (1.0 + elong * w * s)                         # apicobasal elongation
    return xy, z, h


def prism_faces(poly, zc, h):
    """Quad faces (top + walls) of a Voronoi-cell prism [zc-h/2, zc+h/2]."""
    zt, zb = zc + h / 2, zc - h / 2
    faces = [[(p[0], p[1], zt) for p in poly]]            # apical (top) face
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        faces.append([(a[0], a[1], zb), (b[0], b[1], zb), (b[0], b[1], zt), (a[0], a[1], zt)])
    return faces


def render(outdir, nframes=120, seconds=16.0):
    os.makedirs(outdir, exist_ok=True)
    xy0 = hex_disc(R_DISC)
    r = np.linalg.norm(xy0, axis=1)
    patch = r < R_PATCH
    RED = (0.95, 0.35, 0.2); BLUE = (0.3, 0.55, 0.9)
    def draw(ax, t, azim):
        ax.clear(); ax.set_facecolor("black")
        xy, z, h = invaginate(xy0, r, t)
        polys = voronoi_polys(xy, R_DISC)
        tris, cols = [], []
        for i, poly in enumerate(polys):
            if poly is None or len(poly) < 3:
                continue
            base = RED if patch[i] else BLUE
            depth_shade = 0.55 + 0.45 * np.clip(1 + z[i] / 5.0, 0, 1)
            rgba = (base[0] * depth_shade, base[1] * depth_shade, base[2] * depth_shade, 0.7)
            for f in prism_faces(poly, z[i], h[i]):
                tris.append(f); cols.append(rgba)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=(1, 1, 1, 0.08),
                                             linewidths=0.15))
        ax.set_xlim(-R_DISC, R_DISC); ax.set_ylim(-R_DISC, R_DISC); ax.set_zlim(-6, 2)
        ax.invert_zaxis()                                    # flip upside down (pit opens upward)
        try:
            ax.set_box_aspect((1, 1, 0.6), zoom=1.4)
        except TypeError:
            ax.set_box_aspect((1, 1, 0.6))
        ax.set_axis_off(); ax.view_init(elev=32, azim=azim)
        ax.text2D(0.02, 0.98, f"3D apical constriction -> invagination\nframe {int(100*t):3d}%"
                  f"\npatch cells (red) sink+constrict\npit depth={-z.min():.1f}",
                  transform=ax.transAxes, color="white", fontsize=6, va="top", family="monospace")

    picks = [0.0, 0.33, 0.66, 1.0]
    sfig = plt.figure(figsize=(4 * 2.5, 2.6)); sfig.patch.set_facecolor("black")
    for k, tt in enumerate(picks):
        sax = sfig.add_subplot(1, 4, k + 1, projection="3d"); sax.set_facecolor("black")
        draw(sax, tt, 40); sax.set_title(f"{int(100*tt)}%", color="white", fontsize=9)
    sfig.subplots_adjust(0.01, 0.01, 0.99, 0.9, wspace=0.02)
    sfig.savefig(os.path.join(outdir, "strip.png"), dpi=110, facecolor="black"); plt.close(sfig)

    fig = plt.figure(figsize=(5.4, 5.4)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("black"); fig.subplots_adjust(0, 0, 1, 1)
    fps = max(1, round(nframes / seconds))
    w = FFMpegWriter(fps=fps, metadata={"title": "sheet_3d"})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for j in range(nframes):
            draw(ax, min(1.0, 1.2 * j / (nframes - 1)), 30 + 60 * j / (nframes - 1))
            w.grab_frame()
    plt.close(fig)
    print(f"done -> {outdir}/movie.mp4 + strip.png  ({len(xy0)} cells)")


if __name__ == "__main__":
    render(OUT)

#!/usr/bin/env python
"""geico_3d -- a mash-up of `embryo_nca` (Growing-NCA lizard) x a smooth 3D lift.

Grows the 🦎 lizard with the strict-Plexus Growing-NCA (embryo_nca's `growing_nca` field operator)
and renders its FORMATION as a smooth 3D body via a simple **2D->3D rule**: INFLATE the silhouette
by giving each in-plane point a half-thickness equal to its distance-to-the-edge (torso fat, legs
and tail thin rounded tubes), build the occupancy volume { |z| <= t(x,y) }, and extract its surface
with marching cubes. This keeps the four legs and the tail distinguishable (a 3D Voronoi blobs
them, plain voxel columns look columnar) while giving a rounded 3D shape. Faces are coloured by the
NCA's own RGB, lightly shaded. One frame per NCA growth step -> the gecko takes shape in 3D.

    python geico_3d.py --device cuda:0
"""
from __future__ import annotations
import os, sys, argparse, tempfile, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "embryo_nca"))     # embryo_nca_ops (growing_nca)

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure
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

import plexus.operators           # noqa: F401
import embryo_nca_ops             # noqa: F401  growing_nca + seed_nca
import plexus.schema as S
from plexus.engine import run as engine_run

OUT = os.path.join(HERE, "archive_geico")
RES = 72


def grow_lizard(frames=240, device="cuda:0"):
    """Grow the NCA lizard via the Plexus engine; return the FULL grid time series [T,16,72,72]."""
    cfg = {
        "general": {"name": "geico", "seed": 0, "n_frames": frames, "dt": 1.0,
                    "boundary": "free", "world": [1.0, 1.0]},
        "sets": {"seed_cell": {"n": 1, "types": {"a": {"fraction": 1.0}}}},
        "fields": {"nca": {"frame": "grid", "res": RES, "components": 16}},
        "operators": [{"op": "seed_nca", "at": "nca", "before_frame": 1},
                      {"op": "growing_nca", "at": "nca", "fire_rate": 0.5}],
        "schedule": ["seed_nca", "growing_nca"],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f); path = f.name
    sim = S.load(path); os.unlink(path)
    _, out = engine_run(sim, device=device)
    return out["fields"]["nca"]["grid"]                        # [T,16,72,72]


_LIGHT = np.array([0.35, 0.25, 1.0]); _LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def lizard_mesh(grid, amp=7.0, pad=2):
    """The 2D->3D rule: inflate the NCA silhouette into a solid { |z| <= t(x,y) } with half-thickness
    t = amp*sqrt(dist_to_edge / max_dist) (elliptical cross-section -> torso fat, legs/tail thin
    rounded tubes), then marching-cubes its surface. Returns (tri [F,3,3], face_rgb [F,3])."""
    rgb = np.clip(np.transpose(grid[:3], (1, 2, 0)), 0, 1)
    alpha = np.clip(grid[3], 0, 1)
    mask = alpha > 0.1
    if mask.sum() < 12:
        return None, None
    H, W = mask.shape
    dist = distance_transform_edt(mask)
    t = amp * np.sqrt(np.clip(dist, 0, None) / max(dist.max(), 1e-6))          # half-thickness [H,W]
    Zc = int(np.ceil(amp)) + pad
    zz = np.arange(-Zc, Zc + 1)
    V = (np.abs(zz)[None, None, :] <= t[:, :, None]).astype(np.float32)        # [H,W,Dz] occupancy
    V = np.pad(V, ((1, 1), (1, 1), (1, 1)))                                    # pad so surface closes
    try:
        verts, faces, _, _ = measure.marching_cubes(V, level=0.5)
    except Exception:
        return None, None
    vi = verts[:, 0] - 1; vj = verts[:, 1] - 1                                 # undo pad -> (row, col)
    X = vj; Y = (H - 1) - vi; Z = (verts[:, 2] - 1) - Zc                       # display coords, y-up
    P = np.stack([X, Y, Z], 1)
    tri = P[faces]
    ci = np.clip(np.round(vi[faces].mean(1)).astype(int), 0, H - 1)           # sample colour at face
    cj = np.clip(np.round(vj[faces].mean(1)).astype(int), 0, W - 1)
    fcol = np.clip(rgb[ci, cj], 0, 1)                                          # premultiplied colour (clean edges)
    return tri, fcol


def _draw_frame(ax, grid, azim, extent, zoom=3.2):
    ax.clear(); ax.set_facecolor("black")
    (cx, cy), rad = extent
    ax.set_xlim(cx - rad, cx + rad); ax.set_ylim(cy - rad, cy + rad); ax.set_zlim(-12, 12)
    try:
        ax.set_box_aspect((1, 1, 24.0 / (2 * rad)), zoom=zoom)     # zoom in; z sized to real thickness
    except TypeError:
        ax.set_box_aspect((1, 1, 24.0 / (2 * rad)))
    ax.set_axis_off(); ax.view_init(elev=42, azim=azim)
    tri, fcol = lizard_mesh(grid)
    if tri is None:
        return
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm = nrm / np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9, None)
    shade = 0.45 + 0.55 * np.clip(np.abs(nrm @ _LIGHT), 0, 1)                  # simple diffuse shading
    rgba = np.concatenate([np.clip(fcol * shade[:, None], 0, 1),
                           np.full((len(fcol), 1), 0.98)], 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, edgecolors="none"))


def render_growth(grids, outdir, seconds=24.0, max_frames=120):
    """Animate the formation: one voxel frame per NCA growth step, all living voxels, colour kept."""
    os.makedirs(outdir, exist_ok=True)
    T = grids.shape[0]
    idx = list(range(0, T, max(1, -(-T // max_frames))))
    tri, _ = lizard_mesh(grids[-1])
    pts = tri.reshape(-1, 3); lo, hi = pts.min(0)[:2], pts.max(0)[:2]
    ctr = (lo + hi) / 2; rad = (hi - lo).max() / 2 * 1.05
    extent = ((ctr[0], ctr[1]), rad)
    fig = plt.figure(figsize=(5.2, 5.2)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("black")
    fig.subplots_adjust(0, 0, 1, 1)
    _draw_frame(ax, grids[idx[-1]], -55, extent)
    fig.savefig(os.path.join(outdir, "geico_3d.png"), dpi=120, facecolor="black")
    fps = max(1, round(len(idx) / seconds))
    w = FFMpegWriter(fps=fps, metadata={"title": "geico_3d_growth"})
    with w.saving(fig, os.path.join(outdir, "movie.mp4"), dpi=110):
        for j, t in enumerate(idx):
            _draw_frame(ax, grids[t], -55 + 40 * j / max(len(idx) - 1, 1), extent)
            w.grab_frame()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    print("growing lizard (capturing the formation process)...", flush=True)
    grids = grow_lizard(device=a.device)
    nfin = (np.clip(grids[-1][3], 0, 1) > 0.1).sum()
    print(f"grid frames: {grids.shape[0]}   final living voxels: {int(nfin)}   rendering...", flush=True)
    render_growth(grids, OUT)
    print(f"done -> {OUT}/movie.mp4 + geico_3d.png", flush=True)


if __name__ == "__main__":
    main()

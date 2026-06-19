"""External, generic plotting

`plexus.plot` is the ONLY place matplotlib lives. It reads a generated trajectory
(graphs_data/<type>/<name>/trajectory.npz) plus the spec's `plotting:` style block
and renders **generically over the sets present in the data** -- it never branches
on an operator or model name. A new simulation gets figures for free as long as it
writes the standard trajectory layout (`<set>__pos`, `<set>__occ`, optional
`<set>__node_type`).

Dispatch = data structure (which sets exist, do they carry a type) + declared
style (colormap, point size). New entity semantics (orientation arrows, fields as
heatmaps) plug in here as additional generic branches, not per-model code.

Figures land next to the data, in the dataset directory.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plexus.schema import Spec
from plexus.paths import graphs_data_path
from plexus.models.registry import get_entity
import plexus.models.entities  # noqa: F401  register entity render hints
from plexus.models.entities import DEFAULT_RENDER


def _sets_in(npz) -> list[str]:
    """Set names present in a trajectory file (keys '<set>__pos')."""
    return sorted({k[:-len("__pos")] for k in npz.files if k.endswith("__pos")})


def _fields_in(npz) -> list[str]:
    """Field names present in a trajectory file (keys '<field>__grid')."""
    return sorted({k[:-len("__grid")] for k in npz.files if k.endswith("__grid")})


def _composite(frame, colors, gamma=0.7):
    """[C, nx, ny] field + [C, 3] channel colours -> [ny, nx, 3] RGB image."""
    C, nx, ny = frame.shape
    rgb = np.zeros((ny, nx, 3), np.float32)
    for c in range(C):
        rgb += frame[c].T[:, :, None] * colors[c][None, None, :]
    return np.clip(rgb, 0, 1) ** gamma                # gamma lifts faint trails


def _render_meta(sname: str) -> dict:
    """Render hints for a set, from the entity registry (how to color/draw it)."""
    try:
        return getattr(get_entity(sname), "RENDER", None) or DEFAULT_RENDER
    except KeyError:
        return DEFAULT_RENDER


def _point_size(style: dict, n: int) -> float:
    if "point_size" in style:
        return float(style["point_size"])
    return 8.0 if n <= 4000 else (4.0 if n <= 12000 else 2.0)


def plot_dataset(sim: Spec, pre_folder: str, movie: bool = False) -> str:
    """Render a generated simulation's trajectory into its dataset directory.
    Returns the directory written to."""
    folder = pre_folder.rstrip("/")
    data_dir = graphs_data_path(folder, sim.name)
    npz_path = os.path.join(data_dir, "trajectory.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"no trajectory at {npz_path} (run `-o generate` first)")
    d = np.load(npz_path)

    style = sim.plotting or {}
    cmap = plt.get_cmap(style.get("colormap", "tab10"))
    bg = style.get("background", "white")             # figure/axes background colour
    W = float(d["world"]) if "world" in d.files else sim.world
    obstacles = list(getattr(sim, "obstacles", []) or [])   # wall geometry to overlay (grey)

    for sname in _sets_in(d):
        pos = d[f"{sname}__pos"]                      # [T, N, 2]
        occ = d[f"{sname}__occ"]                      # [T, N]
        cby = _render_meta(sname).get("color_by")     # which per-node field colors the points
        nt = d[f"{sname}__{cby}"] if (cby and f"{sname}__{cby}" in d.files) else None
        T, N = pos.shape[0], pos.shape[1]
        s = _point_size(style, N)
        # colour leaves by their PARENT cell when contained (distinguishes distinct
        # bodies -- the two drops, the balls -- and reveals mixing); else by node_type.
        par = d[f"{sname}__parent"] if f"{sname}__parent" in d.files else None
        if par is not None:
            color = cmap(par % cmap.N)
        elif nt is not None:
            color = cmap(nt % cmap.N)
        else:
            color = None
        # a container set (parent of a denser child set) is drawn as its MERGED child
        # cloud -- colour the particles by parent cell, then fuse them into one smooth
        # blob per cell -- instead of a bare centroid dot.
        container = _container_child(d, sname)

        def _draw(ax, i):
            if container is not None:
                cpos, par, ncell = container
                ax.imshow(_merged_rgb(cpos[i], par, ncell, W, cmap), extent=[0, W, 0, 1], origin="upper")
                _draw_obstacles(ax, obstacles)
                ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off"); return
            live = occ[i]
            c = (color[live] if color is not None else "#1f77b4")
            ax.set_facecolor(bg)
            ax.scatter(pos[i, live, 0], pos[i, live, 1], s=s, c=c, linewidths=0)
            _draw_obstacles(ax, obstacles)
            ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

        # evolution montage
        idx = [0, T // 5, 2 * T // 5, 3 * T // 5, T - 1]
        fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * W * 3.2, 3.4))
        fig.patch.set_facecolor(bg)
        for ax, i in zip(np.atleast_1d(axes), idx):
            _draw(ax, i)
        plt.tight_layout()
        evo = os.path.join(data_dir, f"fig_{sname}_evolution.png")
        plt.savefig(evo, dpi=100, facecolor=bg); plt.close(fig)

        # final frame
        fig, ax = plt.subplots(figsize=(6 * W, 6))
        fig.patch.set_facecolor(bg)
        _draw(ax, T - 1)
        plt.tight_layout()
        fin = os.path.join(data_dir, f"fig_{sname}_final.png")
        plt.savefig(fin, dpi=110, facecolor=bg); plt.close(fig)
        print(f"[plot] {sname}: {os.path.basename(evo)}, {os.path.basename(fin)}", flush=True)

        if movie:
            if container is not None:
                cpos, par, ncell = container
                _merged_movie(cpos, par, ncell, W, cmap, T, os.path.join(data_dir, f"movie_{sname}"),
                              bg=bg, obstacles=obstacles)
            else:
                _movie(pos, occ, color, s, W, T, os.path.join(data_dir, f"movie_{sname}"),
                       bg=bg, obstacles=obstacles)

    # --- continuum fields: composite the channels and draw the heatmap ------- #
    gamma = float(style.get("gamma", 0.7))
    fbg = style.get("background", "black")
    for fname in _fields_in(d):
        grid = d[f"{fname}__grid"]                     # [T, C, nx, ny]
        colors = d[f"{fname}__colors"]                 # [C, 3]
        T = grid.shape[0]

        def _fdraw(ax, i):
            ax.set_facecolor(fbg)
            ax.imshow(_composite(grid[i], colors, gamma), origin="lower", extent=[0, W, 0, 1],
                      interpolation="bilinear")
            ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

        idx = [0, T // 5, 2 * T // 5, 3 * T // 5, T - 1]
        fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * W * 3.2, 3.4))
        fig.patch.set_facecolor(fbg)
        for ax, i in zip(np.atleast_1d(axes), idx):
            _fdraw(ax, i)
        plt.tight_layout()
        evo = os.path.join(data_dir, f"fig_{fname}_evolution.png")
        plt.savefig(evo, dpi=100, facecolor=fbg); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6 * W, 6)); fig.patch.set_facecolor(fbg)
        _fdraw(ax, T - 1); plt.tight_layout()
        fin = os.path.join(data_dir, f"fig_{fname}_final.png")
        plt.savefig(fin, dpi=110, facecolor=fbg); plt.close(fig)
        print(f"[plot] {fname} (field): {os.path.basename(evo)}, {os.path.basename(fin)}", flush=True)

        if movie:
            _field_movie(grid, colors, W, os.path.join(data_dir, f"movie_{fname}"), gamma, fbg)

    print(f"[plot] figures -> {data_dir}", flush=True)
    return data_dir


def _ffmpeg() -> str | None:
    """Locate ffmpeg: next to the running interpreter (conda env) or on PATH."""
    import sys, shutil
    cand = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    return cand if os.path.exists(cand) else shutil.which("ffmpeg")


def _save_anim(anim, out_base: str, bg: str, fps: int = 25, dpi: int = 150) -> str:
    """Save an animation as high-quality H.264 (constant-quality CRF, not a fixed
    bitrate). Thin bright structure on a dark background -- slime trails, particle
    fields -- otherwise picks up mosquito/blocking noise at a fixed bitrate. Falls
    back to gif when ffmpeg is absent. Returns the output path."""
    from matplotlib.animation import FFMpegWriter, PillowWriter
    ff = _ffmpeg()
    if ff:
        matplotlib.rcParams["animation.ffmpeg_path"] = ff
        out = out_base + ".mp4"
        anim.save(out, dpi=dpi, savefig_kwargs={"facecolor": bg},
                  writer=FFMpegWriter(fps=fps, codec="libx264",
                                      extra_args=["-crf", "16", "-preset", "slow",
                                                  "-pix_fmt", "yuv420p"]))
    else:
        out = out_base + ".gif"
        anim.save(out, writer=PillowWriter(fps=20), savefig_kwargs={"facecolor": bg})
    return out


def _draw_obstacles(ax, obstacles, color="0.45"):
    """Draw the world's wall geometry (grey): [x0,y0,x1,y1] rectangles / [cx,cy,r] discs."""
    from matplotlib.patches import Rectangle, Circle
    for r in (obstacles or []):
        v = [float(x) for x in r]
        if len(v) == 4:
            ax.add_patch(Rectangle((v[0], v[1]), v[2] - v[0], v[3] - v[1], color=color, zorder=3))
        elif len(v) == 3:
            ax.add_patch(Circle((v[0], v[1]), v[2], color=color, zorder=3))


def _container_child(d, sname: str):
    """If `sname` is a container -- some set names it as `parent_name` and has more
    nodes -- return (child_pos[T,Nc,2], parent_idx[Nc], ncell); else None. A container
    set is then drawn as its MERGED child cloud (each parent's particles fused into one
    smooth blob), which is far more informative than its bare centroid dot."""
    for c in _sets_in(d):
        pn = d.get(f"{c}__parent_name")
        if pn is not None and str(pn) == sname and f"{c}__parent" in d.files:
            return d[f"{c}__pos"], d[f"{c}__parent"], int(d[f"{c}__parent"].max()) + 1
    return None


def _blur(a, sigma):
    """Tiny separable Gaussian blur (no scipy dependency)."""
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    a = np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, k, "same"), 1, a)


def _merged_rgb(xy, par, ncell, W, cmap, res: int = 300, sigma: float = 2.2):
    """Splat each parent's child particles to a blurred density mask and paint it the
    parent's colour: a smooth, uniform per-cell blob composited over white. This is the
    special cell<-MPM 'apply colour then merge the particles' container view -- a soft
    blur (sigma ~2.2) + gentle alpha ramp gives the smooth, slightly-blurry cells the
    user preferred (sharper edges read as noisy particle texture)."""
    Hh = res; Wd = max(1, int(round(res * W)))
    rgb = np.ones((Hh, Wd, 3))
    gx = np.clip((xy[:, 0] / max(W, 1e-9) * Wd).astype(int), 0, Wd - 1)
    gy = np.clip((xy[:, 1] * Hh).astype(int), 0, Hh - 1)
    for c in range(ncell):
        m = par == c
        if not m.any():
            continue
        dens = np.zeros((Hh, Wd)); np.add.at(dens, (gy[m], gx[m]), 1.0)
        dens = _blur(dens, sigma)
        a = np.clip(dens / (dens.max() * 0.18 + 1e-9), 0, 1)[..., None]   # soft -> smooth blob
        rgb = rgb * (1 - a) + np.asarray(cmap(c % cmap.N)[:3])[None, None, :] * a
    return rgb[::-1]                                # image row 0 = top = high y


def _merged_movie(cpos, par, ncell, W, cmap, T, out_base, max_frames: int = 120, bg: str = "white",
                  obstacles=None) -> None:
    """Movie of a container set rendered as merged per-cell blobs."""
    from matplotlib.animation import FuncAnimation
    stride = max(1, T // max_frames); frames = list(range(0, T, stride))
    fig, ax = plt.subplots(figsize=(5 * W, 5)); ax.axis("off"); fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=0)
    im = ax.imshow(_merged_rgb(cpos[0], par, ncell, W, cmap), extent=[0, W, 0, 1], origin="upper")
    _draw_obstacles(ax, obstacles)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")

    def upd(i):
        im.set_data(_merged_rgb(cpos[i], par, ncell, W, cmap)); return im,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg); plt.close(fig)
    print(f"[plot] merged-cell movie -> {os.path.basename(out)}", flush=True)


def _movie(pos, occ, color, s, W, T, out_base, max_frames: int = 120, bg: str = "white",
           obstacles=None) -> None:
    """Render a movie of a set's trajectory. Writes mp4 via ffmpeg when available,
    else falls back to gif. `out_base` is the path without extension."""
    from matplotlib.animation import FuncAnimation
    stride = max(1, T // max_frames)
    frames = list(range(0, T, stride))
    fig, ax = plt.subplots(figsize=(5 * W, 5)); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=0)
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=s, linewidths=0,
                    c=(color if color is not None else "#1f77b4"))
    _draw_obstacles(ax, obstacles)

    def upd(i):
        live = occ[i]
        sc.set_offsets(pos[i, live])
        if color is not None:
            sc.set_color(color[live])
        return sc,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg)
    plt.close(fig)
    print(f"[plot] movie -> {os.path.basename(out)}", flush=True)


def _field_movie(grid, colors, W, out_base, gamma, bg, max_frames: int = 150) -> None:
    """Render a field's composited heatmap over time (mp4 via ffmpeg, else gif)."""
    from matplotlib.animation import FuncAnimation
    T = grid.shape[0]
    stride = max(1, T // max_frames)
    frames = list(range(0, T, stride))
    fig, ax = plt.subplots(figsize=(5 * W, 5)); fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0)
    im = ax.imshow(_composite(grid[0], colors, gamma), origin="lower", extent=[0, W, 0, 1],
                   interpolation="bilinear")

    def upd(i):
        im.set_data(_composite(grid[i], colors, gamma))
        return im,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg)
    plt.close(fig)
    print(f"[plot] field movie -> {os.path.basename(out)}", flush=True)

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


def _typed_palette(sim, sname: str, style: dict):
    """Per-type RGB and per-type size factor for a set, from the spec.

    `plotting.colors` (type_name -> rgb) gives an explicit palette -- e.g. the
    ParticleGraph charge code {q_p2: [1,1,1], ...}; missing names fall back to white.
    `plotting.size_by: <prop>` scales point size by |type prop| (e.g. charge), mapped
    to ~0.5x..1.8x. Both are ordered by the set's declared type order so a node_type
    index selects directly. Returns (palette [K,3] | None, size_factor [K] | None)."""
    types = ((getattr(sim, "sets", {}) or {}).get(sname, {}) or {}).get("types", {}) or {}
    names = list(types.keys())
    if not names:
        return None, None
    from matplotlib.colors import to_rgb
    cmap_colors = style.get("colors")
    pal = None
    if cmap_colors:
        pal = np.array([to_rgb(tuple(cmap_colors[nm])) if nm in cmap_colors else (1.0, 1.0, 1.0)
                        for nm in names], np.float32)
    size_by = style.get("size_by")
    sf = None
    if size_by:
        vals = np.array([abs(float(types[nm].get(size_by, 1.0))) for nm in names], np.float32)
        vmax = float(vals.max()) if vals.size else 1.0
        sf = 0.5 + 1.3 * (vals / (vmax + 1e-9))      # |prop| -> 0.5x .. 1.8x point size
    return pal, sf


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
    world_size = d["world_size"] if "world_size" in d.files else None   # per-axis box (3D splat)
    obstacles = list(getattr(sim, "obstacles", []) or [])   # wall geometry to overlay (grey)

    for sname in _sets_in(d):
        pos = d[f"{sname}__pos"]                      # [T, N, 2]
        occ = d[f"{sname}__occ"]                      # [T, N]
        cby = _render_meta(sname).get("color_by")     # which per-node field colors the points
        nt = d[f"{sname}__{cby}"] if (cby and f"{sname}__{cby}" in d.files) else None
        T, N, D = pos.shape[0], pos.shape[1], pos.shape[2]
        # explicit per-type palette (`plotting.colors`, e.g. the ParticleGraph charge
        # code) and per-type size factor (`plotting.size_by: <prop>`, e.g. ~|charge|).
        pal, sf = _typed_palette(sim, sname, style)
        size_scale = (sf[nt % len(sf)] if (sf is not None and nt is not None) else None)
        # --- 3D set: render with the orbiting 3D gaussian splat, then move on --- #
        if D == 3:
            box = (np.asarray(world_size, np.float32) if world_size is not None
                   and len(world_size) == 3 else np.array([W, 1.0, 1.0], np.float32))
            # auto-frame: if the cloud escaped the world box (a `free`/unbounded run),
            # re-centre and size the render to the trajectory's own extent (+margin); a
            # bounded run (extent ~ box) is left exactly as before.
            flat = pos.reshape(-1, D)
            lo, hi = flat.min(0), flat.max(0)
            extent = (hi - lo).astype(np.float32)
            if bool((extent > box * 1.02).any()):
                box = extent * 1.06
                pos = pos - 0.5 * (lo + hi)[None, None, :] + 0.5 * box[None, None, :]
            if pal is not None and nt is not None:
                color3 = pal[nt % len(pal)]
            else:
                color3 = cmap(nt % cmap.N) if nt is not None else None
            _splat3d_outputs(pos, occ, _point_rgb(color3, N), box, data_dir, sname, style,
                             movie, T, size_scale=size_scale)
            continue
        s = _point_size(style, N)
        if size_scale is not None:                    # scale point area by |type prop| (charge)
            s = s * size_scale
        # colour leaves by their PARENT cell when contained (distinguishes distinct
        # bodies -- the two drops, the balls -- and reveals mixing); else by node_type.
        par = d[f"{sname}__parent"] if f"{sname}__parent" in d.files else None
        if pal is not None and nt is not None:
            color = pal[nt % len(pal)]                # explicit per-type colours (charge code)
        elif par is not None:
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
            sz = (s[live] if np.ndim(s) else s)       # per-node size (e.g. ~|charge|) or scalar
            ax.set_facecolor(bg)
            ax.scatter(pos[i, live, 0], pos[i, live, 1], s=sz, c=c, linewidths=0)
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
        grid = d[f"{fname}__grid"]                     # [T, C, nx, ny] (2D) or [T,C,nx,ny,nz] (3D)
        colors = d[f"{fname}__colors"]                 # [C, 3]
        T = grid.shape[0]

        # --- 3D field: volume-splat the active voxels, then move on --- #
        if grid.ndim == 5:
            _field3d_outputs(grid, colors, W, data_dir, fname, style, movie, T)
            continue

        def _fdraw(ax, i):
            ax.set_facecolor(fbg)
            ax.imshow(_composite(grid[i], colors, gamma), origin="lower", extent=[0, W, 0, 1],
                      interpolation="bilinear")
            ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

        # a STATIC field (e.g. an image-derived stiffness / direction map) is constant in
        # time -> one PNG, no evolution strip and no movie.
        static = T <= 1 or float(np.abs(grid - grid[0:1]).max()) < 1e-6

        if not static:
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
        print(f"[plot] {fname} (field): {os.path.basename(fin)}"
              f"{'' if static else ' (+ evolution)'}{' [static]' if static else ''}", flush=True)

        if movie and not static:
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


# =========================================================================== #
#  Gaussian splatting -- render a point cloud as composited soft gaussians.
#
#  A point set is rasterised by accumulating, per pixel, a normalised colour
#  (sum of w*rgb / sum of w) and an opacity that grows with the local density
#  (1 - exp(-density/ref)). Density is a histogram of the points blurred by a
#  separable gaussian (no scipy) -- this is the 2D-projected equivalent of an
#  isotropic 3D gaussian per splat. The 3D variant rotates the cloud into a
#  camera frame first, then fogs + brightens by depth so the volume reads as 3D;
#  the movie ORBITS the camera (the strongest depth cue). Both share the same
#  accumulate/composite core, so 2D and 3D look consistent.
# =========================================================================== #
def _rot_camera(P, azim: float, elev: float):
    """Rotate centred world points [N,3] into the camera frame: yaw by `azim`
    about the vertical (z) axis, then pitch by `elev` about the screen-x axis.
    Returns (screen_xy [N,2], depth [N]) with depth increasing AWAY from camera."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ca, sa = np.cos(azim), np.sin(azim)
    x1 = ca * x - sa * y
    y1 = sa * x + ca * y
    ce, se = np.cos(elev), np.sin(elev)
    y2 = ce * y1 - se * z
    depth = se * y1 + ce * z
    return np.stack([x1, y2], 1), depth


def _splat_accumulate(xy, rgb, w, res_x: int, res_y: int, sigma: float):
    """Splat weighted, coloured points to (acc_colour[res_y,res_x,3], acc_weight).
    `xy` is in PIXEL coordinates. Vectorised scatter (np.add.at) + separable blur
    -> one isotropic gaussian per point, fast enough for 8k points over a movie."""
    gx = np.clip(xy[:, 0].astype(int), 0, res_x - 1)
    gy = np.clip(xy[:, 1].astype(int), 0, res_y - 1)
    acc_w = np.zeros((res_y, res_x), np.float32)
    acc_c = np.zeros((res_y, res_x, 3), np.float32)
    np.add.at(acc_w, (gy, gx), w)
    for ch in range(3):
        np.add.at(acc_c[..., ch], (gy, gx), w * rgb[:, ch])
    acc_w = _blur(acc_w, sigma)
    for ch in range(3):
        acc_c[..., ch] = _blur(acc_c[..., ch], sigma)
    return acc_c, acc_w


def _composite_splat(acc_c, acc_w, bg, ref: float, gamma: float):
    """Normalised colour over `bg`, opacity = (1 - exp(-w/ref))^gamma. `bg` is any
    matplotlib colour. Row 0 of the returned image is top (high y)."""
    from matplotlib.colors import to_rgb
    col = acc_c / np.maximum(acc_w, 1e-8)[..., None]
    a = (1.0 - np.exp(-acc_w / (ref + 1e-9)))[..., None] ** gamma
    img = col * a + np.asarray(to_rgb(bg), np.float32)[None, None, :] * (1 - a)
    return np.clip(img, 0, 1)[::-1]


def gaussian_splat_2d(points, rgb, W: float, res: int = 520, sigma: float = 2.6,
                      bg="white", gamma: float = 0.8, density: float = 0.18):
    """Render 2D points as composited gaussian blobs over the world [0,W]x[0,1].

    points  : [N,2] world coordinates
    rgb     : [N,3] per-point colour (or [3] broadcast)
    returns : [res, round(res*W), 3] RGB image (origin top-left, high-y up).
    `density` sets the opacity reference as a fraction of the peak blurred density."""
    points = np.asarray(points, np.float32)
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (len(points), 3))
    res_y = res
    res_x = max(1, int(round(res * max(W, 1e-9))))
    xy = np.stack([points[:, 0] / max(W, 1e-9) * res_x, points[:, 1] * res_y], 1)
    acc_c, acc_w = _splat_accumulate(xy, rgb, np.ones(len(points), np.float32), res_x, res_y, sigma)
    return _composite_splat(acc_c, acc_w, bg, acc_w.max() * density + 1e-9, gamma)


def gaussian_splat_3d(points, rgb, box, res: int = 520, sigma: float = 2.6,
                      azim: float = 0.7, elev: float = 0.5, bg="black",
                      gamma: float = 0.85, fog: float = 0.55, density: float = 0.16,
                      intensity=None, zoom: float = 1.0):
    """Render 3D points as a depth-shaded gaussian splat from an orbiting camera.

    points : [N,3] world coordinates inside `box` ([bx,by,bz])
    rgb    : [N,3] per-point colour (or [3] broadcast)
    azim/elev : camera yaw/pitch in radians (vary `azim` per frame to orbit)
    fog    : 0..1, how much far points are dimmed (depth cue); `density` as in 2D.
    intensity : optional [N] per-point weight (e.g. a voxel's trail density) that
                scales its splat brightness -- denser points read brighter.
    returns: [res, res, 3] RGB image. Near points splat brighter, far points fade
    into `bg` -- together with the orbit this gives a volumetric 3D read."""
    points = np.asarray(points, np.float32)
    if len(points) == 0:
        from matplotlib.colors import to_rgb
        return np.tile(np.asarray(to_rgb(bg), np.float32), (res, res, 1))
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (len(points), 3)).copy()
    box = np.asarray(box, np.float32)
    P = points - 0.5 * box[None, :]                       # centre the box at the origin
    screen, depth = _rot_camera(P, azim, elev)
    dn = (depth - depth.min()) / (np.ptp(depth) + 1e-9)   # 0 = farthest, 1 = nearest
    span = (float(np.linalg.norm(box)) * 0.5 + 1e-9) / max(zoom, 1e-6)   # half-diag / zoom (>1 = zoom in)
    sx = (screen[:, 0] / (2 * span) + 0.5) * res
    sy = (screen[:, 1] / (2 * span) + 0.5) * res
    w = (0.4 + 0.6 * dn).astype(np.float32)               # nearer -> heavier (brighter) splat
    if intensity is not None:
        w = w * np.asarray(intensity, np.float32)         # denser points splat brighter
    shade = (1.0 - fog) + fog * dn                        # nearer -> full colour, far -> fades to bg
    acc_c, acc_w = _splat_accumulate(np.stack([sx, sy], 1), rgb * shade[:, None], w, res, res, sigma)
    return _composite_splat(acc_c, acc_w, bg, acc_w.max() * density + 1e-9, gamma)


def gaussian_splat_3d_tight(points, rgb, box, res: int = 600, azim: float = 0.7,
                            elev: float = 0.5, bg="black", sprite_sigma: float = 2.0,
                            fog: float = 0.55, peak: float = 0.92, intensity=None,
                            zoom: float = 1.0, size_scale=None):
    """Sharp 3D gaussian splat: stamp a TIGHT per-point gaussian sprite, depth-sorted
    and alpha-over-composited (far drawn first, so near points occlude far) -- unlike
    `gaussian_splat_3d`, which blurs the whole density grid and reads soft/mushy. Each
    point keeps a defined glowing core of radius ~3*`sprite_sigma` px; depth fog dims
    far points toward `bg`. `intensity` (optional [N]) scales per-point brightness.

    points : [N,3] world coords in `box`; rgb : [N,3] (or [3]); returns [res,res,3]."""
    from matplotlib.colors import to_rgb
    bgc = np.asarray(to_rgb(bg), np.float32)
    points = np.asarray(points, np.float32)
    if len(points) == 0:
        return np.tile(bgc, (res, res, 1))
    rgb = np.broadcast_to(np.asarray(rgb, np.float32), (len(points), 3)).copy()
    box = np.asarray(box, np.float32)
    P = points - 0.5 * box[None, :]
    screen, depth = _rot_camera(P, azim, elev)
    dn = (depth - depth.min()) / (np.ptp(depth) + 1e-9)   # 0 = far, 1 = near
    span = (float(np.linalg.norm(box)) * 0.5 + 1e-9) / max(zoom, 1e-6)   # half-diag / zoom (>1 = zoom in)
    sx = (screen[:, 0] / (2 * span) + 0.5) * res
    sy = (screen[:, 1] / (2 * span) + 0.5) * res
    shade = (1.0 - fog) + fog * dn                        # near -> full colour, far -> fades to bg
    if intensity is not None:
        shade = shade * np.clip(np.asarray(intensity, np.float32), 0, None)
    def _make_sprite(sig):
        r = int(max(1, round(3 * sig)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        sp = np.exp(-(xx * xx + yy * yy) / (2.0 * sig ** 2)).astype(np.float32)
        return sp / sp.max(), r
    # per-point sprite size: a single sprite (fast path), or one per bucketed `size_scale`
    # value (e.g. point size ~ |charge|), looked up per point in the depth-sorted loop.
    if size_scale is None:
        base = _make_sprite(sprite_sigma)
        spr = lambda k: base
        skey = np.zeros(len(points), int)
    else:
        ss = np.asarray(size_scale, np.float32)
        keys = np.round(ss, 2)
        cache = {float(u): _make_sprite(sprite_sigma * max(float(u), 0.05)) for u in np.unique(keys)}
        spr = lambda k: cache[float(k)]
        skey = keys
    img = np.tile(bgc, (res, res, 1)).astype(np.float32)
    sxi = np.round(sx).astype(int); syi = np.round(sy).astype(int)
    for idx in np.argsort(-depth):                        # far -> near, src-over (near occludes)
        sprite, rr = spr(skey[idx])
        px, py = int(sxi[idx]), int(syi[idx])
        x0, x1 = max(0, px - rr), min(res, px + rr + 1)
        y0, y1 = max(0, py - rr), min(res, py + rr + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        a = (sprite[y0 - (py - rr):y1 - (py - rr), x0 - (px - rr):x1 - (px - rr)] * peak)[..., None]
        c = (rgb[idx] * shade[idx])[None, None, :]
        img[y0:y1, x0:x1] = a * c + (1 - a) * img[y0:y1, x0:x1]
    return np.clip(img, 0, 1)[::-1]


def _field_voxel_points(frame, colors, R: float, thresh: float = 0.02):
    """A 3D field frame `[C, nx, ny, nz]` -> (points[M,3], rgb[M,3], density[M]) for
    the voxels whose summed density exceeds `thresh`. Voxel (i,j,k) maps to the world
    centre ((i+.5)/R, (j+.5)/R, (k+.5)/R); colour is the density-weighted channel mix."""
    total = frame.sum(0)                                  # [nx,ny,nz]
    m = total > thresh
    if not m.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    idx = np.argwhere(m).astype(np.float32)               # [M,3]
    dens = total[m].astype(np.float32)
    cols = np.zeros((idx.shape[0], 3), np.float32)
    for c in range(frame.shape[0]):
        cols += np.asarray(colors[c], np.float32)[None, :] * frame[c][m][:, None]
    cols /= dens[:, None]
    pts = (idx + 0.5) / R
    return pts, cols, dens


def _point_rgb(color, N: int):
    """Per-point [N,3] RGB from a matplotlib colour array (or None -> default blue)."""
    if color is None:
        return np.tile(np.array([[0.12, 0.47, 0.71]], np.float32), (N, 1))
    return np.asarray(color, np.float32)[:, :3]


def _splat_style(style: dict) -> dict:
    """Read the 3D-splat appearance from a spec's `plotting:` block. Intuitive knobs
    (with back-compat aliases): `splat_size` (blob radius -> gaussian sigma),
    `splat_sharpness` (opacity contrast -> gamma; higher = punchier/sharper),
    `splat_transparency` (opacity reference; higher = more see-through),
    `splat_fog` (how much far points fade into the background), plus `splat_res`."""
    return dict(
        res=int(style.get("splat_res", 520)),
        sigma=float(style.get("splat_size", style.get("splat_sigma", 2.6))),
        gamma=float(style.get("splat_sharpness", style.get("splat_gamma", 0.85))),
        density=float(style.get("splat_transparency", style.get("splat_density", 0.16))),
        fog=float(style.get("splat_fog", 0.55)),
        elev=float(style.get("camera_elev", 0.5)),
        bg=style.get("background", "black"),
    )


def _splat3d_renderer(style, box):
    """Build the per-frame 3D point renderer `render(points, rgb, azim) -> image` from
    the spec's `plotting:` block. `render_3d` selects the method (default `tight` =
    sharp depth-sorted sprites = method B; `splat` = the soft grid-blur). Returns
    (render_fn, bg, azim0, turns, elev)."""
    bg = style.get("background", "black")
    azim0 = float(style.get("camera_azim", 0.7))
    turns = float(style.get("camera_turns", 0.1))         # revolutions over the movie (slow constant drift)
    rock = float(style.get("camera_rock", 0.0))           # gentle back-and-forth wobble (rad); off by default
    zoom_amp = float(style.get("camera_zoom", 0.12))      # slow zoom-in over the clip (fractional, e.g. 0.12 = +12%)
    elev = float(style.get("camera_elev", 0.5))
    mode = style.get("render_3d", "tight")
    if mode == "splat":                                   # soft grid-blur splat (the original)
        kw = _splat_style(style)
        def render(pts, c, az, zoom=1.0, size=None):
            return gaussian_splat_3d(pts, c, box, azim=az, zoom=zoom, **kw)
    else:                                                 # `tight` (default): sharp per-point sprites
        res = int(style.get("splat_res", 600))
        size = float(style.get("splat_size", style.get("splat_sigma", 2.6)))
        sprite_sigma = max(1.4, 0.62 * size)              # blob radius ~ splat_size
        fog = float(style.get("splat_fog", 0.55))
        def render(pts, c, az, zoom=1.0, size=None):
            return gaussian_splat_3d_tight(pts, c, box, res=res, azim=az, elev=elev, bg=bg,
                                           sprite_sigma=sprite_sigma, fog=fog, zoom=zoom, size_scale=size)
    return render, bg, azim0, turns, rock, zoom_amp, elev


def _cam_zoom(zoom_amp, i, n):
    """Zoom factor at movie frame i of n: a slow linear ramp 1 -> 1+`zoom_amp` (a small
    zoom-in over the clip). 1.0 (no ramp) for the still frames."""
    return 1.0 + zoom_amp * (i / max(n - 1, 1))


def _cam_azim(azim0, turns, rock, i, n):
    """Camera azimuth at movie frame i of n. `turns` (full revolutions over the clip)
    is a constant-speed turntable that loops seamlessly at integer values -- the default
    way to read the cloud as 3D. `rock` (radians) instead gives a back-and-forth wobble
    `azim0 + rock·sin(2π t)`. With both 0 the view is fixed."""
    t = i / max(n - 1, 1)
    if turns:
        return azim0 + 2 * np.pi * turns * t
    return azim0 + rock * np.sin(2 * np.pi * t)


def _splat3d_outputs(pos, occ, rgb, box, data_dir, sname, style, movie, T, size_scale=None):
    """Evolution montage + final frame + (optional) movie for a 3D set. The renderer
    (`plotting.render_3d`: `tight` sharp sprites [default] or `splat` soft grid-blur)
    and its appearance come from the `plotting:` block. `size_scale` ([N], optional)
    scales each point's sprite (e.g. ~|charge|). Mirrors the 2D set outputs."""
    render, bg, azim0, turns, rock, zoom_amp, _elev = _splat3d_renderer(style, box)

    def img(i, az, zoom=1.0):
        live = occ[i]
        ss = size_scale[live] if size_scale is not None else None
        return render(pos[i, live], rgb[live], az, zoom, ss)

    idx = [0, T // 5, 2 * T // 5, 3 * T // 5, T - 1]
    fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * 3.2, 3.4))
    fig.patch.set_facecolor(bg)
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(img(i, azim0)); ax.axis("off")
    plt.tight_layout()
    evo = os.path.join(data_dir, f"fig_{sname}_evolution.png")
    plt.savefig(evo, dpi=100, facecolor=bg); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6)); fig.patch.set_facecolor(bg)
    ax.imshow(img(T - 1, azim0)); ax.axis("off"); plt.tight_layout()
    fin = os.path.join(data_dir, f"fig_{sname}_final.png")
    plt.savefig(fin, dpi=110, facecolor=bg); plt.close(fig)
    print(f"[plot] {sname} (3D splat): {os.path.basename(evo)}, {os.path.basename(fin)}", flush=True)

    if movie:
        _splat3d_movie(img, bg, azim0, turns, rock, zoom_amp, T,
                       os.path.join(data_dir, f"movie_{sname}"))


def _splat3d_movie(img, bg, azim0, turns, rock, zoom_amp, T, out_base,
                   max_frames: int = 180, fps: int = 15) -> None:
    """3D-splat movie from a frame renderer `img(i, azim, zoom)`. The camera does a slow
    constant-speed turntable (`turns` revolutions over the clip) or a `rock` wobble, plus
    a slow `zoom_amp` zoom-in. ~`max_frames`/`fps` seconds per clip (a slow rotation)."""
    from matplotlib.animation import FuncAnimation
    stride = max(1, T // max_frames)
    frames = list(range(0, T, stride))
    fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off"); fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=0)

    def render(i):
        return img(i, _cam_azim(azim0, turns, rock, i, T), _cam_zoom(zoom_amp, i, T))

    im = ax.imshow(render(0))

    def upd(i):
        im.set_data(render(i)); return im,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg, fps=fps); plt.close(fig)
    print(f"[plot] 3D splat movie -> {os.path.basename(out)}", flush=True)


def _field3d_outputs(grid, colors, W, data_dir, fname, style, movie, T):
    """Evolution montage + final frame + (optional) movie for a 3D scalar field,
    rendered as a volume gaussian splat of its active voxels (density -> brightness).
    `grid` is [T, C, nx, ny, nz]; appearance follows the `plotting:` splat knobs."""
    kw = _splat_style(style)
    bg = kw["bg"]
    azim0 = float(style.get("camera_azim", 0.7))
    turns = float(style.get("camera_turns", 0.1))         # slow constant drift (matches the set renderer)
    rock = float(style.get("camera_rock", 0.0))
    zoom_amp = float(style.get("camera_zoom", 0.12))
    R = float(grid.shape[-1])                              # pixels per unit (axes 1,2 span [0,1])
    box = np.array([W, 1.0, 1.0], np.float32)
    thresh = float(style.get("field_thresh", 0.02))

    def img(i, az, zoom=1.0):
        pts, cols, dens = _field_voxel_points(grid[i], colors, R, thresh)
        return gaussian_splat_3d(pts, cols, box, azim=az, intensity=dens, zoom=zoom, **kw)

    idx = [0, T // 5, 2 * T // 5, 3 * T // 5, T - 1]
    fig, axes = plt.subplots(1, len(idx), figsize=(len(idx) * 3.2, 3.4))
    fig.patch.set_facecolor(bg)
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(img(i, azim0)); ax.axis("off")
    plt.tight_layout()
    evo = os.path.join(data_dir, f"fig_{fname}_evolution.png")
    plt.savefig(evo, dpi=100, facecolor=bg); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6)); fig.patch.set_facecolor(bg)
    ax.imshow(img(T - 1, azim0)); ax.axis("off"); plt.tight_layout()
    fin = os.path.join(data_dir, f"fig_{fname}_final.png")
    plt.savefig(fin, dpi=110, facecolor=bg); plt.close(fig)
    print(f"[plot] {fname} (3D field splat): {os.path.basename(evo)}, {os.path.basename(fin)}", flush=True)

    if movie:
        from matplotlib.animation import FuncAnimation
        stride = max(1, T // 180)
        frames = list(range(0, T, stride))
        fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off"); fig.patch.set_facecolor(bg)
        fig.tight_layout(pad=0)
        im = ax.imshow(img(0, azim0))

        def upd(i):
            im.set_data(img(i, _cam_azim(azim0, turns, rock, i, T), _cam_zoom(zoom_amp, i, T))); return im,

        anim = FuncAnimation(fig, upd, frames=frames, interval=50)
        out = _save_anim(anim, os.path.join(data_dir, f"movie_{fname}"), bg, fps=15); plt.close(fig)
        print(f"[plot] 3D field splat movie -> {os.path.basename(out)}", flush=True)


def _merged_rgb(xy, par, ncell, W, cmap, res: int = 820, sigma: float = 1.7):
    """Splat each parent's child particles to a blurred density mask and paint it the
    parent's colour: a smooth, uniform per-cell blob composited over white. This is the
    special cell<-MPM 'apply colour then merge the particles' container view. Renders at
    high internal resolution (res~820) with a tight blur (sigma~1.7) and a steep alpha
    ramp so the blob stays a SOLID body with a crisp rim -- the old res=300/sigma=2.2
    upscaled to the movie frame and read as a soft, washed-out blur."""
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
        # steep ramp: full opacity already at 8% of peak density -> solid interior, the
        # tight blur leaves only a thin crisp rim instead of a wide washed-out gradient.
        a = np.clip(dens / (dens.max() * 0.08 + 1e-9), 0, 1)[..., None]
        rgb = rgb * (1 - a) + np.asarray(cmap(c % cmap.N)[:3])[None, None, :] * a
    return rgb[::-1]                                # image row 0 = top = high y


def _merged_movie(cpos, par, ncell, W, cmap, T, out_base, max_frames: int = 120, bg: str = "white",
                  obstacles=None) -> None:
    """Movie of a container set rendered as merged per-cell blobs."""
    from matplotlib.animation import FuncAnimation
    stride = max(1, T // max_frames); frames = list(range(0, T, stride))
    # render large (8in) at high dpi so the high-res blob mask is not softened on upscale
    fig, ax = plt.subplots(figsize=(8 * W, 8)); ax.axis("off"); fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=0)
    im = ax.imshow(_merged_rgb(cpos[0], par, ncell, W, cmap), extent=[0, W, 0, 1],
                   origin="upper", interpolation="bilinear")
    _draw_obstacles(ax, obstacles)
    ax.set_xlim(0, W); ax.set_ylim(0, 1); ax.set_aspect("equal")

    def upd(i):
        im.set_data(_merged_rgb(cpos[i], par, ncell, W, cmap)); return im,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg, dpi=200); plt.close(fig)
    print(f"[plot] merged-cell movie -> {os.path.basename(out)}", flush=True)


def _movie(pos, occ, color, s, W, T, out_base, max_frames: int = 120, bg: str = "white",
           obstacles=None) -> None:
    """Render a movie of a set's trajectory. Writes mp4 via ffmpeg when available,
    else falls back to gif. `out_base` is the path without extension."""
    from matplotlib.animation import FuncAnimation
    stride = max(1, T // max_frames)
    frames = list(range(0, T, stride))
    # render large (8in) at high dpi so the dense dot cloud stays crisp through H.264
    # -- the old 5in/150dpi (750px) softened the small antialiased points into a blur.
    fig, ax = plt.subplots(figsize=(8 * W, 8)); ax.set_xlim(0, W); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(bg); fig.patch.set_facecolor(bg)
    fig.tight_layout(pad=0)
    sm = (s * 1.6 if np.ndim(s) else s * 1.6)              # crisper dots at the larger canvas
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=sm, linewidths=0,
                    c=(color if color is not None else "#1f77b4"))
    _draw_obstacles(ax, obstacles)

    def upd(i):
        live = occ[i]
        sc.set_offsets(pos[i, live])
        if color is not None:
            sc.set_color(color[live])
        return sc,

    anim = FuncAnimation(fig, upd, frames=frames, interval=50)
    out = _save_anim(anim, out_base, bg, dpi=200)
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

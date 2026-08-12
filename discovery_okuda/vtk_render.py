#!/usr/bin/env python
"""The VTK renderer: a z-buffered, GPU, lit picture of the vesicle -- stills, rotations and movies.

Cedric, 12 August: *"ok c2 and c3 are gorgeous, we go VTK. Could you make two kburns and two mp4
evolve with c2 and c3 in b_star."*

WHY VTK REPLACES `_draw`, stated as a defect and not a preference. `mpl_toolkits.mplot3d` has no
depth buffer: it sorts polygons by mean z and paints back to front, which is only exact when no two
polygons overlap in depth order -- and a closed cellular body is the worst case for it. Measured on
b_star's end frame, 6,124 of 12,272 apical faces point away from the camera at azimuth 310 and are
drawn anyway, so which far-side face wins a tie changes with the angle, and one surface is drawn
two different ways at 0:12 and 0:14 of a single rotation. VTK discards a fragment behind another
per pixel, so the question cannot arise. It is also 29x faster on this mesh: 0.32 s a frame against
9.33 (log/okuda/b_star/render_compare.png).

TWO STYLES, both chosen from that sheet, because they answer different questions:

    flat    per-cell colour with the cell outline drawn. The tissue is a MESH and this is the
            picture that says so -- it is the direct successor to every figure the project has
            made, and the one to read when the question is about cells.
    smooth  interpolated shading, no outline. The picture that says the tissue is a SURFACE: an
            arm reads as a round tube with light running down it, which a flat-shaded image cannot
            express at all. The one to read when the question is about shape.

WHAT IS AND IS NOT CARRIED OVER from `_draw`'s colour semantics. The activator LUT (white -> red)
and magenta-for-non-finite are here, because those are measurements. The green just-divided wash,
the blue dying wash and the teal inhibited wash are NOT yet, and a run that carries them will be
drawn without them -- said here rather than discovered later, because a missing overlay in this
project has twice been read as "the mechanism did not fire".

    python vtk_render.py b_star --kburns --style flat
    python vtk_render.py b_star --evolve  --style smooth
    python vtk_render.py b_star --all                       both clips in both styles
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "log", "okuda")
for _p in (HERE, os.path.join(ROOT, "prototype", "Tyssue"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EGL BEFORE VTK IS IMPORTED, or VTK opens an X window, finds the devcontainer's stub display and
# silently falls back to a software rasteriser. The picture is identical and the speed is not.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

CAM = dict(elev=18.0, azim=30.0)
# 896 = 56 x 16, NOT 900. ffmpeg's macro_block_size is 16 and imageio silently resamples anything
# else -- 900 was being stretched to 912, so every frame of a renderer chosen for PRECISION was
# resampled on the way out. The warning said so and is easy to read past.
SIZE = 896
FPS = 25
KB_SECONDS = 18.0          # one revolution; see kburns_render.SECONDS for why the length IS the speed
KB_ZOOM = 0.55
EV_FPS = 12                # the archive holds ~60 recorded frames; 12 fps makes that a 5 s clip


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])


def frames_of(run):
    """Every recorded (pos, mesh, act) of a finished run, from the archive `movie.mp4` uses."""
    p = os.path.join(LOG, run, "traj.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n = sum(1 for k in z.files if k.startswith("pos_"))
    out = []
    for t in range(n):
        mt = z[f"mesh_{t}"]
        mt = mt.item() if hasattr(mt, "item") else mt
        act = z[f"act_{t}"] if f"act_{t}" in z.files else None
        out.append((np.asarray(z[f"pos_{t}"], float), mt, act))
    return out


def box_of(run, fr):
    """The run's own fixed camera box -- shared with every other picture of it."""
    dj = os.path.join(LOG, run, "diag.json")
    if os.path.exists(dj):
        try:
            L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
            if L:
                return float(L)
        except Exception:
            pass
    from run_one import run_box
    return float(run_box([(p, m, a, None) for p, m, a in fr]))


def mesh_of(pos, mt, act, lo=None, hi=None):
    """The apical shell as PolyData with per-cell RGB. Rebuilt per frame: cells divide."""
    import pyvista as pv
    from tyssue_topology_ops3d import rings_from_flat_3d
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    rings = rings_from_flat_3d(es[live], et[live], ef[live], nF)
    faces, idx = [], []
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        faces.append(len(r)); faces.extend(int(v) for v in r); idx.append(f)
    if not idx:
        return None
    m = pv.PolyData(pos, faces=np.asarray(faces, np.int64))
    if act is None:
        rgb = np.full((len(idx), 3), 235, np.uint8)
    else:
        a = np.asarray(act, float)[:nF][idx]
        ok = np.isfinite(a)
        # THE RANGE IS THE RUN'S, NOT THE FRAME'S, on a movie -- otherwise every frame renormalises
        # and a pattern that is strengthening looks static, which is the same defect as the camera
        # autofit that hid growth.
        _lo = float(np.nanmin(a)) if lo is None else lo
        _hi = float(np.nanmax(a)) if hi is None else hi
        x = np.clip((a - _lo) / (_hi - _lo + 1e-9), 0, 1)
        rgb = (np.asarray(_cmap()(x))[:, :3] * 255).astype(np.uint8)
        rgb[~ok] = (255, 26, 217)                 # magenta: not a cell any more
    m.cell_data["rgb"] = rgb
    return m


def add(p, m, style):
    kw = dict(scalars="rgb", rgb=True, lighting=True)
    if style == "flat":
        p.add_mesh(m, show_edges=True, edge_color="black", line_width=0.4,
                   smooth_shading=False, ambient=0.45, diffuse=0.65, specular=0.05, **kw)
    else:
        p.add_mesh(m, show_edges=False, smooth_shading=True,
                   ambient=0.35, diffuse=0.75, specular=0.12, specular_power=18, **kw)


def aim(p, L, azim=None, elev=None):
    e = np.radians(CAM["elev"] if elev is None else elev)
    a = np.radians(CAM["azim"] if azim is None else azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.position = tuple(d * L * 3.2)
    p.camera.focal_point = (0, 0, 0)
    p.camera.up = (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05


def _ease(u):
    return u * u * (3.0 - 2.0 * u)


def _plotter():
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE))
    p.set_background("black")
    p.enable_anti_aliasing("msaa", multi_samples=8)
    return p


def kburns(run, style, out):
    """The finished specimen, turned once and zoomed in. Geometry fixed, camera moving."""
    fr = frames_of(run)
    if not fr:
        return "no traj.npz"
    L0 = box_of(run, fr)
    pos, mt, act = fr[-1]
    m = mesh_of(pos, mt, act)
    n = int(KB_SECONDS * FPS)
    p = _plotter(); add(p, m, style)
    p.add_text(f"{run}  {style}", position="upper_left", font_size=11, color="white")
    p.open_movie(out, framerate=FPS, quality=8)
    for i in range(n):
        u = i / (n - 1)
        aim(p, L0 * (1.0 - (1.0 - KB_ZOOM) * _ease(u)),
            azim=CAM["azim"] + 360.0 * u,                 # a FULL turn, so the clip loops
            elev=CAM["elev"] + 12.0 * np.sin(np.pi * u))
        p.write_frame()
    p.close()
    return f"{n} frames"


def evolve(run, style, out):
    """The run through time, camera nailed down -- the successor to movie.mp4."""
    fr = frames_of(run)
    if not fr:
        return "no traj.npz"
    L = box_of(run, fr)
    # ONE ACTIVATOR RANGE FOR THE WHOLE CLIP, taken over every recorded frame. Per-frame
    # normalisation would make a strengthening pattern look constant.
    vals = [np.asarray(a, float) for _p, _m, a in fr if a is not None]
    lo = float(min(np.nanmin(v) for v in vals)) if vals else 0.0
    hi = float(max(np.nanmax(v) for v in vals)) if vals else 1.0
    p = _plotter()
    p.open_movie(out, framerate=EV_FPS, quality=8)
    for t, (pos, mt, act) in enumerate(fr):
        m = mesh_of(pos, mt, act, lo, hi)
        if m is None:
            continue
        p.clear()                                  # topology changes every frame: cells divide
        add(p, m, style)
        p.add_text(f"{run}  {style}   frame {t + 1}/{len(fr)}   {int(mt['nF'])} cells",
                   position="upper_left", font_size=11, color="white")
        aim(p, L)
        p.write_frame()
    p.close()
    return f"{len(fr)} frames"


def main():
    global SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--style", default="flat", choices=["flat", "smooth"])
    ap.add_argument("--kburns", action="store_true")
    ap.add_argument("--evolve", action="store_true")
    ap.add_argument("--all", action="store_true", help="both clips in both styles")
    ap.add_argument("--size", type=int, default=SIZE)
    a = ap.parse_args()

    SIZE = a.size
    jobs = []
    if a.all:
        for st in ("flat", "smooth"):
            jobs += [("kburns", st), ("evolve", st)]
    else:
        if a.kburns:
            jobs.append(("kburns", a.style))
        if a.evolve:
            jobs.append(("evolve", a.style))
    if not jobs:
        print("nothing asked for -- use --kburns, --evolve or --all"); return 1

    d = os.path.join(LOG, a.run)
    for kind, st in jobs:
        out = os.path.join(d, f"vtk_{kind}_{st}.mp4")
        t0 = time.perf_counter()
        msg = (kburns if kind == "kburns" else evolve)(a.run, st, out)
        dt = time.perf_counter() - t0
        print(f"  {kind:7s} {st:7s} {msg:12s} {dt:7.1f} s -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

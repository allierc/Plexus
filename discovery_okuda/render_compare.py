#!/usr/bin/env python
"""The same end frame through several renderers, side by side, with the time each took.

Cedric, 12 August, after seeing one surface look different at two azimuths of the same clip:
*"I think there is a render issue in matplotlib... it is time to use another library, more precise,
GPU based"*, and then *"first make a montage of b_star/3d.png plot with different libraries, compare
computation time too."*

WHAT IS ACTUALLY WRONG WITH THE CURRENT PICTURE, so the comparison is against a stated defect and
not a taste. `mpl_toolkits.mplot3d` has NO DEPTH BUFFER. It collects every polygon, sorts them by
mean z and paints back to front, which is exact only for polygons that do not interpenetrate in
depth order -- and a closed cellular body is the worst case. Measured on b_star's end frame: 6,124
of 12,272 apical faces point away from the camera at azimuth 310 and are drawn anyway, together
with every lateral wall. Which of those far-side faces wins a tie depends on the angle, so the same
surface is drawn differently at 0:12 and 0:14 of one rotation. That is not anti-aliasing and it is
not fixable by styling.

A z-buffered renderer resolves it by construction: a fragment behind another is discarded, angle by
angle, pixel by pixel. VTK has one; matplotlib does not.

WHAT IS COMPARED. Every backend is given the SAME geometry (the apical shell), the SAME per-cell
colours from the activator LUT, the same camera and the same output size, so the panels differ in
renderer and nothing else. Time is wall-clock for geometry-to-PNG, measured twice with the first
discarded -- the first call pays for context creation and shader compilation, which is a one-off
per process and would misrepresent a 450-frame movie.

    python render_compare.py                 b_star
    python render_compare.py --run r016_01 --size 900
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
for _p in (HERE, os.path.join(ROOT, "discovery_okuda", "ops"), os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EGL BEFORE VTK IS IMPORTED. Without this VTK opens an X window, finds the devcontainer's stub
# display and falls back to a software rasteriser -- which still produces a correct picture, so the
# only symptom is a timing that says nothing about the GPU. Set first, checked below.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

CAM = dict(elev=18.0, azim=30.0)


def end_frame(run):
    z = np.load(os.path.join(LOG, run, "traj.npz"), allow_pickle=True)
    t = sum(1 for k in z.files if k.startswith("pos_")) - 1
    mt = z[f"mesh_{t}"]
    mt = mt.item() if hasattr(mt, "item") else mt
    act = z[f"act_{t}"] if f"act_{t}" in z.files else None
    return np.asarray(z[f"pos_{t}"], float), mt, act


def shell(pos, mt):
    """The apical surface as (vertices, list-of-rings) -- the same polygons `_draw` builds."""
    from mesh_ops import face_polygons_3d
    from topology_ops import rings_from_flat_3d
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    rings = rings_from_flat_3d(es[live], et[live], ef[live], nF)
    keep = [(f, np.asarray(r, int)) for f, r in enumerate(rings) if r is not None and len(r) >= 3]
    return pos, keep


def lut(act, nF):
    """The white->red activator ramp, as uint8 RGB per cell. Identical to the matplotlib LUT."""
    import matplotlib.cm as cm
    if act is None:
        return np.full((nF, 3), 235, np.uint8)
    a = np.asarray(act, float)[:nF]
    ok = np.isfinite(a)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    x = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("wr", ["white", "#d62728"])
    rgb = (np.asarray(cmap(x))[:, :3] * 255).astype(np.uint8)
    rgb[~ok] = (255, 26, 217)                    # magenta: not a cell any more
    return rgb


# ------------------------------------------------------------------ the backends
def draw_matplotlib(pos, mt, act, L, size, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_tyssue_vesicle import _draw
    nF = int(mt["nF"])
    a = np.asarray(act, float)[:nF] if act is not None else None
    if a is not None:
        ok = np.isfinite(a); lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        col = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1); col[~ok] = np.nan
    else:
        col = None
    fig = plt.figure(figsize=(size / 110.0, size / 110.0)); fig.patch.set_facecolor("black")
    ax = fig.add_subplot(111, projection="3d")
    _draw(ax, pos, mt, 3.90, azim=CAM["azim"], act=col, Lbox=L)
    ax.view_init(**CAM)
    fig.savefig(out, dpi=110, facecolor="black", bbox_inches="tight")
    plt.close(fig)


def _pv_mesh(pos, keep, rgb):
    import pyvista as pv
    faces, cols = [], []
    for f, r in keep:
        faces.append(len(r)); faces.extend(r.tolist()); cols.append(rgb[f])
    m = pv.PolyData(pos, faces=np.asarray(faces, np.int64))
    m.cell_data["rgb"] = np.asarray(cols, np.uint8)
    return m


def _pv_camera(p, L):
    """Match matplotlib's elev/azim and box, so the panels are the same view of the same thing."""
    e, a = np.radians(CAM["elev"]), np.radians(CAM["azim"])
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.position = tuple(d * L * 3.2)
    p.camera.focal_point = (0, 0, 0)
    p.camera.up = (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05


def draw_pyvista(pos, mt, act, L, size, out, edges=True, aa=True, smooth=False):
    import pyvista as pv
    pv.OFF_SCREEN = True
    nF = int(mt["nF"])
    _, keep = shell(pos, mt)
    m = _pv_mesh(pos, keep, lut(act, nF))
    p = pv.Plotter(off_screen=True, window_size=(size, size))
    p.set_background("black")
    p.add_mesh(m, scalars="rgb", rgb=True, show_edges=edges, edge_color="black",
               line_width=0.4, smooth_shading=smooth, lighting=True,
               ambient=0.45, diffuse=0.65, specular=0.05)
    if aa:
        p.enable_anti_aliasing("msaa", multi_samples=8)
    _pv_camera(p, L)
    p.screenshot(out)
    p.close()


def draw_vedo(pos, mt, act, L, size, out):
    import vedo
    vedo.settings.default_backend = "vtk"
    nF = int(mt["nF"])
    _, keep = shell(pos, mt)
    m = vedo.Mesh([pos, [r.tolist() for _f, r in keep]])
    m.cellcolors = np.concatenate([lut(act, nF)[[f for f, _ in keep]],
                                   np.full((len(keep), 1), 255, np.uint8)], axis=1)
    m.linewidth(0.4).linecolor("black")
    e, a = np.radians(CAM["elev"]), np.radians(CAM["azim"])
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    pl = vedo.Plotter(offscreen=True, size=(size, size), bg="black")
    pl.show(m, camera=dict(position=d * L * 3.2, focal_point=(0, 0, 0), viewup=(0, 0, 1),
                           parallel_scale=L * 1.05), resetcam=False)
    pl.screenshot(out)
    pl.close()


BACKENDS = [
    ("matplotlib  mplot3d\n(painter's algorithm, no z-buffer)", draw_matplotlib),
    ("VTK / PyVista\nz-buffer, MSAA x8, flat", lambda *a, **k: draw_pyvista(*a, **k)),
    ("VTK / PyVista\nno edges, smooth shading",
     lambda *a, **k: draw_pyvista(*a, edges=False, smooth=True, **k)),
    ("VTK / vedo\nz-buffer, default AA", draw_vedo),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b_star")
    ap.add_argument("--size", type=int, default=770)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pos, mt, act = end_frame(a.run)
    nF = int(mt["nF"])
    L = None
    dj = os.path.join(LOG, a.run, "diag.json")
    if os.path.exists(dj):
        L = (json.load(open(dj)).get("summary") or {}).get("camera_lbox")
    if not L:
        from run_one import run_box
        L = run_box([(pos, mt, act, None)])
    L = float(L)
    print(f"{a.run}: {nF} cells, box {L:.2f}, {a.size}x{a.size} px\n")

    import vtk
    print(f"  VTK {vtk.vtkVersion.GetVTKVersion()}, "
          f"VTK_DEFAULT_OPENGL_WINDOW={os.environ.get('VTK_DEFAULT_OPENGL_WINDOW')}\n")

    tmp = os.path.join(LOG, "analysis", f"_rc_{a.run}")
    os.makedirs(tmp, exist_ok=True)
    made = []
    for i, (label, fn) in enumerate(BACKENDS):
        p = os.path.join(tmp, f"{i}.png")
        # TWICE, FIRST DISCARDED. The first call in a process pays for the GL context and shader
        # compilation; a movie pays that once over hundreds of frames, so reporting it as the
        # per-frame cost would be wrong by an order of magnitude in the GPU backends' disfavour.
        try:
            t0 = time.perf_counter(); fn(pos, mt, act, L, a.size, p); warm = time.perf_counter() - t0
            t0 = time.perf_counter(); fn(pos, mt, act, L, a.size, p); dt = time.perf_counter() - t0
        except Exception as e:
            print(f"  {label.splitlines()[0]:34s} FAILED: {type(e).__name__}: {str(e)[:70]}")
            continue
        print(f"  {label.splitlines()[0]:34s} {dt:6.2f} s   (first call {warm:6.2f} s)")
        made.append((f"{label}\n{dt:.2f} s / frame", p))

    from PIL import Image, ImageDraw, ImageFont
    if not made:
        print("nothing rendered"); return 1
    W = a.size
    LABH = 78
    sheet = Image.new("RGB", (W * len(made), W + LABH), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except Exception:
        f = ImageFont.load_default()
    for i, (label, p) in enumerate(made):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((W, W), Image.LANCZOS)
            sheet.paste(im, (i * W, LABH))
        for j, line in enumerate(label.split("\n")):
            dr.text((i * W + 10, 8 + j * 22), line, fill=(240, 240, 240), font=f)
    out = a.out or os.path.join(LOG, a.run, "render_compare.png")
    sheet.save(out)
    print(f"\n  -> {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

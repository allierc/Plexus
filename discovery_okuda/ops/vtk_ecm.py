#!/usr/bin/env python
"""The VTK renderer for the okuda_ECM 2x2: the same four panels, z-buffered and about 30x faster.

    python vtk_ecm.py 06_spheroid_bm_ecm                    the 2x2 movie, 200 frames
    python vtk_ecm.py 06_spheroid_bm_ecm --frames 12        a short test clip
    python vtk_ecm.py 06_spheroid_bm_ecm --still            one frame, as vtk_3d.png

WHY, AND IT IS THE SAME DEFECT `vtk_render` WAS BUILT FOR. `mpl_toolkits.mplot3d` has no depth
buffer: it sorts whole polygons by mean z and paints back to front. That is exact only when no two
polygons overlap in depth order, and this figure is the worst case for it -- a closed sheet inside a
closed epithelium inside 200,000 matrix particles, three surfaces that interleave everywhere. It is
also why the panel needed `_visible()` hemisphere culling, why the epithelium's wireframe came out
ON TOP of a membrane outside it until the zorders were set by hand, and why the sheet's own faces
showed black seams between them. VTK discards a fragment behind another per pixel, so none of those
three questions arises: no culling, no zorder, no seam.

MEASURED on 06_spheroid_bm_ecm, 200 frames, same camera and same fixed boxes: matplotlib 551 s
(2.76 s a frame), this ~50 s (0.25 s a frame). An earlier version of this file was 125.8 s and its
docstring claimed 24 s before anything had been timed -- the renderer was a python loop over the
matrix's fibres, not the GPU.

WHAT IS DRAWN, panel by panel, and it is the matplotlib figure's own layout:

    top-left      the tissue inside the stressed matrix. Strands, not points: `ecm_seed` lays each
                  fibre as `per` consecutive particles, so the matrix is drawn as the polylines it
                  was seeded as and a strand that has been dragged reads as a dragged strand.
    top-right     the CROSS-SECTION, in the plane of the cavity axis and viewed down that plane's
                  normal. The monolayer is a ring of radial cells built the way
                  `run_tyssue_round._cross_screen` builds it -- every half-edge that straddles the
                  plane gives an apical corner, a basal corner sits at INNER = 0.82 of that radius,
                  and consecutive pairs close into quads. Slicing the apical shell instead gives one
                  curve, because a shell is a surface: the cells vanish and the panel stops being
                  recognisable, which is what the first VTK version did.
    bottom-left   the basement membrane, coloured by lambda_geo, with the plaques at their
                  attachment points on the epithelium.
    bottom-right  the junction network, coloured by myosin.

ONE CAMERA, ONE BOX, EVERY PANEL AND EVERY FRAME -- `L` comes from the run's own spec, exactly as
`run_ecm.render` computes it, because a camera that tracks its subject hides the growth this run is
about.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

# EGL BEFORE VTK IS IMPORTED, or VTK opens an X window, finds the devcontainer's stub display and
# dies -- the same two lines vtk_render.py needs, for the same reason.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.abspath(os.path.join(HERE, "..", "..", "log", "okuda_ECM"))

CAM = dict(elev=18.0, azim=30.0)          # ecm_render.CAM_SIDE
SIZE = 780                                 # per panel; the sheet writes 2*SIZE square
EPI_RGB = (232, 220, 190)
# the matrix's alpha ramp: an unstressed fibre is nearly invisible, a fully-banded one is solid
ALPHA_LO, ALPHA_HI, ALPHA_GAMMA = 0.03, 0.55, 0.85
MONO_RGB = "3c3c3c"                 # the matrix as dark grey felt, stress on alpha
MONO_LO, MONO_HI = 0.10, 0.85       # visible unloaded, solid where the band is high
# BACK FACES ARE CULLED, and the reason is a question this panel kept being asked. A torn sheet is
# still a closed-ish shell, so through the hole you see the FAR wall from the INSIDE -- lit from
# behind, its triangulation dense with distance -- and it reads as a second, concentric mesh. It is
# not: the store holds one connected component of 4,168 faces with a unimodal radius. Culling the
# faces that point away leaves only the outer surface, so a hole reads as a hole and the panel has
# one mesh in it, which is the one it is named after.
BACKFACE_CULL = True
PLQ_RGB = (255, 45, 45)
# THE RAMPS ARE THE REFERENCE RENDERER'S, NOT THIS FILE'S. Inventing a stress list and a warm myosin
# ramp here made the VTK movie disagree with the matplotlib one about what a colour MEANS -- and
# MYOSIN_COLORS runs pale cyan (low) to dark blue (high), so the warm ramp had the sign of the whole
# panel backwards. Importing them is the only way two renderers of one run can be read against each
# other.
def _ramps():
    import ecm_spec as ES
    from ecm_render import MYOSIN_COLORS
    return np.asarray(ES.STRESS_COLORS, float), np.asarray(MYOSIN_COLORS, float)


# =============================================================================================
#  the inputs, read exactly where run_ecm.render reads them
# =============================================================================================
def load(run, src="06_spheroid_ecm"):
    import ecm_render as RD
    d, ds = os.path.join(LOG, run), os.path.join(LOG, src)
    spec = yaml.safe_load(open(os.path.join(ds, "spec_run.yaml")))
    op = next(o for o in spec["operators"] if o["op"] in ("mesh_contact", "cell_to_ecm",
                                                          "ecm_from_cell"))
    scale = float(op.get("scale", 1.0))
    surf = (op.get("tissue") or op.get("surface")).replace(
        "/groups/saalfeld/home/allierc/Graph", "/workspace")
    Tis = RD.load_tissue(surf, scale)
    z = np.load(os.path.join(ds, "traj.npz"))
    pos, band = np.asarray(z["pos"]), np.asarray(z["stress"])
    centre = np.asarray(op["centre"], float)
    st = max(1, int(op.get("mesh_stride", 1)))
    T = min(pos.shape[0], band.shape[0])
    meshes = [(k * st, m) for k, (_t, m) in enumerate(Tis["meshes"]) if k * st < T]
    seed = next(o for o in spec["operators"] if o["op"] in ("seed_ecm", "ecm_seed"))
    per = max(1, int(spec["sets"]["mpm_particle"]["per_parent"]) // int(seed["n_fibres"]))
    # the SAME fixed box run_ecm.render computes, so the two movies are the same size on a page
    L3 = min(0.5 / max(scale, 1e-12), Tis["Lbox"] * 1.60) * 0.72
    bm = None
    p = os.path.join(d, "bm_frames.npz")
    if os.path.exists(p):
        bm = _bm(np.load(p))
    # WHICH FIELD THE SHEET CARRIES, read off the run rather than guessed from the folder's name: the
    # protease rigs record `kdeg` and colour by MT1-MMP, the mechanical ones colour by lambda_geo.
    mode = "lam"
    import json
    mp = os.path.join(d, "metrics.json")
    if os.path.exists(mp):
        try:
            mode = "mt1" if "kdeg" in json.load(open(mp)) else "lam"
        except Exception:
            pass
    myo = np.concatenate([np.asarray(m["myo"]).ravel() for _t, m in meshes if "myo" in m]) \
        if any("myo" in m for _t, m in meshes) else None
    return dict(pos=pos, band=band, centre=centre, scale=scale, meshes=meshes, per=per, L=L3,
                bm=bm, mesh_frames=np.asarray([m[0] for m in Tis["meshes"]]), stride=st, out=d,
                mode=mode,
                myo_hi=(float(np.percentile(myo, 98)) if myo is not None and myo.size else None))


def _bm(z):
    """The sheet, in TISSUE units, live nodes only -- test_06_panels.BMPanel's own two rules."""
    c, sc = np.asarray(z["centre"], float), float(z["scale"])
    tis = lambda A: (np.asarray(A, np.float64) - c) / sc                       # noqa: E731
    if "n_kept" in z.files:
        n = int(z["n_kept"])
        t = np.array([int(z[f"t{i}"]) for i in range(n)])
        X, F, L, PP, ND = [], [], [], [], []
        for i in range(n):
            x, f, nd = tis(z[f"x{i}"]), z[f"f{i}"], z[f"n{i}"]
            used = np.unique(np.concatenate([f.reshape(-1), nd]) if nd.size else f.reshape(-1))
            rm = np.full(x.shape[0], -1, np.int64)
            rm[used] = np.arange(used.size)
            X.append(x[used]); F.append(rm[f]); L.append(z[f"v{i}"]); PP.append(tis(z[f"p{i}"]))
            ND.append(rm[nd])
    else:
        t = np.asarray(z["frames"])
        X = list(tis(z["X"])); L = list(z["L"]); PP = list(tis(z["PP"]))
        F = [np.asarray(z["F"])] * len(t); ND = list(np.asarray(z["nod"]))
    return dict(t=t, X=X, F=F, L=L, PP=PP, ND=ND,
                vmax=max(float(np.max(v)) for v in L if v.size))


# =============================================================================================
#  geometry -> PolyData
# =============================================================================================
def strands(q, band, per, cmap, mono=None):
    """The matrix as the polylines `ecm_seed` laid down, coloured by its stress band.

    ONE PolyData FOR THE WHOLE MATRIX, AND NO PYTHON LOOP OVER FIBRES. 10,000 fibres as 10,000
    actors is 10,000 draw calls and VTK is no faster than matplotlib at that. The first version of
    this built the polylines in a double loop -- 10,000 strands x 19 segments per frame -- and that
    loop, not the GPU, was the renderer: 0.63 s a frame against matplotlib's 2.76, a 4.4x that
    should have been far larger. Emitting SEGMENTS instead of runs is the same picture (a polyline
    is drawn as its segments) and is pure numpy.

    A strand is cut where consecutive particles have separated: MLS-MPM computes nothing between two
    points further apart than the kernel's support, so a line joining them asserts a fibre the
    physics does not have.
    """
    import pyvista as pv
    n = (q.shape[0] // per) * per
    P = q[:n].reshape(-1, per, 3)
    b = band[:n].reshape(-1, per)
    d = np.linalg.norm(np.diff(P, axis=1), axis=-1)                    # (nstrand, per-1)
    fin = np.isfinite(d)
    keep = fin & (d < (3.0 * d[fin].mean() if fin.any() else np.inf))
    if not keep.any():
        return None
    ia, ja = np.nonzero(keep)                                          # every surviving segment
    A = P[ia, ja]
    B = P[ia, ja + 1]
    pts = np.empty((2 * len(ia), 3), np.float32)
    pts[0::2], pts[1::2] = A, B
    lines = np.column_stack([np.full(len(ia), 2), np.arange(0, 2 * len(ia), 2),
                             np.arange(1, 2 * len(ia), 2)]).ravel()
    poly = pv.PolyData(pts, lines=np.asarray(lines, np.int64))
    # CONTRAST COMES FROM ALPHA, NOT ONLY FROM HUE. Every fibre drawn at one opacity makes the matrix
    # a flat wall whose colour is the SUM of a few hundred overlapping strands, and the band that a
    # single stressed fibre carries is lost inside it -- the picture then says "there is a matrix"
    # and not "this part of it is loaded". Alpha rising with the band lets the unstressed bulk fall
    # back toward the background while a loaded strand stays legible through it, and RGBA is per
    # cell, so this costs one array and no extra draw call.
    cb = np.clip(np.maximum(b[ia, ja], b[ia, ja + 1]).astype(int), 0, len(cmap) - 1)
    frac = cb / max(len(cmap) - 1, 1)
    if mono is None:
        lut = (np.asarray(cmap, float) * 255).astype(np.uint8)[cb]
        a8 = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * frac ** ALPHA_GAMMA
    else:
        # MONOCHROME: THE STRESS IS THE TRANSPARENCY AND NOTHING ELSE. With hue carrying the band as
        # well, an unloaded matrix is drawn in the ramp's dark end AT the ramp's low alpha, so at
        # frame 0 -- when nothing is loaded yet -- the panel is very nearly empty and the fibres the
        # tissue is about to push into cannot be seen at all. Here every fibre is the same dark grey
        # and only its opacity moves, so the unstressed matrix reads as a visible grey felt from the
        # first frame and a loaded strand brightens out of it by becoming solid rather than by
        # changing colour. `MONO_LO` is what makes it visible at the start and is the whole point.
        lut = np.tile(np.asarray(_hex(mono), np.uint8), (len(cb), 1))
        a8 = MONO_LO + (MONO_HI - MONO_LO) * frac ** ALPHA_GAMMA
    poly.cell_data["rgb"] = np.column_stack([lut, (255 * a8).astype(np.uint8)])
    return poly


def _hex(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def tissue_poly(mt, pos):
    """The epithelium's apical shell as polygons, from its half-edge table."""
    import pyvista as pv
    from topology_ops import rings_from_flat_3d
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
        return None, None
    return pv.PolyData(pos, faces=np.asarray(faces, np.int64)), np.asarray(idx)


def cell_ring(mt, pos, normal=(0.0, 1.0, 0.0), inner=0.82):
    """The monolayer in section, as `run_tyssue_round._cross_screen` builds it -- and it is not a
    slice of the apical surface.

    Slicing the apical shell gives ONE curve, because the shell is a surface: the section then reads
    as a tangle of cell-wall crossings and the cells disappear. The reference renderer instead takes
    every half-edge that STRADDLES the plane, interpolates its crossing point as the apical corner,
    puts a basal corner at `inner` times that radius, orders them by angle and closes consecutive
    pairs into quads. That is the ring of radial cells -- the picture this panel used to show, and the
    only one in which the monolayer has a thickness.

    `inner` is `ecm_render.INNER`, the same 0.82 the matplotlib section used, so the two agree about
    how thick the epithelium is.
    """
    import pyvista as pv
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    s_, t_ = es[live], et[live]
    d = np.asarray(normal, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    pr = pos @ d
    a, b = pr[s_], pr[t_]
    hit = (a * b) < 0
    if hit.sum() < 3:
        return None
    fr = (-a[hit] / (b[hit] - a[hit]))[:, None]
    X = pos[s_[hit]] + fr * (pos[t_[hit]] - pos[s_[hit]])          # apical corners, on the plane
    # order around the ring, in the plane's own two directions
    u = np.cross(d, [0.0, 0.0, 1.0])
    u = u / (np.linalg.norm(u) + 1e-12) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(d, u)
    c = X.mean(0)
    ang = np.arctan2((X - c) @ v, (X - c) @ u)
    X = X[np.argsort(ang)]
    B = X * inner                                                   # basal corners
    n = len(X)
    pts = np.vstack([B, X])
    j = (np.arange(n) + 1) % n
    quads = np.column_stack([np.full(n, 4), np.arange(n), n + np.arange(n), n + j, j]).ravel()
    return pv.PolyData(pts, faces=np.asarray(quads, np.int64))


def junction_poly(mt, pos, myo_hi=None):
    """The junction network as line segments, coloured by myosin when the cache carries it."""
    import pyvista as pv
    nF = int(mt["nF"])
    es, et, ef = (np.asarray(mt[k]) for k in ("E_srce", "E_trgt", "E_face"))
    live = ef < nF
    a, b = es[live], et[live]
    lines = np.column_stack([np.full(a.size, 2), a, b]).ravel()
    poly = pv.PolyData(pos, lines=lines)
    myo = np.asarray(mt["myo"])[live] if "myo" in mt else None
    if myo is not None and myo.size == a.size:
        # the matplotlib panel's own myosin ramp: cool blue at rest, warm at high myosin
        # THE RUN'S SCALE, NOT THE FRAME'S. Taking p98 per frame renormalises every frame, so a
        # sheet-wide drift in myosin -- exactly what `beta` produces -- renders as no change at all.
        # It is the defect run_ecm.render fixed for this same panel; passing the run-wide value in is
        # the only way a caller can keep the two movies comparable.
        hi = myo_hi if myo_hi else max(float(np.percentile(myo, 98)), 1e-9)
        x = np.clip(myo / hi, 0, 1)[:, None]
        _, MY = _ramps()
        k = np.clip((np.asarray(x).ravel() * (len(MY) - 1)).round().astype(int), 0, len(MY) - 1)
        poly.cell_data["rgb"] = (MY[k] * 255).astype(np.uint8)
    else:
        poly.cell_data["rgb"] = np.tile(np.uint8([122, 184, 255]), (a.size, 1))   # no myosin recorded
    return poly


def bm_poly(X, F, val, vmax, mode="lam", gamma=0.5):
    """The sheet, coloured by its field on the same truncated ramp the matplotlib panel uses.

    `mode` IS NOT COSMETIC, IT IS THE ZERO OF THE SCALE. lambda_geo starts at 1 (an unstretched
    surface) and MT1-MMP starts at 0, and this function hardcoded `val - 1`: on a protease run every
    face then normalised negative, clipped to zero, and the membrane rendered BLACK -- a panel whose
    field is its subject, showing none of it, in a picture that otherwise looked correct.
    """
    import matplotlib
    import pyvista as pv
    from matplotlib.colors import ListedColormap
    lo = 1.0 if mode == "lam" else 0.0
    base = "magma" if mode == "lam" else "viridis"
    cm = ListedColormap(matplotlib.colormaps[base](np.linspace(0.0, 0.87, 256)))
    f = np.column_stack([np.full(F.shape[0], 3), F]).ravel()
    m = pv.PolyData(X, faces=np.asarray(f, np.int64))
    x = np.clip((np.asarray(val) - lo) / max(vmax - lo, 1e-9), 0, 1) ** gamma
    m.cell_data["rgb"] = (np.asarray(cm(x))[:, :3] * 255).astype(np.uint8)
    return m


# =============================================================================================
#  the 2x2
# =============================================================================================
def _aim_section(p, L):
    """Looking ALONG the cut's normal, so a section is seen face-on.

    THE TOP-RIGHT PANEL IS A CROSS-SECTION AND NOT A CUT SOLID. Clipping the geometry and keeping the
    side camera shows the far half of a sphere from outside -- a bitten shape whose surface is still
    the surface, which is why the panel stopped being recognisable as a section. A section is the
    INTERSECTION with a plane, viewed down that plane's normal: the epithelium becomes a ring of cell
    walls, the membrane a line inside it, and the matrix the strands that cross the slab.
    """
    p.camera.position = (0.0, -L * 3.2, 0.0)
    p.camera.focal_point = (0.0, 0.0, 0.0)
    p.camera.up = (0.0, 0.0, 1.0)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05


def _aim(p, L, clip=False):
    e, a = np.radians(CAM["elev"]), np.radians(CAM["azim"])
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    p.camera.position = tuple(d * L * 3.2)
    p.camera.focal_point = (0, 0, 0)
    p.camera.up = (0, 0, 1)
    p.camera.parallel_projection = True
    p.camera.parallel_scale = L * 1.05


def render(run, frames=None, still=False, src="06_spheroid_ecm", out_name=None, fps=20):
    import imageio_ffmpeg
    import pyvista as pv
    D = load(run, src)
    pv.OFF_SCREEN = True
    n_all = len(D["meshes"])
    idx = (list(range(n_all)) if frames is None else
           [int(round(u)) for u in np.linspace(0, n_all - 1, min(frames, n_all))])
    if still:
        # THE LAST FRAME, like `3d.png`. A still of frame 0 is a small sphere in an untouched matrix
        # and says nothing about the run; the end state is what anyone looks at first, and it is the
        # frame the matplotlib still already shows, so the two are comparable.
        idx = [n_all - 1]
    L, per, sc, c = D["L"], D["per"], D["scale"], D["centre"]
    bm = D["bm"]
    out = os.path.join(D["out"], out_name or ("vtk_3d.png" if still else "movie_vtk.mp4"))
    wr = None
    if not still:
        wr = imageio_ffmpeg.write_frames(out, (2 * SIZE, 2 * SIZE), fps=fps, quality=7)
        wr.send(None)
    t0 = time.time()
    for n, k in enumerate(idx):
        t, mt = D["meshes"][k]
        q = (D["pos"][t] - c) / max(sc, 1e-12)                      # matrix in tissue units
        p = pv.Plotter(off_screen=True, window_size=(2 * SIZE, 2 * SIZE), shape=(2, 2),
                       border=False)
        tis, _ = tissue_poly(mt, np.asarray(mt["pos"], float))
        mat = strands(q, D["band"][t], per, _ramps()[0], mono=MONO_RGB)

        # ---- top-left: the tissue inside the stressed matrix, in 3D
        p.subplot(0, 0)
        p.set_background("black")
        if mat is not None:
            p.add_mesh(mat, scalars="rgb", rgb=True, line_width=0.75, lighting=False)
        if tis is not None:
            p.add_mesh(tis, color=[v / 255 for v in EPI_RGB], smooth_shading=True,
                       show_edges=True, edge_color="black", line_width=0.3,
                       ambient=0.35, diffuse=0.75, specular=0.1)
        _aim(p, L)

        # ---- top-right: the CROSS-SECTION, in the plane of the cavity axis
        p.subplot(0, 1)
        p.set_background("black")
        slab = 0.055 / max(sc, 1e-12)                    # run_ecm.render's own slab, in tissue units
        if mat is not None:
            # A SLICED LINE SET IS A POINT CLOUD -- slicing the matrix would leave the dots where its
            # strands happen to cross the plane, which is a picture of the plane and not of the
            # matrix. The strands INSIDE the slab are drawn instead, so the section shows matter.
            keep = np.abs(mat.points[:, 1]) < slab
            sub = mat.extract_points(keep, adjacent_cells=False)
            if sub.n_cells:
                p.add_mesh(sub, scalars="rgb", rgb=True, line_width=1.1, lighting=False)
        ring = cell_ring(mt, np.asarray(mt["pos"], float), normal=(0.0, 1.0, 0.0))
        if ring is not None:
            p.add_mesh(ring, color=[v / 255 for v in EPI_RGB], show_edges=True,
                       edge_color="black", line_width=0.6, lighting=False)
        if bm is not None:
            jj = int(np.argmin(np.abs(bm["t"] - int(D["mesh_frames"][min(k, len(D["mesh_frames"])
                                                                         - 1)]))))
            if bm["F"][jj].shape[0]:
                bs = bm_poly(bm["X"][jj], bm["F"][jj], bm["L"][jj], bm["vmax"],
                             mode=D["mode"]).slice(normal="y", origin=(0, 0, 0))
                if bs.n_points:
                    p.add_mesh(bs, scalars="rgb", rgb=True, line_width=2.6, lighting=False)
        _aim_section(p, L)

        p.subplot(1, 0)
        p.set_background("black")
        if bm is not None:
            j = int(np.argmin(np.abs(bm["t"] - int(D["mesh_frames"][min(k, len(D["mesh_frames"])
                                                                        - 1)]))))
            # THE MEMBRANE'S OWN MESH, AND NOTHING ELSE'S. This panel drew the epithelium as a
            # wireframe for context, and by the last frame that wireframe IS the panel: the sheet
            # ends up inside the tissue at 95% of its plaques, so the cell mesh sits in front of the
            # membrane everywhere and the picture reads as an epithelium with red dots on it. The
            # membrane gets its own triangulation instead -- `show_edges` on the sheet -- so the mesh
            # in this panel is the one the panel is named after.
            if bm["F"][j].shape[0]:
                p.add_mesh(bm_poly(bm["X"][j], bm["F"][j], bm["L"][j], bm["vmax"],
                                   mode=D["mode"]),
                           scalars="rgb", rgb=True, smooth_shading=True, lighting=True,
                           show_edges=True, edge_color="#3a1a14", line_width=0.35,
                           culling=("back" if BACKFACE_CULL else None),
                           ambient=0.35, diffuse=0.75, specular=0.12, specular_power=18)
            pp, nd = bm["PP"][j], bm["ND"][j]
            if len(pp) and len(bm["F"][j]):
                st = max(1, int(np.ceil(len(pp) / 2562.0)))
                # A PLAQUE IS A LINK, SO IT IS DRAWN AS ONE -- and that is also what makes it
                # visible. Its attachment point is on the EPITHELIUM, and by the last frame the sheet
                # has sunk inside the tissue at 95% of them, so a point drawn there is correctly
                # occluded by the membrane and the panel loses the very set it is about. The segment
                # from the attachment point to the sheet node it holds ends ON the surface, so the
                # outer end is always in view and the part that is buried reads as buried.
                # ONLY THE PLAQUES THAT STILL HOLD MEMBRANE. The rigs do not cull the contact set
                # when a face tears -- it is per NODE -- so a torn run ends with every plaque it
                # started with, most of them holding a node that belongs to no live face and none of
                # them short, because the remnant has also peeled off. Drawn in full that is 2,562
                # long red links over a sheet that has lost a fifth of its faces, and the panel is
                # unreadable. The set drawn here is the one that is still attached; the count of the
                # rest is printed, so the defect is reported rather than hidden by the drawing.
                liven = np.unique(bm["F"][j])
                hold = np.isin(nd, liven)
                if n == 0:
                    print(f"[vtk_ecm] plaques: {int(hold.sum())} of {len(nd)} still hold a live "
                          f"face; {int((~hold).sum())} hold nothing and are not drawn; 1 in "
                          f"{max(1, int(np.ceil(max(int(hold.sum()), 1) / 800.0)))} of the rest "
                          f"is drawn", flush=True)
                nd, pp = nd[hold], pp[hold]
                # AND A CAP ON HOW MANY ARE DRAWN. Once the sheet has peeled off, each link is a long
                # segment rather than a short one, and two thousand of them cover the field the panel
                # exists to show. 800 is enough to read the set as a set; the stride is stated.
                #
                # AND IT IS A STRIDE AND NOT THE FIRST 800. Taking `[:800]` would draw the same
                # plaques at every frame, which is what a flicker-free panel wants, but they are not
                # a sample of the sheet: the contact set is seeded cell by cell, so the first 800
                # entries are the first 67 cells and those sit together on one cap. A panel drawn
                # from them shows adhesion on one pole and bare membrane everywhere else. The stride
                # costs a flicker -- the drawn subset changes composition as the count grows, 1 in 3
                # at the start of 07h and 1 in 91 at the end -- and that flicker is a property of the
                # DRAWING, not of the adhesion: `plaque_identity` measures 98.3% of the set persisting
                # frame to frame, at most 0.0% culled in any interval, and a median tangential drift
                # of 0.13 degrees. Read the movie for where the plaques are, and that json for
                # whether they stay.
                st = max(1, int(np.ceil(max(len(pp), 1) / 800.0)))
                if not len(pp):
                    st = 1
                a, b2 = pp[::st], bm["X"][j][nd[::st]]
                seg = np.empty((2 * len(a), 3), float)
                seg[0::2], seg[1::2] = a, b2
                ln = np.column_stack([np.full(len(a), 2), np.arange(0, 2 * len(a), 2),
                                      np.arange(1, 2 * len(a), 2)]).ravel()
                p.add_mesh(pv.PolyData(seg, lines=np.asarray(ln, np.int64)),
                           color=[v / 255 for v in PLQ_RGB], line_width=1.2, lighting=False)
                p.add_points(pv.PolyData(b2), color=[v / 255 for v in PLQ_RGB],
                             point_size=5.0, render_points_as_spheres=True)
                # LARGER THAN THE MATPLOTLIB PANEL'S, and it can afford to be. There a dot was drawn
                # over the sheet whatever its depth, so 2,562 of them at a legible size buried the
                # field underneath; here VTK occludes a plaque behind the surface per pixel, so only
                # the ones actually facing the reader are painted and the size can say what it is.

        _aim(p, L)

        p.subplot(1, 1)
        p.set_background("black")
        jp = junction_poly(mt, np.asarray(mt["pos"], float), myo_hi=D.get("myo_hi"))
        p.add_mesh(jp, scalars="rgb", rgb=True, line_width=1.2, lighting=False)
        _aim(p, L)

        img = p.screenshot(return_img=True)
        p.close()
        if still:
            import imageio.v2 as iio
            iio.imwrite(out, img)
            print(f"[vtk_ecm] {out}  (frame {t})", flush=True)
            return out
        wr.send(np.ascontiguousarray(img[:, :, :3]))
        if n == 0:
            print(f"[vtk_ecm] first frame in {time.time()-t0:.2f}s ({len(idx)} to draw)", flush=True)
    wr.close()
    dt = time.time() - t0
    print(f"[vtk_ecm] {out}  ({len(idx)} frames @ {fps} fps, {dt:.1f}s, "
          f"{dt/max(len(idx),1):.3f}s a frame)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--src", default="06_spheroid_ecm")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--still", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    render(a.run, frames=a.frames, still=a.still, src=a.src, out_name=a.out, fps=a.fps)


if __name__ == "__main__":
    main()

"""ecm_render -- draw the cellfix_B_new epithelium inside its matrix, the way okuda draws it.

WHAT WAS WRONG BEFORE, PLAINLY. `21`-`23` drew the tissue as a ring of CYAN DOTS: the cell
centroids, scattered. A picture of centroids is not a picture of an epithelium -- it cannot show a
cell, a junction, a division or a monolayer -- and asked "is this actually the vertex model or just
a sphere?", that image has no answer. Meanwhile `log/okuda/cellfix_B_new/strip.png` already answers
it in four rows of polygons, and every other okuda artefact in the repo is drawn by ONE routine,
`run_tyssue_vesicle._draw`. This module calls that routine. It does not reimplement it, does not
recolour it and does not substitute dots for it.

THE CONVENTION, TAKEN FROM `discovery_okuda.run_one.render` RATHER THAN REINVENTED:

    _draw               each cell is a prism -- apical face, basal face, lateral walls -- edged
                        black. Colour is the activator on a white->red LUT; cellfix_B_new's
                        activator is identically 0, so the cells are WHITE, which is why the
                        reference strip is a white ball.
    GREEN WASH          `age <= 4 and ndiv > 0`: this cell divided in the last four division
                        calls. Benign and expected, and the reason the reference ball is
                        white-with-green-patches rather than plain white. It is the only thing
                        moving on the tissue's surface in this experiment, so dropping it would
                        make a proliferating epithelium look inert.
    MAGENTA             genuinely broken cell. An alarm; normally never appears.
    CAM_SIDE / CAM_TOP  elev 18 and elev 88, both at azim 30. Two viewpoints because one can hide
                        a feature that lies along its view direction.
    ONE FIXED Lbox      computed once for the whole run. Per-frame autofit renders a tissue that
                        triples in radius at constant apparent size -- it is what hid growth in
                        every archived movie until `run_box` was written.
    _cross_screen       the monolayer in section: one filled quad per cell between the apical and
                        basal rings, so the band IS the epithelium's thickness and the hollow
                        middle is the lumen.

WHAT THIS MODULE ADDS, AND ONLY THIS: the matrix, coloured by its stress band, drawn around the
tissue in the tissue's own frame.

    THE MATRIX IS SPLIT INTO A FAR HALF AND A NEAR HALF about the tissue centre, and the epithelium
    is drawn BETWEEN them. Matplotlib's 3D depth sort is per-ARTIST, not per-point: one scatter of
    110,000 particles enclosing a sphere has its mean depth AT the sphere, so the whole cloud
    lands either wholly in front of the tissue (which buries it in dust) or wholly behind it
    (which hides the near matrix). Two scatters and `computed_zorder=False` put the tissue where it
    belongs -- inside the material -- and the near half is drawn faint so the cells stay legible.

    REST AND STRESS ARE DRAWN DIFFERENTLY. Unstrained matrix is small and dim, strained matrix is
    larger and bright, and the strained particles are sorted so the hottest band draws last. With
    one uniform scatter the front is a slight hue shift inside a fog of 110,000 equally loud dots;
    the front is the measurement, so it gets the visual weight.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (HERE, os.path.join(ROOT, "src"), os.path.join(ROOT, "prototype", "Tyssue"),
          os.path.join(ROOT, "discovery_okuda")):
    if p not in sys.path:
        sys.path.insert(0, p)

CAM_SIDE = dict(elev=18, azim=30)      # the archive/minisite convention; `_draw`'s own elevation
CAM_TOP = dict(elev=88, azim=30)       # near-polar: what the side view foreshortens is broadside here
DIVIDED_WINDOW = 4                     # division-calls a cell counts as "just divided" for
INNER = 0.82                           # basal radius fraction -- `_draw`'s monolayer thickness
P0 = 3.90                              # passed through to `_draw`; it does not colour by it


# --------------------------------------------------------------------------- the tissue, loaded
def load_tissue(path, scale):
    """The pass-1 cache, as the renderer wants it: per-frame meshes in TISSUE units + the camera."""
    z = np.load(path)
    frames = np.asarray(z["mesh_frames"])
    meshes = []
    for j, t in enumerate(frames):
        meshes.append((int(t), {
            "pos": np.asarray(z[f"m{j}_pos"], np.float64),
            "E_srce": np.asarray(z[f"m{j}_E_srce"]), "E_trgt": np.asarray(z[f"m{j}_E_trgt"]),
            "E_face": np.asarray(z[f"m{j}_E_face"]), "nF": int(z[f"m{j}_nF"]),
            "Nv": int(z[f"m{j}_Nv"]),
            "age": np.asarray(z[f"m{j}_age"], np.float64),
            "ndiv": np.asarray(z[f"m{j}_ndiv"], np.float64),
            # PER-JUNCTION MYOSIN, if the tissue was grown with `junction_myosin` AND the cache is new
            # enough to have recorded it. Absent -> the zoom panel draws junctions grey and says nothing,
            # rather than colouring them by a quantity it does not have.
            **({"myo": np.asarray(z[f"m{j}_myo"], np.float64)}
               if f"m{j}_myo" in z.files else {})}))
    gap = float(z["plate_gap"]) if "plate_gap" in z.files else -1.0
    return {"meshes": meshes, "Lbox": float(z["Lbox"]), "scale": float(scale),
            "n_cells": np.asarray(z["n_cells"]), "r_apical": np.asarray(z["r_apical"]),
            # IN TISSUE UNITS, from the cache -- the same number pass 1 grew the tissue against.
            # The spec carries it in BOX units; taking it from there would mean dividing by the scale
            # and getting a plate drawn at a slightly different place than the one the cells hit.
            "plate_gap": (None if gap <= 0 else gap),
            "r_eq": (np.asarray(z["r_eq"]) if "r_eq" in z.files else None),
            "r_ax": (np.asarray(z["r_ax"]) if "r_ax" in z.files else None)}


def divided_mask(mt):
    """`age <= 4 AND ndiv > 0` -- the green wash, from the division event itself.

    BOTH CONDITIONS ARE NEEDED. `age` starts at 0 for every seeded cell, so `age <= 4` alone paints
    the entire untouched tissue green for the opening frames -- a defect caught by watching a movie,
    not by a check. `ndiv > 0` says the cell has actually divided at least once.
    """
    age, nd = mt.get("age"), mt.get("ndiv")
    if age is None or not np.isfinite(np.asarray(age, float)).any():
        return None                                  # older cache: say nothing rather than lie
    div = np.asarray(age)[:mt["nF"]] <= DIVIDED_WINDOW
    if nd is not None and np.isfinite(np.asarray(nd, float)).any():
        div = div & (np.asarray(nd)[:mt["nF"]] > 0)
    return div


def broken_mask(mt, pos, name=""):
    """Magenta: cells that are not cells any more. Loud when unavailable, never silently absent."""
    try:
        from tyssue_diag import mesh_faults
        return mesh_faults(pos, mt)["broken"]
    except Exception as e:
        print(f"[{name}] broken-cell overlay unavailable ({type(e).__name__}) -- the movie cannot "
              f"show a broken cell, so absence of magenta is not evidence of a healthy mesh",
              flush=True)
        return None


# --------------------------------------------------------------------------- the matrix
def screen_basis(elev, azim):
    """(depth into screen, horizontal right, vertical up) for a matplotlib 3D camera.
    Mirrors `run_tyssue_round._screen_basis`, which is fixed at the side camera."""
    er, az = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(er) * np.cos(az), np.cos(er) * np.sin(az), np.sin(er)])
    v = np.array([0.0, 0.0, 1.0]) - (np.array([0.0, 0.0, 1.0]) @ d) * d
    v /= np.linalg.norm(v) + 1e-12
    u = np.cross(v, d); u /= np.linalg.norm(u) + 1e-12
    return d, u, v


def _matrix_scatter(ax, q, band, cmap, zorder, alpha, s_rest=1.1, s_hot=2.4, three_d=True):
    """Rest (band 0) dim and small, strained bright and larger, hottest band drawn last.

    THE REST STATE IS DIM BUT IT IS NOT ABSENT. `ecm_spec` already made this mistake once at the
    palette level -- band 0 at near-black on black produced a frame containing nothing, and you
    cannot watch a front propagate into fibres you cannot see -- and drawing it at alpha 0.4 and
    half a point wide reintroduced it at the renderer level: frame 0 of the smoke test was an empty
    panel for a matrix of 110,000 particles.
    """
    rest = band == 0
    hot = ~rest
    args = dict(cmap=cmap, vmin=0, vmax=7, marker=".", linewidths=0)
    if three_d:
        args["depthshade"] = False
    if rest.any():
        r = q[rest]
        xs = (r[:, 0], r[:, 1], r[:, 2]) if three_d else (r[:, 0], r[:, 1])
        ax.scatter(*xs, c=band[rest], s=s_rest, alpha=alpha * 0.7, zorder=zorder, **args)
    if hot.any():
        h = q[hot]; b = band[hot]
        o = np.argsort(b)                              # hottest last, so the front is never buried
        h, b = h[o], b[o]
        xs = (h[:, 0], h[:, 1], h[:, 2]) if three_d else (h[:, 0], h[:, 1])
        ax.scatter(*xs, c=b, s=s_hot, alpha=alpha, zorder=zorder + 1, **args)


# THE BLOCK'S OWN RAMP, and it must not be the matrix's. Two materials in one frame coloured by the
# same palette are indistinguishable, and the whole point of an elastic block is telling its
# deformation apart from the matrix's. Slate to white: unstrained block is clearly SOLID (much brighter
# than the matrix's dim rest state), and strain brightens it toward white without entering the inferno
# hues the matrix owns.
BLOCK_COLORS = [
    [0.30, 0.32, 0.36], [0.39, 0.41, 0.45], [0.48, 0.50, 0.55], [0.57, 0.60, 0.65],
    [0.66, 0.70, 0.76], [0.76, 0.81, 0.87], [0.87, 0.91, 0.96], [1.00, 1.00, 1.00],
]

PLATE_FACE = (0.62, 0.64, 0.70, 0.30)      # solid, inert, and not any colour that already means
PLATE_EDGE = (0.85, 0.87, 0.92, 0.85)      # something: white/green are cells, the inferno ramp is stress


def draw_plates_3d(ax, gap, L, zorder=4):
    """The two rigid blocks, as their INNER faces -- the surfaces the tissue is actually stopped by.

    Drawing the full slabs would fill a third of the frame with translucent grey at elev 18 and bury
    the matrix behind them. The inner face IS the constraint; the material above it is inert by
    definition, so a face plus an edge says everything the block does.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as _np
    quads = [_np.array([[-L, -L, z], [L, -L, z], [L, L, z], [-L, L, z]]) for z in (-gap, gap)]
    pc = Poly3DCollection(quads, facecolors=[PLATE_FACE] * 2, edgecolors=[PLATE_EDGE] * 2,
                          linewidths=0.8)
    pc.set_zorder(zorder)
    ax.add_collection3d(pc)


def block_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap(BLOCK_COLORS)


def draw_3d(ax, mt, pos, q, band, cmap, cam, L, div=None, brk=None, tissue=True, cutaway=False,
            plate_gap=None, blk=None):
    """One 3D panel: far matrix -> epithelium -> near matrix, in tissue units.

    `cutaway` removes the octant nearest the camera instead of drawing the tissue. A solid cube of
    matrix hides its own interior from every angle -- the bright front is inside it, and the panel
    reads as "the surface got warmer". With the near octant gone you look in along two cut planes
    and the front is a shell you can see the thickness of.
    """
    from run_tyssue_vesicle import _draw
    if cutaway:
        d, u, v = screen_basis(cam["elev"], cam["azim"])
        keep = ~(((q @ d) < 0) & ((q @ u) > 0) & ((q @ v) > 0))
        q, band = q[keep], band[keep]
    if tissue:
        _draw(ax, pos, mt, P0, azim=cam["azim"], act=None, inner=INNER, Lbox=L,
              divided=div, broken=brk, wall_shade=1.0)
        # `_draw` ends on view_init(elev=18, ...); re-aim for the top camera. Its Poly3DCollection
        # is the only artist on the axis at this point, and it has to sit BETWEEN the two halves of
        # the matrix, so give it an explicit zorder rather than trusting a computed one.
        for c in ax.collections:
            c.set_zorder(5)
    else:
        ax.clear(); ax.set_facecolor("black")
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
        ax.set_box_aspect((1, 1, 1)); ax.axis("off")
    d, _, _ = screen_basis(cam["elev"], cam["azim"])
    depth = q @ d                                      # > 0 is deeper than the tissue centre
    far, near = depth > 0, depth <= 0
    if tissue:
        _matrix_scatter(ax, q[far], band[far], cmap, zorder=0, alpha=0.85)
        _matrix_scatter(ax, q[near], band[near], cmap, zorder=10, alpha=0.28)
    else:
        _matrix_scatter(ax, q[far], band[far], cmap, zorder=0, alpha=0.9)
        _matrix_scatter(ax, q[near], band[near], cmap, zorder=10, alpha=0.9)
    if blk is not None:
        # THE BLOCK, SPLIT THE SAME WAY. It is a body enclosing the tissue from two sides, so a single
        # scatter has the same per-artist depth problem the matrix has -- and the block is opaque
        # enough that getting it wrong hides the experiment rather than blurring it.
        qb, bb = blk
        db = qb @ d
        _matrix_scatter(ax, qb[db > 0], bb[db > 0], block_cmap(), zorder=1, alpha=0.9,
                        s_rest=1.6, s_hot=2.8)
        _matrix_scatter(ax, qb[db <= 0], bb[db <= 0], block_cmap(), zorder=11, alpha=0.22,
                        s_rest=1.6, s_hot=2.8)
    elif plate_gap is not None:
        # ONLY WHEN THERE IS NO BLOCK. A translucent quad drawn over a material that is already there
        # would be a second, disagreeing picture of the same object.
        draw_plates_3d(ax, plate_gap, L)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1)); ax.axis("off")
    ax.view_init(elev=cam["elev"], azim=cam["azim"])


def draw_cross(ax, mt, pos, q, band, cmap, L2, axis_dir, slab, dot_scale=1.0, plate_gap=None,
               blk=None):
    """The monolayer in section + the matrix in the SAME plane, cut in the SCREEN plane.

    `seed_dir=None` on purpose, which makes `_cross_screen` fall back to the camera's own frame: the
    cut is the plane you are looking THROUGH in the 3D panel, so its vertical is that panel's vertical.
    The alternative -- `_cross_screen`'s tube convention, which puts `seed_dir` along the plot
    HORIZONTAL -- is right for a tube and wrong here: it would show the confined axis lying left-to-right
    beside a 3D view with the blocks at top and bottom, and the two panels of one figure would disagree
    about which way is up.

    `axis_dir` is kept in the signature because the CAVITY still has a pinched axis worth recording,
    and callers pass it; the cut plane no longer depends on it.
    """
    from run_tyssue_round import _cross_screen, _screen_basis
    _cross_screen(ax, pos, mt, np.zeros(mt["nF"]), seed_dir=None, inner=INNER, Lbox=L2)
    # THE SAME PLANE, DERIVED THE SAME WAY -- `_cross_screen`'s own fallback frame, recomputed here so
    # the matrix slab and the cell ring are one picture instead of two unrelated ones.
    d, u, v = _screen_basis()
    sl = np.abs(q @ d) < slab
    if sl.any():
        proj = np.stack([q[sl] @ u, q[sl] @ v], axis=1)
        # BIGGER DOTS THAN THE 3D PANELS. A slab holds ~11% of the particles spread over a panel of
        # the same size, so at the 3D dot size the section reads as an empty frame with a cell ring
        # floating in it -- and the section is the panel where the front's TIMING is legible.
        #
        # BUT SCALED TO THE PANEL. A marker size is in POINTS, not in data units, so the same `s`
        # that is right for a full strip panel is four times too big in a small one. `dot_scale` is
        # the caller's panel size, not a taste setting.
        _matrix_scatter(ax, proj, band[sl], cmap, zorder=0, alpha=0.95,
                        s_rest=3.4 * dot_scale, s_hot=7.0 * dot_scale, three_d=False)
    if blk is not None:
        qb, bb = blk
        sb = np.abs(qb @ d) < slab
        if sb.any():
            _matrix_scatter(ax, np.stack([qb[sb] @ u, qb[sb] @ v], axis=1), bb[sb], block_cmap(),
                            zorder=1, alpha=0.95, s_rest=3.4 * dot_scale, s_hot=7.0 * dot_scale,
                            three_d=False)
    elif plate_gap is not None:
        # WHERE THE PLATE ACTUALLY CROSSES THIS PLANE. The plate is the plane z = +/-gap and the cut's
        # vertical `v` is only MOSTLY z (its z-component is 0.951 at elev 18), so the intersection sits
        # at gap / v_z, not at gap. Drawing it at `gap` would misreport the gap by 5% -- small, and
        # exactly the kind of small that turns a measurement into an illustration.
        from matplotlib.patches import Rectangle
        b = plate_gap / max(abs(float(v[2])), 1e-9)
        for lo in (b, -L2):
            ax.add_patch(Rectangle((-L2, lo if lo > 0 else -L2), 2 * L2, L2 - b,
                                   facecolor=PLATE_FACE, edgecolor=PLATE_EDGE, lw=0.8, zorder=2))
    ax.set_xlim(-L2, L2); ax.set_ylim(-L2, L2); ax.set_aspect("equal"); ax.axis("off")


# --------------------------------------------------------------------------- the two new levels
# COLOURS THAT CANNOT BE CONFUSED WITH THE THREE ALREADY IN THE FRAME. The stroma owns the warm inferno
# ramp, the elastic block owns slate-to-white, and the cells own white-with-a-green-wash. So the two new
# entities take the two remaining directions: myosin goes COOL (cyan -> deep blue) and the basement
# membrane goes GREEN -> YELLOW. In the zoom panel the cells are drawn as outlines only, so nothing
# green there is a dividing cell.
MYOSIN_COLORS = [
    [0.55, 0.95, 0.98], [0.42, 0.83, 0.96], [0.31, 0.70, 0.93], [0.23, 0.57, 0.88],
    [0.18, 0.44, 0.80], [0.14, 0.32, 0.70], [0.10, 0.22, 0.58], [0.06, 0.13, 0.45],
]
MEMBRANE_COLORS = [
    [0.20, 0.72, 0.35], [0.34, 0.78, 0.31], [0.50, 0.84, 0.28], [0.66, 0.88, 0.25],
    [0.80, 0.90, 0.22], [0.90, 0.86, 0.20], [0.97, 0.78, 0.18], [1.00, 0.66, 0.15],
]


def _cmap(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def draw_zoom(ax, mt, pos, mem_q=None, mem_s=None, cam=None, frac=0.34, myo_hi=None,
              mem_hi=None, name=""):
    """A ZOOM on one patch of the surface: junctions coloured by MYOSIN, membrane by BOND STRAIN.

    WHY A SEPARATE PANEL AND NOT A COLOUR ON THE EXISTING ONES. Both new entities live in a shell a few
    percent of the tissue radius thick. At the whole-tissue framing that is two or three pixels: the
    junction network is smaller than the line width used to draw a cell, and the membrane is a bright rim
    one dot wide. Neither can be read at the scale the other panels need, so they get their own frame --
    which is also the only place a colour scale can honestly be spent on them.

    THE PATCH IS FIXED IN SPACE, not re-centred per frame. It looks at the +x side of the tissue and stays
    there, so material entering or leaving the window is a real event rather than the camera moving. The
    window WIDTH is a fraction of the current tissue radius, so the patch shows about the same number of
    cells throughout a run in which the radius triples -- otherwise the last frames would be a single cell.
    """
    import numpy as np
    from matplotlib.collections import LineCollection
    cam = cam or CAM_SIDE
    d, u, v = screen_basis(cam["elev"], cam["azim"])
    ax.clear(); ax.set_facecolor("black")

    R = float(np.percentile(np.linalg.norm(pos, axis=1), 98))
    half = frac * R
    # the patch centre: on the surface, on the +x side, projected into the screen plane
    ctr = np.array([R, 0.0, 0.0])

    es, et, ef = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"])
    nF = int(mt["nF"])
    live = ef < nF
    a, b = pos[es[live]], pos[et[live]]
    mid = 0.5 * (a + b)
    # keep junctions in the window AND on the near side, so the far hemisphere does not overprint it
    near = ((mid - ctr) @ d) > -half
    inwin = (np.abs((mid - ctr) @ u) < half) & (np.abs((mid - ctr) @ v) < half) & near
    myo = np.asarray(mt["myo"], float)[live] if "myo" in mt else None
    if inwin.any():
        segs = np.stack([np.stack([(a[inwin] - ctr) @ u, (a[inwin] - ctr) @ v], 1),
                         np.stack([(b[inwin] - ctr) @ u, (b[inwin] - ctr) @ v], 1)], axis=1)
        if myo is None:
            lc = LineCollection(segs, colors="#666", linewidths=0.8, zorder=3)
        else:
            hi = myo_hi or max(float(np.percentile(myo, 98)), 1e-9)
            lc = LineCollection(segs, cmap=_cmap(MYOSIN_COLORS), linewidths=1.6, zorder=3)
            lc.set_array(np.clip(myo[inwin] / hi, 0, 1))
            lc.set_clim(0, 1)
        ax.add_collection(lc)

    if mem_q is not None:
        mq = mem_q
        rel = mq - ctr
        keep = (np.abs(rel @ u) < half) & (np.abs(rel @ v) < half) & ((rel @ d) > -half)
        if keep.any():
            xy = np.stack([rel[keep] @ u, rel[keep] @ v], axis=1)
            if mem_s is None:
                ax.scatter(xy[:, 0], xy[:, 1], s=2.0, c="#3cb85a", marker=".", linewidths=0,
                           zorder=2)
            else:
                s = np.asarray(mem_s, float)[keep]
                hi = mem_hi or max(float(np.percentile(np.asarray(mem_s, float), 99)), 1e-9)
                ax.scatter(xy[:, 0], xy[:, 1], s=3.0, c=np.clip(s / hi, 0, 1),
                           cmap=_cmap(MEMBRANE_COLORS), vmin=0, vmax=1, marker=".",
                           linewidths=0, zorder=2)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")

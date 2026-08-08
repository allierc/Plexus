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
            plate_gap=None, blk=None, mem=None):
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
    if mem is not None:
        # THE BASEMENT MEMBRANE IN THE MAIN PANELS TOO. It was drawn only in the zoom, which made a
        # 30,000-particle body carrying the whole tissue-matrix interface invisible in the two panels
        # people actually look at -- so a sheet that had disintegrated looked like a sheet that was
        # simply not shown. Drawn on its own green-to-amber ramp, between the far and near halves of the
        # stroma, at the surface where it belongs.
        qm, sm = mem
        dm = qm @ screen_basis(cam["elev"], cam["azim"])[0]
        for sel, al, zo in ((dm > 0, 0.95, 2), (dm <= 0, 0.35, 12)):
            if sel.any():
                ax.scatter(qm[sel, 0], qm[sel, 1], qm[sel, 2], c=np.clip(sm[sel], 0, 1),
                           cmap=_cmap(MEMBRANE_COLORS), vmin=0, vmax=1, s=1.4, marker=".",
                           linewidths=0, alpha=al, zorder=zo, depthshade=False)
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
               blk=None, mem=None, zoom_half=None):
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
    if mem is not None:
        qm, sm = mem
        sl_m = np.abs(qm @ d) < slab
        if sl_m.any():
            ax.scatter((qm[sl_m] @ u), (qm[sl_m] @ v), c=np.clip(sm[sl_m], 0, 1),
                       cmap=_cmap(MEMBRANE_COLORS), vmin=0, vmax=1, s=4.0 * dot_scale,
                       marker=".", linewidths=0, zorder=3)
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
    if zoom_half:
        # ZOOM BY MOVING THE AXES LIMITS, not by re-deriving a windowed slice. Everything above already
        # drew at full scale through the SAME routine the whole-tissue panel uses, so the zoom cannot
        # disagree with the panel above it -- it is literally the same drawing, clipped. The window sits
        # on the +u surface crossing, which is where the layering is: lumen, epithelium, basement
        # membrane, stroma, in that order outward.
        rs = float(np.percentile(np.linalg.norm(pos, axis=1), 98))
        ax.set_xlim(rs - zoom_half, rs + zoom_half)
        ax.set_ylim(-zoom_half, zoom_half)
    else:
        ax.set_xlim(-L2, L2); ax.set_ylim(-L2, L2)
    ax.set_aspect("equal"); ax.axis("off")


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
STRAIN_COLORS = ["#1b3a5c", "#1f6f8b", "#3aa17e", "#8cc04f", "#e8d44d", "#f0913a", "#e0452b"]
MEMBRANE_COLORS = [
    [0.20, 0.72, 0.35], [0.34, 0.78, 0.31], [0.50, 0.84, 0.28], [0.66, 0.88, 0.25],
    [0.80, 0.90, 0.22], [0.90, 0.86, 0.20], [0.97, 0.78, 0.18], [1.00, 0.66, 0.15],
]


def _cmap(colors):
    from matplotlib.colors import ListedColormap
    return ListedColormap(colors)


def _myo_of(mt, n_half, myo_new=1.0):
    """`myo` padded to the snapshot's half-edge count, rather than discarded when it is short.

    `junction_myosin` writes `m["myo"]` before `divide_3d` and `reconnect_t1_3d` run, and division
    APPENDS half-edges, so a snapshot's table is routinely longer than the myosin array recorded with it
    -- 28% of frames in run 65, by 6 to 1404 entries. The old guard skipped the colouring on those
    frames and fell back to a flat light blue, so the panel alternated between a colour-mapped network
    and a uniform one: a 28%-duty-cycle flicker that looked like a bad colour scale.

    The missing entries are exactly the junctions born that frame, and a newborn junction starts at
    `myo_new`. So the padding is not a cosmetic patch, it is the value the operator would have written.
    """
    if "myo" not in mt:
        return None
    m = np.asarray(mt["myo"], float)
    if m.shape[0] == n_half:
        return m
    if m.shape[0] < n_half:
        return np.concatenate([m, np.full(n_half - m.shape[0], myo_new)])
    return m[:n_half]


def draw_junctions_3d(ax, mt, pos, cam, L, myo_hi=None, cutaway=True, lw=0.9):
    """THE JUNCTION NETWORK IN 3D, WITH NOTHING ELSE IN FRONT OF IT.

    The same framing as `draw_3d(tissue=False)` -- no cell faces, no matrix, no membrane -- so the
    network is seen as an object rather than as an edge-on rim. Cells are what the epithelium is made of,
    but the junctions are where myosin lives, and drawn together the faces hide them completely.

    `cutaway` drops the half nearest the camera, which is the only way a closed shell of lines reads at
    all: a full sphere of edges is a solid mass of ink from any angle.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.clear(); ax.set_facecolor("black")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1)); ax.axis("off")
    ax.view_init(elev=cam["elev"], azim=cam["azim"])

    es, et, ef = np.asarray(mt["E_srce"]), np.asarray(mt["E_trgt"]), np.asarray(mt["E_face"])
    live = ef < int(mt["nF"])
    a, b = pos[es[live]], pos[et[live]]
    d, _, _ = screen_basis(cam["elev"], cam["azim"])
    keep = ((0.5 * (a + b)) @ d) > 0 if cutaway else np.ones(len(a), bool)
    if not keep.any():
        return
    segs = np.stack([a[keep], b[keep]], axis=1)
    # DEPTH-SORTED, because Line3DCollection is not. Matplotlib draws the segments in array order under a
    # single z-order, and the array order is the half-edge table -- which `divide_3d` and
    # `reconnect_t1_3d` PERMUTE every frame. So which of 12,000 overlapping lines ends up on top changes
    # frame to frame for reasons that have nothing to do with geometry, and the panel shimmers. Measured
    # on run 67 the colour is not moving at all (rendered mean 0.667 -> 0.670, max step 0.006) and the
    # visible set barely is (median 6 edges differ between frames), so the flicker is entirely this.
    # Sorting by depth makes the draw order a function of geometry, which is stable across frames.
    order = np.argsort(-(0.5 * (segs[:, 0] + segs[:, 1]) @ d))
    segs = segs[order]
    _m = _myo_of(mt, live.shape[0])
    myo = _m[live] if _m is not None else None
    if myo is None:
        lc = Line3DCollection(segs, colors="#7ab8ff", linewidths=lw)
    else:
        hi = myo_hi or max(float(np.percentile(myo, 98)), 1e-9)
        lc = Line3DCollection(segs, cmap=_cmap(MYOSIN_COLORS), linewidths=lw)
        lc.set_array(np.clip(myo[keep][order] / hi, 0, 1)); lc.set_clim(0, 1)
    ax.add_collection3d(lc)


def draw_membrane_3d(ax, mem_q, mem_s, cam, L, mem_hi=None, cutaway=True, s_dot=4.5, alive=None,
                     bonds=None, bond_s=None, bond_hi=None):
    """THE BASEMENT MEMBRANE ALONE, framed and cut exactly as `draw_junctions_3d` frames the network.

    The pair is the point: two panels, one per entity, same camera, same cutaway, same tissue-sized box,
    so the sheet and the network can be compared frame to frame instead of hunted for inside a cloud of
    matrix. Neither panel draws the other's material, and neither draws the interstitial ECM.
    """
    ax.clear(); ax.set_facecolor("black")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
    ax.set_box_aspect((1, 1, 1)); ax.axis("off")
    ax.view_init(elev=cam["elev"], azim=cam["azim"])
    if mem_q is None or len(mem_q) == 0:
        return
    d, _, _ = screen_basis(cam["elev"], cam["azim"])
    keep = (mem_q @ d) > 0 if cutaway else np.ones(len(mem_q), bool)
    # the unsecreted reserve is parked at the centre and is not membrane yet
    if alive is not None and len(alive) == len(mem_q):
        keep = keep & np.asarray(alive, bool)
    if not keep.any():
        return
    q = mem_q[keep]
    # NODES AND EDGES, DRAWN DIFFERENTLY, because topology is a property of the EDGES. A cloud of dots
    # cannot distinguish a well-connected sheet from the same dots with every crosslink broken, and
    # "structured / broken" is exactly that distinction. Nodes are a single flat colour so they read as
    # material; the colour scale is spent on the crosslinks, mapped to ELONGATION (L - rest)/rest, which
    # is the quantity that decides whether a bond survives.
    if bonds is not None and len(bonds[0]):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        bi, bj = bonds
        vis = keep[bi] & keep[bj] if keep.dtype == bool else None
        if vis is None or vis.any():
            sel = np.ones(len(bi), bool) if vis is None else vis
            segs = np.stack([mem_q[bi[sel]], mem_q[bj[sel]]], axis=1)
            hi_b = bond_hi or 0.35                       # the break threshold: full scale IS failure
            lc = Line3DCollection(segs, cmap=_cmap(STRAIN_COLORS), linewidths=0.7, alpha=0.9)
            lc.set_array(np.clip(np.asarray(bond_s, float)[sel] / hi_b, 0, 1)); lc.set_clim(0, 1)
            ax.add_collection3d(lc)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=s_dot * 0.5, c="#9fb0c0", marker=".", linewidths=0)
    elif mem_s is None:
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=s_dot, c="#3cb85a", marker=".", linewidths=0)
    else:
        v = np.asarray(mem_s, float)[keep]
        hi = mem_hi or max(float(np.percentile(np.asarray(mem_s, float), 99)), 1e-9)
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=s_dot, c=np.clip(v / hi, 0, 1),
                   cmap=_cmap(MEMBRANE_COLORS), vmin=0, vmax=1, marker=".", linewidths=0)


def draw_zoom(ax, mt, pos, mem_q=None, mem_s=None, cam=None, frac=0.55, myo_hi=None, r_ref=None,
              mem_hi=None, name="", lw=None, junctions=True, bonds=None, bond_s=None,
              bond_hi=None):
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
    # A WINDOW THAT SCALES WITH R CANNOT SHOW GROWTH. Sizing the patch as frac*R(t) keeps roughly the
    # same number of cells in frame all run, which is what the original comment above wanted -- and it
    # makes a tissue whose radius triples look completely static, which is what the inset is watching.
    # `r_ref` sizes the window ONCE, from the final radius, so the surface genuinely grows across it.
    # The centre still follows the current surface, so the patch stays on the sheet instead of drifting
    # off it, and the window never cuts inside the spheroid.
    Rw = float(r_ref) if r_ref else R
    half = frac * Rw
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
    _m = _myo_of(mt, live.shape[0])
    myo = _m[live] if _m is not None else None
    if inwin.any() and junctions:
        segs = np.stack([np.stack([(a[inwin] - ctr) @ u, (a[inwin] - ctr) @ v], 1),
                         np.stack([(b[inwin] - ctr) @ u, (b[inwin] - ctr) @ v], 1)], axis=1)
        if myo is None:
            lc = LineCollection(segs, colors="#7ab8ff", linewidths=(lw or 1.4), zorder=3)
        else:
            hi = myo_hi or max(float(np.percentile(myo, 98)), 1e-9)
            lc = LineCollection(segs, cmap=_cmap(MYOSIN_COLORS), linewidths=(lw or 1.6), zorder=3)
            lc.set_array(np.clip(myo[inwin] / hi, 0, 1))
            lc.set_clim(0, 1)
        ax.add_collection(lc)

    # THE SAME QUANTITY AND RAMP AS THE PANEL AROUND IT. The inset used to draw per-PARTICLE strain on
    # the green->amber membrane ramp while its parent panel drew per-BOND elongation on blue->red: two
    # different quantities on two different scales inside one frame, which invites reading the zoom as a
    # magnified version of the panel when it was measuring something else.
    if bonds is not None and mem_q is not None and len(bonds[0]):
        bi, bj = bonds
        rel_i, rel_j = mem_q[bi] - ctr, mem_q[bj] - ctr
        inb = ((np.abs(rel_i @ u) < half) & (np.abs(rel_i @ v) < half) & ((rel_i @ d) > -half)
               & (np.abs(rel_j @ u) < half) & (np.abs(rel_j @ v) < half))
        if inb.any():
            segs = np.stack([np.stack([rel_i[inb] @ u, rel_i[inb] @ v], 1),
                             np.stack([rel_j[inb] @ u, rel_j[inb] @ v], 1)], axis=1)
            hb = bond_hi or 0.35
            bc = LineCollection(segs, cmap=_cmap(STRAIN_COLORS), linewidths=1.1, zorder=2)
            bc.set_array(np.clip(np.asarray(bond_s, float)[inb] / hb, 0, 1)); bc.set_clim(0, 1)
            ax.add_collection(bc)
        mem_q = None                      # the nodes are drawn by the network above; skip the dot cloud
    if mem_q is not None:
        mq = mem_q
        rel = mq - ctr
        keep = (np.abs(rel @ u) < half) & (np.abs(rel @ v) < half) & ((rel @ d) > -half)
        if keep.any():
            xy = np.stack([rel[keep] @ u, rel[keep] @ v], axis=1)
            if mem_s is None:
                ax.scatter(xy[:, 0], xy[:, 1], s=13.0, c="#3cb85a", marker=".", linewidths=0,
                           zorder=2)
            else:
                s = np.asarray(mem_s, float)[keep]
                hi = mem_hi or max(float(np.percentile(np.asarray(mem_s, float), 99)), 1e-9)
                ax.scatter(xy[:, 0], xy[:, 1], s=13.0, c=np.clip(s / hi, 0, 1),
                           cmap=_cmap(MEMBRANE_COLORS), vmin=0, vmax=1, marker=".",
                           linewidths=0, zorder=2)
    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
    ax.set_aspect("equal"); ax.axis("off")

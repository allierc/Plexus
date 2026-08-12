"""
The bottom-left panel of the 2x2: the basement membrane, in the two forms it takes.

WHY THIS IS ONE FILE AND NOT TWO. Without protease the BM's story is mechanical -- it is stretched by
a growing tissue and held to it by plaques -- so the surface carries lambda_geo and the plaques are
the thing to see. With protease the story is chemical: the surface carries MT1-MMP expression, the
source of the whole cascade, and the plaques still have to be visible because the breach forms where
adhesion and chemistry meet. Same geometry, same camera, same plaques; one field swaps for another.
So it is one drawer with a `mode`, and the panel cannot silently disagree with itself between runs.

THE INSET IS A CROSS-SECTION, not a magnified copy of the 3D view. A surface coloured by a field
shows WHERE; only a section shows the two bodies and the gap between them, which is what a plaque is
a statement about. It sits bottom-right inside the panel, where the neighbouring panel's zoom sits.

AND IT IS CUT BY THE CAMERA, NOT BY THE WORLD. The section used to be a slab of constant y, drawn in
x--z: a second viewpoint inside a figure whose other three panels share one. It is now a slab about the
camera's own screen plane, drawn in the camera's (right, up) basis -- the same view as the panel around
it, magnified -- and the slab is taken at the LIMB, the point of the surface whose outward normal lies
along screen-right. That is the one place where the standoff a plaque spans is a HORIZONTAL distance
rather than a distance into the screen: cut the same slab at the front of the sphere and the sheet, the
plaque and the epithelium all project onto the same few pixels.

PLAQUES ARE DRAWN AT THEIR ATTACHMENT POINT ON THE EPITHELIUM, not at the sheet node -- a plaque is a
statement about where the sheet is HELD, and the two differ by the standoff, which is the quantity
G46 is about. In the section they are drawn as segments, because there they can be.
"""
import matplotlib
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

PLQ_C = "#ff2d2d"                 # red: the plaques, in both modes and both views
EPI_C = "#efe3c0"                 # cream: the epithelium's wireframe
CMAP_LAM = "magma"                # stretch
CMAP_MT1 = "viridis"              # expression -- a different family, so the two modes never read alike


def _ramp(name, top=0.87):
    from matplotlib.colors import ListedColormap
    return ListedColormap(matplotlib.colormaps[name](np.linspace(0.0, top, 256)), name=f"{name}_cut")


def plaque_stride(n, target=2562):
    """Draw one plaque in `stride`, so the marks stay marks.

    2,562 dots on a sphere three hundred pixels across is a marked surface; 40,962 -- what the refining
    sheet ends with, one patch per live node -- is a red one, at any dot size that is still visible.
    Shrinking the dot alone does not help: the count grows sixteenfold and the area does not. So the
    panel draws a fixed NUMBER of them, sampled by a deterministic stride, and the label says which
    fraction is drawn. A picture that quietly showed 6% of the plaques would be worse than either.
    """
    return max(1, int(np.ceil(n / float(target))))


def _visible(P, c, dirv):
    """Keep the hemisphere facing the camera. The cut follows the view; a cut fixed in world space
    puts the interesting side away from the reader, which is exactly how 05h's hole ended up on the
    back."""
    return (P - c) @ dirv > 0.0


def _norm(val, lo, hi, gamma):
    """Where a value sits on the ramp, in [0,1].

    `gamma < 1` LIFTS THE BOTTOM OF THE RAMP, and it is here because lam_geo is a ramp in TIME: it
    climbs from 1.0 to 4.4 over 401 frames, so a linear scale spends its whole range on that climb and
    the first third of the movie renders as a near-black ball -- the sheet invisible under its own
    plaques, at exactly the frames where it is being stretched fastest. The exponent is stated with the
    figure and the per-frame min-max is printed on the panel, so nothing is hidden by it; what it buys
    is that the early frames are the DARK end of a visible ramp rather than the background.
    """
    return np.clip((val - lo) / (hi - lo), 0, 1) ** gamma


def draw_bm(ax, X, F, val, plq_p, c, lim, mode="lam", vmax=None, XE=None, FE=None,
            elev=16.0, azim=-58.0, label=None, inset=True, x_node=None, zoom=1.0, win=None,
            gamma=0.5):
    """One BM panel. `val` is per FACE; `plq_p` are the attachment points on the epithelium.

    mode "lam"  -- val is lambda_geo, coloured from 1 to vmax
    mode "mt1"  -- val is MT1-MMP expression per cell, coloured from 0 to vmax
    """
    ax.set_facecolor("black")
    ax.axis("off")
    dirv = np.array([np.cos(np.radians(elev)) * np.cos(np.radians(azim)),
                     np.cos(np.radians(elev)) * np.sin(np.radians(azim)),
                     np.sin(np.radians(elev))])
    lo = 1.0 if mode == "lam" else 0.0
    hi = float(vmax if vmax is not None else np.percentile(val, 99))
    hi = hi if hi > lo else lo + 1.0
    # THE RAMP IS TRUNCATED BEFORE ITS WHITE END. magma and viridis both finish at a near-white, and the
    # epithelium in this panel is drawn cream -- so a sheet at the top of its scale and the tissue under
    # it came out the same colour, in the 3D view and in the section where they are a line apart. Cutting
    # the last eighth costs nothing that the printed min-max does not already carry, and keeps the two
    # bodies two colours.
    cm = _ramp(CMAP_LAM if mode == "lam" else CMAP_MT1)

    # THE EPITHELIUM GOES DOWN FIRST, AND THE SHEET COVERS IT. These panels are drawn with
    # `computed_zorder=False`, so matplotlib paints collections in the order they are added and not by
    # depth: adding the epithelium second drew the tissue's whole wireframe ON TOP of a basement
    # membrane that is outside it, which is the one spatial relation this panel exists to assert. Under
    # the sheet it shows only where the sheet does not reach -- at the silhouette, and through a breach.
    # AND THE ORDER IS SET EXPLICITLY, because insertion order is not it. Under `computed_zorder=False`
    # matplotlib paints by the artists' own zorder, and a Line3DCollection defaults to 2 against a
    # Poly3DCollection's 1 -- so adding the epithelium first was not enough: its wireframe still came out
    # over the sheet, and the tissue appeared to lie outside its own basement membrane.
    if XE is not None and FE is not None:
        ke = _visible(XE[FE].mean(1), c, dirv)
        ax.add_collection3d(Line3DCollection(XE[FE][ke][:, [0, 1, 2, 0]], colors=EPI_C,
                                             linewidths=0.3, alpha=0.45, zorder=1))
    # EACH FACE IS OUTLINED IN ITS OWN COLOUR. With `linewidths=0` matplotlib antialiases every triangle
    # against the black background and leaves a hairline of it between neighbours: on a 5,120-face sheet
    # that is a dark wireframe over the whole surface, which reads as the epithelium's mesh showing
    # through -- a picture of the sheet having sunk inside the tissue, produced entirely by the renderer.
    # An edge in the face's own colour closes the seam and adds nothing that is not already there.
    keep = _visible(X[F].mean(1), c, dirv)
    fc = cm(_norm(val[keep], lo, hi, gamma))
    tri = Poly3DCollection(X[F][keep], linewidths=0.35, zorder=2)
    tri.set_facecolor(fc)
    tri.set_edgecolor(fc)
    ax.add_collection3d(tri)
    if plq_p is not None and len(plq_p):
        # s=6, NOT 1.4: at the whole-tissue framing this panel shares with its neighbours a 1.4-point
        # dot is under a pixel, and 2,562 of them read as a faint dusting rather than as the attachment
        # points they are. The COUNT, not the size, is what `plaque_stride` holds fixed.
        pp = plq_p[::plaque_stride(len(plq_p))]
        kp = _visible(pp, c, dirv)
        ax.scatter(pp[kp][:, 0], pp[kp][:, 1], pp[kp][:, 2], s=6.0, c=PLQ_C,
                   marker=".", linewidths=0, depthshade=False, zorder=3)
    ax.set_xlim(c[0] - lim, c[0] + lim)
    ax.set_ylim(c[1] - lim, c[1] + lim)
    ax.set_zlim(c[2] - lim, c[2] + lim)
    # `zoom` DEFAULTS TO 1, which is `draw_3d`'s and `draw_junctions_3d`'s. Inside the 2x2 this panel
    # has to render its subject at the same apparent size as the panels around it, and a magnification
    # that only this drawer applies is a second camera convention in one figure. A standalone figure,
    # where there is nothing to be compared against, can ask for more.
    try:
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if label:
        # TOP-RIGHT, at the SAME SIZE as the run's own counter in the top-left panel. The two are the
        # figure's only text and they now read as one convention rather than two; right-aligned so it
        # cannot run into the counter, and the inset has the bottom-right corner.
        ax.text2D(0.98, 0.96, label, transform=ax.transAxes, color="white", fontsize=9,
                  ha="right", va="top")

    if inset:
        _inset_section(ax, X, F, val, plq_p, c, lim, lo, hi, cm, elev, azim, x_node=x_node,
                       XE=XE, FE=FE, win=win, gamma=gamma)


def _edges_2d(P, T, keep, ctr, u, v):
    """The triangles of `T` whose CENTROID is in `keep`, as screen-projected edge segments, and the
    index of the triangle each segment came from.

    THE TEST IS ON THE CENTROID, and it was on the vertices twice before. `any vertex` draws a triangle
    with one corner in the slab and two behind it as a long line running out of the window from a point
    on the surface -- the epithelium's section came out as a spray of cream spurs. `all three vertices`
    then emptied the inset instead: a slab thinner than the mesh spacing contains whole triangles almost
    nowhere, so the sheet vanished from its own section. A centroid is in the cut or it is not, and the
    edges it drags along reach at most one edge-length past the window, which is a section.

    THE SURFACE IS DRAWN AS ITS EDGES, NOT AS ITS NODES, and at this zoom that is the difference between
    a surface and two dots. A window a few standoffs wide holds one or two nodes of a 2,562-node sheet
    once the tissue has quadrupled -- a scatter of those reads as noise -- while the edges through the
    window cross it whatever the node spacing is, and a mesh whose edges are drawn cannot pretend to be
    denser than it is either.
    """
    m = np.asarray(keep, bool)
    if not m.any():
        return None, None
    tri = P[T[m]] - ctr                                     # (nt,3,3)
    xy = np.stack([tri @ u, tri @ v], axis=-1)              # (nt,3,2)
    segs = np.concatenate([xy[:, [0, 1]], xy[:, [1, 2]], xy[:, [2, 0]]], axis=0)
    idx = np.tile(np.flatnonzero(m), 3)
    return segs, idx


def _inset_section(ax, X, F, val, plq_p, c, lim, lo, hi, cm, elev, azim, x_node=None, XE=None,
                   FE=None, frac=0.16, win=None, gamma=1.0):
    """A zoomed section at the LIMB, bottom-right inside the panel and in the PANEL'S OWN CAMERA: the
    epithelium, the sheet, and the plaques as the segments they are.

    The basis is `ecm_render.screen_basis`, the one the three panels around this one are drawn with, so
    the inset is this view magnified rather than a second viewpoint.

    `win` IS THE HALF-WIDTH IN WORLD UNITS, AND IT IS SIZED ON THE STANDOFF, not on a fraction of the
    panel. The junction zoom beside this one takes 0.16 of the panel's extent, and at that width the gap
    a plaque spans -- l0, a few tenths of a micron against cells ten microns across -- is a tenth of a
    pixel: the sheet, the plaque and the epithelium all land on the same dot, and the inset becomes a
    magnified copy of the panel showing the one thing the panel already shows. The caller measures the
    gap over the whole run and passes ~10x it, ONCE, so the window is fixed and the surface grows across
    it. `frac` remains the fallback for a caller that has no such measurement.
    """
    from ecm_render import screen_basis
    d, u, v = screen_basis(elev, azim)

    # NO FRAME. The panel beside it draws its zoom on `axis("off")`; a box here would be the only rule
    # around anything in the figure.
    a = ax.inset_axes([0.62, 0.02, 0.36, 0.36])
    a.set_facecolor("black")
    a.axis("off")

    rel = X - c
    rn = np.linalg.norm(rel, axis=1)
    # THE LIMB, not the front. `ctr = R*u` is the point of the surface whose outward normal is
    # screen-RIGHT, so the sheet, the plaque and the epithelium under it separate horizontally in the
    # inset. At the front of the sphere the same three lie along the view axis and land on one pixel,
    # which is the whole reason the first version of this inset cut a world-fixed x--z plane instead.
    #
    # AND `R` IS THE RADIUS THERE, not the p98 of the whole sheet. On a bumpy surface the p98 sits a
    # few percent outside the material, so the window centred on it opened next to the sheet and the
    # section came out against its left edge. This is the median radius of the nodes pointing within
    # ~15 degrees of `u`, i.e. of the material actually at the limb.
    nh = rel / np.maximum(rn, 1e-12)[:, None]
    at = (nh @ u) > 0.966
    R = float(np.median(rn[at])) if at.sum() >= 8 else float(np.percentile(rn, 98))
    ctr = R * u + c
    # AND THE WINDOW IS CAPPED AT A FIFTH OF THE PANEL. `win` is sized on the plaque gap, which is the
    # right length while the sheet is intact -- but a TORN sheet retracts, its median gap goes from 0.07
    # to 3.8 tissue units, and the window opens to two thirds of the sphere. That is not a section of
    # anything: it is the panel again, in a corner. Above the cap the inset stops magnifying and says so
    # by simply being the widest section this panel will draw.
    w = min(float(win) if win else frac * lim, 0.20 * lim)

    # THE SLAB HAS A REAL THICKNESS AND IT IS THINNER THAN THE WINDOW IS WIDE. A section of zero
    # thickness through a triangulated surface contains no nodes at all, and the far side of the shell
    # has to be excluded or it overprints the near one -- but at `|q.d| < w` the cut holds several rows
    # of the mesh stacked in depth, and the inset draws them all on top of each other: the two bodies
    # and the gap between them disappear into the crosshatch, which is what the drawing exists to show.
    # A quarter of the window keeps roughly one row, and widens only if that catches nothing.
    def slab_of(P, band):
        q = P - ctr
        return (np.abs(q @ d) < band) & (np.abs(q @ u) < w) & (np.abs(q @ v) < w)

    band = 0.4 * w
    for _ in range(4):
        if slab_of(X[F].mean(1), band).sum() >= 6:
            break
        band *= 1.7

    if XE is not None and FE is not None and len(XE):
        # the epithelium's own section, so the gap a plaque spans is a gap between two DRAWN bodies
        se, _ = _edges_2d(XE, FE, slab_of(XE[FE].mean(1), band), ctr, u, v)
        if se is not None:
            a.add_collection(LineCollection(se, colors=EPI_C, linewidths=1.0, alpha=0.75, zorder=1))

    sf, si = _edges_2d(X, F, slab_of(X[F].mean(1), band), ctr, u, v)
    if sf is not None:
        lc = LineCollection(sf, cmap=cm, linewidths=float(np.clip(2.2 * (5120.0 / max(F.shape[0], 1)) ** 0.5, 0.5, 2.2)), zorder=3)
        lc.set_array(_norm(val[si], lo, hi, gamma)); lc.set_clim(0.0, 1.0)
        a.add_collection(lc)
    if plq_p is not None and len(plq_p) and x_node is not None:
        st = plaque_stride(len(plq_p))                 # the same subsample as the panel above
        plq_p, x_node = plq_p[::st], x_node[::st]
        ps = slab_of(x_node, band)
        if ps.any():
            pa, pb = plq_p[ps] - ctr, x_node[ps] - ctr
            a.add_collection(LineCollection(
                np.stack([np.stack([pa @ u, pa @ v], 1), np.stack([pb @ u, pb @ v], 1)], 1),
                colors=PLQ_C, linewidths=1.0, zorder=4))
            a.scatter(pa @ u, pa @ v, s=5.0, c=PLQ_C, marker=".", linewidths=0, zorder=5)
    a.set_xlim(-w, w)
    a.set_ylim(-w, w)
    a.set_aspect("equal")
    return a

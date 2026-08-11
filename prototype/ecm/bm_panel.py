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
a statement about. It sits top-right inside the panel, echoing the section panel above it.

PLAQUES ARE DRAWN AT THEIR ATTACHMENT POINT ON THE EPITHELIUM, not at the sheet node -- a plaque is a
statement about where the sheet is HELD, and the two differ by the standoff, which is the quantity
G46 is about. In the section they are drawn as segments, because there they can be.
"""
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

PLQ_C = "#ff2d2d"                 # red: the plaques, in both modes and both views
EPI_C = "#efe3c0"                 # cream: the epithelium's wireframe
CMAP_LAM = "magma"                # stretch
CMAP_MT1 = "viridis"              # expression -- a different family, so the two modes never read alike


def _visible(P, c, dirv):
    """Keep the hemisphere facing the camera. The cut follows the view; a cut fixed in world space
    puts the interesting side away from the reader, which is exactly how 05h's hole ended up on the
    back."""
    return (P - c) @ dirv > 0.0


def draw_bm(ax, X, F, val, plq_p, c, lim, mode="lam", vmax=None, XE=None, FE=None,
            elev=16.0, azim=-58.0, label=None, inset=True, x_node=None):
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
    cm = CMAP_LAM if mode == "lam" else CMAP_MT1

    keep = _visible(X[F].mean(1), c, dirv)
    tri = Poly3DCollection(X[F][keep], linewidths=0.0)
    import matplotlib.cm as mcm
    tri.set_facecolor(mcm.get_cmap(cm)(np.clip((val[keep] - lo) / (hi - lo), 0, 1)))
    ax.add_collection3d(tri)

    if XE is not None and FE is not None:
        ke = _visible(XE[FE].mean(1), c, dirv)
        ax.add_collection3d(Line3DCollection(XE[FE][ke][:, [0, 1, 2, 0]], colors=EPI_C,
                                             linewidths=0.3, alpha=0.45))
    if plq_p is not None and len(plq_p):
        kp = _visible(plq_p, c, dirv)
        ax.scatter(plq_p[kp][:, 0], plq_p[kp][:, 1], plq_p[kp][:, 2], s=1.4, c=PLQ_C,
                   marker=".", linewidths=0, depthshade=False)
    ax.set_xlim(c[0] - lim, c[0] + lim)
    ax.set_ylim(c[1] - lim, c[1] + lim)
    ax.set_zlim(c[2] - lim, c[2] + lim)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.55)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if label:
        # BOTTOM-left, not top: the inset lives top-right and a two-line label at the top ran under
        # it. The label is the panel's name, so it must not be the thing the inset covers.
        ax.text2D(0.02, 0.03, label, transform=ax.transAxes, color="white", fontsize=10.5,
                  va="bottom")

    if inset:
        _inset_section(ax, X, F, val, plq_p, c, lim, mode, lo, hi, cm, x_node=x_node)


def _inset_section(ax, X, F, val, plq_p, c, lim, mode, lo, hi, cm, x_node=None, band=0.006,
                   frac=0.34):
    """A zoomed x--z section, top-right inside the panel: the sheet's nodes, and the plaques as the
    segments they are. Zoomed to `frac` of the panel's extent about the centre-right of the sheet, so
    the standoff -- the gap a plaque spans -- is a visible distance and not a pixel."""
    a = ax.inset_axes([0.63, 0.63, 0.36, 0.36])
    a.set_facecolor("black")
    for s in a.spines.values():
        s.set_color("#666666")
        s.set_linewidth(0.6)
    a.set_xticks([]); a.set_yticks([])

    # THE BAND IS WIDENED UNTIL THE SECTION EXISTS. A fixed half-width is a fixed fraction of a
    # sphere only while the sphere keeps its size; once the sheet has tripled, or been eaten, the same
    # number catches a handful of nodes and the inset renders as four dots. Widen until ~2% of the
    # live nodes are in it, and stop.
    for _ in range(8):
        sl = np.abs(X[:, 1] - c[1]) < band
        if sl.sum() >= max(24, int(0.02 * X.shape[0])):
            break
        band *= 1.6
    nv = np.zeros(X.shape[0]); cnt = np.zeros(X.shape[0])
    np.add.at(nv, F.reshape(-1), np.repeat(val, 3))
    np.add.at(cnt, F.reshape(-1), 1)
    nv = nv / np.maximum(cnt, 1)
    a.scatter(X[sl][:, 0], X[sl][:, 2], s=6, c=np.clip(nv[sl], lo, hi), cmap=cm, vmin=lo, vmax=hi,
              marker=".", linewidths=0)
    if plq_p is not None and len(plq_p) and x_node is not None:
        ps = np.abs(x_node[:, 1] - c[1]) < band
        if ps.any():
            a.add_collection(LineCollection(
                np.stack([plq_p[ps][:, [0, 2]], x_node[ps][:, [0, 2]]], 1),
                colors=PLQ_C, linewidths=0.7))
            a.scatter(plq_p[ps][:, 0], plq_p[ps][:, 2], s=2.0, c=PLQ_C, marker=".", linewidths=0)
    # THE ZOOM WINDOW IS PLACED ON THE SECTION, not assumed. The first version put it on the right
    # flank at the mean radius, which is where a whole sphere's section is -- but a breached sheet has
    # no material there and the inset came back empty. Centre it on the section points that are
    # actually present, right flank if there are any.
    P = X[sl]
    if P.shape[0]:
        R = P[P[:, 0] > c[0]]
        R = R if R.shape[0] >= 8 else P
        cx, cz = float(R[:, 0].mean()), float(R[:, 2].mean())
    else:
        cx, cz = c[0] + 0.8 * lim, c[2]
    w = frac * lim
    a.set_xlim(cx - w, cx + w)
    a.set_ylim(cz - w, cz + w)
    a.set_aspect("equal")
    return a

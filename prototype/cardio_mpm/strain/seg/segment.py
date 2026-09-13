"""segment -- cells out of the motion, by cutting where the contraction axis turns.

THE ARGUMENT
================================================================================================
Intensity segmentation fails on this sheet because the cells are not delineated -- the raw frames
are a dense phase-contrast texture with no membranes to find. But every cardiomyocyte contracts
along its OWN long axis, and the axis is a property of the cell, not of the image. So:

    a cell is a region over which the contraction axis is constant,
    and a cell boundary is where it turns.

Measured over time, one grid point at a time, before any spatial reasoning: the beat trajectory of
a point is a nearly straight line (median anisotropy l1/l2 = 14), and its direction is that point's
contraction axis. Neighbouring points in the same cell agree; across a boundary the axis steps.

WHY exp(2i.theta) AND NOT theta. An axis has no head or tail, so theta and theta+pi are the same
direction and a gradient of theta would report a huge discontinuity at the wrap. Mapping to
exp(2i.theta) removes the ambiguity: the doubled angle is single-valued and the gradient of the
complex field is a real measure of how fast the axis turns.

WEIGHTED BY HOW MUCH THE AXIS MEANS. A point that barely moves, or moves in a circle, has no
meaningful axis, and letting it vote would put boundaries in the quiet regions. Each point enters
with weight = anisotropy x amplitude, so the axis field is dominated by points that actually have
one.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import beat as B


def axis_field(b, sigma=1.2):
    """The smoothed complex axis field, its boundary strength, and the weights used."""
    ang, amp, aniso = B.axis_and_anisotropy(b)
    w = np.clip((aniso - 1) / 6, 0, 1) * np.clip(amp / np.percentile(amp, 90), 0, 1)
    z = w * np.exp(2j * ang)
    zs = (ndi.gaussian_filter(z.real, sigma) + 1j * ndi.gaussian_filter(z.imag, sigma))
    ws = np.maximum(ndi.gaussian_filter(w, sigma), 1e-9)
    zn = zs / ws                                   # weighted circular mean of the doubled angle
    gy, gx = np.gradient(zn)
    return zn, np.abs(gx) + np.abs(gy), w, ang, amp, aniso


def watershed_cells(bnd, w, h=0.10, min_size=6):
    """Regions enclosed by the ridges of `bnd`.

    Markers are the h-minima of the boundary strength -- the flat interiors -- so the number of
    cells is set by how deep a basin has to be to count, not by a k chosen in advance.
    """
    from skimage.morphology import h_minima, remove_small_objects
    from skimage.segmentation import watershed
    b = bnd / (np.percentile(bnd, 99) + 1e-12)
    seeds = h_minima(ndi.gaussian_filter(b, 0.6), h)
    mk, n = ndi.label(seeds)
    lab = watershed(b, mk, mask=np.ones_like(b, bool))
    # drop slivers back into their largest neighbour rather than leaving holes
    keep = remove_small_objects(lab, min_size=min_size)
    if keep.max() and (keep == 0).any():
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        keep = keep[tuple(idx)]
    lab, _, _ = __import__("skimage.segmentation", fromlist=["relabel_sequential"]) \
        .relabel_sequential(keep)
    return lab, int(lab.max())


def region_stats(lab, ang, w):
    """Per region: size, elongation, its own axis, and whether the two AGREE.

    THE VALIDATION THAT MATTERS. A cardiomyocyte is elongated along the direction it contracts in.
    Nothing in the segmentation knows the region's SHAPE -- it cuts on where the axis turns -- so
    if the regions come out elongated ALONG their own contraction axis, that alignment was not put
    in and is evidence they are cells rather than an arbitrary tiling of a smooth field.
    """
    out = []
    for k in range(1, lab.max() + 1):
        m = lab == k
        n = int(m.sum())
        if n < 4:
            continue
        yy, xx = np.nonzero(m)
        cy, cx = yy.mean(), xx.mean()
        dy, dx = yy - cy, xx - cx
        cxx, cyy, cxy = (dx ** 2).mean(), (dy ** 2).mean(), (dx * dy).mean()
        tr = cxx + cyy
        disc = np.sqrt(max(tr ** 2 / 4 - (cxx * cyy - cxy ** 2), 0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-12)
        shape_ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)          # the region's long axis
        zz = (w[m] * np.exp(2j * ang[m])).sum()
        contract_ang = 0.5 * np.angle(zz)                          # its contraction axis
        d = np.abs(np.angle(np.exp(2j * (shape_ang - contract_ang)))) / 2
        out.append({"label": k, "size": n, "elong": float(np.sqrt(l1 / l2)),
                    "shape_angle": float(shape_ang), "axis_angle": float(contract_ang),
                    "align_rad": float(d)})
    return out

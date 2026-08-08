"""local_affine -- a boundary detector built from what a cell actually IS.

WHY THE AXIS-RIDGE BOUNDARIES WERE WRONG, MEASURED
================================================================================================
Scored on held-out beats, one affine map per region:

    voronoi of the nuclei          FVU 0.113
    my motion watershed            FVU 0.237     twice as bad
    the same, derived from the
      very beats it is scored on   FVU 0.230     still twice as bad

Following the ridges of the axis field produces tortuous regions that straddle mechanical units.
The ridges are stable -- they reproduce at ARI 0.95 -- and they are not cell boundaries. And
Voronoi of DISPLACED nuclei scores 0.113 too, so even the nuclei add nothing: at this region size
any compact tessellation explains a smooth field.

SO DETECT THE BOUNDARY FROM THE DEFINITION INSTEAD OF FROM A PROXY.
A cell is the unit that moves together, which means one affine map covers it. Put a small window on
every node and fit ONE affine map to the whole beat inside it:

    interior of a cell   -> one map fits, residual low
    straddling a border  -> two cells doing two things, no single map fits, residual high

That is a boundary detector whose high values mean the thing we care about, rather than a place
where an angle happened to turn. It also has a matching test: watershed it and score it on
held-out beats with the same affine criterion. If it cannot beat Voronoi there, the field has no
piecewise-affine structure to find at this resolution, and that is an answer too.
"""
from __future__ import annotations

import numpy as np


def residual_map(bb, win=2, stride=1):
    """Per node: unexplained fraction after one affine fit over a (2*win+1)^2 window, whole beat."""
    n_, H, W, _ = bb.shape
    out = np.full((H, W), np.nan, np.float32)
    ys, xs = np.mgrid[-win:win + 1, -win:win + 1]
    des = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float64)
    pinv = np.linalg.pinv(des)                       # the window geometry never changes
    P = des @ pinv                                   # projection onto the affine subspace
    R = np.eye(des.shape[0]) - P
    for i in range(win, H - win, stride):
        for j in range(win, W - win, stride):
            patch = bb[:, i - win:i + win + 1, j - win:j + win + 1, :]      # [n,w,w,2]
            Y = patch.reshape(n_, -1, 2).transpose(1, 0, 2).reshape(des.shape[0], -1)
            Yc = Y - Y.mean(0, keepdims=True)
            res = float((( R @ Y) ** 2).sum())
            tot = float((Yc ** 2).sum())
            out[i, j] = res / max(tot, 1e-12)
    m = np.nanmedian(out)
    return np.where(np.isnan(out), m, out)

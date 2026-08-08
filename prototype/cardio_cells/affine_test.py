"""affine_test -- does this partition predict motion it has never seen?

THE QUESTION THE OTHER TESTS COULD NOT ASK
================================================================================================
Reproducibility says a partition is stable. Nuclei say it has the right density. Neither says it
carves the tissue at its mechanical joints, and that is what a cell IS: the unit that contracts
together.

Inside one cell the displacement is close to a single affine map -- a translation, a rotation, and
a shortening along the myofibril axis. Across a cell boundary two different cells are doing two
different things, and no single affine map covers both. So:

    fit one affine map per region, on beats the partition was NOT derived from,
    and measure what is left over.

A partition that follows real boundaries leaves less. A partition that cuts through the middle of
cells and lumps halves of neighbours together leaves more.

THE COMPARISON IS EVERYTHING, because residual falls with region count no matter what. Every
partition compared here has the SAME NUMBER OF REGIONS, and the geometric control uses the SAME
NUCLEI -- so the only difference is where the borders were put.

  motion        borders from the beat, seeded by nuclei
  voronoi       borders halfway between the same nuclei: geometry, no motion
  shifted       Voronoi of the nuclei displaced as a block -- right statistics, wrong places
  cheating      borders from the very beats being scored: not a fair number, but it bounds how
                much any partition of this K could possibly win by
"""
from __future__ import annotations

import numpy as np

import beat as B
import validate as V
import seeded as SD


def fvu(lab, bb):
    """Fraction of held-out displacement variance NOT explained by one affine map per region."""
    n_, H, W, _ = bb.shape
    yy, xx = np.mgrid[0:H, 0:W]
    U = bb.reshape(n_, H * W, 2)
    L = lab.ravel()
    des = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], 1).astype(np.float64)
    res = 0.0
    tot = float(((U - U.mean(1, keepdims=True)) ** 2).sum())
    for r in np.unique(L):
        m = L == r
        if m.sum() < 4:
            res += float(((U[:, m] - U[:, m].mean(1, keepdims=True)) ** 2).sum())
            continue
        A = des[m]                                   # [p,3]
        Y = U[:, m].transpose(1, 0, 2).reshape(m.sum(), -1)      # [p, n*2]
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        res += float(((Y - A @ coef) ** 2).sum())
    return res / max(tot, 1e-12)


def shifted_seeds(seeds, dy, dx):
    return np.roll(np.roll(seeds, dy, 0), dx, 1)

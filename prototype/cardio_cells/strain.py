"""strain -- the local part of the motion, which is the part a cell owns.

WHY THE DISPLACEMENT AXIS WAS THE WRONG FEATURE
================================================================================================
A point's displacement is what its own cell does PLUS what the whole sheet does around it. In a
confluent monolayer the second term dominates and is smooth over hundreds of microns, so a map of
displacement direction is mostly a map of the tissue's bulk motion, and the domains in it are
tissue-scale, not cell-scale. Segmenting it gave regions whose shape was uncorrelated with their
own contraction axis (44 degrees, and 45 is what a random tiling gives) -- the honest reading of a
failed test.

THE STRAIN IS LOCAL BY CONSTRUCTION. A rigid translation of the whole sheet has zero strain, so
grad(u) removes exactly the part that is not the cell's own doing. A contracting cardiomyocyte
shortens along its long axis, which is a negative principal strain in that direction, and that is
a property of the cell and of nothing else.

Computed here from the smoothed beat displacement rather than from the provided derivative
channels, which are differenced per frame on the 15-pixel grid and are salt-and-pepper noisy at
the scale a cell boundary lives at.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def strain_series(b, sigma=1.0):
    """[n,H,W,3] -- Exx, Eyy, Exy of the symmetric strain, per frame of the mean beat."""
    n = b.shape[0]
    out = np.empty(b.shape[:3] + (3,), np.float32)
    for t in range(n):
        u = ndi.gaussian_filter(b[t, ..., 0], sigma)
        v = ndi.gaussian_filter(b[t, ..., 1], sigma)
        uy, ux = np.gradient(u)
        vy, vx = np.gradient(v)
        out[t, ..., 0] = ux
        out[t, ..., 1] = vy
        out[t, ..., 2] = 0.5 * (uy + vx)
    return out


def contraction_axis(E):
    """Per point: the axis it CONTRACTS along, how much, and how anisotropic that is.

    Taken at each point's own moment of peak strain rather than at a single global frame, because
    the sheet does not contract all at once -- the activation sweeps across it.
    """
    mag = np.sqrt(E[..., 0] ** 2 + E[..., 1] ** 2 + 2 * E[..., 2] ** 2)     # Frobenius
    tpk = np.argmax(mag, 0)                                                  # [H,W]
    H, W = tpk.shape
    ii, jj = np.mgrid[0:H, 0:W]
    Exx, Eyy, Exy = (E[tpk, ii, jj, k] for k in range(3))
    tr = Exx + Eyy
    disc = np.sqrt(np.maximum((Exx - Eyy) ** 2 / 4 + Exy ** 2, 0))
    e1, e2 = tr / 2 + disc, tr / 2 - disc            # e2 is the most NEGATIVE = contraction
    ang = 0.5 * np.arctan2(2 * Exy, Exx - Eyy)       # axis of e1; contraction axis is +pi/2
    contract = ang + np.pi / 2
    aniso = (e1 - e2) / (np.abs(e1) + np.abs(e2) + 1e-12)
    return contract, np.abs(e2), aniso, tpk, mag.max(0)

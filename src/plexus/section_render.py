"""The cross-section drawn as a histological section (exp 11, 2026-09-28): the outlines of the tissue's cells where
the section plane cuts them, not the points that happen to lie in a slab.

    prism_cloud      points filling each apico-basal cell's prism (pos + h sep over its fan triangles), by cell
    section_labels   a label image of the plane: each epithelial pixel the cell whose prism points (in the slab)
                     are densest there; each other pixel the interior cell whose Gaussian particle density, over
                     ALL its particles, passes a level on the plane (the outline follows the particle cloud)
    borders          the pixels where the label changes -- the outlines

Used by `live_movie` (`plotting.cross_section.style: outlines`) and by the exp 11 offline renderers.
"""
from __future__ import annotations

import numpy as np


def prism_cloud(P, S, rings, nF, per_tri=16, heights=5, seed=0):
    rng = np.random.default_rng(seed)
    A, Bv, C, SA, SB, SC, L = [], [], [], [], [], [], []
    for f in range(nF):
        r = rings[f]
        if r is None or len(r) < 3:
            continue
        r = np.asarray(r); cx = P[r].mean(0); cs = S[r].mean(0); r2 = np.roll(r, -1)
        A.append(np.repeat(cx[None], len(r), 0)); Bv.append(P[r]); C.append(P[r2])
        SA.append(np.repeat(cs[None], len(r), 0)); SB.append(S[r]); SC.append(S[r2]); L.append(np.full(len(r), f))
    if not A:
        return np.zeros((0, 3)), np.zeros(0, int)
    A, Bv, C, SA, SB, SC, L = map(np.concatenate, (A, Bv, C, SA, SB, SC, L))
    u = rng.random((len(A), per_tri, 2)); m = u.sum(-1) > 1; u[m] = 1 - u[m]
    xy = A[:, None] + u[..., :1] * (Bv - A)[:, None] + u[..., 1:] * (C - A)[:, None]
    sp = SA[:, None] + u[..., :1] * (SB - SA)[:, None] + u[..., 1:] * (SC - SA)[:, None]
    hs = np.linspace(-0.45, 0.45, heights)
    pts = (xy[:, :, None] + hs[None, None, :, None] * sp[:, :, None]).reshape(-1, 3)
    return pts, np.repeat(L, per_tri * heights)


def section_labels(pts, lab, Q, qc, ax, a, b, y0, xr, yr, npx, th=0.30, sig=0.22):
    """Label image [npx, npx] (row = axis b, col = axis a) over the window xr x yr at plane coordinate y0 on axis
    `ax`. Epithelial labels are the cell index (>= 0), interior labels 100000 + particle-cell id, empty -1."""
    from scipy import ndimage
    px = (xr[1] - xr[0]) / npx
    out = np.full((npx, npx), -1, np.int64)
    inS = np.abs(pts[:, ax] - y0) < th
    if inS.any():
        uv = np.stack([(pts[inS, a] - xr[0]) / px, (pts[inS, b] - yr[0]) / px], 1).astype(int)
        L = lab[inS]; ok = (uv >= 0).all(1) & (uv < npx).all(1); uv, L = uv[ok], L[ok]
        best = np.zeros((npx, npx))
        for k in np.unique(L):
            m = L == k; g = np.zeros((npx, npx)); np.add.at(g, (uv[m, 1], uv[m, 0]), 1.0)
            ys, xs = np.nonzero(g)
            y_0, y_1, x_0, x_1 = max(ys.min() - 6, 0), min(ys.max() + 7, npx), max(xs.min() - 6, 0), min(xs.max() + 7, npx)
            sub = ndimage.gaussian_filter(g[y_0:y_1, x_0:x_1], 1.3)
            bs = best[y_0:y_1, x_0:x_1]; win = (sub > bs) & (sub > 0.05); bs[win] = sub[win]; out[y_0:y_1, x_0:x_1][win] = k
    if len(Q):
        gx = xr[0] + (np.arange(npx) + 0.5) * px; gy = yr[0] + (np.arange(npx) + 0.5) * px
        best = np.zeros((npx, npx))
        for k in np.unique(qc):
            Xk = Q[qc == k]
            if len(Xk) < 8:
                continue
            dy = Xk[:, ax] - y0
            if np.abs(dy).min() > 2.5 * sig:
                continue
            ix = np.nonzero((gx > Xk[:, a].min() - 3 * sig) & (gx < Xk[:, a].max() + 3 * sig))[0]
            iy = np.nonzero((gy > Xk[:, b].min() - 3 * sig) & (gy < Xk[:, b].max() + 3 * sig))[0]
            if not len(ix) or not len(iy):
                continue
            wx = np.exp(-((gx[ix][None, :] - Xk[:, a:a + 1]) ** 2) / (2 * sig ** 2))
            wy = np.exp(-((gy[iy][None, :] - Xk[:, b:b + 1]) ** 2) / (2 * sig ** 2))
            wz = np.exp(-(dy ** 2) / (2 * sig ** 2))
            rho = np.einsum("n,ny,nx->yx", wz, wy, wx)
            d2 = ((Xk[:, None, :] - Xk[None, :, :]) ** 2).sum(-1)
            level = 0.33 * np.median(np.exp(-d2 / (2 * sig ** 2)).sum(1))
            so = out[np.ix_(iy, ix)]; sb = best[np.ix_(iy, ix)]
            win = (rho > level) & (rho > sb) & ~((so >= 0) & (so < 100000))
            so[win] = 100000 + int(k); sb[win] = rho[win]; out[np.ix_(iy, ix)] = so; best[np.ix_(iy, ix)] = sb
    return out


def borders(labels):
    bd = np.zeros(labels.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        s = np.roll(labels, (dy, dx), (0, 1)); bd |= (labels != s) & ((labels >= 0) | (s >= 0))
    return bd

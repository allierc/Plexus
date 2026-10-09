"""exp21 mpm_epithelium -- the rulers: does a plane of cells made of MPM material points hold together as one
epithelium of constant height, do its cells move among their neighbours as the salivary bud's surface cells do
(Wang et al. 2021, their own nucleus tracks), and do opposed streams, divisions and a contact-borne signal pass
through it without wrecking it?

    exp21.sheet     integrity, monolayer, height, coverage, connectivity, packing -- per sampled row and summarised
    exp21.motion    exp11's `cell_motion` (one estimator with Wang's tracks) in the PLANE, plus neighbour retention
                    at 1 h and 2 h by `neighbour_retention`, the estimator `tools/exp21_wang_neighbours.py` runs on
                    the gland's surface nuclei
    exp21.shear     the counter-flow arm: cells of the two opposed halves passing each other across the shear line,
                    and the sense in which the cells in the band turn
    exp21.division  divisions: count, the angle of each daughter pair's axis to the sheet's plane, daughters kept in
                    the monolayer
    exp21.wave      a contact-borne excitation: the fraction of cells reached, the front's speed in cell diameters
                    per hour, and the fraction reached beyond a declared cut

THE CELLS. A cell is a parent of the MPM point set (`cell_set`, `point_set`); its centre is the parent's `pos`
(`aggregate_centroid`), its body its points, grouped by the points' parent-id block (`pid`) when the run records
one, else by the contiguous block of `per_parent` slots. The sheet's PLANE is z (`plane_axis` 2): its normal is z,
heights are read along z, every in-plane quantity on (x, y).

THE INTERIOR. A free disc has a rim that behaves unlike its bulk (T-07, T-27). Every in-plane statistic of a run is
read on the cells whose centre, at the row read, lies more than `rim` cell spacings inside the convex hull of the
sheet's centres; `rim` is 1.5 by default. A whole-sheet number (connectivity, coverage of the hull) says so.

ONE ESTIMATOR FOR THE GLAND AND THE MODEL. `neighbour_retention` takes two position arrays of the same cells at two
times and returns the fraction of the pairs that were neighbours (closer than `r_in` x the median nearest-neighbour
distance d0) still within `r_out` x d0 after the lag. It is scale-free (d0 is each population's own), so the gland's
9.9 um nuclei and the model's 1.0-unit cells are compared through identical arithmetic.
"""
from __future__ import annotations

import math

import numpy as np

from .common import finite, register_run, spec_op

try:                                                    # the gland's own motion estimator, ONE copy
    from .exp11 import cell_motion as _cell_motion_3d, prw_fit, _row_frames
except Exception:                                       # noqa: BLE001
    _cell_motion_3d = prw_fit = _row_frames = None


# ============================================================================ pure estimators (tested)
def neighbour_retention(X0, X1, sel=None, r_in=1.25, r_out=1.5, d0=None):
    """Of the pairs closer than r_in d0 in X0, the fraction still closer than r_out d0 in X1.

    X0, X1  [n, D] positions of the SAME n cells at two times (row-aligned)
    sel     optional [n] bool: only pairs whose two cells are both selected (the surface, the interior)
    d0      the median nearest-neighbour distance of X0 (computed over the selected cells when not given)
    Returns (kept, n_pairs, d0); kept is None when there are fewer than 10 pairs."""
    from scipy.spatial import cKDTree
    X0 = np.asarray(X0, float); X1 = np.asarray(X1, float)
    n = X0.shape[0]
    if sel is None:
        sel = np.ones(n, bool)
    sel = np.asarray(sel, bool)
    idx = np.flatnonzero(sel)
    if idx.size < 3:
        return None, 0, None
    if d0 is None:
        d0 = float(np.median(cKDTree(X0[idx]).query(X0[idx], k=2)[0][:, 1]))
    pairs = cKDTree(X0[idx]).query_pairs(r_in * d0, output_type="ndarray")
    if len(pairs) < 10:
        return None, int(len(pairs)), d0
    a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
    dl = np.linalg.norm(X1[a] - X1[b], axis=1)
    return float(np.mean(dl < r_out * d0)), int(len(pairs)), d0


def interior_mask(XY, rim_spacings=1.5, d0=None):
    """[n] bool: centres more than rim_spacings x d0 inside the convex hull of all centres (2D)."""
    from scipy.spatial import ConvexHull, cKDTree
    XY = np.asarray(XY, float)[:, :2]
    if XY.shape[0] < 4:
        return np.zeros(XY.shape[0], bool)
    if d0 is None:
        d0 = float(np.median(cKDTree(XY).query(XY, k=2)[0][:, 1]))
    h = ConvexHull(XY)
    # signed distance inside every hull facet: n . x + c <= 0 inside; the smallest is the distance to the hull
    eq = h.equations                                     # [m, 3]: normal (2), offset
    depth = -(XY @ eq[:, :2].T + eq[:, 2])               # > 0 inside
    return depth.min(1) > rim_spacings * d0


def cell_heights(P, par, n_cells, axis=2, lo=5.0, hi=95.0):
    """Per cell, the extent of its points along `axis` between the lo and hi percentiles: [n_cells] (nan if empty)."""
    P = np.asarray(P, float); par = np.asarray(par, np.int64)
    h = np.full(n_cells, np.nan)
    order = np.argsort(par, kind="stable")
    ps = par[order]; zs = P[order, axis]
    cut = np.flatnonzero(np.diff(ps)) + 1
    for grp_z, c in zip(np.split(zs, cut), np.split(ps, cut)):
        if c.size and 0 <= c[0] < n_cells and grp_z.size >= 4:
            h[c[0]] = np.percentile(grp_z, hi) - np.percentile(grp_z, lo)
    return h


def contact_components(P, par, live_cells, contact):
    """Connected components of the CONTACT graph: two cells touch when some point of one lies within `contact`
    of some point of the other. Returns (largest component's share of the live cells, number of components,
    the [m, 2] touching pairs as cell ids)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = np.asarray(P, float); par = np.asarray(par, np.int64)
    live = np.asarray(sorted(set(int(c) for c in live_cells)), np.int64)
    if live.size == 0:
        return None, 0, np.zeros((0, 2), np.int64)
    keep = np.isin(par, live)
    Q, q = P[keep], par[keep]
    pr = cKDTree(Q).query_pairs(contact, output_type="ndarray")
    if len(pr):
        a, b = q[pr[:, 0]], q[pr[:, 1]]
        m = a != b
        e = np.unique(np.sort(np.stack([a[m], b[m]], 1), axis=1), axis=0)
    else:
        e = np.zeros((0, 2), np.int64)
    rank = {int(c): i for i, c in enumerate(live)}
    if len(e):
        ii = np.array([rank[int(x)] for x in e[:, 0]]); jj = np.array([rank[int(x)] for x in e[:, 1]])
        A = coo_matrix((np.ones(len(e)), (ii, jj)), shape=(live.size, live.size))
        nc, lab = connected_components(A, directed=False)
    else:
        nc, lab = live.size, np.arange(live.size)
    big = np.bincount(lab).max() / live.size
    return float(big), int(nc), e


def coverage(P, XY_centres, cell_size, res=0.25):
    """Fraction of the hull of the centres (shrunk by one cell) whose top-view raster pixels hold a point."""
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    XY = np.asarray(XY_centres, float)[:, :2]
    if XY.shape[0] < 4:
        return None
    h = ConvexHull(XY)
    poly = XY[h.vertices]
    c = poly.mean(0)
    poly = c + (poly - c) * np.clip(1.0 - cell_size / np.maximum(np.linalg.norm(poly - c, axis=1, keepdims=True), 1e-9),
                                    0.0, 1.0)
    px = res * cell_size
    lo, hi = poly.min(0), poly.max(0)
    gx = np.arange(lo[0], hi[0], px) + 0.5 * px
    gy = np.arange(lo[1], hi[1], px) + 0.5 * px
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    inside = Path(poly).contains_points(G)
    if not inside.any():
        return None
    Q = np.asarray(P, float)[:, :2]
    ix = np.floor((Q[:, 0] - lo[0]) / px).astype(np.int64); iy = np.floor((Q[:, 1] - lo[1]) / px).astype(np.int64)
    ok = (ix >= 0) & (ix < gx.size) & (iy >= 0) & (iy < gy.size)
    occ = np.zeros((gx.size, gy.size), bool)
    occ[ix[ok], iy[ok]] = True
    # a pixel counts as covered when it or one of its 8 neighbours holds a point: the points are a SAMPLE of
    # the material, ~0.3 of a cell apart, so a single empty pixel between them is sampling, not a hole
    from scipy.ndimage import binary_dilation
    occ = binary_dilation(occ, structure=np.ones((3, 3), bool))
    return float(occ.reshape(-1)[inside].mean())


def voronoi_sides(XY, sel):
    """Neighbour counts (Delaunay degree) of the selected centres, 2D: the polygon class of each cell."""
    from scipy.spatial import Delaunay
    XY = np.asarray(XY, float)[:, :2]
    tri = Delaunay(XY)
    nb = [set() for _ in range(XY.shape[0])]
    for s in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    nb[s[i]].add(s[j])
    return np.array([len(nb[i]) for i in np.flatnonzero(sel)], int)


def voronoi_shape_index(XY, sel):
    """Perimeter / sqrt(area) of the Voronoi polygon of each selected centre (2D; unbounded or degenerate cells
    skipped). A regular hexagon reads 3.722; Bi et al. 2015 put the solid-fluid line of a cell sheet at 3.813."""
    from scipy.spatial import Voronoi
    XY = np.asarray(XY, float)[:, :2]
    vor = Voronoi(XY)
    out = []
    for i in np.flatnonzero(sel):
        reg = vor.regions[vor.point_region[i]]
        if not reg or -1 in reg:
            continue
        P = vor.vertices[reg]
        c = P.mean(0)
        P = P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
        Q = np.roll(P, -1, axis=0)
        area = 0.5 * abs(np.sum(P[:, 0] * Q[:, 1] - Q[:, 0] * P[:, 1]))
        per = np.sum(np.linalg.norm(Q - P, axis=1))
        if area > 0:
            out.append(per / math.sqrt(area))
    return np.asarray(out, float)


def velocity_correlation_length(X, V, d0, r_max=8.0, nbins=16):
    """The distance, in spacings d0, at which the velocities of two cells stop being alike: C(r) = <v_i . v_j> /
    <|v|^2> over the pairs at distance r (V with its mean removed), the first r where C falls below 1/e (linear
    between bins); r_max when it never does. Park et al. 2015 (Fig 1d, 2g-h): cooperative packs of 7-26 cells."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float)[:, :2]; V = np.asarray(V, float)[:, :2]
    V = V - V.mean(0)
    v2 = float((V * V).sum(1).mean())
    if v2 <= 0 or X.shape[0] < 10:
        return None
    pr = cKDTree(X).query_pairs(r_max * d0, output_type="ndarray")
    if len(pr) == 0:
        return None
    r = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1) / d0
    c = (V[pr[:, 0]] * V[pr[:, 1]]).sum(1) / v2
    edges = np.linspace(0.5, r_max, nbins + 1)
    mid, C = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 5:
            mid.append(0.5 * (a + b)); C.append(float(c[m].mean()))
    for k in range(len(C)):
        if C[k] < 1.0 / np.e:
            if k == 0:
                return float(mid[0])
            x0, x1, y0, y1 = mid[k - 1], mid[k], C[k - 1], C[k]
            return float(x0 + (1.0 / np.e - y0) * (x1 - x0) / (y1 - y0))
    return float(r_max)


def affine_residual(X0, X1):
    """X1 - X0 with the best affine flow of the whole set removed (least squares d = A x + b): what each cell moved
    RELATIVE to the tissue's own stretch, shear, rotation and drift."""
    X0 = np.asarray(X0, float); D = np.asarray(X1, float) - X0
    A = np.c_[X0, np.ones(len(X0))]
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    return D - A @ coef


def affine_msd(X0, X1, N=None):
    """The mean squared displacement X0 -> X1 of a set of cells with the set's best AFFINE flow removed
    (`affine_residual`) and, given unit normals N [n, 3], the part along each cell's normal removed too: what the
    cells moved AMONG each other, in their own sheet. The salivary bud grows and turns, so a surface nucleus's raw
    displacement carries the tissue's stretch and rotation and its motion off the surface; the model's corral has
    neither. One function for both sides (exp 21 Finding 29: the gland reads 40 um^2 at 1 h this way, 60 raw)."""
    D = affine_residual(X0, X1)
    if N is not None:
        D = D - (D * N).sum(1)[:, None] * N
    return float((D * D).sum(1).mean())


def neighbour_correlation(X, V, d0, r_lo=0.75, r_hi=1.25):
    """<v_i . v_j> / <|v|^2> over the pairs between r_lo and r_hi spacings apart: how alike two NEIGHBOURS' motions
    are (0: independent walkers; 1: carried as one). The gland's surface cells read ~0.1 (exp 21 literature audit,
    2026-10-08: Wang 2021 tracks, 30-min displacements, tissue flow removed)."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float); V = np.asarray(V, float)
    v2 = float((V * V).sum(1).mean())
    if v2 <= 0 or len(X) < 10:
        return None
    pr = cKDTree(X).query_pairs(r_hi * d0, output_type="ndarray")
    if len(pr) == 0:
        return None
    r = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1)
    pr = pr[r >= r_lo * d0]
    if len(pr) < 10:
        return None
    return float((V[pr[:, 0]] * V[pr[:, 1]]).sum(1).mean() / v2)


def neighbour_correlation_lt(X, V, d0, r_lo=0.75, r_hi=1.25):
    """(C_L, C_T): `neighbour_correlation` split into the displacement component ALONG the line joining two
    neighbours (L) and ACROSS it (T), each <a_i . a_j> / <a^2> over the pairs r_lo-r_hi spacings apart. A push
    handed on through a contact correlates L; neighbours carried together sideways -- a sheet that resists shear --
    correlate T. Henkes 2020 (Nat Commun 11:1405): the longitudinal modes' correlation length grows with the
    sheet's bulk + shear modulus, the transverse ones' with its shear modulus only. The gland's surface cells: C_L
    0.20, C_T 0.07 (exp 21 Finding 22). V must lie in the plane (the model's 2D displacements; the gland's after
    projecting out the surface normal)."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, float); V = np.asarray(V, float)
    if len(X) < 10:
        return None, None
    pr = cKDTree(X).query_pairs(r_hi * d0, output_type="ndarray")
    if len(pr) == 0:
        return None, None
    R = X[pr[:, 1]] - X[pr[:, 0]]
    r = np.linalg.norm(R, axis=1)
    k = r >= r_lo * d0
    pr, R, r = pr[k], R[k], r[k]
    if len(pr) < 10:
        return None, None
    u = R / r[:, None]
    ai, aj = (V[pr[:, 0]] * u).sum(1), (V[pr[:, 1]] * u).sum(1)
    ti, tj = V[pr[:, 0]] - ai[:, None] * u, V[pr[:, 1]] - aj[:, None] * u
    sl = 0.5 * float((ai ** 2 + aj ** 2).mean()); st = 0.5 * float(((ti ** 2).sum(1) + (tj ** 2).sum(1)).mean())
    cl = float((ai * aj).mean()) / sl if sl > 0 else None
    ct = float((ti * tj).sum(1).mean()) / st if st > 0 else None
    return cl, ct


# ============================================================================ reading a run
def _units(T):
    g = T.spec.get("general") or {}
    u = g.get("units") or {}
    return float(u.get("length_um", 1.0)), float(u.get("time_s", 60.0)) * float(g.get("dt", 1.0))


def _arr(T, key, t):
    return np.asarray(T.z[key][t], float)


def _cells_points(T, t, cell_set, point_set, pid="pid"):
    """(centres [nc, 3] over ALL slots, live cell mask [nc], points [np, 3] live, their parent [np])."""
    z = T.z
    C = _arr(T, f"{cell_set}__pos", t)
    co = (_arr(T, f"{cell_set}__occ", t).reshape(-1) > 0.5) if f"{cell_set}__occ" in z.files else np.ones(len(C), bool)
    P = _arr(T, f"{point_set}__pos", t)
    po = (_arr(T, f"{point_set}__occ", t).reshape(-1) > 0.5) if f"{point_set}__occ" in z.files else np.ones(len(P), bool)
    if f"{point_set}__{pid}" in z.files:
        par = _arr(T, f"{point_set}__{pid}", t).reshape(-1).round().astype(np.int64)
    else:
        k = P.shape[0] // max(C.shape[0], 1)
        par = np.repeat(np.arange(C.shape[0]), k)[: P.shape[0]]
    keep = po & np.isfinite(P).all(1)
    return C, co & np.isfinite(C).all(1), P[keep], par[keep]


def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


# ============================================================================ exp21.sheet
def sheet(T, cell_set="cell", point_set="pt", every=10, contact=None, rim=1.5, axis=2, **_):
    """Is it ONE epithelium of constant height, every cell one body? Per sampled row, and summarised:

      n_cells                 live cells (first, last)
      integrity_max           largest point-to-own-centre distance over the cell's row-0 rms radius, max over cells
                              and rows (~1.6 for an intact column, > 3 a torn or smeared cell)
      torn_frac               the share of cells that read above 3 at ANY sampled row -- the cap's reading: a crawling
                              cell elongates and one stray point at one row reads 3.1 (round 3: 1-2 cells of 300,
                              the 99th percentile over cells ~2.6), a torn sheet reads many
      monolayer_out_max       fraction of live cells whose centre sits more than half the median height off the
                              sheet's median plane, max over rows (0 for a monolayer; a cell climbing on top reads)
      height_ratio            median cell height at the last row over the first (heights: 5-95 % extent of a cell's
                              points along z)
      height_cv_max           cross-cell CV of height, max over rows
      connected_min           the largest contact component's share of the live cells, min over rows (1 = one sheet)
      coverage_min            the share of the sheet's hull (shrunk by one cell) the points cover from above, min
                              over rows (1 = confluent, no holes)
      sides_mean, hex_frac    neighbour count of interior cells (Delaunay of the centres), last row
      shape_index             median perimeter / sqrt(area) of the interior cells' Voronoi polygons, last row
                              (Bi et al. 2015/2016: a sheet is fluid above 3.81, solid below)
      nonfinite               any non-finite point position in any sampled row (True is a wreck)"""
    # TWO CELLS TOUCH within the run's own contact range: the pair_potential's sigma (+ 0.01), so a run with a wider
    # contact (round 12 sig036: sigma 0.36 against the old fixed 0.25) does not read as disconnected
    if contact is None:
        pp = spec_op(T, "pair_potential")
        contact = float(pp.get("sigma", 0.24)) + 0.01 if pp else 0.25
    rows = _rows(T, every)
    ser = {k: [] for k in ("row", "n_cells", "integrity", "monolayer_out", "height_med", "height_cv", "connected",
                           "coverage")}
    r0 = None
    nonfinite = False
    sides = None
    torn_cells = set()
    for t in rows:
        C, co, P, par = _cells_points(T, t, cell_set, point_set)
        nonfinite |= not np.isfinite(_arr(T, f"{point_set}__pos", t)).all()
        nc = C.shape[0]
        live = np.flatnonzero(co)
        cen = np.zeros((nc, 3)); cnt = np.bincount(par, minlength=nc).astype(float)
        np.add.at(cen, par, P); cen /= np.maximum(cnt, 1)[:, None]
        d = np.linalg.norm(P - cen[par], axis=1)
        rms = np.sqrt(np.bincount(par, d ** 2, minlength=nc) / np.maximum(cnt, 1))
        far = np.zeros(nc); np.maximum.at(far, par, d)
        if r0 is None:
            r0 = np.where(rms > 0, rms, np.nan)
        integ = np.nanmax(far[live] / r0[live]) if live.size else np.nan
        torn_cells |= set(int(c) for c in live[(far[live] / r0[live]) > 3.0])
        h = cell_heights(P, par, nc, axis=axis)
        hm = float(np.nanmedian(h[live])) if live.size else np.nan
        zc = cen[live, axis]
        out_frac = float(np.mean(np.abs(zc - np.median(zc)) > 0.5 * hm)) if live.size else np.nan
        big, _nc, _e = contact_components(P, par, live, contact)
        from scipy.spatial import cKDTree
        d0 = float(np.median(cKDTree(cen[live, :2]).query(cen[live, :2], k=2)[0][:, 1])) if live.size > 3 else 1.0
        cov = coverage(P, cen[live], d0)
        ser["row"].append(int(t)); ser["n_cells"].append(int(live.size)); ser["integrity"].append(finite(integ))
        ser["monolayer_out"].append(finite(out_frac)); ser["height_med"].append(finite(hm))
        ser["height_cv"].append(finite(np.nanstd(h[live]) / max(hm, 1e-12)) if live.size else None)
        ser["connected"].append(finite(big)); ser["coverage"].append(finite(cov))
        if t == rows[-1] and live.size > 10:
            inner = interior_mask(cen[live, :2], rim, d0)
            if inner.sum() >= 5:
                s = voronoi_sides(cen[live, :2], inner)
                q = voronoi_shape_index(cen[live, :2], inner)
                sides = (float(s.mean()), float(np.mean(s == 6)), float(np.median(q)) if q.size else np.nan)

    def _mx(k):
        v = [x for x in ser[k] if x is not None]
        return finite(max(v)) if v else None

    def _mn(k):
        v = [x for x in ser[k] if x is not None]
        return finite(min(v)) if v else None
    hmed = [x for x in ser["height_med"] if x is not None]
    return {"available": True, "series": ser,
            "n_cells_first": ser["n_cells"][0], "n_cells_last": ser["n_cells"][-1],
            "integrity_max": _mx("integrity"), "monolayer_out_max": _mx("monolayer_out"),
            "torn_frac": finite(len(torn_cells) / max(ser["n_cells"][0], 1)),
            "height_first": finite(hmed[0]) if hmed else None, "height_last": finite(hmed[-1]) if hmed else None,
            "height_ratio": finite(hmed[-1] / hmed[0]) if len(hmed) > 1 and hmed[0] else None,
            "height_dev": finite(abs(hmed[-1] / hmed[0] - 1.0)) if len(hmed) > 1 and hmed[0] else None,
            "height_cv_max": _mx("height_cv"), "connected_min": _mn("connected"), "coverage_min": _mn("coverage"),
            "sides_mean": finite(sides[0]) if sides else None, "hex_frac": finite(sides[1]) if sides else None,
            "shape_index": finite(sides[2]) if sides else None,
            "nonfinite": bool(nonfinite)}


register_run("exp21.sheet", sheet, None, "one epithelium of constant height: integrity, monolayer, height, cover")


# ============================================================================ exp21.motion
def motion(T, cell_set="cell", point_set="pt", rim=1.5, lags_h=(1.0, 2.0), max_lag=24, **_):
    """How the cells move in the plane, in the terms of Wang 2021's tracks (exp11's `cell_motion`, one estimator),
    and whether they move AMONG their neighbours:

      speed_um_h_median, prw_v_um_h, prw_P_h, msd_um2, track_speed_cv, turn_deg_median   exp11.cell_motion's, on
                              the interior cells' in-plane centres (z set to 0, the population's drift removed)
      msd_1h_um2              the MSD at a lag of 1 h, interpolated on the row lags
      kept_1h, kept_2h        `neighbour_retention` at 1 h and 2 h lags, averaged over every start row, interior
                              cells at the start (Wang's surface nuclei: tools/exp21_wang_neighbours.py)
      exchange_per_cell_h     (1 - kept_1h) x mean neighbours per cell: neighbours lost per cell per hour
      prw_P_h_1h              the persistence fitted over the first hour of lags (the 2-h fit floors on a caged walk)
      msd_ratio_3h_1h         MSD/lag at 3 h over MSD/lag at 1 h: < 1 caged, > 1 still spreading (gland 1.18)
      corr_1sp                how alike two neighbours' 30-min displacements are, the tissue's affine flow removed
                              (0 independent, 1 carried together; gland ~0.1)
      corr_L_1sp, corr_T_1sp  the same split along / across the line joining the two (gland 0.20 / 0.07): a push
                              passed on (L) or neighbours carried together sideways (T)
      msd_1h_aff_um2, prw_v_aff_um_h, prw_P_aff_h   the MSD at 1 h with the set's affine flow removed per window
                              (`affine_msd`) and the persistent walk (with noise offset) fitted to its curve up to
                              1 h: the cells' own motion among each other, the gland's read the same way
                              (40 um^2, 11.1 um/h, 0.195 h; Finding 29)
      vcorr_len_cells         how far, in cell spacings, the cells' 30-min displacements stay alike (C(r) = 1/e):
                              ~1 for independent walkers, several for a crowd flowing in packs (Park 2015: 7-26 cells)"""
    um, s_per_frame = _units(T)
    n = T.n_rows()
    fr = np.asarray(_row_frames(T) if _row_frames else list(range(n)), float)
    row_h = float(np.median(np.diff(fr))) * s_per_frame / 3600.0 if n > 1 else np.nan
    C = np.array([_arr(T, f"{cell_set}__pos", t) for t in range(n)])
    co = np.array([(_arr(T, f"{cell_set}__occ", t).reshape(-1) > 0.5) if f"{cell_set}__occ" in T.z.files
                   else np.ones(C.shape[1], bool) for t in range(n)])
    alive = co.all(0) & np.isfinite(C).all(axis=(0, 2))       # cells live through the whole run
    from scipy.spatial import cKDTree
    X = C[:, alive, :2]
    d0 = float(np.median(cKDTree(X[0]).query(X[0], k=2)[0][:, 1]))
    inner0 = interior_mask(X[0], rim, d0)
    out = {"available": True, "row_h": finite(row_h), "n_cells": int(alive.sum()), "n_interior": int(inner0.sum()),
           "nn_um": finite(d0 * um)}
    if inner0.sum() < 5 or n < 3:
        return out
    Xi = X[:, inner0] * um
    steps = np.diff(Xi, axis=0)
    drift = np.median(steps, axis=1, keepdims=True)
    sc = steps - drift
    spd = np.linalg.norm(sc, axis=2) / row_h
    a, b = sc[1:], sc[:-1]
    cosang = (a * b).sum(2) / np.maximum(np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2), 1e-12)
    track_v = spd.mean(0)
    Xc = Xi - np.vstack([np.zeros((1, 1, 2)), np.cumsum(drift, axis=0)])
    L = min(max_lag, n - 1)
    msd = np.array([((Xc[k:] - Xc[:-k]) ** 2).sum(2).mean() for k in range(1, L + 1)])
    lag = np.arange(1, L + 1) * row_h
    out.update({"speed_um_h_median": finite(np.median(spd)),
                "speed_um_h_iqr": [finite(np.percentile(spd, 25)), finite(np.percentile(spd, 75))],
                "track_speed_cv": finite(track_v.std() / max(track_v.mean(), 1e-12)),
                "turn_deg_median": finite(np.median(np.degrees(np.arccos(np.clip(cosang, -1, 1))))),
                "msd_um2": [finite(v) for v in msd[:12]],
                "msd_1h_um2": finite(np.interp(1.0, lag, msd)) if lag[-1] >= 1.0 else None})
    if prw_fit is not None:
        f = prw_fit(lag, msd)
        out.update({"prw_v_um_h": finite(f["v"]), "prw_P_h": finite(f["P"]), "prw_r2": finite(f["r2"])})
    nb = cKDTree(X[0][inner0]).query_pairs(1.25 * d0)
    mean_nb = 2.0 * len(nb) / max(int(inner0.sum()), 1)
    for lh in lags_h:
        k = int(round(lh / row_h)) if row_h and np.isfinite(row_h) else 0
        if k < 1 or k >= n:
            out[f"kept_{lh:g}h"] = None
            continue
        vals = []
        for t0 in range(0, n - k, max(1, k // 2)):
            inner = interior_mask(X[t0], rim, d0)
            kept, npairs, _ = neighbour_retention(X[t0], X[t0 + k], sel=inner, d0=d0)
            if kept is not None:
                vals.append(kept)
        out[f"kept_{lh:g}h"] = finite(np.mean(vals)) if vals else None
    k1 = out.get("kept_1h")
    out["exchange_per_cell_h"] = finite((1.0 - k1) * mean_nb) if k1 is not None else None
    # THE CAGE AND THE 1-H PERSISTENCE (literature audit, 2026-10-08): a caged walk's MSD / lag FALLS after its
    # persistence time while the gland's keeps rising (1.18 from 1 h to 3 h); and a PRW fit over 2 h of a caged MSD
    # collapses to its floor, so the persistence is also fitted over the first hour, as the gland's is
    if lag[-1] >= 1.0:
        def _per_h(th):
            return np.interp(th, lag, msd) / th
        Lx = min(int(round(3.0 / row_h)), n - 1) if row_h and np.isfinite(row_h) else 0
        if Lx >= 2:
            m3 = np.array([((Xc[k:] - Xc[:-k]) ** 2).sum(2).mean() for k in range(1, Lx + 1)])
            l3 = np.arange(1, Lx + 1) * row_h
            out["msd_ratio_3h_1h"] = finite((np.interp(min(3.0, l3[-1]), l3, m3) / min(3.0, l3[-1]))
                                            / (np.interp(1.0, l3, m3) / 1.0))
        k1h = int(np.searchsorted(lag, 1.0 + 1e-9))
        if prw_fit is not None and k1h >= 3:
            f1 = prw_fit(lag[:k1h], msd[:k1h])
            out["prw_P_h_1h"] = finite(f1["P"]); out["prw_v_um_h_1h"] = finite(f1["v"])
    # THE CELLS' OWN MOTION, LIKE FOR LIKE WITH THE GLAND'S (Finding 29): the MSD curve over lags up to 1 h with the
    # interior set's affine flow removed per start row (`affine_msd`, the gland's reader projects on its surface
    # too), and the persistent walk fitted to it with a noise offset (exp11.prw_fit, the gland's estimator)
    if row_h and np.isfinite(row_h):
        kmax = min(int(round(1.0 / row_h)), n - 1)
        if kmax >= 3:
            starts = range(0, n - kmax, max(1, kmax // 4))
            cur = np.array([np.mean([affine_msd(Xi[t0], Xi[t0 + L]) for t0 in starts]) for L in range(1, kmax + 1)])
            lh = np.arange(1, kmax + 1) * row_h
            out["msd_1h_aff_um2"] = finite(cur[-1])
            if prw_fit is not None:
                fa = prw_fit(lh, cur)
                out["prw_v_aff_um_h"] = finite(fa["v"]); out["prw_P_aff_h"] = finite(fa["P"])
    # COLLECTIVE FLOW: the velocity correlation length of the interior cells over 30-min displacements (a step is
    # too noisy to correlate), median over start rows
    k30 = max(1, int(round(0.5 / row_h))) if row_h and np.isfinite(row_h) else 1
    vl = []
    for t0 in range(0, n - k30, max(1, k30)):
        vl.append(velocity_correlation_length(Xi[t0] / um, (Xi[t0 + k30] - Xi[t0]) / um, d0))
    vl = [v for v in vl if v is not None]
    out["vcorr_len_cells"] = finite(np.median(vl)) if vl else None
    nc_ = [neighbour_correlation(Xi[t0] / um, affine_residual(Xi[t0], Xi[t0 + k30]) / um, d0)
           for t0 in range(0, n - k30, max(1, k30))]
    nc_ = [v for v in nc_ if v is not None]
    out["corr_1sp"] = finite(np.median(nc_)) if nc_ else None
    lt = [neighbour_correlation_lt(Xi[t0] / um, affine_residual(Xi[t0], Xi[t0 + k30]) / um, d0)
          for t0 in range(0, n - k30, max(1, k30))]
    cl_ = [a for a, _ in lt if a is not None]; ct_ = [b for _, b in lt if b is not None]
    out["corr_L_1sp"] = finite(np.median(cl_)) if cl_ else None
    out["corr_T_1sp"] = finite(np.median(ct_)) if ct_ else None
    return out


register_run("exp21.motion", motion, None, "in-plane cell motion and neighbour retention, Wang's estimators")


# ============================================================================ exp21.shear
def shear(T, cell_set="cell", axis_flow=0, axis_split=1, rim=1.5, **_):
    """The counter-flow arm: the sheet's two halves (split at the row-0 median of `axis_split`) driven in opposite
    directions along `axis_flow`.

      slip_um             the halves' mean displacement difference along the flow (how far they slid past each other)
      crossed_frac        of the cells that started within one spacing of the split line, the share whose
                          row-0 neighbours on the OTHER side of the line are no longer neighbours at the last row
                          -- the two crowds exchanged partners rather than tearing apart or locking
      band_spin_sign      the mean in-plane rotation of the band's cells' neighbourhoods (sign of the vorticity of
                          the cells' velocities in the band) times the sign the shear imposes: +1 the band rolls
                          with the shear, as two people passing turn about each other
      gap_max             the largest hole (1 - coverage) the band opens over the run"""
    um, s_per_frame = _units(T)
    n = T.n_rows()
    C = np.array([_arr(T, f"{cell_set}__pos", t) for t in range(n)])
    co = np.array([(_arr(T, f"{cell_set}__occ", t).reshape(-1) > 0.5) if f"{cell_set}__occ" in T.z.files
                   else np.ones(C.shape[1], bool) for t in range(n)])
    alive = co.all(0) & np.isfinite(C).all(axis=(0, 2))
    X = C[:, alive, :2]
    if X.shape[1] < 10:
        return {"available": False}
    from scipy.spatial import cKDTree
    d0 = float(np.median(cKDTree(X[0]).query(X[0], k=2)[0][:, 1]))
    s0 = np.median(X[0][:, axis_split])
    side = X[0][:, axis_split] > s0
    disp = X[-1] - X[0]
    slip = float(disp[side, axis_flow].mean() - disp[~side, axis_flow].mean())
    band = np.abs(X[0][:, axis_split] - s0) < d0
    pairs = cKDTree(X[0]).query_pairs(1.25 * d0, output_type="ndarray")
    cross = pairs[side[pairs[:, 0]] != side[pairs[:, 1]]]
    if len(cross):
        dl = np.linalg.norm(X[-1][cross[:, 0]] - X[-1][cross[:, 1]], axis=1)
        crossed = float(np.mean(dl > 1.5 * d0))
    else:
        crossed = None
    # vorticity of the band: velocities over the run, curl from a least-squares gradient of (vx, vy) over the band
    V = (X[-1] - X[0])
    xb, vb = X[0][band], V[band]
    spin = None
    if band.sum() >= 6:
        A = np.c_[xb - xb.mean(0), np.ones(len(xb))]
        gx = np.linalg.lstsq(A, vb[:, 0], rcond=None)[0]
        gy = np.linalg.lstsq(A, vb[:, 1], rcond=None)[0]
        w = gy[0] - gx[1]                                   # dvy/dx - dvx/dy
        imposed = -np.sign(slip) if axis_flow == 0 else np.sign(slip)
        spin = float(np.sign(w) * imposed)
    return {"available": True, "slip_um": finite(slip * um), "slip_cells": finite(slip / d0),
            "crossed_frac": finite(crossed), "band_spin_sign": finite(spin), "n_band": int(band.sum())}


register_run("exp21.shear", shear, None, "opposed streams passing: slip, partner exchange across the line, spin")


# ============================================================================ exp21.division
def division(T, cell_set="cell", axis=2, sister_lag_h=24.0, **_):
    """Divisions read from the cell buffer: a slot that turns live at row t is a newborn; its sister is the live
    cell nearest to it at t. Per division, the angle of the sister axis to the sheet's plane (0 = in the plane).

      n_divisions, cells_first, cells_last
      angle_deg_median        median out-of-plane angle of the sister axes
      in_plane_frac           share of divisions whose axis is within 20 deg of the plane
      newborn_out_frac        share of newborns whose centre is off the sheet's median plane by more than half the
                              median cell spacing at the LAST row (extruded or stacked)
      sisters_apart_frac      share of sister pairs no longer neighbours `sister_lag_h` (24 h) after their birth"""
    n = T.n_rows()
    if f"{cell_set}__occ" not in T.z.files:
        return {"available": False}
    occ = np.array([_arr(T, f"{cell_set}__occ", t).reshape(-1) > 0.5 for t in range(n)])
    C = np.array([_arr(T, f"{cell_set}__pos", t) for t in range(n)])
    born = []
    for t in range(1, n):
        new = np.flatnonzero(occ[t] & ~occ[t - 1])
        for s in new:
            others = np.flatnonzero(occ[t]); others = others[others != s]
            if others.size == 0:
                continue
            j = others[np.argmin(np.linalg.norm(C[t][others] - C[t][s], axis=1))]
            born.append((t, int(s), int(j)))
    ang = []
    for t, s, j in born:
        v = C[t][s] - C[t][j]
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            ang.append(np.degrees(np.arcsin(min(1.0, abs(v[axis]) / nv))))
    out = {"available": True, "n_divisions": len(born), "cells_first": int(occ[0].sum()), "cells_last": int(occ[-1].sum())}
    # SISTERS APART AFTER A DAY (Aw et al. 2016 Fig 4D: in the basal epidermis 42 % of sister pairs at E13.5 are
    # separated by at least one cell 24 h after division): a pair is apart when its centres are farther than
    # 1.5 x the median spacing at that row (no longer neighbours), read `sister_lag_h` after the birth row
    um, s_per_frame = _units(T)
    fr = np.asarray(_row_frames(T) if _row_frames else list(range(n)), float)
    hours = fr * s_per_frame / 3600.0
    apart = []
    from scipy.spatial import cKDTree
    for t, s, j in born:
        tt = int(np.searchsorted(hours, hours[t] + float(sister_lag_h)))
        if tt >= n or not (occ[tt][s] and occ[tt][j]):
            continue
        Cl = C[tt][occ[tt]]
        d0 = float(np.median(cKDTree(Cl[:, :2]).query(Cl[:, :2], k=2)[0][:, 1]))
        apart.append(float(np.linalg.norm(C[tt][s, :2] - C[tt][j, :2]) > 1.5 * d0))
    out["sisters_apart_frac"] = finite(np.mean(apart)) if apart else None
    out["sisters_read"] = len(apart)
    if ang:
        out["angle_deg_median"] = finite(np.median(ang))
        out["in_plane_frac"] = finite(np.mean(np.asarray(ang) < 20.0))
        Cl = C[-1][occ[-1]]
        from scipy.spatial import cKDTree
        d0 = float(np.median(cKDTree(Cl[:, :2]).query(Cl[:, :2], k=2)[0][:, 1]))
        zmed = np.median(Cl[:, axis])
        nb = [s for _t, s, _j in born if occ[-1][s]]
        out["newborn_out_frac"] = finite(np.mean(np.abs(C[-1][nb, axis] - zmed) > 0.5 * d0)) if nb else None
    else:
        out.update({"angle_deg_median": None, "in_plane_frac": None, "newborn_out_frac": None})
    return out


register_run("exp21.division", division, None, "in-plane divisions: count, axis angle, newborns kept in the layer")


# ============================================================================ exp21.wave
def wave(T, cell_set="cell", block="chem", channel=0, thresh=0.5, cut=None, **_):
    """A contact-borne excitation (gap-junction coupling between touching cells, exp06's kinetics): each cell's
    activation time is the first row its `block[channel]` exceeds `thresh`.

      reached_frac            share of live cells activated at any row
      speed_cells_h           front speed: slope of distance-from-first-activated-cell (in median spacings) against
                              activation time (h), least squares over activated cells; `speed_um_s` the same in um/s
      beyond_cut_frac         with `cut: {axis: a, at: v}` (world), the share of cells on the far side of the plane
                              x_a = v from the first-activated cell that were reached (0 = the cut blocks)"""
    um, s_per_frame = _units(T)
    n = T.n_rows()
    key = f"{cell_set}__{block}"
    if key not in T.z.files:
        return {"available": False}
    fr = np.asarray(_row_frames(T) if _row_frames else list(range(n)), float)
    hours = fr * s_per_frame / 3600.0
    V = np.array([_arr(T, key, t).reshape(_arr(T, key, t).shape[0], -1)[:, channel] for t in range(n)])
    occ = np.array([(_arr(T, f"{cell_set}__occ", t).reshape(-1) > 0.5) if f"{cell_set}__occ" in T.z.files
                    else np.ones(V.shape[1], bool) for t in range(n)])
    alive = occ.all(0)
    C0 = _arr(T, f"{cell_set}__pos", 0)
    act = np.full(V.shape[1], np.nan)
    hit = V > thresh
    first = np.argmax(hit, axis=0)
    ever = hit.any(0)
    act[ever] = hours[first[ever]]
    sel = alive & ever
    out = {"available": True, "reached_frac": finite(np.mean(ever[alive])) if alive.any() else None}
    if sel.sum() >= 5:
        from scipy.spatial import cKDTree
        X = C0[alive][:, :2]
        d0 = float(np.median(cKDTree(X).query(X, k=2)[0][:, 1]))
        src = np.flatnonzero(sel)[np.nanargmin(act[sel])]
        dist = np.linalg.norm(C0[:, :2] - C0[src, :2], axis=1) / d0
        tt = act[sel] - act[src]
        if np.ptp(tt) > 0:
            out["speed_cells_h"] = finite(np.polyfit(tt, dist[sel], 1)[0])
            out["speed_um_s"] = finite(out["speed_cells_h"] * d0 * um / 3600.0)
        if cut:
            a, v = int(cut["axis"]), float(cut["at"])
            far = alive & (np.sign(C0[:, a] - v) != np.sign(C0[src, a] - v))
            out["beyond_cut_frac"] = finite(np.mean(ever[far])) if far.any() else None
    return out


register_run("exp21.wave", wave, None, "contact-borne excitation: reach, front speed, block at a cut")

"""exp10 organoid_crypt -- the ruler: does a closed shell grow ONE crypt, at its fate patch, as its lumen shrinks?

    exp10.crypt    per recorded row: outward buds (count, depth, neck, body width), whether the bud
                   sits on the fate patch, how round the rest of the shell stays, and the lumen volume
    exp10.mesh_sanity  `tools/mesh_sanity.py` over the run (every cell still a cell of the surface),
                   plus the fractions of COLLAPSED and SLIVER cells, which its lines do not look for

THE REFERENCE SURFACE IS A SPHERE FITTED TO THE REST OF THE SHELL -- the cells off the declared patch
when the fate block exists -- NOT THE MEDIAN RADIUS ABOUT THE CENTROID. A crypt of a tenth of the cells drags the plain centroid toward itself and inflates the
median radius, so the bud reads shallower the deeper it grows. The fit is the algebraic least-squares
sphere |x|^2 = 2 c.x + k (centre c, radius R = sqrt(k + |c|^2)) on the cell centroids, refitted on
the cells within `h_bud` of the previous sphere AND among the three quarters closest to it (`fit_sphere`),
so a bud -- even a broad one -- never sets the surface it is measured from. Each cell's height is h = |x - c| / R - 1, in radii of that sphere.

A BUD IS A CONNECTED PATCH OF RAISED CELLS. Cells with h > `h_bud` (default 0.2, a fifth of the
shell's radius, model choice) are joined through shared edges (`common.neighbour_pairs`), and a
component of at least `min_cells` cells is one outward bud. Components below -`h_bud` are inward
folds, counted apart: an invagination is not the target's crypt. A whole-body elongation of aspect
1.2 (long axis over short) stays below the line at both poles and counts no bud (tested).

THE NECK SITS AT THE SHELL, BELOW THE COUNTING LINE, so the shape is read on a hysteresis region: a
bud is COUNTED on its cells above `h_bud`, and its PROFILE is read on the connected cells above
`h_base` (default 0.05 radii, model choice) that contain it. Without it the profile starts 0.2 radii up
the bud and a ball-shaped crypt on a neck of 0.72 of its diameter read 0.90 (tested).

THE CRYPT'S SHAPE, on the deepest bud, from the mid-surface VERTICES of its region's cells:
    axis d     from the sphere's centre to the bud's farthest vertex
    depth      (farthest projection onto d) - R: the tip's height above the reference sphere, in the
               run's length units; `depth_rel` is the same over R
    profile    the bud's vertices in `n_bins` equal-count bins along d; each bin's diameter is twice
               the median distance of its vertices from the axis
    neck       the tightest pinch: the bin i minimising diam[i] / max(diam[i:]), its diameter over the
               widest section ABOVE it (toward the tip). A cone or a dome narrows all the way up and
               has none (neck = width, ratio 1); a crypt pinched below its body does (the target's
               "narrow neck"). Taken against what lies above, not against the whole profile, because
               the region flares into the shell at its base and that flare can be the widest section
    width      the widest section above the neck (the crypt's body)
    depth_over_width, depth_over_neck, neck_ratio = neck / width
`depth_rel` is 0 on a row with no bud, so the hold arm's depth over the crypt arm's is a number when
the hold blocks the crypt entirely; every other shape field is None on such a row.

YANG ET AL. 2021'S OWN MORPHOMETRICS, measured the way the paper measured them, because its bands are
set on these and not on depth or neck (Yang 2021 Methods, "Light sheet data analysis" and "Time-course
image analysis"):
    Rc_over_Rv   radius of curvature of the crypt over the villus's: the sphere fitted to the bud's
                 mid-surface vertices over the sphere fitted to the rest. The mid-surface is the
                 paper's "average of the apical and basal radii".
    hc_over_hv   epithelial thickness in the crypt over the villus's: 2 |sep| averaged over each
                 cell's ring, then over the bud's cells and over the rest's (None without `sep`).
    lumen_frac   the paper's "lumen ratio": lumen volume over the organoid's total (the volume inside
                 the OUTER surface), per row; `lumen_frac_ratio` is last over first.
    ecc          the paper's 2D eccentricity of the segmented z-projection (skimage regionprops,
                 sqrt(1 - l2/l1) of the mask's second moments). An organoid in a dish lies at any
                 orientation, so the projection is taken along 7 fixed directions -- the 3 axes and
                 the 4 body diagonals -- and averaged; each silhouette is the outer surface's vertices
                 rasterised at 128 pixels across, closed and hole-filled. `ecc_gain` = last - first.

THE PATCH is a declared per-cell block (`patch_block`, default `fate`; a cell with a value above
`patch_min` is a patch cell). `on_patch` is the fraction of the bud's cells that are patch cells,
`patch_in_bud` the fraction of the patch inside the bud. Without the block both are None and
`one_crypt_at_patch` is 0: a crypt with no patch to be at cannot pass that gate.

THE PATCH'S OWN GEOMETRY, bud or no bud, because a patch can fail two ways the bud count cannot tell
apart -- flattening (a taut lid, exp 10 batch 2) or sinking: `patch_R_over_Rv` is the sphere fitted to
the patch cells' mid-surface vertices over the sphere fitted to the OTHER cells' centroids (below 1
bulging, above 1 flatter than the shell), `patch_h_mean` the patch cells' mean height above that
sphere in its radii, `patch_hc_over_hv` their mean thickness 2|sep| over the other cells'.

THE LUMEN is the volume enclosed by the shell's INNER surface. The apico-basal mesh stores the
mid-surface `pos` and a per-vertex half-thickness `sep` (apical = pos + sep, basal = pos - sep,
`vertex_ops.apicobasal_geometry_3d`); each face's ring is fanned from its own centroid and the fans
summed by the divergence theorem, V = (1/6) sum (c_f - o) . ((p_s - o) x (p_t - o)), o the mean
vertex -- the same fan and winding as the cell volumes. Both surfaces are measured and the smaller
enclosed volume is the lumen (`lumen_side` says which: `apical` for an organoid, apical side in).
Without `sep` recorded, the mid-surface is used.
"""
from __future__ import annotations

import numpy as np

from .common import CoreTraj, cells, finite, neighbour_pairs, register_run


# ============================================================================ geometry
def fit_sphere(x, h_bud=0.2, iters=6, trim=0.75):
    """(centre, R) of the least-squares sphere through x, refitted on the points within h_bud of it AND
    among the `trim` fraction closest to it. The trim is what keeps a BROAD crypt out of its own
    reference: a patch of a fifth of the cells raised 0.3 radii pulls an all-cell sphere out until no
    patch cell stands 0.2 above it (a planted mesa read 0 buds at R 5.35 for a shell of 5.0; batch 8's
    v8a and v8b crypts were missed that way). 0.75 leaves room for Yang's crypt size, 0.2 +- 0.06."""
    keep = np.ones(len(x), bool)
    c, R = x.mean(0), 1.0
    for _ in range(iters + 1):
        p = x[keep]
        A = np.c_[2 * p, np.ones(len(p))]
        sol, *_r = np.linalg.lstsq(A, (p * p).sum(1), rcond=None)
        c = sol[:3]
        R = float(np.sqrt(max(sol[3] + c @ c, 1e-30)))
        a = np.abs(np.linalg.norm(x - c, axis=1) / R - 1)
        nk = (a <= h_bud) & (a <= np.quantile(a, trim))
        if nk.sum() < 10 or (nk == keep).all():
            break
        keep = nk
    return c, R


def components(n, pairs, mask):
    """Connected components of the masked nodes over the edge list: list of index arrays."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    m = mask[pairs[:, 0]] & mask[pairs[:, 1]] if len(pairs) else np.zeros(0, bool)
    p = pairs[m]
    G = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    _k, lab = connected_components(G, directed=False)
    lab = lab[idx]
    return [idx[lab == u] for u in np.unique(lab)]


def enclosed_volume(P, es, et, ef, faces):
    """|volume| enclosed by the polygon faces `faces` of the half-edge table, each ring fanned from
    its own centroid (see the module docstring)."""
    keep = np.isin(ef, faces)
    es, et, ef = es[keep], et[keep], ef[keep]
    if len(es) == 0:
        return None
    nF = int(ef.max()) + 1
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, P[es])
    cen /= np.maximum(cnt, 1)[:, None]
    o = P[np.unique(es)].mean(0)
    a, b, q = cen[ef] - o, P[es] - o, P[et] - o
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, q)).sum()) / 6.0)


def lsq_sphere(x):
    """(centre, R) of the plain least-squares sphere through x (no exclusion)."""
    A = np.c_[2 * x, np.ones(len(x))]
    sol, *_r = np.linalg.lstsq(A, (x * x).sum(1), rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 1e-30)))


VIEWS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float)
VIEWS /= np.linalg.norm(VIEWS, axis=1, keepdims=True)


def silhouette_ecc(X, d, px=128):
    """Eccentricity of the filled projection of the point cloud X (a closed surface's vertices)
    along the unit direction d: sqrt(1 - l2/l1), l1 >= l2 the eigenvalues of the mask's pixel
    covariance (skimage regionprops' definition)."""
    from scipy import ndimage
    e1 = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    y = np.c_[X @ e1, X @ e2]
    lo, span = y.min(0), float(np.ptp(y, 0).max())
    if span <= 0:
        return None
    ij = np.clip(((y - lo) / span * (px - 1)).round().astype(int), 0, px - 1)
    m = np.zeros((px + 8, px + 8), bool)
    m[ij[:, 0] + 4, ij[:, 1] + 4] = True
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    q = np.argwhere(m).astype(float)
    lam = np.linalg.eigvalsh(np.cov(q.T))
    return float(np.sqrt(max(0.0, 1 - lam[0] / lam[1])))


def bud_profile(V, c, R, n_bins=8):
    """Depth, width, neck of one bud from its vertices V around the sphere (c, R)."""
    r = V - c
    tip = r[np.argmax(np.linalg.norm(r, axis=1))]
    d = tip / np.linalg.norm(tip)
    z = r @ d
    rho = np.linalg.norm(r - np.outer(z, d), axis=1)
    depth = float(z.max() - R)
    o = np.argsort(z)
    bins = [b for b in np.array_split(o, min(n_bins, max(1, len(o) // 3))) if len(b)]
    diam = np.array([2 * np.median(rho[b]) for b in bins])
    above = np.maximum.accumulate(diam[::-1])[::-1]
    ratio = np.where(above > 0, diam / np.maximum(above, 1e-30), 1.0)
    k = int(np.argmin(ratio))
    width, neck = float(above[k]), float(diam[k])
    return dict(depth=depth, depth_rel=depth / R, width=width, neck=neck,
                depth_over_width=depth / width if width > 0 else None,
                depth_over_neck=depth / neck if neck > 0 else None,
                neck_ratio=neck / width if width > 0 else None, axis=d.tolist())


# ============================================================================ one row
def crypt_row(T, t, patch_block="fate", patch_min=0.5, h_bud=0.2, min_cells=5, n_bins=8, h_base=0.05):
    c = cells(T, t)
    x = c.x
    ok = np.isfinite(x).all(1)
    pb = c.block(patch_block)
    patch = None if pb is None else (pb[:, 0] > patch_min)
    # THE REFERENCE IS THE REST OF THE SHELL, LITERALLY, when the fate is declared: the sphere of the
    # non-patch cells. A broad crypt pulls any sphere fitted through it -- a planted mesa (a fifth of the
    # cells 0.3 radii out) settled on a compromise sphere of radius 5.55 offset 0.95 toward it, and no
    # residual trim separates the two. Without a fate block, the trimmed all-cell fit.
    if patch is not None and (patch & ok).sum() >= 4 and (~patch & ok).sum() >= 10:
        ctr, R = fit_sphere(x[~patch & ok], h_bud)
    else:
        ctr, R = fit_sphere(x[ok], h_bud)
    h = np.linalg.norm(x - ctr, axis=1) / R - 1
    h[~ok] = 0.0
    pairs = neighbour_pairs(T, t, c)
    out_c = [g for g in components(len(x), pairs, h > h_bud) if len(g) >= min_cells]
    in_c = [g for g in components(len(x), pairs, h < -h_bud) if len(g) >= min_cells]
    in_bud = np.zeros(len(x), bool)
    for g in out_c:
        in_bud[g] = True
    rest = ok & ~in_bud
    row = dict(n_buds=len(out_c), n_in=len(in_c), R=R, depth_rel=0.0,
               rest_rms=float(np.sqrt(np.mean(h[rest] ** 2))) if rest.any() else None)
    row["patch_cells"] = None if patch is None else int(patch.sum())
    is_mesh = isinstance(T, CoreTraj)
    cell_th = None
    if is_mesh:
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        P = T.pos(t)
        S = T.vertex_block("sep", t)
        if S is not None and S.shape == P.shape:
            va = enclosed_volume(P + S, es, et, ef, c.slot)
            vb = enclosed_volume(P - S, es, et, ef, c.slot)
            row["lumen"], row["lumen_side"] = (va, "apical") if va <= vb else (vb, "basal")
            outer = P - S if va <= vb else P + S
            row["lumen_frac"] = min(va, vb) / max(va, vb) if max(va, vb) > 0 else None
            th = 2 * np.linalg.norm(S, axis=1)
            nF = int(ef.max()) + 1
            cth = np.bincount(ef, th[es], nF) / np.maximum(np.bincount(ef, minlength=nF), 1)
            cell_th = cth[c.slot]
        else:
            row["lumen"], row["lumen_side"] = enclosed_volume(P, es, et, ef, c.slot), "mid"
            outer, cell_th, row["lumen_frac"] = P, None, None
        live_v = np.unique(es[np.isin(ef, c.slot)])
        eccs = [silhouette_ecc(outer[live_v], v) for v in VIEWS]
        row["ecc"] = float(np.mean(eccs)) if all(e is not None for e in eccs) else None
    if patch is not None and patch.sum() >= 4 and (~patch & ok).sum() >= 10:
        c_r, R_r = fit_sphere(x[~patch & ok], h_bud)
        Vp = P[np.unique(es[np.isin(ef, c.slot[patch])])] if is_mesh else x[patch]
        row["patch_R_over_Rv"] = lsq_sphere(Vp)[1] / R_r if len(Vp) >= 4 else None
        row["patch_h_mean"] = float((np.linalg.norm(x[patch & ok] - c_r, axis=1) / R_r - 1).mean())
        if is_mesh and cell_th is not None:
            row["patch_hc_over_hv"] = float(cell_th[patch].mean() / max(cell_th[~patch].mean(), 1e-12))
    if out_c:
        base = components(len(x), pairs, h > h_base)
        lab = -np.ones(len(x), np.int64)
        for i, b in enumerate(base):
            lab[b] = i
        best = None
        for g in out_c:
            region = base[lab[g[0]]]
            V = P[np.unique(es[np.isin(ef, c.slot[region])])] if is_mesh else x[region]
            pr = bud_profile(V, ctr, R, n_bins)
            if best is None or pr["depth"] > best[1]["depth"]:
                best = (g, pr)
        g, pr = best
        row.update(pr)
        row["bud_cells"] = int(len(g))
        Vb = P[np.unique(es[np.isin(ef, c.slot[g])])] if is_mesh else x[g]
        row["Rc_over_Rv"] = lsq_sphere(Vb)[1] / R if len(Vb) >= 4 else None
        if is_mesh and cell_th is not None and rest.any():
            row["hc_over_hv"] = float(cell_th[g].mean() / max(cell_th[rest].mean(), 1e-12))
        if patch is not None:
            row["on_patch"] = float(patch[g].mean())
            row["patch_in_bud"] = float(patch[g].sum() / max(patch.sum(), 1))
    return row


# ============================================================================ the run
def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


def crypt(T, every=5, patch_block="fate", patch_min=0.5, h_bud=0.2, min_cells=5, n_bins=8, h_base=0.05, **_):
    """The crypt ruler over the run: last-row values, maxima, lumen ratios and the per-row series."""
    ts = _rows(T, every)
    rows = [crypt_row(T, t, patch_block, patch_min, h_bud, min_cells, n_bins, h_base) for t in ts]
    first, last = rows[0], rows[-1]
    out = {"rows": ts}
    for k in ("n_buds", "n_in", "depth_rel", "depth", "width", "neck", "depth_over_width",
              "depth_over_neck", "neck_ratio", "on_patch", "patch_in_bud", "rest_rms", "R",
              "bud_cells", "patch_cells", "lumen", "Rc_over_Rv", "hc_over_hv", "lumen_frac", "ecc",
              "patch_R_over_Rv", "patch_h_mean", "patch_hc_over_hv"):
        v = last.get(k)
        out[f"{k}_last"] = v if isinstance(v, int) or v is None else finite(v)
    out["n_buds_max"] = int(max(r["n_buds"] for r in rows))
    out["depth_rel_max"] = finite(max(r["depth_rel"] for r in rows))
    rr = [r["rest_rms"] for r in rows if r.get("rest_rms") is not None]
    out["rest_rms_max"] = finite(max(rr)) if rr else None
    out["one_crypt_last"] = float(last["n_buds"] == 1)
    op = last.get("on_patch")
    out["one_crypt_at_patch"] = float(last["n_buds"] == 1 and op is not None and op >= 0.5)
    lum = [r.get("lumen") for r in rows]
    out["lumen_side"] = last.get("lumen_side")
    if all(v is not None for v in lum) and lum[0]:
        out["lumen_first"] = finite(lum[0])
        out["lumen_ratio"] = finite(lum[-1] / lum[0])
        out["lumen_min_ratio"] = finite(min(lum) / lum[0])
        dep = [r["depth_rel"] for r in rows]
        if np.ptp(dep) > 0 and np.ptp(lum) > 0:
            from scipy.stats import spearmanr
            out["lumen_depth_rho"] = finite(spearmanr(lum, dep).correlation)
        else:
            out["lumen_depth_rho"] = None
        out["series_lumen"] = [finite(v) for v in lum]
    fr = [r.get("lumen_frac") for r in rows]
    if all(v is not None for v in fr) and fr[0]:
        out["lumen_frac_first"] = finite(fr[0])
        out["lumen_frac_ratio"] = finite(fr[-1] / fr[0])
    ec = [r.get("ecc") for r in rows]
    if all(v is not None for v in ec):
        out["ecc_first"] = finite(ec[0])
        out["ecc_gain"] = finite(ec[-1] - ec[0])
        out["series_ecc"] = [finite(v) for v in ec]
    out["series_n_buds"] = [int(r["n_buds"]) for r in rows]
    out["series_depth_rel"] = [finite(r["depth_rel"]) for r in rows]
    return out


def ring_aspect(P, es, ef, nF):
    """Per cell, the in-plane aspect of its ring: sqrt(l1 / l2) of the two largest eigenvalues of the
    covariance of its vertices about their centroid (1 for a regular polygon, large for a sliver)."""
    cnt = np.maximum(np.bincount(ef, minlength=nF), 1)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, P[es])
    cen /= cnt[:, None]
    d = P[es] - cen[ef]
    C = np.zeros((nF, 3, 3))
    np.add.at(C, ef, d[:, :, None] * d[:, None, :])
    w = np.linalg.eigvalsh(C / cnt[:, None, None])
    return np.sqrt(w[:, 2] / np.maximum(w[:, 1], 1e-30))


def mesh_sanity(T, every=5, collapse=0.1, sliver=4.0, **_):
    """`tools/mesh_sanity.py` on the run, imported so its lines stay its own (Euler characteristic, an
    edge 8x the median, a cell 10x the median area, a vertex 4 edges off the shell): `sane` 1.0 when
    no sampled row breaks one -- the radial line excepted, see below -- else 0.0, with the first
    broken row; `sane_tool` is the tool's verdict with every line. The wrecked cap reads it in place
    of exp 3's growth auditor, per the human (2026-09-26), as exp 12 does.

    THE TOOL LOOKS FOR CELLS THAT BLOW UP, NOT FOR CELLS THAT VANISH, and a vanishing cell is exp 10's
    failure: on batch 1 the free shell lost 363 of 1,280 cells to near-zero area by frame 80 with no
    line of the tool crossed (finding 7). `collapsed_*` is the fraction of cells whose area (the same
    fan area the tool computes) is below `collapse` (default 0.1, model choice) of the row's median.

    NOR FOR A CELL SHEARED INTO A SLIVER: batch 4's crypts stood on villus cells squeezed to a median
    in-plane aspect of 3.2 and a worst tenth above 10 (the seeded shell: 1.25 and 1.5), a texture of
    needles in the movie that no line of the tool crossed. `sliver_*` is the fraction of cells whose
    ring aspect (`ring_aspect`) exceeds `sliver` (default 4, model choice: twice the seeded worst
    tenth, doubled again)."""
    import mesh_sanity as MS
    ts = list(range(0, T.n_rows(), max(1, int(every))))
    if ts[-1] != T.n_rows() - 1:
        ts.append(T.n_rows() - 1)
    rows = [MS.row(T, t) for t in ts]
    first_tool = next((r for r in rows if r["bad"]), None)
    # THE RADIAL LINE IS DROPPED FROM `sane`, AND ONLY IT. It flags a vertex 4 median edges off the
    # shell's MEDIAN radius -- the tool's guard against a vertex flown off an intact shell (exp 14:
    # radius 116 against 6) -- and a crypt is off that radius by construction: batch 4's buds, 0.41 and
    # 0.54 radii deep, put their tips 5.9 and 9.6 edge lengths out and failed it on the very object
    # the experiment asks for. A flown vertex still breaks the EDGE line (8x the median), which stays;
    # the rest's roundness is G-smooth's. The tool's own verdict is kept as `sane_tool`.
    keep = [[b for b in r["bad"] if "off the shell" not in b] for r in rows]
    first_i = next((i for i, b in enumerate(keep) if b), None)
    first = None if first_i is None else dict(rows[first_i], bad=keep[first_i])
    col, sl = [], []
    for t in ts:
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        P = np.asarray(T.pos(t), float)
        nF = T.nF(t)
        fc = np.zeros((nF, 3))
        np.add.at(fc, ef, P[es])
        fc /= np.maximum(np.bincount(ef, minlength=nF), 1)[:, None]
        A = np.zeros(nF)
        np.add.at(A, ef, 0.5 * np.linalg.norm(np.cross(P[es] - fc[ef], P[et] - fc[ef]), axis=1))
        col.append(float((A < collapse * np.median(A)).mean()))
        sl.append(float((ring_aspect(P, es, ef, nF) > sliver).mean()))
    worst = {k: finite(max((r.get(k, 0) or 0) for r in rows)) for k in ("radial", "edge", "area")}
    return {"sane": float(first is None), "first_bad_row": None if first is None else int(first["row"]),
            "first_bad": None if first is None else "; ".join(first["bad"]),
            "sane_tool": float(first_tool is None),
            "first_bad_tool": None if first_tool is None else f"row {first_tool['row']}: " + "; ".join(first_tool["bad"]),
            **{f"worst_{k}": v for k, v in worst.items()},
            "collapsed_last": finite(col[-1]), "collapsed_max": finite(max(col)),
            "series_collapsed": [finite(v) for v in col],
            "sliver_last": finite(sl[-1]), "sliver_max": finite(max(sl)),
            "series_sliver": [finite(v) for v in sl], "rows": ts}


register_run("exp10.mesh_sanity", mesh_sanity, None, "tools/mesh_sanity.py + the collapsed-cell fraction")
register_run("exp10.crypt", crypt, None,
             "outward buds on a closed shell: count, depth/neck/width, on the patch, rest roughness, lumen")

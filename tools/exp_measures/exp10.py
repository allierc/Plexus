"""exp10 organoid_crypt -- the ruler: does a closed shell grow ONE crypt, at its fate patch, as its lumen shrinks?

    exp10.crypt    per recorded row: outward buds (count, depth, neck, body width), whether the bud
                   sits on the fate patch, how round the rest of the shell stays, and the lumen volume
    exp10.mesh_sanity  `tools/mesh_sanity.py` over the run (every cell still a cell of the surface),
                   plus the fractions of COLLAPSED and SLIVER cells, which its lines do not look for
    exp10.choice   phase 2: whether Notch-Delta lateral inhibition chose, at what cell count, and the Wnt
                   patch it drew (the population's decision; `crypt`'s `budded` is its outcome)
    exp10.precrypt phase 2: the organoid inside LSTree's 19-24-cell window -- axis ratio, lumen fraction,
                   wall thickness over radius; `lstree_precrypt(root)` reads the same off the real data

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
`patch_in_bud` the fraction of the patch inside the bud; `patch_col` picks the block's column (phase 2: the
Wnt column of `cell_chem_react[notch_delta]`, 3). Without the block both are None and
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
def crypt_row(T, t, patch_block="fate", patch_min=0.5, h_bud=0.2, min_cells=5, n_bins=8, h_base=0.05, patch_col=0):
    c = cells(T, t)
    x = c.x
    ok = np.isfinite(x).all(1)
    pb = c.block(patch_block)
    patch = None if pb is None or pb.shape[1] <= patch_col else (pb[:, patch_col] > patch_min)
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
        if is_mesh and pr.get("neck") is not None:
            Q = P[np.unique(es[np.isin(ef, c.slot)])]
            Q = Q - Q.mean(0)
            ax = np.linalg.eigh(np.cov(Q.T))[1][:, -1]
            row["neck_over_major"] = float(pr["neck"] / max(np.ptp(Q @ ax), 1e-12))
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


def crypt(T, every=5, patch_block="fate", patch_min=0.5, h_bud=0.2, min_cells=5, n_bins=8, h_base=0.05, patch_col=0,
          **_):
    """The crypt ruler over the run: last-row values, maxima, lumen ratios and the per-row series."""
    ts = _rows(T, every)
    rows = [crypt_row(T, t, patch_block, patch_min, h_bud, min_cells, n_bins, h_base, patch_col) for t in ts]
    first, last = rows[0], rows[-1]
    out = {"rows": ts}
    for k in ("n_buds", "n_in", "depth_rel", "depth", "width", "neck", "depth_over_width",
              "depth_over_neck", "neck_ratio", "on_patch", "patch_in_bud", "rest_rms", "R",
              "bud_cells", "patch_cells", "lumen", "Rc_over_Rv", "hc_over_hv", "lumen_frac", "ecc",
              "patch_R_over_Rv", "patch_h_mean", "patch_hc_over_hv", "neck_over_major"):
        v = last.get(k)
        out[f"{k}_last"] = v if isinstance(v, int) or v is None else finite(v)
    out["n_buds_max"] = int(max(r["n_buds"] for r in rows))
    out["depth_rel_max"] = finite(max(r["depth_rel"] for r in rows))
    rr = [r["rest_rms"] for r in rows if r.get("rest_rms") is not None]
    out["rest_rms_max"] = finite(max(rr)) if rr else None
    out["one_crypt_last"] = float(last["n_buds"] == 1)
    op = last.get("on_patch")
    out["one_crypt_at_patch"] = float(last["n_buds"] == 1 and op is not None and op >= 0.5)
    # THE POPULATION READS THESE TWO, one run = one organoid: the seed mean of `budded` is the fraction
    # of organoids that bud (Serra 2019 Fig. 1g's complement of the enterocysts), and `one_crypt_if_budded`
    # is None on an organoid that did not bud, so its seed mean -- the scorer drops None -- is the
    # fraction of BUDDING organoids that grew exactly one crypt, on the patch (`one_crypt_at_patch`).
    out["budded"] = float(last["n_buds"] >= 1)
    out["one_crypt_if_budded"] = out["one_crypt_at_patch"] if last["n_buds"] >= 1 else None
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
        # A GROWING ORGANOID'S FIRST ROW IS NOT YANG'S "BEFORE BULGING": it is the 12-cell start, whose
        # lumen may open only later. Last over the run's largest lumen fraction is the shrink from the
        # inflated state (Yang Fig. 1f's before -> after); on a non-growing shell whose lumen only
        # shrinks it equals `lumen_frac_ratio`.
        out["lumen_frac_max"] = finite(max(fr))
        out["lumen_frac_last_over_max"] = finite(fr[-1] / max(fr)) if max(fr) > 0 else None
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


# ============================================================================ phase 2: the choice
def choice(T, chem_block="chem", n_col=0, d_col=1, w_col=3, d_on=0.5, n_off=0.5, w_on=0.5, win_max=0.5,
           frames_per_row=None, checkpoints=(8, 16, 32, 64), every=1, **_):
    """Did lateral inhibition choose, when, at what size -- and what patch did its Wnt draw?

    Reads the `cell_chem_react[notch_delta]` columns ([N, D, Y, W] from `n_col`; `d_col` and `w_col`
    absolute) on every sampled row:
        winners   cells with Delta above `d_on` AND Notch below `n_off` -- a lateral-inhibition winner
                  sends Delta and receives none (Collier 1996); the DLL1+ cell of Serra 2019 Fig. 5e
        patch     cells with Wnt above `w_on`: the region the mechanics reads as the crypt
    THE DECISION is the first row from which EVERY later row has at least one winner and no more than
    `win_max` of the cells winning -- a start where every cell's Delta overshoots before Notch catches up
    is not a choice, and a winner that is later lost is not one either. `decided` is 1 with such a row,
    else 0; `cells_at_decision` is the live cell count there (Serra Fig. 2f: the fates branch at the 16-32
    cell stage), `frame_at_decision` the row times `frames_per_row` when given.
    `winners_at_<k>`: winners on the first row with at least k cells (Serra Fig. 5e counts DLL1+ per
    organoid at the 8, 16, 32, 64-cell stages). `patch_frac_last`: the patch's share of the cells at the
    end (Yang 2021's crypt is about a fifth); `patch_parts_last`: its connected pieces (one crypt needs
    one)."""
    n = T.n_rows()
    ts = sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))
    nc, nw, pf, pp = [], [], [], []
    for t in ts:
        c = cells(T, t)
        ch = c.block(chem_block)
        if ch is None or ch.shape[1] <= max(d_col, w_col, n_col):
            raise ValueError(f"exp10.choice: needs cell block `{chem_block}` with columns {n_col}, {d_col}, {w_col}")
        win = (ch[:, d_col] > d_on) & (ch[:, n_col] < n_off)
        pat = ch[:, w_col] > w_on
        nc.append(len(c.x)); nw.append(int(win.sum())); pf.append(float(pat.mean()) if len(c.x) else 0.0)
        pp.append(len(components(len(c.x), neighbour_pairs(T, t, c), pat)) if pat.any() else 0)
    ok = [1 <= w <= win_max * m for w, m in zip(nw, nc)]
    i_dec = None
    for i in range(len(ts) - 1, -1, -1):
        if not ok[i]:
            break
        i_dec = i
    out = {"rows": ts, "decided": float(i_dec is not None), "cells_first": nc[0], "cells_last": nc[-1],
           "winners_last": nw[-1], "patch_frac_last": finite(pf[-1]), "patch_parts_last": pp[-1],
           "series_cells": nc, "series_winners": nw, "series_patch_frac": [finite(v) for v in pf],
           "series_patch_parts": pp}
    out["decision_row"] = None if i_dec is None else int(ts[i_dec])
    out["cells_at_decision"] = None if i_dec is None else int(nc[i_dec])
    out["winners_at_decision"] = None if i_dec is None else int(nw[i_dec])
    out["frame_at_decision"] = (None if i_dec is None or frames_per_row is None
                                else float(ts[i_dec] * frames_per_row))
    for k in checkpoints:
        j = next((i for i, m in enumerate(nc) if m >= k), None)
        out[f"winners_at_{k}"] = None if j is None else int(nw[j])
    return out


def solid_moments(P, es, et, ef, faces):
    """Volume, centre and covariance of the SOLID a closed ring mesh encloses, exactly: every half-edge
    fans a tetrahedron (o, c_f, p_s, p_t) as in `enclosed_volume`, and a tetrahedron (0, a, b, c) has
    integral x x^T dV = V/20 (a a^T + b b^T + c c^T + s s^T), s = a + b + c, first moment V s / 4."""
    live = np.isin(ef, faces)
    es, et, ef = es[live], et[live], ef[live]
    nF = int(ef.max()) + 1
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, P[es])
    cen /= np.maximum(cnt, 1)[:, None]
    o = P[np.unique(es)].mean(0)
    a, b, cc = cen[ef] - o, P[es] - o, P[et] - o
    V = np.einsum("ij,ij->i", a, np.cross(b, cc)) / 6.0
    sgn = 1.0 if V.sum() >= 0 else -1.0
    V = V * sgn
    ssum = a + b + cc
    M1 = (V[:, None] * ssum).sum(0) / 4.0
    outer = lambda u: u[:, :, None] * u[:, None, :]
    M2 = ((V / 20.0)[:, None, None] * (outer(a) + outer(b) + outer(cc) + outer(ssum))).sum(0)
    Vt = V.sum()
    mu = M1 / Vt
    return float(Vt), mu + o, M2 / Vt - np.outer(mu, mu)


def precrypt_row(T, t):
    """The pre-crypt organoid's three LSTree numbers on one row: the solid's axis ratio sqrt(l_min / l_max)
    of its covariance (1 a ball), the lumen's share of the organoid's volume, and the wall thickness over
    the organoid's radius, 1 - r / R, r and R the radii of balls of the lumen's and the organoid's volume."""
    c = cells(T, t)
    es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
    P = T.pos(t)
    S = T.vertex_block("sep", t)
    if S is None or S.shape != P.shape:
        return None
    va = enclosed_volume(P + S, es, et, ef, c.slot)
    vb = enclosed_volume(P - S, es, et, ef, c.slot)
    Vt, _, C = solid_moments(P - S if va <= vb else P + S, es, et, ef, c.slot)
    Vl = min(va, vb)
    w = np.linalg.eigvalsh(C)
    return dict(n_cells=len(c.x), axis_ratio=float(np.sqrt(max(w[0], 0) / w[2])), lumen_frac=float(Vl / Vt),
                thick_over_R=float(1 - (Vl / Vt) ** (1 / 3)))


def precrypt(T, n_lo=19, n_hi=24, **_):
    """`precrypt_row` averaged over the rows whose live cell count is in [n_lo, n_hi] -- LSTree's
    002-Budding window, 19-24 cells (`lstree_precrypt`). None when the run never passes through it."""
    rows = []
    for t in range(T.n_rows()):
        m = len(cells(T, t).x)
        if n_lo <= m <= n_hi:
            r = precrypt_row(T, t)
            if r is not None:
                rows.append(r)
        elif m > n_hi and rows:
            break
    out = {"n_rows": len(rows), "cells_first": len(cells(T, 0).x)}
    for k in ("axis_ratio", "lumen_frac", "thick_over_R"):
        out[k] = finite(np.mean([r[k] for r in rows])) if rows else None
    return out


def lstree_precrypt(root):
    """THE REAL ORGANOID BEFORE ITS CRYPT: LSTree (de Medeiros et al. 2022) example 002-Budding, read from
    its own files. Per timepoint T: cell count and mean neighbour count (features/T.csv, `cell` rows),
    lumen and epithelium volumes, and from lumen_segmentation/*T.tif (1 lumen, 2 epithelium; voxel
    spacing from experiment.json, z first) the organoid mask (label > 0) as a solid: its axis ratio
    sqrt(l_min / l_max) of the voxel coordinates' covariance, the lumen's share of its volume, and the
    wall thickness over radius 1 - (V_lumen / V_organoid)^(1/3) -- the quantities `precrypt` reads on a
    run. The solid, not its boundary voxels: at 2 um z spacing against 0.26 um in-plane the boundary's
    covariance is dominated by the flat top and bottom slices (0.70 on T0301 against the solid's 0.87)."""
    import csv
    import glob
    import json
    import os
    import tifffile
    sp = np.asarray(json.load(open(os.path.join(root, "experiment.json")))["spacing"], float)
    per = {}
    for f in sorted(glob.glob(os.path.join(root, "features", "*.csv"))):
        T = os.path.basename(f)[:-4]
        rows = list(csv.DictReader(open(f)))
        nb = [len(json.loads(r["feature_value"])) for r in rows
              if r["region"] == "cell" and r["feature_name"] == "neighbors"]
        n_cells = sum(1 for r in rows if r["region"] == "cell" and r["feature_name"] == "volume")
        tif = glob.glob(os.path.join(root, "lumen_segmentation", f"*{T}.tif"))
        if not tif:
            continue
        L = tifffile.imread(tif[0])
        X = np.argwhere(L > 0) * sp
        w = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T))
        vox = float(np.prod(sp))
        Vt, Vl = (L > 0).sum() * vox, (L == 1).sum() * vox
        per[T] = dict(n_cells=n_cells, neighbours=float(np.mean(nb)) if nb else None, V_organoid=Vt, V_lumen=Vl,
                      axis_ratio=float(np.sqrt(w[0] / w[2])), lumen_frac=Vl / Vt,
                      thick_over_R=float(1 - (Vl / Vt) ** (1 / 3)))
    vals = list(per.values())
    out = {"per_timepoint": per, "n_timepoints": len(vals)}
    for k in ("n_cells", "axis_ratio", "lumen_frac", "thick_over_R", "neighbours"):
        v = [p[k] for p in vals if p[k] is not None]
        out[f"{k}_mean"] = float(np.mean(v)) if v else None
        out[f"{k}_min"] = float(np.min(v)) if v else None
        out[f"{k}_max"] = float(np.max(v)) if v else None
    return out


register_run("exp10.mesh_sanity", mesh_sanity, None, "tools/mesh_sanity.py + the collapsed-cell fraction")
register_run("exp10.choice", choice, None,
             "Notch-Delta choice: decided, decision row / cell count, winners per size, the Wnt patch's share and pieces")
register_run("exp10.precrypt", precrypt, None,
             "the organoid over LSTree's 19-24-cell window: solid axis ratio, lumen fraction, wall thickness / radius")
register_run("exp10.crypt", crypt, None,
             "outward buds on a closed shell: count, depth/neck/width, on the patch, rest roughness, lumen")

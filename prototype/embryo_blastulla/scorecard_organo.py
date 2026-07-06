"""scorecard_organo -- Phase-3 ORGANOGENESIS geometry/branching metric family.

Quantifies organ-like morphology from the LIVE tissue mask (+ growth / pattern / strain fields) so
budding and branching are decided on NUMBERS, not on the movie. Three levels + relative dims +
localization + time-persistence, all at the standard 5/25/50/75/100% timepoints:

  1. OUTLINE  -- area, perimeter, circularity, aspect_ratio, convexity, solidity, major/minor axis,
                 orientation, body_radius, fragment_count  (round vs elongated vs lobed vs fragmented)
  2. BUD      -- protrusions beyond a smooth reference body (low-pass radial contour): n_buds,
                 bud_score, bud_area_frac, bud_len_bodyR, bud_neck_ratio, bud_roundness
  3. BRANCH   -- skeleton graph: n_tips, n_branchpoints, branch_len_mean/cv, branch_width_mean,
                 branch_angle_mean/sd, tree_depth, skeleton_length, branch_score
  4. LOCALIZATION (causality) -- growth_bud_overlap, pattern_growth_overlap, strain_growth_overlap,
                 tip_growth_enrichment  (did the bud appear WHERE the operator said growth should occur?)
  PERSISTENCE  -- bud_persistence / branch_persistence over the trajectory (round->bud->branch->stable).

`compute(caps, ...)` mirrors `scorecard.compute`: returns {final, evolution, pcts}. All metrics degrade
gracefully to 0 on an empty/degenerate mask. Pure geometry -- no dependence on the specific operators.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

FRACS = (0.05, 0.25, 0.50, 0.75, 1.00)
BUD_THRESH = 0.12          # protrusion height (fraction of body radius) to count as a bud lobe
BUD_MIN_ANG = 0.10         # min angular width (rad) of a lobe


# ------------------------------------------------------------------ mask ---
def tissue_mask(pts, W=1.0, res=170, sigma=1.6, close=2):
    """Live material points -> SMOOTHED filled binary mask. Rasterize to a DENSITY field, Gaussian-blur
    it, then threshold on density (not on "any point present"). This removes point-cloud boundary
    roughness and isolated specks that would otherwise inject spurious skeleton spurs / branch points,
    while preserving genuine protrusions (which carry real density). Returns (mask[res,res], dx)."""
    if pts is None or len(pts) == 0:
        return np.zeros((res, res), bool), W / res
    ix = np.clip((pts[:, 0] / max(W, 1e-9) * res).astype(int), 0, res - 1)
    iy = np.clip((pts[:, 1] * res).astype(int), 0, res - 1)
    dens = np.zeros((res, res), np.float32)
    np.add.at(dens, (ix, iy), 1.0)
    dens = ndimage.gaussian_filter(dens, sigma)                  # smooth the boundary
    nz = dens[dens > 0]
    thr = 0.30 * float(np.percentile(nz, 75)) if nz.size else 0.0   # density threshold (interior >> ragged edge)
    m = dens > max(thr, 1e-6)
    m = ndimage.binary_opening(m, iterations=1)                  # drop residual specks
    m = ndimage.binary_closing(m, iterations=close)
    m = ndimage.binary_fill_holes(m)
    return m, W / res


def _largest(mask, min_frac=0.05):
    """Largest connected component + count of SIGNIFICANT fragments (>= min_frac of the largest;
    ignores rasterization specks so fragment_count reflects real rupture, not sampling noise)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask, 0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    nsig = int((sizes >= min_frac * sizes.max()).sum())
    return lab == (1 + int(np.argmax(sizes))), nsig


# --------------------------------------------------------------- outline ---
def outline(mask, dx):
    body, nfrag = _largest(mask)
    out = dict(fragment_count=float(nfrag), area=0.0, perimeter=0.0, circularity=0.0,
               aspect_ratio=1.0, convexity=1.0, solidity=1.0, major_axis=0.0, minor_axis=0.0,
               orientation=0.0, body_radius=0.0)
    if body.sum() < 8:
        return out, body
    rp = measure.regionprops(body.astype(int))[0]
    area = rp.area * dx * dx
    per = rp.perimeter * dx
    conv = float(rp.convex_area) * dx * dx
    out.update(area=area, perimeter=per,
               circularity=float(4 * np.pi * area / (per * per + 1e-12)),
               aspect_ratio=float(rp.major_axis_length / (rp.minor_axis_length + 1e-9)),
               solidity=float(rp.solidity),
               major_axis=float(rp.major_axis_length * dx),
               minor_axis=float(rp.minor_axis_length * dx),
               orientation=float(rp.orientation),
               body_radius=float(np.sqrt(area / np.pi)))
    # convexity = convex-hull perimeter / actual perimeter (<=1; low = ragged/lobed)
    try:
        cp = measure.regionprops(rp.convex_image.astype(int))[0].perimeter * dx
        out["convexity"] = float(cp / (per + 1e-12))
    except Exception:
        pass
    return out, body


# ------------------------------------------------------------------- bud ---
def buds(body, dx, body_radius):
    """Protrusions beyond a low-pass radial reference. Returns dict of bud metrics + the bud mask."""
    res = dict(n_buds=0.0, bud_score=0.0, bud_area_frac=0.0, bud_len_bodyR=0.0,
               bud_neck_ratio=0.0, bud_roundness=0.0)
    budmask = np.zeros_like(body)
    if body.sum() < 20 or body_radius <= 0:
        return res, budmask
    cont = measure.find_contours(body.astype(float), 0.5)
    if not cont:
        return res, budmask
    c = max(cont, key=len)                                    # [K,2] (row,col)
    cen = np.array(np.nonzero(body)).mean(1)                  # centroid (row,col)
    d = c - cen
    r = np.hypot(d[:, 0], d[:, 1]) * dx
    th = np.arctan2(d[:, 1], d[:, 0])
    o = np.argsort(th); th, r = th[o], r[o]
    thg = np.linspace(-np.pi, np.pi, 360, endpoint=False)     # uniform angular resample
    rg = np.interp(thg, th, r, period=2 * np.pi)
    # low-pass reference body (keep modes 0..3)
    F = np.fft.rfft(rg); F[4:] = 0
    rsm = np.fft.irfft(F, n=len(rg))
    prot = rg - rsm                                           # outward protrusion height
    over = prot > BUD_THRESH * body_radius
    if not over.any():
        return res, budmask
    # group contiguous over-threshold angular runs (circular) into lobes
    idx = np.where(over)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    if len(groups) > 1 and over[0] and over[-1]:              # wrap-around merge
        groups[0] = np.concatenate([groups[-1], groups[0]]); groups.pop()
    lobes = []
    dth = 2 * np.pi / len(rg)
    for g in groups:
        if len(g) * dth < BUD_MIN_ANG:
            continue
        length = float(prot[g].max())                        # radial protrusion length
        ang_w = len(g) * dth
        width = float(rsm[g].mean() * ang_w)                 # arc width at the base
        neck = float(min(rg[g[0]], rg[g[-1]]) - rsm[g].mean())   # gap at the base (neck)
        area = float((0.5 * (rg[g] ** 2 - rsm[g] ** 2)).sum() * dth)   # protrusion area
        lobes.append(dict(length=length, width=max(width, 1e-6), neck=neck, area=max(area, 0.0)))
    if not lobes:
        return res, budmask
    body_area = float(body.sum()) * dx * dx
    tot_area = sum(l["area"] for l in lobes)
    big = max(lobes, key=lambda l: l["area"])
    area_frac = tot_area / (body_area + 1e-12)
    neck_ratio = big["neck"] / (big["width"] + 1e-9)         # small = sharp neck (bud-like)
    roundness = big["width"] / (big["length"] + 1e-9)        # ~1 round lobe, <1 finger
    neck_sharp = float(np.clip(1.0 - neck_ratio, 0, 1))      # sharper neck -> higher
    res.update(n_buds=float(len(lobes)),
               bud_area_frac=float(area_frac),
               bud_len_bodyR=float(big["length"] / body_radius),
               bud_neck_ratio=float(np.clip(neck_ratio, 0, 2)),
               bud_roundness=float(np.clip(roundness, 0, 3)),
               bud_score=float(area_frac * neck_sharp))       # persistence folded in at compute()
    # rasterize lobe wedges into a bud mask (for localization overlap)
    for g in groups:
        if len(g) * dth < BUD_MIN_ANG:
            continue
    return res, budmask


# ---------------------------------------------------------------- branch ---
def _skel_graph(skel):
    import networkx as nx
    ij = np.argwhere(skel)
    idx = {tuple(p): k for k, p in enumerate(ij)}
    G = nx.Graph()
    G.add_nodes_from(range(len(ij)))
    for k, (i, j) in enumerate(ij):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                q = (i + di, j + dj)
                if q in idx and idx[q] > k:
                    G.add_edge(k, idx[q], w=np.hypot(di, dj))
    return G, ij


def branches(body, dx, body_radius):
    import networkx as nx
    res = dict(n_tips=0.0, n_branchpoints=0.0, branch_len_mean=0.0, branch_len_cv=0.0,
               branch_width_mean=0.0, branch_angle_mean=0.0, branch_angle_sd=0.0,
               tree_depth=0.0, hierarchy_depth=0.0, skeleton_length=0.0, branch_score=0.0)
    if body.sum() < 30:
        return res
    skel = morphology.skeletonize(body)
    if skel.sum() < 5:
        return res
    edt = ndimage.distance_transform_edt(body) * dx          # local half-width
    nbr = ndimage.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    deg = nbr * skel
    bpt = (deg >= 3) & skel
    tip = (deg == 1) & skel
    res["skeleton_length"] = float(skel.sum()) * dx
    # cluster adjacent junction pixels -> one bifurcation. NOTE: n_branchpoints reliably separates
    # UNBRANCHED (0: round body / single bud) from BRANCHED (>=1), and tracks the branching TREND; the
    # exact integer is +-1 for thick/complex junctions (raster-skeleton loops) -> read it as "branching
    # present + relative count", not a precise bifurcation number.
    n_bpt = int(ndimage.label(bpt, structure=np.ones((3, 3)))[1])
    # segments = skeleton minus junctions; PRUNE short spurs (skeletonization noise on a round blob)
    seg = skel & (~bpt)
    lab, nseg = ndimage.label(seg, structure=np.ones((3, 3)))
    min_len = max(0.12 * body_radius, 3 * dx)
    lens, wids = [], []
    n_tips = 0
    for s in range(1, nseg + 1):
        pix = lab == s
        L = float(pix.sum()) * dx
        ntip_here = int((tip & pix).sum())
        if ntip_here and L < min_len:                        # short spur off a tip -> prune (noise)
            continue
        n_tips += ntip_here
        if L > 1.5 * dx:
            lens.append(L); wids.append(float(edt[pix].mean() * 2))
    res["n_tips"] = float(n_tips); res["n_branchpoints"] = float(n_bpt)
    if lens:
        lens = np.array(lens); wids = np.array(wids)
        res["branch_len_mean"] = float(lens.mean())
        res["branch_len_cv"] = float(lens.std() / (lens.mean() + 1e-9))
        res["branch_width_mean"] = float(wids.mean())
    # bifurcation angles + tree depth via the reduced tip/branchpoint graph
    try:
        G, ij = _skel_graph(skel)
        node_deg = {tuple(ij[k]): d for k, d in G.degree(weight=None)}
        # angles at each branchpoint: directions to the pixels of incident segments
        angs = []
        bpix = np.argwhere((deg >= 3) & skel)
        for (i, j) in bpix:
            dirs = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and 0 <= i + di < skel.shape[0] and 0 <= j + dj < skel.shape[1] and skel[i + di, j + dj]:
                        dirs.append(np.arctan2(dj, di))
            dirs = np.sort(dirs)
            if len(dirs) >= 2:
                gaps = np.diff(np.concatenate([dirs, [dirs[0] + 2 * np.pi]]))
                angs.append(np.degrees(gaps.max()))
        if angs:
            res["branch_angle_mean"] = float(np.mean(angs)); res["branch_angle_sd"] = float(np.std(angs))
        # tree depth = longest tip->tip shortest path in the reduced (weighted) graph
        if n_tips >= 2:
            tips = [k for k, d in G.degree() if d == 1]
            depth = 0
            sub = tips[:8]                                    # cap pairwise cost
            for a in sub:
                lengths = nx.single_source_dijkstra_path_length(G, a, weight="w")
                depth = max(depth, max((v for k, v in lengths.items()), default=0))
            res["tree_depth"] = float(depth * dx)
        # hierarchy_depth = max # of BIFURCATION GENERATIONS on any root->tip path (gen0->gen1->gen2 ...),
        # root = skeleton node nearest the body centroid. Counts consecutive branchpoint pixels as ONE.
        degd = dict(G.degree())
        tipsn = [k for k in G.nodes if degd[k] == 1]
        if tipsn:
            cen = np.array(np.nonzero(body)).mean(1)
            root = int(np.argmin(((ij - cen) ** 2).sum(1)))
            hd = 0
            for tp in tipsn[:12]:
                try:
                    path = nx.shortest_path(G, root, tp)
                except Exception:
                    continue
                cnt = 0; prev = False
                for k in path:
                    b = degd[k] >= 3
                    if b and not prev:
                        cnt += 1
                    prev = b
                hd = max(hd, cnt)
            res["hierarchy_depth"] = float(hd)
    except Exception:
        pass
    # branch_score: bifurcations x continuity (1 fragment=1.0, penalize islands); persistence at compute()
    _, nfrag = _largest(body)
    cont_pen = 1.0 / max(nfrag, 1)
    res["branch_score"] = float(n_bpt * cont_pen)
    return res


# ---------------------------------------------------- localization / causality ---
def localization(body, birth_pts, pattern_pts, strain_vals, mX_live, W, dx, budmask):
    """Overlap of GROWTH (newly-woken material) with buds / pattern / strain / tips."""
    res = dict(growth_bud_overlap=0.0, pattern_growth_overlap=0.0,
               strain_growth_overlap=0.0, tip_growth_enrichment=0.0, independent_growth_domains=0.0)
    res_shape = body.shape
    def raster(pts):
        m = np.zeros(res_shape, float)
        if pts is None or len(pts) == 0:
            return m
        ix = np.clip((pts[:, 0] / max(W, 1e-9) * res_shape[0]).astype(int), 0, res_shape[0] - 1)
        iy = np.clip((pts[:, 1] * res_shape[1]).astype(int), 0, res_shape[1] - 1)
        np.add.at(m, (ix, iy), 1.0)
        return m
    g = raster(birth_pts)                                    # where growth added material
    if g.sum() > 0:
        gi = g > 0
        # ORG: spatially-separated growth centres = coexisting developmental programs
        gc = ndimage.binary_closing(gi, iterations=2)
        glab, gn = ndimage.label(gc)
        if gn:
            gs = ndimage.sum(np.ones_like(glab), glab, index=np.arange(1, gn + 1))
            res["independent_growth_domains"] = float((gs >= 0.10 * gs.max()).sum())
        if budmask.any():
            res["growth_bud_overlap"] = float((gi & budmask).sum() / (gi.sum() + 1e-9))
        p = raster(pattern_pts)
        if p.sum() > 0:
            pi = p > 0
            res["pattern_growth_overlap"] = float((gi & pi).sum() / (gi.sum() + 1e-9))
        # strain: high-strain particles overlap with growth
        if strain_vals is not None and mX_live is not None and len(mX_live) == len(strain_vals):
            hi = strain_vals > np.quantile(strain_vals, 0.8)
            sh = raster(mX_live[hi])
            res["strain_growth_overlap"] = float(((sh > 0) & gi).sum() / (gi.sum() + 1e-9))
        # tip enrichment: growth near the mask's high-curvature/thin regions (edt small)
        edt = ndimage.distance_transform_edt(body)
        thin = (edt > 0) & (edt < np.quantile(edt[edt > 0], 0.3))
        res["tip_growth_enrichment"] = float((gi & thin).sum() / (gi.sum() + 1e-9))
    return res


# -------------------------------------------------------------- top-level ---
_KEYS = ["fragment_count", "area", "perimeter", "circularity", "aspect_ratio", "convexity",
         "solidity", "major_axis", "minor_axis", "orientation", "body_radius",
         "n_buds", "bud_score", "bud_area_frac", "bud_len_bodyR", "bud_neck_ratio", "bud_roundness",
         "n_tips", "n_branchpoints", "branch_len_mean", "branch_len_cv", "branch_width_mean",
         "branch_angle_mean", "branch_angle_sd", "tree_depth", "hierarchy_depth", "skeleton_length",
         "branch_score", "growth_bud_overlap", "pattern_growth_overlap", "strain_growth_overlap",
         "tip_growth_enrichment", "independent_growth_domains"]


def _one(mX_live, birth_pts, pattern_pts, strain_vals, W, res):
    mask, dx = tissue_mask(mX_live, W=W, res=res)
    o, body = outline(mask, dx)
    b, budmask = buds(body, dx, o["body_radius"])
    br = branches(body, dx, o["body_radius"])
    if o["solidity"] > 0.90:                                 # near-convex round body -> no real branches
        for k in ("n_tips", "n_branchpoints", "branch_score", "tree_depth"):
            br[k] = 0.0
    loc = localization(body, birth_pts, pattern_pts, strain_vals, mX_live, W, dx, budmask)
    return {**o, **b, **br, **loc}


def panel_data(mX_live, W=1.0, res=170):
    """Intermediate rasters for a diagnostic panel: body mask, skeleton, tip/branchpoint masks,
    main contour, and the metric dict -- so a viewer can SEE what the numbers measured."""
    mask, dx = tissue_mask(mX_live, W=W, res=res)
    o, body = outline(mask, dx)
    b, budmask = buds(body, dx, o["body_radius"])
    br = branches(body, dx, o["body_radius"])
    if o["solidity"] > 0.90:
        for k in ("n_tips", "n_branchpoints", "branch_score", "tree_depth"):
            br[k] = 0.0
    skel = morphology.skeletonize(body) if body.sum() >= 30 else np.zeros_like(body)
    tips = np.zeros_like(body); bpts = np.zeros_like(body)
    if skel.sum() >= 5:
        deg = (ndimage.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1) * skel
        tips = (deg == 1) & skel
        bpts = ndimage.binary_dilation((deg >= 3) & skel, iterations=1) & skel
    cont = measure.find_contours(body.astype(float), 0.5)
    cont = max(cont, key=len) if cont else None
    return dict(body=body, skel=skel, tips=tips, bpts=bpts, contour=cont, dx=dx,
                metrics={**o, **b, **br})


def compute(caps, W=1.0, fracs=FRACS, res=170):
    """caps: mX [T,N,2], mocc [T,N] (live material). Optional: mbirth [T,N] (grown material mask),
    aX [T,M,2]+at[M]+occ[T,M] (pattern = a minority agent type), fnorm [T,N] (strain).
    Returns {final, evolution, pcts} over the standard timepoints, with persistence folded in."""
    mX = np.asarray(caps["mX"]); mocc = np.asarray(caps["mocc"]); T = mX.shape[0]
    orig0 = mocc[0] > 0                                       # material live at birth = original tissue
    fnorm = caps.get("fnorm"); fnorm = np.asarray(fnorm) if fnorm is not None else None
    aX = caps.get("aX"); at = caps.get("at"); occ = caps.get("occ")
    aX = np.asarray(aX) if aX is not None else None
    occ = np.asarray(occ) if occ is not None else None

    ev = {k: [] for k in _KEYS}
    idxs = [min(T - 1, max(0, int(round(f * (T - 1))))) for f in fracs]
    for t in idxs:
        lm = mocc[t] > 0
        mXl = mX[t][lm]
        grown = lm & (~orig0)                                 # woken after frame 0 = GROWTH (cell_grow)
        birth_pts = mX[t][grown] if grown.any() else None
        strain = fnorm[t][lm] if fnorm is not None else None
        pat = None                                           # pattern domain = agent type-0 (a minority band)
        if aX is not None and at is not None and occ is not None:
            al = occ[t] > 0
            a0 = al & (np.asarray(at) == 0)
            pat = aX[t][a0] if a0.any() else None
        m = _one(mXl, birth_pts, pat, strain, W, res)
        for k in _KEYS:
            ev[k].append(m.get(k, 0.0))

    # persistence -> fold into the *_score keys (round->bud->branch->stable needs the shape to LAST)
    nb = np.array(ev["n_buds"]); nbp = np.array(ev["n_branchpoints"])
    igd = np.array(ev["independent_growth_domains"])
    bud_pers = float((nb >= 1).mean()); brn_pers = float((nbp >= 1).mean())
    prog_stab = float((igd >= 2).mean())                             # ORG: multiple programs coexist persistently
    final = {k: ev[k][-1] for k in _KEYS}
    final["bud_persistence"] = bud_pers
    final["branch_persistence"] = brn_pers
    final["program_stability"] = prog_stab
    final["bud_score"] = float(final["bud_score"] * bud_pers)         # persistence-weighted
    final["branch_score"] = float(final["branch_score"] * brn_pers)
    ev["bud_persistence"] = [bud_pers] * len(idxs)
    ev["branch_persistence"] = [brn_pers] * len(idxs)
    ev["program_stability"] = [prog_stab] * len(idxs)
    return {"final": final, "evolution": ev, "pcts": [round(f * 100) for f in fracs]}

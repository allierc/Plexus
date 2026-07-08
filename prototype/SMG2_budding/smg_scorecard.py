"""smg_scorecard -- quantitative assessment of the SMG2 topology/shape model.

Follows the embryo_blastulla convention (embryo_metrics.py / scorecard.py):
a family of documented OBSERVABLES, not one scalar loss; reported as `final`
plus `evolution` across the movie so trends/transients are visible; we decide
on NUMBERS, not on visual captions.

Ground truth is annotated on only 2 frames, so assessment has two arms:
  A. GT AGREEMENT at the anchor frames (tube+branch exact-ish, bud +/-20%).
  B. GT-FREE checks on all 553 frames (temporal consistency + geometric
     fidelity of the primitive model) -> evidence the counts are right where
     no GT exists.

Model output per analyzed frame (from smg_topo.fit):
  dict(frame, n_tube, n_branch, n_bud, n_tips, centerline_len,
       centers[K,3], radii[K], occ_volume, vox, density_grid?, ...)
Points per frame are the (N,3) nuclei used for fidelity.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

# GT annotation: frame -> (n_bud, n_branch, n_tube)
GT_ANCHORS = {0: (4, 2, 1), 552: (20, 3, 1)}
TOL = dict(bud_rel=0.20, branch_abs=1, tube_abs=0)   # confirmed pass criteria


# ============================================================ A. GT agreement
def gt_agreement(results_by_frame, anchors=GT_ANCHORS, tol=TOL):
    """Compare predicted counts to GT at the annotated frames.

    results_by_frame: {frame_index: {'n_bud':.., 'n_branch':.., 'n_tube':..}}
    Returns per-anchor errors + pass flags and an overall topo_accuracy
    (fraction of the anchor targets met within tolerance).
    """
    out = {}
    hits = tot = 0
    for f, (gb, gbr, gt) in anchors.items():
        if f not in results_by_frame:
            continue
        r = results_by_frame[f]
        pb, pbr, pt = r["n_bud"], r["n_branch"], r["n_tube"]
        bud_ok = abs(pb - gb) <= tol["bud_rel"] * gb
        br_ok = abs(pbr - gbr) <= tol["branch_abs"]
        tube_ok = abs(pt - gt) <= tol["tube_abs"]
        out[f"f{f}"] = dict(
            bud=(pb, gb, round(abs(pb - gb) / max(gb, 1), 3), bool(bud_ok)),
            branch=(pbr, gbr, bool(br_ok)),
            tube=(pt, gt, bool(tube_ok)),
        )
        hits += int(bud_ok) + int(br_ok) + int(tube_ok)
        tot += 3
    out["topo_accuracy"] = round(hits / max(tot, 1), 3)
    return out


# ================================================= B1. temporal consistency
def temporal_consistency(frames, n_bud, n_branch, n_tube):
    """GT-free sanity of the count time-series over the whole movie."""
    frames = np.asarray(frames, float)
    n_bud = np.asarray(n_bud, float)
    n_branch = np.asarray(n_branch, float)
    n_tube = np.asarray(n_tube, float)
    rho = spearmanr(frames, n_bud).statistic if len(frames) > 2 else np.nan
    dbud = np.abs(np.diff(n_bud)) if len(n_bud) > 1 else np.array([0.0])
    dbr = np.diff(n_branch) if len(n_branch) > 1 else np.array([0.0])
    return dict(
        bud_trend=round(float(rho), 3),                                   # want strongly +
        bud_growth_ratio=round(float(n_bud[-1] / max(n_bud[0], 1)), 2),   # want ~5
        branch_monotonic=round(float((dbr >= 0).mean()), 3),              # want ~1
        count_jitter=round(float(np.median(dbud)), 3),                    # want low
        tube_stability=round(float((n_tube == 1).mean()), 3),            # want ~1
    )


# ================================================= B2. geometric fidelity
def geometric_fidelity(pts, centers, radii, extra=None):
    """How well the union-of-inscribed-ellipsoids explains the point cloud.

    coverage : fraction of nuclei inside some primitive (want ~0.95)
    chamfer  : mean point stick-out beyond nearest primitive surface, world
               units (want ~ one nucleus radius)
    n_primitives / compression : parsimony
    Any precomputed values (volume_ratio, density_corr) merged via `extra`.
    """
    out = {}
    if len(centers) and len(pts):
        tree = cKDTree(np.asarray(centers))
        d, idx = tree.query(np.asarray(pts), k=1)
        r_near = np.asarray(radii)[idx]
        stick = np.maximum(0.0, d - r_near)
        out["coverage"] = round(float((d <= r_near).mean()), 4)
        out["chamfer"] = round(float(stick.mean()), 3)
        out["chamfer_p95"] = round(float(np.percentile(stick, 95)), 3)
        out["n_primitives"] = int(len(centers))
        out["compression"] = round(float(len(centers) / max(len(pts), 1)), 4)
    if extra:
        out.update({k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in extra.items()})
    return out


# ============================================================ aggregate
FRACS = [0.05, 0.25, 0.50, 0.75, 1.00]
PCTS = [5, 25, 50, 75, 100]


def compute(per_frame, points_by_frame=None, anchors=GT_ANCHORS):
    """Assemble the scorecard from a time-ordered list of per-frame results.

    per_frame: list of dicts sorted by frame, each with the topology counts and
               (optionally) centers/radii for fidelity; points_by_frame maps
               frame_index -> (N,3) for fidelity at those frames.
    Returns {agreement, temporal, fidelity_final, fidelity_evolution, pcts}.
    """
    frames = [r["frame"] for r in per_frame]
    by_frame = {r["frame"]: r for r in per_frame}
    agreement = gt_agreement(by_frame, anchors)
    temporal = temporal_consistency(
        frames, [r["n_bud"] for r in per_frame],
        [r["n_branch"] for r in per_frame], [r["n_tube"] for r in per_frame])

    fid_evo = None
    fid_final = None
    if points_by_frame is not None:
        snaps = []
        T = len(per_frame)
        for fr in FRACS:
            i = min(T - 1, max(0, int(round(fr * (T - 1)))))
            r = per_frame[i]
            f = r["frame"]
            if "centers" in r and f in points_by_frame:
                snaps.append(geometric_fidelity(
                    points_by_frame[f], r["centers"], r["radii"],
                    extra=r.get("fidelity_extra")))
            else:
                snaps.append({})
        keys = []
        for s in snaps:
            for k in s:
                if k not in keys:
                    keys.append(k)
        fid_evo = {k: [s.get(k) for s in snaps] for k in keys}
        fid_final = snaps[-1]
    return dict(agreement=agreement, temporal=temporal,
                fidelity_final=fid_final, fidelity_evolution=fid_evo, pcts=PCTS)

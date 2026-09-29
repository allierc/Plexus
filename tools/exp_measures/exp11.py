"""exp11 bm_hole_budding -- the rulers: does the bud grow on the hole's axis, and only there?

    exp11.bud          bud_excess along each declared axis (the archive's metric, reused verbatim)
    exp11.balance      the coupling's force ledger: does every interface sum to zero?
    exp11.smg_buds     bud count by the SMG2 data's own counter, on the model's cells
    exp11.surface      the surface layer's growth: mid-slice outline length fitted as Wang 2021 fit Fig 1H,
                       plus the 3D surface area and the cell count over the same window (Phase 2)
    exp11.persist      the bud sustained: bud_excess on the hole's axis at the last row over its peak (Phase 2)

BUD_EXCESS IS THE ARCHIVE'S DEFINITION, IMPORTED, NOT REWRITTEN. `discovery_okuda/ops/budding_metric.py`
`profile(x, c, d)` returns `bud_index = h_max / R_med - 1` along +d, where h_max is the largest
projection of a tissue vertex onto the unit axis d from the tissue's centroid and R_med the median
vertex radius. bud_excess = bud_index(+d) - bud_index(-d): 0 for a sphere of any size and for a
whole-body elongation, negative when the protrusion is at the wrong pole. The archive's noise floor
is +-0.04 over 80 tissues and its hole runs read +0.975 on axis and -0.055 off it
(`discovery_okuda/BUDDING_08.md` lines 11-14). Using the same function is what makes the live runs
comparable with those numbers.

THE AXIS IS DECLARED BY THE ARM, never inferred from the shape: an arm's gates.yaml entry passes
`axes: {name: [x, y, z]}`, the direction from the tissue centre through the hole's centre, so a
mirrored-hole arm is measured along both the original and the rotated axis and the bud has to be on
the right one.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from .common import ROOT, cells, finite, register_run

_BM = os.path.join(ROOT, "discovery_okuda", "ops", "budding_metric.py")


def _profile():
    spec = importlib.util.spec_from_file_location("budding_metric", _BM)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.profile


def _rows(T, every):
    n = T.n_rows()
    return sorted(set(list(range(0, n, max(1, int(every)))) + [n - 1]))


def bud(T, axes=None, every=20, **_):
    """bud_excess (+d minus -d) and neck ratio along each named unit axis, first/last/max over rows."""
    if not axes:
        raise ValueError("exp11.bud needs `axes: {name: [x, y, z]}` -- the direction through the hole")
    profile = _profile()
    out, ts = {}, _rows(T, every)
    for name, d in axes.items():
        d = np.asarray(d, float)
        d = d / np.linalg.norm(d)
        ex, nk = [], []
        for t in ts:
            x = T.pos(t)
            x = x[np.isfinite(x).all(1)]
            c = x.mean(0)
            b, neck, *_r = profile(x, c, d)
            bf, *_r = profile(x, c, -d)
            ex.append(b - bf)
            nk.append(neck)
        out[f"{name}.excess_last"] = finite(ex[-1])
        out[f"{name}.excess_max"] = finite(np.nanmax(ex))
        out[f"{name}.neck_last"] = finite(nk[-1])
        out[f"{name}.series"] = [finite(v) for v in ex]
    out["rows"] = ts
    return out


def balance(T, sum_scalar="interface_force_sum", max_scalar="interface_force_max", **_):
    """max over rows of |net interface force| / largest single contact force, from the coupling's own
    per-row scalars (`<set>__mesh_scalar_<name>`). The coupling operator must write both; a run
    without them reads `available: False` and the gate that needs it cannot be scored."""
    s = [T.scalar(sum_scalar, t) for t in range(T.n_rows())]
    m = [T.scalar(max_scalar, t) for t in range(T.n_rows())]
    if any(v is None for v in s) or any(v is None for v in m):
        return {"available": False}
    r = np.abs(np.asarray(s, float)) / np.maximum(np.asarray(m, float), 1e-30)
    return {"available": True, "ratio_max": finite(r.max())}


def smg_buds(T, um_per_unit=None, frames=("first", "last"), **_):
    """Buds counted by `prototype/SMG2_budding/smg_topo.analyze_frame` -- the counter the SMG2 ground
    truth was scored with (4 buds at frame 0, 20 at frame 552, `smg_scorecard.GT_ANCHORS`) -- on the
    model's cell centroids scaled to micrometres (`um_per_unit`, else the spec's `units.length_um`)."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "prototype", "SMG2_budding"))
    import smg_topo
    from .common import cells
    s = um_per_unit or ((T.spec.get("general") or {}).get("units") or {}).get("length_um")
    if not s:
        return {"available": False, "why": "no micrometre scale: pass um_per_unit or declare units.length_um"}
    idx = {"first": 0, "last": T.n_rows() - 1}
    out = {"available": True}
    for f in frames:
        t = idx.get(f, f)
        r = smg_topo.analyze_frame(cells(T, int(t)).x * float(s))
        out[f"n_bud_{f}"] = int(r["n_buds"])
    if "n_bud_first" in out and "n_bud_last" in out:
        out["bud_ratio"] = finite(out["n_bud_last"] / max(out["n_bud_first"], 1))
    return out


register_run("exp11.bud", bud, None, "bud_excess along each declared hole axis (archive metric)")
register_run("exp11.balance", balance, "fraction", "interface force ledger, |sum| / max")
register_run("exp11.smg_buds", smg_buds, "count", "bud count by the SMG2 counter")


# ============================================================================ Phase 2 rulers
def _slice_length(P, es, et, ef, nF, c, n):
    """Total length of the curve where the plane through c with unit normal n cuts the surface:
    each face fanned from its vertex mean into triangles (centroid, srce, trgt), each triangle that
    straddles the plane contributing the segment between its two crossing points. Folds and clefts
    that cross the plane count, as they do in a traced outline."""
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, P[es])
    cen /= np.maximum(cnt, 1)[:, None]
    tri = np.stack([cen[ef], P[es], P[et]], 1)                      # [E, 3 corners, 3]
    d = (tri - c) @ n                                               # signed heights [E, 3]
    L = 0.0
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        cross = (da * db) < 0
        t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
        pts.append((cross, tri[:, a] + t[:, None] * (tri[:, b] - tri[:, a])))
    m = np.stack([p[0] for p in pts], 1)                            # [E, 3] which edges cross
    two = m.sum(1) == 2
    if two.any():
        X = np.stack([p[1] for p in pts], 1)[two]                   # [k, 3 edges, 3]
        mm = m[two]
        first = np.argmax(mm, 1)
        last = 2 - np.argmax(mm[:, ::-1], 1)
        k = np.arange(len(X))
        L = float(np.linalg.norm(X[k, first] - X[k, last], axis=1).sum())
    return L


def _fan_area(P, es, et, ef, nF):
    cnt = np.bincount(ef, minlength=nF).astype(float)
    cen = np.zeros((nF, 3))
    np.add.at(cen, ef, P[es])
    cen /= np.maximum(cnt, 1)[:, None]
    return float(0.5 * np.linalg.norm(np.cross(P[es] - cen[ef], P[et] - cen[ef]), axis=1).sum())


def _fit_ratio(hours, y):
    """Wang 2021's Fig 1H fit (`SMG-dynamic-peripheral-line-scan.ipynb`, plot_perimeter_and_nuclei_number):
    y normalised by its 90th percentile, a least-squares line against hours, and the ratio of the
    fitted value at the window's last point to the fitted value at its first. Returns (ratio, r2, slope
    in normalised units per hour)."""
    y = np.asarray(y, float)
    y = y / np.nanpercentile(y, 90)
    A = np.vstack([hours, np.ones_like(hours)]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a * hours + b
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    first, last = b, a * hours[-1] + b
    return (last / first if first > 0 else float("nan")), r2, a


def surface(T, window_h=12.5, normal=(0.0, 1.0, 0.0), every=1, **_):
    """The surface layer's growth over the first `window_h` hours of the run (Wang Fig 1H's window):

        perim_ratio   the mid-slice outline length, fitted as Wang fit Fig 1H -- the plane through the
                      tissue's centroid with unit `normal` (default y, which contains both hole axes, z
                      and x), the length normalised by its 90th percentile, a line against hours, the
                      fitted last over the fitted first
        perim_r2      that fit's r^2 (Wang's 13 glands: 0.80-0.99, all linear)
        area_ratio    the 3D surface area (the faces fanned from their vertex means), last row of the
                      window over the first
        cells_ratio   live cells, last row of the window over the first
        hours         the window actually covered (short when the run is shorter than `window_h`)

    Hours come from the spec's `general.units.time_s` x `general.dt` per frame, and the rows are the
    trajectory's own (every recorded row inside the window, stepped by `every`)."""
    g = (T.spec.get("general") or {})
    ts = float((g.get("units") or {}).get("time_s") or 0.0)
    if ts <= 0:
        return {"available": False, "why": "no time scale: declare general.units.time_s"}
    dt_h = ts * float(g.get("dt", 1.0)) / 3600.0
    frames = _row_frames(T)
    rows = [r for r, f in enumerate(frames) if f * dt_h <= window_h + 1e-9][::max(1, int(every))]
    if len(rows) < 3:
        return {"available": False, "why": f"fewer than 3 rows inside {window_h} h"}
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    L, A, N = [], [], []
    for t in rows:
        P = np.asarray(T.pos(t), float)
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        nF = int(T.nF(t))
        c = P[np.isfinite(P).all(1)].mean(0)
        L.append(_slice_length(P, es, et, ef, nF, c, n))
        A.append(_fan_area(P, es, et, ef, nF))
        N.append(len(cells(T, t)))
    hours = np.asarray([frames[t] * dt_h for t in rows], float)
    ratio, r2, slope = _fit_ratio(hours, L)
    return {"available": True, "perim_ratio": finite(ratio), "perim_r2": finite(r2),
            "perim_slope_per_h": finite(slope), "perim_first": finite(L[0]), "perim_last": finite(L[-1]),
            "area_ratio": finite(A[-1] / max(A[0], 1e-12)), "cells_ratio": finite(N[-1] / max(N[0], 1)),
            "hours": finite(hours[-1])}


def _row_frames(T):
    """The engine frame of every recorded row: the trajectory's own `frame` record when it has one,
    else the row index times the recording stride n_frames / (rows - 1)."""
    n = T.n_rows()
    fr = getattr(T, "frames_of_rows", None)
    if callable(fr):
        try:
            return [int(v) for v in fr()]
        except Exception:                                                    # noqa: BLE001
            pass
    nf = int((T.spec.get("general") or {}).get("n_frames") or (n - 1))
    step = nf / max(n - 1, 1)
    return [int(round(r * step)) for r in range(n)]


def persist(T, axis=None, every=5, **_):
    """Is the bud sustained or receding? bud_excess on the declared hole axis (exp11.bud's metric) every
    `every` rows: `ratio` = the last row's over the peak's (1 = still at its peak; round A read 0.45-0.75),
    `peak`, `last`, `peak_row`. A peak under the noise floor 0.04 reads `ratio` 1 with `grown` False:
    nothing grew, so nothing receded."""
    if axis is None:
        raise ValueError("exp11.persist needs `axis: [x, y, z]` -- the direction through the hole")
    r = bud(T, axes={"a": axis}, every=every)
    ser = np.asarray(r["a.series"], float)
    k = int(np.nanargmax(ser))
    peak, last = float(ser[k]), float(ser[-1])
    grown = peak > 0.04
    return {"peak": finite(peak), "last": finite(last), "peak_row": int(r["rows"][k]), "grown": bool(grown),
            "ratio": finite(last / peak if grown else 1.0)}


register_run("exp11.surface", surface, None, "surface layer growth over Wang Fig 1H's window")
register_run("exp11.persist", persist, "fraction", "bud_excess last / peak on the hole axis")


def membrane(T, set_name="bm_node", **_):
    """Is the membrane still there? Over every row: `finite_min`, the smallest fraction of live membrane
    nodes with finite positions (batch 2 of Phase 2 lost every node to an unstable spring step by row 60
    and its tissues grew free -- Finding 38), and `r_ratio_last`, the membrane's median radius over the
    tissue's at the last row, about the tissue's centroid (≥ ~1: the membrane outside the layer)."""
    z = getattr(T, "z", None)
    k = f"{set_name}__pos"
    if z is None or k not in z.files:
        return {"available": False, "why": f"no {k} in the trajectory"}
    P = z[k]
    ko = f"{set_name}__occ"
    occ = z[ko] if ko in z.files else None
    fmin, rr = 1.0, float("nan")
    for t in range(T.n_rows()):
        p = np.asarray(P[t], float)
        if occ is not None:
            p = p[np.asarray(occ[t]) > 0.5]
        if len(p) == 0:
            continue
        fin = np.isfinite(p).all(1)
        fmin = min(fmin, float(fin.mean()))
        if t == T.n_rows() - 1:
            x = np.asarray(T.pos(t), float)
            x = x[np.isfinite(x).all(1)]
            c = x.mean(0)
            rt = float(np.median(np.linalg.norm(x - c, axis=1)))
            rr = float(np.median(np.linalg.norm(p[fin] - c, axis=1)) / max(rt, 1e-12)) if fin.any() else float("nan")
    return {"available": True, "finite_min": finite(fmin), "r_ratio_last": finite(rr)}


register_run("exp11.membrane", membrane, "fraction", "membrane nodes finite (min over rows), radius vs tissue")


def clefts(T, every=10, thresh=0.1, **_):
    """Inward folds of the layer (Wang 2021 Fig 1J reads clefts as negative curvature of the outline). Per recorded row
    (every `every`), the fraction of mesh vertices whose umbrella vector -- the mean of its ring neighbours minus the
    vertex -- points OUTWARD (along the vertex's direction from the tissue centroid) by more than `thresh` median edge
    lengths: a vertex sitting inward of its neighbours, i.e. at the bottom of a fold. 0 on a smooth convex shell.
    `frac_last`, `frac_max`, and the series."""
    rows = _rows(T, every)
    out = []
    for t in rows:
        P = np.asarray(T.pos(t), float)
        es, et, _ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        n = len(P)
        cnt = np.bincount(np.r_[es, et], minlength=n).astype(float)
        s = np.zeros((n, 3))
        np.add.at(s, es, P[et]); np.add.at(s, et, P[es])
        mean_nb = s / np.maximum(cnt, 1)[:, None]
        c = P[np.isfinite(P).all(1)].mean(0)
        u = P - c
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        L = np.median(np.linalg.norm(P[et] - P[es], axis=1))
        h = ((mean_nb - P) * u).sum(1) / max(L, 1e-12)
        out.append(float(np.mean(h > thresh)))
    return {"frac_last": finite(out[-1]), "frac_max": finite(max(out)), "series": [finite(v) for v in out],
            "rows": rows}


register_run("exp11.clefts", clefts, "fraction", "fraction of layer vertices at the bottom of an inward fold")


def prw_fit(lag_h, msd):
    """The persistent random walk fitted to a mean squared displacement curve, by least squares:

        MSD(t) = 2 v^2 P (t - P (1 - exp(-t / P))) + c

    (Furth 1920; the 3D form with <v(0).v(t)> = v^2 exp(-t / P)), over a grid of P with (v^2, c) in
    closed form. `c` = 6 sigma^2 is the positional noise of a tracked spot (sigma per axis): it adds the
    same offset at every lag, and without it the first lags read as a fast, unpersistent walk. ONE
    ESTIMATOR FOR THE DATA AND THE MODEL: `tools/wang_smg_stats.py` fits Wang 2021's nucleus tracks with
    this same function, so a model's (v, P) is compared with the gland's through identical arithmetic.
    Returns v (length / h), P (h), r2, noise_sigma (length)."""
    lag_h = np.asarray(lag_h, float); msd = np.asarray(msd, float)
    best = (np.inf, np.nan, np.nan, 0.0)
    for P in np.geomspace(0.02, 20.0, 400):
        g = 2.0 * P * (lag_h - P * (1.0 - np.exp(-lag_h / P)))
        A = np.stack([g, np.ones_like(g)], 1)
        (v2, c), *_ = np.linalg.lstsq(A, msd, rcond=None)
        if v2 < 0 or c < 0:
            v2 = max(float((g * msd).sum() / max((g * g).sum(), 1e-30)), 0.0); c = 0.0
        r = float(((msd - v2 * g - c) ** 2).sum())
        if r < best[0]:
            best = (r, float(np.sqrt(max(v2, 0.0))), float(P), float(c))
    ss = float(((msd - msd.mean()) ** 2).sum())
    return {"v": best[1], "P": best[2], "r2": 1.0 - best[0] / max(ss, 1e-30),
            "noise_sigma": float(np.sqrt(max(best[3], 0.0) / 6.0))}


def cell_motion(T, cell_set="icell", point_set="ipt", max_lag=24, um_per_unit=None, min_per_frame=None, **_):
    """How the MPM cells move and deform, in the terms Wang 2021's nucleus tracks give (tools/wang_smg_stats.py,
    experiments/exp11_bm_hole_budding/data/wang_smg_cell_stats.json), so the two sit in one table.

    Tracks are the cells' centroids (`<cell_set>__pos`, kept by `aggregate_centroid`) at every recorded row,
    with the population's drift removed (per row, the median step of every cell), exactly as the data's.
      speed_um_h_median / _iqr    instantaneous speed, step length over the row interval
      track_speed_cv              the spread of per-cell mean speeds across cells (Wang interior: 0.17)
      turn_deg_median             angle between successive steps of one cell
      prw_v_um_h, prw_P_h, prw_r2 `prw_fit` on the drift-corrected MSD over lags 1..`max_lag` rows
      msd_um2                     the MSD curve itself (first 12 lags)
      aspect_median, aspect_cv    each cell's shape: sqrt(largest / smallest eigenvalue) of its points'
                                  gyration tensor, median over cells and rows, and its CV over rows within
                                  a cell (a cell that changes shape as it moves reads > 0)
      align_median                |cos| between a cell's long axis and its step (0.58 for no relation in 3D)
      integrity_max               the largest point-to-centroid distance over the cell's row-0 rms radius,
                                  max over cells and rows: ~1.3 for an intact ball, > 3 a smeared or torn cell
      nn_um_median                centre-to-centre nearest-neighbour distance at row 0 (Wang interior: 9.4 um)
      neighbour_kept              the fraction of row-0 neighbour pairs (within 1.25 x that distance) still
                                  within 1.5 x at the last row: 1 for a welded mass, falling as cells rearrange;
                                  None when the cells are too sparse to have neighbours
    Units: `general.units` of the run (length_um, time_s) unless `um_per_unit` / `min_per_frame` are given."""
    z = T.z
    key = f"{cell_set}__pos"
    n = T.n_rows()
    C = np.array([np.asarray(z[key][t], float) for t in range(n)])
    try:
        P = np.array([np.asarray(z[f"{point_set}__pos"][t], float) for t in range(n)])
    except Exception:                                                        # noqa: BLE001
        P = None
    u = (T.spec.get("general") or {}).get("units") or {}
    um = float(um_per_unit if um_per_unit is not None else u.get("length_um", 1.0))
    fr = _row_frames(T)
    dt_frame = float((T.spec.get("general") or {}).get("dt", 1.0))
    mpf = float(min_per_frame if min_per_frame is not None else u.get("time_s", 60.0) * dt_frame / 60.0)
    dt_h = np.median(np.diff(fr)) * mpf / 60.0
    ok = np.isfinite(C).all(axis=(0, 2))
    C = C[:, ok]
    steps = np.diff(C, axis=0)
    drift = np.median(steps, axis=1, keepdims=True)
    sc = (steps - drift) * um
    spd = np.linalg.norm(sc, axis=2) / dt_h
    a, b = sc[1:], sc[:-1]
    na, nb = np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2)
    cosang = (a * b).sum(2) / np.maximum(na * nb, 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    track_v = spd.mean(0)
    Cc = C * um - np.vstack([np.zeros((1, 1, 3)), np.cumsum(drift, axis=0) * um])
    L = min(max_lag, n - 1)
    msd = np.array([((Cc[k:] - Cc[:-k]) ** 2).sum(2).mean() for k in range(1, L + 1)])
    fit = prw_fit(np.arange(1, L + 1) * dt_h, msd)
    out = {"available": True, "n_cells": int(C.shape[1]), "row_min": finite(dt_h * 60.0),
           "speed_um_h_median": finite(np.median(spd)),
           "speed_um_h_iqr": [finite(np.percentile(spd, 25)), finite(np.percentile(spd, 75))],
           "track_speed_cv": finite(track_v.std() / max(track_v.mean(), 1e-12)),
           "turn_deg_median": finite(np.median(ang)),
           "prw_v_um_h": finite(fit["v"]), "prw_P_h": finite(fit["P"]), "prw_r2": finite(fit["r2"]),
           "prw_noise_sigma_um": finite(fit["noise_sigma"]),
           "msd_um2": [finite(v) for v in msd[:12]]}
    from scipy.spatial import cKDTree
    nn0 = cKDTree(C[0]).query(C[0], k=2)[0][:, 1]
    d0 = float(np.median(nn0))
    out["nn_um_median"] = finite(d0 * um)
    pairs = cKDTree(C[0]).query_pairs(1.25 * d0, output_type="ndarray")
    # IN CONTACT, OR THERE ARE NO NEIGHBOURS TO KEEP: the centres closer than 2.2 cell radii (a ball's
    # radius is sqrt(5/3) x its points' rms distance), read off the points at row 0
    touching = False
    if P is not None and P.shape[1] % max(C.shape[1], 1) == 0 and ok.all():
        _k = P.shape[1] // C.shape[1]
        _par = np.repeat(np.arange(C.shape[1]), _k)
        _rms = np.sqrt(np.bincount(_par, ((P[0] - C[0][_par]) ** 2).sum(1)) / _k)
        touching = d0 < 2.2 * np.sqrt(5.0 / 3.0) * float(np.median(_rms))
    if touching and len(pairs) >= 10:
        dl = np.linalg.norm(C[-1][pairs[:, 0]] - C[-1][pairs[:, 1]], axis=1)
        out["neighbour_kept"] = finite(np.mean(dl < 1.5 * d0))
    else:
        out["neighbour_kept"] = None
    if P is not None and P.shape[1] % max(C.shape[1], 1) == 0 and ok.all():
        k = P.shape[1] // C.shape[1]
        par = np.repeat(np.arange(C.shape[1]), k)
        asp, al, integ = [], [], 0.0
        r0 = None
        for t in range(0, n, max(1, n // 30)):
            X = P[t]
            cen = np.zeros((C.shape[1], 3)); np.add.at(cen, par, X); cen /= k
            d = X - cen[par]
            G = np.zeros((C.shape[1], 3, 3)); np.add.at(G, par, d[:, :, None] * d[:, None, :]); G /= k
            w, V = np.linalg.eigh(G)
            asp.append(np.sqrt(np.maximum(w[:, 2], 1e-12) / np.maximum(w[:, 0], 1e-12)))
            rr = np.sqrt(np.maximum(w.sum(1), 1e-12))
            if r0 is None:
                r0 = rr
            dist = np.zeros(C.shape[1]); np.maximum.at(dist, par, np.linalg.norm(d, axis=1))
            integ = max(integ, float(np.max(dist / r0)))
            if 0 < t < n - 1:
                stp = C[min(t + 1, n - 1)] - C[t - 1]
                sn = np.linalg.norm(stp, axis=1)
                m = sn > 1e-9
                al.append(np.abs((V[:, :, 2] * stp).sum(1))[m] / sn[m])
        asp = np.array(asp)
        out["aspect_median"] = finite(np.median(asp))
        out["aspect_cv"] = finite(np.median(asp.std(0) / np.maximum(asp.mean(0), 1e-12)))
        out["align_median"] = finite(np.median(np.concatenate(al))) if al else None
        out["integrity_max"] = finite(integ)
    return out


register_run("exp11.cell_motion", cell_motion, None, "MPM cells: speed, persistence, turning, shape, neighbour exchange")


def exchange(T, cell_set="icell", point_set="ipt", every=5, **_):
    """The surface <-> interior exchange of Wang 2021's Type I division (Fig 2A-B), and whether the layer
    survives it. Per recorded row every `every`: `surface` (the layer's cells), `interior` (live MPM cells),
    `dived` / `returning` (live cells carrying the `dived` / `ret` mark), `vol_nonpos` (layer cells whose
    apico-basal polyhedron volume is <= 0 -- a crushed or inverted cell), `vol_median` (of the layer's cells),
    `leaked` (interior points farther from the tissue centroid than the apical surface's 99th-percentile
    radius + 0.5, i.e. outside the lumen), `spike` (the farthest layer vertex over the median radius: ~1.1-1.3 for a
    round gland, > 2 an arm or a thrown vertex) and `giant` (the largest layer cell's volume over the median).
    Timings over every row, per cell slot: `dive_to_div_h` (a `dived` cell's time in the lumen before it
    divides -- the mark clears), `ret_to_return_h` (a daughter's time before it re-enters the layer -- its slot
    goes dormant), their medians and counts, and `cycle_h_median` their sum (Wang: dive -> return 1.96 h
    median, 80 % under 4 h)."""
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d
    z = T.z
    n = T.n_rows()
    u = (T.spec.get("general") or {}).get("units") or {}
    fr = _row_frames(T)
    dt_frame = float((T.spec.get("general") or {}).get("dt", 1.0))
    h_row = float(np.median(np.diff(fr))) * float(u.get("time_s", 600.0)) * dt_frame / 3600.0
    def _blk(name, t):
        k = f"{cell_set}__{name}"
        return np.asarray(z[k][t], float).reshape(-1) if k in z.files else None
    rows = _rows(T, every)
    ser = {k: [] for k in ("surface", "interior", "dived", "returning", "vol_nonpos", "vol_median", "leaked",
                           "spike", "giant")}
    for t in rows:
        P = T.pos(t); nF = T.nF(t)
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        S = T.vertex_block("sep", t)
        oc = _blk("occ", t) if f"{cell_set}__occ" in z.files else None
        oc = np.asarray(z[f"{cell_set}__occ"][t], float).reshape(-1) > 0.5
        dv, rt = _blk("dived", t), _blk("ret", t)
        ser["surface"].append(nF); ser["interior"].append(int(oc.sum()))
        ser["dived"].append(int((oc & (dv > 0.5)).sum()) if dv is not None else None)
        ser["returning"].append(int((oc & (rt > 0.5)).sum()) if rt is not None else None)
        if S is not None:
            vp, _a1, _a2, _ = apicobasal_geometry_3d(torch.tensor(P, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                                                     torch.tensor(es), torch.tensor(et), torch.tensor(ef), nF)
            vp = vp.numpy()
            ser["vol_nonpos"].append(int((vp <= 0).sum())); ser["vol_median"].append(finite(np.median(vp)))
            ser["giant"].append(finite(np.max(vp) / max(float(np.median(vp[vp > 0])) if (vp > 0).any() else 1.0, 1e-9)))
            c = P.mean(0)
            _r = np.linalg.norm(P - c, axis=1)
            ser["spike"].append(finite(_r.max() / max(float(np.median(_r)), 1e-9)))
            ra = np.linalg.norm(P + S / 2.0 - c, axis=1)
            Q = np.asarray(z[f"{point_set}__pos"][t], float)
            qo = np.asarray(z[f"{point_set}__occ"][t], float).reshape(-1) > 0.5
            rq = np.linalg.norm(Q[qo] - c, axis=1)
            ser["leaked"].append(int((rq > np.percentile(ra, 99) + 0.5).sum()))
        else:
            ser["vol_nonpos"].append(None); ser["vol_median"].append(None); ser["leaked"].append(None)
            ser["spike"].append(None); ser["giant"].append(None)
    # timings: consecutive recorded rows a slot is live with the mark
    occ = np.array([np.asarray(z[f"{cell_set}__occ"][t], float).reshape(-1) > 0.5 for t in range(n)])
    def _runs(mark):
        k = f"{cell_set}__{mark}"
        if k not in z.files:
            return []
        M = np.array([np.asarray(z[k][t], float).reshape(-1) > 0.5 for t in range(n)]) & occ
        out = []
        for s in range(M.shape[1]):
            col = M[:, s]
            if not col.any():
                continue
            d = np.diff(np.r_[0, col.astype(int), 0])
            st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            for a, b in zip(st, en):
                if b < n:                                   # ended inside the run (divided / returned)
                    out.append((b - a) * h_row)
        return out
    d2d, r2r = _runs("dived"), _runs("ret")
    md = finite(np.median(d2d)) if d2d else None
    mr = finite(np.median(r2r)) if r2r else None
    return {"available": True, "rows": rows, **ser,
            "dive_to_div_h_median": md, "n_divided": len(d2d),
            "ret_to_return_h_median": mr, "n_returned": len(r2r),
            "ret_to_return_under_4h": finite(np.mean(np.asarray(r2r) < 4.0)) if r2r else None,
            "cycle_h_median": finite((md or 0.0) + (mr or 0.0)) if (md is not None and mr is not None) else None}


register_run("exp11.exchange", exchange, None, "surface <-> interior exchange: counts, timings, the layer's health")


def size_health(T, every=10, **_):
    """Is the surface layer's GROWTH-CYCLE healthy? (exp 11, 2026-09-28, the growth-cycle grid.) Per sampled row:
    `vol_cv` (spread over mean of the cells' measured polyhedron volumes; exp 13's working point reads ~0.2),
    `big` (the fraction of cells over 3x the median volume), `target_cv` and `target_max` (the targets `V0f`, the
    largest over the seed's median), `nonpos` (cells whose volume is <= 0), `R_layer` / `R_membrane` (median radii
    about the layer's centre) and `cells`. Over the run: `burst` -- the largest share of all the run's divisions
    that fall in any one sampled interval (1/n_intervals for a perfectly steady rate; ~0.5 for one big wave)."""
    import torch
    from plexus.operators.vertex_ops import apicobasal_geometry_3d
    z = T.z
    rows = _rows(T, every)
    out = {k: [] for k in ("row", "cells", "vol_cv", "big", "target_cv", "target_max", "nonpos", "R_layer", "R_membrane")}
    v0_seed = float(np.median(np.asarray(z["cell__V0f"][0], float).reshape(-1)[:T.nF(0)]))
    for t in rows:
        P = T.pos(t); nF = T.nF(t)
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        S = T.vertex_block("sep", t)
        vp = apicobasal_geometry_3d(torch.tensor(P, dtype=torch.float32), torch.tensor(S, dtype=torch.float32),
                                    torch.tensor(es), torch.tensor(et), torch.tensor(ef), nF)[0].numpy()
        V0 = np.asarray(z["cell__V0f"][t], float).reshape(-1)[:nF]
        med = float(np.median(vp[vp > 0])) if (vp > 0).any() else 1.0
        c = P.mean(0)
        B = np.asarray(z["bm_node__pos"][t], float); ob = np.asarray(z["bm_node__occ"][t]).reshape(-1) > 0.5
        out["row"].append(int(t)); out["cells"].append(int(nF))
        out["vol_cv"].append(finite(float(np.std(vp) / max(np.mean(vp), 1e-12))))
        out["big"].append(finite(float((vp > 3 * med).mean())))
        out["target_cv"].append(finite(float(np.std(V0) / max(np.mean(V0), 1e-12))))
        out["target_max"].append(finite(float(V0.max() / max(v0_seed, 1e-12))))
        out["nonpos"].append(int((vp <= 0).sum()))
        out["R_layer"].append(finite(float(np.median(np.linalg.norm(P - c, axis=1)))))
        out["R_membrane"].append(finite(float(np.median(np.linalg.norm(B[ob] - c, axis=1)))) if ob.any() else None)
    d = np.diff(np.asarray(out["cells"], float)); d = np.clip(d, 0, None)
    out["burst"] = finite(float(d.max() / d.sum())) if d.sum() > 0 else None
    out["n_intervals"] = int(len(d))
    return out


register_run("exp11.size_health", size_health, None, "surface growth-cycle health: volume CV, big cells, targets, bursts")


def _basal_samples(P, S, es, et, ef, nF, sub, sign):
    """Points sampling the layer's BASAL surface: each cell's basal cap as a fan of triangles about its
    centroid (the triangles `bm_contact[live]` meets), each split `sub` times into 4. Returns the sample
    points [n, 3], the area each one stands for [n], its fan-triangle normal [n, 3] and its cell [n]."""
    X = P + sign * S
    ok = (ef >= 0) & (ef < nF)
    es, et, ef = es[ok], et[ok], ef[ok]
    cen = np.zeros((nF, 3)); cnt = np.zeros(nF)
    np.add.at(cen, ef, X[es]); np.add.at(cnt, ef, 1.0)
    cen /= np.maximum(cnt, 1.0)[:, None]
    A, B, C = cen[ef], X[es], X[et]
    n = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(n, axis=1)
    n = n / np.maximum(2.0 * area, 1e-12)[:, None]
    tri = np.stack([A, B, C], 1)                         # [m, 3 corners, 3]
    own = ef.copy(); nor = n
    for _ in range(int(sub)):                            # midpoint split: one triangle -> four
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(q, 1) for q in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
        own = np.tile(own, 4); nor = np.tile(nor, (4, 1))
    pts = tri.mean(1)
    w = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return pts, w, nor, own


def holes(T, every=5, set_name="bm_node", sub=2, cover=1.5, band=0.3, lift=1.5, min_samples=2, link=1.6,
          min_patches=1.0, um_per_unit=None, track=True, **_):
    """HOLES IN THE MEMBRANE, COUNTED AND SIZED OVER TIME (exp 11, 2026-09-29).

    A point of the layer's basal surface is COVERED when a live membrane node lies within `cover` x the
    membrane's own spacing of it IN THE SURFACE'S PLANE (1.5 by default: at 1.0 the healthy L3 membrane,
    mc_L3d_hl012 s2, read 184-331 "holes" of 2-60 um^2 -- the lattice's own irregular gaps; at 1.5 it reads
    0-2 of <= 4 um^2 while Phase 1's declared hole, w2h_hole100 s1, keeps 700 -> 7,555 um^2), and within `band` (world units, the
    `bm_contact[live]` band) of it along the surface normal. The spacing is re-read every row as the median
    nearest-neighbour distance between live nodes, so a membrane that thins as it stretches is judged against
    its own lattice, not the seed's. Uncovered points are joined into holes by `link` x the sample spacing,
    and a hole must hold at least `min_samples` points. A basal point with a node near it in the plane but
    farther than `band` (up to `lift`) along the normal is counted separately as DETACHED: the membrane lifting
    off is not a hole in it.

    The surface is sampled as each cell's basal fan split `sub` times into 4 (sub 2 -> 16 points per fan
    triangle). Which side of the layer is basal (pos + sep or pos - sep) is decided at the first row by
    which one lies nearer the membrane.

    A gap smaller than `min_patches` node patches (one patch = spacing^2 sqrt(3)/2, about 4 um^2 at exp11's
    2.0-2.4 um spacing) is below what the lattice resolves -- Harunaga 2014's tip perforations, 1.6 um^2, are
    that size and live in the model as porosity, not geometry -- so it is counted in `n_small`, not `n_holes`.

    Per sampled row: `n_holes`, `n_small`; `area_total`, `area_median`, `area_max` (um^2 when a length scale is
    known, else world units^2); `frac_open`, the uncovered fraction of the basal area; `frac_detached`;
    `spacing`, the node spacing; and `areas`, every hole's area (the size distribution). With `track`,
    holes are matched from row to row by their centroids, which gives `tracks`: per hole, its first and
    last row and its area over time. Scale: `um_per_unit`, else the spec's `general.units.length_um`."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    z = T.z
    kp, ko = f"{set_name}__pos", f"{set_name}__occ"
    if kp not in z.files:
        return {"available": False, "why": f"no {kp} in the trajectory"}
    s_um = um_per_unit or (((getattr(T, "spec", {}) or {}).get("general") or {}).get("units") or {}).get("length_um")
    a_scale = float(s_um) ** 2 if s_um else 1.0
    rows = _rows(T, every)
    out = {"available": True, "unit": "um^2" if s_um else "world^2",
           **{k: [] for k in ("row", "n_holes", "n_small", "area_total", "area_median", "area_max",
                              "frac_open", "frac_detached", "spacing", "areas", "centroids")}}
    sign = None
    for t in rows:
        P = T.pos(t); nF = T.nF(t)
        es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        S = T.vertex_block("sep", t)
        S = np.zeros_like(P) if S is None else np.asarray(S, float)[: len(P)]
        B = np.asarray(z[kp][t], float)
        if ko in z.files:
            B = B[np.asarray(z[ko][t]).reshape(-1) > 0.5]
        B = B[np.isfinite(B).all(1)]
        if len(B) < 4 or nF == 0:
            continue
        tb = cKDTree(B)
        if sign is None:                                  # which cap faces the membrane
            sign = min((+1.0, -1.0), key=lambda sg: float(np.median(tb.query(P + sg * S)[0])))
        pts, w, nor, _own = _basal_samples(P, S, es, et, ef, nF, sub, sign)
        spacing = float(np.median(tb.query(B, k=2)[0][:, 1]))
        r_in = cover * spacing
        # the nearest nodes of each sample, split into their in-plane and normal parts
        k = min(12, len(B))
        _, j = tb.query(pts, k=k)
        j = j.reshape(len(pts), -1)
        v = B[j] - pts[:, None, :]
        vn = (v * nor[:, None, :]).sum(-1)
        gn = np.abs(vn)
        gp = np.linalg.norm(v - vn[..., None] * nor[:, None, :], axis=-1)
        covered = ((gp <= r_in) & (gn <= band)).any(1)
        detached = ~covered & ((gp <= r_in) & (gn <= lift)).any(1)
        opn = ~covered & ~detached
        tot = float(w.sum())
        areas, cents = [], []
        if opn.any():
            q = pts[opn]; wq = w[opn]
            h = float(np.sqrt(np.median(wq) * 2.0))        # a sample's own spacing
            pr = cKDTree(q).query_pairs(link * h, output_type="ndarray")
            n = len(q)
            g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)) if len(pr) else coo_matrix((n, n))
            nc, lab = connected_components(g, directed=False)
            cnt = np.bincount(lab, minlength=nc)
            aw = np.bincount(lab, weights=wq, minlength=nc)
            cx = np.stack([np.bincount(lab, weights=wq * q[:, c], minlength=nc) for c in range(3)], 1) / np.maximum(aw, 1e-12)[:, None]
            keep = cnt >= int(min_samples)
            big = keep & (aw >= min_patches * spacing ** 2 * np.sqrt(3) / 2)
            n_small = int((keep & ~big).sum())
            areas = (aw[big] * a_scale).tolist(); cents = cx[big].tolist()
        else:
            n_small = 0
        out["row"].append(int(t)); out["n_holes"].append(len(areas)); out["n_small"].append(n_small)
        out["area_total"].append(finite(float(np.sum(areas))) if areas else 0.0)
        out["area_median"].append(finite(float(np.median(areas))) if areas else 0.0)
        out["area_max"].append(finite(float(np.max(areas))) if areas else 0.0)
        out["frac_open"].append(finite(float(w[opn].sum() / max(tot, 1e-12))))
        out["frac_detached"].append(finite(float(w[detached].sum() / max(tot, 1e-12))))
        out["spacing"].append(finite(spacing * (float(s_um) if s_um else 1.0)))
        out["areas"].append([finite(a) for a in areas]); out["centroids"].append(cents)
    if track:
        out["tracks"] = _track_holes(out, a_scale)
    out.pop("centroids")
    return out


def _track_holes(out, a_scale):
    """Holes followed from one sampled row to the next: a hole continues the nearest hole of the
    previous row whose centroid lies within the larger of the two radii (sqrt(area/pi)) + one row's
    reach; each hole continues at most one. Returns [{first, last, rows, areas}] per track."""
    tracks, prev = [], []                                 # prev: [(track index, centroid, radius)]
    for r, A, C in zip(out["row"], out["areas"], out["centroids"]):
        cur, used = [], set()
        for a, c in zip(A, C):
            c = np.asarray(c, float); rad = float(np.sqrt(max(a, 0.0) / a_scale / np.pi))
            best, bd = None, np.inf
            for ti, pc, pr in prev:
                dd = float(np.linalg.norm(c - pc))
                if ti not in used and dd <= max(rad, pr) + 0.5 and dd < bd:
                    best, bd = ti, dd
            if best is None:
                tracks.append({"first": r, "last": r, "rows": [r], "areas": [a]}); best = len(tracks) - 1
            else:
                tk = tracks[best]; tk["last"] = r; tk["rows"].append(r); tk["areas"].append(a)
            used.add(best); cur.append((best, c, rad))
        prev = cur
    return tracks


def holes_summary(T, **kw):
    o = holes(T, **kw)
    if not o.get("available"):
        return o
    tr = o.get("tracks") or []
    return {"available": True, "unit": o["unit"], "n_holes_last": o["n_holes"][-1] if o["n_holes"] else None,
            "n_holes_max": max(o["n_holes"]) if o["n_holes"] else None,
            "area_max": max(o["area_max"]) if o["area_max"] else None,
            "frac_open_last": o["frac_open"][-1] if o["frac_open"] else None,
            "frac_detached_last": o["frac_detached"][-1] if o["frac_detached"] else None,
            "n_tracks": len(tr), "lifetime_rows_median": finite(float(np.median([t["last"] - t["first"] for t in tr]))) if tr else None}


register_run("exp11.holes", holes_summary, None, "membrane holes: count, sizes, open and detached fractions over time")


def wang_cycle(T, cell_set="icell", every=20, **_):
    """Wang 2021's surface-cell cycle, read off a run (exp 11 Phase 3 B0, 2026-09-29; the L3 gate of 2026-09-28).

    From the run's counters (`mesh_scalar_n_*`, last row): `dives` (T2 removals into the interior, `n_apop`),
    `n_type2` (boundary divisions that sent one daughter in), `n_div` (in-place layer divisions), `typeI_frac`
    = (dives - n_type2) / dives (Wang 0.924) and `inplace_not_typeII` = n_div - n_type2 (Wang: none).
    From `exchange`: the dive -> return time (`cycle_h`, Wang median 1.96 h) and `under4h` (Wang 0.80).

    `p_surface` -- WANG'S ON-SURFACE FRACTION (0.930 in control, 0.701 under collagenase; K = p / (1 - p) =
    13.3): of the surface lineage, the share of cell-rows spent in the layer, summed over the recorded rows,
    sum N_S / sum (N_S + N_B), with N_S the layer's cells and N_B the interior bodies carrying the `dived` or
    `ret` mark (the surface lineage while it is inside). The unmarked interior core is not surface lineage and
    is left out. Wang weighs tracked cells by frames; over a steady run the two agree."""
    z = T.z
    sc = {k.split("scalar_")[-1]: np.asarray(z[k]) for k in z.files if "mesh_scalar_n_" in k}
    last = lambda k: float(sc[k][-1]) if k in sc else 0.0            # noqa: E731
    dives, t2, div = last("n_apop"), last("n_type2"), last("n_div")
    ns = nb = 0.0
    ko = f"{cell_set}__occ"
    for t in range(T.n_rows()):
        ns += T.nF(t)
        if ko in z.files:
            oc = np.asarray(z[ko][t], float).reshape(-1) > 0.5
            m = np.zeros_like(oc)
            for mk in ("dived", "ret"):
                k = f"{cell_set}__{mk}"
                if k in z.files:
                    m |= np.asarray(z[k][t], float).reshape(-1) > 0.5
            nb += float((oc & m).sum())
    o = exchange(T, cell_set=cell_set, every=every)
    return {"available": True, "dives": int(dives), "n_type2": int(t2), "n_div": int(div),
            "typeI_frac": finite((dives - t2) / dives) if dives else None,
            "inplace_not_typeII": int(div - t2),
            "p_surface": finite(ns / (ns + nb)) if (ns + nb) else None,
            "cycle_h": o.get("cycle_h_median"), "under4h": o.get("ret_to_return_under_4h"),
            "surface": (o["surface"][0], o["surface"][-1]), "interior": (o["interior"][0], o["interior"][-1])}


register_run("exp11.wang_cycle", wang_cycle, None, "Wang's surface cycle: Type I share, in-place divisions, on-surface fraction, dive->return time")


def ladder_health(T, **_):
    """The Wang ladder's health gate, H1-H6 (exp 11 md, "The Wang cycle ladder"), at row 80 and the last row:
    H1 no inverted cell (0 at row 80, <= 0.5 % at the end), H2 volume CV <= 0.55, H3 cells over 3x the median
    <= 1 %, H4 layer radius within 0.6 of the membrane's, H5 three-sided cells <= 2 %, H6 division burst <= 0.40.
    (H7, the landing audit, is not a trajectory measure.)"""
    z = T.z
    o = size_health(T, every=40)
    n = T.n_rows() - 1
    r80 = o["row"].index(80) if 80 in o["row"] else min(2, len(o["row"]) - 1)
    rl = len(o["row"]) - 1

    def tri(t):
        nF = T.nF(t); es, et, ef = (np.asarray(a, np.int64) for a in T.half_edges(t))
        ns_ = np.bincount(ef[ef < nF], minlength=nF)[:nF]
        return float((ns_ == 3).mean())
    rm = [o["R_membrane"][i] for i in (r80, rl)]
    H = {"H1_inverted": bool(o["nonpos"][r80] == 0 and o["nonpos"][rl] <= 0.005 * o["cells"][rl]),
         "H2_cv": bool(max(o["vol_cv"][r80], o["vol_cv"][rl]) <= 0.55),
         "H3_giants": bool(max(o["big"][r80], o["big"][rl]) <= 0.01),
         "H4_membrane": bool(all(r is not None for r in rm) and
                             max(abs(o["R_layer"][i] - o["R_membrane"][i]) for i in (r80, rl)) <= 0.6),
         "H5_cones": bool(max(tri(80 if n >= 80 else n), tri(n)) <= 0.02),
         "H6_waves": bool((o["burst"] or 0.0) <= 0.40)}
    return {"available": True, "pass": all(H.values()), **H,
            "cells": (o["cells"][r80], o["cells"][rl]), "vol_cv": (o["vol_cv"][r80], o["vol_cv"][rl]),
            "nonpos": (o["nonpos"][r80], o["nonpos"][rl]), "burst": o["burst"],
            "R_layer_membrane": (o["R_layer"][rl], o["R_membrane"][rl]), "tri": (tri(80 if n >= 80 else n), tri(n))}


register_run("exp11.ladder_health", ladder_health, None, "the Wang ladder's health gate H1-H6")

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

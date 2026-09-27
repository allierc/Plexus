"""exp11 bm_hole_budding -- the rulers: does the bud grow on the hole's axis, and only there?

    exp11.bud          bud_excess along each declared axis (the archive's metric, reused verbatim)
    exp11.balance      the coupling's force ledger: does every interface sum to zero?
    exp11.smg_buds     bud count by the SMG2 data's own counter, on the model's cells

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

from .common import ROOT, finite, register_run

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

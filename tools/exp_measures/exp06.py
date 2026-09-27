"""exp06 excitation_wave -- the rulers: did an excitation travel cell to cell, how fast, and where did it stop?

    exp06.wave         the front read off the excitation variable: which cells fired, when, how often;
                       the front speed and whether it stays constant; the cells past a cut line; the
                       arrival spread across the field; whether the sheet stayed intact
    exp06.cell_trace   every cell's recorded (u, v) against an independent numpy integration of the
                       declared kinetics from the same row-0 state: the identity control

A FIRING IS AN UPWARD CROSSING OF `thr` BY THE EXCITATION VARIABLE, WITH HYSTERESIS. A cell fires at the
first row where u >= thr while it is armed; it re-arms only once u has fallen below `rearm` (default
thr / 2). Without the hysteresis a noisy plateau near thr would count as many waves; with it one action
potential is one firing however long its plateau. A cell already above thr at row 0 (a seeded stimulus)
fires at row 0.

DISTANCE IS COUNTED IN CELL DIAMETERS, the median centroid distance between touching cells at row 0, so a
speed reads "cell diameters per unit of model time" and does not change when the sheet is scaled. When
the spec declares `general.units` (length_um, time_s) the speed is also given in cm/s, the unit Kleber &
Rudy 2004 quote conduction velocities in, and the arrival spread in ms, the unit of the Utrecht frame
(41.7 ms at 24 frames a second).

THE ORIGIN IS DECLARED OR IT IS THE EARLIEST FIRING. An arm may pass `origin: [x, y, z]` (the stimulus
centre); otherwise it is the centroid of the cells that fired at the earliest row. The speed is fitted on
cells at least `r_skip` cell diameters from it, because the first few cells are the stimulus itself and
the front's curvature, not conduction. With `direction: [x, y, z]` the distance is measured ALONG that
axis (a planar front launched from a strip); without it, radially (a point source). A radial front is
slower near its source by D * curvature (Aliev & Panfilov 1996 p. 297, c = c0 - D k), so its speed is
not constant by construction; the conduction laws are read on planar fronts.

A ROW'S TIME IS THE ENGINE'S TICK TIMES dt. The engine strides the recording when n_frames exceeds
`record_cap` (engine.py, the recording plan); `_row_times` rebuilds the same ticks from the spec as run
and refuses when they do not match the rows on disk, rather than silently mistiming every arrival.
"""
from __future__ import annotations

import numpy as np

from .common import cells, finite, neighbour_pairs, register_run, spec_op


# ============================================================================ time and space
def _row_times(T, dt=None):
    """Model time of every recorded row: the engine's recorded ticks times general.dt."""
    g = (getattr(T, "spec", None) or {}).get("general") or {}
    n = T.n_rows()
    step = float(dt if dt is not None else g.get("dt", 1.0))
    nf = g.get("n_frames")
    if nf is None:
        return np.arange(n) * step
    nf = int(nf)
    sd = g.get("save_data")
    cap = nf + 1 if sd is True else 1 if sd is False else int(g.get("record_cap", 10000))
    stride = max(1, (nf + cap) // cap)
    ticks = sorted(set(range(0, nf + 1, stride)) | {nf})
    if len(ticks) != n:
        raise ValueError(f"the spec's recording plan gives {len(ticks)} rows (n_frames {nf}, stride "
                         f"{stride}) but the trajectory has {n}: row times would be wrong")
    return np.asarray(ticks, float) * step


def _units(T):
    u = ((getattr(T, "spec", None) or {}).get("general") or {}).get("units") or {}
    return u.get("length_um"), u.get("time_s")


def _series(T, chan):
    """[rows, n] excitation variable of the row-0 cells, NaN where a cell is gone; and the cells."""
    c0 = cells(T, 0)
    n = len(c0)
    U = np.full((T.n_rows(), n), np.nan)
    live = np.zeros(T.n_rows(), int)
    for t in range(T.n_rows()):
        a = T.state("chem", t)
        occ = T.occ(T.c if hasattr(T, "c") and T.c else getattr(T, "s", ""), t)
        if a is None:
            raise ValueError("exp06.wave: the run recorded no `chem` block")
        a = np.asarray(a, float)
        a = a[:, None] if a.ndim == 1 else a
        ok = c0.slot < len(a)
        if occ is not None:
            ok &= np.asarray(occ, bool)[np.minimum(c0.slot, len(occ) - 1)]
        U[t, ok] = a[c0.slot[ok], chan]
        live[t] = int(ok.sum())
    return U, c0, live


def _cell_diameter(T, c0):
    p = neighbour_pairs(T, 0, c0)
    if len(p) == 0:
        return float("nan")
    return float(np.median(np.linalg.norm(c0.x[p[:, 0]] - c0.x[p[:, 1]], axis=1)))


def _firings(U, thr, rearm):
    """List of firing row indices per cell (upward crossings of thr, re-armed below `rearm`)."""
    rows, n = U.shape
    armed = np.ones(n, bool)
    out = [[] for _ in range(n)]
    for t in range(rows):
        u = U[t]
        fire = armed & (u >= thr)
        for i in np.flatnonzero(fire):
            out[i].append(t)
        armed = (armed & ~fire) | (u < rearm)
    return out


def _fit_speed(r, t):
    """(speed, r2) of t = a + r / speed by least squares; None when it cannot be read."""
    if len(r) < 8 or np.ptp(r) <= 0:
        return None, None
    b, a = np.polyfit(r, t, 1)
    res = t - (a + b * r)
    r2 = 1.0 - res.var() / t.var() if t.var() > 0 else None
    return (1.0 / b if b > 0 else None), r2


# ============================================================================ the front
def wave(T, chan=0, thr=0.5, rearm=None, origin=None, direction=None, r_skip=3.0, cut=None,
         cut_margin=1.0, dt=None, **_):
    """The excitation front of one run.

    frac_reached       fraction of the row-0 cells that fired at least once
    frac_reached_far   the same over cells at least r_skip cell diameters from the origin
                       (0 for an uncoupled sheet whatever its stimulus)
    speed              front speed, cell diameters per unit model time, fitted beyond r_skip
    speed_cm_s         the same in cm/s when the spec declares units
    speed_inner/outer  the fit on the nearer / farther half of those cells
    speed_drift_abs    |outer / inner - 1|: 0 for a front at constant speed
    fit_r2             R^2 of arrival time against distance (1 for a clean front)
    n_waves            median firings per cell beyond r_skip: how many waves crossed the sheet
    frac_beyond_cut    with `cut: {point, normal}` (normal pointing AWAY from the stimulus): fraction
                       of cells more than `cut_margin` cell diameters past the line that fired
    frac_before_cut    the same on the stimulus side: the wave had to arrive for a block to mean anything
    arrival_spread     last minus first arrival over the cells that fired, model time (and _ms)
    intact             1 when every row-0 cell is live and finite on every row, else 0
    """
    rearm = thr / 2.0 if rearm is None else rearm
    tt = _row_times(T, dt)
    U, c0, live = _series(T, chan)
    n = U.shape[1]
    cd = _cell_diameter(T, c0)
    L_um, T_s = _units(T)
    intact = float(bool(np.isfinite(U).all()) and int(live.min()) == n)
    fir = _firings(np.nan_to_num(U, nan=-np.inf), thr, rearm)
    nf = np.array([len(f) for f in fir])
    first = np.array([tt[f[0]] if f else np.nan for f in fir])
    out = {"n_cells": n, "cell_diameter": finite(cd), "intact": intact,
           "live_min_over_first": finite(live.min() / max(n, 1)),
           "frac_reached": finite((nf > 0).mean()) if n else None}
    fired = np.isfinite(first)
    if origin is not None:
        o = np.asarray(origin, float)[: c0.x.shape[1]]
    elif fired.any():
        o = c0.x[fired & (first <= np.nanmin(first) + 1e-12)].mean(0)
    else:
        o = None
    if o is None or not np.isfinite(cd) or cd <= 0:
        out.update(frac_reached_far=0.0 if o is None else None, n_waves=0.0 if o is None else None)
        return out
    if direction is not None:                    # a planar front: distance along the propagation axis
        d = np.asarray(direction, float)[: c0.x.shape[1]]
        r = ((c0.x - o) @ (d / np.linalg.norm(d))) / cd
    else:                                        # a point source: radial distance
        r = np.linalg.norm(c0.x - o, axis=1) / cd
    far = r >= r_skip
    out["origin"] = [finite(v) for v in o]
    out["frac_reached_far"] = finite((nf[far] > 0).mean()) if far.any() else None
    out["n_waves"] = finite(np.median(nf[far])) if far.any() else None
    if fired.sum() >= 2:
        out["arrival_spread"] = finite(np.nanmax(first) - np.nanmin(first))
        if T_s:
            out["arrival_spread_ms"] = finite(out["arrival_spread"] * float(T_s) * 1e3)
    use = far & fired
    v, r2 = _fit_speed(r[use], first[use])
    out["speed"], out["fit_r2"] = finite(v), finite(r2)
    if v is not None and L_um and T_s:
        out["speed_cm_s"] = finite(v * cd * float(L_um) * 1e-4 / float(T_s))
    if use.sum() >= 16:
        rm = np.median(r[use])
        vi, _ = _fit_speed(r[use & (r <= rm)], first[use & (r <= rm)])
        vo, _ = _fit_speed(r[use & (r > rm)], first[use & (r > rm)])
        out["speed_inner"], out["speed_outer"] = finite(vi), finite(vo)
        if vi and vo:
            out["speed_drift_abs"] = finite(abs(vo / vi - 1.0))
    if cut:
        p = np.asarray(cut["point"], float)[: c0.x.shape[1]]
        nrm = np.asarray(cut["normal"], float)[: c0.x.shape[1]]
        nrm = nrm / np.linalg.norm(nrm)
        side = ((c0.x - p) @ nrm) / cd
        beyond, before = side > cut_margin, side < -cut_margin
        out["n_beyond_cut"] = int(beyond.sum())
        out["frac_beyond_cut"] = finite((nf[beyond] > 0).mean()) if beyond.any() else None
        # the stimulus side must have fired, or "nothing crossed" only says "nothing propagated"
        out["frac_before_cut"] = finite((nf[before] > 0).mean()) if before.any() else None
    return out


# ============================================================================ the identity control
def _aliev_panfilov(u, v, p):
    """Aliev & Panfilov 1996 (Chaos Solitons Fractals 7:293), the local kinetics:
    du/dt = -k u (u - a)(u - 1) - u v
    dv/dt = eps(u, v) (-v - k u (u - a - 1)),   eps = eps0 + mu1 v / (u + mu2)"""
    k, a, e0, m1, m2 = (float(p.get(x, d)) for x, d in
                        (("k", 8.0), ("a", 0.15), ("eps0", 0.002), ("mu1", 0.2), ("mu2", 0.3)))
    eps = e0 + m1 * v / (u + m2)
    return -k * u * (u - a) * (u - 1.0) - u * v, eps * (-v - k * u * (u - a - 1.0))


def _fitzhugh_nagumo(u, v, p):
    """FitzHugh 1961 (Biophys J 1:445), BVP model with x -> u, y -> v:
    du/dt = c (u - u^3/3 + v + z),   dv/dt = -(u - a + b v) / c"""
    a, b, c, z = (float(p.get(x, d)) for x, d in (("a", 0.7), ("b", 0.8), ("c", 3.0), ("z", 0.0)))
    return c * (u - u ** 3 / 3.0 + v + z), -(u - a + b * v) / c


KINETICS = {"aliev_panfilov": _aliev_panfilov, "fitzhugh_nagumo": _fitzhugh_nagumo}


def cell_trace(T, chan=0, dt=None, model=None, params=None, **_):
    """max |recorded - reference| over every cell and row, for u and v.

    The reference integrates the declared kinetics (the run's own `cell_chem_react` entry: its model
    and parameters, `rate` included) by forward Euler at the frame dt from each cell's recorded row-0
    (u, v). It is meaningful only on an arm with no coupling and no stimulus after row 0 -- the
    identity arm -- where the reaction is the whole right-hand side."""
    op = spec_op(T, "cell_chem_react")
    model = model or op.get("model")
    params = dict(op if params is None else params)
    if model not in KINETICS:
        return {"available": False, "why": f"no reference kinetics for model {model!r}"}
    f = KINETICS[model]
    rate = float(params.get("rate", 1.0))
    g = (getattr(T, "spec", None) or {}).get("general") or {}
    h = float(dt if dt is not None else g.get("dt", 1.0))
    tt = _row_times(T, h)
    U, c0, _ = _series(T, chan)
    V, _, _ = _series(T, chan + 1)
    u, v = U[0].copy(), V[0].copy()
    du = dv = 0.0
    tick = 0
    for t in range(1, T.n_rows()):
        target = int(round(tt[t] / h))
        while tick < target:
            fu, fv = f(u, v, params)
            u, v = u + h * rate * fu, v + h * rate * fv
            tick += 1
        du = max(du, float(np.nanmax(np.abs(U[t] - u))))
        dv = max(dv, float(np.nanmax(np.abs(V[t] - v))))
    return {"available": True, "model": model, "max_dev_u": finite(du), "max_dev_v": finite(dv),
            "max_dev": finite(max(du, dv)), "u_range": finite(np.nanmax(U) - np.nanmin(U))}


register_run("exp06.wave", wave, None, "excitation front: reach, speed, constancy, cut, waves, spread")
register_run("exp06.cell_trace", cell_trace, None, "identity: recorded kinetics vs independent integration")


# ============================================================================ the results table's source
def write_results(root=None):
    """`experiments/specs/exp06/wave.jsonl`: one line per run, the `exp06.wave` numbers the md's derived
    Results table reads (`tools/exp_results_table.py 6`), copied from the scorer's own cache
    (`measures.jsonl`) so the table and the card cannot disagree. The last measurement of a run wins.

        PYTHONPATH=src:tools python -c "from exp_measures.exp06 import write_results; write_results()"
    """
    import json
    import os
    from .common import ROOT
    d = os.path.join(root or ROOT, "experiments", "specs", "exp06")
    last = {}
    for line in open(os.path.join(d, "measures.jsonl")):
        r = json.loads(line)
        if r["measure"] in ("exp06.wave", "exp06.cell_trace"):
            last.setdefault(r["run"], {}).update({k: v for k, v in (r["value"] or {}).items()
                                                  if isinstance(v, (int, float)) or v is None})
    with open(os.path.join(d, "wave.jsonl"), "w") as fh:
        for run, v in last.items():
            fh.write(json.dumps({"spec": run, **v}) + "\n")
    return len(last)


"""exp06 excitation_wave -- the rulers: did an excitation travel cell to cell, how fast, and where did it stop?

    exp06.wave         the front read off the excitation variable: which cells fired, when, how often;
                       the front speed and whether it stays constant; the cells past a cut line; the
                       arrival spread across the field; whether the sheet stayed intact
    exp06.cell_trace   every cell's recorded (u, v) against an independent numpy integration of the
                       declared kinetics from the same row-0 state: the identity control
    exp06.real         Phase 2, the REAL Utrecht sheet: the wave's arrival map over the cells of the
                       segmentation -- spread in ms over the fit's interior cells, and the model's own
                       delay map through `tools/cardio_delay_wave.py` (plane wave, point source,
                       neighbour coherence), the same check the recordings' fitted delays went through
    exp06.apd          APD90 of every activation of every cell: the free pulse's against the vortex's
                       (Aliev & Panfilov Fig 5: the vortex's falls to 0.53 of the free pulse's)
    exp06.rotor        a sustained vortex: its cycle, and the winding number of the activation phase
                       round a scar or a loop -- 1 when it turns round what the loop encloses

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


# ============================================================================ Phase 2: the real sheet
def _on_set(T, set_name):
    """The run read through the named set. A particle-layout run with several positional sets is
    opened on its BIGGEST one by default -- on the real sheet that is the seed's material points, not
    the cells -- so the set is chosen by name, never by size."""
    from plexus.measures import ParticleTraj
    if set_name is None or getattr(T, "s", set_name) == set_name or not isinstance(T, ParticleTraj):
        return T
    U = ParticleTraj(T.path, set_name=set_name)
    U.dir, U.spec = getattr(T, "dir", None), getattr(T, "spec", {})
    return U


def real(T, set="cell", fit=None, chan=0, thr=0.5, rearm=None, dt=None, frame_ms=1000.0 / 24.0,
         cut=None, cut_margin=1.0, **_):
    """The wave on the measured sheet, cell by cell.

    arrival_spread_ms        last minus first arrival over the fit's INTERIOR cells (the cells the
                             cardio fit scores and delay_wave.json reports), ms
    arrival_spread_ms_all    the same over every cell
    frac_reached             fraction of cells that fired
    plane_wave_r2 ...        `tools/cardio_delay_wave.check` on the model's own arrival map, in
                             frames of 1/24 s -- the numbers `data/delay_wave.json` holds for the fits
    frac_before_cut / frac_beyond_cut   with `cut:`, as exp06.wave reports them
    """
    import os
    import tempfile
    import sys
    from .common import ROOT
    U = _on_set(T, set)
    w = wave(U, chan=chan, thr=thr, rearm=rearm, dt=dt, cut=cut, cut_margin=cut_margin)
    rearm = thr / 2.0 if rearm is None else rearm
    tt = _row_times(U, dt)
    X, c0, _ = _series(U, chan)
    fir = _firings(np.nan_to_num(X, nan=-np.inf), thr, rearm)
    first = np.array([tt[f[0]] if f else np.nan for f in fir])
    L_um, T_s = _units(U)
    ms = 1e3 * float(T_s) if T_s else None
    out = {"n_cells": len(first), "frac_reached": w.get("frac_reached"), "intact": w.get("intact")}
    for k in ("frac_before_cut", "frac_beyond_cut", "n_beyond_cut"):
        if k in w:
            out[k] = w[k]
    if ms is None:
        out["why"] = "the spec declares no units.time_s: no milliseconds"
        return out
    interior = np.ones(len(first), bool)
    if fit:
        z = np.load(fit if os.path.isabs(fit) else os.path.join(ROOT, fit))
        m = np.asarray(z["interior"], bool)
        if len(m) == len(first):
            interior = m
    fired = np.isfinite(first)
    a = first * ms
    if (fired & interior).sum() >= 2:
        out["arrival_spread_ms"] = finite(np.nanmax(a[interior]) - np.nanmin(a[interior]))
    if fired.sum() >= 2:
        out["arrival_spread_ms_all"] = finite(np.nanmax(a) - np.nanmin(a))
    if (fired & interior).sum() >= 10:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import cardio_delay_wave as CDW
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "model_delay.npz")
            np.savez(p, delay=np.nan_to_num(a / frame_ms, nan=0.0), interior=interior & fired)
            x = os.path.join(td, "centroids.npy")
            np.save(x, np.asarray(c0.x, float))
            r = CDW.check(p, x)
        for k, v in r.items():
            if isinstance(v, (int, float)):
                out[f"map_{k}"] = finite(v)
    return out


def follow(T, set="cell", block="gam_prev", chan=0, thr=0.5, cut=None, cut_margin=1.0, **_):
    """Does the contraction follow the excitation? Read on the combined rig (active_strain[excitation]).

    peak_ratio_beyond   mean peak activation (`gam_prev`, the active strain's clock value) of the cells
                        more than `cut_margin` cell diameters past the cut, over that of the cells
                        before it (0 when nothing past a closed line contracts)
    frac_excited        fraction of cells whose excitation reached `thr`
    frac_contracting    fraction whose activation rose above 1 % of the sheet's largest
    """
    U = _on_set(T, set)
    n = U.n_rows()
    c0 = cells(U, 0)
    G = np.stack([np.asarray(U.state(block, t), float)[c0.slot, 0] for t in range(n)])
    X = np.stack([np.asarray(U.state("chem", t), float)[c0.slot, chan] for t in range(n)])
    peak = np.nanmax(G, 0)
    out = {"frac_excited": finite((np.nanmax(X, 0) >= thr).mean()),
           "frac_contracting": finite((peak > 0.01 * np.nanmax(peak)).mean()) if np.nanmax(peak) > 0 else 0.0,
           "peak_mean": finite(np.nanmean(peak))}
    if cut:
        cd = _cell_diameter(U, c0)
        p = np.asarray(cut["point"], float)[: c0.x.shape[1]]
        nrm = np.asarray(cut["normal"], float)[: c0.x.shape[1]]
        side = ((c0.x - p) @ (nrm / np.linalg.norm(nrm))) / cd
        before, beyond = side < -cut_margin, side > cut_margin
        pb = np.nanmean(peak[before]) if before.any() else np.nan
        out["peak_before"] = finite(pb)
        out["peak_beyond"] = finite(np.nanmean(peak[beyond])) if beyond.any() else None
        if np.isfinite(pb) and pb > 0 and beyond.any():
            out["peak_ratio_beyond"] = finite(np.nanmean(peak[beyond]) / pb)
    return out


def apd(T, set=None, chan=0, thr=0.5, end=0.1, rearm=None, late=4, dt=None, **_):
    """APD90 of every activation: from the upstroke through `thr` to the fall below `end` (u is 0 at
    rest and ~1 at the plateau, so `end` = 0.1 is 90 % repolarised).

    apd_first    median APD of each cell's FIRST activation -- the free pulse, APD0
    apd_late     median APD of activations number `late` and later -- the vortex after rotations
    apd_ratio    apd_late / apd_first      (Aliev & Panfilov Fig 5: 0.53 for their model)
    n_fire_max, n_fire_median   activations per cell over the run
    """
    U = _on_set(T, set)
    rearm = thr / 2.0 if rearm is None else rearm
    tt = _row_times(U, dt)
    X, c0, _ = _series(U, chan)
    rows, n = X.shape
    armed = np.ones(n, bool); inap = np.zeros(n, bool); t_up = np.full(n, np.nan); k = np.zeros(n, int)
    recs = []
    for t in range(rows):
        u = np.nan_to_num(X[t], nan=-np.inf)
        fire = armed & (u >= thr)
        k[fire] += 1; t_up[fire] = tt[t]; inap |= fire; armed &= ~fire
        down = inap & (u < end)
        for i in np.flatnonzero(down):
            recs.append((k[i], tt[t] - t_up[i]))
        inap &= ~down
        armed |= u < rearm
    R = np.asarray(recs, float).reshape(-1, 2)
    out = {"n_fire_max": int(k.max()) if n else 0, "n_fire_median": finite(np.median(k)) if n else None,
           "n_apd": int(len(R))}
    if len(R) and (R[:, 0] == 1).any():
        out["apd_first"] = finite(np.median(R[R[:, 0] == 1, 1]))
    if len(R) and (R[:, 0] >= late).any():
        out["apd_late"] = finite(np.median(R[R[:, 0] >= late, 1]))
        if out.get("apd_first"):
            out["apd_ratio"] = finite(out["apd_late"] / out["apd_first"])
    return out



def _winding(F, tt, x, ring, centre, t0):
    """Winding number, at time t0, of the activation phase round the closed loop of cells `ring` taken
    in angular order about `centre`. A cell's phase is its own fraction of the way from its last upstroke
    before t0 to its next one after (2 pi per cycle, each cell by its OWN interval), so cells beating at
    slightly different rates do not alias. Sum of the wrapped phase steps round the loop / 2 pi: 1 or -1
    when an activation circulates round something inside the loop, 0 when none does. None when a loop
    cell has no upstroke on either side of t0."""
    o = ring[np.argsort(np.arctan2(x[ring, 1] - centre[1], x[ring, 0] - centre[0]))]
    ph = []
    for k in o:
        f = np.asarray(F[k], int)
        before, after = f[tt[f] <= t0], f[tt[f] > t0]
        if len(before) == 0 or len(after) == 0:
            return None
        a, b = tt[before[-1]], tt[after[0]]
        ph.append(2 * np.pi * (t0 - a) / (b - a))
    ph = np.asarray(ph)
    return round(float(np.angle(np.exp(1j * np.diff(np.r_[ph, ph[0]]))).sum() / (2 * np.pi)), 6) + 0.0   # an integer up to round-off


def rotor(T, set="cell", chan=0, thr=0.5, rearm=None, dt=None, scar=None, loop=None, band=(0.04, 0.12),
          n_probe=5, **_):
    """Is there a sustained vortex, how fast does it turn, and what does it turn round?

    cycle          median interval between a cell's upstrokes, from its second upstroke on (the S1 -> S2
                   interval excluded), in model time; `cycle_frames` in recording frames when the spec
                   declares `active_strain.frames_per_clock_frame` (the contraction rig), with IQR
    n_up_median, n_up_min, n_up_max    upstrokes per cell over the run
    alive          1 when some cell still fires in the last tenth of the run
    winding        median winding number (`_winding`) at `n_probe` times over the second half of the
                   run, round `scar` = {point, normal, half_length} (the cells `band` away from that
                   segment) or round `loop` = {centre, r0, r1} (an annulus); `winding_n` = probes that read
                   (a probe is unreadable when a loop cell has no upstroke on one side of it -- a silent
                   core cell inside the loop does that), `winding_cells` = cells in the loop
    winding_loop   with BOTH `scar` and `loop`: the loop's reading, the scar's stays `winding`
    """
    U = _on_set(T, set)
    rearm = thr / 2.0 if rearm is None else rearm
    tt = _row_times(U, dt)
    X, c0, _ = _series(U, chan)
    F = _firings(np.nan_to_num(X, nan=-np.inf), thr, rearm)
    nu = np.array([len(f) for f in F])
    iv = np.concatenate([np.diff(tt[np.asarray(f, int)])[1:] for f in F if len(f) >= 3] or [np.array([])])
    out = {"n_up_median": finite(np.median(nu)), "n_up_min": int(nu.min()), "n_up_max": int(nu.max()),
           "alive": int(any(len(f) and tt[f[-1]] >= tt[0] + 0.9 * (tt[-1] - tt[0]) for f in F))}
    if len(iv):
        out["cycle"] = finite(np.median(iv))
        op = next((o for o in (getattr(U, "spec", None) or {}).get("operators", [])
                   if o.get("op") == "active_strain" and o.get("frames_per_clock_frame")), None)
        if op is not None:
            fr = float(U.spec["general"]["dt"]) * float(op["frames_per_clock_frame"])
            out["cycle_frames"] = finite(np.median(iv) / fr)
            out["cycle_frames_iqr"] = [finite(np.percentile(iv, 25) / fr), finite(np.percentile(iv, 75) / fr)]
    x = np.asarray(c0.x, float)[:, :2]
    probes = np.linspace(tt[0] + 0.5 * (tt[-1] - tt[0]), tt[-1], n_probe + 2)[1:-1]

    def wind(ring, centre, key):
        if len(ring) < 3:
            return
        w = [v for v in (_winding(F, tt, x, ring, centre, t0) for t0 in probes) if v is not None]
        out[f"{key}_n"], out[f"{key}_cells"] = len(w), int(len(ring))
        if w:
            out[key] = finite(np.median(w))
    if scar is not None:
        p, nrm = np.asarray(scar["point"][:2], float), np.asarray(scar["normal"][:2], float)
        tan = np.array([-nrm[1], nrm[0]]) / np.linalg.norm(nrm)
        h = float(scar.get("half_length", np.inf))
        s_ = np.clip((x - p) @ tan, -h, h)
        d = np.linalg.norm(x - (p + s_[:, None] * tan), axis=1)
        wind(np.flatnonzero((d > band[0]) & (d < band[1])), p, "winding")
    if loop is not None:
        centre = np.asarray(loop["centre"][:2], float)
        d = np.linalg.norm(x - centre, axis=1)
        wind(np.flatnonzero((d > float(loop["r0"])) & (d < float(loop["r1"]))), centre,
             "winding" if scar is None else "winding_loop")
    return out


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
        if r["measure"] in ("exp06.wave", "exp06.cell_trace", "exp06.real", "exp06.apd", "exp06.follow"):
            last.setdefault(r["run"], {}).update({k: v for k, v in (r["value"] or {}).items()
                                                  if isinstance(v, (int, float)) or v is None})
    with open(os.path.join(d, "wave.jsonl"), "w") as fh:
        for run, v in last.items():
            fh.write(json.dumps({"spec": run, **v}) + "\n")
    return len(last)


register_run("exp06.real", real, None, "the wave on the real Utrecht sheet: arrival spread, delay-map check, cut")
register_run("exp06.apd", apd, None, "APD90 per activation: free pulse vs vortex")
register_run("exp06.follow", follow, None, "contraction follows excitation: activation past a cut over before")
register_run("exp06.rotor", rotor, None, "a sustained vortex: cycle, upstrokes per cell, winding round a scar or a loop")

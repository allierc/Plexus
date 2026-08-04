#!/usr/bin/env python
"""time_analysis -- describe a metric's TRAJECTORY, not just its endpoint.

WHY
------------------------------------------------------------------------------------------------
Every run already records each metric frame by frame, and every agent in the loop reads only the
endpoints. `metrics.png` -- the plot of those trajectories -- is referenced ZERO times anywhere in
the codebase. It has been drawn for every run since the beginning and read by nobody.

That is not a cosmetic gap, because a scalar cannot distinguish these two runs:

    A:  protrusion climbs to 2.7 and stays there          -> a tube
    B:  protrusion spikes to 2.7 at the moment the mesh tears, then collapses

`protr_peak` is 2.7 for both, and the campaign ranks on `protr_peak`. Worse, the peak is a maximum
over ALL frames with no validity filter, so B scores exactly as well as A -- which means a long
enough search discovers that the cheapest way to score well is to destroy the tissue.

Classifying the SHAPE of a curve is arithmetic, not judgement. So it is computed here,
deterministically, and written into the record BEFORE any agent reads it -- the same discipline as
the Critic running before the Reflection.

THE SHAPES, and what each one is telling you
------------------------------------------------------------------------------------------------
  flat            nothing happened. If this is your headline metric, the run was pointless.
  rising          still climbing at the final frame -- THE RUN WAS TOO SHORT. The final value is
                  not a result, it is wherever we happened to stop.
  converged       settled to a value and stayed. The final value means something.
  peaked          best in the middle, then declined. THE FINAL FRAME IS NOT THE RESULT -- reading
                  the endpoint reports the decay, not the phenomenon.
  exploded        a late super-linear blow-up. Almost always the mesh failing, not biology.
  pinned          the tail is EXACTLY constant -- not converging, but held against a hard limit.
                  This is the signature of a buffer ceiling, and it is the one that cost us a
                  whole overnight study: 32 runs all ended at exactly 1778 cells because that is
                  where the vertex reservoir runs out. A single glance at `cells` classified as
                  `pinned` would have said so immediately.

`pinned` vs `converged` is the distinction that matters most and it is easy: a converging quantity
still wobbles in its last few samples; a quantity held against a hard limit is *bit-identical*
sample after sample.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

SHAPES = ("too_short", "flat", "pinned", "exploded", "peaked", "rising", "converged")


def classify(y, t=None, tail_frac=0.25, flat_eps=1e-3, decline=0.25, rise=0.15,
             explode_mult=3.0):
    """Describe one trajectory. Pure arithmetic; no model, no judgement.

    Returns a dict with the shape plus the numbers the shape was derived from, so the verdict can
    always be audited against its evidence.
    """
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    n = int(ok.sum())
    out = {"n": n, "n_nonfinite": int((~ok).sum())}
    if n < 4:
        return {**out, "shape": "too_short"}
    y = y[ok]
    t = np.arange(len(y), dtype=float) if t is None else np.asarray(t, dtype=float)[ok]

    lo, hi = float(y.min()), float(y.max())
    rng = hi - lo
    scale = max(abs(float(np.median(y))), 1e-12)
    k = max(3, int(round(tail_frac * len(y))))
    tail = y[-k:]
    i_peak = int(np.argmax(y))
    peak_at = float(i_peak / (len(y) - 1))

    out.update(first=float(y[0]), final=float(y[-1]), peak=hi, min=lo,
               peak_at_frac=round(peak_at, 3),
               tail_spread=float(tail.max() - tail.min()))

    # --- flat: nothing moved, relative to its own magnitude
    if rng <= flat_eps * scale:
        return {**out, "shape": "flat"}

    # --- pinned: the tail is EXACTLY constant. A real quantity converging still wobbles; a
    #     quantity held against a hard ceiling repeats bit-for-bit. This is the 1778 signature.
    if float(tail.max() - tail.min()) == 0.0 and len(y) >= 6:
        return {**out, "shape": "pinned", "pinned_at": float(tail[-1])}

    # --- exploded: the last quarter dwarfs everything before it
    head_rng = float(np.ptp(y[:-k])) if len(y) > k else rng
    tail_jump = float(abs(tail[-1] - tail[0]))
    if head_rng > 0 and tail_jump > explode_mult * head_rng:
        return {**out, "shape": "exploded", "tail_jump": tail_jump, "head_range": head_rng}

    # --- peaked: best in the interior, then a real decline
    fell = (hi - float(y[-1])) / max(rng, 1e-12)
    if 0.05 < peak_at < 0.85 and fell > decline:
        return {**out, "shape": "peaked", "declined_by": round(float(fell), 3)}

    # --- rising vs converged: is it still going anywhere?
    span = max(float(t[-1] - t[-k]), 1e-12)
    slope = float(np.polyfit(t[-k:], tail, 1)[0])
    moved = abs(slope) * span / max(rng, 1e-12)
    out["tail_slope"] = slope
    out["tail_moved_frac_of_range"] = round(moved, 3)
    if moved > rise:
        return {**out, "shape": "rising" if slope > 0 else "peaked"}
    return {**out, "shape": "converged"}


# --------------------------------------------------------------------------- run-level
def classify_npz(path, keys=None):
    """Classify every series in an .npz. Returns {name: verdict}."""
    z = np.load(path)
    t = z["t"] if "t" in z.files else (z["frame"] if "frame" in z.files else None)
    out = {}
    for k in (keys or z.files):
        if k in ("t", "frame"):
            continue
        v = z[k]
        if v.ndim != 1 or v.shape[0] < 2:
            continue
        # A LABEL IS NOT A CURVE. metrics.npz carries the per-frame morphology labels as their own
        # STRING array (tube_analysis saves them separately, deliberately, so the numeric columns
        # are not forced to a string cast). `classify` calls np.asarray(v, float) and raises
        # ValueError: could not convert string to float: 'sphere'.
        #
        # That exception propagated out of `report`, and llm_agents catches it and writes
        # "(trajectory shapes unavailable)" into the Reader's prompt. So from the moment the label
        # column was added, EVERY run has been read with no trajectory information at all -- the
        # whole HOW EACH MEASUREMENT BEHAVED OVER TIME block, the peaked/pinned/exploded warnings,
        # the evidence horizon -- and the failure announced itself only as one parenthesis in a
        # prompt nobody re-read. The curves were computed, plotted to metrics.png and classified;
        # the classification then died on a column that was never meant to be classified.
        if v.dtype.kind not in "fiub":
            continue
        out[k] = classify(v, t if (t is not None and len(t) == len(v)) else None)
    return out


def evidence_horizon(shapes, series, t=None, n_bar=3, first_bar=1):
    """The first frame at which the mesh stops being trustworthy, as an ABSOLUTE CELL COUNT.

    A run is not all-or-nothing: if the mesh fails at frame 380 of 400, the first 379 frames are
    sound physics. So the record carries a HORIZON, every measurement is taken before it, and
    "final" means the last valid frame.

    WHY A COUNT AND NOT A FRACTION (Cedric, 31 July -- the first version used hollow_frac > 0.05).
    A broken cell is a LOCAL topology violation: the physics is wrong there, and having more
    healthy cells elsewhere does not make it less wrong. Worse, a fractional bar gets MORE
    PERMISSIVE AS THE TISSUE GROWS, because the denominator grows with it. Measured on
    r01_03_5e3159_3, the 5% bar meant:

        frame   0   1431 cells -> tolerated  72 broken cells
        frame 461   1966 cells -> tolerated  98
        frame 807   2389 cells -> tolerated 119

    So the runs that grow the most -- the ones we care about -- were granted the most damage. An
    absolute bar of ~20 puts this run's horizon at frame 461 instead of 692.

    WHICH COUNT. Prefer `broken_n`: cells that are genuinely under-connected or whose ring is not
    a valid polygon. Do NOT threshold the legacy blend if the split is available -- the blend also
    counts folded caps and just-divided slivers, and a dividing tissue always has slivers, so a bar
    of 20 on the blend would fire almost immediately and the rule would simply stop being applied.
    An over-strict rule is its own failure mode.

    THE BLEND IS NOT AN ACCEPTABLE FALLBACK, AND ASSUMING IT WAS COST ME A WRONG VERDICT.
    I first reasoned: the blend counts more cells than `broken_n`, so a blend-based horizon fires
    early -- pessimistic, therefore safe. That is wrong. The blend does not over-count damage; it
    counts a DIFFERENT THING. Measured on r01_03_5e3159_3:

        corr(hollow_n, n_tip) = +0.971

    The blended "hollow" count tracks the number of TIP CELLS almost exactly. Over that run the
    tube grows monotonically (length 2.96 -> 19.00) and narrows smoothly (diameter 2.67 -> 1.55)
    with no discontinuity anywhere -- a healthy tube, confirmed by watching the movie -- while the
    blend climbs from 4 to 162. It was counting the tube.

    So the bias does not run toward caution, it runs AGAINST THE TARGET PHENOTYPE: the better the
    tube, the more "damage" is reported. On that basis I condemned 13 of 24 archived runs as
    "peaked after the mesh broke". That verdict is withdrawn.

    Therefore: if only the blend is available, this returns NO HORIZON and says why. A number
    computed from the wrong quantity is worse than no number, because it will be used.

    `broken_n` is the right key precisely because it is TOPOLOGICAL -- under-connected cells and
    rings that are not valid polygons. Curvature and cell size cannot manufacture it, so a tube
    cannot look like damage to it.

    WHY THE BAR IS 3 AND NOT 20. Cedric proposed a low absolute bar of 10-20, but that was
    calibrated by eye against the GREY cells -- the legacy blend, whose healthy baseline is in the
    hundreds because it counts every just-divided sliver. For genuinely broken cells the measured
    baseline is different in kind: `probe_fault_modes` found **exactly zero** across 300 frames of
    a normally dividing, growing tissue. Broken cells are not a background rate to be tolerated,
    they are an event. So the bar is 3 -- enough to ride out a single transient during a division
    or a neighbour swap, low enough that the horizon trips as the tearing STARTS rather than once
    it is well underway. On a synthetic tear the difference matters: a bar of 20 admitted a third
    of the spike, a bar of 3 admits none of it.
    """
    if "broken_n" not in series:
        return {"horizon": None, "counted": None,
                "why": ("no `broken_n` recorded. REFUSING to fall back to the legacy blended "
                        "`hollow_n`: it correlates with the tip-cell count at r=+0.97, i.e. it "
                        "counts the tube, not the damage. Re-run with the three failure modes "
                        "separated. A horizon from the wrong quantity is worse than none.")}
    key = "broken_n"
    y = np.asarray(series[key], dtype=float)
    tt = np.arange(len(y)) if t is None else np.asarray(t)
    out = {"criterion": f"{key} >= {n_bar} cells", "counted": key}

    dmg = np.where(y >= first_bar)[0]
    out["first_damage"] = int(tt[dmg[0]]) if len(dmg) else None

    # THE HORIZON IS "THE LAST FRAME THE MESH WAS STILL CLEAN", not "the frame a threshold was
    # crossed". A real tear never heals: once faces stop being faces they stay broken. So the
    # honest boundary is the start of the damage that PERSISTS to the end of the run, and
    # everything before it is sound physics.
    #
    # A threshold cannot do this job, because damage and the elongation spike are SIMULTANEOUS --
    # the spike IS the tearing. Measured on a synthetic tear, peak elongation admitted:
    #        raw 30.00      bar=20 -> 11.37      bar=3 -> 3.45      last-clean -> 2.05 (the truth)
    # Any bar lets part of the spike through; only the last-clean rule excludes it.
    #
    # A TRANSIENT IS NOT A TEAR. A face may be briefly under-connected mid-division or mid-swap
    # and then recover. Because we take the start of the run of damage that reaches the END, an
    # early blip that heals is correctly ignored.
    clean = np.where(y < first_bar)[0]
    if not len(clean):
        return {**out, "horizon": int(tt[0]), "horizon_idx": 0, "complete": False,
                "valid_frac": 0.0, "why": f"{key} >= {first_bar} from the very first sample"}
    last_clean = int(clean[-1])
    if last_clean == len(y) - 1:
        return {**out, "horizon": int(tt[-1]), "horizon_idx": len(y) - 1, "complete": True,
                "why": f"{key} never sustained damage to the end (max {int(np.nanmax(y))})"}
    sustained = np.nanmax(y[last_clean + 1:])
    return {**out, "horizon": int(tt[last_clean]), "horizon_idx": last_clean, "complete": False,
            "valid_frac": round(last_clean / max(len(y) - 1, 1), 3),
            "sustained_peak": int(sustained),
            "why": (f"last frame with {key} < {first_bar} was {int(tt[last_clean])}; damage from "
                    f"there never recovers (reaching {int(sustained)}). First damage anywhere: "
                    f"frame {out['first_damage']}.")}


# ------------------------------------------------------------------------ temporal reductions
# (the six numbers a trajectory is allowed to become)
#
# WHY A REDUCTION AND NOT ANOTHER SHAPE WORD
# ------------------------------------------------------------------------------------------------
# `classify` returns a WORD. A word cannot be scored: `predict.Clause.check` looks a metric up BY
# EXACT KEY in the run summary and calls `float()` on it, so only a scalar is checkable, and
# `predict.parse` will only recognise a name that exists in `KNOWN_METRICS`. So everything above
# this line -- the whole trajectory channel -- reaches an agent as prose and leaves the record as
# prose. These six turn a curve into six scalars that a prediction can name.
#
# They are also the answer to a specific hole. On `okuda_route` every shape word is reassuring:
# `act_max` is `peaked`, `protr` is `converged`, the tissue is a clean sphere of 3,975 cells with
# genus 0. What no word says is that inside the evidence window `act_max` runs from 0.004 to
# 951,288 -- `_span` says it in one number, 1.3e6.
#
# THREE TIERS, SO COLUMNS HAVE DIFFERENT LENGTHS
# ------------------------------------------------------------------------------------------------
# Measured today on a real 3,975-cell mesh: the full `frame_metrics` costs 1410 ms/frame
# (hollow_flags 583, face_polygons 301, centroids 30) while the CHEMISTRY-ONLY metrics cost
# 0.12 ms. Sampling all of it every frame costs more than the simulation. So the recorder now runs
# three tiers -- chemistry every frame, centroid metrics every frame, mesh metrics every 25 frames
# -- and `frames_1.npz` already looks like that: 901 chemistry samples beside 37 mesh samples.
#
# THEREFORE THE HORIZON IS A FRAME NUMBER AND MUST BE CONVERTED PER COLUMN. `evidence_horizon`
# returns a frame; the columns are indexed by SAMPLE. On this record the horizon is frame 150,
# which is index 150 of 901 in the chemistry tier and index 6 of 37 in the mesh tier. Using the
# frame as a row index -- the bug found and fixed today -- truncates the chemistry tier to its
# first 7 frames and calls the result "the run". Every column therefore carries its own frame
# numbers here (`frames_by_col`), and nothing is truncated without them.
#
# WHY `_floor` AND NOT `_min`
# ------------------------------------------------------------------------------------------------
# Three kept series already END in `_min` -- `shape_idx_min`, `act_min`, `mesh_act_min` -- and
# `predict._METRIC_ALT` is a LONGEST-FIRST alternation over `KNOWN_METRICS`. A suffix `_min` would
# make `shape_idx_min` ambiguous with a hypothetical `shape_idx` + `_min`, and the longest-first
# rule silently decides which reading wins. `_floor` collides with nothing, and there is currently
# no minimum of any kind in `KNOWN_METRICS`, so nothing is being renamed -- only named.
#
# WHAT EACH ONE IS FOR (and what it reads on okuda_route, horizon = frame 150)
#
#   _final          V[h] exactly -- the value AT the horizon, never the last finite value before
#                   it. It is the only reduction tied to one fixed moment, which is what makes the
#                   24 series of a single run comparable to each other; a "last finite" fallback
#                   would quietly compare frame 150 of one series to frame 38 of another. When the
#                   instrument was refusing at that moment the answer is None, and `_measured_frac`
#                   is how you tell refusal from breakage. `corr_act_rad_final` is None here.
#
#   _peak           max over the finite entries.        act_max_peak = 951,288 (at frame 5).
#   _floor          min over the finite entries.        act_max_floor = 0.004.
#
#   _trend          Spearman rank correlation between the finite entries and their FRAME NUMBERS.
#                   Rank-based, so a single monotone spike cannot manufacture a trend the way a
#                   least-squares slope can, and units drop out. Ties get average ranks, which is
#                   not a detail on this data: `red_frac` is exactly 0 in 59% of frames.
#                   n_cells_trend = +0.999 over the whole run -- growth, monotone.
#
#   _span           (peak - floor) / |median|, the blow-up / noise / rail detector. Grounded:
#                   act_max_span = 1.3e6 inside the window and 4.7e6 over the run, while every
#                   shape word for that run says "sphere". A rail reads 0; a live quantity reads
#                   order 1; six orders of magnitude reads six orders of magnitude.
#
#   _measured_frac  finite(V).mean(). Always defined, always in [0,1]. THE DENOMINATOR THAT MAKES
#                   THE OTHER FIVE HONEST: `corr_act_rad` is REFUSED whenever act_cv < 0.05, so on
#                   okuda_route it is finite in 4.6% of the window (23% of the run). A null there
#                   means "there was no pattern to correlate", not "the instrument broke", and
#                   without this number those two are indistinguishable in the record.
#
# NOT-A-NUMBER IS NOT A MEASUREMENT. NaN *and* +/-inf are both excluded everywhere here, and a
# reduction that would have been one comes back None. Two reasons, one of them mechanical: a peak
# of `inf` is not a result, and `json.dump` writes it as bare `Infinity`, which is not valid JSON
# -- a run summary that cannot be re-read is a run that did not happen.

REDUCTIONS = ("final", "peak", "floor", "trend", "span", "measured_frac")

# Columns that ARE the time axis. Reducing the frame column reports that frames increase.
# Skipped in automatic mode only: if a caller names one in `keys` it is reduced, because then it
# is being used as data (the self-test does exactly that to prove where a truncation landed).
_TIME_COLS = ("t", "frame", "frames", "time", "mesh_frame")


def _rankdata(a):
    """Ranks 1..n with TIES AVERAGED -- the same convention as scipy.stats.rankdata.

    Ties are the common case here, not an edge case: `red_frac` is exactly 0.0 in 59% of
    okuda_route's frames and `genus` is 0 in all of them. Ranking ties by their arrival order
    instead would read the arbitrary order of equal values as a trend.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    s = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(y, t):
    """Spearman rho of y against t, or None when it is undefined.

    Undefined means one of the two is CONSTANT: a flat series has no monotone direction, and
    "0.0" would be a claim (no trend) where the truth is "the question does not apply". Constant
    series are everywhere in this record -- genus 0, n_tubes 0 -- so this path is taken often.
    """
    ry, rt = _rankdata(y), _rankdata(t)
    if ry.std() < 1e-12 or rt.std() < 1e-12:
        return None
    r = float(np.corrcoef(ry, rt)[0, 1])
    if not np.isfinite(r):
        return None
    # Clipped and rounded so a perfectly monotone series reads exactly +1.0 rather than
    # 0.9999999999999999. A record that says 0.9999999999999999 invites the reader to wonder what
    # the missing 1e-16 was; it was the correlation of two rank vectors in floating point.
    return round(float(np.clip(r, -1.0, 1.0)), 12)


def _horizon_index(frames, horizon_frame):
    """The index of the last sample AT OR BEFORE a frame number. -1 if there is none.

    This is the whole three-tier correction in two lines. The mesh tier is sampled every 25
    frames, so a horizon of frame 150 is its index 6 and a horizon of frame 160 is ALSO its
    index 6 -- the last mesh measurement that is inside the trustworthy window. Never index 160,
    and never "round up to the next mesh sample": a sample taken after the horizon is taken after
    the mesh stopped being trustworthy, which is what the horizon exists to exclude.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.size == 0:
        return -1
    if horizon_frame is None:
        return frames.size - 1
    inside = np.where(frames <= float(horizon_frame))[0]   # NaN frames compare False -> excluded
    return int(inside[-1]) if inside.size else -1


def reduce_series(y, frames=None, horizon_frame=None):
    """The six scalars for ONE series. Values are floats or None -- never NaN, never inf.

    `frames` are the FRAME NUMBERS of y's samples, one per sample. `horizon_frame` is a frame
    number (what `evidence_horizon` returns), not a row index; everything after it is discarded
    before anything is computed, because those frames are not evidence.

    frames=None means "this series is its own clock" (sample i is frame i). That is exact for
    `_trend`, which is rank-based and so invariant to any increasing relabelling of time, and it
    is why the fallback is allowed at all -- but it is NOT safe for truncation, so `reduce_all`
    refuses it when a horizon is in play rather than guessing.
    """
    y = np.asarray(y, dtype=float).ravel()
    frames = (np.arange(y.size, dtype=float) if frames is None
              else np.asarray(frames, dtype=float).ravel())
    if frames.size != y.size:
        raise ValueError(f"reduce_series: {y.size} samples but {frames.size} frame numbers. "
                         "Each column carries its OWN frame column now (three sampling tiers); "
                         "they cannot be paired by position across tiers.")

    out = {k: None for k in REDUCTIONS}
    out["measured_frac"] = 0.0

    h = _horizon_index(frames, horizon_frame)
    if h < 0:
        # Nothing was sampled at or before the horizon. Not an error -- a mesh tier whose first
        # sample is frame 25 has no measurement inside a horizon of frame 10, and the honest
        # report of that is six nulls and measured_frac 0.
        return out
    V = y[:h + 1]

    finite = np.isfinite(V)                                # excludes NaN AND +/-inf
    out["measured_frac"] = float(finite.mean())
    if finite[-1]:
        out["final"] = float(V[-1])                        # V[h]: the horizon frame itself
    if not finite.any():
        return out

    Vf, Ff = V[finite], frames[:h + 1][finite]
    peak, floor = float(Vf.max()), float(Vf.min())
    out["peak"], out["floor"] = peak, floor

    # Four is the smallest n at which a rank correlation can be anything but +-1 or +-0.8, i.e.
    # the smallest n at which the number carries information rather than the sign of two points.
    if Vf.size >= 4:
        out["trend"] = _spearman(Vf, Ff)

    # SCALE-FREE, so 951,288 and 0.0065 are on the same axis. Median first because a single
    # blow-up frame drags the mean into the blow-up and the span then reports ~1 for a series
    # that moved six decades. Mean only when the median is numerically zero (red_frac: median
    # 0.0, mean 0.29 -- more than half its frames are exactly zero). If both are zero the series
    # is zero, the ratio is 0/0, and the answer is None rather than a fabricated 0.
    denom = abs(float(np.median(Vf)))
    if not denom > 1e-12:
        denom = abs(float(np.mean(Vf)))
    if denom > 1e-12:
        out["span"] = (peak - floor) / denom
    return out


def _frames_for(name, n, frames_by_col):
    """This column's frame numbers, or None if it has none of the right length."""
    if frames_by_col is None:
        return None
    fr = frames_by_col.get(name) if isinstance(frames_by_col, dict) else frames_by_col
    if fr is None:
        return None
    fr = np.asarray(fr, dtype=float).ravel()
    return fr if fr.size == n else None


def reduce_all(cols, frames_by_col=None, horizon_frame=None, keys=None):
    """Reduce many series to one FLAT dict of `name_reduction` -> float|None.

    Flat and scalar because that is the only shape the scorer can read: `predict.Clause.check`
    looks a metric up by exact key in the run summary. `act_max_span` is a name a prediction can
    be written against; `{"act_max": {"span": ...}}` is not.

    COLUMNS HAVE DIFFERENT LENGTHS. `frames_by_col` maps a column name to its frame numbers (or
    is a single array, when every column really does share one clock). Columns whose frames are
    missing are handled by `horizon_frame`:

      no horizon  -> reduced against their own index. Exact: `_trend` is rank-based, and
                     `_final`/`_peak`/`_floor`/`_span`/`_measured_frac` do not use time at all.
      a horizon   -> ValueError. A horizon is a frame number; applying it to a column whose
                     sampling stride you do not know is precisely the arithmetic that truncated
                     folded runs to their opening frames. There is no safe default, so this
                     raises at the seam instead of writing a wrong number into the record.

    Non-numeric and non-1D columns are skipped, for the reason spelled out in `classify_npz`:
    metrics.npz carries the per-frame morphology labels as a STRING column, and the last thing
    that called float() on every column died there and reported it as one parenthesis.
    """
    names = list(cols) if keys is None else [k for k in keys if k in cols]
    out = {}
    for name in names:
        v = np.asarray(cols[name])
        if keys is None and name in _TIME_COLS:
            continue
        if v.ndim != 1 or v.size == 0 or v.dtype.kind not in "fiub":
            continue
        fr = _frames_for(name, v.size, frames_by_col)
        if fr is None:
            if horizon_frame is not None:
                raise ValueError(
                    f"reduce_all: column {name!r} has {v.size} samples and no frame numbers of "
                    f"its own, so the evidence horizon (frame {horizon_frame}) cannot be turned "
                    f"into a row index for it. The tiers are sampled at different rates -- "
                    f"chemistry every frame, mesh every 25 -- so a frame is not a row. Pass "
                    f"frames_by_col[{name!r}].")
            fr = np.arange(v.size, dtype=float)
        for red, val in reduce_series(v, fr, horizon_frame).items():
            out[f"{name}_{red}"] = val
    return out


def report(run_dir, write=True):
    """Classify a run's curves and write curves.json beside them."""
    out = {"run": os.path.basename(run_dir.rstrip("/"))}
    series, t = {}, None

    mech = os.path.join(run_dir, "mechanics.npz")
    if os.path.exists(mech):
        out["mechanics"] = classify_npz(mech)
        z = np.load(mech)
        for k in z.files:
            series.setdefault(k, z[k])
        t = z["t"] if "t" in z.files else None

    # THE EVERY-FRAME TABLE, READ FIRST so the coarse mesh columns cannot overwrite a series
    # that exists at full resolution. okuda_route's activator is a period-53 limit cycle: at the
    # mesh stride of 25 that is 2.1 samples per cycle, below Nyquist, and the classification would
    # be of a beat rather than of the chemistry.
    fnpz = os.path.join(run_dir, "frames.npz")
    if os.path.exists(fnpz):
        zf = np.load(fnpz)
        out["frames"] = classify_npz(fnpz)
        for k in zf.files:
            series[k] = zf[k]
        if "frame" in zf.files:
            t = zf["frame"]

    mnpz = os.path.join(run_dir, "metrics.npz")
    mjson = os.path.join(run_dir, "metrics.json")
    if os.path.exists(mnpz):
        out["metrics"] = classify_npz(mnpz)
        z = np.load(mnpz)
        for k in z.files:
            series.setdefault(k, z[k])          # never over the every-frame table
        t = z["frame"] if "frame" in z.files else t
    elif os.path.exists(mjson):
        d = json.load(open(mjson))
        rows = d.get("series", [])
        if rows:
            cols = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in rows[0]}
            t = cols.get("frame")
            out["metrics"] = {k: classify(v, t) for k, v in cols.items() if k != "frame"}
            series.update(cols)

    # THE TIME COURSE ITSELF, not only a word for its shape. Eight samples per series is small
    # enough to sit in every prompt and is the difference between "act_max peaked" and "act_max
    # went 0.4 -> 17678 -> 0.01": the first is compatible with a healthy pattern, the second is a
    # blow-up followed by extinction, and only the second explains a movie that flashes red and
    # then goes white for the rest of the run.
    out["over_time"] = {k: _spark(v) for k, v in series.items()
                        if k in _CURVES and _spark(v)}

    out["horizon"] = evidence_horizon(out.get("metrics", {}), series, t)

    # The two series are sampled at DIFFERENT rates (mechanics 24, metrics 40 on the run this was
    # written against). They carry their own frame column so they are alignable, but nothing
    # currently aligns them -- flag it rather than silently comparing index to index.
    lens = {n: len(v) for n, v in (("mechanics", series.get("force_mean", [])),
                                   ("metrics", series.get("protr", []))) if len(v)}
    if len(set(lens.values())) > 1:
        out["sampling_mismatch"] = {**lens,
                                    "note": "different sample counts; align on the frame column"}
    if write:
        json.dump(out, open(os.path.join(run_dir, "curves.json"), "w"), indent=1, default=float)
    return out


# WHAT AN AGENT IS ALLOWED TO SEE HAPPEN OVER TIME. `report` classifies EVERY series in
# metrics.npz; this whitelist decides which of them reach a prompt, and it held seven keys, NOT
# ONE of them about the activator. So the chemistry's whole time course -- the thing the campaign
# is about -- was computed, classified `peaked` or `exploded`, written to curves.json, and
# filtered out one step before any agent could read it.
#
# Cedric, watching round 2: "a flash of red activity, 100% red, then a long period of white, no
# activity". okuda_route's act_max went to 17,678 at frame 350 and 0.0105 by frame 807 -- already
# classified `exploded` in that run's curves.json -- and no reader was ever shown the line.
_CURVES = (
    "protr", "protr_p99", "r_cv", "hollow_frac", "cells", "n_cells", "tube_diam",   # shape
    "gyr_prolate", "gyr_asphere", "reduced_volume", "shape_idx_med", "ray_single_frac",
    "act_max", "act_mean", "act_sd", "act_cv", "act_occupancy", "red_frac",         # the pattern
    "n_spots", "spot_frac", "corr_act_rad", "act_at_tip",                           # and its grip
    "force_mean", "tension_mean",                                                   # mechanics
)


def _spark(y, n=8):
    """n evenly spaced samples of a series, as text. A SHAPE WORD IS NOT A TIME COURSE: `peaked`
    is true of a gentle rise-and-fall and of a spike to 17,678 followed by extinction, and an
    agent asked to reason about the chemistry needs to see WHICH. Eight numbers is small enough
    to print for twenty series and enough to see a flash."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return ""
    idx = np.linspace(0, y.size - 1, min(n, y.size)).astype(int)
    return " ".join(f"{v:.3g}" for v in y[idx])


def summarise(rep, keys=_CURVES):
    """One line per interesting curve -- what goes into an agent's prompt.

    The numeric time course comes from `rep["over_time"]`, which `report` fills while the series
    are in hand; nothing here re-reads the disk.
    """
    lines = []
    for grp in ("frames", "metrics", "mechanics"):
        for k, v in (rep.get(grp) or {}).items():
            if keys and k not in keys:
                continue
            extra = ""
            if v["shape"] == "pinned":
                extra = f" at {v.get('pinned_at')}  <-- HELD AGAINST A LIMIT, not converged"
            elif v["shape"] == "peaked":
                extra = (f", best at {v['peak_at_frac']:.0%} through the run "
                         f"(peak {v['peak']:.3g} -> final {v['final']:.3g})"
                         f"  <-- the final frame is NOT the result")
            elif v["shape"] == "rising":
                extra = f" (still climbing at the end: {v['final']:.3g})  <-- run may be too short"
            elif v["shape"] == "exploded":
                extra = f" (late blow-up to {v['final']:.3g})  <-- suspect the mesh, not biology"
            sp = (rep.get("over_time") or {}).get(k) or ""
            lines.append(f"  {k:16} {v['shape']:10}{extra}")
            if sp:
                lines.append(f"  {'':16} over time: {sp}")
    h = rep.get("horizon") or {}
    if h.get("first_damage") is not None:
        lines.append(f"  first damaged cell at frame {h['first_damage']}")
    if h.get("horizon") is not None and not h.get("complete", False):
        lines.append(f"  EVIDENCE HORIZON  frame {h['horizon']} "
                     f"({h.get('valid_frac', 0):.0%} of the run is trustworthy) -- {h['why']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    OK, fails = "\033[92mok\033[0m", []

    def eq(got, want, what):
        if got != want:
            fails.append(f"{what}: got {got!r} want {want!r}")
        print(f"  [{OK if got == want else 'FAIL'}] {what:52} -> {got}")

    n = 40
    x = np.arange(n)
    eq(classify(np.ones(n))["shape"], "flat", "a constant is flat")
    eq(classify(np.r_[np.linspace(1, 5, 20), np.full(20, 5.0)])["shape"], "pinned",
       "tail EXACTLY constant -> pinned, not converged")
    eq(classify(np.linspace(1, 5, n))["shape"], "rising", "a straight climb is rising")
    eq(classify(5 - 4 * np.exp(-x / 5.0) + 1e-6 * np.sin(x))["shape"], "converged",
       "an asymptote that still wobbles is converged")
    eq(classify(np.r_[np.linspace(1, 3, 12), np.linspace(3, 1.2, 28)])["shape"], "peaked",
       "up then down is peaked")
    eq(classify(np.r_[np.linspace(1, 1.4, 30), np.linspace(1.4, 40, 10)])["shape"], "exploded",
       "a late blow-up is exploded")
    eq(classify(np.array([1.0, 2.0]))["shape"], "too_short", "two points is not a curve")

    print("\nthe case this exists for -- 32 runs that all ended at 1778:")
    cells = np.r_[np.linspace(150, 1778, 25), np.full(15, 1778.0)]
    v = classify(cells)
    eq(v["shape"], "pinned", "cells_end 1778 classifies as pinned")
    print(f"     pinned_at = {v['pinned_at']}  <- the vertex buffer, not the biology")

    print("\nplateau vs terminal spike -- identical peak, opposite meaning:")
    plateau = np.r_[np.linspace(1, 2.7, 20), 2.7 - 0.01 * np.abs(np.sin(np.arange(20)))]
    spike = np.r_[np.linspace(1, 1.3, 36), [1.5, 2.0, 2.4, 2.7]]
    eq(float(np.max(plateau).round(2)), 2.7, "same peak (plateau)")
    eq(float(np.max(spike).round(2)), 2.7, "same peak (spike)")
    eq(classify(plateau)["shape"], "converged", "  plateau -> converged")
    eq(classify(spike)["shape"], "exploded", "  spike   -> exploded")

    print("\nhorizon on real data:")
    rd = "/workspace/Plexus/log/okuda/r01_03_5e3159_3"
    if os.path.isdir(rd):
        rep = report(rd, write=False)
        print(summarise(rep))
    else:
        print("  (run dir absent, skipped)")

    # ---------------------------------------------------------------- temporal reductions
    def close(got, want, what, tol=1e-6):
        ok = (got is not None and abs(float(got) - want) <= tol * max(1.0, abs(want)))
        if not ok:
            fails.append(f"{what}: got {got!r} want ~{want!r}")
        print(f"  [{OK if ok else 'FAIL'}] {what:52} -> {got}")

    print("\nreductions -- the degenerate series that must not crash or lie:")
    six = set(REDUCTIONS)
    eq(set(reduce_series(np.array([np.nan, np.nan, np.nan]))), six, "always the same six keys")
    r = reduce_series(np.full(9, np.nan))
    eq((r["final"], r["peak"], r["floor"], r["trend"], r["span"]), (None,) * 5,
       "all-NaN -> five nulls ...")
    eq(r["measured_frac"], 0.0, "  ... and measured_frac 0.0, which is the point")
    r = reduce_series(np.array([np.inf, 1.0, np.inf]))
    eq((r["peak"], r["final"]), (1.0, None), "inf is NOT a measurement (and is not valid JSON)")
    close(r["measured_frac"], 1 / 3, "  it counts against measured_frac like a NaN")
    r = reduce_series(np.array([2.5]))
    eq((r["final"], r["peak"], r["floor"], r["span"], r["trend"], r["measured_frac"]),
       (2.5, 2.5, 2.5, 0.0, None, 1.0), "one sample: defined, span 0, trend refused")
    r = reduce_series(np.full(12, 7.0))
    eq((r["span"], r["trend"]), (0.0, None), "a constant has span 0 and NO trend (not 0.0)")
    r = reduce_series(np.zeros(12))
    eq(r["span"], None, "all zeros: 0/0 is None, not a fabricated 0")
    r = reduce_series(np.array([-4.0, -2.0, -1.0, -8.0]))
    eq((r["peak"], r["floor"]), (-1.0, -8.0), "negatives: peak is the largest, not the biggest")
    close(r["span"], 7.0 / 3.0, "  span uses |median| so it stays positive")
    r = reduce_series(np.array([0.0, 0.0, 0.0, 4.0]))       # red_frac's shape: mostly exactly 0
    close(r["span"], 4.0, "  median 0 -> falls back to |mean| (1.0), not to None")
    r = reduce_series(np.array([1e-3, 1e3, 1.0, 1.0]))
    eq((r["floor"], r["peak"]), (1e-3, 1e3), "six decades survive as floor and peak ...")
    close(r["span"], 999.999, "  ... and span reads them against the median (1.0)")
    r = reduce_series(np.array([5.0, 1.0, 4.0, 2.0, 3.0]), frames=np.arange(5) * 25.0)
    close(r["trend"], -0.3, "trend is Spearman against the FRAME numbers")
    eq(reduce_series(np.arange(3.0))["trend"], None, "3 finite points is not a trend")

    print("\none horizon, two sampling tiers -- a FRAME is not a ROW:")
    fchem, fmesh = np.arange(0, 901.0), np.arange(0, 901.0, 25)
    chem, mesh = np.arange(901.0), np.arange(0, 901.0, 25)          # each column == its own frame
    eq(reduce_series(chem, fchem, 150)["final"], 150.0, "chemistry tier ends AT frame 150")
    eq(reduce_series(mesh, fmesh, 150)["final"], 150.0, "mesh tier ends at frame 150 too ...")
    eq(reduce_series(mesh, fmesh, 160)["final"], 150.0,
       "  ... and at 160, the last CLEAN mesh sample")
    eq(reduce_series(mesh, fmesh, 150)["measured_frac"], 1.0, "  (7 samples, all measured)")
    r = reduce_series(mesh, fmesh, -1)
    eq((r["final"], r["measured_frac"]), (None, 0.0), "nothing sampled inside the horizon")
    flat = reduce_all({"a": chem, "b": mesh}, {"a": fchem, "b": fmesh}, horizon_frame=150)
    eq(sorted(flat), sorted(f"{n}_{r}" for n in "ab" for r in REDUCTIONS), "flat name_reduction")
    eq((flat["a_final"], flat["b_final"]), (150.0, 150.0), "both tiers cut at the same MOMENT")
    eq(flat["a_peak"] == flat["b_peak"] == 150.0, True, "  and neither at the same ROW")
    try:
        reduce_all({"a": chem}, None, horizon_frame=150)
        eq("no raise", "ValueError", "a horizon without frame numbers must REFUSE")
    except ValueError as e:
        eq("ValueError", "ValueError", f"refused: {str(e)[:34]}...")
    eq(set(reduce_all({"morphology": np.array(["sphere"] * 9), "frame": chem})), set(),
       "a label column and the clock are not curves")

    print("\nokuda_route, the real per-frame record (901 chem samples, 37 mesh):")
    fz = "/workspace/Plexus/log/okuda/okuda_route/frames_1.npz"
    if os.path.exists(fz):
        z = np.load(fz)
        f, mf = np.asarray(z["frame"], float), np.asarray(z["mesh_frame"], float)
        cols = {k[5:]: z[k] for k in z.files if k.startswith("chem_")}
        cols.update({k: z[k] for k in ("protr", "r_cv", "corr_act_rad", "n_cells")})
        fbc = {k: f for k in cols}
        for k in z.files:                       # the 25-frame tier, only where it is the ONLY
            if k.startswith("mesh_") and k != "mesh_frame" and k[5:] not in cols:
                cols[k[5:]] = z[k]
                fbc[k[5:]] = mf
        h = evidence_horizon({}, {"broken_n": np.asarray(z["mesh_broken_n"], float)}, mf)
        eq(h["horizon"], 150, "the horizon is FRAME 150 (mesh sample 6 of 37)")
        R = reduce_all(cols, fbc, horizon_frame=h["horizon"])
        eq(R["n_cells_final"], 2058.0, "chemistry truncated at frame 150 (row 6 would say 2000)")
        eq(R["broken_n_peak"], 1.0, "mesh truncated at ROW 6 (the full column peaks at 454)")
        close(R["act_max_span"], 1321209.78, "act_max_span: 0.004 -> 951288 inside the window", 1e-4)
        close(R["act_cv_final"], 0.03479, "act_cv_final: near-UNIFORM field at that same moment",
              1e-3)
        close(R["corr_act_rad_measured_frac"], 0.0464,
              "corr_act_rad refused in 95% of the window", 1e-2)
        eq(R["corr_act_rad_final"], None, "  so its final is null -- no pattern, not no ruler")
        eq(R["act_cv_measured_frac"], 1.0, "  while act_cv itself is measured every frame")
        eq(R["genus_span"], None, "genus is 0 throughout: span None, not 0/0")
        eq(any(k.endswith("_min") for k in R), False, "no key ends in _min (it would shadow "
                                                      "shape_idx_min)")
        eq(("shape_idx_min_floor" in R), True, "  the kept _min series still reduces")
        bad = [k for k, v in R.items() if v is not None and not np.isfinite(v)]
        eq(bad, [], "every value is a float or None -- no NaN, no inf")
        full = reduce_all(cols, fbc)
        eq(full["broken_n_peak"], 454.0, "with no horizon the whole run is read")
        close(full["n_cells_trend"], 0.9989, "n_cells_trend +1.0: it grows, monotonically", 1e-2)
        close(full["act_max_span"], 4740612.9, "act_max_span over the run", 1e-4)
        try:                    # ties are 59% of red_frac, so the tie rule is not a detail
            from scipy.stats import spearmanr
            close(full["red_frac_trend"], float(spearmanr(z["chem_red_frac"], f).statistic),
                  "trend agrees with scipy on a column that is 59% ties", 1e-9)
        except ImportError:
            print("  [--] scipy absent; tie handling not cross-checked")
        print(f"     act_max  floor {full['act_max_floor']:.4g} peak {full['act_max_peak']:.6g}"
              f" final {full['act_max_final']:.4g}   <- and every shape word says 'sphere'")
    else:
        print("  (frames_1.npz absent, skipped)")

    print("\n" + ("time_analysis OK" if not fails else f"{len(fails)} FAILURES:\n  "
                                                     + "\n  ".join(fails)))
    raise SystemExit(1 if fails else 0)

# --------------------------------------------------------------------------- oscillation
def oscillation(y, stride=1, min_cycles=4.0, peak_over_median=6.0, min_spc=4.0):
    """Is this series OSCILLATING, and with what period? Returns {} when it is not.

    WHY THIS EXISTS. `okuda_route`'s activator is not the single flash it looked like: sampled
    every 23 frames it shows fourteen separate 100%-red episodes across 900 frames, swinging
    0.010 -> 17,680 -> 0.010, and the gaps between detections are 69, 46, 46, 116, 46, ... --
    every one an exact multiple of the sampling interval. That is not a measurement of an
    oscillation, it is a measurement of the BEAT between an oscillation and our sampling. The true
    period was unrecoverable, and `red_frac` = 1 means the growth operator is acting on every cell
    at once, which grows a sphere uniformly. Whether that is Okuda's mechanism failing or a
    numerical relaxation oscillation is the difference between a finding and a bug.

    THREE THINGS MAKE THIS WORK ON THIS DATA, and each is a decision, not a default:

      LOG SPACE.  The activator spans six orders of magnitude. In linear space the periodogram is
      dominated by the three largest spikes and reports their spacing, not the period; in log
      space a relaxation oscillation becomes a bounded, roughly periodic wave.

      DETREND AND WINDOW.  A trend is a half-cycle of an infinitely long period and leaks across
      the whole spectrum; a hard-edged record leaks at every frequency. Least-squares line removed,
      Hann window applied.

      A CYCLE COUNT, NOT JUST A PEAK.  A periodogram ALWAYS has a maximum. A "period" of 400
      frames in a 900-frame record is two cycles and is not evidence of anything. Nothing is
      reported below `min_cycles`, and the peak must also stand `peak_over_median` above the
      spectrum's own median -- a robust noise floor that does not assume a noise model.

      AND A FLOOR ON SAMPLES PER CYCLE.  Nyquist says two samples per cycle; that is the limit at
      which a sinusoid is representable, NOT the limit at which one is measurable from noisy data.
      Calibrated against white noise, this detector reported a confident period of 2.7 samples --
      a peak in the top third of the spectrum where noise alone throws up spikes. `min_spc` = 4
      is the same rule used to decide the sampling rate, applied to the detector: below four
      samples per cycle nothing is claimed. It removed the only false positive in the suite.

    Returns {} rather than a null-valued dict, so a prediction naming `..._period` on a series
    that does not oscillate scores `not measured` -> inconclusive, which is the honest verdict.
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 16 or y.std() < 1e-12:
        return {}
    # log space when the series spans decades and is positive throughout
    pos = y > 0
    span = (y[pos].max() / max(y[pos].min(), 1e-300)) if pos.all() else 1.0
    z = np.log10(y) if (pos.all() and span > 100) else y.astype(float)
    space = "log10" if z is not y else "linear"

    t = np.arange(n, dtype=float)
    z = z - np.polyval(np.polyfit(t, z, 1), t)          # detrend
    if z.std() < 1e-12:
        return {}
    # ZERO-PADDED, so a period that does not land on a bin is not halved. 900 samples give bins
    # at 900/k frames: k=19 is period 47.4 and k=20 is 45.0, so a true period of 46 is SPLIT
    # between two bins and each carries half its power -- while the second harmonic, at 39.1,
    # happens to sit almost exactly on a bin and therefore wins. Calibrated: without padding the
    # detector returned 23.1 for a signal of period 46, and lost a period-120 sine entirely.
    # Padding interpolates the spectrum; it adds no information and removes an artefact.
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(z * w, n=4 * n)) ** 2
    P[0] = 0.0                                          # DC carries the window, not the signal

    # AGAINST A LOCAL BACKGROUND, NOT A GLOBAL MEDIAN. Calibrated on 300 random walks -- which is
    # what a real trajectory looks like, not white noise -- the global-median test fired on 6.3%
    # of them AT EVERY THRESHOLD from 6 to 200. Raising the bar could not help: a random walk has
    # a 1/f^2 spectrum, so its largest bin genuinely IS hundreds of times the median, and the test
    # was measuring redness rather than periodicity.
    #
    # Dividing the periodogram by a running median of itself removes any smooth 1/f^alpha
    # background and leaves only what stands out from ITS OWN NEIGHBOURHOOD -- which is what an
    # oscillation does and drifting noise does not. The running median (not mean) is used so the
    # peak cannot inflate the background it is being tested against.
    m = max(9, (P.size // 12) | 1)                      # odd window, ~8% of the spectrum
    half = m // 2
    pad = np.pad(P, half, mode="edge")
    bg = np.array([np.median(pad[i:i + m]) for i in range(P.size)])
    ratio = P / np.maximum(bg, 1e-300)
    ratio[0] = 0.0

    # THE FUNDAMENTAL, NOT ITS SEVENTH HARMONIC. A relaxation oscillation -- a sharp spike and a
    # slow decay, which is exactly what this activator does -- is a HARMONIC COMB: bins at k, 2k,
    # 3k ... all carry power. Taking the largest whitened bin then picks whichever harmonic
    # happens to stand out best against its own neighbourhood, and for the calibration signal of
    # period 46 it returned 6.6 -- that is 46/7, reported with total confidence.
    #
    # Scoring each candidate by the sum of its own harmonic series fixes it: the true fundamental
    # is supported by every harmonic at once, a harmonic of it is supported by only a sparse
    # subset. This is the harmonic-sum spectrum, and it is why a wrong answer of exactly period/7
    # was worth chasing rather than thresholding away.
    # bins now index the PADDED spectrum: bin k is period 4n/k samples
    kmax = max(1, int(ratio.size / max(min_cycles, 1.0)))
    scores = np.zeros(ratio.size)
    for kk in range(1, kmax + 1):
        h = ratio[kk::kk][:4]                            # kk, 2kk, 3kk, 4kk
        scores[kk] = float(h.sum())
    k = int(np.argmax(scores))
    if k == 0:
        return {}
    cycles_seen = 4.0 * n / (4.0 * n / max(k, 1))        # = k/4 cycles in the record
    cycles_seen = k / 4.0
    strength = float(ratio[k])
    period_samples = 4.0 * n / k
    if (cycles_seen < min_cycles or strength < peak_over_median
            or period_samples < min_spc):
        return {}
    return {"period": round(period_samples * stride, 1),      # in FRAMES, not samples
            "cycles_seen": round(cycles_seen, 1),
            "strength": round(strength, 1),
            "space": space}

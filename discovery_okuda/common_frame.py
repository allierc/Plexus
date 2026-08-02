#!/usr/bin/env python
"""common_frame -- compare a batch at a frame every run in it was still valid.

PHASE 0, ITEM 8. THE DEFECT THIS CLOSES
================================================================================================
The evidence horizon is PER RUN: each run gets its own last frame before its mesh started
tearing, and every number after that is withheld. That was the right fix and it is in place.

It does not make a BATCH comparable. Two runs stop being valid at different moments, and the
round still compares their "final" states against each other. If run A tears at frame 300 and
run B holds to 700, then `A_final vs B_final` puts a torn mesh beside a healthy one and reports
the difference as biology. Both sides are individually valid, which is exactly why nobody
notices -- it is the same shape as the two errors that have already cost this campaign a batch
each: the comparison is wrong while both operands are right.

    "The instrument lies before the physics does. Every error was a COMPARISON,
     not a computation."                                    -- RESUME.md, hard-won rules

So a batch is read at ONE frame: the earliest horizon in the batch. Runs are then being asked
the same question at the same moment in their own development, which is the only way a
difference between them can be attributed to the mechanism that differs.

WHAT THIS DELIBERATELY DOES NOT DO
------------------------------------------------------------------------------------------------
It does not replace the per-run reading. A run's own peak and final, at its own horizon, remain
the honest description of THAT run and are what the scoreboard reports. The common frame is for
BETWEEN-run statements only, and the two are reported side by side so the gap is visible: a
mechanism whose effect appears only after the batch's common frame has not been demonstrated on
this batch, and the loop should say so rather than quietly compare endpoints.

It also does not interpolate. A frame that was recorded is used; the nearest recorded frame at
or before the common horizon is taken, and its actual index is reported. Inventing a value
between two samples to make a table line up is the kind of tidiness that hides a tear.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = os.path.join(ROOT, "log", "okuda")


def _series(run_dir):
    """The per-frame table for one run, or None. Never guesses when the file is absent."""
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    s = d.get("series") if isinstance(d, dict) else d
    return s if isinstance(s, list) and s else None


def _horizon(run_dir, series):
    """This run's last trustworthy frame. Falls back to its last recorded frame, and SAYS so."""
    try:
        import curve_shape as CS
        rep = CS.report(run_dir, write=False)
        hz = (rep or {}).get("horizon", {})
        h = hz.get("horizon") if isinstance(hz, dict) else None
        if h is not None:
            return int(h), "measured"
    except Exception:
        pass
    return int(series[-1].get("frame", len(series) - 1)), "no horizon computed -- last frame used"


def batch_common_frame(run_names, log_dir=LOG):
    """The frame at which this batch may be compared, and why it is that frame.

    Returns a dict carrying the decision AND its cost: which run set the limit, and how much of
    every other run is being discarded to meet it. A common frame that throws away 80% of the
    batch is not a comparison, it is one run's early death imposed on everyone, and the caller
    has to be able to see that rather than receive a tidy number.
    """
    runs, missing = {}, []
    for nm in run_names:
        d = os.path.join(log_dir, nm)
        s = _series(d)
        if s is None:
            missing.append(nm)
            continue
        h, why = _horizon(d, s)
        runs[nm] = {"series": s, "horizon": h, "horizon_note": why,
                    "last": int(s[-1].get("frame", len(s) - 1))}
    if not runs:
        return {"frame": None, "why": "no run in the batch has a per-frame record",
                "missing": missing, "runs": {}}

    limiter = min(runs, key=lambda k: runs[k]["horizon"])
    frame = runs[limiter]["horizon"]
    discarded = {nm: round(1.0 - frame / max(1, r["last"]), 3) for nm, r in runs.items()}
    return {
        "frame": frame,
        "limiter": limiter,
        "why": (f"{limiter} stops being valid at frame {frame}; every run is read there so the "
                f"batch is answering one question at one moment"),
        "missing": missing,
        "discarded_fraction": discarded,
        "runs": {nm: {"horizon": r["horizon"], "last": r["last"],
                      "note": r["horizon_note"]} for nm, r in runs.items()},
        "warning": ("the common frame discards more than half of the longest run -- the batch is "
                    "being judged on one run's early failure"
                    if max(discarded.values()) > 0.5 else None),
    }


def read_at(run_name, frame, keys=None, log_dir=LOG):
    """One run's numbers at (or just before) `frame`, with the frame actually used."""
    s = _series(os.path.join(log_dir, run_name))
    if s is None:
        return None
    at = None
    for row in s:
        if int(row.get("frame", -1)) <= frame:
            at = row
        else:
            break
    if at is None:
        return None
    out = {"frame_used": int(at.get("frame", -1))}
    for k, v in at.items():
        if keys is None or k in keys:
            out[k] = v
    return out


def compare_batch(run_names, keys=("protr", "cells", "hollow_frac", "act_max"), log_dir=LOG):
    """The batch at one frame, alongside each run's own final -- so the difference is visible."""
    dec = batch_common_frame(run_names, log_dir=log_dir)
    if dec["frame"] is None:
        return dec
    rows = {}
    for nm in run_names:
        at = read_at(nm, dec["frame"], keys=keys, log_dir=log_dir)
        s = _series(os.path.join(log_dir, nm))
        own = {k: s[-1].get(k) for k in keys} if s else {}
        rows[nm] = {"at_common": at, "at_own_end": own}
    dec["rows"] = rows
    return dec


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or [d for d in sorted(os.listdir(LOG))
                             if os.path.exists(os.path.join(LOG, d, "metrics.json"))][:6]
    print("=" * 92)
    print("COMMON-FRAME COMPARISON -- one question, one moment")
    print("=" * 92)
    r = compare_batch(names)
    if r["frame"] is None:
        print(" ", r["why"])
        raise SystemExit(1)
    print(f"\n  common frame : {r['frame']}   (set by {r['limiter']})")
    print(f"  because      : {r['why']}")
    if r.get("missing"):
        print(f"  NO RECORD    : {', '.join(r['missing'])} -- excluded, not assumed comparable")
    if r.get("warning"):
        print(f"  WARNING      : {r['warning']}")
    print(f"\n  {'run':30}{'horizon':>9}{'last':>7}{'discarded':>11}")
    for nm, d in r["runs"].items():
        print(f"  {nm[:29]:30}{d['horizon']:>9}{d['last']:>7}"
              f"{r['discarded_fraction'][nm]:>11.0%}")
    print(f"\n  {'run':30}{'protr @common':>15}{'protr @own end':>16}")
    for nm, d in r["rows"].items():
        a = (d["at_common"] or {}).get("protr")
        b = (d["at_own_end"] or {}).get("protr")
        flag = ""
        if a is not None and b is not None and abs(b - a) > 0.15 * max(1.0, a):
            flag = "   <-- differs; the endpoint comparison would have been misleading"
        print(f"  {nm[:29]:30}{('%.3f' % a) if a is not None else '--':>15}"
              f"{('%.3f' % b) if b is not None else '--':>16}{flag}")
    print()

#!/usr/bin/env python
"""sweep_eye -- run a queue of archived trials back to back.

    python sweep_eye.py calib      # short probe runs to find the working point
    python sweep_eye.py final      # the full atlas + OKR, at rendering resolution

Each trial lands in `archive/tNN_<label>/` with its own spec.yaml, movie.mp4, strip.png,
curves.npz and diag.json, and the sweep prints one summary line per trial at the end, so
the search is a record and not a memory.
"""
from __future__ import annotations

import os
import sys
import json
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_eye

# ---- the queues ------------------------------------------------------------- #
CALIB = [
    # label,       overrides
    ("c_base",   dict(contract=26.0, tonic=0.20, gain=1.2)),
    ("c_strong", dict(contract=48.0, tonic=0.14, gain=1.6)),
    ("c_max",    dict(contract=75.0, tonic=0.12, gain=1.8)),
    ("c_slack",  dict(contract=48.0, tonic=0.06, gain=1.8)),
    ("c_stiffm", dict(contract=75.0, tonic=0.12, gain=1.8, muscle_youngs=95.0)),
    ("c_softeye", dict(contract=48.0, tonic=0.12, gain=1.8, sclera_youngs=200.0)),
]
CALIB_COMMON = dict(preset="probe", n_particles=30000, n_muscle_particles=1800,
                    n_grid=112, stride=10)

FINAL = [
    ("atlas", dict(preset="atlas", stride=3)),
    ("okr", dict(preset="okr", stride=3)),
]
FINAL_COMMON = dict(n_particles=90000, n_muscle_particles=3000, n_grid=144, stride=3)


def run_queue(queue, common, device="cuda:0"):
    rows = []
    for label, over in queue:
        cfg = dict(common)
        cfg.update(over)
        stride = cfg.pop("stride", 4)
        try:
            d, diag = run_eye.trial(label, device=device, stride=stride, movie=True, **cfg)
            rows.append((label, os.path.basename(d), diag["passed"],
                         diag["range_hvt_deg"], diag["max_settle_error_deg"],
                         diag["recruitment_correct"], diag["peak_shortening_pct"],
                         diag["centroid_drift_max_frac_radius"], diag["strain_p99"]))
        except Exception:
            traceback.print_exc()
            rows.append((label, "FAILED", False, None, None, None, None, None, None))

    print("\n" + "=" * 118)
    print(f"{'trial':<16}{'dir':<16}{'pass':<6}{'range h/v/t':<26}{'err':<7}"
          f"{'recruit':<9}{'short%':<8}{'drift':<8}{'strain':<7}")
    print("-" * 118)
    for r in rows:
        print(f"{r[0]:<16}{r[1]:<16}{str(r[2]):<6}{str(r[3]):<26}{str(r[4]):<7}"
              f"{str(r[5]):<9}{str(r[6]):<8}{str(r[7]):<8}{str(r[8]):<7}")
    print("=" * 118, flush=True)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "calib"
    if which == "calib":
        run_queue(CALIB, CALIB_COMMON)
    elif which == "final":
        run_queue(FINAL, FINAL_COMMON)
    else:
        raise SystemExit(f"unknown queue {which!r} (calib | final)")

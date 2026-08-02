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
# The active stress A and the muscle's PASSIVE stiffness E have to be raised together.
# A muscle shortens until its passive tension balances its active stress, so the steady
# shortening is ~A/(2mu+la) -- i.e. set by A/E alone -- while the FORCE it delivers is
# A x cross-section. Raising A on its own (t02) just made the muscles collapse to 88% and
# crush the globe. These trials sweep the pair at a roughly fixed A/E ~ 0.25.
# Round 2. The ceiling on A/E is a BUCKLING limit, not a force limit: at A/E = 0.30 (t04)
# the straps folded to 90% shortening and threw the globe. So A/E stays near 0.22 and the
# lever becomes GEOMETRY. For a muscle pulling a globe against its antagonist,
#     theta_max ~ (A/E) x (L/R),
# so muscle LENGTH buys gaze range -- and truncating the path to 55% had left the recti at
# L/R = 2.3 and the obliques at 1.15, against ~3.3 in a real orbit. These trials restore the
# length, spread the rectus origins around the annulus of Zinn, and widen the sclera
# stand-off so the belly is not spuriously welded to the globe by shared grid nodes along
# its whole arc of contact.
CALIB = [
    ("c_long",   dict(contract=55.0, muscle_youngs=250.0, tonic=0.12, gain=1.8,
                      mus_frac=0.88, mus_gap=0.038)),
    ("c_longer", dict(contract=55.0, muscle_youngs=250.0, tonic=0.06, gain=1.9,
                      mus_frac=0.95, mus_gap=0.042)),
    ("c_wide",   dict(contract=55.0, muscle_youngs=250.0, tonic=0.10, gain=1.8,
                      mus_frac=0.88, mus_gap=0.038,
                      mus_width=0.044, mus_thickness=0.027)),
]
CALIB_COMMON = dict(preset="probe", n_particles=45000, n_muscle_particles=2200,
                    n_grid=128, stride=6)

# Round 3: the working point from round 2 (long muscles, low tonic, wide stand-off) plus the
# per-muscle strength factor for the obliques and physiological command amplitudes.
# Round 3. Muscle passive stiffness was ABOVE its peak active stress, which is backwards:
# real muscle is passively soft and limits its own shortening through the FORCE-LENGTH
# relation, not through a stiff passive element. With `stretch_activation` (beta) switched on
# a muscle stops pulling once it has shortened by 1/beta, so the passive modulus can drop far
# below the active stress -- and the antagonist, which had been eating most of the agonist's
# force and capping the eye at ~11 deg, gets out of the way.
BEST = dict(contract=115.0, muscle_youngs=110.0, stretch_activation=4.0,
            tonic=0.18, gain=1.9, kp=0.11, kd=0.010, mus_frac=0.95, mus_gap=0.042,
            mus_width=0.040, mus_thickness=0.026)

# Round 4. The force-length relation lifted the force ceiling (52 deg of gaze against 11),
# but a soft, slender strap then BUCKLES: it folds up, losing 60% of its centreline length
# while the fibre stretch stays near 0.75, so the length-tension limiter never engages. Two
# fixes, both what the real plant does -- a chunkier cross-section (bending stiffness goes as
# w h^3) and real TONIC co-contraction, which keeps every muscle in tension and is precisely
# why extraocular motoneurons fire at high rates even in primary position.
VERIFY = [
    ("v_tone",  dict(BEST)),
    ("v_tone2", dict(BEST, contract=150.0, muscle_youngs=150.0, tonic=0.24,
                     stretch_activation=4.5)),
]
VERIFY_COMMON = dict(preset="probe", n_particles=45000, n_muscle_particles=2200,
                     n_grid=128, stride=5)

FINAL = [
    ("atlas", dict(BEST, preset="atlas")),
    ("okr", dict(BEST, preset="okr")),
]
FINAL_COMMON = dict(n_particles=90000, n_muscle_particles=3000, n_grid=128, stride=4)


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
    elif which == "verify":
        run_queue(VERIFY, VERIFY_COMMON)
    elif which == "final":
        run_queue(FINAL, FINAL_COMMON)
    else:
        raise SystemExit(f"unknown queue {which!r} (calib | verify | final)")

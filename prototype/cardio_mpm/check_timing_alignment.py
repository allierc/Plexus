#!/usr/bin/env python
"""check_timing_alignment.py -- verify Phase 1/2 timing alignment with GT.

CRITICAL CHECKS:
1. Atlas Phase 1: which sim frames = which real GT frames?
2. Phase 2 target: does the trainer fit the EXACT SAME loop frames as the atlas measures?
3. Boundary anchoring: should outer nodes follow GT during Phase 2, and at what frames?

Output: frame-by-frame timing ledger, ready for Phase 2 setup.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import find_peaks

import cardio_mpm_atlas as A
import cardio_mpm_data as D


def main():
    print("=" * 80)
    print("TIMING ALIGNMENT CHECK: Atlas Phase 1 vs Real GT vs Phase 2 target")
    print("=" * 80)

    # --- REAL GT: beat detection ---
    P = np.load(D.NPZ)["pos"].astype(np.float32)  # [F, 137^2, 2] in ~[0,1], real domain
    spd = np.linalg.norm(np.diff(P, axis=0), axis=2).mean(1)
    pk, _ = find_peaks(spd, height=spd.mean(), distance=20)
    onsets = [int(p) for p in pk]
    period = int(round(float(np.diff(pk).mean()))) if len(pk) > 1 else 50

    fb = -2 % len(onsets)  # trainer's default fit beat
    onset = onsets[fb]
    end = onsets[fb + 1] if fb + 1 < len(onsets) else min(onset + period, P.shape[0])
    gt_beat_frames = list(range(onset, end + 1))
    gt_duration = len(gt_beat_frames)  # actual count, not end - onset

    print(f"\nREAL GT DATA (cardio_real.npz):")
    print(f"  Total frames: {P.shape[0]}")
    print(f"  Beat onsets detected: {onsets}")
    print(f"  Period (mean inter-onset): {period} frames")
    print(f"  Fit beat: beat #{fb} (onset index in onsets list)")
    print(f"  Fit beat frames: [{onset}:{end+1}] ({gt_duration} frame duration)")
    print(f"  Fit beat frame list: {gt_beat_frames}")

    # --- ATLAS Phase 1: forward simulation window ---
    print(f"\nATLAS Phase 1 FORWARD SIM (cardio_mpm_atlas.py):")
    print(f"  PERIOD constant: {A.PERIOD} frames")
    print(f"  Default n_frames: 105")
    print(f"  Measurement window: frames [{A.PERIOD}:{min(2*A.PERIOD + 3, 105)}]")
    print(f"    → Start: frame {A.PERIOD}")
    print(f"    → End: frame {min(2*A.PERIOD + 3, 105)}")
    print(f"    → Duration: {min(2*A.PERIOD + 3, 105) - A.PERIOD} frames")

    # --- CRITICAL MISMATCH CHECK ---
    atlas_duration = min(2 * A.PERIOD + 3, 105) - A.PERIOD
    print(f"\n⚠️  DURATION MISMATCH CHECK:")
    print(f"  Real GT duration: {gt_duration} frames")
    print(f"  Atlas measurement: {atlas_duration} frames")
    if atlas_duration == gt_duration:
        print(f"  ✓ MATCH: both {gt_duration} frames")
    else:
        print(f"  ✗ MISMATCH: atlas captures {atlas_duration}, GT is {gt_duration}")
        print(f"    → The morphology metrics will measure DIFFERENT temporal scales!")

    # --- PHASE 2 anchoring setup (preview) ---
    print(f"\nPHASE 2 INVERSE FIT (preview for cardio_mpm_train.py):")
    print(f"  When Phase 2 is built, it should:")
    print(f"    (1) Load real_disp from frames [{onset}:{end+1}]")
    print(f"    (2) Anchor outer boundary to GT at those frames")
    print(f"    (3) Fit interior nodes to match GT on those EXACT frames")
    print(f"    (4) Loss computed over [{onset}:{end+1}] ({gt_duration} frames)")
    print(f"\n  Recommended trainer args:")
    print(f"    --fit_beat {fb}           (which beat: {fb})")
    print(f"    --warmup 0                (one period to settle)")
    print(f"    --grad 0                  (full beat = {gt_duration} frames)")
    print(f"    --bwidth 0.06             (outer boundary width)")

    # --- KEY DECISION ---
    print(f"\n" + "=" * 80)
    print("KEY DECISION FOR PHASE 1 ATLAS:")
    print("=" * 80)
    if atlas_duration != gt_duration:
        print(f"❌ CURRENT ATLAS SETUP IS MEASURING THE WRONG DURATION!")
        print(f"\nRECOMMENDATION:")
        print(f"  Change atlas default: n_frames={onset + gt_duration + 10}")
        print(f"  OR: Accept that Phase 1 morphology is measured on a {atlas_duration}-frame window")
        print(f"      (will need to re-normalize when comparing to GT in Phase 2)")
    else:
        print(f"✓ Atlas and GT durations match ({gt_duration} frames)")
        print(f"  Phase 2 can directly compare morphology measured in Phase 1")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

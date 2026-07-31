#!/usr/bin/env python
"""instrument_gate -- the metric bank must reproduce OUR OWN judgement on runs we have labelled
by eye, BEFORE it is allowed to judge runs we have not.

    "If the instrument cannot reproduce our own judgement on known cases,
     it is not ready to judge unknown ones."

This gate is not a formality. Four metrics in this project have now been caught lying:

  * a global-median `hollow` flag that produced thin-tube false positives (390 -> 272 once made
    tube-aware);
  * a spot count thresholded at the 70th percentile that reported "101 spots" where the eye
    saw three (the threshold sat inside the noise);
  * a roughness inflated by unused reservoir vertices parked at the origin;
  * and most recently `tube_len_final = 14.69` scored on a SMALL BUD, giving an "aspect" of 9.30
    -- HIGHER than the archived run that is visibly a long thin tube. (Metrologist M3.)

--------------------------------------------------------------------------------------------
WHY IT MUST BE RUN FRESH, NOT RETROACTIVELY
--------------------------------------------------------------------------------------------
Only 23 of 311 archived runs carry a `metrics.json`, all from early rounds; none of the tube
rounds do. And none carries a trajectory, because the old archive stored only the final frame
(defect D7). So the archive CANNOT be re-scored -- which is precisely the consequence D7 was
recorded for, now realised rather than predicted.

The gate therefore runs the labelled compositions fresh, with full trajectories persisted, so
that every future metric revision CAN re-score them without re-simulating.

--------------------------------------------------------------------------------------------
THE ADMISSIBILITY CRITERION
--------------------------------------------------------------------------------------------
A metric is admissible if it:
  1. ORDERS  -- ranks the eye-labelled classes correctly (tube > bud > sphere);
  2. SEPARATES -- the gap between adjacent classes exceeds the spread within them;
  3. IS NOT FOOLED -- assigns no "tube" score to a run the eye calls a bud or a flood.

Criterion 3 is the one M3 failed, and it is the only one that catches a metric which is
monotone but miscalibrated.

    python instrument_gate.py --submit          # run the labelled set on the L4 partition
    python instrument_gate.py --score           # score whatever has completed
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

LOG = os.path.join(ROOT, "log", "okuda")

# ---------------------------------------------------------------------------------------------
# THE EYE LABELS. Authored by looking at the rendered strips -- the archived ones where they
# exist, and the fresh ones otherwise. `rank` is the ordering the metric must reproduce.
# These are the ground truth of this gate and must not be edited to make a metric pass.
# ---------------------------------------------------------------------------------------------
LABELLED = [
    # ⚠ RE-LABELLED 2026-07-30 FROM THE RENDERED FRAMES OF THESE RUNS.
    # The first version took labels from the REPORT -- which describes the ARCHIVED runs -- while
    # scoring FRESH ones. round_40_mc8 is a long thin tube in the archive and a BUD here, so it
    # was ranked 4 while rendering a 2. That is the gate's own failure mode (judging by
    # provenance instead of by looking) committed one level up, and it made the gate reject
    # every metric. Ground truth for this gate is ONLY what these runs render.
    dict(config="ref_uniform_inflation", eye="exploded", rank=0,
         note="at 900 frames uniform inflation EXPLODES into spikes (the 130-frame smoke was a "
              "clean sphere). Kept as rank 0: it is not a tube, so a metric that scores it high "
              "is fooled -- this is the 'lumpy blob scores well' control."),
    dict(config="round_44_base",     eye="sphere", rank=1,
         note="large smooth sphere with a tiny nub"),
    dict(config="round_41_hertwig",  eye="bud",    rank=2, note="sphere + small bud"),
    dict(config="round_41_relax60",  eye="bud",    rank=2, note="sphere + small bud"),
    dict(config="round_40_mc8",      eye="bud",    rank=2,
         note="sphere + small bud. The ARCHIVE for this preset is a long thin tube; the "
              "clock-fixed substrate renders a bud (Metrologist D1d). We score what ran."),
    dict(config="round_42_k05",      eye="spike",  rank=3,
         note="monolayer, growth-driven: clearly elongated"),
    dict(config="round_42_k05_ex4",  eye="spike",  rank=4,
         note="monolayer + extrusion: the MOST elongated -- a long thin spike"),
    dict(config="round_21_gs",       eye="bud",    rank=2, note="Gray-Scott stable-spot regime"),
]

CLASS_ORDER = ["exploded", "sphere", "bud", "spike"]

# metrics the gate judges. `tube_like` metrics should increase with rank.
JUDGED = ["ta_aspect_len_over_diam", "ta_tube_len_final", "ta_n_tubes_final",
          "protr_final", "protr_peak", "retention", "n_cells_final"]


# Provenance the labels were authored against. A label describes a RENDERED RUN, so it is only
# valid for the run it was written from. `ref_uniform_inflation` was silently re-run from 41
# frames to 901 between the gate storing PASS and today; the gate re-read diag.json, scored a
# DIFFERENT simulation under the same label, and could not tell. Record it, and say so.
LABELLED_FRAMES = 901


def load_scores():
    out = {}
    for row in LABELLED:
        d = os.path.join(LOG, row["config"], "diag.json")
        if not os.path.exists(d):
            continue
        try:
            out[row["config"]] = json.load(open(d)).get("summary", {})
        except Exception:
            pass
    return out


def validity(summary):
    """The campaign's OWN evidence rule, applied to the gate's own inputs.

    `critic.check_posthoc` refuses a run that saturated its buffer or scheduled an operator that
    never acted; `control.score_run` gives it -inf. The gate was exempting itself: it calibrated
    the metric bank on runs the campaign would never score. `ref_uniform_inflation` and
    `round_21_gs` both terminate at n_cells 15002 against a 30000-vertex reservoir -- for a
    trivalent mesh V = 2F-4, so 30000 caps F at exactly 15002. Their late frames are a reservoir
    overflow, not a morphology, and demanding that a metric behave sensibly on them is demanding
    robustness to a crash.
    """
    from critic import check_posthoc
    return [r.code for r in check_posthoc(summary)]


def _separates(vals, ranks):
    """Criterion 2, which was documented in the module docstring and never implemented.

    Between-class separation must exceed within-class spread, or an 'ordering' is noise that
    happens to sort. Returns (ok, why); vacuous when no class has two members.
    """
    import statistics as st
    by = {}
    for v, r in zip(vals, ranks):
        by.setdefault(r, []).append(v)
    rk = sorted(by)
    if len(rk) < 2:
        return False, "only one class present"
    spread = max([st.pstdev(by[r]) for r in rk if len(by[r]) > 1] or [0.0])
    gaps = [st.mean(by[b]) - st.mean(by[a]) for a, b in zip(rk, rk[1:])]
    worst = min(gaps)
    if spread == 0.0:
        return True, f"min gap {worst:+.3f}, no within-class spread to compare (vacuous)"
    return worst > spread, f"min gap {worst:+.3f} vs within-class spread {spread:.3f}"


def score():
    scores = load_scores()
    have = [r for r in LABELLED if r["config"] in scores]
    missing = [r["config"] for r in LABELLED if r["config"] not in scores]

    print("=" * 96)
    print("INSTRUMENT GATE -- can the metric bank reproduce our own eye labels?")
    print("=" * 96)
    if missing:
        print(f"\n  NOT YET RUN ({len(missing)}): {', '.join(missing)}")
    if len(have) < 4:
        print(f"\n  only {len(have)} labelled runs available -- run --submit first. GATE: NOT RUN")
        return 2

    print(f"\n{'config':26} {'eye':8} {'rk':>3}  " +
          "  ".join(f"{m.replace('ta_','').replace('_final',''):>14}" for m in JUDGED))
    for r in sorted(have, key=lambda r: r["rank"]):
        s = scores[r["config"]]
        cells = "  ".join(f"{float(s.get(m, float('nan'))):>14.3f}" for m in JUDGED)
        print(f"{r['config']:26} {r['eye']:8} {r['rank']:>3}  {cells}")

    # ------------------------------------------------------------------ provenance
    drift = [(r["config"], scores[r["config"]].get("frames"))
             for r in have if scores[r["config"]].get("frames") != LABELLED_FRAMES]
    if drift:
        print(f"\n--- PROVENANCE DRIFT (labels were authored at {LABELLED_FRAMES} frames) ---")
        for c, f in drift:
            print(f"  {c:26} ran {f} frames -- the label does not describe this run")

    # ------------------------------------------------------------------ validity
    invalid = {r["config"]: validity(scores[r["config"]]) for r in have}
    invalid = {c: v for c, v in invalid.items() if v}
    valid = [r for r in have if r["config"] not in invalid]
    if invalid:
        print("\n--- REFUSED BY THE CAMPAIGN'S OWN EVIDENCE RULE (critic.check_posthoc) ---")
        for c, codes in invalid.items():
            rk = next(r["rank"] for r in have if r["config"] == c)
            print(f"  rank {rk}  {c:26} {','.join(codes)}")
    # Excluding them makes the gate EASIER, and the ones excluded are the low-rank controls --
    # exactly the cases a metric has to survive. Removing them and declaring PASS is the gate
    # marking its own homework, so record whether the controls are still present and refuse to
    # certify if they are not.
    ctrl_lost = sorted({r["rank"] for r in have if r["config"] in invalid and r["rank"] <= 1})
    verdicts = {}

    print("\n--- admissibility (on VALID runs only) ---")
    for m in JUDGED:
        vals, ranks = [], []
        for r in valid:
            v = scores[r["config"]].get(m)
            if v is None:
                continue
            vals.append(float(v)); ranks.append(r["rank"])
        if len(vals) < 4:
            verdicts[m] = (False, "too few values")
            continue
        # 1. ORDERS -- Spearman-style concordance between metric and eye rank
        import itertools
        conc = dis = 0
        for (a, ra), (b, rb) in itertools.combinations(zip(vals, ranks), 2):
            if ra == rb:
                continue
            if (a - b) * (ra - rb) > 0:
                conc += 1
            elif (a - b) * (ra - rb) < 0:
                dis += 1
        tau = (conc - dis) / max(1, conc + dis)
        # 3. NOT FOOLED -- does any low-rank run outscore the highest-rank run?
        top = max(ranks)
        top_vals = [v for v, rk in zip(vals, ranks) if rk == top]
        low_vals = [v for v, rk in zip(vals, ranks) if rk < top]
        fooled = bool(top_vals and low_vals and max(low_vals) > max(top_vals))
        sep_ok, sep_why = _separates(vals, ranks)                  # 2. SEPARATES (was never run)
        ok = tau >= 0.6 and not fooled and sep_ok
        verdicts[m] = (ok, f"tau={tau:+.2f}" + (" FOOLED: a lower-ranked run outscores the top"
                                                if fooled else "")
                       + ("" if sep_ok else f" NOT SEPARATED: {sep_why}"))
        print(f"  [{'ADMIT ' if ok else 'REJECT'}] {m:28} {verdicts[m][1]}")

    admitted = [m for m, (ok, _) in verdicts.items() if ok]
    print(f"\n  admissible metrics: {admitted or 'NONE'}")
    # A metric bank that orders bud < spike < bigger-spike has NOT been shown to resist a blob.
    # Passing on a set whose blob and sphere controls were thrown out for invalidity is the gate
    # certifying itself on the easy half of its own test.
    certified = bool(admitted) and not ctrl_lost
    if admitted and ctrl_lost:
        print(f"\n  BUT the rank-{ctrl_lost} control(s) were refused as invalid, so this set "
              f"contains no blob/sphere case. The metrics below survived only the "
              f"bud-vs-spike half of the test.")
    passed = certified
    why = ("the bank has at least one metric that reproduces our judgement, controls present"
           if certified else
           "NOT CERTIFIED: the low-rank control(s) are invalid runs -- re-run them below "
           "saturation before trusting any admission" if admitted else
           "NO metric reproduces our judgement; do not score the campaign")
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'} -- {why}")
    print("=" * 96)

    json.dump({"labelled": LABELLED, "labelled_frames": LABELLED_FRAMES, "scores": scores,
               "invalid": invalid, "provenance_drift": drift, "controls_lost": ctrl_lost,
               "verdicts": {m: {"admit": ok, "why": w} for m, (ok, w) in verdicts.items()},
               "admitted": admitted if certified else [], "provisional": admitted,
               "passed": bool(passed), "why": why},
              open(os.path.join(HERE, "_metrology", "instrument_gate.json"), "w"), indent=1)
    return 0 if passed else 1


def submit(frames=900, parallel=8):
    import cluster
    names = [r["config"] for r in LABELLED]
    print(f"[gate] submitting {len(names)} labelled configs, {parallel}-way on the L4 partition")
    cluster.run_batch(names, frames=frames, campaign="instrument_gate", parallel=parallel)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "_metrology"), exist_ok=True)
    if a.submit:
        submit(frames=a.frames, parallel=a.parallel)
    sys.exit(score() if (a.score or not a.submit) else 0)

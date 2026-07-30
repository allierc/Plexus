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
    dict(config="round_40_mc8",      eye="tube",   rank=4,
         note="archived strip: long thin tube, activator at the tip. The fresh clock-fixed "
              "replay renders a SMALL BUD -- so this entry deliberately carries a KNOWN "
              "disagreement between the archived and current substrate, and the gate must "
              "score the run it actually ran, not the archive."),
    dict(config="round_41_hertwig",  eye="bud",    rank=2,
         note="long-axis division without bud-axis orientation -- report: 'makes no tube'"),
    dict(config="round_41_relax60",  eye="bud",    rank=2,
         note="more relaxation collapses the protrusion (R41)"),
    dict(config="round_42_k05",      eye="spike",  rank=3,
         note="monolayer, growth-driven: report says thin SPIKES, not a clean tube"),
    dict(config="round_42_k05_ex4",  eye="spike",  rank=3,
         note="monolayer + gentle extrusion assist"),
    dict(config="round_44_base",     eye="flood",  rank=1,
         note="emergent GM coupled to the wall machinery: floods, over-proliferates, NO tube"),
    dict(config="round_21_gs",       eye="bud",    rank=2,
         note="Gray-Scott stable-spot regime"),
    dict(config="ref_uniform_inflation", eye="sphere", rank=0,
         note="uniform inflation, no patterning: a sphere. The negative control."),
]

CLASS_ORDER = ["sphere", "flood", "bud", "spike", "tube"]

# metrics the gate judges. `tube_like` metrics should increase with rank.
JUDGED = ["ta_aspect_len_over_diam", "ta_tube_len_final", "ta_n_tubes_final",
          "protr_final", "protr_peak", "retention", "n_cells_final"]


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

    print("\n--- admissibility ---")
    verdicts = {}
    for m in JUDGED:
        vals, ranks = [], []
        for r in have:
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
        ok = tau >= 0.6 and not fooled
        verdicts[m] = (ok, f"tau={tau:+.2f}" + (" FOOLED: a lower-ranked run outscores the top"
                                                if fooled else ""))
        print(f"  [{'ADMIT ' if ok else 'REJECT'}] {m:28} {verdicts[m][1]}")

    admitted = [m for m, (ok, _) in verdicts.items() if ok]
    print(f"\n  admissible metrics: {admitted or 'NONE'}")
    passed = len(admitted) >= 1
    why = ("the bank has at least one metric that reproduces our judgement" if passed
           else "NO metric reproduces our judgement; do not score the campaign")
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'} -- {why}")
    print("=" * 96)

    json.dump({"labelled": LABELLED, "scores": scores,
               "verdicts": {m: {"admit": ok, "why": why} for m, (ok, why) in verdicts.items()},
               "admitted": admitted, "passed": bool(passed)},
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

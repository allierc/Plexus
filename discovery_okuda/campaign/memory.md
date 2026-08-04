<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. A STATE DOCUMENT, NOT A LOG.
     A line earns its place only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

The campaign is characterising Okuda-style tissue morphogenesis in Plexus and has,
after round 1, only re-measured 12 archived compositions under the current instruments —
no mechanism hypothesis has yet been posed or tested. The one hard finding is apparatus,
not biology: mechanical relaxation (P5b) fails to keep up with proliferation, so every
growing run under-relaxes and its late-frame shape metrics are suspect. Work is blocked
on that defect — the next real result needs a relaxation fix before any growth lever can
be interpreted.

## What is ESTABLISHED

- "Mechanical relaxation cannot keep up with proliferation (P5b breaks)." — SUPPORTED by
  wk_pressure_pos/wk_curvature_pos/wk_apical_area/cellfix_B_new/okuda_route, force ratio
  2.10–2.75 (frozen recons ~1.00), round 1. Falsifiable by: a growing run holding ratio
  <1.5 to the last frame, or an under-relaxed protrusion reproduced after ↑relaxation.
- "A control replay with prediction `unstated` establishes nothing." — SUPPORTED by all
  12 round-1 runs (NOT CHECKABLE, all inconclusive), round 1. Do not re-propose bare
  control replays; a prediction must name `<metric> <op> <value>`.
- "Frozen 2000-cell ball reads sphere, protr_peak 1.006; round40_mc8 reads tube 1.279
  (mech_p_ratio 2.4, FORCED)." — SUPPORTED by RECON replays, round 1. Baseline anchors.

## What is OPEN

- Whether any lever (pressure/curvature/apical/tension) produces a GROWN morphology
  distinct from sphere — untestable so far because every driven run breaks P5b, so its
  late shape cannot be trusted. Fix relaxation first, then re-measure.
- okuda_route also broke P4 (growth dilutes/feeds morphogen) and P7 (sheet stretches);
  unresolved — cannot separate real biology from under-relaxation until P5b is fixed.

## Known traps

- Prediction `unstated` on protr_peak → NOT CHECKABLE. Guard: every proposal states a
  numeric metric clause. (all 12 round-1 runs)
- Trusting protr_peak/shape_idx once cell count exceeds ~5× seed — under-relaxed artefact.
  Guard: check force ratio <1.5 before believing any late-frame shape. (5 growing runs)

## Frontier and parent

Breed from okuda_route (2000→3975 cells, growth+chemistry+division together) — the only
composition exercising the full stack — but only after a relaxation-increased variant
holds P5b. Comp hash: RECON okuda_route (config r001n_11_okuda_route.yaml).

## Stability envelope

Force ratio force_late/force_early must stay <~1.5 to be integrable; measured 1.00 at
frozen 2000 cells, 1.32–1.35 at 150→~1200 undriven, 2.1–2.75 at 150→~1300 driven
(round 1). Growth from a 150-seed to >~1000 cells exceeds the relaxation budget.

## Track A — the map

Necessary/sufficient/inert operators UNKNOWN — no mechanism run yet. Observed only:
frozen chemistry-on ball stays a sphere (Turing patterning, no shape change);
morphogen-gated growth off (a_sw above reachable activator) gives uniform growth.
Every cell of the operator table is blank.

## Track B — the figure

0 of 4 Okuda morphologies attempted. round40_mc8 yields a FORCED tube (mech_p_ratio 2.4),
which is an artefact tube, not a grown one — does not count as an attempt.

## Next action

Propose the first mechanism batch with numeric predictions, and include at least one
relaxation-increased variant of a growing composition to test whether P5b can be held
<1.5. Changes when a growing run holds the force ratio: then growth levers become
interpretable and Track A can begin.

HEADLINE: Round 1 all controls, none checkable; relaxation breaks (force ×2.1–2.75) in every growing run

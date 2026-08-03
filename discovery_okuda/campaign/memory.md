<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. A STATE DOCUMENT, NOT A LOG. Rewritten IN PLACE.
     A line earns its place only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT.
     History lives in analysis.md and hypotheses.jsonl (append-only, not yours to touch). -->

# Campaign memory

## Abstract

The goal is to reproduce Okuda's tissue morphologies (bud/tube/branch/invagination) from
operator compositions in the Plexus engine. Round 1 spent all 7 runs on bare replay/re-measure
controls that predicted `unstated` — none was checkable, so no mechanism has yet been tested.
Blocked on: we have never run a composition that couples chemistry to shape; the next round must
launch one with a real checkable clause.

## What is ESTABLISHED

- "Chemistry-only cfl_* compositions stay a healthy patterning sphere" — SUPPORTED across 7 runs
  (c000p01→c001p30 × d000p16→d002p00), morphology=sphere, protr_peak=1.006, ta_tube_*/mech_p_tube=0,
  round 1. Falsifiable by: any cfl_* run reaching protr_peak > 1.02 or morphology != sphere.
- "cfl_* runs are numerically clean" — SUPPORTED round 1: P4/P5/P8/P9/P11/P12 pass, genus 0,
  activator finite & non-negative, no self-intersection. Falsifiable by: a broken premise on cfl_*.

## What is OPEN

- Whether ANY composition here can leave the protr_peak≈1.006 sphere floor — untested; no
  growth/morphogen/division operator has been run this campaign.
- The whole map: which operators are necessary/sufficient for a bud or tube. Not one mechanism run.

## Known traps

- Predicting `unstated` (or any non-clause) → NOT CHECKABLE, scores nothing (all 7 runs, r1).
  Guard: every prediction is `<metric> <op> <value>` naming a known metric.
- "re-measure … under the current instruments" framing → watcher_blocks (3/7, r1). Drop the phrase.
- Replaying pre-pipeline archives (wk_curvature_*) → `{}` / no diag.json (5 runs, r1). Re-run from
  config or only replay compositions that emit today's diag.

## Frontier and parent

Nothing bred yet — round 1 produced no checkable mechanism. Start fresh: a minimal composition
that ADDS a chemistry→shape coupling operator (morphogen_growth_3d + divide_3d, or a growth/
curvature driver) onto the known-clean cfl_* sphere. Comp hash: none yet.

## Stability envelope

Measured on the passive sphere (all cfl_*, r1): 2000 cells, mech_force_mean 2.438,
mech_tension_mean 3.724, shape_idx floor 3.545 (min seen 3.663), reduced_volume 0.988, genus 0.
No growth/division stressed yet, so no envelope for those operators exists.

## Track A — the map

Operators exercised: reaction-diffusion chemistry only (necessary for Turing bands, INERT for
shape — produces no deformation alone). Growth, division, morphogen-growth, curvature drivers:
BLANK, never run. Coverage ≈ 0 of the morphogenetic map.

## Track B — the figure

0 of 4 Okuda morphologies attempted. All runs so far are the undeformed sphere control, not an
attempt at bud/tube/branch/invagination.

## Next action

Launch ONE mechanism composition adding a growth+morphogen (or curvature) coupling to a cfl_*
sphere, with a checkable prediction such as `protr_peak > 1.05`. Changes once we have a single
run that moves protr_peak off 1.006 — then breed from it.

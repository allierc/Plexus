# Campaign memory

## Abstract

This campaign builds the causal lever-map of the Okuda mechanism space — for each operator,
what it does alone and in combination — and answers Okuda's morphologies (0 of 4 attempted) as
queries against that map. Round 1 re-measured 12 legacy configs under the current instruments:
every valid parent is a sphere (protr_peak ≤ 1.11, no tube), and the round bought no mechanism
knowledge because all 12 predictions were `unstated`. Blocked on two apparatus facts that must
be respected before the first real edit: predictions need a checkable metric clause, and
growth+division seeded at ~150 cells saturates the 1766-cell buffer and voids the run.

## What is ESTABLISHED
- "All 12 legacy parents are spheres; none makes a protrusion or tube" — SUPPORTED by round-1
  replays, protr_peak ≤ 1.11 AND ta_n_tubes_final=0 AND mech_p_ratio=0 across all. Falsifiable
  by: any parent with protr_peak > 1.2 or ta_n_tubes_final ≥ 1.
- "Fixed-ball cfl_*/coral_fixed_ball (n_cells=2000, no growth) only pattern activator into
  Turing spots; shape unchanged" — SUPPORTED by cfl_c000p080/cfl_c001p300/coral_fixed_ball,
  morphology=sphere, protr_peak=1.006, round 1.
- "cellfix_B_new grows ×52 volume to 4151 cells WITHOUT buffer saturation, stays sphere" —
  SUPPORTED round 1, buf_full=false, protr_peak=1.073.

## What is OPEN
- Whether any operator or combination drives a protrusion/tube — UNTESTED; round 1 ran 0
  mechanism edits (12 controls only).
- The 12 round-1 predictions were `unstated`, so nothing was checked. This is a
  prediction-format defect, not a biology result — re-issue as checkable clauses, not re-runs.

## Known traps
- Prediction = `unstated` (no `<metric> <op> <value>` clause naming an admitted metric) →
  NOT CHECKABLE, run wasted. Round 1, all 12.
- Growth+division seeded at ~150 cells saturates the array at n_cells=1766 (buf_full,
  div_blocked=40, frame 304/401) → P13 fail, valid_evidence=false → NOT EVIDENCE. Killed
  wk_apical_area/wk_curvature/wk_null/wk_pressure/wk_tension, round 1. Guard: size buffer for
  the final cell count.
- mini_coral_nodilute, refute_coral_nocons emit no diag.json (empty {}). Do not re-run.

## Frontier and parent
Breed the first mechanism edits from cellfix_B_new — the only growth+division parent that did
NOT saturate the buffer (4151 cells, buf_full=false) — adding a shape-driving operator. Fixed
balls are inert; wk_* saturate. (No comp hash recorded in round 1.)

## Stability envelope
- Cell buffer caps growth+division at n_cells=1766 when seeded ~150 (measured, wk_*, round 1);
  stay below it or resize the buffer to the final count. cellfix_B_new held 4151 unsaturated.
- Fixed-ball runs stable at 2000 cells indefinitely.
- Shape index floor 3.545 — any value below is a broken measurement, not a finding.
- Reaction must give per_frame ≈ 1 (p1_ph_coral ran dt=0.02/rate=50 → per_frame 1, fine).

## Track A — the map
0 mechanism edits run. Operators seen in parents: morphogen_growth_3d + divide_3d
(cellfix_B_new, wk_*); activator reaction-diffusion only (cfl_*/coral fixed-ball). None
produces a protrusion or tube. Every causal cell is blank.

## Track B — the figure
0 of 4 Okuda morphologies attempted. All parents are spheres — none is a starting point for a
morphology, so the figure needs a shape-driving edit first.

## Next action
Propose the FIRST mechanism edit(s), each carrying a checkable protr_peak or ta_n_tubes_final
clause, bred from cellfix_B_new (non-saturating), adding a shape-driving operator. Changes once
one edit lifts protr_peak > 1.2 — then pivot from breaking the sphere to characterizing it.

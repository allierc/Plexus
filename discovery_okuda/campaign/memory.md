<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. This is the only record the loop consults --
     analysis.md is the append-only human record and no agent reads it.

     Everything here therefore costs context on every single call, which is the discipline: a line
     earns its place only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT.

     A STATE DOCUMENT, NOT A LOG. Rewritten IN PLACE every round.
     If a line stops being true it is CORRECTED, not annotated. The history lives in analysis.md
     and hypotheses.jsonl, which are append-only and are not yours to touch.

     A line earns its place here only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

The campaign is building the causal lever-map of the Okuda mechanism space — what each operator
does alone and in combination — and after round 1 it has NO valid morphogenesis evidence: 0 of 4
Okuda morphologies attempted and every operator still uncharted. Round 1 spent all 12 slots on
controls with unstated predictions, so both failures were Proposer defects, not biology. The
campaign is blocked on emitting its FIRST real edit: one wk_ growth config whose cell-array reserve
is raised above the buffer cap so the scorecard is not voided, carrying one checkable clause.

## What is ESTABLISHED

- "CFL (reaction-diffusion) nodes are Turing chemistry on a RIGID ball with no growth/division
  operator; sphere is their expected NULL, not a finding." — SUPPORTED by all 5 cfl_* replays,
  protr_peak=1.006 & mech_p_ratio=0 & ta_n_tubes=0, round 1. Falsifiable by: any cfl run showing
  protr_peak > 1.02 or a non-sphere morphology.
- "wk_ growth operators are mechanically active but drive final cell count straight into the
  cell-array cap, saturating the buffer and voiding the scorecard." — SUPPORTED by
  wk_apical_area_pos_s0 & _neg_s2, n_cells_final=36749, div_blocked=865, buf_full=true,
  valid_evidence=false, round 1. Falsifiable by: a wk_ run with a raised reserve that ends
  below cap with valid_evidence=true.

## What is OPEN

- Does ANY wk_ growth config (curvature / tension / apical_area / pressure) produce a valid
  protrusion? NEVER cleanly measured — every wk_ slot either returned `{}` (no diag.json) or
  saturated the buffer (valid_evidence=false). The blocker is a raiseable pool line, not the
  operator.
- FORCED (mech_p_ratio ~3) vs GROWN (~1) protrusion — undecidable: mech_p_ratio=0 everywhere
  because no tube has ever formed.

## Known traps

- Controls carry zero info: `replay` / `re-measure … under current instruments` / naming a RECON_
  node as object-of-study returns bit-identical null numbers. Never propose one (all 12 slots, rd 1).
- A prediction that is "unstated", a trend-word, or on a REJECTED metric is NOT CHECKABLE = zero
  info; use ONE clause `<metric> <op> <value>` on an ADMITTED metric (all predictions, rd 1).
- wk_ growth WITHOUT a raised pool line → P2_BUFFER_SATURATED at n≈36749 voids the scorecard.
  Guard: set the reserve above the LIVE cap (re-verify it) and confirm the run ends below it.
- APPARATUS artefacts, cosmetic, never spend a slot: (1) trajectory-classifier ValueError on the
  'sphere' string → analysts read metrics.png instead, verdict unaffected; (2) shape_idx p95 tail
  trips the P7 solid→fluid flag on non-deforming spheres.

## Frontier and parent

No valid parent exists — nothing has produced admissible morphogenesis evidence. Breed the first
real edit from a wk_ growth base (start wk_pressure_pos) with a raised reserve, NOT from any cfl or
RECON_ node (those are characterised nulls). No comp hash yet earns a frontier.

## Stability envelope

Only measured numbers so far are apparatus, not biology: non-growing cfl runs settle at
n_cells_final=2000, wall ~205 s. wk_ growth runs saturate at n≈36749 (div_blocked=865, buf_full),
wall ~1600–1720 s. The reserve must be set above the live cap to keep valid_evidence=true.

## Track A — the map

Operator landscape essentially uncharted. cfl / reaction-diffusion = chemistry only, INERT on
shape (no morphogenesis). wk_ growth operators (curvature, tension, apical_area, pressure) are
mechanically active but UNCHARTED because they saturate or emit no diag. mech_p_ratio is unusable
(0 everywhere). Every combination cell is blank.

## Track B — the figure

0 of 4 Okuda morphologies attempted. None started.

## Next action

Emit ONE non-control wk_ growth edit — start `wk_pressure_pos` — with a pool line raising the
cell-array reserve above the live cap AND one checkable clause (`protr_peak > 1.10`). Changes only
if the pool line proves inexpressible, in which case emit `APPARATUS GAP: cannot raise growth
reserve` and STOP (calls the Diagnostician).

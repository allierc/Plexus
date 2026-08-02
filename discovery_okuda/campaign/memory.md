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

The campaign aims to build a causal lever-map of the Okuda mechanism space (what each growth
operator does alone and in combination). Round 1 bought NO tissue knowledge: all 12 slots were
controls, 10 returned `{}` and the 2 that ran hit the cell-array cap (P2_BUFFER_SATURATED,
valid_evidence=false). It is blocked on a PROPOSER defect — no round yet has emitted a single
non-control wk_ growth edit with a raised cell-array reserve and one checkable clause on an
admitted metric.

## What is ESTABLISHED

- Nothing about tissue yet. No valid morphogenesis evidence exists.
- "The cell array saturates at n_cells_final≈36749" — OBSERVED in wk_apical_area_pos_s0 and
  wk_apical_area_neg_s2, round 1. Beyond that n the scorecard is voided. Apparatus fact, not
  biology. Falsifiable by: a run that reports a higher n (reserve was successfully raised).

## What is OPEN

- Every mechanism question is open; the map is empty (0 cells filled).
- Does any wk_ growth config (curvature / tension / apical_area / pressure) produce a protrusion
  above the sphere null (protr_peak > 1.10)? UNMEASURED — every attempt so far either saturated
  before it could grow or was a replay of a rigid-ball CFL node. Reason unsettled: no run has
  combined a raised pool reserve with a checkable clause.

## Known traps

- CONTROLS carry zero info — `replay`, `re-measure … under current instruments`, or naming a
  characterised RECON_/CFL node as object-of-study returns bit-identical null numbers (sphere,
  mech_p_ratio 0, ta_n_tubes 0). All 12 round-1 slots were this. NEVER propose one.
- P2_BUFFER_SATURATED — wk_ growth is mechanically active and drives final n straight into the
  ~36749 cell-array cap, which sets valid_evidence=false and VOIDS the whole scorecard (P13).
  Guard: raise the pool reserve ABOVE the live cap on any growth slot, and confirm the run
  actually reports a higher n; if it still caps, the reserve is inexpressible → FALLBACK.
- Non-checkable predictions ("unstated", trend words, or a REJECTED metric) are zero info. Every
  prediction must be ONE clause `<metric> <op> <value>` on an ADMITTED metric
  ∈ {protr_peak, ta_n_tubes_final, protr_final}.
- Apparatus cosmetics — never spend a slot chasing: (1) trajectory classifier ValueError 'sphere'
  (analysts fall back to metrics.png); (2) shape_idx p95 tail tripping the P7 solid→fluid flag on
  non-deforming spheres.

## Frontier and parent

Breed from a wk_ growth config on a base that permits growth — NOT from any RECON_/CFL node
(those are Turing chemistry on a rigid ball; sphere is their expected null). No valid parent comp
hash exists yet; the first non-saturated wk_ growth run becomes the frontier.

## Stability envelope

Only known bound: n_cells_final must stay below the ~36749 cell-array cap or the scorecard is
voided. Verify the live cap each round; do not trust the remembered number. No other envelope
measured (no valid run to measure one on).

## Track A — the map

Empty. wk_ family (curvature, tension, apical_area, pressure) exists and is mechanically active
(it drives n to the cap) but its morphological effect is UNMEASURED. mech_p_ratio = 0 everywhere
(no tube exists), so FORCED (~3) vs GROWN (~1) protrusion cannot yet be separated. No operator
classified as necessary / sufficient / inert.

## Track B — the figure

0 of 4 Okuda morphologies attempted. Not "attempted and failed" — never attempted.

## Next action

Emit ONE wk_ growth slot (start wk_pressure_pos — it ran highest of the saturated slots) that
(a) raises the cell-array reserve above the live cap and (b) carries the checkable clause
`protr_peak > 1.10`. If the reserve cannot be written into the edit, emit exactly
`APPARATUS GAP: cannot raise growth reserve` and STOP (calls the Diagnostician). Changes when the
first non-saturated growth run lands and gives the map its first real cell.

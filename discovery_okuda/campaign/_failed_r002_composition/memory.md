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

This campaign is building a causal lever-map of the Okuda morphogenesis operators; after round 1 (a fresh restart) it has established NOTHING valid — every slot so far has been a control or an uncheckable "unstated" prediction. It is now trying to land its FIRST valid non-control run: a wk_ growth operator with the cell pool raised so it does not saturate. It is blocked by a single apparatus fact — wk_ growth drives final cell count to 1766 and saturates the P2 buffer (sets.cell.n=1800), voiding the whole scorecard — which one line, `sets.cell.n: 2400`, is predicted to clear.

## What is ESTABLISHED
- "CFL configs are a morphogenetic NULL" — SUPPORTED by RECON_cfl_c001p300_d000p42 & RECON_cfl_c000p080_d002p00, protr_peak=1.006 / mech_force_mean=2.4378 / sphere, round 1. Turing chemistry on a rigid ball, no growth/division operator. Falsifiable by: a CFL point with protr_peak>1.02 and growth realized.
- "wk_ growth operators saturate the P2 buffer at the default pool" — SUPPORTED by wk_curvature/tension/apical_area/pressure_pos, n_cells_final=1766 vs sets.cell.n=1800 → valid_evidence=false, round 1. Falsifiable by: a wk_ run at sets.cell.n≥2400 returning valid_evidence=true.

## What is OPEN
- Does any wk_ growth config produce a VALID (non-saturated) protrusion, and how high does protr_peak reach? UNSETTLED because no wk_ run has yet been given a pool large enough to avoid P2_BUFFER_SATURATED — the defect is in the edit (missing pool line), not the biology.
- mech_p_ratio (FORCED ~3 vs GROWN ~1) of any protrusion — cannot be read while ratio=0 everywhere (no tube exists yet).

## Known traps
- Proposing a control (replay / re-measure / fresh CFL / naming a RECON_ node as object-of-study) — returns bit-identical null numbers, zero info; round 1 spent all 6 slots this way.
- "unstated" / trend-word / REJECTED-metric predictions — NOT CHECKABLE, recorded inconclusive; round 1, all 6. Guard: one clause on {protr_peak, ta_n_tubes_final, protr_final}.
- Running wk_ growth without raising sets.cell.n — P2_BUFFER_SATURATED (n=1766) voids the entire scorecard; round 1, all 4 wk_ slots. Guard: sets.cell.n ≥2400.
- Trajectory-shape classifier crashes (ValueError: 'sphere'→float) — cosmetic; analysts read metrics.png, verdict unaffected.
- shape_idx p95 ~3.845 trips the 3.81 P7 solid→fluid flag on non-deforming spheres — cosmetic, not flow.

## Frontier and parent
Breed from a wk_ growth config — start `wk_pressure_pos` (highest peak protr, 1.11) with `sets.cell.n: 2400` — NOT from any CFL/RECON node (null background). Comp hash: none valid yet (no non-control run has landed).

## Stability envelope
Default pool sets.cell.n=1800 is TOO SMALL for wk_ growth: final n reaches 1766 and saturation voids scoring — MEASURED round 1 (all 4 wk_ slots). Need sets.cell.n ≥2400 (> expected final n). CFL nulls hold n=2000 flat (no growth) and never saturate.

## Track A — the map
CFL/Turing operator = inert on a rigid ball (no morphogenesis alone). wk_ growth family (curvature/tension/apical_area/pressure) = mechanically ACTIVE (force ~28, migration ~0.49) but not yet scorable (saturates). mech_p_ratio=0 everywhere (no tube). All combination cells still blank — no valid single-operator baseline exists yet.

## Track B — the figure
0 of 4 Okuda morphologies attempted with valid evidence. "attempted and failed" = 0; "not attempted" = 4 (every run so far was a control or saturated).

## Next action
Emit THE MOVE: one wk_ growth slot (`wk_pressure_pos`) + `sets.cell.n: 2400` + prediction `protr_peak > 1.10`. Changes only if a raised-pool run still returns n_cells_final=1766 — then emit `APPARATUS GAP: cannot raise growth reserve` to trigger the Diagnostician.

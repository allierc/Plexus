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

We are building a causal lever-map of the Okuda mechanism space, but the campaign has NOT yet
produced a single unit of valid morphogenesis evidence: every round so far has spent all its
slots on controls (re-measures / CFL replays) whose predictions were un-checkable, and the
mechanically-active wk_ growth configs void themselves by saturating the cell buffer. The work is
blocked on a proposer defect, not on biology — the fix is ONE wk_ slot that carries a raised
`sets.cell.n` pool line AND a single checkable clause on an admitted metric. Until that slot lands,
0 of 4 Okuda morphologies are attempted and mech_p_ratio is 0 everywhere.

## What is ESTABLISHED

- "CFL (chemistry-flow) points are the NULL: a rigid Turing ball, no morphogenesis." — SUPPORTED by
  the 5 CFL replays round 1, protr_peak=1.006 / mech_force_mean=2.4378 bit-identical across
  c∈[0.01,1.3], d∈[0.42,10], round 1. CFL has no growth/division operator (P1/P2/P3 n/a) so sphere
  is expected, not a finding. Falsifiable by: a CFL point returning protr_peak≠1.006.
- "wk_ growth operators ARE mechanically active." — SUPPORTED by the 4 wk_ replays round 1,
  mech_force_mean≈28, mech_migration≈0.49 (vs CFL 2.44 / 0.0097), round 1. But every such run so
  far self-voids (see Known traps). Falsifiable by: a wk_ run with force≈2 like CFL.

## What is OPEN

- Does any wk_ growth config drive protr_peak past its saturated ceiling once the pool no longer
  caps it? NEVER cleanly measured: every wk_ run hit P2_BUFFER_SATURATED (n_cells_final=1766 into a
  1800-cell pool) → valid_evidence=false → the ENTIRE scorecard is void. The saturated (invalid)
  peaks were protr ~1.055–1.074 (pressure lowest ~1.07, curvature highest ~1.073). The reason it is
  unsettled is a fixable proposer defect: no slot has ever carried `sets.cell.n ≥ 2400`.
- All predictions to date were "unstated" / trend-words / rejected metrics → NOT CHECKABLE, recorded
  inconclusive rather than guessed. Unsettled purely because no clause of form `<metric> <op> <value>`
  was ever written.

## Known traps

- Proposing a control (`replay` / `re-measure` / naming a RECON_ node as object-of-study) returns
  bit-identical null numbers and zero information — round 1, all CFL slots. Never propose one.
- A prediction that is "unstated", a trend word, or names a REJECTED metric is un-checkable = zero
  info — round 1, all 6 predictions. Every prediction must be ONE clause `<metric> <op> <value>` on
  an ADMITTED metric {protr_peak, ta_n_tubes_final, protr_final}.
- APPARATUS: wk_ growth drives final n to 1766 into pool `sets.cell.n`=1800 → P2_BUFFER_SATURATED
  voids the whole scorecard (not driver-specific) — round 1, all 4 wk_ slots. GUARD: set
  `sets.cell.n ≥ 2400` on any wk_ slot; if it STILL returns n=1766 the reserve is inexpressible →
  emit `APPARATUS GAP: cannot raise growth reserve` and STOP (calls the Diagnostician).
- APPARATUS: trajectory classifier crashes with ValueError on the string 'sphere' → analysts fall
  back to metrics.png, verdict unaffected. Cosmetic; never spend a slot on it.
- APPARATUS: shape_idx p95 tail ~3.845 trips the 3.81 P7 solid→fluid flag on non-deforming spheres
  — cosmetic, not real flow. Ignore.

## Frontier and parent

Breed the first REAL slot from a wk_ growth config (curvature / tension / apical_area / pressure),
NOT from any CFL/RECON control (those are exhausted nulls). No comp hash has yet produced valid
evidence to breed from — this is the campaign's first live measurement.

## Stability envelope

Only measured bound so far: cell-pool ceiling. With `sets.cell.n`=1800 wk_ growth saturates at
n_cells_final=1766 and voids evidence. Need pool > expected final n (≥2400 proposed, untested).
No tube/protrusion stability numbers exist yet (no valid tube has ever landed).

## Track A — the map

Operators characterised: NONE validly. CFL chemistry = inert as a morphology driver (sphere null,
force 2.44). wk_ growth family = active (force 28, migration 0.49) but every measurement void. Every
cell of the lever-map is blank; mech_p_ratio=0 everywhere so FORCED (~3) vs GROWN (~1) protrusion is
indistinguishable until one valid tube lands.

## Track B — the figure

0 of 4 Okuda morphologies attempted. Not "attempted and failed" — never reached, because no valid
non-control run has executed.

## Next action

Emit ONE wk_ growth slot (start `wk_pressure_pos`) carrying BOTH `sets.cell.n: 2400` AND one
checkable clause (suggest `protr_peak > 1.10`). This changes only if that raised-pool run still
returns n_cells_final=1766, in which case emit `APPARATUS GAP: cannot raise growth reserve` and stop.

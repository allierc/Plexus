# User directives (read + acknowledge each batch; apply going forward)

## ⚠️ ADVANCE TO PHASE 3 NOW (2026-07-06) — MOR is COMPLETE; enter BUD this batch.
MOR (body-scale morphogenesis) has run **~16 batches (b69–b84), far past the 10-batch stage cap** — its
budget is spent and the automatic cap directive is not firing on this running process. **THIS BATCH you MUST:**
adopt MOR's best clean (collapsed=0, escape=0) point as MOR's operating spec, log any open MOR blocker as
`[open]`, **write `BUD` to `current_stage.txt`, and design the first Phase-3 BUD batch.** Do NOT run another
MOR batch. Ladder: …→MOR→**BUD**→BRN→ORG (ORG = terminus).

The first BUD batch = an ISOLATED localized-growth mechanism-validation: a zero-growth no-op control + a small
sweep of localized / pattern-gated `cell_grow` (`mode=anisotropic|tip`, `prestretch`, pattern-gated growth).
**Decide on the NEW organogenesis-geometry family** (`scorecard.json["organo"]` / `org_*` in metrics.json):
`n_buds`, `bud_score`, `bud_len_bodyR`, `bud_neck_ratio`, `bud_persistence`, and `growth_bud_overlap`
(causality) — NOT the movie. Hard-failures (collapsed/escape/nn_min/accel) and the INHERIT-CAPABILITIES rule
still apply: a bud must form WITHOUT rupturing the blastula or scrambling the established pattern.

## Earlier directives (still apply):

1. **Cell movement baseline `move_speed` 0.12**; you may go up to ~0.24 when a stage needs faster flow/migration.
2. **Growth is `cell_grow`, NOT `cell_divide`.** `cell_grow` drives tissue volume + protrusions; `cell_divide`
   only REPOPULATES grown volume (division mixes and destroys pattern above ~1.5×). There is NO fixed division
   multiplier — grow via `cell_grow`, bounded only by what the deforming domain physically holds at `repel.r0`.
3. **~12000 frames** per run so slow dynamics develop; keep each job within the L4 wall (raise `stride` if render grows).

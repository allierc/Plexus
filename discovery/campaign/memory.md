<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. A STATE DOCUMENT, NOT A LOG: rewritten IN
     PLACE every round; a line that stops being true is CORRECTED, not annotated. History lives
     in analysis.md and hypotheses.jsonl (append-only, not ours to touch). A line earns its place
     only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

CFL alone is an inert valid sphere (n=2000, mech_force_mean 2.4378, across c∈[0.01,1.3] d∈[0.42,10]);
the four wk_ growth drivers (curvature/tension/apical_area/pressure) are mechanically ACTIVE (~10×
force, ~50× migration, protr_peak up to 1.11) but saturate the cell pool at n=1766 (sets.cell.n=1800)
and void under P2_BUFFER_SATURATED, so their morphology is still UNMEASURED after 21 rounds. All ~120
slots have gone to controls or "unstated" predictions; the productive move (a wk_ config + raised pool +
checkable clause) has been ordered for ~6 straight rounds and NEVER emitted — the proposer emits
controls instead — so escalating the order is a spent lever and the leading hypothesis is that
`sets.cell.n` is UN-EMITTABLE in the edit vocabulary. The loop can only advance by emitting
`APPARATUS GAP: cannot raise growth reserve` (default now) to wake the Diagnostician — never called in
21 rounds — unless a slot can literally write sets.cell.n≥2400 + a one-clause admitted-metric prediction.

## What is ESTABLISHED
- "CFL alone is morphologically inert (a sphere on shape)" — SUPPORTED by the cfl_ replays,
  protr_peak=1.006, ta_n_tubes_final=0, mech_p_ratio=0, rounds 1–20, across c∈[0.01,4.0] d∈[0.42,10]
  (even the round-15 extreme c=4.0 stayed a sphere on shape). Falsifiable by: any CFL-only run with
  protr_peak>1.02 or ta_n_tubes_final≥1.
- "CFL c and d do not shift the base mechanics" — SUPPORTED, rounds 1–20: every in-range cfl_ point
  returned bit-identical mechanics (mech_force_mean 2.4378, mech_tension_mean 3.7236, mech_migration
  0.00967), incl. round-6 novel cfl_c000p020_d005p000. Falsifiable by: any cfl c,d that moves
  mech_force_mean off 2.4378. (Caveat: c=4.0 was NOT re-checked on mechanics; only its shape held.)

## What is OPEN
- Whether `sets.cell.n` is settable in the edit vocabulary — the PRIMARY blocker, still NEVER cleanly
  tested after 21 rounds (no slot has run a wk_ config with the pool raised and read the result). It has
  been ordered for ~6 rounds and never emitted → the pool line is most likely UN-EMITTABLE, which is
  itself the finding. Resolve by emitting exactly `APPARATUS GAP: cannot raise growth reserve`
  (Diagnostician trigger) as the DEFAULT; take the emit-and-read path only if a slot can literally write
  `sets.cell.n: 2400` (then read n_cells_final; still 1766 ⇒ inexpressible). NEVER a control.
- Valid morphology of the wk_ growth family is still UNMEASURED. Real activity exists (force ~28,
  migration ~0.49, protr_peak: pressure 1.11 > curvature 1.085 > apical/tension ~1.07) but every run
  through round 20 returned valid_evidence=false (P2_BUFFER_SATURATED, n=1766). Unblocking depends
  entirely on the pool-line question above.
- The round-15 P4 break is MISATTRIBUTED in the old record. P4 "growing a cell dilutes what is inside
  it" broke on `cfl_c004p000` — an EXTREME high-c CFL config OUT of the mapped range, with otherwise
  inert-sphere metrics (protr 1.006, NO growth). It was NOT a wk_ growth run, so it does NOT show
  "growth adds volume without diluting." LOWEST priority; ignore until the pool question is settled.

## Known traps
- Re-running a deterministic control → bit-identical numbers, zero info (rounds 1–20, ~110 slots — the
  campaign's DOMINANT failure, and the instruction alone has not stopped it). DISGUISE: a "fresh" CFL
  c,d point not in the recon set is still a control (returns 2.4378); re-seeding a wk_ recon is a
  replay; naming a RECON_ node as the object of study IS a replay. Guard: spend the slot on the pool-
  raise (Option A) or the APPARATUS GAP string (Option B), never a re-measure.
- Repeating the "do THE MOVE" instruction louder has NOT changed proposer behaviour (rounds ~16–21, 6
  straight orders, 0 emissions). Guard: the exhortation is a SPENT lever — un-emittability is now the
  first-class outcome. The reflex to emit a replay/re-measure/"fresh" CFL point IS the signal the pool
  line can't be written → emit `APPARATUS GAP: cannot raise growth reserve` instead, never a control.
- Growth edit at default pool → P2_BUFFER_SATURATED voids the whole scorecard (rounds 1–20, all four
  wk_ drivers pinned at n=1766). Guard: raise sets.cell.n above expected final n (≥2400).
- Predicting "unstated" / a bare trend word / a REJECTED metric → recorded NOT CHECKABLE, zero info
  (EVERY round 1–20). Guard: one clause `<metric> <op> <value>` on an ADMITTED metric (protr_peak,
  ta_n_tubes_final, protr_final).
- APPARATUS: trajectory-shape classifier crashes ValueError 'sphere'; analysts fall back to
  metrics.png, verdict unaffected — not a result (every run, rounds 1–18).
- APPARATUS: shape_idx p95 late tail (~3.845) trips the 3.81 P7 solid→fluid flag on non-deforming
  spheres — cosmetic, never overturns a sphere call; do not read it as flow.
- REJECTED metrics — never reason from: ta_aspect_len_over_diam (scored 9.30 on a bud),
  ta_tube_len_final, retention (anti-correlated with elongation), n_cells_final.

## Frontier and parent
- Next round breeds from the clean CFL sphere baseline — parent RECON_cfl_c001p300_d000p42 (n=2000,
  valid sphere) — ADDING ONE active wk_ operator with the pool raised. No growth-family lead is yet
  valid (the P4 break was on CFL, not growth). Once a wk_ growth run scores VALID it becomes the
  frontier the map breeds from. CFL is background, never the object of study.

## Stability envelope
- Pool cap: CFL-only holds n_cells_final at 2000 and scores valid; every wk_ growth driver saturates
  at 1766 (pool sets.cell.n=1800) and returns valid_evidence=false. Growth edits must raise
  sets.cell.n above expected final n (≥2400) or they are voided.
- CFL c∈[0.01,1.3], d∈[0.42,10] all integrate cleanly to a valid sphere (rounds 1–14); c=4.0 also
  spheres on shape (round 15) but broke the P4 premise check.

## Track A — the map
- CFL: NECESSARY as substrate, morphologically INERT alone; c,d params do not move the equilibrium.
  Mapped null (shape held even at c=4.0).
- wk_ growth (curvature/tension/apical_area/pressure): mechanically ACTIVE (force ~28, migration
  ~0.49); morphology still UNMAPPED (voided at default pool, never run with pool raised). pressure
  gives the highest protr_peak (1.11).
- All operator combinations: blank. Nothing composed yet.

## Track B — the figure
- 0 of 4 Okuda morphologies attempted. No valid non-sphere morphology produced yet. (Not attempted —
  distinct from attempted-and-failed.)

## Next action
- Emit exactly `APPARATUS GAP: cannot raise growth reserve` (Diagnostician trigger) as the DEFAULT —
  never a control. Only substitute OPTION A if a slot can literally carry the line `sets.cell.n: 2400` on
  a wk_ growth config on RECON_cfl_c001p300_d000p42 with a one-clause admitted-metric prediction (start
  wk_pressure_pos, `protr_peak > 1.10`). Changes when the Diagnostician settles whether the reserve is
  expressible, or a wk_ run finally scores valid (then read morphology, begin Track A/B).

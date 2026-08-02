<!-- THE CORE OF THE BUILT KNOWLEDGE. Read at the START of every round by the Proposer, rewritten
     at the END of every round by the Meta-review. A STATE DOCUMENT, NOT A LOG: rewritten IN
     PLACE every round; a line that stops being true is CORRECTED, not annotated. History lives
     in analysis.md and hypotheses.jsonl (append-only, not ours to touch). A line earns its place
     only if a LATER ROUND NEEDS IT AND COULD NOT RE-DERIVE IT. -->

# Campaign memory

## Abstract

After three rounds the campaign has run ONLY controls — zero mechanism edits — and owns exactly
two null facts: CFL alone is an inert equilibrium sphere, and every wk_ growth driver saturates
the reservoir (n_cells 1766, valid_evidence=false) so its morphology is unmeasured. It is now
overdue to issue its first mechanism edit — one active operator composed onto the CFL sphere with
the reservoir oversized above final n, carrying a clause prediction on an admitted metric. It is
blocked entirely by its own behaviour: the Proposer keeps re-running deterministic controls with
"unstated" predictions instead of committing that edit.

## What is ESTABLISHED
- "CFL alone is morphologically inert" — SUPPORTED by the cfl_ replays, protr_peak=1.006,
  ta_n_tubes_final=0, mech_p_ratio=0, rounds 1–3, across c∈[0.01,1.3] d∈[0.42,10]. Falsifiable
  by: any CFL-only run with protr_peak>1.02 or ta_n_tubes_final≥1.
- "CFL c and d do not shift the base equilibrium" — SUPPORTED, rounds 1–3: every cfl_ replay
  returned bit-identical mechanics (mech_force_mean 2.4378, mech_tension_mean 3.7236,
  mech_migration 0.00967). A number change here is not a physics change. Falsifiable by: any
  cfl c,d pair that moves mech_force_mean off 2.4378.

## What is OPEN
- Valid morphology of the wk_ growth family (curvature, tension, apical_area, pressure) is
  UNMEASURED. Rounds 1–3 showed ~10× force (28 vs 2.4), ~50× migration (0.49), protr_peak up to
  1.11 (pressure highest) — real activity — but EVERY run returned valid_evidence=false
  (P2_BUFFER_SATURATED, n_cells 1766), so whether any wk_ driver makes a valid tube/bud is
  unknown. Reason unsettled: reservoir too small; the fix is to re-run oversized, never done.
  The four drivers are indistinguishable at default size (same ~28 force / ~0.49 migration / 0
  tubes).

## Known traps
- Predicting "unstated" / a bare trend word / a REJECTED metric → recorded NOT CHECKABLE, zero
  info (rounds 1–3, all 18 controls). Guard: every prediction is one clause `<metric> <op>
  <value>` on an ADMITTED metric (protr_peak, ta_n_tubes_final, protr_final).
- Growth edit at default reservoir → P2_BUFFER_SATURATED voids the whole scorecard (rounds 1–3,
  all four wk_ drivers at n_cells 1766). Guard: size reservoir above expected final n.
- Re-running a deterministic control → bit-identical numbers, zero info (rounds 2–3 replayed
  round 1's cfl controls verbatim). Guard: run a control once; never replay a settled one.
- APPARATUS: trajectory-shape classifier crashes ValueError 'sphere'; analysts fall back to
  metrics.png, verdict unaffected — not a result (rounds 1–3, every run).
- APPARATUS: shape_idx p95 late tail (~3.845) trips the 3.81 P7 solid→fluid flag on non-deforming
  spheres — cosmetic, never overturns a sphere call; do not read it as flow (rounds 1–3).
- REJECTED metrics — never reason from: ta_aspect_len_over_diam (scored 9.30 on a bud),
  ta_tube_len_final, retention (anti-correlated with elongation), n_cells_final.

## Frontier and parent
- Rounds 1–3 bred nothing (controls only); no mechanism composition hash exists yet. Round 4
  breeds from the clean CFL sphere baseline — parent recon RECON_cfl_c001p300_d000p42 (n_cells
  2000, valid, sphere) — composing ONE active operator onto it, reservoir oversized. CFL is
  background, never the object of study.

## Stability envelope
- Reservoir cap: CFL-only holds n_cells_final at 2000 and scores valid; every wk_ growth driver
  hits 1766 and returns valid_evidence=false (rounds 1–3). Growth edits must size the reservoir
  above expected final n or they are voided.
- CFL c∈[0.01,1.3], d∈[0.42,10] all integrate cleanly to a valid sphere (rounds 1–3).

## Track A — the map
- CFL: NECESSARY as substrate, morphologically INERT alone; c,d params do not move the
  equilibrium. Mapped null.
- wk_ growth (curvature/tension/apical_area/pressure): mechanically ACTIVE (force ~28, migration
  ~0.49), morphology UNMAPPED — all voided at default reservoir. pressure gives the highest
  protr_peak (1.11), the strongest protrusion lead.
- All operator combinations: blank. Nothing has been composed yet.

## Track B — the figure
- 0 of 4 Okuda morphologies attempted. No valid non-sphere morphology produced yet. (Not
  attempted — distinct from attempted-and-failed.)

## Next action
- Round 4: issue the FIRST mechanism edits — compose one active operator (start wk_pressure_pos,
  the 1.11 lead, then curvature/tension/apical_area) onto the CFL sphere, reservoir oversized
  above expected final n, each carrying a clause prediction (e.g. protr_peak>1.10 confirmatory,
  ta_n_tubes_final≥1 adversarial). ≥4 mechanism slots; zero replays. Changes if wk_ still voids
  after oversizing → growth family unusable at this apparatus, pivot to a protrusion/migration
  operator.

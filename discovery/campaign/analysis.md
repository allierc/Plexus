# Analysis log

_APPEND ONLY. One entry per round._

## Round 1 proposal — 2026-07-30

**User input:** no Pending instructions as of 2026-07-30 — nothing to action.

**State going in:** R0 = 5-run vcap parameter sweep on base C5e315998af4/round_40_mc8 →
surprise 0.00, supervisor flagged 92/8 confirmatory drift. Every R0 slot FORCED
(mech_p_ratio 1.8–42.9, body drainage); NO growth-driven (p_ratio~1) tube seen anywhere.
protr_peak vs vcap non-monotone {2.19,4.03,1.73,2.24,3.22}. Solo lever-map still EMPTY
(all 8 operators "insufficient"). Map coverage 0%. Phenotypes {tube:2, spike:3}.

**Batch (6 slots, mode=explore, 3 confirmatory / 2 adversarial = 60/40, leaning
adversarial per supervisor's standing steer).** All edits are single-operator changes on
the round-33 recipe control; NO parameters touched.
- s0 CONTROL — round-33 recipe unchanged (predict protr_peak 2.0–3.5, p_ratio ~3 forced).
- s1 −extrude (conf) — extrude is the forcing op; knockout → protr_peak <1.5, p_ratio→~1.
  Highest-value lever entry + best shot at a GROWN mid-surface tube.
- s2 −morphogen_growth_3d (adv) — if protrusion is FORCED not grown, protr_peak stays ≈control
  (>=2.0); refuted if <1.5. Dissociation partner to s1.
- s3 −cell_geometry_3d (adv) — genuinely uncalled: inert bookkeeping vs load-bearing geometry
  for shape_energy_3d. Predict protr_peak <1.5/degenerate; refuted if >=2.0.
- s4 +divide_3d:hertwig (conf) — oriented long-axis division elongates → protr_peak >=3.0
  and/or ta_n_tubes_final up.
- s5 +vesicle_growth:uniform_ramp (conf) — uniform growth → p_ratio→~1 grown regime,
  protr_peak >=2.0; refuted if body merely inflates (protr_peak <1.5).

**Design rationale:** R0 exhausted the vcap knob and gave zero surprise; the map has no solo
effects at all, so R1 buys causal-map coverage via knockouts (each removal = one clean lever
reading) plus two additions that plausibly produce Okuda-style GROWN morphology rather than
forced-drainage spikes. The −extrude / −morphogen_growth_3d pair is the central falsifiable
test of the "forced not grown" verdict from R0.

## Round 2 proposal — 2026-07-30 (corrected numbering + framing)

**User input:** user_input.md has NO Pending instructions as of 2026-07-30 — nothing to action.

**Numbering/framing correction (per operator note in memory.md):** the earlier entry above
labels the vcap sweep "R0" and the knockout batch "R1"; the operator fixed the counter — the
vcap sweep is **round 1** and this knockout batch is **round 2**. The vcap sweep was a
legitimate Loop-II measurement, NOT a "forbidden"/violation round; that language has been
removed from proposal.json and must not be reintroduced.

**State going in:** round 1 = 5-run vcap sweep on base C5e315998af4/round_40_mc8, surprise 0.00,
92/8 confirmatory. Every slot FORCED (mech_p_ratio 1.8–42.9, body drainage); NO growth-driven
(p_ratio~1) regime anywhere; protr_peak vs vcap non-monotone {2.19,4.03,1.73,2.24,3.22}. Solo
lever-map still EMPTY (all 8 ops "insufficient"). Map coverage 0%. Phenotypes {tube:2, spike:3}.

**Batch (6 slots, mode=explore, 3 confirmatory / 2 adversarial = 60/40).** All single-operator
edits on the round-33 recipe control; NO parameters touched. Predictions tightened to ADMITTED
metrics only (protr_peak / ta_n_tubes_final / protr_final) so every clause is mechanically
checkable — p_ratio is diagnostic commentary, never the checkable clause.
- s0 CONTROL — round-33 recipe (predict protr_peak 1.7–4.0).
- s1 −extrude (conf) — forcing op; knockout → protr_peak <=1.5. Refuted if >=2.0.
- s2 −morphogen_growth_3d (conf) — forced-not-grown ⇒ protr_peak >=2.0. Refuted if <=1.5.
  Dissociation partner to s1 — the central test.
- s3 −cell_geometry_3d (adv) — inert bookkeeping vs load-bearing geometry; predict protr_peak
  >=2.0. Refuted if <=1.5 or ta_n_tubes_final==0.
- s4 +divide_3d:hertwig (conf) — long-axis division sustains a tube; ta_n_tubes_final >=1.
- s5 +vesicle_growth:uniform_ramp (adv) — hunt for the grown regime; protr_peak >=2.0.
  Refuted if <1.5 (mere inflation).

**When results land:** cross-check p_ratio + Q_drop + body-shrink to reject forced-drainage
"tubes" (R1 lessons: high protr_peak ≠ stable tube; watcher gate inert; aspect/tube_len/retention
LIE). If −extrude or +vesicle_growth yields p_ratio~1 with a persistent bulge → first GROWN
regime → robustness-test that composition across seeds next round.


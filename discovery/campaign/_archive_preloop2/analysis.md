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

## Round (this batch) proposal — 2026-08-01 — PIVOT to the unmapped Okuda RD family

**User input:** user_input.md has NO Pending instructions as of 2026-08-01 — nothing to action.

**State going in:** Supervisor's run-counter reset again (evidence header shows "round 0, 0 runs,
coverage 0%") but memory.md/instruction.md carry the real record through round 2. Established:
parent-1 round-33 recipe is a growth-FED FORCED extrusion — `−morphogen_growth_3d` collapsed
protr_peak 4.03→1.026 (sphere) and `−extrude` →1.385 (sphere); BOTH core drivers jointly
necessary (R2 surprise 0.33). Known-invalid edits NOT to repropose: +vesicle_growth:uniform_ramp
(buffer saturates, n_cells 15002), −cell_geometry_3d / −cell_adjacency (load-bearing plumbing,
crash/no diag.json), division-heavy adds (late force/tension blowup). No GROWN (p_ratio~1) regime
found on parent 1.

**Design decision:** parent 1 is mapped enough; pivot the budget to the UNMAPPED, higher-value
family — **parent 2, the reaction-diffusion "growth-driven monolayer (Okuda route)"**
(cell_adjacency+cell_diffuse+cell_geometry_3d+cell_react+divide_3d+morphogen_growth_3d+
reconnect_t1_3d+seed_mesh_3d+shape_energy_3d+shape_to_chem). It has NO baseline yet and is the
closest structure to Okuda's Turing-patterning-on-a-deforming-sheet — the real path to a GROWN
morphology and the (chi,gamma) phase diagram.

**Batch (4 slots, mode=explore, 2 confirmatory / 1 adversarial = 67/33; 2 in_paper /
1 excursion = 67/33).** Single-operator edits spanning the family's mechanism axes; NO parameters
touched. Predictions absolute (new family, no drifting control to diff against).
- s0 CONTROL — parent 2 unchanged, first baseline (predict protr_peak 1.0–4.0, wide per drift lesson).
- s1 +shape_energy_3d:default (conf / in_paper) — type system labels this the "growth-driven
  emergent (target mechanism)". Predict protr_peak >=2.0 and ta_n_tubes_final >=1. REFUTED if <2.0.
- s2 −reconnect_t1_3d (conf / in_paper) — DEFORMATION/topology axis; Okuda's fluidity (γ=100 tube,
  p.7). Jam hypothesis: protr_peak <=1.5. REFUTED if >1.5 (growth bulges a jammed sheet).
- s3 −morphogen_growth_3d (adv / excursion) — GROWTH axis; tests whether R2's morphogen-necessity
  GENERALISES to a family with redundant growth (divide_3d + RD coupling). Adversarial bet it is
  REDUNDANT here: protr_peak >=2.0. REFUTED if <2.0 → morphogen universally necessary.

**When results land:** this establishes parent 2's baseline + three lever readings. Watch for a
p_ratio~1 persistent bulge (first GROWN regime → robustness-test next round). Reject forced-drainage
"tubes" via body-shrink/Q_drop; ignore watcher CONTRADICTS and the rejected aspect/tube_len/retention
metrics. If s1 opens the target mechanism, next round maps its (chi,gamma)-analog routing.

## Round 2 proposal — 2026-08-01 — parent 2, close the reaction-diffusion (chi) axis

**User input:** user_input.md has NO Pending instructions as of 2026-08-01 — nothing to action.

**State going in:** Counter-reset artifact again (header "round 1, 0 runs, coverage 0%,
phenotypes {}"); real record = memory.md/instruction.md + the 2026-08-01 parent-2 pivot on disk.
That prior parent-2 batch spends its three edits on the target shape energy (+shape_energy_3d:default),
the γ/topology axis (−reconnect_t1_3d) and the growth axis (−morphogen_growth_3d) — but leaves the
reaction-diffusion **χ axis untested**. parent 2 has exactly four VALID single-op edits
(vesicle_growth:uniform_ramp saturates the buffer; −cell_geometry_3d/−cell_adjacency are load-bearing
plumbing that crash); the only one the pivot missed is −cell_diffuse.

**Design decision:** make −cell_diffuse the centerpiece and hold two anchors so this round differs
from the pivot by exactly one edit (γ-axis jam test → χ-axis diffusion knockout). Over the two
batches the union covers all four valid parent-2 edits. cell_diffuse is a physics DRIVER (the D in
Turing RD), not bookkeeping — the type system keeps the region valid ("growth-driven monolayer"),
so it should run, not crash.

**Batch (4 slots, mode=explore, 2 confirmatory / 1 adversarial = 67/33; 2 in_paper /
1 excursion = 67/33).** Single-op, no parameters.
- s0 CONTROL — parent 2 unchanged (predict protr_peak 1.0–4.0, wide per drift lesson).
- s2 −cell_diffuse (ADV / EXCURSION) — the χ axis. Knock out morphogen diffusion; bet growth+mechanics
  still bulge the sheet → protr_peak >=2.0. REFUTED if <2.0 → diffusion load-bearing, deformation is a
  genuine Turing pattern. Genuinely uncallable.
- s1 +shape_energy_3d:default (CONF / in_paper) — the "growth-driven emergent (target mechanism)";
  re-confirm the prize against a fresh baseline. protr_peak >=2.0 and ta_n_tubes_final >=1. REFUTED if <2.0.
- s3 −morphogen_growth_3d (CONF / in_paper) — does R2 growth-necessity GENERALISE to the RD family
  (divide_3d + RD could compensate)? Predict collapse → protr_peak <=1.5. REFUTED if >=2.0 →
  necessity is family-specific. NOTE: the pivot bet the OPPOSITE (>=2.0 redundant); if both run this is
  a natural robustness read on the most uncertain edit.

**When results land:** the χ vs γ contrast (−cell_diffuse vs the pivot's −reconnect_t1_3d) is the
first cut at Okuda's (chi,gamma) plane. Hunt a p_ratio~1 persistent bulge (first GROWN regime →
robustness-test next round). Reject forced-drainage "tubes" via body-shrink/Q_drop; ignore watcher
CONTRADICTS and rejected aspect/tube_len/retention.

## Round 4 proposal — 2026-08-01 — OPEN parent 3 (uniform-inflation growth family)

**User input:** user_input.md has NO Pending instructions as of 2026-08-01 — nothing to action.

**State going in:** Header shows "round 3, 0 runs, coverage 0%, phenotypes {}" and an EMPTY
solo-effects table — another counter-reset artifact; real record = memory.md/instruction.md + the
two 2026-08-01 parent-2 batches on disk. Those two batches between them proposed ALL FOUR valid
single-op edits on parent 2 (the reaction-diffusion Okuda route: +shape_energy_3d:default,
−reconnect_t1_3d, −morphogen_growth_3d, −cell_diffuse), so parent 2 is fully PROPOSED and, until its
results actually land, cannot be built on further. Parent 0 (mechanics-only minimal) and parent 3
(uniform inflation) remain completely UNMAPPED.

**Design decision:** open parent 3 = **uniform inflation** (divide_3d + reconnect_t1_3d + seed_mesh_3d
+ shape_energy_3d + vesicle_growth) — the pure MECHANICAL-growth route (growth+division, NO
reaction-diffusion). It is the natural COMPLEMENT to parent 2's RD route: contrasting the two answers
whether a grown Okuda morphology needs Turing reaction-diffusion (chi) or just mechanical growth. The
central falsifiable question for the family: can any single edit break its "no patterning" label into a
patterned protrusion? Chose parent 3 over parent 0 because its growth driver is already ACTIVE, so
each edit has a live morphology to modulate (parent 0 without growth is a static mesh — low info).
Buffer-saturation risk (growth+division stack) is noted but lower than parent 2's triple-growth stack;
two of three edits are removals and stay safe even if the control saturates. Avoided the known-invalid
+vesicle_growth:uniform_ramp (would double-stack growth → saturation).

**Batch (4 slots, mode=explore, 2 confirmatory / 1 adversarial = 67/33; 2 in_paper / 1 excursion =
67/33).** Single-op, no parameters. Absolute predictions (new family, no drifting control).
- s0 CONTROL — parent 3 unchanged, first baseline (predict protr_peak 1.0–1.8 and ta_n_tubes_final <=0,
  a smooth uniform ball).
- s1 =shape_energy_3d:monolayer (ADV / in_paper) — swap shape energy to the monolayer impl; bet it
  buckles the inflating shell into an undulation (Okuda Fig 7, chi=0.1/gamma=100), AGAINST the type
  system's "no patterning" label. protr_peak >=2.0 and ta_n_tubes_final >=1. REFUTED if <2.0.
- s2 −vesicle_growth (CONF / in_paper) — remove the growth driver; family's first solo effect. Predict
  collapse to a static mesh, protr_peak <=1.5. REFUTED if >1.5 (division alone drives shape change).
- s3 −reconnect_t1_3d (CONF / excursion) — remove T1 topology relief under isotropic growth (a regime
  Okuda never isolates). Bet it is inert plumbing here → protr_peak 1.0–1.8, near control. REFUTED if
  >=2.0 → suppressed T1 stores growth stress that buckles the shell, making reconnect_t1_3d a hidden
  anti-buckling regulator (Okuda's gamma / deformation-rate axis).

**When results land:** this establishes parent 3's baseline + three lever readings and lets us start
the parent-2-vs-parent-3 (RD-vs-mechanical growth) contrast. If s1 buckles (patterning from a monolayer
impl swap) that is the family's prize → robustness-test across seeds next round. Apply the standing
guards: reject forced-drainage "tubes" via body-shrink/Q_drop; check the cell-count budget for the
growth+division control; ignore watcher CONTRADICTS and the rejected aspect/tube_len/retention metrics.


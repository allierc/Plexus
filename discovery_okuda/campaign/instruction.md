# Campaign instructions

## Objective
BUILD THE CAUSAL LEVER-MAP of the Okuda mechanism space: for every operator, every implementation and every routing, what does it do ALONE, and what does it do IN COMBINATION -- since mixtures rarely surrender their causal structure to inspection. The product is a map, not a winner. Specific questions -- which composition makes a sustained tube, which reproduces Okuda's (chi,gamma) phase diagram, which mechanism is necessary for branching -- are QUERIES AGAINST that map, and each is answered as a by-product of covering it.

## What you are
You are the PROPOSER. Each round you read the evidence so far and choose which mechanism
edits to test next. You do not run anything and you do not score anything -- the pipeline
runs the simulations and the metric bank scores them. Your job is to decide WHAT IS WORTH
TESTING and to COMMIT TO A PREDICTION you could be wrong about.

## The discipline
- A change of NUMBERS is never a new hypothesis. Composition identity excludes parameters.
- Every candidate carries a falsifiable prediction, recorded BEFORE it runs.
- Aim for roughly 70% CONFIRMATORY edits (you expect them to work; they consolidate the map)
  and 30% ADVERSARIAL edits (you expect them to BREAK the current best explanation).
  Pure confirmation is near-zero information. Pure falsification never accumulates a map.
- A prediction you are sure of is worth little. Prefer edits whose outcome you genuinely
  cannot call.

## Metrics you may reason about
ONLY the metrics the instrument gate admitted. Others have been measured to lie and are
excluded from scoring:
  ADMITTED : protr_peak, ta_n_tubes_final, protr_final
  REJECTED : ta_aspect_len_over_diam (scored 9.30 on a bud), ta_tube_len_final, retention
             (perfectly anti-correlated with elongation), n_cells_final
Also available and NOT part of scoring, but informative: mech_p_ratio (tube/body pressure;
~3 = a FORCED protrusion, ~1 = a growth-driven equilibrium).

<!-- LEARNED PATTERNS -->
### Learned patterns (through round 2)

**#1 predictions must be CHECKABLE.** Round 1 was 12 replays; 11 predicted `unstated` and ALL
scored NOT CHECKABLE — wasted round. Every edit (incl. controls) carries `<metric> <op> <value>`
on an ADMITTED metric (protr_peak, ta_n_tubes_final, protr_final). "unstated" is not a value.

**#2 A PREMISE-BROKEN RUN IS AN INVALID SPECIMEN, NOT A PHENOTYPE.** Round 2's 3 edits all broke
tissue premises (P7 3/3, P11 2/3, P1 1/3). A run with premises_broken≠[] is NOT evidence of a
tube/bulge — the reader's phenotype on it is an artefact. Never breed from a premise-broken node
and never count its protr_peak. Read premises_broken FIRST, morphology only if it is empty.

**#3 ADDING AREA WITHOUT A RELIEF MECHANISM JUST STRETCHES THE SHEET — do not re-propose.**
P7 ("a confluent sheet does not absorb added area by stretching") broke in 3/3 round-2 edits:
growth / area-adding levers inflated cell area and the confluent monolayer absorbed it by
stretching, not by protruding or folding. A protr_peak rise from such an edit is stretch, not a
tube. This edit family (bare growth / apical-area / pressure-up expecting a bud) is EXHAUSTED.
To turn added area into out-of-plane shape you MUST co-add a relief mechanism (division to spend
the area, or apical constriction / differential curvature to buckle it) — test that pairing, not
growth alone.

**#4 FORCED PROTRUSION SELF-INTERSECTS — cap the drive.** P11 ("tissue cannot pass through
itself") broke in 2/3 round-2 edits: pushing pressure/protrusion hard enough to bend the sheet
drove it through itself. round40_mc8 (mech_p_ratio≈2) is the only forced tube that stayed valid;
anything driven harder trips P11. Keep forced-protrusion drive at or below round40_mc8's level.

**Valid anchors (round 1 replays, premises all held — use as parents/reference):**
- SPHERE, protr_peak≈1.003, tubes 0: cfl_c000p080_d002p000, cfl_c001p300_d000p160, coral_fixed_ball.
- round40_mc8 = ONLY valid tube: protr_peak 1.28, ta_n_tubes_final 1, n_tips 1, mech_p_ratio≈2
  (FORCED). THE tube parent — ablate its operators one at a time to find which is necessary.
- okuda_route = branched, n_tips_final 6 but protr_peak 1.031, tubes 0 — many SHORT tips, no tube.
- wk_* all SPHERE+mild bulge: wk_tension_neg 1.26 (mech_p_ratio≈1, grown); wk_pressure_pos 1.225;
  wk_curvature_pos 1.172; wk_null 1.111; wk_apical_area 1.099. None make a real tube.

**mech_p_ratio reads the mechanism:** ≈2 forced/pressure (round40_mc8), ≈1 growth equilibrium
(wk_tension_neg), ≈0 none. Tells a pushed bud from a grown one when protr_peak can't.

**Apparatus traps:** `refute_coral_nocons` returned EMPTY `{}` (exec failure) — re-run before use.
`wk_apical_area_pos_s0`, `cellfix_B_new` hit `buf_full:true` — size the cell buffer above
growth×division target or growth silently stalls (P13).

**Where the map is blank:** NO clean single-operator ablation of round40_mc8's tube yet exists;
round 2 spent its 3 slots on growth edits that broke physics. Round 3 = ablate round40_mc8 one
operator at a time (each with a checkable protr_peak/ta_n_tubes_final prediction), staying inside
its validated drive so P11 holds.

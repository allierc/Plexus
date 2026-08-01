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
## Learned patterns
_Distilled across rounds. Updated R4 (2026-08-01). Drop entries once they stop earning space._

**Edits that keep failing — do not repropose:**
- **BUFFER SATURATION is the dominant failure of every growth family.** R4 parent-3: the plain
  CONTROL saturated (n_cells 20804 → NOT EVIDENCE) AND `−reconnect_t1_3d` saturated (5204 → NOT
  EVIDENCE); only the growth-OFF slot (`−vesicle_growth`) returned valid_evidence. RULE: on any
  family with growth+division active, growth-ON slots saturate → invalid. Single-op can cleanly
  measure only growth-REMOVED states. To map a GROWN regime you MUST drop to Loop-II (lower growth
  rate / cap cell count), not add single-ops. (R2 `+vesicle_growth:uniform_ramp` also hit 15002.)
- **Impl-swaps crash — no diag.json.** R4 `=shape_energy_3d:monolayer` → empty `{}`. Joins the
  crash family with `−cell_geometry_3d` / `−cell_adjacency` (load-bearing plumbing). Do not swap
  implementations and do not knock out bookkeeping ops.
- **Division additions go numerically unstable late.** `+divide_3d:hertwig` → force 1→66,
  tension→1.65, body fragments in the last ~50 frames. Treat any late force/tension spike as blowup.
- **Parameter sweeps disguised as hypotheses.** vcap×5 on one composition → surprise 0.00, 92/8
  drift. Numbers are not a hypothesis; change composition identity/routing every slot.

**Predictions repeatedly WRONG (map miscalibrated):**
- **Growth+division controls do NOT stay smooth — they saturate.** R4 parent-3 control predicted
  protr_peak 1.0–1.8 smooth ball; landed saturated 2.839 "branched" 44 tubes. Expect any grown
  control to trip the buffer, not hold a low-protr ball.
- **The protrusion is GROWTH-FED, not purely forced** (R2 surprise). `−morphogen_growth_3d` AND
  `−extrude` BOTH collapse protr_peak → sphere (~1.0–1.4); both necessary. `−vesicle_growth`
  likewise → sphere 1.003 (R4). Knockout of any core growth/force driver → sphere.
- **protr_peak is NOISY at the top; don't predict tight bands.** Leave headroom; prefer large
  expected effects over single-run diffs vs a drifting control.

**Metrics/artefacts that keep misleading:**
- **On any row flagged valid_evidence:false / NOT EVIDENCE (saturated), ALL metrics LIE — ignore
  the whole row.** R4 saturated control read protr_peak 2.839, 44 tubes, tube_len 89 = pure
  saturation artefact, not patterning. Never read morphology off a saturated slot.
- **ta_aspect_len_over_diam, ta_tube_len, retention** REJECTED (9–35 on buds). analyst "tube"
  consensus + "body shrinks" = forced-drainage artefact, not tubulogenesis. mech_p_ratio is
  DIAGNOSTIC only (~1 grown, ~3 forced, ≥40 degenerate), never a scoring clause.
- **watcher is unreliable** — R2 false-CONTRADICTS on real structures; R4 it worked ("supports"
  sphere). Do not let a watcher verdict overturn admitted-metric evidence either way.
- **Counter-reset artefact every round** — header shows "round N, 0 runs, coverage 0%,
  phenotypes {}". Ignore it; the real record is memory.md / analysis.md / knowledge.md.

**Composition families exhausted / near-exhausted:**
- **parent 1** (round-33 forced / vcap base): forced spikes only, no grown regime; extrude +
  morphogen_growth_3d mapped jointly necessary. DONE.
- **parent 3** (uniform mechanical growth): single-op is near-dead — control saturates, monolayer
  swap crashes, only `−vesicle_growth` valid (confirmatory). Further mapping needs Loop-II rate/cap.
- **parent 2** (RD Okuda route): all 4 valid single-op edits already PROPOSED
  (+shape_energy_3d:default, −reconnect_t1_3d, −cell_diffuse, −morphogen_growth_3d) — do NOT
  re-propose; read its results before building on it.

**Standing steer:** ~70/30 conf/adv (surprise 0.33 = productive; R4 hit 0.00 = drift). Removing a
known driver is confirmatory & zero-surprise; the genuinely uncallable edits on growth families
tend to saturate/crash — so the live frontier is Loop-II grown-regime tuning, not more single-op
knockouts.

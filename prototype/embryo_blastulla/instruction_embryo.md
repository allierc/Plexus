# Embryogenesis loop — Scientific Agent Instructions (Phase 1)

## Mission
Your goal is **not** to optimize a number. It is to **discover which operators and couplings produce
the phenomenology of a flowing, dividing, self-partitioning embryonic blastula** — and to *understand
the mechanism*, not just find a setting. Training/each run is an experiment. Every batch answers **one
scientific question** with an explicit, falsifiable hypothesis.

The scientific object is the mapping **operators + parameters → morphology/dynamics → observables**.
When a knob or operator changes an observable, your finding is *which phenomenon it caused* (membrane
deformation / collapse / migration / partitioning), tagged and regime-noted — not the numeric optimum.

## HARD FAILURES (constraints, NOT tradeoffs — a slot that hits any of these is a FAIL, full stop)
- `collapsed > 0`  (any stacked cells)
- `nn_min < r0`    (a cell pair closer than the exclusion distance)
- `escape > 0`     (a cell outside the blastula / in the membrane)
- `accel` bounded ONLY because `vmax` clipped it (i.e. the run leans on the clamp)
Never trade a hard failure for a prettier observable. Rank: first exclude all hard-failing slots,
THEN judge the target phenomenon. If a slot hard-fails, that is the finding to explain, not a result.

## Phase-1 is a STAGED LADDER — advance to the next stage ONLY when the current one is met with NO
## hard failures. Do not chase later stages early (that yields a beautiful but uninterpretable soup).
- **1A — stable blastula + no collapse.** Cells hold even coverage (sunflower + `repel`), 0 collapse,
  0 escape, bounded accel by balance. The membrane may stay ~round. *Gate to 1B: collapsed=0 & escape=0.*
- **1B — inner flow deforms the membrane.** With 1A holding, find the coupling that lets core flow
  visibly reshape the deep-blue membrane (`deform` ↑) while 1A still holds.
- **1C — division pressure deforms the shell.** Add `cell_divide`; proliferation reshapes the shell.
- **1D — high-density flow / migration.** At confluence cells keep flowing (`flow`>0, not jammed) and
  **collective migration** emerges (`migration` ↑, coherent streams).
- **1E — two-type partitioning.** The two types segregate (e.g. left/right; `segregation` ↑).
- **INT — integrate all (Phase-1 capstone).** Combine every established operator into ONE blastula that holds
  together, flows, divides (bounded), and partitions. ESTABLISHED here: `cell_divide` ALONE caps realistic
  growth at ~1.5× — division is a mechanical mixing/repacking event, so 2× dilutes the demix and 3–4× ruptures.
- **ORI — axis orientation (Phase-1 capstone).** Give the integrated blastula a stable, programmed AXIS — e.g.
  an external field (`gravity`) or asymmetric cue that orients the body. Gate: a reproducible body axis that
  PERSISTS (stable orientation, the 1E partition aligned to it), the substrate Phase 2 builds on.

## PHASE 2 — MORPHOGENESIS. Causal chain: **Blastula → Orientation → GROWTH → PATTERN → MORPHOGENESIS.** Do
## NOT attempt morphogenesis before the model can actually grow tissue. Phase 2 feeds directly into Phase 3
## (organogenesis) below — do NOT jump ahead. Same gate discipline + hard failures throughout.
##
## GROWTH-vs-DIVISION ROLES (Phase 2 — do NOT make division the growth mechanism):
##   `cell_grow`   INCREASES tissue volume + creates protrusions (continuous, anisotropic material addition).
##   `cell_divide` FILLS the newly grown volume (repopulation) — it follows growth, it does not create shape.
- **GRO — growth (the prerequisite; START HERE).** Introduce continuous tissue growth via `cell_grow`
  (MPM-material / rest-volume increase, INDEPENDENT of division). Goal: smooth expansion, buds, branches,
  lobes. Learn how ANISOTROPIC growth generates a protrusion that LATER ROUNDS (grow directional → relax /
  isotropic → the elastic + surface-tension physics rounds the bud). Gate: controlled area/volume growth with
  a clean protrusion→rounding, blastula still intact (collapsed=0, escape=0). First GRO batch = an ISOLATED
  `cell_grow` mechanism-validation (zero-growth no-op control + a small growth sweep), NOT a full morphogenesis test.
- **PAT — patterning.** Stable chemical/mechanical identities on the GROWING tissue — domains persist during
  growth, and growth fields become spatially programmable (a chemical field gates WHERE `cell_grow` acts).
  Gate: `mi_type_x` ↑, domains persistent (low late-time `mixing_entropy` drift) even under active growth.
- **MOR — morphogenesis (BODY-SCALE only).** PATTERN CONTROLS GROWTH: localized anisotropic `cell_grow` (gated
  by the PAT field) creates LARGE-SCALE whole-body morphology — elongation, epiboly, gastrulation,
  convergence/extension, lumen formation; remodeling stabilizes. **Buds and branches are NOT MOR — they are
  Phase-3 (BUD / BRN); MOR stops at whole-body shaping.** Gate: directional, pattern-localized BODY reshaping
  (anisotropic strain bands, `t1_rate` fluidity, area growth, or a lumen) traceable to the PAT domains.

## PHASE 3 — ORGANOGENESIS. Phase 1 gave a stable, oriented embryo; Phase 2 gave continuous growth, persistent
## pattern and body-scale morphogenesis. Phase 3 no longer asks WHETHER the embryo can grow — it asks **which
## operator COMPOSITIONS produce complex organ-like morphologies**: the mapping `growth laws + mechanical
## feedback + pattern fields → organ morphology`, NOT parameter optimization. **NEVER introduce a dedicated
## "branch"/"organ"/"fold" operator** — organogenesis must EMERGE from composing the existing primitives
## (`cell_grow`, pattern fields via `deposit`/`diffuse`/`chemotax`, mechanics, continuum elasticity, growth
## feedback `stress_gain`, remodeling `agent_remodel`). **ORG is the TERMINUS — after ORG the campaign is
## COMPLETE, STOP.** The contribution is a mechanistic ATLAS: operator composition → growth law → morphological
## program → organ geometry.
## **INHERIT EARLIER CAPABILITIES (preserve solved axes).** Each Phase-3 stage MUST preserve the previous
## stage's established phenotype: a BRN experiment must keep the BUD phenotype (a stable, non-ruptured bud) —
## UNLESS the hypothesis explicitly studies bud instability; ORG must preserve branching while adding programs.
## Destroying a solved capability to gain the next is a REGRESSION, not progress (treat it like a hard failure).
- **BUD — localized morphogenesis.** Introduce SPATIALLY LOCALIZED growth; find the mechanism that makes a
  stable tissue bud WITHOUT rupture or loss of pattern. Compose: `cell_grow(mode=anisotropic|tip)`, `prestretch`
  (growth-pressure magnitude), PATTERN-GATED `cell_grow` (growth only inside one domain), growth-rate /
  anisotropy modulation. Question: does localized growth NATURALLY produce a reproducible protrusion that then
  ROUNDS by elastic relaxation (emergent, not a scripted bud)? Gate: one reproducible localized bud · embryo
  integrity preserved (collapsed=0, escape=0) · developmental pattern preserved · reproducible strain localization.
- **BRN — branching morphogenesis.** With a stable bud, find which minimal FEEDBACK law turns ONE growing tip
  into MULTIPLE stable branches (branching must EMERGE from feedback, never an explicit branch operator).
  Compose: stress-dependent growth (`stress_gain`), curvature-dependent growth, tip inhibition, growth
  competition, morphogen-controlled growth fields, multiple interacting growth centres, anisotropic remodeling.
  Branch number, spacing, bifurcation angle and stability are EMERGENT observables, not prescribed targets.
  Gate: reproducible bifurcation · stable branch persistence · tissue continuity preserved · controlled branch spacing.
- **ORG — organogenesis (terminus).** Let MULTIPLE developmental programs COEXIST in one embryo — growth
  controlled simultaneously by tissue identity, morphogen fields, mechanics and developmental timing, so
  different regions run different growth programs (e.g. lung-like branching, glandular budding, gut folding,
  vascular arborization, epithelial invagination, repeated appendages). Goal: discover which operator
  combinations generate different CLASSES of organ morphology — not one specific organ. Gate: multiple
  simultaneous morphogenetic programs · persistent developmental identities · stable organ-level structures ·
  reproducible morphology across seeds.

State which stage the batch targets in the analysis entry; only one stage-transition per batch.

**PER-STAGE BUDGET — ≤2 DAYS (48h) OR ≤10 batches per sub-phase, whichever first.** The driver injects a
`>>> TIME CAP HIT` or `>>> BATCH CAP HIT` directive into your prompt when either budget is up — when you
see it, advance IMMEDIATELY (adopt the best clean point, log any open blocker, write the next stage to
`current_stage.txt`). Every stage — Phase 1, Phase 2 (GRO/PAT/MOR) AND Phase 3 (BUD/BRN/ORG) — is hard-capped
at 10 batches. **Ladder order: 1A→1B→1C→1D→1E→INT→ORI→GRO→PAT→MOR→BUD→BRN→ORG, then STOP.** Phase 1 (through
ORI) and Phase 2 (through MOR) are the substrate; the campaign then enters Phase 3 (organogenesis) at **BUD**.
When the cap fires on **ORG** (the terminus), do NOT advance to any new stage — STOP designing experiments;
the campaign is done. In STAGE STATUS record the batch each stage
STARTED. If a stage's gate is not met within its budget, STOP grinding it: log the blocker as `[open]`,
ADOPT the best clean (escape-free) point achieved as that stage's operating spec, and ADVANCE — or, if the
gate is physically unreachable with the current operator set, relax the numeric target to the best value
found and move on. Never spend >10 batches on one rung; breadth across the ladder beats perfecting one.
**Each batch, write the target sub-phase (e.g. `1E`) to `current_stage.txt`** — the loop uses it to name
archive dirs `embryo_<stage>_b<NN>_<slot>` and to count the per-stage batch budget.

## The QUANTITATIVE SCORECARD — decide on NUMBERS, not on the movie
Every slot writes **`scorecard.json`** (+ a `scorecard.png` evolution panel) via `scorecard.py`, with
FIVE families, EACH computed at **5 / 25 / 50 / 75 / 100 %** of the run (so transients, drift and
steady-state are visible — the 3000-vs-6000-frame trap):
- **shape** — `fourier_m1` (drift) `m2` (elongation) `m3+` (lobing), `circularity`, `shape_index` (p=P/√A, ≈3.81 fluid↔solid), `area`, `perimeter`, `deform_rms`.
- **organization** — `gr_peak`/`gr_peak_r` (RDF), `nn_mean`/`nn_cv`, `density_cv`, `contact_same`.
- **flow** — `speed`, `polar_order`, `enstrophy`/`net_circulation` (swirl vs bulk translation), `msd`, `persistence_frames`, `corr_length_xi` (ξ).
- **topology** — `t1_rate` (neighbour-exchange / fluidity).
- **partition** — `segregation_index`, `mixing_entropy`, `mi_type_x`, `interface_frac`.
- **coupling** — `stress_cell_corr`, `deform_cell_corr`, `flow_deform_lag`, `div_stress_angle` (division axis vs principal-stress, Campinho 2013).
- **organogenesis-geometry** (PHASE 3 decision basis; `scorecard_organo.py`, in `scorecard.json["organo"]`,
  same 5 timepoints; headline values also ride along in `metrics.json` as `org_*`). Computed from the LIVE
  tissue mask (dormant `grow_reserve` excluded) at three levels + localization + persistence:
  *outline* `area`,`perimeter`,`circularity`,`aspect_ratio`,`convexity`,`solidity`,`major_axis`,`minor_axis`,
  `orientation`,`body_radius`,`fragment_count`; *bud* `n_buds`,`bud_score` (protrusion-area×neck-sharpness×
  persistence),`bud_area_frac`,`bud_len_bodyR`,`bud_neck_ratio`,`bud_roundness`,`bud_persistence`; *branch*
  (skeleton graph) `n_tips`,`n_branchpoints`,`branch_len_mean`/`_cv`,`branch_width_mean`,`branch_angle_mean`/`_sd`,
  `tree_depth`,`hierarchy_depth` (branching GENERATIONS gen0→gen1→gen2, for repeated branching),`skeleton_length`,
  `branch_score` (bifurcations×continuity×persistence),`branch_persistence`; *localization/causality*
  `growth_bud_overlap`,`pattern_growth_overlap`,`strain_growth_overlap`,`tip_growth_enrichment` (did the bud
  appear WHERE the operator drove growth?); *ORG developmental programs* `independent_growth_domains` (spatially
  separated growth centres = coexisting programs),`program_stability` (do those domains persist over the run).
  All dimensions are relative
  to `body_radius`/`body_area` so a bigger embryo is not scored as better. **Use this family — NOT the movie —
  to gate BUD/BRN/ORG; persistence matters as much as final shape (round→protrusion→necked bud→branch→stable).**
`metrics.json` = the hard-failure gate PLUS the final scorecard. **Read the numbers AND their 5-point
trajectory; the mp4 / 2×2 only PROPOSE a hypothesis — the scorecard DECIDES whether it survives.**

**METRIC TIERS — decide on the top two, read the third as context (do NOT gate on tier-3):**
- **TIER 1 — HARD GATE** (a fail is a fail, never a tradeoff): `collapsed`, `escape`, `nn_min`, `accel`.
- **TIER 2 — PRIMARY PHENOTYPE** (the decision metrics; robust, low binning-dependence): `deform_rms`,
  `fourier_m1/m2/m3`, `shape_index`, `circularity`, `polar_order`, `segregation_index`, `n_cells`,
  `net_circulation`, `msd`.
- **TIER 3 — SECONDARY DIAGNOSTICS** (interpret, don't gate — sensitive to neighbour-radius / binning /
  capture-stride, so validate before trusting): `gr_peak`, `nn_cv`, `density_cv`, `contact_same`,
  `mixing_entropy`, `interface_frac`, `mi_type_x`, `stress_cell_corr`, `deform_cell_corr`,
  `flow_deform_lag`, and the biology-facing `corr_length_xi`, `t1_rate`, `div_stress_angle`.
Rank a batch by: **1. no Tier-1 failure  2. the batch hypothesis's target Tier-2 metric (right
trajectory)  3. Tier-3 for mechanism/interpretation only.**

## QUANTITATIVE REPORT PROTOCOL (mandatory in every `analysis_embryo.md` entry)
For EVERY claim, pair the visual observation with its scorecard support — never a bare visual claim:
> **visual claim:** "the blastula becomes lobed toward the end"
> **quantitative support:** shape.fourier_m3 0.006→0.013 (2.1×) over 50→100%; circularity 0.92→0.78; deform_rms 0.018→0.031
A claim with no scorecard number behind it is an OPINION, not a finding — do not log it as one.

## `[established]` GATE — replicated + significant, or it stays `[open]`
Promote a mechanism to `[established]` ONLY when (a) it ran on **≥3 seeds** (vary `general.seed` across
slots) and (b) the effect vs its ablation control is **larger than noise: |Δ| > 2·SD** across seeds
(report **mean ± SD**). One run + a movie is `[open]` at best. State the seeds and mean±SD in the
ledger entry. LoopScore discipline: visuals propose, statistics decide.

## The action set — PLAY WITH THE OPERATORS (compose from the whole codebase)
You may change scalar params AND **add / remove / swap operators**. Each slot is a full spec you
author (copy `specs/embryo_base.yaml`, edit). Available operators (see `knowledge_embryo.md` for
roles): couplings `agent_to_mpm`, `mpm_to_agent` (`field: mass|colour`), `mpm_spin`, `flow_align`,
`agent_remodel`, `cell_divide`; cell interactions `repel` (hard min-dist `r0`), `attraction_repulsion`
(`p=[pull,pull_range,push,push_range]`), `separation`, `polar_align`; chemical signalling `deposit`+
`diffuse`+`decay`+`chemotax` (per-type channels → cross-repulsion for partitioning) and `relay`+
`adapt`; motility `glide`; MPM `mpm_strain/p2g/mpm_grid_update/g2p`, `mpm_anchor`. Even init:
`spawn: sunflower`. Two `cell`-level sets are fine; per-type behaviour via `agent[type=X]` selectors.

## RULES (bright lines — follow before creativity)
- **R1 Minimal mechanism first.** Before ADDING a new operator, test whether the phenomenon can be
  produced by changing coupling strength, density, stiffness, or damping in the CURRENT operator set.
  Only introduce a new operator once the existing knobs are shown insufficient.
- **R2 Collapse response.** If collapse appears (`collapsed>0` or `nn_min<r0`), FIRST reduce the
  feedback strength `agent_to_mpm.agent_mass` and `mpm_to_agent.k`. Do NOT first increase `repel` —
  exclusion cannot beat the hydrodynamic self-attraction (see ledger).
- **R3 One new operator family per batch.** Do not add more than one new operator family in a batch
  unless the hypothesis is explicitly ABOUT their coupling. Keep batches interpretable.
- **R4 Always a control.** Every batch includes ≥1 ablation control slot (role=control): remove the
  one operator whose effect the batch's hypothesis claims (e.g. without division / flow_align / spin /
  chemical field / type-specific behaviour) so the attribution is causal.

## Scientific method — per-batch cycle (do ALL, in order; auto-update the files)
1. **Observe** — read the previous batch montage + each slot's `progress.txt`/metrics; start from what
   the movies show, not from a plan.
2. **Hypothesize** — write ONE explicit, predictive hypothesis (e.g. "lowering `mpm_to_agent.k`
   below X stops collapse while keeping membrane deformation").
3. **Design 8 slots** into `embryo_slots.md`, one variable/operator change per slot, roles balanced
   ≈ 4 exploit · 3 explore · 1 control. Isolate one lever per slot for causal attribution.
4. **Predict** the observable change each slot should produce.
5. After results: **verify** — supported / falsified / overturned / inconclusive; note the regime.
6. **Append** a dated section to `analysis_embryo.md` (never overwrite prior batches).
7. **Distill** `knowledge_embryo.md` — merge the new causal finding into the right section, tagged
   `[established]/[open]/[rejected]/[engineering]`; keep it compact.

## Slot schema (one line per slot in `embryo_slots.md`)
`name : SPEC specs/<file>.yaml [KEY val ...]`
- `SPEC` names the spec YAML you authored for this slot (compose operators there).
- optional `KEY val` are dotted overrides applied on top (e.g. `mpm_to_agent.k 0.2`,
  `repel.r0 0.024`, `agent.move_speed 0.05`, `agent_to_mpm.agent_mass 1e-6`, `cell_divide.rate 0.6`,
  `n_grid 64`, `frames 12000`, `stride 16`).
- lines starting `#` are comments. Keep exactly 8 non-comment lines (batches of 8, run in parallel on L4).

## Budget & USER DIRECTIVES (mandatory — 2026-07-02)
- **frames ≈ 12000** on every run (long, so slow dynamics fully develop). A ~12000-frame job is ~25–30
  min on L4 — FINE (wall is 45 min). Do NOT shrink below ~12000. Use `stride ≈ 16` to keep
  render time bounded; that only subsamples the movie, not the physics.
- **move_speed baseline 0.12** (2× faster than before); you MAY go up to ~0.24 when a stage needs
  faster flow/migration.
- **cells SHOULD proliferate via `cell_divide`** to grow and deform the blastula (`div_rate`; `buffer`
  is 3000) — there is NO fixed multiplier target; grow the population as the biology/goal calls for and
  do NOT cap proliferation prematurely when 1C/1D calls for density. The ONLY real limit is physical:
  the population must not exceed what the (deforming) domain can hold at `repel.r0`, else cells over-pack
  and `collapsed` just measures jamming rather than the mechanism under test.
Keep `per_parent`/`n_grid` sane; each job must still finish within the 45-min L4 wall.

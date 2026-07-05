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
State which stage the batch targets in the analysis entry; only one stage-transition per batch.

**PER-STAGE BUDGET — ≤2 DAYS (48h) OR ≤10 batches per sub-phase, whichever first.** The driver injects a
`>>> TIME CAP HIT` or `>>> BATCH CAP HIT` directive into your prompt when either budget is up — when you
see it, advance IMMEDIATELY (adopt the best clean point, log any open blocker, write the next stage to
`current_stage.txt`). **1E (two-type partitioning) is HARD — it is HARD-CAPPED at 10 batches: 1E started
at Batch 24, so advance to INT no later than Batch 33.** In STAGE STATUS record the batch each stage
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

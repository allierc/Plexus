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

## The observables (diagnostics, not a single loss) — from `embryo_metrics.phase1_*`
`collapsed` (frac stacked, want 0) · `nn_min`/`nn_mean` (min distance held) · `deform` (RMS membrane
radial displacement, want ↑) · `flow` (mean cell speed, want >0) · `migration` (velocity polar order,
want ↑) · `segregation` (|⟨x⟩_a−⟨x⟩_b|/R, want ↑) · `accel` (95th-pct, want bounded) · `n_cells`.
Rank a batch by, in order: **1. no-collapse (collapsed≈0)  2. the target phenomenon of the batch's
hypothesis  3. flow/migration  4. the rest.** A pretty run that collapsed is a failure.

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
  `n_grid 64`, `frames 6000`).
- lines starting `#` are comments. Keep exactly 8 non-comment lines (batches of 8, run in parallel on L4).

## Budget & USER DIRECTIVES (mandatory — 2026-07-02)
- **frames ≈ 6000** on every run (2× longer, so slow dynamics develop). A ~6000-frame job is ~16 min
  on L4 — that is FINE (wall is 30 min). Do NOT shrink to 1500/3000. Raise `stride` (6–10) to keep
  render time bounded; that only subsamples the movie, not the physics.
- **move_speed baseline 0.12** (2× faster than before); you MAY go up to ~0.24 when a stage needs
  faster flow/migration.
- **cells may grow up to ~4× via `cell_divide`** (`div_rate`/`max_occ`; `buffer` is 3000) — do not
  cap proliferation prematurely when 1C/1D calls for density.
Keep `per_parent`/`n_grid` sane; each job must still finish within the 30-min L4 wall.

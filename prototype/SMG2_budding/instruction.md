# SMG2 Branching Morphogenesis — Scientific Agent Instructions

## Mission
Build a **phenomenological Plexus FORWARD MODEL** of submandibular-gland (SMG) branching
morphogenesis. **Use optimization to DISCOVER MECHANISMS, not to reduce a loss** — this is a
*per-cell fit*'s opposite. Each batch answers ONE **causal question**: which *mechanism* (a
composition of operators) produces which *observable*. The output that matters is a **variance
ledger** (which mechanism explains which observable, and what remains unexplained), not a best score.

Two disciplines run through everything:
1. **Separate WHAT from HOW.** Observables (geometry, topology, velocity, growth, density) are
   measured by *identical code* on the real data and on every forward run. Mechanisms (pairwise
   mechanics, tissue programs, growth law, polarity, signaling, boundary) are what you compose.
   Every experiment maps **mechanism → observable**, never **observable → fit**.
2. **Delay expressiveness.** Climb from pairwise-only → +static program → +slow program → +learned
   SIREN, and only when the simpler class is EXHAUSTED. An over-expressive program explains
   everything and teaches nothing (the F(x,t) degeneracy).

## FORWARD MODEL — START AS CLOSE AS POSSIBLE TO THE REAL DATA
Every forward sim **INITIALIZES from the real SMG state, not a synthetic disc**: seed the cell
positions (and count / density / footprint / boundary) from the real `x_list` initial frame. The
model's job is then to reproduce the **DYNAMICS** (migration + growth + budding) *from* the real
initial condition — a well-posed inverse problem — and every observable is scored by its distance to
the **real observable trajectory**, computed by the SAME code on sim and data.

## OBSERVABLES — the WHAT (measured identically on real data and forward runs)
- **geometry** — body outline, volume, extent, shape index.
- **topology** — the branch GRAPH (geometry → medial axis → branch graph → metrics; `smg_topo.py`):
  `n_tube`, `n_branch`, `n_bud`, AND **branch genealogy** = main duct → branch → subbranch
  GENERATIONS / hierarchy depth (later morphogenesis depends on hierarchy, not count).
- **velocity** — collective-migration field (dense optical-flow PIV; track ids reshuffle so no
  per-cell velocity).
- **growth map** — proliferation source = continuity residual `∂ρ/∂t + ∇·(ρv)` (production, advection
  removed).
- **cell density** — ρ field, packing, nearest-neighbour statistics.
Real-data reference = `real_data_montage.png` (`real_data_montage.py`). Eventually OPTIMIZE against
the **branch graph itself** (graph-edit distance to the real branch graph), not pixel overlap.

## MECHANISMS — the HOW (the "latent tissue programs")
The prescribed/learnable spatial programs are **tissue programs** (biology), not "fields"
(implementation): **growth, polarity, stiffness, (voltage), morphogen** — each may eventually be a
program evolving in space AND time. Mechanism classes (compose from the operator catalog, `knowledge.md` §2):
- **pairwise mechanics** — `repel`, `attraction_repulsion`, `adhesion`, `tension`, `separation`.
- **growth law** — `cell_grow` (volume) + `cell_divide` (repopulate).
- **polarity** — `polar_align`, `velocity_align` (→ collective migration, unjamming/flocking).
- **tissue program / field** — a spatial program gating growth/polarity/stiffness (see expressiveness ladder).
- **signaling** — `deposit`+`diffuse`+`decay`+`chemotax`, `reaction_diffusion` (→ morphogen program).
- **boundary** — confinement / basement-membrane ECM (e.g. as virtual nodes; cf. the real data's
  boundary-localized residual).

## THE CAUSAL-QUESTION LADDER (mechanism decomposition — mirrors embryo & cardio)
Each rung is ONE falsifiable causal question; advance when it is **ANSWERED** (supported/falsified
with a variance-ledger entry), not when a phenotype "looks right." State the question in
`current_stage.txt`; ≤10 batches per question, then record the answer and advance.
- **Q1 — Can collective migration ALONE generate budding?** (polarity + pairwise, NO growth.) Likely
  falsified → but it fixes migration's variance share of the *velocity* observable and shows whether
  flow alone buds.
- **Q2 — Does differential growth create buds?** (growth law ± a STATIC growth program; no SIREN.)
  Fixes growth's share of the *growth-map* and *bud-topology* observables.
- **Q3 — Can growth + migration COEXIST** without one destroying the other? (compose Q1+Q2) → the coupling.
- **Q4 — What determines branch ORIENTATION?** (polarity program / anisotropic growth / boundary).
- **Q5 — What determines branch NUMBER?** (signaling/Turing gate, tip inhibition, growth competition).
- **Q6 — What determines branch ELONGATION?** (growth-rate program: tip vs stalk).
- **Q7 — Can ONE forward model explain ALL observables** simultaneously? (integrate; the ledger should
  show low residual across every observable).

## METRICS SUMMARY + VARIANCE ATTRIBUTION — the deliverable (cardio's "statistics decide")
Do NOT track a best score. Two records, no redundancy:
- **`metrics_summary.md` — the single source of truth for ALL runs.** `smg_showcase.py` APPENDS one
  row per slot: batch · slot · question · mechanism lever · every observable it produced (topology
  counts + trajectory, migration, growth, distance-to-real, seed). **READ it at the start of every
  batch** so you never repeat an experiment and can see the whole campaign at a glance.
- **`knowledge.md` §5 — the distilled causal STORY (narrative, tagged
  [established]/[open]/[rejected]/[engineering]).** Which MECHANISM explains which OBSERVABLE. Do NOT
  keep a comparison TABLE here — that duplicates `metrics_summary.md`.

**Variance share** (the attribution) = the drop in an observable's **distance-to-the-real-trajectory**
when a mechanism class is ADDED vs its ablation control (PIV RMSE / growth-map correlation /
branch-graph edit distance / geometry). `[established]` only when the share is >2·SD across ≥3 seeds
vs ablation. Knowledge grows even when the score does not.

## MODEL-EXPRESSIVENESS LADDER — DELAY THE SIREN (identifiability guard)
Climb ONLY when the simpler class plateaus, recording the variance each step ADDS:
1. **pairwise ONLY** → how much variance does local mechanics alone explain?
2. **pairwise + STATIC program** → a fixed spatial program (time-independent).
3. **pairwise + SLOW program** → slowly-varying (low temporal bandwidth; the slow-ω idea).
4. **pairwise + LEARNED SIREN** → expressive `f(x)→[program]` (cardio template, `omega_0` bandwidth
   knob, autograd-optimized) — LAST resort, `omega_0` kept LOW first.
If a SIREN's added variance share is large only because it absorbed a mechanism you have not yet
tried explicitly, that is a **RED FLAG**, not a win.

## HARD RULES
- **NO scripted bud/branch/tube operator** — topology must EMERGE from composed primitives.
- **Mechanism → observable, never observable → fit.**
- **Simplest mechanism class first**; expressiveness (→SIREN) only when the simpler class is exhausted.
- **One new mechanism family per batch**; **always an ablation control** (remove the mechanism the
  question credits, so attribution is causal and the variance ledger is valid).
- **No collapse / escape**; if collapse appears, first lower the mechanical feedback gain (not `repel`).

## SCORECARD — decide on NUMBERS (`smg_scorecard.py`), 5/25/50/75/100 % of the run
Per-observable distances feed the ledger: **topology** (`n_bud`/`n_branch`/`n_tube` + `genealogy` vs
target trajectory; bud ±20 %, branch ±1, tube exact), **temporal consistency** (`bud_trend`,
`bud_growth_ratio`, `branch_monotonic`, `count_jitter`, `tube_stability`), **geometric fidelity**
(`coverage`, `chamfer`), **migration** (`polar_order`, `speed`, `corr_length_xi`, PIV RMSE vs real),
**growth** (net ratio, `growth_tip_localization`, growth-map correlation vs real). Read the numbers
AND their 5-point trajectory; the mp4/montage only PROPOSE a hypothesis — the ledger DECIDES.

## QUANTITATIVE REPORT PROTOCOL (every `analysis.md` entry)
Pair every visual claim with its observable-distance + variance-share number, e.g.:
> visual: "adding growth makes buds"; quantitative: bud-topology residual 0.62→0.38 (growth share
> +24 pts vs no-growth ablation); growth-map corr 0.11→0.57. A claim with no number is an opinion.

## `[established]` gate
Promote only with ≥3 seeds AND variance share |Δ| > 2·SD vs the ablation control (report mean±SD).

## REPO RESOURCES — read / mine these (do NOT restart from scratch)
**Plexus framework:** `/workspace/Plexus/paper/plexus.tex|pdf` (+ `plexus.md`); `paper/cardio_mpm.tex|pdf`
(a worked loop as a paper); `paper/fig_ops.*`, `fig_hier.*`.
**Operator sources:** `src/plexus/operators*`; `prototype/active_matter2/am2_ops.py`;
`prototype/embryo_vertex/embryo_vertex_3d_ops.py` + `run_embryo_vertex_3d.py` (**3D vertex** — Okuda
Turing+vertex); `prototype/ops_grow.py`, `grow_engine*.py`.
**Prior agentic loops to MINE (same design→run→score→distil method):**
- `prototype/embryo_blastulla` — **not conclusive but rich with hundreds of specs + paired
  outputs/analyses/scorecards**; GRO/BUD/BRN/ORG = our growth→bud→branch physics. Read
  `knowledge_embryo.md`, `analysis_embryo.md`, grep `archive/embryo_{GRO,BUD,BRN,ORG}_*` for recipes.
- `prototype/cardio_mpm` — the ORIGINAL template (`instruction_cardio_mpm.md`, `knowledge_cardio_mpm.md`,
  `cardio_mpm_loop.py`): the LoopScore/variance discipline AND the **SIREN tissue-program + MPM**
  wiring (`cardio_mpm_train.py`: `f(x)→[stiffness/fibre/gain]`, `omega_0` bandwidth; `mpm_scatter/
  gather/strain/grid_update` + `active_stress`). Use it as the SIREN reference — but only at rung 4.
**Literature (`papers/`):** `Tissue_active_matter.pdf`, `organs_genesis_review.pdf`, `SimuCell3D.pdf`,
`Multiscale_active_matter.pdf` (mechanism→operator map in `knowledge.md` §2). **Cloned repos** in
`papers/`: `tyssue` (3D vertex), `cellGPU` (SPV jamming/migration), `reaction-diffusion`, `cell-sorting`,
`growing-nca`, `SAMoS`, `zebrafish`. **To clone cluster-side** (`knowledge.md` §6): `germannp/yalla`,
`PhysiCell`, `CompuCell3D`, Iber branching-Turing.
**Sibling prototypes:** `embryo_vertex`, `embryo_gray_scott`/`embryo_nca`/`embryo_cell_sorting`/
`embryo_french_flag` (patterning), `cardio_mpm`/`mpm_3d` (MPM). Reuse; don't reinvent.

## Per-batch cycle (do ALL, in order; auto-update files)
1. **Observe** the previous montage + each slot's scorecard/metrics (numbers, not the movie).
2. **Append** a dated `## Batch N` to `analysis.md` (quantitative report protocol) AND update the
   **variance ledger** in `knowledge.md`.
3. **Distill** `knowledge.md` — tag findings `[established]/[open]/[rejected]/[engineering]`.
4. **State the batch's CAUSAL QUESTION** (Q1..Q7) and ONE predictive hypothesis.
5. **Design 8 slots** into `smg_slots.md` (~4 exploit · 3 explore · 1 ablation control), one mechanism
   lever per slot, so each slot contributes a clean variance-share measurement.
6. **Predict** each slot's observable change; after results, **verify** supported/falsified + update ledger.

## Slot schema (one line per slot in `smg_slots.md`)
`name : SPEC specs/<file>.yaml [key val ...]`  — 8 non-comment lines, run in parallel on L4.

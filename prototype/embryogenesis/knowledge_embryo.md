# Embryogenesis (active matter × MPM) — knowledge ledger

Cumulative, curated working memory for the agentic loop. CUMULATIVE: add/curate, never erase.
Tags: **[established]** (causal, reproduced) · **[open]** (hypothesis to test) · **[rejected]**
(falsified) · **[engineering]** (about the tooling/metric, not the biology).

---

## Current objective (Phase 1, one line)
Inner-core cell **flow deforms the outer membrane**; cells **never collapse** (a hard minimum
cell-cell distance holds); motion stays bounded by **parameter balance, not the velocity clamp**;
**division** progressively deforms the blastula; cells **keep flowing at high density** (collective
migration emerges); two types **partition** the blastula (e.g. left/right).

## The system (what exists)
A **blastula**: a thin elastic **membrane** shell (deep blue) enclosing a **water core** (light
blue), held by a **substrate anchor** so the shell contains the fluid and nothing drifts. **Cells**
= active-matter agents living in the core; they are dragged by the fluid, confined to the core, and
deform the membrane. Rendering: cells = coloured dots by type; material = blue (two blues for
membrane vs core); black background. Every run also emits the **2×2 mp4** (cells+material / stress /
deformation / cell tracks) and a VLM caption.

## Operators (the agent's action set — compose freely from the whole codebase)
NEW couplings (src/plexus/operators/):
- `agent_to_mpm` — cells scatter momentum onto the MPM grid → **deform** the material. [established]
- `mpm_to_agent` — grid velocity **drags** cells (`k`) + **confines** them up a density/colour
  gradient (`confine`, `field: mass|colour`). [established]
- `mpm_spin` — drive the disc toward slow solid-body **rotation** (`omega`). [established]
- `flow_align` — cell polarity relaxes toward the local **flow** (SPV polarity–velocity rule). [established]
- `agent_remodel` — cells **soften/rigidify** the tissue (μ,λ) via per-type `remodel_rate`. [open — untested in blastula]
- `cell_divide` — **proliferation** on a fixed `buffer` via occupancy; per-type `div_rate`. [established]
Reused active-matter ops: `repel` (hard-core min distance `r0`), `attraction_repulsion`
(equilibrium spacing; `p=[pull,pull_range,push,push_range]`), `polar_align`, `chemotax`/`deposit`/
`diffuse`/`decay`/`relay`/`adapt` (chemical signalling), `separation` (boids even_spacing),
`glide`. MPM: `mpm_strain/p2g/mpm_grid_update/g2p`, `mpm_anchor`, `mpm_drag`.
Even initialisation: `spawn: sunflower` (Vogel golden-angle lattice — cells start equidistant).

## Established mechanisms
- **[established]** Two-way agent↔MPM coupling routes through the shared grid: cells push the grid
  (agent_to_mpm) and are dragged/confined by it (mpm_to_agent); this gives genuine mutual
  deformation. Reused by every spec.
- **[established]** A **bounded substrate** = elastic body + `mpm_anchor mode:substrate` + density/
  colour confinement → 0 escape, no drift over 1500 frames (disc_growth≈0, aniso≈0.001). A free
  liquid disc instead slowly volume-drifts and sprays its skin — so anchor the body.
- **[established]** **Even coverage recipe**: `spawn: sunflower` + hard `repel` with `r0 ≈` the
  confluent spacing keeps a stable, uniform, non-clustering tiling (min NN held at `r0`, 0 stacking).
- **[established, brief]** Strong two-way coupling causes a **hydrodynamic self-attraction that
  collapses cells into clusters** — the drag reads back a cell-contaminated grid velocity (effective
  velocity-alignment/MIPS). Keep the coupling **weak** (small `agent_mass`, modest `mpm_to_agent k`)
  so cells deform slowly without collapsing. *(Details omitted — a rabbit hole; the lesson is the
  weak-coupling regime.)*
- **[engineering]** `vmax` (per-set speed clamp) exists as a safety net, but the goal is a parameter
  balance where accelerations stay bounded WITHOUT hitting it — treat a run that leans on `vmax` as
  not-yet-balanced.

## Compelling results (the early, vivid runs — keep these as touchstones)
- **water disc** (`agent_mpm_disc_water_v3`): two cell types swim in a cohesive rotating water
  blob, form aligned streams, gently lobe the boundary; 0 escape.
- **elastic disc** (`agent_mpm_disc_elastic_v3`): a coherent stream of cells migrates across an
  elastic tissue (**polar order ≈ 0.46** — strong collective migration); rounder, 0.25% escape.
- **4-type showcase** (`agent_mpm_disc_4types_show`): four cell types with distinct behaviour
  (flocker / disperser / aggregator / noisy) in one disc; deforms strongly.
- **blastula + 4 types** (`agent_mpm_blastula_4types_v1`): the four types inside the two-blue
  membrane+core; collective push deforms the shell into an egg/teardrop; 0 escape.
These are the phenomenology to recover and understand — vivid flow, migration, and shape change.

## Open questions (Phase 1 — design experiments around these)
- **[open]** Which coupling strength (`agent_to_mpm.agent_mass`, `mpm_to_agent.k`) lets inner flow
  visibly **deform the membrane** while keeping cells non-collapsed and still flowing? (the central
  balance)
- **[open]** Does `flow_align` + gentle `mpm_spin` produce **collective migration** at high density
  without jamming? What sustains continuous flow at confluence?
- **[open]** How do two types **partition** (left/right)? Which mechanism drives it — differential
  `move_speed`/`div_rate`, chemical cross-repulsion (`deposit`+`chemotax` with opposite channels),
  or `agent_remodel` making each type stiffen its territory?
- **[open]** Does `cell_divide` (proliferation pressure) deform the blastula, and how does the
  membrane thickness/stiffness gate that?

## GLOBAL open theme — the stress ↔ deformation ↔ active-cell relationship
The central scientific object of this project is the **three-way coupling between the material
STRESS field, the material DEFORMATION field, and the ACTIVE CELLS** — where cells put stress,
how that stress maps to deformation, and how the resulting flow feeds back on the cells. This is to
be **understood empirically through observation + test/validation/falsification in the loop**, NOT
asserted from theory. Every batch should read the 2×2 (cells / stress / deformation / tracks) as
one coupled system and ask a falsifiable question about a link in that chain (e.g. "do cells raise
stress locally where they push?", "does membrane deformation lag cell flow?", "is deformation
localised to shear-stiff material only?"). Promote a link to [established] only after a run confirms
it; keep the rest [open]. The specific claims below are instances of this theme.

## Hypotheses to TEST in-loop (asserted from theory — validate or falsify; do not treat as fact)
- **[open, H1]** *Stress & deformation concentrate in the outer ring because only the elastic
  MEMBRANE is shear-stiff (μ>0); the liquid core (μ=0) bears no shear stress/deformation.* Predicted
  from the fixed-corotated law `2μ(F−R)Fᵀ + λJ(J−1)I` + liquid `mu=0`. **Falsify:** (a) make the core
  elastic (`youngs>0`, no liquid layer) → the stress/deform field should FILL the interior, not just
  the ring; (b) make the membrane liquid → the ring should VANISH. If either fails, our reading of
  where load lives is wrong. *(A general rule: whenever an observation is explained only from theory,
  log it here as [open] with a falsification test and let a batch settle it — do not promote to
  [established] until a run confirms it.)*
- **[open, H2]** *The `collapsed` metric double-counts freshly-divided daughters (spawn offset ≈
  0.2·r0).* Threshold now 0.15·r0 to exclude them; verify a division-only run reports collapsed≈0.
- **[open, H3] Collapse is DENSITY-DEPENDENT** (observed 2026-07-02, 400f, base coupling): 12 cells
  → collapsed 0.0 (+ segregation 0.29); 60 cells → 0.67 (pairs); 265 cells → 0.98. I.e. the
  hydrodynamic self-attraction grows with cell NUMBER; raising `repel.r0` alone does not fix it
  (12-cell r0=0.16 worked because SPARSE, not because r0 was big). **Falsify/validate:** sweep `n` at
  fixed coupling and fixed r0 → expect collapse onset above a critical density; then lower
  `mpm_to_agent.k`/`agent_to_mpm.agent_mass` and expect the critical density to rise. Consequence for
  Phase 1: reach confluence by keeping coupling weak, not by cranking exclusion.

## Rejected / dead ends (one line)
- **[rejected]** Beating the coupling-collapse by cranking `repel` gain/`r0` — exclusion cannot win
  against the hydrodynamic attraction; the fix is weak coupling, not stronger repulsion.

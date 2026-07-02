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

## STAGE STATUS (updated Batch 5, 2026-07-02)
- **1A — stable, no collapse: MET.** Recipe = `mpm_to_agent.confine 0` (see established block). collapsed=0,
  escape=0, nn_min≥r0, accel bounded by balance (not vmax), at n=44 AND confluent n=265. Base = `embryo_1A.yaml`.
  *Caveat (Batch 2):* at RUNAWAY division to n=1600 the disc is OVER-confluent — natural spacing ~0.015 < r0=0.02,
  so nn_min pins ~0.002 (repel can't hold packed cells apart). collapsed stays ~0 (those are daughters); but
  keep n bounded (cap `agent.div_rate`) to stay a true Stage-1A tiling with nn_min≥r0.
- **1B — inner flow deforms membrane: CLOSE, clean route IDENTIFIED (Batch 4).** Best clean points so far
  (escape≈0): deform **0.0148** (s5 m2e4+spin0.6, migr 0.687, escape 0.0105), **0.0115** (s7 n=44 m5e4, escape 0,
  TRUE tiling), **0.0106** (s6 n=224 m5e5, escape 0). None yet crosses the nominal 0.02 CLEANLY (s3 hit 0.0199 but
  at escape 0.042 = hard fail). Mechanism now clear (see deform/escape blocks): deform grows with BOTH mass and n;
  **escape = per-cell push × density (BOTH needed)** — so the clean frontier avoids the high-n×high-mass corner.
  Two clean routes: **sparse-n + high-mass + spin** (s7 pointer) and **dense-n + low-mass** (s6). Batch 5 pushes
  sparse-n(≈44) + mass(→1e-3) + spin(→1.0) to try to cross deform 0.02 at escape 0.
- 1C division / 1D migration / 1E partition — not yet attempted (do not chase early).

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
- **[established, 2026-07-02, Batch 2] CONFINEMENT IS THE COLLAPSE DRIVER — H5 confirmed by crossed
  ablation.** `mpm_to_agent {confine, field: colour}` adds `+confine·grad(colour)` pushing every cell
  inward up the material-colour gradient; that centripetal drift, not any force, stacks cells. Crossed
  ablation at n=44: `drag0` (k=0, confine 3.0) → collapsed=0.568 (unchanged from base); `confine0`
  (confine=0, k on) → collapsed=**0.000**, nn_min=**0.0291 > r0**, escape=0. Collapse tracks `confine`,
  NOT drag `k`. Dose-response is a THRESHOLD not a line: 3.0→1.0→0.5 barely moves it (0.568→0.523→0.477),
  0.5→0 crashes to 0 — critical confine sits in (0, 0.5). **Stage-1A recipe: set `confine 0`** (substrate
  anchor + elastic membrane + density confinement already retain cells — escape stayed 0, r_cell_max even
  dropped). This is the `specs/embryo_1A.yaml` operating point.
- **[rejected→corrected, 2026-07-02, Batch 2] "COLLAPSE ∝ DENSITY" was a CONFOUND.** The Batch-1
  density law (n=44→0.568 … n=265→0.98) held ONLY because every one of those runs had confine=3.0.
  With `confine 0`, n=265 division-ON → collapsed=**0.000**, escape=0 (`confine0_dense`). Density does
  NOT cause collapse. Corrected law: **confinement causes collapse; density AMPLIFIES confine-driven
  collapse.** Remove confine and confluence is collapse-free. (Turning division off still lowers n and
  was the biggest Batch-1 lever only because it reduced the confine-amplifying density.)
- **[rejected, 2026-07-02] H4 — active ops (`flow_align`/`glide`) cause the collapse.** Batch-1 s0
  (flow_align.gain 0), s2 (move_speed 0) and s3 (mass 0, k 0) EACH left collapsed at 0.96–0.985,
  indistinguishable from the 0.977 baseline. None of the active operators nor the passive coupling
  drives collapse. *Caveat: these three ran with division ON (n=265), so density could mask the
  effect; Batch 2 re-tests flow_align/glide at fixed n=44 (s5/s6) to close this cleanly.*
- **[rejected, 2026-07-02] Passive agent↔MPM drag causes collapse.** Both low_k (Batch-0) and s3
  no_couple (mass=k=0, Batch-1) left collapse untouched. The drag `k` is exonerated. *(The historical
  "hydrodynamic self-attraction / coupling-MIPS" story is fully rejected for the DRAG channel.)*
- **[resolved→established, H5, Batch 2] Confinement WAS the collapse driver — see "CONFINEMENT IS THE
  COLLAPSE DRIVER" above.** Crossed ablation settled it: collapse tracks `confine`, not drag `k`; confine 0
  → collapsed 0, nn_min>r0, escape 0, at n=44 AND n=265. Escape-trade risk did not materialise. Open tail:
  the exact critical `confine` in (0, 0.5) is unprobed (transition is a near-switch, not a ramp) — low
  priority now that `confine 0` is a clean Stage-1A point.
- **[established, 2026-07-02, Batch 2] `agent_to_mpm.agent_mass` IS the prime membrane-deform lever — monotone,
  ~15×.** On the confine-0 (Stage-1A) base, deform rises 0.0043→0.0067→0.0526→0.0625 as agent_mass goes
  2e-6→1e-5→5e-5→2e-4 (migration tracks it, 0.22→0.42→0.73). This is the cells→grid push channel: heavier cells
  scatter more momentum onto the grid, displacing the shell. Confirmed the Batch-2 hypothesis direction.
- **[established, 2026-07-02, Batch 2] DEFORM and ESCAPE are CONFOUNDED through agent_mass — big deform at
  mass≥5e-5 is BLOWOUT, not clean reshaping.** escape climbs in lockstep with deform (0.014→0.022→0.146→0.213)
  and r_cell_max exceeds 1.0 (1.22 at 5e-5, 1.27 at 2e-4) — cells are pushed OUTSIDE/through the membrane. So
  raising agent_mass alone cannot pass Stage-1B: the same knob that deforms the shell ejects the cells. Root
  cause (hypothesised, Batch 3 tests it): with `confine 0` there is NO inward force holding cells off the shell,
  and at runaway n=1600 the over-packed core presses cells against the wall; the push then punches them through.
  Stage-1B requires DECOUPLING deform from escape (contain cells while deforming the shell), not a bigger push.
- **[rejected, 2026-07-02, Batch 2] "Softer membrane deforms more" (youngs-limited deform).** youngs 200→80 left
  deform byte-identical at base mass (0.0043) and slightly LOWER at 5e-5 (0.0489 vs 0.0526), same escape. Shell
  stiffness in [80,200] is NOT the deform bottleneck — containment/coupling is. Do not chase membrane softening.
- **[established, 2026-07-02, Batch 2] `mpm_spin` deforms by CLEAN internal flow — best deform-per-escape.**
  omega 0.3→0.6 raised deform 0.0043→0.0069 (1.6×) at the LOWEST escape of the batch (0.011), cells staying
  inside (r_cell_max 0.97). Circulation reshapes the shell from within without ejecting cells — the right *kind*
  of deform (just small). `agent.move_speed` is a poor lever by contrast (raises escape/r_cell_max faster than
  deform). Keep spin as a clean deform amplifier to stack with a contained high-mass regime.
- **[established→REVISED, Batch 5] ESCAPE = per-cell PUSH × DENSITY (BOTH needed) — not confluence-packing
  ALONE.** Batch 3 said escape is purely confluence-pressure-driven; Batch 4 shows it ALSO rises with per-cell
  mass at FIXED n. At n=95, escape climbs 0→0→0.0105→0.0421→**0.1474** (r_max 0.79→0.88→0.93→0.93→**1.20**) as
  mass goes 2e-6→1e-4→2e-4→5e-4→1e-3 — a big enough push ejects cells even at moderate n. BUT density modulates
  the threshold: the SAME mass 5e-4 gives escape **0.042 at n=95** yet **0.000 at n=44** (s7, r_max 0.883, true
  tiling). So escape fires only in the high-push × high-density corner; either lever alone is safe. **Sparse n is
  an escape SHIELD** — drop n and you can push each cell harder without ejection. (Original Batch-3 finding that
  capping n from 1600→95 cut escape 0.146→0 still holds; it was one axis of a two-axis gate.)
- **[established→REVISED, Batch 5] DEFORM grows with BOTH agent_mass AND n — mass IS a lever at fixed n.** Batch 3
  claimed "deform ∝ n×mass, NOT set by mass at fixed n"; that was an artefact of a NARROW mass window (2e-5 vs
  5e-5). Over a wide range at FIXED n=95, deform rises 0.0027→0.0052→0.0105→0.0199→**0.0346** as mass
  2e-6→1e-4→2e-4→5e-4→1e-3 (500× mass → 13× deform, ≈ floor + slope·mass). Deform ALSO grows with n at fixed mass
  (Batch 3: 0.0067@n=95 → 0.0526@n=1600 at 5e-5). So deform is driven by TOTAL scattered momentum, and either
  per-cell mass OR cell number supplies it. Consequence: the decoupling axis WORKS — buy deform from mass at fixed
  n. But deform and escape stay coupled through mass at fixed n (both rise together), so the clean route is sparse
  n (escape shield) + high mass + spin, or dense n + low mass. Same clean deform ceiling ≈0.015 either way so far.
- **[established, Batch 5] TWO CLEAN Stage-1B routes to moderate deform at escape 0 (anti-diagonal frontier).**
  (a) **sparse-n + high-mass:** s7 n=44, mass 5e-4 → deform 0.0115, escape 0, nn_min 0.0229>r0 (true tiling);
  (b) **dense-n + low-mass:** s6 n=224, mass 5e-5 → deform 0.0106, escape 0, r_max 0.777. Both avoid the
  high-n×high-mass escape corner. `mpm_spin` stacks cleanly on EITHER (s5: m2e4+ω0.6 → deform +40% to 0.0148,
  migration 0.291→**0.687**, escape unchanged). Crossing deform 0.02 CLEANLY is the open Stage-1B target.
- **[rejected, 2026-07-02, Batch 3] Sub-threshold boundary confine (`confine 0.2, field: mass`) contains cells
  without collapse.** FALSE: confine 0.2 (field:mass) → collapsed=**0.579**, nn_min 0.0021≪r0. Confinement drives
  collapse regardless of the gradient field (mass or colour); critical confine sits <0.2 (Batch-2 threshold was in
  (0,0.5); now tightened to <0.2). Do not use confine for boundary containment.
- **[rejected/inert, 2026-07-02, Batch 3] Stiffer `g2p.wall_contact` (0.04→0.12) as a containment lever at low n.**
  At n=95 the run is BYTE-IDENTICAL to the 0.04 default (deform 0.0067, escape 0, r_max 0.818) — no cell reaches the
  wall (r_max 0.82 < shell ~0.9), so wall_contact has nothing to gate. It can only matter under wall pressure (high
  n), which is exactly the escape regime; untested there. Not a low-n lever.
- **[engineering, 2026-07-02, Batch 2] At n=1600 the disc is OVER-CONFLUENT (nn_min<r0 is packing, not collapse).**
  Runaway division (div_rate 0.6, 3000 frames) fills to n=1600; the confluent spacing (~0.015) is below r0=0.02,
  so nn_min pins ~0.002 on every slot even at collapsed≈0 (the sub-r0 pairs are freshly-divided daughters, H2).
  Treat nn_min<r0 at high n as expected packing, not a hard fail — but cap `agent.div_rate` to keep a true 1A
  tiling (nn_min≥r0) and to avoid the pressure-eject that drives escape.
- **[engineering]** `vmax` (per-set speed clamp) exists as a safety net, but the goal is a parameter
  balance where accelerations stay bounded WITHOUT hitting it — treat a run that leans on `vmax` as
  not-yet-balanced.
- **[engineering, 2026-07-02] `escape` was never measured — now added.** `embryo_metrics.py` reported
  no `escape` field even though it is a HARD FAILURE in the instruction. Added `escape` = frac of live
  cells with radius > 0.9·Rd (out of the water core, into/through the membrane) + `r_cell_max`. This is
  essential to read the confine ablation: `confine` in `mpm_to_agent` both (a) squeezes cells inward up
  the colour gradient (suspected collapse driver, H5) AND (b) keeps cells inside the core. `confine 0`
  removes BOTH, so a drop in `collapsed` there is only a real Stage-1A win if `escape≈0`. Without this
  metric the confine ablation is uninterpretable. Backward-compatible; does not touch in-flight jobs.

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
  *Update 2026-07-02 (Batch 1):* density-dependence is now **[established]** (n↑→collapsed↑, near
  deterministic — see Established mechanisms). The per-cell clustering force is NOT active-driven
  (flow_align/glide rejected) nor drag-driven; lead suspect is now the **confinement** term (H5).
- **[rejected, H4]** Active `flow_align`/`glide` cause the clumping — see Established mechanisms
  (Batch-1 s0/s2 unchanged at 0.977/0.985). Superseded by **H5 (confinement)**.

## Rejected / dead ends (one line)
- **[rejected]** Beating the collapse by cranking `repel` gain/`r0` — exclusion cannot win against
  the collapse force. Reconfirmed Batch-1 s4: r0 0.02→0.04 & strength 8→20 moved nn_min only
  0.0002→0.0007 (still ≪ r0), collapsed 0.959, and DOUBLED accel (0.0025→0.0052). The fix is to
  remove the squeeze + control density, not stronger repulsion.
- **[rejected]** Passive agent↔MPM drag (`mpm_to_agent.k`, `agent_to_mpm.agent_mass`) as the collapse
  cause — low_k (Batch-0) and no_couple (Batch-1 s3) both inert.
- **[rejected]** `flow_align`/`glide` (active MIPS) as the collapse cause — Batch-1 s0/s2 inert.

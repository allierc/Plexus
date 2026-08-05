# The campaign

*This file is shown to **every** role, before its own instructions. It holds what the campaign is
for and what has been learned; each role's own file holds only how that role does its job.*

*You — Cedric — edit this file between rounds. That is the point of it: it is the one place where a
human can make the next round smarter without touching code. The Analyst writes `knowledge.md` each
round; fold what survives into this file.*

## Objective

Build the **causal lever-map** of the Okuda mechanism space: for every operator, every
implementation and every routing — what does it do ALONE, and what does it do IN COMBINATION?
Mixtures rarely surrender their causal structure to inspection, so the combinations have to be run.

The product is **a map, not a winner**. Specific questions — which composition makes a sustained
tube, which reproduces Okuda's (χ, γ) phase diagram, which mechanism is necessary for branching —
are *queries against* that map, and each is answered as a by-product of covering it.

## The system

A closed epithelial vesicle: a 3D vertex model of polygonal cells, carrying a reaction–diffusion
chemistry on its surface. The chemistry can steer the mechanics (activator → growth, → line
tension, → bending) and the mechanics can steer the chemistry back (curvature → production). Okuda
et al. 2018 report three morphologies out of this coupling: **tubes**, **undulations**, and
**branches**.

## The discipline

- **A change of numbers is never a new mechanism.** Composition identity is deliberately
  parameter-blind: the same operators wired the same way is the same composition, whatever the
  dials say. Retuning is a legitimate and useful experiment — it just is not a new hypothesis, and
  it must be posed as a sweep, not as a discovery.
- **Every candidate carries a falsifiable prediction, recorded before it runs.** One named metric,
  one direction, one threshold. "It will look interesting" is not a prediction.
- **Roughly 70% confirmatory, 30% adversarial.** Confirmatory edits consolidate the map;
  adversarial edits try to break the current best explanation. Pure confirmation carries near-zero
  information. Pure falsification never accumulates anything.
- **A prediction you are sure of is worth little.** Prefer the edits whose outcome you genuinely
  cannot call.
- **One edit per candidate.** Two edits and a result cannot be attributed. The exception is a
  declared pair test, where the point *is* the interaction and both single edits are already run.
- **Slot 0 is always the parent unchanged.** The control. Without it a difference between
  candidates cannot be separated from seed noise.

## What is known

*Established by execution, with the number that makes it a fact. Do not re-derive these.*

- **`sat` at 0 makes the chemistry a limit cycle, not a pattern.** Gierer–Meinhardt saturation
  `a²/(h(1+κa²))` at κ = 0 is unbounded. `okuda_route` ran as a stable limit cycle of period
  **53.0 frames** — 17 cycles, constant amplitude, FFT and peak-count agreeing — reaching
  `red_frac` 0.997 at each peak with `act_cv` 0.129. Near-uniform when it fires, so it grew to
  3,975 cells and stayed a sphere. **Set `sat` > 0.**
- **`coral_gate` is the working chemistry.** `act_cv_peak` 2.20, `act_alive_frac` **1.00**,
  100 spots, `corr_act_rad` 0.435 on every frame, `valid_frac` 1.0, all ten premises holding. **Its
  morphology is a lobed sphere, not a branch** — the classifier labels it *branched* and the eye,
  looking at the frames, refuses that: "these are bulges over the spots, not fingers or tubes;
  nothing elongates, detaches, or self-intersects." The numbers agree with the eye — `n_tubes` 0 and
  `protr_peak` 1.145. Treat `morphology` as a hint and the protrusion metrics as the fact.
- **Gray–Scott needs a seed.** Its u = 1, v = 0 state is a *stable* fixed point: without
  `cell_rd_seed` nothing ever happens.
- **A reservoir caps the tissue.** A closed trivalent sheet obeys V = 2F − 4, so a vertex buffer of
  size V caps the cell count at (V+4)/2 regardless of the biology. 3,552 vertices give exactly
  1,778 cells — the number that voided **59 runs across two batches**, both reported as findings
  before the arithmetic was noticed.
- **`l_th_frac` above its declared range destroys the tissue.** At 1.96 against a declared ceiling
  of 0.12, the T1 flip threshold is nearly twice the mean edge length, so every junction is
  eligible to flip every fourth frame: the mesh rearranges continuously, cannot hold a shape,
  drains volume, thins its cells and folds through itself. Proven by a one-parameter revert.

## What is ruled out

*Do not re-propose these. Each cost a round.*

- **`divide_3d` + `shape_energy_3d:monolayer` is a broken pair.** The 2×2 table is filled: division
  alone is clean, monolayer alone is clean, together the division does not split the basal ring.
  This is a substrate bug, not a biological result. Avoid the pair until it is fixed.
- **The declared search space contains none of the working recipes.** All six pool parents carry
  out-of-range parameters: `l_th_frac` in 6/6 (0.35 vs a 0.12 ceiling), `shape_energy_3d.Lambda` in
  3/6 (3 vs 0.3 — 10×), `morphogen_growth_3d.rate` in 3/6 (below its floor), `a_sw` at 50 vs a
  ceiling of 6. So a `set_param` edit offered by the menu samples a region **no working recipe
  occupies**. Until the boxes are re-derived, treat a value inside the declared range as *unproven*
  rather than safe, and prefer edits that stay near a parent's own measured value.

- **Retunes filed as mechanisms.** 107 of 107 compositions verified parameter-blind, so a retune
  proposed as a new composition is refused before it runs and wastes the slot.
- **`l_th_frac` at 1.96 destroys the tissue, and round 2 was entirely that.** Eleven of twelve
  round-2 runs carried `reconnect_t1_3d.l_th_frac` at **1.96**, against **0.28–0.35 in every working
  recipe**. At that value the T1 flip threshold is nearly twice the mean edge
  length, so every junction is eligible to flip every fourth frame: the mesh rearranges continuously,
  cannot hold a shape, drains volume, thins its cells and folds through itself (P1, P7, P11 broken,
  `valid_frac` 0.0, `protr_peak` railed at 1.001). **Proven by a one-parameter revert** to the
  parent's 0.28 — which is the evidence, and note that the declared range is *not*: the space declares
  `[0.01, 0.12]`, so 0.28 is outside it too and works perfectly. The twelfth run
  had `l_th_frac` 0.28 and broke anyway — it carried `divide_3d` + `morphogen_growth_3d`, already
  ruled out above.

  *A diagnosis that was offered and is FALSE, recorded so it is not proposed again:* "growth against a
  frozen shell radius." `morphogen_growth_3d` has rescaled `R0` from the target volume since
  2026-07-31, five days before round 2 ran, and `r002c_03` broke P7 and P11 with **no growth operator
  at all** — a run that never grows cannot buckle from a growth radius. The claim was produced
  confidently from four runs' metrics by a role that had been given an empty history, and it is what
  every role's history block exists to prevent.

## What is pinned

*The reference loop this campaign is measured against carries an explicit "things you must NOT change"
list, and it is why its slots are never spent re-settling a question. Ours:*

- **The seed cell count** (`seed_mesh_3d.n_cells`). Not an axis of the search: it is grounded on every
  slot so runs stay comparable, and a reservoir big enough for the target is checked before launch.
- **The vertex and cell reservoir sizes.** Derived from the target cell count, never proposed. A
  closed trivalent sheet obeys V = 2F − 4, so a buffer is arithmetic, not a choice.
- **`conserve_amount` on `morphogen_growth_3d`.** Okuda Appendix A: the morphogen is an AMOUNT, so a
  growing cell must dilute its concentration. Turning it off silently creates mass every step.
- **The frame count and the analysis stride.** Campaign-wide, so a metric measured on one run means
  the same as on another.
- **`divide_3d` + `shape_energy_3d:monolayer` together** — a filled 2×2 table says the pair is a
  substrate bug (see *What is ruled out*).

## What is still missing

**A protrusion — but the coupling has moved.** Round 1 (2026-08-05) produced the strongest
chemistry-shape coupling the campaign has measured: `corr_act_rad_peak` **0.739** on `_keep/r001_02`,
with 0.710 and 0.709 on two siblings, against `coral_gate`'s 0.435. All three held `valid_frac` 1.0,
`act_cv_peak` ~4.97, `n_tubes_peak` 1 and `n_tips` 4–6. That is the campaign's own question — does the
pattern grip the shape — answered better than ever before, and those three runs are now the top of the
parent pool. The loop itself recorded none of it: a bad parent's four children exited in under a
minute, the launcher stopped waiting, and every run was measured as nothing. The numbers came off the
disk afterwards.

The best `protr_peak` on an admissible specimen is still **1.169** (`refute_coral_nocons`),
with `coral_gate` at 1.145 — and a tube needs something well above 1.3, so the gap is not a matter of
tuning a run that nearly worked. Nothing has nearly worked. Ten of eleven runs in round 1 sat on
exactly 1.022 and all of round 2 on 1.001: rails, not results. `n_tubes` is 0 everywhere. Okuda's
tubes and branches are the target and nothing has produced one. That is the open problem, and it is
what a round is worth spending on.

<!-- VirialStress -- append below; the driver merges this into campaign/analysis.md -->

# virial_stress (VirialStress)

Read `physics/mechanics/stress.py` (the whole 58-line module, one class), its base
`core/step.py:SimulationStep` (the quasistatic contract, `state.set`), and the real math it
delegates to: `PairwisePotential.virial_pressure` in `physics/mechanics/potentials.py`, plus the
`neighbor_sum` / `safe_norm` / `safe_divide` helpers in `core/geometry.py` and `core/ad_utils.py`.
Cross-read the paper's Methods "Mechanical Stress" paragraph (p. 16) and the main-text usage
(p. 7-8 "Mechanical Control of Cell Proliferation", Fig. 4 / S6).

`VirialStress` itself is a thin quasistatic sensing step: `state.set('stress',
potential.virial_pressure(state))`. It writes a transient per-cell `stress` field
(`heritable=False`), moves nothing, and carries no numeric knob -- its only field is the
`potential`, and the whole point is that a traced potential param (e.g. `epsilon`) stays
optimizable *through* the written stress.

**The surprise (headline).** SOURCE != PAPER. The code computes the textbook Irving-Kirkwood
virial pressure: `p_i = -(1/(2 d V_i)) sum_j r_ij (dU/dr)`, radial projection, normalized by the
cell's own d-ball volume and dimension, with a 1/2 bond split and a minus sign so compression is
positive. The paper (p. 16) instead defines `sigma_i = sum_j [F_ij,x*sgn(dx) + F_ij,y*sgn(dy)]`
-- a *component-wise sign* (taxicab) weighting, no volume/dimension normalization, no 1/2 split,
2D-only. These are genuinely different quantities (differently weighted, and off by ~2 d V_i per
cell). Per the loop rule I recorded the code's form as authoritative and the contradiction in
`surprises:`. Other easy-to-miss details: dU/dr is autodiff of the *cutoff-multiplied* energy;
`safe_norm`/`safe_divide` keep the r=0 diagonal and V_i=0 dead slots finite under debug_nans;
`neighbor_sum` masks self- and dead-pairs; no in-library step *reads* `stress` (its consumer is a
gene-network input in the examples).

**Did NOT establish:** whether this hardened reimplementation's `virial_pressure` reproduces the
*original* jax-morph numeric stress trajectory. I confirmed the FORM disagrees with the paper text
but did not run the oracle to diff a reference stress signal -- that is a validation-stage
question, and I flagged it rather than assuming equivalence. I also did not trace exactly which
example notebook's gene-input node consumes `stress`, only that nothing inside the package does.

---

## Normalizer: verdict `new` -> contract `mechanosense`

**Verdict: `new`** — a standalone quasistatic MECHANOSENSOR. VirialStress reduces the same
pairwise potential and same live-non-self neighbour sum as `attraction_repulsion`, but contracts
it to a per-cell SCALAR observable (the Irving-Kirkwood virial pressure, normalised by `2 d V_i`)
that it writes to a transient `stress` field and that MOVES NOTHING. No registered contract
exposes a per-cell mechanical load as a pure-sensing readout for downstream mechanotransduction:
`active_stress`/`active_force` GENERATE stress/force into the MPM substep, `sense` reads a
diffusible field and steers heading, `aggregate` reduces children onto a parent. Contract:
`mechanosense`, `lateral`/`mechanics` (same taxonomy slot as gravity / mpm_anchor / mpm_spin),
set `cell`; reads position/radius/alive + the potential's coupling field, writes `stress`.

**Single strongest argument against it.** Plexus arguably ALREADY has mechanosensing —
`cell_grow` carries `stress_gain` (`mechano_inhibition`: "growth slows in deformed tissue"), which
is exactly the paper's "stress inhibits proliferation." If the language can already gate a cell's
fate on its local mechanical load, then VirialStress is not new vocabulary but merely the extracted
"sensor half" of a capability Plexus expresses — pushing toward `out_of_scope` (a redundant
intermediate) or a `refinement` of `attraction_repulsion`, whose base literally computes
`virial_pressure` right beside `forces()` and could simply also emit it. The rebuttal I stand on:
`cell_grow` reads the MPM CONTINUUM deformation gradient F (not a pairwise virial) and FUSES
sense+respond in one op, so the mechano-SENSE never exists as a reusable, first-class `stress`
field that any other consumer (a gene network, `cell_divide`, a differentiation switch) can read —
which is precisely the decoupled contract VirialStress mints and the promoted language lacks. But
this is the alternative I had to defeat, not a free win: if a later entry shows `cell_grow`'s
deformation readout and the virial pressure are interchangeable mechanical-load signals, then
`mechanosense` and that fused reading should be reconciled as one contract with two
implementations rather than left as separate vocabulary.

---

## Implementer: operator `mechanosense`

Wrote `src/plexus/operators/candidates/jax_morph_virial_stress.py` and test
`tests/test_jax_morph_virial_stress.py`. Both import/register clean; test passes 7/7.
`status: implemented`.

**Faithful, not fitted.** Modelled the shape on the `adhere:*` torch ports (same
`safe_norm`/`safe_divide`/`_smooth_cutoff`/`_compact_repulsion`) and the `grow_radius`
read-block-or-buffer recipe. Key decisions:

- **Pure sensing = `EMIT=None`, `forward` returns `{}`.** The operator writes the per-cell `stress`
  scalar in place (a schema block if the set declares one, else a lazily-provisioned buffer) and
  moves nothing. Writing a `stress` state block mutates `Level.state`, which the engine's frame-0
  integration-invariant guard (`engine.py:709-723`) flags -- VirialStress is exactly the
  DERIVED-READOUT category that guard exempts, so the class sets `MAY_MUTATE_INTEGRATED_STATE =
  True`. A test asserts pos/vel are byte-identical after the call (it really moves nothing).
- **Pair law as a plug-in (`potential:` selector), reduced -- not a force.** `dU/dr` is taken by
  autodiff of the FULL pair energy (`torch.autograd.grad(U.sum(), r)`, elementwise) so the
  smooth-cutoff switch term rides in it, exactly as `jax.grad(pair_energy)` does. Default `morse`
  (paper mechanics, eps 3.0); also soft_sphere / hertzian / harmonic / lennard_jones, each with its
  own knobs. The reduction (`r_ij . dU/dr` over live non-self j) is the rank-0 SENSED scalar, versus
  `adhere`'s rank-1 `-grad_i U` that MOVES cells -- the whole reason this is `new`, not a widening of
  `adhere`.
- **The three biology-carrying conventions are in and tested:** minus sign (compression-positive;
  the Morse/harmonic adhesive tail correctly reads tension-negative in the smoke run), 1/(2d)
  Irving-Kirkwood + 1/2 bond split, and V_i = the cell's OWN d-ball volume (2r / pi r^2 / 4/3 pi r^3,
  branched on `d = pos.shape[-1]`). `safe_*` keep the r=0 diagonal and a dead cell's V_i=0 finite.
- Differentiable through the coupling: with a per-cell `epsilon_field`, `d stress / d epsilon` is
  finite/correct (the source's "optimizable through the written stress"); inert under the engine's
  `no_grad` rollout.

**Property test (no oracle numbers).** Anchor = compression-positive sign + the analytic
soft_sphere pressure `p = eps r (1 - r/sigma)/(sigma 2 d V)` (derived BY HAND from `dU/dr = -eps(1 -
r/sigma)/sigma`): two overlapping cells read exactly that, strictly > 0. Plus beyond-contact limit
(r>=sigma -> 0), translation/reflection + identical-cell symmetry, size normalization (bigger V ->
smaller pressure at equal overlap), moves-nothing, dead-cell masking, and the no-block buffer path.

**Left for the differ (did NOT establish):** no oracle run yet -- the FORM and calculus-level
properties are verified, but there is no run-vs-reference stress number. One simplification to watch:
I differentiate `dU/dr` against a detached leaf clone of `r`, so the *value* is exact and the
*epsilon*-gradient is preserved, but the position-gradient path through `dU/dr` is dropped (values --
what the differ compares -- are unaffected).

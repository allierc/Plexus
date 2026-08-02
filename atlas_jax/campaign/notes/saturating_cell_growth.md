<!-- SaturatingCellGrowth -- append below; the driver merges this into campaign/analysis.md -->

## saturating_cell_growth (EXCAVATOR, status -> inspected)

**Read:** `jax_morph/physics/growth.py:23` (whole class), its base `SimulationStep`/`StepType`
(`core/step.py`), the state/delta machinery (`core/state.py`: RADIUS, StateFieldSpec, `deltas`,
`_accumulate_dynamic`'s alive-masking), and `tests/physics/test_growth.py`. Also read the SI
"Cell Growth" section of the paper (p. 14) and the main-text sketch (p. 2).

**What it does to the state:** a DYNAMIC step that reads `radius` + per-cell `growth_rate` and
returns a sparse delta `radius += dr`, where `dr = (R - r)(1 - exp(-k*dt/R))` is the *exact* flow
of the von Bertalanffy ODE `dr/dt = k(1 - r/R)`. Growth is fastest at small r, halts at
`max_radius`, bounded in [r0, R] for any dt.

**Biggest surprise (source vs paper contradiction, rule 5):** the PAPER says growth is
*constant-rate with a hard clamp*, `R_i(t+dt) = min(R_i(t) + dR, R_max)`, and explicitly notes the
`min` must be smoothed to be differentiable. The CODE does something else entirely: a smooth
exponential relaxation whose rate decays toward the target -- no fixed `dR`, no `min`, inherently
differentiable. Same endpoint (<= R_max), different trajectory (paper: linear-then-flat; code:
exponential asymptote). Recorded the code as the mechanism and both readings in `equations:`.

**Other things a reimplementer gets wrong:** (1) it returns a `deltas(...)` increment, not an
absolute radius -- return `min(r+dr,Rmax)` and you break the dynamic-phase accumulation contract;
(2) the increment is the *exact* exponential, chosen over forward Euler which diverges once
`dt*k/R > 2` (the stability test drives `dt*k/R = 3`); (3) `max_radius` normalizes the exponent
`-k*dt/R`, so the time constant is `R/k`, not `1/k`; (4) the rate `k` is a *state field*
(`growth_rate`, heritable, default 0), NOT a constructor param -- that's the whole design point
(an upstream controller writes it per-cell and gradients flow back through it); (5) dead-cell
masking lives in the model (`_accumulate_dynamic`), not the step -- ported standalone, dead cells
would grow.

**Not established / uncertainty for the next role:**
- I did NOT run the oracle to numerically confirm the trajectory against the smoke reference; the
  `evidence:` block is left null (that is a downstream role's job). The equations are from static
  reading only.
- I could NOT determine whether the von Bertalanffy law is a faithful re-derivation of the
  ORIGINAL Deshpande `jax-morph` growth code or a fresh choice by whoever refactored this repo into
  the typed-step architecture: the original repo's `growth.py` is not in this tree to diff against,
  so the paper's `min(R+dR, Rmax)` is my only "other" reference. The contradiction is real
  regardless (code disagrees with the paper text), but its *provenance* (deliberate improvement vs
  drift) is open.
- The paper writes `dR` with no cell subscript (a single global increment); the code's `k` is
  per-cell. I read this as an intentional elaboration, but the paper does not say so.
- verdict/contract left null on purpose (normalizer's call, per the loop rules).

## saturating_cell_growth (NORMALIZER, status -> normalized)

**Verdict: `refinement` of `cell_grow`** (`of: cell_grow`, `implementation_of: cell_grow`). Same
biology as the registered growth primitive -- a cell grows toward a maximum size by a *saturating*
law -- so this is not `new` (a second growth contract would inflate the yield), and cell_grow is
unambiguously the closest contract (the only other growth-family op, `cell_divide`, is topology,
not size). But it does not fit cell_grow's *registered* signature, so it is not a clean `alias`
either: the signature must **widen** on four fields -- `kind` (structural, returns `{}` ->
delta-emitting `field` on `radius`, which breaks the invariant that growth adds nothing to the
dynamic-phase accumulation), `writes` (`grow_V` + child particle occ -> a `radius` delta on a
point-cell, dropping the required `mpm_particle` child), `reads`/params (constant `rate` param ->
heritable per-cell state field `growth_rate`, the differentiable-control design point), and the
growth `law` (logistic -> von Bertalanffy exact flow; additive, harmless).

**Strongest argument AGAINST (why this could instead be a plain `alias` + implementation_of):**
cell_grow's own docstring *already* declares the discretisation swappable and the growth law the
invariant -- "swap the discretisation ... and the growth LAW stays identical." Under that reading a
scalar radius on a soft-sphere cell is exactly the anticipated "another discretisation," so nothing
needs to widen; the structural-vs-delta and MPM-vs-scalar differences are realization plumbing
*beneath* the contract, and calling it a refinement over-costs a change the contract's authors
already sanctioned -- inviting the very "refinement hides a breaking change" failure record.py
warns about, in reverse (flagging a break that isn't one). **Why I still chose refinement:** the
frozen baseline record.py compares against is the *promoted language* -- `plexus.operators` and
nothing else -- i.e. the registered signature (`kind=structural`, `EMIT=None`, MPM-child
realization), not the docstring's aspiration. That registered signature genuinely cannot emit a
differentiable `radius` delta without an MPM child; admitting this mechanism *does* mutate the
frozen contract, and a change to the baseline is a refinement by definition -- one a downstream
user of structural-only growth must be shown the cost of. A downstream role that reads the source
and finds the registry can already host a delta-emitting growth realization *without* editing
cell_grow's signature should downgrade this to `alias`.

## saturating_cell_growth (IMPLEMENTER, status -> implemented)

Wrote `src/plexus/operators/candidates/jax_morph_saturating_cell_growth.py` (anti-chamber) and
`tests/test_jax_morph_saturating_cell_growth.py` (9 passing). `module`/`test`/`status` updated.

**The refinement is REJECTED by the registry -- and that settles the normalizer's refinement-vs-alias
debate toward refinement.** The normalizer left the door open: "a downstream role that finds the
registry can already host a delta-emitting growth realization *without* editing cell_grow's signature
should downgrade this to `alias`." It cannot. `@register_operator("cell_grow", implementation=...,
kind="lateral"|"field")` raises at import next to the shipped structural `cell_grow`:
`operator 'cell_grow' implementation 'saturating_radius' has kind 'lateral', but the contract's kind
is 'structural'; implementations may differ only in numerics` (registry.py:131). So this mechanism
CANNOT be a silent second implementation of the frozen contract -- admitting it requires deliberately
widening cell_grow's kind, i.e. a language change a downstream user must be shown the cost of. That is
exactly the "refinement hides a breaking change" the ledger exists to catch, here enforced by the
registry itself, not merely argued. The candidate therefore registers under the distinct name
**`grow_radius`** (family=growth, set=cell); unifying it with cell_grow is the curator's call.

**`kind: field` is read as "delta-emitting"; the faithful concrete kind is `lateral`.** Plexus's
`field` kind is a FIELD's own grid self-dynamics (diffuse/decay), `EMIT=None`, `forward()` returns
`{}` -- it CANNOT emit a per-cell `radius` delta. The operator is a per-cell autonomous ODE with no
neighbour coupling and no field, which is precisely the shipped `attractor_flow`/`signal` shape
(`dx/dt=f(x)`, kind=lateral). So I registered `kind=lateral` and recorded the field->lateral reading
as a surprise. (The entry's contract.kind stays `field` -- the normalizer's artefact; the surprise +
this note flag the concrete kind for the curator rather than my silently rewriting the verdict.)

**Routing (faithful to the source's dynamic-delta contract).** The source returns the EXACT increment
`dr = (R-r)(1-exp(-k*dt/R))` as a dt-scaled delta the Model adds. Plexus integrates a first-order
block as `x += dt*delta`, so -- identical to the `regulate:neural_ode` sibling's self-solved increment
-- I return the mean RATE `delta = dr/dt`; the engine's `dt*` recovers the exact endpoint (dt cancels,
not a second integration). `EMIT="velocity"`, class `INTEGRAND="radius"` (a NON-coordinate first-order
block, so `_resolve_emit` never constrains the cell's spatial order) + instance-`INTEGRAND` routing to
a configurable size block; the delta lands in the `radius` accumulator, summed with any co-writer.
`growth_rate` is read as a per-cell heritable STATE (block or buffer), falling back to a uniform `rate`
param (default 0 -> k=0 -> byte no-op); the sole "target" knob is `max_radius` (guarded > 0). Pure
torch (exp/mul) -> `DIFFERENTIABLE=True`, so gradients flow through r, k, and the optimizable R -- the
whole point of keeping the rate in state.

**Source wins (rule 5):** translated the CODE (von Bertalanffy exact flow, no clamp), not the paper's
prose (`min(R+dR, Rmax)`, constant-rate), as the excavator/normalizer established. Not hard-coded to
the oracle -- the tests assert facts about the ODE, never a fitted number.

Tests (all reference-free): (a) one step reproduces the ODE's KNOWN analytic solution
`r(dt)=R-(R-r0)exp(-k dt/R)` through `radius += dt*delta` to 1e-6; (b) UNCONDITIONAL STABILITY -- even
dt=1e3 never overshoots R while a forward-Euler step with the same dt blows past it (the exact-flow
headline); (c) fixed point at R (dr=0); (d) sign symmetry about R (below grows, above relaxes down);
(e) k=0 exact no-op; (f) 400-step asymptote to R, approached from below, no overshoot; (g) dormant
(occ=0) cells get zero; (h) the end-to-end path through the real engine `_integrate` lands on the
analytic endpoint (the `_delta_blocks` radius routing composes); (i) registration/routing
(kind=lateral, EMIT=velocity, INTEGRAND=radius, MAPS=[], writes=[radius]).

Left for the validator/curator: the differential run against the oracle. jax is deliberately absent
from the Plexus env, and no oracle/smoke scenario is known to exercise SaturatingCellGrowth yet, so
there is no paired trajectory to diff -- `evidence`/`status: validated` untouched. Curator decision:
whether `grow_radius` promotes as a widened second implementation of `cell_grow` (a kind-widening of
the growth contract) or as its own contract.

## saturating_cell_growth (DIFFER, status -> validated)

**PASS, exactly.** `value = max(D_max_A, D_max_B) = 0.0` vs `threshold = 1.0e-5`. The Plexus
`grow_radius` operator and the jax-morph `SaturatingCellGrowth` reference produce **bit-identical
float32 radius trajectories** over all 20 macro-steps in both scenarios (`D_max_A = 0.0`,
`D_max_B = 0.0`). Reproduced from scratch in the Plexus torch env.

**Metric.** `D_max` = max absolute per-cell radius deviation over every recorded frame (t=0..20,
dt=2.0) and every live cell, in radius units, reported as `max(D_max_A, D_max_B)`:
- `D_max_A` -- the 4-cell engine run of `config/atlas/saturating_cell_growth.yaml` exactly as
  `run_spec` runs it (r0=0.30, k=0.40, R=0.6) vs oracle scenario A.
- `D_max_B` -- a 6x6 (r0 x k) 36-cell grid rolled through the real engine `radius += dt*delta` vs
  oracle scenario B; spans k=0 no-op, small-k near-linear, large-k saturation (Euler would
  overshoot at dt=2), r0==R, and r0>R (relaxation DOWN to R -- the no-clamp claim).

**Why isolated.** Growth is run ALONE (no relaxation, no division) so both sides keep the same
cells in the same array slots and align cell-for-cell, frame-for-frame. Division's hazard
p=1-exp(-division_rate*dt) never reads radius (`division.py _dist`), so division -- not growth --
is what makes the anchor's counts diverge (124 vs 82); isolating growth is the faithful test of
THIS operator's contract, the per-cell radius ODE. The KNOWN OPEN 124-vs-82 discrepancy is NOT a
growth defect and stays with division, unexplained by this operator.

**Threshold justification (fixed before the run).** 1e-5 sits ~100x above float32 round-off (the
reference's own float32-vs-float64-analytic error is 9.5e-8) yet ~5e4x below the negative-control
signal (0.55), so it separates bit-level agreement from any real disagreement -- wrong law,
dropped dt-scaling, Euler overshoot, or a min-clamp.

**Discriminating controls (all clean).** Negative control (drop the `/dt` mean-rate convention,
`radius += dt*dr`) scores 0.55, ~5e4x above threshold -- the metric provably catches a wiring
defect precisely because dt=2 makes the convention observable. `frame0_is_ic_residual = 0.0`
(frame 0 is the pure seeded IC), `k0_noop_drift = 0.0`, `aboveR_last_min = 0.6 = R` (no clamp).
Acted ledger valid: `grow_radius` acted 13/20 calls (the later 7 emit a sub-float delta once
radius has reached R -- correct saturation), `seed_state` 1/1.

**Paths.**
- oracle run: `atlas_jax_morph/_oracle/runs/diff_saturating_cell_growth/` (reference.npz, summary.json)
- oracle script: `atlas_jax_morph/_oracle/scripts/saturating_cell_growth.py`
- engine spec A: `config/atlas/saturating_cell_growth.yaml` -> evidence `log/atlas_jax/saturating_cell_growth/`
- grid spec B: `atlas_jax_morph/saturating_cell_growth_gridB.yaml`
- diff driver: `atlas_jax_morph/diff_saturating_cell_growth.py` -> `log/atlas_jax/saturating_cell_growth/diff.json`

**Verdict unchanged.** The differential does not touch the `refinement`-of-`cell_grow` verdict --
either reading is still cell growth toward a maximum size. What it establishes is that the
`grow_radius` contract, as implemented and as the engine runs it, reproduces the source's
saturating von-Bertalanffy per-cell radius flow to the last bit. `status: validated`.

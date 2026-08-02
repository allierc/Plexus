<!-- step type: quasistatic / dynamic / discrete -- append below; the driver merges this into campaign/analysis.md -->

## step type: quasistatic / dynamic / discrete

**Read:** `jax_morph/core/step.py` in full — the `StepType` StrEnum (L50), the `SimulationStep`
`__call__` return contract (L126), `Model._validate`'s per-type write-conflict policy (L394),
`_accumulate_dynamic` (L342), and `Model._run` / `Model.__call__`'s A/B/C phase skeleton (L485,
L519). Also `core/state.py` for `deltas`/`set`/`update` and the field-spec scope model, plus the
`concepts.md` and `core-abstractions.md` guides.

**What it is:** not an operator but a *meta-tag*. Each step declares a class var `step_type` that
fixes (a) which macro-step phase it runs in and (b) how the Model reads its return value.
Quasistatic + discrete return a full state (pipeline/sequential); dynamic returns a sparse dt-scaled
delta that the Model evaluates for every dynamic writer at the SAME post-quasistatic state, then
sums, alive-masks, and applies once. A macro-step is the fixed Lie-Trotter split
`disc o dyn o qs`, then `t += dt`.

**What surprised me (the payload):** the paper (Deshpande 2025, p. 14 "FORWARD SIMULATION") has
NO such taxonomy. It describes the sim as "any subset or combination of the steps detailed below" —
a flat customizable sequence of full-state ops (growth, relaxation, division, diffusion), "each
simulation timestep [consisting] of one cell division." The words quasistatic / dynamic / discrete /
Lie-Trotter / hybrid / macro-step appear ZERO times in the paper. The three-phase time-scale
taxonomy, the operator split, and the dt-scaled sparse-delta contract are a library
re-architecture. Per rule 5, source wins — recorded in `surprises:`. Other traps a reimplementer
hits: dynamic is order-independent (all at same state, summed) while qs/disc are Gauss-Seidel;
dt-scaling is baked into each dynamic step, not the accumulator; discrete steps are exempt from
write-conflict checks (last-writer-wins, silent); phase dispatch is `is`-identity on the StrEnum,
so a raw string `'dynamic'` runs in no phase silently.

**What I did NOT establish:** (1) I did not run the oracle — no differential evidence, and I did
not confirm which concrete steps in the running `smoke` model carry which step_type (I read the
contract, not a live pipeline census). (2) I did not verify the paper's *original* jax-morph GitHub
code (github.com/fmottes/jax-morph) to confirm it truly lacks any implicit qs/dyn/disc split — my
"paper has no taxonomy" claim is from the paper TEXT only; the original source could encode the
distinction informally. (3) The growth example on p. 15 (`R_i(t+dt)=min(R_i+dR, Rmax)`) adds a
FIXED per-step increment, not an explicitly dt-scaled rate — whether the library's "dynamic dt
increment" faithfully reproduces that is a per-step question I left for the growth entry.

**Normalizer verdict: `out_of_scope`.** `step_type` is a class-level classification TAG, not a
forward operator over sets and fields — it declares no physics of its own and only fixes which of
three fixed macro-step phases a step runs in, how the Model reads its return value, and the
per-type write-conflict policy. That is engine/scheduling machinery composing the forward
operators, the same call the campaign already made for `StochasticStep` (order 5, differentiability
= engine concern). The contract fields are a validator formality (R6/R7 demand a typed,
writes-non-empty contract); the load-bearing observation is that step_type is a genuine
TIME-SCALE / integration-phase axis Plexus's IR does not carry — Plexus's own `kind` taxonomy is a
data-flow-SHAPE axis, orthogonal to it — so the axis is a gap in the meta-layer, not a missing
operator. **Strongest argument AGAINST:** step_type is not inert plumbing — it materially changes
what the simulation *computes*. The identical physics tagged `dynamic` (all writers evaluated at
one frozen post-quasistatic state, summed once — Jacobi) versus `quasistatic` (each step slaved to
the current updated state — Gauss-Seidel) yield *different trajectories*; a mis-tag silently runs a
step in no phase at all. Because the tag alters the dynamics and not just the bookkeeping, one
could argue it is modeling content deserving a first-class place in the algebra (an operator
attribute, or even a new time-scale `kind`), not dismissal as numerics. I reject this because
"which integrator composes these operators, in what order, reading which state" is the definition
of an engine/IR concern: it is a property *of* operators (how they compose in time), not an
operator that acts on state — and Plexus already locates operator-classification in its `kind`
meta-layer, exactly where this axis belongs if adopted, never in the forward vocabulary the atlas
counts. Promoting it as `new` would inflate the yield with framework machinery, which is precisely
the measurement out_of_scope protects.

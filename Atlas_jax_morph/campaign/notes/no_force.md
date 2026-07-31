<!-- NoForce -- append below; the driver merges this into campaign/analysis.md -->

# NoForce (potential)

Read the whole `potentials.py` module and all three consumers (`dynamics.py`,
`relaxation.py`) plus the `Potential` base class.

**What it does.** `NoForce` is a direct `Potential` subclass whose `total_energy(positions,
state)` is a bare `jnp.asarray(0.0)` and whose `forces(state)` is *overridden* to return
`jnp.zeros_like(state.position)` directly — bypassing the base class's autodiff-of-energy path.
Net effect on state: nothing moves. It is a null/identity element for the mechanics slot, not a
physical interaction.

**Where it actually lives.** Nobody constructs it by name in the physics. It is what
`potential=None` resolves to inside `BrownianDynamics`, `ActiveBrownianDynamics2D`,
`MechanicalRelaxation`, and `relax_equilibrium` (`potential if potential is not None else
NoForce()`). So it materializes three model idioms: a free Brownian gas, a self-propelled active
gas whose alignment coupling lives in a separate step (Vicsek base), and a no-op relaxation.

**Surprised me.** (1) It is *not in the paper at all* — the paper's cells always interact via a
Morse potential (p. 3, Fig. 1d). NoForce is a pure library affordance, so it is a genuine
"the-code-adds-a-null-element the-paper-never-mentions" case, which I flagged in `surprises`
rather than as a paper/code contradiction. (2) The class docstring advertises "skipping the dense
O(N^2) pairwise sum," but NoForce is a *direct* `Potential`, not a `PairwisePotential` — its own
energy has no sum; the O(N^2) line is general motivation, and the real cost the override avoids is
the grad trace. (3) Under `MechanicalRelaxation` the no-op happens via the *general* FIRE path
(residual 0 <= f_tol at iter 0, loop body never runs), not a special branch — even though
`relax_equilibrium` does carry an explicit `if potential is None` short-circuit that the step
doesn't use (it stores a concrete `NoForce()`).

**Could not determine / did not establish.** I did not run the oracle — a zero potential has no
trajectory signature to diff against, so I left `evidence` null and treated this as a
read-at-source entry. I did not chase whether any *notebook/example* (outside `jax_morph/`) or the
GRN/control side ever passes `NoForce` explicitly; my "constructed only via None" claim is scoped
to the physics package + guides, which is where I looked. I also did not verify the exact default
dtype of the `0.0` scalar under the library's runtime config beyond noting the test suite enables
x64; `forces` sidesteps this by using `zeros_like(position)`.

---

## NORMALIZER — verdict `out_of_scope`

NoForce is the null / identity element of the mechanical-interaction slot — the object
`potential=None` resolves to inside `BrownianDynamics`, `ActiveBrownianDynamics2D`,
`MechanicalRelaxation`, `relax_equilibrium`. Zero energy, zero force, no params, reads no field
values, writes nothing persistent, no paper counterpart. The decisive test: in the Plexus algebra
"no interaction" is the *empty composition* (include zero interaction operators) — the identity of
operator composition, which you get for free by composing nothing and never register as an operator.
jax-morph only materializes NoForce because its `Potential` slot must be filled by *some* object; a
Plexus model expresses the same free/non-interacting regime by omitting the interaction operator, so
there is no forward mechanism to normalize and no vocabulary gap. Not `alias`/`refinement` (there is
no registered no-op/identity contract to point at, and widening an interaction contract to admit
"reads nothing, writes zero, no map" would delete the pairwise force that IS that contract); not
`new` (a do-nothing operator would inflate the yield with content-free plumbing). Contract block
carries a formal `writes: [position]` (an identically-zero force-slot occupant) to clear R7, exactly
as the StochasticStep out_of_scope entry carries a formal `writes: [trace]`.

**Strongest argument AGAINST this verdict.** In an operator *algebra* the identity element is a
first-class member, not something to sweep out of scope. NoForce is type-level interchangeable with
Morse — same `Potential` protocol, consumed by the same integrator steps — so it could instead be
recorded as the degenerate `implementation_of` the pairwise-interaction contract (the zero member of
the Morse/SoftSphere/Hertzian family), a framing that captures a real modeling affordance:
jax-morph deliberately represents "no coupling" as a value of the same type as "Morse coupling," the
base case for active-matter/flocking models whose coupling lives in a separate alignment step. I
reject it because the atlas counts *biological contracts*, not type slots — NoForce reads nothing,
writes nothing real, has no map and no parameters, and is never built in the paper (it is the
`potential=None` sentinel Plexus realizes by leaving the operator out). But if the interaction
contract's registered signature ever made an explicit nullable-potential field load-bearing on a
wrapping step, this flips from `out_of_scope` to `implementation_of`.

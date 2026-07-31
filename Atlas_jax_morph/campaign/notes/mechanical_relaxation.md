<!-- MechanicalRelaxation -- append below; the driver merges this into campaign/analysis.md -->

# MechanicalRelaxation (order 19)

**Read:** `physics/mechanics/relaxation.py` in full -- the `MechanicalRelaxation` step (L221), the
`relax_equilibrium` free fn (L189), and the three internals it composes: `_fire_to_tol` (forward
FIRE loop), `_rigid_body_modes`/`_project` (the physical-subspace projector), and the
`jax.custom_vjp` triple `_relax_impl`/`_relax_fwd`/`_relax_bwd` (the implicit-diff backward). Also
read the `Potential`/`NoForce` base (`potentials.py`) and `SimulationStep`/`StepType` (`core/step.py`)
to pin the read/write contract and the QUASISTATIC phase semantics.

**Biggest surprise -- a clean paper/code contradiction.** The paper (p. 14, "Mechanical Interactions")
says relaxation is *"gradient descent energy minimization of the Morse potential for a fixed number of
steps (except in the case where we learn cell adhesion)."* The source does none of those three things
as written: (1) the minimizer is **FIRE** (momentum + adaptive dt), not plain gradient descent; (2) it
runs to a **force tolerance** `f_tol` (a real equilibrium), with `max_steps` only a fallback bound, not
a fixed step count; (3) the gradient is the **implicit-function-theorem** sensitivity of the
equilibrium (`jax.custom_vjp`), not backprop through the descent. The paper's own aside "(except ...
cell adhesion)" is precisely the case where fixed-step solver-path gradients fail -- which the code's
implicit-diff generalizes away. Recorded in `surprises` with both readings; SOURCE WINS.

**Other reimplementer traps I flagged:** the rigid-body null modes (dead DOFs + global translation +
global rotation of the alive cells) are projected out of the adjoint and carry *zero* gradient by
design (a translation/rotation-sensitive loss just gets none); `ridge` is backward-only and lives on
the *projected* Hessian; the FIRE schedule constants are hardcoded (Bitzek-2006 values), not knobs;
the adjoint also differentiates wrt the incoming **state** (radius -> sigma), because
`params = (potential, state)` rides the custom_vjp as one pytree; and the declared `state_reads`
omits radius/alive/displacement since those are base fields.

**Did NOT establish / open uncertainties:**
- No oracle run -- I did not confirm numerically that the smoke trajectory exercises this step, nor
  that FIRE converges within `max_steps=500` on the reference cluster (the `RuntimeWarning` path is
  untested by me). `evidence` left null for the verifier.
- I did not trace whether the reference `smoke` config uses this quasistatic step at all vs the
  `BrownianDynamics` (kT=0) relaxation route -- both can relax positions, and which one the paper's
  experiments used per-figure is unresolved here.
- I asserted the FIRE constants match Bitzek 2006 from the values/names; I did not cross-check against
  a citation in the paper (the paper names neither FIRE nor these constants).
- The "(except ... cell adhesion)" clause: I inferred it points at the solver-path-gradient problem,
  but the paper does not spell out what alternative they used in that case -- an open question for
  whoever excavates the adhesion-learning experiment.

**Normalization (NORMALIZER role).** Verdict: **`new` -> contract `relax`** (kind `lateral`, family
`mechanics`): drive cell positions to the mechanical equilibrium (force balance `grad_x U = 0`) of a
supplied interaction -- a QUASISTATIC map to the force-balanced configuration, the tissue-mechanics
stance that mechanics equilibrates fast relative to growth so the cluster's shape at each morphogenetic
timepoint IS the equilibrium. Stripping the FIRE/implicit-diff numerics and the pluggable `potential`
(a separate `attraction_repulsion`/Morse contract) leaves the equilibration OPERATION, and no
registered contract expresses it: all 42 are single-application Euler maps (emit a velocity/force,
engine steps once); none runs to a fixed point. This is the quasistatic dual of `brownian_dynamics ->
agitate`. **Single strongest argument AGAINST:** "relax to equilibrium" may be a *driver/solver* over
an operator the language already has, not a new operator -- the equilibrium of `attraction_repulsion`
is just the fixed point of iterating its overdamped motion, and every knob `relax` exposes
(`max_steps`, `f_tol`, `ridge`) is a numerical tolerance with no physical content (contrast `agitate`,
which owns the physical `kT`/`gamma`), which reads as `out_of_scope` plumbing. It loses because the
fixed Lie-Trotter split applies each operator exactly ONCE per macro-step, so the omega-limit of
iterated composition is NOT expressible by composing registered operators (unlike NoForce = the free
empty composition); `relax` also performs a real forward transformation (position -> equilibrium, not
a zero no-op) and is potential-agnostic (the operation, not the force) -- but a verifier could still
land this at `out_of_scope` if they judge a convergence-gated solver to carry no biology of its own,
so this is the live fault line for the entry.

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

---

**Implementation (IMPLEMENTER role).** Wrote `src/plexus/operators/candidates/jax_morph_mechanical_relaxation.py`
(the anti-chamber) and `tests/test_jax_morph_mechanical_relaxation.py`; `status: implemented`.
Torch, not JAX. Registered `@register_operator("relax", family="mechanics", set="cell", kind="lateral")`,
`EMIT="velocity"`, `DIFFERENTIABLE=False`. Imports + registers + 7 property tests pass.

Design decisions (and the impedance mismatches they resolve):
- **Quasistatic overwrite through the delta contract.** The reference step OVERWRITES position with
  the equilibrium `x*` (`state.set('position', x*)`). Plexus has no quasistatic phase and the
  integration invariant forbids an operator writing `pos` directly, so I express the overwrite as a
  velocity: the engine integrates `pos += dt*v`, so emitting `v = (x* - x0)/dt` lands `pos` exactly
  on `x*` in ONE macro-step, for ANY `dt`. That `1/dt` factor is the SIGNATURE of a quasistatic
  step -- the exact dual of `agitate`'s `1/sqrt(dt)` Wiener emit. A test drives three `dt` values and
  confirms the landed configuration is identical (dt-independence).
- **Potential-agnostic solver.** `relax` is the SOLVER, not the force law: a `potential:` param
  selects the energy family it relaxes (`soft_sphere` default / `hertzian` / `harmonic` / `morse` /
  `lennard_jones` / `none`=NoForce no-op). I ported the source's `_compact_repulsion` / `_smooth_cutoff`
  / `safe_divide` / `safe_norm` and the five sigma-relative (`sigma = r_i + r_j`) pair energies, and
  take the force by autodiff of the energy -- so `relax` composes with any of them, orthogonal to
  which interaction. Smoke-ran all five: the adhesive ones (morse/LJ/harmonic) settle to exactly the
  well minimum at contact (sep 1.000), the repulsive ones to at least contact.
- **FIRE, faithfully.** `_fire_to_tol` reproduces the Bitzek-2006 schedule verbatim (carried velocity
  mixed toward the force direction; global power/norm reductions; dt grows after `n_min` downhill
  steps, resets with the velocity on an uphill step); stops at `|grad U|_inf <= f_tol`, `max_steps` a
  fallback. Forward-only: `_force` evaluates `-grad U` on a fresh detached leaf under a local
  `enable_grad` (the engine generates under `no_grad`), so the solver path is never in any outer graph
  -- the source's `custom_vjp` stance ("the answer depends only on `x*`, not how FIRE got there").

What I did NOT reproduce (honest gaps, deliberately):
- **The implicit-diff backward.** `DIFFERENTIABLE=False`. The source's distinctive gradient -- the IFT
  sensitivity `dx*/dp = -H^{-1} d(grad U)/dp`, CG-solved on the physical subspace with the rigid-body
  gauge modes projected out -- is NOT implemented. Reproducing it as a `torch.autograd.Function` with a
  projected-Hessian CG solve is the promotion follow-up. The FORWARD equilibration (what the engine
  runs, what the differ compares) is faithful; the flag correctly steers an inverse loop past this
  implementation.
- **No oracle diff.** `evidence.*` still null -- comparing our relaxation trajectory to `oracle.py` is
  the CURATOR's differential test. I did not run it. My tests assert only contract properties statable
  WITHOUT the reference (force balance `|grad U| <= f_tol`; energy non-increase `U(x*) <= U(x0)`;
  overlap resolved APART to >= contact with centre-of-mass conserved; NoForce no-op limit; translation
  symmetry in free space; dt-independent one-step landing; dead cells held fixed). No fitted constants.
- **Per-cell `alpha` field** (morse well-width as a `StateFieldSpec`) is not supported -- only per-cell
  `epsilon_field`. The source allows either; I judged scope. Note for a reuser.

---

**DIFFER role -- differential test: VALIDATED (`status: validated`).**

*Metric.* `D_eq = max over the 4 LIVE cells of ||x*_plx - x*_ref||_2 / sigma` at the relaxed
equilibrium (frame 1, the single quasistatic macro-step); `sigma = r_i + r_j = 1.0` (dimensionless).
`x*_plx` is the engine's frame-1 position after the `relax` operator's quasistatic emit
`v = (x*-x0)/dt` is integrated (`pos += dt*v`, dt=1.0); `x*_ref` is the jax-morph
`MechanicalRelaxation(Morse)` FIRE-to-`f_tol` equilibrium. **Threshold `1.0e-3` sigma**, pre-registered
in `_analyze_mechanical_relaxation.py` before the diff -- bracketed >100x ABOVE the ~8e-6 sigma
two-solver float32 equilibrium shell (`f_tol/kappa` per side) and >100x BELOW the 0.1135-sigma
relaxation displacement, so it passes legitimate JAX-vs-torch roundoff yet fails any dynamic-creep /
missing-1/dt / fixed-step-descent operator that misses the basin.

*Result.* **`D_eq = 1.91e-06` sigma << 1.0e-3 -> PASS** (~500x margin; per-cell `{0,0,1.9e-6,0}`, one
cell off by ~2 float32 ulp). Corroborators all clean: force cross-check `|grad U|_inf` at the Plexus
equilibrium `1.90e-4 <= 5*f_tol`, matching reference `1.35e-4` (both are genuine force balances of the
SAME Morse energy); gauge-removed Kabsch `D_eq 1.43e-6 ~` raw (no rigid drift); fixed-point plateau
`0.0` both sides (equilibrium held over frames 2-6); frame-0 == IC both (`0.0`); dead slots immobile
both (`0.0`); misaligned-vs-IC `0.1135 =` relaxation displacement (equilibrium genuinely reached).
Summary observables agree: gyration ref-equilibrium `0.6858 ==` plexus final `0.68575`; mean_nn
`0.9698 == 0.9698`.

*Acted ledger* (`log/atlas/mechanical_relaxation/diag.json`): `relax` calls 6 / **acted 1** / moved
0.1123 (acts once, then idempotently emits ~0 on the held equilibrium -- the quasistatic fixed-point
signature), `seed_state` acted, `inert_operators []`, `valid_evidence true`. The IC is BYTE-IDENTICAL
(4-cell diamond, oracle-printed 6-decimal float32; buffer 8 with 4 dead slots at the origin).

*Runs.* Oracle `atlas_jax_morph/_oracle/runs/diff_mechanical_relaxation/` (reference.npz + summary.json);
Plexus `config/atlas/mechanical_relaxation.yaml` -> `log/atlas/mechanical_relaxation/`
(diag.json, metrics.json/.npz, spec_run.yaml, strip.png) + `graphs_data/atlas/mechanical_relaxation/`;
analysis `_oracle/scripts/_analyze_mechanical_relaxation.py` -> `log/atlas/mechanical_relaxation/diff.json`.

*What the diff DOES and does not settle.* It validates the **forward equilibration** -- the FIRE-to-
tolerance solver plus the quasistatic `(x*-x0)/dt` emit reproduce the reference equilibrium to float32
solver noise. It does NOT touch the implicit-diff backward (`DIFFERENTIABLE=False`, unchanged from the
Normalizer's honest gap); that adjoint remains the promotion follow-up and is untested by any
differential here.

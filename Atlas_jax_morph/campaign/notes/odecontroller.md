<!-- ODEController -- append below; the driver merges this into campaign/analysis.md -->

## ODEController (excavator, read at source)

Read the whole `control/ode.py` (base + all three subclasses), the `SimulationStep`/`Model` step
contract in `core/step.py`, `state.deltas`/`StateFieldSpec` in `core/state.py`, and the paper's
"Genetic regulatory interactions" (p. 10) plus fig. 1b. `code_path:L46` still points at the class def.

What it does to state: it is a **DYNAMIC** step. It packs `y = concat(hidden, outputs)` per cell,
freezes the sensed `inputs`, and integrates `dy/dt = vector_field(t, y, inputs)` over `[0, dt]` with
diffrax Dopri5 + `PIDController(rtol=1e-4, atol=1e-6)`, `dt0=dt`, saving only the endpoint. It returns
the **increment** `y(dt) - y0` as a sparse delta, not the new state -- the Model accumulates it. The
base is pure machinery; `vector_field` is abstract.

Surprised me most: the **paper vs code disagreement on where the sensed input enters the sigmoid**.
Paper p. 10: `dg_i/dt = phi(sum_j W_ij g_j + b_i) + I_i - k_i g_i` -- the forcing `I_i` is additive,
OUTSIDE the sigmoid. The closest code subclass `GeneNetworkConnectionist` puts the input INSIDE the
sigmoid (`sigma(W_gene@g + W_in@u + b)`) with no additive input term. Source wins; recorded as a
surprise on the base and it also belongs on the connectionist entry. Also notable: the "sigmoid" is
the *algebraic* `0.5 + 0.5 x/sqrt(1+x^2)` in the connectionist circuit but `jax.nn.sigmoid` (logistic)
in the MWC circuit -- two different saturations under one paper symbol.

Could NOT establish: (1) which concrete subclass the paper's headline results actually used -- the
text says only "simple ODE model inspired by Hiscock", so Connectionist is the best guess but I did
not open the example notebooks to confirm an instantiation; (2) whether the input-placement
difference materially changes fitted dynamics -- I ran nothing (excavator, and jax is deliberately
absent from the Plexus env); (3) I did not verify the diffrax adaptive solver's actual internal step
count given `dt0=dt` -- I read the config, not a trace. These are for the normalizer/validator, not me.

## Normalization (normalizer)

**Verdict: `new`**, contract `regulate` (kind=exchange, family=fields, set=cell). No promoted
operator models a per-cell, continuous-time internal regulatory network -- an intracellular
gene-regulatory ODE that freezes the cell's sensed drivers, evolves a coupled latent+output gene
vector over the macro-step, and persists it as heritable state (the genotype->phenotype decision
function). The nearest registered contracts are `decay` (its `-gamma*g` degradation term is one line
of this network) and `pacemaker` (a per-cell time-varying signal), and widening either to admit an
input-forced, nonlinearly-regulated, stateful multi-gene circuit would destroy what each *is*
(elementwise evaporation; an open-loop clock that "owns only timing"). The base carries the contract;
its three subclasses are interchangeable implementations (`implementation_of: regulate`).

**Strongest argument AGAINST `new`:** because `vector_field` is *abstract*, the base commits to no
particular reaction law -- so one could argue the base ODEController is pure numerics (a diffrax
`diffeqsolve` wrapper: Dopri5 + PIDController + delta bookkeeping) and belongs in `out_of_scope`,
with ALL the biology living only in the three subclasses' vector fields. I reject it because the
base is not agnostic where it counts: it fixes the biologically-loaded *contract* -- per-cell
heritable internal state seeded from current genes, sensed inputs held FIXED across the step (the
quasistatic-chemistry assumption), and the integrated increment returned as an accumulated delta.
Those are modeling commitments, not solver plumbing, and the whole ATLAS measurement turns on
whether the language has this contract-slot at all -- it does not. (A weaker counter, that this is a
`refinement` of `decay`, is answered in the entry's `why:`.)

## Normalization revised -- answering the skeptic (normalizer, 2026-07-31)

The skeptic disputed `new`, claiming `refinement` of `signal` (`src/plexus/operators/signal.py`):
`signal` is a registered per-agent internal-state ODE, `EMIT=velocity`,
`tau*dv/dt = -v + b + sum W*phi(v_pre)`, tag `recurrent` -- the SAME firing-rate-RNN class as
`GeneNetworkConnectionist` `dg/dt = sigma(W@g + W_in@u + b) - gamma*g`, and my predecessor's `why:`
never even named it (it called `decay`/`pacemaker` "closest", which was wrong). I agree with the
critique of the `why:` and have rewritten it to lead with `signal`; the verdict stays **`new`**.

**Strongest argument AGAINST `new` (the skeptic's, stated at full strength):** both operators are
literally the same math object -- a per-node first-order ODE whose drive is an activated linear
combination of node states plus a bias, minus linear decay. `signal` already carries that contract.
Its docstring even advertises the degenerate case ("drop the synapse state and the edge-set
collapses to a plain weighted connectome, one Lateral operator"). So the gene circuit is just
`signal` on a `cell` set with a self-loop connectome and pseudo-node inputs -- a `refinement` that
widens `signal`'s `set`/`maps`, not a new contract; two names for one recurrent-network operator is
exactly the `alias`/`refinement` inflation this loop exists to catch.

**Why it nonetheless fails.** `signal`'s typed identity is a CONNECTOME morphism: weights live on a
first-class `synapse` EDGE-SET read through `pre`/`post` incidence maps (`MAPS=[pre,post]`, "the maps
are PART of the signature"), and it `EMIT`s a velocity the ENGINE integrates. `regulate` has `maps=[]`
(dense per-operator matrix, zero cell-to-cell coupling -- purely intracellular), reads a FIXED sensed
forcing input `signal` lacks, and SELF-SOLVES the whole macro-step (adaptive Dopri5) to return the
exact `y(dt)-y0` delta. The skeptic's own `what_would_settle_it` sets the bar: a widening survives
only if it does NOT break `signal`'s connectome/engine-integration identity. Expressing `regulate`
forces changing the set, emptying the maps, dropping the edge-set, adding a forcing read, and flipping
engine-integration to self-solve -- it breaks exactly that identity. So the mismatch forces a distinct
operator and `new` stands. (`signal` and `regulate` are two contracts of the same recurrent-network
FAMILY, not two implementations of one contract -- their typed signatures differ in kind, set, maps
and integration mode.)

## Implementation (implementer)

Wrote `regulate` at `src/plexus/operators/candidates/jax_morph_odecontroller.py` (the anti-chamber;
promotion is the curator's call after the differential test). Registered
`@register_operator("regulate", family="fields", set="cell", kind="exchange",
implementation="connectionist")` -- the `connectionist` reaction law is the shipped implementation;
`mwc` (log-occupancy) and `neural_ode` (MLP field) are sibling implementations of the SAME contract
(same signature, same self-solve, only the vector field differs), exactly the diffuse
finite_difference/spectral shape. Test: `tests/test_jax_morph_odecontroller.py`, 5 passing.

Three engine-mapping decisions, each faithful to the source but forced by Plexus's integration model:

1. **The evolving state is ONE first-order block `gene`** = the source's `y = concat(hidden, outputs)`
   (surprise #8: they are integrated as one coupled system). Plexus routes a delta by LEVEL, with a
   single `INTEGRAND` block per operator, so the two persisted fields the source slices apart cannot
   be two separate Plexus blocks written by one op; the coupled vector is their honest joint form.
   `hidden_size` is kept as a documented latent/output split but does not change the integration (the
   whole block is solved). This is the FIRST registered operator to use the non-coordinate
   `_delta_blocks` path (grep confirms no other operator sets `INTEGRAND`); the machinery existed
   unused. Class `INTEGRAND="gene"` (so `_resolve_emit` sees a non-`pos` integrand and never
   constrains the cell's coordinate order); instance `self.INTEGRAND` routes to the configured block.

2. **inc/dt EMIT scaling.** The source is a DYNAMIC step: it self-solves and returns the exact
   increment `g(dt)-g0`, which the Model ADDS (surprises #1/#2). Plexus integrates a first-order block
   as `g += dt*delta`, so the operator returns the effective mean rate `delta=(g(dt)-g0)/dt` and the
   engine's `dt*` recovers the exact endpoint -- the dt cancels; it is NOT a second integration. This
   preserves "exact integrated increment, not rate*dt" while obeying `EMIT=velocity`.

3. **SOURCE WINS on the forcing input.** Implemented the code's `sigma(W_gene@g + W_in@u + b) -
   gamma*g` (drive INSIDE the sigmoid), not the paper's additive-outside `+ I_i`, because the
   differential test compares us to the running source. Sigmoid is the ALGEBRAIC
   `0.5+0.5 x/sqrt(1+x^2)` (source's `_rescaled_sigmoid`), computed via `hypot(1,x)` for stability.
   Drive `u` read from a frozen `inputs` block (integration=none), closed over for the whole solve.

Translated `diffrax.Dopri5() + PIDController(rtol=1e-4, atol=1e-6), dt0=dt` as a genuine batched
adaptive Dormand-Prince 5(4): DP tableau, embedded 4th-order error, RMS error norm over all elements
(diffrax's default -> one shared step sequence over the stacked per-cell state), I-controller
`h *= clamp(0.9*err^-0.2, 0.2, 5)`, first step = dt. NOT hard-coded to the oracle. Sanity-checked
the endpoint against a 20000-step fixed RK4 on a nonlinear coupled 2-gene+drive case: max abs diff
7.2e-5 (< rtol), so the tableau and control are correct. Dead cells (occ=0) get a zero delta
(gene state frozen) -- the same net effect as the source's post-hoc alive-mask (surprise #9).

Test properties (all reference-free -- exact solutions of the drive-frozen scalar ODE, since W_gene=0
makes the drive constant in g): (a) linear-decay increment exact -- `dg/dt=0.5-g` from g0=0 lands on
`0.5(1-e^{-dt})` to 1e-4; (b) same through the engine `_integrate` from g0=0.3; (c) fixed point
g*=0.5/gamma is stationary (delta~0); (d) frozen drive forces g*=sigma(k*u) (input path) and the
`drive` block is unchanged after the step; (e) dormant cells hold their gene state.

Could NOT do (left for the validator/curator): the differential run against the oracle -- I did not
run jax (deliberately absent from the Plexus env) and did not build the paired ODE scenario. The
oracle uses adaptive Dopri5 at the same rtol; both should converge to the true solution to ~1e-4, so
tolerance-level agreement is expected, but that is evidence for the validator to produce, not me.
`evidence`/`status: validated` stay untouched (status advanced only to `implemented`).

## Differential test (DIFFER, 2026-07-31)

**Verdict: `validated`.** The `regulate` base-machinery operator (implementation `ode_generic`,
connectionist reaction law) reproduces the jax-morph `GeneNetworkConnectionist` reference
trajectory to solver tolerance.

**Metric (fixed BEFORE running).** `D_inf` = sup-norm |engine_gene - reference_gene| over all 22
recorded frames (tick 0 = seeded IC, ticks 1..21 = the integrated macro-steps) x 4 live cells x 3
genes `[gene_hidden, g_out0, g_out1]`, same IC (all genes 0), same frozen drive `u=[0.8,0.3]`, same
3-gene circuit (W_gene, W_in, b; gamma=1.0), same dt=1.0, 21 steps. Units: gene concentration
(dimensionless; fixed points O(0.3-0.85)).

**Threshold (principled, not tuned): 2e-4.** Both sides integrate the IDENTICAL RHS with an
adaptive Dopri5 at rtol=1e-4/atol=1e-6 (reference: diffrax PIDController; operator: a hand-rolled
DP5(4) at the same tolerances), so two faithful integrations of the same ODE sit apart by at most
~2*(rtol*|y|+atol) ~ 2e-4 on these O(1) values -- an a-priori bound a CORRECT integrator is
guaranteed to meet. Tighter than the >= 1e-2 fixed-point shifts the discriminating failure modes
would cause (logistic vs the algebraic sigmoid; the paper's additive-outside forcing vs the code's
`W_in@u` inside; a mis-scaled / omitted inc-over-dt increment == double integration), so it
separates "same circuit to solver tolerance" from "a different circuit".

**Result: PASS, ~80x under the bar.**
  * `D_max_uniform`  = 2.44e-06  (PRIMARY; the spec run_spec.py executes)
  * `D_max_distinct` = 2.56e-06  (per-cell distinct drives; corroboration)
  * engine frame-0 IC max|gene| = 0.0 -> IC aligned exactly.
  * per-frame error peaks 2.44e-06 mid-trajectory, decays to 0 at the fixed point -- two accurate
    solvers on a CONTRACTING ODE, not accumulating disagreement.

**The distinct run is the behavioral test of the `maps=[]` verdict.** Four cells given four
DIFFERENT frozen drives follow four DIFFERENT reference trajectories, each matched to 2.56e-06 with
zero cross-cell leakage -- the intracellular no-coupling identity that makes `regulate` `new` over
`signal` (a connectome morphism) is not just structural in the code, it is measured. seed_state
cannot express a per-cell drive, so this run was stepped through the engine's own `gene += dt*delta`
(`_integrate`) exactly as `engine.run` does -- the operator as the engine runs it, not a bench rig.

**Paths.**
  * oracle:  `Atlas_jax_morph/_oracle/runs/diff_odecontroller/` (reference.npz, summary.json, diff.json)
  * engine:  `log/atlas/odecontroller/` (diag.json acted ledger: regulate acted 19/21 -- the last two
    near-fixed-point ticks emit a delta below the float32 ledger threshold; convergence, not a
    no-op; `valid_evidence: true`, no inert operators).
  * comparison script: `Atlas_jax_morph/_oracle/scripts/_compare_odecontroller.py` (torch env).

**What this validates and what it does NOT.** It validates the BASE ODEController contract end to
end: the coupled hidden+output `y=concat(hidden,outputs)` block, the frozen-drive quasistatic
assumption, the algebraic sigmoid, linear degradation, the self-solved adaptive Dopri5 endpoint,
AND the inc-over-dt EMIT scaling that lets the engine's first-order step recover `y(dt)` -- all
exercised on the connectionist law (the only concrete law shipped in this module). It does NOT
separately test the `mwc` or `neural_ode` sibling implementations of `regulate` (their own entries;
`neural_ode` is an operator-level weight-export diff, not a spec run, since a free-form MLP cannot
cross the JAX/torch boundary through a YAML spec). The paper-vs-code forcing contradiction is
recorded as a surprise; the test compares against the running SOURCE (drive inside the sigmoid), and
the source wins -- as it must.

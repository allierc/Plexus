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

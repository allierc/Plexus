<!-- GeneNetworkConnectionist -- append below; the driver merges this into campaign/analysis.md -->

# gene_network_connectionist (NORMALIZER)

**Verdict: `new`, contract `regulate` (exchange/fields, set=cell); `implementation_of: regulate`.**
The mechanism is a per-cell INTERNAL regulatory dynamical system — a heritable gene-state vector
whose autonomous per-cell ODE (dense recurrent gene→gene coupling through a saturating
nonlinearity + linear decay + a learnable drive from *sensed external fields*) is integrated over
the macro-step to emit the action-setting outputs that division/secretion/adhesion read. Nothing in
the frozen 42 covers cell-internal regulatory computation: the motion/interaction/mechanics/mpm
families move or couple agents; the fields family sources/reads fields; growth/topology change the
set. The one real neighbour is `signal`. Crucially, this is also the anti-inflation call: the three
`ODEController` subclasses (this connectionist linear-drive form, `GeneNetworkMWC`'s thermodynamic
log-occupancy drive, and `NeuralODE`'s black-box MLP) are interchangeable *implementations* of the
same `regulate` contract — different vector fields under one integration/IO contract — so they
collapse to **one** new contract, not three. That is the convergence the ledger exists to record.

**Strongest argument AGAINST `new` (and why it loses).** `signal` is already a registered
recurrent, nonlinear, first-order ODE network (its own tags say "recurrent"), with the identical
−decay + bias + saturated-weighted-drive skeleton; one could argue `regulate` is merely a
`refinement` of `signal` — widen `set` to `cell`, make `edge_set`/`MAPS` optional, add a
field-input term — and that "gene regulation vs. connectome" is just parameterization of one
"recurrent-ODE-network" contract. This is the counterargument I had to defeat, and it is the
tempting one because the *math* really does rhyme. It loses on the *signature*, not the math:
`signal`'s recurrence runs BETWEEN nodes across a fixed connectome edge-set, and its typed
signature is load-bearing on exactly that topology (`INPUTS ["neuron","synapse"]`,
`MAPS ["pre","post"]`, `REQUIRES_PARAMS ["edge_set"]`, activation on the *presynaptic* input). The
gene circuit has no edge set and no cross-cell coupling at all — `W_gene` is a dense WITHIN-cell
matrix applied per cell (vmap), the sigmoid wraps the WHOLE drive, and the environmental forcing
enters via `W_in` on sensed *fields*, a term `signal` simply lacks. To "widen" `signal` to admit
this you must make its maps/edge-set optional and bolt on field sensing — i.e. delete the
connectome signalling that IS the contract for its only user (a neuron network) and convert a
lateral graph operator into a per-cell field→state controller. A refinement that guts the
signature its sole caller depends on is a breaking change wearing a smaller word, so the honest
verdict is `new`. (Second-order caveat I chose against: I filed `regulate` under the existing
`fields` family rather than minting a `control`/`regulation` family — defensible because its whole
I/O is per-cell scalar fields and the `signal` precedent already parks a control-like ODE inside an
existing family, but a future normalizer could reasonably argue the paper's separate control layer
earns its own family.)

---

# gene_network_connectionist (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_connectionist.py` —
`RegulateConnectionist(Exchange)`, registered `@register_operator("regulate", family="fields",
set="cell", kind="exchange", implementation="connectionist")`. Torch, not JAX. Test:
`tests/test_jax_morph_gene_network_connectionist.py` (5 property tests, all green). `status ->
implemented`; evidence left null for the differ.

**The one real design decision — SELF-SOLVE vs emit-a-rate.** The source is an `ODEController`: it
integrates the ODE over the whole macro-step with adaptive Dopri5 and its DYNAMIC step returns the
*sparse endpoint delta* `g(dt)-g(0)`, not an instantaneous rate. Plexus's engine integrates a
first-order block with one Euler step per tick (`g += dt*delta`). I chose to reproduce the reference
ENDPOINT: integrate internally over `[0,dt]` with fixed-step RK4 (`substeps`, default 8) and return
the *mean rate* `(g(dt)-g(0))/dt`, so the engine's `dt*` recovers `g(dt)` exactly (the dt cancels —
not a second integration). This is the operator's defining behavior per its own equations/surprises
("emits the SPARSE DELTA … integrated over one macro-step by Dopri5"), and it maximizes fidelity for
the coming differential test. Fixed-step RK4 vs adaptive Dopri5 is a numerics choice inside the one
contract; `substeps` tightens it. NB the `mwc` sibling made the OPPOSITE call (emit instantaneous
`dg/dt`, let the engine Euler-step, invoking "integration is the engine's concern") — a genuine
unresolved tension across the four `regulate` files that the curator should reconcile when promoting.

**SOURCE WINS, faithfully ported.** (1) Sensed input enters INSIDE the sigmoid via a trainable
`W_in @ u` (code), NOT as an additive `+ I_i` outside it (paper) — implemented the code. (2) `sigma`
is the ALGEBRAIC `0.5 + 0.5 x/sqrt(1+x^2)` (`_rescaled_sigmoid`), not logistic — ported including the
overflow guard (clip to finite range, rescale by max(1,|x|)). At extreme drives it saturates to
*exactly* 0/1 (float32 underflow of `scaled_one^2`); that is correct, so my sigmoid test asserts
strict `(0,1)` only for moderate drives and mere finiteness at 1e30. (3) Defaults are NOT inert:
zeros for W/W_in/b but `gamma=0.1`, so `dg/dt = 0.5 - 0.1 g` drives every gene to `g*=5.0`. (4)
`gamma` stored verbatim, NOT shape-checked (the one param the source skips through `_resolve_param`);
matrices materialize lazily on first forward and ARE shape-checked. `u` is a frozen quasistatic
snapshot for the whole solve.

**Routing.** `INTEGRAND="gene"` (instance-set to the configured block; class stays `"gene"` so
`_resolve_emit` sees a non-`pos` integrand and does not constrain the coordinate order), `EMIT="velocity"`,
`MAPS=[]`. Sensed drivers `u` are read as per-cell STATE BLOCKS (the reference `_pack`s per-cell input
specs — they are not grid fields), named by `inputs:` (a name or list, concatenated). Dormant cells
(`occ=0`) get a zero delta.

**Property tests (all reference-free — stated from the operator's own definition):** (1) production
`= dg/dt + gamma*g` is strictly in (0,1) for random large drives (algebraic-sigmoid range); (2)
`sigma(0)=0.5`, monotone, overflow-finite; (3) zero-interaction circuit is not inert and vanishes at
`g*=5.0`; (4) end-to-end through `build`+`_integrate`: the engine result equals the internal
self-solve and CONTRACTS toward `g*=5.0` on the original side; (5) dormant cells don't evolve.

**Heads-up for the curator (registration overlap, not a bug).** The `odecontroller` entry (order 4,
the abstract base) also landed a `regulate`/`connectionist` file (`jax_morph_odecontroller.py`, an
adaptive-Dopri5 self-solve) as its concrete representative. Mine is the CANONICAL connectionist
entry (order 1). Candidates are never bulk-imported (the anti-chamber is full of intentional name
clashes), so each imports/tests alone — no runtime collision — but the two `regulate`/`connectionist`
modules are the same operator and should collapse to one on promotion. That collapse (four
ODEController entries → one `regulate` contract with implementations connectionist/mwc/neural_ode) IS
the convergence result the ledger is meant to record.

**Did NOT run** the oracle (differential test is the differ's job; evidence stays null).

<!-- GeneNetworkMWC -- append below; the driver merges this into campaign/analysis.md -->

# gene_network_mwc (NORMALIZER)

**Verdict: `new`, `implementation_of: regulate`.** No registered contract covers a per-cell gene
regulatory network: a stateful, recurrent intracellular circuit that integrates sensed
morphogen/mechanical fields into gene-expression outputs by integrating an ODE over a macro-step.
Contract `regulate` = `exchange`/`fields`/`cell`, reads {gene, driver}, writes {gene}. MWC is not a
separate contract but one of three interchangeable `ODEController` vector-fields (Connectionist =
linear W*g, the paper's eq. 4; MWC = thermodynamic log-occupancy; NeuralODE = MLP) that share one
signature and differ only in the drive's functional form — the Morse-vs-SoftSphere pattern — so the
gene-network family contributes ONE new contract, and `implementation_of` keeps the saturation
ledger from triple-counting it.

**Strongest argument against `new`.** The registered `signal` operator already encodes the *defining*
dynamics of a gene regulatory network — a recurrent leaky integrator over a weighted node network
with a nonlinear per-node drive, a time constant, and a resting bias (`dv/dt = -v/tau + sum
activation(v_pre)*w + bias`). One can read the GRN as `signal` under three *widenings*: (1) let
`edge_set` be optional / admit a dense within-element coupling instead of a sparse connectome, (2)
add a field-forcing input channel, (3) let each node carry a vector of states with several named
outputs. Under that lens this is a `refinement of signal`, and calling it `new` risks the exact
inflation the ledger is built to catch — a leaky-integrator recurrent network is a leaky-integrator
recurrent network whether its nodes are neurons or genes. I rejected it because those three
"widenings" together delete `signal`'s required `edge_set` and pre/post maps and its scalar-per-node,
no-external-input semantics — that is not a widening but a rebuild, and a refinement that changes a
contract's `requires_params` silently breaks every existing `signal` user (a refinement nobody costed
is a breaking change). But the counter-argument is genuine: if Plexus later generalizes `signal` to a
"recurrent regulatory network over an arbitrary coupling structure," `regulate` and `signal` could
well collapse into one contract, and this `new` would be retro-classed a refinement.

---

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_mwc.py` — the `regulate`
contract registered with `implementation="mwc"` (`RegulateMWC(Exchange)`, kind=exchange /
family=fields / set=cell, matching the normalized contract). This is the diffuse
`finite_difference`/`spectral` pattern applied to `regulate`: the connectionist and neural-ODE
siblings can later register `implementation="connectionist"` / `"neural_ode"` on the SAME
contract, so the gene-network family stays ONE contract with three interchangeable vector
fields — the convergence the ledger is meant to record. `get_contract("regulate")` now resolves;
`implementations = {"mwc"}`.

**The load-bearing decomposition choice.** The reference's `ODEController.__call__` self-solves
the ODE over the macro-step with diffrax Dopri5 (adaptive) and returns the increment
`y(dt) - y0`. Plexus separates biology from time-stepping: the operator returns the *vector
field* `dg/dt` as a first-order delta on the cell's `gene` block (`EMIT=velocity`,
`INTEGRAND=gene`), and the ENGINE integrates it (`x += dt*delta`, engine.py `_integrate`). This
is the `signal` precedent verbatim (`signal` returns `dv/dt` and lets the engine integrate the
neuron voltage). So the three ODEController subclasses differ ONLY in `f` — exactly the record's
"share the integration+I/O contract, differ only in the vector field" claim, now realised in
code. Sensed inputs `u` are read from a per-cell `sensed` state block and never written (held
fixed across the step), matching the reference's quasistatic-chemistry `_pack(state,
input_specs)`; the upstream `sense`/`chemotax` step is what fills that block.

**FLAG for the curator's differential run (not a translation bug).** The engine takes ONE
explicit-Euler step per tick where the reference takes an adaptive Dopri5 sub-solve of the same
vector field over the same interval. For a non-stiff field at small `dt` the gap is small but
NONZERO, and it will show up in the differ. It is an Axis-A (integration) difference, not a
difference in the mechanism `f`. To compare `f` itself, evaluate the vector field at `t=0`
(`RegulateMWC.vector_field(evolving, inputs)`) against the reference's `vector_field` on matched
state — that isolates the biology from the solver. Multi-step trajectory agreement would require
substepping the gene block (a `substep_dt` micro-loop) or an RK integrator in the engine; I did
NOT do that, deliberately — self-integrating inside the operator is the "category error" the
engine guards against on frame 0.

**Faithful details carried over (ode.py:449-475):** occupancy uses genes/drivers CLAMPED to >=0
(ln(1+.) would NaN on negatives) while the DECAY term uses the RAW evolving state (restorative
toward 0 — the intentional asymmetry); the g/K ratio is capped at `finfo.max` before `log1p`
(the mixed-sign +inf/-inf overflow guard); rho/tau/K come through `_positive_from_log` (clip to
[log tiny, log max - log 4] then exp, so unset log-0 params give rho=tau=K=1); the sigmoid is
`torch.sigmoid` (logistic), NOT the sibling connectionist's rescaled algebraic sigmoid. `H_gene`
orientation verified non-transposed: `H_gene[i,j]` = weight of regulator j on target i (checked
against an asymmetric 2-gene circuit).

**Test:** `tests/test_jax_morph_gene_network_mwc.py` (6 pass). Headline property (a limit,
reference-free): the production term `rho*sigmoid(F)` is STRICTLY in `(0, rho)` for any genes,
drivers, and finite params — the definition of saturating production, and what separates MWC
from an unbounded linear drive. Plus: the inert circuit's fixed point (`dg = 0.5 - g`), the
activating(+)/inhibitory(-) sign of `H`, the restorative-decay asymmetry on a negative
concentration, dormant cells frozen, and one end-to-end engine-integration check. No oracle
numbers hard-coded.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.

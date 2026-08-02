<!-- energy_sum_composition -- append below; the driver merges this into campaign/analysis.md -->

**Verdict: out_of_scope.** The Hamiltonian-as-sum-of-plugin-terms is not an operator over any set or
field; it is the framework's *composition law* — the meta-rule that every enabled mechanism contributes
an additive energy term and the terms interact only through the summed `dE` the acceptance test reads. It
has no biological content, no parameters, and writes no state. There is nothing to alias or widen because
Plexus's own composition law — operator splitting, where each operator returns a delta the engine
applies/integrates in sequence (the Lie-Trotter split the surprise names) — lives at the same
architectural layer and is likewise *not* a member of the operator vocabulary in `src/plexus/operators/`.
The campaign counts contracts over sets/fields; a calling convention is not one, so I record the finding
loudly (energy summation vs operator splitting = two answers to how mechanisms compose) rather than
inflate the yield with a non-vocabulary item. This is the same cut `celltype` made (a schema is not a
per-step operator), deliberately *not* the cut `cell_as_lattice_domain` made (that earned `new` because
the extended lattice domain is a substrate *thing operators read/write over*, which the point-agent
algebra genuinely cannot represent).

**The strongest argument against.** Energy summation is arguably a genuine *gap* the algebra cannot
express, not mere plumbing: operator splitting applies each delta unconditionally and in order, so it
cannot represent CC3D's simultaneous, order-independent, within-step trade-off resolved by one stochastic
accept/reject (a copy that raises volume energy but lowers contact energy more still gets accepted). By
that reading the algebra *is* incomplete on the composition axis, and burying that under `out_of_scope`
hides exactly the incompleteness this exercise exists to surface — the parallel to `cell_as_lattice_domain`
(also architectural, yet `new`) is real. My rebuttal is that the incompleteness is at the *interpreter/
engine* layer, not the operator *vocabulary*, and the saturation curve measures the vocabulary; forcing a
composition scheme into the contract count would measure our engine's semantics, not the biological
language. But the argument is strong enough that if a future promotion ever makes "how operators compose"
a first-class, parameterizable choice inside the algebra, this entry — together with `metropolis_acceptance`
— is where the energy-summation composition mode should be re-litigated as `new`.

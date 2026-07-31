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

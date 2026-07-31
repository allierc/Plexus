<!-- NeuralODE -- append below; the driver merges this into campaign/analysis.md -->

## Normalization (normalizer)

**Verdict: `new`**, contract `regulate` (kind=exchange, family=fields, set=cell),
`implementation_of: regulate` -- consistent with the `odecontroller` base and the two gene-network
siblings. NeuralODE differs from `GeneNetworkConnectionist`/`GeneNetworkMWC` in exactly one place:
its vector field is a free-form per-cell MLP (`dy/dt = MLP([u, y])`) instead of a sigmoid-linear or
log-occupancy regulatory law. Every other commitment is identical -- it freezes the sensed drivers
`u`, seeds `y0 = concat(hidden, outputs)`, integrates with the same Dopri5/PIDController machinery,
and returns the integrated increment as a sparse dt-delta. That is the Morse/SoftSphere/Hertzian
shape: three interchangeable implementations of the one missing contract-slot -- a per-cell,
sensor-driven, latent-carrying internal regulatory ODE -- which no promoted operator provides
(`decay` is one degradation term of it; `pacemaker` is an autonomous clock with no sensed drive;
`sense` emits a heading, not integrated internal state). So the family yields ONE new contract with
three implementations, not three new contracts, and the yield is not inflated.

**Strongest argument AGAINST `new` (here, `out_of_scope`):** unlike its two siblings, NeuralODE has
NO paper counterpart and an *uninterpretable* right-hand side -- a generic `eqx.nn.MLP` wrapped in a
diffrax solver, doing no gene-specific biology. One could argue it is pure function-approximation
plumbing (a learnable black box + numerics) with no biological content, and that all the modeling
commitment lives in the structured gene circuits. I reject it: biological status here comes from the
contract SLOT the operator fills, not from the legibility of its reaction law. NeuralODE reads the
same sensed drivers, evolves the same coupled latent+output cell state, and persists it as the same
heritable genotype->phenotype decision function -- it is the *learned* implementation of `regulate`,
exactly as a fitted potential is still `adhere`. Recording it `out_of_scope` would hide a real
implementation of a real (and, per the ATLAS measurement, genuinely absent) contract. Its lack of a
paper counterpart is captured instead as a surprise (source wins: a paper-only reimplementer would
build the gene network, never this).

<!-- Hertzian -- append below; the driver merges this into campaign/analysis.md -->

# Hertzian (potentials.py:L345)

Read the whole `Hertzian` class, its base `PairwisePotential` (L128), the `Potential` protocol
(L56), and the shared helper `_compact_repulsion` (L30) plus `safe_divide`/`safe_norm`
(core/ad_utils.py). Hertzian is a two-line subclass: it supplies `pair_params` (sigma = r_i + r_j,
epsilon) and `pair_energy = _compact_repulsion(r, sigma, eps, exponent=2.5, prefactor=0.4)`, i.e.
`U = (2/5) eps (1 - r/sigma)^(5/2)` for r < sigma. Everything else (total energy, autodiff forces,
virial pressure) is inherited from the base. It's the softer sibling of `SoftSphere` (exponent 2 vs
2.5): both the force AND its slope vanish at contact.

Biggest finding: **Hertzian is not in the paper at all.** The paper's mechanics is exclusively the
Morse potential (adhesive soft spheres; p. 9 / p. 14 Methods, Fig. 1d) -- grepped the plaintext for
"hertz" (zero hits) and confirmed the only mechanics energy defined is Morse. Grepped the whole
jax-morph repo: `Hertzian(` is instantiated only in `tests/physics/test_potentials.py` and
`examples/03_potentials.ipynb`; no config, model, or oracle script uses it. So it is a code-only
extension of the pair-potential library, and (being purely repulsive with no adhesive tail) it
cannot even reproduce the paper's cell-cell adhesion. Recorded as the primary surprise + a
PAPER-vs-CODE line in `equations`.

Two guards a reimplementer would miss, both verified in source: (1) the prefactor 2/5 = 1/exponent
normalizes the force prefactor to exactly eps/sigma -- drop it and forces are 2.5x too strong; (2)
the double-`where` in `_compact_repulsion` exists so the fractional power only ever sees a strictly
positive base, keeping the gradient finite for r >= sigma under the always-on `jax_debug_nans`. Also
noted: dead/padded-pair masking is EXTERNAL (sigma=0 gives base=1 -> a spurious 0.4*eps that
`neighbor_sum(u, state.alive)` cancels downstream, not inside the energy).

Did NOT establish / left open: I did not run the oracle (no jax in this env, and no oracle script
exercises Hertzian anyway -- it would need a new script), so there is no numeric confirmation of the
energy/force shape beyond reading -- the equation is transcribed from source, not measured. I read
but did not deeply trace `neighbor_sum` / `pairwise_displacements` in core/geometry.py -- I relied on
the base-class docstring's claim that `neighbor_sum` masks self-pairs and dead cells; a reader who
wants the exact minimum-image / masking mechanics should open that file. Left `verdict`/`contract`
null for the normalizer (Hertzian, SoftSphere, and the `PairwisePotential` base likely collapse to
one contract family, but that is not my call).

**Normalizer verdict — `alias` of `attraction_repulsion` (implementation_of: attraction_repulsion).**
Hertzian is a conservative radial pair force run in its repulsion-only limit — a compact soft core
with contact `sigma = r_i + r_j`, the exact biology of `attraction_repulsion`'s "pull minus push"
with the pull term set to zero. It is the purely-repulsive, C2-soft (force AND its slope vanish at
contact), self-truncating member of the PairwisePotential family. The abstract parent — already alias
→ attraction_repulsion — names Hertzian verbatim as `implementation_of: attraction_repulsion`, and
Morse (order 14) and Harmonic (order 17) followed. Keeping the whole family under one contract is the
several-implementations-per-contract pattern the ledger measures; minting a new contract would inflate
the yield. **Strongest argument against:** the direct sibling `SoftSphere` (order 15), the OTHER
purely-repulsive member, did NOT alias — it minted a NEW contract `adhere`, arguing the registered
`attraction_repulsion` is specifically the D'Orsogna self-propelled-particle law (a hand-coded velocity
of FIXED GLOBAL width `sigma`, keyed to per-TYPE `p=[pull,pull_range,push,push_range]`, message-passed
over a `radius_graph` edge-set, that never reads a cell's physical radius). Hertzian instead reads
`radius` to build a per-pair size-consistent `sigma = r_i + r_j`, takes a per-CELL scalar `epsilon`,
and is an ENERGY over DENSE N×N pairs turned to force by `-jax.grad`. If the contract's identity is
"hand-coded per-type force on a sparse graph" rather than "the conservative radial pull-minus-push
law", then admitting a radius-derived, per-cell, energy-defined, dense-pairs realization is arguably a
`refinement` (or, as SoftSphere ruled, a genuinely distinct `adhere` contract) rather than a free
alias. I judge those to be sub-signature implementation axes — stillinger_weber is already an
energy-defined interaction in this same family, squared_law already carries both all-pairs and a graph,
and attraction_repulsion's own force is the gradient of a radial potential — and I follow the parent +
Morse + Harmonic majority over the SoftSphere dissent so the pair-potential family does not fracture
across two contracts. But the structural gap is real and is the honest case against the alias.

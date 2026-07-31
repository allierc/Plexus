<!-- Harmonic -- append below; the driver merges this into campaign/analysis.md -->

# harmonic (Harmonic pair potential @ potentials.py:L375)

**What I read.** The whole `Harmonic` class (L375-416) and everything it inherits: `PairwisePotential`
(L128-266, giving `total_energy` = 0.5*sum over live non-self pairs, `forces = -jax.grad(energy)`,
`virial_pressure`, and the scalar-or-per-cell-field coupling resolution + arithmetic-mean `mix`) and the
`Potential` protocol (L56). Compared it against its four siblings in the same file to see what Harmonic
does DIFFERENTLY.

**What it does to the state.** Nothing directly — a Potential is a pure energy/force function. `pair_energy`
is a parabola in `(r - sigma)`, `sigma = r_i + r_j` (contact = sum of radii), shifted down by
`(r_c - sigma)^2` and hard-truncated to 0 at `r_c = r_cutoff_frac*sigma` (default 2.5). The shift puts the
well minimum below zero at contact, so it is repulsive when compressed and **adhesive** when stretched
(`sigma < r < r_c`). Params: `k` (stiffness) and `r_cutoff_frac` (range). Forces are consumed by the
wrapping relaxation/Brownian step, which is what moves positions.

**Line/anchor checks.** L375 is exactly `class Harmonic(PairwisePotential)` — code_path unchanged. Paper
anchor: p. 9 ("Mechanical interactions", eq. Vij) and SI p. 14 — but see the surprise.

**What surprised me.** (1) BIGGEST: Harmonic is **not in the paper at all**. The paper defines a single
mechanical potential, the Morse well; Harmonic, SoftSphere, Hertzian and LennardJones are library-only
additions. The source is strictly richer than the paper here — no contradiction, but a paper-only
reimplementer would never write this class. (2) The down-shift is load-bearing: drop `(r_c - sigma)^2` and
you get a purely repulsive, infinite-range spring with no adhesion and no truncation — the opposite of the
intended finite-range well. (3) C0-only cutoff — unlike Morse/LJ (C1 `_smooth_cutoff`) and
SoftSphere/Hertzian (compact C1 tail), Harmonic uses a bare `jnp.where`, so the FORCE jumps at `r_c`; the
docstring calls it harmless because `r_c` sits far beyond resting contact. (4) Two combining rules coexist:
`sigma` additive (`r_i+r_j`), coupling `k` arithmetic-mean `mix` (deliberately not Lorentz-Berthelot — no
sqrt, keeps the gradient NaN-safe).

**What I did NOT establish.** I did not run the oracle — no numeric confirmation of the energy/force
(evidence left null; EXCAVATOR pass). I did not find any campaign config or paper experiment that actually
drives Harmonic (everything uses Morse), so whether this class is exercised downstream is unconfirmed —
`k`-as-a-per-cell-`StateFieldSpec` is a supported mode but I saw no caller use it. `virial_pressure` is
inherited/available but I did not trace whether any step consumes it for Harmonic. Left `verdict`/`contract`
for the normalizer per the role rules.

**Normalizer verdict — `alias` of `attraction_repulsion` (implementation_of: attraction_repulsion).**
Harmonic is a conservative radial pair force with a repulsive core (`r < sigma`) plus an adhesive tail
(`sigma < r < r_c`) and its well minimum at contact `sigma = r_i + r_j` — the exact biology of
`attraction_repulsion`'s "long-range pull minus short-range push". It is a second core+tail member of the
PairwisePotential family alongside Morse (which already landed alias → attraction_repulsion and whose
normalization the parent quotes as naming Harmonic verbatim), differing only in the well SHAPE (a truncated
down-shifted parabola vs the Morse exponential) — the several-implementations-per-contract pattern, so
minting a new contract would inflate the ledger's yield. **Strongest argument against:** attraction_repulsion
is registered as a *first-derivative, hand-coded velocity law message-passed over a `radius_graph` neighbour
graph*, whereas Harmonic is an *energy* defined over a *dense N×N all-pairs* matrix and turned into a force
by `-jax.grad`. If one reads the contract's IDENTITY as "hand-coded force on a sparse graph" rather than "the
conservative radial pull-minus-push law", then admitting an energy-defined dense-pairs implementation is
arguably a `refinement` (widen the operator to accept an energy→autodiff realization and an all-pairs
topology) rather than a free alias. I judge those to be sub-signature implementation axes — stillinger_weber
is already an energy-defined interaction in this same family, squared_law already carries both all-pairs and
a graph, and attraction_repulsion's own hand-coded force *is* the gradient of a radial potential — so no
signature field is forced to change and the alias holds; but the structural gap is real and is the honest
case for refinement.

**Normalizer verdict — REVISED to `new`, implementation_of `adhere` (supersedes the alias paragraph
above).** The prior alias rested on treating the `radius` read as sub-signature. I checked the source and it
does not survive that check. Registered `attraction_repulsion` (attraction_repulsion.py:28-64) reads NO
`radius`, its interaction length is a single GLOBAL scalar (`self.sigma = float(params["sigma"])`), its
force-law `p` is per-TYPE (`REQUIRES_TYPE_PROPS=["p"]`), and it EMITs a hand-coded velocity — the D'Orsogna
point-particle law. It structurally cannot express `sigma = r_i + r_j` or a per-cell coupling. `reads` is a
registered signature field, so attraction_repulsion AS REGISTERED does not cover Harmonic: aliasing drops
the radius read and the size-consistency-under-growth that is the biology of an adhesive soft sphere. And
hosting the family would replace nearly every distinctive attribute at once (velocity→energy/autodiff,
global→per-pair sigma, per-type→per-cell coupling, graph→dense), which is not a one-field widening, so not
`refinement`. Harmonic is a core+tail member of the `adhere` contract SoftSphere (order 15) minted; joining
it adds no contract and is the convergent several-implementations-per-contract shape the ledger rewards.
This is exactly the check the skeptic's `what_would_settle_it` demanded, now confirmed at source; I align
Harmonic with SoftSphere and against the parent/Morse/Hertzian alias entries, whose premise the source
contradicts.

**Strongest argument AGAINST the revision.** The registry is designed to hold several implementations per
contract, and it already treats energy-vs-force (`stillinger_weber`) and dense-vs-graph (`squared_law`) as
below-signature — leaving only the `radius` read, which one can frame as merely another way to source the
interaction length attraction_repulsion already has via its global sigma. On that reading all five pair
potentials, SoftSphere included, are the ONE contract `attraction_repulsion` ("attraction minus repulsion,
a radial pair force moving cells"), and minting/joining `adhere` splits a single biological contract on an
implementation-internal length source, inflating precisely the `new` yield this ledger measures. The parent
plus three siblings landing alias is real evidence that reading is defensible. I reject it because a
per-pair contact tracking each cell's growing radius is a biologically load-bearing feature (size-aware cell
mechanics vs a fixed-width swarming law are distinct model classes), not an alternate length source — but
this is the genuine tension, and the family record stays split until the alias entries are revisited.

<!-- Morse -- append below; the driver merges this into campaign/analysis.md -->

# Morse (order 14) -- excavation note

**Read:** `potentials.py` Morse class (L269) + base `PairwisePotential`/`Potential`, the helpers
`_smooth_cutoff` and `_compact_repulsion`, and the primitives it calls (`safe_norm`, `safe_divide`,
`pairwise_displacements`, `neighbor_sum`). Paper Morse eq. p. 9 (Methods) and its verbatim restatement
p. 14 (SI); fig. 1d prose. Cross-checked `tests/physics/test_potentials.py`.

**What it does to the state:** nothing directly. Morse is an *energy*, not a step -- it supplies
`total_energy` -> (autodiff) `forces` -> `virial_pressure`. A wrapping step (MechanicalRelaxation,
BrownianDynamics, VirialStress) is what actually moves positions / writes a stress field. The energy
is the standard Morse well, minimum `-epsilon` at contact `sigma = r_i + r_j`, repulsive core + adhesive
tail, summed over live non-self pairs with a 0.5 pair-once factor.

**Biggest surprise (paper-vs-code):** the paper's written formula is the *bare* Morse well with no
cutoff. The code multiplies it by a jax-md `multiplicative_isotropic_cutoff` on `[1.5*sigma, 2.5*sigma]`
(C1, so the force has a small slope kink at the window edges). A genuine paper/source gap; the source
wins. The well already decays to 0 at infinity, so the cutoff is a tail-truncation, not a boundedness
fix -- easy to mistake for load-bearing.

**Other reimplementer traps:** two combining rules coexist -- `sigma` is additive (`r_i+r_j`) but
per-cell `epsilon`/`alpha` are arithmetic-mean `mix()` (chosen no-sqrt so gradients survive
`jax_debug_nans`); the 0.5 in `total_energy`; the `safe_divide`/`safe_norm` guards that keep dead-dead
padded pairs (`sigma=0` -> `0/0` in the cutoff) finite. Defaults `epsilon=3.0, alpha=2.8, r_onset=1.5,
r_cutoff=2.5` are magic -- none appear in the paper.

**Could NOT determine:** how the paper's differential-adhesion example (Fig. S1: a full per-pair,
type-dependent well-depth *matrix* = homotypic/heterotypic cadherin SUM, sigmoid-scaled to [0.8, 3.8])
is realized through this class. `__check_init__` rejects any array coupling -- `epsilon` may be only a
shared scalar or a per-cell *scalar* field averaged by `mix()`, which cannot express a sum-based,
type-aware pairwise matrix. Either the model overrides `mix()` or sources `epsilon` by a route I did not
trace in `potentials.py`. Left as an open uncertainty in the entry, not asserted as a contradiction.

**Did NOT run** the oracle (excavation only; evidence left null for the normalizer/harness).

---

**NORMALIZER verdict: `alias` of `attraction_repulsion` (implementation_of: attraction_repulsion).**
Morse is the pairwise cell-cell interaction the paper actually uses -- repulsive core (excluded
volume) + adhesive tail (adhesion), minimum `-epsilon` at `sigma = r_i + r_j` -- which IS
attraction_repulsion's "long-range pull minus short-range push"; the Morse well and the two-Gaussian
force are two SHAPES of one conservative radial pair force, one written as an energy + autodiff, the
other hand-coded. The parent `PairwisePotential` already landed `alias -> attraction_repulsion` and
named Morse as a subclass implementation, so this is the several-implementations-per-contract pattern,
not a new contract; of all the siblings Morse is the canonical fit (only one with BOTH terms).
**Strongest argument AGAINST (and why it loses):** Morse is a differentiable conservative ENERGY that
yields THREE typed outputs (`total_energy`, `forces`, and a per-cell virial STRESS) through an
end-to-end autodiff pipeline, whereas attraction_repulsion is a hand-coded FORCE law emitting only a
velocity -- so one could argue "declares a learnable energy AND exposes a stress readout" is a
distinct contract shape and call this a `refinement` (widen outputs to energy + stress) or even
`new`. It loses because the energy-vs-force distinction sits BELOW the signature (stillinger_weber is
already an energy-defined interaction registered in the same lateral/interaction family, so
autodiffing an energy mints no contract), and the virial stress is the separate VirialStress
operator's output (it writes a `stress` field), not part of the interaction contract -- no field of
attraction_repulsion's signature is forced to change.

---

**Dispute resolved (normalizer, status -> normalized).** A skeptic challenged `alias` -> claimed
`refinement`, on the grounds that registered `attraction_repulsion` carries a GLOBAL scalar interaction
length (`attraction_repulsion.py:43 self.sigma=float(params["sigma"])`) and reads no per-cell `radius`,
so hosting Morse's per-pair `sigma = r_i + r_j` forces a `+radius` read + per-pair contact distance =
costed widening. Both of the skeptic's RECORD facts are wrong, and the substantive point was already
adjudicated. (1) The parent `pairwise_potential` (order 13) is NOT `verdict: null` -- it is
`verdict: alias -> attraction_repulsion` (atlas_record.yaml:2057; the cited "1964" is the wrong line),
and its own `why:` already ruled "the additive contact rule sigma = r_i + r_j read from radius ...
default-compatible and break no existing user, so they are not a costed widening." (2) The very
"division bar" the skeptic invokes (record L1246-1256) is what DEFEATS refinement here: Division tipped
to `refinement` because it reads a `division_axis` field NO promoted operator has AND writes a `radius`
that flips the growth invariant every caller relies on. Morse only adds a `radius` READ -- a field read
across the language (cell_grow, cell_divide, radius_graph), additive, breaking no caller and minting no
novel dependency -- so it clears neither prong of that bar. Consistent with the whole normalized sibling
family (SoftSphere, Hertzian, Harmonic, LennardJones all `alias -> attraction_repulsion`), Morse stays
`alias`.

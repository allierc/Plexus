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

**Normalizer verdict (revised) — `new`, `implementation_of: adhere`.** (This supersedes an earlier
normalizer pass on this entry that landed `alias → attraction_repulsion`.) Hertzian is the
purely-repulsive, C2-soft, self-truncating member of a pairwise cell-cell CONTACT-MECHANICS contract
whose interaction range is set by the cells' physical size (`sigma = r_i + r_j` read from `radius`, so
excluded volume tracks growth and division), defined by an energy that autodiffs to a force + virial,
with a per-cell (or shared) stiffness. The frozen 42 do not carry that contract — it is exactly the
`adhere` contract SoftSphere (order 15) surfaced. The decisive point: Hertzian and SoftSphere are the
two most similar members in the whole family — the same `_compact_repulsion(r, sigma, eps, exponent,
prefactor)` helper, both purely repulsive, single-param, differing ONLY in `(2.5, 0.4)` vs `(2.0,
0.5)` — so they MUST share a verdict, and SoftSphere's `new → adhere` is already merged. The
registered `attraction_repulsion` cannot host this: verified at source it is the D'Orsogna
self-propelled-particle law (`EMIT="velocity"`, a global scalar `sigma = float(params["sigma"])`,
per-TYPE `p`, edge-graph message passing, no `radius` read, no energy/virial), so aliasing overstates
coverage and widening it to size-consistent per-cell energy-mechanics would delete its defining
biology. Recording Hertzian as `implementation_of: adhere` adds ZERO contracts (the family yields ONE
new contract with several implementations), so this is not yield-inflation. **Strongest argument
against:** the family PLURALITY runs the other way — the abstract parent `PairwisePotential` (13),
`Morse` (14) and `Harmonic` (17) all landed `alias → attraction_repulsion`, judging the radius-read,
the energy→autodiff-force strategy, and the dense-vs-graph topology to be sub-signature implementation
axes (stillinger_weber is already an energy-defined member of this family; squared_law already carries
both all-pairs and a graph; attraction_repulsion's own force is the gradient of a radial potential).
If they are right, then `attraction_repulsion` really is "the conservative radial pull-minus-push
contract" abstractly, `adhere` is a spurious mint that fractures a family the parent explicitly
unified and inflates the exact `new` count the ledger measures — and `adhere` is a poor biological
name for a member that is PURELY REPULSIVE and never adheres. I still choose `new` because the
registered operator's `reads` genuinely lacks `radius` (aliasing asserts coverage the signature does
not have) and because splitting the near-identical twins SoftSphere/Hertzian is indefensible; but the
right resolution is family-wide (pull parent + Morse + Harmonic toward `adhere`), and that
reconciliation is flagged for the analysis phase.

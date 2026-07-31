<!-- SoftSphere -- append below; the driver merges this into campaign/analysis.md -->

# SoftSphere (potentials.py:L315)

**What I read.** The whole `potentials.py`: the `Potential` protocol (`total_energy` ->
autodiff `forces` -> `state_reads`), the `PairwisePotential` base (which supplies `total_energy`,
`forces`, `virial_pressure`, the coupling machinery `_coupling`/`mix`/`__check_init__`, and the
`neighbor_sum` reduction), and all six leaf potentials so I could place SoftSphere against its
siblings. `SoftSphere` itself is tiny: `epsilon` (scalar or per-cell spec), `pair_params` returns
`(sigma = r_i+r_j, eps)`, and `pair_energy` is one call to the shared
`_compact_repulsion(r, sigma, eps, 2.0, 0.5)` -> `U = 0.5*eps*(1-r/sigma)^2` for `r<sigma`, else 0.
Everything real lives in the base and the two module helpers `_compact_repulsion` / `_smooth_cutoff`.

**What surprised me (the headline).** The paper does NOT use a harmonic soft sphere. Its cell
mechanics is the **Morse** potential (repulsion + adhesion tail), p. 9 and repeated p. 14:
`V_ij = eps[(1-exp(-alpha(r-sigma)))^2 - 1]`, `sigma = R_i+R_j`. The phrase "adhesive soft spheres"
(p. 4 intro, p. 9 methods) is the paper's *descriptor* for that Morse model, not a literal harmonic
`(eps/2)(1-r/sigma)^2`. So `SoftSphere`-the-class is a purely-repulsive library alternative the
paper describes in name only and does not use for any result. Per the source-wins rule I recorded
both readings in `equations:` and `paper_section:` and flagged the trap in `surprises:`.

**Smaller surprises worth flagging for a reimplementer.** (1) `sigma` is the SUM of radii, not the
mean. (2) The energy has a baked-in `eps/2` prefactor. (3) There are TWO independent 0.5s (the
harmonic prefactor and the pair-double-count in `total_energy`). (4) The double-`where` in
`_compact_repulsion` looks pointless at exponent 2 but exists to keep the fractional-exponent
siblings (Hertzian 5/2) NaN-free under always-on `jax_debug_nans`. (5) per-cell `epsilon` mixes by
ARITHMETIC mean while `sigma` mixes by SUM. (6) forces are autodiff of the energy, not hand-coded.

**Placement (context, not a verdict).** The nearest promoted contract is
`src/plexus/operators/attraction_repulsion.py` (D'Orsogna double-Gaussian, a first-derivative
*velocity* law on an edge graph, receiver-type params, mean aggregation). That is a different object
from this: a purely-repulsive HARMONIC excluded-volume defined by an ENERGY, with contact distance
from radii, autodiff force, and a virial-pressure readout. Whether that counts as a new contract is
the normalizer's call; I left `verdict`/`of`/`contract` null and set `status: inspected`.

**What I did NOT establish.** I read the code statically only -- I did NOT run the oracle to
confirm the numeric energy/force (jax is deliberately absent here; no `oracle_run`). I did not trace
which concrete step wraps `SoftSphere` in any shipped model (I assert the general
Potential->Step relationship from the base/protocol, but did not grep for a model that instantiates
`SoftSphere`). And I did not verify whether the ORIGINAL Deshpande code (vs this hardened
reimplementation) even contains a `SoftSphere` class, or whether it is a Plexus-library addition --
the paper text alone cannot settle that.

---

## Normalization

**Verdict: `new`, `implementation_of: adhere`.** SoftSphere is the purely-repulsive
(zero-adhesion) member of the pairwise cell-cell mechanical-interaction contract `adhere` -- a
Lateral force whose range is set by the two cells' physical radii (contact at sigma = r_i + r_j)
and which drives cell positions through a wrapping integrator step. Following the record's own
`regulate` rule, Morse / SoftSphere / Hertzian share one typed signature and differ only in the
pair-energy law U(r), so they collapse to ONE contract with several implementations, not three;
`adhere` is absent from the frozen 42, hence `new` rather than alias/refinement (R4 rejects a
non-registered `of:`), and the instruction's own gloss -- "`morse_potential` is an implementation
of `adhere`" -- fixes the name. The closest registered slots lose their biology if widened:
`attraction_repulsion` is a fixed-global-width D'Orsogna velocity law with per-type params and no
cell radius, and `separation` is a mean-aggregated 1/|d|^2 boids steering nudge with no contact
distance, energy, or virial.

**Strongest argument against.** SoftSphere has *zero* adhesion -- it is excluded volume and nothing
else -- so calling it an implementation of `adhere` mis-names a purely repulsive cell gas, and one
could argue steric repulsion is its own biological contract (`exclude_volume`) distinct from
adhesion: a cell can exclude volume without adhering, and the two are separable affordances that the
paper's Morse merely happens to bundle. I reject the split because the atlas counts CONTRACTS = typed
signatures, and SoftSphere and Morse have an IDENTICAL signature (read position/radius/epsilon, drive
position via the force slot, expose the same virial) -- they differ only in U(r), which is exactly
the "differ only in the vector field f -> one contract" rule the record already committed to for
`regulate`. Adhesion strength is a knob whose zero limit recovers SoftSphere, not a separate
contract; splitting on the presence of the attractive tail would inflate the contract yield on an
implementation detail -- the precise failure mode the measurement exists to avoid. (If a later pass
finds adhesion carries state or a map that repulsion does not -- a cadherin field, a bond edge-set --
that would be real signature divergence and would reopen the split; the harmonic/Morse pair shows
none.)

---

## Normalization (dispute pass)

**Verdict stands: `new`, `implementation_of: adhere`; status -> normalized.** The skeptic disputed
this as `alias of attraction_repulsion`, invoking "record.py's regulate/consistency rule". Two things
settle it against the skeptic. First, that rule *does not exist*: record.py enforces R0-R12 and none
force siblings to share a verdict -- `validate()` judges each mechanism independently, so the earlier
note's own appeal to a "regulate rule" (and the sibling entries' appeal to the parent) is not a rule
of this ledger. Second, R4 requires an alias `of:` to be a *registered* contract, and the only
candidate -- `attraction_repulsion` -- factually does not cover this: its promoted source has a GLOBAL
scalar `sigma` (attraction_repulsion.py:43), per-TYPE `p` (REQUIRES_TYPE_PROPS), `EMIT=velocity`,
`set=particle`, and no `radius` read, hence no size-consistent per-pair contact and no energy/virial.
The skeptic's own settling test (does attraction_repulsion read radius / use a global sigma?) is thus
met *against* alias. That also means the sibling Morse/Hertzian/Harmonic `alias` verdicts are the
mis-normalization, not this one -- their own skeptics flagged refinement/new.

**Strongest argument against (the honest one, post-dispute).** Not `alias` -- that's refuted by the
source -- but `refinement of attraction_repulsion`: since attraction_repulsion IS the registered
"attraction + repulsion" pairwise-interaction slot and I agree it does not yet cover SoftSphere, the
minimal move is to *widen* it (add a `radius` read, allow a per-pair additive `sigma = r_i + r_j`,
allow a per-cell coupling) rather than mint a second interaction contract -- keeping ONE interaction
force in the language and paying the widening cost openly. I still reject it for `new` because the
widening is not bounded to a field or two: it changes `set` particle -> cell, re-sources the
interaction range from a free global knob to a physical consequence of cell size, and swaps per-type
for per-cell programming -- which deletes the fixed-width, type-programmed D'Orsogna self-propelled
particle model that IS attraction_repulsion and breaks every existing user. Widening that does
violence to the contract's biology is exactly what the record distinguishes from a refinement, so the
size-consistent cell-cell mechanical interaction is a genuinely new contract `adhere`, carrying all
six pair potentials as implementations (one new contract, not six).

<!-- LennardJones -- append below; the driver merges this into campaign/analysis.md -->

# LennardJones (potentials.py:L419)

Read the whole `potentials.py`: the `Potential` protocol, the `PairwisePotential` base (which
supplies `total_energy`, autodiff `forces`, and `virial_pressure` for free), and all five concrete
potentials, plus the shared helpers `_smooth_cutoff`, `_compact_repulsion`, `safe_divide`,
`safe_norm`, `neighbor_sum`, `pairwise_displacements`. LJ is one of the thinnest subclasses: it only
declares `epsilon` + two cutoff fractions, `_couplings`, `_check_config`, `pair_params`, and
`pair_energy`. It is an ENERGY component, not a step -- it writes no state; a wrapping step
(relaxation / Brownian) applies its forces to `position`.

Biggest surprise, and the whole point of this entry: **the paper never uses Lennard-Jones.** Every
mechanical interaction in Deshpande 2025 is the **Morse** potential (p. 9 "Mechanical interactions";
Methods p. 14). `grep` for "Lennard" in the paper text returns nothing; LJ is a source-only
alternative. Per the loop rule, source wins and I recorded the divergence in `surprises:` and
`paper_section:`.

Second surprise: it's the `r_min` LJ form `eps*((sigma/r)^12 - 2 (sigma/r)^6)`, NOT the textbook
`4 eps ((sigma/r)^12 - (sigma/r)^6)`. Here `sigma = r_i + r_j` is the well MINIMUM (contact
distance), value `-epsilon`, not the zero-crossing -- a reimplementer using the 4-eps form gets a
silently-wrong well location/depth. The smooth cutoff multiplies the whole energy (inert on the core
only because `r_on = 1.5*sigma > sigma`), and per-cell `epsilon` mixes by ARITHMETIC mean, not the
geometric/Lorentz-Berthelot rule most LJ codes assume.

Could NOT establish: (1) whether any downstream Plexus operator or oracle script actually
instantiates `LennardJones` -- I only confirmed absence from the paper, not from the library's own
examples/tests, so "unused" is a paper claim, not a library-wide one. (2) I did not run the oracle
or numerically verify the minimum/cutoff continuity; the equations are read from source, not
executed (jax is deliberately absent here). (3) I did not chase which concrete step declares the
`state_writes` that consume these forces -- I inferred the consumers (relaxation/Brownian/virial)
from the base-class docstrings, not from reading those steps.

---

## NORMALIZER note

**Verdict: `refinement` of `attraction_repulsion` (implementation_of: attraction_repulsion).** LJ is
the 12-6 well SHAPE of the one pairwise radial cell-cell interaction the registry already carries --
a hard r^-12 core plus a -2 r^-6 adhesive tail, minimum -epsilon at contact `sigma = r_i + r_j` --
which IS attraction_repulsion's pull-minus-push biology, so `implementation_of` is
attraction_repulsion. It is `refinement` and not `alias` because the FROZEN operator has a single
GLOBAL scalar interaction length (`attraction_repulsion.py:43 self.sigma = float(params["sigma"])`)
and reads no per-cell radius, whereas LJ's contact distance is per-pair and radius-coupled
(`sigma_ij = r_i + r_j`, potentials.py:452) and enters the force-law shape itself -- so admitting LJ
ADDS a `radius` read and generalizes the interaction length to the additive per-pair rule. That is
the same signature-growth the record already accepted as refinement for `cell_divide`
(+division_axis) and `cell_grow` (+growth_rate), and it is the reading the record's skeptic imposed
on the twin `morse` (refuting alias -> refinement on exactly this argument). Of the family LJ is the
strongest refinement case: Hertzian (also refinement) shows the radius coupling only in the
purely-repulsive limit, whereas LJ exercises it in the complete core+tail adhesive well.

**Strongest argument AGAINST (and why it loses):** the two other full core+tail members, `morse` and
`harmonic`, both landed **alias**, and LJ differs from Morse only in the well's functional form
(12-6 vs exponential) -- the textbook interchangeable-implementation case -- so by nearest-twin
consistency LJ "should" be alias too; the alias camp argues the `radius` read is additive and
opt-in (a null-radius / global-sigma path recovers the exact D'Orsogna force), breaking no existing
user, hence default-compatible rather than a costed widening. It loses on three source-checkable
grounds: (a) the frozen operator genuinely does NOT read radius and its sigma is a global scalar
(attraction_repulsion.py:32,43), so the read must truly be ADDED -- a signature change, not a
field the alias contracts glossed as already-present; (b) the record's standing bar (division and
cell_grow, both refinement) treats *adding a read to a frozen signature* as refinement even when
additive/opt-in, and "breaks no existing user" is exactly the deflationary comfort that bar rejects;
(c) the record's own skeptic already refuted this alias reading on the sibling `morse`
(confidence 0.68). The honest cost is that this puts LJ at odds with `morse`/`harmonic` as they
currently stand -- but those are the entries the skeptic flagged, not the Hertzian/refinement line I
follow; the alias label under-reports a real gap (attraction_repulsion cannot, as registered,
express a size-coupled per-pair contact distance), and surfacing that gap is the measurement this
ledger exists for. (Source vs paper: LJ appears nowhere in the paper -- Morse is the paper's only
mechanical potential -- source wins, recorded, verdict unchanged. Oracle not run; jax is absent and
python is sandbox-blocked here, so the entry was checked by inspection against record.py's rules;
the driver runs the validator.)

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
the 12-6 SHAPE of the one pairwise radial cell-cell interaction the language already carries -- a hard
r^-12 excluded-volume core minus a 2 r^-6 adhesive tail, minimum -epsilon at the contact distance
`sigma = r_i + r_j`, force = -grad of that conservative radial energy by autodiff -- which IS
attraction_repulsion's pull-minus-push biology, so its home and closest contract are unambiguous. Not
`new`: the biology is a pairwise radial law, the frozen language already holds an energy-defined
interaction in this family (stillinger_weber), and minting a second pairwise-attraction-repulsion
contract inflates the very yield the ledger measures (this also rejects SoftSphere's `new -> adhere`,
order 15, as over-minting). But NOT a clean `alias`, and that is the entry: the PROMOTED operator
(src/plexus/operators/attraction_repulsion.py, the D'Orsogna model) reads pos/edge_index/node_type/occ
but **never `radius`** (52-61), makes a single GLOBAL scalar length UNCONDITIONALLY required
(`REQUIRES_PARAMS=["sigma"]`, `self.sigma=float(params["sigma"])`, 32/43), and couples **per-TYPE**
(`REQUIRES_TYPE_PROPS=["p"]`, 33/61). Hosting LJ therefore (a) grows the declared reads by `radius`,
(b) relaxes the "exactly one global interaction length" guarantee to a per-pair additive
`sigma=r_i+r_j`, and (c) generalises the coupling from type-indexed to a per-cell field. By the
record's OWN normalized bar that is refinement, not alias: `cell_divide` and `cell_grow` (both
refinement/normalized) establish that GROWING the declared read set -- "a dependency existing
schedulers did not track" -- or FLIPPING an invariant is a costed widening even when additive. LJ does
exactly that; the pure-D'Orsogna path still runs untouched, but the signature and its guarantee change,
and a refinement nobody costed is a breaking change. Source vs paper (rule 5): LJ is in NO paper
experiment -- Morse is the paper's only mechanical potential -- source wins, recorded, verdict
unchanged.

**Strongest argument AGAINST (and why it is close).** The abstract parent `pairwise_potential` (order
13) is already NORMALIZED as plain **alias -> attraction_repulsion**, and its normalized `contract:`
block ALREADY lists `radius` in reads and `sigma=r_i+r_j` in maps while ruling exactly these fields
"default-compatible and break no existing user ... not a costed widening." If that normalized parent
signature -- not the raw registered operator -- is the ledger's operative contract, then the widening
has already been paid and LJ is a clean ALIAS of it; re-costing it as refinement double-charges a
settled result, and the family's landed majority (parent + Morse + Hertzian + Harmonic all alias)
agrees. I resist this because rule 4 pins `of:` to the PROMOTED code in `plexus.operators`, and that
operator demonstrably cannot express a size-coupled per-pair contact distance (no radius read, a
required global sigma) -- the parent's contract block is a PROPOSED signature the frozen language has
not adopted, so against the real baseline the widening is still owed. That is the genuine tension and I
do not pretend it away: if the parent's normalized signature governs, alias is correct; against the
frozen operator, refinement is. I land refinement because it is the only verdict that neither hides the
missing `radius`/per-pair-`sigma` capability (alias) nor mints a whole redundant contract for a
different well shape (adhere/new). (Oracle not run: jax is absent here and `python` is sandbox-blocked,
so the entry was checked by inspection against record.py's twelve rules; the driver runs the validator
on merge.)

---

## NORMALIZER note (landed verdict -- supersedes the refinement draft above)

**Verdict: `new`, `implementation_of: adhere`.** I verified the registered operator directly at the
source: `src/plexus/operators/attraction_repulsion.py` is a D'Orsogna self-propelled-**particle**
velocity law -- `set="particle"`, `EMIT="velocity"`, a required **global scalar** `sigma`
(32/43), **per-TYPE** `p` (33/61), reads `pos/occ/edge_index/node_type` and **never `radius`**
(52-61), no energy, no virial. So `alias` is factually false, and hosting LJ is not a bounded field
add: it changes `set` (particle->cell), swaps a velocity law for an energy whose grad is the force,
re-sources the interaction range from a free global knob to a physical `sigma=r_i+r_j` read from
radius, swaps per-type for per-cell coupling, and grows an energy output -- deleting the D'Orsogna
model. Widening that does violence to the contract's biology, which is the record's `new` test, not
refinement. The six pair potentials share ONE signature and differ only in `U(r)`, so they are ONE
new contract `adhere` with several implementations -- one new contract for the family, not six, which
is the several-implementations result the ledger wants, not inflation. LJ, being adhesive, fits
`adhere` more directly than the repulsion-only SoftSphere that already landed `new -> adhere` (order
15, dispute-survived); I align with it rather than the source-false alias majority.

**Strongest argument AGAINST (and why I still reject it).** The honest counter is not `alias` but
`refinement of attraction_repulsion` (my predecessor's draft above): `attraction_repulsion` IS the
registered "attraction + repulsion" pairwise slot whose *name* matches this pull-minus-push biology,
and the record already accepts refinements that merely grow the declared read set (`cell_divide`,
`cell_grow`) -- so the minimal, non-inflating move is to widen it and keep ONE interaction force in
the language rather than mint a parallel `adhere`, which risks two contracts ("radial pull-minus-push
pair force") that a reader would struggle to tell apart. This is genuinely close, and if the ledger
ever treats a *proposed/normalized* signature (the parent `pairwise_potential` contract block already
lists `radius` + `sigma=r_i+r_j`) as the operative baseline, refinement -- or even alias -- becomes
correct. I reject it because rule 4 pins the comparison to the PROMOTED code, and against that code
the change is not one field but a rewrite of set/emit/coupling/range-ontology plus a new energy
readout, which deletes the fixed-width, type-programmed D'Orsogna particle model and breaks its every
user -- the definition of `new`, not a costed widening. (Oracle not run: jax absent, `python`
sandbox-blocked; entry checked by inspection, driver runs record.py on merge.)

---

## IMPLEMENTER note

**Built:** `src/plexus/operators/candidates/jax_morph_lennard_jones.py` -- the `lennard_jones`
implementation of the `adhere` contract (`@register_operator("adhere", family="interaction",
set="cell", kind="lateral", implementation="lennard_jones")`), a `Lateral`, `EMIT="velocity"`.
Torch, not JAX. It imports/registers cleanly under the neural-graph env.

**Design (ported from the near-identical sibling `jax_morph_hertzian.py`, since the family shares
one signature and differs only in `U(r)`).** The operator builds the total pair energy over live
non-self cell pairs and takes the force by AUTODIFF (`force = -grad_positions E`), exactly as the
source's base does (`forces = -jax.grad(total_energy)`; there is no analytic LJ force in the source
either). The only member-specific pieces vs Hertzian:
* pair energy is the **r_min 12-6 form** `eps*(q^12 - 2 q^6)` with `q = safe_divide(sigma, r)`
  (NOT `r/sigma`), times a **sigma-relative smooth C1 cutoff** `_smooth_cutoff(r, 1.5 sigma,
  2.5 sigma)` that multiplies the WHOLE energy;
* two tunable cutoff fractions `r_onset_frac`/`r_cutoff_frac` with the source's construction-time
  check `r_onset_frac < r_cutoff_frac` (raises ValueError otherwise);
* `_smooth_cutoff` ported verbatim from potentials.py:L42, with a `safe_divide` on its
  `(r_off^2 - r_on^2)^3` denominator to guard the dead-dead (sigma=0) pair.
Shared with Hertzian and kept faithful: `sigma = r_i + r_j` (additive, from a `radius` buffer, with
a scalar fallback); shared-or-per-cell `epsilon` mixed per pair by the ARITHMETIC mean; the 0.5
half-sum so each unordered pair counts once; an EXTERNAL alive+non-self pair mask (the diagonal and
dead pairs already evaluate to a clean 0 under the LJ law, but the mask keeps the `neighbor_sum`
seam identical across the family); `safe_divide`/`safe_norm` double-`where` guards so the r=0
diagonal keeps a finite gradient (a naive sigma/r would NaN a masked pair via 0*inf in backward);
overdamped `velocity = mobility * force`; the `create_graph = outer_grad` autodiff so the force
stays connected under a differentiable rollout.

**Test:** `tests/test_jax_morph_lennard_jones.py`, 11 properties, all stated WITHOUT the reference
(none can be passed by fitting the oracle's numbers); all 11 pass. The headline is the **r_min
discriminator**: two cells at exactly `sigma = r_i + r_j` feel ZERO force (well at contact), and the
force is clearly nonzero at `2^(1/6) sigma` -- so a reimplementer who coded the textbook
`4 eps ((sigma/r)^12 - (sigma/r)^6)` form (well at `2^(1/6) sigma`) FAILS this test. Also checked:
contact is a stable equilibrium (repel inside, adhere just outside; restoring sign flips across
contact), the r^-12 core strengthens monotonically with overlap, Newton's third law (two-body
equal-and-opposite), momentum conservation (net force ~0 over a jittered grid), the smooth cutoff
(exactly 0 beyond `2.5 sigma`, nonzero adhesive tail inside the window), size-consistency (doubling
both radii moves the rest separation 1.0 -> 2.0), the `r_onset_frac < r_cutoff_frac` guard,
dead/masked cells emit nothing, position is not mutated, and 3-D genericity.

**Test gotcha worth recording:** the first momentum-conservation draft used fully random positions
and FAILED -- not a physics bug (Newton's third law passes pairwise) but float32 roundoff: random
placement put cells in deep overlap where the r^-12 core hits ~1e8 and the exact antisymmetric
cancellation is lost at that magnitude. Fixed by using a jittered contact-scale grid (no deep
overlaps) and a RELATIVE tolerance (`net.norm() < 1e-4 * v.norm()`), which is the honest statement:
the residual is roundoff, not net propulsion.

**Could NOT establish / out of scope:** (1) I did NOT run the oracle or diff against the JAX
reference -- jax is deliberately absent from the Plexus env, and (as the reader/normalizer notes
record) NO paper experiment or oracle script instantiates LennardJones, so there is no reference
trajectory to diff against; the differential test is the curator's next step, and `evidence:` stays
null. (2) The tests confirm the operator's own stated invariants, not numeric agreement with the
source (deliberately -- a fitted constant would teach us nothing). (3) The `epsilon_field`
(per-cell well depth) branch is implemented (mirrors Hertzian) but only the shared-scalar path is
exercised by the tests. (4) Per-cell VIRIAL PRESSURE (the base's other consumer of this energy) is
out of scope here, consistent with the sibling entries. Verdict left as the normalizer landed it
(`new`, `implementation_of: adhere`); implementing does not change it.

---

## DIFFER note -- differential test RAN and PASSED (supersedes "evidence stays null")

The earlier "could not establish" note said there was no reference to diff against because jax is
absent from the Plexus env. That was wrong about the mechanism: the oracle runs jax-morph in its
OWN isolated venv (`oracle.py`, jax-morph 0.4.0 / jax 0.11.0, git ace08b8), so a real reference
trajectory exists. It has now been built and diffed.

**Isolation design.** LennardJones is an ENERGY (writes no state; its whole contract is
F = -grad U), so it is isolated from every other mechanism and driven by the reference's OWN
overdamped Langevin step at kT=0: `BrownianDynamics(LennardJones(epsilon=1.0), gamma=1, kT=0)`,
whose update dx = dt*forces/gamma is exactly the Plexus engine's overdamped Euler of the emitted
velocity (mobility = 1/gamma = 1). IC = SIX well-separated dumbbell pairs (centres 5 apart,
> 2.5*sigma, no cross-pair force) at separations [1.15,1.25,1.35,1.50,1.70,2.60]*sigma + 4 dead
padding slots. A purely-adhesive many-cell blob is geometrically frustrated (collapses into the
r^-12 core, explicit integrator explodes -- verified in _probe_lj); the dumbbells relax
monotonically to contact, sweeping the adhesive tail + cutoff ramp + beyond-cutoff, which is the
LJ-DISCRIMINATING regime the repulsion-only siblings lack.

**Metric / threshold (pre-registered).** D_pos = max over all 101 frames and 12 live cells of
||x_plx - x_ref||_2 / sigma. Threshold 1.0e-3 sigma, pre-registered at
`_oracle/scripts/_analyze_lennard_jones.py:83` before the diff was computed.

**Result: D_pos = 0.0 -- PASS (byte-equal).** The Plexus and jax-morph float32 trajectories are
BYTE-EQUAL over all 101x16x2 positions (np.array_equal True). Verified NOT an aliasing artifact:
the two arrays are distinct objects, diverge from the SoftSphere negative control by 0.350 sigma,
and the cells genuinely moved 0.350 sigma. This is the strongest pass in the pair-potential family
(soft_sphere 2.7e-6, harmonic 7.9e-6, hertzian 1.9e-6 were all non-zero, proving 1e-3 is not a
bit-identity gate).

**Corroborators.** frame-0 == pristine IC both sides (0.0/0.0); misaligned x_plx(t) vs x_ref(t-1)
= 0.0268 sigma (27x the bar); per-pair separation trajectory agrees to 0.0 (five pairs -> contact
1.0, sixth frozen at 2.60); negative control (adhesion-off SoftSphere) diverges 0.350 sigma;
dead slots frozen 0.0/0.0; the 9.4e-05 single-step "force residual" is an ORACLE-INTERNAL float32
reassociation (plx and ref steps both differ from the standalone forces(IC) by the SAME 9.4e-07,
i.e. equal each other); oracle self-guard: 2-cell scan vs analytic r_min force 4.6e-06, force
exactly -0.0 at contact and beyond 2.5 sigma, kT=0 deterministic across two PRNG keys, first Euler
step == dt*forces bit-for-bit.

**Acted ledger.** adhere calls 100 / acted 47, moved 2.679 cumulative, valid_evidence true. The
operator acts through the active phase and falls below the move threshold once the pairs reach
contact (~frame 42; gyration 8.5795->8.5678 and nn-distance 1.5917->1.2667 both plateau) -- the
overdamped fixed point, a real run not a still life.

**Scope / what this does NOT prove.** Uniform radius 0.5 (SUM-vs-mean size-consistency at
heterogeneous r_i != r_j is untested here -- carried on the hertzian twin) and shared epsilon
(per-cell arithmetic-mean mix confirmed by _probe_eps but not exercised in the trajectory). The
per-cell VIRIAL pressure the base derives from the same energy is out of scope (separate
VirialStress mechanism). This validates the IMPLEMENTATION reproduces the reference force + its
overdamped dynamics to byte equality; it does not touch the verdict (new -> adhere), which stands
on the signature argument in `why`.

**Runs.** oracle `atlas_jax_morph/_oracle/runs/diff_lennard_jones/` (reference.npz + summary.json);
plexus `log/atlas_jax/lennard_jones/` (spec_run.yaml, diag.json, metrics.json/.npz, strip.png,
movie.mp4); diff `log/atlas_jax/lennard_jones/diff.json`. **Verdict: status validated, D_pos 0.0 <
1e-3, passed.**

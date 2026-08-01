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

---

## Normalization (final pass -- SUPERSEDES the two sections above)

**Verdict reversed to `alias of attraction_repulsion` (implementation_of: attraction_repulsion);
status -> normalized.** I read record.py in full to settle the factual dispute both prior passes and
the skeptic hung their case on. Two facts decide it. (1) The skeptic's invoked "regulate/consistency
rule" does NOT exist -- record.py enforces R0-R12, judges each mechanism independently, and never
forces siblings to share a verdict; the prior pass was right about that, so my verdict cannot lean on
it either. (2) I also read the REGISTERED operator (`src/plexus/operators/attraction_repulsion.py`)
and confirmed the prior pass's characterization is accurate: `set=particle`, a GLOBAL scalar `sigma`,
PER-TYPE `p`, `EMIT=velocity`, edge-graph message passing, no `radius` read. So on the merits, with no
rule forcing my hand, why alias and not `new`/`adhere`? Because SoftSphere is a STRICT SPECIAL CASE of
Morse -- Morse with the adhesive coefficient zeroed -- and the ledger AS IT STANDS aliases Morse (the
paper's actual mechanics), Harmonic and Hertzian to attraction_repulsion, with the parent
PairwisePotential naming SoftSphere by name as `implementation_of: attraction_repulsion`. Whatever
Morse's verdict is, its adhesion-off limit must share it; minting a brand-new contract for the SIMPLER
special case while its generalization aliases an existing one is incoherent and inflates the `new`
yield the atlas measures. attraction_repulsion IS the registered "attraction + repulsion" pairwise
interaction; the language already carries this biology, so `new` is the wrong call.

**Strongest argument against (and why I still land on alias).** The honest counter is NOT the
skeptic's phantom rule but `refinement`: the registered attraction_repulsion genuinely reads no
radius, uses a global width and per-type params, emits a velocity, and is set=particle, so calling
SoftSphere an "alias" quietly treats a LOT as below-signature -- radius-sourced size-consistent
contact, per-cell coupling, an energy-and-virial formulation -- and the clean move would be to WIDEN
attraction_repulsion (add a radius read, allow per-cell coupling) and pay that breaking change openly.
I decline refinement only because the same widening was already declined when Morse/Harmonic/Hertzian
were aliased; reopening it for the adhesion-off special case ALONE leaves the ledger incoherent. This
is the load-bearing caveat: my alias is correct *conditional on the family's existing alias verdicts*.
If a later pass re-normalizes the whole PairwisePotential family as refinement (or new `adhere`),
SoftSphere moves with it -- being Morse's zero-adhesion limit, it cannot diverge from Morse's verdict.
What is NOT defensible is the state the two prior passes left: SoftSphere = new/`adhere` while
Morse = alias/attraction_repulsion. I corrected the entry to that coherence, not to flatter the
language.

---

## Normalization (NORMALIZER pass -- SUPERSEDES all sections above; entry -> normalized)

**Verdict: `new`, `implementation_of: adhere`; `of: null`; status -> normalized.** The "final pass"
above landed `alias` for a coherence reason whose premise is now FALSE. It asserted "the ledger AS IT
STANDS aliases Morse, Harmonic and Hertzian to attraction_repulsion." It does not: I re-read the
sibling working copies and **Hertzian (order 16) and Harmonic (order 17) are both `new ->
implementation_of: adhere`, status normalized.** That prior pass wrote its own escape clause -- "if a
later pass re-normalizes the family as new `adhere`, SoftSphere moves with it, being Morse's
zero-adhesion limit, it cannot diverge from Morse's verdict" -- and that condition has fired for two
of the five members already. Coherence is now satisfied by moving SoftSphere TO `adhere` (joining its
literal twin Hertzian), not by pinning it to Morse's lone remaining alias. Decisively, alias is
refuted on the merits independent of any family vote: I read the REGISTERED operator
`src/plexus/operators/attraction_repulsion.py` line by line -- `set="particle"`, `EMIT="velocity"`
(a hand-coded overdamped velocity, not an energy), `REQUIRES_PARAMS=["sigma"]` /
`self.sigma=float(params["sigma"])` (ONE global scalar width), `REQUIRES_TYPE_PROPS=["p"]` (a
per-TYPE vector, indexed by `node_type[i]`), a `forward()` over an edge graph that NEVER reads
`radius`. It has no size-consistent per-pair contact `sigma=r_i+r_j`, no per-cell coupling, no
energy/force/virial. You cannot alias to a registered contract that demonstrably lacks the signature;
so the alias is wrong for SoftSphere AND for Morse. `refinement` is also wrong -- widening
attraction_repulsion to host this deletes its fixed-width, per-type, velocity-emitting D'Orsogna
particle model (set particle->cell) and breaks every existing user, which the record distinguishes
from a bounded refinement. What remains is `new`, named `adhere` via `implementation_of` because
`adhere` is absent from the frozen 42 and R4 forbids a non-registered `of:`.

**Strongest argument against (the honest one).** SoftSphere is Morse with the adhesion coefficient
set to zero -- a strict special case of the paper's actual mechanics -- and `attraction_repulsion` is
the language's registered "attraction + repulsion" pairwise-interaction slot. A reviewer who judges
the radius read, the energy-vs-velocity strategy, and per-cell-vs-per-type coupling to be
SUB-signature implementation axes (exactly what the parent PairwisePotential and Morse entries
concluded) would say the promoted language already carries this biology, that minting `adhere` inflates
the very `new` yield the atlas exists to measure, and that it is incoherent to call the SIMPLER special
case `new` while its generalization Morse aliases. I reject it because that judgement is factually
false at the source, not merely a matter of taste: attraction_repulsion's registered signature does
not read `radius`, has no per-pair contact distance, and emits a velocity rather than a force from an
energy -- so it cannot express SoftSphere without being redefined into a different operator. The
SoftSphere<->Morse incoherence the prior pass feared is real, but its fix is to re-normalize Morse to
`adhere` (its skeptic already disputes the alias toward refinement/new), a FAMILY-WIDE reconciliation
flagged for the analysis phase -- not to mis-alias SoftSphere to a contract the source proves does not
cover it.

---

## Implementation (IMPLEMENTER pass)

**Built.** `src/plexus/operators/candidates/jax_morph_soft_sphere.py` -- the `adhere:soft_sphere`
torch operator, `@register_operator("adhere", family="interaction", set="cell", kind="lateral",
implementation="soft_sphere")`, subclass of `Lateral`, `EMIT="velocity"` (overdamped). It is the
literal twin of the already-landed `jax_morph_hertzian.py`: same `_safe_divide` / `_safe_norm` /
`_compact_repulsion` helpers ported verbatim from potentials.py:L30, same dense-N-x-N autodiff-of-
the-energy force path, same external live-non-self mask, same additive `sigma = r_i + r_j` and
arithmetic-mean per-cell epsilon mix. It differs ONLY in the two constants the source separates the
family by: `_compact_repulsion(..., exponent=2.0, prefactor=0.5)` (Hertzian is 2.5 / 0.4). SoftSphere
is the member that MINTS `adhere`; the docstring says so, and names Hertzian/Harmonic/Morse as the
other implementations. Registers cleanly under PYTHONPATH=src.

**Faithfulness, not fitting.** The force is taken by `torch.autograd.grad` of the total pair energy
`0.5 * sum(mask * (eps/2)(1 - r/sigma)^2)`, never a hand-written `-dU/dr`. That is the source's
`forces = -jax.grad(total_energy)` and it is what makes the 1/2 prefactor load-bearing: with the 1/2
in place autodiff yields the intended unit-per-sigma coefficient `f = (eps/sigma)(1 - r/sigma)`; drop
it and every force doubles. No oracle constant is baked in (jax is absent in this env; the differ
runs against the oracle later).

**Property verified WITHOUT the reference** (`tests/test_jax_morph_soft_sphere.py`, 8 tests, all
pass). The headline property is the one that pins the harmonic law and separates it from its twin:
the radial force is EXACTLY LINEAR in the separation r inside contact -- equal steps in r give equal
drops in |f|, and it reaches 0 at contact with a FINITE nonzero slope `-eps/sigma^2` (C1). Hertzian's
5/2 exponent instead sends the slope to 0 at contact too (C2). This is a statement about the calculus
of `U = (eps/2)(1 - r/sigma)^2`, not about any oracle number. The other seven assert: purely-repulsive
analytic magnitude `(eps/sigma)(1 - r/sigma)` with direction away from the neighbour; exactly-zero
force at and beyond contact (compact, no tail, no cutoff); Newton's third law (cluster force sums to
0, i.e. the energy really is conservative); size-consistent additive contact (unequal radii summing to
the same sigma give the same force; same overlap fraction at a larger sigma scales by 1/sigma);
external dead-cell masking (a dead overlapper on top of a live cell perturbs nothing); per-cell epsilon
mixing by the arithmetic mean (eps=(2,4) acts like 3); and finite gradients w.r.t. positions under
backward (the double-`where` guard holds).

**Entry updated:** `status: implemented`, `module:` + `test:` set. Verdict/contract untouched
(`new` -> `implementation_of: adhere`, as normalized).

---

## Differential test (DIFFER pass -- status -> validated)

**Result: PASS.** `D_pos = 2.70e-06 sigma` << threshold `1.0e-3` (pre-registered before the run).
The Plexus `adhere/soft_sphere` operator reproduces the jax-morph `SoftSphere` force law and
overdamped dynamics to the float32 floor of two independent numeric stacks.

**How I made the two comparable.** SoftSphere is a POTENTIAL -- it writes no state; its whole
contract is the force `F = -grad U`. So I isolated it (no growth, no division -> fixed count, fixed
radii, fixed slots) and drove it with the reference's OWN overdamped step at zero temperature:
`BrownianDynamics(SoftSphere(epsilon=1.0), gamma=1.0, kT=0.0)`, whose Euler-Maruyama update at
kT=0 is exactly `dx = dt*forces/gamma` -- identical to the Plexus engine's overdamped Euler of the
operator's emitted velocity (`pos += dt*mobility*F`, `mobility == 1/gamma == 1`). So the trajectory
is a pure function of the force law. IC: a fixed 19-cell mutually-overlapping sunflower cluster
(min NN 0.65, 36 overlapping pairs at sigma=1), uniform radius 0.5, 5 dead padding slots at the
origin (to exercise the dead-pair mask + the sigma=0 safe_divide), centred in a free 40x40 world,
60 steps at dt=0.2. The IC is BYTE-IDENTICAL on both sides (the oracle printed its 6-decimal live
coords, pasted verbatim into the spec `start:`); `soft_sphere` is gated `after_frame:1` so frame 0
= the pristine IC and frame t = the IC after t overdamped steps, aligning cell-for-cell,
frame-for-frame with the reference history s_0..s_60.

**Numbers.**
- Primary `D_pos = max over 61 frames x 19 live cells of ||x_plx - x_ref||/sigma = 2.70e-06`
  (threshold 1e-3). Frames 0-4 bit-identical; the discrepancy accumulates slowly and PLATEAUS at
  ~2.7e-6 by frame 41 (does NOT amplify -- overdamped relaxation is a contractive gradient flow).
- Alignment verified: frame 0 == IC on both sides (residual 0.0/0.0); the deliberately-MISALIGNED
  comparison x_plx(t) vs x_ref(t-1) scores 0.0788 (~29000x larger) -> the aligned convention is
  genuinely correct, not accidental.
- Single-step IC force residual `max_i ||(x_plx(1)-x_plx(0))/dt - F_ref(IC)|| / (eps/sigma) =
  6.1e-06` (raw force law before any compounding); run_spec `max|delta| 0.37996304` == oracle
  `force_ic_max 0.37996307` to ~3e-8.
- Radius-of-gyration trajectory relative error `1.4e-07` (1.6818 vs 1.6818); cluster genuinely
  MOVED (max live displacement 0.669 sigma) so there was real repulsive dynamics to disagree on.
- Every DEAD padding slot never moved on either side (0.0/0.0): the alive/self mask and the
  sigma=0 safe_divide reproduce; no phantom repulsion.
- Acted ledger: `adhere` calls 60 / acted 60, max|delta| 0.38, inert_operators [] ->
  valid_evidence true.

**Determinism / convention pinned first (oracle contract).** The kT=0 trajectory is identical at a
fixed key AND across two different keys (the noise is truly off); and jxm's BrownianDynamics update
equals a hand-rolled Euler over `SoftSphere().forces()` to 1.9e-6 (jit-scan vs eager-loop float32
reassociation of the SAME jax force -- a calibration of the jax-internal float32 floor over this
60-step run, not a convention error). This pins `dx = dt*forces` so the trajectory diff isolates
the force law.

**What this certifies (and its scope).** Reproduces the purely-repulsive harmonic law to float32:
the per-pair 1/2 AND the outer 1/2 (a dropped 1/2 doubles the force -> O(sigma) divergence), the
compact support / C1 vanishing at contact, the contact distance sigma = 2*r, and the dead/self
masking. It is a UNIFORM-radius test: the candidate reads a scalar `radius` PARAM (its fallback),
NOT the per-cell `radius` state block, so the headline SUM-vs-mean size-consistency
(sigma = r_i + r_j with heterogeneous radii) is NOT exercised. **Next experiment** to extend it:
give the cell set a per-type `radius` scalar (which registers the `lvl.radius` buffer the candidate
reads) with two distinct radii, seed the reference with the matching per-cell radii, and diff -- a
mean-instead-of-sum contact rule would then diverge by O(sigma). This does not touch the
verdict/dispute (that is the normalizer's axis); it validates the IMPLEMENTATION.

**Paths.**
- Oracle: `atlas_jax_morph/_oracle/runs/diff_soft_sphere/` (reference.npz + summary.json);
  script `atlas_jax_morph/_oracle/scripts/soft_sphere.py`.
- Plexus: `config/atlas/soft_sphere.yaml` -> `log/atlas/soft_sphere/` (diag.json, metrics.json,
  strip.png) + `graphs_data/atlas/soft_sphere/trajectory.npz`.
- Scorer: `atlas_jax_morph/_oracle/scripts/_analyze_soft_sphere.py` -> `log/atlas/soft_sphere/diff.json`.

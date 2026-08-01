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

---

## Implementation (IMPLEMENTER pass)

**Module.** `src/plexus/operators/candidates/jax_morph_harmonic.py` — registered
`@register_operator("adhere", family="interaction", set="cell", kind="lateral",
implementation="harmonic")`, so it is the **first** implementation of the `new` contract `adhere`
(none was registered before; the sibling SoftSphere entry named it but no operator yet carries it).
A later SoftSphere/Morse/Hertzian/LJ candidate joins the same contract via a different
`implementation=` key, exactly as `diffuse` holds `finite_difference` + `spectral`. `AdhereHarmonic`
subclasses `Lateral`, `EMIT="velocity"`, `SUPPORTED_DIMS=[2,3]` (reads `D=pos.shape[-1]`).

**How the reference's shape survives the port to torch.**
- `sigma = r_i + r_j` read off the per-cell `radius` buffer (block-or-buffer `_read_scalar`, the
  grow_radius recipe); `r_c = r_cutoff_frac*sigma`; the C0 hard `torch.where(r<r_c, ., 0)` is kept
  verbatim (no smoothing — the force jumps at `r_c`, the faithful surprise).
- `k` is a shared scalar OR a per-cell field (`k_field`), symmetrised by the **arithmetic-mean** mix
  `0.5*(k_i+k_j)` (no sqrt), matching the base `mix`.
- The down-shift `-(r_c-sigma)^2` is preserved, so the well is adhesive (negative at contact).

**The one real design decision: force = -grad(energy) vs analytic force, and `velocity` vs the
reference's raw force.** The reference `PairwisePotential.forces` returns `-jax.grad(total_energy)`,
a FORCE, and the *wrapping* overdamped step turns it into motion by `1/gamma`. Plexus has no separate
"potential returns force, step divides by gamma" seam — an operator emits an integrable delta. Two
faithful choices existed: (a) emit an `acceleration` (inertial) or (b) emit the overdamped drift
`velocity = mobility*F`. I chose (b) because the whole jax-morph mechanics is overdamped (Brownian /
active-Brownian / gradient-descent relaxation, never inertial), and because the campaign's `agitate`
(the Brownian bath) already documents the split: it emits the thermal velocity and expects the drift
potential to emit `F/gamma`. So `adhere` emits `v = mobility*F`, `mobility=1/gamma` default 1.0 (the
reference's gamma=1); scheduled with `agitate` the two velocities sum into one Euler-Maruyama step.
For the force itself I return the **analytic** radial law `F(r)=k*(sigma-r)` (which IS `-dU/dr`)
rather than autodiff-in-forward: it is exactly the reference's force inside the cutoff, is
differentiable w.r.t. pos and k by plain torch, and avoids a `requires_grad` leaf in the hot path.
This is NOT fitting the oracle — it is the closed-form gradient of the same energy, and the test
below proves the two agree.

**Test** (`tests/test_jax_morph_harmonic.py`, 8 pass). Every assertion is stated without the
reference: (1) **energy-defined force** — the emitted velocity equals `-torch.autograd.grad(
op.total_energy, pos)` to `1e-4` on a heterogeneous cluster with per-cell radii and per-cell `k`
(this is the load-bearing check that the analytic force is genuinely `-grad U`, and that the mix /
down-shift / cutoff are all self-consistent); (2) **zero at contact** (`r=sigma`); (3) **three
regimes + hard cutoff** — repel at `r<sigma`, adhere at `sigma<r<r_c`, exactly zero at `r>=r_c`,
with equal-and-opposite pair forces; (4) **momentum conservation** — `sum_i v_i ≈ 0` (Newton's
third law, any positions/radii/k); (5) **dead/masked** — a dead cell neither moves nor pushes, while
an alive-but-`at:`-masked cell is not driven yet still SOURCES a force (masking gates the actor, not
the field); (6) **pos not mutated** by forward; (7) `r_cutoff_frac<=1` raises (the construction
check); (8) **3-D generic**. Entry updated: `status: implemented`, `module`, `test` set. Evidence
(`oracle_run`/`diff_metric`) left null — the differential comparison against the jax oracle is the
CURATOR's pass; I did not run it and did not hard-code any reference number.

---

## Differential (DIFFER pass) — PASS, `status: validated`

**Verdict: the operator reproduces the reference, INCLUDING the adhesive tail.** D_pos = **7.86e-6 σ**
(threshold 1e-3), a clean pass ~127× below threshold and in the same float32-noise band as the sibling
soft_sphere differential (2.7e-6), despite ~18× larger forces (force_ref_max 6.96 vs 0.39).

**Metric (fixed BEFORE the run).** D_pos = max over recorded frames t=0..160 and live cells i of
‖x_plexus[t,i] − x_ref[t,i]‖₂ / σ, on a deterministic overdamped (kT=0) relaxation. Both sides are the
SAME forward Euler at γ=mobility=1 on a byte-identical float32 IC, so the trajectory is a pure function of
the force law f(r)=k(σ−r) truncated at r_c — repulsive core + **adhesive tail** + hard C0 cutoff. Pass
condition = (D_pos ≤ 1e-3) AND (frame-0 == IC exactly on both sides) AND (dead slots never moved).

**The test is built to exercise the one feature that separates harmonic from the already-validated
purely-repulsive soft_sphere/hertzian: the adhesion.** IC = 19-cell Vogel sunflower (scale 0.5, radius 0.5,
σ=1.0) with 24 overlapping pairs (r<σ, repulsion) + 92 adhesive pairs (σ≤r<r_c=2.5) + 55 beyond-cutoff — both
regimes present at t=0. Under adhesion the cluster CONTRACTS to a bounded equilibrium (gyration 1.54→0.64),
the OPPOSITE of soft_sphere's monotone expansion; the diff tracks that contraction cell-for-cell.

**Negative control (the decisive number).** With k=ε=1, σ=1 the SoftSphere and Harmonic repulsive cores are
IDENTICAL (f=1−r for r<σ), so on the same IC they differ ONLY in the adhesive tail. Deleting the tail
(SoftSphere) diverges the trajectory by **1.48 σ = 1480× threshold** (single-step force gap rel 1.02;
SoftSphere gyration expands to 1.68 vs Harmonic's 0.64). The metric therefore provably resolves the adhesive
tail — a wrong (purely-repulsive) law lands ~10⁶× above the observed pass value.

**Corroborators all agree.** single-step IC force residual 4.0e-5 (units k·σ); gyration rel-err 2.5e-7;
adhesion_contracts_cluster true; dead-slot immobility 0.0 (both sides); frame-0==IC exactly (0.0 both);
misaligned-frame control (plx[t] vs ref[t−1]) 0.26 ≫ D_pos, so the alignment is real not accidental.

**Oracle self-guards (all clean).** jxm BrownianDynamics(kT=0) deterministic at a fixed key AND across two
keys; jxm.simulate ≈ hand-rolled dx=dt·forces Euler (max dev 5.7e-6, float32); 2-cell radial scan of
Harmonic.forces vs the analytic k(σ−r)|_{r<r_c} matches to **6e-8**, with force exactly 0 beyond r_c and a
deepest adhesive pull −1.45 — the reference implements the law the operator claims.

**Numbers & paths.**
- oracle : `atlas_jax_morph/_oracle/runs/diff_harmonic/` (reference.npz + summary.json + reference.png);
  script `_oracle/scripts/harmonic.py`. Model = BrownianDynamics(Harmonic(k=1.0, r_cutoff_frac=2.5), γ=1, kT=0).
- plexus : `log/atlas/harmonic/` (diag.json valid_evidence=true, adhere acted 160/160 max|Δ| 8.45;
  metrics.json/.npz, strip.png, diff.json); spec `config/atlas/harmonic.yaml`; trajectory
  `graphs_data/atlas/harmonic/trajectory.npz`.
- scorer : `_oracle/scripts/_analyze_harmonic.py` → `log/atlas/harmonic/diff.json`. value 7.8642e-06, passed true.
- record : evidence.oracle_run=diff_harmonic, evidence.value=7.8642e-06, evidence.passed=true, status=validated.

**Note on the known 124-vs-82 anchor discrepancy.** Out of scope here: this differential ISOLATES the force
law (no growth, no division, fixed cell count/radii), so the anchor's live-cell-count gap cannot enter — the
force law itself reproduces bit-for-float. That gap lives in the division/growth mechanisms, not in `adhere`.

**Env gotcha (durable).** `run_spec.py` needs `zarr` (field-recording path, engine.py:792); the
`particle-graph` conda env lacks it. Use `/workspace/.conda_envs/neural-graph-linux/bin/python` (torch 2.9 +
zarr 2.18 + plexus) for run_spec and the scorer. The oracle runs in its own pinned jax venv via `oracle.py run`.

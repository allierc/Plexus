# Analysis log — Atlas: jax-morph

Append-only. Newest at the bottom. One heading per mechanism id per call.

---

## Phase 0 — instruments, before any agent ran

- Baseline frozen: **52 registered contracts** (42 canonical + 10 aliases) across 10 families.
  That set is the whole comparison — the promoted language only; unreviewed code in `prototype/`
  and `operators/candidates/` does not enter the measurement.
- Oracle built and verified: jax 0.11.0, jax-morph 0.4.0 (clone @ `ace08b8`), CPU.
- Oracle determinism checked at a fixed PRNG key: two runs of the authors' proliferation model
  are bit-identical in position and alive. A differential test against it is therefore measuring
  us, not the reference's own noise.
- First reference artefact: `_oracle/runs/smoke/` — 4 → 82 cells over 40 macro-steps,
  gyration 0.65 → 3.23, `division_overflow = 0` (no array bound was hit).
- Record seeded mechanically from the clone's AST: **24 candidate mechanisms**, every one at
  status `candidate` — named and located, nothing inspected, nothing believed.

---

## division

Read the whole `Division(StochasticStep)` class (`physics/division.py:L38-L229`), its base
`StochasticStep`/`SimulationStep` and the `Model` macro-step in `core/step.py`, the AD primitives it
calls (`sample_bernoulli_st`, `bernoulli_logp`, `safe_norm` in `core/ad_utils.py`), and the field
contract in `core/state.py`. Fixed `code_path` to the class range and set `status: inspected`.

The event itself is a plain per-cell Bernoulli hazard `p = 1 - exp(-rate*dt)`; the interesting
physics is all in `replay`. What surprised me: (1) volume conservation is dimension-dependent —
`m = 2^(-1/d)` shrinks the radius so the *d-volume* halves, so `n_space_dim` is a physics knob, not
just a shape knob; (2) the daughter offset uses the NEW radius `r*m`, giving exactly-touching
daughters; (3) the mother side is soft/differentiable (straight-through, `d/dp=1`) while the daughter
slot is a hard scatter — the two daughters are not symmetric under autodiff; (4) capacity overflow is
silently capped into a global running `division_overflow` counter (never raises); (5) `replay`
implicitly reads EVERY cell-scope field to fill the daughter (heritable → inherit, else reset to
default), though only `division_rate`/`division_axis` are declared reads.

Could NOT establish: the paper PDF would not render here (no poppler/pdftotext; text streams
compressed), so I anchored `paper_section` to the installed library guides and did NOT verify against
a specific figure/equation whether the authors' paper describes the same `2^(-1/d)` conservation and
oriented placement — that source-vs-paper check is still open. Did not run the oracle for this entry
(Phase 0 smoke already showed `division_overflow=0`, i.e. the cap was never hit at that scale). Did
not confirm whether any shipped example ever sets `division_axis` or a non-zero `orientation_snr`, so
whether oriented division is exercised in practice or is effectively dormant is unverified. Verdict
(alias/refinement/new) left null for the normalizer.


---

## active_brownian_dynamics2_d

<!-- ActiveBrownianDynamics2D -- append below; the driver merges this into campaign/analysis.md -->

# ActiveBrownianDynamics2D

**Read:** `jax_morph/physics/mechanics/dynamics.py:199` (the class), its sibling base
`BrownianDynamics:36`, the `StochasticStep` / `Model` contract in `core/step.py`, the potentials in
`mechanics/potentials.py` (`NoForce`, `PairwisePotential`, `Morse`), `gaussian_logp` in
`core/ad_utils.py`, and `tests/physics/test_active_brownian.py`. Grepped the paper text and notebook
04.

**What it does to the state:** one Euler-Maruyama step. Drift = `F/gamma + v0*e(theta)`; it writes a
position displacement delta and an `active_heading` increment delta, recording four trace fields
(`xi_t`, `dx`, `xi_r`, `dtheta`) so it is both pathwise-differentiable and scorable. `active_speed`
is read-only (heritable, set externally); `active_heading` is read then incremented as a *shared*
dynamic field (so an alignment step can accumulate into it -- the Vicsek pattern).

**Biggest surprise -- paper contradiction (source wins):** the whole class is *absent from the
paper*. The paper's mechanics is passive and, in the SI Methods (p. 14, "Mechanical Interactions"),
even noise-free -- deterministic gradient-descent energy minimization of a Morse potential; the main
text (p. 3 Fig. 1d, p. 19) calls the high-temperature rearrangement "Brownian motion/relaxation". No
self-propulsion, active heading, or rotational diffusion appears anywhere. This is a library-side
active-matter extension: its docstring names it "the natural base for a Vicsek-type model", and
`examples/04_physics_examples.ipynb` pairs it with a `VicsekAlignment` step to build a flocker. I
anchored `paper_section` to the closest passive-mechanics passage and flagged the absence in
`surprises`.

**Other reimplementer traps I recorded:** self-propulsion `v0*e` is NOT divided by gamma (force is);
heading has zero drift (scored against mean 0); the two channels scale noise differently
(`std_r = sqrt(2 D_r dt)` has no gamma); `kT=0`/`D_r=0` are guarded to contribute 0 rather than NaN
(kT=0 is the standard ABP); heading is written as an additive delta, not an absolute angle.

**Did NOT establish:** I did not run the oracle -- this entry is static source+test reading only, so
`evidence.oracle_run` stays null (the verifier's job). The tests assert the free-drift limit, the
rotational-diffusion variance `2 D_r dt`, the trace round-trip, and both gradient estimators, but I
did not numerically execute them here. I did not exhaustively audit every notebook to confirm no
*paper-reproducing* example instantiates this step (I believe none does, consistent with its absence
from the paper). I left `verdict`/`contract` null per the role split -- novelty vs. the promoted
`src/plexus/operators/` is the normalizer's call.


---

## brownian_dynamics

<!-- BrownianDynamics -- append below; the driver merges this into campaign/analysis.md -->

# BrownianDynamics (excavation note)

Read: `physics/mechanics/dynamics.py:36` (the class, its `_dist`/`sample_trace`/`replay`/`logp`),
its base `core/step.py` (`StochasticStep`: trace/replay/score contract, `score_by_default`,
the dynamic-delta accumulation in `Model._run`), `physics/mechanics/potentials.py` (`Potential`,
`NoForce`, the Morse family that supplies `forces = -grad U`), and `core/ad_utils.gaussian_logp`.
Paper: p. 14 "Mechanical Interactions" and the SI p. 19 differential-adhesion experiment.

What it does to the state: one Euler-Maruyama overdamped-Langevin step. `dx = -(grad U/gamma)*dt
+ sqrt(2 kT dt/gamma)*xi`, applied as a **dynamic position delta** (not an absolute position),
with `xi` recorded so the step is both pathwise-differentiable and scorable.

What surprised me:
- Paper vs source. The paper's Methods define relaxation as *deterministic* "gradient descent
  energy minimization of the Morse potential for a fixed number of steps" -- no noise, no gamma,
  no kT written anywhere. Thermal "Brownian relaxation" at "high temperature" appears only in one
  SI experiment (p. 19), with no equation. The source generalizes both into one parameterized
  Langevin step: kT=0 is exactly one gradient-descent step (lr = dt/gamma); kT>0 is the noisy
  regime. This unification is the code's, not the paper's.
- Split dt-scaling (drift ~ dt, noise ~ sqrt(dt)); the kT=0 degenerate-density guard
  (`_gaussian_logp_or_zero`); and `score_by_default=False` overriding the base True (reparameterized
  draw carries its own pathwise gradient, so the score term is off to avoid double-counting).
- `sample_trace` records only `xi`; `replay` derives and records `dx`, and `logp` scores the
  recorded `dx` (not `xi`).

Not established (uncertainty for the next reader):
- I did NOT run the oracle. `evidence.*` is untouched; no numerical check that the Plexus/oracle
  trajectory matches this update. A verifier should drive `oracle.py` on a Brownian/relaxation
  script.
- I did NOT confirm which macro-step `dt` value the paper's "200 relaxation steps" maps to, nor
  whether the paper's original (fmottes) code used a jax-md Brownian integrator with these exact
  gamma/kT semantics -- I only compared the refactored source in this repo against the paper text.
- Whether the paper's gradient-descent relaxation used a fixed learning rate (vs. FIRE/other) is
  not stated; I inferred the kT=0 -> lr=dt/gamma correspondence from the source, not the paper.
- `verdict`/`contract` deliberately left null (normalizer's job); status set to `inspected`.


---

## death

<!-- Death -- append below; the driver merges this into campaign/analysis.md -->

## Death (physics/death.py:L26) -- inspected

Read the whole `Death` class, its `StochasticStep` base (`core/step.py`), the AD primitives it
calls (`sample_bernoulli_st`, `bernoulli_logp` in `core/ad_utils.py`), and the sibling `Division`
it must compose with. It is Division's mirror image: a per-cell Bernoulli hazard
`p = 1 - exp(-death_rate*dt)`, forward-exact with an identity straight-through surrogate, replayed
HARD (flip `alive`, write a float `death` record) and scored by a masked score-function `logp`.

**Biggest surprise -- a code-vs-paper contradiction, and it runs the other direction from the usual
one.** The paper has NO death mechanism at all: "death"/"apoptosis"/"necrosis" occur ZERO times
(grep of the extracted text). The forward model's capability list (p. 2) is exactly "division,
growth, mechanical stress sensing, and morphogen excretion and detection." The paper's only cell
removal is an *external* robustness ablation -- deleting a random fraction of an already-finished
cluster to measure loss sensitivity (p. 8; Fig. caption p. 21), not a dynamic step. So `Death` is a
library step the source ships but the paper never describes. Recorded per rule 5 (source wins).

**Other traps I flagged:** `died` (ephemeral scored trace) vs `death` (persistent float record,
re-derived in replay by `(died>0.5) & alive`, and OVERWRITTEN not accumulated each step); the
Division-before-Death ordering and deferred slot reuse that keep `reconstruct_lineage` correct; the
`death_rate` clip to >=0 that stops a negative rate NaN-ing `bernoulli_logp`; and that the tunable
hazard is a STATE field, not a constructor arg (the only ctor param is `score_by_default`).

**Did NOT establish:** I did not run the oracle (paper ships no death config, and jax is
deliberately absent from this env), so I have no numerical trajectory confirming the hazard/lineage
behaviour -- purely a source read. I also did not trace whether any assembled model in the library's
examples/guides actually *includes* `Death` in a pipeline, so its intended composition partners
beyond the documented Division pairing are unverified. Verdict/contract left for the normalizer.


---

## declared_field_dataflow_validation

<!-- declared field dataflow validation -- append below; the driver merges this into campaign/analysis.md -->

# declared field dataflow validation (order 23)

**Read:** `core/state.py` (StateFieldSpec, merge_specs, BaseState, build_state_from_model) and
`core/step.py` (SimulationStep.state_reads/writes/requires, StochasticStep.trace_writes,
Model.__init__ -> Model._validate, _accumulate_dynamic). Plus guides concepts.md ("Physics and
control compose through fields") and core-abstractions.md ("Model ... validated field dataflow").
Moved `code_path` from state.py:L1 to step.py:364 (`class Model`) -- the summary's claim
"cross-validated when the model is built" is enforced there, in `Model._validate`; the declaration
vocabulary + merge live in state.py, so the contract straddles two files.

**What it actually is:** not a state update. Steps declare reads/writes/traces as StateFieldSpec
tuples; at Model build time `_validate` runs (a) a per-field write-conflict policy (quasistatic =
exactly one writer; dynamic writers many, summed; a field is never both; discrete steps exempt),
(b) trace-field checks (unique per stochastic step, no collision with base/physical names, dynamic
trace default==0), and (c) `merge_specs(BASE_SPECS, state_requires())` which raises if two steps
name one field with non-identical specs. `build_state_from_model` then synthesizes the typed state
class from the merged schema -- that synthesis IS the "coupling only through named fields" guarantee.

**Surprised me:** the name over-sells. Reads are *deliberately never validated* ("The model does
not police reads") -- no DAG, no read-before-write check; a field read with no writer is legal.
And nothing checks that a step's `__call__` body touches only what it declared -- the declaration is
an unenforced promise. Discrete steps are wholly exempt from the write-conflict policy (two can
clobber `alive`) yet still reserve their names against trace fields. `merge_specs` dedups only
EXACTLY-equal specs, so `POSITION(heritable=False)` in one step + plain `POSITION` in another is a
build error.

**Paper contradiction (valuable):** paper (Deshpande 2025, p.14, "I. Forward Simulation") describes
a single `CellState` datatype holding all properties, composed by "any subset or combination of the
steps" -- composition by *convention*, no declared reads/writes, no build-time validation (paper
grep for valid/conflict/schema/declare/contract hits only "validation LOSS"). The declared-dataflow
contract is an invention of the hardened library, invisible to a paper-only reader.

**Did NOT establish:** (1) I did not run the oracle or write a test that trips a validation error --
all validation claims are read statically from `_validate`, not exercised (evidence.oracle_run left
null). (2) I did not confirm whether any *shipped* physics/control step actually relies on the
read-with-no-writer or Emit->React trace-read paths in practice, only that the contract permits
them. (3) I did not audit `_accumulate_dynamic`'s alive-masking against a concrete multi-writer
model. (4) Whether this maps to an existing Plexus operator contract is left to the normalizer
(verdict/of/contract deliberately null).


---

## division

<!-- Division -- append below; the driver merges this into campaign/analysis.md -->


---

## free_screened_diffusion

<!-- FreeScreenedDiffusion -- append below; the driver merges this into campaign/analysis.md -->

# free_screened_diffusion (FreeScreenedDiffusion)

**Read:** `jax_morph/physics/diffusion.py` in full — the `FreeScreenedDiffusion` class (L92), its
`_kernel`, `__call__`, and the module helpers `_k0`/`_k1` (Abramowitz & Stegun polynomial
approximations of modified Bessel K0/K1). Base contract in `core/step.py` (`SimulationStep`,
`StepType.QUASISTATIC`), geometry in `core/geometry.py` (`pairwise_displacements`, free vs periodic
space), `safe_norm` in `core/ad_utils.py`, base field specs in `core/state.py`. Paper Diffusion
section read at p. 15 (M&M) plus the main-text Fig. 1c mention (p. 3).

**Central surprise (paper vs source, source wins):** the paper and code solve the *same* steady
screened-diffusion PDE (`D grad^2 c - K c + S = 0`) by completely different numerics. Paper: a
**graph-Laplacian** lattice solve `c = (K I - D L)^{-1} S` with `L = deg(A) - A` and *explicit*
boundary handling (closed/reflecting `A_ij = 1/dist`, or permeable via a heuristic **ghost sink
node** wired to detected boundary cells). Code: **analytic free-space Green's-function
superposition** — open boundaries automatic, no adjacency, no Laplacian, no ghost node, and a
**finite source radius `a`** (with Bessel/exponential kernels) that has *no* paper counterpart. A
reader trusting the paper would build a different solver and get different boundary behavior.

**Also surprised me:** the self/near field is *included* (r clamped to the source surface, so a cell
reads its own secretion — not the usual self-excluded neighbor sum); three easy-to-miss finite-ness
guards (`a`, `kappa` floors) matter only under `jax_debug_nans`/traced-kappa optimization; and
`state_reads()` declares only `secretion_rate`, silently also consuming `position`/`radius`/`alive`.

**Did NOT establish (open):**
- I did **not** run the oracle or any differential check — no numerical evidence the code and paper
  agree (or how far they diverge, especially near the cluster boundary). That is the validator's job.
- I confirmed by grep that `FreeScreenedDiffusion` is the **sole** diffusion step in the library
  (no graph-Laplacian sibling), so this is a wholesale method replacement, not one of two variants —
  but I did **not** check which example/guide configs actually instantiate it, with what
  `n_space_dim`/`degradation`, or whether any example still expects the paper's closed-system field.
- I recorded the 2-D disk kernel `K0(kappa r)/(2 pi D a kappa K1(kappa a))` verbatim from source but
  did **not** independently derive that it is the correct finite-disk screened Green's function, nor
  verify the `_k0`/`_k1` rational approximations against a reference over the full argument range.


---

## gene_network_connectionist

<!-- GeneNetworkConnectionist -- append below; the driver merges this into campaign/analysis.md -->


---

## gene_network_mwc

<!-- GeneNetworkMWC -- append below; the driver merges this into campaign/analysis.md -->


---

## harmonic

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


---

## hertzian

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


---

## lennard_jones

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

## lie__trotter_macro_step_split

<!-- Lie-Trotter macro-step split -- append below; the driver merges this into campaign/analysis.md -->

## Lie-Trotter macro-step split (order 21)

**Read at source:** `jax_morph/core/step.py` in full -- `Model` (L364), its `__call__` (L519),
the phase skeleton `_run` (L485), and the dynamic accumulator `_accumulate_dynamic` (L342); plus
`SimulationStep`/`StochasticStep`/`StepType` for the return-shape contract. Repointed `code_path`
from `:L1` to `:L364` (the `Model` class, which owns the split). Cross-checked against
`guides/concepts.md` heading "A simulation integrates a hybrid dynamical system", which states the
same operator form and names it Lie-Trotter, O(dt) first-order.

**What it does to the state:** one macro-step = `discrete o dynamic o quasistatic` applied to `s_n`,
then `t += dt`. Quasistatic and discrete phases are sequential pipelines (each step returns a full
state); the dynamic phase is Jacobi -- every dynamic step reads the ONE post-quasistatic state,
returns a sparse `state.deltas(...)` already scaled by `dt`, and the summed, alive-masked deltas are
applied once with no further `dt` multiply.

**Surprised me:** (1) phase order is FIXED and overrides the order steps are listed in; list order
only sequences within the quasistatic/discrete phases. (2) Dynamic is Jacobi, not Gauss-Seidel --
the docstring calls this out. (3) `dt`-scaling is baked into each dynamic delta, so the accumulator
never multiplies by `dt`; double-scaling is the obvious reimplementer error. (4) Alive-masking hits
cell-scope fields only; grids/globals still integrate on dead slots. (5) Forward sampling and
score-time replay share the SAME `_run`, so composition order cannot diverge between them.

**Paper vs code:** the paper NEVER says "Lie-Trotter", "operator split", or "first-order accurate" --
that formalization is the library's guide. The paper (Fig. 1a caption p.2 L100; Methods
"To create an initial state" p.14 L747-748) gives an informal per-timestep sequence
"cell division, cell growth, and mechanical relaxation", one division per step, with division at a
different position than the code's fixed order (discrete/division LAST). Recorded as a surprise;
source wins.

**Did NOT establish:** (a) did not run the oracle, so the O(dt) accuracy claim is unverified
numerically here -- that is the validator's job. (b) Did not re-read `state.py` line-by-line; the
base field set (position/radius/celltype/alive/t) and the `deltas`/`set`/`update`/`n_cells`/`alive`
API are taken from the guide and from usage inside `_accumulate_dynamic`, not confirmed at their
definitions. (c) Did not trace any concrete assembled `Model`/pipeline, so I cannot say which
specific paper steps land in each phase -- "division is discrete/last" is the CONTRACT's phase
order, not a verified property of a shipped model.


---

## mechanical_relaxation

<!-- MechanicalRelaxation -- append below; the driver merges this into campaign/analysis.md -->

# MechanicalRelaxation (order 19)

**Read:** `physics/mechanics/relaxation.py` in full -- the `MechanicalRelaxation` step (L221), the
`relax_equilibrium` free fn (L189), and the three internals it composes: `_fire_to_tol` (forward
FIRE loop), `_rigid_body_modes`/`_project` (the physical-subspace projector), and the
`jax.custom_vjp` triple `_relax_impl`/`_relax_fwd`/`_relax_bwd` (the implicit-diff backward). Also
read the `Potential`/`NoForce` base (`potentials.py`) and `SimulationStep`/`StepType` (`core/step.py`)
to pin the read/write contract and the QUASISTATIC phase semantics.

**Biggest surprise -- a clean paper/code contradiction.** The paper (p. 14, "Mechanical Interactions")
says relaxation is *"gradient descent energy minimization of the Morse potential for a fixed number of
steps (except in the case where we learn cell adhesion)."* The source does none of those three things
as written: (1) the minimizer is **FIRE** (momentum + adaptive dt), not plain gradient descent; (2) it
runs to a **force tolerance** `f_tol` (a real equilibrium), with `max_steps` only a fallback bound, not
a fixed step count; (3) the gradient is the **implicit-function-theorem** sensitivity of the
equilibrium (`jax.custom_vjp`), not backprop through the descent. The paper's own aside "(except ...
cell adhesion)" is precisely the case where fixed-step solver-path gradients fail -- which the code's
implicit-diff generalizes away. Recorded in `surprises` with both readings; SOURCE WINS.

**Other reimplementer traps I flagged:** the rigid-body null modes (dead DOFs + global translation +
global rotation of the alive cells) are projected out of the adjoint and carry *zero* gradient by
design (a translation/rotation-sensitive loss just gets none); `ridge` is backward-only and lives on
the *projected* Hessian; the FIRE schedule constants are hardcoded (Bitzek-2006 values), not knobs;
the adjoint also differentiates wrt the incoming **state** (radius -> sigma), because
`params = (potential, state)` rides the custom_vjp as one pytree; and the declared `state_reads`
omits radius/alive/displacement since those are base fields.

**Did NOT establish / open uncertainties:**
- No oracle run -- I did not confirm numerically that the smoke trajectory exercises this step, nor
  that FIRE converges within `max_steps=500` on the reference cluster (the `RuntimeWarning` path is
  untested by me). `evidence` left null for the verifier.
- I did not trace whether the reference `smoke` config uses this quasistatic step at all vs the
  `BrownianDynamics` (kT=0) relaxation route -- both can relax positions, and which one the paper's
  experiments used per-figure is unresolved here.
- I asserted the FIRE constants match Bitzek 2006 from the values/names; I did not cross-check against
  a citation in the paper (the paper names neither FIRE nor these constants).
- The "(except ... cell adhesion)" clause: I inferred it points at the solver-path-gradient problem,
  but the paper does not spell out what alternative they used in that case -- an open question for
  whoever excavates the adhesion-learning experiment.


---

## morse

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

## neural_ode

<!-- NeuralODE -- append below; the driver merges this into campaign/analysis.md -->


---

## no_force

<!-- NoForce -- append below; the driver merges this into campaign/analysis.md -->

# NoForce (potential)

Read the whole `potentials.py` module and all three consumers (`dynamics.py`,
`relaxation.py`) plus the `Potential` base class.

**What it does.** `NoForce` is a direct `Potential` subclass whose `total_energy(positions,
state)` is a bare `jnp.asarray(0.0)` and whose `forces(state)` is *overridden* to return
`jnp.zeros_like(state.position)` directly — bypassing the base class's autodiff-of-energy path.
Net effect on state: nothing moves. It is a null/identity element for the mechanics slot, not a
physical interaction.

**Where it actually lives.** Nobody constructs it by name in the physics. It is what
`potential=None` resolves to inside `BrownianDynamics`, `ActiveBrownianDynamics2D`,
`MechanicalRelaxation`, and `relax_equilibrium` (`potential if potential is not None else
NoForce()`). So it materializes three model idioms: a free Brownian gas, a self-propelled active
gas whose alignment coupling lives in a separate step (Vicsek base), and a no-op relaxation.

**Surprised me.** (1) It is *not in the paper at all* — the paper's cells always interact via a
Morse potential (p. 3, Fig. 1d). NoForce is a pure library affordance, so it is a genuine
"the-code-adds-a-null-element the-paper-never-mentions" case, which I flagged in `surprises`
rather than as a paper/code contradiction. (2) The class docstring advertises "skipping the dense
O(N^2) pairwise sum," but NoForce is a *direct* `Potential`, not a `PairwisePotential` — its own
energy has no sum; the O(N^2) line is general motivation, and the real cost the override avoids is
the grad trace. (3) Under `MechanicalRelaxation` the no-op happens via the *general* FIRE path
(residual 0 <= f_tol at iter 0, loop body never runs), not a special branch — even though
`relax_equilibrium` does carry an explicit `if potential is None` short-circuit that the step
doesn't use (it stores a concrete `NoForce()`).

**Could not determine / did not establish.** I did not run the oracle — a zero potential has no
trajectory signature to diff against, so I left `evidence` null and treated this as a
read-at-source entry. I did not chase whether any *notebook/example* (outside `jax_morph/`) or the
GRN/control side ever passes `NoForce` explicitly; my "constructed only via None" claim is scoped
to the physics package + guides, which is where I looked. I also did not verify the exact default
dtype of the `0.0` scalar under the library's runtime config beyond noting the test suite enables
x64; `forces` sidesteps this by using `zeros_like(position)`.


---

## odecontroller

<!-- ODEController -- append below; the driver merges this into campaign/analysis.md -->

## ODEController (excavator, read at source)

Read the whole `control/ode.py` (base + all three subclasses), the `SimulationStep`/`Model` step
contract in `core/step.py`, `state.deltas`/`StateFieldSpec` in `core/state.py`, and the paper's
"Genetic regulatory interactions" (p. 10) plus fig. 1b. `code_path:L46` still points at the class def.

What it does to state: it is a **DYNAMIC** step. It packs `y = concat(hidden, outputs)` per cell,
freezes the sensed `inputs`, and integrates `dy/dt = vector_field(t, y, inputs)` over `[0, dt]` with
diffrax Dopri5 + `PIDController(rtol=1e-4, atol=1e-6)`, `dt0=dt`, saving only the endpoint. It returns
the **increment** `y(dt) - y0` as a sparse delta, not the new state -- the Model accumulates it. The
base is pure machinery; `vector_field` is abstract.

Surprised me most: the **paper vs code disagreement on where the sensed input enters the sigmoid**.
Paper p. 10: `dg_i/dt = phi(sum_j W_ij g_j + b_i) + I_i - k_i g_i` -- the forcing `I_i` is additive,
OUTSIDE the sigmoid. The closest code subclass `GeneNetworkConnectionist` puts the input INSIDE the
sigmoid (`sigma(W_gene@g + W_in@u + b)`) with no additive input term. Source wins; recorded as a
surprise on the base and it also belongs on the connectionist entry. Also notable: the "sigmoid" is
the *algebraic* `0.5 + 0.5 x/sqrt(1+x^2)` in the connectionist circuit but `jax.nn.sigmoid` (logistic)
in the MWC circuit -- two different saturations under one paper symbol.

Could NOT establish: (1) which concrete subclass the paper's headline results actually used -- the
text says only "simple ODE model inspired by Hiscock", so Connectionist is the best guess but I did
not open the example notebooks to confirm an instantiation; (2) whether the input-placement
difference materially changes fitted dynamics -- I ran nothing (excavator, and jax is deliberately
absent from the Plexus env); (3) I did not verify the diffrax adaptive solver's actual internal step
count given `dt0=dt` -- I read the config, not a trace. These are for the normalizer/validator, not me.


---

## pairwise_potential

<!-- PairwisePotential -- append below; the driver merges this into campaign/analysis.md -->

# pairwise_potential (PairwisePotential, base class @ potentials.py:L128)

**What I read.** The whole `potentials.py`: the `Potential` protocol (L56, `forces = -jax.grad(total_energy)`),
`NoForce` (L108), the `PairwisePotential` base (L128), the five concrete subclasses (Morse, SoftSphere,
Hertzian, Harmonic, LennardJones), and the two module helpers `_compact_repulsion` / `_smooth_cutoff`. Then
the primitives it leans on: `core/geometry.py` (`pairwise_displacements`, the dense `neighbor_sum` alive/self
mask) and `core/ad_utils.py` (`safe_norm`, `safe_divide`). This entry is the ABSTRACT base only -- Morse,
SoftSphere, Hertzian, Harmonic, LennardJones, NoForce, MechanicalRelaxation, and VirialStress are each their
own record entry (confirmed in atlas_record.yaml), so I scoped this to the contract they inherit: energy ->
force (autodiff) -> virial, over live non-self pairs, with couplings resolved scalar-or-per-cell-field.

**Line/anchor checks.** L128 is exactly `class PairwisePotential` -- code_path unchanged. Paper anchor: the
Morse energy `V_ij(r)` on p. 9 ("Mechanical interactions", also fig. 1d, SI p. 14). The base class itself is
never described in the paper; only its concrete Morse instance is (recorded that gap).

**What surprised me.** (1) SOURCE vs PAPER: the paper says JAX-MD ran the mechanics (Methods, p. 9), but the
source's module docstring is explicit that forces come from autodiff "with no jax-md dependency." Source wins.
(2) Two different halves that look identical: the `0.5` in `total_energy` is a double-count dedup (neighbor_sum
sums both (i,j) and (j,i)); the `1/2` in `virial_pressure` is the Irving-Kirkwood bond split. (3) `mix` is the
ARITHMETIC mean (deliberate -- finite grad, no sqrt), not the geometric/Lorentz-Berthelot mean I expected from
MD. (4) Couplings are hard-restricted to a shared scalar OR a per-cell scalar field; per-type/array couplings
are rejected at `__check_init__`, so the paper's heterotypic cadherin matrix can only be expressed as a per-cell
field mixed to the pair mean -- a real expressiveness limit worth flagging to the normalizer.

**What I did NOT establish.** I did not run the oracle -- no numeric confirmation of the energy/force/virial
values (evidence left null; this is an EXCAVATOR pass). I read `virial_pressure`'s d-ball volume and IK sign
convention off the code/docstring but did not verify the sign against a compression test. I did not trace which
wrapping steps actually call `virial_pressure` vs `forces` (VirialStress / MechanicalRelaxation /
BrownianDynamics are separate entries) -- I only asserted from the signatures that the base writes no state
itself. Whether `mix` is ever overridden by a shipped subclass I did not check (none of the five in this file
override it).


---

## saturating_cell_growth

<!-- SaturatingCellGrowth -- append below; the driver merges this into campaign/analysis.md -->

## saturating_cell_growth (EXCAVATOR, status -> inspected)

**Read:** `jax_morph/physics/growth.py:23` (whole class), its base `SimulationStep`/`StepType`
(`core/step.py`), the state/delta machinery (`core/state.py`: RADIUS, StateFieldSpec, `deltas`,
`_accumulate_dynamic`'s alive-masking), and `tests/physics/test_growth.py`. Also read the SI
"Cell Growth" section of the paper (p. 14) and the main-text sketch (p. 2).

**What it does to the state:** a DYNAMIC step that reads `radius` + per-cell `growth_rate` and
returns a sparse delta `radius += dr`, where `dr = (R - r)(1 - exp(-k*dt/R))` is the *exact* flow
of the von Bertalanffy ODE `dr/dt = k(1 - r/R)`. Growth is fastest at small r, halts at
`max_radius`, bounded in [r0, R] for any dt.

**Biggest surprise (source vs paper contradiction, rule 5):** the PAPER says growth is
*constant-rate with a hard clamp*, `R_i(t+dt) = min(R_i(t) + dR, R_max)`, and explicitly notes the
`min` must be smoothed to be differentiable. The CODE does something else entirely: a smooth
exponential relaxation whose rate decays toward the target -- no fixed `dR`, no `min`, inherently
differentiable. Same endpoint (<= R_max), different trajectory (paper: linear-then-flat; code:
exponential asymptote). Recorded the code as the mechanism and both readings in `equations:`.

**Other things a reimplementer gets wrong:** (1) it returns a `deltas(...)` increment, not an
absolute radius -- return `min(r+dr,Rmax)` and you break the dynamic-phase accumulation contract;
(2) the increment is the *exact* exponential, chosen over forward Euler which diverges once
`dt*k/R > 2` (the stability test drives `dt*k/R = 3`); (3) `max_radius` normalizes the exponent
`-k*dt/R`, so the time constant is `R/k`, not `1/k`; (4) the rate `k` is a *state field*
(`growth_rate`, heritable, default 0), NOT a constructor param -- that's the whole design point
(an upstream controller writes it per-cell and gradients flow back through it); (5) dead-cell
masking lives in the model (`_accumulate_dynamic`), not the step -- ported standalone, dead cells
would grow.

**Not established / uncertainty for the next role:**
- I did NOT run the oracle to numerically confirm the trajectory against the smoke reference; the
  `evidence:` block is left null (that is a downstream role's job). The equations are from static
  reading only.
- I could NOT determine whether the von Bertalanffy law is a faithful re-derivation of the
  ORIGINAL Deshpande `jax-morph` growth code or a fresh choice by whoever refactored this repo into
  the typed-step architecture: the original repo's `growth.py` is not in this tree to diff against,
  so the paper's `min(R+dR, Rmax)` is my only "other" reference. The contradiction is real
  regardless (code disagrees with the paper text), but its *provenance* (deliberate improvement vs
  drift) is open.
- The paper writes `dR` with no cell subscript (a single global increment); the code's `k` is
  per-cell. I read this as an intentional elaboration, but the paper does not say so.
- verdict/contract left null on purpose (normalizer's call, per the loop rules).


---

## soft_sphere

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

## step_type:_quasistatic___dynamic___discrete

<!-- step type: quasistatic / dynamic / discrete -- append below; the driver merges this into campaign/analysis.md -->

## step type: quasistatic / dynamic / discrete

**Read:** `jax_morph/core/step.py` in full — the `StepType` StrEnum (L50), the `SimulationStep`
`__call__` return contract (L126), `Model._validate`'s per-type write-conflict policy (L394),
`_accumulate_dynamic` (L342), and `Model._run` / `Model.__call__`'s A/B/C phase skeleton (L485,
L519). Also `core/state.py` for `deltas`/`set`/`update` and the field-spec scope model, plus the
`concepts.md` and `core-abstractions.md` guides.

**What it is:** not an operator but a *meta-tag*. Each step declares a class var `step_type` that
fixes (a) which macro-step phase it runs in and (b) how the Model reads its return value.
Quasistatic + discrete return a full state (pipeline/sequential); dynamic returns a sparse dt-scaled
delta that the Model evaluates for every dynamic writer at the SAME post-quasistatic state, then
sums, alive-masks, and applies once. A macro-step is the fixed Lie-Trotter split
`disc o dyn o qs`, then `t += dt`.

**What surprised me (the payload):** the paper (Deshpande 2025, p. 14 "FORWARD SIMULATION") has
NO such taxonomy. It describes the sim as "any subset or combination of the steps detailed below" —
a flat customizable sequence of full-state ops (growth, relaxation, division, diffusion), "each
simulation timestep [consisting] of one cell division." The words quasistatic / dynamic / discrete /
Lie-Trotter / hybrid / macro-step appear ZERO times in the paper. The three-phase time-scale
taxonomy, the operator split, and the dt-scaled sparse-delta contract are a library
re-architecture. Per rule 5, source wins — recorded in `surprises:`. Other traps a reimplementer
hits: dynamic is order-independent (all at same state, summed) while qs/disc are Gauss-Seidel;
dt-scaling is baked into each dynamic step, not the accumulator; discrete steps are exempt from
write-conflict checks (last-writer-wins, silent); phase dispatch is `is`-identity on the StrEnum,
so a raw string `'dynamic'` runs in no phase silently.

**What I did NOT establish:** (1) I did not run the oracle — no differential evidence, and I did
not confirm which concrete steps in the running `smoke` model carry which step_type (I read the
contract, not a live pipeline census). (2) I did not verify the paper's *original* jax-morph GitHub
code (github.com/fmottes/jax-morph) to confirm it truly lacks any implicit qs/dyn/disc split — my
"paper has no taxonomy" claim is from the paper TEXT only; the original source could encode the
distinction informally. (3) The growth example on p. 15 (`R_i(t+dt)=min(R_i+dR, Rmax)`) adds a
FIXED per-step increment, not an explicitly dt-scaled rate — whether the library's "dynamic dt
increment" faithfully reproduces that is a per-step question I left for the growth entry.


---

## stochastic_step

<!-- StochasticStep -- append below; the driver merges this into campaign/analysis.md -->

# StochasticStep (core/step.py:L148)

Read the whole `StochasticStep` mixin, its base `SimulationStep`, the `Model` machinery that
drives it (`_reset_traces`, `_run`, `__call__`, `_replay_transition`, `_resolve_score`), the
`check_stochastic_step` round-trip guard, and two concrete subclasses (`physics/division.py`
`Division`, and the `MaybeDivide`/`Kick` toy steps in `guides/extending.md`). Paper anchors:
Methods p. 10 "Gradient calculation" (the `sum_t L_t * grad log pi(a_t|s_t)` REINFORCE estimator)
and the appendix policy-gradient derivation pp. 16-18.

This is an ABSTRACT mixin, not a physics step: it writes nothing itself, it fixes a contract
(trace_writes / _dist / sample_trace / replay / logp; `__call__` is DERIVED = sample_trace then
replay(pathwise=True)). The "update to the state" is that composition plus the scoring path.

Surprises worth flagging: (1) `replay` must co-emit every trace field or the whole record/score
round-trip silently produces garbage -- `check_stochastic_step` is the only guard, and it needs
TWO sentinel pre-fills because a reset-to-default single run can't distinguish "wrote it" from
"forgot it". (2) Reparameterizability is NOT a knob -- it is intrinsic to `replay` and selected by
the model via `pathwise = not scored`; `score_by_default` is only a scoring-inclusion flag. (3)
Trace fields are ephemeral (reset each macro-step), validated to not shadow real fields and to
default 0 when dynamic/additive.

PAPER-vs-CODE contradiction (recorded in `surprises:`): the paper presents ONLY the discrete
score-function/REINFORCE estimator ("stochastic operations have no differentiation rule") and a
uniformly-random division plane; the code additionally supports a reparameterized PATHWISE branch
and oriented placement -- a lower-variance gradient path the paper never describes. Source wins.

NOT established: I did not run the oracle or exercise `trajectory_logp`/`transition_logp` end to
end, so the claim that the forward record round-trips through scoring is read from the source and
`check_stochastic_step`'s docstring, not observed. I also did not enumerate every subclass in the
repo (checked Division + the two guide toys), so whether some step overrides `trace_from_state` or
routes `_dist` unusually is unverified beyond the documented "bespoke layout" escape hatch. Left
verdict/contract null for the normalizer.


---

## stochastic_trace___replay___score

<!-- stochastic trace / replay / score -- append below; the driver merges this into campaign/analysis.md -->

# stochastic trace / replay / score (core/step.py:148 + simulate.py + ad_utils.py)

Read: `StochasticStep` (step.py:148) and its base `SimulationStep`; the driver side --
`Model._reset_traces`, `_run`, `_replay_transition` (step.py:550), `_resolve_score`; the top-level
scorers `trajectory_logp` / `transition_logp` (simulate.py:92,137); the density kernels
`bernoulli_logp` / `gaussian_logp` / `sample_bernoulli_st` (ad_utils.py); and two concrete
subclasses spanning both estimator regimes -- `Division` (discrete, score-only) and
`BrownianDynamics` / `ActiveBrownianDynamics2D` (dynamic, reparameterized = pathwise AND scorable).
The stale `code_path` (`core/logp.py:L1`) does not exist; fixed to step.py:148.

The contract as an update-to-state: forward = `sample_trace` then `replay(pathwise=True)`; scoring
= read the recorded trace back out of the post-step state, then per selected step add
`logp(s_live, trace)` while replaying `pathwise=False`. The model sets `pathwise = not scored`, so
the SAME trace drives either estimator but never both for one choice. `trajectory_logp` returns
one term per macro-step (shape (T,)) = the paper's per-step `log pi(a_t|s_t)`; the return-weight
`G_t` and baseline are deliberately left to user code (no trainer in the library).

Surprised by: (1) the co-emission trap -- a `replay` that forgets a trace field scores garbage
silently, caught only by `check_stochastic_step`'s two-sentinel round-trip. (2) `sample_bernoulli_st`
is forward-EXACT (true draw, temperature-independent) with an identity backward surrogate. (3)
scoring masks by the recorded `divide_eligible` (alive at decision time), not current `alive`.
(4) division uses the competing-risks hazard `p = 1 - exp(-lambda dt)`, not `lambda dt`.

PAPER-vs-CODE (recorded in `surprises:`): the paper formalizes ONLY the score-function/REINFORCE
half for the discrete division event and backprops pathwise only where there is no division; the
code unifies both into one trace/replay/logp contract that also admits reparameterized stochastic
steps. Source wins.

OVERLAP FLAG for the normalizer: this entry and the sibling `stochastic_step` entry cover the SAME
`StochasticStep` mixin from two angles (this one emphasizes the trace/replay/logp PROTOCOL and the
scoring DRIVERS; the other the mixin CLASS). They may collapse to one contract -- see
`stochastic_step.md`. See also `division.md`, `brownian_dynamics.md`, `active_brownian_dynamics2_d.md`.

NOT established: I did NOT run the oracle or exercise `trajectory_logp`/`transition_logp` end to
end -- the round-trip and gradient claims are read from source + docstrings, not observed. I did
not enumerate every StochasticStep subclass (checked Division + the two Brownian steps), so a
subclass with a bespoke `trace_from_state` or an unusual `_dist` routing is unverified beyond the
documented escape hatch. Left verdict/contract null for the normalizer.

--- re-inspection addendum ---
Sharpened one point for `equations`/`surprises`: the SI's clean `grad log P(tau) -> grad log
pi_theta(a|s)` simplification is valid only because it treats the transition kernel `P(s'|s,a)` as
theta-independent. The CODE does NOT: `trajectory_logp` keeps the reconstructed state carry LIVE
(never detached in `_replay_transition`), so the environment's theta-dependence (secretion ->
diffusion -> future division propensity) flows PATHWISE, and reparameterized transitions get their
own Gaussian `logp`. So what the code differentiates is a SUM of a discrete score term and a
pathwise environment term -- broader than the SI's policy-only score. The SI (p. 16) concedes this
("propagated through the environment updates till the very beginning"). Prior working copy was
reverted for a YAML parse error (unquoted colon in prose); this pass moved every colon-bearing
prose value into block scalars. `status: inspected`; evidence still null (no oracle run).


---

## virial_stress

<!-- VirialStress -- append below; the driver merges this into campaign/analysis.md -->

# virial_stress (VirialStress)

Read `physics/mechanics/stress.py` (the whole 58-line module, one class), its base
`core/step.py:SimulationStep` (the quasistatic contract, `state.set`), and the real math it
delegates to: `PairwisePotential.virial_pressure` in `physics/mechanics/potentials.py`, plus the
`neighbor_sum` / `safe_norm` / `safe_divide` helpers in `core/geometry.py` and `core/ad_utils.py`.
Cross-read the paper's Methods "Mechanical Stress" paragraph (p. 16) and the main-text usage
(p. 7-8 "Mechanical Control of Cell Proliferation", Fig. 4 / S6).

`VirialStress` itself is a thin quasistatic sensing step: `state.set('stress',
potential.virial_pressure(state))`. It writes a transient per-cell `stress` field
(`heritable=False`), moves nothing, and carries no numeric knob -- its only field is the
`potential`, and the whole point is that a traced potential param (e.g. `epsilon`) stays
optimizable *through* the written stress.

**The surprise (headline).** SOURCE != PAPER. The code computes the textbook Irving-Kirkwood
virial pressure: `p_i = -(1/(2 d V_i)) sum_j r_ij (dU/dr)`, radial projection, normalized by the
cell's own d-ball volume and dimension, with a 1/2 bond split and a minus sign so compression is
positive. The paper (p. 16) instead defines `sigma_i = sum_j [F_ij,x*sgn(dx) + F_ij,y*sgn(dy)]`
-- a *component-wise sign* (taxicab) weighting, no volume/dimension normalization, no 1/2 split,
2D-only. These are genuinely different quantities (differently weighted, and off by ~2 d V_i per
cell). Per the loop rule I recorded the code's form as authoritative and the contradiction in
`surprises:`. Other easy-to-miss details: dU/dr is autodiff of the *cutoff-multiplied* energy;
`safe_norm`/`safe_divide` keep the r=0 diagonal and V_i=0 dead slots finite under debug_nans;
`neighbor_sum` masks self- and dead-pairs; no in-library step *reads* `stress` (its consumer is a
gene-network input in the examples).

**Did NOT establish:** whether this hardened reimplementation's `virial_pressure` reproduces the
*original* jax-morph numeric stress trajectory. I confirmed the FORM disagrees with the paper text
but did not run the oracle to diff a reference stress signal -- that is a validation-stage
question, and I flagged it rather than assuming equivalence. I also did not trace exactly which
example notebook's gene-input node consumes `stress`, only that nothing inside the package does.

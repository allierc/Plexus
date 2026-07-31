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

**NORMALIZER -- verdict `new`, contract `reorient` (polarity/cell, exchange; reads/writes
`heading`).** ABP2D is a composite step and decomposes into three legs, only one of which the
promoted language lacks. The passive drift `F/gamma` is a registered pair potential under an
overdamped mobility (alias). The self-propulsion `v0*e` + translational noise `std_t` is *exactly*
`glide` with its `noise` param -- an exact param map (`active_speed`->`move_speed`,
`active_heading`->`heading`, `std_t`->`glide.noise`), and glide even carries the `active_brownian`
tag (alias). The one uncovered leg is the **rotational diffusion of the persistent heading**
(`dtheta = sqrt(2 D_r dt) * xi_r`, zero-drift): a single-body, neighbour-independent Brownian
rotation of the cell's own orientation that no registered operator performs. It is the
orientational-decorrelation half of an ABP -- the thing that gives the walk a finite persistence
length and its ballistic->diffusive crossover, and in the `kT=0` textbook limit the *only* source of
wandering. Telling evidence it is a genuine gap and not an oversight: the prototype
`candidates/motility.py` bundled propulsion **and** `cell.heading += rot*randn` in one class; the
promotion to `glide` split off the propulsion and **dropped** the diffusion. Neither closest contract
can widen without violence -- `glide` is deliberately propulsion-only (reads heading, never writes
it; a heading write breaks its composition with the polarity family), and `polarity_align`/
`polarity_flow_align` are *social, deterministic* steering that return `{}` for an isolated cell, the
opposite of a targetless stochastic decorrelation.

**Strongest argument AGAINST this verdict (and why I still hold it):** the honest challenge is that
this should be a flat **alias of `glide`**, not `new`. glide's own docstring advertises "glide +
noise = an active Brownian walker" and tags itself `active_brownian`; if the promoted operator
already *claims* to be the ABP, then recording a `new` contract inflates the yield -- exactly the
failure this loop exists to prevent -- and I am splitting hairs over an implementation detail of how
the heading is maintained. I reject it because glide's claim over-reaches: glide never writes
`heading`, so a glide walker has a **fixed** (or externally-steered) direction and is a persistent
*ballistic* walker with translational jitter, which is not an ABP -- the defining ABP physics
(persistence length `~v0/D_r`, the long-time crossover, and the `kT=0` case where translation is
deterministic) lives entirely in the `D_r` rotational diffusion glide dropped. So the alias would
record a coverage the language does not actually have. The counter-inflation guard is that I scope
`new` to the reorientation leg *only* (2 of 3 legs are explicitly logged as aliases), so the marginal
yield is one contract, not a whole "active dynamics" operator.


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

## Normalizer

**Verdict: `new`** -- a motion-family operator `agitate` on the cell set: overdamped,
temperature-controlled thermal (Brownian) motion of positions, `dx = (F/gamma) dt + sqrt(2 kT dt/gamma) xi`,
with diffusion `D = kT/gamma` (Einstein / fluctuation-dissipation). The reasoning follows the campaign's
own StochasticStep split: the base mixin (order 5) is `out_of_scope` because it writes nothing, but a
concrete subclass with a real state-effect earns a forward contract (cf. `death` -> `new` `apoptose`).
BrownianDynamics writes POSITION, so it is not plumbing; its trace/replay/logp is the mixin's
(out_of_scope), and its drift is delegated to the pluggable Morse `potential` (a separate
cohesion/attraction_repulsion-family contract the engine would integrate). What is left as its own
primitive is a temperature-driven random walk of cell positions, which no frozen contract provides
(`diffuse` is a field/grid Laplacian, wrong set; `drag`/`glide`/`sediment` are dissipative/directed, not
a thermal fluctuation source). Recorded source-vs-paper contradiction (source wins): paper Methods (p. 14)
= deterministic gradient-descent Morse minimization with no kT/gamma; source = full overdamped Langevin,
of which the paper's relaxation is the kT=0 special case.

**Single strongest argument against (verified in the frozen source).** Three registered operators
*already* bolt on an isotropic `noise * randn` and Plexus can therefore jitter cell positions today:
`drag` (`PARAM_ROLES noise: thermal_noise`, `acc = -k*v + noise*randn`, comment "drag + noise = a
Brownian/Langevin bath"), `glide` (`noise: translational_noise`, "an active Brownian walker"), and
`attraction_repulsion` (`noise: exploration_noise`, "exploratory noise on the overdamped velocity"). A
skeptic reads that as: the language has thermal noise, so this is an alias (and the campaign's `death`
entry ruled a borrowed noise model an interchangeable IMPLEMENTATION detail, not contract identity).
My rebuttal is that these are all the *same* ad-hoc modifier -- a bare amplitude times a standard
normal, off-by-default, bolted onto a PRIMARY deterministic force. None carries a temperature, obeys
the Einstein relation (amplitude uncoupled from friction), applies the Wiener sqrt(dt) scaling (the
engine integrates them deterministically, so their diffusion constant is dt-scaling-wrong), or is a
scorable stochastic process; and none stands alone (drag needs velocity, glide a heading,
attraction_repulsion neighbors). That *three* operators independently accreted *three different
role-names* for one jitter is evidence of a MISSING abstraction, not a present one. `agitate` is that
missing thermostat -- constitutive (with `potential=None` the step is nothing but the bath, the exact
mechanism the paper names for differential-adhesion sorting, "high-temperature Brownian relaxation",
p. 19), temperature-parameterized, FDT-calibrated, Wiener-scaled. The closest contract to widen is
`drag`, and widening dissipative inertial friction (acc = -k*v, an acceleration, the energy-SINK half
of fluctuation-dissipation) into an overdamped temperature-scaled fluctuation *source* inverts both
its regime (2nd- to 1st-order) and its sign convention -- violence to its biology, not a widening.


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

## Death -- normalized

**Verdict: `new`.** Contract `apoptose` (kind `structural`, family `growth`, set `cell`; reads
`death_rate`, writes `alive`+`death`). The frozen baseline has no operator that removes a set
member: `cell_divide` and `cell_grow` are the only structural/growth/cell contracts, and both ADD
matter (a daughter, or volume). `apoptose` is `cell_divide`'s biological inverse -- it retires a
live slot rather than waking a dormant one -- so no existing contract covers it and widening
`cell_divide` to also destroy cells would conflate mitosis with apoptosis (the paper itself lists
division as a capability and never death). I gave it `cell_divide`'s exact typing on purpose: same
kind/family/set, so the record shows a birth/death PAIR of structural growth operators, not a lone
outlier.

**Strongest argument AGAINST `new` (the alternative I had to defeat).** Death and Division are so
tightly coupled at the implementation level -- identical hazard `p = 1 - exp(-rate*dt)`, identical
`>=0` clip, identical straight-through discrete draw, identical `{action}_eligible` masking, the
same DISCRETE phase, and a *mandatory* divide-then-die ordering -- that one could argue they are two
IMPLEMENTATIONS (or two directions) of a single abstract contract: a Bernoulli-hazard toggle of cell
occupancy, `occ 0->1` for division and `1->0` for death. Under that reading Death is an alias of
`cell_divide` (or the pair is one operator with a sign), and calling it `new` inflates the atlas's
yield by counting a sign flip as a new contract -- exactly the failure `record.py`'s R5 exists to
catch. I rejected it because Plexus fixes contract identity by what an operator DOES to the state
(its writes and biology), not by the noise law it borrows: division writes position/radius/lineage
and conserves volume across a new inherited slot, while death writes only `alive`+`death` and frees
nothing that step; sharing a random-timing law makes them no more one contract than `diffuse` and
`decay` are for both scaling by `dt`. But the coupling is real, and if Plexus ever adds a
`population_turnover`/occupancy-toggle abstraction, `apoptose` and `cell_divide` would be its first
two implementations -- worth revisiting then.


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

**NORMALIZER verdict: `out_of_scope`.** Declared-field dataflow validation is the declaration
discipline and TYPE SYSTEM of the operator algebra, not an operator in it: it reads no field value
and writes none, it only (a) cross-validates each step's declared reads/writes/traces at Model build
time and (b) synthesizes the typed state class from `merge_specs(BASE_SPECS, union of state_requires)`.
It is the fourth face of the one `core/step.py` engine layer whose other three faces are already
out_of_scope -- Lie-Trotter split (21), step_type taxonomy (22), StochasticStep mixin (5) -- and both
the 21 and 22 why-blocks explicitly name order 23 for the same verdict. Contract fields are a
validator formality (`couple` / exchange / coupling; reads = the BASE_SPECS merged against; writes =
`[alive]` as the formal allocation representative, mirroring the siblings' `t` and `trace`).

**Single strongest argument AGAINST it (and why it still loses):** this is arguably the atlas's most
valuable POSITIVE result rather than a scope-exclusion. Plexus's own IR *is* typed reads/writes per
operator, and `record.py` R7 (reject a writes-nothing operator) is a direct analogue of jax-morph's
"a declared write is the only coupling channel" plus `Model._validate`'s write-conflict policy -- so
the target has *independently reinvented the very discipline the Plexus algebra is built on*, which
one could read as a `new` build-time-dataflow-validator capability the language should register. It
loses because the atlas measures the OPERATOR VOCABULARY (typed forward maps over sets and fields),
and a validator is not a forward map -- it is the type system those maps are *written in*, which
Plexus already embodies; `new` further requires the thing be ABSENT, and this is the opposite of
absent (it is the frame). The honest resolution is out_of_scope WITH the correspondence logged as a
MEASUREMENT NOTE -- the language cannot count its own type discipline among its words without
inflating the yield -- flagging one axis jax-morph carries that Plexus's IR does not: a per-StepType
(time-scale-keyed) write-conflict policy refining R7's flat no-op rule.


---

## division

<!-- Division -- append below; the driver merges this into campaign/analysis.md -->

## Division -- normalized

**Verdict: `refinement` of `cell_divide`.** Same biological contract as the registered
`cell_divide` (structural/growth/cell, tags proliferation|mitosis|growth): identical Bernoulli
hazard `p = 1 - exp(-rate*dt)`, a daughter waking a free buffer slot beside the mother and
inheriting her per-cell state, capacity as a hard wall. Not `new` (proliferation already has a
home; widening does no violence to mitotic biology). Not a bare `alias`, because the two do NOT
fully agree: `cell_divide`'s promoted "default" implementation is ISOTROPIC and RADIUS-PRESERVING
(random-jitter placement, daughter radius cloned so two full-size cells stand where one did),
whereas Division (a) reads a per-cell `division_axis`+`orientation_snr` and places the daughter
ORIENTED along `s*a_hat + xi` (spindle-axis / Hertwig's-rule division) and (b) is VOLUME-CONSERVING
(mother shrinks to `r*2^(-1/d)`, daughters just-touching). The contract must widen its `reads`
(+`division_axis`) and its radius/position write SEMANTICS, and gains `mother`/`division_overflow`.
The cost a refinement must name: existing callers get isotropic full-size daughters, so enabling
volume conservation halves every cell's radius on division (changes packing/contact/virial stress)
and requiring an axis forces a default -- a real breaking change, hence a costed refinement, not a
free alias.

**Strongest argument AGAINST `refinement` (the alternative I had to defeat).** Plexus's registry
explicitly supports MULTIPLE implementations under one contract (`cell_divide` already carries an
`implementations` list), and each implementation may declare its own reads/writes -- so one could
file Division as simply a SECOND implementation of `cell_divide` (oriented + volume-conserving),
leaving the contract signature untouched. That is exactly the Morse/SoftSphere/Hertzian pattern the
campaign celebrates as convergence (and how the three `ODEController` siblings collapsed to one
`regulate`): same biological job, different internal recipe, `implementation_of: cell_divide`,
verdict `alias`. Under that reading nothing "breaks" (the default impl is untouched; you just add
another), the contract count is unchanged either way, and calling it a refinement over-reports
"language incomplete." I rejected it because oriented placement requires READING a `division_axis`
field that NO promoted operator reads today -- the capability genuinely does not exist in the frozen
language, so an alias would flatter it in precisely the way `registry_view.py`'s docstring warns
against ("record an alias without ever asking whether the two contracts actually agree"). The
Morse-family siblings share one I/O signature and differ only in a force's functional form; Division
and `cell_divide` differ in their declared I/O (an orientation input, a volume-conserving radius
write). If Plexus later promotes the widened signature, the current isotropic `cell_divide` becomes
its first implementation and Division the oriented/volume-conserving second -- but until then the
gap is real and belongs on the record as a refinement.


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

## Normalizer

**Verdict: `new` — contract `morphogen`** (kind `exchange`, family `fields`, set `cell`;
`cell -> cell`, reads `secretion_rate`+geometry, writes `chemical`, map `pairwise`), with
`implementation_of: morphogen`. This is the steady-state secreted-signal FIELD — a source `S`, a
degradation/screening `K`, and diffusion `D` fused into one quasistatic constraint that OVERWRITES a
per-cell concentration each macro-step, computed as an all-pairs Green's-function superposition on the
mesh-free cell set. The closest registered contract, `diffuse`, is a different contract, not a
widening: it is `set: field` (a GRID stencil), it is source- and sink-free by design (source is
`deposit`, sink is `decay`, and reaction-diffusion is BUILT by composing the three atoms), and it is
a dt-time-stepper — whereas this is mesh-free CELL state, bundles S/K/D, and returns the t=infinity
equilibrium. The code's free-space Green's-function solve and the paper's graph-Laplacian inverse
`c = (K I - D L)^{-1} S` are two SIBLING implementations of this one contract, which is the
convergence result worth recording.

**Single strongest argument AGAINST:** that `morphogen` is not new at all but merely the fixed point
of `deposit + diffuse + decay` — a COMPOSITION of three registered atoms run to steady state — so the
algebra already covers it and inventing a name inflates the yield. I reject it because the three atoms
iterate a LOCAL update on a GRID (a different discretization, with grid-dependent boundaries) and must
be run to convergence; you cannot wire them to reproduce an EXACT, mesh-free, all-pairs analytic
equilibrium solve on the particle set in a single quasistatic step. If a future contributor showed
that iterating the grid atoms to convergence reproduces this field to within the oracle threshold on a
real config, the honest move would be to demote `morphogen` to that composition — the entry stands or
falls on that unrun differential check.


---

## gene_network_connectionist

<!-- GeneNetworkConnectionist -- append below; the driver merges this into campaign/analysis.md -->

# gene_network_connectionist (NORMALIZER)

**Verdict: `new`, contract `regulate` (exchange/fields, set=cell); `implementation_of: regulate`.**
The mechanism is a per-cell INTERNAL regulatory dynamical system — a heritable gene-state vector
whose autonomous per-cell ODE (dense recurrent gene→gene coupling through a saturating
nonlinearity + linear decay + a learnable drive from *sensed external fields*) is integrated over
the macro-step to emit the action-setting outputs that division/secretion/adhesion read. Nothing in
the frozen 42 covers cell-internal regulatory computation: the motion/interaction/mechanics/mpm
families move or couple agents; the fields family sources/reads fields; growth/topology change the
set. The one real neighbour is `signal`. Crucially, this is also the anti-inflation call: the three
`ODEController` subclasses (this connectionist linear-drive form, `GeneNetworkMWC`'s thermodynamic
log-occupancy drive, and `NeuralODE`'s black-box MLP) are interchangeable *implementations* of the
same `regulate` contract — different vector fields under one integration/IO contract — so they
collapse to **one** new contract, not three. That is the convergence the ledger exists to record.

**Strongest argument AGAINST `new` (and why it loses).** `signal` is already a registered
recurrent, nonlinear, first-order ODE network (its own tags say "recurrent"), with the identical
−decay + bias + saturated-weighted-drive skeleton; one could argue `regulate` is merely a
`refinement` of `signal` — widen `set` to `cell`, make `edge_set`/`MAPS` optional, add a
field-input term — and that "gene regulation vs. connectome" is just parameterization of one
"recurrent-ODE-network" contract. This is the counterargument I had to defeat, and it is the
tempting one because the *math* really does rhyme. It loses on the *signature*, not the math:
`signal`'s recurrence runs BETWEEN nodes across a fixed connectome edge-set, and its typed
signature is load-bearing on exactly that topology (`INPUTS ["neuron","synapse"]`,
`MAPS ["pre","post"]`, `REQUIRES_PARAMS ["edge_set"]`, activation on the *presynaptic* input). The
gene circuit has no edge set and no cross-cell coupling at all — `W_gene` is a dense WITHIN-cell
matrix applied per cell (vmap), the sigmoid wraps the WHOLE drive, and the environmental forcing
enters via `W_in` on sensed *fields*, a term `signal` simply lacks. To "widen" `signal` to admit
this you must make its maps/edge-set optional and bolt on field sensing — i.e. delete the
connectome signalling that IS the contract for its only user (a neuron network) and convert a
lateral graph operator into a per-cell field→state controller. A refinement that guts the
signature its sole caller depends on is a breaking change wearing a smaller word, so the honest
verdict is `new`. (Second-order caveat I chose against: I filed `regulate` under the existing
`fields` family rather than minting a `control`/`regulation` family — defensible because its whole
I/O is per-cell scalar fields and the `signal` precedent already parks a control-like ODE inside an
existing family, but a future normalizer could reasonably argue the paper's separate control layer
earns its own family.)


---

## gene_network_mwc

<!-- GeneNetworkMWC -- append below; the driver merges this into campaign/analysis.md -->

# gene_network_mwc (NORMALIZER)

**Verdict: `new`, `implementation_of: regulate`.** No registered contract covers a per-cell gene
regulatory network: a stateful, recurrent intracellular circuit that integrates sensed
morphogen/mechanical fields into gene-expression outputs by integrating an ODE over a macro-step.
Contract `regulate` = `exchange`/`fields`/`cell`, reads {gene, driver}, writes {gene}. MWC is not a
separate contract but one of three interchangeable `ODEController` vector-fields (Connectionist =
linear W*g, the paper's eq. 4; MWC = thermodynamic log-occupancy; NeuralODE = MLP) that share one
signature and differ only in the drive's functional form — the Morse-vs-SoftSphere pattern — so the
gene-network family contributes ONE new contract, and `implementation_of` keeps the saturation
ledger from triple-counting it.

**Strongest argument against `new`.** The registered `signal` operator already encodes the *defining*
dynamics of a gene regulatory network — a recurrent leaky integrator over a weighted node network
with a nonlinear per-node drive, a time constant, and a resting bias (`dv/dt = -v/tau + sum
activation(v_pre)*w + bias`). One can read the GRN as `signal` under three *widenings*: (1) let
`edge_set` be optional / admit a dense within-element coupling instead of a sparse connectome, (2)
add a field-forcing input channel, (3) let each node carry a vector of states with several named
outputs. Under that lens this is a `refinement of signal`, and calling it `new` risks the exact
inflation the ledger is built to catch — a leaky-integrator recurrent network is a leaky-integrator
recurrent network whether its nodes are neurons or genes. I rejected it because those three
"widenings" together delete `signal`'s required `edge_set` and pre/post maps and its scalar-per-node,
no-external-input semantics — that is not a widening but a rebuild, and a refinement that changes a
contract's `requires_params` silently breaks every existing `signal` user (a refinement nobody costed
is a breaking change). But the counter-argument is genuine: if Plexus later generalizes `signal` to a
"recurrent regulatory network over an arbitrary coupling structure," `regulate` and `signal` could
well collapse into one contract, and this `new` would be retro-classed a refinement.


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

## NORMALIZER note

**Verdict: `refinement` of `attraction_repulsion` (implementation_of: attraction_repulsion).** LJ is
the 12-6 well SHAPE of the one pairwise radial cell-cell interaction the registry already carries --
a hard r^-12 core plus a -2 r^-6 adhesive tail, minimum -epsilon at contact `sigma = r_i + r_j` --
which IS attraction_repulsion's pull-minus-push biology, so its home is attraction_repulsion. It is
`refinement`, not `alias`, because the REGISTERED operator (attraction_repulsion.py, the D'Orsogna
model) has a single required GLOBAL scalar interaction length (`self.sigma = float(params["sigma"])`,
line 43; `REQUIRES_PARAMS=["sigma"]`, line 32), reads pos/edge_index/node_type but NOT `radius`, and
couples PER-TYPE (`type_params`, line 61) -- whereas LJ's contact distance is per-pair and
radius-coupled (`sigma_ij = r_i + r_j`) and its epsilon is per-cell. Admitting LJ therefore FORCES
the signature to (a) add a `radius` read, (b) generalise the length from a global scalar to the
additive per-pair rule, and (c) accept a per-cell coupling; the cost is that the contract's standing
"length is one global scalar sigma" invariant breaks for every consumer of it (the required `sigma`
param, the radius_graph cutoff and range diagnostics derived from it, the D'Orsogna/embryo specs).
This matches the record skeptic's ruling on LJ's nearest twin **Morse** (refuted alias ->
`refinement` of attraction_repulsion, conf 0.68). It rejects `new -> adhere` (the SoftSphere sibling,
order 15): LJ is expressible as a pairwise law, failing the language's own new-contract bar (the
stillinger_weber docstring, which the skeptic invoked to UPHOLD the abstract parent's alias at 0.82),
and the skeptic itself demoted SoftSphere's `adhere` back toward alias (conf 0.90) -- `adhere`
over-mints. Refinement is the precise middle: the alias camp (base/Morse/Hertzian/Harmonic working
copies) under-reports the added radius read; the adhere camp over-mints a whole contract.

**Strongest argument AGAINST (and why it is close):** the same skeptic that refuted Morse's alias
UPHELD the abstract parent `pairwise_potential`'s plain **alias** at HIGHER confidence (0.82), on the
ground that a two-body energy with autodiff force and a radius-sourced sigma "sits BELOW the
signature" -- i.e. adding a `radius` read is an implementation detail, not a signature change. If the
radius-coupling is additive and opt-in (the global-sigma D'Orsogna path still runs untouched), then
by refinement's OWN test -- "name what it breaks for existing users" -- nothing breaks, and a
widening that breaks no user is really an alias. That is the genuine tension, and I do not pretend
it away: the family's skeptic verdicts never converged (Morse refinement, base+SoftSphere alias,
Harmonic+Hertzian new). I land refinement because the break is real though narrow: attraction_repulsion
as registered makes `sigma` an UNCONDITIONALLY REQUIRED single global length, and LJ-mode supplies no
such sigma -- it sources the length per-pair from radius -- so the "one global length" invariant, and
any consumer that reads it, genuinely must change. Calling that alias deflates a real gap
(attraction_repulsion cannot, as registered, express a size-coupled per-pair contact distance);
calling it `new` inflates a contract the language's own pairwise bar forbids. Refinement is the
honest report of a costed widening. (Source vs paper: LJ appears nowhere in the paper -- Morse is the
paper's only mechanical potential -- source wins, recorded, verdict unchanged. Oracle not run; jax is
absent and python is sandbox-blocked here, so the entry was checked by inspection against record.py's
rules; the driver runs the validator.)


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

**NORMALIZE -- verdict `out_of_scope`.** The macro-step split is the INTEGRATOR / composition law
of the operator algebra, not an operator in it: it has no biological name, its only own write is
the global time scalar `t <- t + dt`, and it is explicitly numerics (a first-order Lie-Trotter
operator split). The composition law is part of an algebra's *definition*, not one of its elements;
Plexus's engine already owns operator scheduling, so the promoted VOCABULARY gains nothing. This
sits with the sibling engine-machinery verdicts `stochastic_step` and `no_force`; the vestigial
`contract:` (name `integrate`, aggregate/hierarchy, `writes: [t]`) exists only to satisfy R6/R7,
as in those precedents. **Strongest argument AGAINST:** the split is not inert plumbing -- the
Jacobi (not Gauss-Seidel) dynamic phase, the FIXED quasistatic->dynamic->discrete order that
overrides list order, and the dt-scaled sparse-delta accumulation are load-bearing SEMANTIC choices
a reimplementer must get exactly right, so one could call this a genuine "integration contract" the
atlas is missing (and if Plexus's engine composes operators differently, that disagreement is
exactly the source-vs-language contradiction the exercise prizes). It loses because those choices
are properties of the ENGINE that runs *any* operator pipeline, not of a forward operator over
sets/fields: they deserve DOCUMENTING (they are, in the surprises) but counting them as vocabulary
would inflate the yield with framework machinery -- the same reasoning that put `stochastic_step`
out of scope. Any genuine engine-composition disagreement should be recorded against the engine,
not as a new operator.


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

**Normalization (NORMALIZER role).** Verdict: **`new` -> contract `relax`** (kind `lateral`, family
`mechanics`): drive cell positions to the mechanical equilibrium (force balance `grad_x U = 0`) of a
supplied interaction -- a QUASISTATIC map to the force-balanced configuration, the tissue-mechanics
stance that mechanics equilibrates fast relative to growth so the cluster's shape at each morphogenetic
timepoint IS the equilibrium. Stripping the FIRE/implicit-diff numerics and the pluggable `potential`
(a separate `attraction_repulsion`/Morse contract) leaves the equilibration OPERATION, and no
registered contract expresses it: all 42 are single-application Euler maps (emit a velocity/force,
engine steps once); none runs to a fixed point. This is the quasistatic dual of `brownian_dynamics ->
agitate`. **Single strongest argument AGAINST:** "relax to equilibrium" may be a *driver/solver* over
an operator the language already has, not a new operator -- the equilibrium of `attraction_repulsion`
is just the fixed point of iterating its overdamped motion, and every knob `relax` exposes
(`max_steps`, `f_tol`, `ridge`) is a numerical tolerance with no physical content (contrast `agitate`,
which owns the physical `kT`/`gamma`), which reads as `out_of_scope` plumbing. It loses because the
fixed Lie-Trotter split applies each operator exactly ONCE per macro-step, so the omega-limit of
iterated composition is NOT expressible by composing registered operators (unlike NoForce = the free
empty composition); `relax` also performs a real forward transformation (position -> equilibrium, not
a zero no-op) and is potential-agnostic (the operation, not the force) -- but a verifier could still
land this at `out_of_scope` if they judge a convergence-gated solver to carry no biology of its own,
so this is the live fault line for the entry.


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

## neural_ode

<!-- NeuralODE -- append below; the driver merges this into campaign/analysis.md -->

## Normalization (normalizer)

**Verdict: `new`**, contract `regulate` (kind=exchange, family=fields, set=cell),
`implementation_of: regulate` -- consistent with the `odecontroller` base and the two gene-network
siblings. NeuralODE differs from `GeneNetworkConnectionist`/`GeneNetworkMWC` in exactly one place:
its vector field is a free-form per-cell MLP (`dy/dt = MLP([u, y])`) instead of a sigmoid-linear or
log-occupancy regulatory law. Every other commitment is identical -- it freezes the sensed drivers
`u`, seeds `y0 = concat(hidden, outputs)`, integrates with the same Dopri5/PIDController machinery,
and returns the integrated increment as a sparse dt-delta. That is the Morse/SoftSphere/Hertzian
shape: three interchangeable implementations of the one missing contract-slot -- a per-cell,
sensor-driven, latent-carrying internal regulatory ODE -- which no promoted operator provides
(`decay` is one degradation term of it; `pacemaker` is an autonomous clock with no sensed drive;
`sense` emits a heading, not integrated internal state). So the family yields ONE new contract with
three implementations, not three new contracts, and the yield is not inflated.

**Strongest argument AGAINST `new` (here, `out_of_scope`):** unlike its two siblings, NeuralODE has
NO paper counterpart and an *uninterpretable* right-hand side -- a generic `eqx.nn.MLP` wrapped in a
diffrax solver, doing no gene-specific biology. One could argue it is pure function-approximation
plumbing (a learnable black box + numerics) with no biological content, and that all the modeling
commitment lives in the structured gene circuits. I reject it: biological status here comes from the
contract SLOT the operator fills, not from the legibility of its reaction law. NeuralODE reads the
same sensed drivers, evolves the same coupled latent+output cell state, and persists it as the same
heritable genotype->phenotype decision function -- it is the *learned* implementation of `regulate`,
exactly as a fitted potential is still `adhere`. Recording it `out_of_scope` would hide a real
implementation of a real (and, per the ATLAS measurement, genuinely absent) contract. Its lack of a
paper counterpart is captured instead as a surprise (source wins: a paper-only reimplementer would
build the gene network, never this).


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

## NORMALIZER — verdict `out_of_scope`

NoForce is the null / identity element of the mechanical-interaction slot — the object
`potential=None` resolves to inside `BrownianDynamics`, `ActiveBrownianDynamics2D`,
`MechanicalRelaxation`, `relax_equilibrium`. Zero energy, zero force, no params, reads no field
values, writes nothing persistent, no paper counterpart. The decisive test: in the Plexus algebra
"no interaction" is the *empty composition* (include zero interaction operators) — the identity of
operator composition, which you get for free by composing nothing and never register as an operator.
jax-morph only materializes NoForce because its `Potential` slot must be filled by *some* object; a
Plexus model expresses the same free/non-interacting regime by omitting the interaction operator, so
there is no forward mechanism to normalize and no vocabulary gap. Not `alias`/`refinement` (there is
no registered no-op/identity contract to point at, and widening an interaction contract to admit
"reads nothing, writes zero, no map" would delete the pairwise force that IS that contract); not
`new` (a do-nothing operator would inflate the yield with content-free plumbing). Contract block
carries a formal `writes: [position]` (an identically-zero force-slot occupant) to clear R7, exactly
as the StochasticStep out_of_scope entry carries a formal `writes: [trace]`.

**Strongest argument AGAINST this verdict.** In an operator *algebra* the identity element is a
first-class member, not something to sweep out of scope. NoForce is type-level interchangeable with
Morse — same `Potential` protocol, consumed by the same integrator steps — so it could instead be
recorded as the degenerate `implementation_of` the pairwise-interaction contract (the zero member of
the Morse/SoftSphere/Hertzian family), a framing that captures a real modeling affordance:
jax-morph deliberately represents "no coupling" as a value of the same type as "Morse coupling," the
base case for active-matter/flocking models whose coupling lives in a separate alignment step. I
reject it because the atlas counts *biological contracts*, not type slots — NoForce reads nothing,
writes nothing real, has no map and no parameters, and is never built in the paper (it is the
`potential=None` sentinel Plexus realizes by leaving the operator out). But if the interaction
contract's registered signature ever made an explicit nullable-potential field load-bearing on a
wrapping step, this flips from `out_of_scope` to `implementation_of`.


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

## Normalization (normalizer)

**Verdict: `new`**, contract `regulate` (kind=exchange, family=fields, set=cell). No promoted
operator models a per-cell, continuous-time internal regulatory network -- an intracellular
gene-regulatory ODE that freezes the cell's sensed drivers, evolves a coupled latent+output gene
vector over the macro-step, and persists it as heritable state (the genotype->phenotype decision
function). The nearest registered contracts are `decay` (its `-gamma*g` degradation term is one line
of this network) and `pacemaker` (a per-cell time-varying signal), and widening either to admit an
input-forced, nonlinearly-regulated, stateful multi-gene circuit would destroy what each *is*
(elementwise evaporation; an open-loop clock that "owns only timing"). The base carries the contract;
its three subclasses are interchangeable implementations (`implementation_of: regulate`).

**Strongest argument AGAINST `new`:** because `vector_field` is *abstract*, the base commits to no
particular reaction law -- so one could argue the base ODEController is pure numerics (a diffrax
`diffeqsolve` wrapper: Dopri5 + PIDController + delta bookkeeping) and belongs in `out_of_scope`,
with ALL the biology living only in the three subclasses' vector fields. I reject it because the
base is not agnostic where it counts: it fixes the biologically-loaded *contract* -- per-cell
heritable internal state seeded from current genes, sensed inputs held FIXED across the step (the
quasistatic-chemistry assumption), and the integrated increment returned as an accumulated delta.
Those are modeling commitments, not solver plumbing, and the whole ATLAS measurement turns on
whether the language has this contract-slot at all -- it does not. (A weaker counter, that this is a
`refinement` of `decay`, is answered in the entry's `why:`.)


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

**NORMALIZER.** Verdict: **alias of `attraction_repulsion`** (implementation_of: attraction_repulsion). The
abstract pair-potential base is the energy-form statement of the pairwise attraction-repulsion contract Plexus
already registers: a radial cell-cell force with a repulsive core (excluded volume) and an adhesive tail — the
Morse well IS attraction_repulsion's "long-range pull minus short-range push", and both are the same
conservative radial pair force (attraction_repulsion's `f(r)·(pos_j−pos_i)` is itself the gradient of a radial
potential). Morse/SoftSphere/Hertzian/Harmonic/LennardJones are interchangeable force-law shapes over this one
contract (the pattern the registry is built to hold), and the NoForce normalizer already named exactly this
landing. Plexus's own criterion clinches it: the registered `stillinger_weber` docstring says every prior
interaction op is *pairwise* and only its *three-body* term earns a new contract — PairwisePotential is strictly
two-body, and SW is itself an energy-defined, autodiff-force interaction, so neither "pairwise" nor
"energy+autodiff" is novel here. **Strongest argument AGAINST (and why it loses):** the energy formulation is a
genuine capability attraction_repulsion lacks — a scalar potential U(r) unlocks two things the D'Orsogna
velocity law cannot give: a virial *pressure* readout and relaxation to a differentiable mechanical
*equilibrium* (FIRE / gradient descent). If the atlas counts "carries a conservative energy whose equilibrium is
itself differentiable" as contract-level content, this is at least a **refinement** (widen attraction_repulsion
to carry an energy, a pressure output, and an equilibrium-minimiser consumer), not a drop-in alias. It loses
because (a) attraction_repulsion's radial pair force is *already* conservative, so storing-and-differentiating an
energy vs hand-coding the force is a representation strategy, not new biology; (b) the family already spans both
integration modes (attraction_repulsion EMITs velocity, squared_law/stillinger_weber EMIT acceleration) and both
pair topologies (squared_law's all_pairs option; SW's dense min-image list), so none of that discriminates a
contract; and (c) the pressure and the equilibrium-relaxation are separate *registered* concerns (the VirialStress
and MechanicalRelaxation entries), not outputs of this interaction — folding them in would double-count. The one
residual is honest and recorded as a surprise, not a contract: the scalar/per-cell coupling restriction cannot
express the paper's type×type cadherin matrix — an expressiveness weakness of this implementation, not a wider
contract.


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

## saturating_cell_growth (NORMALIZER, status -> normalized)

**Verdict: `refinement` of `cell_grow`** (`of: cell_grow`, `implementation_of: cell_grow`). Same
biology as the registered growth primitive -- a cell grows toward a maximum size by a *saturating*
law -- so this is not `new` (a second growth contract would inflate the yield), and cell_grow is
unambiguously the closest contract (the only other growth-family op, `cell_divide`, is topology,
not size). But it does not fit cell_grow's *registered* signature, so it is not a clean `alias`
either: the signature must **widen** on four fields -- `kind` (structural, returns `{}` ->
delta-emitting `field` on `radius`, which breaks the invariant that growth adds nothing to the
dynamic-phase accumulation), `writes` (`grow_V` + child particle occ -> a `radius` delta on a
point-cell, dropping the required `mpm_particle` child), `reads`/params (constant `rate` param ->
heritable per-cell state field `growth_rate`, the differentiable-control design point), and the
growth `law` (logistic -> von Bertalanffy exact flow; additive, harmless).

**Strongest argument AGAINST (why this could instead be a plain `alias` + implementation_of):**
cell_grow's own docstring *already* declares the discretisation swappable and the growth law the
invariant -- "swap the discretisation ... and the growth LAW stays identical." Under that reading a
scalar radius on a soft-sphere cell is exactly the anticipated "another discretisation," so nothing
needs to widen; the structural-vs-delta and MPM-vs-scalar differences are realization plumbing
*beneath* the contract, and calling it a refinement over-costs a change the contract's authors
already sanctioned -- inviting the very "refinement hides a breaking change" failure record.py
warns about, in reverse (flagging a break that isn't one). **Why I still chose refinement:** the
frozen baseline record.py compares against is the *promoted language* -- `plexus.operators` and
nothing else -- i.e. the registered signature (`kind=structural`, `EMIT=None`, MPM-child
realization), not the docstring's aspiration. That registered signature genuinely cannot emit a
differentiable `radius` delta without an MPM child; admitting this mechanism *does* mutate the
frozen contract, and a change to the baseline is a refinement by definition -- one a downstream
user of structural-only growth must be shown the cost of. A downstream role that reads the source
and finds the registry can already host a delta-emitting growth realization *without* editing
cell_grow's signature should downgrade this to `alias`.


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

**Normalizer verdict: `out_of_scope`.** `step_type` is a class-level classification TAG, not a
forward operator over sets and fields — it declares no physics of its own and only fixes which of
three fixed macro-step phases a step runs in, how the Model reads its return value, and the
per-type write-conflict policy. That is engine/scheduling machinery composing the forward
operators, the same call the campaign already made for `StochasticStep` (order 5, differentiability
= engine concern). The contract fields are a validator formality (R6/R7 demand a typed,
writes-non-empty contract); the load-bearing observation is that step_type is a genuine
TIME-SCALE / integration-phase axis Plexus's IR does not carry — Plexus's own `kind` taxonomy is a
data-flow-SHAPE axis, orthogonal to it — so the axis is a gap in the meta-layer, not a missing
operator. **Strongest argument AGAINST:** step_type is not inert plumbing — it materially changes
what the simulation *computes*. The identical physics tagged `dynamic` (all writers evaluated at
one frozen post-quasistatic state, summed once — Jacobi) versus `quasistatic` (each step slaved to
the current updated state — Gauss-Seidel) yield *different trajectories*; a mis-tag silently runs a
step in no phase at all. Because the tag alters the dynamics and not just the bookkeeping, one
could argue it is modeling content deserving a first-class place in the algebra (an operator
attribute, or even a new time-scale `kind`), not dismissal as numerics. I reject this because
"which integrator composes these operators, in what order, reading which state" is the definition
of an engine/IR concern: it is a property *of* operators (how they compose in time), not an
operator that acts on state — and Plexus already locates operator-classification in its `kind`
meta-layer, exactly where this axis belongs if adopted, never in the forward vocabulary the atlas
counts. Promoting it as `new` would inflate the yield with framework machinery, which is precisely
the measurement out_of_scope protects.


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

--- normalizer pass ---
VERDICT: out_of_scope. StochasticStep is jax-morph's "hardened core" differentiability contract
(AGENTS.md: "stochastic trace replay/scoring", on which "physics and control steps compose"): an
abstract mixin whose own state_writes() and trace_writes() both default to (), carrying one
non-physical knob (score_by_default). Its entire content is the gradient-estimator machinery --
sample exogenous noise, replay pathwise, score logp -- REINFORCE for discrete events plus a
reparameterized pathwise branch for continuous ones. That is autodiff/optimization plumbing; in
Plexus differentiability is an engine concern (torch autograd), not an operator in the forward
algebra. The biology lives entirely in the subclasses (Division->cell_divide, Death->apoptosis,
Brownian->motion), each already its own atlas entry. No registered contract models gradient
estimation, and widening one to admit trace/replay/score would graft an optimization concern onto
a forward-dynamics signature -- so not new, not alias/refinement. The contract block is
schema-satisfaction only (R6/R7 force a typed contract at status=normalized); the ledger scores
this entry by its out_of_scope verdict and ignores those fields. Flagged the OVERLAP: the sibling
entry stochastic_trace___replay___score (order 24) is the SAME step.py:148 class and should get
the same verdict.

STRONGEST ARGUMENT AGAINST (the skeptic's line -- "find the biology hiding in the plumbing"): one
could argue the trace/replay/score contract is itself a genuinely NEW capability the promoted
language lacks -- Plexus has no notion of a differentiable stochastic step, and if the atlas is
meant to measure whether the vocabulary can express what the target does, then "an operator whose
effect is a scored random draw" is a real gap, arguably a `new` contract (call it `sample` or
`stochastic_event`) rather than out_of_scope. I reject this because it conflates two layers: the
FORWARD effect of every stochastic step already has a biological home in its subclass's own
contract (a Bernoulli division IS cell_divide), and what the mixin adds on top is purely the
gradient path -- REINFORCE vs pathwise -- which is an optimization/AD concern, not a forward
operation on state. Counting it `new` would inflate the operator yield with framework machinery,
exactly the failure the loop's out_of_scope verdict exists to prevent. The one thing that would
genuinely reopen this: if Plexus's engine could NOT recover the same optimization behaviour
through torch autograd on the subclass operators (i.e. if trace/replay/score encoded a forward
semantic the subclass contracts drop), then it would be a real gap -- I did not run that check
(jax is walled off here), so it stays an open question for the verifier, not a blocker on the
verdict.


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

--- normalizer ---
VERDICT: `out_of_scope` (contract stub `sample_score`, kept identical to the sibling `stochastic_step`
entry, which is the SAME core/step.py:148 class and was already normalized out_of_scope with an
explicit directive that this entry match). `StochasticStep` is an abstract differentiability mixin:
base `state_writes()` and `trace_writes()` both default to `()`, it writes no physical state and adds
exactly one non-physical knob (`score_by_default`); everything it fixes is gradient-estimator plumbing
(sample parameter-free noise, replay, score `logp`; the model's `pathwise = not scored` inversion).
In Plexus differentiability is an engine concern, not a forward operator over sets and fields, and the
forward biology already lives in the concrete subclasses (`Division` = `cell_divide`, `BrownianDynamics`
= motion), each its own entry -- so counting this as `new` would inflate the yield with framework
machinery, exactly what out_of_scope guards against.

STRONGEST ARGUMENT AGAINST: the ATLAS exists to find MISSING vocabulary, and one could argue the
sample/record/replay/score protocol is itself a genuinely new STRUCTURAL contract the promoted algebra
lacks -- a first-class "stochastic step" obligation (declare an ephemeral trace, own a `replay`, expose
`logp`) that any operator emitting a sampled action must satisfy. If Plexus cannot express how a
stochastic operator records and scores its own noise, that is arguably a real gap in its capacity for
differentiable stochastic morphogenesis, and calling it out_of_scope discards a contract as "mere
plumbing" that the language may in fact need. I reject this because the protocol has NO forward effect
of its own to normalize -- strip the subclass biology and only the gradient PATH remains (which
estimator, how the trace round-trips, where `stop_gradient` falls), and the typed signature it would
register is empty (no set, no maps, reads nothing, writes only the ephemeral `trace`). An operator with
an empty forward signature is the tell that it is autodiff bookkeeping, not an operation on state; the
biological content is fully accounted for by `cell_divide` and the motion contracts, with nothing left
over for the vocabulary to gain. `status: normalized`; evidence still null (no oracle run).


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

---

## Normalizer: verdict `new` -> contract `mechanosense`

**Verdict: `new`** — a standalone quasistatic MECHANOSENSOR. VirialStress reduces the same
pairwise potential and same live-non-self neighbour sum as `attraction_repulsion`, but contracts
it to a per-cell SCALAR observable (the Irving-Kirkwood virial pressure, normalised by `2 d V_i`)
that it writes to a transient `stress` field and that MOVES NOTHING. No registered contract
exposes a per-cell mechanical load as a pure-sensing readout for downstream mechanotransduction:
`active_stress`/`active_force` GENERATE stress/force into the MPM substep, `sense` reads a
diffusible field and steers heading, `aggregate` reduces children onto a parent. Contract:
`mechanosense`, `lateral`/`mechanics` (same taxonomy slot as gravity / mpm_anchor / mpm_spin),
set `cell`; reads position/radius/alive + the potential's coupling field, writes `stress`.

**Single strongest argument against it.** Plexus arguably ALREADY has mechanosensing —
`cell_grow` carries `stress_gain` (`mechano_inhibition`: "growth slows in deformed tissue"), which
is exactly the paper's "stress inhibits proliferation." If the language can already gate a cell's
fate on its local mechanical load, then VirialStress is not new vocabulary but merely the extracted
"sensor half" of a capability Plexus expresses — pushing toward `out_of_scope` (a redundant
intermediate) or a `refinement` of `attraction_repulsion`, whose base literally computes
`virial_pressure` right beside `forces()` and could simply also emit it. The rebuttal I stand on:
`cell_grow` reads the MPM CONTINUUM deformation gradient F (not a pairwise virial) and FUSES
sense+respond in one op, so the mechano-SENSE never exists as a reusable, first-class `stress`
field that any other consumer (a gene network, `cell_divide`, a differentiation switch) can read —
which is precisely the decoupled contract VirialStress mints and the promoted language lacks. But
this is the alternative I had to defeat, not a free win: if a later entry shows `cell_grow`'s
deformation readout and the virial pressure are interchangeable mechanical-load signals, then
`mechanosense` and that fused reading should be reconciled as one contract with two
implementations rather than left as separate vocabulary.

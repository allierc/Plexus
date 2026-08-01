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

## Normalization revised -- answering the skeptic (normalizer, 2026-07-31)

The skeptic disputed `new`, claiming `refinement` of `signal` (`src/plexus/operators/signal.py`):
`signal` is a registered per-agent internal-state ODE, `EMIT=velocity`,
`tau*dv/dt = -v + b + sum W*phi(v_pre)`, tag `recurrent` -- the SAME firing-rate-RNN class as
`GeneNetworkConnectionist` `dg/dt = sigma(W@g + W_in@u + b) - gamma*g`, and my predecessor's `why:`
never even named it (it called `decay`/`pacemaker` "closest", which was wrong). I agree with the
critique of the `why:` and have rewritten it to lead with `signal`; the verdict stays **`new`**.

**Strongest argument AGAINST `new` (the skeptic's, stated at full strength):** both operators are
literally the same math object -- a per-node first-order ODE whose drive is an activated linear
combination of node states plus a bias, minus linear decay. `signal` already carries that contract.
Its docstring even advertises the degenerate case ("drop the synapse state and the edge-set
collapses to a plain weighted connectome, one Lateral operator"). So the gene circuit is just
`signal` on a `cell` set with a self-loop connectome and pseudo-node inputs -- a `refinement` that
widens `signal`'s `set`/`maps`, not a new contract; two names for one recurrent-network operator is
exactly the `alias`/`refinement` inflation this loop exists to catch.

**Why it nonetheless fails.** `signal`'s typed identity is a CONNECTOME morphism: weights live on a
first-class `synapse` EDGE-SET read through `pre`/`post` incidence maps (`MAPS=[pre,post]`, "the maps
are PART of the signature"), and it `EMIT`s a velocity the ENGINE integrates. `regulate` has `maps=[]`
(dense per-operator matrix, zero cell-to-cell coupling -- purely intracellular), reads a FIXED sensed
forcing input `signal` lacks, and SELF-SOLVES the whole macro-step (adaptive Dopri5) to return the
exact `y(dt)-y0` delta. The skeptic's own `what_would_settle_it` sets the bar: a widening survives
only if it does NOT break `signal`'s connectome/engine-integration identity. Expressing `regulate`
forces changing the set, emptying the maps, dropping the edge-set, adding a forcing read, and flipping
engine-integration to self-solve -- it breaks exactly that identity. So the mismatch forces a distinct
operator and `new` stands. (`signal` and `regulate` are two contracts of the same recurrent-network
FAMILY, not two implementations of one contract -- their typed signatures differ in kind, set, maps
and integration mode.)


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

**IMPLEMENTER -- `reorient` written, imports/registers, 6 property tests pass.**
Module `src/plexus/operators/candidates/jax_morph_active_brownian_dynamics2_d.py`; test
`tests/test_jax_morph_active_brownian_dynamics2_d.py`. Implemented ONLY the uncovered
rotational-diffusion leg (the other two legs stay aliases of a pair potential and of `glide`).

**Key translation decision -- scalar angle -> unit-vector rotation.** The source carries
`active_heading` as a *scalar angle* and adds `dtheta` to it. In Plexus `heading` is a unit VECTOR
`[N,D]` (read by glide/bounce/sense), so I translated "add `dtheta` to the angle" into "**rotate the
heading vector by `dtheta`**", `h <- R(dtheta) h`. Exactly equivalent, and it buys two properties:
(1) **norm-preserving by construction** -- a planar rotation keeps `|heading|=1`, so I deliberately
do NOT renormalize (that keeps the norm-conservation test meaningful: it separates this correct
rotation from a naive additive `h += noise`); (2) **additive composition survives the change of
representation** -- planar rotations compose additively, `R(a)R(b)=R(a+b)`, so `reorient` in place
followed by another rotational heading writer equals summing the angle increments, reproducing the
source's dynamic-additive-delta accumulation under Plexus's in-place heading-steer idiom.

**Routing.** `EMIT=None`, mutates `heading` in place, returns `{}` -- the sibling of
`polarity_align`/`polarity_flow_align` (the polarity family owns every heading write; the engine
does not integrate heading). `SUPPORTED_DIMS=[2]` (scalar rotation angle -> planar rotation; a 3-D
version needs a rotation axis, and the source itself raises for `n_space_dim != 2`).

**No fitted constants.** `std_r = sqrt(2 * rot_diffusion * dt)` is the source's own scale (NO gamma
-- do not reuse the translational `std_t`); zero-drift (mean-0 increment); `rot_diffusion=0` is a
non-crashing identity limit; noise drawn at full padded capacity then gated by `occ`/`mask` (dead &
masked slots get `dtheta=0`), matching the source's `n_cells`-sized sampling; seeded RNG.

**Tests (all statable without the oracle):** zero-diffusion no-op; norm conservation (unit stays
unit); heading actually rotates; **zero-drift mean + variance = `2 D_r dt`** over 40k cells; variance
**linear in dt** (Brownian scaling); dead/masked slots unchanged. `evidence.oracle_run` stays null --
the differential test vs the oracle is the verifier's job.


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

## Implementation (implementer)

Wrote `agitate` at `src/plexus/operators/candidates/jax_morph_brownian_dynamics.py` (anti-chamber;
promotion is the curator's call after the differential test). `@register_operator("agitate",
family="motion", set="cell", kind="lateral")`, base `Lateral`, `EMIT="velocity"`, `SUPPORTED_DIMS=[2,3]`.
Imports and registers clean; 8 property tests pass (`tests/test_jax_morph_brownian_dynamics.py`).

**The one decision that matters: how the split dt-scaling survives the engine's integrator.** The
Plexus engine integrates an `EMIT="velocity"` delta as `pos += dt*v` -- it multiplies whatever we
return by `dt`. But the thermal DISPLACEMENT is a Wiener increment `dx = sqrt(2 kT dt/gamma)*xi`,
which scales as `sqrt(dt)`, not `dt`. To land a `sqrt(dt)` displacement THROUGH a `dt`-multiplying
integrator, `agitate` emits a velocity that scales as `1/sqrt(dt)`:

    v = sqrt(2 kT / (gamma*dt)) * xi     ->     dt*v = sqrt(2 kT dt/gamma) * xi   (correct)

This is the faithful torch translation of the source's `std = sqrt(2 kT dt/gamma)` applied as a
position delta. It also makes the composite reproduce Euler-Maruyama exactly: a drift operator
emits velocity `F/gamma`, the engine sums it with this thermal velocity, and one `pos += dt*v`
integration gives `dt*(F/gamma) + sqrt(2 kT dt/gamma)*xi`. Getting this wrong (a `dt`-independent
noise velocity, or scaling the noise by `dt` like the drift) makes the displacement variance scale
as `dt^2` instead of `dt` -- the wrong diffusion constant. This is the headline test.

**Decomposition honored: `agitate` owns ONLY the thermal leg.** Per the normalized contract, the
deterministic drift `F/gamma` is a SEPARATE pluggable pair-potential operator (soft_sphere /
hertzian / the Morse family / attraction_repulsion -- all already `velocity`-emitting, gamma folded
into their coefficients, confirmed in `attraction_repulsion.py:30`). `agitate` reads no potential
and adds no drift; it is the free Brownian gas (the source's `potential=None`) and composes with a
drift operator in the schedule. The whole StochasticStep trace/replay/logp machinery is engine
plumbing and is out of scope here. This is the sibling shape of `reorient`
(`jax_morph_active_brownian_dynamics2_d.py`), which likewise kept only the one leg
(ActiveBrownianDynamics2D's rotational diffusion) the frozen language lacked.

**Tests -- all reference-free (a limit, a scaling law, a symmetry; no oracle numbers):**
- `kT=0` limit -> emitted velocity exactly zero (the deterministic gradient-descent regime);
- Wiener `sqrt(dt)` scaling (headline) -- same-seed noise, quartering `dt` halves the displacement
  (velocity doubles), exact to 1e-6;
- FDT amplitude -- same-seed, displacement scales as `sqrt(kT)` and `1/sqrt(gamma)`, exact;
- Einstein diffusion + isotropy -- 40k cells, per-dim displacement variance = `2 kT dt/gamma`
  within 5%, zero mean, equal across dims (2-D and a separate 3-D case);
- alive/mask masking -- dead (`occ=0`) or masked-out cells draw a zero kick;
- pos-invariance -- forward returns a delta and never mutates `pos` (the integration guard);
- `n_space_dim` mismatch raises (the source's `_check_dim` surprise, kept as an optional assert).

**Faithful extras.** `kT` (default 0.1) and `gamma` (default 1.0) match the source; `kT<0` /
`gamma<=0` raise; `dt<=0` returns a zero kick. Noise uses `H.rng` (seeded generator) sized to the
full padded capacity and masked by `occ` -- the reference's alive-masking.

**NOT established (for the verifier / curator):**
- Oracle NOT run. `evidence.*` untouched -- no numerical check that a Plexus `agitate` (+ a Morse
  drift operator) trajectory matches `oracle.py` on a Brownian/relaxation script. The differential
  test is the curator's next step; the tests above fix the operator's contract, not oracle
  agreement, by design (a fitted constant would pass the differ and teach us nothing).
- RNG stream differs from the source (torch generator vs jax PRNG key), so a pathwise trajectory
  will NOT match sample-by-sample; the differ must compare distributional/statistical quantities
  (MSD growth, diffusion constant), or the `kT=0` deterministic case where only the drift moves
  cells (that one CAN match a jax-md gradient-descent relaxation numerically).
- The `sqrt(dt)`-velocity convention assumes the engine multiplies a `velocity` emit by `dt` once.
  Verified against the documented integration contract (base.py:524) and the sibling velocity
  operators (glide/sediment/attraction_repulsion), but not exercised end-to-end through
  `engine.run` here.


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


## Death -- implemented

Operator `apoptose` at `src/plexus/operators/candidates/jax_morph_death.py`; test at
`tests/test_jax_morph_death.py` (6 properties, all pass). Written as `cell_divide`'s literal
inverse and modelled on it line-for-line: same `Structural` base, `EMIT=None`, the same
`getattr(lvl, "<rate>", None)`-else-scalar-`rate` fallback (`death_rate` buffer here, `div_rate`
there), the same `getattr(H, "rng", None)` draw. The whole effect is `lvl.occ[die] = 0.0` -- occ
IS the Plexus analog of the source's `alive` mask, and retiring it (rather than parking/zeroing
mass) is the faithful minimal translation, because jax-morph cells are single particles with no
MPM child to retire. Reuse of the freed slot is deferred to a LATER macro-step exactly as the
source intends: `cell_divide` (which allocates `occ==0` slots) runs earlier in the pipeline under
divide-then-die, so a slot freed this step survives for lineage reconstruction.

Deliberate translations of the source's subtleties:
- Hazard `p = -expm1(-clip(rate,0)*dt)` copied verbatim (the `-expm1` form, and the `clamp(min=0)`
  guard so a negative controller output gives p=0, not a NaN score).
- `die` is re-AND'd with the LIVE mask via the eligibility set (`elig = live & at-mask`), so an
  already-dormant slot can never be marked "newly dead" -- the source's `(died>0.5) & alive` guard.
- `death` is a lazily-registered per-node FLOAT buffer, zeroed then set each step (OVERWRITTEN, not
  accumulated), dtype float so it is summable/differentiable while occ stays the boolean liveness.
- The `at:` mask gates eligibility (the source's `die_eligible`); a test confirms masked-out live
  cells survive.

NOT modelled (and why): the trace/`logp` score-function (REINFORCE) layer. Plexus's engine runs the
forward EFFECT only (the source's `replay`), so `died`/`die_eligible` traces and `Death.logp` have
no engine counterpart -- same scope as `cell_divide`, which realises the forward proliferation event
without its scoring term. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation): rate-0 no-op; negative-rate
clip -> no death; huge-rate -> certain death of every live cell (seed-independent, tests the hazard
form + the retire); dormant slots never die/revive and live-count is monotone non-increasing
(apoptose is strictly a remover); the float `death` record equals the exact occ 1->0 flip mask; and
the eligibility mask restricts who may die. No oracle run -- the paper ships no death config and jax
is absent from this env by design -- so `evidence.oracle_run` stays null for the differ/curator.

Name note: `apoptose`, not `death`, so no clash with the pre-existing efflux-boundary `death`
operator (candidates/death.py, kind `lateral`) -- a geometric exit-line sink, an unrelated
mechanism. Candidates are not auto-imported, so nothing registers until the differ/tests import it.


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



## Division -- implemented

Operator registered as `cell_divide` with `implementation="volume_conserving"` at
`src/plexus/operators/candidates/jax_morph_division.py`; test at
`tests/test_jax_morph_division.py` (10 properties, all pass). This is the faithful realization of
the `refinement` / `implementation_of: cell_divide` verdict: the registry keys operators by
`(name, implementation)` and only enforces that co-implementations share the contract `kind`
(structural == structural), so importing the candidate ADDS `volume_conserving` alongside the
promoted isotropic `default` on the SAME `cell_divide` contract -- the widened read/write set lives
declaratively in the atlas entry, the registry does not re-validate it. Default stays the promoted
mass-doubling impl, so the widening is additive for current callers exactly as the entry claims; the
candidate is not auto-imported, so nothing registers until the differ/tests pull it in. Same
multi-implementation pattern as `diffuse` (finite_difference/spectral) and `regulate`
(connectionist/mwc/neural_ode).

State representation (the reason this needed care). Plexus has NO standard `radius` state block --
`radius` is only a spawn-time scalar in `engine.py`, never carried per-cell -- so `radius`,
`division_rate`, and `division_axis` are modelled as per-cell BUFFERS the operator reads/lazily
provisions, mirroring `apoptose`'s `death_rate` convention. `born` is a float buffer, `mother` a long
buffer with the -1 founder sentinel, and `division_overflow` a 0-dim scalar buffer.

Deliberate translations of the source's subtleties:
- Volume factor `m = 2^(-1/d)` reads the live world dim `H.dim` (not a hardcoded 1/2), so both
  daughters take `r*m` and conserve the mother's d-volume (~0.707 in 2D, ~0.794 in 3D).
- The offset uses the NEW radius `(r*m)*dir`; mother moves to `x+offset` and daughter to `x-offset`,
  so the pair is centred on the mother's pre-division position and sits exactly touching (centre gap
  `2*r*m` = sum of radii). I capture `x_old`/`r_old` BEFORE any write so the two placements can't read
  each other back.
- Oriented direction `normalize(orientation_snr*a_hat + xi/sqrt(d))` with `a_hat = axis/(||axis||+1e-12)`;
  a zero axis or `orientation_snr=0` collapses to pure isotropic via the same 1e-12 guard (a test
  confirms the isotropic fallback still conserves volume and just-touches). With no `division_axis`
  buffer at all the direction is pure `xi`.
- Capacity is a hard wall: I draw the movers FIRST, then allocate `cap = min(movers, free)` and add
  `movers - cap` to `division_overflow` -- so the surplus is counted even when the buffer is
  completely full (`cap == 0`), matching the source's `sum(divide) - sum(committed)`. `born`/`mother`
  reset to their defaults every macro-step (a per-step lineage record); `division_overflow` is GLOBAL
  and accumulates (a test checks 4 dropped then 4+8=12 over two steps, deterministic at a huge rate).

One chosen divergence from the source, noted for the curator: jax-morph resets a daughter's
NON-heritable cell fields to their spec default and inherits only heritable ones; I inherit EVERY
per-cell buffer from the mother (like the promoted `cell_divide`'s spawn), then reset `born`/`mother`
explicitly. The heritable drivers (`division_rate`, `division_axis`, `celltype`) inherit correctly
either way; the only difference is that a recycled dead slot's stale non-heritable buffers are
overwritten with the parent's value rather than a default -- arguably safer, and consistent with the
sibling `cell_divide`. Plexus's own `Level.lineage`/`birth` buffers already record parent-slot
provenance, so `mother` is partly redundant with the container, but I write it explicitly because
`reconstruct_lineage` reads that exact field.

NOT modelled (same scope exclusion as `cell_divide`/`apoptose`): the straight-through / pathwise
differentiability and the `logp` score-function term. Plexus's engine runs the forward EFFECT only
(the source's `replay`); the `divided`/`divide_eligible`/`division_dir` traces and `Division.logp`
have no engine counterpart. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation, symmetry): rate-0 and
negative-rate (clip) no-ops; d-volume conservation `r_m^d + r_d^d == r_old^d` in 2D and 3D; the
just-touching centre-distance = sum-of-radii geometry; the symmetric split centred on the mother;
lineage (`born=1`, `mother=parent`, defaults elsewhere) with live-count growth; the overflow cap +
GLOBAL accumulation; the large-`orientation_snr` limit aligning the split with the axis; the
isotropic fallback; and the `at:` mask gating who divides. No oracle run -- jax is deliberately
absent from this env -- so `evidence.oracle_run` stays null for the differ/curator.


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

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_free_screened_diffusion.py` —
`MorphogenFreeSpace(Exchange)`, registered as the `morphogen` contract with
`implementation="free_space_greens_function"` (kind=exchange / family=fields / set=cell, matching
the normalized contract). This is the `diffuse` `finite_difference`/`spectral` pattern applied to
`morphogen`: the paper's graph-Laplacian inverse `c = (K I - D L)^{-1} S` can later register
`implementation="graph_laplacian"` on the SAME contract — two numerical methods, one biological
operator, which is the convergence the ledger is meant to record. `get_contract("morphogen")` now
resolves; `implementations = {"free_space_greens_function"}`.

**The load-bearing decomposition choice — an OVERWRITE, not a delta.** Unlike the gene-network
siblings (`regulate:*` return `dg/dt` for the engine to integrate), this is a QUASISTATIC CONSTRAINT
SOLVE: `dt` is meaningless, the field is the `t=infinity` equilibrium, so the operator OVERWRITES the
`chemical` block each step rather than incrementing it. It is therefore a *derived readout* in exactly
the `aggregate` (centroid) sense: `EMIT=None`, `MAY_MUTATE_INTEGRATED_STATE=True`, mutate state via
clone-and-assign of the `chemical` columns, return `{}`. `MAY_MUTATE_...=True` is required because the
engine's frame-0 integration guard clones the WHOLE `state` tensor and would otherwise flag the
in-place block write (it is the derived-readout exemption, not a force). Verified `pos` is invariant
across a forward (a test), so the exemption is honest — it only ever writes `chemical`.

**Faithful details carried over (diffusion.py:92-264):**
- Dimension-selected kernel by `pos.shape[1]` — 1-D segment `exp(-kappa(r_eff-a))/(2 D kappa)`, 2-D
  disk `K0(kappa r_eff)/(2 pi D a kappa K1(kappa a))`, 3-D sphere
  `exp(-kappa(r_eff-a))/(4 pi D r_eff (1+kappa a))`. The reference's static `n_space_dim` assert is
  preserved as an OPTIONAL param that raises on mismatch (surprise kept reproducible).
- `r_eff = max(r_ij, a_j)` surface clamp → the `i==j` diagonal contributes the on-surface value (SELF
  field included, tested). Source radius `a` is the EMITTER's, broadcast over receiver rows.
- Alive-masking applied TWICE and asymmetrically: sources masked over columns j (`* occ[None,:]`),
  receivers over rows i (`* occ[:,None]`) — tested both directions with a dead big-source slot.
- The three finiteness guards: `a <- max(a,1e-12)`, `kappa <- max(kappa,1e-12)` for 1-D/2-D only, 3-D
  left exact (bounded at kappa=0). The low-dimension screening requirement (`degradation>0` in 1-D/2-D)
  is raised at forward (where the spatial dim is known), matching the reference's constructor check.
- `safe_norm` ported verbatim (value+grad zero at the zero vector — the `where` trick, not a bare
  sqrt), so the diagonal r=0 does not poison gradients. Minimum-image applied if the world is periodic,
  with the documented caveat that free-space kernel + periodic box is a modeling error the user owns.
- JAX `vmap(in_axes=(0,0,1))` over species → a Python loop over the small static species axis;
  `diffusion`/`degradation` broadcast to `(n_species,)` (scalar or per-species). Multi-species
  independence tested.

**Bessel port (the one non-mechanical carry-over).** `_k0`/`_k1` are the reference's Abramowitz &
Stegun 9.8.5–9.8.8 rational/series approximations, ported to torch on `torch.special.i0`/`i1`. I did
NOT use torch's built-in `torch.special.modified_bessel_k0/k1` because those carry no autograd
backward (they would break `DIFFERENTIABLE=True`, the whole point of a jax-morph translation). I
cross-checked my port against torch's builtin K0/K1 over x in [1e-3, 20]: max relative error ~1e-7
(the A&S series precision) — so the disk kernel is faithful, not eyeballed. This also closes the
normalizer's open item "did not verify `_k0`/`_k1` against a reference over the full range."

**One deliberate robustness add (flagged, not hidden).** The reference always has `state.radius`; a
Plexus cell set may not carry a `radius` block. `_radii` reads a per-cell `radius` block/buffer if
present, else falls back to a UNIFORM default (the engine's `spawn radius` default 0.02). For the
oracle differential the source set carries a real per-cell radius, so this is inert there; it only
keeps the operator runnable on a radius-less set. Also: the contract's `READS` lists
`pos`/`radius`/`alive` explicitly — the reference `state_reads()` under-declares them (declares only
`secretion_rate`), the surprise the normalizer flagged.

**Test:** `tests/test_jax_morph_free_screened_diffusion.py` (7 pass). Headline property (reference-free):
SUPERPOSITION — the map `S -> c` is LINEAR (`c(3S)=3c(S)`, `c(S1+S2)=c(S1)+c(S2)`), the defining
property of a Green's-function steady-state solve, tested in 2-D so it exercises the Bessel path. Plus:
non-negative sources → non-negative field (kernel positivity); stronger `K` shortens the range (a
distant cell reads less); the self/diagonal field is included; a dead cell neither emits nor carries;
`pos` invariance under the solve; per-species independence. No oracle numbers hard-coded.

**FLAG for the curator's differential run (not a translation bug).** (1) Use a FREE (non-periodic)
world — the kernel is open-boundary; a periodic oracle would need the paper's method, not this. (2)
This is an OVERWRITE and quasistatic, so it must run BEFORE any step that reads `chemical` in the same
macro-step (the reference's `quasistatic -> dynamic -> discrete` phase order); schedule it first. (3)
Match `n_space_dim`/`diffusion`/`degradation` and per-cell `radius` to the oracle config; the finite
radius `a` has no paper counterpart, so agreement is a code-vs-code check, not code-vs-paper.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.


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

# gene_network_connectionist (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_connectionist.py` —
`RegulateConnectionist(Exchange)`, registered `@register_operator("regulate", family="fields",
set="cell", kind="exchange", implementation="connectionist")`. Torch, not JAX. Test:
`tests/test_jax_morph_gene_network_connectionist.py` (5 property tests, all green). `status ->
implemented`; evidence left null for the differ.

**The one real design decision — SELF-SOLVE vs emit-a-rate.** The source is an `ODEController`: it
integrates the ODE over the whole macro-step with adaptive Dopri5 and its DYNAMIC step returns the
*sparse endpoint delta* `g(dt)-g(0)`, not an instantaneous rate. Plexus's engine integrates a
first-order block with one Euler step per tick (`g += dt*delta`). I chose to reproduce the reference
ENDPOINT: integrate internally over `[0,dt]` with fixed-step RK4 (`substeps`, default 8) and return
the *mean rate* `(g(dt)-g(0))/dt`, so the engine's `dt*` recovers `g(dt)` exactly (the dt cancels —
not a second integration). This is the operator's defining behavior per its own equations/surprises
("emits the SPARSE DELTA … integrated over one macro-step by Dopri5"), and it maximizes fidelity for
the coming differential test. Fixed-step RK4 vs adaptive Dopri5 is a numerics choice inside the one
contract; `substeps` tightens it. NB the `mwc` sibling made the OPPOSITE call (emit instantaneous
`dg/dt`, let the engine Euler-step, invoking "integration is the engine's concern") — a genuine
unresolved tension across the four `regulate` files that the curator should reconcile when promoting.

**SOURCE WINS, faithfully ported.** (1) Sensed input enters INSIDE the sigmoid via a trainable
`W_in @ u` (code), NOT as an additive `+ I_i` outside it (paper) — implemented the code. (2) `sigma`
is the ALGEBRAIC `0.5 + 0.5 x/sqrt(1+x^2)` (`_rescaled_sigmoid`), not logistic — ported including the
overflow guard (clip to finite range, rescale by max(1,|x|)). At extreme drives it saturates to
*exactly* 0/1 (float32 underflow of `scaled_one^2`); that is correct, so my sigmoid test asserts
strict `(0,1)` only for moderate drives and mere finiteness at 1e30. (3) Defaults are NOT inert:
zeros for W/W_in/b but `gamma=0.1`, so `dg/dt = 0.5 - 0.1 g` drives every gene to `g*=5.0`. (4)
`gamma` stored verbatim, NOT shape-checked (the one param the source skips through `_resolve_param`);
matrices materialize lazily on first forward and ARE shape-checked. `u` is a frozen quasistatic
snapshot for the whole solve.

**Routing.** `INTEGRAND="gene"` (instance-set to the configured block; class stays `"gene"` so
`_resolve_emit` sees a non-`pos` integrand and does not constrain the coordinate order), `EMIT="velocity"`,
`MAPS=[]`. Sensed drivers `u` are read as per-cell STATE BLOCKS (the reference `_pack`s per-cell input
specs — they are not grid fields), named by `inputs:` (a name or list, concatenated). Dormant cells
(`occ=0`) get a zero delta.

**Property tests (all reference-free — stated from the operator's own definition):** (1) production
`= dg/dt + gamma*g` is strictly in (0,1) for random large drives (algebraic-sigmoid range); (2)
`sigma(0)=0.5`, monotone, overflow-finite; (3) zero-interaction circuit is not inert and vanishes at
`g*=5.0`; (4) end-to-end through `build`+`_integrate`: the engine result equals the internal
self-solve and CONTRACTS toward `g*=5.0` on the original side; (5) dormant cells don't evolve.

**Heads-up for the curator (registration overlap, not a bug).** The `odecontroller` entry (order 4,
the abstract base) also landed a `regulate`/`connectionist` file (`jax_morph_odecontroller.py`, an
adaptive-Dopri5 self-solve) as its concrete representative. Mine is the CANONICAL connectionist
entry (order 1). Candidates are never bulk-imported (the anti-chamber is full of intentional name
clashes), so each imports/tests alone — no runtime collision — but the two `regulate`/`connectionist`
modules are the same operator and should collapse to one on promotion. That collapse (four
ODEController entries → one `regulate` contract with implementations connectionist/mwc/neural_ode) IS
the convergence result the ledger is meant to record.

**Did NOT run** the oracle (differential test is the differ's job; evidence stays null).


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

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_mwc.py` — the `regulate`
contract registered with `implementation="mwc"` (`RegulateMWC(Exchange)`, kind=exchange /
family=fields / set=cell, matching the normalized contract). This is the diffuse
`finite_difference`/`spectral` pattern applied to `regulate`: the connectionist and neural-ODE
siblings can later register `implementation="connectionist"` / `"neural_ode"` on the SAME
contract, so the gene-network family stays ONE contract with three interchangeable vector
fields — the convergence the ledger is meant to record. `get_contract("regulate")` now resolves;
`implementations = {"mwc"}`.

**The load-bearing decomposition choice.** The reference's `ODEController.__call__` self-solves
the ODE over the macro-step with diffrax Dopri5 (adaptive) and returns the increment
`y(dt) - y0`. Plexus separates biology from time-stepping: the operator returns the *vector
field* `dg/dt` as a first-order delta on the cell's `gene` block (`EMIT=velocity`,
`INTEGRAND=gene`), and the ENGINE integrates it (`x += dt*delta`, engine.py `_integrate`). This
is the `signal` precedent verbatim (`signal` returns `dv/dt` and lets the engine integrate the
neuron voltage). So the three ODEController subclasses differ ONLY in `f` — exactly the record's
"share the integration+I/O contract, differ only in the vector field" claim, now realised in
code. Sensed inputs `u` are read from a per-cell `sensed` state block and never written (held
fixed across the step), matching the reference's quasistatic-chemistry `_pack(state,
input_specs)`; the upstream `sense`/`chemotax` step is what fills that block.

**FLAG for the curator's differential run (not a translation bug).** The engine takes ONE
explicit-Euler step per tick where the reference takes an adaptive Dopri5 sub-solve of the same
vector field over the same interval. For a non-stiff field at small `dt` the gap is small but
NONZERO, and it will show up in the differ. It is an Axis-A (integration) difference, not a
difference in the mechanism `f`. To compare `f` itself, evaluate the vector field at `t=0`
(`RegulateMWC.vector_field(evolving, inputs)`) against the reference's `vector_field` on matched
state — that isolates the biology from the solver. Multi-step trajectory agreement would require
substepping the gene block (a `substep_dt` micro-loop) or an RK integrator in the engine; I did
NOT do that, deliberately — self-integrating inside the operator is the "category error" the
engine guards against on frame 0.

**Faithful details carried over (ode.py:449-475):** occupancy uses genes/drivers CLAMPED to >=0
(ln(1+.) would NaN on negatives) while the DECAY term uses the RAW evolving state (restorative
toward 0 — the intentional asymmetry); the g/K ratio is capped at `finfo.max` before `log1p`
(the mixed-sign +inf/-inf overflow guard); rho/tau/K come through `_positive_from_log` (clip to
[log tiny, log max - log 4] then exp, so unset log-0 params give rho=tau=K=1); the sigmoid is
`torch.sigmoid` (logistic), NOT the sibling connectionist's rescaled algebraic sigmoid. `H_gene`
orientation verified non-transposed: `H_gene[i,j]` = weight of regulator j on target i (checked
against an asymmetric 2-gene circuit).

**Test:** `tests/test_jax_morph_gene_network_mwc.py` (6 pass). Headline property (a limit,
reference-free): the production term `rho*sigmoid(F)` is STRICTLY in `(0, rho)` for any genes,
drivers, and finite params — the definition of saturating production, and what separates MWC
from an unbounded linear drive. Plus: the inert circuit's fixed point (`dg = 0.5 - g`), the
activating(+)/inhibitory(-) sign of `H`, the restorative-decay asymmetry on a negative
concentration, dormant cells frozen, and one end-to-end engine-integration check. No oracle
numbers hard-coded.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.


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

---

## Implementation (candidate `adhere:hertzian`)

Wrote `src/plexus/operators/candidates/jax_morph_hertzian.py` — the `hertzian` implementation of
the `adhere` contract, registered `@register_operator("adhere", family="interaction", set="cell",
kind="lateral", implementation="hertzian")`. Imports/registers clean; the contract reports
`kind=lateral family=interaction set=cell`, `implementations=['hertzian']`. (SoftSphere, the twin,
is not yet coded, so this registration MINTS the `adhere` contract with `hertzian` as its default
implementation; a later `adhere:soft_sphere` will add the second implementation to the same
contract — the normalized `implementation_of: adhere` in the entry is what makes that legal.)

**Faithful to the SOURCE, not the analytic shortcut.** The operator does not hand-write the radial
force. It builds the total conservative pair energy `E = 0.5 * sum_{live i!=j} (2/5) eps (1 -
r/sigma)^(5/2)` and takes `force = -autograd.grad(E, pos)`, exactly mirroring the source's
`forces = -jax.grad(total_energy)` (potentials.py:L84/:L228). This is deliberate: it REALIZES the
2/5 = 1/exponent trick (autodiff of the energy with the 2/5 in place yields the intended unit force
coefficient `f = (eps/sigma)(1-r/sigma)^(3/2)`) rather than smuggling a fitted constant into a
hand-written force — which is the trap the loop warns against. Ported both source guards verbatim:
`_safe_divide` / `_safe_norm` (double-`where`, finite grad at the sigma=0 padded pair and the r=0
self-diagonal) and the `_compact_repulsion` double-`where` (the fractional power only ever sees a
strictly-positive base). Dead/self masking is EXTERNAL (an alive-mask `where` on the energy, the
`neighbor_sum` seam), not inside the per-pair law — matching the surprise the entry records.

**Routing decisions.** `EMIT="velocity"` (mobility * F): the paper's mechanics is overdamped
(gradient-descent `MechanicalRelaxation` / overdamped `BrownianDynamics`), so the force is a
velocity the engine integrates and sums with any other velocity a cell carries — same routing as
the registered near-miss `attraction_repulsion`. `sigma = r_i + r_j` (ADDITIVE) read from the
per-cell `radius` buffer; per-cell `epsilon` (via optional `epsilon_field`) mixed by the ARITHMETIC
MEAN `0.5*(eps_i+eps_j)`, else a shared scalar — the two-different-combining-rules surprise is
implemented and tested. `mobility` (default 1.0) is the sole added knob (the overdamped 1/gamma),
NOT a fitted constant. DENSE N×N half-summed (the source is dense `neighbor_sum`, not a graph);
fine at atlas cell counts, O(N^2) noted in the docstring. Dimension-generic (`SUPPORTED_DIMS=[2,3]`,
reads D from pos). Made `create_graph` follow `pos.requires_grad and grad_enabled` so a plain
rollout under `no_grad` is cheap/detached while a differentiable inverse loop keeps the force
connected to the state graph.

**Test** `tests/test_jax_morph_hertzian.py` — 8 properties, all stated from the calculus of the
energy, none from the oracle: (1) overlap is repulsive with the EXACT analytic magnitude
`f=(eps/sigma)(1-r/sigma)^(3/2)` and correct direction; (2) compact support — zero force AT and
beyond contact (no tail, no cutoff param); (3) C2 softness — |force| falls monotonically to ~0 as
overlap → 0; (4) Newton's third law — total force over a 12-cell cluster sums to 0 (momentum
conservation of a conservative pair energy); (5) size-consistency — additive sigma, same overlap
fraction → same force fraction, and unequal radii with equal sigma give equal force; (6) dead-cell
masking is external (a dead cell coincident with a live one perturbs nothing); (7) per-cell epsilon
mixes by arithmetic mean (eps=(2,4) acts like shared 3); (8) the force is a real autodiff (grads
flow to positions, finite — no NaN from the fractional power). `8 passed`. Also hand-checked
no_grad rollout (detached forces, 0.5^1.5=0.35355), single-cell no-pair (zero), and 3D.

**Did NOT do / left open (unchanged from normalization):** no oracle run — jax is deliberately
absent from this env AND no oracle script instantiates Hertzian (it would need a new script), so the
`evidence` block stays null and the energy/force shape is confirmed only against hand-derived
calculus, not measured against jax-morph. The differential test against the oracle is the curator's
next step; a passing property test is NOT that. The family-wide reconciliation (pull the abstract
parent + Morse + Harmonic toward `adhere`, and reconsider whether `adhere` is the right name for a
purely-repulsive member) remains flagged for the analysis phase — not settled here.


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

**Implementation (IMPLEMENTER role).** Wrote `src/plexus/operators/candidates/jax_morph_mechanical_relaxation.py`
(the anti-chamber) and `tests/test_jax_morph_mechanical_relaxation.py`; `status: implemented`.
Torch, not JAX. Registered `@register_operator("relax", family="mechanics", set="cell", kind="lateral")`,
`EMIT="velocity"`, `DIFFERENTIABLE=False`. Imports + registers + 7 property tests pass.

Design decisions (and the impedance mismatches they resolve):
- **Quasistatic overwrite through the delta contract.** The reference step OVERWRITES position with
  the equilibrium `x*` (`state.set('position', x*)`). Plexus has no quasistatic phase and the
  integration invariant forbids an operator writing `pos` directly, so I express the overwrite as a
  velocity: the engine integrates `pos += dt*v`, so emitting `v = (x* - x0)/dt` lands `pos` exactly
  on `x*` in ONE macro-step, for ANY `dt`. That `1/dt` factor is the SIGNATURE of a quasistatic
  step -- the exact dual of `agitate`'s `1/sqrt(dt)` Wiener emit. A test drives three `dt` values and
  confirms the landed configuration is identical (dt-independence).
- **Potential-agnostic solver.** `relax` is the SOLVER, not the force law: a `potential:` param
  selects the energy family it relaxes (`soft_sphere` default / `hertzian` / `harmonic` / `morse` /
  `lennard_jones` / `none`=NoForce no-op). I ported the source's `_compact_repulsion` / `_smooth_cutoff`
  / `safe_divide` / `safe_norm` and the five sigma-relative (`sigma = r_i + r_j`) pair energies, and
  take the force by autodiff of the energy -- so `relax` composes with any of them, orthogonal to
  which interaction. Smoke-ran all five: the adhesive ones (morse/LJ/harmonic) settle to exactly the
  well minimum at contact (sep 1.000), the repulsive ones to at least contact.
- **FIRE, faithfully.** `_fire_to_tol` reproduces the Bitzek-2006 schedule verbatim (carried velocity
  mixed toward the force direction; global power/norm reductions; dt grows after `n_min` downhill
  steps, resets with the velocity on an uphill step); stops at `|grad U|_inf <= f_tol`, `max_steps` a
  fallback. Forward-only: `_force` evaluates `-grad U` on a fresh detached leaf under a local
  `enable_grad` (the engine generates under `no_grad`), so the solver path is never in any outer graph
  -- the source's `custom_vjp` stance ("the answer depends only on `x*`, not how FIRE got there").

What I did NOT reproduce (honest gaps, deliberately):
- **The implicit-diff backward.** `DIFFERENTIABLE=False`. The source's distinctive gradient -- the IFT
  sensitivity `dx*/dp = -H^{-1} d(grad U)/dp`, CG-solved on the physical subspace with the rigid-body
  gauge modes projected out -- is NOT implemented. Reproducing it as a `torch.autograd.Function` with a
  projected-Hessian CG solve is the promotion follow-up. The FORWARD equilibration (what the engine
  runs, what the differ compares) is faithful; the flag correctly steers an inverse loop past this
  implementation.
- **No oracle diff.** `evidence.*` still null -- comparing our relaxation trajectory to `oracle.py` is
  the CURATOR's differential test. I did not run it. My tests assert only contract properties statable
  WITHOUT the reference (force balance `|grad U| <= f_tol`; energy non-increase `U(x*) <= U(x0)`;
  overlap resolved APART to >= contact with centre-of-mass conserved; NoForce no-op limit; translation
  symmetry in free space; dt-independent one-step landing; dead cells held fixed). No fitted constants.
- **Per-cell `alpha` field** (morse well-width as a `StateFieldSpec`) is not supported -- only per-cell
  `epsilon_field`. The source allows either; I judged scope. Note for a reuser.


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

## Implementation (implementer)

Wrote the `neural_ode` implementation of `regulate` at
`src/plexus/operators/candidates/jax_morph_neural_ode.py` (anti-chamber; promotion is the curator's
call after the differential test). `@register_operator("regulate", family="fields", set="cell",
kind="exchange", implementation="neural_ode")`. The `connectionist` sibling was ALREADY implemented
(`jax_morph_odecontroller.py`); it established the `regulate` interface, and the whole point of the
"three implementations of one contract" framing is that they be genuinely INTERCHANGEABLE. So I did
not invent a second interface -- `neural_ode` is a DROP-IN sibling of `connectionist`, byte-for-byte
identical except the vector field:

* SAME signature/routing: `EMIT="velocity"`, class `INTEGRAND="gene"` + instance-`INTEGRAND` routing,
  `MAPS=[]` (intracellular), one first-order `state:` block (`gene` = the source's `concat(hidden,
  outputs)`), a read-only frozen `inputs:` driver block, `hidden_size` as a documented latent/output
  split that does not change the integration (whole block solved together).
* SAME inc/dt EMIT scaling: the source is a DYNAMIC step returning the exact increment `g(dt)-g0`
  which the Model ADDS; Plexus does `g += dt*delta`, so we return `(g(dt)-g0)/dt` and the engine's
  `dt*` recovers the endpoint (dt cancels; not a second integration).
* SAME adaptive Dopri5(4): I mirrored the sibling's batched Dormand-Prince 5(4) verbatim -- same
  tableau, embedded 4th-order error, RMS error norm over the stacked per-cell state (diffrax's
  default -> one shared step sequence), I-controller `h *= clamp(0.9*err^-0.2, 0.2, 5)`, first step
  = dt. Mirroring (not swapping in torchdiffeq) keeps the two siblings differing ONLY in `_field`,
  which is exactly the contract's promise. NOT hard-coded to the oracle.

The ONE difference -- the vector field. `_field(g, u) = MLP(concat(u, g))` (autonomous, `u` frozen
and closed over for the whole solve). The MLP is a torch `nn.Sequential` mirroring `eqx.nn.MLP(in,
out, width, depth)`: `depth+1` Linear layers, `activation` after every layer but the last, identity
final activation; `make_mlp` defaults reproduced (width=64, depth=2, relu = jax.nn.relu). It maps
`n_in + n_gene -> n_gene` (the source's `in_size + hidden + out_size -> hidden + out_size`). The net
is built LAZILY on first forward once the block widths are known from the actual cell state (the
Plexus `__init__(params, device)` has no Hierarchy, unlike the source ctor that takes a pre-sized
`mlp`); an optional `net=` param injects a pre-built module (the source's "constructor takes a
pre-built mlp" path, used by the tests), and its shape is validated with the source ctor's ValueError
(`must map in+hidden+out -> hidden+out`). `seed` re-inits deterministically without disturbing the
global RNG. `activation` picks relu/tanh/softplus/sigmoid/gelu.

SOURCE WINS / paper: nothing to reconcile inside the field -- NeuralODE has no paper counterpart and
an uninterpretable RHS (already the entry's headline surprise); the paper-vs-code sigmoid/forcing
contradiction is a `connectionist`-only concern. `DIFFERENTIABLE=True` is honest: the RK steps are
plain torch ops, so autograd flows through the solver (the source is diffrax+equinox differentiable).

Tests: `tests/test_jax_morph_neural_ode.py`, 8 passing, all reference-free (a CONFIGURED known field,
never a fitted oracle number): (a) inject a LINEAR field `dg/dt=-k*g` (single Linear, y-columns only)
and check the increment lands on analytic `g0*exp(-k*dt)` to 1e-4 through `g += dt*delta`; (b) the
same endpoint through the real engine `_integrate` (the `_delta_blocks` routing composes); (c)
driver-freezing -- when the net ignores `u`, two different `u` give the identical increment; (d)
zero field -> ~zero delta (float32 solver roundoff ~1e-6 only); (e) dormant (occ=0) cells get zero;
(f) `hidden_size` split integrates the whole `gene` block as one coupled vector; (g) wrong-shape
injected net rejected at forward; (h) `neural_ode` and `connectionist` register as two
implementations of the ONE `regulate` contract with shared `EMIT`/`INTEGRAND`/`MAPS`.

Could NOT do (left for the validator/curator): the differential run against the oracle. jax is
deliberately absent from the Plexus env, and -- per the excavator's UNKNOWN -- no oracle script /
smoke trajectory is known to instantiate NeuralODE (vs the gene-network controllers), so there is no
paired scenario to diff against yet. Both this and the oracle use adaptive Dopri5 at the same rtol,
so tolerance-level agreement is expected once a scenario exists, but that is the validator's evidence
to produce. `status` advanced only to `implemented`; `evidence`/`status: validated` untouched.


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

## Normalization revised -- answering the skeptic (normalizer, 2026-07-31)

The skeptic disputed `new`, claiming `refinement` of `signal` (`src/plexus/operators/signal.py`):
`signal` is a registered per-agent internal-state ODE, `EMIT=velocity`,
`tau*dv/dt = -v + b + sum W*phi(v_pre)`, tag `recurrent` -- the SAME firing-rate-RNN class as
`GeneNetworkConnectionist` `dg/dt = sigma(W@g + W_in@u + b) - gamma*g`, and my predecessor's `why:`
never even named it (it called `decay`/`pacemaker` "closest", which was wrong). I agree with the
critique of the `why:` and have rewritten it to lead with `signal`; the verdict stays **`new`**.

**Strongest argument AGAINST `new` (the skeptic's, stated at full strength):** both operators are
literally the same math object -- a per-node first-order ODE whose drive is an activated linear
combination of node states plus a bias, minus linear decay. `signal` already carries that contract.
Its docstring even advertises the degenerate case ("drop the synapse state and the edge-set
collapses to a plain weighted connectome, one Lateral operator"). So the gene circuit is just
`signal` on a `cell` set with a self-loop connectome and pseudo-node inputs -- a `refinement` that
widens `signal`'s `set`/`maps`, not a new contract; two names for one recurrent-network operator is
exactly the `alias`/`refinement` inflation this loop exists to catch.

**Why it nonetheless fails.** `signal`'s typed identity is a CONNECTOME morphism: weights live on a
first-class `synapse` EDGE-SET read through `pre`/`post` incidence maps (`MAPS=[pre,post]`, "the maps
are PART of the signature"), and it `EMIT`s a velocity the ENGINE integrates. `regulate` has `maps=[]`
(dense per-operator matrix, zero cell-to-cell coupling -- purely intracellular), reads a FIXED sensed
forcing input `signal` lacks, and SELF-SOLVES the whole macro-step (adaptive Dopri5) to return the
exact `y(dt)-y0` delta. The skeptic's own `what_would_settle_it` sets the bar: a widening survives
only if it does NOT break `signal`'s connectome/engine-integration identity. Expressing `regulate`
forces changing the set, emptying the maps, dropping the edge-set, adding a forcing read, and flipping
engine-integration to self-solve -- it breaks exactly that identity. So the mismatch forces a distinct
operator and `new` stands. (`signal` and `regulate` are two contracts of the same recurrent-network
FAMILY, not two implementations of one contract -- their typed signatures differ in kind, set, maps
and integration mode.)

## Implementation (implementer)

Wrote `regulate` at `src/plexus/operators/candidates/jax_morph_odecontroller.py` (the anti-chamber;
promotion is the curator's call after the differential test). Registered
`@register_operator("regulate", family="fields", set="cell", kind="exchange",
implementation="connectionist")` -- the `connectionist` reaction law is the shipped implementation;
`mwc` (log-occupancy) and `neural_ode` (MLP field) are sibling implementations of the SAME contract
(same signature, same self-solve, only the vector field differs), exactly the diffuse
finite_difference/spectral shape. Test: `tests/test_jax_morph_odecontroller.py`, 5 passing.

Three engine-mapping decisions, each faithful to the source but forced by Plexus's integration model:

1. **The evolving state is ONE first-order block `gene`** = the source's `y = concat(hidden, outputs)`
   (surprise #8: they are integrated as one coupled system). Plexus routes a delta by LEVEL, with a
   single `INTEGRAND` block per operator, so the two persisted fields the source slices apart cannot
   be two separate Plexus blocks written by one op; the coupled vector is their honest joint form.
   `hidden_size` is kept as a documented latent/output split but does not change the integration (the
   whole block is solved). This is the FIRST registered operator to use the non-coordinate
   `_delta_blocks` path (grep confirms no other operator sets `INTEGRAND`); the machinery existed
   unused. Class `INTEGRAND="gene"` (so `_resolve_emit` sees a non-`pos` integrand and never
   constrains the cell's coordinate order); instance `self.INTEGRAND` routes to the configured block.

2. **inc/dt EMIT scaling.** The source is a DYNAMIC step: it self-solves and returns the exact
   increment `g(dt)-g0`, which the Model ADDS (surprises #1/#2). Plexus integrates a first-order block
   as `g += dt*delta`, so the operator returns the effective mean rate `delta=(g(dt)-g0)/dt` and the
   engine's `dt*` recovers the exact endpoint -- the dt cancels; it is NOT a second integration. This
   preserves "exact integrated increment, not rate*dt" while obeying `EMIT=velocity`.

3. **SOURCE WINS on the forcing input.** Implemented the code's `sigma(W_gene@g + W_in@u + b) -
   gamma*g` (drive INSIDE the sigmoid), not the paper's additive-outside `+ I_i`, because the
   differential test compares us to the running source. Sigmoid is the ALGEBRAIC
   `0.5+0.5 x/sqrt(1+x^2)` (source's `_rescaled_sigmoid`), computed via `hypot(1,x)` for stability.
   Drive `u` read from a frozen `inputs` block (integration=none), closed over for the whole solve.

Translated `diffrax.Dopri5() + PIDController(rtol=1e-4, atol=1e-6), dt0=dt` as a genuine batched
adaptive Dormand-Prince 5(4): DP tableau, embedded 4th-order error, RMS error norm over all elements
(diffrax's default -> one shared step sequence over the stacked per-cell state), I-controller
`h *= clamp(0.9*err^-0.2, 0.2, 5)`, first step = dt. NOT hard-coded to the oracle. Sanity-checked
the endpoint against a 20000-step fixed RK4 on a nonlinear coupled 2-gene+drive case: max abs diff
7.2e-5 (< rtol), so the tableau and control are correct. Dead cells (occ=0) get a zero delta
(gene state frozen) -- the same net effect as the source's post-hoc alive-mask (surprise #9).

Test properties (all reference-free -- exact solutions of the drive-frozen scalar ODE, since W_gene=0
makes the drive constant in g): (a) linear-decay increment exact -- `dg/dt=0.5-g` from g0=0 lands on
`0.5(1-e^{-dt})` to 1e-4; (b) same through the engine `_integrate` from g0=0.3; (c) fixed point
g*=0.5/gamma is stationary (delta~0); (d) frozen drive forces g*=sigma(k*u) (input path) and the
`drive` block is unchanged after the step; (e) dormant cells hold their gene state.

Could NOT do (left for the validator/curator): the differential run against the oracle -- I did not
run jax (deliberately absent from the Plexus env) and did not build the paired ODE scenario. The
oracle uses adaptive Dopri5 at the same rtol; both should converge to the true solution to ~1e-4, so
tolerance-level agreement is expected, but that is evidence for the validator to produce, not me.
`evidence`/`status: validated` stay untouched (status advanced only to `implemented`).


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

## saturating_cell_growth (IMPLEMENTER, status -> implemented)

Wrote `src/plexus/operators/candidates/jax_morph_saturating_cell_growth.py` (anti-chamber) and
`tests/test_jax_morph_saturating_cell_growth.py` (9 passing). `module`/`test`/`status` updated.

**The refinement is REJECTED by the registry -- and that settles the normalizer's refinement-vs-alias
debate toward refinement.** The normalizer left the door open: "a downstream role that finds the
registry can already host a delta-emitting growth realization *without* editing cell_grow's signature
should downgrade this to `alias`." It cannot. `@register_operator("cell_grow", implementation=...,
kind="lateral"|"field")` raises at import next to the shipped structural `cell_grow`:
`operator 'cell_grow' implementation 'saturating_radius' has kind 'lateral', but the contract's kind
is 'structural'; implementations may differ only in numerics` (registry.py:131). So this mechanism
CANNOT be a silent second implementation of the frozen contract -- admitting it requires deliberately
widening cell_grow's kind, i.e. a language change a downstream user must be shown the cost of. That is
exactly the "refinement hides a breaking change" the ledger exists to catch, here enforced by the
registry itself, not merely argued. The candidate therefore registers under the distinct name
**`grow_radius`** (family=growth, set=cell); unifying it with cell_grow is the curator's call.

**`kind: field` is read as "delta-emitting"; the faithful concrete kind is `lateral`.** Plexus's
`field` kind is a FIELD's own grid self-dynamics (diffuse/decay), `EMIT=None`, `forward()` returns
`{}` -- it CANNOT emit a per-cell `radius` delta. The operator is a per-cell autonomous ODE with no
neighbour coupling and no field, which is precisely the shipped `attractor_flow`/`signal` shape
(`dx/dt=f(x)`, kind=lateral). So I registered `kind=lateral` and recorded the field->lateral reading
as a surprise. (The entry's contract.kind stays `field` -- the normalizer's artefact; the surprise +
this note flag the concrete kind for the curator rather than my silently rewriting the verdict.)

**Routing (faithful to the source's dynamic-delta contract).** The source returns the EXACT increment
`dr = (R-r)(1-exp(-k*dt/R))` as a dt-scaled delta the Model adds. Plexus integrates a first-order
block as `x += dt*delta`, so -- identical to the `regulate:neural_ode` sibling's self-solved increment
-- I return the mean RATE `delta = dr/dt`; the engine's `dt*` recovers the exact endpoint (dt cancels,
not a second integration). `EMIT="velocity"`, class `INTEGRAND="radius"` (a NON-coordinate first-order
block, so `_resolve_emit` never constrains the cell's spatial order) + instance-`INTEGRAND` routing to
a configurable size block; the delta lands in the `radius` accumulator, summed with any co-writer.
`growth_rate` is read as a per-cell heritable STATE (block or buffer), falling back to a uniform `rate`
param (default 0 -> k=0 -> byte no-op); the sole "target" knob is `max_radius` (guarded > 0). Pure
torch (exp/mul) -> `DIFFERENTIABLE=True`, so gradients flow through r, k, and the optimizable R -- the
whole point of keeping the rate in state.

**Source wins (rule 5):** translated the CODE (von Bertalanffy exact flow, no clamp), not the paper's
prose (`min(R+dR, Rmax)`, constant-rate), as the excavator/normalizer established. Not hard-coded to
the oracle -- the tests assert facts about the ODE, never a fitted number.

Tests (all reference-free): (a) one step reproduces the ODE's KNOWN analytic solution
`r(dt)=R-(R-r0)exp(-k dt/R)` through `radius += dt*delta` to 1e-6; (b) UNCONDITIONAL STABILITY -- even
dt=1e3 never overshoots R while a forward-Euler step with the same dt blows past it (the exact-flow
headline); (c) fixed point at R (dr=0); (d) sign symmetry about R (below grows, above relaxes down);
(e) k=0 exact no-op; (f) 400-step asymptote to R, approached from below, no overshoot; (g) dormant
(occ=0) cells get zero; (h) the end-to-end path through the real engine `_integrate` lands on the
analytic endpoint (the `_delta_blocks` radius routing composes); (i) registration/routing
(kind=lateral, EMIT=velocity, INTEGRAND=radius, MAPS=[], writes=[radius]).

Left for the validator/curator: the differential run against the oracle. jax is deliberately absent
from the Plexus env, and no oracle/smoke scenario is known to exercise SaturatingCellGrowth yet, so
there is no paired trajectory to diff -- `evidence`/`status: validated` untouched. Curator decision:
whether `grow_radius` promotes as a widened second implementation of `cell_grow` (a kind-widening of
the growth contract) or as its own contract.


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

---

## Implementer: operator `mechanosense`

Wrote `src/plexus/operators/candidates/jax_morph_virial_stress.py` and test
`tests/test_jax_morph_virial_stress.py`. Both import/register clean; test passes 7/7.
`status: implemented`.

**Faithful, not fitted.** Modelled the shape on the `adhere:*` torch ports (same
`safe_norm`/`safe_divide`/`_smooth_cutoff`/`_compact_repulsion`) and the `grow_radius`
read-block-or-buffer recipe. Key decisions:

- **Pure sensing = `EMIT=None`, `forward` returns `{}`.** The operator writes the per-cell `stress`
  scalar in place (a schema block if the set declares one, else a lazily-provisioned buffer) and
  moves nothing. Writing a `stress` state block mutates `Level.state`, which the engine's frame-0
  integration-invariant guard (`engine.py:709-723`) flags -- VirialStress is exactly the
  DERIVED-READOUT category that guard exempts, so the class sets `MAY_MUTATE_INTEGRATED_STATE =
  True`. A test asserts pos/vel are byte-identical after the call (it really moves nothing).
- **Pair law as a plug-in (`potential:` selector), reduced -- not a force.** `dU/dr` is taken by
  autodiff of the FULL pair energy (`torch.autograd.grad(U.sum(), r)`, elementwise) so the
  smooth-cutoff switch term rides in it, exactly as `jax.grad(pair_energy)` does. Default `morse`
  (paper mechanics, eps 3.0); also soft_sphere / hertzian / harmonic / lennard_jones, each with its
  own knobs. The reduction (`r_ij . dU/dr` over live non-self j) is the rank-0 SENSED scalar, versus
  `adhere`'s rank-1 `-grad_i U` that MOVES cells -- the whole reason this is `new`, not a widening of
  `adhere`.
- **The three biology-carrying conventions are in and tested:** minus sign (compression-positive;
  the Morse/harmonic adhesive tail correctly reads tension-negative in the smoke run), 1/(2d)
  Irving-Kirkwood + 1/2 bond split, and V_i = the cell's OWN d-ball volume (2r / pi r^2 / 4/3 pi r^3,
  branched on `d = pos.shape[-1]`). `safe_*` keep the r=0 diagonal and a dead cell's V_i=0 finite.
- Differentiable through the coupling: with a per-cell `epsilon_field`, `d stress / d epsilon` is
  finite/correct (the source's "optimizable through the written stress"); inert under the engine's
  `no_grad` rollout.

**Property test (no oracle numbers).** Anchor = compression-positive sign + the analytic
soft_sphere pressure `p = eps r (1 - r/sigma)/(sigma 2 d V)` (derived BY HAND from `dU/dr = -eps(1 -
r/sigma)/sigma`): two overlapping cells read exactly that, strictly > 0. Plus beyond-contact limit
(r>=sigma -> 0), translation/reflection + identical-cell symmetry, size normalization (bigger V ->
smaller pressure at equal overlap), moves-nothing, dead-cell masking, and the no-block buffer path.

**Left for the differ (did NOT establish):** no oracle run yet -- the FORM and calculus-level
properties are verified, but there is no run-vs-reference stress number. One simplification to watch:
I differentiate `dU/dr` against a detached leaf clone of `r`, so the *value* is exact and the
*epsilon*-gradient is preserved, but the position-gradient path through `dU/dr` is dropped (values --
what the differ compares -- are unaffected).


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

**IMPLEMENTER -- `reorient` written, imports/registers, 6 property tests pass.**
Module `src/plexus/operators/candidates/jax_morph_active_brownian_dynamics2_d.py`; test
`tests/test_jax_morph_active_brownian_dynamics2_d.py`. Implemented ONLY the uncovered
rotational-diffusion leg (the other two legs stay aliases of a pair potential and of `glide`).

**Key translation decision -- scalar angle -> unit-vector rotation.** The source carries
`active_heading` as a *scalar angle* and adds `dtheta` to it. In Plexus `heading` is a unit VECTOR
`[N,D]` (read by glide/bounce/sense), so I translated "add `dtheta` to the angle" into "**rotate the
heading vector by `dtheta`**", `h <- R(dtheta) h`. Exactly equivalent, and it buys two properties:
(1) **norm-preserving by construction** -- a planar rotation keeps `|heading|=1`, so I deliberately
do NOT renormalize (that keeps the norm-conservation test meaningful: it separates this correct
rotation from a naive additive `h += noise`); (2) **additive composition survives the change of
representation** -- planar rotations compose additively, `R(a)R(b)=R(a+b)`, so `reorient` in place
followed by another rotational heading writer equals summing the angle increments, reproducing the
source's dynamic-additive-delta accumulation under Plexus's in-place heading-steer idiom.

**Routing.** `EMIT=None`, mutates `heading` in place, returns `{}` -- the sibling of
`polarity_align`/`polarity_flow_align` (the polarity family owns every heading write; the engine
does not integrate heading). `SUPPORTED_DIMS=[2]` (scalar rotation angle -> planar rotation; a 3-D
version needs a rotation axis, and the source itself raises for `n_space_dim != 2`).

**No fitted constants.** `std_r = sqrt(2 * rot_diffusion * dt)` is the source's own scale (NO gamma
-- do not reuse the translational `std_t`); zero-drift (mean-0 increment); `rot_diffusion=0` is a
non-crashing identity limit; noise drawn at full padded capacity then gated by `occ`/`mask` (dead &
masked slots get `dtheta=0`), matching the source's `n_cells`-sized sampling; seeded RNG.

**Tests (all statable without the oracle):** zero-diffusion no-op; norm conservation (unit stays
unit); heading actually rotates; **zero-drift mean + variance = `2 D_r dt`** over 40k cells; variance
**linear in dt** (Brownian scaling); dead/masked slots unchanged. `evidence.oracle_run` stays null --
the differential test vs the oracle is the verifier's job.

---

**DIFFER -- `reorient` VALIDATED. D_C = 0.00929 << threshold 0.05 -> PASS.**

**Metric (pre-registered, in the record and baked into the plexus diff script before any plexus
run).** `D_C = max over t=1..40 of |C_plexus(t) - C_oracle(t)|`, where `C(t) = <e(t).e(0)>` is the
ensemble orientational autocorrelation of the per-cell heading unit vector `e=(cos theta,sin theta)`
over all N=20000 cells (dimensionless cosine in [-1,1]). This is the one observable the
rotational-diffusion leg controls: it decays as `exp(-D_r t)`, setting the walker's persistence
length and its ballistic->diffusive crossover. `t=0` excluded (C(0)=1 identically on both sides).
The comparison is DISTRIBUTIONAL: JAX and torch draw independent rotational-noise streams, so no
pathwise seed-0-vs-seed-0 match exists; the N-cell ensemble average of C(t) is the
realization-independent invariant.

**Threshold 0.05.** Per-frame sampling SD at N=20000 is `~1/sqrt(N) ~ 0.007`; 0.05 is ~7x that
floor (not tripped by RNG), yet TIGHT -- `max_t|exp(-0.1 t)-exp(-0.1(1+d)t)|` crosses 0.05 at a
rate error `d ~ 15%`, so 0.05 rejects any run whose effective `D_r` (hence the `sqrt(2 D_r dt)`
scaling constant -- e.g. a dropped factor of 2, which doubles D_r) is off by more than ~15%.

**Isolation.** Both sides run the SAME free active-Brownian gas: NoForce (zero drift), kT=0 (no
translational noise), self-propulsion v0=0.3, N=20000, 40 macro-steps at dt=1.0, uniform initial
headings -- so the ONLY thing driving the heading is the rotational-diffusion leg. Oracle:
`ActiveBrownianDynamics2D(None, n_space_dim=2, kT=0.0, rot_diffusion=0.1)`. Plexus:
`reorient(rot_diffusion=0.1)` + `glide(move_speed=0.3)` from primitives.

**Result.** D_C = 0.00929 at argmax frame t=27 -- the two independent-RNG ABPs agree to ~1% of the
autocorrelation amplitude. Corroborating: D_r_eff 0.09926 (Plexus) vs 0.10060 (oracle) vs input 0.1
(both within ~1.3%); Var(dtheta) 0.19989 (Plexus) vs 0.19985 (oracle) vs theory 2 D_r dt = 0.2;
mean dtheta -6.3e-4 ~ 0 (zero-drift). C(t) at t=5/10/20/40: Plexus 0.607/0.369/0.136/0.020 vs
oracle 0.604/0.367/0.138/0.013.

**Acted-ledger reconciliation (read BEFORE the metric).** run_spec's structural ledger
(`log/atlas/active_brownian_dynamics2_d/diag.json`) flags `reorient` INERT (calls 40, acted 0,
valid_evidence:false). This is a KNOWN instrument blind-spot, NOT a no-op: run_spec fingerprints
only the engine-integrated `state` block, and `reorient` (EMIT=None, like the whole polarity
family) WRITES the auxiliary `heading` buffer in place and returns `{}` -- a write that ledger
cannot see. The HONEST ledger is the companion diff script's heading tap: `reorient` acted on
**40/40 calls**, Var(dtheta)>0, and C(t) decorrelates 1.0->0.020 (a genuinely inert reorient would
leave C(t)==1 for all t). `glide` (which run_spec CAN see) acted 41/41, moving positions at speed
0.3 -- a real spreading ABP cloud (gyration 0.30->7.48).

**Runs.**
- Oracle: `atlas_jax_morph/_oracle/runs/diff_active_brownian_dynamics2_d/` (reference.npz +
  summary.json); script `_oracle/scripts/active_brownian_dynamics2_d.py`.
- Plexus (run_spec evidence): `log/atlas/active_brownian_dynamics2_d/` (spec_run.yaml, diag.json,
  metrics.json, metrics.npz, strip.png, movie.mp4); spec
  `config/atlas/active_brownian_dynamics2_d.yaml`.
- Plexus differential (heading tap + C(t) metric):
  `_oracle/scripts/active_brownian_dynamics2_d_plexus.py` ->
  `log/atlas/active_brownian_dynamics2_d/diff_plexus_summary.json`.

**Verdict: the `reorient` contract reproduces the reference's rotational-diffusion leg.** status ->
`validated`.


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

## Implementation (implementer)

Wrote `agitate` at `src/plexus/operators/candidates/jax_morph_brownian_dynamics.py` (anti-chamber;
promotion is the curator's call after the differential test). `@register_operator("agitate",
family="motion", set="cell", kind="lateral")`, base `Lateral`, `EMIT="velocity"`, `SUPPORTED_DIMS=[2,3]`.
Imports and registers clean; 8 property tests pass (`tests/test_jax_morph_brownian_dynamics.py`).

**The one decision that matters: how the split dt-scaling survives the engine's integrator.** The
Plexus engine integrates an `EMIT="velocity"` delta as `pos += dt*v` -- it multiplies whatever we
return by `dt`. But the thermal DISPLACEMENT is a Wiener increment `dx = sqrt(2 kT dt/gamma)*xi`,
which scales as `sqrt(dt)`, not `dt`. To land a `sqrt(dt)` displacement THROUGH a `dt`-multiplying
integrator, `agitate` emits a velocity that scales as `1/sqrt(dt)`:

    v = sqrt(2 kT / (gamma*dt)) * xi     ->     dt*v = sqrt(2 kT dt/gamma) * xi   (correct)

This is the faithful torch translation of the source's `std = sqrt(2 kT dt/gamma)` applied as a
position delta. It also makes the composite reproduce Euler-Maruyama exactly: a drift operator
emits velocity `F/gamma`, the engine sums it with this thermal velocity, and one `pos += dt*v`
integration gives `dt*(F/gamma) + sqrt(2 kT dt/gamma)*xi`. Getting this wrong (a `dt`-independent
noise velocity, or scaling the noise by `dt` like the drift) makes the displacement variance scale
as `dt^2` instead of `dt` -- the wrong diffusion constant. This is the headline test.

**Decomposition honored: `agitate` owns ONLY the thermal leg.** Per the normalized contract, the
deterministic drift `F/gamma` is a SEPARATE pluggable pair-potential operator (soft_sphere /
hertzian / the Morse family / attraction_repulsion -- all already `velocity`-emitting, gamma folded
into their coefficients, confirmed in `attraction_repulsion.py:30`). `agitate` reads no potential
and adds no drift; it is the free Brownian gas (the source's `potential=None`) and composes with a
drift operator in the schedule. The whole StochasticStep trace/replay/logp machinery is engine
plumbing and is out of scope here. This is the sibling shape of `reorient`
(`jax_morph_active_brownian_dynamics2_d.py`), which likewise kept only the one leg
(ActiveBrownianDynamics2D's rotational diffusion) the frozen language lacked.

**Tests -- all reference-free (a limit, a scaling law, a symmetry; no oracle numbers):**
- `kT=0` limit -> emitted velocity exactly zero (the deterministic gradient-descent regime);
- Wiener `sqrt(dt)` scaling (headline) -- same-seed noise, quartering `dt` halves the displacement
  (velocity doubles), exact to 1e-6;
- FDT amplitude -- same-seed, displacement scales as `sqrt(kT)` and `1/sqrt(gamma)`, exact;
- Einstein diffusion + isotropy -- 40k cells, per-dim displacement variance = `2 kT dt/gamma`
  within 5%, zero mean, equal across dims (2-D and a separate 3-D case);
- alive/mask masking -- dead (`occ=0`) or masked-out cells draw a zero kick;
- pos-invariance -- forward returns a delta and never mutates `pos` (the integration guard);
- `n_space_dim` mismatch raises (the source's `_check_dim` surprise, kept as an optional assert).

**Faithful extras.** `kT` (default 0.1) and `gamma` (default 1.0) match the source; `kT<0` /
`gamma<=0` raise; `dt<=0` returns a zero kick. Noise uses `H.rng` (seeded generator) sized to the
full padded capacity and masked by `occ` -- the reference's alive-masking.

**NOT established (for the verifier / curator):**
- Oracle NOT run. `evidence.*` untouched -- no numerical check that a Plexus `agitate` (+ a Morse
  drift operator) trajectory matches `oracle.py` on a Brownian/relaxation script. The differential
  test is the curator's next step; the tests above fix the operator's contract, not oracle
  agreement, by design (a fitted constant would pass the differ and teach us nothing).
- RNG stream differs from the source (torch generator vs jax PRNG key), so a pathwise trajectory
  will NOT match sample-by-sample; the differ must compare distributional/statistical quantities
  (MSD growth, diffusion constant), or the `kT=0` deterministic case where only the drift moves
  cells (that one CAN match a jax-md gradient-descent relaxation numerically).
- The `sqrt(dt)`-velocity convention assumes the engine multiplies a `velocity` emit by `dt` once.
  Verified against the documented integration contract (base.py:524) and the sibling velocity
  operators (glide/sediment/attraction_repulsion), but not exercised end-to-end through
  `engine.run` here.

---

## Differential test (differ)

**VALIDATED.** `agitate` reproduces the reference free-diffusion constant.

Metric (pre-registered before running): relative error in the free-diffusion constant D of the
PURE thermal bath, MAX over three macro-step sizes --
`metric = max_dt |D_plexus(dt) - D_ref(dt)| / D_ref(dt)`, with
`D(dt) = slope(Rg^2 vs t) / (2*n_dim)`, `Rg^2(t) = mean_alive |r_i - c|^2` fit over every frame of
a pure-bath rollout (`agitate` alone, no drift = the reference `BrownianDynamics(potential=None)`
free Brownian gas). Measured at `dt in {1.0, 0.5, 0.25}` at fixed total time T~40, because D is
dt-INVARIANT only if the noise carries the Wiener sqrt(dt) displacement scaling -- the ONE feature
separating `agitate` from the frozen operators' bolt-on `noise*randn` jitter.

Threshold (pre-registered): **0.03** = ~3x the reference per-side finite-sample error
(1/sqrt(2N)=0.0071 at N=20000) and ~1.7x the oracle's own cross-dt D_hat scatter (0.017). Not
looser: the dt-scaled-noise bug gives D=dt*kT/gamma -> 75% error (negative control, rejected).

Result: **value = 0.010859** (max, at dt=0.5) <= 0.03 -> **PASS**.
- D_ref [slope]    = {1.0: 0.10009, 0.5: 0.09944, 0.25: 0.10025}
- D_plexus [slope] = {1.0: 0.10030, 0.5: 0.10052, 0.25: 0.10004}   (all fits r^2 > 0.9999)
- per-dt rel err   = {1.0: 0.0021, 0.5: 0.0109, 0.25: 0.0021}; both match theory D=kT/gamma=0.1.
- Plexus D dt-invariant to 0.5% -> the sqrt(dt) scaling survives `pos += dt*v` (`agitate` emits
  v = sqrt(2kT/(gamma*dt))*xi).
- Negative control: dt-scaled bug D=0.0250 at dt=0.25, rel_err 0.75, REJECTED -- the metric bites.
- Acted ledger: agitate calls 41, acted 41, moved 2.16, inert_operators [] -> valid_evidence true;
  gyration 0.63 -> 4.05 (free spreading, no wall clamp). Same IC as oracle (all cells at one point;
  Rg is translation-invariant, so the (20,20) offset is not a mismatch).

Paths:
- oracle:  atlas_jax_morph/_oracle/runs/diff_brownian_dynamics/ (script _oracle/scripts/diff_brownian_dynamics.py)
- plexus:  log/atlas/brownian_dynamics/{diag,metrics,diff}.json (specs config/atlas/brownian_dynamics{,_dt05,_dt025}.yaml)

Verdict: `new` -> `agitate` (motion family, cell set), **status: validated**.


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


## Death -- implemented

Operator `apoptose` at `src/plexus/operators/candidates/jax_morph_death.py`; test at
`tests/test_jax_morph_death.py` (6 properties, all pass). Written as `cell_divide`'s literal
inverse and modelled on it line-for-line: same `Structural` base, `EMIT=None`, the same
`getattr(lvl, "<rate>", None)`-else-scalar-`rate` fallback (`death_rate` buffer here, `div_rate`
there), the same `getattr(H, "rng", None)` draw. The whole effect is `lvl.occ[die] = 0.0` -- occ
IS the Plexus analog of the source's `alive` mask, and retiring it (rather than parking/zeroing
mass) is the faithful minimal translation, because jax-morph cells are single particles with no
MPM child to retire. Reuse of the freed slot is deferred to a LATER macro-step exactly as the
source intends: `cell_divide` (which allocates `occ==0` slots) runs earlier in the pipeline under
divide-then-die, so a slot freed this step survives for lineage reconstruction.

Deliberate translations of the source's subtleties:
- Hazard `p = -expm1(-clip(rate,0)*dt)` copied verbatim (the `-expm1` form, and the `clamp(min=0)`
  guard so a negative controller output gives p=0, not a NaN score).
- `die` is re-AND'd with the LIVE mask via the eligibility set (`elig = live & at-mask`), so an
  already-dormant slot can never be marked "newly dead" -- the source's `(died>0.5) & alive` guard.
- `death` is a lazily-registered per-node FLOAT buffer, zeroed then set each step (OVERWRITTEN, not
  accumulated), dtype float so it is summable/differentiable while occ stays the boolean liveness.
- The `at:` mask gates eligibility (the source's `die_eligible`); a test confirms masked-out live
  cells survive.

NOT modelled (and why): the trace/`logp` score-function (REINFORCE) layer. Plexus's engine runs the
forward EFFECT only (the source's `replay`), so `died`/`die_eligible` traces and `Death.logp` have
no engine counterpart -- same scope as `cell_divide`, which realises the forward proliferation event
without its scoring term. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation): rate-0 no-op; negative-rate
clip -> no death; huge-rate -> certain death of every live cell (seed-independent, tests the hazard
form + the retire); dormant slots never die/revive and live-count is monotone non-increasing
(apoptose is strictly a remover); the float `death` record equals the exact occ 1->0 flip mask; and
the eligibility mask restricts who may die. No oracle run -- the paper ships no death config and jax
is absent from this env by design -- so `evidence.oracle_run` stays null for the differ/curator.

Name note: `apoptose`, not `death`, so no clash with the pre-existing efflux-boundary `death`
operator (candidates/death.py, kind `lateral`) -- a geometric exit-line sink, an unrelated
mechanism. Candidates are not auto-imported, so nothing registers until the differ/tests import it.


## Death -- validated (DIFFER)

**PASS. `status: validated`.** The implemented `apoptose` reproduces the reference `Death` hazard.

**Why not a bit-for-bit trajectory diff.** Death is a discrete Bernoulli event and the JAX
(`jax.random.bernoulli`) and torch (`torch.rand`) streams are different RNGs, so seed-0-vs-seed-0
matches nothing stochastic. The paper also ships no death config, and there is no shared IC in the
smoke/anchor run (which has no death). So the differential is run on a purpose-built PURE-DEATH
rollout with a matched IC, and measures the one thing a hazard is defined by -- the pooled
per-macro-step death probability -- which is realization- and record-convention-independent.

**Metric (pre-registered).** `diff_metric = |p_hat_plexus - p_hat_oracle|`, pooled per-macro-step
death hazard, with `p_hat = (n_active[0] - n_active[-1]) / sum(n_active[:-1])` over the 41 recorded
frames (no births => every live-count decrement is one committed death => MLE Bernoulli hazard over
every eligible live-cell-step). `threshold = 3*SE_pooled`, `SE_pooled = sqrt(SE_o^2 + SE_p^2)`,
`SE = sqrt(p_hat(1-p_hat)/eligible)`. Both decided before reading the result; the band is justified
not just as a sampling interval but by discrimination -- it is tighter than the gap between the exact
hazard `1-exp(-lambda*dt)=0.048771` and the linear-approx bug `lambda*dt=0.05` (gap 0.00123), so an
operator that wrote `p=rate*dt` instead of the source's `-expm1(-rate*dt)` would FAIL this test.

**Matched IC.** Oracle `Model([Death])` and Plexus `config/atlas/death.yaml` both start N0=50000
live cells at uniform `death_rate = lambda = 0.05`, `dt=1.0`, 40 macro-steps, death the only
operator. (Death reads no position, so scattered positions in the Plexus box are irrelevant to the
survival metric.) One caveat handled explicitly: Plexus records AFTER applying each op, so its first
recorded frame is already post-step-1 (47506, not 50000; the acted ledger shows apoptose acted 41x
over 41 frames). A frame-index-aligned survival diff would be off-by-one and wrong; the pooled
hazard is stationary and unbiased on any consecutive-frame window, so it sidesteps this cleanly.

**Numbers.**

| side | p_hat | SE | deaths | eligible cell-steps | vs exact 0.048771 |
|------|-------|-----|--------|---------------------|-------------------|
| oracle (jax-morph Death) | 0.048928 | 2.29e-4 | 43236 | 883662 | 0.69 SE |
| Plexus (apoptose)        | 0.048985 | 2.35e-4 | 41169 | 840445 | 0.91 SE |

`diff = 5.66e-05 = 0.17 * SE_pooled  <=  threshold 0.000986`  ->  **PASS**. Both sides also agree
with the exact hazard within 1 SE. Acted ledger: `apoptose` calls 41 / acted 41 / inert []
=> `valid_evidence: true` (checked before any metric).

**Paths.** oracle run `atlas_jax_morph/_oracle/runs/diff_death/` (script
`_oracle/scripts/death.py`); Plexus evidence `log/atlas/death/` (spec `config/atlas/death.yaml`);
differential computed by `_oracle/scripts/_analyze_death.py`.

**Scope (unchanged from implemented).** This validates the forward EFFECT (the source's `replay`):
the exact hazard form, the `>=0` clip, the retire (occ 1->0), and monotone non-increasing counts.
It does NOT exercise the trace/`logp` score-function (REINFORCE) layer -- Plexus's engine has no
counterpart, same as `cell_divide`. The `death` lineage float / divide-then-die ordering are covered
by the reference-free unit tests, not by this population-level differential.


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



## Division -- implemented

Operator registered as `cell_divide` with `implementation="volume_conserving"` at
`src/plexus/operators/candidates/jax_morph_division.py`; test at
`tests/test_jax_morph_division.py` (10 properties, all pass). This is the faithful realization of
the `refinement` / `implementation_of: cell_divide` verdict: the registry keys operators by
`(name, implementation)` and only enforces that co-implementations share the contract `kind`
(structural == structural), so importing the candidate ADDS `volume_conserving` alongside the
promoted isotropic `default` on the SAME `cell_divide` contract -- the widened read/write set lives
declaratively in the atlas entry, the registry does not re-validate it. Default stays the promoted
mass-doubling impl, so the widening is additive for current callers exactly as the entry claims; the
candidate is not auto-imported, so nothing registers until the differ/tests pull it in. Same
multi-implementation pattern as `diffuse` (finite_difference/spectral) and `regulate`
(connectionist/mwc/neural_ode).

State representation (the reason this needed care). Plexus has NO standard `radius` state block --
`radius` is only a spawn-time scalar in `engine.py`, never carried per-cell -- so `radius`,
`division_rate`, and `division_axis` are modelled as per-cell BUFFERS the operator reads/lazily
provisions, mirroring `apoptose`'s `death_rate` convention. `born` is a float buffer, `mother` a long
buffer with the -1 founder sentinel, and `division_overflow` a 0-dim scalar buffer.

Deliberate translations of the source's subtleties:
- Volume factor `m = 2^(-1/d)` reads the live world dim `H.dim` (not a hardcoded 1/2), so both
  daughters take `r*m` and conserve the mother's d-volume (~0.707 in 2D, ~0.794 in 3D).
- The offset uses the NEW radius `(r*m)*dir`; mother moves to `x+offset` and daughter to `x-offset`,
  so the pair is centred on the mother's pre-division position and sits exactly touching (centre gap
  `2*r*m` = sum of radii). I capture `x_old`/`r_old` BEFORE any write so the two placements can't read
  each other back.
- Oriented direction `normalize(orientation_snr*a_hat + xi/sqrt(d))` with `a_hat = axis/(||axis||+1e-12)`;
  a zero axis or `orientation_snr=0` collapses to pure isotropic via the same 1e-12 guard (a test
  confirms the isotropic fallback still conserves volume and just-touches). With no `division_axis`
  buffer at all the direction is pure `xi`.
- Capacity is a hard wall: I draw the movers FIRST, then allocate `cap = min(movers, free)` and add
  `movers - cap` to `division_overflow` -- so the surplus is counted even when the buffer is
  completely full (`cap == 0`), matching the source's `sum(divide) - sum(committed)`. `born`/`mother`
  reset to their defaults every macro-step (a per-step lineage record); `division_overflow` is GLOBAL
  and accumulates (a test checks 4 dropped then 4+8=12 over two steps, deterministic at a huge rate).

One chosen divergence from the source, noted for the curator: jax-morph resets a daughter's
NON-heritable cell fields to their spec default and inherits only heritable ones; I inherit EVERY
per-cell buffer from the mother (like the promoted `cell_divide`'s spawn), then reset `born`/`mother`
explicitly. The heritable drivers (`division_rate`, `division_axis`, `celltype`) inherit correctly
either way; the only difference is that a recycled dead slot's stale non-heritable buffers are
overwritten with the parent's value rather than a default -- arguably safer, and consistent with the
sibling `cell_divide`. Plexus's own `Level.lineage`/`birth` buffers already record parent-slot
provenance, so `mother` is partly redundant with the container, but I write it explicitly because
`reconstruct_lineage` reads that exact field.

NOT modelled (same scope exclusion as `cell_divide`/`apoptose`): the straight-through / pathwise
differentiability and the `logp` score-function term. Plexus's engine runs the forward EFFECT only
(the source's `replay`); the `divided`/`divide_eligible`/`division_dir` traces and `Division.logp`
have no engine counterpart. If a scoring/inverse-design driver lands, that layer is the follow-up.

Tests are reference-free by construction (limits, sign, conservation, symmetry): rate-0 and
negative-rate (clip) no-ops; d-volume conservation `r_m^d + r_d^d == r_old^d` in 2D and 3D; the
just-touching centre-distance = sum-of-radii geometry; the symmetric split centred on the mother;
lineage (`born=1`, `mother=parent`, defaults elsewhere) with live-count growth; the overflow cap +
GLOBAL accumulation; the large-`orientation_snr` limit aligning the split with the axis; the
isotropic fallback; and the `at:` mask gating who divides. No oracle run -- jax is deliberately
absent from this env -- so `evidence.oracle_run` stays null for the differ/curator.


## Division -- validated (differential test)

**Status `validated`. The flagged 124-vs-82 open discrepancy is RNG realization noise, not a wrong
contract.** A single-seed count cannot separate a wrong hazard from noise: jax and torch PRNG streams
differ, so seed-0-vs-seed-0 matches only the deterministic initial POSITIONS, never the stochastic
division draws. So the differ replaces the one-sample count with a realization-INDEPENDENT invariant
-- the **pooled per-macro-step Bernoulli hazard** `p_hat = committed_divisions / eligible_cell_steps`
-- measured identically on both engines over M=48 seeds x 40 macro-steps at `division_rate=0.08`,
`dt=1.0` on the 4 anchor founders (no death, `division_overflow==0` on every seed with `CAP=512`, so
every live-count increment is one committed division).

- **metric** `|p_hat_plexus - p_hat_oracle|` (dimensionless probability per alive cell per macro-step).
- **threshold** `3*SE_pooled = 0.005225` (formula pre-registered; `SE_pooled = sqrt(SE_o^2+SE_p^2)`,
  each `SE = sqrt(p(1-p)/eligible)`). Resolves any hazard divergence >~6.7% rel -- a doubled rate is
  40 SE out -- while not flagging Monte-Carlo scatter between two independent PRNG stacks as
  disagreement. Honest limit: at M=48 it does NOT resolve the O((lambda*dt)^2/2)~0.003 exact-vs-linear
  hazard gap; that is excluded independently (both engines carry the source's `-expm1` law and both
  match the analytic hazard `1-exp(-0.08)=0.076884` to under one SE).
- **result** `|diff| = 0.00109 = 0.63*SE_pooled <= 0.005225` -> **PASS**.
  - `p_hat_oracle = 0.077787` (SE 1.22e-3; 3734 / 48003), vs theory +0.74 SE.
  - `p_hat_plexus = 0.076696` (SE 1.24e-3; 3528 / 46000), vs theory -0.15 SE.
  - Both engines agree with theory AND with each other (triangulation).
- **discrepancy resolved** The oracle final-count distribution (mean 81.8, SD 43.6, min 18, max 234
  over 48 seeds -- a Galton-Watson count whose SD ~ its mean) contains BOTH smoke's 82 (z=+0.005) and
  the anchor/Plexus 124 (z=+0.97) as ordinary draws. Plexus seed 0 gives exactly 124 (the flagged
  value); oracle seed 0 gives 82 (smoke) -- same hazard, different PRNG stream. Volume-conservation /
  placement (not exercised by the count-based hazard) corroborated by the anchor gyration agreeing to
  ~9% (Plexus 3.52 vs reference 3.23).

Runs: oracle `atlas_jax_morph/_oracle/runs/diff_division/` (script `_oracle/scripts/division.py`);
Plexus engine evidence `log/atlas/division/` (`diag.json` acted ledger: `cell_divide` 41 calls /
acted 33, `grow_radius` 41/41, `relax` 41/33, `seed_state` 1/1, `inert_operators []` ->
`valid_evidence true`) plus the pooled-hazard summary `log/atlas/division/diff_plexus_summary.json`
(script `_oracle/scripts/division_plexus.py`, which also proves the division-only reduction is
count-neutral at seed 0: full and division-only both give 124, since only `cell_divide` draws
`H.rng` in this spec). Spec `config/atlas/division.yaml`.


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

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_free_screened_diffusion.py` —
`MorphogenFreeSpace(Exchange)`, registered as the `morphogen` contract with
`implementation="free_space_greens_function"` (kind=exchange / family=fields / set=cell, matching
the normalized contract). This is the `diffuse` `finite_difference`/`spectral` pattern applied to
`morphogen`: the paper's graph-Laplacian inverse `c = (K I - D L)^{-1} S` can later register
`implementation="graph_laplacian"` on the SAME contract — two numerical methods, one biological
operator, which is the convergence the ledger is meant to record. `get_contract("morphogen")` now
resolves; `implementations = {"free_space_greens_function"}`.

**The load-bearing decomposition choice — an OVERWRITE, not a delta.** Unlike the gene-network
siblings (`regulate:*` return `dg/dt` for the engine to integrate), this is a QUASISTATIC CONSTRAINT
SOLVE: `dt` is meaningless, the field is the `t=infinity` equilibrium, so the operator OVERWRITES the
`chemical` block each step rather than incrementing it. It is therefore a *derived readout* in exactly
the `aggregate` (centroid) sense: `EMIT=None`, `MAY_MUTATE_INTEGRATED_STATE=True`, mutate state via
clone-and-assign of the `chemical` columns, return `{}`. `MAY_MUTATE_...=True` is required because the
engine's frame-0 integration guard clones the WHOLE `state` tensor and would otherwise flag the
in-place block write (it is the derived-readout exemption, not a force). Verified `pos` is invariant
across a forward (a test), so the exemption is honest — it only ever writes `chemical`.

**Faithful details carried over (diffusion.py:92-264):**
- Dimension-selected kernel by `pos.shape[1]` — 1-D segment `exp(-kappa(r_eff-a))/(2 D kappa)`, 2-D
  disk `K0(kappa r_eff)/(2 pi D a kappa K1(kappa a))`, 3-D sphere
  `exp(-kappa(r_eff-a))/(4 pi D r_eff (1+kappa a))`. The reference's static `n_space_dim` assert is
  preserved as an OPTIONAL param that raises on mismatch (surprise kept reproducible).
- `r_eff = max(r_ij, a_j)` surface clamp → the `i==j` diagonal contributes the on-surface value (SELF
  field included, tested). Source radius `a` is the EMITTER's, broadcast over receiver rows.
- Alive-masking applied TWICE and asymmetrically: sources masked over columns j (`* occ[None,:]`),
  receivers over rows i (`* occ[:,None]`) — tested both directions with a dead big-source slot.
- The three finiteness guards: `a <- max(a,1e-12)`, `kappa <- max(kappa,1e-12)` for 1-D/2-D only, 3-D
  left exact (bounded at kappa=0). The low-dimension screening requirement (`degradation>0` in 1-D/2-D)
  is raised at forward (where the spatial dim is known), matching the reference's constructor check.
- `safe_norm` ported verbatim (value+grad zero at the zero vector — the `where` trick, not a bare
  sqrt), so the diagonal r=0 does not poison gradients. Minimum-image applied if the world is periodic,
  with the documented caveat that free-space kernel + periodic box is a modeling error the user owns.
- JAX `vmap(in_axes=(0,0,1))` over species → a Python loop over the small static species axis;
  `diffusion`/`degradation` broadcast to `(n_species,)` (scalar or per-species). Multi-species
  independence tested.

**Bessel port (the one non-mechanical carry-over).** `_k0`/`_k1` are the reference's Abramowitz &
Stegun 9.8.5–9.8.8 rational/series approximations, ported to torch on `torch.special.i0`/`i1`. I did
NOT use torch's built-in `torch.special.modified_bessel_k0/k1` because those carry no autograd
backward (they would break `DIFFERENTIABLE=True`, the whole point of a jax-morph translation). I
cross-checked my port against torch's builtin K0/K1 over x in [1e-3, 20]: max relative error ~1e-7
(the A&S series precision) — so the disk kernel is faithful, not eyeballed. This also closes the
normalizer's open item "did not verify `_k0`/`_k1` against a reference over the full range."

**One deliberate robustness add (flagged, not hidden).** The reference always has `state.radius`; a
Plexus cell set may not carry a `radius` block. `_radii` reads a per-cell `radius` block/buffer if
present, else falls back to a UNIFORM default (the engine's `spawn radius` default 0.02). For the
oracle differential the source set carries a real per-cell radius, so this is inert there; it only
keeps the operator runnable on a radius-less set. Also: the contract's `READS` lists
`pos`/`radius`/`alive` explicitly — the reference `state_reads()` under-declares them (declares only
`secretion_rate`), the surprise the normalizer flagged.

**Test:** `tests/test_jax_morph_free_screened_diffusion.py` (7 pass). Headline property (reference-free):
SUPERPOSITION — the map `S -> c` is LINEAR (`c(3S)=3c(S)`, `c(S1+S2)=c(S1)+c(S2)`), the defining
property of a Green's-function steady-state solve, tested in 2-D so it exercises the Bessel path. Plus:
non-negative sources → non-negative field (kernel positivity); stronger `K` shortens the range (a
distant cell reads less); the self/diagonal field is included; a dead cell neither emits nor carries;
`pos` invariance under the solve; per-species independence. No oracle numbers hard-coded.

**FLAG for the curator's differential run (not a translation bug).** (1) Use a FREE (non-periodic)
world — the kernel is open-boundary; a periodic oracle would need the paper's method, not this. (2)
This is an OVERWRITE and quasistatic, so it must run BEFORE any step that reads `chemical` in the same
macro-step (the reference's `quasistatic -> dynamic -> discrete` phase order); schedule it first. (3)
Match `n_space_dim`/`diffusion`/`degradation` and per-cell `radius` to the oracle config; the finite
radius `a` has no paper counterpart, so agreement is a code-vs-code check, not code-vs-paper.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.

---

## Differential run (DIFFER)

**Result: `status: validated`. value = 2.51e-07 (relative) < threshold 1.0e-3 -> PASS.**

`morphogen` is a QUASISTATIC pure state->field map: `__call__` ignores dt/key and OVERWRITES the
`chemical` block with the t=infinity solution, moving no cell. So — unlike the ODEController siblings,
where a trajectory diff conflates the vector field with the integrator — there is NO integrator and NO
trajectory to score. The test is the analog of the gene-network `metric_A`: evaluate the reference
`FreeScreenedDiffusion.__call__` (jax) and the Plexus `MorphogenFreeSpace.forward` (torch) on ONE
identical fixed state and diff the written field directly. float32 both sides (the reference is NOT run
in x64), so the number is pure algorithm agreement, not a dtype gap.

**Metric** (fixed BEFORE the run in `diff_free_screened_diffusion.py`): `value = max` over all configs,
all cells (live AND dead), all species of `|c_plx - c_ref| / max(|c_ref|, 1e-6)` — relative where the
field is resolvable, floored-absolute where it is ~0 (a near-zero true value cannot manufacture a
spurious infinite ratio). Gated on finite + every DEAD slot exactly 0.0 on the Plexus side. Four fixed
configs, each a distinct kernel regime the CODE has and the paper's graph-Laplacian solve does not:
- **C_anchor** (2-D disk): the anchor's 4 founders, uniform radius 0.5, unit secretion, D=K=1 — the
  exact geometry the ENGINE run seeds (translation-invariant, so identical pairwise distances).
- **C2** (2-D disk, adversarial): 12 slots / 9 live, per-cell varied radius, an OVERLAPPING pair
  (r<a -> r_eff=max(r,a) self/near clamp), 2 species with per-species (D,K), secretion incl. zeros and
  a **big S=99 on a DEAD slot** (a source-mask bug would be loud).
- **C3** (3-D sphere, adversarial): species 0 **UNSCREENED (K=0, kappa=0, the deliberately-unfloored
  exact branch)** + species 1 screened, overlapping pair, dead slots.
- **C1** (1-D segment, adversarial): 6 live, D=1, K=0.7.

**Measured** (`_oracle/runs/diff_free_screened_diffusion/diff.json`): value 2.51e-07, max_abs 3.58e-07
— float32 rounding floor, ~4 orders below threshold. Per config: C_anchor 6.7e-08, C2 2.51e-07 (worst;
argmax at the FAR cell where the field is LARGEST 1.426, i.e. float32 accumulation, not a near-FLOOR
artefact), C3 1.16e-07, C1 9.7e-08. All finite; every dead slot exactly 0.0 both sides (the S=99 dead
source emits nothing — the twice-applied alive mask reproduces). The ported Abramowitz-Stegun Bessel
K0/K1 (torch.special.i0/i1 vs jax jsp.i0/i1), the self/near clamp, the per-species vmap-vs-loop, and the
K=0 unfloored 3-D branch all reproduce to rounding.

**Threshold** `1.0e-3` (relative), fixed before the run: float32 has ~7 sig-digits (eps ~1.2e-7); a
dense all-pairs superposition plus (in 2-D) a truncated Bessel series across two INDEPENDENT numeric
stacks accumulates to an expected faithful-port floor ~1e-5..1e-4, so 1e-3 sits ~1 order above that yet
~3 orders BELOW the O(1) divergence any STRUCTURAL error would produce (dropped self-term, wrong mask,
transposed source radius, mis-ported Bessel coefficient, or the paper's graph-Laplacian boundary
condition). Passing certifies the two methods compute the SAME free-space field; it CANNOT pass a
graph-Laplacian re-implementation. Tighter 1e-6 is below the two-library float32 floor and unmeetable;
looser 1e-1 stops distinguishing a faithful port from a ~10% kernel-form error.

**The acted ledger checked first** (`log/atlas/free_screened_diffusion/diag.json`): `morphogen` acted
1/5 (structural write of `chemical` from zero at frame 0; recomputed identically after, so no further
state change — acted>=1, NOT inert), `seed_state` 1/1; `inert_operators: []`, `valid_evidence: true`.
run_spec's position metrics (gyration, nn_distance) are field-INVARIANT here by construction (the solve
moves no cell), so the field itself is scored by the isolated differ, not by this run — this run's job
is to prove the operator SCHEDULES and ACTS in the real engine on the matched IC.

**Runs**
- Oracle (jax venv): `_oracle/runs/diff_free_screened_diffusion/` (reference.npz + summary.json;
  script `_oracle/scripts/free_screened_diffusion.py`). dt/key-invariance asserted (quasistatic).
- Score (torch): `diff_free_screened_diffusion.py` -> `.../diff.json`.
- Engine (torch): spec `config/atlas/free_screened_diffusion.yaml`; evidence
  `log/atlas/free_screened_diffusion/` (diag.json, metrics.json/.npz, spec_run.yaml, strip.png).

**Verdict stands: `new`, `implementation_of: morphogen`.** The reproduction confirms the FREE-SPACE
Green's-function method is a faithful implementation of the `morphogen` contract to float32 precision
across 1-D/2-D/3-D. It does NOT (and cannot) certify the paper's graph-Laplacian sibling — that is a
different numerical method with different boundaries; the code-vs-paper divergence remains the recorded
source-wins contradiction, tested here only in that the free-space form reproduces exactly.


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

# gene_network_connectionist (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_connectionist.py` —
`RegulateConnectionist(Exchange)`, registered `@register_operator("regulate", family="fields",
set="cell", kind="exchange", implementation="connectionist")`. Torch, not JAX. Test:
`tests/test_jax_morph_gene_network_connectionist.py` (5 property tests, all green). `status ->
implemented`; evidence left null for the differ.

**The one real design decision — SELF-SOLVE vs emit-a-rate.** The source is an `ODEController`: it
integrates the ODE over the whole macro-step with adaptive Dopri5 and its DYNAMIC step returns the
*sparse endpoint delta* `g(dt)-g(0)`, not an instantaneous rate. Plexus's engine integrates a
first-order block with one Euler step per tick (`g += dt*delta`). I chose to reproduce the reference
ENDPOINT: integrate internally over `[0,dt]` with fixed-step RK4 (`substeps`, default 8) and return
the *mean rate* `(g(dt)-g(0))/dt`, so the engine's `dt*` recovers `g(dt)` exactly (the dt cancels —
not a second integration). This is the operator's defining behavior per its own equations/surprises
("emits the SPARSE DELTA … integrated over one macro-step by Dopri5"), and it maximizes fidelity for
the coming differential test. Fixed-step RK4 vs adaptive Dopri5 is a numerics choice inside the one
contract; `substeps` tightens it. NB the `mwc` sibling made the OPPOSITE call (emit instantaneous
`dg/dt`, let the engine Euler-step, invoking "integration is the engine's concern") — a genuine
unresolved tension across the four `regulate` files that the curator should reconcile when promoting.

**SOURCE WINS, faithfully ported.** (1) Sensed input enters INSIDE the sigmoid via a trainable
`W_in @ u` (code), NOT as an additive `+ I_i` outside it (paper) — implemented the code. (2) `sigma`
is the ALGEBRAIC `0.5 + 0.5 x/sqrt(1+x^2)` (`_rescaled_sigmoid`), not logistic — ported including the
overflow guard (clip to finite range, rescale by max(1,|x|)). At extreme drives it saturates to
*exactly* 0/1 (float32 underflow of `scaled_one^2`); that is correct, so my sigmoid test asserts
strict `(0,1)` only for moderate drives and mere finiteness at 1e30. (3) Defaults are NOT inert:
zeros for W/W_in/b but `gamma=0.1`, so `dg/dt = 0.5 - 0.1 g` drives every gene to `g*=5.0`. (4)
`gamma` stored verbatim, NOT shape-checked (the one param the source skips through `_resolve_param`);
matrices materialize lazily on first forward and ARE shape-checked. `u` is a frozen quasistatic
snapshot for the whole solve.

**Routing.** `INTEGRAND="gene"` (instance-set to the configured block; class stays `"gene"` so
`_resolve_emit` sees a non-`pos` integrand and does not constrain the coordinate order), `EMIT="velocity"`,
`MAPS=[]`. Sensed drivers `u` are read as per-cell STATE BLOCKS (the reference `_pack`s per-cell input
specs — they are not grid fields), named by `inputs:` (a name or list, concatenated). Dormant cells
(`occ=0`) get a zero delta.

**Property tests (all reference-free — stated from the operator's own definition):** (1) production
`= dg/dt + gamma*g` is strictly in (0,1) for random large drives (algebraic-sigmoid range); (2)
`sigma(0)=0.5`, monotone, overflow-finite; (3) zero-interaction circuit is not inert and vanishes at
`g*=5.0`; (4) end-to-end through `build`+`_integrate`: the engine result equals the internal
self-solve and CONTRACTS toward `g*=5.0` on the original side; (5) dormant cells don't evolve.

**Heads-up for the curator (registration overlap, not a bug).** The `odecontroller` entry (order 4,
the abstract base) also landed a `regulate`/`connectionist` file (`jax_morph_odecontroller.py`, an
adaptive-Dopri5 self-solve) as its concrete representative. Mine is the CANONICAL connectionist
entry (order 1). Candidates are never bulk-imported (the anti-chamber is full of intentional name
clashes), so each imports/tests alone — no runtime collision — but the two `regulate`/`connectionist`
modules are the same operator and should collapse to one on promotion. That collapse (four
ODEController entries → one `regulate` contract with implementations connectionist/mwc/neural_ode) IS
the convergence result the ledger is meant to record.

**Did NOT run** the oracle (differential test is the differ's job; evidence stays null).

---

# gene_network_connectionist (DIFFER)

**Verdict: PASS. `status -> validated`.** The Plexus `regulate:connectionist` operator reproduces
the reference `GeneNetworkConnectionist` gene trajectory to **float32 rounding** — max abs
deviation **6.68e-06** over the whole 21-frame x 6-cell x 5-gene run, against a pre-registered
threshold of **5e-3**.

**Why the test is a DIRECT trajectory comparison, not integrator-matched (the opposite of `mwc`).**
The `mwc` sibling emits an instantaneous `dg/dt` and lets the engine take ONE crude Euler step over
`dt=1`, so its differ HAD to integrate the reference under a matched Euler to avoid measuring the
integrator (its Dopri-vs-Euler one-step gap was 0.48). The connectionist operator instead
SELF-SOLVES each macro-step with fixed-step RK4 (`substeps=64`) and returns `(g(dt)-g0)/dt`, so the
engine's `g += dt*delta` recovers an accurate `g(dt)`. That self-solve is a high-accuracy
integration of the SAME ODE the reference integrates with adaptive Dopri5, so I compare the engine
trajectory DIRECTLY against the reference's own Dopri5 output — the honest end-to-end question, "does
our operator reproduce what the reference actually computes?", with no integrator sleight of hand.

**Metric / threshold (both written into the record BEFORE running).**
`max over {frame 0..20, cell 0..5, gene 0..4} |G_eng - G_ref|`, float32 gene-concentration units.
`G_ref` = the reference Dopri5 trajectory (`reference.npz['gene']`, rtol=1e-4/atol=1e-6);
`G_eng` = the engine's recorded `gene` block (RK4-64 self-solve), from `simulation.zarr`. Both
integrate the byte-identical vector field `dg/dt = sigma_alg(g@W_gene^T + u@W_in^T + b) - gamma*g`
(verified against `ode.py:277-278`) from an identical `g0`/`u`/params, so the only admissible
disagreement is the integrator gap + cross-backend float32 rounding. Threshold **5e-3**: the
reference's rtol=1e-4 over the O(8.8) gene magnitudes bounds the integrator gap at ~1e-3, and it
does not amplify — at large drive the algebraic sigmoid saturates (`sigma'->0`), the Jacobian goes
to `-diag(gamma)`, a contraction that damps error toward `g*~sigma/gamma~10`; 5e-3 gives ~5x margin
over that a-priori ceiling yet stays ~2 orders below the O(0.5-1) divergence any real mechanism error
would make in a few frames (logistic instead of algebraic sigmoid; input added OUTSIDE the sigmoid
per the paper vs inside via `W_in`; a spurious cross-cell coupling scaling the drive by N=6; a decay
sign flip). It certifies formula-identity, not a fitted tolerance.

**Result: 6.68e-06 — even tighter than the 1e-3 integrator bound, because the gap is float32, not
Dopri5.** For this smooth, contractive ODE Dopri5(rtol=1e-4) actually resolves the solution to near
float32 precision, so RK4-64 and Dopri5 agree to rounding. The signature confirms it is
integrator/rounding-limited and NOT a reaction-law error: frame-0 deviation is exactly **0** (the
shared seeded IC), it grows MONOTONE to **4.8e-06** by the final frame, and the max sits at frame 17,
cell 0, **gene 4** — the largest gene (~8.4), where float32 rounding is largest (relative 8.0e-07).
Final cell-0 genes agree component-wise to ~1e-6 (ref `[7.3356, 0.3709, 2.9444, 1.3693, 8.7943]` vs
eng `[7.3356, 0.3709, 2.9444, 1.3693, 8.7942]`). The 20 compounding macro-steps over 6 cells
genuinely exercise the recurrent gene->gene coupling, the `W_in` input inside the sigmoid, the
algebraic sigmoid and per-gene decay — a wrong choice on any of the SOURCE-WINS points would have
diverged by O(1), not sat at 7e-6.

**Acted ledger checked FIRST** (`log/atlas/gene_network_connectionist/diag.json`): `regulate` 20/20
calls acted, `moved` 0.823 (nonzero -> the ODE really evolved the gene block); `seed_state` 1/1;
`inert_operators: []`; `valid_evidence: true`. A metric on an inert operator would be worthless — it
is not inert.

**Runs**
- Oracle (reference `f`, jax/diffrax venv): `_oracle/runs/diff_gene_network_connectionist/`
  (`reference.npz` + `summary.json`; deterministic at fixed input, `u` frozen across the rollout;
  script `_oracle/scripts/gene_network_connectionist.py`).
- Engine (Plexus torch): spec `config/atlas/gene_network_connectionist.yaml`; evidence
  `log/atlas/gene_network_connectionist/` (diag.json, metrics.json/.npz, spec_run.yaml, strip.png);
  gene trajectory in `graphs_data/atlas/gene_network_connectionist/simulation.zarr`.
- Score: `python diff_gene_network_connectionist.py score` ->
  `_oracle/runs/diff_gene_network_connectionist/diff.json` (value 6.676e-06, passed true).

**Same initial condition, confirmed.** Both sides start from the identical `g0` (5-vector) and
frozen `u` (2-vector) embedded once by `_gen_gene_network_connectionist.py`; the spec's
`seed_state@frame0` sets the same IC that the oracle's `State.init_empty(...).update(...)` sets, and
`n_frames=20` records 21 frames = the oracle's `g(0..20 dt)`. Frame-for-frame comparison is valid.

**Verdict stands: `new`, `implementation_of: regulate`.** The reproduction confirms the
connectionist (linear-drive) vector field is a faithful implementation of the `regulate` contract to
float32 precision; with the `mwc` (log-occupancy) and `neural_ode` (MLP) siblings differing ONLY in
the drive law under one integration/IO contract, the ODEController subclasses collapse to ONE new
`regulate` contract — the convergence result the ledger exists to record, not three separate
mechanisms.


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

## Implementation (IMPLEMENTER)

**Built:** `src/plexus/operators/candidates/jax_morph_gene_network_mwc.py` — the `regulate`
contract registered with `implementation="mwc"` (`RegulateMWC(Exchange)`, kind=exchange /
family=fields / set=cell, matching the normalized contract). This is the diffuse
`finite_difference`/`spectral` pattern applied to `regulate`: the connectionist and neural-ODE
siblings can later register `implementation="connectionist"` / `"neural_ode"` on the SAME
contract, so the gene-network family stays ONE contract with three interchangeable vector
fields — the convergence the ledger is meant to record. `get_contract("regulate")` now resolves;
`implementations = {"mwc"}`.

**The load-bearing decomposition choice.** The reference's `ODEController.__call__` self-solves
the ODE over the macro-step with diffrax Dopri5 (adaptive) and returns the increment
`y(dt) - y0`. Plexus separates biology from time-stepping: the operator returns the *vector
field* `dg/dt` as a first-order delta on the cell's `gene` block (`EMIT=velocity`,
`INTEGRAND=gene`), and the ENGINE integrates it (`x += dt*delta`, engine.py `_integrate`). This
is the `signal` precedent verbatim (`signal` returns `dv/dt` and lets the engine integrate the
neuron voltage). So the three ODEController subclasses differ ONLY in `f` — exactly the record's
"share the integration+I/O contract, differ only in the vector field" claim, now realised in
code. Sensed inputs `u` are read from a per-cell `sensed` state block and never written (held
fixed across the step), matching the reference's quasistatic-chemistry `_pack(state,
input_specs)`; the upstream `sense`/`chemotax` step is what fills that block.

**FLAG for the curator's differential run (not a translation bug).** The engine takes ONE
explicit-Euler step per tick where the reference takes an adaptive Dopri5 sub-solve of the same
vector field over the same interval. For a non-stiff field at small `dt` the gap is small but
NONZERO, and it will show up in the differ. It is an Axis-A (integration) difference, not a
difference in the mechanism `f`. To compare `f` itself, evaluate the vector field at `t=0`
(`RegulateMWC.vector_field(evolving, inputs)`) against the reference's `vector_field` on matched
state — that isolates the biology from the solver. Multi-step trajectory agreement would require
substepping the gene block (a `substep_dt` micro-loop) or an RK integrator in the engine; I did
NOT do that, deliberately — self-integrating inside the operator is the "category error" the
engine guards against on frame 0.

**Faithful details carried over (ode.py:449-475):** occupancy uses genes/drivers CLAMPED to >=0
(ln(1+.) would NaN on negatives) while the DECAY term uses the RAW evolving state (restorative
toward 0 — the intentional asymmetry); the g/K ratio is capped at `finfo.max` before `log1p`
(the mixed-sign +inf/-inf overflow guard); rho/tau/K come through `_positive_from_log` (clip to
[log tiny, log max - log 4] then exp, so unset log-0 params give rho=tau=K=1); the sigmoid is
`torch.sigmoid` (logistic), NOT the sibling connectionist's rescaled algebraic sigmoid. `H_gene`
orientation verified non-transposed: `H_gene[i,j]` = weight of regulator j on target i (checked
against an asymmetric 2-gene circuit).

**Test:** `tests/test_jax_morph_gene_network_mwc.py` (6 pass). Headline property (a limit,
reference-free): the production term `rho*sigmoid(F)` is STRICTLY in `(0, rho)` for any genes,
drivers, and finite params — the definition of saturating production, and what separates MWC
from an unbounded linear drive. Plus: the inert circuit's fixed point (`dg = 0.5 - g`), the
activating(+)/inhibitory(-) sign of `H`, the restorative-decay asymmetry on a negative
concentration, dormant cells frozen, and one end-to-end engine-integration check. No oracle
numbers hard-coded.

**Not done (next role):** the differential run against the oracle — `evidence.*` stays null,
`status: implemented`.

---

## Differential run (DIFFER)

**Result: `status: validated`. value = 9.54e-07 < threshold 1.0e-4 -> PASS.**

The comparison is on the per-cell vector field `f`, the contract's only distinguishing content,
NOT on a raw integrated trajectory. A trajectory diff would conflate `f` with the Axis-A
integrator: the reference's own adaptive Dopri5 one-macro-step delta sits **0.482** from the
explicit-Euler step (the diagnostic gap, `diff.json:diagnostic_dopri_vs_euler_one_step`), 3-4
orders above the 1e-4 threshold — so a raw engine-Euler vs reference-Dopri5 differ would measure
the solver, not the biology. The metric therefore puts the reference on the ENGINE's own Euler
step, and additionally probes `f` directly at corners the forward sweep never reaches.

**Metric** `value = max(metric_P, metric_A)`, gated on both adversarial dg batches all-finite:
- **metric_P (trajectory, matched integrator)** = 1.19e-07. max over all (frame, cell, gene) of
  `|g_engine - g_ref|`, both sides Euler-stepping the SAME MWC field over 24 macro-steps (dt=1.0)
  from identical `g0`/`u0`/params (float32). Engine and reference agree to float32 rounding.
- **metric_A (vector field, adversarial)** = 9.54e-07 (main params 4.77e-07; extreme params
  9.54e-07). `|dg/dt_torch - dg/dt_jax|` at t=0 on (i) 6 corner-case cell states (negatives,
  zeros, near-zero, large positives, mixed signs) with a negative driver, and (ii) an EXTREME
  param set (log_K at the float32 underflow clip -> K~1.2e-38 so g/K overflows, mixed-sign large
  H) that exercises the `finfo.max` overflow guard and the (+inf)+(-inf)=NaN it prevents. Both
  dg batches finite -> the guard holds; the clamp/raw-decay asymmetry reproduces to rounding.

**Threshold** `1.0e-4`, fixed BEFORE the run (`diff_gene_network_mwc.py` THRESHOLD constant): ~2
orders above the float32 floor for these O(0.1-1) magnitudes (measured ~1e-6) yet ~4 orders below
both the vector-field scale (max|dg|~5-7) and the Dopri-vs-Euler gap (0.482). Passing certifies
same `f`; immune to the integrator. Looser 1e-2 would stop distinguishing `f`-agreement from a
small drive-law error; tighter 1e-8 is below float32 rounding and unmeetable.

**The acted ledger checked first** (`log/atlas/gene_network_mwc/diag.json`): `regulate` 25/25 calls
acted, moved 1.754 (nonzero -> the ODE genuinely evolved the gene block); `seed_state` 1/1;
`valid_evidence: true`. A metric on an inert operator would be worthless — it is not inert.

**Runs**
- Oracle (reference `f`, jax/diffrax venv): `_oracle/runs/diff_gene_network_mwc/`
  (`reference.npz` + `summary.json`; script `_oracle/scripts/gene_network_mwc.py`).
- Engine (Plexus torch): spec `config/atlas/gene_network_mwc.yaml`; evidence
  `log/atlas/gene_network_mwc/` (diag.json, metrics.json/.npz, spec_run.yaml, strip.png).
- Score: `diff_gene_network_mwc.py score` -> `_oracle/runs/diff_gene_network_mwc/diff.json`.

**Note for a re-run.** `diff_gene_network_mwc.py` loads the candidate operator by file path and
`@register_operator` runs at exec time, raising on a second registration of `regulate:mwc`; the
module load is now cached (`_OP_MODULE`) so the two adversarial evaluations reuse one import.

**Verdict stands: `new`, `implementation_of: regulate`.** The reproduction confirms the third
ODEController vector field (MWC log-occupancy) is a faithful implementation of the same `regulate`
contract as its connectionist/neural-ODE siblings — the differential test measures `f`-agreement,
and `f` agrees to float32 rounding. The code-only MWC drive (paper eq. 4 is the linear form)
reproduces exactly, overflow guard and clamp/raw-decay asymmetry included.


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

---

## Implementation (candidate `adhere:hertzian`)

Wrote `src/plexus/operators/candidates/jax_morph_hertzian.py` — the `hertzian` implementation of
the `adhere` contract, registered `@register_operator("adhere", family="interaction", set="cell",
kind="lateral", implementation="hertzian")`. Imports/registers clean; the contract reports
`kind=lateral family=interaction set=cell`, `implementations=['hertzian']`. (SoftSphere, the twin,
is not yet coded, so this registration MINTS the `adhere` contract with `hertzian` as its default
implementation; a later `adhere:soft_sphere` will add the second implementation to the same
contract — the normalized `implementation_of: adhere` in the entry is what makes that legal.)

**Faithful to the SOURCE, not the analytic shortcut.** The operator does not hand-write the radial
force. It builds the total conservative pair energy `E = 0.5 * sum_{live i!=j} (2/5) eps (1 -
r/sigma)^(5/2)` and takes `force = -autograd.grad(E, pos)`, exactly mirroring the source's
`forces = -jax.grad(total_energy)` (potentials.py:L84/:L228). This is deliberate: it REALIZES the
2/5 = 1/exponent trick (autodiff of the energy with the 2/5 in place yields the intended unit force
coefficient `f = (eps/sigma)(1-r/sigma)^(3/2)`) rather than smuggling a fitted constant into a
hand-written force — which is the trap the loop warns against. Ported both source guards verbatim:
`_safe_divide` / `_safe_norm` (double-`where`, finite grad at the sigma=0 padded pair and the r=0
self-diagonal) and the `_compact_repulsion` double-`where` (the fractional power only ever sees a
strictly-positive base). Dead/self masking is EXTERNAL (an alive-mask `where` on the energy, the
`neighbor_sum` seam), not inside the per-pair law — matching the surprise the entry records.

**Routing decisions.** `EMIT="velocity"` (mobility * F): the paper's mechanics is overdamped
(gradient-descent `MechanicalRelaxation` / overdamped `BrownianDynamics`), so the force is a
velocity the engine integrates and sums with any other velocity a cell carries — same routing as
the registered near-miss `attraction_repulsion`. `sigma = r_i + r_j` (ADDITIVE) read from the
per-cell `radius` buffer; per-cell `epsilon` (via optional `epsilon_field`) mixed by the ARITHMETIC
MEAN `0.5*(eps_i+eps_j)`, else a shared scalar — the two-different-combining-rules surprise is
implemented and tested. `mobility` (default 1.0) is the sole added knob (the overdamped 1/gamma),
NOT a fitted constant. DENSE N×N half-summed (the source is dense `neighbor_sum`, not a graph);
fine at atlas cell counts, O(N^2) noted in the docstring. Dimension-generic (`SUPPORTED_DIMS=[2,3]`,
reads D from pos). Made `create_graph` follow `pos.requires_grad and grad_enabled` so a plain
rollout under `no_grad` is cheap/detached while a differentiable inverse loop keeps the force
connected to the state graph.

**Test** `tests/test_jax_morph_hertzian.py` — 8 properties, all stated from the calculus of the
energy, none from the oracle: (1) overlap is repulsive with the EXACT analytic magnitude
`f=(eps/sigma)(1-r/sigma)^(3/2)` and correct direction; (2) compact support — zero force AT and
beyond contact (no tail, no cutoff param); (3) C2 softness — |force| falls monotonically to ~0 as
overlap → 0; (4) Newton's third law — total force over a 12-cell cluster sums to 0 (momentum
conservation of a conservative pair energy); (5) size-consistency — additive sigma, same overlap
fraction → same force fraction, and unequal radii with equal sigma give equal force; (6) dead-cell
masking is external (a dead cell coincident with a live one perturbs nothing); (7) per-cell epsilon
mixes by arithmetic mean (eps=(2,4) acts like shared 3); (8) the force is a real autodiff (grads
flow to positions, finite — no NaN from the fractional power). `8 passed`. Also hand-checked
no_grad rollout (detached forces, 0.5^1.5=0.35355), single-cell no-pair (zero), and 3D.

**Did NOT do / left open (unchanged from normalization):** no oracle run — jax is deliberately
absent from this env AND no oracle script instantiates Hertzian (it would need a new script), so the
`evidence` block stays null and the energy/force shape is confirmed only against hand-derived
calculus, not measured against jax-morph. The differential test against the oracle is the curator's
next step; a passing property test is NOT that. The family-wide reconciliation (pull the abstract
parent + Morse + Harmonic toward `adhere`, and reconsider whether `adhere` is the right name for a
purely-repulsive member) remains flagged for the analysis phase — not settled here.

---

## Differential test (DIFFER) — `adhere:hertzian` VALIDATED

**Setup.** Hertzian is a pure FORCE law (a pair energy autodiffed to a force), so the differential
isolates the force. TWO comparisons on initial conditions the Plexus side reproduces bit-for-bit,
plus a negative control. The PRIMARY exercises exactly the two combining rules the verdict hinges
on — that soft_sphere's uniform test could NOT.

- **PRIMARY — heterogeneous force FIELD (`force_field_rel_err`).** One fixed 7-cell overlapping
  cluster (11 overlapping pairs) with UNEQUAL radii 0.40..0.70 (so `sigma = r_i + r_j` varies per
  pair — the size-consistency) and PER-CELL epsilon 1.0..3.0 (so the coupling is the per-pair
  arithmetic-mean mix). `F_ref = jax-morph Hertzian(epsilon).forces` (autodiff of
  `E = 0.5 sum (2/5) eps (1-r/sigma)^(5/2)`); `F_plx` = the Plexus `adhere:hertzian.forward()`
  emitted velocity at mobility=1 (= its autodiff force), radii/epsilon registered as per-cell
  buffers, IC = the same POS7 translated by +[20,20]. Metric = `max_{7 cells, both components}
  |F_plx - F_ref| / max|F_ref|`. One forward() call, NO integrator confound.
- **SECONDARY — deterministic overdamped trajectory (`traj_pos_max_abs`).** `config/atlas/hertzian.yaml`
  (uniform radius 0.5, epsilon 2.0, dt 0.1, 40 frames) run through `plexus.engine`, centroid-aligned
  and diffed per-cell per-frame vs the reference `BrownianDynamics(Hertzian, kT=0, gamma=1)` history
  (kT=0 → `dx = dt*forces` deterministic). `adhere` gated `after_frame:1` so frame 0 = shared IC.
- **NEGATIVE CONTROL.** SoftSphere (exponent 2) vs Hertzian (exponent 5/2) on the SAME state — the
  wrong-exponent defect the verdict hinges on — in two forms (reference-internal, and Plexus-vs-
  SoftSphere-reference). Both must land >> the bar, proving the metric resolves the neighbour.

**Numbers.**
- Primary `force_field_rel_err = 1.89e-06` (threshold 1e-4, PRE-REGISTERED at diff_hertzian.py:38).
  **PASS.** Peak force magnitudes agree independently: `|F_ref|_max 0.86514318` vs
  `|F_plx|_max 0.86514449` (~1.3e-6).
- Secondary `traj_pos_max_abs = 6.26e-06` (threshold 1e-3, diff_hertzian.py:39). **PASS.** Frame-0
  IC residual 8.9e-7 (shared IC real); final-frame residual == the max (no runaway amplification —
  contractive gradient flow). The cluster genuinely relaxed: max cell move 0.70 world units,
  gyration 0.799 → 0.984, nn-distance 0.668 → 0.977.
- Negative controls FIRE: reference SoftSphere-vs-Hertzian `0.521`, Plexus-vs-SoftSphere reference
  `0.369` — ~5000x the 1e-4 bar. So PASS is discriminating, not a trivially-met bound; a
  wrong-exponent / dropped-2/5-prefactor / mean-instead-of-sum-sigma reimplementation would be
  rejected by O(1).
- Newton's third law holds both sides: summed internal force ~0 (ref 6.7e-8, plexus 3.7e-8).
- Acted ledger: `adhere` calls 40 / acted 40, moved 0.69977516 (== the oracle's own relaxation),
  inert_operators [] → valid_evidence true. (The PRIMARY is a direct forward() call, not gated;
  the ledger validates the SECONDARY trajectory run.)

**Oracle self-check pinned first (the reference is the real law, not a mis-configured oracle).**
2-cell radial scan matches the analytic `f(r) = (eps/sigma)(1-r/sigma)^1.5` to 8.8e-8;
`BrownianDynamics(kT=0)` is deterministic across two PRNG keys (0.0); first Euler step equals
`dt*forces` to 2.5e-8. Oracle provenance: real jax-morph 0.4.0 / jax 0.11.0, isolated venv
(git ace08b8). The two independent autodiff stacks (jax reverse-mode vs torch autograd, float32)
agree to 1.9e-6 — ~50x under the force bar, so 1e-4 is not so tight only bit-identical arithmetic
passes.

**What this certifies (and its scope) — STRONGER than the soft_sphere twin on the disputed axis.**
Reproduces the Hertzian contact force to float32 AND, crucially, exercises the HETEROGENEOUS
`sigma = r_i + r_j` (per-cell radius 0.40..0.70) and the per-cell arithmetic-mean epsilon mix — the
two combining rules (additive sigma, mean epsilon, per-cell coupling) that distinguish `adhere` from
`attraction_repulsion`'s fixed GLOBAL sigma + per-TYPE params. So where soft_sphere left
size-consistency untested, this differential exercises it directly. NOT touched: the adhesive tail
(Hertzian has none by construction — a member property, not a gap) and the per-cell virial pressure
(out of scope — separate VirialStress mechanism). This validates the IMPLEMENTATION; it does not by
itself decide the verdict/dispute (`new → adhere`), which rests on the signature argument in `why`.
Re-ran the scorer this pass — numbers reproduce exactly; `status: implemented → validated`.

**Paths.**
- Oracle: `atlas_jax_morph/_oracle/runs/diff_hertzian/` (reference.npz + summary.json + reference.png);
  script `atlas_jax_morph/_oracle/scripts/hertzian.py`.
- Plexus: `config/atlas/hertzian.yaml` → `log/atlas/hertzian/` (diag.json, metrics.json, metrics.npz,
  strip.png, movie.mp4).
- Scorer: `atlas_jax_morph/diff_hertzian.py` → `log/atlas/hertzian/diff.json`
  (thresholds PRE-REGISTERED at lines 38-39).


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
plexus `log/atlas/lennard_jones/` (spec_run.yaml, diag.json, metrics.json/.npz, strip.png,
movie.mp4); diff `log/atlas/lennard_jones/diff.json`. **Verdict: status validated, D_pos 0.0 <
1e-3, passed.**


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

**Implementation (IMPLEMENTER role).** Wrote `src/plexus/operators/candidates/jax_morph_mechanical_relaxation.py`
(the anti-chamber) and `tests/test_jax_morph_mechanical_relaxation.py`; `status: implemented`.
Torch, not JAX. Registered `@register_operator("relax", family="mechanics", set="cell", kind="lateral")`,
`EMIT="velocity"`, `DIFFERENTIABLE=False`. Imports + registers + 7 property tests pass.

Design decisions (and the impedance mismatches they resolve):
- **Quasistatic overwrite through the delta contract.** The reference step OVERWRITES position with
  the equilibrium `x*` (`state.set('position', x*)`). Plexus has no quasistatic phase and the
  integration invariant forbids an operator writing `pos` directly, so I express the overwrite as a
  velocity: the engine integrates `pos += dt*v`, so emitting `v = (x* - x0)/dt` lands `pos` exactly
  on `x*` in ONE macro-step, for ANY `dt`. That `1/dt` factor is the SIGNATURE of a quasistatic
  step -- the exact dual of `agitate`'s `1/sqrt(dt)` Wiener emit. A test drives three `dt` values and
  confirms the landed configuration is identical (dt-independence).
- **Potential-agnostic solver.** `relax` is the SOLVER, not the force law: a `potential:` param
  selects the energy family it relaxes (`soft_sphere` default / `hertzian` / `harmonic` / `morse` /
  `lennard_jones` / `none`=NoForce no-op). I ported the source's `_compact_repulsion` / `_smooth_cutoff`
  / `safe_divide` / `safe_norm` and the five sigma-relative (`sigma = r_i + r_j`) pair energies, and
  take the force by autodiff of the energy -- so `relax` composes with any of them, orthogonal to
  which interaction. Smoke-ran all five: the adhesive ones (morse/LJ/harmonic) settle to exactly the
  well minimum at contact (sep 1.000), the repulsive ones to at least contact.
- **FIRE, faithfully.** `_fire_to_tol` reproduces the Bitzek-2006 schedule verbatim (carried velocity
  mixed toward the force direction; global power/norm reductions; dt grows after `n_min` downhill
  steps, resets with the velocity on an uphill step); stops at `|grad U|_inf <= f_tol`, `max_steps` a
  fallback. Forward-only: `_force` evaluates `-grad U` on a fresh detached leaf under a local
  `enable_grad` (the engine generates under `no_grad`), so the solver path is never in any outer graph
  -- the source's `custom_vjp` stance ("the answer depends only on `x*`, not how FIRE got there").

What I did NOT reproduce (honest gaps, deliberately):
- **The implicit-diff backward.** `DIFFERENTIABLE=False`. The source's distinctive gradient -- the IFT
  sensitivity `dx*/dp = -H^{-1} d(grad U)/dp`, CG-solved on the physical subspace with the rigid-body
  gauge modes projected out -- is NOT implemented. Reproducing it as a `torch.autograd.Function` with a
  projected-Hessian CG solve is the promotion follow-up. The FORWARD equilibration (what the engine
  runs, what the differ compares) is faithful; the flag correctly steers an inverse loop past this
  implementation.
- **No oracle diff.** `evidence.*` still null -- comparing our relaxation trajectory to `oracle.py` is
  the CURATOR's differential test. I did not run it. My tests assert only contract properties statable
  WITHOUT the reference (force balance `|grad U| <= f_tol`; energy non-increase `U(x*) <= U(x0)`;
  overlap resolved APART to >= contact with centre-of-mass conserved; NoForce no-op limit; translation
  symmetry in free space; dt-independent one-step landing; dead cells held fixed). No fitted constants.
- **Per-cell `alpha` field** (morse well-width as a `StateFieldSpec`) is not supported -- only per-cell
  `epsilon_field`. The source allows either; I judged scope. Note for a reuser.

---

**DIFFER role -- differential test: VALIDATED (`status: validated`).**

*Metric.* `D_eq = max over the 4 LIVE cells of ||x*_plx - x*_ref||_2 / sigma` at the relaxed
equilibrium (frame 1, the single quasistatic macro-step); `sigma = r_i + r_j = 1.0` (dimensionless).
`x*_plx` is the engine's frame-1 position after the `relax` operator's quasistatic emit
`v = (x*-x0)/dt` is integrated (`pos += dt*v`, dt=1.0); `x*_ref` is the jax-morph
`MechanicalRelaxation(Morse)` FIRE-to-`f_tol` equilibrium. **Threshold `1.0e-3` sigma**, pre-registered
in `_analyze_mechanical_relaxation.py` before the diff -- bracketed >100x ABOVE the ~8e-6 sigma
two-solver float32 equilibrium shell (`f_tol/kappa` per side) and >100x BELOW the 0.1135-sigma
relaxation displacement, so it passes legitimate JAX-vs-torch roundoff yet fails any dynamic-creep /
missing-1/dt / fixed-step-descent operator that misses the basin.

*Result.* **`D_eq = 1.91e-06` sigma << 1.0e-3 -> PASS** (~500x margin; per-cell `{0,0,1.9e-6,0}`, one
cell off by ~2 float32 ulp). Corroborators all clean: force cross-check `|grad U|_inf` at the Plexus
equilibrium `1.90e-4 <= 5*f_tol`, matching reference `1.35e-4` (both are genuine force balances of the
SAME Morse energy); gauge-removed Kabsch `D_eq 1.43e-6 ~` raw (no rigid drift); fixed-point plateau
`0.0` both sides (equilibrium held over frames 2-6); frame-0 == IC both (`0.0`); dead slots immobile
both (`0.0`); misaligned-vs-IC `0.1135 =` relaxation displacement (equilibrium genuinely reached).
Summary observables agree: gyration ref-equilibrium `0.6858 ==` plexus final `0.68575`; mean_nn
`0.9698 == 0.9698`.

*Acted ledger* (`log/atlas/mechanical_relaxation/diag.json`): `relax` calls 6 / **acted 1** / moved
0.1123 (acts once, then idempotently emits ~0 on the held equilibrium -- the quasistatic fixed-point
signature), `seed_state` acted, `inert_operators []`, `valid_evidence true`. The IC is BYTE-IDENTICAL
(4-cell diamond, oracle-printed 6-decimal float32; buffer 8 with 4 dead slots at the origin).

*Runs.* Oracle `atlas_jax_morph/_oracle/runs/diff_mechanical_relaxation/` (reference.npz + summary.json);
Plexus `config/atlas/mechanical_relaxation.yaml` -> `log/atlas/mechanical_relaxation/`
(diag.json, metrics.json/.npz, spec_run.yaml, strip.png) + `graphs_data/atlas/mechanical_relaxation/`;
analysis `_oracle/scripts/_analyze_mechanical_relaxation.py` -> `log/atlas/mechanical_relaxation/diff.json`.

*What the diff DOES and does not settle.* It validates the **forward equilibration** -- the FIRE-to-
tolerance solver plus the quasistatic `(x*-x0)/dt` emit reproduce the reference equilibrium to float32
solver noise. It does NOT touch the implicit-diff backward (`DIFFERENTIABLE=False`, unchanged from the
Normalizer's honest gap); that adjoint remains the promotion follow-up and is untested by any
differential here.


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

## Implementation (implementer)

Wrote the `neural_ode` implementation of `regulate` at
`src/plexus/operators/candidates/jax_morph_neural_ode.py` (anti-chamber; promotion is the curator's
call after the differential test). `@register_operator("regulate", family="fields", set="cell",
kind="exchange", implementation="neural_ode")`. The `connectionist` sibling was ALREADY implemented
(`jax_morph_odecontroller.py`); it established the `regulate` interface, and the whole point of the
"three implementations of one contract" framing is that they be genuinely INTERCHANGEABLE. So I did
not invent a second interface -- `neural_ode` is a DROP-IN sibling of `connectionist`, byte-for-byte
identical except the vector field:

* SAME signature/routing: `EMIT="velocity"`, class `INTEGRAND="gene"` + instance-`INTEGRAND` routing,
  `MAPS=[]` (intracellular), one first-order `state:` block (`gene` = the source's `concat(hidden,
  outputs)`), a read-only frozen `inputs:` driver block, `hidden_size` as a documented latent/output
  split that does not change the integration (whole block solved together).
* SAME inc/dt EMIT scaling: the source is a DYNAMIC step returning the exact increment `g(dt)-g0`
  which the Model ADDS; Plexus does `g += dt*delta`, so we return `(g(dt)-g0)/dt` and the engine's
  `dt*` recovers the endpoint (dt cancels; not a second integration).
* SAME adaptive Dopri5(4): I mirrored the sibling's batched Dormand-Prince 5(4) verbatim -- same
  tableau, embedded 4th-order error, RMS error norm over the stacked per-cell state (diffrax's
  default -> one shared step sequence), I-controller `h *= clamp(0.9*err^-0.2, 0.2, 5)`, first step
  = dt. Mirroring (not swapping in torchdiffeq) keeps the two siblings differing ONLY in `_field`,
  which is exactly the contract's promise. NOT hard-coded to the oracle.

The ONE difference -- the vector field. `_field(g, u) = MLP(concat(u, g))` (autonomous, `u` frozen
and closed over for the whole solve). The MLP is a torch `nn.Sequential` mirroring `eqx.nn.MLP(in,
out, width, depth)`: `depth+1` Linear layers, `activation` after every layer but the last, identity
final activation; `make_mlp` defaults reproduced (width=64, depth=2, relu = jax.nn.relu). It maps
`n_in + n_gene -> n_gene` (the source's `in_size + hidden + out_size -> hidden + out_size`). The net
is built LAZILY on first forward once the block widths are known from the actual cell state (the
Plexus `__init__(params, device)` has no Hierarchy, unlike the source ctor that takes a pre-sized
`mlp`); an optional `net=` param injects a pre-built module (the source's "constructor takes a
pre-built mlp" path, used by the tests), and its shape is validated with the source ctor's ValueError
(`must map in+hidden+out -> hidden+out`). `seed` re-inits deterministically without disturbing the
global RNG. `activation` picks relu/tanh/softplus/sigmoid/gelu.

SOURCE WINS / paper: nothing to reconcile inside the field -- NeuralODE has no paper counterpart and
an uninterpretable RHS (already the entry's headline surprise); the paper-vs-code sigmoid/forcing
contradiction is a `connectionist`-only concern. `DIFFERENTIABLE=True` is honest: the RK steps are
plain torch ops, so autograd flows through the solver (the source is diffrax+equinox differentiable).

Tests: `tests/test_jax_morph_neural_ode.py`, 8 passing, all reference-free (a CONFIGURED known field,
never a fitted oracle number): (a) inject a LINEAR field `dg/dt=-k*g` (single Linear, y-columns only)
and check the increment lands on analytic `g0*exp(-k*dt)` to 1e-4 through `g += dt*delta`; (b) the
same endpoint through the real engine `_integrate` (the `_delta_blocks` routing composes); (c)
driver-freezing -- when the net ignores `u`, two different `u` give the identical increment; (d)
zero field -> ~zero delta (float32 solver roundoff ~1e-6 only); (e) dormant (occ=0) cells get zero;
(f) `hidden_size` split integrates the whole `gene` block as one coupled vector; (g) wrong-shape
injected net rejected at forward; (h) `neural_ode` and `connectionist` register as two
implementations of the ONE `regulate` contract with shared `EMIT`/`INTEGRAND`/`MAPS`.

Could NOT do (left for the validator/curator): the differential run against the oracle. jax is
deliberately absent from the Plexus env, and -- per the excavator's UNKNOWN -- no oracle script /
smoke trajectory is known to instantiate NeuralODE (vs the gene-network controllers), so there is no
paired scenario to diff against yet. Both this and the oracle use adaptive Dopri5 at the same rtol,
so tolerance-level agreement is expected once a scenario exists, but that is the validator's evidence
to produce. `status` advanced only to `implemented`; `evidence`/`status: validated` untouched.

## Differential validation (differ)

**Verdict: `validated`. D_max = 2.22e-16 (machine epsilon) vs threshold 3e-3 -> PASS by ~13 orders
of magnitude.**

The implementer's "no scenario to diff against" concern is real but resolvable: NeuralODE's entire
behaviour IS its MLP vector field, and an MLP cannot cross the JAX/torch boundary through a YAML
spec, so a `run_spec.py` trajectory would compare two DIFFERENT (independently-initialised) fields
and measure nothing. The faithful test is at the OPERATOR level: build the MLP once in JAX, integrate
the reference `ODEController.__call__` over one macro-step on a fixed per-cell IC, export the exact
per-layer weights + the reference endpoint, then reload those weights VERBATIM into the torch
operator and drive it from the same `g0/u/dt` through the REAL engine `_integrate` (`gene += dt*delta`).

Pre-registered metric (written to the record BEFORE the cross-run comparison):
`D_max = max over cells (N=16), evolving components, dt in {0.5,1,2}, and both circuit shapes
(A hidden=0/out=3, B hidden=2/out=2) of |y_plexus(dt) - y_ref(dt)|`; threshold **3.0e-3** = ~3x the
reference's own MEASURED truncation-from-truth (9.9e-4 at dt=2), the tightest bound two independent
adaptive Dopri5 controllers of an identical field can be held to (a tighter 1e-3 risks a false fail
from integrator jitter alone).

Result:
- **D_max (plexus vs reference) = 2.220446e-16** — machine epsilon; verdict does not hinge on the threshold.
- **net-equality (torch MLP vs exported JAX MLP) = 0.0** — bit-identical field, so the endpoint match is the INTEGRATOR.
- `plexus_vs_truth == reference_vs_truth` to ~15 digits (dt=2/A: 9.8942968251614e-4 vs 9.8942968251636e-4) — both controllers take the SAME accept/reject substep sequence; they diverge from machine-truth identically and from each other by ~0.
- **negative control** (drop the /dt mean-rate conversion, endpoint = g0 + delta) at dt=2: A=0.408, B=0.252 — >100x the threshold, so the metric would catch a genuine wiring bug.
- Determinism of the reference confirmed in the oracle (fixed key gives fixed endpoint) before anything was recorded.

Runs:
- oracle (JAX reference): `atlas_jax_morph/_oracle/runs/diff_neural_ode/` (reference.npz + summary.json + _provenance.json); built by `_oracle/scripts/neural_ode.py`.
- plexus differential driver: `atlas_jax_morph/diff_neural_ode.py` -> artefact `log/atlas/neural_ode/diff.json`.

Caveat recorded (not a defect): this is an operator-level differential, not a `run_spec.py`
trajectory, and NeuralODE appears in no reference composition and no paper equation — so the delta
was exercised through the real engine `_integrate`, but there is no morphogenesis trajectory to
compare. The `regulate` contract's black-box implementation reproduces the reference exactly.


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

## saturating_cell_growth (IMPLEMENTER, status -> implemented)

Wrote `src/plexus/operators/candidates/jax_morph_saturating_cell_growth.py` (anti-chamber) and
`tests/test_jax_morph_saturating_cell_growth.py` (9 passing). `module`/`test`/`status` updated.

**The refinement is REJECTED by the registry -- and that settles the normalizer's refinement-vs-alias
debate toward refinement.** The normalizer left the door open: "a downstream role that finds the
registry can already host a delta-emitting growth realization *without* editing cell_grow's signature
should downgrade this to `alias`." It cannot. `@register_operator("cell_grow", implementation=...,
kind="lateral"|"field")` raises at import next to the shipped structural `cell_grow`:
`operator 'cell_grow' implementation 'saturating_radius' has kind 'lateral', but the contract's kind
is 'structural'; implementations may differ only in numerics` (registry.py:131). So this mechanism
CANNOT be a silent second implementation of the frozen contract -- admitting it requires deliberately
widening cell_grow's kind, i.e. a language change a downstream user must be shown the cost of. That is
exactly the "refinement hides a breaking change" the ledger exists to catch, here enforced by the
registry itself, not merely argued. The candidate therefore registers under the distinct name
**`grow_radius`** (family=growth, set=cell); unifying it with cell_grow is the curator's call.

**`kind: field` is read as "delta-emitting"; the faithful concrete kind is `lateral`.** Plexus's
`field` kind is a FIELD's own grid self-dynamics (diffuse/decay), `EMIT=None`, `forward()` returns
`{}` -- it CANNOT emit a per-cell `radius` delta. The operator is a per-cell autonomous ODE with no
neighbour coupling and no field, which is precisely the shipped `attractor_flow`/`signal` shape
(`dx/dt=f(x)`, kind=lateral). So I registered `kind=lateral` and recorded the field->lateral reading
as a surprise. (The entry's contract.kind stays `field` -- the normalizer's artefact; the surprise +
this note flag the concrete kind for the curator rather than my silently rewriting the verdict.)

**Routing (faithful to the source's dynamic-delta contract).** The source returns the EXACT increment
`dr = (R-r)(1-exp(-k*dt/R))` as a dt-scaled delta the Model adds. Plexus integrates a first-order
block as `x += dt*delta`, so -- identical to the `regulate:neural_ode` sibling's self-solved increment
-- I return the mean RATE `delta = dr/dt`; the engine's `dt*` recovers the exact endpoint (dt cancels,
not a second integration). `EMIT="velocity"`, class `INTEGRAND="radius"` (a NON-coordinate first-order
block, so `_resolve_emit` never constrains the cell's spatial order) + instance-`INTEGRAND` routing to
a configurable size block; the delta lands in the `radius` accumulator, summed with any co-writer.
`growth_rate` is read as a per-cell heritable STATE (block or buffer), falling back to a uniform `rate`
param (default 0 -> k=0 -> byte no-op); the sole "target" knob is `max_radius` (guarded > 0). Pure
torch (exp/mul) -> `DIFFERENTIABLE=True`, so gradients flow through r, k, and the optimizable R -- the
whole point of keeping the rate in state.

**Source wins (rule 5):** translated the CODE (von Bertalanffy exact flow, no clamp), not the paper's
prose (`min(R+dR, Rmax)`, constant-rate), as the excavator/normalizer established. Not hard-coded to
the oracle -- the tests assert facts about the ODE, never a fitted number.

Tests (all reference-free): (a) one step reproduces the ODE's KNOWN analytic solution
`r(dt)=R-(R-r0)exp(-k dt/R)` through `radius += dt*delta` to 1e-6; (b) UNCONDITIONAL STABILITY -- even
dt=1e3 never overshoots R while a forward-Euler step with the same dt blows past it (the exact-flow
headline); (c) fixed point at R (dr=0); (d) sign symmetry about R (below grows, above relaxes down);
(e) k=0 exact no-op; (f) 400-step asymptote to R, approached from below, no overshoot; (g) dormant
(occ=0) cells get zero; (h) the end-to-end path through the real engine `_integrate` lands on the
analytic endpoint (the `_delta_blocks` radius routing composes); (i) registration/routing
(kind=lateral, EMIT=velocity, INTEGRAND=radius, MAPS=[], writes=[radius]).

Left for the validator/curator: the differential run against the oracle. jax is deliberately absent
from the Plexus env, and no oracle/smoke scenario is known to exercise SaturatingCellGrowth yet, so
there is no paired trajectory to diff -- `evidence`/`status: validated` untouched. Curator decision:
whether `grow_radius` promotes as a widened second implementation of `cell_grow` (a kind-widening of
the growth contract) or as its own contract.

## saturating_cell_growth (DIFFER, status -> validated)

**PASS, exactly.** `value = max(D_max_A, D_max_B) = 0.0` vs `threshold = 1.0e-5`. The Plexus
`grow_radius` operator and the jax-morph `SaturatingCellGrowth` reference produce **bit-identical
float32 radius trajectories** over all 20 macro-steps in both scenarios (`D_max_A = 0.0`,
`D_max_B = 0.0`). Reproduced from scratch in the Plexus torch env.

**Metric.** `D_max` = max absolute per-cell radius deviation over every recorded frame (t=0..20,
dt=2.0) and every live cell, in radius units, reported as `max(D_max_A, D_max_B)`:
- `D_max_A` -- the 4-cell engine run of `config/atlas/saturating_cell_growth.yaml` exactly as
  `run_spec` runs it (r0=0.30, k=0.40, R=0.6) vs oracle scenario A.
- `D_max_B` -- a 6x6 (r0 x k) 36-cell grid rolled through the real engine `radius += dt*delta` vs
  oracle scenario B; spans k=0 no-op, small-k near-linear, large-k saturation (Euler would
  overshoot at dt=2), r0==R, and r0>R (relaxation DOWN to R -- the no-clamp claim).

**Why isolated.** Growth is run ALONE (no relaxation, no division) so both sides keep the same
cells in the same array slots and align cell-for-cell, frame-for-frame. Division's hazard
p=1-exp(-division_rate*dt) never reads radius (`division.py _dist`), so division -- not growth --
is what makes the anchor's counts diverge (124 vs 82); isolating growth is the faithful test of
THIS operator's contract, the per-cell radius ODE. The KNOWN OPEN 124-vs-82 discrepancy is NOT a
growth defect and stays with division, unexplained by this operator.

**Threshold justification (fixed before the run).** 1e-5 sits ~100x above float32 round-off (the
reference's own float32-vs-float64-analytic error is 9.5e-8) yet ~5e4x below the negative-control
signal (0.55), so it separates bit-level agreement from any real disagreement -- wrong law,
dropped dt-scaling, Euler overshoot, or a min-clamp.

**Discriminating controls (all clean).** Negative control (drop the `/dt` mean-rate convention,
`radius += dt*dr`) scores 0.55, ~5e4x above threshold -- the metric provably catches a wiring
defect precisely because dt=2 makes the convention observable. `frame0_is_ic_residual = 0.0`
(frame 0 is the pure seeded IC), `k0_noop_drift = 0.0`, `aboveR_last_min = 0.6 = R` (no clamp).
Acted ledger valid: `grow_radius` acted 13/20 calls (the later 7 emit a sub-float delta once
radius has reached R -- correct saturation), `seed_state` 1/1.

**Paths.**
- oracle run: `atlas_jax_morph/_oracle/runs/diff_saturating_cell_growth/` (reference.npz, summary.json)
- oracle script: `atlas_jax_morph/_oracle/scripts/saturating_cell_growth.py`
- engine spec A: `config/atlas/saturating_cell_growth.yaml` -> evidence `log/atlas/saturating_cell_growth/`
- grid spec B: `atlas_jax_morph/saturating_cell_growth_gridB.yaml`
- diff driver: `atlas_jax_morph/diff_saturating_cell_growth.py` -> `log/atlas/saturating_cell_growth/diff.json`

**Verdict unchanged.** The differential does not touch the `refinement`-of-`cell_grow` verdict --
either reading is still cell growth toward a maximum size. What it establishes is that the
`grow_radius` contract, as implemented and as the engine runs it, reproduces the source's
saturating von-Bertalanffy per-cell radius flow to the last bit. `status: validated`.


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

---

## Implementer: operator `mechanosense`

Wrote `src/plexus/operators/candidates/jax_morph_virial_stress.py` and test
`tests/test_jax_morph_virial_stress.py`. Both import/register clean; test passes 7/7.
`status: implemented`.

**Faithful, not fitted.** Modelled the shape on the `adhere:*` torch ports (same
`safe_norm`/`safe_divide`/`_smooth_cutoff`/`_compact_repulsion`) and the `grow_radius`
read-block-or-buffer recipe. Key decisions:

- **Pure sensing = `EMIT=None`, `forward` returns `{}`.** The operator writes the per-cell `stress`
  scalar in place (a schema block if the set declares one, else a lazily-provisioned buffer) and
  moves nothing. Writing a `stress` state block mutates `Level.state`, which the engine's frame-0
  integration-invariant guard (`engine.py:709-723`) flags -- VirialStress is exactly the
  DERIVED-READOUT category that guard exempts, so the class sets `MAY_MUTATE_INTEGRATED_STATE =
  True`. A test asserts pos/vel are byte-identical after the call (it really moves nothing).
- **Pair law as a plug-in (`potential:` selector), reduced -- not a force.** `dU/dr` is taken by
  autodiff of the FULL pair energy (`torch.autograd.grad(U.sum(), r)`, elementwise) so the
  smooth-cutoff switch term rides in it, exactly as `jax.grad(pair_energy)` does. Default `morse`
  (paper mechanics, eps 3.0); also soft_sphere / hertzian / harmonic / lennard_jones, each with its
  own knobs. The reduction (`r_ij . dU/dr` over live non-self j) is the rank-0 SENSED scalar, versus
  `adhere`'s rank-1 `-grad_i U` that MOVES cells -- the whole reason this is `new`, not a widening of
  `adhere`.
- **The three biology-carrying conventions are in and tested:** minus sign (compression-positive;
  the Morse/harmonic adhesive tail correctly reads tension-negative in the smoke run), 1/(2d)
  Irving-Kirkwood + 1/2 bond split, and V_i = the cell's OWN d-ball volume (2r / pi r^2 / 4/3 pi r^3,
  branched on `d = pos.shape[-1]`). `safe_*` keep the r=0 diagonal and a dead cell's V_i=0 finite.
- Differentiable through the coupling: with a per-cell `epsilon_field`, `d stress / d epsilon` is
  finite/correct (the source's "optimizable through the written stress"); inert under the engine's
  `no_grad` rollout.

**Property test (no oracle numbers).** Anchor = compression-positive sign + the analytic
soft_sphere pressure `p = eps r (1 - r/sigma)/(sigma 2 d V)` (derived BY HAND from `dU/dr = -eps(1 -
r/sigma)/sigma`): two overlapping cells read exactly that, strictly > 0. Plus beyond-contact limit
(r>=sigma -> 0), translation/reflection + identical-cell symmetry, size normalization (bigger V ->
smaller pressure at equal overlap), moves-nothing, dead-cell masking, and the no-block buffer path.

**Left for the differ (did NOT establish):** no oracle run yet -- the FORM and calculus-level
properties are verified, but there is no run-vs-reference stress number. One simplification to watch:
I differentiate `dU/dr` against a detached leaf clone of `r`, so the *value* is exact and the
*epsilon*-gradient is preserved, but the position-gradient path through `dU/dr` is dropped (values --
what the differ compares -- are unaffected).

---

## Differ: `status: validated` — stress_rel_err 7.6e-8 « 1e-4

**Verdict: PASSES.** The Plexus `mechanosense` operator reproduces the jax-morph
`VirialStress(Morse)` per-cell virial pressure to numerical round-off. The differ answers exactly
the question the implementer flagged as open: does the hardened reimplementation match the original
numeric stress? It does.

**The test.** `mechanosense` MOVES NOTHING, so there is no trajectory — the observable IS the
written scalar on ONE frozen configuration. Oracle (`_oracle/scripts/virial_stress.py` ->
`_oracle/runs/diff_virial_stress/`) builds one rich byte-identical float32 IC: 5 *isolated* pairs
at r = 0.7 / 1.0 / 1.3 / 2.0 / 3.0 (compression / well-min / tension / tapered / beyond-cutoff),
one dense 8-cell sunflower cluster (the multi-neighbour sum), 4 dead padding slots (masking), all
radius 0.5 so sigma = r_i + r_j = 1.0. The spec `config/atlas/virial_stress.yaml` shares that IC
cell-for-cell (`seed_state` sets radius 0.5; `mechanosense` reduces Morse eps 3.0 alpha 2.8) and
run through `plexus.engine` -> `log/atlas/virial_stress/`. Metric = peak-normalized relative stress
error over the 18 live cells on frame 0, plus a dead-slot |stress| guard and a frames-identical
guard; pre-registered at 1e-4 / 1e-6 in `_oracle/scripts/_compare_virial_stress.py` (lines 72, 81,
82).

**Numbers** (`_oracle/runs/diff_virial_stress/compare.json`, `summary.json`):

- PRIMARY engine Morse: **rel 7.61e-8** (max_abs 1.9e-6 on peak 25.06) — 3+ orders under the 1e-4 bar.
- Dead slots: **0.0 on both sides**; **frames_identical true** (sensor is a still life across all 4 frames).
- Acted ledger (`log/atlas/virial_stress/diag.json`): `mechanosense` calls 5 / acted 1, moved 0.0;
  `seed_state` acted; `inert_operators []`; `valid_evidence true`.
- Oracle internal gates: STEP == `PairwisePotential.virial_pressure` bit-for-bit; PRNG-key-independent.
- Analytic cross-check (by-hand single-neighbour Morse): max|dev| 3.2e-6 — r=0.7 -> +11.414
  (compression, sign +), r=1.0 -> 0 (well minimum), r=1.3 -> -1.706 (tension, sign -). The
  compression-positive convention and 2 d V_i normalization are correct, not merely self-consistent.
- SUPPLEMENT (direct operator, paths the equal-radius engine run can't reach): all 10 law/radii
  points pass — 4 other pair laws on the uniform IC (rel <= 1.3e-7) and all 5 laws on a second
  UNEQUAL-RADII IC (rel <= 3.3e-7), exercising sigma = r_i + r_j and per-cell d-ball V_i.

**Runs:** oracle `_oracle/runs/diff_virial_stress/` · plexus `log/atlas/virial_stress/` · scorer
`_oracle/scripts/_compare_virial_stress.py` · value 7.610894538878732e-08 · threshold 1e-4 · **passed**.

**Scope / not settled.** Validates the written VALUE across five laws, equal + unequal radii, masking,
and the sign / normalization / bond-split conventions. It does NOT touch the verdict/dispute axis, nor
the differentiability claim (d stress / d epsilon) — values, not gradients, are compared, and the
port detaches the position-gradient path through dU/dr. The SOURCE-vs-PAPER contradiction is
confirmed as a FORM difference, not resolved numerically: the test scores the code's Irving-Kirkwood
virial pressure (SOURCE WINS), not the paper's taxicab sigma_i.

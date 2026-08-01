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

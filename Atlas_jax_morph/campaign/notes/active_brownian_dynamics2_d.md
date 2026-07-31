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

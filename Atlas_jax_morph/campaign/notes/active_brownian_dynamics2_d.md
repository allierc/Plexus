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

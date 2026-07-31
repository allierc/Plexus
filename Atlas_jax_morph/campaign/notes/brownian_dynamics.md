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

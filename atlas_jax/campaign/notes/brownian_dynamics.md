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
- oracle:  atlas_jax/_oracle/runs/diff_brownian_dynamics/ (script _oracle/scripts/diff_brownian_dynamics.py)
- plexus:  log/atlas_jax/brownian_dynamics/{diag,metrics,diff}.json (specs config/atlas_jax/brownian_dynamics{,_dt05,_dt025}.yaml)

Verdict: `new` -> `agitate` (motion family, cell set), **status: validated**.

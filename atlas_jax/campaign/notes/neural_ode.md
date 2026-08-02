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
- plexus differential driver: `atlas_jax_morph/diff_neural_ode.py` -> artefact `log/atlas_jax/neural_ode/diff.json`.

Caveat recorded (not a defect): this is an operator-level differential, not a `run_spec.py`
trajectory, and NeuralODE appears in no reference composition and no paper equation — so the delta
was exercised through the real engine `_integrate`, but there is no morphogenesis trajectory to
compare. The `regulate` contract's black-box implementation reproduces the reference exactly.

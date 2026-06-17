# The Well in Plexus — what it takes

*Prototype answering: "what does it take to implement* The Well's *simulations in
the Plexus framework (spec → operators, categorical separation)?"*

Dataset cited throughout: **The Well**, Ohana et al., *NeurIPS 2024* —
`github.com/PolymathicAI/the_well` (cloned at `papers/the_well`). 15 TB, 16 PDE
families. We took the three that the existing Plexus/ParticleGraph machinery most
directly covers, simplest first.

---

## 1. The headline: the water prototype proved one half of the framework, the Well proves the other

`plexus.tex` Part I says the container holds **two** kinds of entity: **sets**
(discrete, Lagrangian — `Level`) and **fields** (continuum, Eulerian — `Field`).
The `water/` prototype is *all set*: MPM particles carried by an Exchange operator
(P2G/G2P). It never exercised a field as the *primary* state — `GridField` was only
a scalar morphogen for chemotaxis (diffusion + decay).

**Almost every Well dataset is the opposite: the state lives on a grid, and the
operator IS the field self-update.** So implementing the Well is not "new physics
bolted on" — it is *finishing the categorical square*. Concretely:

| | **state on a SET** (Level) | **state on a FIELD** (grid) |
|---|---|---|
| **carried by Exchange/lateral** | `water/` MPM particles | **`well/` Gray-Scott, acoustic** |
| **example op** | `mpm` (P2G/G2P) | `reaction_diffusion`, `wave_acoustic` |

The one Well dataset that *is* set-shaped — `active_matter` (rod-like swimmers) —
we did with the **same** `Level` + `edge_index` + Rewire machinery the water
particles used (`active_matter` + `radius_graph`), to show a single repo spans both
columns by **changing only the spec**.

---

## 2. What was already there vs. what had to be added

**Reused unchanged (the framework primitives):**
- `plexus.models.base.Field` / `Level` / `Hierarchy` — the entity contracts.
- `plexus.models.registry` — `@register_operator` / `@register_field` dispatch.
- The **spec-is-the-API** loop: a typed YAML validated against the registry before
  any run (`well_schema.py`, mirroring water's `scenario_schema.py`).
- The **obstacle→mask** idea from water: one rasterized wall mask, reused by *both*
  the chemical field BC and the acoustic operator (sound-hard reflectors / maze).

**Added — and it is small (≈3 files of operators+field, ~500 lines):**
- `well_fields.py` — `MultiField`, a **multi-channel** grid `[C, nx, ny]` with
  per-axis BCs (periodic / Neumann-reflecting / open-absorbing) and static
  coefficient maps (acoustic `rho`). `GridField` was single-channel scalar; the
  only genuinely missing primitive was "a field whose state is a *vector* of
  channels and whose update is a *registered operator*, not a hard-wired
  diffusion."
- `well_ops.py` — three operators, one per Well family (see §3).
- `well_engine.py` — a field-aware engine (build fields/sets from spec → init →
  run schedule → record). It is the *same shape* as water's `engine2.py` (build →
  schedule loop → record), minus the MPM-specific cell/particle assumptions.

**The verdict:** implementing a new Well family = **one registered operator + one
`MultiField` channel layout + a YAML**. No engine edits. That is exactly the
"extensible: new behaviour is one new registered operator" claim of `plexus.tex`
Part II, now demonstrated on field PDEs.

---

## 3. The three operators map cleanly onto the operator calculus

The framework's operators are dispatched by **what they change** (state / relations
/ entities) and **the relation they act on**. Each Well family lands on one:

### (a) `reaction_diffusion` — Gray-Scott → kind `exchange`, FIELD self-update, 1st order
```
dA/dt = Da ∆A − A B² + f(1−A)
dB/dt = Db ∆B + A B² − (f+k)B
```
This is **Lateral on a grid**: the Laplacian ∆ is the graph Laplacian of the
4-neighbour lattice — exactly the message-passing form in ParticleGraph's
`generators/RD_Gray_Scott.py` (cited in the operator), where `∆ = Σ_j L_ij (uv)_j`.
The reaction term is a pure node-local ODE. So "reaction-diffusion" = *Laplacian
(Exchange/Lateral) + local ODE*, which is precisely the GNN = "ODE vector field +
Laplacian" decomposition in the glossary. **6-pattern zoo** (spots/gliders/bubbles/
maze/worms) comes from changing only `(f,k)` and the IC — the spec-is-the-API
claim, verbatim.

### (b) `wave_acoustic` — acoustic scattering → kind `exchange`, FIELD, 2nd-order leapfrog
```
p_t = −K(u_x + v_y);   u_t = −(1/ρ) p_x;   v_t = −(1/ρ) p_y
```
A 3-channel field `[p,u,v]` over a heterogeneous medium `ρ(x,y)` (a *static
coefficient field* — the framework's "fields bound to a level" generalized to
"fields that parameterize an operator"). Boundary conditions are per-axis
(reflecting x / open y) exactly as the Well specifies; obstacles reuse the water
mask. The leapfrog order ↔ the framework's `PREDICTION` (2nd-order/inertial). **6
scenarios:** random inclusions, discontinuous interface, focusing lens, refracting
gradient, double-slit diffraction, connected maze.

### (c) `active_matter` — Vicsek swimmers → kind `lateral` on a SET + `radius_graph` (Rewire)
The Well stores the *continuum* active-nematic theory; we implement the
*microscopic* model it coarse-grains: self-propelled particles aligning to
neighbours. This needs **two** operators, and they are two *different kinds*:
- `radius_graph` (**Rewire**): rebuild `edge_index` each tick = all pairs within R.
  Changes the **relation**, returns no delta — the framework's `Rewire`.
- `active_matter` (**Lateral**): align heading to the neighbour mean (a graph
  Laplacian on that relation) + self-propel. Returns a velocity **delta** the
  engine integrates (1st order).

The famous Vicsek order/disorder transition falls out: φ→0.99 at low noise,
φ≈0.53 at high noise (`results.md`). This is the **same** `Level`/`edge_index`/
integrate loop as the water particles — proof the set-half generalizes off MPM.

---

## 4. What it does *not* yet cover (honest gaps)

The Well's harder datasets need primitives this prototype stubs or skips:

1. **Incompressible Navier–Stokes** (`shear_flow`, `rayleigh_benard`,
   `euler_*`): needs a **pressure-projection / divergence-free constraint** — a
   *global* solve (Poisson), not a local stencil. The framework has no "global
   implicit operator" kind yet; it would be a new `exchange`-like operator with an
   internal linear solve. This is the single biggest missing piece.
2. **Spectral / high-order accuracy.** The Well's Gray-Scott is generated with an
   ETDRK4 spectral integrator; we use explicit Euler on a 5-point stencil. Same
   *phenomenology* (the patterns match), lower *fidelity*. A spectral `Field`
   frame (FFT-based Laplacian) would be a drop-in new `@register_field`.
3. **Coupled multiphysics** (`MHD`, `turbulent_radiative_layer`, supernovae):
   several fields advected by a shared velocity + source terms. The framework's
   multi-field `Schedule` already expresses the *composition*; what's missing is an
   **advection** operator (semi-Lagrangian or flux) — another single registered op.
4. **3D.** `MultiField` is 2D. The stencils generalize trivially; the renderer does
   not.
5. **The inverse problem.** The Well is a *learning* benchmark (predict next frame).
   Here we only run the *forward* simulator. But every operator is pure and
   differentiable (Euler/leapfrog, no in-place index tricks), so the same rollout is
   trainable — wiring an autograd loss is future work, not a re-architecture.

**Pattern of the gaps:** every one is "add a new registered operator or a new
`Field` frame," never "change the engine." That is the framework working as
advertised.

---

## 5. Files

```
well/
  well_fields.py     MultiField: multi-channel grid, per-axis BC, coeff maps  (new Field)
  well_ops.py        reaction_diffusion | wave_acoustic | active_matter + radius_graph
  well_engine.py     generic build+run (fields and/or particle set)
  well_schema.py     typed spec loader + registry validation (verifiable boundary)
  render_well.py     field gif + montage   render_am.py   particle gif + montage
  well_suite.py      exhaustive suite -> results.md, run.log, gallery.png
  scenarios/*.yaml   15 specs (the whole "experiment" is these files)
```

Run everything:
```
PYTHONPATH=/workspace/Plexus/src python well_suite.py     # 15 sims, ~8 min on one A6000
```

15/15 pass their intent check (`results.md`). The "experiment" — like water's — is
*only the YAML*: every simulation here is produced by the same engine and the same
three operators, changing nothing but the spec.

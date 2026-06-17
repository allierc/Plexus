# Slime in Plexus — what it takes to port SebLague/Slime-Simulation

A faithful port of [SebLague/Slime-Simulation](https://github.com/SebLague/Slime-Simulation)
(`Assets/Scripts/Slime/SlimeSim.compute`, `SlimeSettings.cs`) — itself Jones 2010's
Physarum transport-network model — onto the Plexus spec / operator / categorical-separation
framework (`paper/plexus.tex`). All code cites the source repo.

Run it:

```bash
cd prototype/slime
PYTHONPATH=../../src SLIME_DEVICE=cuda python run_slime.py          # full suite -> gifs, montages, results.md, gallery.png
PYTHONPATH=../../src python render_slime.py slime_two_repel         # one scenario
```

Every scenario writes `specs/<name>.gif` (full progression), `specs/<name>_montage.png`
(6 frames early→late), and the `specs/<name>.yaml` it came from, so the progression is
saved beside its recipe. The suite also emits `results.md`, `run.log`, `gallery.png`.

## The answer: it takes one field + one operator + one thin engine

The original is **two GPU compute kernels** over an agent buffer and an RGBA trail
texture. Both map **one-to-one** onto Plexus primitives, with **no change to the
abstraction** in `paper/plexus.tex`:

| Slime (Unity / HLSL)                         | Plexus primitive                                   | Where |
|----------------------------------------------|----------------------------------------------------|-------|
| agent buffer (`position`, `angle`)           | a single set `cell` (`S_0`), `state[:, :2]` + `cell.heading` | `slime_engine.py` |
| `SpeciesSettings[]` struct array             | cell **`types`** (one type per species)            | `specs/*.yaml` |
| each `SpeciesSettings` field (moveSpeed, …)  | a **per-type property** → per-agent buffer         | `slime_engine.build` |
| RGBA `TrailMap` (4 channels)                 | a **C-channel `TrailField`** (one channel/type)    | `slime_ops.py` |
| `senseWeight = speciesMask*2 − 1`            | the operator's **`cross`** weight (−1 = repel others) | `slime_ops.py` |
| `Update` kernel (sense→steer→move→deposit)   | the **`slime`** operator (Exchange @ cell)         | `slime_ops.py` |
| `Diffuse` kernel (3×3 blur + decay)          | **`trail.diffuse`** (`TrailField.step`)            | `slime_ops.py` |
| `UpdateColourMap` (composite channels)       | the renderer's `composite()`                       | `render_slime.py` |
| `SlimeSettings` globals (trailWeight, decay…)| field params (`deposit`, `decay`, `diffuse`)       | `specs/*.yaml` |
| `SpawnMode` enum                             | `spawn: random/point/disc/ring_in/ring_out`        | `slime_engine.py` |

### Categorical discipline (plexus.tex Part II, §"Categorical discipline")

The spec obeys the five orthogonal categories — *thing→set, process→operator,
space→world, timing→schedule, style→plotting*. Slime is a clean illustration:

| Quantity | Category → home | Why |
|---|---|---|
| `move_speed`, `turn_speed`, `sensor_angle/dist/size` | **Set** → `sets.cell.types` | per-*species* law params (≙ Unity `SpeciesSettings`); the **type owns**, the operator **reads** them via `REQUIRES_TYPE_PROPS` |
| `cross` (inter-species sense weight) | **Operator** → `operators` | a coefficient of the *process*, not of a node |
| `deposit`, `decay`, `diffuse`, `res` | **Field** | the trail continuum's own coefficients |
| `spawn`, `spawn_radius`, `n`, `fraction` | **Set** | what a node *is* / its initial state |
| `boundary`, `seed`, `n_frames` | **World/clock** → `general` | the shared space + the run |
| `colors`, `background`, `gamma` | **Style** → `plotting` | render only; **never** on a set |

The schema **enforces** this: a `color` placed on a type is rejected up front as a
category error (`category error: 'color' on type cell.a is Style … move it to
plotting.colors`) — the dominant failure mode the paper warns about, caught before
any compute. The spec uses the paper's `general:` / `plotting:` block structure.

### Cost: ~3 new files behind the registry, the abstraction untouched

* **`slime_ops.py`** — the only genuinely new physics. One `@register_field("trail")`
  (multi-channel deposit / windowed species-weighted sense / box-blur+decay step) and
  one `@register_operator("slime")` (the `Update` kernel). This is exactly the paper's
  promise — *"new behaviour is one new registered operator, never an engine change."*
* **`slime_engine.py`** — a single-level agent engine. The slime model has **no
  containment and no mechanics**, so it does *not* fit `engine2.py` (which assumes
  `particle ⊂ cell` + MPM to move cells). A flat agent system needs first-order
  self-propulsion only, so it gets its own ~150-line engine — the same pattern the
  prototype already uses (`engine2.py`, `grow_engine.py`, `grow_engine_mpm.py`: one
  engine per world shape). **This is the one real framework gap the port surfaced:**
  the engine layer lacks a generic first-order single-set integrator; here the `slime`
  operator self-integrates (sanctioned for an Exchange op that mutates `H`), and the
  engine still offers a generic `x += dt·Δ` path for any op that returns a delta.

### Free from the framework (what you *don't* write)

* **Validation / the contract** — reused `scenario_schema.load` unchanged: it checks the
  `slime` op is registered, `from`/`to` reference the declared field, `fraction`s sum to
  1, and — via `REQUIRES_TYPE_PROPS` — that **every species declares all five
  `SpeciesSettings` fields** before anything runs (a missing `sensor_angle` is a one-line
  error, not a crash deep in a kernel).
* **Categorical separation** — multi-species *is* the framework's `types` + per-channel
  field + the `cross` sign. Going 1 → 2 → 3 → 4 species (`slime_default` →
  `slime_two_repel` → `slime_three` → `slime_four`) changes **only the spec**: add a
  `type`, get a channel, no code. The single highest-leverage primitive in the paper
  carries the entire multi-species story.
* **Determinism, gif/montage/intent harness, the spec-as-recipe gallery** — same shapes
  as `prototype/water`.

## The exhaustive suite (17 scenarios)

Same breadth-from-one-registry idea as the paper's ten ant variants: every scenario
below is the **same `slime` operator + `trail` field**, differing only in the spec.

**Single-species morphology** (sweep one knob from `slime_default`):
`slime_fine` (capillary mesh) · `slime_coarse` (large cells) · `slime_filaments`
(narrow cone → straight lines) · `slime_curly` (wide cone + high turn) · `slime_sparse`
(high decay) · `slime_dense` (low decay) · `slime_smear` (high diffusion).

**Spawn modes:** `slime_ring_in` (Lague's signature) · `slime_point` (radial burst) ·
`slime_random_full` (global network) · `slime_torus` (periodic boundary).

**Multi-species / categorical separation:** `slime_two_repel` (cross=−1 → segregated
territories) · `slime_two_attract` (cross=+1 → shared network — *only the sign differs*)
· `slime_two_asym` (two species, different `SpeciesSettings`) · `slime_three` · `slime_four`
(full RGBA showcase).

**Parameter sweeps** (`gen_specs.py` → `specs/sweep_*.yaml`, ~32 more gifs): one axis
swept at a time from the default — `sensor_angle`, `sensor_dist`, `turn_speed`, `decay`,
`diffuse`, `deposit`, `sensor_size` — so each gif isolates the effect of one knob.
Regenerate with `python gen_specs.py`. All `.gif`s land directly in `prototype/slime/`.

### Intent checks (it ran ≠ it worked)

`run_slime.py` measures, per run: **coverage** (network exists), **contrast**
(std/mean over inked pixels — filaments score high, a saturated blob low),
**reinforce** (late/early peak trail — the field self-reinforces), and for
multi-species **overlap** (mean pairwise channel overlap — *low* = segregated). A
repel run must reach low overlap; an attract run high; all must be structured.

### Honesty about regime (the subtle bug, per the paper's lesson)

Multi-species runs first looked like a white blob: at high agent density every channel
**saturates to the clamp (1.0) everywhere**, the trail gradient goes flat, the `cross`
repulsion has nothing to climb, and additive RGB blends to white. This is *correct
physics in the wrong regime* — the same trap the paper calls out for self-secreted
chemotaxis. The fix is in the **spec, not the code**: lower `deposit`, higher `decay`,
fewer agents keep the field below saturation, and segregation appears (`overlap` drops
from 0.36 → 0.03). The four-species case is the most density-restrained for this reason.

## Catalog additions

Extending the paper's operator catalog (§"Operator catalog"):

| op / field | kind | acts on | behaviour |
|---|---|---|---|
| `trail` (field) | exchange frame | cell | C-channel pheromone map; deposit / windowed sense / blur+decay |
| `slime` | exchange | cell | Physarum 3-sensor sense→steer→move→deposit; multi-species via `cross` |

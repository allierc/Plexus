# Methodology: top-level LLM ↔ low-level registry code

What this prototype was *for*: not the tissue physics, but figuring out **how an
LLM should drive a registry of low-level operators**. This is the conclusion of
that assay — the contract, the three failure/extension modes, and a proposed way
to build the repo.

---

## 1. The two layers and the contract between them

```
  INTENT      natural language ("2000 cells, two mechanical types, ...")
     │  Claude compiles  (bounded, schema-checked generation)
  SPEC        scenario.yaml  — sets · fields · operators(+selectors) · schedule
     │  engine resolves names against the registry
  REGISTRY    @register_operator / @register_field  — the only place code lives
     │
  ENGINE      generic: build Hierarchy → run Schedule → zarr     (written once)
```

The **spec is the contract**. The LLM owns everything above it (intent → spec);
the code owns everything below (operators). They never touch each other directly —
they meet at a declarative, validated YAML. This is what makes the system both
LLM-drivable and human-auditable: the spec is small, typed, and reviewable, unlike
the 510-line flat `config.py` it replaces.

Three properties make the contract enforceable, and they map exactly to your three
questions:

| Question | Mechanism | Where |
|---|---|---|
| Q1 archive scenarios | spec + seed + metrics + thumbnail, name-keyed | `archive.py` → `scenarios/archive/`, `GALLERY.md` |
| Q2 missing model | registry lookup errors with the available set; add one `Operator` | `registry.get_operator`, `scenario_schema` |
| Q3 missing property | operators *declare* what they require; validator checks | `Operator.REQUIRES_*`, `scenario_schema` |

---

## 2. Q1 — Archiving scenarios (reproduce / re-visualize)

An archived scenario is **four small files**, keyed by name:

```
scenarios/archive/<name>/
  scenario.yaml   # the exact spec — the source of truth
  metrics.json    # generic measurements + seed
  thumb.png       # start→end visual
  (run.gif)       # optional
```

Because every run is **bit-deterministic given (spec, seed)** (the engine forces
`torch.use_deterministic_algorithms`), the spec *is* the archive — you don't store
the trajectory, you regenerate it:

- **Reproduce exactly:** `python reproduce.py <name>`
- **New realization:** `python reproduce.py <name> --seed 7` (same dynamics, new draw)
- **Visualize differently:** the run writes a generic zarr (sets + fields); point any
  visualizer at it (`viz2.py --dot 0.3`, or a future 3D backend).

`GALLERY.md` is the regenerable index over all archived runs. This scales: a
scenario is ~20 lines of YAML, so thousands of them cost almost nothing, and the
gallery is always the current truth (re-archiving overwrites by name).

> Design rule: **archive the recipe, not the dish.** Store spec+seed (tiny,
> reproducible, re-renderable), not the raw trajectory (huge, frozen to one
> visualization).

## 3. Q2 — When the present code is not enough (a model is missing)

Suppose a request needs field *advection*, or reaction–diffusion, or anything not
yet coded. The pipeline **fails loudly and points at the gap**:

```
operator 'reaction_diffusion' not in registry.
Available: ['adhesion','boids','mechanics','motility','mpm','secrete','sense']
```

The fix is **one new `Operator` subclass** — never an engine change:

1. write a class with `forward(self, H, mask) -> {set: accel}` (or mutate a field, return `{}`);
2. decorate it `@register_operator("name", level=..., kind=...)`;
3. declare its needs via `REQUIRES_PARAMS` / `REQUIRES_TYPE_PROPS`;
4. reference it in the spec.

We exercised this live: `adhesion` (type-specific cohesion → cell *sorting*) was a
capability **no existing operator could produce**. Adding it was ~20 lines in
`ops/adhesion.py`; it immediately unlocked the `sort` scenario. The engine, schema,
and all other scenarios were untouched.

> This is the LLM loop's natural shape: **compile NL → spec; the engine reports
> exactly which operators are missing; the LLM writes those (bounded, single-class,
> fixed-interface tasks) and retries.** The "what to write" is never open-ended —
> it's always "one operator conforming to this contract."

## 4. Q3 — When a cell/particle property does not exist

Two cases, both handled:

**(a) An operator needs a property the spec didn't provide → hard error.**
Operators declare `REQUIRES_TYPE_PROPS` (e.g. `mpm` needs `youngs`). The validator
resolves the property along the **containment chain** (mpm acts on particles, but
`youngs` lives on the parent cell's types) and fails before running:
```
operator 'mpm' requires property 'youngs' on every type of 'cell'; missing on type 'soft'.
```

**(b) The spec declares a property no operator reads → warning, not error.**
Usually a typo or leftover; the run proceeds but flags it:
```
[warn] property 'viscosity' on cell.soft is read by no operator (known: ['fraction','youngs'])
```

> Design rule: **capabilities are declared, not discovered at a crash.** A property
> is meaningful only if some operator consumes it; the operator that consumes it is
> the single place that declares the requirement. The validator turns a would-be
> `AttributeError` 8000 substeps deep into a one-line message before anything runs.

---

## 5. What the process taught us (lessons → repo design)

1. **The intermediate representation is the whole game.** Once sets/fields/
   operators/selectors/schedule existed, ten qualitatively different behaviours
   (aggregate, disperse, swarm, wander, squish, sort, chase, mixed, crystal,
   collapse) were *just different YAML* over the same code. Breadth comes from the
   IR, not from new code.

2. **Selectors are the highest-leverage primitive.** `cell[type=soft]` is how
   "only population 1 does X" is expressed, and how two behaviours coexist (`mixed`:
   stiff flock, soft chemotax). Every operator must honour the selector mask;
   forgetting it (boids did at first) silently breaks subpopulation scoping.

3. **Verification must check *intent*, not just "it ran."** The hard bugs were
   physical, not exceptions: unstable diffusion (`dt·D>0.25`), wrong `p_vol`,
   unclamped boids accel blowing up MPM, cell inversion at very low stiffness,
   GPU non-determinism breaking reproducibility, and — most subtly — *the wrong
   metric* (net displacement vs. clustering). A scenario isn't done until a check
   measures the behaviour the sentence asked for.

4. **Honesty about regime.** Self-secreted chemotaxis only aggregates below a
   density threshold (above it the field is flat — correct physics). The system
   should *state* the regime in the spec/verify output, not silently show the one
   parameter set that worked.

5. **Determinism is a feature you must force.** Bit-reproducibility needed
   `torch.use_deterministic_algorithms` because CUDA scatter atomics diverge over
   thousands of substeps. Without it, "reproduce with a new seed" is meaningless.

---

## 6. Proposed methodology for building the repo

**Layering** (strict; no leaks across):
- `models/base.py` — entities + operator base + the capability-contract fields.
- `models/registry.py` — the only name→code map; three registries (entity/operator/field).
- `models/<domain>/*.py` — operators, one concern per file, each declaring `REQUIRES_*`.
- `scenario.py` — the schema + validator (the contract gatekeeper).
- `engine.py` — generic; contains **no** scenario-specific logic.
- `viz.py` — generic over the Hierarchy/zarr; swappable backends (2D now, 3D later).
- `archive.py` + `scenarios/` — the growing, reproducible gallery.

**The build loop** (how the LLM and code co-evolve):
1. User states intent in NL.
2. LLM compiles → `scenario.yaml`, following `SCENARIO_BIBLE.md` (repeatability).
3. Validator runs: unknown op → §3 (write one operator); missing property → §4
   (fix spec or add the requirement); else it's runnable.
4. Engine runs; **verify** checks intent + repeatability + well-formedness.
5. `archive.py` records spec+seed+metrics+thumb into the gallery.
6. New capability needed → one new registered operator (+ its `REQUIRES_*`), never
   an engine edit. The catalog of operators is the system's vocabulary; it grows
   monotonically, and every past scenario keeps working.

**Invariants to hold the line:**
- Code lives only behind the registry; the engine never branches on scenario names.
- Every operator honours its selector mask and declares its `REQUIRES_*`.
- Every scenario ships an intent check; "it ran" ≠ "it worked".
- Archive the recipe (spec+seed), regenerate the dish.
- When a result holds only in a regime, the spec/verify says so.

This is the contract that lets the LLM operate at the top (intent → spec, plus
occasional single-operator authoring) while the low-level code stays a clean,
growing, verifiable registry underneath.

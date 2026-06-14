# The Scenario Bible

The single source of truth for turning a natural-language simulation request into
a runnable TissueGraph scenario. Its purpose is **repeatability**: the same
request, compiled against this bible, yields the same scenario structure every
time. Claude (or a person) follows these rules; the spec is the contract.

> Pipeline: **NL request → scenario.yaml (this bible) → engine → zarr → verify → viz**
> Claude writes a `scenario.yaml`. New code is written *only* when a needed
> operator is absent from the catalog below.

---

## 1. Vocabulary (the only nouns and verbs)

| Term | Meaning | Code |
|---|---|---|
| **set** | a collection of like nodes at one scale | `Level` |
| **containment** | a set is partitioned into its parent (`parent: <set>`) | `Level.parent` |
| **field** | a continuum on its own grid, bound to one set | `Field` |
| **operator** | a GNN/physics step; one of four *kinds* | `Operator` |
| **selector** | `set` or `set[attr=value]` — which nodes an operator acts on | engine mask |
| **schedule** | ordered, per-frame list of operator/builtin steps | engine loop |

**Four operator kinds** (everything is one of these):
`lateral` (within a set), `aggregate` (children→parent, up), `broadcast`
(parent→children, down), `exchange` (set↔field, scatter/gather).

---

## 2. Scenario file structure (required keys)

```yaml
name: <str>                 # required
seed: <int>                 # determinism
n_frames: <int>
record_every: <int>
sets:                       # >=1 set; a leaf set declares `parent`+`per_parent`
  <name>:
    n: <int>                # for top-level sets
    dim: 2
    types: {<t>: {fraction: <f>, <param>: <v>, ...}, ...}   # optional; fractions sum to 1
    parent: <set>           # for contained sets
    per_parent: <int>
fields:
  <name>: {frame: grid, res: <int>, diffusion: <f>, decay: <f>, couples_to: <set>}
operators:
  - {op: <name>, at: <selector>, <to|from>: <field>, <param>: <v>, ...}
schedule:
  - aggregate | <op> | [<op>, <op>] | <field>.diffuse | mpm | integrate
```

Rules the validator enforces (`scenario_schema.py`) — a spec that loads is runnable:
- every `op` exists in the registry; every `at`/`to`/`from` references a declared set/field;
- type `fraction`s sum to 1; every schedule token resolves to an operator or builtin.
- **Never invent a YAML key that is a YAML 1.1 boolean** (`on/off/yes/no`) — use `at`.

---

## 3. Operator catalog (reuse before writing new code)

| op | kind | acts on | key params | notes |
|---|---|---|---|---|
| `boids` | lateral | cell | radius, cohesion, align, separate | collective motion |
| `mpm` | exchange | particle | n_grid, substeps, dt_sub, a_max, drag | MLS-MPM mechanics; soft/stiff via `youngs` |
| `secrete` | exchange | cell→field | rate | deposit into field (`to:`) |
| `sense` | exchange | cell←field | gain | chemotactic accel up-gradient (`from:`) |
| builtin `aggregate` | aggregate | — | — | cell pos/vel = mean of its particles |
| builtin `<field>.diffuse` | — | — | — | field self-update |
| builtin `integrate` | — | — | — | single-level Euler step (point-cell engine) |

If a request needs a verb not here (adhesion, division, haptotaxis, ...), that is
**one new `Operator` subclass** registered with `@register_operator`, conforming to
`forward(self, H, mask) -> {set: accel}` (or mutating a field and returning `{}`).

---

## 4. NL → spec decision rules (the repeatable mapping)

Apply in order; each rule maps a phrase pattern to a spec construct.

1. **"N cells / particles / agents"** → a `set` with `n: N`.
2. **"made of / composed of K X per cell"** → a contained set `particle` with
   `parent: cell, per_parent: K` (two-level hierarchy → use `engine2`/`scenario_mpm`).
3. **"two/several types differing by P"** → `types:` under the set, one entry each,
   `fraction` summing to 1, with `P` as a per-type param.
4. **"mechanical / elastic / rigid / stiff"** → particles + `mpm`; map the property
   to `youngs` (stiffness). Soft = low, stiff = high.
5. **"move by boids / flock / swarm"** → `boids` lateral op at cell level.
6. **"creates / secretes a chemical / field"** → a `field` (grid) + `secrete` exchange.
7. **"follows / is guided by / chemotaxis"** → `sense` exchange op.
8. **"only population X does Y"** → put the **selector** `at: cell[type=X]` on that
   operator. This is the canonical way to scope behaviour to a subpopulation.
9. **timescales** ("fast/slow") → `schedule` ordering; a fast field substep belongs
   inside the frame before the slow cell step.

### Worked example (canonical)
Request: *"2000 cells, two types differing in mechanics (elasticity, rigidity),
moving by boids, creating a chemical field that guides only the first population."*
→ [scenario_mpm.yaml](scenario_mpm.yaml): cell set with soft/stiff `types`
(`youngs`), contained `particle` set (`per_parent: 100`), a `chemical` grid field,
ops `boids`+`mpm`+`secrete`+`sense`, with `sense`/`secrete` selectored to
`cell[type=soft]`. (Note: at extreme density self-secreted gradients flatten, so the
clean demo runs at moderate cell count — see §6.)

---

## 5. Determinism & verification (non-negotiable)

- Set `seed`. The engine enables `torch.use_deterministic_algorithms` so GPU runs
  are **bit-reproducible** (CUDA scatter atomics are otherwise nondeterministic).
- No wall-clock / unseeded randomness anywhere.
- Every scenario must ship checks in the spirit of [verify2.py](verify2.py):
  **repeatable, well-formed (finite/in-domain), structurally sound (cells cohere),
  intent-met (the requested behaviour measurably happened).** "It ran" is not "it
  worked".

---

## 6. Honesty rules (write the limitation, don't hide it)

- If a behaviour only appears in a parameter regime, **say so in the scenario
  comments and the verify output** (e.g. self-secreted chemotaxis aggregates only
  below a density threshold; above it the field is spatially flat — correct physics).
- If a metric is the wrong one, change the metric, don't tune until it passes
  (net displacement vs. nearest-neighbour clustering was such a case here).
- Report the scale a result was demonstrated at; don't imply an untested scale.

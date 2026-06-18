# Task: land field support in `src/plexus` + decompose slime into primitive operators

## Goal
Make the canonical framework (`src/plexus`) run the slime/Physarum model as **1 set + 1 scalar
field + primitive operators** — *without* a monolithic `slime` operator. This also lands generic
**field support** in the engine (reused by every field-coupled model later).

## Design decisions (already made — do not relitigate)
- **Field = pure state**: a scalar grid with N channels/components, NO behavior. Name it by what it
  *is* (`chemical`/scalar), not "trail".
- **All dynamics are operators**, decomposed by category (replacing the monolithic `slime` op):

  | category | operator | does |
  |---|---|---|
  | set → field | `deposit` | each agent adds to the field at its position (its channel) |
  | field → field | `diffuse` | one Laplacian step ∂c/∂t = D∇²c |
  | field → field | `decay` | linear decay c → c − k·c |
  | field → set | `sense` | sample field at 3 sensors (ahead-L/ahead/ahead-R) → turn heading toward higher conc |
  | set (lateral) | `advance` | advance along heading at speed (self-propulsion) — renamed from prototype `motility` |

  Build the four new ones: **`deposit`, `diffuse`, `decay`, `sense`** (+ **`advance`**) + a pure-state scalar-field class.
- Schedule that reproduces slime: `[deposit, diffuse, decay, sense, advance]`.

> **STATUS: done.** Landed in `src/plexus/operators/{scalar_field,deposit,diffuse,decay,sense,advance}.py`;
> engine builds fields + heading/spawn + per-type scalar broadcast; specs in `config/slime/`
> (`slime_default`, `slime_curly`, `slime_two_repel`, `slime_three`) reproduce the Physarum network.
- A field-dynamics operator writes the field in place and returns `{}` (like the existing Exchange ops).
- `<field>.diffuse` builtin token goes away — `diffuse` is just an operator in the schedule.

## The precise gap in `src` (verified)
- ✅ `src/plexus/schema.py` already parses + validates `fields:` and `<field>.diffuse` tokens.
- ✅ `src/plexus/models/base.py` has `Field` class, `@register_field`, `Hierarchy.add_field`.
- ✅ Engine handles sets, types, selectors, first/second-derivative integration.
- ❌ `src/plexus/engine.py` `build()` never instantiates fields from `sim.fields`.
- ❌ `src/plexus/engine.py` schedule loop raises `NotImplementedError` on `.diffuse` (~line 211).
- ❌ Engine only broadcasts per-type `p`-list (`type_params`); slime needs arbitrary per-type SCALARS
  (`move_speed, turn_speed, sensor_angle, sensor_dist, sensor_size`) broadcast to per-agent buffers
  (extend `_assign_types`).
- ❌ No agent `heading` (engine state is pos+vel); slime needs a `heading` buffer + spawn init.

## Reference implementation to port/decompose from
- `prototype/slime/slime_ops.py` — `TrailField` (deposit/sense/step) + `slime` op. **Decompose** its
  `step()` into `diffuse`+`decay` operators; split its sense→steer→move→deposit into `sense`+`deposit`
  (+ reuse `motility` for the move).
- `prototype/slime/slime_engine.py` — shows the flat-agent build (heading init, per-type scalar
  broadcast, field construction, schedule loop). The engine hooks to add to `src` mirror this.
- All prototype operators are inventoried/copied in `src/plexus/operators/candidates/` (+ README).

## Acceptance
A spec with `sets: {cell:{n, types, spawn}}`, `fields: {chemical:{frame:grid,res,components}}`,
`schedule:[deposit,diffuse,decay,sense,motility]` runs in the canonical engine and produces the
Physarum network (reticulating trails). Multi-species = a `types:` change only, no new code.

## Files to touch
- `src/plexus/engine.py` — build fields from `sim.fields`; dispatch field operators; per-type scalar
  broadcast; agent heading + spawn init.
- `src/plexus/operators/` — new `deposit.py`, `diffuse.py`, `decay.py`, `sense.py` (+ scalar field
  class, or extend the registered field). Self-register on import.
- (do NOT touch `prototype/` or the `dicty` loop.)

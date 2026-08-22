# Seed lifecycle audit (read-only, pre-refactor)

Scope: `/workspace/Plexus`, live code only. `_archive/`, `_archive_runs/`,
`ops/archive/`, `prototype/eye/archive/`, `config/okuda/_superseded_*` were
located but not analysed, per the brief's instruction not to let archived
experiments determine live semantics. `papers/` (vendored third-party code)
excluded throughout.

## Headline

`seed` already exists as a first-class `KIND` string (`base.KINDS`, 8
entries, added 6 Aug), is schema-validated, and is engine-gated to the
opening frames. What does **not** exist is a coherent hierarchy or registry
API behind it:

- `class Seed(Structural)` is defined once and subclassed by exactly **3**
  operators, all in `prototype/eye/`. Every other `kind="seed"` operator
  (16 of them) inherits `Structural`, `Exchange`, `FieldUpdate`, or plain
  `Operator` — `KIND` is stamped onto the class by the registry decorator
  regardless of base class, so `isinstance(op, Seed)` is meaningless today;
  only `cls.KIND` is authoritative.
- Two genuine seed operators are **mis-kinded** and therefore escape the
  engine's seed gate entirely: `IntegrinSeed` (`kind="structural"`) and
  `SeedFromSegmentation` (`kind="exchange"`). Both self-guard with an
  instance `self._done` flag instead.
- `kind` and `family` disagree in both directions: two operators carry
  `family="seed"` without `kind="seed"` (the two above); several carry
  `kind="seed"` with a `family` outside the closed `OPERATOR_FAMILIES` set
  (`"growth"`, `"anatomy"`).
- **The single biggest piece of genuine x₀ establishment is not an operator
  at all.** `engine.build()` (~180 lines) does spawn placement, physics-aware
  velocity ICs (`circular_orbit`, `solid_body`, `radial`), type assignment,
  edge-set construction and child scatter, all read directly from raw
  `sets:` YAML keys (`spawn`, `start`, `vel_init`, `types`) that `schema.py`
  does not validate. One config's own comment documents that an operator
  (`disk_ic`) was deliberately *deleted* and folded into spec keys — i.e.
  the codebase currently holds two contradictory positions on where x₀
  belongs, not just an incomplete migration toward one.
- The engine's "runs once" guarantee is actually `SEED_MAX = 10` opening
  frames, and the window is **global**: the largest `before_frame` declared
  by *any* seed in a spec becomes the shared ceiling for every seed in that
  spec.
- `discovery_okuda` never queries the registry for this (`operators_of_kind`
  has **zero callers anywhere in the repo**). It hard-codes name sets in
  four places (`translate.py`, `run_one.py`, `biologist.py`,
  `composition_space.py`) and maintains a second, independent taxonomy
  (`composition_space.OPERATORS[...]["role"]`) that already disagrees with
  the registry (`cell_chem_seed` is `family="seed"` in the registry, `role=
  "driver"` in `composition_space`). It also references an operator,
  `load_mesh_3d`, in four places that **is not registered anywhere** — a
  dangling seed mechanism today, independent of this refactor.
- Separately, `atlas_jax`/`atlas_cc3d` (`record.py`, `agents/atlas_agents.py`,
  `_work/_check_*.py`) hard-copy `KINDS` **without** `"seed"` for their own
  contract-validation rule (`R6`). Any operator that starts declaring
  `kind="seed"` in those subsystems would fail their validator today — this
  is a landmine for step 4/5 unrelated to anything this refactor touches
  directly, but will break the moment a seed operator crosses into those
  systems.

## Core framework (base.py / registry.py / schema.py / engine.py)

**`base.py`** — `KINDS` (line ~128) already has exactly the 8 target
entries including `"seed"`. Three stale comments describe an older
taxonomy and must be cleaned per step 3: the module docstring's kind table
(6 rows, missing `field` and `seed`), `registry.py`'s "one of the seven",
`engine.py`'s "seven kinds ... dispatched by what they touch" (never names
seed). `Seed`'s own docstring references a constant `SEED_WINDOW` and a
method `engine._schedule` — **neither exists**; the real names are
`engine.SEED_MAX` and `engine._seed_window`. `MAY_MUTATE_INTEGRATED_STATE`
is declared on `Operator` (default `False`) and `Structural` (`True`,
inherited by `Seed`); it is also set `True` on several *derived-readout*
operators (`Centroid`, `VirialStress`) and on *per-frame kinematic*
operators (`prototype/eye/tonic_ops.py`, `forced_gaze_ops.py`) — proof the
flag cannot currently distinguish "writes x₀ once" from "writes xₜ every
frame" (relevant to step 12).

**`registry.py`** — `operators_of_kind(kind)` exists (line ~226) and is
the natural seat for `seed_operators()`/`is_seed()`, but has zero callers
today anywhere in the repo. `register_operator(*names, **tags)` accepts
arbitrary tags including `kind=`/`family=`, stamped as uppercase class
attributes — this is *why* `KIND` can disagree with the actual base class.
`OPERATOR_FAMILIES` (line ~244) already lists `"seed"` first with the
comment "establish initial state on a set" — the family axis already
encodes the intended semantics; several live registrations just don't use
it correctly (`"growth"`, `"anatomy"` aren't in the set at all).

**`schema.py`** — top-level required keys are exactly `sets, fields,
operators, schedule` (line ~108). No `seed:` section exists. Kind
validation exists (`kind in KINDS`, line ~209) but nothing validates seed
placement, uniqueness, or scheduling-control usage: `before_frame`/
`after_frame`/`every` aren't in `_RESERVED`, so they fall into `params`
completely unchecked — a seed with `every: 5` would parse and, worse,
partially work (see engine gate below).

**`engine.py`** — one loop (`for tick in ticks: for step in sim.schedule:
_run_token(...)`), no separate seed pass, no `before_frame`/`after_frame`
*hook* API (only spec params of those names read by the gate).
`_seed_window(sim)` computes one shared ceiling from the max declared
`before_frame` across all seeds (capped at `SEED_MAX=10`); `_gate` rewrites
every `kind="seed"` operator's window to `(0, min(before, seed_window), 1)`
regardless of what the spec says otherwise. This is the sole place the
engine special-cases a kind — contradicting the module's own docstring
claim ("the engine never special-cases a kind"). `build()` does far more
than allocate: `_spawn`/`_spawn3d`/`_spawn_pair3d` (positions),
`_init_velocity` (physics-aware velocity ICs — its own docstring says
"Initialization is a spec concern, not an operator"), `_start_centers`,
`_assign_types`, `_build_edge_set`, child-scatter-in-parent, and
`entity.provision()` hooks (e.g. MPM's per-particle `F`/`C`/`mass`/`mu`
allocation *and* material-band assignment bundled together).

## Classification table

Legend: **A** genuine Seed semantics · **B** runtime allocation only ·
**C** dynamics · **D** derived/readout · **E** compatibility alias ·
**F** historical/archive/dangling.

| file | symbol | current behavior | current kind | class | why / target behavior |
|---|---|---|---|---|---|
| `discovery_okuda/ops/mesh_ops.py:211` | `SeedMesh3D(Structural)` | builds mesh + vertex x₀ + edge table + `A0/P0/V0`, once | seed | **A** | correct kind; wrong base (`Structural`, not `Seed`) → `class SeedMesh3D(Seed):` |
| `discovery_okuda/ops/chem_ops.py:98` | `CellRDSeed(Structural)` | Gray-Scott IC, once — the motivating case for the whole `seed` kind | seed | **A** | → `class CellRDSeed(Seed):` |
| `discovery_okuda/ops/ecm_ops.py:71` | `ECMSeed(Structural)` | fibre matrix layout, once | seed | **A** | → `Seed` |
| `discovery_okuda/ops/block_ops.py:60` | `BlockSeed(Structural)` | MPM slab fill, once (`self._done`) | seed | **A** | → `Seed`; drop the now-redundant `self._done` once engine enforces once-only |
| `discovery_okuda/ops/membrane_ops.py:124` | `BasementMembraneSeed(Structural)` | shell layout, once | seed | **A** | → `Seed` |
| `discovery_okuda/ops/membrane_ops.py:1450` | `AdhesionSeed(Structural)` | places + binds adhesions, once | seed | **A** | → `Seed` |
| `discovery_okuda/ops/integrin_ops.py:77` | `IntegrinSeed(Structural)` | fibre layout, once (`self._done`) | **structural** | **A**, mis-kinded | fix `kind="seed"` + base `Seed`; currently escapes the engine's seed gate entirely |
| `src/plexus/operators/segmentation_seed.py:87` | `SeedFromSegmentation(Exchange)` | seg → cells + particles, once (`self._done`) | **exchange** | **A**, mis-kinded | this is the exact example named in the brief §4; fix `kind="seed"` + base `Seed` |
| `src/plexus/operators/candidates/atlas_seed_state.py:27` | `SeedState(Operator)` | writes constant state blocks | seed | **A**, self-labelled harness | own docstring: delete once a real `seed:` section exists — candidate for removal in step 15, not conversion |
| `prototype/Turing_vertex/{vertex,vertex3d,coupled,turing}_ops.py` | `TissueSeed`, `TissueSeed3D`, `CoupledSeed2D`, `AggregateSeed` (all `Structural`) | cell placement + morphogen IC (+ kNN adjacency in `AggregateSeed`) | seed, family=`growth` | **A** | fix base → `Seed`; fix family → `seed` (or extend `OPERATOR_FAMILIES` if `growth` is meaningfully distinct — decide in step 10) |
| `prototype/embryo_vertex/{vesicle,sheet}_ops.py` | `VesicleSeed`, `SheetSeed` (`Structural`) | shell / monolayer placement | seed, `level=` (deprecated) | **A** | fix base → `Seed`; migrate `level=` → `set=` so `operators_at_set`/registry queries see them |
| `prototype/embryo_nca/embryo_nca_ops.py:125` | `NCASeed(FieldUpdate)` | single live cell in grid centre | seed, `level=` | **A** | the only *field*-target seed found — confirms Seed must address fields, not just sets; fix base → `Seed`, migrate `level=` |
| `prototype/active_matter2/am2_ops.py:266` | `SpiralSeed(Exchange)` | broken-front stamp on tick 1 | seed | **A** | fix base → `Seed` |
| `prototype/eye/blend_mpm_ops.py:458,593`; `capture_rest_ops.py:45` | `BlendGlobe`, `BlendMuscles`, `MusclesFromCapture` | mesh → material points, tissue labels, fibre architecture | seed, family=`anatomy` | **A** | **already correct base** (`Seed`) — the only 3 in the repo; family outside `OPERATOR_FAMILIES`, decide whether to fold into `seed` or extend the family set |
| `prototype/eye/eye_ops.py:104`, `muscle_ops.py:200`, `eye_ops.py:244` | `eye_anatomy`, `muscle_morphogenesis`, `muscle_insertion` | write the rest configuration once, at frame 0 (own comment says so) | **rewire** | **A**, mis-kinded | genuine x₀ hiding under `rewire`; convert to `Seed` |
| `src/plexus/engine.py:44-183` | `_spawn`/`_spawn3d`/`_spawn_pair3d` | initial positions/headings from `sets: spawn:` | *(not an operator)* | **A** | the largest single item: needs to become Seed semantics reachable from a `seed:` section, out of `build()` |
| `src/plexus/engine.py:237-299` | `_init_velocity` | physics-aware initial velocity (`circular_orbit` etc.); own docstring argues *against* being an operator | *(not an operator)* | **A** | same — and its docstring is a real design objection to converting it; see Open Questions |
| `src/plexus/engine.py:388-398` | `_start_centers` | explicit placement from `sets: start:` | *(not an operator)* | **A** | same |
| `src/plexus/models/entities.py:57` | `MPMParticle.provision` | allocates *and* fills per-particle buffers, incl. material-band assignment from `layers`/`core` | *(build hook)* | **A/B mixed** | needs splitting: sizing stays build-time (B), material assignment is Seed (A) |
| `src/plexus/models/base.py:283-299` | `Level.spawn` | wakes dormant slots, clones buffers | primitive | **B** | unchanged |
| `src/plexus/models/base.py:276-281`, `engine.py:515,594` | `free_slots`, `buffer:`, `grow_reserve:` | slot accounting | spec keys | **B** | unchanged |
| `discovery_okuda/round.py:1757`, `critic.py:906`, `translate.py:538` | `_replicate_seed`, `_seed_floors`, `SEED_SENTINEL` | RNG determinism / noise floors | n/a | **B**, homonym | "seed" = PRNG seed here, unrelated to x₀ — must be excluded from any name-based sweep |
| `src/plexus/operators/agent_scatter.py`, `mpm_scatter.py`, `mpm_gather.py` | P2G/G2P | per-tick | exchange | **C** | unchanged |
| `prototype/eye/tonic_ops.py`, `forced_gaze_ops.py` | `TonicActivation`, forced gaze | overwrite state every frame | lateral, `MAY_MUTATE_INTEGRATED_STATE=True` | **C** | unchanged; proof the boolean ≠ seed (step 12) |
| `src/plexus/operators/aggregate.py:30`, `candidates/jax_morph_virial_stress.py:156` | `Centroid`, `VirialStress` | computed from state | aggregate | **D** | unchanged |
| `src/plexus/models/registry.py:221` | `operators_at_level` | deprecated alias | — | **E** | unchanged |
| `src/plexus/models/catalog.py:33-181` | `Spawn`, `Divide`, `Die` stubs | never imported alongside real operators | various | **F** | dead menu, out of scope |
| `discovery_okuda/ops/ckpt.py` / `load_mesh_3d` | referenced 4x, defined nowhere | — | n/a | **F**, dangling | pre-existing bug independent of this refactor — flagging, not fixing, unless it blocks step 9 |
| `atlas_jax/record.py`, `atlas_cc3d/record.py`, `*/agents/atlas_agents.py` | hard-copied `KINDS` without `"seed"` | validation rule R6 | n/a | **F/E** | will reject `kind="seed"` operators if those subsystems ever use one — landmine, not this refactor's job to fix, but worth one line in SEED_MIGRATION.md |

## Where initialization lives outside operators (`sets:` config)

`start:` appears in 935 lines / ~360 live YAML files; `vel_init:` in 44
lines / 32 files. None of it is schema-validated beyond `buffer >= n`.
Three representative cases:

- `config/atlas_jax/brownian_dynamics.yaml` — `start: [[20.0, 20.0]]`
  sitting in the same `sets:` block as `n:`/`buffer:`, nothing
  distinguishing the allocation keys from the initial-condition key.
- `config/inverse_square/spiral_galaxy.yaml` — `spawn: disc` +
  `vel_init: {mode: circular_orbit, ...}`, with an explicit comment
  recording that the operator this replaced (`disk_ic`) was *deleted*.
  This is the strongest evidence the codebase has an active, stated
  position that initialization belongs in the spec, not in an operator —
  which the target architecture (§2/§5 of the brief) contradicts. This
  needs a decision, not just a mechanical fix (see Open Questions).
- `config/material/material_cardio_cells.yaml` — mixes both routes in one
  file (`tissue.start` *and* `seed_from_segmentation` in the schedule),
  with a comment noting the spec value is overwritten by the operator.

## `discovery_okuda` and the registry

Zero calls to `operators_of_kind`/`operators_by_family`/`catalog_summary`
anywhere in `discovery_okuda/`. Four places hard-code seed operator names
instead (`translate.py:539`, `run_one.py:1246-1252`, `biologist.py:686`,
`composition_space.py:339`), and `composition_space.py` maintains an
independent `role="driver"/"substrate"` taxonomy that already disagrees
with the registry's `kind`/`family` for the same operators. §13 of the
brief wants `seed_contracts()` queryable by the discovery system; §18
marks "discovery_okuda agent logic" a non-goal. **These are in tension**:
the four name-set call sites are exactly the code §13 wants replaced, and
also exactly what §18 says not to touch. Flagged for a decision, not
resolved here (see Open Questions).

## Open questions (need a decision before or during implementation)

1. **`_init_velocity`'s own docstring argues against this refactor.** It
   states "Initialization is a spec concern, not an operator" and cites a
   real prior migration (`disk_ic` operator → `vel_init` spec key) as
   precedent for keeping physics-aware ICs out of the operator system.
   Converting `_spawn`/`_init_velocity`/`_start_centers` into Seed
   operators (as §7/§8 ask) reverses that precedent. Options: (a) do it
   anyway, since the new `seed:` section is the "first-class initialization
   phase" this precedent didn't have available; (b) keep placement/velocity
   as spec-level `seed:` *declarations* (data, not operator code) that the
   engine's `seed()` phase interprets — closer to today's shape, still
   satisfies "seed is not in the dynamics schedule." Recommend (b) for
   `spawn`/`vel_init`/`start` specifically, and (a) for the genuinely
   procedural cases (`_build_edge_set`, child scatter, `provision`'s
   material-band assignment). This changes the shape of step 7's work
   substantially — worth confirming before implementing.
2. **`SeedState` is self-admittedly a stand-in for a missing `init:`
   section.** Once `seed:` exists, its own docstring says to delete it.
   Recommend deletion in step 15, not conversion.
3. **Family vs. kind for `growth`/`anatomy`-tagged seeds.** Four
   `Turing_vertex` operators use `family="growth"`, three `eye` operators
   use `family="anatomy"` — neither is in `OPERATOR_FAMILIES`. Either
   fold all seed operators onto `family="seed"` uniformly (simplest,
   matches the brief's "orthogonal" framing in §10) or extend
   `OPERATOR_FAMILIES` to keep the more specific tags. Recommend the
   former for the family axis and let `KIND=seed` alone carry the
   lifecycle meaning — but this is a judgment call, not derivable from
   the audit alone.
4. **`discovery_okuda`'s four hard-coded name sets** — in scope per §13,
   out of scope per §18. Recommend treating this narrowly: give
   `discovery_okuda` the `seed_operators()`/`is_seed()` registry query as
   §13 asks, but leave `composition_space.py`'s `role=` taxonomy and the
   agent-facing logic that consumes it untouched, since rewriting *that*
   is squarely the excluded "agent logic."
5. **`level=` migration.** Three seed operators (`vesicle`, `sheet`,
   `embryo_nca`) still register with the deprecated `level=` kwarg, which
   leaves `SET=None` and makes them invisible to `operators_at_set`/any
   set-scoped registry query. Fixing this is a one-line change per file
   but is a pre-existing deprecation issue this refactor would just be
   the first thing to actually depend on — worth doing alongside the
   `Seed` base-class fix since both touch the same lines.
6. **`atlas_jax`/`atlas_cc3d`'s stale `KINDS` copies (R6 validator).**
   Not touched by this refactor's stated scope, but will actively reject
   `kind="seed"` operators if either subsystem ever registers one. One
   sentence in SEED_MIGRATION.md flagging it seems right; fixing it is a
   separate, small, unrelated PR.

## Numeric summary

- Live `kind="seed"` registrations: 16 correct-kind (3 correctly based on
  `Seed`, 13 needing a base-class fix) + 2 genuine seeds mis-kinded
  (`structural`, `exchange`) = **18 operator classes to convert/fix**.
- Seed-shaped operators with no "seed" in the name: 3
  (`eye_anatomy`, `muscle_morphogenesis`, `muscle_insertion`), all
  `kind="rewire"` today.
- Build-time procedural initialization with no operator representation
  at all: 3 functions (`_spawn`/`_spawn3d`/`_spawn_pair3d`,
  `_init_velocity`, `_start_centers`) plus `_build_edge_set` and the
  child-scatter block in `build()`, plus half of `MPMParticle.provision`.
- Config-level initialization needing a schema decision: `start:` (935
  lines / ~360 files), `vel_init:` (44 lines / 32 files), plus every
  `sets:*:spawn` key.
- `discovery_okuda` call sites needing the new registry query: 4
  (`translate.py`, `run_one.py`, `biologist.py`, `composition_space.py`),
  plus one independent, disagreeing taxonomy (`composition_space.OPERATORS`
  `role=`) to reconcile or explicitly leave alone.

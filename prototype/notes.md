# Prototype notes — refinements from the demos/races build

Working log of changes made while building the three demos (sort / graze / aggregate)
and the two races (pillars / maze), and what they imply for the framework + plexus.tex.
Everything lives under `prototype/`; the engine stays generic (no per-scenario
branching) and all new behaviour is a registered operator or a declarative spec field.

## New operators (registry grew monotonically)

| op | kind | acts on | behaviour | requires |
|----|------|---------|-----------|----------|
| `graze` | exchange | cell ← field | consume a field, capped at local stock; accumulates `H.harvested` | `from`, `rate` |
| `finish` | lateral | cell | mark `done` + count `H.finished` + **freeze** particles at a line | `x` |
| `death` | lateral | cell | mark `done` + count `H.finished` + **remove** (mass→0, park off-domain) at a line | `x` |
| `adhesion` (+`cross`) | lateral | cell | added cross-type term: `cross<0` repels other types → de-mixing/sorting | — |

- `finish` and `death` are two flavours of an **exit/efflux boundary** for the discrete
  set: freeze-in-place vs remove-from-sim. `death` is the counterpart of a field
  `source` (an open sink for cells). Run it **last** in the schedule so the parked
  positions are what gets recorded.
- New selector attribute **`done`**: `cell[done=0]` scopes driving operators to cells
  that have not exited (so finished/dead cells stop being driven and drawn).

## New declarative spec fields (engine reads them generically)

- `world: <W>` — domain is the rectangle `[0,W]×[0,1]` with **square grid cells**
  (`nx = round(W·ny)`). `W>1` is a longitudinal world (the races use `world: 4.0`).
  Default `1.0` → identical to the old unit square. Touches `grid_field`, `mpm`,
  `engine._raster`, `_geodesic_potential`, and the render (equal aspect, undistorted).
- `navigation: geodesic | dynamic` (per field) —
  - `geodesic`: precompute a distance-to-source potential through open corridors (an
    **external** navigation gradient; the proven, no-band forage approach).
  - `dynamic`: the field evolves at runtime (a **self-generated** gradient: secrete /
    `graze` consume + diffuse). Default = geodesic when a `source` is given, else dynamic.
- `init: <v>` — uniform initial fill of a field (used by grazing / self-generated runs).
- **Circle obstacles**: an obstacle entry of length 3 `[cx,cy,r]` is a disc; length 4
  `[x0,y0,x1,y1]` is a rectangle. One mask, reused as MPM wall + field no-flux BC.

## Race scenarios (built like the food-gathering winner)

- `race_pillars` — ~760 round pillars (dense porous medium), `world:4`.
- `race_maze_hard` — a real perfect maze (recursive backtracker, `maze_gen.py`),
  `world:4`, square rooms, many genuine dead ends, open entry corridor on the left.
- Both: hidden **geodesic** `nav` field (not drawn) + a faint **shown** `scent` trail
  (winner aesthetic, no source/reservoir bands) + `motility`/`random_walk`/`sense`/
  **`boids`** driving + `death` at the exit + live escaped counter.
- `maze_gen.py` insets the maze into `x∈[x0,W]` and BFS-verifies left→right passability;
  pillar fields are rejection-sampled and BFS-verified too.
- Render: equal aspect (round pillars stay round), crisp **vector** wall/pillar patches
  (lightly rounded corners) instead of a pixelated mask, dimmed scent, no start box.

## Optimization (cluster, overnight)

- `race_opt.py <scenario>` mirrors `forage_loop.py`: kNN-surrogate **UCB**, no time
  limit, saves `<scenario>_winner_<k>.gif` whenever the escaped count beats the best by
  **≥10**, and rewrites `WINNERS_<scenario>.md`.
- Design vector = food-gathering levers **+ boids params** (12 total):
  `motility.speed/rot, sense.gain, mpm.drag, mpm.a_max, cell.youngs, particle.radius,
  cell.n, boids.cohesion/align/separate/radius`.

## Lessons (for the writeup / "honesty about regime")

1. **Consumption fights navigation.** In a maze, `graze`-consuming the attractant
   depletes the very gradient cells need (early race version: ~13/120). Trailblazing /
   secreting *reinforces* it; the cleanest is a hidden geodesic nav + a shown trail.
2. **Self-generated vs geodesic gradient.** Geodesic = perfect routing (no dead-end
   "fooling"); the Tweedy/Insall fooling needs the `dynamic` self-generated field. The
   easy-vs-hard contrast under geodesic is route length, not fooling — state which.
3. **Spawn clear of walls.** Inset the maze and spawn cells in an open entry corridor,
   else cells initialize inside walls and the MPM explodes.
4. **Rectangular MPM is fine** with square cells (`nx×ny`, `dx=1/ny`); per-axis grid
   index, boundary clamp, and wall BC all generalize cleanly.
5. **Render walls as vector patches**, not a rasterized mask — crisp corners.

## plexus.tex changes (done)

- Operator catalog: added `graze`, `finish`/`death`, and the `adhesion cross` term.
- New paragraph on the **exit/efflux boundary** (`death`) as the dual of a field source.
- Noted the `world` width (longitudinal domain), the `navigation` field mode
  (geodesic vs self-generated), `init`, and circle obstacles in the language table.
- Added the pillars + maze **race** scenarios and the `race_opt` optimizer (boids levers).

---

# Dividing cell — porting the standalone prototype into the framework

`divide_cell.py` was first written **standalone** (bespoke torch, no registry, no
spec, no engine) to validate that a *cardinality-changing* operator can be made
differentiable. This section logs bringing it inside the contract.

## Why `engine2` could not host it (the core finding)

`engine2` allocates `Nc`, `Np`, `parent` once in `build()` and never changes
them; the schedule accumulates into a fixed `H.cell_accel` and MPM moves a fixed
particle array. Nothing can change `|S_k|`. **Cell division and particle
duplication are cardinality-changing operators**, so they can't be a spec over
`engine2` as-is — the concrete reason the prototype lived outside the framework.

## The faithful port (what made it fit the contract)

- **Occupancy mass on a fixed buffer.** Allocate `buffer` slots per level; a
  per-node mass `w ∈ {0,1}` (0 = dormant) marks the active set. Duplication /
  division flip dormant↔active instead of resizing → constant tensor shape (and
  the same trick that keeps it differentiable; here `w` is boolean, the
  continuous-occupancy version of `divide_cell.py` uses `w ∈ [0,1]`).
- **Four registered operators**, real `@register_operator` on `base.py` classes,
  `forward(self, H, mask) -> {level: delta}`:
  - `cohere` (broadcast) — particle pulled toward its parent-cell centroid.
  - `repulse` (lateral) — soft short-range particle–particle repulsion.
  - `duplicate` (**structural**, new kind tag) — wake dormant particle slots.
  - `divide` (**structural**) — split a cell once its mass has doubled.
- **`integrate` builtin**: the schedule grammar already reserves it; `engine2`
  never implemented it. The growth engine implements it as overdamped
  `pos += dt·accel`. So the schedule reads
  `[aggregate, cohere, repulse, integrate, duplicate, divide]` — pure spec.
- Structural ops mutate `H` and return `{}` (no delta) — uniform with every other
  op, so the engine loop doesn't special-case them.

## What the framework still needs (gaps surfaced)

1. **`Level` needs first-class occupancy** (`Level.mass`) + a `buffer` field in
   the set spec, so cardinality change is engine-level, not per-op bookkeeping.
2. **`aggregate` must be mass-weighted** once `w` exists (centroid = Σwx/Σw).
3. **A `structural` operator kind** (changes membership, returns no delta) should
   be a recognised category alongside lateral/aggregate/broadcast/exchange — i.e.
   the Divide/Die row of the taxonomy, wired into the engine + validator.
4. **`engine2` and the growth engine should merge**: one generic engine that
   integrates deltas AND supports cardinality change, so MPM soft-cells can also
   divide.

## Differentiability (kept for the next step)

The continuous-occupancy `divide_cell.py` confirmed it: backprop of an outcome
(colony spread) to a physical parameter (cohesion) **matches finite differences
to <1%** straight through a division. Caveats learned:
- float32 **overflows** the gradient of a long stiff rollout (≈2.4×/tick) → use
  float64 for the sensitivity check, or a short horizon.
- the only non-differentiable residue is the **discrete which-daughter split**
  at the principal-axis plane (a side-assignment flip under perturbation) — the
  ~0.4% FD mismatch, exactly the discrete part the taxonomy flags.

## MPM version (`divide_cell_mpm.yaml` + `grow_engine_mpm.py`)

Second version where the soft cell is a disc of **MLS-MPM** particles (the
registered `mpm` Exchange operator does the elastic mechanics), grown + split the
same way. Learnings:

1. **MPM has no per-cell cohesion** — single-material MPM merges touching blobs
   on the shared grid, so 8 dividing cells would fuse into one body. Fix: keep a
   light **`cohere`** (per-particle pull to the cell centroid) for *identity*, and
   let MPM do the *mechanics*. Cohesion = who-belongs-to-whom; MPM = how it deforms.
2. **MPM only accepted a per-CELL `a_ext`** (`cell_accel[parent]`, uniform per
   cell). Cohesion is per-particle, so I added a tiny backward-compatible hook:
   `mpm` now also adds `H.part_accel` (per-particle) if present. One line; existing
   scenarios unchanged (defaults to 0).
3. **Dormant particles must carry `mass = 0`** so they don't scatter to the grid;
   then the full-buffer `mpm` op runs unchanged and dormant slots are inert. On
   `duplicate`, new slots get `mass = p_vol·ρ`, `F = I`, `C = 0`, `mu/la` from the
   parent — i.e. the structural op must initialise the *operator's* per-node state,
   not just position. (Suggests: operators should be able to declare how a new
   node of their level is initialised.)
4. Look: MPM cells are looser / jigglier than the cohesion-repulsion version
   (real elastic deformation vs. uniform packing). Tightening wants higher `cohere
   k`, more `substeps`, or lower `drag`.

Both engines (`grow_engine`, `grow_engine_mpm`) duplicate most of build/run — they
should fold into one generic engine that integrates deltas AND hosts MPM AND
supports cardinality change.

### Tissue adhesion (the missing force)

With only `cohere` + MPM, after the `divide` push there is nothing holding
daughters together, so the colony **disperses**. Added a `tissue` operator
(cell-level lateral): each active cell is pulled toward its active neighbours
within a radius. Now three forces balance into a packed tissue:

- **`tissue`** (inter-cell attraction) pulls cells together;
- **MPM** incompressibility + **`cohere`** (per-cell binding) resist overlap;
- net result: cells pack as a cohesive cluster instead of dispersing or merging.

Note `tissue` must respect `H.c_active` — inactive (dormant) cell slots have a
centroid of (0,0), so a naive all-cells adhesion would pull the colony toward the
origin. (The stock `adhesion` op isn't occupancy-aware, hence a new operator.)
This is the general lesson: **once sets have occupancy, every set-level operator
must mask on the active set.**

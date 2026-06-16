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

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

## Iterating toward realistic division (indexed specs_X + _X.gif)

Goal: round, realistic dividing cells, **only** via the framework (spec + registered
ops + engine). Each attempt is a named spec → same-named gif.

| # | spec | mechanics | result |
|---|------|-----------|--------|
| v1 | `divide_mpm_v1` | MPM elastic + cohere + tissue | **wedge/triangular** cells |
| v2 | `divide_mpm_v2` | + `tension` (surface tension), softer MPM | rounder but **loose/scattered**, still faceted |
| v3 | `divide_droplet_v3` | droplet: cohere + repulse (no MPM) | **clean round** uniform cells |
| v4 | `divide_droplet_v4` | droplet + `mitosis` (gradual) | **round + realistic division** ✅ |

**The core finding.** Elastic MPM is the *wrong* mechanic for round cells: its
fixed-corotated stress has **shape-memory**, so the flat cut left by a division
freezes into a wedge, and no rounding force fully overcomes it (v1, v2). A
**surface-tension droplet** — `cohere` (pull to centroid) balanced by `repulse`
(incompressibility) — has no shape memory and relaxes to a circle every time (v3).
A real cell is closer to a tense fluid droplet than an elastic solid, so this is
also the physically right call.

**Two operators added, both rational:**
- `tension` (cortical surface tension: boundary particles pulled inward → minimise
  perimeter → round). Tried on MPM (v2); the droplet's `cohere`+`repulse` already
  achieves the same more cleanly, so v3/v4 don't need it.
- `mitosis` (gradual division): on doubling, the cell **elongates** along its
  principal axis and forms a **cleavage furrow** (equatorial particles pulled in
  perpendicular) over `frames` ticks, then splits — replacing the instant `divide`
  relabel with visible, realistic mitosis. **v4 is the deliverable.**

Lesson for the taxonomy: a structural operator can be **gradual** — it carries a
per-cell phase and emits ordinary per-particle deltas during the transition, only
changing membership at completion. So "structural" and "returns a delta" aren't
exclusive; Divide/Die can have dynamics.

### v5–v6: tissue, then internal structure (nucleus + membrane)

| # | spec | change | result |
|---|------|--------|--------|
| v5 | `divide_droplet_v5` | + `tissue` adhesion (strength 1.5) | **regression**: tissue ≫ cohere, all cells merge into one mixed ball |
| v6 | `divide_droplet_v6` | particle roles + `nucleus` + membrane render | **nucleated cells with membranes** ✅ |

- **v5 lesson:** inter-cell `tissue` adhesion must be **≪** the per-cell binding
  (`cohere`), or cells lose identity and fuse. Dropped it; the morula already
  packs from global `repulse`.
- **v6 (better):** cells gain *internal structure*.
  - **Particle roles**: the `particle` set now has `types: {cyto, nucleus}` — the
    engine assigns the innermost fraction of each cell to `nucleus` (so it starts
    central), `duplicate` propagates the role to daughters, and `mitosis` splits
    the nucleus with the cell.
  - **`nucleus` operator** (broadcast): nuclear particles are pulled hard to their
    cell's nuclear centroid → a stiff compact core. `k` large ⇒ rigid.
  - **Membrane**: `tension` already defines the boundary; the render now draws it
    as a **ring** (edge) and the nucleus particles **dark** — each cell reads as a
    membrane-bounded cell with a nucleus, not a dot cloud.
  - General engine lesson: a set can carry **roles (sub-types) on a leaf level**,
    and every per-particle structural op (`duplicate`) must propagate them.

### v7–v8 + a structural correction (entities vs operators)

| # | spec | change | result |
|---|------|--------|--------|
| v7 | `divide_cell_v7` | ~4× particles, dots auto-shrink with density | dense, finely-grained cells |
| v8 | `divide_mpm_v8` | MPM back + membrane **made of particles** | structured cells (nucleus/cyto/membrane); rough at 8 |

**Two corrections to keep the modelling framework-clean (prompted by good pushback):**

1. **A nucleus is an *entity*, not an operator.** First version had a bespoke
   `nucleus` operator — that conflates *what it is* (a contained sub-set) with *the
   force* (cohesion). Fixed: `nucleus` is a **role** on the particle set (a
   sub-set of the cell, à la Fig 1 "cell = particles AND metabolites"), and its
   rigidity is the **generic `cohere` operator scoped to that role** (`role:
   nucleus`, large `k`). Deleted the `nucleus` operator. Same lesson would apply
   to any organelle.
2. **A membrane is also an entity** — and tempting to name a force after the actin
   *cortex*. Same trap. Modelled it generically: `membrane` is a **role**; its
   particles are linked into a **ring** (`ring`, a **rewire** operator that
   rebuilds the loop relation each tick as the cell grows/divides), and held taut
   by **`spring`** (the canonical **Lateral** operator along `edge_index`). So the
   membrane = entity (role) + relation (ring) + generic Lateral force — exercising
   the taxonomy's **rewire** + **lateral** primitives, no "cortex" operator.

General principle reinforced: **name operators after the relation/dynamics
(cohere, spring, rewire), never after the biological entity (nucleus, cortex).**
Entities are sets/roles + containment; operators are the generic dynamics on them.

### v9–v10: grounding division in real mitosis

- **v9** = the winners combined on the **droplet** mechanic (cleanest of all
  versions): cytoplasm + rigid nucleus role + a **membrane made of particles**
  (`ring` rewire + `spring` lateral) + gradual `mitosis`. Validated by montage to
  beat the MPM line (v1/v2/v8, which facet/scatter/blow-out) and match/exceed
  v6/v7 while using a *genuine* particle membrane.
- **v10** = v9 with the missing biology. Real animal-cell division is: mitotic
  rounding → **nucleus splits to two poles (anaphase)** → an **equatorial
  cleavage furrow** (contractile ring perpendicular to the spindle) pinches the
  cell in two → two daughters. My `mitosis` had elongation + furrow but **no
  nuclear separation**; added a `pole` term that drives nucleus-role particles to
  the two poles during the division phase. Result (frame ~19): the single cell
  shows two dark nuclei at opposite poles with the membrane ring pinching into a
  figure-8 between them — textbook anaphase/cytokinesis.

References used to ground the sequence:
[Phases of mitosis (Khan Academy)](https://www.khanacademy.org/science/ap-biology/cell-communication-and-cell-cycle/cell-cycle/a/phases-of-mitosis),
[Cleavage furrow formation & ingression (J Cell Sci)](https://journals.biologists.com/jcs/article/118/8/1549/28714/Cleavage-furrow-formation-and-ingression-during).

Framework note: all of this stayed pure spec + registered operators. `mitosis`
is now a faithful staged process (elongate + nuclear poles + furrow) yet still a
single registered structural operator that emits per-particle deltas and only
relabels membership at completion.

### v11–v25: render cleanup, 3-layer cells, and the MPM-vs-droplet split

Render: removed the drawn membrane circle; dots auto-shrink with density. New
generic operators (all named after dynamics, not entities): `shell` (radial
confinement), `skin` (rewire: re-tag inner=nucleus / outer=membrane by radius
each tick → **uniform membrane + nucleus by construction**, and re-applies
per-layer stiffness for MPM), `tissue` (inter-cell adhesion, weak), plus
`mitosis` gained mitotic **rounding** + anaphase **nuclear poles**.

Two regimes, each pushed to its best:

- **Track A — droplet (cohere + repulse) → division works.** `v18` = the polished
  result: a packed **morula** of 8 distinct 3-layer cells (rigid nucleus core,
  cytoplasm, uniform membrane skin via `skin`), weak `tissue` packing, mitotic
  rounding. Crisp; daughters stay distinct.
- **Track B — MPM unified body → single cell shines, division fuses.** `v21/v22/v24`
  = a single 3-layer MPM cell from the **layer-ball recipe** (elastic core, soft
  cytoplasm, stiff membrane skin): a superb dense **rigid nucleus core**, but the
  membrane edge stays diffuse (a growing MPM body has no surface tension) and any
  division (`v15`, `v23`) **re-fuses the daughters on the shared grid** even with
  a strong push.

**Validated conclusion (montages each step):** for *division*, the droplet is the
only mechanic that keeps cells distinct (A/`v18`); MPM is the right model for a
*single* cell's mechanics (B/`v24`, and the bounce/layer balls) but single-material
MPM cannot separate dividing daughters. Best dividing cell = `divide_cell_v18`;
best single-cell mechanics = `divide_mpm_v24`.

### v26–v30: MPM division solved with a `separate` operator

The MPM fusion was the whole blocker. Fix: a **`separate`** operator (cross-cell
particle repulsion) that holds a gap > the P2G/G2P stencil reach between cells, so
single-material MPM bodies can't merge on the shared grid. With
`separate` (anti-fusion) + `tissue` (pack) + `mitosis` (split) + `cohere`
(within-cell containment), **MPM cells now genuinely divide and pack** like v18:
`v28` = 4 distinct nucleated MPM cells in a 2×2 morula (vs `v15` = one fused
blob). The divide-then-pack dynamics are there (separate pushes daughters apart,
tissue draws them back).

Residual: a *growing* MPM body stays fuzzier at the edges than the droplet
(particles injected by `duplicate` into a deforming body don't immediately
conform; MPM has no strong surface tension). Tried slower growth, high drag,
surface `tension`, stiffer membrane (v29/v30) — improves density but the soft
fuzz persists. So: **MPM division works now** (the goal); droplet is still
crisper. `separate` is the key new primitive.

`cohere` now takes an optional `role` (cohere to the sub-set centroid); `ring`
(rewire) + `spring` (lateral) added; `nucleus` operator removed. MPM build assigns
roles by radius (nucleus innermost, membrane outermost). v8 is rough (MPM edge
blow-out, cells overlap without containment) — needs tuning, but the structured,
particle-membrane cell is proven inside the framework.

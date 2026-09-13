# One seeder for Plexus2: a plan

Written 2026-09-13 on branch `material-seed`, after surveying what a material spec can say today
(twelve scenes seeded and measured, never run) and reading the three seeders that already exist.
Nothing here is coded yet beyond the five placement words committed in 205fa686.

---

## 1. Three seeders, three virtues, no common language

**The cell atlas** (`operators/cell_ops.py`, `seed_cell_atlas`). The best design of the three,
and the one to generalise. It exploits the hierarchy: a `compartment` piece is a POSE (centre,
orientation, type) and its `mpm_particle` child is the matter of that piece, so "how many
mitochondria" is a row count and "what a mitochondrion looks like" is a shape sampler. A table
gives, per compartment, where the pieces go (`place: shell | ball`, radii as fractions of the
cell), which way they point (`axis: radial | random`) and what each one is (`shape: patch, sheet,
rod, ball, filament, cisterna, tubule, stack, mito, barrel`, each a closed-form sampler with a
closed-form volume). What it cannot do: it is hard-wired to ONE spherical cell, its table lives
in Python rather than in the spec, and nothing outside `cell` can use any of it.

**The material placement** (`models/entities.py`, `provision`). Volume-first and honest about it:
`V = per_parent * p_vol` fixes the size, the shape only arranges it. As of 205fa686 the words are
`shape: cube | ball | cylinder | obj`, `hollow`, `rotate`, `repeat`/`pitch`, `scatter`, plus the
axis-aligned `block` with `fill: random | lattice`, and radial structure through `layers`/`core`.
What it cannot do: no per-piece pose (a body is one blob about its parent's `start`), no surface
placement, no way to say "these bodies, arranged like that", and the shapes are a fixed list.

**The Blender importer** (`prototype/eye/blend_mpm_ops.py`, connectome-gnn `feat/oculomotor`).
The only path from real anatomy: a `.blend` is cut into named watertight parts by a `bpy`
subprocess into `parts.npz`, a similarity transform (measured, not assumed: the optic axis is
centre-to-cornea, and the left eye is the enantiomorph, so the transform reflects) carries head
coordinates into the simulation's frame, and points are placed by rejection against the
generalized winding number on a deterministic Hammersley sequence, with per-part material tags.
What it cannot do: it is prototype-local, its parts cache is beside the prototype, and a spec
cannot name a part of an imported model as the source of a body.

The common defect is not any one of them. It is that a scene's SHAPE, its ARRANGEMENT and its
MATTER are tangled differently in each, so nothing composes: the atlas cannot fill a bunny, the
material tab cannot lay twelve pieces on a shell, and the eye importer cannot be used by anything
but the eye.

---

## 2. The formalisation: a body is a source, a pose and a fill

One sentence to hold in mind. **A seeded body is a REGION, placed by a POSE, filled with MATTER
at a DENSITY.** Four independent questions, four independent vocabularies, each usable with any
of the others:

| question | key | values |
|---|---|---|
| what region? | `form:` | `cube`, `ball`, `cylinder`, `capsule`, `torus`, `sheet`, `filament`, `stack`, `mesh:<name>`, `image:<name>`, `field:<expr>` |
| where, and how many? | `arrange:` | `single` (default), `grid`, `scatter`, `shell`, `ring`, `spiral`, `from:<set>` |
| turned how? | `orient:` | `none`, `radial`, `random`, `along:<axis>`, `[rx,ry,rz]` |
| filled how? | `fill:` | `random`, `lattice`, `poisson`, `surface`, `hollow: f` |

and the matter stays where it already is (`material`, `youngs`/`bulk_modulus`, `density`,
`layers`, `core`, `eta`), because that part is not broken.

Two rules keep this from becoming a second language:

- **The volume contract is untouched.** `V = per_parent * p_vol` fixes a body's size whatever its
  form; `arrange` divides both the points and the volume among the copies. A `mesh` form is scaled
  so that ITS volume equals V, which is why a bunny and a cow of the same `particle_mass` weigh
  the same.
- **Everything acts on offsets from a body's centre**, so the four compose without special cases:
  a hollow torus, oriented radially, arranged on a shell of twelve, is four words.

What this buys, stated as scenes that are one type each: a pipe, a coil, a bed of grains, a wheel
of spokes, twelve mitochondria on a shell inside a cell, a bunny of snow, a lattice of 100 cubes
of which ten are water, a muscle strap taken from a scan.

---

## 3. `arrange:` is where the hierarchy pays

The atlas's real insight is that PLACING PIECES and FILLING A PIECE are different jobs at
different levels. Formalised, `arrange` writes the PARENT level's poses and the fill writes the
CHILD level's points:

```yaml
organelle:                     # the pieces: one row per mitochondrion, with a pose
  parent: cell
  types:
    mito: {count: 12, form: capsule, size: 0.12, arrange: {shell: 0.6}, orient: radial}
mito_points:                   # the matter of those pieces
  entity: mpm_particle
  parent: organelle
  body_of: mito
  per_parent: 2000
```

The same two lines describe a cell's organelles, a tissue's basement membrane patches, twelve
muscle straps around an eye, or 100 cubes in a box: `arrange: {grid: [5,4,5], pitch: 0.09}`.
`arrange: {from: cell}` is the case the atlas has today (one piece per parent, at its centroid);
`arrange: {shell: r}` is its Fibonacci spiral, which is worth keeping because a random draw on a
sphere clumps.

A body whose matter needs no piece level keeps writing `per_parent` on the particle set directly,
as material specs do now. Nothing existing has to move.

---

## 4. Shapes from files: one library, one loader, one cache

Today `shape: obj` reads `papers/morph_models/*.obj` (five meshes) and the eye reads a `.blend`
through a `bpy` subprocess into a prototype-local cache. Unify:

```
graphs_data/shapes/<name>/          one folder per shape, whatever it came from
    source.obj | source.blend | source.tif       what a person supplied
    parts.npz                                    watertight parts, in metres, part names
    parts.json                                   per-part volume, centroid, principal axes, tags
    LICENCE / PROVENANCE.md                      where it came from
```

`form: mesh:bunny` and `form: mesh:eye/lateral_rectus` then name a folder and optionally a part.
One loader (`plexus/shapes.py`) that:

1. resolves a name to a folder under `GNN_OUTPUT_ROOT/shapes` or the repo's own `shapes/`;
2. builds `parts.npz` when missing or older than the source, shelling out to `bpy` only for a
   `.blend` (the eye's step 1, promoted verbatim), to `trimesh`/`pyvista` for an `.obj`, and to a
   marching-cubes pass for an image stack;
3. caches the parts in memory per build, as `_obj_points` does now;
4. samples inside a part by rejection on a deterministic low-discrepancy sequence (the eye's
   Hammersley plus winding number, which is better than the current random rejection: no RNG, and
   uniform in volume, which is what MLS-MPM wants);
5. reports what it did in the spec's own terms: part, points, volume, scale factor.

The transform the eye applies (measure the object's own axes, map onto the simulation frame,
reflect for an enantiomorph) becomes `place: {frame: measured, mirror: x}` on the type, since it
is not eye-specific: any scanned body needs it.

**Migration.** `papers/morph_models/*.obj` moves to `graphs_data/shapes/<name>/source.obj` with a
PROVENANCE line each; `shape: obj, obj: bunny` keeps working as an alias of `form: mesh:bunny`
for one release. The eye prototype's `read_blend.py` becomes the `.blend` branch of the loader,
and `blend_mpm_ops` shrinks to the eye-specific part: which parts are muscles, which is the globe.

---

## 5. What it replaces, and what gets shorter

| today | after |
|---|---|
| `si_multimaterial_27`: 27 types, ~90 lines | one type, `arrange: {grid: [3,3,3], pitch: 0.08}` |
| the atlas table in `cell_ops.py` (200 lines of Python) | a table in a spec, one line per compartment |
| `blend_mpm_ops.py` (500 lines, eye-only) | a shape folder plus `form: mesh:eye/<part>` |
| `shape: obj` + `papers/morph_models` | `form: mesh:<name>` + `graphs_data/shapes` |
| a pipe, a coil, a grain bed | one type each |

---

## 6. Stages, each closed by something that runs

| stage | delivers | closed when |
|---|---|---|
| S0 | the five words of 205fa686 (`cylinder`, `hollow`, `rotate`, `repeat`, `scatter`) | done; twelve designed scenes seed and measure as declared |
| S1 | `plexus/shapes.py` and `graphs_data/shapes/`, with the five morph models moved and `shape: obj` aliased | `form: mesh:bunny` seeds the same points as `shape: obj` did, byte for byte under one seed |
| S2 | the `.blend` branch of the loader, from the eye's `read_blend.py` | `form: mesh:eye/globe` seeds the eye's globe with the same point count and volume as `blend_globe` |
| S3 | `arrange:` on a piece level (`single`, `grid`, `scatter`, `shell`, `ring`), the piece/matter pair in the spec | `si_multimaterial_27` rewritten as one type reproduces its archive's first frame within the seed's own tolerance |
| S4 | `form:` unified: the atlas's shapes (capsule, filament, stack, cisterna, tubule, barrel) become forms usable by any set | `seed_cell_atlas` re-expressed as a spec table; the atlas archive re-seeds to the same piece counts |
| S5 | `fill: poisson | surface`, `orient: along`, `place: {frame: measured, mirror}` | a scanned muscle seeded from a blend matches the prototype's rest length and volume |

S1 and S2 are independent of S3; S4 is the one that removes Python.

---

## 7. What this plan deliberately does NOT do

- It does not add an operator. Placement stays a property of a TYPE, read where the buffers are
  allocated, because that is the only place that knows the count, the mass and the parent.
- It does not touch the matter vocabulary (`material`, `youngs`, `layers`, `core`): those describe
  what a body is made of, and they are already right.
- It does not make the seeder a modelling tool. A shape that cannot be said in four words comes
  from a file, and the file is the artefact under `graphs_data/shapes` with its provenance.

---

## 8. One operator or three? Split by JOB, not by domain

Three seeders split by domain is what exists now, and it is why nothing composes: each grew its
own words for the same four questions, and a spec that mixes domains -- a tissue in a matrix with
a scanned bone -- would have to choose one. The split that holds is by WHEN the work must happen:

| job | where it lives | why there |
|---|---|---|
| region + fill of ONE body about its parent (`form`, `fill`, `hollow`, `rotate`) | `models/entities.py: provision`, not an operator | it must run at ALLOCATION time: only there are the count, the particle mass and the parent known, and the buffers are being written anyway |
| the POSES of many pieces (`arrange`: grid, scatter, shell, ring, from) | ONE new seed operator, `place`, at the piece level | it writes a level's rows, which is what a seed operator is for, and it is domain-free: organelles on a shell, 100 cubes in a box, twelve muscle straps |
| turning a file into parts and points (`mesh:<name>`) | a library, `plexus/shapes.py`, called by provision | it is a cache and a loader, not a step in the schedule |

So: one new operator, one extended allocator, one library. Anatomy does not need an operator of
its own -- it is a `form: mesh:eye/globe` source plus `place: {frame: measured, mirror: x}`.

WHEN A DOMAIN OPERATOR IS STILL RIGHT: when it carries knowledge with no geometric meaning. The
atlas's inventory (a cell has 77 mitochondria and 431 membrane patches) and the eye's part
naming (which of the twelve straps is the lateral rectus) are facts about biology, not about
placement. Those stay as thin named adapters -- a table feeding the general machinery -- and lose
their samplers, their transforms and their caches to the three rows above. `seed_cell_atlas`
becomes ~20 lines of table; `blend_globe` becomes a part list.

---

## 9. The scenes that drive the work

THE DEVELOPMENT IS THE SCENES, not the words. Each stage below is a sentence a person says, a
spec that says it, and a picture in the GUI that either shows it or does not. Nothing is called
done because a key parses: a scene is done when it is on screen at
`http://127.0.0.1:8799/material` (or `/bio`), seeded, looking like its sentence, and running
without the solver complaining.

HOW EACH ONE IS DRIVEN. One patch call to the page, which is what the GUI's own buttons send:

    curl -s -X POST localhost:8799/api/scene/patch -H 'Content-Type: application/json' \
         -d '{"form": {...}, "bodies": {"*": {...}, "0-9": {...}}}'

then LOOK: `/api/scene/shot?azim=&elev=&zoom=` writes the picture to a file, `/api/scene/counts`
says how many of what, `/api/scene/run` moves it. A scene that cannot be said in one patch is the
finding, and it names the word the seeder is missing.

### The ladder

| # | say it | needs | the picture that proves it | stage |
|---|---|---|---|---|
| A | "nine boxes on a lattice" | `repeat`/`n_bodies` | nine cubes, one colour each, evenly spaced | S0 (done) |
| B | "a hollow pipe standing on the floor" | `form: cylinder`, `hollow`, `axis` | a tube you can see through, upright | S0 (done) |
| C | "a slab tilted 18 degrees, a ball dropped on it" | `rotate` + a second body | the ball rolls down the ramp when run | S0 (done) |
| D | "a bed of 50 grains" | `scatter` | fifty balls strewn on the floor, none overlapping | S0 (done) |
| E | "a bunny of snow falling into water" | `form: mesh:bunny` + a liquid block | a recognisable bunny, not a blob | S1 |
| F | "the same bunny, ten times, in a row" | `mesh` + `arrange: grid` | ten bunnies, same size, evenly spaced | S1 + S3 |
| G | "a ring of twelve capsules about a centre" | `arrange: ring`, `orient: radial` | twelve capsules pointing outward, evenly spread | S3 |
| H | "a cell with twelve mitochondria on a shell and one nucleus" | the piece/matter pair, `arrange: shell` | the atlas picture, from a spec | S3 + S4 |
| I | "an eye's lateral rectus, from the scan" | `form: mesh:eye/lateral_rectus`, `place: {frame: measured}` | the strap in the right place, right length | S2 + S5 |
| J | "a tissue of 200 cells in a matrix of grains" | bio and material in ONE spec | the cyst sitting in a bed, both seeded | S3 |
| K | "a hollow torus of jelly, spun, dropped" | `form: torus` + `hollow` + `launch` | a doughnut that lands and wobbles | S4 |
| L | "100 boxes, 10 of water, 5000 points each" | `n_bodies` + a range key | the splash: the water goes first | S0 (done) |

A to D and L are on the board already; the ladder's value is E onward, where each rung is the
first scene that the previous vocabulary cannot say.

### What "closed" means for a rung

1. **It seeds.** `/api/scene/counts` reports the bodies and the points the sentence implies.
2. **It looks right.** A shot at two cameras, with the geometry measured, not eyeballed: a pipe's
   inner radius, a ring's twelve centres and their spacing, a bunny's volume against its mesh.
3. **It moves.** 100 frames through `/api/scene/run` with no NaN and no wall-clip warning; the
   thing falls, rolls or wobbles as the sentence says.
4. **It is one sentence away.** The spec's diff against the previous rung is the words the stage
   added, and nothing else -- if a rung needs a hand-written body list, the stage is not done.

### The order to build in

E and F first, because the shape library (S1) is the biggest gap and the bunny is the cheapest
proof. Then G and H, which are `arrange` (S3) and which retire the atlas's Python. Then I, the
scan, which is the only rung that needs Blender. J and K last, because they only compose what the
earlier rungs added.

---

## 10. A spec is read by a person: two rules on what the seeder writes

**No digit that says nothing.** A centre computed as the midpoint of two rounded corners came out
`0.33330000000000004`, and nine of those hide the three numbers that matter. Every float the page
writes is now rounded to six significant digits on the way out (`server._tidy`). The rule is about
the WRITER, not the reader: the engine is happy either way, the person is not.

**No list a word could have said.** Nine bodies on a lattice should not be twenty-seven numbers:

    # what the page writes today, because the form holds a body list
    cell: {n: 9, start: [[0.1667, 0.22, 0.1667], ... nine of them ...], types: {c00: ..., c08: ...}}

    # what it should write, once `arrange` lands (S3)
    cell:
      n: 9
      types:
        cube: {count: 9, form: cube, arrange: {grid: [3, 1, 3], pitch: 0.0833}, material: elastic}

Twenty-seven numbers and nine near-identical type blocks become one line, and the lattice is
STATED rather than sampled -- a reader learns "three by three, 83 mm apart", which is what the
scene is, instead of having to subtract coordinates to recover it. The same rule retires
`si_multimaterial_27`'s ninety lines and the GUI's own `_relay_bodies`, which exists only because
a spec could not say "and nine of them".

The test for both rules is the diff: a scene the page builds should be a spec a person would have
written by hand, and every number in it should be one they chose.

WHAT STOPS `arrange` REPLACING THE LIST TODAY: one colour per body. The 27-cube reference gives
every cube its own hue, which needs one type per cube because colour is a type property. So S3
also needs `color: cycle:<colormap>` (or `colors: [...]`) on a type whose copies should differ --
one word, and the last reason to write a body list disappears.

---

## 11. A population of scanned bodies: 160,000 neurons, not 160,000 lines

THE CASE. `graphs_data/neural_regions/<region>/` already holds what a connectome run fetched from
neuprint: `meshes/<bodyId>.obj` (1,002 of them for `hemibrain_cube_1000`, 740 MB), `skeletons/
<bodyId>.swc`, `morphology_index.json` mapping body id -> file for both, `neurons.npz` with the
soma positions and types, and `manifest.json` with the server, the dataset and the query. The
zebrafish regions carry the same layout at up to 32,768 neurons, and the full datasets run to
~160,000. A spec that names one body per line is not a spec, it is a database dump.

WHAT THE SEEDER NEEDS, and it is one word plus a selector, not a new subsystem:

    sets:
      neuron:                         # the pieces: one row per neuron, pose from the index
        entity: compartment
        source: {region: hemibrain_cube_1000, select: {type: "EPG*", limit: 2000}}
        arrange: from_source          # the poses ARE the somata: no lattice, no scatter
      neuron_points:                  # the matter: points inside each neuron's own mesh
        entity: mpm_particle
        parent: neuron
        per_parent: 500
        form: mesh:from_parent        # each piece fills ITS OWN file, named by the index

Three ideas, each already half-present:

1. **`source:`** -- a set's rows come from a FOLDER, not from a count. It reads
   `morphology_index.json` and `neurons.npz`, writes one row per selected body with its soma as
   the pose and its type as `node_type`, and records the manifest's provenance in the run. The
   selector is the only thing a person writes: a type glob, a body-id list, a region, a limit, a
   bounding box.
2. **`form: mesh:from_parent`** -- the mesh a piece fills is named by the piece's own row, not by
   the type. This is the generalisation of `mesh:<name>/<part>` from section 4: one file per row
   instead of one part per type. The loader caches per file and samples the same way.
3. **A budget, not a count.** At 160,000 neurons the interesting number is not points per neuron
   but points in total: `per_parent: {budget: 20_000_000, by: volume}` spends a total budget
   across the population in proportion to each body's volume, so a giant descending neuron gets
   more points than a small local one and the run fits in memory whatever the selection.

WHY THIS IS THE SAME MACHINERY AS THE EYE. The eye's `.blend` gives named parts of one animal;
neuprint gives one file per body of a population. Both are "a file is the shape", both need the
measured frame of section 4 (`place: {frame: measured}`; neuprint is in nanometres, the sim box is
not), and both want the deterministic Hammersley-plus-winding-number fill. The only new thing a
population needs is that the CHOICE of files be a query rather than a list.

WHAT IT UNLOCKS, as scenes for section 9's ladder:

| # | say it | needs |
|---|---|---|
| M | "500 hemibrain neurons as soft bodies, dropped" | `source` + `mesh:from_parent` |
| N | "every EPG in the cube, coloured by type" | the selector's type glob + `color: by:type` |
| O | "a zebrafish diencephalon region, 8,192 neurons, 20 M points" | the budget |
| P | "the same population, but as skeletons swollen to tubes" | `form: swc:from_parent` with a radius law |

P is worth stating because a 160,000-neuron scene will not hold 160,000 watertight meshes in
memory: the SWC skeleton (a tree of centre-line points with radii) is two orders of magnitude
smaller, and a neuron is well approximated by its skeleton swollen to its radius -- which is a
`form`, exactly like `cylinder`, evaluated per row. The mesh path is the reference; the skeleton
path is what makes the whole dataset runnable.

---

## 12. The payoff: a population that is seen AND simulated

WHAT THIS IS FOR. A neuron set whose rows carry a mesh each, a neural operator that computes their
activity from the connectome those same rows came from, and a picture that colours the meshes by
what the operator computed. Plexus already has the three pieces in three places: the morphology
(`graphs_data/neural_regions/<region>/`: `neurons.npz` with body ids, soma positions in
nanometres, types, neurite direction and span; `connectome.npz` with `edge_index` and `weights`;
`morphology_index.json` naming a skeleton and, for the hemibrain cube, a mesh per body), the
neural operators (`entity: neuron`, `synapse`, the neurons tab), and the renderer's per-row colour
(`color_field`). None of them can be said in ONE spec today, and that is the whole gap.

WHAT THE CONFIG WOULD LOOK LIKE. Nothing new in the language beyond section 11's `source:` and
`form: mesh:from_parent`, plus one word for the colour:

```yaml
general: {name: zf_dien_live, dim: 3, world: [1, 1, 1], n_frames: 2000, dt: 0.001,
          units: {length_um: 1000, time_s: 1}}

sets:
  neuron:                              # 1,017 rows, from the folder, not from a list
    entity: neuron
    source: {region: zf_Diencephalon_1000,       # graphs_data/neural_regions/<region>
             select: {type: "*", limit: 1000},
             pose: soma, frame: measured}        # nanometres -> the box, measured from bounds_*
    state: {v: {width: 1}, r: {width: 1}}
  synapse:                             # the edges of that same folder: 20,911 of them
    edge_set: true
    source: {region: zf_Diencephalon_1000, edges: connectome}
    maps: {pre: neuron, post: neuron}
  morphology:                          # the matter: points inside each neuron's own file
    entity: mpm_particle               # (or `points`, when nothing mechanical acts on them)
    parent: neuron
    per_parent: {budget: 5_000_000, by: span}    # spread over the population, not per row
    form: swc:from_parent              # each row fills ITS OWN skeleton, swollen to its radius
                                       # `mesh:from_parent` where a mesh exists (hemibrain cube)

operators:
- {op: neural_dynamics, at: neuron, over: synapse, model: leaky_rate, tau: 0.05, gain: 1.2}
- {op: stimulus, at: neuron, select: {type: ARTR_L}, kind: step, t: [0.2, 0.4], amp: 1.0}
- {op: paint_children, at: morphology, from: neuron, field: v}   # the row's v onto its points

schedule: [stimulus, neural_dynamics, paint_children]

plotting:
  renderer: vtk_points
  color_field: v                       # the picture IS the activity, on the morphology
  color_range: [-1, 1]
  curve: [{quantity: "mean:neuron:v"}, {quantity: "count:neuron"}]
```

Read it as three sentences: the rows come from a region, the matter of each row is its own
morphology, and the colour of that matter is the state the neural operator computes. The spec is
thirty lines whatever the population is -- 1,000 neurons or 160,000 -- because nothing in it
names a body.

THE ONE NEW OPERATOR, `paint_children`: write a parent's scalar onto its children's colour field.
It is four lines (`child.field = parent.field[child.parent]`), it is not neural at all, and it is
what makes ANY hierarchy visible through its matter -- a cell's chemistry on its organelles, a
body's stress on its particles.

WHAT IS MISSING TODAY, precisely, in case it reads as further off than it is:
1. `source:` on a set (section 11) -- the rows, the poses and the edges from a region folder.
2. `form: swc:from_parent` / `mesh:from_parent` (sections 4 and 11) -- the per-row file.
3. `paint_children` -- the four-line operator above.
4. `per_parent: {budget}` -- a total, not a per-row count.
Everything else (the neural operators, the edge sets, `color_field`, the renderer) exists.

HONEST NOTE ON WHAT I COULD SEE. The `neural_regions` folders are on disk and I read their arrays
to write this: `zf_Diencephalon_1000` has 1,017 neurons, 20,911 edges and 1,017 skeletons but no
meshes; `hemibrain_cube_1000` has 1,002 meshes (740 MB) as well. The zebrafish TRAINING code that
colours meshes by activity is not in this checkout -- `connectome-gnn` here sits on
`feat/symbolic-readout`, and neither it nor `cx/feat/oculomotor` contains a file that reads
`neural_regions` -- so the sketch above is built from the data and from Plexus's own vocabulary,
not from that script. Point me at it and I will reconcile the two.

### 12a. The neuprint spec already exists; the delta is three lines

`config/neural/hemibrain_cube_1000.yaml` is the link to neuprint, written and running: 1,002 real
somata placed by `neural_seed` from `region: hemibrain_cube_1000`, an edge set read from that
region's `connectome.npz`, four dynamical types, `neuron_update` + `neuron_signal` for the
activity, `voxelize` into a 128^3 field, and `renderer: vtk_points`. Its units are measured from
the manifest (the world box IS the 44.5 um cube). Nothing about the morphology demo needs a new
spec -- it needs three additions to that one:

```yaml
  morphology:                       # NEW: the matter of each neuron, from its own file
    entity: points                  # nothing mechanical acts on them; they are what is drawn
    parent: neuron
    per_parent: {budget: 4_000_000, by: span}      # a total, spread by neurite span
    form: mesh:from_parent          # meshes/<bodyId>.obj via morphology_index.json
                                    # `swc:from_parent` for a region that has only skeletons
operators:
  - {op: paint_children, at: morphology, from: neuron, field: voltage}   # NEW
plotting:
  color_field: voltage              # NEW: the picture is the activity, on the morphology
```

and the schedule gains `paint_children` after `neuron_signal`. Everything else in the file stays
as it is, which is the point: the connectome, the dynamics and the units are already right, and
the morphology is one more set hanging off `neuron`.

WHICH REGIONS CAN DO IT TODAY. `hemibrain_cube_1000` has 1,002 meshes (740 MB) and 1,002
skeletons; the zebrafish regions (`zf_Diencephalon_*`, `zf_Forebrain_*`, up to 32,768 neurons)
have skeletons only, so they need `swc:from_parent` -- which is the form that scales anyway.

---

## 13. The neuron seeder, generalised: a region is a SOURCE, not a special case

WHAT THE GUI NOW DOES (committed): the neurons tab has a `source` menu -- synthetic assemblies,
or a frozen neuprint region -- and a list of the 26 regions on this host with what each carries
(`hemibrain_cube_1000`: 1,002 neurons, 1,002 meshes, 1,002 skeletons, 20,259 edges; the zebrafish
trees: up to 32,768 neurons, skeletons only). Choosing one writes the spec that
`config/neural/hemibrain_cube_1000.yaml` was hand-written to be, and it seeds in 4.3 s.

WHY THAT IS STILL A SPECIAL CASE, AND MUST NOT STAY ONE. `neural_seed` takes `region:` and knows
the layout of `graphs_data/neural_regions/<name>`; the material path takes `block:`/`form:` and
knows nothing about files; the cell atlas takes a table in Python. Three readers, three
vocabularies -- the same fault section 1 names, one level up. The general form is ONE key on a
SET, whatever the entity:

    source: {from: <tree>, select: {...}, pose: <field>, frame: measured}

where `<tree>` is a folder with a manifest, and the manifest says what it holds: rows (a table of
bodies, cells, grains), poses (soma positions, centroids), edges (a connectome), morphology (one
file per row), fields. `neural_seed`'s `region:` becomes `source: {from: neural_regions/<name>}`,
the CX circuit becomes `source: {from: drosophila_cx/<run>}`, and a material scene that wants a
scanned population uses exactly the same words.

THE MANIFEST IS THE INTERFACE, and it already exists in three shapes on this host:

| tree | rows | poses | edges | morphology | what it is |
|---|---|---|---|---|---|
| `neural_regions/hemibrain_cube_1000` | `neurons.npz: body_id` | `xyz` (voxels, side in the manifest) | `connectome.npz` | `meshes/`, `skeletons/` | 1,002 neurons of hemibrain:v1.2.1 |
| `neural_regions/zf_Diencephalon_16384` | `neurons.npz: body_id` | `xyz_nm` | `connectome.npz` | `skeletons/` | 16,638 zebrafish neurons |
| `drosophila_cx/<run>` | `circuit_provenance.json: N, type_count` | -- | `net2_Wcon.npz: W2_con` | -- | a built circuit, 338 CX cells + 600 Net2 |

Two vintages of the same tree already disagree (`xyz` vs `xyz_nm`, bounds in voxels vs
nanometres, the side stated only in `manifest.json`), and the fix committed today reads both.
That is the argument for a DECLARED manifest rather than a reader that knows each vintage: a tree
says what it holds and in what units, and one loader serves every set that names it.

WHAT A SOURCE MUST ANSWER, and nothing more:
  rows      how many, and their ids           -> `n`, `node_type`, provenance
  poses     where each row is, in what units  -> `pos`, with the frame measured from the manifest
  types     what each row is                  -> `type_names` + `node_type`, from the table
  edges     who connects to whom              -> an edge set, `edge_index` + `weights`
  morphology one file per row                 -> `form: mesh:from_parent` / `swc:from_parent`
  fields    a recorded volume or image        -> a field's initial condition

A tree that answers three of the six is still a source; a set takes what it needs.

WHAT THIS RETIRES: `neural_seed`'s bespoke reader, the atlas's Python table (section 4), the
material tab's `_relay_bodies`, and every future "importer operator". What it keeps: one loader,
one manifest schema, and a `select:` that is the only thing a person writes.

### The CX figure, and why it is not a one-shot

`docs/drosophila.pdf` figure `fig_cx_circuit_variants.png` is a SCHEMATIC: twelve CX populations
(EPG, EPGt, PEG, ER6, PEN_a, PEN_b, Delta7, PFN_d, PFN_v, PFN_a, hDeltaB, PFR) drawn as boxes,
with the afferent drives and the readouts as arrows, the proposed extension dashed. Reproducing
it as a Plexus scene is the right test of the source vocabulary -- the populations are `types:`,
the arrows are the edge set's type-to-type weights, and the neurons tab's connectivity panel IS
that picture computed rather than drawn -- but it must come from a TREE, not from a script that
knows the CX.

WHAT IS MISSING ON THIS HOST: the CSV pair the loader reads
(`figures/drosophila_cx/drosophila_cx_connectome_338/{neurons,connections}.csv`) is not here; it
is fetched from neuprint by `fetch_cx_connectivity_pfn.py`. The built runs under
`graphs_data/drosophila_cx/` carry `circuit_provenance.json` (N 338, 13 types, the effective J's
sha256) and `net2_Wcon.npz` (the 600-cell Net2 matrix), but not the CX matrix itself. So the
honest order is: fetch or locate the CX tree, give it a manifest of the shape above, and then the
figure is a spec with thirteen types and one edge set -- no CX-specific code anywhere.

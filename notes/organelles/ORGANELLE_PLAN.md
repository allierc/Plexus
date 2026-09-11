# Organelles in Plexus: the plan

Written 2026-09-11 after the bio page's "add a nucleus to every cell" request produced 4,000
unplaced nuclei in one blob at the tissue centre. The refine pass could declare an organelle set
but nothing in the codebase places, keeps or counts organelles inside a vertex-model cell. This
note says what to build, in what order, and what already exists to build it from. Nothing here is
coded yet.

---

## 1. What exists

**Proteins** (`src/plexus/operators/protein_ops.py`, landed 2026-09-10). One contained set,
`protein`, parented to `cell`; each species is a `types:` entry with a `region:` on the cell
(apical cap, basal cap, mid-surface, interior). Three operators, one question each: `protein_seed`
(where they start), `protein_project` (how they stay on their cell, and how a division hands the
daughter half), `protein_express` (how many there are: birth rate `s` per unit area, lifetime
`tau`). A cluster is a point with a parent; it is not matter, so no MPM operator touches it.

**The organelle atlas** (`src/plexus/operators/cell_ops.py:55-267`, runs archived in
`graphs_data/cell/*C_mesh_and_mpm`, `adh_grow_org`). Entity `compartment` alias `organelle`: one
row is one PIECE (one mitochondrion, one Golgi cisterna) with a position, an orientation and a
type. Each compartment has a child set of `mpm_particle` rows (its material), placed by
`seed_cell_atlas` from a shape table (shell/ball placement; patch, sheet, rod, ball, filament,
cisterna, tubule, stack, mito, barrel shapes) around ONE cell of radius `cell_radius`. So an
organelle made of MPM particles already exists, but only for a single spherical cell, placed
relative to a centre and a radius, and never inside a vertex-model polyhedron.

**The gap between them.** The tissue side has cells as apico-basal prisms with faces and a
half-edge mesh; the atlas side has cells as a centre and a radius. An organelle set today can be
declared under a tissue cell (the refine did it) but no operator reads the cell's prism to place
anything in it, nothing keeps it inside when the cell moves or divides, and proteins can only be
parented to a cell.

---

## 2. The design: organelles are the twin of proteins, with a body

**One set, `organelle`, parented to `cell`, species as types.** Same shape as the protein set:

```yaml
organelle:
  parent: cell
  parent_pos: centroid
  per_parent: 60
  types:
    nucleus:      {body: point, n: 1,  radius: 0.30, region: interior, on_divide: duplicate}
    mitochondria: {body: point, n: 40, radius: 0.05, region: interior, on_divide: halve, s: 0.01, tau: 2000}
    golgi:        {body: point, n: 1,  radius: 0.12, region: apical_side, on_divide: duplicate}
```

`n` is the count per cell at seeding (proteins use `density` per unit area; organelles are
counted per cell, which is how a biologist states them). `region` is where inside the cell the
piece lives: `interior` (anywhere in the prism), `apical_side` / `basal_side` (the half of the
prism nearer that cap; the Golgi sits apical, the nucleus basal in a columnar cell). `on_divide`
is the rule at a cell division: `duplicate` (each daughter gets one: nucleus, centrosome),
`halve` (the mother's pieces are shared, nearest half to each daughter: mitochondria), `none`.
`s` and `tau` are the same birth-and-turnover law proteins use, for organelles that have a
biogenesis rate (mitochondria); a species without them keeps its count.

**A piece has a body, and the body is a choice per species.** The piece level is always the
same: one row per organelle with a position, an orientation and a radius. `body:` says whether
that row is ALL there is to the organelle, or whether it owns matter in a child set:

- `point` (default): the row is the organelle. Zero simulation cost; drawn as a sphere or an
  ellipsoid of its radius. The twin of a protein cluster, and the answer to "add a nucleus".
- `mesh`: the row owns a closed half-edge mesh in a child level (an icosphere: the nuclear
  envelope). Faces exist, so proteins can sit on them and a lamina tension can act on them.
- `mpm`: the row owns a cloud of `mpm_particle` rows in a child set, seeded by the atlas shape
  table (`cell_ops.py:98-116`). The organelle has mass and a deformation gradient: it resists
  compression, flows, and pushes on what it touches.

### 2a. How a species picks its body, and what that choice means

The user's question: how can one set be "a mesh or MPM"? It cannot, and it does not have to. The
`organelle` set is always the piece level. A body is a SECOND set in the spec, declared as a child
of `organelle` and naming which species it belongs to. Choosing the body is writing that child
set; `body:` on the species is only the word the operators dispatch on, and the schema refuses a
species whose `body:` has no matching child set (or the reverse). Three complete specs:

```yaml
# point: nothing but the piece level
organelle:
  parent: cell
  parent_pos: centroid
  per_parent: 41
  types:
    nucleus:      {body: point, n: 1,  radius: 0.30, region: basal_side, on_divide: duplicate}
    mitochondria: {body: point, n: 40, radius: 0.05, region: interior,   on_divide: halve}
```

```yaml
# mesh: the nucleus owns an icosphere; the mitochondria stay points
organelle:
  ...
  types:
    nucleus:      {body: mesh,  n: 1,  radius: 0.30, region: basal_side, on_divide: duplicate}
    mitochondria: {body: point, n: 40, radius: 0.05, region: interior,   on_divide: halve}
organelle_vertex:
  parent: organelle
  body_of: nucleus            # rows belong to pieces of this species only
  mesh: half_edge
  per_parent: 42              # icosphere subdivision 1: 42 vertices, 80 faces
  state: {pos, vel}
```

```yaml
# mpm: the nucleus owns a particle cloud; the shape comes from the atlas table
organelle:
  ...
  types:
    nucleus:      {body: mpm,   n: 1,  radius: 0.30, region: basal_side, on_divide: duplicate,
                   shape: ball, youngs: 1000, density: 1.0}
    mitochondria: {body: point, n: 40, radius: 0.05, region: interior,   on_divide: halve}
nucleus_node:
  entity: mpm_particle
  parent: organelle
  body_of: nucleus
  per_parent: 2000
```

What the choice changes is the DIRECTION of information between the piece and its body:

| body  | who moves whom | how it stays inside the cell | what it costs |
|-------|----------------|------------------------------|---------------|
| point | the piece is prescribed: `organelle_project` moves the row | projection onto the region (the corral, as for proteins) | nothing |
| mesh  | the body is simulated: the vertex operators move `organelle_vertex`; the piece's pose is MEASURED back as the centroid of its vertices (`aggregate_centroid`, `cell_ops.py:792`) | contact between the envelope and the cell's caps (`mesh_contact`, `contact_ops.py:87`), a force, not a projection | vertex step on 42 vertices per nucleus |
| mpm   | the body is simulated: the MPM step moves `nucleus_node`; the pose is the centroid of its particles | the cell's caps are a `mesh_contact` boundary for the cloud, and the cloud's pressure pushes the caps back | MPM step on 2,000 particles per nucleus |

So for `point` the piece drives the body (there is none); for `mesh` and `mpm` the body drives
the piece. `organelle_project` therefore does two different things by body: for `point` it
projects; for the other two it only refreshes the pose from the body and applies `on_divide` by
moving the body's rows with the piece (rigid translate at division, then the physics takes over).

**Which organelle gets which body.** The rule is what the question about the organelle needs:

| organelle | point | mesh | mpm |
|-----------|-------|------|-----|
| nucleus | count, position, division plane | envelope with faces: nuclear pores, lamina tension, envelope rupture | resistance to compression, nuclear deformation in a squeezed cell |
| mitochondria | count, turnover, distribution | no (no closed surface worth having at this scale) | bent capsules with cristae, fission/fusion as cloud split/merge |
| Golgi, ER | position only | no (fenestrated, not closed) | stacks, cisternae, tubule networks (the atlas shapes) |
| cytoskeleton | no | no | filaments (atlas `filament`), and only ever mpm |
| centrosome | count, position, spindle axis | no | no |

A real nucleus is BOTH: a mesh envelope with an mpm chromatin content inside it, coupled by
`mesh_contact` between the cloud and the envelope. That is two species in the spec (`envelope:
{body: mesh}`, `chromatin: {body: mpm, parent_region: inner of envelope}`), not a fourth body.
The plan keeps bodies to three words and composes the rest in the spec.

### 2b. What happens at a cell division

**How a division is seen.** `cell_divide` (`vertex_ops.py:1593`, Hertwig long-axis rule) splits
the mother face along a plane through its centroid, inserts the new vertices and edges, and
creates one newborn face; the mother keeps its face id, the newborn gets a fresh one, and both
have their `age` reset. `protein_project` already detects this from the mesh alone: a face that
is live now and was not last frame is the newborn, and the reset-age face nearest to it is the
mother (`protein_ops.py:357-380`). The organelle operators reuse that detection unchanged; the
division plane is recovered as the plane through the two daughters' centroids' midpoint, normal
to the line between them.

**Which daughter gets a piece.** The primary rule is geometric, not "nearest centroid": a piece
belongs to the daughter whose prism contains it (the point-in-prism test of section 3). This is
what the physics would say, and it makes the mother/newborn distinction irrelevant for pieces
that were already on one side of the plane. Only a piece the plane passes through, or a piece
that is transiently outside both prisms in the frame of the split, falls back to the nearest
centroid. Then `on_divide` decides what the count does:

| `on_divide` | point | mesh | mpm |
|-------------|-------|------|-----|
| `duplicate` (nucleus, centrosome, Golgi) | the mother's piece stays where the plane put it; a new piece is spawned in the other daughter at the same relative position (same region, mirrored through the plane) | the mother's envelope is KILLED and two icospheres are re-seeded, one per daughter, at the daughters' nuclear positions: open mitosis, where the envelope breaks down at prometaphase and reforms around each daughter's chromatin | the cloud is CUT by the division plane: particles on each side become the two daughters' pieces (chromatin segregation); each half is topped up to the species' `per_parent` by re-seeding inside its own half so the daughters' nuclei have the same mass as the mother's within one frame |
| `halve` (mitochondria, vesicles) | each piece goes to the daughter containing it; if the counts differ by more than one, the nearest pieces to the smaller daughter's centroid are re-assigned until the split is equal, as proteins do | not used | whole pieces go by the side of the plane their centroid is on; no cloud is cut |
| `none` | the daughter containing it keeps it; nothing is spawned | same | same |

`duplicate` therefore happens AT the division, not before it. The biological order (the nucleus
replicates in S phase, then mitosis, then cytokinesis) is a refinement for O5: a `phase` block on
the cell already exists in the cell-cycle operators, and a nucleus species can read it to swell
before the split and to migrate apically (interkinetic nuclear migration) so that the plane
`cell_divide` chooses passes through the nucleus. In O1 the nucleus reacts to the division; in O5
it helps decide it.

**Proteins on organelles at a division.** Two cases, both handled by the same halving rule
`protein_project` already applies to cell-parented species:

- the organelle is kept whole (`halve`, `none`, or `duplicate` for a `point` body): its proteins
  keep their parent index; nothing to do, since the parent row itself moved to the daughter.
- the organelle is split or re-seeded (`duplicate` for `mesh` and `mpm`, and the spawned twin for
  `point`): the mother organelle's proteins are halved by nearest-to-the-new-piece exactly as a
  cell's proteins are halved, then `protein_project` moves each half onto its own piece's
  `surface` or `inner` region. For a re-seeded envelope this is a projection onto new faces,
  which is what the region provider does every frame anyway.

**What the engine needs for this.** Spawning and killing pieces is what `spawn`/`kill` on a
contained set already do (dormant slots, `reserve_factor`). The new requirement is on the mesh
body: `organelle_vertex` must be able to kill a closed component and seed a new one at run time,
which means the half-edge level's edge occupancy (`eocc`) must extend to whole components; the
tissue mesh only ever adds faces. For the mpm body, cutting a cloud is a relabel of `parent` on
the particle rows plus a top-up seed, both of which exist. The counts the movie panel shows
(`count:organelle:nucleus`) then step by exactly one per division, which is the O1 acceptance
test: over a 100-division run, live nuclei equal live cells at every frame, and the mitochondria
total is conserved at every division to within the express law's birth and turnover.

**Where the body's rows go at seeding.** `organelle_body_seed` runs after `organelle_seed`: it
reads each piece's pose and radius and fills the child set for that piece. For `mesh` it writes an
icosphere scaled to the radius, its faces registered in the child level's half-edge mesh as one
more closed component (the tissue mesh is already one closed component; the level gains N of
them). For `mpm` it calls the atlas sampler for the species' `shape` with the piece as centre and
its orientation as `e3`, which is exactly what `seed_cell_atlas` does per cell today, called per
piece instead.

**Proteins can attach to organelles.** `protein.parent` may be `organelle` (nested containment:
protein in organelle in cell), and the species `region:` then names a place on the organelle:
`surface` or `inner`. What `surface` means depends on the body: the sphere of the piece's radius
for `point`, the faces of its mesh for `mesh`, the outer shell of its particle cloud for `mpm`
(particles whose radial rank within the piece is in the top fraction). `inner` is the rest.

---

## 3. What has to change in the code, by layer

**Regions become a small interface** (the one real refactor). Today `protein_ops._fans` builds
face fans from the tissue mesh and `_project` moves a point to the nearest point of its region.
Both assume the parent is a tissue cell. Replace them with a region provider looked up by
(parent entity, region name), each providing two functions: `sample(parent_ids, n)` (where a new
object goes) and `project(pos, parent_ids)` (the nearest allowed point). Providers:

| parent  | region                          | provider                                              |
|---------|---------------------------------|-------------------------------------------------------|
| cell    | apical, basal, mid              | the existing fans (unchanged)                         |
| cell    | interior, apical_side, basal_side | NEW: point-in-prism test on the fans; project clamps between the caps and inside the ring |
| organelle (point) | surface, inner        | sphere of the piece's radius                          |
| organelle (mesh)  | surface, inner        | the fans of the organelle mesh (same code as the cell caps) |
| organelle (mpm)   | surface, inner        | radial rank within the piece's cloud                  |

`protein_ops` then loops species over providers, and the organelle operators reuse the `cell`
providers. This is what makes organelles a twin rather than a copy.

**The point-in-prism test** is the piece the vertex model does not have yet. A cell is a fan of
triangles on each cap (centroid, source, target of each half-edge) and the prism between them.
A point is inside if, projected along the cell's apico-basal axis, it lands in one of the fan
triangles of the mid-cap (barycentric test), and its height along that axis lies between the
two caps at that in-plane point. Sampling is rejection sampling in the cell's bounding box, which
is cheap because cells are convex-ish prisms with a fill fraction near 0.5.

**Operators** (`src/plexus/operators/organelle_ops.py`, twin of `protein_ops.py`):

- `organelle_seed` (kind seed): per cell and species, `n` pieces in the species' region, with a
  minimum spacing of twice the radius (nuclei do not overlap mitochondria); writes the cell's
  per-species rates (`organelle_s`, `organelle_tau`, `n_organelle`).
- `organelle_project` (structural): keep each piece inside its region as the cell moves;
  apply `on_divide` when a cell divides (the newborn face and the reset-age mother face, exactly
  as `protein_project` finds them); retire orphans; for `mesh` and `mpm` bodies, move the body's
  rows with the piece (rigid translate at this stage).
- `organelle_express` (structural): birth and turnover for species with `s`/`tau`; same code
  path as `protein_express` on a different set, so it should be one function called twice.
- `organelle_body_seed` (kind seed): for `mesh` bodies, an icosphere per piece into
  `organelle_vertex`; for `mpm` bodies, the atlas shape into `<name>_node` (this is
  `seed_cell_atlas` called per piece with the piece's pose instead of per cell).

**Operators that let organelles act on the cell** (the twin of the planned `protein_exert`):
the nucleus position sets where the cell divides (interkinetic nuclear migration: the nucleus
moves apical before mitosis, and the division plane passes through it); the mitochondria count
scales the cell's growth rate. Both are one-line couplings once the sets exist, and both give a
measurable answer (division plane height, growth rate versus count), so each closes a stage.

**Engine.** Nested containment must resolve `parent_pos: centroid` two levels deep (protein in
organelle in cell); `_type_table` and `type_names` already carry per-species properties;
division bookkeeping for a contained set of a contained set (the organelle moves to the daughter,
its proteins move with it) is the one new engine rule, and it is the same rule at both levels.

**Entities.** Reuse `compartment`/`organelle` (`cell_ops.py:61`) with `reserve_factor` and the
same render as `protein_cluster`; add `organelle_vertex` for the mesh body.

**Measures.** `count:organelle:<species>` already works through `type_names`. Add
`organelle_height` (position along the cell's apico-basal axis, for nuclear migration) and
`organelle_overlap` (fraction of piece pairs closer than the sum of their radii; the seed's
spacing guarantee, and the invariant a regression test asserts).

**Bio page and Claude brief.** An Organelles table in the form (species, body, n per cell,
radius, region, on_divide); proteins' parent selector `cell | organelle:<species>`; hierarchy
shows organelle under cell and protein under whichever it names; `/api/bio/counts` already
reports per cell per species. Pieces draw as spheres of their radius, not dots.

---

## 4. Stages, each closed by a measured working point

| stage | delivers | closes when |
|-------|----------|-------------|
| O0 | point body: `organelle_seed` + `organelle_project` with the interior providers; spec `config/tissue/spheroid_organelles.yaml` (nucleus 1, mitochondria 40 per cell) | every cell has exactly its counts at frame 0 and at frame 800, overlap fraction 0, no piece outside its prism; registered as a working point |
| O1 | `on_divide` rules (section 2b) and `organelle_express` (shared with `protein_express`) | over a 100-division run, live nuclei equal live cells at every frame and the mitochondria total is conserved at each division; count curves in the movie panel |
| O2 | proteins parented to organelles, `point` body providers | nuclear-pore species on the nucleus surface, count and projection verified; proteins follow the nucleus through a division |
| O3 | `mesh` body: icosphere envelope per nucleus, proteins on its faces | the envelope stays closed and inside the cell over 800 frames; face-region proteins verified like the cell caps |
| O4 | `mpm` body: atlas shapes per piece inside vertex-model cells | a compressed cell shows nuclear resistance (nucleus strain smaller than cell strain by a measured ratio); the atlas archive runs re-registered on the new path |
| O5 | organelle to cell action: nuclear migration sets the division plane; mitochondria scale growth | division-plane height tracks nucleus height; growth rate versus count is the declared law |

O0 alone answers the request that started this; O2 is where organelles stop being decoration.
O3 and O4 are independent of each other and can go in either order: a species picks `mesh` or
`mpm` by declaring its child set (section 2a), and a spec that declares neither runs on O0.
Each stage adds operators and specs only (no scripts), captions its runs, and registers a
fingerprint before the next stage starts.

---

## 5. Decisions taken here, to be revisited only with a reason

- Counts per cell (`n`) for organelles, density per area for proteins: each stated the way the
  literature states it.
- One `organelle` set with species as types, not one set per organelle kind (the atlas archive's
  layout). Same argument as for proteins: one reservoir, one count path, one division rule.
- The body is a per-species word, so a spec never has to choose between "cheap" and "physical"
  for the whole cell.
- The region interface is built once and used by both families; the point-in-prism provider is
  the only geometry that does not exist today.

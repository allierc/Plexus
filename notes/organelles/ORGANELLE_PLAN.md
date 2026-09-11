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

**A piece has a body, and the body is a choice per species.** `body:` says what the piece is
made of, and only the body changes the cost:

- `point` (default): the piece is a pose (position, orientation, radius). Zero simulation cost;
  drawn as a sphere or an ellipsoid. This is the twin of a protein cluster and the answer to
  "add a nucleus to every cell".
- `mesh`: the piece owns a closed half-edge mesh (an icosphere per nucleus: the nuclear
  envelope). One extra level, `organelle_vertex`, with `mesh: half_edge` and `parent: organelle`,
  all pieces' meshes in one level as disconnected components. Faces exist, so proteins can sit on
  them (nuclear pore complexes, lamins). Cost: the vertex operators over the extra level.
- `mpm`: the piece owns a cloud of `mpm_particle` rows (`<name>_node`, `parent: organelle`), as
  in the atlas archive, seeded by the atlas shape table. The organelle deforms and pushes; a
  nucleus made of MPM particles resists the cell's compression. Cost: the MPM step over the cloud.

The three bodies share the piece level, so `count:organelle:nucleus` and the hierarchy panel read
the same way whichever body is chosen, and a spec upgrades a species from `point` to `mpm` by
changing one word.

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
| O1 | `on_divide` rules and `organelle_express` (shared with `protein_express`) | a dividing spheroid keeps one nucleus per cell and conserves mitochondria at division; count curves in the movie panel |
| O2 | proteins parented to organelles, `point` body providers | nuclear-pore species on the nucleus surface, count and projection verified; proteins follow the nucleus through a division |
| O3 | `mesh` body: icosphere envelope per nucleus, proteins on its faces | the envelope stays closed and inside the cell over 800 frames; face-region proteins verified like the cell caps |
| O4 | `mpm` body: atlas shapes per piece inside vertex-model cells | a compressed cell shows nuclear resistance (nucleus strain smaller than cell strain by a measured ratio); the atlas archive runs re-registered on the new path |
| O5 | organelle to cell action: nuclear migration sets the division plane; mitochondria scale growth | division-plane height tracks nucleus height; growth rate versus count is the declared law |

O0 alone answers the request that started this; O2 is where organelles stop being decoration.
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

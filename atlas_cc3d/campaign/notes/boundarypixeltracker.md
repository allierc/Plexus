<!-- boundarypixeltracker -- append below; the driver merges this into campaign/analysis.md -->

## boundarypixeltracker (BoundaryPixelTrackerPlugin) -- inspected

**What I read.** PyCoreSpecs.py:L5270 (the whole spec class + base `_PyCorePluginSpecs`),
the SWIG bindings in cpp/CompuCell.py (`BoundaryPixelTrackerData` with a single `pixel` field;
`BoundaryPixelTracker` with `pixelSet` + `pixelSetMap`; `BoundaryPixelTrackerAccessor`), the
PySteppables query path (`get_cell_boundary_pixel_list` / `CellBoundaryPixelList` in
iterators.py L759), and the C++ code-ref doc heading. The compiled physics is
`libCC3DBoundaryPixelTracker.so` -- not human-readable here.

**What it does to state.** Nothing to the simulated state. It is a per-cell INDEX: for each
cell it keeps the set of boundary sites (cell-owned sites with a differently-owned neighbour
within `neighbor_order`, medium counting as different), maintained incrementally as pixel
copies are accepted. No energy term, no field write, no lattice write. A run with it loaded is
bit-identical to one without; its whole purpose is to let Curvature / FocalPointPlasticity /
steppables read boundary sites cheaply. This is the pixel-count-set representation the ATLAS
brief flags: a cell is a set of sites, and this plugin is a derived set OVER that set.

**Surprises / gotchas for a reimplementer.** (1) It is a monitor, not an update -- the
Plexus-shaped temptation is to write it as read/write state, which makes it a phantom no-op.
(2) Boundary pixels != all pixels; PixelTracker is the "all sites" sibling, and conflating
them silently changes downstream queries. (3) `<NeighborOrder>` XML is emitted only when
order > 1 (default 1 implicit) -- emit-at-1 diverges from the reference CC3DML though it is
semantically equal. (4) The declared `neighbor_order` bounds which orders are queryable;
asking for an untracked order returns null and raises LookupError -- it is a "which shells to
maintain" declaration, not a query-time radius.

**What I could NOT establish (un-read compiled C++).** Whether the medium gets its own tracked
boundary set (no `track_medium` flag exists here, unlike PixelTracker -- suggests not, but
unverified); the exact per-copy neighbour re-test ordering; and how 2nd+-order sets in
`pixelSetMap` relate to the default-order `pixelSet`. The incremental-watcher model itself is
inferred from CC3D's CellGChangeWatcher architecture, not read line-by-line. No evidence run
exists for this mechanism (not among evidence.py's six), so all of the above is source-only.

## boundarypixeltracker -- normalized (NORMALIZER pass)

**Verdict: out_of_scope.** It maintains a per-cell acceleration cache (the set of a cell's own
lattice sites that touch a foreign owner / medium, out to `neighbor_order`) so downstream plugins
avoid rescanning every pixel. No energy delta, no lattice/field write, bit-identical simulation
without it -- a spatial index, the cellular-Potts analog of a neighbour-list / cell-list, not a
biological process. It exists only because a CC3D cell IS a set of sites; Plexus cells are points
with a derived radius and carry no per-cell pixel set, so the concept has nothing to bind to in
the promoted representation. Contract recorded descriptively (name `boundary_index`,
rewire/topology, set cell, writes `cell.boundary_pixels`) to satisfy the schema; the verdict, not
the contract fields, carries the judgment. `of` and `implementation_of` left null.

**Strongest argument against my verdict.** Plexus already promoted `radius_graph`
(rewire/topology, src/plexus/operators/graph.py), which ALSO maintains a pure index and ALSO
emits no integrable delta -- by the exact test I applied it too "looks like plumbing," yet it is
first-class. Consistency could therefore demand I treat BoundaryPixelTracker the same (alias of
`radius_graph`, or a new rewire/topology contract), and dismissing it as plumbing risks
special-pleading. Rebuttal: `radius_graph` builds the RELATION over the elements of a set
(who-interacts-with-whom), the substrate every lateral message-passing operator consumes;
BoundaryPixelTracker builds no relation among cells -- it classifies a single cell's sub-elements
(boundary vs interior pixel), a distinction that vanishes when a cell is a point. It is
representation-specific bookkeeping, not a topology contract. The tension is genuine, though: if
the algebra ever adopts a pixel-set cell representation, this stops being out_of_scope and becomes
a real rewire/topology candidate. I deliberately did NOT alias it to `radius_graph`, because
letting acceleration caches count toward the vocabulary would inflate the saturation measurement
this campaign exists to protect.

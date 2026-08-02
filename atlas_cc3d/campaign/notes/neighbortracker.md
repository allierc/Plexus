<!-- neighbortracker -- append below; the driver merges this into campaign/analysis.md -->

# NeighborTracker (NeighborTrackerPlugin)

**Read:** PyCoreSpecs.py:L2393 (spec, trivial — no params, emits only a bare
`<Plugin Name="NeighborTracker"/>` header); base `_PyCorePluginSpecs` (L471);
the compiled binding `cpp/CompuCell.py` for `NeighborSurfaceData` (L5625) and
`NeighborTracker` (L5655); and `core/iterators.py` `CellNeighborListFlex` /
`CellNeighborIteratorFlex` (L366–479), which is how Python actually consumes the
per-cell table. Doc: `.../plugins/NeighborTrackerPlugin.rst` (autodoc only).

**What it does to state:** nothing to the energy. It maintains, per cell, an ordered
set `cellNeighbors` of `NeighborSurfaceData{neighborAddress, commonSurfaceArea}`. It is
a listener updated incrementally on accepted pixel copies via
`incrementCommonSurfaceArea`/`decrementCommonSurfaceArea`, pruning a record when its
area reaches 0 (`OKToRemove`). It is a *tracker* — pure derived adjacency state that
other plugins/steppables read.

**Surprised me:**
- `commonSurfaceArea` is an integer boundary-LINK count (shared pixel edges in 2D /
  faces in 3D), not a real area — easy to get wrong as a float length.
- Medium is a first-class neighbour (null address → type 0). Aggregators must include it.
- No full rescan ever; the table is maintained purely by increment/decrement, so a
  bookkeeping error corrupts it silently with no self-healing recompute.

**Could NOT establish (honest gaps):**
- The *exact* increment/decrement rule at a boundary flip lives in the compiled C++
  (`NeighborTrackerPlugin.cpp`), which is not in this env — I inferred it from the
  swig method names and the Python iterator, not from reading the update arithmetic.
- Whether the two cells' stored copies are guaranteed symmetric (A_cb == A_bc) at all
  times or only after both boundary updates settle — I did not verify.
- Which downstream plugins/steppables treat NeighborTracker as a hard prerequisite;
  PySteppables exposes `get_cell_neighbor_data_list` (L1486) but I did not enumerate
  every consumer.

**Plexus framing:** does not fit read/write-a-field or return-a-delta. It emits neither
energy nor a canonical per-cell field update; it maintains an auxiliary *relation* over
the set of cells (adjacency keyed by shared-boundary count). Whether the algebra can
express "auxiliary derived relation over a set, maintained incrementally" is the open
question — flagged for the normalizer, not decided here.

**Verification pass (2026-08-02):** re-read the source to check the entry, not to
re-excavate. Confirmed: (a) class def is at PyCoreSpecs.py:L2393 — code_path correct,
line has not moved; (b) the null→type-0 "medium is a neighbour" claim is literal in
iterators.py at L410, L425, L440 (`cell_type = 0 if not neighbor else neighbor.type`),
in all three aggregators; (c) `commonSurfaceArea` is typed `int` in the iterator rtypes;
(d) `cellNeighbors` is a set queried via `.size()` (iterators.py:L384). Also confirmed
the accessor is `get_cell_neighbor_data_list` → `CellNeighborListFlex(...)` at
PySteppables.py:L1498 (the note's "L1486" is the docstring start; the call is L1498).
Checked the `.rst`: it is a bare autoclass stub (no members, no prose, no equation), so
the library docs add nothing over the compiled class — tightened `paper_section` to name
the exact source anchors and record that the doc is content-free. Unchanged open gap:
the increment/decrement arithmetic still lives in compiled C++ absent from this env.

---

**NORMALIZER (2026-08-02): verdict = refinement of `radius_graph`** (rewire/topology),
`implementation_of: radius_graph`, contract name `contact_graph`.

NeighborTracker maintains, per cell, the set of cells it currently touches plus each edge's
common surface area (integer shared-boundary-link count). That is the *relation among cells*
a rewire operator exists to leave on the set for lateral (contact/adhesion) terms to read —
so unlike its sibling BoundaryPixelTracker (ruled out_of_scope: it classifies one cell's own
pixels and builds no cell-to-cell relation), it does not collapse in the point-particle
representation. Not an alias: radius_graph's registered output is a bare unweighted
`edge_index` and cannot carry the contact-area weight, which is a first-class physical
quantity contact energy scales with. Hence the contract must WIDEN `outputs` from
`[edge_index]` to `[edge_index, edge_weight]` (breaking cost: current radius_graph consumers
assume weightless, purely metric edges; a promised edge weight either forces the distance-based
implementation to synthesise a contact area it has no basis for, or the weight becomes optional
and any lateral op that starts to require it silently fails on radius_graph). The edge criterion
(shared boundary vs Euclidean radius) and incremental-vs-full-rescan maintenance are
implementation, not contract — hence `implementation_of: radius_graph`.

**Strongest argument against (for `out_of_scope`):** the plugin has zero parameters, returns
no energy delta, and on its own changes no dynamics — a run is bit-identical with or without it,
exactly the profile that got BoundaryPixelTracker ruled out_of_scope. One could call it a pure
neighbour-list acceleration cache with no biological content, arguing the "contact graph" is
really owned by whichever contact-energy plugin consumes it. Rebuttal: BoundaryPixelTracker
builds no cell-to-cell relation, whereas this plugin's sole product *is* the tissue adjacency
graph with contact weights — the very substrate radius_graph was promoted to provide — so
treating it as plumbing would discard a real, reusable topology capability (weighted adjacency)
the language currently lacks.

<!-- connectivity -- append below; the driver merges this into campaign/analysis.md -->

## connectivity (ConnectivityPlugin, plain/legacy) — excavated

Read the whole class `PyCoreSpecs.py:L4059-4085` + base `_PyCorePluginSpecs` (L471-509), the
compiled bindings `cpp/CompuCell.py` (grep), and BOTH twedit CC3DML generators
(`CC3DMLGeneratorBase.py:614-632`, `CC3DXMLGenerator.py:656-677`) plus the `Connectivity*` tests.

- **Topological energy term, effectively HARD, like its ConnectivityGlobal sibling** ([[connectivityglobal]]).
  On a proposed pixel copy it penalizes flips that would break a cell's LOCAL connectivity; default
  penalty 10000000 >> T ⇒ Metropolis never accepts a fragmenting copy. Writes no state; only lowers
  acceptance probability. It reads TOPOLOGY (connected-component structure of a lattice site set),
  not a continuous property — no differentiable field exists to read.
- **Zero parameters in the Python spec.** `ConnectivityPlugin()` takes no args; `xml` emits a bare
  `<Plugin Name="Connectivity"/>`. But CC3DML supports one GLOBAL `<Penalty>` (default 1e7), shared
  by ALL cells — confirmed by two independent generators. So via the Python spec the strength is
  pinned at the compiled default and cannot be tuned. Granularity ladder across the family:
  Connectivity = one global scalar; ConnectivityGlobal = per-type; ConnectivityLocalFlex = per-cell
  ([[connectivitylocalflex]]).
- **RESOLVED the sibling note's open question:** the connectivityglobal note "did not confirm the
  folklore that the old Connectivity plugin is 2D-only." Both generators state it plainly:
  "works in 2D and on square lattice only!" AND "requires <NeighborOrder> 1 or 2." Now recorded as a
  surprise (a silent, un-validated applicability restriction), not folklore.
- **NOT SWIG-exposed:** `cc3d.cpp.CompuCell` has ConnectivityGlobalPlugin and
  ConnectivityLocalFlexPlugin but no plain ConnectivityPlugin — it's a compiled Potts plugin
  registered internally (`Potts3D.registerConnectivityConstraint`).

## NORMALIZED — verdict: new, implementation_of `stay_connected` (lateral/topology)

Verdict `new` against the frozen baseline (no promoted contract reads a body's connected-component
structure and vetoes moves that would split it), but `implementation_of: stay_connected` — the
contract the ConnectivityGlobal sibling ([[connectivityglobal]]) introduced in THIS atlas. The
ledger must count the connectivity family ONCE: Connectivity (one global scalar, this entry),
ConnectivityGlobal (per-type flood fill), ConnectivityLocalFlex (per-cell soft) are three
implementations of one contract. What THIS legacy implementation adds beyond the sibling: zero
tunable params (bare `<Plugin Name="Connectivity"/>`, strength pinned at 1e7), one GLOBAL scalar
for all cells (not per-type), 2D/square-lattice/NeighborOrder≤2 only, and a LOCAL check (never the
whole-cell flood fill). Same typed signature as the sibling — reads site-set topology, writes
nothing, kind lateral / family topology.

STRONGEST ARGUMENT AGAINST: this should be `alias` of the registered `cohesion`, not a `new`
contract — both "keep a body together," and a reviewer could argue connectivity is just cohesion
taken to its hard limit. Rebuttal (why I still chose new+implementation_of stay_connected):
cohesion is a metric attractive FORCE between point agents returning an integrable gradient delta;
stay_connected returns no force, is inert until a copy would fragment the SET, then vetoes — a
combinatorial topological rejection rule over an explicit lattice site-set the point-agent
representation does not even possess. Aliasing to cohesion would require cohesion to widen from
"pairwise force between positioned points" to "connected-components indicator over a site-set,"
which is a refinement that breaks every current cohesion user, not a free alias. The honest call is
that `stay_connected` is a genuinely missing contract and this is a second implementation of it.

**Could NOT establish:** the exact `changeEnergy` body — whether `f` is a boolean gate or scales with
the number of connected components introduced, its sign convention — is not readable (no SWIG wrapper,
C++ source absent from this install); reconstructed from the CC3DML comments + standard local-check
behavior, not verified byte-for-byte. **No paper text available**, so `paper_section` cites the Swat
chapter but anchors to in-source generator lines, not a page/eq I read. Not one of the six mechanisms
with reference ablations under `log/atlas_cc3d/`, so behavioural claims are unmeasured.

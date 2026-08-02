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
  Connectivity = one global scalar; ConnectivityGlobal = per-type; ConnectivityLocalFlex = per-cell.
- **RESOLVED the sibling note's open question:** the connectivityglobal note "did not confirm the
  folklore that the old Connectivity plugin is 2D-only." Both generators state it plainly:
  "works in 2D and on square lattice only!" AND "requires <NeighborOrder> 1 or 2." Now recorded as a
  surprise (a silent, un-validated applicability restriction), not folklore.
- **NOT SWIG-exposed:** `cc3d.cpp.CompuCell` has ConnectivityGlobalPlugin and
  ConnectivityLocalFlexPlugin but no plain ConnectivityPlugin — it's a compiled Potts plugin
  registered internally (`Potts3D.registerConnectivityConstraint`).

**Could NOT establish:** the exact `changeEnergy` body — whether `f` is a boolean gate or scales with
the number of connected components introduced, its sign convention — is not readable (no SWIG wrapper,
C++ source absent from this install); reconstructed from the CC3DML comments + standard local-check
behavior, not verified byte-for-byte. **No paper text available**, so `paper_section` cites the Swat
chapter but anchors to in-source generator lines, not a page/eq I read. Not one of the six mechanisms
with reference ablations under `log/atlas_cc3d/`, so behavioural claims are unmeasured.

<!-- connectivityglobal -- append below; the driver merges this into campaign/analysis.md -->

## connectivityglobal (ConnectivityGlobalPlugin) — excavated

Read `PyCoreSpecs.py:L4088-4186` (the whole class + its `xml`/`from_xml`/`cell_type_*` methods),
the compiled bindings `cpp/CompuCell.py:L5106-5178` (`ConnectivityGlobalData` + the plugin's method
list), the C++ doc stub `doc/.../plugins/ConnectivityGlobalPlugin.rst`, and the test
`connectivity_global_fast{,_python}` (`.xml` + steppable) for real usage.

- **It's a topological energy term, not an update, and effectively HARD.** On a proposed pixel copy
  it asks `checkIfCellIsFragmented` — would this copy split the target cell's site set into >1
  connected piece? If so it returns a positive penalty `S` (connectivityStrength); else 0. Large `S`
  ⇒ the Metropolis rule always rejects fragmenting copies. Unlike volume/surface (soft quadratics),
  this does *nothing* until a copy threatens to break the cell — then it vetoes. state_io writes
  nothing.
- **Biggest surprise: the penalty magnitude is unreachable from the Python spec.** The constructor
  is `ConnectivityGlobalPlugin(fast=False, *_cell_types)` — only a fast flag and an opt-in list of
  types (each → `<ConnectivityOn Type=.../>`). The strength `S` lives per-cell in the C++
  `ConnectivityGlobalData.connectivityStrength` (get/setConnectivityStrength) and is never exposed by
  PyCoreSpecs. I marked its role `UNKNOWN` — a magnitude nobody can tune from the spec.
- **Two algorithms, not identical.** Default "global" = whole-cell flood fill (exact, O(cell)); the
  `<FastAlgorithm/>` flag switches to `check_local_connectivity`/`changeEnergyFast`, a local
  approximation that can miss global fragmentations. Fast is a fidelity trade, not a free speedup.
- **Adjacency is external.** "Connected" is defined by the Potts `NeighborOrder`, which is a Potts
  param, not a plugin param — same plugin, different topology under a different neighbor order.
- **Three confusable siblings:** `Connectivity` (L4059, no params — legacy), `ConnectivityGlobal`
  (this), `ConnectivityLocalFlex` (soft local-energy variant, tunable per-cell strength). The naming
  is about the ALGORITHM, not a spatial region. Per-cell `cell.connectivityOn = True` from a
  steppable also enables it outside the type whitelist (test turns it on for `cell.id < 100`).

**Could NOT establish:** the compiled `changeEnergy` body was not read (cc3d not importable here), so
the exact returned value — fixed constant vs `S·(extra components)`, the sign, and the default `S` —
is reconstructed from method names + the emitted CC3DML, not verified byte-for-byte. Whether the
"global" and "fast" checks ever disagree on a real trajectory is asserted from the method split, not
measured. **No paper text available** — `paper_section` names the chapter's home for the connectivity
constraint but is not a page I read; no page/eq number invented. Also did not confirm the folklore
that the old `Connectivity` plugin is 2D-only while `ConnectivityGlobal` is the 3D-capable
replacement — left out of the entry rather than asserted. This mechanism is NOT one of the six with
reference ablations under `log/atlas_cc3d/`, so its behavioural claims are unmeasured.

## NORMALIZED — verdict: new (contract `stay_connected`, lateral/topology)

Verdict `new`: no registered contract, and no jax-morph proposal (adhere/agitate/apoptose/
mechanosense/morphogen/regulate/relax/reorient), expresses a TOPOLOGICAL constraint on a set — one
that reads a body's connected-component structure and vetoes moves that would split it. The 42/52
promoted contracts are metric forces, fields, count-changing structural ops, or index/rewire; none
keeps a body a single connected piece. Contract name `stay_connected` (biological content = cell
integrity: a cell is a single cohesive body and does not spontaneously fragment). It writes nothing
— a Potts energy veto (dE = S on fragmentation, else 0). A second finding: family `topology` only
ever pairs with kind `rewire` in the registry (those BUILD a relation); this is a topology-READING
energy term (kind lateral) that rewires nothing, so the (lateral, topology) slot it needs does not
yet exist.

STRONGEST ARGUMENT AGAINST: it is out_of_scope, collapsing to nothing in the promoted
representation exactly as boundary_index did — a Plexus cell is a POINT, and a point cannot
fragment, so there is literally nothing here to constrain; the constraint is an artifact of CC3D's
site-set representation. My rebuttal (why I still chose `new`): (1) out_of_scope is reserved for
mechanics with NO biological content and NO trajectory effect — boundary_index is bit-identical
with/without and writes only a cache, whereas connectivity changes reachable states and encodes cell
integrity, which is biological; (2) "points can't fragment" is Plexus's representation gap, not
proof the contract is vacuous — the MPM/deformable-body direction (mls_mpm_mechanics, mpm_strain,
cell_grow's woken material points) gives cells extended bodies that CAN tear, and that is precisely
where a connectivity/integrity term becomes the missing contract. If one weights the point-particle
representation as fixed, out_of_scope is defensible; I weight the biological content and the
already-in-flight deformable representation, so `new` is the honest measurement of a real gap.

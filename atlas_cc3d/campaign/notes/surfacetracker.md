<!-- surfacetracker -- append below; the driver merges this into campaign/analysis.md -->

# SurfaceTracker (order 28) — excavator note

**Read:** the spec `PyCoreSpecs.py:L5382` (three mutually-exclusive params, an
if/elif XML emitter, and per-param setters that null the siblings); base
`_PyCorePluginSpecs` (L471) + `_PyCoreSteerableInterface` (L453, gives `steer()`);
the maintained store `CellG.surface` in the compiled binding
(`cpp/CompuCell.py:L527` getter, **L589 read-only setter that raises**); and the
SWIG-exposed sibling `ClusterSurfaceTrackerPlugin` (L10270, a `CellGChangeWatcher`
with `field3DChange`, `getLatticeMultiplicativeFactors`, `getMaxNeighborIndex`,
`updateClusterSurface`). Cross-checked the consumer `SurfacePlugin` (energy, L2225)
which carries its own `neighbor_order`/`scale_surface`.

**What it does to state:** maintains each cell's scalar `surface` — a *weighted*
boundary-link count over a configurable neighbour shell — incrementally on every
accepted pixel copy. ΔH = 0; it is bookkeeping, the live perimeter the Surface
energy term penalises. Same third category as MomentOfInertia/NeighborTracker: a
passive `CellGChangeWatcher`, not an energy plugin and not a scheduled stepper.

**Surprised me:**
- `surface` is **weighted** by lattice multiplicative factors, not a raw face count,
  so it diverges from NeighborTracker's integer `commonSurfaceArea` and its scale is
  tied to the chosen neighbour order — `target_surface` only means something at the
  same order the tracker used.
- The three params are mutually exclusive (each setter nulls the other two) and the
  XML writes exactly one, priority `neighbor_order > max_neighbor_order >
  max_neighbor_distance`. Passing several is silently collapsed, not rejected.
- Unlike PixelTracker and ClusterSurfaceTracker, the plain SurfaceTracker is **not**
  a Python/SWIG class — no `getSurfaceTrackerPlugin`, nothing in `cpp/CompuCell.py`.
  It's a pure internal core plugin normally auto-loaded by the Surface energy plugin.
- `CellG.surface` is read-only from Python (L589 raises) — only the watcher mutates
  it; exposing it as writable Plexus state would be a category error.

**Could NOT establish (honest gaps):**
- The plain plugin's `field3DChange` arithmetic and its use of the lattice
  multiplicative factors are **inferred from the ClusterSurfaceTracker sibling's
  SWIG API and CC3D's BoundaryStrategy, not read** — the `SurfaceTrackerPlugin.cpp`
  is not in this env. High confidence on the family behaviour, but the exact
  increment/decrement ordering and whether `w` is applied identically for the
  non-cluster plugin are unverified.
- The numeric default neighbour order when all three params are `None` (core
  default) — not determined.
- No ablation/evidence run exists for this mechanism (not among the six with
  metrics.json), so all claims are source-only, no measured behaviour.

**Re-verification pass (order-coupling finding):** independently re-read every
anchor above at source — all hold (L5382 class def, L5407-5412 if/elif emitter,
L5450/5467/5484 sibling-nulling setters, L527/L589 getter/read-only-setter,
L10270 `ClusterSurfaceTrackerPlugin(CellGChangeWatcher)` with the four SWIG
methods). One detail worth promoting into the entry: `SurfacePlugin` (the energy
consumer, L2225) carries its OWN `neighbor_order`, and its `__init__` docstring
L2253 says "if SurfacePlugin is specified, its order overrides this." So the
tracker↔energy order consistency (surprise 2) is enforced by an OVERRIDE coupling,
not by the two configs happening to match — added as a surprise. The core default
neighbour order when all three params are `None` remains undetermined (compiled
core, unreadable here).

**Second excavation pass (header found — partly closes the .cpp gap):** the prior
pass said `SurfaceTrackerPlugin.cpp` "is not in this env" and inferred the arithmetic
from the ClusterSurfaceTracker sibling. It IS still absent, but I found the **installed
C++ header** `include/CompuCell3D/CompuCell3D/plugins/SurfaceTracker/SurfaceTrackerPlugin.h`
— which the sibling-only reconstruction didn't use. The header gives the real members
(`LatticeMultiplicativeFactors lmf`, `unsigned int maxNeighborIndex`, `BoundaryStrategy
*boundaryStrategy`, `Potts3D *potts`) and the `field3DChange(pt, newCell, oldCell)` decl
(L45), confirming this is the plain plugin's OWN `lmf` (not borrowed from the cluster
sibling as the old equations said). The stripped `.so` symbol table pins the neighbour
derivation: `getMaxNeighborIndexFromNeighborOrder` (both order params) vs
`getMaxNeighborIndexFromDepth` (distance), plus `getNeighborDirect`.

Upgraded the entry with the **exact four-branch `field3DChange`** (the two ELSE branches
— oldCell GAINS on a newly-exposed internal face, newCell LOSES on a face gone internal
— were missing from the old "old loses, new gains"), the **off-lattice guard**
(`nb.distance == 0` → skip), and a new top surprise on the **two-cells / double-count
trap** (a third-party neighbour's surface is invariant under the flip, so only the two
owner cells are written). Still honest that the four-branch structure is the canonical
CC3D form RECONSTRUCTED from header + `.so` symbols + cluster analogue, not transcribed;
`neighbor_order` vs `max_neighbor_order` operational difference and the None-default order
remain UNKNOWN.

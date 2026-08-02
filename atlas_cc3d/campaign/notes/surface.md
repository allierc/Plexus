<!-- surface -- append below; the driver merges this into campaign/analysis.md -->

# Surface (order 27) — excavation + normalization note

**What I read.** `PyCoreSpecs.py:2167-2320` — `SurfaceEnergyParameter` (per-type block: cell_type,
target_surface, lambda_surface; CC3DML `<SurfaceEnergyParameters CellType TargetSurface
LambdaSurface/>`) and `SurfacePlugin` (plugin-level `neighbor_order` + `scale_surface`, both guarded;
steerable). The physics is the compiled `SurfacePlugin.changeEnergy`, not read here. Also read the
measured reference ablation `log/atlas_cc3d/surface_constraint/metrics.json`.

**Mechanism.** A Potts ENERGY term, the standard partner of the Volume term: `E = lambda_S*(S -
S_t)^2` where `S` is the cell's boundary count — the number of id-change site-pairs out to
`neighbor_order`, a perimeter (2D) / surface (3D) COUNT that rises with both spread and membrane
roughness. Returns `dE` into the Metropolis test; writes nothing. `S` is maintained by a SEPARATE
plugin (SurfaceTracker) — SurfacePlugin only reads it.

**Measured evidence (reference run, not Plexus).** target 16, lambda_S 2.0, seed 42: mean perimeter
`20 -> 19.33` constrained vs `20 -> 22.8` unconstrained (lambda_S 0). Real but MILD — unconstraining
surface lets cells ruffle/spread, it does NOT dissolve them (Volume holds size). Contrast the Volume
ablation, which erased cells to medium. Surface shapes the membrane; Volume anchors the object.

**Surprised me.** (1) `cell.surface` reads a flat 0 without SurfaceTracker — an inert constraint that
looks like a valid measurement. (2) `S` is neighbor-order-dependent: the SAME cell has a different
surface at order 1 vs 2, so `target_surface` is only meaningful relative to the order it was tuned
at. (3) A pixel copy is a two-cell event and the quadratic makes the two boundary-count changes fail
to cancel — perimeter relaxation is coupled, not per-cell-independent.

**Could NOT establish.** The exact `scale_surface` normalization and the neighbor-order weighting
inside compiled `changeEnergy` (the FORM `lambda_S*(S-S_t)^2` is standard-CPM + source-declaration
confirmed; the constants are unread). No extracted paper text exists.

**Verdict (normalizer): `new` → contract `membrane_tension` (lateral/mechanics, set=cell).** No
promoted contract reads a cell's own boundary/perimeter count and returns a restoring energy — a
quadratic Hookean spring on the 1st-order geometric descriptor of a set, an effective membrane /
cortical tension, evaluated as a Potts ENERGY term (returns dE, writes nothing). Closest promoted is
`cell_grow`, rejected on three counts: it targets SIZE (0th moment, isotropic, an integrated state
update) not MEMBRANE AREA (boundary count at fixed volume, a Metropolis-gating energy); widening it
breaks its output contract and its biology. NOT a second sighting of `elongate` (proposed this phase
from LengthConstraint): perimeter is a different descriptor from the inertia-tensor major axis, so
distinct contract, `implementation_of: null`. NOT a jax-morph second sighting (`adhere` is a pairwise
interfacial energy, not a self-boundary spring; `relax`'s meaning is unread).

**Strongest argument AGAINST `new`.** Surface, Volume, and Length are the SAME functional object — a
quadratic Hookean spring `lambda*(moment − target)²` on a per-cell geometric moment, all returning dE
into the same Metropolis test — differing ONLY in which moment they read (0th = site count, 1st =
boundary count, 2nd = inertia axis). The language should arguably register ONE contract,
`constrain_moment(cell, order, target, lambda)`, of which Surface, Volume, and Length are
interchangeable IMPLEMENTATIONS — making `membrane_tension` `implementation_of` that, not `new`, and
turning three "new" contracts into one. Calling all three new risks exactly the yield inflation this
campaign exists to detect. I reject the lump because the moments carry different biology (size vs
membrane-area/tension vs elongation) with different homes in the language — size maps onto
`cell_grow`, the other two have none — so collapsing them would erase real distinctions rather than
reveal a shared one. But this is a genuine call, not a fact: if the campaign later registers a
`constrain_moment`-style contract when it normalizes Volume, then `membrane_tension` AND `elongate`
should both be revisited as implementations of it, and the ledger should count the trio once.

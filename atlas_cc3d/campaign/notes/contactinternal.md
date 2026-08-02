<!-- contactinternal -- append below; the driver merges this into campaign/analysis.md -->

# ContactInternal (ContactInternalPlugin) — excavation note

**What I read.** `PyCoreSpecs.py:L3350` — `ContactInternalPlugin` is a *one-line* subclass:
`ContactInternalPlugin(ContactLocalFlexPlugin)` -> `ContactLocalFlexPlugin(ContactPlugin)` ->
`ContactPlugin(_PyCorePluginSpecs, _PyCoreSteerableInterface)`. The subclass changes only
`name`/`registered_name`; ALL behavior and every tunable are inherited from `ContactPlugin`
(neighbor_order, depth, weight_energy_by_distance, and a list of `ContactEnergyParameter`
type-pair energies). Also read: `ContactEnergyParameter` (L3088), the InternalContact GUI
description (`InternalContactPlugin_descr.py`), and the compartment example deck
(`CompartmentExampleNewStyle.xml`, which runs `<Contact>` and `<ContactInternal>` together).

**What surprised me.** (1) It writes nothing — it is a Potts EnergyFunction; `changeEnergy()`
returns a scalar delta-E that only biases Metropolis acceptance. (2) The ENTIRE difference from
the ordinary Contact plugin is a same-cluster guard: the internal energy fires only between two
*different* cells sharing the same `clusterId` (compartments of one composite cell); it is
silently 0 across clusters and against Medium. (3) It is additive with Contact, not a
replacement — intra-cluster boundaries appear to be scored by both terms. (4) Only one triangle
of the symmetric J matrix is stored (`param_append` rejects the mirror pair). (5) `Depth`
(Euclidean radius) silently overrides `NeighborOrder` (integer shells).

**Source-vs-binary evidence.** The `.cpp` is not on disk (only `libCC3DContactInternal.so`).
I confirmed the same-cluster semantics three ways beyond the Python: the `.so` symbol table
(`internalEnergy(CellG*,CellG*)`, `changeEnergy(Point3D,CellG*,CellG*)`,
`setContactInternalEnergy`, `getMaxNeighborIndexFromDepth/NeighborOrder`,
`Potts3D::getCellFieldG`); the `.so`'s own embedded docstring — *"Handles internal adhesion
energy between members of the same cluster (i.e. between compartments)"*; and the reference
manual heading.

**What I could NOT establish.** (a) I did not read the C++ energy loop line-by-line, so the
exact delta form and the same-cluster/Medium guard ordering are inferred from symbols + docs,
not verified in C++. (b) I did NOT confirm that the ordinary Contact plugin ignores `clusterId`
(the basis for the "additive / double-counted intra-cluster" claim) — a reader should verify
Contact's `changeEnergy` before relying on that. (c) The exact `WeightEnergyByDistance` weight
(assumed 1/d per pair) is inherited-from-Contact and not read at source. (d) No paper text is
available; `paper_section` is anchored to the CC3D reference manual heading I actually read, not
a Swat et al. page. No oracle/ablation run exists for this mechanism (not among the six evidence
runs).

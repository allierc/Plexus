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

**Verdict (normalizer).** `new` against the frozen 42/52 baseline (nothing there is a Cellular-Potts
adhesion energy), with `implementation_of: adhere` — the pure-adhesion contract proposed by jax-morph
and already hosted in this atlas by AdhesionFlex; plain Contact is its canonical implementation and
ContactInternal is that same energy re-scoped by a clusterId guard to compartment-vs-compartment
boundaries. **Strongest argument AGAINST.** The clusterId gate is not obviously "just a domain filter":
sub-cellular compartment adhesion — holding the organelle-like domains of ONE composite cell together
while a separate matrix handles cell-cell contact — is arguably a distinct biological content
(intracellular structural cohesion) from cell-cell sorting, and CompuCell3D ships it as a SEPARATE
plugin run *additively* alongside Contact precisely because the two act on disjoint boundary sets with
independent J matrices. If one takes "adhere" to mean cell–cell adhesion specifically, then a
compartment-scoped adhesion energy is a genuinely new contract, not an implementation of it. I rejected
this because the operator's signature is identical (same output type, same set-of-sites representation,
same boundary-pair sum, same J-matrix read) and only the *predicate selecting which pairs count* differs
— a filter over the same reads, not new content — but it is the one place a reviewer could reasonably
land on `new`-without-implementation_of.

**Addendum (steerability).** Re-reading the chain at source: the immediate parent is
`ContactLocalFlexPlugin(ContactPlugin, _PyCoreSteerableInterface)` (L3339-3347), documented as
"A steerable version of ContactPlugin". So ContactInternal inherits STEERABILITY — its J_int
matrix can be rewritten at runtime by a Python steppable, not fixed once at setup. This does not
change the verdict (the reads/output/signature are unchanged; steerability is a property of when
the coefficients may change, not of the contract), but a reimplementer treating J_int as a static
per-run constant would be wrong. Recorded as a surprise. Confirmed from the class declaration, not
from exercising an actual steer.

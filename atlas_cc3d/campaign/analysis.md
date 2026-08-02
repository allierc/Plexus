

---

## adhesionflex

<!-- adhesionflex -- append below; the driver merges this into campaign/analysis.md -->

# AdhesionFlex (Plugin) — excavation note

**What I read.** `PyCoreSpecs.py:L3525` (`AdhesionFlexPlugin`) plus its helpers
`AdhesionFlexMoleculeDensity` (L3461) and `AdhesionFlexBindingFormula` (L3380); the compiled
interface `cpp/CompuCell.py` (`AdhesionFlexData.adhesionMoleculeDensityVec`, `changeEnergy`,
`adhesionFlexEnergyCustom`, the `set/getAdhesionMoleculeDensity*` family); the library's own
energy descriptor `twedit5/.../adhesion_descr.py`; and the shipped test
`tests/plugin_test_suite/AdhesionFlexPython_test_run/` (XML matrix + a steppable that mutates
densities per cell). The Python spec is a CC3DML emitter only; the physics is compiled.

**The mechanism.** A flexible Contact energy. Sum over neighbouring site pairs whose owners
differ: `E = Σ_ij [ -Σ_mn k_mn · AdhesionFunc(N_m(σ_i), N_n(σ_j)) ] · (1-δ)`. Each cell (and
Medium) carries a *vector* of adhesion-molecule densities `N_m`; `k_mn` is a user interaction
matrix; `AdhesionFunc` is a muParser formula string, default `min(Molecule1,Molecule2)`. Writes
nothing to the lattice — returns dE, a Potts energy term.

**What surprised me.**
- It is not just Contact-with-more-numbers: it introduces genuine **per-cell mutable state** (the
  density vector), seeded from a per-TYPE declaration but thereafter steerable per cell and
  inherited across mitosis (`assignNewAdhesionMoleculeDensityVector` deliberately skips the size
  check for daughter seeding). Plain Contact has no per-cell state.
- **Inverted sign** vs Contact: an explicit leading minus makes positive `k_mn` adhesive; the
  shipped example even uses negative k's.
- The combining kernel is **data** — an arbitrary muParser expression, not a fixed op.

**What I could NOT establish.**
- The (1-δ) restriction: the library HTML says δ is over "cell *types*", but the standard CPM
  contact term and `changeEnergy` are over cell **id** σ. I recorded both readings and flagged
  it; I could not run the compiled core to settle it (cc3d is deliberately not importable in the
  Plexus env), so which governs same-type neighbouring cells is still open.
- No ablation/oracle run exists for this mechanism yet (not among the six with `evidence.py`
  outputs), so every claim here is source-read, not measured.
- No specific Swat et al. (2012) page was read; `paper_section` points to the checkable CC3D
  reference-manual heading + the in-repo descriptor, not a page number I have not seen.


---

## blobinitializer

<!-- blobinitializer -- append below; the driver merges this into campaign/analysis.md -->

## blobinitializer (BlobInitializer) — excavated

Read `PyCoreSpecs.py:L5525-5736` (BlobInitializerRegion + BlobInitializer), `validate_point`
(L7457), the CC3DML generator (`CC3DMLGeneratorBase.py:1228-1250`), a commented example XML block
(`cellsort_2D.xml`), and how the oracle actually uses it (`oracle.py:95-97`).

- **It's a set CONSTRUCTOR, not an energy term.** This is the sharp contrast with everything else
  in this record: BlobInitializer runs ONCE at MCS 0 and genuinely WRITES the cell field — it
  paints a solid disk/sphere of freshly-created cells (each a new SET of lattice sites with a
  unique id) and assigns each a random type. State_io says so plainly: it creates the initial
  partition; downstream trackers (volume, center-of-mass) then maintain it. The energy-plugins
  write nothing; this writes the whole board.
- **The Python class does no painting.** `BlobInitializer`/`BlobInitializerRegion` are pure CC3DML
  serializers — Gap/Width/Radius/Center/Types emitters. The circle-clip, grid-tiling and random
  type draw all live in the compiled core (not importable here). Reading only PyCoreSpecs gives
  the parameters, never the algorithm — so `equations:` is a reconstruction from CC3D convention +
  the emitted fields, flagged as inferred.
- **A declared-validation vs working-use contradiction (surprised me most).** `validate()` bounds-
  checks `center ± radius` on ALL THREE axes; for the oracle's own 2D blob (dim_z=1, center.z=0,
  radius=dim//3) the `z − radius = −radius < 0` term trips `validate_point`'s `c_val < 0` guard and
  would raise "z-min". It works only because the XML-emission path (`.xml.getCC3DXMLElementString()`)
  never calls `validate()`. Recorded in `surprises:`.
- **Invalid-by-default sentinels:** constructor `width=0`/`radius=0` construct fine but `check_dict`
  rejects `<1` — only at validate() time. In `from_xml`, Gap is optional, Width/Radius required.

**Could NOT establish** (all compiled-C++, not readable in this env, and stated as such in the
entry): the exact clip predicate (site-in-sphere vs tile-center-in-sphere), the RNG draw mechanics
for random type assignment (asserted seed-dependent by CC3D convention, not verified), and how
overlapping multi-region blobs resolve painting order. **No paper text available** — `paper_section`
names the chapter's home for initializers plus checkable *library* anchors (PyCoreSpecs.rst:294/299,
the generator comment at L1235); I invented no page/figure. This mechanism is NOT one of the six
with reference runs, so there is no measured evidence — source-read only.


---

## boundarypixeltracker

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


---

## boxwatcher

<!-- boxwatcher -- append below; the driver merges this into campaign/analysis.md -->

## boxwatcher (BoxWatcherSteppable) — excavated

Read `PyCoreSpecs.py:L5490-5519` (the whole class + base `_PyCoreSteppableSpecs` at L512-552),
and the two CC3DML generators that actually emit the steppable's parameters
(`twedit5/.../CC3DMLGeneratorBase.py:L1282-1296`, `.../CC3DProject/CC3DXMLGenerator.py:L1064-1077`).
The compiled core is not readable here and there is NO BoxWatcher string anywhere under `cc3d/cpp`,
so the physics is reconstructed from the docstring, not verified.

- **It is neither an energy term nor a state update.** BoxWatcher traces the minimal bounding box
  of all non-medium cells, pads it by per-axis margins, and hands that box to the Potts solver so
  the pixel-copy sweep only samples sites inside it. It writes the SAMPLER's spatial support, not
  a cell, a field, or a dE. This is the first mechanism I've hit that falls outside the campaign's
  two expected buckets — worth flagging for the normalizer: it may have no Plexus operator analogue
  (nothing returns a delta the engine integrates).
- **The PyCoreSpecs wrapper is parameterless AND lossy.** `xml` emits only
  `<Steppable Type="BoxWatcher"/>`; `from_xml` locates the element then returns a bare `cls()`, so
  XMargin/YMargin/ZMargin present in loaded CC3DML are silently dropped on round-trip. Yet the
  twedit generators emit all three at 7. Genuine source-vs-source discrepancy: the two Python-facing
  paths disagree on whether margins are exposed at all.
- **"May have no effect for parallel version"** (docstring): the optimization is bypassed under the
  parallel sweep, so the realized effect is implementation/threading-dependent, not a model property.
- **The margin default 7 is a magic constant** that lives only in the twedit generators — a
  reimplementer following PyCoreSpecs inherits the (unknown) compiled default instead.

**Could NOT establish:** the exact clamp form at lattice edges, the recompute cadence (per MCS? per
N?), whether the box shrinks as well as grows, the C++ default margins when unset, and — most
important — whether a fixed-seed *serial* run is truly bit-identical with vs without BoxWatcher
(restricting the sampler's support changes the attempt/RNG sequence; I could not confirm it is
behavior-preserving). All of these sit in the compiled core, which is not importable in this
environment. No evidence run exists for this mechanism (not among the six ablated), and no paper
page was read — `paper_section` records that BoxWatcher is a computational optimization absent from
the Swat et al. text, with the generator docstring as the sole anchor.


---

## centerofmass

<!-- centerofmass -- append below; the driver merges this into campaign/analysis.md -->

## centerofmass (CenterOfMassPlugin) -- inspected

**What I read.** PyCoreSpecs.py:L3056 (the whole spec class + base `_PyCorePluginSpecs` at
L471 and its `generate_header`/`depends_on` at L305), the SWIG `CellG` struct in
cpp/CompuCell.py:L518-544 (`xCM/yCM/zCM`, `xCOM/yCOM/zCOM`, `xCOMPrev/yCOMPrev/zCOMPrev`) and
its read-only setters at L595-628, the sibling `MomentOfInertiaPlugin` at L5350, the
"CenterOfMassBased" ExternalPotential algorithm at L2939, and the code-ref doc stub
(PyCoreSpecs.rst:169). The compiled physics is in the un-shipped C++ core -- not readable here.

**What it does to state.** It publishes each cell's centre of mass and nothing else. It is a
CellGChangeWatcher (lattice monitor), NOT an energy term: it returns no delta-E and never tilts
the acceptance probability of a proposed pixel copy. Per accepted spin-flip it carries the
cell's first moment `xCM = sum of site coordinates` incrementally (+/- the changed site) and
stores `xCOM = xCM / volume`. `xCOMPrev` keeps a one-MCS history so consumers can form cell
displacement/velocity. A run with it loaded is bit-identical to one without; its whole purpose
is that COM-based chemotaxis, CenterOfMassBased external potential, mitosis split orientation,
and connectivity READ `xCOM`. In Plexus terms this smells like a REDUCE over a site-set that
publishes a derived field, not an operator returning a force/energy delta.

**Surprises / gotchas for a reimplementer.** (1) Monitor, not update -- forcing it into
read/write energy language makes a phantom no-op. (2) `xCM` (raw moment) and `xCOM`
(= moment/volume) are DIFFERENT CellG fields; dividing a stored sum by a stale volume is the
classic wrong COM; `xCOM` is derived + read-only from Python. (3) Periodic-boundary correction:
a cell straddling a wrap seam needs coordinates unwrapped against the BoundaryStrategy before
summing, then `xCOM` re-wrapped into the lattice -- the one thing most likely to be gotten
wrong. (4) Unguarded prerequisite: the spec makes CenterOfMass depend only on
`[PottsCore, CellTypePlugin]`, and NOTHING declares a `depends_on` CenterOfMass, yet COM-based
consumers silently need it -- omit it and spec validation still passes but COM is zero/stale.
(5) It does NOT compute the inertia tensor -- that is the separate MomentOfInertia plugin.

**What I could NOT establish (un-read compiled C++).** The exact periodic-boundary unwrap
arithmetic; whether `xCOMPrev` is refreshed by this plugin's per-MCS `step()` or elsewhere; and
the precise ordering of the gaining-vs-losing-cell moment updates within one accepted copy. The
incremental-watcher model is inferred from CC3D's CellGChangeWatcher architecture and the CellG
data members, not read line-by-line. No evidence run exists (not among evidence.py's six), so
all of the above is source-only -- no measured ablation backs it.


---

## connectivity

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


---

## connectivityglobal

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


---

## contactinternal

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


---

## contactlocalflex

<!-- contactlocalflex -- append below; the driver merges this into campaign/analysis.md -->

## ContactLocalFlex (order 11) -- read at source

**What I read.** `PyCoreSpecs.py:L3339` `ContactLocalFlexPlugin(ContactPlugin, _PyCoreSteerableInterface)`
-- a *bare* subclass of `ContactPlugin` (L3141): zero extra fields, same `__init__`
(`neighbor_order`, `depth`, `weight_energy_by_distance`, `*ContactEnergyParameter`), same `xml`.
The docstring is one line: "A steerable version of ContactPlugin." All the real difference is in
the compiled core (`cpp/CompuCell.py` L6379-6591): base class is `CellGChangeWatcher`, and it owns
a per-cell attribute `ContactLocalFlexDataContainer{contactDataContainer, localDefaultContactEnergies}`
plus methods `contactEnergy` / `defaultContactEnergy` / `setContactEnergy`.

**The finding.** The energy functional is *identical* to plain Contact:
`E = sum_{nbr pairs} J(sigma_i,sigma_j)(1 - delta(sigma_i,sigma_j))`. The ONLY difference is where
`J` lives: plain Contact reads a global cell-TYPE matrix; LocalFlex reads a PER-CELL container.
The spec's type-pair energies merely *seed* each cell's default table; `setContactEnergy()` then lets
a steppable override `J` for an individual cell (or cell-cell pair) at runtime -- that runtime,
per-cell steerability is the entire point of "local flex". `ContactInternal` (L3350) is the same
class again, just retargeted to compartments within a cluster cell.

**What surprised me.** The "flex" is completely invisible from the Python spec -- reading only
PyCoreSpecs you would conclude it is Contact with a different name. Nothing in the Python API sets a
per-cell `J`; the mechanism only exists once you drive `setContactEnergy` from C++/a steppable.

**What I did NOT establish.** (1) No paper text in this environment -- I could not verify a page/eq
number for the contact-energy term; the anchor is the source, not the paper. (2) I read data-structure
names and method signatures from the SWIG wrapper, not the C++ `changeEnergy` body -- the exact
per-cell lookup/fallback order (per-neighbour map vs `localDefaultContactEnergies` vs type default)
is inferred from the member names, not read line-by-line. (3) No ablation/evidence run exists for
LocalFlex specifically; `contact_adhesion` evidence covers plain Contact, so the measured effect of
per-cell overriding here is unquantified. (4) Whether `weight_energy_by_distance` divides by Euclidean
distance vs a tabulated weight is assumed from the name/parent, not confirmed in the core.

**Re-read addendum.** `_PyCoreSteerableInterface` in the subclass declaration is redundant --
`ContactPlugin` (L3141) already inherits it. So at the Python level the *only* thing that changes
between plain Contact and LocalFlex is `name`/`registered_name`; `registered_name="ContactLocalFlex"`
is the sole hook that binds the spec to the different C++ plugin holding the per-cell J container.


---

## curvature

<!-- curvature -- append below; the driver merges this into campaign/analysis.md -->

## curvature (EXCAVATOR)

Read the actual C++ (found on disk at `papers/CompuCell3D/.../plugins/Curvature/CurvaturePlugin.cpp`,
`.h`, `CurvatureTracker.h`), cross-checked against the installed binary `libCC3DCurvature.so`
(demangled symbols) and the shipped `Curvature_test_generate` XML. Web fetch/search were blocked,
so the on-disk source is the sole evidence — but it is the real thing, not a guess.

What it does: a **Cellular-Potts energy term** that penalizes bending of chains of compartmental
cells linked by junctions it maintains itself. Per triple of consecutive linked cells it adds
`lambda * kappa` where `kappa` is the **Menger curvature (1/circumradius)** of the three centers
of mass — zero when straight. `changeEnergy` returns the dE over affected triples (COMs recomputed
+/-1 volume); it writes no cell state. It ALSO grows a junction graph (activation_energy biases a
bond-forming move; the bond is committed in the `field3DChange` watcher on accepted moves).

Surprises worth the record:
- The function `calculateInverseCurvatureSquare` is a **misnomer** — it returns the plain curvature
  (2*sin(theta)/|chord| = 1/R_circ), neither inverted nor squared. Trust the name and you invert
  the physics.
- Half the plugin is **dead code**: `potentialFunction` (harmonic spring), `targetDistance`,
  `maxDistance`, and all three `diffEnergy*` (return 0). Only `lambda_curve` and `activation_energy`
  matter. Clearly cloned from FocalPointPlasticity and left half-gutted.
- Junctions are the plugin's OWN state (within-cluster only), NOT FPP's — the two are separate
  plugins that merely co-occur in the demo. So Curvature is a **hybrid**: energy term + stateful
  bond-graph watcher. This is the interesting bit for the algebra: `changeEnergy` fits "return a
  delta the engine integrates," but `field3DChange` is a genuine write-on-accept side effect that
  the pure-energy framing does not cover.
- Apparent BUG: the volume-1->0 branch adds three curvature terms WITHOUT the `lambda` factor
  (L674/L682/L687). A faithful port must reproduce it to match the oracle.

Could NOT establish: no oracle/ablation run exists for curvature under `log/atlas_cc3d/` (not one
of the six evidenced mechanisms), so the dynamical magnitudes (does lambda=1000 in the demo
actually straighten the chain, how strong is the activation-energy bias vs Temperature=10) are
unmeasured here — inferred from the formula only. I did not confirm the exact set of "affected
triples" is complete for every geometry; I read the five triple-blocks in each branch but did not
prove they exhaust all triples touched by a COM shift. Left `verdict`/`contract` unset (normalizer's
call); set `status: inspected`.


---

## diffusionsolverfe

<!-- diffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# DiffusionSolverFE (order 14)

**What I read.** `PyCoreSpecs.py:L6242` (`DiffusionSolverFE`) plus its child specs:
`DiffusionSolverFEDiffusionData` (L6070), `...SecretionData` (L6160), and the shared bases
`_PDEDiffusionDataSpecs` (L554), `SecretionParameters` (L597), `PDEBoundaryConditions` (L820),
`_PDESolverFieldSpecs` (L1057), `_PDESolverSpecs` (L1150). The class at L6242 is a thin CC3DML
*emitter* — all physical parameters live on the child DiffusionData/SecretionData/BoundaryConditions
specs, and the integrator is compiled. I grounded the algorithm on the in-tree guide strings:
`diffusion_solvers_descr.py:7` ("Uses Forward Euler method and handles moving boundary conditions")
and `CC3DXMLGenerator.py:884` (FTCS stability limits D>0.16 3D / 0.25 2D at DeltaX=DeltaT=1).

**The mechanism.** One explicit forward-Euler step of `dc/dt = D grad^2 c - lambda c + S` on the
CPM pixel lattice, per MCS. A real field->field WRITE (overwrites c in place) — unlike the Potts
energy-term plugins in this atlas that return nothing and only bias pixel-copy acceptance.

**What surprised me.**
- D and lambda are *cell-type-indexed and spatially heterogeneous* — read from the moving CPM cell
  field at each pixel. Plexus `diffuse`/`decay` carry a single global scalar rate; no set coupling.
- "Secretion" is three semantics under one name: additive rate, ConstantConcentration (a Dirichlet
  *clamp*, i.e. a set not an add), and SecretionOnContact.
- Default BC is Value 0.0 on every face = absorbing sink, not no-flux — silent field leakage.
- FTCS is only conditionally stable, yet the DiffusionSolverFE python spec exposes NO setter for
  DeltaX/DeltaT/ExtraTimesPerMCS — high D blows up silently with no knob at this API layer.

**What I could NOT establish (do not treat as known):**
- The exact neighbour stencil (I assumed von Neumann 4/6) and how D is combined across a cell-type
  interface (destination-pixel vs average) — both are in the compiled core; I did not read the `.so`.
- Whether decay/secretion are applied in the same sub-step as diffusion or in a fixed operator-split
  order; the "one step" form in `equations:` is the standard FTCS reading, not verified against core.
- Whether ExtraTimesPerMCS sub-cycling is auto-chosen by the core or must be user-set — the guide
  implies user-set, but the actual default behaviour is compiled.

**UPDATE — resolved from the OpenCL kernel.** After the notes above, I read the shipped GPU
kernel `cc3d/cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (`uniDiff` L821-L1057, plus the
`secrete*` kernels L192-L346), whose comments cross-reference the CPU code. This upgrades three of
the "could not establish" items:
- **Interface diffusion IS resolved.** The stencil is NOT a single central-D Laplacian. It is two
  half-sums: `isoSum = D_i*(SUM c_j - N c_i)/2` (centre cell's D) plus `varSum = SUM D_j*(c_j-c_i)/2`
  (each *neighbour's own* D), added. This symmetrises the flux across a cell-type interface and only
  collapses to `D*(SUM - N c)` for uniform D. I put the exact form in `equations:`.
- **Operator split IS resolved.** Decay is inside the diffusion pass as a `(1 - dt*lambda_i)*c_i`
  factor; secretion is a *separate* sweep (order-dependent). ConstantConcentration is a hard
  re-pin `c := value` each step; plain Secretion also supports a relative/max uptake sink.
- **Still open:** whether the compiled CPU default path (`gpu=False`) is byte-identical to this
  kernel — I could not read the `.so`; the kernel comments claim equivalence but I did not verify.
  ExtraTimesPerMCS default behaviour is still compiled and unread.

**Adjacent language (for the normalizer, not a verdict):** Plexus already has `diffuse`
(finite_difference + spectral), `decay`, `deposit`, `prescribed_field`, `scalar_field`. CC3D fuses
diffusion+decay+source+BC+cell-type-coupling+mass-compensation into ONE solver; the open question is
whether that fusion and its Potts-field coupling are expressible by composing the existing operators.


---

## focalpointplasticity

<!-- focalpointplasticity -- append below; the driver merges this into campaign/analysis.md -->

## FocalPointPlasticity (order 16) — excavated 2026-08-02

Read: `PyCoreSpecs.py` L4512-4908 — the Python spec layer (`LinkConstituentLaw`,
`FocalPointPlasticityParameters`, `FocalPointPlasticityPlugin`), the twedit ML generator
(`CC3DMLGeneratorBase.py` L859-915), and the shipped test XML
(`tests/.../FocalPointPlasticity.xml`). The compiled FPP core is NOT readable in this env, so the
energy form is inferred, not read line-by-line — flagged in the entry.

What it does: keeps a dynamic set of pairwise **junctions** (focal-point links) between cells of a
type pair, and adds per link a spring energy `lambda*(d - target_distance)^2` on the distance `d`
between the two cells' **centers of mass**. Links form on contact (both cells below max_junctions),
pay a one-time ActivationEnergy at formation, and break when `d > max_distance`. So it is an energy
term that ALSO carries persistent inter-cell state — the interesting bit for the algebra: not a
stateless Potts plugin, it mutates a link registry between steps.

Surprised me: (1) `d` is CoM-to-CoM, long-range — target 7 / break 20 on volume-25 cells, i.e. links
span well beyond cell contact. (2) ActivationEnergy is XML-only, explicitly NOT runtime-steerable
even though targetDistance/lambda/maxDistance are (generator warning L882-884) — a one-time
formation threshold, not a per-step energy. (3) default energy comes from `LinkConstituentLaw`, which
is user-overridable with an arbitrary formula string over Lambda/Length/TargetLength.

Could NOT determine: the exact compiled energy/lifecycle code (inferred from the default
LinkConstituentLaw formula + Swat et al.); the full variable set bindable in a custom
LinkConstituentLaw and how the core parses/evaluates the formula string; whether link formation is
scanned every MCS or only on boundary-changing copies (I assume the latter from the neighbor_order
contact semantics, but did not confirm in core). No ablation run exists for this mechanism yet.

### Verification pass (re-excavation)
Re-checked the two specific citations in this entry against source, both hold:
- ActivationEnergy-XML-only warning is verbatim at `CC3DMLGeneratorBase.py:L882-884`.
- The `Lambda 10 / ActivationEnergy -50 / TargetDistance 7 / MaxDistance 20 / MaxNumberOfJunctions 1`
  defaults are real, but they come from the **twedit generator template** (`CC3DMLGeneratorBase.py`
  L894-898), NOT from the `FocalPointPlasticity.xml` test — that test file has no FPP block at all,
  only Volume(target 25)/Contact/ExternalPotential. So "target 7 / max 20 on volume-25 cells"
  combines two different co-named sources; entry surprise #3 was corrected to say so. The negative
  ActivationEnergy default (-50) corroborates "negative promotes link formation."

### Re-read pass (LinkConstituentLaw variables)
Resolved part of the `law` UNKNOWN by reading `LinkConstituentLaw` (`PyCoreSpecs.py:L4512`) more
closely: Lambda/Length/TargetLength are the built-in default variables, and ARBITRARY extra
variables are bindable via the `variable[name] = value` accessor (L4550), each emitted as a
`<Variable Name=.. Value=..>` child of `<LinkConstituentLaw>` (L4545). Updated the `law` param role to
say so. Still unread: how the compiled core parses/evaluates the Formula string — only the XML
emission is visible from the Python layer, not the evaluator.


---

## kerneldiffusionsolver

<!-- kerneldiffusionsolver -- append below; the driver merges this into campaign/analysis.md -->

## KernelDiffusionSolver (order 17)

Read the class (PyCoreSpecs.py:6642) and all its child specs: KernelDiffusionSolverDiffusionData
(6401), SecretionData (6464), BoundaryConditions (6477), Field (6533), plus the shared bases
`_PDEDiffusionDataSpecs` (554), `_PDESolverFieldSpecs` (1057), `_PDESolverSpecs` (1150). Also the
shipped guide text `diffusion_solvers_descr.py:36`. code_path was already correct.

What it is: a PDE diffusion solver, sibling to DiffusionSolverFE (order 14). Same field->field
write role, but the method differs — it advances the concentration field by CONVOLUTION with a
diffusion kernel (Green's function) rather than a local FTCS stencil. That makes it fast for
LARGE diffusion constants, the exact regime where the FE solver is slow/unstable.

Three things surprised me, all real constraints (not my inference):
- Periodic BCs are HARD-ENFORCED (all six boundary types read-only = periodic, 6477-6505). The
  convolution assumes a torus. This is the sharpest contrast with DiffusionSolverFE.
- D and decay are GLOBAL-ONLY: the DiffusionData.xml (6416-6431) never emits diff_types/
  decay_types even though the base spec_dict carries them — so no per-cell-type coefficients,
  necessarily, since one convolution kernel has one width.
- The guide flags it "legacy" and "approximate" — not an exact solution.

What I could NOT establish: the precise meaning of the two distinguishing params, `kernel` and
`cgfactor`. The source documents them only as "kernel of diffusion solver" and "coarse grain
factor", with validators requiring >= 1 and XML emitted only when > 1. I INFERRED kernel =
number of expansion terms / periodic images in the kernel, and cgfactor = lattice downsampling
factor, from how kernel/convolution solvers usually work — but the actual convolution lives in
the compiled core (.so), which I did not read. Anyone tuning these is tuning core behaviour I
could only describe from the outside. I also did not verify the decay/secretion coupling order
within a step (multiplicative decay vs additive source) against the core; that ordering in the
equations block is the standard reading, not confirmed. No paper text available for this target,
so paper_section points at in-tree checkable anchors only.


---

## lengthconstraint

<!-- lengthconstraint -- append below; the driver merges this into campaign/analysis.md -->

# LengthConstraint (order 18) — excavation note

**What I read.** `PyCoreSpecs.py:3850-4056` — `LengthEnergyParameters` (per-type param block:
cell_type, target_length, lambda_length, optional minor_target_length) and
`LengthConstraintPlugin` (a `_PyCoreSteerableInterface`, so steerable). The Python side only emits
CC3DML `<LengthEnergyParameters CellType TargetLength LambdaLength [MinorTargetLength]/>`; the
physics is the compiled `LengthConstraintPlugin.changeEnergy` (SWIG stubs at
`cpp/CompuCell.py:5219-5264`, with plane-specific `changeEnergy_xy/_xz/_yz` and `_3D`). Also read
the reference sim `elongationFlexTest` (xml + steppable): it drives elongation via
`setLengthConstraintData(cell, lambdaLength, targetLength)` per cell and pairs the plugin with
Volume + CenterOfMass + ConnectivityGlobal.

**Mechanism.** A Potts ENERGY term, not a state update: `E = lambda_L (L - L_t)^2`, plus a
`(W - W_t)^2` minor-axis term when minor_target_length is set. `L` is the cell's extent along the
longest principal axis of its moment-of-inertia tensor about the COM — a mass-weighted continuous
length, not a pixel bounding box. Returns `dE` into the Metropolis test; writes nothing.

**Surprised me.** (1) `setLengthConstraintData(cell, lambda, target)` passes lambda BEFORE target —
opposite of the CC3DML attribute order and of intuition. (2) Constraining only the major axis lets
a cell hit its target length by thinning to a filament; the reference sim needs Volume +
Connectivity to keep the cell physical. (3) The plugin is steerable, so target_length can be ramped
in time — active elongation, not a static prior.

**Could NOT establish.** The exact eigenvalue→scalar-length map inside compiled `changeEnergy`
(normalization/factor — is `L` a diameter, a semi-axis, sqrt-scaled?). I recorded only the
`(L - L_t)^2` penalty FORM as established; the constant is unread. Also unverified: whether the
minor-axis term reuses the same `lambda_L` or a separate coefficient (source exposes one
`LambdaLength` attribute, so I assumed shared — confirm against a 2D run). No extracted paper text
exists; the source class + CC3DML + the elongationFlex reference sim are the only evidence.

**Added from the `.so` symbol table (`libCC3DLengthConstraint.so`).** Two things not visible from
Python: (1) `spring_energy(double,double,double)` confirms the quadratic-spring FORM directly in the
binary. (2) `_get_non_nan_energy(double)` — an explicit NaN guard on the energy. A degenerate cell
(single pixel, undefined inertia) yields a NaN length; without this guard the NaN propagates into
`dE` and silently corrupts the Metropolis test. A reimplementer would almost certainly miss it.
Also: per-cell state is a `LengthConstraintData` ExtraMember, so the plugin carries per-cell (not
just per-type) targets — the local-flex scope `setLengthConstraintData` writes into.


---

## momentofinertia

<!-- momentofinertia -- append below; the driver merges this into campaign/analysis.md -->

# MomentOfInertia (order 19)

## What I read
- Spec at `PyCoreSpecs.py:5350` (`MomentOfInertiaPlugin`) is a stub: no constructor args, emits a
  bare `<Plugin Name="MomentOfInertia"/>`. Fixed nothing — line was correct.
- The physics is in the compiled core. Traced the SWIG bindings in `cpp/CompuCell.py`:
  - Plugin is a **`CellGChangeWatcher`** (line 9002) — invoked by the Potts solver *after* an
    accepted pixel copy, not an energy plugin and not a steppable.
  - It maintains six per-cell tensor fields `CellG.iXX/iYY/iZZ/iXY/iXZ/iYZ` (lines 545-550) and
    derives `CellG.ecc` (554), semiaxes (`getSemiaxes*`, 9030-9043) and orientation
    (`cellOrientation_xy/xz/yz`, 9019-9026).
  - Incremental helper `precalculateInertiaTensorComponentsAfterFlip` (4235) + `eccFromComps`
    (8994) confirm O(1)-per-flip maintenance and the eccentricity-from-components derivation.
- Used by shape-dependent code (oriented-cellsort test XML, `OrientedGrowth` in PySteppables).

## What surprised me
- It's a genuine **third category** for this atlas: not an energy term (returns no dE, no acceptance
  role) and not a modeller update — a passive *change-watcher* running an **incremental reduction**
  that keeps a derived statistic in sync with the pixel set. Worth flagging for the vocabulary
  question: does the Plexus algebra have a "tracker / incremental observable" contract distinct from
  operators that return deltas?
- Depends implicitly on CenterOfMass (measures the tensor about `r_CM`); no param exposes this.

## What I could NOT establish
- The exact diagonal/sign convention in the compiled tensor (physicist `sum(y'^2+z'^2)` vs raw
  central second moment `sum(x'^2)`). Inferred physicist form from names+physics; unverified. The
  downstream `ecc`/semiaxis derivations are what code actually reads and are robust to the choice.
- Exact semiaxis normalisation constant (eigenvalue → length) — compiled, not read.
- No ablation/evidence run exists for this mechanism (not among the six with metrics.json).

## Added this pass (re-verified at source)
- The derived fields are **read-only from Python**: the binding overrides `set_iXX` (etc.) to raise
  `AttributeError "iXX is read only variable"` (`cpp/CompuCell.py:649`). The modeller can *read*
  cell shape but cannot assign it — only the plugin's watcher may. This hard-enforces the observer
  reading and is a concrete reimplementer trap: exposing the tensor as writable Plexus state would
  be a category error. Added as a surprise.
- The watcher hook is `field3DChange` (this plugin at `cpp/CompuCell.py:9016`, base
  `CellGChangeWatcher.field3DChange` at 3000), i.e. per-accepted-cell-id-change, NOT per-MCS. The
  helper returns an `InertiaTensorComponents` struct (`cpp/CompuCell.py:4200`). Added as a surprise
  so a reimplementer doesn't attach a fixed-schedule stepper.
- Made `paper_section` honest: no extracted paper text exists; anchored to the docstring
  (PyCoreSpecs.py:5351) and compiled class (cpp/CompuCell.py:9002), and explicitly flagged the
  Swat page/section as UNREAD rather than citing a page I have not seen.


---

## neighbortracker

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

## pifdumper

<!-- pifdumper -- append below; the driver merges this into campaign/analysis.md -->

# PIFDumper (order 21)

Read `PyCoreSpecs.py:L6012` (the `PIFDumperSteppable` class), its base
`_PyCoreSteppableSpecs` (L512), and the sibling `PIFInitializer` (L5960) that
shares the `pif_name` field. Also skimmed `CMLResultsReader.generate_pif_from_vtk`
to confirm the PIF line format (`cellId cellType xLow xHigh yLow yHigh zLow zHigh`)
lives in the compiled `PlayerPython.FieldWriter`, not in Python.

What it does: a pure-output Steppable. Every `frequency` MCS it reads the whole
cell-id/cell-type lattice and writes a PIF snapshot to disk. It touches NO
simulation state -- no delta, no energy term, no pixel-copy bias. Cleanest
"neither an update nor an energy term" case so far: it's a read-only tap with an
external side effect. Worth flagging for the normalizer as an observer/sink, not
a physics operator.

What surprised me: the `xml` property emits ONLY `PIFName` and silently drops
`frequency`, even though `__init__` stores it, `check_dict` validates it, and
`from_xml` reads a `Frequency` attribute. So Python->CC3DML round-tripping loses
the interval (defaults back to 1). `__getstate__` drops it too. Also `pif_name`
means opposite things in the dumper (output target, not existence-checked) vs the
initializer (input, must exist). And the dumper's `xml` is byte-identical to the
initializer's -- only the Steppable `Type=` attribute distinguishes write from read.

What I could NOT establish: (1) HOW the core steppable actually receives
`frequency` at runtime given the xml omission -- there may be another wiring path
in the compiled core I did not read; I only confirmed the Python spec does not
carry it. (2) The exact on-disk PIF byte format (fixed vs free spacing, header
lines, 2D vs 3D bounds) -- inferred from `generate_pif_from_vtk`/library convention,
not observed from an actual dump. (3) Whether frequency counts MCS or some other
tick, and phase/offset behavior at mcs=0. Did not run evidence.py (not among the
six with reference runs).

## Follow-up pass (corrections against a REAL .piff + the twedit template)

Read a real dumped file,
`tests/plugin_test_suite/AdhesionFlexPython_test_generate/Simulation/initial_configuration.piff`,
and twedit's `CC3DMLGenerator/CC3DMLGeneratorBase.py:1271`
(`generatePIFDumperSteppable`). Three things above are now superseded:

- The PIF line format sketch (`cellId cellType xLow..`) is WRONG. A real line reads
  `8  8  Cell2  119 119  53 53  0 0`, i.e. **clusterId  cellId  typeNAME  x x  y y  z z**.
  The first column is the cluster id (not cell id), the type is a NAME string (not a
  numeric index), the file is prefixed with an `Include Clusters` header line, and
  every line is a single voxel (no run-length box compression). Entry equations +
  a dedicated surprise now carry the corrected format.
- The `frequency` omission is now LOCALIZED, not a mystery: `from_xml` reads it as a
  `<Steppable>` **header attribute** (`Frequency="…"`), which is exactly where the
  canonical twedit CC3DML puts it (default **100**, not 1). `generate_header` never
  emits that attribute — so the round-trip loss is fully explained; no hidden core
  wiring needed for the datum itself, only for the write it never receives.
- twedit also emits `<PIFFileExtension>piff</PIFFileExtension>` and names the file
  from SimulationName; PyCoreSpecs emits neither. Added as a surprise.

Still NOT established: the C++/SWIG writer itself is unread, so traversal order, the
MCS→filename suffix rule, and whether the `Include Clusters` header is conditional
are inferred from one sample + the template, not from the writer code. Still no
evidence.py run (a disk sink has no meaningful ablation).


---

## pifinitializer

<!-- pifinitializer -- append below; the driver merges this into campaign/analysis.md -->

# PIFInitializer (order 22)

**What I read.** `PyCoreSpecs.py:5956` — the whole `PIFInitializer` class plus its base
`_PyCoreSteppableSpecs` (line 512; `generate_header` at 519, emits `<Steppable Type="PIFInitializer">`)
and its sibling `PIFDumperSteppable` (6012).
The Python side is tiny: one param `pif_name`, `validate()` asserts the file exists, `xml`
emits `<PIFName>`. The real work is in `cpp/.../libCC3DPIFInitializer.so` (compiled, not
readable), so I reconstructed the painting behaviour from actual `.piff` files in the test
suite (`AdhesionFlexPython/.../initial_configuration.piff`, `CompartmentExample/.../3D_LINE_BASE.piff`).

**What it does to state.** One-shot initializer at t=0. Reads a PIF text file and stamps the
cell-id lattice directly: each line = `[clusterId] cellId cellType x1 x2 y1 y2 z1 z2`, and every
site in the inclusive box is assigned to that cell. Lines sharing a cellId accrete sites → a cell
is the union of its boxes. This is the cleanest instance of CC3D's "cell = set of lattice sites"
representation in the whole inventory.

**Surprises.** (1) The `Include Clusters` header silently adds a leading column — column meaning
is header-dependent. (2) Box bounds are inclusive both ends. (3) Internal spec name is the typo
`"pif_initiazer"`. (4) `check_dict` only rejects the empty string at construction; existence is
checked later in `validate()`. (5) Cells need not be connected/rectangular.

**What I could NOT establish.** The exact core semantics — pixel iteration order, what happens on
OVERLAPPING boxes (last-writer-wins vs. first?), out-of-lattice-bounds clipping, and whether an
unknown cellType is auto-registered or errors — all live in the unreadable `.so`. I inferred
inclusive boxes and set-union from the file contents, not from core code. No paper text exists for
this target, so `paper_section` is anchored to the source line + the docs `.rst` autoclass, not a
verified page. Did not run evidence.py (initializer has no dynamics/ablation to measure).

**Format re-verification (this pass).** Checked BOTH header modes against real files, not just the
two cluster examples: cluster mode = 9 tokens, and `3D_LINE_BASE.piff` PROVES clusterId is
independent of cellId (clusterId 1 groups cellIds 1-5 — earlier notes asserted this; now confirmed).
Headerless mode = 8 tokens, seen in `ExternalPotential/.../FocalPointInit.piff` and
`amoebae_2D_secretion/.../amoebae_2D.piff`. Made the entry's checkable anchors concrete with these
exact paths. Overlap/last-writer semantics remain the one unresolved gap (still in the `.so`).


---

## pixeltracker

<!-- pixeltracker -- append below; the driver merges this into campaign/analysis.md -->

# PixelTracker (order 23) — excavator note

Read: PyCoreSpecs.py:5311 (the spec, one param `track_medium`), the SWIG stub
CompuCell.py:6002 (`class PixelTrackerPlugin(CellGChangeWatcher)` — the decisive
fact), the attached-data classes PixelTracker/PixelTrackerData (5837, 5854), the
iterator CellPixelList (iterators.py:700), and the consumers in PySteppables.py
(get_cell_pixel_list:2465, get_copy_of_cell_pixels:2584, move_cell/delete_cell).

What it does: it materialises the inverse of the cell_field. The CPM lattice stores
site -> cell id; PixelTracker keeps a live per-cell set of the sites that id owns,
updated incrementally on every accepted pixel copy via field3DChange (erase pt from
old owner, insert into new owner). Delta H = 0 — it is NOT an energy plugin.

Surprised me: this is the plugin that reifies the ATLAS's own framing ("a cell is a
set of lattice sites sharing an id"). In CC3D that set is NOT primitive — you must
load a watcher to build the id->{sites} map; the lattice alone only gives you
site->id. In Plexus the set is the primitive, so PixelTracker looks like an
implementation of a representation Plexus gets for free. Whether the algebra needs
an explicit "index / inverse-map over a field partition" contract is the open
question I flagged in surprises — I did NOT resolve it (that is the normalizer's call).

Could NOT establish: I could not read the compiled C++ field3DChange body itself —
only the SWIG signatures and the Python consumers — so the exact erase/insert
ordering and the medium-guard branch are reconstructed from the SWIG API
(enableMediumTracker, getMediumPixelSet, trackingMedium) plus behaviour, not from
reading the .cpp. High confidence given CellGChangeWatcher semantics, but stated as
reconstruction. No ablation/evidence run exists for this mechanism (not among the
six with metrics.json). track_medium default False is a performance guard, verified
from the spec's conditional <TrackMedium/> emission (PyCoreSpecs.py:5333).

Re-verification pass (correction): the earlier entry claimed get_cell_pixel_list
RAISES when PixelTracker is absent. It does not — it silently returns None
(PySteppables.py:2473-2476). The AttributeError is raised only by the higher-level
wrappers get_copy_of_cell_pixels (2603), delete_cell (2534), and move_cell (469).
Fixed the entry's dependency surprise to reflect the silent-None low-level path (a
worse gotcha than a clean raise: a forgotten None-check surfaces as a confusing
NoneType error downstream). Also confirmed the container is concretely `pixelSet`
on the extra-attrib accessor (iterators.py:703/724). Everything else in the entry
checked out against source; status stays inspected.

Anchor re-verification (this pass): re-read PyCoreSpecs.py:5311 (class def line
correct — code_path unchanged) and located the real SWIG stub at
cc3d/cpp/CompuCell.py (not a bare CompuCell.py): class PixelTrackerPlugin at :6002,
PixelTrackerPlugin_field3DChange at :6010/:6011, enableMediumTracker at :6031, all
confirmed. iterators.py:700 (CellPixelList) and :724 (pixelSet.size) confirmed
verbatim. Corrected the paper_section anchor to the full checkable path
cc3d/cpp/CompuCell.py:6002. Prior non-acceptance was purely a merge-hygiene issue
(atlas_record.yaml touched directly) — the science was unchanged; edited only the
working copy this time.

Line-attribution correction (this pass): re-reading the consumers showed the raise
sites were mis-attributed in the dependency surprise. The verified truth:
get_copy_of_cell_pixels (def 2584, raises 2603) and move_cell (def 2509, raises
2534) both raise AttributeError('Could not find PixelTracker Plugin');
delete_cell (def 2636) raises only INDIRECTLY through get_copy_of_cell_pixels; and
line 469 is merge_cells (def 458), which raises its own Exception "…did you load
PixelTracker plugin?" — NOT move_cell. Entry's third surprise updated to match.


---

## reactiondiffusionsolverfe

<!-- reactiondiffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# ReactionDiffusionSolverFE (order 24)

Read: `PyCoreSpecs.py:L6911` (the class) + its DiffusionData (L6740), SecretionData (L6830),
Field (L6843), and the base PDE classes (`_PDEDiffusionDataSpecs` L554, `_PDESolverSpecs` L1150,
`_PDESolverFieldSpecs` L1057). Cross-read the sibling `diffusionsolverfe` entry (order 14) for the
FTCS discretisation, the FitzHugh-Nagumo example XML
(`tests/pde_solvers/.../ReactionDiffusion_2D_FN.xml`), the twedit generator
(`CC3DMLGeneratorBase.py:1959`), and the in-tree solver blurb (`diffusion_solvers_descr.py:11`).

What it is: DiffusionSolverFE's forward-Euler diffuse-decay-secrete step, PLUS a per-field
`AdditionalTerm` -- a muParser expression that may name the OTHER fields, coupling them into a
reaction-diffusion system. It writes the concentration grids in place (a real field write, not a
Potts energy delta).

Surprised me:
- Same python attribute `additional_term` is emitted here but silently dead in DiffusionSolverFE
  (its DiffusionData.xml never writes it). The coupling is invisible if you compare specs by
  attribute list.
- ConstantConcentration (Dirichlet clamp) is explicitly REFUSED (SpecValueError, L6892).
- A BLANK reaction defaults to `1*<field>` i.e. R=c (exponential growth), not R=0 (twedit
  L2007/2011) -- a magic default.
- init_expression is read by from_xml but never emitted -> lost on write; only init_filename
  round-trips.
- AutoscaleDiffusion is per-steppable in PyCoreSpecs (L6943) but per-field in twedit (L1990).
- (added this pass) from_xml (L7016-7019) PARSES <ConstantConcentration> and forwards
  constant=True to secretion_data_new -- the same method (L6892) that RAISES on that kwarg. So the
  reader accepts a CC3DML the writer forbids: importing a legacy pinned-source model crashes on the
  read, not on write. Reader/writer disagree about whether Dirichlet clamps exist, inside one class.

Could NOT establish (someone must not assume otherwise): the exact FE integration of the reaction
term -- DeltaT scaling, evaluation order relative to the diffusion sweep, and whether
AutoscaleDiffusion rescales R -- all live in the compiled core, which I did not read. The continuous
form and named-field coupling are solid (spec xml + FN example); the discretisation of R is inferred.
DeltaX/DeltaT/ExtraTimesPerMCS remain unreachable from this python spec, same gap as DiffusionSolverFE.


---

## secretion

<!-- secretion -- append below; the driver merges this into campaign/analysis.md -->

## secretion (EXCAVATOR, read at source)

Read `SecretionPlugin` (PyCoreSpecs.py:L4306) plus its two helper specs `SecretionField`
(L4192) and `SecretionParameters` (L597), and the compiled `FieldSecretor` API in
cpp/CompuCell.py (L9082+). The Python layer only emits CC3DML; the physics is in
libCC3DSecretion.so, so I read semantics off the FieldSecretor method names + the amoebae
test XML (`<Secretion Type="Amoeba">20</Secretion>`).

What it does TO STATE: it is a genuine FIELD WRITE (not an energy term). Every MCS it walks
the lattice sites a cell owns and either ADDS a rate (`Secretion`), OVERWRITES a level
(`ConstantConcentration`, a Dirichlet clamp), or adds a rate only at boundary sites touching a
named other type (`SecretionOnContact`). It does NOT transport the chemical — a separate PDE
solver diffuses the same field. This is one of the few CC3D mechanisms that actually writes
state each step, so `state_io.writes` is real, unlike the accept/reject plugins.

Surprised me: (1) `value` is overloaded — a per-step RATE in additive modes but an absolute
CONCENTRATION in constant mode. (2) `ExtraTimesPerMC` (frequency) silently multiplies the
effective rate. (3) The exact same physics can be declared either as this plugin OR inline as
`<SecretionData>` inside a DiffusionField — two spec surfaces, one mechanism. (4) It depends on
PixelTracker/BoundaryPixelTracker for the pixel sets it iterates.

Could NOT establish (compiled core, not read): the intra-MCS ORDERING relative to the diffusion
solve (does secretion inject before or after the field is stepped?), whether the plugin path
uses the "old" or "new" field, and the precise neighbour-iteration order for on-contact. I
inferred additive-vs-overwrite and rate-vs-level purely from method names + XML, not from
running it — no evidence run exists for this mechanism, so those semantics are unverified by
measurement.

### Addendum (second pass, read cpp/CompuCell.py FieldSecretor + PySteppables)

Two corrections/enrichments after reading the compiled `FieldSecretor` (CompuCell.py:L9082) and
`SecretionBasePy` (PySteppables.py:L3392):

- STRICT SUBSET: the XML plugin's `from_xml` maps only Secretion / ConstantConcentration /
  SecretionOnContact onto the three INSIDE-cell variants. FieldSecretor additionally exposes
  UPTAKE (a sink, `uptakeInsideCell*`, absolute + relative-to-max), OUTSIDE-cell boundary
  secretion (`secreteOutsideCellAtBoundary`, writes the medium sites just outside the cell), and
  COM-only point secretion (`secreteInsideCellAtCOM`). These are Python-scripting-only — not
  reachable from the declarative Secretion plugin. Added as a surprise.
- CORRECTED an over-assertion: the previous `state_io.writes` credited `runBeforeMCS=1` to the
  plugin. That flag belongs to the PYTHON steppable `SecretionBasePy`, a different code path. The
  compiled `SecretionPlugin` is in libCC3DSecretion.so; its intra-MCS ordering vs the diffusion
  solve is NOT readable from Python. Downgraded to a hint and moved the uncertainty into a
  surprise, so no one inherits it as fact.

Still could NOT establish (unchanged): the compiled plugin's write ordering relative to the PDE
solve, old-vs-new field buffer, and on-contact neighbour-iteration order. No evidence run exists,
so mode semantics remain inferred from method names + amoebae_2D XML, not measured.

### Addendum (third pass, read the OpenCL secrete KERNELS — actual arithmetic, not method names)

The two prior passes inferred semantics from FieldSecretor symbol names + XML. I read the actual
GPU arithmetic in `cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (the DiffusionSolver's
embedded secretion; the same three modes). This CONFIRMS additive/overwrite/on-contact and adds
three things the name-level reading could not see:

- ZERO IS A NO-OP GUARD, not a clamp-to-zero. Every mode is wrapped in `if (value) { ... }`
  (L216 plain, L262 constant, L314 on-contact). So constant-mode value=0 does NOTHING — it does
  not pin the field to zero. Added as a surprise.
- ON-CONTACT IS NON-ACCUMULATING in this kernel: base conc `c0` is read ONCE (L301) before the
  neighbour loop, then each qualifying neighbour re-assigns `c := c0 + rate` (L315/335) — so N
  contacts do NOT deposit N*rate and the LAST matching neighbour type wins. This DIVERGES from the
  entry's additive `phi += r` equation (correct only for a single contact). I flagged it in the
  note but did NOT rewrite the entry's equation, because this is the GPU DiffusionSolver path and I
  did not disassemble libCC3DSecretion.so to confirm the standalone plugin behaves identically.
- VOLUMETRIC SOURCE: plain/constant write every owned pixel, so total mass scales with cell volume
  (a big cell secretes proportionally more). Added as a surprise.
- Kernel-only detail I did NOT promote to the entry: on-contact uses a medium sentinel of id == -2
  (`NON_CELL`), with medium id == -1 (L302). Whether the plugin's C++ uses the same sentinel is
  unverified, so I left it out of the record to avoid overclaiming.

Net: the OpenCL path corroborates the three declared modes and shows the kernel ALSO implements the
uptake sink the prior pass found by name (`c -= min(c*relUptake, maxUptake)`, L221-233) — confirming
the "Python spec is a strict subset" surprise from a second source. Still unverified: byte-identity
between this GPU kernel and the standalone plugin's compiled CPU path.

### Addendum (resubmission pass)

Re-verified `code_path` L4306 = `class SecretionPlugin` (unmoved) and the `from_xml` mode mapping
(L4435-4451). Made `paper_section` honest: we have NO extracted paper text for this target, so the
anchor now says so and names the source (PyCoreSpecs.py + amoebae_2D XML) as the only evidence,
rather than implying I read a paper section. No analytical claims changed.


---

## steadystatediffusionsolver

<!-- steadystatediffusionsolver -- append below; the driver merges this into campaign/analysis.md -->

# SteadyStateDiffusionSolver (order 26)

Read the class at PyCoreSpecs.py:L7223 (`SteadyStateDiffusionSolver`), its diffusion-data
child `SteadyStateDiffusionSolverDiffusionData` (L7054), the field spec (L7130), the secretion
override (L7194), the base `_PDESolverSpecs`/`_PDESolverFieldSpecs`/`_PDEDiffusionDataSpecs`
(L1150/L1057/L554), and `PDEBoundaryConditions` (L820). In-tree behaviour anchor:
`diffusion_solvers_descr.py` — "Solves Diffusion equation at the steady state i.e. at
time= infinity ... Technically this solver solves Helmholtz Equation." Wizard defaults in
CC3DXMLGenerator.py:997-1054.

What it does to state: unlike DiffusionSolverFE (one explicit forward-Euler step per MCS),
this DISCARDS the transient and writes the equilibrium field — the solution of the Helmholtz
BVP `D∇²c − λc + S = 0` under the per-face boundary conditions. Field->field write, global
(every site depends on every boundary), no timestep, no FTCS limit.

Surprised me:
- λ (decay_global) is structurally load-bearing (screening length √(D/λ)); the wizard default
  decay is 1e-5, NOT 0 — a near-singular guard against the all-Neumann/λ=0 singular case.
- Per-cell-type coefficients (diff_types/decay_types) exist in the base spec_dict but are NOT
  emitted by this solver's DiffusionData.xml → D and λ are uniform through this API (contrast FE).
- Secretion is restricted: secretion_data_new RAISES for ConstantConcentration and
  SecretionOnContact — only additive-rate secretion allowed. Porting an FE spec with a
  constant/Dirichlet secretion crashes at spec time.
- Two registered names via `three_d` (default 2D): SteadyStateDiffusionSolver2D vs …Solver.

Could NOT establish (someone's future false belief if I don't say it):
- The compiled linear-solve kernel is a .so I did not read: exact method (SOR / CG / BiCGSTAB),
  tolerance, iteration cap, and how Neumann/Periodic faces are discretised are UNVERIFIED. I take
  "Helmholtz at steady state" from the guide, not from the numerics.
- Whether init_expression/init_filename are actually consumed as a solver SEED or ignored — I
  inferred "seed, not physical IC" from the steady-state framing but did not confirm in the core.
- No evidence run for this mechanism (not among the six with metrics.json); all above is source-read.

**Addendum (re-excavation, verified anchors).** Confirmed the substance above from source. The
exact guide string I could locate is `CC3DMLGeneratorBase.py:24` ("Solves Diffusion equation at
the steady state i.e. at time= infinity ... Technically this solver solves Helmholtz Equation");
`core/diffusion_solvers_descr.py` has no `steady` hit in this build, so treat that as the anchor.
The "secretion must live inside the solver, Secretion Plugin does not work" restriction is stated
verbatim at `CC3DMLGeneratorBase.py:2456`, and the near-singular default `DecayConstant 0.00001`
is the template default at `CC3DMLGeneratorBase.py:2444`. Everything else stands as written.

**Correction to the addendum (anchor verified by direct read).** The Helmholtz guide quote is
NOT at `CC3DMLGeneratorBase.py:24` — line 24 there is decorator code (`obj = args[0]`). The
verbatim quote ("Solves Diffusion equation at the steady state i.e. at time= infinity ...
Technically this solver solves Helmholtz Equation") lives at
`cc3d/twedit5/Plugins/CC3DProject/diffusion_solvers_descr.py:23-24` (heading at 23, sentence at
24). The addendum's "`core/diffusion_solvers_descr.py` has no `steady` hit" is a wrong-path
grep: the file is under `twedit5/Plugins/CC3DProject/`, not `core/`, and it does contain the
string (also at line 7). `paper_section:` in the entry now points at the correct file. The
`:2456` Secretion-Plugin anchor is confirmed correct.

**Addendum (pymanage).** Added one surprise from direct source read: the field spec's `pymanage`
flag emits a bare `<ManageSecretionInPython/>` element and DROPS the whole SecretionData block
(`SteadyStateDiffusionSolverField.xml`, PyCoreSpecs.py:L7156-L7159). When set, S(x) is supplied by
a user Python steppable each MCS instead of the declared per-type rates — a reimplementer reading
only the SecretionData path would miss that the source term can be externally driven. This is a
declaration-layer branch; whether the compiled core honours the semantics identically is unverified
(the .so was not read).


---

## surfacetracker

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


---

## uniforminitializer

<!-- uniforminitializer -- append below; the driver merges this into campaign/analysis.md -->

## uniforminitializer (UniformInitializer) — excavated

Read `PyCoreSpecs.py:L5742-5953` (`UniformInitializerRegion` + `UniformInitializer`), the base
steppable `_PyCoreSteppableSpecs:L512` (generate_header emits `<Steppable Type="UniformInitializer">`),
and the CC3DML generator `CC3DMLGeneratorBase.py:L1189-1226` whose comment names it exactly:
"Initial layout of cells in the form of rectangular slab."

- **Same shape as blobinitializer: a set CONSTRUCTOR, not an energy term.** It runs once before
  MCS 0 and WRITES the cell field — but tiles a rectangular BOX instead of a disk/sphere. Per
  region it lays cubic cells of edge `width` on a grid with pitch `width+gap`, gap-sites left as
  Medium, each block a fresh cell id with a type from `cell_types`. `state_io.writes` says so
  plainly (cell.id + cell.type over the box); reads nothing dynamic.
- **The Python class only serializes CC3DML.** BoxMin/BoxMax/Gap/Width/Types emitters
  (`xml` property L5795-5809, region L5804-5808). The actual tiling + type draw live in the
  compiled core, not importable here, so `equations:` is a reconstruction from the emitted fields
  and standard CC3D semantics, flagged UNVERIFIED.
- **Sharpest gap I could NOT close: type-assignment rule.** The spec emits only a comma list of
  type names. Whether the core assigns block types by uniform-RANDOM draw (CC3D lore) or CYCLIC
  round-robin is not encoded anywhere readable. Flagged in surprises — a reimplementer must not
  assume deterministic cycling. (Contrast blobinitializer, whose note asserts random-by-convention;
  I declined to assert it here since I have no stronger evidence than the same convention.)
- **Pitch trap:** block pitch = `width+gap`, cell volume = `width^dim`; folding gap into the cell
  size is the obvious error. `gap` guarded non-negative, `width` guarded >=1 (check_dict, L5745).
- **Ordering:** `UniformInitializer(*_regions)` takes many regions positionally, each validated to
  be a `UniformInitializerRegion`; boxes may overlap, so later regions can overwrite earlier —
  recorded as an ordering assumption.

**Could NOT establish** (all compiled-core, not readable here, stated as such in the entry):
random-vs-cyclic type assignment; BoxMax inclusivity and far-edge partial-block handling
(dropped vs clipped); and the empty-`cell_types` meaning (all-Medium vs default type). **No paper
text available** — `paper_section` keeps the chapter reference and adds checkable SOURCE anchors
(class L5833, region L5742, generator comment L1197); no page/figure invented. Not one of the six
mechanisms with reference runs, so source-read only, no measured evidence.

### re-excavation pass (working copy restored after an out-of-band edit to atlas_record.yaml)

Confirmed two more source-readable facts and added them to `surprises:`:
- **width has no usable default.** `__init__` defaults `width=0` (L5754) but the guard rejects
  `width < 1` (L5747) — a region built without an explicit width fails validation. `gap` does
  default to a valid 0 (guard only rejects `<0`), and `from_xml` (L5905/5907) treats Width as
  required, Gap as optional — matching the guard asymmetry.
- **Types round-trips as a comma-joined string** (`xml` L5808 → `from_xml` L5911 split-on-comma,
  spaces stripped): the ordered Python list survives only as text order, which is the only place
  any assignment ordering could live. Still does not resolve random-vs-cyclic — the draw is core-side.


---

## volume

<!-- volume -- append below; the driver merges this into campaign/analysis.md -->

## volume (VolumePlugin) — excavated

Read `PyCoreSpecs.py:L1975-2161` (VolumeEnergyParameter + VolumePlugin) and the generators/
metrics under `_oracle/_evidence/volume_constraint_{on,off}` + `log/atlas_cc3d/_ablations.json`.

- **It's an energy term, not an update.** `E_vol = Σ λ_V (V−V_target)²` over real cells; the plugin
  returns dE for a proposed pixel copy and writes nothing. V(σ) is a lattice-site COUNT. State_io
  says so plainly rather than forcing read/write language.
- **Two-plugin split surprised me most.** The *count* isn't kept by VolumePlugin — it's maintained
  incrementally by a separate auto-loaded `VolumeTrackerPlugin` (CellGChangeWatcher,
  CompuCell.py:L5554). Reimplementers who fold "recount" into the energy term, or forget the
  tracker, read a stale/zero volume. Worth flagging for the normalizer: the "volume" contract may
  really be two — a watcher that maintains a per-set count, and an energy that reads it.
- **A pixel copy is a two-cell event** (gainer V+1, loser V−1); the quadratic doesn't cancel, so dE
  couples both cells' distance-from-target. Medium (id 0) is exempt.
- **Ablation is now MEASURED, not guessed:** λ=0 → n 45→0, volume 25→0 by MCS 200 (OFF run), vs
  ON relaxing 25→59.4 toward target 60. Volume is literally what stops a cell dissolving into
  medium under positive contact energy. Kept the earlier per-type-vs-per-cell surprise (measured
  25→20.9 shrink); confirmed the ON evidence run uses PER-TYPE params, while the growth runs use
  BARE mode — the two paths are mutually exclusive.
- **Guards:** only target_volume is checked (≥0). λ is unchecked (the dissolving 0 passes), and
  there is no coupling guard to the Potts temperature — a large λ vs fixed fluctuation_amplitude
  freezes the boundary, so the constraint's realized effect is inseparable from a Potts param.

**Could NOT establish:** the compiled `changeEnergy` was not read (cc3d C++ is not importable
here), so the exact quadratic form (no leading ½, Medium exemption) is reconstructed from the
CC3DML declaration + standard CPM convention, not verified byte-for-byte. **No paper text is
available** — `paper_section` names the chapter's known home for the term but is not a page I
have read; I did not invent a page/equation number. Confirmed there is NO VolumeFlex/steerable
variant in PyCoreSpecs — VolumePlugin (L2033) is the sole volume energy spec.


---

## adhesionflex

<!-- adhesionflex -- append below; the driver merges this into campaign/analysis.md -->

# AdhesionFlex (Plugin) — excavation note

**What I read.** `PyCoreSpecs.py:L3525` (`AdhesionFlexPlugin`) plus its helpers
`AdhesionFlexMoleculeDensity` (L3461) and `AdhesionFlexBindingFormula` (L3380); the compiled
interface `cpp/CompuCell.py` (`AdhesionFlexData.adhesionMoleculeDensityVec`, `changeEnergy`,
`adhesionFlexEnergyCustom`, the `set/getAdhesionMoleculeDensity*` family); the library's own
energy descriptor `twedit5/.../adhesion_descr.py`; and the shipped test
`tests/plugin_test_suite/AdhesionFlexPython_test_run/` (XML matrix + a steppable that mutates
densities per cell). The Python spec is a CC3DML emitter only; the physics is compiled.

**The mechanism.** A flexible Contact energy. Sum over neighbouring site pairs whose owners
differ: `E = Σ_ij [ -Σ_mn k_mn · AdhesionFunc(N_m(σ_i), N_n(σ_j)) ] · (1-δ)`. Each cell (and
Medium) carries a *vector* of adhesion-molecule densities `N_m`; `k_mn` is a user interaction
matrix; `AdhesionFunc` is a muParser formula string, default `min(Molecule1,Molecule2)`. Writes
nothing to the lattice — returns dE, a Potts energy term.

**What surprised me.**
- It is not just Contact-with-more-numbers: it introduces genuine **per-cell mutable state** (the
  density vector), seeded from a per-TYPE declaration but thereafter steerable per cell and
  inherited across mitosis (`assignNewAdhesionMoleculeDensityVector` deliberately skips the size
  check for daughter seeding). Plain Contact has no per-cell state.
- **Inverted sign** vs Contact: an explicit leading minus makes positive `k_mn` adhesive; the
  shipped example even uses negative k's.
- The combining kernel is **data** — an arbitrary muParser expression, not a fixed op.

**What I could NOT establish.**
- The (1-δ) restriction: the library HTML says δ is over "cell *types*", but the standard CPM
  contact term and `changeEnergy` are over cell **id** σ. I recorded both readings and flagged
  it; I could not run the compiled core to settle it (cc3d is deliberately not importable in the
  Plexus env), so which governs same-type neighbouring cells is still open.
- No ablation/oracle run exists for this mechanism yet (not among the six with `evidence.py`
  outputs), so every claim here is source-read, not measured.
- No specific Swat et al. (2012) page was read; `paper_section` points to the checkable CC3D
  reference-manual heading + the in-repo descriptor, not a page number I have not seen.

---

## Normalizer verdict

**`new` (against the frozen baseline), `implementation_of: adhere` — a second sighting of
`adhere`, first proposed from the jax-morph atlas.**

AdhesionFlex is the pure differential-adhesion energy of the Cellular Potts framework: contacting
cells (and Medium) lower the boundary energy by a double sum over their carried adhesion-molecule
densities, so a positive binding parameter is adhesive. That is the same biology as jax-morph's
proposed `adhere` (cadherin-like surface molecules setting cell-cell stickiness and sorting). No
REGISTERED contract covers it — the nearest, `cohesion` / `attraction_repulsion`, emit a force
(`acceleration`) on point particles and write `position`, whereas AdhesionFlex writes nothing and
only returns a Metropolis `dE` over cross-boundary lattice-site pairs of a cell that IS a set of
sites. Set `implementation_of: adhere` so the ledger counts `adhere` once across repositories, not
twice.

**Strongest argument AGAINST this verdict:** that I should have called it `alias` /
`implementation_of: attraction_repulsion`, exactly as the jax-morph Morse potential was —
"adhesion" is arguably just the attractive tail of the one radial pair interaction, and
`attraction_repulsion` already IS registered whereas `adhere` is not, so mapping to it would avoid
minting fresh yield. The rebuttal: Morse carries BOTH a repulsive core (excluded volume) and an
adhesive tail, so it maps to the combined `attraction_repulsion`; AdhesionFlex carries ONLY
adhesion — excluded volume in CPM lives in the separate Volume constraint, not here — and it is an
energy over shared boundary counts of a site-set cell, not a centre-distance force that integrates
to move a point. Collapsing pure molecule-mediated adhesion into the repulsion-bearing point-force
contract would erase precisely the biology (surface-molecule differential adhesion, per-cell
mutable density vector, no self-repulsion) the campaign exists to measure. If the loop later
PROMOTES `adhere` and it turns out to be defined force-first, the energy-term-vs-force
representational gap recorded in `state_io`/`why` is where this entry should be revisited.


---

## blobinitializer

<!-- blobinitializer -- append below; the driver merges this into campaign/analysis.md -->

## blobinitializer (BlobInitializer) — excavated

Read `PyCoreSpecs.py:L5525-5736` (BlobInitializerRegion + BlobInitializer), `validate_point`
(L7457), the CC3DML generator (`CC3DMLGeneratorBase.py:1228-1250`), a commented example XML block
(`cellsort_2D.xml`), and how the oracle actually uses it (`oracle.py:95-97`).

- **It's a set CONSTRUCTOR, not an energy term.** This is the sharp contrast with everything else
  in this record: BlobInitializer runs ONCE at MCS 0 and genuinely WRITES the cell field — it
  paints a solid disk/sphere of freshly-created cells (each a new SET of lattice sites with a
  unique id) and assigns each a random type. State_io says so plainly: it creates the initial
  partition; downstream trackers (volume, center-of-mass) then maintain it. The energy-plugins
  write nothing; this writes the whole board.
- **The Python class does no painting.** `BlobInitializer`/`BlobInitializerRegion` are pure CC3DML
  serializers — Gap/Width/Radius/Center/Types emitters. The circle-clip, grid-tiling and random
  type draw all live in the compiled core (not importable here). Reading only PyCoreSpecs gives
  the parameters, never the algorithm — so `equations:` is a reconstruction from CC3D convention +
  the emitted fields, flagged as inferred.
- **A declared-validation vs working-use contradiction (surprised me most).** `validate()` bounds-
  checks `center ± radius` on ALL THREE axes; for the oracle's own 2D blob (dim_z=1, center.z=0,
  radius=dim//3) the `z − radius = −radius < 0` term trips `validate_point`'s `c_val < 0` guard and
  would raise "z-min". It works only because the XML-emission path (`.xml.getCC3DXMLElementString()`)
  never calls `validate()`. Recorded in `surprises:`.
- **Invalid-by-default sentinels:** constructor `width=0`/`radius=0` construct fine but `check_dict`
  rejects `<1` — only at validate() time. In `from_xml`, Gap is optional, Width/Radius required.

**Could NOT establish** (all compiled-C++, not readable in this env, and stated as such in the
entry): the exact clip predicate (site-in-sphere vs tile-center-in-sphere), the RNG draw mechanics
for random type assignment (asserted seed-dependent by CC3D convention, not verified), and how
overlapping multi-region blobs resolve painting order. **No paper text available** — `paper_section`
names the chapter's home for initializers plus checkable *library* anchors (PyCoreSpecs.rst:294/299,
the generator comment at L1235); I invented no page/figure. This mechanism is NOT one of the six
with reference runs, so there is no measured evidence — source-read only.

## blobinitializer — normalized

**Verdict: `out_of_scope`.** BlobInitializer runs once at MCS 0 and CONSTRUCTS the initial
partition (the sets themselves) from an empty lattice + a geometric region + a type palette.
Plexus operators are per-step maps returning a delta over state that already exists; the initial
partition is supplied by config/seeding, not by any operator. So this is IC/framework mechanics
for establishing the starting state — out of scope for the OPERATOR algebra whose completeness
we measure. It is one of a family (Blob / Uniform / PIF initializers), three interchangeable ways
to build the same starting partition. I still filled a descriptive `seed` contract
(structural/growth) so the ledger has the typed shape it WOULD take if IC construction were
in-scope — counted as ONE `seed` with those three as implementations, never three separate `new`s.

**Strongest argument AGAINST (i.e. for `new`):** this is the *only* mechanism in the campaign so
far that genuinely writes state and creates sets — literally "how cells come to exist." The
registered algebra has no way to construct sets de-novo: `cell_divide` splits an existing parent
(conserving material, one→two), and nothing seeds a population out of Medium with no parent. If a
cell-based framework must express "instantiate the initial partition," declaring it out_of_scope
hides the single most load-bearing structural gap, and the honest verdict is `new` (a `seed`
contract, Uniform/PIF as co-implementations). I land on out_of_scope because Plexus
*architecturally* seeds initial state via configuration rather than an operator, so there is no
operator here to alias/refine/introduce — but the line between "IC construction is config" and
"IC construction is a missing operator" is the genuine judgement call, and a reasonable normalizer
could put it the other way.


---

## boundarypixeltracker

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


---

## boxwatcher

<!-- boxwatcher -- append below; the driver merges this into campaign/analysis.md -->

## boxwatcher (BoxWatcherSteppable) — excavated

Read `PyCoreSpecs.py:L5490-5519` (the whole class + base `_PyCoreSteppableSpecs` at L512-552),
and the two CC3DML generators that actually emit the steppable's parameters
(`twedit5/.../CC3DMLGeneratorBase.py:L1282-1296`, `.../CC3DProject/CC3DXMLGenerator.py:L1064-1077`).
The compiled core is not readable here and there is NO BoxWatcher string anywhere under `cc3d/cpp`,
so the physics is reconstructed from the docstring, not verified.

- **It is neither an energy term nor a state update.** BoxWatcher traces the minimal bounding box
  of all non-medium cells, pads it by per-axis margins, and hands that box to the Potts solver so
  the pixel-copy sweep only samples sites inside it. It writes the SAMPLER's spatial support, not
  a cell, a field, or a dE. This is the first mechanism I've hit that falls outside the campaign's
  two expected buckets — worth flagging for the normalizer: it may have no Plexus operator analogue
  (nothing returns a delta the engine integrates).
- **The PyCoreSpecs wrapper is parameterless AND lossy.** `xml` emits only
  `<Steppable Type="BoxWatcher"/>`; `from_xml` locates the element then returns a bare `cls()`, so
  XMargin/YMargin/ZMargin present in loaded CC3DML are silently dropped on round-trip. Yet the
  twedit generators emit all three at 7. Genuine source-vs-source discrepancy: the two Python-facing
  paths disagree on whether margins are exposed at all.
- **"May have no effect for parallel version"** (docstring): the optimization is bypassed under the
  parallel sweep, so the realized effect is implementation/threading-dependent, not a model property.
- **The margin default 7 is a magic constant** that lives only in the twedit generators — a
  reimplementer following PyCoreSpecs inherits the (unknown) compiled default instead.

**Could NOT establish:** the exact clamp form at lattice edges, the recompute cadence (per MCS? per
N?), whether the box shrinks as well as grows, the C++ default margins when unset, and — most
important — whether a fixed-seed *serial* run is truly bit-identical with vs without BoxWatcher
(restricting the sampler's support changes the attempt/RNG sequence; I could not confirm it is
behavior-preserving). All of these sit in the compiled core, which is not importable in this
environment. No evidence run exists for this mechanism (not among the six ablated), and no paper
page was read — `paper_section` records that BoxWatcher is a computational optimization absent from
the Swat et al. text, with the generator docstring as the sole anchor.

## boxwatcher — normalized

**Verdict: `out_of_scope`.** BoxWatcher reduces the lattice-site extents of all non-medium cells
to one axis-aligned bounding box (plus fixed per-axis margins) and hands that box to the Potts
solver so pixel-copy attempts are drawn only from the occupied region. It writes the *sampler's*
spatial support — no cell, no field, no energy delta — and is meant to be dynamics-preserving; the
docstring's own rationale is "May speed up calculations." That is framework mechanics with no
biological content, and promoting it (as `new`, or as an `aggregate` alias) would inflate the
saturation yield with plumbing — the one thing this measurement must not do. I recorded a contract
shape (`kind: aggregate`, `set→global` reduction) only to document the typed operation; its
`family: hierarchy` is a forced fit, not a claim that BoxWatcher builds a hierarchy.

**Strongest argument AGAINST out_of_scope:** the typed shape *is* a genuine aggregate — a reduction
over the whole cell set to a global quantity — and Plexus already registers
`aggregate(aggregate/hierarchy)`. If the campaign measured typed *shapes* rather than biological
*meanings*, BoxWatcher would be a legitimate second sighting of `aggregate` (`alias`, or
`implementation_of: aggregate`). I reject that because Plexus `aggregate` coarse-grains child
agents into a *parent biological node* whose output the engine integrates; BoxWatcher's reduced
quantity is a solver sampling window fed back to the numerics, never a modelled entity, and it
emits no delta. Counting it as `aggregate` would let any min/max reduction anywhere in a
framework's bookkeeping masquerade as biology — exactly the naming-habit noise the ledger exists
to exclude. Shape-match noted; out_of_scope stands.


---

## cell_as_lattice_domain

<!-- cell_as_lattice_domain -- append below; the driver merges this into campaign/analysis.md -->

# cell_as_lattice_domain — normalizer note

**Verdict: `new`** (contract `occupy`, aggregate/hierarchy). CC3D's cell is a set of lattice sites
sharing an id; V, S, COM are derived from the label field sigma. The Plexus algebra's agents are
points, so it has no primitive for an extended, deformable lattice domain with a first-class exact
surface. Placed at aggregate/hierarchy on purpose — same kind/family as the registered `aggregate`
(Centroid) — to keep the alias tension honest rather than hide the representation in a distant corner.

**Strongest argument AGAINST (i.e. this is really a `refinement` of `aggregate`, not `new`):** the
registered `aggregate`/Centroid contract *already* computes a cell's position as the occupancy-weighted
centroid of its contained children and writes it as a derived readout — that is precisely COM = mean of
sites. If you read the site→cell label field sigma as just another (dynamic) instance of aggregate's
`parent` containment map, and treat V = |sites| as a trivial count-reduction over the same members, then
two of the three derived quantities fall straight out of the existing contract; nothing about aggregate's
*operation* changed, only how the member set is stored (a field partition vs a member list). A reviewer
could fairly call that a widening of aggregate's `set`/`reads`, not a new contract. My rebuttal is that
(a) surface S is an *inter-parent* boundary count that aggregate cannot express at all — it reads
neighbouring cells' labels, not a parent's own children — and (b) swapping aggregate's member set from
persistent point children to a mutable partition of a shared lattice is a substrate change that breaks
every current point-agent user, so it is different-in-kind, not wider. But the COM overlap is genuine and
I record it in `why:` rather than pretend the two contracts are disjoint.


---

## celltype

<!-- celltype -- append below; the driver merges this into campaign/analysis.md -->

# celltype (CellTypePlugin) — NORMALIZER note

**Verdict: out_of_scope.** CellTypePlugin declares the type PALETTE — (name, id, frozen)
triples with (Medium, 0) fixed — and nothing dynamical: no energy term, no delta, and it does
not even assign tau(sigma) per cell (the initializers do, randomly/seed-dependently). It is the
type SCHEMA that downstream type-keyed operators (contact, chemotaxis, per-type targets) read.
The promoted Plexus algebra already presupposes `type` as a first-class per-agent attribute
(selectors `agent[type=a]`), so there is no operator to alias and no per-step operator to widen.
Recorded the honest forced-fit contract (`differentiate`, structural/hierarchy) only to document
the labelling shape.

**Strongest argument against out_of_scope:** cell type is not decoration — it is differentiation
state, the most biological attribute a cell has, and freezing a type (the Freeze flag) is a
genuine dynamical effect (immovable cells excluded from the Potts sweep). One could argue that a
mechanism whose whole job is to establish cell fate deserves a contract — the jax-morph proposal
`regulate` (fate-state update) is the natural home, making this a second sighting of `regulate`
rather than out-of-scope plumbing. I reject that because CellTypePlugin never CHANGES a type: it
enumerates the legal ones and stops. Putting `differentiate`/`regulate` behind a static
declaration would mint a contract for a dynamic this code does not perform — the mirror-image of
the error the campaign guards against. If a CC3D mechanism that mutates tau(sigma) at runtime
surfaces, that is where `regulate` should be earned; the Freeze mechanic, meanwhile, is a
boundary condition bundled into the schema, not evidence of a typing *process*.


---

## centerofmass

<!-- centerofmass -- append below; the driver merges this into campaign/analysis.md -->

## centerofmass (CenterOfMassPlugin) -- inspected

**What I read.** PyCoreSpecs.py:L3056 (the whole spec class + base `_PyCorePluginSpecs` at
L471 and its `generate_header`/`depends_on` at L305), the SWIG `CellG` struct in
cpp/CompuCell.py:L518-544 (`xCM/yCM/zCM`, `xCOM/yCOM/zCOM`, `xCOMPrev/yCOMPrev/zCOMPrev`) and
its read-only setters at L595-628, the sibling `MomentOfInertiaPlugin` at L5350, the
"CenterOfMassBased" ExternalPotential algorithm at L2939, and the code-ref doc stub
(PyCoreSpecs.rst:169). The compiled physics is in the un-shipped C++ core -- not readable here.

**What it does to state.** It publishes each cell's centre of mass and nothing else. It is a
CellGChangeWatcher (lattice monitor), NOT an energy term: it returns no delta-E and never tilts
the acceptance probability of a proposed pixel copy. Per accepted spin-flip it carries the
cell's first moment `xCM = sum of site coordinates` incrementally (+/- the changed site) and
stores `xCOM = xCM / volume`. `xCOMPrev` keeps a one-MCS history so consumers can form cell
displacement/velocity. A run with it loaded is bit-identical to one without; its whole purpose
is that COM-based chemotaxis, CenterOfMassBased external potential, mitosis split orientation,
and connectivity READ `xCOM`. In Plexus terms this smells like a REDUCE over a site-set that
publishes a derived field, not an operator returning a force/energy delta.

**Surprises / gotchas for a reimplementer.** (1) Monitor, not update -- forcing it into
read/write energy language makes a phantom no-op. (2) `xCM` (raw moment) and `xCOM`
(= moment/volume) are DIFFERENT CellG fields; dividing a stored sum by a stale volume is the
classic wrong COM; `xCOM` is derived + read-only from Python. (3) Periodic-boundary correction:
a cell straddling a wrap seam needs coordinates unwrapped against the BoundaryStrategy before
summing, then `xCOM` re-wrapped into the lattice -- the one thing most likely to be gotten
wrong. (4) Unguarded prerequisite: the spec makes CenterOfMass depend only on
`[PottsCore, CellTypePlugin]`, and NOTHING declares a `depends_on` CenterOfMass, yet COM-based
consumers silently need it -- omit it and spec validation still passes but COM is zero/stale.
(5) It does NOT compute the inertia tensor -- that is the separate MomentOfInertia plugin.

**What I could NOT establish (un-read compiled C++).** The exact periodic-boundary unwrap
arithmetic; whether `xCOMPrev` is refreshed by this plugin's per-MCS `step()` or elsewhere; and
the precise ordering of the gaining-vs-losing-cell moment updates within one accepted copy. The
incremental-watcher model is inferred from CC3D's CellGChangeWatcher architecture and the CellG
data members, not read line-by-line. No evidence run exists (not among evidence.py's six), so
all of the above is source-only -- no measured ablation backs it.

## centerofmass -- normalized

**Verdict: alias of `aggregate`** (implementation_of: aggregate). CC3D's CenterOfMass is the
centroid reduction applied per cell: reduce a cell's owned lattice sites (children) onto the
cell (parent) by the occupancy-weighted mean of their positions, publishing that mean as the
cell's read-only derived position. Same typed shape as the registered `aggregate` centroid
(aggregate.py: children -> parent along the containment map, reads child `pos` + occupancy as
the weight, writes parent `pos`, returns no integrable delta); with occupancy 1 per site the
weighted mean degenerates to CC3D's plain coordinate mean, so no signature field widens.
Incremental accumulation and the raw first moment `xCM` are implementation/intermediate detail;
`xCOMPrev` is side bookkeeping (a cell-velocity readout is a separate concern). The nearest
OTHER contract, `momentofinertia`, is the sibling second-moment reduction -- a different
aggregate output, not this one.

**Strongest argument against (that this is a `refinement`, not an alias):** the periodic-boundary
unwrap. CC3D unwraps coordinates against the BoundaryStrategy before summing and re-wraps `xCOM`
into [0, L); the Plexus `aggregate` does a naive weighted mean that would place a seam-straddling
cell's COM near the lattice centre. One could argue the centroid contract must WIDEN to carry a
boundary/topology parameter to admit periodic domains, making this a refinement that breaks the
current topology-blind implementation. I booked it alias because that correction changes the
reduction's *domain*, not its typed signature (inputs/outputs/reads/writes/maps untouched) -- an
implementation obligation of the same contract on a wrapped lattice. But honestly, "silent
wrong/stale COM on a periodic lattice" is exactly the unspecified precondition a refinement is
meant to force into the open, and a reviewer could book it that way.


---

## chemotaxis

<!-- chemotaxis -- append below; the driver merges this into campaign/analysis.md -->

# chemotaxis

**Verdict: alias of `chemotax`** (implementation_of: chemotax). CC3D's ChemotaxisPlugin is
gradient-following with a per-type sensitivity lambda_chemo — the exact biology of the registered
Keller-Segel `chemotax` contract. It is a second implementation, realized as a Potts energy term
dE = -lambda*(c(x_new)-c(x_old)) that biases pixel-copy acceptance, rather than a velocity delta
the engine integrates.

**Strongest argument against:** if you take `writes` seriously as part of the typed signature, these
are NOT the same contract. Plexus `chemotax` writes a velocity on `pos`; CC3D's plugin writes
nothing and only enters a Hamiltonian. That gap could justify a `refinement` (widen `chemotax` with a
third `emit: energy` / acceptance-bias routing beside `velocity` and `mpm_acceleration`) or even a
`new` energy-term kind the algebra lacks. Calling it an alias risks burying the single most
interesting thing CC3D contributes here — that gradient following is a probability bias on a discrete
move, not a drift. I keep the verdict at alias because the biology is identical and the
energy-vs-update divide is a systemic property of the Potts paradigm shared by nearly every plugin;
minting it per-mechanism would inflate the yield and destroy the measurement. The routing gap is
flagged in `state_io` and `why` so it is recorded once, not counted many times.


---

## connectivity

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


---

## connectivityglobal

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


---

## contact

<!-- contact -- append below; the driver merges this into campaign/analysis.md -->

# contact (ContactPlugin, order 12)

**Verdict: `new` vs the frozen baseline, `implementation_of: adhere`.** This is the CANONICAL
`adhere` implementation and the base class of the whole family — ContactLocalFlex (L3339) and
ContactInternal (L3350) subclass it, AdhesionFlex is its molecule-density generalisation. Energy
`E = sum over cross-boundary site pairs of J(tau_i,tau_j)(1-delta)` with J a static, global,
symmetric per-TYPE matrix is textbook differential adhesion (Steinberg); no registered contract
covers a Potts boundary-energy term. It is the fourth `adhere` sighting in this CC3D atlas and a
second-repository sighting of the jax-morph `adhere`; implementation_of keeps the ledger from
double-counting. It has a real REFERENCE ablation (log/atlas_cc3d/contact_adhesion): heterotypic
boundary 290→167 with differential energy, 290→388 with equal energies — the sign reverses.

**Strongest argument AGAINST this verdict.** The honest alternative is `refinement`, not a clean
`new`/implementation_of. Every `adhere` sighting so far is a Potts ENERGY TERM that returns dE and
writes nothing, and this canonical one most sharply so. The jax-morph `adhere` these are pinned to
was proposed from a force-based, particle world where adhesion returns an integrable force and
writes `position`. Calling both "the same contract" quietly assumes a single `adhere` can host two
incompatible OUTPUT types (energy-bias vs force) and two incompatible `set`s (site-set vs point).
If it cannot, the correct move is to WIDEN `adhere` — a `refinement` whose signature changes
`outputs` from force to {force | contact_energy} and `set` from point to {point | cell-as-site-set},
which is a breaking change for its force-based users. I chose implementation_of because `adhere` is
still UNPROMOTED (nothing to break yet) and the biology — type-dependent boundary cost drives
sorting — is genuinely one verb across both worlds. But that is a bet that the algebra carries the
output-type split BELOW the contract line; if promotion forces the split up to the signature, this
entry should be re-read as the first evidence for widening `adhere`, not an implementation of it.
That tension is recorded in the entry's `why:` rather than resolved.


---

## contactinternal

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


---

## contactlocalflex

<!-- contactlocalflex -- append below; the driver merges this into campaign/analysis.md -->

## ContactLocalFlex (order 11) -- read at source

**What I read.** `PyCoreSpecs.py:L3339` `ContactLocalFlexPlugin(ContactPlugin, _PyCoreSteerableInterface)`
-- a *bare* subclass of `ContactPlugin` (L3141): zero extra fields, same `__init__`
(`neighbor_order`, `depth`, `weight_energy_by_distance`, `*ContactEnergyParameter`), same `xml`.
The docstring is one line: "A steerable version of ContactPlugin." All the real difference is in
the compiled core (`cpp/CompuCell.py` L6379-6591): base class is `CellGChangeWatcher`, and it owns
a per-cell attribute `ContactLocalFlexDataContainer{contactDataContainer, localDefaultContactEnergies}`
plus methods `contactEnergy` / `defaultContactEnergy` / `setContactEnergy`.

**The finding.** The energy functional is *identical* to plain Contact:
`E = sum_{nbr pairs} J(sigma_i,sigma_j)(1 - delta(sigma_i,sigma_j))`. The ONLY difference is where
`J` lives: plain Contact reads a global cell-TYPE matrix; LocalFlex reads a PER-CELL container.
The spec's type-pair energies merely *seed* each cell's default table; `setContactEnergy()` then lets
a steppable override `J` for an individual cell (or cell-cell pair) at runtime -- that runtime,
per-cell steerability is the entire point of "local flex". `ContactInternal` (L3350) is the same
class again, just retargeted to compartments within a cluster cell.

**What surprised me.** The "flex" is completely invisible from the Python spec -- reading only
PyCoreSpecs you would conclude it is Contact with a different name. Nothing in the Python API sets a
per-cell `J`; the mechanism only exists once you drive `setContactEnergy` from C++/a steppable.

**What I did NOT establish.** (1) No paper text in this environment -- I could not verify a page/eq
number for the contact-energy term; the anchor is the source, not the paper. (2) I read data-structure
names and method signatures from the SWIG wrapper, not the C++ `changeEnergy` body -- the exact
per-cell lookup/fallback order (per-neighbour map vs `localDefaultContactEnergies` vs type default)
is inferred from the member names, not read line-by-line. (3) No ablation/evidence run exists for
LocalFlex specifically; `contact_adhesion` evidence covers plain Contact, so the measured effect of
per-cell overriding here is unquantified. (4) Whether `weight_energy_by_distance` divides by Euclidean
distance vs a tabulated weight is assumed from the name/parent, not confirmed in the core.

**Re-read addendum.** `_PyCoreSteerableInterface` in the subclass declaration is redundant --
`ContactPlugin` (L3141) already inherits it. So at the Python level the *only* thing that changes
between plain Contact and LocalFlex is `name`/`registered_name`; `registered_name="ContactLocalFlex"`
is the sole hook that binds the spec to the different C++ plugin holding the per-cell J container.

**Normalizer verdict.** `new` (frozen baseline has no cell-cell contact/adhesion energy) with
`implementation_of: adhere` -- a THIRD sighting of `adhere`, after jax-morph proposed it and
AdhesionFlex logged the first CC3D sighting. The energy is *identical* to plain Contact, the
canonical `adhere` implementation; ContactLocalFlex, plain Contact, and AdhesionFlex are three
interchangeable implementations of the one pure-adhesion contract, which is exactly the many-impls
shape a mature framework should show. Contract mirrors AdhesionFlex's: `set: cell` (a set of lattice
sites), output is a scalar `contact_energy` dE biasing Metropolis acceptance (writes NOTHING to the
lattice, rule 8), interaction counted over cross-boundary site-pairs rather than centre distance.

**Strongest argument AGAINST.** One could push for `alias: adhere` rather than a distinct
implementation -- ContactLocalFlex adds *zero* to the energy functional over plain Contact, so is
the per-cell J storage really a different implementation or just a runtime knob on the same one? If
you weight only the Hamiltonian, plain Contact and LocalFlex are the SAME implementation and this
entry is a duplicate the ledger should collapse. I keep them distinct because the per-cell/per-pair
J container and `setContactEnergy` steering change the READ side of the contract (a per-cell
adhesion attribute, heritable/steerable like AdhesionFlex's density vector) even though the write
side is unchanged -- but that is a genuinely arguable line, and if the ledger counts implementations
by energy form alone, LocalFlex folds into Contact and only `adhere` (once) survives either way.


---

## curvature

<!-- curvature -- append below; the driver merges this into campaign/analysis.md -->

## curvature (EXCAVATOR)

Read the actual C++ (found on disk at `papers/CompuCell3D/.../plugins/Curvature/CurvaturePlugin.cpp`,
`.h`, `CurvatureTracker.h`), cross-checked against the installed binary `libCC3DCurvature.so`
(demangled symbols) and the shipped `Curvature_test_generate` XML. Web fetch/search were blocked,
so the on-disk source is the sole evidence — but it is the real thing, not a guess.

What it does: a **Cellular-Potts energy term** that penalizes bending of chains of compartmental
cells linked by junctions it maintains itself. Per triple of consecutive linked cells it adds
`lambda * kappa` where `kappa` is the **Menger curvature (1/circumradius)** of the three centers
of mass — zero when straight. `changeEnergy` returns the dE over affected triples (COMs recomputed
+/-1 volume); it writes no cell state. It ALSO grows a junction graph (activation_energy biases a
bond-forming move; the bond is committed in the `field3DChange` watcher on accepted moves).

Surprises worth the record:
- The function `calculateInverseCurvatureSquare` is a **misnomer** — it returns the plain curvature
  (2*sin(theta)/|chord| = 1/R_circ), neither inverted nor squared. Trust the name and you invert
  the physics.
- Half the plugin is **dead code**: `potentialFunction` (harmonic spring), `targetDistance`,
  `maxDistance`, and all three `diffEnergy*` (return 0). Only `lambda_curve` and `activation_energy`
  matter. Clearly cloned from FocalPointPlasticity and left half-gutted.
- Junctions are the plugin's OWN state (within-cluster only), NOT FPP's — the two are separate
  plugins that merely co-occur in the demo. So Curvature is a **hybrid**: energy term + stateful
  bond-graph watcher. This is the interesting bit for the algebra: `changeEnergy` fits "return a
  delta the engine integrates," but `field3DChange` is a genuine write-on-accept side effect that
  the pure-energy framing does not cover.
- Apparent BUG: the volume-1->0 branch adds three curvature terms WITHOUT the `lambda` factor
  (L674/L682/L687). A faithful port must reproduce it to match the oracle.

Could NOT establish: no oracle/ablation run exists for curvature under `log/atlas_cc3d/` (not one
of the six evidenced mechanisms), so the dynamical magnitudes (does lambda=1000 in the demo
actually straighten the chain, how strong is the activation-energy bias vs Temperature=10) are
unmeasured here — inferred from the formula only. I did not confirm the exact set of "affected
triples" is complete for every geometry; I read the five triple-blocks in each branch but did not
prove they exhaust all triples touched by a COM shift. Left `verdict`/`contract` unset (normalizer's
call); set `status: inspected`.

## curvature — NORMALIZED

Verdict `new`, contract **`stiffen`** (lateral/interaction, set cell) — a BENDING STIFFNESS: an
angular energy over an ORDERED chain of junction-linked compartmental cells that penalises
curvature (Menger `kappa = 2 sin(theta)/|chord| = 1/R_circ`, linear, zero for straight) and drives
a linked filament toward straight. `implementation_of` left null. Consistent with the excavator's
read, this is a HYBRID: the primary term is the bending stiffness (a Potts `changeEnergy`, writes
nothing); a SECOND stateful half GROWS the junction graph (biases a bond-forming move by
`activation_energy`, commits the bond in the accepted-move watcher) — a topology rewire near
[[radius_graph]] that builds the chain the stiffness acts on. That rewire half is recorded in
writes:/maps:/surprises, not the verdict.

STRONGEST ARGUMENT AGAINST: this should be `implementation_of: stillinger_weber`, not a new
contract. SW is the registry's only three-body angular term, `(cos - cos0)^2`; set `cos0 = -1`
(theta0 = 180°) and SW *is* a straightness/bending penalty — same "angular energy, preferred
angle" idea, and the Menger-vs-cosine functional difference is "just an implementation." Rebuttal
(why I still chose `new`): SW builds its OWN isotropic min-image neighbour list and sums over ALL
geometric triples within a cutoff — it has no ordered backbone. Filament bending requires selecting
only CONSECUTIVE triples along a MAINTAINED 1D bond graph (left/mid/right, ≤3 hops each side);
SW-with-cos0=-1 on a blob would penalise every bent geometric triple, not straighten a defined
chain. Covering Curvature forces SW to take an EXTERNAL ordered bond graph it does not maintain — a
new required input that breaks every current SW user (mW water, silicon) and changes its output
from an integrated Newtonian force to a Metropolis accept/reject energy. That is a signature-
breaking refinement, not a free implementation slot, so `stiffen` is the honest call. (If a
reviewer rules SW's contract already means "any three-body angular stiffness," the fallback is
`implementation_of: stillinger_weber` with `cos0=-1` — and the disagreement is exactly the
maintained-topology axis this campaign exists to surface.)

Not a second sighting of any jax-morph-proposed contract (adhere, agitate, apoptose, mechanosense,
morphogen, regulate, relax, reorient) — none is an angular/bending stiffness. All excavator caveats
(no ablation run, no paper text, C++ read-not-run) carry forward unchanged.


---

## diffusionsolverfe

<!-- diffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# DiffusionSolverFE (order 14)

**What I read.** `PyCoreSpecs.py:L6242` (`DiffusionSolverFE`) plus its child specs:
`DiffusionSolverFEDiffusionData` (L6070), `...SecretionData` (L6160), and the shared bases
`_PDEDiffusionDataSpecs` (L554), `SecretionParameters` (L597), `PDEBoundaryConditions` (L820),
`_PDESolverFieldSpecs` (L1057), `_PDESolverSpecs` (L1150). The class at L6242 is a thin CC3DML
*emitter* — all physical parameters live on the child DiffusionData/SecretionData/BoundaryConditions
specs, and the integrator is compiled. I grounded the algorithm on the in-tree guide strings:
`diffusion_solvers_descr.py:7` ("Uses Forward Euler method and handles moving boundary conditions")
and `CC3DXMLGenerator.py:884` (FTCS stability limits D>0.16 3D / 0.25 2D at DeltaX=DeltaT=1).

**The mechanism.** One explicit forward-Euler step of `dc/dt = D grad^2 c - lambda c + S` on the
CPM pixel lattice, per MCS. A real field->field WRITE (overwrites c in place) — unlike the Potts
energy-term plugins in this atlas that return nothing and only bias pixel-copy acceptance.

**What surprised me.**
- D and lambda are *cell-type-indexed and spatially heterogeneous* — read from the moving CPM cell
  field at each pixel. Plexus `diffuse`/`decay` carry a single global scalar rate; no set coupling.
- "Secretion" is three semantics under one name: additive rate, ConstantConcentration (a Dirichlet
  *clamp*, i.e. a set not an add), and SecretionOnContact.
- Default BC is Value 0.0 on every face = absorbing sink, not no-flux — silent field leakage.
- FTCS is only conditionally stable, yet the DiffusionSolverFE python spec exposes NO setter for
  DeltaX/DeltaT/ExtraTimesPerMCS — high D blows up silently with no knob at this API layer.

**What I could NOT establish (do not treat as known):**
- The exact neighbour stencil (I assumed von Neumann 4/6) and how D is combined across a cell-type
  interface (destination-pixel vs average) — both are in the compiled core; I did not read the `.so`.
- Whether decay/secretion are applied in the same sub-step as diffusion or in a fixed operator-split
  order; the "one step" form in `equations:` is the standard FTCS reading, not verified against core.
- Whether ExtraTimesPerMCS sub-cycling is auto-chosen by the core or must be user-set — the guide
  implies user-set, but the actual default behaviour is compiled.

**UPDATE — resolved from the OpenCL kernel.** After the notes above, I read the shipped GPU
kernel `cc3d/cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (`uniDiff` L821-L1057, plus the
`secrete*` kernels L192-L346), whose comments cross-reference the CPU code. This upgrades three of
the "could not establish" items:
- **Interface diffusion IS resolved.** The stencil is NOT a single central-D Laplacian. It is two
  half-sums: `isoSum = D_i*(SUM c_j - N c_i)/2` (centre cell's D) plus `varSum = SUM D_j*(c_j-c_i)/2`
  (each *neighbour's own* D), added. This symmetrises the flux across a cell-type interface and only
  collapses to `D*(SUM - N c)` for uniform D. I put the exact form in `equations:`.
- **Operator split IS resolved.** Decay is inside the diffusion pass as a `(1 - dt*lambda_i)*c_i`
  factor; secretion is a *separate* sweep (order-dependent). ConstantConcentration is a hard
  re-pin `c := value` each step; plain Secretion also supports a relative/max uptake sink.
- **Still open:** whether the compiled CPU default path (`gpu=False`) is byte-identical to this
  kernel — I could not read the `.so`; the kernel comments claim equivalence but I did not verify.
  ExtraTimesPerMCS default behaviour is still compiled and unread.

**Adjacent language (for the normalizer, not a verdict):** Plexus already has `diffuse`
(finite_difference + spectral), `decay`, `deposit`, `prescribed_field`, `scalar_field`. CC3D fuses
diffusion+decay+source+BC+cell-type-coupling+mass-compensation into ONE solver; the open question is
whether that fusion and its Potts-field coupling are expressible by composing the existing operators.

---

**NORMALIZER — verdict: refinement of `diffuse`** (implementation_of: `diffuse`; the FTCS/
forward-Euler scheme is a third implementation beside finite_difference and spectral). The
mechanism as a whole is a *composite* of three registered contracts — `diffuse` (transport) +
`decay` (turnover) + `deposit` (secretion source): in Plexus you would write it as diffuse→decay→
deposit on one field. That composability is a positive saturation signal — CC3D's flagship PDE
solver adds no new biological verb. It is not a clean alias because its defining capability,
spatially-heterogeneous D_i/λ_i keyed to the moving cell-type lattice, forces `diffuse`/`decay` to
widen `rate` from scalar to a set-coupled coefficient field; that widening breaks the central-D
box-blur (must use the two-half-sum face-average stencil or compute heterotypic diffusion wrong)
and the spectral `exp(-Dk²dt)` implementation (assumes constant D), so it is load-bearing, not a
knob. Two smaller widenings are flagged in `why:` but not elected as the primary verdict: per-face
boundary conditions (fields carry only a `periodic` flag today) and deposit's `mode: add|set` for
the ConstantConcentration Dirichlet clamp.

**Strongest argument AGAINST (that it is `alias`, not `refinement`):** the biology is plain
Fickian diffusion, which `diffuse` already names in full; making D spatial is arguably "a
parameter made spatial" — the same reasoning by which the sibling Chemotaxis entry ruled CC3D's
per-type λ and Michaelis–Menten saturation to be *response-curve choices, not new contracts* and
stayed `alias`. If a per-type coefficient is parameterization rather than signature, then this is
an alias and I am inflating the yield by one refinement. Rebuttal: chemotax *already* reads
`cell.type`, so per-type λ was in its signature; `diffuse` is a pure field→field op with
`set: field` and **no set input at all**, and its variable-diffusion stencil is dead code unless
D genuinely varies in space — so coupling D to the cell lattice is a real signature change. This
is nonetheless the one call in the entry a reasonable reviewer could downgrade to `alias`.


---

## energy_sum_composition

<!-- energy_sum_composition -- append below; the driver merges this into campaign/analysis.md -->

**Verdict: out_of_scope.** The Hamiltonian-as-sum-of-plugin-terms is not an operator over any set or
field; it is the framework's *composition law* — the meta-rule that every enabled mechanism contributes
an additive energy term and the terms interact only through the summed `dE` the acceptance test reads. It
has no biological content, no parameters, and writes no state. There is nothing to alias or widen because
Plexus's own composition law — operator splitting, where each operator returns a delta the engine
applies/integrates in sequence (the Lie-Trotter split the surprise names) — lives at the same
architectural layer and is likewise *not* a member of the operator vocabulary in `src/plexus/operators/`.
The campaign counts contracts over sets/fields; a calling convention is not one, so I record the finding
loudly (energy summation vs operator splitting = two answers to how mechanisms compose) rather than
inflate the yield with a non-vocabulary item. This is the same cut `celltype` made (a schema is not a
per-step operator), deliberately *not* the cut `cell_as_lattice_domain` made (that earned `new` because
the extended lattice domain is a substrate *thing operators read/write over*, which the point-agent
algebra genuinely cannot represent).

**The strongest argument against.** Energy summation is arguably a genuine *gap* the algebra cannot
express, not mere plumbing: operator splitting applies each delta unconditionally and in order, so it
cannot represent CC3D's simultaneous, order-independent, within-step trade-off resolved by one stochastic
accept/reject (a copy that raises volume energy but lowers contact energy more still gets accepted). By
that reading the algebra *is* incomplete on the composition axis, and burying that under `out_of_scope`
hides exactly the incompleteness this exercise exists to surface — the parallel to `cell_as_lattice_domain`
(also architectural, yet `new`) is real. My rebuttal is that the incompleteness is at the *interpreter/
engine* layer, not the operator *vocabulary*, and the saturation curve measures the vocabulary; forcing a
composition scheme into the contract count would measure our engine's semantics, not the biological
language. But the argument is strong enough that if a future promotion ever makes "how operators compose"
a first-class, parameterizable choice inside the algebra, this entry — together with `metropolis_acceptance`
— is where the energy-summation composition mode should be re-litigated as `new`.


---

## externalpotential

<!-- externalpotential -- append below; the driver merges this into campaign/analysis.md -->

# ExternalPotential

**Verdict: `alias` of `sediment`** (registered lateral/motion). ExternalPotential attaches a
constant force vector `lambda` to each cell type and, per attempted pixel copy, contributes
`dE = -lambda . (x_new - x_old)` so the cell drifts persistently along `lambda`. That is exactly
`sediment` -- whose own docstring calls itself "a per-agent constant directional drift ... the
type-selectable sibling of `gravity`", instantiated per type via `at: 'agent[type=a]' gy: -0.1`.
The plugin's per-type `ExternalPotentialParameter(cell_type, x, y, z)` is the same construct; its
global `lambda_x/y/z` mode is the type-blind degenerate case, which is the sibling registered
contract `gravity`. One CC3D plugin thus spans the pair Plexus splits into `sediment` (per-type)
and `gravity` (uniform); I alias to `sediment` because per-type configurability is the defining,
non-degenerate feature and `sediment` subsumes the uniform case as equal params. The
implementations differ only in mechanics (Plexus `sediment` returns a velocity delta the engine
integrates; CC3D returns a Metropolis acceptance-bias `dE` and writes nothing), not in contract.

**Strongest argument against.** The default `com_based=False` mode applies the force PER PIXEL, not
to the centre of mass, so a strong field can *spread or reshape* a cell as it drives it -- a shape
effect `sediment` (a pure COM translation) simply cannot express. If that per-pixel deformation
were considered part of the mechanism's biology, this would not be a clean alias: it would demand
either a `refinement` widening `sediment` with a "distributed vs COM" application mode, or splitting
off a shape-coupled body-force contract. I rejected that because (a) `com_based=True` recovers the
exact COM-drift semantics, making the shape effect an *optional* CPM artefact of the site-set
representation rather than the intent, and (b) the measured ablation (population mean x 32 -> 57.6
driven vs 32 -> 32.2 undriven) is pure directed drift of the cell centres -- the biology being
exercised is a body force, which is `sediment`. The per-pixel spreading is recorded as a surprise,
not folded into the signature.


---

## focalpointplasticity

<!-- focalpointplasticity -- append below; the driver merges this into campaign/analysis.md -->

## FocalPointPlasticity (order 16) — excavated 2026-08-02

Read: `PyCoreSpecs.py` L4512-4908 — the Python spec layer (`LinkConstituentLaw`,
`FocalPointPlasticityParameters`, `FocalPointPlasticityPlugin`), the twedit ML generator
(`CC3DMLGeneratorBase.py` L859-915), and the shipped test XML
(`tests/.../FocalPointPlasticity.xml`). The compiled FPP core is NOT readable in this env, so the
energy form is inferred, not read line-by-line — flagged in the entry.

What it does: keeps a dynamic set of pairwise **junctions** (focal-point links) between cells of a
type pair, and adds per link a spring energy `lambda*(d - target_distance)^2` on the distance `d`
between the two cells' **centers of mass**. Links form on contact (both cells below max_junctions),
pay a one-time ActivationEnergy at formation, and break when `d > max_distance`. So it is an energy
term that ALSO carries persistent inter-cell state — the interesting bit for the algebra: not a
stateless Potts plugin, it mutates a link registry between steps.

Surprised me: (1) `d` is CoM-to-CoM, long-range — target 7 / break 20 on volume-25 cells, i.e. links
span well beyond cell contact. (2) ActivationEnergy is XML-only, explicitly NOT runtime-steerable
even though targetDistance/lambda/maxDistance are (generator warning L882-884) — a one-time
formation threshold, not a per-step energy. (3) default energy comes from `LinkConstituentLaw`, which
is user-overridable with an arbitrary formula string over Lambda/Length/TargetLength.

Could NOT determine: the exact compiled energy/lifecycle code (inferred from the default
LinkConstituentLaw formula + Swat et al.); the full variable set bindable in a custom
LinkConstituentLaw and how the core parses/evaluates the formula string; whether link formation is
scanned every MCS or only on boundary-changing copies (I assume the latter from the neighbor_order
contact semantics, but did not confirm in core). No ablation run exists for this mechanism yet.

### Verification pass (re-excavation)
Re-checked the two specific citations in this entry against source, both hold:
- ActivationEnergy-XML-only warning is verbatim at `CC3DMLGeneratorBase.py:L882-884`.
- The `Lambda 10 / ActivationEnergy -50 / TargetDistance 7 / MaxDistance 20 / MaxNumberOfJunctions 1`
  defaults are real, but they come from the **twedit generator template** (`CC3DMLGeneratorBase.py`
  L894-898), NOT from the `FocalPointPlasticity.xml` test — that test file has no FPP block at all,
  only Volume(target 25)/Contact/ExternalPotential. So "target 7 / max 20 on volume-25 cells"
  combines two different co-named sources; entry surprise #3 was corrected to say so. The negative
  ActivationEnergy default (-50) corroborates "negative promotes link formation."

### Normalization — verdict `new` → contract `bond`
Verdict **`new`** against the frozen 42. Contract `bond`: a persistent, plastic, load-ruptured
cell-cell link network — junctions self-assemble on contact under a per-cell coordination cap
(paying a one-time ActivationEnergy), persist as identified per-pair state, and rupture when their
CoM-CoM distance exceeds a break length. Classified `rewire`/`topology`/set `cell`. NOT
`implementation_of` anything: it is distinct from `adhere` (continuum surface-contact energy; the
CC3D Contact/AdhesionFlex mechanisms are `adhere`), because `bond` is a discrete centroid-pair link
graph, not a boundary-site energy. The restoring spring `lambda*(d-target)^2` is charged separately
to the registered `squared_law`, so only ONE new contract is credited — the plastic topology, not
the spring.

**Strongest argument AGAINST `new`:** FPP may be nothing but a COMPOSITION of two things already in
hand — `radius_graph` (proximity edges) + `squared_law` (a quadratic pair spring) — with no new
atomic contract at all; on that reading it should be recorded as two existing contracts, and minting
`bond` inflates the yield. The rebuttal I rest on: `radius_graph` is deliberately memoryless and
symmetric-threshold (it rebuilds every edge from scratch each tick), so it cannot produce FPP's
hysteresis (form within ~1–2 contact sites, break only past distance ~20), its persistent per-link
identity/attributes, its per-cell coordination cap, or its once-paid formation energy — the
composition genuinely fails to reproduce the dynamics. If that rebuttal is wrong (e.g. a stateful
variant of `radius_graph` is considered fair game to widen into), `bond` collapses to a
`refinement` of `radius_graph` and the honest record is one fewer new contract.

### Re-read pass (LinkConstituentLaw variables)
Resolved part of the `law` UNKNOWN by reading `LinkConstituentLaw` (`PyCoreSpecs.py:L4512`) more
closely: Lambda/Length/TargetLength are the built-in default variables, and ARBITRARY extra
variables are bindable via the `variable[name] = value` accessor (L4550), each emitted as a
`<Variable Name=.. Value=..>` child of `<LinkConstituentLaw>` (L4545). Updated the `law` param role to
say so. Still unread: how the compiled core parses/evaluates the Formula string — only the XML
emission is visible from the Python layer, not the evaluator.


---

## kerneldiffusionsolver

<!-- kerneldiffusionsolver -- append below; the driver merges this into campaign/analysis.md -->

## KernelDiffusionSolver (order 17)

Read the class (PyCoreSpecs.py:6642) and all its child specs: KernelDiffusionSolverDiffusionData
(6401), SecretionData (6464), BoundaryConditions (6477), Field (6533), plus the shared bases
`_PDEDiffusionDataSpecs` (554), `_PDESolverFieldSpecs` (1057), `_PDESolverSpecs` (1150). Also the
shipped guide text `diffusion_solvers_descr.py:36`. code_path was already correct.

What it is: a PDE diffusion solver, sibling to DiffusionSolverFE (order 14). Same field->field
write role, but the method differs — it advances the concentration field by CONVOLUTION with a
diffusion kernel (Green's function) rather than a local FTCS stencil. That makes it fast for
LARGE diffusion constants, the exact regime where the FE solver is slow/unstable.

Three things surprised me, all real constraints (not my inference):
- Periodic BCs are HARD-ENFORCED (all six boundary types read-only = periodic, 6477-6505). The
  convolution assumes a torus. This is the sharpest contrast with DiffusionSolverFE.
- D and decay are GLOBAL-ONLY: the DiffusionData.xml (6416-6431) never emits diff_types/
  decay_types even though the base spec_dict carries them — so no per-cell-type coefficients,
  necessarily, since one convolution kernel has one width.
- The guide flags it "legacy" and "approximate" — not an exact solution.

What I could NOT establish: the precise meaning of the two distinguishing params, `kernel` and
`cgfactor`. The source documents them only as "kernel of diffusion solver" and "coarse grain
factor", with validators requiring >= 1 and XML emitted only when > 1. I INFERRED kernel =
number of expansion terms / periodic images in the kernel, and cgfactor = lattice downsampling
factor, from how kernel/convolution solvers usually work — but the actual convolution lives in
the compiled core (.so), which I did not read. Anyone tuning these is tuning core behaviour I
could only describe from the outside. I also did not verify the decay/secretion coupling order
within a step (multiplicative decay vs additive source) against the core; that ordering in the
equations block is the standard reading, not confirmed. No paper text available for this target,
so paper_section points at in-tree checkable anchors only.

### Normalizer verdict

**`alias` of `diffuse`** (implementation_of: diffuse). This solver advances a scalar field by
periodic convolution with a precomputed diffusion kernel (the Green's function). By the
convolution theorem that is the same operation as `diffuse`'s existing `spectral` implementation
(a Fourier multiply by exp(-D k^2 dt) IS a real-space Gaussian convolution on the torus) — at most
a fourth numerical implementation of the same field->field contract, done in real space with a
truncated kernel + optional lattice coarsening. Crucially, unlike its FE sibling (which I made a
*refinement* because its D_i/lambda_i are spatially heterogeneous, forcing `diffuse`'s scalar
`rate` to widen into a coefficient field), this solver emits ONLY global D and lambda and *cannot*
express per-type coefficients — one kernel has one width. A uniform scalar D is exactly what
`diffuse` already binds, and periodic-only is a *narrowing* onto the spectral implementation, not a
new signature. So nothing widens and nothing breaks: alias, not refinement. The whole mechanism is
the same diffuse+decay+deposit composite as DiffusionSolverFE.

**Strongest argument AGAINST alias (for `refinement`):** the current `diffuse` contract has no way
to say "this implementation is periodic-ONLY and legacy/approximate," nor to carry the
`kernel`/`cgfactor` truncation dials — so one could argue it must widen to advertise a
`boundary: periodic-required` precondition, lest a user asking for `diffuse` with absorbing walls
be silently handed an approximate torus solver. I reject this: those are numeric/accuracy and
implementation-*selection* concerns, not changes to the biological signature (set, inputs,
outputs, reads, writes are identical to base diffuse). Per-implementation domains of validity are
exactly what `implementation_of` is for; promoting a periodicity precondition to a signature change
would do violence to `diffuse`'s biology (transport is transport regardless of scheme) and would
misreport the saturation curve as finding new language where it found a second realisation of a
verb we already have.


---

## lengthconstraint

<!-- lengthconstraint -- append below; the driver merges this into campaign/analysis.md -->

# LengthConstraint (order 18) — excavation note

**What I read.** `PyCoreSpecs.py:3850-4056` — `LengthEnergyParameters` (per-type param block:
cell_type, target_length, lambda_length, optional minor_target_length) and
`LengthConstraintPlugin` (a `_PyCoreSteerableInterface`, so steerable). The Python side only emits
CC3DML `<LengthEnergyParameters CellType TargetLength LambdaLength [MinorTargetLength]/>`; the
physics is the compiled `LengthConstraintPlugin.changeEnergy` (SWIG stubs at
`cpp/CompuCell.py:5219-5264`, with plane-specific `changeEnergy_xy/_xz/_yz` and `_3D`). Also read
the reference sim `elongationFlexTest` (xml + steppable): it drives elongation via
`setLengthConstraintData(cell, lambdaLength, targetLength)` per cell and pairs the plugin with
Volume + CenterOfMass + ConnectivityGlobal.

**Mechanism.** A Potts ENERGY term, not a state update: `E = lambda_L (L - L_t)^2`, plus a
`(W - W_t)^2` minor-axis term when minor_target_length is set. `L` is the cell's extent along the
longest principal axis of its moment-of-inertia tensor about the COM — a mass-weighted continuous
length, not a pixel bounding box. Returns `dE` into the Metropolis test; writes nothing.

**Surprised me.** (1) `setLengthConstraintData(cell, lambda, target)` passes lambda BEFORE target —
opposite of the CC3DML attribute order and of intuition. (2) Constraining only the major axis lets
a cell hit its target length by thinning to a filament; the reference sim needs Volume +
Connectivity to keep the cell physical. (3) The plugin is steerable, so target_length can be ramped
in time — active elongation, not a static prior.

**Could NOT establish.** The exact eigenvalue→scalar-length map inside compiled `changeEnergy`
(normalization/factor — is `L` a diameter, a semi-axis, sqrt-scaled?). I recorded only the
`(L - L_t)^2` penalty FORM as established; the constant is unread. Also unverified: whether the
minor-axis term reuses the same `lambda_L` or a separate coefficient (source exposes one
`LambdaLength` attribute, so I assumed shared — confirm against a 2D run). No extracted paper text
exists; the source class + CC3DML + the elongationFlex reference sim are the only evidence.

**Added from the `.so` symbol table (`libCC3DLengthConstraint.so`).** Two things not visible from
Python: (1) `spring_energy(double,double,double)` confirms the quadratic-spring FORM directly in the
binary. (2) `_get_non_nan_energy(double)` — an explicit NaN guard on the energy. A degenerate cell
(single pixel, undefined inertia) yields a NaN length; without this guard the NaN propagates into
`dE` and silently corrupts the Metropolis test. A reimplementer would almost certainly miss it.
Also: per-cell state is a `LengthConstraintData` ExtraMember, so the plugin carries per-cell (not
just per-type) targets — the local-flex scope `setLengthConstraintData` writes into.

**Verdict (normalizer): `new` → contract `elongate` (lateral/mechanics, set=cell).** No promoted
contract constrains an emergent SHAPE descriptor: this is a quadratic restoring spring on a cell's
inertia-tensor major-axis length toward a target, an ENERGY term (returns dE, writes nothing).
Closest promoted is `cell_grow`, rejected because it targets SIZE (0th moment, isotropic, an
integrated state update) not ANISOTROPY (2nd moment, at fixed volume, a Metropolis-gating energy) —
widening it would break its output contract and its biology. Not a second sighting of any jax-morph
contract (`reorient` is polarity direction, not shape magnitude; `relax`'s meaning is unread).

**Strongest argument AGAINST `new`.** Length and the Volume constraint are the SAME functional
object — a quadratic Hookean spring, lambda·(moment − target)², on a per-cell geometric moment
(Volume on the 0th, Length on the 2nd). One could argue the language should register ONE contract,
say `constrain_moment(cell, order, target, lambda)`, of which Volume and Length are interchangeable
IMPLEMENTATIONS differing only in which moment they read — making `elongate` `implementation_of`
that, not `new`, and inflating yield if I call it new. I reject the lump because the moments carry
different biology (size homeostasis vs elongation) and a different home in the language: Volume's
size-target maps onto the existing `cell_grow` rest-volume machinery, while shape has no home at
all — so collapsing them would erase a real distinction rather than reveal a shared one. But it is
a genuine call, not a fact: if Volume normalizes as a `constrain_moment`-style contract, `elongate`
should be revisited as a second implementation of it.


---

## mcs_time_unit

<!-- mcs_time_unit -- append below; the driver merges this into campaign/analysis.md -->

**Verdict: `out_of_scope`.** The Monte Carlo Step is not an operator over sets or fields; it is
the engine's *temporal-integration contract* -- the definition of the clock itself. It computes no
force, energy term, flux, or division, and none of the seven Plexus kinds describes a scheduler. Its
worth to the atlas is the contradiction it pins down, the sharpest in the record: the promoted Plexus
engine advances state by integrating operator deltas against a real-valued `dt`; CC3D has no `dt` and
no integrator -- time is one attempted pixel copy per lattice site, and a rate must be re-expressed as
a per-MCS acceptance *probability*. The two frameworks disagree on the meaning of the time axis, not
its units. The Plexus algebra can express the energy *terms* of a Potts model but the engine cannot
express its *time*.

**Strongest argument against.** One could call this `new` rather than `out_of_scope`: the campaign
measures whether the language is complete, and here is a genuine capability the promoted vocabulary
lacks -- a discrete, dt-free, per-site stochastic clock. If "the language" is read to include the
engine's execution model and not just the 42 operator contracts, then MCS is precisely a missing
piece and marking it out-of-scope hides a real incompleteness behind "it's plumbing." I keep
`out_of_scope` because the frozen baseline `new` is measured against is a set of *operators* with a
typed `set/inputs/outputs/reads/writes/maps` signature, and a time unit has no set to act on and
writes only the step counter -- promoting it as an operator would corrupt the very saturation metric
the atlas exists to protect. But I record the incompleteness explicitly in `why:` so the measurement
is not lost: it is a gap in the *engine*, logged where it belongs, not smuggled into the operator count.


---

## metropolis_acceptance

<!-- metropolis_acceptance -- append below; the driver merges this into campaign/analysis.md -->

**metropolis_acceptance -> out_of_scope (forced-fit contract `metropolis_step`, structural/growth).**
This is the CPM's core modified-Metropolis Monte Carlo integrator, declared on PottsCore (the
simulation root), not a plugin: it reads the summed dE from every enabled plugin and one T, then
accepts a proposed pixel copy with P=1 if dE<=0 else exp(-(dE+offset)/T). It carries no biology
of its own -- the biology is entirely in the per-plugin dE terms already catalogued -- so it is
framework mechanics, the counterpart to the Plexus engine's integration loop, which sits outside
the operator algebra. Recording it as an operator (`new` or a `cell_divide` alias) would inflate
the yield with the integrator itself and corrupt the saturation measurement.

**Strongest argument AGAINST out_of_scope:** the hand-added surprise says the discrete, stochastic,
accept/reject dynamics "with no pathwise derivative" is itself the finding -- and a gap in the
language is exactly what `new` is meant to flag. If Plexus genuinely cannot express energy-plus-
accept/reject dynamics, one could argue that incapacity is a missing contract, not out-of-scope
plumbing. I reject this because `new` in this loop means a missing *operator* (a typed map over
sets/fields returning a delta), and the Metropolis rule is the integrator that *consumes* deltas/
energies, not one that produces them -- it maps to Plexus's engine, which registers no operator
either. The orthogonality of the two integration paradigms (energies+stochastic accept/reject vs
deltas+deterministic integration) is real and is recorded verbatim in `why:` and `surprises:`; that
is a measurement result about the engine, not a new entry in the operator vocabulary. The honest
move is to state the gap loudly under an out_of_scope verdict rather than mint a fake operator to
represent it.


---

## momentofinertia

<!-- momentofinertia -- append below; the driver merges this into campaign/analysis.md -->

# MomentOfInertia (order 19)

## What I read
- Spec at `PyCoreSpecs.py:5350` (`MomentOfInertiaPlugin`) is a stub: no constructor args, emits a
  bare `<Plugin Name="MomentOfInertia"/>`. Fixed nothing — line was correct.
- The physics is in the compiled core. Traced the SWIG bindings in `cpp/CompuCell.py`:
  - Plugin is a **`CellGChangeWatcher`** (line 9002) — invoked by the Potts solver *after* an
    accepted pixel copy, not an energy plugin and not a steppable.
  - It maintains six per-cell tensor fields `CellG.iXX/iYY/iZZ/iXY/iXZ/iYZ` (lines 545-550) and
    derives `CellG.ecc` (554), semiaxes (`getSemiaxes*`, 9030-9043) and orientation
    (`cellOrientation_xy/xz/yz`, 9019-9026).
  - Incremental helper `precalculateInertiaTensorComponentsAfterFlip` (4235) + `eccFromComps`
    (8994) confirm O(1)-per-flip maintenance and the eccentricity-from-components derivation.
- Used by shape-dependent code (oriented-cellsort test XML, `OrientedGrowth` in PySteppables).

## What surprised me
- It's a genuine **third category** for this atlas: not an energy term (returns no dE, no acceptance
  role) and not a modeller update — a passive *change-watcher* running an **incremental reduction**
  that keeps a derived statistic in sync with the pixel set. Worth flagging for the vocabulary
  question: does the Plexus algebra have a "tracker / incremental observable" contract distinct from
  operators that return deltas?
- Depends implicitly on CenterOfMass (measures the tensor about `r_CM`); no param exposes this.

## What I could NOT establish
- The exact diagonal/sign convention in the compiled tensor (physicist `sum(y'^2+z'^2)` vs raw
  central second moment `sum(x'^2)`). Inferred physicist form from names+physics; unverified. The
  downstream `ecc`/semiaxis derivations are what code actually reads and are robust to the choice.
- Exact semiaxis normalisation constant (eigenvalue → length) — compiled, not read.
- No ablation/evidence run exists for this mechanism (not among the six with metrics.json).

## Added this pass (re-verified at source)
- The derived fields are **read-only from Python**: the binding overrides `set_iXX` (etc.) to raise
  `AttributeError "iXX is read only variable"` (`cpp/CompuCell.py:649`). The modeller can *read*
  cell shape but cannot assign it — only the plugin's watcher may. This hard-enforces the observer
  reading and is a concrete reimplementer trap: exposing the tensor as writable Plexus state would
  be a category error. Added as a surprise.
- The watcher hook is `field3DChange` (this plugin at `cpp/CompuCell.py:9016`, base
  `CellGChangeWatcher.field3DChange` at 3000), i.e. per-accepted-cell-id-change, NOT per-MCS. The
  helper returns an `InertiaTensorComponents` struct (`cpp/CompuCell.py:4200`). Added as a surprise
  so a reimplementer doesn't attach a fixed-schedule stepper.
- Made `paper_section` honest: no extracted paper text exists; anchored to the docstring
  (PyCoreSpecs.py:5351) and compiled class (cpp/CompuCell.py:9002), and explicitly flagged the
  Swat page/section as UNREAD rather than citing a page I have not seen.

## Normalizer verdict
- **`refinement` of `aggregate`** (`of: aggregate`, `implementation_of: aggregate`). CenterOfMass
  was aliased to `aggregate` as the centroid (first-moment) reduction; MomentOfInertia is the
  SECOND-moment sibling of the *same* children→parent reduction (identical kind/family/set/input/
  map). It is not a plain alias because two signature fields must widen: WRITES (position →
  second-moment shape tensor + ecc/semiaxes/orientation) and READS (it takes the moment ABOUT the
  COM, so it newly reads `cell.pos`). Not `new` because that would double-count the same reduction
  family and corrupt the saturation curve — the whole point CenterOfMass's alias was protecting.
- **Strongest argument AGAINST refinement (for `new`):** the OUTPUT is qualitatively different in
  kind, not degree. Centroid publishes a location; MomentOfInertia publishes cell SHAPE and
  ORIENTATION — an observable no promoted contract produces, consumed by a distinct downstream class
  (oriented growth/cellsort). One could argue "shape reduction" is its own biological contract
  (measuring cell morphology) rather than a widened `aggregate`, since widening aggregate to cover
  any statistical moment risks turning it into a catch-all "reduce children to anything" bucket that
  measures our naming laziness rather than the language. I rejected `new` because the children→parent
  reduction structure IS aggregate and both outputs are literally moments of the same site set (0th/
  1st → mean position, 2nd → shape tensor); the honest refinement is to admit higher moments, not to
  fork a near-identical contract. But this is the closest call in the campaign and a reasonable
  reviewer could land on `new`.


---

## neighbortracker

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


---

## pifdumper

<!-- pifdumper -- append below; the driver merges this into campaign/analysis.md -->

# PIFDumper (order 21)

Read `PyCoreSpecs.py:L6012` (the `PIFDumperSteppable` class), its base
`_PyCoreSteppableSpecs` (L512), and the sibling `PIFInitializer` (L5960) that
shares the `pif_name` field. Also skimmed `CMLResultsReader.generate_pif_from_vtk`
to confirm the PIF line format (`cellId cellType xLow xHigh yLow yHigh zLow zHigh`)
lives in the compiled `PlayerPython.FieldWriter`, not in Python.

What it does: a pure-output Steppable. Every `frequency` MCS it reads the whole
cell-id/cell-type lattice and writes a PIF snapshot to disk. It touches NO
simulation state -- no delta, no energy term, no pixel-copy bias. Cleanest
"neither an update nor an energy term" case so far: it's a read-only tap with an
external side effect. Worth flagging for the normalizer as an observer/sink, not
a physics operator.

What surprised me: the `xml` property emits ONLY `PIFName` and silently drops
`frequency`, even though `__init__` stores it, `check_dict` validates it, and
`from_xml` reads a `Frequency` attribute. So Python->CC3DML round-tripping loses
the interval (defaults back to 1). `__getstate__` drops it too. Also `pif_name`
means opposite things in the dumper (output target, not existence-checked) vs the
initializer (input, must exist). And the dumper's `xml` is byte-identical to the
initializer's -- only the Steppable `Type=` attribute distinguishes write from read.

What I could NOT establish: (1) HOW the core steppable actually receives
`frequency` at runtime given the xml omission -- there may be another wiring path
in the compiled core I did not read; I only confirmed the Python spec does not
carry it. (2) The exact on-disk PIF byte format (fixed vs free spacing, header
lines, 2D vs 3D bounds) -- inferred from `generate_pif_from_vtk`/library convention,
not observed from an actual dump. (3) Whether frequency counts MCS or some other
tick, and phase/offset behavior at mcs=0. Did not run evidence.py (not among the
six with reference runs).

## Follow-up pass (corrections against a REAL .piff + the twedit template)

Read a real dumped file,
`tests/plugin_test_suite/AdhesionFlexPython_test_generate/Simulation/initial_configuration.piff`,
and twedit's `CC3DMLGenerator/CC3DMLGeneratorBase.py:1271`
(`generatePIFDumperSteppable`). Three things above are now superseded:

- The PIF line format sketch (`cellId cellType xLow..`) is WRONG. A real line reads
  `8  8  Cell2  119 119  53 53  0 0`, i.e. **clusterId  cellId  typeNAME  x x  y y  z z**.
  The first column is the cluster id (not cell id), the type is a NAME string (not a
  numeric index), the file is prefixed with an `Include Clusters` header line, and
  every line is a single voxel (no run-length box compression). Entry equations +
  a dedicated surprise now carry the corrected format.
- The `frequency` omission is now LOCALIZED, not a mystery: `from_xml` reads it as a
  `<Steppable>` **header attribute** (`Frequency="…"`), which is exactly where the
  canonical twedit CC3DML puts it (default **100**, not 1). `generate_header` never
  emits that attribute — so the round-trip loss is fully explained; no hidden core
  wiring needed for the datum itself, only for the write it never receives.
- twedit also emits `<PIFFileExtension>piff</PIFFileExtension>` and names the file
  from SimulationName; PyCoreSpecs emits neither. Added as a surprise.

Still NOT established: the C++/SWIG writer itself is unread, so traversal order, the
MCS→filename suffix rule, and whether the `Include Clusters` header is conditional
are inferred from one sample + the template, not from the writer code. Still no
evidence.py run (a disk sink has no meaningful ablation).

## Normalizer pass — verdict: out_of_scope

PIFDumper is a serializer, the archetypal out_of_scope mechanism the loop's rules
name explicitly. Zero biological content, zero dynamics: it walks the cell-id lattice
every `frequency` MCS and writes one PIF line per voxel to disk. No delta, no energy
term, no pixel-copy bias — a run with it loaded is bit-identical to one without. I
record a shape-only `checkpoint` contract (whole cell-set read → external bytes) purely
to document the sink's type, borrowing `playback`'s field/fields neighbourhood as an
admitted forced fit.

**Strongest argument AGAINST out_of_scope.** The honest challenger is `playback`:
PIFDumper is its exact inverse (state→disk vs disk→state), and a PIF file round-trips —
PIFInitializer reads it straight back as an initial condition. So one could argue that
dump + init form a single serialization contract with two directions, and the write
direction deserves a promoted name the way `deposit`/`sense` split a field write from a
field read. I reject it because `playback`, `deposit`, and `sense` all produce something
the engine integrates *within the same run*; the dumper produces inert disk state whose
only consumer is a *future, separate* run's initializer — the running model never reads
it back. Admitting it would let checkpoint I/O inflate the operator count without adding
any process the algebra must express, which is exactly the measurement this campaign
protects. If a future framework showed a sink whose output re-enters the *same* running
model, I would reopen this.


---

## pifinitializer

<!-- pifinitializer -- append below; the driver merges this into campaign/analysis.md -->

# PIFInitializer (order 22)

**What I read.** `PyCoreSpecs.py:5956` — the whole `PIFInitializer` class plus its base
`_PyCoreSteppableSpecs` (line 512; `generate_header` at 519, emits `<Steppable Type="PIFInitializer">`)
and its sibling `PIFDumperSteppable` (6012).
The Python side is tiny: one param `pif_name`, `validate()` asserts the file exists, `xml`
emits `<PIFName>`. The real work is in `cpp/.../libCC3DPIFInitializer.so` (compiled, not
readable), so I reconstructed the painting behaviour from actual `.piff` files in the test
suite (`AdhesionFlexPython/.../initial_configuration.piff`, `CompartmentExample/.../3D_LINE_BASE.piff`).

**What it does to state.** One-shot initializer at t=0. Reads a PIF text file and stamps the
cell-id lattice directly: each line = `[clusterId] cellId cellType x1 x2 y1 y2 z1 z2`, and every
site in the inclusive box is assigned to that cell. Lines sharing a cellId accrete sites → a cell
is the union of its boxes. This is the cleanest instance of CC3D's "cell = set of lattice sites"
representation in the whole inventory.

**Surprises.** (1) The `Include Clusters` header silently adds a leading column — column meaning
is header-dependent. (2) Box bounds are inclusive both ends. (3) Internal spec name is the typo
`"pif_initiazer"`. (4) `check_dict` only rejects the empty string at construction; existence is
checked later in `validate()`. (5) Cells need not be connected/rectangular.

**What I could NOT establish.** The exact core semantics — pixel iteration order, what happens on
OVERLAPPING boxes (last-writer-wins vs. first?), out-of-lattice-bounds clipping, and whether an
unknown cellType is auto-registered or errors — all live in the unreadable `.so`. I inferred
inclusive boxes and set-union from the file contents, not from core code. No paper text exists for
this target, so `paper_section` is anchored to the source line + the docs `.rst` autoclass, not a
verified page. Did not run evidence.py (initializer has no dynamics/ablation to measure).

**Format re-verification (this pass).** Checked BOTH header modes against real files, not just the
two cluster examples: cluster mode = 9 tokens, and `3D_LINE_BASE.piff` PROVES clusterId is
independent of cellId (clusterId 1 groups cellIds 1-5 — earlier notes asserted this; now confirmed).
Headerless mode = 8 tokens, seen in `ExternalPotential/.../FocalPointInit.piff` and
`amoebae_2D_secretion/.../amoebae_2D.piff`. Made the entry's checkable anchors concrete with these
exact paths. Overlap/last-writer semantics remain the one unresolved gap (still in the `.so`).

## pifinitializer — normalized

**Verdict: `out_of_scope`, `implementation_of: seed`.** PIFInitializer runs once at t=0 and
CONSTRUCTS the initial cell partition by reading an external `.piff` file and painting cell ids
onto the lattice box-by-box. That is initial-condition construction, which Plexus supplies via
configuration/seeding rather than any per-step operator — nothing in the registered algebra
transforms state here, so there is nothing to alias or widen. It is doubly out-of-scope because
it is specifically a DESERIALIZER whose exact inverse is the PIFDumper serializer (a read/write
pair over the PIF text format is textbook plumbing). It is the third of the Blob / Uniform / PIF
initializer family — three interchangeable ways to build the same starting partition — so I set
`implementation_of: seed`, the descriptive contract the blobinitializer normalizer already typed
with these three as its implementations, keeping the ledger counting `seed` ONCE. Blob/Uniform
seed procedurally from numeric params; PIF replays a recorded layout — same end state, opposite
information source.

**Strongest argument AGAINST (i.e. for `new`):** this and the sibling initializers are the ONLY
mechanisms in the campaign that genuinely write state and CREATE sets — literally how cells come
to exist. The registered algebra has no way to construct a population de-novo: `cell_divide`
splits a pre-existing parent (conserving material, one→two) and nothing seeds cells out of Medium
with no parent. If a cell-based framework must express "instantiate the initial partition,"
calling it out_of_scope hides the single most load-bearing structural gap, and the honest verdict
would be `new` (a `seed` contract, PIF/Blob/Uniform as co-implementations). PIF sharpens the
tension: unlike a procedural blob it carries genuine INFORMATION (an arbitrary,
possibly-disconnected recorded configuration) that no numeric param set can regenerate, which
reads more like a first-class "load a configuration" operator than mere plumbing. I still land on
out_of_scope because Plexus *architecturally* seeds initial state through configuration, not an
operator — but the line between "IC construction is config" and "IC construction is a missing
operator" is the real judgement call, and a reasonable normalizer could put it the other way.


---

## pixel_neighbourhood

<!-- pixel_neighbourhood -- append below; the driver merges this into campaign/analysis.md -->

## pixel_neighbourhood (neighbour order / relation E) — normalized

Read the Potts flip-attempt `neighbor_order` (PyCoreSpecs.py:L1405-1470, emitted L1557) and the
independent per-plugin orders on Surface (L2248), Contact (L3164), a bare NeighborOrder spec
(L3561), and FocalPointPlasticity (L4577). Compared against the registered `radius_graph`
(src/plexus/operators/graph.py:16-45).

**Verdict: refinement of `radius_graph`.** The pixel neighbourhood plays radius_graph's exact
architectural role — the rewire/topology operator that fixes the within-set relation E, emits no
delta, and lets every lateral/contact term read the neighbourhood it leaves. That rules out `new`
(a promoted contract already covers relation-building). But the registered signature is built
around continuous `particle` positions with a continuous `radius` cutoff, rebuilt each tick; the
CPM relation lives over a lattice site-set, reads no dynamic state, and is selected once by a
discrete `neighbor_order` (+ `lattice_type`). Admitting it widens three signature fields
(set, reads, inputs/maps) and relaxes the per-tick-rebuild invariant — the definition of a
refinement, not an alias. Not out_of_scope: radius_graph being promoted means Plexus already
treats "build the interaction relation" as in-scope, so the lattice version is too. No cross-atlas
dedup (no jax-morph proposal is a relation-builder).

**Strongest argument AGAINST (why this could be a plain `alias` instead):** the role is identical
and the widening is purely additive — no existing radius_graph caller breaks, no runtime behaviour
of the current implementation changes. One could argue a `rewire` operator whose `reads` set is
simply empty and whose parameter happens to be discrete is already *expressible* under the
existing contract (radius_graph is just one implementation of it), making lattice adjacency a
sibling implementation rather than a widening — i.e. alias, of: radius_graph. My rebuttal is that
the registered contract is written explicitly as "all live pairs within radius, rebuilt each tick
from pos over a particle set"; taking it to a static lattice-site adjacency where "distance" is a
discrete integer offset breaks the invariant that edge length is a meaningful continuous distance,
which any distance-reading downstream analysis relies on. That is a cost someone must pay, so I
called it a refinement — but the alias reading is genuinely defensible and is the main thing a
reviewer should push on.

**Could NOT establish:** the compiled adjacency tables per (lattice_type, order) and the exact
Metropolis source/target draw are in the C++ core, not readable from this install; reconstructed
from the CC3DML the spec emits and standard CPM behaviour. No paper text available — `paper_section`
cites the Swat chapter but anchors to in-source line numbers. Not among the six mechanisms with
reference ablations under `log/atlas_cc3d/`, so all claims are unmeasured.


---

## pixeltracker

<!-- pixeltracker -- append below; the driver merges this into campaign/analysis.md -->

# PixelTracker (order 23) — excavator note

Read: PyCoreSpecs.py:5311 (the spec, one param `track_medium`), the SWIG stub
CompuCell.py:6002 (`class PixelTrackerPlugin(CellGChangeWatcher)` — the decisive
fact), the attached-data classes PixelTracker/PixelTrackerData (5837, 5854), the
iterator CellPixelList (iterators.py:700), and the consumers in PySteppables.py
(get_cell_pixel_list:2465, get_copy_of_cell_pixels:2584, move_cell/delete_cell).

What it does: it materialises the inverse of the cell_field. The CPM lattice stores
site -> cell id; PixelTracker keeps a live per-cell set of the sites that id owns,
updated incrementally on every accepted pixel copy via field3DChange (erase pt from
old owner, insert into new owner). Delta H = 0 — it is NOT an energy plugin.

Surprised me: this is the plugin that reifies the ATLAS's own framing ("a cell is a
set of lattice sites sharing an id"). In CC3D that set is NOT primitive — you must
load a watcher to build the id->{sites} map; the lattice alone only gives you
site->id. In Plexus the set is the primitive, so PixelTracker looks like an
implementation of a representation Plexus gets for free. Whether the algebra needs
an explicit "index / inverse-map over a field partition" contract is the open
question I flagged in surprises — I did NOT resolve it (that is the normalizer's call).

Could NOT establish: I could not read the compiled C++ field3DChange body itself —
only the SWIG signatures and the Python consumers — so the exact erase/insert
ordering and the medium-guard branch are reconstructed from the SWIG API
(enableMediumTracker, getMediumPixelSet, trackingMedium) plus behaviour, not from
reading the .cpp. High confidence given CellGChangeWatcher semantics, but stated as
reconstruction. No ablation/evidence run exists for this mechanism (not among the
six with metrics.json). track_medium default False is a performance guard, verified
from the spec's conditional <TrackMedium/> emission (PyCoreSpecs.py:5333).

Re-verification pass (correction): the earlier entry claimed get_cell_pixel_list
RAISES when PixelTracker is absent. It does not — it silently returns None
(PySteppables.py:2473-2476). The AttributeError is raised only by the higher-level
wrappers get_copy_of_cell_pixels (2603), delete_cell (2534), and move_cell (469).
Fixed the entry's dependency surprise to reflect the silent-None low-level path (a
worse gotcha than a clean raise: a forgotten None-check surfaces as a confusing
NoneType error downstream). Also confirmed the container is concretely `pixelSet`
on the extra-attrib accessor (iterators.py:703/724). Everything else in the entry
checked out against source; status stays inspected.

Anchor re-verification (this pass): re-read PyCoreSpecs.py:5311 (class def line
correct — code_path unchanged) and located the real SWIG stub at
cc3d/cpp/CompuCell.py (not a bare CompuCell.py): class PixelTrackerPlugin at :6002,
PixelTrackerPlugin_field3DChange at :6010/:6011, enableMediumTracker at :6031, all
confirmed. iterators.py:700 (CellPixelList) and :724 (pixelSet.size) confirmed
verbatim. Corrected the paper_section anchor to the full checkable path
cc3d/cpp/CompuCell.py:6002. Prior non-acceptance was purely a merge-hygiene issue
(atlas_record.yaml touched directly) — the science was unchanged; edited only the
working copy this time.

Line-attribution correction (this pass): re-reading the consumers showed the raise
sites were mis-attributed in the dependency surprise. The verified truth:
get_copy_of_cell_pixels (def 2584, raises 2603) and move_cell (def 2509, raises
2534) both raise AttributeError('Could not find PixelTracker Plugin');
delete_cell (def 2636) raises only INDIRECTLY through get_copy_of_cell_pixels; and
line 469 is merge_cells (def 458), which raises its own Exception "…did you load
PixelTracker plugin?" — NOT move_cell. Entry's third surprise updated to match.

## Normalizer verdict

Verdict: **out_of_scope**, contract `pixel_index` (rewire/topology, set=cell). PixelTracker
is a pure derived-index/observer: Delta H = 0, bit-identical simulation with or without it, its
only product the live cell id -> {sites} inverse of the cell_field. Ruled the same way its
direct sibling BoundaryPixelTracker was (that one indexes only the boundary subset; this one the
full interior set) — a spatial index / acceleration cache, not a process, so it must not count
toward the language's vocabulary.

Strongest argument AGAINST out_of_scope: this is the one tracker plugin that reifies the ATLAS's
OWN premise — "a cell is a set of lattice sites sharing an id." So unlike a neighbour-list cache,
one can argue Plexus really is MISSING a first-class "invert a field partition into per-element
membership sets" contract, and that inverse-map IS a nameable topological operation (it is what
lets any per-cell geometry operator exist at all). If Plexus ever represented cells AS labelled
lattice/voxel fields rather than as particles, this operator would be load-bearing and `new`, not
plumbing. I reject it only because in the PROMOTED representation the partition is primitive (a
cell already is a set with identity; there is no scalar cell-id field to invert), so the operator
reconstructs something the representation gives for free and carries no biological content the
language could name. That rejection is contingent on the representation, not on the mechanism —
worth flagging, because a future voxel-based Plexus backend would flip this verdict.


---

## reactiondiffusionsolverfe

<!-- reactiondiffusionsolverfe -- append below; the driver merges this into campaign/analysis.md -->

# ReactionDiffusionSolverFE (order 24)

Read: `PyCoreSpecs.py:L6911` (the class) + its DiffusionData (L6740), SecretionData (L6830),
Field (L6843), and the base PDE classes (`_PDEDiffusionDataSpecs` L554, `_PDESolverSpecs` L1150,
`_PDESolverFieldSpecs` L1057). Cross-read the sibling `diffusionsolverfe` entry (order 14) for the
FTCS discretisation, the FitzHugh-Nagumo example XML
(`tests/pde_solvers/.../ReactionDiffusion_2D_FN.xml`), the twedit generator
(`CC3DMLGeneratorBase.py:1959`), and the in-tree solver blurb (`diffusion_solvers_descr.py:11`).

What it is: DiffusionSolverFE's forward-Euler diffuse-decay-secrete step, PLUS a per-field
`AdditionalTerm` -- a muParser expression that may name the OTHER fields, coupling them into a
reaction-diffusion system. It writes the concentration grids in place (a real field write, not a
Potts energy delta).

Surprised me:
- Same python attribute `additional_term` is emitted here but silently dead in DiffusionSolverFE
  (its DiffusionData.xml never writes it). The coupling is invisible if you compare specs by
  attribute list.
- ConstantConcentration (Dirichlet clamp) is explicitly REFUSED (SpecValueError, L6892).
- A BLANK reaction defaults to `1*<field>` i.e. R=c (exponential growth), not R=0 (twedit
  L2007/2011) -- a magic default.
- init_expression is read by from_xml but never emitted -> lost on write; only init_filename
  round-trips.
- AutoscaleDiffusion is per-steppable in PyCoreSpecs (L6943) but per-field in twedit (L1990).
- (added this pass) from_xml (L7016-7019) PARSES <ConstantConcentration> and forwards
  constant=True to secretion_data_new -- the same method (L6892) that RAISES on that kwarg. So the
  reader accepts a CC3DML the writer forbids: importing a legacy pinned-source model crashes on the
  read, not on write. Reader/writer disagree about whether Dirichlet clamps exist, inside one class.

Could NOT establish (someone must not assume otherwise): the exact FE integration of the reaction
term -- DeltaT scaling, evaluation order relative to the diffusion sweep, and whether
AutoscaleDiffusion rescales R -- all live in the compiled core, which I did not read. The continuous
form and named-field coupling are solid (spec xml + FN example); the discretisation of R is inferred.
DeltaX/DeltaT/ExtraTimesPerMCS remain unreachable from this python spec, same gap as DiffusionSolverFE.

---

## Normalizer verdict (this pass)

**`new` — contract `react` (kind=field, family=fields, set=field), `implementation_of: react`.**
The diffuse/decay/deposit substrate is already resolved by the sibling `diffusionsolverfe` entry
(refinement of `diffuse`); the sole marginal verb here is the `AdditionalTerm` reaction term — an
arbitrary, generally nonlinear kinetics law coupling M concentration fields (FitzHugh-Nagumo:
F reads H, H reads F). Closest registered contract is `decay`, which is exactly the linear,
single-field, sign-fixed degenerate case of a reaction; widening it to arbitrary multi-field
coupling erases what makes decay decay, so `new`.

**Strongest argument AGAINST it.** That `react` and jax-morph's proposed `regulate` are the SAME
contract, so I should have set `implementation_of: regulate` and let the ledger count the
reaction-kinetics verb once — keeping them separate risks INFLATION, the exact failure this loop
exists to avoid. Both are `dx/dt = f(x_1..x_n)` with a user-supplied law integrated per step; a
maximalist reasonably says the substrate (extracellular pixel field vs per-cell gene vector) is a
parameter, not a new verb, and that FitzHugh-Nagumo chemistry and a gene network are one abstract
reaction network. My rebuttal: Plexus types by set — `regulate` is set=cell/kind=exchange (an
internal genotype→phenotype decision reading a cell's own state), `react` is set=field/kind=field
(pure-grid chemistry, no cell/gene/heritability). Unifying forces `regulate` to widen BOTH its set
(cell→field) and its kind (exchange→field), stripping its defining "internal cell decision"
character — the same violence test rule 2 uses. If that rebuttal is wrong, the right fix is a single
set-polymorphic reaction contract, which is why the entry's `why` flags the merge as a deliberate
ledger-keeper call rather than forcing it silently either way.


---

## secretion

<!-- secretion -- append below; the driver merges this into campaign/analysis.md -->

## secretion (EXCAVATOR, read at source)

Read `SecretionPlugin` (PyCoreSpecs.py:L4306) plus its two helper specs `SecretionField`
(L4192) and `SecretionParameters` (L597), and the compiled `FieldSecretor` API in
cpp/CompuCell.py (L9082+). The Python layer only emits CC3DML; the physics is in
libCC3DSecretion.so, so I read semantics off the FieldSecretor method names + the amoebae
test XML (`<Secretion Type="Amoeba">20</Secretion>`).

What it does TO STATE: it is a genuine FIELD WRITE (not an energy term). Every MCS it walks
the lattice sites a cell owns and either ADDS a rate (`Secretion`), OVERWRITES a level
(`ConstantConcentration`, a Dirichlet clamp), or adds a rate only at boundary sites touching a
named other type (`SecretionOnContact`). It does NOT transport the chemical — a separate PDE
solver diffuses the same field. This is one of the few CC3D mechanisms that actually writes
state each step, so `state_io.writes` is real, unlike the accept/reject plugins.

Surprised me: (1) `value` is overloaded — a per-step RATE in additive modes but an absolute
CONCENTRATION in constant mode. (2) `ExtraTimesPerMC` (frequency) silently multiplies the
effective rate. (3) The exact same physics can be declared either as this plugin OR inline as
`<SecretionData>` inside a DiffusionField — two spec surfaces, one mechanism. (4) It depends on
PixelTracker/BoundaryPixelTracker for the pixel sets it iterates.

Could NOT establish (compiled core, not read): the intra-MCS ORDERING relative to the diffusion
solve (does secretion inject before or after the field is stepped?), whether the plugin path
uses the "old" or "new" field, and the precise neighbour-iteration order for on-contact. I
inferred additive-vs-overwrite and rate-vs-level purely from method names + XML, not from
running it — no evidence run exists for this mechanism, so those semantics are unverified by
measurement.

### Addendum (second pass, read cpp/CompuCell.py FieldSecretor + PySteppables)

Two corrections/enrichments after reading the compiled `FieldSecretor` (CompuCell.py:L9082) and
`SecretionBasePy` (PySteppables.py:L3392):

- STRICT SUBSET: the XML plugin's `from_xml` maps only Secretion / ConstantConcentration /
  SecretionOnContact onto the three INSIDE-cell variants. FieldSecretor additionally exposes
  UPTAKE (a sink, `uptakeInsideCell*`, absolute + relative-to-max), OUTSIDE-cell boundary
  secretion (`secreteOutsideCellAtBoundary`, writes the medium sites just outside the cell), and
  COM-only point secretion (`secreteInsideCellAtCOM`). These are Python-scripting-only — not
  reachable from the declarative Secretion plugin. Added as a surprise.
- CORRECTED an over-assertion: the previous `state_io.writes` credited `runBeforeMCS=1` to the
  plugin. That flag belongs to the PYTHON steppable `SecretionBasePy`, a different code path. The
  compiled `SecretionPlugin` is in libCC3DSecretion.so; its intra-MCS ordering vs the diffusion
  solve is NOT readable from Python. Downgraded to a hint and moved the uncertainty into a
  surprise, so no one inherits it as fact.

Still could NOT establish (unchanged): the compiled plugin's write ordering relative to the PDE
solve, old-vs-new field buffer, and on-contact neighbour-iteration order. No evidence run exists,
so mode semantics remain inferred from method names + amoebae_2D XML, not measured.

### Addendum (third pass, read the OpenCL secrete KERNELS — actual arithmetic, not method names)

The two prior passes inferred semantics from FieldSecretor symbol names + XML. I read the actual
GPU arithmetic in `cpp/CompuCell3DSteppables/OpenCL/DiffusionKernel.cl` (the DiffusionSolver's
embedded secretion; the same three modes). This CONFIRMS additive/overwrite/on-contact and adds
three things the name-level reading could not see:

- ZERO IS A NO-OP GUARD, not a clamp-to-zero. Every mode is wrapped in `if (value) { ... }`
  (L216 plain, L262 constant, L314 on-contact). So constant-mode value=0 does NOTHING — it does
  not pin the field to zero. Added as a surprise.
- ON-CONTACT IS NON-ACCUMULATING in this kernel: base conc `c0` is read ONCE (L301) before the
  neighbour loop, then each qualifying neighbour re-assigns `c := c0 + rate` (L315/335) — so N
  contacts do NOT deposit N*rate and the LAST matching neighbour type wins. This DIVERGES from the
  entry's additive `phi += r` equation (correct only for a single contact). I flagged it in the
  note but did NOT rewrite the entry's equation, because this is the GPU DiffusionSolver path and I
  did not disassemble libCC3DSecretion.so to confirm the standalone plugin behaves identically.
- VOLUMETRIC SOURCE: plain/constant write every owned pixel, so total mass scales with cell volume
  (a big cell secretes proportionally more). Added as a surprise.
- Kernel-only detail I did NOT promote to the entry: on-contact uses a medium sentinel of id == -2
  (`NON_CELL`), with medium id == -1 (L302). Whether the plugin's C++ uses the same sentinel is
  unverified, so I left it out of the record to avoid overclaiming.

Net: the OpenCL path corroborates the three declared modes and shows the kernel ALSO implements the
uptake sink the prior pass found by name (`c -= min(c*relUptake, maxUptake)`, L221-233) — confirming
the "Python spec is a strict subset" surprise from a second source. Still unverified: byte-identity
between this GPU kernel and the standalone plugin's compiled CPU path.

### Addendum (resubmission pass)

Re-verified `code_path` L4306 = `class SecretionPlugin` (unmoved) and the `from_xml` mode mapping
(L4435-4451). Made `paper_section` honest: we have NO extracted paper text for this target, so the
anchor now says so and names the source (PyCoreSpecs.py + amoebae_2D XML) as the only evidence,
rather than implying I read a paper section. No analytical claims changed.

### NORMALIZER

**Verdict: refinement of `deposit`** (implementation_of: deposit). Secretion is the stigmergy
`deposit` verb — a cell adds to a scalar field at the sites it owns (deposit.py REFERENCE =
Grasse 1959, an ant depositing pheromone at its position). Transport is left to the separate
`diffuse` solver, so there is no composite: the standalone plugin is deposit and only deposit.
Plain additive mode alone would be a clean alias/implementation; the refinement is forced by the
plugin's other two modes. ConstantConcentration OVERWRITES/pins (phi := k) whereas deposit only
adds and clamps to 1 — needs a `mode: add|set` widening. SecretionOnContact gates the write on a
heterotypic-boundary predicate — needs an optional boundary write-domain + a neighbour-type read,
where deposit as registered reads only `pos`. Both are the same shape: deposit's unconditional
additive point-scatter must become mode-selectable and optionally domain-restricted over a cell's
pixel SET. The volume-scaled volumetric write is a set-vs-point primitive gap, not a contract
field, so it does not move the verdict.

**Strongest argument against.** You can read ConstantConcentration and SecretionOnContact as
*different verbs* wearing one plugin's name — in which case the honest verdict is `alias` for
plain Secretion plus possibly one `new` contract for the Dirichlet field-pin, not a single
refinement. A pin (phi := k re-applied each step) is arguably not "depositing" at all: it is a
boundary condition ON the field, closer to a field-level clamp than to an agent adding mass, and
folding it into deposit via a `mode` flag risks papering over exactly the source-vs-clamp gap the
loop warns against. I chose refinement because all three modes share one declarative surface, one
per-(field,type) value, and one set->field write target, so they read as a single contract with a
mode axis. But if the ledger prefers clamp/pin kept distinct from additive source, this should
split into alias(deposit) + one `new` Dirichlet-pin contract.


---

## surface

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


---

## surfacetracker

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

---

## Normalizer verdict — refinement of `aggregate` (implementation_of: aggregate)

`surface` is a per-cell scalar reduction — the cell PERIMETER: reduce the cell's own lattice
sites (via their hetero-owner boundary links) onto ONE per-cell number, incrementally, read-only,
no energy / acceptance delta. That is the same children→parent reduction shape (along the
containment map) as its two siblings: CenterOfMass was **aliased** to `aggregate` (first moment),
MomentOfInertia **refined** `aggregate` (second-moment tensor). So this is the aggregate family,
not a new contract — hence `implementation_of: aggregate`, and the ledger counts aggregate ONCE
across the three, not thrice.

**Refinement, not alias**, because two signature fields widen past the registered centroid (which
writes cell.pos and reads only its own children + occupancy): (1) WRITES — a scalar boundary-link
count / perimeter, not a position vector; (2) READS — and this is the genuinely new edge, stronger
than MomentOfInertia's — the summand reads the cell_id **partition** (the OWNER of each neighbour
site: is it c or non-c, Medium counting as non-c) to decide which links are hetero faces. The
centroid depends only on the child set's coordinates; surface depends on the surrounding partition.
Incremental maintenance and the `lmf.surfaceMF` weight are implementation/kernel, not contract.

**Strongest argument AGAINST:** surface(c) is exactly the weighted node-degree of NeighborTracker's
contact graph — surface(c) = Σ_b A_cb over ALL neighbours b (Medium included), modulo the lmf
weighting. NeighborTracker was ruled a refinement of `radius_graph` (contact graph, edge weight =
commonSurfaceArea). If a cell's interface with its non-self surroundings is fundamentally
*relational*, then SurfaceTracker is the degree-marginal of that graph — a reduction OVER incident
contact edges — and belongs to the `radius_graph`/contact-graph rewire contract as its
weighted-degree readout, not to `aggregate`. Under that framing the true children are edges, not
sites, and the operator consumes topology rather than being a fresh aggregate variant. I reject it
on two grounds: (1) at source SurfaceTracker is INDEPENDENT of NeighborTracker — it computes
`surface` directly from the pixel partition via `field3DChange` and never reads the neighbour table,
so the graph-marginal identity is a numerical coincidence, not a dependency; and (2) a scalar
node-marginal is a *reduction* (aggregate/hierarchy) whereas building who-touches-whom is
*rewire/topology* — a different kind. But the coincidence is real: if the algebra ever unifies
"per-cell interface scalar" with "sum of incident contact-graph weights," surface and NeighborTracker
collapse onto one object and this refinement folds into that.


---

## uniforminitializer

<!-- uniforminitializer -- append below; the driver merges this into campaign/analysis.md -->

## uniforminitializer (UniformInitializer) — excavated

Read `PyCoreSpecs.py:L5742-5953` (`UniformInitializerRegion` + `UniformInitializer`), the base
steppable `_PyCoreSteppableSpecs:L512` (generate_header emits `<Steppable Type="UniformInitializer">`),
and the CC3DML generator `CC3DMLGeneratorBase.py:L1189-1226` whose comment names it exactly:
"Initial layout of cells in the form of rectangular slab."

- **Same shape as blobinitializer: a set CONSTRUCTOR, not an energy term.** It runs once before
  MCS 0 and WRITES the cell field — but tiles a rectangular BOX instead of a disk/sphere. Per
  region it lays cubic cells of edge `width` on a grid with pitch `width+gap`, gap-sites left as
  Medium, each block a fresh cell id with a type from `cell_types`. `state_io.writes` says so
  plainly (cell.id + cell.type over the box); reads nothing dynamic.
- **The Python class only serializes CC3DML.** BoxMin/BoxMax/Gap/Width/Types emitters
  (`xml` property L5795-5809, region L5804-5808). The actual tiling + type draw live in the
  compiled core, not importable here, so `equations:` is a reconstruction from the emitted fields
  and standard CC3D semantics, flagged UNVERIFIED.
- **Sharpest gap I could NOT close: type-assignment rule.** The spec emits only a comma list of
  type names. Whether the core assigns block types by uniform-RANDOM draw (CC3D lore) or CYCLIC
  round-robin is not encoded anywhere readable. Flagged in surprises — a reimplementer must not
  assume deterministic cycling. (Contrast blobinitializer, whose note asserts random-by-convention;
  I declined to assert it here since I have no stronger evidence than the same convention.)
- **Pitch trap:** block pitch = `width+gap`, cell volume = `width^dim`; folding gap into the cell
  size is the obvious error. `gap` guarded non-negative, `width` guarded >=1 (check_dict, L5745).
- **Ordering:** `UniformInitializer(*_regions)` takes many regions positionally, each validated to
  be a `UniformInitializerRegion`; boxes may overlap, so later regions can overwrite earlier —
  recorded as an ordering assumption.

**Could NOT establish** (all compiled-core, not readable here, stated as such in the entry):
random-vs-cyclic type assignment; BoxMax inclusivity and far-edge partial-block handling
(dropped vs clipped); and the empty-`cell_types` meaning (all-Medium vs default type). **No paper
text available** — `paper_section` keeps the chapter reference and adds checkable SOURCE anchors
(class L5833, region L5742, generator comment L1197); no page/figure invented. Not one of the six
mechanisms with reference runs, so source-read only, no measured evidence.

### re-excavation pass (working copy restored after an out-of-band edit to atlas_record.yaml)

Confirmed two more source-readable facts and added them to `surprises:`:
- **width has no usable default.** `__init__` defaults `width=0` (L5754) but the guard rejects
  `width < 1` (L5747) — a region built without an explicit width fails validation. `gap` does
  default to a valid 0 (guard only rejects `<0`), and `from_xml` (L5905/5907) treats Width as
  required, Gap as optional — matching the guard asymmetry.
- **Types round-trips as a comma-joined string** (`xml` L5808 → `from_xml` L5911 split-on-comma,
  spaces stripped): the ordered Python list survives only as text order, which is the only place
  any assignment ordering could live. Still does not resolve random-vs-cyclic — the draw is core-side.

### normalization

**Verdict: `out_of_scope`, `implementation_of: seed`.** A one-shot that runs once at MCS 0 and
CONSTRUCTS the initial cell partition (tiles axis-aligned boxes with cubic cells of edge `width`,
`gap` medium between them, each stamped with a palette type). It is not a per-step operator
returning a delta over pre-existing state, so it is out of scope for the dynamical operator
algebra whose completeness this campaign measures. It is the SAME `seed` contract its sibling
BlobInitializer introduced — box-grid clip vs spherical-blob clip is the only difference — so it is
counted once via `implementation_of: seed`, NOT a second `new` (BlobInitializer, this, and
PIFInitializer are three interchangeable implementations of one initial-partition builder).

**Strongest argument AGAINST:** one could hold that population seeding IS a real capability the
algebra is simply missing — every model needs an initial partition, and CC3D exposes three
first-class, validated, parameterized ways to build one, which is exactly the multiplicity a
genuine contract shows; calling it "plumbing" then understates a real gap. I still favor
`out_of_scope` because a Plexus operator is defined by returning a per-step delta the engine
integrates over an EXISTING set/field, and IC construction has no such shape — but I typed and
logged the `seed` contract and marked this `implementation_of` it, so that if the campaign later
rules initialization in, the accounting is already correct and does not re-inflate on the second
and third sightings.


---

## volume

<!-- volume -- append below; the driver merges this into campaign/analysis.md -->

## volume (VolumePlugin) — excavated

Read `PyCoreSpecs.py:L1975-2161` (VolumeEnergyParameter + VolumePlugin) and the generators/
metrics under `_oracle/_evidence/volume_constraint_{on,off}` + `log/atlas_cc3d/_ablations.json`.

- **It's an energy term, not an update.** `E_vol = Σ λ_V (V−V_target)²` over real cells; the plugin
  returns dE for a proposed pixel copy and writes nothing. V(σ) is a lattice-site COUNT. State_io
  says so plainly rather than forcing read/write language.
- **Two-plugin split surprised me most.** The *count* isn't kept by VolumePlugin — it's maintained
  incrementally by a separate auto-loaded `VolumeTrackerPlugin` (CellGChangeWatcher,
  CompuCell.py:L5554). Reimplementers who fold "recount" into the energy term, or forget the
  tracker, read a stale/zero volume. Worth flagging for the normalizer: the "volume" contract may
  really be two — a watcher that maintains a per-set count, and an energy that reads it.
- **A pixel copy is a two-cell event** (gainer V+1, loser V−1); the quadratic doesn't cancel, so dE
  couples both cells' distance-from-target. Medium (id 0) is exempt.
- **Ablation is now MEASURED, not guessed:** λ=0 → n 45→0, volume 25→0 by MCS 200 (OFF run), vs
  ON relaxing 25→59.4 toward target 60. Volume is literally what stops a cell dissolving into
  medium under positive contact energy. Kept the earlier per-type-vs-per-cell surprise (measured
  25→20.9 shrink); confirmed the ON evidence run uses PER-TYPE params, while the growth runs use
  BARE mode — the two paths are mutually exclusive.
- **Guards:** only target_volume is checked (≥0). λ is unchecked (the dissolving 0 passes), and
  there is no coupling guard to the Potts temperature — a large λ vs fixed fluctuation_amplitude
  freezes the boundary, so the constraint's realized effect is inseparable from a Potts param.

**Could NOT establish:** the compiled `changeEnergy` was not read (cc3d C++ is not importable
here), so the exact quadratic form (no leading ½, Medium exemption) is reconstructed from the
CC3DML declaration + standard CPM convention, not verified byte-for-byte. **No paper text is
available** — `paper_section` names the chapter's known home for the term but is not a page I
have read; I did not invent a page/equation number. Confirmed there is NO VolumeFlex/steerable
variant in PyCoreSpecs — VolumePlugin (L2033) is the sole volume energy spec.

## Normalizer verdict — `new`: `volume_elasticity` (lateral/mechanics, set: cell)

Verdict **new**. Volume is the direct sibling of `surface`→`membrane_tension`: the same
quadratic Hookean, two-cell-incremental, writes-nothing energy shape, but on the cell's bulk
site COUNT instead of its boundary count. No registered contract covers an elastic set-point on
cell SIZE — cell_grow only *writes* the target volume forward (a state-writing structural
update), the opposite state_io — so widening it would fuse writer and energy term and erase the
read/write distinction the campaign measures. Named `volume_elasticity` (parallel to
`membrane_tension`, `elongate`); implementation_of null (first sighting — jax-morph's
`relax`/`regulate` are generic homeostasis verbs, not this energy).

**Strongest argument AGAINST `new`:** `volume_elasticity`, `membrane_tension`, and `elongate`
are arguably three implementations of ONE generic contract — a quadratic penalty on a scalar
geometric descriptor of a cell relative to a set-point (bulk count / boundary count /
inertia-axis length). Minting a fresh `new` contract per descriptor is exactly the yield
inflation this exercise warns against: it would measure our naming habits, not the language, and
push the saturation curve up by three when the real novelty may be one. The counter is that each
reads a *different* tracker and encodes a *distinct* biophysical force (bulk incompressibility vs
surface tension vs axial spring), and collapsing them hides that — but the tension is real, and
if a fourth "quadratic-constraint-on-geometry" plugin appears we should seriously consider
retro-fitting a single `geometric_setpoint` contract with these as implementations.

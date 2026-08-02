

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

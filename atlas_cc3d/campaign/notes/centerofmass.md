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

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

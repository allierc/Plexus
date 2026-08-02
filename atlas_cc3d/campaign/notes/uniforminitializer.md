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

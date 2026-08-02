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

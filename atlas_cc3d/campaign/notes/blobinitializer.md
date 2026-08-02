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

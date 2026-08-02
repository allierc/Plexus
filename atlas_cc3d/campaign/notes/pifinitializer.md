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

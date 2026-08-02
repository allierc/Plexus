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

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

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

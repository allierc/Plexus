<!-- cell_as_lattice_domain -- append below; the driver merges this into campaign/analysis.md -->

# cell_as_lattice_domain — normalizer note

**Verdict: `new`** (contract `occupy`, aggregate/hierarchy). CC3D's cell is a set of lattice sites
sharing an id; V, S, COM are derived from the label field sigma. The Plexus algebra's agents are
points, so it has no primitive for an extended, deformable lattice domain with a first-class exact
surface. Placed at aggregate/hierarchy on purpose — same kind/family as the registered `aggregate`
(Centroid) — to keep the alias tension honest rather than hide the representation in a distant corner.

**Strongest argument AGAINST (i.e. this is really a `refinement` of `aggregate`, not `new`):** the
registered `aggregate`/Centroid contract *already* computes a cell's position as the occupancy-weighted
centroid of its contained children and writes it as a derived readout — that is precisely COM = mean of
sites. If you read the site→cell label field sigma as just another (dynamic) instance of aggregate's
`parent` containment map, and treat V = |sites| as a trivial count-reduction over the same members, then
two of the three derived quantities fall straight out of the existing contract; nothing about aggregate's
*operation* changed, only how the member set is stored (a field partition vs a member list). A reviewer
could fairly call that a widening of aggregate's `set`/`reads`, not a new contract. My rebuttal is that
(a) surface S is an *inter-parent* boundary count that aggregate cannot express at all — it reads
neighbouring cells' labels, not a parent's own children — and (b) swapping aggregate's member set from
persistent point children to a mutable partition of a shared lattice is a substrate change that breaks
every current point-agent user, so it is different-in-kind, not wider. But the COM overlap is genuine and
I record it in `why:` rather than pretend the two contracts are disjoint.


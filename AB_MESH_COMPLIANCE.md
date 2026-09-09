# `vertex_ops` against plexus2 chapters 1-5, and the half-edge promotion

Branch `vertex-ops-plexus2-compliance`. Two jobs, in order: say precisely where `vertex_ops`
diverges from the paper, then make the mesh a declared object rather than a private table.

**Nothing here is started.** This file is the audit and the design; the branch exists so the work
does not land on `main` beside a promotion that is mid-flight.

---

## 1. The audit: where `vertex_ops` already complies

Better than expected, and the two real gaps are not the ones that look wrong at first.

### Kinds -- COMPLIANT
Every operator in the module registers one of the paper's eight `KIND`s: 5 `lateral`, 1 `rewire`,
2 `seed`, 23 `structural`. No invented kind.

### Purity -- COMPLIANT, AND THIS IS THE ONE THAT LOOKS WRONG
`cell_grow` and `cell_die` write `m["V0f"]` directly rather than returning a delta, and at first
reading that violates the engine's collect-deltas-then-integrate discipline. It does not. The paper
is explicit (§Kind and purity): *"Dynamics operators are pure: `forward` returns a per-set delta and
the engine integrates once per tick; `field`/`rewire`/`structural`/`seed` operators mutate the
field, the relation, the membership, or the initial state **in place** and return nothing."* Both
are `structural`. Writing in place is their contract.

The five `lateral` operators do return deltas, and `cell_mechanics[apicobasal]` returns two -- one
per `(set, block)` -- which is what R1(a) was built for.

### THE GAP THE PURITY RULE LEAVES OPEN, and it is a real one

The delta discipline is what makes two dynamics operators on one set COMPOSE: the engine sums them.
Structural operators have no such rule. Two of them may write the same buffer and the last writer
wins, silently, at whatever relative rate their `every:` keys give them.

**That is not hypothetical and it is not new.** R6 lost every extrusion to it: `cell_die` shrank
`m["V0f"]` and `cell_grow` overwrote it wholesale from `m["V0f_init"] * s**3` on the next frame, so
three ticks in four erased the shrink and the fourth was erased immediately after. 52 cells marked,
zero extruded.

And the paper already describes this exact failure, one level down, as the reason `seed` was made a
kind of its own (§Operators, on Seed vs Divide/Die): an initial condition *"re-applied on every
timestep rather than once ... silently overwrites the state of any operator writing to the same"*
field. **The pattern was diagnosed and the instance was fixed; the class was not.**

Worth considering on this branch, in increasing cost:

1. **A declared writer set.** Structural operators already declare `m["face_carry"]`; let them
   declare what they WRITE, and have the engine warn when two declare the same buffer. Diagnostic
   only, no behaviour change, catches the whole class at load time.
2. **A composition rule for the buffers that need one.** `V0f` wants growth and death to compose
   multiplicatively, which is a product of factors rather than a sum of deltas.
3. Nothing, and rely on gate rows. This is what happened; it cost R6 a rung and was found only
   because `divisions_and_deaths_both_happened` had been written to catch a vacuous population.

### Aggregate and Broadcast -- ABSENT, THOUGH BOTH ARE RUNNING

`vertex_ops` registers no `aggregate` and no `broadcast` operator, yet `face_geometry_3d` performs
both on every call: `index_add(0, ef, ...)` is $\sum_\pi$ from half-edges to cells, and `pos[es]` is
$\pi^{*}$ from half-edges to vertices. The two cross-level families are being executed as tensor
indexing rather than declared as operators, which is section 2 of this document.

---

## 2. The promotion: `half_edge` as a declared set

### Why `cell_set:` exists and what it is standing in for

A spec declaring `mesh: half_edge` must also declare `cell_set: cell`. That key is none of the
paper's three maps (`parent` for containment, `pre`/`post` for incidence). It is a BIJECTION between
the cell set's rows and the mesh's FACES, and the faces are not a set at all -- they are a table
derived from `E_srce`/`E_trgt`/`E_face`.

It cannot be `parent`, and the reason is the whole design constraint: **$\pi$ has to be a function,
and vertex -> cell is not one.** On a trivalent mesh a vertex belongs to three cells. A function
partitions the children into fibres and it is that partition which makes Aggregate conserve; a
many-to-many relation admits no canonical weighting.

### The route that needs no new primitive

A relation $R \subseteq A \times B$ **is** a set $R$ with two functions $R \to A$ and $R \to B$. Make
the relation a set and both legs are ordinary containment maps. `plexus2.tex` §Hierarchy now says
this; a synapse set with `pre`/`post` is the existing instance.

So: a `half_edge` SET, with $\pi$ to `vertex` and $\pi$ to `cell`. Every half-edge has exactly one
source vertex and exactly one face, so both legs are functions. `cell_set:` then retires entirely --
the face-to-cell pairing becomes the codomain of a declared map instead of a string parameter, and
`edge_flip` could not renumber a set it never declared it needed, which is the defect that put the
key there in the first place.

### What it costs

* A half-edge needs THREE legs -- source vertex, target vertex, face -- and `pre`/`post` supplies
  two. Either a third named map, or `pre`/`post` plus one $\pi$.
* It changes the topological master of every mesh spec in the repo. `edge_flip`,
  `divide_face_3d`, `face_collapse_3d`, `_check_closed` and the whole renumber path index the table
  directly. It needs its own gate ladder with a byte-identical twin against the pristine pin.
* `MESH_KINDS = ("half_edge",)` and `RESERVED`'s five keys are validated in `schema.py`; both move.

### It must be designed WITH `cell_complex`, not before it

Under the `cell_complex` mesh kind `nF != nC`, so the face-to-cell leg stops being a bijection and
becomes a genuine many-to-one -- which is the case $\pi$ was made for, and which also makes the
per-cell `uid` mandatory. Designing the declared-map form against today's mesh would design it
against an object already scheduled for replacement. **The two share one design; do them together
or do neither.**

---

## Order of work on this branch

1. The declared writer set (item 1 above) -- cheap, diagnostic, catches the class that cost R6.
2. `half_edge` as a set, designed jointly with `cell_complex`, behind its own gate ladder.

Neither should start until `AB_R7R8_TODO.md` section 0a is done on `main`: one volume convention
across growth, division, death and the energy. That is a smaller change, it unblocks a red row, and
it touches the same operators this branch would otherwise be rewriting underneath it.

# Restructuring `vertex_ops`: per-cell state, the half-edge set, and `cell_complex`

Branch `vertex-ops-restructure`, cut from `20eb3d06` -- **the working point**: 9 gates, 73 rows,
69 PASS / 4 KNOWN_RED / 0 FAIL, and R6 closed.

This is a restructuring, not a change of model. **Every rung below must be byte-identical to the
working point for every spec that does not opt in.** That is the whole discipline, and the reason
this is doable at all: there is a known-good state to hold against.

---

## What is being fixed, in one sentence each

1. **Per-cell state has two homes.** `area`, `cen`, `chem` live on the `cell` set as declared state
   blocks; `A0`, `P0`, `V0f`, `mg_scale`, `Vbirth`, `divjit`, `age`, `ndiv`, `alive`, `phase` live on
   the `vertex` set's mesh table as face columns. Same level, same cells, two stores, different
   carry mechanisms (`Hierarchy.renumber_set` versus `reindex_faces`/`face_carry`).
2. **`cell_grow` is declared `at: vertex` and is a cell-level operator.** Everything it writes has
   length `nF`. It attaches to the vertex set only because that set owns the table its arrays are in.
3. **`cell_set:` is a map the framework cannot express** -- a bijection between the cell set's rows
   and the mesh's faces, which are not a set. It is a string parameter, so `edge_flip` once
   renumbered a set it never declared it needed.
4. **`structural` means two things.** The paper defines it as Divide/Die -- changing entities. The
   code uses it as "may write in place". `cell_grow` creates and removes nothing and is registered
   `structural` purely for the write permission.

**Every symptom in the R3-R6 campaign traces to one of these.** `cell_die`'s shrink silently
overwritten by `cell_grow` (two structural writers, one face column, no composition rule); the
replay dropping `phase` (face columns read back by a hand-maintained literal); the division trigger
reading a wedge volume the model does not defend. None of these could arise for `area` or `cen`,
which live in the framework's own store.

---

## The byte-identical harness -- BUILD THIS FIRST, CHANGE NOTHING UNTIL IT IS GREEN

`tools/promotion_identical.py` compares okuda against core. This needs core against core: the same
spec, the same seed, before and after a refactor rung. Reuse its `_arrays` digest -- array by array,
`a.tobytes() == b.tobytes()`, reduced to one sha1 per run -- and its measured repeatability floor of
zero over 1,800 frames with division, T1 and chemistry live.

    tools/refactor_identical.py --ref <commit> --specs <list>

Pin `20eb3d06` as the reference, run the covering set on it once, store the digests, and re-run
after every rung.

### Coverage, measured

`vertex_ops` registers **31 operators across 7 contracts**. Of these, **24 are selected by at least
one spec in `config/` and 7 are selected by none**:

    cell_cycle[timer]  cell_cycle[transition_probability]  cell_cycle[inhibitor_dilution]
    cell_divide[adder]  cell_divide[concerted]  cell_die[lonely]  cell_mechanics[warp]

Six of those seven were written in this campaign and have never had a spec. **They need one each
before the refactor starts**, or the harness silently proves nothing about them. `cell_mechanics
[warp]` is the exception that matters most -- it is a second implementation of the energy and the
one most likely to drift unnoticed.

A greedy cover of the other 24 is 16 specs, most contributing one variant each. Do not use it as
found: `config/okuda/_superseded_*` and `_archive_*` are three of them, and a refactor pinned to
archived specs pins itself to whatever those were. **Write a dedicated covering set under
`config/refactor/`, one spec per uncovered variant, short (20-60 frames) and seeded.**

### The three properties the harness must have

* **Fresh both sides.** Never compare against a stored digest from a previous session, for the
  reason `promotion_identical` gives: an artefact proves the artefact has not changed.
* **Every recorded array, not the summary.** `vertex__pos`, `vertex__sep`, every `__mesh_*` column,
  every `cell__*` block. The bug this catches is a per-cell array carried through a renumber by the
  wrong mechanism, which no scalar summary would show.
* **The opt-in specs excluded explicitly, by name.** A rung that moves `A0` to the cell set changes
  the recorded layout for specs that adopt it. Those are listed, and everything else must be bit-equal.

---

## The rungs

Each is a commit, each is byte-identical for every non-opt-in spec, each is revertible alone.

### S0 -- the harness and the covering set. No source change.
Six specs for the uncovered variants, `tools/refactor_identical.py`, digests of `20eb3d06` recorded.
*Proves: the instrument reproduces the working point exactly, twice.*

### S1 -- a declared writer set. Diagnostic only.
Structural operators already declare `m["face_carry"]`; let them declare what they WRITE, and have
the engine warn at load when two declare the same buffer at different rates. **No behaviour change**
-- it would have printed a warning naming `cell_grow` and `cell_die` on `V0f`, which is the defect
that cost R6 a rung and was found only because a gate row had been written to catch a vacuous
population. *Byte-identical everywhere, by construction.*

### S2 -- per-cell state moves to the cell set, one array at a time.
`A0`, `P0`, `V0f`, `V0f_init`, `A0_init`, `P0_init`, `mg_scale`, `Vbirth`, `divjit`, `age`, `ndiv`,
`alive`, `phase` become declared blocks on the `cell` set. The topology operators permute them
through `renumber_set` like `area` and `chem`, and `face_carry` shrinks to what is genuinely
per-half-edge.

**One array per commit**, each byte-identical. The order matters: start with `phase` (newest, one
reader, added this campaign), end with `alive` (read by everything). `A0`/`P0`/`V0f` are the ones
with two writers and should move together with S1's diagnostic already in place.

*Then `cell_grow` becomes `at: cell` and `cell_set:` loses its first job.*

### S3 -- `half_edge` as a declared set.
A set with pi to `vertex` (source), pi to `vertex` (target) and pi to `cell` (face). All three legs
are functions, which is why this needs no new primitive: a relation is a set with two functions out
of it, and `plexus2.tex` sec. Hierarchy now says so. `index_add(0, ef, ...)` becomes a declared
Aggregate and `pos[es]` a declared Broadcast, rather than tensor indexing that happens to implement
them.

*Then `cell_set:` retires entirely* -- the face-to-cell pairing is the codomain of a declared map.

**`pre`/`post` supplies two legs and a half-edge needs three.** Either a third named map or
`pre`/`post` plus one pi. Decide this in S3's design note, not in its code.

### S4 -- `cell_complex`, designed with S3 and not after it.
Under `cell_complex` `nF != nC`, so the face-to-cell leg stops being a bijection and becomes a
genuine many-to-one -- the case pi was made for. It also makes the per-cell `uid` mandatory, and
`_check_closed` needs replacing (an I->H reconnection has dV=+1, dE=+2, dF=+1, which it refuses).

**S3 must not be committed to a design that S4 then has to undo.** If they cannot be designed
together, S3 stops at the design note.

---

## What must NOT change

* Every `half_edge` spec's recorded arrays, for every spec that has not opted in.
* The nine gates: 73 rows, 69 PASS / 4 KNOWN_RED / 0 FAIL. Re-graded at every rung, not at the end.
* `MESH_KINDS`, `RESERVED` and the recorded-arrays rule as narrowed at R0 -- or if they change, the
  rule is re-stated in writing first, as R0 required.

## Do this on `main` first

`AB_R7R8_TODO.md` section 0a: one volume convention across growth, division, death and the energy.
It is smaller, it unblocks a red row (AB-M4 at 11.74 h), and it touches the same operators S2 would
otherwise be moving underneath it.

# Align the codebase with the plexus2 operator algebra

## Context

The tree works — 9 gates, 73 rows, 69 PASS / 4 KNOWN_RED / 0 FAIL at `20eb3d06` — but it does not
implement the algebra its paper describes, and the R3–R6 campaign paid for the gap three times:
`cell_die`'s shrink silently overwritten by `cell_grow`, the replay dropping a face column, the
division trigger reading a volume the model does not defend. Each is a symptom of the same cause.

**The paper contradicts itself about "the eight".** §3 (the algebra) lists **Lateral, Aggregate,
Broadcast, Exchange, Rewire, Divide, Die, Seed**. §Reference implementation calls *"the eight of the
main text"* a different list — `lateral, aggregate, broadcast, exchange, field, rewire, structural,
seed` — which adds `field`, collapses Divide and Die into `structural`, and drops them by name.
`base.KINDS` implements the second, and `schema.py:316` enforces it at spec load. (Its own comment
at `registry.py:10` says "the seven in `base.KINDS`", of which there are eight — the drift is old.)

**Decision taken: §3 is canonical and the code changes to match. Scope is `vertex_ops` and the mesh
first; the other modules follow as separate campaigns.**

### What the audit found (live code, excluding the `candidates/` being deleted)

| kind | live | |
|---|---|---|
| lateral | 59 | correct |
| **structural** | **40** | ~20 are neither Divide nor Die |
| seed | 19 | correct |
| exchange | 19 | includes the operators that actually cross levels |
| **field** | **10** | not a family in §3 |
| rewire | 7 | correct |
| **aggregate** | **1** | `cell_geometry` |
| **broadcast** | **0** | — |

1. **Broadcast has zero registrations, Aggregate has one** — the two families §Hierarchy says are
   the *only* ones crossing the containment map. The traffic exists unlabelled: `mpm_ops.py:476`
   does `a_ext = a_cell[p.parent]`, which is π\*, inside an operator registered `exchange`;
   `face_geometry_3d` does `index_add(0, ef, ·)` (Σ_π) and `pos[es]` (π\*) with no operator at all.
2. **`structural` means "may write in place", not "changes the entities".** It absorbed 13 Modulate-
   like operators (`cell_grow` ×4, `cell_cycle` ×4, `junction_myosin` ×2, `cytokinetic_ring`,
   `ecm_gate_growth`, `bm_sense`), 5 constraint/projection ones, 1 harness (`topo_record`), and
   `ecm_load`.
3. **Per-cell state has two homes.** `area`/`cen`/`chem` are declared blocks on the `cell` set;
   `A0`, `P0`, `V0f`, `mg_scale`, `Vbirth`, `divjit`, `age`, `ndiv`, `alive`, `phase` are face
   columns on the `vertex` set's mesh table, carried by `reindex_faces`/`face_carry` instead of
   `Hierarchy.renumber_set`.
4. **`cell_set:` is a map the framework cannot express** — a bijection to the mesh's faces, which
   are not a set.

**(2) and (3) are one problem.** An operator is misfiled because its output is not declared state;
make the output a declared block and it becomes a properly-kinded Lateral emitting a delta.

### Intended outcome

Every `vertex_ops` operator registers a §3 family and means it; per-cell state lives on the cell
set; `half_edge` is a declared set whose two legs are ordinary π, so Aggregate and Broadcast are
declared rather than performed by tensor indexing; `cell_set:` is gone. No model changes except
where flagged, and every non-opt-in spec byte-identical throughout.

---

## Target mapping for the 31 `vertex_ops` registrations

| contract | now | target | note |
|---|---|---|---|
| `cell_mechanics` (5) | lateral | **lateral** | already correct |
| `edge_flip` (1) | rewire | **rewire** | already correct |
| `seed_mesh` (2) | seed | **seed** | already correct |
| `cell_divide` (6) | structural | **divide** | rename only |
| `cell_die` (12) | structural | **die** | rename only |
| `cell_cycle` (4) | structural | **lateral** | needs a delta — see S4 |
| `topo_record` (1) | structural | *not an operator* | see S5 |

---

## The rungs

Each is one commit, revertible alone, re-graded against all 9 gates.

### S-1 — the branch, before anything else.

    git checkout -b plexus2-algebra-alignment 20eb3d06

Cut from **`20eb3d06`**, the working point (9 gates, 73 rows, 69 PASS / 4 KNOWN_RED / 0 FAIL), not
from current `main`. A fresh branch rather than `vertex-ops-restructure`: that one holds the
*audit and design* documents for the mesh restructuring and should stay a document branch, while
this one carries source changes and must be revertible rung by rung. Its two documents
(`VERTEX_OPS_RESTRUCTURE.md`, `AB_MESH_COMPLIANCE.md`) are copied over in the first commit so the
branch is self-contained, and the plan itself is committed as `PLEXUS2_ALIGNMENT.md` before any
code moves.

*Nothing lands on `main` until every rung is green.*

### S0 — the harness and the covering set. No source change.
`config/tissue` is the comparison base (12 of 31 registrations covered). Add short seeded specs for
the uncovered variants — one per family for the ten `cell_die` and three `cell_cycle` models that
differ only in one predicate, and **real specs for `cell_mechanics[warp]` and `[marinari]`**, the
two that are neither cheap nor similar to anything covered.

`tools/refactor_identical.py --ref 20eb3d06 --specs config/tissue/*.yaml`, reusing
`promotion_identical._arrays` (array-by-array `tobytes()`, one sha1 per run; its repeatability floor
is a measured zero). Compares core-against-core, not okuda-against-core.

*Gate: the harness reproduces `20eb3d06` twice, exactly.*

### S1 — `divide` and `die` become kinds.
`base.KINDS` gains them; `field` and `structural` stay temporarily, since 10 `field` and ~22
`structural` registrations live outside scope. Re-kind `cell_divide` → `divide`, `cell_die` → `die`.
*Byte-identical by construction — `kind` is metadata the engine reads only for scheduling order.*

### S2 — per-cell state moves to the `cell` set, one array per commit.
The thirteen face columns become declared blocks; the topology operators permute them through
`renumber_set` like `chem`; `face_carry` shrinks to what is genuinely per-half-edge. Order: `phase`
first (newest, one reader), `alive` last (read by everything). `A0`/`P0`/`V0f` move together,
because they have two writers.

*Byte-identical. Verified per array, not per rung.*

### S3 — `cell_grow` becomes Lateral on the `cell` set.
With its targets now declared blocks it can emit a delta instead of writing in place, and
`at: vertex` / `cell_set: cell` become `at: cell`.

**Hazard to check first:** `s ← s·(1 + rate)` equals first-order Euler on `ds = s·rate` **only at
`dt = 1`**. `config/tissue` is all `dt: 1.0`, but `log/okuda_ECM` specs run `dt: 0.0032`. Either the
delta carries `1/dt`, or the rate is redefined and those specs are opt-in. **Decide before writing.**

### S4 — `cell_cycle` becomes Lateral.
A discrete per-entity state machine has no home in the eight. Recommended reformulation: a
continuous `cycle_progress` ∈ [0,1) as a first-order block, with `phase` read off thresholds and the
model varying the *rate* of progress rather than jumping the index — timer constant, sizer zero
until the checkpoint passes, `transition_probability` stochastic. That is integrable, so Lateral is
honest.

**This is a model change, not a refactor.** It gets its own gate rung and is not held to
byte-identity; `cycle_phases` is re-run and its phase-fraction panel compared before and after.

### S5 — retire `topo_record`.
Its own first line calls it *"a measurement, as an operator"*, and `engine._setup_recording` already
records the mesh. What it adds is the `hist` list `analyze_forces` reads. Move that to a spec-level
recording option; the eight have no harness family and should not gain one.

### S6 — `half_edge` as a declared set.
π to `vertex` (source), π to `vertex` (target), π to `cell` (face) — all three functions, which is
why no new primitive is needed: a relation is a set with two functions out of it, and §Hierarchy now
says so. `index_add(0, ef, ·)` becomes a declared **Aggregate** and `pos[es]` a declared
**Broadcast**. `cell_set:` retires — the face-to-cell pairing becomes a map's codomain, and
`edge_flip` can no longer renumber a set it never declared.

`pre`/`post` supplies two legs and a half-edge needs three; resolve in S6's design note, not its
code. Touches `edge_flip`, `divide_face_3d`, `face_collapse_3d`, `_check_closed`, the renumber path,
`MESH_KINDS` and `RESERVED`.

### S7 — `cell_complex`, designed with S6, not after it.
There `nF ≠ nC`, so the face-to-cell leg stops being a bijection and becomes the many-to-one π was
made for; it also makes the per-cell `uid` mandatory and needs a replacement for `_check_closed`.
**S6 must not commit to a design S7 would undo.** If they cannot be designed together, S6 stops at
the design note.

---

## Critical files

- `src/plexus/models/base.py` — `KINDS`, and the stale "seven" comment in `registry.py:10`
- `src/plexus/schema.py` — kind validation (`:316`), the `cell_set` validation (`:174`)
- `src/plexus/operators/vertex_ops.py` — re-kinding, `cell_cycle`, `topo_record`, the mesh helpers
- `src/plexus/operators/diffusion_reaction.py` — `cell_grow` (`:874`), `cell_geometry` (the one
  live Aggregate)
- `src/plexus/models/mesh.py` — `FACE_RECORD`, `FACE_ALIAS`, `reindex_faces`, `MESH_KINDS`
- `src/plexus/engine.py` — integration of the newly declared cell blocks
- `tools/refactor_identical.py` (new), reusing `tools/promotion_identical.py::_arrays`
- `config/tissue/*.yaml` — the covering set

## Verification

1. **Byte-identity** after every rung except S4: `tools/refactor_identical.py` over `config/tissue`,
   every recorded array, opt-in specs excluded by name.
2. **The gates**, re-graded every rung, not at the end:
   `PYTHONPATH=src python tools/run_gates.py --gate <id> --device cuda:0 --force` — must hold at
   9 gates / 73 rows / 69 PASS / 4 KNOWN_RED / **0 FAIL**.
3. **The suite**: `PYTHONPATH=src:tools python -m pytest tests -q` — 113 passed, and the 8 known
   pre-existing failures unchanged (7 `aggregate` KeyErrors, 1 `02_ecm_block` frozen-block drift).
4. **S4 only**: re-run `cycle_phases` and compare the phase-fraction panel and the cell-count curve
   against the recorded run.

## Out of scope, recorded

`field` (10) and the ~22 non-`vertex_ops` `structural` registrations — `mpm_ops`, `contact_ops`,
`membrane_ops`, `field_ops` — are stage 2. `KINDS` therefore keeps `field` and `structural` until
that campaign removes them. `AB_R7R8_TODO.md` §0a (one volume convention across growth, division,
death and the energy) stays open on `main`; it is independent of this work but touches `cell_grow`
and `cell_die`, so it should land before S3.

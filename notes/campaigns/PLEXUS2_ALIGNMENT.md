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

---

# Progress log

Appended as each rung lands, so the branch carries its own record and a reader does not have to
reconstruct it from `git log`. Every entry names the evidence, not just the change.

| rung | commit | evidence |
|---|---|---|
| S-1 branch + plan | `777f91aa` | cut from `20eb3d06`, the working point |
| S0 harness | `2f32a470` | `tools/refactor_identical.py`, core-against-core over every recorded array |
| S0 gate | `733ec22b` | 18 of 18 `config/tissue` specs byte-identical against `20eb3d06` |
| S1 divide/die kinds | `e8e61540` | 18 identical, every digest equal to S0's; suite and 9 gates unchanged |
| (gate repair) | `2a3a55dc` | `ab_thickshell` `n_frames` 20 -> 80; roll-up back to 69 PASS / 4 KNOWN_RED / **0 FAIL** |
| (defect fix) | `96c58a06` | daughters inherit the mother through `face_carry`; 17 identical, `cycle_phases` opt-in |
| S2a cycle state -> cell set | `fdd72d76` | `phase_t`, `cyc_inhib`, `cyc_vprev` on `cell`; 18 of 18 byte-identical |
| S2b phase -> cell set | `a77d025a` | one key renamed, 193,002 values over 402 frames unchanged; both renderer paths |

## Two things found on the way, both worth keeping in view

**The recorded working point was not reproducible, and nobody had checked.** `ab_thickshell`'s
AB-C4 row read FAIL 0.9275 against its `within [1.0, 0.02]` band on a fresh run at every commit
tested -- including `58693ed5`, the commit that CREATED the gate -- while the handover recorded
0 FAIL. The cause was not the model: the shell's apical:basal cap-area quotient dips to 0.8750 at
frame 5 as a minority of cells wedge to a basal point, and recovers to 1.0093 by frame 80, entering
the band at frame 48. The gate stopped at frame 20, so `reduce: last` graded a shell still relaxing.
Every number in that gate's prose came from a run made before the cap-area change that shares its
commit. **Lesson for the rest of this campaign: a tally is evidence only if the run that produced it
was made from the code being claimed about.** Each rung here re-grades rather than quoting.

**`face_carry` gave every newly born daughter a stranger's value.** `keep` indexes the rings list,
`divide_face_3d` appends daughter B beyond the end of a length-nF array, and `reindex_faces` clamps
-- so the daughter got cell `nF - 1`'s value. It was found by declaring `cell_cycle`'s arrays on the
cell set and running the two stores side by side, not by reading the code, and it is the third time
this campaign's premise has been confirmed: state kept where the framework cannot see it is state
nothing checks. See `96c58a06`.

## S2 as it is actually being done

Not one rung. The plan's "one array per commit" holds, but the order is set by RECORDING, not by
age:

1. **S2a -- the three unrecorded arrays**: `phase_t`, `cyc_inhib`, `cyc_vprev`. They are not in
   `MeshTable.FACE_RECORD`, so declaring them `record: false` on the `cell` set changes no
   trajectory key and the rung is byte-identical. `cell_cycle` gains a loud failure naming the
   three blocks and the set when a spec has not declared them.
2. **S2b -- `phase`**, which IS recorded and IS what the renderer colours by. Moving it renames
   `vertex__mesh_phase` to `cell__phase`; the renderer and `MeshTable.FACE_RECORD` change in the
   same commit and `cycle_phases` is an opt-in difference.
3. **S2c.. -- `A0`/`P0`/`V0f` together** (two writers), then `Vbirth`, `divjit`, `age`, `ndiv`,
   `mg_scale`, and `alive` last, which everything reads.

`MAY_MUTATE_INTEGRATED_STATE` flips False -> True on `cell_cycle` at S2a and that is a debt made
visible, not incurred: the operator has always written per-cell state in place, and could claim
False only because the state sat outside the tensor `engine._run_token`'s tick-0 invariant guards.
S4 removes the flag and the write together.

## S2c — the eight go together, and the reason is in the code

`A0`, `P0`, `V0f`, `Vbirth`, `divjit`, `age`, `ndiv`, `alive` are not eight independent arrays that
happen to sit on the same table. They are one unit, spelled as a literal tuple in three places:

- `cell_divide` reads all eight off the mesh into python LISTS, extends each list by one entry per
  daughter, and rebuilds every one through `keep` (`vertex_ops.py:1287`, `:1470-1494`);
- `cell_die` carries five of them by name, `("Vbirth", "divjit", "age", "ndiv", "alive")`
  (`vertex_ops.py:2235`);
- `edge_flip` carries all eight by name when a flip drops a face (`vertex_ops.py:3309`).

Moving one of them alone means running two mechanisms side by side inside the same function for
several commits -- a python list rebuilt through `keep` for seven of them, a cell-set block
renumbered by `renumber_set` for the eighth -- and the plan's "one array per commit" was written
before that was known. **The unit is the group.** ~137 references over five live files
(`vertex_ops`, `diffusion_reaction`, `contact_ops`, `membrane_ops`, `mesh`), plus
`tools/test_mesh_carry.py`.

Three of the eight are RECORDED (`A0`, `P0`, `V0f`, `age`, `ndiv` -- five, in fact), so the rung is
an opt-in for every spec that has a mesh, not just for one. That is a much wider blast radius than
S2a or S2b and it is the reason this note exists instead of a commit: the design should be seen
before it is executed.

### The question S2c has to answer first

`cell_divide`'s daughter is APPENDED, and the eight are given explicitly computed values there --
`A0.append(a0e)`, `age.append(0)`, `ndiv.append(ndiv[f])`. On the cell set the equivalent is a write
to row `nF + i` after `renumber_set`, which is a different shape of code from `cst[nF + i] =
cst[mother]` (a copy) and must not be confused with it: **half of these are not inherited, they are
COMPUTED at birth**, and `A0`/`V0f` are split between the daughters rather than copied to both.
Getting that backwards would double the tissue's target volume at every division, which is exactly
the extensive/intensive trap `reindex_faces`'s docstring warns about and `96c58a06` has already
shown this repo can fall into.

### Suggested order once the design is agreed

1. the five `cell_die` carries, which are pure permutation;
2. `cell_divide`'s rebuild, where the daughter values are computed;
3. `edge_flip`'s face-drop path, which is permutation again;
4. delete the three literal tuples.

## Where the branch stands

Five rungs, all green: `e8e61540` (S1), `2a3a55dc` (gate repair), `96c58a06` (daughter carry),
`fdd72d76` (S2a), `a77d025a` (S2b). Suite 113 passed with the same 8 pre-existing failures; all 9
gates re-run from scratch after every rung, 73 rows, **69 PASS / 4 KNOWN_RED / 0 FAIL**.

In `vertex_ops` the only operators still registering a kind that is not one of the eight of §3 are
`cell_cycle` (`structural`, waiting on S4) and `topo_record` (`structural`, waiting on S5). Both are
now unblocked: S4 became possible the moment `cell_cycle`'s state was declared, because an operator
can only emit a delta for state the engine knows about.

**S4 is a MODEL change and should be watched, not run unattended.** It replaces the discrete phase
index with a continuous `cycle_progress`, and `cycle_phases` is re-run and compared by eye rather
than held to byte-identity.

## One model observation, for whoever looks at `cycle_phases` next

With `96c58a06` in place the population cycles in visible WAVES: the phase fractions swing between
about 0 and 90 per cent with a period near 150 frames, and the cell count rises in a staircase with
plateaus around frames 160 and 241. This is not a defect introduced by the move -- it is what the
model does once daughters stop inheriting a stranger's phase. The old behaviour scrambled the
population at every division and hid it. Whether a real tissue should re-synchronise this strongly
under a `g1_size` checkpoint with `phase_cv: 0.15` is a question about the model, and the answer
belongs in the spec or in `cell_cycle`, not in the carry.

---

# The campaign is blocked, and this is the measurement that blocks it

**`gate_00_spheroid`'s three reference pins are byte-identity wearing a different name, and every
remaining rung of S2c fails them.** Not because the model changes -- because the run is chaotic and
the pins are exact.

## What was measured

S2c-2 moves `cell_grow`'s baseline (`mg_scale`, `A0_init`, `P0_init`, `V0f_init`) to the cell set.
Against `808ece7b`, same spec, `PLEXUS_STRICT_DETERMINISM=1`, on `gate_00_spheroid`:

| | ref | now |
|---|---|---|
| positions first differ | frame 33 | max \|dx\| **4.5e-06**, mean 4.2e-07, on 399 of 400 vertices |
| cell count first differs | frame 72 | 277 against 278 -- **one cell** |
| final cell count | frame 401 | **6,914 against 6,749** (2.4%) |

`edge_flip` reconnects on an edge-length comparison, so a perturbation at the 24th bit of a float32
coordinate decides one flip differently and the trajectories part. `promotion_identical`'s own
header already recorded this: *one differing mantissa bit would move an `edge_flip` decision and
diverge from then on*.

Four variants of the same change were measured -- 6,840 (move only), 6,749 (cap rewritten in
volume), 6,749 (plus explicit per-cell absorption of foreign writes), 6,639 (forcing the old global
reset). **Scattered, not shifted.** A different model gives a direction; divergence gives scatter.

## The pins are not broken -- that is the problem

`gate_00_spheroid` re-run at `808ece7b` on **cuda:1** returns `final_cell_count 6914`,
`t1_total 1499`, `apical_radius_fold 3.86064` -- identical to cuda:0. The run is bit-reproducible
across devices, so the pins are meaningful **for a fixed code path** and worthless for a refactor:
anything that changes floating-point association order fails them, and a refactor is nothing but a
change to association order.

    final_cell_count   eq 6914      <- the last frame of a 400-frame chaotic trajectory, as an integer
    t1_total           eq 1499      <- likewise
    apical_radius_fold within [3.8607, 0.002]   <- 0.05%, on the same trajectory

## Three ways out, and the choice is not a code decision

1. **Read the pins before the divergence amplifies.** The cell count is IDENTICAL to frame 72 and
   the positions agree to 4.5e-06 at frame 33. A pin at frame 50 would still catch a real
   regression -- a wrong split, a dropped column, a lost carry all show up in the first ten frames
   -- and would be immune to bit-level reassociation. This is the option that keeps an equality.
2. **Make them bands.** `within [6914, 0.05]` and `within [1499, 0.05]`. Cheap, but it weakens the
   only rows that currently catch a population-level regression, and 5% is a number chosen after
   seeing 2.4%.
3. **Freeze the reference per code-path.** Re-pin deliberately at each rung with
   `--freeze-reference`, which is what that flag is for, and accept that the pin certifies "this
   commit reproduces itself" rather than "this commit matches the paper's run".

Option 1 is the recommendation: it is the only one that keeps an exact assertion while measuring
something a refactor cannot legitimately change.

## What is on which branch

| branch | state |
|---|---|
| `plexus2-algebra-alignment` | `808ece7b`, **green** -- 9 gates, 73 rows, 69 PASS / 4 KNOWN_RED / 0 FAIL; suite 113 passed + the 8 in `TEST_KNOWN_FAILURES.md` |
| `s2c-2-growth-baseline` | `7f6ec238`, the parked rung, with the full diagnosis in its message |

`tests/test_state_census.py` pins the moves already made: `phase`, `phase_t`, `cyc_inhib`,
`cyc_vprev`, `Vbirth` and `divjit` are asserted to be declared blocks on the cell set and NOT
columns on the mesh table, and the remaining per-cell column count is bounded so a new one has to
be argued for.

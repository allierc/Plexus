# Discovery campaign — session log

**Purpose.** Cedric is AFK and reviews periodically. This file is the window into what happened,
what broke, what was decided, and what is next. Append-only; newest hour at the **bottom**.

**Conventions.**
- `HH:MM` entries are detailed running notes.
- `### ⏱ SUMMARY — <hour>` blocks are the hourly report: *done / found / decided / next / blocked*.
- Anything marked **⚠ DECISION** changed the design and should be reviewed.
- Anything marked **🔴 FINDING** is a defect or result that affects the science.

**Reference documents.**
- `discovery/plexus2_discovery.pdf` — the campaign specification (14 pp).
- `discovery/HANDOFF.md` — prior session state, milestone plan (§9), validation ladder (§10).

---

## Hour 1 — 2026-07-30 13:00–14:00 EDT

**13:05** Answered the standing question: **the Grounder** is the agent that reads
`paper/plexus2.tex` (the language contract), `papers/okuda.pdf` (the reference model) and
`papers/okuda_corpus.md` + `papers/` (the literature). It is called at three points — when the
Proposer needs a mechanism grounded, when the Supervisor gates a hypothesis before Discovery
proposes, and when an operator request is filed (does the literature already name this?). It is
the agent that produced the Gierer–Meinhardt unlock by hand last time.

**13:10** Scaffolded `/workspace/Plexus/discovery/{agents,campaign}`, `log/okuda`,
`config/okuda`. Moved `plexus2_discovery.tex` + `.pdf` out of `paper/` into `discovery/` and
rebuilt (14 pp, 0 errors).

**13:20** Wrote `discovery/run_record.py` — the evidence core, ported from
`prototype/SMG2_budding/discovery/` with four campaign-critical fixes. Smoke test passes.

**13:35** Wrote `discovery/composition_space.py` — the Okuda vocabulary (12 operators, typed
ports, 3 stage gates), the one-edit API, structural encoding for proximity clustering, post-hoc
naming, and two reference recipes.

**13:45** Wrote `discovery/translate.py` — the backend adapter, `CompositionGraph ⇄ runnable
Plexus spec`, targeting the same spec shape `run_tyssue_round.make()` produces.

**13:50** Wrote `discovery/validate_space.py` — the 8-part validation battery. First run:
**25/31**. Six failures, all genuine. Fixed three real bugs (below). Second run: **45/45**, and
10 configs written to `config/okuda/`.

**13:58** Committed `dc0fa52` and pushed to `origin/main`.

### 🔴 FINDINGS — hour 1

1. **`divide_3d` slots must be per-implementation.** `hertwig` splits normal to the cell's *own*
   longest axis and needs no morphogen input; `orient_iface` stacks daughters along the bud axis
   and does. Declaring `axis` unconditionally made every hertwig composition look like it had a
   dangling (inert) slot, and the compiler correctly refused to run it. Fixed with `impl_slots`;
   `set_impl` now also drops connections into slots the new implementation does not have.
   *Found by the battery, not by inspection.*

2. **`rd_interface_tension` was a duplicate of `extrude`.** In the engine that one operator
   carries both `K_purse` and `K_extrude`. Exposing it twice would let a single mechanism occupy
   two points of the search space and be counted as two hypotheses. Removed; `extrude` is the
   only node.

3. **`round_40_mc8` and `round_41_relax60` share a comp_hash** (`C5e315998af4`). This is
   **correct and consequential.** `relax_iters` is θ, so round 41 is *the same mechanism
   evaluated closer to equilibrium*, not a different mechanism. It means the R41 finding is
   properly stated as a parameter-sensitivity result — which is exactly what the **Q** metric
   (aspect after driver-off relaxation ÷ aspect before) is designed to measure, as part of every
   composition's evaluation protocol rather than as a separate composition. The identity rule
   worked as intended on its first real test.

4. **Defects D1/D2/D3 are visible in `run_tyssue_round.make()` line 505**:
   `dt = 1.0 if (cones and not rd) else 0.02`. Adding/removing the RD operators — the campaign's
   single most important edit — *also* rescaled chemical:mechanical time by 50×. Every signalling
   verdict would have been confounded. `translate.py` now emits **one global `dt = 0.02`**, and
   pins `every = 1` everywhere plus `topo_snapshot` stride = position stride.

### ⚠ DECISIONS — hour 1

- **Implementation choice is part of composition identity** when it changes the phenomenology
  (`impl_structural=True`): the three reaction kinetics, `shape_energy_3d` default vs monolayer,
  `divide_3d` hertwig vs orient_iface, `morphogen_growth_3d` conserve-amount vs not. plexus2 says
  implementations "differ only in numerics", which holds for finite-difference vs spectral
  diffusion but *not* for Brusselator vs Gierer–Meinhardt. Recorded as a deliberate departure in
  the module docstring — the campaign must be able to ask "which kinetics".
- **Archives are gitignored, ledgers and configs are tracked.** `discovery/_archive*/` and
  `**/traj/` excluded; `config/okuda/*.yaml` committed, each carrying its `comp_hash`.

### ⏱ SUMMARY — Hour 1 (13:00–14:00)

| | |
|---|---|
| **Done** | discovery/ scaffolded; evidence core, composition space, backend adapter, validation battery all written and passing 45/45; 10 configs generated; committed + pushed `dc0fa52` |
| **Found** | 4 findings above — 3 real bugs in the new code (caught by the battery), 1 confirmation that the identity rule behaves correctly on a real case |
| **Decided** | implementation ∈ identity when phenomenology-changing; archives gitignored |
| **Next** | actually *run* a generated config: data generation + movie + metric read. This is the part the battery cannot cover — it proves the spec is not just well-formed but executable. |
| **Blocked** | nothing |

**Validation state:** the space is proven *expressive* (every trusted recipe reachable) and
*faithful* (operator sets identical to the hand-written specs). It is **not yet proven
executable** — no generated config has been run. That is the next gate and it is the important
one.

---
## Hour 2 — 2026-07-30 14:00–15:00 EDT

**14:00** Wrote `discovery/run_one.py` — the executable half of the validation gate. Unlike
`run_tyssue_round.do()` it (a) **asserts** the recording alignment instead of clamping it,
(b) persists the full trajectory + per-frame metric table, (c) records the acted-ledger, and
(d) can run the quasi-static **Q** test.

**14:01** All 10 generated configs load through the real `plexus.schema.load` — the specs are
not merely well-formed, they are accepted by the engine's validator.

**14:02** First real execution. **The D3 assertion fired immediately**, and it was right.

### 🔴 FINDING 5 — the D3 guard caught a bug in *my own translator* on its first run

`ref_uniform_inflation` produced `positions=26 frames but topology=1`. Two causes, both mine:

1. **`seed_mesh_3d` was emitted without `before_frame: 1`**, so it rebuilt the sphere *every
   tick*. `hist` lives inside `_mesh`, so re-seeding wiped the topology history each frame —
   leaving exactly one entry. Under the old `hist[min(t, len(hist)-1)]` clamp this would have
   rendered every frame's coordinates against **frame 0's connectivity** and produced a
   spectacular, entirely fictitious result. It failed loudly instead.
2. **`vesicle_growth` was missing from `SCHEDULE_ORDER`**, so it sorted to position 999 — i.e.
   growth ran *after* the recorder. Added a compile-time guard: any operator missing from
   `SCHEDULE_ORDER` is now a hard error rather than a silent last-place sort.

This is the single best evidence so far that the pre-flight discipline is worth its cost. The
assertion cost nothing and caught a class of bug that previously cost days.

### 🔴 FINDING 6 — defect 1 confirmed verbatim in the recording operator

`TopoSnapshot3D.__init__` keeps `self.every` **and** `self._k`, and gates on them, while the
engine gates the same operator. The effective stride is the product. Confirmed by reading
`tyssue_ops3d.py`. `translate.py` emits `every=1` so the product is 1, but **the operator-side
counters must still be deleted** — the translation fix is necessary, not sufficient. Tracked as
task 9.

**14:02** `ref_uniform_inflation` runs clean: 26 aligned frames, `aspect_final = 1.014` (a
sphere — exactly right for uniform inflation with no patterning), 500 cells, 3.5 s wall.
`strip.png` + `movie.mp4` written to `log/okuda/`, minisite convention (black background).
Trajectory + per-frame metrics persisted to `discovery/_archive/traj/`.

**14:05** Launched `ref_round40_mc8` (the tube recipe) on `cuda:0`, 250 frames. 2 GPUs are
available locally; the L4 partition is untouched so far.

### ⏱ SUMMARY — Hour 2 (14:00–15:00)

| | |
|---|---|
| **Done** | `run_one.py` written; all 10 configs validated against the real schema; first end-to-end execution succeeds; artefacts (strip + movie) render correctly; evidence archived with full trajectory |
| **Found** | 2 findings — the D3 assertion caught a translator bug that would have fabricated a result; defect 1 confirmed verbatim in `TopoSnapshot3D` |
| **Decided** | operators missing from `SCHEDULE_ORDER` are now a compile error, not a silent sort |
| **Next** | confirm the tube phenotype appears in `ref_round40_mc8`; then the metric bank + Q, then the L4 driver |
| **Blocked** | nothing. The acted-ledger reports "not instrumented" for all operators — expected; the D4 operator-side fix is task 9, and until it lands every run is flagged provisional. |

**Validation state:** expressive ✅, faithful ✅, **executable ✅**. Not yet proven to *reproduce
archived phenotypes* — that is the run in flight.

---

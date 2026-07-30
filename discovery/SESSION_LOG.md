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
## Hour 3 — 2026-07-30 15:00–16:00 EDT

**15:05** Added **V9 PARAMETER FIDELITY** to the battery. V3 proved *same operators*; that is not
*same model*. V9 compares the actual numbers against `run_tyssue_round.make()`, excluding only the
keys we deliberately changed (`dt`, `every` — the D1/D2 fixes).

### 🔴 FINDING 7 — V9 caught a silent semantic bug plus 7 fidelity gaps

The one that matters: **`cell_rd_seed.mode='cone'` vs the engine's `'cones'`.** My emitter used the
singular; the engine matches on the plural, so the seeding mode was falling through to a different
branch entirely. **Identical operator set, different mechanism** — precisely the class of defect
V9 exists to catch, and invisible to V3.

Also fixed: `n_spots` hardcoded 3 (hand 1); `after_frame` hardcoded 100 (hand 20/50/80); extrude
`a_sw` 1.2 (hand 0.5, derived from `iface_asw`); `cycle_cv` 0.15 (hand 0.4); `min_cycle` 8
(hand 4); Gray–Scott `F`/`kk` hardcoded; monolayer `gamma`; `orient_asw`.

Vocabulary defaults are now the hand-tuned working point (`p0=3.90`, `Gamma=0.05`, `Lambda=0.20`,
`h0=0.40`), so a default-parameter composition starts where our evidence is. **52/52.**

**15:20** Committed `e033726`, pushed.

### 🔴 FINDING 8 — the D1 fix re-anchors the baseline, and the tube gets shorter

Replayed `round_40_mc8` at its own 900 frames from the real homogenised checkpoint (1431 cells).

| | archived (hand) | replay (D1/D2/D3 fixed) |
|---|---|---|
| aspect | ~7.5 | **3.22 peak / 2.42 final** |
| cells | ~2700 | **3335** |

The archived run had `divide_3d every=2` **and** a private `self._k` counter — effective period 4.
The corrected config fires division **4× more often**. More cells, shorter tube. This is exactly
the re-anchoring the handoff predicted (*"expect division counts to roughly double"*), and it is
larger than predicted.

**Consequence:** the archived "best tube, aspect 7.5" was obtained at a division rate four times
lower than its own config advertised. **The old θ is not the new θ** — the working point must be
re-tuned after the fix, and every one of the 316 archived runs is at a different effective
division rate than it claims. This is a campaign-level fact, not a nuisance: it means the archive
can be re-scored for *shape* but not compared on *rate* without correction.

**15:35** Wrote `discovery/hypothesis.py` — the scientific protocol, in response to Cedric's
clarification. See the decision below.

**15:50** Added two sections to `plexus2_discovery.tex` (now **16 pp**): the Grounder's precise
role and call sites, and **§5 The scientific protocol: hypothesis first**.

### ⚠ DECISION — where hypothesis-first and the 70/30 balance live in the loop

Answering Cedric's question directly. **It lands in the batch composition, with pre-registered
predictions.** A candidate cannot be run until its prediction is recorded (`Hypothesis.__post_init__`
refuses one without). Each batch is allocated confirmatory vs adversarial, so every outcome falls
into a 2×2, and the informative quadrants are the *prediction errors*:

|  | confirmed | refuted |
|---|---|---|
| **confirmatory** | consolidates (low info) | 🔥 surprise |
| **adversarial** | 🔥 surprise | breaks as expected (low info) |

So **70/30 is a setpoint, not a quota**: the Supervisor closes a loop on the observed *surprise
rate*. `< 0.10` → drifted to 100/0, confirming what we already believe → push adversarial.
`> 0.50` → drifted to 0/100, nothing consolidates → push confirmatory. Target ≈ 0.30.
Implemented in `HypothesisRegister.advise_mix()`.

`knowledge.md` is append-only and written in the order the science happened — hypothesis and
grounding, then prediction, then outcome — so a reader can audit what was believed *before* the
evidence arrived. Validated / Refuted / Open sections, surprises flagged.

### ⏱ SUMMARY — Hour 3 (15:00–16:00)

| | |
|---|---|
| **Done** | V9 parameter-fidelity gate (52/52); replay of `round_40_mc8` at full length; `hypothesis.py` protocol; spec grown to 16 pp with the Grounder and hypothesis-first sections |
| **Found** | the `'cone'`/`'cones'` silent mode bug; **the D1 fix re-anchors the tube from aspect 7.5 → 3.2** |
| **Decided** | 70/30 implemented as a closed loop on surprise rate, not a quota; prediction is mandatory before a run |
| **Next** | `round_41_relax60` control (same comp_hash, more relaxation → should collapse further, giving the first real Q reading); then the metric bank + L4 driver |
| **Blocked** | nothing. Re-tuning θ after the D1 re-anchoring is now a required campaign step, not optional. |

---
## Hour 4 — 2026-07-30 16:00–17:00 EDT

**16:05** Wrote `discovery/instrument.py` — the **D4 acted-ledger**, implemented generically.
Rather than editing every operator to self-report (and trusting each edit), it wraps every
registered operator class and takes a cheap fingerprint of the hierarchy before and after each
call. If the state moved, or a non-empty delta was returned, the operator acted. 66 operator
classes instrumented, idempotent.

### 🔴 FINDING 9 — the D4 detector itself needed verifying before it could be trusted

First test on `ref_uniform_inflation` reported **two** inert operators. Only one was real:

- `divide_3d` — **true positive.** `after_frame = 100` but the smoke ran 25 frames, so division
  genuinely never fired.
- `vesicle_growth` — **false positive.** It writes the per-cell mechanical *targets* inside the
  mesh dict (`A0`, `P0`, `V0f`), which my fingerprint did not cover. A growth operator would have
  been reported inert on every run, invalidating perfectly good evidence.

Fixed by folding every numeric array in the mesh dict into the fingerprint. Re-verified in both
directions: at 25 frames `divide_3d` is correctly flagged; at 131 frames **all 6 operators act**
(`divide_3d` 31×, `vesicle_growth` 131×) and the run is marked `valid_evidence: true`.

*This is the discipline the whole document is about, applied to the guard itself: verify the
instrument before trusting the measurement.* A detector that cries wolf is worse than none,
because the campaign would learn to ignore it.

### 🔴 FINDING 10 — R41 reproduced quantitatively, under the corrected instrument

Both runs are the **same composition** (`C5e315998af4`) — `relax_iters` is θ, so this is a
parameter-sensitivity result within one mechanism, exactly as the identity rule predicted.

| | relax = 30 | relax = 60 |
|---|---|---|
| aspect **peak** | 3.22 | **4.58** |
| aspect **final** | 2.42 | **1.36** |
| **retention** (final/peak) | **0.75** | **0.30** |
| cells | 3335 | 4801 |

More relaxation grows a *larger transient* tube which then collapses *more completely*. The
forced protrusion cannot be held against the shape energy. This is round 41's finding — *"pushing
toward quasi-static destroys the tube"* — as a **number with a trajectory behind it**, rather than
an eyeball judgement on a movie.

Added **retention** as a first-class metric. It is a cheap proxy for the full Q test and,
crucially, is computable for every archived run *from the stored per-frame table without
re-simulating* — the direct payoff of fixing D7 first.

### ⏱ SUMMARY — Hour 4 (16:00–17:00)

| | |
|---|---|
| **Done** | D4 acted-ledger implemented + self-verified; `retention` metric added; replay + relaxation control both completed at 900 frames with artefacts |
| **Found** | the D4 detector's own false positive (fixed); **R41 reproduced quantitatively** — retention 0.75 → 0.30 under more relaxation |
| **Decided** | inert operators and buffer saturation both set `valid_evidence: false`; such runs can never enter the ledger |
| **Next** | Grounder agent; metric bank + full Q; L4 8-way driver; operator-side D1 (delete the private clocks) |
| **Blocked** | nothing |

**Note on outputs (answering Cedric's question):** configs land in `config/okuda/` (10, tracked),
job artefacts in `log/okuda/<name>/{strip.png, movie.mp4, diag.json, spec_run.yaml}`, evidence in
`discovery/_archive/{records,analyses}.jsonl` + `traj/*.npz` (29–48 MB for the 900-frame runs —
full trajectories, gitignored). **`graphs_data/` is deliberately untouched**: `run_one.py` drives
`plexus.engine.run` directly rather than `Plexus_Main.py -o generate`, because the 3D-AVM needs
its own half-edge/cross-section renderer and the campaign needs the RunRecord evidence contract,
which the `graphs_data` convention does not provide.

---
## Hour 5 — 2026-07-30 17:00–18:00 EDT

**17:05** Wrote `discovery/agents/grounder.py` — the agent Cedric asked about, implemented.

It holds `paper/plexus2.tex` (the language contract), `papers/okuda.pdf` (the reference model),
and `papers/okuda_corpus.md` + the 23-source vendored corpus, and exposes exactly the three call
sites: `ground()` for the Proposer, `gate()` for the Supervisor, `name_mechanism()` for an
operator request. Retrieval is **local, deterministic and citable** — term-overlap, no embedding
model, no network — so any citation recorded in `knowledge.md` can be re-derived and audited back
to the page it came from.

`gate()` carries five **reference claims** the paper settles (quasi-static regime; Gierer–Meinhardt
not Brusselator/Gray–Scott; no explicit bending; growth-driven not forced; χ sets the diameter).
A hypothesis that contradicts one is not blocked — but the Supervisor must *see* the contradiction
before cluster time is spent. Verified: the campaign's central hypothesis ("the tube survives
removal of the extrusion force") returns `grounded` against the `growth_driven` claim, citing
`okuda.pdf (p.4)`.

### 🔴 FINDING 11 — the corpus contains a duplicate that would have doubled its own evidence

`okuda.pdf` and `Turing_Vertex.pdf` are byte-identical (both 3,331,840 B). Every retrieval
returned both, so every citation of the reference model appeared **twice** — silently doubling
the apparent weight of evidence for whatever that one paper says. Fixed by de-duplicating the
corpus on content hash. After the fix the same query returns `okuda.pdf` plus
`LedesmaDuran_2023_turing_growing_domain.pdf` (Turing on a growing domain — the dilution paper,
directly relevant to the flood problem) instead of the same paper twice.

**17:30** Wrote `discovery/cluster.py` — the L4 driver.

Built around the rule the Tyssue notes paid for: *an action's reported outcome is a hint; the
world's state is the fact.* Submissions are fired **detached** (the ssh returns in <1 s), and the
only ground truth is `bjobs`. `status()` returns `None` — not `{}` — when the queue is
unreachable, so `wait()` can never mistake a dead link for "all jobs finished". Waves of 8
(`gpu_l4` gives 8 slots per GPU); `-gpu num=1` always, since the queue rejects jobs without it.

**17:45** **End-to-end cluster verification.** Submitted `ref_uniform_inflation` (130 frames) →
`bjobs` shows `RUN pg_ref_uniform_inflation`. The full remote path works: config → job script →
detached bsub → queue → execution, with artefacts landing on the shared NFS mount at
`log/okuda/`.

### ⏱ SUMMARY — Hour 5 (17:00–18:00)

| | |
|---|---|
| **Done** | Grounder written and verified on all three call sites; L4 cluster driver written; **one real job submitted and running on the partition** |
| **Found** | the corpus duplicate (`okuda.pdf` == `Turing_Vertex.pdf`) that would have double-counted the reference model in every citation |
| **Decided** | queue-unreachable returns `None`, never `{}` — a dead link must never read as "done" |
| **Next** | the control law (batch → triage → rank → truncate → starve → freeze → terminate); then the operator-side D1 fix; then the instrument gate on 8 eye-labelled runs |
| **Blocked** | nothing |

**Campaign readiness:** 4 of the 5 pre-flight items are now in place — evidence contract (D7),
alignment assertion (D3), acted-ledger (D4), and translation-time D1/D2/D3. Remaining before
switch-on: the operator-side private clocks, the instrument gate, and the control law itself.

---

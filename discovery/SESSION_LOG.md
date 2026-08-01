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
## Hour 6 — 2026-07-30 18:00–19:00 EDT

**18:10** Wrote `discovery/control.py` — the anti-rabbit-hole control law, as a **deterministic
script**. Robin's authors found their orchestrator "almost always called tools in the same order"
and replaced it with a notebook; we do the same. Language models are called only where judgement
is required (propose, watch, explain). Ranking is by measuring.

Components, each answering one named pathology:

| Pathology | Mechanism |
|---|---|
| depth-first drift | `propose_batch` draws across **distinct clusters**, one legal edit each; `truncate` keeps top-K and **drops the losers, never refines them** |
| no terminal state | `Supervisor.terminal()` — computed from statistics, not patience; `escalate()` opens the next stage gate or files an **operator request** |
| near-duplicates | `ProximityIndex` clusters on structural distance; a cluster with 6 evaluations and no best-score gain is **frozen** and its budget reallocated the same round |
| eye/number divergence | any run with an inert operator or a saturated buffer scores `-inf` and can never win a tournament |
| goal drift | `CampaignConfig` holds objective + success criteria + stopping rule; workers cannot amend it |

`rank_btl()` implements Bradley–Terry–Luce over pairwise comparisons (Robin's fix for position
bias, which simple win/loss tallies are highly susceptible to) — but **the comparator is the
metric bank, not a debate**. Verified: BTL recovers a strict order from pairwise comparisons.

The success criteria are authored **before** the search and are falsifiable:
`aspect_final ≥ 3.0` ∧ `retention ≥ 0.6` ∧ `Q ≥ 0.5` ∧ **achieved without the extrude node**.

**18:40** ✅ **FULL STACK VERIFIED ON THE L4 PARTITION.**

`pg_ref_uniform_inflation` ran on `gpu_l4`, host `8*h08u18`, `cuda:0`, 130 frames, 63.5 s wall /
107 s CPU, 1336 MB peak:

```
{"saturated": false, "inert_operators": [], "retention": 0.992,
 "valid_evidence": true, "aspect_final": 1.018, "n_cells_final": 5449, "frames": 131}
D4 ok: all 6 scheduled operators acted ({'seed_mesh_3d': 1, 'divide_3d': 31, 'reconnect_t1_3d': 130} ...)
```

Config → job script → detached bsub → queue → GPU → engine → D3 assertion → D4 ledger → metrics →
archive → strip + movie, all on the shared NFS mount. Nothing in the chain is untested now.

### 📌 NOTE — the archive correctly refused to double-count

The cluster run wrote **no new archive record**, which is right: `run_id` is a content hash of
(composition, θ, seed, backend, IC), so an identical re-run is idempotent. Same inputs, same
evidence, one record. That is exactly the property that makes a crashed multi-week campaign
resumable without duplicating or losing evidence — confirmed here by accident rather than by
design intent, which is the better kind of confirmation.

### ⏱ SUMMARY — Hour 6 (18:00–19:00)

| | |
|---|---|
| **Done** | control law written and exercised end to end; **full pipeline verified on L4**; archive idempotency confirmed on a real duplicate |
| **Found** | nothing broken this hour — first hour without a defect |
| **Decided** | success criteria frozen in `CampaignConfig` before the search, including "achieved without the extrude node" |
| **Next** | operator-side D1 (delete the 5 private clocks); the instrument gate on 8 eye-labelled archived runs; then wire propose→run→rank into one round driver |
| **Blocked** | nothing |

**Pre-flight status:** D1 (translation) ✅ · D2 ✅ · D3 ✅ · D4 ✅ · D7 ✅ · D8 ✅ · D9 ✅ ·
saturation guard ✅ · control law ✅ · Grounder ✅ · cluster ✅.
Remaining: D1 (operator-side), D5 (lying tags), D10/D11 (ranker wired, bounded cost), the
instrument gate, and the round driver.

---
## Hour 7 — 2026-07-30 19:00–20:00 EDT

Cedric raised: *the vocabulary defaults were set at 15:20 with the rationale "so a
default-parameter composition starts where our evidence is" — but FINDING 8 came after that and
nothing marks them stale, so every generated composition starts from a θ tuned for the wrong
clock.* **Verified, and it was worse than stated.**

### 🔴 FINDING 12 — the per-call / per-frame trap, and a masked correction

From `tyssue_ops3d`: `min_cycle`/`max_cycle` are counted in **division-calls** (`age` is
"per-cell age in division-calls") and `max_div`/`max_div_frac` are **per-call** throttles. The
archived configs passed `every: 2`, gated by the engine *and* by the operator's private `self._k`
— product **4**. So `min_cycle=8` meant 32 frames and `max_div_frac=0.03` meant 0.0075/frame.
Correcting the clock multiplied every per-call meaning by 4 — precisely the aspect 7.5 → 3.2 of
FINDING 8.

**A second defect, found while fixing the first.** `cap_div = max(max_div, max_div_frac·nF)`. The
absolute floor **dominates** at realistic cell counts (nF=1431: `max(120, 42) = 120`), so
rescaling `max_div_frac` alone was **entirely masked** — my first correction had literally no
effect. *A correction that looks applied but is masked is worse than none, because it is
believed.* Both had to move.

Because the factor is exact, the fix is **analytic, not a sweep**: rescale the per-call
quantities and it becomes behaviour-preserving by construction. Verified identical to the archived
per-frame budget at nF = 1431 / 2700 / 4000 / 8000 / 16000. New **V10** gate checks the factor.
**59/59.**

### ✅ Operator-side D1 closed

All five private clocks removed. `_engine_owns_clock()` forces the period to 1 and **raises** on
`every > 1`, so the defect cannot return by configuration. `_k` survives only as a monotonic tick
(`divide_3d` seeds an RNG from it). Verified: modules import, `every=2` refused.

### 🆕 The Metrologist — answering "which agent certifies the foundation, and which fixes code"

The honest answer was that **no agent in the roster owned it** — the defect was found by a person
reading code. Added `agents/metrologist.py`:

- owns the ladder, the instrument gate, and **substrate semantics** (units, clocks, per-call vs
  per-frame) — the gap this incident exposed;
- may act **backwards**: a foundation defect issues a **retraction** moving affected claims to
  Open. Append-only — a new record, never an edit, so the history shows what was believed and why
  it was withdrawn;
- **boundary**: no campaign agent may modify the substrate. Metrologist detects and quarantines;
  an **Engineer** patches a *different artefact* and cannot admit its own fix; the Supervisor
  gates resumption; the human approves. *The instrument must not be adjustable by the experiment
  it is measuring.*

Verified: the gate **refuses admission** while invalidating defects are open, naming them.

### 🔴 FINDING 13 — I retracted one of my own claims

I had "proved" `vcap` was not rate-coupled: *"the same cells divide, only sooner."* True of one
cell, **false of the population** — vcap divisions bypass the throttle, so checking 4× more often
means daughters start growing sooner and the total division count differs.

The evidence that forced it: the re-anchored replay recovered the archived **cell count**
(2927 vs ~2700 ✅) but **not the archived aspect** (1.73 vs ~7.5 ❌). A rate-coupled quantity
remains uncorrected. Recorded as defect `D1c` + retraction `RET000`; `PROVISIONAL_THETA = ("vcap",)`.
This was reasoned, not measured — which is exactly the failure mode the hypothesis-first protocol
exists to prevent, and I committed it in my own analysis.

### 📊 Figure added

`plexus2_discovery.pdf` → 18 pp with **Figure 1: the agent roster** — twelve icons, the first ten
producing or judging evidence, the last two (Metrologist, Engineer) deliberately outside that.

### ⏱ SUMMARY — Hour 7 (19:00–20:00)

| | |
|---|---|
| **Done** | clock re-anchoring made analytic and verified at 5 scales; V10 gate; operator-side D1 closed; Metrologist + retraction implemented; agent figure |
| **Found** | the masked `max_div` correction; **and I retracted my own vcap claim** on evidence |
| **Decided** | substrate is not editable by the campaign; retraction is append-only |
| **Next** | 🔴 **OPEN: aspect 1.73 vs archived 7.5.** Cell count matches, so proliferation is re-anchored; something else is rate-coupled. A `vcap` sweep under the fixed clock is the next experiment |
| **Blocked** | the Metrologist correctly refuses evidence admission until D1/D1b/D3/D4 are marked resolved and the instrument gate passes |

---
## Hour 8 — 2026-07-30 20:00–21:00 EDT

### ✅ RESOLVED — the "aspect discrepancy" was my own metric misnaming

First I verified the assumption underneath the whole re-anchoring: **does the engine actually gate
on `every`?** Yes — `engine.py:694`, `max(1, int(o.params.get("every", 1)))`, *"run only when
tick % every == 0"*. So the double-gating is real and the factor of 4 is correct.

Then the actual resolution. `tube_analysis.py:89` defines
`protr = percentile(r,95)/median(r)` — **exactly the formula I had been computing and calling
`aspect`.** But the report's "aspect ~7.5" for `round_40_mc8` is `tube_len/tube_diam`, a different
quantity. I had been comparing **1.73 against 7.5 as if they were the same number.**

Measured with the archive's *own* metric bank:

| | replay (clock-fixed) | archived (report) |
|---|---|---|
| **aspect** = tube_len/tube_diam | **9.30** | ~7.5 |
| tube_len / tube_diam | 14.69 / 1.58 | — |
| n_tubes | 1 | — |
| cells | **2927** | ~2700 |
| area CV | 0.72 | 0.63 |
| hollow_n_peak | 367 | 176 (tube-aware defn) |

**The clock re-anchoring is successful.** The tube is reproduced, at or above the archived
aspect, with a matching cell count. There was never a physics discrepancy.

### 🔴 FINDING 14 — and I committed the identical error again, one function later

Wiring `tube_analysis` in, I merged its output into our summary **unprefixed**, producing
`protr_final 3.124 > protr_peak 1.732` — impossible, because `tube_analysis` computes on 40
sampled frames with its own body-median while ours computes on all 901. Two different quantities
under one name, *again*, minutes after diagnosing exactly that. Caught only because the ordering
was impossible.

Fixed by namespacing every metric-bank key as `ta_*`, so provenance is visible **in the summary
itself** rather than inferred. Recorded as `M1` (the original) and `M2` (the repeat).

The document's own lesson — *a numerical invariant is not the geometric one you meant* — turns out
to apply to the analysis at least as often as to the simulation. Both of my errors today were
comparisons, not computations.

### 🔁 vcap retraction corrected

The vcap retraction (`D1c`/`RET000`) was reasoned **from the false discrepancy**. Its conclusion
survives — vcap's rate-coupling is untested, neither proven nor disproven — but for a different
reason than recorded, and the note in `composition_space.py` now says so. A retraction that is
itself corrected is exactly why the ledger is append-only.

### ⏱ SUMMARY — Hour 8 (20:00–21:00)

| | |
|---|---|
| **Done** | verified the engine really gates on `every`; reconciled against the archive's own metric bank; **clock re-anchoring confirmed successful** (aspect 9.30 vs ~7.5, cells 2927 vs ~2700); metrics namespaced |
| **Found** | the aspect "discrepancy" was a naming error of mine — **and I repeated it one function later** |
| **Decided** | every borrowed metric is namespaced by provenance (`ta_*`); our r95/median is named `protr`, matching the codebase |
| **Next** | mark D1/D1b/D3/D4 resolved in the Metrologist; instrument gate on eye-labelled runs; round driver |
| **Blocked** | nothing — the open inconsistency from Hour 7 is closed |

---
### 🔴🔴 RETRACTION — Hour 8's conclusion was wrong. The Watcher caught it.

I rendered the final frame and looked at it, which I should have done before writing "successful".

| | archived `round_40_mc8` | clock-fixed replay |
|---|---|---|
| **what the eye sees** | a genuine long thin **tube**, activator at the tip | a **small bud** |
| tube_len / tube_diam | — | 14.69 / 1.58 |
| scored "aspect" | ~7.5 | **9.30** |
| cells | ~2700 | 2927 |

**Two separate confirmed problems, not one.**

**`M3` — the metric lies.** `tube_len_final = 14.69` was scored on a bud, giving an "aspect" of
9.30 that is *higher* than the archived tube's 7.5. This is the documented failure mode verbatim
— *a passed metric is not qualitative fidelity* — and it means the **instrument gate is mandatory
before any campaign scoring**, not a nicety. Retraction `RET001` withdraws the
clock-re-anchoring-successful claim.

**`D1d` — the re-anchoring does not restore the tube.** Cell count *is* restored (2927 vs ~2700),
so the proliferation **rate** is correctly re-anchored — but the **phenotype** is not. Something
governing tip dynamics remains uncorrected.

**The suspect is now evidence-backed rather than guessed: `vcap`.** It force-divides oversized
cells *bypassing the throttle*. Checked 4× more often, tip cells are split the moment they cross
the cap instead of ramping while queued — and the Tyssue report attributes the tube tip
specifically to that backlog behaviour (*"tube-tip cells grew far too big because oversized cells
backlog behind the per-call division throttle and keep ramping while queued"*). Its clock
equivalent is probably **not** a simple scaling. A `vcap` sweep under the fixed clock is the next
experiment.

**Three self-corrections today, all of them comparisons rather than computations.** The pattern is
worth naming: every error I made was in deciding *what counts as the same quantity* — my `aspect`
vs their `aspect`, `ta_protr` vs our `protr`, and now a scored tube-length vs a tube. The
simulation was not once at fault.

### ⏱ SUMMARY — Hour 8 revised

| | |
|---|---|
| **Done** | engine gating verified; metric-bank reconciliation; **and then the visual check that overturned it** |
| **Found** | `M3` the metric reports a tube on a bud; `D1d` the clock fix restores cell count but not phenotype |
| **Decided** | the instrument gate now blocks everything downstream; `vcap` is the evidence-backed suspect |
| **Next** | **instrument gate first** — the metric bank must separate eye-labelled archived runs before any score is trusted. Then the vcap sweep. |
| **Blocked** | campaign scoring, correctly — the Metrologist refuses admission and `M3` is invalidating |

---
## Hour 9 — 2026-07-30 21:00–22:00 EDT

### ✅ THE INSTRUMENT GATE RAN — and it works, in both directions

8 labelled configs submitted 8-way to the L4 partition. Running it exposed **three plumbing
defects in an hour**, none of them in the physics:

1. **Tracked configs were not portable.** `translate.py` baked an absolute `/workspace/...`
   checkpoint path into committed configs; the cluster mounts the same export at
   `/groups/.../Graph`, so **7 of 8 jobs died** with `FileNotFoundError`. Now repo-relative,
   resolved by the runner against its own location, and declared deliberate in V9.
2. **`bjobs` hides finished jobs**, so an empty queue could not distinguish *"all finished"* from
   *"never submitted"* — `wait()` would have reported success on 7 dead jobs. Now `-a`.
3. **Names are not sufficient either**: `bjobs -a` returns historical jobs, so a previous `EXIT`
   is indistinguishable from a new `PEND`. Now we parse **job IDs** from a per-submission log.

### 🔴 FINDING 15 — I committed the gate's own failure mode, one level up

First scoring: **GATE FAIL, no metric admissible.** That verdict was wrong, and the reason
matters. I had taken the eye labels **from the report** — which describes the **archived** runs —
while scoring **fresh** ones. `round_40_mc8` is a long thin tube in the archive and renders a
**bud** here (D1d), so I ranked it 4 while it rendered a 2. Judging by provenance instead of by
looking is precisely what this gate exists to prevent.

Re-labelled from a montage of the actual final frames. What they really are:

| run | renders | was labelled |
|---|---|---|
| `ref_uniform_inflation` (900f) | **exploded spiky mess** | "sphere" |
| `round_44_base` | smooth sphere, tiny nub | "flood" |
| `round_41_hertwig` / `relax60` / **`round_40_mc8`** | **buds** | bud / bud / **tube** |
| `round_42_k05`, `round_42_k05_ex4` | **the most elongated** — thin spikes | "spike" |

The monolayer runs are the tube-like ones; `round_40_mc8` is a bud. (Note also that
`ref_uniform_inflation` is a clean sphere at 130 frames and explodes by 900 — a smoke test at one
horizon says nothing about another.)

### ✅ GATE PASS, with correct labels — and it names the liars

| metric | τ | verdict |
|---|---|---|
| `protr_peak` | **+1.00** | ✅ admissible |
| `ta_n_tubes_final` | **+1.00** | ✅ admissible |
| `protr_final` | +0.67 | ✅ admissible |
| `ta_aspect_len_over_diam` | −0.71 | ❌ **FOOLED** — the metric that scored 9.30 on a bud |
| `ta_tube_len_final` | +0.18 | ❌ FOOLED |
| **`retention`** | **−1.00** | ❌ **perfectly anti-correlated** |
| `n_cells_final` | −0.22 | ❌ no signal |

`M3` is now **confirmed and quarantined** rather than merely suspected: the two `tube_len`-derived
metrics are excluded from campaign scoring by measurement, not by argument.

### 🔴 FINDING 16 — `retention`, and therefore **Q**, measures "did not change"

τ = −1.00 is not noise, it is an inversion. A sphere that never moved scores **1.000**; the most
elongated spike scores **0.275**. The ratio rewards **stasis**.

**Q is defined as the same ratio** — aspect after driver-off relaxation ÷ before — so Q inherits
the defect: *a sphere passes Q trivially.* Q is the campaign's **primary discriminator** in
`plexus2_discovery`, and `CampaignConfig.success` requires `Q ≥ 0.5`. As specified, that criterion
would have admitted every sphere in the search space.

Recorded as `M4` (invalidating). The discriminator must be redefined on an **absolute**
post-relaxation elongation with a floor, not a ratio.

### ⏱ SUMMARY — Hour 9 (21:00–22:00)

| | |
|---|---|
| **Done** | instrument gate submitted, run and scored on L4; 6 earlier defects marked resolved; 3 plumbing defects fixed |
| **Found** | I labelled by provenance instead of by looking (F15); **`retention`/`Q` are inverted** (F16, M4) |
| **Decided** | campaign scoring may use only `protr_peak`, `ta_n_tubes_final`, `protr_final` |
| **Next** | redefine Q absolutely; then `vcap` (D1d). The loop still may not start — `D1d` and `M4` are open |
| **Blocked** | correctly. The Metrologist refuses admission. |

**The gate paid for itself on its first run:** it caught the metric I had already published a
conclusion from, and the discriminator the whole campaign was specified around.

---
## Hour 10 — 2026-07-30 22:00–23:00 EDT

### The loop is wired, and the first question exposed a structural limit before it ran

Cedric proposed testing the loop on the open `vcap`/`D1d` question. Choosing it surfaced a limit
that no amount of round-running would have shown:

> **`vcap` is a parameter, not a composition edit.** `comp_hash` excludes θ by design — that is
> the rule that stops a retune posing as a new hypothesis. So the loop as built **structurally
> could not ask the D1d question**: every vcap value is the same hypothesis.

I had built only **Loop I** (mechanism search). D1d is a **Loop II** (parameter) question. A loop
with only Loop I would have answered it by accident or not at all — and this is exactly what the
multi-week analysis predicted, that Loop II dominates once the composition space exhausts.

`round.py` now has two modes. In `--mode theta` the composition hash is **asserted constant**
across the sweep; the constancy is the point, not a defect, and verdicts land in the map as
parameter-sensitivity rather than as new mechanisms.

### The full agent chain is connected

`Grounder → Proposer(LLM) → Critic → Reflection(LLM) → hypotheses → L4 → Analyst×3(LLM) →
Watcher veto(LLM) → Referee + Judge(LLM) → truncate → Interpreter(LLM) → LeverMap → Supervisor →
Meta-review(LLM)`

Two deliberate refusals: if the Proposer yields nothing usable the round **fails rather than
falling back to random** (a round with no reasoned proposal is a failed round, not a random one);
and if Reflection reports *serious* issues the batch is **not run as proposed**.

### 🔴 FINDING 17 — the Critic caught a live bug in my own reference recipe

Consolidating the type guard into 12 enumerable rules (`critic.py`) immediately rejected
`okuda_route`:

```
R4_SLOT_NOT_ON_IMPL: divide_3d:hertwig has no `axis`
```

The recipe connected morphogen → `divide_3d.axis` while the implementation was `hertwig`, which
splits on the cell's *own* long axis and exposes no such slot. `is_runnable()` checked only for
**unrouted** slots, never for a connection into a slot the implementation does not have — so the
edge compiled and was **silently ignored**, through 59/59 validation and a real cluster run.
Third silent no-op found by making a guard enumerable rather than trusting it.

### 🔴 FINDING 18 — stale configs collide by name

`r01_01_4af688.yaml` from an earlier dry-run shadowed a theta slot. Over weeks a job could pick
up a stale config with a matching name. Removed by hand; the real fix (namespacing configs by
round *and* mode, or purging per round) is **not yet done**.

And a smaller one worth recording because it is the same failure in miniature: my verification
glob matched the stale files rather than the theta ones, and I only caught it by listing the
actual files. Trusting the label over the artefact, again.

### Budgets

Per-agent LLM limits (`agents/llm.py`): minutes + `max_turns` + tools, sized per job, in one
auditable table, with a **25 min per-round ceiling** that hard-stops. `max_turns` is the real
lever — it bounds tool-use loops — and the two agents that only map text→JSON (Watcher, Judge)
get **no tools at all**, so they cannot loop. Verified: a worst-case round blocks the 6th call
rather than overrunning.

### ⏱ SUMMARY — Hour 10

| | |
|---|---|
| **Done** | `round.py` wired to the real agents; θ-sweep mode added; `critic.py` consolidated (12 rules); per-agent budgets; Figure 1 → 15 agents + Table 1; objective reframed as the causal lever-map |
| **Found** | the loop could not ask a parameter question at all (F: Loop II missing); the `hertwig`/`axis` silent edge; stale config name collisions |
| **Running** | **vcap sweep, 5 points, on L4** — the first real round |
| **Next** | read the sweep; namespace configs per round; escalation path; caption-per-wave; progress reel |

---
### 🔴 FINDING 19 — a parameter CAN carry a hypothesis; I had conflated two rules

Cedric: *"we can draw hypotheses with parameters: I expect XXX if I increase vcap to XXX."*
Correct. I had inferred from "composition identity excludes θ" (so a retune cannot be a new
MECHANISM — right, and it stays) that a parameter cannot carry a HYPOTHESIS (wrong).

Consequence, and it is not small: θ points were stamped `predicted: "unknown — sensitivity
sweep"` and **excluded from the surprise rate**. Had that stood, all five vcap points would have
been uncounted and the round that taught the most would have reported **surprise 0.00 — "nothing
learned"**. Fixed: parameter hypotheses carry real predictions and count like any other.

### ✅ FIRST REAL RESULT — the vcap sweep, and it refuted me

| vcap | predicted | protr_peak | final | cells | p_ratio | |
|---|---|---|---|---|---|---|
| 0.0 | `< 1.5` | **2.19** | 2.12 | 2152 | **42.9** | 🔥 refuted |
| 0.75 | `< 2.0` | **4.03** | 1.70 | 5881 | 1.84 | 🔥 refuted |
| **1.5** | `< 2.0` | **1.73** | 1.33 | 2927 | 2.51 | confirmed |
| 2.25 | `≥ 2.0` | 2.24 | 1.61 | 2523 | 5.79 | confirmed |
| 3.0 | `≥ 2.0` | **3.22** | **2.81** | 2448 | 4.70 | confirmed |

**Surprise rate 0.40** — in the productive band (target ~0.30), on the very first round.

1. **`vcap = 1.5`, the archived working point, is the WORST value swept.** It sits at the minimum
   of elongation. That is a large part of D1d: the clock re-anchoring left θ stranded at a local
   minimum, which is why the tube was lost.
2. **The response is NON-MONOTONIC** (2.19 → 4.03 → 1.73 → 2.24 → 3.22). My mechanism story —
   *a lower cap splits tip cells sooner so the tube shortens* — predicted a monotone trend and is
   simply wrong.
3. `vcap = 3.0` gives the best sustained protrusion (peak 3.22, final 2.81 — much the highest
   retention). `vcap = 0.0` shows `p_ratio = 42.9`: massively forced, with no relief valve.

D1d moves from Open toward a partial answer, by measurement rather than by my reasoning — which
is the whole point of building the loop.

## Hour 11 — 2026-07-30 23:00–00:00 EDT

### 🔴 FINDING 20 (CRITICAL for the week launch) — the cluster cannot caption, so the Watcher was blind

Every cluster job recorded:

```
VLM caption UNAVAILABLE: ModuleNotFoundError: No module named 'transformers'
```

The partition's `connectome-gnn` environment has no `transformers`. So **the Watcher was blind on
every cluster run** — which is precisely the population a week-long campaign produces. The
eye/number divergence defence, the thing that has caught more errors in this project than
anything else, was inert exactly where it matters most.

It failed *loudly* (wrote `UNAVAILABLE` rather than a caption), which is the only reason I found
it. A silent skip would have left every run looking Watcher-approved.

**Fix**: `caption_wave.py` — captioning moved off the cluster to the devcontainer, which has the
model and the libraries, with **one model load per wave** instead of one per job. Still
always-caption: it runs as part of closing the round, before the Analysts and the Watcher (both
read `description.txt`, and a blind Watcher cannot veto). Verified on the vcap wave: 5 captions,
one load.

### 🔴 FINDING 18 fixed — stale configs can no longer shadow a round

Configs are now namespaced `r{round:03d}{mode}_{slot}_{hash}` and the round's namespace is
**purged before writing**. A leftover from an aborted run of the same round number would
otherwise be picked up by a cluster job — silently running the wrong experiment.

### ✅ Storage bottleneck retracted

I flagged ~400 GB of trajectories as a bottleneck **without checking free space**. Measured: 12
runs = 454 MB → ~38 MB/run → ~380 GB at 10⁴ runs, against **49 TB free**. It is a non-issue and
needs no retention policy. Recorded because an unchecked assertion in a design document is the
same class of error as an unchecked metric.

### ⏱ SUMMARY — Hour 11

| | |
|---|---|
| **Done** | F18 config namespacing + purge; F20 caption-per-wave (the Watcher can see again); storage bottleneck retracted on measurement |
| **Found** | the cluster cannot caption — the single most important week-launch defect so far |
| **Next** | a COMPOSITION round (the LLM Proposer has still never been called); escalation path; progress reel |
| **Blocked** | nothing |

### Standing state for a fresh session

- **Nothing running** on the cluster.
- The loop works end to end in `--mode theta`; `--mode composition` is wired but the **LLM
  Proposer has never actually been invoked**. That is the next thing to exercise, attended.
- Open, not hidden: escalation path unbuilt (and the lever-map reframing made it the *main*
  path); no progress reel yet; Proposer call still blocks the round instead of overlapping.
- `D1d` partially answered: `vcap=1.5` (the archived working point) is the worst value swept;
  response is non-monotonic; `vcap=3.0` gives the best sustained protrusion.

---
## Hour 12 — 2026-07-30 16:50–17:20 EDT  (fresh session; Cedric left at 17:00, running overnight)

### The attended composition round was worth every minute — `--mode composition` could never have run

RESUME.md §2 said to run one attended composition round because the LLM Proposer had never been
called. It was called. **It worked** — valid JSON, control in slot 0, edits copied from the legal
menu, real falsifiable predictions, and it responded to the Supervisor's adversarial steer. Then
the round **crashed**, and the crash was the point:

```
ValueError: intent 'control' not in ('confirmatory', 'adversarial')
```

`proposer.propose()` REJECTS any proposal whose slot 0 is not `intent: "control"`. `hypothesis.
INTENTS` did not contain `"control"`. Two halves of the same protocol disagreed about whether a
control exists and **both enforced their view with a hard error**, so Loop I could never have
completed a single round. Seven more defects came out of the same hour.

### 🔴 FINDING 21 (the biggest) — the Supervisor had NO persistence

`round`, `spent`, `dry`, `best`, `prox`, `stage_gate` were all re-initialised every process;
`supervisor.jsonl` was append-only *output*, never read back. In a single long process this is
invisible. The campaign is specified to run for **weeks across restarts**, and on every restart:

| field | consequence |
|---|---|
| `round → 0` | every round is "round 1": config namespace collides, the purge deletes the previous round's configs, hypothesis ids collide — `hypotheses.jsonl` **already held 5 duplicate hids** |
| `spent → 0` | `terminal()` compares against `budget_runs`, so **the campaign could never stop on budget** |
| `dry → 0` | dry rounds never accumulate → **escalation was dead at the trigger**, not merely unbuilt at the consumer |
| `prox → {}` | no cluster is ever frozen, budget never reallocates — the Co-Scientist mechanism that is the whole antidote to "thirty rounds, four ideas" |

Now atomically checkpointed to `campaign/state.json` after every round and escalation, with a
restart self-test. A corrupt checkpoint **refuses to start** rather than silently resetting the
counter. Bootstrapped the live campaign from the audit log: it now resumes at round 1 → round 2.

### 🔴 FINDING 22 — the config purge was mode-blind

F18 namespaced configs `r{rid}{mode[0]}_…`, but `_purge_round_configs` globbed `r{rid}*`. So a
composition round deletes the θ round's configs of the same number. It really did delete two
committed `r001t_*` files on the first attended run (recovered with `git restore`).

### 🔴 FINDING 23 — the Proposer wrote "forbidden" into persistent memory

It read the ledger, saw the vcap sweep, and recorded it in `memory.md` as *"a forbidden vcap
parameter sweep"* and *"R0's cardinal sin"*. "Do not propose a parameter change" is a **division
of labour** (this batch is Loop I), not a verdict on validity — and F19 already established that a
parameter can carry a hypothesis. Left alone, every future round would inherit the belief that the
campaign's first real result was a rule violation. Prompt now says so explicitly; `memory.md`
corrected in place. The next Proposer call picked the correction up and wrote *"the sweep is
legitimate Loop-II evidence"*.

### 🔴 FINDING 24 — three defects in the prediction scorer, all biasing surprise DOWNWARD

The surprise rate is the campaign's **only** control signal. It was a three-line first-match regex:

- **P1** an *unparseable* prediction returned `True` → recorded **`confirmed`**. The comment
  claimed such a prediction was "uncounted"; it was not, it was counted as correct.
- **P2** the parser was **metric-blind** — first comparison anywhere in the string, applied to
  `protr_peak` whatever metric the clause named. The house error (my metric vs their metric),
  sitting in the scoring path.
- **P3** `REFUTED if …` was parsed as the assertion; only clause order saved it.

Replaced by `predict.py` (16 self-tests): every clause parsed with its metric, ranges supported,
falsifier text discarded, and **`inconclusive` returned rather than a guess**.

### 🔴 FINDING 25 — the headline "surprise 0.40" had never been written down

Measured consequence of P1. The vcap round's five points were posed *before* F19 with the
placeholder `predicted: "unknown -- sensitivity sweep"`, which parsed to nothing and so resolved
**`confirmed`**. The persisted surprise rate for the campaign's most informative round was
**0.00 — "nothing learned"**; the 0.40 in this log had been computed by hand in the narrative and
never reached the ledger. That is precisely the failure F19 was raised to prevent, still sitting
in the record because the fix was never backfilled.

Added an append-only `amend()` (originals retained; `grep amended` shows every correction) and
backfilled the real predictions, re-scored by `predict.score`. **The ledger now computes 0.40
itself** — outcomes matching the narrative exactly. The 8 orphan hypotheses from aborted runs are
closed as `inconclusive` instead of dragging the reported mix to 92/8.

### 🔴 FINDING 26 — the θ-mode intent rule was the same tautology `proposer.py` was built to kill

```python
intent = "confirmatory" if (... or pred.startswith("protr_peak >=")) else "adversarial"
```
"Predicts an increase" ⇒ confirmatory. So the surprise rate measures **the sign of the effect**,
not anyone's belief — exactly the vacuity removed from Loop I (`intent = adversarial if
lbl.startswith("-")`), reintroduced in Loop II.

### 🔴 FINDING 27 — the minisite Turing × vertex videos can no longer be regenerated

Cedric asked to debug/reproduce the minisite Turing + vertex video before the week launch. The
front-page section is generated by `prototype/Tyssue/archive_rounded.py` (two mp4s:
`tyssue_vh_grow_divide`, `tyssue_vh_rd_coral`). It **hard-errors** under the D1 clock fix, because
it passes `every: 2` and `_engine_owns_clock` now raises on any `every > 1`. But the **engine reads
the same `every` key**, so the blanket raise also forbids the correct engine-owned cadence — the
only way to express a multi-rate period at all. Every archived spec carrying `every: 2` was
permanently unloadable.

Worse, two things about *how* it failed:
- `json.dump(rec, …)` sat **outside** the try, so the failure **overwrote `archive/<nm>/diag.json`** —
  the reference metrics — with `{"error": …}`. `archive/` is documented as immutable research
  record. Merely **importing** the module was enough to do it: there was no `__main__` guard, so
  `do(False); do(True)` ran as an import side effect. Both reference files were destroyed and
  restored from git.

Fixes: the raise became an **opt-out** (`engine_clock: true` asserts "this period is written for
the engine-owned clock"), unmigrated specs still fail loudly; `__main__` guard; failures write
`diag.error.json` and **re-raise** instead of clobbering; and a `--repro` mode that writes
`*.repro.*` and **compares to the archive without touching it**.

The clock migration is `every: 2 → 4`, not `2 → 2`: the old true period was 2 (engine) × 2
(operator) = 4, so writing 2 would silently double the division rate and produce a different movie.

### ⏱ SUMMARY — Hour 12

| | |
|---|---|
| **Done** | 8 defects fixed (F21–F27 + the control crash); `predict.py` new with 16 tests; Supervisor checkpointing + restart test; ledger backfilled to a computed 0.40; minisite generator repaired |
| **Found** | `--mode composition` was structurally unrunnable; the Supervisor never survived a restart; the campaign's headline number was never persisted; the minisite videos were unreproducible AND their reference metrics destroyable by an import |
| **Running** | round 2 (6 knockout jobs on L4, first real composition round) · full 500-frame reproduction of both minisite videos |
| **Next** | read round 2; escalation path; progress reel |
| **Blocked** | nothing |

---
## Hour 13 — 2026-07-30 17:20–18:00 EDT  (overnight, unattended)

Cedric clarified the ask: **run the agentic loop on the Turing + vertex settings to get knowledge
about that class of simulation** — not merely replay the movie. The replay became the baseline.

### Both minisite videos reproduce again — and that validates the clock migration

`archive_rounded.py --repro` (new mode: writes `*.repro.*`, compares, never touches the archive):

| | cells_end | vol_cv | hollow_frac | verdict |
|---|---|---|---|---|
| `vh_K4_cv15_d4` archived | 1778 | 0.226 | 0.004 | |
| reproduced | **1778** | 0.215 | 0.000 | **MATCH** |
| `vh_K4_cv15_d4_rd_coral` reproduced | **1778** | 0.215 | 0.000 | **MATCH** |

`cells_end` matching *exactly* is the load-bearing check: it says the division schedule is
identical, which is what confirms the D1 migration is `every: 2 → 4` (old true period = 2 engine ×
2 operator) and not `2 → 2`.

### 🔴 FINDING 28 — a second lying tag, and it only lies when chemistry is present

The coral video needed one more fix. `morphogen_growth_3d` declares
`MAY_MUTATE_INTEGRATED_STATE = False` and mutates it anyway (`conserve_amount` rescales
`cell.chem` in place). The engine's integration invariant refused to run it. This is one of the
four tags HANDOFF §4 flagged; now confirmed by execution rather than inference.

Note *when* it lies: the plain recipe has no RD, so the activator is identically zero and the
branch is never reached. **It fails only in the composition the campaign is actually about.**
The declaration was wrong, not the behaviour — a `kind=structural` operator rescaling a variable
after a volume change is precisely the exempt category.

### 🔴 FINDING 29 (the big one) — the Turing activator is diluted to extinction

Running the loop on the settings, with predictions recorded first:

| conserve_amount | act_max @f200 | act_max @f400 | act_frac_high | corr(act,radius) |
|---|---|---|---|---|
| **1** (the default) | 6.5e-8 | **3.0e-19** | 0.000 | undefined |
| 0 | 0.421 | **0.431** | 0.340 | **+0.292** |

`conserve_amount` divides `cell.chem` by the volume growth ratio **every tick**. Over 400 frames
that is a geometric decay of ~0.88/frame: `3.3e-2 → 4.6e-11 → 2.0e-22` at frames 60/200/400. The
Gray-Scott field is numerically dead long before the movie ends. Consequences, all measured:

- **The coral pattern on the site's front page is a colour map stretched over a field of magnitude
  1e-22.** The renderer normalises to the field's own 5th/99th percentile, so an extinct field
  still renders as vivid coral.
- It explains why both archived movies have identical diagnostics: with no chemistry, the coral
  run *is* the plain run.
- **Waves A and B were null by construction** — ten runs sweeping `a_sw` (50→0.15) and `rho`
  (1.0→0) against a dead field, every one `cells_end = 1778`, `protr` 1.03–1.08. Both knobs
  multiply an activator that was already 1e-19.

### ✅ The lever map, against a live field

| rho | protr | corr(act,radius) | hollow_frac | mesh |
|---|---|---|---|---|
| 1.0 | 1.033 | +0.292 | 0.000 | intact |
| 0.5 | 1.049 | +0.263 | 0.006 | intact |
| 0.2 | 1.060 | +0.283 | 0.008 | intact |
| **0.05** | **1.173** | **+0.627** | 0.197 | straining |
| 0.0 | 32.727 | +0.530 | 0.839 | **destroyed** |

And `a_sw`, swept 50 → 0.1 at `rho = 1.0`: `protr` 1.036–1.051, `hollow` 0.000 throughout.

**`rho` (the growth cap) is the shaping lever; `a_sw` is not.** The strip at `rho = 0.05` shows
the shell clefting *along the activator bands* — corr **+0.63**, the best coupling seen anywhere
in this project (the previous campaign's best was 0.42 on the SPV backend).

**But there is no clean window.** Coupling only appears once the mesh starts to fail: 20% of faces
folded at `rho = 0.05`, 84% at `rho = 0`, where `protr = 32.7` is a needle, not a tube. Recorded
as a **structural limitation**, which is first-class — and it independently agrees with the prior
campaign's conclusion that the gap is representational (missing force-balance iteration and 3D
reconnection), not parametric.

Figure: `_turing_vertex/turing_vertex_lever.png`. Report: `_turing_vertex/FINDINGS.md`.
**Surprise rate 0.235** over 17 mechanism hypotheses — in the productive band.

### Two errors of my own, recorded

- I predicted the cap explained the flat shape and that `rho = 0` would bulge. Wave B **refuted**
  it — but wave B ran against the dead field, so that refutation was worthless. Wave E, with a
  live field, **confirmed** the reading. *A refutation obtained under a broken instrument is not
  evidence, and I nearly filed it as one.*
- Sweeping `rho` while `a_sw` sat at its default 50 would have measured "growth rate → 0", not
  activator-driven bulging — two levers moving with one named. Caught before it ran; added
  `--fix` so the held knobs are explicit.

And one thing that worked: `predict.score` **refused to score all 22 runs** because `protr` was
not in its known-metric list, returning `inconclusive` 22 times rather than inventing 22
confirmations. That is P1 doing its job on me.

### Escalation path built

`escalation.py` (self-tested): the operator-request backlog + a three-action decision table
(open stage gate → file request → declare exhausted), `agents/llm_agents.request_operator`, and
wiring in `round.py`. The trigger was dead in two ways: `terminal()` has always been able to
return `ESCALATE:` and **nothing ever read it**, and `Supervisor.dry` reset to 0 every process so
it could never accumulate. Verified: an opened stage gate now survives a restart.

### ⏱ SUMMARY — Hour 13

| | |
|---|---|
| **Done** | both minisite videos reproduce (MATCH); F28 second lying tag; F29 activator extinction; the rho/a_sw lever map + figure + FINDINGS.md; escalation path built and tested |
| **Found** | the front-page "coral" is an extinct field rendered through a percentile stretch; `rho` not `a_sw` is the shaping lever; coupling and mesh integrity fail together — no clean window |
| **Running** | round 2 slot `r002c_03_560039` (−cell_geometry_3d) — 35+ min vs 5–20 for its siblings, degrading exactly as the Reflection agent warned |
| **Next** | round 2 readout; progress reel; overlap the Proposer with cluster runs |
| **Blocked** | nothing |

---
## Hour 14 — 2026-07-30 18:00–18:40 EDT

### 🔴 FINDING 30 — the extinction defect reaches the MAIN campaign, not just the minisite

Round 2's Watcher **vetoed the control**:

> _"The description shows a red region vanishing and a ring deforming into a teardrop on a static
> sphere — no tube"_

and the VLM caption of that run begins *"A small red region on a large white sphere gradually
shrinks and disappears"*. That is the F29 signature. Checked `config/okuda/round_40_mc8.yaml`:

```yaml
- op: morphogen_growth_3d
  rate: 0.01
  rho: 0.0
  conserve_amount: true      # <-- the same dilution
```

The flagship okuda recipe carries it too, and with `rho: 0.0` it is **worse in kind**: growth only
occurs where the activator is high, so the dilution applies *only to activated cells*. That is a
self-extinguishing activator — a negative feedback whose fixed point is no pattern. It is a
plausible mechanism for three things the campaign has recorded and not explained:

- R44's *"emergent RD cannot sustain a tube"* impossibility claim (already re-opened in RESUME §5);
- the campaign never finding a **grown** rather than **forced** tube;
- the recurring "body shrinks as a thin filament extends" artefact.

**Testing it rather than asserting it.** `conserve_amount` is an *implementation* choice
(`hill_conserve_amount` vs `hill_no_conserve`), so this is a legal Loop-I composition edit, not a
retune — the hashes differ (`Cd13473bf1c3` → `C8acc4aa1fcc`). Submitted both on L4 as
`d1x_cons_d13473` / `d1x_nocons_8acc4a`, 900 frames.

**If this lands, the campaign has been searching mechanism space with its morphogen switched off** —
and every impossibility claim about RD-driven tubes predates the discovery.

### 🔴 FINDING 31 — a degenerate slot could hold a round for 24 hours

Round 2's `-cell_geometry_3d` knockout ran **45+ minutes against 5–20 for its five siblings**, with
empty stdout: degenerate, not crashed — precisely what the Reflection agent had flagged
(*"may not degrade gracefully ... could go degenerate/uninterpretable"*). A composition search
generates combinations no preset ever ran, so this is the normal case, not a rare one.

The only guard was `timeout_h=24`, and `round.py` **discarded the return value**, so "all six
finished" and "we waited a day and gave up" were indistinguishable. Fixed: once a majority has
landed, a job still running past `max(25 min, 4 × median)` is `bkill`ed and reported;
`wait_for_ids` returns a dict; both call sites check it. Note `not {...}` is always `False`, so
the internal caller would otherwise have silently stopped stopping.

Applied the policy by hand to the live round (the process was running the old code): killed the
straggler, and round 2 proceeded to caption and score the five that landed.

### ✅ Caption-per-wave confirmed in production

F20's fix worked on its first real composition round: **5 runs captioned in 135 s on one model
load**, on the devcontainer, before the Analysts and the Watcher — and the Watcher then used a
caption to veto the control. The eye/number defence is live again on cluster runs.

### ✅ Escalation path produced its first real entry

**OR001** in `campaign/operator_backlog.md`: quasi-static force-balance relaxation (a *residual*,
not a step count) plus 3D IH/HI reconnection, with a proposed contract and an acceptance test that
today's best setting fails by ~4× on both integrity metrics. Provenance is recorded in the entry:
filed by me from 32 runs of measured evidence, **not** written by the request agent — so it cannot
later be mistaken for an LLM-generated wish.

### ⏱ SUMMARY — Hour 14

| | |
|---|---|
| **Done** | straggler kill + dict contract; OR001 filed; FINDINGS.md with the 32-run structural limitation; caption-per-wave verified in production |
| **Found** | the extinction defect is in the MAIN campaign's flagship recipe, and `rho=0` makes it self-reinforcing; a degenerate slot could stall a round for a day |
| **Running** | round 2 analysts (20 LLM calls) · `d1x` conserve/no-conserve decisive test on L4 |
| **Next** | read d1x; read round 2; progress reel |
| **Blocked** | nothing |

### ⛔ FINDING 30 RETRACTED — measured within the hour, and it was wrong

The `d1x` test landed. **There is no activator extinction in the okuda campaign.**

| | `conserve_amount: true` | `false` |
|---|---|---|
| `act_max_final` | **0.995** | 1.000 |
| `protr_peak` | 1.876 | 1.945 (+3.7%) |
| `ta_tip_act_final` | 0.676 | 0.661 |
| `n_cells_final` | 569 | 571 |

And on the recipe that actually prompted the hypothesis (round 2's control, comp `5e3159`):
`act_max_final = 0.972`. The morphogen is alive at full amplitude. Turning the dilution off
changes almost nothing.

**Why the Turing x vertex recipe dies and this one does not.** The dilution is driven by the
volume growth ratio, so it only bites while cells are growing:

- Tyssue coral recipe: `rho = 1.0` — *every* cell grows *every* tick, so the dilution is
  continuous and global → amplitude collapses to 1e-22.
- okuda recipe: `rho = 0.0` — only activated cells grow, and the `else` branch clamps the scale
  at `cap`; once a cell saturates, growth stops and **so does its dilution**. The activator
  survives.

I had the sign of the feedback backwards: I argued `rho = 0` would make the dilution
*self-reinforcing*; it is *self-limiting*.

**What the caption actually meant.** *"A small red region … gradually shrinks and disappears"* was
right about the DOMAIN and I read it as amplitude. `red_frac_final = 0.005` — the activated region
has collapsed to 0.5% of cells while its peak stays at 0.97. And the tip tells the real story:

| run | `ta_tip_act_final` | `protr_peak` |
|---|---|---|
| control | **0.27** | 4.03 |
| −morphogen_growth_3d | **0.80** | 1.03 |

With growth on, the activator is *not at the tip*. That is a local effect — growth carries the
activated region away from the protrusion it is supposed to drive — not a global decay, and
`conserve_amount` is not its cause.

Recorded as a retraction rather than a quiet edit: the hypothesis was stated in this log an hour
ago, it was testable, it was tested, and it failed. The test cost two cluster runs because
`conserve_amount` is an *implementation* (`hill_conserve_amount` / `hill_no_conserve`), so the
comparison was a legal one-edit composition change with distinct hashes — not a retune.

### ✅ Round 2 completed — the first composition round, and the Watcher earned its place

```
[KEEP] r002c_01_cba5fe  score 1.71  protr_peak 1.39  phen=sphere      watcher=supports     [confirmed]
[KEEP] r002c_02_9fdce9  score 1.53  protr_peak 1.03  phen=sphere      watcher=supports     [refuted] SURPRISE
[drop] r002c_00_5e3159  score -inf  protr_peak 4.03  phen=tube        watcher=CONTRADICTS  [refuted]
[drop] r002c_04_a413e0  score -inf  protr_peak 3.20  phen=degenerate  watcher=CONTRADICTS  [confirmed]
       r002c_03_560039  straggler, killed
       r002c_05_2399b2  NOT EVIDENCE — P2_BUFFER_SATURATED: n_cells=15002
```

**The Watcher vetoed the two highest-scoring runs.** `protr_peak` 4.03 and 3.20 — the numerical
winners, one of them the control that three Analysts called a *tube* — both scored −∞ and dropped,
because the captions do not show a tube. That is the eye/number divergence defence firing on a
composition round for the first time, and it is the difference between this round reporting "the
control makes a tube (4.03)" and reporting the truth.

The Critic separately refused `+vesicle_growth:uniform_ramp` as **not evidence** — the reservoir
saturated at 15002 cells. Two of six slots produced no evidence, and both said so loudly rather
than returning a plausible number.

**The dissociation result:** both knockouts collapse the protrusion (4.03 → 1.39 for −extrude,
→ 1.03 for −morphogen_growth_3d). The Proposer predicted −morphogen would leave it unchanged if
the protrusion were *forced*; it did not. Recorded as the round's one **surprise**. But the
control is itself Watcher-vetoed, so the 4.03 baseline those ratios are measured against is
suspect — the honest reading is *"both operators are necessary for whatever 4.03 is"*, and what
4.03 is remains open.

---
## Hour 15 — 2026-07-30 18:10–18:20 EDT — deliverables and handover state

### Progress reel built (`reel.py`)

Round montages, per-wave study reels, and a progress reel, with labels burned in. Two things
worth recording about *how* it had to be built:

- The bundled `imageio-ffmpeg` binary has **no `drawtext` filter** (built without libfreetype) and
  there is no system ffmpeg here. Labels are rendered with matplotlib and composited via
  `overlay`, which the binary does have.
- ⚠ **Tiles are not on a common spatial scale.** Each per-run movie was rendered with its own
  camera box (`Lbox = 1.06·Rmax`), so a run with a long spike gets a wide box and its body is
  drawn small. In the round-2 montage the control looks like a tiny body beside two large spheres
  — **that difference is the camera, not the biology**. Every tile is now labelled `(own scale)`
  and the module says so at the top. Recorded rather than silently shipped: an unlabelled montage
  is itself an instrument that lies.

The montage did earn its keep immediately: it shows the control's `protr_peak = 4.03` is a **small
body with a short stub**, not a tube — which is exactly what the Watcher said and the three
Analysts did not.

### `plexus2_discovery.pdf` updated — 21 → 23 pages

New §"First live results", covering: the control-intent deadlock, the three scoring defects and
the 0.00-vs-0.40 ledger, the supervisor's missing persistence, the escalation path and OR001, the
Turing×vertex structural limitation, and both of my retracted mechanism claims. Compiles clean.

### Standing state for the next session

- **Nothing running** locally or on the cluster.
- Supervisor resumes at **round 2 → next round 3**, `spent 9`, `dry 1`, **1 cluster, 0 active**.
  So `terminal()` returns `ESCALATE: all clusters frozen` and `round.py` now *reads* that verdict
  — round 3 should open stage gate 3 by itself. That is the escalation path's first automatic
  firing and it should be watched.
- All gates pass: validate_space 59/59, critic, predict, hypothesis, control, escalation,
  preflight, admission.
- Findings run F1–F31, with **F30 retracted by measurement**.

### ⏱ SUMMARY — Hour 15

| | |
|---|---|
| **Done** | `reel.py` (3 montages incl. a 32-tile study reel); PDF 21→23 pp; RESUME.md rewritten for the next session |
| **Found** | no `drawtext` in the bundled ffmpeg; montage tiles are not spatially comparable and would have invited a false reading |
| **Next** | round 3 (watch escalation fire); overlap the Proposer with cluster runs; analyst calls bypass the LLM budget ledger; common-`Lbox` re-render for a truthful montage |
| **Blocked** | nothing |

## 2026-07-31 15:50 — three defects behind the inert battery, fixed and verified

Cedric looked at a Phase 1 strip and said "everything shrinks, oh no we broke it all?". It was not
a rendering problem and it was not new: all 11 completed runs ended at exactly 500 cells (= the
seeded count), protrusion 1.02 (a sphere) and activator 0.499 (= the seed amplitude). Nothing grew,
divided or reacted in any of them. Bisection found three independent causes.

D1  shape_energy_3d/monolayer had no rest state. Stripping the operator list one at a time: the
    ball holds 5.000 exactly under every other operator and falls 5.00 -> 1.80 in 20 frames the
    moment the mechanics is added. gamma=0 gives 2.95, gamma=0 and kappa_s=0 gives 5.00 -- so both
    tension terms pull in and only the volume spring pushes out, and V_eq had been calibrated
    against the volume term alone. Fixed by solving for the target-volume offset that puts the
    seeded shell at force balance. Solved rather than chosen: the loop sweeps k_v/kappa_s/gamma/h0/
    radius/n_cells and any hardcoded tension collapses at the next setting. Additive not
    multiplicative (the offset is size-independent, so it must survive growth and division). One
    solve was not enough -- it left gamma=0 at x0.80 -- so it iterates against the relaxed shape.
    7 of 9 sweep settings now hold to within 1.5%; kappa_s=0.6 and gamma=0 still drift and are
    RECORDED, not silenced.

    Invisible for the whole campaign because the healthy-looking runs loaded a pre-relaxed
    checkpoint (round_40_mc8 starts at radius 6.14) instead of seeding. The two quarantined runs
    that did seed with this implementation, r01_00_bd318e and r01_01_d1076f, have no diag.json.

D5a chemistry integrated with the mechanics dt. cell_react and cell_diffuse EMIT=velocity into
    `chem`, so 300 frames bought 6 units of Gray-Scott time against the ~500 the validated minisite
    spec needs. Every "no pattern formed" reading was an artefact of the clock. Both are now scaled
    by 1/dt, leaving their ratio -- which selects the Turing wavelength -- untouched.

D5b the growth ceiling sat below the division trigger: vth_frac*v_ref = 1.5 against factor*Vbirth
    = 2.0. Volume-triggered division was arithmetically impossible; every division ever seen came
    from the max_cycle timeout. The ceiling is now derived from the trigger.

Also: grow_after=100 turns out to be a workaround for D1 -- a window to let the shell settle before
growth. It survives as a pattern-formation window, which is a legitimate reason.

VERIFIED. log/okuda cleared, battery relaunched at 500 frames, 12 specs, 2 GPUs x 3.
  p1_ko_morphogen_growth_3d (no growth)  static baseline: folded 0, shape_idx pinned 3.76,
                                         area_cv flat 0.15 -- the ball holds its size.
  p1_ko_divide_3d (growth, no division)  the ball GROWS, a Turing pattern forms (act 0.465 vs the
                                         minisite's 0.43), the shell buckles: folded 0 -> 145 ->
                                         heals to 17, shape_idx crosses 3.81 (Bi rigidity
                                         transition), area_cv 0.16 -> 0.04. broken_n 0 throughout.

METHOD NOTE (Cedric). All three defects are "does this operator do what its name says", and all
three were found by a human looking at a picture. This is the fourth time. The parameter-stability
map I built by hand while fixing D1 is exactly what the loop should be producing on its own -- it
goes in as a Phase 1 operator sweep result, not as constants I picked.

## 2026-07-31 late — Phases 2, 3, 4 and the weekend battery

PHASE 2 (4 of 6).
  R0 frozen at the seed radius while cell targets grew 16x -- the binding constraint on
  reachability, and the root cause of the retracted buckling transition (F004/F005/F006). Fixed:
  R0 follows the enclosing sphere of the current TARGET volume, deliberately not the measured mean
  radius, which would make the spring chase the shell's own excursions and penalise a bud.
  Verified: rays cross exactly once at every frame where they previously crossed 13-17 times.

  shape_to_chem: the missing arrow. ONE contract, FOUR implementations (curvature, tension,
  apical_area, pressure), because WHICH feature the chemistry listens to is the hypothesis, not a
  parameter -- comp_hash includes implementation and excludes theta. `force` and `size` refused,
  with reasons in the docstring. Feature standardised before beta touches it (the F009 lesson
  applied pre-emptively). Certification caught two of my errors: the curvature normalisation
  (2R/L^2, only looked like 1/R because the test held cell count fixed) and my own bump test
  (scaling a cap outward makes it FLATTER). Two blow-ups fixed end-to-end; premise 12 (a
  concentration is non-negative) came out of the second.

PHASE 3 (1 of 3). Ablation compulsory: claim_kind on every hypothesis, critic.check_batch rejects
  A1_NO_ABLATION before any compute, allowed_verb stops the write-up outrunning the experiment.

PHASE 4 (built + calibrated).
  morphology classifier: sphere/undulation/tube/branched/invalid/unclear, certified on built
  shapes. `invalid` beats every morphology. Records the run's PATH, not just its end.
  pattern_scale: n_spots exact at 3/5/12, spot_spacing within 13% of R sqrt(4pi/k).

  THE CALIBRATION WENT SOMEWHERE ELSE (F011/F012). Counting spots over time showed the pattern
  never settles -- 104 -> 62 -> 19 -> 7 -> ONE domain at 53% coverage, stable from step 400 to
  3000. The campaign's chemistry has always been in the LABYRINTH regime. "Five spots" was a
  moment during coarsening. Swept Pearson's diagram for STABLE spots (count at 1500 vs 3000):
  F 0.046 / kk 0.062 gives three spots on all three seeds. Frozen.

  Also retracted F010: the autocorrelation wavelength failed its own certification (the ratio
  true-lambda / first-minimum came out 1.37, 1.79, 1.49 -- not a constant). Kept as a labelled
  diagnostic, not shipped as comparable.

WEEKEND BATTERY: 27 runs, 4 features x 2 signs x 3 seeds + 3 nulls, both GPUs. The null is the
subtractive direction, so the batch satisfies the ablation rule it was built alongside. First
batch in the campaign whose every instrument was certified against a known answer BEFORE it ran.
The gate already earned itself: a smoke run was refused for P4 (chemistry extinguished by growth
dilution) before any result was read.

OPEN FOR MONDAY: P5b fails even at relax_iters 60 -- the residual force still climbs as the tissue
grows. Recorded as a verdict rather than chased. Phase 2 still needs the widened settings and the
published-constants fixture; Phase 3 still needs Evolution, the Referee and the parallel calls.

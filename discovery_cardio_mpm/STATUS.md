# Discovery — cardiomyocytes: status

*Written 2026-08-02, at the point where Phase 0 is closed, Phase 1 is two items in, and the next
work is measurement rather than construction.*

The full narrative is `cardio_note.pdf` (18 pages) and it is what Cedric reads. This file is the
operational summary: what was done, what it cost, what is measured, and what is not.

---

## 1. What this loop is for

The third agentic loop, and the first pointed at a **real measurement** rather than a paper.
`discovery_okuda/` searches for a mechanism nobody wrote down; `atlas_jax/` and `atlas_cc3d/`
decompose published code into the operator algebra. **This one fits a differentiable MPM model of
a beating cardiomyocyte sheet to microscope tracking data**, and asks which mechanisms the data can
actually support.

**Two tracks, judged differently** (this is the design decision of the whole folder):

- **Track A has no success rule.** Its product is a causal map that improves by being filled in,
  including with negatives, reported with a *surprise rate* beside it.
- **Track B has a sharp one:** the difference between the simulated and the recorded loop
  trajectory, **decomposed** into magnitude / opening / direction / orientation / shape — never one
  number.

**The loop is four steps, not thirteen roles** (Cedric's instruction: *make it brutally concrete*):

    propose  ->  gate  ->  measure on data it never saw  ->  interpret
       ^                                                          |
       +--------- the part of the residual nothing explains -------+

`plexus2.tex` §"Agentic mechanistic discovery" calls these Understanding / Fitting / Discovery. The
closing edge is its Loop III. The shape was arrived at independently and it matches.

**The standing rule: no conclusion of the previous 60-batch campaign is inherited.** Its claims sit
in `HYPOTHESES.md` as open questions; `BELIEFS.md` has zero entries. Defaults count as beliefs.

---

## 2. Where it stands

| phase | state | what it produced |
|---|---|---|
| 0 — make it run, make it repeat | **done** | gate 18/18, canaries 6/6; seeded, provenanced, ledger stripped from the source |
| 1 — freeze the recording, seal the test | **6 of 7** | + the resolution ladder, the frozen+sealed split, and `PREMISES.md` (2 of 8 premises fail) |
| 1b — the corpus we already own | **in progress** | 8 stale recipes migrated and loading; first regeneration running |
| 2 — certify the ruler, find the floor | not started | **STOP point** |
| 3 — certify the gradient | not started | the language-vs-switches decision |
| 4 — what a fit may claim, build the gate | not started | **STOP point** |
| 5–8 — first round, campaign, seal-break, method claim | not started | |

**Registers:** 38 inherited hypotheses, all open · **0 beliefs** · 0 retractions.

---

## 3. What has actually been measured

Every number below was measured in this folder, not inherited.

| | |
|---|---|
| the trainer at HEAD | **crashed on every invocation** — a function one branch deleted from under another |
| the recording | **rebuilds bit-exactly** from the microscope derivatives, every array (`ingest.py`) |
| two fits at one seed | **bit-identical over 198,407 parameters** on CPU; two seeds differ |
| GPU determinism | **partial** — blocked by `grid_sampler_2d_backward_cuda` in `plexus.models.base.Field.sample`. Same-seed spread **3.0e-6** (rel 7.5e-8) at 2 iterations |
| LoopScore's zero | **+0.075 ± 0.117**, not 0. Its own docstring says a stub scores ≈0 |
| LoopScore vs coordination | scrambling every particle's timing scores **exactly 1.0000 on all 10 corpus runs** — identical to a perfect match |
| the second score | `interior_r2` null is **−0.875**; the best of 324 archived fits is **−0.912**, i.e. below it |
| the trivial baseline | **copying the previous beat outscores every fit the previous campaign produced** |
| the boundary | with the muscle off, interior displacement is **bit-exact zero** — the pinned edge does not drive the fit |
| the specimen | one healthy sheet + one diseased sheet, under **five filenames**. No independent replicate exists |
| held-out beats | leave-one-out **R² 0.978–0.986** — a held-out beat is not a real test |
| information content | the beat is **90% one spatial pattern, 99% three**; ~109 independent patches |
| **the ceiling** | beat vs beat: LoopScore median **0.705** (0.469–0.850). **No model may score above it.** Even two real beats agree on circulation direction only **93.9%** of the time |
| **the tracker** | two trackings of one movie: **0.996** in time, **0.27–0.29 / −0.06** in space at every beat peak, smoothing and axis conventions tested. As a model of each other: circulation right **50%** of the time, orientation error 0.77 rad (random = 0.79), LoopScore **−0.53** |
| the corpus | 71 MPM runs, 45 GB, intact in `graphs_data/material` — `log/material/` holds only the recipe |
| corpus reproducibility | **7 of 10 active-traction specs would not run**; all 8 migrate and load |
| **temporal integration** | converged: substeps 5/10/20 agree to **0.8%** — *once frame-dt is held fixed* |
| **`--substeps` is not a numerics knob** | it multiplies `dt_sub` to give the frame duration; varying it alone moves enclosed area **2.2×**. The inherited `10` is a claim about the tissue's timescale |
| **spatial discretisation** | **NOT converged.** Enclosure follows particles-per-cell monotonically over **1.56×** across two independent knobs, no plateau. Direction and orientation are untouched |
| **premises** | 6 of 8 hold. **Four operators in the spec never run** (`activation_pulse`, `aggregate`, `apply_material_map`, `pacemaker`). **The model is quiescent 8% of the beat where the tissue is 77%** |
| the split | frozen: fit beat [152,204], 3 held-out beats, 17,499 scored nodes, mask frozen from the recording alone; diseased sheet sealed by content across all 3 files, seal watched refusing |

---

## 4. Defects found that a forward run could not have found

Recorded because they are the argument for the apparatus.

1. **A default was a retired belief.** `--stiff_src` defaulted to an image-sourced field whose
   implementation had been *deleted*, so the default invocation trained a 2-parameter stub against
   a synthetic blob — and the dashboard's `corr(microscope)` panel correlated against that same
   blob. Removed because the code is gone, not because the claim was accepted.
2. **A silent bug in the pulse.** The learnable duration initialised against a hard-coded bound
   while the forward used the CLI one: `--dur0 10 --dur_hi 11` actually started at 8.09. Every
   duration claim in the previous campaign was made through that.
3. **My own deterministic sampler had its axes swapped** and disagreed with the routine it replaced
   by *more than the signal*. Caught only because the replacement was checked before use.
4. **My own magnitude axis was not rotation-invariant** — an L∞ peak, so turning a loop on the spot
   changed its "magnitude". Caught by the descriptor self-test, not by inspection.
5. **Partial aliasing in the library.** Three of six operator renames kept a back-compat alias and
   three did not, with no rule. That is how the rename defect keeps recurring.
6. **`log/material` looks empty and is not.** The data is in `graphs_data/`; the log dir holds only
   the recipe and a completion marker. Nearly cost a regeneration of 45 GB.

---

## 5. What exists here, and what it is

| file | what it is |
|---|---|
| `cardio_note.tex/pdf` | **the document.** Plain English, by phase. Report into it at every boundary, then stop |
| `certify_apparatus.py` | the Phase 0 gate, with `--canary` (breaks it 6 ways, must catch 6) and `--fit` |
| `ingest.py` | rebuild the recording from the microscope derivatives; `--verify` proves bit-exactness |
| `data.py` | **one** path, no fallback; content-hash identity; `specimen_id()` for sealing by content |
| `determinism.py` | seed + pinned arithmetic; called AFTER imports because the trainer set flags at module scope |
| `provenance.py` | a run copies the bytes of every module it imported into its own folder |
| `train.py` | the inherited trainer, repaired, seeded, provenanced, ledger stripped |
| `descriptors.py` | **Track B's measurement.** magnitude/opening/direction/orientation; `shape` deliberately absent |
| `corpus.py` | the 71 owned runs: inventory, descriptor read, and the rebuilt visual instrument |
| `reproduce.py` | migrate the stale recipes and regenerate into `_repro/` — the archive is never written |
| `METRICS.md` | the archaeology: what each inherited instrument was built to answer, and its state |
| `HYPOTHESES.md` | 38 inherited claims, all open |
| `BELIEFS.md` | what we have earned. Empty |
| `harmonic_inherited.py` | the inherited LoopScore, **uncertified**, kept only for comparison |

~3.1k lines of Python, 5 commits, no GPU-hours spent on science yet.

---

## 6. To resume — the five things that close Phase 1

In this order. The first two are cheap and pure measurement; the third can redirect the programme;
the last is irreversible and goes last.

1. ~~**The beat inventory.**~~ **DONE.** Onsets `[2, 51, 101, 152, 204]`, gaps `49/50/51/52`, four
   complete beats, and a truncated tail that is not a beat. The reported "period" of 50 is what
   rounding 50.5 gives, not the mean.
2. ~~**The self-agreement ceiling.**~~ **DONE** (`data_report.py`). LoopScore beat-vs-beat median
   **0.705**. No model may score above it. And even two real beats agree on circulation direction
   only 93.9% of the time — the previous campaign read 0.85 as a deficit against an implicit 1.0.
3. ~~**The tracker-reproducibility number.**~~ **DONE, and it settles the question.** Time course
   0.996; per-node spatial maps 0.27–0.29 / −0.06 at every beat peak; as a model of each other,
   circulation right 50% of the time and LoopScore −0.53. **Learned spatial maps across the sheet
   are not a product of this project.** Aim the loop at whole-sheet mechanisms.
4. ~~**The resolution check.**~~ **DONE** (`resolution.py`). Time converged; space not. And
   `--substeps` turned out to be a physics knob, not a numerics one.
5. ~~**Freeze the split, seal the diseased sheet.**~~ **DONE** (`split.py --freeze`, `--check`
   passes 7/7, seal watched refusing).
6. **The premises** (`PREMISES.md` + `premises.py`) — **in flight, 6 of 8 hold.** The two failures
   are real and neither is waived. Remaining: wire the premise gate into the loop's step 2, and add
   the per-run rail check (no fitted value on its bound) once there are fits to check.

Plus **Phase 1b green**: the eight migrated recipes regenerate and match their archives.

**In flight right now:** `reproduce.py --run material_active_phase_radial` (started 09:31, ~12 s per
frame on `cuda:0`, 250 frames). When it lands:
`python reproduce.py --compare material_active_phase_radial` — a match restores the corpus *and*
validates the operator merge; a mismatch is a silent change in the forward model, which matters
more.

---

## 7. Two decisions waiting on Cedric

1. **Phase 3 — gradients through the Plexus language, or through the trainer's switches?**
   `plexus2.tex` claims *"differentiability becomes a property of the language rather than of a
   particular simulator."* Measured, that is not true today: ask for a gradient through a spec and
   it returns nothing (40 of 45 operator files call `float(params[...])` in `__init__`;
   `Operator.tunable()` exists and no core operator uses it). So this closes a gap against the
   design rather than expressing a preference. **Recommended: through the language, scoped to the
   dozen operators this problem uses.**
2. **Do the two STOPs stand?** Phase 2 can conclude the active model does not beat the trivial
   baselines; Phase 4 that no model size clears the noise. Both would end the project in its
   current form, and both are only worth building if it is agreed *in advance* that reaching one is
   a result. On the evidence already on disk, the Phase 2 stop is a live outcome.

Agreed and not in doubt: a second Utrecht dataset will be requested next, but the approach has to
be demonstrated on what we have first.

---

## 8. The correction that may change every budget

`plexus2.tex` is emphatic that a loss should be **local in space and time** — a subset of entities
over a short rollout — because the backward graph grows as entities × steps, while the evidence
constraining any one operator is usually confined to a small transient region.

The inherited fit is the opposite: it backpropagates through a *whole* beat, ten substeps a frame,
on *all* 16,384 particles — roughly **530 solver steps per gradient** — which is very likely why one
fit costs **5.5 hours**. Nobody chose that; it was inherited. Whether a local loss recovers the same
mechanism far more cheaply is a Phase 3 experiment, and **if it does, every compute estimate in the
note falls.**

---

## 9. Hard-won rules — do not rediscover these

- **The instrument lies before the physics does.** Three of the six defects above were in
  measurement code, two of them in code I had just written to fix a measurement.
- **A gate that cannot fail is not a gate.** Every certification script ships a `--canary` mode that
  injects the faults it claims to catch.
- **A default is a belief in disguise**, and goes in the register with the rest.
- **Never let a loader guess.** One explicit path; a missing file is an error, never a fallback.
  The three-location search was added to fix a lost batch and turned "file missing" into "fitted the
  wrong recording".
- **Scored on data it was fitted to is not evidence.** With gradients, fitting better is nearly free.
- **The archive is never the thing being overwritten.** Regeneration goes to a scratch root.
- **Environment:** `/workspace/.conda_envs/neural-graph-linux/bin/python`,
  `PYTHONPATH=/workspace/Plexus/src`. `git push` may need `--no-verify` (git-lfs absent).

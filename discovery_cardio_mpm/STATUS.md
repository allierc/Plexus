# Discovery — cardiomyocytes: status

*Written 2026-08-02, revised 2026-08-03 after an external audit. **Phases 0, 1 and 1b are closed.**
Three phases stand between here and a running loop — 2, 3 and 4 — and two of them can stop the
project. None of the loop itself is built yet, deliberately: it is Phase 4's work.*

The full narrative is `cardio_note.pdf` (18 pages) and it is what Cedric reads. This file is the
operational summary: what was done, what it cost, what is measured, and what is not.

---

## 1. What this loop is for

The third agentic loop, and the first pointed at a **real measurement** rather than a paper.
`discovery_okuda/` searches for a mechanism nobody wrote down; `atlas_jax/` and `atlas_cc3d/`
decompose published code into the operator algebra. **This one fits a differentiable MPM model of
a beating cardiomyocyte sheet to microscope tracking data**, and asks which mechanisms the data can
actually support.

**Two tracks, and the ORDER is the argument:**

- **Track A — understand the model on its own terms, before any data is involved.** Sweep it,
  ablate it, change what drives what, and record where the system goes. **No recording is
  consulted.** Product: a causal map — which mechanism moves which feature of the beat, alone and
  in combination. No success rule; a dose-confirmed nothing is a result. Instrument:
  `descriptors.describe()`, which reads a trajectory on its own axes.
- **Track B — fit the real recording, spending what A learned.** Differentiable, so far stronger
  and far more dangerous. Instrument: `descriptors.loop_residual()`, built from two `describe()`
  calls so the two tracks cannot drift apart in what they mean by "opening".

A fit will always find something; Track A is what makes it mean something. **A builds the
knowledge, B spends it, and what B cannot explain goes back to A as the next thing to sweep.**

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
| 1 — freeze the recording, seal the test | **done (7 of 7)** | the ceiling, the tracker floor, the resolution ladder, the frozen+sealed split, `PREMISES.md` (2 of 8 fail, recorded not waived) |
| 1b — the corpus we already own | **done** | 8 recipes migrated; **the operator merge is NOT behaviour-preserving** |
| 2 — certify the ruler, find the floor | **done for real 2026-08-09** | 4 instruments CERTIFIED (nothing was promoted until then — see §6 warning). Null bank: bar +0.851 on the fit beat, +0.62 held-out. **STOP point** |
| 3 — certify the gradient | not started | the language-vs-switches decision. **Now the leading route — see §10** |
| 4 — what a fit may claim, build the gate | not started | **STOP point** |
| 5–8 — first round, campaign, seal-break, method claim | not started | |

**Registers:** 38 inherited hypotheses, all open · **0 beliefs** · 0 retractions.

> **→ CURRENT STATE IS §10.** Everything after 2026-08-08 lives there: the acceptance statistic was
> rebuilt, the amplitude gauge withdrawn, and a four-probe investigation settled that P1 is **not**
> a gain verdict. §10 also records three errors in the P0 work itself, one of which mis-calibrates
> every "distinguishable steps" number in this file.

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
| **the ceiling** | beat vs beat: LoopScore median **0.705** (0.469–0.850). No model may score above it **on LoopScore** — which is not an admitted instrument, so Phase 2 must recompute it. Even two real beats agree on circulation direction only **93.9%** of the time |
| **the tracker** (one condition-signature: same specimen, same movie, same algorithm pair — the four beats are repetition, not corroboration) | two trackings of one movie: **0.996** in time, **0.27–0.29 / −0.06** in space at every beat peak, smoothing and axis conventions tested. As a model of each other: circulation right **50%** of the time, orientation error 0.77 rad (random = 0.79), LoopScore **−0.53** |
| the corpus | 71 MPM runs, 45 GB, intact in `graphs_data/material` — `log/material/` holds only the recipe |
| corpus reproducibility | **7 of 10 active-traction specs would not run**; all 8 migrate and load |
| **temporal integration** | converged, checked on **two** backgrounds: substeps 5/10/20 agree to 0.8% at n_grid=128 and 3.7% at n_grid=192 — once frame-dt is held fixed. Weakens as the grid refines (dx–dt coupling) but stays >10× below the 56% spatial swing |
| **`--substeps` is not a numerics knob** | it multiplies `dt_sub` to give the frame duration; varying it alone moves enclosed area **2.2×**. The inherited `10` is a claim about the tissue's timescale |
| **spatial discretisation** | **NOT converged.** Enclosure follows particles-per-cell monotonically over **1.56×** across two independent knobs, no plateau. Direction and orientation are untouched |
| **two open defects** | **the warm-up does not settle (33% shift)** — found only because the audit noticed premise 5 had never been implemented; it underwrites every gradient. And **the model rests 8% of the beat where the tissue rests 77%**, which the objective cannot see |
| **the fitting recipe** | four operators were declared and never stepped; there is now a fitting recipe that says what the fit runs, and the removal is **proved** harmless — fits under both are bit-identical over 198,407 parameters |
| **premises** (after the audit) | **11 of 13 hold; verdict AMBIGUOUS.** Static set is VALID 8/8 (premise 3 closed properly, not waived; coverage check added). The two forward-probe failures are both graded *usual*: **the warm-up does not settle — one extra beat moves the fitted window 33%** (Phase 3 owes an answer), and the model rests 8% of the beat where the tissue rests 77% |
| **the null bank** (Phase 2) | predict nothing **+0.070** · slide the sheet **−0.020** · **copy the previous beat +0.851 (fit), +0.62 (held-out)** · interpolate from the pinned edge **−0.118** · muscle off **+0.070, exactly the do-nothing score** · fields untrained **−0.880**. The previous campaign's best was 0.545 — below even the held-out bar |
| **particle layout is DEVICE-dependent** | same seed, CPU vs GPU → positions differ by the sheet width; changing the seed changes nothing. A CPU fit and a GPU fit are not the same model. Phase 0 compared like with like and could not see it |
| the split | frozen: fit beat [152,204], 3 held-out beats, 17,499 scored nodes, mask frozen from the recording alone; diseased sheet sealed by content across all 3 files |
| **the seal** | attacked 3 times before it held. Now refuses the sealed specimen in 4 unit conventions, cropped, and subsampled. Same specimen r=0.991–1.000, different specimen r=0.151–0.224, threshold 0.90 in the gap |
| **the operator merge** | **NOT behaviour-preserving.** Two runs of the migrated recipe are BIT-IDENTICAL (0.0); the archive differs by 1.5e-4 after 3 frames, growing monotonically from frame 1. So the archive is evidence about a model we can no longer run |
| generate-path cost | 137 s start-up + **24 s per frame** → the 250-frame recipes are ~2 h each. The fitting path is ~5.5 h per fit |

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

## 6. To resume — Phase 2, and it is a STOP point

**Phase 2 is about the METRICS, and it invents none of them.** They were written over three weeks
in `prototype/cardio_mpm` and audited on 4 July; what was never done is saying which quantity is
which. Four names existed for the same things — `loopscore_residual`, `enclosure_row`,
`morphology_row` (withdrawn in July, still printed by every run) and `descriptors.py` (mine, added
in Phase 1 — the same mistake one turn later). `metrics.py` is the registry that reconciles them,
and it *proves* the shared quantities agree numerically rather than assuming it.

The campaign reads loops on a **10×10 grid of the tissue**, comparing each node's path with the
recording's. That stays — it is what a person judges.

**THE INSTRUMENTS ARE CLOSED.** Seven measurements a claim may rest on: one name/definition/implementation each, each moves on its declared axis and holds on the other eight, each has a zero measured on six models that know nothing, each has the tissue's own beat-to-beat variation, and six of seven return the right number against a closed form. Every defect is recorded on the class that carries it, with the measurement that found it.

> **⚠ THAT SENTENCE WAS FALSE FOR SIX DAYS, and it is the origin of everything in §10.** The
> evidence was gathered and the promotion report said four were eligible, but **no metric was ever
> promoted**: `tiers = {provisional: 14, withdrawn: 4}`, `admitted()` empty, so `Metric.cite()`
> refused all seven. A check in the suite asserted exactly that (`add(..., not admitted())`) — a
> Phase-2 placeholder that outlived Phase 2 and passed for as long as nothing was certified.
> Consumers, finding nothing citable, fell back to `loopscore` (the *objective*, 1.6 steps) and
> then built a gauge to make it behave. Closed for real on **2026-08-09**: four certified
> (`orientation_error`, `coordination`, `peak_excursion`, `path_length`), each signed with its
> evidence in the class. Three of the seven remain provisional on resolving power (`interior_r2`
> 2.3, `openness` 3.3, `chirality_match` 2.3, against `MIN_LEVELS` 5).

**THE FITTED FLOOR IS MEASURED, and it did what was predicted.** Seed-to-seed spread is **5–15× the beat-to-beat** floor (coordination 0.0384 vs 0.0025; openness 0.0340 vs 0.0030). Working unit = largest floor, so every `steps` fell:

| | beat only | + seeds | |
|---|---|---|---|
| `orientation_error` | 10.5 | **10.5** | survives |
| `chirality_match` | 10.6 | **8.7** | survives |
| `peak_excursion` | 30.2 | **8.6** | survives |
| `coordination` | 120.5 | **8.0** | survives |
| `path_length` | 39.5 | **6.5** | survives |
| `openness` | 36.6 | 3.3 | **RETIRED** |
| `interior_r2` | 12.1 | 2.3 | **RETIRED** |

It retired two and promoted none. `openness` is the one calibration had already caught (π/4 for every aspect ratio) — two independent tests condemn it. **Five instruments survive.**

Two caveats: (a) the **same-seed floor is still missing** — that run OOM'd on a card another session filled, so every `same seed` cell read `nan`, and `promotion_report` said *fits: yes* anyway (same silence as the captioner; now checks all three floors separately). Re-running. (b) **Depth is load-bearing**: at 300 iterations the fits have barely learned — `orientation_error` lands at 0.770–0.796 against a null of π/4 = 0.785, `chirality_match` at 0.569–0.606 against a null of 0.5. A spread across models that have not learned is not obviously the spread across models that have. (b) the **decomposition** — the five `residual/` dimensions, which are five of the six quantities still lacking a zero (the sixth is `loopscore_sd`), and one of which scored its best in the whole null bank on the do-nothing model. That is the same admission procedure again, and it belongs with the loop that consumes it.

**Done, 5 of 6:**

1. **One name, one quantity** (`metrics.py --check`, 8/8). 18 entries, each with a definition, the
   code that computes it, and a tier. **0 certified · 14 provisional · 4 withdrawn** — the honest
   state. Checks: every live metric has code; every one has a definition; **no live definition may
   name a withdrawn metric**; a withdrawn metric records why; and the two implementations of each
   shared quantity agree numerically.
2. **The reading surface, and the defect in it.** The grid's margin was ten nodes and the pinned
   band is wider: **36 of the 100 nodes sat on the anchor.** Found in July, never fixed — every
   dashboard since has been read with a third of its panels showing the anchor. Corrected to twenty
   (0/100 in the band); both reported. *My own check was wrong first — it said 0/100 against the
   audit's 36, because the grid maps onto a sheet occupying 0.70 of the world and the factor was
   missing. The audit was right.*
3. **Where is zero** (`floors.py`). Bar = **+0.851** fit / **+0.62** held-out, set by copying the
   previous beat. Muscle-off scores exactly the do-nothing score. Everything from here is reported
   as a difference from the do-nothing row.

4. **How big is nothing** (`noise.py`) — *one floor of three.* **Beat to beat is measured** (no
   fitting needed, and it is the floor that matters: no model may score better than the recording
   agrees with itself). On 10 complete beats: openness ±0.0034 · chirality ±0.0140 · orientation
   ±0.0232 rad · coordination ±0.0025 · loopscore ±0.1294. **Still missing:** same-seed-twice and
   seed-to-seed, both of which need fits (`noise.py --fits`). Until they exist the promotion check
   reports **0 of 14 eligible**, mechanically.
5. **Teach the ruler to see coordination** (`metrics.py`). `coordination` (agreement in *when*
   each node beats) and `orientation_error` (angle between the model's principal axis and the
   tissue's) now exist as registry classes and pass the battery. Coordination took **three
   attempts**: the fundamental-phase construction fails because distance-from-centre peaks *twice*
   per beat (rigid shift scored 0.50, must score 1); peak cross-correlation lag fails the same way
   (the two maxima are half a beat apart); mapping the lag onto the **half period** works —
   1.0000 under rigid shift, 0.0778 under scrambled timing. **Declared defect: cannot tell
   in-phase from exactly antiphase.**

**Remaining:**

6. **Deliver the decomposition per run.** The five named dimensions — size, orientation, openness,
   chirality, shape-detail — on the corrected grid, written into every run's record, so step 3 of
   the loop returns *where* the model is wrong rather than how much.

**THE APPROACH — five tests, and each was added after the previous one let something through:**

| | question | blind to |
|---|---|---|
| 1. registry (`metrics.py --check`) | one name, one definition, one implementation; withdrawn refuses; every zero says where it came from | the numbers themselves |
| 2. battery (`metrics.py --certify`) | does it move on its axis and hold on the other eight? | whether the number is *right*; and any change that is a **symmetry of the population** |
| 3. null bank (`floors.py --nulls`) | what do six models that know nothing score? | whether a difference is above the noise |
| 4. floors (`noise.py`) | how much does it wobble when nothing changed? | whether it means what its name says |
| 5. calibration (`calibrate.py`) | ellipses, where area/perimeter/reach/axis/circulation are closed-form — **is the number right?** | whether tissue is an ellipse |

**The rule, declared before the numbers:** `levels = |tissue-vs-itself − knows-nothing| / (3 × largest measured floor)`, and a metric needs **5** steps to carry a claim (four to rank quartiles, one spare). **A role is not a tier:** the quantity a fit descends and the quantity a claim cites are different jobs. `loopscore` = 1.6 steps → marked `OBJECTIVE`, still optimised and reported, `cite()` raises.

| | zero | tissue | steps |
|---|---|---|---|
| `coordination` | 0.078 | 0.997 | 120.5 |
| `path_length` | 0.0042 | 0.0001 | 39.5 |
| `openness` | 0.339 | 0.006 | 36.6 |
| `peak_excursion` | 0.0011 | 0.0000 | 30.2 |
| `interior_r2` | −0.831 | 0.899 | 12.1 |
| `chirality_match` | 0.500 | 0.946 | 10.6 |
| `orientation_error` | 0.785 | 0.058 | 10.5 |
| `loopscore` | 0.070 | 0.710 | **1.6 — objective** |

Registry 17/17 · battery 0 disagreements in 112 cells · calibration 6 metrics vs closed form.
**Outstanding: the fitted-noise floor only** (running). Every `steps` above can only fall.

**What the tests found:** copying the previous beat scores **+0.851** vs the campaign's best 0.545 · the null bank was written in a vocabulary the registry doesn't contain, so 9 of 14 had no zero and `interior_r2`'s was typed −0.875 against a measured −0.8308 · **`coordination` scored a perfect 1.0000 on the do-nothing model** and `residual/shape_detail` +0.3031, both now refuse · `openness`/`path_length`/`peak_excursion` took (model, recording) and **used only the model**, now paired · `path_length` dropped the closing segment of a closed loop so rolling a beat looked like a timing response · **`openness` returns π/4 for every aspect ratio and swings 25% with orientation** — the battery cannot see it, because turning every loop by one angle is a symmetry of a population that already points every which way. That one is open: the fix invalidates every number measured so far, so it is stated not taken, and an `openness` claim must travel with `orientation_error`.

**The figure Phase 2 reports into** (`figure_metrics.py` → `figures/metrics_figure.png`): nine
single distortions across the top drawn as loops, eight readable measurements down the side, and a
cell per pair saying whether it moved or held. **No cell is red** — every measurement moves on its
own axis and holds on the other eight. The right-hand block is the precision, and it carries the
result that matters most so far: **seven of the eight measurements put the best archived fit more
than three beat-to-beat spreads outside the tissue's own variation, and the one that cannot is
`loopscore` — the composite the entire previous campaign ranked on** (0.4987 against 0.7104 ±
0.1294). Orientation is off by 0.431 rad where real beats vary by 0.058; coordination reads 0.582
where real beats read 0.997. That is the argument for item 6, measured rather than asserted.

**THE STOP:** does the fitted model beat the best trivial baseline by more than the noise, with the
comparison and the margin written into code before the numbers exist? On the evidence already on
disk — copying the previous beat outscores every fit the previous campaign produced — this is a
live outcome, not a formality.

**Two defects carry forward as the first things the loop must explain:** the warm-up does not
settle (33% shift), and the model rests 8% of the beat where the tissue rests 77%.

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

---

## 10. 2026-08-09 — P0 closed, and P1 is NOT a gain verdict

*Written after rebuilding the acceptance statistic and running a four-probe investigation
(9 agents, each finding adversarially refuted) into whether the campaign should pivot from
per-cell stiffness to per-cell contraction gain. **It should not.** Read §6's warning box first;
this section is its consequence.*

### 10.1 What P0 fixed

| | |
|---|---|
| four instruments **certified**, each signed with its evidence in the class | `orientation_error` 10.1 steps, `peak_excursion` 8.5, `coordination` 8.0, `path_length` 6.5 |
| the placeholder check replaced by an **audit** | a certification must be checkable against role, declared null, and ≥ `MIN_LEVELS` from the floors that justified it |
| **`accept.py`** — the acceptance statistic | no oracle (reads two `[G,M,2]` loop arrays), ≥3 ticks enforced, worst-channel rule so a good angle cannot pay for a broken amplitude. 9/9 selftest |
| the amplitude **gauge withdrawn** | `gauge_fix`/`gauge_fix2` raise. Over 64 candidate-rollouts the amplitude channel spans 25.0 steps and the pattern channel 1.5 — **the gauge divided out the channel with 17× the information** |
| `discriminating_power` added | resolving power (vs the null) admits an instrument; it does not say the instrument can rank candidates that all roughly work. Across the bank: `orientation_error` 10.1 → **1.4**, `coordination` 8.0 → **1.5** |
| the **box prior cannot be data-anchored** | it was `[0.2,5]×median(naive)` — the attenuated fit it constrained — so it slid with the bias: implied median 128 clean → 40 at realizable noise → **2.2** at high noise, against a true 132. 40/56 configs excluded some planted modulus. `anchor_from_amplitude` now raises `NotInvertible` |

### 10.2 The verdict: no gain pivot

**Per-cell gain is not identifiable, and the case that it might be was an artefact.**

- **The gain channel is structurally ABSENT from most of the beat.** Gain multiplies `act0`, and
  `activation_pulse.py:78` makes the clock exactly 0 for 120/150 ticks, so **‖A_gain‖ = 0.0 exactly
  at 75.8% of frames** (nonzero at 58/240), where the normal equations are singular. Tick 165 —
  the only tick four probes used — is the **global argmax** of ‖act0‖ (4212.6).
- **0/100 cells clear the 5-step bar for either parameter** (all 100 measured, full beat). Best
  gain cell 2.56 steps, best E cell 0.97; 4/100 gain cells clear even one step.
- **"The gain columns of A are exactly F-free" was a degenerate-base artefact.** Probe B
  linearised at θ=0 where `_lame(0)` gives μ=λ=0, so the stress vanishes and the injected F
  multiplies nothing — the bitwise `0.000e+00` was guaranteed by arithmetic, not measured. At base
  E=130 uniform (whose true-F control is *better*), the same derived-F error moves the gain block
  by rel **9.29e-02** and the θ-free offset by rel **2.061**. The attribution inverts: the offset
  does most of the damage, not the E columns.
- **The multi-frame Fisher ceiling says stiffness, not gain.** Stacking the Grams over the active
  window with clean F and tracking noise only: per-cell relative sd at 5 frames = **E 0.0499 vs
  gain 0.0935** — E is 1.9× better determined, and gain's p90 is 5.6 against E's 0.70. Off-pulse
  frames improve E and add **exactly nothing** to gain. The best-determined per-cell direction is
  `(dE/E, dg/g) = (0.999, −0.058)`: **the one per-cell number this data determines is stiffness.**
- **A per-cell gain map would not be gain.** The gain block reproduces **49.9% in norm / 24.9% in
  energy** of the per-cell stiffness signal, and the imitator has **165× the spread** of the true
  gain map with `corr = +0.060` — unstructured debris, not a bias you could regress out.
- **There is no per-cell gain in the forward model.** `material_cardio_cells.yaml` gives
  `active_force` a single global `amplitude: 20.0`; per-cell gain exists only in the estimation
  harness (`assemble.py`'s own docstring says so). Per-cell stiffness *is* in the model
  (`seed_from_segmentation`). PREMISES.md #3: an operator that never acts is not part of the model.

P0's founding 22× is not contradicted — it is about a **uniform** multiplier, these are about
**per-cell** structure. The pivot conflated them.

### 10.3 Three errors in the P0 work itself

1. **The null bar is the wrong tissue's.** `accept.py`'s nulls come from the registry, which
   measured them **on the recording** — `peak_excursion` null 0.0011 is the recording's own median
   excursion. The synthetic sheet beats **8.7× harder** (0.0066–0.0120). The do-nothing model rolled
   out on *this* sheet scores **95.06 steps, not 6.65**. So "the per-frame solve at 19.33 is worse
   than knowing nothing" is wrong in its comparison, and **every steps figure in §10 and in the P0
   record inherits the mis-calibration**. Rankings are unaffected (same statistic, same floors).
   *Fix: `accept.py` must measure the null on the sheet it is scoring.*
2. **"E is unobservable" is a property of the operating point, not the physics.** The spec sits at
   a **stationary point** of its own amplitude-vs-E curve (exponent +0.034, turning point E≈212.6).
   The same sweep gives **117 steps at drag 3, 232 at drive amplitude 80, 364 monotone with the
   pulse always on**.
3. **Per-cell E structure IS visible; one beat was the wrong window.** Over **three** beats the best
   uniform-E impostor of the planted field costs **3.58 steps** (0.73 at one beat), and **8.60** at
   drive 80 — above the null. Flattening the whole per-cell E field costs about as much as a 5%
   gain error.

### 10.4 The apparatus is what the per-cell map measures

`activation_pulse` writes one Gaussian bump, σ=0.12, at the domain centre. Consequence, measured:
per-cell active force spans **max/min = 1.6e4**; 52/100 cells receive <10% of the strongest cell's,
23/100 <1%, **37/100 are essentially frozen**. Per-cell gain sensitivity correlates **ρ=+0.959 with
the active force a cell happens to receive** and only **ρ=+0.142 with its own planted gain**; a
cell's amplitude is 0.43 R² from its *radius* and 0.0095 R² from its own (E, gain).

**A per-cell map on this spec is a map of the stimulus geometry.** `activation_pulse` already
supports `profile: uniform` (line 70) and nobody has run the comparison under it.

Two conditioning facts worth keeping: `cond(G_blockdiag) ≈ cond(G)`, so the near-singularity is
*inside* both blocks and not in the coupling; and the raw `cond` 5.96e11 is inflated by a 50×
column-norm mismatch between blocks, so column scaling recovers most of it.

### 10.5 What to run next, in order, with decision rules

1. **Amplitude-matched Fisher ceiling** (~15 min). Re-run `p1f_stack.py` with the drive scaled so
   peak excursion matches the recording's 0.0011 (`amplitude` 20 → ≈2.3), all 29 active ticks, read
   on a 15-px control grid (~37 nodes/cell) not 100 particles. **Rule:** if median per-cell gain
   relative sd exceeds the planted spread (0.319), per-cell gain is undecidable on the recording
   *even with a perfect F*, and P1 becomes a **global-parameters verdict**. Back-of-envelope
   transfer (arithmetic, not measured) already suggests E ~63% / gain ~119% — i.e. **one beat may
   determine neither parameter per cell.**
2. **The confound forward test** (~25 min). Plant gain **exactly uniform**, E heterogeneous, run the
   estimator that would ship; report `corr(g_hat, E_true)`, `std(g_hat)` against a planted zero.
   **Rule:** `std(g_hat) > 0.1` or `|corr| > 0.3` under any F condition ⇒ a per-cell gain product is
   not a contractility measurement and must not ship as one. Run the converse too.
3. **Uniform-activation operating point** (~30 min). Repeat the P0 sweeps and the Fisher stack with
   `profile: uniform`, period 50, duration 12 (the recording's beat), amplitude-matched. **Rule:** if
   per-cell E identifiability improves >2×, every P0/P1 conclusion must be re-taken at the
   recording's operating point.
4. **Amplitude-blind re-reading of P0's founding sweep** (~5 min). `peak_excursion` was the limiting
   instrument for 37/40 candidates, "steps" is an absolute world distance, and gain is *defined* as
   an amplitude multiplier — comparing an amplitude-defined parameter to a shape parameter on an
   amplitude-limited statistic is near-tautological. **Rule:** if gain's advantage lives only on
   `peak_excursion` and vanishes on `orientation_error`/`coordination`, the 22× is an artefact.
5. **C=472 at real scale** (~25 min). `System(real=True)` and `small_labels_full_472.tif` exist; A is
   [472000 × 944] float64 ≈ 3.6 GB, fits. Cells here are 4.7× larger in area, so every per-cell
   number on record is optimistic by an unquantified amount.
6. **Scoped gradient probe** (half a day engineering). **This is the branch that matters.** The
   entire F blocker exists *only because the algebraic formulation takes F as an input*; a
   differentiable rollout never measures F — it is internal state propagated from the initial
   configuration, and the only observable is x(t), which the recording supplies. Untested in six
   rounds. **Rule:** if it recovers per-cell E to ≤10% from positions alone, P1's premise was wrong
   for a reason nobody tested.

**Do not spend time on:** another single-tick algebraic experiment at tick 165 (the global argmax of
‖act0‖; four probes have saturated it), or any steps number quoted against the imported 6.65 null.

**The parameterisation is settled either way (Cedric, 2026-08-09): one learnable per cell.** That
matches the generative truth exactly — `seed_from_segmentation:209` broadcasts one E per cell to its
particles — so there is no approximation error, unlike F which is a genuine field. 10.2's evidence
says the learnable should be **E**, and 10.5(6) says the machinery should be the gradient route. The
open question is spatial rank: keep one learnable per cell and let a smoothness/shrinkage prior
collapse the unidentifiable directions, so the low-rank basis is *learned* rather than declared.

### 10.6 Artifacts

`prototype/cardio_cells/crash/`: `accept.py` (the statistic), `boxprior.py` (+`_p0`/`_ident.json`),
`p1a_*` (per-cell ladder, ceiling, null, mono, actcorr), `p1b_*` (gaincol, leak, split),
`p1bv_*` (base sweep, phase scan, repro), `p1v_verify.*`, `p1v_impostor.*`, `p1v_long_impostor.*`,
`p1x_verify.*`, `p1c_*` (regime table), `p1e_alias.*` (the confound), `p1f_stack.*` (Fisher).
Plan: `discovery_cardio_mpm/PLAN.md`. Nothing sealed was opened; `prototype/cardio_mpm/archive`
was not written to.

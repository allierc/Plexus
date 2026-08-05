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
| 2 — certify the ruler, find the floor | **the instruments are DONE** | the null bank is measured: **the bar is +0.851 on the fit beat, +0.62 held-out** — copying the previous beat, no physics. **STOP point** |
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

Two things remain and **neither can reverse a decision already taken**: (a) the **fitted floor** — measuring, not deciding; it can only *lower* every `steps` figure, so it can retire an instrument and never promote one. Cost measured: 11.0 s/it alone, 72–85 s/it under a neighbouring session holding four 13 GB jobs on both cards → ~6 h. (b) the **decomposition** — the five `residual/` dimensions, which are five of the six quantities still lacking a zero (the sixth is `loopscore_sd`), and one of which scored its best in the whole null bank on the do-nothing model. That is the same admission procedure again, and it belongs with the loop that consumes it.

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

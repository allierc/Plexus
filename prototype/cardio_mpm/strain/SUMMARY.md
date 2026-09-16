# Cardiomyocyte sheets as per-cell active strain — what was done, what it says

*Written 2026-09-16. `STATUS.md` is the running laboratory record, dated and in order;
`PLAN_R2.md` is the plan that drove the last week. This file is the standing summary: what the
model is, what the learnables mean, what the experiments returned, and what may be said out loud.*

Two dishes of human iPSC-derived cardiomyocytes, grown to a confluent sheet on a 15 kPa gel and
filmed beating: one healthy (`0_B_15kPa_1`, 239 frames), one carrying a hypertrophic
cardiomyopathy mutation (`1_HCM_15kPa_MR44_W3_1`, 299 frames), both at 24 frames a second, both
already tracked on a 137 × 137 lattice of points 15 px apart. Nothing else: no electrical
recording, no calcium dye, no stain, no force measurement.

---

## 1. The three learnables, said plainly

Every cell in the sheet carries its own small set of numbers. Three of them are the core of the
model, and they are three different kinds of thing — one is what the cell *tries* to do, one is
*which way* it does it, one is how it *resists* being deformed. They are not interchangeable and
they are not equally well determined.

### g — the active strain (how hard the cell pulls)

**g is the fractional shortening the cell would reach along its own fibre if nothing resisted it,
at full activation.** It is a strain: dimensionless, a change of length divided by the length at
rest. `g = 0.04` means "this cell wants to be 4% shorter than its rest length".

It is not the shortening you see on the film. A cell that pulls hard while its neighbours pull
hard in the opposite direction barely moves. What the film shows is the *realised* shortening,
which is smaller — about 60% of g for the healthy sheet — and the gap between the two is decided
by the mechanics: the neighbours, the substrate, and the stiffnesses. This is exactly why the fit
exists: the realised shortening can be measured directly from the tracking, but g cannot.

In the model this enters as a change of the cell's **rest length**, not as a force. Once a frame,
every material point of cell *j* has its deformation gradient F multiplied on the left:

    F  <-  ( I + c_j(t) · f_j f_jᵀ )  F

where f_j is the cell's unit fibre direction and c_j(t) is sized so that the accumulated active
part is exactly `I + γ(t) · g_j · f_j f_jᵀ`. The engine computes stress from F, so multiplying F
this way tells the elastic law "you are stretched beyond your rest length along f", and the
material pulls itself shorter along f until the stress balances against its neighbours. The mass
never changes; only the length the material considers unstressed. A sarcomere does the same thing.

The sign is the whole content: `I + c f fᵀ` contracts, `I − c f fᵀ` grows. The growth convention
is what `src/plexus/morph.py` uses to inflate a ball into a cow, and the first version of this
model had it backwards — measured consequence: the fitted axes came out perpendicular to the
tissue's (axis agreement −0.85 instead of +0.85).

**A second active strain, g2, was added later** and turned out to matter more than anything else
tried: the active strain *across* the fibre, `I + γ(t)(g f fᵀ + g2 f⊥ f⊥ᵀ)`. It is negative in
both sheets — the cell *thickens* across the fibre while shortening along it — and adding it took
the fit from explaining 69% of the per-cell motion to 87%. Physically it is area conservation: a
cell that shortens 4% along its fibre and thickens 3% across it keeps nearly its area, which is
what the recording shows and what a rank-1 model cannot express.

### φ — the fibre orientation (which way the cell pulls)

**φ is the direction of the cell's contractile axis**, one angle per cell, defined modulo 180°
because an axis has no head or tail. It is what f_j = (cos φ_j, sin φ_j) is built from.

In real muscle this is the direction the myofibrils run. Here it is inferred from motion, not seen:
nothing in the phase-contrast image shows a myofibril. The initial guess comes from the recording
itself — for each cell, the eigenvector of the most negative principal strain at its peak — and
the fit then moves it by a median of about 10° from that guess.

φ is well determined: in the planted test the fit recovers a cell's axis to 5° when only g, φ and
stiffness are free, and to 10–12° once the transverse strain g2 is also free (g and g2 trade
against each other through the axis, since swapping them and rotating φ by 90° is nearly the same
tissue). Two independent fits of the same sheet agree on φ to 7–10° cell by cell.

### E — the stiffness (how the cell resists being deformed)

**E is the Young's modulus of the cell's material**, the constant relating stress to strain:
stress = E × strain. A stiff cell needs more force for the same deformation, and in a tug-of-war
between two cells the stiffer one imposes its strain on the softer one.

It enters the engine the standard way: E sets the Lamé parameters μ and λ of the elastic law at a
fixed Poisson ratio of 0.3, and those are what turn F into stress.

**E is reported and never claimed.** Three measured reasons:

1. *Not identifiable per cell.* In the planted test, per-cell E recovers only to 0.64 of its
   planted spread (correlation 0.45) against a bar of 0.5, while g reaches 0.32 and φ 10°. This is
   the panel that looks bad in `out/figures/fig3_planted_recovery.png`, and it looks bad correctly.
2. *Only relative.* Scaling every cell's E by the same factor changes no strain except through the
   substrate spring κ, which is fixed by hand. The absolute number (median 236 in the spec's stress
   units) means "behaves this stiffly against this substrate", not a modulus in pascals.
3. *It absorbs model error.* Left completely free on real data, per-cell E spread over a factor of
   200 (values from 3 to 600), which is far outside anything the identifiability test certified. A
   shrinkage prior toward uniform (weight 0.3 on the mean squared deviation of log E) holds it to
   ±12% and costs 0.04 in explained variance. That prior is on in both reported fits.

A retraction follows from this, recorded in `STATUS.md`: an earlier rank-1 fit had the HCM sheet
2.6× stiffer than the healthy one, and that difference disappeared (236 vs 222) the moment the
transverse active strain g2 was added. The rank-1 model had been using stiffness to fake the
thickening it could not express. Withdrawn.

### The rest of the numbers

| per cell | what it is | claimed? |
|---|---|---|
| **g** | active strain along the fibre at full activation | **yes** |
| **g2** | active strain across the fibre (negative = thickening) | **yes** |
| **φ** | fibre axis (mod 180°) | **yes** |
| **δ** | delay of the cell's excitation, in frames of 41.7 ms | **yes** |
| E | Young's modulus, shrunk toward uniform | reported only |
| log τ_rise, log τ_plateau, log τ_decay | scalings of the cell's own excitation time course | machinery |
| a₁, a₂ | weights on two shared temporal modes | machinery |
| log κ | substrate adhesion of that cell | machinery (worth 0.004) |

Shared by the whole sheet: the clock γ(t), four numbers (onset, rise, plateau, decay) giving a
0-to-1 activation pulse, and the two temporal mode shapes ψ₁, ψ₂ (one value per frame each). Fixed
by hand: substrate stiffness κ = 10⁴, drag 150, Poisson ratio 0.3, shrinkage weight 0.3.

About 4,800 numbers per sheet, against roughly 113,000 independent observations per beat.

---

## 2. How the fit works

The initial state is not learned. Every particle starts at the position the segmentation gives it,
with F = I (unstressed) and zero velocity, at a frame where the tissue is measurably at rest. One
rollout of the stock Plexus material-point engine advances the sheet through a beat; the active
strain is injected each frame through `engine.run(on_frame=...)`, functionally so the autograd tape
survives, exactly as `plexus.morph` injects its growth field. **Nothing under `src/plexus` is
edited**; the one new operator, a per-cell substrate spring, is registered from the prototype.

The observable, computed identically from the model's particles and from the recording's tracked
nodes, is the **per-cell affine map**: for each cell and frame, the least-squares 2 × 2 matrix A
and displacement u that best describe how that cell's points moved from rest. Roughly 38 tracked
nodes and 120 particles per cell, so the raw tracking noise averages down while the cell's stretch,
shear, rotation and translation are all retained. The loss is the squared mismatch of A and u,
each scaled by the recording's own spread.

Gradients run back through all 570 sub-steps of the rollout to the parameters; Adam, 250 steps.
**This is parameter estimation, not machine learning**, so the final fits use every beat of the
recording — one rollout per beat from its own rest, gradients accumulated before the step, so the
memory stays at one rollout and the cost scales with the number of beats.

Boundary: the sheet in the film is a crop of a larger monolayer, so particles within one cell width
of the edge are driven directly by the recording, and the 141 healthy (136 HCM) cells touching that
ring are excluded from the loss and from every map and number.

---

## 3. What was measured, in order

**The apparatus, before any biology.** The gradient was certified against central differences in
float64 at three step sizes: ratios 1.0000 for g, φ, log E and the clock. The particle layout was
not innocent — the same parameters gave 0.29, 0.64, 0.55 explained variance at 30, 50, 80 particles
per cell — and the forward converges by 120 per cell on a 128 grid, with two random layouts
agreeing to 0.958. The substrate was ringing: a step test on the clock showed the sheet overshooting
to −32% of its plateau and still 10% off twenty frames later, because the substrate mode was
under-damped; drag 150 removes it.

**Two data premises had to be corrected.** The dataset's second tracking of the healthy movie
(`healthy.npy`) disagrees with the main one at every spatial scale while each repeats itself
beat-to-beat at 0.99. Normalised cross-correlation of 49-pixel patches on the raw frames settles it:
the pixels move as the main tracking says (r = 0.97 in both axes, 0.6 px error on 3 px of motion)
and not as `healthy.npy` says (0.25 / −0.10). The inherited "tracker floor" that had made the whole
spatial field look like noise was that file's failure. And the dataset's beat onsets are peaks of
mean speed, i.e. mid-upstroke; referenced to true rest the beat is an 8-frame rise, a peak mean
shortening of 2.1%, and relaxation complete about 25 frames after onset.

**Identifiability, on planted data with the recording's own rest-frame noise, before touching the
real thing.** The numbers are in section 1. In short: g, g2, φ and δ recover; per-cell stiffness and
per-cell relaxation time do not.

**The fit ladder.** Explained variance of the per-cell strain maps on the healthy sheet, each row
adding one thing to the row above: measured amplitude and axis with no fitting 0.40 → fitted g, φ,
E, one clock 0.69 → per-cell delays 0.70 → **second active axis g2 0.82** → per-cell excitation
time course 0.85 → two shared temporal modes 0.86 → per-cell adhesion 0.866. The recording's own
ceiling, from its singular value decomposition, is 0.87 with one fixed spatial pattern, 0.94 with
two, 0.97 with three; so the model now sits at the two-pattern level and the remaining gap is the
spatial structure of the dominant mode, which no per-cell number tried so far expresses.

---

## 4. The final fits

`out/fits/healthy_allbeats` (beats 1–3, 108 min) and `out/fits/hcm_allbeats` (beats 1–4, 144 min),
120 particles per cell (56,640 and 52,080), 250 iterations.

| | healthy, per beat | HCM, per beat |
|---|---|---|
| strain maps, R² | 0.870 / 0.872 / 0.871 | 0.894 / 0.900 / 0.898 / 0.898 |
| displacements, R² | 0.935 / 0.937 / 0.935 | 0.949 / 0.958 / 0.958 / 0.958 |
| per-cell shortening correlation | 0.94 | 0.97 |

Every beat is fitted, so the meaningful figure is the spread across beats: 0.002 healthy, 0.006
HCM. One parameter set describes every beat equally well, which a set tuned to a single beat could
not. The generalisation statement is instead the **continuous rollout**: the model run once through
beats 1 to 3 without ever being reset, the fitted clock firing once per beat, explains **0.849** of
the per-cell strain — two beat boundaries of accumulated drift cost 0.03. That is what
`out/movies/cont_cells.mp4` shows.

Two independent fits (different optimiser seed, different particle layout) agree per cell at
r = 0.90–0.92 on g, 0.88–0.93 on g2, 0.83–0.88 on the delay, and 7–10° on the axis; the median
cell-to-cell difference is about a fifth of each map's spread. That is the error bar on every map.

---

## 5. Healthy against HCM

One dish per condition. These are descriptions of two specimens, not a statistic.

| | healthy (331 cells) | HCM (298 cells) |
|---|---|---|
| recorded peak shortening per cell, median | 0.025 | 0.040 |
| fitted g (active strain along the fibre) | 0.037 | **0.063** |
| fitted g2 (across the fibre) | −0.035 | −0.033 |
| g2 / g over active cells | −0.70 | **−0.48** |
| cells with g < 0.01 (silent) | 19% | **10%** |
| local axis alignment with neighbours | 0.23 | **0.35** |
| fitted E | 236 | 222 |
| excitation jitter across cells (sd of δ) | 0.079 s | 0.076 s |

Read as biology: the HCM cells pull about 1.7× harder along their fibre with the *same* absolute
thickening across it, so they lose area during contraction where the healthy cells nearly conserve
it; half as many sit out the beat; and neighbouring cells agree on direction far more often. In a
dish nothing imposes a fibre direction, so a healthy monolayer is expected to be disordered and
alignment has to be generated by the cells themselves, mechanically — which is consistent with
harder-pulling cells, not with health.

**Timing, measured without the model**, averaged over each sheet's beats:

| | healthy | HCM |
|---|---|---|
| contraction, 10% → peak | 0.29 s | 0.30 s |
| time above half-peak | 0.60 s | 0.70 s |
| relaxation, peak → 10% | 0.67 s | **0.81 s** |
| beat period | 2.11 s | 2.34 s |

The rise is identical, the fall is 20% slower. Prolonged contraction and impaired relaxation are
the classic diastolic side of hypertrophic cardiomyopathy, and hypercontractile myocytes are the
classic systolic side; the fit finds both without being told either.

**But where the slowness lives is not identifiable.** Freezing the HCM fit's shared excitation
clock at the *healthy* sheet's values and refitting everything else costs 0.002 in explained
variance against the control that freezes HCM's own clock. The per-cell plateau scaling compensates
almost exactly (×1.15 against ×0.78, landing on the same effective duration), so the data fixes the
*product* of the shared and per-cell durations, not either factor. "HCM is driven by a longer
excitation" is therefore not a claim this data supports; "the HCM tissue relaxes 20% more slowly"
is an observation that stands on its own.

---

## 6. What may be said, and what may not

**May be said.** Per cell, from motion alone: the active shortening along the fibre, the thickening
across it, the fibre axis, and the excitation delay — each recovered on planted data, each
reproducible across independent fits. Of the two sheets: the HCM cells contract about 1.7× harder,
lose area rather than conserving it, are half as often silent, are twice as locally aligned, and sit
in a tissue that relaxes 20% more slowly at an unchanged rate of contraction.

**May not be said.** Per-cell stiffness (not identifiable, and only relative to a hand-set
substrate). Per-cell relaxation time (recovers to 0.80 of its spread). That the two sheets differ in
stiffness (retracted). That the HCM excitation is longer (not separable from the cells' own
timing). Anything statistical about the disease: one dish per condition.

**What would move it.** A second dish per condition, which is the only way the comparison becomes
more than a description. An independent stiffness measurement — indentation, or traction on the gel
— which would break the g/E trade-off from the outside. A myofibril stain of the same field, to
check the inferred contraction axis against the actual fibre orientation. And, for the remaining
13% of unexplained motion, a per-cell parameter that can produce the recording's second spatial
pattern; nothing tried so far does.

---

## 7. Where things are

| | |
|---|---|
| `STATUS.md` | the dated laboratory record — read this for how a number was arrived at |
| `PLAN_R2.md` | the plan behind the last week of work |
| `model.py` | the spec, the learnables, the rollout, the loss |
| `recording.py` | the recording in world units, the per-cell affine map, beat windows |
| `fit.py` | the optimiser; `--beats`, `--free`, `--init-from`, `--checkpoints` |
| `segmentation.py` + `seg/` | rebuilds the instance segmentation from the movie (verified bit-identical) |
| `fd_check.py`, `s3_gate.py`, `layout_convergence.py`, `step_test.py`, `rank_ceiling.py` | the apparatus checks |
| `patch_check.py`, `hcm_referee.py`, `tracker_scale.py` | the tracking referees |
| `s4_score.py`, `s4_residual.py`, `compare_sheets.py`, `rounds_table.py` | scoring and comparison |
| `figures.py`, `fig_alignment.py`, `fig_seeds.py` | the seven figures in `out/figures` |
| `movie*.py`, `render_progress.sh` | the three movies in `out/movies` |
| `out/fits/healthy_allbeats`, `out/fits/hcm_allbeats` | the reporting models |

Environment: `/workspace/.conda_envs/neural-graph-linux/bin/python`,
`PYTHONPATH=/workspace/Plexus/src`, two RTX A6000. A fit is about two hours; scoring a fit, a few
minutes; the whole figure and movie set, about twenty minutes.

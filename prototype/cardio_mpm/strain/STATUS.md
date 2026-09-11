# cardio_mpm/strain — status

*Started 2026-09-09. Restart of the cardiomyocyte work on the differentiable Plexus MPM engine
(`src/plexus/morph.py`'s machinery). Nothing under `src/plexus` is edited; everything lives here.
Plan: S0 certify the gradient → S1 the model rests/peaks/rests → S2 loss + cost → **S3 STOP:
identifiability gate on planted truths** → S4 first live round on the healthy sheet → S5 HCM.*

## The model, in one paragraph

472 measured cells on a 2D sheet (the segmentation of `prototype/cardio_cells`, rebuilt here by
`segmentation.py`: same detector, 472 cells, 9.7% of the field off its Voronoi cell, as recorded in
FINDINGS.md). Each cell j carries a fibre-shortening gain g_j (dimensionless strain at full
activation), a fibre axis φ_j and a Young's modulus E_j; the sheet shares ONE clock γ(t) in [0,1]
(4 numbers), because the contraction is synchronous. Once a frame every particle of cell j gets
F ← (I + c_j(t) f_j f_jᵀ) F with c_j chosen so the accumulated active part is exactly
I + γ(t) g_j f_j f_jᵀ (`model.py`). A substrate spring κ = 1e4 (stock `mpm_anchor`,
`applies_to: substrate`) stands for the 15 kPa gel; the outer 0.03 world units (about one cell) are
prescribed from the recording, because the sheet is a crop of a larger monolayer. The observable,
for the model and the recording alike, is the per-cell least-squares affine map A_j(t), u_j(t)
over each cell's ~38 tracking nodes / ~50 particles (`recording.cell_affine`), and the loss is the
mean squared mismatch of A and u, each scaled by the recording's own spread.

## What was established today

| step | result |
|---|---|
| **S0 gradient** (`fd_check.py`, float64 referee, real 472-cell map) | analytic / central-difference ratio at h/16: **g 1.0000, φ 1.0000, log E 1.0000, clock 0.9999**; backward deterministic to 1e-14 in float64, 1e-5 in float32. Certified. |
| **S1 forward** (`s1_forward.py`, 47k particles, 57-frame beat window) | with g_j, φ_j READ OFF the recording and one fitted clock, nothing fitted: per-cell shortening correlation with the recording **+0.87**, axis agreement **+0.85**, strain-component correlations 0.70–0.80. Rest recovery with κ=1e4: last-frame strain **6%** of peak (free sheet: 18%, and it rings). Band prescription lifts centroid-displacement correlation 0.33 → **0.65** and expansion/shortening 0.40 → **0.95** (recording 1.11): the sheet is confined. |
| **S2 loss + cost** (`fit.py --target planted`, no noise) | **7.9 s / iteration, 14.8 GB** at 23.6k particles; 150 iterations = 20 min. Recovery from a flat start: g **0.14** of its planted spread (r 0.91), φ **5.7°**, clock exact (t0 7.31 vs 7.23 frames), **log E 0.51** of its spread (r 0.63). Stiffness is the weak family, as the S0 gradient magnitudes (E ~1000× smaller than g) foretold. |
| **S3 gate** (`s3_gate.py`, 16 planted fits, 200 iterations each, `out/s3_table.json`) | Rule: a family ships only if median-over-seeds of median-over-cells \|estimate − truth\| / spread(truth) < 0.5, with the recording's own rest-frame noise added. **All four families free, 3 seeds: g 0.098 (r 0.93), φ 5.2°, log E 0.400 (r 0.62–0.69) — every family ships, E at the edge.** Diagnostics: E alone with the others at truth recovers to **0.08–0.09** (r 0.92) — E is well determined *in itself*, its 0.40 in the joint fit is entanglement with g and φ, not noise; φ alone 2.2°. At 3× the noise E fails (0.51), g still 0.11. Without φ free, g degrades to 0.22–0.25 and the axis stays at its 10.9° init. E free lowers the planted loss 0.0217 → 0.0166 even on planted data. |

### Two defects found and fixed on the way

- **The sign of the injected strain.** F ← (I − c f fᵀ) F is morph's GROWTH convention: it tells the
  elastic law the material is compressed along f, and the material expands along f. Measured with
  that sign: axis agreement with the recording **−0.85** (perpendicular), expansion twice the
  shortening. Contraction is F ← (I + c f fᵀ) F. One character.
- **The clock must be zero on frame 0.** A clock at γ(0) = 0.256 left the sheet expanded by 0.256 g
  along every fibre once γ returned to 0 — 64% of the peak strain, looking like a model that does
  not rest. γ is now normalised so γ(0) = 0 exactly.

### The beat, as the data actually has it (`recording.beat_window`)

The frozen split's "onsets" are peaks of mean nodal speed, i.e. MID-upstroke. Referenced to them a
beat looks strained for 45 of its 52 frames. Referenced to REST (median position over the 18 frames
before the next beat): rest → 8-frame upstroke → peak mean shortening **0.021** per cell (p90 0.054,
max 0.10) → relaxed by ~25 frames after onset → rest. Expansion ≈ shortening (area-preserving),
mean rotation 0.015. Windows: beat 3 (fit) = frames 144–200; held-out beats 0–47, 43–97, 93–148 —
consecutive windows overlap only in rest frames, each holds one contraction.

## The finding that reorders the premises: the "second tracking" is broken

`healthy.npy` (80×80 top-left of the same lattice) and the 137-grid derivatives disagree on the
beat's spatial pattern at **every** scale 45–375 px (`tracker_scale.py`: per-cell shortening
r = −0.06, axis agreement 0.1), while **each repeats its own pattern beat-to-beat at r = 0.99**.
Normalised cross-correlation of 49 px patches on the raw frames (`patch_check.py`, 3 frame pairs,
400 nodes) referees it: **the pixels move as tracking 1 says (r = 0.97 in x and y, 0.6 px median
error on ~3 px of motion) and not as `healthy.npy` says (r = 0.25 / −0.10).**

Consequences: (1) the inherited premise "two trackings agree only 0.27 in space, so the spatial
field is mostly noise" was measuring a bad tracker, not the recording; (2) tracking 1's per-cell
strain pattern is real and reproducible; (3) the noise model for the gate is tracking 1's own rest
frames (`data/noise_rest.npz`: per-cell |A−I| rms 0.0008 against a peak signal of 0.026), and the
tracking-difference "noise" was deliberately not written.

## Files

| file | what |
|---|---|
| `segmentation.py` | rebuilds the 472-cell map from the movie; writes `data/labels_*.npy`, `data/cells_2560.tif` (nearest-node rasterisation — the old recipe's regular-grid assumption misplaced 44% of nodes) |
| `recording.py` | the recording in world units through `discovery_cardio_mpm/data.py` (seal kept); `cell_affine`, `beat_window`, `clock_init`, `fibre_init` |
| `model.py` | `build_spec` (stock operators, `implementation: differentiable`), `Params`, `rollout` (active strain + E leaf through `on_frame`), `affine_loss`, `band_prescription` |
| `fd_check.py` | S0; `--dtype float64` is the referee |
| `s1_forward.py` | S1; kinematics, recording comparison, cost |
| `fit.py` | the optimiser; `--target planted|recording`, `--free`, `--noise rest`, `--init-truth` |
| `s3_gate.py` | the queue and the verdict table |
| `tracker_noise.py`, `tracker_scale.py`, `patch_check.py` | the tracker investigation |
| `out/` | every number quoted above, as JSON; `out/fits/<tag>/` per fit |

Environment: `/workspace/.conda_envs/neural-graph-linux/bin/python`, `PYTHONPATH=/workspace/Plexus/src`,
two RTX A6000 in the devcontainer.

## Next

1. Read `python s3_gate.py table`. If g ships and E does not, S4 fits g, φ, clock per cell with E
   global (or a low-rank E); if E ships, all four.
2. S4: `fit.py --target recording --beat 3`, then score the three held-out windows with the
   fitted parameters (only the clock's t0 may be re-aligned per beat), against the replay baseline
   computed on the SAME per-cell affine observable.
3. Only then the HCM sheet (sealed; `split.py` refuses it without a token).

## S4 — first live round (2026-09-10, `out/fits/s4_live_s0`)

Fit on beat 3, all four families free, 200 iterations (30 min). Scored on the same per-cell
affine observable (`s4_score.py`), only the clock's onset re-aligned per held-out beat:

| window | model R²(A) | model R²(u) | shortening corr | axis agreement | replay R²(A) | replay R²(u) |
|---|---|---|---|---|---|---|
| beat 1, held out | **0.677** | **0.701** | 0.868 | 0.895 | 0.981 | 0.961 |
| beat 2, held out | **0.679** | **0.701** | 0.867 | 0.898 | 0.985 | 0.968 |
| beat 3, fit | 0.681 | 0.673 | 0.867 | 0.896 | 1 | 1 |

No overfitting (held-out = in-sample), a large gain over the measured init (R²(A) 0.40 → 0.68),
and **it does not beat replay** (0.98), as the plan's rule required. Recorded as such. Replay is a
161k-number lookup table of a beat whose repeats agree at 0.98; the model is 1,420 numbers.

**The residual** (`s4_residual.py`): uniform across components (R² 0.62–0.72 for area change, the
two shears, rotation, u_x, u_y — no single channel fails) but **not across time**: residual/signal
is 0.46 during the plateau (frames 10–20) and rises above 1 in the relaxation (0.89 at frame 24,
1.6 at 30, 2.2 at 34). The model relaxes slower than the tissue; the fit compensated by lengthening
the clock (dur 14.9 → 18.7, tau_d 2.4 → 4.8 frames). Cause, on inspection: with anchor κ = 1e4 and
drag k = 30 the substrate mode is **under-damped** (damping ratio k / 2√κ = 0.15, period 31 frames,
decay 33 frames) — the S1 anchor sweep was confounded by this (κ = 1e5 at k = 30 "broke" because it
rang). Critical damping needs k = 2√κ.

**The parameters**: g median 0.038 (1.7× the amplitude read off the recording; efficiency), but
per-cell g correlates only 0.34 with the measured amplitude; **E ran away** (median 23, p10 3,
p90 893, log-sd 2.0 against a planted-test spread of 0.3): on real data with model error, the weakly
determined E field absorbs residual. Per-cell E needs a prior (shrinkage to uniform) or must stay
global until the mechanics is right. φ moved 19° (median) from the measured axis.

Round 2 (next): critically-damped substrate (κ, k) ∈ {(1e4, 200), (3e4, 346), (1e5, 632)} checked
forward against the relaxation tail; then a refit with E global (or shrunk) and the same scoring.

### Rounds 2–3 (same night)

| round | change | held-out R²(A) | R²(u) | shortening r | axis | E log-sd |
|---|---|---|---|---|---|---|
| 1 | sigmoid clock, all four free | **0.678** | **0.701** | 0.868 | 0.895 | 2.02 |
| 2 | free per-frame clock, **E fixed** | 0.558 | 0.478 | 0.842 | 0.848 | 0 |
| 2b | free per-frame clock, E free | 0.677 | 0.648 | 0.871 | 0.887 | 2.00 |
| 3 | free clock, E shrunk to uniform (λ=1) | 0.624 | 0.526 | 0.866 | 0.875 | 0.12 |

What this says. (i) The clock's functional form is not the problem: the free 57-number clock
changes nothing (round 1 vs 2b), and the relaxation-phase residual (ratio > 1 after frame 26) is
unchanged by it and by critical damping of the substrate (`s1_damp_*`: the tail is identical at
drag 30 and 200). (ii) The per-cell E field's freedom buys 0.06 in R²(A) and 0.17 in R²(u) on
held-out beats (round 1 vs 3), at the price of a spread (3 to 900) far outside the regime the gate
certified (sd 0.3). Because the held-out beats repeat the fit beat at 0.98, "generalising" cannot
tell a physical E map from residual absorbed by an ill-conditioned one. Per-cell E is therefore
reported but not claimed until a λ sweep says how much spread the data insists on.
(iii) **The residual's structure is TIMING.** The recording's per-cell strain over the beat is
88% one temporal mode and 6.7% a second whose time course is early-negative / late-positive: cells
lead or lag. Per-cell peak frames spread p10 7 / p50 13 / p90 21 — 14 frames, 0.6 s — with no
spatial gradient (the old "no wave" finding was about a gradient, not about jitter). A one-clock
model cannot represent that, whatever the clock's shape. Round 4 adds a per-cell delay δ_j
(γ_j(t) = γ(t − δ_j), 472 numbers, zero = the one-clock model).

### Round 4 — per-cell clock delays (`out/fits/s4_live_r4_delay_Efree`)

| round | change | held-out R²(A) | R²(u) | shortening r | axis |
|---|---|---|---|---|---|
| 1 | one clock | 0.678 | 0.701 | 0.868 | 0.895 |
| **4** | one clock + per-cell delay δ_j (E free as in round 1) | **0.703** | **0.742** | 0.878 | 0.901 |

The delays are real numbers of the right size: sd **3.3 frames** (0.14 s), p10 −4.4 / p90 +2.9,
against the directly measured spread of per-cell peak frames (p10 7 / p90 21). The gain is
modest (+0.025 in R²(A), +0.04 in R²(u)) because the timing mode is 6.7% of the variance; the
relaxation-phase residual (ratio > 1 after frame 26) is reduced but not removed, so the remaining
residual is in the SPATIAL pattern of the dominant mode — the mechanics' per-cell affine response
to rank-1 active strain — and in the first upstroke frame (ratio 3.9 at frame 4: the tissue starts
moving one frame before any clock here does).

(Scoring defect found and fixed on the way: `s4_score.py`/`s4_residual.py` did not load `delay`,
so round 4 was first scored WITHOUT its delays at 0.64/0.63 — the loss/R² mismatch gave it away.)

## Where this stands (2026-09-10, 02:15)

- S0–S3 closed with the numbers above; S4 ran four live rounds. **Best held-out: R²(A) 0.70,
  R²(u) 0.74, per-cell shortening r 0.88, axis 0.90, on beats the fit never saw.** Replay of the
  fit beat scores 0.98 on the same observable and remains unbeaten, as the plan's rule anticipated
  might happen; the model is 1,900 numbers, replay 161k.
- What may be claimed per cell from this recording: **g and φ** (gate 0.10 and 5°; the fit's g
  correlates only 0.33–0.50 with the amplitude read directly off the recording, because the fit's g
  is the ACTIVE shortening, the recording's amplitude the REALISED one, and E moves between them);
  **timing δ_j** (new, sd 3.3 frames). **Not yet E**: identifiable in the gate at planted spread
  0.3 but fitted to a spread of 2.0 on real data; a λ sweep of the shrinkage prior is the next
  measurement, and a λ that keeps held-out R² while collapsing the spread would make E claimable.
- Open residual (the loop's next edge): 30% of the per-cell affine variance, spatial, in the
  dominant temporal mode; candidates in order of cheapness: (a) a second active axis / active shear
  per cell, (b) anisotropic passive stiffness along the fibre, (c) ν per cell or a 2D-incompressible
  law (the recording is area-preserving at 1.11 expansion/shortening; the model reaches 0.95),
  (d) the 8 first frames (upstroke onset) where the model lags the tissue by a frame.
- HCM sheet: still sealed, not opened.

## To resume

```
cd /workspace/Plexus/prototype/cardio_mpm/strain
export PYTHONPATH=/workspace/Plexus/src; PY=/workspace/.conda_envs/neural-graph-linux/bin/python
$PY rounds_table.py                      # every live round side by side
$PY s3_gate.py table                     # the gate
$PY fit.py --device cuda:0 --target recording --beat 3 --iters 200 \
   --free g,phi,logE,clock,delay --E-shrink 0.1 --tag s4_live_r5_...     # next: the lambda sweep
$PY s4_score.py --params out/fits/<tag>/params.npz --beats 1,2,3 ; $PY s4_residual.py --params ...
```

Seen in `out/figures/fig4_fitted_maps.png`: fitted g is inflated along the right edge — those cells
sit in the prescribed band, so their own g is unconstrained by the loss; band cells should be masked
from the loss and from any per-cell claim (not yet done). The delay map has spatial clusters of
early cells (blue), i.e. timing is coherent over a few cells; the E map is salt-and-pepper.

## The HCM sheet (2026-09-10 afternoon) — seal broken on Cedric's instruction, `data_hcm/SEAL_BREAK.md`

**Which file.** Three claim to track the HCM movie. `hcm_referee.py` (pixels, 400 nodes, rest 110 →
peak 67): T1 `1_HCM…derivatives.npy` r 0.997 / 0.983 with 0.86 px error on 6.8 px of motion;
T2 `Cardio_0/derivatives.npy` identical to T1; T3 `diseased.npy` r 0.30 / −0.01 — broken like
`healthy.npy`. T1 is used. Segmentation with the healthy sheet's detector setting: **434 cells**,
labels aligned to nodes at 100%. Recording: 299 frames, onsets 17/61/121/181/241; beats 1–4 are
clean 65-frame rest→peak→rest windows (beat 0 starts mid-contraction, excluded); peak mean
shortening **0.032** against the healthy 0.021.

**Like-for-like fits** (E fixed at 80, band cells masked from the loss, per-cell g, φ, δ + one
clock, beat 3 fitted, 200 iterations):

| sheet | held-out beats | R²(A) | R²(u) | shortening r | axis | replay R²(A) |
|---|---|---|---|---|---|---|
| healthy (`s4_live_r5_Efixed_mask_delay`) | 1, 2 | 0.55 | 0.57 | 0.85 | 0.86 | 0.98 |
| HCM (`hcm_r1_Efixed_mask_delay`) | 1, 2, 4 | **0.63** | **0.66** | 0.87 | 0.87 | 0.96–0.99 |

Both are lower than the healthy round 4 (0.70/0.74) because E is fixed here; that is the price of
a comparison the stiffness noise cannot contaminate.

**The comparison** (`compare_sheets.py`, interior cells only, n = 1 sheet each — a description of
two specimens, not a statistic; `out/figures/fig5_healthy_vs_hcm.png`):

| | healthy (330 cells) | HCM (299 cells) |
|---|---|---|
| recorded peak shortening per cell, median | 0.025 | 0.040 |
| fitted g (shortening at full activation), median | 0.040 | **0.090** |
| fitted g, p10 / p90 | 0.000 / 0.094 | 0.026 / 0.157 |
| silent cells (g < 0.01) | **19%** | 6% |
| clock delay per cell, sd | 0.104 s | 0.098 s |
| fibre-axis order (0 random, 1 parallel) | 0.09 | 0.20 |
| clock rise / plateau / decay | 0.07 / 0.75 / 0.18 s | 0.08 / 0.82 / 0.14 s |

What the two sheets differ in: the HCM cells contract about twice as hard (g median 0.090 vs
0.040, the recorded strain 1.6×), far fewer of them are silent (6% vs 19%), and their axes are
twice as aligned (order 0.20 vs 0.09). What they share: the timing jitter (0.10 s in both) and the
clock's shape (rise 70–80 ms, plateau 0.75–0.82 s, decay 140–180 ms). Stiffness is deliberately
not compared yet.

## The stiffness sweep and the comparison with E (2026-09-10, evening)

Healthy sheet, beat 3, band masked, g/φ/δ/E + clock free, `--E-shrink λ` on mean((log E − mean)²);
scored on held-out beats 1–2 (`out/fits/s4_live_r6_Eshrink*`). Hard bounds were added to the
optimiser on the way: log E within 10× of 80 and |δ| ≤ 8 frames, plus a NaN guard that restores the
last finite parameters and halves the learning rates (the λ = 0.03 fit had diverged at iteration 90).

| λ | log E sd | E p10 / p50 / p90 | held-out R²(A) | R²(u) |
|---|---|---|---|---|
| E fixed | 0 | 80 | 0.547 | 0.572 |
| 0.3 | **0.40** | 22 / 37 / 60 | 0.637 | 0.658 |
| 0.1 | 0.70 | 15 / 39 / 92 | 0.661 | 0.677 |
| 0.03 | 1.12 | 10 / 44 / 197 | 0.669 | 0.699 |
| 0.01 | 1.49 | 8 / 44 / 365 | 0.673 | 0.722 |
| 0 (round 4) | 1.92 | 4 / 28 / 630 | 0.703 | 0.741 |

Reading: the step from "E fixed" to "any E freedom" is worth 0.09 in R²(A); the further 0.07 from
a spread of ±60% to one of 100× is bought with values (E = 4 to 630) no gate has certified. λ = 0.3
keeps the spread (sd 0.40) inside the regime the S3 gate certified (planted sd 0.3, recovered to
0.40 of spread), so **λ = 0.3 is the setting under which per-cell stiffness is reported**, for both
sheets.

**HCM at λ = 0.3** (`hcm_r2_Eshrink0.3`): held-out beats 1, 2, 4: R²(A) **0.705–0.709**, R²(u)
0.699–0.712, shortening r 0.91, axis 0.90 — up from 0.63 / 0.66 with E fixed.

**Healthy vs HCM with stiffness** (`compare_sheets.py`, `out/figures/fig5_healthy_vs_hcm_withE.png`,
interior cells, one sheet each):

| | healthy (330) | HCM (299) |
|---|---|---|
| fitted g, median (p10 / p90) | 0.056 (0.007 / 0.107) | 0.084 (0.020 / 0.136) |
| silent cells (g < 0.01) | 11% | 7% |
| fitted E, median (p10 / p90), spec units | **37** (22 / 60) | **96** (54 / 141) |
| clock delay sd | 0.098 s | 0.111 s |
| axis order | 0.12 | 0.20 |
| clock rise / plateau / decay | 0.07 / 0.84 / 0.21 s | 0.08 / 0.79 / 0.15 s |

With stiffness free the amplitude gap narrows (g 0.056 vs 0.084 instead of 0.040 vs 0.090) and part
of it moves into E: the HCM cells come out 2.6× stiffer at the median, with the two E
distributions barely overlapping (healthy p90 60, HCM p10 54). g and E trade off in this model, so
"harder and stiffer" is one reading and "the same drive against stiffer neighbours" is the other;
what is robust across the E-fixed and E-free fits is that the HCM sheet strains 1.6× more, has
fewer silent cells, is twice as aligned, and shares the healthy sheet's excitation timing.

## Caveat found 2026-09-10 evening: the particle layout is not innocent

Cedric asked why the particles are not on a grid. They are the stock seed's uniform-random pixels
per cell, and rolling the SAME fitted parameters (λ = 0.3) out on other layouts changes the answer:
held-out R²(A) **0.29 / 0.64 / 0.55 at 30 / 50 / 80 particles per cell**; two layouts predict each
other to R² 0.80 (50 vs 80), 0.35 (30 vs 50). At 50 per cell on a 128 grid there are ~2.6 particles
per grid cell, below what MPM needs (4–8). So every fit so far is conditional on one under-resolved
random layout, and part of what the parameters fitted is that layout's discretisation noise. The
planted gate could not see this: truth and fit shared the layout.

In progress: (1) `layout_convergence.py` — the forward at 50 / 80 / 120 / 200 per cell, random and
regular-lattice placement (`rollout(lattice=True)`, new in model.py), to find the density where the
result stops changing; (2) refits at that density, both sheets, before any number here is quoted.

## PLAN_R2 steps 1–2 (2026-09-10, night)

**Step 1, discretisation.** `layout_convergence.py`: with the λ = 0.3 parameters, random layouts at
80 → 120 → 200 particles per cell agree to R² 0.81 / 0.92 / 0.96 (grid 128; 0.97 at grid 96, 0.94
at grid 160), and the held-out R² settles at **0.55** — the 0.64 obtained on the 50-particle fit
layout was 0.09 of layout noise fitted. Regular lattices are worse (successive-density R² 0.52–0.77,
and negative held-out R² with random-fitted parameters): MPM aliases against its grid when
particles sit regularly. **Setting: random placement, grid 128, 120 particles per cell** (56,640).
**Two-layout floor** (same parameters, 120 vs 121 per cell): R² **0.958** — no fit may be read
closer to 1 than that.

**Step 2, relaxation.** `step_test.py` (clock 0 → 1 → 0, signed strain along each cell's fibre): at
the reference (κ 1e4, drag 30, ρ 1) the sheet unloads to 37% in 2 frames, then **overshoots to −32%
of the plateau and is still 10% off 20 frames later**: the substrate mode rings (damping ratio
k / 2√(κρ) = 0.15, period 31 frames). Neither grid, particle count, substep, E nor density changes
it; drag does: **drag 150 → decay 3 frames, no overshoot, zero by +10 frames**; 200–300 over-damp
(slow tail). κ = 3e4 with drag 200 is also clean but halves the plateau. **Setting: drag 150.**
(The earlier "damping sweep" that saw no effect was run with the sigmoid clock's slow decay, which
hid the ringing.) The old campaign's spec ran drag 30, i.e. rang on every beat.

Refits at 120 per cell, λ = 0.3, mask, delays: drag 150 (`s4_live_r7_p120_d150`) and drag 30
(`s4_live_r7_p120_d30`) to separate the two effects. Scoring must use the same drag and density.

### Steps 1c/2c done, and the diagnosis that reorders steps 3–4

Refits at 120 particles per cell, λ = 0.3, band masked, delays (`s4_live_r7_p120_d30`, `_d150`):
held-out R²(A) / R²(u) **0.65 / 0.66 at drag 30, 0.69 / 0.66 at drag 150**. The converged refit
lands where the layout test said (0.55 → 0.65 once refitted on the converged layout; +0.04 from
damping). Two-layout floor 0.958.

`rank_ceiling.py` (the recording's own SVD on beat 3, patterns kept on beats 1–2): **one spatial
pattern with one fixed time course predicts held-out beats at R² 0.87; two patterns 0.94; three
0.97.** So 0.9 is reachable by a one-clock model in principle, and what the mechanical model lacks
is SPATIAL: its dominant pattern correlates 0.91 with the recording's (time course 0.99), and 24% of
the model's own motion lies outside the recording's top-3 spatial modes — the mechanics produce
cell-to-cell structure the tissue does not have. By phase, 70% of the residual sits in the plateau
(90% of the signal); the relaxation phase holds 9% (its residual/signal > 1 was a division by a
vanishing signal, not a mechanics defect — PLAN_R2 step 2's exit criterion was mis-specified).
Consequence: step 3 (per-cell relaxation) is worth ≤ 0.07; **step 4 (the spatial pattern) is worth
up to 0.18** and goes first. Poisson ratio is not the lever (forward sweep ν 0.2 / 0.3 / 0.4 / 0.45:
0.695 / 0.692 / 0.628 / 0.495). Running: a second active axis per cell (g2, transverse strain;
`s4_live_r8_p120_d150_g2`) and its planted gate (`gate_g2_s0`).

### Step 4c — a second active axis (2026-09-11)

`model.Params.g2`: active strain ACROSS the fibre, F_a = I + γ (g f fᵀ + g2 f⊥ f⊥ᵀ), one number per
cell, zero = the rank-1 model. Fit `s4_live_r8_p120_d150_g2` (120/cell, drag 150, λ = 0.3, mask,
delays): **held-out R²(A) 0.82 / R²(u) 0.87, shortening r 0.92, axis 0.94** (from 0.69 / 0.66).
The fitted g2 is **−0.9 g at the median** (negative = thickening): the cells thicken across the fibre
by about as much as they shorten along it, i.e. they are area-preserving at the cell level — the
model now reaches what the recording showed from the start (expansion/shortening 1.11). A
scoring defect was found again on the way (the scorers did not load g2; first score 0.45 / −0.43).

The planted gate for g2 with a POSITIVE planted g2 (`gate_g2_s0`) degraded every family (g 0.24,
φ 19°): a positive transverse strain makes the active tensor nearly isotropic, the axis undefined,
and (g, g2, φ) ↔ (g2, g, φ + 90°) a symmetry the recovery metric does not know. Re-planted with the
sign the data has (g2 = −0.9 g, `gate_g2neg_s0`), running. HCM with the second axis
(`hcm_r3_p120_d150_g2`), running.

## Stopped 2026-09-11 (Cedric's call) — what is still running and where to pick up

Background jobs left running (they finish on their own, ~1 h): `gate_g2neg_s0` (planted gate for the
second axis with the observed sign), `hcm_r3_p120_d150_g2` → its scoring, then `gate_g2neg_tau_s0`;
on the other GPU `s4_live_r9_p120_d150_g2_tau` (per-cell relaxation time, step 3a) → its scoring.
Read them with `python rounds_table.py`, `out/*.score.log`, and `out/fits/gate_*/fit.json`
(`log[-1].recovery`). Best model so far: `out/fits/s4_live_r8_p120_d150_g2` — held-out R²(A) 0.82,
R²(u) 0.87, 120 particles/cell, drag 150, λ = 0.3, band masked, per-cell g, g2, φ, E, δ + one clock;
movie `out/movies/s4_live_r8_p120_d150_g2_beat1.mp4`. Score anything with
`s4_score.py --per-parent 120 --drag 150`. Next per PLAN_R2: read the g2 and τ gates, HCM with g2
(`compare_sheets.py --healthy out/fits/s4_live_r8_p120_d150_g2 --hcm out/fits/hcm_r3_p120_d150_g2`),
then step 5 (seeds).

## 2026-09-11, resumed — gates, per-cell relaxation, HCM with the second axis, and a retraction

**Gates (planted, rest noise, 120/cell, drag 150, seed 0).** With g2 in the truth (−0.9 g) and free:
g **0.32** (r 0.68), g2 **0.42** (r 0.53), φ 12°, log E **0.64** — g and g2 ship under the 0.5 rule,
but the two are entangled (rank-1 gate had g at 0.10 and φ at 5°), and **per-cell E no longer
ships** once the second axis is in the model. With a planted per-cell relaxation time as well:
log τ **0.80** — not recoverable; g/g2/φ unchanged.

**Per-cell relaxation (step 3a, `s4_live_r9_p120_d150_g2_tau`):** held-out R²(A) **0.83**, R²(u)
**0.90** (r8: 0.82 / 0.87). Worth +0.01–0.03 and not identifiable per cell → τ stays SHARED in the
reporting model; the small gain is left on the table.

**HCM with the second axis (`hcm_r3_p120_d150_g2`):** held-out beats 1, 2, 4: **R²(A) 0.85, R²(u)
0.87–0.89, shortening r 0.95, axis 0.94** — the best fit of the project, on the diseased sheet.

**Healthy vs HCM, both with g2** (`compare_sheets.py`, `fig5_healthy_vs_hcm_withE_g2.png`):

| | healthy (331) | HCM (298) |
|---|---|---|
| fitted g (along the fibre), median | 0.033 | **0.062** |
| fitted g2 (across; < 0 = thickening), median | −0.041 | −0.042 |
| g2 / g over active cells, median | **−0.91** | **−0.61** |
| silent cells (g < 0.01) | 26% | 13% |
| axis order | 0.09 | 0.18 |
| clock delay sd | 0.089 s | 0.084 s |
| fitted E, median (p10 / p90) | 207 (178 / 246) | 226 (200 / 261) |

**Retraction.** With the rank-1 active strain the sheets came out at E 37 vs 96 ("HCM 2.6× stiffer").
With the second axis the two E distributions coincide (207 vs 226), and the E gate fails. The
stiffness difference was the rank-1 model's way of producing transverse thickening it could not
express; it is withdrawn. What stands, and is stronger than before: HCM cells shorten about **twice
as hard** along the fibre with the SAME absolute thickening across it, so they **lose area** during
contraction where healthy cells conserve it (g2/g −0.61 vs −0.91); half as many are silent; their
axes are twice as aligned locally; the excitation clock is the same. Absolute E is the same for both
sheets and set by the substrate spring, not by the tissue.

Running (step 5): second seeds for both sheets (`*_seed1`, optimiser seed 1 and layout 121/cell).

### Step 5 — seeds (2026-09-11)

Same fit, optimiser seed 1 and layout 121/cell (`*_seed1`): held-out R² within 0.005 of seed 0 on
both sheets. Per-cell agreement between the two fits (`fig_seeds.py`, `fig7_seed_agreement.png`):
g r 0.94 / 0.96, g2 r 0.93 / 0.95, delay r 0.88 / 0.87, axis 6.7° / 4.9° (healthy / HCM); median
cell-to-cell difference ~0.15 of the map's spread. **That is the error bar on every map.** All
figures regenerated from the second-axis models (`figures.py BEST`, `fig_alignment.py`,
`compare_sheets.py`); the reporting model is `s4_live_r8_p120_d150_g2` / `hcm_r3_p120_d150_g2`.

Toward 0.95 (Cedric's target): per-cell excitation time course (delay, rise, plateau, decay per
cell, shrunk to the shared clock; `model.Params.logtr/logdur`) — `s4_live_r10_cellclock`,
`hcm_r4_cellclock` running. Rank ceiling: 0.94 with two temporal modes, 0.97 with three; the
discretisation floor is 0.958 at 120/cell, so 0.95 also needs the density raised (200/cell ≈ 60 GB
a fit: alone on a GPU with a smaller grid, or a two-half rollout).

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

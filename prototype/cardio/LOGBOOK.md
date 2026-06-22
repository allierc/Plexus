# Cardio prototype — learning logbook

Scientific log of building the Plexus synthetic analogue of the real cardiomyocyte
trajectory data. Every entry: **hypothesis → test → verdict**. Negative results are
results. Mirrors the methodology of `prototype/graph_trainer/runs/logbook2.md`.

Data: `cardio_real.npz` (real Utrecht field, 137²=18769 tracked nodes, 238 frames,
dt=0.0417 s). Model: `cardio_stage2.py` + `specs/*.yaml`, archived to `archive/<step>_<name>/`.

---

## E1 — Is the real contraction a propagating wave?
**Hypothesis:** the sheet shows a traveling activation wave (motivating a Nagumo
reaction–diffusion model with a localized trigger).
**Test:** per-node motion-onset time within a beat vs node position
(`cardio_real.npz`); a wave ⇒ strong onset-vs-position gradient.
**Result:** onset-vs-position r ≈ 0 (x +0.01, y +0.09); onset std ~2.7 frames.
**Verdict: FALSIFIED.** The contraction is **synchronous** — the whole field beats
together; any AP/Ca wave is faster than one 42 ms frame. ⇒ model must use
*synchronous* activation, not a slow wave.

## E2 — Units and rhythm of the real data
**Test:** beat detection + spacing on mean speed.
**Result:** dt = 0.0417 s/frame, 9.92 s total; ~6 beats, period ~1.8 s ≈ **0.55 Hz**.
Space: 15 px grid in a 2048 px image (≈0.0073 field-units/node); **physical µm/px is
NOT in the dataset**. Real displacement: mean ~1.7 px, max ~15 px (≈0.0073 field).

## E3 — Active/silent ratio of the real beat
**Test:** threshold mean-speed trace; measure active vs silent fractions.
**Result:** **45% active / 55% silent**, ratio ~1:1.2; beat ≈0.87 s, silent gap ≈1.17 s.
**Verdict:** the target is a brief twitch with a comparable rest — the model must
return to rest between beats (a real silent period).

## E4 — Conduction velocity of the Nagumo operator (sanity check)
**Test:** launch a planar wave, fit wavefront position vs time (`cardio_stage1.py`).
**Result:** ~0.07 node/tick at D=2 (∝√D); a front needs ~1400 ticks to cross the
137-node sheet. Coverage peaks at only ~12–21% excited (a narrow band); D≥20 the
pulse diffuses out and dies.
**Verdict:** the Nagumo signal *does* propagate at a definite speed, but it is **slow
relative to any reasonable pacing period** — confirming E1's conclusion that a
sweeping wave is the wrong analogue here.

## E5 — Model v1: localized trigger + slow wave
**Test:** `coupled4.yaml` (100²/137², corner trigger, D=2, 4 paced pulses).
**Result:** wave covers only a corner before the next pulse; most nodes never move;
cross-cycle similarity 0.04–0.24 (NOT similar).
**Verdict: FALSIFIED** as a data analogue. Archived `1_coupled4` as the wave
sanity-check only.

## E6 — Synchronous `pulse_field` operator
**Hypothesis (from E1):** stimulating ALL nodes together each beat (a global "pulse
field"), while keeping `grid_graph` coupling so nodes stay interconnected, gives
synchronous beats.
**Test:** add `pulse_field` op; `coupled_sync.yaml`, 137², 6 paced global pulses.
**Result:** max_excited = 1.0 (all fire together), 6 clean cycles, cross-cycle
similarity **0.87**.
**Verdict: VALIDATED.** Synchronous activation reproduces the repeated-beat structure.
Archived `2_coupled_sync`.

## E7 — The missing silent period (nodes don't return to rest)
**Observation (user):** the tissue looks continuously active — no silent period.
**Test:** displacement-from-rest trace per cycle.
**Result:** disp oscillates between ~0.0018 and ~0.0035 — it **never returns to
baseline** after beat 1. Cause: heterogeneous contraction makes a net (rigid-mode)
drift that only the weak anchor corrects, with timescale γ/k_anchor ≈ 167 ticks
> period (180). Also the AP (eps=0.08) fills 62% of the cycle vs real 45%.
**Verdict:** two coupled problems — slow mechanical relaxation + too-long AP.

## E8 — Boundary pin (REJECTED)
**Hypothesis:** pin the attached edges to rest so the interior springs back.
**Test:** clamp the boundary ring to X each substep.
**Result:** disp returns closer to rest (~0.0007), but amplitude collapses.
**Verdict: REJECTED by user** — "the nodes should relax to initial positions on their
own." Removed the pin; relaxation must come from the mechanics, not an external BC.

## E9 — Short pulse + self-relaxation (parameter sweep)
**Hypothesis:** a shorter action potential (higher eps) + faster overdamped mechanics
(lower γ, modest k_anchor) lets each node self-relax to rest between beats.
**Test:** sweep eps × k_anchor on a fast 60² grid; measure AP%, peak disp, and
rest/peak (return-to-rest).
**Result:**
| eps | AP% | peak disp | rest/peak |
|-----|-----|-----------|-----------|
| 0.15 | 38% | 0.0083 | 0.02 |
| 0.30 | 23% | 0.0072 | 0.03 |
| 0.50 | 17% | 0.0063 | 0.03 |
(k_anchor=0.1, γ=0.3). All self-relax (rest/peak ≤ 0.03) once γ=0.3 and the AP is short.
**Verdict: VALIDATED.** Chose **eps=0.3** (short pulse, AP 23% → long silent period),
k_anchor=0.1, γ=0.3, β=0.4 — peak disp 0.0072 ≈ the **real max 0.0073**.

## E10 — Full 137² synchronous model with self-relaxation
**Test:** `coupled_sync.yaml` with E9 params, 137², 6 cycles.
**Result:** synchronous (max_excited 1.0), self-relaxing twitches, **cross-cycle
similarity 0.999**, peak disp 0.0155, displacement returns to rest each cycle.
Trajectory PNG = crisp closed loops, repeated across cycles, structured by the
wavelength-10 heterogeneity. Archived `2_coupled_sync`.
**Verdict: VALIDATED** — clean synthetic analogue of the real 4–6-cycle trajectory
data: synchronous, short pulse, silent period, similar curved-string loops.

## E11 — Breaking symmetry via the mechanical-property maps (5 variants)
**Goal:** the baseline `2_coupled_sync` loops were too self-similar (mostly vertical
ellipses). Vary the *smooth* property maps over space (deterministic, no randomness)
and read the trajectory result of each to understand the mechano-signaling map.
Made the heterogeneity spec-driven (`patterns:` block: per-property anisotropic
wavelength `[wx,wy]`, orientation `angle`, `phase`; fibre `mode` = smooth/swirl/radial)
and added a `<name>_properties.png` panel to every archive.

| step | variant | property recipe | trajectory result |
|------|---------|-----------------|-------------------|
| 3 | `p1_aniso` | stiffness stripes [8,26], gain stripes [26,8], fibre smooth | loops vary in orientation & size cell-to-cell; clear break |
| 4 | `p2_diag` | all patterns rotated 45° | diagonal **bands** where loops shrink to dots (low gain) then grow |
| 5 | `p3_swirl` | fibre = tangential swirl about centre | loop orientations **circulate around the centre**; motion collapses at the swirl defect |
| 6 | `p4_multiscale` | coarse stiffness (wl 30), fine gain (wl 6), fibre amp 2π | fast amplitude variation over slow orientation rotation (**scale separation**) |
| 7 | `p5_crossgrad` | stiffness ∥ vs gain ⟂, off-centre swirl | elongated diagonal strokes circulating an off-centre defect; strongest break |

All five stayed synchronous (max_excited 1.0), self-relaxing, cross-cycle similarity ~0.999.

**What the mechano-signaling system does (the readout):**
- **fibre-angle field → loop ORIENTATION field.** Each beat loop's major axis follows
  the local fibre direction; the fibre field's *topology* (stripes, swirls, defects)
  appears directly in the trajectory-orientation pattern.
- **gain field → loop AMPLITUDE.** High gain = large loops; low gain = collapse to a
  dot. Gain bands ⇒ amplitude bands.
- **stiffness field → amplitude (inverse) + roundness.** Stiffer ⇒ smaller, flatter strokes.
- **swirl/defect in the fibre field ⇒ a low-motion "hole"** with circulating orientations
  around it.
- **scale separation works:** fine gain modulates amplitude locally while a coarse fibre
  field rotates orientation globally.

**Verdict:** symmetry is broken cleanly and *interpretably* by the smooth property maps
alone — the trajectory texture is a direct, legible function of (gain, stiffness, fibre).

## E12 — Anchor the boundary with the real data (real = blue, synthetic = green)
**Goal:** ground the model in reality — drive the edge ring with the measured tissue
and compare predicted interior vs real.
**Method:** new `boundary_data` operator (Dirichlet BC): loads `cardio_real.npz`
(137², node ordering verified = y·137+x), resamples to the sim ticks by linear
time-stretch, and sets the boundary node positions each tick (runs LAST in the
schedule). Renderers gained a blue real-data overlay (`real=...`); synthetic stays green.
**Test:** `coupled_anchored.yaml` (step 8) = the synchronous self-relaxing model + p1
property maps + `boundary_data`.
**Result:** max disp 0.0059 (≈ real 0.0073; the pinned edges constrain the interior),
6 cycles, cross-cycle similarity 0.82 (real boundary isn't perfectly periodic across the
6 mapped cycles). Green-vs-blue PNG: loops co-located and same scale; **synthetic loops
are cleaner/simpler, real loops larger and more irregular** — a direct per-node residual.
**Verdict: VALIDATED as a comparison harness.** The blue/green overlay is the diagnostic
for the inverse fit: tune (gain, stiffness, fibre, eps, β) until green matches blue.

## E13 — Learnable, decomposed operators validated vs the forward engine
**Goal (INVERSE_TRAINING step 1):** one differentiable nn.Module per operator, each
validated exact against `cardio_stage2`. **Result** (`cardio_operators.py`): nagumo,
signal_to_force, pulse_field match to 0.0; mpm_mechanics to 6e-7. Learnables inventory:
FHN (D,a,b,eps), signal (theta,eta), mechanics (beta,k_anchor,gamma,aniso); **pulse_field
is GIVEN (a known experimental input), not learned**. Per-node maps (stiffness,gain,fibre)
optional/later. **Verdict: VALIDATED** — a fit on these ops is a fit on the real dynamics.

## E14 — One-step MPM parameter recovery (exact-recovery sanity)
`cardio_fit.py`: from (pos,act)->pos_next transitions of the TRUE forward mechanics,
recover (beta,k_anchor,gamma,aniso) by one-step MSE. **Recovered to ~1e-8** (loss 1e-18).
The gradient inverse on the learnable operator is sound. Archived `train_01_mpm_recovery`.

## E15 — Differentiable ROLLOUT recovery + GT-vs-stages PDF
`cardio_inverse.py`: roll the full loop (pulse GIVEN -> nagumo -> signal -> mpm) with TRUE
mechanics -> GT trajectory + activation; re-roll a differentiable LearnableMpmMechanics from
a WRONG init and backprop through the whole rollout. **All params recovered (max err 4e-3,
loss 2.5e-10).** PDF `inverse_gt_vs_stages.pdf`: page 1 = GT/init/mid/recovered trajectories
over time (recovered green tracks GT blue; init red is off); page 2 = reconstructed params
(bars GT vs stage) + effective contraction/anisotropy maps per stage. **SYNTHETIC GT, not
real data** (validates the machinery first). Archived `train_02_rollout_recovery`.

## E16 — UNet: microscope image -> mechanical tensor (amortized prior)
`cardio_unet.py`: instead of free-fitting 3·N per-node values, a small UNet maps the
microscope frame -> (stiffness, gain, fibre) maps, trained end-to-end through the forward
mechanics against a GT trajectory (maps derived from the image for the sanity).
**Result:** traj MSE 6e-6 -> 5e-8; **stiffness & gain recovered (MAE ~0.045 over a 1.2
range); trajectory matches.** Fibre NOT recovered (MAE 0.82) — a genuine identifiability
fact: anisotropy uses (edge·fibre)², so fibre and fibre+π are mechanically identical (the
trajectory cannot distinguish them; measure orientation mod π). Archived `train_03_unet_mechanical`.
**Verdict: VALIDATED** image-conditioned pipeline; next is training UNet(real image) against
the real anchored trajectory.

## E17 — Fit the model to the REAL cardiomyocyte beat (toward real data)
`cardio_real_fit.py`: target is no longer synthetic. Build a **canonical averaged beat**
from the real trajectory (the ~4–6 near-identical beats aligned + averaged; displacement
time-course, max disp ~0.0067 ≈ the real max), downsample to a grid, and train
**UNet(real microscope frame) → (stiffness,gain,fibre) + global scalars (beta,k_anchor,
gamma)** through the forward single-beat rollout (pulse+nagumo GIVEN, synchronous) to
match the real beat by per-node displacement MSE. **Result:** beat MSE 4.9e-5 → 1.1e-6
(~44×); recovered scalars beta≈0.28, k_anchor≈0.19, gamma≈0.39; the UNet infers
mechanical maps directly from the real image. Archived `train_04_real_fit` (real-vs-fit
PDF + png + mp4, amp×20, 64² grid). **Verdict:** the real-data inverse runs end-to-end and
partially reproduces the measured beat; the fit is imperfect (real motion is richer than
the model — model mismatch + partial observation), which is the expected starting point.

## E18 — Open up the learned loops (amplitude/shape loss; drop substrate anchor)
**Problem:** train_04 collapsed to tiny green dots — per-node position MSE is gamed by
predicting near-zero motion (the model can't reproduce the real's large coherent flow,
so hedging small wins). NOT a model defect: the forward runs make big open loops with
the SAME overdamped mechanics.
**Two fixes, model unchanged** (`cardio_train05.py`):
1. **Dropped the substrate-anchor spring** `k_anchor` (fixed ~0, not fitted). It was a
   crutch from when the sheet was free; now the boundary is anchored to real data, so the
   passive edge springs + fixed boundary already restore the interior to rest. (Distinct
   from the *real-data boundary BC* — different "anchors": one is a force, one is a
   Dirichlet condition on the outline band.)
2. **Amplitude/shape loss:** `MSE + 8·(per-node |disp| mismatch) + 3·(per-frame increment
   mismatch)` — forbids the tiny-motion minimum and matches loop shape.
**Result:** predicted |disp| 0.0013 = real 0.0013 (was ~0); the rendered green loops
**open up to ~real amplitude** (vs train_04 dots). Scalars: beta 0.26, gamma 0.41, aniso
1.05. Archived `train_05_amplitude` (amp_{trajectories.png,mp4,nodes.mp4,unet.png,properties.png}).
**Verdict: VALIDATED.** Amplitude now matches; loop *orientation/shape* is the remaining gap.

**Identifiability note (substrate adhesion):** `k_anchor` could be fitted, but only
GLOBAL — a per-particle adhesion map is degenerate with the gain & stiffness maps (the
trajectory can't separate "stuck by adhesion" from "low contractility / high stiffness").
Tissue *stiffness* is the per-particle elastic field (already a UNet map); substrate
adhesion stays global or dropped.

## E19 — Shape-weighting and the model ceiling (train_06, train_07)
**train_06** (`l_vel`↑ to 30, `l_amp`↓ to 5): the velocity/shape term **competes with
amplitude** — predicted |disp| dropped 0.0013→0.0008, interior loops shrank. So the
amplitude term is what opens loops; train_05's balance (`l_amp=8, l_vel=3`) wins.
**train_07** (`cardio_train_free.py`): **FREE per-node maps** (12k params, no UNet) +
single strongest real beat. Even at maximum flexibility, base MSE plateaus ~1.5e-6 —
**barely better than the UNet (1.75e-6)**, and the per-node error stays ~the signal
magnitude. Amplitude matches (|disp| 0.0012≈0.0013) but the loop **orientations don't**.
**Verdict — MODEL CEILING, not a parameterization/UNet bottleneck.** Synchronous uniform
activation + smooth heterogeneity + overdamped mechanics reproduces the *type* and
*amplitude* of the beat (open loops) but **cannot generate the real's coherent flow
DIRECTIONS** — those need non-local/directional drive (a propagating activation phase,
correct fibre architecture, or more boundary influence), which this (deliberately fixed)
model lacks. Also learned: the **canonical AVERAGED beat under-represents amplitude**
(phase jitter shrinks it) — use the single strongest beat as the target.

## E20 — Drift fix + loss-term sweep
**Drift (user-spotted):** dropping `k_anchor` (E18) caused slow drift over the 6 cycles —
the boundary BC restores *shape* but not *absolute* position, so per-beat hysteresis
residuals accumulate. **Restored `k_anchor=0.06`** (small substrate spring) → loops close,
no drift; the amplitude loss raises β to keep them open. So `k_anchor` is NOT redundant:
boundary fixes shape, spring fixes drift (two different "anchors").
**Sweep** (`cardio_sweep.py`, 14 weighted recipes of normalized terms MSE/amplitude/
loop-area/total-length/shape, grid 48²):
- **`amp+length` wins** for open loops (area-error 0.44, best). The **`area` (open-loop)
  term is useless/harmful** — loop area is too noisy a target; loops open best as a
  *side-effect* of matching amplitude+length.
- **Hard openness-vs-direction trade-off:** `amp+length` → open but wrong direction
  (shape 2.0); `shape`/`amp+shape` → better direction (shape 1.1–1.2) but closes the loops.
  Consistent with the E19 model ceiling.

## E21 — Phase-lag map φ(x,y): the conduction graph is INERT under synchrony
**Key realisation:** in `excitable_nagumo`, `lap = Σ(u_j−u_i)/deg` over the **binary
8-neighbour** grid_graph ⇒ **A_ij = 1 (isotropic)**, and under *synchronous* activation u
is uniform ⇒ the Laplacian ≈ 0 ⇒ **A_ij does nothing**. So directional organization needs
a u-gradient, i.e. non-synchronous activation. **train_08_phase** (`cardio_train08_phase.py`,
now **spec-driven** via `specs/train_phase.yaml`): add one smooth interpretable map — a
per-node activation **lag φ(x,y)** (learnable coarse grid → upsampled, ~10-node wavelength),
fit jointly with the UNet maps + scalars.
**Result:** φ learned a smooth ±6-tick field; `u_traces.png` confirms the activation is
cleanly staggered (the mechanism works). **But shape stayed 1.98 ≈ synchronous** — because
the loss was amp+length (shape weight 0), the optimiser had **no incentive** to use φ for
direction. **Verdict:** the phase DOF is real and works, but a *free* φ under a
direction-agnostic loss won't fix orientation. To exploit it: add the shape term to the
loss, OR give φ a **structured origin** — anisotropic conduction A_ij(fibre/image) that
*generates* directional phase (train_09/10). Archived `train_08_phase`.

## E22 — Full-dataset cycle-batched RMSE + phase (train_08_03)
Switched from fitting one beat to the **whole dataset**: partition the real sequence into
its individual cycles (`real_all_beats`) and minibatch over cycles; loss = plain **RMSE**
(no open/length/shape composite). Model = UNet maps + phase φ + scalars + spring
(`k_anchor=0.06`). **Result:** all-cycle RMSE 0.001→**0.00082**; φ grew to **±17 ticks**;
the green(true)/blue(learned) loops **largely co-locate** across the field — clearly the
best match so far (vs the misaligned 8.1). The `true_vs_learned.mp4` tiles
**beat→rest→beat**, so the **quiescent is respected** (both at rest between beats, aligned).
Also fixed two viz issues: u-traces now plot **raw FHN u** (shows the recovery undershoot),
and the comparison color convention is inverted to **green=true, blue=learned** everywhere.
Archived `train_08_03`. **Open:** orientation still imperfect on some nodes; geometric
`orient` metric + global params (eps/theta/eta/gamma) or structured A_ij(image) next.

---

## Open / next
- **Match active/silent to real (45%)**: eps=0.3 gives 23% (deliberately short). eps≈0.12–0.15 → ~38–45% if a longer beat is wanted.
- **Anchor the boundary with the real data** (node ordering already verified to match: index = y·137+x): drive the sim boundary with `cardio_real.npz` boundary trajectory as a time-dependent BC, then compare the predicted interior to the real interior.
- **Inverse fit** (`INVERSE_TRAINING.md`): recover eps/D/β/gains from the real field; partition over the ~6 beats for statistics (learn the per-beat average).
- **Quantitative match**: compare synthetic vs real per-node loop shape / displacement-direction field, not just amplitude.

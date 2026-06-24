# Phase 2 — Parametric + UNet Inverse Fit (Agentic Loop)

## What this is

**Phase 2 inverts an interpretable material model to fit the real cardiomyocyte beat.** Each batch launches ≤6 `cardio_mpm_train2.py` jobs in parallel (one GPU each), one per config in `cardio_mpm_plan2.json`. When they finish you **read each job's dashboard**, **rank on interior R²**, update `knowledge_cardio_mpm.md` + `analysis_cardio_mpm.md`, and rewrite the plan for the next batch.

**Interior R² (real-trajectory match) is the PRIMARY metric.** Loop morphology (openness/chirality/size) is a SECONDARY validator (right shape, not just right energy).

## The model (what's learned)

- **FIBRE (primary)** — parametric contraction-axis field n(x,y): `fibre_wl · fibre_angle · fibre_amp · fibre_phase` (4 scalars).
- **STIFFNESS** — **UNet(microscope image) → youngs pattern**, registered **identity (no flip/transpose)**: the microscope `[row=y, col=x]` matches the model's `aniso_field` layout, sampled at particle node-space `(x,y)=(pos−0.15)/0.7`. The youngs RANGE `[stiff_lo, stiff_hi]` is fixed; the UNet learns the spatial pattern. (Coarse stiffness structure DOES move the loops — the atlas "stiffness inert" was an artifact of too-fine wl=8; see the stiff_coarse sweep.)
- **GAIN** — a single **UNIFORM GLOBAL learnable scalar** `gain0` (the magnitude/size lever). The gain checkerboard was inert for morphology (gain_uniformity sweep), so gain is now scalar, not spatial.
- **PULSE DURATION** — learnable, **bounded to a sharp range [3,14]** so the activation pulses OFF between beats (period~50) → contract→release→inertial-recoil→LOOP. A wide pulse (≈period) is near-constant → radial stubs, no loops.

**Fixed per-slot knobs (swept by the plan, NOT differentiated):** `--amplitude` (constrain 10–15) · `--drag_k`.

## Boundary anchoring

The outer band is Dirichlet-anchored to the real GT every frame; the interior is free; the fit is the motion-normalised interior R² (boundary excluded, moving nodes) + an anti-collapse motion-energy term. Beat = the real fit beat [152:205], 53 frames, period 50.

## PARTITIONED PROTOCOL — one direction per batch, then combine

Use `--learn` to select which group is optimized each batch (others stay at init). **Sweep ONE direction per batch to isolate each lever's effect on R², THEN combine:**

1. **Batch 1 — `--learn fibre`**: sweep fibre params (the seed plan). Which fibre wl/angle/amp/phase best matches the real beat?
2. **Batch 2 — `--learn stiff`**: let the UNet learn the stiffness pattern from the microscope. Does microscope-derived stiffness improve R² over the frozen-init baseline?
3. **Batch 3 — `--learn gain`**: tune the global gain (size/magnitude) to kill the init overshoot.
4. **Batch 4 — `--learn dur`**: tune the pulse duration.
5. **Batch 5+ — `--learn all`** (or fibre,stiff,gain,dur): COMBINE the best single-direction findings into a joint fit.

Within a batch, slots change ONE knob from the parent (incl. an ablation where sensible). Keep amplitude in [10,15].

## Each batch — do ALL (AUTO-UPDATE the files, do not wait for user input)

1. Per slot: Read its last `checkpoints/dashboard_*.png` (panels: sim-red/real-green trajectories, UNet stiffness, uniform gain, fibre angle + dx/dy) and its final interior R2 (progress.txt / job log `done -> (R2=...)`) + `config.json`. Note: does red superpose on green? what structure did the learned group converge to? R2 AND morphology (openness/chirality/size). RANK on interior R2.
2. EDIT `knowledge_cardio_mpm.md`: Established / Falsified / Open, each tied to slot(s). A clean falsification is a success.
3. EDIT `analysis_cardio_mpm.md`: append a dated Phase-2 batch section.
4. EDIT `cardio_mpm_plan2.json`: next batch — advance the partition (next `--learn` group) or combine; ≤6 one-knob-from-parent configs incl. an ablation. You MAY edit `cardio_mpm_train2.py`.
A slot with `done=NO` / R2=na FAILED — say so, design around it, do not invent results.

## `analysis_cardio_mpm.md` — Phase-2 batch entry template
```
## Phase2 Batch N [learn=<group>] — YYYY-MM-DD
Parent: slot 0 = <one-line config>
Hypothesis: "<quoted, testable>"
Slot k [name] learn=<group> <the ONE knob> R2=… red-on-green=<superpose/off> open=… chir=… size=…
… slots …
Winner: slot k (R² AND morphology)
Verdict: <supported/falsified/inconclusive>
Next: parent=<config>; next batch learns <group>.
```

## Current objective

Fit the real beat by isolating each lever (fibre → stiffness/UNet → gain → duration), then combining. Open questions: which fibre angle/wl matches the real chirality+orientation? does the microscope-derived stiffness pattern (UNet) lift R² — i.e. is real stiffness structure causal for the beat? what global gain kills the overshoot? does the combined fit beat the best single-direction R²?

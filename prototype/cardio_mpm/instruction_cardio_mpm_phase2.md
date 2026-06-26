# Phase 2 — Parametric + UNet Inverse Fit (Agentic Loop)

## What this is

**Phase 2 inverts an interpretable material model to fit the real cardiomyocyte beat.** Each batch launches ≤6 `cardio_mpm_train2.py` jobs in parallel (one GPU each), one per config in `cardio_mpm_plan2.json`. When they finish you **read each job's dashboard**, **rank on interior R²**, update `knowledge_cardio_mpm.md` + `analysis_cardio_mpm.md`, and rewrite the plan for the next batch.

**Interior R² (real-trajectory match) is the PRIMARY metric.** Loop morphology (openness/chirality/size) is a SECONDARY validator (right shape, not just right energy).

## The model (what's learned)

- **FIBRE (primary)** — parametric contraction-axis field n(x,y): `fibre_wl · fibre_angle · fibre_amp · fibre_phase` (4 scalars). **Option `--unet_fibre 1`**: the UNet ALSO predicts a bounded angle deviation `dθ(x,y)` (±`--fibre_dev`, default π/2) ADDED on top of the parametric θ — microscope spatial detail over the smooth parametric base. The deviation is part of the UNet (the `stiff` learn-group), so it co-learns when `--learn` includes `stiff`. **Option `--siren_fibre 1` (PREFERRED over `--unet_fibre`)**: a FREE, image-INDEPENDENT SIREN field f(x,y) supplies the bounded dθ deviation instead — its params are in the `fibre` learn-group (co-learns with `--learn fibre`). Use this to test whether a free direction field (not tied to the microscope) lifts R².
- **STIFFNESS** — youngs pattern over `[stiff_lo, stiff_hi]` (range fixed; the field learns the spatial pattern). Two sources via **`--stiff_src`**: `unet` (default, legacy) = **UNet(microscope image)**, registered identity (microscope `[row=y, col=x]` matches `aniso_field`, sampled at node-space `(x,y)=(pos−0.15)/0.7`); `siren` = a **FREE, image-INDEPENDENT SIREN field f(x,y)**. **Stiffness is the SPATIAL loop-AMPLITUDE lever** (softer→bigger local loop; stiffer→smaller) — the per-region complement to the GLOBAL scalar gain. NB the prior "spatial stiffness/fibre is net-harmful" results (Falsified#8/#9) ALL used `UNet(microscope)`, so they falsify *image-shaped* fields, NOT free fields — `--stiff_src siren` / `--siren_fibre 1` are a genuinely new test, not a re-run.
- **SIREN fields (`--stiff_src siren`, `--siren_fibre 1`)** — image-independent coordinate net; **`--siren_omega`** (default 30) is the frequency/bandwidth knob (LOWER=smoother — the smoothness prior replacing the image constraint; sweep it). `--siren_hidden` (256), `--siren_layers` (3). Dashboard top-right shows the learned fibre dθ; top-middle shows the youngs field. Watch for noisy-overshoot → lower omega.
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

## METHODOLOGICAL RULE — knowledge is CONDITIONAL (no unconditional `FALSIFIED` / `CLOSED`)

**Every validation or falsification is conditional on the experimental REGIME, of which optimization depth is one
important component.** The regime is the whole experimental context — mechanism (force ↔ active-stress), dur (low ↔
high), fibre base (wl40 ↔ wl28), fibre handling (frozen ↔ learned), amplitude, parent config, AND iteration count
(300 → 600 → 1200 …). Every one of these has already changed a conclusion. A verdict is therefore a statement
*"X behaves this way IN this regime"*, never an absolute. This is not optional bookkeeping — it is a core rule of how
this agent reasons:

1. **Always tag provenance.** When you write a verdict, annotate the regime that produced it: `FALSIFIED@600it`,
   `CLOSED@300it/wl28-base`, `exhausted@active-stress/dur30`. A bare `FALSIFIED` / `CLOSED` is incomplete — do not write one.
2. **Verdicts are RE-OPENABLE when the regime moves.** When any controlling variable changes, prior verdicts from the
   old regime become **PROVISIONAL** and must be re-examined, not trusted. Precedent: at 300–600 iter the 4-scalar fit
   looked "plateaued / at its expressiveness limit" and fibre co-learning looked harmful; at 1200+ iter the plateau
   dissolved (−2.62→−1.41) and the co-learn verdict FLIPPED. Treat that as the normal failure mode of regime-bound
   conclusions, not a surprise.
3. **A clean OVERTURN in a changed regime is a SUCCESS**, exactly like a clean falsification — report it as a
   first-class result, and move the old verdict to a "superseded@<regime>" note rather than deleting the history.

4. **Classify every claim into one of THREE kinds — this governs whether and when it is revisited:**
   - **Engineering fact** — implementation correctness, independent of the science: NaN-guard works, active-stress
     sign convention, time alignment, `--amplitude 0` truly ablates. Tag `[engineering]`. **Almost never revisit.**
   - **Mechanistic hypothesis** — a claim about the *model/physics*: "gain is the dominant lever", "stiffness is
     inert", "fibre matters", "dur optimum ≈30". Tag `[mechanism]`. **These are exactly what to revisit when the
     regime changes** — they are what we are actually here to learn, and they are regime-conditional.
   - **Optimization observation** — a claim about *the optimizer's behaviour in a regime*: "plateau at 300it",
     "fibre learning harmful at 600it", "frozen fibre best at 1200it". Tag `[optimization@<regime>]`. Explicitly
     optimizer- and depth-dependent; never promote one of these to a mechanistic conclusion.

   **Do not interpret a lack of improvement after N iterations as evidence against a MECHANISM unless optimization
   itself has been shown to be CONVERGED in that regime.** "The optimizer hasn't exploited X yet" ≠ "the model can't
   do X" — conflating the two is the exact error the overnight run exposed (mechanisms that looked dead at ≤600it
   revived at 1200it).

Operational consequence: pin the converged depth N\* first (sweep iters until ΔR² per doubling is small), then
re-baseline `[mechanism]` verdicts against N\*. Do not relativise against a moving target, do not re-run
`[engineering]` facts, and keep `[optimization@…]` observations explicitly scoped to their regime.

## Each batch — do ALL (AUTO-UPDATE the files, do not wait for user input)

1. Per slot: Read its last `checkpoints/dashboard_*.png` (panels: sim-red/real-green trajectories, UNet stiffness, uniform gain, fibre angle + dx/dy) and its final interior R2 (progress.txt / job log `done -> (R2=...)`) + `config.json`. Note: does red superpose on green? what structure did the learned group converge to? R2 AND morphology (openness/chirality/size). RANK on interior R2.
2. EDIT `knowledge_cardio_mpm.md`: (a) **APPEND this batch's slot rows to the `### Comparison Table — Phase 2` block** (one row per slot: `Batch.slot | Config | learn | R2 | open | chir+ | size | note`, keep it sorted best-R²-first) — this table MUST stay current every batch; (b) update Established / Falsified / Open, each tied to slot(s) **AND, per the METHODOLOGICAL RULE, tagged with (i) its regime (`@<N>it`, base/dur/amp/mechanism) and (ii) its kind — `[engineering]` / `[mechanism]` / `[optimization@<regime>]`**. Never write a bare `FALSIFIED`/`CLOSED`. A clean falsification is a success; so is a clean OVERTURN of an earlier regime-bound verdict. When this batch's regime differs from that of an existing `[mechanism]` verdict, mark that verdict PROVISIONAL and re-examine it rather than citing it as settled — and never cite an `[optimization@…]` non-improvement as evidence against a mechanism unless that regime was shown converged.
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

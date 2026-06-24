# Phase 2 — Parametric Inverse Fit (Agentic Loop)

## What this is

**Phase 2 inverts the parametric pattern family to fit the real cardiomyocyte beat.** Each batch launches ≤6 `cardio_mpm_train2.py` jobs in parallel (one GPU each), one per config in `cardio_mpm_plan2.json`. When they finish you **read each job's dashboard**, **rank on interior R²**, update `knowledge_cardio_mpm.md` + `analysis_cardio_mpm.md`, and rewrite the plan for the next batch.

**Key difference from Phase 1:** Interior R² (real-trajectory match) is the PRIMARY metric. Morphology metrics are SECONDARY validators (confirming the loops have the right shape, not just right energy).

## Learnable parameters (15 total)

### PRIMARY: Fibre field (4)
- `fibre_wl` — wavelength of the fibre pattern (coarser = more elliptical)
- `fibre_angle` — rotation of the fibre pattern (controls chirality/orientation)
- `fibre_amplitude` — modulation strength of the fibre field
- `fibre_phase` — phase shift of the fibre pattern

### SECONDARY: Gain field (4)
- `gain_wl` — wavelength of gain (controls per-pixel active-stress magnitude)
- `gain_phase` — phase of gain pattern
- `gain_lo` — minimum gain value (maps to min active stress)
- `gain_hi` — maximum gain value (maps to max active stress)

### LOW PRIORITY: Stiffness field (4)
- `stiff_wl` — wavelength of stiffness (mostly inert per atlas; kept for completeness)
- `stiff_phase` — phase of stiffness
- `stiff_lo` — minimum youngs modulus
- `stiff_hi` — maximum youngs modulus

### GLOBAL: Mechanics (3)
- `amplitude` — active-contraction strength, **CONSTRAINED to [10, 15]** (not free)
- `drag_k` — overdamping coefficient
- `pulse_duration` — learnable beat envelope duration (locks to ~50 frames = real beat period)

## Boundary anchoring

**The outer band is Dirichlet-anchored to the real GT every frame.** The interior is free to move; the fit is measured over INTERIOR MOVING NODES ONLY (boundary excluded, motion-normalised R²). This is exactly how the old trainer worked.

## Each batch — do ALL

1. **Read every dashboard** (sim-red + real-green trajectories, parametric field maps): does red superpose on green? What STRUCTURE did the learned parameters create? **Rank on interior R²** (the honest metric).

2. **Record per slot:** R² · loop openness · ellipse axis-angle · loop size · chirality agreement · pattern spatial coherence. A clean R² improvement is a success.

3. **Append a dated batch section to `analysis_cardio_mpm.md`** (template below).

4. **Update `knowledge_cardio_mpm.md`:** which patterns converge to the real beat? Which morphology trade-offs emerge (e.g., optimizing fibre_angle worsens gain_wl)?

5. **Rewrite `cardio_mpm_plan2.json`** for the next ≤6 (one-knob-from-parent incl. an ABLATION).

## `analysis_cardio_mpm.md` — batch entry template (Phase 2)

```
## Batch N [name] — YYYY-MM-DD
Parent: slot 0 = <one-line config>
Hypothesis: "<quoted, testable>"
Slot k [name] fibre_wl=… fibre_angle=… gain_wl=… gain_phase=… R2=…  red-on-green=<superpose/off>  openness=… size=…
… slots …
Winner: slot k (R² AND morphology reason)
Verdict: <supported/falsified/inconclusive>
Failures: <slots with done=NO and suspected cause>
Next: parent=<config>; next batch probes <knob>.
```

## Current objective

**Fit the real cardiomyocyte beat on R² + loop-morphology loss, starting from the Phase-1 atlas winner (fibre_wl=40).** Search fibre_angle, gain_wl/phase, stiffness_phase with amplitude CONSTRAINED to [10,15]. Open questions:

- Which fibre_angle best matches the real beat's chirality / major-axis orientation?
- Does gain_wl modulation (spatial) improve R² vs a scalar gain amplitude?
- Can stiffness wavelength be ignored (set to atlas default), or does it couple to fibre when both are learned?
- What is the trade-off between loop morphology match (openness/aspect) and trajectory R²?
- Does amplitude stay in [10,15], or does the inverse pull it outside to reach better R²?

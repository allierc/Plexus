# Working Memory: cardio-MPM inverse fit

The DELIVERABLE is this ledger — defensible claims about which **stiffness + direction fields** and
knobs make the MLS-MPM model reproduce the real cardiomyocyte beat (red loops superposing on green).
Interior R² (motion-normalised, boundary excluded) adjudicates; it is never the goal. A clean
falsification is a success. Read + update EVERY batch. Seeded 2026-06-23 from the forward+inverse build.

## Paper Summary (update at theme boundaries)

- **Model:** UNet → (stiffness, direction) fields → real decomposed MLS-MPM forward → fit one
  beat-aligned cycle; outer band anchored, interior predicted. Two interpretable learned fields.
- **Fit one aligned beat:** SOLVED — 1 model frame = 1 real frame, pulse phase-locked to the real
  onset (period ≈ 50, 5 beats), differentiable window = the FULL inter-onset interval (closed loop).
- **Does it fit?** OPEN — at init R² ≈ −20 (overshoot regime, untuned). The loop's job is to find the
  fields/knobs that drive R² positive (red on green).

## Knowledge Base

### Comparison Table
| Batch.slot | Config | R2 | red-on-green | stiffness | direction |
| ---------- | ------ | -- | ------------ | --------- | --------- |
| init.ref   | directional_cardio, amp25 (pre-fix) | −34087 | no (overshoot) | ~uniform | noisy | 

### Established Principles
1. **The MPM is a stable elastic limit cycle.** Points return to rest; the quiescent state is
   reproducible after one cycle (no cross-cycle drift — the spring model's old streak failure is gone).
   → warm up `no_grad` past one cycle, backprop one beat.
2. **Time must be aligned to the real beat.** 1 model frame = 1 real frame; pulse period+phase locked
   to the detected real onsets (period ≈ 50). The differentiable window = the full inter-onset interval
   so the fitted loop CLOSES (matches `gt_compare.png`).
3. **Amplitude was applied twice (fixed).** The activation must be the gate `env·spatial` (~[0,1]);
   `pulse_to_contraction.amplitude` does the scaling. Double-applying gave ~25× overshoot (R²≈−34087);
   fixing it → R²≈−20 at init. Amplitude is now a single, sweepable knob.
4. **The honest metric is interior R²** (motion-normalised, boundary EXCLUDED, moving nodes). A small
   absolute RMSE is meaningless here (real motion ~6e-4); R²≤0 = worse than predicting no motion.
5. **Dashboard GT must use the canonical selection.** 10×10 / margin-10 nodes + fixed amp ×10 (the
   `gt_trajectories.png` recipe) — auto-amp + denser sampling made loops exceed node spacing → spaghetti.

### Falsified Hypotheses
_(none yet — seed)_

### Open Questions
- **Q1.** Can a *coherent* direction field emerge (vs noisy `dx/dy`), and does it match a sensible
  fibre/active-stress orientation?
- **Q2.** Does the learned stiffness develop structure that improves the per-node fit, or stay ~uniform?
- **Q3.** What `(amplitude, drag_k, dur0)` give a **closed, correctly-sized** loop without overshoot
  (R² → positive)? Is amplitude the dominant overshoot knob?
- **Q4.** Is the boundary-anchored interior R² a good proxy for the FREE (un-anchored) beat, or does
  it overstate the fit (as in the old spring model)?
- **Q5.** speed/fidelity: how low can `--substeps` / `--grad` go before the fit degrades?

---

## Previous Batch Summaries
**RULE: keep the last 4, oldest→newest, before `## Current Batch`.**

_(none yet)_

---

## Current Batch

### Batch info
_(batch number, parent/control config)_

### Current hypothesis
_(the specific testable claim this batch's slots probe)_

### Slots this batch
_(slot → one-knob delta → R2 + morphology, filled as analysed)_

### Emerging observations
**CRITICAL: this section must ALWAYS be at the END of the file.**

_(pre-launch) Seed batch 1 sweeps the overshoot/regime knobs around `directional_cardio`: amplitude
{10,15,25}, lr {2e-3,5e-3}, drag_k 60, dur0 15 — to find the amplitude/drag that close the loop (Q3)
and to see whether the UNet starts structuring stiffness/direction (Q1/Q2). Rank on interior R²._

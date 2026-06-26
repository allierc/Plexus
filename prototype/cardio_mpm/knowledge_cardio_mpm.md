# Knowledge: cardio-MPM inverse fit (distilled)

> **This is a DISTILLED paper, not an experiment log.** Keep it compact — it is reread every batch.
> Express knowledge as **causal statements** ("lower gain reduces overshoot until morphology collapses"),
> NOT numerical optima ("gain0=0.73 is best"). A parameter value matters only insofar as it reveals a
> reproducible mechanism that transfers to future models. Append per-batch detail to `analysis_cardio_mpm.md`
> and DISTILL it up into here — do not grow this file linearly. The full pre-2026-06-26 R²-era ledger is
> archived inside `analysis_cardio_mpm.md` ("Knowledge-ledger snapshot").

> **Objective = LoopScore (LS).** R² is a diagnostic only. Conclusions below established under the **R²**
> objective are marked `provisional@R²→LS` — they are HYPOTHESES to re-evaluate under LoopScore, carried
> forward (not erased). `[engineering]` facts carry over unchanged.

## Current objective

Discover which anisotropic active-stress material mechanisms produce the real cardiomyocyte loop
**morphology** — i.e. learn the mapping `parameters → loop family → LoopScore`, not merely find one
optimum. Maximize node-mean LoopScore (with low LS SD = uniform tissue).

## Current best result

- **Under LoopScore:** **LS = 0.589 ± 0.080** (archive p2_b14_s1: gain0=0.5, learn=fibre,gain,dur,
  2400it, stiff uniform [100,100], amp=10, drag=30, dur→30). Best chirality (chir+=0.69) and best
  uniformity (SD=0.080) of all archive slots. R²=−1.649 (diagnostic).
- **Under R² (diagnostic only):** best R² = −0.999 (2400 it, fibre frozen, gain+dur only, amp10).

## Established mechanisms  `[mechanism]` — causal, regime-conditional

1. **Loops are GENERIC in the active-stress MPM (inertial); structure TUNES morphology, it does not create
   it.** (2×2 test: isotropic loops ≈ structured.) ⇒ the target is loop *morphology* (size/axis/chirality/
   openness), and the forward mechanism is **active stress** (not body force, not rotary). `[engineering-ish
   + mechanism]`, robust.
2. **Gain is the dominant size/overshoot lever** — a single learned global scalar. Lower gain reduces
   contraction magnitude and overshoot; too low and the loops shrink below the real size. `provisional@R²→LS`.
3. **Pulse duration must turn OFF between beats to make loops** (contract→release→inertial recoil→loop). A
   pulse ≈ period is near-constant → radial stubs, openness collapses. Under R² the duration optimum was
   ~30 (~60% of the ~50-frame period), non-monotone with a turnover by ~40. `provisional@R²→LS`.
4. **Amplitude > ~10 is harmful in the parametric (no-rotary) regime** — total impulse ∝ amp×dur drives
   overshoot. Keep amplitude in [10,15]. `provisional@R²→LS`.
5. **Fibre is a MODERATE lever, not essential** (ablating it only mildly hurts) — consistent with #1
   (structure tunes, doesn't create). Parametric 4-scalar fibre; a small positive contraction-axis angle
   was preferred under R². `provisional@R²→LS`.

## Optimization facts  `[optimization@<regime>]` — depth-dependent, never promote to mechanism

- **NEVER TRUST OPTIMIZATION STATE.** Many R²-era conclusions FLIPPED purely because training continued.
  **Do not conclude "the mechanism doesn't work" when the truth may be "the optimizer hasn't discovered it
  yet"** — only after approximate convergence in that regime.
- Under R², depth was monotonically beneficial (600→−2.158, 1200→−1.411, 1800→−1.113, 2400→−0.999; not
  converged). **Under LoopScore, 3600it DEGRADED LS (0.567) vs 2400it (0.589)** at gain0=0.854 — the LS
  landscape may differ: the unbounded training loss keeps decreasing but the clamped LS score can overshoot.
  `[optimization@LoopScore, 2400–3600it, gain0=0.854]`. NEEDS re-test at gain0=0.5.
- Gain init is a lever under BOTH objectives: gain0=0.5 > gain0=0.854 under LoopScore (LS 0.589 vs 0.567
  at 2400it); gain0=0.7 > 0.854 under R² (Δ=0.201 at 1200it). Lower gain reduces overshoot →
  better-calibrated loops. `[optimization@LoopScore+R², 2400it]`.

## Engineering facts  `[engineering]` — stable, almost never revisit

- The MPM is a stable elastic limit cycle → warm up `no_grad` one cycle, backprop one beat.
- Time aligned to the real beat: 1 model frame = 1 real frame; pulse period+phase locked to detected onsets
  (period ≈ 50); differentiable window = full inter-onset interval so the loop closes.
- Active stress (M1) implemented + validated: `σ_act = +A·a(x)·n nᵀ` contracts ALONG axis n (sign verified;
  `−A` contracts perpendicular). Body force / rotary are the superseded force-track.
- NaN-guard: skip `opt.step()` when the clipped grad norm is non-finite (active stress can spike non-finite
  grads).
- `--amplitude 0` truly ablates (sentinel fix); amplitude is one sweepable knob (the double-apply ×25
  overshoot bug is fixed).
- Dashboard/montage GT uses the canonical 10×10 margin-10 node selection, fixed amp ×10.
- LoopScore (`cardio_harmonic.py`): per-node elliptic-Fourier loop morphology; reported LS = clamped score
  mean±SD; training loss = unbounded mean per-node r. `K=4` fixed — a future multiscale `K∈{1,2,4,8}`
  weighting could separate the global ellipse from local wiggles (Open).

## Rejected hypotheses (distilled — regime-tagged; re-openable)

- "Structure is necessary for loops" / "rotary force is required" — **FALSIFIED** (2×2; loops inertial;
  rotary was a scaffold for a force-based model, not a cardiomyocyte property).
- "Spatial stiffness lifts the fit" (UNet-microscope AND free SIREN) — **FALSIFIED@R²**; SIREN converged to
  uniform (no gradient signal under R²). `provisional@R²→LS` — re-test under LoopScore at COARSE wavelength
  (prior SIREN may have been too high-frequency / wrong scale).
- "Per-pixel fibre direction dθ (UNet or SIREN) helps" — **FALSIFIED@R²** (noisy-overshoot, representation-
  independent). `provisional@R²→LS` — re-test COARSE.
- "A travelling-wave phase delay τ(x,y) bends the loops" — **FALSIFIED** (τ stayed tiny; only a shorter-
  pulse effect, never a propagating wave).
- "Active stress is catastrophic / force ≫ stress" — **SUPERSEDED** (was a NaN artifact + an inverse-force-
  track comparison; active stress is now the forward mechanism and works).
- "Zero-motion collapse needs `w_amp` to defend the fit" — **FALSIFIED** (no collapse in the amp10–25
  regime; `w_amp` is a weak knob).

## Open questions

- **Re-evaluate `provisional@R²→LS` claims** — especially spatial stiffness (INERT@R²) and fibre co-learn
  (HARMFUL@R² at depth). SIREN fibre is now CATASTROPHIC under LS too (archive s2; ampL=9.49) — likely
  closed, but stiffness is untested.
- Does COARSE SIREN stiffness (omega=5, range 50–150) lift LoopScore? The per-node objective may provide
  gradient signal that the global R² couldn't. **Batch 1 tests this.**
- Is the gain-init monotone continuing below 0.5, or does it turn over? **Batch 1 tests gain0=0.3.**
- Does fibre co-learn help or hurt under LoopScore at gain0=0.5? (R²-era: hurt at depth.) **Batch 1 tests.**
- Pin converged optimization depth under LoopScore (3600it degraded at gain0=0.854; untested at gain0=0.5).
- What sets loop CHIRALITY, OPENNESS, major-AXIS angle — which lever controls each?
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — future option.

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Force/rotary track (Phase 1, R²):** built the inverse; found drag/amplitude/rotary as overshoot/shape
  levers; superseded by the 2×2 morphology pivot. (Detail: analysis snapshot.)
- **Parametric active-stress inverse (Phase 2, R²):** gain+dur are the load-bearing scalars; spatial
  stiffness/fibre (UNet & SIREN) closed under R²; depth monotone to R²=−0.999 at 2400 it, not converged.
- **Objective shift (2026-06-26):** R² → LoopScore; ledger distilled; prior conclusions → provisional.
- **LoopScore baseline (2026-06-26, archive p2_b14):** LS=0.589 at gain0=0.5, co-learn, 2400it. Tight
  clustering (0.567–0.589). SIREN fibre catastrophic. Depth degrades LS. Scalar model near ceiling.

---

## Current theme
### Current hypothesis
"Spatial stiffness (SIREN, coarse omega=5) may carry LoopScore gradient signal that R² couldn't provide — the
per-node loop-morphology penalty creates a spatially-resolved learning signal for regional stiffness variation."
### Iterations this theme
- Batch 1: baseline reproduction + re-test stiffness, fibre co-learn, gain monotonicity, depth under LS.
### Emerging observations
- LoopScore baseline established: LS=0.589±0.080 (gain0=0.5, co-learn, 2400it).
- Tight clustering of archive slots (LS 0.567–0.589) suggests scalar model near ceiling.
- SIREN fibre confirmed CATASTROPHIC under LS (consistent with R²-era).
- Depth behavior DIFFERS from R²: 3600it worse than 2400it under LS (at gain0=0.854).
**CRITICAL: this section must ALWAYS be at the END of the file.**

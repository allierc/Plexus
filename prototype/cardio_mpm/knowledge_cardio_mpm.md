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

- **Under LoopScore (the new objective):** to be (re)established in the next batches — read the kept
  `archive/p2_b14_*` harmonic runs for the current LS baseline before designing batch 1.
- **Under R² (old objective, diagnostic now):** best interior R² = **−0.999** (2400 it, fibre FROZEN,
  gain+dur only, stiff uniform, amp10). The R² fit was STILL improving at 2400 it (not converged).

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

- **NEVER TRUST OPTIMIZATION STATE.** Optimization depth is itself an experimental variable. Many R²-era
  conclusions FLIPPED purely because training continued (e.g. fibre co-learning helped at 600 it, became
  harmful at 1200–2400 it). The R² fit was not converged even at 2400 it (depth monotone: 600→−2.158,
  1200→−1.411, 1800→−1.113, 2400→−0.999; Δ/doubling still ~0.41). **Do not conclude "the mechanism doesn't
  work" when the truth may be "the optimizer hasn't discovered it yet"** — only after approximate
  convergence in that regime.
- Warm-start / init of the learned scalars (e.g. gain init) measurably shifted the R² result — an optimizer
  property, re-checkable under LoopScore.

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

- **Re-evaluate every `provisional@R²→LS` claim under LoopScore.** Which mechanisms that looked dead under R²
  improve LoopScore (especially while degrading R² — that is a success)?
- Does a COARSE free SIREN stiffness / fibre field lift LoopScore (the R²-era closure may be a wavelength
  artifact — see the coarse-region rule)?
- What sets loop CHIRALITY, OPENNESS, and major-AXIS angle under LoopScore — which lever controls each?
- Pin the converged optimization depth under LoopScore before trusting any mechanism verdict.
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — would separating global ellipse from local wiggle sharpen
  the objective?

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Force/rotary track (Phase 1, R²):** built the inverse; found drag/amplitude/rotary as overshoot/shape
  levers; superseded by the 2×2 morphology pivot. (Detail: analysis snapshot.)
- **Parametric active-stress inverse (Phase 2, R²):** gain+dur are the load-bearing scalars; spatial
  stiffness/fibre (UNet & SIREN) closed under R²; depth monotone to R²=−0.999 at 2400 it, not converged.
- **Objective shift (2026-06-26):** R² → LoopScore; ledger distilled; prior conclusions → provisional.

---

## Current theme
### Current hypothesis
### Iterations this theme
### Emerging observations
**CRITICAL: this section must ALWAYS be at the END of the file.**

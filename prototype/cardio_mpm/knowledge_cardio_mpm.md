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

- **Under LoopScore (corrected metric, floor=0.02):** **LS ≈ 0.14–0.15** (reproducible across
  5 runs of amp=12, stiff + floor, ω=5, gain0=0.5, 2400it: LS=0.159, 0.149, 0.149, 0.144, 0.140).
  Peak single-node: **+0.76** (with SIREN fibre, B6). ALL runs have exactly 1 catastrophic node,
  position depends on fibre init basin.
- **Best floored config:** stiff [80,300], LS=0.149, SD=0.178 (B5). stiff [80,200]: LS≈0.140 (B6).
- **Fibre-only (no stiffness):** LS ≈ 0.118 at amp=12 (B5). Stiffness adds ~0.02 net.
- **Under R² (diagnostic only):** best R² = −0.912 (gain0=0.3, fibre+gain+dur, 2400it).

## Established mechanisms  `[mechanism]` — causal, regime-conditional

1. **Loops are GENERIC in the active-stress MPM (inertial); structure TUNES morphology, it does not create
   it.** (2×2 test: isotropic loops ≈ structured.) ⇒ the target is loop *morphology* (size/axis/chirality/
   openness), and the forward mechanism is **active stress** (not body force, not rotary). `[engineering-ish
   + mechanism]`, robust.
2. **Gain is a size/overshoot lever** — a single learned global scalar. Lower gain reduces overshoot; at
   gain0=0.3 the mean LS is unchanged vs 0.5 but uniformity improves dramatically (SD 0.212→0.152).
   `[mechanism@LoopScore, 2400it, corrected metric]`. (R²-era: gain0=0.5 > 0.854; confirmed directionally.)
3. **Pulse duration must turn OFF between beats to make loops** (contract→release→inertial recoil→loop).
   Duration saturates at dur_hi in ALL slots under LoopScore — the optimizer wants maximum pulse energy.
   But **raising dur_hi is NOT beneficial**: dur_hi=40 HURTS (alone: LS 0.117 < 0.136; with stiffness:
   catastrophic LS=-0.070). The optimizer maximizes total impulse (gradient signal), but more pulse energy
   does not improve loop morphology. `[mechanism@LoopScore, 2400it, FALSIFIED@binding-constraint]`.
4. **Amplitude 12 is the CONFIRMED optimum.** amp=12 reproduces at LS≈0.15 (2 runs: 0.159, 0.149).
   amp=14 is CATASTROPHIC (3 outlier nodes, LS=-0.247). The amp=12→14 transition is sharp — amp=12
   is the upper stability boundary with wide stiffness. Keep amplitude ≤12.
   `[mechanism@LoopScore, 2400it, confirmed@2-runs]`.
5. **Fibre co-learning is LOAD-BEARING under LoopScore** — freezing fibre drops LS from 0.119→0.088
   (Δ=−0.031). This OVERTURNS the R²-era finding (fibre hurt at depth under R²). The parametric fibre
   provides orientation structure that the LS per-node gradient rewards. `[mechanism@LoopScore, 2400it]`.
6. **Coarse SIREN stiffness is ACTIVE under LoopScore with a BASIN-DEPENDENT OUTLIER.** SIREN (ω=5)
   converges to a binary spatial pattern adding ~0.02 LS over fibre-only. Every run has 1 catastrophic
   node (LS=-1.00), but the catastrophic node's POSITION is NOT fixed — it depends on the fibre init
   basin (B5: default→(2,3), angle=0.5→(1,1), phase=1.2→(1,1)). The stiffness × fibre basin
   INTERACTION creates the catastrophe, not a structural tissue property at one location. Stiffness
   FLOOR (80-100) prevents MULTI-node catastrophe and improves uniformity (SD 0.254→0.175). Wider
   range [80,300] is beneficial (LS=0.149 vs 0.137 at [80,200]). ω=5 confirmed sweet spot.
   `[mechanism+optimization@LoopScore, 2400it, ω=5, updated B5]`.
7. **Stiffness × duration interaction is DESTRUCTIVE.** Longer pulses (dur_hi=40) are tolerable with
   uniform stiffness (LS=0.117) but catastrophic with spatial stiffness (LS=-0.070). Soft regions
   amplify the extra pulse energy into runaway overshoot. Keep dur_hi=30 when stiffness is active.
   `[mechanism@LoopScore, 2400it]`.
8. **SIREN fibre dθ REDISTRIBUTES catastrophes when jointly optimized with SIREN stiffness.** Tight-bound
   SIREN fibre (dev=0.15–0.5 rad) CAN fix specific catastrophic nodes (per-node ceiling jumps from +0.31
   to +0.76) but CREATES NEW catastrophes at other nodes. Wider dev → more catastrophes (monotonic).
   The per-node ceiling increase proves the model COULD match morphology much better — the bottleneck is
   the JOINT stiffness×fibre optimization landscape, not the physics representation. Whether SIREN fibre
   works WITHOUT stiffness is UNTESTED (B7 experiment).
   `[mechanism+optimization@LoopScore, 2400it, joint stiff+fibre SIREN, B6]`.
9. **Gain is FLAT in [0.4, 0.5] with stiffness.** gain0=0.4 ≈ gain0=0.5 (LS=0.139 vs 0.140). Combined
   with gain0=0.3 catastrophic (B3) and gain0=0.7 catastrophic (B4) with wide stiffness, the viable gain
   window is 0.4–0.5. `[mechanism@LoopScore, 2400it, stiff [80,200], B6]`.

## Optimization facts  `[optimization@<regime>]` — depth-dependent, never promote to mechanism

- **NEVER TRUST OPTIMIZATION STATE.** Many R²-era conclusions FLIPPED purely because training continued.
- **Under corrected LoopScore, the model is CONVERGED at 2400it** — 3600it ≈ 2400it (LS 0.120 vs 0.119).
  The prior finding (3600it degraded LS at gain0=0.854 under old metric) may have been an artifact of the
  inflated metric. `[optimization@LoopScore-corrected, 2400-3600it, gain0=0.5]`.
- **Gain viable range is [0.4, 0.5] with wide stiffness.** gain0=0.3 CATASTROPHIC (LS=-0.406, B3);
  gain0=0.7 CATASTROPHIC (LS=-0.272, B4); gain0=0.4 ≈ 0.5 (LS=0.139 vs 0.140, B6). Without
  stiffness or with narrow stiffness, gain0 is flexible. `[optimization@LoopScore-corrected, 2400it]`.
- **SIREN stiffness convergence is STOCHASTIC but REPRODUCIBLE with floor.** Without floor:
  identical config can produce 1 or 3 catastrophic nodes (LS range -0.208 to +0.159). With floor=80:
  all runs have exactly 1 outlier, LS reproducible at ~0.135-0.140. B4+B5 controls reproduced:
  LS=0.149, 0.137. `[optimization@LoopScore, 2400it, ω=5]`.
- **Fibre parametric init landscape is HIGHLY NON-CONVEX.** angle=0.5 vs 0.17 changes LS by 0.093
  (0.044 vs 0.137). phase=1.2 vs 0.41 changes LS by 0.052. The optimizer does NOT escape bad fibre
  basins in 2400it. Different basins move the catastrophic node to different positions. The
  parametric fibre field may lack the expressiveness to avoid ALL catastrophic nodes simultaneously.
  `[optimization@LoopScore, 2400it, fibre init]`.
- **w_amp increase does NOT help.** w_amp=0.6 (vs 0.3 default) did not tame outliers and introduced
  mid-range negatives. The w_amp penalty conflicts with LoopScore optimization.
  `[optimization@LoopScore, 2400it]`.

## Engineering facts  `[engineering]` — stable, almost never revisit

- The MPM is a stable elastic limit cycle → warm up `no_grad` one cycle, backprop one beat.
- Time aligned to the real beat: 1 model frame = 1 real frame; pulse period+phase locked to detected onsets
  (period ≈ 50); differentiable window = full inter-onset interval so the loop closes.
- Active stress (M1) implemented + validated: `σ_act = +A·a(x)·n nᵀ` contracts ALONG axis n (sign verified;
  `−A` contracts perpendicular). Body force / rotary are the superseded force-track.
- NaN-guard: skip `opt.step()` when the clipped grad norm is non-finite.
- `--amplitude 0` truly ablates (sentinel fix); amplitude is one sweepable knob.
- Dashboard/montage GT uses the canonical 10×10 margin-10 node selection, fixed amp ×10.
- **LoopScore metric FIXED (commit 3dc8188):** energy floor lowered 0.05→0.02; per-node score = clamp(1-r,
  −1, 1). Old metric inflated scores for thin-loop nodes (~0.5 for stubs). Archive LS values (≤0.589)
  are INCOMPARABLE with corrected-metric values (≤0.133). `[engineering, 2026-06-26]`.
- **LoopScore sensitivity ranking** (from `make_loopscore_sensitivity.py`, GT-perturbation sweep):
  **chirality (1.97) ≈ size (1.96) >> axis orientation (0.77) > openness/aspect (0.62) >> temporal phase
  = position = 0**. Confirms designed invariances. Chirality and size are equally dominant — a mechanism
  that fixes either dimension has the most LS impact. `[engineering, regime-robust]`.
- LoopScore (`cardio_harmonic.py`): per-node elliptic-Fourier loop morphology; reported LS = clamped score
  mean±SD; training loss = unbounded mean per-node r. `K=4` fixed.

## Rejected hypotheses (distilled — regime-tagged; re-openable)

- "Structure is necessary for loops" / "rotary force is required" — **FALSIFIED** (2×2; loops inertial;
  rotary was a scaffold for a force-based model, not a cardiomyocyte property).
- "Spatial stiffness lifts the fit" — **FALSIFIED@R²; OVERTURNED@LoopScore.** SIREN stiffness converged to
  uniform under R² (no gradient signal) but is ACTIVE under LoopScore (binary pattern, LS +0.014). Moved
  to Established mechanisms #6.
- "Per-pixel fibre direction dθ (UNet or SIREN) helps" — **FALSIFIED@R², @LS at fibre_dev≥π/2, AND
  @LS with tight bounds (0.15–0.5 rad) WHEN JOINTLY OPTIMIZED WITH SIREN STIFFNESS** (B6). Tight-bound
  SIREN fibre fixes individual nodes (ceiling +0.76) but REDISTRIBUTES catastrophes to other nodes.
  `CLOSED for joint stiffness+fibre SIREN optimization.` **RE-OPENED for SIREN fibre WITHOUT stiffness**
  — the interaction may be the culprit, not the mechanism. `B7 test`.
- "A travelling-wave phase delay τ(x,y) bends the loops" — **FALSIFIED** (τ stayed tiny).
- "Active stress is catastrophic / force ≫ stress" — **SUPERSEDED** (NaN artifact).
- "Zero-motion collapse needs `w_amp` to defend the fit" — **FALSIFIED** (no collapse at amp10–25).
- "Fibre co-learn hurts under LS" — **FALSIFIED@LoopScore** (freezing fibre drops LS by 0.031). Fibre
  is load-bearing under LS (opposite of R²-era at depth). Moved to Established mechanisms #5.
- "dur_hi=30 is a binding constraint on loop SIZE" — **FALSIFIED@LoopScore, 2400it.** Raising dur_hi
  to 40 HURTS (alone: LS drops; with stiffness: catastrophic). Duration saturation ≠ energy starvation.
- "Coarser stiffness (ω=3) helps" — **FALSIFIED@LoopScore, 2400it.** ω=3 regions are too large, LS
  drops. ω=5 is better-sized for this tissue. `CLOSED for ω<5`.
- "Finer stiffness (ω=7) helps" — **FALSIFIED@LoopScore, 2400it.** ω=7 catastrophic (LS=-0.217);
  finer field creates more overshoot regions. `CLOSED for ω>5`. ω=5 confirmed as sweet spot.
- "Stiffness floor tames outliers" — **PARTIALLY SUPPORTED@LoopScore, 2400it.** Floor prevents
  MULTI-node catastrophe (SD 0.254→0.175). The single outlier survives but its POSITION varies
  with fibre init (B5) — it's a fibre×stiffness basin interaction, not position-fixed.
- "Different fibre init eliminates the persistent outlier" — **PARTIALLY FALSIFIED@LoopScore, 2400it.**
  Different angle/phase MOVES the catastrophic node to different positions but never eliminates it
  entirely. The parametric fibre field may lack sufficient local expressiveness. `B5`.
- "Coarser fibre (wl=40) improves chirality/orientation" — **FALSIFIED@LoopScore, 2400it.** LS=-0.051,
  3 catastrophic nodes. Coarser fibre destabilizes SIREN stiffness optimization.
- "gain0=0.3 + wide stiffness synergizes" — **FALSIFIED@LoopScore, 2400it.** LS=-0.406, catastrophic.
- "gain0=0.7 + wide stiffness helps" — **FALSIFIED@LoopScore, 2400it.** LS=-0.272, 3 outliers.
  Only gain0=0.5 survives with wide stiffness [30,200].
- "w_amp=0.6 tames outlier overshoot" — **FALSIFIED@LoopScore, 2400it.** LS=0.144 (below control
  0.149); introduced mid-range negatives. Anti-collapse penalty conflicts with morphology loss.
- "amp=14 extends amp=12 benefit" — **FALSIFIED@LoopScore, 2400it.** amp=14 catastrophic (3
  outliers, LS=-0.247). The amp=12→14 transition is sharp. `CLOSED for amp>12`.
- "Higher parametric fibre amplitude (0.8 vs 0.39) improves orientation" — **FALSIFIED@LoopScore,
  2400it.** LS drops from 0.140 to 0.098 with 3 negative nodes. Higher fibre_amp destabilizes
  stiffness convergence. `B6`.
- "gain0=0.4 differs from gain0=0.5" — **FALSIFIED@LoopScore, 2400it, stiff [80,200].**
  LS=0.139 vs 0.140 (identical). Gain is flat in [0.4,0.5]. `B6`.

## Open questions

- **Does SIREN fibre dθ work WITHOUT SIREN stiffness (uniform stiffness)?** B6 showed SIREN fibre
  REDISTRIBUTES catastrophes when jointly optimized with SIREN stiffness. The interaction — not
  the fibre dθ mechanism itself — may be the cause. If SIREN fibre with uniform stiffness exceeds
  the fibre-only ceiling (LS≈0.118) without catastrophes, the mechanism works in isolation.
  **#1 priority.** `B7 test`.
- **Does coarser SIREN fibre (ω=3 instead of ω=5) produce more coherent directional corrections?**
  The B6 dθ maps were noisy/speckled at ω=5. With uniform stiffness, ω only affects the fibre
  SIREN — testing ω=3 may produce coarser, more effective directional corrections. `B7 test`.
- **What is the dominant BOTTLENECK dimension across nodes?** Need residual decomposition to
  quantify. SIZE and CHIRALITY are the top-sensitivity dimensions (engineering).
- **Can amp<12 help?** amp=12 is optimal, amp=14 catastrophic. amp=10 is untested at [80,300].
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — future option.

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 3 (2026-06-27):** CONTROL FAILED — stiffness stochasticity. 1 vs 3 outlier nodes = LS sign.
  ω=7 catastrophic. amp=12 marginal best (0.159). Single-run stiffness comparisons unreliable.
- **Batch 4 (2026-06-27):** Stiffness floor (80, 100) fixes UNIFORMITY (SD 0.254→0.175) but NOT the
  persistent outlier at node (2,3). Control reproduced (LS=0.149). amp=14 and gain0=0.7 catastrophic.
  w_amp=0.6 unhelpful. The outlier is STRUCTURAL, not stiffness-related. Redirects to fibre direction.
- **Batch 5 (2026-06-27):** Fibre init sensitivity MASSIVE (ΔLS=0.093 from angle change). Catastrophic
  node is NOT position-fixed — moves with fibre basin. stiff_hi=300 matches best (LS=0.149). Fibre-only
  ceiling at amp=12 confirmed: LS=0.118. Parametric fibre lacks local expressiveness. Redirects to
  tight-bound SIREN fibre as the next mechanism.
- **Batch 6 (2026-06-28):** SIREN fibre with tight bounds (0.15–0.5 rad) REDISTRIBUTES catastrophes
  rather than eliminating them — fixes specific nodes (ceiling +0.76!) but creates new catastrophes at
  other nodes. Joint stiffness×fibre SIREN optimization is the culprit. gain0=0.4≈0.5 (flat).
  fibre_amp=0.8 destabilizes. CLOSED for joint SIREN optimization. Redirects to fibre SIREN isolation.

---

## Current theme
### Current hypothesis
"The catastrophe redistribution observed in B6 is caused by the SIREN fibre × SIREN stiffness
INTERACTION — not by the SIREN fibre mechanism itself. SIREN fibre with UNIFORM stiffness (no
stiffness SIREN) should provide local directional corrections without the destabilizing interaction,
potentially exceeding the fibre-only ceiling (LS≈0.118). A coarser fibre SIREN (ω=3) may produce
more coherent corrections than the noisy ω=5 patterns seen in B6."
### Iterations this theme
- Batch 1–4: stiffness is active (ω=5), floor helps uniformity, outlier persists regardless of
  stiffness params. Fibre is load-bearing. Redirected to fibre direction.
- Batch 5: fibre init sensitivity confirmed. Catastrophic node MOVES with fibre basin (not
  position-fixed). Fibre-only ceiling = LS≈0.118 at amp=12.
- Batch 6: SIREN fibre (tight bounds 0.15–0.5 rad) + SIREN stiffness → catastrophe REDISTRIBUTION.
  Per-node ceiling jumps to +0.76. The mechanism works locally but joint optimization destabilizes.
  CLOSED for joint SIREN optimization. RE-OPENED for fibre SIREN without stiffness.
- Batch 7: test SIREN fibre with UNIFORM stiffness (isolate the interaction). Compare ω=5 vs ω=3
  for fibre granularity. Also probe amp=10 and establish stiff [80,300] as new parent.
### Emerging observations
- Reproducible best: LS≈0.14–0.15 with amp=12, stiff [80,200-300], ω=5 (5 runs: 0.159–0.140).
- Every stiffness-active run has 1 catastrophic node. Position varies with fibre basin.
- Per-node ceiling: **+0.76** (with SIREN fibre, B6). Model CAN match morphology much better.
- Eliminating the outlier would lift mean LS from ~0.15 to ~0.26 — the single largest possible gain.
- SIREN fibre + stiffness SIREN: CLOSED (catastrophe redistribution). Without stiffness: UNTESTED.
- gain0 flat in [0.4,0.5] with stiffness. Viable window: 0.4–0.5 with wide stiffness.
- Sensitivity: chirality ≈ size >> orientation > openness (engineering, regime-robust).
**CRITICAL: this section must ALWAYS be at the END of the file.**

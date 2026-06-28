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
  7+ runs of amp=10-12, stiff [80,300], ω=5, gain0=0.5, 2400it: LS=0.159, 0.151, 0.150, 0.149×2,
  0.144, 0.140). Peak single-node: **+1.00** (B7 s3/s5). ALL runs have exactly 1 catastrophic node,
  position depends on fibre init basin.
- **Best floored config:** stiff [80,300], LS=0.151, SD=0.178 (B7 control). amp=10 ≈ amp=12.
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
4. **Amplitude is FLAT in [10, 12] and CATASTROPHIC at 14.** amp=10 ≈ amp=12 (LS=0.150 vs 0.151,
   identical per-node patterns, B7). amp=14 is catastrophic (LS=-0.247, B4). The model compensates
   for amplitude changes via gain/stiffness adjustment — amplitude is not a morphology lever within
   [10,12]. Keep amplitude in [10,12].
   `[mechanism@LoopScore, 2400it, confirmed@B4+B7]`.
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
8. **SIREN fibre dθ is INTRINSICALLY DESTABILIZING — CLOSED across all configurations.** Without
   stiffness: CATASTROPHIC (ω=5: LS=-0.222; ω=3: LS=-0.047, B7). With stiffness: catastrophe
   redistribution (B6: LS≈0.098–0.140; B7: LS=0.011). Stiffness actually STABILIZES the fibre SIREN
   (opposite of the B6 hypothesis). The per-pixel SIREN fibre has too many degrees of freedom for the
   optimizer to solve globally — it improves some nodes (ceiling +0.76) while destroying others.
   The per-node ceiling proves the model CAN match morphology; the bottleneck is the OPTIMIZATION
   LANDSCAPE of per-pixel direction fields. `CLOSED for SIREN fibre dθ at current architecture.`
   `[mechanism+optimization@LoopScore, 2400it, dev=0.3, ω=3-5, ±stiffness, B6+B7]`.
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
- "Per-pixel fibre direction dθ (UNet or SIREN) helps" — **FALSIFIED across ALL configurations.**
  With stiffness: catastrophe redistribution (B6). Without stiffness: WORSE — LS=-0.222 (ω=5) or
  -0.047 (ω=3), B7. Stiffness STABILIZES fibre SIREN (opposite of hypothesis). The SIREN dθ
  optimization landscape is intrinsically too rough at per-pixel resolution. `CLOSED. B6+B7`.
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
- "SIREN fibre without stiffness avoids catastrophe redistribution" — **FALSIFIED@LoopScore, 2400it.**
  Without stiffness: LS=-0.222 (ω=5), -0.047 (ω=3). Far WORSE than with stiffness. Stiffness
  STABILIZES fibre SIREN. `B7`.
- "amp=10 differs from amp=12" — **FALSIFIED@LoopScore, 2400it, stiff [80,300].** LS=0.150 vs 0.151,
  identical per-node patterns. Amplitude is flat in [10,12]. `B7`.

## Open questions

- **What is the dominant BOTTLENECK dimension across nodes?** Need residual decomposition to
  quantify. SIZE and CHIRALITY are the top-sensitivity dimensions (engineering). **#1 priority.**
- **Does drag_k affect loop morphology?** drag_k=30 is COMPLETELY UNTESTED. It controls the
  damping timescale (overdamped drag). Higher drag → slower response → potentially different loop
  shape/chirality. Lower drag → more dynamic/inertial → larger loops. `B8 test`.
- **Does w_amp=0 improve LS?** w_amp=0.3 is default; 0.6 failed (B4). With LoopScore as objective,
  the anti-collapse penalty may conflict with morphology optimization. Ablating it may free the
  optimizer. `B8 test`.
- **Can deeper optimization (3600it) at the best config [80,300] break the LS≈0.15 plateau?**
  Previously tested at a weaker config. `B8 test`.
- **Can a coarser parametric fibre (wl=35) with stiffness improve over wl=28.8?** wl=40 was
  catastrophic but wl=35 is untested. The fibre+stiffness interaction may benefit from slightly
  coarser direction. `B8 test`.
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — future option.

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 4 (2026-06-27):** Stiffness floor (80, 100) fixes UNIFORMITY (SD 0.254→0.175) but NOT the
  persistent outlier. Control reproduced (LS=0.149). amp=14 and gain0=0.7 catastrophic.
- **Batch 5 (2026-06-27):** Fibre init sensitivity MASSIVE (ΔLS=0.093 from angle change). Catastrophic
  node is NOT position-fixed — moves with fibre basin. stiff_hi=300 best (LS=0.149). Fibre-only
  ceiling: LS=0.118. Parametric fibre lacks local expressiveness.
- **Batch 6 (2026-06-28):** SIREN fibre with tight bounds REDISTRIBUTES catastrophes — fixes specific
  nodes (ceiling +0.76!) but creates new catastrophes. gain0=0.4≈0.5 (flat). CLOSED for joint SIREN.
- **Batch 7 (2026-06-28):** SIREN fibre WITHOUT stiffness is WORSE (LS=-0.222), not better — FALSIFIES
  the stiffness-interaction hypothesis. SIREN fibre is intrinsically destabilizing. Stiffness
  STABILIZES. amp=10 ≈ amp=12. SIREN fibre CLOSED across ALL configurations. Controls reproduce.

---

## Current theme
### Current hypothesis
"The parametric model has plateaued at LS≈0.15 — all spatial-field levers (stiffness, fibre direction)
are exhausted at the current SIREN architecture. Progress requires probing UNTESTED physical parameters
(drag_k, w_amp) that affect the DYNAMIC RESPONSE (loop shape via damping timescale), or optimization
strategies (deeper training, lower lr) that improve the global solution. Drag_k controls the overdamped
timescale — different drag may shift the loop morphology family (chirality/size balance) toward the
target."
### Iterations this theme
- Batch 1–7: stiffness [80,300] at ω=5 is the stable spatial lever (+0.03 LS over fibre-only).
  SIREN fibre dθ CLOSED (intrinsically destabilizing). amp flat in [10,12]. gain flat in [0.4,0.5].
  Parametric fibre ceiling LS≈0.118. Best reproducible: LS≈0.15. Per-node ceiling: +1.00 (B7).
- Batch 8: probe drag_k (untested), w_amp=0 (ablation), deeper training (3600it), and coarser
  fibre wavelength. The goal: break the LS≈0.15 plateau via physical or optimization parameters.
### Emerging observations
- Reproducible best: LS≈0.15 with amp=10-12, stiff [80,300], ω=5 (7+ runs: 0.140–0.159).
- Every stiffness-active run has 1 catastrophic node. Position varies with fibre basin.
- Per-node ceiling: **+1.00** (B7 s3/s5). Model CAN perfectly match individual nodes.
- SIREN fibre dθ: CLOSED across all configs (with/without stiffness, ω=3-5, dev=0.15-0.5).
- Eliminating the outlier would lift mean LS from ~0.15 to ~0.26 — the single largest possible gain.
- gain0 flat in [0.4,0.5]; amp flat in [10,12]. Viable window is narrow but stable.
- Sensitivity: chirality ≈ size >> orientation > openness (engineering, regime-robust).
- UNTESTED: drag_k, w_amp=0, 3600it at best config, fibre_wl=35.
**CRITICAL: this section must ALWAYS be at the END of the file.**

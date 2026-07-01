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

- **Under LoopScore (corrected metric, floor=0.02):** **LS = 0.323** (B14 sgain_amp10),
  spatial gain (SIREN ω=5), stiff [80,300], gain0=0.5, amp=10, dur_hi=11→dur=10.0, 2400it.
  1 barely-negative node (-0.02). Per-node ceiling = 0.87 (B14 dur0_8).
  **BROKE the LS≈0.20 uniform-gain ceiling** (Δ=+0.118 vs ctrl 0.205, +58%).
- **ALL-POSITIVE config:** LS=0.277 (B14 sgain_deep, 3600it). ZERO negatives. First
  reliably ALL-positive with spatial gain at depth.
- **Previous plateau (uniform gain):** LS ≈ 0.20 (B12–B13), now superseded.
- **Fibre-only (no stiffness):** LS ≈ 0.118 at amp=12 (B5). Spatial gain without stiffness:
  LS=0.125 (B14). Stiffness + spatial gain together = 0.323.
- **Under R² (diagnostic only):** best R² = −0.912 (gain0=0.3, fibre+gain+dur, 2400it).

## Established mechanisms  `[mechanism]` — causal, regime-conditional

1. **Loops are GENERIC in the active-stress MPM (inertial); structure TUNES morphology, it does not create
   it.** (2×2 test: isotropic loops ≈ structured.) ⇒ the target is loop *morphology* (size/axis/chirality/
   openness), and the forward mechanism is **active stress** (not body force, not rotary). `[engineering-ish
   + mechanism]`, robust.
2. **Gain is a size/overshoot lever** — a single learned global scalar. Lower gain reduces overshoot; at
   gain0=0.3 the mean LS is unchanged vs 0.5 but uniformity improves dramatically (SD 0.212→0.152).
   `[mechanism@LoopScore, 2400it, corrected metric]`. (R²-era: gain0=0.5 > 0.854; confirmed directionally.)
3. **Pulse duration controls a MONOTONIC LS–UNIFORMITY TRADEOFF, but the "Goldilocks zone" is
   STOCHASTIC, not deterministic.** Four regimes mapped: (a) dur≈30 (init trap, LS≈0.06-0.12);
   (b) dur≈19-21 (intermediate, LS≈0.16); (c) dur≈9.5-10 (best mean LS≈0.208); (d) dur≈8.5
   (LS=0.211 but more negatives). B11's "zero-negative" result at dur=10 was a STOCHASTIC
   outcome — B12's identical config produced 2 negatives (-0.52, -0.54). The catastrophe is
   energy overshoot controlled by duration, but FULL elimination depends on SIREN init luck.
   The LS≈0.208 plateau is reached by both dur_hi=11 (2400it) and dur_hi=12 (3600it).
   `[mechanism@LoopScore, 2400-3600it, UPDATED@B12 — stochastic uniformity]`.
4. **Amplitude×duration×gain-type INTERACTION (3-way).** At dur≈19: amp=10 ≈ amp=12 (B7).
   At dur≈11 with UNIFORM gain: amp=10 HURTS (B11: 0.184 < 0.191). At dur≈11 with SPATIAL
   gain: amp=10 is BEST (B14: 0.323 > 0.218 at amp=12). The mechanism: spatial gain
   compensates for lower base amplitude by varying contraction regionally, while lower
   amplitude reduces the OVERSHOOT ceiling globally. amp=14 remains catastrophic.
   `[mechanism@LoopScore, 2400it, UPDATED@B14, amp×dur×gain-type 3-way interaction]`.
5. **Fibre co-learning is LOAD-BEARING under LoopScore** — freezing fibre drops LS from 0.119→0.088
   (Δ=−0.031). This OVERTURNS the R²-era finding (fibre hurt at depth under R²). The parametric fibre
   provides orientation structure that the LS per-node gradient rewards. `[mechanism@LoopScore, 2400it]`.
6. **Coarse SIREN stiffness is CRITICAL — range [80,300] is a HARD OPTIMUM.** SIREN (ω=5)
   converges to a binary spatial pattern. At dur≈11, stiffness adds ~0.10 LS (0.191→0.092
   without it, B11). The wide range [80,300] is essential: NARROWING to [100,200] HURTS badly
   (Δ=-0.036 at durhi12, Δ=-0.035 at durhi10 + catastrophic, B12); raising floor to [100,300]
   also HURTS (Δ=-0.017). The spatial contrast between very soft (80) and very stiff (300)
   regions is how the SIREN controls local overshoot. ω=5 confirmed. The stiffness SIREN
   convergence is STOCHASTIC: identical configs produce different per-node outcomes depending
   on initialization.
   `[mechanism@LoopScore, 2400it, ω=5, UPDATED@B12, stiff range is hard optimum]`.
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
9. **Gain is FLAT in [0.4, 0.5] with stiffness — confirmed at 3600it.** gain0=0.4 ≈ gain0=0.5
   at both 2400it (LS=0.139 vs 0.140, B6) and 3600it (LS=0.159 vs 0.160, B9). Combined with
   gain0=0.3 catastrophic (B3) and gain0=0.7 catastrophic (B4) with wide stiffness, the viable
   gain window is 0.4–0.5 and does NOT differentiate at any tested depth.
   `[mechanism@LoopScore, 2400-3600it, stiff [80,300], B6+B9]`.
10. **Drag has an ASYMMETRIC floor at ~30.** drag_k=50 ≈ 30 (LS=0.152 both), but drag_k=20
    HURTS (LS=0.112, 2 catastrophic nodes). Lower damping fails to control elastic recoil
    overshoot. Above the floor, drag is inert — the system is overdamped enough.
    `[mechanism@LoopScore, 2400it, B8]`.
11. **w_amp (anti-collapse penalty) is LOAD-BEARING at 0.3.** w_amp=0 drops LS to 0.132 with
    much worse SD (0.272 vs 0.181). Without it, vulnerable nodes lose motion. w_amp=0.6 also
    HURTS (B4). The optimal w_amp is ~0.3 (neither 0 nor higher). `[mechanism@LoopScore, 2400it, B8]`.
12. **Stiffness range has an upper limit at stiff_hi=300.** stiff_hi=400 is catastrophic
    (LS=0.076, 5/9 negative). Too much contrast creates multiple catastrophic nodes.
    `[mechanism@LoopScore, 2400it, B8]`.
13. **Problem nodes (positions 0, 5) are SOLVABLE with spatial gain.** Under uniform gain,
    positions 0 and 5 were persistently negative (range -0.03 to -0.79, B12–B13). With spatial
    gain: position 0 → +0.32, position 5 → -0.02 (B14 amp10). At 3600it: ALL nodes positive
    (B14 deep). The problem was uniform gain's inability to vary contraction per-region, NOT
    a fundamental model limitation.
    `[mechanism@LoopScore, B14, UPDATED — OVERTURNS "physics model limitation" for these nodes]`.
14. **The LS≈0.20 plateau WAS a uniform-gain ceiling — BROKEN by spatial gain (LS=0.323).**
    B13 confirmed the plateau was not SIREN capacity (hidden384 collapsed). B14 confirmed
    it was uniform gain: spatial gain (SIREN ω=5, [0.1, 2.5]) broke through to LS=0.323
    (+58% over ctrl). The mechanism is per-region contraction amplitude decoupled from
    stiffness (amplitude vs frequency/timing).
    `[mechanism@LoopScore, 2400-3600it, durhi11, stiff [80,300], ω=5, UPDATED@B14]`.
15. **SPATIAL GAIN is the CEILING-BREAKING mechanism.** The SIREN gain field g(x,y) ∈ [0.1, 2.5]
    (ω=5) decouples per-region active-stress AMPLITUDE from material STIFFNESS. Stiffness
    controls the elastic response (natural frequency, loop shape/timing); gain controls the
    driving force magnitude (loop size). Both are needed: gain alone (no stiffness) gives only
    LS=0.125 (uniform weak loops); stiffness alone (no spatial gain) caps at LS≈0.20.
    Together: LS=0.323. Gain+stiffness provide COMPLEMENTARY spatial control channels.
    `[mechanism@LoopScore, 2400it, B14, NEW]`.
16. **Spatial gain is NOT converged at 2400it.** s0 (2400it, LS=0.218) → s1 (3600it, LS=0.277):
    +0.059 in 1200 extra iterations, with ALL negatives eliminated. The spatial gain SIREN
    has more parameters than scalar gain, and continues improving past 2400it. The 2400it
    best (amp10, LS=0.323) may improve further with depth.
    `[optimization@LoopScore, B14, spatial gain, 2400-3600it]`.

## Optimization facts  `[optimization@<regime>]` — depth-dependent, never promote to mechanism

- **NEVER TRUST OPTIMIZATION STATE.** Many conclusions FLIPPED purely because training continued or
  init changed. B9 is a textbook case: "dur=24 is the interior optimum" (B8) was itself an optimization
  artifact — dur0=10 at 2400it reaches dur≈19.4, matching 4150it. Depth was only needed because the
  default init (dur0=14) trapped duration in the dur≈30 basin.
- **3600it does NOT break the uniform-gain LS≈0.20 plateau** (B13) **but DOES matter with
  spatial gain:** +0.059 (0.218→0.277) and eliminates ALL negatives (B14). Depth is more
  valuable when the physics model has enough degrees of freedom to exploit.
  `[optimization@LoopScore, B12-B14, UPDATED@B14]`.
- **Wider SIREN (hidden384) HURTS optimization.** LS=0.146 (Δ=-0.052 vs ctrl 0.198). The
  larger parameter space creates a harder optimization landscape. More capacity ≠ better
  convergence for SIREN stiffness. `[optimization@LoopScore, 2400it, B13]`.
- **lr=5e-4 is NEUTRAL vs lr=1e-3.** LS=0.201 ≈ ctrl 0.198. Lower lr does not improve
  SIREN convergence quality. `[optimization@LoopScore, 2400it, B13]`.
- **Gain viable range is [0.4, 0.5] with wide stiffness.** gain0=0.3 CATASTROPHIC (LS=-0.406, B3);
  gain0=0.7 CATASTROPHIC (LS=-0.272, B4); gain0=0.4 ≈ 0.5 (LS=0.139 vs 0.140, B6). Without
  stiffness or with narrow stiffness, gain0 is flexible. `[optimization@LoopScore-corrected, 2400it]`.
- **SIREN stiffness convergence is STOCHASTIC — controls 0-3 negative nodes.** B12 ctrl
  (identical to B11 durhi12 which had ZERO negatives) produced 2 negatives (-0.52, -0.54).
  LS varies ±0.014 between identical runs. The all-positive property is NOT reproducible.
  With floor=80: LS reproducible at ~0.19-0.21 but negative-node count varies.
  `[optimization@LoopScore, 2400it, ω=5, UPDATED@B12]`.
- **Fibre parametric init landscape is HIGHLY NON-CONVEX and INTERACTS with duration.**
  angle=0.5 vs 0.17 changes LS by 0.093 (0.044 vs 0.137). phase=1.2 vs 0.41 changes LS by 0.052.
  The optimizer does NOT escape bad fibre basins in 2400it. **NEW (B10): angle=0.5 TRAPS duration
  at dur≈28 even with dur0=10** (vs default angle=0.17 which allows dur→19). The fibre init basin
  determines WHICH duration basin the optimizer can access — a cross-parameter interaction.
  `[optimization@LoopScore, 2400it, fibre init, UPDATED@B10]`.
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
- "amp=10 differs from amp=12" — **Complex 3-way interaction.** At dur≈19: FLAT (B7). At dur≈11
  with UNIFORM gain: amp=10 HURTS (B11). At dur≈11 with SPATIAL gain: amp=10 is BEST (B14:
  LS=0.323 vs amp=12 LS=0.218). The interaction depends on the gain type. `B7+B11+B14`.
- "Spatial gain without stiffness is sufficient" — **FALSIFIED@B14.** LS=0.125 (nostiff) vs
  0.218-0.323 (with stiff). Stiffness and gain are complementary mechanisms. `B14`.
- "drag_k=50 or drag_k=20 changes loop morphology vs drag_k=30" — drag_k=50 **FALSIFIED**
  (LS=0.152≈ctrl). drag_k=20 **HARMFUL** (LS=0.112, 2 catastrophes). Drag is flat above 30
  and destructive below. `B8`.
- "w_amp=0 frees the optimizer" — **FALSIFIED@LoopScore, 2400it.** LS=0.132, SD=0.272. The
  anti-collapse penalty is load-bearing. `B8`.
- "stiff_hi=400 extends stiff_hi=300 benefit" — **FALSIFIED@LoopScore, 2400it.** LS=0.076,
  5/9 negative. Upper limit is 300. `B8`.
- "Duration saturates at dur_hi in ALL slots" — **OVERTURNED@3600it.** Duration finds an
  interior optimum at dur=24 when trained to 3600it. The saturation at 2400it was an
  optimization-depth artifact. `B8`.
- "dur=24 is the duration interior optimum" — **OVERTURNED@B9→B10.** dur≈19-21 was itself
  suboptimal; dur_hi=15 forces dur=11.3 → LS=0.196. Three basins: 30, 19-21, 11. `B9→B10`.
- "dur≈19-21 is the true duration optimum" — **OVERTURNED@B10.** dur_hi=15 → dur=11.3 →
  LS=0.196, beating dur≈19 (LS≈0.16). The improvement comes from taming the catastrophic
  node (LS=-1.00 → -0.45). Short duration limits overshoot energy. `B10`.
- "dur_hi=25 forces duration into the optimal range at 2400it" — **FALSIFIED@LoopScore, B9.**
  dur_hi=25 pins duration at 24.4 (ceiling), which is the WRONG basin. LS=0.157, worst in
  batch. The ceiling prevents escape to the true optimum at ~20. `B9`.
- "3600it provides a broad optimization benefit beyond duration" — **FALSIFIED@LoopScore, B9.**
  dur0=10 at 2400it (LS=0.165) matches 4150it (LS=0.166). The 3600it benefit was specifically
  duration escaping its basin, not general optimization quality. `B9`.
- "gain0=0.4 differs from gain0=0.5 at 3600it" — **FALSIFIED@LoopScore, 3600it, B9.**
  LS=0.159 vs 0.160. Gain remains flat at all tested depths. `B9`.
- "fibre_phase=1.2 with dur0=10 eliminates catastrophe" — **FALSIFIED@LoopScore, 2400it, B10.**
  LS=0.158 vs ctrl 0.161. Phase change is neutral/slightly negative; catastrophic node remains
  at LS=-1.00. `B10`.
- "fibre_angle=0.5 with dur0=10 is viable" — **FALSIFIED@LoopScore, 2400it, B10.** LS=0.060.
  angle=0.5 TRAPS duration at dur≈28 even with dur0=10 — a fibre×duration cross-interaction.
  Only angle=0.17 allows duration to escape to the short-duration basin. `B10`.
- "wl=35 improves over wl=28.8" — **FALSIFIED@LoopScore, 2400it, B10.** LS=0.165 ≈ ctrl 0.161.
  wl=35 is neutral, unlike wl=40 (catastrophic). Viable wl range is [28.8, 35]. `B10`.
- "Narrowing stiffness range [100,200] extends the Goldilocks zone" — **FALSIFIED@LoopScore,
  2400it, B12.** At durhi12: LS=0.158 (Δ=-0.036); at durhi10: LS=0.159 (catastrophic -1.00
  returns). Narrowing removes essential spatial contrast. [100,300] (raised floor) also HURTS
  (Δ=-0.017). Range [80,300] is a HARD OPTIMUM. `B12`.
- "The all-positive property at dur=10 is deterministic" — **FALSIFIED@B12.** Identical config
  to B11 durhi12 (LS=0.200, zero negatives) produced 2 negatives (LS=0.194) in B12 ctrl.
  The zero-negative outcome depends on SIREN stiffness initialization, not duration alone. `B12`.
- "dur_hi=11 is intermediate between durhi12 and durhi10" — **PARTIALLY SUPPORTED@B12.**
  dur_hi=11→dur=9.5→LS=0.208, matching deep3600. But the regime-specific morphology is not
  distinct from durhi12 — it just reaches the same LS≈0.208 plateau more efficiently. `B12`.
- "SIREN capacity (hidden384 or layers4) breaks the LS≈0.20 plateau" — **FALSIFIED@B13.**
  hidden384 COLLAPSED (LS=0.146, Δ=-0.052); layers4 neutral (LS=0.206, Δ=+0.008, stochastic).
  The plateau is NOT a network capacity ceiling — it is a physics model ceiling. `B13`.
- "Lower learning rate (5e-4) improves SIREN convergence" — **FALSIFIED@B13.** LS=0.201 ≈
  ctrl 0.198. The stochastic outcome quality is unchanged. `B13`.
- "dur0=8 vs dur0=10 matters" — **PARTIALLY FALSIFIED@B13.** Mean LS is neutral (0.204 ≈ 0.198),
  but dur0=8 produced the best position-5 outcome (-0.03 vs ctrl -0.31). The effect is on
  SIREN basin selection, not mean quality. `B13`.

## Open questions

- **Is spatial gain + amp=10 converged at 2400it?** s0→s1 showed +0.059 in 1200it. The
  amp10 slot (LS=0.323) at 3600it could reach ~0.38+. **#1 priority (B15).**
- **Is SIREN ω=5 optimal for the gain field?** ω=5 was inherited from stiffness. Coarser
  (ω=3) or finer (ω=7) may suit the gain field differently. **#2 priority (B15).**
- **What is the dominant RESIDUAL BOTTLENECK with spatial gain?** The bottleneck dimensions
  may have shifted from size/chirality (uniform gain) to orientation/shape-detail. Residual
  decomposition script `run_decompose_b14.sh` created. **#3 priority.**
- **Can gain bounds [0.1, 2.5] be widened/narrowed?** Too narrow = insufficient contrast;
  too wide = overshoot risk. Analogous to stiffness range optimization.
- ~~**Spatial gain breaks the plateau?**~~ **ANSWERED (B14): YES.** LS 0.205→0.323 (+58%).
- ~~**amp=10 hurts at dur≈11?**~~ **OVERTURNED (B14): only with UNIFORM gain.** With spatial
  gain, amp=10 is BEST.
- ~~**Problem nodes are a physics model limitation?**~~ **OVERTURNED (B14): solvable with
  spatial gain.** ALL-positive at 3600it.

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 11 (2026-06-29):** dur_hi=12 → dur=10.0 → LS=0.200, FIRST EVER ALL-POSITIVE
  config. dur_hi=10 → dur=8.5 → LS=0.211 (best mean but 3 negatives). Goldilocks zone.
  Stiffness 3-5× MORE load-bearing at short dur. amp=10 HURTS at dur≈11 (uniform gain).
- **Batch 12 (2026-06-30):** The "all-positive" Goldilocks is STOCHASTIC (ctrl failed to
  reproduce). LS≈0.208 PLATEAU reached by two routes (durhi11@2400it, durhi12@3600it).
  Stiffness narrowing FALSIFIED in ALL configs. [80,300] is a hard stiffness optimum.
- **Batch 13 (2026-06-30):** SIREN capacity hypothesis FALSIFIED. hidden384 COLLAPSED
  (LS=0.146, Δ=-0.052); layers4 neutral; lr/depth/dur-init all neutral. The LS≈0.20
  plateau is a PHYSICS MODEL ceiling (uniform gain), not architecture. Spatial gain
  implemented (`--gain_src siren`).
- **Batch 14 (2026-07-01):** **BREAKTHROUGH — SPATIAL GAIN BROKE THE LS≈0.20 CEILING.**
  LS=0.323 (amp=10+sgain, +58% over ctrl 0.205). ALL spatial-gain+stiffness slots beat ctrl.
  amp=10 is BEST (OVERTURNS B11 amp×dur finding — 3-way interaction with gain type).
  Deep (3600it): ALL-positive nodes (LS=0.277). Stiffness remains load-bearing with sgain
  (0.125 without vs 0.323 with). Gain+stiffness = complementary spatial control.

---

## Current theme
### Current hypothesis
"Spatial gain (SIREN, ω=5) has BROKEN the LS≈0.20 ceiling (LS=0.323). The gain field is NOT
converged at 2400it (+0.059 from 2400→3600it). Deeper training of the amp=10 + spatial gain
configuration should push LS further. Simultaneously, the gain field's spatial frequency (ω)
and the gain bounds may not be optimal — ω was inherited from stiffness, not independently
tuned. The next bottleneck may have shifted from size/chirality to orientation/shape-detail."
### Iterations this theme
- Batches 1–12: scalar levers + SIREN architecture saturated at LS≈0.20 plateau.
- Batch 13: SIREN capacity hypothesis FALSIFIED. Spatial gain implemented.
- **Batch 14: SPATIAL GAIN CONFIRMED as ceiling-breaking mechanism (LS=0.323, +58%).**
  amp=10+sgain is best. Depth matters more with spatial gain. Stiffness still load-bearing.
- Batch 15: optimize spatial gain configuration (depth, ω, gain bounds, amp).
### Emerging observations
- **LS=0.323** is new best (B14 sgain_amp10). Per-node ceiling = 0.87 (B14 dur0_8).
- **Spatial gain NOT converged at 2400it** — +0.059 from 2400→3600it (B14 s0→s1).
- **amp=10 BEST with spatial gain** — OVERTURNS B11 (uniform gain: amp=10 hurts).
- **ALL-positive achievable** at 3600it with spatial gain (B14 deep).
- **Stiffness+gain are COMPLEMENTARY** — gain alone LS=0.125; together LS=0.323.
- Sensitivity: chirality ≈ size >> orientation > openness (engineering, regime-robust).
- **OPEN: is ω=5 optimal for gain SIREN?** Stiffness ω=5 confirmed; gain may differ.
- **OPEN: are gain bounds [0.1, 2.5] optimal?** Analogous to stiffness range.
**CRITICAL: this section must ALWAYS be at the END of the file.**

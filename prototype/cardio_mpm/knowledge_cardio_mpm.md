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

- **Under LoopScore (corrected metric, floor=0.02):** **LS = 0.196, SD=0.227** (B10 durhi15:
  stiff [80,300], ω=5, gain0=0.5, amp=12, dur_hi=15→dur=11.3, 2400it). The **very-short-
  duration regime (dur≈11)** TAMES the catastrophic node from LS=-1.00 to LS=-0.45, lifting
  the mean by +0.035 over the prior best. SD is higher (0.227 vs 0.187) because the per-node
  range widened (-0.45 to +0.51) — but the mean improvement is real.
- **Previous best:** LS=0.166 (B9 deep4800 at 4150it, dur→21.1) ≈ LS=0.165 (B9 dur0_10, 2400it).
- **Best 2400it config:** dur_hi=15, dur0=10, stiff [80,300], LS=0.196 (B10). Prior: 0.165.
- **Fibre-only (no stiffness):** LS ≈ 0.118 at amp=12 (B5). Stiffness adds ~0.03 net.
- **Under R² (diagnostic only):** best R² = −0.912 (gain0=0.3, fibre+gain+dur, 2400it).

## Established mechanisms  `[mechanism]` — causal, regime-conditional

1. **Loops are GENERIC in the active-stress MPM (inertial); structure TUNES morphology, it does not create
   it.** (2×2 test: isotropic loops ≈ structured.) ⇒ the target is loop *morphology* (size/axis/chirality/
   openness), and the forward mechanism is **active stress** (not body force, not rotary). `[engineering-ish
   + mechanism]`, robust.
2. **Gain is a size/overshoot lever** — a single learned global scalar. Lower gain reduces overshoot; at
   gain0=0.3 the mean LS is unchanged vs 0.5 but uniformity improves dramatically (SD 0.212→0.152).
   `[mechanism@LoopScore, 2400it, corrected metric]`. (R²-era: gain0=0.5 > 0.854; confirmed directionally.)
3. **Pulse duration landscape has THREE basins; very-short (dur≈11) is the current best.**
   The landscape: (a) dur≈30 (default init trap, LS≈0.06-0.12), (b) dur≈19-21 (intermediate,
   LS≈0.16), (c) **dur≈11 (best, LS=0.196, reachable via dur_hi=15).** The very-short regime
   TAMES the catastrophic node from LS=-1.00 to -0.45 because shorter pulses limit overshoot
   energy. This proves the catastrophe is an ENERGY OVERSHOOT, not a structural tissue defect.
   Duration INIT and dur_hi CEILING are first-class experimental variables: dur0=10 + dur_hi=15
   reaches dur≈11 at 2400it. Morphology: very-short dur has highest chirality (0.70) and lowest
   ampL (0.049); medium-dur (19-21) has lower chir (0.63-0.65), higher ampL.
   `[mechanism+optimization@LoopScore, 2400it, UPDATED@B10, OVERTURNED@B9-dur19-21]`.
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

## Optimization facts  `[optimization@<regime>]` — depth-dependent, never promote to mechanism

- **NEVER TRUST OPTIMIZATION STATE.** Many conclusions FLIPPED purely because training continued or
  init changed. B9 is a textbook case: "dur=24 is the interior optimum" (B8) was itself an optimization
  artifact — dur0=10 at 2400it reaches dur≈19.4, matching 4150it. Depth was only needed because the
  default init (dur0=14) trapped duration in the dur≈30 basin.
- **3600it improves over 2400it modestly in the dur≈19 regime.** With dur0=10, 3600it→LS=0.175
  (dur=18.8) vs 2400it→LS=0.161 (dur=19.3). The +0.014 benefit is real but small. However,
  dur_hi=15 at 2400it (LS=0.196) beats both — init constraints are more powerful than depth.
  `[optimization@LoopScore, B10, UPDATED from B9]`.
- **Gain viable range is [0.4, 0.5] with wide stiffness.** gain0=0.3 CATASTROPHIC (LS=-0.406, B3);
  gain0=0.7 CATASTROPHIC (LS=-0.272, B4); gain0=0.4 ≈ 0.5 (LS=0.139 vs 0.140, B6). Without
  stiffness or with narrow stiffness, gain0 is flexible. `[optimization@LoopScore-corrected, 2400it]`.
- **SIREN stiffness convergence is STOCHASTIC but REPRODUCIBLE with floor.** Without floor:
  identical config can produce 1 or 3 catastrophic nodes (LS range -0.208 to +0.159). With floor=80:
  all runs have exactly 1 outlier, LS reproducible at ~0.135-0.140. B4+B5 controls reproduced:
  LS=0.149, 0.137. `[optimization@LoopScore, 2400it, ω=5]`.
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
- "amp=10 differs from amp=12" — **FALSIFIED@LoopScore, 2400it, stiff [80,300].** LS=0.150 vs 0.151,
  identical per-node patterns. Amplitude is flat in [10,12]. `B7`.
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

## Open questions

- **What is the dominant BOTTLENECK dimension across nodes?** Need residual decomposition to
  quantify (scripts ready: `run_decompose_b10.sh`). SIZE and CHIRALITY are the top-sensitivity
  dimensions (engineering). **#1 priority.**
- **Is dur≈11 the floor, or does even shorter duration improve LS?** dur_hi=12 and dur_hi=10
  are untested. Below some floor, pulse energy may be too low for any loop. **#2 priority.**
- **Does depth (3600it) help in the dur≈11 regime?** At dur≈19, depth gave +0.014. In the
  dur≈11 regime the benefit may differ. **#3 priority.**
- **Can the remaining catastrophic node (LS=-0.45 at dur≈11) be further tamed?** Reducing
  amplitude from 12→10 or adjusting stiffness topology might help. The catastrophe is now
  confirmed as an ENERGY OVERSHOOT mechanism.
- **Does amp=10 + dur_hi=15 synergize?** amp=10≈12 was established at dur≈19; in the dur≈11
  regime, the interaction may differ.
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — future option.
- ~~**Can wl=35 improve over wl=28.8?**~~ **ANSWERED (B10): NO.** wl=35 neutral (LS=0.165).
- ~~**Is dur≈19-21 the GLOBAL optimum?**~~ **ANSWERED (B10): NO.** dur≈11 (via dur_hi=15) gives
  LS=0.196, +18% improvement. Three basins discovered.
- ~~**Does dur0=10 + 3600it improve?**~~ **PARTIALLY ANSWERED (B10):** 3600it gives LS=0.175
  (dur→18.8) vs 2400it 0.161 (dur→19.3). Modest +0.014 gain. But dur_hi=15 at 2400it beats it.

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 7 (2026-06-28):** SIREN fibre WITHOUT stiffness is WORSE (LS=-0.222). SIREN fibre is
  intrinsically destabilizing. Stiffness STABILIZES. amp=10 ≈ amp=12. SIREN fibre CLOSED.
- **Batch 8 (2026-06-28):** 3600it BREAKS the LS≈0.15 plateau (LS=0.162) — duration escapes
  dur≈30 basin to interior optimum dur=24. drag_k=50≈30 but 20 HURTS. w_amp IS load-bearing.
  stiff_hi=400 catastrophic. Duration saturation was an optimization-depth artifact.
- **Batch 9 (2026-06-29):** dur0=10 at 2400it (LS=0.165) MATCHES 4150it (LS=0.166). True
  duration optimum is ~19-21, not 24. Duration init substitutes for depth. gain0 FLAT at 3600it.
- **Batch 10 (2026-06-29):** dur_hi=15 → dur=11.3 → **LS=0.196 (NEW BEST, +18%).** THIRD
  duration basin discovered at dur≈11. Very-short pulses TAME the catastrophic node from
  LS=-1.00 to -0.45 by limiting overshoot energy. fibre_angle=0.5 TRAPS duration at dur≈28
  even with dur0=10 (fibre×duration cross-interaction). wl=35 neutral. 3600it gives modest
  +0.014 in dur≈19 regime but can't compete with dur_hi=15's basin change.

---

## Current theme
### Current hypothesis
"The very-short-duration regime (dur≈11, via dur_hi=15) broke the LS plateau by TAMING
the catastrophic node. The catastrophe is an ENERGY OVERSHOOT: shorter pulses limit the
energy available for recoil. The question is whether dur≈11 is the floor of this basin
or whether even shorter duration (dur_hi=12 or 10) continues to improve. Secondary:
can depth (3600it) or amplitude reduction (amp=10) further improve the dur≈11 regime?"
### Iterations this theme
- Batch 1–7: all continuous levers mapped; SIREN fibre CLOSED; best at 2400it: LS≈0.15.
- Batch 8: depth broke the plateau via duration (LS=0.162). Drag/w_amp/stiff bounds mapped.
- Batch 9: CONFIRMED duration-init as the mechanism. True optimum dur≈19-21. LS=0.166.
- Batch 10: dur_hi=15 → dur=11.3 → LS=0.196 (NEW BEST). Third duration basin. Catastrophe
  tamed (-1.00→-0.45). fibre×duration interaction discovered. wl=35 neutral.
- Batch 11: explore the dur≈11 regime — shorter, deeper, amplitude, stiffness topology.
### Emerging observations
- **NEW BEST: LS=0.196** (dur_hi=15, dur=11.3, 2400it).
- Duration landscape has THREE basins: dur≈30 (trap), dur≈19-21 (intermediate), dur≈11 (best).
- The catastrophic node is an ENERGY OVERSHOOT: shorter pulses tame it (-1.00→-0.45).
- fibre_angle=0.5 traps duration at dur≈28 even with dur0=10 — fibre×duration interaction.
- Very-short-dur regime: highest chirality (0.70), lowest ampL (0.049).
- The dur≈11 regime's SD (0.227) is higher than dur≈19 (0.187) — room for uniformity improvement.
- Sensitivity: chirality ≈ size >> orientation > openness (engineering, regime-robust).
- Scalar levers (gain, amp, drag, w_amp) are FLAT in viable ranges — duration ceiling is the
  new active lever.
**CRITICAL: this section must ALWAYS be at the END of the file.**

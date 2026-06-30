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

- **Under LoopScore (corrected metric, floor=0.02):** **LS = 0.211, SD=0.234** (B11 durhi10:
  stiff [80,300], ω=5, gain0=0.5, amp=12, dur_hi=10→dur=8.5, 2400it). Highest mean LS, but
  3 negative nodes (-0.36, -0.25, -0.06). Driven by high ceiling (0.71).
- **Best UNIFORM config (ZERO negative nodes):** **LS = 0.200, SD=0.216** (B11 durhi12:
  dur_hi=12→dur=10.0). FIRST config ever with ALL per-node LS ≥ 0 (min=0.00, max=0.58).
  The persistent catastrophic node is ELIMINATED. This proves the catastrophe is purely
  energy overshoot, not a structural tissue defect.
- **Previous best:** LS=0.196 (B10 durhi15, dur→11.3), LS=0.166 (B9).
- **Fibre-only (no stiffness):** LS ≈ 0.118 at amp=12 (B5). Stiffness adds ~0.10 at dur≈11.
- **Under R² (diagnostic only):** best R² = −0.912 (gain0=0.3, fibre+gain+dur, 2400it).

## Established mechanisms  `[mechanism]` — causal, regime-conditional

1. **Loops are GENERIC in the active-stress MPM (inertial); structure TUNES morphology, it does not create
   it.** (2×2 test: isotropic loops ≈ structured.) ⇒ the target is loop *morphology* (size/axis/chirality/
   openness), and the forward mechanism is **active stress** (not body force, not rotary). `[engineering-ish
   + mechanism]`, robust.
2. **Gain is a size/overshoot lever** — a single learned global scalar. Lower gain reduces overshoot; at
   gain0=0.3 the mean LS is unchanged vs 0.5 but uniformity improves dramatically (SD 0.212→0.152).
   `[mechanism@LoopScore, 2400it, corrected metric]`. (R²-era: gain0=0.5 > 0.854; confirmed directionally.)
3. **Pulse duration controls a MONOTONIC LS–UNIFORMITY TRADEOFF with a Goldilocks zone at dur≈10.**
   Four regimes mapped: (a) dur≈30 (init trap, LS≈0.06-0.12); (b) dur≈19-21 (intermediate,
   LS≈0.16); (c) **dur≈10 (Goldilocks, LS=0.200, ZERO negative nodes — ALL ≥ 0);** (d) dur≈8.5
   (highest mean LS=0.211 but 3 negative nodes re-introduced). Shorter duration raises the ceiling
   for good nodes but eventually breaks uniformity: dur=10 is the ONLY regime with zero catastrophes.
   The persistent catastrophic node progressed: LS=-1.00 (B1-B9) → -0.45 (B10, dur≈11) → **0.00
   (B11, dur≈10) → GONE.** This proves the catastrophe is purely ENERGY OVERSHOOT. Duration INIT
   and dur_hi CEILING are first-class variables. `[mechanism@LoopScore, 2400it, UPDATED@B11]`.
4. **Amplitude×duration INTERACTION: amp=10≈12 at dur≈19, but amp=10 HURTS at dur≈11.**
   At dur≈19: amp=10 ≈ amp=12 (LS=0.150 vs 0.151, B7). At dur≈11: amp=10 drops LS from 0.191
   to 0.184 and worsens the catastrophic node (-0.19 → -0.47, B11). Short pulses need full
   amplitude=12 to avoid energy starvation in stiff regions while soft regions overshoot.
   amp=14 remains catastrophic (LS=-0.247, B4). Keep amplitude=12 at dur≈10.
   `[mechanism@LoopScore, 2400it, UPDATED@B11, amp×dur interaction]`.
5. **Fibre co-learning is LOAD-BEARING under LoopScore** — freezing fibre drops LS from 0.119→0.088
   (Δ=−0.031). This OVERTURNS the R²-era finding (fibre hurt at depth under R²). The parametric fibre
   provides orientation structure that the LS per-node gradient rewards. `[mechanism@LoopScore, 2400it]`.
6. **Coarse SIREN stiffness is CRITICAL — 3-5× MORE load-bearing at short duration.** SIREN (ω=5)
   converges to a binary spatial pattern. At dur≈19, stiffness adds ~0.02-0.03 LS. At **dur≈11,
   stiffness adds ~0.10 LS** (0.191→0.092 without it; ampL spikes 0.048→0.272, B11). The stiffness
   spatial pattern controls where energy goes when pulse energy is limited — without it, the short-
   pulse regime is catastrophic. Stiffness FLOOR (80-100) prevents multi-node catastrophe. Wider
   range [80,300] is beneficial. ω=5 confirmed sweet spot. The stiffness×duration interaction is
   the dominant remaining lever for improving uniformity.
   `[mechanism@LoopScore, 2400it, ω=5, UPDATED@B11, stiff×dur interaction]`.
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
- **3600it barely helps at dur≈11 (+0.007) and dur≈19 (+0.014).** At dur≈11: 3600it→LS=0.198
  vs 2400it→LS=0.191 (B11). At dur≈19: 3600it→0.175 vs 2400it→0.161 (B10). Init constraints
  (dur_hi ceiling) are far more powerful than depth in the short-duration regime.
  `[optimization@LoopScore, B11, UPDATED from B10]`.
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
- "amp=10 differs from amp=12" — **FALSIFIED at dur≈19 (B7: 0.150≈0.151); OVERTURNED at dur≈11
  (B11: 0.184 < 0.191).** Amplitude×duration interaction: amp=10 HURTS at short duration (catastrophic
  node worsens -0.19→-0.47). Amplitude is flat only at longer durations. `B7+B11`.
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
  quantify (scripts ready: `run_decompose_b11.sh`). SIZE and CHIRALITY are the top-sensitivity
  dimensions (engineering). **#1 priority.**
- **Can we get BOTH high LS AND zero negatives?** dur=10 (durhi12) eliminates all negatives
  (LS=0.200); dur=8.5 (durhi10) has best LS=0.211 but 3 negatives. Can depth (3600it in
  durhi12) push mean LS above 0.211 while keeping all-positive? **#2 priority.**
- **Does narrowing stiffness contrast extend the Goldilocks zone to shorter duration?**
  At dur=8.5, soft regions overshoot while stiff regions are fine. If stiff_hi is reduced
  (less contrast), can durhi10 keep its high mean while eliminating negatives? **#3 priority.**
- **Is dur_hi=11 better than dur_hi=12?** Squeezing the ceiling further might find a
  tighter Goldilocks zone (dur≈9-10) with higher mean but still zero negatives.
- Multiscale LoopScore (`K∈{1,2,4,8}` weighted) — future option.
- ~~**Is dur≈11 the floor?**~~ **ANSWERED (B11): NO.** dur=8.5 gives LS=0.211. BUT dur=10 is
  a UNIFORMITY optimum (zero negatives).
- ~~**Does depth help at dur≈11?**~~ **ANSWERED (B11): BARELY.** +0.007, marginal.
- ~~**Does amp=10 help at dur≈11?**~~ **ANSWERED (B11): NO.** amp=10 HURTS (0.184 vs 0.191).
- ~~**Can the catastrophic node be tamed?**~~ **ANSWERED (B11): YES.** dur_hi=12 → dur=10.0
  ELIMINATES it completely (LS=0.00, first ever all-positive config).

---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 8 (2026-06-28):** 3600it BREAKS the LS≈0.15 plateau (LS=0.162) — duration escapes
  dur≈30 basin to interior optimum dur=24. drag_k=50≈30 but 20 HURTS. w_amp IS load-bearing.
  stiff_hi=400 catastrophic. Duration saturation was an optimization-depth artifact.
- **Batch 9 (2026-06-29):** dur0=10 at 2400it (LS=0.165) MATCHES 4150it (LS=0.166). True
  duration optimum is ~19-21, not 24. Duration init substitutes for depth. gain0 FLAT at 3600it.
- **Batch 10 (2026-06-29):** dur_hi=15 → dur=11.3 → LS=0.196 (NEW BEST, +18%). THIRD
  duration basin discovered at dur≈11. Very-short pulses TAME the catastrophic node from
  LS=-1.00 to -0.45. fibre_angle=0.5 TRAPS duration at dur≈28.
- **Batch 11 (2026-06-29):** dur_hi=12 → dur=10.0 → **LS=0.200, FIRST EVER ALL-POSITIVE
  config** (zero negative per-node LS). dur_hi=10 → dur=8.5 → LS=0.211 (new best mean but
  3 negatives). Goldilocks zone at dur≈10: shorter raises ceiling but breaks uniformity.
  Stiffness 3-5× MORE load-bearing at short dur (0.10 vs 0.02 ΔLS). amp=10 HURTS at dur≈11.
  3600it marginal (+0.007).

---

## Current theme
### Current hypothesis
"The dur=10 regime (durhi12) occupies a GOLDILOCKS ZONE: pulse energy too low for overshoot
at ANY node, yet sufficient for all nodes to form loops. This zone depends on the stiffness
contrast — wider ranges narrow the zone (soft regions still overshoot), narrower ranges may
widen it. The path to best-of-both-worlds (high LS + zero negatives) is either: (1) deepen
durhi12 to push its mean LS above 0.211, or (2) narrow stiffness in durhi10 to eliminate its
negatives. Stiffness topology, not duration or amplitude, is the dominant remaining lever."
### Iterations this theme
- Batch 1–7: all continuous levers mapped; SIREN fibre CLOSED; best at 2400it: LS≈0.15.
- Batch 8: depth broke the plateau via duration (LS=0.162). Drag/w_amp/stiff bounds mapped.
- Batch 9: CONFIRMED duration-init as the mechanism. True optimum dur≈19-21. LS=0.166.
- Batch 10: dur_hi=15 → dur=11.3 → LS=0.196. Third duration basin. Catastrophe tamed.
- Batch 11: dur_hi=12 → dur=10.0 → LS=0.200 (ZERO negatives!). dur_hi=10 → dur=8.5 →
  LS=0.211 (best mean but 3 negatives). Goldilocks zone at dur≈10. Stiffness 3-5× more
  load-bearing at short duration. amp=10 HURTS at dur≈11. 3600it marginal.
- Batch 12: exploit the Goldilocks zone — depth, stiffness topology, squeezed ceiling.
### Emerging observations
- **NEW BEST MEAN: LS=0.211** (dur_hi=10, dur=8.5, 2400it). But 3 negative nodes.
- **FIRST ALL-POSITIVE: LS=0.200** (dur_hi=12, dur=10.0). Zero negatives, min=0.00.
- Duration-uniformity TRADEOFF: shorter dur → higher ceiling, lower floor. dur=10 is sweet spot.
- Catastrophe FULLY CONTROLLED: -1.00 (B1-B9) → -0.45 (B10) → 0.00 (B11) → GONE.
- Stiffness is 3-5× more load-bearing at dur≈10 than dur≈19 (ΔLS: 0.10 vs 0.02).
- amp=10 HURTS at dur≈10 (energy starvation in stiff regions).
- Sensitivity: chirality ≈ size >> orientation > openness (engineering, regime-robust).
- Scalar levers (gain, amp, drag, w_amp) are saturated — stiffness topology is the next frontier.
**CRITICAL: this section must ALWAYS be at the END of the file.**

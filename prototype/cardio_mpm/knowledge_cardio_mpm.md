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

> **Confidence notation (project-wide):** prefix EVERY claim with **✓ established** (dose-confirmed: monotone
> ladder or independent reproduction) · **◐ provisional** (single-draw high or untested hypothesis — cannot
> close an axis/question) · **✗ open|refuted** (no mechanism yet, or a dose-confirmed negative). This composes
> with the `[class@regime]` tag (e.g. `✓ [mechanism@LoopScore,2400it]`). Promote ◐→✓ on dose confirmation;
> demote ✓→◐ if a frozen axis regresses. On each distill (step 12), add/refresh the marker on every line.

## Current objective  (Phase 2 — see `current_phase.txt`)

**Identify the minimal physical mechanism controlling each remaining trajectory morphology axis** —
magnitude, enclosure, direction, shape, uniformity, size — thereby **completing the causal decomposition of
LoopScore** before entering generative trajectory discovery (Phase 3). LoopScore is the operational metric;
the deliverable is a mechanism *per axis* (solved OR demonstrated structurally limited — a dose-confirmed null
result counts). Confidence: **✓ established** (dose-confirmed) · **◐ provisional** (single-draw/hypothesis, not
closable) · **✗ open**. Axis status: magnitude ✓ (gain+stiffness) · enclosure ✓ (rotation) · direction ✓ ·
shape ✓ · uniformity ✓ (fibre heterogeneity) · **SIZE ◐ (REOPENED@B33 — the GAIN CEILING + fibre_dev raise
peak_ratio 0.49→0.53 @rot1.0; single-draw, B34 replicates)**. **FREEZE solved axes:** every experiment must improve SIZE *while preserving* the ✓ axes (a slot that
lifts size but regresses enclosure/chirality is not progress; the regressed axis reverts to ◐). Phase 2
completes only when SIZE is ✓ AND no ✓ axis has regressed; then advance `current_phase.txt` → `PHASE3`.

## Current best result

- **Under LoopScore (corrected metric, floor=0.02): LS = 0.509** (B33 ghi20 = dev18 + gain_hi 1.5→**2.0**,
  CONVERGED@2399it) — NEW RECORD, ALL-POSITIVE nodes (min +0.20), best peak_ratio (0.533) + area_ratio (0.387) +
  energy (0.873) of the batch. **SIZE-IN-ROTATION RESOLVED: the GAIN CEILING is a per-region size+uniformity lever
  @rot1.0** — raising gain_hi lets the SIREN gain field push harder LOCALLY at the small radial-stub nodes, and
  because the axis rotates that extra local drive becomes loop AREA (peak 0.50→0.53) not overshoot. `fibre_dev`
  0.18→0.20 (dev20, LS=0.505) reproduces the lift; base amplitude (amp12/amp14) and floor softening (slo20) do NOT
  (global drive still overshoots, #4/#25). This OVERTURNS fact #28 ("gain_hi>1.5 buys nothing") as regime-bound to
  rot=0. **SINGLE DRAW — B34 replicates (campaign law: 9+ single-draw clean points have regressed).** See #28
  (overturned), #31 (size reopened), #30 (rotation is the enclosure channel that gates this).
- **Prior record: LS = 0.492** (B32 dev18 = rot1.0 + soft-floor stiff[30,300] +
  SIREN fibre_dev 0.18, CONVERGED@2399it) — batch best, best area_ratio (0.354) + chir_match (0.853). This is the
  PEAK of a DOSE-CONFIRMED fibre lever, not a lottery draw: the fibre_dev ladder rises MONOTONE — dev0.05=0.447,
  dev0.08=0.465, dev0.12=0.473, dev0.18=0.492 — then rolls off at dev0.25=0.482 (peak dev~0.18–0.20). **The prior
  "LS=0.493 fdev12" was a SINGLE-DRAW HIGH: its B32 replicate REGRESSED to 0.473 (9th single-draw regression) — but
  the independent monotone dose ladder RESCUED the lever (fibre heterogeneity is REAL, see #5).** fibre_wl 28.8→20
  (finer SCALE) = 0.473 ≡ fdev12 → the win is dev MAGNITUDE, not spatial scale. New provisional OPERATING POINT =
  dev18. **RESIDUAL FLIPPED BACK TO SIZE:** rotation SOLVED enclosure (loopiness_ratio 1.06–1.18, at/ABOVE real),
  so the dominant gap is now MAGNITUDE — peak_ratio ≈0.49 (sim peak = half real), area_ratio ≈0.35; dashboard red
  loops are loopy+correctly-chiral but sit INSIDE green. See #5 (fibre dose-confirmed + scale-inert), #30 (rot
  saturates at 1.0), #31 (residual=size in the rotating regime; size-in-rotation is the LIVE B33 question).
- **B31 slo30 = 0.475 was a LOTTERY HIGH DRAW — did NOT reproduce (B32 replicate 0.453).** The B31 stiffness-floor
  ladder (slo20/30/40 = 0.456/0.453/0.459) is FLAT within noise: softening is inert @rot1.0 (see #31).
- **Prior single-draw record: LS = 0.481** (B28 rot10, CONVERGED@2399it) —
  the rotating-contraction-axis breakthrough. Config = the b27 record family (stiff[50,300] ω5, drag40, amp10,
  gain[0.2,1.5], fibre-ON dev005, dur_hi11, substeps10) PLUS `--rot_stress 1.0` (contraction axis swings
  θ(x,y)+1.0·sin(2π(fr−onset)/period) over the beat). Lifts LS **+0.11 / +30% over the 2-month 0.365 ceiling** by
  FILLING the area-enclosure residual, not by adding force: the rot dose ladder (0→0.3→0.6→1.0) is MONOTONE in LS
  (0.332→0.430→0.461→0.481), area_ratio (0.100→0.189→0.284→0.360, 3.6×), loopiness_ratio (0.424→0.712→0.992→1.107,
  hits real at rot0.6), and minor_axis_ratio (0.446→0.794). Dashboards: rot0 red loops are thin radial stubs inside
  green; rot1.0 red loops are FAT closed ellipses superposing on green (per-node LS up to +0.87). **SINGLE DRAW —
  B29 replicates (campaign law: single-seed clean points routinely regress).** See fact #30 (ESTABLISHED) + #29.
- **Prior record (rot-OFF radial regime): LS = 0.365** (B23 slo50, CONVERGED@2400it).
  Config: stiff[50,300] (SOFTENED floor, SIREN ω=5) + narrow gain[0.2,1.5] + SIREN fibre dev=0.05,
  gain0=0.5, amp=10, drag=40, dur_hi=11→dur≈9.4. ZERO negative nodes; **lowest overshoot ever ampL=0.004**;
  learned gain field HALVED vs stiff[80,300]. Beats the old 0.358 fibre-lottery peak with LOW variance.
  **B24 caveat: the record did NOT reproduce** — the exact-config ctrl replicate came in at LS=0.343 (1 neg
  node); drag50 (0.354) and wide50_400 (0.340) also sit in the [0.34,0.365] fibre-SIREN stochastic band. So
  0.365 is a HIGH draw of the same family, not a stable point; treat the family center as ≈0.35±0.02.
  **B26 UPDATE: the 0.365 recurred at `stiff_hi400` (wide400) with ampL=0.002 — the LOWEST overshoot ever
  recorded (below B23's 0.004), zero negatives.** But B26 ctrl drew LOW (0.319), so the batch spread
  (0.308–0.365) is again dominated by the fibre-SIREN lottery (±0.05); wide400's LS may be a high draw, but its
  ampL=0.002 is a genuine shape signal — high-ceiling contrast + soft floor + drag40 is the cleanest-overshoot
  regime. stiff_hi400 (with the soft floor) is the current best config to reconfirm.
  **B27 RECONFIRMS: hi400 drew LS=0.369, ampL=0.001 (lowest overshoot ever), zero negatives — the 0.365/0.369
  wide400 family is a stable, reproducible high draw; stiff_hi300 ctrl 0.360 alongside.**
- **RESIDUAL MORPHOLOGY (2026-07-04 independent audit, record wide400) — the residual is INSUFFICIENT
  CIRCULATION, not insufficient displacement.** Decomposed on the `mov` FIT nodes as sim | real | ratio
  (ratio = sim/real, 1.0 = perfect; real is the fixed data target, ~constant across runs — keeping raw sim
  AND real makes any change in the ratio attributable):

  | axis | metric | sim | real | ratio |
  |------|--------|-----|------|-------|
  | magnitude | energy (√Σdisp², total work) | 0.63 | 0.66 | **0.955** |
  | magnitude | peak excursion | — | — | 0.59 |
  | enclosure | \|enclosed area\| | — | — | **0.16** |
  | enclosure | loopiness (\|area\|/bbox) | 0.15 | 0.34 | 0.44 |
  | direction | chirality sense match | — | — | **0.81** |
  | shape | minor-axis var frac (λ₂/(λ₁+λ₂)) | 0.17 | 0.39 | **0.44** |

  Reading: the tissue does about the **right amount of work** (energy 0.95) and mostly the **right rotation
  sense** (0.81), but the motion **collapses onto one principal axis instead of exploring two** (minor-axis
  0.17 vs 0.39) so it **encloses ~1/6 the area** (0.16). Right magnitude coexists with wrong enclosure — the
  `size`/`ampL` diagnostics the loop optimized are MAGNITUDE only and were blind to this. The old "lift is
  overshoot reduction, not size" story is a MAGNITUDE statement; the real gap is CIRCULATION. See fact #24
  (tempered) and #29 (root cause). The loop now reports this full sim|real|ratio decomposition per checkpoint
  (`enclosure_row` → `progress.txt` RESIDUAL_MORPHOLOGY block); reproduce standalone via `audit_trajectories.py`
  on a `--eval_dump` npz.
- **Prior peak: LS = 0.358** (B17 gnarr+fdev01) — a high draw of the fibre-SIREN lottery (family band
  [0.257, 0.358], ±0.05). B23 slo50 supersedes it as a CONVERGED, reproducible-family record.
- **Prior CONVERGED family (B21+B22@2400it):** drag40/amp10 fibre-ON, stiff[80,300] sits at LS≈0.32–0.34.
- **ALL-POSITIVE config:** LS=0.312 (B15 deep4800, ~3950it). ALL nodes positive (worst=0.17).
  With fibre SIREN: deep (4100it) gives LS=0.305, only 1 negative (node 1, -0.49; B17).
  **NEW (B18@1100it):** gnarr+fdev005 ALL-POSITIVE at 1100it (LS=0.279, SD=0.267) — needs
  convergence confirmation.
- **Previous plateaus:** LS ≈ 0.31 (spatial gain, no fibre SIREN, B15); LS ≈ 0.20 (uniform gain, B12–B13).
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
5. **Fibre co-learning is LOAD-BEARING under LoopScore — CONFIRMED AT CONVERGENCE.** Freezing fibre
   drops LS: B14 shallow 0.119→0.088 (Δ=−0.031); **B21@2400it converged: fibre-ON 0.320 vs
   fibre-OFF 0.241 (Δ=+0.079), fibre-OFF also worst overshoot (ampL 0.098)**. This OVERTURNS the
   R²-era finding (fibre hurt at depth under R²). The parametric fibre provides orientation
   structure that the LS per-node gradient rewards, and the effect GROWS with depth.
   **ELEVATED by the 2026-07-04 audit — FIBRE CONTROLS AREA-ENCLOSURE, not displacement magnitude.**
   Ablating fibre (`nofibre` dump) does NOT mainly shrink the motion; it FLATTENS it: enclosed-area ratio
   collapses 0.16→0.087 (median sim/real |area|) and the loops go MORE degenerate/radial, while total energy
   only eases (0.955→0.688).
   That is exactly what an anisotropic constitutive law should do — set how displacement is DISTRIBUTED across
   directions, not how much there is. This makes fibre architecture the leading lever for the real residual
   (enclosure, fact #29), and reframes "fibre is load-bearing" as "fibre is the directionality/enclosure
   channel." **B31 CONFIRMS fibre stays load-bearing UNDER ROTATION:** dropping fibre from `--learn` at rot=1.0
   (nofib) is the WORST B31 slot (LS=0.394, area_ratio 0.230 — lowest) — the rotating axis does NOT make the
   spatial fibre redundant. **B32 ELEVATES fibre to the LIVE LS LEVER under rotation — DOSE-CONFIRMED:** the SIREN
   fibre_dev ladder is MONOTONE in LS (dev 0.05/0.08/0.12/0.18 → 0.447/0.465/0.473/0.492, rolling off at 0.25→0.482),
   so more per-region axis variation genuinely rescues the radial-node uniformity residual under the rotating axis
   (peak dev~0.18–0.20; the quiver gets wavier). REAL, not lottery: the prior single "fdev12=0.493" REGRESSED to 0.473
   on replicate, but the independent dose ladder validated the lever. **The win is dev MAGNITUDE, not spatial SCALE:
   fibre_wl 28.8→20 (finer) = 0.473 ≡ fdev12 replicate (scale-inert).**
   `[mechanism@LoopScore, 2400it converged, B14+B21+B31+B32; dose-confirmed@B32; enclosure/directionality role from 2026-07-04 audit + B32]`.
6. **Coarse SIREN stiffness is CRITICAL, and WIDER CONTRAST WINS @drag40 — [80,300] was NOT the hard
   optimum; the RANGE IS DRAG-GATED.** SIREN (ω=5) converges to a binary spatial pattern; stiffness adds
   ~0.10 LS. NARROWING [100,200] and floor-RAISING [100,300] both HURT (B12) — contrast is essential.
   **B23 OVERTURN (@drag40):** WIDENING helps in BOTH directions — floor-LOWERING stiff_lo 80→50 gives
   LS 0.332→0.365 (RECORD, ampL 6× down, gain field halved), and ceiling-RAISING stiff_hi 300→400 gives
   +0.022 (but reintroduces 1 negative node). Floor optimum ≈50 (slo40→0.350; B24 slo30→0.321, 3 negs — softer
   than 50 HURTS uniformity). The [80,300]
   "hard optimum" (fact-6 pre-B23) was an artifact of testing ONLY narrowing/floor-raising @drag30. The
   soft floor works ONLY with drag40 damping (slo50@drag30 = 0.333, benefit gone — fact #10, #26). ω=5
   confirmed. SIREN convergence is STOCHASTIC per-node.
   `[mechanism@LoopScore, 2400it, ω=5, OVERTURNED@B23 — range is drag-gated, softer floor wins @drag40]`.
7. **Stiffness × duration interaction is DESTRUCTIVE.** Longer pulses (dur_hi=40) are tolerable with
   uniform stiffness (LS=0.117) but catastrophic with spatial stiffness (LS=-0.070). Soft regions
   amplify the extra pulse energy into runaway overshoot. Keep dur_hi=30 when stiffness is active.
   `[mechanism@LoopScore, 2400it]`.
8. **SIREN fibre dθ is BENEFICIAL at dev∈[0.05,0.15], NOT intrinsically destabilizing.**
   Pre-spatial-gain (B6+B7): dev=0.3 catastrophic. With spatial gain: dev=0.3 NEUTRAL,
   dev=0.2 NEUTRAL, dev=0.1 BENEFICIAL (LS=0.345, B16), dev=0.05 BENEFICIAL (LS=0.332, B17),
   dev=0.15 BENEFICIAL (LS=0.311, B17). **B17 UPDATE:** the dev dose-response within
   [0.05, 0.15] is FLAT — all values produce LS within the stochastic band (~±0.05).
   The critical threshold is dev≤0.15 (helps) vs dev≥0.2 (neutral). Spatial gain STABILIZES
   the fibre SIREN. The original "CLOSED" conclusion was DOSAGE-limited and CONTEXT-limited.
   `[mechanism@LoopScore, 3600it, dev∈[0.05,0.15], ω=5, spatial gain, B16+B17, OVERTURNS B6+B7]`.
9. **Gain is FLAT in [0.4, 0.5] with stiffness — confirmed at 3600it.** gain0=0.4 ≈ gain0=0.5
   at both 2400it (LS=0.139 vs 0.140, B6) and 3600it (LS=0.159 vs 0.160, B9). Combined with
   gain0=0.3 catastrophic (B3) and gain0=0.7 catastrophic (B4) with wide stiffness, the viable
   gain window is 0.4–0.5 and does NOT differentiate at any tested depth.
   `[mechanism@LoopScore, 2400-3600it, stiff [80,300], B6+B9]`.
10. **Drag has an ASYMMETRIC floor at ~30; above it, drag is a mild OVERSHOOT lever, FLAT on mean LS
    across [30,50].** B8 (pre-spatial-gain): drag_k=50 ≈ 30, drag_k=20 HURTS. B21: drag_k=40
    LS-neutral vs 30 but HALVES overshoot (ampL 0.017 vs 0.027). **B22 UPDATE: drag_k=50 confirms the
    plateau — LS=0.333 ≈ drag40 (within noise), lowest overshoot tier (ampL 0.014).** So drag∈[40,50]
    is the low-overshoot envelope, all LS-equivalent. Drag buys overshoot headroom, not mean LS, and
    (B22) that headroom does NOT convert into bigger loops (see fact #24). **B23 UPGRADE: drag40's overshoot
    headroom is what UNLOCKS the wider-stiffness-contrast regimes — softer floor (stiff_lo50) and higher
    ceiling (stiff_hi400) both help @drag40 but NOT @drag30 (softening benefit vanishes: 0.333 vs 0.365).
    So drag40 is a load-bearing ENABLER of the B23 record, even though it is LS-neutral on its own.**
    `[mechanism@LoopScore, 2400it, B8+B21+B22+B23]`.
11. **w_amp (anti-collapse penalty) is LOAD-BEARING at 0.3.** w_amp=0 drops LS to 0.132 with
    much worse SD (0.272 vs 0.181). Without it, vulnerable nodes lose motion. w_amp=0.6 also
    HURTS (B4). The optimal w_amp is ~0.3 (neither 0 nor higher). `[mechanism@LoopScore, 2400it, B8]`.
12. **stiff_hi=400 is NOT an intrinsic ceiling — it was DRAG/OVERSHOOT-limited.** @drag30 (B8) it was
    catastrophic (LS=0.076, 5/9 negative). **B23 OVERTURN @drag40:** stiff_hi=400 gives LS=0.354 (+0.022
    over ctrl), only 1 negative node. The extra high-stiff contrast is tolerable once drag40 damps the
    soft-region recoil. So the upper contrast bound is set by DAMPING, not a fixed value.
    `[mechanism@LoopScore, 2400it, OVERTURNED@B23 — drag-gated, not an intrinsic ceiling]`.
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
16. **Spatial gain converges by ~3600it; depth beyond that improves UNIFORMITY not mean.**
    B14: 2400→3600it gave +0.059 (at amp=12). B15: 3600it=0.313, 3950it=0.312 (flat at amp=10).
    However, 3950it achieved ALL-POSITIVE nodes (node 5: 0.00→0.46) — depth rescues the weakest
    nodes without improving the tissue average. The remaining LS gap is node-specific.
    `[optimization@LoopScore, B14-B15, spatial gain, 2400-4800it, UPDATED@B15]`.
17. **Gain SIREN ω=5 is confirmed optimal (ω=3 HURTS).** ω=3 (coarser gain field) drops LS
    from 0.313 to 0.218 (Δ=-0.095), with 3 negatives and worst uniformity (SD=0.341). The
    gain field needs the SAME spatial resolution as stiffness (ω=5). ω parallels between
    gain and stiffness SIRENs: both need ω=5. `[mechanism@LoopScore, 2400it, B15]`.
18. **amp=10 confirmed as HARD OPTIMUM with spatial gain.** amp=11 drops LS from 0.313 to
    0.271 (Δ=-0.042, 2 negatives). Combined with B14 (amp=12 worse): the amplitude ceiling
    is SHARP at 10. With spatial gain, the gain field provides per-region amplitude control,
    so the base amplitude should be LOW to avoid overshoot. `[mechanism@LoopScore, 2400it, B15]`.
19. **dur_hi=11 is the OPTIMAL constraint.** dur_hi=13 drops LS from 0.313 to 0.238 (dur
    settled at 11.5 vs 10.0). The longer pulse allows overshoot at vulnerable nodes (node 5:
    LS=-0.64). The duration CONSTRAINT is the mechanism — it limits pulse energy globally.
    `[mechanism@LoopScore, 2400it, B15]`.
20. **Spatial gain STABILIZES the stiffness field AND the fibre SIREN.** Without spatial gain,
    the stiffness SIREN fragments into noisy high-frequency; with spatial gain, clean binary.
    Similarly, fibre SIREN dθ at dev=0.3 is catastrophic pre-sgain (B6+B7) but neutral with
    sgain (B16). The gain channel absorbs amplitude variation, freeing stiffness for elastic
    structure and fibre for orientation. `[mechanism@LoopScore, 2400-3600it, B15+B16]`.
21. **Gain SIREN upper bound is LOAD-BEARING and STIFFNESS-FLOOR-GATED.** Narrow bounds [0.2,1.5] neutral
    (LS=0.314≈ctrl). Wide [0.05,4.0] CATASTROPHIC (LS=-0.266, 4/9 at -1.00). The bound caps local gain to
    prevent extreme overshoot. **B24 OVERTURN of "gain_hi=2.5 is default-safe":** at stiff80 (B16) gain_hi=2.5
    was safe (only 4.0 catastrophic); at the SOFT floor (stiff_lo=50) gain_hi=2.5 is CATASTROPHIC (LS=-0.075,
    ampL=16.5, 2 nodes runaway to -1.00) — the *healthy* nodes stay small, only 2 diverge. Soft regions + high
    local gain = runaway recoil ⇒ **the tolerable gain ceiling SHRINKS as the stiffness floor softens.** Keep
    gain_hi=1.5 in the soft-floor record regime. `[mechanism@LoopScore, B16+B24 — gain-ceiling × stiff-floor
    interaction, regime-gated]`.
22. **Node 1 is the UNIVERSAL BOTTLENECK — but its rescue mechanism is UNCERTAIN.** Node 1
    (top-center) is negative in 5/6 B17 slots (range -0.91 to +0.20) and dominates LS variance.
    B17 found **ONLY wl=35** rescued node 1 (+0.20 at 3600it). **B18 UPDATE (PARTIAL@1100it):**
    wl=35 did NOT rescue node 1 (-0.19), while 3 other slots got node 1 positive (+0.22 to
    +0.34) at wl=28.8. This suggests node 1's sign at 1100it is STOCHASTIC (init-dependent),
    and the node-1 collapse may EMERGE during mid-to-late optimization (1100→3600it). The B17
    wl=35 finding may have been a late-optimization coincidence, not a causal wl mechanism.
    **Needs convergence (2400-3600it) to resolve.**
    **B20 UPDATE (matched causal pair @150it): the "fibre PRESENCE causes node-1 collapse" lead
    (B19) is REFUTED.** With fibre the ONLY variable, fibre-ON was ALL-POSITIVE (node 1 = +0.21,
    LS=0.151) while fibre-OFF carried the catastrophe (node 5 = −0.56, node 0 <0, LS=0.128) — the
    OPPOSITE of B19's 4-vs-1 pattern. B19 was a small-sample artifact (1 nofibre draw vs 4 fibre).
    **Conclusion: which node collapses early is STOCHASTIC (init-draw); fibre-ON is at least as
    uniform as fibre-OFF and is the better parent.** Node 1 is NOT a fibre-caused bottleneck; it
    is one of several nodes that can draw a bad early basin. The "fibre = variance amplifier that
    sacrifices node 1" story is unsupported by the controlled test.
    `[mechanism@LoopScore, 3600it→CHALLENGED@1100it→LEAD@450it→REFUTED@150it-matched-pair, B17+B18+B19+B20]`.
23. **MPM `--substeps` is SCIENCE-CRITICAL — loop morphology depends on integrator resolution.**
    At matched 2400it, substeps=6 gives LS=0.283 vs substeps=10's 0.320 (Δ=−0.037), WORSE
    uniformity (SD 0.300), and ~3.3× overshoot (ampL 0.090 vs 0.027); the gain SIREN also blows
    up (field max ~0.45 vs 0.20) compensating for the under-resolved dynamics. The elastic-
    overshoot limit cycle that SETS loop size is not well-resolved at substeps=6. **Keep
    substeps=10.** The "free depth multiplier" hope (B21) is dead. **B26 UPDATE: FINER integration
    (substeps=14) is NOT a size lever either** — size FLAT (1.09e-3, if anything marginally UP), LS DOWN to
    0.308, one node −0.40. So substeps=10 is a TWO-SIDED sweet spot: coarser (6) AND finer (14) both worse.
    Integrator resolution sets stability, NOT loop size — the size residual (fact #24) is NOT integrator-limited.
    `[mechanism@LoopScore, 2400it, B21+B26]`.
24. **⚠ SUPERSEDED BY #29 + the 2026-07-04 audit — the residual is ENCLOSURE, not SIZE; "size" was a
    sim-only, magnitude-blind diagnostic.** Kept for the falsification record. What this fact ESTABLISHED and
    still holds: NO explored parameter moved the residual — drive, stiffness, duration, damping, gain-ceiling,
    integrator resolution, waveform asymmetry all leave it flat. What it got WRONG: it read that as "loop SIZE
    is a HARD forward-model-STRUCTURAL limit." The correct, TEMPERED claim is: **within the explored parameter
    family, no parameter increased AREA-ENCLOSURE** — which is NOT the same as "this forward model cannot
    generate area." The audit shows magnitude is ~right (energy 0.95) while enclosure is deficient (area 0.16,
    minor-axis 0.17 vs 0.39): the motion is preferentially RADIAL, not circulating. The right question is no
    longer "make loops bigger" (a magnitude/force ask — all falsified) but **"what transfers motion from the
    major axis to the minor axis?"** — mechanisms NOT yet ruled out: fibre architecture (see fact #29 — nofibre
    HALVES area), local fibre residuals, anisotropic active stress, activation TIMING (travelling wave),
    viscoelastic lag, transverse compliance. None of these are "more force." Original evidence below (valid as
    the parametric-exhaustion record; reinterpret "size" as "enclosure").
    `[mechanism@LoopScore → SUPERSEDED by #29; tempered 2026-07-04 audit].`
    Across all B21–B23 converged montages, red (sim) loops sit INSIDE the
    larger green (real) loops; chirality/axis look broadly right. `size` diagnostic is stuck at ~1.02–1.08e-03
    across drive (amp10/11/12, gain_hi1.5/2.5 — B22), stiffness (stiff_lo80/50/40, stiff_hi300/400 — B23),
    duration (dur_hi11/13 — B23), AND damping (drag30/40/50). **B24 CLOSES the last parametric lead:**
    re-testing drive in the NEW tamed-overshoot soft-floor regime (stiff_lo50, ampL≈0.004) — amp12 left size
    FLAT (1.04e-03) + lower LS, ghi25 DIVERGED (not enlarged). So size is drive-invariant ACROSS regimes
    (stiff80 → soft-floor stiff50). LoopScore is co-dominated by size≈chirality (sensitivity 1.96/1.97), so
    size is the highest-leverage residual, but **NO material/drive parameter moves it in ANY regime — the
    parametric frontier for size is EXHAUSTED.** The residual is STRUCTURAL: set by the forward
    integrator/limit-cycle (substeps, fact #23) or the activation-waveform shape (symmetric Gaussian →
    reversible → small excursion). **B26 CLOSES BOTH structural sub-leads: finer substeps=14 left size FLAT
    (fact #23), and `--pulse_skew`=2.0 left size FLAT/smaller (fact #27, → overshoot instead).** So size is now
    invariant to EVERY in-model lever: drive · stiffness · duration · damping · gain-ceiling · integrator
    resolution · activation-waveform asymmetry. [TEMPERED: this exhausts the MAGNITUDE/parametric family — it
    does NOT prove the forward model cannot enclose area; it means the residual is a DIRECTIONALITY/enclosure
    deficit no magnitude lever addresses. See #29.] Only two
    never-touched structural constraints remain: (i) the BOUNDARY Dirichlet anchor (`--bwidth`) — the GT-pinned
    outer ring may COMPRESS interior excursion; (ii) the SETTLE window (`--warmup`) — the scored beat may run
    before the limit cycle reaches full amplitude. → B26 tests both. `[mechanism@LoopScore, 2400it converged,
    B21–B26 — size is invariant to ALL in-model levers; frontier = boundary/settle structure].`
25. **LOOP SIZE IS NOT DRIVE-LIMITED — amplitude/gain set OVERSHOOT, not primary loop size.** B22
    (drag40, converged): the `size` diagnostic (mean node max|disp|) is FLAT at 1.03–1.06e-03 across
    amp10/11/12 AND gain_hi 1.5/2.5. Raising amplitude 10→12 leaves loop size unchanged, monotonically
    LOWERS LS (0.344→0.310), and only raises overshoot (ampL 0.012→0.020). Extra active stress is
    dissipated as recoil overshoot in the overdamped MPM, not converted into a larger excursion. So the
    size residual (#24) is set by the ELASTIC RESPONSE (stiffness: size≈stress/stiffness → softer =
    bigger) and/or CONTRACTION TRAVEL (duration), NOT by stress magnitude. amp=10, drag∈[40,50],
    gain_hi=1.5 is the confirmed drive envelope. `[mechanism@LoopScore, 2400it, drag40, B22, NEW —
    redirects the size agenda off drive].`
26. **STIFFNESS SOFTENING is an OVERSHOOT/SHAPE lever (NOT a size lever) — the current LS mechanism.**
    Lowering stiff_lo (softer soft regions) makes the material convert active stress to strain more
    EFFICIENTLY, so the optimizer needs LESS drive: the learned gain field HALVES (dashboard max 0.20→0.11)
    and overshoot collapses (ampL 0.023→0.004, ≈6×) — while loop SIZE stays flat. The LS gain (+0.033,
    0.332→0.365) comes from cleaner loop SHAPE (less recoil overshoot), not bigger loops. This is GATED by
    drag40: at drag30 the softer floor over-recoils and the benefit vanishes (0.333). So the "softer →
    bigger loop" static-excursion intuition is WRONG in this overdamped active limit cycle — softening
    buys overshoot headroom, which drag40 must be present to realize. `[mechanism@LoopScore, 2400it, drag40,
    B23, NEW — softening = overshoot lever, drag-gated].`
27. **ACTIVATION TIME-ASYMMETRY (`--pulse_skew`) is a SHAPE/OVERSHOOT lever, NOT a size lever.** skew=2.0
    (fast contract, slow release — the physiological twitch) left loop SIZE flat/smaller (1.02e-3, the smallest
    in B26) while RAISING overshoot (ampL 0.011→0.024, highest in batch), openness (0.213→0.233, highest), and
    driving the SIREN gain field UP (dashboard max 0.21→0.35). The asymmetric twitch is dissipated as recoil
    overshoot and changes loop openness — it does NOT grow peak excursion. Same signature as stiffness softening
    (#26): a shape/overshoot channel, not a size channel. LS=0.331 (mid-batch, within lottery). So neither the
    integrator (#23) NOR the activation-waveform shape moves the size residual. `[mechanism@LoopScore, 2400it,
    B26, NEW — waveform asymmetry = shape/overshoot lever, size-invariant].`
28. **THE GAIN CEILING IS A PER-REGION SIZE+UNIFORMITY LEVER @rot1.0 — OVERTURNED@B33 (was "buys nothing" @rot=0).**
    B24/B26 (soft-floor, NON-rotating): raising gain_hi above 1.5 bought NOTHING (size flat, LS flat) and 2.5 was
    RUNAWAY (ghi25 ampL=16.5, 2 nodes −1.00). **B33 OVERTURN @rot1.0:** gain_hi 1.5→2.0 (ghi20) is the RECORD
    (LS=0.474→0.509, +0.035), raising peak_ratio 0.50→0.53 and area 0.34→0.39 by rescuing the small radial-stub
    nodes to ALL-POSITIVE (min +0.20). MECHANISM: in the rotating (enclosure-solved) regime the extra LOCAL gain
    headroom converts to loop AREA, not overshoot — the same energy the fixed-axis regime dissipated as recoil is
    now circulated. So "gain-ceiling invariant / 2.0 is the edge" (old #28) was REGIME-BOUND to rot=0. Whether
    gain_hi>2.0 keeps helping or hits the old runaway edge is the B34 question (ghi22/ghi25 @rot1.0). Global drive
    (base amplitude) still overshoots (#4/#25) — the size gate opens only for PER-REGION drive (gain ceiling,
    fibre_dev). `[mechanism@LoopScore, 2399it, rot1.0, OVERTURNED@B33 — gain ceiling = size/uniformity lever,
    regime-bound; single-draw, replicating@B34].`
29. **THE RESIDUAL IS AREA-ENCLOSURE (loopiness), NOT size — ESTABLISHED. The radial-motion ROOT CAUSE is a
    WORKING HYPOTHESIS that B27 tests, NOT yet an established mechanism.** What the 2026-07-04 audit + B26
    ESTABLISH (three claims, well-supported): (i) the residual is area-enclosure — the audit reframed it off
    `size` (unsound, sim-only/boundary-contaminated) onto loopiness: sim traces near-radial 1-D paths (loopiness
    ~0.21 vs real ~0.50, enclosed-area ratio ~0.16); (ii) fibre INFLUENCES enclosure — the no-fibre ablation
    HALVES area (fact #5), a convincing directional signal; (iii) the explored PARAMETER FAMILY does not fix
    enclosure. **B26 CLOSED the last two structural SIZE leads:** (c1) boundary anchor `--bwidth` — a monotone but
    tiny effect (bwnar 0.03 LS 0.341 ≥ ctrl 0.336 ≥ bwwide 0.10 0.308), within the ±0.05 lottery and its `size`
    trend is the boundary set moving inside the contaminated diagnostic, NOT interior enlargement (bwidth 0.03 is a
    safe EXPLOIT). (c2) settle window `--warmup` — FALSIFIED with OPPOSITE sign: warmup=100 (~2 beats) is the WORST
    slot (LS 0.267, ampL 0.044 highest); a longer settle does not grow the stable cycle, it accumulates recoil.
    **WORKING HYPOTHESIS (montage-SUGGESTED, b26_ctrl — B27 will confirm or falsify):** global SYNCHRONOUS
    activation, acting through a LARGELY UNIAXIAL fibre architecture (the learned quiver looks roughly single-axis
    in the montages), may bias particle motion toward RADIAL trajectories that enclose little area. IF true, this
    would explain why enclosure was
    invariant to every in-model lever (none breaks the temporal/directional symmetry), and introducing SPATIAL
    activation TIMING should raise enclosed area. That is exactly what B27 decides — a TRAVELLING-WAVE activation
    phase (staggered regional contraction): a positive result (enclosure_row area/loop/minor ratios rise) supports
    the hypothesis; a null falsifies it and means radial motion is intrinsic to the uniaxial-active-stress model
    (→ time-varying/biaxial stress or a rotational fibre field). **STATUS (B27 RAN, converged 2399it): the
    travelling-wave hypothesis is FALSIFIED — staggered timing LOWERS enclosure, monotonically.** Over the tw dose
    (0→6→12→20): area_ratio 0.130→0.108→0.104→0.085, loopiness_ratio 0.503→0.405→0.365, energy 0.94→0.66, ampL
    0.004→0.114, LS 0.360→0.320→0.249; direction test (tw12y ang90) also < ctrl (0.308). FAILURE MECHANISM:
    desync DECOHERES a still-uniaxial pull — Dirichlet-pinned interior regions fire out of phase and partly cancel
    (energy↓), mistimed recoil becomes overshoot (ampL↑); timing does NOT rotate a particle's FORCE DIRECTION, so
    loops get THINNER not fatter. **DEEPER: radial motion is TIME-REVERSIBILITY-intrinsic — a single axis n(x,y)
    with a near-symmetric envelope contracts along n then releases back along n → retraces → ~zero area; the ~0.13
    area is only inertial/damping lag (thin loops). Enclosure REQUIRES the contraction axis to ROTATE during the
    beat** (contract and release push along DIFFERENT directions). **CONFIRMED@B28: `--rot_stress` (a rotating axis,
    fact #30) raises enclosure MONOTONE (area_ratio 0.10→0.36, loopiness→real) AND breaks the LS ceiling
    (0.332→0.481) — the time-reversibility prediction is validated; rotation works where staggered timing (B27)
    failed.** `[ESTABLISHED: residual=enclosure, fibre-gated, radial=time-reversibility-intrinsic, ROTATION-fixable
    (#30) | travelling-wave FALSIFIED@B27 — timing decoheres, does not rotate — 2399it, B26+B27+B28+2026-07-04 audit].`

30. **ROTATING CONTRACTION AXIS (`--rot_stress`) IS THE AREA-ENCLOSURE MECHANISM — ESTABLISHED, and it is the
    RECORD-BREAKING LS lever.** A mean-zero, phase-locked axis swing `theta(x,y) + rot_stress·sin(2π(fr−onset)/period)`
    (radians; 0=OFF byte-identical, applied at all four frame-stepping sites, differentiable in θ) moves the residual
    the whole campaign called invariant (facts #24–#29), MONOTONE and large, AND breaks the 2-month LS≈0.365 ceiling:
    over rot 0→0.3→0.6→1.0, LS 0.332→0.430→0.461→0.481 (RECORD), area_ratio 0.100→0.189→0.284→0.360 (3.6×),
    loopiness_ratio 0.424→0.712→0.992→1.107 (hits REAL at rot0.6, overshoots at rot1.0), minor_axis_ratio
    0.446→0.794, chir_match 0.789→0.844. **MECHANISM: rotation REDISTRIBUTES, it does not add force** — energy_ratio
    stays ~0.85 flat and peak_ratio DROPS (0.544→0.488), so radial (in-and-out) motion is converted into circulation.
    This is exactly why the enclosure residual was invariant to every MAGNITUDE lever (drive/stiff/dur/drag/waveform)
    but yields to a DIRECTIONAL one, and it directly CONFIRMS fact #29's deep prediction (a fixed axis with a
    symmetric envelope is time-reversible → retraces → ~0 area; the axis must ROTATE during the beat). SIGN is WEAK:
    rotneg (−1.0) gives LS 0.441 ≈ rot +1.0, chir_match 0.833 vs 0.844 — magnitude dominates, sign barely flips
    chirality. LS still RISING at rot=1.0 (not peaked); optimizer shortens the pulse (dur 9.6→6.4–7.7). Caveat:
    rot10 (0.481) is a SINGLE draw → B29 replicates. **B31 REPLICATES + adds SATURATION:** rot10 redrew LS=0.462
    (within ±0.05 of 0.481) → mechanism STABLE. But rot is a SATURATING enclosure knob for LoopScore — over rot
    1.0→1.5→2.0, area_ratio keeps climbing MONOTONE (0.308→0.357→0.435) while LS does NOT (0.462→0.430→0.438): past
    rot1.0 the extra swing DEGRADES chir_match (0.838→0.809) and dips loopiness, so more raw area does NOT buy LS. LS
    is decoupled from area once loopiness≈real (hit at rot1.0). rot_stress OPERATING POINT = 1.0. **B32 RECONFIRMS the
    saturation at an intermediate dose:** rot14 (rot 1.0→1.4 @soft-floor) gave the HIGHEST area_ratio (0.402) + minor_axis
    (0.906) of B32, yet LS=0.448 ≤ the rot1.0 replicate (0.453), chir flat (0.837). Sign weak-but-NOT-neutral: B32 rotneg
    (−1.0) is the WORST slot (LS=0.432), the ONLY one with loopiness BELOW real (0.844) + lowest chir (0.817) — reversing
    the swing slightly degrades BOTH enclosure and chirality (not pure magnitude). `[mechanism@LoopScore,
    2399it converged, ESTABLISHED@B28, REPLICATED+SATURATION@B31+B32 — rotating axis = enclosure channel via redistribution,
    LS peaks at rot1.0 (area saturates above); parent=b27 record family stiff[50,300]].`
31. **SIZE-LEVER × ROTATING-REGIME — RESOLVED@B31→OVERTURNED@B32: NEITHER softening NOR drive is a robust LS lever
    @rot1.0; the residual is per-node UNIFORMITY, and FIBRE HETEROGENEITY (not the material floor) is what moves it.**
    In the rot=1.0 regime the median loop is a fat closed ellipse; the residual is absolute AREA (~0.35 vs real 1.0) +
    per-node UNIFORMITY (LS_SD ~0.30, a handful of still-radial nodes). **(a) stiff_lo SOFTENING does NOT reopen — B31's
    slo30 "win" (0.475) was a LOTTERY HIGH DRAW.** B32 replicated it at 0.453 and the whole floor ladder is FLAT within
    noise: slo20/slo30/slo40 = 0.456/0.453/0.459 (area drifts mildly UP with a FIRMER floor, 0.323/0.286/0.351, but LS
    is flat). So softening is INERT across stiff_lo∈[20,40] @rot1.0 — the B31 "compliance→circulation" story is
    RETRACTED. **(b) AMPLITUDE does NOT reopen either** (B31 amp12: LS 0.443<rot10, still overshoot — facts #4/#25 hold
    under rotation). **(c) The LIVE lever is SIREN FIBRE HETEROGENEITY (fact #5):** raising fibre_dev 0.05→0.12 (fdev12)
    WON B32 by rescuing radial nodes — DOSE-CONFIRMED (fibre_dev 0.05→0.18 monotone 0.447→0.492, peak dev~0.18; #5), more
    per-region orientation variation, NOT softer material, cuts the uniformity residual under the rotating axis. **B33
    RE-ATTRIBUTES the residual: enclosure is now SOLVED (loopiness_ratio 1.06–1.18 ≥ real across all B32 slots), so the
    dominant gap is back to MAGNITUDE/SIZE — clean real-referenced peak_ratio ≈0.49 (sim peak = HALF real), area_ratio
    ≈0.35; dashboard red loops loopy+correctly-chiral but INSIDE green.** SIZE was declared "invariant to every lever"
    (#24/#25) but ONLY at rot=0 (radial/time-reversible) — with the axis now rotating, whether drive/gain/compliance
    converts to size is the LIVE B33 question (see Open-Q #1-NEW). **B33 RESOLVES IT: SIZE IS MOVABLE @rot1.0, but
    ONLY via PER-REGION drive — the GAIN CEILING (gain_hi 1.5→2.0 = record, #28) and fibre_dev (0.18→0.20) both raise
    peak_ratio 0.50→0.53 + area 0.34→0.39 by rescuing the small radial nodes; GLOBAL drive (base amplitude amp12/amp14)
    and floor softening (slo20) do NOT (amp still overshoots per #4/#25, floor inert per #31a).** So facts #24/#25/#28
    ("size/gain-ceiling invariant to every lever") are REGIME-BOUND to rot=0: the fixed axis dissipates extra drive as
    recoil; the rotating axis circulates the same energy into loop area. Size is now ◐ (single-draw, replicating@B34),
    not ✗. `[mechanism@LoopScore, 2399it, OVERTURNED@B32 (floor inert), RESOLVED@B33 (size = per-region-drive lever
    @rot1.0, gain-ceiling+fibre; amp/floor null); single-draw].`

## Optimization facts  `[optimization@<regime>]` — depth-dependent, never promote to mechanism

- **NEVER TRUST OPTIMIZATION STATE.** Many conclusions FLIPPED purely because training continued or
  init changed. B9 is a textbook case: "dur=24 is the interior optimum" (B8) was itself an optimization
  artifact — dur0=10 at 2400it reaches dur≈19.4, matching 4150it. Depth was only needed because the
  default init (dur0=14) trapped duration in the dur≈30 basin.
- **Spatial gain converges by ~3600it.** 3600→3950it: LS flat (0.313→0.312). The 2400→3600it
  benefit (+0.059, B14) is the LAST significant depth gain. Beyond 3600it, extra depth
  improves uniformity (all-positive at ~4000it) but not mean LS. 3600it is the recommended
  depth for spatial-gain configs.
  `[optimization@LoopScore, B14-B15, UPDATED@B15]`.
- **Wider SIREN (hidden384) HURTS optimization.** LS=0.146 (Δ=-0.052 vs ctrl 0.198). The
  larger parameter space creates a harder optimization landscape. More capacity ≠ better
  convergence for SIREN stiffness. `[optimization@LoopScore, 2400it, B13]`.
- **lr=5e-4 is NEUTRAL vs lr=1e-3.** LS=0.201 ≈ ctrl 0.198. Lower lr does not improve
  SIREN convergence quality. `[optimization@LoopScore, 2400it, B13]`.
- **Gain viable range is [0.4, 0.5] with wide stiffness.** gain0=0.3 CATASTROPHIC (LS=-0.406, B3);
  gain0=0.7 CATASTROPHIC (LS=-0.272, B4); gain0=0.4 ≈ 0.5 (LS=0.139 vs 0.140, B6). Without
  stiffness or with narrow stiffness, gain0 is flexible. `[optimization@LoopScore-corrected, 2400it]`.
- **SIREN convergence is STOCHASTIC — variance ~±0.05 LS (UPDATED@B18).** With fibre SIREN
  (dev=0.1), identical configs produce LS ∈ [0.257, 0.358] (B17 ctrl=0.257 vs B16=0.345).
  The variance is DOMINATED by node 1's basin draw (range -0.91 to +0.20). Without fibre
  SIREN: variance ±0.01 (B14-B15). The fibre SIREN AMPLIFIES stochastic variance ~5× by
  adding a non-convex orientation landscape. Negative-node count varies 0-3.
  **B18 (PARTIAL@1100it):** At 1100it, node 1 is positive in 3/6 slots regardless of wl.
  Node collapse may EMERGE during 1100→3600it optimization (SIREN basin rearrangement).
  `[optimization@LoopScore, 3600it, ω=5, fibre SIREN, UPDATED@B18]`.
- **TIGHTER CONSTRAINTS may improve uniformity.** B18@1100it: gnarr+fdev005 (narrow gain
  [0.2,1.5] + fibre dev=0.05) produced ALL-POSITIVE nodes with best uniformity (SD=0.267).
  Hypothesis: double constraint shrinks the SIREN optimization landscape, preventing the
  catastrophic basin entries that cause node collapse. **Needs convergence confirmation.**
  `[optimization@LoopScore, PROVISIONAL@1100it, B18]`.
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
- **THE SECONDARY DIAGNOSTICS (`size`/`open`/`chir+`/`ampL`) ARE UNSOUND FOR RESIDUAL ATTRIBUTION — do NOT
  interpret WHY LS is stuck from them.** `[engineering, 2026-07-04, human audit]` (a) `size`/`open`/`chir+`
  (`morphology_row`, train.py:198-204) are SIM-ONLY and computed on the 100-node dashboard `idx` set (NOT the
  `mov` set LoopScore uses); 36/100 of those nodes are Dirichlet boundary nodes PINNED to GT, so `size`≈1.07e-3
  tracks the boundary anchor, not the interior fit. The TRUE interior sim loop is median 5.8e-4 ≈ **0.6× real**
  (centered, on `mov`). So "size flat ~1e-3, matches real, stuck" (fact #24, Open-Q #1) was never a real-vs-sim
  residual. (b) `ampL` is a GLOBAL √ΣE ratio dominated by a few big nodes; ampL≈0.002 ("cleanest overshoot")
  coexists with the median node at 0.57× real rms — low ampL ≠ good loops. LoopScore ITSELF is sound (per-node,
  normalized, invariant); only these interpretive diagnostics are not. FIX: log a real-referenced per-node
  residual (sim/real peak · |area| · loopiness on `mov`) — see `audit_trajectories.py`.
- **THE DOMINANT RESIDUAL IS AREA-ENCLOSURE (openness/loopiness), NOT size.** `[mechanism@LoopScore, converged,
  2026-07-04 audit]` On wide400 (record): sim/real peak ratio med 0.59 but sim/real |enclosed-area| ratio med
  0.16; loopiness med sim 0.21 vs real 0.50; aggregate minor-axis variance frac sim 0.17 vs real 0.39 → the sim
  traces near-RADIAL (1-D) paths, right-ish magnitude but too little enclosed area (overlays: red line inside
  green loop even where size matches). area ≈ size²·loopiness ⇒ the loopiness deficit outweighs the size deficit.
  Openness/area-enclosure is co-dominant and NOT solved — reopen it as a live mechanism (regional anisotropy that
  breaks radial symmetry). Independent sim/real size ratio also VARIES with levers (nofibre 0.45, ctrl 0.64,
  wide400 0.59), so "size invariant to every lever" (fact #24) is partly a contaminated-diagnostic artifact.
- **INSTRUMENT THROUGHPUT is DOMINATED by external GPU contention, and it VARIES — 2400it IS
  reachable (do not assume otherwise).** `[engineering, 2026-07-02, B20→CORRECTED@B21]`
  B20's "~9 s/iter FIXED, 2400it needs ~6 h, UNREACHABLE" conclusion was a TRANSIENT-CONTENTION
  ARTIFACT: **B21 ran ALL 4 slots to full 2399/2400 convergence** — the other workspace projects
  (embryogenesis, active_matter2) freed the shared cuda:0. So throughput is set by WORKSPACE-WIDE
  GPU load (not my slot count — that part of B20 stands: cutting slots buys nothing), and that load
  is time-varying: some windows converge, others stall at 150–450it. **Always request n_iter=2400
  and read whatever depth the run reached** (dashboards checkpoint every 50it). Do NOT pre-declare
  the instrument the binding constraint — B18–B20 burned four batches on that false premise.
  substeps stays at 10 (science-critical, fact #23); it is NOT a depth lever.
- **CLUSTER SSH CREDENTIAL CAN DIE MID-CAMPAIGN → whole batch submit-fails, ZERO data (NOT a science result).
  B24→B37 = 14 CONSECUTIVE LOSSES (35% of the 40-batch budget) — but RESOLVED at B24-reissue by an operator
  RESTART.** `[engineering/ops, 2026-07-03, B24–B37→restart]` Symptom: all 6 slots fail `bsub` with
  `allierc@login1: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)` → every slot
  `NOT-SUBMITTED / done=NO / LS=na`, slot dirs hold ONLY `config.json`. Same shared credential that stalled the
  embryogenesis campaign — affects BOTH. **Diagnosis rule:** a whole batch with `LS=na`/no checkpoints → grep
  the session log for `SUBMIT FAILED` BEFORE treating it as a scientific null; an infra loss must NOT enter the
  physics ledger. On this fault, RE-ISSUE the same slots (the blocker is INTERMITTENT — a later window may work,
  as embryo b02/b03 did); don't redesign around a non-null.
  **DRIVER IDENTITY (corrected — the B27→B37 chain named the wrong file):** the live driver is
  **`cardio_mpm_loop.py`**, which does `import cardio_mpm_cluster as L` and calls `L.submit_cluster()` /
  `L.save_state()`. cluster.py's `no jobs submitted -- aborting batch` string surfaced in the logs and misled
  those analyses into "the running driver is cluster.py, guard unloaded." The batch loop is in `loop.py:main()`,
  and the HOLD-and-retry guard (commit 1440971) IS there at **`loop.py:145`** (`while not ids:` → `SUBMIT OUTAGE
  … HOLDING batch`, 10-min cadence, env `CARDIO_SUBMIT_RETRY_MIN`; never advances `save_state` on a total
  outage). The B24→B37 burn happened because the *running resume3 process* held an OLDER in-memory loop.py that
  predated the guard — precisely what a restart cures.
  **RESOLUTION (B24-reissue):** the operator RESTARTED the driver (new session log `campaign_resume4.out`;
  `cardio_mpm_loop_state.json` rewound `{"batch":37}`→`{"batch":24}` to re-run the lost science from the first
  lost batch). The restart reloaded the on-disk loop.py WITH the guard, so a still-dead credential now makes the
  loop HOLD instead of burn. The B27→B37 prescription ("RESTART is the credential-independent #1 fix") was
  correct even though it named cluster.py; renewing the Kerberos/SSH credential remains #2 (lets the held jobs
  actually run). **Two standing caveats:** (a) `[loop] cluster preflight OK` is NOT a reliable gate — it logged
  OK while the real `bsub` failed; (b) no in-sandbox escape — `nvidia-smi`/`torch.cuda`/`ssh` probes are all
  denied non-interactively, so the agent can neither renew, test, nor restart; those are operator-only.
  **B27 RECURRENCE (2026-07-04):** the submit-loss RETURNED — the travelling-wave batch launched (11:51:16, all
  configs correct, `--tw_amp` present) but every `p3_b27_s*` slot archived ONLY `config.json`, no data, LS=na. So
  the post-restart HOLD guard does NOT immunize against the loss; it converts a burn into a hold, but a batch can
  still archive empty stubs (cred died in an OFF window, or `bsub` IDs died on the cluster — indistinguishable in
  sandbox). Handled per the IDEMPOTENCY rule (state still `{batch:27}` + "Batch 27" design section already exists +
  `p3_b27_*` config-only stubs = design-step re-invocation): NOT logged as science, NOT redesigned; the completed
  travelling-wave slots LEFT VERBATIM (b27_*) so the loop re-submits batch 27 into the same dirs. Confirms the
  blocker is still INTERMITTENT and operator-gated.
- **CODE-CRASH is a DISTINCT 0-data loss mode from the SSH/submit loss — TRIAGE by the presence of `.out`/`.err`.**
  `[engineering/ops, 2026-07-05, B30 (=b29 code-crash)]` A whole batch with only `config.json` per slot is NOT
  automatically an SSH loss. Two signatures: (a) SUBMIT loss — NO `.out`/`.err` at all (bsub never launched the job);
  (b) CODE-CRASH — a `.out`/`.err` PAIR exists, the job ran on a host and died in ~10–15 s (`Run time ~15 s`,
  `Max Memory ~600 MB`, exit 1). **ALWAYS read a slot's `.err` FIRST** (`loop_logs/p3_b{N}_s*_*.err`): a Python
  traceback ⇒ code-crash; renew-the-credential does nothing, you must FIX code + re-issue. **B30 instance:** the M2
  operator refactor (commit 6737189) merged the drag op `mpm_drag`→`drag` (`emit: mpm_acceleration`); the spec was
  migrated but `cardio_mpm_train.py` still keyed `ops["mpm_drag"]` (lines 588 AND 589 — force_ops too) →
  `KeyError 'mpm_drag'` ~15 s ×6. FIXED (both keys → `"drag"`; behaviorally identical — new `Drag` exposes `.k` and
  returns `{mpm_particle: -k·v}` that `step_frame` routes as before). Sibling M3 refactor
  (`pulse_stimulus`+`phase_delay_pulse`→`activation_pulse`) left 3 HARMLESS `p_op("pulse_stimulus",…)` fallbacks at
  train.py:512–514 (defaults matched the spec) — renamed for correctness. **DURABLE RULE: after ANY operator refactor
  in the Plexus src, grep cardio's OWN `cardio_mpm_train.py` (not just specs/am2_ops) for every renamed op token —
  train.py has hardcoded `ops["…"]` and `p_op("…")` string lookups that a src rename silently breaks.** Was an
  EXECUTION LOSS, not science: last real data stays B28; the b29 rot_stress replication is re-issued as Batch 30.

## Rejected hypotheses (distilled — regime-tagged; re-openable)

- "Stiffness-floor softening REOPENS as a size/enclosure lever in the rotating regime (fact #31a)" —
  **OVERTURNED@B32.** B31's slo30=0.475 was a lottery high draw; the B32 replicate came in at 0.453 and the floor
  ladder is FLAT (slo20/30/40 = 0.456/0.453/0.459). Softening is INERT across stiff_lo∈[20,40] @rot1.0. The live
  uniformity lever is FIBRE heterogeneity, not material compliance. Moved to fact #31/#5. `B31→B32, CLOSED for floor.`
- "GLOBAL drive (base amplitude) or floor softening converts to loop SIZE in the rotating regime" —
  **FALSIFIED@B33.** @rot1.0 amp10→12/14 left peak_ratio flat (~0.50) with LS ≤ parent (overshoot, #4/#25 hold under
  rotation); stiff_lo 30→20 inert (#31a). The size gate opens ONLY for PER-REGION drive (gain ceiling #28 /
  fibre_dev), not global magnitude. Moved to facts #28/#31. `B33 — CLOSED for global-drive size @rot1.0.`
- "Structure is necessary for loops" / "rotary force is required" — **FALSIFIED** (2×2; loops inertial;
  rotary was a scaffold for a force-based model, not a cardiomyocyte property).
- "Spatial stiffness lifts the fit" — **FALSIFIED@R²; OVERTURNED@LoopScore.** SIREN stiffness converged to
  uniform under R² (no gradient signal) but is ACTIVE under LoopScore (binary pattern, LS +0.014). Moved
  to Established mechanisms #6.
- "Per-pixel fibre direction dθ (UNet or SIREN) helps" — **OVERTURNED@B16.** Originally FALSIFIED
  at dev=0.3 pre-spatial-gain (B6+B7). With spatial gain context + dev=0.1 (±6°): LS=0.345,
  NEW RECORD (+14% over ctrl). The conclusion was DOSAGE-limited (only dev=0.3 tested) and
  CONTEXT-limited (no spatial gain). Moved to Established mechanisms #8 (reopened).
- "A travelling-wave phase delay τ(x,y) bends the loops / raises area-enclosure" — **FALSIFIED TWICE.**
  (1) Learnable τ went to zero under R² (never used). (2) B27 (2399it) FIXED/swept `--tw_amp` 6/12/20: staggered
  timing LOWERS enclosure monotonically (area_ratio 0.130→0.085, loopiness 0.503→0.365, LS 0.360→0.249, ampL up
  ~28×) — it DECOHERES a still-uniaxial pull (Dirichlet-pinned regions cancel), it does not rotate the force
  direction. Enclosure needs a ROTATING axis, not staggered timing (fact #29→#30). `CLOSED for the timing route.`
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
- "stiff_hi=400 extends stiff_hi=300 benefit" — **FALSIFIED@drag30 (B8) → OVERTURNED@drag40 (B23).**
  @drag30: LS=0.076, 5/9 negative. @drag40: LS=0.354 (+0.022, 1 negative). The ceiling was drag-limited,
  not intrinsic. Moved to Established mechanisms #12. `B8→B23`.
- "stiffness range [80,300] is a HARD optimum" — **OVERTURNED@B23 (@drag40).** Only narrowing/floor-raising
  had been tested (all HURT). Floor-LOWERING to 50 gives the RECORD (0.365); the range is drag-gated.
  Moved to Established mechanisms #6. `B12→B23`.
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
- "Spatial gain is NOT converged at 2400it and deeper training pushes LS further" —
  **PARTIALLY FALSIFIED@B15.** 3600→3950it: mean LS flat (0.313→0.312). Depth improves
  UNIFORMITY (all-positive at ~3950it) but NOT mean LS beyond 3600it. `B15`.
- "Coarser gain SIREN (ω=3) better matches the tissue amplitude scale" — **FALSIFIED@B15.**
  LS=0.218, Δ=-0.095 vs ω=5 (0.313). Three negatives. Gain needs ω=5 like stiffness. `B15`.
- "amp=11 improves on amp=10 with spatial gain" — **FALSIFIED@B15.** LS=0.271, Δ=-0.042
  vs amp=10 (0.313). Two negatives. amp=10 is HARD optimum. `B15`.
- "dur_hi=13 allows a better duration optimum" — **FALSIFIED@B15.** LS=0.238, dur→11.5,
  node 5 catastrophic (-0.64). dur_hi=11 constraint IS the mechanism. `B15`.
- "fibre_dev=0.2 or 0.3 with spatial gain helps" — **FALSIFIED@B16.** dev=0.2: LS=0.300≈ctrl
  (0.303); dev=0.3: LS=0.303≈ctrl. Only dev=0.1 is beneficial (LS=0.345). The dose-response
  is non-monotonic — dev≥0.2 gives back the gain. `B16`.
- "Wider gain bounds [0.05, 4.0] improve flexibility" — **FALSIFIED@B16.** CATASTROPHIC
  (LS=-0.266, 4 nodes -1.00). Upper gain bound ≤2.5 is essential. `B16`.
- "Narrower gain bounds [0.2, 1.5] improve uniformity" — **NEUTRAL@B16, INCONCLUSIVE@B17.**
  Alone (B16): LS=0.314≈ctrl (0.303). Combined with fdev01 (B17): LS=0.358 (record) — but
  within stochastic range. Synergy unconfirmed. `B16+B17`.
- "dev=0.05 is too constrained for the fibre SIREN" — **FALSIFIED@B17.** LS=0.332, competitive
  with dev=0.1 draws. The dose-response in [0.05, 0.15] is FLAT within stochastic noise. `B17`.
- "dev=0.15 is the sweet spot between dev=0.1 and dev=0.2" — **FALSIFIED@B17.** LS=0.311,
  within stochastic band. No improvement over dev=0.1. `B17`.
- "Deeper training (4800it) with fibre SIREN rescues node 1" — **FALSIFIED@B17.** 4100it gave
  LS=0.305 with node 1=-0.49. Depth rescues nodes 0,3,6,7 but NOT node 1. Node 1 is a
  parametric-fibre-seed problem. `B17`.
- "The LS=0.345 fdev01 record (B16) is reproducible" — **FALSIFIED@B17.** Control replication
  gave LS=0.257 (Δ=-0.088). Stochastic variance ~±0.05. The record was a high-draw. `B17`.
- "wl=35 reliably rescues node 1" — **FALSIFIED@B18+B19.** B18@1100it: wl=35 got node 1=
  -0.19 while ctrl got +0.22. B19@450it: wl=35 got 3 negatives (worst negative count). Node 1
  sign is STOCHASTIC, not wl-controlled. B17's finding was a single-draw coincidence.
  `B18@1100it + B19@450it, CLOSED`.
- "angle=0.25 with default gain bounds is viable" — **FALSIFIED@B18 (PARTIAL@1100it).**
  LS=0.171 (worst in batch, ampL=0.061 overshoot). Confounded: also lacks narrow gain.
  Confirms B10 finding. `B18@1100it, partially confounded`.
- "Tighter fibre constraints (smaller dev) prevent node collapse" — **FALSIFIED@B19
  (PARTIAL@450it).** fdev003 (tightest) produced WORST node 1 (-0.90, most catastrophic in
  batch). The dose-response at 450it is OPPOSITE to predicted: tighter → more negatives.
  `B19@450it, CAVEAT: very shallow depth`.
- "gnarr+fdev005 is reproducibly all-positive" — **FALSIFIED@B19 (PARTIAL@450it).** Both
  replicates (s0, s5) had 2 negatives at 450it. B18's all-positive at 1100it may have been
  stochastic or may emerge during optimization (450→1100it). `B19@450it, provisional`.
  **B21 UPDATE@2400it: ctrl (fibre-ON, drag30, dev005) IS all-positive (worst +0.13) at
  convergence** — the all-positive property returns at depth; the 450it negatives were undertrained.
- "substeps=6 gives a free 1.67× depth with unchanged morphology" — **FALSIFIED@B21, 2400it.**
  substeps=6 LS=0.283 vs substeps=10 0.320 (Δ=−0.037), worse SD, 3.3× overshoot. Integrator
  resolution sets loop size; substeps is science-critical, not a cost knob. Moved to Established
  mechanisms #23. `B21, CLOSED`.
- "The instrument (GPU) is a FIXED binding constraint that makes 2400it unreachable" —
  **FALSIFIED@B21.** All 4 slots converged to 2399/2400. B20's 6-hour estimate was transient
  contention. Throughput varies with workspace-wide GPU load. `B21, CLOSED — do not re-assume.`
- "The amplitude/gain SIZE ceiling is DRAG-DEPENDENT (drag40 unlocks bigger loops)" —
  **FALSIFIED@B22, 2400it, drag40.** amp10/11/12 @drag40: LS 0.344→0.324→0.310 (monotone DOWN),
  size FLAT 1.04–1.06e-03, overshoot UP. gain_hi 2.5 also grew no size (LS=0.316). drag30 vs drag40
  at amp12 tie (0.320≈0.310). Deeper truth: drive does not grow size at ANY drag — it sets overshoot
  only (fact #25). `CLOSED — drive is the wrong size lever.`
- "amp>10 helps if overshoot is damped" / "gain_hi=2.5 grows loop size" — **FALSIFIED@B22.** Neither
  moves the `size` diagnostic; both lower LS. amp=10, gain_hi=1.5 confirmed as the drive envelope. `B22.`
- "The soft-floor/halved-gain regime finally lets DRIVE convert to loop SIZE" (B24 lead a) — **REFUTED@B24,
  2400it, stiff_lo50.** amp12 kept size FLAT (1.04e-03) + dropped LS (0.299); gain_hi=2.5 DIVERGED (ampL=16.5,
  2 nodes -1.00), its size=1.29e-03 an artifact of runaway, not enlargement. Size is drive-invariant in the
  soft regime too ⇒ across ALL regimes. `CLOSED — size is not parametric; frontier is structural (fact #24).`
- "FINER integration (substeps=14) grows loop size" (B25 lead b1) — **FALSIFIED@B26, 2400it.** size FLAT
  (1.09e-3), LS DOWN (0.308), one node −0.40. substeps=10 is a two-sided sweet spot; integrator resolution is
  not a size lever. Moved to fact #23. `B26, CLOSED.`
- "Time-asymmetric activation (pulse_skew=2.0) grows peak excursion" (B25 lead b2) — **FALSIFIED@B26, 2400it.**
  size FLAT/smaller (1.02e-3); the skew added OVERSHOOT (ampL 0.024) + openness instead. Waveform asymmetry is
  a shape/overshoot lever, not a size lever. Moved to fact #27. `B26, CLOSED.`
- "gain_hi=2.0 diverges like gain_hi=2.5 in the soft regime" — **FALSIFIED@B26.** ghi20 was SAFE (no runaway,
  size flat, LS≈ctrl). Tolerance edge is 2.0–2.5; raising the ceiling buys no size. Moved to fact #28. `B26.`
- "A longer SETTLE window (warmup~2 beats) grows the limit-cycle amplitude / loop size" (B26 lead c2) —
  **FALSIFIED@B26, OPPOSITE SIGN.** warmup=100 was the WORST slot (LS 0.267, ampL 0.044 highest, SD 0.337).
  The cycle is stable; extra settle accumulates recoil, it does not enlarge. Moved to fact #29. `B26, CLOSED.`
- "Freeing the interior (narrower `--bwidth`) grows loop SIZE" (B26 lead c1) — **NOT A SIZE LEVER@B26.** bwidth
  0.03→0.06→0.10 is monotone in LS (0.341/0.336/0.308) but within the ±0.05 lottery, and the `size` trend is the
  boundary node set moving inside the contaminated dashboard diagnostic, not interior enlargement. bwidth 0.03 is
  a mild safe EXPLOIT (cleanest overshoot), NOT the size mechanism. Moved to fact #29. `B26.`
- "The residual bottleneck is loop SIZE, invariant to all levers, hence structural" (B22–B26 framing) —
  **SUPERSEDED by the 2026-07-04 audit + B26.** The driving `size` diagnostic was boundary-contaminated/sim-only;
  the real dominant residual is AREA-ENCLOSURE (loopiness ~0.21 vs 0.50), rooted in the single-global-pulse radial
  motion, NOT a size ceiling. Reopens openness/area-enclosure as the live mechanism. Moved to fact #29. `→B27.`

## Open questions

- ~~**Is the amplitude/gain SIZE ceiling DRAG-DEPENDENT?**~~ **ANSWERED (B22): NO — and deeper,
  size is not drive-limited at all (fact #25).** Drive sets overshoot, not size. `CLOSED.`
- ~~**What NON-DRIVE mechanism grows loop SIZE? (B23 question.)**~~ **ANSWERED (B23): NONE of stiffness
  or duration does.** Softening stiff_lo lifts LS (record 0.365) but via OVERSHOOT reduction, not size —
  `size` stayed flat (fact #26). dur_hi13 neutral. So drive (B22), stiffness AND duration (B23) all leave
  size invariant (fact #24). `CLOSED — the mapped mechanisms do not move size.`
- ~~**#1 — Does a mechanism create ENCLOSED AREA (raise loopiness toward real)?**~~ **ANSWERED — YES, a ROTATING
  contraction axis (`--rot_stress`) does it (B28, fact #30); the travelling-wave route was FALSIFIED (B27).** Timing
  decoheres a uniaxial pull; ROTATION redistributes radial motion into circulation → area_ratio 0.10→0.36,
  loopiness→real, LS 0.332→0.481 (RECORD). `CLOSED — enclosure is rotation-fixable.`
- **#1 (NEW) — Does a SIZE lever now convert to loop SIZE in the ROTATING regime? (B33, LIVE — now with clean evidence.)**
  With enclosure SOLVED (loopiness ≥ real, B32), the dominant residual is a CLEAN real-referenced peak_ratio ≈0.49 (sim
  peak = half real) + area_ratio ≈0.35 on the mov set. Every size lever (amp, gain_hi, stiff_lo) was falsified ONLY at
  rot=0 (radial/time-reversible, where drive→overshoot, facts #24/#25). PREDICTION: with the axis rotating, extra
  drive/gain/compliance may now ENLARGE the loop (raise peak_ratio) instead of dissipating as radial recoil, because
  rotation redistributes radial motion into circulation. **B33 tests amp12/amp14 (drive) + gain_hi2.0 (ceiling) + stiff_lo20
  (compliance) @rot1.0, dev18 parent.** A clean null (all leave peak≈0.49 + lower LS) → absolute size is capped INDEPENDENT
  of rotation → structural (boundary Dirichlet compliance `--bwidth`, or a bigger-strain constitutive change), not a
  drive/material lever. An overturn (any lever raises peak_ratio while holding LS+chir) → facts #24/#25 are regime-bound to rot=0.
- **#1b (NEW) — Where is the fibre_dev LS peak, dev0.18 or 0.20? (B33.)** The dose ladder rises to dev0.18 (0.492) and rolls
  off at dev0.25 (0.482); dev0.20 fills the peak and pins the operating point.
- **#2 (NEW) — Where is the rot_stress LS optimum, and is fibre still load-bearing under rotation? (B29.)** LS still
  RISING at rot=1.0 (find the peak: rot1.5/2.0). And does the LEARNED fibre SIREN still add value once rot swings the
  axis, or has rotation made it redundant (nofib ablation)?
- ~~**What STRUCTURAL mechanism sets loop SIZE? (was THE #1 question, B25–B26.)**~~ **RETIRED — wrong question.**
  The `size` diagnostic that framed it was unsound (audit); the two remaining structural leads closed @B26 —
  boundary `--bwidth` a tiny lottery-band effect (not interior enlargement) and settle `--warmup` FALSIFIED with
  opposite sign (worst slot). The real residual is area-enclosure, not size (fact #29). See #1 above. `CLOSED.`
- ~~**Does stiffness CONTRAST STACK — is stiff[50,400] > stiff[50,300]?**~~ **ANSWERED (B24): NO net gain.**
  stiff[50,400] = LS 0.340 ≈ ctrl 0.343 (no negative node this draw, unlike B23's shi400). The soft floor
  absorbs the high-ceiling overshoot but contrast does not STACK into extra LS. `CLOSED — neutral.`
- ~~**Does `--substeps 6` preserve morphology for free depth?**~~ **ANSWERED (B21): NO** — substeps
  is science-critical (fact #23); and convergence is reachable at substeps=10 anyway. `CLOSED`.
- ~~**Does the fibre SIREN CAUSALLY drive the node-1 collapse?**~~ **ANSWERED (B20): NO.** The
  matched pair at 150it showed fibre-ON all-positive (node1=+0.21) and fibre-OFF carrying the
  catastrophe (node5=−0.56). Early node collapse is stochastic, not fibre-caused. `CLOSED`.
- **Can the no-fibre-SIREN family (LS≈0.31±0.01) be improved beyond 0.31 with a NEW mechanism?**
  Fibre SIREN adds peak draws (0.358) but ~5× more variance. Without fibre SIREN, the config
  is reproducible but capped. What mechanism can push the reproducible family above 0.31?
  **#2 priority (post-B20).**
- **What is the RESIDUAL BOTTLENECK dimension?** **PARTIALLY ANSWERED (B21): loop SIZE** — sim
  loops sit inside larger real loops in every converged montage (fact #24). A quantitative residual
  decomposition on the B21 converged models (`run_decompose_*.sh` exist) would confirm the
  size-vs-chirality split. **#3 priority — run once, cheaply.**
- ~~**Does gnarr+fdev005 preserve all-positive to convergence?**~~ **ANSWERED (B21@2400it): YES.**
  ctrl (fibre-ON, drag30, dev005) converged all-positive (worst +0.13). B19's 450it negatives were
  undertrained. `CLOSED`.
- ~~**Is wl=35 → node-1 rescue reproducible?**~~ **ANSWERED (B18+B19): NO.** wl=35 failed in
  both B18 (1100it) and B19 (450it). B17's finding was a coincidence. `CLOSED`.
- ~~**RUNTIME: cut slots to converge?**~~ **FALSIFIED@B20.** Throughput does NOT scale with my
  slot count: 2 slots ran at ~9.1 s/iter = 6 slots (9.6). GPU is shared with the whole workspace.
  Cutting slots is USELESS for depth. The only depth lever is per-iter cost (substeps). See
  engineering fact "PER-ITERATION COST IS ~9 s AND FIXED". `CLOSED — model was wrong.`
---

## Previous theme summaries (last 4, oldest→newest; MUST precede ## Current theme)

- **Batch 27 (2026-07-04, CONVERGED@2399it):** the TRAVELLING-WAVE hypothesis is FALSIFIED — staggered timing
  LOWERS enclosure monotonically (area_ratio 0.130→0.085, loopiness 0.503→0.365, LS 0.360→0.249, ampL up ~28×).
  Desync DECOHERES a still-uniaxial pull (Dirichlet-pinned regions cancel); timing does not rotate the force
  direction. DEEPER: radial motion is TIME-REVERSIBILITY-intrinsic — enclosure REQUIRES the contraction axis to
  ROTATE during the beat. → B28 tests `--rot_stress` (a phase-locked axis swing).
- **Batch 28 (2026-07-05, CONVERGED@2399it) — DECISIVE WIN:** `--rot_stress` (rotating contraction axis) IS the
  area-enclosure mechanism (fact #30, ESTABLISHED) and it BREAKS the 2-month LS≈0.365 ceiling. rot 0→0.3→0.6→1.0:
  LS 0.332→0.481 (RECORD), area_ratio 0.10→0.36 (3.6×), loopiness→real, minor_axis→0.79 — all MONOTONE. Rotation
  REDISTRIBUTES radial motion into circulation (energy flat ~0.85, peak DROPS) — it works where every magnitude
  lever failed, confirming fact #29's time-reversibility prediction. Sign weak (rotneg ≈ rot+1.0). rot10 single
  draw → B29 replicates + maps the SHIFTED residual (absolute size, now in the rotating regime).
- **Batch 31 (2026-07-05, CONVERGED@2399it):** rot10 mechanism REPLICATED (fact #30 stable). rot is a SATURATING
  enclosure knob — over 1.0→2.0 area climbs while LS does not (peaks at 1.0). Fact #31 SPLIT (later overturned by
  B32): slo30 drew the batch high (LS=0.475), amp12 did not reopen. Residual = per-node uniformity (LS_SD ~0.30);
  fibre quiver still near-uniaxial. → B31/32 tests floor curve (slo20/40), higher rot (rot14), fibre heterogeneity
  (fdev12), rotation sign (rotneg).
- **Batch 32 (2026-07-05→06, CONVERGED@2399it):** FIBRE HETEROGENEITY is a DOSE-CONFIRMED live LS lever under rotation.
  The fibre_dev ladder rises MONOTONE (dev 0.05/0.08/0.12/0.18 → 0.447/0.465/0.473/0.492, rolls off 0.25→0.482; peak
  dev~0.18, new op point dev18 LS=0.492). The earlier single "fdev12=0.493" REGRESSED to 0.473 on replicate (campaign
  law) but the dose ladder rescued the lever. The win is dev MAGNITUDE not spatial SCALE (fibre_wl20 inert = 0.473).
  **Residual RE-ATTRIBUTED: rotation SOLVED enclosure (loopiness ≥ real everywhere), so the dominant gap flips BACK to
  SIZE — clean peak_ratio ≈0.49, area ≈0.35, red loops loopy-but-inside-green.** → B33 re-opens size in the rotating
  regime (does drive/gain/compliance now convert to peak excursion?).

---

## Current theme
### Current hypothesis
"ESTABLISHED (B28, replicated B31): a ROTATING contraction axis (`--rot_stress`) IS the area-enclosure mechanism —
it filled the residual the whole campaign called invariant AND broke the 2-month LS≈0.365 ceiling (fact #30). rot is
a SATURATING knob: LS peaks at 1.0 (rot OPERATING POINT), area keeps climbing above but LS does not. Enclosure SHAPE
is solved (loopiness≈real); the residual SHIFTED to per-node UNIFORMITY (LS_SD ~0.30: a handful of still-radial
nodes) + absolute AREA (~0.35 vs real 1.0).
**B32 CONFIRMED the fibre lever by DOSE-RESPONSE and RE-ATTRIBUTED the residual to SIZE.** (1) FIBRE HETEROGENEITY is
REAL, not lottery: the fibre_dev ladder rises MONOTONE (dev 0.05/0.08/0.12/0.18 → 0.447/0.465/0.473/0.492), rolling off
at dev0.25 (0.482) → peak dev~0.18–0.20 (fact #5). The earlier single 'fdev12=0.493' REGRESSED to 0.473 on replicate
(9th single-draw regression) but the independent monotone dose ladder rescued the lever. The win is dev MAGNITUDE, not
spatial SCALE — fibre_wl 28.8→20 (finer) = 0.473 ≡ replicate (scale-inert). New op point = dev18 (LS=0.492).
(2) **THE RESIDUAL FLIPPED BACK TO SIZE.** Rotation SOLVED enclosure: loopiness_ratio is 1.06–1.18 (at/ABOVE real) in
every B32 slot. So the dominant gap is now MAGNITUDE — a CLEAN real-referenced peak_ratio ≈0.49 (sim peak = HALF real),
area_ratio ≈0.35; the dev18 dashboard shows red loops loopy + correctly-chiral but sitting INSIDE the green.
**B33 HYPOTHESIS (the ONE question): in the enclosure-solved rotating regime, is the SIZE residual now DRIVE-limited?**
Facts #24/#25 ('drive→overshoot, not size') were established ONLY at rot=0 (radial/time-reversible). With the axis
rotating — which redistributes radial motion into circulation — extra drive/gain/compliance may now CONVERT to loop
size (raise peak_ratio) instead of overshooting. B33 fans out the causal candidates: amp12/amp14 (drive), gain_hi2.0
(ceiling), stiff_lo20 (compliance), each vs a dev18 replicate control; plus dev20 to pin the fibre-dose peak. A clean
NULL (all leave peak≈0.49 + lower LS) → size is capped INDEPENDENT of rotation → structural (boundary `--bwidth`
compliance or a constitutive strain change); an OVERTURN (any lever raises peak_ratio holding LS+chir) → #24/#25 are
regime-bound to rot=0. Settled context: substeps=10, rot1.0, drag40, gain[0.2,1.5], amp10, stiff[30,300], dur_hi11.
Parent = dev18 (fibre_dev0.18)."
### Iterations this theme
- Batches 1–12: scalar levers + SIREN architecture saturated at LS≈0.20 plateau.
- Batch 13: SIREN capacity hypothesis FALSIFIED. Spatial gain implemented.
- Batch 14: SPATIAL GAIN CONFIRMED as ceiling-breaking mechanism (LS≈0.31, +50%).
- Batch 15: ALL scalar knobs CLOSED. Converged at 3600it. gain ω=5 confirmed.
- Batch 16: SIREN fibre dθ REOPENED — dev=0.1 gives LS=0.345 NEW RECORD (+14%).
- Batch 17: dose-response FLAT in [0.05,0.15]; B16 record stochastic; node 1 = bottleneck.
- Batch 18 (PARTIAL@1100it): constraint-driven uniformity hypothesis formed.
- Batch 19 (PARTIAL@450it): constraint hypothesis FALSIFIED. Fibre SIREN variance confirmed.
  Re-read surprise: nofibre keeps node1 POSITIVE (4-vs-1 vs all fibre slots).
- Batch 20 (@150it): BOTH premises overturned. Throughput does NOT scale with my slots; matched
  fibre pair REFUTES the fibre-collapse lead. Declared instrument binding (later disproven).
- Batch 21 (CONVERGED@2400it): Instrument was NOT binding — all 4 slots converged. substeps=6
  speedup FALSIFIED (integrator resolution science-critical). Fibre ON/OFF reversal CONFIRMED at
  depth (0.320 vs 0.241). drag40 halves overshoot (ampL 0.017), LS-neutral. Residual = loop SIZE.
- Batch 22 (CONVERGED@2400it): drag-unlock FALSIFIED. LOOP SIZE IS NOT DRIVE-LIMITED (fact #25) —
  `size` flat across amp10/11/12 + gain_hi1.5/2.5; amp>10 only adds overshoot, lowers LS. drag∈[40,50]
  flat on LS. Best LS=0.344 (drag40/amp10). Redirect size agenda off drive → stiffness/duration.
- Batch 23 (CONVERGED@2400it): NEW RECORD LS=0.365 (stiff_lo 80→50) — but via OVERSHOOT reduction, not
  size (fact #26: ampL 6× down, gain field halved, `size` flat). Softening = overshoot lever, drag40-gated.
  [80,300] and stiff_hi=400-catastrophic both OVERTURNED (facts #6, #12). Size now invariant to
  drive+stiffness+duration → likely integrator/structural.
- **Batch 24 (reissue): parametric size frontier EXHAUSTED.** amp12 + gain_hi2.5 @stiff_lo50 both refuted
  (gain_hi2.5 diverged, revealing the stiffness-floor-gated gain ceiling). Record 0.365 did not reproduce.
- **Batch 25: BOTH structural size sub-leads FALSIFIED.** substeps=14 (finer) → size flat, LS down (fact #23);
  pulse_skew=2.0 → size flat/smaller, overshoot+openness UP = a shape lever (fact #27). Size invariant to ALL
  in-model levers. stiff_hi400 matched record (0.365, ampL=0.002, cleanest ever). gain_hi=2.0 safe (fact #28).
- **Batch 26: the size frontier goes to FORWARD-MODEL STRUCTURE.** Test the two never-touched structural
  constraints (existing knobs, no code): boundary Dirichlet anchor `--bwidth` (bracket 0.03/0.10 — does freeing
  the interior grow loop size?) and settle window `--warmup` (~2 beats — under-developed limit cycle?). Plus
  exploit the cleanest-overshoot regime (stiff_hi400 repro, stiff_hi450) + control. Parent = b25 record family.
- **Batch 27: NEW MECHANISM — travelling-wave activation phase (attack area-enclosure).** B26 closed the size
  agenda (settle FALSIFIED opposite-sign; boundary a tiny lottery effect). Acting on the audit: the real residual
  is loopiness/area-enclosure (ESTABLISHED); WORKING HYPOTHESIS (B27 tests) — it is rooted in near-radial motion
  from global synchronous activation through a largely uniaxial fibre architecture. Added
  `--tw_amp`/`--tw_angle` (coarse plane-wave activation delay = AP propagation); the operator's enclosure_row
  (area_ratio/loop_ratio/minor_ratio) is the real-referenced instrument. Sweep tw dose 0/6/12/20 + direction + a
  tw-OFF LS anchor. Read the enclosure ratios + montage loop fatness, not LS alone.
- **Batch 27: travelling-wave FALSIFIED (fact #29→#30 pivot).** Staggered timing LOWERS enclosure monotonically —
  it decoheres a uniaxial pull rather than rotating the force. DEEPER: radial motion is time-reversibility-intrinsic;
  enclosure REQUIRES the axis to ROTATE during the beat. Added `--rot_stress` (phase-locked axis swing). → B28.
- **Batch 28: DECISIVE WIN — `--rot_stress` (rotating axis) is the enclosure mechanism (fact #30, ESTABLISHED).**
  rot 0→1.0: LS 0.332→0.481 (NEW RECORD, +30% over the 0.365 ceiling), area_ratio 0.10→0.36, loopiness→real, all
  MONOTONE. Rotation redistributes radial motion into circulation (energy flat, peak drops). Enclosure SHAPE solved;
  the residual SHIFTS to absolute SIZE (area/peak still ~0.4–0.5). → B29 replicates + re-opens size in the rotating regime.
- **Batch 29→31: rotating-axis mechanism REPLICATED + mapped.** rot10 redrew within noise (fact #30 stable). rot
  SATURATES for LS (peaks at 1.0; area climbs above, LS does not). B31 size-lever split (slo30 wins, amp12 doesn't)
  looked like "softening reopens" — but see B32.
- **Batch 32 (CONVERGED@2399it): FIBRE HETEROGENEITY is a DOSE-CONFIRMED lever; residual FLIPS to SIZE.** The
  fibre_dev ladder rises MONOTONE (dev 0.05/0.08/0.12/0.18 → 0.447/0.465/0.473/0.492, rolls off 0.25→0.482; peak
  dev~0.18, new op point dev18 LS=0.492). The earlier "fdev12=0.493" REGRESSED to 0.473 on replicate but the dose
  ladder rescued the lever (win = dev MAGNITUDE; fibre_wl20 SCALE inert = 0.473). Rotation SOLVED enclosure
  (loopiness ≥ real everywhere) → dominant gap is now SIZE (peak_ratio ≈0.49, area ≈0.35, loops loopy-but-inside-green).
- **Batch 33: RE-OPEN SIZE in the rotating regime.** Enclosure solved → the clean residual is peak_ratio 0.49 (sim =
  half real). Facts #24/#25 killed drive/gain/compliance as size levers but ONLY at rot=0 (radial/time-reversible).
  Test amp12/amp14 (drive) + gain_hi2.0 (ceiling) + stiff_lo20 (compliance) @rot1.0, dev18 parent + dev20 (dose peak).
  Null → size structural (boundary/constitutive); overturn → #24/#25 regime-bound to rot=0.
### Emerging observations
- **LS=0.492** is the campaign best (B32 dev18 = rot1.0 + soft-floor + fibre_dev0.18, CONVERGED@2399it), the PEAK of a
  DOSE-CONFIRMED fibre lever (monotone dev0.05→0.18: 0.447→0.492; win = dev magnitude, fibre_wl SCALE inert). Prior
  record LS=0.481 (B28 rot10) was the rotating-axis breakthrough (enclosure fill, not more force).
- **FIBRE-DOSE lesson: a regressed single-draw can still be a REAL lever — validate with an independent DOSE LADDER, not
  a replicate alone.** "fdev12=0.493" regressed to 0.473 (would have been called luck), but the monotone dev ladder
  proved fibre heterogeneity real. Report dose-response, not point estimates.
- **RESIDUAL RE-ATTRIBUTED enclosure→SIZE (B32):** with rotation, loopiness_ratio is 1.06–1.18 (≥ real) everywhere —
  enclosure is SOLVED. The clean real-referenced residual is now peak_ratio ≈0.49 (sim peak = half real) + area ≈0.35.
  This REVIVES the size question (facts #24/#25 killed drive/gain/compliance as size levers ONLY at rot=0) → B33 tests
  whether rotation makes size drive-limited.
- **rot_stress works by REDISTRIBUTION, not more force** (energy flat ~0.85, peak DROPS 0.544→0.488) — it converts
  radial (in-and-out) motion into circulation. This is why magnitude levers (drive/stiff/dur/waveform) all failed on
  enclosure but a DIRECTIONAL lever succeeds. The rot dose is MONOTONE in LS + area + loopiness + minor_axis.
- **rot_stress SIGN is weak:** rotneg (−1.0) LS 0.441 ≈ rot +1.0 (0.481), chir_match 0.833 vs 0.844 — the swing
  magnitude sets enclosure, the sign barely flips chirality. Optimizer SHORTENS the pulse under rotation (dur 9.6→6.4).
- **LS=0.365** was the prior record (B23 slo50, rot-OFF radial regime, ampL=0.004) — now superseded by rot10.
- **THREE spatial control channels confirmed:** stiffness (elastic response), gain (contraction
  amplitude), fibre dθ (orientation fine-tuning). Each is SIREN-based at ω=5.
- **Spatial gain STABILIZES both stiffness AND fibre SIRENs** — the key enabler.
- **Fibre SIREN is a HIGH-VARIANCE LOTTERY.** Peak LS=0.358 but range [0.257, 0.358],
  with catastrophic nodes (node 1 = -0.91). Constraint tightening does NOT help
  (B19: tighter → worse). The no-fibre family (LS≈0.31±0.01) is more reproducible.
- **CONFIRMED at convergence (B21): fibre is load-bearing** — fibre-ON 0.320 vs fibre-OFF 0.241
  (Δ+0.079), and fibre-OFF also has the worst overshoot (ampL 0.098). The fibre-collapse story is
  fully dead; fibre-ON is the parent.
- **INSTRUMENT (CORRECTED @B21): NOT a fixed constraint.** B21 converged all 4 slots to 2400it —
  B20's "unreachable" model was transient GPU contention. Throughput varies with workspace-wide
  load. Request n_iter=2400 and read reached depth; never pre-declare the instrument binding.
- **substeps=10 is a TWO-SIDED sweet spot (B21+B26):** substeps=6 degrades LS + triples overshoot; substeps=14
  (finer) leaves size flat and drops LS. Integrator resolution sets stability, NOT loop size.
- **RESIDUAL = AREA-ENCLOSURE (loopiness), reframed off SIZE (2026-07-04 audit + B26, fact #29) — ESTABLISHED.**
  The `size` diagnostic that looked "invariant to all levers" (B21–B26) was boundary-contaminated/sim-only; the
  real dominant deficit is loopiness (sim ~0.21 vs real ~0.50), and fibre gates it (no-fibre ablation halves area).
  B26 closed the last size leads (settle `--warmup` FALSIFIED opposite-sign = worst slot; boundary `--bwidth` a
  tiny lottery effect). WORKING HYPOTHESIS (B27 tests, not yet established): the loopiness deficit comes from
  near-RADIAL motion because global synchronous activation acts through a largely UNIAXIAL fibre architecture. → B27
  adds a TRAVELLING-WAVE activation phase to test whether spatial timing breaks radial symmetry and encloses area
  (enclosure_row area/loop/minor_ratio now logged in progress.txt).
- **Activation asymmetry + stiffness softening are SHAPE/OVERSHOOT levers, not size levers (facts #26, #27).**
  pulse_skew=2.0 (fast contract/slow release) raises overshoot + openness and ramps the gain field, but leaves
  the loop small — same as softening stiff_lo. Nothing tried converts drive/waveform energy into excursion.
- **Stiffness range is DRAG-GATED, not fixed:** widening in BOTH directions helps @drag40 (softer floor
  cleanly = record; higher ceiling with 1 negative) but NOT @drag30. drag40 is a load-bearing ENABLER
  even though LS-neutral alone. Next: does contrast STACK (stiff[50,400])?
- **Node 5 (and 0, 1) collapse stochastically** — no single node is a fixed bottleneck.
  Per-node ceiling = 0.88 (B17 node 8). Sensitivity: chirality ≈ size >> orientation.
- **Fields converge FASTER than loop morphology** — stiffness (binary domains), gain
  (checkerboard), fibre (combed) form early; per-node loop morphology differentiates by ≳1000it.
**CRITICAL: this section must ALWAYS be at the END of the file.**

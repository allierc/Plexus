# Working Memory: cardio-MPM inverse fit

The DELIVERABLE is this ledger — defensible claims about which **stiffness + direction fields** and
knobs make the MLS-MPM model reproduce the real cardiomyocyte beat (red loops superposing on green).
Interior R² (motion-normalised, boundary excluded) adjudicates; it is never the goal. A clean
falsification is a success. Read + update EVERY batch. Seeded 2026-06-23 from the forward+inverse build.

## Paper Summary (update at theme boundaries)

- **⏵ OBJECTIVE PIVOT (2026-06-24).** Renamed: NOT "find rotary parameters" — NOW **"learn which
  anisotropic active-stress MATERIAL PATTERNS generate the real loop MORPHOLOGY."**
  **2×2 result:** loops are GENERIC in active-stress MPM; **structure is NOT required for loop
  existence** (isotropic loops as much as structured: non-affine openness ~0.2–0.3 across A/B/C/D; the
  intrinsic loops are INERTIAL — overdamp `drag_k=2000` → isotropic 0.029 ≈ native line level). The
  native `p1_aniso` (overdamped graph-spring) DOES show structure→loops cleanly (structured 0.395 vs
  isotropic 0.013, 29×). **Therefore openness alone is NOT the mechanism test.** The inverse target is
  **loop MORPHOLOGY: size, axis, chirality, openness, spatial pattern, and match to real trajectories.**
  > The 2×2 test FALSIFIES "structure is necessary for loops" in the MLS-MPM port, but SUPPORTS a
  > better objective: loops are available dynamics, and structure TUNES their morphology.
  **New parent:** forward = active-stress MPM, NO rotary, uniform pulse, learn PARAMETRIC
  stiffness/gain/fibre patterns, fit real trajectories + loop-shape metrics. **Phase 1 = SHAPE ATLAS**
  (sweep pattern wavelengths/angle/beta/drag, record loop families); **Phase 2 = UCB tree** over pattern
  families, objective = real-traj R² + loop-morphology loss. Every batch reports R² · openness · ellipse
  angle err · loop size err · chirality agreement · pattern spatial coherence. (Evidence:
  `archive/aniso_loop_test/SUMMARY.md`.) The rotary/amplitude track below is SUPERSEDED but retained.
- **Model:** UNet → (stiffness, direction) fields → real decomposed MLS-MPM forward → fit one
  beat-aligned cycle; outer band anchored, interior predicted. Two interpretable learned fields.
- **Fit one aligned beat:** SOLVED — 1 model frame = 1 real frame, pulse phase-locked to the real
  onset (period ≈ 50, 5 beats), differentiable window = the FULL inter-onset interval (closed loop).
- **Does it fit?** OPEN, force track, but BREAKING THROUGH. Best HONEST R²≈**−0.189** (batch 11 **force amp25** on the
  rotary parent, drag180; avg of dup −0.219). **AMPLITUDE is the live lever and the monotone-UP CONTINUES to amp25 (Est.#27,
  b11):** amp7 −0.437 < amp10 −0.354 < amp15 −0.267 ≈ amp20 −0.267 < amp25 −0.219 (avg). ampL is cleanly MONOTONE-DOWN with amp
  (amp15 0.191 > amp20 0.147 > amp25 0.115) → the curved loops are STILL UNDER-sized even at amp25, so more amplitude keeps
  improving the energy match; the R² is flat amp15≈amp20 then breaks lower at amp25 (diminishing R²/Δamp) and the TURNOVER is
  NOT yet reached (→ amp30/amp35, b12). This OVERTURNS the pre-rotary amp-monotone-DOWN (Est.#6 was an OVERSHOOT-regime artifact):
  with curvature making motion efficient, amplitude is a SIZE knob, not an overshoot knob. **A SECOND, independent lever appeared:
  PHASE/short-pulse (b11 Q19):** spiral_md40 on amp15 (−0.208) helps as much as amp25 (Δ+0.06 over the amp15 parent) BUT τ STAYS
  TINY (used [0.32,9.9]/40, the b02/b10 signature) and dur COLLAPSES to 17.3 → it is a SHORTER-PULSE effect, NOT a spiral wave
  (Falsified#3 re-confirmed). So amplitude (size) and pulse-timing each reach ≈−0.21 from −0.267; **whether they STACK on amp25 is
  the new Q21 (b12).** **DRAG is STILL saturated at amp15 (b11 Q12 re-closed):** drag240 (−0.267) = parent exactly — higher amplitude
  does NOT re-open the drag lever (no overshoot to tame; curvature makes the extra motion useful). **ROTARY remains ESSENTIAL:** the
  rotary0 ablation at amp15 (−0.417) sits ≈0.15–0.20 below the rotary slots (the "amp15 rotary0 worse than the amp10 floor" prediction
  was FALSIFIED — −0.417 is slightly BETTER than the amp10 floor −0.461; the line-stub floor rises a hair with amplitude). The
  **ROTARY lever (scalar AND spatial) is EXHAUSTED** (Est.#24/#25/#26): `--rotary R` sweeps d(x,y,t)=Rot(R·(phase−0.5))·d, turning
  OUT-AND-BACK LINE stubs into area-enclosing CURVED arcs; the global scalar optimum is a POSITIVE-handed PLATEAU at ≈+270°–+360°
  (no turnover by +360; CHIRALITY decisive — wrong-handed side flat-then-HARMFUL); the SPATIAL per-pixel field is a NEAR-DEAD lever
  (saturates to a uniform + magnitude nudge, cannot bootstrap chirality from a 0 base, does not amplify stiffness — Falsified#6).
  **Q2 (stiffness) LOAD-BEARING under high + rotary** (Falsified#2 REVISED; the amp25 winner shows the most coherent stiffness yet,
  youngs→200). The OVERSHOOT/OPTIMIZATION track is closed (DRAG saturated Q12, sub8 dead Q13, iter600 flat Q15, passive floor
  **−0.845 DRAG-INVARIANT**); mechanism settled (M0 force ≫ M1 stress, b03). → Batch 12 pushes the amp bracket higher (amp30/amp35 —
  find the turnover), tests whether phase/short-pulse STACKS on amp25 (spiral_amp25, Q21), decouples phase from pulse-duration
  (dur0_18 — short pulse WITHOUT phase), and ablates rotary0 at amp25.

## Knowledge Base

### Comparison Table — Phase 2 (PARAMETRIC INVERSE, real-beat interior R², `cardio_mpm_train2.py`)
| Batch.slot | Config (one knob from parent) | learn | R2 | open | chir+ | size | note |
| ---------- | ----------------------------- | ----- | -- | ---- | ----- | ---- | ---- |
| **p2_b01.s5** | **fibre_amp 1.5 (WINNER; amp→1.52)** | fibre | **−5.45** | 0.258 | 0.61 | 1.46e-3 | least overshoot (ampL 0.315); high fibre_amp survives |
| p2_b01.s3 | fibre_wl 28 (finer) | fibre | −8.27 | 0.295 | 0.53 | 1.81e-3 | finer wl helps the INVERSE (opposite of forward atlas) |
| p2_b01.s1 | fibre_angle 0.3 | fibre | −10.01 | 0.265 | 0.67 | 1.97e-3 | lower angle helps |
| p2_b01.s0 | fibre_parent (fibre_amp 1.0) | fibre | −14.45 | 0.255 | 0.59 | 2.35e-3 | optimizer COLLAPSES fibre_amp→0.01 |
| p2_b01.s2 | fibre_angle 0.9 | fibre | −17.03 | 0.264 | 0.64 | 2.55e-3 | higher angle hurts |
| p2_b01.s4 | fibre_wl 52 (coarser) | fibre | −21.21 | 0.320 | 0.70 | 3.05e-3 | WORST; coarse wl hurts the inverse |

_Phase-2 regime note: all R² deeply negative — fibre alone (4 scalars) at amp10/drag30/**dur8 frozen** (16% duty) can't fit; batches 2–4 add stiff(UNet)/gain/dur, then combine (Est.#30, Q25)._

### Comparison Table — Phase 1 / rotary track (forward atlas + old UNet-force inverse)
| Batch.slot | Config | R2 | red-on-green | stiffness | direction |
| ---------- | ------ | -- | ------------ | --------- | --------- |
| init.ref   | directional_cardio, amp25 (pre-fix) | −34087 | no (overshoot) | ~uniform | noisy |
| **b11.s2** | **FORCE amp25 rotary+360 rfield±90 drag180 (WINNER, best honest of ALL batches; amp monotone-up continues)** | **−0.189** (avg −0.219) | slightly BIGGER curved arcs, more on green (best ampL 0.115) | MOST coherent (connected yellow network, youngs→200) | coherent; field SATURATES + |
| b11.s3     | FORCE spiral_md40 amp15 rotary+360 rfield±90 drag180 (phase on amp15 — helps but shorter-pulse) | −0.211 (avg −0.208) | arcs on green ≈ parent | purple + green/yellow blobs | coherent; **τ TINY [0.32,9.9]/40** (b02 sig); dur→17.3, ampL 0.215 |
| b11.s0     | FORCE amp15 rotary+360 rfield±90 drag180 (b11 parent = b10 winner, reproduced) | −0.242 (avg −0.267) | small curved arcs partly on green | coherent yellow network (youngs→200) | coherent; field saturates + |
| b11.s1     | FORCE amp20 rotary+360 rfield±90 drag180 (amp UP, flat vs amp15) | −0.279 (avg −0.267) | curved arcs ≈ parent | teal-washed network (purple holes) | coherent; field saturates + |
| b11.s4     | FORCE drag240 amp15 rotary+360 rfield±90 (drag STILL saturated at amp15 = parent) | −0.261 (avg −0.267) | arcs ≈ parent | purple + green/yellow blobs | coherent; field saturates + |
| b11.s5     | FORCE rotary0 amp15 drag180 (b11 mechanism ABL — rotary still essential) | −0.418 (avg −0.417) | out-and-back LINE stubs + tiny loops (zero-area; ampL 0.327) | INERT purple interior + bright frame | coherent domains |
| **b10.s2** | **FORCE amp15 rotary+360 rfield±90 drag180 (b10 WINNER; amp optimum FLIPPED UP)** | **−0.261** | bigger curved arcs ON green (loops no longer under-sized) | MOST coherent (connected yellow network, youngs→200) | coherent; field SATURATES +90° |
| b10.s1     | FORCE spiral_md40 amp10 rotary+360 rfield±90 drag180 (phase ON rotary — mild help, NO spiral) | −0.325 | arcs on green ≈ parent | coherent green/yellow blobs | coherent; **τ TINY [0.31,9.9]/40** (b02 signature), dur→18.6 |
| b10.s4     | FORCE spread45 (±45°) amp10 rotary+360 drag180 (field→scalar as spread→0) | −0.345 | small curved arcs ≈ parent | coherent yellow blobs | coherent; field saturates +45° (between scalar & ±90) |
| b10.s0     | FORCE amp10 rotary+360 rfield±90 drag180 (b10 parent = b09 winner, reproduced) | −0.354 | small curved arcs partly on green | coherent yellow network (youngs→200) | coherent; field saturates +90° |
| b10.s3     | FORCE amp7 rotary+360 rfield±90 drag180 (amp DOWN → under-driven) | −0.437 | OFF, under-driven line-ish stubs | yellow blobs (less than winner) | coherent; field saturates +90° |
| b10.s5     | FORCE rotary0 amp10 drag180 (b10 mechanism ABL = b06/b07/b08/b09 rotary0, reproduced) | −0.461 | out-and-back LINE stubs (zero-area) | INERT purple interior + bright frame | coherent domains |
| **b09.s2** | **FORCE rfield_b360 SPREAD±90° drag180 amp10 (b9 winner; but only Δ+0.012 over scalar)** | **−0.341** | small curved arcs on green (≈ scalar control) | scattered small blobs (LESS coherent than scalar s0) | coherent; field SATURATES to +90° (uniform + nudge) |
| b09.s0     | FORCE scalar+360 drag180 amp10 (b9 parent = b08 winner, field OFF, reproduced) | −0.353 | small curved arcs on green | MOST coherent of batch (large connected yellow network, youngs→180+) | coherent domains |
| b09.s1     | FORCE rfield_b360 SPREAD±180° drag180 amp10 (wider spread HURTS) | −0.388 | small arcs, worse than control | mostly purple + few blobs (less coherent) | coherent; field saturates +180° |
| b09.s4     | FORCE rfield PURE 0-base SPREAD±360° (UNet does NOT find + sense from 0) | −0.457 | OFF, ≈ ablation | INERT yellow-frame + purple | coherent-ish; field BALANCED/zero-mean (no + bias) |
| b09.s3     | FORCE rfield_b360 SPREAD±360° drag180 amp10 (wide → flips chirality → ≈ ablation) | −0.462 | OFF, ≈ ablation | INERT yellow-frame + purple | noisier; field red(+) w/ noise |
| b09.s5     | FORCE rotary0 drag180 amp10 (b9 mechanism ABL = b06/b07/b08 rotary0, reproduced) | −0.464 | out-and-back LINE stubs (zero-area) | INERT purple interior + bright frame | coherent domains |
| b08.s2     | FORCE ROTARY+360° drag180 amp10 (b8 winner; magnitude PLATEAU; reproduced −0.353 as b09.s0) | −0.351 | curved area-enclosing loops ON green | MOST coherent of ANY batch (connected yellow network, youngs→180) | coherent domains |
| b08.s1     | FORCE ROTARY+270° drag180 amp10 (≈ winner, within noise) | −0.357 | more curved arcs/loops on green | MORE coherent bright-yellow domains (youngs→200) | coherent domains |
| b08.s0     | FORCE ROTARY+180° drag180 amp10 (b8 parent = b07 winner, reproduced) | −0.400 | curved arcs/some closed loops on green | purple + green/yellow blobs + frame | coherent domains |
| b08.s3     | FORCE ROTARY−180° drag180 amp10 (wrong handedness ≈ rotary0) | −0.450 | line-stub-ish (wrong sense), barely > rotary0 | purple + scattered blobs (less coherent) | coherent domains |
| b08.s5     | FORCE rotary0 drag180 amp10 (b8 mechanism ABL = b06 drag180 parent, reproduced) | −0.459 | out-and-back LINE stubs (zero-area) | INERT purple interior + bright frame | coherent domains |
| b08.s4     | FORCE ROTARY−270° drag180 amp10 (wrong handedness + magnitude = WORSE than rotary-off) | −0.494 | biggest/most-wasted motion (ampL 0.381), below rotary0 | purple + scattered green/yellow blobs | coherent domains |
| b07.s2     | FORCE ROTARY+180° drag180 amp10 (b7 winner; reproduced −0.400 in b8) | −0.394 | CURVED/elliptical red arcs ON green (area-enclosing); LOWEST ampL 0.275 | MOST coherent yet (green domains, youngs→180) | coherent sharper domains |
| b07.s1     | FORCE ROTARY+90° drag180 amp10 | −0.440 | curved arcs, better-on-green than parent | purple+coherent green blobs+frame | coherent domains |
| b07.s4     | FORCE drag240 amp10 (drag asymptote holds, now DOMINATED by rotary) | −0.449 | line stubs slightly smaller than drag180 | purple+faint blobs+frame | coherent domains |
| b07.s3     | FORCE ROTARY−90° drag180 amp10 (WRONG handedness ≈ parent) | −0.468 | line stubs ≈ parent (no benefit) | purple+green blobs+frame | coherent domains |
| b07.s0     | FORCE rotary0 drag180 amp10 (b7 parent = b06 winner, reproduced) | −0.472 | out-and-back LINE stubs (zero-area) | purple+scattered blobs+frame | coherent domains |
| b06.s2     | FORCE drag180 amp10 md0 (b6 winner) | −0.464 | smallest/most-contained loops | SMOOTHEST (least texture, faint frame) | coherent (dx one big domain) |
| b06.s1     | FORCE drag150 amp10 md0 | −0.475 | slightly smaller than drag120 | washed purple, texture fading + frame | coherent domains |
| b06.s4     | FORCE drag120 iter600 (NOT optimization-limited: ≈parent) | −0.482 | ≈parent | washed purple, less texture (dur drifted→52.9) | coherent, smoother |
| b06.s0     | FORCE drag120 amp10 md0 (b6 parent, reproduces b05 −0.502 at −0.488) | −0.488 | small/contained | purple + green/yellow blobs + frame | coherent domains |
| b06.s3     | FORCE drag120 sub8 (brake EXHAUSTED: < parent) | −0.490 | ≈parent | interior teal/green texture + frame | coherent domains |
| b05.s2     | FORCE drag120 amp10 md0 (b5 winner) | −0.502 | smallest/most-contained loops | uniform-low+frame + MOST interior texture (green/yellow blobs) | coherent domains |
| b05.s1     | FORCE drag90 amp10 md0 | −0.519 | tighter than drag60 parent | dark interior + yellow blobs + frame | coherent domains |
| b05.s3     | FORCE drag60 sub8 amp10 md0 (brake stacks WEAKLY) | −0.544 | slightly better than drag60 | interior teal texture + frame | coherent domains |
| b05.s0     | FORCE drag60 amp10 md0 (parent, =b04.s3 reproduced) | −0.592 | small/contained | uniform-low+frame, faint blobs | coherent smooth domains |
| b04.s3     | FORCE drag60 amp10 md0 (b4 winner; reproduced −0.592 in b5) | −0.602 | smallest/most-contained, least overshoot | uniform-low+frame + interior texture | coherent domains |
| b05.s4     | FORCE drag60 amp5 md0 (amp brake FAILS to stack: < drag60) | −0.620 | WORSE than drag60 parent | dark interior + scattered blobs | coherent domains |
| b04.s5/b05.s5/b03.s4 | FORCE amp0 md0 (TRUE-zero ABLATION = MOTION FLOOR, drag-invariant) | −0.845 | red=tiny passive stubs, no loops | SALT-AND-PEPPER NOISE (untrained) | SALT-AND-PEPPER NOISE (untrained) |
| b04.s5     | FORCE sub8 amp10 md0 (drag=spec) | −0.737 | better-aligned than parent | uniform-low+frame | coherent domains |
| b04.s1     | FORCE amp5 md0 (drag=spec) | −0.799 | small, best-aligned of standard slots | uniform-low+bright frame | coherent domains |
| b04.s0     | FORCE amp10 md0 drag=spec (b4 parent) | −0.978 | small, modest align | uniform-low+frame (inert) | coherent smooth domains |
| b04.s2     | FORCE drag30 amp10 md0 (below spec default) | −1.045 | same as parent, slightly worse | uniform-low+frame | coherent |
| b03.s1     | FORCE amp10 md0 lr1e-3 (b3 winner) | −1.045 | small, modestly aligned | uniform-low + frame, faint texture | coherent smooth domains |
| ~~b02.s4~~ | ~~STRESS amp10 md40~~ −0.845 **FALSIFIED: NaN artifact** (b03 clean stress = −117) | n/a | n/a (NaN, blank) | n/a (NaN, blank) |
| b02.s5     | force amp10 md40 lr1e-3 | −1.138 | no | uniform-low + frame | coherent; τ used[0,9.9]/40 |
| b02.s0     | force amp10 md0 lr2e-3 | −1.141 | no | uniform-low + frame | coherent domains |
| b03.s2     | STRESS amp5 md0 (best stress) | −8.452 | no (overshoot) | low+frame+blobs | coherent |
| b03.s0     | STRESS amp10 md0 (clean) | −117.0 | no (red spaghetti streaks) | uniform-low+frame | coherent blocky |
| b03.s3     | STRESS amp20 md0 | −208.4 | no (wild streaks) | green mottled | coherent |
| b03.s5     | STRESS amp10 md40 | −407.6 | no (streaks) | green structured | coherent; τ tiny[0.25,2.3]/40 |
| b03.s4     | STRESS amp0→25 (ablation INVALID, trainer bug) | −1081 | no (worst) | green structured | coherent |
| b01.s3     | force amp10 w_amp0.3 (b1 winner) | −1.433 | no (best aligned) | uniform-low + frame | coherent domains |
| b01.s4     | force amp25 w_amp0.3 | −2.122 | no (worst) | uniform-low + strong frame | coherent domains |

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
6. **Amplitude is the dominant overshoot knob (Q3) — but the SIGN of its optimum is REGIME-DEPENDENT (REVISED b10, Est.#27).**
   In the PRE-ROTARY/overshoot regime amplitude was monotone DOWN (batch 1, force, static direction): amp10 R²=−1.43 > amp15 −1.55
   > amp25 −2.12 — lower amplitude reduced wrong-direction overshoot. The fit improved ~1700× from init (−34087 fixed → −20 init →
   −1.4 tuned). BUT this monotone-DOWN is an OVERSHOOT-REGIME property: once rotary makes the excursion CURVE (efficient), the amp
   optimum FLIPS to monotone UP (b10: amp7 −0.437 < amp10 −0.354 < amp15 −0.261) → see Est.#27. So amplitude is the overshoot knob
   ONLY while motion is wasteful out-and-back; with curvature it becomes a SIZE knob that grows under-sized loops onto green.
7. **A COHERENT direction field emerges (Q1 = yes).** Every batch-1 slot's learned dx/dy are smooth
   low-frequency domains, NOT salt-and-pepper noise. The UNet self-organises orientation. BUT the
   domains do not yet reproduce the real per-node beat directions (R² still <0) — coherence ≠ correct.
8. **Motion MAGNITUDE matching ≠ good fit; DIRECTION is the bottleneck.** amp25 matched real motion
   energy best (ampL=0.010) yet had the WORST R² (−2.12); amp10 under-shot energy (ampL=0.150) but fit
   best. So R² is gated by directional alignment, not by getting the displacement magnitude right.
9. **The learnable pulse duration self-tunes to ≈ one period.** From dur0=30 it converged to ~47–53
   (period≈52) in every well-behaved slot. lr5e-3 broke this (ran to 111 ≫ period) and hurt R²:
   keep lr ≤ 2e-3 for stable duration learning. (lr1e-3 also beats lr2e-3 on R²: b02.s5 −1.138 vs
   b02.s2 −1.195 at md40 — lower lr is consistently better.)
10. **M0 force >> M1 stress for the INVERSE REAL-FIT (Q7 CLOSED, batch 3 clean — reverses batch 2; OLD INVERSE-FORCE-TRACK ONLY — SUPERSEDED by Phase 1 atlas which uses ACTIVE-STRESS).** ~~With the
    NaN-guard ON and matched amp10/lr1e-3/md0, clean active-stress is catastrophic (b03.s0 R²=−117) while
    the force body-force wins decisively (b03.s1 R²=−1.045)~~ **[This comparison tested the UNet inverse trainer under FORCE vs STRESS; Phase 1 atlas uses ACTIVE-STRESS without issue; the criticism applies only to the old inverse-force track.]** The batch-2 stress −0.845 was a NaN ARTIFACT
    (blank field panels = the forward had already diverged; the metric was computed on a degenerate state).
    The mechanism question is SETTLED FOR THE FORCE INVERSE; the atlas PIVOT uses ACTIVE-STRESS as the forward mechanism.
11. **The active-stress forward is NUMERICALLY CHAOTIC; force is reproducible.** Each b03 slot was
    submitted TWICE (same config). The two R² diverge wildly for stress (s3 −980 vs −208; s5 −2727 vs −408;
    s0 −14 vs −117) but are nearly identical for force (s1 −1.027 vs −1.045). Stress overshoot is
    seed-sensitive/unstable — another reason it is unusable here, beyond the much worse mean R².
12. **Stress can spike non-finite gradients (NaN-guard, still valid).** b02.s4 (stress×phase) drove
    log_dur+UNet to NaN — clip_grad_norm leaves a NaN norm so opt.step() corrupted the params. FIXED: skip
    opt.step() when the clipped grad norm is non-finite. The guard worked in batch 3 (no blank panels) —
    which is exactly how the −0.845 was exposed as the artifact it was.
13. **`--amplitude 0` did NOT ablate (trainer bug, FIXED + VERIFIED).** `cardio_mpm_train.py:202` read
    `amp = args.amplitude or spec_default`, so `--amplitude 0` (falsy) fell through to the spec default
    25.0; b03.s4 "amp0 ablation" actually ran amp25-stress (progress.txt amp=25.0, ampL=537, worst −1081).
    FIXED: `--amplitude` default → −1 sentinel ("<0=spec; 0=true zero"); selection `spec if
    args.amplitude<0 else args.amplitude`. b04.s4 confirms the fix (progress.txt amp=0.0) — the floor is now measured (#14).
14. **DRAG (overdamping) is the lever that breaks the −1.0 force plateau (Q11 SUPPORTED, b04).** Monotonic in
    drag: drag30 (−1.045) < spec-default (−0.978) < drag60 (−0.602, new best across ALL batches). More drag
    suppresses momentum-driven wrong-direction overshoot → smallest, most-contained red loops. drag30 being
    WORSE than the parent places the spec default drag_k in (30,60); only damping ABOVE the default helps.
    (NB `--drag_k` default 0 is falsy → spec default; `if args.drag_k` at train.py:232 means 0 cannot zero drag.)
15. **The MOTION FLOOR is −0.845, and it BEATS the active parent — the reframing (Q6/Q8, b04.s4).** A TRUE
    amp0 run (no active contraction) gives R²=−0.845: passive boundary-anchoring alone explains most interior
    motion. The active force at amp10/default-drag (−0.978) is WORSE than this floor, i.e. it injects
    NET-HARMFUL overshoot. Only amp5 (−0.799), sub8 (−0.737), drag60 (−0.602) clear −0.845. The bar is −0.845,
    not 0 — and most settings fail it. (Partial Q4: a large share of interior R² is the anchored boundary, not the fit.)
16. **Field learning REQUIRES active force (b04.s4).** At amp0 the dx/dy AND stiffness fields stay at
    salt-and-pepper init and dur stays at 30.0 (no field gradient without active force). So the coherent
    direction DOMAINS seen in every active slot are a LEARNED product of the active-force gradient, not init.
17. **Substeps is a real fidelity lever, not just stability (Q5, b04.s5).** sub8 (−0.737) > sub5 parent
    (−0.978): some residual misalignment was a numerical-integration artifact. UNIFYING THEME of b04: the
    three improvements (drag↑, substeps↑, amplitude↓) are ALL overshoot-suppression — nailing Est.#8 that
    OVERSHOOT (not magnitude/mechanism) is the bottleneck, with DRAG the strongest single brake.
18. **The DRAG monotone is SUSTAINED but DIMINISHING — optimum is near (Q12, b05).** drag60 (−0.592) →
    drag90 (−0.519, Δ+0.073) → drag120 (−0.502, Δ+0.017, new best of all batches). The gain stays monotone
    and motion is NOT yet starved (ampL even rises slightly at drag120, 0.387 vs 0.355) but the marginal
    return collapses ~4× per +30. The over-damp turnover has not been reached but is close — bracket it with
    drag150/180. (Spec default drag_k ∈ (30,60) from b04; useful range is roughly [60,~150].)
19. **The overshoot brakes DON'T cleanly STACK — they tap one reservoir (Q13 PARTLY FALSIFIED, b05).** On the
    drag60 base, sub8 helps WEAKLY (−0.544 vs −0.592) but amp5 makes it WORSE (−0.620 vs −0.592) — even though
    amp5 ALONE beat the old parent (b04 −0.799 vs −0.978). Once drag has drained the overshoot, cutting
    amplitude just STARVES useful aligned motion (ampL rose to 0.438). The three brakes (drag/substeps/amp) are
    NOT additive; drag is dominant and near-sufficient. Don't co-tune amp down with high drag.
20. **The passive motion floor (−0.845) is DRAG-INVARIANT (Q14 ANSWERED, b05.s5 + b06.s5).** amp0+drag60 AND
    amp0+drag120 both = −0.845, IDENTICAL to the b03/b04 amp0 floor (drag=spec). Drag does NOT reshape the passive
    boundary-driven motion at ANY level; it wins PURELY by taming ACTIVE overshoot. Clean attribution of the entire
    drag win to active-overshoot suppression. (amp0 fields = untrained salt-and-pepper noise again — re-confirms Est.#16.)
21. **DRAG is SATURATED — asymptotic, no turnover (Q12 CLOSED, b06).** The monotone CONTINUES past drag120 with
    NO reversal out to drag180 (drag120 −0.488 → drag150 −0.475 → drag180 −0.464), gains ~constant at ~+0.012/+30
    and ampL stays ~0.386 (motion NOT starved — the predicted over-damp collapse to dots never occurs). Drag is a
    shallow PLATEAU, not a peak: useful but effectively maxed; chasing it buys ~0.01/step. Stiffness texture is
    NON-monotone in drag — it PEAKS ~drag120 and WASHES OUT by drag180, so it is a mid-drag transient, not a
    load-bearing fit component (refines Falsified#2). Bottom line: the overshoot/damping lever is exhausted.
22. **The residual misfit is ARCHITECTURE/LOSS-limited, NOT optimization-limited (Q15 CLOSED, b06.s4 — PIVOTAL).**
    50% more iterations (iter600 −0.482) do NOT beat the 400-it parent (−0.488, Δ+0.006 ≈ noise) and the learnable
    duration drifts UP to 52.9 without helping. The coherent-but-wrong direction field does NOT refine onto green
    with more training. ⇒ training longer / lower lr is a dead end; the fit is gated by the model's expressiveness.
23. **STRUCTURAL: a scalar envelope × static direction can only make LINE loops, not the real ELLIPSES (b06) —
    now CONFIRMED by the rotary fix (b07, Est.#24).** The force forward is F=amplitude·a(t)·d with a(t) a SCALAR pulse
    and d(x,y) a STATIC unit field → every node moves OUT-AND-BACK along one axis (degenerate, zero-area loop). The real
    green per-node loops are area-enclosing ELLIPSES. The model structurally LACKED the rotational DOF to trace one —
    the mechanistic ROOT of "direction is coherent-but-wrong" (Est.#7/#8) and of why drag/substeps/iters couldn't close
    the gap. Supplying the DOF (`--rotary`) is the FIRST lever to beat drag and turns red stubs into curved arcs (Est.#24).
24. **ROTARY (rotational DOF) is the directional lever — line→ellipse CONFIRMED, and HANDEDNESS matters (Q16, b07).**
    `--rotary R` rotates the body-force direction d(x,y,t)=Rot(R·(beat_phase−0.5))·d through R radians over the beat,
    so each node's force sweeps an angle → the excursion CURVES (area-enclosing) instead of out-and-back. Positive rotary
    monotonically beats the rotary0 parent: rotary0 (−0.472) < +90° (−0.440, Δ+0.032) < +180° (−0.394, Δ+0.078) — the best
    HONEST R² across all batches and the first lever to beat drag, by SHAPE not overshoot. Red goes from degenerate LINE
    stubs (rotary0) to CURVED arcs on green. HANDEDNESS is real: −90° (−0.468) ≈ the parent (−0.472), the SAME line-stub
    morphology, while +90° clearly helps → the real beat rotates with a DEFINITE sense (the +/CCW sweep); wrong chirality
    buys nothing (strong evidence the gain is genuine elliptical tracing, not just added motion). The winner has the LOWEST
    ampL (0.275) yet the BEST R² — curvature makes motion EFFICIENT, replacing wasted out-and-back overshoot (re-decouples
    R² from motion energy, Est.#8). Magnitude still CLIMBING at π (+180 > +90); optimum is at/beyond π (bracket b08).
25. **The GLOBAL scalar rotary optimum is a POSITIVE-handed PLATEAU at ≈+270°–+360°; wrong handedness is flat-then-HARMFUL (Q17 CLOSED, b08).**
    The positive monotone CONTINUES past π but SATURATES: rotary0 (−0.459) < +90 (−0.440) < +180 (−0.400) < +270 (−0.357) < +360 (−0.351), with
    +270→+360 Δ+0.006 = noise → the optimum is a shallow plateau spanning ≈+270° to +360° (≈ one full force-direction turn over the beat); there is
    NO turnover/degradation even at a full turn. CHIRALITY is decisive and the handedness gap WIDENS with magnitude: at matched |R| positive ≫ negative
    (+180 −0.400 vs −180 −0.450, Δ0.050; +270 −0.357 vs −270 −0.494, Δ0.137). The WRONG-handed side is non-monotone and DAMAGING — −180 (−0.450) ≈
    rotary0 (−0.459) but −270 (−0.494) drops BELOW the rotary-off ablation: over-rotating the wrong way is WORSE than not rotating. ⇒ the rotary gain is
    genuine CORRECT-SENSE elliptical tracing, not "any rotation/extra motion" (rotary0 and −270 BOTH have the highest ampL 0.381 yet the worst R²; Est.#8).
    The scalar rotary is now bracketed and EXHAUSTED as a lever → make it SPATIAL (b09, the new `--rotary_field`).
26. **The SPATIAL rotary field is a NEAR-DEAD lever; the ROTARY frontier (scalar+spatial) is EXHAUSTED (Q18 CLOSED, b09).** A
    LEARNABLE per-pixel rotary deviation R(x,y) around the +360 base gives only Δ+0.012 at TIGHT spread (±90 −0.341, the new
    best of all batches) and is HARMFUL wider: R² is MONOTONE in spread — ±90 (−0.341) > scalar/0 (−0.353) > ±180 (−0.388) >
    ±360 (−0.462 ≈ rotary0 −0.464). THREE clean sub-results: (a) the field does NOT learn "spatial fiber rotation" — it
    SATURATES to +spread (red almost everywhere, a few blue islands), i.e. a near-UNIFORM positive MAGNITUDE nudge (eff ~+450°
    at ±90), consistent with the +positive plateau still rising a hair past +360; so the marginal win is just slightly more
    positive magnitude, not spatial structure. (b) a WIDE spread that can flip local chirality COLLAPSES to the ablation level
    (±360 −0.462 ≈ rotary0) — the locally wrong-handed islands cancel enclosed area (re-confirms chirality, Est.#25). (c) the
    +360 scalar base is a NEEDED PRIOR: from a 0 base the field stays BALANCED/zero-mean (no global + bias) and collapses to
    rotary0 (s4 −0.457 ≈ −0.464) — the UNet CANNOT bootstrap global chirality from zero; it must be seeded by the scalar and the
    field only refines around it. The new `--rotary_field` code ran CLEAN (all field panels finite, no b02-style NaN artifact) —
    it is now runtime-VALIDATED; its verdict is simply that it is a marginal lever. ⇒ the fit is stuck at ≈−0.34; pivot OFF rotary.
27. **AMPLITUDE RE-OPENED — its optimum FLIPS UP in the rotary regime (Q20 ANSWERED-YES, b10; new best of all batches).** On the
    rotary parent (force/+360/rfield±90/drag180) amplitude is MONOTONE UP: amp7 (−0.437) < amp10 (−0.354) < amp15 (−0.261, best
    HONEST R² of ALL batches, Δ+0.08). This OVERTURNS the pre-rotary amp-monotone-DOWN (Est.#6) and re-confirms it as an
    OVERSHOOT-regime artifact: with a STATIC direction every added amplitude unit was wrong-direction overshoot, so less was better;
    once rotary makes the excursion CURVE/area-enclosing, the loops are EFFICIENT but UNDER-SIZED vs green, so MORE amplitude grows
    them toward green. ampL CONFIRMS the size reading: amp15 has the LOWEST ampL (0.182 = best motion-energy match) AND best R²,
    amp7 the highest (0.396) and worst — at amp10/amp7 the curved loops were UNDER-driven (note this RE-COUPLES R² with energy match
    in the rotary regime, unlike the overshoot regime where they decoupled, Est.#8). Biggest one-knob jump since rotary itself.
    Amplitude is now a SIZE knob, not an overshoot knob. **The monotone-UP CONTINUES to amp25 (b11), no turnover yet:** amp15 −0.267
    ≈ amp20 −0.267 < amp25 −0.219 (avg of dup); amp25 is the new best of ALL batches (−0.189 single). ampL stays cleanly MONOTONE-DOWN
    (amp15 0.191 > amp20 0.147 > amp25 0.115) → the loops are STILL under-sized even at amp25 (energy still climbing toward green). R²
    is FLAT amp15≈amp20 then breaks lower at amp25 — diminishing R²/Δamp but NO peak; the turnover (where extra size becomes overshoot)
    is still UNREACHED (b12 pushes amp30/amp35). NB the b11 drag re-test FALSIFIED "higher amp re-opens drag" (drag240 at amp15 −0.267
    = parent) — the bigger excursions are NOT wasteful overshoot a brake can tame, they are useful curved motion (consistent with the
    SIZE-knob reading). Amplitude (size) and the phase/short-pulse lever (Q19/Q21) are so far INDEPENDENT, each reaching ≈−0.21.

28. **LOOPS ARE GENERIC in the active-stress MPM continuum; structure TUNES morphology, it is not the
    source (the 2×2 test, 2026-06-24 — the OBJECTIVE PIVOT).** A 2×2 (boundary {free wall, anchored ring}
    × structure {isotropic, structured p1_aniso patterns}) under active-stress + uniform pulse, judged on
    NON-AFFINE (global-affine-removed) per-node loop openness: A free+iso 0.314 · B free+struct 0.211 ·
    C anch+iso 0.321 · D anch+struct 0.214 → **D/C = 0.7×**, i.e. isotropic loops AS MUCH as structured.
    A drag sweep on C/D shows the isotropic loops are INERTIAL: drag30 C=0.321 → drag600 0.305 → drag2000
    **0.029** (≈ native line level 0.013), and only overdamped does structure flip the right way (D/C=1.9×)
    but WEAKLY (struct 0.057 ≪ native 0.395). The NATIVE `p1_aniso` (overdamped graph-spring) is clean:
    structured 0.395 vs isotropic 0.013 (29×). ⇒ MLS-MPM's inertia/PIC ringing makes per-node loops even
    for UNIFORM material, so loops are "available dynamics"; structure modulates their SHAPE. New objective
    = inverse-tune loop MORPHOLOGY (size/axis/chirality/openness/spatial pattern) to the real beat with
    PARAMETRIC active-stress patterns, NOT rotary. Files: `archive/aniso_loop_test/`. New op `mpm_anchor`
    (boundary/substrate rest-anchor); per-particle gain in `pulse_to_active_stress`; spec
    `material_aniso_cardio`.

29. **MORPHOLOGY ATLAS (Phase 1, b11): pattern params decouple along morphology axes; fibre WAVELENGTH controls ellipticity/axis-angle; STIFFNESS wavelength is INERT; AMPLITUDE collapses without inverse structure; DRAG trades openness for size (Est.#Q22, Phase 1). [FORWARD ATLAS ONLY — Phase 2 inverse shows DIFFERENT ranking, see Est.#30.]** The 2×2 test falsified "structure required for loops" — loops are inertial, available without structure, so the objective pivots to: learn which ANISOTROPIC ACTIVE-STRESS patterns generate the REAL loop MORPHOLOGY. Forward-sweep `cardio_mpm_atlas.py` on `material_aniso_cardio` base (stiff_wl 8, gain_wl 26, fibre_wl 16, fibre_angle 0.6, amp 10, drag 30): (s0) base → openness 0.258, aspect 0.23, angle 1.54, size 5.32e-03, chirality 0.47. (s1) fibre_angle=0 → open↑ 0.303, chir↓ 0.42 — fibre rotation couples openness/chirality. (s2 WINNER) fibre_wl=32 → aspect↑ 0.34, angle↑ 2.29, open 0.276, chir↑ 0.51 — **coarser fibre INCREASES ellipticity and major-axis rotation.** (s3) stiff_wl=24 → no visible morphology change — stiffness wavelength is INACTIVE. (s4 FAILED) amp=25 → collapsed (open raw 0.013, size 1.09e-03) — naive forward cannot harness high amplitude; inverse inverse-training context is required (Est.#27 showed amp25 was best there). (s5) drag=300 → open↑ 0.306, angle↑ 2.77, size↓ 1.95e-03 — extreme drag maximizes openness/angle but shrinks absolute size (inertial→quasi-static, open-thin vs closed-fat trade-off). **Finding: Pattern parameters decouple cleanly — fibre wavelength controls loop SHAPE (ellipticity/rotation), NOT just size; drag-amplitude-stiffness have secondary/non-linear effects. Phase-2 inverse will tune the best atlas family (fibre_wl40 leading from Phase 1 batches 11–15) to real per-node morphology distribution.**

30. **PARAMETRIC INVERSE (Phase 2 b1, learn=fibre): fibre_amp is the CRITICAL fibre param — the optimizer KILLS it at default init; only HIGH init (1.5) survives AND wins. All R² deeply negative (−5 to −21) — fibre alone (4 scalars) at amp=10/drag=30/dur=8 is far from fitting the real beat; the other levers (stiff UNet, gain, dur) are needed.** The Phase-2 parametric inverse (REAL-beat fit with `cardio_mpm_train2.py`, learn=fibre, 300 iterations) shows: (a) the fibre_amp=1.0 parent COLLAPSES amp to 0.01 (the optimizer zeroes out fibre structure because at these settings anisotropy HURTS by generating wrong-direction overshoot), R²=−14.5. Only fibre_amp=1.5 init SURVIVES (converges to 1.52) and wins (R²=−5.45), because the higher-amplitude fibre pattern creates a denser interference lattice in the active stress that constrains motion (ampL=0.315, UNDER-driven = smallest overshoot). (b) fibre_wl is a SLOW gradient lever (moves <0.5 in 300 iterations), so init placement matters: wl=28 (finer, R²=−8.3) > wl=40 (R²=−14.5, but confounded by amp collapse) > wl=52 (coarser, R²=−21.2, WORST). Finer wavelength helps the inverse by providing more spatial resolution, even though the forward atlas found coarser wl=40 most elliptical. (c) Lower fibre_angle helps: 0.3 (−10.0) > 0.6 (−14.5) > 0.9 (−17.0), though confounded by amp survival. (d) The REGIME is deeply negative because dur=8 (frozen, learn=fibre) is far below the period (~50) → 16% duty cycle → violent kick → inertial ringing → wrong-direction overshoot. The other levers (stiff, gain, dur) are needed to close the gap. (Evidence: archive/p2_b01_s0–s5.)

### Mechanism: learnable per-pixel rotary field -- IMPLEMENTED + runtime-VALIDATED (b09)
- `--rotary_field>0` adds a UNet output channel → a per-pixel rotary DEVIATION map R(x,y)=`rotary_spread`·tanh(o) ∈
  [−spread,+spread] rad; `dir_at` then rotates EACH pixel by (`--rotary`+R(x,y))·(beat_phase−0.5), so chirality/magnitude
  vary spatially (the real myocardium has spatially-varying fiber rotation). The scalar path (`--rotary_field 0`) is
  byte-identical to b08 (rfield=None ⇒ dir_at unchanged) so parent/ablation slots are safe. The learned field renders in the
  dashboard 3rd column ("learned rotary field (rad, dev)", RdBu). RUNTIME-VALIDATED in b09: all 4 field slots produced finite
  RdBu maps (no blank/NaN), and the verdict is Est.#26 — the field SATURATES to +spread (uniform magnitude nudge, not spatial
  chirality), helps only marginally at tight spread, and cannot bootstrap chirality from a 0 base.
- NOTE phase τ (`--max_delay>0`) and rotary/rotary_field are INDEPENDENT UNet channels + forward paths (`pulse_env` τ vs `dir_at`
  rotary) → they COEXIST (UNet out=3+phase+rfield). So a rotary+phase "spiral" slot is valid (b10 s1 spiral_md40).

### Mechanism: active stress (M1) -- IMPLEMENTED + VALIDATED
- `pulse_to_active_stress` writes `sigma_act = +amplitude*a(x)*n n^T` to `H.active_stress`; `p2g` adds
  it to the elastic stress before the affine scatter (additive, default-off; snow/liquid untouched).
  Trainer toggle `--mechanism {force,stress}`; stress-gain amplitude needs its OWN calibration (amp 200
  on real-data scale overshoots ~7x; ~amp 30 ballpark).
- **Validated** by `active_stress_test.py` (4 specs `material_active_{horizontal,vertical,radial_in,
  swirl}`, the active-stress counterparts of `material_directional_*`): with axis n the sheet contracts
  ALONG n (horiz->x -6.2%, vert->y -6.3%, radial->isotropic -4.3%, swirl -3.8%) with ~ZERO centroid
  drift, vs the body-force specs which DRIFT / contract perpendicular. Confirms active stress acts
  through its divergence (no net force), the "direction = contraction AXIS" reading.
- **Sign:** `+A n n^T` (cardiac active tension) gives contraction along n in this MPM convention;
  `-A n n^T` contracts perpendicular (fixed empirically via the test).

### Falsified Hypotheses
1. **"Zero-motion collapse is the active failure mode, and w_amp is needed to defend the fit" — FALSIFIED
   in the amp10–25 regime (b01.s0/s1/s2).** No slot collapsed: with w_amp=0 (s1) the sim still kept
   substantial motion (ampL=0.075, dur→47) and R² was only marginally worse than w_amp=1.0 (−1.64 vs
   −1.52). GD is failing by directional MISALIGNMENT, not by sliding into the tiny-dot basin. w_amp is
   a weak knob here; keep it small (0.3) but it is not the lever. (Re-test at very low amplitude, where
   collapse may re-emerge.)
2. **"The UNet will structure the stiffness field to fit" — FALSIFIED in the overshoot regime (b01–b06) but now PARTLY
   RE-INSTATED: stiffness IS LOAD-BEARING under high-magnitude positive rotary (Q2, b08).** In the pre-rotary/overshoot
   regime learned stiffness stayed ~uniform-low interior with a bright anchored-boundary frame, the fit carried by direction;
   interior texture rose with drag, PEAKED ~drag120, then WASHED OUT by drag180 (a non-load-bearing mid-drag transient). BUT
   b08 OVERTURNS this once the correct rotational DOF exists at sufficient magnitude: the +270°/+360° rotary winners show the
   MOST coherent stiffness of ANY batch — large CONNECTED bright-yellow domains (youngs→180–200) forming a network pattern,
   qualitatively unlike the inert purple+frame, while the rotary0 ablation in the SAME batch stays inert purple+frame. So
   stiffness has flipped from non-load-bearing to a genuine fit component when the direction (now elliptical, correct-sense)
   is right. Q2 is RE-OPENED as ANSWERED-YES under rotary; b09 tests whether the spatial rotary field amplifies this further.
3. **"A travelling-wave phase-delay τ(x,y) bends the red loops onto green (under the force mechanism)"
   — FALSIFIED (b02, force phase sweep).** The no-phase md0 control (−1.141) beats EVERY nonzero
   max_delay: md20 −1.217, md40 −1.195, md60 −1.164, md80 −1.253. The learnable τ self-organises into a
   coherent low-frequency map but converges to a SMALL delay (max used ~9–19 f, far under the allowed
   max and the period ≈50) and mildly HURTS R². Phase is not the lever for directional misalignment in
   the force mechanism. (Now CLOSED under stress too — see #5.) **PARTIAL sign-flip ON rotary (b10, Q19):**
   with the ellipse present md40 mildly HELPS (−0.325 > parent −0.354) instead of hurting — but τ STILL
   stays tiny (used [0.31,9.9]/40, same signature) and dur collapses to 18.6, so it is a shorter-pulse
   effect, NOT the predicted travelling/spiral wave. The "spiral" mechanic remains unsupported.
   **RE-CONFIRMED on the amp15 parent (b11, Q19):** phase HELPS more clearly now (spiral_md40 −0.208 avg vs
   amp15 parent −0.267, Δ+0.06 — as big as the amp25 gain) but the SAME signature holds: τ stays TINY (used
   [0.32,9.9]/40), dur COLLAPSES to 17.3, ampL RISES to 0.215. So the gain is again SHORTER-PULSE, not a
   travelling/spiral wave — the spiral mechanic is still unsupported. What HAS changed is the regime: under
   rotary, phase flipped from harmful (b02) to a real independent ≈−0.21 lever (Q21 tests whether it stacks
   with amplitude / is just pulse-duration, decoupled by the b12 dur0_18 short-pulse-without-phase slot).
4. **"Active stress M1 gives the coordinated motion the body force can't / is the breakthrough" —
   FALSIFIED FOR THE INVERSE-FORCE-TRACK (b03 clean; OLD ONLY — SUPERSEDED by Phase 1 atlas which WORKS with active-stress).** ~~The batch-2 −0.845 stress winner was a NaN artifact (degenerate state, blank
   panels). With the NaN-guard ON and matched amp10/lr1e-3/md0, clean stress is catastrophic (−117, wild
   overshoot streaks) and force md0 (−1.045) wins by ~100×. Stress is also run-to-run chaotic.~~ **[This result is from the UNet inverse trainer comparing FORCE vs STRESS; Phase 1 atlas uses active-stress freely and achieves good morphology without issue — the criticism is specific to the inverse real-fit under the force/UNet paradigm, not to active-stress as a forward mechanism.]** M0 force is the mechanism for the inverse-force-track (Est.#10/#11).
5. **"Phase τ behaves differently under STRESS than under force" — FALSIFIED (b03.s5).** stress_md40
   (−408) ≪ stress_md0 (−117); τ self-organised to a TINY delay (used [0.25,2.3] of 40) — the SAME
   small-τ signature as under force. Phase is a non-lever in both mechanisms.
6. **"A learnable SPATIAL rotary field beats the global scalar / rediscovers the global sense / amplifies
   stiffness" — FALSIFIED on all three counts (b09, Est.#26).** (a) BEATS scalar: only at TIGHT spread and by
   Δ+0.012 (±90 −0.341 vs scalar −0.353 ≈ noise); MONOTONE-worse wider (±180 −0.388, ±360 −0.462 ≈ ablation) —
   spatial structure is NOT the lever, the field just saturates to a uniform + magnitude nudge. (b) REDISCOVERS
   the + sense from a 0 base: NO — the pure 0-base field stays balanced/zero-mean and collapses to rotary0 (s4
   −0.457 ≈ −0.464); the scalar base is a needed prior. (c) AMPLIFIES stiffness (the b09 hypothesis extending
   Falsified#2's revision): NO — the SCALAR control s0 has the MOST coherent stiffness of the batch (connected
   yellow network); the field FRAGMENTS it into scattered blobs. The load-bearing stiffness (Est.#25) is a
   +360-MAGNITUDE-regime property, NOT enhanced by the spatial DOF.

7. **"Structure is NECESSARY for loops" / "rotary force is required for loops" — FALSIFIED (the 2×2,
   Est.#28).** In the MLS-MPM port isotropic material loops as much as structured (D/C=0.7×) because the
   loops are INERTIAL, available without any structure or rotation. So rotary was a SCAFFOLD compensating
   for a force-based, inertial model — NOT a cardiomyocyte property. Honest statement: *a force-based
   inertial model needed a rotary correction to recover loops* ≠ *cardiomyocytes require rotary force*.
   The tell was Falsified#6 (the learned rotary field saturated to a near-uniform scalar = a constant
   correction, not a latent field). SUPPORTED replacement objective: loops are available dynamics, and
   anisotropic active-stress STRUCTURE tunes their MORPHOLOGY (→ Est.#28, the new parent).

### Open Questions
- **Q22 [EXTENDED b12 — Phase 1 atlas denser sampling — Est.#29 REFINED].** **FIBRE WAVELENGTH sweeps CONTINUOUSLY separate ellipticity and major-axis angle:** fibre_wl16 (b11 base) aspect 0.23 → wl24 0.27 → wl32 0.34 → wl40 0.35. The trend is MONOTONE UP through wl40 (the coarser, the more elliptical). Angle similarly climbs: wl16 1.54 → wl24 1.74 → wl32 2.29 → wl40 3.06 rad. **Fibre_wl40 is the morphology LEADER (most elliptical 0.35, most rotated 3.06 rad)** — closest to rich elliptical structure in real cardiomyocytes. **Fibre angle (0.6→0) DECOUPLES openness from chirality:** removing rotation OPENS the loops (0.276→0.322 openness) but KILLS handedness (chir 0.51→0.45), showing angle sets the morphology chirality. **Amplitude trade:** amp15 delivers 10× larger loops (5.1e-02 vs 5.3e-03 at amp10) but COLLAPSES openness (0.225 vs 0.276) — in forward-atlas, high amplitude drives inertial/closed loops (inefficient), unlike the inverse regime where learned structure harnesses amp25. **Stiffness wavelength (still inert in b12).** **Phase-1 winner = fibre_wl40** (the morphology extreme; Phase 2 will inverse-tune this family and explore amp/drag/angle variants to match real distribution).
- **Q24 [REFINED b14 — Phase 1 atlas clarifies amplitude/angle/drag trade-offs on fibre_wl40].** The forward atlas (b14) on fibre_wl40 reveals non-monotone morphology trade-offs: (1) **AMPLITUDE INVERSE EFFECT:** amp15/amp20 COLLAPSE in the forward atlas (amp15 aspect↓0.24, angle↓0.11, size↑3.94e-02 inertial overshoot; amp20 aspect↓0.06 flattened, size↓3.29e-03 tiny, chir↓0.12 lost), contradicting the inverse finding where amp15 was the winner. The divergence arises because forward is unstructured random init while inverse learns structure; **Phase 2 should return to the amp10–15 bracket, NOT push to amp20+.** (2) **FIBRE ANGLE REVERSAL (Est.#29 FALSIFIED):** angle0.3 (vs parent 0.6) shows aspect 0.32 (similar), angle 1.91 (half), BUT chir 0.58 (BEST — opposite direction of prediction "angle=0 kills handedness"). The real data's chirality is orthogonal to fibre-angle; this parameter axis decouples handedness from rotation. (3) **DRAG OPENS BUT KILLS ROTATION:** drag60 highest open 0.276, aspect 0.36, but angle 0.06 (lost rotation) — the inertial→quasi-static trade-off. (4) **AMPLITUDE0 ABLATION FAILS:** all metrics zero, confirming active-stress required (Est.#16 re-confirmed). **WINNER:** parent s0 (fibre_wl40 balanced) for Phase-2 inverse on the real beat, targeting morphology match via learned gain/fibre-angle variants (NOT amplitude up). Inverse on real data should constrain amplitude to the 10–15 range, letting learned pattern structure carry the fit.
- **Q1.** [PARTLY ANSWERED b01: coherent domains DO emerge] — do they ever match the real per-node
  beat directions, or do they stay coherent-but-wrong? This is now the central question (R² gate).
- **Q2.** [ANSWERED-so-far b01: stiffness stays uniform-inert] — can anything (mechanism change, a
  stiffness-only regularizer, longer training) make stiffness do useful work, or is direction enough?
- **Q3.** [ANSWERED b01: amplitude IS the dominant overshoot knob, monotonic] — how low does amplitude
  go before motion is too small (collapse) vs the sweet spot? amp5/amp0 probed batch 2.
- **Q4.** [PARTLY ANSWERED b04: the floor is high] the amp0 passive floor (−0.845) shows a LARGE share of
  interior R² is just the anchored boundary — so the anchored metric DOES overstate the active fit's value.
  Still open whether it tracks the FREE (un-anchored) beat.
- **Q5.** [ANSWERED b04.s5: substeps IS a fidelity lever] sub8 (−0.737) > sub5 (−0.978); finer substeps
  reduce misalignment, not just instability. Open: how high is worth it (sub10) vs cost; how low can `--grad` go?
- **Q6.** [ANSWERED b04.s4: NO collapse — the opposite] the TRUE amp0 floor is −0.845 (not a collapse to
  zero-motion basin; passive boundary motion dominates). w_amp remains a weak knob. Closed.
- **Q7.** [CLOSED b03: NO — M0 force ≫ M1 stress] clean stress is catastrophic (−117) and chaotic; force
  md0 wins (−1.045). The batch-2 stress signal was a NaN artifact. Mechanism settled = directional force.
- **Q8.** [CLOSED b04: motion floor = −0.845 (force, true amp0)] — and it BEATS the active parent (−0.978):
  active force at amp10/default-drag is net-harmful overshoot. The bar to beat is −0.845. (Stress side closed b03.)
- **Q9.** [CLOSED b03.s5: NO] phase τ does NOT help under stress either (md40 −408 ≪ md0 −117; τ tiny).
- **Q10.** [ANSWERED b03: direction still carries it] even with stress rendering (no NaN), stiffness stays
  uniform-low+frame under the force winner; stress slots show more stiffness texture but in service of
  overshoot, not fit. Q2 stands.
- **Q11.** [ANSWERED-YES b04: DRAG is the lever] raising drag_k suppresses overshoot and pulls red onto
  green: drag60 (−0.602) breaks the −1.0 force plateau, monotonic above the spec default. The strongest brake found.
- **Q12.** [CLOSED b06: SATURATED, no turnover] the monotone CONTINUES out to drag180 (−0.464) with no reversal,
  gains ~constant (~+0.012/+30) and ampL not starved — drag is an asymptotic PLATEAU, not a peak. The over-damp
  collapse prediction was FALSIFIED; drag is effectively maxed (~0.01/step). One last bracket (drag240, b07) only
  to confirm the asymptote vs a very late turnover; the lever is functionally exhausted regardless.
  **RE-CLOSED at amp15 (b11): higher amplitude does NOT re-open drag** — drag240 (−0.267 avg) = the amp15 parent
  exactly (ampL even rises to 0.220 = slight motion starve). The hypothesised "more amp → more overshoot for drag
  to tame" is FALSIFIED: under rotary the extra amplitude is useful curved motion, not wasteful overshoot, so there
  is nothing for drag to brake. Drag is saturated independent of amplitude.
- **Q13.** [CLOSED b06: NO] sub8 on the drag120 base (−0.490) is WORSE than the parent (−0.488) — the one brake
  that helped weakly at drag60 no longer helps once drag has drained the overshoot reservoir. Brakes tap one
  (now-empty) reservoir; drag alone is the lever and it is saturated.
- **Q14.** [CLOSED b06: DRAG-INVARIANT at all levels] amp0+drag120 = −0.845 EXACTLY = b03/b04/b05 floor → drag
  wins purely by taming ACTIVE overshoot, never by reshaping passive motion. Higher drag does NOT move the floor.
- **Q15.** [CLOSED b06: NOT optimization-limited — PIVOTAL] iter600 (−0.482) ≈ 400-it parent (−0.488); the
  coherent-but-wrong direction field does NOT refine with 50% more training (dur drifts up uselessly). The misfit
  is ARCHITECTURE/LOSS-limited. ⇒ stop tuning overshoot/optimization; add a richer DIRECTIONAL mechanism.
- **Q16.** [ANSWERED-YES, b07 — line→ellipse CONFIRMED] A ROTARY force DOES bend red onto green and beats drag180:
  rotary_p180 (−0.394) ≫ rotary0 (−0.472), the best honest R² of all batches and the first lever to beat drag, by
  SHAPE not overshoot. Red goes from line stubs to curved arcs. HANDEDNESS matters (−90 ≈ parent, +90 helps → definite
  rotation sense). Magnitude still climbing at π. The directional bottleneck (Q1) is now addressable. → Est.#24.
- **Q17.** [CLOSED b08 — optimum is a POSITIVE PLATEAU at ≈+270°–+360°, chirality decisive] The positive monotone CONTINUES
  past π but SATURATES (+180 −0.400 < +270 −0.357 < +360 −0.351, Δ+0.006 = noise; no turnover at a full turn). Wrong handedness
  is flat-then-HARMFUL (−180 −0.450 ≈ rotary0 −0.459; −270 −0.494 BELOW rotary-off) ⇒ correct-sense elliptical tracing, not
  "any motion." Stiffness IS now LOAD-BEARING under the +270/+360 winners (Q2, Falsified#2 revised). → Est.#25.
- **Q18.** [CLOSED b09 — spatial rotary is a NEAR-DEAD lever, rotary frontier EXHAUSTED] Spatial beats the scalar only at TIGHT
  spread by Δ+0.012 (±90 −0.341, new best) and is MONOTONE-worse wider (±360 −0.462 ≈ ablation); the field SATURATES to +spread
  (a uniform magnitude nudge, NOT spatial chirality); from a 0 base it stays zero-mean and collapses to rotary0 (scalar base is a
  NEEDED PRIOR — no chirality bootstrap); and it does NOT amplify stiffness (the scalar control is the most coherent). → Est.#26,
  Falsified#6. The rotary lever (scalar+spatial) is done; pivot OFF rotary (b10).
- **Q19.** [ANSWERED b10 — sign-FLIPS vs b02 but NOT a spiral] Phase ON rotary mildly HELPS (md40 −0.325 > parent −0.354, Δ+0.029),
  reversing b02 where every nonzero md HURT (Falsified#3). BUT τ STAYS TINY (used [0.31,9.9]/40, the SAME b02 small-τ signature) and
  the dur COLLAPSED to 18.6 — so it is NOT forming a travelling/spiral wave; the gain is a SHORTER-PULSE effect, not coordinated
  propagation. The "spiral wave" hypothesis is NOT supported. **RE-CONFIRMED on amp15 (b11):** phase helps MORE clearly now
  (spiral_md40 −0.208 avg vs amp15 parent −0.267, Δ+0.06 ≈ the amp25 gain) but the SAME signature holds — τ tiny ([0.32,9.9]/40),
  dur collapsed to 17.3, ampL up to 0.215 → still a SHORTER-PULSE effect, NOT a spiral. Phase is now a real INDEPENDENT ≈−0.21
  lever under rotary. NEW open thread → Q21. (Distrust any blank/NaN τ panel — none occurred.)
- **Q20.** [ANSWERED-YES b10, EXTENDED b11 — amp optimum FLIPS UP and the monotone CONTINUES to amp25, Est.#27] Amplitude is MONOTONE
  UP in the rotary regime: amp7 (−0.437) < amp10 (−0.354) < amp15 (−0.267) ≈ amp20 (−0.267) < amp25 (−0.219 avg, −0.189 single = new
  best of ALL batches). ampL is cleanly MONOTONE-DOWN (0.191 > 0.147 > 0.115) → loops STILL under-sized at amp25. The pre-rotary
  amp-monotone-DOWN (Est.#6) was an overshoot-regime artifact; amplitude is now a SIZE knob. STILL OPEN: the turnover is UNREACHED —
  R² is flat amp15≈amp20 then breaks lower at amp25 (diminishing R²/Δamp, no peak). b12 pushes amp30/amp35.
- **Q21.** [NEW b12 — do the two ≈−0.21 levers STACK?] Amplitude (SIZE, amp25 −0.219) and phase/short-pulse (TIMING, spiral_md40 on
  amp15 −0.208) each independently lift the −0.267 parent to ≈−0.21. b12 tests: (a) spiral_amp25 — does phase ADD to amp25 (→≈−0.15
  if additive) or tap the same reservoir (→≈−0.21)? (b) dur0_18 — does forcing a SHORT pulse WITHOUT phase reproduce the spiral gain
  (⇒ the lever is pulse-DURATION, phase incidental), or does duration just self-tune back to ~period (⇒ phase is REQUIRED to hold the
  short pulse, Est.#9)? This decouples phase from pulse-duration and tells whether to pursue a duration knob or the phase channel.
- **Q25.** [NEW Phase-2 b2 — does UNet stiffness improve R² in the PARAMETRIC inverse?] Phase-2 b1 (learn=fibre) established the
  fibre baseline (R²=−5.448, s5 winner with fibre_amp=1.5). Batch 2 freezes fibre at s5's converged values and learns stiffness
  (UNet→youngs pattern). Questions: (a) does stiffness learning lower R² from the −5.45 baseline? (b) does the stiffness RANGE
  (stiff_lo/stiff_hi) matter — wide vs narrow? (c) does finer fibre (wl=28 from s3) combine better with stiffness? (d) does
  slightly more amplitude (12 vs 10) address the under-driving (ampL=0.315)? Ablation = uniform stiffness (stiff_lo=stiff_hi=100).

---

## Previous Batch Summaries
**RULE: keep the last 4, oldest→newest, before `## Current Batch`.**

**Phase 1 Atlas Batch 1b (2026-06-24, MORPHOLOGY ATLAS — parametric variants on fibre_wl40, forward atlas NOT inverse):** Forward
sweep of active-stress patterns under uniform pulse, no rotary, no phase. Base = material_aniso_cardio with fibre_wl 40. Ranking by
morphology: s0 parent (open 0.262, aspect 0.35, angle 3.06) > s4 drag60 (most open/elliptical but lost rotation) > s3 angle0.3 (best
chirality 0.58) > s1 amp15 (inertially collapsed) > s2 amp20 (degenerate) > s5 amp0 (ablation, all zero). HEADLINES: (1) Amplitude
COLLAPSES in forward atlas (amp15/20 → degenerate), contradicting inverse where learned structure harnesses high amp. (2) Fibre angle
reversal: angle0.3 chirality UP to 0.58 (opposite prediction). (3) Drag opens but kills rotation. (4) Parent s0 = balanced morphology
winner for Phase-2 inverse.

**Phase 2 Batch 1 (2026-06-24, PARAMETRIC INVERSE learn=fibre, parent material_aniso_cardio fibre_wl40/angle0.6/amp1.0/phase0.7,
amp=10, drag=30, dur0=8; archive prefix p2_b01_*):** Swept fibre init params: s0 parent (control), s1 fibre_angle=0.3, s2
fibre_angle=0.9, s3 fibre_wl=28, s4 fibre_wl=52, s5 fibre_amp=1.5. All 6 ran 300 iterations. Ranking: s5 fibre_amp=1.5 (R²=−5.448,
WINNER) > s3 wl=28 (−8.267) > s1 angle=0.3 (−10.005) > s0 parent (−14.451) > s2 angle=0.9 (−17.029) > s4 wl=52 (−21.213). HEADLINES:
(1) ALL R² deeply negative (−5 to −21) — fibre alone (4 scalars) far from fitting the real beat at amp=10/drag=30/dur=8 (Est.#30).
(2) fibre_amp is THE critical fibre param: the parent COLLAPSED fibre_amp from 1.0→0.01 (optimizer zeroes fibre structure = anisotropy
HURTS at these settings); only s5 (init 1.5→1.52) survived AND won. (3) fibre_wl is a SLOW gradient lever (barely moves from init);
finer wl=28 beats coarser wl=40/52 in the inverse (opposite of forward atlas). (4) ampL inversely correlated: s5 ampL=0.315 (under-driven)
= best R²; s4 ampL=5.071 (5× over-driven) = worst. (5) dur=8 frozen far below period ~50 → violent kick → inertial ringing. New parent
= s5 converged fibre (wl=40.2, angle=0.72, amp=1.52, phase=1.53); batch 2 advances to --learn stiff (Q25).
artifact): curvature made the loops efficient but UNDER-SIZED, so more amplitude grows them onto green; amp15 has the LOWEST ampL (0.182,
best energy match) AND best R² (re-couples R² with energy, unlike the overshoot regime). (2) Q19 — phase ON rotary SIGN-FLIPS vs b02
(md40 −0.325 mildly BEATS the parent, no longer HURTS) BUT τ STAYS TINY (used [0.31,9.9]/40, same b02 signature) and dur collapsed to
18.6 → NOT a spiral wave, just a shorter-pulse effect (Falsified#3 partially revised; spiral mechanic unsupported). (3) spread45 (−0.345)
lands BETWEEN scalar and ±90, field saturates +45° → confirms the spatial field is a near-scalar magnitude nudge (Est.#26). (4) Q2 — amp15
winner shows the MOST coherent stiffness (yellow network, youngs→200), load-bearing in the high-effective-magnitude regime (Est.#25). New
parent = FORCE/amp15/rotary+360/rfield±90/drag180; batch 11 brackets the new amp turnover (amp20/amp25), re-tests phase on amp15, re-tests
whether higher amp re-opens drag (drag240), ablates rotary0 at amp15.

**Batch 11 (2026-06-24, AMP-bracket + re-test phase/drag at higher amp, parent force/md0/rotary+360/rfield±90/amp15/lr1e-3/sub5/drag180;
archive prefix mpm_b10_*):** s0 control (amp15), s1 amp20 / s2 amp25 (amp bracket UP), s3 spiral_md40 (phase on amp15), s4 drag240 (drag
re-test), s5 rotary0 ablation. DUP submission (s0/s1/s2 twice in-dir; s3/s4/s5 renamed dup dir) — authoritative = in-dir final run, pair as
variance (Δ0.02–0.06). Ranking (avg): s3 spiral(−0.208) ≈ s2 amp25(−0.219) > [s0 amp15 ≈ s1 amp20 ≈ s4 drag240 −0.267] > s5 rotary0(−0.417);
by best-single s2 amp25(−0.189, new best of ALL batches). WINNER = s2 amp25 (cleaner mechanistic continuation than spiral). HEADLINES:
(1) Q20 — the amp monotone-UP CONTINUES to amp25, NO turnover (Est.#27): ampL cleanly MONOTONE-DOWN (0.191>0.147>0.115) ⇒ loops STILL
under-sized; R² flat amp15≈amp20 then breaks lower at amp25. (2) Q19/Falsified#3 RE-CONFIRMED — phase on amp15 (spiral −0.208) helps as
much as amp25 (Δ+0.06) BUT via SHORTER-PULSE not a spiral: τ tiny [0.32,9.9]/40, dur→17.3, ampL↑0.215. Two INDEPENDENT ≈−0.21 levers (size
vs timing) → Q21 (do they stack?). (3) Q12 RE-CLOSED — drag240 at amp15 (−0.267) = parent: higher amplitude does NOT re-open drag (the extra
motion is useful curvature, not overshoot). (4) rotary still ESSENTIAL — rotary0_amp15 (−0.417, ampL 0.327) ≈0.15–0.20 below the rotary slots;
the "amp15 rotary0 worse than the amp10 floor −0.461" prediction was FALSIFIED (−0.417 slightly BETTER — the line-stub floor rises a hair with
amp). (5) Q2 — amp25 winner has the MOST coherent stiffness (yellow network, youngs→200). New parent = FORCE/amp25/rotary+360/rfield±90/drag180;
batch 12 pushes amp30/amp35 (turnover?), tests phase-stack on amp25 + short-pulse-without-phase (Q21), ablates rotary0 at amp25.

---

## Current Batch

### Batch info
**PHASE 2 BATCH 2 [learn=stiff] — PARAMETRIC INVERSE.** The UNet learns the stiffness spatial pattern from the microscope image
while fibre params are FROZEN at the Phase-2 b1 winner's converged values (s5: fibre_wl=40.2, fibre_angle=0.72, fibre_amp=1.52,
fibre_phase=1.53). Runner = `cardio_mpm_train2.py`. Sweep: stiffness range variants (stiff_lo/stiff_hi), alternative fibre base
(s3 wl=28), amplitude variant (amp=12 to address under-driving), and ablation (uniform stiffness = no spatial variation).

### Current hypothesis
"The UNet-derived microscope stiffness pattern will improve R² from the fibre-only baseline (−5.448). The microscope image encodes
real structural variation (cell boundaries, cytoskeleton density) that should create a spatially-varying youngs modulus field making
the anisotropic contraction pattern more physically accurate. A wider stiffness range gives the UNet more expressiveness; amplitude=12
may help the under-driven winner (ampL=0.315)."

### Slots this batch
STIFFNESS SWEEP (fibre frozen at s5 winner converged values unless noted; base = material_aniso_cardio, amp=10, drag=30, dur0=8):
- s0 stiff_parent — stiff [50,150], learn=stiff (CONTROL: does stiff learning improve R² from −5.448 fibre-only baseline?)
- s1 stiff_wide — stiff [20,250], learn=stiff (ONE knob: wider range → more expressiveness for the UNet)
- s2 stiff_soft — stiff [20,80], learn=stiff (ONE knob: softer regime → different elastic response)
- s3 stiff_on_wl28 — fibre from b1 s3 (wl=28.1, angle=0.69, amp=0.34, phase=0.71), stiff [50,150], learn=stiff (test: does finer fibre + stiff combine better?)
- s4 stiff_amp12 — amplitude=12, stiff [50,150], learn=stiff (ONE knob: address under-driving ampL=0.315)
- s5 stiff_uniform_abl — stiff [100,100], learn=stiff (ABLATION: no spatial stiff variation → tests whether SPATIAL pattern matters vs just uniform elasticity)

### Emerging observations
**CRITICAL: this section must ALWAYS be at the END of the file.**

_(Phase 2 b1, 2026-06-24) Phase-2 PARAMETRIC inverse first batch: learn=fibre only (4 scalars). ALL R² deeply negative (−5 to −21)
— fibre alone at amp=10/drag=30/dur=8 is far from fitting the real beat (Est.#30). The CRITICAL finding: fibre_amp is the decisive
fibre param. The parent (fibre_amp=1.0) COLLAPSED amp to 0.01 (the optimizer kills fibre structure because anisotropy HURTS at these
settings); only the fibre_amp=1.5 init SURVIVED (→1.52) and won (R²=−5.448, ampL=0.315 under-driven = smallest overshoot). Finer
fibre_wl=28 was second-best (R²=−8.267). Coarser wl=52 was worst (R²=−21.2, ampL=5.1 massive overshoot). The forward-atlas morphology
ranking (wl=40 best) does NOT predict the inverse ranking — more spatial resolution helps the parametric fit. fibre_wl is a SLOW
gradient lever (<0.5 movement in 300 iterations), so init placement matters. WATCH batch 2: does UNet stiffness close the gap from
−5.45? does amplitude=12 help the under-driven winner? does the microscope image encode useful structural information?_

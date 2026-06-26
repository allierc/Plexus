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
- **Does it fit?** OPEN. **Phase 1 (rotary force) best:** R²≈**−0.189** (b11 amp25 rotary+360, drag180). **Phase 2 (parametric active-stress) best:** R²≈**−0.999** (b12.s0, 2400it, dur=30, gain+dur only, fibre FROZEN, stiff=[100,100]) — FIRST TIME R² crosses −1.0. The depth monotone continues: 600→−2.158, 1200→−1.411, 1800→−1.113, 2400→−0.999 (Δ per doubling: 600→1200=0.747, 1200→2400=0.412, NOT converged — need 3600+). **gain0 INIT MATTERS** (b12.s5): gain0=0.7 at 1200it (−1.210) beats gain0=0.854 at 1200it (−1.411) by Δ=0.201 — a lower gain init is a NEW lever, untested at depth. Fibre co-learning gap WIDENS with depth: frozen wins by Δ=0.064 at 2400it (was 0.026 at 1200it). Amplitude >10 CONFIRMED harmful at depth (amp12 −1.746, amp15 −2.380 vs amp10 −1.411 at 1200it). The fit is governed by 2 LEARNABLE scalars (gain + dur) + optimization depth + gain INIT, with 4 FROZEN fibre params providing the fixed material pattern.

## Knowledge Base

### Comparison Table — Phase 2 (PARAMETRIC INVERSE, real-beat interior R², `cardio_mpm_train2.py`)
| Batch.slot | Config (one knob from parent) | learn | R2 | open | chir+ | size | note |
| ---------- | ----------------------------- | ----- | -- | ---- | ----- | ---- | ---- |
| **p2_b02.s3** | **stiff [50,150] on wl28 fibre (NEW BEST)** | stiff | **−5.18** | 0.321 | 0.52 | 1.49e-3 | wl28+stiff BEATS b1 winner; ampL=0.431 (less overshoot than wl40+stiff) |
| **p2_b01.s5** | **fibre_amp 1.5 (b1 WINNER; amp→1.52)** | fibre | **−5.45** | 0.258 | 0.61 | 1.46e-3 | least overshoot (ampL 0.315); high fibre_amp survives |
| p2_b02.s5 | stiff UNIFORM ABL [100,100] (beats spatial!) | stiff | −6.48 | 0.274 | 0.62 | 1.64e-3 | NO spatial pattern → BETTER than spatial s0; ampL=0.507 |
| p2_b02.s1 | stiff WIDE [20,250] | stiff | −6.62 | 0.302 | 0.56 | 1.66e-3 | wider range helps slightly vs parent; ampL=0.764 |
| p2_b02.s0 | stiff PARENT [50,150] on wl40 fibre | stiff | −7.43 | 0.304 | 0.55 | 1.77e-3 | spatial stiff HURTS vs uniform abl; ampL=0.801 |
| p2_b01.s3 | fibre_wl 28 (finer) | fibre | −8.27 | 0.295 | 0.53 | 1.81e-3 | finer wl helps the INVERSE (opposite of forward atlas) |
| p2_b02.s4 | stiff [50,150] amp=12 | stiff | −9.74 | 0.276 | 0.52 | 1.97e-3 | amp12 HURTS — more overshoot at dur=8; ampL=1.442 |
| p2_b01.s1 | fibre_angle 0.3 | fibre | −10.01 | 0.265 | 0.67 | 1.97e-3 | lower angle helps |
| p2_b01.s0 | fibre_parent (fibre_amp 1.0) | fibre | −14.45 | 0.255 | 0.59 | 2.35e-3 | optimizer COLLAPSES fibre_amp→0.01 |
| p2_b01.s2 | fibre_angle 0.9 | fibre | −17.03 | 0.264 | 0.64 | 2.55e-3 | higher angle hurts |
| p2_b01.s4 | fibre_wl 52 (coarser) | fibre | −21.21 | 0.320 | 0.70 | 3.05e-3 | WORST; coarse wl hurts the inverse |
| p2_b02.s2 | stiff SOFT [20,80] (CATASTROPHIC) | stiff | −25.04 | 0.261 | 0.63 | 3.43e-3 | too-soft material → massive overshoot; ampL=7.278 |
| **p2_b03.s3** | **gain_dur (gain0→0.817, NEW BEST)** | gain,dur | **−4.16** | 0.273 | 0.63 | 1.36e-3 | gain SHRINKS contraction → lowest ampL 0.093 → best overlap |
| p2_b03.s0 | dur_only (CONTROL — dur moves 8→8.8) | dur | −5.08 | 0.275 | 0.62 | 1.50e-3 | tiny dur shift still helps; ampL=0.229 |
| p2_b03.s4 | fibre_dur_wl28 (amp 0.34→0.56, angle 0.69→0.56) | fibre,dur | −6.74 | 0.313 | 0.60 | 1.82e-3 | wl28 fibre more stable than wl40 for co-learning; ampL=0.736 |
| p2_b03.s2 | stiff_dur (binary yellow/purple UNet pattern) | stiff,dur | −7.27 | 0.320 | 0.57 | 1.78e-3 | stiff active; similar to b2.s0; ampL=0.745 |
| p2_b03.s1 | fibre_dur wl40 (amp 1.52→1.38, DESTABILIZED) | fibre,dur | −13.23 | 0.285 | 0.52 | 2.27e-3 | fibre co-learning on wl40 HURTS (ampL=2.481 overshoot) |
| p2_b03.s5 | all_combine (CATASTROPHIC, learn=all) | all | −16.83 | 0.290 | 0.55 | 2.47e-3 | too many DOF → optimizer destabilizes; ampL=3.785 worst |

| **p2_b04.s1** | **fibre+gain wl28 (NEW BEST, ampL=0.010!)** | fibre,gain,dur | **−2.620** | 0.306 | 0.61 | 1.20e-3 | wl28 fibre+gain SYNERGY; angle 0.69→0.17, amp 0.34→0.39; near-PERFECT energy match |
| p2_b04.s4 | gain_dur dur0=14 (HIGH-DUR BASIN EXISTS) | gain,dur | −3.880 | 0.253 | 0.62 | 1.39e-3 | dur STAYS at 14.0 bound; gain→0.838; 2nd best |
| p2_b04.s2 | gain_dur amp=12 (amp UP hurts) | gain,dur | −4.722 | 0.277 | 0.61 | 1.47e-3 | gain→0.672 compensates but not enough; amp12 net-harmful |
| p2_b04.s3 | gain_only ABL (dur frozen=8) | gain | −5.241 | 0.275 | 0.61 | 1.49e-3 | gain→0.830; dur frozen HURTS vs gain+dur (−4.16) |
| p2_b04.s5 | stiff+gain+dur wl40 (stiff HURTS) | stiff,gain,dur | −6.060 | 0.362 | 0.55 | 1.65e-3 | binary stiff pattern net-harmful; youngs→[50,350] |
| p2_b04.s0 | fibre+gain wl40 (DESTABILIZED) | fibre,gain,dur | −7.307 | 0.266 | 0.49 | 1.75e-3 | fibre_amp collapses 1.52→0.54; gain→0.746; wl40 UNSTABLE |

| **p2_b05.s3** | **fibre_frozen gain+dur dur0=14 (BATCH-5 WINNER)** | gain,dur | **−2.992** | 0.291 | 0.64 | 1.29e-3 | fibre FROZEN better than co-learned at dur=14; still < b4.s1 |
| p2_b05.s1 | combine dur_hi=20 (dur→20.0 BOUND HIT) | fibre,gain,dur | −3.142 | 0.307 | 0.55 | 1.22e-3 | dur wants >20; ampL=0.017 |
| p2_b05.s0 | combine dur0=14 (two b4 wins combined) | fibre,gain,dur | −3.383 | 0.298 | 0.62 | 1.29e-3 | did NOT stack with fibre+gain — Q28 FALSIFIED |
| p2_b05.s5 | combine drag=60 on dur14 base | fibre,gain,dur | −3.443 | 0.305 | 0.61 | 1.30e-3 | drag redundant on combined base |
| p2_b05.s2 | combine angle=0 init (Q29 test) | fibre,gain,dur | −3.671 | 0.294 | 0.59 | 1.35e-3 | angle=0 WORSE than 0.17 — Q29 answered |
| p2_b05.s4 | combine +stiff active (CATASTROPHIC) | fibre,stiff,gain,dur | −10.498 | 0.368 | 0.60 | 2.27e-3 | binary stiffness → ampL=1.621 massive overshoot |

| **p2_b06.s2** | **dur_hi30 fibre co-learn (BATCH-6 WINNER)** | fibre,gain,dur | **−2.814** | 0.259 | 0.56 | 1.08e-3 | fibre co-learn REVERSES at dur=30 (beats frozen!); dur→30.0 bound; ampL=0.003 |
| p2_b06.s0 | dur_hi30 fibre frozen | gain,dur | −3.087 | 0.230 | 0.64 | 1.19e-3 | dur→30.0 bound; fibre frozen worse than co-learned at high dur |
| p2_b06.s1 | dur_hi50 fibre frozen | gain,dur | −3.223 | 0.179 | 0.66 | 1.19e-3 | dur→49.9 bound; WORSE than dur=30 → turnover 30–50; openness collapses |
| p2_b06.s5 | fibre_amp=0 ABL (no fibre) | gain,dur | −3.224 | 0.242 | 0.56 | 1.48e-3 | fibre ablation NOT catastrophic — comparable to frozen |
| p2_b06.s3 | fibre wl24 dur_hi=14 | fibre,gain,dur | −3.317 | 0.312 | 0.57 | 1.24e-3 | wl24 doesn't help; dur→14.0 bound |
| p2_b06.s4 | fibre wl24 lowdur dur0=8 | fibre,gain,dur | −3.746 | 0.320 | 0.58 | 1.30e-3 | wl24 at low dur WORST in batch; dur→9.0 |

| p2_b07.s3 | dur_hi30 angle=0.3 (fibre perturbation, Q34) | fibre,gain,dur | −2.842 | 0.245 | 0.53 | 1.08e-3 | angle 0.3 init; WORSE than parent −2.814; ampL=0.002 |
| p2_b07.s0 | dur_hi40 fibre co-learn (dur bracket, Q33) | fibre,gain,dur | −2.871 | 0.225 | 0.53 | 1.06e-3 | dur→39.9 (bound!); dur=40 WORSE than dur=30 → optimum IS 30 |
| p2_b07.s4 | dur_hi30 fibre_amp=0.6 (fibre perturbation) | fibre,gain,dur | −2.956 | 0.260 | 0.58 | 1.10e-3 | higher fibre_amp hurts; ampL=0.010 |
| p2_b07.s2 | dur_hi30 wl=24 (wl sweep, Q31 reconfirmed) | fibre,gain,dur | −3.716 | 0.264 | 0.57 | 1.23e-3 | wl24 HURTS at dur=30 too; ampL=0.098 |
| p2_b07.s1 | dur_hi30 amp=12 (amp UP at high dur) | fibre,gain,dur | −3.719 | 0.258 | 0.58 | 1.22e-3 | amp12 HURTS at dur=30 (total impulse too high) |
| p2_b07.s5 | dur_hi30 gain FROZEN ablation | fibre,dur | −4.006 | 0.260 | 0.56 | 1.28e-3 | gain essential — ablation costs 1.19 R² units |
| p2_b08.s4 | lowdur_control NO unet_fibre (BATCH-8 WINNER) | fibre,gain,dur | −4.002 | 0.306 | 0.60 | 1.35e-3 | parametric-only BEATS all UNet fibre; stiff [50,150] frozen |
| p2_b08.s1 | unet_fibre_hidur dur0=14 dur_hi=30 | fibre,stiff,gain,dur | −6.255 | 0.339 | 0.60 | 1.91e-3 | best UNet fibre slot; dur→30.0 bound; smoother dθ map |
| p2_b08.s0 | unet_fibre_lowdur (MAIN TEST, Q35) | fibre,stiff,gain,dur | −13.664 | 0.266 | 0.57 | 2.38e-3 | UNet fibre massive overshoot; noisy dθ; ampL=2.394 |
| p2_b08.s5 | unet_fibre_noparam_abl (fibre FROZEN) | stiff,gain,dur | −14.651 | 0.303 | 0.61 | 2.48e-3 | parametric fibre learning barely matters under UNet fibre |
| p2_b08.s2 | unet_fibre_tight (fibre_dev=π/4) | fibre,stiff,gain,dur | −15.597 | 0.270 | 0.55 | 2.56e-3 | tighter dev doesn't help; ampL=3.304 |
| p2_b08.s3 | unet_fibre_stiff (stiff [50,150] active) | fibre,stiff,gain,dur | −22.666 | 0.289 | 0.55 | 2.89e-3 | spatial stiff + UNet fibre = WORST; ampL=5.659 |

| **p2_b09.s2** | **iter600_hidur (dur_hi=30, 600it, NEW OVERALL BEST)** | fibre,gain,dur | **−2.158** | 0.255 | 0.56 | 9.40e-4 | dur→29.9(bound); 600 iter BREAKS plateau; ampL=0.014; beats b4.s1 by Δ=0.462 |
| p2_b09.s4 | dur_hi25 (300it) | fibre,gain,dur | −2.810 | 0.277 | 0.56 | 1.09e-3 | dur→25.0(bound); ampL=0.003; ≈b6.s2 level |
| p2_b09.s3 | dur_hi20 (300it) | fibre,gain,dur | −2.918 | 0.291 | 0.58 | 1.16e-3 | dur→20.0(bound); ampL=0.005 |
| p2_b09.s1 | iter600_lowdur (dur_hi=14, 600it) | fibre,gain,dur | −3.884 | 0.300 | 0.58 | 1.36e-3 | 600 iter helps low dur but NOT reproducible of b4.s1 (−2.620) |
| p2_b09.s0 | clean_reproduce (dur_hi=14, 300it) | fibre,gain,dur | −5.020 | 0.310 | 0.61 | 1.49e-3 | b4.s1 NOT reproducible from converged inits + stiff=[100,100] |
| p2_b09.s5 | gain_reset_abl (gain0=1.0) | fibre,gain,dur | −6.861 | 0.314 | 0.61 | 1.70e-3 | gain init=1.0 → worst; gain warm-start critical (Δ=1.84 penalty) |

| p2_b10.s0 | free_stiff (SIREN stiff ω=30, stiff=[100,100]) | stiff,gain,dur | −2.354 | 0.228 | 0.63 | 1.07e-3 | SIREN stiff → UNIFORM (≡ ablation); 600it; dur→29.9 |
| p2_b10.s1 | free_stiff_smooth (SIREN stiff ω=15) | stiff,gain,dur | −2.354 | 0.228 | 0.63 | 1.07e-3 | ω=15 → IDENTICAL to ω=30 and ablation |
| p2_b10.s2 | free_stiff_fine (SIREN stiff ω=60) | stiff,gain,dur | −2.354 | 0.228 | 0.63 | 1.07e-3 | ω=60 → IDENTICAL to ω=30 and ablation |
| p2_b10.s3 | free_dir (SIREN fibre ω=30) | fibre,gain,dur | −7.591 | 0.302 | 0.66 | 2.00e-3 | SIREN fibre → NOISY dθ, massive overshoot ampL=0.916 |
| p2_b10.s4 | free_stiff_dir (both SIREN ω=30) | fibre,stiff,gain,dur | −11.102 | 0.285 | 0.69 | 2.24e-3 | both free → WORST; stiff still uniform, fibre noisy; ampL=1.658 |
| p2_b10.s5 | uniform_abl (CONTROL, no SIREN) | gain,dur | −2.354 | 0.228 | 0.63 | 1.07e-3 | fibre FROZEN gain+dur only; ≡ SIREN stiff slots exactly |

| **p2_b11.s5** | **iter1200_frozen_abl (FROZEN fibre, 1200it, NEW BEST)** | gain,dur | **−1.411** | 0.229 | 0.63 | 8.48e-4 | fibre FROZEN + deep gain/dur BEATS co-learn; ampL=0.083 |
| p2_b11.s0 | iter1200 (fibre co-learn, 1200it) | fibre,gain,dur | −1.437 | 0.243 | 0.57 | 7.84e-4 | co-learn 0.026 WORSE than frozen at 1200it; ampL=0.124 |
| p2_b11.s3 | fibre_phase0 (phase init=0.0, 600it) | fibre,gain,dur | −2.108 | 0.251 | 0.54 | 9.32e-4 | phase=0 marginal +0.050 over parent; basin broad |
| p2_b11.s4 | dur_hi35 (dur ceiling=35, 600it) | fibre,gain,dur | −2.159 | 0.236 | 0.55 | 9.31e-4 | dur→34.9(bound); ≈parent; dur=30 optimum confirmed |
| p2_b11.s1 | fibre_wl32 (coarser wl=32, 600it) | fibre,gain,dur | −2.199 | 0.243 | 0.58 | 9.52e-4 | wl=32 slightly worse than 28.8 |
| p2_b11.s2 | fibre_amp06 (higher amp=0.6, 600it) | fibre,gain,dur | −2.215 | 0.256 | 0.55 | 9.51e-4 | fibre_amp=0.6 slightly worse than 0.39 |

| **p2_b12.s0** | **iter2400 (WINNER, NEW OVERALL BEST, FIRST R²>−1)** | gain,dur | **−0.999** | 0.232 | 0.62 | 6.96e-4 | fibre FROZEN; 2400it; depth monotone continues; ampL=0.287 |
| p2_b12.s4 | iter2400_fibre_co (fibre co-learn at 2400it) | fibre,gain,dur | −1.063 | 0.236 | 0.62 | 6.96e-4 | co-learn Δ=0.064 WORSE than frozen — gap WIDENING with depth |
| p2_b12.s1 | iter1800 (intermediate depth checkpoint) | gain,dur | −1.113 | 0.230 | 0.62 | 7.47e-4 | fibre FROZEN; fills 1200→2400 curve; ampL=0.204 |
| p2_b12.s5 | iter1200_gain07 (lower gain init=0.7) | gain,dur | −1.210 | 0.230 | 0.62 | 7.84e-4 | gain0=0.7 BEATS gain0=0.854 at 1200it by Δ=0.201 — NEW lever |
| p2_b12.s2 | iter1200_amp12 (amplitude=12 at 1200it) | gain,dur | −1.746 | 0.228 | 0.64 | 9.37e-4 | amp12 HURTS at depth (ampL=0.022, energy-match trap) |
| p2_b12.s3 | iter1200_amp15 (amplitude=15 at 1200it) | gain,dur | −2.380 | 0.229 | 0.64 | 1.07e-3 | amp15 HURTS MORE (ampL=0.004, severe energy-match trap) |

_Phase-2 regime note: **b12.s0 (dur=30, 2400 iter, fibre FROZEN, gain+dur only, R²=−0.999) is the NEW OVERALL BEST — FIRST TIME R² crosses −1.0.** Batch 12 continues the depth push: 1800→−1.113, 2400→−0.999. The depth monotone is SUSTAINED but DECELERATING (Δ per doubling: 600→1200=0.747, 1200→2400=0.412). NOT converged per the user's Δ<0.05 criterion — need 3600+ to pin N\*. Fibre co-learn gap WIDENS with depth (0.026 at 1200it → 0.064 at 2400it — co-learning gets MORE harmful as depth increases). NEW FINDING: gain init=0.7 (b12.s5, −1.210 at 1200it) beats default 0.854 (b11.s5, −1.411) by Δ=0.201 — gain INIT is a previously unsuspected lever that must be tested at depth. Amplitude >10 re-confirmed harmful at 1200it depth (amp12 −1.746, amp15 −2.380). The spatial-field track remains DEFINITIVELY CLOSED (b10, Falsified#10/#11). The fit is governed by 2 learnable scalars (gain + dur) + optimization depth + gain INIT._

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

_Each entry is tagged by KIND (see the METHODOLOGICAL RULE in `instruction_cardio_mpm_phase2.md`):_
_`[engineering]` = implementation correctness, regime-independent → almost never revisit._
_`[mechanism]` = a claim about the model/physics → revisit when the regime (depth/dur/base/mechanism) changes._
_`[optimization@<regime>]` = a claim about the optimizer's behaviour in a regime → optimizer/depth-dependent, never promote to a mechanistic conclusion._
_Tags added 2026-06-25 in a one-time retro-pass; entries below were written before the rule, so most still carry their verdict prose ("CLOSED"/"FALSIFIED") un-regime-tagged — read those as conditional per their `[kind]`._

1. `[engineering]` **The MPM is a stable elastic limit cycle.** Points return to rest; the quiescent state is
   reproducible after one cycle (no cross-cycle drift — the spring model's old streak failure is gone).
   → warm up `no_grad` past one cycle, backprop one beat.
2. `[engineering]` **Time must be aligned to the real beat.** 1 model frame = 1 real frame; pulse period+phase locked
   to the detected real onsets (period ≈ 50). The differentiable window = the full inter-onset interval
   so the fitted loop CLOSES (matches `gt_compare.png`).
3. `[engineering]` **Amplitude was applied twice (fixed).** The activation must be the gate `env·spatial` (~[0,1]);
   `pulse_to_contraction.amplitude` does the scaling. Double-applying gave ~25× overshoot (R²≈−34087);
   fixing it → R²≈−20 at init. Amplitude is now a single, sweepable knob.
4. `[engineering]` **The honest metric is interior R²** (motion-normalised, boundary EXCLUDED, moving nodes). A small
   absolute RMSE is meaningless here (real motion ~6e-4); R²≤0 = worse than predicting no motion.
5. `[engineering]` **Dashboard GT must use the canonical selection.** 10×10 / margin-10 nodes + fixed amp ×10 (the
   `gt_trajectories.png` recipe) — auto-amp + denser sampling made loops exceed node spacing → spaghetti.
6. `[mechanism]` **Amplitude is the dominant overshoot knob (Q3) — but the SIGN of its optimum is REGIME-DEPENDENT (REVISED b10, Est.#27).**
   In the PRE-ROTARY/overshoot regime amplitude was monotone DOWN (batch 1, force, static direction): amp10 R²=−1.43 > amp15 −1.55
   > amp25 −2.12 — lower amplitude reduced wrong-direction overshoot. The fit improved ~1700× from init (−34087 fixed → −20 init →
   −1.4 tuned). BUT this monotone-DOWN is an OVERSHOOT-REGIME property: once rotary makes the excursion CURVE (efficient), the amp
   optimum FLIPS to monotone UP (b10: amp7 −0.437 < amp10 −0.354 < amp15 −0.261) → see Est.#27. So amplitude is the overshoot knob
   ONLY while motion is wasteful out-and-back; with curvature it becomes a SIZE knob that grows under-sized loops onto green.
7. `[mechanism]` **A COHERENT direction field emerges (Q1 = yes).** Every batch-1 slot's learned dx/dy are smooth
   low-frequency domains, NOT salt-and-pepper noise. The UNet self-organises orientation. BUT the
   domains do not yet reproduce the real per-node beat directions (R² still <0) — coherence ≠ correct.
8. `[mechanism]` **Motion MAGNITUDE matching ≠ good fit; DIRECTION is the bottleneck.** amp25 matched real motion
   energy best (ampL=0.010) yet had the WORST R² (−2.12); amp10 under-shot energy (ampL=0.150) but fit
   best. So R² is gated by directional alignment, not by getting the displacement magnitude right.
9. `[optimization@force-track]` **The learnable pulse duration self-tunes to ≈ one period.** From dur0=30 it converged to ~47–53
   (period≈52) in every well-behaved slot. lr5e-3 broke this (ran to 111 ≫ period) and hurt R²:
   keep lr ≤ 2e-3 for stable duration learning. (lr1e-3 also beats lr2e-3 on R²: b02.s5 −1.138 vs
   b02.s2 −1.195 at md40 — lower lr is consistently better.)
10. `[mechanism]` **M0 force >> M1 stress for the INVERSE REAL-FIT (Q7 CLOSED, batch 3 clean — reverses batch 2; OLD INVERSE-FORCE-TRACK ONLY — SUPERSEDED by Phase 1 atlas which uses ACTIVE-STRESS).** ~~With the
    NaN-guard ON and matched amp10/lr1e-3/md0, clean active-stress is catastrophic (b03.s0 R²=−117) while
    the force body-force wins decisively (b03.s1 R²=−1.045)~~ **[This comparison tested the UNet inverse trainer under FORCE vs STRESS; Phase 1 atlas uses ACTIVE-STRESS without issue; the criticism applies only to the old inverse-force track.]** The batch-2 stress −0.845 was a NaN ARTIFACT
    (blank field panels = the forward had already diverged; the metric was computed on a degenerate state).
    The mechanism question is SETTLED FOR THE FORCE INVERSE; the atlas PIVOT uses ACTIVE-STRESS as the forward mechanism.
11. `[engineering]` **The active-stress forward is NUMERICALLY CHAOTIC; force is reproducible.** Each b03 slot was
    submitted TWICE (same config). The two R² diverge wildly for stress (s3 −980 vs −208; s5 −2727 vs −408;
    s0 −14 vs −117) but are nearly identical for force (s1 −1.027 vs −1.045). Stress overshoot is
    seed-sensitive/unstable — another reason it is unusable here, beyond the much worse mean R².
12. `[engineering]` **Stress can spike non-finite gradients (NaN-guard, still valid).** b02.s4 (stress×phase) drove
    log_dur+UNet to NaN — clip_grad_norm leaves a NaN norm so opt.step() corrupted the params. FIXED: skip
    opt.step() when the clipped grad norm is non-finite. The guard worked in batch 3 (no blank panels) —
    which is exactly how the −0.845 was exposed as the artifact it was.
13. `[engineering]` **`--amplitude 0` did NOT ablate (trainer bug, FIXED + VERIFIED).** `cardio_mpm_train.py:202` read
    `amp = args.amplitude or spec_default`, so `--amplitude 0` (falsy) fell through to the spec default
    25.0; b03.s4 "amp0 ablation" actually ran amp25-stress (progress.txt amp=25.0, ampL=537, worst −1081).
    FIXED: `--amplitude` default → −1 sentinel ("<0=spec; 0=true zero"); selection `spec if
    args.amplitude<0 else args.amplitude`. b04.s4 confirms the fix (progress.txt amp=0.0) — the floor is now measured (#14).
14. `[mechanism]` **DRAG (overdamping) is the lever that breaks the −1.0 force plateau (Q11 SUPPORTED, b04).** Monotonic in
    drag: drag30 (−1.045) < spec-default (−0.978) < drag60 (−0.602, new best across ALL batches). More drag
    suppresses momentum-driven wrong-direction overshoot → smallest, most-contained red loops. drag30 being
    WORSE than the parent places the spec default drag_k in (30,60); only damping ABOVE the default helps.
    (NB `--drag_k` default 0 is falsy → spec default; `if args.drag_k` at train.py:232 means 0 cannot zero drag.)
15. `[mechanism]` **The MOTION FLOOR is −0.845, and it BEATS the active parent — the reframing (Q6/Q8, b04.s4).** A TRUE
    amp0 run (no active contraction) gives R²=−0.845: passive boundary-anchoring alone explains most interior
    motion. The active force at amp10/default-drag (−0.978) is WORSE than this floor, i.e. it injects
    NET-HARMFUL overshoot. Only amp5 (−0.799), sub8 (−0.737), drag60 (−0.602) clear −0.845. The bar is −0.845,
    not 0 — and most settings fail it. (Partial Q4: a large share of interior R² is the anchored boundary, not the fit.)
16. `[mechanism]` **Field learning REQUIRES active force (b04.s4).** At amp0 the dx/dy AND stiffness fields stay at
    salt-and-pepper init and dur stays at 30.0 (no field gradient without active force). So the coherent
    direction DOMAINS seen in every active slot are a LEARNED product of the active-force gradient, not init.
17. `[engineering]` **Substeps is a real fidelity lever, not just stability (Q5, b04.s5).** sub8 (−0.737) > sub5 parent
    (−0.978): some residual misalignment was a numerical-integration artifact. UNIFYING THEME of b04: the
    three improvements (drag↑, substeps↑, amplitude↓) are ALL overshoot-suppression — nailing Est.#8 that
    OVERSHOOT (not magnitude/mechanism) is the bottleneck, with DRAG the strongest single brake.
18. `[mechanism]` **The DRAG monotone is SUSTAINED but DIMINISHING — optimum is near (Q12, b05).** drag60 (−0.592) →
    drag90 (−0.519, Δ+0.073) → drag120 (−0.502, Δ+0.017, new best of all batches). The gain stays monotone
    and motion is NOT yet starved (ampL even rises slightly at drag120, 0.387 vs 0.355) but the marginal
    return collapses ~4× per +30. The over-damp turnover has not been reached but is close — bracket it with
    drag150/180. (Spec default drag_k ∈ (30,60) from b04; useful range is roughly [60,~150].)
19. `[mechanism]` **The overshoot brakes DON'T cleanly STACK — they tap one reservoir (Q13 PARTLY FALSIFIED, b05).** On the
    drag60 base, sub8 helps WEAKLY (−0.544 vs −0.592) but amp5 makes it WORSE (−0.620 vs −0.592) — even though
    amp5 ALONE beat the old parent (b04 −0.799 vs −0.978). Once drag has drained the overshoot, cutting
    amplitude just STARVES useful aligned motion (ampL rose to 0.438). The three brakes (drag/substeps/amp) are
    NOT additive; drag is dominant and near-sufficient. Don't co-tune amp down with high drag.
20. `[mechanism]` **The passive motion floor (−0.845) is DRAG-INVARIANT (Q14 ANSWERED, b05.s5 + b06.s5).** amp0+drag60 AND
    amp0+drag120 both = −0.845, IDENTICAL to the b03/b04 amp0 floor (drag=spec). Drag does NOT reshape the passive
    boundary-driven motion at ANY level; it wins PURELY by taming ACTIVE overshoot. Clean attribution of the entire
    drag win to active-overshoot suppression. (amp0 fields = untrained salt-and-pepper noise again — re-confirms Est.#16.)
21. `[mechanism]` **DRAG is SATURATED — asymptotic, no turnover (Q12 CLOSED, b06).** The monotone CONTINUES past drag120 with
    NO reversal out to drag180 (drag120 −0.488 → drag150 −0.475 → drag180 −0.464), gains ~constant at ~+0.012/+30
    and ampL stays ~0.386 (motion NOT starved — the predicted over-damp collapse to dots never occurs). Drag is a
    shallow PLATEAU, not a peak: useful but effectively maxed; chasing it buys ~0.01/step. Stiffness texture is
    NON-monotone in drag — it PEAKS ~drag120 and WASHES OUT by drag180, so it is a mid-drag transient, not a
    load-bearing fit component (refines Falsified#2). Bottom line: the overshoot/damping lever is exhausted.
22. `[optimization@force-track/600it]` **The residual misfit is ARCHITECTURE/LOSS-limited, NOT optimization-limited (Q15 CLOSED, b06.s4 — PIVOTAL).**
    50% more iterations (iter600 −0.482) do NOT beat the 400-it parent (−0.488, Δ+0.006 ≈ noise) and the learnable
    duration drifts UP to 52.9 without helping. The coherent-but-wrong direction field does NOT refine onto green
    with more training. ⇒ training longer / lower lr is a dead end; the fit is gated by the model's expressiveness.
23. `[mechanism]` **STRUCTURAL: a scalar envelope × static direction can only make LINE loops, not the real ELLIPSES (b06) —
    now CONFIRMED by the rotary fix (b07, Est.#24).** The force forward is F=amplitude·a(t)·d with a(t) a SCALAR pulse
    and d(x,y) a STATIC unit field → every node moves OUT-AND-BACK along one axis (degenerate, zero-area loop). The real
    green per-node loops are area-enclosing ELLIPSES. The model structurally LACKED the rotational DOF to trace one —
    the mechanistic ROOT of "direction is coherent-but-wrong" (Est.#7/#8) and of why drag/substeps/iters couldn't close
    the gap. Supplying the DOF (`--rotary`) is the FIRST lever to beat drag and turns red stubs into curved arcs (Est.#24).
24. `[mechanism]` **ROTARY (rotational DOF) is the directional lever — line→ellipse CONFIRMED, and HANDEDNESS matters (Q16, b07).**
    `--rotary R` rotates the body-force direction d(x,y,t)=Rot(R·(beat_phase−0.5))·d through R radians over the beat,
    so each node's force sweeps an angle → the excursion CURVES (area-enclosing) instead of out-and-back. Positive rotary
    monotonically beats the rotary0 parent: rotary0 (−0.472) < +90° (−0.440, Δ+0.032) < +180° (−0.394, Δ+0.078) — the best
    HONEST R² across all batches and the first lever to beat drag, by SHAPE not overshoot. Red goes from degenerate LINE
    stubs (rotary0) to CURVED arcs on green. HANDEDNESS is real: −90° (−0.468) ≈ the parent (−0.472), the SAME line-stub
    morphology, while +90° clearly helps → the real beat rotates with a DEFINITE sense (the +/CCW sweep); wrong chirality
    buys nothing (strong evidence the gain is genuine elliptical tracing, not just added motion). The winner has the LOWEST
    ampL (0.275) yet the BEST R² — curvature makes motion EFFICIENT, replacing wasted out-and-back overshoot (re-decouples
    R² from motion energy, Est.#8). Magnitude still CLIMBING at π (+180 > +90); optimum is at/beyond π (bracket b08).
25. `[mechanism]` **The GLOBAL scalar rotary optimum is a POSITIVE-handed PLATEAU at ≈+270°–+360°; wrong handedness is flat-then-HARMFUL (Q17 CLOSED, b08).**
    The positive monotone CONTINUES past π but SATURATES: rotary0 (−0.459) < +90 (−0.440) < +180 (−0.400) < +270 (−0.357) < +360 (−0.351), with
    +270→+360 Δ+0.006 = noise → the optimum is a shallow plateau spanning ≈+270° to +360° (≈ one full force-direction turn over the beat); there is
    NO turnover/degradation even at a full turn. CHIRALITY is decisive and the handedness gap WIDENS with magnitude: at matched |R| positive ≫ negative
    (+180 −0.400 vs −180 −0.450, Δ0.050; +270 −0.357 vs −270 −0.494, Δ0.137). The WRONG-handed side is non-monotone and DAMAGING — −180 (−0.450) ≈
    rotary0 (−0.459) but −270 (−0.494) drops BELOW the rotary-off ablation: over-rotating the wrong way is WORSE than not rotating. ⇒ the rotary gain is
    genuine CORRECT-SENSE elliptical tracing, not "any rotation/extra motion" (rotary0 and −270 BOTH have the highest ampL 0.381 yet the worst R²; Est.#8).
    The scalar rotary is now bracketed and EXHAUSTED as a lever → make it SPATIAL (b09, the new `--rotary_field`).
26. `[mechanism]` **The SPATIAL rotary field is a NEAR-DEAD lever; the ROTARY frontier (scalar+spatial) is EXHAUSTED (Q18 CLOSED, b09).** A
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
27. `[mechanism]` **AMPLITUDE RE-OPENED — its optimum FLIPS UP in the rotary regime (Q20 ANSWERED-YES, b10; new best of all batches).** On the
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

28. `[mechanism]` **LOOPS ARE GENERIC in the active-stress MPM continuum; structure TUNES morphology, it is not the
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

29. `[mechanism]` **MORPHOLOGY ATLAS (Phase 1, b11): pattern params decouple along morphology axes; fibre WAVELENGTH controls ellipticity/axis-angle; STIFFNESS wavelength is INERT; AMPLITUDE collapses without inverse structure; DRAG trades openness for size (Est.#Q22, Phase 1). [FORWARD ATLAS ONLY — Phase 2 inverse shows DIFFERENT ranking, see Est.#30.]** The 2×2 test falsified "structure required for loops" — loops are inertial, available without structure, so the objective pivots to: learn which ANISOTROPIC ACTIVE-STRESS patterns generate the REAL loop MORPHOLOGY. Forward-sweep `cardio_mpm_atlas.py` on `material_aniso_cardio` base (stiff_wl 8, gain_wl 26, fibre_wl 16, fibre_angle 0.6, amp 10, drag 30): (s0) base → openness 0.258, aspect 0.23, angle 1.54, size 5.32e-03, chirality 0.47. (s1) fibre_angle=0 → open↑ 0.303, chir↓ 0.42 — fibre rotation couples openness/chirality. (s2 WINNER) fibre_wl=32 → aspect↑ 0.34, angle↑ 2.29, open 0.276, chir↑ 0.51 — **coarser fibre INCREASES ellipticity and major-axis rotation.** (s3) stiff_wl=24 → no visible morphology change — stiffness wavelength is INACTIVE. (s4 FAILED) amp=25 → collapsed (open raw 0.013, size 1.09e-03) — naive forward cannot harness high amplitude; inverse inverse-training context is required (Est.#27 showed amp25 was best there). (s5) drag=300 → open↑ 0.306, angle↑ 2.77, size↓ 1.95e-03 — extreme drag maximizes openness/angle but shrinks absolute size (inertial→quasi-static, open-thin vs closed-fat trade-off). **Finding: Pattern parameters decouple cleanly — fibre wavelength controls loop SHAPE (ellipticity/rotation), NOT just size; drag-amplitude-stiffness have secondary/non-linear effects. Phase-2 inverse will tune the best atlas family (fibre_wl40 leading from Phase 1 batches 11–15) to real per-node morphology distribution.**

30. `[mechanism]` **PARAMETRIC INVERSE (Phase 2 b1, learn=fibre): fibre_amp is the CRITICAL fibre param — the optimizer KILLS it at default init; only HIGH init (1.5) survives AND wins. All R² deeply negative (−5 to −21) — fibre alone (4 scalars) at amp=10/drag=30/dur=8 is far from fitting the real beat; the other levers (stiff UNet, gain, dur) are needed.** The Phase-2 parametric inverse (REAL-beat fit with `cardio_mpm_train2.py`, learn=fibre, 300 iterations) shows: (a) the fibre_amp=1.0 parent COLLAPSES amp to 0.01 (the optimizer zeroes out fibre structure because at these settings anisotropy HURTS by generating wrong-direction overshoot), R²=−14.5. Only fibre_amp=1.5 init SURVIVES (converges to 1.52) and wins (R²=−5.45), because the higher-amplitude fibre pattern creates a denser interference lattice in the active stress that constrains motion (ampL=0.315, UNDER-driven = smallest overshoot). (b) fibre_wl is a SLOW gradient lever (moves <0.5 in 300 iterations), so init placement matters: wl=28 (finer, R²=−8.3) > wl=40 (R²=−14.5, but confounded by amp collapse) > wl=52 (coarser, R²=−21.2, WORST). Finer wavelength helps the inverse by providing more spatial resolution, even though the forward atlas found coarser wl=40 most elliptical. (c) Lower fibre_angle helps: 0.3 (−10.0) > 0.6 (−14.5) > 0.9 (−17.0), though confounded by amp survival. (d) The REGIME is deeply negative because dur=8 (frozen, learn=fibre) is far below the period (~50) → 16% duty cycle → violent kick → inertial ringing → wrong-direction overshoot. The other levers (stiff, gain, dur) are needed to close the gap. (Evidence: archive/p2_b01_s0–s5.)

31. `[mechanism]` **SPATIAL STIFFNESS (UNet) is a NEAR-DEAD lever on the wl40 fibre base — the UNIFORM ABLATION BEATS the spatial pattern (Phase 2 b2, learn=stiff, Q25 ANSWERED-NO on wl40 but YES on wl28).** (a) On the wl40/high-fibre-amp base, spatial stiffness HURTS: the uniform ablation s5 (R²=−6.48) beats the spatial parent s0 (R²=−7.43) — the UNet learns a mottled yellow/purple pattern that generates net-harmful spatial variation. (b) BUT on the wl28/low-fibre-amp base, stiffness learning HELPS: s3 (R²=−5.18) is the new Phase-2 best, beating the b1 fibre-only winner (−5.45). The finer fibre provides more spatial resolution, and the LOW fibre_amp (0.34) means less fibre-driven overshoot → stiffness can modulate motion beneficially where strong fibre cannot. (c) Wider stiffness range [20,250] helps marginally (s1 −6.62 vs s0 −7.43); the SOFT regime [20,80] is CATASTROPHIC (s2 −25.04, ampL=7.278 massive overshoot — the material can't resist contraction). (d) amp=12 at dur=8 HURTS (s4 −9.74 vs s0 −7.43) — more amplitude at short duty cycle adds overshoot (consistent with pre-rotary regime Est.#6). (e) ALL R² still deeply negative; dur=8 frozen remains the root cause — user confirmed this diagnosis and advised co-learning dur with each group. (Evidence: archive/p2_b02_s0–s5.)

32. `[mechanism]` **GAIN is the dominant OVERSHOOT lever in the parametric inverse — a single scalar SIZE brake (Phase 2 b3, Q26 PARTLY ANSWERED).** gain0 learned from 1.0→0.817, reducing the effective contraction magnitude (amp×gain = 10×0.817 = 8.17) → ampL=0.093 (LOWEST of any Phase-2 slot) → R²=−4.164 (NEW Phase-2 best, beating b2.s3 −5.18). Gain is a CLEAN size/overshoot lever: it does exactly what amplitude does but WITHIN the optimizer's control. The gain winner has the best red-on-green overlap, smallest loop size (1.36e-3), and highest chirality (0.63). The stiffness and fibre panels are frozen (same as dur_only control), so the entire R² gain is from one scalar: gain0. This parallels the old overshoot regime where drag/amplitude were the brakes (Est.#6/#14/#18), but gain is a LEARNED brake, not a swept one — the optimizer found the optimal contraction scaling autonomously.

33. `[optimization@dur0=8]` **DUR has a BIMODAL landscape — the high-dur basin EXISTS at ≥14 but is UNREACHABLE from dur0=8 (Phase 2 b3 REFINED by b4, Q26 EXTENDED).** b3 showed dur moved only 8→8.7–9.0 from dur0=8, BUT b4.s4 (dur0=14) STAYED at 14.0 (the [3,14] upper bound) AND gave R²=−3.880, beating the previous Phase-2 best (−4.164). So dur IS a real lever, with TWO basins: (a) a LOCAL minimum near ~8–9 (the optimizer gets stuck there from dur0=8); (b) a BETTER basin at ≥14 (unreachable from below, sits at the bound → the true optimum may be BEYOND 14). The dur gradient near 8 is flat/noisy → the optimizer cannot cross the barrier. Added `--dur_hi` CLI argument to widen the bound beyond 14 for exploration.

34. `[optimization@wl40-vs-wl28]` **Fibre co-learning DESTABILIZES on the wl40 base but is more stable on wl28 (Phase 2 b3).** fibre+dur on wl40 (s1, R²=−13.23) is MUCH worse than fibre-only (b1.s5, −5.45) — the optimizer moves fibre_amp 1.52→1.38 and angle 0.72→0.56 into a high-overshoot regime (ampL=2.481). But fibre+dur on wl28 (s4, R²=−6.74) is less catastrophic — fibre_amp moves UP from 0.34→0.56 and angle DOWN from 0.69→0.56, a MORE productive direction. The wl40 high-fibre-amp starting point is unstable to co-optimization; the wl28 low-fibre-amp starting point is more robust. This suggests future fibre sweeps should start from wl28 or use gain as a stabilizing co-learner.

35. `[optimization@300it]` **All-combine (learn=all) is CATASTROPHIC at this stage (Phase 2 b3.s5, R²=−16.83, ampL=3.785).** Combining fibre+stiff+gain+dur simultaneously gives the WORST R² of the batch. fibre_amp drops 1.52→1.14, gain barely moves (1.0→1.017), stiffness learns a high-contrast binary pattern — the parameters fight each other. The PARTITIONED protocol is validated: one lever at a time, combine only after each is tuned.

36. `[optimization@b4.s1/300it]` **wl28 FIBRE + GAIN CO-LEARNING is the BREAKTHROUGH combination (Phase 2 b4.s1, R²=−2.620, NEW PHASE-2 BEST).** On the wl28 fibre base with gain as co-learner: fibre_angle drives DOWN (0.69→0.17, nearly zero rotation), fibre_amp stays moderate (0.34→0.39), gain=0.854 (mild brake). Result: ampL=0.010 (VIRTUALLY PERFECT motion energy match — the red loops are almost exactly the right SIZE), R²=−2.620 (best by 1.5 units over prev best −4.164). The wl28 base is STABLE for co-optimization (confirming Est.#34 — wl40 destabilizes), and gain prevents the overshoot that killed fibre co-learning in b3. The optimizer CHOSE to minimize fibre rotation angle, suggesting near-zero fibre angle is optimal for the real beat. Contrast: wl40 fibre+gain (b4.s0, R²=−7.307) STILL destabilizes (fibre_amp 1.52→0.54, collapse), proving the instability is wl40-specific.

37. `[mechanism]` **AMPLITUDE ≥12 HURTS in the parametric inverse at dur~8–10 (Phase 2 b4.s2, R²=−4.722 vs parent −4.164, consistent with pre-rotary Est.#6).** Even with gain as a learned brake (gain compensated DOWN to 0.672), amp=12 at short duty cycle adds net-harmful overshoot. The parametric inverse (without rotary) is in the same overshoot regime as the old force-track — amplitude UP is NOT the lever. Keep amplitude at 10.

38. `[mechanism]` **GAIN-ONLY ablation (dur frozen) CONFIRMS dur is load-bearing (Phase 2 b4.s3, R²=−5.241 vs b3.s3 gain+dur −4.164).** With dur frozen at 8.0, gain alone converges to 0.830 (similar to b3.s3's 0.817) but the R² is 1.1 units WORSE. The small dur shift (8→8.7–9.0) that was dismissed as "near-inert" actually contributes ~1 R² unit. And dur0=14 contributes ~0.3 more (b4.s4 −3.880). Together: gain is the DOMINANT lever, but dur is a real SECONDARY lever — especially at the high-dur basin.

39. `[mechanism]` **SPATIAL STIFFNESS (UNet) is CONSISTENTLY harmful on the wl40 fibre base (Phase 2 b4.s5, R²=−6.060 vs gain-only b3.s3 −4.164; extends b2 Est.#31).** stiff+gain+dur produces a high-contrast binary yellow/purple stiffness pattern (youngs [50,350] — the UNet pushes BEYOND the init range) that is net-harmful. The stiffness lever on wl40 is EXHAUSTED — every attempt (b2.s0, b3.s2, b4.s5) produces net-harmful spatial patterns. On wl28 it may differ (b2.s3 helped); test in b5.

40. `[optimization@300it]` **The two b4 wins (fibre+gain + high-dur) do NOT STACK — dur=14 disrupts the fibre+gain optimum (Phase 2 b5, Q28 FALSIFIED).** Combining wl28 fibre+gain (b4.s1, −2.620 at dur≈8.7) with dur0=14 → best result −2.992 (fibre frozen) or −3.383 (fibre co-learning). Both WORSE than b4.s1. The fibre params (angle=0.17, amp=0.39) were tuned by the b4.s1 optimizer for short dur; at dur=14 they are suboptimal. The improvements tap OVERLAPPING variance — they are NOT independent levers. b4.s1 (dur≈8.7) remains the overall Phase-2 best.

41. `[optimization@dur14/300it]` **Fibre co-learning is HARMFUL at high dur — freezing is better (Phase 2 b5.s3 vs b5.s0).** At dur=14, fibre-frozen (−2.992) beats fibre-co-learning (−3.383, Δ=0.39 R² units). The b4.s1-learned fibre params are a better FROZEN prior than what the optimizer finds when re-tuning fibre at dur=14 in 300 iterations. Co-learning destabilizes because the fibre optimum landscape shifts with dur, and the optimizer overshoots in the new regime.

42. `[optimization@dur-bound]` **Duration wants BEYOND 20 — the high-dur basin center is UNREACHED (Phase 2 b5.s1, Q30 EXTENDED).** With dur_hi=20, dur converged to 20.0 (the bound) and R²=−3.142 (better than dur=14 at −3.383). The optimizer ALWAYS pushes dur to the upper bound: dur=14 at [3,14], dur=20 at [3,20]. The period is ~50; the basin center may be ~25 (half-period) or even at the period itself. Wider bounds needed.

43. `[mechanism]` **Spatial stiffness is DEFINITIVELY harmful across ALL bases (Phase 2 b5.s4, CLOSES Q2/Q25 for parametric inverse).** b5.s4 (wl28+dur14+stiff active, R²=−10.498, ampL=1.621) adds to the pile: b2.s0 (wl40+stiff, −7.43), b3.s2 (wl40+stiff+dur, −7.27), b4.s5 (wl40+stiff+gain, −6.06), NOW b5.s4 (wl28+combined, −10.50). The UNet consistently learns an extreme binary pattern that generates catastrophic overshoot. Spatial stiffness from the microscope image is NOT a useful lever in the parametric inverse — the microscope texture does not carry load-bearing material information at this resolution.

44. `[mechanism]` **Fibre angle=0 is NOT optimal — a small positive angle (~0.17 rad ≈ 10°) IS preferred (Phase 2 b5.s2 vs b5.s0, Q29 ANSWERED).** angle=0 init (−3.671) is worse than angle=0.17 init (−3.383). The b4.s1 optimizer drove angle from 0.69→0.17 — not to zero, but to a specific small positive value. A ~10° rotation of the contraction axis from the parametric fibre pattern's principal direction is part of the real beat's structure.

45. `[mechanism]` **Drag is REDUNDANT on the gain-controlled base (Phase 2 b5.s5, extends Est.#21).** drag=60 (−3.443) ≈ drag=30 (−3.383) on the wl28+dur14+gain base. With gain0=0.854 already reducing the effective contraction, additional drag provides no marginal benefit. Drag was the brake in the pre-gain overshoot regime; gain SUPERSEDES it as the learned contraction-scaling lever.

46. `[mechanism]` **Duration has a NON-MONOTONE optimum — turnover between dur=30 and dur=50 (Phase 2 b6, Q30 EXTENDED).** dur=30 (s0 frozen −3.087, s2 co-learn −2.814) > dur=50 (s1 frozen −3.223). At dur≈50 ≈ period, the pulse is near-constant → openness collapses (0.179 vs 0.230) and R² degrades. The optimum is between 30 and 50 — need dur_hi=40 to bracket. This is the FIRST evidence of a dur turnover (all previous tests hit the bound monotonically).

47. `[optimization@dur30/300it]` **Fibre co-learning REVERSES at high dur (dur_hi=30) — OVERTURNS Est.#41 (Phase 2 b6.s2 vs b6.s0).** At dur=30, fibre co-learning (s2, −2.814) BEATS fibre frozen (s0, −3.087, Δ=0.27). At dur=14 (b5), fibre co-learning HURT (−3.383 vs frozen −2.992). Explanation: the fibre optimum SHIFTS with dur; the b4.s1 fibre params (angle=0.17, amp=0.39) were tuned for dur≈8.7 and are suboptimal at dur=30. At dur=30 the optimizer has room to find the new fibre optimum — the regime is stable enough (unlike wl40). Est.#41 was a DUR-REGIME artifact, not a general rule.

48. `[mechanism]` **Fibre structure is a MODERATE lever, not essential (Phase 2 b6.s5, fibre_amp=0 ablation).** fibre_amp=0 (isotropic, −3.224) is only Δ=0.14 worse than fibre frozen (−3.087) at dur=30. Fibre anisotropy helps but is not transformative at the high-dur setting — gain and dur are the dominant levers. Consistent with Est.#28 (loops are generic/inertial; structure tunes morphology).

49. `[mechanism]` **Duration optimum is AT 30 (~60% of period) — the optimizer ALWAYS pushes dur to the ceiling, and dur=30 > dur=40 > dur=50 (Phase 2 b7.s0, Q33 CLOSED, REFINES Est.#46).** The controlled ceiling sweep: dur_hi=30 → R²=−2.814 (b6.s2), dur_hi=40 → R²=−2.871 (b7.s0, dur→39.9 bound), dur_hi=50 → R²=−3.223 (b6.s1). The turnover is between 30 and 40 (not 30–50 as estimated in b6). Duration at ~80% of period (dur=40) mildly degrades; at ~100% (dur=50) openness collapses. The optimizer's gradient ALWAYS favors longer pulses, so dur hits whatever ceiling you set — but the best ceiling is 30.

50. `[optimization@dur30/300-600it]` **Fibre-param perturbations at dur=30 do NOT improve — the b4.s1-derived inits are near-optimal for the high-dur path (Phase 2 b7, Q34 CLOSED).** angle=0.3 (b7.s3, −2.842, Δ=−0.028 vs parent), fibre_amp=0.6 (b7.s4, −2.956, Δ=−0.142), wl=24 (b7.s2, −3.716, Δ=−0.902): ALL worse than the b6.s2 parent (−2.814) which uses b4.s1-derived values (angle=0.17, amp=0.39, wl=28.8). The fibre params from b4.s1 generalize across dur regimes.

51. `[mechanism]` **Gain learning is ESSENTIAL at dur=30 — ablation degrades R² by 1.19 units (Phase 2 b7.s5, STRENGTHENS Est.#32).** s5 gain frozen at 0.854 (−4.006) vs parent with gain learning (−2.814). Gain is MORE critical at high dur (Δ=1.19) than at low dur (b4.s3 Δ=1.06): longer pulse injects more total energy → gain must brake harder. Without gain learning, ampL rises to 0.114 (overshoot).

52. `[mechanism]` **Amplitude >10 HURTS at high dur (Phase 2 b7.s1, consistent with Est.#37).** amp=12 (−3.719) is Δ=0.905 WORSE than amp=10 parent (−2.814) at dur=30. Total impulse = amplitude × duration; amp12×dur30=360 vs amp10×dur30=300 → 20% more energy → net-harmful overshoot that gain cannot fully compensate. amp=10 is the correct setting for the parametric inverse at any dur.

53. `[optimization@300→600it]` **~~BOTH dur basins are PLATEAUED~~ — REVISED: the HIGH-DUR basin was optimization-depth-limited, NOT expressiveness-limited (Phase 2 b9.s2 OVERTURNS for high-dur).** b6.s2 (300 iter, dur=30, −2.814) → b9.s2 (600 iter, dur=30, −2.158, Δ=+0.656 gain). The 4-scalar parametric fibre HAS more room with 600 iterations at dur=30. The LOW-DUR basin (b4.s1, −2.620) remains non-reproducible and fragile (b9.s0 = −5.020 from converged inits). **Use 600 iterations as default going forward.**

54. `[mechanism]` **UNet fibre-angle deviation (per-pixel microscope dθ(x,y)) is a NET-HARMFUL lever — ALL spatial UNet channels are now CLOSED (Phase 2 b8, Q35 ANSWERED-NO).** The parametric-only control (s4, R²=−4.002) DECISIVELY beats every UNet fibre slot: s1 hidur (−6.255), s0 lowdur (−13.664), s5 frozen-fibre (−14.651), s2 tight-dev (−15.597), s3 stiff+fibre (−22.666). The UNet learns a noisy/speckled dθ map at low dur (ampL 2–6 = massive overshoot) and a smoother but still harmful map at high dur (ampL=0.487). THREE sub-findings: (a) tighter deviation range (π/4 vs π/2) does NOT help (s2 WORSE than s0); (b) parametric fibre co-learning barely matters under UNet fibre (s0 ≈ s5); (c) spatial stiffness + UNet fibre = the WORST combination (s3, re-confirms Falsified#8). This CLOSES the spatial-information track: NEITHER spatial stiffness NOR spatial fibre direction from the microscope improves the parametric inverse. The microscope image does NOT carry material information at the resolution/representation the UNet can extract under 300 iterations of L2 optimization.

55. `[optimization@600it]` **600 iterations is a SIGNIFICANT optimization-depth lever at high dur (Phase 2 b9).** High-dur: 300→600 iter gains Δ=+0.656 R² (−2.814→−2.158). Low-dur: 300→600 iter gains Δ=+1.136 (−5.020→−3.884) but from a worse base and still nowhere near b4.s1 (−2.620). The high-dur basin is deeper and rewards optimization effort. **Default n_iter=600 for all future high-dur runs.**

56. `[optimization@300it]` **Gain warm-start (gain0=0.854) is CRITICAL — resetting to 1.0 costs Δ=1.84 R² units (Phase 2 b9.s5 vs b9.s0, STRENGTHENS Est.#32).** At identical config, gain0=1.0 (−6.861) vs gain0=0.854 (−5.020). The gain learned in b4.s1 is a non-trivial warm-start; the optimizer cannot recover the 1.84 deficit in 300 iters from gain=1.0.

57. `[optimization@b4.s1]` **The low-dur b4.s1 result (R²=−2.620) is NOT REPRODUCIBLE from converged-value inits (Phase 2 b9.s0/s1).** b9.s0 (stiff=[100,100], b4.s1's converged fibre/gain/dur inits, 300 it) = −5.020, a 2.4-unit degradation. b9.s1 (same, 600 it) = −3.884, still 1.26 worse. The b4.s1 convergence was trajectory-dependent: the specific init conditions (fibre_amp=0.34 from b1.s5→b2.s3, stiff=[50,150]) enabled a favorable path through the loss landscape that cannot be reached by starting at the converged endpoint. **Treat b4.s1 as a lucky local minimum, not a reproducible baseline.**

58. `[mechanism]` **FREE (SIREN) spatial stiffness is COMPLETELY INERT — converges to a UNIFORM field at ALL bandwidths (Phase 2 b10, Falsified#10).** SIREN coordinate-network stiffness (stiff_src=siren) at ω=15/30/60 produces metrics PIXEL-IDENTICAL to the uniform ablation (R²=−2.354 in all four slots, same ampL=0.003, same openness=0.228). The optimizer receives NO gradient signal for spatial stiffness variation — the loss landscape is FLAT in the stiffness-field direction. The youngs panel in all three SIREN slots shows flat uniform teal (≡ stiff=100 everywhere). Combined with Falsified#8 (UNet stiffness harmful) and Est.#43 (UNet stiffness harmful across all bases), this DEFINITIVELY closes spatial stiffness as a lever — it is inert under BOTH image-shaped (UNet) AND free (SIREN) representations. The parametric inverse's fit depends ONLY on global scalars, not spatial material patterns.

59. `[mechanism]` **FREE (SIREN) fibre-direction deviation is HARMFUL — the SAME noisy-overshoot failure mode as UNet fibre (Phase 2 b10.s3/s4, Falsified#11).**
60. `[optimization@1200it]` **1200 ITERATIONS gives another MASSIVE R² jump — the optimization-depth monotone CONTINUES beyond 600 (Phase 2 b11, Q37 ANSWERED-YES).** 600→1200 iter gains Δ=0.72–0.75: b9.s2 (600it, −2.158) → b11.s5 (1200it, −1.411, Δ=+0.747) / b11.s0 (1200it, −1.437, Δ=+0.721). This is COMPARABLE to the 300→600 jump (Δ=0.66, b9). The parametric model's expressiveness limit is STILL UNREACHED. The gain is NOT diminishing: roughly constant Δ≈0.7 per doubling. Push to 1800/2400 in b12.
61. `[optimization@1200it]` **Fibre co-learning REVERSES to SLIGHTLY HARMFUL at 1200 iter — FROZEN fibre is the new paradigm (Phase 2 b11, REVISES Est.#47).** b11.s5 frozen (−1.411) beats b11.s0 co-learn (−1.437) by Δ=0.026. At 600 iter, co-learning helped (Δ=0.196 from b10); at 1200 iter the gain reverses. The extra fibre DOF introduces gradient competition at deep optimization. The b4.s1-derived fibre params (wl=28.8, angle=0.17, amp=0.39, phase=0.41) are near-optimal and should be FROZEN; all optimization budget goes to gain+dur. This SIMPLIFIES the model to 2 learnable scalars (gain + dur).
62. `[optimization@1200it]` **Deeper optimization GROWS the red loops (ampL increases) — the 600-iter energy-match trap (Phase 2 b11).** 1200-iter ampL = 0.083 (s5) / 0.124 (s0) vs 600-iter ampL ≈ 0.003–0.019. The 600-iter optimizer was stuck at ampL≈0 (near-perfect energy match, poor per-node alignment); 1200 iter finds that GROWING the loops (allowing energy mismatch) improves directional R². ampL≈0 was a trap, not an optimum — it zeroed the motion penalty but didn't align loop shape/phase.
63. `[optimization@600it]` **Fibre perturbations (wl, amp, phase) are INERT at 600 iter — the convergence basin is BROAD (Phase 2 b11, extends Est.#50).** wl32 (−2.199, Δ=−0.041), amp06 (−2.215, Δ=−0.057), phase0 (−2.108, Δ=+0.050): no perturbation significantly differs from the parent (−2.158). Combined with b7 at 300 iter, the b4.s1-derived fibre values are ROBUST to init perturbation across optimization depths. dur_hi=35 (−2.159, dur→34.9 bound) re-confirms dur=30 optimal (Est.#49).

64. `[optimization@2400it]` **The depth monotone CONTINUES to 2400 iter — FIRST R² crossing −1.0, but the gain is DECELERATING (Phase 2 b12, Q38 ANSWERED-PARTLY).** The depth sweep with frozen fibre + gain+dur only: 600→−2.158, 1200→−1.411, 1800→−1.113, 2400→−0.999. Δ per step: 0.747 (600→1200), 0.298 (1200→1800), 0.114 (1800→2400). Δ per doubling: 600→1200=0.747, 1200→2400=0.412. The curve is decelerating but NOT converged per the user's criterion (Δ per doubling < 0.05). Need 3600 and 4800 to bracket convergence. The model's 2-scalar capacity (gain + dur) is STILL being exploited by the optimizer.
65. `[optimization@2400it]` **Fibre co-learning gap WIDENS with depth — frozen fibre wins MORE decisively at 2400it (Phase 2 b12, STRENGTHENS Est.#61).** b12.s0 frozen (−0.999) vs b12.s4 co-learn (−1.063), Δ=0.064. At 1200it the gap was 0.026 (b11). The gradient competition from the extra fibre DOF is NOT a transient — it gets WORSE with depth, not better. Fibre should be FROZEN at b4.s1 values at ALL depths.
66. `[optimization@1200it]` **Gain INIT is a previously unsuspected lever — gain0=0.7 beats gain0=0.854 at 1200it by Δ=0.201 (Phase 2 b12.s5, NEW).** The b4.s1-derived gain0=0.854 was assumed near-optimal (Est.#56 showed resetting to 1.0 costs Δ=1.84). But going LOWER to 0.7 at 1200it gives R²=−1.210 vs −1.411 — a significant improvement, about 2/3 of the 1200→1800 depth gain. The optimizer benefits from starting at a SMALLER contraction; Est.#56's "warm-start critical" is refined to "warm-start DIRECTION matters — lower > higher > reset." Must test at 2400+ iter depth.
67. `[mechanism]` **Amplitude >10 is CONFIRMED harmful at deep optimization in the parametric inverse (Phase 2 b12.s2/s3, STRENGTHENS Est.#37/#52).** At 1200it: amp10 (−1.411) ≫ amp12 (−1.746, Δ=0.335) ≫ amp15 (−2.380, Δ=0.969). The amplitude penalty is MONOTONE and GROWS with amplitude: total impulse amp×dur ∝ overshoot energy. Higher amplitude pushes the optimizer into the energy-match trap (ampL=0.022 at amp12, 0.004 at amp15 — the same trap seen at 600it). The amp10 optimum is DEPTH-ROBUST.

 SIREN fibre (siren_fibre=1, ω=30) produces R²=−7.591 (3.2× worse than ablation) with ampL=0.916. The dashboard shows a NOISY, mottled dθ map (orange/blue blobs) — the SIREN learns high-frequency angular deviations that amplify contraction into overshoot, the SAME pathology as UNet fibre (Falsified#9, b8: noisy dθ → massive overshoot). Both-SIREN (s4, R²=−11.102, ampL=1.658) is WORST. The failure is NOT specific to the image/UNet representation — it is a fundamental property of the loss landscape: per-pixel direction freedom at the scale of the inverse L2 loss leads to noisy overshoot regardless of whether the field comes from a microscope image (UNet) or free coordinates (SIREN).

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
1. `[mechanism]` **"Zero-motion collapse is the active failure mode, and w_amp is needed to defend the fit" — FALSIFIED
   in the amp10–25 regime (b01.s0/s1/s2).** No slot collapsed: with w_amp=0 (s1) the sim still kept
   substantial motion (ampL=0.075, dur→47) and R² was only marginally worse than w_amp=1.0 (−1.64 vs
   −1.52). GD is failing by directional MISALIGNMENT, not by sliding into the tiny-dot basin. w_amp is
   a weak knob here; keep it small (0.3) but it is not the lever. (Re-test at very low amplitude, where
   collapse may re-emerge.)
2. `[mechanism]` **"The UNet will structure the stiffness field to fit" — FALSIFIED in the overshoot regime (b01–b06) but now PARTLY
   RE-INSTATED: stiffness IS LOAD-BEARING under high-magnitude positive rotary (Q2, b08).** In the pre-rotary/overshoot
   regime learned stiffness stayed ~uniform-low interior with a bright anchored-boundary frame, the fit carried by direction;
   interior texture rose with drag, PEAKED ~drag120, then WASHED OUT by drag180 (a non-load-bearing mid-drag transient). BUT
   b08 OVERTURNS this once the correct rotational DOF exists at sufficient magnitude: the +270°/+360° rotary winners show the
   MOST coherent stiffness of ANY batch — large CONNECTED bright-yellow domains (youngs→180–200) forming a network pattern,
   qualitatively unlike the inert purple+frame, while the rotary0 ablation in the SAME batch stays inert purple+frame. So
   stiffness has flipped from non-load-bearing to a genuine fit component when the direction (now elliptical, correct-sense)
   is right. Q2 is RE-OPENED as ANSWERED-YES under rotary; b09 tests whether the spatial rotary field amplifies this further.
3. `[mechanism]` **"A travelling-wave phase-delay τ(x,y) bends the red loops onto green (under the force mechanism)"
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
4. `[mechanism]` **"Active stress M1 gives the coordinated motion the body force can't / is the breakthrough" —
   FALSIFIED FOR THE INVERSE-FORCE-TRACK (b03 clean; OLD ONLY — SUPERSEDED by Phase 1 atlas which WORKS with active-stress).** ~~The batch-2 −0.845 stress winner was a NaN artifact (degenerate state, blank
   panels). With the NaN-guard ON and matched amp10/lr1e-3/md0, clean stress is catastrophic (−117, wild
   overshoot streaks) and force md0 (−1.045) wins by ~100×. Stress is also run-to-run chaotic.~~ **[This result is from the UNet inverse trainer comparing FORCE vs STRESS; Phase 1 atlas uses active-stress freely and achieves good morphology without issue — the criticism is specific to the inverse real-fit under the force/UNet paradigm, not to active-stress as a forward mechanism.]** M0 force is the mechanism for the inverse-force-track (Est.#10/#11).
5. `[mechanism]` **"Phase τ behaves differently under STRESS than under force" — FALSIFIED (b03.s5).** stress_md40
   (−408) ≪ stress_md0 (−117); τ self-organised to a TINY delay (used [0.25,2.3] of 40) — the SAME
   small-τ signature as under force. Phase is a non-lever in both mechanisms.
6. `[mechanism]` **"A learnable SPATIAL rotary field beats the global scalar / rediscovers the global sense / amplifies
   stiffness" — FALSIFIED on all three counts (b09, Est.#26).** (a) BEATS scalar: only at TIGHT spread and by
   Δ+0.012 (±90 −0.341 vs scalar −0.353 ≈ noise); MONOTONE-worse wider (±180 −0.388, ±360 −0.462 ≈ ablation) —
   spatial structure is NOT the lever, the field just saturates to a uniform + magnitude nudge. (b) REDISCOVERS
   the + sense from a 0 base: NO — the pure 0-base field stays balanced/zero-mean and collapses to rotary0 (s4
   −0.457 ≈ −0.464); the scalar base is a needed prior. (c) AMPLIFIES stiffness (the b09 hypothesis extending
   Falsified#2's revision): NO — the SCALAR control s0 has the MOST coherent stiffness of the batch (connected
   yellow network); the field FRAGMENTS it into scattered blobs. The load-bearing stiffness (Est.#25) is a
   +360-MAGNITUDE-regime property, NOT enhanced by the spatial DOF.

7. `[mechanism]` **"Structure is NECESSARY for loops" / "rotary force is required for loops" — FALSIFIED (the 2×2,
   Est.#28).** In the MLS-MPM port isotropic material loops as much as structured (D/C=0.7×) because the
   loops are INERTIAL, available without any structure or rotation. So rotary was a SCAFFOLD compensating
   for a force-based, inertial model — NOT a cardiomyocyte property. Honest statement: *a force-based
   inertial model needed a rotary correction to recover loops* ≠ *cardiomyocytes require rotary force*.
   The tell was Falsified#6 (the learned rotary field saturated to a near-uniform scalar = a constant
   correction, not a latent field). SUPPORTED replacement objective: loops are available dynamics, and
   anisotropic active-stress STRUCTURE tunes their MORPHOLOGY (→ Est.#28, the new parent).

8. `[mechanism]` **"Spatial stiffness (UNet from microscope) can lift R² in the parametric inverse" — DEFINITIVELY
   FALSIFIED across ALL bases (b2–b5, Est.#43).** Every attempt to add UNet-learned spatial stiffness was
   net-harmful: b2.s0 wl40 (−7.43 vs uniform ablation −6.48), b3.s2 wl40+dur (−7.27), b4.s5 wl40+gain
   (−6.06), b5.s4 wl28+combined (−10.50, ampL=1.621 catastrophic). The UNet consistently learns an extreme
   binary yellow/purple stiffness pattern that drives massive overshoot. The microscope image does NOT carry
   load-bearing spatial material information at this resolution for the parametric inverse. (Exception: b2.s3
   wl28+stiff-only helped — but that gain came from the fibre base change, not stiffness expressiveness.)

9. `[mechanism]` **"UNet fibre-angle deviation from the microscope breaks the R²≈−2.6 plateau" — FALSIFIED (b8, Q35 CLOSED).** All 5 UNet fibre slots (both dur basins, tight/wide deviation, with/without spatial stiffness, with/without parametric co-learning) produced R²=−6 to −23, DECISIVELY worse than the parametric-only control (−4.002). The UNet dθ(x,y) map is noisy at low dur and smooth-but-harmful at high dur. This CLOSES the spatial-information track entirely: NEITHER spatial stiffness (Falsified#8) NOR spatial fibre direction from the microscope image improves the parametric inverse.

10. `[mechanism]` **"A FREE, image-independent SIREN stiffness field can lift R² (the UNet falsifications were confounded by the microscope)" — FALSIFIED (b10, Est.#58).** The user hypothesised that Falsified#8/#9 confounded "a field shaped like the microscope" with "a spatial field" — a FREE SIREN f(x,y) might succeed where UNet(microscope) failed. RESULT: SIREN stiffness at ω=15/30/60 converges to a UNIFORM field with metrics IDENTICAL to the no-SIREN ablation in all cases. The stiffness direction has ZERO gradient signal in the loss landscape. The confounding hypothesis is itself FALSIFIED: the issue was never the image constraint — it is that the inverse loss does not reward spatial stiffness variation, period.

11. `[mechanism]` **"A FREE SIREN fibre-direction field provides the spatial resolution needed to break the plateau" — FALSIFIED (b10.s3/s4, Est.#59).** SIREN fibre (ω=30) produces R²=−7.591 (3.2× worse than ablation) with a noisy dθ map and massive overshoot (ampL=0.916). Combined SIREN stiff+fibre even WORSE (−11.102). The noisy-overshoot pathology is representation-INDEPENDENT: both UNet (image-shaped, b8) and SIREN (free, b10) produce it. Per-pixel direction freedom is a NET-HARMFUL DOF under L2 inverse optimization, regardless of source.

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
- **Q25.** [CLOSED b2 — ANSWERED MIXED: spatial stiffness HURTS on wl40 but HELPS on wl28.] (a) stiffness learning does NOT lower R²
  from the −5.45 baseline on the wl40 fibre base — the spatial parent s0 (−7.43) is WORSE than the fibre-only b1 winner, and even
  the UNIFORM ablation s5 (−6.48) beats it. So on wl40+high-fibre-amp, the UNet spatial stiffness pattern is a net-negative lever.
  (b) YES on wl28: s3 (wl28+stiff, −5.18) is the new Phase-2 best — finer fibre + stiffness learning DOES help, but the gain comes
  from the fibre base change not stiffness expressiveness. (c) Wider range [20,250] helps marginally; SOFT [20,80] is catastrophic.
  (d) amp=12 HURTS at dur=8 (more overshoot). The KEY insight: dur=8 is the root cause of all deeply-negative R²; the stiffness
  question is MASKED by the short-duty-cycle regime. → batch 3 co-learns dur with each lever (user guidance).
- **Q26.** [ANSWERED b3 — dur is NEAR-INERT; GAIN is the lever, not dur.] Co-learning dur alongside each group did NOT break the
  deeply-negative regime via dur — dur moved only 0.7–1.0 units (8→8.7–9.0) in ALL 6 slots. The hypothesis that dur→14 (longer pulse
  → less ringing) is FALSIFIED at [3,14] bound / 300 iterations. BUT the batch DID find a winner: gain+dur (s3, R²=−4.164, NEW Phase-2
  best). The R² gain is almost entirely from GAIN (1.0→0.817), not dur. Answers per sub-question: (a) dur_only (−5.08) marginally beats
  b1 winner (−5.45) — tiny dur shift helps a little; (b) fibre+dur on wl40 HURTS (−13.23, destabilized); on wl28 −6.74 (OK but worse
  than b2 best); (c) stiff+dur (−7.27) ≈ stiff-only b2.s0 (−7.43), tiny improvement; (d) gain+dur = WINNER (−4.16); (e) all_combine
  CATASTROPHIC (−16.83). → batch 4 builds on gain, tests fibre+gain, and probes the high-dur basin directly (dur0=14 init).
- **Q27.** [ANSWERED b4 — YES on wl28, NO on wl40; high-dur basin EXISTS.] Fibre+gain on wl28 (b4.s1, R²=−2.620) is the NEW Phase-2
  best — gain stabilizes fibre co-learning and the optimizer drives fibre_angle→0.17 (near-zero rotation), producing nearly perfect
  energy match (ampL=0.010). On wl40 (b4.s0, −7.307) fibre STILL destabilizes (amp collapse). High-dur basin CONFIRMED: dur0=14 stays
  at the upper bound and gives R²=−3.880 > prev best −4.164. The [3,14] bound is limiting — dur may want >14. Added `--dur_hi` to test.
- **Q28.** [CLOSED b5 — FALSIFIED: the two b4 wins do NOT stack.] wl28 fibre+gain (−2.620 at dur≈8.7) + dur0=14 → best −2.992
  (fibre frozen), −3.383 (fibre co-learning). Both worse than b4.s1. The improvements overlap — dur=14 disrupts the fibre+gain
  optimum. b4.s1 remains the Phase-2 best.
- **Q29.** [CLOSED b5 — angle=0 is NOT optimal; ~0.17 IS preferred.] angle=0 (−3.671) < angle=0.17 (−3.383). The b4.s1 optimum is
  a genuine small positive angle, not "heading to zero." A ~10° contraction-axis rotation is load-bearing.
- **Q30.** [ANSWERED b6 — dur turnover is between 30 and 50; optimum BRACKETED.] dur_hi=30 → dur hits bound at 30.0
  (s0 −3.087, s2 −2.814). dur_hi=50 → dur hits 49.9, R²=−3.223 (WORSE than dur=30, openness collapses 0.179). So the
  optimum is between 30 and 50 — the first TURNOVER evidence. Need dur_hi=40 to pin it down.
- **Q31.** [ANSWERED b6 — wl24 does NOT improve the low-dur regime.] s4 (wl24, dur=9.0, −3.746) is WORSE than b4.s1
  (wl28, dur≈8.7, −2.620). Finer wavelength does not help; wl28 remains optimal.
- **Q32.** [PARTLY ANSWERED b6 — fibre co-learning at dur=30 CLOSES the gap but doesn't cross.] s2 fibre co-learn
  at dur=30 gives −2.814, the closest to b4.s1 (−2.620) yet, but Δ=0.19 short. Fibre FROZEN at dur=30 (−3.087) is worse
  → co-learning IS needed at high dur (overturns Est.#41). The question shifts: can dur_hi=40 (finding the true optimum)
  or fibre-param perturbations (angle, amp inits) close the remaining 0.19?
- **Q33.** [CLOSED b7 — dur optimum IS 30; dur_hi=40 → dur saturated at 39.9, R²=−2.871, WORSE than dur=30 (−2.814).]
  The controlled ceiling sweep pins the optimum: dur=30 > dur=40 > dur=50. The turnover is between 30 and 40. Est.#49.
- **Q34.** [CLOSED b7 — fibre perturbations do NOT improve.] angle=0.3 (−2.842), fibre_amp=0.6 (−2.956), wl=24 (−3.716)
  all WORSE than parent (−2.814). The b4.s1-derived inits are near-optimal at dur=30. Est.#50.
- **Q35.** [CLOSED b8 — ANSWERED-NO: UNet fibre deviation does NOT break the plateau; it is net-harmful.] ALL 5 UNet fibre
  slots are 2–19 R² units WORSE than the parametric-only control (s4, −4.002). The microscope image does not carry
  fibre-direction information usable by the UNet/L2 inverse. Both spatial UNet channels (stiffness, fibre) are now closed.
  Est.#54, Falsified#9.
- **Q36.** [ANSWERED b9/b10 — YES, optimization depth is the remaining lever; spatial fields CLOSED.] (a) 600 iterations BROKE the
  plateau: b9.s2 (600it, dur=30, −2.158) beats b4.s1 (−2.620) by Δ=0.462 (Est.#55). (b) intermediate dur=20/25 are MONOTONE
  below dur=30 at 300it (Est.#49). (c) uniform stiff fix confirmed; b10 re-optimized gain+dur from b4.s1 inits at 600it/dur=30
  and got −2.354, which IS the gain+dur-only level (fibre frozen). Fibre co-learning contributes Δ=0.196 (−2.354→−2.158). SIREN
  free-fields are DEAD (Falsified#10/#11, Est.#58/#59). The remaining lever is optimization depth + fibre-param exploration at ≥600 iters.
- **Q37.** [ANSWERED-YES b11 — 1200 iter gives Δ=0.72–0.75 over 600 iter, MASSIVE, NOT converged] b11.s5 (1200it, −1.411) and b11.s0
  (1200it, −1.437) both dramatically beat b9.s2 (600it, −2.158). The jump is COMPARABLE to 300→600 (Δ=0.66). The frozen-fibre ablation
  (s5, gain+dur only) BEATS fibre co-learning (s0) by Δ=0.026 — fibre is near-optimal and should be FROZEN at deep optimization. If 1200 helps, continue pushing; if flat,
  the parametric model is at its expressiveness limit. Also: do fibre-param perturbations (wl, amp, phase inits) help at 600 iter
  when they failed at 300 iter (b7)? ANSWERED: 1200 iter = MASSIVE gain; perturbations inert; fibre should be FROZEN.
- **Q38.** [ANSWERED-PARTLY b12 — YES, the depth monotone continues to 2400 but is DECELERATING; N\* NOT YET PINNED.] 1200→1800 Δ=0.298, 1800→2400 Δ=0.114; Δ per doubling 1200→2400=0.412, still >>0.05. The curve is decelerating but N\* requires 3600/4800 runs. Amplitude >10 CONFIRMED harmful at depth (Est.#67). Gain init=0.7 found as a new lever (Est.#66). b13 continues depth push + gain-init sweep.
- **Q39.** [NEW b13 — does gain0=0.7 beat gain0=0.854 at 2400+ iter depth?] At 1200it, gain0=0.7 (−1.210) beats gain0=0.854 (−1.411) by Δ=0.201. If the gain-init advantage TRANSFERS to deeper optimization (2400, 3600it), the new parent should use gain0=0.7. Also: does going even LOWER (gain0=0.5) help further, or is 0.7 the sweet spot? This tests whether gain init is a monotone lever or has a turnover.

---

## Previous Batch Summaries
**RULE: keep the last 4, oldest→newest, before `## Current Batch`.**

**Phase 2 Batch 9 (2026-06-24, PARAMETRIC INVERSE optimization depth + dur gap-fill; archive prefix p2_b09_*):** Tested 600 iterations,
intermediate dur=20/25, clean b4.s1 reproduction, gain reset. WINNER: s2 iter600_hidur (R²=−2.158, NEW OVERALL BEST, beats b4.s1 by Δ=0.462).
KEY FINDINGS: (1) 600 iter BREAKS the high-dur plateau (Est.#55, OVERTURNS Est.#53 for high-dur). (2) Low-dur basin NOT reproducible from
converged inits (Est.#57). (3) Intermediate dur=20/25 monotone below dur=30 at 300it. (4) Gain warm-start critical — Δ=1.84 penalty from reset (Est.#56).

**Phase 2 Batch 10 (2026-06-24, SIREN FREE-FIELD TEST; archive prefix p2_b10_*):** Tested SIREN free fields (image-independent coordinate
networks) for stiffness and fibre direction — the genuinely new spatial test that Falsified#8/#9 did NOT cover. ALL 6 SLOTS DONE. RESULT:
SIREN stiffness COMPLETELY INERT at all ω (15/30/60) — converges to UNIFORM, metrics IDENTICAL to ablation (R²=−2.354 in s0/s1/s2/s5).
SIREN fibre HARMFUL (s3 −7.591, ampL=0.916). Combined WORST (s4 −11.102, ampL=1.658). (Falsified#10/#11, Est.#58/#59, Q36 answered.)
b9.s2 (−2.158) remains overall best.

**Phase 2 Batch 11 (2026-06-25, OPTIMIZATION DEPTH 1200it + FIBRE PERTURBATIONS; archive prefix p2_b11_*):** Pushed optimization depth
to 1200 iterations and tested fibre-param perturbations at 600it. WINNER: s5 iter1200_frozen_abl (R²=−1.411, NEW OVERALL PHASE-2 BEST,
beats b9.s2 −2.158 by Δ=+0.747). KEY FINDINGS: (1) 1200 iter is a MASSIVE leap — Δ≈0.72–0.75 over 600 iter, comparable to 300→600 jump
(Q37 ANSWERED-YES, Est.#60). The parametric model is NOT converged. (2) Fibre FROZEN (gain+dur only) BEATS fibre co-learn at 1200it
(−1.411 vs −1.437, Δ=0.026) — REVISES Est.#47: co-learning is slightly harmful at deep optimization; fibre should be frozen at b4.s1
values (Est.#61). (3) Fibre perturbations at 600it are all near-parent — convergence basin is broad (Est.#63). (4) dur_hi=35 ≈ parent —
dur=30 confirmed (Est.#49). (5) Deeper optimization GROWS red loops (ampL 0.083–0.124 > 600-iter 0.003–0.019) — the 600-iter energy-match
trap was NOT the optimum (Est.#62). (Est.#60–63, Q37 answered, Q38 opened.)

**Phase 2 Batch 12 (2026-06-25, OPTIMIZATION DEPTH 1800/2400it + AMPLITUDE/GAIN-INIT PERTURBATIONS; archive prefix p2_b12_*):** Continued
depth push to 1800/2400 iter with fibre FROZEN, tested amp12/amp15 at 1200it, and gain init=0.7. WINNER: s0 iter2400 (R²=−0.999,
NEW OVERALL PHASE-2 BEST, FIRST TIME R² crosses −1.0). KEY FINDINGS: (1) Depth monotone CONTINUES but DECELERATES: 1200→−1.411,
1800→−1.113, 2400→−0.999; Δ per doubling 1200→2400=0.412, NOT converged (Q38 partly answered, Est.#64). (2) Fibre co-learn gap WIDENS
with depth: Δ=0.064 at 2400it vs 0.026 at 1200it (Est.#65). (3) gain0=0.7 BEATS gain0=0.854 at 1200it by Δ=0.201 — NEW lever (Est.#66,
Q39 opened). (4) Amplitude >10 re-confirmed harmful at depth: amp12 −1.746, amp15 −2.380 (Est.#67). (Est.#64–67, Q38 partly answered,
Q39 opened.)

---

## Current Batch

### Batch info
**PHASE 2 BATCH 13 [learn=gain,dur — PIN CONVERGED DEPTH N\* + GAIN-INIT SWEEP] — PARAMETRIC INVERSE.**
Parent: b12.s0 (R²=−0.999, 2400 iter, dur_hi=30, learn=gain,dur, fibre FROZEN at b4.s1 values, stiff=[100,100], gain0=0.854).
Batch 12 showed the depth monotone continues (Δ per doubling 1200→2400=0.412, NOT converged) and that gain init=0.7 is a new lever
(Δ=0.201 at 1200it). The model is 2 learnable scalars (gain + dur). Runner = `cardio_mpm_train2.py`.

### Current hypothesis
"The optimizer has NOT converged at 2400 iter (Δ per doubling=0.412 >> 0.05), so deeper runs (3600, 4800) should continue the
monotone — eventually bracketing N\* where Δ per doubling < 0.05. Meanwhile, gain0=0.7 beat 0.854 by Δ=0.201 at 1200it — if
this advantage transfers to 2400+ depth, it is a free +0.2 lift. A gain-init sweep (0.5, 0.7) at 2400it + deeper runs at 0.7
should find the best combination of depth × gain-init. Fibre co-learning is CLOSED (gap widening with depth)."

### Slots this batch
PIN DEPTH N\* + GAIN-INIT SWEEP (base = b12.s0, stiff=[100,100], dur_hi=30, amp=10, drag=30, fibre FROZEN, material_aniso_cardio):
- s0 iter3600 — gain,dur (fibre FROZEN), 3600 iters, gain0=0.854 (continue depth push — Δ from 2400→3600?)
- s1 iter4800 — gain,dur (fibre FROZEN), 4800 iters, gain0=0.854 (bracket N\*: if Δ(2400→4800)<0.05 per doubling, N\*≈2400)
- s2 iter2400_gain07 — gain,dur (fibre FROZEN), 2400 iters, gain0=0.7 (does gain-init advantage transfer to depth? Q39)
- s3 iter3600_gain07 — gain,dur (fibre FROZEN), 3600 iters, gain0=0.7 (combine best init with deeper opt)
- s4 iter2400_gain05 — gain,dur (fibre FROZEN), 2400 iters, gain0=0.5 (sweep gain init lower — monotone or turnover?)
- s5 iter2400_gain10_abl — gain,dur (fibre FROZEN), 2400 iters, gain0=1.0 (ABLATION: is warm-start STILL critical at 2400it?)

### Emerging observations
**CRITICAL: this section must ALWAYS be at the END of the file.**

_(Phase 2 b12, 2026-06-25) OPTIMIZATION DEPTH 2400it — FIRST R² crossing −1.0 (b12.s0, −0.999). The depth monotone continues
but is DECELERATING: Δ per doubling drops from 0.747 (600→1200) to 0.412 (1200→2400). NOT converged per Δ<0.05 criterion —
need 3600/4800 to pin N\*. SURPRISE: gain init=0.7 (b12.s5, −1.210 at 1200it) BEATS default 0.854 (−1.411) by Δ=0.201 — a
previously unsuspected lever. This changes the parent going forward: if the advantage holds at depth, gain0=0.7 is the new
default. Fibre co-learn gap WIDENS with depth (0.026→0.064) — co-learning is closed. Amplitude >10 re-confirmed harmful at
1200it depth (amp12 −1.746, amp15 −2.380). The two live levers are DEPTH (continue pushing) and GAIN INIT (sweep downward)._

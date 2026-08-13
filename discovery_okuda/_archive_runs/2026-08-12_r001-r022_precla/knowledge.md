# Knowledge — Okuda campaign (fresh, from r001)

## Facts established this round
- **The round-1 recipe family makes lobed spheres, not fingers.** 13 Route-B runs, `protr_peak`
  1.152–1.342, all read by the eye as bulges/undulations; `n_tubes` (1–3) and
  `protrusion_aspect_max_peak` (up to 5.909, `r001_15`) both fire but with
  `protrusion_aspect_max_final` 0.0 — transient flicker, no sustained protrusion. `n_tips` 0.
- **Campaign bests so far (all on lobed spheres):** `protr_peak` 1.342 (`r001_15`), `grip_peak`
  0.101 (`r001_06`), `n_tubes_peak` 3 (`r001_11`/`r001_15`). No tube exists yet.
- **Spot count is far too high for Okuda's scale.** `n_spots_peak` 144–338 this round vs ~10 in
  Fig. 5. `spot_spacing_cells` measurable (2.4–10.3), unlike the previous campaign, because there
  are many spots — the opposite of the target.

## SURPRISES
- `r001_10` `gyr_prolate_peak` **1.71** vs control 1.01 (+69%), unpredicted — the round's most
  elongated body, yet the eye calls it a faceted polyhedral blob, not a tube. Elongation of the
  gyration tensor ≠ a finger.
- **Control `r001_00` chemistry rail:** `act_alive_frac`/`act_max_final`/`n_spots` all exactly
  0.0 while the eye reports a stable 4–6-domain orange-on-blue pattern. Either a field is present
  and unmeasured or the colouring is a static overlay — a measurement gap on the control.
- `r001_09` `P12` rail: chemistry NaN from frame 1635, metrics empty. A crash, not a zero.
- `n_apop` fired 35–203 (`r001_12` 203, `r001_02` 122, `r001_13` 110) in death-twin runs, but
  `invagination_peak` reached only 0.41 (`r001_11`) — apoptosis is active yet no inward fold.

## STANDING LAWS
- **L1** `cell_mechanics.K_bend` raises BOTH `grip` and `protr` on `b_gs_shaping_soft_lo`.
  evidence: 0.03 → grip 0.083 / protr 1.233; 0.05 → grip 0.096 / protr 1.274. status: HOLDS (2 pts).
- **L2** `cell_die.max_mark_frac` at 0.001–0.002 raises `protr` with no premise broken.
  evidence: 0.001 → protr 1.274; 0.002 → protr 1.342 (round best), cells flat ~17,500.
  status: HOLDS (2 pts, break untested above 0.002).
- **L3** `protrusion_aspect_max_peak` and `n_tubes` do NOT discriminate a finger from a lobe.
  evidence: all runs with aspect_peak > 1 (`r001_03/04/05/15`) read as lobes by the eye,
  aspect_final 0.0, `n_tips` 0. status: HOLDS.

## Route A closures
- **`cell_grow.rho` on `b_gs_plain_soft_lo`:** one value only, 0.05 → 5952 cells, protr 1.201,
  grip 0.079, premises intact. On `b_bru_question` rho 0.05 returned empty (crashed). NOT a curve —
  1 value each; sweep incomplete.
- **`cell_die.mode` on `b_gs_plain_soft_lo_death`:** older → 6848 cells/protr 1.153;
  smaller → 6813/protr 1.181. Mode is nearly inert on shape at this cap. 2 values.
- **`cell_die.max_mark_frac` on `b_gs_shaping_sharp_hi_death`:** 0.001 → 17,446 cells/protr
  1.274; 0.002 → 17,505/protr 1.342/grip 0.090. Cells flat; protr rises. Use 0.002; break above it
  untested. 2 values.
- **`cell_mechanics.K_bend` on `b_gs_shaping_soft_lo`:** 0.03 → protr 1.233/grip 0.083;
  0.05 → protr 1.274/grip 0.096. Both rise (see L1). Use 0.05; break above untested. 2 values.

## Facts established r002
- **Cell count is inverse to grip; it is the coarsening lever.** Fewer cells → few large spots →
  budding + strong coupling; more cells → many spots → faceted sphere. grip_peak vs cells_final:
  `r002_01` 3463→0.136, `r002_04` 5468→0.141, `r002_03` 22245→0.068, `r002_07` 41865→0.087.
- **Campaign bests, all on two-lobe BUDS at low cell count:** `corr_act_rad_peak` 0.933 (`r002_04`),
  0.928 (`r002_01`) vs control 0.763; `grip_peak` 0.141 (`r002_04`), 0.136 (`r002_01`) vs prior best
  0.101; `protr_peak` 1.405 (`r002_04`). Still buds not fingers (aspect_final 0.0, L3). No tube.
- **`r002_00` control is a lobed invaginating body, not a sphere** — protr_peak 1.245, invagination
  0.396, reduced_volume 0.747, 17964 cells; morphology='sphere' and n_tubes=1 both wrong (eye).

## SURPRISES (r002)
- **`n_spots_peak` rails at exactly 100** in 7 runs (`r002_01,02,04,06,09,10,11`) — a measurement
  cap, not a count; every n_spots prediction this round scored against it and refuted. `r002_03`
  reaches 427 and `r002_07` 786, so the 100 is a ceiling those two escape, not the tissue.
- `r002_02` `act_mean_floor` 5.6e-45 and `act_cv_peak` 7.80 with P4 broken — chemistry flashed once
  then died to numerical zero; `mech_p_ratio` 0.
- `n_apop` railed 2438 (`r002_13`) and 1103 (`r002_12`) — 10-20x the r001 death twins — yet
  `invagination_peak` only 0.37; apoptosis does not fold the sheet inward at this cap.

## STANDING LAWS (r002 update)
- **L2 REFUTED as monotone.** `cell_die.max_mark_frac` raises `protr` only up to ~0.01, then
  FALLS: 0.001→1.274, 0.002→1.342 (r001); 0.01→1.321, 0.02→1.246 (r002, b_gs_shaping_sharp_hi_death).
  Peaks near 0.01. status: REFUTED (non-monotone; earlier up-claim held only 0.001–0.002).
- **L3 HOLDS.** `r002_01` aspect_peak 3.973, `r002_04` 2.672 — both buds by the eye, aspect_final 0.
- **L4** grip_peak and protr_peak rise as cell count FALLS (few coarse spots vs many). evidence: 4
  runs, cells 3463/5468/22245/41865 → grip 0.136/0.141/0.068/0.087. status: HOLDS (no run inverts).

## Route A closures (r002)
- **`cell_grow.rho` on `b_gs_plain_soft_lo`:** 0.1→7261 cells/protr 1.176/grip 0.072;
  0.25→12205/1.185/0.081. rho drives cell count but protr is FLAT (~1.18) — growth buys lobes, not
  elongation. Use for coarsening cell count only. 2 values.
- **`cell_die.mode` on `b_gs_plain_soft_lo_death`:** crowded→6838 cells/protr 1.169;
  dimmer→6641/1.168. Mode inert on shape (reconfirms r001). CLOSED, 2 values.
- **`cell_die.max_mark_frac` on `b_gs_shaping_sharp_hi_death`:** 0.01→15622 cells/protr 1.321/
  grip 0.104; 0.02→12428/1.246/0.085. protr and grip both FALL — the knob overshot here (see L2).
  Use ≤0.01. 2 values.
- **`cell_mechanics.K_bend` on `b_gs_shaping_soft_lo`:** 0.12→7810 cells/protr 1.320/grip 0.094.
  1 value this round; extends L1 upward (0.05→1.274 < 0.12→1.320). Break still untested. 1 value.

## Facts established r003
- **The low-cell two-lobe BUD is a reproducible attractor, capped at grip ~0.135.** Three runs at
  3518/5302/8773 cells (`r003_06/07/04`) → grip 0.136/0.137/0.134, corr_act_rad_peak 0.919/0.935/
  0.886, protr_peak 1.344/1.400/1.293, invagination 0.580/0.585/0.394. All fat buds by the eye,
  aspect_final 0 (L3). Reconfirms L4 (low cells → strong coupling).
- **Campaign-best invagination = 0.585 (`r003_07`)** vs control 0.396 — the deepest inward fold,
  and it sits on a bud, not a fold. protr_peak 1.400 ties r002_04's 1.405 as campaign best.
- **The buds are GROWN not forced:** `mech_p_ratio` 2.234/2.486 (`r003_06/07`) with no extrude
  operator in the space — the >2 stand-in flags grown budding, not forcing (user §3).

## SURPRISES (r003)
- **`r003_01` `gyr_prolate_peak` 4.802** vs control 1.337 (3.6×) — the campaign's most elongated
  body, a smooth prolate egg (eye), yet grip COLLAPSES (grip_final -0.0017, corr_act_rad_final
  -0.006). Whole-body stretch, unpredicted (round tested aspect); elongation without a spot.
- **`r003_07` `protrusion_aspect_max_peak` 3.169** (control 0.0) — highest aspect the campaign has
  recorded, but aspect_final 0 and eye sees a bud (L3 HOLDS: aspect_peak still ≠ finger).
- `n_spots_peak` rails at exactly 100 on `r003_01`–`07`; `r003_10` escapes to 463. Reconfirmed cap.
- Three P4 deaths (`r003_02/03/05`): act_mean_floor 5.6e-45, chemistry flashed once and died,
  mech_p_ratio 0 — a settings/death-twin failure, not biology.

## STANDING LAWS (r003 update)
- **L1 REFUTED as monotone.** `cell_mechanics.K_bend` on `b_gs_shaping_soft_lo` peaks near 0.12:
  0.03→1.233, 0.05→1.274, 0.12→1.320, 0.2→1.244 (`r003_08`). status: REFUTED (non-monotone).
- **L3 HOLDS.** `r003_07` aspect_peak 3.169, `r003_06` 2.036 — both buds, aspect_final 0.
- **L4 HOLDS.** 6 more runs, no inversion: cells 3518/5302/8773 → grip 0.136/0.137/0.134;
  11936/30134 → grip 0.080/0.087.

## Route A closures (r003)
- **`cell_grow.rho` on `b_gs_plain_soft_lo`:** 0.5→30134 cells/protr 1.189/grip 0.087 (`r003_10`).
  With r002 (0.1→7261, 0.25→12205): rho drives cells monotonically but protr is FLAT ~1.18 across
  the whole range — growth buys cells/lobes, never elongation. CLOSED, 3 values. Stop sweeping rho.
- **`cell_mechanics.K_bend` on `b_gs_shaping_soft_lo`:** 0.2→3544 cells/protr 1.244/grip 0.080
  (`r003_08`). Protr FELL from the 0.12 peak (see L1). Use ~0.12; CLOSED above it. 4 values total.
- **`cell_mechanics.Lambda` on `b_gs_shaping_soft_lo`:** 0.3→11936 cells/protr 1.186/grip 0.080
  (`r003_15`). 1 value; sweep incomplete.
- **`cell_die.max_mark_frac` on `b_gs_plain_soft_lo_death`:** 0.001→6876 cells/protr 1.174/
  grip 0.081 (`r003_09`), premises intact. 1 value; extends nothing new.

## Facts established r004
- **First sustained aspect signal, but still a bud.** `r004_06` is the only run with
  `protrusion_aspect_max_final` non-zero (**2.207**) and `n_tips_final` 1, at 4121 cells / one spot
  (`red_frac` 0.116) — `protr_peak` 1.454, `corr_act_rad_peak` 0.954, `grip` 0.135, `mech_p_ratio`
  2.965 (grown). The eye calls it a fat pear (aspect ~1.09), no finger. Closest to a tube yet; not one.
- **Grip has TWO routes to ~0.14.** Low-N coarsening (`r004_06` 4121 cells → grip 0.135) AND high-N
  invagination (`r004_07` 14809 cells → grip 0.14649 via `r_cv` 0.217, `reduced_volume` 0.553). L4's
  low-cell claim holds but is not exclusive.

## SURPRISES (r004)
- `r004_07` `reduced_volume_final` **0.553** vs control 0.747 — campaign-deepest volume collapse
  (invagination 0.541), at high cell count; unpredicted (round tested protr).
- `r004_10` `protrusion_aspect_max_peak` **7.742** vs control 0.0 — campaign record aspect_peak, yet
  `aspect_final` 0, eye sees pointed-star lobes. L3 holds: aspect_peak ≠ finger.
- `n_spots_peak` rails at exactly 100 (`r004_02/05/06/08/14/15`); reconfirmed cap.
- `r004_02` P4 death: `act_mean_floor` 5.6e-45, `mech_p_ratio` 0 — chemistry flashed once and died.

## STANDING LAWS (r004 update)
- **L3 HOLDS.** `r004_10` aspect_peak 7.742, `r004_06` 3.411 — bud/star by the eye. But `r004_06`
  `aspect_final` 2.207 is the first non-zero final; watch whether L3 weakens at low-N one-spot states.
- **L4 HOLDS but not exclusive.** Low cells → high grip still holds (`r004_06` 4121→0.135); however
  `r004_07` reaches grip 0.147 at 14809 cells by invagination, a second mechanism.

## Route A closures (r004)
- **`cell_mechanics.Lambda` on `b_gs_shaping_soft_lo`:** 0.45→11443 cells/protr 1.210/grip 0.069;
  0.6→7106/1.330/grip 0.103; 0.8→2794/1.197/grip 0.041 (pattern collapses — `act_cv` 17.3, `corr`
  −0.012). Lambda drives cells DOWN monotone; protr and grip PEAK at 0.6. Use 0.6; chemistry dies at
  0.8. With r003 (0.3→11936): CLOSED, 4 values.
- **`cell_grow.vth_frac` on `b_gs_plain_soft_lo`:** 4→10803 cells/protr 1.344/grip 0.125; 6→10822/
  1.348/0.124. INERT — identical across the range. CLOSED, 2 values. Stop sweeping vth_frac.
- **`cell_grow.rho` on `b_bru_question`:** both 0.05 and 0.1 returned empty (crashed). Base rules
  itself out — do not reuse `b_bru_question`.
- **`cell_die.max_mark_frac` on `b_gs_plain_soft_lo_death`:** 0.002→6962 cells/protr 1.171/grip
  0.080. 1 value, consistent with r003; extends nothing.

## Facts established r005
- **New campaign grip best = `r005_04` grip_peak 0.17961**, on a 5–6-armed undulating star at 9260
  cells via deep folding: `reduced_volume_final` 0.5339 (ties r004_07's campaign-deepest 0.553),
  `invagination` 0.5202, `corr_act_rad_peak` 0.876. High-N invagination, not low-N coarsening — the
  r004_07 route, now the campaign's strongest coupling. Still a `complex`/undulation; aspect_final 0,
  no finger (L3).
- **The r005 control is a stronger parent than r001–r004:** `r005_00` protr_peak 1.277 / grip_peak
  0.11957 / 10749 cells (vs the old 4-cusp control protr 1.245 / grip 0.090). `r005_06` bit-identical
  → seed floor ~0 on this recipe. Re-baseline grip claims here.
- **Elongation ⊥ grip, reconfirmed.** `r005_01` gyr_prolate_peak 4.619 (fat directional bud) with
  corr_act_rad_peak 0.946 but grip_peak 0.0158 — whole-body stretch, amplitude normalised away.
  Matches r003_01 (gyr_prolate 4.802, grip collapsed).

## SURPRISES (r005)
- **`r005_01` gyr_prolate_peak 4.619** vs control 1.10 (4.2×), unpredicted (round tested aspect) — the
  round's most elongated body, a prolate egg, yet grip collapses to 0.0158.
- **`r005_03` n_spots_peak 415** vs control 154 (2.7×) and n_spots_final 406 — the round's finest
  pattern; unpredicted, and the opposite of Okuda's ~10-spot target.
- `r005_09` P12 rail: chemistry NaN from frame 1635 (5th such crash: r001_09/r002_08/r003_11/
  r004_11); metrics empty, a crash not a zero. `r005_10` empty.
- `n_apop` fired 932 (`r005_14`) / 332 (`r005_13`) / 115 (`r005_01`) yet invagination_peak ≤ 0.30 on
  the death runs — apoptosis active, no inward fold (reconfirms r001–r003).

## STANDING LAWS (r005 update)
- **L3 HOLDS.** `r005_08` aspect_peak 7.742, `r005_03` 5.578, `r005_04` 0.498 — all lobes/stars by the
  eye, aspect_final 0, n_tips 0. No finger.
- **L4 UNTESTED cleanly (cross-recipe this round).** Within-recipe low-N→high-grip not isolated;
  note the apparent inversion — `r005_04` grip 0.180 at 9260 cells (HIGH) via folding, `r005_01` grip
  0.0158 at 6580 cells (LOW). High-N folding now reaches higher grip than low-N coarsening.

## Route A closures (r005)
- **`cell_grow.vth_frac` on `b_gs_plain_soft_lo`:** 10→10822 cells/protr 1.348/grip 0.124. With r004
  (4→1.344, 6→1.348): INERT across 4/6/10. CLOSED, 3 values. Stop sweeping vth_frac.
- **`cell_mechanics.K_bend` on `b_gs_shaping_soft_lo`:** 0→10516 cells/protr 1.234/grip 0.070;
  0.005→10620/1.208/0.077. Both below the 0.12 peak (L1); protr flat, grip near-flat at the low end.
  Use ~0.12. CLOSED region 0–0.005.
- **`cell_die.max_mark_frac` on `b_gs_plain_soft_lo_death`:** 0.01→6595 cells/protr 1.159/grip
  0.075/n_apop 332; 0.02→5855/1.171/0.073/n_apop 932. Cells fall, protr near-flat, no invagination
  (≤0.21). Reconfirms L2 non-monotone plateau near 0.01. 2 values.
- **`cell_divide.factor` on `b_gs_plain_soft_lo`:** 2.4→3493 cells/protr 1.106/grip 0.048/v_cell_mean
  0.458. 1 value; low division → few cells but lowest protr/grip of the round (fat cells, weak lobing).
  Sweep incomplete.
- **`cell_grow.rho` on `b_bru_question`:** 0.05 and 0.1 both empty (crashed) again — base RULED OUT
  (reconfirms r001/r004). Do not reuse `b_bru_question`.

## Facts established r006
- **New campaign grip best = `r006_06` grip_peak 0.18254** (prev best r005_04 0.17961), on a 5–6-lobe
  star at 8512 cells, `n_tubes_peak` 6, `corr_act_rad_peak` 0.872, `reduced_volume` 0.526,
  `invagination` 0.5136 — high-N folding route again (r004_07/r005_04), still a lobed star, aspect_final
  0, no finger (L3).
- **Two-lobe BUD attractor reconfirmed at low N.** `r006_04` (5468 cells) protr_peak 1.405 /
  corr_act_rad_peak 0.933 / aspect_max_peak 2.672 / grip 0.141; `r006_11` (2661 cells) protr_peak 1.432
  / corr_act_rad_peak 0.903 / aspect_max_peak 3.09 / grip 0.151 — red-capped dumbbells, both buds by the
  eye, aspect_final 0.
- **Apoptosis deepens the fold.** `r006_03` (n_apop 117, the round's only death run) reached
  invagination_peak 0.5503 / reduced_volume 0.545 at 14647 cells, grip 0.14681 — high-N invagination.

## SURPRISES (r006)
- **`r006_11` act_cv_peak 11.67** vs control 2.20 (5.3×) — the round's spikiest chemistry (4 spots
  final, spot_spacing 14.19 cells, closest to Okuda's ~10-spot scale), unpredicted (round tested aspect).
- **`r006_12` and `r006_14` bit-identical** (cell_divide.factor 5 vs 8 → both 2000 cells / protr 1.063) —
  a rail: division saturates above factor 3, the two Route-A rungs measure the same tissue.
- `r006_09` P12 NaN crash from frame 1635 (6th: r001_09/r002_08/r003_11/r004_11/r005_09); r006_13/15 empty.
- `r006_02` P4 death: act_max_final 0, mech_p_ratio 0 — chemistry flashed once and died.

## STANDING LAWS (r006 update)
- **L3 HOLDS.** `r006_11` aspect_peak 3.09, `r006_04` 2.672 — buds by the eye, aspect_final 0, n_tips 0.
- **L4 mixed, reconfirmed non-exclusive.** grip best `r006_06` 0.183 at 8512 cells (high-N folding) >
  low-N `r006_11` 0.151 at 2661 cells — folding route out-grips coarsening route again (r005_04).

## Route A closures (r006)
- **`cell_divide.factor` on `b_gs_plain_soft_lo`:** 3→2156 cells/protr 1.064/grip 0.032; 5→2000/1.063/0.032;
  8→2000/1.063/0.032. INERT above 3 (5 and 8 bit-identical). CLOSED, 3 values. Stop sweeping factor.
- **`cell_mechanics.K_bend` on `b_gs_shaping_soft_lo`:** 0.08→9129 cells/protr 1.298/grip 0.098;
  0.3→2661/protr 1.432/grip 0.151/act_cv 11.674. Higher K_bend → fewer cells, higher protr+grip, but
  chemistry destabilises (act_cv 11.7). Use ~0.3; break in act_cv above it. 2 values, sweep open upward.

## Facts established r007
- **Sustained aspect returns at LOW cell count, still a bud.** Three runs carry
  `protrusion_aspect_max_final` non-zero: `r007_05` 3.044 (2646 cells, n_spots_final 5, n_tips_final 1,
  protr_peak 1.316, grip 0.105, mech_p_ratio 1.393), `r007_06` 2.959 (3375 cells, n_tips 1,
  tube_diam_final 1.373, mech_p_ratio 2.604 grown), `r007_03` 1.636 (3420 cells, n_tips 2, grip 0.053).
  Eye reads all three as fat lobes/buds over a red domain, no finger — reconfirms L3 (r004_06 was 2.207).
- **`r007_05` is the round's closest-to-Okuda pattern:** n_spots_final 5, spot_spacing 12.15 cells —
  a handful of large domains, not the 150+ specks the control carries.

## SURPRISES (r007)
- **`r007_02` gyr_prolate_peak 2.404** vs control 1.10 (2.2×) and `act_at_tip_peak` 7.758 vs 2.11 —
  the round's most elongated body plus a tip-focused chemistry spike, both unpredicted (round tested
  n_tubes). Eye sees a dented heart-shaped lobed blob, not a finger; grip only 0.081.
- **`r007_10` cells 18632** (a_sw 0.1) vs control 10749 (1.7×), red_frac_peak 0.756 — the round's
  largest, reddest tissue, unpredicted.
- `r007_08` and `r007_14` empty. `r007_08` = P12 NaN crash from frame 1635 (7th: r001_09/r002_08/
  r003_11/r004_11/r005_09/r006_09) — a crash, not a zero.
- `r007_04` P1+P4 broken: act_cv_peak 23.77 rail, act_mean_floor 5.6e-45, mech_p_ratio 0 — chemistry
  flashed once and died; a second stable field survives unmeasured (eye), gripping nothing.

## STANDING LAWS (r007 update)
- **L3 HOLDS, weakening at low-N.** `r007_05` aspect_final 3.044 / `r007_06` 2.959 / `r007_03` 1.636 —
  all sustained non-zero finals, all buds by the eye, n_tips 1–2. Sustained aspect now recurs at
  low-N one-few-spot states (r004_06, r007_05/06/03) but no finger yet.
- **L4 INVERTED on the a_sw sweep.** grip RISES with cell count here: 3076→0.041, 4471→0.058,
  12631→0.100, 18632→0.107. High-N folding out-grips low-N coarsening (matches r005_04/r006_06); L4's
  low-N→high-grip claim is route-dependent, not general.

## Route A closures (r007)
- **`cell_grow.a_sw` on `b_gs_plain_soft_lo`:** 0.1→18632 cells/protr 1.236/grip 0.107; 0.2→12631/1.238/
  0.100; 0.5→4471/1.133/0.058; 0.7→3076/1.098/0.041. a_sw (activation switch width) drives cells,
  protr AND grip all DOWN monotone — a low switch width grows more tissue and grips harder. Use a_sw
  0.1; no break, chemistry intact throughout. CLOSED, 4 values.
- **`cell_mechanics.K_lumen` on `b_gs_shaping_soft_lo`:** 0→10119 cells/protr 1.261/grip 0.083;
  0.1→10352/1.247/grip 0.070. Near-inert — K_lumen slightly LOWERS protr and grip. Use 0. 2 values.

## Facts established r008
- **No tube, no aspect, no prediction scored.** Route-A-only round (5 real runs, slots 05–08
  empty). All lobed/undulating spheres; `protrusion_aspect_max_final` 0, `n_tips_peak` ≤1,
  `n_tubes_final` 0 everywhere. Round-best `protr_peak` 1.279 (`r008_01`) still < 1.3.

## SURPRISES (r008)
- **`r008_01` red_frac_final 0.430** vs control 0.011 (39×) and cells 10296 vs 3397 (3×) — the
  round's largest, reddest tissue (K_lumen 2, shaping recipe), unpredicted; eye: tetrahedral blob.
- `r008_02`/`r008_03` `mech_p_ratio` 0.0 with chemistry alive (act_max 0.82) — a stalled ratio.

## STANDING LAWS (r008 update)
- **L3 HOLDS.** No aspect signal this round to test it — aspect_final 0 on all 5 runs.

## Route A closures (r008)
- **`cell_grow.hill` on `b_gs_plain_soft_lo`:** 1→7887 cells/protr 1.151/grip 0.073;
  2→6998/1.168/0.078; 8→7480/1.188/0.077. Near-INERT — faint monotone protr rise, grip flat.
  CLOSED, 3 values. Stop sweeping hill.
- **`cell_mechanics.K_lumen` on `b_gs_shaping_soft_lo`:** 2→10296 cells/protr 1.279/grip 0.0915.
  With r007 (0→1.261/0.083; 0.1→1.247/0.070): NON-MONOTONE, 2 mildly beats 0 on protr AND grip —
  REVISES the r007 "use 0, lowers grip" closure. 3 values; one rung (5) left to confirm or reject.

## Facts established r009
- **New campaign grip best = `r009_05` grip_peak 0.190 AND invagination best = 0.566** (prev
  0.183 r006_06, 0.550 r006_03), on a 3–6-armed red-tipped STAR at 9038 cells: protr_peak 1.454,
  gyr_prolate 1.429, corr_act_rad_peak 0.907, reduced_volume 0.534, mech_p_ratio 2.155 (grown).
  High-N folding route. Arms aspect ~1.9, aspect_final 0 — no finger (L3).
- **Round-best protr = `r009_02` protr_peak 1.479** (9023 cells, grip 0.183, invagination 0.489,
  n_apop 98) — the campaign's strongest protrusion in this fresh run, still a lobed star, aspect_final 0.
- **Chemistry death cost 3 of 8 Route-B slots** (`r009_01/03/07`): P4 broken, act_mean_floor
  5.6e-45, protr rails 1.02, mech_p_ratio 0 — a settings failure, not biology.

## SURPRISES (r009)
- **`r009_05` invagination_peak 0.566 and grip_peak 0.190** — both campaign records, but the round
  posed protr, so both records are UNPREDICTED (surprise + record).
- **`r009_02` n_apop 98** — the round's only apoptosis; invagination 0.489 vs control 0.193 (2.5×),
  unpredicted (round tested invagination threshold, which it missed).
- **`r009_15` n_spots_peak 334 / cells 22565** vs control 100/3397 — round's finest, largest tissue,
  unpredicted; opposite of Okuda's ~10-spot scale.
- **`r009_12` shape_idx_p95_peak 4.59** vs control 4.19 (+9.5%) at 10100 cells, gyr_prolate 1.55 —
  mesh distending under a lumen load (K_lumen 8); a mild rail on the shape index, not a phenotype.
- `r009_10` P12 NaN crash from frame ~1562 (8th: r001_09/r002_08/r003_11/r004_11/r005_09/r006_09/
  r007_08) — a crash, not a zero.

## STANDING LAWS (r009 update)
- **L3 HOLDS.** `r009_04` aspect_final 1.699 / n_tips 1 (morphology 'tube') read as a lobe by the eye;
  `r009_02`/`r009_05` protr 1.48/1.45 stars, aspect_final 0. aspect_peak/n_tubes ≠ finger.
- **L4 HOLDS as folding-route, non-exclusive.** grip rises with cells to ~9000 then falls:
  3486→0.041, 6917→0.079, 9038→0.190 / 9023→0.183, 22565→0.104. High-N folding peaks mid, out-grips
  low-N coarsening — matches r005_04/r006_06/r007's a_sw inversion.

## Route A closures (r009)
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.000433→3486 cells/protr 1.079/grip 0.041;
  0.001732→22565/1.270/0.105. rate drives cells ×6.5 and protr AND grip UP monotone — the growth
  lever that also lifts grip (unlike rho, which left protr flat). Break untested above 0.001732.
  2 values, sweep OPEN upward.
- **`cell_grow.hill` on `b_gs_plain_soft_lo`:** 16→7902 cells/protr 1.213/grip 0.082. With r008
  (1/2/8 → protr 1.151/1.168/1.188, grip flat ~0.076): INERT across 1–16. CLOSED, 4 values.
- **`cell_mechanics.K_lumen` on `b_gs_shaping_soft_lo`:** 8→10100 cells/protr 1.249/grip 0.080.
  With r007/r008 (0/0.1/2 → grip 0.083/0.070/0.0915, protr 1.261/1.247/1.279): grip capped ~0.09,
  protr non-monotone peaks at 2. Use 2. CLOSED 0–8, 4 values.
- **`interface_tension.K_purse` on `b_gs_shaping_soft_lo`:** 0→6917 cells/protr 1.186/
  grip 0.079 — the purse-string at 0 = baseline. 1 value; sweep open.
- **`cell_grow.rho` on `b_bru_question`:** empty (crashed) again — base RULED OUT (reconfirms
  r001/r004/r005). Do not reuse `b_bru_question`.

## Facts established r010
- **This round's control is the WEAKEST parent family run so far:** `r010_00` protr_peak 1.168 /
  grip_peak 0.078 / 3397 cells (vs r005–r009 controls at grip ~0.12, protr ~1.28). `r010_06`
  bit-identical → seed floor ~0 on this recipe. Re-baseline grip claims to 0.078 here.
- **No tube, no finger, no sustained aspect anywhere.** `protrusion_aspect_max_final` 0 on all 10
  real runs, `n_tips_final` 0, `n_tubes_final` (1–2) disputed by the eye as lobes. L3 HOLDS.
- **Round-best grip = `r010_01` grip_peak 0.140 but it is whole-body STRETCH, not a spot:**
  gyr_prolate_peak 4.923 (a smooth prolate egg by the eye), corr_act_rad_peak 0.896, 5791 cells,
  n_apop 149. Elongation ⊥ local grip reconfirmed (matches r003_01/r005_01).
- **Best chemistry–shape coupling on a single spot = `r010_05`:** corr_act_rad_peak 0.935,
  act_at_tip_peak 9.796, act_cv_peak 11.29, n_spots_final 1, 2242 cells — one fat BUD (aspect_peak
  1.41, aspect_final 0), grip 0.105. Closest to Okuda's ~10-cell single-spot scale this round.

## SURPRISES (r010)
- **`r010_01` gyr_prolate_peak 4.923** vs control 1.205 (4.1×) — round's most elongated body, a
  prolate egg, unpredicted (round tested aspect); grip only 0.140, no finger.
- **`r010_05` act_at_tip_peak 9.796** vs control 3.569 (2.7×) and n_spots_final 1 — round's sharpest
  tip chemistry on the coarsest (one-spot) pattern, unpredicted.
- `r010_03` P4 death: act_mean_floor 5.6e-45, act_cv_peak 33.97 rail, mech_p_ratio 0 — chemistry
  flashed once and died; a settings failure, not biology.
- `r010_11` P12 NaN crash from frame 1562 (9th: r001_09/r002_08/r003_11/r004_11/r005_09/r006_09/
  r007_08/r009_10) — a crash, not a zero. `r010_08/09/10/15` empty.
- `r010_13` mech_p_ratio 0 with chemistry alive (act_max 0.692) at 10739 cells — stalled ratio at
  high N (reconfirms r008_02/03).

## STANDING LAWS (r010 update)
- **L3 HOLDS.** aspect_final 0 / n_tips_final 0 on all 10 real runs; `r010_01` protr 1.377 is a
  whole-body egg, `r010_05` aspect_peak 1.41 a bud. aspect_peak/n_tubes ≠ finger.

## Route A closures (r010)
- **`cell_chem_react.F` on `b_gs_plain_soft_lo`:** 0.03→5413 cells/protr 1.155/grip 0.068 (`r010_14`).
  1 value; sweep incomplete.
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.003464 and 0.006928 both EMPTY (crashed). With r009
  (0.000433/0.001732 ran clean): rate CRASHES this base above ~0.0017. Ceiling found. Do not exceed
  0.001732 on b_gs_plain_soft_lo.
- **`interface_tension.K_purse` on `b_gs_shaping_soft_lo`:** 0.25→8620 cells/protr 1.263/
  grip 0.083 (`r010_12`); 3→10739/1.214/0.062 (`r010_13`). With r009 (0→6917/1.186/0.079): K_purse
  raises cells monotone; protr PEAKS at 0.25 then falls, grip falls. Use 0.25. 3 values.
- **`cell_grow.rho` on `b_bru_question`:** empty (crashed) again — base RULED OUT (reconfirms
  r001/r004/r005/r009). Do not reuse `b_bru_question`.

## Facts established r011
- **New fresh-campaign protr best AND new campaign invagination best = `r011_04`:** protr_peak
  **1.598** (prev fresh best 1.479 r009_02), invagination_peak **0.593** (prev 0.566 r009_05),
  grip 0.184, reduced_volume 0.651, 4762 cells, mech_p_ratio 2.586 (grown), n_tubes 4. A 3–4-armed
  red-tipped star (eye), aspect_final 0 — no finger (L3). Confirmed protr>1.479.
- **New campaign-deepest volume collapse = `r011_06` reduced_volume_final 0.459** (prev 0.526
  r006_06), grip_peak 0.189 (≈record 0.190 r009_05), invagination 0.561, 13172 cells, red_frac 0.434,
  n_tubes 5 — a 4–5-armed star, high-N folding route. mech_p_ratio 1.885.
- **The branched target is being MADE and not MEASURED.** `r011_04`/`r011_06` are red-tipped 3–5-arm
  stars by the eye, yet n_tips_peak 0 and morphology='sphere' on both. The tip detector does not fire
  on broad-based arms — the metric, not the tissue, is the limit.
- **b_bru_question is inert-NaN, not empty.** rho 0.05/0.1/0.25 RAN (60066/47291/45352 cells,
  campaign-largest tissue) but P4+P12 broke every value (act_cv 244/216/211 NaN, grip 0.003, protr
  ~1.02). It grows enormous tissue and cannot pattern. RETIRE — reverses the r001–r010 "empty" note.

## SURPRISES (r011)
- **`r011_04` invagination_peak 0.593** — campaign record, but the round posed protr>1.479, so the
  invagination record is UNPREDICTED (surprise + record).
- **`r011_06` reduced_volume_final 0.459** vs control 0.873 — campaign-deepest collapse, unpredicted
  (round tested n_tips).
- **`r011_09` cells 60066** vs control 3397 (17.7×) — campaign-largest tissue, but chemistry NaN
  (P12); a rail on the apparatus (b_bru_question rho 0.05), not the tissue.
- `n_spots_peak` rails at exactly 100 across `r011_00/01/02/03/05/10/13/14`; reconfirmed cap.
- `r011_09/12/15` P12 NaN blowup (act_cv 200+); `r011_01/02/05` P4/P1 chemistry deaths.

## STANDING LAWS (r011 update)
- **L3 HOLDS.** `r011_04` aspect_peak 6.097, `r011_06`/`r011_03` multi-arm stars — all aspect_final 0,
  n_tips_final 0, read as broad arms/lobes by the eye. aspect_peak/n_tubes ≠ finger.
- **L4 HOLDS as folding-route, non-exclusive.** grip vs cells: `r011_14` 6142→0.077, `r011_10`
  10973→0.058, `r011_04` 4762→0.184, `r011_06` 13172→0.189 — high-N folding out-grips low-N (0.189 at
  13172 is the highest-N high-grip on file). No clean monotone; route-dependent, as r007/r009.

## Route A closures (r011)
- **`cell_grow.rho` on `b_bru_question`:** 0.05→60066 cells/protr 1.030/grip 0.003; 0.1→47291/1.020/
  0.003; 0.25→45352/1.016/0.003 — all P4+P12 (NaN). rho near-INERT on cell count; base grows huge
  inert tissue, chemistry diverges every value. CLOSED, 3 values — RETIRE base.
- **`cell_chem_react.F` on `b_gs_plain_soft_lo`:** 0.038→6142 cells/protr 1.185/grip 0.077. With r010
  (0.03→5413/1.155/0.068): F raises cells, protr, grip mildly monotone. 2 values, open.
- **`interface_tension.K_purse` on `b_gs_shaping_soft_lo`:** 6→10973 cells/protr 1.165/grip
  0.058. With r009/r010 (0→0.079, 0.25→0.083, 3→0.062): grip peaks at 0.25, falls monotone to 6.
  Use 0.25. CLOSED 0–6, 4 values.
- **`cell_mechanics.K_V` on `b_gs_shaping_soft_lo`:** 5→2010 cells/protr 1.012/grip 0/P1 broken/
  act_cv 21.9. K_V 5 freezes growth (cells flat 2000) and kills the pattern. 1 value; sweep down.

## Facts established r012
- **Grip ceiling ~0.19 reconfirmed as a hard wall.** Three runs aimed at it: `r012_05` grip_peak
  **0.1903** (5–6-armed red-tipped star, 9038 cells, protr 1.454, n_tubes 6, reduced_volume 0.534)
  TIES the campaign best (r009_05 0.190); `r012_06` 0.180 (16680 cells); `r012_02` 0.086. No run
  exceeds 0.19. High-N folding route; aspect_final 0 — no finger (L3).
- **New campaign invagination best = `r012_02` invagination_peak 0.60786** (prev 0.593 r011_04), a
  bilobed cleft heart at 6432 cells, gyr_prolate_peak 3.073 — but grip only 0.086. Elongation ⊥ grip
  (reconfirms r003_01/r005_01/r010_01).
- **`r012_06` reduced_volume_final 0.4519** near campaign-deepest (0.459 r011_06), a 4–6-cusp star
  with DEPLETED tips: red_at_tip 0.209 (eye confirms white tips) yet act_at_tip 2.03 — tip-activator
  metric disputed.
- Control this round is the WEAK family: protr 1.168 / grip 0.078 / 3397 cells (same as r010).

## SURPRISES (r012)
- **`r012_02` invagination_peak 0.608** — campaign record, UNPREDICTED (round posed grip>0.19).
- **`r012_02` gyr_prolate_peak 3.073** vs control 1.205 (2.5×) — round's most elongated body, a
  bilobed heart, unpredicted; grip collapses to 0.086.
- **`r012_06` reduced_volume_final 0.4519** vs control 0.873 — near-deepest collapse, unpredicted
  (round tested grip).
- `r012_14` act_max_peak 3.3e9 / cells 32246 (b_bru_question vth_frac 6) — NaN blowup, a rail on the
  apparatus not the tissue. `n_spots_peak` rails at exactly 100 across most runs (reconfirmed cap).

## STANDING LAWS (r012 update)
- **L3 HOLDS.** `r012_15` aspect_final 4.04 / n_tips_final 1 / morphology='tube' — eye sees a lobed
  sphere (broad bulges, aspect 1.205); `r012_05` n_tubes 6 star, aspect_final 0. Sustained aspect ≠
  finger.
- **L4 HOLDS as folding-route, non-exclusive.** grip vs cells: `r012_02` 6432→0.086, `r012_05`
  9038→0.190, `r012_13` 12748→0.119, `r012_06` 16680→0.180. High-N folding out-grips low-N; no clean
  monotone, route-dependent (as r007/r009/r011).

## Route A closures (r012)
- **`cell_mechanics.K_V` on `b_gs_shaping_soft_lo`:** 10→2004 cells/protr 1.044/grip −0.000/P4
  (growth frozen, chemistry dead); 40→12748/1.322/grip 0.119; 80→12755/1.303/grip 0.096. Strong
  lever with a THRESHOLD between 10 and 40 — below ~10 growth freezes and the pattern dies (reconfirms
  r011 K_V 5→2010/P1), 40 is the peak, 80 plateaus/slightly falls. Use 40. 3 values (+r011's 5).
- **`cell_grow.vth_frac` on `b_bru_question`:** 4→empty (crashed); 6→32246 cells/protr 1.020/grip
  0.003/P4+P12 (act_cv 179, NaN). Huge inert tissue that cannot pattern — RETIRE (reconfirms r011).
- **`cell_grow.rho` on `b_bru_question`:** 0.5→empty (crashed). Base ruled out (reconfirms
  r001/r004/r005/r009/r010).
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.003464 and 0.006928 both empty (crashed) — reconfirms
  the r010 ceiling ~0.0017. Do not exceed.

## Facts established r013
- **The ~0.19 grip wall is BROKEN — new campaign best `r013_05` grip_peak 0.273** (prev 0.190
  r009_05/r012_05, ×1.44), on a 7–8-armed red-tipped STAR at 12201 cells. Same run holds the
  **campaign-best invagination 0.617** (prev 0.608 r012_02) AND **campaign-deepest reduced_volume
  0.285** (prev 0.459 r011_06): protr_peak 1.408, n_tubes_peak 11, corr_act_rad_peak 0.921,
  mech_p_ratio 2.196 (grown). aspect_final 0 — a star by the eye, no finger by the metric (L3).
- **A NEW, much stronger control family this round.** `r013_00` protr_peak 1.295 / grip_peak 0.118 /
  12984 cells / invagination 0.411 / reduced_volume 0.654 — a 3–4-lobe undulating body (eye), vs the
  r010–r012 weak family (3397 cells, grip 0.078). `r013_07` bit-identical → seed floor ~0; differences
  are real. Re-baseline grip to 0.118 here.
- **Two more high-N folding stars clear the old ceiling region:** `r013_04` grip 0.179 / protr 1.392 /
  13111 cells; `r013_06` grip 0.173 / protr 1.405 / 8240 cells / n_tips_peak 7. Eye reads both as
  genuine red-tipped multi-arm stars.
- **The fold is NOT driven by apoptosis.** `r013_05` (n_apop 0) has the deepest invagination 0.617;
  the death runs `r013_01` (n_apop 157) / `r013_04` (115) / `r013_06` (87) are all shallower
  (invag ≤0.503). Apoptosis active, no inward fold — reconfirms r001–r012.

## SURPRISES (r013)
- **`r013_05` grip 0.273, invagination 0.617, reduced_volume 0.285 — three campaign records, all
  UNPREDICTED** (the round posed `protrusion_aspect_max_peak > 5.234`, refuted at 0.324).
- **`r013_14` act_cv_peak 16.77** vs control 2.20 (7.6×), n_spots_final 17, one-sided red hemisphere
  cap (Lambda 1) — round's spikiest, coarsest chemistry, unpredicted; nearest Okuda's ~10-spot scale.
- `r013_13` P12 NaN crash (chemistry non-finite from frame 1352; empty metrics). Five slots
  (`r013_02/03/09/10/11`) empty — 6 of 16 lost.

## STANDING LAWS (r013 update)
- **L3 HOLDS but CONTESTED by the eye.** `r013_05` n_tubes 11 / `r013_06` n_tips_peak 7 / `r013_04` —
  all aspect_final 0, yet the eye calls all three red-tipped multi-arm STARS and "the sharpest
  protrusions in this campaign." The metric zeroes tips the picture shows; for the star morphotype
  escalate to the eye + n_tubes, not aspect/n_tips.
- **L4 HOLDS as folding-route.** grip 0.273 at 12201 cells (`r013_05`) is the highest-N high-grip on
  file; high-N folding out-grips low-N coarsening again (r005_04/r006_06/r009/r011/r012).

## Route A closures (r013)
- **`cell_mechanics.Lambda` on `b_gs_shaping_soft_lo`:** 0→12318 cells/protr 1.257/grip 0.093;
  0.2→12166/1.238/0.096; 1→2688/1.189/grip 0.071/act_cv 16.77 (chemistry destabilizes, cells collapse
  to 2688). protr/grip near-flat 0–0.2, pattern dies at 1. Use ≤0.2. With r003/r004 (0.3/0.45/0.6/0.8
  peaked at 0.6): CLOSED at the low end.
- **`cell_chem_react.F` on `b_gs_plain_soft_lo`:** 0.054→7638 cells/protr 1.185/grip 0.087. With r010/r011
  (0.03→1.155/0.068, 0.038→1.185/0.077): F raises cells and grip mildly monotone. 3 values.
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.003464 and 0.006928 both empty again — reconfirms the
  r010 ceiling ~0.0017.
- **`b_bru_question` (`cell_grow.rho` 0.5, `vth_frac` 4):** empty (crashed) — base RULED OUT, reconfirmed.

## Facts established r014
- **`r013_05` (the star) is seed-floor-robust, not a fluke.** `r014_02` — a set_impl edit that turned
  out inert — is **bit-identical** to the verbatim control `r014_00`: grip 0.273, invag 0.617,
  reduced_volume 0.285, n_tubes 11, 12201 cells all reproduce exactly. The campaign records stand.
- **The star's arms are growth-buckling; the purse-string adds ~3 arms, not the star.** `r014_01`
  removed `interface_tension` from r013_05: n_tubes 11→8, grip 0.273→0.220, protr 1.408→1.337,
  invag 0.617→0.608 — still a 6–8-petal rosette (eye). Line tension is a modifier, not the generator.
- **The cell_chem_from_shape feedback leg is INERT on b_star (beta=0).** `r014_02` `set_impl cell_chem_from_shape →
  apical_area` is bit-identical to the curvature control — the impl swap changes nothing because
  beta=0, so the loop is not closed. Any feedback-geometry claim requires beta≠0 first.
- **Halving `cell_chem_react.kk` 0.062→0.031 coarsens to Okuda's ~10-cell scale but runs the reaction
  hot.** `r014_04`: spot_spacing_cells 4.04→12.01 (Okuda scale reached), but cells 12201→17182 (+41%),
  red_frac_peak 0.472→0.814, act_max 1.47→2.19, grip 0.247, protr 1.358. Eye: 7–9 red-tipped pointed
  arms ("first clear multi-armed protrusion field"). kk is the wavelength lever but NOT clean —
  coupled to activator amplitude and overgrowth.
- **A mechanics leg at low growth does not finger r007_03.** Adding `interface_tension`
  (`r014_05`, aspect_peak 1.082, eye: never protrudes, worse than r007_03's own aspect_final 1.636)
  or halving kk (`r014_06`, aspect 1.091, broad lobes) both refuted aspect>2.152. The Okuda-envelope
  tube hypothesis is falsified on both routes.

## SURPRISES (r014)
- **`r014_02` bit-identical to control** — `set_impl cell_chem_from_shape → apical_area` is a no-op RAIL:
  the feedback operator did literally nothing (beta=0), so the impl choice is unmeasurable here.
- **`r014_04` red_frac_peak 0.814** vs control 0.472 (1.7×) and **cells 17182** vs 12201 (1.4×) —
  round's reddest, largest tissue, UNPREDICTED (the round posed spot_spacing). Halving kk runs hot.
- **`r014_05` aspect_final 0.685 / n_tips_final 1** on a near-dead pattern (grip 0.022, red_frac
  0.007) — a sustained-aspect flicker with no chemistry behind it; the opposite of a gripped finger.
- `r014_12`/`r014_14` P12 NaN from frame 1352 (recurring crash); 5 of 15 slots lost.

## STANDING LAWS (r014 update)
- **L3 HOLDS.** `r014_04` eye "clear multi-armed protrusion field / pointed fingers", yet
  aspect_final 0, n_tips 0. The metric still zeroes the tips the picture shows.
- **L5 (new) — `cell_chem_from_shape.beta = −0.5` extinguishes the activator, morphotype-independent.**
  evidence: `r014_03` (star r013_05) P4 broken, invag 0.617→0.031, act extinct; `r014_07` (bud
  r011_04) P4 broken, invag 0.593→0.060. Negative feedback at this magnitude is a kill switch, not a
  dimple. status: HOLDS (2 runs). Sweep beta small (±0.05) before trusting the sign as a fold lever.

## Route A closures (r014)
- **`cell_chem_react.F` on `b_gs_plain_soft_lo`:** 0.06→7929 cells/protr 1.188/grip 0.091. With
  r010/r011/r013 (0.03→0.068, 0.038→0.077, 0.054→0.087): F raises cells and grip mildly monotone.
  4 values, no break — extend or use ~0.06.
- **`cell_chem_react.kk` on `b_gs_plain_soft_lo`:** 0.052→10302 cells/protr 1.185/grip 0.089 — FIRST kk
  value on this base, ≈ cell_chem_react.F's effect. 1 value; sweep open.
- **`cell_mechanics.Lambda` on `b_gs_shaping_soft_lo`:** 2→2006 cells/protr 1.018/grip −0.000/act_cv
  9.74/P1+P4 (growth frozen, pattern dead). Reconfirms r013 (Lambda 1→2688 dead). Use ≤0.2; Lambda≥1
  kills. CLOSED upward.

## Facts established r015 — THE FIRST FINGER
- **`r015_06` is the campaign's first FINGER, confirmed by the eye AND the aspect metric at once —
  L3 broken.** `protrusion_aspect_max_final` **7.544** (all 200+ prior runs 0.0), `n_tips_final` 2 /
  `n_tips_peak` 6, `n_tubes_peak` 5, morphology 'tube'. The eye: "campaign's first convincing branched
  protrusion — thin red-tipped arms, genuine fingers not bulges," genus intact. Every prior sustained
  aspect either read 0 at end or the eye called it a lobe; here BOTH say finger. This is the campaign's
  answer.
- **`r015_06` also takes the grip AND protr records:** `grip_peak` **0.344** (prev best 0.273
  r013_05, ×1.26), `protr_peak` **2.199** (prev 1.598 r011_04 / 1.408 star, ×1.38). At only **5690
  cells** with `n_spots_final` 3 / `spot_spacing_cells` 25.94 — a THIRD grip route (few coarse spots →
  genuine fingers), distinct from low-N coarsening and high-N folding. `act_max` 3.41 (control 1.47),
  `act_at_tip_peak` 8.95, `red_at_tip` 0.979 — chemistry chases the extruding tips; `mech_p_ratio`
  3.765. Predicted grip>0.247, CONFIRMED. **SINGLE SEED — the edit that made it is not identified from
  metrics; REPLICATE and name the composition before building on it** (the coarse tip-chasing pattern +
  hot activator is the signature of a closed geometry→chemistry feedback, `cell_chem_from_shape` beta≠0, or of
  `cell_chem_seed:cones` — either would be the first of its kind here).
- **Apoptosis DEGRADES the star's fold, does not deepen it.** `r015_04` death twin of the star:
  n_apop 77, invagination 0.617→**0.601**, grip 0.273→0.250, protr 1.408→1.367. Predicted invag>0.617,
  REFUTED. Reconfirms r001–r013: apoptosis active, no inward fold.
- **The star REQUIRES growth; its arms are growth-buckling.** `r015_03` (growth ablated): protr 1.03,
  2021 cells (≈seed 2000), red_frac 0, grip 0.010, mech_p_ratio 0 — collapses to a coarsened-spot
  sphere. Predicted protr<1.1, CONFIRMED. Reconfirms r014_01.
- **`r015_02` bit-identical to control** (grip 0.273, invag 0.617, n_tubes 11, 12201 cells) — inert
  edit, seed floor ~0 on b_star reconfirmed (matches r014_02).

## SURPRISES (r015)
- **`r015_06` `protrusion_aspect_max_final` 7.544, `n_tips_final` 2, `protr_peak` 2.199 — records, all
  UNPREDICTED** (the round posed grip>0.247, which it also cleared). The aspect_final 7.544 against a
  universal prior 0.0 is the single most informative number the campaign has produced.
- **`r015_15` `corr_act_rad_final` −0.210 / `grip_final` −0.004** (b_bru_question, divide factor 2.4,
  32333 cells, red_frac 0.939) — first strong NEGATIVE coupling, activator anti-correlated with radius.
  b_bru PATTERNED this time (P intact, no NaN, unlike r011/r012) yet anti-grips. Base reconfirmed useless.
- **`r015_04` n_apop 77 yet invagination FELL** 0.617→0.601 — apoptosis subtracts from the fold.

## STANDING LAWS (r015 update)
- **L3 REFUTED.** `r015_06` `protrusion_aspect_max_final` 7.544 / `n_tips_final` 2 / `n_tubes` 5, and
  the eye INDEPENDENTLY calls it genuine fingers — first agreement of the sustained-aspect metric with
  the picture. status: REFUTED (metric fires and picture confirms; the "aspect ≠ finger" claim held only
  while every aspect_final was 0 or the eye dissented).
- **L4 extended — a THIRD high-grip route.** grip 0.344 at 5690 cells (`r015_06`) via genuine fingers,
  below the low-N coarsening (~4000 cells) and high-N folding (~9000–13000) regimes. High grip is not
  cell-count-bound; the finger route out-grips both. status: HOLDS as route-dependent, now 3 routes.

## Route A closures (r015)
- **`cell_chem_react.kk` on `b_gs_plain_soft_lo`:** 0.057→8406 cells/protr 1.175/grip 0.080; 0.066→6101/
  1.161/0.074. With r014 (0.052→10302/1.185/0.089): kk UP → cells DOWN, protr and grip mildly DOWN
  monotone. Use low kk ~0.052. 3 values, no break.
- **`cell_divide.factor` on `b_bru_question`:** 2.4→32333 cells/protr 1.027/grip −0.004/corr −0.210/
  red_frac 0.939/act_cv 1.41 — patterns cleanly but ANTI-grips. RETIRE reconfirmed (no NaN this time).
- **`cell_grow.rho`/`vth_frac` on `b_bru_question`, `cell_grow.rate` on `b_gs_plain_soft_lo`:** empty
  (crashed). rate ceiling ~0.0017 reconfirmed; b_bru RULED OUT reconfirmed.

## Facts established r016 — THE FINGER GENERATOR ISOLATED
- **The finger's arm is GROWTH-BUCKLING + a LIVE `cell_chem_from_shape` feedback leg; the purse-string is
  only a modifier.** Ablation of the finger `r015_06` (baseline: protr 2.199, aspect_final 7.544,
  ~5690 cells):
  - remove `interface_tension` (`r016_03`): finger SURVIVES & STRENGTHENS — protr_peak 1.83,
    `protrusion_aspect_max_final` **21.094** (new campaign record, prev 7.544 r015_06), n_tips 2,
    n_tubes 6, morphology 'tube', red_at_tip 0.996, mech_p_ratio 2.511 (grown), eye "first true
    protrusions, tapering pointed arms." Tension is a modifier (as on star r014_01), not the generator.
  - remove `cell_chem_from_shape` (`r016_04`): finger → bulge (aspect 0.435, protr 1.088) AND growth runs
    away to **50532 cells** with pattern UNIFORM (act_cv_final 0.0, red_frac 1.0). So `cell_chem_from_shape`
    is LIVE (beta≠0) on r015_06 — the campaign's first CLOSED-FEEDBACK composition — and it BRAKES
    growth; removing it un-brakes it (~9× the finger's 5690 cells).
  - halve feedback F0 0.046→0.023 (`r016_01`): finger lost, protr 1.272, cells 22231, aspect_peak
    0.726 — needs ~full F0 (a knee, not linear).
  - remove growth (`r016_02`): protr 1.015, 2006 cells, pattern faded. remove division (`r016_06`):
    2000 cells, P4 broken. Both trivially required.
- **The finger route is SEED-ROBUST, not a single-seed fluke.** r015_06 (seed A) and r016_03 (fresh
  seed, tension removed) both finger — answers the r015 replication worry; the ablated seed is stronger.
- **Apoptosis does not invaginate (reconfirmed).** `r016_07` (star r013_05 + cell_die, n_apop
  85): invag_peak 0.605 < the star's 0.617.

## SURPRISES (r016)
- **`r016_03` `protrusion_aspect_max_final` 21.094** — campaign record (prev 7.544 r015_06),
  UNPREDICTED (round posed n_tubes<5). The strongest finger on file, made by REMOVING the purse-string.
- **`r016_04` cells 50532** vs finger 5690 (~9×) — campaign-largest patterning tissue, UNPREDICTED
  (posed aspect<4); pattern washes uniform (act_cv_final 0.0). Removing the feedback leg un-brakes
  growth — a mechanism for the project's growth-overshoot.
- **`r016_15` `corr_act_rad_final` −0.284 / grip −0.006** (b_bru_question, cell_divide.factor 3, 6122
  cells) — anti-grip reconfirmed (matches r015_15).
- `r016_11` P4+P12 NaN (act_max 3.3e9, b_bru vth_frac 4) — apparatus rail.

## STANDING LAWS (r016 update)
- **L3 REFUTED (reconfirmed).** `r016_03` aspect_final 21.094 + eye "genuine fingers" — metric and
  picture agree again.
- **L4 HOLDS — finger route.** aspect_final 21.094 at 5511 cells (`r016_03`); finger route out-grips
  folding and is now seed-robust.
- **L6 (new) — the `cell_chem_from_shape` feedback leg both MAKES the finger and BRAKES growth.** evidence:
  remove it (`r016_04`) → aspect 7.544→0.435 AND cells 5690→50532 uniform; halve F0 (`r016_01`) →
  protr 2.199→1.272, cells→22231. status: HOLDS (2 runs). Sweep F0/beta both signs from r016_03.
- **L5 UNTESTED this round** (no negative-beta run).

## Route A closures (r016)
- **`cell_chem_react.kk` on `b_gs_plain_soft_lo`:** 0.07→5351 cells/protr 1.164/grip 0.068 (`r016_12`).
  With r014/r015 (0.052→10302/0.089, 0.057→8406/0.080, 0.066→6101/0.074): kk UP → cells DOWN, grip
  DOWN monotone. Use low kk ~0.052. CLOSED, 4 values.
- **`cell_chem_diffuse.d_a` on `b_gs_plain_soft_lo`:** 0.02→6909 cells/protr 1.181/grip 0.036/act_cv_peak
  3.228/n_spots_final 446 (`r016_14`) — FIRST d_a value, finest pattern and lowest grip of the base.
  1 value, sweep open.
- **`cell_divide.factor` on `b_bru_question`:** 3→6122 cells/protr 1.035/grip −0.006/corr_act_rad
  −0.284 (`r016_15`) — patterns cleanly but ANTI-grips (reconfirms r015_15). RETIRE base.
- **`cell_grow.vth_frac` on `b_bru_question`:** 4→32246 cells, P4+P12 NaN (`r016_11`); `cell_grow.rho`
  0.5 and `cell_grow.rate` empty (crashed). b_bru RULED OUT reconfirmed.

## Facts established r017
- **The finger is seed-robust at 3 seeds and out-folds the star.** `r017_05` reproduces `r015_06`
  (purse-string ON) exactly: protr_peak 2.199, grip_peak 0.34356, aspect_final 7.544, n_tips 6,
  n_tubes 5, 5690 cells, mech_p_ratio 3.765 (grown), corr_act_rad_peak 0.944 — PLUS new campaign
  records **invagination_peak 0.754** (prev 0.617 r013_05) and **protrusion_aspect_max_peak 34.616**
  (prev 21.094 r016_03). Finger now confirmed at 3 seeds (r015_06/r016_03/r017_05); the low-N finger
  (5690 cells) folds DEEPER than the high-N star (12201 cells, 0.617).
- **The control is now the FINGER** (`r016_03` promoted): `r017_00` protr 1.83, aspect_final 21.094,
  grip 0.173, invagination 0.628, n_tubes 6, 5511 cells, morphology 'tube'. Re-baseline to the finger.
- **Four `b_star` basis variations EXTINGUISH the activator (P4).** `r017_02/03/04/06` all collapse to
  a sphere, protr ≤1.024, act_max_final 0, activator extinct within ~3 frames (eye: "flashes once then
  dies"). The new seeding/gate levers (b_star_{relgate,oriented,sharp,avoid,cones,pressure}, exact
  slot→spec map not given this round) mostly destabilize the RD; only the plain finger patterns.
  Attribute each death to its variation before reuse.

## SURPRISES (r017)
- **`r017_05` invagination_peak 0.754 and protrusion_aspect_max_peak 34.616 — two campaign records,
  UNPREDICTED** (round posed spot_spacing_cells_peak>66.82, hit exactly 66.82 → refuted at the bound).
  The finger folds deeper than any star.
- **`r017_12`/`r017_14` v_cell_mean 0.608** vs control 0.338 (1.8×), cells flat ~2000, divide factor
  5≈8 bit-near-identical — division SATURATED on b_bru, a RAIL (reconfirms r006_12/14).
- `r017_11` cells 32246 / act_max_peak 3.3e9 NaN (b_bru vth_frac 10) — apparatus rail (reconfirms
  r012_14/r016_11).

## STANDING LAWS (r017 update)
- **L3 REFUTED (reconfirmed).** Control `r016_03` aspect_final 21.094 and `r017_05` aspect_final 7.544
  — both fingers by eye AND metric.
- **L4 HOLDS — finger route, now the DEEPEST fold too.** `r017_05` invagination 0.754 at 5690 cells
  out-folds the 12201-cell star (0.617). High grip AND deep fold both live on the low-N finger.
- **L7 (new) — `cell_chem_diffuse.d_a` is the wavelength/coarsening lever on b_gs_plain: higher d_a →
  fewer, wider spots, higher grip.** evidence: d_a 0.02→n_spots 446/grip 0.036 (r016_14), 0.04→250/
  0.056 (r017_13), 0.16→9/0.119 (r017_15, spot_spacing 21.15 = Okuda scale). status: HOLDS (3 values,
  grip monotone up, n_spots monotone down).

## Route A closures (r017)
- **`cell_chem_diffuse.d_a` on `b_gs_plain_soft_lo`:** 0.04→8121 cells/protr 1.159/grip 0.056/n_spots 250;
  0.16→3407/1.327/grip 0.119/invag 0.560/n_spots 9/spot_spacing 21.15 (bilobed bud, eye). With r016_14
  (0.02→6909/1.181/0.036/n_spots 446): d_a UP → n_spots DOWN monotone (coarsens), grip UP monotone,
  protr non-monotone (min at 0.04). At 0.16 reaches Okuda ~10-spot scale — the clean wavelength knob
  (cleaner than kk, which coupled to overgrowth, r014). Use high d_a; grip 0.119 = bud, not finger.
  3 values, open upward.
- **`cell_divide.factor` on `b_bru_question`:** 5→2024 cells/v_cell 0.607/protr 1.018/grip −0.001;
  8→2000/0.608/1.018/corr −0.167 — division SATURATED (both ~2000, cells 0.608), anti-grip. RETIRE
  reconfirmed (r006/r015/r016).
- **`cell_grow.vth_frac` on `b_bru_question`:** 10→32246 cells, P4+P12 NaN (act_cv 179). RETIRE reconfirmed.
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.003464/0.006928 empty (crashed) — ceiling ~0.0017
  reconfirmed (r010/r012/r013).

## Facts established r018
- **The finger recipe is LOCKED at seed floor 0 across 4 slots.** Control `r018_00` and three
  Route-B slots `r018_01/02/06` are **bit-identical**: protr_peak 1.83, `protrusion_aspect_max_final`
  21.094, grip 0.173, invagination 0.628, n_tubes 6, 5511 cells. Their prediction passes
  (aspect_peak>5, n_tips_peak>2) restate the control. STOP spending seed slots on it — the finger is
  fixed at 3+ seeds (r015_06/r016_03/r017_05/here).
- **Apoptosis on the finger no longer subtracts from the fold, but shrinks the arm.** `r018_07`
  (n_apop 75): invagination_peak 0.635 > control 0.628 — the FIRST death run (vs r001–r015, all fell)
  not to reduce the fold — but aspect_final 21.094→6.067, protr 1.83→1.797. The +0.007 fold gain is
  within coupling noise; death trades the finger for a hair of fold, deepens nothing.
- **`r018_03` reaches Okuda's spot scale but buds instead of fingering.** protr 1.615, 3220 cells,
  red_frac_final 0.155 (control 0.041, 3.8×), n_spots_final 5, spot_spacing_cells 23.85 (Okuda
  ~10-spot scale), corr_act_rad_peak 0.949, grip 0.173, aspect_final 2.792 — eye: 5 fat red-tipped
  buds. Refuted invag>0.754 (0.309). Coarsening the finger's pattern to the target wavelength gives
  up the buckling arm.
- **Two more `b_star` variations extinguish the activator (P4).** `r018_04/05`: sphere, protr ≤1.021,
  act_max_final 0, extinct within ~3 frames. Reconfirms r017 — only the plain finger patterns.

## SURPRISES (r018)
- **`r018_07` n_apop 75 yet invagination ROSE 0.628→0.635** — first death run not to subtract from
  the fold (all r001–r015 fell), UNPREDICTED as a reversal; but arm degraded (aspect 21.094→6.067).
- **`r018_03` red_frac_final 0.155** vs control 0.041 (3.8×) — coarser, hotter pattern, UNPREDICTED
  (round posed invagination); the coarsening that hit Okuda's spot scale.
- **`r018_15` cells 29974** vs control 5511 (5.4×), act_max NaN — apparatus rail (b_bru a_sw 0.5,
  P4+P12), not tissue.
- **`r018_14` n_spots 143 / spot_spacing 2.78** (chi 0.3) — finest pattern on file; wrong direction
  for Okuda's ~10.
- No new records: r017_05 still holds invag 0.754, aspect_peak 34.616, grip 0.344.

## STANDING LAWS (r018 update)
- **L3 REFUTED (reconfirmed).** Control finger aspect_final 21.094 + eye "genuine fingers".
- **L4 HOLDS — finger route.** aspect_final 21.094 at 5511 cells; seed-floor-0 across 4 slots.
- **L7 HOLDS, ceiling found.** `cell_chem_diffuse.d_a` 0.3 → P4 death (`r018_12`, 2051 cells, extinct
  ~frame 220). Wavelength knob dies between 0.16 (r017, alive, n_spots 9) and 0.3. Use ≤0.16.
- **L5 (negative beta kill switch) UNTESTED this round.**

## Route A closures (r018)
- **`cell_chem_diffuse.chi` on `b_gs_plain_soft_lo`:** 0.3→4488 cells/protr 1.257/grip 0.100/act_cv 3.455/
  n_spots 143/spot_spacing 2.78/reduced_vol 0.752 (`r018_14`). The FIRST chi value off the
  campaign-wide fixed 1.3 (see campaign §F009). chi 0.3 makes the FINEST pattern on file — chi is a
  wavelength lever, low chi → many fine spots. 1 value; sweep UP (chi>1.3) to coarsen toward Okuda.
- **`cell_chem_diffuse.d_a` on `b_gs_plain_soft_lo`:** 0.3→2051 cells/protr 1.014/grip 0.000/P4 dead
  (`r018_12`). Extends L7: d_a alive at 0.16, dead at 0.3 — ceiling between. CLOSED upward at 0.3.
- **`cell_grow.a_sw` on `b_bru_question`:** 0.5→29974 cells/P4+P12 NaN (`r018_15`, act_max NaN, act_cv
  173). b_bru RULED OUT reconfirmed (grows huge inert tissue, cannot pattern).
- **`cell_grow.rate` on `b_gs_plain_soft_lo`:** 0.003464/0.006928 empty (crashed) — ceiling ~0.0017
  reconfirmed.

## Facts established r019
- **New campaign protr best = `r019_06` protr_peak 2.296** (prev 2.199 r015_06/r017_05, ×1.05), a
  5–6-fingered star at 5825 cells: grip_peak 0.322, protrusion_aspect_max_peak 25.018, n_tips_peak 6,
  n_tubes_peak 7, corr_act_rad_peak 0.927, reduced_volume_final 0.417, mech_p_ratio 3.746. Eye: "5–6
  sharp fingers grow from sphere." A stronger low-N finger sibling of the control; its edit is NOT
  identifiable from metrics — REPLICATE and name it. mech_p_ratio 3.746 is NOT forcing (no
  interface_push in the space, user §3) — grown.
- **New campaign invagination = `r019_01` invagination_peak 0.978 BUT P11 broken** (self-intersecting
  mesh, not a clean tissue): 23527 cells, reduced_volume_final 0.288, grip_peak 0.251, protr 1.557 — a
  multi-armed star that over-grows and folds THROUGH itself at ~frame 1550 (eye). The clean
  invagination record stays r017_05 0.754.
- **High-N grown folding trades the finger's arm for a hollow star, cleanly to ~17k cells.** `r019_02`
  (16828 cells): protr 1.269, grip_peak 0.202, invagination_peak 0.598, reduced_volume_final 0.283,
  corr_act_rad_peak 0.898, n_tubes 4 — a 5–6-armed hollow star, red at tips, all premises intact,
  mech_p_ratio 1.508. Cell count is the arm→fold lever; rupture (P11) appears by 23527 (r019_01).
- **Finger control reproduced (seed floor).** `r019_00` = the r016/r017/r018 finger: protr 1.83,
  protrusion_aspect_max_final 21.094, grip 0.173, invagination 0.628, n_tubes 6, 5511 cells.

## SURPRISES (r019)
- **`r019_06` protr_peak 2.296 — campaign record, UNPREDICTED** (round posed n_tips_peak>6, refuted at
  6). The strongest protrusion on file.
- **`r019_01` invagination_peak 0.978 — deepest fold on file but P11 broken, UNPREDICTED** (round posed
  aspect<21.094). Records the over-growth rupture, not a clean fold.
- **`r019_15` act_max_peak 10.69 / act_mean_peak 1.837** vs control 3.61/0.70 (~3×) — campaign-hottest
  chemistry, yet corr_act_rad_final −0.254 (negative) and grip_final −0.001: a fully DECOUPLED hot
  pattern (b_bru_question a_sw 0.7), UNPREDICTED. Patterned clean (no NaN), unlike prior b_bru.
- `n_spots_peak` rails at exactly 100 on control and 01/02/05/06 — reconfirmed measurement cap.

## STANDING LAWS (r019 update)
- **L3 REFUTED (reconfirmed).** Control finger aspect_final 21.094 and `r019_06` aspect_peak 25.018 +
  eye "sharp fingers" — metric and picture agree.
- **L4 HOLDS — finger route.** `r019_06` protr 2.296 / grip 0.322 at 5825 cells; the low-N finger
  out-protrudes every high-N folding star (r019_01/02 at 16–23k cells).
- **L8 (new) — over-growing the finger past ~15k cells trades the arm for a deep hollow fold, and
  ruptures (P11) by ~23k.** evidence: finger 5511 cells (aspect_final 21.094, clean) → r019_02 16828
  cells (n_tubes 4 hollow star, invag 0.598, clean) → r019_01 23527 cells (P11 broken, invag 0.978,
  self-fold). Cell count is the arm-vs-fold lever with a rupture ceiling near 23k. status: HOLDS (3 pts).

## Route A closures (r019)
- **`cell_chem_diffuse.chi` on `b_gs_plain_soft_lo`:** 0.65→6649 cells/protr 1.198/grip 0.079/n_spots 187/
  reduced_vol 0.777; 2→6986/1.164/0.077/n_spots 29/0.799. chi UP (0.65→2) COARSENS n_spots 187→29 but
  protr and grip stay FLAT (~1.18/0.078). With r018 (chi 0.3→143): chi 2 is nearest Okuda's ~10 (29
  spots) yet still 3× too many and grip never lifts off baseline. Use chi 2 for coarsening only; sweep
  open above 2.
- **`cell_grow.a_sw` on `b_bru_question`:** 0.1/0.2 empty (crashed); 0.7→23883 cells/protr 1.023/grip
  0.002/act_cv 1.357 — grows huge, patterns HOT (act_max 9.31, no NaN this time) but fully decoupled.
  b_bru RULED OUT reconfirmed (anti-grip, not NaN; matches r015_15/r017). Retire.
- **`cell_grow.rate` on `b_gs_plain_soft_lo`** (0.003464/0.006928) and **`cell_grow.rho` on
  `b_bru_question`** (0.5): both empty — rate ceiling ~0.0017 and b_bru ruled out, both reconfirmed.

## Facts established r020
- **The finger lives ONLY in the ~5500-cell window; no Route-B edit reproduced it.** Control
  `r020_00` = the finger (protr 1.83, aspect_final 21.094, grip 0.173, invag 0.628, n_tubes 6, 5511
  cells). All 8 non-control runs read aspect_final 0 / n_tips_final 0 — the arm gone. The edits either
  OVER-grew (`r020_01` 24661, `r020_03` 17182, `r020_02` 12976, `r020_04` 12235 cells → folding/lobed
  stars) or UNDER-grew (`r020_05` 3144, `r020_07` 2224 → buds/dead). Pushing the finger's growth
  HARDER multiplies nothing — n_tubes>6/>11, n_tips>6, protr>1.598 all refuted, all by over-growth.
- **New campaign-deepest CLEAN volume collapse = `r020_03` reduced_volume_final 0.267** (prev 0.283
  r019_02 / 0.285 r013_05; r019_01's 0.288 was P11-broken), a 17182-cell folding star with grip_peak
  0.24678, n_tubes 9, invag_peak 0.607, act_max 2.189, all premises intact; eye "~7 red-tipped
  fingers." Strongest `complex`/`branched` specimen of the round — folded, not fingered (aspect_final 0).
- **Apoptosis still does not deepen the fold.** `r020_04` death twin (n_apop 89, 12235 cells):
  reduced_volume 0.462, SHALLOWER than the clean folding stars (0.267). Reconfirms r001–r019.

## SURPRISES (r020)
- **`r020_03` reduced_volume_final 0.267 — campaign-deepest clean collapse, UNPREDICTED** (round posed
  n_tips_peak>6, refuted at 0).
- **`r020_14` grip 0.0006 / corr_act_rad_final 0.033 with a LIVE 43-spot pattern** (act_cv 1.825,
  act_max 0.942) — a fully DECOUPLED live pattern on a sphere: the geometry ignores the chemistry.
  UNPREDICTED (Route A, cell_chem_from_shape.beta −2).
- **`r020_01` cells 24661** vs control 5511 (4.5×), reduced_volume 0.308 heavy crumple-fold yet
  aspect_final 0, P11 INTACT — over-growth above r019_01's 23527-cell tear without rupturing.
  UNPREDICTED (posed n_tubes). The L8 rupture ceiling is stochastic near 23–25k, not a hard line.
- `n_spots_peak` rails at exactly 100 on control/01/03/05/12/14 — reconfirmed measurement cap.

## STANDING LAWS (r020 update)
- **L3 REFUTED (reconfirmed, both ways).** Control finger aspect_final 21.094 — the metric fires on the
  true buckled arm. But `r020_03` eye "~7 fingers" reads aspect_final 0 / n_tips 0 on a FOLDING star:
  the finger metric fires only on the growth-buckled arm, not on folding-star arms.
- **L4 HOLDS — finger route is a ~5500-cell window.** Every r020 edit that moved cell count off ~5500
  lost the arm (aspect_final 0 at 3144→24661 cells). High grip AND the arm both require the window.
- **L8 HOLDS, rupture ceiling NOT clean.** `r020_01` 24661 cells → reduced_volume 0.308, P11 intact,
  ABOVE r019_01's 23527-cell tear. Rupture near 23–25k is stochastic, not a hard threshold.
- **L9 (new) — `cell_chem_from_shape.beta` < 0 is base-dependent: KILLS on b_star, DECOUPLES on b_gs_plain.**
  evidence: r014 beta −0.5 on b_star → P4, activator extinct (L5); `r020_14` beta −2 on b_gs_plain →
  activator ALIVE (act_max 0.942) but grip 0.0006 / corr 0.033. status: HOLDS (2 bases).

## Route A closures (r020)
- **`cell_chem_diffuse.chi` on `b_gs_plain_soft_lo`:** 2.8→6194 cells/protr 1.229/grip 0.123/n_spots_final
  14/spot_spacing 11.01/reduced_vol 0.696 (`r020_12`). With r018/r019 (0.3→143, 0.65→187, 2→29): chi UP
  COARSENS n_spots monotone; 2.8 = 14 spots, NEAREST Okuda's ~10 on file, but grip stays FLAT ~0.12
  across the whole range. Use chi 2.8 for coarsening only; grip never lifts off baseline. CLOSED, 4
  values — the wavelength knob does not buy coupling.
- **`cell_chem_from_shape.beta` on `b_gs_plain_soft_lo`:** −2→3276 cells/protr 1.034/grip 0.0006/corr_act_rad
  0.033/act_cv 1.825, chemistry ALIVE (act_max 0.942, `r020_14`). Negative beta DECOUPLES the pattern
  from the shape (grip→0) WITHOUT extinguishing the activator on this base (see L9). 1 value; sweep
  small negative to find where decoupling begins.

## Facts established r021
- **New campaign grip record = `r021_02` grip_peak 0.35083** (prev 0.34356 r017_05 / 0.344 r015_06,
  ×1.02), on a NON-finger: 7550 cells, corr_act_rad_peak 0.927, r_cv_peak 0.456, n_apop 59, morphology
  'sphere', aspect_final 0. Eye reads 5–6 red-tipped fingers; the arm metric zeroes them (over-grown
  past the ~5500 window). Predicted invagination_peak>0.598 → CONFIRMED 0.646. The grip wall (~0.34) is
  nudged, not broken; high grip no longer requires the buckled arm.
- **The 2.296 finger is SEED-ROBUST — `r019_06` REPLICATED by `r021_03`** (metrics reproduce exactly):
  protr_peak 2.296, 5825 cells, grip_peak 0.322, protrusion_aspect_max_peak 25.018 / aspect_final
  14.719, n_tips_peak 6, mech_p_ratio 3.746 (grown), eye "genuine sharp radial spikes." Predicted
  n_tips_peak>6 → refuted at 6. r019_06 is now the campaign's 2nd named finger composition (with the
  5511-cell control), confirmed at 2 seeds; both live in the ~5500–5825-cell window (L4).
- **`cell_chem_from_shape` is DEAD at beta 0 on b_gs_plain — a labyrinth, not a coupled field.** `r021_12`
  (beta 0): grip 0.006, mech_p_ratio 0, eye "clearest Turing stripe/labyrinth field the campaign has
  shown" on a perfect sphere — chemistry patterns, mechanics never respond. The `set_impl cell_chem_from_shape`
  edit changed nothing measurable (null). Direct evidence the feedback leg needs beta≠0 (reconfirms L6).
- **Over-grown finger folds deep and CLEAN at 21832 cells.** `r021_06`: reduced_volume_final 0.288,
  grip 0.283, invagination 0.816, 23 spots, aspect_final 0, P11 INTACT. Predicted reduced_volume<0.40 →
  CONFIRMED. Above r019_02's 16828 clean, below r019_01's 23527 tear — L8 fold regime, rupture ceiling
  stochastic (21.8k intact, matches r020_01's 24661 intact).
- **`r021_05` P4 death** (one-sided lobed dome, act_max_final 0, extinct early) — refuted n_spots>3 at 1.

## SURPRISES (r021)
- **`r021_02` grip_peak 0.35083 — campaign record, UNPREDICTED** (round posed invagination). The first
  grip record NOT on a buckled-arm finger (aspect_final 0), at 7550 cells.
- **`r021_02` n_apop 59 with invagination_peak 0.646** — the round's only death run and it did NOT
  subtract from the fold (control finger 0.628); apoptosis+over-growth co-occur with the grip record.
- **`r021_12` grip 0.006 on a bold labyrinth** vs control 0.173 (0.03×) — a fully decoupled live stripe
  field, the campaign's cleanest Turing pattern, UNPREDICTED (posed a null set_impl).
- **`r021_06` cells 21832** vs control 5511 (4×), reduced_volume 0.288 — near-deepest clean collapse.
- `n_spots_peak` rails at exactly 90–100 on 01/02/03/04/06 — reconfirmed measurement cap.

## STANDING LAWS (r021 update)
- **L3 REFUTED (reconfirmed, both ways).** `r021_03` aspect_final 14.719 + eye "spikes" agree; but
  `r021_01`/`r021_02` eye "fingers" read aspect_final 0 (over-grown to 7058/7550 cells) — the arm metric
  fires only inside the ~5500-cell buckling window, not on over-grown folding stars.
- **L4 HOLDS — finger route is a ~5500–5825-cell window, now seed-robust at TWO compositions.**
  `r021_03` = `r019_06` replicated (protr 2.296, 5825 cells); the 5511-cell control the other.
- **L8 HOLDS, rupture ceiling stochastic.** `r021_06` 21832 cells → reduced_volume 0.288, P11 intact.
- **L10 (new) — `cell_chem_from_shape` coupling scales with beta on b_gs_plain; beta 0 = decoupled labyrinth.**
  evidence: beta 0→grip 0.006/corr flat (`r021_12`), beta 1→grip 0.103/corr_act_rad_peak 0.834
  (`r021_14`); with r020 beta −2→grip 0.0006. status: HOLDS (3 pts, positive beta buys grip).

## Route A closures (r021)
- **`cell_chem_from_shape.beta` on `b_gs_plain_soft_lo`:** 0→4199 cells/protr 1.027/grip 0.010/act_cv 2.202
  (labyrinth, decoupled, `r021_12`); 1→7775/1.230/grip 0.103/corr_act_rad_peak 0.834/red_frac 0.461
  (`r021_14`). beta 0→1 raises cells, protr AND grip monotone (0.010→0.103); with r020 (−2→0.0006), grip
  is monotone in beta across [−2,1]. Positive beta couples pattern to shape — the wavelength knobs never
  did this. Use beta≥1; break above 1 untested. 2 values (+r020's −2), open upward.

## Facts established r022
- **The 2.296 finger is seed-robust at THREE seeds.** `r022_03` = `r019_06` = `r021_03`: protr_peak
  2.296, 5825 cells, grip 0.322, aspect_final 14.719 / aspect_peak 25.018, n_tubes 7, n_tips 3,
  mech_p_ratio 3.746 (grown), corr_act_rad_peak 0.927. Predicted protr>2.296 REFUTED at the bound
  (replicate vs copied prediction). The 5511-cell control finger (`r022_00_ctrl`, `r022_05`
  bit-identical) is the other named composition; both live in the ~5500–5825-cell window (L4).
- **Apoptosis on the finger PRESERVES the arm and can deepen the fold — reversing r015's "death
  degrades the arm."** `r022_07` (n_apop 80): protr 2.225, aspect_final 9.197, invagination_peak 0.700
  (> control 0.628), gyr_prolate_peak 2.830. `r022_06` (n_apop 87): protr 2.031, aspect_final 17.787,
  invagination 0.507 (< control), grip 0.236. Both keep the finger; the fold goes OPPOSITE ways at
  equal death. Clean-fold record stays r017_05 0.754.
- **Grip climbs past the arm window.** `r022_02` (6280 cells): grip_peak 0.221 > control 0.173
  (CONFIRMED), protr 1.937, aspect_final 0 / n_tips 0 — arm zeroed just above the ~5825 ceiling; eye
  sees fingers (L3).
- **Half the round lost to execution:** 7 of 14 slots empty (08–12,14); `r022_14` = P12 NaN blowup
  (chemistry non-finite from frame ~1710). No P4 chemistry death this round.

## SURPRISES (r022)
- **`r022_07` invagination_peak 0.700** vs control 0.628 AND gyr_prolate_peak 2.830 vs 1.293 (2.2×) —
  round's deepest fold, on a death+finger, UNPREDICTED (posed invag>0.754, missed).
- **`r022_06` n_apop 87 yet invag 0.507 < control 0.628** — apoptosis SHALLOWER here, opposite of
  `r022_07` at near-equal death: death→fold is not monotone in n_apop.
- `n_spots_peak` rails at 90–103 across finger runs — reconfirmed measurement cap.

## STANDING LAWS (r022 update)
- **L3 REFUTED (reconfirmed, both ways).** `r022_03` aspect_final 14.719 + eye "spikes" agree;
  `r022_02` eye "fingers" reads aspect_final 0 at 6280 cells (over the window).
- **L4 HOLDS — finger window ~5500–5825, now seed-robust at 3 seeds** (`r022_03`). `r022_02` 6280 cells
  → arm gone; grip still climbs (0.221).
- **L10 REVISED — `cell_chem_from_shape.beta` coupling SATURATES above beta 1.** beta 0/1/2 → grip
  0.006/0.103/0.107 (`r021_12`/`r021_14`/`r022_13`); monotone up to 1, FLAT above. status: HOLDS,
  plateau found — stop sweeping beta up.
- **L11 (new) — apoptosis on the finger preserves the arm but its effect on the fold is NOT monotone
  in death.** evidence: `r022_07` n_apop 80 → invag 0.700 (+0.072), aspect_final 9.197; `r022_06`
  n_apop 87 → invag 0.507 (−0.121), aspect_final 17.787. status: OPEN (2 runs, opposite signs at equal
  n_apop) — needs a max_mark_frac ladder on the finger death twin.

## Route A closures (r022)
- **`cell_chem_from_shape.beta` on `b_gs_plain_soft_lo`:** 2→8763 cells/protr 1.235/grip 0.107/n_spots 49/
  reduced_vol 0.692 (`r022_13`). With r020/r021 (−2→0.0006, 0→0.010, 1→0.103): grip monotone to beta 1
  then FLAT (2≈1). Use beta 1; SATURATES above. CLOSED upward, 4 values.
- **`cell_chem_from_shape.F0` on `b_gs_plain_soft_lo`:** 0.02→5801 cells/protr 1.108/grip 0.043/mech_p_ratio
  0.0 (`r022_15`). Halving F0 kills the coupling (no mechanical response), matching r016_01's knee
  (0.023→finger lost). Feedback needs ~full F0. 1 value, knee reconfirmed.

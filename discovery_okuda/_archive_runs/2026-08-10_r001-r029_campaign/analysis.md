# Round r001

## 1. What happened
Control `r001_00`: inert sphere — no growth, no chemistry, `protr_peak` 1.014, `grip` not
measurable. Three families moved off it:

- **Growing GM/coral (01,02,03,10,11,14,15).** Chemistry ignites and coarsens; tissue grows.
  High-growth 01/02/03 reach 14003/13700/11554 cells, low-growth 10/11/14/15 ~3.3k. All land
  `corr_act_rad_peak` 0.80–0.833 and `grip_peak` 0.026–0.047 — the campaign's strongest
  coupling to date — yet `protr_peak` only 1.063–1.103, `n_tubes_peak` 0, no tip.
- **Saturated activator (12,13).** `act_max` 9.6, `red_frac` railed 1.0: the whole surface goes
  high. `corr_act_rad_peak` collapses to 0.016 (floor −0.643), `grip` negative. protr flat.
- **Ablations/extinction (07,08,09).** 08/09 rho=0 (P1 broken, no growth) = control twins.
  07 activator dies (`act_max_final` 0, P4 broken, `red_frac` 1.0); a second field coarsens
  (`act_cv_peak` 2.19) but grips nothing (`grip_peak` 0.0006).
- **Lost: 04,05 empty; 06 eye-only.** Execution loss, three slots.

## 2. What was learned
- 01 refuted `grip>0.08` (0.042); 02 refuted `protr>1.20` (1.103); 03 refuted `protr>1.15`
  (1.063). The coupling is real and record-high but **grips colour, not shape** — the eye is
  unanimous: coarsening red domains on a "lumpy/faceted ball," no finger in any run.
- 07 confirmed `act_cv>1.0` (2.19) only as an extinct specimen: uniform-field coarsening, not a
  gripping pattern.
- protr does not track growth: 4× the cells (14k vs 3.3k) gives the same protr (~1.09).
  reduced_volume falling to 0.925 is packing/faceting, not protrusion.

## 3. What went wrong
- **Metric vs eye, control included.** Every eye report on 00/08/09 sees a persistent 5–6 orange
  spot pattern while `act_cv`=`act_max`=`n_spots`=0. A fixed non-activator colour channel, or the
  metric reads the wrong field. Flag before trusting act_* on a no-chemistry run.
- `rd_interface_tension` inert in 14/15 (null result for that operator at this setting).
- 12/13 show overdriving the reaction is counter-productive — saturation anti-couples.

## 4. What to do next
- The coupling ceiling is ~0.047 grip / 0.83 corr with pattern gripping only colour. Next edit
  must convert grip to relief: add a mechanical leg (line tension or bending keyed to activator)
  on the 01/03 recipe, predict `protr_peak` and check `mech_p_ratio` for a forced signature.
- Do NOT push activator harder — 12/13 prove saturation kills the grip.
- Re-run 04/05/06; three slots produced nothing.

# Round r002

## 1. What happened
Control `r002_00`: grows 2000→13700 cells, `protr_peak` 1.103, `grip` 0.047, `n_tubes` 0 — a
lumpy spotted sphere, the r001 story unchanged. The batch spreads around it:

- **`r002_12` is the round's best specimen:** `protr_peak` 1.129, `grip_peak` 0.05735,
  `reduced_volume` 0.8935 at only 3223 cells — a scalloped/lobed ball (eye: "shallow soccer-ball
  lobes"), deepest lobing of the campaign. Still no finger (`n_tubes` 0, `protrusion_aspect_max` 0).
- **`r002_09` best grip after control:** 0.0489 at 6765 cells, `protr` 1.105 — lobed sphere.
- **`r002_05`/`06` are bit-identical to the control** across every field (`grip` 0.0467).
- **Overdrive twins `r002_10`/`11`:** `act_max` 6.75/6.88, `red_frac` 0.95/0.97 → `corr_act_rad`
  −0.23/−0.13, `grip` ~0, smooth sphere; `r002_11` reaches 19236 cells, the largest tissue yet.
- **Zero-chemistry-metric `r002_14`/`15`:** all `act_*` = 0 yet the eye sees a coarsening
  orange/blue two-phase field.
- **Low/no-growth `r002_01`(2000c)/`02`/`03`/`07`/`08`/`13`:** protr 1.038–1.094, patterned
  spheres, no protrusion.

## 2. What was learned
- The best admissible `protr_peak` moved 1.103→1.129 (`r002_12`) and `reduced_volume` 0.914→0.894,
  i.e. lobing deepened — but `protrusion_aspect_max` stays 0 and `n_tubes` 0. **Deeper lobes, no
  finger; the ~1.3 wall is intact.** grip>0.06 refuted on `r002_05`/`06`, but those measured the
  control (below).
- **Overdrive reconfirmed a 2nd time** (`r002_10`/`11` vs r001_12/13): saturating the reaction
  drives `corr_act_rad` NEGATIVE and kills grip. Settled.
- `r002_07` confirmed `act_cv>0.5` (2.20) — trivially, the pattern always ignites; uninformative.
- protr still ⊥ growth: `r002_08` blips `protrusion_aspect_max` 1.602 (1 tip) and `r002_13`
  0.754 (10 tips) but both hold `protr_peak` ~1.09 — single-frame specks, not sustained fingers.

## 3. What went wrong
- **`r002_05`/`06` never differed from the control** — every metric bit-identical. The edit did
  not land, and both slots carried a grip>0.06 prediction scored "refuted": two slots spent
  re-measuring `r002_00`. Verify the diff applies before a run is scored.
- **Field-read bug recurs** (`r002_14`/`15`, as r001_00/08/09): `act_*` reads flat 0 while a
  two-phase field visibly coarsens. `act_*` is on the wrong channel for these recipes.
- `rd_interface_tension` inert on `r002_12`/`13` — null across settings, now twice.

## 4. What to do next
- Lobing is deepening on low-cell recipes (`r002_12`: 3223 cells, `reduced_volume` 0.894). Push
  that leg: `r002_12`'s composition with a mechanical amplifier (bending/line-tension keyed to
  activator), predict `protr_peak`>1.2 and read `mech_p_ratio` for a forced signature.
- Fix the `r002_05`/`06` no-op and the `r002_14`/`15` field-read before spending more slots.
- Do NOT re-push the reaction; overdrive is closed.

# Round r003

## 1. What happened
Control `r003_00`: `protr_peak` 1.127, `grip` 0.0528, `act_cv_peak` 2.202, 3209 cells, `n_tubes` 0
— a lobed spotted sphere. **Nothing beat the control's protr.** The batch spread along a growth
axis and hit the same ceilings:
- **Growth ladder (cells → protr_peak):** 3209→1.127, 3331→1.106, 3649→1.119, 6335→1.111,
  6683→1.095, 9982→1.019, 13920→1.028, 15679→1.036. protr **falls** as cells rise; the biggest
  tissues are the roundest.
- **Overdrive twins `r003_08`/`09`** (bit-identical): `act_max` 9.47, `red_frac` 1.0 →
  `corr_act_rad` −0.137, `grip` −0.0006, smooth sphere at 9982 cells.
- **Zero-chemistry-metric `r003_12`/`13`:** all `act_*`=0 at 15679/13920 cells, yet the eye sees a
  coarsening orange/blue field. protr 1.036/1.028 — pure growth, no shape.
- **Bit-identical pairs:** `01`==`00`(control), `05`==`06`, `08`==`09`, `14`==`15`.
- `r003_07` best pattern: `act_cv_peak` 2.647, 120 spots, 6335 cells; still `protr` 1.111,
  `grip` 0.055.

## 2. What was learned
- **protr is INVERSE to growth here, not independent.** 3209c→1.127 down to 9982c→1.019: more
  cells pack smoother, the opposite of "growth drives protrusion." The r001/r002 "protr ⊥ growth"
  is now "protr falls with growth." Best protr = the smallest tissue = control.
- **Overdrive reconfirmed a 3rd time** (`08`/`09` vs r001_12/13, r002_10/11): `corr_act_rad`
  −0.137, `grip` −0.0006. Settled beyond doubt.
- `r003_07` confirmed `act_cv>2.5` (2.647) — the strongest pattern of the campaign (120 spots),
  and it grips colour not shape: `grip` 0.055, no finger. Pattern strength ≠ protrusion.
- All `grip_peak` < 0.06; best 0.0551 (`07`). The grip>0.06 gate and the 1.3 protr wall both hold.

## 3. What went wrong
- **Four bit-identical pairs, seed spread STILL unmeasured.** `01` re-measured the control (no-op
  edit, refuted `protr>1.3`); `05`/`06`, `08`/`09`, `14`/`15` are each two runs at the SAME seed —
  identical to the last digit, so they bound noise at 0 and waste the replicate. Replicates must
  change seed or they cost a slot for nothing.
- **Field-read bug a 3rd recurrence** (`12`/`13`, as r001_00/08/09, r002_14/15): `act_*` flat 0
  while a two-phase field coarsens on screen. Wrong channel; unfixed across three rounds.
- `rd_interface_tension` inert a 3rd time (`10`/`11`) — null result, now three rounds running.

## 4. What to do next
- Growth is exhausted as a protr lever — it lowers protr. Stop climbing cell count. Return to the
  low-cell lobing leg (control at 3209c already gives the round's best protr 1.127) and add a
  mechanical amplifier (bending/line-tension keyed to activator) as r002 recommended but was never
  run; predict `protr_peak`>1.2, read `mech_p_ratio` for a forced signature.
- Fix the replicate-seed bug and the `act_*` field-read before spending more slots.
- Do NOT push the reaction (overdrive closed 3×) and do NOT sweep `rd_interface_tension` (inert 3×).

# Round r004

## 1. What happened
Control `r004_00`: `protr_peak` 1.127, `grip_peak` 0.0528, `act_cv_peak` 2.20, `corr_act_rad_peak`
0.825, 3209 cells, `n_tubes` 0 — the lobed sphere, again the round's best protr. Nothing beat it.
- **Near-control lobers (02,03,04,05,06,08,09,12,14).** `protr_peak` 1.074–1.127, `grip` 0.027–0.053,
  all `n_tubes` 0. `r004_05` deepened prolateness (`gyr_prolate` 1.222, `gyr_oblate` 0.172) at 3637c
  but protr 1.122 < control. `r004_03` is **bit-identical to control** (every field equal) — edit
  never applied.
- **Overdrive (13,15).** `act_max` 9.47/8.25, `red_frac` 0.97, `corr_act_rad` −0.137/−0.28,
  `grip` −0.0006/−0.003 at 9982/5101c. Smooth sphere.
- **Field-read-zero (10,11).** `act_*`=`n_spots`=0 while the eye sees coarsening two-phase domains;
  protr 1.029/1.024, roundest of the round.
- **No-growth (07).** 2001c, `red_frac` 0, stable spot/maze, protr 1.01.

## 2. What was learned
- **protr↓growth, 4th round.** Round's best protr = control at 3209c; `r004_13` at 9982c = 1.019.
  Growth remains the opposite of a protr lever.
- **Overdrive kills grip — 4th AND 5th replication** (`r004_13`,`r004_15`): `corr_act_rad`
  negative, `grip`≈0. Closed beyond doubt.
- **`protrusion_aspect_max`>0.5 is a transient speck, not a finger.** `r004_04` scored "confirmed"
  (0.622, 5 tips) but its `protr_peak` is 1.074 — *below* control — and the eye saw "shallow
  undulations, no protrusion." `r004_02` same (0.742, 4 tips, protr 1.086). The metric fires on
  single-frame blips; do not pose predictions on it (3rd round it has misled).

## 3. What went wrong
- **Replicate-seed bug, 4th round.** `r004_03`==control bit-identical (same seed). Seed spread
  STILL unmeasured after four rounds — every difference in the round is unbounded by noise.
- **Field-read bug, 4th recurrence** (`r004_10`,`r004_11`): `act_*`=0 vs a patterned eye.
- **`rd_interface_tension` inert a 4th time** (`r004_08`,`r004_09`).

## 4. What to do next
- The lobing leg is saturated by tuning alone: 4 rounds, no run clears control's 1.127. A
  **mechanical amplifier keyed to the activator (bending / apical line-tension), NOT growth, NOT a
  stronger reaction** is the only untried lever and was recommended r002/r003 yet never run. Predict
  `protr_peak`>1.2, read `mech_p_ratio` (≈1 grown vs ≈3 forced).
- FIX the replicate-seed and `act_*` field-read bugs before spending more slots; they have each
  wasted 4 rounds.
- Retire `protrusion_aspect_max` and `rd_interface_tension` from the menu.

# Round r005

## 1. What happened
Control `r005_00`: `protr_peak` 1.127, `grip` 0.053, 3209c, lumpy sphere. Nothing moved off it.
- **Same-seed replicates (05,06,07)** bit-identical to control (`protr` 1.127, `grip` 0.05278,
  every field equal) — 5th round the seed spread is unmeasured. Their predictions scored on the
  control: 05 `act_cv>1` conf (2.20), 06 `n_spots<20` refuted (100), 07 `n_spots>50` conf (100).
- **Mild edits (01,02).** 01 → 1.104/3661c, 02 → 1.113/3708c (`gyr_prolate` 1.26, round's most
  elongated), both refuted `protr>1.3`; eye sees lobing/undulation over red domains, no finger.
- **Low-growth (10,11).** ~2000c, `protr` 1.063–1.064, `grip` 0.030 — *below* control.
- **Overdrive (12,13).** `act_max` 7.31/7.49, `red_frac` 0.947/0.966, `corr_act_rad` −0.558/−0.542,
  `grip` negative — 6th/7th replication.
- **`rd_interface_tension` (14,15)** bit-identical pair, inert, `protr` 1.087.
- **Lost/degenerate: 08** extinct field (`act_*`=0, `red_frac` 0) vs a patterned eye = field-read
  bug 5th time; **09** empty `{}` = execution loss.

## 2. What was learned
Five rounds, the 1.3 protrusion wall stands and no run beats control's 1.127. Growth is not a
lever in *either* direction: low-growth 10/11 (~2000c) land 1.063, *below* the 3209c control, so
3209c is a shallow optimum, not a floor to climb off. Overdrive anti-couples again (6th/7th).

## 3. What went wrong
Seed spread STILL unmeasured after 5 rounds (05=06=07=ctrl, 14=15 same-seed). Field-read `act_*`=0
bug 5th recurrence (08). `rd_interface_tension` inert 5th time (14/15). One execution loss (09).

## 4. What to do next
Unchanged and now 5 rounds overdue: the only untried lever is a **mechanical leg keyed to the
activator (bending / apical line-tension), NOT growth, NOT a hotter reaction**. Predict
`protr_peak`>1.2, read `mech_p_ratio` (≈1 grown, ≈3 forced). Fix the same-seed replicate and
`act_*` field-read bugs before more slots burn on them.

# Round r006

## 1. What happened
Control `r006_00`: the usual lobed ball — `protr_peak` 1.127, `grip_peak` 0.053, 3209 cells,
`n_tubes` 0. For the first time in six rounds two runs edged past it: `r006_04` `protr_peak`
**1.153** (`gyr_prolate_peak` 1.327, the round's most elongated, 3608c), and `r006_15`
`grip_peak` **0.0633** — the first run ever past the 0.06 grip gate `r002_12` missed —
(`reduced_volume` 0.865, most deformed; `corr_act_rad_peak` 0.773, 5597c). Every other run
landed 1.087–1.134. All still spheres/lobed balls: `n_tubes` 0 everywhere, `aspect_max` ≤0.394
bar a single-frame `r006_03` blip (1.109, 2 tips, protr 1.134), and the eye reports no finger in
any run.

## 2. What was learned
The two beats are +0.026 protr / +0.010 grip over control — and seed spread is STILL unmeasured
(the only two replicate pairs, 01=05 and 10=11, are bit-identical same-seed), so neither clears a
noise floor nobody has bounded. Growth-off decouples pattern from shape cleanly: `r006_07`
(2001c, no division) holds vivid chemistry `act_cv_peak` 2.15 yet `protr` 1.014, `grip` 0.006 — a
flat sphere (confirmed <1.05). Grip needs the tissue to be growing to grip anything.

## 3. What went wrong
Overdrive anti-couples an **8th** time (`r006_09`: `act_max_peak` 7.49, `red_frac` 0.966,
`corr_act_rad` −0.542, `grip` −0.0029). `rd_interface_tension` inert a **6th** time (10/11,
flagged null). Same-seed replicate bug a **6th** round (01=05, 10=11 bit-identical) — seed spread
unmeasured after six rounds. Three execution losses (12/13/14 empty).

## 4. What to do next
Same as five rounds running: a **mechanical leg keyed to the activator (bending / apical
line-tension), not growth, not a hotter reaction**. But first probe what moved `r006_04`/`r006_15`
above control — re-run BOTH at a fresh seed to see if 1.153 / 0.0633 survive, because until the
same-seed replicate bug is fixed a "record" is indistinguishable from luck.

# Round r007

## 1. What happened
Control `r007_00`: `protr_peak` 1.13, `grip_peak` 0.06338, 5580c, `act_cv_peak` 2.20, n_tubes 0.
No run beats it on either lever. Best `protr_peak` 1.134 (`r007_03` 3641c, `r007_14` 4470c) = ctrl
within noise; best `grip_peak` 0.0598 (`r007_07`, `r007_14`) — BELOW ctrl. `r007_04` bit-identical
to ctrl (same seed). `r007_12`/`13` empty (execution loss).

## 2. What was learned
protr↓growth reconfirmed as a shallow optimum ~3600–4470c: cells→protr 2054→1.014, 2336→1.079,
3179→1.129, 3641/4470→1.134, 5580→1.130, 7744→1.097, 11877→1.021, 12355→1.019. `act_cv>0.5`
(`r007_06` 2.647) and `n_spots>10` (`r007_07` 106) confirmed but grip colour not shape (grip
0.055/0.060). corr_act_rad rides high on near-spheres — `r007_15` 0.830, `r007_07` 0.827 — yet
grip 0.032/0.060: Pearson without amplitude, the grip discipline holding.

## 3. What went wrong
Overdrive anti-couples a **9th/10th** time (`r007_08`/`09`: `act_max` 6.18/7.18, `red_frac` 1.0,
`corr_act_rad` −0.364/−0.277, grip negative, ~12000c). `rd_interface_tension` inert a **7th** time
(`r007_10`/`11`; `r007_10` also P1-broken, division near-off 2054c). Same-seed replicate bug a
**7th** round (`r007_04`==ctrl) — seed spread unmeasured after seven rounds. Two execution losses.

## 4. What to do next
Unchanged and now seven rounds overdue: an activator-keyed **mechanical** operator (bending /
apical line-tension), not growth, not a hotter reaction. Fix the same-seed replicate bug before
any "record" is posed — it has cost a slot every round since r001.

# Round r008

## 1. What happened
Control `r008_00` (=`05`=`06`, bit-identical, same seed): `protr_peak` 1.13, `grip_peak` 0.0634,
`corr_act_rad_peak` 0.762, 5580c, lobed sphere. `r008_09` beats it on BOTH headline axes —
`protr_peak` **1.144**, `grip_peak` **0.0711**, `corr_act_rad_peak` 0.880 — at only 3801c, the
first double-beat in eight rounds. Its low-growth siblings `r008_07`/`08` (3417/3640c) sit just
below (grip 0.052/0.062). High-growth `r008_01`/`02` (7559/7744c) round out to `protr` 1.096/1.097,
grip 0.041. Overdrive `r008_13`/`14` and losses `r008_10`/`11` fill the rest.

## 2. What was learned
The beats are real but sit on ONE seed and inside the unbounded seed noise; `n_tubes` 0,
`aspect_max` 0, and the eye reports a 6–8-fold undulating raspberry, no finger. The mechanism
behind them is COARSENING: `r008_09` holds the fewest, widest, best-separated domains
(`n_spots_final` 21, `spot_spacing_cells` 8.05) and the highest `corr_act_rad`. Grip rises as the
pattern coarsens and as growth falls — grip↓growth now joins protr↓growth. r008_01 predicted
protr>1.3 (refuted, 1.096) and r008_02 gyr_prolate>1.2 (refuted, 1.118): the wall stands.

## 3. What went wrong
Overdrive killed grip an 11th/12th time (`r008_13`/`14`: `act_max` 9.45/9.11, `corr_act_rad`
−0.30/−0.19). `rd_interface_tension` inert an 8th time (`r008_08`/`09`). Same-seed replicate bug an
8th round — `r008_05`/`06` copy the control, and `r008_06` was scored on a protr>1.13 prediction
against the very run it duplicates. `r008_10`/`11` empty. `r008_12` `aspect_max` 0.648 / `n_tips` 3
is the 4th single-frame blip on a 1.066 sphere.

## 4. What to do next
Re-run `r008_09` at 2–3 fresh seeds to bound the double-beat before believing it — and fix the
same-seed replicate so the bound is possible. The coarser-grips-harder signal says push toward
fewer, larger domains (lower growth, coarser chemistry), not a hotter reaction. The mechanical
operator remains the only route to a finger.

# Round r009

## 1. What happened
Control `r009_00`: `protr_peak` **1.158**, `grip_peak` **0.07315**, `corr_act_rad_peak` 0.880,
3767c — the highest control on both axes in nine rounds, so the whole round rides a lucky seed.
`r009_04` still beat it on both: `protr_peak` **1.198**, `grip_peak` **0.09512** (campaign max grip,
prior best 0.0711 `r008_09`), `r_cv` 0.113, `reduced_volume` 0.705 (round's deepest lobing), 4149c.
The rest bracket growth: 05 (3381c) protr 1.105, 06 (5439c) 1.144, 03 (7576c) 1.104.

## 2. What was learned
The double-beat is one seed over a seed-inflated control, so it confirms nothing new — grip>0.071
scored confirmed on `r009_04`, but seed spread is still unmeasured a 9th round. `n_tubes` 0,
`aspect_max` 0, eye sees a scalloped/undulating ball, no finger. protr↓growth holds a 9th round
(4149c peaks, 7576c rounds to 1.104); the shallow optimum ~3800–4150c persists. Grip is the only
axis that moved, and it moved with deeper lobing (`reduced_volume` 0.705), not with a tube.

## 3. What went wrong
Overdrive killed grip a 13th/14th time (`r009_12`/`13`: `act_max` 8.67/8.82, `red_frac` 1.0,
`corr_act_rad` −0.175/−0.181, `grip` −0.001). `rd_interface_tension` inert a 9th time
(`r009_14`/`15`, explicit null). Same-seed replicate bug a 9th round — `r009_01`==`r009_02`
bit-identical (protr 1.091), both scored on n_tubes>0 / protr>1.3 predictions (refuted). Two
execution losses (`r009_08`/`09` empty). Predictions protr>1.3 refuted twice more (01/07, ≤1.145).

## 4. What to do next
Re-run `r009_04` at 2–3 fresh seeds against a fresh-seed control to see if the +0.02 protr / +0.02
grip survives — nine rounds of single-seed beats have never been bounded, and until they are, no
tuning claim can stand. Fix the same-seed replicate. The finger still needs a mechanical operator,
not a parameter.

# Round r010

## 1. What happened
This round IS the r009 recommendation: `r009_04`'s recipe became the control (`r010_00`, bit-identical
— protr 1.198, grip 0.09512, 4149c) and was re-run at fresh seeds. `r010_03` (4544c) landed protr
1.184 / grip 0.09873; `r010_04` (4216c) protr 1.176 / grip 0.09014; `r010_05` failed to reseed
(bit-identical to control). The rest bracket growth: 07 (3819c) 1.138, 08 (3592c) 1.105, 01/02
(5092c, a same-seed pair) 1.112, 06 (4371c) 1.11, 09/10 overdrive (10523/11428c) 1.017.

## 2. What was learned
**The seed spread is measured at last, and it clears the beat.** Across control + the two fresh
seeds, protr = 1.186±0.011 and grip = 0.0947±0.004 — span ≤0.022 / ≤0.0086, an order below the
recipe's +0.06 protr / +0.03 grip lead over the ~1.13/0.06 historical baseline. So the nine-round
run of single-seed double-beats was NOT seed luck: this composition really does sit above the old
optimum. `r010_03` grip 0.09873 is a new campaign max, on the round's coarsest field (n_spots 16,
spacing 10.03) — coarser-grips-harder holds. Still `n_tubes` 0, `aspect_max` 0; no finger.
protr↓growth holds a 10th round (4149c peaks; 11428c → 1.018).

## 3. What went wrong
Overdrive killed grip a 15th/16th time (`r010_09`/`10`: act_max 9.24/9.42, red_frac 1.0,
corr_act_rad −0.30/−0.21, grip ≈0). `rd_interface_tension` inert a 10th time (`r010_11`/`12`).
Same-seed bit-identical bug a 10th round — `r010_05`==ctrl (scored protr>1.20 against the control
it copies, refuted), `r010_01`==`r010_02`. `r010_12` broke P1 (no growth, 2140c). Three execution
losses (`r010_13`/`14`/`15` empty). Five predictions refuted (01/02 protr>1.3, 03 grip>0.10 at
0.09873, 04/05 protr>1.20); 07 confirmed grip>0.071 (0.07161).

## 4. What to do next
The tuning question is now answered: this recipe is the confirmed lobing optimum (protr ~1.19,
grip ~0.095) and no parameter on it makes a finger — `n_tubes` 0 for a 10th round, wall at 1.3
un-breached. Stop re-confirming it. The only remaining move is a mechanical operator that couples
the (now seed-robust) pattern to bending or line tension; a `set_param` sweep will keep landing
inside ±0.01 of control. Also fix the reseed bug so replicate slots stop copying the control.

# Round r011

## 1. What happened
Control `r011_00` is the standing lobing optimum again (protr 1.184, grip 0.0987, 4544c).
`r011_06` reproduced the r009_04 seed exactly (protr 1.198, grip 0.0951, 4149c); `r011_05`
bit-identical to control; `r011_07` protr 1.176 / grip 0.0901 (4216c); `r011_01`==`r011_02`
same-seed pair (protr 1.141, grip 0.0605, 5508c). The unpredicted slots sweep growth:
`r011_13` 2338c → 1.05, `r011_14` 6399c → 1.16, `r011_15` 19135c → **1.241**.

## 2. What was learned
**First mech_p_ratio > 0 of the whole campaign, and it is the only thing that ever pushed
protr past 1.198.** `r011_15` (19135c) hit campaign-max protr **1.241** and n_tubes 2 /
tube_diam 0.833 — but `mech_p_ratio` **1.752** (every other run 0.0) flags a FORCING operator,
not grown mechanics; the eye reads a 5–6-lobed berry and calls the "2 tubes" two bulges. Per the
campaign rule (forced protr answers nothing; grown≈1, forced≈3), this 1.241 is a pushed lobe,
not a finger — aspect_max 0, no neck. Among the UNFORCED runs protr↓growth holds an 11th round:
4149c peaks at 1.198, 9008c overdrive rounds to 1.019.

## 3. What went wrong
`r011_09` chemistry blew up at frame 48 (NaN; P1/P4/P12 broken, act_max→3.8e17) — a static dead
sphere for 850 frames. Overdrive killed grip a 17th time (`r011_08`: act_max_peak 9.46, red_frac
0.979, corr_act_rad −0.142, grip −0.0008, 9008c). Three execution losses (`r011_10`/`11`/`12`
empty). Same-seed bit-identical bug an 11th round (`r011_05`==ctrl; `r011_01`==`r011_02`).
Predictions: 01 protr>1.3, 02 p99>1.4, 05/06 grip>0.10 (0.0987/0.0951) all refuted; 07 act_cv>0.5
confirmed (2.20).

## 4. What to do next
The forcing operator is the round's real signal: mech_p_ratio 1.752 sits between grown (≈1) and
forced (≈3), so `r011_15` is a HALF-forced lobe worth isolating — run it against a
mech_p_ratio≈1 sibling at matched cells to see whether any of the 1.241 is grown. But a pushed
protrusion is not the campaign's answer, so the priority stays a pattern→bending/line-tension
coupling that fingers a tissue the mechanics build themselves. Fix the reseed bug (an 11th round
of copied controls).

# Round r012

## 1. What happened
Control `r012_00`: the standing lobing optimum again — `protr_peak` 1.184, `grip_peak` 0.09873,
4544c, `n_tubes` 0, mech_p_ratio 0. Nothing UNFORCED beat it on either axis.
- **Lobers 01–05 (5089–6459c):** `protr_peak` 1.113–1.163, `grip` 0.055–0.071, all `n_tubes` 0,
  all refuted `protr>1.3` (01/03/04) or `>1.241` (05). Eye: coarsening red patches on a
  scalloped/lobed ball, no finger.
- **`r012_06` FORCED (18717c):** `protr_peak` **1.218**, `mech_p_ratio` **2.011**, `n_tubes` 1,
  `tube_diam` 0.618 — second forcing run of the campaign (after r011_15). refuted `>1.241`; eye
  reads a lobed cauliflower berry, the "tube" a bulge (`aspect_max` 0).
- **`r012_07` growth-off (2001c):** `red_frac` 0, `protr` 1.014, `grip` 0.006 — flat sphere.
- **`r012_11`/`12` non-finite:** chemistry blew up (`act_max` NaN, `act_mean` ±1e26; P1/P4/P12
  broken), then a static dead sphere. protr railed 1.014.
- **`r012_13`/`15` (2966/3152c):** `protr` 1.078/1.076, mildly lobed spheres.
- **Lost: 08/09/10/14 empty.**

## 2. What was learned
Twelfth round, wall stands: no unforced run beats control `protr_peak` 1.184 or `grip_peak`
0.09873, `n_tubes` 0 across every unforced run. **Forcing is the ONLY thing that clears 1.2 again,
and it scales with cell count:** `r012_06` mech_p_ratio 2.011 at 18717c gives protr 1.218 — pushed,
not grown, so it answers nothing (grown≈1, forced≈3). protr↓growth holds among unforced runs:
2001→1.014, 2966→1.078, 3152→1.076, 4544→1.184 (opt), 5089–5532→1.11–1.13, 6459→1.163. Growth-off
decouples pattern from shape a 2nd time (`r012_07` vs r006_07): grip needs a growing tissue.

## 3. What went wrong
The reaction went **non-finite a 2nd AND 3rd time** (`r012_11`/`12`, after r011_09): act_max NaN,
P1/P4/P12 broken — read P12 first, a blow-up masquerades as a sphere in the shape metrics. Four
execution losses (08/09/10/14 empty). No same-seed replicate this round, so seed spread is not
re-bounded (still the r010 measure: protr 1.186±0.011, grip 0.0947±0.004).

## 4. What to do next
Nothing new was learned that a mechanical operator would not settle — twelve rounds of tuning have
not made a finger and the only lever that moves protr past the optimum is FORCING, which is
disqualified. Stop re-confirming the lobing optimum. Isolate `r012_06`'s forcing (mech_p_ratio
2.011): what operator is pushing, and does any grown component survive at matched cells? Else
commit the round to a pattern→bending/line-tension coupling. Guard the reaction against blow-up
(r012_11/12 lost two slots to NaN).

# Round r013

## 1. What happened
Control `r013_00`: the standing lobing optimum — `protr_peak` 1.184, `grip_peak` 0.09873, 4544c,
`n_tubes` 0, mech_p_ratio 0. No unforced run beat it on either axis.
- **Lobers 01/02/04/05 (4386–4883c):** `protr_peak` 1.165–1.193, `grip` 0.084–0.099, `n_tubes` 0;
  all refuted `protr>1.2/1.25/1.3`. `r013_02` (4883c) protr 1.193 / grip 0.099 is the round's high
  but sits inside the r010 seed band (1.186±0.011, 0.0947±0.004) — a tie with control, not a beat.
- **Labyrinth/stripe regime at low growth — new phenomenology.** `r013_03` (2664c; corr_act_rad_peak
  0.791, confirmed `<0.80`), `r013_06` (2000c, division OFF, v_cell doubled 0.24→0.50), `r013_15`
  (3535c): the eye reads a connected red/white maze, not spots. All grip weakly (0.014/0.042/0.046),
  shape stays a sphere (protr ≤1.095). A real Turing labyrinth forms and deforms nothing.
- **Spot-fineness ladder:** `n_spots_final` 16(ctrl)→27(`04`)→36(`05`)→46(`07`), spacing 10.0→6.7→
  6.0→4.9; `r013_07` confirmed `n_spots>25`. Finer pattern grips LESS (grip 0.099→0.096→0.084→0.076).
- **Overdrive `r013_11`/`12` (7903/8443c):** `act_max_peak` 10.36/10.28, `red_frac` 0.98/0.996,
  `corr_act_rad` −0.076/−0.157, `grip` <0. `r013_12` went non-finite (P12 broken).
- **Lost: 08/09/10/14 empty.**

## 2. What was learned
Thirteenth round, wall stands: no run beats control `protr_peak` 1.184 or `grip_peak` 0.09873,
`n_tubes` 0 everywhere, no forced run this round. protr↓growth holds a 13th round: 2000→1.094,
2664→1.029, 3443→1.086, 3535→1.095, 3974→1.165, 4386→1.174, 4544→1.184 (opt), 4883→1.193,
7903/8443→1.017. Optimum shallow ~4544–4883c. Coarser-grips-harder holds: control's 16 spots grip
hardest, the 46-spot `r013_07` weakest. The labyrinth regime is genuinely new pattern topology but
couples to colour only — a striped field grips no better than a spotted one.

## 3. What went wrong
Reaction went non-finite a **4th time** (`r013_12`, P12 broken; after r011_09, r012_11/12) —
read P12 first. Four execution losses (08/09/10/14 empty). No same-seed replicate, so seed spread
stays the r010 bound.

## 4. What to do next
Nothing tuning can settle remains — thirteen rounds, no finger, and the labyrinth regime just added
confirms strong pattern ⊥ shape. Stop re-confirming the optimum and stop sweeping pattern scale
(spots and stripes both grip colour, not surface). Commit the next round to a
pattern→bending/line-tension mechanical operator; it is the only untested route to a protrusion.

# Round r014

## 1. What happened
Control `r014_00` (4883c) = the standing recipe: `protr_peak` 1.193, `grip_peak` 0.099, 16 spots,
lobed sphere, `mech_p_ratio` 0.0. Four regimes moved off it:
- **Single-pole BUD (01,02) — the wall finally broke.** `protr_peak` **1.453**/**1.420** (prior
  unforced ceiling 1.184, prior forced 1.241), `grip_peak` 0.139/0.130, `protrusion_aspect_max_peak`
  1.331/1.458, `n_tips` 2, `gyr_prolate` 2.00/2.06, at LOW growth (2286/2325c). Chemistry collapses
  to ONE pole (`n_spots_final` 1/2, `red_frac` 0.137/0.151, `act_cv_peak` 9.65/7.67). BUT
  `mech_p_ratio` **1.958**/**1.783** — first nonzero since the r011/r012 forced lobes — so the bud
  carries a FORCING operator. Eye: a single rounded bud with a constricting neck, "budding-off, not
  a finger"; the `n_tubes_peak` 1 / `tube_diam` 2.05 are the whole-body prolate, not a tube.
- **Grown lober (03).** `protr` 1.204, `grip` 0.115, `mech_p_ratio` 0.0, 4001c — a scalloped star,
  ties control regime, no finger.
- **Overdrive (04,09,10).** 04 held its pattern (act_cv 2.21, refuted act_cv<0.05). 09/10: `act_max`
  8.38/9.04, `red_frac` 0.996/0.999, `corr_act_rad` floor −0.65/−0.66, `grip` ~0, spheres (9021/8737c).
- **Dead-activator (05,06,07).** P4 broken, red extinct; `grip` 0.002–0.008, spheres. The eye flags
  a persistent blue/yellow SECOND field coarsening to the end in all three that no metric records.
- **Lost: 08,11,12,13 empty** (11 eye-only). Four execution losses.

## 2. What was learned
The bud (01,02) is the first protrusion in fourteen rounds with `aspect_max`>1.3 and a visible neck
— morphologically Okuda's Fig-5b budding case, not a lobe. It replicates across the two runs
(protr 1.45/1.42, grip 0.14/0.13, aspect 1.33/1.46, mech_p_ratio 1.96/1.78 — tight). But
`mech_p_ratio` ~1.9 (grown≈1, forced≈3) says it is roughly half PUSHED, so it does not yet answer
the open problem: the campaign's own rule is that a forced protrusion is not a grown finger. What is
new versus the r011/r012 forced lobes (aspect 0, ~19000c) is that forcing at LOW growth with
chemistry collapsed to one pole yields a bud SHAPE, not a berry. Overdrive kills grip — 20th/21st
replication (09/10). Coarser-grips-harder taken to its limit: `n_spots_final` 1 → grip 0.139, the
round's hardest grip.

## 3. What went wrong
Four execution losses (08/11/12/13). The dead-activator second field (05/06/07) is the field-read
gap again — a patterned channel the eye sees and act_* reads as extinct. No same-seed replicate, so
seed spread stays the r010 bound.

## 4. What to do next
- **Decide grown vs pushed on the bud.** Dial the forcing gain on the 01/02 recipe DOWN in steps:
  if `protr_peak` stays >1.3 as `mech_p_ratio`→1 it is grown (an answer); if it collapses toward
  1.2 it was pushed. This is the decisive test and it is one edit.
- **Replicate 01/02 at a fresh seed** to bound the bud, and turn division ON — does a growing
  single-pole bud elongate into a finger, or does growth re-round it (protr↓growth)?

# Round r015

## 1. What happened
Control `r015_00` = the r014 bud recipe, forced: `protr_peak` 1.453, `grip` 0.139, `mech_p_ratio`
1.958, one pole, 2286c. This round is the r014 decisive test — a FORCING-GAIN LADDER down the bud
recipe. Ordered by `mech_p_ratio` → `protr_peak` / grip:
- 1.958 → 1.453 / 0.139 (ctrl, and `r015_04` bit-identical to it)
- 1.532 → 1.353 / 0.110 (`r015_07`, one-sided bud, 2539c)
- 1.439 → 1.346 / 0.152 (`r015_03`, 5–6 fat lobes, n_spots 6, 3089c)
- 1.348 → 1.276 / 0.090 (`r015_06`, single bud, 2397c)
- 0.0 → 1.085 / 0.042 (`r015_14`, UNFORCED, 3131c, mildly lumpy sphere, aspect 0, n_tips 0)
- 0.0 → 1.017/1.018 (`r015_09`/`15`, unforced overdrive spheres, 9179/8702c)

`r015_01`==`r015_02` bit-identical (protr 1.004, P4 broken, chemistry extinct, 2019c). `r015_08`
reaction blew up (act_max 5.7e17, NaN, P1/P4/P12 broken, sphere). Four losses (10/11/12/13 empty).

## 2. What was learned
**The r014 bud was PUSHED, not grown — the r014 "wall break" is a forcing artifact.** protr is
MONOTONE in `mech_p_ratio`: dial the gain to zero and the bud collapses to a lumpy sphere (1.453 →
1.085), aspect 1.331 → 0, no neck. The decisive test the r014 note posed is answered NEGATIVE —
protr does not survive as `mech_p_ratio`→1. No unforced run beats protr 1.085; the 1.3 GROWN wall
still stands after fifteen rounds, `n_tubes` 0 everywhere.
Forcing needs a live pattern to gate on: `r015_01`/`02` push a dead field → protr 1.004 (flat).
Round's best grip `r015_03` 0.152 (>ctrl 0.139) is moderate-forcing MULTI-lobe (n_spots 6) — still
forced, still no finger; the extra lobes buy grip but not a tube. Overdrive kills grip — 22nd/23rd
replication (`r015_09`/`15`, corr −0.226/−0.189, grip <0).

## 3. What went wrong
`r015_04` bit-identical to control (same-seed replicate bug, 12th round) — its aspect<1.331
prediction tied at exactly 1.331 and refused. `r015_08` non-finite reaction a 5th time (read P12
first). Four execution losses.

## 4. What to do next
- **Retire the forced-bud line.** protr from `extrude`-class forcing is settled: linear in gain,
  zero at gain 0. Stop spending slots pushing it past 1.3 — the campaign rule already voids it.
- **Return to the grown regime with a NEW lever.** Fifteen rounds of division+chemistry cap grown
  protr at ~1.18. Pattern topology (r013 stripes), spot coarseness, growth rate all tried. The
  missing ingredient is anisotropy — a line-tension or bending term co-located with the pole that
  could thin a lobe into a finger without a radial push. Test `shape_energy`/line-tension gated on
  the activator at the single-pole recipe, `mech_p_ratio` held at 0.
- **Bound `r015_03` at a fresh seed** — is grip 0.152 real or the same single-seed luck that has
  regressed elsewhere?

# Round r016

## 1. What happened
Control `r016_00` = the r015_03 forced multi-lobe recipe: `protr_peak` 1.346, `grip` 0.15244,
`mech_p_ratio` 1.439, n_spots 6, 3089c. The round's job was the r015 recommendation — bound that
0.152 at a FRESH seed — and it collapsed to the replicate bug at its worst yet. Distinct results:
- FORCED control (1.346 / 0.152) — and `r016_03` **bit-identical** to it (`mech_p_ratio` 1.439,
  protr 1.346, grip 0.15244). Its "grip>0.152 confirmed" scores against the control it copies.
- one UNFORCED composition (`mech_p_ratio` 0, 4001c, protr 1.204, grip 0.11538, n_spots 18) run
  **four times at one seed** — `r016_01`==`r016_02`==`r016_05`==`r016_07`, all bit-identical.
- `r016_12` unforced lober: protr 1.089, grip 0.0417, 2918c, n_spots 23.
- `r016_15` unforced fine field: protr 1.083, grip 0.0184, 3148c, **n_spots 166**, roundest.
- `r016_11`/`r016_13` reaction blew up (act_max 1.3e30 / 1.7e23 → NaN, P1/P4/P12 broken, sphere).
- Four losses (`r016_08`/`09`/`10`/`14` empty).

## 2. What was learned
**The central question went UNANSWERED — the forced grip 0.152 is still n=1.** `r016_03` failed to
reseed (bit-identical to control), so the r015_03 point has no fresh-seed replicate; do not read
its "confirmed" as replication. The wall stands a **16th round**: no UNFORCED run beats protr 1.204
or grip 0.115, `n_tubes` 0 everywhere unforced, the only run above that is the forced control.
**Coarser-grips-harder holds a 14th round**, cleanly across three unforced runs at one growth
scale: n_spots_final 18/23/166 → grip 0.115/0.042/0.018, and the 166-spot fine field is also the
roundest (protr 1.083). The 4001c/18-spot run also has the round's highest unforced protr (1.204):
coarseness, not cell count, sets it (2918c/23sp→1.089, 3148c/166sp→1.083, 4001c/18sp→1.204).

## 3. What went wrong
Same-seed replicate bug a **13th round and the worst on file** — FIVE slots collapsed into TWO
results (01/02/05/07 identical; 03==ctrl). Of twelve non-empty slots the round bought two distinct
science points. Reaction non-finite a **6th and 7th time** (`r016_11`/`13`; read P12 first — both
read as intact spheres in the shape metrics). Four execution losses.

## 4. What to do next
- **Re-issue the r015_03 fresh-seed bound with the reseed actually verified** before launch — the
  campaign's one open question (is forced grip 0.152 real?) has now cost a round to the seed bug.
- **The grown line still needs a NEW lever, not another seed of the same recipe.** As r015 said:
  anisotropic line-tension / bending gated on the single pole, `mech_p_ratio` held at 0 — nothing
  tried this round tests it. Every unforced composition here is a spot-coarseness variant and all
  sit under grip 0.115.
- **Fix or route around the replicate bug** — a round that yields two points from twelve slots is
  not worth launching.

# Round r017

## 1. What happened
Control `r017_00` is the FORCED recipe again (`mech_p_ratio` 1.439, protr 1.346, grip 0.152, 3089c,
n_spots 6). The round splits cleanly by forcing. **Forced slots set new campaign maxima on both
axes and still make no finger:** `r017_07` (2486c, `mech_p_ratio` 2.284) reached protr_peak **1.588**
and grip_peak **0.198** — the highest either has ever been — with `gyr_prolate` 2.253, aspect_max
1.748, n_tubes 3; the eye sees two fat red-tipped buds on a body whose cross-section stays a clean
circle. `r017_06` (1.958) reproduces `r014_01` at protr 1.453; `r017_15` (2.025) 1.268. **Unforced
slots (`mech_p_ratio` 0) all sit under the standing wall:** `r017_02` protr 1.204/grip 0.115 (4001c,
18 spots) ties the r016_01 ceiling; `r017_03`/`05`/`13` land 1.19/1.15/1.074.

## 2. What was learned
Forcing stays **linear past protr 1.5** — mech_p_ratio 2.284/2.025/1.958 → protr 1.588/1.268/1.453,
and grip rides with it to 0.198. `gyr_prolate>2` (r017_07 confirmed) is the whole body stretching
under the push, not a tube: cross-section round, aspect on a bud. So the r014/r015 verdict stands —
protr and grip past the wall are bought with forcing, not growth. The **grown wall stands a 17th
round**: no unforced run beats protr 1.204 or grip 0.115, n_tubes 0 unforced. **Coarser-grips-harder
holds a 15th round** (unforced n_spots 18/41/40/83 → grip 0.115/0.097/0.071/0.028).

## 3. What went wrong
`r017_04` P4 broken — chemistry extinct, grip 7e-05, a dead-field specimen (its grip<0.115 is
trivially confirmed). Overdrive killed grip a **24th/25th** time: `r017_11` uniform-saturated
(act_cv 0.007, red_frac 1.0), `r017_12` (corr_act_rad −0.228, grip −0.001). Four execution losses
(`r017_08`/`09`/`10`/`14` empty). No same-seed replicate this round — the r010 seed bound still
holds; the forced grip point remains n=1 across seeds.

## 4. What to do next
- **The forced line is exhausted — retire it.** Three rounds now show protr/grip past the wall are
  linear in forcing gain and produce round-sectioned buds, not fingers. Another gain rung buys a
  bigger push, no new mechanism.
- **The grown line still lacks a NEW lever.** Anisotropic line-tension or bending gated on the pole,
  `mech_p_ratio` held at 0 — still untried; every unforced slot this round is a spot-coarseness
  variant under grip 0.115.
- **Verify the reseed before any n=1 forced point is re-posed** — the seed bug has cost this
  question two rounds already.

# Round r018

## 1. What happened
Control `r018_00` is a NEW recipe (1801 frames, 7424c) and a step change on both axes: `protr_peak`
**1.513**, `grip_peak` **0.228**, `n_tubes_peak` **7**, `corr_act_rad_peak` 0.944, `act_cv_peak`
4.34, `mech_p_ratio` 1.658. Both records beat every prior FORCED max (1.588/0.198) and dwarf the
prior unforced wall (1.204/0.115). The eye — across three bit-identical views (02/04/07) — calls it
a genuine 4–7 armed star of red-tipped fingers, "grown not pushed"; the first specimen in 18 rounds
it reads as real fingers rather than buds or lobes. Variants off it, all worse on grip:
- **`r018_05`==`06`** (seed pair, 3750c): chemistry to one pole → a single elongated finger,
  `gyr_prolate_peak` **7.86**, `protr_peak` 1.543 (round max), `n_tubes_final` 1 — eye "closest to
  a genuine tube" — but grip 0.187 and `mech_p_ratio` 2.103 (more forced).
- **`r018_14`**: bilobed bud, 3059c, `mech_p_ratio` 2.337 (most forced), protr 1.349, grip 0.117.
- **`r018_15`**: multi-lobed undulating ball, 5609c, `mech_p_ratio` 1.74, protr 1.276, grip 0.126.

## 2. What was learned
The multi-arm control grips HARDER and forces LESS than every concentrated variant (0.228/mp1.658
vs single-finger 0.187/2.103, lobers 0.117–0.126/1.74–2.34). **protr↓growth REVERSES** in this
recipe: the largest tissue (7424c) makes the strongest grip; concentrating to fewer poles at lower
growth trades multi-arm grip for one longer, more-forced finger. `mech_p_ratio` 1.658 (nearer
grown≈1 than forced≈3) plus the eye's "grown" verdict make this the strongest grown-signature
protrusion on file — but forcing is present (mp≠0), so a gain ablation is owed before calling it
purely grown.

## 3. What went wrong
Replicate bug a **14th round**: 02/04/07 bit-identical to control, so their confirmed predictions
(act_cv>3, grip>0.16, spacing>12) are self-comparisons, worthless; 05==06 likewise. Reaction
non-finite an **8th time** (`r018_09`, P4+P12, act_max→6.8e8→NaN at 15989c overgrowth, static
sphere). Chemistry extinct (`r018_13`, P4, flashes once then dies, 2051c no-growth sphere). Four
execution losses (`r018_08`/`10`/`11`/`12` empty). The replicate bug denied a seed bound on the
campaign's best-ever point.

## 4. What to do next
- **Reseed the control at fresh seeds.** It is the strongest point the campaign has produced and it
  is n=1 across seeds — the 02/04/07 "replicates" were bit-identical.
- **Forcing-gain ablation on the control** (mp 1.658→0): do the 7 arms survive as `mech_p_ratio`→1?
  The decisive grown-vs-forced test, as the r015 ladder settled the bud.
- **Bracket growth above 7424c** — protr↓growth reversed, so test whether more growth still buys
  more grip or there is an optimum past the control.

# Round r019

## 1. What happened
Control `r019_00` repeats the r018 base (7424c): `protr_peak` **1.513**, `grip_peak` **0.228**,
`n_tubes_peak` 7, `mech_p_ratio` 1.658. Two lines moved off it.
- **`r019_07` is the campaign's strongest protrusion on record.** `protr_peak` **1.817**,
  `grip_peak` **0.294**, `gyr_prolate_peak` **5.857**, `protrusion_aspect_max_peak` **11.42**,
  `n_tubes` 2, at 4368c with chemistry concentrated to one pole (`n_spots_final` 10,
  `reduced_volume` 0.625). The eye: "Red-tipped bud elongates into a tapered finger — first real
  protrusion... reads as grown, not merely bulged." BUT `mech_p_ratio` **2.019** (grown≈1,
  forced≈3) — forcing is present, ~half. Same concentrate-to-one-pole move as r018_05/06 but
  stronger on every axis. Refuted its own `grip<0.20` (0.294).
- **`r019_03`** (7491c): a finer-spot variant, `n_spots_final` 42 vs control's 22 → `grip` 0.190 <
  0.228, `mech_p_ratio` 1.704. Refuted `grip>0.228`.
- **Lobers `r019_14`/`15`** (4488/5659c): `mech_p_ratio` 1.864/1.65, protr 1.257/1.243, grip
  0.100/0.104 — bulges over spots, eye sees no finger; their `n_tubes` 1–2 are spurious.

## 2. What was learned
- **First UNFORCED seed bound in the new recipe.** `r019_08`/`09` are one composition at two seeds
  (`mech_p_ratio` **0** both, `rd_interface_tension` inert both, ~6.5k c): protr **1.194±0.015**,
  grip **0.0865±0.0005**, `n_spots` 75/72 — an undulating lobed ball, no finger. A difference under
  ±0.015 protr / ±0.0005 grip is seed noise, not signal.
- **`rd_interface_tension` is INERT** — null on `r019_08`/`09`/`15` (3 runs), reconfirming r001/r002.
  Stop proposing it.
- **Grip needs a growing tissue — reconfirmed twice.** `r019_01` (rho→0, 2000c, no growth): protr
  1.003, grip 0.0003, sphere. `r019_06` (P4 broken, activator extinct, 2174c): grip 0.00026, sphere.
- **Coarser-grips-harder holds** in the new recipe: 10-spot 0.294 (forced) / 22-spot 0.228 (ctrl) /
  42-spot 0.190 (`r019_03`).

## 3. What went wrong
Replicate bug a **15th round**: `r019_02`/`04`/`05` bit-identical to control (protr 1.513) — their
predictions are self-comparisons (02's `gyr>2.5` refuted against ctrl gyr 1.533; 04's `protr>1.513`;
05's `protr<1.20`). The control STILL has no fresh-seed bound. Four execution losses
(`r019_10`/`11`/`12`/`13` empty).

## 4. What to do next
- **Forcing-gain ablation on `r019_07`** (mp 2.019→0): does the tapered finger survive as
  `mech_p_ratio`→1? The r015 bud went linear in gain and collapsed to a lump at gain 0 — this is the
  decisive grown-vs-forced test, and 1.817/0.294 is worth it only if it survives.
- **Reseed the control** — owed since r018, still n=1 across seeds.
- **Concentrate-to-one-pole is the lever past the multi-arm wall**, but every strong point (r019_07,
  r018_05/06) carries mp>2. Test whether ANY concentrated finger holds protr>1.5 at mp near 1.

# Round r020

## 1. What happened
Control `r020_00` again repeats the r018/r019 base (7424c): protr_peak 1.513, grip_peak 0.228,
n_tubes 7, mech_p_ratio 1.658. Two clusters moved off it:
- **Multi-arm forced stars (02/03/04/06).** Eye across all four: 4–8 red-tipped fingers, "grown not
  pushed." grip 0.216–0.2617, protr 1.408–1.607, mp 1.68–1.77, protrusion_aspect_max 3.3–4.87.
  `r020_06` is round-max grip **0.2617** (>ctrl 0.228; n_spots_final 11, gyr_prolate 3.008);
  `r020_03` grip 0.240 (gyr_prolate 2.925). Both confirmed grip>0.228. `r020_04` largest tissue
  (8259c) yet protr 1.408, grip 0.216 — most cells ≠ most grip.
- **Unforced/low lobers.** `r020_12`/`14` (mp **0**): protr 1.184/1.190, grip 0.089/0.0915,
  n_spots 83/93 — lobed balls, no finger. `r020_08` (4498c, mp 1.671) protr 1.229 faceted blob;
  `r020_13` (n_spots 179, mp 1.933) protr 1.215 raspberry.
- Replicates 05/07 bit-identical to ctrl. Losses 09/10/11/15 empty.

## 2. What was learned
- **Unforced seed bound reconfirmed a 3rd time.** `r020_12`/`14` (mp 0): protr 1.184/1.190, grip
  0.089/0.0915 sit inside r019_08/09's 1.194±0.015 / 0.0865±0.0005. The unforced wall stands: no
  mp-0 run beats protr ~1.19 / grip ~0.09, n_tubes 0.
- **Round-max grip beats control but is still forced.** `r020_06` 0.2617 at mp 1.69; the whole
  multi-arm cluster is mp 1.68–1.77. The eye calls them grown fingers, mp says ~half-forced — the
  same unresolved conflict as r018/r019, and the gain ablation is still owed.
- **Coarser-grips-harder holds:** n_spots_final 11/14/83/93/179 → grip
  0.2617/0.240/0.089/0.0915/0.080 (06/03/12/14/13).
- **`rd_interface_tension` INERT** on 08/12/14, reconfirming r001/r002/r019. Stop proposing it.

## 3. What went wrong
Replicate bug a **16th round**: `r020_05`==`r020_07`==ctrl (protr 1.513) — "grip>0.15"/"protr>1.513"
are self-comparisons. Control STILL n=1 across seeds, owed since r018. n_tubes noisy: eye disputes
the single "tube" on 08/13, calling it one lobe of a blob. Four execution losses (09/10/11/15 empty).

## 4. What to do next
- **Forcing-gain ablation on `r020_06`** (mp 1.69→0): does the multi-arm star survive as
  mech_p_ratio→1? Decisive grown-vs-forced test, owed three rounds; 0.2617/1.595 counts only if it
  survives.
- **Reseed the control** — n=1 across seeds since r018.
- Retire rd_interface_tension and stop re-running the base without a gain ladder.

# Round r021

## 1. What happened
Control `r021_00` is the `r020_06` FORCED multi-arm recipe (`mech_p_ratio` 1.69, `protr_peak`
1.595, `grip_peak` 0.2617, 6143c) — the round is at last a (bimodal) forcing ablation on it, plus
two rho=0/extinct ablations. Splits by `mech_p_ratio`:
- **FORCED (mp 1.6–2.0):** `r021_05` (3782c, mp 1.635) protr **1.652** round-max, grip 0.217;
  `r021_03` (7200c, mp 1.694) protr 1.417, grip 0.202, eye "5–7 red-tipped fingers"; `r021_04`
  (5773c, mp 2.023) protr 1.438 but **P11 fold**; `r021_11`/`r021_15` (7106/6091c) protr
  1.330/1.244, grip 0.103/0.120.
- **UNFORCED (mp 0):** `r021_13` (6986c) protr 1.164, grip 0.077, n_spots 29; `r021_02` (2000c,
  no-division, 3 domains) protr **1.257**, grip 0.080; `r021_12` (2794c, one pole) protr 1.197,
  grip −0.0015 (`corr_act_rad_final` −0.012).
- **rho=0 / extinct (mp 0, P4 broken):** `r021_01` grip 0.0003, `r021_07` grip 0.0001 — spheres.
- **Lost: 08/09/10/14 empty.**

## 2. What was learned
The owed ablation, answered bimodally. At MATCHED growth (~7k c) removing forcing drops the star
from protr 1.595 / grip 0.2617 (ctrl) to **1.164 / 0.077** (`r021_13`, mp 0) — a lobed ball, no
finger; every "fingers" verdict the eye gives (`r021_03`) is on an mp ≥ 1.65 run. Forcing carries
~0.43 protr and ~0.18 grip; the multi-arm star is forcing-dependent, reconfirming r015/r020 a 4th
way. The unforced protr wall nudges to **1.257** (`r021_02`) — but that is a 2000c no-division
3-domain BULGE (grip 0.080, `protrusion_aspect_max` 0, `n_tubes` 0), not a finger. Over-forcing
BUCKLES: `r021_04` mp 2.023 → P11 fold. protr is NOT monotone in mp across these runs
(`r021_05` at the lowest forced mp 1.635 has the highest protr 1.652) because growth co-varies —
protr↓growth holds within the forced set (3782c→1.652 > 7200c→1.417). Round-max protr 1.652 is a
BUD: `protrusion_aspect_max_peak` 12.034 is a single-frame spike and the eye reads 2–3 rounded
lobes, not fingers.

## 3. What went wrong
Grip needs a growing tissue — reconfirmed: rho→0 `r021_01` (grip 0.0003, P4) and extinct `r021_07`
(grip 0.0001, P4) are spheres. Coarser-grips-harder holds: `n_spots_final` 11/6/25/77 → grip
0.2617/0.217/0.120/0.103 (ctrl/`r021_05`/`r021_15`/`r021_11`). No replicate bug this round —
predictions are genuine (6 refuted, 1 confirmed: `r021_06` grip<0.10 at 0.093). Four execution
losses (08/09/10/14 empty).

## 4. What to do next
The forcing verdict is now settled a FOURTH way (r015 gain ladder, r018/19/20 mp, r021
matched-growth ablation): the star's protr/grip are forcing-carried; unforced this recipe makes a
≤1.26 lobed ball. Stop ablating — the answer will not move. The grown line has sat at protr ~1.2
for 21 rounds under division+chemistry+forcing; the untried lever remains **anisotropic
line-tension / bending gated on the pole at mp 0**, owed since r015 and never once run. Second:
turn DIVISION on over the single-pole unforced bulge (`r021_02`, 2000c) — does it elongate or
re-round? — the one growth×concentration corner untested unforced.

# Round r022

## 1. What happened
Control `r022_00` is again the FORCED multi-arm star (mp 1.69, protr_peak 1.595, grip_peak 0.2617,
6143c). The round splits three ways:
- **Forced growth-ladder (01–07, mp 1.49–1.93):** grip and protr fall monotone as the tissue grows.
  cells→protr/grip: 6143→1.595/0.262 (ctrl) > 6515→1.540/0.238 (`_05`) ≈ 6744→1.464/0.243 (`_01`) >
  7929→1.434/0.229 (`_02`) > 7991→1.415/0.207 (`_07`) > 8259→1.408/0.216 (`_04`). `_03` (6003c, mp
  1.494) protr 1.601 (round-max, ties ctrl) but grip only 0.180. Eye reads all as grown red-tipped
  stars/lobes.
- **NEW fine near-uniform regime (09/10/11):** reduced_volume 0.80–0.82 (ctrl 0.48), act_max 0.73–0.83
  (ctrl 0.589), act_cv 0.67–0.80 (ctrl 4.49), n_spots 73/70/120, grip ~0.08. `_09` mp 0, `_10`/`_11`
  mp 1.68/1.615. `_11` = 10708c, largest tissue on file. Eye: lumpy multi-lobed inflating ball, no
  finger/tube on all three.
- **Dead Gray-Scott field (`_15`):** activator extinct (act_mean 0.018, act_cv 4.83, corr_act_rad
  −0.25), shape a clean sphere (protr 1.03, grip −0.004, 2903c). Eye: two-phase blue/yellow
  segregation coarsens but never deforms the surface.
- **Lost: 08/12/13/14 empty.** `_06` bit-identical to ctrl (replicate bug, 17th round).

## 2. What was learned
The lever is now bracketed on both axes of the field. **Along growth**, the forced star's protr/grip
decline monotone with cell count (0.2617→0.216 over 6143→8259c); ctrl at the LOWEST growth is the
max-grip point, which is why every "beat ctrl" prediction refuted (01/02 grip, 03 protr, 04 oblate).
**Along field coarseness**, forcing carries grip ONLY on a coarse field: `_10`/`_11` are FORCED
(mp 1.6) yet on a 70–120-spot fine field give grip 0.087/0.087 ≈ the UNFORCED `_09` 0.079 — the ball
inflates to reduced_volume 0.80 instead of fingering. So the r021 "forcing carries ~0.18 grip" is
conditional: it needs a coarse (n_spots ~11) field to act on. Coarser-grips-harder is now the same
statement as forcing-works. `_15` reconfirms a dead activator grips nothing however the second field
segregates.

## 3. What went wrong
No premises broke anywhere — a mechanically clean round. `_06`==ctrl (17th replicate-bug round); its
"protr>1.5 confirmed" is a self-comparison. Only genuine confirmations: `_05` gyr_oblate 0.263>0.253,
`_07` grip 0.207>0.20. Four execution losses (08/12/13/14). Grown-protrusion wall stands a 22nd round:
best unforced `_09` protr 1.177 / grip 0.079, n_tubes 0.

## 4. What to do next
The (growth × coarseness) map of the forced star is now filled: max grip is low-growth + coarse
(ctrl), and forcing is inert on a fine field. Stop sweeping this recipe — both axes only lower grip.
The fine regime (09/10/11) is a genuinely NEW chemistry setting the campaign had not run — near-uniform
100-spot pattern, reduced_volume 0.80 — and it inflates to the largest tissue on file WITHOUT
fingering; if a coarse forced star and a fine inflating ball are the two poles, the untried middle is
a COARSE field at HIGH growth held together (the anisotropic line-tension/bending gate owed since
r015, never run). Second: seed the Gray-Scott field (`_15`) so its two-phase pattern lives — a live
segregating field has never been coupled to the mechanics.

# Round r023

## 1. What happened
Control `r023_00` is the FORCED multi-arm star yet again (mp 1.69, protr_peak 1.595, grip_peak
0.2617, 6143c, n_spots_final 11) — the 4th consecutive round on this one recipe. The round is a
(growth × coarseness) re-sweep of it, already bracketed in r022:
- **Coarse growth variant (`_02`, 7548c, mp 1.683):** protr 1.432, grip 0.228, n_spots_final 14,
  `protrusion_aspect_max_peak` **3.327**, `n_tips_peak` 8, `n_tubes_peak` 8. The eye's BEST read
  of the round — "6–8 red-tipped fingers with narrow necks, grown not pushed." Confirmed
  corr_act_rad_peak 0.917>0.7. Falls exactly on the r022 growth line (6143→0.262 > 7548→0.228).
- **Fine-field lobers (`_15`/`_14`/`_08`, 8627/7937/10691c, mp 1.61/1.76/1.37):** n_spots_final
  58/68/91 → grip 0.1153/0.0975/0.074, reduced_volume 0.66/0.72/0.84. Eye: 5–6-lobed balls / an
  inflating tri-lobed blob, no finger on any.
- **Unforced STRIPE field (`_13`, mp 0, 4228c):** protr 1.026, grip 0.0058, a clean sphere top to
  bottom; the chemistry coarsens into a labyrinthine worm-stripe maze that never touches the shape.
- **`_06`==`_07`==ctrl** bit-identical (replicate bug, 18th round). **Lost: 09/10/11/12 empty.**

## 2. What was learned
Nothing the r022 map did not already hold — this round re-fills it. **Coarser-grips-harder is the
cleanest 5-point ladder on file, all in one round:** n_spots_final 11/14/58/68/91 → grip
0.2617/0.228/0.1153/0.0975/0.074 (ctrl/`_02`/`_15`/`_14`/`_08`), monotone. **Growth dilutes on a
coarse field** (ctrl 6143→0.262 vs `_02` 7548→0.228, both n_spots ~11–14), reproducing r022 exactly.
`_02` is the round's most finger-like specimen (aspect 3.327, 8 tips, eye "grown fingers") yet
`mech_p_ratio` 1.683 = FORCED and grip 0.228 < ctrl 0.262 — forcing-carried, beats nothing, no new
mechanism. **The one unforced run is a stripe sphere:** `_13` grip 0.0058 reconfirms grip needs
forcing and adds that stripe (vs spot) topology grips nothing either — the lowest grip of the round.

## 3. What went wrong
A mechanically clean round — no premise broke, no overdrive, no blow-up. But it bought no new science:
four rounds now (r020–r023) have swept the forced-star recipe along growth and coarseness and every
value only lowers grip below the ctrl. Replicate bug an **18th round** (`_06`==`_07`==ctrl); both
posed grip>0.262 and scored refuted against the control they copy (grip 0.2617 exactly). Four
execution losses (09/10/11/12). The grown-protrusion wall stands a **23rd round**: the only unforced
run is a 1.026 stripe sphere, no unforced run near the wall, n_tubes 0 unforced.

## 4. What to do next
- **STOP re-running the forced star.** Its (growth × coarseness) map is filled twice over; max grip
  is ctrl (low-growth + coarse), every other cell of the map is lower. Another sweep is a wasted round.
- **The owed lever is STILL untried, 8 rounds on:** anisotropic line-tension / bending gated on the
  pole at `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never once launched.
- **Seed the unforced stripe field (`_13`) into a live spot field and turn division on** — a coarse
  UNFORCED spot pattern on a growing tissue is the one corner never run; every unforced run to date
  is either a stripe, a fine lober, or a no-growth bulge.

# Round r024

## 1. What happened
Control `r024_00` is the FORCED multi-arm star a 5th round (mp 1.69, protr_peak 1.595, grip_peak
0.2617, 6143c, n_spots 11). This round finally varied FORCING MAGNITUDE at matched growth and swept a
new high-act chemistry regime:
- **Forcing ladder at ~7k c (`_05`/`_02`/`_06`/`_07`, mp 1.303/1.672/1.942/1.932):** grip
  0.214/0.204/0.218/0.243, protr 1.34/1.433/1.397/1.464. Flat in grip across a 1.30→1.94 mp span.
- **Forced high-act/fine regime (`_14`/`_15`, mp 1.844/1.941):** act_max 0.99/0.84, act_cv 0.86/1.52,
  n_spots 29/7 → grip 0.093/0.072. High forcing, low grip — inflates/lobes, no finger.
- **Unforced lobers (`_11`/`_13`, mp 0, 5784/6395c):** protr 1.105/1.137, grip 0.0465/0.056, n_spots
  26/37 — bumpy balls, same high-act regime.
- **Unforced bulge (`_03`, mp 0, 2000c no-div):** protr 1.222 (round's best unforced), grip 0.074, 2
  domains — a lumpy sphere. **Ablations (`_04` 2000c grip 0.00067; `_01` P4 extinct grip 5e-05):**
  spheres. **Lost: 08/09/10/12 empty.**

## 2. What was learned
**Forcing magnitude is not the dial — it saturates.** At matched 6.7–7.1k growth, mp 1.303→1.942
leaves grip on a plateau 0.204–0.243 (`_05`/`_02`/`_06`/`_07`); forcing is an on/off GATE (mp 0 →
grip <0.08) but pushing harder than ~1.3 adds nothing. Grip is set by field coarseness × growth, as
r022/r023 held — the new fact is that the third axis (forcing strength) is flat. **Forcing still needs
a coarse field:** `_14`/`_15` carry ctrl-level forcing (mp 1.84/1.94) on the new high-act near-uniform
field (act_max 0.92–0.99, act_cv 0.68–0.86) and grip only 0.072–0.093 — a third of ctrl. That regime
inflates and lobes, never fingers. The eye's "first genuine multi-finger" (`_07`, aspect 4.332, 6
tubes) is FORCED and grip 0.243 < ctrl 0.262: forcing-carried, no new mechanism.

## 3. What went wrong
Mechanically clean — one premise break (`_01` P4, an intended extinction ablation), no overdrive, no
blow-up. But no new science beyond nailing the forcing-magnitude plateau: this is the recipe's 5th
consecutive round and every variant sits at or below ctrl grip. `_06`'s aspect>5 prediction refuted
(1.617 — fat lobes). Grown-protrusion wall stands a **24th round**: best unforced is `_03`'s 1.222
2-domain bulge, n_tubes 0 on every unforced run. Four execution losses (08/09/10/12). No replicate bug.

## 4. What to do next
- **STOP the forced star for good.** Its (growth × coarseness × FORCING) map is now filled in all
  three axes; ctrl (low-growth, coarse) is the max and no cell beats it. Any further sweep is wasted.
- **The owed lever is STILL untried, 9 rounds on:** anisotropic line-tension / bending gated on the
  pole at `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never launched.
- **Grow an UNFORCED coarse SPOT field:** every unforced run is a stripe, a fine lober, or a
  no-growth bulge (`_03`); a coarse spot pattern on dividing tissue at mp 0 is the corner never run.

# Round r025

## 1. What happened
Control `r025_00_ctrl` is the FORCED multi-arm star a **6th** round (mp 1.69, protr_peak 1.595,
grip_peak 0.2617, 6143c, n_spots 11) — eye: "4–5 red-tipped arms, genuine star." Nothing this round
left the map already filled r022–r024:
- **Forced star cluster (`_03`/`_04`/`_05`, mp 1.748/1.77/1.761):** protr 1.607/1.408/1.54, grip
  0.240/0.216/0.238, cells 6424/8259/6515, n_spots 14/20/15. All below ctrl grip.
- **Two bit-identical replicate copies:** `r025_02` AND `r025_06` == ctrl to every digit (protr
  1.595, grip 0.2617). Their "confirmed" act_cv>1 / grip>0.1 are self-comparisons.
- **Unforced lobers (`_11`/`_15`, mp 0):** protr 1.115/1.17, grip 0.0527/0.065, n_spots_final
  37/106, cells 4388/7905, reduced_volume 0.89/0.83 — bumpy inflated balls, no finger.
- **Ablations, P1 broken:** `_13` rho→0 (2005c, no growth, protr 1.012, grip 0.0035, reticulated
  maze on a rigid sphere); `_14` (2028c, 40% volume loss, v_cell 0.14, protr 1.092, grip 0.0088).
- **Lost: 08/09/10/12 empty.**

## 2. What was learned
**Nothing new — a 6th confirmatory round on a closed map.** Forced coarse cluster grips 0.216–0.240,
all < ctrl 0.2617; growth dilutes as before (`_04` 8259c is lowest grip 0.216). Coarser-grips-harder
holds: n_spots_final 14/15/20 (forced) → grip 0.240/0.238/0.216 vs unforced 37/106 → 0.0527/0.065.
The eye again reads the forced stars (`_04`/`_05`/`_06`) as "genuine grown fingers," but every one is
mp ~1.75 — forcing-carried, no mechanism. **Grip needs a growing tissue** reconfirmed: `_13`/`_14`
are non-growing/shrinking spheres at grip <0.01. spot_spacing_cells_peak refuted (`_03` 19.61 vs >21)
— ~20 cells is the coarse-field spacing ceiling.

## 3. What went wrong
The replicate bug returns a **19th round** — TWO copies this time (`_02`, `_06`). `_14` is a
diagnostic: P1's 40% volume loss (485.7→283.6) leaves the ball the same apparent size (eye) — a
volume break invisible in every shape metric; read P1, not geometry. Grown-protrusion wall stands a
**25th round**: best unforced `_15` protr 1.17 / grip 0.065, n_tubes 0 everywhere unforced.

## 4. What to do next
- **STOP the forced star.** Six rounds, all three axes filled; ctrl is the max, no cell beats it.
- **The owed lever is STILL untried, 10 rounds on:** anisotropic line-tension / bending gated on the
  pole at `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never launched.
- **Fix the replicate loss** — 2 of 12 slots this round measured the control.

# Round r026

## 1. What happened
Control `r026_00_ctrl` is the FORCED multi-arm star a **7th** round (mp 1.69, protr_peak 1.595,
grip_peak 0.2617, 6143c, n_spots 11). Nothing left the map filled r022–r025:
- **Forced star (`_04`/`_07`, mp 1.77/1.626):** protr 1.408/1.415, grip 0.216/0.207, cells
  8259/7991, n_spots 20/13 — both below ctrl grip, both higher growth.
- **Two bit-identical replicate copies:** `r026_05` AND `r026_06` == ctrl to every digit (protr
  1.595, grip 0.2617). Their "grip>0.09 confirmed" are self-comparisons; `_04`/`_07`'s grip>0.09
  is trivially true for a forced coarse star.
- **Unforced lober `_13` (mp 0):** protr 1.17, grip 0.0554, 8144c, n_spots_final 123,
  reduced_volume 0.83 — a knobbly inflated ball, no finger (n_tubes 0).
- **Reaction non-finite TWICE (`_14`/`_15`):** P1+P4+P12 broken, act_max→2.5e28/4.8e34→NaN by
  frame ~15, 2001c static sphere, grip ~2e-5.
- **Lost: 08/09/10/11/12 empty (5).**

## 2. What was learned
**Nothing new — a 7th confirmatory round on a closed map.** Growth dilutes grip on the coarse
forced field: ctrl 6143c→0.2617 > `_07` 7991c→0.207 > `_04` 8259c→0.216; ctrl at lowest growth is
the max, cannot be beaten by growing the recipe. Coarser-grips-harder holds: n_spots_final
11/13/20/123 → grip 0.2617/0.207/0.216/0.0554 (ctrl/`_07`/`_04`/`_13`). Grown-protrusion wall
stands a **26th round**: best unforced `_13` protr 1.17 / grip 0.0554, n_tubes 0.

## 3. What went wrong
Replicate bug returns a **20th round** — TWO copies (`_05`,`_06`). Reaction non-finite an 9th+
time (`_14`/`_15`, P4+P12): activator diverges to 1e28–1e34 then NaN ~1% into the run, mechanics
never fire, body a 2001c sphere. Five execution losses — the worst slot yield of the campaign.

## 4. What to do next
- **STOP the forced star.** Seven rounds, all axes filled; ctrl is the max, no cell beats it.
- **The owed lever is STILL untried, 11 rounds on:** anisotropic line-tension / bending gated on
  the pole at `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never launched.
- **Fix the launcher:** 5 empty + 2 replicate = 7 of 15 slots produced no new science.

# Round r027

## 1. What happened
Control `r027_00_ctrl` is the FORCED multi-arm star an **8th** round (mp 1.69, protr_peak 1.595,
grip_peak 0.2617, 6143c, n_spots 11). Nothing left the map:
- **Forced coarse cluster (`_01`/`_02`/`_05`/`_07`, mp 1.591/1.683/1.761/1.626):** grip
  0.215/0.228/0.238/0.207, protr 1.41/1.432/1.54/1.415, cells 7780/7548/6515/7991, n_spots
  21/14/15/13 — all below ctrl grip, all ≥ ctrl growth.
- **Replicate copy `_03` == ctrl to every digit** (protr 1.595, grip 0.2617); its "act_cv>4.49"
  refuted is a self-comparison (ctrl act_cv_peak 4.488 < 4.49).
- **Triple-identical `_13`==`_14`==`_15`** (one run, three slots): protr 1.348, grip 0.1244,
  10822c, mp 2.266, n_spots 136, act_max 0.889, reduced_volume 0.6525 — a fine-field spiky ball,
  no finger (n_tips 0).
- **Lost: 08/09/10/11/12 empty (5).**

## 2. What was learned
**Nothing new — 8th confirmatory round on the closed forced-star map.** Growth dilutes grip: ctrl
6143c→0.2617 > all four forced runs (6515–7991c) 0.207–0.238; ctrl at lowest growth is the max,
unbeatable by growing the recipe. Coarser-grips-harder holds: n_spots_final 11/14/15/21/136 → grip
0.2617/0.228/0.238/0.215/0.1244. NEW cautious point: high forcing (mp 2.266) on a FINE 136-spot
field held P11 and gave grip 0.1244 (~2× typical fine-field ~0.06) yet stayed a spiky ball
(reduced_volume 0.6525, n_tips 0) — no buckle, unlike r021's mp 2.023 coarse buckle. Fingering and
buckling depend on field coarseness, not forcing magnitude.

## 3. What went wrong
Replicate/duplicate bug a **21st round** — 4 wasted slots: `_03`==ctrl plus the `_13`/`_14`/`_15`
triple. NO unforced (mp 0) run this round, so the grown-protrusion wall is untested. Five
execution losses (08–12). 9 of 14 slots produced no new science — worst yield on file.

## 4. What to do next
- **STOP the forced star** (8 rounds, ctrl is the max, no cell beats it).
- **The owed lever, untried 12 rounds on:** anisotropic line-tension / bending gated on the pole
  at `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never launched.
- **Fix launcher + de-duplicate seeds:** 5 empty + 4 replicate is the worst yield on file.

# Round r028

## 1. What happened
Control `r028_00_ctrl` (== `r028_06`, a duplicate slot) is the FORCED multi-arm star a **9th**
round (mp 1.69, protr_peak 1.595, grip_peak 0.2617, 6143c, n_spots 11) — eye: "4-5 red-tipped
arms, starfish." Nothing left the map filled r022-r027:
- **Forced star (`_03`/`_05`):** `_03` mp 1.796, 7559c, protr 1.366, grip 0.1888, n_spots_final 30
  — growth-diluted below ctrl; `_05` mp 1.857, 3083c, protr 1.269, grip 0.0899, n_spots_final 20 —
  a knobbly mulberry sphere the eye says has NO tube (disputes n_tubes 2). Both below ctrl grip.
- **Unforced cell-volume-growth bulges (mp 0, no division, v_cell 0.24→0.54):** `_02` (2000c, 5
  domains) protr 1.215 (round's best unforced), grip 0.078; `_13`==`_14`==`_15` (one run, 18-spot
  high-act field, act_max 0.806) protr 1.063, grip 0.0183 — patterned lumpy spheres, no finger.
- **Ablations (mp 0):** `_01` rho→0 (2000c, grip 0.0007), `_04` P4 extinct (2201c, grip 0.00011),
  `_07` activator fully extinct (act_cv 0, n_spots 0, 2188c) — all spheres.
- **Lost: 08/09/10/11/12 empty (5).**

## 2. What was learned
**Nothing new — a 9th confirmatory round on the closed forced-star map.** Growth dilutes grip on
the forced coarse field again: ctrl 6143c→0.2617 > `_03` 7559c→0.1888; ctrl at lowest growth is the
max. Coarser-grips-harder holds: n_spots_final 11/20/30 → grip 0.2617/0.0899/0.1888 (the 20-spot
`_05` grips less than the 30-spot `_03` because its field is lower-growth and less coarsened, act_cv
1.66 vs 0.84). **Grip needs a growing tissue** reconfirmed three ways (`_01`/`_04`/`_07`, grip
<0.001). **A no-division tissue can still GROW by inflating cell volume** (`_02`/`_13`, v_cell
0.24→0.54) — it reaches an unforced coarse BULGE (protr 1.215) but no finger, so cell-volume growth
is not a substitute for the forcing that makes the star.

## 3. What went wrong
Replicate/duplicate bug a **22nd round** — 4 wasted slots: `_06`==ctrl plus the `_13`/`_14`/`_15`
triple. **Dead-field second-channel flag recurs (`_07`):** act_* reads extinct while the eye sees a
coarse two-tone field coarsening on every frame — same substrate read as r001_00/r014_05. Grown-
protrusion wall stands a **27th round**: best unforced `_02` protr 1.215 2-domain bulge, n_tubes 0
on every unforced run. Five execution losses.

## 4. What to do next
- **STOP the forced star** (9 rounds, ctrl is the max, no cell beats it).
- **The owed lever, untried 13 rounds on:** anisotropic line-tension / bending gated on the pole at
  `mech_p_ratio` 0 — the only path to a GROWN finger, owed since r015, never launched.
- **Fix launcher + de-duplicate seeds:** 5 empty + 4 replicate = 9 of 14 slots produced no new
  science, tying the worst yield on file.

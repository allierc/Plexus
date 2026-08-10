# Knowledge — surviving facts

## Round r001
- **Record coupling, but it grips colour not shape.** Growing GM/coral recipes reach
  `corr_act_rad_peak` 0.833 (`r001_14`) and `grip_peak` 0.047 (`r001_02`) — the campaign's
  strongest — while `protr_peak` stays ≤1.103, `n_tubes_peak` 0, no tip. The pattern coarsens
  into red domains over a faceted ball; the eye finds no finger in any run.
- **protr tracks crowding, not fingering.** High-growth (11.5k–14k cells) protr 1.098–1.103 vs
  low-growth (~3.3k cells) 1.077–1.091 — comparable despite 4× the cells. reduced_volume falls
  to 0.925 = faceting from packing, not protrusion.
- **Overdriving the activator kills the grip.** `r001_12/13`: `act_max` 9.6, `red_frac` railed
  1.0 (whole surface high) → `corr_act_rad_peak` collapses to 0.016 (floor −0.643), `grip`
  negative. A uniform-high field anti-correlates with radius. Do not push the reaction harder.
- **Extinct-field specimen.** `r001_07`: activator dies (`act_max_final` 0, P4 broken,
  `red_frac` 1.0); a secondary field coarsens (`act_cv_peak` 2.19) but grips nothing
  (`grip_peak` 0.0006). A high act_cv is not evidence of a gripping pattern.
- **Substrate flag: chemistry metric vs eye.** On no-chemistry runs (`r001_00`/08/09) the eye
  sees a persistent 5–6 orange-spot pattern while `act_cv`=`act_max`=`n_spots`=0 — a fixed
  non-activator channel or a wrong-field read. Verify before trusting act_* on such runs.
- `rd_interface_tension` inert on `r001_14`/`15` (null at that setting).

## Round r002
- **Best admissible specimen to date: `r002_12`, `protr_peak` 1.129, `grip_peak` 0.05735,
  `reduced_volume` 0.8935 — a scalloped/lobed ball, still no finger** (`n_tubes` 0,
  `protrusion_aspect_max` 0). Beats control 1.103 but misses the grip>0.06 gate. The 1.3
  protrusion wall stands; lobing deepens (`reduced_volume` 0.89) without ever fingering.
- **`r002_05`/`r002_06` are bit-identical to control** (`grip` 0.0467, `protr` 1.103, all fields
  equal) — the edit did not apply. Both carried grip>0.06 predictions, both scored refuted:
  two slots spent measuring the control. Check the diff actually lands before scoring.
- **Overdrive reconfirmed (`r002_10`/`11`):** `act_max` 6.75/6.88, `red_frac` 0.95–0.97 →
  `corr_act_rad` NEGATIVE (−0.23/−0.13), `grip` ~0, smooth sphere. Second replication of the
  r001 finding. Do not push the reaction.
- **Field-read bug recurs (`r002_14`/`15`):** every `act_*` reads 0 (P4 not even fired) while the
  eye sees a two-phase orange/blue field coarsening into large domains. `act_*` is reading the
  wrong channel on these recipes — same substrate flag as r001_00/08/09.
- **protr still tracks growth, not fingering.** `protrusion_aspect_max_peak` blips (`r002_08`
  1.602/1 tip; `r002_13` 0.754/10 tips) are transient single-frame specks, not sustained — those
  runs' `protr_peak` stays 1.089/1.093, below control.
- `rd_interface_tension` inert again on `r002_12`/`13` — null across settings.

## Round r003
- **protr FALLS with cell count.** Growth ladder (cells → `protr_peak`): 3209→1.127, 6683→1.095,
  9982→1.019, 15679→1.036. The largest tissues are the roundest; growth is not a protrusion lever,
  it is the opposite. Best `protr_peak` of the round = control at 3209 cells = 1.127. The r001/r002
  "protr ⊥ growth" sharpens to "protr ↓ growth." Stop climbing cell count for a finger.
- **Overdrive kills grip — 3rd replication.** `r003_08`/`09`: `act_max` 9.47, `red_frac` 1.0,
  `corr_act_rad` −0.137, `grip` −0.0006 at 9982 cells. Third independent confirmation after
  r001_12/13 and r002_10/11. Do not push the reaction; this is closed.
- **Pattern strength ≠ protrusion.** `r003_07` reached `act_cv_peak` 2.647 and 120 spots — the
  strongest chemistry of the campaign — with `protr` 1.111, `grip` 0.055, `n_tubes` 0. A stronger
  pattern grips colour, not shape.
- **Bit-identical replicates measure noise at 0.** `r003_01`==control, `05`==`06`, `08`==`09`,
  `14`==`15`: pairs run at the SAME seed, identical every field. Seed spread remains unmeasured
  after three rounds; a replicate must change seed to bound it.
- **Field-read bug, 3rd recurrence** (`r003_12`/`13`): `act_*`=0 at 15679/13920 cells while the eye
  sees a coarsening orange/blue two-phase field. Same wrong-channel read as r001_00/08/09 and
  r002_14/15; unfixed.
- `rd_interface_tension` inert a 3rd time (`r003_10`/`11`) — null across three rounds. Stop sweeping it.

## Round r004
- **The lobing leg is tuning-saturated: 4 rounds, no run beats control's `protr_peak` 1.127.**
  Round r004 best = control (3209c); every edit landed 1.074–1.127, `n_tubes` 0. protr↓growth holds
  (`r004_13` 9982c → 1.019). A finger will not come from parameters on this composition — needs a
  new mechanical operator (activator→bending/line-tension), not growth, not a hotter reaction.
- **Overdrive kills grip — 4th and 5th replication.** `r004_13`/`r004_15`: `act_max` 9.47/8.25,
  `red_frac` 0.97, `corr_act_rad` −0.137/−0.28, `grip` ≤−0.0006. Five confirmations; do not push
  the reaction, ever.
- **`protrusion_aspect_max_peak`>0.5 is a single-frame blip, not fingering.** `r004_04` 0.622 (5 tips)
  and `r004_02` 0.742 (4 tips) both have `protr_peak` BELOW control (1.074/1.086) and the eye saw no
  protrusion. Do not pose predictions on this metric — it has misled 3 rounds running.
- `rd_interface_tension` inert a 4th time (`r004_08`/`09`). Field-read `act_*`=0 bug a 4th time
  (`r004_10`/`11`). Replicate-seed bug a 4th time (`r004_03`==control bit-identical); seed spread
  still unmeasured after four rounds.

## Round r005
- **5th round, wall stands: no run beats control `protr_peak` 1.127; `n_tubes` 0.** Best edits
  01/02 land 1.104/1.113 (lobed, not fingered). The lobing leg is closed to tuning.
- **Growth is not a lever in either direction.** Low-growth `r005_10`/`11` (~2000c) give `protr`
  1.063, `grip` 0.030 — *below* the 3209c control's 1.127/0.053. 3209c is a shallow optimum, so
  neither climbing nor cutting cell count fingers the tissue.
- **Overdrive kills grip — 6th/7th replication.** `r005_12`/`13`: `act_max` 7.31/7.49, `red_frac`
  0.947/0.966, `corr_act_rad` −0.558/−0.542, `grip` negative. Closed beyond doubt.
- `rd_interface_tension` inert a 5th time (`r005_14`/`15`). Field-read `act_*`=0 bug 5th
  (`r005_08`, extinct vs patterned eye). Same-seed replicate bug 5th (05=06=07=ctrl, 14=15); seed
  spread unmeasured after five rounds. Execution loss: `r005_09` empty.

## Round r006
- **First runs past control in six rounds — but inside unbounded seed noise.** `r006_04`
  `protr_peak` **1.153** (control 1.127, `gyr_prolate` 1.327) and `r006_15` `grip_peak` **0.0633**
  (control 0.053; first ever past the 0.06 gate; `reduced_volume` 0.865, `corr_act_rad_peak`
  0.773, 5597c). Both still `n_tubes` 0, `aspect_max` ≤0.394, eye sees no finger. The beats are
  +0.026 / +0.010; seed spread is STILL unmeasured, so treat as unconfirmed until re-run at a
  fresh seed.
- **Pattern grips nothing without growth.** `r006_07` (2001c, division off) keeps `act_cv_peak`
  2.15 but `protr_peak` 1.014, `grip` 0.006 — a flat sphere. Grip requires an actively growing
  tissue; a strong static pattern on a fixed shell couples to colour only.
- **Overdrive kills grip — 8th replication.** `r006_09`: `act_max_peak` 7.49, `red_frac` 0.966,
  `corr_act_rad` −0.542, `grip` −0.0029. Closed.
- `rd_interface_tension` inert a **6th** time (`r006_10`/`11`). Same-seed replicate bug a **6th**
  round (01=05, 10=11 bit-identical) — seed spread unmeasured after six rounds. Three execution
  losses (`r006_12`/`13`/`14` empty).

## Round r007
- **7th round, wall stands: no run beats control `protr_peak` 1.13 or `grip_peak` 0.06338;
  `n_tubes` 0.** Best `protr_peak` 1.134 (`r007_03` 3641c, `r007_14` 4470c) = control within noise;
  best `grip_peak` 0.0598 (`r007_07`/`14`) is BELOW control. Lobing leg closed to tuning.
- **protr optimum is shallow at ~3600–4470c.** cells→`protr_peak`: 2054→1.014, 3179→1.129,
  3641/4470→1.134, 5580→1.130 (ctrl), 7744→1.097, 12355→1.019. Both extremes round the ball;
  protr↓growth holds at high count for a 7th round.
- **Overdrive kills grip — 9th/10th replication.** `r007_08`/`09`: `act_max` 6.18/7.18,
  `red_frac` 1.0, `corr_act_rad` −0.364/−0.277, `grip` negative at ~12000c. Closed.
- **corr_act_rad without amplitude is not grip.** `r007_15` `corr_act_rad_peak` 0.830 /
  `r007_07` 0.827 on near-spheres (low r_cv) yet `grip` 0.032/0.060 — lead with grip, not corr.
- `rd_interface_tension` inert a **7th** time (`r007_10`/`11`). Same-seed replicate bug a **7th**
  round (`r007_04` bit-identical to control) — seed spread STILL unmeasured. Two execution losses
  (`r007_12`/`13` empty).

## Round r008
- **First run past control on BOTH axes in eight rounds — still no finger.** `r008_09`
  `protr_peak` **1.144** (ctrl 1.13) AND `grip_peak` **0.0711** (ctrl 0.0634), `corr_act_rad_peak`
  0.880, at 3801c. `n_tubes` 0, `protrusion_aspect_max` 0, eye sees a 6–8-fold undulating raspberry,
  no finger. Beats are +0.014/+0.008 on ONE seed; seed spread unmeasured, so unconfirmed.
- **A COARSER pattern grips harder.** Low-growth `r008_07`/`08`/`09` (3417/3640/3801c;
  `n_spots_final` 32/20/21, `spot_spacing_cells` 5.0/8.3/8.1) reach `corr_act_rad_peak`
  0.80/0.86/0.88 and `grip` 0.052/0.062/0.071 — above high-growth `r008_01`/`02` (7559/7744c,
  corr 0.68/0.66, grip 0.041). protr↓growth holds an **8th** round; grip↓growth too.
- **Overdrive kills grip — 11th/12th replication.** `r008_13`/`14`: `act_max_peak` 9.45/9.11,
  `red_frac_peak` 0.91/0.76, `corr_act_rad` −0.30/−0.19, `grip` negative (6997/4729c). Closed.
- **`protrusion_aspect_max`>0.5 misleads a 4th time.** `r008_12` 0.648 / `n_tips` 3 on a
  `protr_peak` 1.066 sphere (2219c, eye: round). Do not pose predictions on it.
- `rd_interface_tension` inert an **8th** time (`r008_08`/`09`). Same-seed replicate bug an **8th**
  round (`r008_05`, `r008_06` bit-identical to control; `r008_06` even scored a protr>1.13
  prediction against the control it copies). Two execution losses (`r008_10`/`11` empty).

## Round r009
- **Control set a new high on BOTH axes and one edit beat it — still no finger.** Control
  `r009_00` `protr_peak` **1.158**, `grip_peak` **0.07315** (3767c) — highest control ever,
  seed-dependent. `r009_04` beat it on both: `protr_peak` **1.198**, `grip_peak` **0.09512**
  (campaign max grip, prior best 0.0711 `r008_09`), `corr_act_rad_peak` 0.874, `r_cv` 0.113,
  `reduced_volume` 0.705 (round's deepest lobing), 4149c. `n_tubes` 0, `protrusion_aspect_max` 0,
  eye saw a scalloped/undulating ball, no finger. ONE seed; seed spread unmeasured a **9th** round.
- **protr↓growth holds a 9th round.** cells→`protr_peak`: 3241→1.072, 3381→1.105, 3767→1.158 (ctrl),
  4149→1.198, 5439→1.144, 7576→1.104. Optimum shallow at ~3800–4150c; the two largest tissues are
  the roundest.
- **Overdrive kills grip — 13th/14th replication.** `r009_12`/`13`: `act_max_peak` 8.67/8.82,
  `red_frac` 1.0, `corr_act_rad` −0.175/−0.181, `grip` ≈−0.001 (6832/7675c). Closed beyond doubt.
- `rd_interface_tension` inert a **9th** time (`r009_14`/`15`, explicit null). Same-seed replicate
  bug a **9th** round (`r009_01`==`r009_02` bit-identical, protr 1.091 grip 0.0417). Two execution
  losses (`r009_08`/`09` empty).

## Round r010
- **Seed spread finally bounded — and the r009_04 double-beat SURVIVES it.** The standing best
  recipe (`r009_04`, now this round's control) replicated across 3 seeds: `protr_peak`
  {1.198 (ctrl), 1.184 (`r010_03`), 1.176 (`r010_04`)} = 1.186±0.011; `grip_peak`
  {0.09512, 0.09873, 0.09014} = 0.0947±0.004. Span ≤0.022 protr / ≤0.0086 grip — an order
  smaller than the recipe's +0.06 protr / +0.03 grip LEAD over the historical ~1.13/0.06 baseline.
  Ten rounds of single-seed beats are now bounded: the beat is real, not seed luck. Caveat:
  `r010_05` failed to reseed (bit-identical to control), so only 2 truly fresh seeds back this.
- **New campaign-max grip: `r010_03` `grip_peak` 0.09873** (prior 0.09512), `n_spots_final` 16,
  `spot_spacing_cells` 10.03 — the round's COARSEST pattern grips hardest; coarser-grips-harder
  holds. Still `n_tubes` 0, `protrusion_aspect_max` 0, eye sees an undulating/scalloped shell.
- **protr↓growth holds a 10th round.** cells→`protr_peak`: 2140→1.012, 2598→1.058, 3592→1.105,
  3819→1.138, 4149→1.198 (opt), 5092→1.112, 10523→1.017, 11428→1.018. Shallow optimum ~4149c;
  both extremes round the ball.
- **Overdrive kills grip — 15th/16th replication.** `r010_09`/`10`: `act_max` 9.24/9.42,
  `red_frac` 1.0, `corr_act_rad` −0.30/−0.21, `grip` ≈0 (10523/11428c). Closed.
- `rd_interface_tension` inert a **10th** time (`r010_11`/`12`, explicit null). Same-seed
  bit-identical replicate bug a **10th** round (`r010_05`==ctrl; `r010_01`==`r010_02`).
  `r010_12` broke P1 (no growth, 2140c, volume flat). Three execution losses
  (`r010_13`/`14`/`15` empty). `protrusion_aspect_max` 0.461/5 tips on `r010_11` is a blip
  (protr 1.058 sphere) — misleads a 5th time.

## Round r011
- **Campaign-max protr 1.241 was FORCED, not grown — first nonzero `mech_p_ratio` on file.**
  `r011_15` (19135c) reached protr_peak 1.241 (prior max 1.198) with n_tubes 2 / tube_diam 0.833,
  but `mech_p_ratio` 1.752 while every other run in eleven rounds reads 0.0 — a pushing operator is
  in this composition. Grown≈1, forced≈3, so 1.241 is a half-forced lobe (eye: 5–6-lobed berry,
  the "2 tubes" are two bulges, aspect_max 0). A protrusion produced with forcing does not answer
  the open problem; it is not counted as a finger.
- **protr↓growth holds an 11th round among the UNFORCED runs.** cells→protr_peak: 2338→1.05,
  4149→1.198, 4544→1.184 (ctrl), 5508→1.141, 6399→1.16, 9008→1.019 (overdrive). Optimum shallow
  at ~4149c; both extremes round the ball. n_tubes 0, aspect_max 0 on every unforced run.
- **Overdrive kills grip — 17th replication.** `r011_08`: act_max_peak 9.46, red_frac 0.979,
  corr_act_rad −0.142, grip −0.0008 at 9008c. Closed beyond any doubt.
- **The reaction can still go non-finite.** `r011_09` blew up at frame 48 (act_max→3.8e17, NaN;
  P1/P4/P12 broken), then 850 frames of a static dead sphere. A blow-up masquerades as a sphere in
  the shape metrics — read P12 first.
- Same-seed bit-identical replicate bug an **11th** round (`r011_05`==ctrl; `r011_01`==`r011_02`).
  Three execution losses (`r011_10`/`11`/`12` empty).

## Round r012
- **12th round, wall stands: no UNFORCED run beats control `protr_peak` 1.184 or `grip_peak`
  0.09873, `n_tubes` 0 everywhere unforced.** The 01–05 lobers (5089–6459c) land 1.113–1.163,
  grip 0.055–0.071 — all below control, all refuted `protr>1.3`/`>1.241`.
- **Forcing scales with cell count, and remains the only lever past 1.2 — 2nd forced run.**
  `r012_06` (18717c) `mech_p_ratio` **2.011** (every unforced run 0.0) gives `protr_peak` 1.218,
  `n_tubes` 1, `tube_diam` 0.618 — a pushed lobe, `aspect_max` 0, not a grown finger (after
  r011_15 at 19135c/mech_p_ratio 1.752). A forced protrusion answers nothing.
- **protr↓growth holds a 12th round among unforced runs.** cells→`protr_peak`: 2001→1.014,
  2966→1.078, 3152→1.076, 4544→1.184 (opt), 5089–5532→1.11–1.13, 6459→1.163. Shallow optimum
  ~4544c.
- **Growth-off decouples pattern from shape — 2nd replication** (`r012_07` vs r006_07): 2001c,
  division off, `red_frac` 0 → `protr` 1.014, `grip` 0.006. Grip needs a growing tissue.
- **Reaction went non-finite a 2nd/3rd time** (`r012_11`/`12`, after r011_09): `act_max` NaN,
  `act_mean` ±1e26, P1/P4/P12 broken, then a static dead sphere (protr railed 1.014). Read P12
  first — a blow-up reads as a sphere in the shape metrics.
- Four execution losses (`r012_08`/`09`/`10`/`14` empty). No same-seed replicate this round, so
  seed spread stays the r010 bound (protr 1.186±0.011, grip 0.0947±0.004).

## Round r013
- **13th round, wall stands: no run beats control `protr_peak` 1.184 or `grip_peak` 0.09873,
  `n_tubes` 0 everywhere.** `r013_02` (4883c) protr 1.193 / grip 0.099 ties control inside the r010
  seed band; the lobers 01/04/05 land 1.165–1.174, grip 0.084–0.096, all refuted `protr>1.2/1.25`.
- **A low-growth STRIPE/LABYRINTH regime exists and grips nothing.** `r013_03` (2664c) corr_act_rad_peak
  0.791, `r013_06` (2000c, division off, v_cell 0.50), `r013_15` (3535c): eye sees a connected
  red/white maze, not spots; grip 0.014/0.042/0.046, shape a sphere (protr ≤1.095). Pattern topology
  (stripes vs spots) does not change whether it deforms the shell.
- **Coarser-grips-harder holds a 13th round.** `n_spots_final` 16(ctrl)/27/36/46 → grip
  0.099/0.096/0.084/0.076; finer pattern grips LESS. `r013_07` confirmed `n_spots_final>25` (46).
- **protr↓growth holds a 13th round.** cells→`protr_peak`: 2000→1.094, 2664→1.029, 3535→1.095,
  3974→1.165, 4544→1.184 (opt), 4883→1.193, 7903/8443→1.017. Shallow optimum ~4544–4883c.
- **Overdrive kills grip — 18th/19th replication.** `r013_11`/`12` act_max_peak 10.36/10.28,
  red_frac 0.98/0.996, corr_act_rad −0.076/−0.157, grip <0. Reaction went non-finite a **4th time**
  (`r013_12`, P12 broken). Four execution losses (08/09/10/14 empty).

## Round r014
- **The protr wall broke — a single-pole BUD, but half-FORCED.** `r014_01`/`r014_02` reached
  `protr_peak` **1.453**/**1.420** (prior unforced ceiling 1.184, prior forced 1.241) with
  `protrusion_aspect_max_peak` 1.331/1.458, `n_tips` 2, `grip_peak` 0.139/0.130 — the first
  protrusion in 14 rounds with aspect>1.3 AND a neck the eye called "budding-off, not a finger"
  (Okuda Fig-5b budding shape). Chemistry collapsed to ONE pole (`n_spots_final` 1/2, `red_frac`
  0.137/0.151, `act_cv_peak` 9.65/7.67) at LOW growth (2286/2325c). BUT `mech_p_ratio`
  **1.958**/**1.783** (grown≈1, forced≈3; every other r014 run 0.0) → ~half PUSHED, so NOT yet a
  grown finger. The two runs agree tightly (protr 1.45/1.42, grip 0.14/0.13, aspect 1.33/1.46) — the
  bud regime is reproducible. Decisive next test: lower the forcing gain and watch whether protr
  survives as `mech_p_ratio`→1.
- **Coarser-grips-harder to the limit:** `n_spots_final` 1 → `grip` 0.139, the round's hardest grip.
- **Overdrive kills grip — 20th/21st replication.** `r014_09`/`10`: `act_max_peak` 8.38/9.04,
  `red_frac` 0.996/0.999, `corr_act_rad` floor −0.65/−0.66, `grip` ~0, spheres (9021/8737c).
- **Dead-activator second field, unmeasured a 6th time.** `r014_05`/`06`/`07` (P4 broken, red
  extinct, `grip` 0.002–0.008) show a persistent blue/yellow field the eye sees coarsening to the
  end that act_* reads as extinct. Four execution losses (`r014_08`/`11`/`12`/`13` empty).

## Round r015
- **The r014 bud was FORCED, not grown — decisive test answered NEGATIVE.** A forcing-gain ladder
  down the bud recipe makes `protr_peak` MONOTONE in `mech_p_ratio`: 1.958→1.453, 1.532→1.353
  (`r015_07`), 1.439→1.346 (`r015_03`), 1.348→1.276 (`r015_06`), 0.0→1.085 (`r015_14`, unforced,
  3131c). At gain 0 the bud collapses to a mildly lumpy sphere — `protrusion_aspect_max` 1.331→0,
  n_tips 2→0, no neck. protr does NOT survive as `mech_p_ratio`→1, so the r014 protr-1.45 wall
  break is an `extrude`-class push, not a grown finger. **The 1.3 GROWN-protrusion wall stands a
  15th round; no unforced run beats protr 1.085, `n_tubes` 0 everywhere. Retire the forced-bud
  line — protr from forcing is linear in gain and zero at gain 0.**
- **Forcing needs a live pattern to gate on.** `r015_01`==`r015_02` (bit-identical, P4 broken,
  chemistry extinct) push a dead field → `protr` 1.004, flat sphere. Forcing on nothing does
  nothing.
- **Best grip is moderate-forcing MULTI-lobe, but still forced.** `r015_03` `grip_peak` 0.152
  (>ctrl 0.139) at `mech_p_ratio` 1.439, `n_spots_final` 6, protr 1.346 — extra lobes buy grip,
  not a tube (eye: broad bulges, `protrusion_aspect_max` 1.75 transient). n=1, unreplicated.
- **Overdrive kills grip — 22nd/23rd replication.** `r015_09`/`15` act_max 9.54/9.55, red_frac
  0.976/0.97, corr_act_rad −0.226/−0.189, grip <0, spheres (9179/8702c), unforced.
- Reaction non-finite a **5th time** (`r015_08`, act_max 5.7e17, P1/P4/P12 broken; read P12
  first). Same-seed bit-identical replicate bug a **12th** round (`r015_04`==ctrl). Four execution
  losses (`r015_10`/`11`/`12`/`13` empty).

## Round r016
- **The forced grip 0.152 (`r015_03`/`r016_00` ctrl) is STILL n=1 — the reseed failed.** `r016_03`
  came back bit-identical to control (`mech_p_ratio` 1.439, protr 1.346, grip 0.15244), so its
  "grip>0.152 confirmed" is a self-comparison, NOT a fresh-seed replicate. That point remains
  unbounded.
- **Grown-protrusion wall stands a 16th round.** No unforced run beats `protr_peak` 1.204 or
  `grip_peak` 0.11538 (`r016_01`, 4001c, `mech_p_ratio` 0); `n_tubes` 0 on every unforced run. Only
  the FORCED control exceeds it.
- **Coarser-grips-harder holds a 14th round, cleanly at one growth scale.** n_spots_final 18/23/166
  → grip 0.115/0.042/0.018 (`r016_01`/`12`/`15`); the 166-spot fine field is also roundest
  (protr 1.083). Coarseness, not cell count, sets protr: 4001c/18sp→1.204 > 3148c/166sp→1.083.
- Reaction non-finite a **6th and 7th time** (`r016_11`/`13`, act_max 1.3e30/1.7e23→NaN, P1/P4/P12
  broken, read as intact spheres — read P12 first). Same-seed bit-identical replicate bug a **13th
  round and the worst on file**: five slots collapsed to two results (`r016_01`==`02`==`05`==`07`;
  `r016_03`==ctrl). Four execution losses (`r016_08`/`09`/`10`/`14` empty).

## Round r017
- **Campaign-max protr 1.588 AND grip 0.198 — both FORCED, both answer nothing.** `r017_07`
  (2486c) `mech_p_ratio` **2.284** (every unforced run 0.0) → `protr_peak` **1.588** (prior forced
  max 1.453), `grip_peak` **0.198** (prior 0.152), `gyr_prolate_peak` 2.253, `protrusion_aspect_max`
  1.748, `n_tubes` 3 — yet the eye sees TWO fat red-tipped buds on a body whose cross-section stays
  a clean circle (a prolate two-bud peanut, not fingers). Forcing stays linear past 1.5:
  `mech_p_ratio` 2.284→protr 1.588 (`r017_07`), 2.025→1.268 (`r017_15`), 1.958→1.453 (`r017_06`,
  reproduces `r014_01`). Confirmed `gyr_prolate`>2 is the forced elongation, not a grown tube.
- **Grown-protrusion wall stands a 17th round.** No unforced run beats `protr_peak` 1.204 or
  `grip_peak` 0.115 (`r017_02`, 4001c, `mech_p_ratio` 0, `n_spots_final` 18) — ties the r016_01
  ceiling exactly; `n_tubes` 0 on every unforced run.
- **Coarser-grips-harder holds a 15th round.** Unforced `n_spots_final` 18/41/40/83 → grip
  0.115/0.097/0.071/0.028 (`r017_02`/`03`/`05`/`13`). protr↓growth holds: unforced 3468→1.074,
  3909→1.19, 4001→1.204 (opt), 4729→1.15.
- **Overdrive kills grip — 24th/25th replication.** `r017_11` uniform-saturated (`act_max` 4.02,
  `red_frac` 1.0, `act_cv` 0.007, grip ~0, 7872c); `r017_12` (`act_max` 9.98, `red_frac` 0.942,
  `corr_act_rad` −0.228, grip −0.001, 8809c). `r017_04` P4 broken (chemistry extinct, grip 7e-05) —
  dead-field specimen. Four execution losses (`r017_08`/`09`/`10`/`14` empty).

## Round r018
- **NEW base recipe, campaign's FIRST grown multi-armed protrusion — a step change.** Control
  `r018_00` (1801 frames, 7424c): `protr_peak` **1.513**, `grip_peak` **0.228**, `n_tubes_peak`
  **7**, `corr_act_rad_peak` 0.944, `act_cv_peak` 4.34, `mech_p_ratio` 1.658 — both protr and grip
  beat every prior FORCED max (1.588/0.198) and dwarf the prior unforced wall (1.204/0.115). Eye
  across 3 bit-identical seeds (02/04/07): a genuine 4–7 armed star of red-tipped fingers, "grown
  not pushed." First specimen in 18 rounds the eye reads as real fingers, not buds/lobes.
  `mech_p_ratio` 1.658 (grown≈1, forced≈3) + eye = grown-dominant, but forcing is present (mp≠0),
  so a gain ablation is still owed before calling it purely grown.
- **Multi-arm grips harder AND forces less than any concentrated variant.** grip / mech_p_ratio:
  control multi-arm 0.228/1.658 > single-finger `r018_05`/`06` 0.187/2.103 > multi-lobe `r018_15`
  0.126/1.74 > bilobed bud `r018_14` 0.117/2.337.
- **Concentrating chemistry to one pole at low growth = one extreme finger, not more grip.**
  `r018_05`==`06` (3750c): `gyr_prolate_peak` **7.86** (prior campaign norm ~1.5–2.3), `protr_peak`
  1.543 (round max), `n_tubes_final` 1 — eye "closest to a genuine tube" — yet grip 0.187 < the
  multi-arm control 0.228.
- **protr↓growth REVERSES in this regime.** Largest tissue grips hardest: cells→grip 7424→0.228,
  5609→0.126, 3059→0.117; the old law (small tissue, higher protr) does not carry to the new recipe.
- Replicate bug a **14th round** (`r018_02`/`04`/`07`==ctrl; `05`==`06`) — their "confirmed"
  predictions are self-comparisons. Reaction non-finite an **8th time** (`r018_09`, P4+P12,
  act_max→6.8e8→NaN, 15989c overgrowth, static sphere). Chemistry extinct a further time (`r018_13`,
  P4, flashes once then dies, 2051c no-growth sphere). Four execution losses (`r018_08`/`10`/`11`/`12`
  empty).

## Round r019
- **Campaign-max protrusion AND grip, but half-FORCED — `r019_07`.** `protr_peak` **1.817**,
  `grip_peak` **0.294**, `gyr_prolate_peak` **5.857**, `protrusion_aspect_max_peak` **11.42**,
  `n_tubes` 2, 4368c, chemistry to one pole (`n_spots_final` 10). Eye: "first real protrusion... reads
  as grown, not merely bulged." But `mech_p_ratio` **2.019** (grown≈1, forced≈3) — forcing present,
  ~half. A gain ablation is owed before it counts as a grown finger; the r015 bud was linear in gain
  and zero at gain 0.
- **First UNFORCED seed bound on the new recipe** (`r019_08`/`09`, one composition at two seeds,
  `mech_p_ratio` 0, ~6.5k c): `protr_peak` **1.194±0.015**, `grip_peak` **0.0865±0.0005**,
  `n_spots` 75/72 — undulating lobed ball, no finger. Any protr/grip difference inside this band is
  seed noise.
- **`rd_interface_tension` INERT — 3 more runs** (`r019_08`/`09`/`15`), reconfirming r001/r002. Do not
  propose it again.
- **Grip needs a growing tissue — reconfirmed.** rho→0 (`r019_01`, 2000c) protr 1.003, grip 0.0003;
  activator extinct (`r019_06`, P4 broken, 2174c) grip 0.00026 — both spheres.
- **Coarser-grips-harder holds in the new recipe:** `n_spots_final` 10/22/42 → grip
  0.294(forced)/0.228(ctrl)/0.190 (`r019_07`/ctrl/`r019_03`).
- Replicate bug a **15th round** (`r019_02`/`04`/`05`==ctrl, protr 1.513) — control still n=1 across
  seeds. Four execution losses (`r019_10`/`11`/`12`/`13` empty).

## Round r020
- **Unforced seed bound reconfirmed a 3rd time.** `r020_12`/`14` (`mech_p_ratio` 0): `protr_peak`
  1.184/1.190, `grip_peak` 0.089/0.0915, `n_spots` 83/93 — inside r019_08/09's 1.194±0.015 /
  0.0865±0.0005. No unforced run beats protr ~1.19 / grip ~0.09; `n_tubes` 0.
- **Round-max grip `r020_06` 0.2617 (>ctrl 0.228) but FORCED.** `mech_p_ratio` 1.69,
  `n_spots_final` 11, `gyr_prolate_peak` 3.008, `protrusion_aspect_max_peak` 4.87; the whole
  multi-arm cluster (`r020_02`/`03`/`04`/`06`) sits mp 1.68–1.77, grip 0.216–0.2617, protr
  1.408–1.607. Eye reads all four as genuine grown fingers, but mp≠1 → gain ablation still owed.
- **Coarser-grips-harder holds again:** `n_spots_final` 11/14/83/93/179 → grip
  0.2617/0.240/0.089/0.0915/0.080 (`r020_06`/`03`/`12`/`14`/`13`).
- **`rd_interface_tension` INERT** on `r020_08`/`12`/`14` (3 more) — reconfirms r001/r002/r019. Retire.
- Replicate bug a **16th round** (`r020_05`==`07`==ctrl, protr 1.513, grip 0.228) — control still
  n=1 across seeds. Four execution losses (`r020_09`/`10`/`11`/`15` empty).

## Round r021
- **Forcing ablation answered a 4th way — at MATCHED growth.** At ~7k c, removing forcing drops the
  multi-arm star from protr 1.595 / grip 0.2617 (`r021_00` ctrl, `mech_p_ratio` 1.69) to protr
  **1.164** / grip **0.077** (`r021_13`, mp 0) — a lobed ball, no finger. Forcing carries ~0.43
  protr and ~0.18 grip; every "5–7 fingers" the eye reports (`r021_03`) is on an mp ≥ 1.65 run. The
  star's protr/grip are forcing-dependent — reconfirms r015/r020.
- **Unforced protr wall nudges to 1.257, still no finger.** `r021_02` (mp 0) protr 1.257 is the
  highest unforced protr on file, but it is a 2000c no-division 3-domain BULGE (grip 0.080,
  `protrusion_aspect_max` 0, `n_tubes` 0). The 1.3 GROWN-finger wall stands a **21st round**;
  `n_tubes` 0 on every unforced run.
- **Over-forcing buckles the shell.** `r021_04` `mech_p_ratio` 2.023 → P11 broken (inward fold),
  protr 1.438. Above mp ~2 the radial push folds the tissue rather than fingering it.
- **Round-max protr 1.652 is a BUD, not a finger.** `r021_05` (3782c, mp 1.635) protr 1.652,
  `protrusion_aspect_max_peak` 12.034 — but the aspect is a single-frame spike and the eye reads
  2–3 rounded lobes. Highest protr ≠ finger; protr↓growth holds within the forced set
  (3782c→1.652 > 7200c→1.417).
- **Grip needs a growing tissue — reconfirmed.** rho→0 `r021_01` (grip 0.0003, P4 broken) and
  activator-extinct `r021_07` (grip 0.0001, P4) are both spheres.
- **Coarser-grips-harder holds again:** `n_spots_final` 11/6/25/77 → grip 0.2617/0.217/0.120/0.103
  (ctrl/`r021_05`/`r021_15`/`r021_11`).
- No replicate bug this round (predictions genuine, 6 refuted / 1 confirmed). Four execution losses
  (`r021_08`/`09`/`10`/`14` empty).

## Round r022
- **Forcing carries grip ONLY on a coarse field — the r021 "~0.18 grip" is conditional.** `r022_10`/`11`
  are FORCED (`mech_p_ratio` 1.68/1.615) yet on a FINE 70/120-spot field give `grip_peak` 0.087/0.087
  ≈ the UNFORCED `r022_09` (mp 0) 0.079 — all three inflate to `reduced_volume` 0.80–0.82 (near-sphere,
  ctrl 0.48), no finger. Coarser-grips-harder and forcing-works are the same statement: forcing needs
  a coarse (`n_spots_final` ~11) field to act on.
- **Within the forced star, protr AND grip fall monotone with growth.** cells→protr/grip: 6143→1.595/0.262
  (ctrl) > 7929→1.434/0.229 > 7991→1.415/0.207 > 8259→1.408/0.216 (`r022_02`/`07`/`04`). ctrl at the
  LOWEST growth is the max-grip point, so it cannot be beaten by growing the same recipe — every
  "beat-ctrl" prediction (`r022_01`/`02` grip, `03` protr) refuted for this reason.
- **A NEW fine near-uniform chemistry regime, mapped:** `act_max` 0.73–0.83 (ctrl 0.589), `act_cv`
  0.67–0.80 (ctrl 4.49), `n_spots` 70–120, grew to the largest tissue on file (`r022_11`, 10708c) —
  and it INFLATES (`reduced_volume` 0.80–0.82) rather than fingering. grip ~0.08 forced or unforced.
- **Dead Gray-Scott field grips nothing (reconfirmed).** `r022_15`: activator extinct (`act_mean`
  0.018, `corr_act_rad` −0.25), a two-phase blue/yellow field segregates (eye) but the shape stays a
  clean sphere (`protr` 1.03, `grip` −0.004, 2903c).
- **Grown-protrusion wall stands a 22nd round:** best unforced `r022_09` protr 1.177 / grip 0.079,
  `n_tubes` 0. Replicate bug a 17th round (`r022_06`==ctrl). Four execution losses (08/12/13/14 empty).

## Round r023
- **Coarser-grips-harder — the cleanest 5-point ladder on file, one round, one recipe:**
  `n_spots_final` 11/14/58/68/91 → `grip_peak` 0.2617/0.228/0.1153/0.0975/0.074
  (ctrl/`r023_02`/`r023_15`/`r023_14`/`r023_08`), monotone. Fine fields also inflate
  (`reduced_volume` 0.66/0.72/0.84), no finger.
- **Growth dilutes grip on a coarse field — reproduces r022 exactly.** ctrl 6143c→1.595/0.262 vs
  `r023_02` 7548c→1.432/0.228 (both `n_spots` 11–14, `mech_p_ratio` 1.69/1.683). ctrl at lowest
  growth is the max-grip point; growing the recipe cannot beat it.
- **Best eye-finger of the round is still FORCED and beats nothing.** `r023_02` (mp 1.683)
  `protrusion_aspect_max_peak` 3.327, `n_tips_peak` 8, `n_tubes_peak` 8, eye "6–8 grown red-tipped
  fingers, narrow necks" — yet grip 0.228 < ctrl 0.262. Forcing-carried, no new mechanism.
- **An UNFORCED stripe field grips nothing — and stripe topology adds no grip over spots.**
  `r023_13` (mp 0, 4228c): a labyrinthine worm-stripe Turing field, `protr_peak` 1.026, `grip_peak`
  0.0058 — a clean sphere, the round's lowest grip. Grip needs forcing (reconfirmed).
- **Grown-protrusion wall stands a 23rd round:** the only unforced run is `r023_13`'s 1.026 stripe
  sphere; `n_tubes` 0 unforced. Replicate bug an 18th round (`r023_06`==`r023_07`==ctrl, both
  refuted grip>0.262 against the control they copy). Four execution losses (09/10/11/12 empty).
- **The forced-star (growth × coarseness) map is now filled TWICE (r022+r023); stop sweeping it.**
  The owed grown-finger lever — anisotropic line-tension / bending gated on the pole at
  `mech_p_ratio` 0 — remains untried 8 rounds after r015.

## Round r024
- **Forcing MAGNITUDE saturates — at matched ~7k growth it barely moves grip.** `mech_p_ratio`
  1.303/1.672/1.942/1.932 → `grip_peak` 0.214/0.204/0.218/0.243 (`r024_05`/`_02`/`_06`/`_07`, all
  6.7–7.1k c, n_spots 15–22). Forcing is a GATE (mp 0 → grip <0.08), not a dial above ~1.3; grip is
  set by field coarseness × growth, not by how hard you push. Third axis of the forced-star map, flat.
- **Forcing needs a coarse field — reconfirmed (r022/r024).** Forced runs on a NEW high-act
  near-uniform regime (`act_max` 0.92–0.99 vs ctrl 0.589, `act_cv` 0.68–0.86) give grip 0.093/0.072
  despite mp 1.844/1.941 (`r024_14` 29-spot 7183c / `_15` 7-spot 3529c) — a third of ctrl's grip at
  ctrl's forcing. This regime inflates/lobes, never fingers.
- **Grown-protrusion wall stands a 24th round.** Best unforced is `r024_03` `protr_peak` 1.222 — a
  2000c no-division 2-domain BULGE (grip 0.074, `n_tubes` 0); unforced growing runs `_11`/`_13` lober
  at protr 1.105/1.137, grip 0.0465/0.056. `n_tubes` 0 on every unforced run.
- **Grip needs a growing tissue — reconfirmed.** `r024_04` (2000c, grip 0.00067) and `r024_01` (P4
  extinct, grip 5e-05) are both spheres.
- Aspect prediction refuted (`r024_06` `protrusion_aspect_max_peak` 1.617 vs >5): forced arms are fat
  lobes. Eye's "first genuine multi-finger" (`r024_07`, aspect 4.332, 6 tubes) is FORCED, grip 0.243 <
  ctrl 0.262 — no new mechanism. No replicate bug; four execution losses (08/09/10/12).

## Round r025
- **6th confirmatory round on the closed forced-star map — nothing new.** Forced coarse cluster
  `r025_03`/`_04`/`_05` (mp 1.748/1.77/1.761) grip 0.240/0.216/0.238, all below ctrl 0.2617; growth
  dilutes (`_04` 8259c → lowest grip 0.216). Coarser-grips-harder holds: n_spots_final 14/15/20 →
  0.240/0.238/0.216 (forced) vs unforced 37/106 → 0.0527/0.065.
- **Grown-protrusion wall stands a 25th round:** best unforced `r025_15` (mp 0, 7905c) protr 1.17 /
  grip 0.065 — a 106-spot lobed ball; `r025_11` (4388c) protr 1.115 / grip 0.0527. n_tubes 0 on every
  unforced run.
- **Grip needs a growing tissue — reconfirmed.** `r025_13` rho→0 (2005c, no growth, grip 0.0035,
  reticulated maze on a rigid sphere) and `r025_14` (P1 broken, 40% volume loss, v_cell 0.14, grip
  0.0088) are both spheres.
- **A P1 break can be invisible in geometry** (`r025_14`): tissue volume 485.7→283.6 (−40%) with the
  ball the same apparent size and every shape metric normal — read P1, not the shape, on shrink runs.
- **Replicate bug a 19th round — TWO copies** (`r025_02` AND `r025_06` == ctrl to every digit, protr
  1.595, grip 0.2617); their "confirmed" act_cv>1 / grip>0.1 are self-comparisons. `spot_spacing_cells_peak`
  refuted (`r025_03` 19.61 vs >21) — ~20 cells is the coarse-field spacing ceiling. Four execution
  losses (08/09/10/12).

## Round r026
- **7th confirmatory round on the closed forced-star map — nothing new.** Forced coarse cluster
  `r026_04`/`_07` (mp 1.77/1.626) grip 0.216/0.207, both below ctrl 0.2617; growth dilutes
  (`_04` 8259c → grip 0.216, `_07` 7991c → 0.207). Coarser-grips-harder holds: n_spots_final
  11/13/20/123 → grip 0.2617/0.207/0.216/0.0554 (ctrl/`_07`/`_04`/`_13`).
- **Grown-protrusion wall stands a 26th round:** best unforced `r026_13` (mp 0, 8144c) protr 1.17
  / grip 0.0554 — a 123-spot lobed ball, reduced_volume 0.83, n_tubes 0.
- **Reaction non-finite again (`r026_14`/`_15`, P4+P12 broken):** act_max diverges to 2.5e28/4.8e34
  then NaN ~1% into the run, mechanics never fire, 2001c static sphere, grip ~2e-5.
- **Replicate bug a 20th round — TWO copies** (`r026_05` AND `r026_06` == ctrl to every digit,
  protr 1.595, grip 0.2617); their "grip>0.09 confirmed" are self-comparisons. Five execution
  losses (08/09/10/11/12) — worst slot yield on file: 7 of 15 slots produced no new science.

## Round r027
- **8th confirmatory round on the closed forced-star map — nothing new.** Forced coarse cluster
  `r027_01`/`_02`/`_05`/`_07` (mp 1.591/1.683/1.761/1.626) grip 0.215/0.228/0.238/0.207, all below
  ctrl 0.2617; growth dilutes (cells 6515–7991 vs ctrl 6143). Coarser-grips-harder holds:
  n_spots_final 11/14/15/21/136 → grip 0.2617/0.228/0.238/0.215/0.1244.
- **High forcing does NOT buckle or finger a FINE field.** `r027_13` (=`_14`=`_15`, one run) mp
  2.266 on a 136-spot fine field: grip 0.1244, protr 1.348, reduced_volume 0.6525, P11 held,
  n_tips 0 — a spiky ball, not the buckle r021 got at mp 2.023 on a coarse field. Buckling and
  fingering depend on field coarseness, not forcing magnitude.
- **Grown-protrusion wall untested this round** — no mp 0 run; all 5 real slots forced (mp
  1.59–2.27). Replicate/duplicate bug a 21st round: `_03`==ctrl, `_13`/`_14`/`_15` triple-identical.
  Five execution losses (08–12).

## Round r028
- **9th confirmatory round on the closed forced-star map — nothing new.** Forced cluster `_03`/`_05`
  (mp 1.796/1.857) grip 0.1888/0.0899, both below ctrl 0.2617; growth dilutes (`_03` 7559c → 0.1888).
  Coarser-grips-harder holds: n_spots_final 11/20/30 → grip 0.2617/0.0899/0.1888 (ctrl/`_05`/`_03`;
  `_05`'s 20-spot field is lower-growth/less-coarsened, act_cv 1.66).
- **A no-division tissue GROWS by inflating cell volume (v_cell 0.24→0.54) and still makes no
  finger.** Unforced `_02` (2000c, 5 domains, mp 0) reaches protr 1.215 / grip 0.078 — round's best
  unforced, a coarse BULGE; `_13`==`_14`==`_15` (18-spot high-act, act_max 0.806) protr 1.063 / grip
  0.0183. Cell-volume growth is NOT a substitute for forcing.
- **Grip needs a growing tissue — reconfirmed 3 ways.** `_01` rho→0 (2000c, grip 0.0007), `_04` P4
  extinct (2201c, grip 0.00011), `_07` activator fully extinct (act_cv 0, n_spots 0, 2188c) — spheres.
- **Grown-protrusion wall stands a 27th round:** best unforced `_02` protr 1.215, n_tubes 0 on every
  unforced run.
- **Dead-field second-channel flag recurs (`_07`):** act_* reads extinct while the eye sees a coarse
  two-tone field coarsening every frame — same substrate read as r001_00/r014_05. Replicate/duplicate
  bug a 22nd round (`_06`==ctrl; `_13`/`_14`/`_15` triple). Five execution losses (08–12).

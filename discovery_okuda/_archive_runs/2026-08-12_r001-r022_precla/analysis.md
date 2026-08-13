# Round r001

## 1. What happened
Control `r001_00`: a static sphere, `protr_peak` 1.014, 2001 cells, no growth and no
reaction operator — yet the eye reports a stable 4–6-domain orange pattern while
`act_alive_frac`/`act_max`/`n_spots` all read 0.0 (see Surprises). Every Route-B run
(`r001_01`–`08`, `10`–`15`) is the SAME morphology: a growing (2000 → up to 19,421 cells)
ball that roughens into broad lobes/undulations. `protr_peak` spans 1.152–1.342, `grip_peak`
0.075–0.101. The eye is unanimous across all 13: **bulges over spots, no finger, no tube** —
and it contradicts the `n_tubes` metric (1–3) on nine of them. `r001_09` is lost (empty
metrics; chemistry NaN from frame 1635).

## 2. What was learned
- 4 confirmed / 3 refuted, but the confirmations are weak. All three `protrusion_aspect_max_peak
  > 0.5` passes (`r001_01` 0.795, `r001_03` 2.687, `r001_04` 1.742, `r001_15` 5.909) have
  `protrusion_aspect_max_final` **0.0**: the aspect signal is a single-frame flicker, not a
  sustained protrusion, and the eye calls every one a lobe. `protrusion_aspect_max_peak` does
  not discriminate a finger — treat it like `protr`.
- `r001_06` refuted `protr_peak > 1.25` (1.22) and `r001_07` refuted aspect > 0.5 (0.0): more
  growth (19k cells) does not buy elongation; it buys more lobes.
- Campaign bests this round, all on lobed spheres: `protr_peak` 1.342 (`r001_15`), `grip_peak`
  0.101 (`r001_06`), `n_tubes_peak` 3 (`r001_11`, `r001_15`). None is a tube.

## 3. What went wrong
- `r001_09`: integrator blew up (`P12`, chemistry not finite from frame 1635); one slot lost.
  The eye flagged it; metrics are empty, not zero.
- `n_tubes`/`tube_diam` fire on 9 runs the eye reads as tube-free — the tube classifier reads
  lobe asymmetry as a tube. Believe the eye and `n_tips` (0 almost everywhere).
- Control chemistry is unmeasured while visibly patterned (see Surprises) — a measurement gap
  on the one run every difference is judged against.

## 4. What to do next
- **Fewer, larger spots.** `n_spots_peak` ran 144–338 where Okuda's Fig. 5 wants ~10 of ~10
  cells. Every finger target is blocked until the pattern coarsens; F009 (no `dx`/χ scale)
  stands. Propose the coarsening knob before any more forcing.
- **Push `cell_die.max_mark_frac` and `K_bend` past this round's window** — both raised
  `protr` monotonically here with no premise broken; find where each breaks.
- **Fix or flag the control chemistry read** before resting a grip claim on it.

# Round r002

## 1. What happened
Control `r002_00`: a lobed, deeply-invaginating body (eye), NOT a sphere — `protr_peak` 1.245,
`grip_peak` 0.090, `corr_act_rad_peak` 0.763, 17964 cells, `n_spots_peak` 187, `invagination` 0.396.
Two runs break from the lobe family: `r002_01` and `r002_04` are two-lobe BUDS at LOW cell count
(3463 / 5468 — division suppressed), with `n_spots_final` collapsed to 9 / 17 (spots coarsened),
and they hold the round's — and the campaign's — best coupling: `corr_act_rad_peak` **0.928 / 0.933**
(vs control 0.763), `grip_peak` **0.136 / 0.141** (prior best 0.101), `protr_peak` 1.357 / 1.405.
The eye calls both fat buds, not fingers (aspect_final 0.0, L3 holds). The high-growth runs go the
other way: `r002_07` at 41865 cells is a faceted tetrahedron, grip down to 0.087.

## 2. What was learned
- **Cell count IS the coarsening lever, and it is inverse to grip.** Fewer cells → few large spots →
  budding with strong grip; more cells → many spots → faceted sphere. grip_peak vs cells: 3463→0.136,
  5468→0.141, 22245→0.068, 41865→0.087. This is the round's headline (L4).
- 6 refuted / 1 confirmed. All 4 `n_spots_peak` predictions refuted because `n_spots_peak` **rails at
  exactly 100** in 7 runs — a measurement cap, not a count (see Surprises). `r002_06` confirmed
  cells<2500 (2000, division off).
- **`max_mark_frac` reverses direction at higher cap:** 0.01→protr 1.321, 0.02→1.246. L2's monotone-up
  claim (0.001–0.002) does not extend; protr peaks near 0.01 then falls.

## 3. What went wrong
- `r002_02`: P4 broken — chemistry extinct after one flash, dead growing sphere, `mech_p_ratio` 0.
- `r002_08`: lost. Metrics empty; eye reports chemistry non-finite from frame 1635 — a NaN crash
  (same failure as `r001_09`), not a zero.
- `n_apop` railed high on `r002_13` (2438) and `r002_12` (1103) yet `invagination_peak` only 0.37 —
  apoptosis active, no inward fold (reconfirms r001).

## 4. What to do next
- **Push division-suppressed budding.** `r002_01/04` (low cells, coarse spots, grip ~0.14) are the
  best-coupled specimens the campaign has. Drive cell count DOWN further and test whether the two
  buds neck into fingers, or set `bud` parents from them.
- **Retune `max_mark_frac` around 0.005–0.01** — the protr peak sits there, not higher.
- **Fix the `n_spots` cap at 100** before resting any spot-count prediction on it.

# Round r003

## 1. What happened
Control `r003_00`: a 4-cusped undulating body (eye), not a sphere — `protr_peak` 1.245,
`grip_peak` 0.090, `corr_act_rad_peak` 0.763, `invagination` 0.396, 17964 cells. Three runs
break the lobe family, all TWO-LOBE BUDS at low cell count: `r003_07` (5302 cells), `r003_06`
(3518), `r003_04` (8773). They hold the round's best coupling — `grip_peak` 0.137/0.136/0.134
(vs control 0.090), `corr_act_rad_peak` 0.935/0.919/0.886 (vs 0.763), `invagination` 0.585/0.580
/0.394, `protr_peak` 1.400/1.344/1.293. Eye calls all three fat buds, no finger (aspect_final 0,
L3 holds). Three runs (`r003_02/03/05`) died — P4 broken, chemistry flashed once to 5.6e-45. Four
lost (`r003_11`–`14`; `r003_11` NaN-blows-up, P12). Route A = `r003_08/09/10/15`.

## 2. What was learned
- 1 confirmed / 6 refuted. `r003_07` confirmed `invagination_peak > 0.45` (0.585) — the campaign's
  deepest inward fold, on a bud. `r003_06` refuted `grip>0.14` at 0.13617 — a whisker short, not a
  miss; the low-cell bud route sits at grip ~0.135, a ceiling three runs now share.
- **L4 reconfirmed and extended.** grip/coupling peaks at low cell count: 3518/5302/8773 cells →
  grip 0.136/0.137/0.134, corr 0.919/0.935/0.886; the high-growth Route-A runs (11936/30134) sit
  at 0.080/0.087. No run inverts it.
- **`mech_p_ratio` 2.2–2.5 on the buds** (`r003_06` 2.234, `r003_07` 2.486) with no extrude operator
  in the space — these are GROWN, not forced (per user §3); the >2 stand-in is off and irrelevant.

## 3. What went wrong
- `r003_02/03/05`: P4 broken — chemistry extinct after one flash (act_mean_floor 5.6e-45), dead
  growing spheres, `mech_p_ratio` 0. A recurring death-twin/settings failure, three this round.
- `r003_11`–`14` lost: `r003_11` NaN from the final frame (P12, same crash as r001_09/r002_08).
- `n_spots_peak` rails at exactly 100 again on `r003_01`–`07`; `r003_10` escapes to 463.

## 4. What to do next
- **Neck the buds.** `r003_07/06` (low cells, grip ~0.135, invagination ~0.58, aspect_peak 3.169/
  2.036) are the best-coupled bodies the campaign has. Drive cell count lower or add an inward lever
  and test whether the two lobes neck into fingers — every run stops at a bud.
- **Chase r003_01's elongation.** gyr_prolate_peak 4.802 (3.6× control) is a real whole-body stretch
  with grip collapsed — find what elongates it and couple that to a spot.
- **K_bend peaks near 0.12** (Route A): stop sweeping it upward.

# Round r004

## 1. What happened
Control `r004_00`: the 4-cusp invaginating body, `protr_peak` 1.245, `grip_peak` 0.090,
`invagination` 0.396, 17964 cells. One run breaks the family: `r004_06` — classifier `tube`,
`protr_peak` 1.454 (campaign best on an admissible run), `protrusion_aspect_max_peak` 3.411 with
`aspect_final` **2.207** (the FIRST sustained non-zero aspect_final in the campaign), `n_tips_final`
1, `corr_act_rad_peak` 0.954, `act_cv_peak` 8.742, at only 4121 cells / `red_frac` 0.116 (one
dominant spot). `mech_p_ratio` 2.965 — grown, not forced. **The eye DISAGREES:** a fat red-capped
pear, aspect ~1.09, "a bulge, not a finger." Two runs lost: `r004_11` (NaN, P12) and `r004_12`
(empty). `r004_02` P4-dead.

## 2. What was learned
- 1 confirmed / 6 refuted. `r004_06` confirmed `grip<0.141` (0.135); `r004_07` refuted `protr>1.293`
  by 0.001 (1.292 — a rail-adjacent miss, not a finding).
- **A second route to grip, at HIGH cell count.** `r004_07` reaches the round's top `grip_peak`
  0.14649 at **14809 cells** — via invagination (`r_cv` 0.217, `reduced_volume_final` 0.553 the
  campaign deepest), not coarsening. L4 (grip from few coarse spots) still holds for the low-cell
  route but is no longer the only path; deep folding at high N reaches the same grip.
- `r004_06` pushes the low-cell attractor one spot further than r003's two-lobe buds: a single
  red-capped lobe at grip 0.135, the shared ceiling. Necking still does not happen (eye, L3).

## 3. What went wrong
- `r004_11` NaN from frame 1635 (P12, 4th such crash: r001_09/r002_08/r003_11); `r004_12` empty.
- `r004_02` P4 broken — chemistry flashed to 5.6e-45, `mech_p_ratio` 0, dead sphere.
- `n_spots_peak` rails at 100 again (`r004_02/05/06/08/14/15`).
- `protrusion_aspect_max_peak` 7.742 (`r004_10`) and `n_tubes_peak` 5 (`r004_09/10`) both fire on
  bodies the eye calls pointed-star lobes, `aspect_final` 0 — L3 holds, these metrics still lie.

## 4. What to do next
- **Neck `r004_06`.** Lowest cells / one spot / sustained aspect_final 2.207 is the closest the
  campaign has come; add an inward/purse lever (`interface_tension`, `cell_die`) at this
  low-N one-spot state and test whether the lobe necks.
- **Combine the two grip routes:** low-N coarsening (`r004_06`) + high-N invagination (`r004_07`,
  reduced_volume 0.553) are different mechanisms reaching grip ~0.14 — pair-test them.
- **Lambda 0.6 is the coarsening op point** (Route A); stop above it.

# Round r005

## 1. What happened
Control `r005_00` is a STRONGER parent than r001–r004: `protr_peak` 1.277, `grip_peak` 0.11957,
`corr_act_rad_peak` 0.815, 10749 cells, `invagination` 0.359 — a knobbly lobed ball (eye). `r005_06`
is bit-identical to it (replicate, seed floor ~0). One run beats it: **`r005_04` — `grip_peak`
0.17961 (round best, +50% over control), `invagination` 0.5202, `reduced_volume_final` 0.5339
(campaign-deep), `corr_act_rad_peak` 0.876, 9260 cells, a 5–6-armed undulating star (eye).** It is
the round's `complex`-target result. `r005_01` goes the opposite way: `gyr_prolate_peak` **4.619**
(one fat directional bud) but `grip_peak` collapses to 0.0158. `r005_08` marginally above control
(protr 1.348, grip 0.124). Route A = `r005_11`–`15`, `r005_08`, `r005_15`. Two lost:
`r005_09` (NaN, P12, frame 1635) and `r005_10` (empty).

## 2. What was learned
- **0 confirmed / 5 refuted — every Route-B prediction missed against the stronger control.**
  `r005_04` refuted aspect>1.0 (0.498), `r005_03` and `r005_06` refuted grip>0.15 (0.084 / 0.120),
  `r005_01` refuted aspect>3.5 (0.701), `r005_02` refuted n_tubes>6 (0). The aspect/tube targets are
  as blocked as ever; L3 holds (every aspect_peak fire has aspect_final 0, eye sees lobes).
- **`r005_04` is a real grip advance and it is an undulation, not a bud.** grip 0.180 at high cell
  count (9260) via deep invagination (`reduced_volume` 0.534) — the r004_07 high-N folding route, now
  the campaign grip best. Serves `complex`, not `tube`.
- **Elongation and grip are anti-correlated (again).** `r005_01` gyr_prolate 4.619 with corr_act_rad
  0.946 but grip 0.0158 — whole-body stretch normalises the amplitude away, mirroring r003_01
  (gyr_prolate 4.802, grip collapsed). A prolate egg is not a gripped tube.

## 3. What went wrong
- `r005_09` NaN from frame 1635 (P12) — the 5th such crash (r001_09/r002_08/r003_11/r004_11);
  `r005_10` empty. Two slots lost.
- No P4 deaths this round — the death-twin failure did not recur.
- `n_tubes`/`tube_diam` fire on control (3) and 5 runs the eye reads as tube-free lobes; `n_tips` 0
  everywhere. Metrics still lie; believe the eye.

## 4. What to do next
- **Push `r005_04`'s invagination route.** grip 0.180 / reduced_volume 0.534 via deep folding at high
  N is the campaign's best coupling. Add an inward lever (`cell_die`, `interface_tension`)
  on this parent and test whether an arm necks.
- **Stop sweeping vth_frac** — inert across 4/6/10 (protr 1.344/1.348/1.348). Closed.
- **Re-anchor: the control changed.** r001–r004 grip ceiling ~0.14 is superseded by this parent
  (0.120 baseline, 0.180 on r005_04); re-baseline grip predictions on r005_00, not the old family.

# Round r006

## 1. What happened
Control `r006_00`: lobed growing ball, protr_peak 1.277, grip_peak 0.11957, 10749 cells (=r005_00).
Every Route-B run made a lobe/bud/star, none a finger. The round's best coupling, `r006_06`, is a new
campaign grip best (0.18254 > r005_04 0.17961), a 5–6-lobe star at 8512 cells via folding
(reduced_volume 0.526, invagination 0.5136, corr_act_rad 0.872). The low-N two-lobe buds returned:
`r006_04` (5468 cells, protr 1.405, aspect_max_peak 2.672, grip 0.141) and `r006_11` (2661 cells,
protr 1.432, aspect_max_peak 3.09, grip 0.151), both red-capped dumbbells.

## 2. What was learned
All 7 scored predictions REFUTED — but three narrowly, and each names a bud/star that is a legitimate
parent by its own target seat. `r006_06` grip 0.18254 vs threshold 0.19 is a MISS on the number and a
campaign RECORD on the metric. `r006_04` aspect 1.854<2 and `r006_11` aspect 3.09 both remain buds
(aspect_final 0, L3 HOLDS). `r006_03` apoptosis (n_apop 117) reached invagination 0.5503<0.60 —
deepest fold of the round, inward lever works but under-shoots. Chemistry couples but never necks:
protr and grip both live on lobes/buds, never a sustained protrusion.

## 3. What went wrong
`r006_02` P4 death (chemistry flashed once, mech_p_ratio 0). `r006_09` NaN crash frame 1635 (6th of
the family); `r006_13`/`r006_15` empty. `r006_12`==`r006_14` bit-identical — cell_divide.factor 5 and 8
measure the same 2000-cell tissue (saturated above factor 3): one wasted rung.

## 4. What to do next
- **Combine the two best routes.** `r006_06` folding-star (grip 0.183) + `r006_03`'s apoptosis inward
  lever on ONE parent — does an arm neck instead of lobe? Predict invagination >0.55 or a non-zero
  aspect_final.
- **Push K_bend above 0.3 on b_gs_shaping_soft_lo**, but watch act_cv (already 11.67 at 0.3) — a
  chemistry-death ceiling is near. One rung at 0.5.
- **Stop sweeping cell_divide.factor** — inert above 3; closed at 3 values.

# Round r007

## 1. What happened
Control `r007_00`: lobed sphere, `protr_peak` 1.277, `grip_peak` 0.11957, 10749 cells, 154 spots,
no protrusion (aspect_final 0). Route B (6 predictions) went **0/6 confirmed** — every prediction
refuted. Two slots empty (`r007_08` NaN crash frame 1635; `r007_14`). The signal is in the aspect
finals, not the predictions: three low-N runs held a sustained `protrusion_aspect_max_final`
(`r007_05` 3.044, `r007_06` 2.959, `r007_03` 1.636) with n_tips 1–2 — the eye calls all three fat
buds over a red domain, no finger.

## 2. What was learned
- `r007_02` n_tubes>6 refuted (1): coarsening measure catches a lobe, not a tube.
- `r007_03` grip>0.15 refuted (0.053), `r007_06` grip>0.20 refuted (0.078): at low N grip stays low
  even when aspect_final fires — sustained aspect and grip are decoupled.
- `r007_04` gyr_prolate>2.5 refuted (1.024) on a P1+P4-broken specimen: chemistry died, no shape.
- `r007_05` n_spots<20 refuted on peak (100 rail) but n_spots_FINAL is 5 — the round's best pattern
  scale; the prediction lost to the 100-spot ceiling, not to the tissue.
- `r007_07` protr>1.4 refuted (1.284): coarsening the control does not reach a finger.

## 3. What went wrong
Two empty runs. `r007_08` diverged to NaN at frame 1635 (7th identical P12 crash) — any metric
averaging it is contaminated; the eye caught it, the pipeline did not. `r007_04` broke P1 and P4:
the deliberate ablation (rho 0 / chem off) killed the field, act_cv railed 23.77.

## 4. What to do next
- **Combine `r007_05`'s few-large-spot state (n_spots_final 5) with the folding lever.** It is the
  scale closest to Okuda and already holds aspect_final 3.044. Predict aspect_final >2 with grip >0.12.
- **Adopt a_sw 0.1 as the coarsening knob** — grip 0.107 at 18632 cells, monotone; stop sweeping it.
- **Drop K_lumen** — near-inert, lowers grip. Do not sweep further.

# Round r008

## 1. What happened
A thin, Route-A-only round: 5 real runs, slots 05–08 empty. Control `r008_00`: `protr_peak`
1.168, `grip_peak` 0.078, 3397 cells, a ~7-lobe spotted sphere (eye), no finger. Every run a
lobed/undulating ball; the eye calls three of them undulations the classifier mislabels `sphere`.
`n_tips_peak` ≤1 everywhere, `protrusion_aspect_max_final` 0 everywhere. No tube, none near one.

## 2. What was learned
No Route-B prediction was posed, so nothing scored. The two sweeps ARE the round.
- `cell_grow.hill` on `b_gs_plain_soft_lo` is near-INERT: 1/2/8 → protr 1.151/1.168/1.188,
  grip 0.073/0.078/0.077, cells 7887/6998/7480. Faint monotone protr rise, grip flat. Close it.
- `cell_mechanics.K_lumen`=2 on `b_gs_shaping_soft_lo` (`r008_01`): protr 1.279, grip 0.0915,
  10296 cells, red_frac 0.43 — round best on all three, but a faceted tetrahedral blob (eye).
  This CONTRADICTS the r007 K_lumen closure ("inert, lowers grip, use 0"): at 0/0.1/2 →
  grip 0.083/0.070/0.0915, protr 1.261/1.247/1.279 — non-monotone, 2 slightly beats 0.

## 3. What went wrong
Four empty slots (05–08), no .err shown — half the round produced nothing; treat as execution
loss, not biology. `r008_02` and `r008_03` report `mech_p_ratio` 0.0 with chemistry intact
(act_max 0.82) — a stalled ratio, not a dead field.

## 4. What to do next
- **Stop sweeping `cell_grow.hill`** — inert (3 values).
- **Re-open K_lumen upward**, since 2 mildly beats 0 on `b_gs_shaping_soft_lo` — one more rung
  (5) to see if it is a real weak riser or noise; predict grip <0.11.
- **Escape Route A.** Six rounds of sweeps on plain/shaping recipes have not cleared protr 1.3.
  Spend Route-B slots on the few-large-spot state (a_sw 0.1) crossed with folding, not on knobs.

# Round r009

## 1. What happened
Control `r009_00`: `protr_peak` 1.168, `grip_peak` 0.078, 3397 cells, a ~6-lobe spotted ball,
no finger. The round's two strongest specimens are multi-armed STARS at high cell count:
`r009_05` (9038 cells) protr 1.454 / grip **0.190** / invagination **0.566** / gyr_prolate 1.429,
and `r009_02` (9023 cells) protr **1.479** / grip 0.183 / invagination 0.489, n_apop 98. The eye
calls both 3–6-armed red-tipped lobed stars — real outward arms (aspect ~1.9), NOT thin fingers;
`protrusion_aspect_max_final` 0 on both. Everything else is a lobed sphere or a dead-chemistry ball.

## 2. What was learned
- **r009_05 CONFIRMED protr_peak>1.45 (1.454)** — the only scored prediction that held. Its grip
  0.190 and invagination 0.566 are both campaign bests (prev 0.183 r006_06, 0.550 r006_03), via the
  high-N folding route (r004_07/r005_04/r006_06), and grown not forced (mech_p_ratio 2.155).
- **r009_02 REFUTED invagination>0.55 (0.489)** yet is the round's highest protr (1.479); the fold
  fell just short but the arms are the campaign's strongest. Two big stars, no tube.
- Three predictions (r009_01/03/07) refuted BY CHEMISTRY DEATH not by mechanism: P4 broken,
  activator extinct, protr rails 1.02, mech_p_ratio 0 — a settings failure, unscorable as biology.
- r009_04 (aspect_final 1.699, n_tips 1, morphology 'tube') and r009_06 (aspect_peak 1.224) both
  refuted their aspect thresholds; eye sees lobes, aspect_final≈0 → L3 holds.

## 3. What went wrong
`r009_09/10/11` empty. `r009_10` = P12 NaN blow-up from frame ~1562 (8th such crash: r001_09/
r002_08/r003_11/r004_11/r005_09/r006_09/r007_08) — a crash, not a zero. Chemistry deaths in
r009_01/03/07 (act_mean_floor 5.6e-45, mech_p_ratio 0) wasted three Route-B slots.

## 4. What to do next
- **Lift r009_05, the new best, as a parent** (star, grip 0.190) and REPLICATE it at a second seed
  to bound the noise — its grip/invagination records are single-seed.
- **Sweep cell_grow.rate upward** — it drove cells 3486→22565 and grip 0.041→0.105 monotone over 4×;
  the break is untested. Predict grip <0.12 (folding ceiling).
- **Stop the chemistry deaths.** Three of eight Route-B slots died on activator extinction; diagnose
  before spending more.

# Round r010

## 1. What happened
Control `r010_00`: protr_peak 1.168, grip_peak 0.078, 3397 cells, a ~6-lobe spotted ball — the
WEAKEST parent family run so far (r005–r009 controls sat at grip ~0.12). `r010_06` is bit-identical
to it → seed floor ~0. No run this round reaches a tube, a finger, or a sustained aspect: the two
notable specimens are a whole-body prolate egg (`r010_01`, gyr_prolate 4.923, protr 1.377, grip
0.140) and a single fat BUD on a one-spot pattern (`r010_05`, corr_act_rad 0.935, act_at_tip 9.796,
2242 cells). Everything else is a lobed/potato ball or a dead-chemistry sphere.

## 2. What was learned
- **Every aspect prediction refuted; no finger exists.** r010_01/02/05/06 all predicted
  aspect_peak >1.8–4.2 and all fell (0.8 / 1.086 / 1.41 / 4.136), aspect_final 0 everywhere, n_tips
  0. L3 HOLDS — protrusion_aspect_max_peak and n_tubes still do not discriminate a finger from a
  lobe or a stretch.
- **r010_01's grip 0.140 is elongation, not coupling.** gyr_prolate 4.923 with the eye reading a
  smooth prolate egg — a global stretch that inflates grip's r_cv term while corr falls; reconfirms
  elongation ⊥ local grip (r003_01/r005_01).
- **r010_07 CONFIRMED protr_peak<1.25 (1.014)** — the only held prediction: a patterned sphere that
  never deforms (grip 0.003). A clean null.
- **r010_05 is the round's Okuda-closest pattern:** one spot, act_at_tip 9.796, act_cv 11.29 — a
  coarse single-domain bud, still aspect_final 0.

## 3. What went wrong
`r010_08/09/10/15` empty. `r010_11` = P12 NaN crash from frame 1562 (9th such: r001_09/r002_08/
r003_11/r004_11/r005_09/r006_09/r007_08/r009_10). `r010_03` chemistry death (P4 broken,
act_mean_floor 5.6e-45, act_cv 33.97 rail, mech_p_ratio 0) — a settings failure, unscorable. Two
cell_grow.rate Route-A rungs (0.003464/0.006928) crashed to empty: rate CRASHES b_gs_plain_soft_lo
above ~0.0017 (r009 ran clean at 0.001732).

## 4. What to do next
- **Abandon this weak recipe family as a parent line** (control grip 0.078, half the r009 family);
  return to the high-N folding stars (r009_05 grip 0.190) that this round did not touch.
- **Cap cell_grow.rate at 0.001732 on b_gs_plain_soft_lo** — higher values 0-archive; stop sweeping up.
- **K_purse 0.25 on b_gs_shaping_soft_lo is the protr optimum** (protr 1.263 vs 0→1.186, 3→1.214);
  test it as a Route-B ingredient rather than sweeping further.

# Round r011

## 1. What happened
Control `r011_00`: same weak family as r010 — protr_peak 1.168, grip_peak 0.078, 3397 cells, a
lumpy ball of shallow domes (eye), n_tubes 1 disputed. Two Route-B runs blew past it: `r011_04`
protr_peak **1.598** / grip 0.184 / invagination_peak **0.593** / 4762 cells / mech_p_ratio 2.586
(grown), a 3–4-armed red-tipped star (eye); `r011_06` grip_peak **0.189** / reduced_volume_final
**0.459** / invagination 0.561 / 13172 cells / red_frac 0.434, a 4–5-armed star. `r011_03`
protr_peak 1.354 / grip 0.138 / mech_p_ratio 2.608 at 3282 cells — 5–6-lobe bud. All other slots
were Route A or chemistry deaths.

## 2. What was learned
- **`r011_04` protr 1.598 (>1.479) and `r011_03` 1.354 (>1.3) confirmed** — the fresh campaign's two
  strongest protrusions, both grown (mech_p_ratio 2.6), both multi-arm stars, aspect_final 0. No
  finger (L3 holds): `r011_04` aspect_peak 6.097 flickers, ends 0; eye sees broad tapering arms.
- **`r011_06` n_tips>5 refuted (n_tips_peak 0)** despite the eye counting 4–5 red arm-tips — the tip
  detector does not fire on a broad-armed star, exactly as morphology='sphere' mislabels it. The
  branched target is being made and not measured.
- Chemistry deaths (`r011_01/02/05`) refuted their predictions: P4/P1 broken, protr rails ~1.01–1.06,
  a settings failure, not biology.

## 3. What went wrong
- `r011_07/08/11` empty. `r011_09/12/15` are **b_bru_question rho** — they RAN this time (contrary to
  r001–r010 "empty/crashed") but P4+P12 broke: act_cv 244/216/211 (NaN blowup), grip 0.003, protr
  ~1.02. Enormous inert tissue (60066/47291/45352 cells), chemistry diverges every value.
- `r011_13` (K_V 5) P1 broken: 2010 cells flat, no growth, chemistry near-dead (act_cv 21.9, one spot).

## 4. What to do next
- **`r011_04` and `r011_06` are the round's parents** — grown red-tipped stars, the best branched/
  complex specimens yet (invagination 0.593, reduced_volume 0.459 both campaign records). Add
  `cell_die` to `r011_04` to test whether an inward fold narrows the broad arms toward tips.
- **Retire b_bru_question outright** — not empty but inert-NaN at every rho; it cannot pattern.
- **Do not sweep K_purse further** — grip peaks at 0.25 (0.083) and falls monotone to 6 (0.058).

# Round r012

## 1. What happened
Control `r012_00`: the WEAK family again — protr_peak 1.168, grip 0.078, 3397 cells, a lobed spotty
sphere (eye), n_tubes 1 disputed. Four scored Route-B runs, all aimed at the grip/invagination
ceiling, all refuted or dead:
- `r012_05` invagination<0.489 **refuted** (0.566): protr_peak 1.454, grip_peak **0.1903** (ties the
  campaign best, r009_05 0.190), 9038 cells, n_tubes 6, reduced_volume 0.534 — a 5–6-armed red-tipped
  star (eye), aspect_final 0. Round's strongest coupling.
- `r012_06` grip>0.189 **refuted** (0.180): protr 1.419, 16680 cells, reduced_volume_final 0.4519,
  red_frac 0.474 — a 4–6-cusp star with DEPLETED white tips (red_at_tip 0.209).
- `r012_02` grip>0.19 **refuted** (0.086): 6432 cells, invagination_peak **0.60786**, gyr_prolate_peak
  3.073 — a bilobed cleft heart (eye), grip collapsed.
- `r012_04` grip<0.05 **confirmed** (2e-05) but on a P4-dead specimen (K_V 10 froze growth, chemistry
  flashed then died) — a null, not a shape result.

## 2. What was learned
- **The ~0.19 grip wall holds.** Three deliberate attempts to clear 0.189/0.19; best 0.1903 (r012_05)
  is the wall itself (r009_05 0.190, r011_06 0.189). High-N folding route; no new mechanism to exceed
  it was tried.
- **Fold DEPTH and grip AMPLITUDE ride different bodies.** Campaign-record invagination 0.608 (r012_02)
  and near-record reduced_volume 0.452 (r012_06) both landed where grip did NOT — 0.086 and 0.180 —
  while grip peaked on a third body (r012_05). Elongation ⊥ grip reconfirmed (r003_01/r005_01/r010_01).
- **L3 holds.** `r012_15` aspect_final 4.04 / n_tips 1 / morphology='tube' — eye reads a lobed sphere
  (broad bulges, aspect 1.205). Sustained aspect ≠ finger.

## 3. What went wrong
- P4 chemistry deaths `r012_04`, `r012_09` (act_mean_floor 5.6e-45, mech_p_ratio 0 — flashed once, died).
- `r012_14` P4+P12 NaN blowup: act_max 3.3e9, 32246 cells (b_bru_question vth_frac 6) — apparatus rail.
- `r012_12` P12 NaN crash (empty metrics; eye saw the blowup) — 10th such crash on file.
- Empty `r012_01/03/08/10/11`; the two inconclusive predictions (aspect, invagination) rode dead slots.

## 4. What to do next
- **Parents:** `r012_05` (grip 0.1903 star, n_tubes 6 → complex seat) and `r012_02` (invagination 0.608
  bilobed heart → bud/branched seat).
- **To beat the 0.19 grip wall needs a NEW mechanism, not a retune** — add `cell_die` to `r012_05`
  and test whether an inward fold pushes grip past 0.19 (folding is the only route that has reached it).
- **Use K_V 40 as a strong Route-A base** (protr 1.322 / grip 0.119, up from a dead K_V 10). Retire
  b_bru_question — NaN at vth_frac 6 as at every rho.

# Round r013

## 1. What happened
The control changed families and got much stronger: `r013_00` protr_peak 1.295, grip_peak 0.118,
12984 cells, invagination 0.411, reduced_volume 0.654 — a 3–4-lobe undulating body (eye), not the
r010–r012 weak 3397-cell family. `r013_07` is bit-identical (seed floor ~0), so every difference is
real. Six of 16 slots produced nothing: `r013_02/03/09/10/11` empty, `r013_13` a P12 NaN crash.
Three runs beat the control decisively, all high-N folding stars:
- `r013_05` grip_peak **0.273**, invagination_peak **0.617**, reduced_volume_final **0.285**,
  protr 1.408, n_tubes 11, corr_act_rad_peak 0.921, mech_p_ratio 2.196, 12201 cells — a 7–8-armed
  red-tipped star (eye).
- `r013_04` grip 0.179 / protr 1.392 / 13111 cells; `r013_06` grip 0.173 / protr 1.405 / 8240 cells /
  n_tips_peak 7. Both red-tipped multi-arm stars (eye).

## 2. What was learned
- **The ~0.19 grip wall (held four rounds) is broken: 0.273.** It came free — the round posed aspect,
  not grip. It rode the stronger control family, not a new operator; grip 0.118 baseline vs 0.078
  before. Fold depth followed it this time (invag 0.617, reduced_volume 0.285 all on the same body),
  unlike r012 where depth and grip rode different bodies.
- **The fold is NOT apoptotic.** Deepest invagination (`r013_05`) had n_apop 0; the three death runs
  (157/115/87) folded less. The r012 hypothesis — apoptosis pushes grip past 0.19 — is not what did it.
- **Every prediction refuted or dead.** All four aspect/invagination thresholds missed
  (aspect >5.234→0.324, >6.097→0.657/0.738; invag >0.608→0.492); `r013_07` protr>1.303 refuted at the
  control's own 1.295 (a replicate). L3 HOLDS on the metric — but the eye insists r013_04/05/06 are
  the sharpest red-tipped arms yet, tips the aspect/n_tips detectors zero out.

## 3. What went wrong
- 6 of 16 slots lost (37%): five empty, `r013_13` P12 NaN from frame 1352. The two inconclusive
  predictions (r013_02 aspect, r013_03 invagination) rode empty slots.
- The metric bank cannot name the star morphotype: n_tips 0 / aspect_final 0 where the eye sees 5–8
  red-capped arms. The instrument, not the tissue, is the wall.

## 4. What to do next
- **Parent `r013_05`** — the campaign's strongest coupling (grip 0.273 / invag 0.617) → complex and
  branched seats. Replicate it once to bound the seed floor on this new family.
- **Escalate the star morphotype to the eye + n_tubes**; stop scoring these on protrusion_aspect_max.
- **Route A: Lambda ≤0.2** on b_gs_shaping_soft_lo (1 collapses cells to 2688, act_cv 16.8).

# Round r014

## 1. What happened
First round on `b_star` (r013_05 promoted to basis): 7 real Route-B probes of the star + 2 bud/star
siblings, 3 Route-A rungs (slots 11/13/15), 5 lost (08/09/10/12/14; 12+14 are P12 NaN-at-1352).
Control `r014_00` = r013_05 verbatim: protr 1.408, grip 0.273, invag 0.617, 12201 cells, n_tubes 11.
`r014_02` is **bit-identical** to it (seed floor ~0). Nothing beat the control.

## 2. What was learned
- **The star's arms are growth-buckling, not the purse-string.** `r014_01` removed
  interface_tension: n_tubes 11→8, grip 0.273→0.220, protr 1.408→1.337, invag 0.617→0.608,
  still a 6–8-petal rosette (eye). Line tension adds ~3 arms and ~0.05 grip; it is not what makes the
  star. Adversarial, confirmed.
- **The feedback leg is INERT and beta<0 is a KILL SWITCH.** `r014_02` (set_impl cell_chem_from_shape →
  apical_area) is bit-identical to curvature — beta=0, so the impl swap does nothing; the loop is not
  closed on b_star. Turning it on negative extinguishes the reaction: `r014_03` (star, beta −0.5) and
  `r014_07` (bud r011_04, beta −0.5) BOTH break P4, act extinct, invag collapses to 0.031/0.060. The
  sign at −0.5 is not a dimple — it kills the chemistry, morphotype-independent. Both refuted.
- **Halving Gray-Scott kk (0.062→0.031) coarsens to Okuda scale but runs the reaction hot.**
  `r014_04`: spot_spacing 4.04→12.01 cells (confirmed >11.3), yet cells 12201→17182 (+41%),
  red_frac_peak 0.472→0.814, act_max 1.47→2.19, grip 0.247. Eye: 7–9 red-tipped pointed arms, "first
  clear multi-armed protrusion field." kk is the wavelength lever but NOT clean — it also amplifies
  activator and growth.
- **A mechanics leg at low growth does NOT finger r007_03.** `r014_05` (add purse-string) aspect_peak
  1.082 — eye sees a sphere that never protrudes, WORSE than r007_03 alone (aspect_final 1.636);
  `r014_06` (kk 0.031) aspect 1.091, broad lobes. Both refuted >2.152. Grounder's "tube at Okuda's
  growth envelope" hypothesis falsified on both routes. L3 HOLDS.

## 3. What went wrong
- 5 of 15 slots lost (33%); r014_12/14 are the recurring P12 NaN from frame 1352. The whole second
  half of the Route-B menu (r011_04-kk, r009_05×2, r011_06×2, r009_02×2) landed on empty slots — no
  data on the F0 gain sweep or the apoptosis fold-vs-subtraction pair.
- beta −0.5 as a chosen edit destroyed two otherwise-good parents. Sweep beta small (±0.05) or not at
  all until the leg is shown active.

## 4. What to do next
- **Sweep cell_chem_react.kk between 0.031 and 0.062** on b_star to separate coarsening from the hot-run
  side effect — is there a kk that spaces arms ~10 cells WITHOUT the +41% overgrowth?
- **Do not re-run beta<0 at −0.5** on any live parent; it is a kill switch (L5). If the feedback leg
  is wanted, first raise beta from 0 by a small positive step and check the run still patterns.
- **Re-issue the lost F0-gain and apoptosis-subtraction pair** (r009_05 F0 0.023, r009_02 remove
  apoptosis) — the fold-vs-apoptosis question is still unsettled and rode empty slots.
- **Route A: cell_chem_react.F/kk raise grip mildly** (F 0.06→0.091, kk 0.052→0.089 on b_gs_plain); Lambda
  ≥1 still kills (Lambda 2 → cells frozen 2006, P1+P4).

# Round r015

## 1. What happened
Second round on `b_star`. Control `r015_00` = the star verbatim (protr 1.408, grip 0.273, invag 0.617,
n_tubes 11, 12201 cells); `r015_02` is bit-identical (seed floor ~0). Four Route-B probes scored, four
slots lost (08/09/10/11/13 empty), three Route-A rungs (12/14/15). **`r015_06` broke the campaign open:
the first FINGER seen by the eye and the aspect metric at the same time** — protr_peak 2.199, grip_peak
0.344 (both records), `protrusion_aspect_max_final` 7.544, n_tips_final 2, n_tubes 5, 5690 cells.
Everything else consolidated known facts.

## 2. What was learned
- **`r015_06` is the answer the campaign was built to find.** Against a universal prior of
  `protrusion_aspect_max_final` 0.0 across 200+ runs, it reads **7.544**, `n_tips_final` 2, morphology
  'tube', and the eye — independently — calls it "the campaign's first convincing branched protrusion,
  thin red-tipped arms, genuine fingers not bulges," genus intact. grip 0.344 (×1.26 the r013_05 record)
  and protr 2.199 (×1.38) both fall out of it. It sits at only 5690 cells with 3 coarse spots
  (spot_spacing 25.94), act_max 3.41 (control 1.47), act_at_tip 8.95, red_at_tip 0.979 — the chemistry
  chases the extruding tips. This is a THIRD grip route: genuine fingers at low-N, out-gripping both
  low-N coarsening and high-N folding. Predicted grip>0.247, confirmed. **It is single-seed and its edit
  is not identifiable from the metrics I was given** — the tip-chasing hot pattern is the signature of a
  closed geometry→chemistry feedback (cell_chem_from_shape beta≠0) or of cell_chem_seed:cones, both firsts here.
- **Apoptosis subtracts from the fold.** `r015_04` (star death twin, n_apop 77): invag 0.617→0.601,
  grip 0.273→0.250, protr 1.408→1.367. Predicted invag>0.617, refuted. Reconfirms r001–r013 — apoptosis
  is active and does NOT drive the sheet inward here.
- **The star needs growth; its arms are growth-buckling.** `r015_03` (growth off): protr 1.03, 2021
  cells, red_frac 0, grip 0.010 — a bare patterned sphere. Predicted protr<1.1, confirmed. Reconfirms
  r014_01.

## 3. What went wrong
- 4 of 13 slots empty (08–11/13). Given r015_06's importance, the risk is that its sibling probes —
  likely the rest of the b_star_{cones,oriented,tension,pressure,sharp} family — rode those empty slots,
  so we have ONE finger and no idea which of its neighbours also finger.
- `r015_15` (b_bru_question, divide factor 2.4) patterned cleanly (32333 cells, red_frac 0.939, no NaN)
  but `corr_act_rad` −0.210, grip −0.004 — first strong NEGATIVE coupling; the base is reconfirmed
  useless, now for anti-grip rather than NaN.

## 4. What to do next
- **REPLICATE `r015_06` at 2–3 fresh seeds and identify its composition** before anything else. Eight
  single-seed clean points have regressed on replication across this project's history; a first finger
  is exactly the result most likely to be seed-luck, and its edit must be named to be built on.
- **Re-issue the lost sibling slots** (the other b_star variations) — if r015_06 is `cones` or a beta≠0
  feedback, its neighbours in that family are the immediate map to fill.
- **Sweep the mechanism once identified:** if feedback, raise beta from 0 in small positive steps
  (L5: −0.5 is a kill switch); if cones, sweep N to make "how many tubes" a controlled variable.
- **Route A:** cell_chem_react.kk on b_gs_plain is monotone (0.052/0.057/0.066 → cells 10302/8406/6101,
  grip 0.089/0.080/0.074) — use low kk; stop sweeping b_bru_question.

# Round r016 — THE FINGER GENERATOR ISOLATED

## 1. What happened
Control `r016_00` is the STAR `b_star` reproduced verbatim (grip 0.273, invag 0.617,
reduced_volume 0.285, n_tubes 11, 12201 cells; eye: 8-armed star). Every scored Route-B slot is a
DISSECTION of the campaign's only finger `r015_06` (baseline from r015: protr 2.199, aspect_final
7.544, ~5690 cells) or the star `r013_05` — no new base. The payoff is that the finger's generator
is now isolated by ablation. 4 of 13 slots lost (08–10, 13) and the `cones` probe (slot 05) with
them.

## 2. What was learned — the finger, taken apart
- **The purse-string is NOT the generator; the finger is growth-buckled.** `r016_03` REMOVED
  `interface_tension` from the finger and it SURVIVES and STRENGTHENS: protr_peak 1.83,
  `protrusion_aspect_max_final` **21.094** (new campaign record, prev 7.544), n_tips 2, n_tubes 6,
  morphology 'tube', red_at_tip 0.996, mech_p_ratio 2.511 (grown). The eye INDEPENDENTLY: "the first
  true protrusions I have seen — tapering pointed arms, genuine fingers." Predicted n_tubes<5,
  refuted (6) — the refutation IS the finding: tension is a modifier (as on the star, r014_01), not
  the arm-maker.
- **This is a SECOND seed of the finger route — not a single-seed fluke.** r015_06 (seed A) and
  r016_03 (fresh seed, tension removed) both finger; the r015 replication worry is answered, and the
  strongest specimen is the ablated one.
- **The closed `cell_chem_from_shape` feedback leg IS required — and it BRAKES growth.** `r016_04` REMOVED
  `cell_chem_from_shape` → the finger collapses to a bulge (aspect 0.435, protr 1.088) AND growth runs away
  to **50532 cells** (campaign-largest patterning tissue, ~9× the finger's 5690) with the pattern
  washing UNIFORM (act_cv_final 0.0, red_frac 1.0). Predicted aspect<4, confirmed. So on r015_06
  `cell_chem_from_shape` is LIVE (beta≠0): r015_06 is the campaign's first CLOSED-FEEDBACK composition,
  confirming the r015 hypothesis. Its removal un-brakes growth — a direct mechanism for the project's
  growth-overshoot (user §5).
- **Halving the feedback (F0 0.046→0.023) also loses the finger.** `r016_01` protr 1.272 (predicted
  >1.6, refuted), cells 22231, aspect_peak 0.726 — feedback strength is a knee, not linear; the
  finger needs ~full F0.
- **Growth and division are needed, trivially.** `r016_02` (growth off): protr 1.015, 2006 cells,
  pattern faded (predicted <1.3, confirmed). `r016_06` (division off): 2000 cells, P4 broken
  (chemistry died), aspect 1.276 (predicted <4, confirmed).
- **Apoptosis does not invaginate.** `r016_07` (star + cell_die, n_apop 85): invag_peak 0.605 <
  the star's 0.617 (predicted >0.617, refuted). Reconfirms r001–r015.

## 3. What went wrong
- 4 lost slots (08–10, 13) plus `cones` (slot 05 — `.out` exists, no scorecard). The Okuda-Fig-5
  cones probe — the only route that makes tube NUMBER a controlled variable — did not land.
- `r016_11` (b_bru_question vth_frac 4) NaN, P4+P12, act_max 3.3e9 — apparatus rail, base ruled out.

## 4. What to do next
- **Build the next round from `r016_03`, not r015_06** — it holds the aspect record 21.094 with the
  purse-string already removed, so the composition is simpler and stronger.
- **Sweep `cell_chem_from_shape.F0` (and beta, both signs) from r016_03** — the leg is now identified as the
  finger's generator. Predict aspect rises with F0 to a ceiling, collapses at F0=0 (r016_04) and at
  half (r016_01); L5 warns beta −0.5 is a kill switch, so step small.
- **Re-issue `cones`** (lost) and **replicate r016_03** at 2 seeds to lock the 21.094 record.

# Round r017

## 1. What happened
The control is now the FINGER (`r016_03` promoted, purse-string removed): `r017_00` protr 1.83,
aspect_final 21.094, grip 0.173, invag 0.628, n_tubes 6, 5511 cells, morphology 'tube'. Three slots
empty (08/09/10). The rest split: one finger reproduced (05), FOUR P4 chemistry deaths (02/03/04/06),
one high-N apoptosis star (07), one NaN rail (11), a b_bru divide pair (12/14), a d_a sweep (13/15).

## 2. What was learned
- **The finger reproduces at a THIRD seed and out-folds the star.** `r017_05` = `r015_06`
  (purse-string ON): protr_peak 2.199, grip_peak 0.34356, aspect_final 7.544 — identical to r015_06 —
  PLUS invagination_peak **0.754** (new campaign record, prev 0.617 r013_05) and aspect_max_peak
  **34.616** (record, prev 21.094). n_tips 6, n_tubes 5, corr 0.944, red_at_tip 0.979, mech_p_ratio
  3.765 (grown), 5690 cells. Finger now confirmed at 3 seeds (r015_06/r016_03/r017_05); the low-N
  finger folds DEEPER than the 12201-cell star. Scored refuted only as a replicate filed against
  spot_spacing>66.82 (hit exactly 66.82).
- **Four b_star variations EXTINGUISH the chemistry (P4).** `r017_02/03/04/06` collapse to spheres,
  protr ≤1.024, act extinct (eye: "flashes once then dies"). The new b_star seeding/gate levers
  (relgate/oriented/sharp/avoid) mostly destabilize the RD; only the plain finger patterns. The map
  around the finger is still blank.
- **d_a is the clean wavelength/coarsening lever** (Route A): higher d_a → fewer, wider spots, higher
  grip; at 0.16 reaches Okuda scale (9 spots, spacing 21.15) as a bilobed bud, grip 0.119.
- **L3 REFUTED reconfirmed** — control and r017_05 both finger, eye and metric agree.

## 3. What went wrong
- 3 empty (08/09/10). Four of six Route-B probes were P4 deaths — the b_star variations meant to map
  the finger's neighbours mostly killed the chemistry.
- `r017_11` P4+P12 NaN (act_max 3.3e9, 32246 cells, b_bru vth_frac 10) — apparatus rail.
- `r017_12/14` (b_bru divide 5/8): v_cell 0.608, cells flat ~2000, corr −0.26 anti-grip — division
  saturated rail, base retired again.

## 4. What to do next
- **Parent `r017_05`/`r016_03` finger** — tube + branched + complex seats; invag record 0.754 → complex.
- **The b_star variations need gentler settings** — their seeding/gate changes break P4. Start each
  variation FROM the plain finger and change ONE lever, or re-baseline the gate as a fraction (relgate)
  before reissuing.
- **Route A:** high d_a (~0.16) is the coarse-pattern lever on b_gs_plain; retire b_bru_question
  (division saturated, NaN at vth_frac 10).

# Round r018

## 1. What happened
Control `r018_00` is the FINGER (r016_03 promoted): protr_peak 1.83, `protrusion_aspect_max_final`
21.094, grip 0.173, invagination 0.628, n_tubes 6, 5511 cells, morphology 'tube'. Three Route-B
slots (`r018_01/02/06`) are **bit-identical to it** — protr 1.83, aspect_final 21.094, grip 0.173,
invag 0.628, 5511 cells all reproduce exactly. The finger is seed-floor-0 across four slots now.
One genuine variation (`r018_03`) coarsened it to fat buds; one death twin (`r018_07`) held it; two
`b_star` variations (`r018_04/05`) extinguished the activator to a sphere. Five slots lost
(`r018_08/09/10/11/13` empty).

## 2. What was learned
- **The finger recipe is LOCKED at seed floor 0.** `r018_01/02/06` reproduce the control to the bit,
  so their confirmations (aspect_peak>5, n_tips_peak>2) are the control's numbers, not new evidence.
  Stop re-running it — three seeds are spent confirming what r015/r016/r017 already fixed.
- **Apoptosis on the finger no longer subtracts from the fold, but degrades the arm.** `r018_07`
  (n_apop 75): invagination_peak 0.635 > control 0.628 — the FIRST death run in r001–r015 not to
  reduce the fold — yet aspect_final 21.094→6.067, protr 1.83→1.797. Confirmed invag>0.628, but the
  gain (+0.007) is within the coupling's noise while the arm clearly shrinks. Death deepens nothing;
  it trades a finger for a hair of fold.
- **`r018_03` coarsens the finger to Okuda's spot scale but loses the arm.** protr 1.615, 3220 cells,
  red_frac_final 0.155 (control 0.041, 3.8×), n_spots_final 5, spot_spacing_cells 23.85 (Okuda
  ~10-spot scale), corr_act_rad_peak 0.949, grip 0.173 — but aspect_final 2.792, eye: 5 fat
  red-tipped buds, not fingers. Refuted invag>0.754 (0.309). A coarser, hotter pattern that buds
  instead of fingering: the edit reached the target wavelength and gave up the buckling.

## 3. What went wrong
- **`r018_04/05` P4 deaths** — two more `b_star` variations extinguish the activator within ~3
  frames (act_max_final 0, protr ≤1.021, sphere). Reconfirms r017: the seeding/gate variations
  mostly destabilise the RD; only the plain finger patterns.
- **`r018_12` d_a 0.3 P4 death** (2051 cells, act extinct ~frame 220); **`r018_15` b_bru a_sw 0.5**
  P4+P12 NaN (29974 cells, act_max NaN). Five empty slots.

## 4. What to do next
- **Retire the finger replicate.** It is bit-stable; no more seed slots on it.
- **Identify and re-buckle `r018_03`.** It hit spot_spacing 23.85 (Okuda scale) with red_frac 3.8×
  control but lost the arm — restore the K_V/buckling lever on that coarsened pattern to test whether
  few coarse spots + incompressible cells finger instead of bud.
- **Sweep `cell_chem_diffuse.chi` on the finger.** `r018_14` is the campaign's FIRST chi value off the
  fixed 1.3 (chi 0.3 → n_spots 143, spot_spacing 2.78 — finest yet). chi is the untouched wavelength
  knob; sweep it UP to coarsen without the d_a death at 0.3.

# Round r019

## 1. What happened
Control `r019_00` is the FINGER (protr 1.83, protrusion_aspect_max_final 21.094, grip 0.173,
invagination 0.628, n_tubes 6, 5511 cells; eye: 4–5-armed pointed star). Six of fourteen slots empty
(04/08/09/10/11/13). Two axes split the round. Low-N: `r019_06` (5825 cells) beats the control's
protrusion — protr_peak **2.296** (campaign record, prev 2.199), grip 0.322, aspect_peak 25.018,
n_tips 6, n_tubes 7, eye "5–6 sharp fingers." High-N: `r019_01` (23527 cells) and `r019_02` (16828)
over-grow into deep-folding hollow stars — invagination 0.978 / 0.598, reduced_volume 0.288 / 0.283 —
but `r019_01` folds THROUGH itself (P11 broken, self-fold ~frame 1550). Two P4 chemistry deaths
(03/05); one decoupled hot ball (15).

## 2. What was learned
- **`r019_06` is a stronger finger sibling and a new protr record (2.296).** Grown — mech_p_ratio
  3.746 is NOT forcing (no extrusion operator exists, user §3; the eye's "forced signature" read rests
  on the retired stand-in). Its edit is not identifiable from the metrics I was given — it must be
  replicated and named. Predicted n_tips>6, refuted at exactly 6 — a whisker miss on a genuine finger.
- **Cell count is the finger's arm-vs-fold lever, with a rupture ceiling near 23k (new L8).** The
  5511-cell finger holds a clean arm (aspect 21.094); grown to 16828 (`r019_02`) it trades the arm for
  a clean hollow star (n_tubes 4, invag 0.598, all premises intact); grown to 23527 (`r019_01`) the
  fold deepens to invag 0.978 but tears the sheet (P11). The deepest fold on file is a broken mesh, not
  a tissue — a direct picture of the project's growth-overshoot (user §5).
- **`r019_01` confirmed aspect<21.094 (1.352) and `r019_05` confirmed n_tubes<5 (0)** — both trivial:
  the first on a P11-broken specimen, the second on a P4-dead sphere. `r019_02` refuted n_tubes>6 (4).
- **`r019_15` is a textbook decoupled case:** b_bru a_sw 0.7, act_max 9.31 (campaign-hottest), yet
  corr_act_rad −0.254 and grip −0.001. Strong pattern, zero shape response — the base cannot grip.

## 3. What went wrong
- 6 of 14 slots empty (43%) — the largest loss since r013/r014. `r019_04` (b_bru rho 0.5) and the two
  cell_grow.rate rungs crashed to empty (rate ceiling ~0.0017 reconfirmed).
- `r019_03/05` P4 deaths — chemistry flashed once and died (act_max_final 0, sphere); b_star
  variations again destabilising the RD (reconfirms r017/r018).
- `r019_15` patterned HOT with no NaN yet anti-grips — b_bru retired again, now for decoupling not
  divergence.

## 4. What to do next
- **Replicate and name `r019_06`** (protr 2.296, the record) at 2 fresh seeds before building on it —
  it is single-seed and its edit is unknown from metrics; eight single-seed clean points have regressed
  across this project.
- **Bracket the rupture ceiling.** `r019_02` (16828 cells, clean, invag 0.598) vs `r019_01` (23527,
  P11 tear) — sweep the growth cap between them to find where the fold ruptures, and hold the finger
  below it. This is the growth-overshoot brake (user §5).
- **chi 2 is the coarse-pattern lever off the fixed 1.3** (29 spots) but grip stays flat — cross chi 2
  coarsening with the finger's K_V buckling rather than sweeping chi further.

# Round r020

## 1. What happened
Control `r020_00` IS the finger (promoted r016_03): protr_peak 1.83, `protrusion_aspect_max_final`
**21.094**, grip 0.173, invag 0.628, n_tubes 6, 5511 cells, morphology 'tube'; eye "5-armed pointed
star with genuine fingers." **Not one Route-B edit reproduced it.** Every non-control run reads
`protrusion_aspect_max_final` 0.0 / `n_tips_final` 0 — the arm is gone. The edits split two ways off
the finger's ~5500-cell window: OVER-grew (`r020_01` 24661, `r020_03` 17182, `r020_02` 12976,
`r020_04` 12235 cells → lobed/folding/crumpled stars) or UNDER-grew (`r020_05` 3144, `r020_07` 2224
→ broad buds / dead chemistry). Five slots empty (08/09/10/11/13); `r020_15` NaN-crashed.

## 2. What was learned
- 1 confirmed (`r020_05` protr 1.42<1.83), 5 refuted. Every "more tubes/tips/protr" prediction on the
  finger (`r020_01` n_tubes>6, `_02` >11, `_03` n_tips>6, `_04` protr>1.598) FAILED the same way:
  pushing growth harder does not multiply arms, it over-grows the finger into a fold. Reconfirms L8
  and L4 — the finger is a ~5500-cell phenomenon; move cell count off it and the arm collapses.
- `r020_03` is the round's strongest specimen: 17182 cells, grip_peak 0.24678, **reduced_volume_final
  0.267 (campaign-deepest CLEAN, prev 0.283 r019_02)**, n_tubes 9, invag 0.607, all premises intact;
  eye "~7 red-tipped fingers, clearest protrusions." A `complex`/`branched` parent, folded not fingered.
- Route A: `cell_chem_diffuse.chi` 2.8 (`r020_12`) → 14 spots, nearest Okuda ~10 on file, grip flat 0.123.
  `cell_chem_from_shape.beta` −2 (`r020_14`) DECOUPLES (grip 0.0006, corr 0.033) with the activator ALIVE.

## 3. What went wrong
- `r020_07` P4 death (2224 cells, chemistry coalesces to 2 domains then extinct) and `r020_15` P4+P12
  NaN (15166 cells, act→NaN) — a b_star variation and a blowup, not biology.
- `r020_04` death twin (n_apop 89): reduced_volume 0.462, SHALLOWER than the clean folding stars
  (`r020_03` 0.267) — apoptosis still does not deepen the fold (reconfirms r001–r019).
- `n_spots_peak` rails at exactly 100 on control/01/03/05/12/14 — measurement cap.

## 4. What to do next
- **Hold cells at ~5500 and vary ONE mechanism** — every edit this round lost the finger by moving
  cell count. Use the untried `set_impl` levers (user §7: `cell_divide:orient_iface`, `cell_chem_from_shape:
  tension/pressure`, `cell_chem_seed:cones`) at the finger's growth setting, not stronger growth.
- **`r020_03` (grip 0.247, reduced_vol 0.267) is the round's parent for `complex`/`branched`** — a
  clean deep-fold star; replicate at a fresh seed and name its edit before building on it.
- **Stop pushing the finger harder.** n_tubes>6/n_tips>6 refuted four ways; the arm-count lever is not
  growth. Cross chi 2.8 coarsening (14 spots, Okuda scale) with the finger's K_V buckling instead.

# Round r021

## 1. What happened
Control `r021_00` = the finger: protr_peak 1.83, aspect_final 21.094, grip 0.173, invagination 0.628,
n_tubes 6, 5511 cells (`r021_04` bit-identical → seed floor ~0). Six real runs, six slots empty
(08–11/13/15), `r021_15` a P12 NaN crash. Two runs beat the control on its own terms: `r021_03`
reproduces the 2.296 finger; `r021_02` sets a new grip record on a NON-finger.

## 2. What was learned
- `r021_03` (n_tips>6, refuted at 6) **replicates `r019_06` exactly** — protr 2.296, 5825 cells, grip
  0.322, aspect_final 14.719, mech_p_ratio 3.746. The 2.296 finger is seed-robust; r019_06 is now a
  named 2nd finger composition. The refuted prediction is the only miss — n_tips caps at 6, the arm
  count is not the growth lever (reconfirms r019/r020).
- `r021_02` (invagination>0.598, confirmed 0.646) carries **grip 0.35083, a campaign record**, at 7550
  cells with aspect_final 0 — high grip WITHOUT the buckled arm, the first grip record off a finger.
- `r021_12` `set_impl cell_chem_from_shape` = a **null**: beta 0 gives a bold labyrinth on a perfect sphere,
  grip 0.006, mech_p_ratio 0. The feedback leg is dead unless beta≠0 (Route A: beta 1 → grip 0.103).
- `r021_06` (reduced_vol<0.40, confirmed 0.288) over-grows the finger to 21832 cells and folds deep and
  CLEAN (P11 intact) — L8 fold regime, rupture ceiling stochastic above 21.8k.
- `r021_01` (aspect>25.018, refuted 1.214): over-grew to 7058 cells, lost the arm.

## 3. What went wrong
- `r021_05` P4 death (n_spots>3 refuted at 1): one-sided lobed dome, act_max_final 0, activator extinct
  early — a settings failure, not biology.
- Six empty slots + one NaN crash (`r021_15`): 7 of 14 lost, the round's real cost.

## 4. What to do next
- **Sweep `cell_chem_from_shape.beta` up from 1** on the finger base — the ONLY lever this round that lifted
  grip off baseline (0.010→0.103 as beta 0→1) by coupling, not by wavelength. Find where it breaks.
- **Cross beta≠0 feedback INTO the ~5500-cell finger** — the finger has a live feedback leg (r016);
  test whether stronger beta deepens its fold without losing the arm.
- **Name `r021_02`'s edit** (grip 0.351, aspect_final 0) and replicate — a high-grip folding star
  distinct from the finger; the `complex`/`bud` seat's parent.

# Round r022

## 1. What happened
Control `r022_00_ctrl` = the finger: protr_peak 1.83, aspect_final 21.094, grip 0.173, invagination
0.628, n_tubes 6, 5511 cells (`r022_05` bit-identical → seed floor ~0). Seven real slots, SEVEN empty
(08–12/14), `r022_14` a P12 NaN blowup. Every scored prediction refuted except one; two death twins
carry the round.

## 2. What was learned
- `r022_03` (protr>2.296, refuted at the bound) **replicates the 2.296 finger a THIRD seed**
  (`r019_06`=`r021_03`): protr 2.296, 5825 cells, grip 0.322, aspect_final 14.719/aspect_peak 25.018,
  n_tubes 7, mech_p_ratio 3.746. Seed-robust at 3 seeds. The miss is the replicate scored against the
  copied `>2.296`.
- **Apoptosis now twins the finger WITH a deep fold — and it is not monotone in death.** `r022_07`
  (n_apop 80): protr 2.225, aspect_final 9.197, invagination 0.700 (>control 0.628), gyr_prolate 2.830 —
  refuted invag>0.754. `r022_06` (n_apop 87): protr 2.031, aspect_final 17.787, but invag 0.507
  (<control), grip 0.236 (refuted >0.30). At equal death the fold goes opposite ways; the arm survives
  both (reverses r015's "death degrades the arm").
- `r022_02` (grip>0.173, **confirmed 0.221**): 6280 cells, protr 1.937, aspect_final 0 / n_tips 0 — the
  arm zeroes just past the ~5825 window while grip climbs; eye sees fingers, metric doesn't (L3/L4).

## 3. What went wrong
- Half the round lost: 7 empty slots + `r022_14` P12 NaN (eye: chemistry → NaN from frame 1710). The
  round's real cost; no P4 chemistry death this time.
- Every replicate/inert prediction (`r022_03` >2.296, `r022_05` >21.094) refused AT the bound — copied
  predictions scored against bit-identical or seed-floor reruns, not new tests.

## 4. What to do next
- **Bracket apoptosis on the finger**: `r022_07`/`r022_06` disagree at equal n_apop (invag 0.700 vs
  0.507). Sweep `max_mark_frac` on the finger death twin — is the deep fold reproducible or seed luck?
- **Stop re-seeding the finger control** — 3 slots this round restated it (05 bit-identical). Spend the
  seat on the `complex`/`bud` targets instead.
- **`cell_chem_from_shape.beta` SATURATES above 1** (2→grip 0.107 ≈ 1→0.103); stop sweeping beta up, push it
  INTO the finger base instead.

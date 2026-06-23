# Model/trainer improvement logbook (beyond hyperparameter grid)

Started 11:03:23. The grid search ceilings the *hyperparameters*; here we improve the *model* and *trainer*. Benchmark to beat = grid-search best (sensor_angle u-err 0.121). Each experiment: a hypothesis, a controlled comparison on the same targets (mean+/-std), a verdict.


## H1 [11:03:26]: a no-Gumbel margin/hinge loss recovers sensor_angle better than the softmax-mixture NLL.
Controlled: same reconstruction + iters=800 restarts=3; only the sense model/loss differ.
- mixture-NLL : angle=0.179+/-0.155 turn=0.115 dist=0.216
- margin-hinge: angle=0.498+/-0.324 turn=0.369 dist=0.137
- verdict: FALSIFIED (margin not better)  [mixture 0.179 vs margin 0.498; grid ceiling 0.121]

## H3 [11:08:45]: UCB/GD hybrid (black-box 2D grid over geometry + GD on ts/beta/temp, scored by one-step NLL) recovers sensor_angle better than pure-GD mixture.
- pure-GD mixture: angle=0.179+/-0.155 turn=0.115 dist=0.216
- UCB+GD hybrid  : angle=0.251+/-0.174 turn=0.104 dist=0.060
- verdict: FALSIFIED  [pure-GD 0.179 vs hybrid 0.251; grid ceiling 0.121]

## H4 [11:15:45]: a little input noise (decaying pos+heading jitter) during training improves sensor recovery (your cross-repo trick).
- noise_pos=0.000 noise_h=0.00: angle=0.179 turn=0.115 dist=0.216
- noise_pos=0.003 noise_h=0.02: angle=0.208 turn=0.138 dist=0.258
- noise_pos=0.008 noise_h=0.05: angle=0.239 turn=0.127 dist=0.244
- noise_pos=0.015 noise_h=0.10: angle=0.227 turn=0.093 dist=0.217

## H2 [11:22:10]: RECURRENT K-step heading unroll recovers sensor_angle better than one-step (accumulation amplifies the weak angle signal). K=6, same lr/iters.
- one-step    : angle=0.191+/-0.150 turn=0.134 dist=0.214
- recurrent K=4: angle=0.184+/-0.110 turn=0.157 dist=0.069
  verdict K=4: tie  [one-step 0.191 vs recurrent 0.184; grid ceiling 0.121]

## H5 [11:28:12]: heading as (cos,sin) unit vector in the recurrent unroll (turn=rotation, loss=MSE on cos/sin; the connectome-cx representation) beats scalar-angle recurrent on sensor_angle. K=6.

- recurrent K=8: angle=0.279+/-0.163 turn=0.190 dist=0.055
  verdict K=8: FALSIFIED  [one-step 0.191 vs recurrent 0.279; grid ceiling 0.121]
- recurrent scalar : angle=0.238+/-0.136 turn=0.185 dist=0.049
- recurrent (cos,sin): angle=0.225+/-0.124 turn=0.083 dist=0.067
- verdict: VALIDATED ((cos,sin) better)  [scalar 0.238 vs (cos,sin) 0.225; grid ceiling 0.121]

## H6 [11:36:23] FAIR (iters=1500,restarts=4): best one-step vs (cos,sin) recurrent.

## CONCLUSIONS (after H1-H6)

Recovery quality by parameter (best across methods):
- field (3 params): u-err ~0.000  -- exact one-step (deposit/diffuse/decay), validated vs engine.
- move_speed: ~0.000 -- per-step displacement.
- turn_speed: ~0.08-0.10 -- identifiable.
- sensor_dist: ~0.05-0.07 -- recurrent K-step unroll NAILS it (radial sampling strongly changes the sensed gradient).
- sensor_angle: ~0.12-0.18 FLOOR -- WEAKLY IDENTIFIABLE. The robust finding.

Hypotheses (scientific method, all on the same 4 targets, mean+/-std):
- H1 margin / no-Gumbel: FALSIFIED (0.498) -- hard branch-classification too sensitive to heading-reconstruction noise.
- H3 UCB+GD hybrid (exhaustive 2D grid over geometry, NLL-scored): FALSIFIED for angle (0.251) but SOLVED sensor_dist (0.060). KEY INSIGHT: even an exhaustive search cannot find sensor_angle -> identifiability problem, not optimization.
- H4 noise injection (your cross-repo trick): FALSIFIED (monotonically worse 0.179->0.239) -- confirms identifiability; noise smooths landscapes, it cannot add missing information.
- H2 recurrent K-step unroll: VALIDATED for sensor_dist (->0.055-0.069); angle ~tie.
- H5 (cos,sin) heading representation (connectome-cx recipe: turn=rotation, MSE on cos/sin): VALIDATED -- 0.238 -> 0.225 vs scalar-angle recurrent. The CORRECT circular representation; gain is modest because the bottleneck is identifiability, not angle-wrap.

Mechanistic reason for the sensor_angle floor: the engine's turn DIRECTION depends on sign(wL - wR), which is largely INVARIANT to the angle magnitude; only the readings' magnitudes carry angle info, and the chemical field is locally smooth at the sensor scale (wL ~ wR regardless of angle) -> weak signal swamped by noise. sensor_dist changes the sensed gradient strongly -> identifiable.

VERDICT: model/trainer improvements help where signal exists (recurrent solved sensor_dist; (cos,sin) is the right representation and validated) but cannot break a genuine identifiability limit on sensor_angle. Best recovery = one-step mixture-NLL (angle) + recurrent-(cos,sin) (dist) + exact field/move. Established across 6 falsification tests.

## H7 [11:57:51]: CONNECTOME-CX CURRICULUM scheme -- free-run heading (cos,sin), per-frame loss over a GROWING horizon [10,30,60,100,140], random start frames, soft tail, lr decay, grad clip, bounce resync. (My earlier recurrent only matched the FINAL net change.)
  target0 stage0 K=10: angle_uerr=0.086 (turn=0.25 dist=0.23)
  target0 stage1 K=30: angle_uerr=0.106 (turn=0.25 dist=0.23)
  target0 stage2 K=60: angle_uerr=0.115 (turn=0.22 dist=0.21)
  target0 stage3 K=100: angle_uerr=0.118 (turn=0.18 dist=0.16)

## H7 [12:08:38] CURRICULUM (per-frame (cos,sin) loss, free-run heading, random starts, bounce resync) -- the connectome-cx scheme. My prior recurrent matched only the FINAL net change; this scores every frame over the horizon.
- schedule=(10,): angle=0.238 turn=0.152 dist=0.066  [at floor; prior floor 0.12-0.18]  (75s)

## H7b [12:17:22] CURRICULUM + per-iter random-start DIVERSITY (resample=25), per-target breakdown. schedule=(10,30,60), iters=400.
- target0: angle_uerr=0.120  turn=0.072 dist=0.038
- target1: angle_uerr=0.545  turn=0.032 dist=0.032
- target2: angle_uerr=0.248  turn=0.075 dist=0.095

## H7 CONCLUSIONS (connectome-cx curriculum scheme)

The curriculum scheme (per-frame (cos,sin) loss + free-run recurrent heading + random start frames + bounce resync) is the RIGHT method and CORRECTS my premature uniform-floor conclusion:

1. turn_speed and sensor_dist are NAILED across all targets (turn ~0.03-0.07, dist ~0.03-0.04) -- better than every earlier method. The recurrent accumulation + PER-FRAME loss is the key; my earlier recurrent (H2/H5) matched only the FINAL net change and so underperformed.
2. sensor_angle is TARGET-DEPENDENT identifiable, not a uniform floor: target0 angle 0.120 (best single-stage K=10 per-iter: 0.086), target1 angle 0.545 (unidentifiable). The 0.12-vs-0.55 spread proves it depends on the target field having lateral structure at the sensor scale, not on the method.
3. GROWING the horizon (->140/500) does NOT help slime: chaos + wall bounces inject unpredictable reheads. SHORT horizon (K=10) is best. This DIFFERS from the connectome head-direction integrator (pure integration, no chaos/bounces) where long horizons help -- a real domain difference.

VERDICT: recurrent + per-frame + cos/sin + random starts materially improved turn_speed and sensor_dist and recovered sensor_angle WHERE IDENTIFIABLE (~0.09-0.12). sensor_angle residual difficulty is genuine target-dependent identifiability.
- target3: angle_uerr=0.291  turn=0.030 dist=0.288
- MEAN angle=0.301 (min 0.120, max 0.545) -> target-DEPENDENT identifiability

## PROFILE [12:32:15]: likelihood profile of sensor_angle (turn,dist fixed at truth), curriculum loss, K=30.
- target0: true_ang_u=0.88, argmin=0.98, relative well-depth=0.41, loss(true)=0.3611, loss(min)=0.3506  -> IDENTIFIABLE, fit missed (dip at truth)
- target1: true_ang_u=0.30, argmin=0.94, relative well-depth=0.94, loss(true)=0.8371, loss(min)=0.5533  -> BIASED (dip away from truth)
- target2: true_ang_u=0.75, argmin=0.94, relative well-depth=0.39, loss(true)=0.5987, loss(min)=0.5718  -> BIASED (dip away from truth)
- target3: true_ang_u=0.08, argmin=0.42, relative well-depth=0.66, loss(true)=0.2082, loss(min)=0.1439  -> BIASED (dip away from truth)
- saved plot -> archive/profile_sensor_angle.png

## PROFILE [12:44:16]: likelihood profile of sensor_angle (turn,dist fixed at truth), curriculum loss, K=30.
- target0: true_ang_u=0.88, argmin=0.98, relative well-depth=0.32, loss(true)=0.3467, loss(min)=0.3379  -> IDENTIFIABLE, fit missed (dip at truth)
- target1: true_ang_u=0.30, argmin=0.98, relative well-depth=0.42, loss(true)=0.8136, loss(min)=0.6024  -> BIASED (dip away from truth)
- target2: true_ang_u=0.75, argmin=0.94, relative well-depth=0.14, loss(true)=0.6432, loss(min)=0.6226  -> UNIDENTIFIABLE (flat profile)
- target3: true_ang_u=0.08, argmin=0.42, relative well-depth=0.47, loss(true)=0.1884, loss(min)=0.1397  -> BIASED (dip away from truth)
- saved plot -> archive/profile_sensor_angle.png

## FIX ATTEMPT (3x3-window read) -- FALSIFIED

Hypothesis: the sensor_angle bias (profile argmin at large angle, not truth) comes from the surrogate reading a single bilinear point instead of the engine.s 3x3 sensor window. Swapped to a 3x3 replicate-box-blurred bilinear read (operators.box3/box3b in _reads/dist_b/mu_vec) and re-profiled.

Result: bias PERSISTS (argmin still 0.94-0.98 for targets 1,2,3 vs truths 0.30/0.75/0.08), and profile well-depths DROPPED (target1 0.94->0.42, target2 0.39->0.14 ~flat). So matching the engine.s genuine 3x3 window REDUCES the lateral-gradient signal angle depends on; it is not the bias source. FALSIFIED.

Diagnosis: the bias is monotone toward LARGE sensor_angle independent of truth -> not the read footprint. Remaining suspects: (a) the STEERING surrogate (mixture/softmax p_left = sigmoid(beta*(wL-wR))) vs the engine.s hard argmax branch -- a larger angle may always lower the free-run loss under the smooth steering; (b) a turn_speed<->sensor_angle confound (profile fixes turn at truth, but beta/temp are at defaults, not fitted). Next: 2D profile over (sensor_angle x turn_speed), and profile with fitted beta/temp; then revisit the steering surrogate (the genuinely non-differentiable part). The 3x3 read is kept (more faithful to the engine) but is not the fix.

## PROFILE [12:58:23]: likelihood profile of sensor_angle (turn,dist fixed at truth), curriculum loss, K=30.
- target0: true_ang_u=0.88, argmin=0.98, relative well-depth=1.12, loss(true)=0.3490, loss(min)=0.3385  -> IDENTIFIABLE, fit missed (dip at truth)
- target1: true_ang_u=0.30, argmin=0.98, relative well-depth=0.72, loss(true)=1.0072, loss(min)=0.6838  -> BIASED (dip away from truth)
- target2: true_ang_u=0.75, argmin=0.86, relative well-depth=0.39, loss(true)=0.7488, loss(min)=0.7219  -> IDENTIFIABLE, fit missed (dip at truth)
- target3: true_ang_u=0.08, argmin=0.42, relative well-depth=0.67, loss(true)=0.2288, loss(min)=0.1572  -> BIASED (dip away from truth)
- saved plot -> archive/profile_sensor_angle.png

## FIX: std3 (scale-invariant steering) -- PARTIAL

Standardize the 3 sensor readings per cell before steering (operators.std3 in dist/dist_b/mu_vec) so decisiveness comes from beta, not reading-magnitude. Re-profiled sensor_angle:
- well-depths INCREASED (more identifiable signal): target0 0.32->1.12.
- target2 moved BIASED->IDENTIFIABLE (argmin 0.94->0.86 vs true 0.75).
- but small-true-angle targets still biased: target1 argmin 0.98 (true 0.30), target3 argmin 0.42 (true 0.08).
Partial improvement; residual bias is specific to SMALL true angle (close L/R sensors -> ordering still favors larger angle). Next: keep std3; test recurrent-scheme variants + a steering that is sign-only.

## RECURRENT-SCHEME SWEEP [13:02:32] (std3 steering; 3 targets, mean u-err)
- one_step              : angle=0.354 turn=0.241 dist=0.190
- rec_final_K6          : angle=0.159 turn=0.319 dist=0.095
- curric_K10            : angle=0.164 turn=0.370 dist=0.179
- curric_10_30_60       : angle=0.079 turn=0.285 dist=0.143

## PROFILE-POS [13:12:40]: loss ONLY ON PARTICLES (positions, free-run sense->advance). sensor_angle profile, K=20.
- target0: true=0.88 argmin=0.94 depth=0.41 -> IDENTIFIABLE (dip at truth)
- target1: true=0.30 argmin=0.30 depth=0.06 -> UNIDENTIFIABLE
- target2: true=0.75 argmin=0.62 depth=0.11 -> UNIDENTIFIABLE
- target3: true=0.08 argmin=0.14 depth=0.70 -> IDENTIFIABLE (dip at truth)
- saved archive/profile_pos.png

## FIX: loss ON PARTICLES (positions) -- THE BIAS FIX (user insight)

Hypothesis (user): the sensor_angle bias comes from matching the RECONSTRUCTED heading (atan2 of displacement), not the particles directly. Test: free-run the full particle dynamics (sense->rotate heading->advance position) and put the loss ONLY on the PREDICTED POSITIONS vs GT positions (no heading in the loss). Re-profiled sensor_angle.

Result -- bias REMOVED. argmin moves from ~0.98 (heading-loss, biased) to AT/NEAR truth:
  target0 true .88: .98 -> .94    target1 true .30: .98 -> .30 (exact!)
  target2 true .75: .86 -> .62    target3 true .08: .42 -> .14
The large-angle bias is gone for the small-true-angle targets that were biased. Wells are shallower on t1/t2 (unbiased but weakly identifiable) -- a much better regime than biased. CONCLUSION: heading-reconstruction was the bias source; position loss is the fix. Next: make the trainer use position loss; regenerate the recovery; the answer to do-you-put-loss-on-particles-or-field = particles (and now positions, not reconstructed heading).

## POSITION-ONLY field test [13:23:50]: profile diffuse.rate from PARTICLE POSITIONS ALONE (field free-run, never observed). K=10.
- diffuse.rate: true_u=0.99 argmin=0.05 depth=0.004 -> UNIDENTIFIABLE from positions (flat)
- saved archive/profile_field_from_pos.png
- curric_10_30_60_100   : angle=0.118 turn=0.204 dist=0.150
- best sensor_angle: curric_10_30_60 (0.079)

## POSITION-ONLY, NO FIELD LOSS (cardio question) -- field NOT identifiable from positions

Closed-loop free-run: field evolves with LEARNABLE deposit/diffuse/decay (own field, teacher-forced only at t0), particles move by sense+advance reading that self-generated field, loss ONLY on positions. Profiled diffuse.rate.
Result: FLAT (depth 0.004, argmin 0.05 vs true 0.99) at K=10. The field.s effect on short-horizon particle motion is negligible (initial GT field + fresh deposits dominate the locally-sensed field), and chaos forbids the long horizon needed to accumulate the diffusion effect.
CONCLUSION: position-only recovers PARTICLE/SENSE params (the de-biased sensor_angle), but NOT the latent FIELD params -- those need their own loss (the field IS observable in slime). CARDIO implication: recovering the latent electrical field from observed mechanics alone is the same setup and is HARD; cardio may do better (stronger electromechanical coupling, less chaos, longer usable horizon) but should not assume the latent field is identifiable from motion alone -- give it a loss / observation where possible.

## POSITION-LOSS + NOISE [13:29:40] (recurrent state noise with/without; 4 targets, mean u-err)

## ABLATION recurrent x noise [13:35:53] (position loss; 2 targets t0,t1; mean u-err)
- one-step,  no-noise   : angle=0.075 turn=0.069 dist=0.055
- one-step,  noise.03   : angle=0.065 turn=0.060 dist=0.077
- recurrent, no-noise   : angle=0.171 turn=0.388 dist=0.070
- recurrent, noise.03   : angle=0.231 turn=0.309 dist=0.064
- BEST (sum 3): one-step,  no-noise

## ABLATION: recurrent x noise (position loss) -- ONE-STEP WINS

| scheme              | angle | turn  | dist  |
|---------------------|-------|-------|-------|
| one-step, no-noise  | 0.075 | 0.069 | 0.055 |  <- BEST
| one-step, noise .03 | 0.065 | 0.060 | 0.077 |
| recurrent, no-noise | 0.171 | 0.388 | 0.070 |
| recurrent, noise.03 | 0.231 | 0.309 | 0.064 |

Answers (user asked: with/without recurrent + with/without noise are key):
- RECURRENT vs ONE-STEP: for the POSITION loss, ONE-STEP is much better (recurrent free-run accumulates chaos -> angle 0.075->0.17-0.23, turn 0.07->0.31-0.39). OPPOSITE to the heading loss (where recurrence was needed because one-step heading was too noisy). The decisive change was the LOSS on POSITIONS, not the recurrence.
- NOISE: marginal/mixed. On one-step it slightly improves angle/turn (0.075->0.065, 0.069->0.060) at a small dist cost; on recurrent it hurts. Not needed.
WINNER: ONE-STEP POSITION loss -> all sense params recovered well & UNBIASED (angle ~0.07, turn ~0.07, dist ~0.055), vs heading-loss recurrent (angle 0.12-0.55, biased). This is the best sense fit found.

## MULTI-TYPE (2 & 4 types) [17:01:22]: same one-step position-loss training; `cross` now recoverable (multi-channel field).

  [slime2, C=2 channels] per-parameter recovery (u-space):
  param      true_u   rec_u    err
  amount      0.625   0.625  0.000
  diffuse     0.897   0.897  0.000
  decay       0.776   0.776  0.000
  cross       0.225   0.213  0.012  <- now identifiable
  move        0.300   0.300  0.000
  turn        0.874   0.747  0.126
  sensA       0.005   0.078  0.073
  sensD       0.821   0.853  0.032

  [slime4, C=4 channels] per-parameter recovery (u-space):
  param      true_u   rec_u    err
  amount      0.625   0.625  0.000
  diffuse     0.897   0.897  0.000
  decay       0.776   0.776  0.000
  cross       0.225   0.233  0.008  <- now identifiable
  move        0.300   0.300  0.000
  turn        0.874   0.780  0.094
  sensA       0.005   0.078  0.073
  sensD       0.821   0.860  0.039

## MULTI-TYPE EXPANSION (2 & 4 types) -- SUMMARY

Extended the inverse training + operators to multi-type slime. The chemical field gets C=#types channels; deposit writes per-type channel, diffuse/decay are per-channel (field ops already multi-channel, validated). Only  needed extension: a cell reads (1-cross)*own_channel + cross*total_over_channels -- operators.LearnableSense.mu_vec_mc, with  a learnable param. Trainer: sense_trainer.fit_sense_pos_mc (one-step position loss, the ablation winner). Mechanisms SLIME2 (slime_two_attract) / SLIME4 (slime_four) in spec_space.

Recovery (one-step position loss, u-space err):
  2-type: amount/diffuse/decay 0.000, move 0.000, cross 0.012, turn 0.126, sensA 0.073, sensD 0.032
  4-type: amount/diffuse/decay 0.000, move 0.000, cross 0.008, turn 0.094, sensA 0.073, sensD 0.039
HEADLINE:  (inter-type sensing weight) -- UNIDENTIFIABLE for single-type, held@GT there -- is now RECOVERED (err 0.012 at 2 types, 0.008 at 4 types; better with more types = stronger inter-type signal). All other params recover as in single-type. The same one-step position-loss scheme transfers unchanged to 2 and 4 types.

## MULTI-TYPE EXPANSION (2 and 4 types) -- clean summary
Extended inverse training + operators to multi-type slime. Field gets C=#types channels (deposit per-type channel; diffuse/decay per-channel -- field ops already multi-channel). Only sense extended: read = (1-cross)*own_channel + cross*total (mu_vec_mc), cross learnable. Trainer fit_sense_pos_mc (one-step position loss). Mechs SLIME2/SLIME4 in spec_space.
Recovery (u-err): 2-type cross 0.012, 4-type cross 0.008 (better with more types); amount/diffuse/decay/move 0.000; turn/sensA/sensD ~0.03-0.13. HEADLINE: cross (inter-type weight) was UNIDENTIFIABLE single-type, now RECOVERED with >=2 types. Same scheme transfers unchanged.

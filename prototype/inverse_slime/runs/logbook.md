# Sense-surrogate optimisation logbook

Started 10:48:29. Objective: minimise sensing u-error (esp. sensor_angle) across 4 held-out slime targets. Method: coordinate ascent over trainer hyperparameters then random refinement; each trial evaluated on all targets (mean+/-std = replication); verdict logged vs running best. Field+move are recovered exactly elsewhere.


## Baseline [10:48:58]
cfg={'lr': 0.05, 'iters': 400, 'restarts': 2, 'beta0': 12.0, 'temp0': 0.05, 'anneal': False, 'wd': 0.0, 'field_offset': 1, 'max_turn': 1.0, 'n_per_frame': 300, 'sched': 'cos', 'max_frames': 40}
result: score=0.225 angle=0.228+/-0.146 turn=0.189 dist=0.257

  -> archived spec+mp4 to archive/baseline/

## Coordinate-ascent sweep 1 [10:52:19]  (best score 0.225)
- T1 [10:52:31] iters=200: score=0.243 angle=0.229+/-0.145 turn=0.240 dist=0.260  [falsified]
- T2 [10:53:18] iters=800: score=0.196 angle=0.204+/-0.153 turn=0.119 dist=0.266  [IMPROVED]
- T3 [10:54:47] iters=1500: score=0.138 angle=0.119+/-0.080 turn=0.078 dist=0.216  [IMPROVED]
  => adopt iters=1500 (score 0.138, angle 0.119)
- T4 [10:55:33] restarts=1: score=0.179 angle=0.185+/-0.163 turn=0.087 dist=0.263  [falsified]
- T5 [10:57:49] restarts=3: score=0.135 angle=0.121+/-0.066 turn=0.101 dist=0.183  [IMPROVED]
- T6 [11:02:24] restarts=6: score=0.105 angle=0.130+/-0.071 turn=0.069 dist=0.116  [IMPROVED]
  => adopt restarts=6 (score 0.105, angle 0.130)
- T7 [11:06:48] lr=0.02: score=0.185 angle=0.187+/-0.142 turn=0.097 dist=0.269  [falsified]
- T8 [11:11:11] lr=0.1: score=0.107 angle=0.139+/-0.073 turn=0.080 dist=0.102  [falsified]
- T9 [11:15:32] lr=0.2: score=0.109 angle=0.183+/-0.107 turn=0.079 dist=0.066  [falsified]
- T10 [11:19:50] beta0=6.0: score=0.105 angle=0.130+/-0.071 turn=0.069 dist=0.116  [falsified]
- T11 [11:24:11] beta0=30.0: score=0.105 angle=0.130+/-0.071 turn=0.069 dist=0.115  [falsified]
- T12 [11:28:24] beta0=60.0: score=0.105 angle=0.130+/-0.071 turn=0.069 dist=0.115  [falsified]
- T13 [11:32:42] temp0=0.02: score=0.155 angle=0.215+/-0.124 turn=0.061 dist=0.189  [falsified]
- T14 [11:37:01] temp0=0.1: score=0.074 angle=0.052+/-0.050 turn=0.059 dist=0.112  [IMPROVED]
- T15 [11:41:16] temp0=0.2: score=0.105 angle=0.124+/-0.050 turn=0.066 dist=0.124  [IMPROVED]
  => adopt temp0=0.1 (score 0.074, angle 0.052)
  -> archived spec+mp4 to archive/r1_temp0/
- T16 [11:50:25] anneal=True: score=0.082 angle=0.072+/-0.044 turn=0.059 dist=0.113  [falsified]
- T17 [11:54:55] field_offset=0: score=0.074 angle=0.050+/-0.048 turn=0.059 dist=0.112  [IMPROVED]
- T18 [11:59:19] field_offset=2: score=0.071 angle=0.046+/-0.046 turn=0.059 dist=0.110  [IMPROVED]
  => adopt field_offset=2 (score 0.071, angle 0.046)
- T19 [12:03:34] max_turn=0.6: score=0.095 angle=0.064+/-0.034 turn=0.107 dist=0.113  [falsified]
- T20 [12:07:53] max_turn=1.5: score=0.071 angle=0.046+/-0.046 turn=0.059 dist=0.110  [falsified]
- T21 [12:12:10] n_per_frame=200: score=0.078 angle=0.059+/-0.044 turn=0.060 dist=0.115  [falsified]
- T22 [12:16:34] n_per_frame=400: score=0.072 angle=0.048+/-0.044 turn=0.059 dist=0.109  [falsified]
- T23 [12:20:52] n_per_frame=800: score=0.073 angle=0.056+/-0.038 turn=0.058 dist=0.105  [falsified]
- T24 [12:25:09] wd=0.0001: score=0.070 angle=0.041+/-0.039 turn=0.060 dist=0.109  [IMPROVED]
- T25 [12:29:29] wd=0.001: score=0.070 angle=0.043+/-0.038 turn=0.060 dist=0.107  [IMPROVED]
  => adopt wd=0.0001 (score 0.070, angle 0.041)
  -> archived spec+mp4 to archive/r1_wd/
- T26 [12:38:29] sched=none: score=0.080 angle=0.062+/-0.031 turn=0.064 dist=0.114  [falsified]

## Coordinate-ascent sweep 2 [12:38:29]  (best score 0.070)
- T27 [12:39:06] iters=200: score=0.095 angle=0.086+/-0.053 turn=0.100 dist=0.098  [falsified]
- T28 [12:40:20] iters=400: score=0.087 angle=0.099+/-0.023 turn=0.060 dist=0.102  [falsified]
- T29 [12:42:47] iters=800: score=0.075 angle=0.069+/-0.026 turn=0.059 dist=0.096  [falsified]
- T30 [12:43:33] restarts=1: score=0.087 angle=0.076+/-0.017 turn=0.079 dist=0.106  [falsified]

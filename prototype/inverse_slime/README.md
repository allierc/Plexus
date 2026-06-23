# Plexus inverse_slime — one-step gradient descent (beats UCB)

The ParticleGraph training pattern (teacher-forced one step, `t -> t+1`, Adam)
applied to Plexus parameter recovery, as the **alternative to the UCB tree**
(`../RL/`). Instead of a black-box full-rollout search, predict the next state from
the *true* current state and minimise the one-step error. No rollout → no chaos → a
dense exact gradient on the parameters.

## Files
- `field_model.py` — DIFFERENTIABLE reimplementation of the Plexus field tick
  `field_{t+1} = decay(diffuse(deposit(field_t, pos_t)))`, exact to the engine
  (validated `max|err| ~ 1e-7`). Params (`amount`, `diffuse.rate`, `decay.rate`) are
  learnable tensors.
- `fit_field.py` — teacher-forced one-step fit on a real engine-generated slime
  trajectory (≤160 frames ⇒ field recorded every tick), Adam on the one-step field
  MSE; `move_speed` read directly from per-step displacement. Head-to-head vs the UCB
  winner on the same target.

## Result (slime target 4)
| lever | UCB u-err | gradient-descent u-err |
|---|---|---|
| deposit.amount | 0.101 | 0.000 |
| diffuse.rate | **0.741** | 0.005 |
| decay.rate | 0.284 | 0.000 |
| move_speed | 0.007 | 0.000 |
| **mean** | **0.283** | **0.001** (≈224× better) |

UCB needed ~250 full rollouts/target and still missed `diffuse.rate` entirely;
gradient descent nails all four in 400 cheap one-step iterations.

## Scope / next
These are the **differentiable** slime parameters (3 field + move_speed). The 4
**sensing** params (`turn_speed`, `sensor_angle`, `sensor_dist`, `cross`) go through
`sense`/`bounce`, which use hard `torch.where` branches + a random turn — non
differentiable. Next step: a **smooth surrogate of `sense`** (softmax over the 3
sensor readings + bilinear field sampling at the sensor positions) so the sensor
geometry params also get a one-step gradient. Then the same idea extends to the
smooth mechanisms: **attraction_repulsion** and **boids** (pure differentiable force
laws — exactly ParticleGraph's setting, where UCB did worst).

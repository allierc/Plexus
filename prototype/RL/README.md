# Plexus RL prep — inverse design (observation → spec)

The goal: learn to recover the **spec** that generated a given simulation (or one
that produces a very similar simulation). This directory prepares the ground for an
RL implementation by building and validating the three things RL needs — an
**environment**, a **reward**, and **data + a baseline** — all on top of the
codebase's registered operators and engine (nothing is reimplemented).

## Building blocks (the RL interface)

| File | Role for RL |
|------|-------------|
| `spec_space.py` | **Action space + random spec generator.** A `Mechanism` = a fixed family of codebase operators (a base config) + a box of continuous `LEVERS`. `apply_u(u)` maps a unit-cube vector → a runnable `Spec`; `sample_u` draws a random one. Slime is the testbed (8 levers). |
| `rollout.py` | **Environment step.** `rollout(mech, u, seed)` → cell trajectory `(pos:(F,N,2), node_type:(N,))` via `plexus.engine.run`. Returns `None` on blow-up. |
| `metric.py` | **Reward.** `constellation_dist` = per-frame, per-type, sliced-Wasserstein OT on the joint **(position, velocity)** cloud (velocity = `pos[t+1]-pos[t]`), Hungarian-matched over exchangeable types. `reward = -dist`. |
| `cem.py` | **Black-box inverse baseline** (CEM) — the bar a learned policy must beat. |

## Overnight tests (what they produce)

| Script | Output | Why RL needs it |
|--------|--------|-----------------|
| `test_characterize_metric.py` | `characterize.json`, `pairwise.npy` | Reward **signal-to-noise**: seed floor vs random-pair distance, and a **degeneracy fraction** (how many distinct specs are mutually indistinguishable). Decides whether RL targets *behaviour* or *parameters*. |
| `test_inverse_baseline.py` | `results.jsonl` | CEM inversion over many held-out targets: achievable `best_dist` + `u_err` + **per-lever error** (which params are identifiable). The baseline RL must beat. |
| `build_dataset.py` | `shard_*.npz` | `(u, observation)` pairs — observation = `(K frames, M cells, 4)` pos+vel cloud. Supervised **warm-start** for the recognition net `q(spec|obs)`; RL fine-tunes from there. |
| `overnight.py` | `runs/overnight/...` | Orchestrates all three, time-boxed (`--hours`) and resumable. |

## Run

```bash
cd prototype/RL
# full battery overnight (resumable; re-run to continue)
PYTHONPATH=../../src nohup python overnight.py --hours 10 > runs/overnight.log 2>&1 &
# or a single phase
PYTHONPATH=../../src python test_characterize_metric.py
```

## Validated so far (slime, n=6000, 200 frames, CPU ~1 s/sim)

- constellation reward: self=0, same-spec/diff-seed ≈ **0.06** (seed floor),
  random specs ≈ **0.31** → ~**5×** separation.
- CEM drives `best_dist` 0.069 → 0.040 toward the floor, but `u_err` stays high →
  **behavioural match ≫ parameter recovery** (genuine non-identifiability). RL
  should be rewarded on behaviour, not parameter recovery.

## Next (the RL itself, not yet built)

- **Recognition net** `q(u | observation)`: set/temporal encoder over the
  `(K,M,4)` cloud → Gaussian over `u`. Supervised pretrain on the dataset.
- **Policy/value heads**: continuous tanh-Gaussian actor over the lever box +
  critic baseline; reward = `metric.reward`. REINFORCE/PPO fine-tune, or pathwise
  gradients if the metric is made differentiable.
- **Mechanism search** (later): lift the fixed-family assumption — a discrete head
  picks the operator family, this box tunes its interior.

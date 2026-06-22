# Inverse training for cardio — reusing the slime methodology

How to recover the cardio model's parameters (FitzHugh–Nagumo + couplings) from the
real velocity field, reusing the training methodology developed and **validated** on
the slime inverse problem (`prototype/inverse_slime/`, scientific log in
`prototype/inverse_slime/runs/logbook2.md`). This replaces "Stage 4 = UCB search"
(`README.md`) with a gradient-first recipe; UCB stays only for the discrete pieces.

The recipe below is not theory — every line is a verdict from a falsification test on
slime. Read it as "do this, not that, because we measured it."

---

## 1. The cardio inverse problem

- **Observed**: the real velocity/displacement field — `pos [F,N,2]`, `vel [F,N,2]`,
  `F=238`, `N=18769` tracked nodes, `dt=0.0417 s` (`cardio_real.npz`). The nodes are
  a **fixed tracked grid** → node *i* at frame *t* corresponds to node *i* at *t+1*.
- **Latent**: the electrical wave `(u,v)` on the cell grid (FHN), which drives
  contraction through the coupling. We never see it directly.
- **Recover**: FHN params `a,b,ε`, diffusion `D`, the trigger, and the
  electrical→mechanical (and back) coupling gains.

So it is a **partially-observed** inverse problem (mechanics observed, electrics
latent) over a system with **node identity** and **smooth** dynamics.

## 2. What's EASIER than slime (exploit it)

- **Node correspondence exists.** The tracking grid is Lagrangian with identity, so
  node *i* is the same node across frames. Use a **direct per-node loss**
  (`‖pred_i − gt_i‖`) — no optimal transport, no constellation metric. (Slime had
  chaotic anonymous particles and needed OT; you don't.)
- **FHN is smooth and differentiable.** No hard `torch.where` branch like slime's
  `sense`. So one-step gradient descent should recover `a,b,ε,D` *exactly*, the way it
  nailed slime's `deposit/diffuse/decay` (u-err ~0.000, **224× better than UCB**).
- **Waves are more deterministic than slime** (less chaos), so **longer rollout
  horizons should help** — unlike slime, where chaos+bounces capped the useful
  horizon at K≈10. Expect the head-direction integrator's long curriculum
  (100→800) to transfer better here than it did to slime.

## 3. What's HARDER (where the slime lessons bite)

- **The electrical state is latent.** Mechanical params fit from `vel_t → vel_{t+1}`
  directly; electrical params only show up through the **closed loop**
  (electric → force → displacement). Recovering them needs the **recurrent
  curriculum** (§5), not one-step.
- **Rotors have a phase.** The spiral-wave phase / tip angle is a **circular**
  variable → represent it as **(cos θ, sin θ)**, never a raw angle (§6).
- **Identifiability is not free.** Some params will be weakly constrained or the
  objective biased. **Check the profile** before trusting any point estimate (§7).

---

## 4. Recipe step 1 — decomposed learnable operators, validated EXACT vs the engine

One `nn.Module` per registered operator (`excitable_nagumo`, `trigger_pulse`,
`signal_to_mpm_force`, `mpm`, the aggregate/broadcast couplings), each with its
parameter as a learnable tensor, **and each with a `test_vs_engine()`** that builds a
tiny `engine.build()` Hierarchy, runs the real engine operator one step, runs the
learnable one on the same state, and asserts they match (~1e-6).

> Slime evidence: `operators.py` — deposit/diffuse/decay/advance all matched the
> engine to ≤1e-7; `sense` (the one non-differentiable op) was the only surrogate.
> This pattern is the thing the user liked: *the model is validated exact against the
> engine*, so a fit on the learnable op is a fit on the real dynamics.

Copy the structure from `prototype/inverse_slime/operators.py` (`LearnableDeposit`
… + `test_deposit` … + `run_all_tests`).

## 5. Recipe step 2 — one-step first, then recurrent CURRICULUM

**(a) One-step teacher-forced** (cheap, exact for smooth params): from the true
state at `t` predict `t+1`, minimise the per-node MSE by Adam. Recovers the
**mechanical** params and the smooth FHN params directly. This is the
`prototype/inverse_slime/fit_field.py` pattern.

**(b) Recurrent curriculum** (for the latent/closed-loop params) — the
connectome-cx scheme, the single most important lesson:

- **FREE-RUN** the state (feed the model's own output back), don't teacher-force the
  whole rollout.
- **PER-FRAME loss over the whole horizon** — score every frame, *not just the final
  state*. (Slime: matching only the final net change underperformed badly; switching
  to per-frame loss is what extracted the signal.)
- **GROWING horizon curriculum**: `n_steps_schedule` like `[100,200,200,300,400,
  500,600,700,800]` per epoch (connectome `drosophila_cx_both.yaml`). For cardio,
  expect the long end to help (deterministic waves); on slime it didn't (chaos).
- **RANDOM start frames** each iteration (diversity; rebuild the window pool every
  ~25 iters for speed).
- **Soft tail** weight (`coeff_tail_loss≈0.035`) on post-horizon frames.
- **Per-stage LR decay** + **grad clip** (`≈2.5`) + skip non-finite steps.
- **Event resync**: where GT has an unpredictable event (slime: wall bounce →
  random rehead; cardio: a new trigger / boundary), teacher-force the state back to
  GT at that frame so the free-run stays consistent.

Reuse `prototype/inverse_slime/sense_trainer.py::fit_sense_curriculum` as the
template (the unroll loop, window builder `_curr_batch`, soft tail, resync).

## 6. Recipe step 3 — circular variables as (cos, sin)

Any phase/angle (rotor tip angle, wave phase): encode as **(cos θ, sin θ)**, apply
updates as **rotations** (not scalar addition), and use **MSE on (cos, sin)** as the
loss; decode with `atan2`. No von Mises needed — the Euclidean MSE on the unit-circle
embedding *is* the circular metric.

> Source: connectome-gnn-cx `graph_trainer.py:2637` (`F.mse_loss(y_hat, y_in)` on the
> 2-vector). Slime evidence (H5): (cos,sin) recurrent beat scalar-angle 0.238→0.225.
> Template: `operators.py::LearnableSense.mu_vec` + `fit_sense_recurrent_cossin`.

## 7. Recipe step 4 — identifiability by LIKELIHOOD PROFILE (do not trust point estimates)

For each parameter, **scan it across its range** with the others fixed at their
recovered values, and plot the fit loss. The shape is the verdict:
- **flat** → genuinely unidentifiable (no amount of optimisation helps);
- **a well at the truth** → identifiable (any failure is the optimiser's);
- **a well away from the truth** → the objective is **biased / model-misspecified**
  (fix the model, not the optimiser).

> Slime saga: a point estimate of `sensor_angle` looked like an "identifiability
> floor"; the **profile** (`exp_profile.py`, plot `archive/profile_sensor_angle.png`)
> showed the profiles are *not flat* — the minimum is biased toward large angle. That
> changed the diagnosis from "impossible" to "fixable bias". **Always profile.**
> Template: `prototype/inverse_slime/exp_profile.py`.

## 8. Recipe step 5 — use BOTH gradient and black-box

Gradient descent for the **continuous, differentiable** interior (FHN params,
coupling gains, mechanics); **UCB/CEM** only for the **discrete / structural** choices
(which coupling family, trigger protocol topology — the Stage-4 *mechanism* search).
Don't UCB-search a parameter that has a gradient.

> Slime: gradient descent was 224× more accurate than UCB on the differentiable
> params; UCB stalled on the rugged ones. The paper's "use both" recipe.

---

## Work like a scientist — keep a logbook

Mirror `runs/logbook2.md`: every change is a **hypothesis**, a **controlled
comparison** on the same target(s), and a **verdict (validate / falsify)**. Evaluate
on **≥2 targets** (healthy `0_B` vs diseased `1_HCM`; multiple beats) so you measure
**replication** (mean ± std), not a single lucky fit. Negative results are results —
the slime campaign's value was as much in what got falsified (margin loss, noise
injection, long horizons) as in what worked.

## Suggested build order for the cardio inverse trainer

1. `cardio_operators.py` — learnable FHN / trigger / coupling / MPM ops, each with a
   `test_vs_engine()`; `run_all_tests()` from `__main__`.
2. `fit_mech.py` — one-step per-node MSE on the velocity field → mechanical + smooth
   FHN params (exact-recovery sanity, like slime's field fit).
3. `fit_cardio_curriculum.py` — free-run, per-frame, growing-horizon, random-start
   recurrent fit of the closed-loop (latent electrical) params; phase as (cos,sin).
4. `profile.py` — likelihood profiles for every recovered param before believing it.
5. `logbook.md` — hypotheses, verdicts, replication across healthy/diseased.

## Pointers

- Reusable code: `/workspace/Plexus/prototype/inverse_slime/` —
  `operators.py` (decomposed + tests), `sense_trainer.py` (one-step + curriculum +
  recurrent + cos/sin), `fit_field.py` (exact one-step), `exp_profile.py` (profiles).
- The methodology log it came from: `prototype/inverse_slime/runs/logbook2.md`.
- Cardio model + data: `cardio.tex`, `cardio_data.py`, `cardio_real.npz`, `README.md`.

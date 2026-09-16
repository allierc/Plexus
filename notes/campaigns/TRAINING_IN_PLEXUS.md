# Training in Plexus: learnables as operator substitutions

Plan, not yet implemented. Written for whoever implements it — assumes familiarity with
`src/plexus/operators/`, `engine.py` and `src/plexus/tasks/`, and with nothing else.

## The claim this rests on

**An operator's contract is already the substitutability test.** A learnable may replace an
operator exactly when it satisfies that operator's declared signature, and the signature is
written down today: `INPUTS`, `OUTPUTS`, `READS`, `WRITES`, `EMIT`, `INTEGRAND`. So none of this
needs a new language — it needs a third variant axis beside the two the registry already has,
distinguished by **who determines the law**:

| axis | who decides | example |
|---|---|---|
| `model` | the modeller — a different biological hypothesis | `neuron_signal[type_pre]` vs `[type_pairwise]` |
| `implementation` | the numericist — same law, different arithmetic | `organ_mechanics[explicit]` vs the implicit default |
| **`learnable`** | **the data** | `neuron_signal[siren]`, fitted |

The forward spec does not change. A `learnable:` block names operators to substitute; everything
else about the model stays as written, so the same file describes the analytic model and the
fitted one and the two can be run against each other.

## What already exists, so the plan does not re-derive it

- `engine.run(grad=True)` keeps the autograd tape across a whole rollout. Verified in use by
  `prototype/inverse_slime`.
- Operators are `nn.Module`s already (they carry `register_buffer`; they are called through
  `__call__`). Collecting their parameters is a walk, not a redesign.
- `plexus.tasks` generates `(stimulus, target)` corpora with the truth in `teacher.pt`, and
  `spectral_coverage` decides identifiability **before** fitting.
- `plexus.tasks.trainer` is a working reference trainer, but outside the engine. Its
  `test_trainer_circuit_is_arithmetically_the_operators` is the parity discipline to keep.
- `config/task/t*` is a 10-rung ladder with analytic answers, which is the test corpus below.

## R0 — a batch axis in the engine

**The only real blocker.** Fitting is hundreds of trials × hundreds of epochs and
`engine.run(grad=True)` runs one world.

**Do not use trials-as-entities.** It would reuse existing machinery, and it would make the
hierarchy assert something false: 32 trials are 32 independent runs of *one* world, not 32 worlds
coexisting, and `Aggregate` over them would be meaningless. A batch is a **numerical** fact, so it
belongs where numerical facts live.

`Level.state` becomes `[B, N, W]` with `B = 1` the default. Invisible to the language — no spec
key, no operator change that is not a broadcast. Touches `engine.build`, `_integrate`, the delta
accumulators, and `H.gather` / `H.scatter_along` (which index `N`, so they gain a leading axis).

**Gate.** `config/neural/ctrnn_eyeG_rig.yaml` run at `B = 32` with 32 different prescribed inputs
must reproduce 32 separate `B = 1` runs to 1e-6 of gaze. Nothing else proves the axis is inert.

## R1 — parameters, collected not declared

`H.parameters()` walks the instantiated operators and yields their `nn.Parameter`s. An operator
with none contributes none; nothing changes for the ~146 that have none.

**Gate.** A spec whose one operator holds a tensor leaf returns exactly that leaf, and
`loss.backward()` on the final state puts a finite gradient on it.

## R2 — the learnable registry

New file `src/plexus/learnables/registry.py`, deliberately **not** folded into
`models/registry.py`: an operator is a mechanism and a learnable is an approximator, and one
registry holding both would make "how many mechanisms does Plexus have" unanswerable — the
question `catalog.py` used to exist to answer.

```python
@register_learnable("siren", family="implicit")
class Siren(Learnable):
    """Sinusoidal implicit network. Fits a smooth scalar law of a few inputs."""
```

First five, chosen because they fail differently:

| family | good at | the case it is for |
|---|---|---|
| `mlp` | anything, badly | the baseline that must be beaten |
| `siren` | smooth laws of few inputs, high frequency | `neuron_signal`'s ψ, `muscle_pose_map`'s g |
| `ngp` | sharp spatial structure | a field-coupled exchange |
| `table` | per-type or per-edge lookup, no smoothness assumed | a transfer function **per cell type** |
| `gnn` | a law over a relation | anything traversing `pre`/`post` |

Each declares the contract it can satisfy — which blocks it reads, what it emits — so
substitutability is checked against the learnable's own declaration, not guessed from its shape.

## R3 — the substitutability check, and the part that is missing

Block-level matching is necessary and **not sufficient**, and the counter-example is the one that
matters: replace `neuron_signal` with an MLP on each neuron's own state and it satisfies every
declared field — same sets, same blocks, same `EMIT` — while silently dropping the connectome. It
would train, score well, and mean nothing.

What separates them is the **relation traversed**, which is exactly the query `MAPS` claimed to
answer and could not be trusted for (declared on 23 of ~150 operators; removed in 6f9a86b0). This
is its first real consumer, so bring it back **derived, not declared**: instrument `H.gather` and
`H.scatter_along`, run the spec corpus once, and record per operator which legs it actually walked.

The check then has four parts, in order of how quietly each fails:

1. `EMIT` and `INTEGRAND` must match **exactly** — the engine resolves a whole set's integration
   order from `EMIT`, so a mismatch changes how the set moves in time with nothing in the output
   to show for it.
2. `WRITES` must match exactly.
3. `READS` may **narrow** but never widen. A learnable reading a block the operator did not is a
   different model wearing its name.
4. The measured relation must match. This is the connectome case above.

**Gate.** Three refusals, each asserted: a learnable emitting `velocity` where the operator emits
`acceleration`; one reading a block the operator does not; one traversing no relation where the
operator traverses `pre`/`post`. Plus one acceptance that runs.

## R4 — the spec surface

```yaml
learnable:
  - replaces: neuron_signal
    at: neuron
    with: siren
    params: {hidden: 64, layers: 3, omega0: 30}
    init_from: operator          # pre-fit to the law it replaces, then depart from it

training:
  task: t2_resonator_damping     # a corpus under graphs_data/task/
  epochs: 200
  batch: 32
  lr: 1e-3
  seed: 0
  curriculum: {horizon: [0.05, 0.25, 0.5, 1.0]}
  score: [mse, pole_recovery]
```

`init_from: operator` is the part worth arguing for. Pre-fit the network to the operator it
replaces so training **starts at the known law**. Then what the learnable does afterwards is the
residual, and a residual measured against a stated model is interpretable — which is the paper's
Loop III claim made operational rather than asserted. Starting from noise discards it.

**The loss does not go in the engine.** `training:` carries hyperparameters; the objective is a
function of the trajectory *and the task*, and the engine knows nothing about tasks. `run(grad=True)`
returns a Hierarchy with live tensors and the driver forms the loss. The moment the engine owns a
loss, every spec acquires an opinion about what it is being fitted to.

## R5 — trainer, tester, analyser

Three programs, because they answer three questions and fail independently.

```
log/<name>/                      TRAINING writes here
    config.yaml                  the run spec, copied — self-describing
    models/best.pt               by validation score
    models/epoch_*.pt            checkpoints for the analyser's trajectory-over-training
    training.log
    results/                     TESTER and ANALYSER write here
        <name>_test.json         held-out rollout, per condition cell
        <name>_traces.png        target vs prediction, residual on the same scale
        <name>_recovery_<q>.png  one per recovered quantity
        <name>_bode.png          measured vs true transfer function
        report.json
```

**trainer** — `-o train`. Fits. Writes checkpoints and the loss history. Knows nothing about
whether the fit is any good.

**tester** — `-o test`. Rolls the checkpoint out on held-out data. One number per condition cell,
never only an aggregate: a grid exists to be read per cell.

**analyser** — `-o analyse`. The one that says whether the law was recovered, and the only one
whose output is an argument rather than a measurement. Mirror
`connectome-gnn/src/connectome_gnn/plot.py:plot_recovery_panels` — a 2×2 used for **every**
recovered quantity so all of them are read the same way:

- true vs learned scatter, with R², slope, N, RMSE
- histogram of the **error** (learned − true), in the quantity's own units
- histogram of the **relative** error, which makes a small parameter's 10% miss comparable to a
  large one's
- violins of |error| per group, sorted by the group's mean true value

For an LTI task the recovered quantities are the **poles** — and unlike a connectome fit, the
truth is analytic and complete, recorded in `teacher.pt` at generation. So the analyser also
drives the trained model with the matching `probe_chirp_*` corpus, takes the ratio of output to
input spectra, and overlays the measured Bode against the teacher's closed form.

## The RLC ladder as the test, and what each rung is for

The ten corpora already exist and each fails differently. Use them as the acceptance suite, not as
a demo:

| task | what a learnable must show |
|---|---|
| `t0_gain_unity`, `t0_delay_100ms` | the harness works; a failure here invalidates everything above |
| `t1_integrator_perfect` | a pole at the origin — a line attractor, the hardest thing to hold |
| `t2_resonator_damping` | ζ = 0.1 / 0.4 / 0.9 — does the learnable get damping, or only frequency |
| `t2_eye_plant` | 3-in 3-out **coupled** — the MIMO case, where a per-channel learnable must fail |
| `t3_lowpass_order` | order 2 / 4 / 8 — how many poles can a learnable of a given size hold |
| `t4_unexcited_12hz` | **the negative control.** Must score well and recover the poles badly. A learnable that "succeeds" here has told you the analyser is broken, not that the fit is good |

`t4` is the rung that validates the analyser. The reference trainer already reproduced it: lowest
error of eight, worst dynamics of eight.

## Known gap this plan does not close

**Contextual unidentifiability.** Measured on the reference trainer: grids over the *teacher*
average 0.326 normalised MSE against 0.0009 for no grid — a factor of 360 — because
`t1_integrator_tau_sweep` asks one circuit to be a τ = 0.5 s integrator *and* a τ = 32 s one with
nothing in the input saying which. `spectral_coverage` passes all three, because it tests the
spectral kind of unidentifiability and not this one. The fix is a second check in
`tasks/schema.py`: refuse a teacher-varying grid unless the stimulus varies with it or the cell
identity is an input channel. Independent of everything above; do it before trusting any result
on those three tasks.

## Order, and why

R0 → R1 → R3 → R2 → R4 → R5. The batch axis first because it blocks everything and is
self-contained; parameter collection next because it is small and R2 needs it; the **derived
relation check before the registry**, because without it the registry would accept the
connectome-dropping MLP and the first thing anyone tries would be wrong in a way that trains
fine. R0, R1 and R3 are each worth having even if `learnable:` is never written.

---

# Progress

## Done

**R1, R3** (`f03f084e`) — `H.operators` / `H.parameters()`, and `H.measured_maps()`. The whole
promoted library returns zero parameters, asserted, so a future operator quietly acquiring one is
a red test. Relations are measured through `gather` / `scatter_along` / `lift_index`; instrumenting
`lift_index` was not optional, since an Aggregate traverses containment there and
`muscle_pose_map` would otherwise have measured as walking nothing.

**R2** (`c140fb43`) — six learnable families (three nets × two shapes) and `check_substitution`.
The refusals are the content: pointwise-for-relational, relational-for-pointwise, wrong `EMIT`,
widened `READS`, a relational substitution with no edge set. A live substitution on
`ctrnn_eyeG_rig` measures the same legs as `neuron_signal` itself.

**R4** (`5c78e63c`, `39f4f88b`) — `learnable:` in the spec language, both forms, plus
`spec_trainer` and the eye rig.

## The eye rig, measured

`config/run/eye_rig_fit*.yaml` fits W_in, W and W_out of the 64-unit circuit through the frozen
eye against `t5_gaze_tracking`, by running the spec — not a transcription of it.

| | best val | of variance | what changed |
|---|---|---|---|
| 12 epochs, no curriculum | — | 0.647 | the starting point |
| 60 epochs, curriculum, `accum: 1` | 18.06 deg² | 0.446 | truncated BPTT, 60 → 480 frames |
| 60 epochs, curriculum, `accum: 8` | **2.48 deg²** | **0.061** | gradient averaged over 8 trials |

Two things that mattered more than expected. **The horizon curriculum** is not a refinement: a
480-step rollout from an unfitted W sends every late frame's gradient through hundreds of tanh's
and it arrives uninformative. **Stepping per trial cost a factor of seven** — validation bouncing
27 → 76 deg² between consecutive epochs was an optimiser handed a different task each step, not a
failure to learn.

Gradient accumulation is **not** a substitute for R0. Both 60-epoch runs took 1831 s for the same
3,360 rollouts; it buys the variance reduction and none of the wall-clock. Training at the scale
`tasks.trainer` uses (512 trials) still needs the engine to carry B trials in one rollout.

## Still open

**R0 — the batch axis.** Now the only thing between this and training at scale. The gate stands:
`ctrnn_eyeG_rig` at B = 32 must reproduce 32 separate B = 1 runs to 1e-6 of gaze.

**R5 — tester and analyser.** `spec_trainer` has `-o train` only. The analyser is the one that
says whether the law was recovered rather than whether the output is close, and for these tasks
the truth is analytic and in `teacher.pt`.

**Contextual unidentifiability**, unchanged: grids over the *teacher* average 0.326 normalised MSE
against 0.0009 for no grid, and `spectral_coverage` passes all of them.

## Two traps worth not repeating

`python3` is the system interpreter and has neither torch nor yaml. A heredoc using it failed
silently and a training run launched against a spec that was never written.

A command piped through `grep` reports **grep's** exit code. A run that crashed was recorded as
successful, and the traceback was filtered out by the pattern meant to summarise it.

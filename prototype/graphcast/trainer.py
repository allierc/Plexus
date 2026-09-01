"""The TRAINER ENGINE. The fitting counterpart of `plexus.engine.run`, built the same way.

`plexus.engine.run(spec, out)` builds the operators a spec names, applies them in the order the
`schedule:` gives, and records the result. This does the same for the `trainer:` section: it builds
the operators that section names, applies them in the order its schedule gives, and records the
result. The loop body below is short on purpose -- everything that could differ between two fits
lives in the operators, which is the claim.

TWO SCHEDULES, TWO CLOCKS, ONE FILE:

    model:    fields + operators + schedule    ->  a Plexus HIERARCHY, stepped to make a prediction
    trainer:  operators + schedule             ->  predict -> loss -> regularize -> step

The first is a Plexus spec in every sense: `plexus.schema` validates it and `plexus.engine` builds
it (see `model_hierarchy.py`). The second is a schedule over the trainer's own five roles. They are
deliberately separate vocabularies -- `plexus.models.base.KINDS` says how an operator moves data
inside a hierarchy during a step, and a loss does not do that -- but they are the same IDEA applied
twice, which is what makes the fit composable rather than configurable.

WHAT IS NOT IN THIS FILE, AND MUST NOT DRIFT BACK IN: no loss expression, no regularisation
coefficient, no learning rate, no optimiser choice, no clip value, no dataset name. If a number
appears here it is a bug in the same way a dataset name in `engine.py` is a bug (G2). What this
file knows is how to read a batch out of a recorded trajectory and how to run two lists in order.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

import ops_trainer
from model_hierarchy import ModelHierarchy


def _load_field(run_dir: str, name: str, device: str) -> torch.Tensor:
    """The recorded field as [T, *res].

    Read out of the ZARR rather than off a live hierarchy, because the zarr is the artifact a
    consumer sees -- the same discipline the generator follows, and it means the trainer is
    exercising the archived data rather than a privileged in-memory copy of it.
    """
    import zarr
    z = zarr.open(os.path.join(run_dir, "field.zarr"), "r")
    if name not in z:
        raise KeyError(f"field {name!r} not in {run_dir}/field.zarr "
                       f"(has {sorted(z.group_keys())})")
    return torch.as_tensor(np.asarray(z[name]["grid"]), device=device)   # [T, C, *res]


def _step_weights(weighting: str, n_steps: int, gamma: float) -> list[float]:
    """Per-step weights over a rollout: uniform | discount | linear_decay | last.

    Copied in form from `connectome_gnn.models.recurrent_step._rollout_step_weights`, including the
    property that matters: EVERY SCHEME RETURNS [1.0] AT K = 1, so the recurrent objective at
    horizon 1 is arithmetically the nominal one and the two paths can be compared without a
    confound. Unnormalised; the caller divides by the weight it actually applied.

    The weekend benchmark measured `last` at -0.028 and `discount` at -0.002 in R^2_W against
    `uniform`, and GraphCast scores every lead time uniformly. `uniform` is the default for both
    reasons, and the others exist so that finding can be re-checked here rather than assumed.
    """
    if weighting == "uniform":
        return [1.0] * n_steps
    if weighting == "discount":
        return [gamma ** k for k in range(n_steps)]
    if weighting == "linear_decay":
        return [(n_steps - k) / n_steps for k in range(n_steps)]
    if weighting == "last":
        return [0.0] * (n_steps - 1) + [1.0]
    raise ValueError(f"unknown rollout weighting {weighting!r} (expected uniform, discount, "
                     f"linear_decay or last)")


def _load_all(model, batch, b):
    """Put sample `b` of every INITIALISED field into the hierarchy."""
    for name, arr in batch.items():
        model.load(name, arr[b])


def _predict_increment(model, field, x, n_ticks):
    """The model's INCREMENT over `n_ticks` of its own schedule, per tick. `x` is [B, C, *res].

    ONE HIERARCHY, ONE SAMPLE AT A TIME. A Plexus hierarchy holds one system, not a batch of them,
    so a batch is a loop rather than a leading tensor dimension. Slower, and honest: pretending a
    hierarchy is batched is how a model stops being the thing it claims to simulate.
    """
    batch = x if isinstance(x, dict) else {field: x}
    n = next(iter(batch.values())).shape[0]
    out = []
    for b in range(n):
        _load_all(model, batch, b)
        before = model.read(field)
        model.step(n_ticks)
        out.append(model.read(field) - before)
    return torch.stack(out)


def _predict_state_all(model, batch, n_ticks):
    """Advance EVERY field the model carries, returning them all -- so a rollout keeps the latent
    state (`u`, `v`) as well as the observed one (`s`). Re-loading only the observation each step
    would silently re-anchor the latents to their initial values and make a K-step rollout a
    sequence of one-step predictions."""
    n = next(iter(batch.values())).shape[0]
    outs = {k: [] for k in batch}
    for b in range(n):
        _load_all(model, batch, b)
        model.step(n_ticks)
        for k in batch:
            outs[k].append(model.read(k))
    return {k: torch.stack(v) for k, v in outs.items()}


def _predict_state(model, field, x, n_ticks):
    """The model's STATE after `n_ticks`. The rollout path hands the model back its own output."""
    batch = x if isinstance(x, dict) else {field: x}
    n = next(iter(batch.values())).shape[0]
    out = []
    for b in range(n):
        _load_all(model, batch, b)
        model.step(n_ticks)
        out.append(model.read(field))
    return torch.stack(out)


def _train_step_nominal(train, model, field, losses, stride, batch_frames, device):
    """ONE STEP. The model takes a single tick and is scored against a single recorded increment.

    The target is divided by the stride because a recorded pair spans that many simulation frames
    while the model's tick is one; the PREDICTION is not, because it already is a per-frame
    increment. Dividing both was a real bug here and scaled every recovered constant by the stride.
    """
    obs = train[field]
    j = torch.randint(0, obs.shape[0] - 1, (batch_frames,), device=device)
    x = {k: v[j] for k, v in train.items()}
    target = (obs[j + 1] - obs[j]) / stride
    pred = _predict_increment(model, field, x, 1)
    return sum(op.forward(pred, target) for op in losses)


def _train_step_recurrent(train, model, field, losses, stride, batch_frames, device,
                          horizon, weighting, gamma):
    """ROLLOUT. Unroll `horizon` records from one start, FEEDING THE MODEL ITS OWN OUTPUT, and score
    every step against what was actually recorded.

    WHY THIS EXISTS AND WHAT IT IS FOR HERE. It is not a general accuracy improvement -- the weekend
    benchmark found plain t+1 wins as an objective, and GraphCast spends 96% of its updates at K=1
    before a short tail at a learning rate 3,300x below peak. It is here because of a specific
    IDENTIFIABILITY failure measured on this toy:

        r_i = omega_i + K * coupling_i,   omega FREE PER PIXEL

    At any single instant `omega` can absorb `K * coupling` entirely, so a one-step loss barely
    constrains `K` -- measured, it reached 0.037 against a true 0.90 while omega's spread came out
    almost exactly right. What distinguishes the two is that the coupling CHANGES OVER TIME and
    omega does not, and a rollout is what puts that difference in front of the loss.

    BPTT RUNS THROUGH THE WHOLE UNROLL, not a truncated window. GraphCast does not truncate, and the
    benchmark's `pushforward` arm -- a one-step BPTT window -- was the worst of the five at -0.082.
    The state is never detached between steps.

    ONE STEP IS ONE RECORD, so the model takes `stride` ticks per rollout step; comparing a one-tick
    prediction against a record that is `stride` frames later would silently fit a velocity that is
    `stride` times too small.
    """
    obs = train[field]
    K = int(horizon)
    hi = obs.shape[0] - K
    if hi <= 0:
        raise ValueError(f"rollout horizon {K} needs {K + 1} records; the split holds "
                         f"{obs.shape[0]}")
    j = torch.randint(0, hi, (batch_frames,), device=device)
    w = _step_weights(weighting, K, gamma)
    # THE LATENT FIELDS ARE INITIALISED ONCE, at the start of the window, and then never again:
    # after step 1 the model carries its OWN u and v forward, so only the first frame is given.
    state = {k: v[j] for k, v in train.items()}
    total, wsum = 0.0, 0.0
    for k in range(K):
        state = _predict_state_all(model, state, stride)
        if w[k] == 0.0:
            continue
        total = total + w[k] * sum(op.forward(state[field], obs[j + k + 1]) for op in losses)
        wsum += w[k]
    return total / max(wsum, 1e-12)


def run(fit, out_root: str, device: str = "cuda", log_every: int = 50) -> dict:
    """Fit the spec's own model to the run its `fit.data.run` names. Returns the history."""
    fb = fit.fit
    field = fb.field
    run_dir = fb.run if os.path.isdir(fb.run) else os.path.join(out_root, fb.run)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"fit.data.run is {fb.run!r}; no such run directory "
                                f"(looked in {out_root}). Generate it first.")
    # `init:` NAMES THE FIELDS THE MODEL IS STARTED FROM; `field:` names the one it is SCORED on.
    # For a single-mechanism fit they are the same. For a fit on the SUM they are not: the model is
    # given the latent u and v at the start of a window and scored on the observation s, which
    # separates "can the parameters be recovered through a superposition" from "can the state be
    # estimated from it" -- two questions that a single number would confound.
    init_names = list(fb.data.get("init") or [field])
    fields = {n: _load_field(run_dir, n, device).float() for n in dict.fromkeys(init_names + [field])}
    u = fields[field]
    n_rec = u.shape[0]

    # THE RECORD STRIDE IS PART OF THE TARGET'S UNIT. Records are `stride` simulation frames apart,
    # so the increment between two records is that many frames of change. A rate quoted per FRAME
    # must divide by it. Getting this wrong scales every recovered constant by a constant factor
    # and looks exactly like a converged fit to the wrong answer -- the most expensive kind.
    stride = fb.record_stride or max(1, round(fit.n_frames / max(1, n_rec - 1)))
    lo, hi = fb.split.get("train", [0, fit.n_frames])
    a, b = int(lo * n_rec / fit.n_frames), int(hi * n_rec / fit.n_frames)
    train = {k: v[a:b] for k, v in fields.items()}

    # THE MODEL IS THE SPEC. Loaded from the same file, by plexus.schema, as a Plexus spec.
    model = ModelHierarchy(fit.path, device=device)
    # THE SUPPORT, read off the data: a cell the field is never non-zero at is outside the mask.
    # Exact here because the generator multiplies by it, and it keeps the mask a property of the
    # OBSERVATION rather than a second copy of the generator's config that could drift from it.
    # A MASK PER FIELD, and each operator gets the one belonging to the field it acts on. Using
    # the observation's mask everywhere would hand the Kuramoto operator a mask covering the whole
    # domain, since the sum is non-zero everywhere the coarse rule is.
    masks = {n: (v.abs().amax(dim=0).amax(dim=0) > 1e-12).to(v.dtype) for n, v in fields.items()}
    model.bind_shapes({op_name: masks.get(model.at_of(op_name), masks[field])
                       for op_name in model.names})

    ctx = {"field": field, "dim": fit.dim, "device": device, "stride": stride, "model": model}
    ops = {r: [] for r in ops_trainer.ROLES}
    for line in fb.operators:
        op = ops_trainer.build(line, ctx)
        ops[op.ROLE].append(op)
    for role in fb.schedule:
        if not ops.get(role):
            raise ValueError(f"fit.schedule names role {role!r} but no operator declares it")
    if len(ops["step"]) != 1:
        raise ValueError(f"a fit needs exactly one `step` operator, got {len(ops['step'])}")
    if not ops["loss"]:
        raise ValueError("a fit needs at least one `loss` operator")

    step = ops["step"][0]
    named = model.named_parameters()
    step.bind(named)

    ro = fb.rollout or {}
    recurrent, horizon = bool(ro.get("recurrent", False)), int(ro.get("horizon", 1))
    print(f"model     {model.describe()}")
    print(f"fit       {' -> '.join(fb.schedule)}")
    for role in fb.schedule:
        for op in ops[role]:
            print(f"  {op.describe()}")
    print(f"learn     {', '.join(f'{k}{tuple(v.shape)}' for k, v in named.items())}")
    print(f"data      {run_dir}  records [{a}, {b}) of {n_rec}, stride {stride} sim-frames")
    print(f"objective {'rollout, horizon %d, %s' % (horizon, ro.get('weighting', 'uniform')) if recurrent else 'one step (t+1)'}")

    hist = {"iter": [], "loss": [], **{k: [] for k in named}}
    for it in range(step.n_iter):
        # THE ONE BRANCH IN THE LOOP, and it is the same partition connectome-gnn's graph_trainer
        # makes between `run_nominal_train_step` and `run_recurrent_train_step`: each returns the
        # data loss and the caller owns backward, the step and the logging tail.
        if recurrent:
            total = _train_step_recurrent(train, model, field, ops["loss"], stride,
                                          fb.batch_frames, device, horizon,
                                          ro.get("weighting", "uniform"),
                                          float(ro.get("gamma", 0.5)))
        else:
            total = _train_step_nominal(train, model, field, ops["loss"], stride,
                                        fb.batch_frames, device)
        data_loss = float(total.detach())
        for op in ops["regularize"]:
            pen = op.forward(named)
            if pen is not None:
                total = total + pen
        step.set_lr(it)
        step.step(total, named)
        if it % log_every == 0 or it == step.n_iter - 1:
            hist["iter"].append(it)
            hist["loss"].append(data_loss)
            for k, p in named.items():
                # A SCALAR IS LOGGED WHOLE; A FIELD IS LOGGED AS STATISTICS. `omega` is one number
                # per pixel -- 1,048,576 of them -- and writing it 30 times produced a 7.5 MB
                # trainer.json that no one can read and that says less than four numbers would.
                v = p.detach()
                hist[k].append(float(v) if v.ndim == 0 else
                               {"mean": float(v.mean()), "std": float(v.std()),
                                "min": float(v.min()), "max": float(v.max())})

    out = {"history": hist,
           "learned": {k: (float(p.detach()) if p.ndim == 0 else
                           {"shape": list(p.shape), "mean": float(p.detach().mean()),
                            "std": float(p.detach().std()), "file": f"{k}.npy"})
                       for k, p in named.items()},
           "train_records": [a, b], "record_stride": stride, "n_iter": step.n_iter,
           "model_schedule": model.describe()}
    # THE FIELDS THEMSELVES GO BESIDE THE JSON, as .npy. They are the scientific output -- omega
    # IS the heterogeneity map -- so they must be kept; they just do not belong inside a summary.
    for k, p in named.items():
        if p.ndim:
            np.save(os.path.join(run_dir, f"{k}.npy"), p.detach().cpu().numpy())
    with open(os.path.join(run_dir, "trainer.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

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


def _train_step_nominal(train, predict, losses, stride, batch_frames, device):
    """ONE STEP. The model takes a single tick and is scored against a single recorded increment.

    The target is divided by the stride because a recorded pair spans that many simulation frames
    while the model's tick is one; the PREDICTION is not, because it already is a per-frame
    increment. Dividing both was a real bug here and scaled every recovered constant by the stride.
    """
    j = torch.randint(0, train.shape[0] - 1, (batch_frames,), device=device)
    x = train[j]
    target = (train[j + 1] - x) / stride
    pred = predict.forward(x)
    return sum(op.forward(pred, target) for op in losses)


def _train_step_recurrent(train, predict, losses, stride, batch_frames, device,
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
    K = int(horizon)
    hi = train.shape[0] - K
    if hi <= 0:
        raise ValueError(f"rollout horizon {K} needs {K + 1} records; the split holds "
                         f"{train.shape[0]}")
    j = torch.randint(0, hi, (batch_frames,), device=device)
    w = _step_weights(weighting, K, gamma)
    s = train[j]
    total, wsum = 0.0, 0.0
    for k in range(K):
        s = predict.state_after(s, stride)          # the model's own state, never re-anchored
        if w[k] == 0.0:
            continue
        total = total + w[k] * sum(op.forward(s, train[j + k + 1]) for op in losses)
        wsum += w[k]
    return total / max(wsum, 1e-12)


def run(fit, run_dir: str, device: str = "cuda", log_every: int = 50) -> dict:
    """Execute `fit.trainer` against the trajectory in `run_dir`. Returns the history."""
    tr = fit.trainer
    field = tr.field
    u = _load_field(run_dir, field, device).float()
    n_rec = u.shape[0]

    # THE RECORD STRIDE IS PART OF THE TARGET'S UNIT. Records are `stride` simulation frames apart,
    # so the increment between two records is that many frames of change. A rate quoted per FRAME
    # must divide by it. Getting this wrong scales every recovered constant by a constant factor
    # and looks exactly like a converged fit to the wrong answer -- the most expensive kind.
    stride = tr.record_stride or max(1, round(fit.n_frames / max(1, n_rec - 1)))
    lo, hi = fit.data.split["train"]
    a = int(lo * n_rec / fit.n_frames)
    b = int(hi * n_rec / fit.n_frames)
    train = u[a:b]

    model = ModelHierarchy(tr.model,
                           {"name": f"{fit.name}_fit", "dim": fit.dim, "seed": fit.seed,
                            "dt": fit.dt, "n_frames": 1,
                            "world": fit.general.get("world", 1.0),
                            "boundary": fit.general.get("boundary", "periodic"),
                            "units": fit.general.get("units")},
                           run_dir, device=device)
    # THE SUPPORT OF THE FINE RULE, read off the data: a pixel the field is never non-zero at is
    # outside the mask. Exact here because the generator multiplies by the mask, and it keeps the
    # mask a property of the OBSERVATION rather than a second copy of the generator's config that
    # could drift away from it.
    mask = (u.abs().amax(dim=0).amax(dim=0) > 1e-12).to(u.dtype)   # over time AND channels
    model.bind_shapes({n: mask for n in model.names})

    ctx = {"field": field, "dim": fit.dim, "device": device,
           "stride": stride, "model": model}
    ops = {r: [] for r in ops_trainer.ROLES}
    for line in tr.operators:
        op = ops_trainer.build(line, ctx)
        ops[op.ROLE].append(op)
    for role in tr.schedule:
        if not ops.get(role):
            raise ValueError(f"trainer.schedule names role {role!r} but no operator declares it")
    for role in ("predict", "step"):
        if len(ops[role]) != 1:
            raise ValueError(f"a trainer needs exactly one {role!r} operator, got {len(ops[role])}")

    predict, step = ops["predict"][0], ops["step"][0]
    named = {k: p for k, p in predict.parameters().items() if p.requires_grad}
    if not named:
        raise ValueError("the model hierarchy exposes no learnable parameter; "
                         "does an operator's `learn:` name anything?")
    step.bind(named)

    print(f"model schedule    {model.describe()}")
    print(f"trainer schedule  {' -> '.join(tr.schedule)}")
    for role in tr.schedule:
        for op in ops[role]:
            print(f"  {op.describe()}")
    print(f"learnable         {', '.join(f'{k}{tuple(v.shape)}' for k, v in named.items())}")
    print(f"train records     [{a}, {b}) of {n_rec}, stride {stride} sim-frames")

    ro = tr.rollout or {}
    recurrent = bool(ro.get("recurrent", False))
    horizon = int(ro.get("horizon", 1))
    print(f"objective         {'recurrent, horizon %d, %s weights' % (horizon, ro.get('weighting', 'uniform')) if recurrent else 'one step (t+1)'}")

    hist = {"iter": [], "loss": [], **{k: [] for k in named}}
    for it in range(step.n_iter):
        # THE ONE BRANCH IN THE LOOP, and it is the same partition connectome-gnn's graph_trainer
        # makes between `run_nominal_train_step` and `run_recurrent_train_step`: each returns the
        # data loss and the caller owns backward, the step and the logging tail.
        if recurrent:
            total = _train_step_recurrent(train, predict, ops["loss"], stride, tr.batch_frames,
                                          device, horizon, ro.get("weighting", "uniform"),
                                          float(ro.get("gamma", 0.5)))
        else:
            total = _train_step_nominal(train, predict, ops["loss"], stride, tr.batch_frames,
                                        device)
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

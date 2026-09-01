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

    hist = {"iter": [], "loss": [], **{k: [] for k in named}}
    for it in range(step.n_iter):
        j = torch.randint(0, train.shape[0] - 1, (tr.batch_frames,), device=device)
        x = train[j]
        target = (train[j + 1] - x) / stride            # increment per SIM-FRAME
        # NOT DIVIDED BY THE STRIDE. The model steps its own schedule once at its own dt, so its
        # increment is already per SIM-FRAME; the TARGET is what needs dividing, because a recorded
        # pair spans `stride` of them. Dividing both scaled the recovered velocity by the stride.
        pred = predict.forward(x)
        total = sum(op.forward(pred, target) for op in ops["loss"])
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

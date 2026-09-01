"""The TRAINER: a fit written as a schedule of operators, the same way a simulation is.

THE CLAIM, and it is the whole reason this file exists. `plexus.engine.run` takes a `schedule:` --
a list of named operators, each with its params -- and applies them in order. That is not a
simulation-specific idea; it is a way of writing a composition down so that every term is named,
parameterised from the file, and separately inspectable. A TRAINING STEP HAS THE SAME SHAPE. So the
`training:` section stops being a bag of scalars and becomes a second schedule:

    forward   schedule:  advect_field -> kuramoto_field                     what the world does
    fitting   schedule:  predict -> loss -> regularize -> step              what we do about it

What this buys, one line each:

  * A LOSS STOPS BEING A HARD-CODED LINE IN A TRAIN LOOP. `coeff_g_phi_diff: 750` is the largest
    term in the production objective and it is a number with no operator attached; as
    `- op: smoothness_reg` with `coeff: 750` it has a name, a signature and a place in a printed
    schedule.
  * THE RESIDUAL BECOMES ATTRIBUTABLE TO A MECHANISM, which is the reason the prototype is in
    Plexus at all (plexus2.tex, mechanistic inverse modelling). That only holds if each learnable
    thing is its own operator.
  * `known_ode` AND `gnn` BECOME TWO IMPLEMENTATIONS OF ONE ROLE rather than two code paths.
  * AN ABLATION BECOMES AN EDIT TO A LIST. Dropping the regulariser is deleting a line.

WHY THE ROLES ARE A SEPARATE VOCABULARY FROM `plexus.models.base.KINDS`, and this was a real
decision rather than a default. The simulation kinds -- lateral / aggregate / broadcast / exchange /
field / rewire / structural / seed -- say HOW AN OPERATOR MOVES DATA INSIDE A HIERARCHY DURING A
STEP. A trainer operator does something categorically different: it consumes a trajectory and
returns a scalar, or it mutates parameters. Overloading `kind` would have made one word mean two
things, and `kind` is what the registry dispatches on. So trainer operators carry a `ROLE` and the
two vocabularies never mix.

    ROLE          signature                                    implementations here
    loss          prediction, target -> scalar                 mse_loss
    regularize    parameters -> scalar                         l1_reg, l2_reg
    step          scalars, parameters -> updated parameters    adamw

THERE IS NO `predict` ROLE, AND THERE MUST NOT BE. The spec's OWN `operators:` and `schedule:` are
the model -- a fit spec is a Plexus spec whose operators are the learnable twins of a simulation's.
Prediction is therefore running that schedule, and an operator whose job is "run the model" would
name the framework rather than a mechanism. An earlier version had exactly that, called
`plexus_model`, and it was unreadable for precisely this reason.

Nor is there a `graph` role. Building a neighbour graph changes the edge set, which is ALREADY a
Plexus kind -- `radius_graph` is `kind="rewire"` in `operators/interaction_ops.py` -- so it belongs
in the model's own schedule, exactly as `config/active_matter/vicsek_4t.yaml` writes it:

    operators: [{op: radius_graph, at: particle, radius: 0.05}, {op: velocity_align, ...}]
    schedule:  [radius_graph, velocity_align]
"""

from __future__ import annotations

import math

import torch

ROLES = ("loss", "regularize")

_REGISTRY: dict[str, type] = {}


def register_trainer_op(name: str):
    """Same shape as `plexus.models.registry.register_operator`, for the trainer vocabulary."""
    def wrap(cls):
        if cls.ROLE not in ROLES:
            raise ValueError(f"{name}: ROLE {cls.ROLE!r} is not one of {ROLES}")
        cls.NAME = name
        _REGISTRY[name] = cls
        return cls
    return wrap


def build(line: dict, ctx: dict):
    """One `- op: ...` line from `trainer.operators` into an instance. `ctx` carries the run."""
    name = line.get("op")
    if name not in _REGISTRY:
        raise ValueError(f"unknown trainer operator {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](line, ctx)


class TrainerOp:
    ROLE: str = ""
    NAME: str = ""

    def __init__(self, params: dict, ctx: dict):
        self.params = params
        self.ctx = ctx

    def describe(self) -> str:
        keys = [k for k in self.params if k != "op"]
        return f"{self.NAME}({self.ROLE}) " + " ".join(f"{k}={self.params[k]}" for k in keys)


def build_trainer(params: dict):
    """`fit.trainer:` -> the optimiser. One of them, not a list."""
    return AdamW(params)


# ------------------------------------------------------------------ loss ------------------------
@register_trainer_op("mse_loss")
class MSELoss(TrainerOp):
    """ROLE loss. Mean square error, optionally normalised by the variance of the INCREMENT.

    `norm: increment_variance` is GraphCast's `s_j` (supplement 4.2): every target is unit-variance
    AS AN INCREMENT, not as a state, which is what lets one loss weight transfer across quantities
    of very different activity. Off by default because on a single-field toy it is a no-op scaling.
    """

    ROLE = "loss"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.norm = params.get("norm", "none")
        if self.norm not in ("none", "increment_variance"):
            raise ValueError(f"mse_loss norm {self.norm!r} is not none|increment_variance")

    def forward(self, pred, target):
        r = (pred - target) ** 2
        if self.norm == "increment_variance":
            r = r / target.var().clamp_min(1e-30)
        return r.mean()


# ------------------------------------------------------------------ regularize ------------------
class _Penalty(TrainerOp):
    ROLE = "regularize"
    POWER = 1

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.coeff = float(params.get("coeff", 0.0))
        self.on = list(params.get("params", []))

    def forward(self, named):
        if self.coeff == 0.0:
            return None                     # a zero coefficient contributes NOTHING, not 0.0*x
        hit = [named[k] for k in self.on if k in named]
        if not hit:
            raise ValueError(f"{self.NAME} names {self.on} but the predict operator exposes "
                             f"{sorted(named)}")
        return self.coeff * sum((p.abs() ** self.POWER).sum() for p in hit)


@register_trainer_op("l1_reg")
class L1Reg(_Penalty):
    """ROLE regularize. L1, for sparsity in `W`."""
    POWER = 1


@register_trainer_op("l2_reg")
class L2Reg(_Penalty):
    """ROLE regularize. L2 AS A LOSS TERM, deliberately not as the optimiser's `weight_decay`.

    Weight decay in AdamW is applied to every parameter in a group, and the scientific outputs --
    `W`, the embedding `a`, the stimulus gain `b` -- are IN those groups. Shrinking them is not a
    prior, it is a bias in the reported answer, and that bug has already been paid for once here.
    As an explicit term it names what it acts on.
    """
    POWER = 2


# ------------------------------------------------------------------ step ------------------------
class AdamW:
    NAME = "adamw"
    ROLE = "step"

    """Not a registered operator: `fit.trainer:` is a PARAMETER BLOCK.

    An optimiser is not a term in the objective. `mse_loss` and `l1_reg` are -- they take
    tensors and return a scalar that is summed with the others, so they belong in a list and
    an ablation is deleting a line. AdamW takes the RESULT of that sum and mutates the
    parameters; there is exactly one of it, it composes with nothing, and calling it an
    operator put a singleton in a list to make it look like a composition. `fit.schedule`
    still ends with the token `step`, because the update IS the last thing an iteration
    does and the schedule should say so -- but its params are `fit.trainer:`.
    """

    """ROLE step. AdamW with PARAMETER GROUPS, a schedule, and global gradient clipping.

    The groups are the point. Production configs carry three learning rates -- `lr_W 0.0009`,
    `lr 0.0018`, `lr_embedding 0.002325` -- as three scalars whose relationship to the parameters
    is implicit in the trainer's source. Here they are what they always were: groups, written down.

    Defaults follow GraphCast where they are load-bearing and measured (supplement 4.4): beta2 =
    0.95 rather than 0.999, global grad-norm clip at 32, and cosine decay TO ZERO -- the weekend
    benchmark's one-checkpoint collapses of 0.2-0.5 in R^2_W are the symptom of never annealing.
    """

    def __init__(self, params, ctx=None):
        self.params = params
        self.n_iter = int(params.get("n_iter", 1000))
        self.grad_clip = float(params.get("grad_clip", 32.0))
        self.warmup = int(params.get("warmup_iters", 0))
        self.scheduler = params.get("scheduler", "none")
        if self.scheduler not in ("none", "cosine_to_zero"):
            raise ValueError(f"adamw scheduler {self.scheduler!r} is not none|cosine_to_zero")
        b = params.get("betas", [0.9, 0.95])
        self.betas = (float(b[0]), float(b[1]))
        self.groups_spec = params.get("groups") or [{"params": ["*"], "lr": 1.0e-3}]
        self.opt = None

    def describe(self) -> str:
        return "adamw  " + " ".join(f"{k}={v}" for k, v in self.params.items())

    def bind(self, named):
        groups, claimed = [], set()
        for g in self.groups_spec:
            want = g.get("params", ["*"])
            hit = list(named) if want == ["*"] else [k for k in want if k in named]
            if not hit:
                raise ValueError(f"adamw group {want} matches none of {sorted(named)}")
            claimed.update(hit)
            groups.append({"params": [named[k] for k in hit],
                           "lr": float(g.get("lr", 1.0e-3)),
                           "weight_decay": float(g.get("weight_decay", 0.0)),
                           "name": ",".join(hit)})
        # EVERY LEARNABLE PARAMETER MUST BE IN A GROUP. A parameter silently left out of the
        # optimiser trains at zero and reports as "did not converge", which is indistinguishable
        # from a modelling failure and costs a run to diagnose.
        orphan = sorted(set(named) - claimed)
        if orphan:
            raise ValueError(f"parameters {orphan} are learnable but in no adamw group")
        self.opt = torch.optim.AdamW(groups, betas=self.betas)
        self._lr0 = [g["lr"] for g in self.opt.param_groups]
        return self.opt

    def set_lr(self, it):
        for g, lr0 in zip(self.opt.param_groups, self._lr0):
            f = 1.0
            if self.warmup and it < self.warmup:
                f = (it + 1) / self.warmup
            elif self.scheduler == "cosine_to_zero":
                p = (it - self.warmup) / max(1, self.n_iter - self.warmup)
                f = 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
            g["lr"] = lr0 * f

    def step(self, total, named):
        self.opt.zero_grad(set_to_none=True)
        total.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(named.values()), self.grad_clip)
        self.opt.step()

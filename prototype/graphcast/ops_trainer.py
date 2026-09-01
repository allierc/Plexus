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
    predict       state, t -> state at t+1 (or its increment)  known_ode (via ops_known_ode), gnn
    loss          prediction, target -> scalar                 mse
    regularize    parameters -> scalar                         l1, l2
    step          scalars, parameters -> updated parameters    adamw

`graph` is the fifth role in the design and is deliberately NOT here: building a kNN graph over
positions changes the edge set, which is ALREADY a Plexus kind (`rewire`), so it is a plain
registered operator that can also sit in a forward schedule. Reusing that is the point; inventing a
trainer-only duplicate of it would not be.
"""

from __future__ import annotations

import math

import torch

ROLES = ("graph", "predict", "loss", "regularize", "step")

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


# ------------------------------------------------------------------ predict ---------------------
@register_trainer_op("plexus_model")
class PlexusModelPredict(TrainerOp):
    """ROLE predict. THE MODEL IS A PLEXUS HIERARCHY AND PREDICTION IS RUNNING ITS SCHEDULE.

    This is the one trainer operator that had to be got right, because it is where the framework's
    claim is either honoured or quietly dropped. The alternative -- a `nn.Module` with a `forward`
    -- would keep the arithmetic and throw away all four of plexus2.tex's compositions: the fields
    the model is about, the several mechanisms that may act on one of them, the levels they reach
    each other through, and the SCHEDULE that says in what order and HOW OFTEN each runs.

    So `model:` in the yaml is a Plexus spec fragment, `ModelHierarchy` loads it through
    `plexus.schema` and builds it through `plexus.engine`, and one prediction is:

        load the observed frame into the hierarchy  ->  step its schedule K times  ->  read it back

    Two consequences that a `forward()` does not have. The learnable parameters live on the
    OPERATORS, so a residual is attributable to a named mechanism rather than to a model. And the
    schedule carries `every:`, so a level whose motion is slower than the observation cadence is
    integrated on ITS OWN CLOCK -- which is the whole reason the 64^2 coarse dataset exists.

    `implementation:` selects which operators the model spec names -- `known_ode` for the true
    equation with its constants learnable, `gnn` for a general learnable message-passing rule.
    Nothing here branches on it: it is a different list of operators in the file, not a code path.
    """

    ROLE = "predict"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
        self.model = ctx["model"]                    # a ModelHierarchy, built by the trainer engine
        self.field = params.get("field", ctx["field"])
        self.steps = int(params.get("steps", 1))
        self.target = params.get("target", "increment")
        if self.target not in ("increment", "state"):
            raise ValueError(f"plexus_model target {self.target!r} is not increment|state")

    def parameters(self):
        return self.model.named_parameters()

    def forward(self, x):
        """`x` is [B, *res]. Returns the prediction in the same shape as the target.

        ONE HIERARCHY, ONE SAMPLE AT A TIME. A Plexus hierarchy holds one system, not a batch of
        them, so a batch is a loop rather than a leading tensor dimension. That is slower and it is
        honest: pretending a hierarchy is batched is how a model stops being the thing it claims to
        simulate. For the field sizes here the loop is not the cost.
        """
        out = []
        for b in range(x.shape[0]):
            self.model.load(self.field, x[b])
            before = x[b]
            self.model.step(self.steps)
            after = self.model.read(self.field)
            out.append(after - before if self.target == "increment" else after)
        return torch.stack(out)


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
@register_trainer_op("adamw")
class AdamWStep(TrainerOp):
    """ROLE step. AdamW with PARAMETER GROUPS, a schedule, and global gradient clipping.

    The groups are the point. Production configs carry three learning rates -- `lr_W 0.0009`,
    `lr 0.0018`, `lr_embedding 0.002325` -- as three scalars whose relationship to the parameters
    is implicit in the trainer's source. Here they are what they always were: groups, written down.

    Defaults follow GraphCast where they are load-bearing and measured (supplement 4.4): beta2 =
    0.95 rather than 0.999, global grad-norm clip at 32, and cosine decay TO ZERO -- the weekend
    benchmark's one-checkpoint collapses of 0.2-0.5 in R^2_W are the symptom of never annealing.
    """

    ROLE = "step"

    def __init__(self, params, ctx):
        super().__init__(params, ctx)
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

"""The model under fit, as a PLEXUS HIERARCHY -- not as a wrapper around one.

THE DESIGN POINT, and it is the whole reason this file is separate from `trainer.py`. plexus2.tex
builds a model by four compositions: entities into structures, mechanisms within a level, levels
through aggregate/broadcast, and *mechanisms in time* -- the schedule, which "specifies both the
order and the frequency with which operators are evaluated", making a model "a multi-rate
composition of biological mechanisms rather than a monolithic numerical solver".

A fitting model that is a `nn.Module` holding a `forward()` throws all four away and keeps only a
function. So the model here is built the same way the generator is:

    fields + operators + schedule  ->  plexus.schema.load  ->  plexus.engine.build/seed  ->  H

and PREDICTION IS RUNNING ITS SCHEDULE. The learnable parameters live on the operators, exactly
where the mechanism they belong to lives, which is what makes a residual attributable to a
mechanism rather than to a model (plexus2.tex, mechanistic inverse modelling).

WHAT `step()` DOES AND WHY IT IS NOT A FORK OF THE ENGINE. `plexus.engine.run` owns its tick loop
inline; there is no public one-tick entry point. What that loop does per tick is: set `H.frame`,
zero the deltas, call `op.forward(H)` for each schedule entry in order, then integrate the
accumulated deltas. For a FIELD-ONLY model -- every operator `kind="field"`, writing `fld.grid`
directly and returning `{}` -- the delta stage is a no-op, so the tick reduces to exactly the same
sequence of `op.forward(H)` calls, which is what `step()` makes. It is the engine's call, not a
reimplementation of the engine's arithmetic.

    SUPPORTED   field-only models (the toy's transport and Kuramoto rules, and their known-ODE
                twins), with per-operator `every:` for multi-rate schedules.
    DEFERRED    any model carrying a SET -- deltas, `EMIT`, aggregate/broadcast between levels.
                Those must go through `plexus.engine.run(..., grad=True)`, which keeps the tape
                across a whole rollout. `step()` raises rather than silently dropping a delta.

MULTI-RATE IS THE POINT, NOT AN EXTRA. The 64^2 coarse dataset is bit-identical between 71.5% of
consecutive records: its motion is below the observation cadence, and there is no timestep at which
it is a smooth transport. A schedule entry `{op: ..., every: 4}` lets that level advance on its own
clock while a fine level advances on every tick -- which is precisely the composition the paper
describes and precisely what a single-rate fit cannot express.
"""

from __future__ import annotations

import os

import torch
import yaml

import ops_graphcast   # noqa: F401  registers the generator's rules
import ops_known_ode   # noqa: F401  registers their known-ODE twins
from plexus import engine as plexus_engine
from plexus import schema as plexus_schema


class ModelHierarchy:
    """A Plexus hierarchy whose operators carry the learnable parameters."""

    def __init__(self, model_cfg: dict, general: dict, out_dir: str, device: str = "cpu"):
        spec_raw = {
            "general": {**general, **(model_cfg.get("general") or {})},
            "sets": model_cfg.get("sets") or {},
            "fields": model_cfg["fields"],
            "operators": model_cfg["operators"],
            "schedule": [s["op"] if isinstance(s, dict) else s for s in model_cfg["schedule"]],
        }
        path = os.path.join(out_dir, "_model_spec.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(spec_raw, f, sort_keys=False)
        # LOADED THROUGH `plexus.schema`, so the model spec is validated by the same code that
        # validates a forward spec. A model that would not run as a simulation is not a model.
        self.spec = plexus_schema.load(path)
        self.H = plexus_engine.build(self.spec, device)
        plexus_engine.seed(self.H, self.spec, device)

        # The operator INSTANCES, in schedule order, each with its own rate.
        from plexus.models.registry import get_operator
        by_name = {o["op"]: o for o in model_cfg["operators"]}
        self.ops, self.every, self.names = [], [], []
        for entry in model_cfg["schedule"]:
            entry = {"op": entry} if isinstance(entry, str) else dict(entry)
            line = by_name[entry["op"]]
            cls = get_operator(line["op"], variant=line.get("model"))
            if getattr(cls, "KIND", None) != "field":
                raise NotImplementedError(
                    f"{line['op']} has kind={cls.KIND!r}; ModelHierarchy steps FIELD operators "
                    f"only, because for those the engine's delta stage is a no-op and a tick is "
                    f"exactly the sequence of forward() calls made here. A set-carrying model must "
                    f"go through plexus.engine.run(..., grad=True).")
            self.ops.append(cls({**line, "_at": line.get("at"), "dim": self.spec.dim},
                                device=device))
            self.every.append(int(entry.get("every", 1)))
            self.names.append(line["op"])
        self.tick = 0

    def named_parameters(self):
        out = {}
        for name, op in zip(self.names, self.ops):
            for k, p in op.named_parameters():
                out[f"{name}.{k}"] = p
        return out

    def load(self, field: str, value: torch.Tensor, channel: int = 0):
        """Put an observed frame into the hierarchy. The fit's initial condition is DATA.

        THE GRID IS REPLACED BY A DETACHED CLONE, NOT WRITTEN INTO, and the reason is a bug this
        cost. A hierarchy is persistent: the same `fld.grid` tensor survives from one training
        iteration to the next, and after a backward pass it still carries that iteration's autograd
        history. Assigning into it in place leaves the old graph attached to the new sample, so the
        second iteration walks a graph whose buffers were freed by the first --

            RuntimeError: Trying to backward through the graph a second time

        -- and the tempting fix, `retain_graph=True`, would have silenced it while quietly
        accumulating every past iteration into one ever-growing graph.

        The right statement is the modelling one: THE OBSERVATION IS DATA. It enters the fit as a
        LEAF, carrying no history, and the only tensors that carry gradient are the operators' own
        parameters -- which is exactly the property that makes a residual attributable to a
        mechanism instead of to the trainer's bookkeeping.
        """
        fld = self.H.fields[field]
        fld.grid = fld.grid.detach().clone()
        fld.grid[channel] = value.detach()

    def read(self, field: str, channel: int = 0) -> torch.Tensor:
        return self.H.fields[field].grid[channel]

    def step(self, n: int = 1):
        """`n` ticks of the model's own schedule, honouring each operator's `every:`."""
        for _ in range(n):
            self.H.frame = self.tick
            if hasattr(self.H, "frame_t"):
                self.H.frame_t.fill_(float(self.tick))
            for op, every in zip(self.ops, self.every):
                if self.tick % every == 0:
                    op.forward(self.H)
            self.tick += 1

    def describe(self) -> str:
        return "  ".join(f"{n}{'' if e == 1 else f' every {e}'}"
                         for n, e in zip(self.names, self.every))

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

import ops_toy        # noqa: F401  registers the generator's rules
import ops_known_ode   # noqa: F401  registers their known-ODE twins
from plexus import engine as plexus_engine
from plexus import schema as plexus_schema


class ModelHierarchy:
    """A Plexus hierarchy whose operators carry the learnable parameters."""

    def __init__(self, spec_path: str, device: str = "cpu"):
        """`spec_path` is the FIT SPEC ITSELF. It is a Plexus spec and is loaded as one.

        There is no separate `model:` block and no temporary yaml. A fit spec declares the same
        `general` / `sets` / `fields` / `operators` / `schedule` its simulation twin does -- the
        only differences are that its operators are the LEARNABLE twins (`kuramoto_known_ode` for
        `kuramoto_field`), that they carry a `learn:` list, and that a `fit:` section is appended.
        `plexus.schema.load` ignores unknown top-level keys, so `fit:` costs nothing, and unknown
        operator params are carried through, so `learn:` reaches the operator.

        That is the whole point of the arrangement: a reader who knows the simulation spec can read
        the fit spec, and the two can be diffed. An earlier version nested a `model:` dict inside a
        `trainer:` block and reached it through an operator called `plexus_model` -- an indirection
        that named the framework instead of the mechanism and told a reader nothing.
        """
        self.spec = plexus_schema.load(spec_path)
        self.H = plexus_engine.build(self.spec, device)
        plexus_engine.seed(self.H, self.spec, device)

        from plexus.models.registry import get_operator
        with open(spec_path) as f:
            raw = yaml.safe_load(f)
        by_name = {o["op"]: o for o in raw["operators"]}
        self.ops, self.every, self.names, self.learn = [], [], [], {}
        for entry in raw["schedule"]:
            entry = {"op": entry} if isinstance(entry, str) else dict(entry)
            line = by_name[entry["op"]]
            cls = get_operator(line["op"], variant=line.get("model"))
            self.ops.append(cls({**line, "_at": line.get("at"), "dim": self.spec.dim},
                                device=device))
            self.every.append(int(entry.get("every", 1)))
            self.names.append(line["op"])
            self.learn[line["op"]] = list(line.get("learn", []))
            self._at = getattr(self, "_at", {})
            self._at[line["op"]] = line.get("at", "")
        self.tick = 0

    def at_of(self, op_name: str) -> str:
        """Which field an operator acts on, so a caller can give it that field's mask."""
        return self._at.get(op_name, "")

    def bind_shapes(self, masks: dict | None = None):
        """Let operators whose PARAMETER SHAPE is the field's allocate it, now that the field exists.

        `KuramotoKnownODE.omega` is one natural frequency per pixel, so its shape is not knowable
        when the operator is constructed from a spec line -- only once the hierarchy has a grid. An
        operator that needs this declares a `bind(shape, mask)`; one that does not is untouched, so
        this is a capability check rather than a special case for a named operator.

        THE MASK IS GIVEN, NOT FITTED. `m_i` says where the fine rule acts. Learning the support as
        well is a different experiment -- *where does the fast mechanism live* -- and confounding it
        into the recovery of `K` and `omega` would make a failure of either unattributable.
        """
        for name, op in zip(self.names, self.ops):
            if not hasattr(op, "bind"):
                continue
            fld = self.H.fields[getattr(op, "field_name", None) or self.spec.fields[0]]
            shape = tuple(fld.grid.shape[1:])
            m = (masks or {}).get(name)
            op.bind(shape, torch.ones(shape, device=fld.grid.device) if m is None else m)
        return self

    def named_parameters(self):
        """`<operator>.<param>` for every parameter the spec's `learn:` lists.

        ONLY WHAT `learn:` NAMES. A constant an operator owns but the spec does not list stays
        frozen at its init, so "which constants are unknown" is a statement in the file rather than
        a property of a class -- and the same operator can be an oracle in one spec and the thing
        under test in another.
        """
        out = {}
        for name, op in zip(self.names, self.ops):
            want = self.learn.get(name, [])
            for k, p in op.named_parameters():
                p.requires_grad_(k in want)
                if k in want:
                    out[f"{name}.{k}"] = p
        missing = [f"{n}.{k}" for n in self.learn for k in self.learn[n]
                   if f"{n}.{k}" not in out]
        if missing:
            raise ValueError(f"spec asks to learn {missing}, but no operator exposes them; "
                             f"available: {sorted(out)}")
        return out

    def load(self, field: str, value: torch.Tensor):
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
        # THE WHOLE GRID, EVERY CHANNEL. A phase oscillator's faithful state is the pair
        # (sin phi, cos phi): sin alone is many-to-one, so no rule written in phi can be fitted to
        # it. Loading channel 0 only would have left cos at whatever the seed put there.
        fld.grid = value.detach().clone()

    def read(self, field: str) -> torch.Tensor:
        return self.H.fields[field].grid

    def step(self, n: int = 1):
        """`n` ticks of the model's own schedule, honouring each operator's `every:`.

        ONE TICK IS WHAT THE ENGINE'S TICK IS: zero the deltas, run the schedule in order, let each
        operator either mutate a field in place (`field`, `rewire`) or return a per-set delta
        (`lateral`, `exchange`, `aggregate`, `broadcast`), then integrate. The integration is
        `plexus.engine._integrate`, CALLED rather than reimplemented -- it resolves the order from
        `H.emit_order`, reads each set's `StateSchema` to find which block is the coordinate, and
        applies the boundary. Reproducing that here would be a second definition of what a state
        is, and the two would drift.

        An earlier version refused anything but `kind="field"` on the grounds that the delta stage
        was a no-op for those. That was true and it was also the thing blocking the canonical
        Plexus composition -- `radius_graph` (rewire) followed by a message-passing `Lateral` over
        the edges it builds, which is how every interaction model in the codebase is written.
        """
        for _ in range(n):
            self.H.frame = self.tick
            if hasattr(self.H, "frame_t"):
                self.H.frame_t.fill_(float(self.tick))
            if self.H.levels:        # a field-only model has no set, hence no accumulator
                self.H.zero_delta()
            for op, every in zip(self.ops, self.every):
                if self.tick % every:
                    continue
                out = op.forward(self.H)
                for set_name, delta in (out or {}).items():
                    self.H.add_delta(set_name, delta)
            if getattr(self.H, "emit_order", None):
                plexus_engine._integrate(self.H, self.spec.dt)
            self.tick += 1

    def describe(self) -> str:
        return "  ".join(f"{n}{'' if e == 1 else f' every {e}'}"
                         for n, e in zip(self.names, self.every))

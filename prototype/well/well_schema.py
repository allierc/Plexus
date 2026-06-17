"""well_schema: typed, validated spec loader for The Well PDE scenarios.

Same contract philosophy as the water prototype's scenario_schema.py: a YAML spec
is the verifiable boundary between intent and code. It declares *fields* (continua)
and/or *sets* (particle levels), the registered *operators* that act on them, and a
*schedule*. Loading validates against the operator registry and the per-operator
capability contract (REQUIRES_PARAMS) and fails loudly before any run.

A spec that loads here is guaranteed runnable by well_engine.run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml

import well_fields   # noqa: F401  registers MultiField ("grid_nd")
import well_ops      # noqa: F401  registers the operators
from plexus.models.registry import get_operator


@dataclass
class OpSpec:
    op: str
    on: str                         # set or field name the operator acts on
    params: dict = field(default_factory=dict)


@dataclass
class WellScenario:
    name: str
    seed: int
    n_frames: int
    record_every: int
    world: float
    fields: dict
    sets: dict
    operators: list
    schedule: list
    init: dict                      # initial-condition directives, per field/set


_RESERVED = {"op", "at"}


def load(path: str) -> WellScenario:
    with open(path) as f:
        raw = yaml.safe_load(f)
    for key in ("name", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"scenario missing required key: {key!r}")

    fields = raw.get("fields", {})
    sets = raw.get("sets", {})

    ops = []
    for o in raw["operators"]:
        name = o["op"]
        try:
            cls = get_operator(name)
        except KeyError:
            from plexus.models.registry import _OPERATOR_REGISTRY
            raise ValueError(f"operator {name!r} not registered. "
                             f"Available: {sorted(_OPERATOR_REGISTRY)}")
        at = o["at"]
        if at not in fields and at not in sets:
            raise ValueError(f"operator {name!r} acts on unknown set/field {at!r}")
        params = {k: v for k, v in o.items() if k not in _RESERVED}
        for req in getattr(cls, "REQUIRES_PARAMS", []):
            if req not in params:
                raise ValueError(
                    f"operator {name!r} requires param {req!r} "
                    f"(declared in {cls.__name__}.REQUIRES_PARAMS); add it to the line.")
        ops.append(OpSpec(op=name, on=at, params=params))

    op_names = {o.op for o in ops}
    for step in raw["schedule"]:
        for tok in (step if isinstance(step, list) else [step]):
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator")

    return WellScenario(
        name=raw["name"],
        seed=int(raw.get("seed", 0)),
        n_frames=int(raw.get("n_frames", 200)),
        record_every=int(raw.get("record_every", 2)),
        world=float(raw.get("world", 1.0)),
        fields=fields,
        sets=sets,
        operators=ops,
        schedule=raw["schedule"],
        init=raw.get("init", {}),
    )

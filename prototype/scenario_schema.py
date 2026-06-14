"""Scenario spec: load + VALIDATE.  (the 'verifiable' entry point)

The spec is the contract between intent and code. This module parses it into
typed objects and fails loudly with a precise message if anything is off:
unknown operator, missing field, malformed selector, fractions that don't sum to
one. A spec that loads here is guaranteed to be runnable by the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

# importing tissue_graph populates base+registry but NOT the stub catalog
from tissue_graph.models import registry  # noqa: F401

_SELECTOR_RE = re.compile(r"^(?P<set>\w+)(?:\[(?P<attr>\w+)=(?P<val>\w+)\])?$")


@dataclass
class Selector:
    set: str
    attr: Optional[str] = None
    val: Optional[str] = None

    @classmethod
    def parse(cls, s: str) -> "Selector":
        m = _SELECTOR_RE.match(str(s).strip())
        if not m:
            raise ValueError(f"bad selector {s!r} (expected 'set' or 'set[attr=val]')")
        return cls(m["set"], m["attr"], m["val"])


@dataclass
class OpSpec:
    op: str
    on: Selector
    to: Optional[str] = None       # target field (Exchange scatter)
    frm: Optional[str] = None      # source field (Exchange gather)
    params: dict = field(default_factory=dict)


@dataclass
class Scenario:
    name: str
    seed: int
    n_frames: int
    dt: float
    record_every: int
    sets: dict
    fields: dict
    operators: list[OpSpec]
    schedule: list


_RESERVED = {"op", "at", "to", "from"}
_BUILTIN_STEPS = {"integrate", "aggregate"}    # plus '<field>.diffuse'


def load(path: str) -> Scenario:
    with open(path) as f:
        raw = yaml.safe_load(f)

    for key in ("name", "sets", "fields", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"scenario missing required key: {key!r}")

    # --- sets: type fractions must sum to 1 ---
    for sname, s in raw["sets"].items():
        types = s.get("types", {})
        if types:
            total = sum(t["fraction"] for t in types.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"set {sname!r} type fractions sum to {total}, must be 1.0")

    # --- operators: names registered, selectors valid, fields exist ---
    ops = []
    for o in raw["operators"]:
        name = o["op"]
        try:
            registry.get_operator(name)
        except KeyError:
            raise ValueError(
                f"operator {name!r} not in registry. Available: "
                f"{sorted(registry._OPERATOR_REGISTRY)}"
            )
        sel = Selector.parse(o["at"])
        if sel.set not in raw["sets"]:
            raise ValueError(f"operator {name!r} acts on unknown set {sel.set!r}")
        for fref in (o.get("to"), o.get("from")):
            if fref is not None and fref not in raw["fields"]:
                raise ValueError(f"operator {name!r} references unknown field {fref!r}")
        params = {k: v for k, v in o.items() if k not in _RESERVED}
        ops.append(OpSpec(op=name, on=sel, to=o.get("to"), frm=o.get("from"), params=params))

    # --- schedule: every token resolves ---
    op_names = {o.op for o in ops}
    for step in raw["schedule"]:
        tokens = step if isinstance(step, list) else [step]
        for tok in tokens:
            if tok in _BUILTIN_STEPS:
                continue
            if tok.endswith(".diffuse"):
                fld = tok[: -len(".diffuse")]
                if fld not in raw["fields"]:
                    raise ValueError(f"schedule step {tok!r} references unknown field")
                continue
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator or builtin")

    return Scenario(
        name=raw["name"],
        seed=int(raw.get("seed", 0)),
        n_frames=int(raw.get("n_frames", 200)),
        dt=float(raw.get("dt", 0.05)),
        record_every=int(raw.get("record_every", 2)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
    )

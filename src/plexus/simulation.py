"""The simulation spec: load + VALIDATE -- the contract gatekeeper.

A complete Plexus run is a single declarative `spec.yaml`: `sets`, `fields`,
`operators` (+ selectors), and a `schedule`. This module parses it into typed
objects and fails loudly with a precise message if anything is off -- an
unregistered operator, an unknown set/field reference, a malformed selector, type
fractions that do not sum to one, a required property missing along the
containment chain, or a schedule token that does not resolve. The guarantee:

    a spec that loads here is runnable by the engine.

It imports the registry (to resolve operator names + their capability contracts)
but NEVER the engine -- pure validation, no execution. The operator modules must
already be imported by the caller so the registry is populated.

(Promoted from the prototype's `scenario_schema.py`; renamed `Scenario` ->
`Simulation` to match the build plan, which uses "simulation" throughout. Two
additions over the prototype: the `buffer` set-spec field -- the occupancy
allocation a cardinality-changing run needs -- and a defensive check that every
operator carries a recognised `KIND`.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

# importing plexus populates base + registry; operator modules register themselves
from plexus.models import registry
from plexus.models.base import KINDS

_SELECTOR_RE = re.compile(r"^(?P<set>\w+)(?:\[(?P<attr>\w+)=(?P<val>\w+)\])?$")


@dataclass
class Selector:
    """`set` (every node) or `set[attr=val]` (a subset, re-checked each tick)."""
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
    to: Optional[str] = None        # target field (Exchange scatter)
    frm: Optional[str] = None       # source field (Exchange gather)
    params: dict = field(default_factory=dict)


@dataclass
class Simulation:
    name: str
    seed: int
    n_frames: int
    dt: float
    record_every: int
    sets: dict
    fields: dict
    operators: list[OpSpec]
    schedule: list
    obstacles: list = field(default_factory=list)   # wall rectangles [x0,y0,x1,y1] or discs [cx,cy,r]
    boundary: str = "wall"                           # 'wall' (clamp) or 'periodic' (torus)
    world: float = 1.0                               # domain width; world is [0,world]x[0,1]


_RESERVED = {"op", "at", "to", "from"}
_BUILTIN_STEPS = {"integrate", "aggregate"}         # plus '<field>.diffuse'


def load(path: str) -> Simulation:
    with open(path) as f:
        raw = yaml.safe_load(f)

    for key in ("name", "sets", "fields", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"simulation missing required key: {key!r}")

    # --- sets: type fractions sum to 1; buffer (if given) >= n -------------- #
    for sname, s in raw["sets"].items():
        types = s.get("types", {})
        if types:
            total = sum(t["fraction"] for t in types.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"set {sname!r} type fractions sum to {total}, must be 1.0")
        # `buffer` is the allocated slot count for a cardinality-changing set
        # (occupancy `occ` marks the live subset). It must hold the initial set.
        buf = s.get("buffer")
        if buf is not None:
            live0 = s.get("n", s.get("per_parent"))
            if live0 is not None and int(buf) < int(live0):
                raise ValueError(
                    f"set {sname!r} buffer={buf} is smaller than its initial size {live0}; "
                    f"a structural (divide/duplicate) run needs buffer >= initial.")

    # --- operators: names registered, valid KIND, selectors + fields exist -- #
    ops = []
    for o in raw["operators"]:
        name = o["op"]
        try:
            cls = registry.get_operator(name)
        except KeyError:
            raise ValueError(
                f"operator {name!r} not in registry. Available: "
                f"{sorted(registry._OPERATOR_REGISTRY)}"
            )
        kind = getattr(cls, "KIND", None)
        if kind not in KINDS:
            raise ValueError(
                f"operator {name!r} has unrecognised kind {kind!r}; expected one of {KINDS}.")
        sel = Selector.parse(o["at"])
        if sel.set not in raw["sets"]:
            raise ValueError(f"operator {name!r} acts on unknown set {sel.set!r}")
        for fref in (o.get("to"), o.get("from")):
            if fref is not None and fref not in raw["fields"]:
                raise ValueError(f"operator {name!r} references unknown field {fref!r}")
        params = {k: v for k, v in o.items() if k not in _RESERVED}

        # --- capability contract: operator declares what it requires --------- #
        have = set(o.keys())
        for req in getattr(cls, "REQUIRES_PARAMS", []):
            if req not in have:
                raise ValueError(
                    f"operator {name!r} requires param {req!r} (declared in "
                    f"{cls.__name__}.REQUIRES_PARAMS); add it to the operator line.")
        # a required property may live on the set's types or be inherited from a
        # parent set (mpm acts on particles but reads `youngs` off the cell type)
        def types_in_chain(set_name):
            seen = set()
            while set_name and set_name not in seen:
                seen.add(set_name)
                ts = raw["sets"][set_name].get("types")
                if ts:
                    return set_name, ts
                set_name = raw["sets"][set_name].get("parent")
            return None, {}
        for prop in getattr(cls, "REQUIRES_TYPE_PROPS", []):
            owner, set_types = types_in_chain(sel.set)
            if not set_types:
                raise ValueError(
                    f"operator {name!r} requires per-type property {prop!r}, but neither "
                    f"{sel.set!r} nor its parents declare `types`.")
            for tname, t in set_types.items():
                if prop not in t:
                    raise ValueError(
                        f"operator {name!r} requires property {prop!r} on every type of "
                        f"{owner!r}; missing on type {tname!r}. "
                        f"(declared in {cls.__name__}.REQUIRES_TYPE_PROPS)")
        ops.append(OpSpec(op=name, on=sel, to=o.get("to"), frm=o.get("from"), params=params))

    # --- warn about per-type properties no operator reads (typo guard) ------ #
    used_props = set()
    for o in raw["operators"]:
        used_props |= set(getattr(registry.get_operator(o["op"]), "REQUIRES_TYPE_PROPS", []))
    _KNOWN_TYPE_KEYS = {"fraction", "core", "layers"} | used_props  # core/layers: per-particle stiffness (engine build)
    for sname, s in raw["sets"].items():
        for tname, t in s.get("types", {}).items():
            for k in t:
                if k not in _KNOWN_TYPE_KEYS:
                    print(f"[warn] property {k!r} on {sname}.{tname} is read by no operator "
                          f"(known: {sorted(_KNOWN_TYPE_KEYS)})")

    # --- schedule: every token resolves to an operator or a builtin --------- #
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

    return Simulation(
        name=raw["name"],
        seed=int(raw.get("seed", 0)),
        n_frames=int(raw.get("n_frames", 200)),
        dt=float(raw.get("dt", 0.05)),
        record_every=int(raw.get("record_every", 2)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
        obstacles=raw.get("obstacles", []),
        boundary=raw.get("boundary", "wall"),
        world=float(raw.get("world", 1.0)),
    )

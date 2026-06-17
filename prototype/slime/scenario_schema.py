"""Slime spec: load + VALIDATE, conforming to the modified plexus.tex spec language
(Part II, "Categorical discipline").

The spec has FIVE orthogonal categories, each with its own home (Table tab:cat):

  general:    World + clock -- name, seed, n_frames, dt, boundary, world, obstacles.
  sets:       Entities -- what a node IS: count, composition (types + per-type law
              params), initial state (spawn). NEVER style, NEVER a rate.
  fields:     continua (PDE coeffs live here).
  operators:  Process -- how state changes: the law and its parameters.
  schedule:   composition / timing.
  plotting:   Style ONLY -- colours, background, gamma. Read by the visualizer,
              never by the dynamics.

This validator enforces the discipline: a quantity filed under the wrong category
is rejected up front. In particular `color` on a type is a CATEGORY ERROR (colour
is Style -> plotting.colors), not a property of what a node is.

A spec that loads here is guaranteed runnable. Back-compatible: a flat top-level
(name/seed/... at the root, no `general:` block) is still accepted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from plexus.models import registry  # noqa: F401  populate base+registry

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
    to: Optional[str] = None
    frm: Optional[str] = None
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
    operators: list
    schedule: list
    obstacles: list = field(default_factory=list)
    boundary: str = "wall"
    world: float = 1.0
    plotting: dict = field(default_factory=dict)     # Style category (read by viz only)


_RESERVED = {"op", "at", "to", "from"}
_BUILTIN_STEPS = {"integrate", "aggregate"}
# keys that legitimately live in the `general:` (World+clock) category
_GENERAL_KEYS = {"name", "seed", "n_frames", "dt", "record_every",
                 "boundary", "world", "obstacles"}


def load(path: str) -> Scenario:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # --- general: block (World + clock). Back-compat: fall back to top-level. ---
    gen = dict(raw.get("general", {}))
    def G(key, default=None):
        return gen.get(key, raw.get(key, default))

    for key in ("sets", "fields", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"scenario missing required key: {key!r}")
    if G("name") is None:
        raise ValueError("scenario missing required key: 'name' (in general: or top level)")

    plotting = dict(raw.get("plotting", {}))

    # --- sets: type fractions sum to 1; colour on a type is a CATEGORY ERROR ---
    for sname, s in raw["sets"].items():
        types = s.get("types", {})
        if types:
            total = sum(t["fraction"] for t in types.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"set {sname!r} type fractions sum to {total}, must be 1.0")
            for tname, t in types.items():
                if "color" in t or "colour" in t:
                    raise ValueError(
                        f"category error: 'color' on type {sname}.{tname} is Style, not a "
                        f"property of what a node is. Move it to plotting.colors.{tname} "
                        f"(plexus.tex Part II, Categorical discipline).")

    # --- operators: names registered, selectors valid, fields exist, caps declared ---
    ops = []
    for o in raw["operators"]:
        name = o["op"]
        try:
            registry.get_operator(name)
        except KeyError:
            raise ValueError(
                f"operator {name!r} not in registry. Available: "
                f"{sorted(registry._OPERATOR_REGISTRY)}")
        sel = Selector.parse(o["at"])
        if sel.set not in raw["sets"]:
            raise ValueError(f"operator {name!r} acts on unknown set {sel.set!r}")
        for fref in (o.get("to"), o.get("from")):
            if fref is not None and fref not in raw["fields"]:
                raise ValueError(f"operator {name!r} references unknown field {fref!r}")
        params = {k: v for k, v in o.items() if k not in _RESERVED}
        cls = registry.get_operator(name)
        have = set(o.keys())
        for req in getattr(cls, "REQUIRES_PARAMS", []):
            if req not in have:
                raise ValueError(
                    f"operator {name!r} requires param {req!r} (declared in "
                    f"{cls.__name__}.REQUIRES_PARAMS); add it to the operator line.")

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

    # --- warn about per-type properties no operator reads (typo / category tripwire) ---
    used_props = set()
    for o in raw["operators"]:
        used_props |= set(getattr(registry.get_operator(o["op"]), "REQUIRES_TYPE_PROPS", []))
    _KNOWN_TYPE_KEYS = {"fraction", "core", "layers", "block"} | used_props
    for sname, s in raw["sets"].items():
        for tname, t in s.get("types", {}).items():
            for k in t:
                if k not in _KNOWN_TYPE_KEYS:
                    print(f"[warn] property {k!r} on {sname}.{tname} is read by no operator "
                          f"(known: {sorted(_KNOWN_TYPE_KEYS)})")

    # --- schedule: every token resolves ---
    op_names = {o.op for o in ops}
    for step in raw["schedule"]:
        for tok in (step if isinstance(step, list) else [step]):
            if tok in _BUILTIN_STEPS:
                continue
            if tok.endswith(".diffuse"):
                if tok[:-len(".diffuse")] not in raw["fields"]:
                    raise ValueError(f"schedule step {tok!r} references unknown field")
                continue
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator or builtin")

    return Scenario(
        name=G("name"),
        seed=int(G("seed", 0)),
        n_frames=int(G("n_frames", 200)),
        dt=float(G("dt", 0.05)),
        record_every=int(G("record_every", 2)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
        obstacles=G("obstacles", []) or [],
        boundary=G("boundary", "wall"),
        world=float(G("world", 1.0)),
        plotting=plotting,
    )

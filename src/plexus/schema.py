"""The spec schema: load + VALIDATE -- the contract gatekeeper.

A complete Plexus run is a single declarative `spec.yaml` (`general`, `sets`,
`fields`, `operators` (+ selectors), `schedule`, `plotting`). This module is the
schema: it parses the file into typed objects (`Spec`, `OpSpec`, `Selector`) and
fails loudly with a precise message if anything is off -- an unregistered
operator, an unknown set/field reference, a malformed selector, type fractions
that do not sum to one, a required property missing along the containment chain,
or a schedule token that does not resolve. The guarantee:

    a spec that loads here is runnable by the engine.

It defines and validates the spec; the engine *runs* a `Spec`. This module
imports the registry (to resolve operator names + their capability contracts) but
NEVER the engine -- pure validation, no execution. The operator modules must
already be imported by the caller so the registry is populated.
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
class Spec:
    name: str
    seed: int
    n_frames: int
    dt: float
    sets: dict
    fields: dict
    operators: list[OpSpec]
    schedule: list
    obstacles: list = field(default_factory=list)   # wall rectangles [x0,y0,x1,y1] or discs [cx,cy,r]
    boundary: str = "wall"                           # 'wall' (clamp) or 'periodic' (torus)
    world: float = 1.0                               # domain width (= world_size[0]); legacy 2D scalar
    dim: int = 2                                     # spatial dimensions (the global dimension contract)
    world_size: list = field(default_factory=lambda: [1.0, 1.0])   # per-axis box [w0 .. w_{D-1}]
    plotting: dict = field(default_factory=dict)     # render STYLE (colormap, point_size, ...) — read by plexus.plot


_RESERVED = {"op", "at", "to", "from"}
# No schedule builtins: `aggregate` and `diffuse` are ordinary registered operators
# now, and integration is implicit (end of tick). Every schedule token must resolve
# to a declared operator.
_BUILTIN_STEPS: set = set()


def load(path: str) -> Spec:
    with open(path) as f:
        raw = yaml.safe_load(f)

    # run/world scalars live under `general:` (name, seed, n_frames, dt, boundary,
    # world, obstacles); fall back to the top level for back-compat.
    g = raw.get("general", {})
    def gv(key, default=None):
        return g.get(key, raw.get(key, default))

    for key in ("sets", "fields", "operators", "schedule"):
        if key not in raw:
            raise ValueError(f"simulation missing required key: {key!r}")
    if gv("name") is None:
        raise ValueError("simulation missing required key: 'name' (under general:)")

    # --- the dimension contract: dim (default 2) + per-axis world box -------- #
    dim = int(gv("dim", 2))
    world_raw = gv("world", 1.0)
    if isinstance(world_raw, (list, tuple)):
        world_size = [float(x) for x in world_raw]
    else:
        world_size = [float(world_raw)] + [1.0] * (dim - 1)   # scalar: axis-0 = width, rest = 1
    if len(world_size) != dim:
        raise ValueError(f"general.world has {len(world_size)} entries but dim={dim}; "
                         f"give a length-{dim} box e.g. world: {[1.0] * dim}")

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
        supported = getattr(cls, "SUPPORTED_DIMS", [2])
        if dim not in supported:
            raise ValueError(
                f"operator {name!r} supports dims {supported}, not dim={dim} "
                f"(set general.dim, or use a dimension-generic / *_3d operator).")
        sel = Selector.parse(o["at"])
        # `at:` names a SET (set/exchange operators) or a FIELD (field-internal
        # operators like diffuse/decay, which read & write the field at `at:`).
        if sel.set not in raw["sets"] and sel.set not in raw["fields"]:
            raise ValueError(f"operator {name!r} acts on unknown set or field {sel.set!r}")
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
        cls = registry.get_operator(o["op"])
        used_props |= set(getattr(cls, "REQUIRES_TYPE_PROPS", []))
        used_props |= set(getattr(cls, "OPTIONAL_TYPE_PROPS", []))   # read only in some modes (e.g. alignment per_type)
    # core/layers/block: consumed by an entity provision hook (e.g. mpm_particle), not by an operator
    _KNOWN_TYPE_KEYS = {"fraction", "core", "layers", "block"} | used_props
    for sname, s in raw["sets"].items():
        for tname, t in s.get("types", {}).items():
            for k in t:
                if k not in _KNOWN_TYPE_KEYS:
                    print(f"[warn] property {k!r} on {sname}.{tname} is read by no operator "
                          f"(known: {sorted(_KNOWN_TYPE_KEYS)})")

    # --- schedule: every token resolves to an operator or a builtin --------- #
    # A step may be a token, a list of tokens (run in sequence), or a substep micro-loop
    # `{substep_dt: <dt>, steps: [...]}` whose inner tokens run once per substep; the count
    # is round(general.dt / dt). (e.g. the MPM strain->P2G->grid->G2P cycle.)
    op_names = {o.op for o in ops}
    for step in raw["schedule"]:
        if isinstance(step, dict) and "substep" in step:
            raise ValueError("the `{substep: N, dt: X}` schedule form was removed; write "
                             "`{substep_dt: X, steps: [...]}` and set `general.dt` to the "
                             "per-frame sim time (substeps = round(general.dt / X)).")
        if isinstance(step, dict) and "substep_dt" in step:
            if not isinstance(step.get("steps"), list):
                raise ValueError("a `{substep_dt: …}` schedule step needs a `steps:` list")
            tokens = step["steps"]
        else:
            tokens = step if isinstance(step, list) else [step]
        for tok in tokens:
            if tok in _BUILTIN_STEPS:
                continue
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator or builtin")

    return Spec(
        name=gv("name"),
        seed=int(gv("seed", 0)),
        n_frames=int(gv("n_frames", 200)),
        dt=float(gv("dt", 0.05)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
        obstacles=gv("obstacles", []),
        boundary=gv("boundary", "wall"),
        world=world_size[0],
        dim=dim,
        world_size=world_size,
        plotting=raw.get("plotting", {}),
    )

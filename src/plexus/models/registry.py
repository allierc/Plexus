"""Registries for Plexus.

Three registries, all decorator-based (same pattern as cell-gnn's registry):

  - ENTITY   : node *kinds* (a `Level`'s state schema)  e.g. "cell", "particle", "molecule"
  - OPERATOR : the learnable dynamics (a GNN = ODE vector field + Laplacian)
  - FIELD    : continuum discretizations  e.g. "grid", "mesh", "implicit"

Operators and fields are tagged with the *level* they act on and the *kind* of
operator — one of the seven in `base.KINDS`: the four set-dynamics kinds that
return a delta (lateral / aggregate / broadcast / exchange), `field` (a field's
own self-dynamics: diffuse / decay / playback), `rewire` (changes the relation E),
and `structural` (changes the entity set |S|) — so a `Schedule`, or the LLM agentic
loop, can enumerate "everything that can run at level k".

Usage
-----
    @register_operator("signal", level="cell", kind="lateral")
    class SignalOperator(Operator):
        ...

    op_cls   = get_operator("signal")
    at_cell  = operators_at_level("cell")        # -> {name: cls, ...}
"""

from __future__ import annotations

_ENTITY_REGISTRY: dict[str, type] = {}
_OPERATOR_REGISTRY: dict[str, type] = {}
_FIELD_REGISTRY: dict[str, type] = {}


def _make_decorator(registry: dict, label: str, **tag_defaults):
    def register(*names: str, **tags):
        def decorator(cls):
            meta = {**tag_defaults, **tags}
            for k, v in meta.items():
                setattr(cls, k.upper(), v)
            cls.REGISTERED_NAMES = list(names)
            for name in names:
                if name in registry:
                    raise ValueError(
                        f"{label} name '{name}' already registered to "
                        f"{registry[name].__name__}"
                    )
                registry[name] = cls
            return cls
        return decorator
    return register


# tag_defaults declare which tags every entry carries (None until set).
register_entity = _make_decorator(_ENTITY_REGISTRY, "Entity", level=None)
register_operator = _make_decorator(_OPERATOR_REGISTRY, "Operator", level=None, kind=None)
register_field = _make_decorator(_FIELD_REGISTRY, "Field", couples_to=None, frame=None)


def get_entity(name: str) -> type:
    return _ENTITY_REGISTRY[name]


def get_operator(name: str) -> type:
    return _OPERATOR_REGISTRY[name]


def get_field(name: str) -> type:
    return _FIELD_REGISTRY[name]


def operators_at_level(level: str) -> dict[str, type]:
    """All operators registered to act at `level` (the LLM's action set)."""
    return {n: c for n, c in _OPERATOR_REGISTRY.items() if getattr(c, "LEVEL", None) == level}


def operators_of_kind(kind: str) -> dict[str, type]:
    return {n: c for n, c in _OPERATOR_REGISTRY.items() if getattr(c, "KIND", None) == kind}


def catalog_summary() -> str:
    """Human-readable table of everything registered — printed by docs/CLI."""
    def tag(c, name):
        return str(getattr(c, name, None))
    lines = ["# entities"]
    for n, c in sorted(_ENTITY_REGISTRY.items()):
        lines.append(f"  {n:18s} level={tag(c, 'LEVEL')}")
    lines.append("# operators")
    for n, c in sorted(_OPERATOR_REGISTRY.items()):
        lines.append(f"  {n:18s} level={tag(c, 'LEVEL'):10s} kind={tag(c, 'KIND')}")
    lines.append("# fields")
    for n, c in sorted(_FIELD_REGISTRY.items()):
        lines.append(f"  {n:18s} couples_to={tag(c, 'COUPLES_TO'):10s} frame={tag(c, 'FRAME')}")
    return "\n".join(lines)

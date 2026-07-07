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


# The closed set of operator FAMILIES -- a conceptual taxonomy over the registry (not a
# directory layout). Every core operator declares `family=` in @register_operator; the
# audit (tools/audit_operator_registry.py) fails if a family is missing or not in this set,
# so families do not proliferate. This turns the flat registry into a browsable taxonomy:
#   operators_by_family("mechanics") -> {active_force, active_stress, gravity, ...}
OPERATOR_FAMILIES = {
    "motion",       # individual self-propulsion + kinematics (glide, drag, bounce, velocity_cruise, sediment)
    "interaction",  # neighbour/pairwise forces (attraction_repulsion, squared_law, cohesion, separation, velocity_align)
    "polarity",     # heading steering (polarity_align, polarity_flow_align)
    "fields",       # scalar/vector field ops (diffuse, decay, deposit, sense, chemotax, activation_pulse, pacemaker, playback)
    "mechanics",    # body forces / active stress on the continuum (active_force, active_stress, gravity, mpm_anchor, mpm_spin)
    "mpm",          # the MLS-MPM substep machinery (mpm_strain/scatter/gather/grid_update, mls_mpm_mechanics, apply_material_map)
    "coupling",     # cross-substrate transfer (agent_scatter, agent_gather, agent_remodel)
    "hierarchy",    # parent<->child plumbing (aggregate, broadcast)
    "growth",       # structural population change (cell_divide, cell_grow)
    "topology",     # graph rewire (radius_graph)
}


def operators_by_family(family: str) -> dict[str, type]:
    """All CANONICAL operators in `family` (aliases excluded)."""
    return {n: c for n, c in _OPERATOR_REGISTRY.items()
            if getattr(c, "FAMILY", None) == family
            and getattr(c, "REGISTERED_NAMES", [n])[0] == n}


def catalog_summary() -> str:
    """Human-readable table of everything registered — printed by docs/CLI.

    Each operator also surfaces its declarative metadata: MECHANISM_TAGS (a capability
    index — "find every operator that does long_range") and PARAM_ROLES (a per-knob
    glossary — what each tunable param *means*). This is the single consumer that makes
    that metadata live rather than dead: it feeds the operator catalog the docs and the
    agentic mechanism-search layer read, so declaring the metadata now has an effect."""
    def tag(c, name):
        return str(getattr(c, name, None))
    lines = ["# entities"]
    for n, c in sorted(_ENTITY_REGISTRY.items()):
        lines.append(f"  {n:18s} level={tag(c, 'LEVEL')}")
    lines.append("# operators (by family; canonical names, aliases in parens)")
    canon = [(n, c) for n, c in _OPERATOR_REGISTRY.items()
             if getattr(c, "REGISTERED_NAMES", [n])[0] == n]        # skip aliases
    for fam in sorted(OPERATOR_FAMILIES):
        ops = sorted((n, c) for n, c in canon if getattr(c, "FAMILY", None) == fam)
        if not ops:
            continue
        lines.append(f"## {fam}")
        for n, c in ops:
            al = getattr(c, "REGISTERED_NAMES", [n])[1:]
            alias = f"  (alias {', '.join(al)})" if al else ""
            lines.append(f"  {n:20s} level={tag(c, 'LEVEL'):10s} kind={tag(c, 'KIND'):10s} emit={tag(c, 'EMIT')}{alias}")
            tags = getattr(c, "MECHANISM_TAGS", None)
            if tags:
                lines.append(f"      tags:  {', '.join(tags)}")
            roles = getattr(c, "PARAM_ROLES", None)
            if roles:
                for p, role in roles.items():
                    lines.append(f"      · {p:11s} {role}")
    orphan = sorted(n for n, c in canon if getattr(c, "FAMILY", None) not in OPERATOR_FAMILIES)
    if orphan:
        lines.append(f"## (NO/UNKNOWN FAMILY -- audit will fail): {', '.join(orphan)}")
    lines.append("# fields")
    for n, c in sorted(_FIELD_REGISTRY.items()):
        lines.append(f"  {n:18s} couples_to={tag(c, 'COUPLES_TO'):10s} frame={tag(c, 'FRAME')}")
    return "\n".join(lines)

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
    @register_operator("signal", set="neuron", kind="lateral")
    class SignalOperator(Operator):
        ...

    op_cls   = get_operator("signal")
    at_neuron = operators_at_set("neuron")       # -> {name: cls, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ENTITY_REGISTRY: dict[str, type] = {}
_OPERATOR_REGISTRY: dict[str, type] = {}    # name -> DEFAULT implementation class (enumeration / back-compat)
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


# tag_defaults declare which tags every entry carries (None until set). Canonical tags:
#   entity `depth=` -- the hierarchy DEPTH integer (0 = leaf).
register_entity = _make_decorator(_ENTITY_REGISTRY, "Entity", depth=None)
register_field = _make_decorator(_FIELD_REGISTRY, "Field", couples_to=None, frame=None)


# --------------------------------------------------------------------------- #
#  Operator contract + implementations (Plexus2 sec. 5)
# --------------------------------------------------------------------------- #
@dataclass
class OperatorContract:
    """A biological operator: a FIXED typed signature (its biology) with one or more
    interchangeable numerical IMPLEMENTATIONS (their numerics). The contract owns
    name/kind/family/signature; each implementation owns only how the delta is computed
    plus its capabilities (supported dims, differentiability). Selecting an
    implementation is a numerical choice, never a biological one --
    `{op: neuron_voltage, implementation: hodgkin_huxley}` vs `... implementation: gnn}`
    is the same contract. The default is the sole / first-registered implementation, so
    every existing single-implementation spec is unaffected."""
    name: str
    kind: str | None = None
    family: str | None = None
    set: str | None = None
    signature: dict = field(default_factory=dict)
    implementations: dict = field(default_factory=dict)   # impl_name -> class
    default: str | None = None

    def get(self, implementation: str | None = None) -> type:
        impl = implementation or self.default
        if impl not in self.implementations:
            raise KeyError(
                f"operator {self.name!r} has no implementation {impl!r}; "
                f"available: {sorted(self.implementations)}")
        return self.implementations[impl]

    def capabilities(self) -> dict:
        """Per-implementation capability table: supported dims + differentiability."""
        return {i: {"dims": list(getattr(c, "SUPPORTED_DIMS", [])),
                    "differentiable": bool(getattr(c, "DIFFERENTIABLE", True))}
                for i, c in self.implementations.items()}


_OP_CONTRACTS: dict[str, OperatorContract] = {}   # name -> contract (signature + all implementations)


def register_operator(*names: str, implementation: str | None = None, **tags):
    """Register an operator implementation. The FIRST registration of a name creates its
    contract (from the class's typed `signature()`); a later registration of the SAME name
    with a different `implementation=` adds an interchangeable implementation to that same
    contract. `set=`/`kind=`/`family=` and any other tag are stamped as UPPER-case class
    attributes (SET / KIND / FAMILY ...). One decorator may list alias names, which each
    resolve to the same class."""
    tags.setdefault("set", None)
    tags.setdefault("kind", None)
    def decorator(cls):
        for k, v in tags.items():
            setattr(cls, k.upper(), v)
        cls.REGISTERED_NAMES = list(names)
        impl = implementation or "default"
        cls.IMPLEMENTATION = impl
        for name in names:
            contract = _OP_CONTRACTS.get(name)
            if contract is None:
                contract = OperatorContract(
                    name=name, kind=getattr(cls, "KIND", None),
                    family=getattr(cls, "FAMILY", None), set=getattr(cls, "SET", None),
                    signature=cls.signature() if hasattr(cls, "signature") else {},
                    default=impl)
                _OP_CONTRACTS[name] = contract
                _OPERATOR_REGISTRY[name] = cls          # default impl: enumeration + back-compat
            else:
                if impl in contract.implementations:
                    raise ValueError(
                        f"operator {name!r} already has implementation {impl!r} "
                        f"({contract.implementations[impl].__name__})")
                # implementations of one contract must share its biology (kind); only the
                # numerics may differ.
                if getattr(cls, "KIND", None) != contract.kind:
                    raise ValueError(
                        f"operator {name!r} implementation {impl!r} has kind "
                        f"{getattr(cls, 'KIND', None)!r}, but the contract's kind is "
                        f"{contract.kind!r}; implementations may differ only in numerics.")
            contract.implementations[impl] = cls
        return cls
    return decorator


def get_entity(name: str) -> type:
    return _ENTITY_REGISTRY[name]


# --------------------------------------------------------------------------------------------
# LEGACY OPERATOR NAMES
#
# Cedric, 6 August: *"I found inconsistency in the naming, some seed operator are named xxx_seed
# other seed_xxx, can you uniform to seed_xxx"*. He is right and it was not two cases: of the 18
# operators that establish an initial condition, 14 read `xxx_seed` and 2 read `seed_xxx`.
#
# An operator name is a VERB ON A NOUN everywhere else in this registry -- `divide_3d`,
# `reconnect_t1_3d`, `shape_to_chem` -- and `cell_rd_seed` inverts that for no reason, which is
# how the same act ended up with two spellings and a reader has to know which is which.
#
# WHY AN ALIAS AND NOT A CLEAN BREAK. 305 files name `cell_rd_seed`, nearly all of them archived
# specs and run records. Unlike `mode: tip`, a rename changes NO semantics -- the old name runs the
# identical class -- so resolving it silently is safe, where silently reinterpreting `tip` as
# `scatter` would not have been. Archived specs keep loading; new specs get one spelling.
#
# The map is applied at RESOLUTION, not at registration, so only the canonical name is in
# `_OP_CONTRACTS` and every enumeration (coverage, the menu, the battery) sees one entry per
# operator rather than two.
LEGACY_OPERATOR_NAMES = {
    "aggregate_seed": "seed_aggregate",
    "basement_membrane_seed": "seed_basement_membrane",
    "block_seed": "seed_block",
    "cell_rd_seed": "seed_cell_rd",
    "cell_seed": "seed_cell",
    "coupled_seed_2d": "seed_coupled_2d",
    "ecm_seed": "seed_ecm",
    "mesh_seed": "seed_mesh",
    "nca_seed": "seed_nca",
    "sheet_seed": "seed_sheet",
    "spiral_seed": "seed_spiral",
    "tissue_seed": "seed_tissue",
    "tissue_seed_3d": "seed_tissue_3d",
    "vesicle_seed": "seed_vesicle",
}


def canonical_operator(name: str) -> str:
    """The current name for `name`, translating a pre-6-August seed operator."""
    return LEGACY_OPERATOR_NAMES.get(name, name)


def get_operator(name: str, implementation: str | None = None) -> type:
    """The implementation class for operator `name` -- the default, or the named
    `implementation`. Raises KeyError if the operator or the implementation is unknown."""
    return _OP_CONTRACTS[canonical_operator(name)].get(implementation)


def get_contract(name: str) -> OperatorContract:
    """The full operator contract (signature + every registered implementation)."""
    return _OP_CONTRACTS[canonical_operator(name)]


def get_field(name: str) -> type:
    return _FIELD_REGISTRY[name]


def operators_at_set(set_name: str) -> dict[str, type]:
    """All operators registered to act on `set_name` (the LLM's action set for that
    biological set). `set_name` is a set like "cell"/"neuron", not a depth."""
    return {n: c for n, c in _OPERATOR_REGISTRY.items() if getattr(c, "SET", None) == set_name}


def operators_at_level(level: str) -> dict[str, type]:
    """Deprecated alias for `operators_at_set` (the arg names a set, never a depth)."""
    return operators_at_set(level)


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
        lines.append(f"  {n:18s} depth={tag(c, 'DEPTH')}")
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
            lines.append(f"  {n:20s} set={tag(c, 'SET'):10s} kind={tag(c, 'KIND'):10s} emit={tag(c, 'EMIT')}{alias}")
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

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
    implementations: dict = field(default_factory=dict)   # variant name -> class (models + impls)
    default: str | None = None
    # WHICH AXIS EACH VARIANT VARIES. `model` = a different biological hypothesis at this slot;
    # `implementation` = the same biology computed differently. See the contract paragraph in
    # plexus2.tex: conflating them left every finding recorded against `cell_chem_react` actually
    # about Gray-Scott, and the search unable to tell an experiment from a control.
    axis: dict = field(default_factory=dict)              # variant name -> "model" | "implementation"

    def get(self, implementation: str | None = None, model: str | None = None,
            variant: str | None = None) -> type:
        """The class for a named variant.

        `model=` / `implementation=` are the SPEC's keys and are checked against the axis the
        variant was registered on -- naming one where the other is meant is refused, because the
        word is the claim. `variant=` is the axis-agnostic lookup the engine uses once the schema
        has already made that check; re-checking there would fail every model the schema just
        admitted.
        """
        if model is not None and implementation is not None:
            raise KeyError(f"operator {self.name!r}: name a model or an implementation, not both")
        if variant is not None:
            if variant not in self.implementations:
                raise KeyError(f"operator {self.name!r} has no variant {variant!r}; "
                               f"models: {self.models() or '-'}  "
                               f"implementations: {self.impls() or '-'}")
            return self.implementations[variant]
        want = model if model is not None else implementation
        variant = want or self.default
        if variant not in self.implementations:
            kinds = {"model": sorted(v for v, a in self.axis.items() if a == "model"),
                     "implementation": sorted(v for v, a in self.axis.items()
                                              if a == "implementation")}
            raise KeyError(
                f"operator {self.name!r} has no variant {variant!r}; "
                f"models: {kinds['model'] or '-'}  implementations: {kinds['implementation'] or '-'}")
        got = self.axis.get(variant, "implementation")
        if want is not None:
            asked = "model" if model is not None else "implementation"
            if got != asked:
                raise KeyError(
                    f"operator {self.name!r}: {variant!r} is a {got}, not a{'n' if asked[0] == 'i' else ''}"
                    f" {asked}. Write `{got}: {variant}`. "
                    + ("A model is a different biological hypothesis at this slot, so swapping it "
                       "is an experiment; an implementation is the same biology computed "
                       "differently." if got == "model" else
                       "An implementation is the same biology computed differently; a result that "
                       "changes under the swap is about discretisation, not about the tissue."))
        return self.implementations[variant]

    def models(self) -> list:
        return sorted(v for v, a in self.axis.items() if a == "model")

    def impls(self) -> list:
        return sorted(v for v, a in self.axis.items() if a == "implementation")

    def capabilities(self) -> dict:
        """Per-implementation capability table: supported dims + differentiability."""
        return {i: {"dims": list(getattr(c, "SUPPORTED_DIMS", [])),
                    "differentiable": bool(getattr(c, "DIFFERENTIABLE", True))}
                for i, c in self.implementations.items()}


_OP_CONTRACTS: dict[str, OperatorContract] = {}   # name -> contract (signature + all implementations)


def register_operator(*names: str, implementation: str | None = None,
                      model: str | None = None, **tags):
    """Register an operator implementation. The FIRST registration of a name creates its
    contract (from the class's typed `signature()`); a later registration of the SAME name
    with a different `implementation=` adds an interchangeable implementation to that same
    contract. `set=`/`kind=`/`family=` and any other tag are stamped as UPPER-case class
    attributes (SET / KIND / FAMILY ...). One decorator may list alias names, which each
    resolve to the same class."""
    tags.setdefault("set", None)
    tags.setdefault("kind", None)
    if implementation is not None and model is not None:
        raise ValueError(f"operator {names[0]!r}: a variant is a model OR an implementation")
    def decorator(cls):
        for k, v in tags.items():
            setattr(cls, k.upper(), v)
        cls.REGISTERED_NAMES = list(names)
        impl = model or implementation or "default"
        axis = "model" if model is not None else "implementation"
        cls.IMPLEMENTATION = impl
        cls.VARIANT_AXIS = axis
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
                        f"operator {name!r} already has variant {impl!r} "
                        f"({contract.implementations[impl].__name__})")
                # Variants of one contract must share its KIND. This is a weak guard and always
                # was: all four `cell_chem_from_shape` variants are `lateral` while each senses a
                # different physical quantity, so the check passed on four distinct biological
                # hypotheses wearing one label. The `model`/`implementation` axis is what
                # actually carries that distinction.
                if getattr(cls, "KIND", None) != contract.kind:
                    raise ValueError(
                        f"operator {name!r} variant {impl!r} has kind "
                        f"{getattr(cls, 'KIND', None)!r}, but the contract's kind is "
                        f"{contract.kind!r}; a variant may not change the kind.")
            contract.implementations[impl] = cls
            contract.axis[impl] = axis
        return cls
    return decorator


def get_entity(name: str) -> type:
    return _ENTITY_REGISTRY[name]


def get_operator(name: str, implementation: str | None = None,
                 model: str | None = None, variant: str | None = None) -> type:
    """The class for operator `name` -- the default variant, or a named `model` /
    `implementation`. Raises KeyError if the operator or the variant is unknown, or if the
    variant exists on the other axis (naming a model an implementation is refused)."""
    return _OP_CONTRACTS[name].get(implementation, model, variant)


def get_contract(name: str) -> OperatorContract:
    """The full operator contract (signature + every registered implementation)."""
    return _OP_CONTRACTS[name]


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
    "growth",       # structural population change (agent_divide, agent_grow)
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

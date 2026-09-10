"""The spec schema: load + VALIDATE -- the contract gatekeeper.

A complete Plexus run is a single declarative `spec.yaml` (`general`, `sets`,
`fields`, `seed` (optional -- x_0, once, before dynamics), `operators`
(+ selectors), `schedule`, `plotting`). This module is the
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

from plexus.paths import warn

import re
from dataclasses import dataclass, field
from typing import Optional

import yaml

from plexus.units import Units, parse as parse_units

# importing plexus populates base + registry; operator modules register themselves
from plexus.models import registry
from plexus.models.base import KINDS, EMITS
from plexus.models.state import INTEGRATIONS

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
    impl: Optional[str] = None      # which VARIANT of the op's contract -- a `model` (a different
                                    # biological hypothesis at this slot) or an `implementation`
                                    # (the same biology computed differently). One field, because
                                    # the engine instantiates a class either way; the SPEC keeps
                                    # them as separate keys, because the word is the claim.
    params: dict = field(default_factory=dict)


@dataclass
class Spec:
    name: str
    seed: int                       # RNG seed (general.seed) -- NOT the seed: PHASE below;
                                     # kept as `seed` for back-compat, disambiguated by type.
    n_frames: int
    dt: float
    sets: dict
    fields: dict
    operators: list[OpSpec]
    schedule: list
    seed_ops: list[OpSpec] = field(default_factory=list)   # the seed: section (x_0), NOT
                                                            # a schedule -- see `engine.seed()`
    obstacles: list = field(default_factory=list)   # wall rectangles [x0,y0,x1,y1] or discs [cx,cy,r]
    boundary: str = "wall"                           # 'wall' (clamp) or 'periodic' (torus)
    world: float = 1.0                               # domain width (= world_size[0]); legacy 2D scalar
    dim: int = 2                                     # spatial dimensions (the global dimension contract)
    world_size: list = field(default_factory=lambda: [1.0, 1.0])   # per-axis box [w0 .. w_{D-1}]
    plotting: dict = field(default_factory=dict)     # render STYLE (colormap, point_size, ...) — read by plexus.plot
    # WHICH DRIVER RUNS THE SCHEDULE. `default` is plexus.engine.run; a name N resolves to
    # plexus.<N>_engine.run. A mode of RUNNING, not a different physics -- `continuous` still
    # delegates to the same loop and exists only to refuse specs its operators cannot check.
    engine: str = "default"
    record_cap: int = 10000                          # max recorded SET (position) frames; the trajectory is strided if n_frames exceeds it
    field_record_cap: int = 256                      # max recorded FIELD (grid) frames — fields are large, so a tighter cap
    # THE PHYSICAL SCALE, declared once under `general.units:` and never inferred. Three base scales
    # (length_um, time_s, force_nN) because mechanics needs three; everything else is derived (see
    # plexus/units.py). Absent => the run is dimensionless and no result from it may carry a unit.
    # `time_s` defaults to 1.0, i.e. THE CONVENTION IS THAT `dt` IS IN SECONDS.
    units: "Units" = field(default_factory=lambda: Units(declared=False))
    # SAVE THE TRAJECTORY, OR DO NOT. None = the legacy `record_cap` path.
    save_data: bool = None


_RESERVED = {"op", "at", "to", "from", "implementation", "model"}
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
            # TWO WAYS TO SAY THE SAME THING, AND EXACTLY ONE OF THEM PER SET. `fraction` is a
            # share of the set and `count` is a number of elements; a set may state either, never
            # both, because two statements of one partition are two chances to disagree and
            # nothing downstream could tell which was meant. `count` exists because an inventory
            # -- a cell atlas's 421 plasma-membrane pieces, 77 mitochondria -- is naturally
            # counted, and writing it as 0.100790 of 4,177 both hides the number and reaches it
            # only by rounding. Counts are converted to fractions HERE, so every other reader
            # (`_type_fracs`, the operators, the atlas) is untouched; `_assign_types` then uses
            # the count itself, so the assignment is exact by construction.
            n_counted = sum(1 for t in types.values() if "count" in t)
            n_frac = sum(1 for t in types.values() if "fraction" in t)
            if n_counted and n_counted != len(types):
                missing = [k for k, t in types.items() if "count" not in t]
                raise ValueError(
                    f"set {sname!r}: {n_counted} of {len(types)} types declare `count:` -- "
                    f"missing on {missing}. A partition is stated one way for the whole set.")
            if n_counted and n_frac:
                both = [k for k, t in types.items() if "count" in t and "fraction" in t]
                raise ValueError(
                    f"set {sname!r} types {both} declare BOTH `count:` and `fraction:`. They are "
                    f"two statements of one partition -- give the count, or give the share.")
            if n_counted:
                # `count` is measured against what the set actually holds: `n` for a root set,
                # `per_parent` for a contained one (the per-parent inventory, which is what
                # `per_parent` already means).
                live = s.get("n", s.get("per_parent"))
                if isinstance(live, dict):
                    raise ValueError(
                        f"set {sname!r} uses `count:` on its types and a per-type `per_parent:` "
                        f"mapping. `count` counts THIS set's elements and the mapping counts its "
                        f"CHILDREN's -- put the mapping on the child set, where it belongs.")
                if live is None:
                    raise ValueError(f"set {sname!r} uses `count:` but declares neither `n:` nor "
                                     f"`per_parent:`, so there is nothing for the counts to add up to")
                tot = sum(int(t["count"]) for t in types.values())
                if tot != int(live):
                    raise ValueError(
                        f"set {sname!r} type counts sum to {tot:,}, but the set holds "
                        f"{int(live):,}. Every element belongs to exactly one type.")
                for t in types.values():
                    t["fraction"] = int(t["count"]) / float(live)
            else:
                total = sum(t["fraction"] for t in types.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(f"set {sname!r} type fractions sum to {total}, must be 1.0")
        # `buffer` is the allocated slot count for a cardinality-changing set
        # (occupancy `occ` marks the live subset). It must hold the initial set.
        buf = s.get("buffer")
        if buf is not None:
            live0 = s.get("n", s.get("per_parent"))
            if isinstance(live0, dict):
                live0 = max(int(v) for v in live0.values())   # the largest block a parent may hold
            if live0 is not None and int(buf) < int(live0):
                raise ValueError(
                    f"set {sname!r} buffer={buf} is smaller than its initial size {live0}; "
                    f"a structural (divide/duplicate) run needs buffer >= initial.")
        # optional `mesh:` -- the set's elements carry a half-edge surface (see `plexus.models.mesh`).
        #
        # VALIDATED HERE AND LOUDLY, because the failure mode of not validating is silence: before
        # this, `mesh: half-edge` (a hyphen) or `mesh:` on the wrong set parsed clean, allocated
        # nothing, and the run proceeded with a spec that claimed a topology it did not have.
        # ---- `mesh:` NAMES A SET NOW, NOT A KIND -------------------------------------------------
        # It used to name one of `MESH_KINDS` -- the single string "half_edge" -- and the topology
        # it stood for was declared nowhere: `E_srce`, `E_trgt` and `E_face` were three parallel
        # arrays on a table hanging off this set, and the CELL those faces are was a second key,
        # `cell_set:`, repeated as a parameter on every operator that crossed between them.
        #
        # A HALF-EDGE IS AN ELEMENT, SO IT IS A SET. plexus2 sec. Hierarchy: a many-to-many relation
        # needs no new primitive, because a relation IS a set with functions out of it. Every
        # half-edge has exactly one source vertex, exactly one target vertex and exactly one face --
        # three functions, so three maps, and the whole mesh is those three plus the sets they land
        # in. `index_add(0, E_face, ...)` in `face_geometry_3d` is then a declared AGGREGATE along
        # half_edge -> cell and `pos[E_srce]` a declared BROADCAST along half_edge -> vertex, which
        # are the two families sec. Hierarchy says are the only ones allowed to cross.
        #
        # `cell_set:` RETIRES INTO `maps.face`. The pairing is no longer a key that names a set; it
        # is the codomain of a declared map, so `edge_flip` cannot renumber a set the spec never
        # said it touches, and no operator has to be told a relation the mesh already states.
        #
        # WHY THE MAPS ARE NOT `pre`/`post`. An edge-set is built from a STATIC `edges:` list or an
        # npz -- the connectome is given and does not change. A half-edge table is rewritten by
        # every division and every T1, so its maps are OWNED BY THE TOPOLOGY OPERATORS, and its `n:`
        # is a capacity rather than a count. Same idea, different lifetime; sharing the `pre`/`post`
        # keys would have hidden that.
        mk = s.get("mesh")
        if mk is not None:
            if s.get("pre") is not None or s.get("post") is not None:
                raise ValueError(
                    f"set {sname!r} declares `mesh:` and is an EDGE-SET (it has pre/post). An "
                    f"edge-set's elements are connections; a mesh's are vertices.")
            if int(raw.get("general", {}).get("dim", 2)) != 3:
                raise ValueError(
                    f"set {sname!r} declares `mesh: {mk}` but the model is "
                    f"{raw.get('general', {}).get('dim')}D -- every mesh operator declares "
                    f"SUPPORTED_DIMS = [3].")
            if mk not in raw["sets"]:
                raise ValueError(
                    f"set {sname!r} declares `mesh: {mk}`, which is not a set in this spec "
                    f"(have: {', '.join(sorted(raw['sets']))}). `mesh:` names the half-edge SET "
                    f"whose `maps:` say what the topology is; it is no longer a kind.")
            if s.get("cell_set") is not None:
                raise ValueError(
                    f"set {sname!r} still declares `cell_set: {s['cell_set']}`. That key retired: "
                    f"a face of the mesh IS a cell, and the pairing is now `maps.face` on the "
                    f"{mk!r} set. Delete it from here and from every operator line.")
            hs = raw["sets"][mk]
            for role in ("srce", "trgt", "face"):
                if role not in (hs.get("maps") or {}):
                    raise ValueError(
                        f"set {mk!r} is used as {sname!r}'s mesh but declares no `maps.{role}:`. A "
                        f"half-edge has exactly one source vertex, one target vertex and one face; "
                        f"all three are functions out of it and all three must be declared.")

        # ---- `maps:` -- named functions OUT of this set, one row to one row of the codomain -------
        mp = s.get("maps")
        if mp is not None:
            if not isinstance(mp, dict) or not mp:
                raise ValueError(f"set {sname!r}: `maps:` must be a mapping of role -> set name")
            for role, tgt in mp.items():
                if tgt not in raw["sets"]:
                    raise ValueError(
                        f"set {sname!r} map {role!r} lands in {tgt!r}, which is not a set in this "
                        f"spec (have: {', '.join(sorted(raw['sets']))})")
                if tgt == sname:
                    raise ValueError(
                        f"set {sname!r} map {role!r} lands in itself. A map goes OUT of a set.")
            if s.get("pre") is not None or s.get("post") is not None:
                raise ValueError(
                    f"set {sname!r} declares both `maps:` and `pre`/`post`. They are the same idea "
                    f"with different lifetimes -- pre/post are a static edge list, `maps` are "
                    f"owned by the topology operators -- so a set has one or the other.")

        # `entity:` -- WHICH REGISTERED KIND PROVIDES THIS SET'S STATE, when the set's own name is
        # not that kind. Validated here because the failure is otherwise silent and late: an
        # unregistered name falls back to the pos/vel default, no `provision` runs, and an MPM set
        # dies several hundred lines later on a missing `F`.
        ent = s.get("entity")
        if ent is not None:
            from plexus.models.registry import _ENTITY_REGISTRY
            if ent not in _ENTITY_REGISTRY:
                raise ValueError(
                    f"set {sname!r} declares `entity: {ent}`, which is not a registered entity "
                    f"(have: {', '.join(sorted(_ENTITY_REGISTRY))})")

        # optional `state:` block -- the set's StateSchema (the fifth primitive). Absent =>
        # the spatial pos/vel default. Each entry is a width (int) or {width, integration,
        # boundary, role, record, unit}. Validate here so a malformed schema fails at load.
        #
        # `unit:` IS NOT VALIDATED HERE, AND THAT IS THE POINT. It is a `plexus.units` quantity
        # string -- "volume", "1/T", "volume[polyhedron]" -- and an unreadable one parses to
        # UNKNOWN rather than raising, because the units layer is WARN-ONLY: a typo in an
        # annotation must never stop a run. The checker reports it; the loader does not refuse it.
        st = s.get("state")
        if st is not None:
            if not isinstance(st, dict) or not st:
                raise ValueError(f"set {sname!r} `state:` must be a non-empty mapping of block -> width|dict")
            for bname, decl in st.items():
                if isinstance(decl, dict):
                    if int(decl.get("width", 1)) < 1:
                        raise ValueError(f"set {sname!r} state block {bname!r}: width must be >= 1")
                    integ = decl.get("integration", "first_order")
                    if integ not in INTEGRATIONS:
                        raise ValueError(f"set {sname!r} state block {bname!r}: integration {integ!r} "
                                         f"not one of {INTEGRATIONS}")
                    bnd = decl.get("boundary", "free")
                    if bnd not in ("free", "world"):
                        raise ValueError(f"set {sname!r} state block {bname!r}: boundary {bnd!r} "
                                         f"must be 'free' or 'world'")
                elif int(decl) < 1:
                    raise ValueError(f"set {sname!r} state block {bname!r}: width must be >= 1")
        # optional `edge_set` -- a set whose elements are connections, joined to endpoint
        # sets by `pre`/`post` incidence maps. Requires a containing `parent`, both endpoint
        # sets, and an inline `edges: [[pre, post], ...]` list (PR2: inline for determinism).
        if s.get("edge_set"):
            if "parent" not in s:
                raise ValueError(f"edge-set {sname!r} needs a `parent:` (its containing set, e.g. network)")
            for role in ("pre", "post"):
                if role not in s:
                    raise ValueError(f"edge-set {sname!r} needs a `{role}:` endpoint set")
                if s[role] not in raw["sets"]:
                    raise ValueError(f"edge-set {sname!r} `{role}: {s[role]}` is not a declared set")
            # the connections: an inline list, or an `.npz` for anything connectome-sized.
            # EXACTLY ONE of the two, because a spec that gives both is a spec whose author
            # believes two different things about which connectome it runs.
            has_inline, has_file = "edges" in s, bool(s.get("edges_file"))
            if has_inline and has_file:
                raise ValueError(
                    f"edge-set {sname!r} gives both `edges:` and `edges_file:` -- name one. "
                    f"The file is the connectome; an inline list beside it is a second answer "
                    f"to the same question.")
            if not has_inline and not has_file:
                raise ValueError(f"edge-set {sname!r} needs `edges: [[pre, post], ...]` or "
                                 f"`edges_file: <path.npz>` (edge_index [2, E] + weights [E])")
            if has_file:
                if "weights" in s:
                    raise ValueError(
                        f"edge-set {sname!r} sets `weights:` beside `edges_file:` -- the weights "
                        f"belong in the npz, next to the edges they weight.")
            else:
                if not isinstance(s["edges"], list) or not s["edges"]:
                    raise ValueError(f"edge-set {sname!r} needs a non-empty inline `edges: [[pre, post], ...]` list")
                for e in s["edges"]:
                    if not (isinstance(e, (list, tuple)) and len(e) in (2, 3)):
                        raise ValueError(f"edge-set {sname!r}: each edge must be [pre, post] or "
                                         f"[pre, post, weight], got {e!r}")

    # --- shared operator-line resolution, used for both `operators:` and the
    # `seed:` section: names registered, valid KIND, selectors + fields exist,
    # capability contract satisfied. Returns (cls, kind, OpSpec). ------------ #
    def types_in_chain(set_name):
        # a required property may live on the set's types or be inherited from a
        # parent set (mpm acts on particles but reads `youngs` off the cell type)
        seen = set()
        while set_name and set_name not in seen:
            seen.add(set_name)
            ts = raw["sets"][set_name].get("types")
            if ts:
                return set_name, ts
            set_name = raw["sets"][set_name].get("parent")
        return None, {}

    def resolve_op_line(o):
        name = o["op"]
        # `model:` and `implementation:` are separate keys and naming one where the other is meant
        # is refused by the contract. Gray-Scott and Brusselator are not two ways of computing one
        # reaction -- their parameter sets are disjoint -- so calling the swap an implementation
        # made every finding recorded against `cell_chem_react` silently a finding about Gray-Scott.
        impl = o.get("implementation")
        modl = o.get("model")
        if impl is not None and modl is not None:
            raise ValueError(
                f"operator {o['op']!r} names both `model: {modl}` and "
                f"`implementation: {impl}`; a variant is one or the other")
        # resolve the SELECTED implementation of the operator's contract; capability
        # checks below then run against the implementation that will actually execute.
        try:
            cls = registry.get_operator(name, impl, modl)
        except KeyError:
            try:
                contract = registry.get_contract(name)
            except KeyError:
                raise ValueError(
                    f"operator {name!r} not in registry. Available: "
                    f"{sorted(registry._OPERATOR_REGISTRY)}")
            raise ValueError(
                f"operator {name!r} has no variant {(modl or impl)!r}; "
                f"models: {contract.models() or '-'}  "
                f"implementations: {contract.impls() or '-'}")
        kind = getattr(cls, "KIND", None)
        if kind not in KINDS:
            raise ValueError(
                f"operator {name!r} has unrecognised kind {kind!r}; expected one of {KINDS}.")
        supported = getattr(cls, "SUPPORTED_DIMS", [2])
        if dim not in supported:
            raise ValueError(
                f"operator {name!r} supports dims {supported}, not dim={dim} "
                f"(set general.dim, or use a dimension-generic / *_3d operator).")
        # integration-order vocabulary (Axis A): the class default and the optional spec
        # override must both name a recognised state (one vocabulary, no synonyms).
        cls_emit = getattr(cls, "EMIT", None)
        if cls_emit is not None and cls_emit not in EMITS:
            raise ValueError(
                f"operator {name!r} has invalid EMIT {cls_emit!r}; "
                f"expected one of {EMITS} or None.")
        emit = o.get("emit")
        if emit is not None and emit not in EMITS:
            raise ValueError(
                f"operator {name!r} sets emit: {emit!r}, not one of {EMITS}.")
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
        for prop in getattr(cls, "REQUIRES_TYPE_PROPS", []):
            owner, set_types = types_in_chain(sel.set)
            if not set_types:
                raise ValueError(
                    f"operator {name!r} requires per-type property {prop!r}, but neither "
                    f"{sel.set!r} nor its parents declare `types`.")
            # ONE PROPERTY, MORE THAN ONE WAY TO SPELL IT. `youngs` and `bulk_modulus` are two
            # routes to the same Lame pair -- for a liquid, where mu = 0, they are the SAME number
            # reached differently -- so requiring the first by name would forbid the second. The
            # operator declares the alternatives and the type satisfies any one of them.
            alts = getattr(cls, "TYPE_PROP_ALTERNATIVES", {}).get(prop, ())
            for tname, t in set_types.items():
                if prop not in t and not any(a in t for a in alts):
                    also = f" (or {' / '.join(alts)})" if alts else ""
                    raise ValueError(
                        f"operator {name!r} requires property {prop!r}{also} on every type of "
                        f"{owner!r}; missing on type {tname!r}. "
                        f"(declared in {cls.__name__}.REQUIRES_TYPE_PROPS)")
        opspec = OpSpec(op=name, on=sel, to=o.get("to"), frm=o.get("from"),
                        impl=(modl or impl), params=params)
        return cls, kind, opspec

    # --- operators: names registered, valid KIND, selectors + fields exist -- #
    ops = []
    _legacy_seed_ops = set()          # kind="seed" operators declared the old way -- see below
    for o in raw["operators"]:
        cls, kind, opspec = resolve_op_line(o)
        ops.append(opspec)
        if kind == "seed":
            _legacy_seed_ops.add(opspec.op)

    # --- seed: establishes x_0, once, before any dynamics (Model = S then Phi) #
    # A distinct, optional section. Rules (rejected at load, not special-cased by
    # the engine at run time):
    #   - every line here must resolve to a kind="seed" operator (a dynamics
    #     operator declared in `seed:` is a category error, not a convenience);
    #   - a seed line may not carry a dynamic scheduling control (`every`,
    #     `before_frame`, `after_frame`) -- those are schedule concepts, and
    #     seed has no schedule: it runs exactly once, always;
    #   - a seed-kind operator declared here may not ALSO appear in `schedule:`
    #     (checked below, once `op_names`/seed names are both known).
    _SEED_ONLY_FORBIDDEN = ("every", "before_frame", "after_frame")
    seed_ops = []
    for o in raw.get("seed", []):
        cls, kind, opspec = resolve_op_line(o)
        if kind != "seed":
            raise ValueError(
                f"seed: {o['op']!r} has kind {kind!r}, not \"seed\" -- only operators that "
                f"establish x_0 may be declared in the seed: section. A dynamics operator "
                f"belongs in operators:/schedule:.")
        bad = [k for k in _SEED_ONLY_FORBIDDEN if k in o]
        if bad:
            raise ValueError(
                f"seed: {o['op']!r} sets {bad}, a dynamic scheduling control -- seed has no "
                f"schedule, it runs exactly once. Remove {bad} from this seed: line.")
        seed_ops.append(opspec)

    # A seed-kind operator declared under `operators:` is the deprecated (pre-`seed:`)
    # spelling: still accepted so existing specs are not broken in one pass (see
    # notes/campaigns/SEED_MIGRATION.md), but warned about, and it is exactly what the next check
    # (seed in schedule:) would reject if that operator is ALSO scheduled.
    if _legacy_seed_ops:
        warn(f"[warn] deprecated: {sorted(_legacy_seed_ops)} declared in operators: with "
              f"kind=\"seed\" -- move to the seed: section, or run "
              f"`tools/migrate_seed_section.py` (see notes/campaigns/SEED_MIGRATION.md, which records that a "
              f"`before_frame` wider than 1 CANNOT be migrated: `seed:` has no window). Still "
              f"accepted for now via the legacy engine seed-window path.")

    # --- warn about per-type properties no operator reads (typo guard) ------ #
    used_props = set()
    for o in raw["operators"]:
        cls = registry.get_operator(o["op"])
        used_props |= set(getattr(cls, "REQUIRES_TYPE_PROPS", []))
        used_props |= set(getattr(cls, "OPTIONAL_TYPE_PROPS", []))   # read only in some modes (e.g. alignment per_type)
    # core/layers/block: consumed by an entity provision hook (e.g. mpm_particle), not by an operator
    #
    # `material`, `density` and `tau` join them because the provision hook reads all three -- per
    # the PARTICLE's own type, not its parent's -- so a warning that they are read by nothing would
    # be false. (Were the hook NOT to read them, that warning would be the only notice that a
    # cytosol declaring `material: liquid` had silently built as whatever its parent cell was.)
    # `bulk_modulus` is read by mpm_scatter via TYPE_PROP_ALTERNATIVES rather than by name, so the
    # used_props scan does not see it and it would be reported as read by no operator.
    # `count` is consumed by the schema itself (converted to `fraction` above) and by
    # `_assign_types`, not by an operator, so it joins the keys the typo guard already knows.
    _KNOWN_TYPE_KEYS = {"fraction", "count", "core", "layers", "block",
                        "material", "density", "tau", "bulk_modulus", "shape",
                        "eta"} | used_props          # per-type dynamic viscosity (mpm_viscosity)
    for sname, s in raw["sets"].items():
        for tname, t in s.get("types", {}).items():
            for k in t:
                if k not in _KNOWN_TYPE_KEYS:
                    warn(f"[warn] property {k!r} on {sname}.{tname} is read by no operator "
                          f"(known: {sorted(_KNOWN_TYPE_KEYS)})")

    # --- schedule: every token resolves to an operator or a builtin --------- #
    # A step may be a token, a list of tokens (run in sequence), or a substep micro-loop
    # `{substep_dt: <dt>, steps: [...]}` whose inner tokens run once per substep; the count
    # is round(general.dt / dt). (e.g. the MPM strain->P2G->grid->G2P cycle.)
    op_names = {o.op for o in ops}
    seed_op_names = {o.op for o in seed_ops}       # new-style seed:, excluded from schedule
    for step in raw["schedule"]:
        if isinstance(step, dict) and "substep" in step:
            raise ValueError("the `{substep: N, dt: X}` schedule form was removed; write "
                             "`{substep_dt: X, steps: [...]}` and set `general.dt` to the "
                             "per-frame sim time (substeps = round(general.dt / X)).")
        if isinstance(step, dict) and "substep_dt" in step:
            if not isinstance(step.get("steps"), list):
                raise ValueError("a `{substep_dt: …}` schedule step needs a `steps:` list")
            # THE KEY SET IS CLOSED, so `complie: true` is an error and not a silently ignored
            # line. A performance flag that does nothing looks exactly like a performance flag
            # that does not help, and the run it was measured against is then meaningless.
            _ok = {"substep_dt", "steps", "capture", "compile", "compile_mode",
                   "compile_recompile_limit", "compile_verbose"}
            _bad = set(step) - _ok
            if _bad:
                raise ValueError(f"unknown key(s) {sorted(_bad)} on a `{{substep_dt: …}}` step; "
                                 f"allowed: {sorted(_ok)}")
            tokens = step["steps"]
        else:
            tokens = step if isinstance(step, list) else [step]
        for tok in tokens:
            if tok in _BUILTIN_STEPS:
                continue
            # A seed: operator is never dispatched by the schedule loop -- engine.seed()
            # runs it once, before engine.run() starts. Declaring it here too is not a
            # convenience, it is the exact bug `seed:` exists to make unrepresentable (see
            # the KINDS comment in base.py): a seed re-running on the schedule's terms.
            if tok in seed_op_names:
                raise ValueError(
                    f"schedule step {tok!r} is a seed: operator (kind=\"seed\") -- seed runs "
                    f"exactly once, before the schedule, via engine.seed(). Remove it from "
                    f"schedule: (and from operators:, if declared there too).")
            if tok not in op_names:
                raise ValueError(f"schedule step {tok!r} is not a declared operator or builtin")

    _spec = Spec(
        name=gv("name"),
        seed=int(gv("seed", 0)),
        n_frames=int(gv("n_frames", 200)),
        dt=float(gv("dt", 0.05)),
        sets=raw["sets"],
        fields=raw["fields"],
        operators=ops,
        schedule=raw["schedule"],
        seed_ops=seed_ops,
        obstacles=gv("obstacles", []),
        boundary=gv("boundary", "wall"),
        world=world_size[0],
        dim=dim,
        world_size=world_size,
        plotting=raw.get("plotting", {}),
        engine=str(gv("engine", "default")).lower(),
        record_cap=int(gv("record_cap", 10000)),
        field_record_cap=int(gv("field_record_cap", 256)),
        units=parse_units(gv("units", None)),
        save_data=gv("save_data", None),
    )
    # THE UNITS CHECK: ONE PASS, WARNING ONLY, AND IT CANNOT STOP THE LOAD. It runs here because
    # this is the first moment the whole declaration is visible at once -- the base scales, every
    # set's state blocks, and every operator that resolved -- and it is outside anything that runs
    # per frame, so it costs one pass and can never slow a simulation. `units_check.check` catches
    # its own exceptions, and this call is wrapped again: a defect in the checker costs one printed
    # line and nothing else. See `plexus/units_check.py` for what it can and cannot see.
    try:
        from plexus.units_check import check as _units_check
        # THE VARIANT MAP, NOT `get_operator`. `OpSpec.impl` is `(model or implementation)` -- the
        # axis the variant was selected on is not kept -- so calling `get_operator(name, impl, impl)`
        # passes one string as BOTH axes and resolves nothing for every `model:` variant, which is
        # `cell_mechanics[apicobasal]`, `cell_die[prescribed]` and `seed_mesh[apicobasal]` on one
        # tissue spec alone. `contract.implementations` is keyed by variant name across both axes,
        # so one lookup covers them, with the contract's own default when a line names no variant.
        _classes = {}
        for _o in list(ops) + list(seed_ops):
            try:
                _c = registry.get_contract(_o.op)
                _classes[_o.op] = (_c.implementations.get(_o.impl or _c.default),
                                   dict(_o.params or {}))
            except Exception:                    # noqa: BLE001
                pass
        _units_check(_spec, _classes)
    except Exception as _e:                      # noqa: BLE001
        print(f"[units] check skipped ({type(_e).__name__}: {_e})", flush=True)
    return _spec

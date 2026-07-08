"""State: the fifth Plexus primitive, made first-class in code.

A Set (``Level``) carries a ``StateSchema`` -- a typed record of named ``Block``s.
This is the code-level form of the paper's fifth primitive (sets, state, fields,
maps, operators). ``pos``/``vel`` is just one schema; ``voltage``/``calcium``,
synaptic ``g``, a metabolite ``conc`` are others. The engine sizes, integrates, and
records state by *reading the schema*, never by hard-coding ``pos``/``vel``.

Integration is declared per block, decoupled from any spatial meaning
(``Block.integration`` is the load-bearing field):

===========================  =========================================================
integration                  meaning
===========================  =========================================================
``none``                     not integrated -- a parameter, readout, or frozen feature
``first_order``              overdamped coordinate ``x``:  ``x += dt * delta``
``second_order_coordinate``  inertial coordinate ``x``, advanced by its rate block
``second_order_rate``        the rate ``v`` of a 2nd-order coordinate:
                             ``v += dt * delta;  x += dt * v``
===========================  =========================================================

So ``EMIT=velocity`` means *"the delta for the first-order integrated block"* and
``EMIT=acceleration`` means *"the delta for the second_order_rate block"* -- never
literally spatial velocity. ``Block.boundary`` says whether the block lives in the
world box (``world`` -> clamp/wrap) or is unbounded (``free`` -> e.g. a voltage).

Backward compatibility: a legacy ``{block: (c0, c1)}`` dict still works everywhere.
``StateSchema.normalize`` turns it into a schema, and ``StateSchema`` behaves like
that dict (``schema['pos'] == (c0, c1)``, ``'pos' in schema``), so existing call
sites (``lvl.get('pos')``, ``state_schema['pos']``) are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- integration kinds (Block.integration) -------------------------------------- #
NONE = "none"
FIRST_ORDER = "first_order"
SECOND_ORDER_COORDINATE = "second_order_coordinate"
SECOND_ORDER_RATE = "second_order_rate"
INTEGRATIONS = (NONE, FIRST_ORDER, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE)

# --- boundary kinds (Block.boundary) -------------------------------------------- #
BOUNDARY_WORLD = "world"    # clamp / wrap to the world box (a spatial coordinate)
BOUNDARY_FREE = "free"      # unbounded (voltage, concentration, gating)
BOUNDARIES = (BOUNDARY_WORLD, BOUNDARY_FREE, None)


@dataclass(frozen=True)
class Block:
    """One named, contiguous chunk of a set's per-element state."""
    name: str
    width: int
    role: Optional[str] = None          # semantic label (coordinate/rate/voltage/gating/...) for docs + selectors
    integration: str = NONE             # one of INTEGRATIONS -- how the engine advances it in time
    boundary: Optional[str] = None      # BOUNDARY_WORLD | BOUNDARY_FREE | None
    record: bool = True                 # store this block in the trajectory

    def __post_init__(self):
        if self.integration not in INTEGRATIONS:
            raise ValueError(f"block {self.name!r}: integration {self.integration!r} not in {INTEGRATIONS}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"block {self.name!r}: boundary {self.boundary!r} not in {BOUNDARIES}")
        if self.width < 1:
            raise ValueError(f"block {self.name!r}: width must be >= 1, got {self.width}")


class StateSchema:
    """An ordered list of ``Block``s with contiguous column offsets.

    Dict-compatible: ``schema['pos']`` returns the ``(c0, c1)`` slice, ``'pos' in
    schema`` works, and iteration yields block names -- so every legacy call site
    that treated the schema as a ``{name: (c0, c1)}`` dict keeps working unchanged.
    """

    def __init__(self, blocks):
        self.blocks: list[Block] = list(blocks)
        self._slices: dict[str, tuple[int, int]] = {}
        self._by_name: dict[str, Block] = {}
        c = 0
        for b in self.blocks:
            if b.name in self._slices:
                raise ValueError(f"duplicate block name {b.name!r} in StateSchema")
            self._slices[b.name] = (c, c + b.width)
            self._by_name[b.name] = b
            c += b.width
        self.dim = c                     # total state width

    # --- dict compatibility (legacy `state_schema['pos'] == (c0, c1)`) ---------- #
    def __getitem__(self, name): return self._slices[name]
    def __contains__(self, name): return name in self._slices
    def __iter__(self): return iter(self._slices)
    def __len__(self): return len(self._slices)
    def keys(self): return self._slices.keys()
    def items(self): return self._slices.items()
    def values(self): return self._slices.values()
    def get(self, name, default=None): return self._slices.get(name, default)

    def block(self, name) -> Block: return self._by_name[name]
    def slice(self, name) -> tuple[int, int]: return self._slices[name]

    # --- the integration roles the engine's integrator reads -------------------- #
    @property
    def coordinate(self) -> Optional[Block]:
        """The position-like block the engine advances: the ``second_order_coordinate``
        if the set is inertial, else the ``first_order`` block if it is overdamped,
        else ``None`` (a set with no engine-integrated state)."""
        for b in self.blocks:
            if b.integration == SECOND_ORDER_COORDINATE:
                return b
        for b in self.blocks:
            if b.integration == FIRST_ORDER:
                return b
        return None

    @property
    def rate(self) -> Optional[Block]:
        """The rate block of an inertial (second-order) set, or ``None`` for a
        first-order / non-integrated set."""
        for b in self.blocks:
            if b.integration == SECOND_ORDER_RATE:
                return b
        return None

    @property
    def recorded(self) -> list[Block]:
        return [b for b in self.blocks if b.record]

    def as_dict(self) -> dict[str, tuple[int, int]]:
        return dict(self._slices)

    def __repr__(self):
        body = ", ".join(f"{b.name}[{b.width},{b.integration}]" for b in self.blocks)
        return f"StateSchema({body})"

    # --- the compatibility shim ------------------------------------------------- #
    @classmethod
    def normalize(cls, schema) -> "StateSchema":
        """Return ``schema`` if it is already a ``StateSchema``; otherwise treat it as
        a legacy ``{block: (c0, c1)}`` dict and normalize it.

        The legacy dict is the spatial ``pos``/``vel`` layout every current spec uses;
        it MUST normalize to a schema that integrates byte-identically to the old
        hard-coded path: ``pos`` -> ``second_order_coordinate`` in the ``world`` box,
        ``vel`` -> ``second_order_rate``. Any other legacy block is carried through as
        a non-integrated (``none``) block so nothing is silently integrated.
        """
        if isinstance(schema, StateSchema):
            return schema
        items = sorted(schema.items(), key=lambda kv: kv[1][0])   # contiguous by start column
        blocks = []
        for name, (c0, c1) in items:
            w = c1 - c0
            if name == "pos":
                blocks.append(Block("pos", w, role="coordinate",
                                    integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD))
            elif name == "vel":
                blocks.append(Block("vel", w, role="rate", integration=SECOND_ORDER_RATE))
            else:
                blocks.append(Block(name, w, integration=NONE))
        return cls(blocks)


def spatial_schema(dim: int) -> StateSchema:
    """The dimension-aware ``pos``/``vel`` schema (state width ``2*dim``) -- the spatial
    default every current set uses. ``pos`` is an inertial coordinate clamped/wrapped to
    the world box; ``vel`` is its rate. Byte-identical replacement for the old
    ``{'pos': (0, D), 'vel': (D, 2D)}`` dict from ``engine._dim_schema``."""
    return StateSchema([
        Block("pos", dim, role="coordinate", integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD),
        # vel is NOT recorded: the trajectory has always stored pos only, so recording vel
        # would add a state/vel group and break byte-identical output on every spatial spec.
        Block("vel", dim, role="rate", integration=SECOND_ORDER_RATE, record=False),
    ])


def schema_from_spec(state_decl: dict) -> StateSchema:
    """Build a ``StateSchema`` from a spec ``state:`` block. Two accepted forms::

        state: {voltage: 1, calcium: 1}                      # width only
        state: {voltage: {width: 1, integration: first_order, boundary: free}}

    A bare integer is shorthand for a ``first_order``, ``free``, recorded block of
    that width (the common non-spatial case: a scalar that integrates its own ODE and
    lives in no world box). The explicit-dict form overrides any field.
    """
    blocks = []
    for name, decl in state_decl.items():
        if isinstance(decl, dict):
            w = int(decl.get("width", 1))
            blocks.append(Block(
                name, w,
                role=decl.get("role"),
                integration=decl.get("integration", FIRST_ORDER),
                boundary=decl.get("boundary", BOUNDARY_FREE),
                record=bool(decl.get("record", True)),
            ))
        else:
            blocks.append(Block(name, int(decl), integration=FIRST_ORDER, boundary=BOUNDARY_FREE))
    return StateSchema(blocks)

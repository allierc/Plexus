"""The hierarchical graph container: sets + fields + operators (the contract layer).

This module defines the *vocabulary* every Plexus simulation is built from. It is
deliberately small and engine-agnostic: it says what a set, a field, and an
operator ARE, not how a run is stepped (that is `engine.py`). The shapes here are
exactly the ones the prototype validated across ~30 scenarios, promoted into the
package with the gaps `prototype/notes.md` surfaced closed.

Entities
--------
* **Level** -- a *set* of like nodes of one type (membrane particle / cytoplasm
  particle / nucleus / molecule / cell / population). Stored as flat batched
  tensors (a `state` matrix, an optional learnable `embedding`), never one object
  per node. Containment is a **parent map to another set**: `parent` (indices)
  `+ parent_name` (the containing set), the map `pi_{this -> parent}`. MANY sets
  may share one parent -- a cell contains membrane, cytoplasm, nucleus and
  molecule sets at once -- so the container is a *typed containment graph*, not a
  linear scale ladder, and a parent entity is a bundle of fibres (one per child
  set). An optional `edge_index` is a lateral relation. A Level is allocated at a
  fixed **buffer** size and a per-node **occupancy** `occ in [0,1]` marks the live
  subset, so a *cardinality-changing* operator (divide / die) lives inside a
  constant-shape contract.
* **Field** -- a continuum `f: Omega -> R^c` on its own discretization frame,
  bound to exactly one Level via `couples_to`. Fields do not nest. Subclasses
  supply the frame and the transfer kernels (`scatter`/`gather`/`step`).

Container
---------
* **Hierarchy** -- the ordered Levels + the flat set of Fields, plus the run-time
  scratch the engine attaches (config, rng, per-level accel accumulators, world
  geometry). It is an `nn.Module`, so operators and fields register as submodules
  and `.to(device)` moves the whole thing.

Operators
---------
* **Operator** -- a unit of dynamics. Proven contract::

      def __init__(self, params: dict, device="cpu"): ...
      def forward(self, H: Hierarchy, mask=None) -> dict[str, Tensor]: ...

  It returns a dict `{level_name: delta}` of *time-derivative contributions*. The
  engine sums same-level deltas into that level's accumulator and integrates each
  set once at the end of the tick (the order -- 1st-derivative velocity vs
  2nd-derivative acceleration -- comes from the operator's `PREDICTION`). An
  operator that changes *membership* or *relations* (structural / rewire) or that
  writes a field (exchange) mutates `H` in place and returns `{}` -- uniform with
  every other operator, so the engine never special-cases a kind.

  **The integration invariant.** An operator NEVER writes the engine-integrated
  state (`pos`/`vel`) directly -- a change to position/velocity must flow through
  the returned delta, which the engine integrates. Everything *else* is mutated in
  place and returns `{}`: relations `E` (`edge_index`), entities `|S|`
  (`occ`/buffers, structural only), fields `F` (`grid`), and auxiliary per-node
  control buffers (e.g. a slime agent's `heading`, which is not integrated state).
  So a dynamics operator that self-Euler-steps `pos` is a category error; the
  engine guards against it on the first frame.

  Six kinds, dispatched by the relation an operator acts on:

  | kind        | relation              | examples                              |
  |-------------|-----------------------|---------------------------------------|
  | `lateral`   | within a set          | signalling, boids, MPM particle forces|
  | `aggregate` | children -> parent    | particles -> cell centroid            |
  | `broadcast` | parent -> children    | cell decision -> particle force       |
  | `exchange`  | set <-> field / set   | P2G/G2P, secrete/sense, reaction      |
  | `structural`| changes |S_k| / membership | divide, duplicate, die           |
  | `rewire`    | rebuilds E (edge_index)   | membrane ring, neighbour graph    |

  `params` is the operator's spec line merged with its field refs (`to`/`from`);
  operators read their tunables from it in `__init__`. The `mask` is the live
  boolean selection of the operator's `at:` selector, recomputed every tick by
  the engine (so state-dependent selectors like `cell[done=0]` track the run).

Capability contract
-------------------
Operators declare what they need so the validator fails *before* a run, not deep
in a substep:

* `REQUIRES_PARAMS`      -- param keys the spec line must provide.
* `REQUIRES_TYPE_PROPS`  -- per-type node properties (resolved along the
  containment chain, e.g. `mpm` acts on particles but reads `youngs` off the
  parent cell's types).

`KIND` and `LEVEL` are stamped onto each class by the `@register_operator`
decorator; do not set them by hand.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# The recognised operator kinds (dispatch tags), grouped by what they change:
# the *dynamics* kinds (lateral / aggregate / broadcast / exchange) move a SET's
# STATE and return a delta; `field` is a FIELD's own self-dynamics (diffuse / decay /
# react / playback -- field->field, mutates the field, returns {}); `exchange` couples
# a SET to a field (set->field deposit, field->set sense/chemotaxis); `rewire` changes
# the RELATIONS (the edge set E); `structural` changes the ENTITIES (the node set |S|).
# Naming them lets the registry enumerate "what can change the state, the field, the
# relation, or the set".
KINDS = ("lateral", "aggregate", "broadcast", "exchange", "field", "structural", "rewire")


# --------------------------------------------------------------------------- #
#  Entities: Level (a set) and Field (a continuum)
# --------------------------------------------------------------------------- #
class Level(nn.Module):
    """A set `S_k` of like nodes at one scale, stored as flat batched tensors.

    Allocated at a fixed **buffer** size; `occ` (occupancy in [0,1], default all
    ones) marks the live subset so a cardinality-changing operator can wake or
    retire slots without resizing. `active` is the boolean live mask. Domain
    operators register their own per-node buffers on the Level (e.g. MPM's
    `mass`/`F`/`C`, a `node_type` for roles) -- the contract only mandates
    `state`, `occ`, and (for a contained set) `parent`.
    """

    def __init__(
        self,
        name: str,
        level: int,
        state: torch.Tensor,                        # [N, d]   dynamic state
        embedding: Optional[torch.Tensor] = None,   # [N, e]   learnable a_i
        parent: Optional[torch.Tensor] = None,      # [N]      index into the PARENT SET (the map pi_{this->parent})
        edge_index: Optional[torch.Tensor] = None,  # [2, E]   lateral relation
        occ: Optional[torch.Tensor] = None,         # [N]      occupancy in [0,1] (default ones)
        state_schema: Optional[dict] = None,        # {block: (c0,c1)} column semantics (from the entity registry)
        parent_name: Optional[str] = None,          # name of the set that contains this one (a containment EDGE)
        role: Optional[str] = None,                 # this set's role inside its parent (membrane / cytoplasm / nucleus...)
    ):
        super().__init__()
        self.name = name
        self.level = level                          # a depth hint only; containment is by `parent_name`, not by integer
        self.parent_name = parent_name              # which set contains this one (None for a top-level set)
        self.role = role
        self.state_schema = state_schema or {"pos": (0, 2), "vel": (2, 4)}
        N = state.shape[0]
        self.register_buffer("state", state)
        self.embedding = nn.Parameter(embedding) if embedding is not None else None
        self.register_buffer(
            "parent",
            parent if parent is not None else torch.empty(0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "edge_index",
            edge_index if edge_index is not None else torch.empty(2, 0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "occ",
            occ if occ is not None else torch.ones(N, device=state.device),
        )
        # lineage bookkeeping for cardinality-changing (structural) operators:
        #   birth   -- the occupancy baseline a node was born with (drives the
        #              "mass has doubled -> divide" trigger; caller sets/splits it).
        #   lineage -- the slot this node split/spawned from (-1 = founder).
        self.register_buffer("birth", self.occ.clone())
        self.register_buffer("lineage", torch.full((N,), -1, dtype=torch.long, device=state.device))

    @property
    def n(self) -> int:
        """Buffer size (allocated slots, live or dormant)."""
        return self.state.shape[0]

    def get(self, block: str) -> torch.Tensor:
        """A view of a named state block (e.g. 'pos', 'vel') per the schema."""
        a, b = self.state_schema[block]
        return self.state[:, a:b]

    @property
    def active(self) -> torch.Tensor:
        """Boolean mask of live nodes (`occ > 0`)."""
        return self.occ > 0

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def __repr__(self):
        return (f"Level({self.name!r}, level={self.level}, n={self.n}, "
                f"active={self.n_active}, d={self.state.shape[-1]})")

    # --- cardinality primitives (the engine-level structural machinery) ----- #
    # A `structural` operator (divide / duplicate / die) orchestrates these
    # instead of hand-scanning occupancy and hand-initialising buffers. They keep
    # tensor shapes constant (a fixed buffer); `occ` marks the live subset.
    def per_node_buffers(self):
        """Yield (name, tensor) for every registered buffer indexed by node, so a
        structural op touches them all uniformly. Excludes the relation
        `edge_index` (shaped [2, E], not per-node) and the immutable `birth` /
        `lineage` / `occ` which `spawn`/`kill` manage explicitly."""
        for name, b in self.named_buffers(recurse=False):
            if name in ("edge_index", "occ", "birth", "lineage"):
                continue
            if b.dim() >= 1 and b.shape[0] == self.n:
                yield name, b

    def free_slots(self, k: int) -> torch.Tensor:
        """Up to `k` dormant slot indices (`occ == 0`), fewer if the buffer is
        nearly full. An empty result means the buffer is exhausted -- the op
        should stop dividing rather than error (the proven 'buffer full' guard)."""
        dormant = (self.occ == 0).nonzero(as_tuple=True)[0]
        return dormant[:k]

    def spawn(self, src_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wake free slots as clones of `src_idx`: copy EVERY per-node buffer
        (state, parent, node_type, and any domain buffer like MPM mass/F/C/mu/la)
        from the source nodes, set `occ = 1`, and record lineage. This is the
        init-safe fix for the trap in notes.md -- a structural op no longer has to
        remember to initialise each operator's per-node state by hand; it inherits
        the parent's. The caller then overrides only what must differ (e.g. nudge
        position, zero velocity, reset deformation F). Returns (new_idx, src_used)
        truncated to the number of free slots actually available."""
        dst = self.free_slots(int(src_idx.numel()))
        src = src_idx[: dst.numel()]
        for _, b in self.per_node_buffers():
            b[dst] = b[src]
        self.occ[dst] = 1.0
        self.birth[dst] = self.birth[src]          # default: inherit baseline (divide overrides per-daughter)
        self.lineage[dst] = src
        return dst, src

    def kill(self, idx: torch.Tensor, park: Optional[torch.Tensor] = None) -> None:
        """Retire live slots (the efflux/death boundary): `occ = 0`, drop physical
        `mass` to 0 if present so the node stops contributing to MPM scatter /
        mass-weighted aggregation, and optionally park its position off-domain at
        `park` so it is neither drawn nor sensed."""
        self.occ[idx] = 0.0
        if "mass" in dict(self.named_buffers(recurse=False)):
            self.get_buffer("mass")[idx] = 0.0
        if park is not None:
            self.state[idx, : park.numel()] = park


class Field(nn.Module):
    """A continuum `f: Omega -> R^c` on its own frame, bound to one Level.

    Subclasses (registered via `@register_field`) supply the discretization and
    the transfer kernels. `scatter` writes object state into the field; `gather`
    reads the field back onto objects (a delta); `step` advances the field's own
    PDE one tick (Laplacian diffusion + decay). P2G/G2P, secrete/sense and
    morphogen sampling are all (scatter, gather) pairs over a Field.
    """

    COUPLES_TO: Optional[str] = None    # stamped by @register_field
    FRAME: Optional[str] = None

    def __init__(self, name: str, couples_to: Optional[str] = None):
        super().__init__()
        self.name = name
        # the Level whose types this field's channels mirror (slime: one channel per
        # species). Optional: a prescribed field (e.g. a video) binds to no set --
        # its coupling is established by the operator that reads it (`from:`), not here.
        self.couples_to = couples_to

    def scatter(self, level: "Level") -> None:          # object -> field
        raise NotImplementedError

    def gather(self, level: "Level") -> torch.Tensor:   # field -> object (delta)
        raise NotImplementedError

    def step(self) -> None:                             # advance the field's own PDE
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Container: Hierarchy
# --------------------------------------------------------------------------- #
class Hierarchy(nn.Module):
    """Ordered Levels (bottom-up) + a flat set of Fields, plus run-time scratch.

    The engine attaches per-run state as plain attributes (an `nn.Module` allows
    it): `config` (the validated Spec), `rng` (a seeded generator for
    determinism), world geometry (`world_width`, `periodic`), and the per-level
    **delta accumulators** that realise the integration model.

    Integration model (the contract every operator/engine honours): operators are
    pure and return per-level deltas `Δ`; the engine sums same-level deltas here
    and integrates each set once at the end of the tick. Use
    `zero_delta`/`add_delta`/`delta` so operators and the engine share one
    convention. The delta is a velocity or an acceleration depending on the
    operator's `PREDICTION`; it is not necessarily an acceleration, hence `delta`.
    """

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleDict()         # name -> Level (insertion order = bottom-up)
        self.fields = nn.ModuleDict()         # name -> Field
        self._delta: dict[str, torch.Tensor] = {}

    # --- structure -------------------------------------------------------- #
    def add_level(self, lvl: Level) -> Level:
        self.levels[lvl.name] = lvl
        return lvl

    def add_field(self, fld: Field) -> Field:
        self.fields[fld.name] = fld
        return fld

    def level(self, name: str) -> Level:
        return self.levels[name]

    def field(self, name: str) -> Field:
        return self.fields[name]

    # --- the typed containment graph (parent maps by name, not a linear ladder) -- #
    def children(self, parent_name: str) -> list[str]:
        """The child sets contained by `parent_name`. A parent may have MANY
        children of different roles (membrane / cytoplasm / nucleus / molecule),
        so a parent entity is a *bundle of fibres*, one per child set."""
        return [n for n, l in self.levels.items() if getattr(l, "parent_name", None) == parent_name]

    def parent_of(self, name: str) -> Optional[str]:
        """The set that contains `name` (None for a top-level set)."""
        return getattr(self.levels[name], "parent_name", None)

    # --- per-level delta accumulators (the integration scratch) ----------- #
    def zero_delta(self, dim: int = 2) -> None:
        """Reset every level's delta accumulator to zeros (called once per tick)."""
        dev = next(iter(self.levels.values())).state.device
        self._delta = {name: torch.zeros(l.n, dim, device=dev)
                       for name, l in self.levels.items()}

    def add_delta(self, level_name: str, delta: torch.Tensor) -> None:
        """Add an operator's returned delta into its level's accumulator."""
        if level_name not in self._delta:
            self._delta[level_name] = delta.clone()
        else:
            self._delta[level_name] = self._delta[level_name] + delta

    def delta(self, level_name: str) -> torch.Tensor:
        """The accumulated delta for a level (zeros if nothing wrote it)."""
        if level_name not in self._delta:
            lvl = self.levels[level_name]
            self._delta[level_name] = torch.zeros(lvl.n, 2, device=lvl.state.device)
        return self._delta[level_name]


# --------------------------------------------------------------------------- #
#  Operators: a unit of dynamics, dispatched by relation
# --------------------------------------------------------------------------- #
class Operator(nn.Module):
    """Base operator. Proven contract: `__init__(params, device)` reads tunables
    from the spec line; `forward(H, mask) -> {level_name: delta}` returns
    per-level time-derivative contributions (or `{}` if it mutates `H`).

    `KIND` and `LEVEL` are stamped by `@register_operator`. `REQUIRES_PARAMS` /
    `REQUIRES_TYPE_PROPS` are the capability contract the validator checks.

    Subclasses below are thin semantic tags (one per kind); an operator inherits
    the one matching the relation it acts on so its kind reads from the class.
    Plain torch is the norm (index_add / pairwise); an operator that wants
    message-passing machinery can mix in a PyG `MessagePassing` when those ports
    land (signalling, interaction) -- the contract does not require it.
    """

    KIND: Optional[str] = None
    LEVEL: Optional[str] = None
    # What this operator's returned delta IS, so the engine knows how to integrate
    # it: "first_derivative" (the delta is a velocity -> x += dt*delta) or
    # "second_derivative" (an acceleration -> v += dt*delta; x += dt*v). None for
    # operators that emit no force (rewire/structural/exchange-into-field).
    PREDICTION: Optional[str] = None
    REQUIRES_PARAMS: list = []          # param keys this operator must be given
    REQUIRES_TYPE_PROPS: list = []      # per-type node properties it reads (e.g. "youngs")
    # The integration invariant is enforced per-operator on frame 0 (see engine.run):
    # an operator that legitimately writes a set's `state` -- a structural op (divide/
    # die rewrites the buffer) or a derived-state readout (aggregate centroid) -- sets
    # this True to opt out of the guard. Everything else must NOT touch pos/vel.
    MAY_MUTATE_STATE: bool = False
    # World-model ledger metadata (spec -> mechanistic language; see plexus.tex Part IV).
    # Declarative, optional: what mechanism this operator embodies, what morphologies it
    # tends to produce, and what each tunable param *means* mechanistically.
    MECHANISM_TAGS: list = []           # e.g. ["long_range_attraction", "coarsening"]
    MORPHOLOGY_PRIOR: list = []         # e.g. ["single_cluster", "filaments"]
    PARAM_ROLES: dict = {}              # e.g. {"sigma": "interaction_length", "gain": "field_sensitivity"}

    def __init__(self, params: Optional[dict] = None, device: str = "cpu"):
        super().__init__()
        self.params = params or {}
        self.device = device

    def forward(self, H: Hierarchy, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class Lateral(Operator):
    """Within-set dynamics on `E subset S_k x S_k` (interaction + discrete Laplacian)."""
    KIND = "lateral"


class Aggregate(Operator):
    """children -> parent reduction over the partition `Level.parent` (occupancy-weighted)."""
    KIND = "aggregate"


class Broadcast(Operator):
    """parent -> children lift along the partition `Level.parent`."""
    KIND = "broadcast"


class Exchange(Operator):
    """set <-> field (or set <-> set) scatter/gather. Unifies P2G/G2P, secrete/sense,
    chemotaxis. Mutates a Field (or auxiliary set state) in place; returns `{}` -- OR,
    for a field->set coupling, returns a velocity/accel delta the engine integrates."""
    KIND = "exchange"


class FieldUpdate(Operator):
    """A field's OWN dynamics: field -> field (diffuse, decay, react, or playback of
    prescribed data). Mutates the Field's grid in place and returns `{}`. Distinct from
    `exchange`, which couples a SET to a field; here no set is involved."""
    KIND = "field"


class Structural(Operator):
    """Changes cardinality / membership (divide, duplicate, die) on a fixed buffer
    via occupancy. May emit per-node deltas during a gradual transition (e.g.
    mitosis) and only relabel membership at completion; returns `{}` otherwise."""
    KIND = "structural"
    MAY_MUTATE_STATE = True             # waking/retiring slots rewrites the state buffer


class Rewire(Operator):
    """Rebuilds a relation `E` (`edge_index`) -- e.g. the membrane ring or a
    neighbour graph -- each tick, so the relation tracks growth/division. Stores
    the new edges on `H`; emits no delta."""
    KIND = "rewire"

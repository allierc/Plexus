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
  2nd-derivative acceleration -- comes from the operator's `EMIT`). An
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

  Eight kinds. Seven are DYNAMICS, dispatched by the relation an operator acts
  on; `seed` is not one of the seven -- it is a separate LIFECYCLE PHASE (see
  `Seed` below and `docs/` for the full model: `x_0 = S(theta_S)`, then
  `x_{t+1} = Phi(x_t; theta_D)`). A seed operator never appears in a model's
  `schedule:` and is never dispatched by the per-tick loop; it runs once, via
  `engine.seed()`, before the dynamics loop starts.

  | kind        | relation              | examples                              |
  |-------------|-----------------------|---------------------------------------|
  | `lateral`   | within a set          | signalling, boids, MPM particle forces|
  | `aggregate` | children -> parent    | particles -> cell centroid            |
  | `broadcast` | parent -> children    | cell decision -> particle force       |
  | `exchange`  | set <-> field / set   | P2G/G2P, secrete/sense, reaction      |
  | `field`     | field -> field        | diffuse, decay, react, playback       |
  | `structural`| changes |S_k| / membership | divide, duplicate, die           |
  | `rewire`    | rebuilds E (edge_index)   | membrane ring, neighbour graph    |
  | `seed`      | writes x_0, once, before dynamics | mesh/tissue placement, field IC |

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

from plexus.models.state import StateSchema


# The recognised operator kinds (dispatch tags). SEVEN are DYNAMICS kinds, grouped by
# what they change: `lateral` / `aggregate` / `broadcast` / `exchange` move a SET's
# STATE and return a delta; `field` is a FIELD's own self-dynamics (diffuse / decay /
# react / playback -- field->field, mutates the field, returns {}); `exchange` couples
# a SET to a field (set->field deposit, field->set sense/chemotaxis); `rewire` changes
# the RELATIONS (the edge set E); `structural` changes the ENTITIES (the node set |S|)
# DURING the run (divide, extrude, apoptosis). The eighth, `seed`, is not a dynamics
# kind at all: it WRITES THE INITIAL STATE, once, before any dynamics kind runs -- see
# `Seed` below, and `docs/` for the two-phase model this implements:
# `x_0 = S(theta_S)`, then `x_{t+1} = Phi(x_t; theta_D)`.
#
# WHY `seed` IS ITS OWN KIND, SEPARATE FROM `structural`, NOT A FLAVOUR OF IT. Added 6
# August; the argument is a measured failure rather than a taxonomy preference.
#
# `structural` used to conflate two acts that differ in the one property that matters:
# establishing the initial state (once) and changing membership as the simulation runs
# (divide, extrude, apoptosis -- throughout). Both mutate in place and return nothing, so
# the purity contract could not tell them apart, and neither could the engine.
#
# The cost of that was `cell_rd_seed mode: tip`. It re-stamped an activation cap on the
# outermost cell EVERY frame, which makes it a moving boundary condition rather than an
# initial condition -- and nothing in the language could say so. Three hand-written parents
# carried `before_frame: 3`; the whole campaign lineage carried nothing; both were equally
# legal specs. The measured consequence: the chemistry was overwritten every tick, so
# `cell_chem_from_shape` could accumulate nothing, and 8 same-seed `beta` edits across 13 rounds
# moved the trajectory by EXACTLY zero while each was recorded as a refuted hypothesis.
#
# A `seed` operator runs exactly once, in `engine.seed()`, before the dynamics `schedule:`
# starts -- not "gated to the opening frames of the schedule", a genuinely separate
# lifecycle phase a spec cannot express seed operators inside of. A per-frame initialiser
# is UNEXPRESSIBLE rather than merely discouraged. This is the same move as EMIT making the
# engine own integration: a discipline every spec had to remember becomes a guarantee the
# language provides.
KINDS = ("lateral", "aggregate", "broadcast", "exchange", "field", "structural", "rewire",
         "seed")


# The recognised temporal-integration states (Axis A: how a SET moves in time), shared by
# the class attribute `Operator.EMIT` and the spec `emit:` param -- ONE vocabulary,
# no translation table:
#   velocity / acceleration -> engine-integrated (per-set order agreement enforced);
#   mpm_acceleration        -> an acceleration routed to the MPM substep (p2g reads it as
#                              a_ext), NOT engine-integrated -- same order as acceleration,
#                              differs only in routing.
# `None` (the default) means the operator emits no set delta at all (rewire / structural /
# field / exchange-into-field). Spatial field math (grad/laplacian/sample, Axis B) is a
# SEPARATE axis and lives on the Field, not here.
EMITS = ("velocity", "acceleration", "mpm_acceleration")


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
        depth: int = 0,                             # hierarchy DEPTH (0 = leaf); a scale hint only.
                                                    # (Was `level` -- renamed to free the word "level"
                                                    #  from the set/depth overload; `level=` still accepted.)
        state: torch.Tensor = None,                 # [N, d]   dynamic state
        embedding: Optional[torch.Tensor] = None,   # [N, e]   learnable a_i
        parent: Optional[torch.Tensor] = None,      # [N]      index into the PARENT SET (the map pi_{this->parent})
        edge_index: Optional[torch.Tensor] = None,  # [2, E]   lateral relation
        occ: Optional[torch.Tensor] = None,         # [N]      occupancy in [0,1] (default ones)
        state_schema=None,                          # StateSchema (the fifth primitive), or a legacy {block:(c0,c1)} dict
        parent_name: Optional[str] = None,          # name of the set that contains this one (a containment EDGE)
        role: Optional[str] = None,                 # this set's role inside its parent (membrane / cytoplasm / nucleus...)
        pre: Optional[torch.Tensor] = None,         # [E] INCIDENCE map to the pre-endpoint set (edge-set only)
        post: Optional[torch.Tensor] = None,        # [E] INCIDENCE map to the post-endpoint set (edge-set only)
        pre_name: Optional[str] = None,             # the set `pre` indexes into (e.g. neuron)
        post_name: Optional[str] = None,            # the set `post` indexes into (e.g. neuron)
    ):
        super().__init__()
        self.name = name
        # `depth` is a scale hint only; containment is by `parent_name`, not by integer.
        self.depth = depth
        self.parent_name = parent_name              # which set contains this one (None for a top-level set)
        self.role = role
        # Incidence maps (the second kind of map): an EDGE-SET's elements are connections,
        # sent to their endpoints by `pre`/`post` -- index buffers of the same shape as
        # `parent`, but answering "whom do I connect" not "who owns me". Empty for an
        # ordinary set, so this is inert for every existing (spatial) spec.
        self.pre_name = pre_name
        self.post_name = post_name
        # State is first-class: normalize a legacy {block:(c0,c1)} dict into a StateSchema
        # (the shim). A StateSchema is still dict-indexable (schema['pos'] == (c0,c1)), so
        # `get('pos')` and every legacy call site are unchanged.
        self.state_schema = StateSchema.normalize(state_schema or {"pos": (0, 2), "vel": (2, 4)})
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
            "pre",
            pre if pre is not None else torch.empty(0, dtype=torch.long, device=state.device),
        )
        self.register_buffer(
            "post",
            post if post is not None else torch.empty(0, dtype=torch.long, device=state.device),
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
    def is_edge_set(self) -> bool:
        """True if this set carries incidence maps -- its elements are connections
        (a synapse/junction/vessel), sent to endpoints by `pre`/`post`."""
        return self.pre_name is not None or self.post_name is not None

    def incidence(self, role: str) -> torch.Tensor:
        """The incidence-map index buffer for `role` ('pre' or 'post')."""
        return getattr(self, role)

    def incidence_name(self, role: str) -> str:
        """The endpoint set name for `role`."""
        return getattr(self, f"{role}_name")

    @property
    def active(self) -> torch.Tensor:
        """Boolean mask of live nodes (`occ > 0`)."""
        return self.occ > 0

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def __repr__(self):
        return (f"Level({self.name!r}, depth={self.depth}, n={self.n}, "
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

    # --- Axis B: spatial field math on the grid (same category as `pix`) ----- #
    def grad_at(self, pos, channel=0, periodic=False):
        """gradient of a grid channel, bilinear-sampled at world positions `pos`
        [N, D] -> [N, D]. Boundary-aware central difference: wrap under a periodic
        world, replicate under a wall (a wall edge has no artificial gradient).
        `channel=None` sums all channels (follow any trail). 2D grid fields
        (`self.grid` is [C, nx, ny]); operators that use it declare
        SUPPORTED_DIMS=[2] -- N-D is a follow-up. This is the gradient the
        chemotactic operators used to compute inline; it lives on the field so
        spatial (Axis B) math is not duplicated across operators."""
        import torch.nn.functional as F
        g = self.grid.sum(0) if channel is None else self.grid[int(channel)]      # [nx, ny]
        W = self.width
        mode = "circular" if periodic else "replicate"
        gp = F.pad(g[None, None], (1, 1, 1, 1), mode=mode)[0, 0]                   # [nx+2, ny+2]
        gx = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5 * (self.nx / W)                  # d/dx per world unit
        gy = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5 * self.ny                        # d/dy per world unit
        grad = torch.stack([gx, gy], 0)[None]                                     # [1, 2, nx, ny]
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]                            # grid_sample: x=ny, y=nx
        return F.grid_sample(grad, grid, mode="bilinear", padding_mode="border",
                             align_corners=True)[0, :, 0].t()                     # [N, D]

    def sample(self, pos, channel=None):
        """bilinear read of the grid at world positions `pos` [N, D]. `channel=None`
        -> [N, C] (all channels); `channel=k` -> [N] (that one channel). Border
        padding; same coordinate convention as `grad_at`. 2D grid fields. Replaces
        the bilinear grid_sample block the field-coupled operators used inline."""
        import torch.nn.functional as F
        data = self.grid if channel is None else self.grid[int(channel):int(channel) + 1]   # [C, nx, ny]
        W = self.width
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        g = torch.stack([gyn, gxn], -1)[None, None]
        out = F.grid_sample(data[None], g, mode="bilinear", padding_mode="border",
                            align_corners=True)[0, :, 0].t()                      # [N, C']
        return out[:, 0] if channel is not None else out


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
    operator's `EMIT`; it is not necessarily an acceleration, hence `delta`.
    """

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleDict()         # name -> Level (insertion order = bottom-up)
        self.fields = nn.ModuleDict()         # name -> Field
        self._delta: dict[str, torch.Tensor] = {}          # per-set COORDINATE-block delta (MPM reads this via delta())
        self._delta_blocks: dict[str, dict] = {}           # per-set EXTRA first-order block deltas: set -> {block: tensor}
        self.dim = 2                          # spatial dimensions (set by the engine from the spec)

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

    # --- incidence maps: gather/scatter along a named map (pre/post), not containment -- #
    def gather(self, edge_set: str, role: str, block: str) -> torch.Tensor:
        """Gather an endpoint set's `block` state onto each edge along its `role`
        incidence map -- a lift along an incidence map. Returns `[E, w]`: the pre/post
        endpoint's block value per edge (e.g. `v_pre`, `v_post` for a synapse_ode)."""
        es = self.level(edge_set)
        idx = es.incidence(role)                          # [E] endpoint index per edge
        ep = self.level(es.incidence_name(role))
        return ep.get(block)[idx]

    def scatter_along(self, edge_set: str, role: str, values: torch.Tensor) -> torch.Tensor:
        """Sum per-edge `values` `[E, w]` onto the endpoint set along the `role`
        incidence map -- an Aggregate along an incidence map (e.g. synaptic current onto
        the post neuron). Occupancy-weighted, so dormant edges contribute nothing.
        Returns `[N_endpoint, w]`."""
        es = self.level(edge_set)
        idx = es.incidence(role)
        ep = self.level(es.incidence_name(role))
        out = torch.zeros(ep.n, values.shape[-1], device=values.device)
        out.index_add_(0, idx, values * es.occ[:, None])
        return out

    # --- per-level delta accumulators (the integration scratch) ----------- #
    def _delta_dim(self, lvl: "Level") -> int:
        """Width of a level's delta = its integrated coordinate block's width. For a
        spatial set that is `pos` (== self.dim, so byte-identical to the old
        `H.dim`-sized accumulator); for a neuron it is voltage's width, etc."""
        coord = lvl.state_schema.coordinate
        return coord.width if coord is not None else self.dim

    def zero_delta(self, dim: int = None) -> None:
        """Reset every level's delta accumulator to zeros (called once per tick). Each
        level's delta is sized to its coordinate block (pos for a spatial set, voltage
        for a neuron), not a global spatial dim."""
        dev = next(iter(self.levels.values())).state.device
        self._delta = {name: torch.zeros(l.n, self._delta_dim(l) if dim is None else dim, device=dev)
                       for name, l in self.levels.items()}
        self._delta_blocks = {}                            # extra (non-coordinate) block deltas, filled lazily

    def add_delta(self, level_name: str, delta: torch.Tensor, block: str = None) -> None:
        """Add an operator's returned delta into its level's accumulator. `block` is the
        state block the delta integrates into (an operator's `INTEGRAND`); `None` or the
        coordinate block -> the COORDINATE accumulator (unchanged, what MPM reads via
        `delta()`); any OTHER dynamical block -> its own accumulator, so one set can carry
        several independently-integrated blocks (pos + chem + a0)."""
        coord = self.levels[level_name].state_schema.coordinate
        coord_name = coord.name if coord is not None else None
        if block is None or block == coord_name:
            if level_name not in self._delta:
                self._delta[level_name] = delta.clone()
            else:
                self._delta[level_name] = self._delta[level_name] + delta
        else:
            db = self._delta_blocks.setdefault(level_name, {})
            db[block] = delta.clone() if block not in db else db[block] + delta

    def renumber_set(self, level_name: str, keep, n_new: int | None = None) -> bool:
        """Permute a set's rows through `keep` (new index -> old index). Returns True if it acted.

        WHY THE ENGINE OWNS THIS. Removal is the first operation in this engine that MOVES A ROW --
        `cell_divide` appends, so every existing cell keeps its index, and nothing before apoptosis
        and face-dropping T1s ever renumbered anything. When they arrived, each wrote its own
        renumber, and each had to remember every store the engine keeps per set: the state, the
        occupancy, the coordinate delta accumulator, and the extra first-order block deltas.

        THE ONE THEY BOTH GOT WRONG is the last. `_delta_blocks` is keyed by LEVEL NAME
        (`add_delta`: `self._delta_blocks.setdefault(level_name, {})`), and both call sites guard on
        `isinstance(key, tuple) and key[0] == cell_set` -- always False, so the extra blocks are
        never permuted. It is inert only because `chem` happens to be the cell set's COORDINATE
        block today and therefore travels in `_delta`; the moment a model declares a second
        `first_order` block on that set, its deltas scramble on every death.

        AND THE COORDINATE ONE IS NOT HYPOTHETICAL. The engine zeroes the accumulator once per TICK
        and integrates at the END of the schedule, so the chemistry deposits early and the engine
        applies last; an operator that renumbers in between leaves every delta pointing at a
        different cell. Measured before it was fixed: the activator reached -0.1529 at frame 50 on
        every mode that killed anything, while the no-death control never left 0.0000.

        THE TAIL OF `state` IS LEFT AS IT WAS, deliberately -- the call sites clone, write `[:n]`,
        and leave the rest stale, and `occ` is what says those rows are not there. Zeroing it would
        be a different behaviour, and the promotion is gated on bit-equality.
        """
        import numpy as _np
        lvl = self.levels.get(level_name) if hasattr(self.levels, "get") else None
        if lvl is None or getattr(lvl, "state", None) is None:
            return False
        n = int(len(keep) if n_new is None else n_new)
        idx = torch.as_tensor(_np.asarray(keep), dtype=torch.long, device=lvl.state.device)
        if lvl.state.shape[0] < n:
            return False
        st = lvl.state.clone()
        st[:n] = lvl.state[idx]
        lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(lvl.state.shape[0], device=lvl.state.device)
            occ[:n] = 1.0
            lvl.occ = occ
        d = self._delta.get(level_name)
        if d is not None:
            k = idx.to(d.device)
            d[:n] = d[k]
            d[n:] = 0.0
        for _b, t in (self._delta_blocks.get(level_name) or {}).items():
            if t is None:
                continue
            k = idx.to(t.device)
            t[:n] = t[k]
            t[n:] = 0.0
        return True

    def block_deltas(self, level_name: str) -> dict:
        """Accumulated deltas for a set's EXTRA (non-coordinate) dynamical blocks: {block: tensor}."""
        return self._delta_blocks.get(level_name, {})

    def delta(self, level_name: str) -> torch.Tensor:
        """The accumulated delta for a level (zeros if nothing wrote it)."""
        if level_name not in self._delta:
            lvl = self.levels[level_name]
            self._delta[level_name] = torch.zeros(lvl.n, self._delta_dim(lvl), device=lvl.state.device)
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
    # The biological SET this operator acts on -- its primary INPUT set. Stamped by
    # `@register_operator(set=...)`. (Was `LEVEL`, a misnomer: `set="cell"` names a
    # *set*, never a hierarchy depth or a runtime Level. See `signature()`.)
    SET: Optional[str] = None
    # --- typed signature (Plexus2 sec. 2.1): an operator is a typed MORPHISM between
    # sets. Declarative metadata, like MECHANISM_TAGS -- the validator, atlas and docs
    # read it; the engine does not. Maps are PART of the signature: Plexus2 folds maps
    # INTO the operator (there is no standalone Map primitive), so an operator names the
    # maps it traverses here rather than the language carrying a separate `M`. Defaults
    # describe a single-set morphism on SET; fill INPUTS/OUTPUTS/READS/WRITES/MAPS on
    # operators that gather/scatter along maps (signal, aggregate, deposit, ...).
    INPUTS: list = []                   # input sets (empty => [SET])
    OUTPUTS: list = []                  # output sets (empty => [SET])
    READS: list = []                    # state blocks consumed, by name, e.g. ["voltage", "w"]
    WRITES: list = []                   # state blocks produced, by name, e.g. ["voltage"]
    MAPS: list = []                     # named maps traversed, e.g. ["pre", "post"] or ["parent"]
    # Which state BLOCK this operator's returned delta integrates into. `None` (default) =>
    # the set's coordinate block (the common case; unchanged). Set it to a NON-coordinate
    # dynamical block (e.g. "chem", "a0") so one set can carry several independently-
    # integrated blocks at once -- pos integrated by the mechanics, chem by the RD, a0 by growth.
    INTEGRAND: Optional[str] = None
    # What this operator's returned delta IS, so the engine knows how to integrate it.
    # One vocabulary (Axis A; = the spec `emit:` value on merged operators), see EMITS:
    #   "velocity"         -- delta is dx/dt     -> engine: x += dt*delta
    #   "acceleration"     -- delta is d2x/dt2   -> engine: v += dt*delta; x += dt*v
    #   "mpm_acceleration" -- a d2x/dt2 body accel routed to the MPM substep (a_ext), NOT
    #                         engine-integrated; same order as acceleration, differs in routing.
    # None: emits no set delta at all (rewire / structural / field / exchange-into-field).
    EMIT: Optional[str] = None
    REQUIRES_PARAMS: list = []          # param keys this operator must be given
    REQUIRES_TYPE_PROPS: list = []      # per-type node properties it reads (e.g. "youngs")
    # Spatial dimensions this operator supports. The language/container is dimension-
    # generic (general: dim: D, world: [w0..w_{D-1}]); an operator declares which D it
    # implements so the schema rejects an incompatible spec BEFORE the run. Default 2D
    # only; a dimension-generic operator (reads D = pos.shape[-1], no hard-coded 2) sets
    # [2, 3] (or more). The 2D-specific kernels (MLS-MPM) stay [2].
    SUPPORTED_DIMS: list = [2]
    # A per-IMPLEMENTATION capability (Plexus2 sec. 5): whether gradients flow through this
    # implementation's forward. The contract is fixed; two implementations of it may differ
    # here (a differentiable Neural-ODE vs a black-box/non-diff solver), so an inverse loop
    # can filter `get_contract(name).capabilities()` for the differentiable ones. Default
    # True -- the pure-torch operators are differentiable.
    DIFFERENTIABLE: bool = True
    # The integration invariant is enforced per-operator on frame 0 (see engine.run):
    # an operator that legitimately writes a set's `state` -- a structural op (divide/
    # die rewrites the buffer) or a derived-state readout (aggregate centroid) -- sets
    # this True to opt out of the guard. Everything else must NOT touch pos/vel.
    MAY_MUTATE_INTEGRATED_STATE: bool = False
    # World-model ledger metadata (spec -> mechanistic language; see plexus.tex Part IV).
    # Declarative, optional: what mechanism this operator embodies, what morphologies it
    # tends to produce, and what each tunable param *means* mechanistically.
    MECHANISM_TAGS: list = []           # e.g. ["long_range_attraction", "coarsening"]
    PARAM_ROLES: dict = {}              # e.g. {"sigma": "interaction_length", "gain": "field_sensitivity"}

    # --- the transitional fence (plexus.tex Part IV) ----------------------- #
    # A *normal* operator obeys the whole contract: one concern, returns a delta,
    # never integrates, never resizes. A *transitional* operator is the explicit
    # exception -- it wraps a mature, validated multi-mechanism subsystem (e.g. the
    # MLS-MPM solver: P2G + grid solve + stress + plasticity + boundary + G2P) that
    # is too costly to decompose immediately. It is allowed to break the ideal
    # architecture ONLY when the violation is fenced: explicit (`TRANSITIONAL=True`),
    # enumerated (`ARCHITECTURAL_DEBT`), isolated, and scheduled for decomposition.
    # The fence stops the exception from spreading -- an agent enumerates the
    # `TRANSITIONAL` operators and treats everything else as ideal.
    TRANSITIONAL: bool = False
    ARCHITECTURAL_DEBT: list = []       # human-readable list of the contract clauses it breaks
    # Engine-provisioning requirements (distinct from REQUIRES_PARAMS / REQUIRES_TYPE_PROPS,
    # which are spec-time): per-node buffers the operator reads off its Level *besides*
    # `state`, and Hierarchy run-time scratch the engine must attach. Declared so the
    # dependency is visible in the contract, not hidden inside a substep. Checked at
    # forward() entry (these are provisioned by the engine/entity build, not by a spec line,
    # so the schema cannot see them -- a missing one must fail loudly, with a precise message).
    REQUIRES_BUFFERS: list = []         # per-node Level buffers, e.g. ["C", "F", "mass", "mu", "la", "p_vol"]
    REQUIRES_HSTATE: list = []          # Hierarchy scratch, e.g. ["cell_accel"] (parent->child broadcast)

    def __init__(self, params: Optional[dict] = None, device: str = "cpu"):
        super().__init__()
        self.params = params or {}
        self.device = device

    def tunable(self, value, default=None):
        """Read a knob so that a LEARNABLE one survives construction.

        Every operator in the library writes `self.rate = float(params.get("rate", 0.0))`, and
        that `float()` is where the inverse half of Plexus dies: a spec may hand in a tensor with
        `requires_grad`, and the constructor silently casts the tape away before `forward` ever
        sees it. Two independent instruments (`operators/diff/audit.py` past the constructor,
        `operators/diff/certify.py` through a real spec) disagreed on exactly this set of knobs,
        and the gap between them WAS the coercion.

        The reference states the rule outright -- *store as an array anything you want to learn* --
        so: pass a tensor through untouched (moved to the device, tape intact), and coerce
        anything else to a float exactly as before. A forward run cannot tell the difference,
        which is why this is safe to apply to an operator whose numbers are already validated.

            self.epsilon = self.tunable(params.get("epsilon"), 1.0)
        """
        v = default if value is None else value
        if isinstance(v, torch.Tensor):
            return v.to(self.device)
        return float(v)

    def forward(self, H: Hierarchy, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def signature(cls) -> dict:
        """The operator's typed signature (Plexus2 sec. 2.1): the sets it maps
        between, the state it reads/writes, and the maps it traverses. `inputs`/
        `outputs` default to the single acting set `SET` when not declared. This is
        the machine-readable form of "operator as typed morphism between sets"; the
        atlas and (later) the OperatorContract read it -- the engine never does."""
        one = [cls.SET] if cls.SET else []
        return {
            "inputs":  list(cls.INPUTS) or one,
            "outputs": list(cls.OUTPUTS) or one,
            "reads":   list(cls.READS),
            "writes":  list(cls.WRITES),
            "maps":    list(cls.MAPS),
        }


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
    via occupancy, DURING the dynamics phase. May emit per-node deltas during a
    gradual transition (e.g. mitosis) and only relabel membership at completion;
    returns `{}` otherwise. A dynamics kind: appears in `schedule:`, runs every
    tick it is scheduled for, for the length of the run."""
    KIND = "structural"
    MAY_MUTATE_INTEGRATED_STATE = True             # waking/retiring slots rewrites the state buffer


class Seed(Operator):
    """WRITES THE INITIAL STATE, once, before any dynamics runs. NOT a dynamics
    kind and NOT a `Structural` -- the two were merged once (Seed subclassed
    Structural) and that inheritance is exactly the taxonomy error this class
    now exists to rule out.

    A `Seed` and a `Structural` op both write a buffer directly rather than
    returning a delta, so the low-level mechanics MAY overlap (both can call
    `Level.spawn`, both set `MAY_MUTATE_INTEGRATED_STATE = True`) -- but they do
    not share a semantic ancestor, because they differ in the one property that
    matters: WHEN, not HOW. `agent_divide` and `apoptosis` (`Structural`) change
    membership throughout a run; a `Seed` establishes the state the run starts
    from and then never runs again. Conflating the two via inheritance is what
    let `cell_rd_seed mode: tip` re-stamp an activation cap every frame -- a
    `Structural` subclass has no way to say "and never again" -- annihilating
    every operator writing to the same channel.

    A `Seed` operator is not scheduled at all: it runs exactly once, in
    `engine.seed()`, before `engine.run()`'s dynamics loop starts. It never
    appears in a model's `schedule:`, and the schema rejects a spec where it
    does. The guarantee is the language's to keep, not each spec's to remember.
    """
    KIND = "seed"
    MAY_MUTATE_INTEGRATED_STATE = True             # establishing x_0 IS writing the state buffer


class Rewire(Operator):
    """Rebuilds a relation `E` (`edge_index`) -- e.g. the membrane ring or a
    neighbour graph -- each tick, so the relation tracks growth/division. Stores
    the new edges on `H`; emits no delta."""
    KIND = "rewire"

"""The hierarchical graph container: sets + fields + operators.

Design summary (see docs/tissue_graph.pdf):

  * A **Level** is a *set* S_k of like nodes at one scale, stored as batched
    tensors (state + learnable embedding) plus a `parent` map (the partition
    into the level above) and an optional `edge_index` (a lateral relation).
  * A **Field** is a continuum f: Omega -> R^c on its own discretization frame,
    bound to exactly one level via `couples_to`.
  * A **Hierarchy** holds the ordered levels and the set of fields.
  * An **Operator** is a GNN that returns a *time-derivative contribution*
    (ODE vector field + Laplacian) for one relation. Four kinds:
        Lateral    : within a set            (uses Level.edge_index)
        Aggregate  : children -> parent  (up)   (uses Level.parent)
        Broadcast  : parent -> children  (down) (uses Level.parent)
        Exchange   : set <-> field/other set    (scatter / gather, bipartite)
  * A **Schedule** is an ordered, multi-rate list of operators. It *integrates*
    the accumulated derivatives (operators stay pure -> composable rollouts).

Operators return deltas; the Schedule sums and integrates them. This keeps
multiple operators writing the same level safe and keeps rollout differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# --------------------------------------------------------------------------- #
#  Entities: Level (a set) and Field (a continuum)
# --------------------------------------------------------------------------- #
class Level(nn.Module):
    """A set S_k of like nodes at one scale (cells, particles, molecules...).

    Storage is flat/batched (PyG `batch`-vector style), never one module per
    node. The containment tree lives in `parent`.
    """

    def __init__(
        self,
        name: str,
        level: int,
        state: torch.Tensor,                  # [N, d]  dynamic state
        embedding: Optional[torch.Tensor] = None,   # [N, e]  learnable a_i
        parent: Optional[torch.Tensor] = None,      # [N]     index into level+1
        edge_index: Optional[torch.Tensor] = None,  # [2, E]  lateral relation
    ):
        super().__init__()
        self.name = name
        self.level = level
        self.register_buffer("state", state)
        self.embedding = nn.Parameter(embedding) if embedding is not None else None
        self.register_buffer("parent", parent if parent is not None else torch.empty(0, dtype=torch.long))
        self.register_buffer("edge_index", edge_index if edge_index is not None else torch.empty(2, 0, dtype=torch.long))

    @property
    def n(self) -> int:
        return self.state.shape[0]

    def __repr__(self):
        return f"Level({self.name!r}, level={self.level}, n={self.n}, d={self.state.shape[-1]})"


class Field(nn.Module):
    """A continuum f: Omega -> R^c on its own frame, bound to one Level.

    Subclasses (registered via @register_field) supply the discretization and
    the transfer kernels. `scatter` writes object state into the field;
    `gather` reads the field back onto objects. P2G/G2P, secrete/sense and
    morphogen sampling are all (scatter, gather) pairs.
    """

    def __init__(self, name: str, couples_to: str):
        super().__init__()
        self.name = name
        self.couples_to = couples_to        # the Level this field exchanges with

    def scatter(self, level: Level) -> None:        # object -> field
        raise NotImplementedError

    def gather(self, level: Level) -> torch.Tensor:  # field -> object (delta)
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Container: Hierarchy
# --------------------------------------------------------------------------- #
class Hierarchy(nn.Module):
    """Ordered levels (bottom-up) + a flat set of fields. The thing everything
    is built upon."""

    def __init__(self):
        super().__init__()
        self.levels = nn.ModuleDict()       # name -> Level
        self.fields = nn.ModuleDict()       # name -> Field

    def add_level(self, lvl: Level) -> Level:
        self.levels[lvl.name] = lvl
        return lvl

    def add_field(self, fld: Field) -> Field:
        self.fields[fld.name] = fld
        return fld

    def level(self, name: str) -> Level:
        return self.levels[name]


# --------------------------------------------------------------------------- #
#  Operators: GNN = ODE vector field + Laplacian, dispatched by relation
# --------------------------------------------------------------------------- #
@dataclass
class Relation:
    """Names the operands an operator acts on within a Hierarchy."""
    src: str                       # source level name
    dst: Optional[str] = None      # target level/field (None -> same as src)


class Operator(MessagePassing):
    """Base operator. `forward` returns a delta dict {target_name: dstate}.

    KIND and LEVEL are stamped by the registry decorator.
    """

    KIND: str = None
    LEVEL: str = None

    def __init__(self, config=None, device=None, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.config = config
        self.device = device

    def forward(self, H: Hierarchy, rel: Relation) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class Lateral(Operator):
    """Within-set message passing (ODE interaction + discrete Laplacian)."""
    KIND = "lateral"


class Aggregate(Operator):
    """children -> parent reduction over the partition `Level.parent`."""
    KIND = "aggregate"


class Broadcast(Operator):
    """parent -> children lift along the partition `Level.parent`."""
    KIND = "broadcast"


class Exchange(Operator):
    """set <-> field (or set <-> set) scatter/gather. The bipartite operator;
    unifies P2G/G2P, secrete/sense, and reaction stoichiometry."""
    KIND = "exchange"


# --------------------------------------------------------------------------- #
#  Dynamics: Schedule (multi-rate, integrates accumulated deltas)
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    op: str                        # operator name (registry key)
    rel: Relation
    rate: int = 1                  # run every `rate`-th outer tick (timescale)


class Schedule(nn.Module):
    """A model = an ordered, multi-rate list of operator steps. Pure operators
    return deltas; the Schedule sums per level and integrates (Euler by
    default)."""

    def __init__(self, steps: list[Step], dt: float = 1.0):
        super().__init__()
        self.steps = steps
        self.dt = dt

    def forward(self, H: Hierarchy, tick: int = 0) -> Hierarchy:
        from tissue_graph.models.registry import get_operator

        deltas: dict[str, torch.Tensor] = {}
        for s in self.steps:
            if tick % s.rate != 0:
                continue
            op = get_operator(s.op)(config=getattr(H, "config", None))
            for tgt, d in op(H, s.rel).items():
                deltas[tgt] = deltas.get(tgt, 0) + d
        for name, d in deltas.items():           # integrate
            lvl = H.levels[name]
            lvl.state = lvl.state + self.dt * d
        return H

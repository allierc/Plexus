"""Boids (Reynolds flocking) decomposed into its three classic steering rules as
separate Lateral operators: cohesion, alignment, separation. The engine SUMS their
deltas ("several operators on one set, deltas add"), and because each is a mean over
the same neighbour graph and the mean is linear, their sum equals the monolithic
PDE_B `boids` operator to float precision:

    cohesion_i   =  w^c_i a1  mean_j (pos_j - pos_i)            # steer toward neighbours
    alignment_i  =  w^a_i a2  mean_j (vel_j - vel_i)            # match neighbour velocity
    separation_i = -w^s_i a3  mean_j (pos_j - pos_i)/|d_ij|^2   # avoid crowding (short range)

Each is a second-derivative law (returns an acceleration) over the edges a `rewire`
op (e.g. `radius_graph`) left on the set. The per-receiver weights w^c/w^a/w^s are
named per-type properties (`cohesion`/`alignment`/`separation`), broadcast to
per-agent buffers by the engine -- so the spec reads the three rules by name instead
of a positional `p:` list. Fixed scales a1=0.5e-5, a2=5e-4, a3=1e-8 (PDE_B), each an
overridable `scale`. Separation's 1/|d|^2 relies on the radius graph's `min_radius`.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


def _neighbour_mean(lvl, H, msg_fn):
    """Mean over each receiver i's neighbours j of msg_fn(i, j, d_ij) -> acc [N, 2]."""
    pos = lvl.get("pos")
    occ = lvl.occ
    N = pos.shape[0]
    ei = lvl.edge_index                                  # row0 = receiver i, row1 = neighbour j
    if ei.numel() == 0:
        return torch.zeros(N, 2, device=pos.device)
    i, j = ei[0], ei[1]
    d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                      getattr(H, "world_width", 1.0))    # j - i  [E, 2]
    msg = msg_fn(i, j, d) * occ[j, None]                 # ignore dormant neighbours
    acc = torch.zeros(N, 2, device=pos.device)
    acc.index_add_(0, i, msg)
    deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
    return (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]


def _masked(acc, mask):
    return acc if mask is None else acc * mask[:, None].float()


@register_operator("cohesion", level="particle", kind="lateral")
class Cohesion(Lateral):
    PREDICTION = "second_derivative"
    REQUIRES_TYPE_PROPS = ["cohesion"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 0.5e-5))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.cohesion
        acc = _neighbour_mean(lvl, H, lambda i, j, d: w[i, None] * self.a * d)
        return {self.at: _masked(acc, mask)}


@register_operator("alignment", level="particle", kind="lateral")
class Alignment(Lateral):
    PREDICTION = "second_derivative"
    REQUIRES_TYPE_PROPS = ["alignment"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 5e-4))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.alignment
        vel = lvl.get("vel")
        acc = _neighbour_mean(lvl, H, lambda i, j, d: w[i, None] * self.a * (vel[j] - vel[i]))
        return {self.at: _masked(acc, mask)}


@register_operator("separation", level="particle", kind="lateral")
class Separation(Lateral):
    PREDICTION = "second_derivative"
    REQUIRES_TYPE_PROPS = ["separation"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 1e-8))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.separation

        def msg(i, j, d):
            d2 = (d * d).sum(-1, keepdim=True)           # |d|^2 (> 0 via the graph's min_radius)
            return -w[i, None] * self.a * d / d2

        acc = _neighbour_mean(lvl, H, msg)
        return {self.at: _masked(acc, mask)}

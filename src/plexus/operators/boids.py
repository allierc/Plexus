"""Boids (Reynolds flocking) -- a second-derivative Lateral operator.

For a receiver particle i, averaged over its neighbours j (the edges left by a
`rewire` operator such as `radius_graph`), the acceleration is the sum of three
classic rules with per-type weights p = [cohesion, alignment, separation]:

    cohesion   =  p1 * a1 * (pos_j - pos_i)              # steer toward neighbours
    alignment  =  p2 * a2 * (vel_j - vel_i)              # match neighbour velocity
    separation = -p3 * a3 * (pos_j - pos_i) / |d_ij|^2   # avoid crowding (short range)
    acc_i = mean_j (cohesion + alignment + separation)

with fixed scales a1=0.5e-5, a2=5e-4, a3=1e-8 (the canonical PDE_B constants;
overridable). Second-derivative law -> returns an acceleration (set
`prediction: second_derivative`); the engine integrates vel += dt·acc, pos +=
dt·vel. Separation's 1/|d|^2 needs a `min_radius` on the radius graph so near
coincidences don't blow up. Per-type weights come from the spec `types:` block.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("boids", level="particle", kind="lateral")
class Boids(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration
    REQUIRES_TYPE_PROPS = ["p"]                 # per-type [cohesion, alignment, separation]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a1 = float(params.get("a1", 0.5e-5))   # cohesion scale
        self.a2 = float(params.get("a2", 5e-4))     # alignment scale
        self.a3 = float(params.get("a3", 1e-8))     # separation scale
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        vel = lvl.get("vel")
        occ = lvl.occ
        N = pos.shape[0]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, 2, device=pos.device)}
        i, j = ei[0], ei[1]

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_width", 1.0))   # j - i  [E, 2]
        dv = vel[j] - vel[i]                                 # neighbour velocity difference
        d2 = (d * d).sum(-1, keepdim=True)                  # |d|^2  (>0: min_radius on the graph)
        p = lvl.type_params[lvl.node_type[i]]               # receiver-type weights [E, 3]

        cohesion = p[:, 0:1] * self.a1 * d
        alignment = p[:, 1:2] * self.a2 * dv
        separation = -p[:, 2:3] * self.a3 * d / d2
        msg = (cohesion + alignment + separation) * occ[j, None]

        acc = torch.zeros(N, 2, device=pos.device)
        acc.index_add_(0, i, msg)
        deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
        acc = acc / deg.clamp(min=1.0)[:, None]             # mean aggregation
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

"""alignment -- a boids steering rule (Lateral, second-derivative).

Match neighbours' velocity: `a2 * w^a_i * mean_j (vel_j - vel_i)`, with the
per-receiver weight `w^a` the type's named `alignment` property. One of the three
rules whose deltas the engine sums to make a flock; see also cohesion, separation.
"""
from __future__ import annotations

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import neighbour_mean


@register_operator("alignment", level="particle", kind="lateral")
class Alignment(Lateral):
    PREDICTION = "second_derivative"
    REQUIRES_TYPE_PROPS = ["alignment"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 5e-4))       # PDE_B alignment scale a2
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.alignment
        vel = lvl.get("vel")
        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False), getattr(H, "world_width", 1.0),
                             lambda i, j, d: w[i, None] * self.a * (vel[j] - vel[i]))
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

"""cohesion -- a boids steering rule (Lateral, second-derivative).

Steer toward the local centre of mass: the mean offset to neighbours,
`a1 * w^c_i * mean_j (pos_j - pos_i)`, with the per-receiver weight `w^c` the type's
named `cohesion` property. One of the three rules whose deltas the engine sums to
make a flock; see also alignment, separation.
"""
from __future__ import annotations

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import neighbour_mean


@register_operator("cohesion", level="particle", kind="lateral")
class Cohesion(Lateral):
    PREDICTION = "second_derivative"
    REQUIRES_TYPE_PROPS = ["cohesion"]
    MECHANISM_TAGS = ["cohesion", "collective_motion"]
    MORPHOLOGY_PRIOR = ["flock", "cluster"]
    PARAM_ROLES = {"scale": "cohesion_strength"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 0.5e-5))     # PDE_B cohesion scale a1
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.cohesion
        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False), getattr(H, "world_width", 1.0),
                             lambda i, j, d: w[i, None] * self.a * d)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

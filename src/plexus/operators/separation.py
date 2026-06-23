"""separation -- a boids steering rule (Lateral, second-derivative).

Avoid crowding at short range: `-a3 * w^s_i * mean_j (pos_j - pos_i)/|d_ij|^2`, with
the per-receiver weight `w^s` the type's named `separation` property. The 1/|d|^2
relies on the radius graph's `min_radius` so coincident neighbours don't blow up. One
of the three rules whose deltas the engine sums to make a flock; see also cohesion,
alignment.
"""
from __future__ import annotations

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import neighbour_mean


@register_operator("separation", level="particle", kind="lateral")
class Separation(Lateral):
    PREDICTION = "second_derivative"
    SUPPORTED_DIMS = [2, 3]                          # neighbour_mean is N-D; the rule is dimension-generic
    REQUIRES_TYPE_PROPS = ["separation"]
    MECHANISM_TAGS = ["short_range_repulsion", "collision_avoidance"]
    MORPHOLOGY_PRIOR = ["even_spacing"]
    PARAM_ROLES = {"scale": "separation_strength"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("scale", 1e-8))       # PDE_B separation scale a3
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        w = lvl.separation

        def msg(i, j, d):
            d2 = (d * d).sum(-1, keepdim=True)          # |d|^2 (> 0 via the graph's min_radius)
            return -w[i, None] * self.a * d / d2

        acc = neighbour_mean(lvl.get("pos"), lvl.occ, lvl.edge_index,
                             getattr(H, "periodic", False),
                             getattr(H, "world_size", getattr(H, "world_width", 1.0)), msg)
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

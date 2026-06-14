"""boids: Lateral operator @ cell. Collective motion (separation/alignment/cohesion).

State convention for a 2-D cell set: Level.state[:, :2] = position,
Level.state[:, 2:4] = velocity. Operators return {set_name: accel [N, dim]}.
"""

from __future__ import annotations

import torch

from tissue_graph.models.base import Lateral
from tissue_graph.models.registry import register_operator


@register_operator("boids", level="cell", kind="lateral")
class BoidsOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.r = float(params.get("radius", 0.07))
        self.w_coh = float(params.get("cohesion", 0.8))
        self.w_align = float(params.get("align", 0.5))
        self.w_sep = float(params.get("separate", 1.5))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos, vel = cell.state[:, :2], cell.state[:, 2:4]
        d = torch.cdist(pos, pos)
        W = ((d < self.r) & (d > 0)).float()
        count = W.sum(1).clamp(min=1.0)[:, None]

        cohesion = (W @ pos) / count - pos
        alignment = (W @ vel) / count - vel
        S = W / d.clamp(min=0.1 * self.r) ** 2          # soften 1/d^2 singularity
        separation = pos * S.sum(1)[:, None] - S @ pos

        accel = self.w_coh * cohesion + self.w_align * alignment + self.w_sep * separation
        return {"cell": accel}

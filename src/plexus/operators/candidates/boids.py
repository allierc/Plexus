"""boids: Lateral operator @ cell. Collective motion (separation/alignment/cohesion).

State convention for a 2-D cell set: Level.state[:, :2] = position,
Level.state[:, 2:4] = velocity. Operators return {set_name: accel [N, dim]}.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


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
        diff = pos[None, :, :] - pos[:, None, :]        # [N,N,2]  i -> j
        if getattr(H, "periodic", False):
            diff = torch.remainder(diff + 0.5, 1.0) - 0.5   # minimum-image (bc_dpos)
        d = diff.norm(dim=2)
        W = ((d < self.r) & (d > 0)).float()
        if mask is not None:                            # flock only within the selected subpop
            W = W * mask.float()[None, :]
        count = W.sum(1).clamp(min=1.0)[:, None]
        cohesion = (W[..., None] * diff).sum(1) / count        # toward neighbour centroid
        alignment = (W @ vel) / count - vel
        S = W / d.clamp(min=0.1 * self.r) ** 2          # soften 1/d^2 singularity
        separation = -(S[..., None] * diff).sum(1)             # push away from neighbours
        accel = self.w_coh * cohesion + self.w_align * alignment + self.w_sep * separation
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}

"""motility: Lateral operator @ cell. Active self-propulsion (active Brownian).

Each cell carries a heading that random-walks (rotational diffusion); the cell is
pushed along it at constant strength. This makes *every* cell wander on its own,
independent of neighbours or fields -- so the stiff population (no chemotaxis,
few boids neighbours) still moves. Reproducible: uses the Hierarchy's seeded RNG.
"""

from __future__ import annotations

import math
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("motility", level="cell", kind="lateral")
class MotilityOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.speed = float(params.get("speed", 30.0))   # propulsion accel
        self.rot = float(params.get("rot", 0.4))        # rotational diffusion / frame

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        if not hasattr(cell, "heading"):
            cell.heading = torch.rand(cell.n, generator=H.rng, device=dev) * 2 * math.pi
        cell.heading = cell.heading + self.rot * torch.randn(cell.n, generator=H.rng, device=dev)
        accel = self.speed * torch.stack([torch.cos(cell.heading), torch.sin(cell.heading)], dim=1)
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}

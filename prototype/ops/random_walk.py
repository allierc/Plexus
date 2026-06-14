"""random_walk: Lateral operator @ cell. Small uncorrelated Brownian jitter.

Adds a small random acceleration to each cell every frame (unlike `motility`,
which is a *persistent* heading). This breaks up ballistic, asteroid-like
trajectories and gives cells a more diffusive, biological wander. Uses the
Hierarchy's seeded RNG, so runs stay reproducible.
"""

from __future__ import annotations

import torch

from tissue_graph.models.base import Lateral
from tissue_graph.models.registry import register_operator


@register_operator("random_walk", level="cell", kind="lateral")
class RandomWalkOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.strength = float(params.get("strength", 25.0))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        a = self.strength * torch.randn(cell.n, 2, generator=H.rng, device=cell.state.device)
        if mask is not None:
            a = a * mask.float()[:, None]
        return {"cell": a}

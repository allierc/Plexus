"""adhesion: Lateral operator @ cell. Type-specific cohesion (differential adhesion).

Each cell is pulled toward the centroid of nearby cells *of its own type*. Two
mixed populations therefore sort into same-type domains (Steinberg differential
adhesion hypothesis) -- a pattern none of the other operators can produce.

Added to demonstrate the extension path: a missing capability becomes one new
`Operator` subclass conforming to `forward(H, mask) -> {set: accel}`, registered
with a name. No engine change.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("adhesion", level="cell", kind="lateral")
class AdhesionOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.r = float(params.get("radius", 0.1))
        self.strength = float(params.get("strength", 1.0))   # same-type cohesion (>0 pull together)
        self.cross = float(params.get("cross", 0.0))         # cross-type force: <0 repel -> de-mix

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos, typ = cell.state[:, :2], cell.node_type
        diff = pos[None, :, :] - pos[:, None, :]              # [N,N,2]  i -> j (toward neighbour)
        if getattr(H, "periodic", False):
            diff = torch.remainder(diff + 0.5, 1.0) - 0.5     # minimum-image
        d = diff.norm(dim=2)
        near = ((d < self.r) & (d > 0)).float()
        same = (typ[:, None] == typ[None, :]).float()
        Wsame = near * same                                  # same-type neighbours
        Wcross = near * (1.0 - same)                         # other-type neighbours
        cnt_s = Wsame.sum(1).clamp(min=1.0)[:, None]
        cnt_c = Wcross.sum(1).clamp(min=1.0)[:, None]
        accel = (self.strength * (Wsame[..., None] * diff).sum(1) / cnt_s     # cohere with own type
                 + self.cross * (Wcross[..., None] * diff).sum(1) / cnt_c)    # cross<0 -> push off other type
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}

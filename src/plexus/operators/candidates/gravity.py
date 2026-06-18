"""gravity: Lateral operator @ cell. A uniform body force (constant acceleration).

Applies the same acceleration to every selected cell; `mpm` broadcasts it down to
that cell's particles as the external accel `a_ext`. This is the body force that
makes a soft body fall. The *bounce* is deliberately NOT a force here: it is the
emergent elastic response when the falling body compresses against the reflective
floor boundary condition (grid velocity into a wall is clamped to zero, the
fixed-corotated stress then springs the body back). Stiffer `youngs` -> livelier
bounce; `drag` on the mpm operator sets how fast successive bounces decay.

Default direction is -y (down). Override `gx`/`gy` for an incline or sideways pull.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("gravity", level="cell", kind="lateral")
class GravityOperator(Lateral):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.g = float(params.get("g", 10.0))            # magnitude (world units / time^2)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        accel = torch.zeros(cell.n, 2, device=dev)
        accel[:, 0] = self.gx
        accel[:, 1] = self.gy
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}

"""advance_3d -- set (lateral). Self-propulsion along a 3D heading vector.

The 3D counterpart of `advance`: where the 2D agent carries a scalar `heading`
angle and moves along `(cos h, sin h)`, the 3D agent carries a unit `heading`
VECTOR and moves along it: `v_i = move_speed_i * heading_i`. First-derivative law
(emits a velocity; the ENGINE integrates the position). Purely propulsion -- wall
reflection is the separate `bounce_3d` operator, so this stays one concern.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("advance_3d", level="cell", kind="lateral")
class Advance3D(Lateral):
    PREDICTION = "first_derivative"             # emits a velocity; the ENGINE integrates pos
    SUPPORTED_DIMS = [3]
    REQUIRES_TYPE_PROPS = ["move_speed"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading                                   # [N, 3] unit heading vector
        spd = lvl.move_speed                              # [N]
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = spd[:, None] * h                            # move along the heading
        return {self.at: vel * m[:, None]}

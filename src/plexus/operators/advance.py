"""advance -- set (lateral). Self-propulsion: move along the heading at speed.

A first-derivative dynamics operator: it returns a velocity delta
`v_i = move_speed_i * (cos h_i, sin h_i)` and lets the ENGINE integrate the
position (`pos += dt * v`), exactly like attraction_repulsion returns a dpos.
It does NOT touch `pos`, and it is purely propulsion -- domain/obstacle reflection
is a separate `bounce` operator (run before advance), so this stays one concern.

Renamed from the prototype `motility` (it is the slime "move" step).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("advance", level="cell", kind="lateral")
class Advance(Lateral):
    PREDICTION = "first_derivative"             # emits a velocity; the ENGINE integrates pos
    REQUIRES_TYPE_PROPS = ["move_speed"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading
        spd = lvl.move_speed
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = spd[:, None] * torch.stack([torch.cos(h), torch.sin(h)], dim=1)
        return {self.at: vel * m[:, None]}

"""advance -- set (lateral). Self-propulsion: move along the heading at speed.

A first-derivative dynamics operator: it returns a velocity delta
`v_i = move_speed_i * (cos h_i, sin h_i)` and lets the ENGINE integrate the
position (`pos += dt * v`), exactly like attraction_repulsion returns a dpos.
It does NOT touch `pos` itself -- the integrator is the engine's job.

The heading is auxiliary control state (set by `sense`, not integrated by the
engine). On a wall, advance re-heads *proactively*: if a step along the current
heading would leave the domain, it picks a fresh random heading this tick (so the
emitted velocity already points back inward) -- Lague's bounce, without the
operator ever writing a position.

Renamed from the prototype `motility` (it is the slime "move" step).
"""
from __future__ import annotations

import math
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
        pos = lvl.get("pos")
        h = lvl.heading
        spd = lvl.move_speed
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = m > 0

        if not getattr(H, "periodic", False):
            # proactive wall bounce: if a dt-step along the heading would leave the
            # domain, re-head NOW (heading is auxiliary control state, not the
            # integrated pos), so the velocity we emit already points back inward.
            dt = float(getattr(H.config, "dt", 1.0))
            W = float(getattr(H, "world_width", 1.0))
            nx = pos[:, 0] + dt * spd * torch.cos(h)
            ny = pos[:, 1] + dt * spd * torch.sin(h)
            out = (nx < 0) | (nx >= W) | (ny < 0) | (ny >= 1)
            rand = torch.rand(N, generator=H.rng, device=dev) * 2 * math.pi
            h = torch.where(out & keep, rand, h)
            lvl.heading = torch.where(keep, h, lvl.heading)

        vel = spd[:, None] * torch.stack([torch.cos(h), torch.sin(h)], dim=1)
        return {self.at: vel * m[:, None]}

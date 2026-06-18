"""advance -- set (lateral). Self-propulsion: move along the heading at speed.

Each agent steps forward `dt * move_speed` along its current heading (the heading
that `sense` set). On a wall it picks a fresh random heading (Lague's bounce); on a
periodic domain it wraps. A first-class self-propelled mover: it integrates its own
position and returns {} (sanctioned by the operator contract -- like the other
self-moving/exchange ops), so the set needs no separate integration order.

Renamed from the prototype `motility` (it is the slime "move" step); the active-
Brownian rotational-diffusion variant can layer on top as a separate operator.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("advance", level="cell", kind="lateral")
class Advance(Lateral):
    REQUIRES_TYPE_PROPS = ["move_speed"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.state[:, :2]
        h = lvl.heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        W = float(getattr(H, "world_width", 1.0))

        step = dt * spd
        nx = pos[:, 0] + torch.cos(h) * step
        ny = pos[:, 1] + torch.sin(h) * step

        if getattr(H, "periodic", False):
            nx = torch.remainder(nx, W)
            ny = torch.remainder(ny, 1.0)
            new_h = h
        else:
            out = (nx < 0) | (nx >= W) | (ny < 0) | (ny >= 1)         # hit a wall
            rand = torch.rand(N, generator=H.rng, device=dev) * 2 * math.pi
            new_h = torch.where(out, rand, h)                         # pick a fresh direction
            nx = nx.clamp(0, W - 1e-6)
            ny = ny.clamp(0, 1 - 1e-6)

        new_pos = torch.stack([nx, ny], dim=1)
        m = (mask.float() if mask is not None else torch.ones(N, device=dev))
        keep = m > 0
        lvl.heading = torch.where(keep, new_h, h)
        lvl.state[:, :2] = torch.where(keep[:, None], new_pos, pos)
        return {}

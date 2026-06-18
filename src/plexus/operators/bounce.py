"""bounce -- set. Reflect a self-propelled agent off the domain walls / obstacles.

A boundary condition on *orientation*, kept separate from propulsion: if a dt-step
along the current heading would leave the domain (or enter an obstacle), the agent
picks a fresh random heading. Run it AFTER `sense` (which steers) and BEFORE
`advance` (which emits the propulsion velocity), so the velocity already points
back inside -- proactive bounce, no agent ever crosses the wall.

It mutates the heading (auxiliary control state, not the integrated position) and
returns {} -- like the other in-place set ops, it is not a force.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _in_obstacles(x, y, obstacles):
    """Bool mask: is (x, y) inside any obstacle? rect=[x0,y0,x1,y1], disc=[cx,cy,r]."""
    hit = torch.zeros_like(x, dtype=torch.bool)
    for o in (obstacles or []):
        if len(o) == 4:
            x0, y0, x1, y1 = o
            hit = hit | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        elif len(o) == 3:
            cx, cy, r = o
            hit = hit | (((x - cx) ** 2 + (y - cy) ** 2) <= r * r)
    return hit


@register_operator("bounce", level="cell", kind="lateral")
class Bounce(Lateral):
    REQUIRES_TYPE_PROPS = ["move_speed"]        # needs the step length it is about to take

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        if getattr(H, "periodic", False):
            return {}                                       # torus: nothing to bounce off
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.get("pos")
        h = lvl.heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        W = float(getattr(H, "world_width", 1.0))
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = m > 0

        nx = pos[:, 0] + dt * spd * torch.cos(h)
        ny = pos[:, 1] + dt * spd * torch.sin(h)
        out = (nx < 0) | (nx >= W) | (ny < 0) | (ny >= 1)
        out = out | _in_obstacles(nx, ny, getattr(H, "obstacles", []))
        rand = torch.rand(N, generator=H.rng, device=dev) * 2 * math.pi
        lvl.heading = torch.where(out & keep, rand, h)
        return {}

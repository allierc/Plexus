"""bounce_3d -- set (lateral). Reflect a 3D agent's heading off the domain walls.

The 3D counterpart of `bounce`: if a dt-step along the current heading vector would
leave the box on some axis, that axis's heading component is reflected (specular
wall bounce), and the heading is renormalised. Run AFTER `sense_3d` (which steers)
and BEFORE `advance_3d` (which emits the propulsion velocity), so the velocity
already points back inside -- no agent ever crosses the wall.

It mutates the heading vector (auxiliary control state, not the integrated
position) and returns {} -- like the other in-place set ops, it is not a force.
A torus domain has nothing to bounce off, so it is a no-op under periodic BC.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("bounce_3d", level="cell", kind="lateral")
class Bounce3D(Lateral):
    SUPPORTED_DIMS = [3]
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
        pos = lvl.get("pos")                                # [N, 3]
        h = lvl.heading                                     # [N, 3] unit heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        box = H.world_size                                  # [3] per-axis box size
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]

        nxt = pos + dt * spd[:, None] * h                   # tentative next position
        out = (nxt < 0) | (nxt > box[None, :])              # which axes would exit the box
        new_h = torch.where(out, -h, h)                     # reflect the exiting components
        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)
        lvl.heading = torch.where(keep, new_h, h)
        return {}

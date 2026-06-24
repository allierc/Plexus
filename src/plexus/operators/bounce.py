"""bounce -- set. Reflect a self-propelled agent off the domain walls / obstacles.

A boundary condition on *orientation*, kept separate from propulsion: if a dt-step
along the current heading would leave the box (or enter an obstacle), the heading is
turned back inside. Run it AFTER `sense` (which steers) and BEFORE `advance` (which
emits the propulsion velocity), so the velocity already points back inside --
proactive bounce, no agent ever crosses the wall.

Dimension-generic (the dimension contract): `heading` is a unit VECTOR [N, D] in
every dimension. Walls use **specular reflection** -- the heading component on any
axis the step would exit is negated (`heading[out_axis] *= -1`), which works
identically in 2D and 3D and replaces the old scalar-angle random re-heading.
Obstacles (rect/disc walls, a 2D maze feature) still re-head randomly, since they
have no single axis-aligned normal. Replaces the old 2D `bounce` + `bounce_3d`.

It mutates the heading (auxiliary control state, not the integrated position) and
returns {} -- like the other in-place set ops, it is not a force.
"""
from __future__ import annotations

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


def _random_unit(n, D, rng, device):
    """A random unit vector per agent [n, D] (isotropic re-heading off obstacles)."""
    v = torch.randn(n, D, generator=rng, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)


@register_operator("bounce", level="cell", kind="lateral")
class Bounce(Lateral):
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic specular wall reflection
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
        pos = lvl.get("pos")                                # [N, D]
        h = lvl.heading                                     # [N, D] unit heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        box = H.world_size                                  # [D] per-axis box size
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]

        nxt = pos + dt * spd[:, None] * h                   # tentative next position
        out = (nxt < 0) | (nxt > box[None, :])              # which axes would exit the box
        new_h = torch.where(out, -h, h)                     # specular reflect the exiting components

        # obstacles (2D maze rects/discs): re-head isotropically where the step would
        # enter one -- they carry no single axis-aligned normal to reflect against.
        obs = getattr(H, "obstacles", [])
        if obs and pos.shape[1] == 2:
            hit = _in_obstacles(nxt[:, 0], nxt[:, 1], obs)
            new_h = torch.where(hit[:, None], _random_unit(N, 2, H.rng, dev), new_h)

        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)
        lvl.heading = torch.where(keep, new_h, h)
        return {}

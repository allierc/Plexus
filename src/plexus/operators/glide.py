"""glide -- set (lateral). Self-propulsion: glide along the heading at constant speed (overdamped, first-order; the heading-kinematic sibling of `cruise`).

A first-derivative dynamics operator: it returns a velocity delta
`v_i = move_speed_i * heading_i` and lets the ENGINE integrate the position
(`pos += dt * v`), exactly like attraction_repulsion returns a dpos. It does NOT
touch `pos`, and it is purely propulsion -- domain/obstacle reflection is a separate
`bounce` operator (run before glide), so this stays one concern.

Dimension-generic (the dimension contract): `heading` is a unit VECTOR [N, D] in
every dimension, so `vel = speed * heading` is the same law in 2D and 3D -- no
`cos/sin` special case. Replaces the old scalar-angle 2D law and the `advance_3d`
counterpart with one operator.

Renamed from the prototype `motility` (it is the slime "move" step).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("glide", level="cell", kind="lateral")
class Glide(Lateral):
    PREDICTION = "first_derivative"             # emits a velocity; the ENGINE integrates pos
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_TYPE_PROPS = ["move_speed"]
    PARAM_ROLES = {"noise": "translational_noise"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.noise = float(params.get("noise", 0.0))      # isotropic translational noise (active Brownian; off by default)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading                                   # [N, D] unit heading vector
        spd = lvl.move_speed                              # [N]
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = spd[:, None] * h                            # move along the heading
        if self.noise > 0.0:                              # glide + noise = an active Brownian walker
            vel = vel + self.noise * torch.randn(N, h.shape[-1], generator=getattr(H, "rng", None), device=dev)
        return {self.at: vel * m[:, None]}

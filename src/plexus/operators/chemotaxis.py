"""chemotaxis -- set <- field. Particles climb (or flee) a field's gradient.

The Keller-Segel coupling: each particle gets a velocity `gain * grad(field)` sampled
at its position (gain<0 to flee). A field->set Exchange that returns a first-derivative
velocity delta -- the ENGINE integrates it, and it simply sums with any other velocity
the particle has (e.g. attraction_repulsion), the framework's "deltas add" rule. The
field may be deposited (slime), reaction-diffused, or prescribed from a video.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("chemotaxis", level="particle", kind="exchange")
class Chemotaxis(Exchange):
    PREDICTION = "first_derivative"             # emits a velocity; the ENGINE integrates
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["gradient_following", "field_templated_aggregation"]
    MORPHOLOGY_PRIOR = ["single_cluster", "field_outline"]
    PARAM_ROLES = {"gain": "field_sensitivity"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.gain = float(params.get("gain", 1.0))
        self.channel = int(params.get("channel", 0))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        g = fld.grid[self.channel]                              # [nx, ny]
        W = float(getattr(H, "world_width", 1.0))
        # boundary-aware central difference (x spans [0,W], y spans [0,1]): wrap under a
        # periodic world, replicate under a wall -- so a wall edge has no artificial gradient.
        mode = "circular" if getattr(H, "periodic", False) else "replicate"
        gp = Fnn.pad(g[None, None], (1, 1, 1, 1), mode=mode)[0, 0]   # [nx+2, ny+2]
        gx = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5 * (fld.nx / W)     # d/dx, per world unit
        gy = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5 * fld.ny           # d/dy, per world unit
        grad = torch.stack([gx, gy], 0)[None]                       # [1, 2, nx, ny]
        # bilinear sample the gradient at particle positions
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]         # grid_sample: last dim (x=ny, y=nx)
        vel = Fnn.grid_sample(grad, grid, mode="bilinear", padding_mode="border",
                              align_corners=True)[0, :, 0].t()  # [N, 2]
        vel = self.gain * vel * lvl.occ[:, None]
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}

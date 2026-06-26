"""chemo_force -- set <- field. A chemotactic BODY FORCE (acceleration along grad).

Identical sampling to `chemotaxis` (bilinear grad(field) at each particle), but it
returns an ACCELERATION (`gain * grad`) rather than a velocity. `PREDICTION = None`
(it does not vote on a set's integration order), which makes it a universal body
force consumed two ways:

* on an MLS-MPM set (e.g. `mpm_particle`): `p2g` reads `H.delta(set)` as the
  per-particle external acceleration `a_ext` (the same path as `mpm_drag` /
  `pulse_to_contraction`), so the WATER flows along another set's deposited field --
  a river streaming through the slime organic pattern;
* on a second-order set (e.g. a boids flock): the set's order is fixed by boids and
  this acceleration simply adds into the summed delta the engine integrates.

gain<0 to flee the field; `channel=None` sums all channels (follow any slime trail).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("chemo_force", level="particle", kind="exchange")
class ChemoForce(Exchange):
    PREDICTION = None                           # body force: MPM p2g consumes it, or a boids set integrates it
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["gradient_following", "field_templated_flow"]
    PARAM_ROLES = {"gain": "field_sensitivity"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.gain = float(params.get("gain", 1.0))
        self.channel = params.get("channel", None)          # None -> sum all channels (any slime trail)
        self.by_material = bool(params.get("by_material", False))  # solids climb (+|gain|), liquids flee (-|gain|)
        self.after = int(params.get("after", 0))            # gate: off until frame >= after (phased forcing)
        self.before = int(params.get("before", 1 << 30))    # gate: off once frame >= before
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        if not (self.after <= int(getattr(H, "frame", 0)) < self.before):
            return {self.at: torch.zeros_like(lvl.get("pos"))}      # outside the active window -> no force
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        g = fld.grid.sum(0) if self.channel is None else fld.grid[int(self.channel)]   # [nx, ny]
        W = float(getattr(H, "world_width", 1.0))
        mode = "circular" if getattr(H, "periodic", False) else "replicate"
        gp = Fnn.pad(g[None, None], (1, 1, 1, 1), mode=mode)[0, 0]
        gx = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5 * (fld.nx / W)
        gy = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5 * fld.ny
        grad = torch.stack([gx, gy], 0)[None]                       # [1, 2, nx, ny]
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]
        acc = Fnn.grid_sample(grad, grid, mode="bilinear", padding_mode="border",
                              align_corners=True)[0, :, 0].t()      # [N, 2]
        if self.by_material and getattr(lvl, "is_liquid", None) is not None:
            # same field, opposite pull per phase: elastic/solid climbs onto the
            # filaments (+|gain|), liquid is pushed into the voids (-|gain|).
            sign = torch.where(lvl.is_liquid, -1.0, 1.0).to(acc.dtype)[:, None]
            acc = abs(self.gain) * sign * acc
        else:
            acc = self.gain * acc
        acc = acc * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

"""deposit -- set -> field. Each agent adds to the field at its own pixel.

The stigmergy write: agent i adds `amount*dt` to channel `node_type[i]` (its own
species) at the pixel under its position, clamped to 1 (the shader's `min(1,...)`).
This is the only place the agents imprint the trail they then sense.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("deposit", level="cell", kind="exchange")
class Deposit(Exchange):
    """object -> field. Writes `to:` field in place; returns {}."""

    SUPPORTED_DIMS = [2, 3]                     # N-D scatter onto the grid field
    REQUIRES_PARAMS = ["to"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.amount = float(params.get("amount", 0.9))
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        fld = H.fields[self.field_name]
        pos = lvl.get("pos")                                      # [N, D] (D = 2 or 3)
        D = pos.shape[1]
        nt = lvl.node_type
        dt = float(getattr(H.config, "dt", 1.0))
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ

        gidx = fld.pix(*[pos[:, k] for k in range(D)])           # D-tuple of voxel indices
        # channel-major, row-major flat index over the N-D grid (== the 2D
        # `nt*(nx*ny) + gx*ny + gy` exactly when D == 2).
        ravel = torch.zeros(N, dtype=torch.long, device=dev)
        stride = 1
        for k in reversed(range(D)):
            ravel = ravel + gidx[k] * stride
            stride *= fld.shape[k]
        flat = nt * stride + ravel                               # stride == prod(shape)
        amt = torch.full((N,), self.amount * dt, device=dev) * m
        fld.grid.view(-1).index_add_(0, flat, amt)
        fld.grid.clamp_(max=1.0)
        return {}

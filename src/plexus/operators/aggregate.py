"""aggregate -- children -> parent. Reduce a contained set onto its parent.

The `centroid` reduction: a parent's position is the occupancy-weighted mean of its
children's positions (a cell's position = the centroid of its particles). This is a
DERIVED readout, not an integrated force, so it writes the parent's position directly
and declares MAY_MUTATE_STATE to opt out of the integration guard; returns {}.
"""
from __future__ import annotations

import torch

from plexus.models.base import Aggregate
from plexus.models.registry import register_operator


@register_operator("aggregate", level="cell", kind="aggregate")
class Centroid(Aggregate):
    MAY_MUTATE_STATE = True             # writes the parent's derived position (a readout)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.child = params.get("child")           # optional: which contained set (default: first child)

    def forward(self, H, mask=None):
        parent = H.level(self.at)
        kids = H.children(self.at)
        if not kids:
            return {}
        child = H.level(self.child) if self.child else H.level(kids[0])
        pidx = child.parent                        # [Nc] parent slot per child
        if pidx.numel() == 0:
            return {}
        dev = parent.state.device
        px0, px1 = parent.state_schema["pos"]
        cpos = child.get("pos"); cocc = child.occ
        s = torch.zeros(parent.n, 2, device=dev).index_add_(0, pidx, cpos * cocc[:, None])
        w = torch.zeros(parent.n, device=dev).index_add_(0, pidx, cocc)
        centroid = s / w.clamp(min=1.0)[:, None]
        new = parent.state.clone()                 # only live parents take the readout
        new[:, px0:px1] = torch.where(parent.occ[:, None] > 0, centroid, parent.state[:, px0:px1])
        parent.state = new
        return {}

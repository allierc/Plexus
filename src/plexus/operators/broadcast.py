"""broadcast -- parent -> children. Lift a parent quantity onto its children.

The `containment` lift: each child gets a velocity delta `stiffness * (parent_pos -
child_pos)` -- pulled toward its parent's (e.g. aggregated centroid) position, so a
cell holds its particles together. Unlike `aggregate` (a derived readout that writes
the parent), this RETURNS a delta on the children that the engine integrates.
"""
from __future__ import annotations

import torch

from plexus.models.base import Broadcast
from plexus.models.registry import register_operator


@register_operator("broadcast", level="particle", kind="broadcast")
class BroadcastLift(Broadcast):
    PREDICTION = "first_derivative"            # emits a velocity; the engine integrates
    REQUIRES_PARAMS = ["stiffness"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params.get("stiffness", 1.0))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        child = H.level(self.at)
        dev = child.state.device
        pname = getattr(child, "parent_name", None)
        if pname is None:
            return {self.at: torch.zeros(child.n, 2, device=dev)}
        parent = H.level(pname)
        ppos = parent.get("pos")[child.parent]     # each child's parent position
        vel = self.k * (ppos - child.get("pos")) * child.occ[:, None]
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}

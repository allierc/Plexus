"""finish: Lateral operator @ cell. Race finish line + freeze + counter.

A cell whose centroid crosses x > `x` is marked done: it is counted once in
H.finished (the race objective) and its particles are frozen (velocity and affine
momentum zeroed) so it STOPS at the line instead of drifting on. Driving operators
(motility/sense/secrete) select `cell[done=0]`, so finished cells also stop being
pushed. Mirrors how `forage` records H.food_delivered -- a discrete state event,
no engine change.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("finish", level="cell", kind="lateral")
class FinishOperator(Lateral):
    REQUIRES_PARAMS = ["x"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.x = float(params.get("x", 0.9))      # finish line (cross x > this)

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        if not hasattr(cell, "done"):
            cell.done = torch.zeros(cell.n, dtype=torch.bool, device=dev)
        crossed = cell.state[:, 0] > self.x
        newly = crossed & ~cell.done
        H.finished = getattr(H, "finished", 0) + int(newly.sum())
        cell.done = cell.done | crossed
        # freeze the particles of finished cells -> they stop at the line
        part = H.level("particle")
        fz = cell.done[part.parent]
        if fz.any():
            part.state[fz, 2:4] = 0.0
            if hasattr(part, "C"):
                part.C[fz] = 0.0
        return {}

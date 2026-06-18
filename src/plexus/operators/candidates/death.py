"""death: Lateral operator @ cell. Removal at an exit (a sink/efflux boundary).

A cell whose centroid crosses x > `x` is removed from the simulation: it is counted
once in H.finished (the race/escape objective), its particles' mass is zeroed (so it
stops contributing to the MPM grid and can no longer block the cells behind it), and
its particles are parked off-domain so it visibly disappears at the exit rather than
piling up. Removal is sticky (cell.done stays set), and driving operators select
`cell[done=0]`, so a removed cell never moves or is drawn again.

Biologically this is an efflux / apoptosis / outflow boundary -- an open sink for the
discrete set, the counterpart of a field's source. Run it LAST in the schedule (after
mpm) so the parked positions are what gets recorded each frame.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("death", level="cell", kind="lateral")
class DeathOperator(Lateral):
    REQUIRES_PARAMS = ["x"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.x = float(params.get("x", 0.9))      # exit line: remove cells with centroid x > this

    def forward(self, H, mask=None):
        cell = H.level("cell")
        dev = cell.state.device
        if not hasattr(cell, "done"):
            cell.done = torch.zeros(cell.n, dtype=torch.bool, device=dev)
        crossed = cell.state[:, 0] > self.x
        newly = crossed & ~cell.done
        H.finished = getattr(H, "finished", 0) + int(newly.sum())
        cell.done = cell.done | crossed
        part = H.level("particle")
        dead = cell.done[part.parent]
        if dead.any():
            part.mass = torch.where(dead, torch.zeros_like(part.mass), part.mass)  # stop blocking
            part.state[dead, 0] = -1.0                                             # park off-domain (disappear)
            part.state[dead, 1] = -1.0
            part.state[dead, 2:4] = 0.0
            if hasattr(part, "C"):
                part.C[dead] = 0.0
        return {}

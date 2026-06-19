"""gravity -- a uniform body force at the cell level.

Applies the same acceleration to every selected cell and RETURNS it as a delta
`{cell: a}`. In an MPM scene the cell is a derived aggregate (its position is the
centroid of its particles), so the engine does not integrate it; instead the
`mls_mpm_mechanics` operator reads the cell's accumulated delta and feeds it into
the substep as the external body force `a_ext`. So gravity is an ordinary operator
-- it does not touch `pos`/`vel`, it just contributes a force.

The *bounce* of a dropped soft body is NOT a force here: it is the emergent elastic
response when the body compresses against the reflective floor (grid velocity into a
wall is clamped to zero, the fixed-corotated stress springs it back). Stiffer
`youngs` -> livelier bounce; `drag` on the mpm operator sets how fast it decays.

Default direction is -y (down). Override `gx`/`gy` for an incline or sideways pull
(e.g. a tilted-gravity slosh).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("gravity", level="cell", kind="lateral")
class GravityOperator(Lateral):
    # PREDICTION is intentionally None: gravity emits a force the MPM substep consumes,
    # it is NOT integrated on the cell (the cell is a centroid readout). So `cell` never
    # enters H.predict and the engine never advects it under gravity.
    PARAM_ROLES = {"g": "gravity_magnitude", "gx": "gravity_x", "gy": "gravity_y"}
    MECHANISM_TAGS = ["body_force", "uniform_acceleration"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 10.0))            # magnitude (world units / time^2)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        dev = cell.state.device
        accel = torch.zeros(cell.n, 2, device=dev)
        accel[:, 0] = self.gx
        accel[:, 1] = self.gy
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {cell.name: accel}

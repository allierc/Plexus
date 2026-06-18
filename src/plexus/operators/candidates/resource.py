"""graze: Exchange operator, cell <- field (consumption).

Each (masked) cell removes chemical from the field at its position, capped at
what is actually present (so the field never goes negative and a cell cannot eat
more than the local stock). The consumed amount accumulates in `H.harvested`,
the optimization objective for grazing-style scenarios -- exactly the way
`forage` accumulates `H.food_delivered`.

This is also the half of the *self-generated gradient* mechanism (Tweedy &
Insall, Science 2020): cells deplete a diffusible attractant, and that local
depletion -- together with field diffusion and no-flux walls -- *creates* the
gradient that `sense` then climbs. No engine change: one registered operator.
"""

from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("graze", level="cell", kind="exchange")
class GrazeOperator(Exchange):
    REQUIRES_PARAMS = ["from", "rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params.get("rate", 1.0))   # consumption rate (>0 eats the field)
        self.field_name = params.get("from")

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        fld = H.fields[self.field_name]
        avail = fld.sample(pos).clamp(min=0.0)                       # stock at each cell
        want = torch.full_like(avail, self.rate * fld.dt)
        take = torch.minimum(avail, want)                            # cannot eat more than present
        if mask is not None:
            take = take * mask.float()
        fld.scatter(pos, -take)                                      # remove from the field
        H.harvested = getattr(H, "harvested", 0.0) + float(take.sum())
        return {}                                                    # no acceleration

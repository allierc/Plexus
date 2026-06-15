"""forage: Lateral operator @ cell. The pick-up / drop-off state machine.

Each cell carries a boolean `loaded`. In the food region an unloaded cell picks
up (loaded:=True); in the home region a loaded cell drops off (loaded:=False) and
increments H.food_delivered (the optimization objective). The loaded flag is what
the state-gated `sense` operators read to switch a cell between seeking food and
returning home. Returns no acceleration -- it only mutates state.
"""

from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("forage", level="cell", kind="lateral")
class ForageOperator(Lateral):
    REQUIRES_PARAMS = ["food", "home"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.food = params.get("food")     # [x0,y0,x1,y1]
        self.home = params.get("home")

    @staticmethod
    def _inside(pos, r):
        return ((pos[:, 0] >= r[0]) & (pos[:, 0] <= r[2]) &
                (pos[:, 1] >= r[1]) & (pos[:, 1] <= r[3]))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        if not hasattr(cell, "loaded"):
            cell.loaded = torch.zeros(cell.n, dtype=torch.bool, device=pos.device)
        in_food = self._inside(pos, self.food)
        in_home = self._inside(pos, self.home)
        H.food_delivered = getattr(H, "food_delivered", 0) + int((in_home & cell.loaded).sum())
        cell.loaded = torch.where(in_food, torch.ones_like(cell.loaded), cell.loaded)
        cell.loaded = torch.where(in_home, torch.zeros_like(cell.loaded), cell.loaded)
        return {}

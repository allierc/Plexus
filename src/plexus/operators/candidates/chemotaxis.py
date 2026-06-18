"""Exchange operators (set <-> field).

secrete : object -> field. Every (masked) cell deposits chemical at its position.
sense   : field -> object. (Masked) cells read the chemical gradient and get a
          chemotactic acceleration up-gradient.

The `mask` argument is the operator's selector, computed by the engine:
secrete's `on: cell` -> all cells; sense's `on: cell[type=soft]` -> soft only.
That single mask is how "the field guides only the first population" is expressed.
"""

from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("secrete", level="cell", kind="exchange")
class SecreteOperator(Exchange):
    REQUIRES_PARAMS = ["to", "rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params.get("rate", 1.0))
        self.field_name = params.get("to")
        # optional recency-discounted deposit (ant recruitment): if runout>0, a cell's deposit
        # fades linearly to 0 over `runout` ticks since it last left a goal (read from the
        # clock the `forage` operator maintains as cell.t_since_goal). Default 0 -> constant rate.
        self.runout = float(params.get("runout", 0.0))

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        fld = H.fields[self.field_name]
        amount = torch.full((pos.shape[0],), self.rate * fld.dt, device=pos.device)
        if self.runout > 0 and hasattr(cell, "t_since_goal"):
            recency = (1.0 - cell.t_since_goal / self.runout).clamp(min=0.0, max=1.0)
            amount = amount * recency
        if mask is not None:
            amount = amount * mask.float()
        fld.scatter(pos, amount)                  # side effect on the field
        return {}                                 # no acceleration


@register_operator("sense", level="cell", kind="exchange")
class SenseOperator(Exchange):
    REQUIRES_PARAMS = ["from", "gain"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.gain = float(params.get("gain", 1.0))
        self.field_name = params.get("from")

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        fld = H.fields[self.field_name]
        accel = self.gain * fld.gather_grad(pos)  # [N,2] up-gradient
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {"cell": accel}

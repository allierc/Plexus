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

from tissue_graph.models.base import Exchange
from tissue_graph.models.registry import register_operator


@register_operator("secrete", level="cell", kind="exchange")
class SecreteOperator(Exchange):
    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params.get("rate", 1.0))
        self.field_name = params.get("to")

    def forward(self, H, mask=None):
        cell = H.level("cell")
        pos = cell.state[:, :2]
        fld = H.fields[self.field_name]
        amount = torch.full((pos.shape[0],), self.rate * fld.dt, device=pos.device)
        if mask is not None:
            amount = amount * mask.float()
        fld.scatter(pos, amount)                  # side effect on the field
        return {}                                 # no acceleration


@register_operator("sense", level="cell", kind="exchange")
class SenseOperator(Exchange):
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

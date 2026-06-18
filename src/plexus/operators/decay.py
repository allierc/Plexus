"""decay -- field -> field. Linear evaporation of a scalar field.

c <- max(0, c - k*dt). Without decay the deposited trail would only accumulate;
decay is what lets unused filaments fade so the network coarsens (stigmergy).
Writes the field in place; returns {}.
"""
from __future__ import annotations

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("decay", level="cell", kind="exchange")
class Decay(Exchange):
    REQUIRES_PARAMS = ["to"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("to")
        self.rate = float(params.get("rate", 0.012))    # evaporation per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        dt = float(getattr(H.config, "dt", 1.0))
        fld.grid = (fld.grid - self.rate * dt).clamp(min=0.0)
        return {}

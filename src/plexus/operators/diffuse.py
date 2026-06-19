"""diffuse -- field -> field. One discrete diffusion step on a scalar field.

A per-channel 3x3 box-blur lerp: c <- (1-w)*c + w*mean_3x3(c), with w = saturate(
rate*dt). mean_3x3(c) - c is a discrete Laplacian, so this is one explicit step of
dc/dt = D nabla^2 c (the shader's `Diffuse`, edge-clamped). Writes the field in
place; returns {}.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("diffuse", level="field", kind="field")
class Diffuse(FieldUpdate):
    """field -> field: acts on the field named by `at:` (no set involved)."""

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # the field at `at:`
        self.rate = float(params.get("rate", 0.35))     # diffusion weight per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, nx, ny]
        dt = float(getattr(H.config, "dt", 1.0))
        gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode="replicate")    # edge-clamp
        blur = Fnn.avg_pool2d(gp, 3, stride=1).squeeze(0)              # 3x3 mean, same size
        dw = min(max(self.rate * dt, 0.0), 1.0)                        # saturate(rate*dt)
        fld.grid = g * (1.0 - dw) + blur * dw
        return {}

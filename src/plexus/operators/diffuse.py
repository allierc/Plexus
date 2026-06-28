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

    SUPPORTED_DIMS = [2, 3]                     # 3x3 (2D) / 3x3x3 (3D) box-blur step

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # the field at `at:`
        self.rate = float(params.get("rate", 0.35))     # diffusion weight per unit time

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, *shape]
        dt = float(getattr(H.config, "dt", 1.0))
        # periodic field -> wrap the blur across the seam (`circular`); else edge-clamp.
        pmode = "circular" if getattr(fld, "periodic", False) else "replicate"
        if g.dim() == 3:                                               # 2D field [C, nx, ny]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool2d(gp, 3, stride=1).squeeze(0)             # 3x3 mean, same size
        else:                                                         # 3D field [C, nx, ny, nz]
            gp = Fnn.pad(g.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode=pmode)
            blur = Fnn.avg_pool3d(gp, 3, stride=1).squeeze(0)            # 3x3x3 mean, same size
        dw = min(max(self.rate * dt, 0.0), 1.0)                        # saturate(rate*dt)
        fld.grid = g * (1.0 - dw) + blur * dw
        return {}

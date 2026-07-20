"""diffuse -- field -> field. One discrete diffusion step on a scalar field.

A per-channel 3x3 box-blur lerp: c <- (1-w)*c + w*mean_3x3(c), with w = saturate(
rate*dt). mean_3x3(c) - c is a discrete Laplacian, so this is one explicit step of
dc/dt = D nabla^2 c (the shader's `Diffuse`, edge-clamped). Writes the field in
place; returns {}.
"""
from __future__ import annotations

import math

import torch
import torch.fft as fft
import torch.nn.functional as Fnn

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="finite_difference")
class Diffuse(FieldUpdate):
    """field -> field: acts on the field named by `at:` (no set involved).

    The `finite_difference` implementation of the `diffuse` contract: a 3x3 box-blur
    lerp (an explicit Laplacian step). `spectral` below is a second implementation of
    the SAME contract -- select it with `{op: diffuse, at: chemical, implementation:
    spectral}`; both advance dc/dt = D nabla^2 c one step, differing only in numerics."""

    EMIT = None                                # field->field: writes the grid in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]                     # 3x3 (2D) / 3x3x3 (3D) box-blur step
    REQUIRES_PARAMS = []                        # no required params — target field comes from `at:` (engine-injected)
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "laplacian"]
    PARAM_ROLES = {"rate": "diffusion_rate"}
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."

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


@register_operator("diffuse", family="fields", set="field", kind="field",
                   implementation="spectral")
class DiffuseSpectral(FieldUpdate):
    """`spectral` implementation of the `diffuse` contract: one EXACT heat-kernel step of
    dc/dt = D nabla^2 c on a periodic grid -- c_hat *= exp(-D k^2 dt) in Fourier space.
    Same contract as the finite-difference box-blur (Diffuse); differs only in numerics
    (spectral accuracy, periodic boundary). Differentiable via torch.fft, so an inverse
    loop that filters `capabilities()` for `differentiable` keeps it."""

    EMIT = None
    SUPPORTED_DIMS = [2]                        # FFT step is 2D here (N-D is a follow-up)
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diffusion", "field_smoothing", "spectral"]
    PARAM_ROLES = {"rate": "diffusion_coefficient"}
    REFERENCE = "Fick, A. (1855). Ueber Diffusion. Ann. Phys. 170:59-86; Turing, A. M. (1952). Phil. Trans. R. Soc. B 237:37-72."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")
        self.rate = float(params.get("rate", 0.35))     # diffusion coefficient D

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        g = fld.grid                                                    # [C, nx, ny]
        if g.dim() != 3:
            raise NotImplementedError("diffuse:spectral is 2D-only (grid must be [C, nx, ny])")
        dt = float(getattr(H.config, "dt", 1.0))
        _, nx, ny = g.shape
        kx = fft.fftfreq(nx, device=g.device) * (2 * math.pi)          # radians / cell
        ky = fft.fftfreq(ny, device=g.device) * (2 * math.pi)
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2                        # [nx, ny]
        decay = torch.exp(-self.rate * dt * k2)                         # exact heat kernel
        ghat = fft.fftn(g, dim=(-2, -1))
        fld.grid = fft.ifftn(ghat * decay, dim=(-2, -1)).real
        return {}

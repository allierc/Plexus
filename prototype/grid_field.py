"""GridField: a continuum on a regular grid over the unit square [0,1]^2.

Implements the Field contract: scatter (object -> field, bilinear deposit),
gather_grad (field -> object, bilinear-sampled gradient), and step (self-update:
discrete Laplacian diffusion + decay). This is the only place grid numerics live.
"""

from __future__ import annotations

import torch

from tissue_graph.models.base import Field
from tissue_graph.models.registry import register_field


@register_field("grid", frame="eulerian")
class GridField(Field):
    def __init__(self, name, couples_to, res=96, diffusion=0.1, decay=0.0,
                 dt=0.05, device="cpu"):
        super().__init__(name, couples_to)
        self.res = int(res)
        self.D = float(diffusion)
        self.decay = float(decay)
        self.dt = float(dt)
        self.device = device
        self.register_buffer("grid", torch.zeros(self.res, self.res, device=device))

    # --- index helpers (bilinear) ---
    def _corners(self, pos):
        g = pos.clamp(0, 1 - 1e-6) * self.res          # [N,2] continuous grid coords
        i0 = g.floor().long()
        f = g - i0.float()
        i1 = (i0 + 1).clamp(max=self.res - 1)
        return i0, i1, f                                # f in [0,1]

    def scatter(self, pos, amount):                    # object -> field (deposit)
        i0, i1, f = self._corners(pos)
        wx0, wy0 = (1 - f[:, 0]), (1 - f[:, 1])
        wx1, wy1 = f[:, 0], f[:, 1]
        flat = self.grid.view(-1)
        for (ix, iy, w) in (
            (i0[:, 0], i0[:, 1], wx0 * wy0),
            (i1[:, 0], i0[:, 1], wx1 * wy0),
            (i0[:, 0], i1[:, 1], wx0 * wy1),
            (i1[:, 0], i1[:, 1], wx1 * wy1),
        ):
            flat.index_add_(0, ix * self.res + iy, w * amount)

    def gather_grad(self, pos):                        # field -> object (sampled gradient)
        gx, gy = torch.gradient(self.grid, spacing=1.0 / self.res)   # physical units
        i0, i1, f = self._corners(pos)
        def sample(field2d):
            a = field2d[i0[:, 0], i0[:, 1]]; b = field2d[i1[:, 0], i0[:, 1]]
            c = field2d[i0[:, 0], i1[:, 1]]; d = field2d[i1[:, 0], i1[:, 1]]
            wx, wy = f[:, 0], f[:, 1]
            return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
        return torch.stack([sample(gx), sample(gy)], dim=1)   # [N,2]

    def step(self):                                    # self-update: diffusion + decay
        c = self.grid
        # discrete Laplacian in grid-index units; stable while dt*D <= 0.25
        lap = (
            torch.roll(c, 1, 0) + torch.roll(c, -1, 0)
            + torch.roll(c, 1, 1) + torch.roll(c, -1, 1) - 4 * c
        )
        self.grid = (c + self.dt * (self.D * lap - self.decay * c)).clamp(min=0.0)

"""GridField: a continuum on a regular grid over the world rectangle [0,W]x[0,1].

Implements the Field contract: scatter (object -> field, bilinear deposit),
gather_grad (field -> object, bilinear-sampled gradient), and step (self-update:
discrete Laplacian diffusion + decay). This is the only place grid numerics live.

The world may be wider than tall (W = `width` >= 1, e.g. a longitudinal maze). The
grid uses SQUARE cells of size dx = 1/ny, so it is [nx, ny] with nx = round(W*ny).
For the default W=1 this reduces exactly to the old square ny x ny grid.
"""

from __future__ import annotations

import torch

from plexus.models.base import Field
from plexus.models.registry import register_field


@register_field("grid", frame="eulerian")
class GridField(Field):
    def __init__(self, name, couples_to, res=96, diffusion=0.1, decay=0.0,
                 dt=0.05, device="cpu", walls=None, source=None, source_rate=0.0,
                 periodic=False, width=1.0):
        super().__init__(name, couples_to)
        self.width = float(width)
        self.ny = int(res)
        self.nx = int(round(self.width * self.ny))         # square cells dx = 1/ny
        self.res = self.ny                                 # back-compat alias (square cells)
        self.dx = 1.0 / self.ny
        self.D = float(diffusion)
        self.decay = float(decay)
        self.dt = float(dt)
        self.device = device
        self.source_rate = float(source_rate)
        self.periodic = bool(periodic)
        self.register_buffer("grid", torch.zeros(self.nx, self.ny, device=device))
        self.register_buffer("walls", walls if walls is not None
                             else torch.zeros(self.nx, self.ny, dtype=torch.bool, device=device))
        self.register_buffer("source", source if source is not None
                             else torch.zeros(self.nx, self.ny, dtype=torch.bool, device=device))

    def equilibrate(self, n):                              # pre-solve a steady gradient
        for _ in range(n):
            self.step()

    # --- index helpers (bilinear); grid coords g = pos / dx = pos * ny ---
    def _corners(self, pos):
        p = torch.nan_to_num(pos, nan=0.5)
        gx = (torch.remainder(p[:, 0], self.width) if self.periodic
              else p[:, 0].clamp(0, self.width - 1e-6)) * self.ny
        gy = (torch.remainder(p[:, 1], 1.0) if self.periodic
              else p[:, 1].clamp(0, 1 - 1e-6)) * self.ny
        if self.periodic:
            i0x = gx.floor().long() % self.nx; fx = gx - gx.floor(); i1x = (i0x + 1) % self.nx
            i0y = gy.floor().long() % self.ny; fy = gy - gy.floor(); i1y = (i0y + 1) % self.ny
        else:
            i0x = gx.floor().long().clamp(0, self.nx - 1); fx = gx - i0x.float(); i1x = (i0x + 1).clamp(max=self.nx - 1)
            i0y = gy.floor().long().clamp(0, self.ny - 1); fy = gy - i0y.float(); i1y = (i0y + 1).clamp(max=self.ny - 1)
        return (i0x, i0y), (i1x, i1y), (fx, fy)

    def scatter(self, pos, amount):                        # object -> field (deposit)
        (i0x, i0y), (i1x, i1y), (fx, fy) = self._corners(pos)
        wx0, wy0 = (1 - fx), (1 - fy)
        flat = self.grid.view(-1)
        for (ix, iy, w) in (
            (i0x, i0y, wx0 * wy0),
            (i1x, i0y, fx * wy0),
            (i0x, i1y, wx0 * fy),
            (i1x, i1y, fx * fy),
        ):
            flat.index_add_(0, ix * self.ny + iy, w * amount)

    def sample(self, pos):                                 # bilinear field value at pos
        (i0x, i0y), (i1x, i1y), (fx, fy) = self._corners(pos)
        g = self.grid
        a = g[i0x, i0y]; b = g[i1x, i0y]; c = g[i0x, i1y]; d = g[i1x, i1y]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy

    def gather_grad(self, pos):                            # field -> object (sampled gradient)
        if self.periodic:                                  # toroidal central difference
            g = self.grid
            gx = (torch.roll(g, -1, 0) - torch.roll(g, 1, 0)) * (0.5 / self.dx)
            gy = (torch.roll(g, -1, 1) - torch.roll(g, 1, 1)) * (0.5 / self.dx)
        else:
            gx, gy = torch.gradient(self.grid, spacing=self.dx)        # physical units
        (i0x, i0y), (i1x, i1y), (fx, fy) = self._corners(pos)

        def samp(field2d):
            a = field2d[i0x, i0y]; b = field2d[i1x, i0y]; c = field2d[i0x, i1y]; d = field2d[i1x, i1y]
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
        return torch.stack([samp(gx), samp(gy)], dim=1)               # [N,2]

    def step(self):                                        # self-update: diffusion + decay (+ walls, source)
        c = self.grid
        w = self.walls
        lap = c.new_zeros(c.shape)                          # no-flux at walls
        for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            cn = torch.roll(c, d, ax)
            wn = torch.roll(w, d, ax)
            cn = torch.where(wn, c, cn)                     # wall neighbour -> no flux
            lap = lap + (cn - c)
        g = (c + self.dt * (self.D * lap - self.decay * c)).clamp(min=0.0)
        if self.source_rate > 0:
            g = g + self.dt * self.source_rate * self.source.float()
        g = torch.where(w, torch.zeros_like(g), g)          # walls hold no chemical
        self.grid = g

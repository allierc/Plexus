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
                 dt=0.05, device="cpu", walls=None, source=None, source_rate=0.0):
        super().__init__(name, couples_to)
        self.res = int(res)
        self.D = float(diffusion)
        self.decay = float(decay)
        self.dt = float(dt)
        self.device = device
        self.source_rate = float(source_rate)
        self.register_buffer("grid", torch.zeros(self.res, self.res, device=device))
        # optional static masks (maze): walls block diffusion, source injects chemical
        self.register_buffer("walls", walls if walls is not None
                             else torch.zeros(self.res, self.res, dtype=torch.bool, device=device))
        self.register_buffer("source", source if source is not None
                             else torch.zeros(self.res, self.res, dtype=torch.bool, device=device))

    def equilibrate(self, n):                           # pre-solve the maze gradient
        for _ in range(n):
            self.step()

    # --- index helpers (bilinear) ---
    def _corners(self, pos):
        g = torch.nan_to_num(pos, nan=0.5).clamp(0, 1 - 1e-6) * self.res   # [N,2] grid coords
        i0 = g.floor().long().clamp(0, self.res - 1)
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

    def sample(self, pos):                             # bilinear field value at pos
        i0, i1, f = self._corners(pos)
        wx, wy = f[:, 0], f[:, 1]
        g = self.grid
        a = g[i0[:, 0], i0[:, 1]]; b = g[i1[:, 0], i0[:, 1]]
        c = g[i0[:, 0], i1[:, 1]]; d = g[i1[:, 0], i1[:, 1]]
        return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy

    def gather_grad(self, pos):                        # field -> object (sampled gradient)
        gx, gy = torch.gradient(self.grid, spacing=1.0 / self.res)   # physical units
        i0, i1, f = self._corners(pos)
        def sample(field2d):
            a = field2d[i0[:, 0], i0[:, 1]]; b = field2d[i1[:, 0], i0[:, 1]]
            c = field2d[i0[:, 0], i1[:, 1]]; d = field2d[i1[:, 0], i1[:, 1]]
            wx, wy = f[:, 0], f[:, 1]
            return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
        return torch.stack([sample(gx), sample(gy)], dim=1)   # [N,2]

    def step(self):                                    # self-update: diffusion + decay (+ walls, source)
        c = self.grid
        w = self.walls
        # no-flux at walls: a wall neighbour contributes no gradient (treated as == c)
        lap = c.new_zeros(c.shape)
        for d, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            cn = torch.roll(c, d, ax)
            wn = torch.roll(w, d, ax)
            cn = torch.where(wn, c, cn)               # wall neighbour -> no flux
            lap = lap + (cn - c)
        g = (c + self.dt * (self.D * lap - self.decay * c)).clamp(min=0.0)
        if self.source_rate > 0:
            g = g + self.dt * self.source_rate * self.source.float()
        g = torch.where(w, torch.zeros_like(g), g)    # walls hold no chemical
        self.grid = g

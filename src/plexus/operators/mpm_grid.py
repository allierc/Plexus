"""mpm_grid -- the Eulerian background grid FIELD + the shared transfer kernel for the
decomposed MLS-MPM operators (mpm_strain / p2g / mpm_grid_update / g2p).

The grid is a FIELD, not an entity: transient scratch rebuilt every substep. p2g
scatters mass/momentum/colour into it, mpm_grid_update solves on it, g2p reads it back.
The B-spline weight helper lives here because both p2g (scatter) and g2p (gather)
recompute it from the (pre-advection) particle positions -- so no per-substep scratch
needs threading and the numbers match the `mls_mpm_mechanics` oracle.
"""
from __future__ import annotations

import torch

from plexus.models.base import Field
from plexus.models.registry import register_field

# 3x3 stencil offsets, built once (CPU constant; moved to device in each operator)
_OFFSETS = torch.tensor([[i, j] for i in range(3) for j in range(3)], dtype=torch.float32)


@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid on [0,width]x[0,1] with square cells dx = 1/n_grid.
    Channels: m (mass), mv (momentum), c (liquid colour for CSF), v (velocity). Pure
    scratch: p2g zeroes + scatters into it each substep, grid_update solves on it, g2p
    reads it back."""

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, device="cpu", **kw):
        super().__init__(name)
        self.ny = int(n_grid)
        self.nx = int(round(float(width) * self.ny))
        self.width = float(width)
        self.dx = 1.0 / self.ny
        self.inv_dx = float(self.ny)
        n = self.nx * self.ny
        self.register_buffer("m", torch.zeros(n, device=device))
        self.register_buffer("mv", torch.zeros(n, 2, device=device))
        self.register_buffer("c", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(n, 2, device=device))

    @property
    def grid(self):                                  # [C,nx,ny] view for the recorder (mass density)
        return self.m.view(1, self.nx, self.ny)


def bspline(X, inv_dx, offsets, nx, ny, periodic):
    """Quadratic B-spline weights of each particle over its 3x3 grid stencil.
    Returns (fx, weight[N,9], flat[N*9]) -- identical to the oracle's P2G block."""
    base = (X * inv_dx - 0.5).floor().long()
    fx = X * inv_dx - base.float()
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)                 # [N,3,2]
    oi = offsets[:, 0].long(); oj = offsets[:, 1].long()
    weight = w[:, oi, 0] * w[:, oj, 1]                             # [N,9]
    gpos = base[:, None, :] + offsets.long()[None]
    if periodic:
        gpos = torch.stack([gpos[..., 0] % nx, gpos[..., 1] % ny], dim=-1)
    else:
        gpos = torch.stack([gpos[..., 0].clamp(0, nx - 1), gpos[..., 1].clamp(0, ny - 1)], dim=-1)
    flat = (gpos[..., 0] * ny + gpos[..., 1]).reshape(-1)
    return fx, weight, flat


def sub_dt(H, fallback):
    """The current substep dt: the schedule's `{substep: N, dt}` loop sets `H.sub_dt`;
    fall back to the operator's own `dt_sub` param if run outside a substep loop."""
    sd = getattr(H, "sub_dt", None)
    return float(sd if sd is not None else fallback)

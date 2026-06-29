"""mpm_grid -- the Eulerian background grid FIELD + the shared transfer kernel for the
decomposed MLS-MPM operators (mpm_strain / p2g / mpm_grid_update / g2p).

The grid is a FIELD, not an entity: transient scratch rebuilt every substep. p2g
scatters mass/momentum/colour into it, mpm_grid_update solves on it, g2p reads it back.
The B-spline weight helper lives here because both p2g (scatter) and g2p (gather)
recompute it from the (pre-advection) particle positions -- so no per-substep scratch
needs threading and the numbers match the `mls_mpm_mechanics` oracle.

Dimension-generic (the dimension contract): the grid is `dim`-dimensional -- a
[nx, ny] square grid in 2D, an [nx, ny, nz] cube in 3D -- and the B-spline kernel
spans the 3^dim stencil. The 2D path is bit-identical to the original (3x3 stencil,
`gx*ny + gy` row-major flatten); the 3D path mirrors MPM_pytorch's MPM_3D (27-cell
stencil, `x*ny*nz + y*nz + z`). axis 0 spans the world width, the others span 1.
"""
from __future__ import annotations

import itertools

import torch

from plexus.models.base import Field
from plexus.models.registry import register_field


def stencil_offsets(dim: int, device="cpu") -> torch.Tensor:
    """The 3^dim quadratic-B-spline stencil offsets, row-major (last axis fastest).
    2D -> [9,2] == `[[i,j] for i in 0..2 for j in 0..2]`; 3D -> [27,3] (matches the
    MPM_3D offset ordering: idx//9, (idx%9)//3, idx%3)."""
    return torch.tensor(list(itertools.product(range(3), repeat=dim)),
                        dtype=torch.float32, device=device)


# 2D stencil kept as a module constant for back-compat (p2g/g2p now build per-dim)
_OFFSETS = stencil_offsets(2)


@register_field("mpm_grid")
class MPMGrid(Field):
    """MLS-MPM background grid on [0,width]x[0,1](x[0,1]) with square cells dx = 1/n_grid.
    Channels: m (mass), mv (momentum [.,dim]), c (liquid colour for CSF), v (velocity
    [.,dim]). Pure scratch: p2g zeroes + scatters into it each substep, grid_update
    solves on it, g2p reads it back."""

    RECORD = False                                   # transient scratch -- not recorded/rendered

    def __init__(self, name, width=1.0, n_grid=128, dim=2, device="cpu", **kw):
        super().__init__(name)
        self.dim = int(dim)
        self.ny = int(n_grid)                        # cells per unit length (axes 1..)
        self.nx = int(round(float(width) * self.ny)) # axis 0 spans the world width
        self.width = float(width)
        self.dx = 1.0 / self.ny
        self.inv_dx = float(self.ny)
        if self.dim == 2:
            self.shape = (self.nx, self.ny)
        else:                                        # 3D cube: axes 1,2 span [0,1]
            self.nz = self.ny
            self.shape = (self.nx, self.ny, self.nz)
        n = 1
        for s in self.shape:
            n *= s
        self.n_cells = n
        self.register_buffer("m", torch.zeros(n, device=device))
        self.register_buffer("mv", torch.zeros(n, self.dim, device=device))
        self.register_buffer("c", torch.zeros(n, device=device))
        self.register_buffer("v", torch.zeros(n, self.dim, device=device))

    @property
    def grid(self):                                  # [1,*shape] view for the recorder (mass density)
        return self.m.view((1,) + self.shape)


def bspline(X, inv_dx, offsets, shape, periodic):
    """Quadratic B-spline weights of each particle over its 3^dim grid stencil.
    Returns (fx [N,D], weight [N,S], flat [N*S]) where S = 3^D. Dimension-generic; the
    2D call reduces bit-identically to the original `w[:,oi,0]*w[:,oj,1]`, `gx*ny+gy`."""
    D = X.shape[1]
    base = (X * inv_dx - 0.5).floor().long()                          # [N,D]
    fx = X * inv_dx - base.float()                                    # [N,D]
    w = torch.stack([0.5 * (1.5 - fx) ** 2,
                     0.75 - (fx - 1) ** 2,
                     0.5 * (fx - 0.5) ** 2], dim=1)                   # [N,3,D]
    oidx = offsets.long()                                            # [S,D]
    weight = torch.ones(X.shape[0], offsets.shape[0], device=X.device)
    for k in range(D):                                               # prod_k w[:, o_k, k]
        weight = weight * w[:, oidx[:, k], k]
    gpos = base[:, None, :] + oidx[None]                             # [N,S,D]
    comps = []
    for k in range(D):
        comps.append(gpos[..., k] % shape[k] if periodic
                     else gpos[..., k].clamp(0, shape[k] - 1))
    flat = comps[0]                                                  # row-major flatten over `shape`
    for k in range(1, D):
        flat = flat * shape[k] + comps[k]
    return fx, weight, flat.reshape(-1)


def sub_dt(H, fallback):
    """The current substep dt: the schedule's `{substep: N, dt}` loop sets `H.sub_dt`;
    fall back to the operator's own `dt_sub` param if run outside a substep loop."""
    sd = getattr(H, "sub_dt", None)
    return float(sd if sd is not None else fallback)

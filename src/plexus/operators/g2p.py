"""g2p (mpm_grid -> particle): the MLS-MPM grid-to-particle gather + advection.

Gathers the grid velocity back onto particles (new velocity + new affine matrix C),
applies the inelastic wall-contact restitution and the CFL velocity cap, then advects
the particle positions. Recomputes the B-spline weights from the (pre-advection)
positions so they match p2g exactly. Advection is integrated state, so this opts out
of the engine guard (MAY_MUTATE_INTEGRATED_STATE) -- the substep loop owns it, like the
oracle. Step 4 of the decomposed MLS-MPM (oracle: `mls_mpm_mechanics`).
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator
from plexus.operators.mpm_grid import _OFFSETS, bspline, sub_dt


@register_operator("g2p", level="particle", kind="exchange")
class G2P(Exchange):
    MAY_MUTATE_INTEGRATED_STATE = True             # advects pos/vel inside the substep (like the oracle)
    MECHANISM_TAGS = ["grid_to_particle", "advection"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.frm = params.get("from", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.wall_damp = float(params.get("wall_damp", 1.0))
        self.wall_contact = float(params.get("wall_contact", 0.04))
        self.vmax = float(params.get("vmax", 1e9))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.frm); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        nx, ny, inv_dx, dx = g.nx, g.ny, g.inv_dx, g.dx
        periodic = bool(getattr(H, "periodic", False))
        width = float(getattr(H, "world_width", 1.0))
        offsets = _OFFSETS.to(dev)
        X, V = p.get("pos"), p.get("vel")
        fx, weight, flat = bspline(X, inv_dx, offsets, nx, ny, periodic)
        gvn = g.v[flat].view(p.n, 9, 2)
        new_V = (weight[..., None] * gvn).sum(1)
        dpos_grid = offsets[None] - fx[:, None, :]
        new_C = 4 * inv_dx * (weight[..., None, None] * (gvn[..., :, None] @ dpos_grid[..., None, :])).sum(1)
        new_V = torch.nan_to_num(new_V)
        if self.wall_damp != 1.0 and not periodic:                 # inelastic wall contact (solids)
            cb = self.wall_contact
            near = ((X[:, 0] < cb) | (X[:, 0] > width - cb) | (X[:, 1] < cb) | (X[:, 1] > 1.0 - cb))
            liquid = getattr(p, "is_liquid", None)
            if liquid is not None:
                near = near & ~liquid
            new_V = torch.where(near[:, None], new_V * self.wall_damp, new_V)
        sp = new_V.norm(dim=1, keepdim=True).clamp(min=1e-9)
        vmax = min(self.vmax, 0.4 * dx / dt)                       # CFL velocity cap
        new_V = new_V * (sp.clamp(max=vmax) / sp)
        new_C = torch.nan_to_num(new_C)
        Xn = torch.nan_to_num(X + dt * new_V, nan=0.5)
        if periodic:
            Xn = torch.stack([torch.remainder(Xn[:, 0], width), torch.remainder(Xn[:, 1], 1.0)], dim=1)
        else:
            Xn = torch.stack([Xn[:, 0].clamp(2 * dx, width - 2 * dx), Xn[:, 1].clamp(2 * dx, 1 - 2 * dx)], dim=1)
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]; va, vb = p.state_schema["vel"]
        new[:, pa:pb] = Xn; new[:, va:vb] = new_V
        p.state = new
        p.C = new_C
        return {}

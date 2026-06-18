"""dicty_ops_mpm.py -- the source operator for the soft-MPM dicty arm.

inflow_mpm (structural @ cell): a SOURCE of cells entering the volume. Each tick wake ~`rate`
dormant cells; waking a cell = activate its `ppc` MPM particles as a fresh small disc at an
INDEPENDENT random position (cells streaming in), with reset MPM state (mass=p_mass0, F=I, C=0).
The MPM `mpm` operator then gives volume exclusion + deformation; chemotaxis pulls the new cell in.
Same dormant-buffer semantics as the point-cell `inflow`, but it activates a particle group.
"""
from __future__ import annotations

import math
import torch
from plexus.models.base import Operator
from plexus.models.registry import register_operator


@register_operator("inflow_mpm", level="cell", kind="structural")
class InflowMPM(Operator):
    REQUIRES_PARAMS = ["rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = float(params["rate"])
        self.region = params.get("region")              # optional [x0,y0,x1,y1]

    def forward(self, H, mask=None):
        cell = H.level("cell"); part = H.level("particle")
        dev = cell.state.device
        dormant = (~H.c_active).nonzero(as_tuple=True)[0]
        if dormant.numel() == 0 or self.rate <= 0:
            return {}
        base = int(math.floor(self.rate))
        extra = int(torch.rand(1, generator=H.rng, device=dev).item() < (self.rate - base))
        k = min(base + extra, dormant.numel())
        if k == 0:
            return {}
        ppc = H.ppc; rad = H.rad; W = float(getattr(H, "world_width", 1.0))
        new = dormant[:k]
        # independent entry positions
        ctr = torch.rand(k, 2, generator=H.rng, device=dev)
        if self.region is not None:
            x0, y0, x1, y1 = self.region
            ctr[:, 0] = x0 + ctr[:, 0] * (x1 - x0); ctr[:, 1] = y0 + ctr[:, 1] * (y1 - y0)
        else:
            ctr[:, 0] *= W
        cell.state[new, :2] = ctr; cell.state[new, 2:] = 0.0
        H.c_active[new] = True
        # particles of cell c are the contiguous block [c*ppc:(c+1)*ppc]
        pidx = (new[:, None] * ppc + torch.arange(ppc, device=dev)[None, :]).reshape(-1)
        npart = pidx.numel()
        r = torch.sqrt(torch.rand(npart, generator=H.rng, device=dev)) * rad
        th = torch.rand(npart, generator=H.rng, device=dev) * 2 * math.pi
        ctr_rep = ctr.repeat_interleave(ppc, dim=0)
        part.state[pidx, :2] = ctr_rep + torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)
        part.state[pidx, 2:] = 0.0
        part.mass[pidx] = H.p_mass0
        part.F[pidx] = torch.eye(2, device=dev)
        part.C[pidx] = 0.0
        part.mu[pidx] = H.mu0; part.la[pidx] = H.la0
        return {}

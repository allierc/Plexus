"""Operators for the growing/dividing soft cell — registered, spec-drivable.

Four operators on the two-level hierarchy (particle in cell), all real
`@register_operator` classes following the engine contract
`forward(self, H, mask) -> {level: delta}`:

  cohere    (broadcast) : particle pulled toward its parent cell's centroid
  repulse   (lateral)   : soft short-range particle-particle repulsion
  duplicate (structural): wake dormant particle slots  -> the cell grows
  divide    (structural): split a cell whose mass has DOUBLED, along its
                          principal axis -> two daughters

`cohere`/`repulse` return a particle-level acceleration (a delta). The two
structural ops mutate the hierarchy (occupancy `H.p_w`, `parent`, the active
cell count) and return `{}` — uniform with every other operator, so the engine
never special-cases them. Occupancy `w in {0,1}` on a fixed buffer is what lets a
cardinality-changing operator live inside a fixed-shape contract.
"""
from __future__ import annotations

import math
import torch

from plexus.models.base import Lateral, Broadcast, Operator
from plexus.models.registry import register_operator

EPS = 1e-6


@register_operator("cohere", level="particle", kind="broadcast")
class Cohere(Broadcast):
    """Broadcast the parent cell's centroid down to its particles and pull each
    particle toward it (keeps the cell round and together)."""
    REQUIRES_PARAMS = ["k"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.k = float(params["k"])

    def forward(self, H, mask=None):
        part, cell = H.level("particle"), H.level("cell")
        ppos = part.state[:, :2]
        centroid = cell.state[:, :2][part.parent]          # parent's centroid (down-lift)
        accel = -self.k * (ppos - centroid) * H.p_w[:, None]
        return {"particle": accel}


@register_operator("repulse", level="particle", kind="lateral")
class Repulse(Lateral):
    """Soft short-range repulsion among active particles (incompressibility ->
    the cell has area; daughters push apart)."""
    REQUIRES_PARAMS = ["r0", "k"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.r0 = float(params["r0"]); self.k = float(params["k"])

    def forward(self, H, mask=None):
        part = H.level("particle"); w = H.p_w
        pos = part.state[:, :2]; N = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        d = torch.sqrt((diff * diff).sum(2) + 1e-9)
        off = 1.0 - torch.eye(N, device=pos.device)
        mag = (self.k * (1.0 - d / self.r0)).clamp_min(0.0) * off * w[None, :]
        f = (mag[:, :, None] * diff / d[:, :, None]).sum(1)
        return {"particle": f * w[:, None]}


@register_operator("duplicate", level="particle", kind="structural")
class Duplicate(Operator):
    """Wake `rate` dormant particle slots, each next to a random active particle
    of the same cell (cell grows). Mutates occupancy; returns no delta."""
    REQUIRES_PARAMS = ["rate"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.rate = int(params["rate"]); self.r0 = float(params.get("r0", 0.013))

    def forward(self, H, mask=None):
        part = H.level("particle"); w = H.p_w
        dormant = (w < EPS).nonzero(as_tuple=True)[0]
        active = (w > EPS).nonzero(as_tuple=True)[0]
        k = min(self.rate, dormant.numel())
        if k == 0 or active.numel() == 0:
            return {}
        new = dormant[:k]
        par = active[torch.randint(active.numel(), (k,), generator=H.rng, device=w.device)]
        ang = torch.rand(k, generator=H.rng, device=w.device) * 2 * math.pi
        off = 0.5 * self.r0 * torch.stack([torch.cos(ang), torch.sin(ang)], 1)
        part.state[new, :2] = part.state[par, :2] + off
        part.state[new, 2:] = 0.0
        part.parent[new] = part.parent[par]
        w[new] = 1.0
        return {}


@register_operator("divide", level="cell", kind="structural")
class Divide(Operator):
    """Split any active cell whose mass has reached `ratio`x its birth mass, along
    the cell's principal axis: half its particles get a fresh cell id, both
    daughters are re-baselined and nudged apart. Mutates membership; no delta."""
    REQUIRES_PARAMS = ["ratio"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.ratio = float(params["ratio"]); self.push = float(params.get("push", 0.012))

    def forward(self, H, mask=None):
        part = H.level("particle"); cell = H.level("cell")
        w = H.p_w; par = part.parent; dev = w.device
        for c in [c for c in range(cell.n) if H.c_active[c]]:
            mem = (par == c) & (w > EPS)
            mc = float(w[mem].sum())
            if mc < self.ratio * H.cell_birth[c]:
                continue
            if H.c_active.sum() >= cell.n:                 # buffer full -> stop dividing
                break
            idx = mem.nonzero(as_tuple=True)[0]
            p = part.state[idx, :2]
            cen = p.mean(0)
            cov = ((p - cen).t() @ (p - cen)) / p.shape[0]
            axis = torch.linalg.eigh(cov)[1][:, -1]
            side = ((p - cen) @ axis) > 0
            newid = int((~H.c_active).nonzero(as_tuple=True)[0][0])   # first free cell slot
            par[idx[side]] = newid
            H.c_active[newid] = True
            m_side = float(w[idx[side]].sum())
            H.cell_birth[c] = mc - m_side; H.cell_birth[newid] = m_side
            part.state[idx[side], :2] += axis * self.push
            part.state[idx[~side], :2] -= axis * self.push
        return {}

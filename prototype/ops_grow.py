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
        # if the particle level carries MPM state, initialise the new slots too
        if hasattr(part, "mass"):
            part.mass[new] = H.p_mass0
        if hasattr(part, "F"):
            part.F[new] = torch.eye(2, device=w.device)
            part.C[new] = 0.0
        if hasattr(part, "mu"):
            part.mu[new] = part.mu[par]; part.la[new] = part.la[par]
        return {}


@register_operator("tension", level="particle", kind="lateral")
class Tension(Lateral):
    """Cortical surface tension: a particle on the cell BOUNDARY (few same-cell
    neighbours) is pulled inward toward the cell centroid; interior particles
    (full neighbourhood) feel nothing. Minimising the exposed surface rounds the
    cell -- the real force that makes a soft cell spherical, here fighting MPM's
    elastic shape-memory so divided daughters round up instead of staying wedges."""
    REQUIRES_PARAMS = ["strength", "radius"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.s = float(params["strength"]); self.r = float(params["radius"])
        self.nfull = float(params.get("nfull", 12.0))

    def forward(self, H, mask=None):
        part = H.level("particle"); cell = H.level("cell"); w = H.p_w
        pos = part.state[:, :2]
        diff = pos[:, None, :] - pos[None, :, :]
        d = torch.sqrt((diff * diff).sum(2) + 1e-9)
        same = (part.parent[:, None] == part.parent[None, :]).float()
        n = (((d < self.r) & (d > 1e-5)).float() * same * w[None, :]).sum(1)
        bw = (1.0 - n / self.nfull).clamp(0.0, 1.0)              # boundary weight
        cen = cell.state[:, :2][part.parent]
        return {"particle": self.s * bw[:, None] * (cen - pos) * w[:, None]}


@register_operator("tissue", level="cell", kind="lateral")
class Tissue(Lateral):
    """Inter-cell adhesion: each active cell is pulled toward the centroids of its
    active neighbours within `radius`. This is the attraction that holds the colony
    together as a tissue; MPM incompressibility + per-cell `cohere` fight it, so the
    cells pack instead of either dispersing or collapsing. Cell-level accel ->
    broadcast to the cell's particles by `mpm`."""
    REQUIRES_PARAMS = ["strength", "radius"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.s = float(params["strength"]); self.r = float(params["radius"])

    def forward(self, H, mask=None):
        cell = H.level("cell"); pos = cell.state[:, :2]
        act = H.c_active.float()
        diff = pos[None, :, :] - pos[:, None, :]              # j - i  -> toward neighbour
        d = torch.sqrt((diff * diff).sum(2) + 1e-9)
        W = ((d < self.r) & (d > 1e-5)).float() * act[None, :] * act[:, None]
        cnt = W.sum(1).clamp(min=1.0)[:, None]
        accel = self.s * (W[..., None] * diff).sum(1) / cnt
        return {"cell": accel * act[:, None]}


@register_operator("mitosis", level="cell", kind="structural")
class Mitosis(Operator):
    """Gradual, realistic division. When a cell's mass doubles it enters mitosis:
    over `frames` ticks it (i) ELONGATES along its principal axis (the two halves
    drift apart) and (ii) forms a CLEAVAGE FURROW (particles near the equator are
    pulled in perpendicular to the axis, pinching the waist); at the end the two
    halves become separate cells. Returns a per-particle accel (integrated by the
    engine) and relabels membership on completion."""
    REQUIRES_PARAMS = ["ratio", "frames"]

    def __init__(self, params, device="cpu"):
        super().__init__()
        self.ratio = float(params["ratio"]); self.frames = int(params["frames"])
        self.elong = float(params.get("elong", 0.15))
        self.furrow = float(params.get("furrow", 3.0))
        self.furrow_w = float(params.get("furrow_w", 0.018))

    def forward(self, H, mask=None):
        part = H.level("particle"); cell = H.level("cell")
        w = H.p_w; par = part.parent; dev = w.device
        if not hasattr(H, "cell_phase"):
            H.cell_phase = torch.zeros(cell.n, device=dev)
            H.cell_axis = torch.zeros(cell.n, 2, device=dev)
        accel = torch.zeros(part.n, 2, device=dev)
        for c in [c for c in range(cell.n) if H.c_active[c]]:
            mem = (par == c) & (w > EPS)
            mc = float(w[mem].sum())
            ph = float(H.cell_phase[c])
            if ph == 0.0 and mc >= self.ratio * H.cell_birth[c]:
                idx = mem.nonzero(as_tuple=True)[0]
                p = part.state[idx, :2]; cen = p.mean(0)
                cov = ((p - cen).t() @ (p - cen)) / p.shape[0]
                H.cell_axis[c] = torch.linalg.eigh(cov)[1][:, -1]
                H.cell_phase[c] = 1.0; ph = 1.0
            if ph > 0.0:
                idx = mem.nonzero(as_tuple=True)[0]
                axis = H.cell_axis[c]
                p = part.state[idx, :2]; cen = p.mean(0)
                rel = p - cen
                proj = rel @ axis                                   # along the spindle
                perp = rel - proj[:, None] * axis                   # equatorial offset
                s = torch.sign(proj)
                a = self.elong * s[:, None] * axis[None, :]         # elongate: halves apart
                eq = proj.abs() < self.furrow_w                     # cleavage furrow at the waist
                a = a - self.furrow * eq[:, None].float() * perp    # pinch inward
                accel[idx] += a
                H.cell_phase[c] = ph + 1.0
                if ph + 1.0 >= self.frames:                         # complete the split
                    if H.c_active.sum() < cell.n:
                        newid = int((~H.c_active).nonzero(as_tuple=True)[0][0])
                        side = proj > 0
                        par[idx[side]] = newid; H.c_active[newid] = True
                        m_side = float(w[idx[side]].sum())
                        H.cell_birth[c] = mc - m_side; H.cell_birth[newid] = m_side
                        H.cell_phase[newid] = 0.0
                    H.cell_phase[c] = 0.0
        return {"particle": accel}


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

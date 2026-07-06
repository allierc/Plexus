"""embryo_vertex_ops -- Plexus operator for the **Self-Propelled Voronoi / Active Vertex model**.

A strict-Plexus reproduction of the confluent-tissue vertex model — **D. Bi, X. Yang, M. C.
Marchetti & M. L. Manning, "Motility-Driven Glass and Jamming Transitions in Biological Tissues"
(PRX 2016)** and **D. Barton et al., "Active Vertex Model..." (PLoS Comput. Biol. 2017)**; PDFs
in `papers/zebrafish/`, and geometry inspired by the dual cell/vertex graph of `cell-gnn`. Unlike
the earlier point-agent embryos, here each cell has a real SHAPE: the tissue is the **Voronoi
tessellation** of the cell centres, and the mechanics come from a cell **shape energy**

    E = Σ_i [ K_A (A_i − A₀)² + K_P (P_i − P₀)² ]

where A_i, P_i are the cell's Voronoi area and perimeter. The dimensionless **target shape index**
p₀ = P₀/√A₀ controls a **rigidity transition** at p₀* ≈ 3.81: below it the tissue is a SOLID
(cells jam, no rearrangement), above it a FLUID (cells flow via T1 neighbour exchanges). T1s are
AUTOMATIC — they fall out of re-tessellating each step, no explicit rule. Self-propulsion (speed
v₀ along a slowly-rotating polarity) drives the tissue through the jamming/unjamming transition.

`vertex_tension` -- the shape-energy force on cell centres + self-propulsion; `kind=lateral`,
`EMIT=velocity` (overdamped). The Voronoi topology is retessellated each step (periodic Delaunay);
the force is the EXACT gradient of E, obtained by autodiff through differentiable circumcenters.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _circumcenters_np(P):
    """Circumcentre of each triangle. P: [M,3,2] -> [M,2] (numpy, for angular ordering)."""
    ax, ay = P[:, 0, 0], P[:, 0, 1]
    bx, by = P[:, 1, 0], P[:, 1, 1]
    cx, cy = P[:, 2, 0], P[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.stack([ux, uy], 1)


def _circumcenters_torch(P):
    """Differentiable circumcentre. P: [M,3,2] -> [M,2] (autodiff w.r.t. the triangle points)."""
    ax, ay = P[:, 0, 0], P[:, 0, 1]
    bx, by = P[:, 1, 0], P[:, 1, 1]
    cx, cy = P[:, 2, 0], P[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return torch.stack([ux, uy], 1)


def periodic_voronoi_rings(pos_np, L, N):
    """Periodic Voronoi of N cell centres in a box [0,L)^2 via a 3x3 tiling + Delaunay.

    Returns (tri [M,3] into the tiled point array, tiled_orig [9N] original cell id,
    tiled_shift [9N,2] image offset, ring_idx [N,Vmax] triangle index per cell's ordered vertex
    ring (padded by the last vertex), ok [N] valid-cell mask). Each cell's Voronoi polygon is the
    circumcentres of the Delaunay triangles incident to its CENTRAL image (tiled indices 0..N-1)."""
    shifts = np.array([[dx * L, dy * L] for dx in (0, -1, 1) for dy in (0, -1, 1)], dtype=np.float64)
    tiled_orig = np.tile(np.arange(N), 9)                      # central image first (0..N-1)
    tiled_shift = np.repeat(shifts, N, axis=0)
    tiled = pos_np[tiled_orig] + tiled_shift                   # [9N,2]
    tri = Delaunay(tiled, qhull_options="QJ").simplices        # [M,3]
    cc = _circumcenters_np(tiled[tri])                         # [M,2]
    central = tri < N                                          # central-image vertices
    ts_idx, k_idx = np.where(central)
    cells = tri[ts_idx, k_idx]                                 # [P] central cell id
    tri_id = ts_idx                                            # [P] triangle index
    order = np.argsort(cells, kind="stable")
    cells_s, tri_s = cells[order], tri_id[order]
    bounds = np.searchsorted(cells_s, np.arange(N + 1))
    rings, Vmax = [None] * N, 3
    for i in range(N):
        ts = tri_s[bounds[i]:bounds[i + 1]]
        if len(ts) >= 3:
            v = cc[ts]
            ang = np.arctan2(v[:, 1] - pos_np[i, 1], v[:, 0] - pos_np[i, 0])
            rings[i] = ts[np.argsort(ang)]
            Vmax = max(Vmax, len(ts))
    ring_idx = np.zeros((N, Vmax), dtype=np.int64)
    ok = np.zeros(N, dtype=np.float32)
    for i in range(N):
        r = rings[i]
        if r is not None:
            ring_idx[i, :len(r)] = r
            ring_idx[i, len(r):] = r[-1]                       # pad by last vertex (zero-length edges)
            ok[i] = 1.0
    return tri, tiled_orig, tiled_shift, ring_idx, ok


def cell_polygons(pos_np, L, N):
    """Voronoi polygon (vertex ring), area and perimeter for each central cell -- for rendering
    and diagnostics. Returns (polys: list of [k,2] arrays or None, area [N], perim [N], ok [N])."""
    tri, torig, tshift, ring_idx, ok = periodic_voronoi_rings(pos_np, L, N)
    tiled = pos_np[torig] + tshift
    cc = _circumcenters_np(tiled[tri])
    polys = []
    area = np.zeros(N); perim = np.zeros(N)
    for i in range(N):
        if not ok[i]:
            polys.append(None); continue
        v = cc[ring_idx[i]]
        keep = [0]
        for k in range(1, len(v)):
            if not np.allclose(v[k], v[keep[-1]]):
                keep.append(k)
        vv = v[keep]
        polys.append(vv)
        x, y = vv[:, 0], vv[:, 1]
        area[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perim[i] = np.sum(np.sqrt(((np.roll(vv, -1, 0) - vv) ** 2).sum(1)))
    return polys, area, perim, ok


def delaunay_neighbors(pos_np, L, N):
    """Set of unordered central cell-cell neighbour pairs (for T1 / rearrangement counting)."""
    tri, torig, _, _, _ = periodic_voronoi_rings(pos_np, L, N)
    orig = torig[tri]                                          # [M,3] original ids
    pairs = set()
    for a, b, c in orig:
        for u, v in ((a, b), (b, c), (a, c)):
            if u != v:
                pairs.add((u, v) if u < v else (v, u))
    return pairs


@register_operator("vertex_tension", level="cell", kind="lateral")
class VertexTension(Lateral):
    """Self-Propelled Voronoi shape-energy force + self-propulsion. Retessellates the Voronoi
    each step (automatic T1s), computes E = Σ K_A(A−A₀)² + K_P(P−P₀)² over cell polygons, and
    returns the overdamped velocity v = μ·(−∇E) + v₀·n̂ with a rotationally-diffusing polarity n̂."""
    SUPPORTED_DIMS = [2]
    EMIT = "velocity"
    MECHANISM_TAGS = ["vertex_model", "self_propelled_voronoi", "shape_energy",
                      "rigidity_transition", "T1_transition", "confluent_tissue", "morphogenesis"]
    PARAM_ROLES = {"p0": "target_shape_index", "v0": "self_propulsion_speed",
                   "Dr": "rotational_diffusion", "K_A": "area_stiffness", "K_P": "perimeter_stiffness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_A = float(params.get("K_A", 1.0))
        self.K_P = float(params.get("K_P", 1.0))
        self.A0 = float(params.get("A0", 1.0))
        self.p0 = float(params.get("p0", 3.85))               # target shape index (transition ~3.81)
        self.P0 = self.p0 * math.sqrt(self.A0)
        self.v0 = float(params.get("v0", 0.2))                # self-propulsion speed
        self.Dr = float(params.get("Dr", 1.0))                # rotational diffusion (1/persistence)
        self.mu = float(params.get("mu", 1.0))                # mobility
        self.dt = float(params.get("dt", 0.05))

    def _theta(self, lvl, N, dev):
        if not hasattr(lvl, "theta"):
            lvl.register_buffer("theta", 2 * math.pi * torch.rand(N, generator=getattr(lvl, "rng", None),
                                                                  device=dev))
        return lvl.theta

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")                             # [N,2] in [0,L)^2 (N = buffer)
        N = pos_full.shape[0]
        dev = pos_full.device
        L = float(H.world_size[0])
        v_full = torch.zeros_like(pos_full)
        # tessellate only LIVE cells (division wakes dormant buffer slots -> occ 0 must be excluded)
        live = lvl.occ > 0
        idx = live.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 4:
            return {self.at: v_full}
        pos_live = pos_full[idx]
        pos_np = pos_live.detach().cpu().numpy().astype(np.float64) % L
        tri, tiled_orig, tiled_shift, ring_idx, ok = periodic_voronoi_rings(pos_np, L, n)
        tri_t = torch.as_tensor(tri, device=dev)
        orig_t = torch.as_tensor(tiled_orig, device=dev, dtype=torch.long)
        shift_t = torch.as_tensor(tiled_shift, device=dev, dtype=pos_full.dtype)
        ring_t = torch.as_tensor(ring_idx, device=dev)
        ok_t = torch.as_tensor(ok, device=dev)
        with torch.enable_grad():
            pos = (pos_live.detach() % L).requires_grad_(True)
            tiled = pos[orig_t] + shift_t                     # [9n,2] differentiable
            cc = _circumcenters_torch(tiled[tri_t])           # [M,2]
            verts = cc[ring_t]                                # [n,Vmax,2]
            nxt = torch.roll(verts, -1, dims=1)
            cross = verts[..., 0] * nxt[..., 1] - nxt[..., 0] * verts[..., 1]
            area = 0.5 * cross.sum(1).abs()                   # [n]
            perim = (nxt - verts).norm(dim=-1).sum(1)         # [n]
            E = (self.K_A * (area - self.A0) ** 2 + self.K_P * (perim - self.P0) ** 2)
            E = (E * ok_t).sum()
            grad = torch.autograd.grad(E, pos)[0]
        F = torch.nan_to_num(-grad) * ok_t[:, None]           # shape-energy force on live centres
        # self-propulsion with rotational diffusion (polarity buffer over the full buffer)
        th = self._theta(lvl, N, dev)
        th = th + math.sqrt(2 * self.Dr * self.dt) * torch.randn(N, generator=getattr(H, "rng", None),
                                                                 device=dev)
        lvl.theta = th
        n_hat = torch.stack([torch.cos(th), torch.sin(th)], 1)[idx]
        v_full[idx] = self.mu * F + self.v0 * n_hat
        return {self.at: v_full}

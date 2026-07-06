"""embryo_vertex_3d_ops -- the Self-Propelled Voronoi tissue in **3D**.

The 3D generalisation of `vertex_tension` (Merkel & Manning, "A geometrically controlled rigidity
transition in a model for confluent 3D tissues", New J. Phys. 2018). Cells are points in a periodic
box; the tissue is their 3D Voronoi tessellation (cells = convex polyhedra). The shape energy is

    E = Σ_i [ K_V (V_i − V₀)² + K_S (S_i − S₀)² ]

with V_i, S_i the cell's Voronoi volume and surface area. The dimensionless target shape index
s0 = S0 / V0^(2/3) drives a rigidity transition at s0* ~= 5.41 (below -> solid, above -> fluid).

The force is the exact −∇E, by autodiff through differentiable **tetrahedron circumcentres** (the
3D Voronoi vertices). Each cell's polyhedron = the convex hull of the circumcentres of the Delaunay
tetrahedra incident to it; volume & surface come from the hull's triangulation (topology from scipy,
values differentiable in torch). `kind=lateral`, `EMIT=velocity` (overdamped) + 3D self-propulsion.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay, ConvexHull

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _tet_circumcenters_np(P):
    """Circumcentre of each tetrahedron. P: [M,4,3] -> [M,3] (numpy)."""
    a = P[:, 0]; u = P[:, 1] - a; v = P[:, 2] - a; w = P[:, 3] - a
    cvw = np.cross(v, w); cwu = np.cross(w, u); cuv = np.cross(u, v)
    denom = 2.0 * (u * cvw).sum(-1)
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    o = ((u * u).sum(-1)[:, None] * cvw + (v * v).sum(-1)[:, None] * cwu
         + (w * w).sum(-1)[:, None] * cuv) / denom[:, None]
    return a + o


def _tet_circumcenters_torch(P):
    """Differentiable circumcentre. P: [M,4,3] -> [M,3]."""
    a = P[:, 0]; u = P[:, 1] - a; v = P[:, 2] - a; w = P[:, 3] - a
    cvw = torch.cross(v, w, dim=-1); cwu = torch.cross(w, u, dim=-1); cuv = torch.cross(u, v, dim=-1)
    denom = 2.0 * (u * cvw).sum(-1)
    denom = torch.where(denom.abs() < 1e-9, torch.full_like(denom, 1e-9), denom)
    o = ((u * u).sum(-1, keepdim=True) * cvw + (v * v).sum(-1, keepdim=True) * cwu
         + (w * w).sum(-1, keepdim=True) * cuv) / denom[:, None]
    return a + o


def periodic_delaunay_3d(pos_np, L, n):
    """3D periodic Delaunay via a 27-image tiling. Returns (tetra [M,4] into the tiled array,
    tiled_orig [27n], tiled_shift [27n,3], and per central-cell lists of incident tetra indices)."""
    shifts = np.array([[dx * L, dy * L, dz * L]
                       for dx in (0, -1, 1) for dy in (0, -1, 1) for dz in (0, -1, 1)], dtype=np.float64)
    tiled_orig = np.tile(np.arange(n), 27)                    # central image first (0..n-1)
    tiled_shift = np.repeat(shifts, n, axis=0)
    tiled = pos_np[tiled_orig] + tiled_shift
    tetra = Delaunay(tiled, qhull_options="QJ").simplices     # [M,4]
    verts = tetra.reshape(-1)
    tet_id = np.repeat(np.arange(tetra.shape[0]), 4)
    central = verts < n
    cells, tets = verts[central], tet_id[central]
    order = np.argsort(cells, kind="stable")
    cells_s, tets_s = cells[order], tets[order]
    bounds = np.searchsorted(cells_s, np.arange(n + 1))
    incident = [tets_s[bounds[i]:bounds[i + 1]] for i in range(n)]
    return tetra, tiled_orig, tiled_shift, incident


def cell_polyhedra(pos_np, L, n):
    """Per central cell: convex-hull vertices of its Voronoi polyhedron, volume, surface, ok mask.
    For rendering + diagnostics. Returns (verts_list, vol[n], surf[n], ok[n])."""
    tetra, torig, tshift, incident = periodic_delaunay_3d(pos_np, L, n)
    tiled = pos_np[torig] + tshift
    cc = _tet_circumcenters_np(tiled[tetra])                  # [M,3]
    verts_list = [None] * n
    vol = np.zeros(n); surf = np.zeros(n); ok = np.zeros(n, np.float32)
    for i in range(n):
        pts = cc[incident[i]]
        if len(pts) < 4:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        verts_list[i] = pts[hull.vertices]
        vol[i] = hull.volume; surf[i] = hull.area; ok[i] = 1.0
    return verts_list, vol, surf, ok


def cell_faces(pos_np, L, n):
    """Per central cell: the triangulated faces of its Voronoi polyhedron (for 3D rendering) and
    its shape index s = S/V^(2/3). Returns (faces_list of [F,3,3] triangle arrays, s [K], ok count)."""
    tetra, torig, tshift, incident = periodic_delaunay_3d(pos_np, L, n)
    cc = _tet_circumcenters_np((pos_np[torig] + tshift)[tetra])
    faces, svals = [], []
    for i in range(n):
        inc = incident[i]
        if len(inc) < 4:
            continue
        pts = cc[inc]
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        faces.append(pts[hull.simplices])                    # [F,3,3] absolute triangle coords
        svals.append(hull.area / max(hull.volume, 1e-9) ** (2.0 / 3.0))
    return faces, np.array(svals)


@register_operator("vertex_tension_3d", level="cell", kind="lateral")
class VertexTension3D(Lateral):
    """3D SPV shape-energy force + self-propulsion. Retessellates the periodic 3D Voronoi each
    step; E = Σ K_V(V−V₀)² + K_S(S−S₀)², force by autodiff through tetra circumcentres."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    MECHANISM_TAGS = ["vertex_model_3d", "self_propelled_voronoi", "shape_energy_3d",
                      "rigidity_transition", "confluent_tissue", "morphogenesis"]
    PARAM_ROLES = {"s0": "target_shape_index_3d", "v0": "self_propulsion_speed",
                   "Dr": "rotational_diffusion", "K_V": "volume_stiffness", "K_S": "surface_stiffness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_V = float(params.get("K_V", 1.0))
        self.K_S = float(params.get("K_S", 1.0))
        self.V0 = float(params.get("V0", 1.0))
        self.s0 = float(params.get("s0", 5.4))                # target shape index (transition ~5.41)
        self.S0 = self.s0 * self.V0 ** (2.0 / 3.0)
        self.v0 = float(params.get("v0", 0.1))
        self.Dr = float(params.get("Dr", 1.0))
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 0.05))

    def _polarity(self, lvl, N, dev):
        if not hasattr(lvl, "pol"):
            p = torch.randn(N, 3, generator=getattr(lvl, "rng", None), device=dev)
            lvl.register_buffer("pol", p / p.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        return lvl.pol

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")                             # [N,3]
        N = pos_full.shape[0]
        dev = pos_full.device
        L = float(H.world_size[0])
        v_full = torch.zeros_like(pos_full)
        live = lvl.occ > 0
        idx = live.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 5:
            return {self.at: v_full}
        pos_live = pos_full[idx]
        pos_np = pos_live.detach().cpu().numpy().astype(np.float64) % L
        tetra, torig, tshift, incident = periodic_delaunay_3d(pos_np, L, n)
        cc_np = _tet_circumcenters_np((pos_np[torig] + tshift)[tetra])
        # per-cell convex-hull triangulation topology (numpy)
        cell_tets, cell_tris, ok = [], [], np.zeros(n, np.float32)
        for i in range(n):
            inc = incident[i]
            if len(inc) < 4:
                cell_tets.append(None); cell_tris.append(None); continue
            pts = cc_np[inc]
            try:
                hull = ConvexHull(pts)
            except Exception:
                cell_tets.append(None); cell_tris.append(None); continue
            cell_tets.append(inc)                             # global tetra (=circumcentre) indices
            cell_tris.append(hull.simplices)                 # [F,3] into `inc`
            ok[i] = 1.0
        tetra_t = torch.as_tensor(tetra, device=dev)
        orig_t = torch.as_tensor(torig, device=dev, dtype=torch.long)
        shift_t = torch.as_tensor(tshift, device=dev, dtype=pos_full.dtype)
        with torch.enable_grad():
            pos = (pos_live.detach() % L).requires_grad_(True)
            tiled = pos[orig_t] + shift_t                     # [27n,3]
            cc = _tet_circumcenters_torch(tiled[tetra_t])     # [M,3]
            E = pos.new_zeros(())
            for i in range(n):
                if not ok[i]:
                    continue
                verts = cc[torch.as_tensor(cell_tets[i], device=dev)]     # [k,3]
                tris = torch.as_tensor(cell_tris[i], device=dev)
                a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
                nrm = torch.cross(b - a, c - a, dim=-1)                   # [F,3]
                S_i = 0.5 * nrm.norm(dim=-1).sum()
                ri = pos[i]
                V_i = (((a - ri) * torch.cross(b - ri, c - ri, dim=-1)).sum(-1)).sum().abs() / 6.0
                E = E + self.K_V * (V_i - self.V0) ** 2 + self.K_S * (S_i - self.S0) ** 2
            grad = torch.autograd.grad(E, pos)[0]
        F = torch.nan_to_num(-grad)
        # 3D self-propulsion: unit polarity with rotational diffusion
        pol = self._polarity(lvl, N, dev)
        pol = pol + math.sqrt(2 * self.Dr * self.dt) * torch.randn(N, 3, generator=getattr(H, "rng", None),
                                                                   device=dev)
        pol = pol / pol.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        lvl.pol = pol
        v_full[idx] = self.mu * F + self.v0 * pol[idx]
        return {self.at: v_full}

"""vertex3d_ops -- the 3D vertex (Self-Propelled Voronoi) MECHANICS, plexus2.

The 3D generalisation of vertex_ops (Merkel & Manning, NJP 2018): cells are points, the
tissue is their 3D Voronoi tessellation (cells = convex polyhedra), and the mechanics come
from a cell shape energy

    E = sum_i [ K_V (V_i - V0_i)^2 + K_S (S_i - S0_i)^2 ]

with V_i, S_i the Voronoi volume and surface area; target shape index s0 = S0/V0^(2/3)
(rigidity transition ~5.41). Force = -grad E by autodiff through differentiable tetrahedron
circumcentres (the 3D Voronoi vertices).

FINITE tissue via GHOST boundary points, giving the paper's two initial tissues:
  * WITHOUT lumen -- a compacted aggregate: cells fill a ball, an outer ghost shell bounds it;
  * WITH lumen -- a monolayer vesicle: cells on a sphere, inner + outer ghost shells bound the
    shell so the central cavity (lumen) stays open.

Operators:
  tissue_seed_3d     (structural) -- seed a ball (no lumen) or a shell (vesicle); init V0
  voronoi_graph_3d   (rewire)     -- Delaunay cell-cell adjacency (the RD graph, for coupling)
  voronoi_tension_3d (lateral)    -- 3D shape-energy force on the centres (EMIT=velocity)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay, ConvexHull

from plexus.models.base import Lateral, Structural, Rewire
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  3D Voronoi geometry (finite, ghost-bounded, differentiable circumcentres)
# --------------------------------------------------------------------------- #
def _circ_np(P):
    a = P[:, 0]; u = P[:, 1] - a; v = P[:, 2] - a; w = P[:, 3] - a
    cvw = np.cross(v, w); cwu = np.cross(w, u); cuv = np.cross(u, v)
    den = 2.0 * (u * cvw).sum(-1); den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    o = ((u * u).sum(-1)[:, None] * cvw + (v * v).sum(-1)[:, None] * cwu
         + (w * w).sum(-1)[:, None] * cuv) / den[:, None]
    return a + o


def _circ_torch(P):
    a = P[:, 0]; u = P[:, 1] - a; v = P[:, 2] - a; w = P[:, 3] - a
    cvw = torch.cross(v, w, dim=-1); cwu = torch.cross(w, u, dim=-1); cuv = torch.cross(u, v, dim=-1)
    den = 2.0 * (u * cvw).sum(-1); den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
    o = ((u * u).sum(-1, keepdim=True) * cvw + (v * v).sum(-1, keepdim=True) * cwu
         + (w * w).sum(-1, keepdim=True) * cuv) / den[:, None]
    return a + o


def _fib_sphere(nn, R, c):
    k = np.arange(max(nn, 8)) + 0.5
    phi = np.arccos(1 - 2 * k / max(nn, 8)); th = np.pi * (1 + 5 ** 0.5) * k
    return c + R * np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def ghost_points(pos_np, lumen, pad):
    """Fixed boundary points that bound the finite Voronoi: an OUTER ghost shell always, and
    (for a vesicle) an INNER ghost shell inside the lumen so the shell cells are bounded."""
    c = pos_np.mean(0)
    r = np.linalg.norm(pos_np - c, axis=1)
    n = len(pos_np)
    gs = [_fib_sphere(max(64, n // 3), r.max() + pad, c)]
    if lumen:
        gs.append(_fib_sphere(max(48, n // 4), max(r.min() - pad, 0.15 * r.max()), c))
    return np.concatenate(gs, 0)


def _tessellate(pos_np, ghosts):
    """Finite Delaunay of real+ghost points -> (tet [M,4], incident-tetra list per real cell)."""
    allp = np.concatenate([pos_np, ghosts], 0)
    tet = Delaunay(allp, qhull_options="QJ").simplices
    n = len(pos_np)
    verts = tet.reshape(-1); tid = np.repeat(np.arange(len(tet)), 4)
    real = verts < n
    cells, tets = verts[real], tid[real]
    order = np.argsort(cells, kind="stable")
    cs, ts = cells[order], tets[order]
    b = np.searchsorted(cs, np.arange(n + 1))
    incident = [ts[b[i]:b[i + 1]] for i in range(n)]
    return allp, tet, incident


def cell_shape_3d(pos_np, lumen, pad):
    """Per real cell: Voronoi volume, surface, shape index s=S/V^(2/3), ok mask (diagnostics/render)."""
    ghosts = ghost_points(pos_np, lumen, pad)
    allp, tet, incident = _tessellate(pos_np, ghosts)
    cc = _circ_np(allp[tet])
    n = len(pos_np)
    vol = np.zeros(n); surf = np.zeros(n); ok = np.zeros(n, np.float32)
    for i in range(n):
        inc = incident[i]
        if len(inc) < 4:
            continue
        try:
            h = ConvexHull(cc[inc]); vol[i] = h.volume; surf[i] = h.area; ok[i] = 1.0
        except Exception:
            pass
    s = np.where(ok > 0, surf / np.maximum(vol, 1e-9) ** (2.0 / 3.0), np.nan)
    return vol, surf, s, ok


def cell_faces_3d(pos_np, lumen, pad):
    """Per real cell: the triangulated FACES of its Voronoi polyhedron ([F,3,3] absolute coords),
    its shape index s=S/V^(2/3), and its centre -- for drawing the actual 3D Voronoi cells."""
    ghosts = ghost_points(pos_np, lumen, pad)
    allp, tet, incident = _tessellate(pos_np, ghosts)
    cc = _circ_np(allp[tet])
    n = len(pos_np)
    faces, svals, cens = [], [], []
    for i in range(n):
        inc = incident[i]
        if len(inc) < 4:
            continue
        pts = cc[inc]
        try:
            h = ConvexHull(pts)
        except Exception:
            continue
        faces.append(pts[h.simplices])                    # [F,3,3]
        svals.append(h.area / max(h.volume, 1e-9) ** (2.0 / 3.0))
        cens.append(pos_np[i])
    return faces, np.array(svals), (np.array(cens) if cens else np.zeros((0, 3)))


def delaunay_edges_3d(pos_np, lumen, pad):
    """Undirected central cell-cell Delaunay adjacency (real cells only) -> edge_index [2,E]."""
    ghosts = ghost_points(pos_np, lumen, pad)
    allp, tet, _ = _tessellate(pos_np, ghosts)
    n = len(pos_np)
    pairs = set()
    for t in tet:
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = t[a], t[b]
                if u < n and v < n and u != v:
                    pairs.add((u, v) if u < v else (v, u))
    if not pairs:
        return np.zeros((2, 0), np.int64)
    e = np.array(sorted(pairs)).T
    return np.concatenate([e, e[::-1]], 1)


# --------------------------------------------------------------------------- #
#  Seed: ball (no lumen) or shell/vesicle (with lumen)
# --------------------------------------------------------------------------- #
@register_operator("tissue_seed_3d", set="cell", kind="structural", family="growth")
class TissueSeed3D(Structural):
    """Frame-0 IC (`before_frame: 1`): place N cell centres as a solid BALL (`lumen: false`,
    compacted aggregate) or a monolayer SHELL (`lumen: true`, vesicle around a cavity) of
    radius `radius`, centred in the world; set every cell's target volume V0 (state `v0`)."""
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["tissue_3d", "initial_condition", "aggregate", "vesicle", "lumen"]
    PARAM_ROLES = {"radius": "aggregate_radius", "lumen": "hollow_vesicle", "v0": "target_cell_volume"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.radius = float(params["radius"])
        self.lumen = bool(params.get("lumen", False))
        self.v0 = float(params.get("v0", 1.0))
        self.noise = float(params.get("noise", 0.2))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        N = lvl.state.shape[0]
        dev = lvl.state.device
        c = 0.5 * H.world_size[:3].to(dev)
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        d = torch.randn(N, 3, generator=g); d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
        if self.lumen:                                       # monolayer shell (vesicle)
            r = self.radius + self.noise * torch.randn(N, 1, generator=g)
        else:                                                # solid ball (aggregate)
            r = torch.rand(N, 1, generator=g).pow(1.0 / 3.0) * self.radius
        pos = (c + (d * r).to(dev))
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos
        if "v0" in lvl.state_schema:
            a, b = lvl.state_schema["v0"]; st[:, a:b] = self.v0
        lvl.state = st
        return {}


# --------------------------------------------------------------------------- #
#  Rewire: Delaunay cell-cell adjacency (the RD graph, for Stage 4 coupling)
# --------------------------------------------------------------------------- #
@register_operator("voronoi_graph_3d", set="cell", kind="rewire", family="topology")
class VoronoiGraph3D(Rewire):
    """Re-tessellate the live 3D cell centres each tick and set `cell.edge_index` to the
    Delaunay cell-cell adjacency (the graph the Turing RD runs on)."""
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["voronoi_3d", "delaunay", "retessellation", "confluent"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.lumen = bool(params.get("lumen", False))
        self.pad = float(params.get("pad", 0.15 * float(params["radius"])))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        live = lvl.occ > 0
        idx = live.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 8:
            lvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            return {}
        pos_np = lvl.get("pos")[idx].detach().cpu().numpy().astype(np.float64)
        ei_local = delaunay_edges_3d(pos_np, self.lumen, self.pad)
        ei = idx.cpu().numpy()[ei_local] if ei_local.size else ei_local
        lvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        return {}


# --------------------------------------------------------------------------- #
#  Lateral: 3D shape-energy force on the cell centres (overdamped)
# --------------------------------------------------------------------------- #
@register_operator("voronoi_tension_3d", set="cell", kind="lateral", family="mechanics")
class VoronoiTension3D(Lateral):
    """3D SPV shape-energy force: E = sum K_V(V-V0)^2 + K_S(S-S0)^2 over the (ghost-bounded)
    Voronoi polyhedra, force = -grad E by autodiff through the tetra circumcentres, overdamped
    (EMIT=velocity). V0 is per-cell (state `v0`, the growth handle) or a uniform param."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["s0", "radius"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos", "v0"]; WRITES = ["pos"]
    MAPS = ["edge_index"]
    MECHANISM_TAGS = ["vertex_model_3d", "self_propelled_voronoi", "shape_energy_3d",
                      "rigidity_transition", "confluent_tissue", "morphogenesis"]
    PARAM_ROLES = {"s0": "target_shape_index_3d", "K_V": "volume_stiffness", "K_S": "surface_stiffness",
                   "mu": "mobility", "lumen": "hollow_vesicle"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_V = float(params.get("K_V", 1.0)); self.K_S = float(params.get("K_S", 1.0))
        self.s0 = float(params.get("s0", 5.4)); self.V0 = float(params.get("V0", 1.0))
        self.mu = float(params.get("mu", 1.0))
        self.lumen = bool(params.get("lumen", False))
        self.pad = float(params.get("pad", 0.15 * float(params["radius"])))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")
        dev = pos_full.device
        v_full = torch.zeros_like(pos_full)
        live = lvl.occ > 0
        idx = live.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 8:
            return {self.at: v_full}
        pos_live = pos_full[idx]
        pos_np = pos_live.detach().cpu().numpy().astype(np.float64)
        ghosts_np = ghost_points(pos_np, self.lumen, self.pad)
        allp_np, tet, incident = _tessellate(pos_np, ghosts_np)
        cc_np = _circ_np(allp_np[tet])
        # per-cell hull triangulation topology (numpy)
        cell_tets, cell_tris, ok = [], [], np.zeros(n, np.float32)
        for i in range(n):
            inc = incident[i]
            if len(inc) < 4:
                cell_tets.append(None); cell_tris.append(None); continue
            try:
                h = ConvexHull(cc_np[inc])
            except Exception:
                cell_tets.append(None); cell_tris.append(None); continue
            cell_tets.append(torch.as_tensor(inc, device=dev))
            cell_tris.append(torch.as_tensor(h.simplices, device=dev)); ok[i] = 1.0
        tet_t = torch.as_tensor(tet, device=dev)
        ghosts_t = torch.as_tensor(ghosts_np, device=dev, dtype=pos_full.dtype)
        v0 = (lvl.get("v0")[idx, 0] if "v0" in lvl.state_schema
              else torch.full((n,), self.V0, device=dev))
        S0 = self.s0 * v0.clamp(min=1e-6) ** (2.0 / 3.0)
        with torch.enable_grad():
            pos = pos_live.detach().requires_grad_(True)
            allp = torch.cat([pos, ghosts_t], 0)
            cc = _circ_torch(allp[tet_t])
            E = pos.new_zeros(())
            for i in range(n):
                if not ok[i]:
                    continue
                verts = cc[cell_tets[i]]; tris = cell_tris[i]
                a, b, cc3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
                S_i = 0.5 * torch.cross(b - a, cc3 - a, dim=-1).norm(dim=-1).sum()
                ri = pos[i]
                V_i = (((a - ri) * torch.cross(b - ri, cc3 - ri, dim=-1)).sum(-1)).sum().abs() / 6.0
                E = E + self.K_V * (V_i - v0[i]) ** 2 + self.K_S * (S_i - S0[i]) ** 2
            grad = torch.autograd.grad(E, pos)[0]
        v_full[idx] = self.mu * torch.nan_to_num(-grad)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}

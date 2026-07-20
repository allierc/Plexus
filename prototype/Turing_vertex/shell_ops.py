"""shell_ops -- monolayer-epithelium 3D vertex mechanics (apical/basal + lumen pressure).

The Stage-3 `voronoi_tension_3d` bounds the finite Voronoi with GLOBAL spherical ghost shells
(inner+outer Fibonacci spheres at r.min/r.max). That is fine for a near-spherical vesicle but
FIGHTS localized deformation: the instant one region bulges, the global ghost sphere jumps to
that radius and corrupts every other cell's Voronoi volume. So the vesicle can only undulate
mildly, never sprout tubes.

This module replaces the global ghosts with PER-CELL apical/basal ghosts placed along each
cell's LOCAL surface normal (estimated by neighbourhood PCA). The boundary then follows the
deforming sheet -- a patch can fold out of plane without wrecking its neighbours -- which is
what lets differential (activator-driven) growth buckle the monolayer into tubes (Okuda Fig 5).
A `lumen_pressure` operator adds a soft enclosed-volume constraint so area growth goes into
buckling rather than uniform inflation.

Operators:
  voronoi_tension_shell (lateral) -- apical/basal-bounded shape-energy force (drop-in for
                                     voronoi_tension_3d on a monolayer); INTEGRAND="pos"
  lumen_pressure        (lateral) -- outward pressure ~ (V_target - V_lumen) on the shell;
                                     INTEGRAND="pos"
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import Delaunay, ConvexHull, cKDTree, SphericalVoronoi
from scipy.spatial import QhullError

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from vertex3d_ops import _circ_np, _circ_torch


# --------------------------------------------------------------------------- #
#  Per-cell surface normals (local PCA) + apical/basal ghost placement
# --------------------------------------------------------------------------- #
def cell_normals(pos_np, k=14):
    """Unit outward surface normal per cell: the smallest-variance direction of its k nearest
    neighbours (PCA), oriented away from the tissue centroid. Robust on a deforming monolayer
    (unlike the radial direction, which fails once tubes form)."""
    n = len(pos_np)
    k = min(k, n - 1)
    tree = cKDTree(pos_np)
    _, idx = tree.query(pos_np, k=k + 1)                     # includes self
    c = pos_np.mean(0)
    normals = np.zeros((n, 3))
    for i in range(n):
        X = pos_np[idx[i]] - pos_np[idx[i]].mean(0)
        w, V = np.linalg.eigh(X.T @ X)
        nrm = V[:, 0]                                        # smallest eigenvalue -> normal
        if np.dot(nrm, pos_np[i] - c) < 0.0:
            nrm = -nrm                                       # orient outward
        normals[i] = nrm
    return normals


def shell_ghosts(pos_np, normals, thickness):
    """Apical (inward) + basal (outward) ghost points, one pair per cell, along its normal."""
    apical = pos_np - thickness * normals
    basal = pos_np + thickness * normals
    return np.concatenate([apical, basal], 0)


def _tessellate_shell(pos_np, ghosts):
    """Delaunay of cells + per-cell apical/basal ghosts -> (allp, tet, incident-tets per cell)."""
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


def shell_cell_shape(pos_np, thickness, k=14):
    """Per cell: Voronoi volume, surface, ok mask (apical/basal-bounded). For diagnostics/render."""
    normals = cell_normals(pos_np, k)
    ghosts = shell_ghosts(pos_np, normals, thickness)
    allp, tet, incident = _tessellate_shell(pos_np, ghosts)
    cc = _circ_np(allp[tet]); n = len(pos_np)
    vol = np.zeros(n); surf = np.zeros(n); ok = np.zeros(n, np.float32)
    for i in range(n):
        inc = incident[i]
        if len(inc) < 4:
            continue
        try:
            h = ConvexHull(cc[inc]); vol[i] = h.volume; surf[i] = h.area; ok[i] = 1.0
        except Exception:
            pass
    return vol, surf, ok


def shell_faces(pos_np, act, thickness, k=14):
    """Per valid cell: triangulated Voronoi-polyhedron faces [F,3,3], activator, centre (render)."""
    normals = cell_normals(pos_np, k)
    ghosts = shell_ghosts(pos_np, normals, thickness)
    allp, tet, incident = _tessellate_shell(pos_np, ghosts)
    cc = _circ_np(allp[tet]); n = len(pos_np)
    faces, avals, cens = [], [], []
    for i in range(n):
        inc = incident[i]
        if len(inc) < 4:
            continue
        try:
            h = ConvexHull(cc[inc])
        except Exception:
            continue
        faces.append(cc[inc][h.simplices]); avals.append(act[i]); cens.append(pos_np[i])
    return faces, np.array(avals), (np.array(cens) if cens else np.zeros((0, 3)))


# --------------------------------------------------------------------------- #
#  Mechanics: apical/basal-bounded shape-energy force
# --------------------------------------------------------------------------- #
@register_operator("voronoi_tension_shell", set="cell", kind="lateral", family="mechanics")
class VoronoiTensionShell(Lateral):
    """Monolayer shape-energy force E = sum K_V(V-V0)^2 + K_S(S-S0)^2 over Voronoi polyhedra
    bounded by PER-CELL apical/basal ghosts (along the local PCA normal), so the sheet can fold
    locally. V0 is the per-cell growing target volume (state `v0`). Overdamped (EMIT=velocity),
    per-cell speed clamp for the degenerate-polyhedron guard. INTEGRAND='pos'."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["s0"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos", "v0"]; WRITES = ["pos"]
    MAPS = []
    MECHANISM_TAGS = ["vertex_model_3d", "monolayer", "apical_basal", "shape_energy_3d",
                      "epithelium", "morphogenesis"]
    PARAM_ROLES = {"s0": "target_shape_index_3d", "K_V": "volume_stiffness", "K_S": "surface_stiffness",
                   "thickness": "monolayer_half_thickness", "mu": "mobility"}
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (3D vertex monolayer, apical/basal); Merkel, M. & Manning, M. L. (2018). New J. Phys. 20:022002."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_V = float(params.get("K_V", 10.0)); self.K_S = float(params.get("K_S", 1.0))
        self.s0 = float(params.get("s0", 5.4)); self.V0 = float(params.get("V0", 1.0))
        self.mu = float(params.get("mu", 0.02)); self.k = int(params.get("k", 14))
        self.thickness = float(params.get("thickness", 0.0))  # 0 -> auto from V0
        self.vmax = float(params.get("vmax", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos"); dev = pos_full.device
        v_full = torch.zeros_like(pos_full)
        idx = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 8:
            return {self.at: v_full}
        pos_live = pos_full[idx]
        pos_np = pos_live.detach().cpu().numpy().astype(np.float64)
        v0 = (lvl.get("v0")[idx, 0] if "v0" in lvl.state_schema
              else torch.full((n,), self.V0, device=dev))
        thick = self.thickness or 0.7 * float(v0.mean().clamp(min=1e-6) ** (1.0 / 3.0))
        normals = cell_normals(pos_np, self.k)
        ghosts_np = shell_ghosts(pos_np, normals, thick)
        allp_np, tet, incident = _tessellate_shell(pos_np, ghosts_np)
        cc_np = _circ_np(allp_np[tet])
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
                a, b, c3 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
                S_i = 0.5 * torch.cross(b - a, c3 - a, dim=-1).norm(dim=-1).sum()
                ri = pos[i]
                V_i = (((a - ri) * torch.cross(b - ri, c3 - ri, dim=-1)).sum(-1)).sum().abs() / 6.0
                E = E + self.K_V * (V_i - v0[i]) ** 2 + self.K_S * (S_i - S0[i]) ** 2
            grad = torch.autograd.grad(E, pos)[0]
        v = self.mu * torch.nan_to_num(-grad)
        if self.vmax > 0:
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[idx] = v
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


# --------------------------------------------------------------------------- #
#  Lumen pressure: soft enclosed-volume constraint (area growth -> buckling)
# --------------------------------------------------------------------------- #
@register_operator("membrane_bending", set="cell", kind="lateral", family="mechanics")
class MembraneBending(Lateral):
    """Discrete bending rigidity / surface smoothing over the cell graph: pull each cell toward
    the mean position of its graph neighbours (a graph Laplacian). This DAMPS single-cell radial
    spikes (high spatial frequency) while permitting coherent multi-cell folds (low frequency),
    so the monolayer buckles as a smooth SHEET into tubes instead of fragmenting into spikes.
    `tangential=False` uses the full Laplacian; a small `k_bend` only removes the roughness.
    Reads the cell adjacency (`edge_index`, set by voronoi_graph_3d). INTEGRAND='pos'."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    DIFFERENTIABLE = False
    INTEGRAND = "pos"
    REQUIRES_PARAMS = ["k_bend"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["edge_index"]
    MECHANISM_TAGS = ["bending_rigidity", "membrane", "laplacian_smoothing", "epithelium", "coherence"]
    PARAM_ROLES = {"k_bend": "bending_stiffness"}
    REFERENCE = "Helfrich, W. (1973). Elastic properties of lipid bilayers: theory and possible experiments. Z. Naturforsch. C 28:693-703."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k_bend = float(params["k_bend"])
        self.vmax = float(params.get("vmax", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos"); dev = pos.device
        v_full = torch.zeros_like(pos)
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {self.at: v_full}
        i, j = ei[0], ei[1]
        N = pos.shape[0]
        agg = torch.zeros_like(pos).index_add_(0, i, pos[j])
        deg = torch.zeros(N, device=dev).index_add_(0, i, torch.ones_like(i, dtype=pos.dtype))
        mean_nb = agg / deg.clamp(min=1)[:, None]
        lap = (mean_nb - pos) * (deg > 0).float()[:, None]   # toward neighbour centroid
        v = self.k_bend * lap * lvl.occ[:, None]
        if self.vmax > 0:
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        return {self.at: v}


@register_operator("surface_lloyd", set="cell", kind="lateral", family="mechanics")
class SurfaceLloyd(Lateral):
    """Tangential Lloyd relaxation ON THE SIMULATED cells (not the render): each tick move every
    cell toward the centroid of its spherical-Voronoi region (kept at the cell's current radius),
    equalising cell areas. A regular hexagonal epithelial paving then emerges from the DYNAMICS --
    the 3D analog of the 2D vertex model relaxing to hexagons -- so division-scattered cells
    continuously re-even. Tangential only (radius preserved by the mechanics/bending). INTEGRAND='pos'."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    DIFFERENTIABLE = False
    INTEGRAND = "pos"
    REQUIRES_PARAMS = ["k_lloyd"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos"]; WRITES = ["pos"]
    MAPS = []
    MECHANISM_TAGS = ["lloyd_relaxation", "area_equalisation", "epithelium", "hexagonal_packing"]
    PARAM_ROLES = {"k_lloyd": "relaxation_rate"}
    REFERENCE = "Lloyd, S. P. (1982). Least squares quantization in PCM. IEEE Trans. Inf. Theory 28(2):129-137 (algorithm 1957)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k_lloyd = float(params["k_lloyd"])
        self.vmax = float(params.get("vmax", 1.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos"); dev = pos.device
        v_full = torch.zeros_like(pos)
        idx = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 12:
            return {self.at: v_full}
        pos_np = pos[idx].detach().cpu().numpy().astype(np.float64)
        c = pos_np.mean(0); d = pos_np - c
        r = np.linalg.norm(d, axis=1); dirs = d / np.clip(r[:, None], 1e-9, None)
        try:
            sv = SphericalVoronoi(dirs, radius=1.0, center=np.zeros(3))
            sv.sort_vertices_of_regions()
        except (QhullError, ValueError, IndexError):     # genuine geometric degeneracy only --
            return {self.at: v_full}                      # NOT a blanket swallow (that once hid a missing import)
        Vv = sv.vertices
        tgt = np.array([Vv[reg].mean(0) if len(reg) >= 3 else dirs[i]
                        for i, reg in enumerate(sv.regions)])
        tgt = tgt / np.clip(np.linalg.norm(tgt, axis=1)[:, None], 1e-9, None)
        target_pos = c + r[:, None] * tgt                # same radius, Voronoi-centroid direction
        v = self.k_lloyd * torch.as_tensor(target_pos - pos_np, device=dev, dtype=pos.dtype)
        if self.vmax > 0:
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[idx] = v
        return {self.at: v_full}


@register_operator("lumen_pressure", set="cell", kind="lateral", family="mechanics")
class LumenPressure(Lateral):
    """Soft enclosed-volume (lumen) constraint on the vesicle: an outward force along each shell
    cell's normal, magnitude P = K_L * (V_target - V_enc), where V_enc is the volume enclosed by
    the shell (convex-hull estimate). When the monolayer AREA grows (division) but V_target lags,
    the excess area cannot inflate the lumen -> it buckles out of plane (the driver of tubulation).
    `v_target_scale`: target lumen volume as a multiple of the initial enclosed volume.
    INTEGRAND='pos'."""
    SUPPORTED_DIMS = [3]
    EMIT = "velocity"
    DIFFERENTIABLE = False
    INTEGRAND = "pos"
    REQUIRES_PARAMS = ["k_lumen"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos"]; WRITES = ["pos"]
    MAPS = []
    MECHANISM_TAGS = ["lumen", "pressure", "enclosed_volume", "buckling", "vesicle"]
    PARAM_ROLES = {"k_lumen": "lumen_stiffness", "v_target_scale": "target_lumen_volume_multiple"}
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (monolayer vesicle lumen / enclosed-volume constraint)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.k_lumen = float(params["k_lumen"])
        self.v_target_scale = float(params.get("v_target_scale", 1.0))
        self.k = int(params.get("k", 14))
        self.mu = float(params.get("mu", 0.02))
        self.vmax = float(params.get("vmax", 1.0))
        self._v0_lumen = None                                # initial enclosed volume (set on first call)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos"); dev = pos_full.device
        v_full = torch.zeros_like(pos_full)
        idx = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 8:
            return {self.at: v_full}
        pos_np = pos_full[idx].detach().cpu().numpy().astype(np.float64)
        try:
            V_enc = float(ConvexHull(pos_np).volume)
        except Exception:
            return {self.at: v_full}
        if self._v0_lumen is None:
            self._v0_lumen = V_enc
        V_target = self.v_target_scale * self._v0_lumen
        P = self.k_lumen * (V_target - V_enc)                # >0 push out (inflate), <0 pull in
        normals = torch.as_tensor(cell_normals(pos_np, self.k), device=dev, dtype=pos_full.dtype)
        v = (self.mu * P) * normals
        if self.vmax > 0:
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[idx] = v
        return {self.at: v_full}

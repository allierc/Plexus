"""coupled_ops -- Stage 4: the FULL coupled Turing x vertex model (Okuda et al. 2018), 2D.

Closes the loop from the two halves: reaction-diffusion on the cell graph patterns the
activator; the activator (a mitogen) grows each cell's target area a0 (Eqs 7-8, Hill);
a cell divides when a0 reaches 2x its base (v_th=(4/3)v_ref); the vertex mechanics deform
the tissue toward the growing target areas; re-tessellation updates the diffusion graph and
the mechanics together. A FREE (finite, ghost-bounded) tissue so it can grow and deform.

Fixed buffer + occupancy: the `cell` set is allocated at `buffer` slots, `occ` marks live;
division wakes dormant slots (no resizing). Multi-block state (the engine extension): pos is
integrated by the mechanics, chem=[act,inh] by the RD, a0 by growth -- each its own block.

Operators (this file):
  coupled_seed_2d   (structural) -- seed n0 cells on a fixed buffer; init chem, a0, occ
  voronoi_graph_2d  (rewire)     -- finite Voronoi -> cell.edge_index (the diffusion graph)
  growth            (lateral)    -- activator -> da0/dt (Hill, Eqs 7-8);  INTEGRAND="a0"
  divide_2x         (structural) -- a0 >= 2*a0_base -> wake a daughter, split a0, inherit chem
  vertex_tension_2d (lateral)    -- shape-energy force toward the (growing) a0; INTEGRAND="pos"
Reused from turing_ops: `graph_diffuse`, `react` (INTEGRAND="chem"); the RD kinetics.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay

from plexus.models.base import Lateral, Structural, Rewire
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Finite (ghost-bounded) 2D Voronoi geometry
# --------------------------------------------------------------------------- #
def _circ_np(P):
    ax, ay = P[:, 0, 0], P[:, 0, 1]; bx, by = P[:, 1, 0], P[:, 1, 1]; cx, cy = P[:, 2, 0], P[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return np.stack([ux, uy], 1)


def _circ_torch(P):
    ax, ay = P[:, 0, 0], P[:, 0, 1]; bx, by = P[:, 1, 0], P[:, 1, 1]; cx, cy = P[:, 2, 0], P[:, 2, 1]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1e-9), d)
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return torch.stack([ux, uy], 1)


def _ghost_ring(pos_np, pad):
    c = pos_np.mean(0)
    r = np.linalg.norm(pos_np - c, axis=1)
    ng = max(24, len(pos_np) // 3)
    th = 2 * np.pi * np.arange(ng) / ng
    return c + (r.max() + pad) * np.stack([np.cos(th), np.sin(th)], 1)


def finite_voronoi_2d(pos_np, pad):
    """Finite Voronoi of the cells (bounded by a ghost ring). Returns (tri [M,3] into the
    real+ghost array, ghosts [g,2], ring_idx [n,Vmax] triangle index per cell's ordered
    vertex ring (padded), ok [n])."""
    ghosts = _ghost_ring(pos_np, pad)
    allp = np.concatenate([pos_np, ghosts], 0)
    n = len(pos_np)
    tri = Delaunay(allp, qhull_options="QJ").simplices
    cc = _circ_np(allp[tri])
    verts = tri.reshape(-1); tid = np.repeat(np.arange(len(tri)), 3)
    real = verts < n
    cells, tris = verts[real], tid[real]
    order = np.argsort(cells, kind="stable")
    cs, ts = cells[order], tris[order]
    b = np.searchsorted(cs, np.arange(n + 1))
    rings, Vmax = [None] * n, 3
    for i in range(n):
        t = ts[b[i]:b[i + 1]]
        if len(t) >= 3:
            v = cc[t]
            ang = np.arctan2(v[:, 1] - pos_np[i, 1], v[:, 0] - pos_np[i, 0])
            rings[i] = t[np.argsort(ang)]; Vmax = max(Vmax, len(t))
    ring_idx = np.zeros((n, Vmax), dtype=np.int64); ok = np.zeros(n, np.float32)
    for i in range(n):
        rr = rings[i]
        if rr is not None:
            ring_idx[i, :len(rr)] = rr; ring_idx[i, len(rr):] = rr[-1]; ok[i] = 1.0
    return tri, ghosts, ring_idx, ok


def delaunay_edges_2d(tri, n):
    pairs = set()
    for a, b, c in tri:
        for u, v in ((a, b), (b, c), (a, c)):
            if u < n and v < n and u != v:
                pairs.add((u, v) if u < v else (v, u))
    if not pairs:
        return np.zeros((2, 0), np.int64)
    e = np.array(sorted(pairs)).T
    return np.concatenate([e, e[::-1]], 1)


def cell_polygons_2d(pos_np, pad):
    """Per live cell: Voronoi polygon (for rendering), area. Returns (polys, area, ok)."""
    tri, ghosts, ring_idx, ok = finite_voronoi_2d(pos_np, pad)
    allp = np.concatenate([pos_np, ghosts], 0); cc = _circ_np(allp[tri])
    n = len(pos_np); polys, area = [], np.zeros(n)
    for i in range(n):
        if not ok[i]:
            polys.append(None); continue
        v = cc[ring_idx[i]]
        keep = [0]
        for k in range(1, len(v)):
            if not np.allclose(v[k], v[keep[-1]]):
                keep.append(k)
        vv = v[keep]; polys.append(vv)
        x, y = vv[:, 0], vv[:, 1]
        area[i] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return polys, area, ok


# --------------------------------------------------------------------------- #
#  Seed: n0 cells on a fixed buffer (occupancy)
# --------------------------------------------------------------------------- #
@register_operator("coupled_seed_2d", set="cell", kind="structural", family="growth")
class CoupledSeed2D(Structural):
    """Frame-0 IC (`before_frame: 1`): place the live cells (occ>0) as a disc, init the
    morphogens (activator noise around a mean, substrate/inhibitor), and the target area a0.
    The set is allocated at `buffer` slots (occ marks live); division wakes the dormant ones."""
    SUPPORTED_DIMS = [2]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["radius"]
    MECHANISM_TAGS = ["tissue", "initial_condition", "fixed_buffer", "occupancy"]
    PARAM_ROLES = {"radius": "disc_radius", "a0": "base_target_area", "a_mean": "activator_seed"}
    REFERENCE = "Plexus (this work); cf. Okuda, S. et al. (2018). Sci. Rep. 8:2386."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.radius = float(params["radius"])
        self.a0 = float(params.get("a0", 1.0))
        self.a_mean = float(params.get("a_mean", 1.0))
        self.h_mean = float(params.get("h_mean", 1.0))
        self.noise = float(params.get("noise", 0.05))
        self.seed_mode = params.get("seed_mode", "noise")    # noise (Brusselator/GM) | scatter (Gray-Scott)
        self.seed_frac = float(params.get("seed_frac", 0.04))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        c = 0.5 * H.world_size[:2].to(dev)
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(live.numel())
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        idx = torch.arange(n).float() + 0.5                  # sunflower disc
        r = torch.sqrt(idx / n) * self.radius
        th = math.pi * (1.0 + 5.0 ** 0.5) * idx
        pos = torch.stack([c[0] + r * torch.cos(th), c[1] + r * torch.sin(th)], 1).to(dev)
        st = lvl.state.clone()
        px0, px1 = lvl.state_schema["pos"]; st[live, px0:px1] = pos
        ca, _ = lvl.state_schema["chem"]
        if self.seed_mode == "scatter":                      # Gray-Scott: full substrate + scattered activator nuclei
            v = 0.02 * torch.rand(n, generator=g).to(dev)
            u = torch.ones(n, device=dev)
            nuc = torch.rand(n, generator=g).to(dev) < self.seed_frac
            v[nuc] = 0.5
            st[live, ca] = (v + self.noise * torch.randn(n, generator=g).to(dev)).clamp(min=0.0)  # activator v
            st[live, ca + 1] = u                                                                  # substrate u
        else:                                                # Brusselator/GM: steady state + noise
            st[live, ca] = (self.a_mean + self.noise * torch.randn(n, generator=g)).to(dev)      # activator
            st[live, ca + 1] = (self.h_mean + self.noise * torch.randn(n, generator=g)).to(dev)  # inhibitor
        a0a, _ = lvl.state_schema["a0"]; st[live, a0a] = self.a0
        lvl.state = st
        lvl.a0_base = self.a0                                 # remember the base target area (division threshold)
        return {}


# --------------------------------------------------------------------------- #
#  Rewire: finite Voronoi -> the cell-cell diffusion graph
# --------------------------------------------------------------------------- #
@register_operator("voronoi_graph_2d", set="cell", kind="rewire", family="topology")
class VoronoiGraph2D(Rewire):
    """Re-tessellate the live cells (finite, ghost-bounded) and set `cell.edge_index` to the
    Delaunay adjacency the Turing RD diffuses on. Automatic T1s / new neighbours as the tissue
    grows and deforms."""
    SUPPORTED_DIMS = [2]
    DIFFERENTIABLE = False
    MECHANISM_TAGS = ["voronoi", "delaunay", "retessellation", "confluent"]
    REFERENCE = "Delaunay, B. (1934). Sur la sphere vide. Bull. Acad. Sci. URSS 6:793-800."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.pad = float(params.get("pad", 0.4))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(live.numel())
        if n < 4:
            lvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev); return {}
        pos_np = lvl.get("pos")[live].detach().cpu().numpy().astype(np.float64)
        tri, _, _, _ = finite_voronoi_2d(pos_np, self.pad)
        ei_local = delaunay_edges_2d(tri, n)
        ei = live.cpu().numpy()[ei_local] if ei_local.size else ei_local
        lvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        return {}


# --------------------------------------------------------------------------- #
#  Growth: activator -> target-area growth (Hill; Okuda Eqs 7-8)
# --------------------------------------------------------------------------- #
@register_operator("growth", set="cell", kind="lateral", family="growth")
class Growth(Lateral):
    """The mechanochemical coupling (Okuda Eqs 7-8): the activator is a mitogen, so a cell's
    target area grows at a Hill-saturating rate in the activator concentration,

        da0/dt = lam_ref * ( rho_lam + (a^n / (a^n + a_sw^n)) )

    Reads chem[activator]; writes the a0 block (INTEGRAND='a0', integrated as its own block)."""
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    INTEGRAND = "a0"
    REQUIRES_PARAMS = ["lam_ref"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["chem"]; WRITES = ["a0"]
    MAPS = []
    MECHANISM_TAGS = ["growth", "mitogen", "hill", "mechanochemical_coupling", "proliferation"]
    PARAM_ROLES = {"lam_ref": "reference_growth_rate", "a_sw": "activator_half_saturation",
                   "rho_lam": "basal_growth", "hill": "hill_coefficient"}
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (Eqs. 7-8, activator-driven growth); Hill, A. V. (1910). J. Physiol. 40:iv-vii."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.lam_ref = float(params["lam_ref"])
        self.a_sw = float(params.get("a_sw", 1.0))
        self.rho_lam = float(params.get("rho_lam", 0.0))
        self.hill = float(params.get("hill", 2.0))
        self.cap = float(params.get("cap", 2.0))             # stop growing at cap*base (the division size)
        self.block = params.get("block", "a0")               # size block: a0 (2D target area) | v0 (3D target volume)
        self.INTEGRAND = self.block                          # per-instance override of the class INTEGRAND

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        a = lvl.get("chem")[:, 0].clamp(min=0.0)             # activator concentration
        an = a.pow(self.hill)
        lam = self.lam_ref * (self.rho_lam + an / (an + self.a_sw ** self.hill + 1e-9))
        # cap at the division size: when the buffer is full a cell cannot divide, so without this
        # its target size would grow unbounded and over-compress the (jammed) core.
        base = float(getattr(lvl, self.block + "_base", 1.0))
        below = (lvl.get(self.block)[:, 0] < self.cap * base).to(lam.dtype)
        d = (lam * lvl.occ * below).unsqueeze(1)             # dormant / saturated cells do not grow
        return {self.at: d}


# --------------------------------------------------------------------------- #
#  Divide: size-controlled proliferation at 2x (Okuda v_th)
# --------------------------------------------------------------------------- #
@register_operator("divide_2x", set="cell", kind="structural", family="growth")
class Divide2x(Structural):
    """Size-controlled division (Okuda): when a cell's target area a0 reaches `ratio`x its base
    (v_th=(4/3)v_ref), it divides -- wake a dormant slot beside the mother, split a0 back to the
    base for both, and the daughter inherits the mother's morphogen state. Fixed buffer + occ;
    when the buffer is full, division stops (capacity-limited)."""
    SUPPORTED_DIMS = [2, 3]
    EMIT = None
    DIFFERENTIABLE = False                                    # structural (mutates entities), not in the differentiable rollout
    MECHANISM_TAGS = ["proliferation", "mitosis", "size_control", "occupancy"]
    PARAM_ROLES = {"ratio": "division_size_ratio", "offset": "daughter_placement"}
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (size-controlled in-plane division, v_th); Okuda, S. et al. (2015). Biomech. Model. Mechanobiol. 12:627-644."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.ratio = float(params.get("ratio", 2.0))
        self.offset = float(params.get("offset", 0.15))
        self.block = params.get("block", "a0")               # size block: a0 (2D area) | v0 (3D volume)
        self.tangential = bool(params.get("tangential", True))  # 3D: place daughter IN the tissue-surface plane (Okuda)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        base = float(getattr(lvl, self.block + "_base", 1.0))
        sa, _ = lvl.state_schema[self.block]
        sz = lvl.state[:, sa]
        live = lvl.occ > 0
        ready = (live & (sz >= self.ratio * base)).nonzero(as_tuple=True)[0]
        free = (~live).nonzero(as_tuple=True)[0]
        cap = min(int(ready.numel()), int(free.numel()))
        if cap == 0:
            return {}
        parents = ready[:cap]; slots = free[:cap]
        # daughter inherits every per-node buffer (state incl chem, node_type, ...)
        lvl.state[slots] = lvl.state[parents].clone()
        for _, b in list(lvl.named_buffers()):
            if b is not None and b.dim() >= 1 and b.shape[0] == lvl.n and b is not lvl.state:
                b[slots] = b[parents].clone()
        # split target size back to base for BOTH; place the daughter beside the mother
        lvl.state[parents, sa] = base
        lvl.state[slots, sa] = base
        px0, px1 = lvl.state_schema["pos"]; D = px1 - px0    # dim-aware (2 or 3)
        parent_pos = lvl.state[parents, px0:px1]
        rnd = torch.rand(cap, D, generator=getattr(H, "rng", None), device=dev) - 0.5
        if D == 3 and self.tangential:
            # Okuda in-plane division: place the daughter WITHIN the tissue-surface plane
            # (perpendicular to the local radial normal), not radially -> the monolayer stays
            # coherent and cells stay uniform (a random radial offset is what roughened the shell).
            c = lvl.state[lvl.occ > 0, px0:px1].mean(0)
            nrm = parent_pos - c; nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp(min=1e-9)
            rnd = rnd - (rnd * nrm).sum(1, keepdim=True) * nrm          # project onto the tangent plane
        jit = torch.nn.functional.normalize(rnd, dim=1, eps=1e-9) * self.offset
        lvl.state[slots, px0:px1] = parent_pos + jit
        lvl.occ[slots] = 1.0
        if hasattr(lvl, "birth"):
            lvl.birth[slots] = 1.0
        return {}


# --------------------------------------------------------------------------- #
#  Mechanics: finite 2D shape-energy force toward the (growing) target area
# --------------------------------------------------------------------------- #
@register_operator("vertex_tension_2d", set="cell", kind="lateral", family="mechanics")
class VertexTension2D(Lateral):
    """Finite (ghost-bounded) 2D shape-energy force: E = sum K_A(A - a0)^2 + K_P(P - P0)^2 over
    the Voronoi polygons, force = -grad E by autodiff through the circumcentres, overdamped
    (EMIT=velocity). A0 is the per-cell GROWING target area (state `a0`), so growth drives the
    tissue to expand and deform. P0 = p0*sqrt(a0). INTEGRAND='pos' (the coordinate)."""
    SUPPORTED_DIMS = [2]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    INTEGRAND = "pos"
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos", "a0"]; WRITES = ["pos"]
    MAPS = ["edge_index"]
    MECHANISM_TAGS = ["vertex_model", "shape_energy", "confluent_tissue", "morphogenesis"]
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness"}
    REFERENCE = "Bi, D. et al. (2016). A density-independent rigidity transition in biological tissues. Nat. Phys. 11:1074-1079; Nagai, T. & Honda, H. (2001). Phil. Mag. B 81:699-719."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_A = float(params.get("K_A", 1.0)); self.K_P = float(params.get("K_P", 0.5))
        self.p0 = float(params.get("p0", 3.9)); self.mu = float(params.get("mu", 0.2))
        self.pad = float(params.get("pad", 0.4))
        self.vmax = float(params.get("vmax", 2.0))           # clamp |velocity| -- degenerate boundary polygons blow up

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        dev = pos_full.device
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(live.numel())
        if n < 4:
            return {self.at: v_full}
        pos_live = pos_full[live]
        pos_np = pos_live.detach().cpu().numpy().astype(np.float64)
        tri, ghosts, ring, ok = finite_voronoi_2d(pos_np, self.pad)
        a0 = lvl.get("a0")[live, 0].clamp(min=1e-4)
        P0 = self.p0 * a0.sqrt()
        tri_t = torch.as_tensor(tri, device=dev)
        ghosts_t = torch.as_tensor(ghosts, device=dev, dtype=pos_full.dtype)
        ring_t = torch.as_tensor(ring, device=dev)
        ok_t = torch.as_tensor(ok, device=dev)
        with torch.enable_grad():
            pos = pos_live.detach().requires_grad_(True)
            allp = torch.cat([pos, ghosts_t], 0)
            cc = _circ_torch(allp[tri_t])
            verts = cc[ring_t]                               # [n, Vmax, 2]
            nxt = torch.roll(verts, -1, dims=1)
            area = 0.5 * (verts[..., 0] * nxt[..., 1] - nxt[..., 0] * verts[..., 1]).sum(1).abs()
            perim = (nxt - verts).norm(dim=-1).sum(1)
            E = ((self.K_A * (area - a0) ** 2 + self.K_P * (perim - P0) ** 2) * ok_t).sum()
            grad = torch.autograd.grad(E, pos)[0]
        v = self.mu * torch.nan_to_num(-grad) * ok_t[:, None]
        vn = v.norm(dim=1, keepdim=True)                     # clamp per-cell speed (boundary-degeneracy guard)
        v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[live] = v
        return {self.at: v_full}


# --------------------------------------------------------------------------- #
#  Planar Lloyd relaxation -> uniform hexagonal cells (the 2D analog of surface_lloyd)
# --------------------------------------------------------------------------- #
@register_operator("lloyd_2d", set="cell", kind="lateral", family="mechanics")
class Lloyd2D(Lateral):
    """Planar Lloyd relaxation: move each live cell toward the area-centroid of its (finite)
    Voronoi polygon, equalising cell areas -> a uniform hexagonal packing emerges from the
    dynamics (the 2D analog of surface_lloyd; the vertex_tension already relaxes shape, this
    evens the SIZES that division scatters). INTEGRAND='pos'."""
    SUPPORTED_DIMS = [2]
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
        self.pad = float(params.get("pad", 0.4))
        self.vmax = float(params.get("vmax", 2.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos"); dev = pos.device
        v_full = torch.zeros_like(pos)
        idx = (lvl.occ > 0).nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 4:
            return {self.at: v_full}
        pos_np = pos[idx].detach().cpu().numpy().astype(np.float64)
        polys, area, ok = cell_polygons_2d(pos_np, self.pad)
        cent = pos_np.copy()
        for i in range(n):
            P = polys[i]
            if ok[i] <= 0 or P is None or len(P) < 3:
                continue
            x, y = P[:, 0], P[:, 1]; xn, yn = np.roll(x, -1), np.roll(y, -1)
            cross = x * yn - xn * y; A = 0.5 * cross.sum()
            if abs(A) > 1e-9:                                # polygon area-centroid (proper Lloyd target)
                cent[i, 0] = ((x + xn) * cross).sum() / (6 * A)
                cent[i, 1] = ((y + yn) * cross).sum() / (6 * A)
        v = self.k_lloyd * (torch.as_tensor(cent, device=dev, dtype=pos.dtype) - pos[idx])
        if self.vmax > 0:
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[idx] = v
        return {self.at: v_full}

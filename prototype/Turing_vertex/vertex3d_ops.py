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
  seed_tissue_3d     (structural) -- seed a ball (no lumen) or a shell (vesicle); init V0
  voronoi_graph_3d   (rewire)     -- Delaunay cell-cell adjacency (the RD graph, for coupling)
  voronoi_tension_3d (lateral)    -- 3D shape-energy force on the centres (EMIT=velocity)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay, ConvexHull, SphericalVoronoi

from plexus.models.base import Lateral, Structural, Rewire
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Epithelial PRISM builder: monolayer = 2D surface cells (spherical Voronoi)
#  extruded radially, NOT a 3D Voronoi (which slivers on a thin shell).
# --------------------------------------------------------------------------- #
def _poly_area_3d(P):
    """Area of a (near-planar) 3D polygon, vertices ordered: 0.5*||sum_j P_j x P_{j+1}||."""
    x = np.cross(P, np.roll(P, -1, axis=0)).sum(0)
    return 0.5 * float(np.linalg.norm(x))


def cell_prisms_3d(pos_np, thickness, center=None, relax=0):
    """Model a monolayer epithelium as radially EXTRUDED PRISMS.

    Each cell's APICAL face is its cell on the sphere -- a spherical Voronoi polygon of the cell
    DIRECTIONS (compact, near-regular hexagons/pentagons, unlike the radial slivers a 3D Voronoi
    gives on a thin shell). Extrude radially by `thickness` to the BASAL face; lateral quads join
    them -> a one-cell-thick prism. Voronoi vertices take the mean radius of the cells sharing
    them, so the paving is watertight even on a slightly non-uniform shell.

    Returns (prism_faces, apical_faces, vol, surf, ok):
      prism_faces[i] -- list of polygon faces (each [k,3]) of cell i's prism (apical, basal, walls)
      apical_faces[i] -- the apical polygon [k,3] (the outer surface patch)
      vol[i], surf[i] -- prism volume and total surface area; ok[i] -- valid mask
    """
    n = len(pos_np)
    c = pos_np.mean(0) if center is None else np.asarray(center, float)
    d = pos_np - c
    r = np.linalg.norm(d, axis=1)
    dirs = d / np.clip(r[:, None], 1e-9, None)
    for _ in range(relax):                                   # optional Lloyd relaxation -> equal areas
        sv = SphericalVoronoi(dirs, radius=1.0, center=np.zeros(3))
        sv.sort_vertices_of_regions()
        newd = np.array([sv.vertices[reg].mean(0) if len(reg) >= 3 else dirs[i]
                         for i, reg in enumerate(sv.regions)])
        dirs = newd / np.clip(np.linalg.norm(newd, axis=1)[:, None], 1e-9, None)
    sv = SphericalVoronoi(dirs, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    V = sv.vertices                                          # [m,3] unit-sphere Voronoi vertices
    regions = sv.regions                                     # ordered vertex-index list per cell
    vrad = np.zeros(len(V)); vcnt = np.zeros(len(V))         # per-vertex radius = mean of sharing cells
    for i, reg in enumerate(regions):
        for vi in reg:
            vrad[vi] += r[i]; vcnt[vi] += 1
    vrad = np.where(vcnt > 0, vrad / np.maximum(vcnt, 1), float(np.mean(r)))
    h = float(thickness)
    prism_faces, apical_faces = [None] * n, [None] * n
    vol = np.zeros(n); surf = np.zeros(n); ok = np.zeros(n, np.float32)
    for i, reg in enumerate(regions):
        if len(reg) < 3:
            continue
        uv = V[reg]; rr = vrad[reg]                          # ordered unit vertices + their radii
        apical = uv * (rr + h / 2.0)[:, None] + c            # outer polygon
        basal = uv * (rr - h / 2.0)[:, None] + c             # inner polygon
        k = len(reg)
        walls = [np.array([apical[j], apical[(j + 1) % k], basal[(j + 1) % k], basal[j]])
                 for j in range(k)]
        prism_faces[i] = [apical, basal[::-1]] + walls
        apical_faces[i] = apical
        Aa, Ab = _poly_area_3d(apical), _poly_area_3d(basal)
        surf[i] = Aa + Ab + sum(_poly_area_3d(w) for w in walls)
        vol[i] = 0.5 * (Aa + Ab) * h                         # prism volume ~ mean cross-section * height
        ok[i] = 1.0
    return prism_faces, apical_faces, vol, surf, ok


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
@register_operator("seed_tissue_3d", set="cell", kind="structural", family="growth")
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
    REFERENCE = "Okuda, S. et al. (2018). Combining Turing and 3D vertex models... Sci. Rep. 8:2386 (monolayer vesicle / compacted aggregate)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.radius = float(params["radius"])
        self.lumen = bool(params.get("lumen", False))
        self.v0 = float(params.get("v0", 1.0))
        self.v0noise = float(params.get("v0noise", 0.0))     # per-cell target-volume spread -> ASYNCHRONOUS division
        self.noise = float(params.get("noise", 0.2))
        self.a_mean = float(params.get("a_mean", 1.0))       # coupling: activator seed (Brusselator steady state)
        self.h_mean = float(params.get("h_mean", 3.0))       # coupling: inhibitor seed
        self.cnoise = float(params.get("cnoise", 0.04))
        self.seed_mode = params.get("seed_mode", "noise")    # noise (Brusselator/GM) | scatter (Gray-Scott)
        self.seed_frac = float(params.get("seed_frac", 0.04))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        c = 0.5 * H.world_size[:3].to(dev)
        live = (lvl.occ > 0).nonzero(as_tuple=True)[0]       # seed LIVE cells only (growing buffer)
        N = int(live.numel())
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        if self.lumen:                                       # monolayer shell (vesicle): even Fibonacci sphere
            d = torch.as_tensor(_fib_sphere(N, 1.0, np.zeros(3)) , dtype=torch.float32)
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            r = self.radius + self.noise * torch.randn(N, 1, generator=g)
        else:                                                # solid ball (aggregate): uniform interior
            d = torch.randn(N, 3, generator=g); d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            r = torch.rand(N, 1, generator=g).pow(1.0 / 3.0) * self.radius
        pos = (c + (d * r).to(dev))
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[live, px0:px1] = pos
        if "v0" in lvl.state_schema:
            a, b = lvl.state_schema["v0"]
            spread = (1.0 + self.v0noise * torch.randn(N, 1, generator=g)).clamp(min=0.2).to(dev)
            st[live, a:b] = self.v0 * spread                 # noisy start -> cells cross v_th at spread times
            lvl.v0_base = self.v0                             # base target volume (division threshold)
        if "chem" in lvl.state_schema:                       # coupling: seed the two morphogens on live cells
            ca, _ = lvl.state_schema["chem"]
            gg = torch.Generator(device="cpu"); gg.manual_seed(1)
            if self.seed_mode == "scatter":                  # Gray-Scott: full substrate + scattered activator nuclei
                v = 0.02 * torch.rand(N, generator=gg).to(dev)
                v[torch.rand(N, generator=gg).to(dev) < self.seed_frac] = 0.5
                st[live, ca] = (v + self.cnoise * torch.randn(N, generator=gg).to(dev)).clamp(min=0.0)
                st[live, ca + 1] = torch.ones(N, device=dev)                                        # substrate u
            else:                                            # Brusselator/GM: steady state + noise
                st[live, ca] = (self.a_mean + self.cnoise * torch.randn(N, generator=gg)).to(dev)
                st[live, ca + 1] = (self.h_mean + self.cnoise * torch.randn(N, generator=gg)).to(dev)
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
    REFERENCE = "Delaunay, B. (1934). Sur la sphere vide. Bull. Acad. Sci. URSS 6:793-800."

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
#  Vectorised shape energy (padded batched -> 32x over a per-cell Python loop;
#  torch.compile fuses it for another ~1.5x). One scalar E over all cells.
# --------------------------------------------------------------------------- #
def _shape_energy_3d(pos, ghosts, tet_t, tri_idx, tmask, okt, v0, S0, KV, KS):
    allp = torch.cat([pos, ghosts], 0)
    cc = _circ_torch(allp[tet_t])                             # [M,3] tetra circumcentres
    verts = cc[tri_idx]                                       # [n, Fmax, 3, 3] hull-triangle vertices per cell
    a, b, c3 = verts[:, :, 0], verts[:, :, 1], verts[:, :, 2]
    S = (0.5 * torch.linalg.cross(b - a, c3 - a, dim=-1).norm(dim=-1) * tmask).sum(1)
    ri = pos[:, None, :]                                      # tetrahedra from the cell centre to each hull face
    V = (((a - ri) * torch.linalg.cross(b - ri, c3 - ri, dim=-1)).sum(-1) * tmask).sum(1).abs() / 6.0
    return (okt * (KV * (V - v0) ** 2 + KS * (S - S0) ** 2)).sum()


_shape_energy_3d_compiled = None
def _energy_fn_3d(use_compile):
    """Return the (optionally torch.compiled, dynamic-shape) shape-energy function."""
    global _shape_energy_3d_compiled
    if not use_compile:
        return _shape_energy_3d
    if _shape_energy_3d_compiled is None:
        _shape_energy_3d_compiled = torch.compile(_shape_energy_3d, dynamic=True)
    return _shape_energy_3d_compiled


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
    REFERENCE = "Merkel, M. & Manning, M. L. (2018). A geometrically controlled rigidity transition in a model for confluent 3D tissues. New J. Phys. 20:022002."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_V = float(params.get("K_V", 1.0)); self.K_S = float(params.get("K_S", 1.0))
        self.s0 = float(params.get("s0", 5.4)); self.V0 = float(params.get("V0", 1.0))
        self.mu = float(params.get("mu", 1.0))
        self.lumen = bool(params.get("lumen", False))
        self.pad = float(params.get("pad", 0.15 * float(params["radius"])))
        self.vmax = float(params.get("vmax", 0.0))           # >0: clamp |velocity| (degenerate-polyhedron guard; needed when V0 grows)
        self.compile = bool(params.get("compile", False))    # torch.compile the vectorised shape energy
        self.retess_every = max(1, int(params.get("retess_every", 1)))  # topology refresh cadence (multi-rate; 1=exact/tick)
        self._topo = None; self._tick = 0                    # cached CONNECTIVITY (tet + hull tris); ghosts recomputed/tick

    def _topology(self, pos_np, n, dev, dtype):
        """Tessellate + per-cell hull -> cached connectivity (tet, tri_idx, mask, ok). The expensive
        part (Delaunay + N ConvexHulls); reused for `retess_every` ticks, refreshed on cell-count change."""
        ghosts_np = ghost_points(pos_np, self.lumen, self.pad)
        allp_np, tet, incident = _tessellate(pos_np, ghosts_np)
        cc_np = _circ_np(allp_np[tet])
        rows, ok, Fmax = [], np.zeros(n, np.float32), 4
        for i in range(n):
            inc = incident[i]
            if len(inc) < 4:
                rows.append(None); continue
            try:
                g = np.asarray(inc)[ConvexHull(cc_np[inc]).simplices]
            except Exception:
                rows.append(None); continue
            rows.append(g); ok[i] = 1.0; Fmax = max(Fmax, len(g))
        tri_idx = np.zeros((n, Fmax, 3), np.int64); tmask = np.zeros((n, Fmax), np.float32)
        for i in range(n):
            if rows[i] is not None:
                F = len(rows[i]); tri_idx[i, :F] = rows[i]; tmask[i, :F] = 1.0
        return dict(n=n, ng=len(ghosts_np),
                    tet=torch.as_tensor(tet, device=dev),
                    tri_idx=torch.as_tensor(tri_idx, device=dev),
                    tmask=torch.as_tensor(tmask, device=dev, dtype=dtype),
                    okt=torch.as_tensor(ok, device=dev))

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
        # multi-rate topology cache: rebuild connectivity every `retess_every` ticks OR when the cell
        # count changes (division). The boundary GHOSTS + circumcentres are recomputed EVERY tick from
        # the current positions (cheap), so the boundary tracks inflation; only the connectivity lags.
        self._tick += 1
        if (self._topo is None or self._topo["n"] != n or self._tick % self.retess_every == 0):
            self._topo = self._topology(pos_np, n, dev, pos_full.dtype)
        T = self._topo
        ghosts_t = torch.as_tensor(ghost_points(pos_np, self.lumen, self.pad), device=dev, dtype=pos_full.dtype)
        if ghosts_t.shape[0] != T["ng"]:                     # ghost count drifted -> force a rebuild
            self._topo = self._topology(pos_np, n, dev, pos_full.dtype); T = self._topo
            ghosts_t = torch.as_tensor(ghost_points(pos_np, self.lumen, self.pad), device=dev, dtype=pos_full.dtype)
        tet_t, tri_idx_t, tmask_t, okt = T["tet"], T["tri_idx"], T["tmask"], T["okt"]
        v0 = (lvl.get("v0")[idx, 0] if "v0" in lvl.state_schema
              else torch.full((n,), self.V0, device=dev))
        S0 = self.s0 * v0.clamp(min=1e-6) ** (2.0 / 3.0)
        efn = _energy_fn_3d(self.compile)
        with torch.enable_grad():
            pos = pos_live.detach().requires_grad_(True)
            E = efn(pos, ghosts_t, tet_t, tri_idx_t, tmask_t, okt, v0, S0, self.K_V, self.K_S)
            grad = torch.autograd.grad(E, pos)[0]
        v = self.mu * torch.nan_to_num(-grad)
        if self.vmax > 0:                                     # clamp per-cell speed (boundary-degeneracy guard)
            vn = v.norm(dim=1, keepdim=True)
            v = torch.where(vn > self.vmax, v * (self.vmax / vn.clamp(min=1e-9)), v)
        v_full[idx] = v
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}

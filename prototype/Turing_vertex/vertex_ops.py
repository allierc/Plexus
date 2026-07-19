"""vertex_ops -- the MECHANICS half of the Turing x vertex prototype (2D), plexus2.

A confluent 2D tissue as a Self-Propelled Voronoi / active vertex model (Bi et al.
PRX 2016; Barton et al. PLoS CB 2017): each cell is a CENTRE, the tissue is the
Voronoi tessellation of the centres (re-tessellated each tick -> automatic T1s), and
the mechanics come from a cell shape energy

    E = sum_i [ K_A (A_i - A0_i)^2 + K_P (P_i - P0)^2 ]

with A_i, P_i the cell's Voronoi area and perimeter. Force = -grad E by autodiff
through differentiable circumcentres; overdamped (EMIT=velocity). The target shape
index p0 = P0/sqrt(A0) sets a rigidity transition at ~3.81 (solid below, fluid above).

Why SPV for the coupled model: the Voronoi re-tessellation gives BOTH the polygon
geometry (mechanics) AND the Delaunay cell-cell adjacency (the RD diffusion graph),
so one tessellation feeds `voronoi_tension` and `graph_diffuse`. Per-cell target area
A0 is the coupling handle: the activator raises A0 -> cells grow -> tissue deforms.

Operators:
  tissue_seed      (structural) -- place cell centres in a periodic box; init A0
  voronoi_graph    (rewire)     -- re-tessellate: cell.edge_index (Delaunay) + rings
  voronoi_tension  (lateral)    -- shape-energy force on the centres (EMIT=velocity)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Delaunay

from plexus.models.base import Lateral, Structural, Rewire
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Voronoi geometry helpers (periodic, differentiable circumcentres)
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


def periodic_voronoi(pos_np, L, N):
    """Periodic Voronoi of N centres in [0,L)^2 via a 3x3 tiling + Delaunay. Returns
    tri [M,3] (into the tiled array), tiled_orig [9N], tiled_shift [9N,2], ring_idx
    [N,Vmax] (triangle index per cell's ordered vertex ring, padded), ok [N]."""
    shifts = np.array([[dx * L, dy * L] for dx in (0, -1, 1) for dy in (0, -1, 1)], dtype=np.float64)
    tiled_orig = np.tile(np.arange(N), 9)
    tiled_shift = np.repeat(shifts, N, axis=0)
    tiled = pos_np[tiled_orig] + tiled_shift
    tri = Delaunay(tiled, qhull_options="QJ").simplices
    cc = _circ_np(tiled[tri])
    central = tri < N
    ts_idx, k_idx = np.where(central)
    cells = tri[ts_idx, k_idx]; tri_id = ts_idx
    order = np.argsort(cells, kind="stable")
    cells_s, tri_s = cells[order], tri_id[order]
    bounds = np.searchsorted(cells_s, np.arange(N + 1))
    rings, Vmax = [None] * N, 3
    for i in range(N):
        ts = tri_s[bounds[i]:bounds[i + 1]]
        if len(ts) >= 3:
            v = cc[ts]
            ang = np.arctan2(v[:, 1] - pos_np[i, 1], v[:, 0] - pos_np[i, 0])
            rings[i] = ts[np.argsort(ang)]; Vmax = max(Vmax, len(ts))
    ring_idx = np.zeros((N, Vmax), dtype=np.int64); ok = np.zeros(N, dtype=np.float32)
    for i in range(N):
        r = rings[i]
        if r is not None:
            ring_idx[i, :len(r)] = r; ring_idx[i, len(r):] = r[-1]; ok[i] = 1.0
    return tri, tiled_orig, tiled_shift, ring_idx, ok


def delaunay_edges(tri, tiled_orig, N):
    """Undirected central cell-cell Delaunay neighbour pairs -> edge_index [2,E] (symmetric)."""
    orig = tiled_orig[tri]
    pairs = set()
    for a, b, c in orig:
        for u, v in ((a, b), (b, c), (a, c)):
            if u != v and u < N and v < N:
                pairs.add((u, v) if u < v else (v, u))
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    e = np.array(sorted(pairs)).T
    return np.concatenate([e, e[::-1]], axis=1)


def cell_polygons(pos_np, L, N):
    """Voronoi polygon, area, perimeter per central cell (rendering + diagnostics)."""
    tri, torig, tshift, ring_idx, ok = periodic_voronoi(pos_np, L, N)
    tiled = pos_np[torig] + tshift
    cc = _circ_np(tiled[tri])
    polys, area, perim = [], np.zeros(N), np.zeros(N)
    for i in range(N):
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
        perim[i] = np.sqrt(((np.roll(vv, -1, 0) - vv) ** 2).sum(1)).sum()
    return polys, area, perim, ok


# --------------------------------------------------------------------------- #
#  Seed: place cell centres, init target area
# --------------------------------------------------------------------------- #
@register_operator("tissue_seed", set="cell", kind="structural", family="growth")
class TissueSeed(Structural):
    """Frame-0 IC (`before_frame: 1`): scatter N cell centres in the periodic box
    [0,L)^2 (jittered grid) and set every cell's target area A0 (state block `a0`)."""
    SUPPORTED_DIMS = [2]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["tissue", "initial_condition", "confluent"]
    PARAM_ROLES = {"a0": "target_cell_area", "jitter": "grid_disorder"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.a0 = float(params.get("a0", 1.0))
        self.jitter = float(params.get("jitter", 0.25))
        self.lattice = params.get("lattice", "triangular")   # triangular (hex Voronoi) | square

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        N = lvl.state.shape[0]
        dev = lvl.state.device
        L = float(H.world_size[0])
        g = torch.Generator(device="cpu"); g.manual_seed(0)
        if self.lattice == "triangular":                     # -> hexagonal Voronoi ground state
            ncol = max(2, int(round((N * 2.0 / math.sqrt(3.0)) ** 0.5)))
            nrow = max(2, int(round(N / ncol)))
            rr, cc = torch.meshgrid(torch.arange(nrow), torch.arange(ncol), indexing="ij")
            off = 0.5 * (rr % 2).float()                     # offset alternate rows -> triangular
            x = ((cc.float() + off) * (L / ncol)).reshape(-1) % L
            y = (rr.float() * (L / nrow)).reshape(-1)
            cen = torch.stack([x, y], 1)
        else:                                                # square grid -> square Voronoi
            m = int(math.ceil(math.sqrt(N)))
            gx, gy = torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
            cen = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).float()
            cen = (cen + 0.5) / m * L
        cen = cen[:N] if cen.shape[0] >= N else torch.cat([cen, cen[: N - cen.shape[0]]])
        cen = cen + (torch.rand(N, 2, generator=g) - 0.5) * self.jitter * (L / N ** 0.5)
        pos = (cen % L).to(dev)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos
        if "a0" in lvl.state_schema:
            a0, a1 = lvl.state_schema["a0"]; st[:, a0:a1] = self.a0
        lvl.state = st
        return {}


# --------------------------------------------------------------------------- #
#  Rewire: re-tessellate -> Delaunay adjacency (the RD graph) + polygon rings
# --------------------------------------------------------------------------- #
@register_operator("voronoi_graph", set="cell", kind="rewire", family="topology")
class VoronoiGraph(Rewire):
    """Re-tessellate the live cell centres each tick: set `cell.edge_index` to the
    Delaunay cell-cell adjacency (the graph `graph_diffuse` runs the Turing RD on) and
    stash the Voronoi ring connectivity (`_tri`, `_ring`, `_torig`, `_tshift`, `_ok`,
    `_live`) that `voronoi_tension` differentiates for the shape-energy force. The
    tessellation is rebuilt from the live configuration, so T1 neighbour exchanges are
    automatic (they fall out of re-tessellating)."""
    SUPPORTED_DIMS = [2]
    DIFFERENTIABLE = False
    MECHANISM_TAGS = ["voronoi", "delaunay", "retessellation", "T1_transition", "confluent"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        L = float(H.world_size[0])
        live = lvl.occ > 0
        idx = live.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n < 4:
            lvl.edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
            lvl._vg = None
            return {}
        pos_np = (lvl.get("pos")[idx].detach().cpu().numpy().astype(np.float64)) % L
        tri, torig, tshift, ring, ok = periodic_voronoi(pos_np, L, n)
        ei_local = delaunay_edges(tri, torig, n)             # [2,E] into LIVE index
        ei = idx.cpu().numpy()[ei_local] if ei_local.size else ei_local
        lvl.edge_index = torch.as_tensor(ei, dtype=torch.long, device=dev)
        # stash the connectivity the tension operator needs (numpy/torch, live-indexed)
        lvl._vg = dict(idx=idx,
                       tri=torch.as_tensor(tri, device=dev),
                       torig=torch.as_tensor(torig, device=dev, dtype=torch.long),
                       tshift=torch.as_tensor(tshift, device=dev, dtype=lvl.state.dtype),
                       ring=torch.as_tensor(ring, device=dev),
                       ok=torch.as_tensor(ok, device=dev), n=n, L=L)
        return {}


# --------------------------------------------------------------------------- #
#  Lateral: shape-energy force on the cell centres (overdamped)
# --------------------------------------------------------------------------- #
@register_operator("voronoi_tension", set="cell", kind="lateral", family="mechanics")
class VoronoiTension(Lateral):
    """Self-Propelled-Voronoi shape-energy force on the cell centres:
    E = sum K_A (A - A0)^2 + K_P (P - P0)^2 over the Voronoi polygons (connectivity from
    `voronoi_graph`), force = -grad E by autodiff through the circumcentres, overdamped
    (EMIT=velocity). A0 is per-cell (state block `a0`, the growth handle); P0 = p0*sqrt(A0)."""
    SUPPORTED_DIMS = [2]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["cell"]; OUTPUTS = ["cell"]
    READS = ["pos", "a0"]; WRITES = ["pos"]
    MAPS = ["edge_index"]                                    # the Voronoi/Delaunay connectivity
    MECHANISM_TAGS = ["vertex_model", "self_propelled_voronoi", "shape_energy",
                      "rigidity_transition", "confluent_tissue", "morphogenesis"]
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "mu": "mobility"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.K_A = float(params.get("K_A", 1.0))
        self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.85))
        self.mu = float(params.get("mu", 1.0))
        self.A0 = float(params.get("A0", 1.0))               # uniform target area if no per-cell `a0` state
        self.v0 = float(params.get("v0", 0.0))               # self-propulsion speed (0 = pure relaxation)
        self.Dr = float(params.get("Dr", 1.0))               # rotational diffusion of the polarity
        self.dt = float(params.get("dt", 0.05))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        vg = getattr(lvl, "_vg", None)
        if vg is None:
            return {self.at: v_full}
        idx, n, L = vg["idx"], vg["n"], vg["L"]
        tri, torig, tshift, ring, ok = vg["tri"], vg["torig"], vg["tshift"], vg["ring"], vg["ok"]
        a0 = (lvl.get("a0")[idx, 0] if "a0" in lvl.state_schema
              else torch.full((n,), self.A0, device=pos_full.device))
        P0 = self.p0 * a0.clamp(min=1e-6).sqrt()
        with torch.enable_grad():
            pos = (pos_full[idx].detach() % L).requires_grad_(True)
            tiled = pos[torig] + tshift
            cc = _circ_torch(tiled[tri])
            verts = cc[ring]
            nxt = torch.roll(verts, -1, dims=1)
            area = 0.5 * (verts[..., 0] * nxt[..., 1] - nxt[..., 0] * verts[..., 1]).sum(1).abs()
            perim = (nxt - verts).norm(dim=-1).sum(1)
            E = ((self.K_A * (area - a0) ** 2 + self.K_P * (perim - P0) ** 2) * ok).sum()
            grad = torch.autograd.grad(E, pos)[0]
        v_full[idx] = self.mu * torch.nan_to_num(-grad) * ok[:, None]
        # self-propulsion along a rotationally-diffusing polarity -> the tissue can FLOW
        # (T1s) above the rigidity transition; below it, the jammed solid resists.
        if self.v0 > 0:
            Nf = pos_full.shape[0]; rng = getattr(H, "rng", None)
            if not hasattr(lvl, "theta"):
                lvl.register_buffer("theta", 2 * math.pi * torch.rand(Nf, generator=rng, device=pos_full.device))
            th = lvl.theta + math.sqrt(2 * self.Dr * self.dt) * torch.randn(Nf, generator=rng, device=pos_full.device)
            lvl.theta = th
            n_hat = torch.stack([torch.cos(th), torch.sin(th)], 1)
            v_full[idx] = v_full[idx] + self.v0 * n_hat[idx]
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}

"""tyssue_ops -- a TRUE vertex-model (AVM) implementation of epithelial mechanics, plexus2.

This is the sibling implementation of the Turing_vertex prototype's Self-Propelled
Voronoi (SPV) mechanics. Same biological contract -- an epithelial shape energy

    E = sum_f [ K_A (A_f - A0_f)^2 + K_P (P_f - P0_f)^2 ]  (+ line tension, contractility)

-- but a genuinely different NUMERICAL implementation (plexus2 sec. 5):

  SPV (Turing_vertex/vertex_ops.py) : the cell is a POINT; the polygon is its Voronoi
      cell; the mesh is RE-TESSELLATED every tick, so T1 neighbour exchanges are IMPLICIT
      (they fall out of the Delaunay flip). DOF = cell centres.

  AVM (here, after DamCB/tyssue)     : the DOF are the VERTICES of a half-edge mesh
      (vert / edge(srce,trgt,face) / face); cells SHARE edges; T1 is an EXPLICIT local
      operation (collapse a short edge, split the merged vertex); mechanics is a
      quasistatic force balance (gradient descent to residual) between topology events.

Why bother: the Turing_vertex report found that clean tubulation was gated on exactly the
two ingredients the SPV route lacks -- (i) force-balance iteration and (ii) an explicit
reconnection (T1) operator -- both of which the vertex-mesh AVM supplies natively. We also
VECTORISE it: per-face area/perimeter come from a single scatter-add over the half-edge
table, and the force is one autograd pass -- so this AVM has none of tyssue's per-cell
pandas/scipy cost (the report's other complaint) and runs on the GPU.

Master topology = the half-edge table (E_srce, E_trgt, E_face), ordered CCW around each
face, stashed on the vertex Level (like VoronoiGraph stashes its rings). Per-face targets
(A0, P0) and liveness live alongside it. The seed bootstraps a clean honeycomb by taking
the Voronoi tessellation of a triangular lattice ONCE, then the AVM takes over.

Operators:
  seed_mesh       (structural) -- build a honeycomb half-edge mesh on the `vertex` set;
                                   stash the edge table + per-face A0/P0/alive/pin on H.
  shape_energy  (lateral)    -- AVM shape-energy force on vertices (vectorised scatter-add
                                   + autograd), with an inner overdamped relax loop so each
                                   frame is a quasistatic step. EMIT=velocity.

(t1_transition / face_divide / face_extrude are added once the mechanics are validated.)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import Voronoi

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


# --------------------------------------------------------------------------- #
#  Honeycomb half-edge mesh from a Voronoi tessellation of a triangular lattice
# --------------------------------------------------------------------------- #
def hex_lattice(nx, ny, a=1.0):
    """Triangular lattice of cell CENTRES (rows offset by a/2). Voronoi of these is a
    honeycomb of regular hexagons with centre spacing `a`."""
    xs, ys = [], []
    dy = a * math.sqrt(3.0) / 2.0
    for j in range(ny):
        for i in range(nx):
            xs.append(i * a + (j % 2) * (a / 2.0))
            ys.append(j * dy)
    return np.stack([np.array(xs), np.array(ys)], 1)


def build_honeycomb(nx, ny, a=1.0, border=1, jitter=0.0, seed=0):
    """Return a bounded honeycomb half-edge mesh.

    vertices : [Nv,2] float   -- the Voronoi vertices actually used by live faces
    E_srce/E_trgt/E_face : [E] -- half-edges, CCW around each face
    face_center : [F,2]        -- the seeding cell centre of each LIVE face
    pin : [Nv]  bool           -- vertices on the outer boundary (held fixed)
    a0 : float                 -- regular-hexagon area at spacing a (target area)

    The outer `border` rings of cells are dropped (their Voronoi regions are open / ragged);
    the vertices they share with kept cells are pinned, giving a fixed-border patch -- the
    standard bounded-sheet setup in tyssue's 2D demos.
    """
    centers = hex_lattice(nx, ny, a)
    if jitter > 0:
        g = np.random.default_rng(seed)
        centers = centers + (g.random((centers.shape[0], 2)) - 0.5) * jitter * a
    vor = Voronoi(centers)

    # which seeding cells are "interior" (away from the ragged Voronoi border)
    cx = centers[:, 0]; cy = centers[:, 1]
    xlo, xhi = cx.min() + border * a, cx.max() - border * a
    ylo = cy.min() + border * a * math.sqrt(3) / 2; yhi = cy.max() - border * a * math.sqrt(3) / 2

    faces_rings = []       # list of ordered vertex-index rings (into vor.vertices)
    face_center = []
    for i, c in enumerate(centers):
        if not (xlo <= c[0] <= xhi and ylo <= c[1] <= yhi):
            continue
        reg = vor.regions[vor.point_region[i]]
        if len(reg) < 3 or -1 in reg:
            continue
        ring = np.array(reg, dtype=np.int64)
        v = vor.vertices[ring]
        ang = np.arctan2(v[:, 1] - c[1], v[:, 0] - c[0])        # order CCW around the centre
        ring = ring[np.argsort(ang)]
        faces_rings.append(ring)
        face_center.append(c)

    # compact the vertex set to only those used by kept faces
    used = np.unique(np.concatenate(faces_rings))
    remap = -np.ones(vor.vertices.shape[0], dtype=np.int64); remap[used] = np.arange(used.size)
    vertices = vor.vertices[used]

    E_srce, E_trgt, E_face = [], [], []
    for f, ring in enumerate(faces_rings):
        r = remap[ring]
        for k in range(len(r)):
            E_srce.append(int(r[k])); E_trgt.append(int(r[(k + 1) % len(r)])); E_face.append(f)
    E_srce = np.array(E_srce, np.int64); E_trgt = np.array(E_trgt, np.int64); E_face = np.array(E_face, np.int64)

    # pin boundary vertices: those incident to fewer half-edges than their interior degree
    # (an interior honeycomb vertex has 3 incident faces -> 3 outgoing half-edges).
    deg = np.bincount(E_srce, minlength=vertices.shape[0])
    pin = deg < 3

    a0 = (math.sqrt(3) / 2.0) * a * a                            # regular-hexagon area
    return (vertices.astype(np.float64), E_srce, E_trgt, E_face,
            np.array(face_center, np.float64), pin, a0)


# --------------------------------------------------------------------------- #
#  Seed: build the mesh, place vertices, stash topology on the Level
# --------------------------------------------------------------------------- #
@register_operator("seed_mesh", set="vertex", kind="structural", family="growth")
class MeshSeed(Structural):
    """Frame-0 IC (`before_frame: 1`): build a bounded honeycomb half-edge mesh, write the
    vertex positions into the `vertex` set, and stash the edge table + per-face targets on
    the Level (`lvl._mesh`). The `vertex` set is the mechanical DOF; faces are carried as a
    buffer, exactly as VoronoiGraph carries its Voronoi rings."""
    SUPPORTED_DIMS = [2]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["epithelium", "vertex_model", "honeycomb", "half_edge_mesh", "initial_condition"]
    PARAM_ROLES = {"nx": "lattice_cols", "ny": "lattice_rows", "a": "cell_spacing",
                   "jitter": "positional_disorder", "p0": "target_shape_index"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.nx = int(params.get("nx", 16))
        self.ny = int(params.get("ny", 18))
        self.a = float(params.get("a", 1.0))
        self.border = int(params.get("border", 1))
        self.jitter = float(params.get("jitter", 0.0))
        self.p0 = float(params.get("p0", 3.85))
        self.seed = int(params.get("seed", 0))
        self.pin_border = bool(params.get("pin_border", True))   # False -> free boundary (tissue can expand/grow)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device; dt = lvl.state.dtype
        verts, es, et, ef, fc, pin, a0 = build_honeycomb(
            self.nx, self.ny, self.a, self.border, self.jitter, self.seed)
        Nv = verts.shape[0]; Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"mesh has {Nv} vertices but buffer n={Nbuf}; raise sets.vertex.n")
        pos = torch.zeros(Nbuf, 2, dtype=dt, device=dev)
        pos[:Nv] = torch.as_tensor(verts, dtype=dt, device=dev)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos; lvl.state = st
        # occupancy: only the real vertices are live
        if hasattr(lvl, "occ") and lvl.occ is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        if not self.pin_border:
            pin = np.zeros(Nv, dtype=bool)                       # free boundary: tissue can expand as it grows
        nF = int(ef.max()) + 1 if ef.size else 0
        A0 = torch.full((nF,), a0, dtype=dt, device=dev)
        P0 = self.p0 * A0.clamp(min=1e-9).sqrt()
        lvl._mesh = dict(
            E_srce=torch.as_tensor(es, device=dev), E_trgt=torch.as_tensor(et, device=dev),
            E_face=torch.as_tensor(ef, device=dev), nF=nF, Nv=Nv,
            A0=A0, P0=P0, alive=torch.ones(nF, dtype=dt, device=dev),
            pin=torch.as_tensor(pin, device=dev), face_center=torch.as_tensor(fc, dtype=dt, device=dev),
        )
        return {}


# --------------------------------------------------------------------------- #
#  Mechanics: vectorised AVM shape-energy force on the vertices (force-balanced)
# --------------------------------------------------------------------------- #
def _face_geometry(pos, es, et, ef, nF):
    """Vectorised per-face signed area & perimeter from the half-edge table.
    area = 1/2 sum_edges (x_s y_t - x_t y_s) ; perim = sum_edges |t - s|. CCW => area > 0."""
    s = pos[es]; t = pos[et]
    cross = s[:, 0] * t[:, 1] - t[:, 0] * s[:, 1]
    length = (t - s).norm(dim=-1)
    area = 0.5 * torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, cross)
    perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, length)
    return area, perim


@register_operator("shape_energy", set="vertex", kind="lateral", family="mechanics")
class ShapeEnergy(Lateral):
    """True vertex-model shape-energy force on the mesh vertices (the AVM implementation of
    the epithelial-mechanics contract). Energy per face f:

        K_A (A_f - A0_f)^2 + K_P (P_f - P0_f)^2  [+ 0.5 Gamma P_f^2 + Lambda * l_e / 2]

    with A_f, P_f the polygon area/perimeter from the half-edge table (`seed_mesh`). Force =
    -grad E on the vertices by one autograd pass; overdamped (EMIT=velocity). To emulate
    tyssue's quasistatic MINIMISE-to-residual between topology events, an inner loop takes
    `relax_iters` gradient steps per tick and returns the net displacement as a velocity, so
    each rendered frame is (near-)force-balanced rather than a single explicit Euler step --
    exactly the force-balance iteration the Turing_vertex report identified as missing.
    Boundary vertices (`pin`) are held fixed."""
    SUPPORTED_DIMS = [2]
    EMIT = "velocity"
    DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]
    READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "active_vertex_model", "shape_energy",
                      "force_balance", "rigidity_transition", "confluent_tissue"]
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "Gamma": "contractility", "Lambda": "line_tension", "mu": "mobility",
                   "relax_iters": "force_balance_iterations", "eta": "relax_step"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.K_A = float(params.get("K_A", 1.0))
        self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.85))
        self.Gamma = float(params.get("Gamma", 0.0))
        self.Lambda = float(params.get("Lambda", 0.0))
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 0.05))
        self.relax_iters = int(params.get("relax_iters", 1))
        self.eta = float(params.get("eta", 0.05))              # inner GD step (fraction of overdamped)
        self.v0 = float(params.get("v0", 0.0))                 # self-propulsion speed (0 = pure relaxation)
        self.Dr = float(params.get("Dr", 1.0))                 # rotational diffusion of the polarity
        self.cap_frac = float(params.get("cap_frac", 0.15))    # max per-substep move as a fraction of the
        #                                                        mean edge length -> stable overdamped step
        #                                                        (EulerSolver): no vertex can cross a cell in
        #                                                        one step, so the floppy regime cannot diverge

    def _energy(self, pos, m):
        area, perim = _face_geometry(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        alive = m["alive"]
        E = (self.K_A * (area.abs() - m["A0"]) ** 2 + self.K_P * (perim - m["P0"]) ** 2)
        if self.Gamma:
            E = E + 0.5 * self.Gamma * perim ** 2
        E = (E * alive).sum()
        if self.Lambda:
            s = pos[m["E_srce"]]; t = pos[m["E_trgt"]]
            E = E + self.Lambda * 0.5 * (t - s).norm(dim=-1).sum()
        return E

    def _grad(self, pos, m):
        with torch.enable_grad():                              # engine calls forward() under no_grad
            p = pos.detach().requires_grad_(True)
            E = self._energy(p, m)
            g = torch.autograd.grad(E, p)[0]
        return torch.nan_to_num(g)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        # cross-set read: if the model declares a genuine `cell` set, the per-cell target areas
        # a0 are BIOLOGICAL state living there (not on the vertex mesh), so shape_energy reads a0
        # from the cell set and broadcasts the resulting force back to the vertices -- a two-level
        # (vertex <-> cell) operator. Without a cell set it falls back to the mesh buffer (M1).
        try:
            from tyssue_cell_ops import cell_level
            clvl = cell_level(H)
        except Exception:
            clvl = None
        if clvl is not None and "a0" in getattr(clvl, "state_schema", {}):
            a0c = clvl.get("a0")[:m["nF"], 0].to(m["A0"].dtype)
            m["A0"] = a0c
            m["P0"] = self.p0 * a0c.clamp(min=1e-9).sqrt()
        Nv = m["Nv"]; pin = m["pin"]
        free = (~pin).to(pos_full.dtype)[:, None]
        x0 = pos_full[:Nv].detach().clone()
        x = x0.clone()
        with torch.no_grad():                                  # displacement cap = fraction of mean edge
            Lmean = (x[m["E_trgt"]] - x[m["E_srce"]]).norm(dim=1).mean()
        cap = self.cap_frac * torch.clamp(Lmean, min=1e-6)

        def _capped(step):                                     # bound every move -> no cell inverts in one step
            nrm = step.norm(dim=1, keepdim=True)
            return step * torch.clamp(cap / (nrm + 1e-12), max=1.0)

        # stable overdamped relaxation toward force balance (bounded step: EulerSolver / QSSolver)
        for _ in range(max(1, self.relax_iters)):
            g = self._grad(x, m)
            x = x + _capped(-(self.eta * self.mu) * g) * free
        # active self-propulsion (also bounded): fluid FLOWS via T1, solid stays caged
        if self.v0 > 0:
            dev = pos_full.device; rng = getattr(H, "rng", None)
            th = getattr(lvl, "theta", None)
            if th is None or th.shape[0] < Nv:
                th = 2 * math.pi * torch.rand(Nv, generator=rng, device=dev)
            th = th + math.sqrt(2 * self.Dr * self.dt) * torch.randn(Nv, generator=rng, device=dev)
            lvl.theta = th
            n_hat = torch.stack([torch.cos(th), torch.sin(th)], 1)
            x = x + _capped(self.v0 * self.dt * n_hat) * free
        v_full[:Nv] = (x - x0) / max(self.dt, 1e-9)             # engine does x += dt*v -> x
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


# --------------------------------------------------------------------------- #
#  Diagnostics helper (shared by the driver): per-face polygons for rendering
# --------------------------------------------------------------------------- #
def face_polygons(pos_np, mesh):
    """Return (polys, area, perim, shape_index) for the live faces, from the half-edge table.
    `mesh` is the dict stashed on the Level (numpy views)."""
    es, et, ef = mesh["E_srce"], mesh["E_trgt"], mesh["E_face"]
    nF = mesh["nF"]
    polys = [[] for _ in range(nF)]
    for k in range(len(ef)):
        polys[int(ef[k])].append(int(es[k]))           # ordered srce ring per face (CCW)
    out_polys, area, perim = [], np.zeros(nF), np.zeros(nF)
    for f in range(nF):
        ring = polys[f]
        v = pos_np[ring]
        out_polys.append(v)
        x, y = v[:, 0], v[:, 1]
        area[f] = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        perim[f] = np.sqrt(((np.roll(v, -1, 0) - v) ** 2).sum(1)).sum()
    shape = np.where(area > 1e-9, perim / np.sqrt(np.maximum(area, 1e-9)), np.nan)
    return out_polys, area, perim, shape

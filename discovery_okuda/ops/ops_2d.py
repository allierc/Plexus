"""ops_2d -- a TRUE vertex-model (AVM) implementation of epithelial mechanics, plexus2.

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

def _face_geometry(pos, es, et, ef, nF):
    """Vectorised per-face signed area & perimeter from the half-edge table.
    area = 1/2 sum_edges (x_s y_t - x_t y_s) ; perim = sum_edges |t - s|. CCW => area > 0."""
    s = pos[es]; t = pos[et]
    cross = s[:, 0] * t[:, 1] - t[:, 0] * s[:, 1]
    length = (t - s).norm(dim=-1)
    area = 0.5 * torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, cross)
    perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, length)
    return area, perim



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

"""tyssue_ops3d -- the 3D (surface) vertex model: an epithelial VESICLE. A closed half-edge mesh on
a sphere (spherical Voronoi), vertices in 3D, cells are polygons on the surface. The AVM shape
energy is computed with 3D geometry (Newell area vector, 3D edge lengths) plus a LUMEN-VOLUME term
that keeps the shell inflated against surface tension -- the closure a 2D bounded patch got from
pinning. This is the true-vertex-model sibling of Turing_vertex's spherical-Voronoi vesicle, and the
substrate for morphogen-driven budding/tubulation in 3D.

Operators:
  seed_mesh_3d    (structural) -- build a closed spherical half-edge mesh; stash it + A0/P0/V0
  shape_energy_3d (lateral)    -- 3D AVM shape energy + lumen volume; force by autodiff; bounded Euler
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import SphericalVoronoi

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator


def fib_sphere(n, r=1.0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); theta = np.pi * (1 + 5 ** 0.5) * i
    return r * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


def build_sphere_mesh(n, r=1.0, jitter=0.0, seed=0):
    """Closed spherical half-edge mesh from a spherical Voronoi of a (jittered) Fibonacci sphere.
    Returns vertices [Nv,3], E_srce/E_trgt/E_face, nF. Every edge is shared by exactly two faces."""
    pts = fib_sphere(n, r)
    if jitter > 0:
        g = np.random.default_rng(seed)
        pts = pts + jitter * (g.random((n, 3)) - 0.5)
        pts = r * pts / np.linalg.norm(pts, axis=1, keepdims=True)
    sv = SphericalVoronoi(pts, radius=r, center=np.zeros(3)); sv.sort_vertices_of_regions()
    faces = [np.array(reg, np.int64) for reg in sv.regions]
    V = sv.vertices                                          # reorient every face CCW as seen from OUTSIDE
    for i, rr in enumerate(faces):                           # (Newell normal points away from the centre) so the
        P = V[rr]; N = np.cross(P, np.roll(P, -1, 0)).sum(0) # mesh is consistently oriented: each directed edge
        if float(np.dot(N, P.mean(0))) < 0:                  # appears once (needed by the topology ops) and every
            faces[i] = rr[::-1]                              # per-cell wedge volume is positive.
    es, et, ef = [], [], []
    for f, rr in enumerate(faces):
        k = len(rr)
        for i in range(k):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % k])); ef.append(f)
    return (sv.vertices.astype(np.float64), np.array(es, np.int64), np.array(et, np.int64),
            np.array(ef, np.int64), len(faces))


def face_geometry_3d(pos, es, et, ef, nF):
    """Per-face 3D area (Newell area-vector magnitude), perimeter, centroid, and the PER-CELL wedge
    volume v_f = (1/3)(cen_f . N_f) -- the volume of the pyramid from the sphere centre to the face.
    The lumen volume is just sum_f v_f, but keeping it per-cell lets each cell carry its own volume
    elasticity (a distributed term that resists local buckling). All differentiable in `pos`."""
    s = pos[es]; t = pos[et]
    length = (t - s).norm(dim=-1)
    cross = torch.cross(s, t, dim=-1)                        # consecutive-vertex cross products
    dev, dt = pos.device, pos.dtype
    N = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, cross)   # area vector / face
    area = N.norm(dim=-1)
    perim = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, length)
    cnt = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, torch.ones_like(length))
    cen = torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, s) / cnt.clamp(min=1)[:, None]
    vf = (1.0 / 3.0) * (cen * N).sum(dim=-1)                 # per-cell wedge volume (sum = lumen volume)
    return area, perim, cen, vf


@register_operator("seed_mesh_3d", set="vertex", kind="structural", family="growth")
class SeedMesh3D(Structural):
    """Frame-0: build a closed spherical half-edge mesh (spherical Voronoi), write the 3D vertex
    positions, and stash the edge table + per-face targets (A0, P0) and the lumen target V0."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["vesicle", "epithelial_shell", "spherical", "half_edge_mesh", "initial_condition"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.n = int(params.get("n_cells", 150)); self.R = float(params.get("radius", 5.0))
        self.jitter = float(params.get("jitter", 0.15)); self.p0 = float(params.get("p0", 3.9))
        self.seed = int(params.get("seed", 0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device; dt = lvl.state.dtype
        verts, es, et, ef, nF = build_sphere_mesh(self.n, self.R, self.jitter, self.seed)
        Nv = verts.shape[0]; Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"sphere mesh has {Nv} vertices but buffer n={Nbuf}")
        pos = torch.zeros(Nbuf, 3, dtype=dt, device=dev)
        pos[:Nv] = torch.as_tensor(verts, dtype=dt, device=dev)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos; lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        est = torch.as_tensor(es, device=dev); ett = torch.as_tensor(et, device=dev)
        eft = torch.as_tensor(ef, device=dev)
        area, perim, cen, vf = face_geometry_3d(pos[:Nv], est, ett, eft, nF)
        A0 = float(area.mean()); P0 = self.p0 * (A0 ** 0.5)
        lvl._mesh = dict(E_srce=est, E_trgt=ett, E_face=eft, nF=nF, Nv=Nv,
                         A0=torch.full((nF,), A0, dtype=dt, device=dev),
                         P0=torch.full((nF,), P0, dtype=dt, device=dev),
                         alive=torch.ones(nF, dtype=dt, device=dev),
                         V0f=vf.detach().clone(),               # PER-CELL target wedge volume (v_eq per cell)
                         Vbirth=vf.detach().clone(),            # volume at birth -> cell divides when it doubles
                         V0=float(vf.sum()),
                         R0=float(np.linalg.norm(verts, axis=1).mean()), verts0=verts)
        return {}


@register_operator("shape_energy_3d", set="vertex", kind="lateral", family="mechanics")
class ShapeEnergy3D(Lateral):
    """3D AVM shape-energy force on the vesicle vertices:
        E = sum_f [ K_A(A_f-A0)^2 + K_P(P_f-P0)^2 + K_V(v_f - v_eq_f)^2 ] + Lambda*sum_e l_e .
    K_V is a PER-CELL volume elasticity on each cell's wedge volume v_f (Turing_vertex Eq.3 / tyssue
    ClosedMonolayer), not a single global lumen term: it keeps every cell inflated and resists local
    buckling, so growth (ramping v_eq per cell) inflates the shell smoothly. Force = -grad E by one 3D
    autograd pass; bounded overdamped Euler (displacement capped at cap_frac x mean edge). EMIT=velocity."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "shape_energy", "cell_volume_elasticity", "vesicle", "force_balance"]
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "Lambda": "surface_tension", "K_V": "cell_volume_elasticity", "cap_frac": "stability_cap"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.K_A = float(params.get("K_A", 1.0)); self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.9)); self.Lambda = float(params.get("Lambda", 0.1))
        self.K_V = float(params.get("K_V", 0.5)); self.K_R = float(params.get("K_R", 0.0))
        self.Gamma = float(params.get("Gamma", 0.0))             # cortical contractility (1/2)Gamma*P^2 -> rounds cells
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0)); self.relax_iters = int(params.get("relax_iters", 6))
        self.eta = float(params.get("eta", 0.08)); self.cap_frac = float(params.get("cap_frac", 0.12))
        # Lloyd-like tangential regularization (AVM analog of Turing's surface_lloyd): rounds cells
        self.smooth_iters = int(params.get("smooth_iters", 0)); self.smooth_w = float(params.get("smooth_w", 0.0))

    def _energy(self, pos, m):
        area, perim, cen, vf = face_geometry_3d(pos, m["E_srce"], m["E_trgt"], m["E_face"], m["nF"])
        alive = m["alive"]
        E = (self.K_A * (area - m["A0"]) ** 2 + self.K_P * (perim - m["P0"]) ** 2) * alive
        if self.Gamma:                                           # cortical contractility (rounds cells; tyssue FaceContractility)
            E = E + 0.5 * self.Gamma * perim ** 2 * alive
        E = E.sum() + self.Lambda * (pos[m["E_trgt"]] - pos[m["E_srce"]]).norm(dim=-1).sum()
        # PER-CELL volume elasticity (Turing_vertex Eq.3 / tyssue ClosedMonolayer): each cell holds its
        # own wedge volume near v_eq, so growth (ramping v_eq per cell) inflates every cell locally and
        # a local dip is penalised -- the shell expands smoothly and resists buckling, no kinematic push.
        E = E + self.K_V * ((vf - m["V0f"]) ** 2 * alive).sum()
        if self.K_R:                                             # soft radial constraint (a bending-like term):
            E = E + self.K_R * ((pos.norm(dim=1) - m["R0"]) ** 2).sum()   # keeps vertices near the shell -> smooth
        return E

    def _grad(self, pos, m):
        with torch.enable_grad():
            p = pos.detach().requires_grad_(True)
            g = torch.autograd.grad(self._energy(p, m), p)[0]
        return torch.nan_to_num(g)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = m["Nv"]; x0 = pos_full[:Nv].detach().clone(); x = x0.clone()
        with torch.no_grad():
            cap = self.cap_frac * (x[m["E_trgt"]] - x[m["E_srce"]]).norm(dim=-1).mean().clamp(min=1e-6)
        for _ in range(max(1, self.relax_iters)):
            step = -(self.eta * self.mu) * self._grad(x, m)
            nrm = step.norm(dim=1, keepdim=True)
            x = x + step * torch.clamp(cap / (nrm + 1e-12), max=1.0)
        if self.smooth_iters and self.smooth_w > 0:      # Lloyd-like tangential regularization -> rounder cells
            es, et = m["E_srce"], m["E_trgt"]
            ones = torch.ones(es.shape[0], device=x.device, dtype=x.dtype)
            with torch.no_grad():
                for _ in range(self.smooth_iters):
                    nbr = torch.zeros_like(x).index_add(0, es, x[et])          # sum of edge-neighbours per vertex
                    deg = torch.zeros(Nv, device=x.device, dtype=x.dtype).index_add(0, es, ones).clamp(min=1)
                    xs = (1 - self.smooth_w) * x + self.smooth_w * (nbr / deg[:, None])
                    x = xs * (x.norm(dim=1, keepdim=True) / xs.norm(dim=1, keepdim=True).clamp(min=1e-9))  # keep on shell
        v_full[:Nv] = (x - x0) / max(self.dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


@register_operator("vesicle_growth", set="vertex", kind="structural", family="growth")
class VesicleGrowth(Structural):
    """Grow the vesicle self-similarly by ramping only the per-cell TARGETS (no vertex is moved by
    hand -- the expansion emerges from shape_energy_3d's force balance). Each tick a linear scale grows
    by `rate`, so per cell: A0 <- A0*(1+rate)^2, P0 <- P0*(1+rate), v_eq <- v_eq*(1+rate)^3, and the
    shell radius R0 <- R0*(1+rate) (area~R^2, volume~R^3). The per-cell volume elasticity then pushes
    each cell's vertices outward to restore its target wedge volume, so the whole shell inflates
    smoothly and resists buckling -- the physically-correct mechanism, matching tyssue's ClosedMonolayer
    (grow prefered_vol, minimise) and Turing_vertex Eq.3 (grow v_eq)."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["growth", "isotropic_inflation", "vesicle", "volume_target_scaling"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.rate = float(params.get("rate", 0.004)); self.every = int(params.get("every", 1))
        self._k = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1
        if self._k % self.every != 0:
            return {}
        g = 1.0 + self.rate
        m["A0"] = m["A0"] * (g * g)                              # area ~ radius^2
        m["P0"] = m["P0"] * g                                    # perimeter ~ radius
        m["V0f"] = m["V0f"] * (g ** 3)                           # per-cell target volume ~ radius^3
        m["V0"] = float(m["V0f"].sum())
        m["R0"] = float(m["R0"]) * g                             # shell radius ~ radius
        return {}


@register_operator("divide_3d", set="vertex", kind="structural", family="growth")
class Divide3D(Structural):
    """In-surface cell division on the vesicle -- the sheet-division analog (tyssue
    sheet_topology.cell_division) lifted to the closed sphere. A cell divides when its wedge volume
    reaches `factor` x its BIRTH volume (the volume-doubling cell cycle): it is split by an
    edge-midpoint septum (divide_face_3d), which also splits the shared edges of its two neighbours so
    the mesh stays closed (Euler=2). Each daughter inherits half the area & target volume and resets
    its birth volume to half; shape_energy_3d re-balances."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["division", "cell_division", "vesicle", "proliferation", "volume_doubling"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.factor = float(params.get("factor", 2.0))           # divide when volume >= factor x birth volume
        self.reset_noise = float(params.get("reset_noise", 0.12))  # per-cell threshold jitter -> staggered (gradual) divisions
        self.p0 = float(params.get("p0", 3.72))
        self.every = int(params.get("every", 3)); self._k = 0
        self.max_div = int(params.get("max_div", 20))            # cap divisions per call for stability

    def forward(self, H, mask=None):
        from tyssue_topology_ops3d import rings_from_flat_3d, flat_from_rings_3d, divide_face_3d
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1
        if self._k % self.every != 0:
            return {}
        dev = lvl.state.device; dt = lvl.state.dtype; buf = lvl.state.shape[0]
        Nv = m["Nv"]
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].detach().cpu().numpy(); et = m["E_trgt"].detach().cpu().numpy()
        ef = m["E_face"].detach().cpu().numpy(); nF = int(m["nF"])
        _, _, _, vf = face_geometry_3d(torch.as_tensor(pos_np), torch.as_tensor(es),
                                       torch.as_tensor(et), torch.as_tensor(ef), nF)
        vf = vf.numpy()                                          # per-cell CURRENT wedge volume
        rings = rings_from_flat_3d(es, et, ef, nF)
        pos = [p for p in pos_np]
        A0 = m["A0"].detach().cpu().numpy().tolist()
        V0f = m["V0f"].detach().cpu().numpy().tolist()
        Vbirth = m["Vbirth"].detach().cpu().numpy().tolist()
        alive = m["alive"].detach().cpu().numpy().tolist()
        rng = np.random.default_rng(12345 + self._k)
        djit = m.get("divjit")                                   # per-cell threshold jitter -> staggered divisions
        if djit is None:
            djit = (1.0 + self.reset_noise * (np.random.default_rng(777).random(nF) * 2 - 1)).tolist()
        else:
            djit = djit.detach().cpu().numpy().tolist()
        cand = [f for f in range(nF) if alive[f] > 0 and vf[f] >= self.factor * djit[f] * Vbirth[f]
                and rings[f] is not None and len(rings[f]) >= 4]
        rng.shuffle(cand)                                        # unbiased when more cells are ready than max_div
        cand = cand[:self.max_div]                               # (else cand[:n] sweeps in pole-to-pole face order)
        ndone = 0
        for f in cand:
            if len(pos) + 2 > buf:                                # respect the vertex buffer
                break
            r = rings[f]
            P = np.array([pos[v] for v in r]); c = P.mean(0)      # LONG-AXIS (Hertwig) split: septum perpendicular
            try:                                                 # to the cell's longest axis -> compact daughters
                _, _, vh = np.linalg.svd(P - c, full_matrices=False); u = vh[0]
                n = c / (np.linalg.norm(c) + 1e-9)               # outward (radial) face normal on the sphere
                w = np.cross(n, u); w = w / (np.linalg.norm(w) + 1e-9)   # short-axis direction in the tangent plane
                mids = 0.5 * (P + np.roll(P, -1, 0)); proj = (mids - c) @ w
                ea, eb = int(np.argmax(proj)), int(np.argmin(proj))
            except Exception:
                ea, eb = 0, len(r) // 2
            res = divide_face_3d(rings, pos, f, ea=ea, eb=eb)
            if res is None:
                continue
            half = vf[f] * 0.5                                    # each daughter is born at half the volume
            A0[f] *= 0.5; V0f[f] *= 0.5; Vbirth[f] = half         # daughter A (kept at index f)
            djit[f] = 1.0 + self.reset_noise * (rng.random() * 2 - 1)                       # fresh jittered thresholds
            A0.append(A0[f]); V0f.append(V0f[f]); Vbirth.append(half); alive.append(1.0)   # daughter B
            djit.append(1.0 + self.reset_noise * (rng.random() * 2 - 1))
            ndone += 1
        if ndone == 0:
            return {}
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        A0a = np.array([A0[i] for i in keep], np.float64)
        V0fa = np.array([V0f[i] for i in keep], np.float64)
        Vba = np.array([Vbirth[i] for i in keep], np.float64)
        dja = np.array([djit[i] for i in keep], np.float64)
        alv = np.array([alive[i] for i in keep], np.float64)
        P0a = self.p0 * np.sqrt(np.maximum(A0a, 1e-9))
        Nv2 = len(pos)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone()
        st[:Nv2, px0:px1] = torch.as_tensor(np.asarray(pos), dtype=dt, device=dev)
        lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(buf, device=dev); occ[:Nv2] = 1.0; lvl.occ = occ
        m["E_srce"] = torch.as_tensor(es2, device=dev); m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev); m["nF"] = nF2; m["Nv"] = Nv2
        m["A0"] = torch.as_tensor(A0a, dtype=dt, device=dev)
        m["P0"] = torch.as_tensor(P0a, dtype=dt, device=dev)
        m["V0f"] = torch.as_tensor(V0fa, dtype=dt, device=dev)
        m["Vbirth"] = torch.as_tensor(Vba, dtype=dt, device=dev)
        m["divjit"] = torch.as_tensor(dja, dtype=dt, device=dev)
        m["alive"] = torch.as_tensor(alv, dtype=dt, device=dev)
        m["n_div"] = int(m.get("n_div", 0)) + ndone
        return {}


@register_operator("topo_snapshot_3d", set="vertex", kind="structural", family="growth")
class TopoSnapshot3D(Structural):
    """Record the current mesh (flat half-edge table + vertex count) each frame, so a growing/dividing
    vesicle -- whose topology changes over time -- can be rendered frame by frame."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False

    def __init__(self, params, device="cpu"):
        super().__init__(params, device); self.at = params.get("_at", "vertex")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        m.setdefault("hist", []).append(dict(
            E_srce=m["E_srce"].detach().cpu().numpy().copy(),
            E_trgt=m["E_trgt"].detach().cpu().numpy().copy(),
            E_face=m["E_face"].detach().cpu().numpy().copy(),
            nF=int(m["nF"]), Nv=int(m["Nv"])))
        return {}


def face_polygons_3d(pos_np, mesh):
    """3D face polygons + per-face area/perimeter/shape-index (for rendering)."""
    es, et, ef, nF = mesh["E_srce"], mesh["E_trgt"], mesh["E_face"], mesh["nF"]
    rings = [[] for _ in range(nF)]
    for k in range(len(ef)):
        rings[int(ef[k])].append(int(es[k]))
    polys, area, perim = [], np.zeros(nF), np.zeros(nF)
    for f in range(nF):
        v = pos_np[rings[f]]; polys.append(v)
        N = 0.5 * np.abs(np.cross(v, np.roll(v, -1, 0)).sum(0))
        area[f] = np.linalg.norm(0.5 * np.cross(v, np.roll(v, -1, 0)).sum(0))
        perim[f] = np.linalg.norm(np.roll(v, -1, 0) - v, axis=1).sum()
    shape = np.where(area > 1e-9, perim / np.sqrt(np.maximum(area, 1e-9)), np.nan)
    return polys, area, perim, shape

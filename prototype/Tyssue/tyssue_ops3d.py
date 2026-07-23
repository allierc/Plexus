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


def face_geometry_3d(pos, es, et, ef, nF, eocc=None):
    """Per-face 3D area (Newell area-vector magnitude), perimeter, centroid, and the PER-CELL wedge
    volume v_f = (1/3)(cen_f . N_f) -- the volume of the pyramid from the sphere centre to the face.
    The lumen volume is just sum_f v_f, but keeping it per-cell lets each cell carry its own volume
    elasticity (a distributed term that resists local buckling). All differentiable in `pos`."""
    s = pos[es]; t = pos[et]
    length = (t - s).norm(dim=-1)
    cross = torch.cross(s, t, dim=-1)                        # consecutive-vertex cross products
    dev, dt = pos.device, pos.dtype
    if eocc is not None:                                     # reservoir: zero out dead (padding) half-edges
        length = length * eocc; cross = cross * eocc[:, None]; sw = s * eocc[:, None]; w = eocc
    else:
        sw = s; w = torch.ones_like(length)
    N = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, cross)   # area vector / face
    area = N.norm(dim=-1)
    perim = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, length)
    cnt = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, w)
    cen = torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, sw) / cnt.clamp(min=1)[:, None]
    vf = (1.0 / 3.0) * (cen * N).sum(dim=-1)                 # per-cell wedge volume (sum = lumen volume)
    return area, perim, cen, vf


def _shape_energy_core(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, K_A, K_P, K_V, K_R, Lam, Gam,
                       eocc, vocc, K_bend=0.0, twin_face=None, K_lumen=0.0):
    """Explicit-arg AVM shape energy on a FIXED-size RESERVOIR (torch.compile-friendly: shapes never
    change, so it compiles once even under division). Dead slots are masked out: `alive` (faces),
    `eocc` (half-edges), `vocc` (vertices, for the radial term). R0 is a tensor (changes each frame);
    the K_* / Lam / Gam coefficients are compile-time constants."""
    area, perim, cen, vf = face_geometry_3d(pos, es, et, ef, nF, eocc)
    E = (K_A * (area - A0) ** 2 + K_P * (perim - P0) ** 2 + 0.5 * Gam * perim ** 2) * alive
    line = (pos[et] - pos[es]).norm(dim=-1) * eocc          # line tension over live half-edges only
    E = E.sum() + Lam * line.sum()
    E = E + K_V * ((vf - V0f) ** 2 * alive).sum()
    E = E + K_R * (((pos.norm(dim=1) - R0) ** 2) * vocc).sum()   # radial over live vertices only
    if K_bend > 0 and twin_face is not None:
        # DIHEDRAL bending (Wardetzky hinge): penalise the angle between a cell face's outward normal and
        # the normal of the cell across each shared edge -> smooths sharp cell-to-cell FOLDS (the hollow /
        # inverted-cap tilt the metric flags), high-frequency, WITHOUT flattening gentle whole-shell curvature
        # (unlike the radial K_R). Newell area vector per face -> unit normal; sum over half-edges (edge twice).
        s = pos[es]; t = pos[et]
        crossN = torch.cross(s, t, dim=-1) * eocc[:, None]
        Nf = 0.5 * torch.zeros(nF, 3, device=pos.device, dtype=pos.dtype).index_add(0, ef, crossN)
        nhat = Nf / (Nf.norm(dim=-1, keepdim=True) + 1e-12)
        cosang = (nhat[ef] * nhat[twin_face]).sum(dim=-1)       # cos(dihedral) per half-edge
        E = E + K_bend * ((1.0 - cosang) * eocc).sum()
    if K_lumen > 0:
        # GLOBAL LUMEN INCOMPRESSIBILITY: penalise the shell for enclosing LESS volume than a sphere of
        # its current total area. A buckled/wrinkled shell ALWAYS encloses less (isoperimetric inequality),
        # so this is the one constraint that distinguishes a sphere from a per-cell-volume-preserving
        # buckle -- which the per-cell wedge term (a pyramid from the origin) and local area/perim cannot.
        # Relative deficit -> scale-invariant. (Coral only; a tube is NOT the max-volume shape -> off there.)
        A_tot = (area * alive).sum()
        V_sphere = A_tot ** 1.5 / (6.0 * 3.14159265358979 ** 0.5)
        V_tot = (vf * alive).sum()
        E = E + K_lumen * ((V_tot - V_sphere) / (V_sphere + 1e-9)) ** 2
    return E


def _relax_subset(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, mech, move_mask, iters):
    """Bounded-Euler shape-energy descent that moves ONLY the vertices in `move_mask` (the fresh
    daughters + their one-ring after a division). Heals the just-cut caps in place so a division never
    leaves an inverted cap for the global relaxation to (fail to) fix. Reuses `_shape_energy_core`."""
    dev, dt = pos.device, pos.dtype
    eocc = torch.ones(es.shape[0], device=dev, dtype=dt); vocc = torch.ones(pos.shape[0], device=dev, dtype=dt)
    R0t = torch.as_tensor(float(R0), device=dev, dtype=dt)
    with torch.no_grad():
        cap = mech["cap_frac"] * (pos[et] - pos[es]).norm(dim=-1).mean().clamp(min=1e-6)
    mm = move_mask.to(dt)[:, None]
    x = pos.clone()
    for _ in range(iters):
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            E = _shape_energy_core(xg, es, et, ef, nF, A0, P0, V0f, alive, R0t, mech["K_A"], mech["K_P"],
                                   mech["K_V"], mech["K_R"], mech["Lambda"], mech["Gamma"], eocc, vocc)
            g = torch.nan_to_num(torch.autograd.grad(E, xg)[0])
        step = -mech["eta"] * g
        step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
        x = x + step * mm                                        # only the fresh region moves
    return x.detach()


@register_operator("seed_mesh_3d", set="vertex", kind="structural", family="growth")
class SeedMesh3D(Structural):
    """Frame-0: build a closed spherical half-edge mesh (spherical Voronoi), write the 3D vertex
    positions, and stash the edge table + per-face targets (A0, P0) and the lumen target V0."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["vesicle", "epithelial_shell", "spherical", "half_edge_mesh", "initial_condition"]
    REFERENCE = "Okuda, S. et al. (2013). Reversible network reconnection model for simulating large deformation in 3D tissues. Biomech. Model. Mechanobiol. 12:627-644; tyssue (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.n = int(params.get("n_cells", 150)); self.R = float(params.get("radius", 5.0))
        self.jitter = float(params.get("jitter", 0.15)); self.p0 = float(params.get("p0", 3.9))
        self.seed = int(params.get("seed", 0))
        self.vseed_cv = float(params.get("vseed_cv", 0.0))       # STOCHASTIC VOLUME SEED: per-cell random cell-cycle
        #   phase at t=0 (spread of the initial division threshold) -> desynchronises the FIRST division wave

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
        if self.vseed_cv > 0:                                    # random initial cell-cycle phase per cell
            dj = np.clip(1.0 + self.vseed_cv * np.random.default_rng(self.seed + 101).standard_normal(nF), 0.4, 1.8)
        else:
            dj = np.ones(nF)                                     # all cells born in phase (synchronised)
        lvl._mesh = dict(E_srce=est, E_trgt=ett, E_face=eft, nF=nF, Nv=Nv,
                         A0=torch.full((nF,), A0, dtype=dt, device=dev),
                         P0=torch.full((nF,), P0, dtype=dt, device=dev),
                         alive=torch.ones(nF, dtype=dt, device=dev),
                         divjit=torch.as_tensor(dj, dtype=dt, device=dev),   # per-cell division-threshold multiplier
                         V0f=vf.detach().clone(),               # PER-CELL target wedge volume (v_eq per cell)
                         Vbirth=vf.detach().clone(),            # volume at birth -> cell divides when it doubles
                         V0=float(vf.sum()),
                         v_ref=float(vf.median()),              # REFERENCE cell volume (Okuda v_ref) -> uniform cells:
                         #   morphogen growth caps v_eq at (4/3)v_ref, cells cycle in [2/3,4/3]v_ref centred on v_ref
                         R0=float(np.linalg.norm(verts, axis=1).mean()), verts0=verts,
                         # RESERVOIR fixed sizes for the compiled mechanics (verts<=Nbuf; faces~V/2; half-edges~3V)
                         Nv_max=Nbuf, nF_max=Nbuf // 2 + 64, Ebuf=4 * Nbuf)
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
    REFERENCE = "Farhadifar, R. et al. (2007). Curr. Biol. 17:2095-2104 (vertex-model shape energy); Okuda, S. et al. (2015). Biomech. Model. Mechanobiol. 14:413-421 (3D volume/surface)."
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "Lambda": "surface_tension", "K_V": "cell_volume_elasticity", "cap_frac": "stability_cap"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.K_A = float(params.get("K_A", 1.0)); self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.9)); self.Lambda = float(params.get("Lambda", 0.1))
        self.K_V = float(params.get("K_V", 0.5)); self.K_R = float(params.get("K_R", 0.0))
        # DIHEDRAL bending (Wardetzky): penalises adjacent-cell normal deviation -> smooths the local folds
        # the hollow metric flags (division-injected cap tilts). High-frequency: unlike the radial K_R it
        # does NOT flatten gentle whole-shell/tube curvature. 0 = off (default). See _shape_energy_core.
        self.K_bend = float(params.get("K_bend", 0.0))
        # GLOBAL LUMEN incompressibility (isoperimetric): penalise enclosing less volume than a sphere of
        # the current area -> distinguishes sphere from a per-cell-volume-preserving buckle. 0=off. Coral only.
        self.K_lumen = float(params.get("K_lumen", 0.0))
        self.Gamma = float(params.get("Gamma", 0.0))             # cortical contractility (1/2)Gamma*P^2 -> rounds cells
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0)); self.relax_iters = int(params.get("relax_iters", 6))
        self.eta = float(params.get("eta", 0.08)); self.cap_frac = float(params.get("cap_frac", 0.12))
        # ANTI-INVERSION filtered step (IPC-analog, differentiable): hollow caps are inverting faces
        # (signed wedge volume v_f flips sign at the division septum). Each bounded-Euler substep, scale
        # back the move of any vertex whose incident face would drop v_f below `antiinv` x median(v_f) --
        # a move that only makes an already-inverted face WORSE is blocked, a recovering move is allowed.
        # Straight-through (scale detached) so the rollout stays differentiable. 0 = off (default).
        self.antiinv = float(params.get("antiinv", 0.0))
        # Lloyd-like tangential regularization (AVM analog of Turing's surface_lloyd): rounds cells
        self.smooth_iters = int(params.get("smooth_iters", 0)); self.smooth_w = float(params.get("smooth_w", 0.0))
        # torch.compile the (autograd-differentiated) energy: ~2.4x on a FIXED mesh, but DIVISION changes
        # nF every other frame -> torch.compile recompiles each time -> 20x SLOWER. So default OFF; only
        # enable (compile=True) for fixed-topology runs (static RD, growth without division).
        # torch.compile the energy via a fixed-size RESERVOIR (occ-masked padding so shapes never change
        # under division). Measured: a CLEAR ~2.4x win only for FIXED-topology runs (buffer == live count
        # -> no padding); for growing/dividing meshes the padding computes over the oversized buffer and
        # the one-time compile cost only amortizes over very long runs, so it is net SLOWER. Hence
        # OPT-IN (compile=True) -- use it for the static RD; the default is the fast live-only path.
        self.use_compile = bool(params.get("compile", False))
        self._efn = torch.compile(_shape_energy_core, dynamic=False) if self.use_compile else _shape_energy_core

    def _antiinv_scale(self, x, step, es, et, ef, nF, eocc, vf_ref, floor):
        """Straight-through per-vertex step scale (detached) that halves the move of any vertex whose
        incident face would drop its signed wedge volume below `floor` AND below its current value ---
        i.e. block moves that push a face toward inversion, allow recovering moves. A few backtracks."""
        scale = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        for _ in range(5):
            _, _, _, vf = face_geometry_3d(x + step * scale, es, et, ef, nF, eocc)
            bad = (vf < floor) & (vf < vf_ref)                   # per-face: inverting and getting worse
            if not bool(bad.any()):
                break
            badv = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            badv.index_add_(0, es, bad.to(x.dtype)[ef] * eocc)   # half-edge src vertices of the bad faces
            scale = torch.where((badv > 0)[:, None], scale * 0.5, scale)
        return scale.detach()

    def _grad(self, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc, twin_face=None):
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            E = self._efn(p, es, et, ef, nF, A0, P0, V0f, alive, R0t, self.K_A, self.K_P,
                          self.K_V, self.K_R, self.Lambda, self.Gamma, eocc, vocc, self.K_bend,
                          twin_face, self.K_lumen)
            g = torch.autograd.grad(E, p)[0]
        return torch.nan_to_num(g)

    @staticmethod
    def _twin_faces(es, et, ef, Nv):
        """Per half-edge -> the face on the OTHER side of that (undirected) edge, for the dihedral term.
        Twin of half-edge (u->v) is the half-edge (v->u); match by integer key. Closed mesh: always found."""
        key = es * Nv + et; twinkey = et * Nv + es
        order = torch.argsort(key); ks = key[order]
        pos = torch.searchsorted(ks, twinkey).clamp(max=key.shape[0] - 1)
        found = ks[pos] == twinkey
        return torch.where(found, ef[order[pos]], ef)          # fallback to self (no penalty) if no twin

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos")
        v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = int(m["Nv"]); nF = int(m["nF"]); es = m["E_srce"]; et = m["E_trgt"]; ef = m["E_face"]
        E = es.shape[0]; dev = pos_full.device; dt = pos_full.dtype
        m["mech"] = dict(K_A=self.K_A, K_P=self.K_P, K_V=self.K_V, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.Gamma, eta=self.eta, cap_frac=self.cap_frac)   # for divide_3d local relax
        x0 = pos_full[:Nv].detach().clone()
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        if self.use_compile:
            # RESERVOIR path: pad to fixed buffer sizes (dead slots masked by occ) -> compiled once
            Nvm = int(m.get("Nv_max", Nv)); Fm = int(m.get("nF_max", nF)); Em = int(m.get("Ebuf", E))
            z = torch.zeros
            xp = z(Nvm, 3, device=dev, dtype=dt); xp[:Nv] = x0
            esp = z(Em, dtype=torch.long, device=dev); esp[:E] = es
            etp = z(Em, dtype=torch.long, device=dev); etp[:E] = et
            efp = z(Em, dtype=torch.long, device=dev); efp[:E] = ef
            eocc = z(Em, device=dev, dtype=dt); eocc[:E] = 1.0
            vocc = z(Nvm, device=dev, dtype=dt); vocc[:Nv] = 1.0
            A0p = z(Fm, device=dev, dtype=dt); A0p[:nF] = m["A0"]
            P0p = z(Fm, device=dev, dtype=dt); P0p[:nF] = m["P0"]
            V0fp = z(Fm, device=dev, dtype=dt); V0fp[:nF] = m["V0f"]
            alivep = z(Fm, device=dev, dtype=dt); alivep[:nF] = m["alive"]
            for _ in range(max(1, self.relax_iters)):
                step = -(self.eta * self.mu) * self._grad(xp, esp, etp, efp, Fm, A0p, P0p, V0fp,
                                                          alivep, R0t, eocc, vocc)
                xp = xp + step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            x = xp[:Nv]
        else:
            # LIVE-ONLY path (default, fast): relax the live vertices, energy over live faces/edges
            eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
            twin = self._twin_faces(es, et, ef, Nv) if self.K_bend > 0 else None   # dihedral neighbour faces
            x = x0.clone()
            floor = None
            if self.antiinv > 0:                                 # anti-inversion floor = frac of median live wedge vol
                _, _, _, vf0 = face_geometry_3d(x, es, et, ef, nF, eocc)
                floor = self.antiinv * (vf0[vf0 > 0].median() if (vf0 > 0).any() else vf0.new_tensor(1e-9)).clamp(min=1e-9)
            for _ in range(max(1, self.relax_iters)):
                step = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, m["A0"], m["P0"],
                                                          m["V0f"], m["alive"], R0t, eocc, vocc, twin)
                step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
                if floor is not None:                            # block any substep that drives a face toward inversion
                    _, _, _, vf_cur = face_geometry_3d(x, es, et, ef, nF, eocc)
                    step = step * self._antiinv_scale(x, step, es, et, ef, nF, eocc, vf_cur, floor)
                x = x + step
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
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.rate = float(params.get("rate", 0.004)); self.every = int(params.get("every", 1))
        self.max_scale = float(params.get("max_scale", 1e9))    # cap the linear growth -> the shell PLATEAUS
        self._k = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1
        if self._k % self.every != 0:
            return {}
        gs = float(m.get("gscale", 1.0))
        if gs >= self.max_scale:                                # plateaued: stop inflating (division then stops too)
            return {}
        g = 1.0 + self.rate
        m["gscale"] = gs * g                                     # cumulative linear scale (radius factor)
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
    REFERENCE = "Hertwig, O. (1884) (long-axis division rule); tyssue cell_division (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.factor = float(params.get("factor", 2.0))           # divide when volume >= factor x birth volume
        self.reset_noise = float(params.get("reset_noise", 0.12))  # per-cell threshold jitter -> staggered (gradual) divisions
        self.cycle_cv = float(params.get("cycle_cv", 0.0))       # STOCHASTIC CELL CYCLE: Gaussian CV of each daughter's
        #   cell-cycle length (fresh division threshold). >0 keeps division waves broken up (desynchronised) as the
        #   tissue proliferates -- essential at scale so max-rate division never outruns relaxation. 0 -> uniform reset_noise.
        self.p0 = float(params.get("p0", 3.72))
        self.every = int(params.get("every", 3)); self._k = 0
        self.max_div = int(params.get("max_div", 20))            # cap divisions per call for stability (absolute floor)
        # LIVE-scaled division cap: max_div is otherwise a FIXED absolute count set from the INITIAL cell count,
        # so on a long run the live count grows (150->~1400) while the cap stays 10 -> ready cells backlog behind
        # it, keep ramping v_eq while queued, then divide oversized -> tip strain -> hollow caps. A fractional cap
        # bounds the division RATE (<= frac of live cells per call), so the cap grows with the tissue. Off (0) by
        # default so existing presets are unchanged.
        self.max_div_frac = float(params.get("max_div_frac", 0.0))
        # HARD CELL-SIZE CAP: force-divide ANY cell whose current volume >= vcap x v_ref, BYPASSING the
        # per-call throttle (max_div/max_div_frac) so oversized cells never backlog and keep growing. This
        # bounds the maximum cell size directly (the tube-tip cells that grew "far too big"). 0 = off.
        self.vcap = float(params.get("vcap", 0.0))
        # LOCAL DAUGHTER RELAX: after the septum, run a few bounded-Euler shape-energy steps on ONLY the
        # fresh daughters + their one-ring, so a division never hands the global relaxation an inverted
        # cap (the sole source of hollow cells -- growth alone never makes one). Uses the coeffs stashed
        # on m["mech"] by shape_energy_3d. 0 = off (default); the coral/tube fix sets it > 0.
        self.local_relax = int(params.get("local_relax", 0))
        # DIVISION MODEL = volume-primary + bounded DURATION (Okuda: divide at 2x volume, but the cell-cycle
        # PERIOD is constrained). min_cycle: a cell may not divide before this many division-calls since birth
        # (a fast-growing tip cell can't divide instantly -> tighter size CV); max_cycle: force division after
        # this many calls even if volume < 2x (a stalled cell still cycles). 0/inf = pure volume-doubling.
        self.min_cycle = int(params.get("min_cycle", 0)); self.max_cycle = int(params.get("max_cycle", 10 ** 9))
        self.cell_set = params.get("cell_set", None)             # if set, daughters inherit the mother's cell state (morphogen)
        # G1 RAMP (SimuCell3D/tyssue "birth-at-target"): set each daughter's TARGET volume v_eq to its ACTUAL
        # birth volume instead of mother_target/2. The division trigger is ACTUAL volume (vf>=2*Vbirth) but v_eq
        # is set by ramped morphogen growth, so an actively-growing tip cell has mother_V0f >> vf; halving it
        # leaves a fresh daughter with target >> actual and (since K_V dominates tension 5-50x) K_V drives the
        # tiny face hard -> inverted/hollow caps at the proliferating tip. Birth-at-target removes the mismatch;
        # morphogen_growth_3d then re-ramps activated daughters as their G1 regrowth. Off by default so other
        # presets (vesicle_divide/fig4) are unchanged; the tube preset turns it on.
        self.g1_ramp = bool(params.get("g1_ramp", False))
        # ORIENTED division at the red/white interface (Okuda's tube mechanism, user issue 3): the dividing
        # plane of an ACTIVATED (red) cell is oriented so the daughters stack ALONG the bud axis (the
        # direction from the vesicle centre to the activated tip) instead of by the cell's own long axis.
        # This adds new cells NORMAL to the body surface -> builds the tube WALL and EXTENDS the protrusion,
        # rather than widening it. orient_asw = activator threshold that flags an interface/red cell. 0 = off.
        self.orient_iface = bool(params.get("orient_iface", False))
        self.orient_asw = float(params.get("orient_asw", 1.0))

    def _fresh_djit(self, rng, n=1):
        """Fresh per-cell division-threshold multiplier. Gaussian CV (cycle_cv) when set -> desynchronised
        cell cycles; otherwise the legacy uniform +/-reset_noise jitter. Clamped so thresholds stay sane."""
        if self.cycle_cv > 0:
            v = np.clip(1.0 + self.cycle_cv * rng.standard_normal(n), 0.4, 1.8)
        else:
            v = 1.0 + self.reset_noise * (rng.random(n) * 2 - 1)
        return v if n > 1 else float(v[0])

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
            djit = self._fresh_djit(np.random.default_rng(777), nF).tolist()
        else:
            djit = djit.detach().cpu().numpy().tolist()
        age = m.get("age")                                       # per-cell age in division-calls since birth
        age = ([0] * nF) if (age is None or age.shape[0] != nF) else (age.detach().cpu().numpy() + 1).tolist()
        # volume-primary + bounded duration: divide if (2x volume AND old enough) OR (past max cycle length)
        def _ready(f):
            if rings[f] is None or len(rings[f]) < 4 or alive[f] <= 0:
                return False
            vol_ok = vf[f] >= self.factor * djit[f] * Vbirth[f]
            return (vol_ok and age[f] >= self.min_cycle) or (age[f] >= self.max_cycle)
        cand = [f for f in range(nF) if _ready(f)]
        rng.shuffle(cand)                                        # unbiased when more cells are ready than the cap
        cap_div = self.max_div if self.max_div_frac <= 0 else max(self.max_div, int(self.max_div_frac * nF))
        if self.vcap > 0:                                        # HARD size cap: oversized cells ALWAYS divide (not throttled)
            vref = float(m.get("v_ref", 0.0)) or (float(np.median(vf[vf > 0])) if (vf > 0).any() else 1.0)
            over = [f for f in range(nF) if alive[f] > 0 and rings[f] is not None and len(rings[f]) >= 4 and vf[f] >= self.vcap * vref]
            oset = set(over); cand = over + [f for f in cand if f not in oset]   # oversized first
            cap_div = max(cap_div, len(over))                    # never throttle oversized cells
        cand = cand[:cap_div]                                    # (else cand[:n] sweeps in pole-to-pole face order)
        ndone = 0
        daughter_mothers = []                                    # mother face index of each appended daughter (order)
        bud_axis = None; a_cells = None                          # ORIENTED interface division: bud axis = centre->red tip
        if self.orient_iface and self.cell_set is not None:
            clvl0 = H.level(self.cell_set)
            if clvl0 is not None and "chem" in clvl0.state_schema:
                ci0, _ = clvl0.state_schema["chem"]
                a_cells = clvl0.state[:nF, ci0].detach().cpu().numpy()
                rc = [np.array([pos[v] for v in rings[f]]).mean(0) for f in range(nF)
                      if a_cells[f] > self.orient_asw and rings[f] is not None and len(rings[f]) >= 3]
                if len(rc):
                    ba = np.mean(rc, 0); nba = float(np.linalg.norm(ba))
                    if nba > 1e-6:
                        bud_axis = ba / nba
        for f in cand:
            if len(pos) + 2 > buf:                                # respect the vertex buffer
                break
            r = rings[f]
            P = np.array([pos[v] for v in r]); c = P.mean(0)      # LONG-AXIS (Hertwig) split: septum perpendicular
            try:                                                 # to the cell's longest axis -> compact daughters
                _, _, vh = np.linalg.svd(P - c, full_matrices=False); u = vh[0]
                n = c / (np.linalg.norm(c) + 1e-9)               # outward (radial) face normal on the sphere
                if bud_axis is not None and a_cells is not None and f < len(a_cells) and a_cells[f] > self.orient_asw:
                    ut = bud_axis - float(np.dot(bud_axis, n)) * n   # bud axis in this cell's tangent plane ->
                    if np.linalg.norm(ut) > 1e-6:                    # daughters separate ALONG the protrusion (extend the
                        u = ut / np.linalg.norm(ut)                  # wall) instead of by the cell's own long axis
                w = np.cross(n, u); w = w / (np.linalg.norm(w) + 1e-9)   # short-axis direction in the tangent plane
                mids = 0.5 * (P + np.roll(P, -1, 0)); proj = (mids - c) @ w
                ea, eb = int(np.argmax(proj)), int(np.argmin(proj))
            except Exception:
                ea, eb = 0, len(r) // 2
            res = divide_face_3d(rings, pos, f, ea=ea, eb=eb)
            if res is None:
                continue
            half = vf[f] * 0.5                                    # each daughter is born at half the actual volume
            if self.g1_ramp:                                     # birth-at-target: v_eq = actual birth volume (no K_V mismatch);
                iso = A0[f] / max(V0f[f], 1e-12) ** (2.0 / 3.0)  # keep A0 isoperimetric-consistent A0 ~ v_eq^{2/3}
                a0d = iso * half ** (2.0 / 3.0); v0d = half       # (P0 = p0*sqrt(A0) recomputed below)
            else:
                a0d = A0[f] * 0.5; v0d = V0f[f] * 0.5             # legacy: half the mother's targets
            A0[f] = a0d; V0f[f] = v0d; Vbirth[f] = half           # daughter A (kept at index f)
            djit[f] = self._fresh_djit(rng); age[f] = 0           # fresh (desync'd) thresholds; reset cell-cycle age
            A0.append(a0d); V0f.append(v0d); Vbirth.append(half); alive.append(1.0)   # daughter B
            djit.append(self._fresh_djit(rng)); age.append(0)
            daughter_mothers.append(f)
            ndone += 1
        if ndone == 0:
            m["age"] = torch.as_tensor(np.asarray(age), dtype=dt, device=dev)   # persist ageing even without division
            return {}
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        A0a = np.array([A0[i] for i in keep], np.float64)
        V0fa = np.array([V0f[i] for i in keep], np.float64)
        Vba = np.array([Vbirth[i] for i in keep], np.float64)
        dja = np.array([djit[i] for i in keep], np.float64)
        agea = np.array([age[i] for i in keep], np.float64)
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
        m["age"] = torch.as_tensor(agea, dtype=dt, device=dev)
        m["alive"] = torch.as_tensor(alv, dtype=dt, device=dev)
        m["n_div"] = int(m.get("n_div", 0)) + ndone
        if self.local_relax > 0 and "mech" in m and Nv2 > Nv:    # heal the fresh caps in place at birth
            esT = m["E_srce"]; etT = m["E_trgt"]; efT = m["E_face"]
            newv = torch.zeros(Nv2, dtype=torch.bool, device=dev); newv[Nv:Nv2] = True   # appended septum verts
            touch = newv[esT] | newv[etT]                        # half-edges incident to a fresh vertex
            ring = newv.clone(); ring[etT[touch]] = True; ring[esT[touch]] = True         # + their one-ring
            posf = lvl.state[:Nv2, px0:px1].detach().clone()
            xr = _relax_subset(posf, esT, etT, efT, nF2, m["A0"], m["P0"], m["V0f"], m["alive"],
                               float(m["R0"]), m["mech"], ring, self.local_relax)
            st2 = lvl.state.clone(); st2[:Nv2, px0:px1] = xr; lvl.state = st2
        # propagate the cell morphogen to daughters: each appended cell (new index nF+i) inherits its
        # mother's cell state so the RD pattern rides along through division (seg_A keeps the mother's).
        if self.cell_set is not None and daughter_mothers:
            clvl = H.level(self.cell_set)
            if clvl is not None and clvl.state.shape[0] >= nF2:
                cst = clvl.state.clone()
                for i, mother in enumerate(daughter_mothers):
                    cst[nF + i] = cst[mother]                    # daughter inherits all mother cell state (chem, ...)
                clvl.state = cst
                if getattr(clvl, "occ", None) is not None:
                    cocc = torch.zeros(clvl.state.shape[0], device=clvl.state.device); cocc[:nF2] = 1.0; clvl.occ = cocc
        return {}


@register_operator("topo_snapshot_3d", set="vertex", kind="structural", family="growth")
class TopoSnapshot3D(Structural):
    """Record the current mesh (flat half-edge table + vertex count) each frame, so a growing/dividing
    vesicle -- whose topology changes over time -- can be rendered frame by frame."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["recording", "topology_history", "diagnostic"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device); self.at = params.get("_at", "vertex")
        self.every = int(params.get("every", 1)); self._k = 0   # store at the RECORDING stride (else OOM on long runs)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                                            # store at tick 1 (~frame 0) then every `every` ticks,
        if not (self._k == 1 or self._k % self.every == 0):     # matching the engine's recorded set frames -> aligned
            return {}
        def cp(k):                                              # per-cell mechanical targets (for offline force/stress
            v = m.get(k)                                        # analysis) -- None-safe numpy copies
            return v.detach().cpu().numpy().copy() if v is not None and hasattr(v, "detach") else None
        m.setdefault("hist", []).append(dict(
            E_srce=m["E_srce"].detach().cpu().numpy().copy(),
            E_trgt=m["E_trgt"].detach().cpu().numpy().copy(),
            E_face=m["E_face"].detach().cpu().numpy().copy(),
            nF=int(m["nF"]), Nv=int(m["Nv"]),
            A0=cp("A0"), P0=cp("P0"), V0f=cp("V0f")))           # targets -> analyze_forces reconstructs the energy
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

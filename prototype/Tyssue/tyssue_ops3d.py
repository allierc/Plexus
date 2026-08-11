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

# Every frame whose relaxation ran WITHOUT the per-junction myosin because the array and the half-edge
# buffer had different lengths: (myosin length, half-edge count). Empty is the only correct value.
MYO_SKIPPED: list = []


def _carry_face_state(m, keep, dt, dev):
    """Reindex any EXTRA per-face array an operator has declared, through the same `keep` map.

    `keep` maps new face -> old face, and `divide_3d` / `apoptosis_3d` already use it to carry A0, P0,
    V0f, Vbirth, divjit, age, ndiv and alive across a rebuild. That list was a literal tuple, so a
    per-face state introduced by a NEW operator was silently left indexed against faces that had
    moved -- the same defect class as per-half-edge myosin before `junction_myosin_sync`, one level up
    and with no vertex-pair key to recover from.

    `m["face_carry"]` makes the list open. An operator declares its own array once and the topology
    operators still know nothing about what is in it, which is the whole point: the alternative is
    every topology operator learning the name of every state, and the next state added edits them all
    again.

    NOTE ON SEMANTICS. `keep` COPIES the parent's value onto both daughters, so what is carried this
    way must be an INTENSIVE quantity -- a density, a concentration, an age. Carrying an extensive
    one (an amount, a mass) doubles it at every division. `medioapical_myosin` stores an areal density
    for exactly this reason.
    """
    names = m.get("face_carry")
    if not names:
        return
    idx = torch.as_tensor(np.asarray(keep, np.int64), device=dev)
    for nm in sorted(names):
        a = m.get(nm)
        if a is None:
            continue
        a = a if torch.is_tensor(a) else torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        m[nm] = a.to(dev)[idx.clamp(max=max(a.shape[0] - 1, 0))].to(dt)


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
                       eocc, vocc, K_bend=0.0, twin_face=None, K_lumen=0.0, myo_e=None):
    """Explicit-arg AVM shape energy on a FIXED-size RESERVOIR (torch.compile-friendly: shapes never
    change, so it compiles once even under division). Dead slots are masked out: `alive` (faces),
    `eocc` (half-edges), `vocc` (vertices, for the radial term). R0 is a tensor (changes each frame);
    the K_* / Lam / Gam coefficients are compile-time constants."""
    area, perim, cen, vf = face_geometry_3d(pos, es, et, ef, nF, eocc)
    E = (K_A * (area - A0) ** 2 + K_P * (perim - P0) ** 2 + 0.5 * Gam * perim ** 2) * alive
    line = (pos[et] - pos[es]).norm(dim=-1) * eocc          # line tension over live half-edges only
    # PER-JUNCTION MYOSIN, when a junction operator has supplied it. `Lam` alone is one number for the
    # whole tissue, so no junction can be weaker than its neighbours and myosin cannot be recruited where
    # tension is high. `myo_e` is a per-half-edge multiplier on exactly that term -- which is where
    # actomyosin enters an AVM -- and defaults to None, in which case this reduces to `Lam * line.sum()`
    # exactly and every existing run is bit-identical.
    E = E.sum() + (Lam * line.sum() if myo_e is None else Lam * (myo_e * line).sum())
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


def _engine_owns_clock(params, default=1):
    """D1: the ENGINE owns the operator period; operators must not gate themselves as well.

    Several operators kept a private `every`/`_k` AND were gated by the engine, so the effective
    period was the PRODUCT. A config saying `every: 2` fired once every FOUR frames, and every
    per-call quantity (min_cycle, max_div, max_div_frac) silently meant 4x what it said. That is
    the defect that re-anchored the whole archive.

    Returns 1 always -- the operator never gates.

    `every > 1` still hard-errors, because a spec written under the OLD semantics means something
    different now: its author wrote `every: 2` intending period 2 and silently got 4, so honouring
    it as 2 would change the run without telling anyone. Such a spec must be read by a human.

    BUT the engine reads THE SAME `every` key (engine.py `max(1, int(o.params.get("every", 1)))`),
    so a blanket raise also forbids the correct, engine-owned multi-rate cadence -- which is the
    only way to express "this operator runs every k frames" at all. That made every archived spec
    carrying `every: 2` permanently unloadable, including the two that generate the Turing x vertex
    videos on the site's front page.

    So the raise is now an opt-OUT, not a wall: a spec that has been migrated declares
    `engine_clock: true` next to its `every`, which asserts "this period is written for the
    engine-owned clock; do not second-guess it". Unmigrated specs still fail loudly.
    """
    e = int(params.get("every", default))
    if e > 1 and not bool(params.get("engine_clock", False)):
        raise ValueError(
            f"D1: operator-side `every={e}` is no longer supported -- the engine owns the clock. "
            f"(A private period multiplied the engine's, so `every: 2` meant once every 4 frames "
            f"and every per-call quantity meant 4x what it said.) If this spec was written for "
            f"the OLD semantics, its true period was {e}x the engine's -- convert it and set "
            f"`every: {e * e}`. Then declare `engine_clock: true` beside it to confirm the period "
            f"is now engine-owned.")
    return 1



@register_operator("seed_mesh_3d", set="vertex", kind="seed", family="growth")
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

    def _grad(self, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc, twin_face=None,
              myo_e=None):
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            E = self._efn(p, es, et, ef, nF, A0, P0, V0f, alive, R0t, self.K_A, self.K_P,
                          self.K_V, self.K_R, self.Lambda, self.Gamma, eocc, vocc, self.K_bend,
                          twin_face, self.K_lumen, myo_e)
            g = torch.autograd.grad(E, p)[0]
        return torch.nan_to_num(g)

    def _grad_myo(self, m, *a, **kw):
        """`_grad` with the mesh's per-junction myosin, if a junction operator has supplied one.

        Read through the MESH rather than passed down the call chain, because the operator that writes it
        (`junction_ops.junction_myosin`) runs at a different point in the schedule and the two never see
        each other. Absent -> None -> the energy reduces to the scalar-Lambda form exactly.
        """
        myo = m.get("myo") if isinstance(m, dict) else None
        if myo is not None and myo.shape[0] != a[1].shape[0]:
            # A FRAME RELAXED WITH NO MYOSIN AT ALL, which is the operator being off rather than the
            # operator being approximated -- and it used to happen in silence. It should now be
            # unreachable: `junction_myosin_sync` re-keys the array after every topology operator, so a
            # mismatch here means the sync is missing from the schedule or is placed before the operator
            # that resized the buffer. Counted and announced once rather than raised, because raising
            # would take down every archived specification that predates the sync operator.
            MYO_SKIPPED.append((int(myo.shape[0]), int(a[1].shape[0])))
            if len(MYO_SKIPPED) == 1:
                print(f"[shape_energy_3d] myosin array is {myo.shape[0]} long against "
                      f"{a[1].shape[0]} half-edges -- relaxing WITHOUT myosin this frame. Schedule "
                      f"`junction_myosin_sync` after the topology operators.", flush=True)
            myo = None
        return self._grad(*a, myo_e=myo, **kw)

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
                step = -(self.eta * self.mu) * self._grad_myo(m, x, es, et, ef, nF, m["A0"], m["P0"],
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


# `vesicle_growth` (class VesicleGrowth) WAS HERE AND IS DELETED. Cedric, 8 August: "it is a
# bummer to have two growth competing operators... simplicity needs erasing here."
#
# It duplicated `grow_3d` and CONTRADICTED it: both wrote the same mesh targets, this one
# multiplicatively (V0f <- V0f * g^3), the other by assignment from its own snapshot
# (V0f <- V0f_init * s^3). Scheduled together -- which the discovery loop did twice, r001_07
# and r002_03 -- grow_3d ran second and overwrote this one every frame, silently.
#
# Every call site is ported: uniform body-wide inflation is `grow_3d` with `rho = 1` and the
# gate open (`a_sw = 0`). `max_scale` capped the LINEAR scale and `vth_frac` caps per-cell
# VOLUME, so the same plateau is `vth_frac = max_scale ** 3`.


@register_operator("divide_3d", set="vertex", kind="structural", family="growth")
class Divide3D(Structural):
    """In-surface cell division on the vesicle -- the sheet-division analog (tyssue
    sheet_topology.cell_division) lifted to the closed sphere. A cell divides when its wedge volume
    reaches `factor` x v_ref, THE SEED-TIME MEDIAN CELL VOLUME -- an absolute size checkpoint, the
    default since 8 August; see `_trigger` for why, and `model: doubler` for the birth-relative
    rule it replaced. It is split by an
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
        self.cycle_cv = float(params.get("cycle_cv", 0.0))       # STOCHASTIC CELL CYCLE: Gaussian CV of each daughter's
        #   cell-cycle length (fresh division threshold). >0 keeps division waves broken up (desynchronised) as the
        #   tissue proliferates -- essential at scale so max-rate division never outruns relaxation. 0 -> uniform reset_noise.
        self.p0 = float(params.get("p0", 3.72))
        self.every = _engine_owns_clock(params, default=3); self._k = 0
        # max_div / max_div_frac / vcap: WITHDRAWN, and not read anywhere. An
        # archived spec may still carry them; they are ignored, not honoured.
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
        # grow_3d then re-ramps activated daughters as their G1 regrowth. Off by default so other
        # presets (vesicle_divide/fig4) are unchanged; the tube preset turns it on.
        self.g1_ramp = bool(params.get("g1_ramp", False))
        # ORIENTED division at the red/white interface (Okuda's tube mechanism, user issue 3): the dividing
        # plane of an ACTIVATED (red) cell is oriented so the daughters stack ALONG the bud axis (the
        # direction from the vesicle centre to the activated tip) instead of by the cell's own long axis.
        # This adds new cells NORMAL to the body surface -> builds the tube WALL and EXTENDS the protrusion,
        # rather than widening it. orient_asw = activator threshold that flags an interface/red cell. 0 = off.
        self.orient_iface = bool(params.get("orient_iface", False))
        self.orient_asw = float(params.get("orient_asw", 1.0))

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        """Has this cell earned a division? THE ONLY THING A `model=` VARIANT OF divide_3d CHANGES.

        THE DEFAULT IS `sizer`: an ABSOLUTE threshold, v >= factor * v_ref, where v_ref is the
        seed-time median cell volume. Ginzberg, Kafri & Kirschner (Science 2015) are explicit that
        this is what size control requires -- "both the cell's target size and its actual size
        must be evaluated on ABSOLUTE rather than relative scales" -- and it is the mechanism they
        review as the G1/S size checkpoint, where small cells are held longer so they can catch up.
        A sizer corrects a size deviation in ONE generation.

        The rule this replaced as default, `v >= factor * v_birth`, is kept as `model: doubler`.
        It is relative, so it corrects nothing: a cell born 30% small divides 30% small and passes
        the deviation on undiminished. Combined with exponential growth it does worse than nothing,
        which is what this campaign measured -- vol_cv 0.160 at seed to 0.53 by frame 900.
        """
        return v_now >= self.factor * jit * v_ref

    def _fresh_djit(self, rng, n=1):
        """Fresh per-cell division-threshold multiplier. Gaussian CV (cycle_cv) when set -> desynchronised
        cell cycles; otherwise the legacy uniform +/-reset_noise jitter. Clamped so thresholds stay sane."""
        # `reset_noise` REMOVED 6 August. It was the legacy jitter, read only on the `cycle_cv == 0`
        # branch, and every parent this campaign has run sets cycle_cv > 0 -- so the battery
        # measured it DEAD (a same-seed edit moved the trajectory by exactly zero) and the search
        # could not reach it anyway. Two ways of doing one thing, one of them unreachable.
        if self.cycle_cv > 0:
            v = np.clip(1.0 + self.cycle_cv * rng.standard_normal(n), 0.4, 1.8)
        else:
            v = np.ones(n)                     # cycle_cv = 0 means synchronous, not "jittered a bit"
        return v if n > 1 else float(v[0])

    def forward(self, H, mask=None):
        from tyssue_topology_ops3d import rings_from_flat_3d, flat_from_rings_3d, divide_face_3d
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
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
        # HAS THIS CELL EVER DIVIDED? `age` alone cannot answer it: it starts at 0 for every SEEDED
        # cell and is only RESET to 0 by a division, so in the opening frames a whole untouched
        # tissue looks "just divided" -- the movie flashed entirely green in p1_ph_rd_only, a run in
        # which division never fires at all. Counting divisions separates born-this-way from
        # divided-just-now; the renderer needs both.
        ndiv = m.get("ndiv")
        ndiv = ([0] * nF) if (ndiv is None or ndiv.shape[0] != nF) else ndiv.detach().cpu().numpy().tolist()
        # volume-primary + bounded duration: divide if (2x volume AND old enough) OR (past max cycle length)
        v_ref = float(m.get("v_ref", 1.0))                       # SEED-TIME MEDIAN cell volume
        def _ready(f):
            if rings[f] is None or len(rings[f]) < 4 or alive[f] <= 0:
                return False
            vol_ok = self._trigger(vf[f], Vbirth[f], djit[f], age[f], v_ref)
            return (vol_ok and age[f] >= self.min_cycle) or (age[f] >= self.max_cycle)
        cand = [f for f in range(nF) if _ready(f)]
        rng.shuffle(cand)                                        # unbiased when more cells are ready than the cap
        # NO THROTTLE, NO BYPASS. Every cell that is READY divides, and READY is the test two
        # lines above: current volume >= factor x jitter x its OWN birth volume, after min_cycle
        # (or past max_cycle, which is a backstop and must be set long or it becomes the rate).
        # That is P3 -- "a cell divides because it got big" -- and it is the mechanism the paper
        # is about. The pace therefore comes from how fast grow_3d inflates cells:
        # ONE number, and it is a growth rate, in the place where the biology is.
        #
        # THREE NUMERICAL SHORTCUTS USED TO OVERRIDE IT, and each in turn WAS the rate:
        #   max_div         an absolute divisions-per-call floor. cap_div = max(max_div,
        #                   frac x N), and the floor won at every realistic cell count -- 30 per
        #                   call whatever the growth was doing.
        #   max_div_frac    the same as a fraction; entirely masked by the floor above it.
        #   vcap            force-divided any cell over 1.5 x the REFERENCE volume, bypassing the
        #                   x2 trigger, the jitter and min_cycle. v_ref drifts DOWN as cells
        #                   divide and shrink, so it fired earlier and earlier.
        #
        # Measured before removal: okuda_route (vcap on) divided 7x faster than its growth rate;
        # cellfix_B_new (vcap off) 1.2x. Two runs overran to 25,898 and 8,982 cells against
        # Okuda's 4,000 while the rate written in the spec was masked twentyfold. Every previous
        # run in this campaign measured a counter rather than a tissue.
        ndone = 0
        daughter_mothers = []                                    # mother face index of each appended daughter (order)
        bud_axis = None; a_cells = None                          # ORIENTED interface division: bud axis = centre->red tip
        if self.orient_iface and self.cell_set is not None:
            clvl0 = H.level(self.cell_set)
            if clvl0 is not None and "chem" in clvl0.state_schema:
                ci0, _ = clvl0.state_schema["chem"]
                a_cells = clvl0.state[:nF, ci0].detach().cpu().numpy()
                # RELATIVE TO THE FIELD, for the reason `rd_interface_tension.a_sw` is: an absolute
                # threshold on a field whose scale the chemistry sets is one edit from selecting
                # nothing. `orient_asw` defaulted to 1.0 with the same (0.2, 6.0) range, and only
                # 20 of 78 campaign runs ever reached act_max > 1.0 -- so in 74% of runs
                # `divide_3d:orient_iface`, a named Okuda mechanism, could orient nothing and was
                # behaviourally `hertwig`. A `set_impl ... orient_iface` edit was a silent no-op.
                amax = float(a_cells.max()) if a_cells.size else 0.0
                thr = self.orient_asw * amax
                rc = [np.array([pos[v] for v in rings[f]]).mean(0) for f in range(nF)
                      if amax > 0 and a_cells[f] > thr
                      and rings[f] is not None and len(rings[f]) >= 3]
                if len(rc):
                    ba = np.mean(rc, 0); nba = float(np.linalg.norm(ba))
                    if nba > 1e-6:
                        bud_axis = ba / nba
        blocked = 0                     # divisions the RESERVOIR refused, not the biology
        for f in cand:
            if len(pos) + 2 > buf:
                # THE VERTEX BUFFER IS FULL. Counted and reported, never silent.
                #
                # This `break` used to be the whole story: division simply stopped and nothing
                # said why. `wk_pressure_pos_s0` grew 150 -> 1778 cells by frame 323 of 900 and
                # then added ZERO for the remaining 575 frames, because 1778 is exactly the
                # (V+4)/2 cap of a buffer sized for a 150-cell start. Two thirds of that run
                # measured a full array, every specimen check passed it, and the only thing that
                # ever noticed was Cedric watching the green stop two seconds into the movie.
                #
                # A run truncated by its own reservoir is not evidence about growth, and the
                # difference between "the tissue stopped dividing" and "the tissue was not
                # ALLOWED to divide" is the difference between a result and an artefact.
                blocked = len(cand) - len(daughter_mothers)
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
            ndiv[f] = ndiv[f] + 1
            A0.append(a0d); V0f.append(v0d); Vbirth.append(half); alive.append(1.0)   # daughter B
            djit.append(self._fresh_djit(rng)); age.append(0); ndiv.append(ndiv[f])
            daughter_mothers.append(f)
            ndone += 1
        if ndone == 0:
            m["age"] = torch.as_tensor(np.asarray(age), dtype=dt, device=dev)   # persist ageing even without division
            m["ndiv"] = torch.as_tensor(np.asarray(ndiv), dtype=dt, device=dev)
        # RESERVOIR PRESSURE, reported every frame it bites. `div_blocked` is how many cells
        # wanted to divide this frame and were refused for want of room; `buf_full` says the
        # array is at its ceiling. Both travel with the mesh so run_one can record them.
        m["div_blocked"] = int(blocked)
        m["buf_full"] = bool(len(pos) + 2 > buf)
        if blocked:
            print(f"[divide_3d] RESERVOIR FULL: {blocked} division(s) refused for want of vertex "
                  f"buffer ({len(pos)}/{buf}). This run is capped by its array, not by its "
                  f"biology -- every later measurement describes the reservoir.", flush=True)
            return {}
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        A0a = np.array([A0[i] for i in keep], np.float64)
        V0fa = np.array([V0f[i] for i in keep], np.float64)
        Vba = np.array([Vbirth[i] for i in keep], np.float64)
        dja = np.array([djit[i] for i in keep], np.float64)
        agea = np.array([age[i] for i in keep], np.float64)
        ndva = np.array([ndiv[i] for i in keep], np.float64)
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
        m["ndiv"] = torch.as_tensor(ndva, dtype=dt, device=dev)
        m["alive"] = torch.as_tensor(alv, dtype=dt, device=dev)
        _carry_face_state(m, keep, dt, dev)
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



@register_operator("apoptosis_3d", set="vertex", kind="structural", family="death")
class Apoptosis3D(Structural):
    """Cell elimination on the closed vesicle -- the DIE family, and the inverse of divide_3d.

    Plexus2 lists eight elementary operator families and Die, "removal of biological entities", is
    one of them. This vesicle had no implementation of it: growth inflates, division subdivides,
    the purse-string measured inert and extrusion is the disqualified forcing term, so every
    mechanism it owned deformed the sheet OUTWARD. Invagination is one of Okuda's three target
    morphologies and nineteen rounds of search never produced one, because nothing in the
    vocabulary could pull inward.

    DECOMPOSED INTO PRIMITIVES, NOT A BEHAVIOUR. Monier et al. (2015) show apoptotic force driving
    fold formation in Drosophila, and the 2D `apoptosis` operator already renders that as three
    steps rather than a monolith. The same three exist here and two were already written:

        1. the dying cell's target volume SHRINKS each tick   (this operator)
        2. `reconnect_t1_3d` sheds its neighbours, one short edge at a time, until it is a triangle
        3. `face_collapse_3d` extrudes the triangle to a point and the sheet closes by force
           balance in `shape_energy_3d`

    So this operator does step 1 and asks for step 3; step 2 belongs to T1 and MUST be scheduled or
    a marked cell shrinks forever without ever reaching three sides. That is not a hidden coupling,
    it is the mechanism: a cell leaves an epithelium by losing neighbours.

    WHY IT WAITS FOR A TRIANGLE. Collapsing a k-sided face merges k vertices into one and leaves a
    vertex of degree k, which a trivalent sheet cannot represent -- a rosette. At k = 3 the count
    is V-2, E-3, F-1 and the Euler characteristic is unchanged. `face_collapse_3d` refuses anything
    else and validates closure before committing, so a refused collapse costs a frame, not a mesh.
    """
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["apoptosis", "cell_elimination", "extrusion", "delamination", "die"]
    REFERENCE = ("Monier, B. et al. (2015). Apico-basal forces exerted by apoptotic cells drive "
                 "epithelium folding. Nature 518:245-248; tyssue B-Apoptosis (DamCB).")
    PARAM_ROLES = {"cells": "explicit apoptotic cell indices", "mode": "how dying cells are chosen",
                   "shrink_rate": "target-volume shrink per tick",
                   "critical_frac": "extrude below this fraction of v_ref",
                   "frac": "(band/cone/chem_low) size of the dying population",
                   "a_sw": "(chem_low) die where the activator is below this fraction of its max",
                   "max_mark_frac": "cap on the fraction of the tissue under sentence at once"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex"); self.cat = params.get("cell_set", "cell")
        self.cells = [int(c) for c in (params.get("cells") or [])]
        # WHICH CELLS DIE is targeting, not a second mechanism: the shrink-shed-extrude pathway is
        # identical in every mode, so these are one operator rather than four.
        # DEFAULT `competition`, NOT `list`. The loop's `add_op` edit writes only the numeric
        # parameters an operator declares, so a non-numeric default is what every composition the
        # Proposer builds will actually run. `list` with no `cells` can never fire -- injecting it
        # would have handed the search an operator that is inert by construction, which is the
        # failure this project has now paid for three times. `competition` needs no chemistry (so
        # it is live on the b_none bases too), is the canonical cell-competition hypothesis, and
        # measured 69 deaths on r020_03 with protr 1.549 vs the parent's 1.596 and no premise
        # broken. Every existing spec sets `mode` explicitly, so nothing else moves.
        self.mode = str(params.get("mode", "competition"))
        #   list|band|cone|small|stalled|chem_low
        #   LOCAL (vs neighbours): competition|smaller|dimmer|older|crowded|lonely
        self.frac = float(params.get("frac", 0.04))
        self.cone_deg = float(params.get("cone_deg", 22.8))       # 10 cells across on a 2,000-cell ball
        self.band_deg = float(params.get("band_deg", 8.0))        # half-width of each ring
        self.n_bands = int(params.get("n_bands", 1))              # >1 -> that many latitude rings
        self.small_frac = float(params.get("small_frac", 0.35))   # (small) die below this x v_ref
        # THE THRESHOLD IS PER MODE, because the quantities have different dynamic ranges and one
        # number cannot serve them. Measured on r010_12 at a shared 0.5: `dimmer` removed ~212
        # cells by frame 480 and `competition`, `stalled`, `smaller` and `older` removed NONE.
        # The activator spans 0 to its maximum across a spot boundary, so a factor-of-two cut sees
        # it easily; volume, age and growth vary by roughly +-30% between neighbours, so demanding
        # HALF the neighbour mean asks for a cell that essentially cannot exist in a healthy sheet.
        # A silent rule is not a conservative rule -- it is an untested one, which this project has
        # now paid for three times (rd_interface_tension twice, chem_low once).
        _STALL_DEFAULT = {"dimmer": 0.5, "competition": 0.7, "older": 0.7,
                          "stalled": 0.8, "smaller": 0.85}
        self.stall_frac = float(params.get(
            "stall_frac", _STALL_DEFAULT.get(self.mode, 0.5)))
        #                                                          (stalled) die below this x the
        self.stall_margin = float(params.get("stall_margin", 0.02))  # population's MEDIAN growth,
        self.min_age = int(params.get("min_age", 4))              # once older than min_age, and
        self.n_max = int(params.get("n_max", 9))          # (crowded) die at this many neighbours
        self.n_min = int(params.get("n_min", 3))          # (lonely) die at this few
        #   only while the median cell has actually grown by stall_margin -- see _marked
        # WHICH SPECIES DECIDES WHO DIES: 0 by default; 2 reads the second RD system, which is
        # what lets one chemistry drive growth and another drive death in the same tissue.
        self.chan = int(params.get("chan", 0))
        self.a_sw = float(params.get("a_sw", 0.25))               # fraction of the activator's own max
        self.shrink = float(params.get("shrink_rate", 0.04))
        self.crit = float(params.get("critical_frac", 0.12))      # x v_ref, the seed-time median
        self.p0 = float(params.get("p0", 3.72))                   # target shape index, for P0
        # HOW FAST, as distinct from WHO -- see forward(). 0.5% of the tissue marked at once is
        # ~37 cells on the 7,400-cell parents this has to survive, against the 1,660 that `smaller`
        # took uncapped.
        self.max_mark_frac = float(params.get("max_mark_frac", 0.005))
        self.after_frame = int(params.get("after_frame", 0))
        self.every = _engine_owns_clock(params, default=1); self._k = 0
        self.seed = int(params.get("seed", 0))

    @staticmethod
    def _ring_neighbours(rings, f):
        """The faces sharing an EDGE with f, read off the rings before f is collapsed."""
        r = rings[f]
        if r is None:
            return []
        edges = {(int(r[i]), int(r[(i + 1) % len(r)])) for i in range(len(r))}
        edges |= {(b, a) for a, b in edges}
        out = set()
        for g, rg in enumerate(rings):
            if rg is None or g == f:
                continue
            for i in range(len(rg)):
                if (int(rg[i]), int(rg[(i + 1) % len(rg)])) in edges:
                    out.add(g); break
        return sorted(out)

    def _nb(self, m, nF, q):
        """(neighbour mean of `q`, neighbour count) on the CELL ADJACENCY GRAPH.

        Cedric, 9 August: "we could make a comparison graph network between adjacent cells to make
        a proper rule of cell competition?" -- and then: "this death operator could have different
        modes: game of life, too small vs others, too slow growth vs others, cell duration too long
        vs others". They are one mechanism: every useful death rule is a LOCAL comparison, and they
        differ only in which per-cell quantity is compared. So the graph is built once, here.

        WHY LOCAL AND NOT GLOBAL. `chem_low` compared each cell against the whole field and marked
        the tissue when the pattern weakened -- 2,000 cells shrank to 21.6% of their volume and not
        one was extruded. `stalled` compared growth against the population median and marked NOBODY
        on r010_12, because at rho = 0.1 every white cell grows at the SAME slow rate and none sits
        below half the median. Uniformly slow is not the same as losing, and only the neighbour
        graph can tell them apart. A local rule is also self-limiting: it cannot mark a uniform
        field however extreme that field is.

        Adjacency is the face across each half-edge -- the same relation rd_interface_tension uses
        for the red/white ring, so "adjacent" means one thing in this file.
        """
        import torch as _t
        es = _t.as_tensor(m["E_srce"], dtype=_t.long)
        et = _t.as_tensor(m["E_trgt"], dtype=_t.long)
        ef = _t.as_tensor(m["E_face"], dtype=_t.long)
        twin = ShapeEnergy3D._twin_faces(es, et, ef, int(m["Nv"])).cpu().numpy()
        efn = ef.cpu().numpy()
        ok = (efn < nF) & (twin < nF) & (twin >= 0)
        ssum = np.zeros(nF); cnt = np.zeros(nF)
        np.add.at(ssum, efn[ok], q[twin[ok]])
        np.add.at(cnt, efn[ok], 1.0)
        mean = np.zeros(nF)
        hit = cnt > 0
        mean[hit] = ssum[hit] / cnt[hit]
        return mean, cnt

    def _loser(self, m, nF, q, below=True):
        """Cells whose `q` is far from their neighbours' -- the shared body of every local mode."""
        ag = m.get("age")
        nb, cnt = self._nb(m, nF, q)
        old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
            else np.ones(nF, bool)
        live = (cnt > 0) & old_enough
        if below:
            # the neighbours must be doing the thing, or a quiet patch culls itself from inside
            return set(np.where(live & (nb > self.stall_margin)
                                & (q < self.stall_frac * nb))[0].tolist())
        return set(np.where(live & (q > nb / max(self.stall_frac, 1e-9)))[0].tolist())

    def _q(self, m, H, nF, what):
        """The per-cell quantity a local mode compares."""
        if what == "growth":                                  # fractional growth SINCE BIRTH
            v = m.get("V0f"); vb = m.get("Vbirth")
            if v is None or vb is None:
                return None
            return np.maximum(v.detach().cpu().numpy()[:nF]
                              / np.maximum(vb.detach().cpu().numpy()[:nF], 1e-12) - 1.0, 0.0)
        if what == "volume":
            v = m.get("V0f")
            return None if v is None else v.detach().cpu().numpy()[:nF]
        if what == "age":
            a = m.get("age")
            return None if a is None else a.detach().cpu().numpy()[:nF]
        if what == "act":
            clvl = H.level(self.cat)
            if clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
                return None
            h0, _ = clvl.state_schema["chem"]
            return clvl.state[:nF, h0 + self.chan].detach().cpu().numpy()
        return None

    def _admit(self, flag, want, m, H, nF):
        """Turn the mode's proposal into the marks actually TAKEN this tick.

        A DEATH RATE, NOT A DEATH SET. Every state-defined rule re-evaluates each frame, so on a
        growing tissue the marks ACCUMULATE: measured on the campaign's own best parents, `smaller`
        killed 1,660 of 7,424 cells and took the run from protr 1.513 / grip 0.228 to 1.131 /
        0.049, and every mode but `crowded` did something comparable. The thresholds were
        calibrated on a static 400-cell sheet where a population is marked ONCE; nothing in them
        bounds a flux, and a threshold on a state variable never can -- it names a set, and the set
        refills as fast as the tissue can produce members of it.

        `max_mark_frac` caps how much of the tissue may be under sentence AT ONCE, so the mode
        chooses WHO dies and this chooses HOW FAST. `crowded` is the existence proof that a gentle
        rate leaves a run intact: 104 deaths, no premise broken, grip 0.216 -> 0.116 where the
        others reached ~0.01.

        THE NAMED POPULATIONS ARE EXEMPT, because the cap bounds a flux and a set chosen once has
        none: `apopgeo_half` deliberately sentences 45% of the sheet in one act, and capping it
        would silently delete the hardest topology test this operator has.
        """
        if self.mode in ("list", "band", "cone"):
            for f in want:
                flag[f] = 1.0
            return flag
        held = np.where(flag > 0)[0]
        room = max(1, int(self.max_mark_frac * nF)) - len(held)
        if room <= 0:                       # the queue is full: let it drain before sentencing more
            return flag
        fresh = np.array(sorted(want - {int(x) for x in held}), dtype=np.int64)
        if len(fresh) > room:
            # WORST FIRST once the cap bites, so the rule still means what it says: a competition
            # rule that culls an arbitrary subset of its losers is a lottery wearing the rule's name.
            sev = self._severity(m, H, nF, fresh)
            fresh = fresh[np.argsort(sev)[:room]] if sev is not None else fresh[:room]
        flag[fresh] = 1.0
        return flag

    def _severity(self, m, H, nF, idx):
        """How badly each of `idx` qualifies -- SMALLEST DIES FIRST. Only consulted when the rate
        cap bites and the mode has named more cells than may die at once; a competition rule that
        then culls an arbitrary subset of its losers would be a lottery wearing the rule's name."""
        _Q = {"competition": "growth", "smaller": "volume", "dimmer": "act", "older": "age"}
        if self.mode in _Q:
            q = self._q(m, H, nF, _Q[self.mode])
            if q is None:
                return None
            nb = self._nb(m, nF, q)[0]
            r = q[idx] / np.maximum(nb[idx], 1e-12)
            return -r if self.mode == "older" else r        # `older` loses by being ABOVE
        if self.mode in ("small", "stalled"):
            q = self._q(m, H, nF, "volume" if self.mode == "small" else "growth")
            return None if q is None else q[idx]
        if self.mode == "chem_low":
            q = self._q(m, H, nF, "act")
            return None if q is None else q[idx]
        if self.mode == "crowded":
            return -self._nb(m, nF, np.ones(nF))[1][idx].astype(float)   # most crowded first
        if self.mode == "lonely":
            return self._nb(m, nF, np.ones(nF))[1][idx].astype(float)
        return None                                          # list/band/cone: a named population

    def _marked(self, m, H, nF):
        """The set of cell indices currently marked to die."""
        if self.mode == "list":
            return {c for c in self.cells if c < nF}
        cen = m.get("cen_np")
        if cen is None:                                            # per-cell centroid from the rings
            return set()
        r = np.linalg.norm(cen, axis=1) + 1e-12
        u = cen / r[:, None]
        if self.mode == "band":                                    # `n_bands` latitude rings
            lat = np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1)))   # -90 .. +90
            if self.n_bands <= 1:
                return set(np.where(np.abs(lat) < self.band_deg)[0].tolist())
            # evenly spaced in latitude, endpoints excluded so no ring sits on a pole
            centres = np.linspace(-90.0, 90.0, self.n_bands + 2)[1:-1]
            hit = np.zeros(len(lat), bool)
            for c in centres:
                hit |= np.abs(lat - c) < self.band_deg
            return set(np.where(hit)[0].tolist())
        if self.mode == "cone":                                     # one contiguous cap
            ax = np.array([0.0, 0.0, 1.0])
            return set(np.where(np.degrees(np.arccos(np.clip(u @ ax, -1, 1)))
                                < self.cone_deg)[0].tolist())
        if self.mode == "small":
            # DEATH BY SIZE, and the one selection that is genuinely about the cell's own state
            # rather than where it sits. A cell whose volume has fallen below `small_frac` of
            # v_ref -- the seed-time median -- is squeezed out, which is what an epithelium does
            # with a cell it can no longer accommodate. It re-evaluates for the same reason
            # `chem_low` does: a cell arrives in this set by shrinking, not by being pushed.
            v = m.get("V0f")
            if v is None:
                return set()
            vv = v.detach().cpu().numpy()[:nF]
            v_ref = float(m.get("v_ref", 1.0))
            return set(np.where(vv < self.small_frac * v_ref)[0].tolist())
        if self.mode == "stalled":
            # CELL COMPETITION: a cell that is not growing while its neighbours are gets removed.
            # Cedric, 9 August: "the apoptosis is working but not what triggers it -- would it be
            # possible to kill cells that do not grow?"
            #
            # RELATIVE, WHICH IS THE WHOLE POINT. `chem_low` marks every cell below a fraction of
            # the activator's maximum, so when the pattern weakens it marks the TISSUE: measured on
            # r019_02_apop_low, every cell shrank to 21.6% of its volume, act_max went to zero, and
            # not one cell was ever extruded, because death needs a cell squeezed to a triangle by
            # neighbours that are not shrinking too. A threshold against the population's own
            # median cannot do that: it always names a minority, by construction.
            #
            # V0f/Vbirth is growth since birth and `age` is time since birth. Both are already
            # carried across renumbering by `keep` -- in divide_3d and in this operator -- so this
            # needs no new bookkeeping, which is the part that has gone wrong twice already.
            v = m.get("V0f"); vb = m.get("Vbirth"); ag = m.get("age")
            if v is None or vb is None:
                return set()
            vv = v.detach().cpu().numpy()[:nF]
            vbb = np.maximum(vb.detach().cpu().numpy()[:nF], 1e-12)
            # THE EXCESS OVER BIRTH SIZE, NOT THE RATIO -- and the first version compared ratios,
            # which cannot work. V0f/Vbirth is >= 1 for any cell that has grown at all and exactly
            # 1.0 for one that has not, so with a median near 1.4 a cut at 0.5 x median = 0.7 sits
            # BELOW the floor and marks nobody. Measured: zero deaths across three runs and
            # hundreds of frames. `g = ratio - 1` is fractional growth SINCE BIRTH -- 0 for a
            # stalled cell -- and half the median excess is the comparison that was intended. On a
            # synthetic population of 400 stalled cells among 1,600 growing ones: the ratio form
            # marks 0, the excess form marks 618.
            g = np.maximum(vv / vbb - 1.0, 0.0)
            med = float(np.median(g))
            # A GLOBALLY STALLED TISSUE IS NOT A TISSUE FULL OF LOSERS. If the median cell has not
            # grown, there is no competition to lose and this marks nobody -- otherwise the same
            # runaway that chem_low produced would return by another route.
            if med < self.stall_margin:
                return set()
            old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
                else np.ones(nF, bool)
            return set(np.where((g < self.stall_frac * med) & old_enough)[0].tolist())
        # ---- THE LOCAL FAMILY. All five compare a cell with the cells TOUCHING it; they differ
        # only in what is compared, which is why they share `_loser`. Each is a different
        # biological hypothesis about why a cell is eliminated, not a different way to compute one.
        if self.mode == "competition":            # grows slower than its neighbours -- Myc-style
            q = self._q(m, H, nF, "growth")       # cell competition, the loser is out-proliferated
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "smaller":                # smaller than its neighbours: squeezed out by a
            q = self._q(m, H, nF, "volume")       # tissue that can no longer accommodate it
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "dimmer":                 # less activator than its neighbours -- the LOCAL
            q = self._q(m, H, nF, "act")          # form of chem_low, which failed by being global
            return set() if q is None else self._loser(m, nF, q, below=True)
        if self.mode == "older":                  # has gone longer without dividing than its
            q = self._q(m, H, nF, "age")          # neighbours: a cell that stopped cycling
            return set() if q is None else self._loser(m, nF, q, below=False)
        if self.mode in ("crowded", "lonely"):
            # GAME OF LIFE, on the real adjacency rather than a lattice. A closed trivalent sheet
            # gives every cell about six neighbours, so `crowded` fires where division has packed a
            # region and `lonely` only after deaths have already thinned one -- which makes
            # `lonely` a rule about how a wound SPREADS or heals, and worth having for that alone.
            deg = self._nb(m, nF, np.ones(nF))[1]
            ag = m.get("age")
            old_enough = (ag.detach().cpu().numpy()[:nF] >= self.min_age) if ag is not None \
                else np.ones(nF, bool)
            if self.mode == "crowded":
                return set(np.where(old_enough & (deg >= self.n_max))[0].tolist())
            return set(np.where(old_enough & (deg > 0) & (deg <= self.n_min))[0].tolist())
        if self.mode == "chem_low":                                 # die BETWEEN the spots
            clvl = H.level(self.cat)
            if clvl is None or "chem" not in getattr(clvl, "state_schema", {}):
                return set()
            h0, _h1 = clvl.state_schema["chem"]
            a = clvl.state[:nF, h0 + self.chan].detach().cpu().numpy()
            amax = float(np.nanmax(a)) if a.size else 0.0
            if amax <= 1e-9:
                return set()
            return set(np.where(a < self.a_sw * amax)[0].tolist())
        return set()

    def forward(self, H, mask=None):
        # THE ACTED LEDGER IS BLIND TO THIS OPERATOR, AND TO THE WHOLE DIE FAMILY. Measured over
        # rounds r001-r012 of the live campaign: 24 runs killed cells, 6,693 deaths in total, and
        # `inert_operators` recorded `apoptosis_3d` as having acted in ZERO of them -- including
        # runs that extruded 200 cells.
        #
        # The cause is structural, in both senses. `instrument._wrap` decides `acted` from
        # `_nonzero(out) or (before != after)`, where `before`/`after` fingerprint the LEVEL STATE.
        # A Die operator returns `{}` -- it has no delta to contribute -- and does its work by
        # mutating the mesh: `nF`, the half-edge table, `alive`, and the per-face arrays. None of
        # that is in the fingerprint, so the engine cannot see it and infers "nothing happened".
        #
        # HARMLESS TODAY AND NOT SAFE TO LEAVE. It never lands in `inert_operators`, so the
        # inert-implies-inconclusive rule in `round.score` cannot misfire on it. But the ledger is
        # the only cheap evidence that an operator RAN, and for this family it says nothing at all
        # -- so a death run and a run where death was silently unreachable are indistinguishable in
        # the record. That is the failure this project has paid for with `rd_interface_tension`
        # (written off inert twice, never having fired) and `shape_to_chem` (100% acted, changing
        # nothing).
        #
        # THE FIX IS FOR THE OPERATOR TO REPORT, NOT FOR THE ENGINE TO GUESS. Fingerprinting the
        # mesh would work and is the wrong shape: the engine would be inferring from a hash what
        # this operator already knows exactly. `m["n_apop"]` is incremented here on every
        # extrusion, so the operator can hand back a liveness record -- deaths this tick, cells
        # marked, cells waiting to be shed -- and the ledger can record a fact instead of a
        # deduction. That record is also what gate G13 needs: the frame a cell crosses the
        # extrusion threshold, separate from the frame it is shed, which no downstream sampling
        # stride can recover.
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        # LOCAL, as Divide3D does: tyssue_topology_ops3d imports from this module, so a
        # module-level import here is circular.
        from tyssue_topology_ops3d import (rings_from_flat_3d, flat_from_rings_3d,
                                           face_collapse_3d)
        self._k += 1
        dev = m["V0f"].device; dt = m["V0f"].dtype
        nF = int(m["nF"]); Nv = int(m["Nv"])
        px0, px1 = lvl.state_schema["pos"]
        pos_t = lvl.state[:Nv, px0:px1].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].cpu().numpy(); et = m["E_trgt"].cpu().numpy(); ef = m["E_face"].cpu().numpy()
        rings = [np.asarray(r, np.int64) for r in rings_from_flat_3d(es, et, ef, nF)]
        # per-cell centroid, for the geometric modes
        m["cen_np"] = np.array([pos_t[r].mean(0) if r is not None and len(r) else np.zeros(3)
                                for r in rings])
        # MARKED CELLS ARE A PER-FACE FLAG, NOT A LIST OF INDICES, and the first version was a
        # list. Every collapse renumbers the faces through `keep`, so `cells: [900]` pointed at a
        # DIFFERENT cell after each death: marking one cell killed seven in the first smoke run,
        # a cascade down the index space. The flag rides the same `keep` map as `alive` and `age`,
        # so a cell stays the cell it was.
        flag = m.get("apop_flag")
        if flag is None or len(flag) != nF:
            base = np.zeros(nF, np.float64)
            if flag is not None:                       # carried across a topology change already
                base[:min(len(flag), nF)] = flag[:min(len(flag), nF)]
            flag = base
        # AN INDEX IS NOT AN IDENTITY, A REGION IS -- so `list` marks ONCE and the state-defined
        # modes re-evaluate. `cells: [900]` names the cell at index 900 AT THE MOMENT IT IS READ,
        # and every collapse renumbers the faces through `keep`; re-reading it each frame therefore
        # hands the mark to a fresh victim after every death. Measured before this guard: marking
        # one cell killed three, at ticks 86, 164 and 236, each reporting exactly one marked cell.
        #
        # This is the seed-window argument of the Plexus2 paper in a second place. There, an
        # initial condition re-applied every timestep "converts it from an initial condition into a
        # moving boundary condition"; here a one-off selection re-applied every timestep converts a
        # death sentence into a death RATE. `seed` is a kind so the runtime can impose the window;
        # this operator is `structural`, so it imposes its own.
        _want = set()
        if self.mode in ("list", "band", "cone"):     # a POPULATION chosen once; see below
            if not m.get("apop_marked_once"):
                _want = {f for f in self._marked(m, H, nF) if f < nF}
                m["apop_marked_once"] = True
        else:
            # chem_low IS THE ONLY MODE THAT MAY RE-EVALUATE, and band/cone sat on this branch
            # until it was measured. A geometric region LOOKS like state and is not: as the equator
            # constricts, fresh cells rotate INTO the band and are marked, so a ring of 24 became a
            # mouth that ate the sphere -- 2,000 cells down to 497 by frame 270. A ring or a cap is
            # a POPULATION chosen at one moment, exactly like an index list. Only a chemical
            # threshold is a genuine ongoing condition: a cell whose activator has fallen should
            # now die, and no cell enters that set merely by being pushed.
            _want = {f for f in self._marked(m, H, nF) if f < nF}
        flag = self._admit(flag, _want, m, H, nF)
        m["apop_flag"] = flag
        marked = {int(f) for f in np.where(flag > 0)[0]}
        if not marked:
            return {}
        V0f = m["V0f"].detach().cpu().numpy().astype(np.float64)
        A0 = m["A0"].detach().cpu().numpy().astype(np.float64)
        v_ref = float(m.get("v_ref", 1.0))
        crit = self.crit * v_ref
        # 1. SHRINK. shape_energy_3d contracts the cell toward the smaller target; T1 then finds its
        #    edges short and sheds neighbours. Nothing is removed here.
        for f in marked:
            if f < nF:
                V0f[f] = max(V0f[f] * (1.0 - self.shrink), 1e-9)
                A0[f] = max(A0[f] * (1.0 - self.shrink) ** (2.0 / 3.0), 1e-9)
        m["V0f"] = torch.as_tensor(V0f, dtype=dt, device=dev)
        m["A0"] = torch.as_tensor(A0, dtype=dt, device=dev)
        # P0 FOLLOWS A0, as it does everywhere else in this file: the target perimeter of a cell
        # with target area A is p0*sqrt(A). Leaving it behind would make the shrinking cell chase a
        # perimeter its area no longer supports, which is a shape-index error, not a size one.
        m["P0"] = torch.as_tensor(self.p0 * np.sqrt(np.maximum(A0, 1e-9)), dtype=dt, device=dev)
        # 2. EXTRUDE the ones that are now triangles AND small enough.
        # A DYING CELL'S CONTENTS GO SOMEWHERE. Cedric, 9 August: "a rule should enforce that the
        # sum of activity in the dying cell's vicinity does not change much by construction."
        #
        # Until now they did not: the row was dropped from the cell state and its activator and
        # inhibitor left the tissue. That is a discontinuity in a conserved quantity, injected at
        # every death, and nothing in the model accounted for it -- a cell is extruded and its
        # chemistry is simply gone. Real extrusion hands the contents to the neighbours that close
        # over the gap.
        #
        # AMOUNT, NOT CONCENTRATION. The cell holds a * v; the neighbours receive that amount and
        # each converts it back to a concentration by its own volume. Distributing the
        # CONCENTRATION would create or destroy material whenever the neighbours are a different
        # size from the deceased, which is exactly the accounting error this fixes.
        clvl_c = H.level(self.cat)
        _has_chem = clvl_c is not None and "chem" in getattr(clvl_c, "state_schema", {})
        _bequest = []                       # (neighbour indices, amount per channel)
        gone = 0
        for f in sorted(marked, reverse=True):
            if f >= nF or rings[f] is None or len(rings[f]) != 3:
                continue
            if V0f[f] > crit:
                continue
            nbrs = [int(g) for g in self._ring_neighbours(rings, f) if g < nF]
            if face_collapse_3d(rings, pos_t, f):
                gone += 1
                if _has_chem and nbrs:
                    h0, h1 = clvl_c.state_schema["chem"]
                    amt = clvl_c.state[f, h0:h1].detach().clone() * float(V0f[f])
                    _bequest.append((nbrs, amt))
        if gone == 0:
            return {}
        if _bequest:
            cs = clvl_c.state.clone()
            h0, h1 = clvl_c.state_schema["chem"]
            for nbrs, amt in _bequest:
                # DO NOT POUR INTO A CELL THAT IS ITSELF VANISHING. The divisor is the recipient's
                # volume, and a cell that is MARKED but cannot reach a triangle shrinks to the
                # 1e-9 floor and stays there -- so a bequest of ~0.04 divided by 1e-9 injects ~1e7
                # concentration in one step, and a few of those compound. Measured before this
                # guard, on r020_00_ctrl + `smaller`: act_min -1.04e10.
                #
                # This was my own code, added on 9 August to enforce Cedric's rule that "the sum of
                # activity in the dying cell's vicinity should not change much by construction" --
                # and it turned a conservation law into an amplifier by dividing by a number the
                # model allows to reach zero.
                live = [g for g in nbrs if float(V0f[g]) >= crit]
                if not live:
                    # every neighbour is on its way out too: the material leaves with the cell,
                    # which is what happened before conservation existed and is the honest
                    # alternative to inventing somewhere to put it
                    continue
                # AND THE BEQUEST MUST LEAVE THE RECIPIENT INSIDE THE INTEGRATOR'S BASIN.
                # Conservation is correct as conservation and unbounded as CONCENTRATION: the
                # increment is share/V_g, and Gray-Scott integrates stably only while its
                # activator stays in the region its own kinetics bound. A bequest that pushes a
                # neighbour outside it diverges under explicit Euler, and the divergence is not
                # gentle -- measured on tsd_cap10 and tsd_cap25, act_mean_floor reached -7.29e11
                # and -2.42e23 before going NaN at frame 1430, while tsd_cap10_fast with FIVE
                # TIMES the deaths stayed clean at 0.0465. The amount of death is not the
                # variable; the two failing runs are the SLOW ones (shrink 0.05 against 0.15), so
                # cells linger near the extrusion threshold and their neighbours receive many
                # small bequests instead of few.
                #
                # `ceil` is the field's own running maximum, which is the only scale available
                # that the chemistry itself sets -- the same argument that makes a_sw a fraction
                # rather than an absolute. Material that cannot be placed without leaving the
                # basin is DROPPED AND COUNTED, never silently injected: a conservation law that
                # is quietly violated is worse than one that reports where it failed, because the
                # first is indistinguishable from a working one.
                share = amt / float(len(live))
                # TORCH, NOT NUMPY: `cs` is the cell state and lives on the GPU. `np.nanmax` on a
                # CUDA tensor raises "can't convert cuda:0 device type tensor to numpy", which
                # killed the two runs this bound exists to fix -- so the fix's first outing failed
                # for a reason unrelated to the fix. Hoisted out of the per-neighbour loop too: it
                # is a property of the field, not of the recipient.
                # ELEMENTWISE, AND PER COLUMN. The bequest spans every chemistry channel the cell
                # carries, so `inc` and `room` are vectors: a scalar comparison on them raises
                # "Boolean value of Tensor with more than one value is ambiguous", and a single
                # ceiling taken from channel 0 would clamp the substrate against the activator's
                # maximum. Each column is bounded by its own.
                ceil = torch.nan_to_num(cs[:nF, h0:h1], nan=0.0).amax(dim=0) if nF else None
                for g in live:
                    inc = share / float(V0f[g])
                    if ceil is None:
                        cs[g, h0:h1] += inc
                        continue
                    room = (ceil - cs[g, h0:h1]).clamp(min=0.0)
                    fits = torch.minimum(inc, room)
                    over = (inc - fits).sum()
                    if float(over) > 0:
                        m["apop_spill"] = m.get("apop_spill", 0.0) + float(over) * float(V0f[g])
                    cs[g, h0:h1] += fits
            clvl_c.state = cs
        # 3. REBUILD, exactly as divide_3d does: `keep` maps new face -> old face and carries every
        #    per-face array across, so nothing can fall out of step with the mesh.
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        def _car(name, arr=None):
            a = (arr if arr is not None else m[name].detach().cpu().numpy().astype(np.float64))
            return torch.as_tensor(np.asarray([a[i] for i in keep], np.float64), dtype=dt, device=dev)
        m["A0"] = _car("A0", A0); m["V0f"] = _car("V0f", V0f)
        m["P0"] = torch.as_tensor(self.p0 * np.sqrt(np.maximum(
            m["A0"].detach().cpu().numpy(), 1e-9)), dtype=dt, device=dev)
        for nm in ("Vbirth", "divjit", "age", "ndiv", "alive"):
            if nm in m:
                m[nm] = _car(nm)
        _carry_face_state(m, keep, dt, dev)
        m["apop_flag"] = np.asarray([flag[i] for i in keep], np.float64)   # identity, across renumbering
        st = lvl.state.clone()
        st[:len(pos_t), px0:px1] = torch.as_tensor(pos_t, dtype=dt, device=dev)
        lvl.state = st
        m["E_srce"] = torch.as_tensor(es2, device=dev); m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev); m["nF"] = nF2
        # THE CELL SET FOLLOWS THE MESH. `chem` is indexed by face, so a death that compacts the
        # faces and not the chemistry silently re-assigns every activator value to a different
        # cell -- the same class of error as a per-face target left behind by division.
        clvl = H.level(self.cat)
        if clvl is not None and clvl.state.shape[0] >= nF:
            # MEASURED INNOCENT, and worth recording because it was the obvious suspect. At every
            # death min(a) BEFORE this copy equals min(a) AFTER it -- tick 34 read -0.03872 on
            # both sides. The negative activator that breaks P12 on every run where cells die
            # arrives from the reaction/diffusion step operating on a mesh a death has just
            # perturbed, not from this reindex. Ruled out by experiment as well: removing
            # shape_to_chem left it at -0.0948, and quartering chi left it at -0.0689.
            cst = clvl.state.clone()
            cst[:nF2] = clvl.state[torch.as_tensor(keep, device=clvl.state.device)]
            clvl.state = cst
            if getattr(clvl, "occ", None) is not None:
                cocc = torch.zeros(clvl.state.shape[0], device=clvl.state.device)
                cocc[:nF2] = 1.0; clvl.occ = cocc
        # THE PENDING DELTAS MUST BE RENUMBERED TOO, AND THIS IS THE BUG THAT BROKE P12 ON EVERY
        # RUN WHERE A CELL DIED.
        #
        # The engine zeroes the delta accumulator once per TICK and integrates at the END of the
        # schedule, so `cell_diffuse` and `cell_react` deposit their per-cell deltas early and the
        # engine applies them last. An operator that RENUMBERS the set in between leaves every one
        # of those deltas pointing at a different cell -- a large negative flux meant for one cell
        # lands on another that has almost no activator, and the concentration goes negative. That
        # is exactly what was measured: the activator hit -0.1529 at frame 50 on every mode that
        # killed anything, decaying afterwards as the scrambled deltas diffused away, while the
        # no-death control never left 0.0000.
        #
        # `divide_3d` never hit it because appending is not renumbering: daughters go to indices
        # >= nF and every existing cell keeps its own. Removal is the first operation in this
        # engine that moves a row.
        #
        # Permuting the accumulator with the SAME `keep` map fixes it wherever the operator sits
        # in the schedule. Confirmed independently by moving apoptosis ahead of the chemistry --
        # act_min held at 0.0000 for the whole run and P12 and P4 both cleared -- but that is a
        # discipline every spec would have to remember, and this is the guarantee instead.
        try:
            _d = getattr(H, "_delta", None)
            if isinstance(_d, dict) and self.cat in _d and _d[self.cat] is not None:
                _kt = torch.as_tensor(keep, device=_d[self.cat].device, dtype=torch.long)
                _d[self.cat][:nF2] = _d[self.cat][_kt]
                _d[self.cat][nF2:] = 0.0
            _db = getattr(H, "_delta_blocks", None)
            if isinstance(_db, dict):
                for _k, _v in _db.items():
                    if isinstance(_k, tuple) and _k and _k[0] == self.cat and _v is not None:
                        _kt = torch.as_tensor(keep, device=_v.device, dtype=torch.long)
                        _v[:nF2] = _v[_kt]; _v[nF2:] = 0.0
        except Exception as _e:
            print(f"[apoptosis_3d] could not renumber the pending deltas "
                  f"({type(_e).__name__}) -- chemistry may be scrambled this tick", flush=True)
        m["n_apop"] = int(m.get("n_apop", 0)) + gone
        # LOUD, because a death is not recoverable and the count must be auditable against the
        # cell count: nF -> nF2 should differ by exactly `gone` and by nothing else.
        print(f"[apoptosis_3d] tick {self._k}: extruded {gone} cell(s), {nF} -> {nF2} faces "
              f"(marked {len(marked)}, total {m['n_apop']})", flush=True)
        return {}


@register_operator("divide_3d", model="doubler", set="vertex", kind="structural", family="growth")
class Divide3DDoubler(Divide3D):
    """Divide at `factor` x THIS CELL'S OWN BIRTH VOLUME -- the rule that was the default until
    8 August, kept because it is the null the sizer has to beat and because every result in this
    project's record up to round 4 was measured under it.

    Under exponential growth this is a TIMER wearing a sizer's clothes: doubling from any birth
    volume takes the same time, so it never consults size in any way that could correct one. The
    review is direct about the consequence: size disparities are amplified, not constrained.
    """
    MECHANISM_TAGS = ["division", "volume_doubling", "relative_threshold", "no_size_control"]

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return v_now >= self.factor * jit * v_birth


@register_operator("divide_3d", model="timer", set="vertex", kind="structural", family="growth")
class Divide3DTimer(Divide3D):
    """Divide on the CLOCK: `age >= cycle * jit` division-calls since birth, size ignored entirely.

    Alone this is the worst of the three -- a cell divides whether or not it has grown, so size
    variance is set by whatever growth did in the interval and nothing corrects it. It exists to
    be paired with `grow_3d model: timer`, which sets the growth rate from the size deficit so the
    cell arrives at its target size exactly when the clock fires. That pair gives every cell the
    same age AND the same size at division, the strongest homeostasis available here, and it is
    the upper bound the sizer and the balance model should be read against.

    `cycle` is in DIVISION-CALLS, the same unit as `min_cycle` and `max_cycle`, not frames.
    """
    MECHANISM_TAGS = ["division", "timer", "cell_cycle_clock", "size_independent"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.cycle = float(params.get("cycle", 8.0))

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return age >= self.cycle * jit


@register_operator("topo_snapshot_3d", set="vertex", kind="structural", family="growth")
class TopoSnapshot3D(Structural):
    """Record the current mesh (flat half-edge table + vertex count) each frame, so a growing/dividing
    vesicle -- whose topology changes over time -- can be rendered frame by frame."""
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = False
    MECHANISM_TAGS = ["recording", "topology_history", "diagnostic"]
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device); self.at = params.get("_at", "vertex")
        self.every = _engine_owns_clock(params); self._k = 0   # engine owns the stride (D1)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        def cp(k):                                              # per-cell mechanical targets (for offline force/stress
            v = m.get(k)                                        # analysis) -- None-safe numpy copies
            return v.detach().cpu().numpy().copy() if v is not None and hasattr(v, "detach") else None
        m.setdefault("hist", []).append(dict(
            E_srce=m["E_srce"].detach().cpu().numpy().copy(),
            E_trgt=m["E_trgt"].detach().cpu().numpy().copy(),
            E_face=m["E_face"].detach().cpu().numpy().copy(),
            nF=int(m["nF"]), Nv=int(m["Nv"]),
            A0=cp("A0"), P0=cp("P0"), V0f=cp("V0f"),            # targets -> analyze_forces reconstructs the energy
            # `age` = division-calls since this cell was born (divide_3d resets it to 0 on
            # division). Recorded so a renderer can colour RECENTLY DIVIDED cells. The first
            # attempt inferred "just divided" from cell AREA -- a sliver test -- and it never
            # fired: a division splits a cell into two roughly equal halves, so a normal daughter
            # is ~50-70% of its neighbours while the sliver test looks below 15%. It detects
            # DEGENERATE cells, not new ones. Age is the actual event, not a proxy for it.
            age=cp("age"), ndiv=cp("ndiv"),
            # PER-JUNCTION MYOSIN, when a junction operator has written one. Recorded for the same
            # reason `age`/`ndiv` are: a renderer cannot colour by a quantity that only existed inside
            # one frame's forward pass. `cp` is None-safe, so a run without the operator records None.
            myo=cp("myo"),
            # PER-CELL GROWTH INHIBITION, when a second morphogen is switching growth off. Recorded
            # for exactly the reason `age` and `myo` are: a renderer cannot colour by a quantity
            # that only existed inside one frame's forward pass, and "where is growth stopped" is
            # the whole point of an inhibitor -- an invisible mechanism is one nobody can check.
            # None-safe, so a run without an inhibitor records None and the renderer draws nothing.
            inhib=cp("inhib_frac"),
            # THE CONSERVATION LAW'S OWN ERROR TERM. Material a dying cell could not bequeath
            # without pushing a neighbour out of the integrator's basin is dropped and counted
            # here rather than injected. It must be ~0 on a healthy run; a large value says the
            # bequest is being refused, which is the diagnostic for the -7.3e11 divergence.
            apop_spill=float(m.get("apop_spill", 0.0)),
            # THE TWO-POOL STATE, when `medioapical_myosin` is in the schedule. `myo_med` is the
            # AREAL density on each cell and `myo_amount` the AMOUNT on each half-edge; the pair is
            # what makes the conservation ledger measurable offline, since `myo` alone is normalised
            # to a tissue mean of 1 and so cannot say how much myosin there is.
            myo_med=cp("myo_med"), myo_amount=cp("myo_amount"),
            # THE RESERVOIR, PER FRAME. divide_3d sets these on the mesh and nothing carried them
            # into the history, so run_one read them and always found nothing -- a run that
            # plateaued at 98.5% of its array reported buf_full False. The flag existed, the
            # counter existed, and the one structure anybody reads afterwards did not have them.
            div_blocked=int(m.get("div_blocked") or 0),
            # CUMULATIVE DEATHS, for the same reason div_blocked is here: apoptosis_3d counts
            # every extrusion on the mesh and nothing carried it into the history, so the only
            # visible trace of a death was the cell COUNT -- which cannot distinguish "nothing
            # died" from "deaths were masked by divisions". Measured on r019_02_apop_small: cells
            # went 2000 -> 3089 with death running, and on r019_02_apop_low they held at 2000 with
            # death running and nothing dying. Both read identically from the count alone.
            n_apop=int(m.get("n_apop") or 0),
            # WHICH CELLS ARE MARKED TO DIE, recorded for the same reason `age` is: a renderer
            # cannot colour a state that only existed inside one frame's forward pass. Without it
            # a dying cell is drawn exactly like a living one, the sheet closes over the gap, and
            # the mechanism is invisible in the movie -- which is what happened: a 293-cell patch
            # dying at the north pole could not be seen even looking straight down at it.
            apop=(np.asarray(m["apop_flag"]).copy()
                  if isinstance(m.get("apop_flag"), np.ndarray) else None),
            buf_full=bool(m.get("buf_full"))))
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

"""The 3D vertex model: an epithelium as a closed surface of polygonal cells, where the degrees
of freedom are the VERTICES and everything a cell does is a consequence of where they sit.

The state is a closed half-edge mesh. A cell is a FACE, a junction an EDGE, and the mechanics an
energy over faces and edges whose gradient moves the vertices. Because the representation is a
mesh rather than a lattice, the topology can change -- a cell divides by inserting a septum, dies
by contracting to nothing, and neighbours exchange by flipping an edge -- and every one of those
is an operator here rather than a special case in a solver.

In the order they appear below:

    seed_mesh        seed        the closed half-edge surface, built once at frame 0
    cell_mechanics   lateral     the shape energy, and the force that is its gradient
    cell_divide      structural  a septum through a face: one cell becomes two
    cell_die         structural  contract a cell to a point and remove it from the surface
    topo_record      structural  one recorded frame of topology per tick, so it can be rendered
    edge_flip        structural  the T1: a reversible neighbour exchange

then the models -- different hypotheses in one slot -- and the implementations, which change only
the arithmetic:

    seed_mesh[apicobasal]     the same epithelium seeded WITH its thickness
    cell_mechanics            models: monolayer, apicobasal, marinari
                              implementations: warp (default), autograd, compile
    cell_divide               models: doubler, timer
    cell_die                  models: competition, smaller, dimmer, older, crowded, lonely,
                              small, stalled, chem_low, prescribed, field

Reference: Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y. & Adachi, T. (2013). Reversible network
reconnection model for simulating large deformation in dynamic tissue morphogenesis. Biomech.
Model. Mechanobiol. 12(4):627-644 -- the reconnection `edge_flip` implements; Okuda, S., Miura,
T., Inoue, Y., Adachi, T. & Eiraku, M. (2018). Combining Turing and 3D vertex models reproduces
autonomous multicellular morphogenesis of the tissue. Sci. Rep. 8:2386; the ancestor is Honda, H.,
Tanemura, M. & Nagai, T. (2004). A three-dimensional vertex dynamics cell model of space-filling
polyhedra simulating cell division. J. Theor. Biol. 226(4):439-453. The shape energy is
Farhadifar, R., Roper, J.-C., Aigouy, B., Eaton, S. & Julicher, F. (2007). The influence of cell
mechanics, cell-cell interactions, and proliferation on epithelial packing. Curr. Biol.
17:2095-2104. The mesh representation follows Tyssue.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import SphericalVoronoi
from plexus.models.base import Lateral, Structural
from plexus.models.mesh import MeshTable, declare_vertex_carry
from plexus.models.registry import register_operator
from plexus.models.base import Rewire
from plexus.models.topology import (rings_from_flat_3d, flat_from_rings_3d,
                                    _edge_face_map, _check_closed)
from plexus.models.base import Lateral
# (was `from mesh_ops import face_geometry_3d`) -- same module now


MYO_SKIPPED: list = []


def _carry_face_state(m, keep, dt, dev):
    """Reindex any EXTRA per-face array an operator has declared, through the same `keep` map.

    `keep` maps new face -> old face, and `cell_divide` / `cell_die` already use it to carry A0, P0,
    V0f, Vbirth, divjit, age, ndiv and alive across a rebuild. That list was a literal tuple, so a
    per-face state introduced by a NEW operator was silently left indexed against faces that had
    moved -- the same defect class as per-half-edge myosin before `junction_sync`, one level up
    and with no vertex-pair key to recover from.

    `m["face_carry"]` makes the list open. An operator declares its own array once and the topology
    operators still know nothing about what is in it, which is the whole point: the alternative is
    every topology operator learning the name of every state, and the next state added edits them all
    again.

    NOTE ON SEMANTICS. `keep` COPIES the parent's value onto both daughters, so what is carried this
    way must be an INTENSIVE quantity -- a density, a concentration, an age. Carrying an extensive
    one (an amount, a mass) doubles it at every division. `medioapical_myosin` stores an areal density
    for exactly this reason.

    ONE IMPLEMENTATION, FOUR CALLERS. The body of this moved to `MeshTable.reindex_faces` so the
    carry is a property of the mesh rather than a helper each topology operator has to remember to
    call -- `edge_flip` did not, for its whole life, and dropped the medioapical myosin every time a
    flip lost a face. The semantics here are unchanged, clamp included.
    """
    if not m.get("face_carry"):
        return
    if hasattr(m, "reindex_faces"):
        m.reindex_faces(keep, dt=dt, dev=dev)
        return
    # A BARE DICT still reaches this: four operator self-tests build fake meshes as plain dicts and
    # must keep working, and every archived run predates the table.
    idx = torch.as_tensor(np.asarray(keep, np.int64), device=dev)
    for nm in sorted(m.get("face_carry") or ()):
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


def build_hexagon_mesh(n=1, r=1.0, jitter=0.0, seed=0):
    """ONE regular hexagon as an open half-edge mesh: a single face, six vertices, CCW from +z.

    THE ONLY SOLID IN THIS PROMOTION WHOSE SHAPE INDEX IS KNOWN ON PAPER, and that is its whole
    reason for existing. A regular hexagonal prism of side a = 1 and height h = 1 has

        A_cap = (3 sqrt3 / 2) a^2 = 2.598076      A_lat = 6 a h = 6
        S     = 2 A_cap + A_lat = 3 sqrt3 + 6 = 11.196152
        V     = A_cap h = 3 sqrt3 / 2 = 2.598076        V^(2/3) = 1.889882
        s     = S / V^(2/3) = 5.924261377933605

    and that ONE number is only reachable if the fan triangulation, the cap centroid convention,
    both cap orientations and the two-triangle wall split are simultaneously right. It is gate row
    AB-C2, and `build_disc_mesh` cannot produce it: its lattice lays equal x- and y-pitch with
    alternate rows offset by half a step, so its Voronoi cells are hexagon-ISH and never regular.

    A VALUE ON `shape:`, NOT A MODEL, by the vocabulary's own written test -- "a flat patch and a
    closed shell are the same hypothesis about the tissue seeded into two different geometries". A
    single regular cell is the third such geometry.

    THE CIRCUMRADIUS IS THE SIDE, for a hexagon and only for a hexagon, so `radius: 1.0` gives side
    1. `n` and `jitter` are accepted and ignored: the builders share one call signature, and a
    jittered hexagon is not a hexagon.
    """
    th = np.arange(6) * (np.pi / 3.0)
    verts = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(6)], 1).astype(np.float64)
    es = np.arange(6, dtype=np.int64)
    et = (np.arange(6, dtype=np.int64) + 1) % 6
    return verts, es, et, np.zeros(6, np.int64), 1


def build_disc_mesh(n, r=1.0, jitter=0.0, seed=0):
    """A FLAT Voronoi patch as an open half-edge mesh: vertices [Nv,3] (z=0), E_srce/E_trgt/E_face, nF.

    The planar twin of `build_sphere_mesh`, and it exists for one reason: a sheet is where the
    monolayer's apical/basal offset can be judged with no curvature in the picture. On a closed
    shell a normal offset that is going unstable and a shell that is merely bumpy look alike; on a
    flat patch the mid-surface should stay in its plane, and any departure IS the instability.

    OPEN, AND DELIBERATELY SO. Only the Voronoi regions that are BOUNDED and lie wholly inside the
    disc are kept, so the patch has a rim: rim edges belong to one face, not two. Euler is not 2 and
    `_check_closed` will say so -- which is correct, this is not a closed surface. The operators that
    care refuse a rim face rather than crashing (`divide_face_3d` returns None when a split edge has
    no neighbour), so the interior grows and the rim does not.

    CCW AS SEEN FROM +z, so the Newell area vector points along +z and every wedge volume taken about
    an origin BELOW the patch is positive -- which is what makes `cell_grow` and `cell_divide` work
    on a sheet without changing them: at height h the wedge volume is (h/3)A, exactly proportional to
    the cell's area.
    """
    from scipy.spatial import Voronoi
    g = np.random.default_rng(seed)
    # a jittered hex-ish lattice covering the disc, which gives rounder cells than pure Poisson
    m = int(np.ceil(np.sqrt(n / 0.7854)))                    # disc packs pi/4 of its bounding square
    step = 2.0 * r / max(m, 1)
    xs, ys = np.meshgrid(np.arange(m + 1) * step - r, np.arange(m + 1) * step - r)
    xs = xs + (np.arange(m + 1)[:, None] % 2) * 0.5 * step   # offset alternate rows -> hexagonal
    pts = np.stack([xs.ravel(), ys.ravel()], 1)
    if jitter > 0:
        pts = pts + jitter * step * (g.random(pts.shape) - 0.5)
    pts = pts[np.linalg.norm(pts, axis=1) <= r * 1.35]       # a collar OUTSIDE the disc, so the
    #                                                          cells we keep have bounded regions
    vor = Voronoi(pts)
    keep_r = r                                               # cells whose vertices all sit inside
    faces, V = [], vor.vertices
    for ip, ir in enumerate(vor.point_region):
        reg = vor.regions[ir]
        if not reg or -1 in reg:                             # unbounded region: outside the collar
            continue
        P = V[reg]
        if np.linalg.norm(P, axis=1).max() > keep_r:
            continue
        faces.append(np.asarray(reg, np.int64))
    if not faces:
        raise ValueError(f"build_disc_mesh: no bounded cell inside r={r} for n={n}")
    used = np.unique(np.concatenate(faces))                  # drop the collar's vertices
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    verts = np.zeros((len(used), 3), np.float64); verts[:, :2] = V[used]
    es, et, ef = [], [], []
    for f, reg in enumerate(faces):
        rr = remap[reg]
        P = verts[rr, :2]
        c = P.mean(0)
        if np.cross(P[0] - c, P[1] - c) < 0:                 # CCW seen from +z
            rr = rr[::-1]
        k = len(rr)
        for i in range(k):
            es.append(int(rr[i])); et.append(int(rr[(i + 1) % k])); ef.append(f)
    return (verts, np.array(es, np.int64), np.array(et, np.int64),
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
                       eocc, vocc, K_bend=0.0, twin_face=None, K_lumen=0.0, myo_e=None, Gam_l=0.0):
    """Explicit-arg vertex-model shape energy on a FIXED-size RESERVOIR (torch.compile-friendly: shapes never
    change, so it compiles once even under division). Dead slots are masked out: `alive` (faces),
    `eocc` (half-edges), `vocc` (vertices, for the radial term). R0 is a tensor (changes each frame);
    the K_* / Lam / Gam coefficients are compile-time constants."""
    area, perim, cen, vf = face_geometry_3d(pos, es, et, ef, nF, eocc)
    E = (K_A * (area - A0) ** 2 + K_P * (perim - P0) ** 2 + 0.5 * Gam * perim ** 2) * alive
    line = (pos[et] - pos[es]).norm(dim=-1) * eocc          # line tension over live half-edges only
    # PER-JUNCTION MYOSIN, when a junction operator has supplied it. `Lam` alone is one number for the
    # whole tissue, so no junction can be weaker than its neighbours and myosin cannot be recruited where
    # tension is high. `myo_e` is a per-half-edge multiplier on exactly that term -- which is where
    # actomyosin enters a vertex model -- and defaults to None, in which case this reduces to `Lam * line.sum()`
    # exactly and every existing run is bit-identical.
    E = E.sum() + (Lam * line.sum() if myo_e is None else Lam * (myo_e * line).sum())
    # A PER-JUNCTION QUADRATIC, WHICH IS NOT THE SAME TERM AS `Gam`. `Gam` above is Farhadifar's
    # cortical contractility, (Gamma/2) * PERIMETER^2, one number per CELL. `Gam_l` is Marinari's
    # (Nature 484:542, Supplementary p.1), (Gamma/2) * l^2 per JUNCTION -- they state the substitution
    # explicitly, "proportional to the square of the junction length, instead of the square of
    # perimeter length, and thus we model individual junctions as elastic springs with equilibrium
    # length l0 = -Lambda/Gamma". The two are rival hypotheses about one force, not additive
    # contributions, which is why `cell_mechanics[model: marinari]` sets one and zeroes the other
    # rather than the spec being free to set both.
    #
    # THE SUM IS OVER HALF-EDGES AND THEIR SUM IS OVER JUNCTIONS. A closed half-edge mesh carries two
    # half-edges per junction, so `line.sum()` is 2x a junction sum. The factor is NOT applied here --
    # `Lam` has always meant "per half-edge" and changing that would move every existing run -- it is
    # applied by the marinari variant when it maps the paper's parameters in. See ShapeEnergy3DMarinari.
    if Gam_l:
        E = E + 0.5 * Gam_l * ((line ** 2) * eocc).sum()
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
        if mech.get("plane_axis") is not None:                    # 2D vertex model: see ShapeEnergy3D
            step[:, int(mech["plane_axis"])] = 0.0
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



# CANONICAL NAME `seed_mesh`, ALIAS `mesh_seed`, and the alias is not politeness. Archived
# specifications use both spellings in comparable numbers, so registering only one would make half
# the corpus fail to load -- loudly, with `operator not in registry`, but fail. The canonical form
# follows the `seed_<noun>` convention core already uses for `seed_from_segmentation`, while the
# okuda classes kept the old spelling, so 325 specs, including every `r0*` the campaign has ever
# written, could not be run at all. `seed_cell_chem` (320 specs) had the same break.
@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed")
class SeedMesh3D(Structural):
    """Build the surface the whole model lives on, once, at frame 0: a closed half-edge mesh whose
    faces are the cells.

    vertex -> vertex: writes every vertex position, and stashes the half-edge table and the
    per-face targets on the mesh.

    `n` cells are placed as a spherical Voronoi tessellation of `n` points on a sphere of radius R,
    both in world units, so each face is a polygon and each vertex is shared by three of them.
    `jitter` disorders the generating points, from 0 (a regular arrangement) to 1; a regular one is
    not neutral, because the mechanics then inherits its symmetry.

    It also writes the targets the energy is measured against: A0 and P0 per face -- the target
    area and perimeter, in world units squared and world units -- and V0, the target lumen volume.
    Those are what growth raises and what the mechanics tries to reach.

    `shape` selects the geometry: `sphere` is the closed vesicle, `disc` an open flat patch. It is
    a VALUE and not a model, because the hypothesis about the tissue is identical in both -- the
    same energy, the same operators -- seeded into two different geometries. A disc has a rim whose
    cells have fewer neighbours, so the run has to be read knowing the rim does not divide.

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644.
    """
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["vesicle", "epithelial_shell", "spherical", "half_edge_mesh", "initial_condition"]
    REFERENCE = "Okuda, S. et al. (2013). Reversible network reconnection model for simulating large deformation in 3D tissues. Biomech. Model. Mechanobiol. 12:627-644; tyssue (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.n = int(params.get("n_cells", 150)); self.R = float(params.get("radius", 5.0))
        self.jitter = float(params.get("jitter", 0.15)); self.p0 = float(params.get("p0", 3.9))
        self.seed = int(params.get("seed", 0))
        # WHERE THE SPHERE GOES. `build_sphere_mesh` returns it about the ORIGIN, and until now
        # nothing could move it: every spec that used this ran in a `free` world centred on 0, so a
        # unit box put the vesicle in a corner. Default [0,0,0], so the 461 specs that never asked
        # are byte-identical.
        self.centre = [float(v) for v in params.get("centre", [0.0, 0.0, 0.0])]
        # `shape: sphere | disc` -- THE GEOMETRY OF THE INITIAL CONDITION, and it is a value on this
        # operator rather than a `model:` because a flat patch and a closed shell are the same
        # hypothesis about the tissue seeded into two different geometries. `seed_plate` already
        # spells its geometry this way.
        #
        # A DISC IS AN OPEN MESH, which the rest of the vertex stack has never seen: `divide_face_3d`
        # refuses a face whose split edges lack two distinct interior neighbours, and every rim face
        # is one. That is the point of seeding it -- a flat sheet is where the monolayer's normal
        # offset can be tuned with no curvature in the picture -- but it means the rim does not
        # divide, and the run has to be read knowing that.
        #
        # `hexagon` IS THE THIRD VALUE, added for gate row AB-C2: one regular hexagonal cell, the
        # only solid in the apico-basal promotion whose shape index is known on paper. See
        # `build_hexagon_mesh`. It is a value for the same reason `disc` is -- the same hypothesis
        # about the tissue, seeded into a geometry chosen so a closed form is reachable.
        self.shape = str(params.get("shape", "sphere")).lower()
        if self.shape not in ("sphere", "disc", "hexagon"):
            raise ValueError(f"mesh_seed: shape must be 'sphere', 'disc' or 'hexagon', "
                             f"got {self.shape!r}")
        # PARTITION THE TYPES BY POSITION, NOT AT RANDOM. `type_layout` on a SET is applied at build
        # (engine.py), over that set's own coordinates -- which a mesh cell does not have yet: the
        # cells do not exist until this operator runs, so the build-time split assigns every one of
        # them to type 0 and a two-type spheroid comes out uniform. Declaring it here instead cuts
        # the shell at the equator of the axis named, tiling the declared `fraction`s along it, at
        # the moment the faces first have centroids.
        self.type_layout = str(params.get("type_layout", "random")).lower()
        self.cell_set = params.get("cell_set", "cell")
        # THE PREFERRED AREA, SET APART FROM THE GEOMETRY. `A0` was always `mean(area)` of the mesh
        # just built, so every run started exactly at its target and the ONLY direction it could go
        # was down, under whatever line tension the mechanics carried: a sweep of six specs that all
        # ended between A/A0 0.44 and 0.72 and never once tested expansion.
        #
        # It is also the missing half of the Marinari crowding transform. `cell_mechanics
        # [model: marinari]` rescales K, Lambda and Gamma by g^-2, g^-1/2 and g^-1, which leaves W
        # invariant ONLY together with A0' = g A0 -- and that line is the seeder's, so `crowding: 4`
        # weakened the area term 16x against a line term weakened 2x and collapsed the shell to
        # radius 1.4 with CV(junction length) 1.9. With this, `a0_scale: g` completes the transform,
        # and the paper's own words become the gate: a tissue "free to expand so that its total area
        # was larger by a factor g" must settle at A = g x its seeded area.
        self.a0_scale = float(params.get("a0_scale", 1.0))
        if self.a0_scale <= 0:
            raise ValueError(f"mesh_seed: a0_scale must be > 0, got {self.a0_scale}")
        self.vseed_cv = float(params.get("vseed_cv", 0.0))       # STOCHASTIC VOLUME SEED: per-cell random cell-cycle
        #   phase at t=0 (spread of the initial division threshold) -> desynchronises the FIRST division wave
        # `age_seed` -- THE OTHER WAY TO DESYNCHRONISE, AND THE ONLY ONE A CLOCK CAN USE. `vseed_cv`
        # spreads the THRESHOLD (`divjit`), which is a phase only for a rule that reads size:
        # `cell_divide[timer]` fires on `age >= cycle * jit` and every cell is seeded with age 0, so
        # a clock-driven tissue divides in ONE frame however wide `vseed_cv` is -- measured on
        # ab_06_population: 1279 of 1280 cells divided at frame 480 and on no other frame.
        #
        # AND SPREADING THE THRESHOLD IS THE WRONG KNOB WHEN THE CELLS MUST COME OUT THE SAME SIZE.
        # Paired with `cell_grow[model: timer]`, which drives every cell to one target volume in
        # `cycle_frames`, a cell that divides early divides SMALL: threshold noise buys sparseness
        # by spending size uniformity. Spreading the initial AGE instead leaves every cell the same
        # cycle length -- so they all reach the same size at division -- and merely offsets them in
        # time. Uniform volumes and sparse division are then not in tension.
        #
        # In DIVISION-CALLS, the unit `cycle`, `min_cycle` and `max_cycle` use; set it to `cycle` to
        # spread the population evenly over one generation. 0 (the default) seeds age 0 as before.
        self.age_seed = float(params.get("age_seed", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device; dt = lvl.state.dtype
        _build = {"sphere": build_sphere_mesh, "disc": build_disc_mesh,
                  "hexagon": build_hexagon_mesh}[self.shape]
        verts, es, et, ef, nF = _build(self.n, self.R, self.jitter, self.seed)
        Nv = verts.shape[0]; Nbuf = lvl.state.shape[0]
        if Nv > Nbuf:
            raise ValueError(f"sphere mesh has {Nv} vertices but buffer n={Nbuf}")
        verts = verts + np.asarray(self.centre, verts.dtype)
        pos = torch.zeros(Nbuf, 3, dtype=dt, device=dev)
        pos[:Nv] = torch.as_tensor(verts, dtype=dt, device=dev)
        px0, px1 = lvl.state_schema["pos"]
        st = lvl.state.clone(); st[:, px0:px1] = pos; lvl.state = st
        if getattr(lvl, "occ", None) is not None:
            occ = torch.zeros(Nbuf, device=dev); occ[:Nv] = 1.0; lvl.occ = occ
        est = torch.as_tensor(es, device=dev); ett = torch.as_tensor(et, device=dev)
        eft = torch.as_tensor(ef, device=dev)
        area, perim, cen, vf = face_geometry_3d(pos[:Nv], est, ett, eft, nF)
        A_seed = float(area.mean())
        A0 = A_seed * self.a0_scale; P0 = self.p0 * (A0 ** 0.5)
        if self.a0_scale != 1.0:
            print(f"[mesh_seed] preferred area A0 = {self.a0_scale:g} x the seeded mean area "
                  f"{A_seed:.4f} = {A0:.4f}: every cell starts at {1.0 / self.a0_scale:.3f} of its "
                  f"target, so the shell is under-inflated and must EXPAND "
                  f"(radius x {self.a0_scale ** 0.5:.3f} if the area term wins outright)", flush=True)
        if self.vseed_cv > 0:                                    # random initial cell-cycle phase per cell
            dj = np.clip(1.0 + self.vseed_cv * np.random.default_rng(self.seed + 101).standard_normal(nF), 0.4, 1.8)
        else:
            dj = np.ones(nF)                                     # all cells born in phase (synchronised)
        # UNIFORM AND NOT GAUSSIAN, because this is a PHASE and a phase is uniform on its cycle: a
        # tissue in steady state has as many cells just born as just about to divide. A Gaussian
        # would pile the population at the middle of the cycle and still give a division wave, only
        # a rounder one. See `age_seed` in __init__ for why the phase and not the threshold.
        ag = (np.random.default_rng(self.seed + 202).random(nF) * self.age_seed
              if self.age_seed > 0 else np.zeros(nF))
        # A `MeshTable`, WHICH IS A `dict` -- see `plexus.models.mesh`. Every reader is unchanged
        # by construction: the type still passes `isinstance(m, dict)` (which the D4 acted-ledger
        # tests, and whose failure would score every mesh-only operator as inert), still iterates,
        # still takes `get`/`setdefault`/`m[k] = v` on an open namespace. What it adds is a name for
        # the thing, a place for the carry and the snapshot to live, and an engine that knows it
        # exists -- `grep -rn "_mesh" src/plexus/` returned NOTHING before this.
        #
        # FILLED, NOT REBOUND, when the set declared `mesh: half_edge`. `engine._build_mesh`
        # allocated the table during `build`, and rebinding here would hand every consumer a
        # DIFFERENT OBJECT from the one the engine allocated -- which is the whole failure mode the
        # declaration exists to end: anything that took a reference before the seed ran (the
        # recorder, a probe, the salvage) would keep writing into an orphan. So the seed writes
        # THROUGH the existing table when there is one, and only creates one for a spec that has
        # not declared it yet, which is nearly all of them.
        if self.type_layout.startswith("split_"):
            from plexus.engine import retype
            _ax = "xyz".index(self.type_layout[-1])
            _cl = H.level(self.cell_set) if self.cell_set in H.levels else None
            _fr = getattr(_cl, "_type_fracs", None) if _cl is not None else None
            if _cl is None or _fr is None or int(_fr.numel()) < 2:
                print(f"[mesh_seed] type_layout={self.type_layout} ignored: set "
                      f"{self.cell_set!r} declares fewer than two types", flush=True)
            else:
                order = torch.argsort(cen[:, _ax])              # faces, sorted along the axis
                nt = torch.zeros(int(_cl.n), dtype=torch.long, device=dev)
                cuts = (torch.cumsum(_fr / _fr.sum(), 0) * nF).round().long().tolist()
                lo = 0
                for tid, hi in enumerate(cuts):
                    nt[order[lo:min(int(hi), nF)]] = tid
                    lo = int(hi)
                retype(_cl, nt)
                _names = list(getattr(_cl, "type_names", []) or [])
                _tally = "  ".join(f"{(_names[t] if t < len(_names) else t)}={int((nt[:nF] == t).sum())}"
                                   for t in range(int(_fr.numel())))
                print(f"[mesh_seed] type_layout={self.type_layout}: {nF} cells cut at the "
                      f"{'xyz'[_ax]} equator -> {_tally}", flush=True)
        seeded = dict(E_srce=est, E_trgt=ett, E_face=eft, nF=nF, Nv=Nv,
                         A0=torch.full((nF,), A0, dtype=dt, device=dev),
                         P0=torch.full((nF,), P0, dtype=dt, device=dev),
                         alive=torch.ones(nF, dtype=dt, device=dev),
                         divjit=torch.as_tensor(dj, dtype=dt, device=dev),   # per-cell division-threshold multiplier
                         age=torch.as_tensor(ag, dtype=dt, device=dev),      # per-cell cell-cycle PHASE at t=0
                         # THE VOLUME AND RADIUS TARGETS FOLLOW `a0_scale`, on the isotropic rescale
                         # it implies: a preferred AREA g times larger is a preferred LENGTH sqrt(g)
                         # larger, hence a volume g^1.5 larger. Leaving V0f at the seeded value would
                         # set `K_V` against the very expansion `a0_scale` asks for, so the shell
                         # would stall short of its area target and the gate would fail on a
                         # constraint the paper's 2D model does not even have. `Vbirth` follows V0f
                         # for the same reason: a cell must still divide at twice its OWN target,
                         # not at twice a volume it was never meant to hold.
                         V0f=vf.detach().clone() * self.a0_scale ** 1.5,   # PER-CELL target wedge volume (v_eq per cell)
                         Vbirth=vf.detach().clone() * self.a0_scale ** 1.5,   # volume at birth -> cell divides when it doubles
                         V0=float(vf.sum()) * self.a0_scale ** 1.5,
                         v_ref=float(vf.median()) * self.a0_scale ** 1.5,   # REFERENCE cell volume (Okuda v_ref) -> uniform cells:
                         #   morphogen growth caps v_eq at (4/3)v_ref, cells cycle in [2/3,4/3]v_ref centred on v_ref
                         R0=float(np.linalg.norm(verts - np.asarray(self.centre, verts.dtype),
                                                axis=1).mean()) * self.a0_scale ** 0.5, verts0=verts)
        m = getattr(lvl, "_mesh", None)
        if isinstance(m, MeshTable):
            m.clear(); m.update(seeded)          # the engine's table, filled in place
        else:
            lvl._mesh = MeshTable(**seeded)      # spec has no `mesh:` declaration yet
        return {}


@register_operator("seed_mesh", "mesh_seed", set="vertex", kind="seed", family="seed",
                   implementation="apicobasal")
class SeedMeshApicoBasal(SeedMesh3D):
    """`apicobasal` IMPLEMENTATION of seed_mesh -- the same epithelium, seeded WITH ITS THICKNESS.

    R2 of the apico-basal promotion, and the whole of it: the representation exists and does not
    move yet.

    WHAT IT ADDS. One per-vertex block, `sep`, the apico-basal HALF-separation, so that

        apical_i = pos_i + sep_i          basal_i = pos_i - sep_i

    `(pos, sep)` is an invertible linear map on R^6 of `(apical, basal)`, so this IS the full
    doubled degree-of-freedom set -- it removes the kinematic identity `a_i, b_i = x_i +/- (H_i/2)n_i`
    that `monolayer_shells` imposes, and nothing else. Apical and basal can now slide past each
    other, which is what wedging, bottle cells and apical constriction are made of and what the
    mid-surface hypothesis forbids at every parameter value.

    IT IS AN IMPLEMENTATION AND NOT A MODEL, by this operator's own written test: `shape: sphere |
    disc` is a VALUE "because a flat patch and a closed shell are the same hypothesis about the
    tissue seeded into two different geometries", and a thickness at birth is the same kind of
    statement about the initial condition. The HYPOTHESIS lives in `cell_mechanics[model:
    apicobasal]` at R3; this only writes x_0. Registering a separate class also leaves the default
    class untouched, so every existing spec is byte-identical.

    THE SEPARATION IS ALONG THE VERTEX NORMAL, half the cell thickness each way. Vertex normals
    rather than face normals is the load-bearing choice inherited from `monolayer_design.md`: on a
    curved sheet it makes the apical and basal caps DIFFER in area, which is what gives bending
    stiffness ~ kappa_s h^2 without an explicit K_bend. Face normals give parallel caps and no
    single-cell bending term at all.

    `h0` IS THE FULL CELL THICKNESS, and `sep` is half of it, so that `|a_i - b_i| = h0`. Stated
    because the monolayer's own `h0` means the same thing and a factor of two here would be
    invisible until a cap-area ratio came out wrong.
    """
    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.h0 = float(params.get("h0", 0.4))                 # FULL thickness; sep is h0/2
        self.sep_block = str(params.get("sep_block", "sep"))

    def forward(self, H, mask=None):
        out = super().forward(H, mask)                         # pos, occ and the mesh table
        lvl = H.level(self.at)
        if self.sep_block not in lvl.state_schema:
            raise ValueError(
                f"seed_mesh[apicobasal] needs a `{self.sep_block}` block on the {self.at!r} set -- "
                f"declare it under `sets:` as `{self.sep_block}: {{width: 3, integration: "
                f"first_order}}`. Without it there is nowhere to put the apico-basal separation and "
                f"the run would be the mid-surface model wearing a different operator's name.")
        # THE SPATIAL SCHEMA MUST STILL BE THE SPATIAL SCHEMA. `_resolve_schema` says "the spec
        # always wins", so a `state:` block REPLACES the set's schema rather than extending it, and
        # `schema_from_spec` defaults every block to first_order/free. A spec that declares `sep`
        # and restates `pos` without `boundary: world` runs perfectly well and silently loses the
        # world clamp -- positions stop being confined and nothing says so. Checked here because
        # this operator is the only thing that knows the spec meant to keep a spatial set spatial.
        _coord = lvl.state_schema.coordinate
        if _coord is None or _coord.name != "pos":
            raise ValueError(
                f"seed_mesh[apicobasal]: the {self.at!r} set's coordinate block is "
                f"{_coord.name if _coord else None!r}, not `pos`. Declaring a `state:` block "
                f"REPLACES the schema; restate `pos` as "
                f"{{width: 3, role: coordinate, integration: second_order_coordinate, "
                f"boundary: world}} alongside `{self.sep_block}`.")
        if _coord.boundary != "world":
            raise ValueError(
                f"seed_mesh[apicobasal]: `pos` has boundary {_coord.boundary!r}, not 'world'. The "
                f"spec's `state:` block dropped the world clamp -- the run would not be confined "
                f"and nothing else would say so.")
        m = getattr(lvl, "_mesh", None)
        Nv = int(m["Nv"])
        pos = lvl.get("pos")[:Nv]
        # THE SAME NORMAL THE ENERGY USES, not a second definition of it. `monolayer_shells`
        # returns (apical, basal, vertex_normal, vertex_thickness) and its docstring is explicit
        # about why one formula with two callers beats two copies: "Two copies of 'offset along the
        # vertex normal by h/2' would agree until one of them changed, and the picture would then
        # stop being a picture of the model." The thickness argument is a dummy here -- only the
        # normal is taken -- because at R2 the separation is uniform and `h0` sets it directly.
        nF = int(m["nF"])
        _h = torch.full((nF,), self.h0, dtype=pos.dtype, device=pos.device)
        _a, _b, n, _hv = monolayer_shells(pos, m["E_srce"], m["E_trgt"], m["E_face"], nF, _h)
        c0, c1 = lvl.state_schema[self.sep_block]
        st = lvl.state.clone()
        st[:, c0:c1] = 0.0
        st[:Nv, c0:c1] = (0.5 * self.h0) * n
        lvl.state = st
        # DECLARED FOR THE CARRY, so a vertex born on a septum inherits its parents' separation
        # instead of the buffer's zero -- a cell of zero height along the seam it just grew.
        # R1(b) built `carry_vertices`; this is its first real consumer.
        declare_vertex_carry(m, self.sep_block)
        print(f"[seed_mesh:apicobasal] {Nv} vertices carry an apico-basal separation of "
              f"h0/2 = {0.5 * self.h0:g} along the vertex normal (cell thickness {self.h0:g})",
              flush=True)
        return out


@register_operator("cell_mechanics", set="vertex", kind="lateral", family="mechanics")
class ShapeEnergy3D(Lateral):
    """3D vertex-model shape-energy force on the vesicle vertices:
        E = sum_f [ K_A(A_f-A0)^2 + K_P(P_f-P0)^2 + K_V(v_f - v_eq_f)^2 ] + Lambda*sum_e l_e .
    NOT AN "AVM", WHICH IS WHAT THIS SAID AND IS A DIFFERENT MODEL. `AVM` is the Active Vertex
    Model of Barton, D. L., Henkes, S., Weijer, C. J. & Sknepnek, R. (2017), PLoS Comput. Biol.
    13(6):e1005569, and it differs from this in both of its defining ingredients: it is ACTIVE (the
    cells are self-propelled, active-matter dynamics on top of the vertex energy) and it is
    CENTRE-BASED (contacts are generated dynamically from cell CENTRE positions, a Voronoi
    construction). This operator is passive -- force = -grad E, overdamped, no active term -- and
    the degrees of freedom ARE the vertices of a half-edge mesh, which is the true-vertex lineage
    (Tyssue, Okuda). Borrowing the name imports
    a claim of self-propulsion the code does not make.

    K_V is a PER-CELL volume elasticity on each cell's wedge volume v_f (Turing_vertex Eq.3 / tyssue
    ClosedMonolayer), not a single global lumen term: it keeps every cell inflated and resists local
    buckling, so growth (ramping v_eq per cell) inflates the shell smoothly. Force = -grad E by one 3D
    autograd pass; bounded overdamped Euler (displacement capped at cap_frac x mean edge). EMIT=velocity."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    REQUIRES_PARAMS = ["p0"]
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "shape_energy", "cell_volume_elasticity", "vesicle", "force_balance"]
    # THE SECOND CITATION WAS THE WRONG OKUDA 2015, AND THE PAGES WERE WRONG TOO. It read
    # "Biomech. Model. Mechanobiol. 14:413-421 (3D volume/surface)": that paper is 413-425,
    # and it is the VISCOSITY paper -- local velocity fields to make vertex dynamics Galilean
    # invariant -- which this operator does not implement. The 3D volume/surface energy is the
    # other Okuda 2015, in Biophysics and Physicobiology.
    REFERENCE = ("Farhadifar, R., Roper, J.-C., Aigouy, B., Eaton, S. & Julicher, F. (2007). "
                 "Curr. Biol. 17(24):2095-2104, doi:10.1016/j.cub.2007.11.049 -- the vertex-model "
                 "shape energy (area, perimeter, line tension) this generalises; "
                 "Okuda, S., Inoue, Y. & Adachi, T. (2015). Three-dimensional vertex model for "
                 "simulating multicellular morphogenesis. Biophys. Physicobiol. 12:13-20 -- the "
                 "3D volume/surface form. NOT an Active Vertex Model: see the class docstring.")
    PARAM_ROLES = {"p0": "target_shape_index", "K_A": "area_stiffness", "K_P": "perimeter_stiffness",
                   "Lambda": "surface_tension", "K_V": "cell_volume_elasticity", "cap_frac": "stability_cap",
                   "centre": "origin_of_the_K_R_radial_term"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.K_A = float(params.get("K_A", 1.0)); self.K_P = float(params.get("K_P", 1.0))
        self.p0 = float(params.get("p0", 3.9)); self.Lambda = float(params.get("Lambda", 0.1))
        self.K_V = float(params.get("K_V", 0.5)); self.K_R = float(params.get("K_R", 0.0))
        _c = params.get("centre", None)
        self._centre = None if _c is None else torch.tensor([float(v) for v in _c])
        # DIHEDRAL bending (Wardetzky): penalises adjacent-cell normal deviation -> smooths the local folds
        # the hollow metric flags (division-injected cap tilts). High-frequency: unlike the radial K_R it
        # does NOT flatten gentle whole-shell/tube curvature. 0 = off (default). See _shape_energy_core.
        self.K_bend = float(params.get("K_bend", 0.0))
        # GLOBAL LUMEN incompressibility (isoperimetric): penalise enclosing less volume than a sphere of
        # the current area -> distinguishes sphere from a per-cell-volume-preserving buckle. 0=off. Coral only.
        self.K_lumen = float(params.get("K_lumen", 0.0))
        self.Gamma = float(params.get("Gamma", 0.0))             # cortical contractility (1/2)Gamma*P^2 -> rounds cells
        self.Gam_l = 0.0                # per-JUNCTION (1/2)Gamma*l^2; set by model=marinari, not by a spec
        self.mu = float(params.get("mu", 1.0))
        self.dt = float(params.get("dt", 1.0)); self.relax_iters = int(params.get("relax_iters", 6))
        self.eta = float(params.get("eta", 0.08)); self.cap_frac = float(params.get("cap_frac", 0.12))
        # ANTI-INVERSION filtered step (IPC-analog, differentiable): hollow caps are inverting faces
        # (signed wedge volume v_f flips sign at the division septum). Each bounded-Euler substep, scale
        # back the move of any vertex whose incident face would drop v_f below `antiinv` x median(v_f) --
        # a move that only makes an already-inverted face WORSE is blocked, a recovering move is allowed.
        # Straight-through (scale detached) so the rollout stays differentiable. 0 = off (default).
        # `plane_axis` -- CONSTRAIN THE SOLVER TO A PLANE, which is what makes this a 2D vertex
        # model. The ENERGY is unchanged; only the descent is projected, so a flat sheet stays in
        # its plane by construction rather than by a restoring force fighting it.
        #
        # A PENALTY CANNOT DO THIS JOB. This loop takes `relax_iters` free 3D steps per frame and
        # an external penalty corrects only once afterwards, so the sheet undulates by many times
        # the declared gap in between. Raising the penalty stiffness does not help: an explicit
        # penalty is stability-limited exactly where it would start to bite, and the run diverges
        # before it becomes strong enough to hold the sheet flat.
        #
        # None (the default) leaves every existing run untouched.
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)
        self.antiinv = float(params.get("antiinv", 0.0))
        # Lloyd-like tangential regularization (vertex-model analog of Turing's surface_lloyd): rounds cells
        self.smooth_iters = int(params.get("smooth_iters", 0)); self.smooth_w = float(params.get("smooth_w", 0.0))

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
        """dE/d(pos), through the hand-written warp kernels when they apply, autograd otherwise.

        WARP IS THE DEFAULT. The energy is unchanged -- `vertex_warp` implements the derivative OF
        `_shape_energy_core`, checked term by term against `torch.autograd.grad` in
        `tests/test_vertex_warp.py` at a relative error of ~3.5e-07, which is float32 round-off.
        What changes is the cost: the autograd backward was the single largest entry in the frame's
        profile (25.6 s of 85.8 s on `mesh_mpm_spheroid_nominal` at frame 380), and it is
        LAUNCH-bound rather than arithmetic-bound -- 3.27 ms per call whether the mesh has 1,188
        half-edges or 71,988. The warp path is 0.40 ms at every one of those sizes: 8.1x on the
        gradient, 3.5x on the whole frame (120.5 -> 34.1 ms/frame end to end).

        IT FALLS BACK RATHER THAN REFUSING, and says so once: on CPU, in float64, without warp
        installed, or when `K_bend` / `K_lumen` is on -- those two terms are not ported, and a term
        silently dropped from a gradient is a different model, not a faster one. The fallback is the
        ONLY way the autograd route is reached; there is no operator key that forces it, because
        which route differentiates one and the same energy is a backend choice and not a model.

        NOT BIT-IDENTICAL, and neither is the thing it replaces. Both accumulate per-face sums with
        float32 atomics whose order is not fixed -- warp's `atomic_add` and torch's `index_add` --
        so two runs of the UNMODIFIED default already differ from each other.
        """
        # LOCAL AND LATE: `try_shape_energy_grad` is defined at the BOTTOM of this file, below the
        # class that calls it, so a module-level import would be a forward reference.
        from plexus.operators.vertex_ops import try_shape_energy_grad
        g = try_shape_energy_grad(self, p, es, et, ef, nF, A0, P0, V0f, alive, R0t,
                                  eocc, vocc, twin_face, myo_e)
        if g is not None:
            return g
        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            # THE RADIAL TERM'S ORIGIN, AND WHY IT HAS TO BE DECLARABLE. `K_R` penalises
            # (|pos| - R0)^2 with |pos| measured from the WORLD ORIGIN, so an energy that reads as
            # translation-invariant is not: place the tissue at [25,25,25] of a 50-unit box and every
            # vertex sits 43 units out against an R0 of 4.65, and the term tears the shell apart --
            # 685 cells instead of 6,786, measured. Every other term (areas, perimeters, volumes,
            # edge lengths) IS translation-invariant, so evaluating the whole energy on `p - centre`
            # changes only this one and leaves the gradient of the rest identical.
            #
            # DEFAULT [0, 0, 0], so the 461 specs that never asked are bit-identical.
            if self._centre is not None:
                p_e = p - self._centre.to(p.device, p.dtype)
            else:
                p_e = p
            E = _shape_energy_core(p_e, es, et, ef, nF, A0, P0, V0f, alive, R0t, self.K_A, self.K_P,
                                   self.K_V, self.K_R, self.Lambda, self.Gamma, eocc, vocc,
                                   self.K_bend, twin_face, self.K_lumen, myo_e, self.Gam_l)
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
            # unreachable: `junction_sync` re-keys the array after every topology operator, so a
            # mismatch here means the sync is missing from the schedule or is placed before the operator
            # that resized the buffer. Counted and announced once rather than raised, because raising
            # would take down every archived specification that predates the sync operator.
            MYO_SKIPPED.append((int(myo.shape[0]), int(a[1].shape[0])))
            if len(MYO_SKIPPED) == 1:
                print(f"[cell_mechanics] myosin array is {myo.shape[0]} long against "
                      f"{a[1].shape[0]} half-edges -- relaxing WITHOUT myosin this frame. Schedule "
                      f"`junction_sync` after the topology operators.", flush=True)
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
        # `plane_axis` TRAVELS WITH THE REST. `cell_divide` heals the two fresh daughters and
        # their one-ring with `_relax_subset`, a SECOND descent loop reading these numbers -- and
        # a constraint the main loop honours and that one does not is a constraint the run does
        # not have. Measured: with the main relax projected into the plane and this omitted, the
        # sheet still drifted to sd(z) = 5.5% of an edge, and the drift tracked division activity.
        m["mech"] = dict(K_A=self.K_A, K_P=self.K_P, K_V=self.K_V, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.Gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)   # for cell_divide local relax
        x0 = pos_full[:Nv].detach().clone()
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        # Relax the LIVE vertices only: the energy is summed over the live faces and half-edges, so
        # `eocc` and `vocc` are all-ones here and the occupancy masking inside `_shape_energy_core`
        # is a no-op. (There is no padded-reservoir variant any more: whether the body below is
        # traced by a fusing compiler is the engine's `compile:` decision, not this operator's.)
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
            if self.plane_axis is not None:                   # 2D vertex model: no motion off the sheet
                step[:, self.plane_axis] = 0.0
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
        # DIVIDED BY THE ENGINE'S dt, NOT BY THIS OPERATOR'S, so the two cancel BY CONSTRUCTION.
        #
        # This is a UNIT CONVERSION and nothing else: the relaxation above solved a DISPLACEMENT
        # (`x - x0`), using `eta`, `cap_frac` and `relax_iters`, none of which involve a timestep.
        # `EMIT = "velocity"` means the engine will apply `pos += v * general.dt`, so the only
        # divisor that returns the displacement actually solved is `general.dt`.
        #
        # IT USED TO DIVIDE BY `self.dt`, and the two cancelled in every spec ever written because
        # both were 1.0. The moment a spec lowered `general.dt` to co-schedule an MPM continuum --
        # 0.0032 -- and left this at 1.0, the tissue relaxed at 0.32% of the rate it had solved for:
        # a vesicle that reaches 7,814 cells reached 270. NOTHING LOOKED WRONG. The shell stayed
        # round, every shape diagnostic passed, the contact behaved; it read as a growth-rate
        # problem and it was a unit-of-time problem, and it took a spec-for-spec ablation to find.
        # A parameter that must always equal another parameter is not a parameter.
        #
        # BIT-IDENTICAL FOR THE CORPUS: 1,606 of the 1,610 operator lines that declare a `dt` in
        # `config/` already set it to their spec's `general.dt`, so the ratio was already 1. The
        # four that did not were the specs this was found in.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics: `dt: {self.params['dt']}` is IGNORED -- the divisor is "
                 f"general.dt ({_dt}) so that the engine's own multiplication cancels it. Remove "
                 f"the parameter; leaving it in the spec suggests a knob that is not there.")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


# There is ONE growth operator on this mesh, `cell_grow` in diffusion_reaction, and deliberately
# so. A second one duplicated it and contradicted it: both wrote the same mesh targets, one
# multiplicatively (V0f <- V0f * g^3) and the other by assignment from its own snapshot
# (V0f <- V0f_init * s^3). Scheduled together, whichever ran second silently overwrote the first
# every frame, so the growth rate a specification declared was not the one it got.
#
# Every call site is ported: uniform body-wide inflation is `cell_grow` with `rho = 1` and the
# gate open (`a_sw = 0`). `max_scale` capped the LINEAR scale and `vth_frac` caps per-cell
# VOLUME, so the same plateau is `vth_frac = max_scale ** 3`.


@register_operator("cell_divide", set="vertex", kind="structural", family="population")
class Divide3D(Structural):
    """Cell division: a septum through a face, so one cell becomes two, and the surface gains a
    cell without losing its closure.

    vertex -> vertex: rebuilds the half-edge table, the vertex positions and every per-face array.

        divide cell j when   v_j  >  factor * v_ref

    v_j is the cell's current wedge volume and v_ref the SEED-TIME MEDIAN cell volume, both in
    world units cubed, so `factor` is dimensionless -- how many times the size of a typical cell at
    the start a cell must reach before it divides. That makes it an ABSOLUTE size checkpoint, and
    absolute is the point: a cell that is born small must grow more than one born large before it
    divides, which is what corrects size variance rather than perpetuating it.

    The septum runs between the midpoints of two edges of the face, and the two neighbouring faces
    that shared those edges are split as well, so the mesh stays a valid closed half-edge surface
    with every vertex still of degree three. Every per-face array -- A0, P0, V0f, age, ndiv, and
    anything an operator has registered for carry -- is reindexed through the new-to-old face map,
    so no state is left pointing at a face that has moved.

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644; the in-surface
    division follows the sheet topology of Tyssue.
    """
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["division", "cell_division", "vesicle", "proliferation", "volume_doubling"]
    REFERENCE = "Hertwig, O. (1884) (long-axis division rule); tyssue cell_division (DamCB)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        # A POPULATION CEILING, off by default so nothing archived changes. `before_frame` stops
        # division at a TIME; this stops it at a SIZE, which is what "grow to 30k cells and then
        # just let the chemistry run" needs -- the frame at which a given spec reaches a given
        # count is not knowable in advance, and guessing it either truncates the growth or wastes
        # the rest of the run.
        self.max_cells = int(params.get("max_cells", 0) or 0)
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
        # on m["mech"] by cell_mechanics. 0 = off (default); the coral/tube fix sets it > 0.
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
        # cell_grow then re-ramps activated daughters as their G1 regrowth. Off by default so other
        # presets (vesicle_divide/fig4) are unchanged; the tube preset turns it on.
        self.g1_ramp = bool(params.get("g1_ramp", False))
        # ORIENTED division at the red/white interface (Okuda's tube mechanism, user issue 3): the dividing
        # plane of an ACTIVATED (red) cell is oriented so the daughters stack ALONG the bud axis (the
        # direction from the vesicle centre to the activated tip) instead of by the cell's own long axis.
        # This adds new cells NORMAL to the body surface -> builds the tube WALL and EXTENDS the protrusion,
        # rather than widening it. orient_asw = activator threshold that flags an interface/red cell. 0 = off.
        self.orient_iface = bool(params.get("orient_iface", False))
        self.orient_asw = float(params.get("orient_asw", 1.0))
        # `project` -- PUSH THE TWO NEW MIDPOINTS BACK OUT TO THE LOCAL SHELL RADIUS. True is
        # right for a closed vesicle, where a septum midpoint left at the chord would dimple the
        # surface inward at every division. It is a SPHERE assumption, and on a flat sheet it is
        # simply wrong: |p| is measured from the world origin, so it lifts the midpoints off the
        # plane. On a flat disc it is the entire residual out-of-plane drift once the solver
        # itself is constrained: the sheet acquires an out-of-plane spread of several percent of an
        # edge length purely from division, even though its mechanics can no longer leave the
        # plane at all.
        self.project = bool(params.get("project", True))

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        """Has this cell earned a division? THE ONLY THING A `model=` VARIANT OF cell_divide CHANGES.

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
        # There is ONE way to desynchronise the cycle, `cycle_cv`. A second jitter parameter read
        # only on the `cycle_cv == 0` branch is unreachable whenever cycle_cv is set, so a search
        # editing it would measure exactly zero change and record that as evidence about the
        # mechanism rather than about the parameter being dead.
        if self.cycle_cv > 0:
            v = np.clip(1.0 + self.cycle_cv * rng.standard_normal(n), 0.4, 1.8)
        else:
            v = np.ones(n)                     # cycle_cv = 0 means synchronous, not "jittered a bit"
        return v if n > 1 else float(v[0])

    def forward(self, H, mask=None):
        from plexus.models.topology import rings_from_flat_3d, flat_from_rings_3d, divide_face_3d
        # THE VERTEX PARENTAGE, collected here and spent below. Empty `vertex_carry` -> a no-op, so
        # every existing spec is byte-identical; the list is built regardless because it costs two
        # tuples per division and a conditional would be one more thing to get wrong.
        births: list = []
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
        # THE POPULATION CEILING, and it is a HARD STOP rather than a throttle. Once the tissue has
        # `max_cells` cells nothing further divides, so the run keeps integrating -- chemistry,
        # mechanics, edge flips all continue -- on a population that has stopped changing. That is
        # the difference between this and `before_frame`: a frame number cannot say "when the
        # tissue is this big", because the frame at which a given spec reaches a given count is not
        # knowable before the run. Zero (the default) means no ceiling, so nothing archived moves.
        #
        # IT DROPS THE CANDIDATES RATHER THAN CAPPING THEM. Letting the last few through to land
        # exactly on `max_cells` would make the stop depend on how many cells happened to ripen on
        # the same tick, which is noise; refusing the whole tick makes the ceiling reproducible.
        if self.max_cells and nF >= self.max_cells:
            cand = []
        rng.shuffle(cand)                                        # unbiased when more cells are ready than the cap
        # NO THROTTLE, NO BYPASS. Every cell that is READY divides, and READY is the test two
        # lines above: current volume >= factor x jitter x its OWN birth volume, after min_cycle
        # (or past max_cycle, which is a backstop and must be set long or it becomes the rate).
        # That is P3 -- "a cell divides because it got big" -- and it is the mechanism the paper
        # is about. The pace therefore comes from how fast cell_grow inflates cells:
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
        # A volume cap on the division trigger decouples the division rate from the growth rate
        # entirely: cells divide and shrink, so the trigger fires earlier and earlier and the
        # tissue overruns its target cell count by a large factor while the rate written in the
        # specification is masked. What such a run measures is the counter, not the tissue.
        ndone = 0
        daughter_mothers = []                                    # mother face index of each appended daughter (order)
        bud_axis = None; a_cells = None; orient_thr = None       # ORIENTED interface division: bud axis = centre->red tip
        if self.orient_iface and self.cell_set is not None:
            clvl0 = H.level(self.cell_set)
            if clvl0 is not None and "chem" in clvl0.state_schema:
                ci0, _ = clvl0.state_schema["chem"]
                a_cells = clvl0.state[:nF, ci0].detach().cpu().numpy()
                # RELATIVE TO THE FIELD, for the reason `rd_interface_tension.a_sw` is: an absolute
                # threshold on a field whose scale the chemistry sets is one edit from selecting
                # nothing. `orient_asw` defaulted to 1.0 with the same (0.2, 6.0) range, and only
                # 20 of 78 campaign runs ever reached act_max > 1.0 -- so in 74% of runs
                # `cell_divide:orient_iface`, a named Okuda mechanism, could orient nothing and was
                # behaviourally `hertwig`. A `set_impl ... orient_iface` edit was a silent no-op.
                amax = float(a_cells.max()) if a_cells.size else 0.0
                thr = orient_thr = self.orient_asw * amax
                rc = [np.array([pos[v] for v in rings[f]]).mean(0) for f in range(nF)
                      if amax > 0 and a_cells[f] > thr
                      and rings[f] is not None and len(rings[f]) >= 3]
                if len(rc):
                    ba = np.mean(rc, 0); nba = float(np.linalg.norm(ba))
                    if nba > 1e-6:
                        bud_axis = ba / nba
        blocked = 0                     # divisions the RESERVOIR refused, not the biology
        # THE EDGE->FACE MAP IS BUILT ONCE FOR THE FRAME, not once per dividing cell.
        #
        # `divide_face_3d` needs two lookups out of it and used to rebuild the whole O(E) dict for
        # them. That made the operator quadratic in the tissue: both the edge count and the number
        # of cells ripening on a tick grow together, so the rebuild cost grew with the product.
        # Profiled on `mesh_mpm_spheroid_nominal` at frame 380 it was 19.4 s of `cell_divide`'s
        # 24.2 s, over 5,480 rebuilds in 96 frames -- 57 rebuilds a frame, one per division.
        # Built here and maintained inside `divide_face_3d`, it is one build a frame.
        emap = _edge_face_map(rings)
        for f in cand:
            if len(pos) + 2 > buf:
                # THE VERTEX BUFFER IS FULL. Counted and reported, never silent.
                #
                # A bare `break` would be the whole story: division simply stops and nothing says
                # why. A buffer sized for the seed cell count caps the tissue at (V+4)/2 cells, and
                # a run that reaches it adds ZERO cells for every remaining frame. Such a run
                # measures a full array rather than a tissue, and every numerical check still
                # passes it -- the cell count is simply constant, which is a legal trajectory.
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
                # THE SAME RELATIVE THRESHOLD THAT BUILT THE AXIS. This test used to read
                # `a_cells[f] > self.orient_asw` -- the ABSOLUTE value -- while the axis above was
                # already relative to `amax`, so the two disagreed about which cells are the bud.
                # At orient_asw 0.6 on a field peaking at 1.47 the axis takes the top 40% and this
                # took everything above 0.6, a different and larger set; on any run whose field
                # peaks below orient_asw the axis exists and NO cell passes here, which is the
                # silent-`hertwig` failure the comment above says was fixed.
                if bud_axis is not None and orient_thr is not None and f < len(a_cells) and a_cells[f] > orient_thr:
                    ut = bud_axis - float(np.dot(bud_axis, n)) * n   # bud axis in this cell's tangent plane ->
                    if np.linalg.norm(ut) > 1e-6:                    # daughters separate ALONG the protrusion (extend the
                        u = ut / np.linalg.norm(ut)                  # wall) instead of by the cell's own long axis
                w = np.cross(n, u); w = w / (np.linalg.norm(w) + 1e-9)   # short-axis direction in the tangent plane
                mids = 0.5 * (P + np.roll(P, -1, 0)); proj = (mids - c) @ w
                ea, eb = int(np.argmax(proj)), int(np.argmin(proj))
            except Exception:
                ea, eb = 0, len(r) // 2
            res = divide_face_3d(rings, pos, f, ea=ea, eb=eb, emap=emap, project=self.project,
                                 births=births)
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
            print(f"[cell_divide] RESERVOIR FULL: {blocked} division(s) refused for want of vertex "
                  f"buffer ({len(pos)}/{buf}). This run is capped by its array, not by its "
                  f"biology -- every later measurement describes the reservoir.", flush=True)
            return {}
        # THE PER-VERTEX CARRY, spent here for the same reason `reindex_faces` is spent beside the
        # per-face one: a topology edit that changes who a row IS must be followed by every array
        # that is indexed by that row, in the same place, or the fourth operator does not get the
        # memo. With no `vertex_carry` declared this walks an empty name list and returns, so every
        # existing spec is byte-identical -- which is the claim R1 has to make and the twin measures.
        if hasattr(m, "carry_vertices") and births:
            m.carry_vertices(births, dt=dt, dev=dev, level=lvl)
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
            # AND ITS TYPE, which is not part of `state` and was therefore not inherited. `node_type`
            # is its own buffer, so a daughter kept whatever type its BUFFER SLOT was assigned at
            # build -- which for the usual contiguous fraction split is type 0 for every low index.
            # Measured on a 200-cell spheroid seeded 100 soft / 100 tense: by frame 401 it was 4,104
            # soft and 100 tense, every one of the 4,004 daughters born soft, and the type predicted
            # which hemisphere a cell was in 53.8% of the time -- chance. Any per-type property
            # (myosin drive, adhesion, a clone label) silently washed out of a growing tissue, and
            # the run still looked plausible because the SHAPE change it produced early on persisted.
            nt = getattr(clvl, "node_type", None) if clvl is not None else None
            if nt is not None and int(nt.numel()) >= nF2:
                from plexus.engine import retype
                nt = nt.clone()
                for i, mother in enumerate(daughter_mothers):
                    nt[nF + i] = nt[mother]
                retype(clvl, nt)                                 # and put the derived buffers back
        return {}



@register_operator("cell_die", set="vertex", kind="structural", family="population")
class Apoptosis3D(Structural):
    """Cell elimination: the Die family, and the inverse of `cell_divide`. A marked cell contracts
    its own ring to a point over several frames and is then removed from the surface.

    vertex -> vertex: shrinks the marked cells' target area and perimeter, then rebuilds the
    half-edge table when a ring has contracted far enough.

        A0_j <- A0_j * (1 - 1/tau)          while the cell is dying
        remove cell j when its ring has contracted to a triangle

    tau is the number of frames a cell takes to disappear. Extruding gradually rather than deleting
    outright is what keeps the surface valid at every frame: a face removed in one step would leave
    a hole its neighbours have no rule for closing.

    It matters that this family exists at all. Growth inflates and division subdivides, so a vertex
    model with no Die operator can only deform its sheet OUTWARD, and invagination -- one of the
    canonical morphologies -- is unreachable however the other parameters are set.

    The base contract carries no criterion of its own; every `model:` below differs ONLY in what
    marks a cell for death, exactly as `cell_divide`'s models differ only in what triggers a
    division.

    Reference: Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12(4):627-644; live-cell
    extrusion from an epithelium, Rosenblatt, J., Raff, M. C. & Cramer, L. P. (2001). Curr. Biol.
    11:1847-1857.
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
        # failure worth guarding against. `competition` is the default because it needs no
        # chemistry, so it is available on every base, and it is the canonical cell-competition
        # hypothesis.
        #
        # What makes a cell die is on the `model:` AXIS, not a `mode:` setting. It is the same kind
        # of choice as what makes a cell divide, and `cell_divide` puts that on the model axis --
        # two mechanisms for one question, in adjacent classes in this file.
        #
        # TWO OF THE THIRTEEN COLLAPSED INTO VALUES rather than becoming models, because they were
        # never distinct hypotheses:
        #   `list`/`band`/`cone` -> `model: prescribed` with `region:` -- three GEOMETRIES of one
        #     claim ("the experimenter ablates this patch"), the `mesh_seed.shape` case.
        #   `field_high`/`field_low` -> `model: field` with `compare:` -- the docstring below already
        #     says these "carry no criterion of their own", so the direction is a value on one model.
        if "mode" in params:
            raise ValueError(
                "cell_die: `mode` is gone -- write `model:`. What makes a cell die is a hypothesis, "
                "the same kind `cell_divide` carries on `model:`. `list`/`band`/`cone` are now "
                "`model: prescribed` with `region:`, and `field_high`/`field_low` are "
                "`model: field` with `compare: high|low`.")
        _death = getattr(type(self), "DEATH", "competition")
        if _death == "prescribed":
            _death = str(params.get("region", "list"))
            if _death not in ("list", "band", "cone"):
                raise ValueError(f"cell_die[prescribed]: region must be list|band|cone, "
                                 f"got {_death!r}")
        elif _death == "field":
            _cmp = str(params.get("compare", "high"))
            if _cmp not in ("high", "low"):
                raise ValueError(f"cell_die[field]: compare must be high|low, got {_cmp!r}")
            _death = "field_" + _cmp
        self.mode = _death
        #   list|band|cone|small|stalled|chem_low
        #   LOCAL (vs neighbours): competition|smaller|dimmer|older|crowded|lonely
        #   PUBLISHED FIELD: field_high|field_low -- see `_marked`. These two carry no criterion of
        #   their own; `field` names a per-cell quantity some other operator measured and published
        #   on the mesh, and `field_frac` is the multiple of its live MEDIAN that counts as
        #   qualifying. Death stops needing a new branch for every new measurement.
        self.field = str(params.get("field", "elong"))
        self.field_frac = float(params.get("field_frac", 1.5))
        self.frac = float(params.get("frac", 0.04))
        self.cone_deg = float(params.get("cone_deg", 22.8))       # 10 cells across on a 2,000-cell ball
        self.band_deg = float(params.get("band_deg", 8.0))        # half-width of each ring
        self.n_bands = int(params.get("n_bands", 1))              # >1 -> that many latitude rings
        self.small_frac = float(params.get("small_frac", 0.35))   # (small) die below this x v_ref
        # THE THRESHOLD IS PER MODE, because the quantities have different dynamic ranges and one
        # number cannot serve them. The activator spans zero to its maximum across a spot
        # boundary, so a factor-of-two cut on it selects cells easily. Volume, age and growth vary
        # by a few tens of percent between neighbours, so the same factor-of-two cut on those asks
        # for a cell that essentially cannot exist in a healthy sheet, and the model then kills
        # nothing while appearing to be under test.
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
        # ANY FIELD PUBLISHED ON THE MESH, BY NAME. The four above are hard-coded because they are
        # the engine's own bookkeeping (volume, birth volume, age, chemistry); anything an operator
        # measures and publishes arrives here instead, so a new probe needs no branch of its own.
        # `cell_shape_probe` writes `elong`; asking for a name nothing published returns None,
        # which every caller reads as "unknown", never as zero.
        v = m.get(what)
        if v is None:
            return None
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return v[:nF] if v.ndim == 1 and v.shape[0] >= nF else None

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
        if self.mode in ("field_high", "field_low"):
            q = self._q(m, H, nF, self.field)
            if q is None:
                return None
            # SMALLEST DIES FIRST is this method's contract, so `field_high` -- where the WORST
            # offender is the largest value -- has to be negated or the cap would spare exactly
            # the cells the mode exists to remove.
            return (-q[idx]) if self.mode == "field_high" else q[idx]
        return None                                          # list/band/cone: a named population

    def _marked(self, m, H, nF):
        """The set of cell indices currently marked to die."""
        if self.mode == "list":
            return {c for c in self.cells if c < nF}
        # THE CENTROID IS A PRECONDITION OF TWO MODES, NOT OF DEATH. This guard used to stand here
        # unconditionally -- `if cen is None: return set()` -- so every mode in the operator was
        # silently gated behind a quantity only `band` and `cone` read. A geometric precondition
        # that switches off a competition rule, or a probe-driven one, is the same defect as an
        # absolute threshold on a relative field: the operator goes quiet for a reason unrelated to
        # its own mechanism, and quiet is indistinguishable from "nothing qualified".
        u = None
        if self.mode in ("band", "cone"):
            cen = m.get("cen_np")                                  # per-cell centroid from the rings
            if cen is None:
                return set()
            u = cen / (np.linalg.norm(cen, axis=1) + 1e-12)[:, None]
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
            #
            # RELATIVE, WHICH IS THE WHOLE POINT. `chem_low` marks every cell below a fraction of
            # the activator's maximum, so when the pattern weakens it marks the WHOLE TISSUE:
            # every cell shrinks, the activator maximum goes to zero, and not one cell is ever
            # extruded -- because death needs a cell squeezed to a triangle by neighbours that are
            # NOT shrinking too. A threshold against the population's own
            # median cannot do that: it always names a minority, by construction.
            #
            # V0f/Vbirth is growth since birth and `age` is time since birth. Both are already
            # carried across renumbering by `keep` -- in cell_divide and in this operator -- so this
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
        if self.mode in ("field_high", "field_low"):
            # DEATH PLUGGED ONTO A MEASUREMENT SOMEONE ELSE MADE, which is the whole point of the
            # pair. Every other mode in this operator computes its own criterion internally --
            # eleven bespoke measurements living inside one Die -- and the only one that did not
            # was `chem_low`, which reads the `chem` another operator wrote. These two generalise
            # that: name a field, and any operator that publishes it can drive death without a
            # line changing here. `cell_shape_probe` publishes `elong`; the next probe will
            # publish something else and this will already work.
            q = self._q(m, H, nF, self.field)
            if q is None:
                # UNPUBLISHED IS NOT ZERO. No probe ran, or its precondition was absent -- either
                # way nothing is known about these cells, and killing none of them is the only
                # honest reading. Killing all of them is what a `0 > thr` comparison on a
                # zero-filled array would have done for `field_low`.
                return set()
            ok = np.isfinite(q)
            if not ok.any():
                return set()
            # RELATIVE TO THE FIELD'S OWN SPREAD, for the reason every threshold here is: `elong`
            # is a shape index near 3.7 and an aspect near 1.5, two quantities with no common
            # scale, and one `frac` has to mean the same thing on both. The reference is the
            # MEDIAN of the live cells, not the max: one runaway cell must not set the scale of
            # the quantity that exists to find it. `frac` 1.5 with mode field_high therefore reads
            # "half again the typical cell".
            med = float(np.median(q[ok]))
            if not np.isfinite(med) or abs(med) < 1e-12:
                return set()
            thr = self.field_frac * med
            hit = (q > thr) if self.mode == "field_high" else (q < thr)
            return set(np.where(ok & hit)[0].tolist())
        return set()

    def forward(self, H, mask=None):
        # THE ACTED LEDGER IS BLIND TO THIS OPERATOR, AND TO THE WHOLE DIE FAMILY. Measured over
        # rounds r001-r012 of the live campaign: 24 runs killed cells, 6,693 deaths in total, and
        # `inert_operators` recorded `cell_die` as having acted in ZERO of them -- including
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
        # (written off inert twice, never having fired) and `cell_chem_from_shape` (100% acted, changing
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
        # LOCAL, as Divide3D does: topology_ops imports from this module, so a
        # module-level import here is circular.
        from plexus.models.topology import (rings_from_flat_3d, flat_from_rings_3d,
                                           face_collapse_3d)
        births: list = []                        # see Divide3D: the per-vertex carry's parentage
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
        # 1. SHRINK. cell_mechanics contracts the cell toward the smaller target; T1 then finds its
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
        # A DYING CELL'S CONTENTS GO SOMEWHERE: the sum of activity in the dying cell's vicinity
        # should not change much, by construction.
        #
        # Simply dropping the row from the cell state takes its activator and inhibitor out of the
        # tissue with it. That is a discontinuity in a conserved quantity, injected at every death
        # and accounted for nowhere -- a cell is extruded and its
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
            if face_collapse_3d(rings, pos_t, f, births=births):
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
                # 1e-9 floor and stays there -- so a bequest divided by that floor injects a
                # concentration many orders of magnitude too large in one step, and a few of those
                # compound. Without this guard the conservation rule above becomes an AMPLIFIER,
                # by dividing by a number the model allows to reach zero.
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
        # 3. REBUILD, exactly as cell_divide does: `keep` maps new face -> old face and carries every
        #    per-face array across, so nothing can fall out of step with the mesh.
        # THE PER-VERTEX CARRY, spent here for the same reason `reindex_faces` is spent beside the
        # per-face one: a topology edit that changes who a row IS must be followed by every array
        # that is indexed by that row, in the same place, or the fourth operator does not get the
        # memo. With no `vertex_carry` declared this walks an empty name list and returns, so every
        # existing spec is byte-identical -- which is the claim R1 has to make and the twin measures.
        if hasattr(m, "carry_vertices") and births:
            m.carry_vertices(births, dt=dt, dev=dev, level=lvl)
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
            # cell_chem_from_shape left it at -0.0948, and quartering chi left it at -0.0689.
            # ONE ENGINE CALL: state, occupancy, the coordinate delta accumulator AND the extra
            # first-order block deltas, which the hand-rolled version below never reached (its
            # guard tests `isinstance(key, tuple)` and `_delta_blocks` is keyed by level NAME).
            if not H.renumber_set(self.cat, keep, n_new=nF2):
                # LOUD, BECAUSE THE SILENT VERSION COST A WEEK. This returned False on every call
                # for the whole promotion and both call sites discarded it; the code it replaced at
                # least printed a warning. The counter also rides `MeshTable.SCALAR_RECORD`, so a
                # gate row can assert it stayed zero instead of a human having to read a log.
                m["renumber_failed"] = int(m.get("renumber_failed", 0)) + 1
                print(f"[cell_die] renumber_set({self.cat!r}) DID NOT ACT -- the cell state, its "
                      f"occupancy and its pending deltas are now mis-indexed against the mesh, and "
                      f"the chemistry will scramble from here", flush=True)
        # THE PENDING DELTAS MUST BE RENUMBERED TOO, AND THIS IS THE BUG THAT BROKE P12 ON EVERY
        # RUN WHERE A CELL DIED.
        #
        # The engine zeroes the delta accumulator once per TICK and integrates at the END of the
        # schedule, so `cell_chem_diffuse` and `cell_chem_react` deposit their per-cell deltas early and the
        # engine applies them last. An operator that RENUMBERS the set in between leaves every one
        # of those deltas pointing at a different cell -- a large negative flux meant for one cell
        # lands on another that has almost no activator, and the concentration goes negative. That
        # is exactly what happens: the activator goes NEGATIVE on every model that kills anything,
        # decaying afterwards as the scrambled deltas diffuse away, while a no-death control never
        # leaves zero.
        #
        # `cell_divide` never hit it because appending is not renumbering: daughters go to indices
        # >= nF and every existing cell keeps its own. Removal is the first operation in this
        # engine that moves a row.
        #
        # Permuting the accumulator with the SAME `keep` map fixes it wherever the operator sits
        # in the schedule. Confirmed independently by moving apoptosis ahead of the chemistry --
        # act_min held at 0.0000 for the whole run and P12 and P4 both cleared -- but that is a
        # discipline every spec would have to remember, and this is the guarantee instead.
        # (the deltas were renumbered by `H.renumber_set` above, together with the state)
        m["n_apop"] = int(m.get("n_apop", 0)) + gone
        # LOUD, because a death is not recoverable and the count must be auditable against the
        # cell count: nF -> nF2 should differ by exactly `gone` and by nothing else.
        print(f"[cell_die] tick {self._k}: extruded {gone} cell(s), {nF} -> {nF2} faces "
              f"(marked {len(marked)}, total {m['n_apop']})", flush=True)
        return {}


@register_operator("cell_die", model="competition", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DCompetition(Apoptosis3D):
    """`competition` MODEL of cell_die -- grows slower than its neighbours -- the Myc-style loser.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "competition"


@register_operator("cell_die", model="smaller", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DSmaller(Apoptosis3D):
    """`smaller` MODEL of cell_die -- smaller than its neighbours: squeezed out by a fitter crowd.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "smaller"


@register_operator("cell_die", model="dimmer", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DDimmer(Apoptosis3D):
    """`dimmer` MODEL of cell_die -- less activator than its neighbours -- the LOCAL chemical loser.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "dimmer"


@register_operator("cell_die", model="older", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DOlder(Apoptosis3D):
    """`older` MODEL of cell_die -- has gone longer without dividing than its neighbours.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "older"


@register_operator("cell_die", model="crowded", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DCrowded(Apoptosis3D):
    """`crowded` MODEL of cell_die -- has more neighbours than its neighbours do -- density-driven extrusion.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "crowded"


@register_operator("cell_die", model="lonely", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DLonely(Apoptosis3D):
    """`lonely` MODEL of cell_die -- has fewer -- the tissue closes over a gap.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "lonely"


@register_operator("cell_die", model="small", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DSmall(Apoptosis3D):
    """`small` MODEL of cell_die -- below an ABSOLUTE volume threshold, not a relative one.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "small"


@register_operator("cell_die", model="stalled", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DStalled(Apoptosis3D):
    """`stalled` MODEL of cell_die -- below an absolute GROWTH-RATE threshold.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "stalled"


@register_operator("cell_die", model="chem_low", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DChemLow(Apoptosis3D):
    """`chem_low` MODEL of cell_die -- below an absolute activator threshold -- dies BETWEEN the spots.

    The only thing a `model=` variant of cell_die changes is what marks a cell for death, exactly
    as `cell_divide._trigger` is the only thing its models change.
    """
    DEATH = "chem_low"


@register_operator("cell_die", model="prescribed", set="vertex", kind="structural",
                   family="population")
class Apoptosis3DPrescribed(Apoptosis3D):
    """`prescribed` MODEL of cell_die -- the EXPERIMENTER ablates a patch; the tissue did not choose.

    ONE MODEL, THREE GEOMETRIES, and the geometry is a VALUE. `region: list | band | cone` names a
    set of cells (explicit indices), a set of latitude rings, or one contiguous cap. All three make
    the same claim -- "these cells are removed from outside" -- applied to different shapes, which is
    the `mesh_seed.shape: sphere | disc` case: "the same hypothesis about the tissue seeded into two
    different geometries". Splitting them into three models would have claimed three hypotheses where
    there is one.

    IT IS NOT A MECHANISM AND SHOULD NOT BE READ AS ONE. Every other model of this operator answers
    "what makes a cell die"; this one answers "which cells did we kill", which is a protocol.
    """
    DEATH = "prescribed"


@register_operator("cell_die", model="field", set="vertex", kind="structural", family="population")
class Apoptosis3DField(Apoptosis3D):
    """`field` MODEL of cell_die -- death keyed to a per-cell quantity SOME OTHER OPERATOR published.

    The generic one, and the base class already says why it exists: these "carry no criterion of
    their own; `field` names a per-cell quantity some other operator measured and published on the
    mesh, and `field_frac` is the multiple of its live MEDIAN that counts as qualifying. Death stops
    needing a new branch for every new measurement."

    `compare: high | low` is a VALUE, not two models: which tail of one distribution qualifies is a
    threshold direction, not a different claim about the tissue.
    """
    DEATH = "field"


@register_operator("cell_divide", model="doubler", set="vertex", kind="structural", family="population")
class Divide3DDoubler(Divide3D):
    """Divide at `factor` x THIS CELL'S OWN BIRTH VOLUME -- the rule that was the default until
    kept because it is the null the absolute-size rule has to beat, and because a large part of
    this project's recorded results were measured under it.

    Under exponential growth this is a TIMER wearing a sizer's clothes: doubling from any birth
    volume takes the same time, so it never consults size in any way that could correct one. The
    review is direct about the consequence: size disparities are amplified, not constrained.
    """
    MECHANISM_TAGS = ["division", "volume_doubling", "relative_threshold", "no_size_control"]

    def _trigger(self, v_now, v_birth, jit, age, v_ref):
        return v_now >= self.factor * jit * v_birth


@register_operator("cell_divide", model="timer", set="vertex", kind="structural", family="population")
class Divide3DTimer(Divide3D):
    """Divide on the CLOCK: `age >= cycle * jit` division-calls since birth, size ignored entirely.

    Alone this is the worst of the three -- a cell divides whether or not it has grown, so size
    variance is set by whatever growth did in the interval and nothing corrects it. It exists to
    be paired with `cell_grow model: timer`, which sets the growth rate from the size deficit so the
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


@register_operator("topo_record", set="vertex", kind="structural", family="harness")
class TopoSnapshot3D(Structural):
    """A measurement, as an operator: record the current mesh once per frame, so a vesicle whose
    topology changes can be rendered frame by frame.

    vertex -> vertex: reads the half-edge table and the vertex count, writes nothing; appends one
    row per frame to the recording.

    It is needed because the mesh is not fixed. A renderer handed only the final topology would
    draw every earlier frame with faces that did not exist yet, and one handed only the first would
    lose every division. What is recorded is the flat half-edge table, which is enough to
    reconstruct the surface at that instant and nothing more.

    Reference: none -- this is a recording, not a mechanism. Plexus (this work).
    """
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
            # `age` = division-calls since this cell was born (cell_divide resets it to 0 on
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
            # THE RESERVOIR, PER FRAME. cell_divide sets these on the mesh and nothing carried them
            # into the history, so run_one read them and always found nothing -- a run that
            # plateaued at 98.5% of its array reported buf_full False. The flag existed, the
            # counter existed, and the one structure anybody reads afterwards did not have them.
            div_blocked=int(m.get("div_blocked") or 0),
            # CUMULATIVE DEATHS, for the same reason div_blocked is here: cell_die counts
            # every extrusion on the mesh and nothing carried it into the history, so the only
            # visible trace of a death was the cell COUNT -- which cannot distinguish "nothing
            # died" from "deaths were masked by divisions". A run whose count grows and a run
            # whose count holds can both have death running, and the count alone cannot say which
            # is which.
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


def _vertex_faces(rings):
    """vertex -> set of faces incident to it (used to find the third face at u / at v)."""
    vf = {}
    for f, r in enumerate(rings):
        if r is None or len(r) < 3:
            continue
        for w in r:
            vf.setdefault(w, set()).add(f)
    return vf


def _insert_before(ring, anchor, w):
    """New ring with w inserted immediately BEFORE the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        if x == anchor and not done:
            out.append(w); done = True
        out.append(x)
    return out if done else None


def _insert_after(ring, anchor, w):
    """New ring with w inserted immediately AFTER the first occurrence of anchor (or None)."""
    out, done = [], False
    for x in ring:
        out.append(x)
        if x == anchor and not done:
            out.append(w); done = True
    return out if done else None


def _ring_ok(r):
    return r is not None and len(r) >= 3 and len(set(r)) == len(r)


# --------------------------------------------------------------------------------------------------
#  local closed-surface (manifold) check on the 4 changed faces
# --------------------------------------------------------------------------------------------------
def _boundary_de(faces_map):
    """Directed edges of a face patch; returns the set of BOUNDARY directed edges (those whose reverse
    is not in the patch), or None if any directed edge repeats inside the patch (-> non-manifold)."""
    de = set()
    for r in faces_map.values():
        k = len(r)
        for i in range(k):
            e = (r[i], r[(i + 1) % k])
            if e in de:
                return None                                  # directed edge used twice -> non-manifold
            de.add(e)
    return set(e for e in de if (e[1], e[0]) not in de)


def _local_manifold_ok(old_map, new_map):
    """A T1 only touches the four faces {A,B,C,D}; every changed directed edge stays inside that patch.
    So the mesh stays closed iff the patch mates with the (unchanged) exterior exactly as before, i.e.
    the patch's boundary directed edges are unchanged -- and no directed edge repeats in the new patch."""
    b_old = _boundary_de(old_map); b_new = _boundary_de(new_map)
    return b_new is not None and b_old is not None and b_new == b_old


# --------------------------------------------------------------------------------------------------
#  3D per-face validity: non-degenerate, outward-facing, and SIMPLE (no self-crossing)
# --------------------------------------------------------------------------------------------------
def _seg_cross(p1, p2, p3, p4):
    """Proper 2D segment intersection (touching endpoints do NOT count)."""
    ccw = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1); d2 = ccw(p3, p4, p2); d3 = ccw(p1, p2, p3); d4 = ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _polygon_simple_2d(Q):
    """True iff the 2D polygon Q has no two NON-adjacent edges properly crossing (catches bow-ties)."""
    k = len(Q)
    for i in range(k):
        for j in range(i + 1, k):
            if j == i + 1 or (i == 0 and j == k - 1):
                continue                                     # adjacent edges share a vertex
            if _seg_cross(Q[i], Q[(i + 1) % k], Q[j], Q[(j + 1) % k]):
                return False
    return True


def _face_ok_3d(ring, getp):
    """Face is valid: >=3 verts, non-zero area, Newell normal points OUTWARD (dot with centroid > 0),
    and the polygon is SIMPLE when projected onto its own plane. `getp(i)` returns vertex i's 3-vector."""
    P = np.array([np.asarray(getp(i), float) for i in ring])
    k = len(P)
    if k < 3:
        return False
    c = P.mean(0)
    N = 0.5 * np.cross(P, np.roll(P, -1, 0)).sum(0)          # Newell area vector (|N| = area)
    a = np.linalg.norm(N)
    if a < 1e-9 or float(np.dot(N, c)) <= 0.0:               # degenerate, or inward-facing
        return False
    n = N / a                                                # project to the face plane, test simplicity
    e1 = P[0] - c; e1 = e1 - np.dot(e1, n) * n
    if np.linalg.norm(e1) < 1e-9:
        e1 = P[1] - c; e1 = e1 - np.dot(e1, n) * n
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    Q = np.stack([(P - c) @ e1, (P - c) @ e2], 1)
    return _polygon_simple_2d(Q)


# --------------------------------------------------------------------------------------------------
#  the T1 flip
# --------------------------------------------------------------------------------------------------
# Per frame: (frame, flips this frame, cumulative flips, live faces). The RATE per cell per frame is
# what discriminates a homogenising (length-keyed) myosin feedback from a destabilising (tension-keyed)
# one; junction-length CV moves the same way under both.
T1_TRACE: list = []


def t1_flip_3d(rings, pos, e_uv, new_len=None, emap=None, vf=None, plane_axis=None):
    """One surface T1 on interior edge e_uv=(u,v): rewire the four rings A,B,C,D and move u,v apart
    along the tangent-plane perpendicular (projected onto the shell). Mutates `rings` and pos[u],pos[v]
    in place and returns (u,v) on success; returns None (no-op) if the flip is impossible or would break
    an invariant: boundary edge, non-trivalent u/v, C==D, a face < 3 verts, a duplicated vertex in a
    ring, a broken closed-surface, or a non-simple/inward face for BOTH chiralities x BOTH signs.
    OPTIMISATION: pass `emap` (edge->face) and `vf` (vertex->faces) built ONCE by the caller; a successful
    flip updates them in place (only the 4 faces A,B,C,D change), avoiding an O(F) rebuild per candidate."""
    u, v = int(e_uv[0]), int(e_uv[1])
    if emap is None:
        emap = _edge_face_map(rings)
    A = emap.get((u, v)); B = emap.get((v, u))
    if A is None or B is None or A == B:
        return None                                          # boundary / degenerate interior edge
    if vf is None:
        vf = _vertex_faces(rings)
    Cs = vf.get(u, set()) - {A, B}; Ds = vf.get(v, set()) - {A, B}
    if len(Cs) != 1 or len(Ds) != 1:
        return None                                          # u or v not trivalent -> no clean T1
    C = Cs.pop(); D = Ds.pop()
    if C == D or len({A, B, C, D}) != 4:
        return None                                          # the two "far" cells must be distinct
    rA, rB, rC, rD = rings[A], rings[B], rings[C], rings[D]

    nA = [w for w in rA if w != v]                           # A loses v
    nB = [w for w in rB if w != u]                           # B loses u
    if not (_ring_ok(nA) and _ring_ok(nB)):
        return None                                          # would leave a face with < 3 verts

    # geometry: collapse u,v to their midpoint then reopen perpendicular in the local tangent plane
    pu = np.asarray(pos[u], float); pv = np.asarray(pos[v], float)
    mid = 0.5 * (pu + pv); rmid = np.linalg.norm(mid)
    d = pv - pu; L = np.linalg.norm(d)
    if L < 1e-12 or (plane_axis is None and rmid < 1e-9):
        return None
    # THE SHELL NORMAL, AND ON A SHEET IT IS NOT THE RADIAL. For a closed vesicle the outward
    # radial from the world origin is the surface normal, which is what this assumed. On a FLAT
    # patch it is not: a sheet at height 5 has a radial that tilts away from +z everywhere except
    # the axis, so reopening the junction perpendicular to it, and then re-projecting both new
    # vertices onto a sphere below, lifts them off the plane at every flip. Ablation measured it
    # exactly: with `edge_flip` removed the sheet stayed at sd(z) = 0.00e+00, with it 4.23% of an
    # edge -- the whole residual, and none of it from division or from the relaxation.
    if plane_axis is None:
        n = mid / rmid                                       # outward radial (shell normal)
    else:
        n = np.zeros(3); n[int(plane_axis)] = 1.0            # the sheet's own normal
    perp = np.cross(n, d / L); pn = np.linalg.norm(perp)
    if pn < 1e-9:                                            # edge is radial -> perpendicular undefined
        return None
    perp = perp / pn
    half = 0.5 * (new_len if new_len is not None else L)     # reopen at ~ new_len (default: same length)
    rm = 0.5 * (np.linalg.norm(pu) + np.linalg.norm(pv))     # target shell radius for both split verts

    old_map = {A: rA, B: rB, C: rC, D: rD}
    #   two chiralities: insert v/u BEFORE (v->u in C) or AFTER (u->v in C) -- both are closed T1s;
    #   geometry (simple + outward) picks the correct, non-folding one.
    for insert in (_insert_before, _insert_after):
        nC = insert(rC, u, v)                                # C gains v next to u
        nD = insert(rD, v, u)                                # D gains u next to v
        if not (_ring_ok(nC) and _ring_ok(nD)):
            continue
        new_map = {A: nA, B: nB, C: nC, D: nD}
        if not _local_manifold_ok(old_map, new_map):
            continue                                         # would break the closed surface
        for sign in (+1.0, -1.0):
            nu = mid - sign * perp * half; nv = mid + sign * perp * half
            if plane_axis is None:
                nu = nu * (rm / (np.linalg.norm(nu) + 1e-12))  # back onto the shell
                nv = nv * (rm / (np.linalg.norm(nv) + 1e-12))
            else:                                              # back onto the SHEET
                nu[int(plane_axis)] = mid[int(plane_axis)]
                nv[int(plane_axis)] = mid[int(plane_axis)]
            getp = lambda i: (nu if i == u else nv if i == v else pos[i])
            if all(_face_ok_3d(r, getp) for r in (nA, nB, nC, nD)):
                for fid, ro, rn in ((A, rA, nA), (B, rB, nB), (C, rC, nC), (D, rD, nD)):
                    for i in range(len(ro)):                # keep the passed maps in sync: only A,B,C,D changed
                        emap.pop((ro[i], ro[(i + 1) % len(ro)]), None)
                    for i in range(len(rn)):
                        emap[(rn[i], rn[(i + 1) % len(rn)])] = fid
                    for w in ro:
                        s = vf.get(w)
                        if s is not None:
                            s.discard(fid)
                    for w in rn:
                        vf.setdefault(w, set()).add(fid)
                rings[A] = nA; rings[B] = nB; rings[C] = nC; rings[D] = nD
                pos[u] = nu; pos[v] = nv
                return (u, v)
    return None


# --------------------------------------------------------------------------------------------------
#  plexus operator
# --------------------------------------------------------------------------------------------------
@register_operator("edge_flip", set="vertex", kind="rewire", family="topology")
class ReconnectT1_3D(Rewire):
    """The T1 transition: a reversible neighbour exchange, and the ingredient that makes this a
    dynamic vertex model rather than a repeatedly re-tessellated one.

    vertex -> vertex: reads pos, flips short interior edges and repositions their two vertices.

        flip edge e when   l_e  <  l_T1

    l_T1 is the threshold length in world units. The edge is removed and reconnected across the
    other diagonal of the two faces sharing it, so two cells that were neighbours cease to be and
    two that were not become so. The two vertices are repositioned to give the new edge a length
    just above the threshold, otherwise it would flip straight back.

    It preserves the counts of vertices, edges and faces, and therefore the Euler characteristic:
    nothing is created or destroyed, only reconnected. That is what distinguishes it from division
    and death, and it is why a flip that would produce an invalid configuration is simply not
    committed rather than repaired afterwards.

    A route that re-tessellates the surface each frame cannot have this operator, because it never
    represents the intermediate: intercalation is then an artefact of re-meshing rather than a
    mechanism with a threshold anyone can set.

    Reference: Okuda, S., Inoue, Y., Eiraku, M., Sasai, Y. & Adachi, T. (2013). Reversible network
    reconnection model for simulating large deformation in dynamic tissue morphogenesis. Biomech.
    Model. Mechanobiol. 12(4):627-644.
    """
    SUPPORTED_DIMS = [3]; DIFFERENTIABLE = False; MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["T1_transition", "reversible_network_reconnection", "intercalation",
                      "neighbour_exchange", "vertex_model", "vesicle", "surface"]
    PARAM_ROLES = {"l_th": "reconnection_threshold_length (absolute; 0 -> use l_th_frac)",
                   "l_th_frac": "threshold as fraction of the mean edge length",
                   "max_flips": "cap on reconnections per call", "every": "call period (ticks)"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.l_th = float(params.get("l_th", 0.0))           # absolute; <=0 -> l_th_frac x mean edge
        self.l_th_frac = float(params.get("l_th_frac", 0.15))
        self.max_flips = int(params.get("max_flips", 20))
        # (was `from mesh_ops import _engine_owns_clock`) -- same module now
        self.every = _engine_owns_clock(params); self._k = 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at); m = getattr(lvl, "_mesh", None)
        if m is None:
            return {}
        self._k += 1                    # monotonic tick only -- D1: the engine owns the period
        dev = lvl.state.device; dt = lvl.state.dtype
        Nv = int(m["Nv"])
        pos_np = lvl.get("pos")[:Nv].detach().cpu().numpy().astype(np.float64)
        es = m["E_srce"].detach().cpu().numpy(); et = m["E_trgt"].detach().cpu().numpy()
        ef = m["E_face"].detach().cpu().numpy(); nF = int(m["nF"])
        rings = rings_from_flat_3d(es, et, ef, nF)
        # READ OFF `mech`, NOT DECLARED AGAIN HERE. `cell_mechanics` stashes the constraint it is
        # solving under, so a spec says `plane_axis` once and every topology operator that moves a
        # vertex honours the same plane. Repeating it per operator is how three of them end up
        # disagreeing about which surface the tissue lives on.
        _plane = (m.get("mech") or {}).get("plane_axis")
        emap = _edge_face_map(rings); vf = _vertex_faces(rings)   # build the adjacency maps ONCE; t1_flip_3d
        #   updates them incrementally on each successful flip (was rebuilt O(F) per candidate = the hot spot)
        pos = [p.copy() for p in pos_np]
        length = np.linalg.norm(pos_np[et] - pos_np[es], axis=1)
        thr = self.l_th if self.l_th > 0 else self.l_th_frac * float(length.mean())
        order = np.argsort(length)                           # shortest interior edges first
        seen = set(); used = set(); ndone = 0                # `used` verts -> one flip per vertex / call
        for k in order:
            if ndone >= self.max_flips or length[k] >= thr:
                break
            a, b = int(es[k]), int(et[k]); key = (min(a, b), max(a, b))
            if key in seen or a in used or b in used:
                continue
            seen.add(key)
            if t1_flip_3d(rings, pos, (a, b), new_len=thr, emap=emap, vf=vf,
                          plane_axis=_plane) is not None:
                used.add(a); used.add(b); ndone += 1
        # TRACED PER FRAME, because the RATE is the observable that separates a length-keyed myosin
        # feedback from a tension-keyed one. A length feedback homogenises junction lengths and should
        # SUPPRESS T1s; a tension feedback is the destabilising one the germband-extension literature
        # describes and should PRODUCE them. Both move junction-length CV the same way, so CV cannot
        # tell them apart -- this can. `n_t1` was already accumulated; only the per-frame series was
        # missing, and without it the discriminating measurement was unavailable.
        T1_TRACE.append((int(getattr(H, "frame", -1) or -1), ndone, int(m.get("n_t1", 0)) + ndone,
                         int(nF)))
        if ndone == 0:
            return {}
        es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
        # "T1 DROPS NO FACE" WAS AN ASSUMPTION, NOT AN INVARIANT, and it held only while nothing
        # shrank a cell to nothing. `flat_from_rings_3d` discards any ring that falls below three
        # vertices; a flip on a cell that is already a triangle can do exactly that. This line
        # threw `keep` away, so when it happened T1 lost a face and left EVERY per-face array --
        # A0, P0, V0f, Vbirth, divjit, age, ndiv, alive -- shifted by one against the mesh.
        #
        # Found by cell_die, which is the first operator that shrinks a cell far enough to
        # reach the case: marking ONE cell for death produced three deaths at ticks 86, 164 and
        # 236, each reporting exactly one marked cell, because the shift kept handing the flag to
        # a new victim. The control with T1 and no apoptosis holds at 2000 cells, which is why
        # this went eighteen rounds unseen.
        if nF2 != nF:
            print(f"[edge_flip] a flip left {nF - nF2} face(s) below three sides -- "
                  f"reindexing {nF} -> {nF2}. Per-face targets follow the mesh; a cell was lost "
                  f"to topology, not to biology.", flush=True)
            for _nm in ("A0", "P0", "V0f", "Vbirth", "divjit", "age", "ndiv", "alive"):
                if _nm in m:
                    _a = m[_nm].detach().cpu().numpy()
                    m[_nm] = torch.as_tensor(np.asarray([_a[i] for i in keep]),
                                             dtype=m[_nm].dtype, device=m[_nm].device)
            if isinstance(m.get("apop_flag"), np.ndarray):
                m["apop_flag"] = np.asarray([m["apop_flag"][i] for i in keep], np.float64)
            # AND THE OPEN NAMES, which this branch has never carried. The tuple above is the
            # CLOSED list every topology operator knows; `face_carry` is the open one an operator
            # declares for itself, and `cell_divide` and `cell_die` have routed it through
            # `_carry_face_state` since it existed. This one did not, so a flip that lost a face
            # left `medioapical_myosin`'s per-face density indexed against faces that had moved --
            # the same defect class as per-half-edge myosin before `junction_sync`, one level up.
            #
            # UNREACHABLE IN THE GATED RUN, and asserted so: this branch needs `nF2 != nF`, which
            # `_ring_ok` refuses to produce (it rejects any new ring below three sides). The fix is
            # covered by `tools/test_mesh_carry.py` instead, which constructs the condition.
            _carry_face_state(m, keep, m["A0"].dtype if "A0" in m else torch.float32,
                                        m["E_srce"].device)
            # THE CELL STATE AND THE PENDING DELTAS FOLLOW TOO, and leaving them behind is the
            # same defect cell_die had: `chem` is indexed by face, and the engine zeroes its
            # delta accumulator once per TICK and integrates at the END of the schedule, so
            # cell_chem_diffuse and cell_chem_react deposit per-cell deltas that are applied after this runs.
            # Renumber the faces without renumbering those and every activator value, and every
            # pending flux, lands on a different cell.
            #
            # It stayed hidden because a flip only drops a face when a cell is ALREADY a triangle,
            # which needs something to have shrunk it -- so nothing reached this branch until
            # apoptosis existed. It then fails immediately and enormously -- the activator goes
            # many orders of magnitude negative when division and death run in the same tick,
            # where the same operator on a non-growing tissue holds it at exactly zero.
            # edge_flip declares no `cell_set`, so the cell level is looked up by name and
            # a model without one simply skips this -- H.level raises rather than returning None.
            # ONE ENGINE CALL, for the same reason `cell_die` makes it: state, occupancy, the
            # coordinate delta accumulator and the extra first-order block deltas are four stores
            # the engine keeps per set, and a renumber that forgets one scrambles it silently. The
            # version this replaces forgot the fourth -- its guard tested `isinstance(key, tuple)`
            # while `_delta_blocks` is keyed by level NAME, so the branch could never run.
            #
            # `edge_flip` declares no `cell_set`, so the cell level is looked up by name and a model
            # without one simply skips: `renumber_set` returns False rather than raising.
            _cat = getattr(self, "cat", None) or "cell"
            if not H.renumber_set(_cat, keep, n_new=nF2):
                m["renumber_failed"] = int(m.get("renumber_failed", 0)) + 1
                print(f"[edge_flip] renumber_set({_cat!r}) DID NOT ACT after a face drop -- the "
                      f"cell state is now mis-indexed against the mesh", flush=True)
        m["E_srce"] = torch.as_tensor(es2, device=dev)
        m["E_trgt"] = torch.as_tensor(et2, device=dev)
        m["E_face"] = torch.as_tensor(ef2, device=dev)
        m["nF"] = nF2
        px0, px1 = lvl.state_schema["pos"]                   # write the two moved verts back (like cell_divide)
        st = lvl.state.clone()
        st[:Nv, px0:px1] = torch.as_tensor(np.asarray(pos), dtype=dt, device=dev)
        lvl.state = st
        m["n_t1"] = int(m.get("n_t1", 0)) + ndone
        return {}


# --------------------------------------------------------------------------------------------------
#  standalone self-test (no engine): build a jittered vesicle, run many T1 flips, assert it stays
#  CLOSED with Euler=2 and that a T1 keeps V,E,F constant.
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # (was `from mesh_ops import build_sphere_mesh`) -- same module now

    verts, es, et, ef, nF = build_sphere_mesh(150, 5.0, 0.15, 0)
    rings = rings_from_flat_3d(es, et, ef, nF)
    pos = [v.copy() for v in verts]
    ok0, V0, E0, F0, eu0 = _check_closed(rings)
    print(f"start:  closed={ok0} V={V0} E={E0} F={F0} euler={eu0}")

    def undirected_edges(rr):
        s = set()
        for r in rr:
            if r is None or len(r) < 3:
                continue
            k = len(r)
            for i in range(k):
                a, b = r[i], r[(i + 1) % k]; s.add((min(a, b), max(a, b)))
        return s

    ndone = 0
    for sweep in range(6):                                   # several sweeps: each flip reopens the edge
        E = sorted(undirected_edges(rings),                  #   at ~mean length so it exits the short set
                   key=lambda ab: np.linalg.norm(pos[ab[1]] - pos[ab[0]]))
        me = float(np.mean([np.linalg.norm(pos[b] - pos[a]) for a, b in E]))
        thr = 0.9 * me                                       # target the shorter ~half of the edges
        used = set(); fired = 0
        for (a, b) in E:
            if np.linalg.norm(pos[b] - pos[a]) >= thr:
                break
            if a in used or b in used:
                continue
            if t1_flip_3d(rings, pos, (a, b), new_len=me) is not None:  # reopen at ~mean length
                used.add(a); used.add(b); fired += 1
        ndone += fired
        ok, V, E_, F, eu = _check_closed(rings)
        print(f"sweep {sweep}: fired {fired:3d}  closed={ok} V={V} E={E_} F={F} euler={eu}")
        assert ok and eu == 2, "mesh broke the closed-surface invariant"

    ok, V, E, F, eu = _check_closed(rings)
    print(f"after {ndone} T1 flips:  closed={ok} V={V} E={E} F={F} euler={eu}  (want euler=2)")
    assert ok and eu == 2
    assert (V, E, F) == (V0, E0, F0), "a T1 must keep V,E,F constant"    # dV=dE=dF=0
    es2, et2, ef2, nF2, keep = flat_from_rings_3d(rings)
    print(f"rebuilt flat: nF={nF2} half-edges={len(es2)} (each real edge twice -> {len(es2)//2} edges)")
    assert ndone >= 20, "expected a meaningful number of flips"
    print(f"SELF-TEST OK  ({ndone} flips, closed, euler=2, V/E/F unchanged)")


def apical_basal_shells(pos, es, et, ef, nF, h_cell):
    """Apical (outer) and basal (inner) vertex positions a_i, b_i = x_i +/- (H_i/2) n_i, for RENDERING
    the monolayer as two offset shells with a visible thickness. Same offset the energy uses."""
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(s, t, dim=-1))
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    cnt = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, torch.ones(es.shape[0], device=dev, dtype=dt))
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef]) / cnt.clamp(min=1e-9)
    return pos + 0.5 * hv[:, None] * n, pos - 0.5 * hv[:, None] * n


def monolayer_shells(pos, es, et, ef, nF, h_cell, eocc=None):
    """The apical and basal surfaces of a monolayer, from its MID-surface and a per-cell thickness.

    ONE DEFINITION, TWO CALLERS. `monolayer_geometry_3d` needs these to build the prism the energy
    is written on, and the renderer needs them to draw the epithelium as something with a thickness
    rather than as the mid-surface it stores. Two copies of "offset along the vertex normal by h/2"
    would agree until one of them changed, and the picture would then stop being a picture of the
    model -- so the formula lives here and both call it.

    The normal at a vertex is the normalised sum of the area vectors of its incident faces, and the
    thickness at a vertex is the mean over its incident cells; the shell straddles the mid-surface
    symmetrically, apical OUTWARD. Returns (apical, basal, vertex_normal, vertex_thickness).
    """
    dev, dt = pos.device, pos.dtype
    Nv = pos.shape[0]
    s, t = pos[es], pos[et]
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    # mid-surface face area vectors -> vertex normals (sum of incident face area vectors)
    crossm = torch.cross(s, t, dim=-1) * ones_e[:, None]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, crossm)
    vn = torch.zeros(Nv, 3, device=dev, dtype=dt).index_add(0, es, Nf[ef] * ones_e[:, None])
    n = vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)
    # thickness at a vertex = mean thickness of incident cells
    cnt_v = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, ones_e)
    hv = torch.zeros(Nv, device=dev, dtype=dt).index_add(0, es, h_cell[ef] * ones_e) / cnt_v.clamp(min=1e-9)
    return pos + 0.5 * hv[:, None] * n, pos - 0.5 * hv[:, None] * n, n, hv


def monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc=None):
    """Per-cell prism volume v_f and surface s_f (apical+basal+lateral), plus the apical/basal cap areas.
    All differentiable in `pos`. h_cell is per-cell thickness [nF]. eocc masks dead half-edges (or None)."""
    dev, dt = pos.device, pos.dtype
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    crossm = torch.cross(pos[es], pos[et], dim=-1) * ones_e[:, None]
    Nf = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, crossm)
    a, b, _, _ = monolayer_shells(pos, es, et, ef, nF, h_cell, eocc)
    a_s, a_t, b_s, b_t = a[es], a[et], b[es], b[et]
    # cell VOLUME = mid-surface area x thickness (v_j = A_mid*h_j). Exact for a flat cell, first-order in
    # curvature (the O((h/R)^2) prism correction ~0.3% is dropped); ALWAYS positive & differentiable, and
    # -- the key point -- bending resistance comes from the SURFACE term below (apical!=basal area under
    # curvature), NOT from the volume, so A_mid*h is the physically correct choice, not just the simple one.
    v_f = Nf.norm(dim=-1) * h_cell
    # apical / basal cap areas (Newell magnitude, origin-independent)
    Na = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(a_s, a_t, dim=-1) * ones_e[:, None])
    Nb = 0.5 * torch.zeros(nF, 3, device=dev, dtype=dt).index_add(0, ef, torch.cross(b_s, b_t, dim=-1) * ones_e[:, None])
    A_ap, A_ba = Na.norm(dim=-1), Nb.norm(dim=-1)
    # lateral quad area per edge = tri(a_s,a_t,b_t) + tri(a_s,b_t,b_s)
    la = (0.5 * torch.cross(a_t - a_s, b_t - a_s, dim=-1).norm(dim=-1)
          + 0.5 * torch.cross(b_t - a_s, b_s - a_s, dim=-1).norm(dim=-1)) * ones_e
    A_lat = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, la)
    s_f = A_ap + A_ba + A_lat
    return v_f, s_f, A_ap, A_ba


def apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc=None):
    """Per-cell TRUE polyhedron volume `v_f` and surface `s_f` from the DOUBLED degree-of-freedom set.

    The twin of `monolayer_geometry_3d`, and the difference between them is the whole promotion.
    There the cell's two surfaces are `pos +/- (h/2) n` -- a kinematic identity that ties them to the
    mid-surface and to each other -- and the volume is `A_mid * h`, the flat-cell answer with the
    `O((h/R)^2)` prism correction dropped. Here apical `= pos + sep` and basal `= pos - sep` with
    `sep` a free per-vertex vector, and the volume is the volume OF THE POLYHEDRON:

        V_f = (1/6) sum over the cell's triangles of (p-o) . ((q-o) x (r-o))     [divergence theorem]

          apical cap    the ring on `a`, fanned from the cap's own centroid, outward winding
          basal cap     the ring on `b`, REVERSED, because it faces the other way
          lateral wall  one quad per ring edge, (a_s, b_s, b_t) + (a_s, b_t, a_t)

    THE WINDING IS THE SAME AS `gate_measures._cell_polyhedron_volume`, term for term, and that is
    deliberate: a gate row that disagreed with the energy would be measuring a different solid than
    the one the tissue relaxed. The measure was written first and caught its own wall wound INWARD
    -- a hexagonal prism of +2.598 reading -0.866, with the closure row still green, because an
    inward-wound wall is still closed. Both now read (a_s, b_s, b_t), (a_s, b_t, a_t).

    `o` IS THE CELL'S OWN MID-RING CENTROID, NOT THE WORLD ORIGIN. The volume of a closed surface is
    origin-independent in exact arithmetic and this state is float32, so the choice is a numerical
    one and it is the load-bearing kind: about the world origin every term is `O(R^3)` and they
    cancel down to `O(h*A)`, losing three or four significant figures on a shell at R = 5. About the
    cell's own centroid every term is already the size of the answer.

    Returns (v_f, s_f, A_ap, A_ba). All differentiable in BOTH `pos` and `sep`.
    """
    dev, dt = pos.device, pos.dtype
    ones_e = torch.ones(es.shape[0], device=dev, dtype=dt) if eocc is None else eocc
    a, b = pos + sep, pos - sep
    a_s, a_t, b_s, b_t = a[es], a[et], b[es], b[et]
    z3 = lambda: torch.zeros(nF, 3, device=dev, dtype=dt)                        # noqa: E731
    cnt = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, ones_e).clamp(min=1e-9)
    cen = z3().index_add(0, ef, pos[es] * ones_e[:, None]) / cnt[:, None]        # mid-ring centroid
    ca = z3().index_add(0, ef, a_s * ones_e[:, None]) / cnt[:, None]             # apical cap centroid
    cb = z3().index_add(0, ef, b_s * ones_e[:, None]) / cnt[:, None]             # basal cap centroid
    o = cen[ef]

    def tri(p, q, r):
        return torch.einsum("ij,ij->i", p - o, torch.cross(q - o, r - o, dim=-1)) * ones_e

    v6 = torch.zeros(nF, device=dev, dtype=dt)
    v6 = v6.index_add(0, ef, tri(ca[ef], a_s, a_t))                              # apical cap
    v6 = v6.index_add(0, ef, tri(cb[ef], b_t, b_s))                              # basal cap, reversed
    v6 = v6.index_add(0, ef, tri(a_s, b_s, b_t))                                 # wall, triangle 1
    v6 = v6.index_add(0, ef, tri(a_s, b_t, a_t))                                 # wall, triangle 2
    v_f = v6 / 6.0
    # THE CAP'S AREA IS MEASURED ON THE SURFACE ITS VOLUME IS MEASURED ON -- the SAME centroid fan,
    # summed as true triangle areas. It used to be the Newell magnitude ||1/2 sum a_s x a_t||, chosen
    # so that on a right prism the caps were the same arithmetic as the monolayer's and AB-C1 was an
    # identity rather than a near miss. IT BOUGHT THAT IDENTITY WITH A NULL SPACE, and the null space
    # is the whole failure mode of the free separation.
    #
    # The Newell magnitude is the area of the ring's PLANAR PROJECTION, so a crumpled cap and a flat
    # one of the same outline measure the SAME, and the area is stationary to first order when one
    # vertex moves normal to the cap. The volume, meanwhile, was already the fan -- so the energy was
    # pricing one surface and enclosing another. With nothing but the lateral wall resisting, `sep`
    # relaxes downhill into a checkerboard: measured on `gate_ab_thickshell` extended to 80 frames
    # at sep_mu 1, neighbouring vertices anti-correlate across an edge (Pearson r -0.42 by frame 20),
    # the thickness varies five times more WITHIN a cell's ring (cv 1.19) than between cells (0.24),
    # the median thickness collapses 1.428 -> 0.237 while a tail reaches 7.47, and spans start
    # inverting at frame 40. Quartering the step to eta 0.02 over 120 iterations reproduces it and
    # collapses slightly faster, so it is this energy's own descent direction and not the stepping.
    #
    # THE IDENTITIES SURVIVE, WHICH IS WHY THIS IS A FIX AND NOT A CHANGE OF MODEL. For a planar
    # convex ring fanned from its own centroid the triangle areas sum to the polygon area exactly, so
    # AB-C1 (flat patch, caps exactly planar) and AB-C2 (regular hexagonal prism) are unchanged to
    # the last bit. What moves is any curved shell, where the ring is not planar and the cap's true
    # area is the larger of the two -- which is the case the null space lived in.
    A_ap = torch.zeros(nF, device=dev, dtype=dt).index_add(
        0, ef, 0.5 * torch.cross(a_s - ca[ef], a_t - ca[ef], dim=-1).norm(dim=-1) * ones_e)
    A_ba = torch.zeros(nF, device=dev, dtype=dt).index_add(
        0, ef, 0.5 * torch.cross(b_s - cb[ef], b_t - cb[ef], dim=-1).norm(dim=-1) * ones_e)
    la = (0.5 * torch.cross(a_t - a_s, b_t - a_s, dim=-1).norm(dim=-1)
          + 0.5 * torch.cross(b_t - a_s, b_s - a_s, dim=-1).norm(dim=-1)) * ones_e
    A_lat = torch.zeros(nF, device=dev, dtype=dt).index_add(0, ef, la)
    return v_f, A_ap + A_ba + A_lat, A_ap, A_ba


def _apicobasal_energy_core(pos, sep, es, et, ef, nF, V_eq, alive, k_v, kappa_s, Lam, K_R, R0,
                            eocc, vocc, gamma=0.0, surface="apical"):
    """The monolayer's energy on the polyhedron: same functional, different geometry.

        U = sum_j [ 1/2 k_v (V_j - V_eq_j)^2 + kappa_s S_j + 1/2 gamma P_j^2 ]
            + Lam sum_e l_e + K_R sum_i (|x_i| - R0)^2

    THE FUNCTIONAL IS UNCHANGED ON PURPOSE. The hypothesis under test is that a cell is a polyhedron
    with two independent caps, not that it obeys some new constitutive law -- so every coefficient
    keeps its meaning and a spec can be moved between the two `model:` values without retuning. If
    this energy also changed form, a difference between the arms would have two possible causes and
    would settle neither.

    `surface:` -- WHICH RING THE PERIMETER, THE LINE TENSION AND THE RADIAL SPRING ACT ON, and the
    default is `apical` BECAUSE THE MID-SURFACE IS NOT A MEMBRANE. `pos` is a coordinate choice, not
    an object: the design sets apical = pos + sep and basal = pos - sep, so `pos` is IDENTICALLY the
    midpoint of the two caps (measured on gate_ab_curved: max |pos - (apical+basal)/2| = 0.000e+00
    over every vertex) and carries no information the caps do not. It is kept as the coordinate
    block for compatibility -- it is the surface the incumbent `monolayer` model integrates, which
    is what makes the R3 reduction a bit-level identity rather than a comparison -- and that is a
    statement about the SOLVER, not about the cell.

    So writing `P_j`, the line tension and the radial spring on it put three forces on the average
    of two membranes, which is an object no epithelium has: a junctional belt in this model ran
    along a surface that is not there. `surface: apical` puts them on a real one. `mid` is kept as
    a value so a run made before this change can be reproduced exactly, and `basal` because which
    surface a belt is on is a claim about the tissue and belongs in the spec.

    WHAT THIS DOES AND DOES NOT MOVE. On a FLAT patch with a uniform frozen `sep` the apical ring is
    the mid ring translated by one constant vector, so every edge length and every perimeter is
    unchanged and AB-C1 stays exact to the bit. On a curved shell the apical ring is larger by the
    offset, so any spec with a non-zero `gamma` or `Lambda` moves -- which is the point: it was
    being evaluated in the wrong place. `K_R` is a declared suppression either way
    (`graphs_data/mesh_mpm/README.md`) and every apicobasal gate runs it at 0.

    The volume and the total surface `S_j` are untouched: they are properties of the whole
    polyhedron and were never mid-surface quantities.
    """
    v_f, s_f, _, _ = apicobasal_geometry_3d(pos, sep, es, et, ef, nF, eocc)
    E = (0.5 * k_v * (v_f - V_eq) ** 2 * alive).sum() + kappa_s * (s_f * alive).sum()
    ring = {"apical": pos + sep, "basal": pos - sep, "mid": pos}[surface]
    if gamma != 0.0:
        perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(
            0, ef, (ring[et] - ring[es]).norm(dim=-1) * eocc)
        E = E + 0.5 * gamma * (perim ** 2 * alive).sum()
    if Lam != 0.0:
        E = E + Lam * ((ring[et] - ring[es]).norm(dim=-1) * eocc).sum()
    if K_R != 0.0:
        E = E + K_R * (((ring.norm(dim=1) - R0) ** 2) * vocc).sum()
    return E


def _monolayer_energy_core(pos, es, et, ef, nF, h_cell, V_eq, alive, k_v, kappa_s, Lam, K_R, R0, eocc, vocc, gamma=0.0):
    """U = sum_j [ 1/2 k_v (v_j - v_eq_j)^2 + kappa_s s_j + 1/2 gamma P_j^2 ] + Lam*sum_e l_e + K_R*sum_i (|x_i|-R0)^2 .
    gamma is a cortical CONTRACTILITY (perimeter^2) that rounds cells and resists shear -- a cell-shape
    regularizer standing in for the RNR/T1 remeshing Okuda relies on (without it the bare volume+surface
    energy shears/spikes under large deformation). Lam/K_R are optional dials (both default 0 in the op)."""
    v_f, s_f, _, _ = monolayer_geometry_3d(pos, es, et, ef, nF, h_cell, eocc)
    E = (0.5 * k_v * (v_f - V_eq) ** 2 * alive).sum() + kappa_s * (s_f * alive).sum()
    if gamma != 0.0:
        perim = torch.zeros(nF, device=pos.device, dtype=pos.dtype).index_add(0, ef, (pos[et] - pos[es]).norm(dim=-1) * eocc)
        E = E + 0.5 * gamma * (perim ** 2 * alive).sum()
    if Lam != 0.0:
        E = E + Lam * ((pos[et] - pos[es]).norm(dim=-1) * eocc).sum()
    if K_R != 0.0:
        E = E + K_R * (((pos.norm(dim=1) - R0) ** 2) * vocc).sum()
    return E


@register_operator("cell_mechanics", model="monolayer", set="vertex", kind="lateral", family="mechanics")
class MonolayerShapeEnergy3D(Lateral):
    """The MONOLAYER implementation of the cell_mechanics contract (plexus2 sec. 5: same biological
    operator -- the mechanical force that shapes the epithelial vesicle -- different NUMERICS). The
    default implementation is a mid-surface model with a lumen-wedge volume; this one gives every cell
    its OWN 3D volume + surface (apical+basal+lateral, Okuda Eq. 3): per-cell 3D volume elasticity +
    linear surface tension. Force = -grad U by one autograd pass; bounded overdamped Euler (displacement
    capped at cap_frac x mean edge). EMIT=velocity. Selected by {op: cell_mechanics, model:
    monolayer} -- `model:`, because giving every cell its own 3D volume is a different HYPOTHESIS
    about the tissue, not the same one computed differently; `implementation: monolayer` is refused
    by the schema, and this docstring said it for months. Emergent bending (thin undulate / thick straight) falls out of the vertex-normal offset;
    no explicit K_bend. See monolayer_design.md."""
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos"]; WRITES = ["pos"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "monolayer", "cell_3d_volume", "surface_tension", "emergent_bending", "force_balance"]
    REFERENCE = "Okuda, S. et al. (2018). Sci. Rep. 8:2386 (monolayer 3D vertex model, Eq. 3)."
    PARAM_ROLES = {"k_v": "cell_volume_elasticity", "kappa_s": "surface_tension", "h0": "cell_thickness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.k_v = float(params.get("k_v", 4.0)); self.kappa_s = float(params.get("kappa_s", 0.2))
        self.h0 = float(params.get("h0", 0.4))                    # uniform cell thickness (v1: fixed field)
        self.gamma = float(params.get("gamma", 0.0))             # cortical contractility (cell-shape regularizer)
        self.Lambda = float(params.get("Lambda", 0.0)); self.K_R = float(params.get("K_R", 0.0))
        self.mu = float(params.get("mu", 1.0)); self.dt = float(params.get("dt", 1.0))
        self.relax_iters = int(params.get("relax_iters", 30)); self.eta = float(params.get("eta", 0.08))
        self.cap_frac = float(params.get("cap_frac", 0.12))
        # REST STATE. "force_balance" makes the seeded vesicle an equilibrium; "volume_only" is the
        # original behaviour, kept so the collapse can be reproduced deliberately. See _rest_offset.
        self.rest_calibration = str(params.get("rest_calibration", "force_balance"))
        # THE VOLUME TARGET'S SCALE, DECLARABLE INSTEAD OF INFERRED. `V_eq = mono_k * V0f` converts
        # the wedge target `cell_grow` scales into this model's cell volume, and `mono_k` is
        # normally calibrated once from the seeded mesh -- median rest volume over median wedge.
        # That is right for a run and wrong for a CONTROLLED COMPARISON: two models that agree on
        # the volume to float32 still calibrate to constants differing in the last bits, and at the
        # seeded state `v - V_eq` is a difference of two nearly equal numbers, so those last bits
        # are a large RELATIVE perturbation of the volume force exactly where it is smallest.
        # Measured on gate_ab_flat: 3e-7 in `mono_k` opened the two arms to 1.1e-2 of an edge within
        # one frame, decaying back to 1.2e-4 once the cells had left their targets. Declaring it
        # holds the calibration fixed ACROSS arms, which is what makes AB-C1 a comparison of two
        # energies rather than of two calibrations. Default None: every existing spec calibrates
        # exactly as before.
        _mk = params.get("mono_k", None)
        self.mono_k = None if _mk is None else float(_mk)
        # THE SAME PLANAR CONSTRAINT THE DEFAULT IMPLEMENTATION TAKES. This class does not derive
        # from `ShapeEnergy3D` -- it is a different energy on the same contract -- so it parses
        # and applies `plane_axis` itself. Without this the key is accepted, ignored, and a flat
        # monolayer buckles for a reason that has nothing to do with its thickness.
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)

    def _rest_offset(self, x0, es, et, ef, nF, h, V_eq0, alive, R0t, eocc, vocc):
        """The constant to add to every cell's target volume so the SEEDED SHELL IS AT REST.

        WHY THIS EXISTS. The energy is  U = 1/2 k_v (v-V_eq)^2 + kappa_s s + 1/2 gamma P^2.  Only the
        first term can push outward; the other two always pull the surface in. Calibrating V_eq to
        the rest volume alone -- which is what the `mono_k` line below does -- balances the volume
        term against nothing, so a freshly seeded ball is NOT in equilibrium and collapses under its
        own tension. Measured: radius 5.00 -> 1.80 in 20 frames at the shipped settings, and still
        falling; with gamma=0 it is 2.95, with gamma=0 AND kappa_s=0 it is exactly 5.00. Runs that
        looked healthy avoided this only by loading a pre-relaxed checkpoint (round_40_mc8 starts at
        radius 6.14) rather than seeding, so the collapse was invisible for the whole campaign.

        WHY A CONSTANT, NOT A FACTOR. At force balance
            V_eq - v = (2 kappa_s + 1/2 gamma p0^2) / (k_v h0),
        because both tension terms contribute a SIZE-INDEPENDENT amount to dU/dA (P = p0 sqrt(A), so
        1/2 gamma P^2 = 1/2 gamma p0^2 A, whose derivative carries no A). The offset must therefore
        survive unchanged when a cell grows or divides -- a multiplicative correction would shrink
        with V0f and quietly reintroduce the collapse after the first division.

        WHY SOLVED, NOT TYPED IN. The loop sweeps k_v, kappa_s, gamma, h0, the seed radius and the
        cell count; a tension that happens to balance at one setting collapses at the next. The
        offset enters the force exactly linearly --
            g(delta) = g(0) - k_v delta grad(V_total)
        because U_vol is quadratic in (v - V_eq) -- so one gradient pair gives the exact root. What
        is zeroed is the RADIAL component: tangential forces are the cell-shape relaxation we want
        to keep, and only the radial resultant inflates or deflates the vesicle.

        WHY IT ITERATES. One solve is not enough in practice: it zeroes the radial force on the
        SEEDED mesh, but the first thing the relaxation does is even out cell shapes tangentially,
        and that moves the balance. Measured with a single solve, the defaults held to x1.014 but
        gamma=0 still drifted to x0.80 and gamma=0.3 to x1.10. So alternate -- relax the shape, put
        it back on the target sphere, re-solve -- until the shape stops changing. The rescaling is
        what pins the answer: it asks for the offset whose equilibrium radius IS the seed radius,
        rather than whatever radius the relaxation happens to wander to.
        """
        R_target = x0.norm(dim=1).mean().clamp(min=1e-9)
        delta = torch.zeros((), dtype=x0.dtype, device=x0.device)
        x = x0.clone()
        for _ in range(8):
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            u = x / x.norm(dim=1, keepdim=True).clamp(min=1e-9)         # outward radial direction
            g0 = self._grad(x, es, et, ef, nF, h, V_eq, alive, R0t, eocc, vocc)
            with torch.enable_grad():                                   # grad of TOTAL cell volume
                xg = x.detach().requires_grad_(True)
                vf, _, _, _ = monolayer_geometry_3d(xg, es, et, ef, nF, h, eocc)
                gV = torch.autograd.grad((vf * alive).sum(), xg)[0]
            den = self.k_v * (u * torch.nan_to_num(gV)).sum()
            if not torch.isfinite(den) or den.abs() < 1e-12:
                break
            step = ((u * g0).sum() / den).detach()
            if not torch.isfinite(step):
                break
            delta = delta + step
            # let the shape relax under the new offset, then put it back on the target sphere so the
            # next solve is asked about the radius we actually want
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            with torch.no_grad():
                cap = self.cap_frac * (x[et] - x[es]).norm(dim=-1).mean().clamp(min=1e-6)
            for _ in range(max(1, self.relax_iters)):
                s = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, h, V_eq, alive,
                                                       R0t, eocc, vocc)
                x = x + s * torch.clamp(cap / (s.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            x = x * (R_target / x.norm(dim=1).mean().clamp(min=1e-9))
        return delta.detach()

    def _grad(self, x, es, et, ef, nF, h, V_eq, alive, R0t, eocc, vocc):
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            E = _monolayer_energy_core(x, es, et, ef, nF, h, V_eq, alive, self.k_v, self.kappa_s,
                                       self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma)
            g = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(g)

    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        h_cell = torch.full((nF,), self.h0, dtype=dt, device=dev)   # v1: uniform fixed thickness
        # PUBLISHED FOR THE RENDERER. The mesh this operator works on is the MID-surface, so a
        # picture of it is a picture of a sheet with no thickness -- which is the one thing the
        # monolayer model adds. Recording the per-cell thickness (a per-face column, so it rides
        # `mesh_face_offsets` into the trajectory like `A0` does) lets the cross section rebuild
        # the apical and basal shells through `monolayer_shells`, the same function the energy
        # below is written on.
        m["mono_h"] = float(self.h0)
        # target monolayer volume: calibrate ONCE so V_eq matches the rest prism volume, then track the
        # growth op's scaling of the wedge target V0f (cell_grow scales V0f per cell) -> reuse it.
        v_rest, _, _, _ = monolayer_geometry_3d(x0, es, et, ef, nF, h_cell, eocc)
        if "mono_k" not in m:
            if self.mono_k is not None:
                m["mono_k"] = self.mono_k                     # declared: see __init__
            else:
                wedge = face_geometry_3d(x0, es, et, ef, nF, eocc)[3]
                m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            if "mono_delta" not in m:
                m["mono_delta"] = self._rest_offset(x0, es, et, ef, nF, h_cell, V_eq,
                                                    m["alive"], R0t, eocc, vocc)
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
        # STASHED FOR THE TOPOLOGY OPERATORS, exactly as the default implementation does it:
        # `edge_flip` reads the plane off `m["mech"]`, so a monolayer sheet keeps its T1s in plane.
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        x = x0.clone()
        for _ in range(max(1, self.relax_iters)):
            step = -(self.eta * self.mu) * self._grad(x, es, et, ef, nF, h_cell, V_eq, m["alive"], R0t, eocc, vocc)
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
        # THE DIVISOR IS `general.dt`, NOT A DECLARED ONE -- the same by-construction fix as the
        # default implementation (see the long note at ShapeEnergy3D). This operator emits a
        # VELOCITY that the engine immediately multiplies by `general.dt`, so the two must cancel;
        # a declared `dt` that differs from the harness's silently rescales the relaxation rate and
        # nothing looks wrong. Both implementations of the same contract now read the harness.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics[monolayer]: `dt: {self.params['dt']}` is IGNORED -- the "
                 f"divisor is general.dt ({_dt}) so that the engine's own multiplication cancels "
                 f"it. Remove the parameter; leaving it in the spec suggests a knob that is not there.")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        return {self.at: v_full}


@register_operator("cell_mechanics", model="apicobasal", set="vertex", kind="lateral",
                   family="mechanics")
class ApicoBasalShapeEnergy3D(Lateral):
    """R3 OF THE APICO-BASAL PROMOTION: the energy, written on the doubled degree-of-freedom set.

    `model:` AND NOT `implementation:`, by the axis test. An
    implementation computes the SAME equation differently; this is a different equation about a
    different object. `cell_mechanics[model: monolayer]` says a cell's two surfaces are tied to its
    mid-surface by `a_i, b_i = x_i +/- (H_i/2) n_i`, so apical and basal can never slide past each
    other at any parameter value; this says they are free, and wedging, bottle cells and apical
    constriction are what the difference is made of. `cell_mechanics[model: apicobasal]` sits beside
    `monolayer` and `marinari` on the one contract.

    THE FUNCTIONAL IS THE MONOLAYER'S, TERM FOR TERM -- see `_apicobasal_energy_core`. Only the
    geometry the terms are evaluated on changes, which is what makes the promotion's first rung a
    REDUCTION rather than a comparison of two tunings: on a flat patch with `sep` frozen at
    (h0/2)n the polyhedron volume IS `A_mid * h`, so the two models must produce the same force.
    That is AB-C1, and it is the row that says the doubling generalises the incumbent.

    `sep_mu` IS A MOBILITY AND ZERO IS A LIMIT, NOT A MODE. The second degree-of-freedom group has
    its own drag, and `sep_mu` is its mobility relative to the mid-surface's `mu`. At `sep_mu: 0` the
    apico-basal separation is infinitely damped -- it cannot move, so the cell keeps the thickness it
    was seeded with while the mid-surface relaxes under the polyhedral energy. THAT IS A PHYSICAL
    LIMIT OF THIS MODEL, not a staging switch: it is the "thickness is set by the cell and defended
    absolutely" case, and it is the case in which AB-C1 and AB-C5 are closed forms. R3 and R4 run at
    0 because a closed form for the volume of a prism of KNOWN thickness is only a closed form while
    the thickness is known; R5 raises it to 1 and the separation becomes a solver outcome.

    WHAT IT DOES NOT DO. It does not own the apico-basal AXIS: nothing here normalises `sep` or
    enforces its sign, because a normalisation of another operator's state is not a mechanism. The
    cost of that decision is gate row AB-B1, which counts the vertices whose span has inverted, and
    that cost is paid in the table rather than hidden in a clamp.
    """
    SUPPORTED_DIMS = [3]; EMIT = "velocity"; DIFFERENTIABLE = True
    INPUTS = ["vertex"]; OUTPUTS = ["vertex"]; READS = ["pos", "sep"]; WRITES = ["pos", "sep"]
    MAPS = ["E_srce", "E_trgt", "E_face"]
    MECHANISM_TAGS = ["vertex_model", "apicobasal", "cell_polyhedron", "cell_3d_volume",
                      "surface_tension", "emergent_bending", "force_balance"]
    REFERENCE = ("Okuda, S. et al. (2013). Biomech. Model. Mechanobiol. 12:627-644 (3D vertex model "
                 "with independent apical and basal surfaces); Okuda, S. et al. (2018). Sci. Rep. "
                 "8:2386 (the monolayer reduction this generalises).")
    PARAM_ROLES = {"k_v": "cell_volume_elasticity", "kappa_s": "surface_tension",
                   "sep_mu": "apicobasal_mobility",
                   "surface": "which surface the ring terms act on"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "vertex")
        self.sep_block = str(params.get("sep_block", "sep"))
        self.k_v = float(params.get("k_v", 4.0)); self.kappa_s = float(params.get("kappa_s", 0.2))
        self.gamma = float(params.get("gamma", 0.0))
        self.Lambda = float(params.get("Lambda", 0.0)); self.K_R = float(params.get("K_R", 0.0))
        self.mu = float(params.get("mu", 1.0)); self.dt = float(params.get("dt", 1.0))
        self.relax_iters = int(params.get("relax_iters", 30)); self.eta = float(params.get("eta", 0.08))
        self.cap_frac = float(params.get("cap_frac", 0.12))
        self.sep_mu = float(params.get("sep_mu", 1.0))
        # WHICH SURFACE THE RING TERMS ACT ON -- see `_apicobasal_energy_core`. `apical` by default,
        # because the mid-surface is a coordinate choice and not a membrane, so a perimeter, a line
        # tension or a radial spring written on it is a force on the average of two surfaces.
        self.surface = str(params.get("surface", "apical")).lower()
        if self.surface not in ("apical", "basal", "mid"):
            raise ValueError(f"cell_mechanics[apicobasal]: surface must be apical|basal|mid, "
                             f"got {self.surface!r}")
        self.rest_calibration = str(params.get("rest_calibration", "force_balance"))
        _mk = params.get("mono_k", None)                  # see MonolayerShapeEnergy3D.__init__
        self.mono_k = None if _mk is None else float(_mk)
        _pa = params.get("plane_axis", None)
        self.plane_axis = None if _pa is None else int(_pa)

    # ---------------------------------------------------------------- gradients
    def _grad(self, x, s, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, want_sep):
        """(dU/dx, dU/ds). ONE autograd pass over both, because they are one energy.

        `want_sep` is False at `sep_mu: 0`, where the second gradient would be computed, multiplied
        by zero and thrown away -- and it is not free: it doubles the graph the relaxation walks
        thirty times a frame.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            s = s.detach().requires_grad_(want_sep)
            E = _apicobasal_energy_core(x, s, es, et, ef, nF, V_eq, alive, self.k_v, self.kappa_s,
                                        self.Lambda, self.K_R, R0t, eocc, vocc, self.gamma,
                                        self.surface)
            if want_sep:
                gx, gs = torch.autograd.grad(E, (x, s))
                return torch.nan_to_num(gx), torch.nan_to_num(gs)
            gx = torch.autograd.grad(E, x)[0]
        return torch.nan_to_num(gx), None

    def _rest_offset(self, x0, s0, es, et, ef, nF, V_eq0, alive, R0t, eocc, vocc):
        """The constant added to every cell's target volume so the SEEDED SHELL IS AT REST.

        THE SAME SOLVE THE MONOLAYER DOES, on this geometry -- see `MonolayerShapeEnergy3D._rest_offset`
        for why it exists at all (only the volume term can push outward; calibrating `V_eq` to the
        rest volume balances it against nothing and a freshly seeded ball collapses under its own
        tension, radius 5.00 -> 1.80 in 20 frames), why the correction is a CONSTANT and not a factor,
        why it is solved rather than typed in, and why it iterates. Repeated here rather than
        inherited because the gradient is this energy's, and a shared solve that called the wrong
        `_grad` would calibrate one model against the other's rest state.

        `sep` IS HELD FIXED THROUGHOUT THE SOLVE, at every `sep_mu`. The question the offset answers
        is which target volume puts the MID-SURFACE at radius R; letting the thickness relax inside
        the calibration would answer it with a different tissue than the one that then runs.
        """
        R_target = x0.norm(dim=1).mean().clamp(min=1e-9)
        delta = torch.zeros((), dtype=x0.dtype, device=x0.device)
        x = x0.clone()
        for _ in range(8):
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            u = x / x.norm(dim=1, keepdim=True).clamp(min=1e-9)
            g0, _ = self._grad(x, s0, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, False)
            with torch.enable_grad():
                xg = x.detach().requires_grad_(True)
                vf, _, _, _ = apicobasal_geometry_3d(xg, s0, es, et, ef, nF, eocc)
                gV = torch.autograd.grad((vf * alive).sum(), xg)[0]
            den = self.k_v * (u * torch.nan_to_num(gV)).sum()
            if not torch.isfinite(den) or den.abs() < 1e-12:
                break
            step = ((u * g0).sum() / den).detach()
            if not torch.isfinite(step):
                break
            delta = delta + step
            V_eq = (V_eq0 + delta).clamp(min=1e-9)
            with torch.no_grad():
                cap = self.cap_frac * (x[et] - x[es]).norm(dim=-1).mean().clamp(min=1e-6)
            for _ in range(max(1, self.relax_iters)):
                gx, _ = self._grad(x, s0, es, et, ef, nF, V_eq, alive, R0t, eocc, vocc, False)
                st = -(self.eta * self.mu) * gx
                x = x + st * torch.clamp(cap / (st.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            x = x * (R_target / x.norm(dim=1).mean().clamp(min=1e-9))
        return delta.detach()

    # ---------------------------------------------------------------- forward
    def forward(self, H, mask=None):
        lvl = H.level(self.at); pos_full = lvl.get("pos"); v_full = torch.zeros_like(pos_full)
        m = getattr(lvl, "_mesh", None)
        if m is None:
            return {self.at: v_full}
        if self.sep_block not in lvl.state_schema:
            raise ValueError(
                f"cell_mechanics[apicobasal] needs a `{self.sep_block}` block on the {self.at!r} "
                f"set -- seed it with `{{op: seed_mesh, implementation: apicobasal}}`. Without it "
                f"there is no second surface and the run would be the mid-surface model wearing "
                f"this operator's name.")
        Nv = int(m["Nv"]); nF = int(m["nF"]); es, et, ef = m["E_srce"], m["E_trgt"], m["E_face"]
        E = es.shape[0]; dev, dt = pos_full.device, pos_full.dtype
        x0 = pos_full[:Nv].detach().clone()
        s0 = lvl.get(self.sep_block)[:Nv].detach().clone()
        eocc = torch.ones(E, device=dev, dtype=dt); vocc = torch.ones(Nv, device=dev, dtype=dt)
        R0t = torch.as_tensor(float(m["R0"]), dtype=dt, device=dev)
        # PUBLISHED FOR THE RENDERER, and it is now a MEASUREMENT rather than a declared constant:
        # the monolayer writes its `h0` here because its thickness is one, and this model's is a
        # per-vertex vector. The cross section wants one number, so it gets the mean cell thickness
        # |a - b| = 2|sep|, which is what `h0` meant on the other model.
        m["mono_h"] = float(2.0 * s0.norm(dim=1).mean())
        # THE SAME V_eq THE MONOLAYER GETS, and that is what makes AB-C1 a controlled comparison:
        # both arms calibrate the wedge target `V0f` -- the quantity `cell_grow` scales -- against
        # their own rest volume, and on a flat patch those rest volumes are the same number.
        v_rest, _, _, _ = apicobasal_geometry_3d(x0, s0, es, et, ef, nF, eocc)
        if "mono_k" not in m:
            if self.mono_k is not None:
                m["mono_k"] = self.mono_k                     # declared: see the monolayer's __init__
            else:
                wedge = face_geometry_3d(x0, es, et, ef, nF, eocc)[3]
                m["mono_k"] = float((v_rest.median() / wedge.median().clamp(min=1e-9)).item())
        V_eq = (m["mono_k"] * m["V0f"]).clamp(min=1e-9)
        if self.rest_calibration == "force_balance":
            if "mono_delta" not in m:
                m["mono_delta"] = self._rest_offset(x0, s0, es, et, ef, nF, V_eq, m["alive"],
                                                    R0t, eocc, vocc)
            V_eq = (V_eq + m["mono_delta"]).clamp(min=1e-9)
        with torch.no_grad():
            cap = self.cap_frac * (x0[et] - x0[es]).norm(dim=-1).mean().clamp(min=1e-6)
            # THE SEPARATION'S CAP IS IN THE SEPARATION'S OWN UNIT -- a fraction of the mean cell
            # HALF-thickness, not of the mean junction length. A cap borrowed from the mid-surface
            # would be several times the whole thickness on a thin epithelium, which is not a cap.
            cap_s = self.cap_frac * s0.norm(dim=1).mean().clamp(min=1e-6)
        m["mech"] = dict(K_A=0.0, K_P=0.0, K_V=self.k_v, K_R=self.K_R, Lambda=self.Lambda,
                         Gamma=self.gamma, eta=self.eta, cap_frac=self.cap_frac,
                         plane_axis=self.plane_axis)
        move_sep = self.sep_mu != 0.0
        x = x0.clone(); s = s0.clone()
        for _ in range(max(1, self.relax_iters)):
            gx, gs = self._grad(x, s, es, et, ef, nF, V_eq, m["alive"], R0t, eocc, vocc, move_sep)
            step = -(self.eta * self.mu) * gx
            step = step * torch.clamp(cap / (step.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
            if self.plane_axis is not None:
                step[:, self.plane_axis] = 0.0
            x = x + step
            if move_sep:
                ds = -(self.eta * self.mu * self.sep_mu) * gs
                ds = ds * torch.clamp(cap_s / (ds.norm(dim=1, keepdim=True) + 1e-12), max=1.0)
                s = s + ds
        # THE DIVISOR IS `general.dt`, NOT A DECLARED ONE -- the same by-construction fix both other
        # implementations of this contract carry: the engine multiplies an emitted velocity by
        # `general.dt`, so the two must cancel or the relaxation rate is silently rescaled.
        _dt = float(getattr(H, "dt", self.dt) or self.dt)
        if "dt" in self.params and abs(float(self.params["dt"]) - _dt) > 1e-12 * max(_dt, 1.0) \
                and not getattr(self, "_dt_warned", False):
            self._dt_warned = True
            from plexus.paths import warn
            warn(f"[warn] cell_mechanics[apicobasal]: `dt: {self.params['dt']}` is IGNORED -- the "
                 f"divisor is general.dt ({_dt}).")
        v_full[:Nv] = (x - x0) / max(_dt, 1e-9)
        if mask is not None:
            v_full = v_full * mask[:, None].float()
        out = {self.at: v_full}
        if move_sep:
            # THE SECOND DELTA, ROUTED BY `(set, block)` -- what R1(a) was built for. Without that
            # key the engine folds every delta for a set into its COORDINATE block, so a separation
            # velocity would be added to `pos` and the tissue would translate instead of thicken.
            vs = torch.zeros_like(pos_full)
            vs[:Nv] = (s - s0) / max(_dt, 1e-9)
            if mask is not None:
                vs = vs * mask[:, None].float()
            out[(self.at, self.sep_block)] = vs
        return out


@register_operator("cell_mechanics", model="marinari", set="vertex", kind="lateral",
                   family="mechanics")
class ShapeEnergy3DMarinari(ShapeEnergy3D):
    """The work function of Marinari et al., Nature 484:542 (2012), Supplementary p.1:

        W = SUM_cells (K/2)(A - A0)^2  +  SUM_junctions [ Lambda * l  +  (Gamma/2) * l^2 ]

    A DIFFERENT MODEL, NOT A DIFFERENT IMPLEMENTATION, which is why it is on the `model=` axis and
    not `implementation=`. The default body is Farhadifar's: its contractility is (Gamma/2) * P^2,
    one number per CELL. This one puts the quadratic on each JUNCTION instead, so every junction is
    an elastic spring of equilibrium length l0 = -Lambda/Gamma -- and with the paper's own values,
    Lambda = 56.8 and Gamma = 49.9, that length is NEGATIVE (-1.14). No junction has a rest length it
    can reach; each is under tension all the way down to zero, resisted only by the area term. That
    is the whole engine of the paper: junctions shrink, T1s fire, cells lose neighbours, and the ones
    that reach three junctions and a quarter of their target area are extruded. The two forms are
    rival hypotheses about one force, so a spec must not be able to set both -- this variant zeroes
    the perimeter terms rather than adding to them.

    THE PAPER'S PARAMETERS ARE THE SPEC'S PARAMETERS, and the factor-of-two bookkeeping happens here
    rather than in anyone's yaml:

      K       -> K_A = K/2         the core writes K_A*(A-A0)^2, the paper writes (K/2)(A-A0)^2
      Lambda  -> Lam  = Lambda/2   the core sums over HALF-EDGES, the paper over JUNCTIONS, and a
      Gamma   -> Gam_l = Gamma/2   closed half-edge mesh has two half-edges per junction, so a
                                   half-edge sum is exactly twice a junction sum. `Gam_l` is then
                                   halved again inside the core (it writes 0.5*Gam_l*l^2), giving
                                   (Gamma/4)*SUM_halfedges l^2 = (Gamma/2)*SUM_junctions l^2.

    Getting this wrong is silent: it does not crash, it runs the tissue at twice the tension the spec
    asked for, and the delamination rate -- the paper's entire observable -- is a steep function of
    exactly that.

    WHAT THIS CLASS DOES NOT GIVE YOU, because they are not the energy: their dynamics are Metropolis
    Monte Carlo on vertex positions (accept if dW<0, else with probability exp(-dW)), their T1 is a
    Metropolis-accepted move rather than a threshold flip, their boundary is a periodic box of FIXED
    area, and their T2 removes a cell at exactly 3 junctions and area < A0/4. This operator is the
    force law alone; on the inherited overdamped-descent mover it will relax to a minimum of W rather
    than fluctuate through it, so it reproduces their ENERGY and not yet their PHENOMENON.
    """

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        K = float(params.get("K", 160.0))                     # Table S1, uncrowded
        Lambda = float(params.get("Lambda", 56.8))
        Gamma = float(params.get("Gamma", 49.9))
        # CROWDING, AS THE PAPER APPLIES IT: not by shrinking the box but by rescaling the parameters
        # together, "values [that] correspond to a tissue that would have the same value of the work
        # function ... were it free to expand so that its total area was larger by a factor g".
        # A0' = g A0 is the seeder's business; the three coefficients are this operator's.
        g = float(params.get("crowding", 1.0))
        if g <= 0:
            raise ValueError(f"cell_mechanics[marinari]: crowding must be > 0, got {g}")
        K = K * g ** -2.0
        Lambda = Lambda * g ** -0.5
        Gamma = Gamma / g
        self.K_A = 0.5 * K
        self.Lambda = 0.5 * Lambda
        self.Gam_l = 0.5 * Gamma
        # THE RIVAL TERM IS OFF, not merely defaulted off. `Gamma` is NOT rejected here even though
        # the default model has a parameter of that name: in THIS model `Gamma` is the per-junction
        # spring constant, read above, and rejecting it would refuse the paper's own notation. It is
        # the PERIMETER terms that must not also be live -- `K_P` unambiguously names one, and the
        # default's perimeter `Gamma` is zeroed below after its value has been consumed as the
        # junction spring. (The first version of this guard rejected `Gamma` and so refused every
        # spec written in the paper's own symbols.)
        if params.get("K_P") is not None:
            raise ValueError(
                "cell_mechanics[model: marinari]: 'K_P' is the perimeter elasticity of the default "
                "(Farhadifar) model, which this model replaces with a per-junction spring. Running "
                "both would be neither model. Drop 'K_P', or use the default model if you want it.")
        self.K_P = 0.0
        self.Gamma = 0.0
        self._said_marinari = False

    def forward(self, H, mask=None):
        if not self._said_marinari:
            self._said_marinari = True
            _l0 = (-2.0 * self.Lambda) / (2.0 * self.Gam_l) if self.Gam_l else float("nan")
            print(f"[cell_mechanics/marinari] W = (K/2)(A-A0)^2 + sum_junctions[Lambda*l + "
                  f"(Gamma/2)l^2]  ->  K_A {self.K_A:g}, Lam {self.Lambda:g} (per half-edge), "
                  f"Gam_l {self.Gam_l:g}; junction rest length l0 = -Lambda/Gamma = {_l0:.3f}"
                  f"{'  (NEGATIVE: every junction is under tension to zero length)' if _l0 < 0 else ''}",
                  flush=True)
        return super().forward(H, mask)


# ----------------------------------------------------------------------------------------------
# Every variant of `cell_mechanics` lives in this file -- they are one contract, and reading them
# side by side is how anyone can tell which one a specification is getting.
#
# `HAVE_WARP` and `_wp_launch` come from `mpm_ops`, and that is a smell worth naming rather than
# hiding: the Warp launch helper is shared infrastructure that happens to live in an MPM module, so
# a vertex operator imports from an MPM one to get it. Moving it is a separate change with its own
# blast radius.
#
# NVIDIA Warp gradient for the vertex-model shape energy: `cell_mechanics[implementation: warp]`.
# ==========================================================================================================
from plexus.operators.mpm_ops import HAVE_WARP, _wp_launch      # noqa: E402  see the note above

if HAVE_WARP:
    import warp as wp

    @wp.kernel
    def face_accum(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                   ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                   EOCC: wp.array(dtype=float),
                   N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                   CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3)):
        """Per-face area vector, perimeter, live-edge count and centroid sum -- `face_geometry_3d`.

        The half-sum in N_f = (1/2) sum (s x t) is NOT applied here: it is applied once per face in
        `face_scalars`, so this kernel is exactly the accumulation and the factor lives in one place.
        """
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:                                # dead reservoir slot: contributes to nothing
            return
        f = EF[e]
        s = POS[ES[e]]
        t = POS[ET[e]]
        wp.atomic_add(N, f, wp.cross(s, t) * w)
        wp.atomic_add(P, f, wp.length(t - s) * w)
        wp.atomic_add(CNT, f, w)
        wp.atomic_add(CSUM, f, s * w)

    @wp.kernel
    def face_scalars(N: wp.array(dtype=wp.vec3), P: wp.array(dtype=float),
                     CNT: wp.array(dtype=float), CSUM: wp.array(dtype=wp.vec3),
                     A0: wp.array(dtype=float), P0: wp.array(dtype=float),
                     V0F: wp.array(dtype=float), ALIVE: wp.array(dtype=float),
                     K_A: float, K_P: float, K_V: float, Gam: float,
                     G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                     CG: wp.array(dtype=wp.vec3)):
        """The three per-face derivatives, folded into what the half-edge pass actually needs:
        `G` (the covector on N_f, already halved), `DP` (dE/dP_f) and `CG` (the centroid route)."""
        f = wp.tid()
        a = ALIVE[f]
        Nf = N[f] * 0.5                             # the half-sum, applied once
        area = wp.length(Nf)
        cnt = wp.max(CNT[f], 1.0)
        cen = CSUM[f] / cnt
        vf = wp.dot(cen, Nf) / 3.0

        dA = 2.0 * K_A * (area - A0[f]) * a
        DP[f] = (2.0 * K_P * (P[f] - P0[f]) + Gam * P[f]) * a
        dv = 2.0 * K_V * (vf - V0F[f]) * a

        nhat = wp.vec3(0.0, 0.0, 0.0)
        if area > 1.0e-20:
            nhat = Nf / area
        # HALVED HERE, not in the half-edge kernel: dN_f/d(cross_e) = 1/2, and folding it in once
        # per face keeps the scatter pass to the three cross products it is really doing.
        G[f] = (nhat * dA + cen * (dv / 3.0)) * 0.5
        CG[f] = Nf * (dv / (3.0 * cnt))

    @wp.kernel
    def edge_grad(POS: wp.array(dtype=wp.vec3), ES: wp.array(dtype=wp.int32),
                  ET: wp.array(dtype=wp.int32), EF: wp.array(dtype=wp.int32),
                  EOCC: wp.array(dtype=float), MYO: wp.array(dtype=float),
                  G: wp.array(dtype=wp.vec3), DP: wp.array(dtype=float),
                  CG: wp.array(dtype=wp.vec3),
                  Lam: float, Gam_l: float, has_myo: int,
                  GRAD: wp.array(dtype=wp.vec3)):
        """Scatter every half-edge's contribution onto its two endpoints."""
        e = wp.tid()
        w = EOCC[e]
        if w <= 0.0:
            return
        f = EF[e]
        i = ES[e]
        j = ET[e]
        s = POS[i]
        t = POS[j]

        # --- through the area vector and the wedge volume (both act on N_f) ---------------- #
        Gf = G[f]
        wp.atomic_add(GRAD, i, wp.cross(t, Gf) * w)
        wp.atomic_add(GRAD, j, wp.cross(Gf, s) * w)

        # --- through the centroid: cen_f averages the SOURCE vertices only ------------------ #
        wp.atomic_add(GRAD, i, CG[f] * w)

        # --- through the edge length: perimeter + line tension + junction spring ------------ #
        d = t - s
        L = wp.length(d)
        if L > 1.0e-20:
            u = d / L
            m = 1.0
            if has_myo == 1:
                m = MYO[e]
            c = (DP[f] + Lam * m + Gam_l * L) * w
            wp.atomic_add(GRAD, j, u * c)
            wp.atomic_add(GRAD, i, u * (-c))

    @wp.kernel
    def vertex_radial(POS: wp.array(dtype=wp.vec3), VOCC: wp.array(dtype=float),
                      R0: float, K_R: float, GRAD: wp.array(dtype=wp.vec3)):
        """K_R (|x| - R0)^2 over live vertices -- the one term that is not a sum over faces."""
        v = wp.tid()
        w = VOCC[v]
        if w <= 0.0:
            return
        x = POS[v]
        r = wp.length(x)
        if r > 1.0e-20:
            wp.atomic_add(GRAD, v, x * (2.0 * K_R * (r - R0) * w / r))


_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        from plexus.paths import warn
        warn(msg)


def shape_energy_grad_warp(pos, es, et, ef, nF, A0, P0, V0f, alive, R0, K_A, K_P, K_V, K_R,
                           Lam, Gam, eocc, vocc, myo_e=None, Gam_l=0.0, buffers=None):
    """dE/d(pos) for `_shape_energy_core`, in four warp kernels instead of an autograd backward.

    `buffers` -- an optional dict reused across the relax loop's iterations, so the per-face and
    per-vertex scratch is allocated once per frame rather than once per descent step. The arrays are
    zeroed here, which is a torch `zero_` per buffer and cheaper than reallocating them.
    """
    dev = pos.device
    Nv = pos.shape[0]
    E = es.shape[0]
    b = buffers if buffers is not None else {}
    if b.get("_nF") != nF or b.get("_Nv") != Nv:
        b.clear()
        b["_nF"] = nF; b["_Nv"] = Nv
        z = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float32)  # noqa: E731
        b["N"] = z(nF, 3); b["P"] = z(nF); b["CNT"] = z(nF); b["CSUM"] = z(nF, 3)
        b["G"] = z(nF, 3); b["DP"] = z(nF); b["CG"] = z(nF, 3); b["GRAD"] = z(Nv, 3)
    for k in ("N", "P", "CNT", "CSUM", "GRAD"):
        b[k].zero_()

    wpos = wp.from_torch(pos.contiguous(), dtype=wp.vec3)
    wes = wp.from_torch(es.to(torch.int32).contiguous())
    wet = wp.from_torch(et.to(torch.int32).contiguous())
    wef = wp.from_torch(ef.to(torch.int32).contiguous())
    weo = wp.from_torch(eocc.contiguous())
    wN = wp.from_torch(b["N"], dtype=wp.vec3); wP = wp.from_torch(b["P"])
    wC = wp.from_torch(b["CNT"]); wCS = wp.from_torch(b["CSUM"], dtype=wp.vec3)
    wG = wp.from_torch(b["G"], dtype=wp.vec3); wDP = wp.from_torch(b["DP"])
    wCG = wp.from_torch(b["CG"], dtype=wp.vec3)
    wGRAD = wp.from_torch(b["GRAD"], dtype=wp.vec3)
    has_myo = 1 if myo_e is not None else 0
    wmyo = wp.from_torch((myo_e if myo_e is not None else eocc).contiguous())

    _wp_launch(face_accum, E, dev, [wpos, wes, wet, wef, weo, wN, wP, wC, wCS])
    _wp_launch(face_scalars, nF, dev,
               [wN, wP, wC, wCS, wp.from_torch(A0.contiguous()), wp.from_torch(P0.contiguous()),
                wp.from_torch(V0f.contiguous()), wp.from_torch(alive.contiguous()),
                float(K_A), float(K_P), float(K_V), float(Gam), wG, wDP, wCG])
    _wp_launch(edge_grad, E, dev,
               [wpos, wes, wet, wef, weo, wmyo, wG, wDP, wCG,
                float(Lam), float(Gam_l), int(has_myo), wGRAD])
    _wp_launch(vertex_radial, Nv, dev,
               [wpos, wp.from_torch(vocc.contiguous()), float(R0), float(K_R), wGRAD])
    return b["GRAD"]


def try_shape_energy_grad(op, p, es, et, ef, nF, A0, P0, V0f, alive, R0t, eocc, vocc,
                          twin_face=None, myo_e=None):
    """The gradient for `op` in warp, or None if this run is not one the kernels can serve.

    CALLED BY `ShapeEnergy3D._grad`, WHICH IS WHY IT IS A FUNCTION AND NOT A SUBCLASS. Warp is the
    default backend, and the default has to reach `cell_mechanics`, `cell_mechanics[model:
    marinari]` and anything else that inherits that gradient. A subclass reaches exactly one of them
    -- `marinari` derives from `ShapeEnergy3D`, so a `ShapeEnergy3DWarp` sibling would have left the
    Marinari sweep on autograd while reporting that warp was the default.

    RETURNS None RATHER THAN RAISING, so the caller simply carries on into autograd. The reason is
    printed once per distinct cause: a fallback that is invisible is a performance cliff nobody can
    find, and one that is printed per frame is noise in a 500-frame log.
    """
    why = None
    from plexus.engine import STRICT_DETERMINISM
    if STRICT_DETERMINISM:
        # STRICT MEANS STRICT, AND THESE KERNELS CANNOT HONOUR IT. `PLEXUS_STRICT_DETERMINISM` sets
        # `torch.use_deterministic_algorithms(True, warn_only=False)` precisely so that a kernel
        # with no deterministic implementation FAILS rather than quietly running the
        # nondeterministic path -- see the note at engine.py:58. `wp.atomic_add` is outside torch's
        # machinery altogether, so it would not fail and would not be deterministic either: the run
        # would claim reproducibility it does not have. That is the one outcome the flag exists to
        # prevent, so the warp path steps aside whenever it is set.
        #
        # It is also what makes a bit-identity comparison against an older commit possible at all:
        # with the flag set on both sides, both take the autograd path, where otherwise one side
        # would compare a Warp gradient against an autograd one and no row could match.
        why = "PLEXUS_STRICT_DETERMINISM is set, and wp.atomic_add is outside torch's guarantee"
    elif not HAVE_WARP:
        why = "warp is not installed"
    elif p.device.type != "cuda":
        why = f"the run is on {p.device.type}, and these kernels are CUDA-only"
    elif p.dtype != torch.float32:
        why = f"positions are {p.dtype}; the kernels are float32"
    elif getattr(op, "K_bend", 0.0) > 0:
        why = "K_bend > 0 -- the dihedral hinge is not ported"
    elif getattr(op, "K_lumen", 0.0) > 0:
        why = "K_lumen > 0 -- the global isoperimetric term is not ported"
    if why is not None:
        _warn_once(why, f"[warn] cell_mechanics: the warp gradient is unavailable because {why}, so "
                        f"this run uses autograd. The physics is identical; the speed is not.")
        return None
    if op._centre is not None:
        # EVERY TERM BUT THE RADIAL ONE IS TRANSLATION-INVARIANT, and the caller evaluates the whole
        # energy on `p - centre`; d/dp and d/d(p-centre) are the same map, so shifting the positions
        # here reproduces it exactly.
        p = p - op._centre.to(p.device, p.dtype)
    if not hasattr(op, "_wbuf"):
        op._wbuf = {}
    g = shape_energy_grad_warp(p, es, et, ef, nF, A0, P0, V0f, alive, float(R0t),
                               op.K_A, op.K_P, op.K_V, op.K_R, op.Lambda, op.Gamma,
                               eocc, vocc, myo_e=myo_e, Gam_l=op.Gam_l, buffers=op._wbuf)
    return torch.nan_to_num(g)


@register_operator("cell_mechanics", implementation="warp", set="vertex", kind="lateral",
                   family="mechanics")
class ShapeEnergy3DWarp(ShapeEnergy3D):
    """`cell_mechanics` with the shape-energy gradient in warp -- WHICH IS THE DEFAULT.

    Kept only as a name a spec can still write, so archived specs that say `implementation: warp`
    keep loading; it selects exactly the behaviour the bare operator has. Which route computes the
    derivative is a backend choice, not a model, so there is no longer an `autograd` variant to
    select in the other direction -- `_grad` uses the warp kernels wherever they apply and falls
    back to `torch.autograd` where they do not, saying so once.
    """
    MECHANISM_TAGS = ShapeEnergy3D.MECHANISM_TAGS + ["warp"]

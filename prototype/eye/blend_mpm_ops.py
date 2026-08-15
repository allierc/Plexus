"""blend_mpm_ops -- build the MPM eye from the Blender model (PROTOTYPE-LOCAL, not promoted).

Importing this module registers two `seed` operators, so a spec can start its material
points from `260802_s2_EYE_MUSCLES_MODEL.blend` instead of from the analytic ovoid and
the six generated straps:

    blend_globe     seed @ mpm_particle      the globe: retina shell + cornea + lens
    blend_muscles   seed @ muscle_particle   the six extraocular straps, with their fibres

They are DROP-IN REPLACEMENTS for `eye_anatomy` and `muscle_morphogenesis`: they write
exactly the buffers the rest of the plant reads, so `eye_pose`, `muscle_geometry`,
`oculomotor_drive`, `muscle_contract`, `bone_anchor`, `muscle_sleeve` and `orbit_socket`
are untouched. Swap the two operators in the spec and the same pipeline now runs on
scanned anatomy.

WHY `seed` AND NOT `rewire`. `eye_anatomy` and `muscle_morphogenesis` predate the `seed`
kind and are declared `rewire`; what they do -- write the initial configuration once and
never again -- is exactly what `plexus.models.base.Seed` was added for. The engine gates
a seed to the opening frames whatever the spec says, so the guarantee that scanned
anatomy is an INITIAL CONDITION and not a per-frame boundary condition is the language's,
not this file's.

FROM .blend TO PARTICLES, in four steps

  1. READ. `archive/run_03/read_blend.py` cuts the .blend into named watertight parts
     (`blend_parts/parts.npz`). That step needs Blender's own loader, so it runs under a
     `bpy` interpreter; this module SHELLS OUT to it when the npz is missing or older than
     the .blend, and otherwise reads the cache. A spec therefore names the .blend and gets
     particles, with one subprocess on the first run of a new blend.

  2. PLACE. The blend is a whole larva in head coordinates (+x the animal's right, +y
     caudal, +z dorsal); the plant runs in `fish_anatomy`'s per-eye frame inside a unit
     box (+x caudal, +y dorsal, +z the optic axis, globe at `center`). One similarity
     transform carries one across:

         x_sim = center + scale * R_eye @ (x_blend - globe_centre) ,
         scale = a_eq / (the globe's measured equatorial semi-axis)

     `R_eye` comes from `parts.json`, where it was measured (the optic axis is the
     direction centre -> cornea, not an assumption). For the RIGHT eye R is a proper
     rotation; the LEFT eye is its enantiomorph, so its frame is left-handed and the
     transform includes a REFLECTION -- which is what maps a left eye onto the same
     canonical layout the right one has, and the only thing it changes downstream is the
     sign of torsion. `side: R` is the default because it needs no reflection at all.

  3. FILL. Points are placed by rejection: a 3-D Hammersley sequence over the part's
     bounding box, kept where the generalized winding number says the point is inside the
     mesh. Deterministic (no RNG), uniform in volume (which is what MLS-MPM wants), and it
     needs nothing of the mesh but that it be closed -- all twelve muscle straps and all
     six eye tissues in this blend are watertight, with zero non-manifold edges.

  4. LABEL. The globe's points get a tissue id -- lens and cornea from the meshes
     themselves, sclera / choroid / vitreous from bands of the normalized ellipsoidal
     radius, pupil and iris from the polar angle -- so `eye_pose` can fit on the shell and
     the renderer can colour by tissue exactly as before. A muscle's points get the fibre
     coordinate `s` (0 at the ORIGIN, 1 at the INSERTION, as `muscle_morphogenesis`
     defines it), the local fibre direction from the measured centreline, and the two caps
     (`anchored` at the bone, `tendon` in the sclera).

WHAT IS DIFFERENT FROM THE GENERATED PLANT, and it is the point of the exercise:

  * the straps are the artist's geometry, so their bellies, their curvature and their
    arcs of contact are whatever the model says -- no `strap_path`, no tangent
    construction, no `frac` truncation, no `gap`/`embed` knobs;
  * the six muscles have DIFFERENT VOLUMES (IO 0.0062 against MR 0.0031 blend-units^3, a
    factor 2.0 of the smallest), and since the engine gives every muscle the same number
    of points, per-particle volume carries that difference. Force is active stress x
    cross-section, so the volumes ARE the relative strengths, exactly as
    `fish_anatomy.strap_widths` intended;
  * the globe is a triaxial ellipsoid (semi-axes 0.688 / 0.575 / 0.468 blend units,
    axial/equatorial = 0.741 against the 0.676 measured off Fig. 12.1A) rather than a
    body of revolution.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from plexus.models.base import Seed
from plexus.models.registry import register_operator

import eye_anatomy as EA

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BLEND = os.path.join(HERE, "260802_s2_EYE_MUSCLES_MODEL.blend")
READER = os.path.join(HERE, "archive", "run_03", "read_blend.py")
DEFAULT_PARTS = os.path.join(HERE, "archive", "run_03", "blend_parts")

MUSCLE_KEYS = EA.MUSCLE_KEYS                       # LR, SR, MR, IR, SO, IO -- the set's order


# --------------------------------------------------------------------------- #
#  1. the cut: read it, or make it
# --------------------------------------------------------------------------- #
def load_cut(blend=DEFAULT_BLEND, parts_dir=DEFAULT_PARTS, rebuild=False):
    """(arrays, manifest) for the blend, regenerating the cut if it is stale.

    The cut is a CACHE of the .blend, so it is rebuilt when it is missing or older than
    the file it came from -- one subprocess under the bpy interpreter, which this process
    cannot import (bpy ships its own Python and its own numpy).
    """
    # relative paths resolve against THIS prototype, not the cwd: a spec that names
    # "archive/run_03/blend_parts" has to mean the same thing in the devcontainer
    # (/workspace/Plexus/...) and on the cluster (/groups/saalfeld/.../Plexus/...), which
    # mount the same NFS export at different roots.
    blend = blend if os.path.isabs(blend) else os.path.join(HERE, blend)
    parts_dir = parts_dir if os.path.isabs(parts_dir) else os.path.join(HERE, parts_dir)
    npz = os.path.join(parts_dir, "parts.npz")
    js = os.path.join(parts_dir, "parts.json")
    stale = (not os.path.exists(npz) or not os.path.exists(js)
             or os.path.getmtime(npz) < os.path.getmtime(blend))
    if rebuild or stale:
        print(f"[blend] cutting {os.path.basename(blend)} -> {parts_dir}", flush=True)
        subprocess.run([sys.executable, READER, "--blend", blend, "--out", parts_dir],
                       check=True)
    return np.load(npz), json.load(open(js))


# --------------------------------------------------------------------------- #
#  2. the transform: head coordinates -> the plant's per-eye frame
# --------------------------------------------------------------------------- #
class BlendFrame:
    """The similarity that carries the blend's head coordinates into the sim's unit box.

    Holds the measured globe (centre, semi-axes in the eye frame) so the operators can
    normalize radii the way `eye_anatomy` did, and reports the scale it chose.
    """

    def __init__(self, manifest, arrays, side="R", a_eq=EA.A_EQ, center=EA.GLOBE_CENTER,
                 inflate=1.0):
        if side not in manifest["eyes"]:
            raise ValueError(f"side {side!r} not in the cut; have {sorted(manifest['eyes'])}")
        e = manifest["eyes"][side]
        self.side = side
        # INFLATE scales the GLOBE only, about its own centre, leaving the six straps
        # exactly where the artist drew them. It is the one knob that changes which tissue
        # the insertions are in: at 1.0 the tendon tips graze the sclera, at 1.2 the globe
        # has grown out past them and they sit inside it, so the shared MLS-MPM grid welds
        # tendon to sclera over a volume instead of at a surface.
        self.inflate = float(inflate)
        self.R = np.asarray(e["frame"], float)              # rows: caudal, dorsal, lateral
        self.det = float(np.linalg.det(self.R))
        self.c_blend = np.asarray(e["globe_center"], float)
        self.center = np.asarray(center, float)

        # semi-axes IN THE EYE FRAME, from the retina shell: for a smooth closed ellipsoid
        # the half-extent along an axis IS the semi-axis, and unlike a covariance fit it
        # does not care that the mesh's vertices are unevenly spread.
        loc = (np.asarray(arrays[f"{side}_retina__v"], float) - self.c_blend) @ self.R.T
        self.semi_blend = np.abs(loc).max(axis=0)           # (caudal, dorsal, lateral)
        self.equatorial = float(self.semi_blend[:2].mean())
        self.axial_ratio = float(self.semi_blend[2] / self.equatorial)
        self.scale = float(a_eq) / self.equatorial
        self.semi = self.semi_blend * self.scale * self.inflate       # sim units, AS BUILT

    def __call__(self, X):
        """[n,3] blend points -> [n,3] sim points. Muscles, bones: everything but the globe."""
        return self.center[None, :] + self.scale * ((np.asarray(X, float) - self.c_blend) @ self.R.T)

    def globe(self, X):
        """The same map for the GLOBE's meshes, with `inflate` applied about its centre."""
        return self.center[None, :] + (self.scale * self.inflate) * (
            (np.asarray(X, float) - self.c_blend) @ self.R.T)

    def describe(self):
        infl = "" if abs(self.inflate - 1.0) < 1e-9 else f", globe inflated x{self.inflate:.2f}"
        return (f"side {self.side} (frame det {self.det:+.0f}"
                f"{', reflected' if self.det < 0 else ''}), scale {self.scale:.4f} sim/blend, "
                f"semi-axes {np.round(self.semi, 4).tolist()} (axial/equatorial "
                f"{self.axial_ratio:.3f}){infl}")


# --------------------------------------------------------------------------- #
#  3. the fill: uniform points inside a closed triangle mesh
# --------------------------------------------------------------------------- #
def _halton(n, offset=0, bases=(2, 3, 5)):
    """[n,3] of the Halton sequence -- `muscle_morphogenesis`'s Hammersley fill, made
    RESUMABLE. Hammersley's first coordinate is (j+0.5)/N, so it depends on how many points
    you meant to draw; rejection sampling does not know that in advance and draws again
    until enough have landed inside. Halton has no N in it, so round two continues round
    one instead of re-covering it."""
    j = np.arange(offset + 1, offset + n + 1, dtype=np.int64)
    out = []
    for base in bases:
        v = np.zeros(len(j), float)
        f, i = 1.0 / base, j.copy()
        while np.any(i > 0):
            v += f * (i % base)
            i //= base
            f /= base
        out.append(v)
    return np.stack(out, axis=1)


def _dev(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inside_mesh(pts, V, F, device=None):
    """Generalized winding number: True where `pts` lie inside the closed mesh (V, F).

    Van Oosterom & Strackee's solid angle, summed over triangles: exactly 4*pi inside a
    watertight mesh and 0 outside, so the 2*pi threshold sits as far from both as a
    threshold can. In torch and chunked over points, because the intermediate is
    [n_points, n_triangles, 3] and a 45 000-point globe against 1024 triangles is 138 M
    triangle-point terms -- two minutes of numpy, under a second on the GPU the run is
    already using.
    """
    dev = _dev(device)
    P = torch.as_tensor(np.asarray(pts, np.float32), device=dev)
    Vt = torch.as_tensor(np.asarray(V, np.float32), device=dev)
    Ft = torch.as_tensor(np.asarray(F, np.int64), device=dev)
    A, B, C = Vt[Ft[:, 0]], Vt[Ft[:, 1]], Vt[Ft[:, 2]]
    out = torch.zeros(len(P), dtype=torch.bool, device=dev)
    chunk = max(1, int(2.0e7 // max(len(Ft), 1)))
    for lo in range(0, len(P), chunk):
        p = P[lo:lo + chunk].unsqueeze(1)
        a, b, c = A.unsqueeze(0) - p, B.unsqueeze(0) - p, C.unsqueeze(0) - p
        na, nb, nc = a.norm(dim=2), b.norm(dim=2), c.norm(dim=2)
        num = (a * torch.cross(b, c, dim=2)).sum(2)
        den = (na * nb * nc + (a * b).sum(2) * nc + (b * c).sum(2) * na + (c * a).sum(2) * nb)
        out[lo:lo + chunk] = (2.0 * torch.atan2(num, den)).sum(1).abs() > 2.0 * math.pi
    return out.cpu().numpy()


def surface_radius(origin, dirs, meshes, device=None):
    """Distance from `origin` to the OUTERMOST surface along each unit direction.

    Moller-Trumbore against every triangle, keeping the largest positive hit, so a
    direction that leaves through both the retina and the corneal bulge measures the bulge.
    This is what makes the normalized radius `rn` mean the same thing on scanned anatomy as
    it did on the analytic ovoid: 1.0 ON the surface, whatever shape the surface is. A
    fitted ellipsoid would not do -- this globe encloses 0.43 blend-units^3 against the
    0.78 of its own bounding ellipsoid, so an ellipsoidal `rn` would put the whole sclera
    band outside the tissue.
    """
    dev = _dev(device)
    O = torch.as_tensor(np.asarray(origin, np.float32), device=dev)
    D = torch.as_tensor(np.asarray(dirs, np.float32), device=dev)
    best = torch.zeros(len(D), device=dev)
    for V, F in meshes:
        Vt = torch.as_tensor(np.asarray(V, np.float32), device=dev)
        Ft = torch.as_tensor(np.asarray(F, np.int64), device=dev)
        A = Vt[Ft[:, 0]]
        e1, e2 = Vt[Ft[:, 1]] - A, Vt[Ft[:, 2]] - A
        s = (O.unsqueeze(0) - A)                                   # [T,3]
        chunk = max(1, int(1.5e7 // max(len(Ft), 1)))
        for lo in range(0, len(D), chunk):
            d = D[lo:lo + chunk].unsqueeze(1)                      # [n,1,3]
            h = torch.cross(d, e2.unsqueeze(0).expand(d.shape[0], -1, -1), dim=2)
            det = (e1.unsqueeze(0) * h).sum(2)
            ok = det.abs() > 1e-12                       # ray parallel to the triangle
            inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
            u = inv * (s.unsqueeze(0) * h).sum(2)
            q = torch.cross(s.unsqueeze(0).expand(d.shape[0], -1, -1),
                            e1.unsqueeze(0).expand(d.shape[0], -1, -1), dim=2)
            v = inv * (d * q).sum(2)
            t = inv * (e2.unsqueeze(0) * q).sum(2)
            hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
            tmax = torch.where(hit, t, torch.zeros_like(t)).max(dim=1).values
            best[lo:lo + chunk] = torch.maximum(best[lo:lo + chunk], tmax)
    return best.cpu().numpy()


def path_coordinate(pts, seed_pt, k=12):
    """[n] fibre coordinate along a strap: the GEODESIC distance from the insertion end,
    through the muscle's own material points, normalized to [0, 1] and flipped so 0 is the
    ORIGIN (the convention `muscle_morphogenesis` set).

    A k-nearest-neighbour graph over the points and Dijkstra from the points nearest the
    insertion. Projecting onto a straight principal axis -- the obvious alternative, and
    what the cut's centreline does -- FAILS on a muscle that wraps the globe: the strap is
    not monotone along any straight line, so points from the two ends of the wrap land in
    the same bin and others land in none. That is not hypothetical, it is the measured
    failure this replaced: SR came out with one of `muscle_geometry`'s 14 bins EMPTY, the
    empty bin's centroid defaulted to the origin of the world, and the muscle reported a
    length 12x its rest length (a shortening of -1107%). Distance THROUGH THE TISSUE has no
    such failure mode: it is monotone along the strap by construction.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    n = len(pts)
    k = int(min(k, n - 1))
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k + 1)
    rows = np.repeat(np.arange(n), k)
    G = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)                                     # an undirected neighbourhood
    src = np.argsort(((pts - np.asarray(seed_pt)[None, :]) ** 2).sum(1))[:max(8, n // 200)]
    d = dijkstra(G, directed=False, indices=src, min_only=True)
    if not np.isfinite(d).all():                           # a detached island, if any
        d[~np.isfinite(d)] = d[np.isfinite(d)].max()
    return 1.0 - d / max(d.max(), 1e-12), idx


def fibre_from_s(pts, s, idx, ridge=1e-9):
    """[n,3] unit fibre directions: the local gradient of the fibre coordinate.

    Least squares over each point's neighbourhood -- (x_j - x_i).g = s_j - s_i -- so the
    fibre follows the strap wherever it curves, which a single tangent per bin cannot.
    Points toward INCREASING s, i.e. origin -> insertion, as `muscle_morphogenesis` did.
    """
    dx = pts[idx[:, 1:]] - pts[:, None, :]
    ds = s[idx[:, 1:]] - s[:, None]
    A = np.einsum('nki,nkj->nij', dx, dx) + ridge * np.eye(3)[None]
    b = np.einsum('nki,nk->ni', dx, ds)
    g = np.linalg.solve(A, b[..., None])[..., 0]     # numpy 2: b must carry its own column
    return g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-12)


def binned_length(pts, s, bins=14):
    """The length `muscle_geometry` will measure at frame 0, by its own recipe.

    Rest length has to be measured the same way as the running length or every shortening
    percentage carries a constant bias. Returns (length, n_empty_bins); a non-zero second
    value means the runtime readout will place a bin centroid at the world origin.
    """
    b = np.clip((s * bins).astype(int), 0, bins - 1)
    cen, empty = [], 0
    for j in range(bins):
        sel = b == j
        if sel.any():
            cen.append(pts[sel].mean(axis=0))
        else:
            empty += 1
    cen = np.asarray(cen)
    return float(np.linalg.norm(np.diff(cen, axis=0), axis=1).sum()), empty


def relieve_overlap(pts, s, centre, shell, standoff, embed, cap, device=None):
    """Push the belly of a strap OUT of the globe, leaving its tendon embedded.

    The artist's straps intersect the eyeball -- 17% of SR's points, 21% of IR's, 11% of
    IO's lie inside the retina surface. In the animal that volume is Tenon's capsule: the
    belly SLIDES over the sclera and only the tendon is attached. In MLS-MPM an overlap is
    not a drawing convention, it is a weld: the shared grid transfers momentum wherever two
    bodies share cells, so an embedded belly glues the whole arc of contact to the globe and
    the eye cannot rotate. Measured, with `standoff: 0` (the raw scanned geometry): the gaze
    range collapses to 5.9 / 6.0 / 7.7 degrees against commands of 25 / 16 / 9.

    Every point is moved radially to sit at least `target(s)` clear of the globe's surface,
    where the target tapers from `standoff` over the belly to `embed` (NEGATIVE -- the
    tendon bites into the sclera) over the distal `cap` fraction. Points already clear are
    left exactly where the artist put them.
    """
    loc = pts - np.asarray(centre)[None, :]
    r = np.linalg.norm(loc, axis=1).clip(1e-12)
    dirs = loc / r[:, None]
    r_surf = surface_radius(centre, dirs, shell, device=device).clip(1e-9)
    w = np.clip((s - (1.0 - cap)) / max(cap, 1e-6), 0.0, 1.0)      # 0 belly -> 1 insertion
    target = standoff * (1.0 - w) + embed * w
    need = (r_surf + target) - r
    push = np.maximum(need, 0.0)
    return pts + push[:, None] * dirs, float((push > 0).mean())


def fill_meshes(meshes, n, oversample=3.0, max_rounds=8, device=None):
    """`n` points uniform in the UNION of closed meshes, plus the volume they imply.

    Rejection sampling on the bounding box against a Halton sequence: uniform in the box ->
    uniform in whatever fraction of it the meshes occupy, deterministically and with no RNG
    to seed. The accepted fraction is also a Monte-Carlo estimate of the volume, which is
    what sizes the particles -- the stock provision sized `p_vol` for the ball the engine
    seeded, and the union of the globe's three tissues has no closed form anyway.

    The draw is ADAPTIVE because the fill fraction is not knowable in advance and spans an
    order of magnitude here: the globe fills 47% of its box, a curved muscle strap under 4%.
    A fixed oversample either wastes work or comes up short.
    """
    allv = np.concatenate([v for v, f in meshes])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    box_vol = float(np.prod(hi - lo))

    kept, n_try, n_hit, drawn = [], 0, 0, 0
    want = int(max(n * oversample, 4096))
    for _ in range(max_rounds):
        u = _halton(want, offset=drawn)
        drawn += want
        cand = lo[None, :] + u * (hi - lo)[None, :]
        ins = np.zeros(len(cand), bool)
        for v, f in meshes:                 # union: only test what no mesh has claimed yet
            todo = ~ins
            if not todo.any():
                break
            ins[todo] = inside_mesh(cand[todo], v, f, device=device)
        kept.append(cand[ins])
        n_try += len(cand)
        n_hit += int(ins.sum())
        got = sum(len(k) for k in kept)
        if got >= n:
            break
        rate = max(n_hit / max(n_try, 1), 1e-4)
        want = int(min(max((n - got) / rate * 1.35, 4096), 4_000_000))
    X = np.concatenate(kept)
    if len(X) < n:
        raise RuntimeError(f"fill_meshes: only {len(X)} of {n} points landed inside "
                           f"after {n_try} candidates ({100.0 * n_hit / max(n_try, 1):.1f}% fill)")
    vol = box_vol * n_hit / max(n_try, 1)
    return X[:n], vol


# --------------------------------------------------------------------------- #
#  4. blend_globe (seed @ mpm_particle)
# --------------------------------------------------------------------------- #
@register_operator("blend_globe", family="anatomy", set="particle", kind="seed")
class BlendGlobe(Seed):
    """Fill the scanned globe with material points and give each one its tissue.

    The globe is the UNION of three watertight meshes -- the retina shell (which is the
    whole eyeball's outer surface), the corneal cap that bulges beyond it, and the lens --
    so the material points fill a real eye rather than an ellipsoid of revolution, and the
    cornea is a bulge rather than a band of the same sphere.

    The labels are the same eight `eye_anatomy` uses, and they are assigned in the same
    spirit: LENS and CORNEA from the meshes that define them, SCLERA / CHOROID / VITREOUS
    from bands of the normalized ellipsoidal radius `rn`, PUPIL / IRIS / FLECK from the
    polar angle about the optic axis (cosmetic, and the flecks are what make torsion
    visible in the movie). Lens and cornea get their own Lame parameters.

    Buffers written -- these are the contract with the rest of the plant:
        tissue    per-point label id, indexing TISSUE_NAMES
        rest      rest OFFSETS from the globe centre, for `eye_pose`'s Kabsch fit
        rest_dir  rest unit directions
        rest_rn   normalized ellipsoidal radius, so `shell_min` still selects the shell
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True          # a seed writes the initial state, by definition
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["morphogenesis_static", "regional_identity", "material_heterogeneity",
                      "imported_geometry"]
    PARAM_ROLES = {"blend": "source_geometry", "side": "which_eye", "a_eq": "globe_scale",
                   "lens_youngs": "lens_stiffness", "cornea_youngs": "cornea_stiffness"}
    REFERENCE = "Plexus (this work); geometry: 260802_s2_EYE_MUSCLES_MODEL.blend (96 hpf zebrafish head)."

    TISSUE_NAMES = ["vitreous", "choroid", "sclera", "cornea", "iris", "fleck", "pupil", "lens"]
    VITREOUS, CHOROID, SCLERA, CORNEA, IRIS, FLECK, PUPIL, LENS = range(8)

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.blend = params.get("blend", DEFAULT_BLEND)
        self.parts_dir = params.get("parts", DEFAULT_PARTS)
        self.side = str(params.get("side", "R"))
        self.a_eq = float(params.get("a_eq", EA.A_EQ))
        self.center = [float(x) for x in params.get("center", EA.GLOBE_CENTER)]
        self.lens_youngs = float(params.get("lens_youngs", EA.LENS_YOUNGS))
        self.cornea_youngs = float(params.get("cornea_youngs", 320.0))
        self.pupil_deg = float(params.get("pupil_deg", EA.PUPIL_DEG))
        self.iris_deg = float(params.get("iris_deg", EA.IRIS_DEG))
        self.inflate = float(params.get("inflate", 1.0))       # grow the globe, not the straps
        self.density = float(params.get("density", 1.0))
        self.nu = float(params.get("poisson", 0.2))
        self._done = False

    def _lame(self, E):
        return E / (2 * (1 + self.nu)), E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        d, man = load_cut(self.blend, self.parts_dir)
        fr = BlendFrame(man, d, self.side, self.a_eq, self.center, self.inflate)

        def mesh(part):
            return fr.globe(d[f"{part}__v"]), np.asarray(d[f"{part}__f"], np.int64)

        retina, cornea, lens = (mesh(f"{self.side}_{k}") for k in ("retina", "cornea", "lens"))
        X, vol = fill_meshes([retina, cornea, lens], p.n, device=dev)

        c = np.asarray(self.center, float)
        loc = X - c[None, :]
        r = np.linalg.norm(loc, axis=1).clip(1e-12)
        dirs = loc / r[:, None]
        # NORMALIZED RADIUS, measured against the globe's own surface along each point's own
        # direction rather than against a fitted ellipsoid -- see `surface_radius`. rn = 1 on
        # the surface, so EA's bands (vitreous 0.60, choroid 0.88, shell 0.94) and
        # `eye_pose`'s `shell_min` keep the meaning they had on the analytic ovoid.
        rn = np.clip(r / surface_radius(c, dirs, [retina, cornea], device=dev).clip(1e-9), 0.0, 1.0)

        in_lens = inside_mesh(X, *lens, device=dev)
        in_cornea = inside_mesh(X, *cornea, device=dev) & ~in_lens
        polar = np.degrees(np.arccos(np.clip(dirs[:, 2], -1.0, 1.0)))
        azim = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0])) % 360.0

        tissue = np.full(p.n, self.VITREOUS, np.int64)
        tissue[rn > EA.R_VITREOUS] = self.CHOROID
        tissue[rn > EA.R_INNER] = self.SCLERA
        shell = rn > EA.R_SHELL
        on_iris = shell & (polar < self.iris_deg) & (polar >= self.pupil_deg)
        fleck = np.zeros_like(on_iris)
        for a0 in EA.IRIS_FLECK_DEG:
            fleck |= on_iris & (np.abs((azim - float(a0) + 180.0) % 360.0 - 180.0)
                                < EA.IRIS_FLECK_WIDTH_DEG)
        tissue[shell & (polar < self.pupil_deg)] = self.PUPIL
        tissue[on_iris] = self.IRIS
        tissue[fleck] = self.FLECK
        tissue[in_cornea] = self.CORNEA
        tissue[in_lens] = self.LENS

        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = new

        mu, la = p.mu.clone(), p.la.clone()
        for sel, E in ((in_lens, self.lens_youngs), (in_cornea, self.cornea_youngs)):
            s = torch.as_tensor(sel, device=dev)
            m_, l_ = self._lame(E)
            mu = torch.where(s, torch.full_like(mu, m_), mu)
            la = torch.where(s, torch.full_like(la, l_), la)
        p.mu, p.la = mu, la
        p.p_vol = torch.full((p.n,), vol / p.n, device=dev)
        p.mass = p.p_vol * self.density

        p.register_buffer("tissue", torch.as_tensor(tissue, device=dev))
        p.register_buffer("rest", torch.as_tensor(loc, dtype=torch.float32, device=dev))
        p.register_buffer("rest_dir", torch.as_tensor(dirs, dtype=torch.float32, device=dev))
        p.register_buffer("rest_rn", torch.as_tensor(rn, dtype=torch.float32, device=dev))
        print(f"[blend_globe] {fr.describe()}; {p.n} points, volume {vol:.5f}, "
              f"{int(shell.sum())} on the shell, {int(in_lens.sum())} lens, "
              f"{int(in_cornea.sum())} cornea", flush=True)
        self._done = True
        return {}


# --------------------------------------------------------------------------- #
#  5. blend_muscles (seed @ muscle_particle)
# --------------------------------------------------------------------------- #
@register_operator("blend_muscles", family="anatomy", set="muscle_particle", kind="seed")
class BlendMuscles(Seed):
    """Fill the six scanned straps with material points, fibres and attachments.

    Each muscle's points come from its own closed surface, and its FIBRE ARCHITECTURE from
    the centreline measured with it (`read_blend.muscle_geometry`): a point is projected
    onto the polyline, which gives it the fibre coordinate `s` -- 0 at the ORIGIN, 1 at the
    INSERTION, the convention `muscle_morphogenesis` set and `muscle_geometry`,
    `muscle_contract` and `muscle_sleeve` all read -- and the local tangent there is the
    fibre direction the active stress acts along.

    The two caps are the muscle's two attachments: `anchored`, the proximal `cap` fraction
    that `bone_anchor` pins to the skull, and `tendon`, the distal fraction embedded in the
    sclera. Nothing is embedded by hand here: the strap already reaches the globe, because
    the artist drew it there (its insertion end sits 0.04-0.10 blend units off the retina,
    i.e. under a fifth of a globe radius, and the shared MLS-MPM grid welds it).

    Per-muscle volume comes from the mesh, so the six particle clouds carry the six
    measured cross-sections -- see the module docstring on why that IS the strength table.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MAPS = ["parent"]
    MECHANISM_TAGS = ["morphogenesis_static", "fibre_architecture", "tendon_attachment",
                      "imported_geometry"]
    PARAM_ROLES = {"blend": "source_geometry", "side": "which_eye", "cap": "attachment_cap_fraction",
                   "youngs": "muscle_stiffness"}
    REFERENCE = "Plexus (this work); geometry: 260802_s2_EYE_MUSCLES_MODEL.blend (96 hpf zebrafish head)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.blend = params.get("blend", DEFAULT_BLEND)
        self.parts_dir = params.get("parts", DEFAULT_PARTS)
        self.side = str(params.get("side", "R"))
        self.a_eq = float(params.get("a_eq", EA.A_EQ))
        self.center = [float(x) for x in params.get("center", EA.GLOBE_CENTER)]
        self.keys = list(params.get("keys", MUSCLE_KEYS))
        self.cap = float(params.get("cap", 0.10))
        self.youngs = float(params.get("youngs", 240.0))
        self.density = float(params.get("density", 1.0))
        self.nu = float(params.get("poisson", 0.2))
        # the belly must ride clear of the sclera; the tendon must bite into it. 0 keeps the
        # artist's geometry exactly, welds and all -- see `relieve_overlap`.
        self.standoff = float(params.get("standoff", 0.008))
        self.embed = float(params.get("embed", -0.006))
        self.inflate = float(params.get("inflate", 1.0))       # must match blend_globe
        self.bins = int(params.get("bins", 14))            # match `muscle_geometry`'s readout
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        d, man = load_cut(self.blend, self.parts_dir)
        fr = BlendFrame(man, d, self.side, self.a_eq, self.center, self.inflate)
        par = p.parent.detach().cpu().numpy()
        M = int(par.max()) + 1
        if M != len(self.keys):
            raise ValueError(f"the spec has {M} muscles but {len(self.keys)} keys: {self.keys}")

        # the globe as BUILT (inflated), because that is the surface the straps meet
        shell = [(fr.globe(d[f"{self.side}_{k}__v"]),
                  np.asarray(d[f"{self.side}_{k}__f"], np.int64))
                 for k in ("retina", "cornea")]
        X = np.zeros((p.n, 3))
        fib = np.zeros((p.n, 3))
        sarr = np.zeros(p.n)
        pvol = np.zeros(p.n)
        rest_len = np.zeros(M)
        report = []
        for mi, key in enumerate(self.keys):
            sel = np.nonzero(par == mi)[0]
            part = f"{self.side}_{key}"
            V = fr(d[f"{part}__v"])
            F = np.asarray(d[f"{part}__f"], np.int64)
            pts, vol = fill_meshes([(V, F)], sel.size, device=dev)

            # the INSERTION, measured in the cut, is where the fibre coordinate starts
            ins = fr(d[f"{part}__centreline__v"])[0]
            s, idx = path_coordinate(pts, ins)
            if self.standoff != 0.0 or self.embed != 0.0:
                pts, moved = relieve_overlap(pts, s, self.center, shell, self.standoff,
                                             self.embed, self.cap, device=dev)
                s, idx = path_coordinate(pts, ins)          # the push is small, but s is cheap
            else:
                moved = 0.0
            L, empty = binned_length(pts, s, self.bins)
            if empty:
                print(f"[blend_muscles] WARNING {key}: {empty} of {self.bins} bins empty -- "
                      f"`muscle_geometry` will read a broken length", flush=True)
            X[sel] = pts
            fib[sel] = fibre_from_s(pts, s, idx)
            sarr[sel] = s
            pvol[sel] = vol / sel.size
            rest_len[mi] = L
            # HOW FAR THE INSERTION SITS INSIDE THE GLOBE -- the quantity this whole
            # geometry argument is about, so it is measured and printed, not assumed.
            tip = pts[s > 1.0 - self.cap]
            inside = 0.0
            if len(tip):
                loc = tip - np.asarray(self.center)[None, :]
                r = np.linalg.norm(loc, axis=1).clip(1e-12)
                rs = surface_radius(self.center, loc / r[:, None], shell, device=dev)
                inside = float(np.mean(np.maximum(rs - r, 0.0)))
            report.append(f"{key}={L:.3f}(moved {100 * moved:.0f}%, tendon {inside:+.4f} in)")

        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = new
        mu = self.youngs / (2 * (1 + self.nu))
        la = self.youngs * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        p.mu = torch.full((p.n,), mu, device=dev)
        p.la = torch.full((p.n,), la, device=dev)
        p.p_vol = torch.as_tensor(pvol, dtype=torch.float32, device=dev)
        p.mass = p.p_vol * self.density
        p.register_buffer("fibre", torch.as_tensor(fib, dtype=torch.float32, device=dev))
        p.register_buffer("s", torch.as_tensor(sarr, dtype=torch.float32, device=dev))
        p.register_buffer("rest", torch.as_tensor(X, dtype=torch.float32, device=dev))
        p.register_buffer("anchored", torch.as_tensor(sarr < self.cap, device=dev))
        p.register_buffer("tendon", torch.as_tensor(sarr > 1.0 - self.cap, device=dev))
        p.register_buffer("active_stress", torch.zeros(p.n, 3, 3, device=dev))
        m = H.level(p.parent_name)
        m.register_buffer("rest_length",
                          torch.as_tensor(rest_len, dtype=torch.float32, device=dev))
        print(f"[blend_muscles] {fr.describe()}; {M} straps x {p.n // M} points; "
              f"standoff {self.standoff:+.3f} / embed {self.embed:+.3f}; rest lengths "
              + " ".join(report), flush=True)
        self._done = True
        return {}


# --------------------------------------------------------------------------- #
#  what the spec needs to know about the blend, without running the engine
# --------------------------------------------------------------------------- #
def plant(blend=DEFAULT_BLEND, parts_dir=DEFAULT_PARTS, side="R",
          a_eq=EA.A_EQ, center=EA.GLOBE_CENTER, keys=None, inflate=1.0):
    """Where the muscles start and how big the globe is, in SIM units.

    `eye_spec` writes a `start` position per muscle so the engine seeds each strap's points
    near the strap; this returns those, plus the numbers the socket and the drive need
    (cup radius, globe semi-axes). Read by `run_eye_G.build_spec`.
    """
    d, man = load_cut(blend, parts_dir)
    fr = BlendFrame(man, d, side, a_eq, center, inflate)
    keys = list(keys or MUSCLE_KEYS)
    rec = {r["part"]: r for r in man["parts"]}
    starts, lengths = [], []
    for k in keys:
        part = f"{side}_{k}"
        line = fr(d[f"{part}__centreline__v"])
        starts.append(line.mean(axis=0))
        lengths.append(float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()))
    return dict(frame=fr, starts=np.asarray(starts), rest_lengths=np.asarray(lengths),
                semi=fr.semi, a_eq=float(a_eq), scale=fr.scale,
                axial_ratio=fr.axial_ratio, side=side,
                volumes={k: rec[f"{side}_{k}"]["volume"] * fr.scale ** 3 for k in keys})

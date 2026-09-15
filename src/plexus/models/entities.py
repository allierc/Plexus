"""Entity semantics: the per-node STATE SCHEMA and RENDER hints, in the registry.

This is the single source of truth for "what a node's state columns mean". The
engine reads `state_schema` to size and slice a set's state; operators read named
blocks (`lvl.get('pos')`, `lvl.get('vel')`) instead of hardcoding `[:, :2]`; the
plotter reads `render` (color-by, arrows) so it draws generically. A new entity
declares its layout here once and everyone downstream just works.

  state_schema : how this entity's state is laid out (see the three forms below)
  render       : {color_by: <per-node int field>, arrows: <vector block or None>}

THE THREE ACCEPTED FORMS OF `state_schema`, and only two of them are honoured:

  StateSchema           a fixed layout, dimension-independent -- for a set whose
                        state does not scale with the world's dimension.
  dim -> StateSchema    a CALLABLE, resolved at build time with the run's `dim`.
                        This is what every spatial entity uses: `spatial_schema`.
  {block: (c0, c1)}     the LEGACY dict. A hint only. NEVER honoured as a schema.

WHY THE LEGACY DICT IS NOT HONOURED, which is the whole reason this file changed.
Until this commit the engine never consulted the registry at all: `_resolve_schema`
was `if "state" in s: ... else spatial_schema(D)`, and all three `_entity_meta` call
sites threw the schema away (`_, render, depth = ...`). So every dict here was dead
code -- and, being written `{"pos": (0, 2), "vel": (2, 4)}`, it was dead code that
hard-codes TWO dimensions. The engine has been dimension-generic since the `dim`
contract landed, and six entities in `src/plexus/` carried that 2D dict while being
used in 3D runs (`mpm_block`, `basement_membrane_particle`, `integrin_particle` are
all ECM/membrane sets, and those specs are `dim: 3`). Making the registry live and
honouring the dict would therefore have TRUNCATED every one of those runs from
`[pos_xyz | vel_xyz]` to `[pos_xy | vel_xy]` -- silently, since a shorter state
tensor raises nothing. So the dict stays a hint, the callable is the way to declare
a spatial layout, and the six that carried a dict are ported below and in
`ecm_ops` / `membrane_ops`. `spatial_schema(D)` is exactly what the engine already
substituted for them, so the port moves no byte -- which is what makes it checkable
by bit-equality (`tools/promotion_identical.py --phase A`) rather than by argument.

(`prototype/eye/muscle_ops.py` still carries one; it is outside `src/plexus/` and is
left for its own commit.)

Importing this module registers the entities. The engine imports it alongside the
operator library.
"""
from __future__ import annotations

import math

import torch

from plexus.models.registry import register_entity
from plexus.models.state import (
    Block, StateSchema, spatial_schema,
    NONE, FIRST_ORDER, BOUNDARY_FREE, BOUNDARY_WORLD,
)


@register_entity(
    "particle", depth=0,
    state_schema=spatial_schema,                 # dim -> StateSchema (pos|vel, D-wide each)
    render={"color_by": "node_type", "arrows": "vel"},
)
class Particle:
    """A point with position + velocity (the interacting-particle / boid leaf)."""


_NU = 0.2                          # Poisson ratio (shared; near-incompressible MPM materials)



# ---------------------------------------------------------------------------------------------
# PLACEMENT, IN FIVE WORDS. A body is where it is, how it is turned, whether it is hollow and how
# many of it there are; every one of these was previously either impossible or a hand-written
# type per copy (`si_multimaterial_27` spells 27 cubes as 27 types, 90 lines that differ in a
# corner and a colour). They are per-TYPE properties read in one place, so the vocabulary grows
# by five words and the operator by one function:
#
#   shape: cube | ball | cylinder | obj     the body's form; `aspect` is a cylinder's length/diameter
#   hollow: f                               f of the radius (or of the box) is EMPTY: a shell, a pipe
#   rotate: [rx, ry, rz]                    degrees about the body's own centre, x then y then z
#   repeat: [nx, ny, nz] (+ pitch)          the body tiled into a lattice of copies, points split evenly
#   scatter: [x0,y0,z0,x1,y1,z1]            the copies' centres drawn in that box instead of tiled
#
# All four act on OFFSETS from a body's centre, which is why they compose: a hollow rotated
# cylinder repeated nine times is four words, and nothing about the volume contract changes --
# `V = per_parent * p_vol` still fixes the size, `repeat` divides the points among the copies.
def _ATLAS_FORMS():
    """The atlas's shape names, imported lazily: `models` must not import `operators` at load."""
    try:
        from plexus.operators.cell_ops import ATLAS_FORMS
        return tuple(n for n in ATLAS_FORMS if n not in ("ball",))   # `ball` is the seeder's own
    except Exception:                                                # noqa: BLE001
        return ()


def _rot_matrix(deg, D, device):
    r = [math.radians(float(v)) for v in (list(deg) + [0.0, 0.0, 0.0])[:3]]
    if D == 2:
        c, s = math.cos(r[2]), math.sin(r[2])
        return torch.tensor([[c, -s], [s, c]], device=device, dtype=torch.float32)
    cx, sx, cy, sy, cz, sz = (math.cos(r[0]), math.sin(r[0]), math.cos(r[1]),
                              math.sin(r[1]), math.cos(r[2]), math.sin(r[2]))
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    return Rz @ Ry @ Rx


def _unit_offsets(kind, n, D, H, device, aspect=2.0, hollow=0.0, axis=2, fill=None):
    """`n` points in a unit body centred on the origin: a cube of side 1, a ball of radius 1, a
    cylinder of radius 1 and length `aspect` * 2 along `axis`, or a torus of ring radius 1 and tube
    radius `aspect` about `axis`. `hollow` empties the inner fraction, so 0.7 on a ball is a shell,
    on a cylinder a pipe, and on a torus a doughnut whose dough is a pipe. Scaled by the caller."""
    h = min(max(float(hollow), 0.0), 0.98)
    if kind in ("ball", "sphere"):
        u = torch.randn(n, D, generator=H.rng, device=device)
        u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-12)
        f = torch.rand(n, 1, generator=H.rng, device=device)
        r = (h ** D + (1.0 - h ** D) * f) ** (1.0 / D)        # uniform in the shell h..1
        return u * r
    if kind == "torus":
        # A DOUGHNUT: a ring of radius 1 about `axis`, of tube radius `aspect` (in units of that
        # ring radius), and `hollow` empties the tube's own core -- so 0.6 is a doughnut whose
        # dough is itself a pipe. The hole points along `axis`.
        if D != 3:
            raise ValueError("shape: torus is a 3D body; in 2D a ring is `shape: ball` with `hollow`")
        ax = int(axis) % D
        lat = [i for i in range(D) if i != ax]
        a = max(float(aspect), 1e-6)
        # UNIFORM IN VOLUME, NOT IN THE TUBE ANGLE. The ring of material at tube angle th sits at
        # distance 1 + a cos(th) from the axis, so its circumference -- and the volume it carries --
        # is proportional to that. Drawing th uniformly would pile mass on the INSIDE of the
        # doughnut, where the circumference is smallest, which an MPM run reads as a density step.
        # One rejection pass with acceptance (1 + a cos th) / (1 + a) fixes it.
        th = torch.empty(0, device=device)
        for _ in range(24):
            if int(th.numel()) >= n:
                break
            cand = torch.rand(int((n - int(th.numel())) * 1.6) + 32, generator=H.rng,
                              device=device) * 2.0 * math.pi
            u = torch.rand(int(cand.numel()), generator=H.rng, device=device)
            th = torch.cat([th, cand[u <= (1.0 + a * torch.cos(cand)) / (1.0 + a)]])
        th = th[:n]
        phi = torch.rand(n, generator=H.rng, device=device) * 2.0 * math.pi
        f = torch.rand(n, generator=H.rng, device=device)
        r = a * (h ** 2 + (1.0 - h ** 2) * f) ** 0.5          # the annulus h..1 ACROSS THE TUBE
        rad = 1.0 + r * torch.cos(th)
        out = torch.zeros(n, D, device=device)
        out[:, lat[0]] = rad * torch.cos(phi)
        out[:, lat[1]] = rad * torch.sin(phi)
        out[:, ax] = r * torch.sin(th)
        return out
    if kind == "cylinder":
        ax = int(axis) % D
        lat = [i for i in range(D) if i != ax]
        out = torch.zeros(n, D, device=device)
        th = torch.rand(n, generator=H.rng, device=device) * 2.0 * math.pi
        f = torch.rand(n, generator=H.rng, device=device)
        r = (h ** 2 + (1.0 - h ** 2) * f) ** 0.5              # uniform in the annulus h..1
        out[:, lat[0]] = r * torch.cos(th)
        if len(lat) > 1:
            out[:, lat[1]] = r * torch.sin(th)
        out[:, ax] = (torch.rand(n, generator=H.rng, device=device) - 0.5) * 2.0 * float(aspect)
        return out
    if str(fill or "").lower() == "lattice":
        # `fill: lattice` ON A SHAPE, not only on a `block`. The block path has had it since the
        # 27-cube scene was written; saying the same 27 cubes as ONE type with `repeat` meant
        # asking for a cube whose points sit on a grid, and `shape: cube` could only draw them at
        # random -- which is a different first frame, not a different spelling of the same one.
        # Same construction as the block path: k = round(n^(1/D)) per axis, points at the cell
        # CENTRES, and a jitter of a fifth of a cell so no two land on one line of the MPM grid.
        k = max(1, int(round(n ** (1.0 / D))))
        ax = [(torch.arange(k, device=device, dtype=torch.float32) + 0.5) / k for _ in range(D)]
        g = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, D)
        if g.shape[0] < n:                                         # not a perfect power: pad at random
            g = torch.cat([g, torch.rand(n - g.shape[0], D, generator=H.rng, device=device)], 0)
        g = g[:n] + (torch.rand(n, D, generator=H.rng, device=device) - 0.5) * (0.2 / k)
        return g.clamp(0.0, 1.0) - 0.5
    u = torch.rand(n, D, generator=H.rng, device=device) - 0.5     # a cube of side 1
    if h > 0:                                                     # a box shell: push points outward
        k = u.abs().max(dim=1, keepdim=True).values.clamp(min=1e-9)
        want = 0.5 * (h + (1.0 - h) * torch.rand(n, 1, generator=H.rng, device=device))
        u = u * (want / k)
    return u


def _place(t, off, cps, n, D, H, device, out=None):
    """A body's offsets, TURNED, then assigned to its copies. `rotate` is degrees about the body's
    own centre; the copies come from `repeat`/`pitch`, or from `scatter`, a box the centres are
    drawn in (a grain bed: many bodies of one type, nowhere in particular)."""
    if t.get("rotate") is not None:
        off = off @ _rot_matrix(t["rotate"], D, device).T
    radial = str(t.get("orient", "")).lower() == "radial" and D == 3
    sc = t.get("scatter")
    if sc is not None:
        v = [float(x) for x in sc]
        lo = torch.tensor(v[:D], device=device); hi = torch.tensor(v[D:2 * D], device=device)
        k = int(cps.shape[0])
        c = lo + torch.rand(k, D, generator=H.rng, device=device) * (hi - lo)
        c = c - c.mean(0)                                    # relative to the body's own centre
    else:
        c = cps
    if int(c.shape[0]) <= 1:
        if out is not None:
            out["copy"] = torch.zeros(n, device=device)
        return off
    which = torch.arange(n, device=device) % int(c.shape[0])  # the points split evenly among copies
    if radial:
        # EACH COPY TURNED TO FACE OUTWARD. A ring of rods all pointing the same way is a bundle;
        # a rosette, a corolla or a shell of organelles means every piece's own axis points AWAY
        # from the centre. Rotate each copy's offsets from the body axis onto its own radius --
        # the shortest rotation, so nothing spins about its own length for no reason.
        ax = {"x": 0, "y": 1, "z": 2}.get(str(t.get("axis", "z")).lower(), 2)
        a = torch.zeros(D, device=device); a[ax] = 1.0
        d = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)      # each copy's outward direction
        v = torch.cross(a.expand_as(d), d, dim=1)
        cth = (a.expand_as(d) * d).sum(1, keepdim=True)
        K = torch.zeros(len(d), D, D, device=device)
        K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
        K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
        R = torch.eye(D, device=device).expand(len(d), D, D) + K + K @ K / (1.0 + cth).clamp_min(1e-9)[:, :, None]
        off = torch.einsum("nij,nj->ni", R[which], off)
    if out is not None:
        # WHICH COPY A POINT BELONGS TO, when the set asks for it (`state: {copy: {width: 1}}`).
        # Colour is a TYPE property, so ten copies of one type were one colour; a per-particle copy
        # index is the honest way to tell them apart -- `plotting.color_field: copy` then paints
        # each body its own hue without splitting one type into ten.
        out["copy"] = which.to(torch.float32)
    return off + c[which]


def _write_copy(lvl, mask, sink):
    """Record the copy index on the particles, if the set declared a `copy` block for it."""
    if "copy" not in sink or "copy" not in getattr(lvl, "state_schema", {}):
        return
    a, b = lvl.state_schema["copy"]
    lvl.state[mask, a:b] = sink["copy"].reshape(-1, 1).to(lvl.state.dtype)


def _copies(t, D, device):
    """The centres of a type's copies, relative to its own centre. Returns [k, D].

      repeat: [nx,ny,nz] (+ pitch)   a lattice of copies
      ring: {n, radius, axis}        `n` copies evenly around a circle -- a rosette, a corolla, a
                                     wheel of spokes, twelve organelles about a cell's centre
    """
    sh = t.get("shell")
    if sh is not None:
        # COPIES OVER A SPHERE, on a Fibonacci spiral. A random draw on a sphere clumps -- the atlas
        # says so and uses the spiral for exactly this -- and a shell of organelles that clumps
        # reads as a scatter, not as a lining.
        r = dict(sh) if isinstance(sh, dict) else {"n": int(sh)}
        n = max(1, int(r.get("n", 1)))
        rad = float(r.get("radius", 1.0))
        i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
        phi = torch.acos(1.0 - 2.0 * i / n)                       # uniform in area
        gold = math.pi * (1.0 + 5.0 ** 0.5)
        th = gold * i
        out = torch.zeros(n, D, device=device)
        out[:, 0] = rad * torch.cos(th) * torch.sin(phi)
        if D > 1:
            out[:, 1] = rad * torch.cos(phi)
        if D > 2:
            out[:, 2] = rad * torch.sin(th) * torch.sin(phi)
        return out
    ring = t.get("ring")
    if ring is not None:
        r = dict(ring) if isinstance(ring, dict) else {"n": int(ring)}
        n = max(1, int(r.get("n", 1)))
        rad = float(r.get("radius", 1.0))
        ax = {"x": 0, "y": 1, "z": 2}.get(str(r.get("axis", "y")).lower(), 1) % D
        lat = [i for i in range(D) if i != ax]
        th = torch.arange(n, device=device, dtype=torch.float32) * (2.0 * math.pi / n) \
            + float(r.get("phase", 0.0)) * math.pi / 180.0
        out = torch.zeros(n, D, device=device)
        out[:, lat[0]] = rad * torch.cos(th)
        if len(lat) > 1:
            out[:, lat[1]] = rad * torch.sin(th)
        return out
    rep = t.get("repeat")
    if rep is None:
        return torch.zeros(1, D, device=device)
    rep = [int(v) for v in (list(rep) + [1, 1, 1])[:D]]
    pitch = t.get("pitch", 1.0)
    pitch = [float(pitch)] * D if isinstance(pitch, (int, float)) else [float(v) for v in (list(pitch) + [0, 0, 0])[:D]]
    axes = [(torch.arange(rep[i], device=device, dtype=torch.float32) - 0.5 * (rep[i] - 1)) * pitch[i]
            for i in range(D)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, D)


def _obj_points(t, nb, vol, D, H, device, cache, par=None):
    """`nb` points uniform inside the mesh named by `t["obj"]`, scaled to volume `vol`, centred at 0.

    Returns `(points [nb, D] on `device`, longest extent after scaling)`. 3-D only: an OBJ is a
    surface in space and a 2-D run has nothing for it to enclose.

    THE HOLE IS THE FIRST THING TO DEAL WITH. The Stanford bunny is open at its base (223 open
    edges) and the Utah teapot at its spout and lid (160); a ray from an interior point escapes
    through the opening and parity reports "outside" for half the body. `fill_holes` closes them,
    and `compute_normals(auto_orient_normals=True)` makes inside/outside consistent, which the
    enclosure test then relies on. The five files in papers/morph_models were checked: bunny
    and teapot need the fill, cow, spot and armadillo are watertight.

    REJECTION AGAINST THE SURFACE, NOT A VOXELISATION, so the sample is the solid and the density
    is uniform in it -- MPM resolves a material by how evenly its particles fill the cells, and a
    voxel grid would put an aliasing pattern into the first frame's stress.
    """
    import os
    import numpy as np
    import pyvista as pv
    if D != 3:
        raise ValueError(f"shape: obj needs a 3-D run, this one is {D}-D")
    name = str(t.get("obj", "")).strip()
    if not name:
        raise ValueError("shape: obj needs `obj: <name or path>` on the same type")
    path = name if (os.sep in name or name.endswith(".obj")) else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                         os.path.abspath(__file__))))), "papers", "morph_models", f"{name}.obj")
    if not os.path.exists(path):
        raise FileNotFoundError(f"shape: obj -- {path} does not exist")
    if path not in cache:
        m = pv.read(path).clean().triangulate().fill_holes(1e9).clean()
        m = m.compute_normals(auto_orient_normals=True, consistent_normals=True)
        if m.volume <= 0:
            raise ValueError(f"{path}: non-positive volume after hole filling -- surface not closed")
        cache[path] = m
    m = cache[path]
    lo, hi = np.array(m.bounds).reshape(3, 2).T
    ctr = 0.5 * (lo + hi)
    sc = (float(vol) / float(m.volume)) ** (1.0 / 3.0)         # so the SCALED mesh has volume `vol`
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=H.rng, device=device).item())
    rng = np.random.default_rng(seed)
    out, have = [], 0
    while have < nb:
        q = rng.random((4 * max(nb, 1), 3)) * (hi - lo) + lo
        sel = pv.PolyData(q).select_enclosed_points(m, tolerance=0.0, check_surface=False)
        keep = q[np.asarray(sel["SelectedPoints"]) > 0]
        if len(keep) == 0:
            raise ValueError(f"{path}: no interior points found -- is the surface closed?")
        out.append(keep); have += len(keep)
    pts = (np.concatenate(out)[:nb] - ctr) * sc
    # `obj_rotate: random` -- EACH BODY IN ITS OWN ORIENTATION, drawn once from the run's seed.
    # A lattice of identical bodies all facing the same way is a crystal, and a crystal falls as
    # one; sixty cows that each landed on the same hoof would say nothing about elasticity that
    # one cow does not. The rotation is uniform on SO(3) -- the Q of a QR decomposition of a
    # Gaussian matrix, sign-fixed so it is a rotation and not a reflection -- and it is drawn per
    # PARENT from `H.rng`, so the same seed gives the same sixty orientations every time.
    if str(t.get("obj_rotate", "none")).lower() in ("random", "rand", "true") and par is not None:
        par_np = np.asarray(par.detach().cpu().numpy() if hasattr(par, "detach") else par)
        rot = {}
        for k in np.unique(par_np):
            g = np.random.default_rng(
                int(torch.randint(0, 2 ** 31 - 1, (1,), generator=H.rng, device=device).item()))
            q, r = np.linalg.qr(g.standard_normal((3, 3)))
            q = q * np.sign(np.diag(r))[None, :]
            if np.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            rot[int(k)] = q
        for k, R in rot.items():
            sel = par_np == k
            pts[sel] = pts[sel] @ R.T
    return (torch.as_tensor(pts, dtype=torch.float32, device=device),
            float((hi - lo).max() * sc))


def _lame(E, nu: float = _NU):
    """Young's modulus E -> Lame parameters (mu shear, la bulk) at Poisson ratio nu."""
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, la


@register_entity(
    # THE SET IS NAMED FOR WHAT IT IS, NOT FOR HOW IT IS COMPUTED. `mpm_particle` names a
    # numerical method, which is exactly the separation the paper rests on -- semantics in the
    # language, discretisation in the implementation. `cytosol` and `nucleus` are two different
    # biological objects that happen to share a substrate, and naming them apart is what makes a
    # cell a COMPOSITION rather than one material at several stiffnesses. The old name stays first
    # and stays valid: 461 specs use it.
    "mpm_particle", "cytosol", "nucleus", "protein", depth=0,
    state_schema=spatial_schema,                 # dim -> StateSchema; MLS-MPM runs in 2D and 3D
    render={"color_by": "node_type", "arrows": None},
)
class MPMParticle:
    """A material point for MLS-MPM: position + velocity PLUS the per-particle
    continuum buffers the solver carries (deformation gradient F, affine velocity C,
    mass, Lame parameters mu/la, particle volume p_vol, material masks is_liquid/
    is_snow, plastic ratio Jp). These are provisioned at build time from the parent
    cell's per-type material config -- this is the entity-side half of the MPM
    subsystem (the operator side is the fenced `mls_mpm_mechanics`)."""

    @classmethod
    def provision(cls, lvl, parent, s, H, device):
        """Allocate the MPM per-particle buffers from the parent cell's per-type
        config: `youngs` -> mu/la; concentric `layers` (each {frac, youngs, material})
        -> radial material bands (liquid / snow / elastic); optional stiffer `core`;
        optional per-type `block` rectangle that the type FILLS (a pool/cube) instead
        of a disc. Mirrors the validated prototype build."""
        # A PARENT IS OPTIONAL, AND WITHOUT ONE THE SET DECLARES ITSELF.
        #
        # The container above an MPM cloud earns its place when it is a BODY -- a cell with a
        # nucleus, a jelly block next to a water pool -- because then `youngs`, `density`, `layers`
        # and `core` describe that body and the particles inherit them. It earns nothing when
        # there is exactly one of it and no operator touches it: a slab of gel indented by a plate
        # then declares a set of one element whose only job is to hold `types`, and the spec has to
        # name a body that is not in the model. `mpm_particle` may therefore carry its own `n` and
        # its own `types` and stand alone.
        #
        # WHAT THE SHIM DOES: every line below indexes per-parent arrays as `x[pidx]`. With no
        # parent each particle IS its own body, so `pidx` is the identity and the per-parent arrays
        # are per-particle. Nothing downstream changes, and the parented path is untouched.
        Np = lvl.n
        if parent is None:
            types = s.get("types") or {}
            type_list = list(types.values())
            ntp = getattr(lvl, "node_type", None)
            if ntp is None:
                ntp = torch.zeros(Np, dtype=torch.long, device=device)
            pidx = torch.arange(Np, device=device)
            n_par = Np
            # RADIAL BANDS NEED A CENTRE, AND THERE IS NONE. `layers` and `core` are concentric
            # shells measured from the parent's position; with no parent every particle sits at
            # r = 0 from itself, so `core` would swallow the whole cloud and `layers` would collapse
            # to their innermost shell -- silently, and looking like a uniform material.
            for _t in type_list:
                if _t.get("layers") or _t.get("core"):
                    raise ValueError(
                        f"{lvl.name}: `layers`/`core` are radial bands about a parent body's "
                        f"centre, and this set has no parent. Give it a `parent:` or declare the "
                        f"materials as separate types with their own `block:`.")
                if _t.get("block") is None:
                    raise ValueError(
                        f"{lvl.name}: a parentless MPM set is placed by its types' `block:` -- "
                        f"there is no parent centre to scatter a disc around. Add a `block: "
                        f"[x0,y0,z0,x1,y1,z1]` to every type, or give the set a `parent:`.")
        else:
            # THE NEAREST ANCESTOR THAT DECLARES `types`, not necessarily the immediate parent.
            #
            # Reading only `parent.types_raw` gives, with two levels, the same
            # thing; with `cell -> compartment -> mpm_particle` it is not, and the failure is
            # silent both ways round: a material declared on the CELL is invisible to a particle
            # whose parent is a compartment, and every particle then builds at the 100.0 default.
            # Walking up finds whichever level actually says what the body is made of, and stops
            # at the FIRST one -- the finer statement wins, the same precedence a child set's own
            # `types` already has over its parent's below.
            #
            # `pidx` is the COMPOSITION of the parent maps down from that ancestor, built here by
            # hand rather than through `H.lift_index` because `provision` runs inside `build`,
            # before this level is added to the hierarchy, so the chain cannot be walked from H.
            anc, pidx = parent, lvl.parent
            while not (getattr(anc, "types_raw", None) or {}):
                nxt_name = getattr(anc, "parent_name", None)
                if nxt_name is None or nxt_name not in H.levels:
                    break
                pidx = anc.parent[pidx]
                anc = H.level(nxt_name)
            types = getattr(anc, "types_raw", None) or {}
            type_list = list(types.values())
            ntp = anc.node_type if hasattr(anc, "node_type") else \
                torch.zeros(anc.n, dtype=torch.long, device=device)   # [Nc] per-body type id
            n_par = anc.n
            parent = anc                                         # radial `layers`/`core` are measured
                                                                 # from the body that declares them
        rho = float(s.get("density", 1.0)); rad = float(s.get("radius", 0.02))
        # DENSITY MAY VARY BY TYPE, and it has to for buoyancy to mean anything. It was a single
        # set-level scalar, so two species of different density needed two SETS -- two particle
        # clouds scattering to one grid, when what the physics wants is one cloud whose particles
        # differ. `youngs` and `material` were already per type; density is the third property of
        # the same kind and was the one left behind.
        # THE PARTICLE'S OWN TYPE, NOT ITS PARENT'S. `type_list` above is the PARENT's types --
        # it is what `youngs`/`layers`/`core` read, because those describe the CELL. Density is
        # different: two protein species of different density live inside ONE cell, so the
        # variation is between particles, not between their parents. The child set declares its
        # own `types` and the engine already assigns it a `node_type`; this reads that.
        # ... AND THE PARENT'S TYPE TOO, when the bodies themselves are what differ. The note
        # above is about two protein species inside ONE cell, where the variation really is between
        # particles. It is not the only case: a scene of a jelly block, a water pool and a snow
        # block declares those on the PARENT's types, so per-BODY density was unreachable and snow
        # could not be made to float on water. Parent-type density is applied first; a child-type
        # declaration still wins, being the finer statement.
        rho_p = rho
        if type_list and any("density" in t for t in type_list):
            rho_p = torch.as_tensor([float(t.get("density", rho)) for t in type_list],
                                    device=device, dtype=torch.float32)[ntp][pidx]
        _ct = list((s.get("types") or {}).values())
        _nt = getattr(lvl, "node_type", None)
        if _ct and _nt is not None and len(_ct) > 1 and any("density" in t for t in _ct):
            rho_p = torch.as_tensor([float(t.get("density", rho)) for t in _ct],
                                    device=device, dtype=torch.float32)[_nt]
        # HOW MANY POINTS SHARE ONE BODY, which is what `p_vol = V_body / ppc` divides by. It is a
        # SCALAR when `per_parent` is one number and a per-particle TENSOR when `per_parent` is a
        # mapping from the parent's type -- a nuclear-envelope patch of 50,000 points and a
        # plasma-membrane patch of 50 cannot share a divisor, and using one would give the membrane
        # points 1,000x the volume, hence 1,000x the mass, of the nucleus points beside them.
        # `build` computes the vector and passes it down as `_per_particle`.
        _ppv = s.get("_per_particle")
        ppc = _ppv.to(device) if _ppv is not None else \
            (int(s["per_parent"]) if parent is not None else Np)
        if _ppv is not None and s.get("particle_mass") is not None:
            raise ValueError(
                f"{lvl.name}: `particle_mass` sizes a body from ONE point mass and a per-type "
                f"`per_parent` gives each body a different point count, so the two disagree about "
                f"every body's volume. Declare one.")
        px0, px1 = lvl.state_schema["pos"]
        D = H.dim                                                # particle dimension (2D or 3D; the global dim contract)
        pos = lvl.state[:, px0:px1].clone()
        # WITH NO PARENT A PARTICLE IS ITS OWN CENTRE, so `r` is 0 everywhere -- which is why
        # `layers`/`core` are refused above rather than allowed to read it.
        cpos = parent.get("pos")[pidx] if parent is not None else pos
        r = (pos - cpos).norm(dim=1)                             # radial distance (for layer bands)

        # per-cell youngs / core / layers, broadcast to particles
        youngs_c = torch.full((n_par,), 100.0, device=device)
        core_y = torch.zeros(n_par, device=device); core_f = torch.zeros(n_par, device=device)
        type_layers = {}
        type_mat = {}
        for tid, t in enumerate(type_list):
            sel = ntp == tid
            youngs_c[sel] = float(t.get("youngs", 100.0))
            core = t.get("core")
            if core is not None:
                core_y[sel] = float(core["youngs"]); core_f[sel] = float(core.get("frac", 0.5))
            layers = t.get("layers")
            if layers is not None:
                type_layers[tid] = [(float(L["frac"]), float(L["youngs"]), L.get("material", "elastic"),
                                     float(L.get("tau", 0.0)))                  # tau: viscoelastic relaxation time
                                    for L in layers]
            elif t.get("material"):
                # `material` DIRECTLY ON THE TYPE, so a uniform body does not have to be written as
                # a one-element `layers` list. `layers` describes RADIAL STRUCTURE -- a nucleus
                # inside a cytosol inside a membrane -- and spelling a homogeneous snow block as
                # `layers: [{frac: 1.0, youngs: 90, material: snow}]` says "one concentric shell
                # covering the whole body", which is a true but unreadable way to say "this block
                # is snow". It also duplicated `youngs` at both levels, since the type-level value
                # is the fallback the layer overrides.
                type_mat[tid] = (t["material"], float(t.get("tau", 0.0)))

        # THE VOLUME CAN COME FROM THE MASS INSTEAD OF FROM A BOX. `per_parent` is the hierarchy's
        # own call -- how many children a parent has -- and it should not double as an MPM sampling
        # knob. So a spec may instead say what ONE PARTICLE WEIGHS, and the body's volume follows:
        #
        #     p_vol = particle_mass / density        V = per_parent * p_vol       side = V^(1/3)
        #
        # and the particles are seeded in a cube of that side centred on their parent. Declaring N
        # and m_p fixes the volume; declaring N and a block fixes the mass. Both together is one
        # statement too many, which is what `_volume_conflict` below is for.
        _pm = s.get("particle_mass")
        _mesh_cache: dict = {}                             # `shape: obj`: one load per file per build
        if _pm is not None and float(_pm) <= 0:
            raise ValueError(f"particle_mass must be > 0, got {_pm}")
        if _pm is not None:
            _side = None
            for tid, t in enumerate(type_list):
                if t.get("block") is not None:
                    continue                                  # an explicit box wins; see the check
                bm = ntp[pidx] == tid
                nb = int(bm.sum())
                if nb == 0:
                    continue
                _rho_t = float(t.get("density", rho if not torch.is_tensor(rho) else 1.0))
                _pv = float(_pm) / _rho_t
                _vol = float(ppc) * _pv                       # ppc here is per_parent, the count
                # THE SHAPE THE VOLUME TAKES. A cube by default, because it is what a `block` would
                # have given; `shape: ball` makes it a sphere of the same volume instead, which is
                # what you want for anything thrown, dropped or rolled. Either way the VOLUME is
                # the derived quantity and the shape only decides how it is arranged.
                _shape = str(t.get("shape", "cube")).lower()
                # THE VOLUME IS THE CONTRACT, THE SHAPE ONLY ARRANGES IT. `V = per_parent * p_vol`
                # is fixed by the mass; `repeat` divides both the points and the volume among the
                # copies, so nine cubes of one type weigh what one did.
                _cp = _copies(t, D, device)
                _k = int(_cp.shape[0])
                _volk = _vol / _k
                _hollow = float(t.get("hollow", 0.0) or 0.0)
                _fill_frac = max(1.0 - _hollow ** D, 1e-6)    # a shell holds less than its envelope
                if _shape in ("ball", "sphere", "cylinder", "torus"):
                    if _shape == "torus":
                        # `aspect` IS THE TUBE, as a fraction of the ring radius: 0.3 is a doughnut,
                        # 0.05 a wire hoop. V = 2 pi^2 R a^2 (1 - h^2) with a = aspect * R, so the
                        # ring radius follows from the volume the mass already fixed.
                        _asp = float(t.get("aspect", 0.3))
                        _ax = {"x": 0, "y": 1, "z": 2}.get(str(t.get("axis", "z")).lower(), 2)
                        _r = (_volk / (2.0 * math.pi ** 2 * _asp ** 2
                                       * max(1.0 - _hollow ** 2, 1e-6))) ** (1.0 / 3.0)
                        _side = 2.0 * _r * (1.0 + _asp)      # the outer diameter of the doughnut
                    elif _shape == "cylinder":
                        _asp = float(t.get("aspect", 2.0))
                        _ax = {"x": 0, "y": 1, "z": 2}.get(str(t.get("axis", "z")).lower(), 2)
                        # V = pi r^2 L (1 - h^2), L = 2 * aspect * r
                        _r = (_volk / (2.0 * math.pi * _asp * max(1.0 - _hollow ** 2, 1e-6))) ** (1.0 / 3.0) \
                            if D == 3 else (_volk / (4.0 * _asp * max(1.0 - _hollow, 1e-6))) ** 0.5
                        _side = 2.0 * _r * _asp
                    else:
                        _r = (_volk * 3.0 / (4.0 * math.pi * _fill_frac)) ** (1.0 / 3.0) if D == 3 \
                            else (_volk / (math.pi * _fill_frac)) ** 0.5
                        _side = 2.0 * _r
                        _asp, _ax = 2.0, 2
                    # UNIFORM IN THE BODY, not in the radius: `_unit_offsets` draws a direction and
                    # a radius with the right power, so the density is flat before anything moves.
                    _off = _unit_offsets(_shape, nb, D, H, device, aspect=_asp, hollow=_hollow, axis=_ax) * _r
                    _sink = {}
                    pos[bm] = cpos[bm] + _place(t, _off, _cp, nb, D, H, device, out=_sink)
                    _write_copy(lvl, bm, _sink)
                elif _shape == "cube":
                    _side = (_volk / _fill_frac) ** (1.0 / D)
                    if str(t.get("fill", "")).lower() == "lattice" and _k > 1:
                        # EVERY COPY GETS THE SAME LATTICE, which is what 27 hand-written blocks
                        # were. Building one lattice over the whole body and letting `_place` deal
                        # it out gives each copy a SCATTERED SUBSET of a 33^3 grid rather than its
                        # own 11^3 one: same count, same volume, and a body 3% wider whose points
                        # do not sit in planes. `_place` sends point i to copy i % k, so the
                        # per-copy lattice is interleaved to match.
                        _per = max(int(nb) // _k, 1)
                        _u = _unit_offsets("cube", _per, D, H, device, hollow=_hollow, fill="lattice")
                        _off = _u.repeat_interleave(_k, 0)[:nb] * _side
                        if _off.shape[0] < nb:                   # a count that does not divide evenly
                            _off = torch.cat([_off, _off[: nb - _off.shape[0]]], 0)
                    else:
                        _off = _unit_offsets("cube", nb, D, H, device, hollow=_hollow,
                                             fill=t.get("fill")) * _side
                    _sink = {}
                    pos[bm] = cpos[bm] + _place(t, _off, _cp, nb, D, H, device, out=_sink)
                    _write_copy(lvl, bm, _sink)
                elif _shape in _ATLAS_FORMS():
                    # THE ATLAS'S OWN SHAPES, AVAILABLE TO ANY BODY. `patch`, `sheet`, `rod`,
                    # `filament`, `cisterna`, `tubule`, `stack`, `mito` and `barrel` are closed-form
                    # samplers with closed-form volumes, written for `seed_cell_atlas` and reachable
                    # only through it; `plexus.operators.cell_ops.form_points` is the door in, so a
                    # material body can BE a filament or a Golgi stack and there is one
                    # implementation of each rather than two. The volume contract is untouched: the
                    # sampler is asked for a piece of volume `_volk` and returns it.
                    # THE FORM'S OWN WORDS LIVE UNDER `form:`, AND THEY HAVE TO. `layers` already
                    # means concentric MATERIAL bands on a type (`layers: [{frac, youngs}]`) and a
                    # Golgi stack means it as a count of discs -- one word, two meanings, and the
                    # spec cannot hold both at the top level. A nested block also keeps the
                    # schema's "property read by no operator" check honest: `bend`, `cristae` and
                    # `skew` are read HERE, by the sampler, not by any operator.
                    from plexus.operators.cell_ops import form_points
                    _fp = t.get("form")
                    if not isinstance(_fp, dict):
                        raise ValueError(
                            f"shape: {_shape} takes its proportions from a `form:` block -- e.g. "
                            f"`form: {{size: 0.3, thickness: 0.08}}`. `size` and `thickness` are "
                            f"the piece's PROPORTIONS; its absolute scale comes from the volume.")
                    _off = form_points(_shape, nb, _fp, _volk, copies=_k, device=device,
                                       seed=int(getattr(H, "seed", 0)) + tid).to(pos.dtype)
                    _side = 2.0 * float(_off.abs().max()) if nb else 0.0
                    _sink = {}
                    pos[bm] = cpos[bm] + _place(t, _off, _cp, nb, D, H, device, out=_sink)
                    _write_copy(lvl, bm, _sink)
                elif _shape.startswith("mesh:") or _shape in ("obj", "mesh"):
                    # `shape: obj` -- THE VOLUME TAKES THE SHAPE OF A MESH FILE. Same contract as
                    # ball and cube: `V = per_parent * p_vol` is fixed by the mass, and the mesh is
                    # scaled so that ITS volume equals V, so a bunny and a cow of the same
                    # `particle_mass` weigh the same and displace the same, whatever their extent.
                    # Scaling to a longest axis instead -- what `morph.sample_inside` does for the
                    # morphing script -- would make the density a function of the file's aspect
                    # ratio, which is not a property of any material.
                    #
                    # `obj:` is a bare name under papers/morph_models, or a path. The mesh is loaded
                    # once per spec however many bodies wear it, and the points are drawn with a
                    # numpy generator seeded from `H.rng`, so the run stays reproducible under the
                    # same seed it always had.
                    if _shape.startswith("mesh:"):
                        # THE LIBRARY, NOT A PATH. `form: mesh:<name>[/<part>]` asks `plexus.shapes`
                        # for points inside that shape, scaled so ITS volume is the volume the mass
                        # fixes -- the same contract cube and ball keep, and the same one `obj:`
                        # had, now with a folder, a cache and a provenance behind the name.
                        from plexus import shapes as _shapes
                        import torch as _t
                        _pp = _shapes.points(_shape.split(":", 1)[1], int(nb), float(_volk))
                        _pts = _t.as_tensor(_pp, dtype=pos.dtype, device=device)
                        _sink = {}
                        _pts = _place(t, _pts, _cp, nb, D, H, device, out=_sink)
                        _write_copy(lvl, bm, _sink)
                        _side = float(abs(_pp).max() * 2.0)   # numpy is not imported in this module
                    else:
                        _pts, _side = _obj_points(t, nb, _vol, D, H, device, _mesh_cache,
                                                  par=pidx[bm])
                    pos[bm] = cpos[bm] + _pts
                else:
                    raise ValueError(f"shape must be 'cube', 'ball', 'cylinder', 'obj' or "
                                     f"'mesh:<name>', got {_shape!r}")
            if _side is not None:
                print(f"[build] {lvl.name}: particle_mass {float(_pm):.4g} / density -> p_vol "
                      f"{float(_pm) / float(rho if not torch.is_tensor(rho) else 1.0):.4g}, "
                      f"{ppc:,} per parent -> body side {_side:.6g} (volume {_side ** D:.4g})",
                      flush=True)

        # block-fill: a type FILLS an axis-aligned box (pool/cube) instead of a disc
        # around the centre. 2D block = [x0,y0,x1,y1]; 3D block = [x0,y0,z0,x1,y1,z1].
        for tid, t in enumerate(type_list):
            blk = t.get("block")
            if blk is None:
                continue
            bm = ntp[pidx] == tid
            nb = int(bm.sum())
            if nb == 0:
                continue
            v = [float(x) for x in blk]
            lo = torch.tensor(v[:D], device=device); hi = torch.tensor(v[D:2 * D], device=device)
            # A BOX IS A CUBE WITH A SIZE PER AXIS, so it takes the same four words. Turned or
            # tiled, the points are built as offsets from the box's centre and put back.
            if t.get("hollow") or t.get("rotate") is not None or t.get("repeat") is not None or t.get("scatter"):
                ctr, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
                _cp = _copies(t, D, device)
                _u = _unit_offsets("cube", nb, D, H, device, hollow=float(t.get("hollow", 0.0) or 0.0))
                _u = _u / max(int(_cp.shape[0]), 1) ** (1.0 / D)      # the copies share the box
                pos[bm] = ctr + _place(t, _u * (2.0 * half), _cp, nb, D, H, device)   # pitch is a world length
                continue
            if str(t.get("fill", "random")).lower() == "lattice":
                # `fill: lattice` -- THE POINTS ON A REGULAR GRID inside the box (the cube root of
                # the count per axis, a small jitter so no two lie on one grid line of the MPM
                # grid), which reads as a solid body where a uniform draw reads as a cloud.
                k = max(1, int(round(nb ** (1.0 / D))))
                ax = [(torch.arange(k, device=device, dtype=torch.float32) + 0.5) / k for _ in range(D)]
                g = torch.stack(torch.meshgrid(*ax, indexing="ij"), -1).reshape(-1, D)
                if g.shape[0] < nb:                      # the count is not a perfect power: pad at random
                    g = torch.cat([g, torch.rand(nb - g.shape[0], D, generator=H.rng, device=device)], 0)
                g = g[:nb] + (torch.rand(nb, D, generator=H.rng, device=device) - 0.5) * (0.2 / k)
                pos[bm] = lo + g.clamp(0.0, 1.0) * (hi - lo)
                continue
            u = torch.rand(nb, D, generator=H.rng, device=device)
            pos[bm] = lo + u * (hi - lo)
        lvl.state[:, px0:px1] = pos                              # commit block positions

        # per-particle stiffness + material masks (inner->outer radial bands)
        is_core = (core_y[pidx] > 0) & (r < core_f[pidx] * rad)
        p_y = torch.where(is_core, core_y[pidx], youngs_c[pidx])
        # A CHILD SET'S OWN TYPES SET ITS OWN MATERIAL, and until now they did not. Everything
        # above reads `type_list`, which is the PARENT's types -- correct for `layers` and `core`,
        # which describe how a CELL is built in radial bands, and wrong for a composition of
        # several distinct child sets, where each set IS a material.
        #
        # MEASURED: a spec declaring nucleus youngs 4000 / elastic beside cytosol youngs 15 /
        # liquid produced mu = 16.67 and is_liquid = 0.00 for BOTH -- the cell type's youngs 40,
        # twice. The cytosol was never a liquid and the nucleus was never stiff, so three
        # compartments that were supposed to be three substrates were one material wearing three
        # colours. It is why changing either value changed nothing: two runs of cell_03 with
        # youngs 300 and 4000 gave bit-identical shape statistics.
        _ct = list((s.get("types") or {}).values())
        _cnt = getattr(lvl, "node_type", None)
        _own_mat = {}
        if _ct and _cnt is not None:
            _cy = torch.as_tensor([float(t.get("youngs", 100.0)) for t in _ct],
                                  device=device, dtype=p_y.dtype)[_cnt]
            p_y = torch.where(is_core, core_y[pidx], _cy)     # `core` still overrides, per parent
            for _tid, _t in enumerate(_ct):
                _own_mat[_tid] = (_t.get("material", "elastic"), float(_t.get("tau", 0.0)))
        is_liquid = torch.zeros(Np, dtype=torch.bool, device=device)
        is_snow = torch.zeros(Np, dtype=torch.bool, device=device)
        is_visco = torch.zeros(Np, dtype=torch.bool, device=device)          # viscoelastic (Maxwell) band
        visco_tau = torch.full((Np,), 1e9, device=device)                    # 1e9 = no relaxation (pure elastic)

        def _mark(mat, sel_band, tau):                                       # set the material mask(s) for a band
            nonlocal is_liquid, is_snow, is_visco, visco_tau
            if mat == "liquid":
                is_liquid = is_liquid | sel_band
            elif mat == "snow":
                is_snow = is_snow | sel_band
            elif mat == "viscoelastic":
                is_visco = is_visco | sel_band
                visco_tau = torch.where(sel_band, torch.full_like(visco_tau, max(tau, 1e-6)), visco_tau)

        # the child's own `material`, applied per type -- the same precedence as its `youngs`.
        # `layers` still wins where a parent declares them, because a layered CELL is a statement
        # about radial structure that a flat per-set material cannot express.
        for _tid, (_mat, _tau) in _own_mat.items():
            if _mat and _mat != "elastic":
                _mark(_mat, (_cnt == _tid), _tau)
        # uniform parent types: no radial bands, the whole body is one material
        for _tid, (_m, _tu) in type_mat.items():
            _mark(_m, ntp[pidx] == _tid, _tu)
        if type_layers:
            rnorm = r / max(rad, 1e-9)
            nt = ntp[pidx]
            for tid, lyrs in type_layers.items():
                sel = nt == tid
                assigned = torch.zeros_like(sel)
                for (frac, yng, mat, tau) in lyrs:               # first band that contains the particle
                    band = sel & (~assigned) & (rnorm <= frac)
                    p_y = torch.where(band, torch.full_like(p_y, yng), p_y)
                    _mark(mat, band, tau)
                    assigned = assigned | band
                rem = sel & (~assigned)                          # rounding slop -> outermost layer
                p_y = torch.where(rem, torch.full_like(p_y, lyrs[-1][1]), p_y)
                _mark(lyrs[-1][2], rem, lyrs[-1][3])
        mu, la = _lame(p_y)
        mu = torch.where(is_liquid, torch.zeros_like(mu), mu)    # liquid: no shear modulus -> pressure only
                                                                 # (viscoelastic KEEPS mu -- it relaxes F, not mu)
        # `bulk_modulus` -- SAY WHAT A LIQUID ACTUALLY HAS. Young's modulus is defined by pulling a
        # rod with free sides: it stretches AND thins. A fluid at rest carries no shear, so it cannot
        # hold that stress state at all; formally nu -> 1/2 and E = 3K(1-2nu) -> 0 while K stays
        # finite. Water's Young's modulus is ZERO and its bulk modulus is 2.2 GPa.
        #
        # What `youngs: 200` on a liquid in this codebase actually sets, once `mu` is zeroed on the
        # line above, is K = la = E*nu/((1+nu)(1-2nu)) = 55.6 at the default nu = 0.2 -- a roundabout
        # bulk modulus reached through a modulus the material does not possess and a Poisson ratio
        # that is wrong for it. `bulk_modulus` sets K directly, in the same units as any other
        # stress, and is the only sane way to write a liquid in a spec that carries units.
        #
        # WHY IT MATTERS BEYOND TIDINESS. K is the number that sets the Mach number, and four
        # separate defects measured in this codebase are one defect in K being far too low:
        # a "water" impacting at Mach 0.449 with 10.1% self-weight volumetric strain; a drop that
        # compacts monotonically and never settles; a mean(J) that creeps for the whole run; and a
        # surface-tension implosion in which the Laplace pressure 2*sigma/R reaches 0.47 OF K and the
        # drop squeezes itself to a point. Real water in a 0.1 m box has 2*sigma/R / K = 1.3e-8.
        #
        # LIQUID ONLY, and an ERROR alongside `youngs`, because two ways to set one number is two
        # chances to disagree. Absent -> every existing spec is byte-identical.
        _bk = [t for t in type_list if t.get("bulk_modulus") is not None]
        if _bk:
            k_c = torch.full((n_par,), float("nan"), device=device)
            for tid, t in enumerate(type_list):
                K = t.get("bulk_modulus")
                if K is None:
                    continue
                if t.get("youngs") is not None:
                    raise ValueError(
                        f"a material type declares BOTH youngs={t['youngs']} and "
                        f"bulk_modulus={K}. For a liquid `mu` is zeroed, so `youngs` is only a "
                        f"roundabout way of setting the same K -- give one or the other.")
                if str(t.get("material", "elastic")) != "liquid":
                    raise ValueError(
                        f"bulk_modulus={K} on material {t.get('material', 'elastic')!r}. It is "
                        f"defined here for `material: liquid` only, where mu = 0 makes K = lambda "
                        f"exactly; on a solid the bulk modulus is K = lambda + 2*mu/3 and setting "
                        f"lambda from it would silently be the wrong number.")
                k_c[ntp == tid] = float(K)
            _has_k = ~torch.isnan(k_c[pidx])
            la = torch.where(_has_k, torch.nan_to_num(k_c[pidx]), la)   # mu is already 0 -> K = la


        # per-particle volume: ball footprint (disc pi*r^2 in 2D, sphere 4/3 pi r^3 in
        # 3D) / ppc, or the box volume / ppc for a block-filled pool.
        unit_vol = math.pi * rad * rad if D == 2 else (4.0 / 3.0) * math.pi * rad ** 3
        p_vol = (unit_vol / ppc).to(device=device, dtype=torch.float32) if torch.is_tensor(ppc) \
            else torch.full((Np,), unit_vol / ppc, device=device)
        for tid, t in enumerate(type_list):
            blk = t.get("block")
            if blk is not None:
                v = [float(x) for x in blk]
                vol = 1.0
                for k in range(D):
                    vol *= abs(v[D + k] - v[k])
                _pv_t = (vol / ppc).to(p_vol.dtype) if torch.is_tensor(ppc) \
                    else torch.full_like(p_vol, vol / ppc)
                p_vol = torch.where(ntp[pidx] == tid, _pv_t, p_vol)
        # A DECLARED PARTICLE MASS SETS p_vol OUTRIGHT, and then the geometry is only a placement.
        # WARN, DO NOT SILENTLY PICK ONE: a block says the body occupies THIS much space and a
        # particle mass says it occupies THAT much, and when they disagree the run means neither.
        # The tolerance is 1% because a block is usually written to 3 figures.
        if _pm is not None:
            _rho_scalar = float(rho) if not torch.is_tensor(rho) else None
            for tid, t in enumerate(type_list):
                _rt = float(t.get("density", _rho_scalar if _rho_scalar is not None else 1.0))
                _pv = float(_pm) / _rt
                _sel = ntp[pidx] == tid
                if not bool(_sel.any()):
                    continue
                blk = t.get("block")
                if blk is not None:
                    v = [float(x) for x in blk]
                    _vg = 1.0
                    for k in range(D):
                        _vg *= abs(v[D + k] - v[k])
                    _vm = float(ppc) * _pv
                    if abs(_vm / _vg - 1.0) > 0.01:
                        import warnings
                        warnings.warn(
                            f"{lvl.name}: TWO VOLUMES, AND THEY DISAGREE. The `block` on type "
                            f"{(list(types.keys())[tid] if types else tid)!r} encloses "
                            f"{_vg:.6g}, while per_parent {ppc:,} x particle_mass {float(_pm):.4g} "
                            f"/ density {_rt:g} = {_vm:.6g} -- a factor of {_vm / _vg:.4g}. The "
                            f"particle_mass wins for p_vol (and therefore for the physics); the "
                            f"block only places the particles. Drop one of the two.",
                            RuntimeWarning, stacklevel=2)
                p_vol = torch.where(_sel, torch.full_like(p_vol, _pv), p_vol)

        lvl.register_buffer("C", torch.zeros(Np, D, D, device=device))
        lvl.register_buffer("F", torch.eye(D, device=device).expand(Np, D, D).contiguous())
        lvl.register_buffer("mu", mu)
        lvl.register_buffer("la", la)
        # PER-TYPE VISCOSITY, built the same way mu and la are. `eta` was the one material property
        # that lived on the OPERATOR rather than on the particle, so every body in a set shared it:
        # a spec with a water drop, a gel blob and a snowball got one eta for all three, and the
        # obvious fix -- `at: mpm_particle[type=jelly]` -- cannot work, because the types are on the
        # PARENT and only a set declaring `types:` carries node_type.
        #
        # Registered ONLY when some type actually declares `eta`. Absent, `mpm_viscosity` uses its
        # own scalar exactly as before, so every existing spec is byte-identical.
        _etas = [t.get("eta") for t in type_list]
        _own_etas = [t.get("eta") for t in _ct] if _ct else []
        if any(e is not None for e in _etas + _own_etas):
            _eta = torch.full((Np,), float("nan"), device=device)
            for _tid, _t in enumerate(type_list):           # the PARENT's types, per particle
                if _t.get("eta") is not None:
                    _eta = torch.where(ntp[pidx] == _tid,
                                       torch.full_like(_eta, float(_t["eta"])), _eta)
            if _ct and _cnt is not None:                    # a child set's OWN types win
                for _tid, _t in enumerate(_ct):
                    if _t.get("eta") is not None:
                        _eta = torch.where(_cnt == _tid,
                                           torch.full_like(_eta, float(_t["eta"])), _eta)
            lvl.register_buffer("eta", _eta)                # NaN = "not declared, use the operator's"
        lvl.register_buffer("is_liquid", is_liquid)
        lvl.register_buffer("is_snow", is_snow)
        lvl.register_buffer("is_visco", is_visco)
        lvl.register_buffer("visco_tau", visco_tau)
        lvl.register_buffer("Jp", torch.ones(Np, device=device))
        lvl.register_buffer("p_vol", p_vol)
        lvl.register_buffer("mass", p_vol * rho_p)          # volume x the particle's own density
        lvl.register_buffer("density", rho_p if torch.is_tensor(rho_p)
                            else torch.full_like(p_vol, float(rho_p)))


@register_entity(
    # AN ORGANELLE PIECE IS A POINT WITH A RADIUS, contained in a cell (see
    # operators/organelle_ops.py): one nucleus, one mitochondrion. Its species are the set's
    # `types:` with a `count:` per cell. The reserve holds the pieces a division will spawn
    # (`on_divide: duplicate`), one extra block per seeded one.
    "organelle", depth=0,
    state_schema=spatial_schema,
    reserve_factor=1,
    render={"color_by": "node_type", "arrows": None},
)
class OrganellePiece:
    """A subcellular body reduced to its pose: position and radius, inside a vertex-model cell.
    What the piece is made of (nothing, a mesh, or MPM particles) is a child set declared in the
    spec, see notes/organelles/ORGANELLE_PLAN.md section 2a."""


@register_entity(
    # A PROTEIN CLUSTER IS A POINT, NOT MATTER: position and velocity, contained in a cell, no
    # deformation gradient and no grid (see operators/protein_ops.py). `integrin` and `myosin`
    # are the same layout under the names the specs use.
    "protein_cluster", "integrin", "myosin", depth=0,
    state_schema=spatial_schema,
    reserve_factor=3,                        # dormant slots per seeded one, unless the spec says `grow_reserve`
    render={"color_by": "node_type", "arrows": None},
)


@register_entity(
    "cell", depth=1,
    state_schema=spatial_schema,                 # dim -> StateSchema
    render={"color_by": "node_type", "arrows": "vel"},
)
class Cell:
    """A set of particles/molecules; its position is an aggregate of its children."""


@register_entity(
    # A DRAWN POINT AND NOTHING ELSE: position, velocity, and a `paint` scalar the renderer can
    # colour by. This is what a neuron's own morphology is made of -- the points sampled along its
    # skeleton or inside its mesh -- and it deliberately carries no deformation gradient, no mass
    # and no grid, because nothing integrates it. The parent neuron is the mechanism; these are its
    # picture, filled once by `morphology_seed` and painted every frame by `paint_children`.
    "points", depth=0,
    reserve_factor=0,      # nothing spawns a drawn point; the reserve would be dead rows
    state_schema=lambda D: StateSchema([Block("pos", D, role="coordinate", integration="none"),
                                        Block("vel", D, role="rate", integration="none", record=False),
                                        Block("paint", 1, integration="none")]),
    render={"color_by": "paint", "arrows": None},
)
class DrawnPoints:
    """The points a parent entity is DRAWN as: its skeleton swollen to its radius, or its mesh
    filled. Geometry, not mechanism -- no operator integrates them."""



# --------------------------------------------------------------------------- #
#  The neural sets: neuron, the assembly that contains them, and the synapse.
#
#  THREE THINGS ARE KEPT APART HERE, and the separation is the point rather than a
#  tidiness preference:
#
#    IDENTITY   what the neuron IS -- a connectome root id, a cell type, a NeuPrint
#               key. Dataclass fields on the entity class. Not numerical state.
#    GEOMETRY   where it is -- `pos`. A neuron does not MOVE, so `pos` is fixed
#               geometry (`integration: none`), not an integrated coordinate. A
#               skeleton/mesh, when one is imported, is a further attachment and
#               still not state.
#    DYNAMICS   what it DOES -- `voltage`, and the per-type parameters of its update
#               equation. This is the only part an operator integrates.
#
#  Keeping geometry out of the dynamics is what later makes "does the mechanism
#  depend on morphology?" a question that can be asked at all: morphology can be
#  attached, removed or varied without touching the neural state.
# --------------------------------------------------------------------------- #
def neuron_schema(dim: int) -> StateSchema:
    """`pos` (fixed geometry) | `voltage` (the integrated coordinate) | `omega` (an
    external modulation channel).

    `voltage` IS THE COORDINATE, not `pos`, and that inversion is the whole reason a
    neuron cannot use `spatial_schema`. `StateSchema.coordinate` returns the first
    `second_order_coordinate` and otherwise the first `first_order` block, so declaring
    `pos` as `none` and `voltage` as `first_order` makes the engine integrate the
    voltage and leave the position alone -- and sizes the set's delta accumulator to
    one column instead of `dim`. A spatial set is a body that moves and carries state;
    a neuron is a state that sits still.

    `omega` is the per-neuron value of an external field Omega_i(t) (see
    `operators/neural.py`). It is `none`-integrated -- written by an exchange operator,
    never advanced -- and unrecorded, since it is an input the run already knows.
    """
    return StateSchema([
        Block("pos", dim, role="geometry", integration=NONE, boundary=BOUNDARY_WORLD),
        Block("voltage", 1, role="coordinate", integration=FIRST_ORDER, boundary=BOUNDARY_FREE),
        Block("omega", 1, role="modulation", integration=NONE, boundary=BOUNDARY_FREE,
              record=False),
        # THE PRINCIPAL NEURITE DIRECTION: the axis along which this cell's arbour is most
        # extended, as a unit vector, pointing away from the soma. It is GEOMETRY, like `pos`
        # -- a fixed property of the cell, `none`-integrated, never advanced. It is here rather
        # than in a renderer because it is a fact about the neuron: two cells at the same place
        # with opposite projection axes are different cells, and an operator that cared about
        # anisotropy (a direction-dependent connection rule, a polarised conductance) would read
        # this block. `record=False` because it is static -- storing 2,001 identical copies per
        # run buys nothing, and a consumer reads it from the region's `neurons.npz`.
        Block("neurite_dir", dim, role="orientation", integration=NONE,
              boundary=BOUNDARY_FREE, record=False),
        # THE SOMA RADIUS, MEASURED FROM THE TISSUE rather than assumed. fish2 populates
        # `somaRadius` on 0 of 177,513 bodies, so the size of a cell body has until now been a
        # stated literature constant applied to every neuron alike. It does not have to be:
        # dense EM tracing cannot route a neurite through a soma, so the largest ball around a
        # soma centre containing no OTHER neuron's traced neurite is an upper bound on that
        # soma, and it is per-cell. Geometry like `pos` and `neurite_dir` -- fixed, never
        # advanced, unrecorded because it does not change over a run.
        Block("soma_radius", 1, role="geometry", integration=NONE,
              boundary=BOUNDARY_FREE, record=False),
    ])


def assembly_schema(dim: int) -> StateSchema:
    """`pos` (the assembly's location) | `activity` (a readout of its neurons).

    An assembly is NOT a new computational primitive -- it is a set at another scale,
    related to its neurons by the same `parent` containment map that relates particles
    to a cell. `activity` is `none`-integrated because it is a derived readout written
    by an aggregate operator, not a state with dynamics of its own.
    """
    return StateSchema([
        Block("pos", dim, role="geometry", integration=NONE, boundary=BOUNDARY_WORLD),
        Block("activity", 1, role="readout", integration=NONE, boundary=BOUNDARY_FREE),
    ])


def synapse_schema(dim: int) -> StateSchema:
    """`w` -- one fixed weight per connection. The CONNECTIVITY MATRIX, in sparse form.

    W is a first-class mechanistic object, not an implementation detail: it is what an
    inverse model reconstructs. So it lives where the language can see it -- as the
    state of an EDGE-SET whose elements are connections, joined to the neuron set by
    the `pre`/`post` incidence maps -- rather than as a dense tensor hidden inside an
    operator. Everything a synapse might later grow (plasticity, a delay, a
    transmitter type, a geometry) is another block here, and none of it disturbs the
    neuron abstraction.
    """
    return StateSchema([
        Block("w", 1, role="weight", integration=NONE, boundary=BOUNDARY_FREE, record=False),
    ])


@register_entity(
    "neuron", depth=0,
    state_schema=neuron_schema,
    render={"color_by": "node_type", "arrows": None},
)
class Neuron:
    """A neuron's biological identity and structural metadata -- NOT its dynamic state.

    The numerical quantities live in the `Level`'s state tensor under `neuron_schema`
    above; the per-type parameters of its update equation live in the set's `types:`
    table (`lvl.type_params[lvl.node_type]`). What belongs HERE is what a connectome
    knows about the cell and a simulation does not derive: which neuron it is, and what
    it is called.

    These fields are populated by an importer (a NeuPrint / FlyWire reader), not by the
    engine, and are absent for a synthetic network -- which is why they all default to
    None rather than being required.
    """

    root_id: int | None = None            # connectome body/root id
    cell_type: str | None = None          # e.g. "EPG", "T4a", "L1"
    neuprint_id: str | None = None        # source key, when imported from a NeuPrint server


@register_entity(
    "neural_assembly", "assembly", depth=1,
    state_schema=assembly_schema,
    render={"color_by": "node_type", "arrows": None},
)
class NeuralAssembly:
    """A group of neurons, as an ordinary contained set one scale up.

    Deliberately not a special class: `brain -> assembly -> neuron -> synapse` uses the
    same containment machinery as `organism -> tissue -> cell -> particle`, which is
    what makes the neural case a test of the hierarchy claim rather than a subsystem
    bolted beside it.
    """


@register_entity(
    "connection", "synapse", depth=0,
    state_schema=synapse_schema,
    render={"color_by": "node_type", "arrows": None},
)
class Connection:
    """One weighted link between two sets: an EDGE-SET element carrying the weight W_e.

    `synapse` IS AN ALIAS, and the general name leads for a reason. What this entity provides is
    a `w` block on a relation, which is what EVERY weighted map needs -- a sensor population into
    a circuit, a circuit into an effector, one circuit into another, or a free matrix with no
    anatomy in it at all. Only some of those are synapses. A free 64x64 recurrent matrix in a
    prototype is a CONNECTIVITY matrix and calling it a synapse overclaims: a synapse is a
    measured thing, and the word should stay available for when the weights actually came from
    tissue.

    So a spec names its set for what that set is -- `recurrent`, `afferent`, `junction`,
    `synapse` -- and points `entity:` here for the state. The set's NAME is what it is; the
    entity is what provides its layout."""


# default for any set whose name is not a registered entity. Kept as the legacy dict
# because it is a HINT, in the same sense as the dicts described at the top of this
# file: the engine's fallback for an unregistered name is `spatial_schema(dim)`, not
# this. Anything reading it gets the 2D shape of the default spatial layout.
DEFAULT_STATE_SCHEMA = {"pos": (0, 2), "vel": (2, 4)}
DEFAULT_RENDER = {"color_by": "node_type", "arrows": None}

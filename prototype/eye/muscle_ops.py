"""muscle_ops -- the six extraocular muscles as CONTRACTING MPM BODIES (prototype-local).

In the first version of this prototype a muscle was a line of action and its pull was a
body force injected at the insertion. That is a caricature: nothing contracted, nothing
deformed, and the tendon's grip on the sclera was asserted rather than simulated. Here the
muscles are real tissue -- their own set of material points -- and the mechanics does the
rest:

    a muscle SHORTENS because an active stress acts along its fibre axis
        (the standard  sigma_a = A * a(t) * f f^T  of muscle mechanics),
    it CANNOT run away because its origin end is anchored to bone,
    and it MOVES THE EYE because its tendon end is embedded in the sclera and the two
        bodies share one MLS-MPM background grid, which is what transmits the force.

No operator applies a force "to the eye". The globe rotates because a muscle got shorter.

WHY A SECOND PARTICLE SET. The globe and the muscles are different biological entities with
different material laws, different counts and different operators acting on them, so they are
different Plexus sets: `mpm_particle` (parent: eye) and `muscle_particle` (parent: muscle).
They are coupled by SHARING the `mpm_grid` field -- which is exactly how MLS-MPM couples any
two bodies. The stock `mpm_scatter` overwrites the grid, so a second implementation of that
same contract, `implementation: accumulate`, is registered here: identical biology (a
particle -> grid scatter), different numerics (it adds to the grid rather than replacing it,
and it reads its active stress from a per-set buffer instead of a global side-channel). Per
Plexus2 that is the correct way to extend -- a new implementation of an existing contract,
not an edit to the engine.

Operators
    muscle_morphogenesis  rewire     shape each muscle's points into a strap; fibres, caps
    muscle_geometry       aggregate  points -> muscle: length, insertion, line of action, axis
    muscle_contract       exchange   activation -> active stress along the fibre
    bone_anchor           lateral    hold the origin end to the skull
    mpm_scatter[accumulate]  exchange   the second body's scatter into the shared grid
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Lateral, Aggregate, Exchange, Rewire
from plexus.models.registry import register_operator, register_entity
from plexus.models.entities import MPMParticle
from plexus.operators.mpm_grid import stencil_offsets, bspline, sub_dt

import eye_anatomy as EA


# --------------------------------------------------------------------------- #
#  entity: a muscle's material point (same continuum buffers as an MPM particle)
# --------------------------------------------------------------------------- #
@register_entity(
    "muscle_particle", depth=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": None},
)
class MuscleParticle:
    """A material point of an extraocular muscle. Identical continuum state to the globe's
    `mpm_particle` (F, C, mass, Lame mu/la, p_vol) -- it is a distinct SET because it is a
    distinct biological entity with its own operators, not because it needs different state.
    `provision` is the stock one; `muscle_morphogenesis` then reshapes and re-masses it."""
    provision = MPMParticle.provision


# --------------------------------------------------------------------------- #
#  strap geometry
# --------------------------------------------------------------------------- #
def _ovoid_radius(d, a, c):
    """Distance from the centre to the ovoid surface along the unit direction `d`."""
    d = np.atleast_2d(d)
    return 1.0 / np.sqrt((d[:, 0] ** 2 + d[:, 1] ** 2) / a ** 2 + d[:, 2] ** 2 / c ** 2)


def _radical_inverse(n, base):
    out = np.zeros_like(n, dtype=np.float64)
    f, i = 1.0 / base, n.astype(np.int64).copy()
    while np.any(i > 0):
        out += f * (i % base)
        i //= base
        f /= base
    return out


def strap_path(centre, n_ins, origin, a, c, arc_deg, gap_prox, embed, frac=0.55, n_arc=48):
    """The centreline of one muscle, from ORIGIN (s=0) to INSERTION (s=1).

    Two pieces, which is how the muscle actually runs: a straight belly from the origin down
    to the point where it meets the globe, then an ARC OF CONTACT hugging the sclera to the
    insertion. The stand-off from the sclera tapers from `gap_prox` (the belly rides clear of
    the globe, so grid contact does not glue the whole arc down and freeze the rotation) to
    `embed` at the insertion, which is NEGATIVE -- the tendon bites into the sclera, so the
    shared grid welds the two bodies exactly where a tendon is anchored.
    """
    n_hat = np.asarray(n_ins, float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    ins = centre + float(_ovoid_radius(n_hat, a, c)[0]) * n_hat
    to_org = np.asarray(origin, float) - ins
    tang = to_org - np.dot(to_org, n_hat) * n_hat
    nt = np.linalg.norm(tang)
    tang = tang / nt if nt > 1e-9 else np.cross(n_hat, [0.0, 0.0, 1.0])
    phis = np.radians(np.linspace(0.0, arc_deg, n_arc))
    dirs = np.cos(phis)[:, None] * n_hat[None, :] + np.sin(phis)[:, None] * tang[None, :]
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    along = phis / max(phis[-1], 1e-9)
    gap = embed + (gap_prox - embed) * along ** 0.6
    arc = centre[None, :] + (_ovoid_radius(dirs, a, c) + gap)[:, None] * dirs
    path = np.vstack([np.asarray(origin, float)[None, :], arc[::-1]])   # origin -> insertion
    if frac < 1.0:
        # Keep only the DISTAL `frac` of the path. The four recti share one bony origin at
        # the orbital apex, so drawing them to it piles all four onto one point and buries
        # the globe. Truncating them in mid-orbit is also the better mechanics: Demer's
        # active-pulley hypothesis puts each rectus' EFFECTIVE origin at a connective-tissue
        # pulley part-way down the orbit, not at the apex. The line of action at the
        # insertion -- the only thing the torque depends on -- is unchanged, because the
        # path is the same curve.
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        sc = np.concatenate([[0.0], np.cumsum(seg)])
        keep = sc >= (1.0 - frac) * sc[-1]
        cut = np.interp((1.0 - frac) * sc[-1], sc, np.arange(len(sc)))
        lo = int(np.floor(cut))
        w = cut - lo
        start = path[lo] * (1 - w) + path[min(lo + 1, len(path) - 1)] * w
        path = np.vstack([start[None, :], path[keep]])
    binormal = np.cross(n_hat, tang)
    return path, binormal / np.linalg.norm(binormal)


def resample(path, m=160):
    """Arc-length resampling + unit tangents. Returns (points [m,3], tangents [m,3], length)."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    u = np.linspace(0.0, L, m)
    pts = np.stack([np.interp(u, s, path[:, k]) for k in range(3)], 1)
    tan = np.gradient(pts, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True).clip(1e-12)
    return pts, tan, L


def _taper(sv, end=0.30):
    """Cross-section profile along the muscle: a thick belly tapering to thin tendons."""
    return end + (1.0 - end) * np.sin(np.pi * np.clip(sv, 0, 1)) ** 0.5


# --------------------------------------------------------------------------- #
#  1. muscle_morphogenesis (rewire)
# --------------------------------------------------------------------------- #
@register_operator("muscle_morphogenesis", family="anatomy", set="muscle_particle", kind="rewire")
class MuscleMorphogenesis(Rewire):
    """Give each muscle its shape, its fibres and its two attachments.

    The engine seeds a contained set as a ball around its parent; a muscle is not a ball, so
    at frame 0 this operator moves every point onto the strap described by `strap_path`: a
    Hammersley sequence fills the strap uniformly (deterministic, and uniform density is what
    MLS-MPM needs), a tapered elliptical cross-section gives it a belly and two tendons, and
    each point records

        s          where it lies along the muscle, 0 at the origin, 1 at the insertion
        fibre      the local fibre direction -- the axis the active stress will act along
        anchored   the origin cap, held to the skull by `bone_anchor`
        tendon     the insertion cap, embedded in the sclera

    Particle volume and mass are recomputed from the strap's TRUE volume, since the stock
    provision sized them for the ball it seeded.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True       # writes the rest configuration at frame 0
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MAPS = ["parent"]
    MECHANISM_TAGS = ["morphogenesis_static", "fibre_architecture", "tendon_attachment"]
    PARAM_ROLES = {"width": "belly_width", "thickness": "belly_thickness",
                   "arc_deg": "arc_of_contact", "gap": "sclera_standoff",
                   "embed": "tendon_embedding_depth", "youngs": "muscle_stiffness"}
    REFERENCE = "Demer, J. L. (2002). Invest. Ophthalmol. Vis. Sci. 43:2179 (orbital pulleys and paths)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.center = np.asarray(params.get("center", EA.GLOBE_CENTER), float)
        self.a = float(params.get("a_eq", EA.A_EQ))
        self.c = float(params.get("c_ax", EA.C_AX))
        self.width = float(params.get("width", 0.034))
        self.thickness = float(params.get("thickness", 0.021))
        self.arc_deg = float(params.get("arc_deg", 30.0))
        self.frac = float(params.get("frac", 0.88))     # distal fraction kept (pulley origin)
        self.gap = float(params.get("gap", 0.038))
        self.embed = float(params.get("embed", -0.013))
        self.cap = float(params.get("cap", 0.10))          # fraction of the length that is a cap
        self.youngs = float(params.get("youngs", 60.0))
        self.density = float(params.get("density", 1.0))
        self.nu = 0.2
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        par = p.parent.detach().cpu().numpy()
        M = int(par.max()) + 1
        n_ins_all = EA.insertion_dirs()
        org_all = EA.origins_world()

        X = np.zeros((p.n, 3))
        fib = np.zeros((p.n, 3))
        sarr = np.zeros(p.n)
        pvol = np.zeros(p.n)
        rest_len = np.zeros(M)
        centrelines = []
        for mi in range(M):
            sel = np.nonzero(par == mi)[0]
            n = sel.size
            path, binorm = strap_path(self.center, n_ins_all[mi], org_all[mi], self.a, self.c,
                                      self.arc_deg, self.gap, self.embed, self.frac)
            pts, tan, L = resample(path)
            centrelines.append(pts)
            rest_len[mi] = L
            # Hammersley fill: s along the muscle, (r, th) over the elliptical cross-section
            j = np.arange(n)
            sv = (j + 0.5) / n
            u1 = _radical_inverse(j, 2)
            u2 = _radical_inverse(j, 3)
            rr = np.sqrt(u1)
            th = 2 * np.pi * u2
            k = np.clip((sv * (len(pts) - 1)).astype(int), 0, len(pts) - 1)
            t_hat = tan[k]
            b_hat = np.tile(binorm, (n, 1))
            b_hat = b_hat - (b_hat * t_hat).sum(1, keepdims=True) * t_hat
            b_hat /= np.linalg.norm(b_hat, axis=1, keepdims=True).clip(1e-12)
            r_hat = np.cross(t_hat, b_hat)
            tap = _taper(sv)
            off = (0.5 * self.width * tap * rr * np.cos(th))[:, None] * b_hat \
                + (0.5 * self.thickness * tap * rr * np.sin(th))[:, None] * r_hat
            X[sel] = pts[k] + off
            fib[sel] = t_hat
            sarr[sel] = sv
            # true strap volume -> per-particle volume (the ball-seeded value is meaningless)
            ds = L / len(pts)
            vol = float(np.sum(0.25 * np.pi * self.width * self.thickness * _taper(
                np.linspace(0, 1, len(pts))) ** 2 * ds))
            pvol[sel] = vol / n

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
        m.register_buffer("rest_length", torch.as_tensor(rest_len, dtype=torch.float32, device=dev))
        print(f"[muscle_morphogenesis] {M} muscles x {p.n // M} points; rest lengths "
              + " ".join(f"{k}={l:.3f}" for k, l in zip(EA.MUSCLE_KEYS, rest_len)), flush=True)
        self._done = True
        return {}


# --------------------------------------------------------------------------- #
#  2. muscle_geometry (aggregate): points -> muscle
# --------------------------------------------------------------------------- #
@register_operator("muscle_geometry", family="hierarchy", set="muscle", kind="aggregate")
class MuscleGeometry(Aggregate):
    """Measure each muscle from its own material points, along the `parent` map.

    Points are binned by their fibre coordinate `s` and the bin centroids traced: the
    polyline through them is the muscle's current CENTRELINE, its total length is the
    muscle's LENGTH (so shortening is measured, not assumed), its last vertex is the
    insertion, and the direction of its final segment is the current LINE OF ACTION. The
    rotation axis  n_hat x u_hat  follows, and `oculomotor_drive` reads it next frame.

    So the textbook actions of the six muscles are never tabulated anywhere in this
    prototype: they are re-measured every frame from where the tissue actually is.
    A derived readout, hence MAY_MUTATE_INTEGRATED_STATE (as the stock `aggregate`).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle"]
    READS = ["pos"]
    WRITES = ["pos", "length"]
    MAPS = ["parent"]
    MECHANISM_TAGS = ["length_readout", "line_of_action", "hierarchical_readout"]
    PARAM_ROLES = {"bins": "centreline_resolution"}
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.child = params.get("child", "muscle_particle")
        self.eye = params.get("eye", "eye")
        self.bins = int(params.get("bins", 14))

    def forward(self, H, mask=None):
        m = H.level(self.at)
        p = H.level(self.child)
        if not hasattr(p, "s"):
            return {}
        dev = p.state.device
        M, B = m.n, self.bins
        X = p.get("pos")
        b = (p.s * B).long().clamp(0, B - 1)
        key = p.parent * B + b
        acc = torch.zeros(M * B, 3, device=dev).index_add_(0, key, X)
        cnt = torch.zeros(M * B, device=dev).index_add_(0, key, torch.ones_like(p.s))
        cen = (acc / cnt.clamp(min=1.0)[:, None]).view(M, B, 3)
        seg = (cen[:, 1:] - cen[:, :-1]).norm(dim=2)
        length = seg.sum(1)

        ins = cen[:, -1]                                    # the insertion end of the centreline
        u = cen[:, -3] - cen[:, -1]                         # heading from the insertion, proximally
        u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-9)
        eye_c = H.level(self.eye).get("pos")[0]
        n_hat = ins - eye_c[None, :]
        n_hat = n_hat / n_hat.norm(dim=1, keepdim=True).clamp(min=1e-9)
        axis = torch.cross(n_hat, u, dim=1)
        axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-9)

        m.ins_pos = ins.detach()
        m.pull = u.detach()
        m.axis = axis.detach()
        m.centreline = cen.detach()
        new = m.state.clone()
        pa, pb = m.state_schema["pos"]
        la, lb = m.state_schema["length"]
        new[:, pa:pb] = cen.mean(1)
        new[:, la:lb] = length[:, None]
        m.state = new
        return {}


# --------------------------------------------------------------------------- #
#  3. muscle_contract (exchange): activation -> active stress along the fibre
# --------------------------------------------------------------------------- #
@register_operator("muscle_contract", family="mechanics", set="muscle_particle", kind="exchange")
class MuscleContract(Exchange):
    """The contractile machinery: an activated muscle carries a tension along its fibres.

    Each point gets the active stress

        sigma_a(x) = A * a_m * (1 + beta*(lambda-1)) * f f^T ,     f = the local fibre axis,

    which the MLS-MPM scatter adds to the elastic stress before forming the affine momentum
    matrix. The tissue therefore feels contraction through the DIVERGENCE of a stress field,
    which is what a muscle is: it shortens along f and hauls on whatever its ends are attached
    to. It is not a force applied to a point. The optional `stretch_activation` beta is the
    length--tension relation (a stretched sarcomere pulls harder).

    The stress is published on a per-set buffer, not on a global side-channel, because two
    particle sets share the grid here and only ONE of them is contractile; the `accumulate`
    implementation of `mpm_scatter` reads it from the set it is scattering.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["amplitude"]
    INPUTS = ["muscle", "muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["act"]
    WRITES = []
    MAPS = ["parent"]
    MECHANISM_TAGS = ["active_contraction", "active_stress_tensor", "length_tension"]
    PARAM_ROLES = {"amplitude": "peak_active_stress", "strength": "per_muscle_strength",
                   "stretch_activation": "length_tension_slope",
                   "taper": "tendon_passivity"}
    REFERENCE = "Hill, A. V. (1938). Proc. R. Soc. B 126:136; Niederer, S. A. et al. (2006). Biophys. J. 90:1697 (active-stress formulation)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.muscles = params.get("muscles", "muscle")
        self.amplitude = float(params["amplitude"])
        self.stretch_activation = float(params.get("stretch_activation", 0.0))
        self.taper = bool(params.get("taper", True))
        # per-muscle strength. The obliques reach the globe over a much shorter post-pulley
        # path than the recti (L/R ~ 2.0 against ~3.3), and a muscle's reachable rotation
        # scales as (A/E) x (L/R), so at equal active stress they are the weak pair. Their
        # real compliance lives in the long belly BEHIND the trochlea, which this model
        # truncates; a per-muscle strength factor is how that is paid back.
        self.strength = params.get("strength") or list(EA.peak_tensions())

    def forward(self, H, mask=None):
        p = H.level(self.at)
        m = H.level(self.muscles)
        if not hasattr(p, "fibre"):
            return {}
        if not hasattr(self, "_strength_t"):
            self._strength_t = torch.as_tensor(self.strength, dtype=torch.float32,
                                               device=p.state.device)
        act = m.get("act")[:, 0].clamp(min=0.0)[p.parent]         # broadcast along `parent`
        f = p.fibre
        gate = self.amplitude * self._strength_t[p.parent] * act
        if self.taper:                                            # tendons are passive, the belly pulls
            gate = gate * torch.sin(math.pi * p.s.clamp(0, 1)).clamp(min=0.0) ** 0.5
        if self.stretch_activation != 0.0:
            # THE FORCE-LENGTH RELATION, and the thing that makes this model stable.
            # lambda = |F f| is the fibre stretch; tension falls off as the sarcomere shortens,
            #     T = A a (1 + beta (lambda - 1)) ,   clamped at zero,
            # so a muscle stops pulling once it has shortened by 1/beta and CANNOT reel itself
            # up. That is what lets the passive modulus be far SOFTER than the peak active
            # stress, as it is in real muscle. Coupling A to a stiff passive element instead
            # (the earlier attempt) bought stability at the cost of an antagonist so stiff that
            # it ate most of the agonist's force, and capped the eye at ~11 degrees.
            lam = torch.bmm(p.F, f[:, :, None]).squeeze(-1).norm(dim=1).clamp(min=1e-6)
            gate = gate * (1.0 + self.stretch_activation * (lam - 1.0)).clamp(min=0.0)
        if mask is not None:
            gate = gate * mask.float()
        p.active_stress = gate[:, None, None] * (f[:, :, None] * f[:, None, :])
        return {}


# --------------------------------------------------------------------------- #
#  4. bone_anchor (lateral): the origin end is fixed to the skull
# --------------------------------------------------------------------------- #
@register_operator("bone_anchor", family="mechanics", set="muscle_particle", kind="lateral")
class BoneAnchor(Lateral):
    """Hold a tagged cap of points at their rest positions -- the muscle's bony origin.

    Without it a contracting muscle simply reels itself up: a force needs something to pull
    against. The annulus of Zinn, the trochlea and the orbital floor are bone, so the origin
    cap is pinned (stiffness `k`, damping `c`) and every newton the muscle develops is
    delivered to the other end, where the tendon is embedded in the sclera.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k"]
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos", "vel"]
    WRITES = []
    MECHANISM_TAGS = ["boundary_condition", "rest_restoring", "skeletal_attachment"]
    PARAM_ROLES = {"k": "anchor_stiffness", "c": "anchor_damping", "flag": "anchored_subset"}
    REFERENCE = "Plexus (this work); cf. `mpm_anchor` (substrate attachment)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.k = float(params["k"])
        self.c = float(params.get("c", 40.0))
        self.flag = str(params.get("flag", "anchored"))

    def forward(self, H, mask=None):
        p = H.level(self.at)
        sel = getattr(p, self.flag, None)
        if sel is None or not hasattr(p, "rest"):
            return {}
        s = sel.float()[:, None]
        acc = (self.k * (p.rest - p.get("pos")) - self.c * p.get("vel")) * s
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


# --------------------------------------------------------------------------- #
#  5. mpm_scatter, implementation "accumulate": a second body into the shared grid
# --------------------------------------------------------------------------- #
@register_operator("mpm_scatter", implementation="accumulate",
                   family="mpm", set="particle", kind="exchange")
class MPMScatterAccumulate(Exchange):
    """The MLS-MPM particle-to-grid scatter, ACCUMULATING into a grid another set has
    already written. Same contract, same biology, same stress law as the stock
    implementation; two differences in the numerics only:

      * it ADDS mass / momentum / colour to the grid instead of replacing them, so several
        bodies can share one background grid -- which is how MLS-MPM couples them, and the
        only reason the muscles can move the eye at all;
      * it reads its optional active stress from the SET's own `active_stress` buffer rather
        than from the global `H.active_stress` side-channel, because with more than one
        particle set a global channel is ambiguous (and shaped for the wrong set).

    The default implementation must run FIRST in the spec's operator order -- it is the one
    that resets the grid for the tick.
    """

    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["particle_to_grid", "fixed_corotated_stress", "active_stress",
                      "multi_body_coupling"]
    PARAM_ROLES = {"dt_sub": "MLS-MPM substep dt", "drag": "Stokes drag coefficient",
                   "a_max": "external-acceleration clamp"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM P2G)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.to = params.get("to", "mpm_grid")
        self.dt_sub = float(params.get("dt_sub", 2e-4))
        self.drag = float(params.get("drag", 0.0))
        self.a_max = float(params.get("a_max", 200.0))

    def forward(self, H, mask=None):
        p = H.level(self.at); g = H.field(self.to); dev = p.state.device
        dt = sub_dt(H, self.dt_sub)
        inv_dx, dx = g.inv_dx, g.dx
        D = p.F.shape[-1]
        periodic = bool(getattr(H, "periodic", False))
        offsets = stencil_offsets(D, dev)
        X, V = p.get("pos"), p.get("vel")
        pn = getattr(p, "parent_name", None)
        if pn is not None:
            a_cell = H.delta(pn)
            a_cell = torch.nan_to_num(a_cell, posinf=self.a_max, neginf=-self.a_max)
            a_cell = a_cell.clamp(-self.a_max, self.a_max)[:, :D]
            a_ext = a_cell[p.parent]
        else:
            a_ext = torch.zeros(p.n, D, device=dev)
        a_ext = a_ext + torch.nan_to_num(H.delta(p.name))
        V = V + dt * (a_ext - self.drag * V)

        F, C, mass = p.F, p.C, p.mass
        eye = torch.eye(D, device=dev).expand(p.n, D, D)
        J = torch.linalg.det(F)
        U, S, Vh = torch.linalg.svd(F)
        U = U.clone(); Vh = Vh.clone()
        U[torch.det(U) < 0, :, -1] *= -1
        Vh[torch.det(Vh) < 0, -1, :] *= -1
        R = U @ Vh
        stress = 2 * p.mu[:, None, None] * ((F - R) @ F.transpose(-2, -1)) \
            + eye * (p.la * J * (J - 1))[:, None, None]
        act = getattr(p, "active_stress", None)          # PER-SET active stress (muscle)
        if act is not None:
            stress = stress + act
        stress = (-dt * 4 * inv_dx * inv_dx) * p.p_vol[:, None, None] * stress
        affine = stress + mass[:, None, None] * C

        fx, weight, flat = bspline(X, inv_dx, offsets, g.shape, periodic)
        occ = getattr(p, "occ", None)
        if occ is not None:
            weight = weight * (occ > 0).to(weight.dtype)[:, None]
        dpos_phys = (offsets[None] - fx[:, None, :]) * dx
        mom = mass[:, None, None] * V[:, None, :] + (affine[:, None] @ dpos_phys[..., None]).squeeze(-1)
        gm = torch.zeros(g.n_cells, device=dev)
        gmv = torch.zeros(g.n_cells, D, device=dev)
        gm.index_add_(0, flat, (weight * mass[:, None]).reshape(-1))
        gmv.index_add_(0, flat, (weight[..., None] * mom).reshape(-1, D))
        g.m = g.m + gm                                    # ACCUMULATE (the one real difference)
        g.mv = g.mv + gmv
        return {}

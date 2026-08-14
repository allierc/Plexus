"""bench_ops -- the two operators the minimal rig needs, and nothing else.

The eye has six muscles, a socket, orbital fat, a lens and five antagonists, and a
question about the muscle answered there is answered against all of that at once. This
rig has a bone, one muscle and one ball:

    bone (fixed)  ====[ muscle tube ]====  ( deformable sphere, held in place )

so the only thing that can absorb a contraction is the muscle itself. Everything else
is stock: `muscle_contract`, `bone_anchor`, `mpm_*`, all unchanged.

    muscle_morphogenesis [tube]  a straight cylinder from bone to sphere, fibres along
                                 its axis, an anchored cap at one end and an embedded
                                 cap at the other -- the same four buffers the eye's
                                 strap version writes, so every downstream operator is
                                 untouched
    pin_region                   holds the sphere's far side. "Fixed but deformable" is
                                 the point: the ball may not fly away, but it must be
                                 free to dimple where the tendon pulls, because that
                                 dimple is one of the places the contraction can go.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Lateral, Rewire
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_operator, register_entity


@register_entity(
    "bone_particle", depth=0,
    state_schema={"pos": (0, 2), "vel": (2, 4)},
    render={"color_by": "node_type", "arrows": None},
)
class BoneParticle:
    """A material point of bone. Same continuum state as any MPM particle -- it is a
    separate SET because it is a separate body with its own operators (it is pinned,
    and it is not contractile), not because it needs different state."""
    provision = MPMParticle.provision


def _radical_inverse(n, base):
    out = np.zeros_like(n, dtype=np.float64)
    f, i = 1.0 / base, n.astype(np.int64).copy()
    while np.any(i > 0):
        out += f * (i % base)
        i //= base
        f /= base
    return out


@register_operator("muscle_morphogenesis", implementation="tube",
                   family="anatomy", set="muscle_particle", kind="rewire")
class TubeMorphogenesis(Rewire):
    """Shape the muscle's points into a straight cylinder between two points.

    A Hammersley sequence fills it uniformly, because MLS-MPM wants uniform density and
    a random fill gives clumps that read as material inhomogeneity. `s` runs 0 at the
    BONE end to 1 at the SPHERE end, the fibre is the axis everywhere, and the two end
    caps are labelled `anchored` and `tendon` exactly as the eye's version labels them.
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
    MECHANISM_TAGS = ["morphogenesis_static", "fibre_architecture", "tendon_attachment"]
    PARAM_ROLES = {"radius": "tube_radius", "youngs": "muscle_stiffness",
                   "cap": "attachment_cap_fraction"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        # one endpoint pair per muscle: the rig grew a second, antagonist muscle and a
        # tube generator that can only make one is a generator that decides the anatomy
        self.a = np.atleast_2d(np.asarray(params.get("bone_ends",
                               [params.get("bone_end", (0.22, 0.5, 0.5))]), float))
        self.b = np.atleast_2d(np.asarray(params.get("sphere_ends",
                               [params.get("sphere_end", (0.50, 0.5, 0.5))]), float))
        self.radius = float(params.get("radius", 0.022))
        self.cap = float(params.get("cap", 0.10))
        self.youngs = float(params.get("youngs", 240.0))
        self.density = float(params.get("density", 1.0))
        self.nu = float(params.get("poisson", 0.2))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        par = p.parent.detach().cpu().numpy()
        M = int(par.max()) + 1
        X = np.zeros((p.n, 3))
        fib = np.zeros((p.n, 3))
        sv = np.zeros(p.n)
        pvol = np.zeros(p.n)
        rest_len = np.zeros(M)
        for mi in range(M):
            sel = np.nonzero(par == mi)[0]
            n = sel.size
            a = self.a[mi % len(self.a)]
            b = self.b[mi % len(self.b)]
            ax = b - a
            L = float(np.linalg.norm(ax))
            t_hat = ax / L
            u = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(u, t_hat)) > 0.9:
                u = np.array([0.0, 1.0, 0.0])
            e1 = np.cross(t_hat, u)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(t_hat, e1)
            j = np.arange(n)
            s = (j + 0.5) / n
            rr = self.radius * np.sqrt(_radical_inverse(j, 2))
            th = 2 * np.pi * _radical_inverse(j, 3)
            X[sel] = (a[None, :] + (s * L)[:, None] * t_hat[None, :]
                      + (rr * np.cos(th))[:, None] * e1[None, :]
                      + (rr * np.sin(th))[:, None] * e2[None, :])
            fib[sel] = t_hat
            sv[sel] = s
            pvol[sel] = float(np.pi * self.radius ** 2 * L) / n
            rest_len[mi] = L
        n = p.n
        s = sv
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = new
        mu = self.youngs / (2 * (1 + self.nu))
        la = self.youngs * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        p.mu = torch.full((n,), mu, device=dev)
        p.la = torch.full((n,), la, device=dev)
        p.p_vol = torch.as_tensor(pvol, dtype=torch.float32, device=dev)
        p.mass = p.p_vol * self.density
        p.register_buffer("fibre", torch.as_tensor(fib, dtype=torch.float32, device=dev))
        p.register_buffer("s", torch.as_tensor(s, dtype=torch.float32, device=dev))
        p.register_buffer("rest", torch.as_tensor(X, dtype=torch.float32, device=dev))
        p.register_buffer("anchored", torch.as_tensor(s < self.cap, device=dev))
        p.register_buffer("tendon", torch.as_tensor(s > 1.0 - self.cap, device=dev))
        p.register_buffer("active_stress", torch.zeros(n, 3, 3, device=dev))
        m = H.level(p.parent_name)
        m.register_buffer("rest_length",
                          torch.as_tensor(rest_len, dtype=torch.float32, device=dev))
        L = float(rest_len.mean())
        print(f"[muscle_morphogenesis:tube] {n} points, length {L:.4f}, radius {self.radius}, "
              f"E {self.youngs}", flush=True)
        self._done = True
        return {}


@register_operator("pin_region", family="mechanics", set="particle", kind="lateral")
class PinRegion(Lateral):
    """Hold one side of a body to its rest position with a stiff spring.

    "Fixed but deformable": the sphere may not translate away when the muscle pulls --
    that would confound the measurement with rigid-body motion, which is exactly the
    confound this rig exists to remove -- but the near side must stay free, because a
    tendon that simply dimples the surface instead of moving the body is one of the
    answers the rig is looking for.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos", "vel"]
    WRITES = []
    MECHANISM_TAGS = ["boundary_condition", "kinematic_constraint"]
    PARAM_ROLES = {"k": "pin_stiffness", "c": "pin_damping", "beyond": "pinned_side"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.k = float(params.get("k", 60000.0))
        self.c = float(params.get("c", 200.0))
        self.axis = int(params.get("axis", 0))
        self.beyond = float(params.get("beyond", 0.5))     # pin everything past this
        self._m = None

    def forward(self, H, mask=None):
        p = H.level(self.at)
        X, V = p.get("pos"), p.get("vel")
        if self._m is None:
            if not hasattr(p, "rest_pos"):
                p.register_buffer("rest_pos", X.clone())
            self._m = (p.rest_pos[:, self.axis] > self.beyond).float()[:, None]
        a = (self.k * (p.rest_pos - X) - self.c * V) * self._m
        return {self.at: a}


@register_operator("bone_anchor", implementation="clamp",
                   family="mechanics", set="muscle_particle", kind="lateral")
class BoneAnchorClamp(Lateral):
    """Hold the origin cap to the bone as a CONSTRAINT, not as a spring.

    The stock `bone_anchor` is a penalty force, k(x_rest - x) - c v, and a penalty force
    yields: measured on this rig, the anchored cap slid 0.0632 world off its bone while
    the tendon moved 0.0007. Ninety-nine per cent of the contraction went into the muscle
    pulling itself off the skull, and the load never felt it. That is not a stiffness to
    be tuned -- the eye sweep raised k from 9,000 to 300,000 and the run destabilised
    before the slip stopped, which is what a penalty always does: it either yields or it
    goes stiff enough to break the substep.

    A bone is a Dirichlet condition. This zeroes the anchored points' velocity and
    returns them to their rest position each step, so they cannot move at any load and
    no stiffness has to be chosen. `MAY_MUTATE_INTEGRATED_STATE` is declared because
    that is exactly what a kinematic boundary condition does.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["muscle_particle"]
    OUTPUTS = ["muscle_particle"]
    READS = ["pos", "vel"]
    WRITES = ["pos", "vel"]
    MECHANISM_TAGS = ["boundary_condition", "kinematic_constraint", "bone_attachment"]
    PARAM_ROLES = {}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if not hasattr(p, "anchored"):
            return {}
        m = p.anchored
        if not bool(m.any()):
            return {}
        px0, px1 = p.state_schema["pos"]
        vx0, vx1 = p.state_schema["vel"]
        new = p.state.clone()
        new[m, px0:px1] = p.rest[m]        # back to the bone
        new[m, vx0:vx1] = 0.0              # and not moving
        p.state = new
        return {}


@register_operator("bone_block", family="anatomy", set="particle", kind="rewire")
class BoneBlock(Rewire):
    """Fill a box with material points: the bone, as a BODY rather than a constraint.

    The muscle is then attached to it the way the tendon is attached to the sphere --
    the two bodies overlap and the shared MLS-MPM grid carries the force. That is the
    same mechanism at both ends of the muscle, which is the point: an attachment
    modelled as a spring on one end and as a material on the other is not a comparison.

    STIFFNESS IS BOUNDED BY THE SUBSTEP, not by how rigid a bone feels. The wave speed
    is sqrt(E/rho) and the substep must resolve it: at dx = 1/112 and dt_sub = 2e-4 the
    ceiling is E ~ 2000, and anything stiffer silently breaks the MPM rather than
    modelling a harder bone. Rigidity therefore comes from PINNING the block, not from
    its modulus -- pinned points do not move at any load, which is what a bone does.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["bone_particle"]
    OUTPUTS = ["bone_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["morphogenesis_static", "rigid_body"]
    PARAM_ROLES = {"youngs": "bone_stiffness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bone_particle")
        self.lo = np.asarray(params.get("lo", (0.12, 0.41, 0.41)), float)
        self.hi = np.asarray(params.get("hi", (0.24, 0.59, 0.59)), float)
        self.youngs = float(params.get("youngs", 1600.0))
        self.density = float(params.get("density", 2.0))
        self.nu = float(params.get("poisson", 0.25))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        dev = p.state.device
        n = p.n
        j = np.arange(n)
        u = np.stack([(j + 0.5) / n, _radical_inverse(j, 2), _radical_inverse(j, 3)], 1)
        X = self.lo[None, :] + u * (self.hi - self.lo)[None, :]
        new = p.state.clone()
        pa, pb = p.state_schema["pos"]
        new[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = new
        mu = self.youngs / (2 * (1 + self.nu))
        la = self.youngs * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        p.mu = torch.full((n,), mu, device=dev)
        p.la = torch.full((n,), la, device=dev)
        vol = float(np.prod(self.hi - self.lo))
        p.p_vol = torch.full((n,), vol / n, device=dev)
        p.mass = p.p_vol * self.density
        p.register_buffer("rest", torch.as_tensor(X, dtype=torch.float32, device=dev))
        p.register_buffer("active_stress", torch.zeros(n, 3, 3, device=dev))
        print(f"[bone_block] {n} points, {self.lo} -> {self.hi}, E {self.youngs}", flush=True)
        self._done = True
        return {}


@register_operator("pin_region", implementation="clamp",
                   family="mechanics", set="particle", kind="lateral")
class PinRegionClamp(Lateral):
    """The same region held as a CONSTRAINT rather than a penalty.

    A penalty yields under load -- measured on this rig, the stock spring anchor let the
    muscle slide 0.063 world off its bone while the load moved 0.0007. For the bone that
    is not a stiffness to tune but a kind of object: it does not move, at any load.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos", "vel"]
    WRITES = ["pos", "vel"]
    MECHANISM_TAGS = ["boundary_condition", "kinematic_constraint"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.axis = int(params.get("axis", 0))
        self.beyond = float(params.get("beyond", -1e9))     # default: the whole body
        self._m = None

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if self._m is None:
            if not hasattr(p, "rest"):
                p.register_buffer("rest", p.get("pos").clone())
            self._m = p.rest[:, self.axis] > self.beyond
            if not bool(self._m.any()):
                return {}
        px0, px1 = p.state_schema["pos"]
        vx0, vx1 = p.state_schema["vel"]
        new = p.state.clone()
        new[self._m, px0:px1] = p.rest[self._m]
        new[self._m, vx0:vx1] = 0.0
        p.state = new
        return {}

"""run07_ops -- the two things run_01..06 say the scanned eye is missing.

Neither touches `blend_mpm_ops`; both run AFTER it, on the points it seeded.

    seat_attachments   pull each tendon ONTO the globe and push each belly OFF it
    bone_from_origins  give every origin a bone to be embedded in

WHY, from the rig:

  run_01  `bone_anchor` is a penalty spring, and a penalty yields. The anchored cap slid
          0.063 world off its bone -- 99% of the contraction -- while the load moved
          0.0007. Stiffening it did not help: at k = 300,000 the run destabilised before
          the slip stopped, which is what a penalty always does.
  run_04  embedding the origin in a PINNED BODY instead cut the slip to 28% and tripled
          delivery, on a load that kept its shape.
  run_05  a pair attached OFF-AXIS to a socket-held globe turns it 27 deg horizontal and
          36 deg torsion. The simulator can do this; the geometry has to let it.
  audit   in the scanned eye every tendon sits 6-17% of a radius OFF the globe (about 1.3
          grid cells) while the bellies of SR, IR and IO penetrate it by 15-16%. The grip
          is strongest where the muscle should slide and weakest where it should attach.
          `relieve_overlap` fixes the first half -- it pushes bellies out -- but it can
          only push (`max(need, 0)`), so a tendon already outside is never brought in.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Rewire
from plexus.models.registry import register_operator, register_entity
from plexus.models.entities import MPMParticle


# `bench_ops` (the rig) registers the same entity; whichever module is imported first
# owns it, and registering twice is an error rather than a merge
try:
    @register_entity("bone_particle", depth=0,
                     state_schema={"pos": (0, 2), "vel": (2, 4)},
                     render={"color_by": "node_type", "arrows": None})
    class BoneParticle:
        """A material point of bone: the same continuum state as any MPM particle, a
        separate SET because it is a separate body with its own operators (pinned, not
        contractile)."""
        provision = MPMParticle.provision
except ValueError:
    pass


def _radical_inverse(n, base):
    out = np.zeros_like(n, dtype=np.float64)
    f, i = 1.0 / base, n.astype(np.int64).copy()
    while np.any(i > 0):
        out += f * (i % base)
        i //= base
        f /= base
    return out


@register_operator("seat_attachments", family="anatomy", set="muscle_particle", kind="rewire")
class SeatAttachments(Rewire):
    """Put each muscle's two ends where an attachment and a slide belong.

    Radially, against the globe's own measured surface:

      the TENDON cap (s > 1 - cap) is moved to sit `embed` INSIDE the surface, whether it
      started outside or inside. This is the half `relieve_overlap` cannot do, and it is
      the half that matters: a tendon 1.3 grid cells clear of the sclera is not attached
      to it, it is merely near it, and the B-spline stencil transfers a fraction of the
      force it should.

      the BELLY (s < 1 - cap) is moved to sit at least `standoff` OUTSIDE it. An overlap
      in MLS-MPM is a weld -- the shared grid transfers momentum wherever two bodies share
      cells -- so a belly inside the sclera glues the whole arc of contact and the globe
      cannot turn.

    The globe's surface is measured from the globe's OWN particles rather than assumed
    spherical, because the scanned retina is not a sphere: its radius runs 0.33 to 0.76 of
    the mesh's units. A spherical approximation would seat some tendons in the vitreous
    and leave others in mid-air.
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
    MECHANISM_TAGS = ["morphogenesis_static", "tendon_attachment", "contact_relief"]
    PARAM_ROLES = {"embed": "tendon_embedding_depth", "standoff": "belly_clearance",
                   "cap": "tendon_cap_fraction"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle_particle")
        self.globe = params.get("globe", "mpm_particle")
        self.centre = np.asarray(params.get("center", (0.5, 0.5, 0.46)), float)
        self.embed = float(params.get("embed", 0.012))       # depth INTO the sclera
        self.standoff = float(params.get("standoff", 0.010))  # belly clearance
        self.cap = float(params.get("cap", 0.12))
        self.n_dir = int(params.get("n_dir", 64))
        self._done = False

    def _surface(self, X, dirs):
        """Radius of the globe's surface along each of `dirs`, from its own points.

        A direction-binned maximum: the globe's particles are bucketed by their unit
        direction from the centre and the largest radius in each bucket is that
        direction's surface. Crude, but it follows a non-spherical globe, which is the
        only property required here.
        """
        loc = X - self.centre[None, :]
        r = np.linalg.norm(loc, axis=1).clip(1e-12)
        u = loc / r[:, None]
        from scipy.spatial import cKDTree
        # a fixed set of probe directions, nearest-neighbour interpolated
        g = np.random.default_rng(0).normal(size=(self.n_dir * 8, 3))
        g /= np.linalg.norm(g, axis=1, keepdims=True)
        tree = cKDTree(u)
        rad = np.array([r[tree.query_ball_point(d, 0.25) or [int(tree.query(d)[1])]].max()
                        for d in g])
        gt = cKDTree(g)
        return gt, rad

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        q = H.level(self.globe)
        dev = p.state.device
        X = p.get("pos").detach().cpu().numpy()
        G = q.get("pos").detach().cpu().numpy()
        s = p.s.detach().cpu().numpy()
        par = p.parent.detach().cpu().numpy()

        gt, rad = self._surface(G, None)
        loc = X - self.centre[None, :]
        r = np.linalg.norm(loc, axis=1).clip(1e-12)
        u = loc / r[:, None]
        r_surf = rad[gt.query(u)[1]]

        is_tendon = s > (1.0 - self.cap)
        target = np.where(is_tendon, r_surf - self.embed, r_surf + self.standoff)
        # tendons are pulled IN if they are out; bellies are pushed OUT if they are in
        new_r = np.where(is_tendon, np.minimum(r, target), np.maximum(r, target))
        Xn = self.centre[None, :] + new_r[:, None] * u

        moved_t = float(np.mean(np.abs(new_r - r)[is_tendon] > 1e-6))
        moved_b = float(np.mean(np.abs(new_r - r)[~is_tendon] > 1e-6))
        st = p.state.clone()
        pa, pb = p.state_schema["pos"]
        st[:, pa:pb] = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
        p.state = st
        p.register_buffer("rest", torch.as_tensor(Xn, dtype=torch.float32, device=dev))
        gap = (r - r_surf)[is_tendon]
        print(f"[seat_attachments] tendons seated {self.embed:+.3f} inside the surface "
              f"(they were {gap.mean():+.4f} +- {gap.std():.4f} from it); moved "
              f"{100 * moved_t:.0f}% of tendon points and {100 * moved_b:.0f}% of belly "
              f"points", flush=True)
        self._done = True
        return {}


@register_operator("bone_from_origins", family="anatomy", set="particle", kind="rewire")
class BoneFromOrigins(Rewire):
    """A bone nodule around every muscle origin, so the origin is EMBEDDED, not sprung.

    One sphere per muscle, centred on that muscle's anchored cap and large enough to
    swallow it. They are filled uniformly and then pinned, so a muscle pulling on one
    pulls on something that does not move -- which is what a skull is, and what a penalty
    spring measurably is not.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["bone_particle"]
    OUTPUTS = ["bone_particle"]
    READS = ["pos"]
    WRITES = ["pos"]
    MECHANISM_TAGS = ["morphogenesis_static", "rigid_body", "bone_attachment"]
    PARAM_ROLES = {"youngs": "bone_stiffness", "pad": "nodule_margin"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "bone_particle")
        self.muscles = params.get("muscles", "muscle_particle")
        self.pad = float(params.get("pad", 1.45))
        self.youngs = float(params.get("youngs", 1600.0))
        self.density = float(params.get("density", 2.0))
        self.nu = float(params.get("poisson", 0.25))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        p = H.level(self.at)
        m = H.level(self.muscles)
        dev = p.state.device
        Y = m.get("pos").detach().cpu().numpy()
        s = m.s.detach().cpu().numpy()
        par = m.parent.detach().cpu().numpy()
        M = int(par.max()) + 1
        anch = m.anchored.detach().cpu().numpy()

        cen, rad = [], []
        for i in range(M):
            sel = (par == i) & anch
            if not sel.any():
                sel = (par == i) & (s < 0.15)
            pts = Y[sel]
            c = pts.mean(0)
            cen.append(c)
            rad.append(self.pad * float(np.linalg.norm(pts - c, axis=1).max()))
        cen, rad = np.asarray(cen), np.asarray(rad)

        per = p.n // M
        j = np.arange(per)
        u1, u2, u3 = (j + 0.5) / per, _radical_inverse(j, 2), _radical_inverse(j, 3)
        rr = rad[:, None] * u1[None, :] ** (1.0 / 3.0)
        ct = 2 * u2 - 1.0
        st_ = np.sqrt(np.clip(1 - ct ** 2, 0, None))
        ph = 2 * np.pi * u3
        X = np.concatenate([
            cen[i][None, :] + np.stack([rr[i] * st_ * np.cos(ph), rr[i] * st_ * np.sin(ph),
                                        rr[i] * ct], 1) for i in range(M)])
        if len(X) < p.n:
            X = np.concatenate([X, np.repeat(X[-1:], p.n - len(X), axis=0)])
        X = X[:p.n]

        st = p.state.clone()
        pa, pb = p.state_schema["pos"]
        st[:, pa:pb] = torch.as_tensor(X, dtype=torch.float32, device=dev)
        p.state = st
        mu = self.youngs / (2 * (1 + self.nu))
        la = self.youngs * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        p.mu = torch.full((p.n,), mu, device=dev)
        p.la = torch.full((p.n,), la, device=dev)
        vol = float((4.0 / 3.0 * np.pi * rad ** 3).sum())
        p.p_vol = torch.full((p.n,), vol / p.n, device=dev)
        p.mass = p.p_vol * self.density
        p.register_buffer("rest", torch.as_tensor(X, dtype=torch.float32, device=dev))
        p.register_buffer("active_stress", torch.zeros(p.n, 3, 3, device=dev))
        print(f"[bone_from_origins] {M} nodules, radii "
              + " ".join(f"{v:.3f}" for v in rad), flush=True)
        self._done = True
        return {}

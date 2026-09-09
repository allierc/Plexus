"""The extracellular matrix as MPM (material point method) material, and the stiff blocks that
confine it.

The matrix is a continuum the growing tissue has to push its way into. Every operator here acts
on a set of MPM particles, and the tissue enters as a prescribed moving BOUNDARY rather than as
a second body -- so these describe how an epithelium loads a matrix, not how the two negotiate.

In the order they appear below:

    ecm_seed        seed       the box MINUS a cavity, as aligned fibres, once at frame 0
    ecm_from_cell   lateral    the tissue surface as a moving boundary the particles feel
    ecm_stress      lateral    |J - 1|, banded, so the stress front is visible in a movie
    cell_exclude    structural the hard backstop: no matrix particle inside the lumen
    mpm_block       entity     a material point of a solid block
    block_seed      seed       two slabs beyond a free gap, as a second and much stiffer set
    block_stress    lateral    the block's own strain, at its own full scale

then the two implementations of `ecm_from_cell`, which differ in where the surface comes from:

    ecm_from_cell[sphere]   a prescribed ball of radius r(t): a stand-in with a known answer
    ecm_from_cell[replay]   a recorded epithelium's own apical surface, frame by frame

`ecm_seed` and `block_seed` are not redundant: one fills the complement of a cavity with aligned
fibres, the other fills two slabs with a jittered lattice. Same family, same module, different
geometry. `ecm_stress` and `block_stress` ARE the same body, kept apart only by a module-level
history list per set; moving that onto the Level is what would let one operator serve both.
"""
from __future__ import annotations
import math
import torch
from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator
from plexus.models.entities import MPMParticle
from plexus.models.registry import register_entity, register_operator
from plexus.models.state import spatial_schema


STRESS_HISTORY: list = []
BALL_RADIUS: list = []

# The raw scalar, not only the band. Banding at SIMULATION time would make the colour scale a
# property of the run: `stress_scale` baked into 8 levels, everything above it clipped to the top
# band, and changing the palette costing a full re-simulation. A scale that resolves the front
# early in a run leaves most of the matrix saturated late in it, and once the numbers are banded
# no re-render can recover the gradient. Kept as float16, about 0.1% of a particle's trajectory,
# so that the renderer bands it and a new palette or scale costs only a re-render.
STRESS_RAW: list = []

# The reaction the tissue never felt. `ecm_from_cell` computes the force the tissue puts on the
# matrix, and by Newton's third law that force has an equal and opposite partner on the tissue --
# which a REPLAY has nowhere to put, its tissue pass having finished before the matrix pass began.
# Recording it here is what makes the second half of the coupling possible: `ecm_load` reads this
# map in a later tissue pass and pushes back with it. One row per frame, as an equirectangular map
# of pressure by direction.
PRESSURE_HISTORY: list = []


# --------------------------------------------------------------------------- seeding
# Canonical `ecm_seed`, alias `seed_ecm`. Both spellings stay registered because archived
# specifications use each of them, and a rename that leaves the corpus behind is a rename that
# deletes the corpus.


# --------------------------------------------------------------------------- the coupling
@register_operator("ecm_from_cell", family="mechanics", set="particle", kind="lateral",
                   implementation="sphere")
class CellToECMSphere(Lateral):
    """The growing cell ball as a moving boundary the matrix feels, with the ball prescribed
    rather than simulated: a sphere of radius r(t).

    particle -> particle: reads pos, emits an external acceleration the MPM substep consumes.

        r(t) = min(r0 + v t, r_max)
        d_i  = r(t) - |x_i - c|                      penetration depth, positive when inside
        a_i  = k d_i (x_i - c)/|x_i - c|  -  c_d v_i     only where d_i > 0

    r0 is the initial radius, v is `growth` in world units per unit time and r_max the final
    radius, all in world units. k is the contact stiffness in inverse time squared, so k d is an
    acceleration; c_d is `damp`, a contact damping in inverse time that bleeds the energy a pure
    penalty would ring with.

    The force is a ONE-SIDED penalty on penetration depth, applied only where the ball has
    actually reached. It is not a spring to a rest position: the matrix must be free to be pushed
    and to STAY pushed, because permanent displacement is the observable -- an elastic matrix that
    springs back tells you the ball was there, not what it did.

    This implementation is a stand-in and is labelled one. The real cell ball is a vertex mesh
    with its own energy minimisation; a prescribed sphere reproduces the loading the matrix sees
    without needing it, so the cavity, the fibres, the stable substep and the rendering can all be
    settled against something whose answer is known -- a sphere of radius r(t) pushes exactly as
    hard as r(t) says -- before any of it is trusted with a mesh whose answer is not. Swapping in
    the mesh is then one word on the same operator.

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the vesicle this stands in for). The
    penalty on penetration depth is the normal half of the particle-to-surface contact of Chen,
    Z., Qiu, X., Zhang, X. & Lian, Y. (2015). An improved coupling of finite element method with
    material point method. Comput. Methods Appl. Mech. Engrg. 293:1-19 -- with the surface
    analytic and PRESCRIBED, so unlike that scheme no reaction is returned to it and momentum is
    not conserved across the interface. `mesh_contact` is the two-way form.
    """
    EMIT = "mpm_acceleration"       # consumed by the substep as a_ext, like mpm_anchor
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["cell_matrix_contact", "moving_boundary", "confinement"]
    PARAM_ROLES = {"k": "contact_stiffness", "r0": "initial_radius", "growth": "growth_rate",
                   "r_max": "final_radius", "damp": "contact_damping"}
    REFERENCE = ("Okuda, S. et al. (2018) Sci. Rep. 8:2386 (the vesicle this stands in for). The "
                 "penalty on penetration depth is the normal half of the particle-to-surface "
                 "contact of Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015) Comput. Methods Appl. "
                 "Mech. Engrg. 293:1-19 (ICFEMP), doi:10.1016/j.cma.2015.04.005 -- with the "
                 "surface analytic and PRESCRIBED, so unlike ICFEMP no reaction is returned to it "
                 "and momentum is not conserved across the interface. `mesh_contact` is the "
                 "two-way form.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.k = float(params["k"])
        self.r0 = float(params.get("r0", 0.05))
        self.r_max = float(params.get("r_max", 0.30))
        # PER FRAME, NOT PER SUBSTEP. The radius is the experiment's clock, and it must not depend
        # on how finely the mechanics is integrated -- otherwise halving the substep for stability
        # would halve the growth rate, and a numerical choice would silently become a biological
        # one.
        self.growth = float(params.get("growth", 0.0008))
        self.damp = float(params.get("damp", 0.0))
        self._r = self.r0
        self._frame = -1

    def radius(self):
        return self._r

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype

        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:                       # advance ONCE per frame, not per substep
            self._frame = f
            self._r = min(self.r_max, self._r + self.growth)

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        depth = (self._r - r).squeeze(1)                       # > 0 where the ball has reached
        hit = depth > 0
        acc = torch.zeros_like(pos)
        if hit.any():
            n = d[hit] / r[hit]
            a = self.k * depth[hit][:, None] * n                # push outward along the normal
            if self.damp > 0:
                # `lvl.state` is the TENSOR, not a dict of blocks: `"vel" in lvl.state` raises
                # inside Tensor.__contains__. Never fired because `damp` defaults to 0, which is
                # exactly what a landmine looks like -- the first run to set `damp` would have died.
                v = lvl.get("vel")[hit] if "vel" in lvl.state_schema else None
                if v is not None:
                    a = a - self.damp * (v * n).sum(1, keepdim=True).clamp(max=0.0) * n
            acc[hit] = a
        BALL_RADIUS.append(float(self._r))          # the ball's own clock, for the render
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc * lvl.occ[:, None].float()}


# --------------------------------------------------------------------------- what you can see
@register_operator("ecm_stress", family="hierarchy", set="particle", kind="lateral")
class ECMStress(Lateral):
    """A measurement, as an operator: colour the matrix by how hard it is being squeezed, so the
    stress front is the thing a movie shows rather than the positions.

    particle -> particle: reads the deformation gradient F, writes a colour band on node_type.

        s_i    = |det F_i - 1|                       the local volume change
        band_i = floor(K min(s_i / S, 1))            an integer in [0, K)

    det F is the ratio of a particle's current volume to its rest volume, so s is dimensionless
    and reads zero for unstrained material, positive for both compression and extension. S is
    `scale`, the strain at which the palette saturates, and K is `bands`.

    A propagation is invisible in a movie of positions: the fibres near the ball move a little,
    the ones behind them less, and by eye that reads as the middle wiggling, when what is
    happening is a stress front travelling outward. |det F - 1| is chosen over a von Mises
    invariant because it needs no material constants, so the colour means the same thing across a
    sweep in which stiffness is exactly what varies.

    It is banded into integers rather than written as a float because the renderer's `color_by` is
    a palette INDEX and not a continuous map. The set declares K types carrying identical material
    and different colours, so the index is decoration and cannot change the physics it draws.

    Reference: the deformation gradient is that of Hu, Y. et al. (2018). A moving least squares
    material point method with displacement discontinuity and two-way rigid body coupling. ACM
    Trans. Graph. 37(4):150.
    """
    EMIT = None                     # writes a colour channel in place; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "strain_visualisation"]
    PARAM_ROLES = {"scale": "strain_full_scale", "bands": "colour_bands"}
    REFERENCE = ("Hu, Y. et al. (2018). A moving least squares material point method with "
                 "displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. "
                 "37(4):150.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        # THE FULL-SCALE VALUE IS FIXED, NOT PER-FRAME. Auto-scaling each frame to its own
        # maximum would make every run look identically stressed -- the brightest fibre is always
        # full brightness -- and a sweep over stiffness would then show no difference at all,
        # which is the one thing it exists to show.
        self.scale = float(params.get("scale", 0.15))
        self.bands = int(params.get("bands", 8))
        self.channel = str(params.get("channel", "node_type"))
        # WHICH STRAIN. `vol` is |J-1|, the local VOLUME change, and it is the honest default because it
        # needs no material constants. But it is blind to the deformation this experiment actually
        # produces: a matrix pushed outward by a growing sphere is SHEARED and dragged far more than it
        # is compressed, and MLS-MPM's fixed-corotated material resists volume change stiffly, so
        # |J-1| stays near zero while the fibres are visibly splayed. That is why movies of a plainly
        # moving matrix read as unstressed -- the MEASURE, not the mechanics. `dev` is the
        # volume-normalised equivalent deviatoric strain, which is where the signal is.
        # `vol` |J-1| volume change | `dev` equivalent deviatoric STRAIN | `vonmises` the von Mises
        # invariant of the CAUCHY STRESS the solver itself computed (`mpm_scatter: store_stress: true`).
        # `vonmises` is the physical one: it is in stress units, it weights shear the way this material's
        # own mu and la do, and it is the same tensor that generated the grid forces -- not a proxy
        # re-derived from F with a constitutive law chosen by the diagnostic.
        self.measure = str(params.get("measure", "vol"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        if self.measure == "vonmises":
            sig = getattr(lvl, "sigma", None)
            if sig is None:
                if not getattr(self, "_warned_vm", False):
                    print("[ecm_stress] measure=vonmises but no `sigma` buffer -- the scatter was not "
                          "asked to keep it (`store_stress: true`). Falling back to |J-1|, which is a "
                          "DIFFERENT quantity: read the colours accordingly.", flush=True)
                    self._warned_vm = True
            else:
                tr = sig.diagonal(dim1=-2, dim2=-1).sum(-1)
                eye = torch.eye(sig.shape[-1], device=sig.device, dtype=sig.dtype)
                dv = sig - (tr / 3.0)[:, None, None] * eye
                vm = torch.sqrt((1.5 * (dv * dv).sum((-1, -2))).clamp_min(0.0))
                band = ((vm / max(self.scale, 1e-9)).clamp(0, 1)
                        * (self.bands - 1)).round().long()
                ch = getattr(lvl, self.channel, None)
                if ch is not None:
                    ch[:] = band.to(ch.dtype).reshape(ch.shape)
                STRESS_HISTORY.append(band.detach().to("cpu", torch.uint8).numpy())
                STRESS_RAW.append(vm.detach().to("cpu", torch.float16).numpy())
                return {}
        J = torch.linalg.det(F)
        if self.measure == "dev":
            # Volume-normalised left Cauchy-Green, then its deviator: shape change with the volume
            # change divided out, so `dev` and `vol` are independent readings of the same F rather than
            # two views of mostly the same number.
            B = F @ F.transpose(-1, -2)
            Bb = B / J.abs().clamp_min(1e-9).pow(2.0 / 3.0)[:, None, None]
            tr = Bb.diagonal(dim1=-2, dim2=-1).sum(-1)
            eye = torch.eye(Bb.shape[-1], device=Bb.device, dtype=Bb.dtype)
            dev = Bb - (tr / 3.0)[:, None, None] * eye
            s = torch.sqrt((1.5 * (dev * dev).sum((-1, -2))).clamp_min(0.0)) / max(self.scale, 1e-9)
        else:
            s = (J - 1.0).abs() / max(self.scale, 1e-9)
        band = (s.clamp(0, 1) * (self.bands - 1)).round().long()
        # `node_type` IS A BUFFER, NOT A STATE BLOCK. It is registered by the provision
        # (base.py:348) alongside mass/F/C, so `lvl.get()` -- which slices the state matrix --
        # raises KeyError on it. Written directly, in place, so the buffer the renderer reads and
        # the buffer the material was assigned from stay the same object.
        ch = getattr(lvl, self.channel, None)
        if ch is None:
            if not getattr(self, "_warned", False):
                print(f"[ecm_stress] no `{self.channel}` buffer -- the movie will not be "
                      f"stress-coloured", flush=True)
                self._warned = True
            return {}
        ch[:] = band.to(ch.dtype).reshape(ch.shape)
        STRESS_HISTORY.append(band.detach().to("cpu", torch.uint8).numpy())
        STRESS_RAW.append((s * max(self.scale, 1e-9)).detach().to("cpu", torch.float16).numpy())
        return {}


@register_operator("ecm_from_cell", family="mechanics", set="particle", kind="lateral",
                   implementation="replay")
class CellToECMReplay(Lateral):
    """The same contact, against a real epithelium's recorded surface instead of a sphere.

    particle -> particle: reads pos and a recorded surface, emits an external acceleration.

        d_i = S R(theta_i, phi_i) - |x_i - c|        penetration depth
        a_i = k d_i (x_i - c)/|x_i - c|                 only where d_i > 0

    R(theta, phi) is an angular radius map: per frame, the distance from the tissue centroid to
    the furthest apical vertex in that direction. S is `scale`, a dimensionless rescaling of the
    recorded surface, and k the contact stiffness in inverse time squared.

    A particle's own direction gives its bin, and one comparison decides whether the tissue has
    reached it -- O(1) per particle rather than a point-in-mesh test against thousands of faces,
    which is what makes hundreds of thousands of particles affordable per frame. The price is an
    assumption: the vesicle must be STAR-SHAPED about its centroid, and a run in which it stops
    being has left this operator's domain of validity.

    The map is centroid-referenced, so `centre` pins the tissue at the box centre and the
    vesicle's own translational drift is dropped. That is deliberate. The drift is a few percent
    of the radius, and it would otherwise slide the tissue off the cavity it was seeded into,
    turning a symmetric loading experiment into an accidentally one-sided one. What the matrix
    sees is the tissue's shape and growth, not its wandering.

    The coupling is one-way, and the reason is bookkeeping rather than modelling: a replay has no
    live tissue for the reaction to be returned to, its tissue pass having already finished. So
    this shows how a growing epithelium LOADS a matrix, and does not show the matrix shaping the
    epithelium back. The reaction is recorded in PRESSURE_HISTORY for a later pass to apply.

    Reference: Okuda, S. et al. (2018). Sci. Rep. 8:2386 (the tissue being replayed). The contact
    is the penalty half of Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg. 293:1-19,
    against a RECORDED surface, so it is one-way by construction; `mesh_contact` returns the
    reaction to the vertices and is the operator to use where the tissue must feel the matrix.
    """
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k", "surface"]
    MECHANISM_TAGS = ["cell_matrix_contact", "moving_boundary", "recorded_tissue"]
    PARAM_ROLES = {"k": "contact_stiffness", "scale": "surface_rescale"}
    REFERENCE = ("Okuda, S. et al. (2018) Sci. Rep. 8:2386 (the tissue being replayed). The "
                 "contact is the penalty half of Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015) "
                 "Comput. Methods Appl. Mech. Engrg. 293:1-19 (ICFEMP), "
                 "doi:10.1016/j.cma.2015.04.005, against a RECORDED surface: the reaction has "
                 "nowhere to go, so this is one-way by construction. `mesh_contact` returns it to "
                 "the vertices and is the operator to use where the tissue must feel the matrix.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.k = float(params["k"])
        self.scale = float(params.get("scale", 1.0))
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(z["smap"], dtype=torch.float32) * self.scale   # [T, nth, nph]
        self.T = int(self.smap.shape[0])
        self._frame = -1
        self._t = 0
        self._dom = None                 # per-row solid angle, built on the first call

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))
        M = self.smap[self._t].to(dev, dt_)
        nth, nph = M.shape

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        it = (th / math.pi * nth).long().clamp(0, nth - 1)
        ip = (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)
        R = M[it, ip]

        depth = R - r
        hit = depth > 0
        acc = torch.zeros_like(pos)
        if hit.any():
            acc[hit] = self.k * depth[hit][:, None] * u[hit]

        # THE REACTION, BINNED BY DIRECTION AND TURNED INTO A PRESSURE. Sum the contact force in each
        # (theta, phi) bin and divide by the AREA that bin covers on the tissue surface,
        # R^2 * dOmega -- otherwise the poles, whose bins are slivers, would report a pressure many
        # times the equator's for the same force, and a later tissue pass would grow a waist.
        # dOmega = (2pi/nph) * (cos theta_lo - cos theta_hi), exact per row rather than sin(theta)
        # d(theta), which diverges from it precisely at the poles where it matters.
        if self._dom is None or self._dom.shape[0] != nth:
            e = torch.linspace(0, math.pi, nth + 1, device=dev, dtype=dt_)
            self._dom = (e[:-1].cos() - e[1:].cos()) * (2 * math.pi / nph)
        load = torch.zeros(nth * nph, device=dev, dtype=dt_)
        if hit.any():
            load.index_add_(0, (it[hit] * nph + ip[hit]),
                            (self.k * depth[hit]).to(dt_))
        area = (M * M) * self._dom[:, None]
        PRESSURE_HISTORY.append(
            (load.reshape(nth, nph) / area.clamp_min(1e-12)).detach().to("cpu").numpy())
        BALL_RADIUS.append(float(M.median()))
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc * lvl.occ[:, None].float()}


@register_operator("cell_exclude", family="boundary", set="particle", kind="structural")
class CellExclude3D(Structural):
    """A hard non-penetration backstop: no matrix particle may end a frame inside the tissue.

    particle -> particle: reads pos and the recorded surface, writes pos and vel in place.

        R_i = S R(theta_i, phi_i) (1 + skin)
        x_i <- c + R_i u_i          and     v_i <- v_i - min(v_i . u_i, 0) u_i

    u_i is the unit direction from the tissue centre to the particle. Any particle found inside is
    projected out onto the surface with a thin skin, `skin` being that clearance as a fraction of
    the local radius, and its INWARD radial velocity is removed so it does not simply re-enter on
    the next substep.

    It exists because `ecm_from_cell` is a penalty, which punishes penetration after the fact
    rather than preventing it, and three things let a penalty lose. The MPM scatter CLAMPS the
    external acceleration, so past a depth of a_max/k the restoring force stops growing no matter
    how deep the particle is -- the right behaviour for stability and the wrong one for a
    constraint. The tissue surface sweeps, advancing every frame whether or not the matrix has got
    out of the way, so a particle only has to be out-accelerated once to be left behind. And the
    surface is a smoothed angular map, so where the smoothing cuts a bump, particles sit inside
    the true mesh while the map says they are outside it.

    The penalty is kept, because it is what generates the stress being measured; this runs after
    it. A boundary that must not be crossed is a projection, not a force.

    It is RIGID, which is honest here only because the coupling is one-way: the tissue's shape is
    prescribed, so nothing is decided by letting it win every contact -- it was always going to.
    In a two-way run this operator would be taking a side, and the projection would have to become
    a correction shared between the two bodies.

    Reference: Plexus (this work); the surface is Okuda, S. et al. (2018). Sci. Rep. 8:2386. It
    backs up the penalty contact of Chen, Z. et al. (2015). Comput. Methods Appl. Mech. Engrg.
    293:1-19, whose own reported weakness is exactly this one -- penetration through a severely
    deformed contact surface. Forbidding it outright rather than projecting afterwards is BFEMP:
    Li, X., Fang, Y., Li, M. & Jiang, C. (2022). Comput. Methods Appl. Mech. Engrg. 390:114350.
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["surface"]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    MECHANISM_TAGS = ["non_penetration", "rigid_contact", "moving_boundary"]
    PARAM_ROLES = {"skin": "projection_skin_fraction", "scale": "surface_rescale"}
    REFERENCE = ("Plexus (this work); the surface is Okuda, S. et al. (2018) Sci. Rep. 8:2386. It "
                 "backs up the penalty contact of Chen, Z., Qiu, X., Zhang, X. & Lian, Y. (2015) "
                 "Comput. Methods Appl. Mech. Engrg. 293:1-19, doi:10.1016/j.cma.2015.04.005, "
                 "whose own reported weakness is exactly this one -- penetration through a "
                 "severely deformed contact surface. Forbidding it outright instead of projecting "
                 "afterwards is BFEMP (Li, X., Fang, Y., Li, M. & Jiang, C. (2022) CMAME 390:114350, "
                 "doi:10.1016/j.cma.2021.114350), at the cost of an implicit solve.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        import numpy as _np
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.scale = float(params.get("scale", 1.0))
        self.skin = float(params.get("skin", 0.004))
        z = _np.load(str(params["surface"]))
        self.smap = torch.as_tensor(z["smap"], dtype=torch.float32) * self.scale
        self.T = int(self.smap.shape[0])
        self._frame, self._t, self._n = -1, 0, 0

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        dev, dt_ = pos.device, pos.dtype
        f = int(getattr(H, "frame", -1) or -1)
        if f != self._frame:
            self._frame = f
            self._t = min(self.T - 1, max(0, f))
        M = self.smap[self._t].to(dev, dt_)
        nth, nph = M.shape

        c = torch.tensor(self.centre, device=dev, dtype=dt_)
        d = pos - c
        r = d.norm(dim=1).clamp_min(1e-9)
        u = d / r[:, None]
        th = torch.acos(u[:, 2].clamp(-1, 1))
        ph = torch.atan2(u[:, 1], u[:, 0]) % (2 * math.pi)
        R = M[(th / math.pi * nth).long().clamp(0, nth - 1),
              (ph / (2 * math.pi) * nph).long().clamp(0, nph - 1)]
        inside = r < R
        # A MASSLESS PARTICLE IS NOT MATERIAL. `bm_secrete` parks its unsecreted reserve
        # at the centre with mass 0; without this line the projection would fire on every dormant
        # particle and paste the entire reserve onto the surface as a second, fake sheet.
        m = getattr(lvl, "mass", None)
        if m is not None and m.shape[0] == inside.shape[0]:
            inside = inside & (m > 0)
        n_in = int(inside.sum())
        if n_in:
            # ONTO THE SURFACE PLUS A SKIN. Exactly onto it would leave the particle at depth 0, where
            # the penalty is also 0, so the next substep's own motion puts it straight back in.
            target = R * (1.0 + self.skin)
            pos[inside] = c + u[inside] * target[inside][:, None]
            if "vel" in lvl.state_schema:
                v = lvl.get("vel")
                vr = (v * u).sum(1)                       # radial component, inward is negative
                v[inside] = v[inside] - torch.minimum(
                    vr[inside], torch.zeros_like(vr[inside]))[:, None] * u[inside]
        if f <= 1 or (n_in and n_in > self._n * 4 + 50):
            print(f"[cell_exclude] frame {f}: {n_in} particle(s) projected out of the tissue",
                  flush=True)
        self._n = n_in
        return {}


BLOCK_STRESS: list = []
BLOCK_RAW: list = []            # the un-banded scalar -- see `ecm_ops.STRESS_RAW`


# --------------------------------------------------------------------------- the entity
@register_entity(
    # `spatial_schema` is a `dim -> StateSchema` callable, not a fixed dict: this set runs in 3D
    # specifications, and a hard-coded 2D schema would silently truncate its state.
    "mpm_block", depth=0,
    state_schema=spatial_schema,
    render={"color_by": "node_type", "arrows": None},
)
class MPMBlock:
    """A material point of a solid block: identical continuum state to the matrix's
    `mpm_particle` -- the deformation gradient F, the affine velocity C, mass, the Lame parameters
    mu and lambda, and the rest volume p_vol. The stock provision allocates it.

    It exists as a separate registration because the entity is resolved BY SET NAME: the registry
    is looked up under the set's own name and falls back to a bare position/velocity schema for
    anything unregistered. A set called `mpm_block` with every MPM operator pointed at it would
    otherwise have no F, and the run would die in `mpm_strain` with a missing attribute -- which
    reads like a bug in the operator and is a missing registration.
    """
    provision = MPMParticle.provision


@register_operator("block_seed", family="seed", set="particle", kind="seed")
class BlockSeed(Structural):
    """Fill the two slabs beyond a free gap with particles, once, at frame 0: the rigid walls the
    matrix and the tissue are confined between.

    particle -> particle: rewrites every position in the set at the opening of the trajectory.

        occupied = { x : |x_a - c_a| > g }           a = the confined axis

    g is `gap_half`, the half-width of the free gap in world units, and c the domain centre. The
    slabs run from there to the box wall, less `margin`. The particles are a jittered lattice
    rather than fibres, because a block is isotropic where the matrix is not.

    The block is a SECOND MPM set with its own Lame parameters, typically two orders of magnitude
    stiffer than the matrix, which is what lets it act as a wall while still being a continuum
    that deforms measurably under load.

    Reference: Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for
    history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196 (MPM).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["gap_half"]
    MECHANISM_TAGS = ["solid_obstacle", "material_seeding"]
    PARAM_ROLES = {"gap_half": "free_half_gap", "axis": "confined_axis",
                   "centre": "domain_centre", "margin": "wall_clearance"}
    REFERENCE = ("Sulsky, D., Chen, Z. & Schreyer, H. L. (1994). A particle method for "
                 "history-dependent materials. Comput. Methods Appl. Mech. Eng. 118:179-196.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_block")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        self.axis = int(params.get("axis", 2))
        self.gap_half = float(params["gap_half"])
        # A WALL CLEARANCE, not a cosmetic margin. `mpm_gather` treats the outer `wall_contact` shell
        # specially and a particle seeded exactly on the boundary starts inside that shell, so the
        # block's outermost layer would begin the run in a contact correction it never leaves.
        self.margin = float(params.get("margin", 0.012))
        self.seed = int(params.get("seed", 0))
        self._done = False

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n, D = pos.shape
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        ax, c = self.axis, self.centre[self.axis]
        lo, hi = self.margin, 1.0 - self.margin

        # A JITTERED LATTICE, NOT UNIFORM NOISE. MPM resolves a material by how evenly its particles
        # fill the cells: uniform random sampling leaves holes and clumps at this density, and a clump
        # is a stiffness the material was not given. One particle per lattice site plus a fraction of
        # the spacing is what the stock block provisions do.
        half = n // 2
        per = []
        for sgn in (+1.0, -1.0):
            t0, t1 = c + sgn * self.gap_half, c + sgn * (0.5 - self.margin)
            th = abs(t1 - t0)
            # cube-root split of the slab's aspect ratio, so the lattice is roughly isotropic
            k = max(1, int(round((half * th * th / max((hi - lo) ** 2, 1e-9)) ** (1.0 / 3.0))))
            nx = max(1, int(round((half / max(k, 1)) ** 0.5)))
            grid = torch.stack(torch.meshgrid(
                torch.linspace(lo, hi, nx), torch.linspace(lo, hi, nx),
                torch.linspace(min(t0, t1), max(t0, t1), k), indexing="ij"), -1).reshape(-1, 3)
            per.append(grid)
        p = torch.cat(per)
        if p.shape[0] < n:
            p = torch.cat([p, p[: n - p.shape[0]]])
        p = p[:n]
        # reorder the columns so column `ax` is the slab-normal one
        if ax != 2:
            idx = [0, 1, 2]; idx[2], idx[ax] = idx[ax], idx[2]
            p = p[:, idx]
        spacing = (hi - lo) / max(int(round(n ** (1.0 / 3.0))), 1)
        p = p + (torch.rand(p.shape, generator=g) - 0.5) * spacing * 0.35
        p[:, ax] = p[:, ax].clamp(min(lo, c - 0.5 + self.margin), max(hi, c + 0.5 - self.margin))
        p = p.clamp(lo, hi)
        # anything that landed in the free gap is pushed back into its own slab
        d = p[:, ax] - c
        bad = d.abs() < self.gap_half
        if bad.any():
            p[bad, ax] = c + torch.sign(torch.where(d[bad] == 0, torch.ones_like(d[bad]), d[bad])) \
                * (self.gap_half + spacing * 0.5)
        lvl.get("pos")[:] = p.to(pos.device, pos.dtype)
        self._done = True
        print(f"[block_seed] {n} particles in two slabs beyond +/-{self.gap_half:.4g} of "
              f"{c:.3g} on axis {ax} ({100 * (1 - 2 * self.gap_half):.0f}% of the box)", flush=True)
        return {}


@register_operator("block_stress", family="hierarchy", set="particle", kind="lateral")
class BlockStress(Lateral):
    """The block's own local volume change, banded, so its deformation is visible at its own
    scale rather than at the matrix's.

    particle -> particle: reads F, writes a colour band on node_type. Same body as `ecm_stress`:

        s_i    = |det F_i - 1|
        band_i = floor(K min(s_i / S, 1))

    Separate from `ecm_stress` only because `scale` must differ: a block two orders of magnitude
    stiffer strains two orders of magnitude less under the same load, so sharing a full-scale S
    would leave it uniformly in the zero band.

    Reference: the deformation gradient is that of Hu, Y. et al. (2018). ACM Trans. Graph.
    37(4):150 (MLS-MPM).
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "strain_visualisation"]
    PARAM_ROLES = {"scale": "strain_full_scale", "bands": "colour_bands"}
    REFERENCE = ("Hu, Y. et al. (2018). A moving least squares material point method with "
                 "displacement discontinuity and two-way rigid body coupling. ACM Trans. Graph. "
                 "37(4):150.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_block")
        # A DIFFERENT FULL SCALE FROM THE MATRIX'S, and it has to be: the block is ~130x stiffer, so
        # the same load strains it ~130x less. At the matrix's 0.08 the block would read as
        # uniformly unstrained in every frame, which is the claim "it is rigid" -- the claim this
        # operator exists to test.
        self.scale = float(params.get("scale", 0.004))
        self.bands = int(params.get("bands", 8))
        # Same three options as `ecm_stress`; `vonmises` reads the Cauchy stress the accumulate scatter
        # cached (`store_stress: true`) rather than re-deriving anything from F.
        self.measure = str(params.get("measure", "vol"))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        sig = getattr(lvl, "sigma", None) if self.measure == "vonmises" else None
        if sig is not None:
            tr = sig.diagonal(dim1=-2, dim2=-1).sum(-1)
            eye = torch.eye(sig.shape[-1], device=sig.device, dtype=sig.dtype)
            dv = sig - (tr / 3.0)[:, None, None] * eye
            s = torch.sqrt((1.5 * (dv * dv).sum((-1, -2))).clamp_min(0.0)) / max(self.scale, 1e-9)
        else:
            if self.measure == "vonmises" and not getattr(self, "_warned", False):
                print("[block_stress] measure=vonmises but no `sigma` buffer -- falling back to "
                      "|J-1|, a DIFFERENT quantity", flush=True)
                self._warned = True
            J = torch.linalg.det(F)
            s = (J - 1.0).abs() / max(self.scale, 1e-9)
        band = (s.clamp(0, 1) * (self.bands - 1)).round().long()
        BLOCK_STRESS.append(band.detach().to("cpu", torch.uint8).numpy())
        BLOCK_RAW.append((s * max(self.scale, 1e-9)).detach().to("cpu", torch.float16).numpy())
        return {}

"""ecm_ops -- an extracellular matrix the growing cell ball has to push its way into.

THE EXPERIMENT. Fill the box with a soft fibrous solid, leave a DISC-SHAPED CAVITY in the
middle, drop the cell ball into the cavity and let it grow. Early on the ball is free and grows
as it would in isolation. Then it touches the matrix, and from that moment its shape is no longer
its own: the disc confines it in one axis and lets it spread in the other two, the matrix builds
stress where it is pressed, and the ball either flattens, or breaks out where the matrix is
thinnest, or stalls.

WHY MPM AND NOT A WALL. A rigid obstacle would confine the ball and tell us nothing, because the
answer is fixed before the run starts. A material-point matrix DEFORMS, STORES STRESS AND PUSHES
BACK, so the confinement is an outcome rather than a boundary condition, and it can fail --
which is the interesting case and the one a wall cannot produce.

TWO-WAY, THROUGH ONE GRID. Both bodies scatter into the SAME background grid and both gather
from it, so the coupling is momentum exchange rather than a force model anyone had to invent.
This is the pattern `prototype/eye/` already runs for the globe and its muscles: the first
`mpm_scatter` resets the grid, the second uses `implementation: accumulate` and adds to it, and
one schedule token runs both. One-way coupling was considered and rejected: it deforms the matrix
decoratively while leaving the ball's shape untouched, which produces a convincing movie of
nothing.

THE FIBRES ARE GEOMETRY, NOT YET MECHANICS -- and the distinction matters. `ecm_seed` lays the
particles along straight segments with a shared orientation, so the matrix LOOKS fibrous and the
movie shows fibres being dragged and splayed. But MPM interpolates every particle onto a
continuum grid, so a fibrous ARRANGEMENT of an isotropic material still responds isotropically.
Real fibre reinforcement needs the constitutive model to know the direction. Each particle
therefore carries its unit fibre vector in `fibre`, unused by the stock material and ready for an
anisotropic term -- so the anisotropy is a switch to add, not a rewrite. Until that term exists,
no claim about fibre alignment is supported by this substrate, and saying so here is cheaper than
discovering it from a movie that looks right.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Structural
from plexus.models.registry import register_operator

# THE STRESS OVER TIME, WHICH THE RECORDER DOES NOT KEEP. `node_type` is a BUFFER, and the
# engine saves a buffer once -- its final value -- not once per frame. So a movie coloured by it
# shows the stress AT THE END painted onto particles moving through the whole run: every frame
# looks equally stressed, including the ones before contact, and a propagating front is exactly
# what cannot be seen. The one thing this experiment is for.
#
# Appended here instead, one row per frame, and rendered by run_ecm. A module-level list because
# the operator instance is built inside the engine and there is no handle to it afterwards.
STRESS_HISTORY: list = []
BALL_RADIUS: list = []


# --------------------------------------------------------------------------- seeding
@register_operator("ecm_seed", family="growth", set="particle", kind="structural")
class ECMSeed(Structural):
    """Lay the matrix out ONCE, at frame 0: the box minus a cavity, as fibres.

    A structural operator rather than a set provision because the stock provision seeds a block
    or a ball, and the matrix is neither -- it is the COMPLEMENT of a shape. Writing it as an
    operator also means the cavity is a parameter of the experiment, visible in the spec beside
    the stiffness it is being tested against, rather than a number buried in a seeder.

    The particles are not deleted from the cavity, they are never placed there: `ecm_seed`
    rewrites every position in the set. A cut-out would leave the discarded particles occupying
    memory and mass, and an MPM particle with zero occupancy still costs a scatter.
    """
    EMIT = None                       # rewrites positions in place at frame 0; no integrable delta
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["matrix_seeding", "fibre_alignment", "cavity"]
    PARAM_ROLES = {"cavity_r": "cavity_radius", "cavity_h": "cavity_half_height",
                   "fibre_len": "fibre_length", "n_fibres": "fibre_count",
                   "align": "fibre_alignment_strength"}
    REFERENCE = "Sulsky, D. et al. (1994) Comput. Methods Appl. Mech. Eng. 118:179 (MPM)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.centre = [float(v) for v in params.get("centre", [0.5, 0.5, 0.5])]
        # THE CAVITY IS A DISC, and its two radii are separate numbers on purpose. A sphere would
        # confine the ball equally in every direction, which is the one case where the matrix
        # cannot tell us anything the ball's own mechanics does not: it would stay a ball. A disc
        # is anisotropic confinement -- free in two axes, pinched in the third -- so the shape the
        # tissue takes is a fact about the coupling rather than about the ball.
        self.cavity_r = float(params.get("cavity_r", 0.22))     # in-plane radius
        self.cavity_h = float(params.get("cavity_h", 0.07))     # half-thickness along `axis`
        self.axis = int(params.get("axis", 1))                  # the pinched axis
        self.margin = float(params.get("margin", 0.03))         # keep clear of the domain walls
        self.n_fibres = int(params.get("n_fibres", 900))
        self.fibre_len = float(params.get("fibre_len", 0.16))
        # 0 = isotropic directions, 1 = every fibre parallel to `align_dir`. Anything between is a
        # von-Mises-like spread about it, which is what a real matrix looks like near a boundary.
        self.align = float(params.get("align", 0.0))
        self.align_dir = [float(v) for v in params.get("align_dir", [1.0, 0.0, 0.0])]
        self.jitter = float(params.get("jitter", 0.004))        # across-fibre thickness
        self.seed = int(params.get("seed", 0))
        self._done = False

    def _outside_cavity(self, pos):
        """True where a point is IN the matrix -- i.e. outside the disc-shaped cavity."""
        c = torch.tensor(self.centre, device=pos.device, dtype=pos.dtype)
        d = pos - c
        ax = self.axis
        along = d[:, ax].abs()
        radial = torch.sqrt((d ** 2).sum(1) - d[:, ax] ** 2 + 1e-12)
        return ~((radial < self.cavity_r) & (along < self.cavity_h))

    def forward(self, H, mask=None):
        if self._done:
            return {}
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        n, D = pos.shape
        dev, dt_ = pos.device, pos.dtype
        g = torch.Generator(device="cpu").manual_seed(self.seed)

        # FIBRE DIRECTIONS. Isotropic by default; `align` mixes toward a chosen axis. Normalised
        # AFTER mixing so a partially aligned fibre is still a unit vector.
        d = torch.randn(self.n_fibres, D, generator=g)
        d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
        if self.align > 0:
            a = torch.tensor(self.align_dir, dtype=d.dtype)
            a = a / a.norm().clamp_min(1e-9)
            d = (1.0 - self.align) * d + self.align * a
            d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)

        # FIBRE CENTRES, REJECTION-SAMPLED OUT OF THE CAVITY. Rejection rather than a closed-form
        # placement because the cavity is a parameter: a formula would have to be rewritten for
        # every new cavity shape, and a rejection loop would not.
        lo, hi = self.margin, 1.0 - self.margin
        keep = []
        for _ in range(64):
            c = torch.rand(self.n_fibres * 2, D, generator=g) * (hi - lo) + lo
            c = c[self._outside_cavity(c.to(dev)).cpu()]
            keep.append(c)
            if sum(x.shape[0] for x in keep) >= self.n_fibres:
                break
        centres = torch.cat(keep)[: self.n_fibres]
        if centres.shape[0] < self.n_fibres:                  # cavity larger than the box
            centres = centres.repeat((self.n_fibres // max(centres.shape[0], 1)) + 1, 1)
            centres = centres[: self.n_fibres]

        # PARTICLES ALONG THE FIBRES, evenly spaced with a little across-fibre thickness so a
        # fibre reads as a strand rather than a line of dots.
        per = max(1, n // self.n_fibres)
        t = (torch.arange(per, dtype=torch.float32) / max(per - 1, 1) - 0.5) * self.fibre_len
        p = (centres[:, None, :] + t[None, :, None] * d[:, None, :]).reshape(-1, D)
        f = d[:, None, :].expand(-1, per, -1).reshape(-1, D)
        if p.shape[0] < n:                                    # top up with whole fibres
            k = n - p.shape[0]
            p = torch.cat([p, p[:k]])
            f = torch.cat([f, f[:k]])
        p, f = p[:n], f[:n]
        p = p + torch.randn(p.shape, generator=g) * self.jitter
        p = p.clamp(lo, hi)

        # A FIBRE THAT CROSSES THE CAVITY IS PULLED OUT OF IT. Seeding by centres alone leaves
        # particles inside the cavity whenever a fibre straddles the boundary -- and a single
        # matrix particle sitting where the cell ball is about to be is not a small error: it is
        # in contact from frame 0, so "the moment of first contact", which is the one event this
        # experiment exists to observe, would be frame 0 for the whole run.
        inside = ~self._outside_cavity(p.to(dev)).cpu()
        if inside.any():
            c = torch.tensor(self.centre, dtype=p.dtype)
            v = p[inside] - c
            rad = v.clone(); rad[:, self.axis] = 0
            rn = rad.norm(dim=1, keepdim=True).clamp_min(1e-9)
            push = torch.zeros_like(v)
            push += rad / rn * (self.cavity_r + self.jitter * 2)        # out through the rim
            push[:, self.axis] = torch.sign(v[:, self.axis]) * (self.cavity_h + self.jitter * 2)
            p[inside] = (c + push).clamp(lo, hi)

        # WRITTEN THROUGH THE VIEW. `Level.get` returns a slice of the state matrix, so an
        # in-place assignment IS the write; there is no setter, and `lvl.set(...)` does not exist.
        lvl.get("pos")[:] = p.to(dev, dt_)
        # THE FIBRE DIRECTIONS LIVE ON THE OPERATOR, not in the state matrix: the schema is fixed
        # at construction and has no `fibre` block. Keeping them here means an anisotropic term
        # can read them without a schema migration, and means they cannot be silently zeroed by
        # something that rewrites state.
        self.fibre = f.to(dev, dt_)
        self._done = True
        n_in = int((~self._outside_cavity(lvl.get("pos"))).sum())
        print(f"[ecm_seed] {n} particles on {self.n_fibres} fibres; cavity r={self.cavity_r} "
              f"h={self.cavity_h} about axis {self.axis}; {n_in} left inside the cavity",
              flush=True)
        return {}


# --------------------------------------------------------------------------- the coupling
@register_operator("cell_to_ecm", family="mechanics", set="particle", kind="lateral",
                   implementation="sphere")
class CellToECMSphere(Lateral):
    """The growing cell ball, as a moving boundary the matrix particles feel.

    IMPLEMENTATION `sphere` IS A STAND-IN, AND IS LABELLED ONE. The real cell ball is a Tyssue
    vertex mesh with its own energy minimisation; this is a prescribed sphere of radius r(t),
    which reproduces the loading the matrix sees without needing the mesh. It exists so the
    matrix, the cavity, the fibres, the stable substep and the rendering can all be settled
    against something whose answer is known -- a sphere of radius r(t) pushes exactly as hard as
    r(t) says -- before any of it is trusted with a mesh whose answer is not known.

    Swapping in the mesh is then `implementation: vertex_mesh` on this same operator, which is
    the point of registering it as an implementation rather than as a second operator: the spec
    changes by one word and everything else is already certified.

    THE FORCE IS A ONE-SIDED PENALTY on penetration depth, applied only to particles the ball has
    actually reached. Not a spring to a rest position: the matrix must be free to be pushed and
    STAY pushed, because permanent displacement is the observable -- an elastic matrix that
    springs back tells you the ball was there, not what it did.
    """
    EMIT = "mpm_acceleration"       # consumed by the substep as a_ext, like mpm_anchor
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["cell_matrix_contact", "moving_boundary", "confinement"]
    PARAM_ROLES = {"k": "contact_stiffness", "r0": "initial_radius", "growth": "growth_rate",
                   "r_max": "final_radius", "damp": "contact_damping"}
    REFERENCE = "Okuda, S. et al. (2018) Sci. Rep. 8:2386 (the vesicle this stands in for)."

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
                v = lvl.get("vel")[hit] if "vel" in getattr(lvl, "state", {}) else None
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
    """Colour the matrix by how hard it is being squeezed, so the stress is the thing you watch.

    THE POINT OF THE WHOLE EXPERIMENT IS A PROPAGATION, and a propagation is invisible in a movie
    of positions. The fibres near the ball move a little, the ones behind them move less, and by
    eye that reads as "the middle wiggled" -- when what is actually happening is a stress front
    travelling outward through the material, which is the mechanics the matrix was added to show.

    The scalar is |J - 1|, J = det F: the LOCAL VOLUME CHANGE. Compression reads positive,
    extension reads positive, unstrained material reads zero. Chosen over a von Mises invariant
    because it needs no material constants -- so the colour means the same thing across a sweep
    in which stiffness is exactly what varies, and two runs at different Young's modulus can be
    put side by side without the palette having quietly changed meaning between them.

    IT IS BANDED INTO INTEGERS, not written as a float, because this renderer's `color_by` is a
    PALETTE INDEX -- `pal[nt % len(pal)]` -- and not a continuous map. So the set declares K types
    carrying identical material and different colours, and this writes the band. The material is
    untouched by construction: every type is the same material, so the index is decoration and
    cannot change the physics it is drawing.
    """
    EMIT = None                     # writes a colour channel in place; no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["diagnostic", "strain_visualisation"]
    PARAM_ROLES = {"scale": "strain_full_scale", "bands": "colour_bands"}
    REFERENCE = "Hu, Y. et al. (2018). ACM Trans. Graph. 37(4):150 (MLS-MPM)."

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

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        F = getattr(lvl, "F", None)
        if F is None:
            return {}
        J = torch.linalg.det(F)
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
        return {}


@register_operator("cell_to_ecm", family="mechanics", set="particle", kind="lateral",
                   implementation="replay")
class CellToECMReplay(Lateral):
    """The REAL tissue's recorded surface, pushing the matrix.

    `implementation: sphere` is a prescribed ball and knows nothing about cells. This one replays
    an actual cellfix_B_new run: 200 epithelial cells growing and dividing to 3,170 under their own
    vertex mechanics, at their own scale, with their own parameters -- none of it rescaled, because
    rescaling was measured to break it (see `tissue.py`).

    THE SURFACE IS AN ANGULAR RADIUS MAP OF THE APICAL VERTICES. Each frame carries R(theta, phi),
    the distance to the furthest apical vertex in that direction (`tissue.apical_map`; it used to be
    the furthest cell CENTROID, which sampled the same surface with half the points). A particle's
    own direction gives its bin, and one comparison decides whether the tissue has reached it --
    O(1) per particle rather than a point-in-mesh test against four thousand faces, which is what
    makes 110,000 particles affordable per frame. It assumes the vesicle is STAR-SHAPED.
    cellfix_B_new is; P11 is the premise that reports when a tissue stops being, and a run that
    trips it has left this operator's domain of validity.

    THE MAP IS CENTROID-REFERENCED, SO `centre` PINS THE TISSUE AT THE BOX CENTRE and the vesicle's
    own translational drift is dropped. That is deliberate: the drift is a few percent of the radius
    and would otherwise slide the tissue off the cavity it was seeded into, turning a symmetric
    loading experiment into an accidental one-sided one. What the matrix sees is the tissue's SHAPE
    and GROWTH, not its wandering.

    THE COUPLING IS ONE-WAY HERE, and the reason is scale, not modelling. The reaction force is
    computed and returned by the `sphere` implementation for a live tissue; a replay has no live
    tissue to return it to -- pass 1 finished before pass 2 began. So this shows how a growing
    epithelium LOADS a matrix, and does not show the matrix shaping the epithelium back.
    """
    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["k", "surface"]
    MECHANISM_TAGS = ["cell_matrix_contact", "moving_boundary", "recorded_tissue"]
    PARAM_ROLES = {"k": "contact_stiffness", "scale": "surface_rescale"}
    REFERENCE = "Okuda, S. et al. (2018) Sci. Rep. 8:2386."

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
        BALL_RADIUS.append(float(M.median()))
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc * lvl.occ[:, None].float()}

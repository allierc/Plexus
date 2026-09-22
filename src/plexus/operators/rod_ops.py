"""A cilium as a DISCRETE ELASTIC ROD, driven by a moment at its anchored base.

WHY THIS EXISTS BESIDE `cilia_ops.py`, AND WHY IT IS NOT A REFINEMENT OF IT. Every cilium in this
repository so far has been made of MLS-MPM material points, and that choice has a hard floor: MPM
carries one velocity per grid node, so a body thinner than a grid cell cannot move relative to the
fluid it shares nodes with -- it is simply advected. A real Platynereis cilium is 0.25 um thick
against a grid cell of 4.08 um, so representing one faithfully needs a 3,000-cube grid on this
domain. The response up to now was to widen the cilium until the grid could hold it, which
produced a 6 um "paddle" and then a 25 x 10 um "blade": defensible as a stand-in for a whole
ciliary tuft, and not a cilium.

A ROD HAS NO SUCH FLOOR. Its nodes are points on a curve, its thickness is a DRAG COEFFICIENT
rather than a volume, and nothing about it has to be resolved by a background grid. That is how
cilia and flagella are actually modelled -- a slender elastic filament coupled to the fluid
through resistive-force theory or an immersed boundary -- and it is the only way this model gets a
thin cilium at all.

WHAT IS IN HERE, and each operator answers one question:

    rod_seed          where the nodes start, and what "straight" means for this rod
    rod_stretch       the rod does not stretch                      (segment springs)
    rod_bend          the rod resists being bent                    (curvature springs)
    rod_base_moment   the rod is DRIVEN, by a moment at its base    <- the one that was missing
    rod_pin           the base is held, by a wall or by a cell
    rod_drag          the fluid resists the rod, and more across it than along it

EVERY INTERNAL FORCE SUMS TO ZERO BY CONSTRUCTION. Stretch is an equal-and-opposite pair, bend is
a triple whose weights are (-1, +2, -1), and the base moment is a pair. So the rod cannot push
itself: any net motion of the whole comes from `rod_drag` (the fluid) or `rod_pin` (the wall),
which are the two places momentum is allowed to leave, and both are named as external.

THE UNITS ARE PER UNIT NODE MASS, following `bm_bond`: every stiffness here is an acceleration
per unit of whatever it multiplies, so `k_stretch` and `k_bend` are in inverse seconds squared and
no operator needs a mass buffer. That also means these constants are not Young's moduli and are
not written as if they were.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral, Seed
from plexus.models.registry import register_operator


def _nodes(H, at):
    """The rod set, its node count per rod, and its positions reshaped to [n_rod, per, 3]."""
    p = H.level(at)
    X = p.get("pos")
    n_rod = int(getattr(p, "n_rod", 1))
    per = p.n // max(n_rod, 1)
    return p, X, n_rod, per


@register_operator("rod_seed", family="mechanics", set="particle", kind="seed")
class RodSeed(Seed):
    """Lay each rod out straight from its base, and remember what straight meant.

    particle -> particle: writes `pos`, and stores the rest segment length, the base's anchor
    point and the axis the base moment turns about.

        x_k = base + (k / (N-1)) * L * d,        k = 0 .. N-1

    with `d` the direction the rod points and `L` its length. Node 0 IS the base: `rod_pin` holds
    it and `rod_base_moment` drives the joint between it and node 1, so the ordering is not a
    convention that could be flipped without consequence.

    `n_rod` LETS ONE SET HOLD MANY RODS, laid out contiguously -- node k of rod r is row
    r*N + k -- which is what keeps every operator here a batched tensor op rather than a loop over
    filaments. With the default of one rod it is the single-cilium test rig, which is what this
    whole file was written for first: one cilium on a fixed base, driven, and nothing else in the
    scene to confuse the question of whether it oscillates.

    `beat_axis` IS THE NORMAL OF THE STROKE PLANE, not the direction of the stroke. The rod turns
    ABOUT it, so the tip sweeps in the plane perpendicular to it; naming it after the sweep would
    describe a rod that beats edge-on.
    """

    EMIT = None
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = []
    WRITES = ["pos"]
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    MAY_MUTATE_INTEGRATED_STATE = True
    REQUIRES_PARAMS = ["length"]
    MECHANISM_TAGS = ["initial_condition", "cilia", "elastic_rod", "anatomy"]
    PARAM_ROLES = {"length": "rod_length_in_world_units",
                   "base": "where_node_zero_sits",
                   "direction": "the_way_the_rod_points_at_rest",
                   "beat_axis": "normal_of_the_stroke_plane",
                   "n_rod": "how_many_rods_share_this_set"}
    REFERENCE = ("Discrete elastic rods: Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63. "
                 "Cilium as a driven elastic filament: Machin, K.E. (1958). J. Exp. Biol. 35:796.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.length = float(params["length"])
        self.n_rod = int(params.get("n_rod", 1))
        self.base = [float(v) for v in (params.get("base") or [0.5, 0.5, 0.2])]
        self.dir = [float(v) for v in (params.get("direction") or [0.0, 0.0, 1.0])]
        self.axis = [float(v) for v in (params.get("beat_axis") or [0.0, 1.0, 0.0])]

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev, dt = p.state.device, p.state.dtype
        per = p.n // max(self.n_rod, 1)
        if per < 3:
            raise ValueError(
                f"rod_seed: {per} node(s) per rod. A rod needs at least three -- two make a "
                f"single segment with no interior node, so `rod_bend` has nothing to act on and "
                f"the rod is a rigid stick however it is driven.")
        d = torch.tensor(self.dir, device=dev, dtype=dt)
        d = d / d.norm().clamp_min(1e-12)
        n = torch.tensor(self.axis, device=dev, dtype=dt)
        n = n - (n @ d) * d                       # the stroke normal must be perpendicular to the rod
        if float(n.norm()) < 1e-9:
            raise ValueError(
                f"rod_seed: `beat_axis` {self.axis} is parallel to `direction` {self.dir}. The rod "
                f"turns ABOUT the beat axis, so an axis along the rod names a twist and not a "
                f"stroke, and the tip would not sweep at all.")
        n = n / n.norm()
        b = torch.tensor(self.base, device=dev, dtype=dt)

        k = (torch.arange(p.n, device=dev, dtype=dt) % per) / float(per - 1)
        pos = b[None, :] + (k[:, None] * self.length) * d[None, :]

        st = p.state.clone()
        a, bb = p.state_schema["pos"]
        st[:, a:bb] = pos
        p.state = st
        p.register_buffer("rod_rest", torch.full((p.n,), self.length / (per - 1),
                                                 device=dev, dtype=dt))
        p.register_buffer("rod_axis", n[None, :].expand(p.n, 3).contiguous().clone())
        p.register_buffer("rod_anchor", b[None, :].expand(self.n_rod, 3).contiguous().clone())
        p.n_rod = self.n_rod
        print(f"[rod_seed] {self.n_rod} rod(s), {per} nodes each, length {self.length:g}, "
              f"segment {self.length / (per - 1):.5f}, beating about "
              f"[{float(n[0]):.2f}, {float(n[1]):.2f}, {float(n[2]):.2f}]", flush=True)
        return {}


@register_operator("rod_stretch", family="mechanics", set="particle", kind="lateral")
class RodStretch(Lateral):
    """The rod does not stretch: a spring along every segment, equal and opposite.

    particle -> particle: reads pos, emits an acceleration.

        a_i  = +k (|e| - rest) e / |e| ,   a_{i+1} = -that ,   e = x_{i+1} - x_i

    `k` is in inverse seconds squared, so k times a length is an acceleration -- the same
    convention `bm_bond` uses, and for the same reason: writing the force in terms of STRAIN
    instead divides by the rest length, which for a segment of a few hundredths of a world unit
    multiplies the declared stiffness by a factor of a hundred or more, and the rod then rings
    itself apart at a stiffness that reads as mild.

    IT MUST BE STIFF, and stiffness here costs timestep rather than realism. A cilium is
    essentially inextensible: what bends is not what stretches. The rod's fastest mode is this
    spring, so the explicit timestep must satisfy dt < 2/sqrt(k) -- and `rod_probe.py` reports the
    margin rather than leaving it to be discovered as a blow-up.

    `zeta` IS A DASHPOT ALONG THE SEGMENT, AND THE ROD NEEDS ONE. A spring this stiff with no
    damping of its own rings forever, and the only other sink in the rig is `rod_drag` -- which is
    the FLUID, and which has to be turned down to put the rod at a biological sperm number. The
    first Step-1 rig had exactly that combination and its base angle grew from 20 to 50 degrees
    over three seconds with visible high-frequency jitter riding on it: an instability that reads,
    on the amplitude alone, exactly like a rod beating harder.

        a_damp = -2 zeta sqrt(k) [ (v_{i+1} - v_i) . e_hat ] e_hat        on i, and +that on i+1

    It acts ONLY along the segment, so it damps stretching and leaves bending and rotation
    untouched -- a dashpot on the full relative velocity would quietly damp the beat itself. It is
    equal and opposite, so it is internal and takes no momentum out of the rod. `zeta` is the
    damping ratio of the segment mode: 1 is critical, and the rig uses it.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["elastic_rod", "spring", "inextensible", "momentum_conserving",
                      "internal_damping"]
    PARAM_ROLES = {"k": "stretch_stiffness_per_unit_mass_in_inverse_seconds_squared",
                   "zeta": "damping_ratio_of_the_segment_mode_1_is_critical"}
    REFERENCE = "Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63 (discrete elastic rods)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.k = float(params["k"])
        self.zeta = float(params.get("zeta", 0.0))

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        e = Q[:, 1:, :] - Q[:, :-1, :]                              # [R, per-1, 3]
        L = e.norm(dim=2, keepdim=True).clamp_min(1e-12)
        rest = p.rod_rest.view(n_rod, per)[:, :-1, None]
        u = e / L
        f = self.k * (L - rest) * u                                 # pull i toward i+1
        if self.zeta:
            V = p.get("vel").view(n_rod, per, 3).to(X.dtype)
            dv = ((V[:, 1:, :] - V[:, :-1, :]) * u).sum(2, keepdim=True)   # stretch RATE only
            f = f + (2.0 * self.zeta * math.sqrt(self.k)) * dv * u
        a = torch.zeros_like(Q)
        a[:, :-1, :] += f
        a[:, 1:, :] -= f
        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}


@register_operator("rod_bend", family="mechanics", set="particle", kind="lateral")
class RodBend(Lateral):
    """The rod resists being bent, and a straight rod is its rest state.

    particle -> particle: reads pos, emits an acceleration.

        c_i = x_{i-1} - 2 x_i + x_{i+1}                 the discrete curvature at interior node i
        a_{i-1} = -k c_i ,  a_i = +2 k c_i ,  a_{i+1} = -k c_i

    which is the gradient of E = (k/2) * sum |c_i|^2, and whose three weights sum to zero -- so
    bending is an internal force and cannot move the rod's centre of mass however hard it curls.

    `k_bend` IS WHERE "STIFF CILIUM" AND "ELASTIC CILIUM" DIFFER, and it is the whole of that
    difference. Large `k` propagates the base's motion to the tip almost rigidly, so the rod
    sweeps as a straight stick and its stroke is exactly its command's. Small `k` lets the fluid's
    drag bend the rod as it sweeps, so the tip LAGS the base -- the rod takes on the curved shape a
    real cilium has, and the lag is what breaks the time symmetry of the stroke without anything
    asymmetric being commanded. That is Machin's 1958 result and it is the reason a flexible
    filament can swim while a rigid one waving back and forth cannot.

    `zeta` DAMPS THE BENDING MODES, and the rig needs it for the same reason `rod_stretch` needs
    its own dashpot: the only other sink is `rod_drag`, which is the FLUID and which has to be
    turned down to put the rod at a biological sperm number. With drag that low the bending modes
    ring for the whole run -- measured, the base angle grew from 20 to 50 degrees over three
    seconds with high-frequency jitter on it, which on the amplitude alone reads exactly like a
    rod beating harder rather than a rod ringing.

        a_damp = damp * (-1, +2, -1) weighted on the rate of change of c_i,   damp = 2 zeta sqrt(k)

    The same three weights, so it is internal and takes no momentum out of the rod.

    THE CURVATURE IS UNWEIGHTED BY SEGMENT LENGTH, which is exact here and an approximation in
    general: `rod_seed` lays every segment the same length and `rod_stretch` keeps them that way,
    so the second difference IS the curvature times a constant. On a rod with unequal segments
    this term would have to be divided by the local length, and it would be wrong to pretend
    otherwise by leaving the division out silently.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["elastic_rod", "bending_stiffness", "momentum_conserving",
                      "internal_damping"]
    PARAM_ROLES = {"k": "bending_stiffness_per_unit_mass_in_inverse_seconds_squared",
                   "zeta": "damping_ratio_of_the_bending_modes"}
    REFERENCE = ("Machin, K.E. (1958). J. Exp. Biol. 35:796 (a driven flexible filament); "
                 "Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.k = float(params["k"])
        self.zeta = float(params.get("zeta", 0.0))

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        c = Q[:, :-2, :] - 2.0 * Q[:, 1:-1, :] + Q[:, 2:, :]        # [R, per-2, 3]
        g = self.k * c
        if self.zeta:
            V = p.get("vel").view(n_rod, per, 3).to(X.dtype)
            cd = V[:, :-2, :] - 2.0 * V[:, 1:-1, :] + V[:, 2:, :]   # the curvature's RATE
            g = g + (2.0 * self.zeta * math.sqrt(self.k)) * cd
        a = torch.zeros_like(Q)
        a[:, :-2, :] -= g
        a[:, 1:-1, :] += 2.0 * g
        a[:, 2:, :] -= g
        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}


@register_operator("rod_base_moment", family="mechanics", set="particle", kind="lateral")
class RodBaseMoment(Lateral):
    """DRIVE THE ROD BY A MOMENT AT ITS ANCHORED BASE. This is the operator that was missing.

    particle -> particle: reads pos, emits an acceleration on the base node and the first node
    above it, equal and opposite.

    A cilium is driven at its base: the basal body applies a moment to the axoneme, and the
    reaction is felt by the cell the basal body is embedded in. Everything in `cilia_ops.py`
    before this drove a shaft with a PURE COUPLE spread over the whole shaft, which is what you
    need when the shaft is free-floating MPM material -- but it is not what a basal body does, and
    it cannot be pointed at an anchor. Here the moment acts across ONE JOINT, the base:

        e  = x_1 - x_0                          the first segment
        a_1 = (M / |e|^2) (n x e)               perpendicular to the segment, in the stroke plane
        a_0 = -a_1                              the reaction, on the anchored node

    and the torque this delivers about x_0 is exactly M n:

        e x a_1 = (M/|e|^2) [ n (e.e) - e (e.n) ] = M n     whenever n is perpendicular to e

    THE REACTION LANDS ON NODE 0, WHICH IS THE POINT. If `rod_pin` is holding node 0 to a wall,
    the wall takes it -- an external constraint, honestly external. If `rod_pin` is holding node 0
    to a CELL, the cell takes it, and the cilium's drive pushes back on the body exactly as a real
    one does. Switching between those two is the difference between "watch the cilium beat" and
    "watch the cilium move the cell", and it is one parameter of a different operator rather than
    a different model of the drive.

    THE COMMAND IS AN OSCILLATION, not a constant:

        M(t) = moment * g(phi + omega t)
        g    = sin                                  waveform: sine   -- symmetric in time
             = an asymmetric sawtooth, `duty` up    waveform: stroke -- fast one way, slow back

    `duty` matters only where the fluid is viscous enough for it to: at Reynolds numbers well
    below one a time-symmetric stroke moves no net fluid however hard it is driven (Purcell's
    scallop theorem), so a real cilium beats asymmetrically. Measured on the MPM larva at
    Re = 14,721 the swim speed RISES as the stroke is made symmetric, which is the theorem not
    applying rather than the theorem being wrong -- worth knowing before reading anything into a
    `duty` sweep run in the wrong regime.

    `n` IS TAKEN FROM THE ROD'S STORED BEAT AXIS and re-orthogonalised against the live segment
    every step, because `e` rotates as the rod beats and a moment computed against a stale
    perpendicular would deliver a torque that drifts off-plane and slowly twists the rod.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    REQUIRES_PARAMS = ["moment"]
    MECHANISM_TAGS = ["cilia", "ciliary_beat", "active_moment", "momentum_conserving",
                      "excitation_contraction_coupling"]
    PARAM_ROLES = {"moment": "peak_specific_moment_world_length_squared_per_second_squared",
                   "omega": "beat_rate_in_radians_per_second",
                   "waveform": "sine_or_stroke",
                   "duty": "fraction_of_the_cycle_in_the_power_stroke",
                   "phase": "radians_of_offset"}
    REFERENCE = ("Machin, K.E. (1958). J. Exp. Biol. 35:796; Purcell, E.M. (1977). "
                 "Am. J. Phys. 45:3-11 (the scallop theorem).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.moment = float(params["moment"])
        self.omega = float(params.get("omega", 1.0))
        self.waveform = str(params.get("waveform", "sine")).lower()
        if self.waveform not in ("sine", "stroke"):
            raise ValueError(f"rod_base_moment: `waveform` must be sine or stroke, "
                             f"got {self.waveform!r}")
        self.duty = float(params.get("duty", 0.5))
        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"rod_base_moment: `duty` is the fraction of the cycle spent in the "
                             f"power stroke and must lie strictly between 0 and 1, "
                             f"got {self.duty}")
        self.phase = float(params.get("phase", 0.0))

    def wave(self, t: float) -> float:
        """The command at time `t`, in [-1, 1]. Separated out so a probe can plot it."""
        u = ((self.omega * t + self.phase) / (2.0 * math.pi)) % 1.0
        if self.waveform == "sine":
            return math.sin(2.0 * math.pi * u)
        if u < self.duty:
            return 2.0 * (u / self.duty) - 1.0
        return 2.0 * (1.0 - (u - self.duty) / (1.0 - self.duty)) - 1.0

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        dt = X.dtype
        e = Q[:, 1, :] - Q[:, 0, :]                                  # [R, 3] the first segment
        n = p.rod_axis.view(n_rod, per, 3)[:, 0, :].to(dt)
        # RE-ORTHOGONALISED AGAINST THE LIVE SEGMENT. `e` turns as the rod beats; a moment taken
        # about a stale perpendicular delivers torque with a component ALONG the rod, which is a
        # twist, and it accumulates.
        n = n - (n * e).sum(1, keepdim=True) / (e * e).sum(1, keepdim=True).clamp_min(1e-24) * e
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

        t = float(getattr(H, "frame", 0)) * float(getattr(H, "dt", 1.0))
        M = self.moment * self.wave(t)
        a1 = (M / (e * e).sum(1, keepdim=True).clamp_min(1e-24)) * torch.cross(n, e, dim=1)

        a = torch.zeros_like(Q)
        a[:, 1, :] = a1
        a[:, 0, :] = -a1                                             # the reaction, on the anchor
        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(dt)
        return {self.at: a}


@register_operator("rod_pin", family="mechanics", set="particle", kind="lateral")
class RodPin(Lateral):
    """Hold the rod's base -- to a fixed point, or to a body that is free to move.

    particle -> particle: reads pos and vel, emits an acceleration on node 0 (and, when the base
    is a set rather than a wall, the equal and opposite on that set).

        a_0 = k (target - x_0) - c (v_0 - v_target)          k = omega_n^2, c = 2 zeta omega_n

    `omega_n` IS THE ATTACHMENT'S CORNER FREQUENCY in radians per second, which is a more useful
    thing to state than a stiffness because it is directly comparable to the beat: an anchor at
    40 rad/s under a beat at 1.25 is a hinge, while one at 2 rad/s is a spring the stroke swings
    on and the "fixed" base is not fixed at all.

    `anchor: fixed` IS AN EXTERNAL CONSTRAINT AND SAYS SO. A wall absorbs whatever momentum the rod
    pushes into it, so a scene with a pinned rod does NOT conserve momentum and should not be
    measured as though it does. That is the right model for the first test rig -- one cilium on a
    bench, beating, with nothing else in the scene -- and the wrong one for a swimmer, which is
    why it is a parameter and not an assumption.

    `clamp` HOLDS THE BASE'S DIRECTION, NOT ONLY ITS POSITION, AND THE ROD NEEDS IT. A basal body
    does both: it fixes where the cilium is attached AND which way the axoneme leaves the cell, and
    the beat is a deviation from that rest direction. Without the second half there is nothing at
    all resisting a RIGID ROTATION of the whole rod about the pin -- `rod_bend` penalises curvature
    and a rigid rotation has none -- so the rod is free to windmill.

    It does, and the mechanism is real rather than numerical: `rod_drag` is anisotropic, so over a
    symmetric cycle the rod ratchets round in the direction it grips least. Measured on the
    unclamped rig, a mean-zero sine drove a clean 1 Hz beat whose MEAN drifted to +35 degrees over
    ten seconds, the rod oscillating about a steadily-curling rest position. That ratchet is the
    same asymmetry that makes a filament swim; what it is not is a cilium beating about a fixed
    rest direction.

        theta      = signed angle from the rest direction to the live first segment, about n
        a_1        = [ -(clamp^2) theta - 2 zeta clamp theta_dot ] (n x e),    a_0 = -a_1

    WRITTEN SO THAT `clamp` IS THE CORNER FREQUENCY AND NOT MERELY CALLED ONE. Node 1's angular
    equation about the base, per unit mass, is |e|^2 theta_ddot = M, so a moment written
    M = -(clamp^2) theta gives a natural frequency of clamp/|e| -- and |e| is ONE SEGMENT, a few
    hundredths of a world unit, so the mode ran at 57 times the number in the spec and its damping
    term at 6,600 per second against a stable limit near 2,000. Every clamp above zero returned a
    rod stretched to 400-700% of its own length. Folding the |e|^2 in makes the equation
    theta_ddot = -(clamp^2) theta - 2 zeta clamp theta_dot exactly, which is what the parameter
    name promises.

    the same force pair `rod_base_moment` uses, so it is equal and opposite and the wall (or the
    cell) takes the reaction. `clamp` is a corner frequency in radians per second: it must sit
    BELOW the drive's own authority or the cilium cannot beat at all, and above the ratchet's, or
    it windmills anyway. Zero turns it off, which is the free-pivot rod.

    `anchor: <set>` PUTS THE REACTION ON THAT SET instead, spread over the `n_body` nearest points,
    and then momentum IS conserved across the pair: the same force with opposite signs, so
    sum(m a) is identically zero. Going from the first to the second is exactly the step from
    "does it beat" to "does the beat move the cell".
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["anchor", "boundary_condition", "attachment"]
    PARAM_ROLES = {"omega_n": "attachment_corner_frequency_rad_per_s",
                   "zeta": "damping_ratio_of_the_attachment",
                   "anchor": "fixed_or_the_name_of_the_set_that_takes_the_reaction",
                   "clamp": "corner_frequency_holding_the_base_DIRECTION_rad_per_s",
                   "clamp_zeta": "damping_ratio_of_that_direction_clamp",
                   "n_body": "points_of_that_set_the_reaction_is_spread_over"}
    REFERENCE = "The basal body is embedded in its cell; Machin, K.E. (1958). J. Exp. Biol. 35:796."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.omega_n = float(params.get("omega_n", 200.0))
        self.zeta = float(params.get("zeta", 1.0))
        # NOT `to`, WHICH THE SPEC SCHEMA RESERVES. `resolve_op_line` reads `to:` as the name
        # of a FIELD the operator writes into (it is how `mpm_scatter to: mpm_grid` is spelt), so
        # `to: fixed` was rejected with "operator 'rod_pin' references unknown field 'fixed'".
        self.anchor = str(params.get("anchor", "fixed"))
        self.n_body = int(params.get("n_body", 16))
        self.clamp = float(params.get("clamp", 0.0))
        self.clamp_zeta = float(params.get("clamp_zeta", 1.0))
        self._idx = None

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        V = p.get("vel").view(n_rod, per, 3)
        dt = X.dtype
        x0, v0 = Q[:, 0, :], V[:, 0, :]

        out = {}
        if self.anchor == "fixed":
            target = p.rod_anchor.to(dt)
            v_t = torch.zeros_like(v0)
        else:
            b = H.level(self.anchor)
            Xb, Vb = b.get("pos").to(dt), b.get("vel").to(dt)
            if self._idx is None:
                self._idx = torch.cdist(p.rod_anchor.to(dt), Xb).topk(
                    min(self.n_body, b.n), largest=False).indices
            target = Xb[self._idx].mean(1)
            v_t = Vb[self._idx].mean(1)

        k, c = self.omega_n ** 2, 2.0 * self.zeta * self.omega_n
        a0 = k * (target - x0) - c * (v0 - v_t)

        a = torch.zeros_like(Q)
        a[:, 0, :] = a0

        if self.clamp:
            # THE DIRECTION CLAMP. `rod_seed` stored the rest direction as the rod's own layout, so
            # it is recovered from the anchor-to-node-1 vector at rest rather than kept twice.
            e = Q[:, 1, :] - Q[:, 0, :]
            n = p.rod_axis.view(n_rod, per, 3)[:, 0, :].to(dt)
            d0 = getattr(p, "rod_rest_dir", None)
            if d0 is None:
                d0 = (e / e.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach().clone()
                p.register_buffer("rod_rest_dir", d0)
            u = torch.cross(n, d0, dim=1)                    # in-plane, perpendicular to rest
            th = torch.atan2((e * u).sum(1), (e * d0).sum(1))
            # the rate, from the same two nodes' velocities projected on the same in-plane axis
            ve = V[:, 1, :] - V[:, 0, :]
            th_d = (ve * u).sum(1) / e.norm(dim=1).clamp_min(1e-12)
            # THE |e|^2 IS ALREADY FOLDED IN: a = (theta_ddot) * (n x e), and |n x e| = |e|, so
            # theta_ddot comes out as the angular acceleration the two constants name directly.
            th_dd = -(self.clamp ** 2) * th - (2.0 * self.clamp_zeta * self.clamp) * th_d
            a1 = th_dd[:, None] * torch.cross(n, e, dim=1)
            a[:, 1, :] += a1
            a[:, 0, :] -= a1                                  # equal and opposite, onto the anchor

        if self.anchor != "fixed":
            # THE REACTION, SPREAD OVER THE SAME POINTS THE TARGET WAS READ FROM. Accelerations do
            # not add across different masses, forces do -- but every node here is per-unit-mass by
            # this file's convention, so the honest statement is that the BODY takes the same total
            # force the base does, divided among its points.
            n_b = self._idx.shape[1]
            b_acc = torch.zeros_like(H.level(self.anchor).get("pos"))
            b_acc.index_add_(0, self._idx.reshape(-1),
                             (-a0 / n_b)[:, None, :].expand(-1, n_b, -1).reshape(-1, 3).to(b_acc.dtype))
            out[self.anchor] = b_acc
        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(dt)
        out[self.at] = a
        return out


@register_operator("rod_drag", family="mechanics", set="particle", kind="lateral")
class RodDrag(Lateral):
    """The fluid resists the rod, and resists it MORE ACROSS than ALONG. That asymmetry is thrust.

    particle -> particle: reads vel and pos, emits an acceleration.

        t_hat   = the local tangent of the rod at this node
        v_par   = (v . t_hat) t_hat                   the part of the motion along the rod
        v_perp  = v - v_par                           the part across it
        a       = -( zeta_par v_par + zeta_perp v_perp )

    RESISTIVE-FORCE THEORY, AND THE RATIO IS THE WHOLE MECHANISM. A slender body in Stokes flow
    feels roughly twice the drag moving sideways as lengthwise -- zeta_perp / zeta_par is about 2
    for a long thin filament. That single fact is how a filament waving back and forth produces
    net thrust: on the power stroke the rod is broadside to its own motion and grips the fluid,
    on the recovery it is edge-on and slips through it. Set the ratio to 1 and the rod is a
    perfectly symmetric paddle that cannot swim no matter what waveform it is given, which is
    worth running once as the control.

    THIS IS A STAND-IN FOR THE FLUID, NOT THE FLUID. It is a local, instantaneous drag law: it
    knows nothing about the flow the rod itself creates, so it cannot show a vortex, cannot let
    one cilium feel another, and cannot push water past a cell. It is exactly right for the first
    question -- does a driven rod beat, and with what shape -- and it must be REPLACED by a real
    coupling to the water, not supplemented, once the question becomes where the fluid goes.

    IT TAKES MOMENTUM OUT OF THE SCENE, on purpose, and unlike every other operator in this file
    it is not an equal-and-opposite pair: the momentum goes into a fluid that is not represented.
    A scene with `rod_drag` in it therefore does not conserve momentum and must not be measured as
    if it did.

    Reference: Gray, J. & Hancock, G.J. (1955). J. Exp. Biol. 32:802 (resistive force theory).
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = ["zeta_par"]
    MECHANISM_TAGS = ["drag", "resistive_force_theory", "dissipation", "external_momentum_sink"]
    PARAM_ROLES = {"zeta_par": "drag_along_the_rod_in_inverse_seconds",
                   "ratio": "zeta_perp_over_zeta_par_about_2_for_a_slender_filament"}
    REFERENCE = "Gray, J. & Hancock, G.J. (1955). J. Exp. Biol. 32:802."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.zeta_par = float(params["zeta_par"])
        self.ratio = float(params.get("ratio", 2.0))

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        V = p.get("vel").view(n_rod, per, 3)
        # THE TANGENT AT A NODE IS THE CENTRED DIFFERENCE OF ITS NEIGHBOURS, and one-sided at the
        # ends. A per-SEGMENT tangent would leave the nodes -- which are what carry the velocity --
        # without one of their own.
        t = torch.zeros_like(Q)
        t[:, 1:-1, :] = Q[:, 2:, :] - Q[:, :-2, :]
        t[:, 0, :] = Q[:, 1, :] - Q[:, 0, :]
        t[:, -1, :] = Q[:, -1, :] - Q[:, -2, :]
        t = t / t.norm(dim=2, keepdim=True).clamp_min(1e-12)

        v_par = (V * t).sum(2, keepdim=True) * t
        a = -(self.zeta_par * v_par + (self.zeta_par * self.ratio) * (V - v_par))
        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}

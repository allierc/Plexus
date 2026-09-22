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

WHAT IS IN HERE -- TWO OPERATORS AND A SEED, and it was six before.

    rod_seed      where the nodes start, and what "straight" means for this rod
    rod_elastic   the rod's own material: it does not stretch, and it resists being bent
    rod_base      the base: held in PLACE, held in DIRECTION, and DRIVEN -- one moment, three laws

The first draft had `rod_stretch` and `rod_bend` apart, `rod_base_moment` and `rod_pin` apart, and
a `rod_drag` of its own. That is five words for three mechanisms, and a registry that spells one
mechanism twice makes a reader compare two docstrings to find out which one a spec is getting.
Stretch and bend are one material, always declared together and acting on the same set. The base
clamp and the base drive are the SAME MOMENT ACROSS THE SAME JOINT -- one takes a restoring law,
the other a command -- so they are one operator with two terms, not two operators. And the drag
was never a new mechanism at all: `motion_ops.drag` already existed and only lacked a tangent, so
it gained `along: chain` and this file lost an operator.

WHAT IS NOT MERGED, AND WHY THAT IS THE SAME RULE. The drag stays outside `rod_elastic` even
though both act on every node, because they differ in what they do to the INVARIANTS: every force
in `rod_elastic` is an equal-and-opposite pair or a (-1, +2, -1) triple, so the rod cannot push
itself, while drag is a sink into a fluid that is not represented. Merging them would hide a
momentum leak inside an operator whose name promises there is none. Subtract what is redundant,
not what is load-bearing.

EVERY INTERNAL FORCE SUMS TO ZERO BY CONSTRUCTION. So any net motion of the whole comes from the
drag (the fluid) or from `rod_base` (the wall, or the cell), which are the two places momentum is
allowed to leave, and both are named as external.

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



@register_operator("rod_elastic", family="mechanics", set="particle", kind="lateral")
class RodElastic(Lateral):
    """THE ROD'S OWN MATERIAL: it does not stretch, and it resists being bent.

    particle -> particle: reads pos and vel, emits an acceleration. Every force below is either an
    equal-and-opposite PAIR or a triple whose weights are (-1, +2, -1), so all of it is internal
    and the rod cannot accelerate its own centre of mass however hard it is driven.

        stretch   f_i = +[ k_s (|e| - rest) + c_s d|e|/dt ] e_hat ,  f_{i+1} = -f_i
        bend      c_i = x_{i-1} - 2 x_i + x_{i+1}                    the discrete curvature
                  a_{i-1} = -g_i ,  a_i = +2 g_i ,  a_{i+1} = -g_i ,  g_i = k_b c_i + c_b dc_i/dt

    with c_s = 2 zeta_s sqrt(k_s) and c_b = 2 zeta_b sqrt(k_b).

    ONE OPERATOR AND NOT TWO, because stretching and bending are one material. They are always
    declared together, act on the same set, are both internal, and a rod given one without the
    other is not a simpler rod -- it is a chain of beads or a rigid stick.

    THE UNITS ARE PER UNIT NODE MASS, following `bm_bond`: k times a length is an acceleration, so
    both stiffnesses are in inverse seconds squared and no mass buffer is needed. They are NOT
    Young's moduli and are not written as if they were.

    `k_stretch` MUST BE STIFF and stiffness costs timestep, not realism: a cilium is essentially
    inextensible, and what bends is not what stretches. `k_bend` IS WHERE "STIFF CILIUM" AND
    "ELASTIC CILIUM" DIFFER, and it is the whole of that difference -- large, and the base's motion
    reaches the tip almost rigidly so the stroke is the command's; small, and drag curls the rod as
    it sweeps so the tip LAGS, which is the curved shape a real cilium has and the time asymmetry
    a symmetric command cannot otherwise supply (Machin 1958).

    BOTH NEED THEIR OWN DASHPOT, and that is not a detail. The only other sink in a rod rig is the
    drag -- which is the FLUID, and which has to be turned DOWN to reach a biological sperm number.
    With drag that low the springs ring for the whole run: measured, a base angle growing from 20
    to 50 degrees over three seconds with high-frequency jitter on it, which on the amplitude alone
    reads exactly like a rod beating harder. Both dashpots have explicit-integration ceilings of
    their own, and both were found by returning NaN: zeta_stretch 1.0 and zeta_bend 0.15.

    THE STABILITY LIMITS, because a rod blows up silently at first. The stretch spring needs
    dt < 2/sqrt(k_stretch); the bending operator's stiffest mode is about 16 k_bend, so it needs
    dt < 0.5/sqrt(k_bend) -- which is the tighter of the two and is why k_bend 5e5 at dt 1e-3
    returns NaN while 5e4 is stable.

    THE CURVATURE IS UNWEIGHTED BY SEGMENT LENGTH, which is exact here and an approximation in
    general: `rod_seed` lays every segment the same length and the stretch term keeps them that
    way, so the second difference IS the curvature times a constant. On a rod with unequal
    segments this would have to be divided by the local length, and leaving the division out
    silently would be pretending otherwise.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = ["k_stretch", "k_bend"]
    MECHANISM_TAGS = ["elastic_rod", "spring", "bending_stiffness", "inextensible",
                      "internal_damping", "momentum_conserving"]
    PARAM_ROLES = {"k_stretch": "stretch_stiffness_per_unit_mass_inverse_seconds_squared",
                   "k_bend": "bending_stiffness_per_unit_mass_inverse_seconds_squared",
                   "zeta_stretch": "damping_ratio_of_the_segment_mode",
                   "zeta_bend": "damping_ratio_of_the_bending_modes"}
    REFERENCE = ("Bergou, M. et al. (2008). ACM Trans. Graph. 27(3):63 (discrete elastic rods); "
                 "Machin, K.E. (1958). J. Exp. Biol. 35:796 (a driven flexible filament).")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.k_s = float(params["k_stretch"])
        self.k_b = float(params["k_bend"])
        self.z_s = float(params.get("zeta_stretch", 0.0))
        self.z_b = float(params.get("zeta_bend", 0.0))

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        V = p.get("vel").view(n_rod, per, 3).to(X.dtype) if (self.z_s or self.z_b) else None
        a = torch.zeros_like(Q)

        # ---- stretch: an equal-and-opposite pair along every segment
        e = Q[:, 1:, :] - Q[:, :-1, :]
        L = e.norm(dim=2, keepdim=True).clamp_min(1e-12)
        u = e / L
        f = self.k_s * (L - p.rod_rest.view(n_rod, per)[:, :-1, None]) * u
        if self.z_s:
            # THE STRETCH RATE ONLY, projected on the segment: a dashpot on the full relative
            # velocity would quietly damp the beat itself rather than the spring.
            dv = ((V[:, 1:, :] - V[:, :-1, :]) * u).sum(2, keepdim=True)
            f = f + (2.0 * self.z_s * math.sqrt(self.k_s)) * dv * u
        a[:, :-1, :] += f
        a[:, 1:, :] -= f

        # ---- bend: a (-1, +2, -1) triple at every interior node
        g = self.k_b * (Q[:, :-2, :] - 2.0 * Q[:, 1:-1, :] + Q[:, 2:, :])
        if self.z_b:
            g = g + (2.0 * self.z_b * math.sqrt(self.k_b)) * (
                V[:, :-2, :] - 2.0 * V[:, 1:-1, :] + V[:, 2:, :])
        a[:, :-2, :] -= g
        a[:, 1:-1, :] += 2.0 * g
        a[:, 2:, :] -= g

        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(a.dtype)
        return {self.at: a}


@register_operator("rod_base", family="mechanics", set="particle", kind="lateral")
class RodBase(Lateral):
    """THE BASE: held in PLACE, held in DIRECTION, and DRIVEN. One joint, one moment, three laws.

    particle -> particle: reads pos and vel, emits an acceleration on node 0 and node 1, and --
    when the base is held to a set rather than to a wall -- the equal and opposite on that set.

    A basal body does three things at once, and this operator is those three things:

        POSITION   a_0 += omega_n^2 (target - x_0) - 2 zeta omega_n (v_0 - v_target)
        DIRECTION  theta_dd = -(clamp^2) theta - 2 zeta_c clamp theta_dot
        DRIVE      M(t)    = moment * g(phase + omega t)
        both moments  a_1 += (theta_dd + M/|e|^2) (n x e) ,   a_0 -= that

    ONE OPERATOR AND NOT TWO, because the clamp and the drive are THE SAME MOMENT ACROSS THE SAME
    JOINT. One takes a restoring law and the other a command; nothing else about them differs, and
    splitting them meant two operators recomputing the same segment, the same perpendicular and
    the same force pair. The position pin joins them because it is the other half of one anatomical
    fact: a basal body fixes where the cilium is attached AND which way the axoneme leaves.

    THE DRIVE WAS THE OPERATOR THAT WAS MISSING. Everything before it drove a shaft with a pure
    couple spread over the whole shaft, which is what free-floating material needs and is not what
    a basal body does -- and it could not be pointed at an anchor. The torque delivered here about
    x_0 is exactly M n:

        e x a_1 = (M/|e|^2) [ n (e.e) - e (e.n) ] = M n     whenever n is perpendicular to e

    and the reaction lands on node 0. If the base is held to a WALL the wall takes it; if to a
    CELL, the cell does, and the cilium's drive pushes back on the body exactly as a real one does.
    That switch is `anchor`, one parameter -- the difference between "watch the cilium beat" and
    "watch the cilium move the cell" is not a different model of the drive.

    `clamp` IS NOT OPTIONAL IN PRACTICE. `rod_elastic` penalises CURVATURE, and a rigid rotation
    about the pin has none, so without a direction clamp nothing at all resists the rod windmilling
    -- and it does, because the drag is anisotropic and over a symmetric cycle the rod ratchets
    round in the direction it grips least. Measured: a clean 29-degree beat whose CENTRE drifted to
    +32 degrees over ten seconds, which on amplitude alone reads as a beat.

    `clamp` IS A CORNER FREQUENCY AND NOT MERELY CALLED ONE. Node 1's angular equation about the
    base, per unit mass, is |e|^2 theta_dd = M, so writing M = -(clamp^2) theta gives a natural
    frequency of clamp/|e| -- and |e| is ONE SEGMENT, so the mode ran 57 times the number in the
    spec and every clamp above zero stretched the rod to 400-700% of its length. Folding the |e|^2
    in makes theta_dd = -(clamp^2) theta - 2 zeta_c clamp theta_dot exactly. It must sit above the
    ratchet's authority and below the drive's: 40 rad/s against a 6.28 rad/s beat centres the
    stroke with the largest amplitude of any value tried; 160 overpowers the drive.

    `waveform: stroke` matters only where the fluid is viscous enough for it to. Below Reynolds
    one a time-symmetric stroke moves no net fluid however hard it is driven (Purcell's scallop
    theorem), which is why a real cilium beats asymmetrically -- but measured on the MPM larva at
    Re = 14,721 the swim speed RISES as the stroke is made symmetric. That is the theorem not
    applying, and it is worth knowing before reading anything into a `duty` sweep.

    `n` IS RE-ORTHOGONALISED AGAINST THE LIVE SEGMENT every step, because `e` turns as the rod
    beats and a moment taken about a stale perpendicular has a component ALONG the rod, which is a
    twist, and it accumulates.
    """

    EMIT = "acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel"]
    WRITES = []
    REQUIRES_PARAMS = []
    MECHANISM_TAGS = ["cilia", "ciliary_beat", "active_moment", "anchor", "attachment",
                      "boundary_condition", "momentum_conserving",
                      "excitation_contraction_coupling"]
    PARAM_ROLES = {"moment": "peak_specific_moment_world_length_squared_per_second_squared",
                   "omega": "beat_rate_in_radians_per_second",
                   "waveform": "sine_or_stroke",
                   "duty": "fraction_of_the_cycle_in_the_power_stroke",
                   "phase": "radians_of_offset",
                   "omega_n": "position_pin_corner_frequency_rad_per_s",
                   "zeta": "damping_ratio_of_the_position_pin",
                   "clamp": "corner_frequency_holding_the_base_DIRECTION_rad_per_s",
                   "clamp_zeta": "damping_ratio_of_that_direction_clamp",
                   "anchor": "fixed_or_the_name_of_the_set_that_takes_the_reaction",
                   "n_body": "points_of_that_set_the_reaction_is_spread_over"}
    REFERENCE = ("Machin, K.E. (1958). J. Exp. Biol. 35:796; Purcell, E.M. (1977). Am. J. Phys. "
                 "45:3-11 (the scallop theorem); Veraszto, C. et al. (2017). eLife 6:e26000.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "rod_node")
        self.moment = float(params.get("moment", 0.0))
        self.omega = float(params.get("omega", 1.0))
        self.waveform = str(params.get("waveform", "sine")).lower()
        if self.waveform not in ("sine", "stroke"):
            raise ValueError(f"rod_base: `waveform` must be sine or stroke, got {self.waveform!r}")
        self.duty = float(params.get("duty", 0.5))
        if not 0.0 < self.duty < 1.0:
            raise ValueError(f"rod_base: `duty` is the fraction of the cycle spent in the power "
                             f"stroke and must lie strictly between 0 and 1, got {self.duty}")
        self.phase = float(params.get("phase", 0.0))
        self.omega_n = float(params.get("omega_n", 200.0))
        self.zeta = float(params.get("zeta", 1.0))
        self.clamp = float(params.get("clamp", 0.0))
        self.clamp_zeta = float(params.get("clamp_zeta", 1.0))
        # NOT `to`, WHICH THE SPEC SCHEMA RESERVES. `resolve_op_line` reads `to:` as the name of a
        # FIELD the operator writes into (it is how `mpm_scatter to: mpm_grid` is spelt), so
        # `to: fixed` was rejected with "references unknown field 'fixed'".
        self.anchor = str(params.get("anchor", "fixed"))
        self.n_body = int(params.get("n_body", 16))
        self._idx = None

    def wave(self, t: float) -> float:
        """The drive command at time `t`, in [-1, 1]. Separated so a probe can plot it."""
        u = ((self.omega * t + self.phase) / (2.0 * math.pi)) % 1.0
        if self.waveform == "sine":
            return math.sin(2.0 * math.pi * u)
        if u < self.duty:
            return 2.0 * (u / self.duty) - 1.0
        return 2.0 * (1.0 - (u - self.duty) / (1.0 - self.duty)) - 1.0

    def forward(self, H, mask=None):
        p, X, n_rod, per = _nodes(H, self.at)
        Q = X.view(n_rod, per, 3)
        V = p.get("vel").view(n_rod, per, 3).to(X.dtype)
        dt = X.dtype
        a = torch.zeros_like(Q)
        out = {}

        # ---- the base is held in PLACE
        if self.anchor == "fixed":
            target, v_t = p.rod_anchor.to(dt), torch.zeros_like(V[:, 0, :])
        else:
            b = H.level(self.anchor)
            Xb, Vb = b.get("pos").to(dt), b.get("vel").to(dt)
            if self._idx is None:
                self._idx = torch.cdist(p.rod_anchor.to(dt), Xb).topk(
                    min(self.n_body, b.n), largest=False).indices
            target, v_t = Xb[self._idx].mean(1), Vb[self._idx].mean(1)
        a0 = (self.omega_n ** 2) * (target - Q[:, 0, :]) \
            - (2.0 * self.zeta * self.omega_n) * (V[:, 0, :] - v_t)
        a[:, 0, :] = a0

        # ---- ONE MOMENT ACROSS THE BASE JOINT, carrying both the clamp and the drive
        e = Q[:, 1, :] - Q[:, 0, :]
        e2 = (e * e).sum(1).clamp_min(1e-24)
        n = p.rod_axis.view(n_rod, per, 3)[:, 0, :].to(dt)
        n = n - ((n * e).sum(1) / e2)[:, None] * e
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)

        ang = torch.zeros(n_rod, device=X.device, dtype=dt)     # angular acceleration, rad/s^2
        if self.clamp:
            d0 = getattr(p, "rod_rest_dir", None)
            if d0 is None:
                d0 = (e / e.norm(dim=1, keepdim=True).clamp_min(1e-12)).detach().clone()
                p.register_buffer("rod_rest_dir", d0)
            u = torch.cross(n, d0, dim=1)
            th = torch.atan2((e * u).sum(1), (e * d0).sum(1))
            th_d = ((V[:, 1, :] - V[:, 0, :]) * u).sum(1) / e.norm(dim=1).clamp_min(1e-12)
            ang = ang - (self.clamp ** 2) * th - (2.0 * self.clamp_zeta * self.clamp) * th_d
        if self.moment:
            t = float(getattr(H, "frame", 0)) * float(getattr(H, "dt", 1.0))
            ang = ang + (self.moment * self.wave(t)) / e2
        a1 = ang[:, None] * torch.cross(n, e, dim=1)
        a[:, 1, :] += a1
        a[:, 0, :] -= a1                                        # the reaction, on the anchored node

        if self.anchor != "fixed":
            # THE REACTION ON THE BODY, spread over the same points the target was read from.
            k = self._idx
            n_b = k.shape[1]
            b_acc = torch.zeros_like(H.level(self.anchor).get("pos"))
            b_acc.index_add_(0, k.reshape(-1),
                             (-(a0 - a1) / n_b)[:, None, :].expand(-1, n_b, -1)
                             .reshape(-1, 3).to(b_acc.dtype))
            out[self.anchor] = b_acc

        a = a.reshape(-1, 3)
        if mask is not None:
            a = a * mask[:, None].to(dt)
        out[self.at] = a
        return out

"""Single-body motion: how one element moves when nothing else is acting on it.

Every contract here reads one element's own state and writes one element's own delta. There is
no neighbour relation and no interaction term anywhere in the module, which is what makes it
obvious when a new operator does not belong here.

In the order they appear below:

    drag             lateral   velocity-proportional damping: the overdamped limit's other half
    glide            lateral   move along the heading at the type's own speed
    sediment         lateral   a constant settling drift
    attractor_flow   lateral   ride a prescribed chaotic vector field, one of ten systems
    velocity_cruise  lateral   relax the speed toward a target without turning
    bounce           lateral   the wall and obstacle response: reflect the heading
    gravity          lateral   a uniform body force
"""
from __future__ import annotations
import torch
from plexus.models.base import Lateral
from plexus.models.registry import register_operator


def _rod_edges(lvl):
    """The chain's segment table, or the one its contiguous layout implies.

    `drag [along: chain]` is the one place outside `rod_ops` that needs to know which nodes are
    joined, and it used to answer that from row adjacency. It asks the table instead, so that a
    ring, a branch or a cross-linked axoneme gets the right tangent -- and so there is ONE
    statement of the rod's topology rather than two that can disagree. The fallback builds the
    chain table a bare `n_rod`/`per` set implies, so a spec that never declared a `mesh:` is
    unaffected.
    """
    m = getattr(lvl, "mesh", None)
    if m is not None and "E_srce" in m and m["E_srce"].numel():
        return m["E_srce"], m["E_trgt"], m.get("rest"), m.get("valence")
    n_rod = int(getattr(lvl, "n_rod", 1))
    per = lvl.n // max(n_rod, 1)
    idx = torch.arange(lvl.n, device=lvl.state.device, dtype=torch.long)
    srce = idx[(idx % per) < (per - 1)]
    return srce, srce + 1, None, None


@register_operator("drag", family="motion", set="particle", kind="lateral",
                   equation=r"""$$\mathbf a_i \;=\; -\,k\,\mathbf v_i$$""")
class Drag(Lateral):
    """Viscous drag: a force opposing the velocity and proportional to it. Composed with a
    force law it produces the overdamped limit; composed with noise it is a Langevin bath.

    particle -> particle: reads vel, emits an acceleration.

        d2x_i/dt2 = -k v_i  +  eta xi_i

    k is the drag coefficient, in inverse time -- 1/k is the time a particle takes to lose
    1 - 1/e of its speed, so large k means the velocity forgets its history within a step and
    the dynamics become effectively first-order. eta is `noise`, the amplitude of an isotropic
    random acceleration in world units per time squared, and xi_i a standard normal vector.
    Drag alone is dissipative; drag plus noise is a thermal bath whose equilibrium temperature
    is set by the ratio eta^2 / k, which is the fluctuation-dissipation relation.

    `field: <grid>` DRAGS AGAINST A MOVING FLUID INSTEAD OF A STILL ONE, and for an immersed
    filament it is the difference between a model and nothing.

        a = -k [ (v - u) resolved into tangential and transverse parts as below ]

    with u the fluid velocity INTERPOLATED AT THE PARTICLE from the named grid, by the same
    quadratic B-spline weights the MPM transfer uses. With no `field` the fluid is taken to be at
    rest and this is the plain law above.

    WHY IT HAD TO EXIST. The obvious way to immerse a rod in MLS-MPM water is to declare its nodes
    `entity: mpm_particle` and let `mpm_scatter` and `mpm_gather` carry the momentum. That fails
    for a SLENDER body, and the measurement is stark: a rod node weighs 7.3e-03 while the 3x3x3
    stencil it scatters over holds 1.39 of water, so the node is 0.53% of the mass its own velocity
    is averaged with -- and `mpm_gather` REPLACES a particle's velocity with that average. Over
    99% of whatever the drive did in a substep is discarded before it comes back. Measured on the
    cilium: the base swing fell from 60.3 degrees in air to 1.0 in water, and no moment or clamp
    recovered it (40 rad/s of clamp let the base flop to +107 degrees, 8e4 gave exactly 0.00, and
    the operator's own arithmetic is fine -- a unit test returns the quasi-static angle it should).
    Making the rod 100x denser moved the tip from 0.00 to 4.79 degrees, which identified the cause,
    and 1000x diverged.

    `react: true` PUTS THE EQUAL AND OPPOSITE BACK INTO THE FLUID, which is what makes this a
    coupling rather than a one-way sink. The force taken from the particle is scattered onto the
    same grid nodes, with the same weights, as a velocity increment -w m_p a dt / m_i. Without it
    the rod would feel the water and the water would never feel the rod, and a swimmer built that
    way would move with nothing pushed the other way.

    IT REPLACES `mpm_scatter`/`mpm_gather` ON THAT SET, it does not join them: a set that both
    gathers and drags is told its velocity twice.

    `along: chain` MAKES IT ANISOTROPIC, which is what a slender body in a fluid actually feels
    and is the whole mechanism by which a filament waving back and forth produces net thrust.

        t_hat  = the local tangent of the chain at this node
        a      = -k [ (v.t_hat) t_hat + ratio (v - (v.t_hat) t_hat) ]

    A long thin body in Stokes flow feels roughly TWICE the drag moving sideways as lengthwise
    (Gray & Hancock 1955), so `ratio` is about 2. On the power stroke the filament is broadside to
    its own motion and grips the fluid; on the recovery it is edge-on and slips through. Set
    `ratio: 1` and it is a perfectly symmetric paddle that cannot swim whatever waveform it is
    given, which is the control worth running once.

    The tangent is the centred difference of a node's neighbours along the chain, one-sided at the
    ends, and it is READ OFF THE SEGMENT TABLE (`_rod_edges` above) rather than off row adjacency:
    every segment hands its vector to both of its ends, which reproduces that difference exactly on
    a chain and is also defined on a ring, a branch, or a cross-linked axoneme, where there is no
    "previous row" to difference against. A per-SEGMENT tangent would leave the nodes, which are
    what carry the velocity, without one of their own.

    THIS IS A PARAMETER AND NOT A SECOND OPERATOR. `rod_drag` was written as its own contract and
    it should not have been: it was this law with one extra term, and a registry that spells the
    same mechanism twice makes a reader compare two docstrings to find out which one a spec is
    getting. Perfection is when there is nothing left to remove.

    IT TAKES MOMENTUM OUT OF THE SCENE, isotropic or not. The momentum goes into a fluid that is
    not represented, so a scene whose only sink is this operator does not conserve momentum and
    must not be measured as though it did.

    Reference: Stokes, G. G. (1851). On the effect of the internal friction of fluids on the
    motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106. Anisotropy: Gray, J. & Hancock, G.J.
    (1955). J. Exp. Biol. 32:802 (resistive force theory).
    """

    EMIT = "acceleration"            # second-order: a force on a body that has inertia
    SUPPORTED_DIMS = [2, 3]                      # acts on the D-vector velocity, dimension-generic
    REQUIRES_PARAMS = ["k"]                     # drag coefficient
    MECHANISM_TAGS = ["viscous_drag", "friction", "damping", "resistive_force_theory"]
    PARAM_ROLES = {"k": "drag_coefficient", "noise": "thermal_noise",
                   "along": "chain_to_make_the_drag_anisotropic_about_the_local_tangent",
                   "ratio": "transverse_over_tangential_drag_about_2_for_a_slender_filament",
                   "field": "grid_whose_velocity_the_drag_is_measured_against",
                   "react": "put_the_equal_and_opposite_back_into_that_fluid",
                   "coupling": "dial_the_fluid_structure_exchange_from_0_to_1",
                   "react_set": "the_fluid_s_particle_set_that_takes_the_reaction"}
    REFERENCE = ("Stokes, G. G. (1851). On the effect of the internal friction of fluids on "
                 "the motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.noise = float(params.get("noise", 0.0))     # isotropic Langevin noise (off by default)
        self.at = params.get("_at", "particle")
        self.along = str(params.get("along", "") or "").lower()
        if self.along not in ("", "chain"):
            raise ValueError(f"drag: `along` is `chain` or nothing, got {self.along!r}")
        self.ratio = float(params.get("ratio", 2.0))
        self.field = params.get("field")
        self.field = str(self.field) if self.field else None
        self.react = bool(params.get("react", False))
        self.react_set = params.get("react_set")
        self.react_set = str(self.react_set) if self.react_set else None
        # ---- THE OVERDAMPED BRANCH -----------------------------------------------------------
        # `emit: velocity` on this operator says the ROD has no inertia, which is what a cilium at
        # Reynolds number 1e-5 actually is (see builder step 0343: a real Platynereis cilium is
        # overdamped by a factor of 6.1 million, and carrying that explicitly would need a drag of
        # 6.1e7 per second and a 3.3e-08 s timestep). In that regime this operator's job changes
        # completely. It no longer applies a drag FORCE -- the rod's own mobility already is the
        # drag, since `rod_elastic` and `rod_motor` emit f/zeta directly -- so all that is left for
        # it to do on the rod is ADVECTION: the overdamped law in a moving fluid is
        #
        #     v_rod = u + f_internal / zeta
        #
        # and the operators supply the second term, so this one supplies `u`. The rod's stored
        # velocity is then u + f/zeta, which makes `v_rod - u` exactly f/zeta and gives the
        # reaction on the fluid for free: f_on_fluid = -zeta (v_rod - u).
        #
        # `zeta` IS A DRAG COEFFICIENT IN SIM MASS PER SECOND, not the per-unit-mass `k` above, and
        # it has to be because the reaction is a FORCE handed to a set whose masses are real. For
        # the 8-node Platynereis rod it is zeta_perp x h / (1 sim mass) = 2.0974e-03 Pa s x 2.857
        # um / 1.25e-13 kg = 4.794e+04. The check that it is the right number is that the rod's
        # own mobility already encodes the same zeta: k_bend = B/(zeta_perp h^4) = 4150 per second,
        # which is what run 0347 used, to a ratio of 1.000.
        self.zeta = float(params.get("zeta", 0.0))
        # `coupling` DIALS THE FLUID-STRUCTURE EXCHANGE FROM 0 TO 1, and it exists because going
        # from an uncoupled rod straight to a fully coupled one changes two things at once -- the
        # rod starts feeling the flow AND the flow starts feeling the rod -- so a run that breaks
        # cannot say which. At 0 this operator is inert in both directions and the spec is exactly
        # its own baseline; at 1 it is the physical coupling. Anything between is a weaker fluid,
        # not a different model: it scales the advection the rod feels and the reaction the fluid
        # receives by the SAME factor, so momentum still balances exactly and the residual stays at
        # machine precision whatever the dial is set to.
        self.coupling = float(params.get("coupling", 1.0))
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError(f"drag: coupling must be in [0, 1], got {self.coupling}")
        self.overdamped = str(params.get("emit", "") or "").lower() == "velocity"
        if self.overdamped and self.react and self.zeta <= 0.0:
            raise ValueError(
                "drag [emit: velocity] hands a FORCE back to the fluid, so it needs `zeta:`, the "
                "node drag coefficient in sim mass per second. The per-unit-mass `k` cannot serve: "
                "an overdamped rod has no mass for it to be per unit of.")
        if self.react and not self.field:
            raise ValueError("drag: `react` needs a `field` to react against -- there is nothing "
                             "to push back on otherwise.")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        occ = lvl.occ
        V = lvl.get("vel")
        # THE FLUID'S OWN VELOCITY AT THIS PARTICLE, by the same B-spline the MPM transfer uses,
        # so the rod feels the flow it is actually sitting in rather than a still medium.
        _u = None
        if self.field is not None:
            from plexus.operators.mpm_ops import bspline, stencil_offsets
            g = H.fields[self.field]
            D = V.shape[1]
            off = stencil_offsets(D, V.device)
            inv_dx = 1.0 / float(g.dx)
            _, w, flat = bspline(lvl.get("pos"), inv_dx, off, g.shape,
                                 bool(getattr(H, "periodic", False)))
            _u = (w[..., None] * g.v[flat].view(lvl.n, off.shape[0], D)).sum(1)
            _u = torch.nan_to_num(_u)
            V = V - _u
        if self.along == "chain":
            # THE LOCAL TANGENT, READ OFF THE SEGMENT TABLE rather than off adjacent rows of the
            # state array. Every segment (i, j) hands its vector e = x_j - x_i to BOTH its ends, so
            # an interior node of a chain accumulates e_prev + e_next = x_{i+1} - x_{i-1} and an end
            # node accumulates its one segment -- which is exactly the centred/one-sided difference
            # this used to write out by slicing, and is now also defined on a ring, a branch or a
            # cross-linked axoneme, none of which have a "previous row".
            X = lvl.get("pos")
            i, j, _, _ = _rod_edges(lvl)
            e = X[j] - X[i]
            t = torch.zeros_like(X)
            t.index_add_(0, i, e)
            t.index_add_(0, j, e)
            t = t / t.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self._tan = t                                  # reused by the overdamped anisotropy below
            v_par = (V * t).sum(1, keepdim=True) * t
            acc = -self.k * (v_par + self.ratio * (V - v_par)) * occ[:, None]
        else:
            self._tan = None
            acc = -self.k * V * occ[:, None]
        if self.noise > 0.0:                             # drag + noise = a Brownian/Langevin bath
            N, D = acc.shape
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=acc.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # ---- OVERDAMPED: the rod is ADVECTED, and the force it hands the fluid is zeta * (v - u).
        # `acc` above computed a drag force per unit mass, which an inertialess rod has no use for:
        # its mobility already IS the drag. What it still needs is the fluid's own motion, so the
        # emitted quantity becomes `u` itself -- the engine reads it as dx/dt and the rod is carried
        # by the flow on top of the f/zeta its own operators supply. `dv` is then exactly f/zeta,
        # so the reaction is a force this operator can state without ever seeing f.
        if self.overdamped:
            V_rod = lvl.get("vel").to(acc.dtype)
            dv = V_rod - (_u if _u is not None else 0.0)
            acc = (_u if _u is not None else torch.zeros_like(V_rod)) * (self.coupling * occ[:, None])
            if mask is not None:
                acc = acc * mask[:, None].to(acc.dtype)
            # THE SIGN, AND IT IS THE ONE THING HERE THAT CANNOT BE GUESSED. `_od_force` has to be
            # the force ON THE ROD, because the reaction scatter below negates whatever it is given
            # -- in the inertial branch it receives m_p * acc, and `acc` is already -k (v - u), so
            # the negation hands the fluid +zeta (v - u), which is the drag the rod exerts on it.
            # Writing +zeta * dv here instead handed the fluid -zeta (v - u): the water was pushed
            # AWAY from the rod's motion, which is anti-drag, so the relative velocity grew, the
            # push grew with it, and the loop ran away. Measured on cil_s31_odwater2 -- the
            # coupling itself was exact (residual 4e-08 with 3,298 fluid particles touched at
            # frame 1) and the rod still left the domain, ending at |pos| 1.01e+04 with the box at
            # 0 to 1 while holding its shape at 19.95 um of 20. A sign, not a magnitude.
            # ANISOTROPIC DRAG, Gray-Hancock: a slender filament resists NORMAL motion about twice
            # as hard as TANGENTIAL (`ratio` = zeta_perp/zeta_par ~ 2). The overdamped branch was
            # isotropic -- every rate carried the one zeta -- and run 0473 measured the consequence:
            # a released cell whose cilia beat perfectly did not swim, because a periodic stroke
            # with isotropic drag pumps no net fluid (the mean force over a beat is zero). The thrust
            # of a cilium IS this anisotropy: broadside on the power stroke (full zeta), edge-on on
            # recovery (zeta/ratio), so the two half-strokes push unequal fluid and the net is
            # forward. Applied ONLY when the drag declares `along: chain` (the filament flag); every
            # existing overdamped spec leaves it unset and is unchanged, and `ratio: 1` with the flag
            # ON is the paddle control that must not swim. The fluid reaction is scaled, its equal
            # and opposite still lands on the same water, so the momentum residual stays at zero.
            dvf = dv
            if getattr(self, "_tan", None) is not None and self.ratio != 1.0:
                dv_par = (dv * self._tan).sum(1, keepdim=True) * self._tan
                dvf = dv - (1.0 - 1.0 / self.ratio) * dv_par     # tangential drag reduced by ratio
            self._od_force = (-self.zeta * self.coupling) * dvf * occ[:, None]
        if self.react and _u is not None and self.react_set:
            # THE REACTION GOES BACK AS A DELTA ON THE FLUID'S OWN SET, not as a write to the grid,
            # and the difference is the whole reason the first two takes returned NaN.
            #
            # The grid is REBUILT from scratch by `mpm_scatter` every substep, so anything written
            # into g.v is erased before the next one reads it. Worse, an engine-integrated operator
            # placed inside a substep block is EVALUATED every substep but INTEGRATED once per tick
            # (engine.py:2695, `_integrate(H, sim.dt)`), so the rod moved once per frame while its
            # reaction was applied to the grid sixty-one times.
            #
            # `mpm_scatter` reads its own set's delta as a body acceleration every substep
            # (`a_ext = a_ext + H.delta(p.name)`, mpm_ops.py:552) and the engine snapshots and
            # restores deltas around the substep loop, so a delta returned HERE is seen once per
            # substep at its own value -- which is what an acceleration means. One force, applied
            # once, through a route that already existed.
            #
            # Localised by the same B-spline both ways: the rod's force is scattered onto the grid
            # nodes it overlaps, then gathered onto whichever fluid particles overlap those nodes.
            b = H.level(self.react_set)
            m_p = getattr(lvl, "mass", None)
            m_p = torch.ones(lvl.n, device=acc.device) if m_p is None else m_p.to(acc.dtype)
            f_node = torch.zeros_like(g.v)
            # WHICH FORCE IS HANDED BACK. Inertial: minus the drag force the rod felt, m_p * acc.
            # Overdamped: minus zeta (v_rod - u), computed above -- `acc` no longer holds a force
            # in that branch, it holds the advection velocity, so reusing it here would hand the
            # fluid a quantity with the wrong units AND the wrong sign.
            Xb = b.get("pos")
            _, wb, fb = bspline(Xb, inv_dx, off, g.shape, bool(getattr(H, "periodic", False)))
            m_b = getattr(b, "mass", None)
            m_b = torch.ones(b.n, device=acc.device) if m_b is None else m_b.to(acc.dtype)
            # a node's force is shared among the particles that overlap it, weighted as they are
            n_node = torch.zeros(g.v.shape[0], device=acc.device)
            n_node.index_add_(0, fb, wb.reshape(-1))
            # ---- THE IMPLICIT EXCHANGE, which is what lets the full coupling run at all -------
            # The explicit form hands the fluid zeta (v - u) dt every step, and that is unstable
            # whenever the fluid a node pushes on is light enough to overshoot within the step:
            # the condition is zeta dt / m_local < 1. Here m_local is the water mass under a rod
            # node's 3x3x3 stencil, about 0.24 sim-mass (8 particles per grid cell), so the ratio
            # is 1.96 at coupling 1.0 and 0.20 at 0.1 -- measured, and exactly why 0.1 "worked"
            # while 1.0 sent the rod to |pos| 1.4e25 with the startup ramp already in place
            # (builder 0442). The remedy is not a smaller dt (5x cost) but the implicit-Euler
            # solution of the two-body exchange within the step,
            #
            #     u_new = (u + (zeta dt/m) v) / (1 + zeta dt/m)
            #     force to the fluid = zeta (v - u) / (1 + zeta dt / m_local)
            #
            # which is unconditionally stable, identical to the explicit form when the ratio is
            # small, and NOT a fudge: it is the same physics integrated implicitly. The factor is
            # applied to `_od_force` in place so the momentum residual below is computed against
            # the force the fluid actually received. The smallest factor is printed with the
            # residual, because a factor well below 1 means the grid is too coarse for this
            # zeta and the run is leaning on the integrator.
            _f_src = self._od_force if self.overdamped else (m_p[:, None] * acc)
            self._implicit_min = 1.0
            if self.overdamped and self.zeta > 0.0:
                mass_node = torch.zeros(g.v.shape[0], device=acc.device, dtype=acc.dtype)
                mass_node.index_add_(0, fb, (wb * m_b[:, None]).reshape(-1))
                m_local = (w * mass_node[flat].view(w.shape)).sum(1).clamp_min(1e-12)   # per rod node
                _dt = float(getattr(H, "dt", 0.0))
                fac = 1.0 / (1.0 + (self.coupling * self.zeta * _dt) / m_local)
                self._od_force = self._od_force * fac[:, None]
                _f_src = self._od_force
                self._implicit_min = float(fac.min())
            f_node.index_add_(0, flat,
                              ((-_f_src)[:, None, :] * w[..., None]
                               ).reshape(-1, _f_src.shape[1]))
            share = f_node[fb] / n_node[fb].clamp_min(1e-12)[:, None]
            a_b = (wb[..., None] * share.view(b.n, off.shape[0], -1)).sum(1) / m_b[:, None]
            a_b = torch.nan_to_num(a_b)

            # ---- THE GATE: is momentum actually exchanged? --------------------------------------
            # Copied in spirit from `mesh_contact._record`, which has measured its own exchange
            # this way all along: the force the rod takes from the fluid and the force handed back
            # to the fluid must cancel, so
            #
            #     residual = |sum f_rod + sum f_water| / sum |f_rod|
            #
            # is dimensionless, is 0 for an exact exchange, and is 1 when the reaction never landed.
            # THIS IS THE NUMBER THE COUPLING IS JUDGED ON, and it exists because the previous three
            # attempts were judged on a movie instead: the water was bit-identical across 46,612,200
            # velocity components while the printed accelerations looked healthy, which a residual
            # would have caught in one frame. `n_react` counts the water particles that received
            # anything at all -- a residual of 0 with n_react 0 means the rod felt no drag either,
            # which is a different failure wearing the same number.
            f_rod = self._od_force if self.overdamped else (m_p[:, None] * acc)
            f_wat = m_b[:, None] * a_b
            den = f_rod.abs().sum().clamp_min(1e-30)
            resid = float((f_rod.sum(0) + f_wat.sum(0)).abs().sum() / den)
            n_react = int((a_b.abs().sum(1) > 0).sum())
            self._resid = max(getattr(self, "_resid", 0.0), resid)
            if int(getattr(H, "frame", 0)) in (1, 5, 20, 50, 200, 500, 800):
                print(f"[drag f{int(getattr(H, 'frame', 0))}] react -> {self.react_set}: "
                      f"momentum residual {resid:.3e} (0 = exact, 1 = the reaction never landed); "
                      f"{n_react} of {b.n} fluid particles touched; "
                      f"|f_rod| sum {float(den):.3e}, |f_water| sum {float(f_wat.abs().sum()):.3e}; "
                      f"|a_water| max {float(a_b.abs().max()):.3e}; implicit factor min {self._implicit_min:.3f}; "
                      f"fluid u max {float(_u.abs().max()):.3e}", flush=True)
            out_extra = {self.react_set: a_b}
        else:
            out_extra = {}
        out_extra[self.at] = acc
        return out_extra


@register_operator("glide", family="motion", set="cell", kind="lateral",
                   equation=r"""$$\dot{\mathbf x}_i=s_i\,\mathbf n_i+\eta\,\boldsymbol\xi_i$$""")
class Glide(Lateral):
    """Self-propulsion: move along the heading at the speed the element's own type carries.
    The heading is state that other operators steer; this one only walks it.

    cell -> cell: reads heading and the per-type move_speed, emits a velocity.

        dx_i/dt = s_i n_i  +  eta xi_i

    n_i is the unit heading vector and s_i the per-type `move_speed`, in world units per time.
    eta is `noise`, an isotropic translational noise of the same units, and xi_i a standard
    normal vector; with it the element is an active Brownian particle, without it a straight
    walker that only turns when something else rewrites its heading.

    Emits a velocity, not an acceleration: this is the overdamped, first-order sibling of
    `velocity_cruise`, which drives the same speed through inertia instead.

    Reference: the noisy case is the active Brownian particle; see Romanczuk, P., Bar, M.,
    Ebeling, W., Lindner, B. & Schimansky-Geier, L. (2012). Active Brownian particles: from
    individual to collective stochastic dynamics. Eur. Phys. J. Spec. Top. 202:1-162.
    """

    EMIT = "velocity"             # first-order: the engine integrates pos from this
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (heading is a [N,D] unit vector)
    REQUIRES_PARAMS = []                        # no required params — speed from `move_speed` type prop; noise optional
    MECHANISM_TAGS = ["self_propulsion", "motility", "active_brownian"]
    REQUIRES_TYPE_PROPS = ["move_speed"]
    PARAM_ROLES = {"noise": "translational_noise"}
    REFERENCE = ("Romanczuk, P., Bar, M., Ebeling, W., Lindner, B. & Schimansky-Geier, L. "
                 "(2012). Active Brownian particles: from individual to collective stochastic "
                 "dynamics. Eur. Phys. J. Spec. Top. 202:1-162.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.noise = float(params.get("noise", 0.0))      # isotropic translational noise (active Brownian; off by default)
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        h = lvl.heading                                   # [N, D] unit heading vector
        spd = lvl.move_speed                              # [N]
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = spd[:, None] * h                            # move along the heading
        if self.noise > 0.0:                              # glide + noise = an active Brownian walker
            vel = vel + self.noise * torch.randn(N, h.shape[-1], generator=getattr(H, "rng", None), device=dev)
        return {self.at: vel * m[:, None]}


@register_operator("sediment", family="motion", set="cell", kind="lateral",
                   equation=r"""$$\dot{\mathbf x}_i=\mathbf g$$""")
class Sediment(Lateral):
    """Sedimentation: a constant drift velocity, the terminal speed at which drag already
    balances the settling force, so no acceleration is ever resolved.

    cell -> cell: reads nothing, emits a velocity.

        dx_i/dt = (gx, gy)

    gx and gy are the drift components in world units per time. `g` is a convenience: it sets
    gy = -g, i.e. straight down, since the y axis is the screen's vertical in 2D. Note that the
    drift is written on axes 0 and 1 only, so in a 3D specification there is no settling along
    z -- use `gravity` with `gz` if a genuine third component is wanted.

    Distinguished from `gravity` by being first-order: gravity is an acceleration the medium
    has not yet damped, this is the steady state after it has.

    Reference: Stokes, G. G. (1851). On the effect of the internal friction of fluids on the
    motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106 (Stokes settling).
    """

    EMIT = "velocity"                                # first-order: a terminal velocity, not a force
    SUPPORTED_DIMS = [2, 3]                           # uniform drift is dimension-generic
    REQUIRES_PARAMS = []                             # no required params — all knobs optional (defaults in __init__)
    PARAM_ROLES = {"g": "sediment_magnitude", "gx": "sediment_x", "gy": "sediment_y"}
    REFERENCE = ("Stokes, G. G. (1851). On the effect of the internal friction of fluids on "
                 "the motion of pendulums. Trans. Camb. Phil. Soc. 9:8-106 (Stokes settling).")
    MECHANISM_TAGS = ["body_force", "differential_sedimentation"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 0.0))             # magnitude (world units / time)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        D = int(getattr(H, "dim", 2))                    # drift is a D-vector; -y (axis 1) is "down"
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        vel = torch.zeros(N, D, device=dev)
        vel[:, 0] = self.gx
        vel[:, 1] = self.gy
        return {self.at: vel * m[:, None]}


def _halvorsen(x, y, z, p):
    a = p.get("a", 1.4)
    return (-a * x - 4.0 * y - 4.0 * z - y * y,
            -a * y - 4.0 * z - 4.0 * x - z * z,
            -a * z - 4.0 * x - 4.0 * y - x * x)


def _lorenz(x, y, z, p):
    s = p.get("sigma", 10.0); r = p.get("rho", 28.0); b = p.get("beta", 8.0 / 3.0)
    return (s * (y - x), x * (r - z) - y, x * y - b * z)


def _aizawa(x, y, z, p):
    a = p.get("a", 0.95); b = p.get("b", 0.7); c = p.get("c", 0.6)
    d = p.get("d", 3.5); e = p.get("e", 0.25); f = p.get("f", 0.1)
    return ((z - b) * x - d * y,
            d * x + (z - b) * y,
            c + a * z - (z ** 3) / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * (x ** 3))


def _sprott_b(x, y, z, p):
    a = p.get("a", 1.0)
    return (a * y * z, x - y, 1.0 - x * y)


def _thomas(x, y, z, p):
    b = p.get("b", 0.208)
    return (torch.sin(y) - b * x, torch.sin(z) - b * y, torch.sin(x) - b * z)


def _rossler(x, y, z, p):
    a = p.get("a", 0.2); b = p.get("b", 0.2); c = p.get("c", 5.7)
    return (-y - z, x + a * y, b + z * (x - c))


def _dadras(x, y, z, p):
    a = p.get("a", 3.0); b = p.get("b", 2.7); c = p.get("c", 1.7)
    d = p.get("d", 2.0); e = p.get("e", 9.0)
    return (y - a * x + b * y * z, c * y - x * z + z, d * x * y - e * z)


def _chen(x, y, z, p):
    a = p.get("a", 35.0); b = p.get("b", 3.0); c = p.get("c", 28.0)
    return (a * (y - x), (c - a) * x - x * z + c * y, x * y - b * z)


def _chua(x, y, z, p):
    a = p.get("alpha", 15.6); b = p.get("beta", 28.58)
    m0 = p.get("m0", -1.1428571); m1 = p.get("m1", -0.7142857)
    h = m1 * x + 0.5 * (m0 - m1) * (torch.abs(x + 1.0) - torch.abs(x - 1.0))
    return (a * (y - x - h), x - y + z, -b * y)


def _rabinovich_fabrikant(x, y, z, p):
    al = p.get("alpha", 1.1); g = p.get("gamma", 0.87)
    return (y * (z - 1.0 + x * x) + g * x,
            x * (3.0 * z + 1.0 - x * x) + g * y,
            -2.0 * z * (al + x * y))


_FIELDS = {
    "halvorsen": _halvorsen, "lorenz": _lorenz, "aizawa": _aizawa, "sprott_b": _sprott_b,
    "thomas": _thomas, "rossler": _rossler, "dadras": _dadras, "chen": _chen, "chua": _chua,
    "rabinovich_fabrikant": _rabinovich_fabrikant,
}
ATTRACTOR_SYSTEMS = tuple(_FIELDS)


def attractor_velocity(system: str, pos: torch.Tensor, params: dict | None = None) -> torch.Tensor:
    """The strange-attractor vector field f(x): dx/dt for every point of `pos` [N, 3].
    `system` is one of `ATTRACTOR_SYSTEMS`; `params` overrides that system's constants.
    Returns a velocity tensor [N, 3] on the same device/dtype as `pos`."""
    if system not in _FIELDS:
        raise ValueError(f"attractor_flow: unknown system {system!r}; "
                         f"choose one of {list(ATTRACTOR_SYSTEMS)}")
    p = params or {}
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xd, yd, zd = _FIELDS[system](x, y, z, p)
    return torch.stack([xd, yd, zd], dim=-1)


@register_operator("attractor_flow", family="motion", set="particle", kind="lateral", title="Flow along a strange attractor",
                   equation=r"""$$\dot{\mathbf x} \;=\; f_{\text{system}}(\mathbf x),
\qquad
\text{e.g. Lorenz: }\;
\dot x=\sigma(y-x),\;\; \dot y=x(\rho-z)-y,\;\; \dot z=xy-\beta z$$""")
class AttractorFlow(Lateral):
    """Ride a prescribed chaotic vector field: every particle is advected by the same
    dissipative flow, so the set traces out that system's strange attractor.

    particle -> particle: reads pos, emits a velocity.

        dx_i/dt = c f_system(x_i)

    f_system is one of ten named autonomous 3D vector fields (`system`), each with its own
    constants, which a specification overrides by naming them directly on the operator line:
    lorenz (sigma, rho, beta), rossler (a, b, c), halvorsen (a), aizawa (a..f), sprott_b (a),
    thomas (b), dadras (a..e), chen (a, b, c), chua (alpha, beta, m0, m1) and
    rabinovich_fabrikant (alpha, gamma). c is `scale`, a dimensionless time rescaling: c > 1
    runs the same trajectory faster. `clamp` caps |dx/dt| in world units per time and is a
    safety device, not physics -- a nonzero value distorts the field wherever it binds.

    3D only, and that is a theorem rather than a limitation of the code: by Poincare-Bendixson
    a continuous autonomous flow in the plane cannot be chaotic. The engine integrates with
    forward Euler, so the timestep must stay small; too large a one drifts off the attractor
    that a dissipative flow would otherwise stay pinned to.

    Reference: Lorenz, E. N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci. 20:130-141;
    Rossler, O. E. (1976). Phys. Lett. A 57:397-398; Chua, L. O. et al. (1986). IEEE Trans.
    Circuits Syst. 33:1072-1118; Rabinovich, M. I. & Fabrikant, A. L. (1979). Sov. Phys. JETP
    50:311-317; Chen, G. & Ueta, T. (1999). Int. J. Bifurcat. Chaos 9:1465-1466; Sprott, J. C.
    (1994). Phys. Rev. E 50:R647-R650; Thomas, R. (1999). Int. J. Bifurcat. Chaos 9:1889-1905.
    """

    EMIT = "velocity"                 # delta IS dx/dt; engine integrates x += dt * f(x)
    SUPPORTED_DIMS = [3]              # continuous autonomous chaos requires >=3D (Poincare-Bendixson)
    REQUIRES_PARAMS = ["system"]      # which attractor to ride
    MECHANISM_TAGS = ["strange_attractor", "deterministic_chaos", "dissipative_flow",
                      "sensitive_dependence", "phase_space_contraction", "dynamical_system"]
    PARAM_ROLES = {"system": f"which attractor: one of {list(ATTRACTOR_SYSTEMS)}",
                   "scale": "time-rescale the flow (f -> scale*f); >1 = faster",
                   "clamp": "max |velocity| safety cap (0 = off)"}
    REFERENCE = ("Lorenz, E. N. (1963). Deterministic nonperiodic flow. J. Atmos. Sci. "
                 "20:130-141; Rossler (1976) Phys. Lett. A 57:397; Chua et al. (1986) IEEE "
                 "TCAS 33:1072; Rabinovich & Fabrikant (1979) Sov. Phys. JETP 50:311; Chen & "
                 "Ueta (1999) IJBC 9:1465; Sprott (1994) Phys. Rev. E 50:R647; Thomas (1999) "
                 "IJBC 9:1889.")

    # spec-line keys that are plumbing/knobs, not per-system physical constants
    _NON_CONST = {"op", "at", "to", "from", "_at", "system", "scale", "clamp",
                  "emit", "after_frame", "before_frame"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.system = str(params["system"])
        if self.system not in _FIELDS:
            raise ValueError(f"attractor_flow: unknown system {self.system!r}; "
                             f"choose one of {list(ATTRACTOR_SYSTEMS)}")
        self.scale = float(params.get("scale", 1.0))
        self.clamp = float(params.get("clamp", 0.0))
        self.const = {k: float(v) for k, v in params.items()
                      if k not in self._NON_CONST and isinstance(v, (int, float))}

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        vel = attractor_velocity(self.system, pos, self.const) * self.scale
        if self.clamp > 0:                                       # safety cap on |v|
            mag = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            vel = vel * (mag.clamp(max=self.clamp) / mag)
        vel = vel * occ[:, None]                                 # dormant points hold still
        if mask is not None:
            vel = vel * mask[:, None].float()
        return {self.at: vel}


@register_operator("velocity_cruise", "cruise", family="motion", set="particle", kind="lateral",
                   equation=r"""$$\ddot{\mathbf x}_i=k\big(v_0-\lVert\mathbf v_i\rVert\big)\frac{\mathbf v_i}{\lVert\mathbf v_i\rVert}+\eta\,\boldsymbol\xi_i+c\,(-v_{iy},v_{ix})$$""")
class VelocityCruise(Lateral):
    """Cruising: drive the speed toward a target without turning the particle. The inertial,
    second-order sibling of `glide` -- the same self-propulsion, reached through a force.

    particle -> particle: reads vel, emits an acceleration.

        d2x_i/dt2 = k (v0 - |v_i|) v_i/|v_i|  +  eta xi_i  +  c (-v_iy, v_ix)

    v0 is the cruising speed in world units per time and k the restoring stiffness in inverse
    time, so 1/k is how long the particle takes to recover its speed after being slowed. The
    restoring term points along the current heading v_i/|v_i|, which is why it never turns the
    particle: speed and direction are decoupled, and only the noise and chirality terms rotate
    it. eta is `noise`, an isotropic random acceleration, and xi_i a standard normal vector --
    the Vicsek control parameter, trading order against disorder. c is `chirality`, in inverse
    time, a force at 90 degrees to the velocity; it makes trajectories curve consistently one
    way and so produces swirls. Chirality is 2D only, since a single rotation sense needs a
    plane to be defined in.

    Reference: Schweitzer, F., Ebeling, W. & Tilch, B. (1998). Complex motion of Brownian
    particles with energy depots. Phys. Rev. Lett. 80:5044-5047.
    """

    EMIT = "acceleration"            # second-order: self-propulsion through a force
    SUPPORTED_DIMS = [2, 3]                     # speed restoration + isotropic noise are dimension-generic
    REQUIRES_PARAMS = ["v0"]
    MECHANISM_TAGS = ["self_propulsion", "vicsek", "active_matter"]
    PARAM_ROLES = {"v0": "cruising_speed", "noise": "orientation_noise", "chirality": "rotational_bias"}
    REFERENCE = ("Schweitzer, F., Ebeling, W. & Tilch, B. (1998). Complex motion of Brownian "
                 "particles with energy depots. Phys. Rev. Lett. 80:5044-5047.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.v0 = float(params["v0"])
        self.k = float(params.get("k", 1.0))                  # speed-restoring stiffness
        self.noise = float(params.get("noise", 0.0))          # isotropic orientation noise
        self.chirality = float(params.get("chirality", 0.0))  # 2D rotational bias (swirls)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        vel, occ = lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        dev = vel.device
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        acc = self.k * (self.v0 - speed) * (vel / speed)              # restore cruising speed along heading
        if self.noise > 0.0:
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        if self.chirality != 0.0 and D == 2:
            acc = acc + self.chirality * torch.stack([-vel[:, 1], vel[:, 0]], dim=-1)   # 90deg -> swirls
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}


def _in_obstacles(x, y, obstacles):
    """Bool mask: is (x, y) inside any obstacle? rect=[x0,y0,x1,y1], disc=[cx,cy,r]."""
    hit = torch.zeros_like(x, dtype=torch.bool)
    for o in (obstacles or []):
        if len(o) == 4:
            x0, y0, x1, y1 = o
            hit = hit | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        elif len(o) == 3:
            cx, cy, r = o
            hit = hit | (((x - cx) ** 2 + (y - cy) ** 2) <= r * r)
    return hit


def _random_unit(n, D, rng, device):
    """A random unit vector per agent [n, D] (isotropic re-heading off obstacles)."""
    v = torch.randn(n, D, generator=rng, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)


@register_operator("bounce", family="boundary", set="cell", kind="lateral",
                   equation=r"""$$v_{i,a}\leftarrow-\,v_{i,a}\quad\text{on the axis }a\text{ of the wall struck}$$""")
class Bounce(Lateral):
    """The boundary response for a heading-driven walker: turn it around before it leaves the
    world, rather than letting it exit and clamping it back.

    cell -> cell: reads pos, heading and move_speed, writes heading in place.

    The element's tentative next position is x_i + dt s_i n_i, one step of `glide`. For every
    axis on which that would fall outside the box, the heading's component on that axis is
    negated -- which is specular reflection off an axis-aligned wall, and preserves the speed:

        n_ia <- -n_ia   for each axis a where the step would exit,   then n_i <- n_i / |n_i|

    An obstacle (a 2D rectangle [x0, y0, x1, y1] or disc [cx, cy, r]) has no single axis-aligned
    normal to reflect against, so the element is re-headed instead. `noise` is the fraction of
    that re-heading that is random, from 0 to 1: 0 reverses the heading exactly, 1 picks an
    isotropic random direction, and intermediate values blend the two before renormalising.

    Under periodic boundaries the operator returns immediately: a torus has no wall.

    Reference: none -- specular reflection off a box is standard practice, not a result.
    Plexus (this work).
    """

    EMIT = None                                 # writes heading in place, returns no delta
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic specular wall reflection
    REQUIRES_PARAMS = []                         # no required params — `noise` optional
    MECHANISM_TAGS = ["boundary_condition", "wall_reflection", "obstacle_avoidance", "steering"]
    REQUIRES_TYPE_PROPS = ["move_speed"]        # needs the step length it is about to take
    PARAM_ROLES = {"noise": "obstacle_reheading_randomness"}
    REFERENCE = "Plexus (this work); specular reflection off a box is standard practice."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.noise = float(params.get("noise", 0.0))    # obstacle re-head: 0 = reverse (deterministic), 1 = isotropic
        self.at = params.get("_at", "cell")

    def forward(self, H, mask=None):
        if getattr(H, "periodic", False):
            return {}                                       # torus: nothing to bounce off
        lvl = H.level(self.at)
        dev = lvl.state.device
        N = lvl.n
        pos = lvl.get("pos")                                # [N, D]
        h = lvl.heading                                     # [N, D] unit heading
        spd = lvl.move_speed
        dt = float(getattr(H.config, "dt", 1.0))
        box = H.world_size                                  # [D] per-axis box size
        m = (mask.float() if mask is not None else torch.ones(N, device=dev)) * lvl.occ
        keep = (m > 0)[:, None]

        nxt = pos + dt * spd[:, None] * h                   # tentative next position
        out = (nxt < 0) | (nxt > box[None, :])              # which axes would exit the box
        new_h = torch.where(out, -h, h)                     # specular reflect the exiting components

        # obstacles (2D maze rects/discs): re-head where the step would enter one --
        # they carry no single axis-aligned normal to reflect against. The `noise` knob
        # (default 0) sets how random that re-heading is: 0 reverses the heading (-h,
        # deterministic), 1 picks an isotropic random direction (the old behaviour).
        obs = getattr(H, "obstacles", [])
        if obs and pos.shape[1] == 2:
            hit = _in_obstacles(nxt[:, 0], nxt[:, 1], obs)
            rehead = -h if self.noise <= 0.0 else \
                (1.0 - self.noise) * (-h) + self.noise * _random_unit(N, 2, H.rng, dev)
            new_h = torch.where(hit[:, None], rehead, new_h)

        new_h = new_h / new_h.norm(dim=1, keepdim=True).clamp(min=1e-9)
        lvl.heading = torch.where(keep, new_h, h)
        return {}


@register_operator("gravity", family="mechanics", set="cell", kind="lateral", title="Gravity",
                   equation=r"""$$\mathbf a_{\text{ext}} \;=\; \mathbf g \qquad (\text{default } \mathbf g=(0,-g_y))$$""")
class GravityOperator(Lateral):
    """A uniform body force: the same acceleration on every element, independent of its state.

    cell -> cell: reads nothing, emits an acceleration the MPM substep consumes.

        a_i = (gx, gy, gz)

    The three components are in world units per time squared. `g` is a convenience setting
    gy = -g. The default direction is -y because in 2D that is the screen's vertical; in 3D the
    screen's vertical is z, as both mplot3d and VTK put z up, so a 3D specification wanting a
    fall that looks vertical writes `gy: 0.0, gz: -9.0`. gz defaults to 0, so a specification
    written before z existed keeps the -y fall it had.

    Emits `mpm_acceleration`, which the MPM substep consumes as an external body acceleration.
    It is deliberately NOT engine-integrated on the cell set: a cell here is a centroid read out
    from its material points, so integrating it directly would make it fall twice.

    Reference: Newton, I. (1687). Philosophiae Naturalis Principia Mathematica.
    """

    EMIT = "mpm_acceleration"                  # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]                           # uniform body force is dimension-generic
    REQUIRES_PARAMS = []                              # no required params — direction/magnitude optional (default -y down)
    PARAM_ROLES = {"g": "gravity_magnitude", "gx": "gravity_x", "gy": "gravity_y",
                   "gz": "gravity_z"}
    REFERENCE = "Newton, I. (1687). Philosophiae Naturalis Principia Mathematica."
    MECHANISM_TAGS = ["body_force", "uniform_acceleration"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")              # the set this acts on (engine-injected)
        self.g = float(params.get("g", 10.0))            # magnitude (world units / time^2)
        self.gx = float(params.get("gx", 0.0))           # x-component (default 0)
        self.gy = float(params.get("gy", -self.g))       # y-component (default -g: down)
        self.gz = float(params.get("gz", 0.0))

    def forward(self, H, mask=None):
        cell = H.level(self.at)
        dev = cell.state.device
        D = int(getattr(H, "dim", 2))                    # gravity is a D-vector
        accel = torch.zeros(cell.n, D, device=dev)
        accel[:, 0] = self.gx
        accel[:, 1] = self.gy
        if D > 2:
            accel[:, 2] = self.gz
        if mask is not None:
            accel = accel * mask.float()[:, None]
        return {cell.name: accel}

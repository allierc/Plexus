"""The bacterial flagellar motor: stator units pushing a rotor.

THE MOTOR IS NOT AN OPERATOR THAT SAYS "ROTATE". builder/instruction.md section 0b: the torque
comes from stator units acting on a rotor, and the rotation rate has to COME OUT of that torque
against whatever load the rotor carries. So the one contract in this module is the push itself --
each stator unit, an element of its own set, applies a tangential force on the rotor material
within its reach -- and nothing here knows what speed that produces.

In the order they appear below:

    stator_push      exchange   each stator unit pushes the rotor material in its reach, tangentially
                                -- the stated baseline: a constant force per unit
    stator_contact   exchange   the elastic linkage between a unit and the ring: torque = kappa x
    stator_step      lateral    the unit's ion-driven stepping, feeding the stretch x against its load
    aggregate        aggregate  Sigma_pi: the stators' transits summed into the cell's motor current
    broadcast        broadcast  pi^*: the cell's potential handed to every stator
    membrane_potential lateral  the cell's capacitor: respiration in, motors and leak out
    stator_membrane  exchange   the membrane as a place: each unit's current deposited where it stands

`stator_push` is the reduction section 0b allows as a baseline; the pair below it is the machine,
from which the torque-speed curve is to EMERGE (the switch, CheY-P flipping the direction, is
next). A helix seed and a resistive-force helix drive lived here for one day
(builder/exp_02_bacterium steps 0011-0041) and were removed with those steps on 2026-09-23: the
drive propelled its cell by a fitted drag law rather than by the water, and the seed had no user
left once the drive was gone. The filament as a 1D mesh, the drag that couples it to water and
the cell body live in `rod_ops` and `motion_ops` and are reused, not restated -- a merge of
`rod_ops` and this module into one cilium/flagellum module is under discussion with the session
that owns `rod_ops`.
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Aggregate, Broadcast, Exchange, Lateral
from plexus.models.registry import register_operator


@register_operator("stator_push", family="motility", set="particle", kind="exchange",
                   equation=r"""$$\mathbf r_p=(\mathbf x_p-\mathbf c)-\big((\mathbf x_p-\mathbf c)\cdot\hat{\mathbf a}\big)\hat{\mathbf a},\qquad \mathbf t_p=s\,\frac{\hat{\mathbf a}\times\mathbf r_p}{\lVert\mathbf r_p\rVert}$$""")
class StatorPush(Exchange):
    """Each stator unit pushes the rotor material within its reach, tangentially about the axis.

    stator -> particle: reads the stator set's positions and the rotor particles' positions and
    masses; emits an external acceleration the MPM substep consumes (`EMIT = mpm_acceleration`,
    the same route `mesh_contact` and `mpm_anchor` take).

        S_k                 the position of stator unit k
        P_k = { p : |x_p - S_k| < reach }          the rotor particles unit k is in contact with
        r_p = (x_p - c) - ((x_p - c) . a) a        the particle's lever arm from the axis
        t_p = s (a x r_p) / |r_p|                  the tangential direction, s = +1 CCW about a
        f_p = F / |P_k| t_p                        unit k's force F, shared among its particles
        a_p = f_p / m_p                            what the substep receives

    F is `force`, the force ONE stator unit exerts, in simulation force units; `reach` the contact
    distance in world units; `axis` the rotor's axis (unit vector, default z); `centre` the point
    the axis passes through (default: the centroid of the stator set, which is derived from the
    anatomy rather than declared twice); `sign` +1 for counter-clockwise rotation seen from the
    +axis side, which is the E. coli run direction viewed from outside the cell.

    THE TORQUE IS A MEASUREMENT, NOT A PARAMETER. Each unit's torque is F times the lever arm of
    the material it actually touches, so the total is N F r only if every unit touches material at
    radius r. The operator sums (r_p x f_p) . a over the particles it pushed and prints it on the
    same frames `drag` prints its momentum residual, beside the nominal N F r, so a stator that
    reaches nothing -- placed off the rim, or with a reach the grid cannot see -- is loud rather
    than a motor that quietly turns slower.

    WHAT THIS IS AND WHAT IT IS NOT. A stator unit is MotA5B2, an ion-driven rotary machine that
    steps against FliG on the C-ring rim; a single unit exerts about 7.3 pN (Ryu, Berry & Berg 2000,
    Nature 403:444, as used by Drobnic et al. 2025 to predict 1,606 pN nm for eleven units on the
    E. coli/Salmonella C-ring). The plateau of Berg's torque-speed curve -- constant torque up to a
    knee near 170 Hz at 23 C -- is exactly what a constant force per unit gives, and that is what
    this operator implements. STATED LIMITATIONS: (1) the force is constant in speed, so the
    knee and the fall to zero torque at ~300 Hz (Chen & Berg 2000) are not here; removing that
    needs the unit's stepping kinetics, a rate that saturates with the ion flux; (2) the number of
    units is the size of the stator set and does not remodel with load (Lele et al. 2013); (3) the
    equal and opposite force on each unit goes into the cell wall the stator is anchored to and is
    not returned to any set -- a cell body that should counter-rotate will receive it when the
    body exists. None of these is a rotation rate: that is measured off the run.

    Reference: Berg, H.C. (2003). Annu. Rev. Biochem. 72:19-54; Ryu, W.S., Berry, R.M. & Berg, H.C.
    (2000). Nature 403:444-447; Reid, S.W. et al. (2006). PNAS 103:8066-8071 (>= 11 units,
    1,260 +- 190 pN nm); Drobnic, T. et al. (2025). Nat. Microbiol. 10:1723-1740.
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["stator", "particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = []
    REQUIRES_PARAMS = ["stator", "force"]
    MECHANISM_TAGS = ["rotary_motor", "stator_unit", "torque_generation", "active_force"]
    PARAM_ROLES = {"stator": "the_set_whose_elements_are_the_stator_units",
                   "force": "force_one_unit_exerts_in_sim_force_units",
                   "reach": "contact_distance_from_a_unit_in_world_units",
                   "axis": "the_rotor_axis_unit_vector",
                   "centre": "a_point_on_the_axis_default_the_stator_centroid",
                   "sign": "+1_counter_clockwise_seen_from_the_plus_axis_side"}
    PARAM_UNITS = {"force": "force", "reach": "length"}
    REFERENCE = ("Berg, H.C. (2003). Annu. Rev. Biochem. 72:19-54; Ryu, Berry & Berg (2000). "
                 "Nature 403:444; Reid et al. (2006). PNAS 103:8066; Drobnic et al. (2025). "
                 "Nat. Microbiol. 10:1723.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.stator = str(params["stator"])
        self.force = self.tunable(params.get("force"), 0.0)
        self.reach = float(params.get("reach", 0.05))
        self.axis = [float(v) for v in (params.get("axis") or [0.0, 0.0, 1.0])]
        self.centre = params.get("centre")
        self.sign = float(params.get("sign", 1.0))
        if self.reach <= 0.0:
            raise ValueError(f"stator_push: reach must be > 0, got {self.reach}")
        if self.sign not in (1.0, -1.0):
            raise ValueError(f"stator_push: sign is +1 (CCW seen from +axis) or -1, got {self.sign}")
        self._cvec = {}
        self.torque = 0.0                      # the last frame's delivered torque, for probes
        self.torque_nominal = 0.0

    def _const(self, key, values, dev, dtype):
        """A spec constant as a device tensor, built once (a per-call host copy breaks capture)."""
        ck = (key, str(dev), str(dtype))
        if ck not in self._cvec:
            self._cvec[ck] = torch.tensor([float(v) for v in values], device=dev, dtype=dtype)
        return self._cvec[ck]

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if self.stator not in H.levels:
            raise ValueError(f"stator_push: `stator: {self.stator}` is not a set in this model "
                             f"(have: {', '.join(H.levels)})")
        st = H.level(self.stator)
        X = p.get("pos")
        dev, dt = X.device, X.dtype
        S = st.get("pos").to(dt)
        S = S[st.active] if st.occ.numel() else S
        m_p = getattr(p, "mass", None)
        m_p = torch.ones(p.n, device=dev, dtype=dt) if m_p is None else m_p.to(dt)

        a = self._const("axis", self.axis, dev, dt)
        a = a / a.norm().clamp_min(1e-12)
        c = self._const("centre", self.centre, dev, dt) if self.centre is not None else S.mean(0)
        rel = X - c
        r_perp = rel - (rel @ a)[:, None] * a
        r_len = r_perp.norm(dim=1)
        t_hat = self.sign * torch.cross(a.expand_as(r_perp), r_perp, dim=1) / r_len.clamp_min(1e-12)[:, None]

        live = p.occ > 0
        if mask is not None:
            live = live & mask
        # WHICH PARTICLES EACH UNIT TOUCHES: within `reach` of its centre. [N, n_stator]
        d2 = torch.cdist(X, S) ** 2
        touch = (d2 < self.reach ** 2) & live[:, None]
        n_k = touch.sum(0).to(dt)                                   # particles per unit
        share = torch.where(n_k > 0, self.force / n_k.clamp_min(1.0), torch.zeros_like(n_k))
        f_mag = (touch.to(dt) * share[None, :]).sum(1)              # force magnitude per particle
        f = f_mag[:, None] * t_hat
        acc = f / m_p[:, None]

        # THE TORQUE ACTUALLY DELIVERED, against the nominal N F r_mean of the touched material.
        T = float((r_len * f_mag).sum())
        n_touch = int((f_mag > 0).sum())
        r_mean = float((r_len * (f_mag > 0)).sum() / max(n_touch, 1))
        self.torque = T
        self.torque_nominal = float(self.force) * int(S.shape[0]) * r_mean
        fr = int(getattr(H, "frame", 0))
        if fr in (1, 5, 20, 50, 200, 500, 800, 1000, 1400):
            lo, hi = int(n_k.min()), int(n_k.max())
            print(f"[stator_push f{fr}] {int(S.shape[0])} units, {n_touch} particles touched "
                  f"({lo}-{hi} per unit), mean lever arm {r_mean:.4f} world; torque delivered "
                  f"{T:.4e} vs N F r {self.torque_nominal:.4e} (sim force x world)"
                  + (";  A UNIT TOUCHES NOTHING" if lo == 0 else ""), flush=True)
        return {self.at: acc}


# ============================================================================================
# THE STATOR AS A MACHINE: an ion-driven stepper on an elastic linkage, instead of a force
# ============================================================================================
#
# `stator_push` above puts a constant force on the rotor. These two operators replace it with
# what a stator unit is: a MotA5B2 complex whose MotA pentamer turns around MotB one step per ion
# transit (Santiveri et al. 2020; Deme et al. 2020), each step advancing its grip on the C-ring by
# one FliG spacing, through a linkage that is elastic (the MotB plug and peptidoglycan anchor, and
# here the PflA/PflB scaffold). Torque is then the linkage's stretch times its stiffness, and the
# rate at which the stretch is fed is set by ion kinetics against the work the stretch costs, so
# the torque-speed relation -- Berg's plateau, knee and fall -- is an OUTCOME of the two numbers a
# stator is made of (the free energy per ion and the step size) and not a curve written in.
#
# The split follows the language: `stator_contact` is the EXCHANGE between the stator set and the
# rotor's material points (the mechanical contact: force out, ring speed in), `stator_step` is the
# LATERAL dynamics of the stator set itself (the chemistry: how fast the stretch is fed). They
# meet on the stator's own state block `x`, the linkage stretch in radians, which `stator_step`
# integrates and `stator_contact` turns into force. Two buffers carry the contact's readings to
# the stepper (`stator_torque`, `stator_omega`), the way rod_ops passes its anchor.


@register_operator("stator_contact", family="motility", set="particle", kind="exchange",
                   equation=r"""$$P_k=\{p:\lVert\mathbf x_p-\mathbf S_k\rVert<\text{reach}\},\qquad \tau_k=\kappa\,x_k,\qquad \mathbf f_p=\frac{\tau_k}{\sum_{q\in P_k} r_q}\,\mathbf t_p$$""", title="Stator-to-rotor linkage")
class StatorContact(Exchange):
    """The elastic contact between each stator unit and the rotor material under it.

    stator -> particle: reads the stator set's positions and its stretch block `x`, the rotor
    particles' positions, velocities and masses; emits an external acceleration the MPM substep
    consumes (`EMIT = mpm_acceleration`). Writes two buffers on the stator set for `stator_step`:
    `stator_torque` (what each unit delivered) and `stator_omega` (how fast the ring under it
    turns).

        P_k = { p : |x_p - S_k| < reach }         the rotor particles unit k grips
        tau_k = kappa x_k                         the torque of the stretched linkage
        f_p = tau_k / sum_{q in P_k} r_q  t_p     shared so that sum_p r_p f_p = tau_k exactly
        omega_k = < (v_p . t_p) / r_p >_{P_k}     the ring's angular rate under the unit

    `kappa` is the linkage stiffness in sim torque per radian, `x` the stretch in radians that
    `stator_step` feeds and this contact relaxes as the ring turns; `reach`, `axis`, `centre`,
    `sign` as in `stator_push`. A unit that grips no material delivers nothing and reads
    omega = 0, and says so.

    WHY THE HANDOVER NEEDS NO BOOKKEEPING. A real unit steps from one FliG protomer to the next;
    here the ring is a continuum of material points and the unit always grips what is under it,
    so the stretch is the only memory: it grows by one step per ion and shrinks as the ring moves
    on, and its rate of change is `stator_step`'s to integrate. What is lost against a protomer
    model is the ratchet's discreteness -- the 34 teeth -- which the map may put back.

    Reference: Meacci, G. & Tu, Y. (2009). PNAS 106:3746-3751 (the elastic linkage and the
    torque-speed curve from it); Santiveri, M. et al. (2020). Cell 183:244-257; Deme, J.C. et al.
    (2020). Nat. Microbiol. 5:1553-1564 (MotA turning about MotB).
    """

    EMIT = "mpm_acceleration"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = True
    INPUTS = ["stator", "particle"]
    OUTPUTS = ["particle"]
    READS = ["pos", "vel", "x"]
    WRITES = []
    REQUIRES_PARAMS = ["stator", "kappa"]
    MECHANISM_TAGS = ["rotary_motor", "stator_unit", "elastic_linkage", "torque_generation"]
    PARAM_ROLES = {"stator": "the_set_whose_elements_are_the_stator_units",
                   "kappa": "linkage_stiffness_sim_torque_per_radian",
                   "anchor": "the_material_set_each_unit_is_held_by", "anchor_reach": "how_far_a_unit_binds_its_anchor",
                   "reach": "contact_distance_from_a_unit_in_world_units",
                   "axis": "the_rotor_axis_unit_vector",
                   "centre": "a_point_on_the_axis_default_the_stator_centroid",
                   "sign": "+1_counter_clockwise_seen_from_the_plus_axis_side"}
    # UNITS, IN THE PAPER'S THREE SCALES: a torque per radian is an energy (F L); the stretch is an
    # angle. The buffers this writes (torque, angular rate) are not blocks and carry no annotation.
    PARAM_UNITS = {"reach": "length", "kappa": "energy", "kT": "energy", "zeta_rot": "F*L*T", "anchor_reach": "length"}
    MAY_MUTATE_INTEGRATED_STATE = True                    # the units' positions ride on their anchor
    BLOCK_UNITS = {"x": "1"}
    REFERENCE = ("Meacci, G. & Tu, Y. (2009). PNAS 106:3746; Santiveri, M. et al. (2020). Cell "
                 "183:244; Deme, J.C. et al. (2020). Nat. Microbiol. 5:1553.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.stator = str(params["stator"])
        self.kappa = float(params["kappa"])
        self.reach = float(params.get("reach", 0.05))
        self.axis = [float(v) for v in (params.get("axis") or [0.0, 0.0, 1.0])]
        self.centre = params.get("centre")
        self.sign = float(params.get("sign", 1.0))
        # THERMAL NOISE ON THE ROTOR, the fluctuating half of the load's dissipation. The rotor's
        # rotational drag zeta (the placeholder `drag` times the rotor's moment of inertia, or a
        # filament's) is the dissipative half of one Langevin equation whose other half is a random
        # torque of variance 2 zeta kT / dt (fluctuation-dissipation; Berg 2003 treats the rotor
        # so, Mora & Wingreen 2009 give its rotational diffusion). Held constant over one frame
        # and delivered on the same material the stators grip, as a tangential force sharing the
        # lever arms. It adds NO friction: the friction is already the drag; what it adds is the
        # angle's diffusion, D = kT / zeta, and no change of the mean rate. `kT` and `zeta_rot` in
        # sim units; zero (the default) is off.
        self.kT = float(params.get("kT", 0.0))
        self.zeta_rot = float(params.get("zeta_rot", 0.0))
        self.seed = int(params.get("seed", 0))
        self._gen = None
        # THE ANCHOR (E2). `anchor: <set>` names the material each unit is held by -- MotB's own
        # periplasmic domain, and through the grid the scaffold behind it. The equal-and-opposite
        # force of the torque a unit puts on the ring goes INTO that material (tangential, the
        # opposite way, spread over the anchor points within `anchor_reach` of the unit, so that
        # the two sets' torques cancel exactly), and the unit's position RIDES on it: each frame
        # the unit sits at its seeded place plus the mean displacement of the anchor points it was
        # bound to at the first frame. Until now the reaction fell into a wall that was not there
        # and the scaffold was a picture; with this the linkage's stiffness has a second term the
        # scaffold supplies, and a scaffold that gives lets the unit slip back.
        self.anchor = params.get("anchor")
        self.anchor_reach = float(params.get("anchor_reach", self.reach))
        self._anchor_idx = None                    # [n_stator] lists of bound anchor points, bound once
        if self.kappa <= 0.0 or self.reach <= 0.0:
            raise ValueError(f"stator_contact: kappa and reach must be > 0, got {self.kappa}, {self.reach}")
        if self.sign not in (1.0, -1.0):
            raise ValueError(f"stator_contact: sign is +1 (CCW seen from +axis) or -1, got {self.sign}")
        self._cvec = {}
        self.torque = 0.0

    def _const(self, key, values, dev, dtype):
        ck = (key, str(dev), str(dtype))
        if ck not in self._cvec:
            self._cvec[ck] = torch.tensor([float(v) for v in values], device=dev, dtype=dtype)
        return self._cvec[ck]

    def forward(self, H, mask=None):
        p = H.level(self.at)
        if self.stator not in H.levels:
            raise ValueError(f"stator_contact: `stator: {self.stator}` is not a set in this model")
        st = H.level(self.stator)
        if "x" not in st.state_schema._slices:
            raise ValueError(f"stator_contact: set {self.stator!r} declares no state block `x` (the "
                             f"linkage stretch, width 1, integration first_order) for stator_step to feed")
        X = p.get("pos")
        dev, dt = X.device, X.dtype
        V = p.get("vel").to(dt) if "vel" in p.state_schema._slices else torch.zeros_like(X)
        S = st.get("pos").to(dt)
        x_k = st.get("x").to(dt).reshape(-1)
        m_p = getattr(p, "mass", None)
        m_p = torch.ones(p.n, device=dev, dtype=dt) if m_p is None else m_p.to(dt)

        a = self._const("axis", self.axis, dev, dt)
        a = a / a.norm().clamp_min(1e-12)
        c = self._const("centre", self.centre, dev, dt) if self.centre is not None else S.mean(0)
        rel = X - c
        r_perp = rel - (rel @ a)[:, None] * a
        r_len = r_perp.norm(dim=1)
        t_hat = self.sign * torch.cross(a.expand_as(r_perp), r_perp, dim=1) / r_len.clamp_min(1e-12)[:, None]

        live = p.occ > 0
        if mask is not None:
            live = live & mask
        touch = ((torch.cdist(X, S) ** 2) < self.reach ** 2) & live[:, None]     # [N, n_stator]
        T = touch.to(dt)
        n_k = T.sum(0)
        R_k = (T * r_len[:, None]).sum(0)                                      # sum of lever arms per unit
        tau_k = self.kappa * x_k                                               # the linkage's torque
        share = torch.where(R_k > 0, tau_k / R_k.clamp_min(1e-12), torch.zeros_like(tau_k))
        f_mag = (T * share[None, :]).sum(1)                                    # per particle
        if self.kT > 0.0 and self.zeta_rot > 0.0:
            if self._gen is None:
                self._gen = torch.Generator(device=dev); self._gen.manual_seed(self.seed)
            h = float(getattr(H, "dt", 1.0))
            tau_th = math.sqrt(2.0 * self.zeta_rot * self.kT / h) * float(torch.randn(1, generator=self._gen, device=dev))
            gripped = (T.sum(1) > 0)
            R_all = float((r_len * gripped.to(dt)).sum())
            if R_all > 0:
                f_mag = f_mag + gripped.to(dt) * (tau_th / R_all)               # sum r f = tau_th exactly
            self.torque_thermal = tau_th
        acc = (f_mag[:, None] * t_hat) / m_p[:, None]

        # THE RING'S RATE UNDER EACH UNIT, from the velocities of what it grips.
        w_p = (V * t_hat).sum(1) / r_len.clamp_min(1e-12)
        omega_k = torch.where(n_k > 0, (T * w_p[:, None]).sum(0) / n_k.clamp_min(1.0), torch.zeros_like(n_k))
        delivered = torch.where(R_k > 0, tau_k, torch.zeros_like(tau_k))
        for name, val in (("stator_torque", delivered), ("stator_omega", omega_k)):
            if hasattr(st, name):
                getattr(st, name).copy_(val.detach())
            else:
                st.register_buffer(name, val.detach().clone())
        self.torque = float(delivered.sum())
        fr = int(getattr(H, "frame", 0))
        out = {self.at: acc}
        anchor_note = ""
        if self.anchor and self.anchor in H.levels:
            an = H.level(self.anchor)
            Xa = an.get("pos"); ma = getattr(an, "mass", None)
            ma = torch.ones(an.n, device=dev, dtype=dt) if ma is None else ma.to(dt)
            if self._anchor_idx is None:                 # bound once, on the first frame, to what is under each unit
                Ta = (((torch.cdist(Xa, S) ** 2) < self.anchor_reach ** 2) & (an.occ > 0)[:, None])
                self._anchor_T = Ta.to(dt)
                self._anchor_ref = Xa.detach().clone()                         # the anchor points' first positions
                self._stator_ref = S.detach().clone()                          # the units' seeded positions
                self._anchor_idx = True
                na = self._anchor_T.sum(0)
                print(f"[stator_contact] anchor {self.anchor!r}: {int((self._anchor_T.sum(1) > 0).sum())} points bound, "
                      f"{int(na.min())}-{int(na.max())} per unit" + (";  A UNIT HOLDS NOTHING" if int(na.min()) == 0 else ""), flush=True)
            Ta = self._anchor_T
            rel_a = Xa - c
            rpa = rel_a - (rel_a @ a)[:, None] * a
            rla = rpa.norm(dim=1)
            ta = self.sign * torch.cross(a.expand_as(rpa), rpa, dim=1) / rla.clamp_min(1e-12)[:, None]
            Ra = (Ta * rla[:, None]).sum(0)                                    # lever arms of the anchor points
            share_a = torch.where(Ra > 0, -delivered / Ra.clamp_min(1e-12), torch.zeros_like(delivered))
            fa = (Ta * share_a[None, :]).sum(1)                                # sum r f = -tau_k: the reaction
            out[self.anchor] = (fa[:, None] * ta) / ma[:, None]
            # the unit rides on its anchor: seeded place + the mean displacement of its bound points
            na = Ta.sum(0).clamp_min(1.0)
            disp = (Ta.T @ (Xa - self._anchor_ref)) / na[:, None]             # [n_stator, 3]
            new_S = self._stator_ref + disp
            a0, b0 = st.state_schema["pos"]
            sts = st.state.clone(); sts[:, a0:b0] = new_S.to(st.state.dtype); st.state = sts
            anchor_note = f"; reaction on {self.anchor} {float(-(rla * fa).sum()):.4e}, units moved {float(disp.norm(dim=1).mean()) * 1e3:.3f}e-3 world"
        if fr in (1, 5, 20, 50, 200, 500, 800, 1000, 1400):
            lo, hi = int(n_k.min()), int(n_k.max())
            print(f"[stator_contact f{fr}] {int(S.shape[0])} units, {int((f_mag != 0).sum())} particles gripped "
                  f"({lo}-{hi} per unit); torque delivered {self.torque:.4e} = kappa sum x "
                  f"(mean stretch {float(x_k.mean()):.4f} rad); ring under the units {float(omega_k.mean()):.4f} rad/s"
                  + (";  A UNIT GRIPS NOTHING" if lo == 0 else "") + anchor_note, flush=True)
        return out


@register_operator("stator_step", family="motility", set="particle", kind="lateral",
                   equation=r"""$$W_k=\tau_k\,\delta,\qquad r^{+}_k=r_0\,e^{\theta(\epsilon-W_k, title="Ion-driven stepping of a stator unit")/k_BT},\qquad r^{-}_k=r_0\,e^{-(1-\theta)(\epsilon-W_k)/k_BT}$$""")
class StatorStep(Lateral):
    """The ion-driven stepping of each stator unit, feeding its linkage's stretch.

    stator -> stator: reads the stretch block `x` and the buffers `stator_torque`, `stator_omega`
    that `stator_contact` wrote; integrates `x` (`INTEGRAND = "x"`, first order) and, if the set
    declares it, `ions` (transits, signed).

    One cycle of a unit is a chemical step -- an ion binds, at a load-independent rate k_c -- and
    a mechanical step, the power stroke that turns MotA by one step of angle `delta` and stretches
    the linkage by the same, against the work that stretch costs at the present torque:

        W_k       = tau_k delta                        work of one step against the linkage
        r+_k      = r0 exp( theta (eps - W_k) / kT )   forward stroke
        r-_k      = r0 exp( -(1 - theta) (eps - W_k) / kT )   back-stroke, the ion returned
        nu_k      = (r+ - r-) / (1 + (r+ + r-)/k_c)     steps per unit time: the chain's own mean
                                                        (bind, then stroke forward or back, then bind again)
        dx_k/dt   = delta nu_k - omega_k               fed by steps, relaxed by the ring's motion

    `eps` is the free energy of one transit, q e Delta-psi (two protons across 150 mV is 48 pN nm),
    `kT` the thermal energy, both in sim energy units (sim force times world length); `theta` in
    [0, 1] is where the transition state sits along the stroke (Meacci & Tu 2009's load
    distribution; 0.1 puts nearly all the load on the back-stroke, which is what gives a plateau);
    `r0` and `k_c` are rates in 1/sim-s; `delta` the step in radians (2 pi / 26 for the 26 steps
    per revolution of Sowa et al. 2005); `ions` the transits per step (q). With `stochastic: true`
    the two halves are drawn as Bernoulli trials per frame -- the unit is then either waiting for
    an ion or stroking, and a run shows steps -- otherwise the mean rate above is integrated.

    WHAT EMERGES AND WHAT IS STILL GIVEN. Stall torque per unit, eps / delta, is the ion's free
    energy over the step and is not a parameter; the plateau and the knee of the torque-speed
    curve come from theta, r0 and k_c against whatever load the rotor carries. Given, and stated:
    the zero-load rate (r0, k_c), the transition-state position theta, the linkage stiffness in
    `stator_contact`, and the membrane potential in eps, which does not yet deplete with the flux
    it drives.

    Reference: Meacci, G. & Tu, Y. (2009). PNAS 106:3746-3751; Sowa, Y. et al. (2005). Nature
    437:916-919 (26 steps per revolution); Mandadapu, K.K. et al. (2015). PNAS 112:E4381 (the
    power stroke); Chen, X. & Berg, H.C. (2000). Biophys. J. 78:1036-1041 (the curve to be earned).
    """

    EMIT = "velocity"
    INTEGRAND = "x"
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    INPUTS = ["stator"]
    OUTPUTS = ["stator"]
    READS = ["x"]
    WRITES = ["x", "ions"]
    REQUIRES_PARAMS = ["r0", "k_c", "eps", "kT", "delta"]
    MAY_MUTATE_INTEGRATED_STATE = True                    # writes `flux` in place, a rate, not a state it integrates
    MECHANISM_TAGS = ["rotary_motor", "stator_unit", "ion_motive_force", "stepping", "power_stroke"]
    PARAM_ROLES = {"r0": "power_stroke_attempt_rate_per_sim_s", "k_c": "ion_binding_rate_per_sim_s",
                   "eps": "free_energy_per_transit_sim_energy", "kT": "thermal_energy_sim_energy",
                   "delta": "step_angle_radians", "theta": "transition_state_position_0_to_1",
                   "ions": "transits_per_step", "stochastic": "bernoulli_steps_per_frame",
                   "seed": "rng_seed"}
    # UNITS: energies are force x length (the paper derives energy from the three scales, never
    # declared apart); rates are 1/time; a transit is a count; the potential block is e psi, an
    # energy per elementary charge, i.e. an energy since the charge is a count.
    PARAM_UNITS = {"r0": "rate", "k_c": "rate", "eps": "energy", "kT": "energy", "delta": "1",
                   "theta": "1", "ions": "count", "pool_alpha": "time"}
    BLOCK_UNITS = {"x": "1", "ions": "count", "psi": "voltage", "flux": "current"}
    REFERENCE = ("Meacci, G. & Tu, Y. (2009). PNAS 106:3746; Sowa, Y. et al. (2005). Nature 437:916; "
                 "Mandadapu, K.K. et al. (2015). PNAS 112:E4381; Chen, X. & Berg, H.C. (2000). "
                 "Biophys. J. 78:1036.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "stator_unit")
        self.r0 = float(params["r0"])
        self.k_c = float(params["k_c"])
        self.eps = float(params["eps"])
        self.kT = float(params["kT"])
        self.delta = float(params["delta"])
        self.theta = float(params.get("theta", 0.1))
        self.q = float(params.get("ions", 2.0))
        self.stochastic = bool(params.get("stochastic", False))
        self.seed = int(params.get("seed", 0))
        self._gen = None
        # E3 -- THE PROTON POOL AS A NUMBER, NOT A FIELD. The buffered pool a unit draws from is
        # depleted around it by its own current I_k = q nu_k; the steady depletion of a point sink
        # in a medium of diffusivity D at distance r is I / (4 pi D r), so the pool at the channel's
        # mouth is c0 (1 - alpha I_k) with alpha = 1 / (4 pi D r c0), and the binding rate follows it:
        # k_c,eff = k_c (1 - alpha q nu_k). At D ~ 1e9 nm^2/s, c0 = 1 mM, r = 2 nm and 5,200 protons/s
        # the depletion is 3e-4. A grid field of the pool at this box (voxels of 10 nm^3 holding
        # 0.006 protons, a sink of 0.05 protons per frame) cannot represent that number and a
        # Green's function can; this is the spatial term, computed where it acts. `pool_alpha` is
        # alpha in sim time per proton (the generator converts); zero (the default) is off.
        self.pool_alpha = float(params.get("pool_alpha", 0.0))
        if not (0.0 <= self.theta <= 1.0):
            raise ValueError(f"stator_step: theta is in [0, 1], got {self.theta}")
        if min(self.r0, self.k_c, self.kT, self.delta) <= 0.0:
            raise ValueError("stator_step: r0, k_c, kT and delta must be > 0")

    def forward(self, H, mask=None):
        st = H.level(self.at)
        x = st.get("x").reshape(-1)
        dev, dt = x.device, x.dtype
        n = x.shape[0]
        tau = getattr(st, "stator_torque", None)
        omega = getattr(st, "stator_omega", None)
        tau = torch.zeros(n, device=dev, dtype=dt) if tau is None else tau.to(dt)
        omega = torch.zeros(n, device=dev, dtype=dt) if omega is None else omega.to(dt)
        W = tau * self.delta
        # THE POTENTIAL FROM THE CELL WHEN THERE IS ONE: a `psi` block on the stator set, written by
        # `broadcast` from the cell's `membrane_potential`, makes eps = q e psi a state; without it
        # eps is the constant the spec gave.
        eps = self.q * st.get("psi")[:, 0].to(dt) if "psi" in st.state_schema._slices else \
            torch.full((n,), self.eps, device=dev, dtype=dt)
        g = (eps - W) / self.kT
        r_f = self.r0 * torch.exp(self.theta * g)
        r_b = self.r0 * torch.exp(-(1.0 - self.theta) * g)
        h = float(getattr(H, "dt", 1.0))
        if self.stochastic:
            if self._gen is None:
                self._gen = torch.Generator(device=dev); self._gen.manual_seed(self.seed)
            if not hasattr(st, "stator_bound"):
                st.register_buffer("stator_bound", torch.zeros(n, device=dev, dtype=dt))
            b = st.stator_bound
            # THE CHAIN'S CLOCK IS FINER THAN THE FRAME. One Bernoulli trial per frame at k_c h = 0.1
            # and r+ h = 0.2 is not a Poisson process: a trial that cannot fire twice, nor fire and
            # bind in the same frame, lengthens the mean cycle by the frame's own length, and the
            # first stochastic run (exp_02 D10) turned at 87 Hz where the mean-rate motor turns at
            # 100 with the same parameters. The frame is cut into m sub-intervals so that every
            # probability stays below 2%, and the trials walk them in order.
            p_max = max(float(self.k_c * h), float((r_f * h).max()), float((r_b * h).clamp(max=1e3).max()))
            m = int(max(1, min(200, math.ceil(p_max / 0.02))))
            hs = h / m
            steps = torch.zeros(n, device=dev, dtype=dt)
            for _ in range(m):
                u = torch.rand(3, n, generator=self._gen, device=dev, dtype=dt)
                bind = (b < 0.5) & (u[0] < min(self.k_c * hs, 1.0))
                fwd = (b > 0.5) & (u[1] < (r_f * hs).clamp(max=1.0))
                bck = (b > 0.5) & ~fwd & (u[2] < (r_b * hs).clamp(max=1.0))
                steps = steps + fwd.to(dt) - bck.to(dt)
                b.copy_(torch.where(bind, torch.ones_like(b), torch.where(fwd | bck, torch.zeros_like(b), b)))
            nu = steps / h
        else:
            mech = r_f - r_b
            k_c = self.k_c
            if self.pool_alpha > 0.0:                    # the pool at the mouth, depleted by last frame's current
                nu_prev = st.get("flux")[:, 0].to(dt) / self.q if "flux" in st.state_schema._slices else torch.zeros(n, device=dev, dtype=dt)
                depletion = (self.pool_alpha * self.q * nu_prev).clamp(0.0, 0.99)
                k_c = self.k_c * (1.0 - depletion)
                self.depletion = float(depletion.mean())
            # THE MEAN OF THE CHAIN ITSELF, NOT A GUESS AT IT. The first mean-rate law, nu = 1/(1/k_c +
            # 1/(r+ - r-)), treated the stroke as one step at the net rate; in the chain a bound unit
            # strokes forward with probability r+/(r+ + r-) after a mean wait of 1/(r+ + r-), or
            # back -- and either way it must bind an ion again. Its mean advance per cycle is
            # (r+ - r-)/(r+ + r-) steps in a time 1/k_c + 1/(r+ + r-), so
            #     nu = (r+ - r-) / (1 + (r+ + r-) / k_c),
            # which is what the stochastic runs (D10, E1s) measured against the old law: 75-82% of
            # its rate, a discrepancy that was the law's, not the chain's (2026-09-24 00:20). One
            # kinetics, two integrations: the mean-rate motor now IS the chain's average.
            nu = mech / (1.0 + (r_f + r_b) / k_c)
        dx = self.delta * nu - omega
        if "flux" in st.state_schema._slices:                    # transits per sim s, for `aggregate` -> the cell
            a, b = st.state_schema["flux"]
            stt = st.state.clone(); stt[:, a:b] = (self.q * nu)[:, None]; st.state = stt
        fr = int(getattr(H, "frame", 0))
        if fr in (1, 5, 20, 50, 200, 500, 800, 1000, 1400):
            print(f"[stator_step f{fr}] W/eps {float((W / eps).mean()):.3f} (eps {float(eps.mean()):.3e}); r+ {float(r_f.mean()):.3e}, "
                  f"r- {float(r_b.mean()):.3e}, k_c {self.k_c:.3e} -> {float(nu.mean()):.3e} steps/s = "
                  f"{float(nu.mean()) * self.delta:.4f} rad/s fed; ring {float(omega.mean()):.4f} rad/s; "
                  f"stretch {float(x.mean()):.4f} rad = {float(x.mean()) / self.delta:.2f} steps"
                  + (f"; pool depletion at the mouth {getattr(self, 'depletion', 0.0):.2e}" if self.pool_alpha > 0 else ""), flush=True)
        out = {(self.at, "x"): dx[:, None]}
        if "ions" in st.state_schema._slices:
            out[(self.at, "ions")] = (self.q * nu)[:, None]
        return out


# ============================================================================================
# THE ION ECONOMY: the membrane potential as an outcome, not a constant
# ============================================================================================
#
# `stator_step` above takes the free energy per transit, eps = q e psi, as a number. Here psi is a
# state of the CELL: a membrane capacitor charged by respiration and discharged by the stators
# and a leak, the zero mode of Poisson-Nernst-Planck at a scale where the Debye length is 1 nm
# and protons cross the box in 20 us against 380 us between one stator's steps. Three operators
# and the two hierarchy families the paper names: `membrane_potential` is the cell's own
# (Lateral) law, `aggregate` sums the stators' transits up the containment map (Sigma_pi) into
# the cell's motor current, `broadcast` hands psi down it (pi^*) so every stator steps against
# the potential the cell actually has. The stator set is therefore a CHILD of the cell.


@register_operator("aggregate", family="hierarchy", set="particle", kind="aggregate",
                   equation=r"""$$\mathbf x_P \;=\; \frac{\sum_{i\in P} o_i\,\mathbf x_i}{\sum_{i\in P} o_i}$$""", title="Sum the children onto their parent")
class AggregateBlock(Aggregate):
    """Sigma_pi: a parent's block as the sum (or mean) of its children's, along `Level.parent`.

    child -> parent: reads the children's `block`, writes the parent's `target` block in place
    (`target`, not `to`: to the schema `to`/`from` name FIELDS, as in mpm_scatter's grid).

        P_j = reduce_{i : pi(i) = j} c_i          reduce = sum | mean, over live children

    The generic lift of a per-child quantity to the thing that contains it -- the motor current
    of a cell from the flux of its stator units, the load of a tissue from its cells. Writes
    state rather than returning a delta because a sum is not dynamics: it is true the instant its
    terms are, and an integrator would lag it by a frame.

    Reference: Plexus (this work), section 2 of the paper: exactly two families cross a level.
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    INPUTS = ["child"]; OUTPUTS = ["parent"]; READS = []; WRITES = []
    REQUIRES_PARAMS = ["block", "target"]
    MECHANISM_TAGS = ["hierarchy", "aggregate"]
    PARAM_ROLES = {"block": "the_children_block_read", "target": "the_parent_block_written", "reduce": "sum_or_mean"}
    REFERENCE = "Plexus (this work)."
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.block = str(params["block"])
        self.to = str(params["target"])
        self.reduce = str(params.get("reduce", "sum")).lower()
        if self.reduce not in ("sum", "mean"):
            raise ValueError(f"aggregate: reduce is sum or mean, got {self.reduce!r}")

    def forward(self, H, mask=None):
        c = H.level(self.at)
        pname = getattr(c, "parent_name", None)
        if not pname or getattr(c, "parent", None) is None:
            raise ValueError(f"aggregate: set {self.at!r} declares no parent (needs `parent:` and `per_parent:`)")
        p = H.level(pname)
        idx = c.parent.reshape(-1).long()
        v = c.get(self.block).to(p.state.dtype)
        live = (c.occ > 0).to(v.dtype)[:, None]
        out = torch.zeros(p.n, v.shape[1], device=v.device, dtype=v.dtype).index_add_(0, idx, v * live)
        if self.reduce == "mean":
            cnt = torch.zeros(p.n, 1, device=v.device, dtype=v.dtype).index_add_(0, idx, live)
            out = out / cnt.clamp_min(1.0)
        a, b = p.state_schema[self.to]
        st = p.state.clone(); st[:, a:b] = out[:, : b - a]; p.state = st
        return {}


@register_operator("broadcast", family="hierarchy", set="particle", kind="broadcast",
                   equation=r"""$$\dot{\mathbf x}_i \;=\; k\,\big(\mathbf x_{P(i, title="Hand the parent's value to its children")}-\mathbf x_i\big)$$""")
class BroadcastBlock(Broadcast):
    """pi^*: a parent's block handed down to every child, along `Level.parent`.

    parent -> child: reads the parent's `source` block, writes the children's `block` in place.

        c_i = P_{pi(i)}

    The generic lift the other way: the membrane potential of a cell to each of its stator units,
    a tissue's signal to its cells. In place for the same reason `aggregate` is.

    Reference: Plexus (this work).
    """
    EMIT = None
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    INPUTS = ["parent"]; OUTPUTS = ["child"]; READS = []; WRITES = []
    REQUIRES_PARAMS = ["block", "source"]
    MECHANISM_TAGS = ["hierarchy", "broadcast"]
    PARAM_ROLES = {"block": "the_children_block_written", "source": "the_parent_block_read"}
    REFERENCE = "Plexus (this work)."
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.block = str(params["block"])
        self.src = str(params["source"])

    def forward(self, H, mask=None):
        c = H.level(self.at)
        pname = getattr(c, "parent_name", None)
        if not pname or getattr(c, "parent", None) is None:
            raise ValueError(f"broadcast: set {self.at!r} declares no parent (needs `parent:` and `per_parent:`)")
        p = H.level(pname)
        idx = c.parent.reshape(-1).long()
        v = p.get(self.src)[idx].to(c.state.dtype)
        a, b = c.state_schema[self.block]
        st = c.state.clone(); st[:, a:b] = v[:, : b - a]; c.state = st
        return {}


@register_operator("membrane_potential", family="metabolism", set="particle", kind="lateral",
                   equation=r"""$$C\frac{d\psi}{dt}=J_{\mathrm{pump}}(\psi)-J_{\mathrm{motor}}-g\psi,\qquad J_{\mathrm{pump}}=J_{\max}\max\!\left(0,\,1-\frac{\psi}{\psi_{\mathrm{rev}}}\right)$$""")
class MembranePotential(Lateral):
    """The cell's membrane potential as a capacitor charged by respiration and drained by its
    motors and a leak -- the zero mode of Poisson-Nernst-Planck.

    cell -> cell: reads the cell's `psi` (e psi, sim energy per elementary charge) and its motor
    current block `flux` (charges per sim s, written by `aggregate`); integrates `psi`
    (`INTEGRAND = "psi"`) and, if declared, the proton counts `h_peri` and `h_cyto`.

        C dpsi/dt = J_pump(psi) - J_motor - g psi
        J_pump    = J_max max(0, 1 - psi / psi_rev)      respiration, stalling at psi_rev
        dh_peri/dt = J_pump - J_motor - g psi,  dh_cyto/dt = -dh_peri/dt

    The counts carry the SAME three currents as the capacitor, the leak included (charge that
    returns through everything that is not a stator), so that C dpsi = e dh_peri holds to the
    digit; a first version counted only pump and motor and the counts rose while the potential
    fell (exp_02 step 0036: +6,557 protons out against -4.4 mV -- a bookkeeping error, fixed).

    `capacitance` C is in charges per unit of psi (the membrane's C_m A / e^2 in these units),
    `pump_max` and `leak` in charges per sim s (per unit psi for the leak), `pump_rev` the
    potential at which the pump stalls, `psi0` the potential written once at the first frame
    (the cell starts charged), `h0` the two counts' start. On the first frame the operator writes
    psi0 and h0 in place; from then on it only integrates.

    WHY AN ODE AND NOT THE PDE. Nernst-Planck plus Poisson in a 140 nm box collapses to this law:
    the cytoplasm and periplasm are each equipotential (Debye length 1 nm), the whole drop sits
    across the membrane, and protons cross the box in 20 us against 380 us between one stator's
    steps, so both sides are well mixed. The spatial version is the refinement for the day a
    stator depletes its own neighbourhood, and the number that decides it is the flux, 88,000
    protons/s for 17 units at 100 Hz against a diffusive supply orders of magnitude larger.

    Reference: Berg, H.C. (2003). Annu. Rev. Biochem. 72:19-54 (proton-driven); Fung, D.C. & Berg,
    H.C. (1995). Nature 375:809-812 (speed proportional to the potential); Gabel, C.V. & Berg, H.C.
    (2003). PNAS 100:8748-8751; Plexus (this work).
    """
    EMIT = "velocity"
    INTEGRAND = "psi"
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = False
    INPUTS = ["cell"]; OUTPUTS = ["cell"]; READS = ["psi"]; WRITES = ["psi", "h_peri", "h_cyto"]
    REQUIRES_PARAMS = ["capacitance", "pump_max", "pump_rev"]
    MECHANISM_TAGS = ["ion_motive_force", "membrane_capacitor", "respiration", "metabolism"]
    PARAM_ROLES = {"capacitance": "charges_per_unit_psi", "pump_max": "respiration_charges_per_sim_s",
                   "pump_rev": "psi_at_which_the_pump_stalls", "leak": "charges_per_sim_s_per_unit_psi",
                   "psi0": "initial_psi", "h0": "initial_proton_counts", "flux": "the_motor_current_block",
                   "shunt": "resistor_switched_across_the_membrane", "shunt_from": "the_frame_it_switches_on"}
    # UNITS: psi is carried as e psi, an energy per elementary charge -- `voltage` in units.py, F L
    # through the constant e, reported in mV; the capacitance is charges per unit of e psi, 1/(F L);
    # currents are `current`, charges per time, reported in fA;
    # the leak is a current per unit e psi. Counts are counts, and may be negative: `h_peri` is
    # the NET number of protons that left the cytoplasm since frame 0 (pump - motor - leak), so
    # a cell whose potential falls has a negative h_peri, equal to C dpsi / e to the digit.
    PARAM_UNITS = {"capacitance": "1/(F*L)", "pump_max": "current", "pump_rev": "voltage", "leak": "1/(T*F*L)",
                   "psi0": "voltage", "h0": "count", "shunt": "1/(T*F*L)", "shunt_from": "count"}
    BLOCK_UNITS = {"psi": "voltage", "h_peri": "count", "h_cyto": "count", "j_motor": "current"}
    REFERENCE = ("Berg, H.C. (2003). Annu. Rev. Biochem. 72:19; Fung & Berg (1995). Nature 375:809; "
                 "Gabel & Berg (2003). PNAS 100:8748; Plexus (this work).")
    MAY_MUTATE_INTEGRATED_STATE = True

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.C = float(params["capacitance"])
        self.J_max = float(params["pump_max"])
        self.psi_rev = float(params["pump_rev"])
        self.g = float(params.get("leak", 0.0))
        self.psi0 = float(params.get("psi0", 0.0))
        self.h0 = [float(v) for v in (params.get("h0") or [0.0, 0.0])]
        self.flux = str(params.get("flux", "j_motor"))
        # THE RESISTOR, the sanity check the human asked for (2026-09-23): a shunt conductance
        # switched across the membrane at frame `shunt_from`, in the leak's units. A load that is
        # not the motor must pull psi down and the motor must slow with it -- proportionally, since
        # near stall its torque is proportional to psi -- or the coupling is not what it claims.
        self.shunt = float(params.get("shunt", 0.0))
        self.shunt_from = int(params.get("shunt_from", 0))
        self._started = False
        if self.C <= 0 or self.psi_rev <= 0:
            raise ValueError("membrane_potential: capacitance and pump_rev must be > 0")

    def forward(self, H, mask=None):
        c = H.level(self.at)
        if not self._started:
            st = c.state.clone()
            a, b = c.state_schema["psi"]; st[:, a:b] = self.psi0
            for k, v0 in (("h_peri", self.h0[0]), ("h_cyto", self.h0[1])):
                if k in c.state_schema._slices:
                    a, b = c.state_schema[k]; st[:, a:b] = v0
            c.state = st
            self._started = True
        psi = c.get("psi")[:, 0]
        J_m = c.get(self.flux)[:, 0] if self.flux in c.state_schema._slices else torch.zeros_like(psi)
        J_p = self.J_max * (1.0 - psi / self.psi_rev).clamp_min(0.0)
        fr = int(getattr(H, "frame", 0))
        g = self.g + (self.shunt if (self.shunt > 0.0 and fr >= self.shunt_from) else 0.0)
        net = J_p - J_m - g * psi                                # every current the capacitor sees
        dpsi = net / self.C
        if fr in (1, 5, 20, 50, 200, 500, 800, 1000, 1400):
            print(f"[membrane_potential f{fr}] psi {float(psi.mean()):.4e} (sim e psi); pump {float(J_p.mean()):.4e}, "
                  f"motor {float(J_m.mean()):.4e}, leak+shunt {float((g * psi).mean()):.4e} charges/s"
                  f"{' (shunt ON)' if g > self.g else ''}; dpsi/dt {float(dpsi.mean()):.3e}", flush=True)
        out = {(self.at, "psi"): dpsi[:, None]}
        if "h_peri" in c.state_schema._slices:
            out[(self.at, "h_peri")] = net[:, None]
        if "h_cyto" in c.state_schema._slices:
            out[(self.at, "h_cyto")] = (-net)[:, None]
        return out


@register_operator("stator_membrane", family="motility", set="particle", kind="exchange",
                   equation=r"""$$j^{\mathrm{in}}_p=\sum_k\frac{\Phi_k}{\lvert M_k\rvert},\qquad M_k=\{p:\lVert\mathbf x_p-\mathbf S_k\rVert<\text{reach}\}$$""", title="Ion current into the membrane")
class StatorMembrane(Exchange):
    """Where the motor's current enters: each stator unit deposits its transit rate on the
    membrane points it stands in.

    stator -> membrane: reads the stator set's positions and its `flux` block (transits per sim
    time, written by `stator_step`); writes the membrane's `j_in` block in place -- each unit's
    flux shared equally over the membrane points within `reach` of it, zero elsewhere.

        j_in,p = sum_k  flux_k / |M_k|  for p in M_k = { p : |x_p - S_k| < reach }

    THE MEMBRANE IS A PLACE, NOT A FIELD WITH ITS OWN POTENTIAL. A patch of membrane the size of
    one point (a few nm^2) holds half an elementary charge per volt, so a per-point potential
    would swing by volts at every transit; the aqueous sides equalise it in nanoseconds, and the
    potential is uniform over the sheet by physics (Debye length ~1 nm). So psi reaches the sheet
    by `broadcast` from the cell and is the same everywhere on it, and what this operator puts
    on the sheet is the one thing that IS local: the current, seventeen spots of inflow under
    the stators against the pump's uniform outflow. A unit standing on no membrane point deposits
    nothing and says so.

    Reference: Plexus (this work); the membrane's electrostatics after Hille, Ion Channels of
    Excitable Membranes, ch. 1 (capacitance 1 uF/cm^2, Debye screening).
    """
    EMIT = None
    SUPPORTED_DIMS = [3]
    DIFFERENTIABLE = False
    INPUTS = ["stator", "membrane"]; OUTPUTS = ["membrane"]; READS = ["pos", "flux"]; WRITES = ["j_in"]
    REQUIRES_PARAMS = ["stator"]
    MECHANISM_TAGS = ["rotary_motor", "stator_unit", "ion_motive_force", "membrane"]
    PARAM_ROLES = {"stator": "the_set_whose_elements_are_the_stator_units",
                   "reach": "the_membrane_points_a_unit_stands_in", "block": "the_membrane_block_written"}
    PARAM_UNITS = {"reach": "length"}
    BLOCK_UNITS = {"j_in": "current", "flux": "current", "j_avg": "current"}
    PARAM_UNITS = {"reach": "length", "tau": "count"}
    MAY_MUTATE_INTEGRATED_STATE = True
    REFERENCE = "Plexus (this work); Hille, B. Ion Channels of Excitable Membranes (2001), ch. 1."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "membrane")
        self.stator = str(params["stator"])
        self.reach = float(params.get("reach", 0.05))
        self.block = str(params.get("block", "j_in"))
        # `tau` (frames): an exponential running mean of the inflow into a second block `avg`
        # (`j_avg` unless said), so that a sheet fed by stochastic strokes -- each spot swinging
        # between forward and back transits within a frame -- still shows WHERE the protons enter
        # on average. Zero (the default) writes no average.
        self.tau = float(params.get("tau", 0.0))
        self.avg = str(params.get("avg", "j_avg"))

    def forward(self, H, mask=None):
        m = H.level(self.at)
        st = H.level(self.stator)
        X = m.get("pos"); dev, dt = X.device, X.dtype
        S = st.get("pos").to(dt)
        flux = st.get("flux")[:, 0].to(dt) if "flux" in st.state_schema._slices else torch.zeros(S.shape[0], device=dev, dtype=dt)
        T = (((torch.cdist(X, S) ** 2) < self.reach ** 2) & (m.occ > 0)[:, None]).to(dt)       # [N, n_stator]
        n_k = T.sum(0)
        share = torch.where(n_k > 0, flux / n_k.clamp_min(1.0), torch.zeros_like(flux))
        j = (T * share[None, :]).sum(1)
        a, b = m.state_schema[self.block]
        stt = m.state.clone(); stt[:, a:b] = j[:, None]
        if self.tau > 0.0 and self.avg in m.state_schema._slices:
            a2, b2 = m.state_schema[self.avg]
            w = 1.0 - math.exp(-1.0 / self.tau)
            stt[:, a2:b2] = stt[:, a2:b2] + w * (j[:, None] - stt[:, a2:b2])
        m.state = stt
        fr = int(getattr(H, "frame", 0))
        if fr in (1, 50, 500, 1400):
            lo, hi = int(n_k.min()), int(n_k.max())
            print(f"[stator_membrane f{fr}] {int(S.shape[0])} units on {int((j != 0).sum())} membrane points "
                  f"({lo}-{hi} per unit); current entering {float(j.sum()):.4e} charges per sim s"
                  + (";  A UNIT STANDS ON NO MEMBRANE" if lo == 0 else ""), flush=True)
        return {}

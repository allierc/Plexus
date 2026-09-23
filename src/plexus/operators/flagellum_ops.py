"""The bacterial flagellar motor: stator units pushing a rotor.

THE MOTOR IS NOT AN OPERATOR THAT SAYS "ROTATE". builder/instruction.md section 0b: the torque
comes from stator units acting on a rotor, and the rotation rate has to COME OUT of that torque
against whatever load the rotor carries. So the one contract in this module is the push itself --
each stator unit, an element of its own set, applies a tangential force on the rotor material
within its reach -- and nothing here knows what speed that produces.

One operator, for now:

    stator_push      exchange   each stator unit pushes the rotor material in its reach, tangentially

That push is a stated reduction -- a constant force per unit, not the product of ion flow -- and
the operators that would make the torque emerge (a stepping stator with its own phase driven by
the proton motive force, an elastic contact to a FliG protomer, the switch) are the next thing
this module takes. A helix seed and a resistive-force helix drive lived here for one day
(builder/exp_02_bacterium steps 0011-0041) and were removed with those steps on 2026-09-23: the
drive propelled its cell by a fitted drag law rather than by the water, and the seed had no user
left once the drive was gone. The filament as a 1D mesh, the drag that couples it to water and
the cell body live in `rod_ops` and `motion_ops` and are reused, not restated -- a merge of
`rod_ops` and this module into one cilium/flagellum module is under discussion with the session
that owns `rod_ops`.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("stator_push", family="motility", set="particle", kind="exchange")
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

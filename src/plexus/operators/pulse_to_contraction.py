"""pulse_to_contraction -- the FORCE half: activation field -> per-particle MPM force.

Reads the activation field a(x,t) (the `from:` field) and converts its gradient into
a per-particle body force, RETURNED as a particle delta. The engine sums it (with any
`mpm_drag` delta) into H.delta(mpm_particle), which the MLS-MPM `p2g` scatter consumes
as the body force (`p2g`: a_ext += H.delta(particle)) -- the per-particle counterpart
of the parent-delta path that carries `gravity`. For a Gaussian activation bump grad(a)
points toward the centre, so

    F_i = sign * amplitude * grad(a)(x_i)          sign = +1 (inward) / -1 (outward)

contracts (mode: inward) or expands (mode: outward) the sheet. It owns only the
mechanical mapping -- not WHEN (`pacemaker`) nor WHERE (`pulse_stimulus`).

`kind=exchange` (field -> set); `EMIT=None`, so the engine never integrates the
particle set (g2p owns advection) -- the force enters mechanics only through the MPM
substep. The engine's `zero_delta` resets the delta each outer tick, and the substep
loop reads that constant body force on each of its iterations.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("pulse_to_contraction", level="particle", kind="exchange")
class PulseToContraction(Exchange):
    EMIT = "mpm_acceleration"           # a body accel the MPM substep consumes as a_ext, not engine-integrated
    REQUIRES_PARAMS = ["from"]                # the activation field to read
    MECHANISM_TAGS = ["active_contraction", "field_gradient_force", "directed_active_stress"]
    PARAM_ROLES = {"amplitude": "contraction_strength", "mode": "gradient_or_directional"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.amplitude = float(params.get("amplitude", 50.0))
        self.channel = int(params.get("channel", 0))
        self.mode = str(params.get("mode", "inward"))
        # gradient modes (inward/outward): direction = +/- grad(activation). directional:
        # direction = a unit-vector field, magnitude = the (uniform) activation value.
        self.sign = {"inward": 1.0, "outward": -1.0}.get(self.mode, 1.0)
        self.direction_from = params.get("direction_from")
        if self.mode == "directional" and self.direction_from is None:
            raise ValueError("pulse_to_contraction mode: directional needs `direction_from:` "
                             "(a vector_grid field giving the contraction direction)")
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]

        if self.mode == "directional":
            # F_i = amplitude * a(x_i) * d(x_i): uniform activation sets WHEN/how much,
            # the vector field sets WHERE to push (the active-stress orientation map).
            a = fld.sample(pos, self.channel)                             # [N] activation
            d = H.fields[self.direction_from].sample(pos)                 # [N, 2] direction
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            acc = self.amplitude * a[:, None] * d
        else:
            # gradient mode: direction = +/- grad(activation), sampled at each particle.
            grad = fld.grad_at(pos, self.channel, periodic=getattr(H, "periodic", False))  # [N, 2]
            acc = self.sign * self.amplitude * grad                       # inward for sign>0

        acc = acc * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # return a per-particle force delta; the engine sums it (with mpm_drag's) into
        # H.delta(mpm_particle), which p2g consumes as the MPM body force. EMIT=None,
        # so the engine never integrates the particle set (g2p owns advection).
        return {self.at: acc}

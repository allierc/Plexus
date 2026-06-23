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

`kind=exchange` (field -> set); `PREDICTION=None`, so the engine never integrates the
particle set (g2p owns advection) -- the force enters mechanics only through the MPM
substep. The engine's `zero_delta` resets the delta each outer tick, and the substep
loop reads that constant body force on each of its iterations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fnn

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("pulse_to_contraction", level="particle", kind="exchange")
class PulseToContraction(Exchange):
    PREDICTION = None                         # force is consumed by the MPM substep, not engine-integrated
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

    def _sample(self, field_grid, pos, W):
        """Bilinear-sample a `[C, nx, ny]` field at particle positions -> `[N, C]`."""
        gxn = (pos[:, 0] / W) * 2 - 1
        gyn = (pos[:, 1] / 1.0) * 2 - 1
        grid = torch.stack([gyn, gxn], -1)[None, None]              # grid_sample expects (x=ny, y=nx)
        return Fnn.grid_sample(field_grid[None], grid, mode="bilinear",
                               padding_mode="border", align_corners=True)[0, :, 0].t()

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        W = float(getattr(H, "world_width", 1.0))

        if self.mode == "directional":
            # F_i = amplitude * a(x_i) * d(x_i): uniform activation sets WHEN/how much,
            # the vector field sets WHERE to push (the active-stress orientation map).
            a = self._sample(fld.grid[self.channel:self.channel + 1], pos, W)[:, 0]   # [N]
            d = self._sample(H.fields[self.direction_from].grid, pos, W)              # [N, 2]
            d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-9)
            acc = self.amplitude * a[:, None] * d
        else:
            # gradient mode: direction = +/- grad(activation), bilinear-sampled (chemotaxis idiom).
            g = fld.grid[self.channel]                                  # [nx, ny]
            pad = "circular" if getattr(H, "periodic", False) else "replicate"
            gp = Fnn.pad(g[None, None], (1, 1, 1, 1), mode=pad)[0, 0]
            gx = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5 * (fld.nx / W)    # d a/d x
            gy = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5 * fld.ny          # d a/d y
            grad = torch.stack([gx, gy], 0)                            # [2, nx, ny]
            acc = self.sign * self.amplitude * self._sample(grad, pos, W)  # inward for sign>0

        acc = acc * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        # return a per-particle force delta; the engine sums it (with mpm_drag's) into
        # H.delta(mpm_particle), which p2g consumes as the MPM body force. PREDICTION=None,
        # so the engine never integrates the particle set (g2p owns advection).
        return {self.at: acc}

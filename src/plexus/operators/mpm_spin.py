"""mpm_spin -- drive an MLS-MPM body toward slow solid-body rotation (a body force).

A proportional controller toward the rigid-rotation velocity field about a centre `c`:

    v_rot(x) = omega * perp(x - c)          perp((dx,dy)) = (-dy, dx)     [2D]
                                            omega x (x - c) about `axis`  [3D]
    acc_i    = spin_k * (v_rot(x_i) - v_i)

Returned as a particle-level delta the engine sums into `H.delta(mpm_particle)`; the MLS-MPM
`p2g` scatter consumes it as an external body force (`a_ext += H.delta(particle)`), exactly like
`drag` and `gravity`. `EMIT=mpm_acceleration` (the engine does not integrate the particle set; g2p
owns advection), so the spin enters mechanics only through the substep. The `-v_i` term damps
toward the target rate, so a disc started at rest spins UP to `omega` and then rotates steadily
-- a single "rotate the disc slowly" knob, no external swirl map or pacemaker needed. Self-
contained counterpart of the `pulse_to_contraction` + swirl-`vector_grid` recipe.

Dimension-generic: 2D uses the analytic perp; 3D rotates about `axis` (default z) via
omega * (axis x (x - c)).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("mpm_spin", level="particle", kind="lateral")
class MPMSpin(Lateral):
    EMIT = "mpm_acceleration"   # consumed by the MPM substep as a_ext, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["omega"]
    MECHANISM_TAGS = ["solid_body_rotation", "swirl"]
    PARAM_ROLES = {"omega": "angular_velocity", "spin_k": "spin_gain"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.omega = float(params["omega"])                # target angular velocity (rad / time)
        self.spin_k = float(params.get("spin_k", 30.0))    # controller gain toward v_rot
        self.center = params.get("center", None)           # rotation centre; default = domain centre
        self.axis = params.get("axis", [0.0, 0.0, 1.0])    # 3D rotation axis
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        X = lvl.get("pos"); V = lvl.get("vel")
        D = X.shape[1]
        if self.center is not None:
            c = torch.tensor([float(x) for x in self.center][:D], device=dev)
        else:                                              # domain centre: axis 0 = width, rest = 1
            box = [float(b) for b in getattr(H, "world_size", [getattr(H, "world_width", 1.0)] + [1.0] * (D - 1))][:D]
            c = 0.5 * torch.tensor(box, device=dev)
        rel = X - c
        if D == 2:
            v_rot = self.omega * torch.stack([-rel[:, 1], rel[:, 0]], dim=1)
        else:
            ax = torch.tensor([float(a) for a in self.axis][:3], device=dev)
            ax = ax / ax.norm().clamp(min=1e-9)
            v_rot = self.omega * torch.cross(ax.expand_as(rel), rel, dim=1)
        acc = self.spin_k * (v_rot - V) * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

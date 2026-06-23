"""mpm_drag -- viscous body drag for MLS-MPM particles (a force, not an integrator knob).

acc = -k * vel, returned as a particle-level delta the engine sums into
H.delta(mpm_particle); the MLS-MPM `p2g` scatter consumes it as a body force
(a_ext += H.delta(particle)). `PREDICTION=None`: the engine does not integrate the
particle set (g2p owns advection), so the drag enters mechanics only through the
substep -- exactly like the contraction force. This is the composable-operator form of
friction (plexus.tex Part IV: "friction is the `drag` operator, not a knob in the
integrator"); it replaces `p2g`'s built-in `drag` param, so set `p2g drag: 0` and add
`mpm_drag` to the schedule instead. Large k -> the overdamped (quasi-static, tissue)
regime where the sheet creeps to a/k and relaxes without ringing.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("mpm_drag", level="particle", kind="lateral")
class MPMDrag(Lateral):
    PREDICTION = None                 # consumed by the MPM substep as a body force, not engine-integrated
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = ["k"]
    MECHANISM_TAGS = ["viscous_drag", "overdamped"]
    PARAM_ROLES = {"k": "drag_coefficient"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        acc = -self.k * lvl.get("vel") * lvl.occ[:, None]           # -k v: opposes the particle velocity
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

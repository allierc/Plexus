"""tonic_ops -- a constant BASELINE active stress on all six muscles, independent of any
gaze command (PROTOTYPE-LOCAL, not promoted).

    op: tonic_activation
    at: muscle
    tonic: 0.14          # constant activation level, held from frame 0 (never ramped)

Every other activation source in this project (`oculomotor_drive`) computes act's TARGET as
`tonic + gain*relu(...)` and lets the engine integrate toward it over the tau electromechanical
delay. This operator skips both the recruitment term (there is no gaze COMMAND to recruit
toward in a forced-duction spec -- the globe's motion is prescribed, not driven) and the ramp
(real resting motor-unit firing is not something that switches on partway through a trial; it
is already there at t=0). It overwrites `act` directly every frame, the same "this is a
boundary condition, not a force" contract `forced_gaze`/`bone_anchor` use elsewhere here.

Built to test one of three candidate explanations for why an unactivated muscle buckles under
forced duction (see run_forced_duction.py's --tonic): real muscle is rarely fully silent, and
a small baseline tone may be enough to keep a slack muscle's path taut without contaminating
the passive-response measurement as much as a large, targeted pre-tension would.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("tonic_activation", family="mechanics", set="muscle", kind="lateral")
class TonicActivation(Lateral):
    """Every muscle's `act` held at a constant floor, overwritten every frame."""

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["tonic"]
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["muscle"]
    OUTPUTS = ["muscle"]
    READS = ["act"]
    WRITES = ["act"]
    MECHANISM_TAGS = ["boundary_condition", "resting_innervation"]
    PARAM_ROLES = {"tonic": "resting_innervation"}
    REFERENCE = ("Baseline motor-unit firing keeps even 'resting' skeletal muscle under some "
                "tension; extraocular muscle in particular is rarely fully silent.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.tonic = float(params["tonic"])

    def forward(self, H, mask=None):
        m = H.level(self.at)
        a, b = m.state_schema["act"]
        val = torch.full((m.n, 1), self.tonic, dtype=m.state.dtype, device=m.state.device)
        if mask is not None:
            mf = mask[:, None].float()
            val = val * mf + m.state[:, a:b] * (1.0 - mf)
        new = m.state.clone()
        new[:, a:b] = val
        m.state = new
        return {}

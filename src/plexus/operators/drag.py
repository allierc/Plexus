"""Drag -- viscous friction as a composable operator (not an integration knob).

A `drag` force opposes velocity: acc = -k * vel. Modelling friction as an
operator (rather than a `damping` field baked into the integrator) keeps the
integrator generic and lets drag compose in the schedule with the other forces.
It is a second-derivative contribution (an acceleration): `emit: acceleration`
(the class default) sums it into a set the ENGINE integrates (e.g. alongside boids);
`emit: mpm_acceleration` routes the same -k*v into the MPM substep as a body force
(p2g reads H.delta as a_ext) -- this replaces the former `mpm_drag` operator.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("drag", level="particle", kind="lateral")
class Drag(Lateral):
    EMIT = "acceleration"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                      # acts on the D-vector velocity, dimension-generic
    REQUIRES_PARAMS = ["k"]                     # drag coefficient
    MECHANISM_TAGS = ["viscous_drag", "friction", "damping"]
    PARAM_ROLES = {"k": "drag_coefficient", "noise": "thermal_noise"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.noise = float(params.get("noise", 0.0))     # isotropic Langevin noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        occ = lvl.occ
        acc = -self.k * lvl.get("vel") * occ[:, None]
        if self.noise > 0.0:                             # drag + noise = a Brownian/Langevin bath
            N, D = acc.shape
            acc = acc + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None),
                                                 device=acc.device) * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

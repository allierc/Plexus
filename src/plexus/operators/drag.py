"""Drag -- viscous friction as a composable operator (not an integration knob).

A `drag` force opposes velocity: acc = -k * vel. Modelling friction as an
operator (rather than a `damping` field baked into the integrator) keeps the
integrator generic and lets drag compose in the schedule with the other forces.
It is a second-derivative contribution (an acceleration), so it lives on sets
integrated with `prediction: second_derivative` (e.g. alongside boids).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("drag", level="particle", kind="lateral")
class Drag(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                      # acts on the D-vector velocity, dimension-generic
    REQUIRES_PARAMS = ["k"]                     # drag coefficient

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.k = float(params["k"])
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        acc = -self.k * lvl.get("vel") * lvl.occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

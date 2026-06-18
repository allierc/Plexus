"""ScalarField -- a pure-state continuum: a C-channel scalar grid, NO behaviour.

A field is *what it is* (a discretized scalar density over the domain), not what is
done to it. All dynamics -- deposit (set->field), diffuse/decay (field->field), and
sense (field->set) -- are OPERATORS in the schedule, never methods here. The field
only holds its grid and the geometry of its discretization (the world<->pixel map),
so every field-coupled model (slime, reaction-diffusion, morphogen) reuses it.

One channel per coupling target: the slime case uses one channel per species (per
`type` of the coupled set), so categorical separation needs no new field code.
"""
from __future__ import annotations

import torch

from plexus.models.base import Field
from plexus.models.registry import register_field


@register_field("grid", frame="grid")
class ScalarField(Field):
    """A C-channel scalar field on a square grid over [0,W]x[0,1] (pixels dx=1/R).

    Pure state: a `grid` buffer `[C, nx, ny]` plus the geometry to map a world
    position to a pixel (`pix`). Operators read/write `grid` directly.
    """

    def __init__(self, name, couples_to, components=1, res=200, width=1.0, device="cpu"):
        super().__init__(name, couples_to)
        self.C = int(components)
        self.R = int(res)
        self.width = float(width)
        self.nx = int(round(self.width * self.R))      # square pixels, dx = 1/R
        self.ny = self.R
        self.register_buffer("grid", torch.zeros(self.C, self.nx, self.ny, device=device))

    def pix(self, x, y):
        """Nearest pixel (gx, gy) of world positions x, y (the int-cast of the shader)."""
        gx = (x.clamp(0, self.width - 1e-6) * self.R).long().clamp(0, self.nx - 1)
        gy = (y.clamp(0, 1 - 1e-6) * self.R).long().clamp(0, self.ny - 1)
        return gx, gy

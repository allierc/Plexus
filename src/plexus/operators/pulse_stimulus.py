"""pulse_stimulus -- the SPATIAL half of an active stimulus: paint an activation field.

Reads the scalar clock p(t) from `H.signals[clock]` (written by a `pacemaker`) and
writes a Gaussian activation bump, centred at `center` with width `radius`, into the
scalar field named by `at:`:

    a(x, t) = p(t) * exp( -||x - x0||^2 / (2 * sigma^2) ),    sigma = radius

It owns WHERE the stimulus is -- not WHEN (the pacemaker) nor the mechanical effect
(`pulse_to_contraction`). `kind=field`: writes the field grid in place, returns {}.
With no pacemaker run before it (the key is absent from `H.signals`) p(t) defaults to
0, so the field is blank -- schedule the pacemaker first.
"""
from __future__ import annotations

import torch

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("pulse_stimulus", level="field", kind="field")
class PulseStimulus(FieldUpdate):
    MECHANISM_TAGS = ["activation_field", "gaussian_source"]
    PARAM_ROLES = {"radius": "stimulus_width", "center": "stimulus_site", "clock": "pacemaker_signal"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # the field at `at:`
        self.clock = str(params.get("clock", "pacemaker"))         # H.signals key to read p(t)
        self.profile = str(params.get("profile", "gaussian"))      # "gaussian" (localised) | "uniform" (global)
        c = params.get("center", [0.5, 0.5])
        self.center = (float(c[0]), float(c[1]))
        self.sigma = float(params.get("radius", 0.12))
        self.channel = int(params.get("channel", 0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        dev = fld.grid.device
        nx, ny, R = fld.nx, fld.ny, fld.R
        # pixel-centre world coordinates: axis 0 spans [0, width], axis 1 spans [0, 1]
        xs = (torch.arange(nx, device=dev) + 0.5) / R
        ys = (torch.arange(ny, device=dev) + 0.5) / R
        gx = xs[:, None].expand(nx, ny)
        gy = ys[None, :].expand(nx, ny)
        if self.profile == "uniform":
            bump = torch.ones(nx, ny, device=dev)                  # global stimulus: a(x,t) = p(t)
        else:
            r2 = (gx - self.center[0]) ** 2 + (gy - self.center[1]) ** 2
            bump = torch.exp(-r2 / (2.0 * self.sigma * self.sigma))   # localised Gaussian site
        p = float((getattr(H, "signals", None) or {}).get(self.clock, 0.0))   # this tick's clock value
        fld.grid[self.channel] = p * bump
        return {}

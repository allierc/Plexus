"""phase_delay_pulse -- a clocked activation field with a SPATIAL phase delay tau(x,y).

The spatial generalization of the (pacemaker -> pulse_stimulus) pair. Instead of one
GLOBAL clock p(t) painted with a fixed spatial profile, every pixel runs the SAME pulse
waveform but offset by a per-pixel delay read from a map:

    a(x, y, t) = pulse( t - tau(x, y) ),     tau(x, y) = max_delay * delay_map(x, y)

so a region with a larger tau contracts LATER. A spatial delay GRADIENT (e.g.
tau = max_delay * x, or tau ~ atan2(y-.5, x-.5)) makes neighbouring regions fire in
sequence -- a travelling activation wave. Under an elastic tissue (with an active-stress
or contraction operator downstream) that timing gradient turns into curved / rotating
motion -- the substrate for rotary, cardiac-like trajectories. The delay map is an
`image` field (a TIFF, exactly like a stiffness map), so it is analytic now and learnable
later.

The pulse is the same smooth raised bump as `pacemaker` (so force<->stress<->wave is a
clean swap), evaluated per pixel at its own local time `t - tau`:

    s = (t - tau + phase) mod period;   pulse = sin(pi * s / duration)  if s < duration else 0

`kind=field`: writes the activation grid named by `at:` in place and returns {}. It
SUBSUMES pacemaker+pulse_stimulus -- it owns both WHEN (per pixel) and WHERE (via the
map) -- so schedule it in their place. Downstream `pulse_to_active_stress` /
`pulse_to_contraction` read the delayed activation field exactly as before.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fnn

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("phase_delay_pulse", level="field", kind="field")
class PhaseDelayPulse(FieldUpdate):
    PREDICTION = None                       # writes a prescribed field; never engine-integrated
    REQUIRES_PARAMS = ["delay_from"]
    SUPPORTED_DIMS = [2]
    MECHANISM_TAGS = ["activation_field", "phase_delay", "travelling_wave", "spatial_clock"]
    PARAM_ROLES = {"period": "beat_interval", "duration": "active_width",
                   "max_delay": "phase_delay_gain", "delay_from": "delay_map"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # activation field at `at:`
        self.delay_from = params.get("delay_from")
        if self.delay_from is None:
            raise ValueError("phase_delay_pulse needs `delay_from:` (an image field giving the "
                             "normalised delay map tau(x,y) in [0,1])")
        self.period = float(params.get("period", 150.0))     # ticks between beats
        self.duration = float(params.get("duration", 30.0))  # active width (ticks)
        self.phase = float(params.get("phase", 0.0))         # global tick offset
        self.max_delay = float(params.get("max_delay", 10.0))  # ticks of delay at map==1
        self.channel = int(params.get("channel", 0))

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        out = fld.grid[self.channel]                          # [nx, ny] activation channel to write
        delay = H.fields[self.delay_from].grid[0].to(out.device)   # [nx, ny] normalised 0..1
        if delay.shape != out.shape:                          # map at a different resolution: resample
            delay = Fnn.interpolate(delay[None, None].float(), size=tuple(out.shape),
                                    mode="bilinear", align_corners=True)[0, 0]
        tau = self.max_delay * delay                          # per-pixel delay (ticks)
        t = float(getattr(H, "frame", 0))
        s = torch.remainder(t - tau + self.phase, self.period)   # local phase, handles t-tau < 0
        act = torch.where(s < self.duration,
                          torch.sin((math.pi / max(self.duration, 1e-9)) * s),
                          torch.zeros_like(s))                 # smooth bump while active, else 0
        fld.grid[self.channel] = act
        return {}

"""activation_pulse -- paint a clocked activation field. ONE operator, two timing modes,
chosen by whether a per-pixel delay map is given:

* no `delay_from` (the old `pulse_stimulus`) -- a SHARED clock painted with a spatial
  profile. Reads the scalar clock p(t) from `H.signals[clock]` (written by a `pacemaker`)
  and writes
      a(x, t) = p(t) * profile(x),    profile = Gaussian bump at `center` (width `radius`)
                                                or `uniform`.
  It owns WHERE; the pacemaker owns WHEN -- so one clock can drive many sites.

* with `delay_from` (the old `phase_delay_pulse`) -- a PER-PIXEL delayed wave. Every pixel
  runs the same raised bump but offset by a local delay tau(x,y) = max_delay * delay_map(x,y):
      s = (t - tau + phase) mod period;   a = sin(pi s / duration) if s < duration else 0
  a delay GRADIENT makes neighbouring regions fire in sequence -- a travelling activation
  wave. The delay map is an `image` field. This mode owns both WHEN (per pixel) and WHERE
  (via the map), so it needs no pacemaker.

`kind=field`: writes the activation grid named by `at:` in place, returns {}. Downstream
`pulse_to_contraction` / `pulse_to_active_stress` read the activation field either way; the
waveform is the same raised bump as `pacemaker`, so force<->stress<->wave stay clean swaps.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as Fnn

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("activation_pulse", level="field", kind="field")
class ActivationPulse(FieldUpdate):
    EMIT = None                       # writes a prescribed field; never engine-integrated
    SUPPORTED_DIMS = [2]
    REQUIRES_PARAMS = []              # no required params — field target from `at:`; all timing knobs optional
    MECHANISM_TAGS = ["activation_field", "gaussian_source", "phase_delay", "travelling_wave", "spatial_clock"]
    PARAM_ROLES = {"radius": "stimulus_width", "center": "stimulus_site", "clock": "pacemaker_signal",
                   "period": "beat_interval", "duration": "active_width",
                   "max_delay": "phase_delay_gain", "delay_from": "delay_map"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("_at") or params.get("to")   # activation field at `at:`
        self.channel = int(params.get("channel", 0))
        self.delay_from = params.get("delay_from")                 # None -> shared clock; set -> per-pixel wave
        # shared-clock mode (old pulse_stimulus):
        self.clock = str(params.get("clock", "pacemaker"))         # H.signals key to read p(t)
        self.profile = str(params.get("profile", "gaussian"))      # "gaussian" (localised) | "uniform" (global)
        c = params.get("center", [0.5, 0.5])
        self.center = (float(c[0]), float(c[1]))
        self.sigma = float(params.get("radius", 0.12))
        # per-pixel wave mode (old phase_delay_pulse):
        self.period = float(params.get("period", 150.0))           # ticks between beats
        self.duration = float(params.get("duration", 30.0))        # active width (ticks)
        self.phase = float(params.get("phase", 0.0))               # global tick offset
        self.max_delay = float(params.get("max_delay", 10.0))      # ticks of delay at map==1

    def forward(self, H, mask=None):
        fld = H.fields[self.field_name]
        if self.delay_from is None:
            # --- shared clock x spatial profile (old pulse_stimulus) ------------------ #
            dev = fld.grid.device
            nx, ny, R = fld.nx, fld.ny, fld.R
            # pixel-centre world coordinates: axis 0 spans [0, width], axis 1 spans [0, 1]
            xs = (torch.arange(nx, device=dev) + 0.5) / R
            ys = (torch.arange(ny, device=dev) + 0.5) / R
            gx = xs[:, None].expand(nx, ny)
            gy = ys[None, :].expand(nx, ny)
            if self.profile == "uniform":
                bump = torch.ones(nx, ny, device=dev)              # global stimulus: a(x,t) = p(t)
            else:
                r2 = (gx - self.center[0]) ** 2 + (gy - self.center[1]) ** 2
                bump = torch.exp(-r2 / (2.0 * self.sigma * self.sigma))   # localised Gaussian site
            p = float((getattr(H, "signals", None) or {}).get(self.clock, 0.0))   # this tick's clock value
            fld.grid[self.channel] = p * bump
        else:
            # --- per-pixel delayed wave (old phase_delay_pulse) ----------------------- #
            out = fld.grid[self.channel]                           # [nx, ny] activation channel to write
            delay = H.fields[self.delay_from].grid[0].to(out.device)   # [nx, ny] normalised 0..1
            if delay.shape != out.shape:                           # map at a different resolution: resample
                delay = Fnn.interpolate(delay[None, None].float(), size=tuple(out.shape),
                                        mode="bilinear", align_corners=True)[0, 0]
            tau = self.max_delay * delay                           # per-pixel delay (ticks)
            t = float(getattr(H, "frame", 0))
            s = torch.remainder(t - tau + self.phase, self.period)   # local phase, handles t-tau < 0
            act = torch.where(s < self.duration,
                              torch.sin((math.pi / max(self.duration, 1e-9)) * s),
                              torch.zeros_like(s))                 # smooth bump while active, else 0
            fld.grid[self.channel] = act
        return {}

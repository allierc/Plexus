"""pacemaker -- a clocked scalar SOURCE: a periodic activation signal p(t).

The temporal half of an active-stimulus decomposition
(clock -> activation field -> contraction -> mechanics). It owns ONLY timing: it
emits a scalar p(t) in [0,1] on a fixed period and writes it to `H.signals[name]`
-- a scalar side-channel (the same pattern as `H.part_accel`) that any downstream
operator reads by name. It knows nothing about space or mechanics, so one pacemaker
can later drive a FitzHugh-Nagumo activation, a calcium transient, a contraction, or
a secretion; `pacemaker_A`/`pacemaker_B` with different `period`/`phase` make a
multi-site pacing graph.

p(t) is a smooth raised bump of width `duration`, once per `period`, offset by
`phase` (all in OUTER ticks, i.e. `H.frame`):

    s = (frame + phase) mod period;   p = sin(pi * s / duration)  if s < duration else 0

Set `period >= n_frames` for a one-shot pulse. `kind=field` (emits no set delta); its
`at:` is only a scheduling anchor (any declared field) -- it mutates only `H.signals`.
"""
from __future__ import annotations

import math

from plexus.models.base import FieldUpdate
from plexus.models.registry import register_operator


@register_operator("pacemaker", level="field", kind="field")
class Pacemaker(FieldUpdate):
    SUPPORTED_DIMS = [2, 3]
    MECHANISM_TAGS = ["periodic_source", "clock", "pacemaker"]
    PARAM_ROLES = {"period": "beat_interval", "duration": "active_width", "phase": "beat_offset"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.signal = str(params.get("name", "pacemaker"))   # the H.signals key it writes
        self.period = float(params.get("period", 180.0))     # ticks between beats
        self.duration = float(params.get("duration", 20.0))  # active width (ticks)
        self.phase = float(params.get("phase", 0.0))         # tick offset

    def clock(self, frame: int) -> float:
        s = (frame + self.phase) % self.period
        if s < self.duration:
            return math.sin(math.pi * s / max(self.duration, 1e-9))   # smooth 0 -> 1 -> 0 bump
        return 0.0

    def forward(self, H, mask=None):
        if getattr(H, "signals", None) is None:
            H.signals = {}
        H.signals[self.signal] = float(self.clock(int(getattr(H, "frame", 0))))
        return {}

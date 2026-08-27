"""length_drive_ops -- activation from a per-muscle LENGTH target, not a gaze-angle PID
(PROTOTYPE-LOCAL, not promoted).

    op: length_tracking_drive
    at: muscle
    target_len: [0.1261, 0.1382, 0.1679, 0.1447, 0.1628, 0.1499]   # one per muscle, sim units
    ramp_s: 1.5             # seconds to ramp from rest_length to target_len, then hold
    tonic: 0.0
    kp: 8.0
    tau: 0.02

`oculomotor_drive` computes a desired angular velocity from the GAZE error and projects it
onto each muscle's rotation axis -- correct in principle, but every attempt to reach beyond
about +-7 deg through it pushed a muscle's shortening to 25-50% while achieving less rotation
than that, not more: it was recruiting the muscle far past what the geometry needs and paying
for it in buckling, not gaze (see run_forced_duction.py's own investigation).

A forced-duction run (the globe's rotation PRESCRIBED, no active stress at all) gives the
answer directly: whatever LENGTH each muscle settles into while being passively dragged to a
given gaze angle is, by construction, the geometrically correct length for that angle,
independent of whether the shortening was produced by a drag or by the muscle's own
contraction. For the eye_H -20deg sweep that length was 2.34% for MR -- an order of magnitude
under what oculomotor_drive was asking for.

This operator replays that number as a TARGET rather than deriving a torque and hoping the
resulting shortening comes out right:

    deficit_m = max(0, length_m(t) - target_len_m(t))          # positive: still too long
    a_target_m = tonic + kp * deficit_m / rest_length_m
    da/dt = (a_target - a) / tau                                # same first-order dynamics
                                                                 # oculomotor_drive integrates

The same Sherrington rectification `oculomotor_drive` uses falls out for free: a muscle whose
recorded target is LONGER than rest (an antagonist that was passively STRETCHED, like LR
here) has deficit clamped to zero and just sits at tonic -- it is never asked to actively
stretch itself, only to get out of the way.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("length_tracking_drive", family="mechanics", set="muscle", kind="lateral")
class LengthTrackingDrive(Lateral):
    EMIT = "velocity"
    INTEGRAND = "act"
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["target_len"]
    INPUTS = ["muscle"]
    OUTPUTS = ["muscle"]
    READS = ["act", "length"]
    WRITES = ["act"]
    MECHANISM_TAGS = ["motor_command", "length_feedback", "activation_dynamics",
                      "reciprocal_innervation"]
    PARAM_ROLES = {"target_len": "recorded_forced_duction_length", "ramp_s": "ramp_duration_s",
                   "tonic": "resting_innervation", "kp": "length_error_gain",
                   "tau": "activation_time_constant"}
    REFERENCE = ("Target lengths recorded from a forced-duction run (forced_gaze_ops) rather "
                "than tabulated; see length_drive_ops module docstring.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.target_len = torch.as_tensor(params["target_len"], dtype=torch.float32,
                                          device=device)
        self.ramp_s = float(params.get("ramp_s", 1.5))
        self.tonic = float(params.get("tonic", 0.0))
        self.kp = float(params.get("kp", 8.0))
        self.tau = float(params.get("tau", 0.02))
        self.dt = float(params.get("dt", 0.003))
        self.last = {}

    def forward(self, H, mask=None):
        m = H.level(self.at)
        dev = m.state.device
        frame = int(getattr(H, "frame", 0))
        t = frame * self.dt
        frac = min(max(t / max(self.ramp_s, 1e-9), 0.0), 1.0)
        rest = m.rest_length
        target_now = rest + frac * (self.target_len.to(dev) - rest)

        length = m.get("length")[:, 0]
        deficit = (length - target_now).clamp(min=0.0)
        a_target = (self.tonic + self.kp * deficit / rest.clamp(min=1e-9)).clamp(0.0, 1.0)

        a = m.get("act")[:, 0]
        d_act = (a_target - a) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"target_now": target_now.detach().cpu().numpy(),
                     "a_target": a_target.detach().cpu().numpy()}
        return {self.at: d_act[:, None]}

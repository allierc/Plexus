"""probe_ops -- open-loop activation injection, for plant identification (PROTOTYPE-LOCAL).

Phase 1a identified the plant from a run with the controller CLOSED. In that data the input is a
function of the state -- that is what feedback means -- so the regressor is correlated with the
residual and least squares is biased. The surrogate that came out predicted a 1.7 degree excursion
where the eye had made 17, failed its own fidelity check, and the tuner refused to report.

`muscle_probe` is the fix, and it is the standard autotune move: OPEN the loop and inject a step on
one muscle at a time. The input is then prescribed, independent of the state by construction, and
the closed-loop bias is gone -- not reduced, gone.

What one probe run buys that the closed-loop fit could not:

    the STATIC GAIN, read straight off the plateau.  Delta theta_infinity / Delta a  is the
    quantity that decides whether the standing gaze error is a control problem or a mechanical
    one, and it is exactly what an acceleration fit (theta_ddot = Bu - C theta_dot - K theta) has
    no leverage on -- that fit is dominated by transients.

Six runs, one per muscle, give the full 3x6 static gain matrix plus the rise time and overshoot per
channel: the plant, measured rather than assumed.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

import eye_anatomy as EA


@register_operator("muscle_probe", family="signalling", set="muscle", kind="lateral")
class MuscleProbe(Lateral):
    """Drive the six activations from a PRESCRIBED waveform, ignoring the gaze entirely.

    A drop-in replacement for `oculomotor_drive` in an identification run: same set, same
    integrated block, same first-order activation dynamics (`tau`), no feedback. Every muscle is
    held at `tonic`; the one named by `muscle` steps to `a_hi` between `t_on` and `t_off` and
    returns to `tonic` afterwards, so a single run carries a step ON and a step OFF and the release
    transient is measured too.

    `step_frames` smooths the edge over a few frames. That is not cosmetic: an instantaneous jump in
    commanded activation excites the MPM substep far above anything the closed loop ever does, and
    an identification run should probe the plant, not the integrator.
    """

    EMIT = "velocity"                        # delta is da/dt ...
    INTEGRAND = "act"                        # ... into the muscle's `act` block
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["muscle"]
    INPUTS = ["muscle"]
    OUTPUTS = ["muscle"]
    READS = ["act"]
    WRITES = ["act"]
    MECHANISM_TAGS = ["open_loop_probe", "system_identification", "step_response"]
    PARAM_ROLES = {"muscle": "probed_muscle_index", "a_hi": "step_amplitude",
                   "tonic": "resting_innervation", "t_on": "step_onset_frame",
                   "t_off": "step_release_frame", "tau": "activation_time_constant",
                   "step_frames": "edge_smoothing"}
    REFERENCE = "Ljung, L. (1999). System Identification: Theory for the User, 2nd ed. (open-loop step response); Astrom, K. J. & Hagglund, T. (1984). Automatica 20:645 (autotuning)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "muscle")
        self.muscle = int(params["muscle"])          # -1 = all muscles held at tonic (a null run)
        self.a_hi = float(params.get("a_hi", 1.0))
        self.tonic = float(params.get("tonic", 0.14))
        self.t_on = float(params.get("t_on", 60))
        self.t_off = float(params.get("t_off", 240))
        self.tau = float(params.get("tau", 0.02))
        self.step_frames = float(params.get("step_frames", 4.0))
        self.last = {}

    def level(self, frame: float) -> float:
        """The commanded activation of the probed muscle at `frame` (a smoothed boxcar)."""
        if self.muscle < 0:
            return self.tonic
        w = max(self.step_frames, 1e-6)
        on = float(np.clip((frame - self.t_on) / w, 0.0, 1.0))
        off = float(np.clip((frame - self.t_off) / w, 0.0, 1.0))
        return self.tonic + (self.a_hi - self.tonic) * (on - off)

    def forward(self, H, mask=None):
        m = H.level(self.at)
        dev = m.state.device
        frame = int(getattr(H, "frame", 0))
        cmd = torch.full((m.n,), self.tonic, device=dev)
        if self.muscle >= 0:
            cmd[self.muscle] = self.level(frame)
        a = m.get("act")[:, 0]
        d_act = (cmd - a) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"commanded": cmd.detach().cpu().numpy().copy()}
        return {self.at: d_act[:, None]}


def probe_spec(base_spec: dict, muscle: int, a_hi=1.0, tonic=0.14,
               t_on=60, t_off=240, n_frames=320) -> dict:
    """Turn an archived closed-loop spec into an OPEN-LOOP probe of one muscle.

    The base spec is loaded from the archive rather than rebuilt, so the probe measures exactly the
    configuration that was archived -- not today's defaults, which have moved on. Only two things
    change: `oculomotor_drive` is swapped for `muscle_probe`, and the frame count is cut to the
    probe window. Everything mechanical is untouched.
    """
    import copy
    spec = copy.deepcopy(base_spec)
    spec["general"] = dict(spec["general"])
    spec["general"]["n_frames"] = int(n_frames)
    spec["general"]["name"] = f"{spec['general']['name']}_probe_{EA.MUSCLE_KEYS[muscle] if muscle >= 0 else 'null'}"

    ops = []
    for o in spec["operators"]:
        if o["op"] == "oculomotor_drive":
            ops.append({"op": "muscle_probe", "at": "muscle", "muscle": int(muscle),
                        "a_hi": float(a_hi), "tonic": float(tonic),
                        "t_on": int(t_on), "t_off": int(t_off),
                        "tau": float(o.get("tau", 0.02))})
        else:
            ops.append(o)
    spec["operators"] = ops
    spec["schedule"] = ["muscle_probe" if s == "oculomotor_drive" else s
                        for s in spec["schedule"]]
    return spec

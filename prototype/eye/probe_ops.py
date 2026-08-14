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
    spec["schedule"] = ["muscle_probe" if s in ("oculomotor_drive", "muscle_probe") else s
                        for s in spec["schedule"]]
    return spec


@register_operator("muscle_probe", implementation="tour",
                   family="signalling", set="muscle", kind="lateral")
class MuscleProbeTour(MuscleProbe):
    """The same open-loop probe, but stepping every muscle in turn within ONE run.

    Six separate probe runs are the right thing for identification -- each gives a clean
    static gain with nothing else moving. They are the wrong thing for LOOKING at the
    plant: six files, six colour scales, six different moments to compare. Here the six
    steps are laid end to end on one timeline, each muscle held for `hold` frames and
    then released for `rest` so the globe comes back to rest before the next one pulls.
    The rendered panels then share one strain scale, one stress scale and one gaze axis
    across all six muscles, so their excursions can be read against each other directly.

    Same contract as `muscle_probe`: same set, same integrated block, same activation
    time constant, no feedback anywhere. Only the waveform differs.
    """

    REQUIRES_PARAMS = []
    PARAM_ROLES = dict(MuscleProbe.PARAM_ROLES, order="probe_sequence",
                       hold="frames_each_muscle_is_on", rest="frames_between_muscles",
                       lead="frames_before_the_first_step")

    def __init__(self, params, device="cpu"):
        params = dict(params)
        params.setdefault("muscle", -1)
        super().__init__(params, device)
        self.order = [int(i) for i in params.get("order", range(EA.N_MUSCLE))]
        self.hold = float(params.get("hold", 70))
        self.rest = float(params.get("rest", 45))
        self.lead = float(params.get("lead", 40))

    def window(self, slot):
        """(t_on, t_off) of the `slot`-th step."""
        t_on = self.lead + slot * (self.hold + self.rest)
        return t_on, t_on + self.hold

    def n_frames(self, tail=60):
        return int(self.lead + len(self.order) * (self.hold + self.rest) + tail)

    def levels(self, frame: float, n=None) -> np.ndarray:
        """Commanded activation of every muscle at `frame`.

        Sized from the SET, not from the eye's six: the minimal rig in `archive/`
        drives an antagonist pair, and a probe that insists on six returns a delta the
        engine cannot integrate."""
        cmd = np.full(EA.N_MUSCLE if n is None else int(n), self.tonic)
        w = max(self.step_frames, 1e-6)
        for slot, mi in enumerate(self.order):
            if mi >= len(cmd):
                continue
            t_on, t_off = self.window(slot)
            on = float(np.clip((frame - t_on) / w, 0.0, 1.0))
            off = float(np.clip((frame - t_off) / w, 0.0, 1.0))
            cmd[mi] = self.tonic + (self.a_hi - self.tonic) * (on - off)
        return cmd

    def level(self, frame: float) -> float:
        return float(self.levels(frame).max())

    def active(self, frame: float):
        """Which muscle is being driven at `frame`, or None between steps."""
        for slot, mi in enumerate(self.order):
            t_on, t_off = self.window(slot)
            if t_on <= frame <= t_off:
                return EA.MUSCLE_KEYS[mi]
        return None

    def forward(self, H, mask=None):
        m = H.level(self.at)
        dev = m.state.device
        frame = int(getattr(H, "frame", 0))
        cmd = torch.as_tensor(self.levels(frame, n=m.n), dtype=torch.float32, device=dev)
        a = m.get("act")[:, 0]
        d_act = (cmd - a) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"commanded": cmd.detach().cpu().numpy().copy()}
        return {self.at: d_act[:, None]}


def tour_spec(base_spec: dict, a_hi=1.0, tonic=0.14, hold=70, rest=45, lead=40,
              order=None, n_frames=None) -> dict:
    """`probe_spec`'s sibling: swap the closed loop for the six-step tour."""
    import copy
    spec = copy.deepcopy(base_spec)
    order = list(range(EA.N_MUSCLE)) if order is None else list(order)
    n_frames = int(n_frames if n_frames is not None
                   else lead + len(order) * (hold + rest) + 60)
    spec["general"] = dict(spec["general"])
    spec["general"]["n_frames"] = n_frames
    spec["general"]["name"] = f"{spec['general']['name']}_tour"
    ops = []
    for o in spec["operators"]:
        # accept either a closed-loop spec or an already-archived single-muscle probe:
        # the archived models A-E only kept probe specs, and the tour replaces the drive
        # in exactly the same slot either way.
        if o["op"] in ("oculomotor_drive", "muscle_probe"):
            ops.append({"op": "muscle_probe", "implementation": "tour", "at": "muscle",
                        "a_hi": float(a_hi), "tonic": float(tonic),
                        "hold": int(hold), "rest": int(rest), "lead": int(lead),
                        "order": [int(i) for i in order],
                        "tau": float(o.get("tau", 0.02))})
        else:
            ops.append(o)
    spec["operators"] = ops
    spec["schedule"] = ["muscle_probe" if s in ("oculomotor_drive", "muscle_probe") else s
                        for s in spec["schedule"]]
    return spec


@register_operator("muscle_probe", implementation="staircase",
                   family="signalling", set="muscle", kind="lateral")
class MuscleProbeStaircase(MuscleProbe):
    """A DESCENDING AMPLITUDE STAIRCASE on one muscle: 1.00, 0.75, 0.50, 0.25, tonic.

    Every probe in this archive so far has been a full-on step held for 0.54 s (t36-t43)
    or 0.10 s (the tour). The fitted plant is omega_n = 9.8 rad/s, zeta = 0.26, so settling
    to 2% takes 4/(zeta.omega_n) = 1.28 s: NO STEADY STATE HAS EVER BEEN MEASURED HERE.
    Every static gain in the identification therefore rests on an endpoint that was still
    moving. Each level below is held `hold` frames, 2.0 s by default -- about 1.5 settling
    times -- which is the whole point of the operator.

    And it sweeps rather than stepping to full on, because a step at u = 1 pins the
    activation-to-torque map Phi at its endpoints and says nothing in between. The
    curvature that distinguishes a muscle from a linear gain lives at intermediate
    amplitudes, which is where the circuit actually operates.

    DESCENDING, in one run, with no return to rest between levels: each transition is
    also a step-down whose transient carries omega_n and zeta, so the same run feeds both
    the static and the dynamic half of the fit and nothing is spent twice.
    """

    REQUIRES_PARAMS = ["muscle"]
    PARAM_ROLES = dict(MuscleProbe.PARAM_ROLES, levels="amplitude_staircase",
                       hold="frames_per_level", lead="frames_before_the_first_level",
                       tail="frames_held_at_tonic_afterwards")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.levels = [float(v) for v in params.get("levels", (1.0, 0.75, 0.50, 0.25))]
        self.hold = int(params.get("hold", 667))          # 2.0 s at dt = 0.003
        self.lead = int(params.get("lead", 167))          # 0.5 s settling at tonic first
        self.tail = int(params.get("tail", 667))          # 2.0 s back at tonic at the end

    def n_frames(self):
        return int(self.lead + len(self.levels) * self.hold + self.tail)

    def windows(self):
        """[(level, t_on, t_off)] -- when each level is commanded."""
        return [(lv, self.lead + i * self.hold, self.lead + (i + 1) * self.hold)
                for i, lv in enumerate(self.levels)]

    def level(self, frame: float) -> float:
        """Commanded activation of the probed muscle, with every edge smoothed.

        The edges are ramped over `step_frames` for the reason the parent class gives:
        an instantaneous jump excites the MPM substep far harder than anything the
        closed loop does, and this run is meant to probe the plant, not the integrator.
        """
        t = float(frame) - self.lead
        if t < 0:
            return self.tonic
        i = int(t // self.hold)
        local = t - i * self.hold
        prev = self.levels[i - 1] if 0 < i <= len(self.levels) else self.tonic
        cur = self.levels[i] if i < len(self.levels) else self.tonic
        w = float(np.clip(local / max(self.step_frames, 1e-6), 0.0, 1.0))
        return prev + (cur - prev) * w

    def levels_all(self, frame: float) -> np.ndarray:
        cmd = np.full(EA.N_MUSCLE, self.tonic)
        if self.muscle >= 0:
            cmd[self.muscle] = self.level(frame)
        return cmd


def staircase_spec(base_spec: dict, muscle: int, levels=(1.0, 0.75, 0.50, 0.25),
                   hold=667, lead=167, tail=667, tonic=0.14) -> dict:
    """`probe_spec`'s long-hold sibling: one muscle, one descending staircase."""
    import copy
    spec = copy.deepcopy(base_spec)
    n_frames = int(lead + len(levels) * hold + tail)
    spec["general"] = dict(spec["general"])
    spec["general"]["n_frames"] = n_frames
    spec["general"]["record_cap"] = max(int(spec["general"].get("record_cap", 4000)), n_frames)
    spec["general"]["name"] = (f"{spec['general']['name']}_stair_"
                               f"{EA.MUSCLE_KEYS[muscle] if muscle >= 0 else 'null'}")
    ops = []
    for o in spec["operators"]:
        if o["op"] in ("oculomotor_drive", "muscle_probe"):
            ops.append({"op": "muscle_probe", "implementation": "staircase", "at": "muscle",
                        "muscle": int(muscle), "levels": [float(v) for v in levels],
                        "hold": int(hold), "lead": int(lead), "tail": int(tail),
                        "tonic": float(tonic), "tau": float(o.get("tau", 0.02))})
        else:
            ops.append(o)
    spec["operators"] = ops
    spec["schedule"] = ["muscle_probe" if s in ("oculomotor_drive", "muscle_probe") else s
                        for s in spec["schedule"]]
    return spec


@register_operator("muscle_probe", implementation="hold_vector",
                   family="signalling", set="muscle", kind="lateral")
class MuscleProbeHoldVector(MuscleProbeStaircase):
    """Hold an ARBITRARY SUBSET of muscles at fixed levels -- the protocol's unit of work.

    Stage 1 drives one muscle, stage 2 drives two, and the static map is a function of
    all six, so the probe takes a VECTOR rather than an index. Everything else is the
    staircase's: ramp over `step_frames` (a jump excites the substep harder than the
    controller ever will), hold, release to tonic.

    The muscles that are not named are not silent -- they sit at `tonic`, exactly as they
    do in the closed loop, because that resting pull is part of the plant being measured.
    """

    REQUIRES_PARAMS = []
    PARAM_ROLES = dict(MuscleProbeStaircase.PARAM_ROLES,
                       muscles="driven_muscle_indices", levels_vec="their_levels")

    def __init__(self, params, device="cpu"):
        params = dict(params)
        params.setdefault("muscle", -1)
        super().__init__(params, device)
        self.driven = [int(i) for i in params.get("muscles", [])]
        self.driven_levels = [float(v) for v in params.get("levels_vec", [])]
        if len(self.driven) != len(self.driven_levels):
            raise ValueError("muscle_probe[hold_vector]: one level per driven muscle")

    def n_frames(self):
        return int(self.lead + self.hold + self.tail)

    def _w(self, frame):
        """0 before the ramp, 1 through the hold, 0 again after the release."""
        t = float(frame) - self.lead
        w = max(self.step_frames, 1e-6)
        return (float(np.clip(t / w, 0.0, 1.0))
                - float(np.clip((t - self.hold) / w, 0.0, 1.0)))

    def levels_all(self, frame: float, n=None) -> np.ndarray:
        """The command vector. `n` defaults to the eye's six, but the set decides:
        the minimal transmission rig in `archive/` has ONE muscle, and a probe that
        insists on six returns a delta the engine cannot integrate."""
        cmd = np.full(EA.N_MUSCLE if n is None else int(n), self.tonic)
        w = self._w(frame)
        for mi, lv in zip(self.driven, self.driven_levels):
            if mi < len(cmd):
                cmd[mi] = self.tonic + (lv - self.tonic) * w
        return cmd

    def level(self, frame: float) -> float:
        return float(self.levels_all(frame).max())

    def forward(self, H, mask=None):
        m = H.level(self.at)
        dev = m.state.device
        cmd = torch.as_tensor(self.levels_all(int(getattr(H, "frame", 0)), n=m.n),
                              dtype=torch.float32, device=dev)
        d_act = (cmd - m.get("act")[:, 0]) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"commanded": cmd.detach().cpu().numpy().copy()}
        return {self.at: d_act[:, None]}


@register_operator("muscle_probe", implementation="groups",
                   family="signalling", set="muscle", kind="lateral")
class MuscleProbeGroups(MuscleProbeStaircase):
    """Drive SETS of muscles in turn -- the synergies, not the muscles one at a time.

    `muscle_probe[tour]` steps one muscle per slot, which is right for identification and
    wrong for asking what the plant DOES: no single extraocular muscle moves the eye along
    a cardinal axis. Elevation is a pair, depression is the opposite pair, and it is the
    pair that the circuit commands. Each entry of `groups` is a list of muscle indices
    driven together to `a_hi`, held, and released before the next.

    Written for the question "does this geometry move the eye up when the two elevators
    pull?", which a per-muscle tour cannot answer.
    """

    REQUIRES_PARAMS = []
    PARAM_ROLES = dict(MuscleProbeStaircase.PARAM_ROLES, groups="synergies_in_order")

    def __init__(self, params, device="cpu"):
        params = dict(params)
        params.setdefault("muscle", -1)
        super().__init__(params, device)
        self.groups = [list(map(int, g)) for g in params.get("groups", [])]
        self.rest = int(params.get("rest", 250))

    def n_frames(self):
        return int(self.lead + len(self.groups) * (self.hold + self.rest) + self.tail)

    def window(self, slot):
        t_on = self.lead + slot * (self.hold + self.rest)
        return t_on, t_on + self.hold

    def levels_all(self, frame: float, n=None) -> np.ndarray:
        cmd = np.full(EA.N_MUSCLE if n is None else int(n), self.tonic)
        w = max(self.step_frames, 1e-6)
        for slot, g in enumerate(self.groups):
            t_on, t_off = self.window(slot)
            on = float(np.clip((frame - t_on) / w, 0.0, 1.0))
            off = float(np.clip((frame - t_off) / w, 0.0, 1.0))
            lvl = self.tonic + (self.a_hi - self.tonic) * (on - off)
            for mi in g:
                if mi < len(cmd):
                    cmd[mi] = max(cmd[mi], lvl)
        return cmd

    def level(self, frame: float) -> float:
        return float(self.levels_all(frame).max())

    def active(self, frame: float):
        for slot, g in enumerate(self.groups):
            t_on, t_off = self.window(slot)
            if t_on <= frame <= t_off:
                return "+".join(EA.MUSCLE_KEYS[i] for i in g)
        return None

    def forward(self, H, mask=None):
        m = H.level(self.at)
        dev = m.state.device
        cmd = torch.as_tensor(self.levels_all(int(getattr(H, "frame", 0)), n=m.n),
                              dtype=torch.float32, device=dev)
        d_act = (cmd - m.get("act")[:, 0]) / self.tau
        if mask is not None:
            d_act = d_act * mask.float()
        self.last = {"commanded": cmd.detach().cpu().numpy().copy()}
        return {self.at: d_act[:, None]}


def groups_spec(base_spec: dict, groups, hold=500, rest=300, lead=100, tail=250,
                a_hi=1.0, tonic=0.14) -> dict:
    """`tour_spec`'s sibling: drive synergies in turn instead of single muscles."""
    import copy
    spec = copy.deepcopy(base_spec)
    n_frames = int(lead + len(groups) * (hold + rest) + tail)
    spec["general"] = dict(spec["general"])
    spec["general"]["n_frames"] = n_frames
    spec["general"]["record_cap"] = max(int(spec["general"].get("record_cap", 4000)), n_frames)
    spec["general"]["name"] = f"{spec['general']['name']}_synergies"
    ops = []
    for o in spec["operators"]:
        if o["op"] in ("oculomotor_drive", "muscle_probe"):
            ops.append({"op": "muscle_probe", "implementation": "groups", "at": "muscle",
                        "groups": [list(map(int, g)) for g in groups],
                        "a_hi": float(a_hi), "tonic": float(tonic), "hold": int(hold),
                        "rest": int(rest), "lead": int(lead), "tail": int(tail),
                        "tau": float(o.get("tau", 0.02))})
        else:
            ops.append(o)
    spec["operators"] = ops
    spec["schedule"] = ["muscle_probe" if s in ("oculomotor_drive", "muscle_probe") else s
                        for s in spec["schedule"]]
    return spec

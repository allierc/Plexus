"""forced_gaze_ops -- rotate the globe KINEMATICALLY and read the passive muscle stress
that motion drags into the muscles (PROTOTYPE-LOCAL, not promoted).

Every other run in this prototype goes activation -> active stress -> contraction ->
rotation: a muscle pulls and the globe is what moves. This inverts the causal chain. The
globe's rotation is PRESCRIBED -- written directly into `mpm_particle`'s pos/vel every
frame, the same "the object does not move, at any load" pattern `pin_region [clamp]` uses
for a pinned bone (`archive/bench_ops.py`) -- and NO muscle carries any active stress
(`muscle_contract` is not in the spec at all). Whatever stress a muscle then shows is purely
its PASSIVE elastic response to being dragged by the globe through the tendon end and held
by the origin end: a forced duction test, the same manipulation an ophthalmologist performs
on an anaesthetised eye to tell a muscle that is not pulling from one that cannot be moved.

    op: forced_gaze
    at: mpm_particle
    axis: h                # h (about the sim's +y, this model's own "abduction" axis --
                            # see oculomotor_drive's own omega convention) | v | t
    amplitude: 20.0         # deg, the sweep's peak
    period_s: 3.0           # seconds for one full 0 -> amplitude -> 0 sweep

THE SWEEP is a single raised-cosine bump, not a step or a ramp with corners:

    theta(t) = (amplitude/2) * (1 - cos(2 pi t / period))

chosen because its derivative theta'(t) = (amplitude pi / period) sin(2 pi t / period) is
ZERO at t=0, t=period/2 (the peak) and t=period: the globe starts from rest, decelerates
smoothly INTO the peak rather than being stopped by a boundary condition, and returns to
rest -- no velocity discontinuity anywhere for the shared grid to feel as an impulse.

WHY THIS WRITES POS AND VEL, not just pos. MLS-MPM's P2G scatter carries a particle's
CURRENT velocity into the grid every substep (`mpm_scatter`'s `mom = mass*V + affine*offset`);
a particle whose position jumps but whose velocity is stale would scatter the WRONG
momentum for one substep, which is exactly the sort of one-tick impulse this operator exists
to avoid. Velocity is the closed-form rigid-body field v = omega x r for the SAME theta(t),
not a finite difference of position -- no lag, no frame-rate dependence.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

AXES = {"h": np.array([0.0, 1.0, 0.0]),      # oculomotor_drive's own convention:
        "v": np.array([-1.0, 0.0, 0.0]),     # omega = (-e_v, e_h, e_t) i.e. rotation about
        "t": np.array([0.0, 0.0, 1.0])}      # +y is abduction, about -x is elevation, +z torsion


def _rodrigues(axis, theta):
    """[3,3] rotation by `theta` radians about a unit `axis`."""
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


@register_operator("forced_gaze", family="mechanics", set="particle", kind="lateral")
class ForcedGaze(Lateral):
    """Prescribe the globe's rotation; read the muscles' PASSIVE response elsewhere.

    Writes `pos`/`vel` directly every frame (a kinematic constraint, the same contract as
    `pin_region [clamp]`), so the globe's motion is authoritative and no force operator
    acting on `mpm_particle` (`orbit_socket`, drag) has any lasting effect -- whatever they
    compute is overwritten the next tick. Both should be left out of a forced-gaze spec
    rather than merely outvoted, so the schedule says what is actually happening.
    """

    EMIT = None
    SUPPORTED_DIMS = [3]
    REQUIRES_PARAMS = ["amplitude", "period_s"]
    MAY_MUTATE_INTEGRATED_STATE = True
    INPUTS = ["mpm_particle"]
    OUTPUTS = ["mpm_particle"]
    READS = ["pos", "vel"]
    WRITES = ["pos", "vel"]
    MECHANISM_TAGS = ["boundary_condition", "kinematic_constraint", "forced_duction"]
    PARAM_ROLES = {"axis": "rotation_axis", "amplitude": "peak_angle_deg",
                   "period_s": "sweep_period_s"}
    REFERENCE = ("Forced duction testing: the passive rotation of an eye under external "
                "torque, used clinically to distinguish a muscle that will not pull from "
                "one that cannot be moved.")

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "mpm_particle")
        self.center = np.asarray(params.get("center"), float)
        self.axis = AXES[str(params.get("axis", "h"))]
        self.amplitude = math.radians(float(params["amplitude"]))
        self.period = float(params["period_s"])
        self.dt = float(params.get("dt", 0.003))
        self.last_theta_deg = 0.0                      # read by the caller for logging/HUD

    def forward(self, H, mask=None):
        p = H.level(self.at)
        dev = p.state.device
        if not hasattr(p, "rest"):
            return {}
        frame = int(getattr(H, "frame", 0))
        # clamp AT period: past the nominal sweep the schedule re-enters the same periodic
        # formula and drifts off theta=0 again -- frames beyond `period` exist to let the
        # globe HOLD at rest so the muscles can relax, not to run a second partial sweep
        t = min(frame * self.dt, self.period)
        phase = 2.0 * math.pi * t / self.period
        theta = 0.5 * self.amplitude * (1.0 - math.cos(phase))
        omega = (self.amplitude * math.pi / self.period) * math.sin(phase)   # rad/s
        self.last_theta_deg = math.degrees(theta)

        R = torch.as_tensor(_rodrigues(self.axis, theta), dtype=torch.float32, device=dev)
        c = torch.as_tensor(self.center, dtype=torch.float32, device=dev)
        ax = torch.as_tensor(self.axis, dtype=torch.float32, device=dev)

        new_pos = c[None, :] + p.rest @ R.T
        rel = new_pos - c[None, :]
        new_vel = omega * torch.cross(ax[None, :].expand_as(rel), rel, dim=1)

        pa, pb = p.state_schema["pos"]
        va, vb = p.state_schema["vel"]
        if mask is not None:
            m = mask[:, None].float()
            new_pos = m * new_pos + (1 - m) * p.state[:, pa:pb]
            new_vel = m * new_vel + (1 - m) * p.state[:, va:vb]
        new = p.state.clone()
        new[:, pa:pb] = new_pos
        new[:, va:vb] = new_vel
        p.state = new
        return {}

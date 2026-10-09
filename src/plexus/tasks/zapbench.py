"""ZAPBench's visual session as a drive for the two-eye rig (Cedric, 2026-10-08: "can we generate zapbench-like data
with the rig simulation?"). One process, no teacher of its own -- the corpus pairs it with the rig's `statespace` law.

    zapbench_slip   a process: the 2-h session's horizontal whole-field image slip, deg/s, the same on both retinas
                    (the conjugate drive the rig was trained on: t6_binocular_gaze, `independent: false`), held over
                    each of the recording's 0.914-s frames and resampled to the rig's clock

THE FEATURES ARE CONDITION CODES, NOT MOTION. ZAPBench's covariates (Lueckmann et al. 2025, App. B.6; the recording's
`stimulus` [7879, 22], columns = the paper's dimensions 1-22) give a direction or an on/off flag, never a speed; only two
conditions move the whole field sideways, and only they reach the rig:
    rotation  column 19 (dim 20): +1 leftward / -1 rightward rotating grating, the whole condition (10 x 30 s each way)
    turning   column 9 (dim 10) 1 while the grating drifts; columns 10, 11 (dims 11, 12) the direction's (cos, sin).
              The drifts cycle (cos, sin) = (0, 1), (1, 0), (-1, 0), (0, -1) and the paper's order is forward, leftward,
              rightward, backward, so cos = +1 is leftward: the lateral slip is column 9 x column 10. Forward and
              backward drifts slip the two eyes in opposite directions, which the rig never saw: 0 here.
Every other condition is 0: dots (column 2 is +-1 for two settings over the whole condition, so its three 20-s coherent
rightward periods cannot be picked out), flash / taxis / dark (luminance; the rig has no luminance input), gain /
position / open loop (forward gratings). + is leftward throughout.

THE SPEEDS ARE PARAMETERS, NOT DATA (the release has no stimulus speed). The rig has no quick phase: a held drive u
holds the eyes at the teacher's steady state 0.5 (u + u) / 0.125 = 8 u deg (tau 8 s), so the default 1 deg/s keeps a
30-s rotation at ~8 deg, inside the trained gaze range (target RMS 6.5 deg; drive SD 7.37 deg/s band-limited to 1 Hz).
"""
from __future__ import annotations

import numpy as np

from plexus.tasks import register_stimulus


@register_stimulus("zapbench_slip", family="zapbench")
def zapbench_slip(rng, T, dt, channels, v_rotation_deg_s=1.0, v_turning_deg_s=1.0, **_):
    """[T, channels]: the conjugate slip (deg/s) on every channel, frame k at time k dt."""
    from plexus.paths import graphs_data_path
    d = np.load(graphs_data_path("zebrafish", "zapbench_recording.npz"), allow_pickle=True)
    S, t_s = d["stimulus"].astype(np.float64), d["t_s"].astype(np.float64)
    slip = float(v_rotation_deg_s) * S[:, 19] + float(v_turning_deg_s) * S[:, 9] * np.round(S[:, 10])
    frame = np.clip(np.searchsorted(t_s, np.arange(T) * dt, side="right") - 1, 0, len(t_s) - 1)
    return np.repeat(slip[frame][:, None], channels, axis=1)

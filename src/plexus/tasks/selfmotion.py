"""Self-motion: the swim-event drive and the heading it integrates to (Cedric, 2026-10-08: "the ARTR twin, restricted to
angle integration"). The task of Cedric's note "Connectome-constrained self-motion integration in the larval zebrafish
hindbrain" (connectome-gnn-cx docs/zebrafish.tex, Sect. 2, "Swim-integration task"), heading only, as its code generates
it (connectome-gnn-cx graph_data_generator.py:937-1041, audited 2026-10-08):

    swim_heading   a process: channel 0 the angular velocity omega in DEG/S, channels 1, 2 the starting heading's cue
                   (cos theta0, sin theta0) on the first frame only, 0 after
    heading        a teacher: theta(t) = theta0 + sum omega dt (theta0 read back from the cue, theta[0] = theta0),
                   the targets (cos theta, sin theta)

THE EVENTS. Each frame a swim starts with probability rate_hz * dt; its kind is left / right / forward / backward with
fractions (0.3, 0.3, 0.2, 0.2); only left (+) and right (-) turn the heading. A turn's size |dtheta| is lognormal
(median 0.785 rad, sigma_log 0.51), spread as a boxcar of `event_s` (0.3 s) at height dtheta / event_s; overlapping
events add. omega is fed in deg/s, as the reference feeds it (the unit sets the scale of the gate's gains).
"""
from __future__ import annotations

import numpy as np

from plexus.tasks import register_stimulus, register_teacher


@register_stimulus("swim_heading", family="selfmotion")
def swim_heading(rng, T, dt, channels, rate_hz=0.5, event_s=0.3, fractions=(0.3, 0.3, 0.2, 0.2),
                 dtheta_median_rad=0.785, dtheta_sigma_log=0.51, **_):
    """[T, 3]: omega (deg/s), cos theta0 at frame 0, sin theta0 at frame 0."""
    if channels != 3:
        raise ValueError(f"swim_heading writes 3 channels (omega, cos theta0, sin theta0); `general.channels` is {channels}")
    f = np.asarray(fractions, np.float64)
    f = f / f.sum()
    n_ev = max(1, int(round(float(event_s) / dt)))
    omega = np.zeros(T)
    onset = rng.random(T) < float(rate_hz) * dt
    kind = rng.choice(4, size=T, p=f)                                  # 0 left, 1 right, 2 forward, 3 backward
    size = np.exp(np.log(float(dtheta_median_rad)) + float(dtheta_sigma_log) * rng.standard_normal(T))
    for t in np.flatnonzero(onset & (kind < 2)):
        sgn = 1.0 if kind[t] == 0 else -1.0
        omega[t:t + n_ev] += np.degrees(sgn * size[t] / (n_ev * dt))   # a boxcar of height dtheta / event_s, in deg/s
    th0 = rng.uniform(0.0, 2.0 * np.pi)
    u = np.zeros((T, 3))
    u[:, 0] = omega
    u[0, 1], u[0, 2] = np.cos(th0), np.sin(th0)
    return u


@register_teacher("heading", family="selfmotion")
def heading(u, dt, **_):
    """[N, T, 2]: (cos theta, sin theta), theta = theta0 + cumsum(omega) dt, theta[0] = theta0."""
    th0 = np.arctan2(u[:, 0, 2], u[:, 0, 1])                          # [N]
    th = th0[:, None] + np.cumsum(np.radians(u[:, :, 0]), axis=1) * dt
    th[:, 0] = th0
    return np.stack([np.cos(th), np.sin(th)], -1)


heading.n_targets = lambda channels, **_: 2

"""The input ensembles: what goes IN to a task.

Since the target is COMPUTED from the input by the teacher, the input is the only free choice in
making a task -- and the choice is not open-ended. System identification settled it: an input is
good for identifying a system exactly to the extent that it EXCITES the system's modes, and the
short list below is the list of ways to do that, each with a different trade between how much it
excites and how realistic it looks.

    impulse               everything at once, for one instant -- the impulse response h(t)
    step                  DC gain, rise time, overshoot, settling: what tau IS, read directly
    chirp                 one frequency at a time, swept -- magnitude and phase, per frequency
    band_limited_noise    the whole band at once, continuously -- the best signal per trial
    prbs                  a flat spectrum at bounded amplitude -- the classical ID input
    trajectory            the task's own statistics, when realism is the point

WHICH TO USE IS DECIDED BY WHAT YOU WANT TO KNOW, NOT BY WHICH LOOKS MOST LIFELIKE. `step` and
`impulse` are for reading a constant off a trace; `chirp` is for MEASURING a fitted circuit's
transfer function afterwards and comparing it with the teacher's; `band_limited_noise` and
`prbs` are for fitting, because they put power everywhere the system can respond and so leave
the fewest unconstrained directions. `trajectory` is for the case where the ensemble itself is
the claim -- a target that moves like prey moves.

THE ONE PROPERTY THAT MATTERS IS PERSISTENT EXCITATION. A mode the input never excites cannot be
identified, however long the fit runs, and no amount of data repairs it. That is a property of
the STIMULUS ALONE -- its power spectrum against the frequencies the teacher's poles occupy --
so it is computable before a single training step, and `generate.py` computes it and writes it
into the provenance rather than leaving it assumed. The alternative is what already happened
once in this lineage: a 12 Hz oscillation that the objective could not see, discovered after
five training runs rather than predicted before any.

Every process has the same signature and none of them knows what it will be fed to:

    f(rng, T, dt, channels, **params) -> [T, channels] float64

`rng` is a `numpy.random.Generator` seeded per trial by the caller, so a corpus is a function of
its spec and its seeds and of nothing else.
"""
from __future__ import annotations

import numpy as np

from plexus.tasks import register_stimulus


def _t(T, dt):
    return np.arange(T, dtype=np.float64) * dt


# --------------------------------------------------------------------------- #
#  the deterministic probes
# --------------------------------------------------------------------------- #
@register_stimulus("impulse", family="probe", deterministic=True)
def impulse(rng, T, dt, channels, at_s=0.0, amplitude=1.0, **_):
    """A single sample of height `amplitude/dt`, so the impulse has unit AREA at any dt.

    Dividing by dt is what makes this a discrete approximation of a Dirac delta rather than of a
    one-sample pulse: the area is `amplitude` whatever the sampling rate, so the response is the
    impulse response h(t) and does not change scale when the corpus is resampled. `at_s` is the
    onset in seconds, and is nonzero when a baseline before the event is wanted.
    """
    u = np.zeros((T, channels))
    k = int(round(float(at_s) / dt))
    if not 0 <= k < T:
        raise ValueError(f"impulse at_s={at_s} s is outside the trial (0 to {T * dt:.3f} s)")
    u[k, :] = float(amplitude) / dt
    return u


@register_stimulus("step", family="probe", deterministic=True)
def step(rng, T, dt, channels, at_s=0.0, amplitude=1.0, **_):
    """Zero, then `amplitude`, held to the end of the trial.

    The most readable probe there is: the steady value IS the DC gain times the amplitude, the
    time to 63 % of it IS tau for a first-order system, and any overshoot is second-order or
    higher. A step excites low frequencies strongly and high ones only at the edge, so it is a
    good READOUT and a poor fitting signal.
    """
    u = np.zeros((T, channels))
    k = int(round(float(at_s) / dt))
    u[k:, :] = float(amplitude)
    return u


@register_stimulus("chirp", family="probe", deterministic=True)
def chirp(rng, T, dt, channels, f0_hz=0.05, f1_hz=None, amplitude=1.0, method="logarithmic", **_):
    """A sine sweeping f0 -> f1 over the trial: one frequency at a time, all of them by the end.

    This is the measuring instrument. Drive a fitted circuit with it, take the ratio of output to
    input spectra, and you have its transfer function -- magnitude and phase, per frequency --
    which can be compared with the teacher's H(s) pole by pole. That is a far stronger statement
    than a loss value, and it is the reason a task built on a Laplace teacher can be VERIFIED and
    not merely scored.

    `f1_hz` defaults to the Nyquist frequency 1/(2 dt), the highest frequency the sampling can
    represent at all. `method` is `logarithmic` (equal time per octave -- what you want when the
    poles span decades) or `linear`.
    """
    from scipy.signal import chirp as _chirp
    f1 = float(f1_hz) if f1_hz is not None else 0.5 / dt
    x = _chirp(_t(T, dt), f0=float(f0_hz), t1=(T - 1) * dt, f1=f1, method=str(method))
    return float(amplitude) * np.repeat(x[:, None], channels, axis=1)


# --------------------------------------------------------------------------- #
#  the fitting signals
# --------------------------------------------------------------------------- #
@register_stimulus("band_limited_noise", family="fit", deterministic=False)
def band_limited_noise(rng, T, dt, channels, f_max_hz=2.0, f_min_hz=0.0, amplitude=1.0,
                       independent=True, **_):
    """Gaussian noise with its power confined to [f_min, f_max], built in the frequency domain.

    FILTERED IN THE SPECTRUM, NOT WITH AN IIR FILTER, and the difference is not cosmetic: drawing
    the Fourier coefficients directly gives EXACTLY zero power outside the band and no transient
    at the start of the trial, whereas filtering white noise leaves both a skirt beyond the
    corner and a settling period at t = 0 that every trial then shares. A corpus in which every
    trial begins with the same artefact teaches the circuit that artefact.

    The result is scaled to unit standard deviation and then by `amplitude`, so `amplitude` is
    the signal's RMS in the units of whatever block it is written into -- not a peak, which for
    a Gaussian would be unbounded.

    `independent` draws each channel separately (two uncorrelated velocity components); False
    gives every channel the same realisation, which is what a one-dimensional task wants when it
    is still carrying a two-channel input for shape reasons.
    """
    f = np.fft.rfftfreq(T, d=dt)
    keep = (f >= float(f_min_hz)) & (f <= float(f_max_hz))
    if not keep.any():
        raise ValueError(
            f"band [{f_min_hz}, {f_max_hz}] Hz contains no representable frequency at dt={dt} "
            f"(resolution {1.0 / (T * dt):.4f} Hz, Nyquist {0.5 / dt:.2f} Hz). The trial is too "
            f"short for the band, or the band is above Nyquist.")
    n_draw = channels if independent else 1
    out = np.empty((T, n_draw))
    for c in range(n_draw):
        spec = np.zeros(f.shape, complex)
        n = int(keep.sum())
        spec[keep] = rng.normal(size=n) + 1j * rng.normal(size=n)
        x = np.fft.irfft(spec, n=T)
        sd = x.std()
        out[:, c] = x / sd if sd > 0 else x
    if not independent:
        out = np.repeat(out, channels, axis=1)
    return float(amplitude) * out


@register_stimulus("prbs", family="fit", deterministic=False)
def prbs(rng, T, dt, channels, hold_s=0.25, amplitude=1.0, independent=True, **_):
    """A binary signal switching between +/- amplitude, holding each level for `hold_s`.

    The classical system-identification input, and it is classical for a reason that matters
    here: it has a nearly flat spectrum up to about 1/(2 hold_s) while never leaving +/-
    amplitude, so it delivers the most excitation available under a hard bound on the drive. A
    muscle drive, a current, a concentration -- anything with a physical ceiling -- is fitted
    better by PRBS than by Gaussian noise of the same peak, because the Gaussian spends most of
    its time well inside the bound.

    `hold_s` sets the bandwidth: shorter holds push power higher in frequency. It must be at
    least one sample, and a hold shorter than a couple of samples is a square wave at Nyquist
    rather than a broadband signal, so it is refused.
    """
    n_hold = int(round(float(hold_s) / dt))
    if n_hold < 2:
        raise ValueError(
            f"hold_s={hold_s} s is {n_hold} samples at dt={dt}. Below two samples PRBS is a "
            f"square wave at the Nyquist frequency, not a broadband input; lengthen the hold or "
            f"shorten dt.")
    n_seg = int(np.ceil(T / n_hold))
    n_draw = channels if independent else 1
    levels = rng.integers(0, 2, size=(n_seg, n_draw)) * 2.0 - 1.0
    x = np.repeat(levels, n_hold, axis=0)[:T]
    if not independent:
        x = np.repeat(x, channels, axis=1)
    return float(amplitude) * x


# --------------------------------------------------------------------------- #
#  the naturalistic ensemble
# --------------------------------------------------------------------------- #
@register_stimulus("trajectory", family="naturalistic", deterministic=False)
def trajectory(rng, T, dt, channels, speed=0.4, turn_rate_hz=0.5, smooth_s=0.3,
               amplitude=1.0, **_):
    """The VELOCITY of a point wandering at roughly constant speed with smooth turns.

    For the case where the ensemble is itself the claim -- a target that moves the way a thing
    being tracked moves. The heading performs a smoothed random walk at `turn_rate_hz` and the
    speed is held near `speed`, so the output is a velocity of roughly constant magnitude whose
    DIRECTION varies: which is what distinguishes a pursued object from band-limited noise, and
    what makes the tracking problem about turns rather than about amplitude.

    `smooth_s` is the correlation time of the heading; below it the path is straight. Note this
    is a KINEMATIC generator and not the piecewise-waypoint one of the dot-tracking corpus --
    that corpus crosses shape, motion, speed and angle as explicit conditions, which belongs in
    a task spec's `conditions:` grid rather than inside one process.
    """
    n_smooth = max(1, int(round(float(smooth_s) / dt)))
    steps = rng.normal(size=T) * float(turn_rate_hz) * np.sqrt(dt) * 2 * np.pi
    kern = np.ones(n_smooth) / n_smooth
    heading = np.cumsum(np.convolve(steps, kern, mode="same"))
    v = float(speed) * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    if channels == 1:
        v = v[:, :1]
    elif channels > 2:
        v = np.concatenate([v, np.zeros((T, channels - 2))], axis=1)
    return float(amplitude) * v

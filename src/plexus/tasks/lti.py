"""The teacher laws: what the target IS, given the stimulus.

A task's target is not generated, it is COMPUTED -- `y = teacher(u)` -- so the teacher is the
whole of the task's scientific content. "The target is the leaky integral of velocity with
tau = 8 s" is a hypothesis about what a circuit must do, and every law here says such a thing in
a form that can be read, compared between tasks, and checked against a fit afterwards.

    laplace    a rational transfer function H(s) = num(s)/den(s), given outright
    lti        the same, named: butter / cheby1 / cheby2 / ellip / bessel, low/high/band/stop
    statespace A, B, C, D -- the MIMO form, C inputs to K outputs with coupled dynamics
    integrate  1/s, the perfect integrator, which no filter family can express
    delay      a pure time shift, exactly, by indexing rather than by a Pade approximant
    gain       a constant, the trivial law, kept because a baseline should be expressible

WHY LINEAR AND TIME-INVARIANT IS THE RIGHT PLACE TO START, despite a circuit with tanh in it
being neither. Because an LTI teacher comes with its own ANALYTIC answer. Its poles say what
time constants a fitted circuit must have; its frequency response says what a chirp must measure
coming out; and its pole frequencies say which parts of the input band carry information, so
whether a stimulus can identify it at all is decidable before any training happens. A teacher
you can only evaluate gives you a loss curve; a teacher you can solve gives you a verdict.

The eye already in this repository is such a law: `organ_mechanics` is
u_ddot + C u_dot + K u = K u_inf, i.e. H(s) = K/(s^2 + Cs + K), a second-order low-pass at
omega ~ 6.4 rad/s and damping ratio ~0.40. So the oculomotor rig is not a separate kind of thing
from a task defined here -- it is one entry, at particular parameters.

WHAT IS NOT HERE. Static nonlinear laws (logic gates, thresholds, categorisation) are memoryless
and have no H(s); they belong in a sibling module, and the union of the two -- a static map
composed with an LTI block -- is the Hammerstein cascade the eye plant already is. Nothing about
this module's interface needs to change to add them.

SISO-PER-CHANNEL VERSUS GENUINELY MIMO, because the difference is silent and matters. `laplace`,
`lti`, `integrate` and `delay` apply ONE H(s) to EACH channel independently: a task with three
channels is three copies of one problem, not a three-input system. That is the right reading for
a filter bank and the wrong one for anything whose inputs interact -- and the eye is the latter,
since its C and K are full 3x3 matrices and a horizontal command leaks into torsion through the
dynamics themselves. `statespace` is the form that can say so, and every transfer matrix is one,
so nothing is given up by making it the MIMO entry point rather than nesting polynomials.

Every law has the same signature:

    f(u [N, T, C], dt, **params) -> y [N, T, K]

and exposes two things `generate.py` reads:

    .poles(dt, **params)     the continuous-time poles, to decide whether a stimulus excites them
    .n_targets(channels, **params)  K, so the target width is DERIVED from the law rather than
                             declared in the spec and trusted. A spec that states a width the law
                             does not produce is refused instead of silently reshaped.
"""
from __future__ import annotations

import numpy as np

from plexus.tasks import register_teacher

# The analog IIR prototypes scipy can design, and the bands they can be designed into. Crossed
# with an order, this is 160 distinct H(s) before any parameter is chosen -- every one of them a
# canonical R-L-C network reduced to its transfer function, which is what makes an existing
# filter-design library the right source for a task vocabulary and a circuit SIMULATOR the wrong
# one: a simulator would hand back a trajectory from its own integrator, and the analytic answer
# is the entire point.
FAMILIES = ("butter", "cheby1", "cheby2", "ellip", "bessel")
BANDS = ("lowpass", "highpass", "bandpass", "bandstop")


def _lsim(num, den, u, dt):
    """Apply H(s) = num/den to every trial, exactly, by zero-order hold.

    `scipy.signal.lsim` discretises the continuous system and steps it with a matrix exponential,
    which is EXACT for an input held constant between samples -- which is what a sampled stimulus
    is. So the target carries no integration error of its own, and a circuit fitted to it is
    being compared against the law rather than against a solver's approximation of the law. This
    is the property that would be given up by generating targets with a circuit simulator.
    """
    from scipy.signal import lsim, lti as _lti
    sys = _lti(np.atleast_1d(num), np.atleast_1d(den))
    N, T, C = u.shape
    t = np.arange(T) * dt
    out = np.empty((N, T, C), np.float64)
    for n in range(N):
        for c in range(C):
            _, y, _ = lsim(sys, U=u[n, :, c], T=t)
            out[n, :, c] = y
    return out


def _poles_of(num, den):
    return np.roots(np.atleast_1d(den))


# --------------------------------------------------------------------------- #
@register_teacher("laplace", family="lti")
def laplace(u, dt, num=(1.0,), den=(1.0, 1.0), **_):
    """H(s) = num(s)/den(s), coefficients highest power first.

        den: [8.0, 1.0]  num: [1.0]      ->  1/(8s + 1)      a leaky integrator, tau = 8 s
        den: [1.0, 0.8, 1.0]             ->  a resonator at omega = 1, damping ratio 0.4
        den: [1.0, 0.0]  num: [1.0]      ->  1/s             a perfect integrator

    The general form, for when a law is easier to write as a polynomial than to name. Everything
    else in this module is a way of CONSTRUCTING one of these.
    """
    return _lsim(num, den, u, dt)


laplace.poles = lambda dt, num=(1.0,), den=(1.0, 1.0), **_: _poles_of(num, den)
laplace.n_targets = lambda channels, **_: channels        # SISO applied per channel


@register_teacher("lti", family="lti")
def lti(u, dt, family="butter", band="lowpass", order=2, cutoff_hz=1.0, rp=1.0, rs=40.0, **_):
    """A named analog filter: `family` of `order`, as a `band` at `cutoff_hz`.

    `cutoff_hz` is a scalar for low/high and a pair for band/stop. `rp` is the passband ripple in
    dB (Chebyshev I, elliptic) and `rs` the stopband attenuation in dB (Chebyshev II, elliptic);
    both are ignored by the families that do not take them.

    ORDER IS A DIFFICULTY AXIS AND IT IS AN HONEST ONE: order N is exactly how many poles a
    fitted circuit has to reproduce, so the same task at N = 2 and N = 8 differs in a way that is
    stated rather than guessed at. Bessel is worth singling out -- it is the family with maximally
    flat GROUP DELAY, so it distorts a waveform's shape least, which makes it the right teacher
    when the question is whether a circuit preserves timing rather than whether it attenuates.
    """
    from scipy.signal import iirfilter
    kw = {}
    if family in ("cheby1", "ellip"):
        kw["rp"] = float(rp)
    if family in ("cheby2", "ellip"):
        kw["rs"] = float(rs)
    wn = (2 * np.pi * np.asarray(cutoff_hz, float))          # rad/s: analog design is in rad/s
    num, den = iirfilter(int(order), wn, btype=band, ftype=family, analog=True, output="ba", **kw)
    return _lsim(num, den, u, dt)


def _lti_poles(dt, family="butter", band="lowpass", order=2, cutoff_hz=1.0, rp=1.0, rs=40.0, **_):
    from scipy.signal import iirfilter
    kw = {}
    if family in ("cheby1", "ellip"):
        kw["rp"] = float(rp)
    if family in ("cheby2", "ellip"):
        kw["rs"] = float(rs)
    wn = (2 * np.pi * np.asarray(cutoff_hz, float))
    _num, den = iirfilter(int(order), wn, btype=band, ftype=family, analog=True, output="ba", **kw)
    return np.roots(den)


lti.poles = _lti_poles
lti.n_targets = lambda channels, **_: channels            # SISO applied per channel


@register_teacher("integrate", family="lti")
def integrate(u, dt, tau_s=None, gain=1.0, **_):
    """The integrator, leaky when `tau_s` is given and perfect when it is not.

        tau_s given:  dy/dt = -y/tau + gain * u      H(s) = gain / (s + 1/tau)
        tau_s None:   dy/dt =          gain * u      H(s) = gain / s

    Named separately from `laplace` because it is the task this whole lineage exists for -- the
    oculomotor integrator converts an eye VELOCITY command into an eye ANGLE, and how long it can
    hold that angle is what tau measures. A perfect integrator has no steady state and its output
    grows without bound, which is deliberate: a circuit that cannot hold will show its leak as a
    growing gap, and that gap is the measurement.
    """
    if tau_s is None or float(tau_s) <= 0:
        return _lsim([float(gain)], [1.0, 0.0], u, dt)
    a = 1.0 / float(tau_s)
    return _lsim([float(gain)], [1.0, a], u, dt)


integrate.poles = (lambda dt, tau_s=None, gain=1.0, **_:
                   np.array([0.0]) if tau_s is None or float(tau_s) <= 0
                   else np.array([-1.0 / float(tau_s)]))
integrate.n_targets = lambda channels, **_: channels


@register_teacher("delay", family="lti")
def delay(u, dt, seconds=0.1, **_):
    """A pure time shift: y(t) = u(t - T), zero before it.

    BY INDEXING, NOT BY A PADE APPROXIMANT. `e^{-sT}` is not rational, so putting a delay through
    `laplace` means approximating it with a rational function whose phase is right only up to
    some frequency and which overshoots at the step. Since a sampled delay of a whole number of
    samples is exact, that approximation buys nothing here and costs the one property the LTI
    teachers are for. The delay is rounded to the nearest sample and the rounding is reported.
    """
    k = int(round(float(seconds) / dt))
    if k < 0:
        raise ValueError(f"delay seconds={seconds} is negative; a teacher may not see the future")
    y = np.zeros_like(u)
    if k < u.shape[1]:
        y[:, k:, :] = u[:, :u.shape[1] - k, :]
    return y


delay.poles = lambda dt, seconds=0.1, **_: np.array([])      # a delay has no poles
delay.n_targets = lambda channels, **_: channels


@register_teacher("gain", family="static")
def gain(u, dt, k=1.0, **_):
    """y = k u. The trivial law, and a baseline should be expressible.

    Useful for exactly one thing: establishing what a fit achieves when the task requires no
    dynamics at all. A circuit that cannot beat `gain` on an integration task has not learnt to
    integrate, whatever its loss says.
    """
    return float(k) * u


gain.poles = lambda dt, k=1.0, **_: np.array([])
gain.n_targets = lambda channels, **_: channels


# --------------------------------------------------------------------------- #
#  the MIMO form
# --------------------------------------------------------------------------- #
@register_teacher("statespace", family="lti")
def statespace(u, dt, A=None, B=None, C=None, D=None, **_):
    """x_dot = A x + B u,  y = C x + D u. C inputs to K outputs, with COUPLED dynamics.

    The general linear teacher, and the one to reach for whenever the inputs interact. A transfer
    matrix is a state-space realisation, so nothing is expressible as a matrix of polynomials
    that is not expressible here -- and writing an 18-entry 3x6 matrix of (num, den) pairs in
    yaml is a worse spec than four matrices.

    THE EYE IS THIS. `organ_mechanics` is u_ddot + C_d u_dot + K u = K u_inf with full 3x3 C_d
    and K, which as a state-space over x = (u, u_dot) is

        A = [[0, I], [-K, -C_d]]        B = [[0], [K]]        C = [I, 0]        D = 0

    and its off-diagonal terms are exactly why a horizontal command leaks into torsion. A
    diagonal, per-channel teacher cannot express that at all, which is the reason this law
    exists rather than a nesting of the polynomial ones.

    Shapes are checked rather than broadcast: A (n, n), B (n, C), C (K, n), D (K, C) or absent.
    """
    from scipy.signal import lsim, lti as _lti
    A = np.atleast_2d(np.asarray(A, float))
    B = np.atleast_2d(np.asarray(B, float))
    Cm = np.atleast_2d(np.asarray(C, float))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"statespace: A is {A.shape}, must be square")
    if B.shape[0] != n:
        raise ValueError(f"statespace: B is {B.shape}, must have {n} rows to match A")
    if Cm.shape[1] != n:
        raise ValueError(f"statespace: C is {Cm.shape}, must have {n} columns to match A")
    n_in, n_out = B.shape[1], Cm.shape[0]
    if u.shape[2] != n_in:
        raise ValueError(
            f"statespace: B takes {n_in} input(s) but the stimulus has {u.shape[2]} channel(s). "
            f"Set `general.channels: {n_in}`.")
    Dm = np.zeros((n_out, n_in)) if D is None else np.atleast_2d(np.asarray(D, float))
    sys = _lti(A, B, Cm, Dm)
    N, T, _ = u.shape
    t = np.arange(T) * dt
    out = np.empty((N, T, n_out), np.float64)
    for i in range(N):
        _, y, _ = lsim(sys, U=u[i], T=t)
        out[i] = np.atleast_2d(y).reshape(T, n_out)
    return out


statespace.poles = lambda dt, A=None, **_: np.linalg.eigvals(np.atleast_2d(np.asarray(A, float)))
statespace.n_targets = lambda channels, C=None, **_: int(np.atleast_2d(np.asarray(C, float)).shape[0])

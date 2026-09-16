"""Gates for `plexus.tasks`: the teacher laws against their ANALYTIC answers.

The point of building tasks on Laplace teachers is that the right answer is known in closed form,
so these tests do not check that a law "runs" or that its output "looks reasonable" -- they check
it against the number theory says it must produce. A first-order low-pass driven by a step must
reach 63.2 % of its final value at exactly t = tau; a resonator's ringing must be at exactly
f_n sqrt(1 - zeta^2); an integrator's output must be exactly the cumulative sum times dt. A law
that passes those is the law it claims to be.

The excitation check gets the same treatment: it is asserted to FIRE on a case constructed to be
unidentifiable, not merely to return a dict. A check that never says no has not been shown to
work.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import plexus.tasks as T
from plexus.tasks.generate import spectral_coverage
from plexus.tasks.schema import load_task

DT = 1.0 / 60.0


def _u(T_frames, channels=1, kind="step", dt=DT, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return T.get_stimulus(kind)(rng, T_frames, dt, channels, **kw)[None]     # [1, T, C]


# --------------------------------------------------------------------------- #
#  the laws, against closed form
# --------------------------------------------------------------------------- #
def test_leaky_integrator_step_response_reaches_63_percent_at_one_tau():
    """The defining property of a first-order lag, and the one a fitted tau is read from."""
    tau, T_f = 1.0, 600
    u = _u(T_f, kind="step")
    y = T.get_teacher("integrate")(u, DT, tau_s=tau)[0, :, 0]
    final = tau                                          # H(0) = gain/(1/tau) = tau for gain 1
    k = int(round(tau / DT))
    assert y[k] / final == pytest.approx(1 - np.exp(-1.0), abs=5e-3), (
        f"at t = tau the step response is {y[k] / final:.4f} of final, must be 0.6321")
    assert y[-1] / final == pytest.approx(1.0, abs=1e-2), "it must settle at H(0)"


def test_perfect_integrator_is_the_cumulative_sum():
    """1/s applied to a sampled input is the running integral, to the quadrature's own accuracy."""
    T_f = 300
    u = _u(T_f, kind="step")
    y = T.get_teacher("integrate")(u, DT)[0, :, 0]
    exact = np.cumsum(u[0, :, 0]) * DT
    # zero-order hold integrates each sample over its own interval, so it leads the left-hand
    # Riemann sum by exactly one sample; compare where they are defined to agree
    assert np.abs(y[1:] - exact[:-1]).max() < 1e-9, "1/s is not integrating"


def test_resonator_rings_at_the_damped_frequency():
    """f_d = f_n sqrt(1 - zeta^2), read off the zero crossings of the impulse response."""
    f_n, zeta, T_f = 2.0, 0.1, 1200
    wn = 2 * np.pi * f_n
    u = _u(T_f, kind="impulse")
    y = T.get_teacher("laplace")(u, DT, num=[wn ** 2], den=[1.0, 2 * zeta * wn, wn ** 2])[0, :, 0]
    zc = np.where(np.diff(np.signbit(y[5:])))[0]         # skip the onset sample
    assert len(zc) > 6, "the impulse response does not ring"
    half_period = np.median(np.diff(zc)) * DT
    f_meas = 0.5 / half_period
    f_d = f_n * np.sqrt(1 - zeta ** 2)
    assert f_meas == pytest.approx(f_d, rel=0.02), (
        f"rings at {f_meas:.3f} Hz, theory says f_d = {f_d:.3f} Hz")


def test_delay_is_exact_and_shifts_by_whole_samples():
    T_f, shift = 200, 0.1
    u = _u(T_f, kind="prbs", hold_s=0.2, seed=3)
    y = T.get_teacher("delay")(u, DT, seconds=shift)
    k = int(round(shift / DT))
    assert np.array_equal(y[0, k:, 0], u[0, :T_f - k, 0]), "a delay must be an exact shift"
    assert np.all(y[0, :k, 0] == 0), "and zero before it"


def test_statespace_reproduces_the_eye_plants_own_modes():
    """The MIMO law, against C and K of the fitted eye computed independently.

    x = (u, u_dot), A = [[0, I], [-K, -C]]. Its eigenvalues must be the plant's, so the damped
    frequencies must equal f_n sqrt(1 - zeta^2) with f_n from K and zeta from C -- three coupled
    modes, not three independent copies.
    """
    C = np.array([[4.619, -0.045, 0.173], [-0.045, 5.098, 0.046], [0.173, 0.046, 5.712]])
    K = np.array([[37.499, 6.344, 4.904], [6.344, 44.668, 5.353], [4.904, 5.353, 42.699]])
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-K, -C]])
    poles = T.get_teacher("statespace").poles(DT, A=A)
    f_meas = np.sort([abs(p.imag) / (2 * np.pi) for p in poles if p.imag > 0])

    wn = np.sqrt(np.linalg.eigvalsh(K))
    zeta = np.linalg.eigvalsh(C) / (2 * wn)
    f_expect = np.sort(wn * np.sqrt(1 - zeta ** 2) / (2 * np.pi))
    assert np.allclose(f_meas, f_expect, rtol=0.02), (
        f"state-space modes {np.round(f_meas, 3)} Hz against {np.round(f_expect, 3)} Hz from C, K")
    assert T.get_teacher("statespace").n_targets(3, C=np.hstack([np.eye(3), np.zeros((3, 3))])) == 3


def test_a_mimo_law_refuses_a_channel_count_it_cannot_take():
    A = np.array([[-1.0, 0.0], [0.0, -2.0]])
    B = np.array([[1.0], [1.0]])                        # ONE input
    C = np.array([[1.0, 1.0]])
    with pytest.raises(ValueError, match="channel"):
        T.get_teacher("statespace")(np.zeros((1, 10, 3)), DT, A=A, B=B, C=C)   # three given


# --------------------------------------------------------------------------- #
#  the stimulus ensembles
# --------------------------------------------------------------------------- #
def test_band_limited_noise_has_no_power_outside_its_band():
    """Built in the spectrum, so the band edge is exact rather than a filter's skirt."""
    T_f, f_max = 1024, 2.0
    u = _u(T_f, kind="band_limited_noise", f_max_hz=f_max, seed=1)
    f = np.fft.rfftfreq(T_f, d=DT)
    p = np.abs(np.fft.rfft(u[0, :, 0])) ** 2
    assert p[f > f_max * 1.001].max() < 1e-18 * max(p.max(), 1e-30), "power leaks past the band"
    assert p[(f > 0) & (f <= f_max)].max() > 0, "and there is power inside it"


def test_impulse_carries_unit_area_at_any_dt():
    """Height 1/dt, so the response is the impulse response and not dt-dependent."""
    for dt in (1 / 60, 1 / 600):
        u = _u(120, kind="impulse", dt=dt)
        assert (u.sum() * dt) == pytest.approx(1.0), f"area is not 1 at dt={dt}"


def test_prbs_refuses_a_hold_shorter_than_two_samples():
    with pytest.raises(ValueError, match="Nyquist"):
        _u(100, kind="prbs", hold_s=DT * 0.5)


# --------------------------------------------------------------------------- #
#  the excitation check -- it must SAY NO
# --------------------------------------------------------------------------- #
def test_excitation_fires_on_a_pole_the_stimulus_cannot_reach():
    """THE NEGATIVE CONTROL. 12 Hz resonance, 0.5 Hz stimulus -- the FINDINGS limit cycle.

    A check that never returns False has not been shown to work, so the case it must catch is
    asserted directly rather than left to a corpus.
    """
    T_f = 480
    u = _u(T_f, kind="band_limited_noise", f_max_hz=0.5, seed=2)
    poles = T.get_teacher("laplace").poles(DT, num=[5685.0], den=[1.0, 9.4, 5685.0])
    rep = spectral_coverage(u, DT, poles)
    assert not rep["identifiable"], "a 12 Hz pole under a 0.5 Hz stimulus must be unidentifiable"
    assert rep["poles"][0]["freq_hz"] == pytest.approx(12.0, rel=0.02)


def test_excitation_passes_when_the_stimulus_covers_the_pole():
    T_f = 480
    u = _u(T_f, kind="band_limited_noise", f_max_hz=20.0, seed=2)
    poles = T.get_teacher("laplace").poles(DT, num=[5685.0], den=[1.0, 9.4, 5685.0])
    assert spectral_coverage(u, DT, poles)["identifiable"]


# --------------------------------------------------------------------------- #
#  the spec language
# --------------------------------------------------------------------------- #
def _write(tmp_path, body):
    import yaml
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(body))
    return str(p)


BASE = {"general": {"name": "t", "duration_s": 1.0, "dt": DT, "channels": 1},
        "stimulus": {"process": "band_limited_noise", "f_max_hz": 2.0},
        "teacher": {"law": "integrate", "tau_s": 8.0},
        "splits": {"train": {"n_per_cond": 4, "seed0": 0},
                   "test": {"n_per_cond": 2, "seed0": 1000}}}


def test_a_spec_with_overlapping_split_seeds_is_refused():
    """Splits that share seeds share TRIALS, and a val score on training data means nothing."""
    import copy
    bad = copy.deepcopy(BASE)
    bad["splits"]["test"]["seed0"] = 2            # train covers [0, 4)
    with pytest.raises(ValueError, match="share seeds"):
        load_task(_write(pytest.importorskip("pathlib") and __import__("pathlib").Path(
            os.environ.get("PYTEST_TMPDIR", "/tmp")), bad))


def test_a_condition_naming_nothing_is_refused(tmp_path):
    import copy
    bad = copy.deepcopy(BASE)
    bad["conditions"] = {"not_a_parameter": [1, 2]}
    with pytest.raises(ValueError, match="name no parameter"):
        load_task(_write(tmp_path, bad))


def test_an_unregistered_law_is_refused(tmp_path):
    import copy
    bad = copy.deepcopy(BASE)
    bad["teacher"] = {"law": "telepathy"}
    with pytest.raises(KeyError, match="teacher law"):
        load_task(_write(tmp_path, bad))


def test_the_grid_is_the_full_cross_product(tmp_path):
    import copy
    ok = copy.deepcopy(BASE)
    ok["conditions"] = {"tau_s": [1.0, 8.0], "f_max_hz": [0.5, 2.0, 8.0]}
    spec = load_task(_write(tmp_path, ok))
    assert len(spec.cells) == 6
    assert spec.n_trials("train") == 6 * 4


def test_registries_are_populated_and_every_law_reports_its_poles_and_width():
    assert {"laplace", "lti", "integrate", "delay", "gain", "statespace"} <= set(T.teachers())
    assert {"impulse", "step", "chirp", "band_limited_noise", "prbs", "trajectory"} <= set(T.stimuli())
    for n in T.teachers():
        f = T.get_teacher(n)
        assert hasattr(f, "poles") and hasattr(f, "n_targets"), f"{n} is missing a contract method"


def test_a_name_cannot_be_registered_twice():
    """Two laws under one name is the defect a registry exists to prevent."""
    with pytest.raises(ValueError, match="already registered"):
        T.register_teacher("integrate")(lambda u, dt, **k: u)

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


# --------------------------------------------------------------------------- #
#  the trainer's circuit, against the registered operators
# --------------------------------------------------------------------------- #
def test_trainer_circuit_is_arithmetically_the_operators():
    """`CircuitRNN` claims to be `project -> neuron_update -> neuron_signal -> readout`.

    That claim is what licenses training outside the engine, so it is asserted against the
    REGISTERED operators rather than restated in a docstring. If the two ever diverge, a spec and
    the thing fitted to it stop describing one model and nothing downstream would notice.
    """
    import torch
    import plexus.operators                                    # noqa: F401  self-registers
    from plexus.models.registry import get_operator
    from plexus.tasks.trainer import CircuitRNN

    torch.manual_seed(0)
    H_UNITS, N_IN, N_OUT, T_F, dt = 6, 2, 1, 40, 1 / 60
    m = CircuitRNN(N_IN, N_OUT, hidden=H_UNITS, tau0=0.25, dt=dt, w_init="random", seed=1)
    with torch.no_grad():                      # a zero W would make the recurrent half vacuous
        m.W.copy_(torch.randn(H_UNITS, H_UNITS) * 0.3)
    u = torch.randn(1, T_F, N_IN) * 0.5
    ref = m(u)[0].detach().numpy()

    # the same three steps, each taken from the operator that owns it
    class _L:
        def __init__(s, n, sch, wdt, name):
            s.n, s.name = n, name
            s.state = torch.zeros(n, wdt); s.state_schema = sch; s.occ = torch.ones(n)
        def get(s, b):
            a, z = s.state_schema[b]; return s.state[:, a:z]

    class _E(_L):
        def __init__(s, pre, post, w, pre_name, post_name):
            super().__init__(len(w), {"w": (0, 1)}, 1, "edges")
            s.pre, s.post = torch.as_tensor(pre), torch.as_tensor(post)
            s.pre_name, s.post_name = pre_name, post_name
            s.state[:, 0] = torch.as_tensor(w, dtype=torch.float32)
        def incidence(s, role): return s.pre if role == "pre" else s.post
        def incidence_name(s, role): return s.pre_name if role == "pre" else s.post_name

    class _H:
        def __init__(s, lv): s.levels = lv; s.config = type("C", (), {"dt": dt})()
        def level(s, n): return s.levels[n]
        def gather(s, es, role, blk):
            e = s.levels[es]; return s.levels[e.incidence_name(role)].get(blk)[e.incidence(role)]
        def scatter_along(s, es, role, vals):
            e = s.levels[es]; ep = s.levels[e.incidence_name(role)]
            out = torch.zeros(ep.n, vals.shape[-1]); out.index_add_(0, e.incidence(role), vals)
            return out

    src = _L(N_IN, {"signal": (0, 1)}, 1, "src")
    neu = _L(H_UNITS, {"voltage": (0, 1), "tau": (1, 2)}, 2, "neuron")
    out = _L(N_OUT, {"y": (0, 1)}, 1, "out")
    with torch.no_grad():
        neu.state[:, 1] = m.log_tau.exp()
    aff = _E(*zip(*[(j, i) for i in range(H_UNITS) for j in range(N_IN)]),
             [float(m.W_in[i, j].detach()) for i in range(H_UNITS) for j in range(N_IN)], "src", "neuron")
    rec = _E(*zip(*[(j, i) for i in range(H_UNITS) for j in range(H_UNITS)]),
             [float(m.W[i, j].detach()) for i in range(H_UNITS) for j in range(H_UNITS)], "neuron", "neuron")
    jnc = _E(*zip(*[(j, k) for k in range(N_OUT) for j in range(H_UNITS)]),
             [float(m.W_out[k, j].detach()) for k in range(N_OUT) for j in range(H_UNITS)], "neuron", "out")
    Hh = _H({"src": src, "neuron": neu, "out": out, "aff": aff, "rec": rec, "jnc": jnc})

    proj = get_operator("project")({"_at": "neuron", "edge_set": "aff", "block": "signal"})
    upd = get_operator("neuron_update")({"_at": "neuron", "tau": "tau"})
    sig = get_operator("neuron_signal", model="shared")(
        {"_at": "neuron", "edge_set": "rec", "activation": "tanh"})
    rd = get_operator("readout")({"_at": "out", "edge_set": "jnc", "block": "voltage",
                                  "into": "y", "send": "tanh"})
    got = []
    for t in range(T_F):
        src.state[:, 0] = u[0, t]
        # the SAME 1/tau scales the leak, the input and the recurrent message, as the reference
        # writes it: v += (dt/tau)(-v + W r + I)
        inv_tau = (1.0 / neu.get("tau"))
        d = (proj.forward(Hh)["neuron"] * inv_tau + upd.forward(Hh)["neuron"]
             + sig.forward(Hh)["neuron"] * inv_tau)
        rd.forward(Hh)
        got.append(float(out.get("y")[0, 0]))
        neu.state[:, 0:1] = neu.state[:, 0:1] + dt * d
    got = np.asarray(got)
    err = np.abs(got - ref[:, 0]).max()
    assert np.abs(ref).max() > 0.01, "the reference barely moves; the comparison would be vacuous"
    assert err < 1e-5, (
        f"CircuitRNN departs from project/neuron_update/neuron_signal/readout by {err:.3e}. The "
        f"trainer's claim to be evaluating the spec is then false.")


# --------------------------------------------------------------------------- #
#  the R-L-C realisation
# --------------------------------------------------------------------------- #
def test_a_second_order_pole_pair_gives_back_its_own_w0_and_zeta():
    """The ladder is derived from the poles, so it must reproduce them.

    A series RLC has w0 = 1/sqrt(LC) and zeta = (R/2)sqrt(C/L). Deriving L and R from a pole
    pair and then recomputing those two is a round trip: if it closes, the schematic is of the
    teacher and not of something adjacent to it.
    """
    from plexus.tasks.lti import rlc_stages
    f_n, zeta = 1.0, 0.4
    wn = 2 * np.pi * f_n
    poles = T.get_teacher("laplace").poles(
        DT, num=[wn ** 2], den=[1.0, 2 * zeta * wn, wn ** 2])
    st = rlc_stages(poles)
    assert len(st) == 1, "a conjugate PAIR is one section, not two"
    s0 = st[0]
    assert s0["kind"] == "RLC"
    w0_back = 1.0 / np.sqrt(s0["L"] * s0["C"])
    zeta_back = (s0["R"] / 2) * np.sqrt(s0["C"] / s0["L"])
    assert w0_back == pytest.approx(wn, rel=1e-6), "L, C do not give back w0"
    assert zeta_back == pytest.approx(zeta, rel=1e-6), "R does not give back zeta"


def test_a_real_pole_gives_an_rc_whose_time_constant_is_right():
    from plexus.tasks.lti import rlc_stages
    tau = 8.0
    st = rlc_stages(T.get_teacher("integrate").poles(DT, tau_s=tau))
    assert len(st) == 1 and st[0]["kind"] == "RC"
    assert st[0]["R"] * st[0]["C"] == pytest.approx(tau, rel=1e-9), "RC is not tau"


def test_a_pole_at_the_origin_is_not_a_passive_network():
    """1/s has infinite DC gain, which no combination of R, L and C provides."""
    from plexus.tasks.lti import rlc_stages
    st = rlc_stages(T.get_teacher("integrate").poles(DT))
    assert len(st) == 1 and st[0]["kind"] == "integrator"
    assert "op-amp" in st[0]["note"]


def test_order_is_the_number_of_sections():
    """`order: N` on a spec is N/2 boxes on a breadboard -- the difficulty axis, made literal."""
    from plexus.tasks.lti import rlc_stages
    for order, expect in ((2, 1), (4, 2), (8, 4)):
        st = rlc_stages(T.get_teacher("lti").poles(
            DT, family="butter", band="lowpass", order=order, cutoff_hz=1.0))
        assert len(st) == expect, f"order {order} should be {expect} section(s), got {len(st)}"


def test_engineering_notation_reads_as_a_component():
    from plexus.tasks.lti import eng
    assert eng(4700.0, "Ω").startswith("4.7 k")
    assert eng(1e-6, "F").startswith("1 u")
    assert eng(0.22, "H").startswith("220 m")

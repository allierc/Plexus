"""exp18's rulers on planted inputs: each reads what is planted."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from exp_measures import exp18 as R                      # noqa: E402
from exp_measures.common import TrainingRun              # noqa: E402


def test_coupling_at_zero_phase_and_angle_is_the_scaled_connectome():
    W = np.array([[0.0, 2.0], [-1.0, 0.0]])
    J = R.coupling(W, np.zeros((1, 1)), [0, 0], [0, 0], 0.0, [3.0, 1.0], [1.0, 0.5])
    assert np.allclose(J, [[0.0, 3.0 * 2.0 * 0.5], [-1.0, 0.0]])


def test_coupling_reads_the_pair_phase_sender_then_receiver():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    phi = np.array([[0.0, 0.9], [0.2, 0.0]])            # phi[type_pre, type_post]
    J = R.coupling(W, phi, [0, 1], [0, 1], 0.4, [1, 1], [1, 1])
    assert J[1, 0] == pytest.approx(math.cos(0.9 - 0.4))  # sender 0 (type 0) -> receiver 1 (type 1)
    assert J[0, 1] == pytest.approx(math.cos(0.2 - 0.4))


def test_gap_reads_a_planted_isolated_eigenvalue():
    assert R.gap([10.0, 2.0, 1.0, 0.0]) == pytest.approx((10 - 2) / (2 - 0))
    assert R.gap([1.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_gap_counts_a_conjugate_pair_as_one_mode():
    lam = [5 + 2j, 5 - 2j, 1.0, 0.5, 0.0]
    assert R.gap(lam) == pytest.approx((5 - 1) / (1 - 0))


def test_half_turn_identity_holds_on_a_random_pencil():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(12, 12))
    phi = rng.uniform(0, 2 * np.pi, size=(3, 3))
    t = rng.integers(0, 3, size=12)
    J0 = R.coupling(W, phi, t, t, 0.7, np.ones(12), np.ones(12))
    J1 = R.coupling(W, phi, t, t, 0.7 + np.pi, np.ones(12), np.ones(12))
    assert R.halfturn_err(J0, J1) < 1e-10
    J2 = R.coupling(W, phi, t, t, 0.7 + 0.5, np.ones(12), np.ones(12))
    assert R.halfturn_err(J0, J2) > 1e-3


def test_overlap_and_circular_distance():
    assert R.overlap(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert R.overlap(np.array([1.0, 1j]), np.array([1j, -1.0])) == pytest.approx(1.0)
    assert R.circ_dist(0.1, 2 * np.pi - 0.1) == pytest.approx(0.2)


def test_fit_reads_the_worst_law_and_the_diverged_fraction(tmp_path):
    d = tmp_path / "training" / "neural" / "r1"
    (d / "results").mkdir(parents=True)
    (d / "config.yaml").write_text("name: r1\n")
    (d / "model.yaml").write_text("general: {name: m}\n")
    json.dump({"cell_names": ["integrate", "delay"], "normalised_per_cell": {"0": 0.002, "1": 0.03},
               "normalised_mse": 0.016}, open(d / "results" / "r1_test.json", "w"))
    json.dump({"n_trials": 64, "batch": 32, "epochs": 5,
               "history": [{"diverged_steps": 1}, {"diverged_steps": 0}, {}, {}, {}]},
              open(d / "results" / "report.json", "w"))
    out = R.fit(TrainingRun(str(d)))
    assert out["worst_cell_nmse"] == pytest.approx(0.03)
    assert out["nmse_integrate"] == pytest.approx(0.002) and out["nmse_delay"] == pytest.approx(0.03)
    assert out["diverged_frac"] == pytest.approx(1 / 10)


def test_coupling_on_a_torus_is_the_mean_of_the_axes():
    W = np.array([[0.0, 1.0], [1.0, 0.0]])
    phi = np.array([[[0.0, 0.9], [0.2, 0.0]], [[0.0, -0.3], [1.4, 0.0]]])
    J = R.coupling(W, phi, [0, 1], [0, 1], np.array([0.4, 1.0]), [1, 1], [1, 1])
    assert J[1, 0] == pytest.approx(0.5 * (math.cos(0.9 - 0.4) + math.cos(-0.3 - 1.0)))
    Jpi = R.coupling(W, phi, [0, 1], [0, 1], np.array([0.4, 1.0]) + np.pi, [1, 1], [1, 1])
    assert np.allclose(Jpi, -J)


def test_coupling_gain_only_is_half_one_plus_cos():
    W = np.array([[0.0, 2.0], [0.0, 0.0]])
    J = R.coupling(W, np.zeros((1, 1)), [0, 0], [0, 0], np.pi, [1, 1], [1, 1], nonneg=True)
    assert J[0, 1] == pytest.approx(0.0, abs=1e-12)

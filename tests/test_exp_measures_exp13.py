"""exp13's rulers on planted inputs whose answer is known.

    PYTHONPATH=src:tools python -m pytest tests/test_exp_measures_exp13.py -q

The growth ruler is run on a tiny particle-layout trajectory.npz written to tmp_path and opened with
`open_run`, so it goes through the same facade (`cells`, `block`) as a real run.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_measures  # noqa: E402
from exp_measures import exp13  # noqa: E402
from exp_measures.common import open_run  # noqa: E402

LAM = 0.005          # the planted growth rate of the rest of the tissue, per frame


def plant(tmp_path, n_frames=400, n_live=20, n_clone=10, clone_mult=2.0, stop=None, clone=True):
    """A particle run: `n_live` live cells in a 2x larger buffer (dead slots hold junk volumes, so a
    reader that ignores occupancy is caught); the first `n_clone` are the clone. Cell volume grows as
    exp(rate t), the clone's rate `clone_mult` x LAM, until frame `stop`, flat after it."""
    t = np.arange(n_frames, dtype=float)
    te = t if stop is None else np.minimum(t, stop)
    nb = 2 * n_live
    occ = np.zeros((n_frames, nb), bool)
    occ[:, :n_live] = True
    V = np.full((n_frames, nb, 1), 99.0)
    rest = np.exp(LAM * te)
    fast = np.exp(clone_mult * LAM * te)
    V[:, :n_live, 0] = rest[:, None]
    V[:, :n_clone, 0] = fast[:, None]
    z = {"cell__pos": np.random.default_rng(0).normal(size=(n_frames, nb, 3)), "cell__occ": occ, "cell__volume": V}
    if clone:
        k = np.zeros((n_frames, nb, 1))
        k[:, :n_clone, 0] = 1.0
        k[:, n_live:, 0] = 1.0                       # dead slots flagged too: must not count
        z["cell__clone"] = k
    d = tmp_path / "run"
    d.mkdir()
    np.savez(d / "trajectory.npz", **z)
    return open_run(str(d))


def test_growth_exponential_never_arrests(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0))
    assert abs(r["rate_peak"] - LAM) < 1e-6
    assert abs(r["late_over_peak"] - 1.0) < 1e-3     # still growing as fast as it ever did
    assert r["arrest_frame"] is None
    assert r["size_cv_last"] < 1e-9                  # every live cell the same volume; dead slots ignored


def test_growth_arrest_frame_and_late_rate(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0, stop=200), window=100)
    assert abs(r["late_over_peak"]) < 1e-6           # flat after frame 200
    assert 200 <= r["arrest_frame"] <= 252           # the centred 100-frame window reaches past 200 by 50 frames
    assert abs(r["size_ratio"] - np.exp(LAM * (200 - 60))) < 1e-6


def test_clone_rate_ratio(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=2.0))
    assert r["clone.available"]
    assert abs(r["clone.rate"] - 2 * LAM) < 1e-9 and abs(r["clone.rate_rest"] - LAM) < 1e-9
    assert abs(r["clone.rate_ratio"] - 2.0) < 1e-6 and abs(r["clone.excess"] - 1.0) < 1e-6
    assert abs(r["clone.excess_late"] - 1.0) < 1e-6
    assert r["clone.cells_first"] == 10              # the flagged dead slots are not the clone


def test_clone_uniform_is_zero_excess(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=1.0))
    assert abs(r["clone.excess"]) < 1e-9


def test_clone_late_excess_sees_the_adaptation(tmp_path):
    """Shraiman Fig. 4a: the clone outgrows the rest early, then runs parallel. The late excess is 0,
    the whole-window one is not."""
    import numpy as np
    d = tmp_path / "adapt"
    d.mkdir()
    n, t = 400, np.arange(400, dtype=float)
    fast = np.exp(LAM * t + LAM * np.minimum(t, 150))       # rate 2 LAM until frame 150, LAM after
    V = np.ones((n, 20, 1))
    V[:, :, 0] = np.exp(LAM * t)[:, None]
    V[:, :10, 0] = fast[:, None]
    k = np.zeros((n, 20, 1))
    k[:, :10, 0] = 1.0
    np.savez(d / "trajectory.npz", cell__pos=np.zeros((n, 20, 3)), cell__occ=np.ones((n, 20), bool),
             cell__volume=V, cell__clone=k)
    r = exp_measures.run_measure("exp13.growth", open_run(str(d)))
    assert abs(r["clone.excess_late"]) < 1e-9 and r["clone.excess"] > 0.1


def test_clone_window_ends_at_arrest(tmp_path):
    """After arrest nobody grows; the rate is read over the growth window, not diluted by the plateau."""
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone_mult=2.0, stop=200))
    assert r["clone.window"][1] == r["arrest_frame"]
    assert abs(r["clone.rate_ratio"] - 2.0) < 1e-6


def test_no_clone_block(tmp_path):
    r = exp_measures.run_measure("exp13.growth", plant(tmp_path, clone=False))
    assert r["clone.available"] is False and "clone.excess" not in r


def test_arm_of():
    assert exp13.arm_of("tissue/exp03_g1_sizer_s4") == "g1_sizer"
    assert exp13.arm_of("tissue/exp13_clone_k2_s1") == "clone_k2"


def _table(tmp_path, rows):
    f = tmp_path / "sig.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return exp13.load_table(str(f), keys=("S1_slope", "S2_rho"))


def test_compare_identity_and_distinct(tmp_path):
    T = _table(tmp_path, [
        {"spec": "tissue/exp03_g1_sizer_s1", "S1_slope": -1.2, "S2_rho": -0.1},
        {"spec": "tissue/exp03_g1_sizer_s2", "S1_slope": -1.0, "S2_rho": -0.1},
        {"spec": "tissue/exp03_g1_timer_s1", "S1_slope": 0.7, "S2_rho": 0.8},
        {"spec": "tissue/exp03_g1_timer_s2", "S1_slope": 0.9, "S2_rho": 0.9},
        {"spec": "tissue/exp03_g1_timer_s2", "S1_slope": 0.8, "S2_rho": 0.9},   # a rerun: last line wins
    ])
    assert T["g1_timer"]["S1_slope"] == pytest.approx((0.75, 0.1))
    keys = ("S1_slope", "S2_rho")
    # sizer mean (-1.1, -0.1), spreads (0.2, 0 -> guarded); timer mean (0.75, 0.85), spreads (0.1, 0.1)
    same = exp13.compare({"S1_slope": -1.1, "S2_rho": -0.1}, T, keys=keys)
    assert same["identity_z"] == pytest.approx(0.0, abs=1e-9)
    assert same["nearest_arm"] == "g1_sizer" and same["distinct_margin"] == pytest.approx(0.0, abs=1e-9)
    near = exp13.compare({"S1_slope": -1.3, "S2_rho": -0.1}, T, keys=keys)
    assert near["identity_z"] == pytest.approx(1.0) and near["identity_key"] == "S1_slope"
    # halfway between the arms on S1, off both on S2: the nearest arm decides the margin
    new = exp13.compare({"S1_slope": 0.0, "S2_rho": 0.5}, T, keys=keys)
    assert new["nearest_arm"] == "g1_timer"
    assert new["distinct_margin"] == pytest.approx(max(0.75 / 0.2, 0.35 / 0.2))
    assert new["distinct_key"] == "S1_slope"


def test_exp03_members_are_inside_their_own_arm():
    """On exp 3's recorded table: every g1_sizer seed is within one spread of its arm's mean (identity_z
    <= 1) and not separated from its own arm (distinct_margin <= 0.5)."""
    if not os.path.exists(exp13.EXP03_TABLE):
        pytest.skip("exp03 signatures.jsonl not on disk")
    T = exp13.load_table()
    rows = [json.loads(l) for l in open(exp13.EXP03_TABLE)]
    sizer = [r for r in rows if exp13.arm_of(r["spec"]) == "g1_sizer"]
    assert len(sizer) == 4
    for r in sizer:
        c = exp13.compare(r, T)
        assert c["identity_z"] <= 1.0 + 1e-9
        assert c["distinct_margin"] <= 0.5 + 1e-9


def test_write_results_derives_the_table_source(tmp_path):
    rows = [{"run": "tissue/exp13_g0_s1", "measure": "exp13.growth", "value": {"late_over_peak": 0.5, "size_cv_last": 0.2}},
            {"run": "tissue/exp13_g0_s1", "measure": "exp13.growth", "value": {"late_over_peak": 0.9, "size_cv_last": 0.25}},
            {"run": "tissue/exp13_g0_s1", "measure": "shared.growth_audit", "value": {"score": 8.3, "band": "good"}}]
    (tmp_path / "audit.jsonl").write_text(json.dumps({"spec": "tissue/exp13_g0_s1", "score": 8.3, "growth": 14.2}) + "\n")
    (tmp_path / "measures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    exp13.write_results(str(tmp_path / "measures.jsonl"), str(tmp_path), audit=False)
    fb = [json.loads(l) for l in open(tmp_path / "feedback.jsonl")]
    au = [json.loads(l) for l in open(tmp_path / "audit.jsonl")]
    assert fb == [{"spec": "tissue/exp13_g0_s1", **{k: None for k in exp13.RESULT_KEYS},
                   "late_over_peak": 0.9, "size_cv": 0.25}]          # the last measurement wins
    assert au == [{"spec": "tissue/exp13_g0_s1", "score": 8.3, "growth": 14.2}]    # the auditor's record, kept

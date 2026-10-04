"""exp20's rulers on planted inputs: the arithmetic, and a fake training run whose trace test and cached trial scores
carry known numbers (the trial scorer is not re-run: its cache is newer than the fake checkpoint)."""
import json
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")

from exp_measures import run_measure  # noqa: E402
from exp_measures import exp20 as M  # noqa: E402


def test_window_skill_is_one_minus_the_summed_ratio():
    assert abs(M.window_skill([[1, 1], [2]], [[2, 2], [4]]) - 0.5) < 1e-12
    assert M.window_skill([[3.0]], [[3.0]]) == 0.0
    assert M.window_skill([[4.0]], [[2.0]]) == -1.0


def test_evoked_reads_a_planted_step():
    before = np.zeros((9, 5))
    win = np.zeros((50, 5))
    win[:10] = 0.3                                   # 0.3 over the first 10 of 20 evoked frames
    assert abs(M.evoked(win, before, 20) - 0.15) < 1e-12
    assert abs(M.evoked(win + 1.0, before + 1.0, 20) - 0.15) < 1e-12      # a level shift cancels


def test_ctrl_excess_and_gvgm():
    assert M.ctrl_excess(0.10, 0.133) == 0.0
    assert abs(M.ctrl_excess(1.0, 0.133) - 0.867) < 1e-12
    rec = {"mid": 1.18, "hind": 0.30}
    assert abs(M.gvgm_rel(rec, rec) - 1.0) < 1e-12
    assert abs(M.gvgm_rel({"mid": 0.5, "hind": 0.5}, rec)) < 1e-12


def fake_run(tmp, gut_law=(0.10, 0.12), gut_sta=(0.135, 0.135), ev_law=(0.02, 0.20, 0.22)):
    d = os.path.join(tmp, "fake_gb")
    os.makedirs(os.path.join(d, "results"))
    os.makedirs(os.path.join(d, "models"))
    yaml.safe_dump({"name": "fake_gb", "task": {"reference": {"trace_recording": "gutbrain_glucose_f1"}}},
                   open(os.path.join(d, "config.yaml"), "w"))
    yaml.safe_dump({"general": {}}, open(os.path.join(d, "model.yaml"), "w"))
    open(os.path.join(d, "models", "best.pt"), "w").close()
    time.sleep(0.01)
    res = {"name": "fake_gb", "mode": "trace", "skill": [0.01 * h for h in range(1, 33)], "skill_short": 0.05,
           "skill_long": 0.02, "identity_gap": 2e-9, "origins": 84,
           "free": {"r2_raw": -0.4, "r2_denoised": -0.2, "finite": 1.0}}
    json.dump(res, open(os.path.join(d, "results", "fake_gb_test.json"), "w"))
    H = 3
    tr = [{"onset": 913, "kind": "control", "site": 1, "mse_resp": [0.1] * H, "mse_all": [0.05] * H,
           "evoked_law": ev_law[0], "evoked_rec": 0.0286, "mse_sta_resp": [0.12] * H, "mse_sta_all": [0.06] * H}]
    for k, (law, sta, ev) in enumerate(zip(gut_law, gut_sta, ev_law[1:])):
        tr.append({"onset": 1563 + k, "kind": "gut", "site": 2 + k, "mse_resp": [law] * H, "mse_all": [0.04] * H,
                   "evoked_law": ev, "evoked_rec": 0.21, "mse_sta_resp": [sta] * H, "mse_sta_all": [0.05] * H})
    json.dump({"name": "fake_gb", "trials": tr}, open(os.path.join(d, "results", "fake_gb_trials.json"), "w"))
    return d


def test_forecast_and_free_read_the_planted_numbers(tmp_path):
    d = fake_run(str(tmp_path))
    v = run_measure("exp20.forecast", d)
    assert abs(v["skill_short"] - 0.05) < 1e-12 and abs(v["skill_h16"] - 0.16) < 1e-12 and v["identity_gap"] == 2e-9
    f = run_measure("exp20.free", d)
    assert f["finite"] == 1.0 and abs(f["r2_denoised"] + 0.2) < 1e-12


def test_trial_reads_the_planted_windows(tmp_path):
    v = run_measure("exp20.trial", fake_run(str(tmp_path)))
    assert abs(v["skill_gut"] - (1 - 0.22 / 0.27)) < 1e-12           # summed over both gut trials
    assert abs(v["skill_gut_all"] - (1 - 0.04 / 0.05)) < 1e-12
    assert abs(v["r_law"] - 0.02 / 0.21) < 1e-12 and abs(v["r_rec"] - 0.0286 / 0.21) < 1e-12
    assert v["ctrl_excess"] == 0.0 and v["n_gut_trials"] == 2


def test_trial_control_answered_like_gut_scores_excess(tmp_path):
    v = run_measure("exp20.trial", fake_run(str(tmp_path), ev_law=(0.21, 0.20, 0.22)))
    assert abs(v["ctrl_excess"] - (1.0 - 0.0286 / 0.21)) < 1e-12


def test_trial_silent_law_is_not_read_as_a_matching_ratio(tmp_path):
    # the smoke run's case: no gut response, small negative changes everywhere -- the old ratio of the law's own
    # responses read 0.13; over the recorded gut response it reads the law's tiny control change, no excess
    v = run_measure("exp20.trial", fake_run(str(tmp_path), ev_law=(-0.0034, -0.0048, -0.0463)))
    assert abs(v["r_law"] - (-0.0034 / 0.21)) < 1e-12 and v["ctrl_excess"] == 0.0

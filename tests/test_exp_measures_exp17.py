"""exp17's rulers on planted inputs: a fake training run whose trace test carries known numbers."""
import json
import os
import sys

import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
os.environ.setdefault("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData")

from exp_measures import run_measure  # noqa: E402

NAMES = ["gain", "dots", "flash", "taxis", "turning", "position", "open loop", "rotation", "dark"]


def fake_run(tmp, short=0.12, long=0.30, lookup=0.25, by_cond=(0.3, 0.2, -0.1, 0.4, 0.1, -0.2, 0.5, 0.3, 0.2),
             ami=0.17, finite=1.0):
    d = os.path.join(tmp, "fake_zap")
    os.makedirs(os.path.join(d, "results"))
    yaml.safe_dump({"name": "fake_zap"}, open(os.path.join(d, "config.yaml"), "w"))
    yaml.safe_dump({"general": {}}, open(os.path.join(d, "model.yaml"), "w"))
    res = {"name": "fake_zap", "mode": "trace", "names": NAMES, "skill": [0.01 * h for h in range(1, 33)],
           "skill_short": short, "skill_long": long, "lookup_skill_long": lookup,
           "skill_long_minus_lookup": long - lookup, "skill_long_by_condition": list(by_cond),
           "n_conditions_long_positive": sum(1 for v in by_cond if v > 0), "identity_gap": 3e-9,
           "free": {"r2_raw": -0.4, "r2_raw_sd": 0.2, "r2_denoised": -0.3, "r2_denoised_sd": 0.25, "finite": finite}}
    json.dump(res, open(os.path.join(d, "results", "fake_zap_test.json"), "w"))
    json.dump({"k": 8, "ami_tuning": ami}, open(os.path.join(d, "results", "fake_zap_clusters.json"), "w"))
    return d


def test_forecast_reads_the_planted_numbers(tmp_path):
    v = run_measure("exp17.forecast", fake_run(str(tmp_path)))
    assert abs(v["skill_short"] - 0.12) < 1e-12 and abs(v["skill_long"] - 0.30) < 1e-12
    assert abs(v["skill_long_minus_lookup"] - 0.05) < 1e-12
    assert v["n_conditions_long_positive"] == 7 and abs(v["skill_h16"] - 0.16) < 1e-12
    assert abs(v["skill_long_open_loop"] - 0.5) < 1e-12


def test_free_and_embedding(tmp_path):
    d = fake_run(str(tmp_path), ami=0.23, finite=0.0)
    f = run_measure("exp17.free", d)
    assert f["finite"] == 0.0 and abs(f["r2_denoised"] + 0.3) < 1e-12
    assert abs(run_measure("exp17.embedding", d)["ami_tuning"] - 0.23) < 1e-12


def test_a_field_test_is_refused(tmp_path):
    d = fake_run(str(tmp_path))
    p = os.path.join(d, "results", "fake_zap_test.json")
    r = json.load(open(p))
    r["mode"] = "full"
    json.dump(r, open(p, "w"))
    try:
        run_measure("exp17.forecast", d)
    except Exception:
        return
    raise AssertionError("a non-trace test file was read as a trace test")

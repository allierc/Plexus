"""exp16's ruler on planted inputs: a fake training run whose test file carries known numbers."""
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tools")]
import pytest  # noqa: E402

if not (os.environ.get("PLEXUS_OUTPUT_ROOT") or os.environ.get("GNN_OUTPUT_ROOT")):
    pytest.skip("the frozen redox recording lives under $GNN_OUTPUT_ROOT, which is not set", allow_module_level=True)

from exp_measures import exp16, run_measure  # noqa: E402


def fake_run(tmp, slope_model=-0.003, slope_rec=-0.0036, rmse=(0.10, 0.11, 0.12), pers=(0.14, 0.15, 0.16)):
    d = os.path.join(tmp, "fake_gc")
    os.makedirs(os.path.join(d, "results"))
    yaml.safe_dump({"name": "fake_gc", "task": {"reference": {"field_recording": "hlo_washout"}}},
                   open(os.path.join(d, "config.yaml"), "w"))
    yaml.safe_dump({"general": {}}, open(os.path.join(d, "model.yaml"), "w"))
    t = np.arange(1, 16) / 6.0
    res = {"name": "fake_gc", "test_volumes": [55, 69],
           "horizons": {str(h): {"rmse": r, "persistence_rmse": p, "skill": 1 - r * r / (p * p)}
                        for h, r, p in zip((1, 3, 6), rmse, pers)},
           "free": {"rmse": [0.1] * 14 + [0.2], "persistence_rmse": [0.15] * 15,
                    "organoid_mean_model": list(0.59 + slope_model * t),
                    "organoid_mean_recorded": list(0.59 + slope_rec * t)}}
    json.dump(res, open(os.path.join(d, "results", "fake_gc_test.json"), "w"))
    return d


def test_planted_numbers_read_back(tmp_path):
    v = run_measure("exp16.forecast", fake_run(str(tmp_path)))
    assert abs(v["rmse_h3"] - 0.11) < 1e-12 and abs(v["persistence_h6"] - 0.16) < 1e-12
    assert abs(v["skill_h1"] - (1 - 0.01 / 0.0196)) < 1e-12
    assert abs(v["identity_gap"] - 0.04) < 1e-12
    assert abs(v["slope_model_per_h"] + 0.003) < 1e-9 and abs(v["slope_recorded_per_h"] + 0.0036) < 1e-9
    assert abs(v["free_frac_beats_persistence"] - 14 / 15) < 1e-12
    assert abs(v["free_rmse_ratio_last"] - 0.2 / 0.15) < 1e-12


def test_xlsx_target_is_the_labs_trend():
    """The G20 target is read from the frozen recording: -0.0036 per h over volumes 55-69, SE 0.0007."""
    slope, se, jitter = exp16.xlsx_slope("hlo_washout", (55, 69))
    assert -0.0040 < slope < -0.0032 and 0.0005 < se < 0.0009 and 0.0015 < jitter < 0.0025


def test_slope_error_is_against_the_xlsx(tmp_path):
    v = run_measure("exp16.forecast", fake_run(str(tmp_path), slope_model=0.0))
    assert abs(v["slope_error_per_h"] - abs(v["xlsx_slope_per_h"])) < 1e-9


def test_embedding_correlation_length_scales_with_planted_domains(tmp_path):
    """Planted 'cells': square domains of side c voxels with random PC1 values. The 1/e length must grow
    with c (8 voxels -> longer than 4 voxels -> longer than 1 voxel)."""
    from plexus.tasks import field_recording as FR
    rec = FR.coarsen(FR.load("hlo_washout"), 4)
    Z, Y, X = rec["ratio"].shape[1:]
    out = {}
    for c in (1, 4, 8):
        d = fake_run(str(tmp_path / f"c{c}"))
        yaml.safe_dump({"name": "fake_gc", "task": {"reference": {"field_recording": "hlo_washout", "coarsen": 4,
                        "n_test": 15, "n_val": 6}}}, open(os.path.join(d, "config.yaml"), "w"))
        rng = np.random.default_rng(0)
        blocks = rng.normal(size=(Z, -(-Y // c), -(-X // c)))
        a = np.kron(blocks, np.ones((1, c, c)))[:, :Y, :X][None].astype(np.float32)
        np.savez_compressed(os.path.join(d, "results", "fake_gc_embedding.npz"), a=a)
        out[c] = run_measure("exp16.embedding", d)["corr_len_um"]
    assert out[1] < out[4] < out[8]
    assert out[1] < 1.5 * rec["dx_um"]

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


# ------------------------------------------------------------------ the network question (2026-10-10): planted runs
import numpy as np  # noqa: E402

import exp_measures.exp17 as E  # noqa: E402


def planted_movie(tmp, n_in=6, n_out=10, n_frames=240, rollout=""):
    """A run whose free rollout is PERFECT on the input neurons and FLAT on the others after block 1, and WRONG
    (sign-flipped) inside block 1 -- which the steady part must leave out. The recording is planted in E._REC."""
    rng = np.random.default_rng(0)
    n = n_in + n_out
    X = np.zeros((n_frames, n))
    X[:, :n_in] = rng.normal(size=(n_frames, n_in)) + rng.normal(size=(n_frames, 1))   # own signal + a shared one
    s = rng.normal(size=(n_frames, n_out // 2))
    X[:, n_in::2], X[:, n_in + 1::2] = s, -s           # the others in +- pairs: they add nothing to the brain mean,
    P = X.copy()                                       # so the learned and the recorded brain means coincide and a
    P[:, n_in:] = 0.3                                  # perfect input trace reads r = 1 exactly; flat on the others
    first = n_frames // E.N_BLOCKS                                             # block 1 of 24
    P[:first] = -X[:first]
    d = os.path.join(tmp, "planted_zap")
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    yaml.safe_dump({"name": "planted_zap", "task": {"reference": {"trace_recording": "planted"}}},
                   open(os.path.join(d, "config.yaml"), "w"))
    yaml.safe_dump({}, open(os.path.join(d, "model.yaml"), "w"))
    np.savez(os.path.join(d, "results", f"planted_zap{rollout}_movie.npz"), frames=np.arange(n_frames), pred=P,
             mean_obs_all=X.mean(1), mean_pred_all=P.mean(1))
    json.dump({"name": "planted_zap", "mode": "trace"}, open(os.path.join(d, "results", "planted_zap_test.json"), "w"))
    mask = np.zeros((n, 3), bool)
    mask[:n_in, 1] = True
    np.savez(os.path.join(tmp, "mask.npz"), mask=mask)
    E._REC["planted"] = X
    return d


def test_steady_reads_the_planted_split_and_leaves_block_one_out(tmp_path):
    d = planted_movie(str(tmp_path))
    v = run_measure("exp17.steady", d, input_mask=os.path.join(str(tmp_path), "mask.npz"), input_array="mask")
    assert v["n_input"] == 6 and v["n_other"] == 10 and v["n"] == 16
    assert abs(v["pn_r_input"] - 1.0) < 1e-9                     # perfect after block 1 (block 1 was sign-flipped)
    assert v["pn_r_other"] == 0.0 and v["n_flat_other"] == 10    # a flat learned trace scores 0, counted as flat
    assert abs(v["pn_r"] - 6 / 16) < 1e-9 and v["n_flat"] == 10
    assert abs(v["bm_r"] - 1.0) < 1e-9 and v["frames_steady"] == 240 - 240 // E.N_BLOCKS


def test_steady_reads_a_rollout_variant(tmp_path):
    d = planted_movie(str(tmp_path), rollout="_Wall0")
    v = run_measure("exp17.steady", d, rollout="_Wall0", input_mask=os.path.join(str(tmp_path), "mask.npz"))
    assert abs(v["pn_r_input"] - 1.0) < 1e-9


CLAMP = {"max_abs_w0_minus_clamp_in_w0": 0.0, "n_free_scored": 10,
         "laws": {k: {"local_r": {"estimate": v}} for k, v in
                  (("full", 0.57), ("W = 0", 0.47), ("clamp_in", 0.58), ("clamp_in, W = 0", 0.47))},
         "tests": [{"a": "clamp_in", "b": "clamp_in, W = 0", "local_r": {"difference": 0.11, "p": 1e-4}},
                   {"a": "full", "b": "W = 0", "local_r": {"difference": 0.10, "p": 2e-4}}]}


def test_clamp_reads_the_planted_file_and_is_pending_without_it(tmp_path, monkeypatch):
    d = planted_movie(str(tmp_path))
    monkeypatch.setattr(E, "DATA", str(tmp_path))
    v = run_measure("exp17.clamp", d)
    assert all(x is None for x in v.values())                    # no file: every key pending, none 0
    json.dump(CLAMP, open(os.path.join(str(tmp_path), "clamp_planted_zap.json"), "w"))
    v = run_measure("exp17.clamp", d)
    assert abs(v["relay_gap_pn"] - 0.11) < 1e-12 and abs(v["plain_gap_pn"] - 0.10) < 1e-12
    assert v["identity_max_abs"] == 0.0 and v["n_free"] == 10 and abs(v["clamp_in_pn"] - 0.58) < 1e-12
    assert v["beyond_relay_share"] is None                       # no clamp_in_cut rollout in the file
    c = json.loads(json.dumps(CLAMP))
    c["laws"]["clamp_in_cut"] = {"local_r": {"estimate": 0.525}}
    json.dump(c, open(os.path.join(str(tmp_path), "clamp_planted_zap.json"), "w"))
    assert abs(run_measure("exp17.clamp", d)["beyond_relay_share"] - 0.5) < 1e-9     # (0.525 - 0.47) / 0.11


def test_spectrum_counts_slow_brainwide_modes(tmp_path, monkeypatch):
    d = planted_movie(str(tmp_path))
    monkeypatch.setattr(E, "DATA", str(tmp_path))
    assert all(x is None for x in run_measure("exp17.spectrum", d).values())
    # slowest leak 0.01 /s (tau 100 s): eigenvalues -0.005 and +0.002 are slower, -0.02 and -0.5 are not;
    # of the two slow ones only the first spreads over >= half the uniform spread (200 um)
    J = {"abscissa": 0.002, "eig_re": [-0.005, 0.002, -0.02, -0.5], "mode_spread_um": [150.0, 20.0, 300.0, 10.0],
         "uniform_spread_um": 200.0, "max_abs_im": 0.3, "impulse": {"outside_share_all": 0.4}}
    json.dump({"leak": {"slowest_rate_per_s": 0.01, "abscissa": -0.01}, "J": J},
              open(os.path.join(str(tmp_path), "jacobian_planted_zap.json"), "w"))
    v = run_measure("exp17.spectrum", d)
    assert v["n_slow"] == 2 and v["n_unstable"] == 1 and v["n_slow_brainwide"] == 1
    assert abs(v["leak_abscissa"] + 0.01) < 1e-12 and abs(v["max_abs_im"] - 0.3) < 1e-12
    # a phase law: one J per block, the max over the blocks
    J2 = dict(J, eig_re=[-0.5, -0.6, -0.7, -0.8], abscissa=-0.5)
    json.dump({"leak": {"slowest_rate_per_s": 0.01, "abscissa": -0.01}, "per_block": [J2, J]},
              open(os.path.join(str(tmp_path), "jacobian_planted_zap.json"), "w"))
    v = run_measure("exp17.spectrum", d)
    assert v["n_slow_brainwide"] == 1 and abs(v["abscissa"] - 0.002) < 1e-12

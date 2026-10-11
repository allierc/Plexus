"""The graph phase's rollout kinds and spec keys (trainer, exp17 2026-10-10): `prune:`, `zero_input:`, `pulse:` and
`clock:` refused and accepted, their contexts applied and restored, `task.graph` read or refused."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plexus import trainer as T  # noqa: E402


def _spec(rollouts, drive=True, graph=None):
    t = {"reference": {"trace_recording": "x"}, "observe": {"set": "n", "block": "dff"}, "rollouts": rollouts}
    if drive:
        t["drive"] = {"set": "s", "block": "u", "window": [0, 0]}
    if graph is not None:
        t["graph"] = graph
    return {"task": t, "learnable": [{"param": "W_short", "op": "state_diffuse"}, {"param": "W_mid", "op": "state_diffuse"},
                                     {"block": "input", "of": "n"}, {"block": "tau", "of": "n"}]}


@pytest.mark.parametrize("ro", [
    [{"name": "a", "prune": {"thresholds": {"W_nope": 0.1}}}],            # not a learnable
    [{"name": "a", "prune": {"thresholds": {"W_short": -0.1}}}],          # a negative threshold
    [{"name": "a", "prune": {"W_short": 0.1}}],                           # not the {thresholds: {...}} form
    [{"name": "a", "prune": {"thresholds": {}}}],
    [{"name": "a", "zero_input": {"quantile": 1.0}}],                     # quantile in [0, 1)
    [{"name": "a", "zero_input": {"quantile": 0.5, "threshold": 1.0}}],   # one of them
    [{"name": "a", "zero_input": {"threshold": -1.0}}],
    [{"name": "a", "zero_input": {"mask": "zebrafish/m.npy"}}],           # an npz
    [{"name": "a", "zero_input": {"quantile": 0.5, "columns": []}}],      # columns: a non-empty list of indices
    [{"name": "a", "zero_input": {"quantile": 0.5, "columns": [0, -1]}}],
    [{"name": "a", "pulse": {"rois": [{"box": [[0, 1], [0, 1]]}], "level_z": 2.0, "start_s": 0.0}}],   # no duration
    [{"name": "a", "pulse": {"rois": [{"box": [[0, 1], [0, 1]]}], "mask": "zebrafish/m.npz", "level_z": 2.0,
                             "start_s": 0.0, "duration_s": 5.0}}],        # rois and mask
    [{"name": "a", "pulse": {"rois": [], "level_z": 2.0, "start_s": 0.0, "duration_s": 5.0}}],
    [{"name": "a", "pulse": {"rois": [{"box": [[0, 1], [0, 1]]}], "level_z": 2.0, "start_s": -1.0, "duration_s": 5.0}}],
    [{"name": "a", "clock": {"shift_frames": 2, "freeze_at": 3}}],        # one of them
    [{"name": "a", "clock": {"freeze_at": -1}}],
    [{"name": "a", "clock": {"shift_frames": 1.5}}],
])
def test_graph_kinds_refused(ro):
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec(ro), "trace_recording")


def test_graph_kinds_accepted_on_a_trace_law_only():
    ok = [{"name": "cut", "prune": {"thresholds": {"W_short": 0.027, "W_mid": 0.072}}},
          {"name": "b_half", "zero_input": {"quantile": 0.5}}, {"name": "b_t", "zero_input": {"threshold": 0.3}},
          {"name": "b_m", "zero_input": {"mask": "zebrafish/m.npz", "array": "mask"}},
          {"name": "b_c", "zero_input": {"quantile": 0.5, "columns": [1]}},
          {"name": "pulse_x", "drive": "off", "pulse": {"rois": [{"box": [[0, 0.2], [0, 1]], "units": "fraction"}],
                                                        "level_z": 2.0, "start_s": 120.0, "duration_s": 10.0}},
          {"name": "pulse_m", "pulse": {"mask": "zebrafish/m.npz", "level_z": 1.0, "start_s": 0.0, "duration_s": 1.0}},
          {"name": "shift", "clock": {"shift_frames": -600}}, {"name": "frozen", "clock": {"freeze_at": 4000}}]
    T._check_rollouts("spec", _spec(ok), "trace_recording")
    for ro in ok:
        with pytest.raises(ValueError):
            T._check_rollouts("spec", _spec([ro]), "field_recording")
    with pytest.raises(ValueError):                              # zero_input needs a drive
        T._check_rollouts("spec", _spec([ok[1]], drive=False), "trace_recording")


class _L:
    def __init__(self):
        f64 = dict(dtype=torch.float64)                                     # exact decimals in `tolist`
        self.p = {"state_diffuse.W_short": torch.tensor([0.1, 0.6, -0.2, 0.9], **f64), "state_diffuse.W_mid": torch.ones(3, **f64),
                  "n.input": torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [0.1, 0.1]], **f64), "n.tau": torch.zeros(4, **f64)}


def test_pruned_cuts_below_the_threshold_and_restores():
    L, spec = _L(), _spec([])
    with T._pruned(L, spec, {"W_short": 0.5}) as cut:
        assert cut == {"W_short": {"cut": 2, "of": 4, "threshold": 0.5}}
        assert L.p["state_diffuse.W_short"].tolist() == [0.0, 0.6, 0.0, 0.9]
        assert float(L.p["state_diffuse.W_mid"].sum()) == 3                  # the other set untouched
    assert L.p["state_diffuse.W_short"].tolist() == [0.1, 0.6, -0.2, 0.9]
    with T._pruned(L, spec, {"W_short": 0.0}) as cut:                        # the identity cut
        assert cut["W_short"]["cut"] == 0


def test_input_rows_quantile_threshold_mask(tmp_path):
    L, spec = _L(), _spec([])
    P = np.zeros((4, 3))
    box = {"rec": {"pos_um": P}, "X": torch.zeros(6, 4), "norm": (0.0, 1.0, 1.0)}
    # |B_i| = 0, 1, 5, 0.141: the input elements are 1, 2, 3; the median of their norms is 1.0
    sel, rec = T._input_rows(L, spec, box, {"quantile": 0.5})
    assert sel.tolist() == [False, False, False, True] and rec["n_input"] == 3 and rec["n_cut"] == 1
    sel, rec = T._input_rows(L, spec, box, {"threshold": 2.0})
    assert sel.tolist() == [False, True, False, True] and rec["by"] == "threshold"
    np.savez(tmp_path / "m.npz", mask=np.array([1, 1, 1, 0], np.float32))
    sel, rec = T._input_rows(L, spec, box, {"mask": str(tmp_path / "m.npz")})
    assert sel.tolist() == [False, True, True, False]                         # masked AND an input element
    with T._input_zeroed(L, spec, box, {"threshold": 2.0}) as rec:
        assert rec["n_cut"] == 2 and L.p["n.input"].tolist() == [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]
    assert L.p["n.input"].tolist() == [[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [0.1, 0.1]]
    # columns [1]: |B_i| over column 1 only = 0, 0, 4, 0.1 -> the input elements are 2 and 3, their median 2.05
    sel, rec = T._input_rows(L, spec, box, {"quantile": 0.5, "columns": [1]})
    assert sel.tolist() == [False, False, False, True] and rec["n_input"] == 2 and rec["columns"] == 1
    with T._input_zeroed(L, spec, box, {"quantile": 0.5, "columns": [1]}) as rec:
        assert L.p["n.input"].tolist() == [[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [0.1, 0.0]]     # column 0 kept
    assert L.p["n.input"].tolist() == [[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [0.1, 0.1]]
    with pytest.raises(ValueError):
        T._input_rows(L, spec, box, {"quantile": 0.5, "columns": [2]})                       # beyond the block's width


def test_pulse_holds_the_selected_elements_for_its_frames():
    g = np.random.default_rng(0)
    P = g.uniform(0, 100, (50, 3))
    box = {"rec": {"pos_um": P}, "X": torch.zeros(10, 50), "norm": (0.1, 0.05, 1.0)}
    p = {"rois": [{"box": [[0, 50], [0, 100]]}], "level_z": 2.0, "start_s": 2 * 0.914, "duration_s": 2 * 0.914}
    perturb, rec = T._pulse_fn(p, box, 0.914)
    sel = P[:, 0] <= 50
    assert rec["n_pulsed"] == int(sel.sum()) and rec["start_frame"] == 2 and rec["frames"] == 2
    assert abs(rec["level_dff"] - 0.2) < 1e-12
    blk = torch.zeros(50, 3)
    assert perturb(1, blk) is None and perturb(4, blk) is None
    out = perturb(2, blk)
    assert torch.allclose(out[torch.as_tensor(sel)], torch.full((int(sel.sum()), 3), 0.2))
    assert float(out[~torch.as_tensor(sel)].abs().sum()) == 0 and float(blk.abs().sum()) == 0   # the input untouched
    with pytest.raises(ValueError):                                           # every element pulsed: refused
        T._pulse_fn({"rois": [{"box": [[-1, 101], [-1, 101]]}], "level_z": 1.0, "start_s": 0.0, "duration_s": 1.0}, box, 0.914)


@pytest.mark.parametrize("g", [
    {"stride": 3},                                                            # an unread key
    {"controls": {"no_w": 3}},                                                # a name or null
    {"controls": {"twin": "x"}},
    {"regions": "atlas.npy"},
    {"steady_skip_min": -1},
    {"bootstrap": {"blocks": 24, "n": 10}},
    {"spectrum": {"state": "mean"}},
    {"movies": "yes"},
])
def test_task_graph_refusals(g):
    with pytest.raises(ValueError):
        T._check_graph("spec", _spec([], graph=g), "trace_recording")


def test_task_graph_accepted_on_a_trace_task_only():
    g = {"controls": {"no_w": "zap_now", "mean_field": None}, "inputs": {"mask": "zebrafish/m.npz", "array": "mask"},
         "regions": "zebrafish/atlas.npz", "steady_skip_min": 5, "bootstrap": {"blocks": 24, "resamples": 1000, "seed": 0},
         "prune": {"ladder": 7, "span": [0.111, 3.0]}, "prune_b": {"quantiles": [0.5, 0.9]},
         "spectrum": {"k": 20, "state": "rest"}, "pulse": {"level_z": 2.0, "duration_s": 10}, "movies": False,
         "every_frame": True}
    T._check_graph("spec", _spec([], graph=g), "trace_recording")
    T._check_graph("spec", _spec([]), "trace_recording")                      # absent: the defaults
    with pytest.raises(ValueError):
        T._check_graph("spec", _spec([], graph=g), "corpus")

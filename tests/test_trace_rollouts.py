"""`task.rollouts` (trainer): the free-rollout variants -- ROI masks, the spec's refusals, zeroed learnables restored."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plexus import trainer as T  # noqa: E402


def test_roi_mask_box_fraction_box_um_sphere():
    g = np.random.default_rng(0)
    P = g.uniform(0, 1, (5000, 3)) * [800.0, 400.0, 200.0] + [100.0, -50.0, 0.0]
    m = T._roi_mask([{"box": [[0.0, 0.25], [0.0, 1.0]], "units": "fraction"}], P)       # 2-D: every z
    assert np.array_equal(m, P[:, 0] <= 100.0 + 0.25 * np.ptp(P[:, 0]) + P[:, 0].min() - 100.0)
    m3 = T._roi_mask([{"box": [[100, 300], [-50, 350], [0, 100]]}], P)
    assert np.array_equal(m3, (P[:, 0] <= 300) & (P[:, 2] <= 100))
    ms = T._roi_mask([{"sphere": [500, 150, 100, 80]}], P)
    assert np.array_equal(ms, ((P - [500, 150, 100]) ** 2).sum(1) <= 80 ** 2)
    assert np.array_equal(T._roi_mask([{"box": [[100, 300], [-50, 350], [0, 100]]}, {"sphere": [500, 150, 100, 80]}], P),
                          m3 | ms)                                                    # a list of ROIs: their union


def _spec(rollouts, drive=True):
    t = {"reference": {"trace_recording": "x"}, "observe": {"set": "n", "block": "dff"}, "rollouts": rollouts}
    if drive:
        t["drive"] = {"set": "s", "block": "u", "window": [0, 0]}
    return {"task": t, "learnable": [{"param": "W_short", "op": "state_diffuse"}, {"block": "input", "of": "n"}]}


@pytest.mark.parametrize("ro", [
    [{"name": "a", "zero": ["W_nope"]}],                                    # not a learnable
    [{"name": "a"}, {"name": "a"}],                                         # names unique
    [{"name": "a", "drive": "maybe"}],
    [{"name": "a", "clamp": {"rois": []}}],
    [{"name": "a", "clamp": {"rois": [{"box": [[0, 1]]}]}}],                # one axis
    [{"name": "a", "clamp": {"rois": [{"box": [[0, 1], [0, 1]], "sphere": [0, 0, 0, 1]}]}}],
    [{"name": "a", "clamp": {"rois": [{"sphere": [0, 0, 1], "units": "um"}]}}],
    [{"name": "a", "clamp": {"rois": [{"box": [[0, 1], [0, 1]], "units": "mm"}]}}],
    [{"name": "a", "stride": 2}],                                           # an unread key
])
def test_refusals(ro):
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec(ro), "trace_recording")


def test_accepts_the_four_kinds_and_drive_off_needs_a_drive():
    ok = [{"name": "nominal"}, {"name": "W0", "zero": ["W_short"]}, {"name": "no_stimulus", "drive": "off"},
          {"name": "lead", "clamp": {"rois": [{"box": [[0, 0.25], [0, 1]], "units": "fraction"}]}}]
    T._check_rollouts("spec", _spec(ok), "trace_recording")
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec([{"name": "no_stimulus", "drive": "off"}], drive=False), "trace_recording")
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec(ok), "field_recording")


def test_zeroed_restores(monkeypatch):
    class L:
        p = {"op.W_short": torch.ones(4), "n.input": torch.full((3, 2), 2.0)}
    spec = {"learnable": [{"param": "W_short", "op": "state_diffuse"}, {"block": "input", "of": "n"}]}
    # monkeypatch restores the class's own staticmethod object (a bare `T.Learnables.key` read through the class is
    # the plain function, and writing it back made it an instance method for every later test -- exp19, 2026-10-03)
    monkeypatch.setattr(T.Learnables, "key", staticmethod(lambda e: "op.W_short" if e.get("param") else "n.input"))
    with T._zeroed(L, spec, ["W_short"]):
        assert float(L.p["op.W_short"].abs().sum()) == 0 and float(L.p["n.input"].sum()) == 12
    assert float(L.p["op.W_short"].sum()) == 4


def test_learnables_key_is_still_static_after_the_test_above():
    assert isinstance(T.Learnables.__dict__["key"], staticmethod)


def test_clamp_mask_from_a_file(tmp_path):
    """`clamp: {mask:, array:}` (exp17, 2026-10-10): the elements nonzero in the npz's array, one value per element."""
    g = np.random.default_rng(1)
    P = g.uniform(0, 100, (500, 3))
    m = (g.uniform(size=500) < 0.2).astype(np.float32)
    np.savez(tmp_path / "m.npz", mask=m, other=1.0 - m, by_input=np.ones((500, 3)))
    f = str(tmp_path / "m.npz")                         # absolute: graphs_data_path joins it as given
    assert np.array_equal(T._clamp_mask({"mask": f}, P), m != 0)
    assert np.array_equal(T._clamp_mask({"mask": f, "array": "other"}, P), m == 0)
    for bad in ({"mask": f, "array": "by_input"}, {"mask": f}):     # [N, F], and a length other than N
        with pytest.raises(ValueError):
            T._clamp_mask(bad, P if bad.get("array") else P[:400])
    box = [{"box": [[0, 50], [0, 100]]}]
    assert np.array_equal(T._clamp_mask({"rois": box}, P), T._roi_mask(box, P))   # the ROI path unchanged


@pytest.mark.parametrize("cl", [
    {"mask": "zebrafish/m.npz", "rois": [{"box": [[0, 1], [0, 1]]}]},   # both
    {"array": "mask"},                                                   # no file
    {"mask": "zebrafish/m.npy"},
    {"mask": "zebrafish/m.npz", "array": "a b"},
])
def test_clamp_mask_refusals(cl):
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec([{"name": "a", "clamp": cl}]), "trace_recording")


def test_clamp_mask_accepted_on_a_trace_law_only():
    ok = [{"name": "clamp_in", "clamp": {"mask": "zebrafish/input_mask_destripe_bal20.npz", "array": "mask"}},
          {"name": "clamp_in_W0", "zero": ["W_short"], "clamp": {"mask": "zebrafish/input_mask_destripe_bal20.npz"}}]
    T._check_rollouts("spec", _spec(ok), "trace_recording")
    with pytest.raises(ValueError):
        T._check_rollouts("spec", _spec(ok[:1], drive=False), "field_recording")

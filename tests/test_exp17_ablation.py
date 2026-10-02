"""tools/exp17_ablation.py: the left/right split and the edge-weight zeroing, on a toy graph (no data, CPU)."""
import os
import types

import numpy as np
import torch

import exp17_ablation as A


def test_left_right_splits_the_shorter_horizontal_axis_at_the_median(tmp_path):
    rng = np.random.default_rng(0)
    pos = np.stack([rng.uniform(0, 500, 101), rng.uniform(0, 800, 101), rng.uniform(0, 250, 101)], 1)
    f = os.path.join(tmp_path, "p.npz")
    np.savez(f, pos_um=pos)
    left, d = A.left_right(types.SimpleNamespace(_pos_file=(f, "pos_um")))
    assert d["left_right_axis"] == "x"                 # x spans ~500 um, y ~800 um: x is left-right
    assert d["n_left"] + d["n_right"] == 101 and abs(d["n_left"] - d["n_right"]) <= 1
    assert np.all(pos[left, 0] < d["midline_um"]) and np.all(pos[~left, 0] >= d["midline_um"])


def test_weights_zeroed_masks_then_restores():
    w = {"state_diffuse.W_short": torch.nn.Parameter(torch.arange(1.0, 7.0)),
         "state_diffuse.W_mid": torch.nn.Parameter(torch.ones(3))}
    learn = types.SimpleNamespace(p=w)
    keys = {"short": "state_diffuse.W_short", "mid": "state_diffuse.W_mid"}
    m = torch.tensor([True, False, True, False, False, True])
    with A.weights_zeroed(learn, keys, {"short": m, "mid": None}):
        assert w["state_diffuse.W_short"].tolist() == [0, 2, 0, 4, 5, 0]
        assert w["state_diffuse.W_mid"].tolist() == [0, 0, 0]
    assert w["state_diffuse.W_short"].tolist() == [1, 2, 3, 4, 5, 6]
    assert w["state_diffuse.W_mid"].tolist() == [1, 1, 1]

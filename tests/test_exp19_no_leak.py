"""exp19, Cedric's point 1 (2026-10-03): NO LEAK -- after its origin, a free rollout reads nothing recorded.

The field trainer's free rollout (`trainer._field_rollout`, used by `_test_field_full` and `_test_field`) is
seeded once from the recorded volumes origin, origin - 1, ... origin - (inputs - 1); after that only the law's
own field and the learned global forcing I(t) may drive it. Checked on a tiny synthetic recording with a law
given random (non-zero) weights and forcing:

    identity    every recorded volume AFTER the origin replaced by noise: the rollout is bit-identical
    planted     the origin volume itself changed: the rollout changes (the test can see a dependence)
    planted     the law is not persistence (the weights are live, so the identity case means something)
"""
import numpy as np
import pytest
import torch
import yaml

from plexus import trainer
from plexus.tasks import field_recording as FR

T, Z, Y, X, K = 12, 3, 6, 8, 2           # volumes, grid, input volumes


def _rec(seed, x=None):
    rng = np.random.default_rng(seed)
    if x is None:
        x = (1 + 0.3 * rng.standard_normal((T, Z, Y, X))).astype(np.float32)
    return {"ratio": x, "mask": np.ones((T, Z, Y, X), bool), "t_s": np.arange(T) * 3.8,
            "dx_um": 2.6, "dz_um": 9.0, "name": "synthetic"}


@pytest.fixture
def spec(tmp_path):
    fwd = {"general": {"name": "leak_fwd", "seed": 0, "n_frames": 6, "dt": 1.0, "boundary": "wall", "dim": 3,
                       "world": [Z / Y, 1.0, X / Y], "save_data": False},
           "sets": {},
           "fields": {"ratio": {"frame": "grid", "res": Y, "per_axis": True, "components": K}},
           "operators": [{"op": "diffuse", "model": "graphcast", "at": "ratio", "latent": 8, "layers": 2,
                          "inputs": K, "forcing": T, "spacing": [9.0, 2.6, 2.6], "seed": 0}],
           "schedule": ["diffuse"]}
    fp = tmp_path / "leak_fwd.yaml"
    fp.write_text(yaml.safe_dump(fwd))
    tr = {"name": "leak_tr", "model": str(fp),
          "learnable": [{"param": "theta", "op": "diffuse"}, {"param": "I", "op": "diffuse"}],
          "task": {"reference": {"field_recording": "synthetic", "coarsen": 1, "split": "all", "n_val": 2},
                   "observe": {"field": "ratio"}, "loss": "masked_mse"},
          "training": {"optimizer": "adam", "lr": 1e-3, "iters": 0, "horizon": [1], "batch": 1, "seed": 0}}
    tp = tmp_path / "leak_tr.yaml"
    tp.write_text(yaml.safe_dump(tr))
    return trainer.load(str(tp))


def _rollout(spec, rec, learn, monkeypatch, o, n):
    monkeypatch.setattr(FR, "load", lambda name, group=FR.GROUP: rec)
    box = trainer._field_setup(spec, "cpu")
    sims = trainer._field_sims(spec, [n], train=False)
    with torch.no_grad():
        return trainer._field_rollout(sims, learn, box, o, n, "cpu", False)[:, 0].numpy()


def test_free_rollout_reads_nothing_after_its_origin(spec, monkeypatch):
    o, n = K - 1 + 2, 6                                         # origin volume 3, 6 steps
    a = _rec(0)
    learn = trainer.Learnables(spec["learnable"], "cpu")
    _rollout(spec, a, learn, monkeypatch, o, n)                 # makes the parameters from the model
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                                       # live weights and a non-zero forcing
        for p in learn.p.values():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    base = _rollout(spec, a, learn, monkeypatch, o, n)

    x = a["ratio"].copy()
    x[o + 1:] = np.random.default_rng(7).standard_normal(x[o + 1:].shape).astype(np.float32) * 5
    after = _rollout(spec, _rec(0, x), learn, monkeypatch, o, n)
    assert np.array_equal(base, after), "a recorded volume after the origin changed the free rollout: a LEAK"

    y = a["ratio"].copy()
    y[o] += 1.0
    seeded = _rollout(spec, _rec(0, y), learn, monkeypatch, o, n)
    assert not np.allclose(base, seeded), "the origin volume does not reach the rollout: the test is blind"
    assert not np.allclose(base, np.broadcast_to(a["ratio"][o], base.shape)), "the law is persistence"


def test_segmented_rollout_equals_the_single_one(spec, monkeypatch):
    """tools/exp19_ablation.py rolls out in segments, each seeded from the MODEL's own last field: it must be the
    same rollout as one long `_field_rollout` (and so read nothing recorded after the origin either)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import exp19_ablation as A
    a = _rec(0)
    learn = trainer.Learnables(spec["learnable"], "cpu")
    o = K - 1
    n = T - 1 - o
    _rollout(spec, a, learn, monkeypatch, o, n)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for p in learn.p.values():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    whole = _rollout(spec, a, learn, monkeypatch, o, n)
    box = trainer._field_setup(spec, "cpu")
    seg, o2, n2 = A.rollout(spec, learn, box, "cpu", True, seg=3)
    assert (o2, n2) == (o, n)
    np.testing.assert_allclose(seg, whole, rtol=0, atol=1e-6)

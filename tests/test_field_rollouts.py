"""`task.rollouts` on a FIELD recording (exp19, 2026-10-03, ported from exp17's trace path, commit 03be5324).

Checked on a tiny synthetic recording and a law with live random weights and forcing:
    validation   a field spec's variants are read (messages, zero, clamp); `drive:` is refused (a field has none)
    identity     the segmented free rollout (`_field_free`) equals one `_field_rollout`
    identity     `messages: off` equals the rollout with the switch off; `zero: [I]` equals the forcing at 0, and
                 the learnables are restored after
    planted      `clamp:` gives the clamped voxels their recorded values at every frame; with messages off, the
                 voxels outside the clamp are untouched by it (nothing travels without messages)
    planted      the brain-mean metrics read R2 = 1 when the prediction is the recording
"""
import numpy as np
import pytest
import torch
import yaml

from plexus import trainer
from plexus.tasks import field_recording as FR

T, Z, Y, X, K = 12, 3, 6, 8, 2
ROLLOUTS = [{"name": "nominal"}, {"name": "W0", "messages": "off"}, {"name": "no_stimulus", "zero": ["I"]},
            {"name": "lead", "clamp": {"rois": [{"box": [[0.0, 0.3], [0.0, 1.0]], "units": "fraction"}]}}]


def _rec(x=None):
    rng = np.random.default_rng(0)
    if x is None:
        x = (1 + 0.3 * rng.standard_normal((T, Z, Y, X))).astype(np.float32)
    return {"ratio": x, "mask": np.ones((T, Z, Y, X), bool), "t_s": np.arange(T) * 3.8,
            "dx_um": 2.6, "dz_um": 9.0, "name": "synthetic"}


def _write(tmp_path, rollouts):
    fwd = {"general": {"name": "ro_fwd", "seed": 0, "n_frames": 6, "dt": 1.0, "boundary": "wall", "dim": 3,
                       "world": [Z / Y, 1.0, X / Y], "save_data": False},
           "sets": {}, "fields": {"ratio": {"frame": "grid", "res": Y, "per_axis": True, "components": K}},
           "operators": [{"op": "diffuse", "model": "graphcast", "at": "ratio", "latent": 8, "layers": 2,
                          "inputs": K, "forcing": T, "spacing": [9.0, 2.6, 2.6], "seed": 0}],
           "schedule": ["diffuse"]}
    fp = tmp_path / "ro_fwd.yaml"
    fp.write_text(yaml.safe_dump(fwd))
    tr = {"name": "ro_tr", "model": str(fp),
          "learnable": [{"param": "theta", "op": "diffuse"}, {"param": "I", "op": "diffuse"}],
          "task": {"reference": {"field_recording": "synthetic", "coarsen": 1, "split": "all", "n_val": 2},
                   "observe": {"field": "ratio"}, "loss": "masked_mse", "rollouts": rollouts},
          "training": {"optimizer": "adam", "lr": 1e-3, "iters": 0, "horizon": [1], "batch": 1, "seed": 0}}
    tp = tmp_path / "ro_tr.yaml"
    tp.write_text(yaml.safe_dump(tr))
    return str(tp)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rec = _rec()
    monkeypatch.setattr(FR, "load", lambda name, group=FR.GROUP: rec)
    spec = trainer.load(_write(tmp_path, ROLLOUTS))
    box = trainer._field_setup(spec, "cpu")
    learn = trainer.Learnables(spec["learnable"], "cpu")
    o = box["n_in"] - 1
    with torch.no_grad():
        trainer._field_rollout(trainer._field_sims(spec, [2], False), learn, box, o, 2, "cpu", False)
        g = torch.Generator().manual_seed(1)
        for p in learn.p.values():                      # live weights and a non-zero forcing
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    return spec, box, learn, rec, o, T - 1 - o


def _one(spec, box, learn, o, n, **kw):
    with torch.no_grad():
        return trainer._field_rollout(trainer._field_sims(spec, [n], False), learn, box, o, n, "cpu", False,
                                      **kw)[:, 0].numpy()


def test_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "load", lambda name, group=FR.GROUP: _rec())
    trainer.load(_write(tmp_path, ROLLOUTS))
    with pytest.raises(ValueError):
        trainer.load(_write(tmp_path, [{"name": "x", "drive": "off"}]))        # a field has no drive
    with pytest.raises(ValueError):
        trainer.load(_write(tmp_path, [{"name": "x", "messages": "maybe"}]))
    with pytest.raises(ValueError):
        trainer.load(_write(tmp_path, [{"name": "x", "zero": ["W_short"]}]))  # not a learnable of this run


def test_segmented_equals_single(setup):
    spec, box, learn, rec, o, n = setup
    free, o2, n2, clm = trainer._field_free(spec, learn, box, "cpu", seg=3)
    assert (o2, n2) == (o, n) and not clm.any()
    np.testing.assert_allclose(free, _one(spec, box, learn, o, n), rtol=0, atol=1e-6)


def test_messages_and_zero(setup):
    spec, box, learn, rec, o, n = setup
    v = {r["name"]: r for r in ROLLOUTS}
    w0, *_ = trainer._field_free(spec, learn, box, "cpu", variant=v["W0"], seg=4)
    np.testing.assert_allclose(w0, _one(spec, box, learn, o, n, messages=False), rtol=0, atol=1e-6)
    before = learn.p["diffuse.I"].clone()
    ns, *_ = trainer._field_free(spec, learn, box, "cpu", variant=v["no_stimulus"], seg=4)
    assert torch.equal(learn.p["diffuse.I"], before), "the forcing must be restored after the variant"
    with torch.no_grad():
        learn.p["diffuse.I"].zero_()
    ref = _one(spec, box, learn, o, n)
    with torch.no_grad():
        learn.p["diffuse.I"].copy_(before)
    np.testing.assert_allclose(ns, ref, rtol=0, atol=1e-6)
    assert not np.allclose(ns, _one(spec, box, learn, o, n)), "the forcing must matter for the test to mean anything"


def test_clamp(setup):
    spec, box, learn, rec, o, n = setup
    lead = dict(next(r for r in ROLLOUTS if r["name"] == "lead"), messages="off")
    free, _, _, clm = trainer._field_free(spec, learn, box, "cpu", variant=lead, seg=4)
    assert clm.any() and not clm.all()
    np.testing.assert_allclose(free[:, clm], rec["ratio"][o + 1:o + 1 + n][:, clm], rtol=0, atol=1e-6)
    w0 = _one(spec, box, learn, o, n, messages=False)
    np.testing.assert_allclose(free[:, ~clm], w0[:, ~clm], rtol=0, atol=1e-6)


def test_brain_mean_scores(setup):
    spec, box, learn, rec, o, n = setup
    obs = rec["ratio"][o + 1:o + 1 + n]
    sc, fm, om = trainer._field_scores(obs.copy(), rec, rec, o, n, np.zeros((Z, Y, X), bool))
    assert sc["brain_mean_r2"] == pytest.approx(1.0) and sc["brain_mean_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert sc["r2"] == pytest.approx(1.0)

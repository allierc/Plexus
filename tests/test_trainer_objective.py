"""The trainer's objective: evidence terms from `task.loss`, priors from `learnable:`, annealing.

What must hold, each asserted: a loss LIST loads and a term or reduction the reference kind cannot
score is refused; every prior kind computes the stated formula; `sign` groups an edge set's weights
by sender; annealing multiplies every prior by 1 - exp(-r t); and the objective records each term
under its own name.
"""
from __future__ import annotations

import copy
import math
import os
import tempfile

import pytest
import torch
import yaml

from plexus import trainer as T

BASE = "config/training/neural/t1_integrator_perfect_zf285.yaml"


def _load(mut):
    d = yaml.safe_load(open(BASE))
    mut(d)
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    try:
        return T.load(f.name)
    finally:
        os.unlink(f.name)


def test_a_loss_list_loads_and_bad_terms_are_refused():
    s = _load(lambda d: d["task"].update(loss=[{"term": "mse", "weight": 0.5, "reduction": "norm2"}]))
    assert T._loss_terms(s["task"], s["_kind"]) == [{"term": "mse", "weight": 0.5, "reduction": "norm2"}]
    with pytest.raises(ValueError, match="does not score"):
        _load(lambda d: d["task"].update(loss=[{"term": "log_mse"}]))
    with pytest.raises(ValueError, match="reduction"):
        _load(lambda d: d["task"].update(loss=[{"term": "mse", "reduction": "cubic"}]))
    with pytest.raises(ValueError, match="read by nothing"):
        _load(lambda d: d["task"].update(loss=[{"term": "mse", "gain": 2}]))


def test_every_prior_kind_computes_its_formula():
    x = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    kinds = {"shrink": x.pow(2).mean(), "shrink_to_mean": (x - x.mean()).pow(2).mean(),
             "smooth": (x[..., 1:] - x[..., :-1]).pow(2).sum(), "l1": x.abs().sum(),
             "l2": x.norm(2), "group_l1": x.norm(2, dim=0).sum()}
    for k, want in kinds.items():
        L = T.Learnables([{"block": "b", "of": "s", "prior": {k: 2.0}}])
        L.p["s.b"] = torch.nn.Parameter(x.clone())
        parts = {}
        got = L.prior(parts=parts)
        assert torch.allclose(got, 2.0 * want), k
        assert parts == {f"prior.s.b.{k}": pytest.approx(float(2.0 * want))}


def test_sign_prior_is_zero_when_every_sender_keeps_one_sign():
    L = T.Learnables([{"block": "w", "of": "syn", "prior": {"sign": 1.0}}])
    L._pre["syn.w"] = torch.tensor([0, 0, 1, 1])
    L.p["syn.w"] = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [-1.0], [-3.0]]))
    assert float(L.prior()) == pytest.approx(0.0, abs=1e-6)
    L.p["syn.w"] = torch.nn.Parameter(torch.tensor([[1.0], [-1.0], [-1.0], [-3.0]]))
    assert float(L.prior()) > 0.5


def test_annealing_multiplies_every_prior():
    s = _load(lambda d: d["training"].update(anneal={"rate": 0.5}))
    assert T._anneal(s, 0) == 0.0
    assert T._anneal(s, 2) == pytest.approx(1 - math.exp(-1.0))
    assert T._anneal(_load(lambda d: None), 7) == 1.0
    with pytest.raises(ValueError, match="anneal"):
        _load(lambda d: d["training"].update(anneal=0.5))


def test_the_objective_weighs_and_records_each_term():
    s = _load(lambda d: d["task"].update(loss=[{"term": "mse", "weight": 3.0}]))
    L = T.Learnables([{"block": "b", "of": "s", "prior": {"l2": 1.0}}])
    L.p["s.b"] = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    parts = {}
    r = torch.tensor([1.0, -1.0])
    tot = T._objective(s, L, {"mse": lambda red: T._reduce(r, red)}, 0, parts)
    assert float(tot) == pytest.approx(3.0 * 1.0 + 5.0)
    assert parts == {"loss.mse": pytest.approx(3.0), "prior.s.b.l2": pytest.approx(5.0)}


def test_point_mse_needs_its_targets_named():
    """A correspondence term without `points:` would have nothing to match each particle against."""
    d = yaml.safe_load(open("config/training/morph/cow.yaml"))
    d["task"]["loss"] = [{"term": "log_mse"}, {"term": "point_mse", "weight": 1.0}]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    with pytest.raises(ValueError, match="points"):
        T.load(f.name)
    os.unlink(f.name)


def test_smooth_on_a_lattice_differences_neighbouring_nodes_not_components():
    """A lattice is [K^3, width]; `smooth` must step between nodes along x, y, z of [K, K, K, width]."""
    K = 3
    L = T.Learnables([{"block": "rate", "of": "p", "with": "lattice", "over": "material", "K": K,
                       "extent": [0, 0, 0, 1], "prior": {"smooth": 1.0}}])
    g = torch.zeros(K, K, K, 6)
    g[..., 0] = torch.arange(K).float()[:, None, None]          # a ramp along x in component 0
    L.p["p.rate"] = torch.nn.Parameter(g.reshape(K ** 3, 6))
    assert float(L.prior()) == pytest.approx((K - 1) * K * K)   # (K-1)*K*K unit steps along x
    L.p["p.rate"] = torch.nn.Parameter(torch.ones(K ** 3, 6) * torch.arange(6.0))   # varies only across components
    assert float(L.prior()) == pytest.approx(0.0)

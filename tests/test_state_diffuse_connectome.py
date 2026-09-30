"""state_diffuse[model: connectome] (exp17): connectome-gnn's law (a message MLP, an update MLP, one weight per edge,
a per-node embedding) on GraphCast's multi-level mesh. Identity, planted, gradients, and its own priors."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("state_diffuse").implementations["connectome"]


def cloud(tmp, n=400, seed=0):
    g = np.random.default_rng(seed)
    P = (g.uniform(0, 100, (n, 3)) * [1.0, 1.6, 0.5]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    return torch.as_tensor(P), f


def op(f, **kw):
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": 16.0, "mesh_levels": 3, "inputs": 3,
         "hidden": 16, "substeps": 4, "positions_file": f, "seed": 0}
    p.update(kw)
    return CLS(p)


def test_untrained_is_persistence_exactly(tmp_path):
    P, f = cloud(str(tmp_path))
    x = torch.rand(len(P), 3)
    d = op(f).step(x, P)
    assert torch.equal(d[:, 0], torch.zeros(len(P))) and torch.equal(x[:, 1:] + d[:, 1:], x[:, :-1])


def test_W_is_one_weight_per_mesh_edge(tmp_path):
    P, f = cloud(str(tmp_path))
    o = op(f)
    assert o.W.numel() == o.mesh(P)["mm"][0].numel() and o.mesh_embedding.shape[0] == o.mesh(P)["n_mesh"]


def test_gradients_reach_W_the_embedding_and_the_three_mlps(tmp_path):
    P, f = cloud(str(tmp_path))
    o = op(f)
    g = torch.Generator().manual_seed(1)
    o.theta_n = o.theta_n + 0.05 * torch.randn(o.theta_n.shape, generator=g)     # a trained element update
    for name in ("theta_g", "theta_f", "theta_n", "W", "mesh_embedding"):
        setattr(o, name, getattr(o, name).clone().requires_grad_(True))
    d = o.step(torch.rand(len(P), 3), P)
    (d[:, 0] ** 2).mean().backward()
    for name in ("theta_g", "theta_f", "theta_n", "W", "mesh_embedding"):
        assert float(getattr(o, name).grad.norm()) > 0, name


def test_its_priors_evaluate(tmp_path):
    P, f = cloud(str(tmp_path))
    o = op(f, message="conductance")
    for kind in ("monotone", "pin", "input_group_l1"):
        v = o.prior_term("theta_g", kind)
        assert torch.isfinite(v) and float(v) >= 0
    assert o._shapes()["theta_g"] == 2 + 2 * o.mesh_embedding_dim


def test_other_positions_are_refused(tmp_path):
    P, f = cloud(str(tmp_path))
    Q = P + 50.0
    try:
        op(f).step(torch.rand(len(Q), 3), Q * 1.7)
    except ValueError:
        return
    raise AssertionError("positions other than the file's were accepted")

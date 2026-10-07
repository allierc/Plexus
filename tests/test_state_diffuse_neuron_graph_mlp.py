"""state_diffuse[model: neuron_graph_mlp] (exp17 batch 12): connectome-gnn's GNN on the neuron graph."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("state_diffuse").implementations["neuron_graph_mlp"]


def setup(tmp, n=1500, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 2,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22,
         "w_init": 0.05, "hidden": 16}
    p.update(kw)
    o = CLS(p)
    nb = {"input": 0.1 * torch.randn(n, 22), "embedding": torch.ones(n, 2)}
    return o, torch.as_tensor(P), nb


@pytest.mark.parametrize("message", ["sender", "pair"])
def test_untrained_is_persistence_and_learns(tmp_path, message):
    o, P, nb = setup(str(tmp_path), message=message)
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    assert float(o.step(x, P, None, u, nb=nb).abs().max()) == 0.0           # f_theta's last layer at 0
    o.theta_f = o.theta_f.clone()
    o.theta_f[-1] = 0.1                                                     # the output bias: now it moves
    for name in ("theta_f", "theta_g", "W_short", "W_long"):
        setattr(o, name, getattr(o, name).clone().requires_grad_(True))
    nb["embedding"] = nb["embedding"].clone().requires_grad_(True)
    o.theta_f.data[-1 - o.hidden: -1] = 0.1                                 # the last layer's weights: g matters
    d = o.step(x, P, None, u, nb=nb)
    d.pow(2).sum().backward()
    for name in ("theta_f", "theta_g", "W_short", "W_long"):
        assert float(getattr(o, name).grad.abs().sum()) > 0, (message, name)
    assert float(nb["embedding"].grad.abs().sum()) > 0


def test_priors_and_refusals(tmp_path):
    o, P, nb = setup(str(tmp_path))
    for k in ("monotone", "pin", "input_group_l1"):
        assert torch.isfinite(o.prior_term("theta_g", k))
    with pytest.raises(ValueError):
        setup(str(tmp_path), integrator="exponential")
    with pytest.raises(ValueError):
        setup(str(tmp_path), message="edge")


@pytest.mark.parametrize("model", ["neuron_graph_mlp", "neuron_graph_mlp_leak"])
def test_context_block_onehot(tmp_path, model):
    """context: block (exp17, 2026-10-07): a one-hot of the stimulus block enters g_phi (and f_theta): the input widths
    grow by the number of blocks, the law needs the frame, and at one state the update changes with the block alone."""
    n = 600
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(str(tmp_path), "rec.npz")
    np.savez(f, pos_um=P, offsets=np.array([0, 100, 250, 400]))
    cls = get_contract("state_diffuse").implementations[model]
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 1,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22,
         "w_init": 0.05, "hidden": 16}
    o0, o = cls(dict(p)), cls({**p, "context": "block"})
    for name, w in o0._shapes().items():
        assert o._shapes()[name] == w + 3
    assert o.FRAME_CLOCK
    nb = {"input": 0.1 * torch.randn(n, 22), "embedding": torch.ones(n, 2), "tau": torch.zeros(n, 1),
          "rest": torch.zeros(n, 1)}
    if "theta_f" in o._shapes():
        o.theta_f = o.theta_f.clone()
        o.theta_f[-1 - o.hidden:] = 0.1                                     # f_theta's last layer: the update moves
    x, u = torch.randn(n, 1), torch.randn(22, 1)
    o.frame = 50
    d0 = o.step(x, torch.as_tensor(P), None, u, nb=nb)
    o.frame = 300
    d1 = o.step(x, torch.as_tensor(P), None, u, nb=nb)
    assert not torch.allclose(d0, d1)                                       # the same state, another block
    with pytest.raises(ValueError):
        cls({**p, "context": "hour"})

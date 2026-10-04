"""state_diffuse[model: neuron_graph_mlp_leak] (exp17 batch 12, arms 9-12): the known ODE's leak with the GNN-MLP's
learned message."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

IMPL = get_contract("state_diffuse").implementations
CLS, KNOWN = IMPL["neuron_graph_mlp_leak"], IMPL["neuron_graph"]


def setup(tmp, n=1500, cls=CLS, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 2,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22}
    if cls is CLS:
        p["hidden"] = 16
    p.update(kw)
    o = cls(p)
    nb = {"input": 0.1 * torch.randn(n, 22), "embedding": torch.ones(n, 2), "tau": torch.randn(n, 1),
          "rest": 0.3 * torch.randn(n, 1)}
    return o, torch.as_tensor(P), nb


@pytest.mark.parametrize("message", ["sender", "pair"])
def test_w0_is_the_known_ode(tmp_path, message):
    """Every W = 0: the law is the known ODE with W = 0 (the same leak, rest, stimulus and integrator), to the bit."""
    o, P, nb = setup(str(tmp_path), message=message)
    k, _, _ = setup(str(tmp_path), cls=KNOWN, integrator="exponential")
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    assert o.integrator == "exponential"                                   # this law's default
    assert torch.equal(o.step(x, P, None, u, nb=nb), k.step(x, P, None, u, nb=nb))


@pytest.mark.parametrize("message", ["sender", "pair"])
def test_every_learnable_gets_a_gradient(tmp_path, message):
    o, P, nb = setup(str(tmp_path), message=message)
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    for name in ("theta_g", "W_short", "W_mid", "W_long"):
        setattr(o, name, (getattr(o, name).clone() + 0.05).requires_grad_(True))
    nb = {k: v.clone().requires_grad_(True) for k, v in nb.items()}
    o.step(x, P, None, u, nb=nb).pow(2).sum().backward()
    for name in ("theta_g", "W_short", "W_mid", "W_long"):
        assert float(getattr(o, name).grad.abs().sum()) > 0, (message, name)
    for k in ("embedding", "tau", "rest", "input"):
        assert float(nb[k].grad.abs().sum()) > 0, (message, k)
    assert not hasattr(o, "theta_f")                                       # no update MLP: the leak is the update


def test_priors_and_refusals(tmp_path):
    o, P, nb = setup(str(tmp_path))
    for k in ("monotone", "pin", "input_group_l1"):
        assert torch.isfinite(o.prior_term("theta_g", k))
    for bad in ({"synapse": "conductance"}, {"modulation": "siren"}, {"activation": "relu"}, {"message": "edge"}):
        with pytest.raises(ValueError):
            setup(str(tmp_path), **bad)

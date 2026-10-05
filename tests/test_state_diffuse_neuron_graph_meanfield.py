"""state_diffuse[model: neuron_graph_meanfield] (exp17 batch 15, the mean-field control): the known ODE with the graph's
message replaced by each element's gain on the mean over all elements of tanh(z)."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

IMPL = get_contract("state_diffuse").implementations
CLS, KNOWN = IMPL["neuron_graph_meanfield"], IMPL["neuron_graph"]


def setup(tmp, cls=CLS, n=800, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 2,
         "forcing": "stimulus.u", "forcing_dim": 22, "integrator": "exponential"}
    if cls is KNOWN:
        p.update({"short_k": 6, "mid_um": 32.0, "long_um": 128.0})
    p.update(kw)
    o = cls(p)
    nb = {"input": 0.1 * torch.randn(n, 22), "tau": torch.randn(n, 1), "rest": 0.3 * torch.randn(n, 1)}
    return o, torch.as_tensor(P), nb


def test_zero_gain_is_the_known_ode(tmp_path):
    """Every a_i = 0: the law is the known ODE with W = 0, to the bit."""
    o, P, nb = setup(str(tmp_path))
    k, _, _ = setup(str(tmp_path), cls=KNOWN)
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    assert torch.equal(o.step(x, P, None, u, nb=nb), k.step(x, P, None, u, nb=nb))


def test_the_message_is_the_gain_times_the_mean(tmp_path):
    """One substep, a hand-computed update: m_i = a_i * mean_j tanh(z_j)."""
    o, P, nb = setup(str(tmp_path), substeps=1)
    o.W_mean = torch.randn(len(P))
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    mu, sd, _ = o.norm
    z = (x - mu) / sd
    m = o.W_mean[:, None] * torch.tanh(z).mean()
    want = sd * o._frac(o._rate(nb["tau"])) * (-z + nb["rest"] + m + o._input_drive(nb, u))
    assert torch.allclose(o.step(x, P, None, u, nb=nb), want, atol=1e-5)


def test_gradient_and_refusals(tmp_path):
    o, P, nb = setup(str(tmp_path))
    o.W_mean = (o.W_mean + 0.1).requires_grad_(True)
    o.step(torch.randn(len(P), 1), P, None, torch.randn(22, 1), nb=nb).pow(2).sum().backward()
    assert float(o.W_mean.grad.abs().sum()) > 0
    assert o.EDGE_SETS == () and not hasattr(o, "W_short")
    for bad in ({"short_k": 6}, {"graph": "mesh"}, {"synapse": "conductance"}):
        with pytest.raises(ValueError):
            setup(str(tmp_path), **bad)

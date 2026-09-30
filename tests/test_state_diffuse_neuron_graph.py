"""state_diffuse[model: neuron_graph] (exp17): the known ODE on a multi-scale graph between the elements, no mesh."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("state_diffuse").implementations["neuron_graph"]


def setup(tmp, n=2000, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 4,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22}
    p.update(kw)
    o = CLS(p)
    nb = {"tau": torch.full((n, 1), -1.0), "rest": 0.1 * torch.randn(n, 1), "input": 0.1 * torch.randn(n, 22)}
    return o, torch.as_tensor(P), nb


def test_zero_weights_is_each_elements_own_leaky_filter(tmp_path):
    """W = 0: every element relaxes toward V_i + B_i.u at its own rate, nothing from any other."""
    o, P, nb = setup(str(tmp_path))
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    d = o.step(x, P, None, u, nb=nb)
    z, rate, tgt = x.clone(), torch.nn.functional.softplus(nb["tau"]), nb["rest"] + (nb["input"] * u.reshape(1, -1)).sum(1, keepdim=True)
    for _ in range(4):
        z = z + rate * (-z + tgt) / 4
    assert torch.allclose(d, z - x, atol=1e-6)


def test_a_planted_edge_moves_its_receiver_only(tmp_path):
    o, P, nb = setup(str(tmp_path), substeps=1)
    nb["input"] = torch.zeros_like(nb["input"])
    s, r = o._E["long"]
    o.W_long = torch.zeros_like(o.W_long)
    o.W_long[0] = 1.0
    x = torch.zeros(len(P), 1)
    x[s[0]] = 2.0
    base = o.step(x, P, None, torch.zeros(22, 1), nb=nb)
    o.W_long[0] = 0.0
    d = (base - o.step(x, P, None, torch.zeros(22, 1), nb=nb)).abs().reshape(-1)
    assert int((d > 1e-7).sum()) == 1 and int(d.argmax()) == int(r[0])


def test_edge_sets_have_their_reach(tmp_path):
    o, P, nb = setup(str(tmp_path))
    st = o.graph_stats
    assert st["short"]["per_element"] == 6
    assert 16 < st["mid"]["mean_um"] < 48 and 64 < st["long"]["mean_um"] < 192
    assert st["mid"]["edges"] > 0 and st["long"]["edges"] > 0
    for name in ("mid", "long"):
        s, r = o._E[name]
        assert bool((s != r).all())


def test_gradients_reach_every_weight_set_and_block(tmp_path):
    o, P, nb = setup(str(tmp_path), w_init=0.01)
    for k in ("W_short", "W_mid", "W_long"):
        setattr(o, k, getattr(o, k).clone().requires_grad_(True))
    for k in nb:
        nb[k].requires_grad_(True)
    o.step(torch.randn(len(P), 1), P, None, torch.randn(22, 1), nb=nb).pow(2).sum().backward()
    for k in ("W_short", "W_mid", "W_long"):
        assert float(getattr(o, k).grad.norm()) > 0, k
    for k in nb:
        assert float(nb[k].grad.norm()) > 0, k


def test_a_reach_of_zero_drops_that_set(tmp_path):
    o, P, nb = setup(str(tmp_path), long_um=0.0)
    assert o.W_long.numel() == 0 and o.graph_stats["long"]["edges"] == 0
    assert torch.isfinite(o.step(torch.randn(len(P), 1), P, None, torch.randn(22, 1), nb=nb)).all()

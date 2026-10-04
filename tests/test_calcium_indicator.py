"""calcium_indicator (exp17): the forward model of a calcium recording -- a latent activity v read out by a
first-order kernel -- and the linear inverse that starts the latent from the recorded frames."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("calcium_indicator").implementations[None] if None in get_contract(
    "calcium_indicator").implementations else next(iter(get_contract("calcium_indicator").implementations.values()))


class _Lvl:
    def __init__(self, st, schema):
        self.state, self.state_schema, self.name = st, schema, "neuron"


class _H:
    def __init__(self, lvl):
        self.lvl = lvl

    def level(self, _):
        return self.lvl


def make(n=50, width=6, **kw):
    p = {"_at": "neuron", "block": "dff", "latent": "v", "width": width, "tau_s": 2.0, "frame_s": 0.914}
    p.update(kw)
    o = CLS(p)
    st = torch.zeros(n, width + 1)
    return o, _H(_Lvl(st, {"dff": (0, width), "v": (width, width + 1)}))


def test_the_tick_is_the_first_order_kernel():
    o, H = make()
    g = torch.Generator().manual_seed(0)
    H.lvl.state = torch.randn(50, 7, generator=g)
    d = o.forward(H)[("neuron", "dff")]
    k = 1 - math.exp(-0.914 / 2.0)
    c, v = H.lvl.state[:, :6], H.lvl.state[:, 6:]
    assert torch.allclose(d[:, :1], k * (v - c[:, :1]), atol=1e-6)
    assert torch.allclose(c[:, 1:] + d[:, 1:], c[:, :-1])          # the history shifts by one


def test_the_inverse_start_recovers_the_latent_one_frame_back():
    """A trace made by the kernel from a known v: the inverse taps return v(t0 - 1) exactly (held to t0)."""
    o, H = make()
    k = float(o.k())
    g = torch.Generator().manual_seed(1)
    v = torch.randn(50, 20, generator=g)
    c = torch.zeros(50, 21)
    for t in range(20):
        c[:, t + 1] = c[:, t] + k * (v[:, t] - c[:, t])
    t0 = 20
    H.lvl.state[:, :6] = torch.stack([c[:, t0 - j] for j in range(6)], 1)   # newest first
    o.init_latent(H)
    assert torch.allclose(H.lvl.state[:, 6], v[:, t0 - 1], atol=1e-5)


def test_identity_start_is_the_recorded_value():
    o, H = make(encoder_init="identity")
    H.lvl.state[:, :6] = torch.randn(50, 6)
    o.init_latent(H)
    assert torch.equal(H.lvl.state[:, 6], H.lvl.state[:, 0])


def test_gradients_reach_the_rate_and_the_taps():
    o, H = make()
    o.rate = o.rate.clone().requires_grad_(True)
    o.taps = o.taps.clone().requires_grad_(True)
    H.lvl.state = torch.randn(50, 7)
    o.init_latent(H)
    o.forward(H)[("neuron", "dff")][:, 0].pow(2).sum().backward()
    assert float(o.rate.grad.abs().sum()) > 0 and float(o.taps.grad.abs().sum()) > 0

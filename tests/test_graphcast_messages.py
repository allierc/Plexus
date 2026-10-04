"""diffuse[graphcast] `messages: false` (exp19, Cedric 2026-10-03): the network controls of exp17's slides 26-27.

Identity: absent and `messages: true` are the same law (bit-identical step).
Planted: with messages on, a change of one voxel changes its neighbour's update; with them off it does not, while
it still changes its OWN update -- each voxel then evolves from its own inputs alone. Flipping the switch on a
trained instance (`op.messages = False`) is the "network weights zeroed, forcing kept" evaluation.
"""
import torch

from plexus.operators.field_ops import DiffuseGraphCast

Z, Y, X = 3, 5, 7


def _op(**kw):
    op = DiffuseGraphCast({"_at": "f", "latent": 8, "layers": 2, "inputs": 1, "spacing": [9.0, 2.6, 2.6],
                           "seed": 0, **kw})
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():                                   # live weights: the decoder starts at 0 otherwise
        op.theta.copy_(0.3 * torch.randn(op.theta.shape, generator=g))
    return op


def _pair():
    s = 1 + 0.2 * torch.randn(1, Z, Y, X, generator=torch.Generator().manual_seed(5))
    s2 = s.clone()
    s2[0, 1, 2, 3] += 1.0                                   # one voxel changed
    return s, s2


def test_default_is_messages_on():
    s, _ = _pair()
    assert torch.equal(_op().step(s), _op(messages=True).step(s))


def test_no_messages_isolates_each_voxel():
    s, s2 = _pair()
    on, off = _op(), _op(messages=False)
    d_on = on.step(s2) - on.step(s)
    d_off = off.step(s2) - off.step(s)
    nb = (0, 1, 2, 4)                                       # the changed voxel's x neighbour
    assert d_on[nb].abs() > 1e-6, "with messages a neighbour must feel the change"
    assert d_off[nb].abs() == 0, "without messages a neighbour must not feel it"
    assert d_off[0, 1, 2, 3].abs() > 1e-6, "the voxel itself still reads its own input"
    mask = torch.ones_like(d_off, dtype=torch.bool)
    mask[0, 1, 2, 3] = False
    assert torch.all(d_off[mask] == 0)


def test_switch_on_a_trained_instance():
    s, _ = _pair()
    a, b = _op(), _op(messages=False)
    a.messages = False                                      # the trained law, its network zeroed at inference
    assert torch.equal(a.step(s), b.step(s))

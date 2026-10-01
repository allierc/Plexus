"""diffuse[model: graphcast] (exp16): the learned field law. Identity where it must agree with
persistence, planted where it must differ, and the structural claims of its docstring."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("diffuse").implementations["graphcast"]
SP = [5.0, 3.3, 3.3]


def op(**kw):
    p = {"_at": "ratio", "latent": 8, "layers": 2, "spacing": SP, "seed": 0}
    p.update(kw)
    return CLS(p)


def planted(o, scale=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    o.theta = o.theta + scale * torch.randn(o.theta.shape, generator=g)
    return o


def test_untrained_is_persistence_exactly():
    s = torch.rand(1, 5, 12, 10)
    assert torch.equal(op().step(s), s)


def test_planted_differs_and_carries_a_gradient():
    o = planted(op())
    o.theta.requires_grad_(True)
    s = torch.rand(1, 5, 12, 10)
    y = o.step(s)
    assert float((y - s).abs().max()) > 1e-4
    ((y - s) ** 2).mean().backward()
    assert float(o.theta.grad.norm()) > 0


def test_no_wrap_across_the_box():
    """A change in the first z-plane reaches at most `layers` planes: it never wraps to the last one."""
    o = planted(op(layers=2))
    s = torch.rand(1, 8, 6, 6)
    s2 = s.clone()
    s2[0, 0] += 1.0
    d = (o.step(s2) - o.step(s)).abs().amax((0, 2, 3))
    assert float(d[0]) > 0 and float(d[-1]) == 0.0 and float(d[3:].max()) == 0.0


def test_layers_are_unshared():
    n1 = op(layers=1).theta.numel()
    n4 = op(layers=4).theta.numel()
    n7 = op(layers=7).theta.numel()
    assert n7 - n4 == n4 - n1 > 0                 # linear in depth: each layer its own weights


def test_two_inputs_shift_and_persist():
    s = torch.rand(2, 4, 6, 6)
    y = op(inputs=2).step(s)
    assert torch.equal(y[0], s[0]) and torch.equal(y[1], s[0])   # untrained: newest kept, and shifted down
    o = planted(op(inputs=2))
    s_b = s.clone()
    s_b[1] += 0.5                                 # a different PREVIOUS state
    assert float((o.step(s_b)[0] - o.step(s)[0]).abs().max()) > 1e-5   # the law reads it


def test_wrong_field_is_refused():
    try:
        op(inputs=2).step(torch.rand(1, 4, 6, 6))
    except ValueError as e:
        assert "input state" in str(e)
    else:
        raise AssertionError("a one-state field under inputs: 2 was accepted")


def mop(**kw):
    p = {"_at": "ratio", "latent": 8, "layers": 1, "spacing": SP, "seed": 0,
         "mesh_levels": 3, "mesh_stride": [1, 2, 2]}
    p.update(kw)
    return CLS(p)


def test_mesh_untrained_is_persistence_exactly():
    s = torch.rand(1, 6, 20, 18)
    assert torch.equal(mop().step(s), s)


def test_mesh_planted_differs_and_carries_a_gradient():
    o = planted(mop())
    o.theta.requires_grad_(True)
    s = torch.rand(1, 6, 20, 18)
    y = o.step(s)
    assert float((y - s).abs().max()) > 1e-4
    ((y - s) ** 2).mean().backward()
    assert float(o.theta.grad.norm()) > 0


def test_multimesh_long_edges_reach_in_one_layer():
    """One processor layer: a change at voxel (0, 0, 0) reaches voxel (0, 9, 0) through a level-2 edge
    (4 mesh steps = 8 voxels) when there are 3 levels, and cannot with 1 level (1 mesh step = 2 voxels)."""
    s = torch.rand(1, 4, 24, 8)
    s2 = s.clone()
    s2[0, 0, 0, 0] += 1.0
    far = {}
    for L in (1, 3):
        o = planted(mop(mesh_levels=L, layers=1))
        far[L] = float((o.step(s2) - o.step(s))[0, 0, 9, 0].abs())
    assert far[3] > 0 and far[1] == 0.0


def test_mesh_graph_sizes():
    G = mop().mesh((14, 128, 128), "cpu")
    assert G["n_mesh"] == 14 * 64 * 64
    assert G["g2m"][0].numel() == 14 * 128 * 128                   # every voxel sends once
    # own block + the nearer block along y and along x and both: none past the rim (126 of 128 rows have one)
    assert G["m2g"][0].numel() == 14 * (128 + 126) ** 2


def test_embedding_is_read_and_its_zero_is_neutral_at_init():
    for kw in ({}, {"mesh_levels": 2, "mesh_stride": [1, 2, 2]}):
        s = torch.rand(1, 4, 12, 10)
        a = torch.zeros(2, 4, 12, 10)
        assert torch.equal(op(embedding="embed", embedding_dim=2, **kw).step(s, a), s)   # untrained: persistence
        o = planted(op(embedding="embed", embedding_dim=2, **kw))
        a2 = a.clone()
        a2[:, :, :6] = 1.0                        # two groups of voxels, told apart only by the embedding
        d = (o.step(s, a2) - o.step(s, a)).abs()
        assert float(d[:, :, :6].max()) > 1e-5


def test_embedding_shape_is_checked():
    try:
        op(embedding="embed", embedding_dim=2).step(torch.rand(1, 4, 6, 6), torch.zeros(3, 4, 6, 6))
    except ValueError as e:
        assert "embedding" in str(e)
    else:
        raise AssertionError("a 3-wide embedding was accepted by a 2-wide law")


def test_forcing_is_global_and_read_at_the_right_volume():
    """I(t) enters every voxel alike: with a uniform input field, a planted law's output stays uniform in
    space whatever I is; and the volume index selects the entry of I."""
    o = planted(op(forcing=5))
    o.I = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0])
    s = torch.full((1, 4, 6, 6), 0.5)
    y2, y1 = o.step(s, t=2), o.step(s, t=1)
    assert float((y2 - y1).abs().max()) > 1e-5                       # t = 2 reads I = 3, t = 1 reads 0
    assert torch.equal(op(forcing=5).step(s, t=2), s)                # untrained: persistence


def test_checkpoint_gives_the_same_gradient():
    class Hh:                                                          # a minimal hierarchy: one field
        pass
    g = {}
    for ck in (False, True):
        o = planted(op(checkpoint=ck))
        o.theta.requires_grad_(True)
        H = Hh()
        H.fields = {"ratio": type("F", (), {})()}
        H.fields["ratio"].grid = torch.rand(1, 4, 6, 6, generator=torch.Generator().manual_seed(3))
        for k in range(3):
            H.frame = k
            o.forward(H)
        (H.fields["ratio"].grid[0] ** 2).mean().backward()
        g[ck] = o.theta.grad.clone()
    assert torch.allclose(g[False], g[True], atol=1e-7)


def test_transport_untrained_is_persistence_exactly():
    for kw in ({}, {"mesh_levels": 2, "mesh_stride": [1, 2, 2]}):
        s = torch.rand(1, 4, 12, 10)
        assert torch.equal(op(transport=True, **kw).step(s), s)


def test_transport_learns_a_translation():
    """A velocity head saturated at +1 voxel per tick along x and no local change: the field moves by one
    voxel along x, exactly, and the voxel entering from outside the box is 0."""
    o = op(transport=True)
    W = o._views(1, 3)
    W["dec_v.b2"].data[:] = torch.tensor([0.0, 0.0, 30.0])       # tanh(30) = 1 in float32 -> v = (0, 0, 1)
    s = torch.rand(1, 4, 6, 7)
    y = o.step(s)
    assert torch.allclose(y[..., 1:], s[..., :-1], atol=1e-6) and float(y[..., 0].abs().max()) < 1e-6


def test_transport_moves_the_embedding_with_the_field():
    class Hh:
        pass
    o = op(transport=True, transport_embedding=True, embedding="emb", embedding_dim=2)
    W = o._views(1, 3)
    W["dec_v.b2"].data[:] = torch.tensor([0.0, 30.0, 0.0])       # +1 voxel along y
    H = Hh()
    H.fields = {"ratio": type("F", (), {})(), "emb": type("F", (), {})()}
    H.fields["ratio"].grid = torch.rand(1, 3, 6, 5)
    e0 = torch.rand(2, 3, 6, 5)
    H.fields["emb"].grid = e0.clone()
    H.frame = 0
    o.forward(H)
    assert torch.allclose(H.fields["emb"].grid[:, :, 1:], e0[:, :, :-1], atol=1e-6)


def test_normalisation_keeps_persistence_and_scales_the_increment():
    s = torch.rand(1, 4, 8, 6)
    for kw in ({}, {"mesh_levels": 2, "mesh_stride": [1, 2, 2]}):
        o = op(**kw)
        o.norm = (0.55, 0.19, 0.15)
        assert torch.equal(o.step(s), s)                               # untrained: still persistence
        o = planted(op(**kw))
        d1 = o.step(s) - s
        o.norm = (0.0, 1.0, 2.0)                                      # the same law, its increment x2
        assert torch.allclose(o.step(s) - s, 2 * d1, atol=1e-6)

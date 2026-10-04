"""state_diffuse[model: graphcast] (exp17): GraphCast on a point cloud. Identity where it must agree with
persistence, planted where it must differ, and the structural claims of its docstring."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

CLS = get_contract("state_diffuse").implementations["graphcast"]


def cloud(n=400, seed=0):
    g = np.random.default_rng(seed)
    return torch.as_tensor(g.uniform(0, 100, (n, 3)) * [1.0, 1.6, 0.5], dtype=torch.float32)


def op(**kw):
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "mesh_spacing": 16.0, "mesh_levels": 3,
         "latent": 8, "layers": 2, "inputs": 3, "seed": 0}
    p.update(kw)
    return CLS(p)


def planted(o, scale=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    o.theta = o.theta + scale * torch.randn(o.theta.shape, generator=g)
    return o


def test_untrained_is_persistence_exactly():
    """Zero decoder: the increment is exactly 0 and every older value moves to its newer neighbour's."""
    P = cloud()
    x = torch.rand(len(P), 3)
    d = op().step(x, P)
    assert torch.equal(d[:, 0], torch.zeros(len(P)))
    assert torch.equal(x[:, 1:] + d[:, 1:], x[:, :-1])


def test_planted_differs_and_carries_a_gradient():
    P = cloud()
    o = planted(op())
    o.theta.requires_grad_(True)
    d = o.step(torch.rand(len(P), 3), P)
    assert float(d[:, 0].abs().max()) > 1e-4
    (d[:, 0] ** 2).mean().backward()
    assert float(o.theta.grad.norm()) > 0


def test_every_element_reaches_the_mesh_and_back():
    P = cloud(800)
    G = op(mesh_levels=4).mesh(P)
    s, r, _ = G["g2m"]
    assert set(s.tolist()) == set(range(len(P)))
    s, r, _ = G["m2g"]
    assert set(r.tolist()) == set(range(len(P)))


def test_coarse_levels_add_long_edges():
    P = cloud(800)
    n1 = op(mesh_levels=1).mesh(P)["stats"]["mm_edges"]
    n3 = op(mesh_levels=3).mesh(P)["stats"]["mm_edges"]
    f = op(mesh_levels=3).mesh(P)["mm"][2]
    assert n3 > n1 and float(f[:, 3].max()) >= 3.5          # a level-2 edge spans 4 fine spacings


def test_forcing_reaches_every_element():
    P = cloud()
    o = planted(op(forcing="stimulus.u", forcing_dim=6))
    x = torch.rand(len(P), 3)
    u0, u1 = torch.zeros(2, 3), torch.ones(2, 3)
    d = (o.step(x, P, None, u1)[:, 0] - o.step(x, P, None, u0)[:, 0]).abs()
    assert float(d.min()) > 0                               # a global input, read at every element


def test_embedding_is_read():
    P = cloud()
    o = planted(op(embedding="embedding", embedding_dim=2))
    x = torch.rand(len(P), 3)
    a0, a1 = torch.zeros(len(P), 2), torch.zeros(len(P), 2)
    a1[0] = 1.0
    d = (o.step(x, P, a1)[:, 0] - o.step(x, P, a0)[:, 0]).abs()
    assert float(d[0]) > 0


def test_checkpoint_same_gradient():
    P = cloud()
    x = torch.rand(len(P), 3)
    grads = []
    for ck in (False, True):
        o = planted(op(checkpoint=ck))
        o.theta.requires_grad_(True)
        from torch.utils.checkpoint import checkpoint
        d = checkpoint(o.step, x, P, None, None, use_reentrant=False) if ck else o.step(x, P)
        (d[:, 0] ** 2).sum().backward()
        grads.append(o.theta.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_width_must_match_inputs():
    P = cloud()
    try:
        op(inputs=3).step(torch.rand(len(P), 2), P)
    except ValueError:
        return
    raise AssertionError("a block narrower than `inputs:` was accepted")


def test_levels_are_nested_as_graphcast():
    """Every node of a coarse level is a node of every finer level (M0 c M1 c ... as GraphCast's refinements)."""
    P = cloud(3000)
    L = op(mesh_levels=4, mesh_spacing=8.0).mesh(P)["level_nodes"]
    for k in range(1, 4):
        assert set(L[k].tolist()) <= set(L[k - 1].tolist()), f"level {k} has nodes that are not level-{k - 1} nodes"
        assert len(L[k]) < len(L[k - 1])


def test_lattice_is_centred_on_the_cloud():
    """The cloud's centre is a coarsest-level vertex, so a mirror-symmetric cloud gets a mirror-symmetric mesh."""
    g = np.random.default_rng(3)
    half = g.uniform(0, 1, (1500, 3)) * [60.0, 150.0, 40.0]
    P = torch.as_tensor(np.concatenate([half, half * [-1, 1, 1]]), dtype=torch.float32)   # mirror in x
    G = op(mesh_levels=3, mesh_spacing=10.0).mesh(P)
    lat = G["lattice"]
    k = (lat["mid_um"] - lat["origin_um"]) / lat["coarsest_um"]
    assert np.allclose(k, np.round(k), atol=1e-6)                   # the centre is a coarsest-level vertex
    c = G["centre_um"]
    mirrored = np.round(c * [-1, 1, 1] + [2 * lat["mid_um"][0], 0, 0], 4)
    assert set(map(tuple, np.round(c, 4))) == set(map(tuple, mirrored))   # every mesh node has its mirror


def test_grid2mesh_is_a_radius_query_and_mesh2grid_the_8_corners():
    P = cloud(500)
    G = op(mesh_levels=3).mesh(P)
    s, r, f = G["g2m"]
    n = torch.bincount(s, minlength=len(P))
    assert int(n.min()) >= 1 and int(n.max()) <= 8                # every element reaches the mesh
    assert float(f[:, 3].max()) <= 3 ** 0.5 / 2 + 1e-4            # within sqrt(3)/2 L0 (edge length in L0 units)
    s2, r2, _ = G["m2g"]
    assert torch.equal(torch.bincount(r2, minlength=len(P)), torch.full((len(P),), 8))


def test_mirror_axis_makes_an_asymmetric_cloud_symmetric():
    g = np.random.default_rng(4)
    P = torch.as_tensor(g.uniform(0, 1, (2000, 3)) * [80.0, 150.0, 40.0] + [-30.0, 0, 0], dtype=torch.float32)
    G = op(mesh_levels=3, mesh_spacing=10.0, mesh_mirror_axis=0).mesh(P)
    c, x0 = G["centre_um"], G["lattice"]["mid_um"][0]
    mirrored = np.round(c * [-1, 1, 1] + [2 * x0, 0, 0], 4)
    assert set(map(tuple, np.round(c, 4))) == set(map(tuple, mirrored))

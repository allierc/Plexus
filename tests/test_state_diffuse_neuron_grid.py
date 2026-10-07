"""state_diffuse[neuron_grid]: the known ODE whose coupling runs through GraphCast's lattice grid (exp17 batch 21)."""
import os

import numpy as np
import pytest
import torch

from plexus.operators.cell_ops import StateDiffuseNeuronGraph, StateDiffuseNeuronGrid


def setup(tmp, n=1500, cls=StateDiffuseNeuronGrid, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [120.0, 160.0, 64.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P)
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 1,
         "forcing": "stimulus.u", "forcing_dim": 4, "integrator": "exponential"}
    p.update(kw)
    o = cls(p)
    nb = {"tau": torch.full((n, 1), -1.0), "rest": 0.1 * torch.randn(n, 1, generator=torch.Generator().manual_seed(1)),
          "input": torch.zeros(n, 4)}
    x = torch.randn(n, 1, generator=torch.Generator().manual_seed(2))
    return o, x, torch.as_tensor(P), nb


def test_grid_builds(tmp_path):
    """The slides' lattice: corners of 16-um cubes at 3 levels, every neuron sent to >= 1 corner and read from 8."""
    o = setup(str(tmp_path))[0]
    st = o.grid_stats
    assert st["mesh_nodes"] > 0 and len(st["nodes_per_level"]) == 3
    assert o.W_grid.numel() == st["mm_edges_with_self"] == st["mm_edges"] + st["mesh_nodes"]
    gs, gr = o._grid["g2m"]
    assert set(gs.tolist()) == set(range(o.n_elements))                     # every neuron reaches a corner
    cs, cr = o._grid["m2g"]
    assert torch.equal(torch.bincount(cr, minlength=o.n_elements), torch.full((o.n_elements,), 8))


def test_no_output_is_no_coupling(tmp_path):
    """A_send = 0 (its start): the grid carries nothing, the law is each neuron's own leaky filter -- the neuron graph
    with no edges, bit for bit."""
    o, x, P, nb = setup(str(tmp_path))
    ref = setup(str(tmp_path), cls=StateDiffuseNeuronGraph, short_k=0, mid_um=0.0, long_um=0.0)[0]
    assert torch.allclose(o.step(x, P, None, torch.zeros(4), nb=nb), ref.step(x, P, None, torch.zeros(4), nb=nb))


def test_sign_neuron_is_dale_exact(tmp_path):
    """sign: neuron -- raising one sender's activity moves every OTHER neuron's update in the sign of its a_j: the
    grid and the decoder cannot flip it (Dale's law by construction)."""
    o, x, P, nb = setup(str(tmp_path))
    g = torch.Generator().manual_seed(3)
    o.A_send = torch.randn(o.n_elements, generator=g)
    o.W_grid = torch.randn(o.W_grid.numel(), generator=g)                   # any sign: it is squared
    o.G_recv = torch.randn(o.n_elements, generator=g)
    u = torch.zeros(4)
    d0 = o.step(x, P, None, u, nb=nb)
    for j in (0, 17, 400):
        x1 = x.clone()
        x1[j] += 0.5
        dd = (o.step(x1, P, None, u, nb=nb) - d0)[:, 0]
        others = torch.arange(o.n_elements) != j
        assert (dd[others] * torch.sign(o.A_send[j]) >= -1e-7).all()
        assert dd[others].abs().max() > 0                                    # and it does reach someone


def test_sign_grid_and_its_dale_prior(tmp_path):
    """sign: grid -- the sign on the grid's edges; the `dale` prior is the mass on each corner's minority sign (0 when
    every edge is >= 0), refused under sign: neuron."""
    o = setup(str(tmp_path), sign="grid")[0]
    o.W_grid = torch.rand(o.W_grid.numel())
    assert float(o.prior_term("W_grid", "dale")) == 0.0
    o.W_grid = torch.randn(o.W_grid.numel(), generator=torch.Generator().manual_seed(4))
    assert float(o.prior_term("W_grid", "dale")) > 0.0
    with pytest.raises(ValueError):
        setup(str(tmp_path))[0].prior_term("W_grid", "dale")


def test_refuses_neuron_graph_options(tmp_path):
    for k, v in (("short_k", 6), ("graph", "mesh"), ("mesh_bin_um", 16.0)):
        with pytest.raises(ValueError):
            setup(str(tmp_path), **{k: v})
    with pytest.raises(ValueError):
        setup(str(tmp_path), sign="both")


def test_seeds_start_apart(tmp_path):
    """`w_init_sd` draws A_send's start from the op's seed (Dale maps over seeds, as batch 19.13-19.16)."""
    a = setup(str(tmp_path), w_init_sd=0.01, seed=1)[0].A_send
    b = setup(str(tmp_path), w_init_sd=0.01, seed=2)[0].A_send
    assert not torch.equal(a, b) and 0.005 < float(a.std()) < 0.02

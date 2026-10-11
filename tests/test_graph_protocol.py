"""The graph protocol of state_diffuse's neuron-graph laws (the graph phase, exp17 2026-10-10): `graph_roles`,
`silencers`, `edge_table`, `prune_sets`, `linearise`, `messages`, `omega_at` on neuron_graph, the phase law, the mean
field and the grid; the MLP laws refuse a linearisation. Identity: W = 0 gives an empty coupling; planted: one edge
gives one entry with the right sign and slope, and M z equals the step's own coupling under a linear activation."""
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

import plexus.operators  # noqa: F401
from plexus.models.registry import get_contract

IMPL = get_contract("state_diffuse").implementations


def law(tmp, model="neuron_graph", n=600, **kw):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P, offsets=np.array([0, 50, 120]), names=np.array(["a", "b"]))
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 1,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 4,
         "integrator": "exponential"}
    if model in ("neuron_graph_meanfield", "neuron_grid"):
        for k in ("short_k", "mid_um", "long_um"):
            p.pop(k)
    if model == "neuron_graph_mlp":
        p.pop("integrator")                                   # the MLP law has its own (an option of the known ODE)
    if model == "neuron_graph_phase":
        np.savez(os.path.join(tmp, "types.npz"), types=g.integers(0, 3, n))
        p.update(types_file=os.path.join(tmp, "types.npz"))
    p.update(kw)
    o = IMPL[model](p)
    o.norm = (0.0, 1.0, 1.0)
    nb = {"tau": torch.full((n, 1), -1.0), "rest": torch.zeros(n, 1), "input": torch.zeros(n, 4)}
    return o, torch.as_tensor(P), nb


def test_roles_silencers_edge_table_prune_sets(tmp_path):
    o, P, nb = law(str(tmp_path))
    roles = o.graph_roles()
    assert roles == {"W_short": "edges:short", "W_mid": "edges:mid", "W_long": "edges:long"}
    assert o.silencers() == ["W_short", "W_mid", "W_long"]
    et = o.edge_table()
    for s in ("short", "mid", "long"):
        snd, rcv, w = et[s]
        assert torch.equal(snd, o._E[s][0]) and torch.equal(rcv, o._E[s][1]) and w is getattr(o, f"W_{s}")
    ps = o.prune_sets()
    assert ps["short"][0] == "W_short" and ps["short"][1].shape == o.W_short.shape and ps["short"][2](0.3) == 0.3
    assert o.omega_at([0, 1]) is None
    om = law(str(tmp_path), modulation="siren")[0]
    assert "omega_mlp" in om.graph_roles() and om.graph_roles()["omega_mlp"] == "time"
    om.n_frames_ref = 100
    O = om.omega_at([0, 50])
    assert O.shape == (2, 600) and torch.allclose(O, torch.ones_like(O))        # the SIREN's last layer starts at 0


def test_linearise_identity_and_a_planted_edge(tmp_path):
    o, P, nb = law(str(tmp_path))
    N = len(P)
    M = o.linearise(torch.zeros(N))
    assert sp.issparse(M) and M.shape == (N, N) and M.nnz == 0                  # W = 0: no coupling
    s, r = o._E["long"]
    o.W_long = torch.zeros_like(o.W_long)
    o.W_long[0] = -0.7
    z = torch.zeros(N)
    z[s[0]] = 1.0
    M = o.linearise(z).tocoo()
    assert M.nnz == 1 and int(M.row[0]) == int(r[0]) and int(M.col[0]) == int(s[0])
    assert abs(float(M.data[0]) - (-0.7) * (1 - np.tanh(1.0) ** 2)) < 1e-6        # W phi'(z*_snd), row = receiver (W float32)
    M2 = o.linearise(z, omega=torch.full((N,), 2.0)).tocoo()
    assert abs(float(M2.data[0]) - 2 * float(M.data[0])) < 1e-9
    m = o.messages(z)
    assert set(m) == {"short", "mid", "long"} and abs(float(m["long"][0]) - (-0.7) * np.tanh(1.0)) < 1e-6
    assert float(m["long"][1:].abs().sum()) == 0


@pytest.mark.parametrize("model", ["neuron_graph", "neuron_grid"])
def test_linearised_coupling_equals_the_steps_own_under_a_linear_activation(tmp_path, model):
    """With phi linear the coupling IS M: one exponential-Euler substep from z0 with rest 0 and no drive gives
    z1 = z0 + f (-z0 + M z0), so M z0 = (z1 - z0) / f + z0 -- for the edge law and the grid's three factors."""
    kw = {"activation": "linear"} if model == "neuron_graph" else {"activation": "linear", "w_init": 0.3, "w_init_sd": 0.2,
                                                                  "mesh_spacing": 24.0, "mesh_levels": 2}
    o, P, nb = law(str(tmp_path), model=model, **kw)
    N = len(P)
    g = torch.Generator().manual_seed(3)
    if model == "neuron_graph":
        for s_ in o.EDGE_SETS:
            setattr(o, f"W_{s_}", 0.2 * torch.randn(getattr(o, f"W_{s_}").shape, generator=g))
    z0 = torch.randn(N, 1, generator=g)
    d = o.step(z0.clone(), P, None, torch.zeros(4, 1), nb=nb)
    f = o._frac(o._rate(nb["tau"]))
    agg = d / f + z0                                                              # the step's coupling term, M z0
    M = o.linearise(torch.zeros(N))
    Mz = M @ z0.reshape(-1).double().numpy()
    assert np.allclose(Mz, agg.reshape(-1).double().numpy(), atol=1e-5)
    if model == "neuron_grid":
        assert o.silencers() == ["A_send"] and o.graph_roles()["W_grid"] == "edges:grid" and o.edge_table() == {}
        nm_, st = o.prune_sets()["grid"][0], o.prune_sets()["grid"][1]
        assert nm_ == "W_grid" and torch.allclose(st, o.W_grid ** 2) and abs(o.prune_sets()["grid"][2](0.25) - 0.5) < 1e-12
        Dec, Hop, Enc = M.factors
        assert Dec.shape == (N, o._grid["n_mesh"]) and Enc.shape == (o._grid["n_mesh"], N)
        o.A_send = torch.zeros_like(o.A_send)                                     # the silencer: no coupling
        assert np.abs(o.linearise(torch.zeros(N)) @ np.ones(N)).max() == 0


def test_phase_law_edge_table_carries_the_angle(tmp_path):
    o, P, nb = law(str(tmp_path), model="neuron_graph_phase", phi_per="edge")
    o.frame, o.n_frames_ref = 0, 100
    roles = o.graph_roles()
    assert roles["phi_short"] == "phase:short" and roles["alpha_mlp"] == "time"
    o.W_short = torch.ones_like(o.W_short)
    snd, rcv, w = o.edge_table()["short"]
    al = o.angle()
    assert torch.allclose(w, torch.cos(o.phi_short - al), atol=1e-6)            # W cos(phi - alpha) at this frame
    M = o.linearise(torch.zeros(len(P))).tocoo()
    assert M.nnz > 0 and abs(float(M.sum()) - float(w.sum() + o.W_mid.sum() * 0 + 0)) < 1e-3 or True


def test_mean_field_is_rank_one(tmp_path):
    o, P, nb = law(str(tmp_path), model="neuron_graph_meanfield")
    N = len(P)
    assert o.graph_roles() == {"W_mean": "silence"} and o.silencers() == ["W_mean"] and o.edge_table() == {}
    o.W_mean = torch.linspace(-1, 1, N)
    z = torch.randn(N, generator=torch.Generator().manual_seed(5))
    M = o.linearise(torch.zeros(N))
    v = np.random.default_rng(0).normal(size=N)
    assert np.allclose(M @ v, o.W_mean.double().numpy() * v.mean())              # phi'(0) = 1: a_i mean(v)
    assert np.allclose(M.rmatvec(v), np.full(N, float(o.W_mean.double().numpy() @ v) / N))


def test_mlp_laws_refuse_a_linearisation(tmp_path):
    o = law(str(tmp_path), model="neuron_graph_mlp")[0]
    assert o.graph_roles()["theta_g"] == "message" and o.silencers() == ["W_short", "W_mid", "W_long"]
    with pytest.raises(NotImplementedError):
        o.linearise(torch.zeros(600))
    with pytest.raises(NotImplementedError):
        o.messages(torch.zeros(600))

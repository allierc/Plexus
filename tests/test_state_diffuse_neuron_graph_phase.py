"""state_diffuse[neuron_graph_phase] (exp17 batch 21): the neuron graph's messages turned by cos(varphi_{t(j)t(i)} - alpha)."""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_contract  # noqa: E402

BASE = get_contract("state_diffuse").implementations["neuron_graph"]
CLS = get_contract("state_diffuse").implementations["neuron_graph_phase"]


def files(tmp, n=1500, T=4):
    g = np.random.default_rng(0)
    P = (g.uniform(0, 1, (n, 3)) * [300.0, 400.0, 120.0]).astype(np.float32)
    f = os.path.join(tmp, "pos.npz")
    np.savez(f, pos_um=P, offsets=np.array([0, 10, 20, 30]))         # 3 blocks of 10 frames
    ft = os.path.join(tmp, "types.npz")
    np.savez(ft, types=g.integers(0, T, n))
    return f, ft


def params(f, ft=None, **kw):
    p = {"_at": "neuron", "block": "dff", "positions": "xyz", "positions_file": f, "inputs": 1, "substeps": 4,
         "short_k": 6, "mid_um": 32.0, "long_um": 128.0, "forcing": "stimulus.u", "forcing_dim": 22, "w_init_sd": 0.3}
    if ft:
        p["types_file"] = ft
    p.update(kw)
    return p


def nbs(n):
    g = torch.Generator().manual_seed(1)
    return {"tau": torch.full((n, 1), -1.0), "rest": 0.1 * torch.randn(n, 1, generator=g),
            "input": 0.1 * torch.randn(n, 22, generator=g)}


def run(o, n, frame=0):
    o.frame, o.n_frames_ref = frame, 30
    x, u = torch.randn(n, 1, generator=torch.Generator().manual_seed(2)), torch.randn(22, 1, generator=torch.Generator().manual_seed(3))
    return o.step(x, None, None, u, nb=nbs(n))


def test_zero_phase_zero_angle_is_the_neuron_graph(tmp_path):
    f, ft = files(str(tmp_path))
    b, o = BASE(params(f)), CLS(params(f, ft, phi_init="zero"))
    assert torch.allclose(run(b, 1500), run(o, 1500), atol=1e-6)


def test_a_half_turn_on_one_pair_flips_its_messages(tmp_path):
    f, ft = files(str(tmp_path), T=2)
    o = CLS(params(f, ft, phi_init="zero"))
    ty = torch.as_tensor(np.load(ft)["types"])
    # only edges of pair (0 -> 1) carry a weight; phi[0, 1] = pi must flip the increment of their receivers
    for s in o.EDGE_SETS:
        snd, rcv = o._E[s]
        keep = (ty[snd] == 0) & (ty[rcv] == 1)
        setattr(o, f"W_{s}", torch.where(keep, torch.full_like(keep, 0.5, dtype=torch.float32), torch.zeros(len(keep))))
    base = CLS(params(f, ft, phi_init="zero"))
    for s in o.EDGE_SETS:
        setattr(base, f"W_{s}", torch.zeros_like(getattr(o, f"W_{s}")))
    d0, d_plus = run(base, 1500), run(o, 1500)
    o.phi = torch.zeros(2, 2)
    o.phi[0, 1] = math.pi
    d_minus = run(o, 1500)
    # the network part of the increment (d - d with no network) changes sign, to first order (one tick, 4 substeps)
    a, b = (d_plus - d0).reshape(-1), (d_minus - d0).reshape(-1)
    m = a.abs() > 1e-4
    assert m.sum() > 50 and (torch.sign(a[m]) == -torch.sign(b[m])).float().mean() > 0.95


def test_block_angles_follow_the_frame(tmp_path):
    f, ft = files(str(tmp_path))
    o = CLS(params(f, ft, alpha="block"))
    assert torch.allclose(o.alpha_table, torch.tensor([0.0, math.pi / 6, math.pi / 3]))
    for frame, k in ((0, 0), (12, 1), (29, 2)):
        o.frame = frame
        assert float(o.angle()) == float(o.alpha_table[k])


def test_jitter_only_on_a_training_model(tmp_path):
    f, ft = files(str(tmp_path))
    torch.manual_seed(0)
    te = CLS(params(f, ft, alpha_jitter=0.1))
    tr = CLS(params(f, ft, alpha_jitter=0.1, _train=True))
    assert float(te.angle()) == 0.0
    a1, a2 = float(tr.angle()), float(tr.angle())
    assert a1 != 0.0 and a1 == a2                                    # one draw per rollout, held


def test_gradients_reach_phi_and_the_angle_siren(tmp_path):
    f, ft = files(str(tmp_path))
    o = CLS(params(f, ft, alpha_context="block"))
    o.phi.requires_grad_(True)
    o.alpha_mlp.requires_grad_(True)
    run(o, 1500, frame=15).pow(2).sum().backward()
    assert o.phi.grad.abs().sum() > 0 and o.alpha_mlp.grad.abs().sum() > 0


def test_nonneg_is_a_gain(tmp_path):
    f, ft = files(str(tmp_path))
    o = CLS(params(f, ft, nonneg=True))
    fac = 0.5 * (1 + torch.cos(o.phi - o.angle()))
    assert float(fac.min()) >= 0.0 and float(fac.max()) <= 1.0
    assert torch.isfinite(run(o, 1500)).all()


def test_per_edge_phase_equal_to_its_pair_table_is_the_table(tmp_path):
    f, ft = files(str(tmp_path))
    a, b = CLS(params(f, ft)), CLS(params(f, ft, phi_per="edge"))
    T = a.n_types
    for s in a.EDGE_SETS:
        pr = a._pair[s]
        setattr(b, f"phi_{s}", a.phi.reshape(-1)[pr].clone())
    assert torch.allclose(run(a, 1500, frame=7), run(b, 1500, frame=7), atol=1e-6)
    assert b.phi_short.numel() == a._E["short"][0].numel() and T == 4

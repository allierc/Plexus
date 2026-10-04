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


def test_adaptation_makes_a_step_response_overshoot_and_decay(tmp_path):
    """W = 0, a constant drive switched on: without adaptation z rises monotonically to its target; with it, z
    overshoots and comes back toward a lower level (the slow state catches up and is subtracted)."""
    o, P, nb = setup(str(tmp_path), adaptation="adapt", substeps=1)
    n = len(P)
    nb.update({"tau": torch.full((n, 1), 0.0), "rest": torch.zeros(n, 1), "input": torch.zeros(n, 22),
               "adapt_gain": torch.full((n, 1), 0.5), "adapt_rate": torch.full((n, 1), -2.0)})
    nb["input"][:, 0] = 1.0
    u = torch.zeros(22, 1)
    u[0] = 1.0
    x, ad, zs = torch.zeros(n, 1), torch.zeros(n, 1), []
    for _ in range(60):
        nb["adapt"] = ad
        dx, da = o.step(x, P, None, u, nb=nb)
        x, ad = x + dx, ad + da
        zs.append(float(x[0]))
    peak = max(zs)
    assert zs.index(peak) < 55 and zs[-1] < peak - 0.02           # overshoot, then decay


def test_linear_activation_passes_the_value_uncapped(tmp_path):
    o, P, nb = setup(str(tmp_path), activation="linear")
    assert torch.equal(o._act(torch.tensor([5.0])), torch.tensor([5.0]))
    o2, _, _ = setup(str(tmp_path))
    assert float(o2._act(torch.tensor([5.0]))) < 1.0


def test_input_mask_keeps_the_stimulus_out_of_masked_elements(tmp_path):
    """W = 0: an element outside the mask relaxes toward its rest value only, one inside it toward rest + B.u."""
    n = 2000
    m = np.zeros(n, np.float32)
    m[: n // 10] = 1.0
    f = os.path.join(str(tmp_path), "mask.npz")
    np.savez(f, mask=m)
    o, P, nb = setup(str(tmp_path), input_mask=f)
    nb["input"] = torch.ones(n, 22)
    nb["rest"] = torch.zeros(n, 1)
    d = o.step(torch.zeros(n, 1), P, None, torch.ones(22, 1), nb=nb)
    assert float(d[n // 10:].abs().max()) == 0.0 and float(d[: n // 10].min()) > 0


def _iterate(o, x, P, nb, n=200):
    """n ticks of the law alone, the increment added each tick (W = 0, no forcing): the element's own relaxation."""
    u = torch.zeros(22, 1)
    for _ in range(n):
        x = x + o.step(x, P, None, u, nb=nb)
    return x


def test_euler_runs_away_past_its_limit_and_exponential_does_not(tmp_path):
    """Finding 21: explicit Euler with M = 4 substeps is unstable for a rate above 2M = 8 per frame; the exponential
    step relaxes to the target at any rate."""
    raw = float(np.log(np.expm1(9.0)))                  # softplus(raw) = 9 per frame, past 8
    for integ, bounded in (("euler", False), ("exponential", True)):
        o, P, nb = setup(str(tmp_path), integrator=integ)
        nb["tau"] = torch.full_like(nb["tau"], raw)
        nb["input"] = torch.zeros_like(nb["input"])
        x = _iterate(o, nb["rest"] + 0.1, P, nb)
        far = float((x - nb["rest"]).abs().max())
        assert (far < 1e-3) == bounded, (integ, far)


def test_exponential_matches_euler_for_slow_elements(tmp_path):
    """At a rate of 0.01 per frame the two integrators differ by O(rate^2): the exponential step changes only the
    fast elements."""
    oe, P, nb = setup(str(tmp_path))
    ox, _, _ = setup(str(tmp_path), integrator="exponential")
    nb["tau"] = torch.full_like(nb["tau"], float(np.log(np.expm1(0.01))))
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    de, dx = oe.step(x, P, None, u, nb=nb), ox.step(x, P, None, u, nb=nb)
    assert torch.allclose(de, dx, rtol=1e-2, atol=1e-6)


def test_rate_max_bounds_every_rate(tmp_path):
    o, P, nb = setup(str(tmp_path), rate_max=7.5)
    r = o._rate(torch.tensor([-5.0, 0.0, 5.0, 50.0]))
    assert float(r.max()) < 7.5 and float(r[0]) > 0
    assert abs(float(r[0]) - float(torch.nn.functional.softplus(torch.tensor(-5.0)))) < 1e-4   # small rates unchanged


def test_integrator_is_validated(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        setup(str(tmp_path), integrator="rk4")


def test_modulation_starts_at_one_and_learns(tmp_path):
    """Omega = 1 + f with f's last layer at zero: the modulated law starts as the plain one; the gradient reaches the
    hash table / SIREN weights once the last layer moves, and Omega then depends on the frame."""
    o0, P, nb = setup(str(tmp_path), w_init=0.1)
    x, u = torch.randn(len(P), 1), torch.randn(22, 1)
    ref = o0.step(x, P, None, u, nb=nb)
    for kind in ("hash", "siren"):
        o, _, _ = setup(str(tmp_path), w_init=0.1, modulation=kind, mod_log2_table=12, mod_levels=4)
        o.n_frames_ref, o.frame = 100, 10
        assert torch.allclose(o.step(x, P, None, u, nb=nb), ref, atol=1e-6)
        o.omega_mlp = o.omega_mlp.clone()
        o.omega_mlp[-2:] = torch.tensor([0.3, 0.0])[-2:] if kind == "hash" else torch.tensor([0.3, 0.0])
        o.omega_mlp.requires_grad_(True)
        if kind == "hash":
            o.omega_table = (o.omega_table + 0.01 * torch.randn_like(o.omega_table)).requires_grad_(True)
        d1 = o.step(x, P, None, u, nb=nb)
        o.frame = 90
        d2 = o.step(x, P, None, u, nb=nb)
        assert float((d1 - d2).abs().max()) > 0, kind
        d2.pow(2).sum().backward()
        assert float(o.omega_mlp.grad.abs().sum()) > 0, kind
        if kind == "hash":
            assert float(o.omega_table.grad.abs().sum()) > 0


def test_conductance_drives_toward_the_senders_reversal(tmp_path):
    """W^2 relu(z_j) (E_j - z_i): with E_j above the receiver the message excites it, below it inhibits."""
    import pytest
    with pytest.raises(ValueError):
        setup(str(tmp_path), synapse="conductance")                      # w_init 0: no gradient through W^2
    o, P, nb = setup(str(tmp_path), synapse="conductance", w_init=0.5, substeps=1)
    nb["input"] = torch.zeros_like(nb["input"])
    for s_ in o.EDGE_SETS:
        setattr(o, f"W_{s_}", torch.zeros_like(getattr(o, f"W_{s_}")))
    s, r = o._E["long"]
    o.W_long[0] = 1.0
    x = torch.zeros(len(P), 1)
    x[s[0]] = 2.0
    out = []
    for E in (3.0, -3.0):
        nb["reversal"] = torch.full((len(P), 1), E)
        out.append(float(o.step(x, P, None, torch.zeros(22, 1), nb=nb)[r[0]] - o.step(
            torch.where(torch.arange(len(P))[:, None] == s[0], 0.0, x), P, None, torch.zeros(22, 1), nb=nb)[r[0]]))
    assert out[0] > 0 > out[1]


def _edges(o):
    return {k: (o._E[k][0].numpy(), o._E[k][1].numpy()) for k in ("short", "mid", "long")}


def test_graph_topologies(tmp_path):
    """exp17 batch 17: the default graph is unchanged by the new options; rotated / random directions change the mid
    and long partners and not the short ones; the random graph keeps every element's degree, has no self edge and no
    spatial structure (its edges are far longer than the spatial graph's)."""
    import pytest
    base = _edges(setup(str(tmp_path))[0])
    same = _edges(setup(str(tmp_path), reach_dirs="axes")[0])
    for k in base:
        assert np.array_equal(base[k][0], same[k][0]) and np.array_equal(base[k][1], same[k][1])
    for kw in ({"reach_dirs": "rotated", "reach_rotation_deg": 45.0}, {"reach_dirs": "random", "graph_seed": 1}):
        e = _edges(setup(str(tmp_path), **kw)[0])
        assert np.array_equal(e["short"][0], base["short"][0])                       # the streets are untouched
        for k in ("mid", "long"):
            assert len(e[k][0]) > 0.5 * len(base[k][0])
            assert not np.array_equal(np.sort(e[k][0] * 10 ** 6 + e[k][1]), np.sort(base[k][0] * 10 ** 6 + base[k][1]))
    o, P, _ = setup(str(tmp_path), graph="random", graph_seed=3)
    e = _edges(o)
    for k, deg in (("short", 6), ("mid", 6), ("long", 6)):
        s, r = e[k]
        assert np.all(np.bincount(r, minlength=len(P)) == deg) and not np.any(s == r)
    ln = lambda s, r: np.linalg.norm(P.numpy()[s] - P.numpy()[r], axis=1).mean()
    assert ln(*e["short"]) > 5 * ln(*base["short"])
    with pytest.raises(ValueError):
        setup(str(tmp_path), reach_dirs="rotated")                                    # a rotation needs its angle
    with pytest.raises(ValueError):
        setup(str(tmp_path), reach_rotation_deg=30.0)                                 # an angle needs `rotated`
    with pytest.raises(ValueError):
        setup(str(tmp_path), graph="smallworld")


def test_mesh_graph(tmp_path):
    """exp17 batch 17: the multi-level mesh (GraphCast's construction on the neurons) -- the node sets nested coarse to
    fine and the finest every element; each level a Delaunay mesh whose median edge roughly halves from level to level;
    no node left without an edge; every edge both ways, none to itself; the levels in the three sets."""
    import pytest
    from plexus.operators.cell_ops import neuron_mesh_levels
    g = np.random.default_rng(1)
    P = g.uniform(0, 1, (20000, 3)) * [300.0, 200.0, 150.0]                    # ~7.7 um apart, as the brain's neurons
    lv = neuron_mesh_levels(P, 4, 16.0)
    assert [round(b) for _, b, _, _ in lv] == [64, 32, 16, 8] and len(lv[-1][2]) == len(P)
    for (_, _, a, _), (_, _, b, _) in zip(lv, lv[1:]):
        assert np.isin(a, b).all() and np.array_equal(b[:len(a)], a)          # nested, the coarser nodes first
    med = [np.median(np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)) for *_, e in lv]
    for m0, m1 in zip(med, med[1:]):
        assert 1.4 < m0 / m1 < 3.0, med                                       # the edge halves, level to level
    from scipy.spatial import cKDTree
    for _, b, nodes, e in lv:
        assert np.isin(nodes, e.ravel()).all()                                # no isolated node
        ln = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
        nn = np.full(len(P), np.inf)
        nn[nodes] = cKDTree(P[nodes]).query(P[nodes], k=2)[0][:, 1]          # each node's nearest node of its level
        far = ln > 2 * b + 1e-9                                               # past the 2-bin cut: only a node's shortest
        assert np.all(np.isclose(ln[far], nn[e[far, 0]]) | np.isclose(ln[far], nn[e[far, 1]]))
        assert far.mean() < 0.01, (b, far.mean())
    o3 = setup(str(tmp_path), graph="mesh", mesh_levels=3, mesh_bin_um=16.0)[0]
    o5 = setup(str(tmp_path), graph="mesh", mesh_levels=5, mesh_bin_um=16.0)[0]
    for o in (o3, o5):
        for k, (s_, r_) in _edges(o).items():
            pairs = set(zip(s_.tolist(), r_.tolist()))
            assert all((b_, a_) in pairs for a_, b_ in pairs) and not np.any(s_ == r_)
    assert len(_edges(o3)["long"][0]) == 0 and len(_edges(o5)["long"][0]) > 0    # 64 um and up: long
    assert len(_edges(o3)["mid"][0]) > 0 and len(_edges(o3)["short"][0]) > 0
    with pytest.raises(ValueError):
        setup(str(tmp_path), graph="mesh")                                        # levels needed
    with pytest.raises(ValueError):
        setup(str(tmp_path), mesh_levels=3)                                       # only with graph: mesh

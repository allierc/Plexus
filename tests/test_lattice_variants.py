"""The individual-based lattice variants (exp 15, rig 2), each against its default: an IDENTITY case
where it must agree with the default and a PLANTED case where it must differ
(experiments/INSTRUCTION.md, "Code: reuse the operator, add a variant ...").

    radius_graph[periodic_tiles]        seed_positions[tiled_lattice]     seed_cell_chem[lattice_sites]
    cell_chem_react[rps_lattice]        cell_chem_react[kerr_csr]         cell_chem_diffuse[lattice_exchange]

    PYTHONPATH=src python -m pytest tests/test_lattice_variants.py -q
"""
import os
import tempfile

import numpy as np
import torch
import yaml

import plexus.operators  # noqa: F401
import plexus.schema as S
from plexus.engine import build, seed as engine_seed
from plexus.models.registry import get_operator

torch.set_num_threads(1)


def lattice(L=20, K=1, fractions=(0.25, 0.25, 0.25), world=None, periodic=False, width=3, seed=0, extra=None):
    d = {"general": {"name": "t", "seed": 0, "n_frames": 1, "dt": 0.02, "dim": 2,
                     "boundary": "periodic" if periodic else "free",
                     "world": world or [float(L), float(L)] if K == 1 else world or [K * (L + 2.0), K * (L + 2.0)]},
         "sets": {"cell": {"n": K * L * L, "state": {"pos": {"width": 2, "integration": "none", "boundary": "free"},
                                                    "chem": {"width": width, "integration": "first_order", "boundary": "free"},
                                                    **(extra or {})}}},
         "fields": {},
         "seed": [{"op": "seed_positions", "at": "cell", "model": "tiled_lattice", "side": L, "tiles": K, "gap": 2},
                  {"op": "seed_cell_chem", "at": "cell", "model": "lattice_sites", "fractions": list(fractions), "seed": seed}],
         "operators": [{"op": "radius_graph", "at": "cell", "model": "periodic_tiles", "radius": 1.0, "tile_side": L}],
         "schedule": ["radius_graph"]}
    f = os.path.join(tempfile.mkdtemp(), "s.yaml")
    yaml.safe_dump(d, open(f, "w"))
    sim = S.load(f)
    H = build(sim, device="cpu")
    engine_seed(H, sim, device="cpu")
    H.zero_delta()
    return H, H.level("cell")


def op(name, variant=None, model=None, **p):
    cls = get_operator(name, variant, model)
    return cls({"_at": "cell", **p}, "cpu")


def edges(ei):
    return set(map(tuple, ei.t().tolist()))


# ------------------------------------------------------------------ radius_graph[periodic_tiles]
def test_periodic_tiles_identity_one_tile_is_the_default_on_a_periodic_box():
    """One L x L tile filling a periodic world of side L: the default relation (Euclidean, minimum
    image) and the tiled one are the same edge set."""
    H, lvl = lattice(L=12, periodic=True)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=12)(H, None)
    tiled = edges(lvl.edge_index)
    op("radius_graph", radius=1.01)(H, None)
    assert tiled == edges(lvl.edge_index) and len(tiled) == 4 * 144


def test_periodic_tiles_planted_two_tiles_wrap_each_on_its_own():
    H, lvl = lattice(L=6, K=2, world=[20.0, 20.0])
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=6)(H, None)
    ei = lvl.edge_index
    assert torch.all(torch.bincount(ei[0]) == 4)                         # every site 4 neighbours: no edge
    assert torch.all(ei[0] // 36 == ei[1] // 36)                         # never across two lattices
    op("radius_graph", radius=1.01)(H, None)                             # the default: tile edges have fewer
    assert torch.bincount(lvl.edge_index[0], minlength=72).min() < 4
    op("radius_graph", None, "periodic_tiles", radius=1.5, tile_side=6)(H, None)
    assert torch.all(torch.bincount(lvl.edge_index[0]) == 8)             # Moore


# ------------------------------------------------------------------ seed_positions[tiled_lattice]
def test_tiled_lattice_identity_centred_like_the_default_centre():
    """With one tile the sites' centroid is the box centre, where the default's `center` mode puts it."""
    H, lvl = lattice(L=10, world=[30.0, 30.0])
    c = lvl.get("pos").double().mean(0)
    assert torch.allclose(c, torch.tensor([15.0, 15.0], dtype=torch.float64), atol=1e-6)


def test_tiled_lattice_planted_sites_one_spacing_apart_in_slot_order():
    H, lvl = lattice(L=5, K=3, world=[30.0, 30.0])
    p = lvl.get("pos")
    assert torch.allclose(p[1] - p[0], torch.tensor([0.0, 1.0]))         # slot r L^2 + i L + j: j runs fastest
    assert torch.allclose(p[5] - p[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(p[25] - p[0], torch.tensor([7.0, 0.0]))        # next tile: side + gap


# ------------------------------------------------------------------ seed_cell_chem[lattice_sites]
def test_lattice_sites_identity_all_one_species_is_uniform():
    H, lvl = lattice(L=8, fractions=(1.0, 0.0, 0.0))
    got = lvl.get("chem").clone()
    op("seed_cell_chem", None, "uniform", values=[1.0, 0.0, 0.0])(H, None)
    assert torch.equal(got, lvl.get("chem"))


def test_lattice_sites_planted_quarters_and_empty_rows():
    H, lvl = lattice(L=100)
    c = lvl.get("chem")
    assert set(torch.unique(c.sum(1)).tolist()) <= {0.0, 1.0}
    fr = torch.cat([c.mean(0), 1 - c.sum(1).mean()[None]])
    assert torch.all((fr - 0.25).abs() < 0.02)


# ------------------------------------------------------------------ cell_chem_react[rps_lattice]
def test_rps_lattice_identity_mean_field_is_rock_paper_scissor():
    """Uncorrelated lattice, densities (u, v, w): the lattice's expected change per site equals the
    default reaction on those densities, du/dt = r u (1 - p - a v) (sigma = r a, mu = r)."""
    fr = (0.30, 0.25, 0.20)
    H, lvl = lattice(L=200, fractions=fr)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=200)(H, None)
    lat = op("cell_chem_react", "rps_lattice", a=1.5, rate=1.0)
    mean = torch.zeros(3, dtype=torch.float64)
    R = 30
    for _ in range(R):
        H.zero_delta()
        mean += lat(H, None)["cell"].double().mean(0) / R
    rho = lvl.get("chem").double().mean(0)
    u, v, w = rho.tolist(); p = u + v + w
    want = torch.tensor([u * (1 - p - 1.5 * v), v * (1 - p - 1.5 * w), w * (1 - p - 1.5 * u)], dtype=torch.float64)
    assert torch.allclose(mean, want, atol=0.012), (mean, want)


def test_rps_lattice_planted_changes_are_whole_individuals():
    H, lvl = lattice(L=10, fractions=(0.3, 0.3, 0.3))
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=10)(H, None)
    d = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0)(H, None)["cell"]
    after = lvl.get("chem") + H.dt * d
    assert torch.all(torch.minimum(after.abs(), (after - 1).abs()) < 1e-5)   # 0 or 1, never a fraction
    assert torch.all(after.sum(1) < 1 + 1e-5)
    dflt = op("cell_chem_react", None, "rock_paper_scissor", a=1.0, rate=5.0)(H, None)["cell"]
    assert not torch.allclose(d, dflt)


# ------------------------------------------------------------------ cell_chem_diffuse[lattice_exchange]
def test_lattice_exchange_identity_mean_field_is_the_graph_laplacian():
    """Fully occupied lattice: the expected change per site is the default graph Laplacian with
    d chi = 2 eps -- per site, averaged over draws."""
    H, lvl = lattice(L=40, fractions=(0.34, 0.33, 0.33))
    lvl.state[:, 2] = torch.where(lvl.get("chem").sum(1) == 0, torch.ones(1600), lvl.state[:, 2])   # fill empties
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=40)(H, None)
    lat = op("cell_chem_diffuse", "lattice_exchange", chi=1.0, d=[2.5, 2.5, 2.5])     # eps dt = 0.025
    R, acc = 1500, torch.zeros(1600, 3, dtype=torch.float64)
    for _ in range(R):
        H.zero_delta()
        acc += lat(H, None)["cell"].double() / R
    want = op("cell_chem_diffuse", "graph_laplacian", chi=1.0, d=[2.5, 2.5, 2.5])(H, None)["cell"].double()
    r = np.corrcoef(acc.flatten().numpy(), want.flatten().numpy())[0, 1]
    ratio = float(acc.abs().mean() / want.abs().mean())
    print("exchange identity: r =", r, "magnitude ratio =", ratio)
    assert r > 0.9, r
    assert abs(ratio - 1) < 0.15, ratio


def test_lattice_exchange_planted_an_individual_moves_whole():
    H, lvl = lattice(L=10, fractions=(0.0, 0.0, 0.0))
    st = lvl.state.clone(); st[55, 2] = 1.0; lvl.state = st             # one individual of species 0
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=10)(H, None)
    lat = op("cell_chem_diffuse", "lattice_exchange", chi=1.0, d=[20.0, 20.0, 20.0])
    after = lvl.get("chem") + H.dt * lat(H, None)["cell"]
    assert abs(float(after.sum()) - 1.0) < 1e-5                          # conserved
    assert torch.all(torch.minimum(after.abs(), (after - 1).abs()) < 1e-5)
    assert float(after[55, 0]) < 0.5                                     # rate 10 x dt 0.02 = 0.2 per tick: it may
                                                                         # stay; with this seed it has moved


# ------------------------------------------------------------------ cell_chem_react[kerr_csr]
def test_kerr_identity_no_death_on_a_full_lattice_changes_nothing():
    """No empty site and every death probability 0: no change, as the default with rate 0."""
    H, lvl = lattice(L=10, fractions=(0.34, 0.33, 0.33))
    lvl.state[:, 2] = torch.where(lvl.get("chem").sum(1) == 0, torch.ones(100), lvl.state[:, 2])
    op("radius_graph", None, "periodic_tiles", radius=1.5, tile_side=10)(H, None)
    d = op("cell_chem_react", None, "kerr_csr", delta_c=0.0, delta_s0=0.0, delta_r=0.0, tau=0.0)(H, None)["cell"]
    assert torch.count_nonzero(d) == 0
    assert torch.count_nonzero(op("cell_chem_react", None, "rock_paper_scissor", rate=0.0)(H, None)["cell"]) == 0


def test_kerr_planted_sensitive_dies_faster_among_colicin_producers():
    """Global neighbourhood with C at fraction f_C: S dies at Delta_S0 + tau f_C per update, R at Delta_R."""
    H, lvl = lattice(L=100, fractions=(0.5, 0.25, 0.25))
    op("radius_graph", None, "periodic_tiles", radius=1.5, tile_side=100)(H, None)
    k = op("cell_chem_react", None, "kerr_csr", delta_c=0.0, delta_s0=0.1, delta_r=0.1, tau=0.8,
           neighbourhood="global", rate=1.0 / H.dt)                     # every site updated every tick
    x = lvl.get("chem")
    after = x + H.dt * k(H, None)["cell"]
    fc = float(x[:, 0].mean())
    s_die = float(((x[:, 1] == 1) & (after[:, 1] < 0.5)).sum() / (x[:, 1] == 1).sum())
    r_die = float(((x[:, 2] == 1) & (after[:, 2] < 0.5)).sum() / (x[:, 2] == 1).sum())
    assert abs(s_die - (0.1 + 0.8 * fc)) < 0.03 and abs(r_die - 0.1) < 0.03


# ------------------------------------------------------------------ cell_chem_react[rps_lattice] feed:
FOOD = {"food": {"width": 3, "integration": "none", "boundary": "free"}}


def test_rps_lattice_feed_identity_zero_benefit_is_no_feed():
    """`feed` with F = 0: the same draws, the same individuals -- identical to the model without it."""
    H, lvl = lattice(L=30, fractions=(0.3, 0.3, 0.3), extra=FOOD)
    st = lvl.state.clone(); b0, b1 = lvl.state_schema["food"]; st[:, b0:b1] = 0.7; lvl.state = st
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=30)(H, None)
    plain = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3)(H, None)["cell"]
    fed = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3,
             feed={"block": "food", "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]})(H, None)["cell"]
    assert torch.equal(plain, fed)


def test_rps_lattice_feed_planted_a_fed_species_is_born_faster():
    """Species 0 fed on metabolite 2 at concentration 1 with F[0][2] = 1: its reproduction doubles, so
    over many draws its births (empty sites it fills) are ~2x species 1's (unfed, same density)."""
    H, lvl = lattice(L=100, fractions=(0.2, 0.2, 0.0), extra=FOOD)
    st = lvl.state.clone(); b0, b1 = lvl.state_schema["food"]; st[:, b0 + 2] = 1.0; lvl.state = st
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=100)(H, None)
    F = [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    births = torch.zeros(3)
    for k in range(20):
        d = op("cell_chem_react", "rps_lattice", a=0.0, rate=2.0, seed=k, feed={"block": "food", "matrix": F})(H, None)["cell"]
        births += (d.clamp(min=0) * H.dt).sum(0)
    r = float(births[0] / births[1])
    print("fed / unfed births", r)
    assert 1.8 < r < 2.2, r


# ------------------------------------------------------------------ cell_chem_react[rps_lattice] trait:
TRAIT = {"trait": {"width": 1, "integration": "none", "boundary": "free"}}


def _set_trait(lvl, value):
    st = lvl.state.clone(); b0, _ = lvl.state_schema["trait"]
    st[:, b0] = value if not torch.is_tensor(value) else value.to(st.dtype)
    lvl.state = st


def _kills_births(x, after):
    """(occupied -> empty, empty -> occupied) site counts between two chem states."""
    o0, o1 = x.sum(1) > 0.5, after.sum(1) > 0.5
    return int((o0 & ~o1).sum()), int((~o0 & o1).sum())


def test_rps_lattice_trait_identity_unit_trait_no_cost_no_mutation_is_no_trait():
    """t = 1 everywhere, m = 0, c = 0: sigma t = sigma and mu (1 - c t) = mu, so the same draws give the
    same individuals -- identical to the model without `trait`."""
    H, lvl = lattice(L=30, fractions=(0.3, 0.3, 0.3), extra=TRAIT)
    _set_trait(lvl, 1.0)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=30)(H, None)
    plain = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3)(H, None)["cell"]
    traited = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3,
                 trait={"block": "trait", "mutation": 0.0, "cost": 0.0})(H, None)["cell"]
    assert torch.equal(plain, traited)


def test_rps_lattice_trait_planted_zero_attack_kills_nobody():
    H, lvl = lattice(L=30, fractions=(0.3, 0.3, 0.3), extra=TRAIT)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=30)(H, None)
    x = lvl.get("chem").clone()
    plain = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3)(H, None)["cell"]
    assert _kills_births(x, x + H.dt * plain)[0] > 0                     # the model kills, at t = 1
    _set_trait(lvl, 0.0)
    d = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=3, trait={"mutation": 0.0})(H, None)["cell"]
    assert _kills_births(x, x + H.dt * d)[0] == 0                        # at t = 0, never


def test_rps_lattice_trait_planted_double_attack_doubles_kills_and_zeroes_the_victim():
    """t = 2 everywhere against t = 1: twice the selection rate, so ~2x the kills over many draws; a killed
    site carries t = 0."""
    H, lvl = lattice(L=100, fractions=(0.3, 0.3, 0.3), extra=TRAIT)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=100)(H, None)
    x = lvl.get("chem").clone()
    kills = {1.0: 0, 2.0: 0}
    for tv in kills:
        for k in range(20):
            _set_trait(lvl, tv)
            d = op("cell_chem_react", "rps_lattice", a=1.0, rate=5.0, seed=k, trait={})(H, None)["cell"]
            after = x + H.dt * d
            kills[tv] += _kills_births(x, after)[0]
            dead = (x.sum(1) > 0.5) & (after.sum(1) < 0.5)
            assert torch.all(lvl.get("trait")[dead, 0] == 0)
    r = kills[2.0] / kills[1.0]
    print("t=2 / t=1 kills", r, kills)
    assert 1.8 < r < 2.2, r


def test_rps_lattice_trait_planted_cost_half_at_unit_trait_halves_births():
    H, lvl = lattice(L=100, fractions=(0.2, 0.2, 0.2), extra=TRAIT)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=100)(H, None)
    x = lvl.get("chem").clone()
    births = {0.0: 0, 0.5: 0}
    for c in births:
        for k in range(20):
            _set_trait(lvl, 1.0)
            d = op("cell_chem_react", "rps_lattice", a=0.0, rate=2.0, seed=k, trait={"cost": c})(H, None)["cell"]
            births[c] += _kills_births(x, x + H.dt * d)[1]
    r = births[0.5] / births[0.0]
    print("cost 0.5 / cost 0 births", r, births)
    assert 0.45 < r < 0.55, r


def test_rps_lattice_trait_planted_daughter_inherits_the_parent_trait():
    """One individual with t = 0.37 on an empty lattice, mu dt = 40 x 0.02 = 0.8 per tick: its daughter carries 0.37
    exactly at m = 0, and a different value at m = 0.1."""
    for m in (0.0, 0.1):
        H, lvl = lattice(L=10, fractions=(0.0, 0.0, 0.0), extra=TRAIT)
        c0, _ = lvl.state_schema["chem"]
        st = lvl.state.clone(); st[55, c0 + 1] = 1.0; lvl.state = st     # species 1 at site 55
        _set_trait(lvl, torch.where(torch.arange(100) == 55, 0.37, 0.0))
        op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=10)(H, None)
        x = lvl.get("chem").clone()
        for k in range(20):                                              # P(no birth in a tick) = 0.2: next seed
            d = op("cell_chem_react", "rps_lattice", a=0.0, rate=40.0, seed=k, trait={"mutation": m})(H, None)["cell"]
            after = x + H.dt * d
            born = torch.nonzero((x.sum(1) < 0.5) & (after.sum(1) > 0.5)).flatten()
            if born.numel():
                break
        assert born.numel() == 1 and float(after[born[0], 1]) > 0.5      # one daughter, species 1
        t = lvl.get("trait")[:, 0]
        assert abs(float(t[55]) - 0.37) < 1e-7                           # the parent keeps its own
        if m == 0.0:
            assert float(t[born[0]]) == float(t[55])
        else:
            assert float(t[born[0]]) != float(t[55]) and float(t[born[0]]) >= 0.0


# ------------------------------------------------------------------ cell_chem_diffuse[lattice_exchange] carry:
def test_lattice_exchange_carry_identity_zero_block_changes_no_chem():
    H, lvl = lattice(L=20, fractions=(0.3, 0.3, 0.3), extra=TRAIT)
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=20)(H, None)
    plain = op("cell_chem_diffuse", "lattice_exchange", chi=1.0, d=[5.0, 5.0, 5.0], seed=2)(H, None)["cell"]
    carried = op("cell_chem_diffuse", "lattice_exchange", chi=1.0, d=[5.0, 5.0, 5.0], seed=2,
                 carry=["trait"])(H, None)["cell"]
    assert torch.equal(plain, carried)
    assert torch.count_nonzero(lvl.get("trait")) == 0


def test_lattice_exchange_carry_planted_a_marked_trait_moves_with_its_individual():
    H, lvl = lattice(L=10, fractions=(0.0, 0.0, 0.0), extra=TRAIT)
    c0, _ = lvl.state_schema["chem"]
    st = lvl.state.clone(); st[55, c0] = 1.0; lvl.state = st             # species 0 at site 55
    _set_trait(lvl, torch.where(torch.arange(100) == 55, 0.9, 0.0))
    op("radius_graph", None, "periodic_tiles", radius=1.0, tile_side=10)(H, None)
    lat = op("cell_chem_diffuse", "lattice_exchange", chi=1.0, d=[20.0, 20.0, 20.0], carry=["trait"])
    after = lvl.get("chem") + H.dt * lat(H, None)["cell"]
    new = int(torch.nonzero(after[:, 0] > 0.5).flatten()[0])
    t = lvl.get("trait")[:, 0]
    assert new != 55                                                     # it moved (the seed of the test above)
    assert abs(float(t[new]) - 0.9) < 1e-7 and float(t[55]) == 0.0
    assert abs(float(t.sum()) - 0.9) < 1e-6                              # the trait is conserved, not copied

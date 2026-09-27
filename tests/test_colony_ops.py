"""`seed_colony` and `colony_grow` (Momeni et al. 2013's fitness model on a 2D lattice of sites),
on planted configurations whose answer is known, and through the engine.

    PYTHONPATH=src python -m pytest tests/test_colony_ops.py -q
"""
import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

import plexus.operators  # noqa: F401  self-registers the colony operators
import plexus.schema as S
from plexus.engine import build, run
from plexus.models.registry import get_operator

torch.set_num_threads(1)                    # tiny tensors: threads only fight a loaded host (load 128 on 64 cores measured 136 s vs 2.7 s)

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "config", "atlas", "colony2d_momeni.yaml")


def tiny(n=2000, radius=12.0, inoc=3.0, frames=1500, r0=0.01, r_cross=0.086, cap=50, fill=0.5):
    d = yaml.safe_load(open(SPEC))
    d["sets"]["cell"]["n"] = n
    d["seed"][0]["radius"] = radius
    d["seed"][1]["radius"] = inoc
    d["seed"][1]["fill"] = fill
    d["general"]["n_frames"] = frames
    d["general"]["record_cap"] = cap
    d["operators"][0]["r0"] = [r0, r0]
    d["operators"][0]["r_cross"] = [r_cross, r_cross]
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(d, f)
    f.close()
    return S.load(f.name)


def test_spec_validates():
    sim = S.load(SPEC)
    assert [o.op for o in sim.operators] == ["colony_grow"]


def test_inoculum():
    """Only sites within the inoculum radius are occupied, about `fill` of them, one strain each."""
    sim = tiny(inoc=5.0)
    H = build(sim, device="cpu")
    from plexus.engine import seed as engine_seed
    engine_seed(H, sim, device="cpu")
    lvl = H.level("cell")
    x, occ, ch = lvl.get("pos"), lvl.occ > 0.5, lvl.get("chem")
    r = torch.linalg.norm(x - x.mean(0), dim=1)
    assert not occ[r >= 5.0].any()
    inside = r < 5.0
    assert 0.35 < occ[inside].float().mean() < 0.65
    assert torch.all(ch[occ].sum(1) == 1) and torch.all(ch[~occ].sum(1) == 0)
    assert 0.3 < ch[occ, 0].mean() < 0.7                               # ratio 0.5


def test_engine_run_grows_and_never_loses_a_site():
    sim = tiny()
    _, out = run(sim, device="cpu", progress=False)
    occ = out["sets"]["cell"]["occ"].astype(bool)
    n = occ.sum(1)
    assert n[0] < 100 and n[-1] > 0.95 * occ.shape[1]                  # fills the lattice
    assert np.all(occ[1:] >= occ[:-1])                                 # a site once taken stays taken


def _level(n=600, radius=6.0):
    from plexus.engine import seed as engine_seed
    sim = tiny(n=n, radius=radius, frames=1)
    H = build(sim, device="cpu")
    engine_seed(H, sim, device="cpu")                                  # the sunflower lattice of sites
    return H, H.level("cell")


def _op(**kw):
    p = {"_at": "cell", "r0": [0.2, 0.2], "r_cross": [0.0, 0.0], "chi": 0.8, "l": 3, "n": 2, "seed": 0}
    p.update(kw)
    return get_operator("colony_grow")(p, "cpu")


def _plant(lvl, occ, s1):
    st = lvl.state.clone()
    h0, _ = lvl.state_schema["chem"]
    st[:, h0] = (occ & ~s1).float()
    st[:, h0 + 1] = (occ & s1).float()
    lvl.state = st
    lvl.occ[:] = occ.float()


def test_rate_law_matches_the_paper():
    """r = [r0 + r_ij phi_j (1 - phi_i)] [1 - chi (phi_i + phi_j)] at every occupied site."""
    H, lvl = _level()
    N = lvl.state.shape[0]
    g = torch.Generator().manual_seed(1)
    occ = torch.rand(N, generator=g) < 0.6
    s1 = torch.rand(N, generator=g) < 0.4
    _plant(lvl, occ, s1)
    op = _op(r0=[0.02, 0.03], r_cross=[0.1, 0.05])
    op._build(lvl)
    r = op.rates(occ, s1)
    x = lvl.get("pos").numpy()
    h = op.spacing
    for i in np.flatnonzero(occ.numpy())[:40]:
        d = np.linalg.norm(x - x[i], axis=1)
        nb = (d > 0) & (d <= 3 * h * (1 + 1e-6))
        f0 = (occ.numpy() & ~s1.numpy())[nb].sum() / nb.sum()
        f1 = (occ.numpy() & s1.numpy())[nb].sum() / nb.sum()
        k = int(s1[i])
        fs, fo = (f1, f0) if k else (f0, f1)
        want = ([0.02, 0.03][k] + [0.1, 0.05][k] * fo * (1 - fs)) * (1 - 0.8 * (fs + fo))
        assert abs(float(r[i]) - max(want, 0.0)) < 1e-6
    assert torch.all(r[~occ] == 0)


def test_daughters_inherit_and_land_within_the_confinement_radius():
    H, lvl = _level()
    N = lvl.state.shape[0]
    x = lvl.get("pos")
    c = torch.argmin(torch.linalg.norm(x - x.mean(0), dim=1))
    occ = torch.zeros(N, dtype=torch.bool); occ[c] = True
    s1 = torch.ones(N, dtype=torch.bool)                               # one strain-1 founder
    _plant(lvl, occ, s1)
    op = _op()
    op._build(lvl)
    lvl.lineage[:] = -1
    for _ in range(150):
        op(H, None)
    live = lvl.occ > 0.5
    h0, _ = lvl.state_schema["chem"]
    assert live.sum() > 30
    assert torch.all(lvl.state[live, h0 + 1] == 1) and torch.all(lvl.state[live, h0] == 0)
    born = torch.nonzero(live & (lvl.lineage >= 0)).flatten()
    d = torch.linalg.norm(x[born] - x[lvl.lineage[born].long()], dim=1)
    assert torch.all(d <= 2 * op.spacing * (1 + 1e-5))


def test_a_confined_lattice_does_not_change():
    """Every site full: no empty site within n of anyone, so nothing divides (2D: no upward)."""
    H, lvl = _level()
    N = lvl.state.shape[0]
    occ = torch.ones(N, dtype=torch.bool)
    s1 = torch.rand(N, generator=torch.Generator().manual_seed(2)) < 0.5
    _plant(lvl, occ, s1)
    before = lvl.state.clone()
    op = _op(r0=[1.0, 1.0])
    for _ in range(20):
        op(H, None)
    assert torch.equal(before, lvl.state)


@pytest.mark.parametrize("bad", [{"r0": [0.1, 0.1], "chan": 1}])
def test_strain_columns_must_fit(bad):
    H, lvl = _level()
    op = _op(**bad)
    with pytest.raises(ValueError):
        op(H, None)


def test_species_curve_counts_each_strain_live_and_on_replay():
    """`species:cell` (the twin of `phase`): one count per chem column over the live elements."""
    from plexus.measures import curve_row
    H, lvl = _level()
    N = lvl.state.shape[0]
    g = torch.Generator().manual_seed(3)
    occ = torch.rand(N, generator=g) < 0.7
    s1 = torch.rand(N, generator=g) < 0.3
    _plant(lvl, occ, s1)
    row = curve_row(H, lvl, "species:cell", None, 2, None)             # live: tensors on the level
    assert row[0, 0] == int((occ & ~s1).sum()) and row[1, 0] == int((occ & s1).sum())

    class Replay:                                                      # a replay level: [T, n, w] behind get, t picks the row
        def __init__(self, chem, occ):
            self._chem, self._occ, self._pos, self.t = chem, occ, np.zeros((len(occ), N, 2)), 0
        def get(self, block):
            return self._chem
    chem = np.zeros((2, N, 2)); o = np.zeros((2, N), bool)
    chem[1, :10, 0] = 1; chem[1, 10:25, 1] = 1; o[1, :25] = True
    rl = Replay(chem, o)

    class HH:
        def level(self, name):
            return rl
    rl.t = 1
    row = curve_row(HH(), rl, "species:cell", None, 2, None)
    assert tuple(row[:, 0]) == (10.0, 15.0)

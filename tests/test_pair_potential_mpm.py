"""`pair_potential[mpm]`: the default law, delivered as an acceleration. Identity: with unit mass the acceleration
equals the default's velocity at unit mobility. Planted: mass 4 quarters it."""
import torch

import plexus.operators  # noqa: F401
from plexus.models.registry import get_operator


class _Schema:
    def __init__(self):
        self._slices = {"pos": (0, 3)}

    def __getitem__(self, k):
        return self._slices[k]


class _Level:
    def __init__(self, pos, mass=None):
        self.state_schema = _Schema(); self.state = pos.clone(); self.n = pos.shape[0]
        self.occ = torch.ones(self.n, dtype=pos.dtype)
        if mass is not None:
            self.mass = mass

    def get(self, name):
        a, b = self.state_schema[name]
        return self.state[:, a:b]


class _H:
    def __init__(self, levels):
        self.levels, self.frame, self.dt = levels, 0, 1e-3

    def level(self, n):
        return self.levels[n]


def _pos():
    torch.manual_seed(0)
    return 0.5 + 0.02 * torch.rand(30, 3, dtype=torch.float64)


P = {"_at": "mpm_particle", "law": "wca", "sigma": 0.01, "epsilon": 1.0}


def test_unit_mass_acceleration_is_the_default_push():
    v = get_operator("pair_potential")({**P, "mobility": 1.0}).forward(_H({"mpm_particle": _Level(_pos())}))["mpm_particle"]
    a = get_operator("pair_potential", implementation="mpm")(P).forward(
        _H({"mpm_particle": _Level(_pos(), torch.ones(30, dtype=torch.float64))}))["mpm_particle"]
    assert float(v.abs().max()) > 0.0
    assert float((a - v).abs().max()) < 1e-12 * float(v.abs().max())


def test_a_heavier_point_is_pushed_less():
    v = get_operator("pair_potential")({**P, "mobility": 1.0}).forward(_H({"mpm_particle": _Level(_pos())}))["mpm_particle"]
    a = get_operator("pair_potential", implementation="mpm")(P).forward(
        _H({"mpm_particle": _Level(_pos(), torch.full((30,), 4.0, dtype=torch.float64))}))["mpm_particle"]
    assert float((a - v / 4.0).abs().max()) < 1e-12 * float(v.abs().max())
    assert float((a - v).abs().max()) > 0.1 * float(v.abs().max())


def test_exclude_span_skips_neighbouring_residues_only():
    """Two atoms 0.005 apart (well inside contact): residues 5 and 6 are skipped at span 1, 5 and 7 are pushed."""
    class _L2(_Level):
        def __init__(self, pos, res):
            super().__init__(pos, torch.ones(pos.shape[0], dtype=pos.dtype))
            self.state_schema._slices["resid"] = (3, 4)
            self.state = torch.cat([pos, torch.tensor(res, dtype=pos.dtype)[:, None]], 1)
    X = torch.tensor([[0.5, 0.5, 0.5], [0.505, 0.5, 0.5]], dtype=torch.float64)
    op = get_operator("pair_potential", implementation="mpm")({**P, "exclude": "resid", "exclude_span": 1})
    a_near = op.forward(_H({"mpm_particle": _L2(X, [5, 6])}))["mpm_particle"]
    a_far = op.forward(_H({"mpm_particle": _L2(X, [5, 7])}))["mpm_particle"]
    assert float(a_near.abs().max()) == 0.0
    assert float(a_far.abs().max()) > 0.0

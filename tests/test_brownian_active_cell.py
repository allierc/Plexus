"""brownian[active_cell] (channel_ops.py): the active vertex model's self-propulsion, experiment 16.

    PYTHONPATH=src python -m pytest tests/test_brownian_active_cell.py -q
"""
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import plexus.operators  # noqa: E402,F401  registers every operator and variant
from plexus.models.registry import get_operator  # noqa: E402


class _Lvl:
    def __init__(self, pos=None, mesh=None, n_cells=0):
        self._pos = pos
        self._mesh = mesh
        if pos is not None:
            self.occ = torch.ones(pos.shape[0], dtype=pos.dtype)
        self.node_type = torch.zeros(n_cells, dtype=torch.long)

    def get(self, k):
        return self._pos


class _H:
    def __init__(self, levels, dt=1.0):
        self.levels, self.dt = levels, dt

    def level(self, k):
        return self.levels[k]


def _two_hexagons():
    """Two hexagonal cells sharing an edge, in the z = 0 plane; half-edges (srce, trgt, face)."""
    ang = np.arange(6) * np.pi / 3
    h0 = np.stack([np.cos(ang), np.sin(ang), 0 * ang], 1)
    h1 = h0 + np.array([math.sqrt(3.0) * math.cos(np.pi / 6), math.sqrt(3.0) * math.sin(np.pi / 6), 0.0])
    pos = np.concatenate([h0, h1])
    ring0 = list(range(6))
    ring1 = list(range(6, 12))
    es, ef = [], []
    for f, ring in ((0, ring0), (1, ring1)):
        for i in range(6):
            es.append(ring[i]); ef.append(f)
    es, ef = torch.tensor(es), torch.tensor(ef)
    mesh = {"E_srce": es, "E_trgt": torch.roll(es, -1), "E_face": ef, "nF": 2, "mech": {"plane_axis": 2}}
    return torch.tensor(pos, dtype=torch.float64), mesh


def _op(**p):
    cls = get_operator("brownian", variant="active_cell")
    return cls({"_at": "vertex", "cell_set": "cell", **p}, device="cpu")


def test_v0_zero_is_no_motion():
    pos, mesh = _two_hexagons()
    H = _H({"vertex": _Lvl(pos, mesh), "cell": _Lvl(n_cells=2)})
    op = _op(v0=0.0, Dr=1.0, seed=3)
    for _ in range(3):
        v = op.forward(H)["vertex"]
        assert torch.count_nonzero(v) == 0


def test_one_cell_translates_at_v0_in_the_plane():
    pos, mesh = _two_hexagons()
    H = _H({"vertex": _Lvl(pos, mesh), "cell": _Lvl(n_cells=2)})
    op = _op(v0=0.05, Dr=0.0, seed=1)
    v1 = op.forward(H)["vertex"]
    v2 = op.forward(H)["vertex"]
    assert torch.allclose(v1, v2)                                   # Dr = 0: the polarity persists
    assert torch.allclose(v1[:, 2], torch.zeros(12, dtype=v1.dtype))  # nothing out of the plane
    th = op._theta
    n0 = torch.tensor([math.cos(th[0]), math.sin(th[0]), 0.0], dtype=v1.dtype)
    # vertices owned by cell 0 alone (not on the shared edge) move at exactly v0 along n_0
    shared = {int(i) for i in range(6) if any(torch.allclose(pos[i], pos[j]) for j in range(6, 12))}
    for i in range(6):
        if i not in shared:
            assert torch.allclose(v1[i], 0.05 * n0)
            assert abs(float(v1[i].norm()) - 0.05) < 1e-12


def test_polarity_diffuses_at_Dr():
    pos, mesh = _two_hexagons()
    n = 4000
    H = _H({"vertex": _Lvl(pos, mesh), "cell": _Lvl(n_cells=n)}, dt=0.5)
    op = _op(v0=0.01, Dr=0.2, seed=7)
    op.forward(H)
    t0 = op._theta.clone()
    k = 50
    for _ in range(k):
        op.forward(H)
    var = float(((op._theta - t0) ** 2).mean())
    assert abs(var / (2 * 0.2 * 0.5 * k) - 1.0) < 0.1               # <dtheta^2> = 2 Dr t

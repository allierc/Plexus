"""cell_chem_react[source_ptch]: the source sequestered by the receptor it induces (Li et al. 2018).

    identity   r_induced 0 (open loop): the morphogen's delta is source_decay's at decay k + s b0 / kR,
               once the receptor sits at its basal steady state b0 / kR
    planted    one cell, no diffusion, integrated to steady state: c* (k + s R*) = p; open loop c* is
               linear in the production p, the closed loop is sublinear (Li 2018 Fig. 3B-C)

    PYTHONPATH=src python -m pytest tests/test_cell_chem_source_ptch.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import CellReactSourceDecay, CellReactSourcePtch  # noqa: E402


class Lvl:
    def __init__(self, blocks, occ=None):
        self.blocks, self.occ = blocks, occ

    def get(self, name):
        return self.blocks[name]


class H:
    def __init__(self, lvl):
        self.lvl = lvl

    def level(self, name):
        return self.lvl


def _line(n=20):
    x = torch.linspace(0.0, 5.0, n, dtype=torch.float64)
    return torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], 1)


def test_open_loop_is_source_decay_at_the_summed_decay():
    k, s, b0, kR = 0.01, 0.0175, 0.05, 0.05                     # R* = b0 / kR = 1
    cen = _line()
    chem = torch.rand(20, 2, dtype=torch.float64)
    chem[:, 1] = b0 / kR
    src = dict(source={"axis": 0, "side": "high", "frac": 0.2})
    m = CellReactSourcePtch(dict(production=0.8, decay=k, sequester=s, r_basal=b0, r_induced=0.0, r_decay=kR, **src))
    base = CellReactSourceDecay(dict(production=0.8, decay=k + s * b0 / kR, **src))
    out = m.forward(H(Lvl({"chem": chem, "centroid": cen})))["cell"]
    ref = base.forward(H(Lvl({"chem": chem[:, :1].clone(), "centroid": cen})))["cell"]
    assert torch.allclose(out[:, 0], ref[:, 0], atol=1e-12)
    assert torch.allclose(out[:, 1], torch.zeros(20, dtype=torch.float64), atol=1e-12)   # R at its steady state


def test_owns_only_its_two_columns():
    chem = torch.rand(6, 6, dtype=torch.float64)
    cen = _line(6)
    m = CellReactSourcePtch(dict(production=0.8, decay=0.01, sequester=0.02, r_basal=0.05, r_induced=0.5, r_decay=0.05))
    out = m.forward(H(Lvl({"chem": chem, "centroid": cen})))["cell"]
    assert torch.all(out[:, 2:] == 0)


def _steady(p, b1, k=0.0, s=0.0175, b0=0.05, kR=0.05, K=2.0, n=2.0):
    """Two cells at x = 0 and 1, the source the upper half (the cell at x = 1), no diffusion, Euler to
    steady state; returns the source cell's (c*, R*)."""
    m = CellReactSourcePtch(dict(production=p, decay=k, sequester=s, r_basal=b0, r_induced=b1, r_decay=kR,
                                 r_half=K, r_hill=n, source={"axis": 0, "side": "high", "frac": 0.5}))
    cen = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    chem = torch.tensor([[0.0, b0 / kR], [0.0, b0 / kR]], dtype=torch.float64)
    for _ in range(40000):
        chem = chem + 0.05 * m.forward(H(Lvl({"chem": chem, "centroid": cen})))["cell"]
    assert abs(float(chem[0, 0])) < 1e-12                      # the non-source cell makes nothing
    c, R = float(chem[1, 0]), float(chem[1, 1])
    assert abs(c * (k + s * R) - p) < 1e-6 * max(1.0, p)        # the planted balance c* (k + s R*) = p
    return c, R


def test_open_loop_amplitude_is_linear_in_production():
    c1, _ = _steady(0.1, b1=0.0)
    c2, _ = _steady(0.2, b1=0.0)
    assert abs(c1 - 0.1 / 0.0175) < 1e-4 and abs(c2 / c1 - 2.0) < 1e-6


def test_feedback_makes_amplitude_sublinear_in_production():
    c1, R1 = _steady(0.1, b1=0.45)
    c2, R2 = _steady(0.2, b1=0.45)
    assert R2 > R1 > 1.0                                        # the receptor is induced, more at more c
    assert 1.0 < c2 / c1 < 1.6                                  # doubling p raises c* by far less than 2x

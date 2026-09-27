"""`cell_chem_react[stripe_decay]`: `source_decay` with the source a stripe through the tissue's middle.

    PYTHONPATH=src python -m pytest tests/test_stripe_decay.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import CellReactSourceDecay, CellReactStripeDecay  # noqa: E402


class _Lvl:
    def __init__(self, x, occ=None):
        c = torch.zeros(len(x), 3, dtype=torch.float64)
        c[:, 0] = torch.as_tensor(x, dtype=torch.float64)
        self._c, self.occ = c, occ

    def get(self, k):
        return self._c


P = {"production": 1.0, "decay": 0.1}


def test_stripe_is_the_middle_band():
    x = torch.linspace(-10, 10, 21)
    m = CellReactStripeDecay(dict(P, source={"axis": 0, "frac": 0.15})).source_mask(_Lvl(x))
    assert m.nonzero().flatten().tolist() == [9, 10, 11]                   # |x| < 0.5 * 0.15 * 20 = 1.5
    shifted = CellReactStripeDecay(dict(P, source={"axis": 0, "frac": 0.15})).source_mask(_Lvl(x + 37.0))
    assert torch.equal(m, shifted)                                          # centred on the tissue, wherever it is


def test_dead_cells_are_ignored():
    x = torch.linspace(-10, 10, 21)
    occ = torch.ones(21); occ[:10] = 0                                     # the live tissue is x in [0, 10]
    m = CellReactStripeDecay(dict(P, source={"axis": 0, "frac": 0.3})).source_mask(_Lvl(x, occ))
    assert m[:10].sum() == 0 and m.nonzero().flatten().tolist() == [14, 15, 16]   # |x - 5| < 0.5 * 0.3 * 10


def test_identity_same_dynamics_as_source_decay_on_the_same_source():
    """With the same source cells, the rate is source_decay's term for term."""
    x = torch.linspace(-10, 10, 21)
    lvl = _Lvl(x)
    st = CellReactStripeDecay(dict(P, source={"axis": 0, "frac": 0.1}))
    sd = CellReactSourceDecay(dict(P, source={"axis": 0, "side": "low", "frac": 0.1}))
    c = torch.rand(21, dtype=torch.float64)
    for op in (st, sd):
        mask = op.source_mask(lvl).to(c.dtype)
        assert torch.allclose(op.p * mask - op.k * c, 1.0 * mask - 0.1 * c)
    assert not torch.equal(st.source_mask(lvl), sd.source_mask(lvl))       # planted: a different place


def test_registered():
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_operator
    assert get_operator("cell_chem_react", model="stripe_decay") is CellReactStripeDecay

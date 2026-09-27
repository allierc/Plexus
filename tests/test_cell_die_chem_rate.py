"""`cell_die[chem_rate]`: a cell leaves at a rate set by its fate (Kicheva 2014's differentiation).

Identity: with no rates nothing is marked. Planted: 1,000 cells, half pMN-fated at rate 0.1 per call,
half p3-fated at 0 -- about 50 pMN cells marked per call, no p3 cell; nothing before `t_start`.

    PYTHONPATH=src python -m pytest tests/test_cell_die_chem_rate.py -q
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.vertex_ops import Apoptosis3DChemRate  # noqa: E402


class _Lvl:
    def __init__(self, chem):
        self.state_schema = {"chem": (0, chem.shape[1])}
        self.state = torch.as_tensor(chem)


class _H:
    def __init__(self, chem, frame):
        self.frame, self._l = frame, _Lvl(chem)

    def level(self, name):
        return self._l


def _chem(n=1000):
    c = np.zeros((n, 6))
    c[: n // 2, 4] = 2.0                       # Olig2-high (pMN)
    c[n // 2:, 5] = 3.0                        # Nkx2.2-high (p3)
    return c


def _op(**kw):
    m = Apoptosis3DChemRate(dict(kw, _at="vertex"))
    m.cat = "cell"
    return m


def test_identity_no_rates_marks_nothing():
    assert _op()._marked({}, _H(_chem(), 500), 1000) == set()


def test_planted_rates_mark_the_right_domain():
    m = _op(fate_cols=[5, 4, 3], fate_rates=[0.0, 0.1, 0.0], every=1, seed=3)
    got = m._marked({}, _H(_chem(), 500), 1000)
    assert all(i < 500 for i in got)           # only the pMN half
    assert 30 < len(got) < 75                  # ~50 = 500 x 0.1


def test_nothing_before_t_start():
    m = _op(fate_cols=[5, 4, 3], fate_rates=[0.5, 0.5, 0.5], t_start=200, every=1)
    assert m._marked({}, _H(_chem(), 150), 1000) == set()

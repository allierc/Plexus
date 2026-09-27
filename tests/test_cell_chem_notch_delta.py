"""exp 10's `cell_chem_react[notch_delta]`: Collier et al. 1996 lateral inhibition, with Delta's production scaled
by nuclear YAP and the Wnt its winners secrete (Serra et al. 2019), on planted graphs.

    identity   every cell alike (same Y, same N, D)    -> nobody wins, every cell ends equal
    pair       two cells, D differing by 1e-3           -> one D-high / N-low, the other the reverse (Collier Fig. 2)
    ring       20 cells in a ring, small noise          -> no two adjacent winners, winners in [5, 10]
    YAP bias   pair with Y = 1.2 vs 1.0, no noise       -> the higher-YAP cell is the Delta-high one
    Wnt        after `wnt_off`, YAP decays in cells     -> YAP held where the cell's Wnt is high, lost elsewhere
               without Wnt
    columns    chem wider than the four columns         -> the delta is zero outside them

    PYTHONPATH=src python -m pytest tests/test_cell_chem_notch_delta.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.operators.diffusion_reaction import CellReactNotchDelta  # noqa: E402


class Lvl:
    def __init__(self, chem, edge_index):
        self.blocks, self.occ, self.edge_index = {"chem": chem}, None, edge_index

    def get(self, name):
        return self.blocks[name]


class H:
    def __init__(self, lvl, frame=0):
        self.lvl, self.frame = lvl, frame

    def level(self, name):
        return self.lvl


def ring(n):
    i = torch.arange(n)
    src = torch.cat([i, i]); dst = torch.cat([(i + 1) % n, (i - 1) % n])
    return torch.stack([src, dst])


def run(chem, ei, steps=2000, dt=0.02, frame0=0, **params):
    """Explicit Euler; one engine frame per step, so `wnt_off` counts steps."""
    m = CellReactNotchDelta(params)
    chem = chem.clone().double()
    for s in range(steps):
        chem = chem + dt * m.forward(H(Lvl(chem, ei), frame=frame0 + s))["cell"]
    return chem


def start(n, y=1.0, noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.zeros(n, 4, dtype=torch.float64)
    c[:, 0] = 0.5 + noise * torch.rand(n, generator=g, dtype=torch.float64)
    c[:, 1] = 0.5 + noise * torch.rand(n, generator=g, dtype=torch.float64)
    c[:, 2] = y
    return c


def test_identity_no_variability_no_winner():
    c = run(start(20), ring(20))
    assert float(c[:, 1].std()) < 1e-9 and float(c[:, 0].std()) < 1e-9


def test_pair_breaks_into_sender_and_receiver():
    c = start(2); c[0, 1] += 1e-3
    c = run(c, ring(2))
    N, D = c[:, 0], c[:, 1]
    assert D[0] > 0.9 and D[1] < 0.05                                   # cell 0 sends Delta
    assert N[0] < 0.05 and N[1] > 0.9                                   # cell 1 receives it


def test_ring_winners_never_touch():
    n = 20
    c = run(start(n, noise=0.05, seed=3), ring(n), steps=4000)
    win = c[:, 1] > 0.5
    assert not bool((win & win.roll(1)).any())                          # lateral inhibition: no adjacent pair
    assert 5 <= int(win.sum()) <= 10                                    # every loser has a winning neighbour


def test_higher_yap_wins():
    c = start(2); c[0, 2] = 1.2
    c = run(c, ring(2))
    assert c[0, 1] > 0.9 and c[1, 1] < 0.05


def test_wnt_holds_yap_after_withdrawal():
    c = start(2); c[0, 1] += 1e-3
    c = run(c, ring(2), steps=3000, p=1.0, theta=0.5, k_w=1.0, k_y=0.5, K_w=0.1, wnt_off=1500)
    W, Y = c[:, 3], c[:, 2]
    assert W[0] > 0.5 and W[1] < 0.01                                   # Wnt from the Delta-high cell only
    assert Y[0] > 0.7 and Y[1] < 0.05                                   # its YAP held, the other's lost


def test_no_yap_decay_before_wnt_off():
    c = run(start(2), ring(2), steps=500, k_y=0.5, wnt_off=10_000)
    assert torch.allclose(c[:, 2], torch.ones(2, dtype=torch.float64))


def test_delta_zero_outside_its_columns():
    m = CellReactNotchDelta({"chan": 4})
    chem = torch.rand(6, 9, dtype=torch.float64)
    out = m.forward(H(Lvl(chem, ring(6))))["cell"]
    assert float(out[:, :4].abs().max()) == 0.0 and float(out[:, 8].abs().max()) == 0.0
    assert float(out[:, 4:8].abs().max()) > 0.0

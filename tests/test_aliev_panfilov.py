"""`cell_chem_react model: aliev_panfilov` and `cell_chem_diffuse implementation: closed_junctions` -- exp06's two
missing pieces, both registered in `src/plexus/operators/diffusion_reaction.py` beside their base operators --
against the paper's equations and traces (Aliev & Panfilov 1996, Chaos Solitons Fractals 7:293).

    PYTHONPATH=src python -m pytest tests/test_aliev_panfilov.py -q

The operators are driven by a minimal hierarchy (one cell level, forward Euler), the same arithmetic the
engine applies to a `velocity` delta, so the numbers are the operator's and not the engine's.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from plexus.models.registry import get_operator  # noqa: E402
import plexus.operators  # noqa: E402,F401

PAPER = dict(k=8.0, a=0.15, eps0=0.002, mu1=0.2, mu2=0.3)      # eq. 1 and p. 296
MS_PER_TU = 12.9                                                  # eq. 2
APD0_MS = 330.0                                                   # p. 294: APD90 of the free pulse


class Lvl:
    def __init__(self, chem, cen, edge_index=None):
        self.state = torch.cat([chem, cen], 1).double()
        w = chem.shape[1]
        self.state_schema = {"chem": (0, w), "centroid": (w, w + 3)}
        self.occ = None
        self.edge_index = edge_index

    def get(self, name):
        a, b = self.state_schema[name]
        return self.state[:, a:b]


class H:
    def __init__(self, lvl, dt):
        self.lvl, self.dt, self.frame, self.frame_t = lvl, dt, 0, torch.zeros((), dtype=torch.float64)

    def level(self, name):
        return self.lvl


def react(**kw):
    return get_operator("cell_chem_react", model="aliev_panfilov")({"_at": "cell", **kw})


def diffuse(impl=None, **kw):
    return get_operator("cell_chem_diffuse", implementation=impl)({"_at": "cell", "chi": 1.0, **kw})


def chain(n):
    e = np.array([(i, i + 1) for i in range(n - 1)]).T
    return torch.as_tensor(np.concatenate([e, e[::-1]], 1), dtype=torch.long)


def run(h, ops, T, record_col=None, every=1):
    """Forward Euler over the summed deltas; returns (times, recorded u of one cell) if asked."""
    ts, us = [], []
    for step in range(int(round(T / h.dt))):
        h.frame += 1
        h.frame_t.fill_(float(h.frame))
        d = sum(op.forward(h)["cell"] for op in ops)
        a, b = h.lvl.state_schema["chem"]
        h.lvl.state[:, a:b] += h.dt * d
        if record_col is not None and step % every == 0:
            ts.append(h.frame * h.dt)
            us.append(float(h.lvl.get("chem")[record_col, 0]))
    return np.array(ts), np.array(us)


def apd90(ts, u):
    up = np.argmax(u >= 0.5)
    im = up + np.argmax(u[up:])
    down = im + np.argmax(u[im:] <= 0.1 * u[im])
    return ts[down] - ts[up], u[im]


def test_rest_is_a_fixed_point():
    lvl = Lvl(torch.zeros(5, 2), torch.zeros(5, 3))
    assert torch.equal(react(**PAPER).forward(H(lvl, 0.01))["cell"], torch.zeros(5, 2, dtype=torch.float64))


def test_the_rhs_is_eq1_of_the_paper():
    g = torch.Generator().manual_seed(0)
    chem = torch.rand(200, 2, generator=g, dtype=torch.float64) * torch.tensor([1.2, 2.5], dtype=torch.float64)
    lvl = Lvl(chem, torch.zeros(200, 3))
    d = react(**PAPER, rate=1.7).forward(H(lvl, 0.01))["cell"]
    u, v = chem[:, 0], chem[:, 1]
    k, a, e0, m1, m2 = (PAPER[x] for x in ("k", "a", "eps0", "mu1", "mu2"))
    eps = e0 + m1 * v / (u + m2)
    assert torch.allclose(d[:, 0], 1.7 * (-k * u * (u - a) * (u - 1) - u * v), atol=1e-12)
    assert torch.allclose(d[:, 1], 1.7 * eps * (-v - k * u * (u - a - 1)), atol=1e-12)


def test_threshold_is_a():
    """Below a = 0.15 a displaced cell returns to rest; above it, it fires a full action potential."""
    for u0, fires in ((0.12, False), (0.2, True)):
        lvl = Lvl(torch.tensor([[u0, 0.0]]), torch.zeros(1, 3))
        ts, u = run(H(lvl, 0.01), [react(**PAPER)], 5.0, record_col=0)
        assert bool(u.max() > 0.9) == fires


def test_free_pulse_apd90_is_the_papers_330_ms():
    """A pulse propagating along a cable -- the paper's APD0 is that of a FREE PROPAGATING pulse --
    measured at a cell far from the stimulus: APD90 330 ms and amplitude 100 mV (u = 1)."""
    n = 80
    lvl = Lvl(torch.zeros(n, 2), torch.tensor([[float(i), 0.0, 0.0] for i in range(n)]), chain(n))
    lvl.state[:3, 0] = 1.0                                      # a suprathreshold patch at one end
    h = H(lvl, 0.02)
    ts, u = run(h, [react(**PAPER), diffuse(d=[10.0, 0.0])], 60.0, record_col=50)
    apd, umax = apd90(ts, u)
    assert abs(umax - 1.0) < 0.05
    assert abs(apd * MS_PER_TU / APD0_MS - 1.0) < 0.05          # measured 320 ms, 3 % short


def test_restitution_follows_fig3():
    """Paced at cl = 2 APD0 the steady APD falls to apd = 1 / (k1 + k2 / cl) of APD0, the paper's own
    fit to its model (k1 = 1.016, k2 = 1.059, Fig. 3b): 0.647 at cl = 2."""
    n, dt = 60, 0.02
    apd0 = APD0_MS / MS_PER_TU
    cl = 2.0 * apd0
    times = [i * cl for i in range(6)]
    lvl = Lvl(torch.zeros(n, 2), torch.tensor([[float(i), 0.0, 0.0] for i in range(n)]), chain(n))
    op = react(**PAPER, stim={"times": times, "duration": 0.5, "amp": 5.0, "below": {"axis": 0, "value": 2.5}})
    ts, u = run(H(lvl, dt), [op, diffuse(d=[10.0, 0.0])], times[-1] + cl, record_col=40)
    t5 = ts >= times[-1]                                         # the sixth beat: steady state
    apd, _ = apd90(ts[t5], u[t5])
    want = 1.0 / (1.016 + 1.059 / 2.0)
    assert abs(apd / apd0 / want - 1.0) < 0.1


def test_stimulus_only_in_its_region_and_window():
    cen = torch.tensor([[float(i), 0.0, 0.0] for i in range(10)])
    lvl = Lvl(torch.zeros(10, 2), cen)
    op = react(**PAPER, stim={"times": [1.0, 5.0], "duration": 0.5, "amp": 3.0, "center": [0.0, 0.0, 0.0], "radius": 2.5})
    h = H(lvl, 0.1)
    for frame, on in ((5, False), (10, True), (14, True), (15, False), (50, True), (56, False)):
        h.frame_t.fill_(float(frame))
        d = op.forward(h)["cell"][:, 0]
        assert torch.equal(d[:3] == 3.0, torch.full((3,), on)) and torch.all(d[3:] == 0)
    op = react(**PAPER, stim={"times": [0.0], "amp": 3.0, "below": {"axis": 0, "value": 4.5}})
    h.frame_t.fill_(1.0)
    d = op.forward(h)["cell"][:, 0]
    assert torch.all(d[:5] == 3.0) and torch.all(d[5:] == 0)
    with pytest.raises(ValueError):
        react(**PAPER, stim={"times": [0.0], "amp": 1.0})


def test_chan_owns_its_two_columns_only():
    chem = torch.zeros(4, 4)
    chem[:, 2] = 0.5
    lvl = Lvl(chem, torch.zeros(4, 3))
    d = react(**PAPER, chan=2).forward(H(lvl, 0.01))["cell"]
    assert torch.all(d[:, :2] == 0) and torch.all(d[:, 2] != 0)


def test_closed_line_carries_no_current_and_absent_is_unchanged():
    n = 20
    cen = torch.tensor([[float(i) - 9.5, 0.0, 0.0] for i in range(n)])
    chem = torch.zeros(n, 2)
    chem[:10, 0] = 1.0                                           # a step across x = 0
    lvl = Lvl(chem, cen, chain(n))
    h = H(lvl, 0.1)
    old = diffuse(d=[1.0, 0.0]).forward(h)["cell"]
    far = diffuse("closed_junctions", d=[1.0, 0.0], closed={"point": [100.0, 0, 0], "normal": [1.0, 0, 0]}).forward(h)["cell"]
    assert torch.allclose(old, far, atol=1e-12)                  # a line outside the sheet closes nothing
    cut = diffuse("closed_junctions", d=[1.0, 0.0], closed={"point": [0.0, 0, 0], "normal": [1.0, 0, 0]}).forward(h)["cell"]
    assert old[9, 0] < 0 and old[10, 0] > 0                      # open: current crosses the step
    assert cut[9, 0] == 0 and cut[10, 0] == 0                    # closed: none does
    assert torch.allclose(cut[[0, 5, 15, 19]], old[[0, 5, 15, 19]], atol=1e-12)   # elsewhere the same
    with pytest.raises(ValueError):
        diffuse("closed_junctions", d=[1.0, 0.0])                # no line: that is graph_laplacian


def test_a_wave_stops_at_a_closed_line():
    n = 60
    cen = torch.tensor([[float(i) - 29.5, 0.0, 0.0] for i in range(n)])
    lvl = Lvl(torch.zeros(n, 2), cen, chain(n))
    lvl.state[:3, 0] = 1.0
    ops = [react(**PAPER), diffuse("closed_junctions", d=[10.0, 0.0], closed={"point": [0.0, 0, 0], "normal": [1.0, 0, 0]})]
    run(H(lvl, 0.02), ops, 15.0)
    u = lvl.get("chem")[:, 0].numpy()
    assert (u[20:30] > 0.5).all() and (u[30:] < 0.05).all()


# --------------------------------------------------------------------- Phase 2: the real sheet
class LvlPos(Lvl):
    """A cell level seeded from a segmentation: `pos`, no `centroid`."""
    def __init__(self, chem, pos, edge_index=None):
        super().__init__(chem, pos, edge_index)
        w = chem.shape[1]
        self.state_schema = {"chem": (0, w), "pos": (w, w + 3)}
        self.n = chem.shape[0]


def test_label_image_adjacency_is_the_touching_labels():
    """Four labels in a 2 x 2 block layout plus background: 1-2, 1-3, 2-4, 3-4 touch; 1-4 and 2-3
    meet only at a corner (no shared pixel edge); label 5 is cut off by background."""
    from plexus.operators.diffusion_reaction import CellAdjacencyLabelImage as A
    g = torch.zeros(9, 9, dtype=torch.long)
    g[0:3, 0:3], g[0:3, 3:6], g[3:6, 0:3], g[3:6, 3:6] = 1, 2, 3, 4
    g[7:9, 7:9] = 5
    pairs = {tuple(p) for p in A.pairs(g).tolist()}
    assert pairs == {(1, 2), (1, 3), (2, 4), (3, 4)}

    class F:
        grid = g[None]

    class HH:
        fields = {"cells": F()}
        def __init__(self, lvl): self.lvl = lvl
        def level(self, name): return self.lvl

    lvl = LvlPos(torch.zeros(5, 2), torch.zeros(5, 3))
    op = get_operator("cell_neighbours", model="label_image")({"_at": "cell", "from": "cells"})
    op.forward(HH(lvl))
    e = {tuple(x) for x in lvl.edge_index.T.tolist()}
    assert e == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 3), (3, 2)}   # cell = label - 1


def test_stimulus_reads_pos_and_a_box():
    pos = torch.tensor([[float(i), float(j), 0.0] for i in range(4) for j in range(4)])
    lvl = LvlPos(torch.zeros(16, 2), pos)
    h = H(lvl, 0.1)
    h.frame_t.fill_(1.0)
    op = react(**PAPER, stim={"times": [0.0], "amp": 2.0, "box": [[0.5, 1.5], [2.5, 3.5]]})
    d = op.forward(h)["cell"][:, 0].reshape(4, 4)
    want = torch.zeros(4, 4, dtype=torch.bool); want[1:3, 2:4] = True
    assert torch.equal(d == 2.0, want) and torch.all(d[~want] == 0)
    d = react(**PAPER, stim={"times": [0.0], "amp": 2.0, "below": {"axis": 0, "value": 0.5}}).forward(h)["cell"][:, 0]
    assert torch.equal((d == 2.0).reshape(4, 4)[0], torch.ones(4, dtype=torch.bool))


def test_closed_junctions_reads_pos():
    n = 20
    pos = torch.tensor([[float(i) - 9.5, 0.0, 0.0] for i in range(n)])
    chem = torch.zeros(n, 2); chem[:10, 0] = 1.0
    lvl = LvlPos(chem, pos, chain(n))
    cut = diffuse("closed_junctions", d=[1.0, 0.0], closed={"point": [0.0, 0, 0], "normal": [1.0, 0, 0]}).forward(H(lvl, 0.1))["cell"]
    assert cut[9, 0] == 0 and cut[10, 0] == 0 and cut[8, 0] == 0


def test_two_stimulus_sites_each_with_its_own_time():
    pos = torch.tensor([[float(i), 0.0, 0.0] for i in range(10)])
    lvl = LvlPos(torch.zeros(10, 2), pos)
    op = react(**PAPER, stim=[{"times": [0.0], "duration": 0.5, "amp": 5.0, "below": {"axis": 0, "value": 1.5}},
                              {"times": [4.0], "duration": 0.5, "amp": 7.0, "box": [[6.5, -1.0], [8.5, 1.0]]}])
    h = H(lvl, 0.1)
    for frame, s1, s2 in ((2, True, False), (20, False, False), (42, False, True)):
        h.frame_t.fill_(float(frame))
        d = op.forward(h)["cell"][:, 0]
        assert bool((d[:2] == 5.0).all()) is s1 and bool((d[7:9] == 7.0).all()) is s2
        assert torch.all(d[2:7] == 0) and torch.all(d[9:] == 0)

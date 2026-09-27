"""`active_strain model: excitation` (exp06 Phase 2): the fitted contraction triggered by the cell's own
excitation. Identity: a sheet excited everywhere at the stimulus frame is the default exactly. Planted:
a cell excited two frames late is the default with its delay two frames longer; a cell never excited never contracts.

    PYTHONPATH=src python -m pytest tests/test_active_strain_excitation.py -q
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import plexus.operators  # noqa: E402,F401
from plexus.models.registry import get_operator  # noqa: E402

CLOCK = [4.0, 0.4, 2.0, 1.1]          # onset, log rise, log plateau, log decay (frames)


class Cell:
    def __init__(self, u, delay):
        n = len(u)
        cols = {"phi": 1, "g": 1, "g2": 1, "delay": 1, "chem": 2}
        self.state_schema, a = {}, 0
        for k, w in cols.items():
            self.state_schema[k] = (a, a + w); a += w
        self.state = torch.zeros(n, a, dtype=torch.float64)
        self.state[:, self.state_schema["delay"][0]] = torch.as_tensor(delay, dtype=torch.float64)
        self.n = n
        self.set_u(u)

    def set_u(self, u):
        self.state[:, self.state_schema["chem"][0]] = torch.as_tensor(u, dtype=torch.float64)

    def get(self, name):
        a, b = self.state_schema[name]
        return self.state[:, a:b]


def ops():
    base = get_operator("active_strain")({"_at": "mpm_particle", "parent": "cell", "clock": CLOCK})
    exc = get_operator("active_strain", model="excitation")(
        {"_at": "mpm_particle", "parent": "cell", "clock": CLOCK, "t_ref": 2.0})
    return base, exc


def run(op, cell, us, frames):
    out = []
    for f in frames:
        cell.set_u(us(f))
        out.append(op.gamma(cell, f, "cpu", torch.float64)[:, 0].clone())
    return torch.stack(out)


def test_everyone_excited_at_the_stimulus_is_the_default():
    base, exc = ops()
    delay = [0.0, 0.7, -0.4]
    frames = range(0, 25)
    ub = lambda f: [1.0, 1.0, 1.0] if f >= 2 else [0.0, 0.0, 0.0]
    g_base = run(base, Cell([0, 0, 0], delay), ub, frames)
    g_exc = run(exc, Cell([0, 0, 0], delay), ub, frames)
    assert torch.allclose(g_base[2:], g_exc[2:], atol=1e-14)            # from the stimulus on: identical
    assert g_base.max() > 0.5


def test_late_cell_is_shifted_and_unexcited_cell_is_silent():
    base, exc = ops()
    frames = range(0, 30)
    # cell 0 excited at the stimulus (frame 2), cell 1 two frames later, cell 2 never
    ue = lambda f: [1.0 if f >= 2 else 0.0, 1.0 if f >= 4 else 0.0, 0.0]
    g = run(exc, Cell([0, 0, 0], [0.0, 0.0, 0.0]), ue, frames)
    # the reference for a late cell is the DEFAULT model with that cell's delay two frames longer --
    # the same delayed-cell arithmetic (normalised to rest at frame 0) the fit's own delays use
    ref = run(base, Cell([0, 0, 0], [0.0, 2.0, 0.0]), lambda f: [0, 0, 0], frames)
    assert torch.allclose(g[:, 0][2:], ref[2:, 0], atol=1e-14)
    assert torch.allclose(g[:, 1][4:], ref[4:, 1], atol=1e-14)
    assert torch.all(g[:4, 1] == 0)                                     # nothing before its excitation
    assert torch.all(g[:, 2] == 0)


def test_paced_beats_retrigger_and_a_silent_cell_stays_silent():
    """Two beats (segments 0 and 20): cells excited at 2 and again at 22 contract in both beats, each
    beat the default's; a cell excited only in beat 1 does not contract in beat 2."""
    base = get_operator("active_strain")({"_at": "mpm_particle", "parent": "cell", "clock": CLOCK, "segments": [0, 20]})
    exc = get_operator("active_strain", model="excitation")(
        {"_at": "mpm_particle", "parent": "cell", "clock": CLOCK, "t_ref": 2.0, "segments": [0, 20]})
    pulse = lambda f, on: 1.0 if (on <= f < on + 4) else 0.0
    ue = lambda f: [max(pulse(f, 2), pulse(f, 22)), pulse(f, 2), 0.0]
    frames = range(0, 40)
    g = run(exc, Cell([0, 0, 0], [0.0, 0.0, 0.0]), ue, frames)
    ref = run(base, Cell([0, 0, 0], [0.0, 0.0, 0.0]), lambda f: [0, 0, 0], frames)[:, 0]
    assert torch.allclose(g[2:20, 0], ref[2:20], atol=1e-14) and torch.allclose(g[22:, 0], ref[22:], atol=1e-14)
    assert torch.allclose(g[2:20, 1], ref[2:20], atol=1e-14) and torch.all(g[20:, 1] == 0)
    assert torch.all(g[:, 2] == 0)

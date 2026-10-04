"""The engine's unrecorded WARM-UP (2026-09-28): `general.warmup` (default = the seed's settle window `ref_frame`).

    PYTHONPATH=src:tools python -m pytest tests/test_engine_warmup.py -q

The seed's settle window runs BEFORE frame 0 and is not recorded; the size rules idle and `cell_grow` holds through
it, so frame 0 of the data is the relaxed tissue and growth, the cycle and the record start together. `warmup: 0`
restores the old behaviour (the window is recorded, growth runs through it). exp 13's spheroid, shrunk, on CPU.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from plexus import schema
from plexus.engine import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES, WINDOW = 12, 6


def _small(warmup):
    sim = schema.load(os.path.join(ROOT, "config", "tissue", "exp13_fb_a2_s1.yaml"))
    sim.n_frames = FRAMES
    sim.record_cap = FRAMES + 1
    sim.warmup = warmup
    for o in sim.seed_ops:
        if getattr(o, "op", None) == "seed_mesh":
            o.params.update(n_cells=60, radius=3.0, ref_frame=WINDOW)
    for o in sim.operators:
        if getattr(o, "op", None) == "cell_grow":
            o.params.update(rate=0.01)                    # growth visible within a few frames
    return sim


def _v0f(traj):
    c = traj["sets"]["cell"]
    occ = np.asarray(c["occ"], bool)
    return [np.asarray(c["state"]["V0f"])[t][occ[t]][..., 0] if "V0f" in c["state"] else None for t in range(len(occ))]


@pytest.fixture(scope="module")
def runs():
    return {w: run(_small(w), out_path=None, device="cpu") for w in (None, 0)}


def test_warmup_keeps_the_recorded_length(runs):
    for w, (H, traj) in runs.items():
        occ = np.asarray(traj["sets"]["cell"]["occ"])
        assert len(occ) == FRAMES + 1, (w, len(occ))
        assert len(traj["frame_ms"]) == FRAMES + 1, (w, len(traj["frame_ms"]))


def test_default_warmup_is_the_settle_window_and_off_is_zero(runs):
    assert int(runs[None][0].warmup) == WINDOW
    assert int(runs[0][0].warmup) == 0


def test_growth_holds_through_the_warmup(runs):
    """PLANTED: with the warm-up, frame 0's targets are the seed's plus ONE step (the median within 2 % of the
    no-warm-up run's frame 0, which is also seed + one step); without it the targets grow through the settle window
    (the median up > 3 % by the frame the window ends)."""
    on, off = _v0f(runs[None][1]), _v0f(runs[0][1])
    assert on[0] is not None and off[0] is not None, "V0f is not recorded on the cell set"
    assert abs(np.median(on[0]) / np.median(off[0]) - 1.0) < 0.02
    assert np.median(off[WINDOW]) / np.median(off[0]) > 1.03

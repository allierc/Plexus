"""tools/exp_land.py's health check on planted trajectories (2026-09-27)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "tools"), os.path.join(ROOT, "src")]

import exp_land  # noqa: E402


def _run(tmp_path, monkeypatch, frames=2000, rows=183, nan_set=None):
    d = tmp_path / "tissue" / "expXX_a_s1"
    d.mkdir(parents=True)
    pos = np.zeros((rows, 10, 3), np.float32)
    pos[:, :, 0] = np.linspace(0, 1, 10)[None] * np.linspace(1, 2, rows)[:, None]
    mem = np.zeros((rows, 5, 3), np.float32)
    if nan_set == "membrane":
        mem[40:] = np.nan
    np.savez(d / "trajectory.npz", frame_ms=np.full(frames, 3.0), cell__pos=pos,
             cell__occ=np.ones((rows, 10, 1)), membrane__pos=mem)
    monkeypatch.setattr(exp_land, "_gd", lambda: str(tmp_path))
    return exp_land.health("tissue/expXX_a_s1", audit=False)


def test_recorded_rows_are_not_simulated_frames(tmp_path, monkeypatch):
    h = _run(tmp_path, monkeypatch)                  # exp08: 2,000 frames recorded as 183 rows
    assert h["frames"] == 2000 and h["rows"] == 183
    assert h["sets"]["cell"]["count"] == [10, 10]
    assert h["sets"]["cell"]["extent"][1] > h["sets"]["cell"]["extent"][0]


def test_non_finite_in_a_set_that_is_not_the_subject(tmp_path, monkeypatch):
    h = _run(tmp_path, monkeypatch, nan_set="membrane")
    assert any(b.startswith("membrane__pos") and "from frame 40" in b for b in h["non_finite"])
    assert any("NON-FINITE" in l for l in exp_land.health_lines(h))

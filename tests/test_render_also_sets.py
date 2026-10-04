"""`plotting.also_sets: [<set>, ...]` -- the movie draws more point sets over its subject (exp 15).

The point renderer draws ONE subject set, so exp 15's host cells under the community lattice were
absent from the engine movie and the talk movie needed its own script. A two-set synthetic run: a
`cell` subject and a smaller `host` set whose one-column `chem` pulses on one host per frame.

    PYTHONPATH=src python -m pytest tests/test_render_also_sets.py -q
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("pyvista")

T, NH = 4, 2


def _traj(d):
    g = np.stack(np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 6)), -1).reshape(-1, 2)
    hpos = np.array([[0.3, 0.5], [0.7, 0.5]], np.float32)
    hchem = np.zeros((T, NH, 1), np.float32)
    for t in range(T):
        hchem[t, t % NH, 0] = 1.0
    np.savez(os.path.join(d, "trajectory.npz"), world=np.float64(1.0),
             world_size=np.array([1.0, 1.0], np.float32),
             cell__pos=np.tile(g.astype(np.float32), (T, 1, 1)), cell__occ=np.ones((T, len(g)), bool),
             cell__chem=np.tile(np.array([1.0, 0.0, 0.0], np.float32), (T, len(g), 1)),
             host__pos=np.tile(hpos, (T, 1, 1)), host__occ=np.ones((T, NH), bool), host__chem=hchem)
    return hpos


def _replay(tmp_path, style):
    from plexus import live_movie as LM
    d = str(tmp_path)
    hpos = _traj(d)
    rec = {"cloud": [], "also": [], "n_also": None}

    class Rec(LM.LiveMovie):
        def __call__(self, H, tick):
            super().__call__(H, tick)
            assert not self.failed, self.failed
            rec["cloud"].append(np.asarray(self.cloud["rgb"]).copy())
            rec["n_also"] = len(getattr(self, "_also", []) or [])
            if rec["n_also"]:
                nm, pd = self._also[0]
                rec["also"].append((nm, np.asarray(pd.points).copy(), np.asarray(pd["rgb"]).copy()))

    sim = types.SimpleNamespace(plotting=dict(style, render_px=160), sets={}, units=None, dt=1.0,
                                n_frames=T - 1, name="two", world=1.0)
    orig = LM.LiveMovie
    LM.LiveMovie = Rec
    try:
        LM.replay(d, sim, out=os.path.join(d, "movie.mp4"), stills=0)
    finally:
        LM.LiveMovie = orig
    return rec, hpos


BASE = {"renderer": "mpl2d", "background": "black", "blend": "additive",
        "species": ["#e8262a", "#2f5fd0", "#f2d21b"], "subject": "cell"}


def test_without_also_sets_only_the_subject_is_drawn(tmp_path):
    rec, _ = _replay(tmp_path, BASE)
    assert rec["n_also"] == 0 and not rec["also"]
    assert len(rec["cloud"]) == T


def test_also_sets_draws_the_host_over_the_subject_per_frame(tmp_path):
    rec, hpos = _replay(tmp_path, dict(BASE, also_sets=["host"]))
    base, _ = _replay(tmp_path, BASE)
    assert all(np.array_equal(a, b) for a, b in zip(rec["cloud"], base["cloud"]))   # subject untouched
    assert rec["n_also"] == 1 and len(rec["also"]) == T
    for t, (nm, pts, rgb) in enumerate(rec["also"]):
        assert nm == "host"
        assert np.allclose(pts[:, :2], hpos) and (pts[:, 2] > 0).all()   # lifted above the 2D subject
        lit = rgb.sum(1) > 0                                              # additive: quiet = black
        assert np.array_equal(lit, np.arange(NH) == t % NH), t            # the pulse moves


def test_also_colors_fixes_a_set_colour(tmp_path):
    rec, _ = _replay(tmp_path, dict(BASE, also_sets=["host"], also_colors={"host": "#ffffff"}))
    assert all((rgb == 255).all() for _, _, rgb in rec["also"])

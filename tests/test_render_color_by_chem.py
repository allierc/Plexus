"""`plotting.color_by: chem` -- a segmentation-seeded sheet's movie shows its chemistry (exp 6).

`seed_from_segmentation` puts material points (`mpm_particle`, parent `cell`) in each segmented cell;
the chemistry lives on `cell`. The points are the larger set, so they are the movie's subject, and with
no `chem` of their own they were painted by parent BODY (one tab10 hue per cell id, fixed at t = 0): a
static confetti square while the wave crossed the tissue. The mpl2d `-o plot` pass drew `cell` as a
container (merged blobs, one hue per id) and threw the chem colour it had just computed away.

A three-cell synthetic sheet, four particles per cell, a pulse that moves one cell per frame.

    PYTHONPATH=src python -m pytest tests/test_render_color_by_chem.py -q
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("pyvista")

T, NC, PER = 4, 3, 4


def _sheet(d):
    cpos = np.tile(np.array([[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]], np.float32), (T, 1, 1))
    chem = np.zeros((T, NC, 2), np.float32)
    for t in range(T):
        chem[t, t % NC, 0] = 1.0                       # the pulse: cell t % 3 is excited at frame t
    par = np.repeat(np.arange(NC), PER)
    jit = np.random.default_rng(0).normal(0, 0.01, (len(par), 2))
    ppos = np.tile((cpos[0][par] + jit).astype(np.float32), (T, 1, 1))
    np.savez(os.path.join(d, "trajectory.npz"), world=np.float64(1.0),
             world_size=np.array([1.0, 1.0], np.float32),
             cell__pos=cpos, cell__occ=np.ones((T, NC), bool), cell__chem=chem,
             mpm_particle__pos=ppos, mpm_particle__occ=np.ones((T, len(par)), bool),
             mpm_particle__parent=par, mpm_particle__parent_name=np.array("cell"))
    return chem, par


def _replay_rgb(tmp_path, style):
    """The subject cloud's colour at every frame of a real `live_movie.replay`."""
    from plexus import live_movie as LM
    d = str(tmp_path)
    _sheet(d)
    frames = []

    class Rec(LM.LiveMovie):
        def __call__(self, H, tick):
            super().__call__(H, tick)
            assert not self.failed, self.failed
            frames.append(np.asarray(self.cloud["rgb"]).copy())

    sim = types.SimpleNamespace(plotting=dict(style, render_px=160), sets={}, units=None, dt=1.0,
                                n_frames=T - 1, name="sheet", world=1.0)
    orig = LM.LiveMovie
    LM.LiveMovie = Rec
    try:
        LM.replay(d, sim, out=os.path.join(d, "movie.mp4"), stills=0)
    finally:
        LM.LiveMovie = orig
    return frames


def test_live_movie_without_color_by_keeps_the_parent_body_hues(tmp_path):
    fr = _replay_rgb(tmp_path, {"renderer": "mpl2d", "background": "black"})
    assert len(fr) == T
    assert all(np.array_equal(f, fr[0]) for f in fr)           # static: fixed at t = 0, as before
    import matplotlib.pyplot as plt
    tab = (np.array([plt.get_cmap("tab10")(k)[:3] for k in range(NC)]) * 255).astype(np.uint8)
    assert np.array_equal(fr[0], tab[np.repeat(np.arange(NC), PER)])


def test_live_movie_color_by_chem_paints_each_particle_its_cells_chem(tmp_path):
    fr = _replay_rgb(tmp_path, {"renderer": "mpl2d", "background": "black", "color_by": "chem"})
    assert len(fr) == T
    for t, f in enumerate(fr):
        red = f[:, 1] < 200                                    # subtractive: excited = red, rest = white
        want = np.repeat(np.arange(NC), PER) == (t % NC)
        assert np.array_equal(red, want), t


def _plot_calls(tmp_path, style, monkeypatch):
    """What the mpl2d path hands to its two movie writers, per set."""
    from plexus import plot as P
    d = str(tmp_path)
    chem, par = _sheet(d)
    calls = {}
    monkeypatch.setattr(P, "graphs_data_path", lambda folder, name: d)
    monkeypatch.setattr(P, "_movie", lambda pos, occ, color, *a, **k:
                        calls.__setitem__(os.path.basename(a[3]), ("dots", color)))
    monkeypatch.setattr(P, "_merged_movie", lambda cpos, par_, ncell, W, cmap, T_, out, **k:
                        calls.__setitem__(os.path.basename(out), ("merged", None)))
    sim = types.SimpleNamespace(plotting=dict(style), sets={}, fields={}, name="sheet", world=1.0,
                                world_size=[1.0, 1.0], obstacles=[], boundary="wall")
    P.plot_dataset(sim, "tissue", movie=True)
    return calls, chem, par


def test_mpl2d_without_color_by_is_unchanged(tmp_path, monkeypatch):
    calls, _, _ = _plot_calls(tmp_path, {"renderer": "mpl2d", "background": "black"}, monkeypatch)
    assert calls["movie_cell"][0] == "merged"                  # the container view, as before
    kind, col = calls["movie_mpm_particle"]
    assert kind == "dots" and np.ndim(col) == 2                # one static hue per parent id


def test_mpl2d_color_by_chem_draws_the_chemistry(tmp_path, monkeypatch):
    calls, chem, par = _plot_calls(tmp_path, {"renderer": "mpl2d", "background": "black",
                                              "color_by": "chem"}, monkeypatch)
    from plexus.live import chem_rgb
    kind, col = calls["movie_cell"]
    assert kind == "dots" and np.shape(col) == (T, NC, 3)      # dots in the chem colour, per frame
    want = np.stack([chem_rgb(chem[t], background="black")[0] for t in range(T)])
    assert np.allclose(col, want)
    kind, col = calls["movie_mpm_particle"]
    assert kind == "dots" and np.shape(col) == (T, len(par), 3)
    assert np.allclose(col, want[:, par])                      # each particle its cell's colour

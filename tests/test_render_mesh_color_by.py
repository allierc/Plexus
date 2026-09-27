"""`plotting.mesh_color_by: node_type` and `mesh_color_scale` -- the mesh coloured per cell, per frame.

exp 9 drew its LEP/MEP sheet in one colour (`mesh_color: '#ffffff'`): `mesh_color_by` read state
blocks only, and a cell's type is `node_type`, a column of the cell set that is not a block -- so it
was never found, and a cell retyped during the run (Part B) could not be recoloured either. And a
float block was always rounded into categories. Two square cells; cell 1 is retyped 0 -> 1 at frame 2.

Both paths are covered: `render_vtk` (renderer `vtk_mesh`) and `live_movie` (a mesh subject under the
point renderer, which is what exp 9 Phase 2 runs), the latter through a real `live_movie.replay`.

    PYTHONPATH=src python -m pytest tests/test_render_mesh_color_by.py -q
"""
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pytest.importorskip("pyvista")

T = 4
POS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], np.float32)
ES = np.array([0, 1, 4, 3, 1, 2, 5, 4])
ET = np.array([1, 4, 3, 0, 2, 5, 4, 1])
EF = np.array([0, 0, 0, 0, 1, 1, 1, 1])
NT_T = np.array([[0, 0], [0, 0], [0, 1], [0, 1]], np.int16)       # cell 1 retyped at frame 2
GF = np.array([0.2, 0.7], np.float32)                               # a float block, not a label
PAL = ["#0000ff", "#ff8000"]


def _traj(d):
    np.savez(os.path.join(d, "trajectory.npz"), world=np.float64(3.0),
             world_size=np.array([3.0, 3.0, 3.0], np.float32),
             vertex__pos=np.tile(POS, (T, 1, 1)), vertex__occ=np.ones((T, 6), bool),
             vertex__mesh_E_srce=np.tile(ES, T), vertex__mesh_E_trgt=np.tile(ET, T),
             vertex__mesh_E_face=np.tile(EF, T), vertex__mesh_nF=np.full(T, 2),
             vertex__mesh_Nv=np.full(T, 6), vertex__mesh_offsets=np.arange(T + 1) * 8,
             vertex__mesh_face_offsets=np.arange(T + 1) * 2,
             cell__age=np.full((T, 2, 1), 50.0, np.float32), cell__gf=np.tile(GF[None, :, None], (T, 1, 1)),
             cell__node_type=NT_T[-1].astype(np.int64), cell__node_type_t=NT_T,
             cell__type_names=np.array(["LEP", "MEP"]))
    return os.path.join(d, "trajectory.npz")


def _pal(k):
    from matplotlib.colors import to_rgb
    return (np.asarray(to_rgb(PAL[k])) * 255).astype(np.uint8)


# ---- render_vtk: the `vtk_mesh` renderer -----------------------------------------------------------

def _vtk_rgb(tmp_path, style, frame):
    from plexus import render_vtk as R
    R.use_plotting(style)
    try:
        fr = R._core_frames(_traj(str(tmp_path)))
        pos, mt, act, chem = fr[frame]
        m = R.mesh_of(pos, mt, act, show_div=False, chem=chem, color_rng=R.mesh_color_rng(fr))
        return np.asarray(m.cell_data["rgb"]).astype(int)
    finally:
        R.use_plotting(None)


def test_vtk_mesh_without_mesh_color_by_is_one_colour(tmp_path):
    rgb = _vtk_rgb(tmp_path, {"mesh_color": "#ffffff"}, 3)
    assert np.array_equal(rgb[0], rgb[1])                               # one body colour, as before


def test_vtk_mesh_float_block_without_scale_keeps_the_categorical_law(tmp_path):
    from plexus.measures import label_rgb
    rgb = _vtk_rgb(tmp_path, {"mesh_color_by": "gf"}, 0)
    want = (np.array([label_rgb(0), label_rgb(1)]) * 255).astype(np.uint8)   # rint(0.2), rint(0.7)
    assert np.array_equal(rgb, want)


def test_vtk_mesh_node_type_per_frame(tmp_path):
    st = {"mesh_color_by": "node_type", "label_colors": PAL}
    assert np.array_equal(_vtk_rgb(tmp_path, st, 0), np.stack([_pal(0), _pal(0)]))
    assert np.array_equal(_vtk_rgb(tmp_path, st, 3), np.stack([_pal(0), _pal(1)]))   # retyped


def test_vtk_mesh_continuous_scale(tmp_path):
    from matplotlib import colormaps
    rgb = _vtk_rgb(tmp_path, {"mesh_color_by": "gf", "mesh_color_scale": "continuous",
                              "mesh_color_range": [0.0, 1.0]}, 0)
    want = (np.asarray(colormaps["viridis"](GF.astype(float)))[:, :3] * 255).astype(np.uint8)
    assert np.array_equal(rgb, want)
    # `auto` on an integer-valued block stays categorical
    rgb = _vtk_rgb(tmp_path, {"mesh_color_by": "node_type", "mesh_color_scale": "auto",
                              "label_colors": PAL}, 3)
    assert np.array_equal(rgb, np.stack([_pal(0), _pal(1)]))


# ---- live_movie: the mesh as the point renderer's subject (exp 9 Phase 2) ---------------------------

def _replay_faces(tmp_path, style):
    from plexus import live_movie as LM
    d = str(tmp_path)
    _traj(d)
    frames = []

    class Rec(LM.LiveMovie):
        def __call__(self, H, tick):
            super().__call__(H, tick)
            assert not self.failed, self.failed
            frames.append(np.asarray(self._meshes[0][2].cell_data["rgb"]).copy())

    sets = {"vertex": {"mesh": "half_edge"}, "half_edge": {"maps": {"face": "cell"}}}
    sim = types.SimpleNamespace(plotting=dict(style, render_px=160, mesh_mark_division=False),
                                sets=sets, units=None, dt=1.0, n_frames=T - 1, name="mesh", world=3.0)
    orig = LM.LiveMovie
    LM.LiveMovie = Rec
    try:
        LM.replay(d, sim, out=os.path.join(d, "movie.mp4"), stills=0)
    finally:
        LM.LiveMovie = orig
    return frames


def test_live_mesh_without_mesh_color_by_is_one_colour(tmp_path):
    fr = _replay_faces(tmp_path, {"mesh_color": "#ffffff"})
    assert len(fr) == T and all((f == 255).all() for f in fr)


def test_live_mesh_node_type_recoloured_when_retyped(tmp_path):
    fr = _replay_faces(tmp_path, {"mesh_color_by": "node_type", "label_colors": PAL})
    assert len(fr) == T
    for t in range(T):
        assert np.array_equal(fr[t], np.stack([_pal(int(NT_T[t, 0])), _pal(int(NT_T[t, 1]))])), t


def test_live_mesh_continuous_scale(tmp_path):
    from matplotlib import colormaps
    fr = _replay_faces(tmp_path, {"mesh_color_by": "gf", "mesh_color_scale": "continuous",
                                  "mesh_color_range": [0.0, 1.0]})
    want = (np.asarray(colormaps["viridis"](GF.astype(float)))[:, :3] * 255).astype(np.uint8)
    assert all(np.array_equal(f, want) for f in fr)

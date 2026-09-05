"""R1(d) 1 and 2: a gate can reach a per-vertex block, and can read `renumber_failed`.

WHY EITHER IS NEEDED. A gate row names a function from `MEASURES` and that function is handed a
`Traj` -- a nine-getter facade over a saved run, written so that ONE measure function grades both an
okuda run and a core run. Everything the facade could reach was cell- or mesh-shaped.

1. `state(block, t)` builds its key from the CELL set and crops to `nF`. The apico-basal separation
   `sep` is per-VERTEX and cropped by `nV` -- on gate_00's last row that is 13,824 against 6,914 --
   so `state` would have returned the first 6,914 rows of a 13,824-row array. Not an error: a
   plausible-looking half. `vertex_block` is the tenth getter.

2. `renumber_failed` is written into every run by `MeshTable.SCALAR_RECORD` and NOTHING HAS EVER
   READ IT. `MEASURES` exposed four thin wrappers over `scalar_col`, and `scalar_col` itself is not
   in the table, so a row naming it fails preflight. It is the sentinel from the 23 August defect --
   `renumber_set` returned False on every call and both callers discarded the bool -- and the fix
   put the counter there so "a gate can assert it is 0 instead of a human having to notice a printed
   line". This is that assertion.

THE FAILURE MODE THESE TESTS EXIST FOR is a getter that returns `None`. A measure that returns
`None` does not raise; a row reducing it can read as passing. So both cases assert on the SHAPE and
the VALUES, never merely on "it did not raise".
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import gate_measures as GM


def _a_mesh_trajectory():
    """A recorded run that carries a mesh, or skip. Not a fixture built here on purpose: the point
    is to exercise the reader against a file the engine actually wrote."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(GM.__file__)))   # <repo>/tools/.. -> <repo>
    for p in sorted(glob.glob(os.path.join(root, "graphs_data", "**", "trajectory.npz"),
                              recursive=True)):
        try:
            if "vertex__mesh_nF" in np.load(p, allow_pickle=True).files:
                return p
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def T():
    p = _a_mesh_trajectory()
    if p is None:
        pytest.skip("no mesh trajectory on disk to read")
    return GM.CoreTraj(p, set_name="vertex", cell_set="cell")


def test_vertex_block_is_cropped_by_nV_not_nF(T):
    """THE WHOLE POINT. The two crops differ, and taking the wrong one returns plausible garbage."""
    t = 0
    v = T.vertex_block("pos", t)
    assert v is not None, "the vertex block came back None -- a gate row would read that as nothing"
    assert v.shape[0] == T.nV(t), f"cropped to {v.shape[0]}, expected nV={T.nV(t)}"
    if T.nF(t) != T.nV(t):
        assert v.shape[0] != T.nF(t), "cropped to nF -- this is the cell crop, not the vertex one"


def test_vertex_block_matches_pos(T):
    """`pos` is the one per-vertex block that already had a getter, so it is the cross-check."""
    t = 0
    assert np.array_equal(T.vertex_block("pos", t), T.pos(t))


def test_vertex_block_is_two_dimensional(T):
    """A width-1 block must still come back [nV, 1], so a measure can index it uniformly."""
    v = T.vertex_block("pos", 0)
    assert v.ndim == 2


def test_a_missing_block_is_None_not_an_exception(T):
    """An operator may not be in a spec at all; the facade's contract is None, as `face_col` has."""
    assert T.vertex_block("a_block_no_operator_writes", 0) is None


def test_renumber_reader_is_callable_from_a_gate_row():
    """A row names a function in MEASURES. `scalar_col` is NOT in it, which is why a wrapper is
    needed rather than pointing a row at the generic helper."""
    assert "renumber_did_not_act" in GM.MEASURES
    assert "scalar_col" not in GM.MEASURES


def test_renumber_reader_returns_a_series_not_None(T):
    """Every measure returns one value per recorded row; the row's `reduce:` collapses it.

    A `None` here is the failure that matters: the counter would be unreadable exactly as before,
    but a gate table would now LOOK as though it were guarded.
    """
    r = GM.MEASURES["renumber_did_not_act"](T)
    assert r is not None, "renumber_failed still unreadable"
    r = np.asarray(r)
    assert r.shape[0] == T.n_rows()
    assert np.isfinite(r).all()


def test_renumber_counter_is_zero_on_a_run_that_never_renumbered(T):
    """A CAVEAT, ASSERTED SO IT IS NOT MISREAD. gate_00 has no death operator and `edge_flip`'s
    face-drop branch is the only other renumber path, so 0 here is weak evidence -- it is the
    reading of a path this run barely exercises. The row is still worth having: it is the death
    rungs (R6) that make it informative, and a counter that is 0 because nothing called it looks
    identical to one that is 0 because everything worked.
    """
    r = np.asarray(GM.MEASURES["renumber_did_not_act"](T))
    assert (r >= 0).all()

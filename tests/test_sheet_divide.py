"""`cell_divide[adder_sheet]`'s edge choice on hand-built flat meshes whose right answer is known.

    PYTHONPATH=src python -m pytest tests/test_sheet_divide.py -q
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tests")]

import plexus.models.topology as TOP  # noqa: E402
from plexus.operators.vertex_ops import sheet_septum, sheet_split_edges  # noqa: E402
from test_cell_chem_steady_uptake import square_disc  # noqa: E402


def rect_mesh(start_short):
    """One 2 x 1 rectangle cell (long along x) in z = 0, plus a neighbour across every edge, so every
    edge has an interior twin. The ring starts on a short edge when `start_short`."""
    pos = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (1, -1, 0), (3, 0.5, 0), (1, 2, 0), (-1, 0.5, 0)]
    ring = [1, 2, 3, 0] if start_short else [0, 1, 2, 3]
    rings = [ring, [0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]
    return rings, [np.array(p, float) for p in pos]


def long_edges(ring):
    k = len(ring)
    return {i for i in range(k) if {ring[i], ring[(i + 1) % k]} in ({0, 1}, {2, 3})}


def test_the_septum_crosses_the_long_axis_whatever_the_ring_start():
    for start_short in (False, True):
        rings, pos = rect_mesh(start_short)
        pair = sheet_split_edges(rings[0], pos, TOP._edge_face_map(rings), 0, axis=2)
        assert set(pair) == long_edges(rings[0])          # Hertwig: through the two LONG edges


def test_the_old_fallback_depends_on_the_ring_start():
    """What the shell code does on a plane: every edge projects to 0, ea = eb = 0, and the fallback
    (0, k/2) is the short pair when the ring starts on a short edge -- the defect this module fixes."""
    rings, pos = rect_mesh(True)
    assert {0, 2} != long_edges(rings[0])


def test_a_rim_cell_keeps_hertwig_across_its_free_edge():
    """Remove the neighbour across a LONG edge. Allowed to cross the free edge, the septum stays
    Hertwig's (the two long edges); restricted to interior edges, it falls back to the short ones."""
    rings, pos = rect_mesh(False)
    rings = rings[:3] + [None, rings[4]]                  # no face across edge 2 (the top long edge)
    em = TOP._edge_face_map(rings)
    assert set(sheet_split_edges(rings[0], pos, em, 0)) == {0, 2}
    assert set(sheet_split_edges(rings[0], pos, em, 0, allow_open=False)) == {1, 3}


def test_rim_faces_of_a_disc_divide_where_the_shell_rule_refuses():
    pos, es, et, ef, nF, cen = square_disc(8.0)
    rings = [[] for _ in range(nF)]
    for s_, f in zip(es, ef):
        rings[f].append(int(s_))
    P = [np.asarray(p, float) for p in pos]
    emap = TOP._edge_face_map(rings)
    rim = [f for f in range(nF) if any(emap.get((rings[f][(i + 1) % 4], rings[f][i])) is None for i in range(4))]
    one_open = [f for f in rim if sum(emap.get((rings[f][(i + 1) % 4], rings[f][i])) is None for i in range(4)) == 1]
    old = sum(TOP.divide_face_3d([list(r) for r in rings], list(P), f, project=False, ea=0, eb=2) is not None
              for f in one_open)
    new = sum(sheet_split_edges(rings[f], P, emap, f) is not None for f in one_open)
    assert new == len(one_open) and old < len(one_open)   # every one-open-edge rim cell can divide now


def _euler(rings):
    V = {v for r in rings if r for v in r}
    E = {tuple(sorted((r[i], r[(i + 1) % len(r)]))) for r in rings if r for i in range(len(r))}
    return len(V) - len(E) + sum(1 for r in rings if r)


def test_dividing_across_the_free_edge_grows_the_rim_and_keeps_the_disc_a_disc():
    from plexus.operators.vertex_ops import divide_face_open
    pos, es, et, ef, nF, cen = square_disc(6.0)
    rings = [[] for _ in range(nF)]
    for s_, f in zip(es, ef):
        rings[f].append(int(s_))
    P = [np.asarray(p, float) for p in pos]
    em = TOP._edge_face_map(rings)
    rim = lambda: sum(any(em.get((r[(i + 1) % len(r)], r[i])) is None for i in range(len(r))) for r in rings)
    f = next(f for f in range(nF) if sum(em.get((rings[f][(i + 1) % 4], rings[f][i])) is None for i in range(4)) == 1)
    i_open = next(i for i in range(4) if em.get((rings[f][(i + 1) % 4], rings[f][i])) is None)
    n_rim, chi = rim(), _euler(rings)
    births = []
    res = divide_face_open(rings, P, f, i_open, (i_open + 2) % 4, em, births)
    assert res is not None and len(births) == 2
    assert rim() == n_rim + 1                              # the perimeter gained a cell
    assert _euler(rings) == chi == 1                       # still one open disc
    assert em == TOP._edge_face_map(rings)                 # the maintained map equals a rebuild


def test_the_patch_is_restored_even_on_error():
    orig = TOP.divide_face_3d
    with pytest.raises(RuntimeError):
        with sheet_septum(2):
            assert TOP.divide_face_3d is not orig
            raise RuntimeError("boom")
    assert TOP.divide_face_3d is orig


def test_registered_as_an_implementation():
    import plexus.operators  # noqa: F401
    from plexus.models.registry import get_operator
    cls = get_operator("cell_divide", variant="adder_sheet")
    assert cls.__name__ == "Divide3DAdderSheet"


def test_identity_divide_face_open_equals_the_default_on_interior_edges():
    """Where both split edges have a neighbour, the open-mesh division is the default's, exactly: the
    same rings, the same new vertices, the same maintained edge map."""
    from plexus.operators.vertex_ops import divide_face_open
    pos, es, et, ef, nF, cen = square_disc(6.0)
    rings = [[] for _ in range(nF)]
    for s_, f in zip(es, ef):
        rings[f].append(int(s_))
    P = [np.asarray(p, float) for p in pos]
    em = TOP._edge_face_map(rings)
    f = next(f for f in range(nF) if all(em.get((rings[f][(i + 1) % 4], rings[f][i])) is not None for i in range(4)))
    ra, pa, ma = [list(r) for r in rings], list(P), dict(em)
    rb, pb, mb = [list(r) for r in rings], list(P), dict(em)
    out_a = TOP.divide_face_3d(ra, pa, f, project=False, ea=0, eb=2, emap=ma)
    out_b = divide_face_open(rb, pb, f, 0, 2, mb)
    assert out_a == out_b and ra == rb and ma == mb
    assert all(np.array_equal(x, y) for x, y in zip(pa, pb))

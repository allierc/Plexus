"""A vertex born mid-run gets a value -- R1(b) of the apico-basal promotion.

WHAT WAS MISSING. `MeshTable.reindex_faces` carries every per-FACE array across a topology edit, and
its docstring records why it had to become one operation: the carry was written four times and
`edge_flip`'s copy silently omitted the medioapical myosin. There was no per-VERTEX analogue at all,
because until now `pos` was the only per-vertex quantity and both topology functions write it
themselves -- `divide_face_3d` sets the midpoint, `face_collapse_3d` sets the centroid.

It stops being invisible the moment a second per-vertex block exists. A monolayer's apico-basal
separation would be 0 at a vertex born on a septum -- a cell of zero height along the seam it just
grew -- and at an extrusion the survivor would keep `r[0]`'s value while its position jumped to the
centroid of three.

AND THE PARENTAGE WAS ALREADY COMPUTED AND THROWN AWAY. `divide_face_3d` builds each midpoint from a
named edge's two endpoints; `face_collapse_3d` computes `keep, drop` and moves `pos[keep]` to the
centroid. Both now append `(new_vertex, (parents...))` to an optional `births` list, so nothing is
recomputed and nothing is guessed.

THE FIRST TEST IS THE ONE R1 HAS TO PASS. With no `vertex_carry` declared the carry walks an empty
name list and returns, so every one of the 2,456 specs in the corpus is untouched -- that is the
byte-identity claim the twin suite measures, in miniature.
"""
from __future__ import annotations

import numpy as np
import torch

from plexus.models.mesh import MeshTable, declare_vertex_carry
from plexus.models.topology import divide_face_3d, face_collapse_3d, rings_from_flat_3d


# --------------------------------------------------------------------------------------------- #
#  parentage: reported, not recomputed
# --------------------------------------------------------------------------------------------- #
def _cube_rings():
    """A closed cube as rings: 8 vertices, 6 quad faces. The smallest closed surface `divide_face_3d`
    will act on -- it refuses a ring of fewer than four edges.

    RINGS ARE LISTS OF PLAIN ints, which is what `rings_from_flat_3d` returns and therefore what
    `cell_divide` passes. Built as numpy arrays first, this fixture died inside `_insert_after` on
    `ring.insert` -- a fixture that cannot reproduce production is not a test, and this one could
    not have caught anything about the caller it stands in for.
    """
    pos = [np.array(p, float) for p in
           [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]]
    rings = [list(r) for r in
             [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]]
    return rings, pos


def test_divide_reports_the_two_midpoints_and_their_parents():
    rings, pos = _cube_rings()
    births = []
    res = divide_face_3d(rings, pos, 0, project=False, births=births)
    assert res is not None, "the cube's quad face should divide"
    _daughter, m1, m2 = res
    assert [b[0] for b in births] == [m1, m2], "births must name the two new vertices, in order"
    for new_i, parents in births:
        assert len(parents) == 2, "a midpoint has exactly two parents -- the split edge's endpoints"
        # and it really is their midpoint, which is what makes a MEAN the right blend
        assert np.allclose(pos[new_i], 0.5 * (pos[parents[0]] + pos[parents[1]]))


def test_divide_without_births_is_unchanged():
    """The out-parameter is opt-in: every existing caller passes nothing and sees nothing."""
    a_rings, a_pos = _cube_rings()
    b_rings, b_pos = _cube_rings()
    ra = divide_face_3d(a_rings, a_pos, 0, project=False)
    rb = divide_face_3d(b_rings, b_pos, 0, project=False, births=[])
    assert ra == rb
    assert all(np.array_equal(x, y) for x, y in zip(a_pos, b_pos))
    assert a_rings == b_rings


# --------------------------------------------------------------------------------------------- #
#  the carry itself
# --------------------------------------------------------------------------------------------- #
def _table(n=8, value=None):
    m = MeshTable()
    m["h"] = torch.arange(float(n)) if value is None else torch.full((n,), float(value))
    return m


def test_carry_is_a_no_op_when_nothing_is_declared():
    """THE BYTE-IDENTITY CLAIM. No `vertex_carry` -> the array is not touched at all."""
    m = _table()
    before = m["h"].clone()
    m.carry_vertices([(0, (1, 2)), (3, (4, 5))], dt=torch.float32, dev="cpu")
    assert torch.equal(m["h"], before), "an undeclared array was rewritten"


def test_declared_array_is_blended_from_its_parents():
    m = _table()
    declare_vertex_carry(m, "h")
    m.carry_vertices([(0, (2, 4))], dt=torch.float32, dev="cpu")
    assert float(m["h"][0]) == 3.0, "h[0] should be the mean of h[2]=2 and h[4]=4"
    assert float(m["h"][1]) == 1.0, "an untouched vertex must not move"


def test_a_merge_blends_all_three_parents():
    """`face_collapse_3d` reports (survivor, (survivor, dropped, dropped)) -- three parents, not two.

    Without the survivor in its own parent list the extrusion site would jump to the mean of the two
    cells it absorbed, which is a different point from the centroid `pos[keep]` is moved to.
    """
    m = _table()
    declare_vertex_carry(m, "h")
    m.carry_vertices([(1, (1, 3, 5))], dt=torch.float32, dev="cpu")
    assert float(m["h"][1]) == 3.0, "(1 + 3 + 5) / 3"


def test_births_are_applied_in_order():
    """A vertex that is a parent of a later birth contributes its NEW value.

    This is the case a set-based or vectorised implementation would get wrong, and it is the real
    one: a collapse following a division in the same tick sees the divided sheet, not the old one.
    """
    m = _table()
    declare_vertex_carry(m, "h")
    m.carry_vertices([(0, (2, 4)), (1, (0, 0))], dt=torch.float32, dev="cpu")
    assert float(m["h"][0]) == 3.0
    assert float(m["h"][1]) == 3.0, "the second birth must see the first birth's result"


def test_a_short_array_clamps_rather_than_raises():
    """`reindex_faces` clamps on a short array; this keeps the same contract.

    The reservoir is sized by the spec and an operator that has not grown its own array yet must not
    take the run down -- changing that is a behaviour change and belongs in its own step.
    """
    m = MeshTable()
    m["h"] = torch.arange(4.0)
    declare_vertex_carry(m, "h")
    m.carry_vertices([(99, (0, 1)), (0, (1, 2))], dt=torch.float32, dev="cpu")   # 99 is past the end
    assert float(m["h"][0]) == 1.5, "the in-range birth still applied"


def test_a_declared_but_absent_array_is_skipped():
    """An operator may declare a name before it has written the array. That is not an error."""
    m = _table()
    declare_vertex_carry(m, "not_written_yet")
    m.carry_vertices([(0, (1, 2))], dt=torch.float32, dev="cpu")                 # must not raise
    assert "not_written_yet" not in m

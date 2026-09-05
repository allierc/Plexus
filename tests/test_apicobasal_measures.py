"""The four R2 gate measures, checked against solids whose volume is known on paper.

WHY AN ANALYTIC SOLID AND NOT ONLY A RUN. `polyhedron_volume_closure` compares one cell's volume
taken about two different origins, and a closed surface gives the same answer from any origin. That
makes it a strong test of CLOSURE and a useless test of CORRECTNESS -- and the difference is not
academic: the first version of `_cell_polyhedron_volume` wound the lateral wall INWARD, and an
inward-wound wall is still closed. Both origins agreed to 1e-15 while the volume of a hexagonal
prism came out -0.866 for a cell of +2.598. The gate row was passing on a geometry that was inside
out.

So the closure row keeps its place -- it is the only thing that catches a single wall wound the
wrong way, which the maintained C++ reference needs an explicit `polygonDirections_` array to avoid
-- but it is not left alone. These cases pin the volume against `3*sqrt(3)/2` for a unit hexagonal
prism and against `l^2 h` for a box, which no orientation error can survive.
"""
from __future__ import annotations

import numpy as np
import pytest

import gate_measures as GM


def _prism(k=6, r=1.0, h=1.0, centre=(0.0, 0.0, 0.0)):
    """A single k-gon prism as (a, b, es, et, ef): ring CCW seen from +z, apical at +z."""
    th = np.arange(k) * 2 * np.pi / k
    pos = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(k)], 1) + np.asarray(centre)
    sep = np.tile([0.0, 0.0, 0.5 * h], (k, 1))
    es = np.arange(k)
    et = (np.arange(k) + 1) % k
    ef = np.zeros(k, int)
    return pos + sep, pos - sep, es, et, ef


def test_hexagonal_prism_volume_is_exact():
    """3*sqrt(3)/2 * h for a regular hexagon of side 1. The case that caught the inward wall."""
    a, b, es, et, ef = _prism(k=6, r=1.0, h=1.0)
    v = GM._cell_polyhedron_volume(a, b, es, et, ef, 1, np.zeros((1, 3)))[0]
    assert v == pytest.approx(3 * np.sqrt(3) / 2, rel=1e-12)
    assert v > 0, "a positive volume means the surface is wound OUTWARD"


def test_square_prism_volume_is_exact():
    """A second solid, because one analytic case can be matched by a coincidence of factors."""
    a, b, es, et, ef = _prism(k=4, r=np.sqrt(2) / 2, h=3.0)     # side 1, height 3
    v = GM._cell_polyhedron_volume(a, b, es, et, ef, 1, np.zeros((1, 3)))[0]
    assert v == pytest.approx(3.0, rel=1e-12)


def test_volume_is_origin_independent():
    """The property the gate row asserts, on a case where the volume is also known to be right."""
    a, b, es, et, ef = _prism()
    vs = [GM._cell_polyhedron_volume(a, b, es, et, ef, 1, o)[0]
          for o in (np.zeros((1, 3)), np.array([[0.3, -0.2, 0.1]]), np.array([[100.0, 7.0, -3.0]]))]
    assert vs[0] == pytest.approx(vs[1], rel=1e-12)
    assert vs[0] == pytest.approx(vs[2], rel=1e-9)


def test_an_inward_wall_is_still_closed_which_is_why_the_prism_case_exists():
    """THE POINT OF THIS FILE, asserted rather than described.

    Flip the separation and the solid is inside out; the closure check cannot see it, because both
    origins still agree. Only the signed volume does.
    """
    a, b, es, et, ef = _prism()
    a_flip, b_flip = b, a                                        # apical and basal swapped
    v0 = GM._cell_polyhedron_volume(a_flip, b_flip, es, et, ef, 1, np.zeros((1, 3)))[0]
    v1 = GM._cell_polyhedron_volume(a_flip, b_flip, es, et, ef, 1, np.array([[9.0, 2.0, 5.0]]))[0]
    assert v0 == pytest.approx(v1, rel=1e-9), "closure holds even inside out -- that is the trap"
    assert v0 < 0, "an inverted solid must show as a NEGATIVE volume"


def test_cap_ratio_of_a_cylinder_is_one():
    """Parallel caps of equal radius: no curvature, so the apical:basal ratio is exactly 1.

    The closed form ((R+h/2)/(R-h/2))^2 only departs from 1 because a curved shell's caps sit at
    different radii; a flat prism is the degenerate case, and getting 1 here is what says the
    measure is reading cap areas and not something else.
    """
    a, b, es, et, ef = _prism(k=6, r=1.0, h=1.0)

    class _T:
        def n_rows(self): return 1
        def nF(self, t): return 1
        def pos(self, t): return 0.5 * (a + b)
        def half_edges(self, t): return es, et, ef
        def vertex_block(self, name, t): return 0.5 * (a - b)

    assert GM.MEASURES["cap_area_ratio"](_T())[0] == pytest.approx(1.0, rel=1e-12)


def test_span_invalid_count_flags_an_inverted_vertex():
    """AB-B1 must actually fire. A row that has only ever returned 0 is indistinguishable from a
    row that returns 0 unconditionally."""
    a, b, es, et, ef = _prism()
    pos = 0.5 * (a + b)
    sep = 0.5 * (a - b)
    sep[2] = -sep[2]                                             # one vertex points the wrong way

    class _T:
        def n_rows(self): return 1
        def nF(self, t): return 1
        def pos(self, t): return pos
        def half_edges(self, t): return es, et, ef
        def vertex_block(self, name, t): return sep

    assert GM.MEASURES["apicobasal_span_invalid_count"](_T())[0] == 1.0


def test_recorded_fraction_is_zero_without_a_span():
    """AB-B7's whole job is to notice a run where the block is absent or all-zero."""
    class _Absent:
        def n_rows(self): return 2
        def vertex_block(self, name, t): return None

    class _Zero:
        def n_rows(self): return 2
        def vertex_block(self, name, t): return np.zeros((5, 3))

    assert GM.MEASURES["apicobasal_span_recorded_fraction"](_Absent()) == [0.0, 0.0]
    assert GM.MEASURES["apicobasal_span_recorded_fraction"](_Zero()) == [0.0, 0.0]

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

from plexus.models.base import Level
from plexus.models.mesh import MeshTable, declare_vertex_carry
from plexus.models.state import (Block, StateSchema, FIRST_ORDER, SECOND_ORDER_COORDINATE,
                                 SECOND_ORDER_RATE, BOUNDARY_WORLD)
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


# --------------------------------------------------------------------------------------------- #
#  THE INTEGRATION CASE. Everything above calls `carry_vertices` and the topology functions
#  DIRECTLY. None of it runs `cell_divide`, so none of it can tell whether the operator actually
#  collects the births and spends them -- which is the wiring this rung is, and the part a unit test
#  on a cube cannot reach.
# --------------------------------------------------------------------------------------------- #
def test_cell_divide_carries_a_declared_array_on_a_real_tissue(tmp_path):
    """Seed a 24-cell vesicle, declare a per-vertex array, divide, and check every NEW vertex.

    The array is `arange`, so each vertex starts holding its own index. A vertex left at its stale
    reservoir value therefore holds a number >= Nv0, while any carried value is a blend of SEEDED
    vertices and must be <= the largest seeded index. That is the assertion, and it is the one that
    separates "carried" from "happens to have changed".

    NOT ALWAYS A HALF-INTEGER, and the first version of this test asserted that it was and failed on
    h[50] = 23.25. Within ONE `cell_divide` call, `divide_face_3d` inserts each midpoint into the
    rings of the two NEIGHBOURING faces; a face divided later in the same sweep can therefore split
    an edge whose endpoint is a vertex born moments earlier, and its midpoint is the mean of a mean.
    That is the chained-birth case `test_births_are_applied_in_order` covers in miniature -- here it
    arises on its own, which is the better evidence that the ordering matters in production.
    """
    import yaml
    from plexus.engine import build
    from plexus.models.registry import get_operator
    from plexus.schema import load as load_spec

    raw = {"general": {"name": "carry_probe", "seed": 0, "n_frames": 1, "dt": 1.0, "dim": 3,
                       "world": [40.0, 40.0, 40.0]},
           "sets": {"vertex": {"n": 4096, "mesh": "half_edge", "cell_set": "cell"},
                    "cell": {"n": 1024, "state": {"area": {"width": 1}, "cen": {"width": 3}}}},
           "fields": {},
           "operators": [{"op": "mesh_seed", "at": "vertex", "before_frame": 1, "cell_set": "cell",
                          "n_cells": 24, "radius": 5.0, "jitter": 0.18, "p0": 3.5, "seed": 0},
                         {"op": "cell_divide", "at": "vertex", "cell_set": "cell", "every": 1,
                          "factor": 0.0, "p0": 3.5}],
           "schedule": ["mesh_seed", "cell_divide"]}
    path = str(tmp_path / "carry_probe.yaml")
    yaml.safe_dump(raw, open(path, "w"))
    sim = load_spec(path)
    H = build(sim, device="cpu")

    def _op(o):
        return get_operator(o.op, variant=o.impl)(
            {**o.params, "to": o.to, "from": o.frm, "_at": o.on.set}, "cpu")

    # `mesh_seed` IS RUN THROUGH THE TICK PATH, NOT `engine.seed()`. It is declared under
    # `operators:` with `kind: seed` -- the deprecated spelling the loader still accepts via the
    # legacy seed-window -- so `seed(H, sim)` does not run it and the mesh has no `A0`. Driving the
    # operators in schedule order is what `test_operator_dt` does and is the honest reproduction.
    _op(sim.operators[0]).forward(H)
    lvl = H.level("vertex")
    m = lvl.mesh
    Nv0 = int(m["Nv"])

    m["h"] = torch.arange(float(lvl.state.shape[0]))
    declare_vertex_carry(m, "h")
    before = m["h"].clone()

    _op(sim.operators[-1]).forward(H)

    Nv1 = int(m["Nv"])
    assert Nv1 > Nv0, "no vertex was added -- `factor: 0.0` should divide every cell"
    seeded_max = float(before[:Nv0].max())
    for i in range(Nv0, Nv1):
        v = float(m["h"][i])
        assert v != float(before[i]), f"vertex {i} kept its stale reservoir value {v}"
        assert v <= seeded_max, (
            f"h[{i}]={v} exceeds the largest seeded value {seeded_max}, so it is not a blend of "
            f"seeded vertices -- the carry did not run and this is reservoir data")
        assert v >= 0.0


def test_carry_reaches_a_level_state_block_not_only_a_mesh_column():
    """A declared name may live on the LEVEL, not in the mesh table -- and R2 walked into this.

    A per-vertex quantity can be a mesh column or a state block on the Level. The apico-basal
    separation is the second kind, deliberately: a state block reaches the trajectory through the
    generic per-set recording path, so it never touches FACE_RECORD/EDGE_RECORD/snapshot() and does
    not trip the NO NEW RECORDED ARRAYS rule.

    The first `carry_vertices` only looked in the table. `declare_vertex_carry(m, "sep")` succeeded,
    `m.get("sep")` returned None, the loop skipped it -- SILENTLY -- and every vertex born by
    division kept the buffer's zero. Measured on `ab_sphere` at 60 frames: all 66 newly born
    vertices held |sep| = 0.0000 against a seeded 0.2000, which is a cell of zero height along the
    seam it had just grown. That is the exact failure the carry exists to prevent, so it gets a test
    that fails without the fix rather than a comment.
    """
    schema = StateSchema([
        Block("pos", 3, role="coordinate", integration=SECOND_ORDER_COORDINATE,
              boundary=BOUNDARY_WORLD),
        Block("vel", 3, role="rate", integration=SECOND_ORDER_RATE, boundary="free"),
        Block("sep", 3, integration=FIRST_ORDER, boundary="free"),
    ])
    lvl = Level("vertex", depth=0, state=torch.zeros(6, schema.dim), state_schema=schema)
    c0, c1 = schema["sep"]
    lvl.state[0, c0:c1] = torch.tensor([0.0, 0.0, 2.0])
    lvl.state[1, c0:c1] = torch.tensor([0.0, 0.0, 4.0])

    m = MeshTable()
    declare_vertex_carry(m, "sep")
    assert "sep" not in m, "the fixture must NOT put the block in the table -- that is the point"

    m.carry_vertices([(2, (0, 1))], dt=torch.float32, dev="cpu", level=lvl)
    assert float(lvl.state[2, c1 - 1]) == 3.0, (
        "the new vertex kept the buffer's zero: the carry looked only in the mesh table")
    assert float(lvl.state[3, c1 - 1]) == 0.0, "an untouched vertex moved"


def test_a_level_block_is_not_needed_when_the_name_is_a_mesh_column():
    """The table still wins when the name is there, so passing `level` changes nothing else."""
    m = MeshTable()
    m["h"] = torch.arange(6.0)
    declare_vertex_carry(m, "h")
    schema = StateSchema([Block("pos", 3, role="coordinate",
                                integration=SECOND_ORDER_COORDINATE, boundary=BOUNDARY_WORLD)])
    lvl = Level("vertex", depth=0, state=torch.zeros(6, schema.dim), state_schema=schema)
    m.carry_vertices([(0, (2, 4))], dt=torch.float32, dev="cpu", level=lvl)
    assert float(m["h"][0]) == 3.0

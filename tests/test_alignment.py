"""Behavioural tests for the contact-gated `alignment` operator.

Run as a script (no pytest needed):
    PYTHONPATH=src python tests/test_alignment.py
or under pytest:
    PYTHONPATH=src pytest tests/test_alignment.py

These exercise the *physics*, not just registration: the alignment direction and
scale, the smoothstep contact gate (weighting two neighbours at different
distances), the hard exclusion of out-of-contact neighbours, the no-contact ->
zero case, occupancy masking, and periodic minimum-image wrap.
"""
from __future__ import annotations

import torch

import plexus.operators  # noqa: F401  self-registers `alignment`
from plexus.models.base import Hierarchy, Level
from plexus.models.registry import _OPERATOR_REGISTRY

SCHEMA = {"pos": (0, 2), "vel": (2, 4)}
Alignment = _OPERATOR_REGISTRY["alignment"]


def make_H(pos, vel, edges, *, occ=None, periodic=False, world_width=1.0):
    """A one-level Hierarchy of particles with an explicit edge list.

    `edges` is [2, E] with row0 = receiver i, row1 = neighbour j (the operator's
    convention). pos/vel are [N, 2] lists; occ defaults to all-live.
    """
    pos = torch.tensor(pos, dtype=torch.float32)
    vel = torch.tensor(vel, dtype=torch.float32)
    state = torch.cat([pos, vel], dim=1)                # [N, 4]
    N = state.shape[0]
    o = torch.ones(N) if occ is None else torch.tensor(occ, dtype=torch.float32)
    lvl = Level("particle", level=0, state=state, occ=o,
                edge_index=torch.tensor(edges, dtype=torch.long),
                state_schema=SCHEMA)
    H = Hierarchy()
    H.add_level(lvl)
    H.periodic = periodic
    H.world_width = world_width
    return H


def approx(a, b, tol=1e-6):
    return torch.allclose(a, torch.as_tensor(b, dtype=a.dtype), atol=tol)


# --------------------------------------------------------------------------- #


def test_direction_and_scale():
    """A single in-contact neighbour: acc = a * (vel_j - vel_i)."""
    H = make_H(pos=[[0.0, 0.0], [0.01, 0.0]],
               vel=[[1.0, 0.0], [0.0, 1.0]],
               edges=[[0], [1]])                        # receiver 0 <- neighbour 1
    op = Alignment({"a": 5e-4, "r": 0.05, "softness": 0.5})
    acc = op(H)["particle"]
    assert approx(acc[0], [5e-4 * -1.0, 5e-4 * 1.0]), acc[0]   # a * (dv = [-1, 1])
    assert approx(acc[1], [0.0, 0.0]), acc[1]                  # no edge into node 1
    print("ok  direction_and_scale")


def test_contact_cutoff_excludes_distant():
    """A neighbour beyond r contributes nothing -> result equals the contact-only case."""
    H = make_H(pos=[[0.0, 0.0], [0.01, 0.0], [0.5, 0.0]],     # node 2 is far (> r)
               vel=[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]],
               edges=[[0, 0], [1, 2]])                        # receiver 0 sees both 1 and 2
    op = Alignment({"a": 5e-4, "r": 0.05, "softness": 0.5})
    acc = op(H)["particle"]
    # node 2 (vel 5,5 at dist 0.5) is masked out; only node 1 (in contact) counts.
    assert approx(acc[0], [5e-4 * -1.0, 5e-4 * 1.0]), acc[0]
    print("ok  contact_cutoff_excludes_distant")


def test_no_contact_is_zero():
    """Only out-of-contact neighbours -> zero alignment (degree clamps, no NaN)."""
    H = make_H(pos=[[0.0, 0.0], [0.5, 0.0]],
               vel=[[1.0, 0.0], [9.0, 9.0]],
               edges=[[0], [1]])
    op = Alignment({"a": 5e-4, "r": 0.05, "softness": 0.5})
    acc = op(H)["particle"]
    assert torch.isfinite(acc).all()
    assert approx(acc[0], [0.0, 0.0]), acc[0]
    print("ok  no_contact_is_zero")


def test_smooth_gate_weights_two_neighbours():
    """The smoothstep weight differentiates two neighbours at different distances.

    With r=1, softness=0.5 -> r_in=0.5: neighbour at 0.3 has weight 1, neighbour
    at 0.75 has weight smoothstep(t=0.5)=0.5. Receiver vel 0; neighbour A drives
    +x, neighbour B drives +y, so acc ∝ (w_A, w_B) = (1, 0.5).
    """
    H = make_H(pos=[[0.0, 0.0], [0.30, 0.0], [0.0, 0.75]],
               vel=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
               edges=[[0, 0], [1, 2]])
    op = Alignment({"a": 1.0, "r": 1.0, "softness": 0.5})
    acc = op(H)["particle"]
    wA, wB = 1.0, 0.5
    expected = [wA / (wA + wB), wB / (wA + wB)]          # a=1, weighted mean
    assert approx(acc[0], expected, tol=1e-5), (acc[0], expected)
    # x-component must dominate y (closer neighbour weighted more):
    assert acc[0, 0] > acc[0, 1]
    print("ok  smooth_gate_weights_two_neighbours")


def test_softness_zero_is_hard_cutoff():
    """softness=0: in (just inside r) vs out (just outside r) is a clean step."""
    op = Alignment({"a": 1.0, "r": 1.0, "softness": 0.0})
    inside = make_H(pos=[[0.0, 0.0], [0.9, 0.0]], vel=[[0.0, 0.0], [1.0, 0.0]],
                    edges=[[0], [1]])
    outside = make_H(pos=[[0.0, 0.0], [1.1, 0.0]], vel=[[0.0, 0.0], [1.0, 0.0]],
                     edges=[[0], [1]])
    a_in = op(inside)["particle"][0]
    a_out = op(outside)["particle"][0]
    assert approx(a_in, [1.0, 0.0]), a_in        # full alignment inside
    assert approx(a_out, [0.0, 0.0]), a_out       # nothing outside
    print("ok  softness_zero_is_hard_cutoff")


def test_occupancy_masks_neighbour():
    """A dead neighbour (occ=0), even in contact, is ignored."""
    H = make_H(pos=[[0.0, 0.0], [0.01, 0.0]],
               vel=[[1.0, 0.0], [0.0, 1.0]],
               edges=[[0], [1]], occ=[1.0, 0.0])
    op = Alignment({"a": 5e-4, "r": 0.05, "softness": 0.5})
    acc = op(H)["particle"]
    assert approx(acc[0], [0.0, 0.0]), acc[0]
    print("ok  occupancy_masks_neighbour")


def test_periodic_minimum_image():
    """Across a periodic wall the wrapped separation (0.02) is in contact, not 0.98."""
    pos = [[0.99, 0.0], [0.01, 0.0]]
    vel = [[1.0, 0.0], [0.0, 1.0]]
    op = Alignment({"a": 5e-4, "r": 0.05, "softness": 0.5})
    flat = op(make_H(pos, vel, [[0], [1]], periodic=False, world_width=1.0))["particle"]
    wrap = op(make_H(pos, vel, [[0], [1]], periodic=True, world_width=1.0))["particle"]
    assert approx(flat[0], [0.0, 0.0]), flat[0]                       # dist 0.98 -> no contact
    assert approx(wrap[0], [5e-4 * -1.0, 5e-4 * 1.0]), wrap[0]        # wrapped dist 0.02 -> contact
    print("ok  periodic_minimum_image")


def test_empty_edges():
    """No edges -> zeros of the right shape (the early-return path)."""
    H = make_H(pos=[[0.0, 0.0], [0.5, 0.0]], vel=[[1.0, 0.0], [2.0, 0.0]],
               edges=[[], []])
    op = Alignment({"a": 5e-4})
    acc = op(H)["particle"]
    assert acc.shape == (2, 2)
    assert approx(acc, [[0.0, 0.0], [0.0, 0.0]])
    print("ok  empty_edges")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall alignment tests passed.")

"""Behavioural tests for the `alignment` operator.

Run as a script (no pytest needed):
    PYTHONPATH=src python prototype/dicty/test_alignment.py
or under pytest:
    PYTHONPATH=src pytest prototype/dicty/test_alignment.py

These exercise the *physics*, not just registration: the alignment direction and
scale, the smoothstep contact gate (weighting two neighbours at different
distances), the hard exclusion of out-of-contact neighbours, the no-contact ->
zero case, occupancy masking, and periodic minimum-image wrap -- plus the
`gate: none` regime (ungated plain mean) and its exact equivalence to the
`boids` operator's alignment term when `per_type` is on.
"""
from __future__ import annotations

import os
import sys

import torch

# this folder is a standalone prototype: import the sibling operator module to
# register `alignment` (PYTHONPATH=src is still needed for the plexus package).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alignment  # noqa: F401,E402  self-registers `alignment`
from plexus.models.base import Hierarchy, Level                       # noqa: E402
from plexus.models.registry import _OPERATOR_REGISTRY                 # noqa: E402

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


def add_types(H, node_type, type_params, at="particle"):
    """Attach per-type props (what the engine builds) so per-type operators run.

    `node_type` is an [N] type id per node; `type_params` is [n_types, k] (e.g.
    boids' p = [cohesion, alignment, separation] per type).
    """
    lvl = H.level(at)
    lvl.register_buffer("node_type", torch.tensor(node_type, dtype=torch.long))
    lvl.register_buffer("type_params", torch.tensor(type_params, dtype=torch.float32))
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


def test_gate_none_is_ungated_plain_mean():
    """gate='none' ignores distance: every graph neighbour counts equally.

    Two neighbours, one far beyond any contact radius. With the contact gate the
    far one is dropped; with gate='none' both count equally -> acc is the plain
    mean of the two heading differences.
    """
    H = make_H(pos=[[0.0, 0.0], [0.01, 0.0], [0.5, 0.0]],     # node 2 is far
               vel=[[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]],
               edges=[[0, 0], [1, 2]])                        # receiver 0 sees both
    op = Alignment({"a": 1.0, "gate": "none"})
    acc = op(H)["particle"]
    # plain mean of dv_1=(2,0) and dv_2=(0,4) -> (1, 2)
    assert approx(acc[0], [1.0, 2.0]), acc[0]
    print("ok  gate_none_is_ungated_plain_mean")


def test_gate_none_matches_boids_alignment():
    """gate='none' + per_type reproduces the boids operator's alignment term exactly.

    Run the real `boids` op with cohesion/separation zeroed (p=[0, p2, 0]) so only
    its alignment term survives, and compare to `alignment` with gate='none',
    per_type=True, type_index=1. They must agree to floating-point.
    """
    from plexus.operators.boids import Boids

    pos = [[0.0, 0.0], [0.30, 0.0], [0.0, 0.20], [0.7, 0.7]]
    vel = [[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [-2.0, 1.0]]
    edges = [[0, 0, 1], [1, 2, 3]]                            # a few receivers/neighbours
    p2 = 7.0                                                  # per-type alignment weight
    # single type carrying p = [cohesion=0, alignment=p2, separation=0]
    def build():
        H = make_H(pos, vel, edges, periodic=False, world_width=1.0)
        return add_types(H, node_type=[0, 0, 0, 0], type_params=[[0.0, p2, 0.0]])

    boids = Boids({"a2": 5e-4})(build())["particle"]
    align = Alignment({"a": 5e-4, "gate": "none",
                       "per_type": True, "type_index": 1})(build())["particle"]
    assert approx(align, boids, tol=1e-9), (align, boids)
    print("ok  gate_none_matches_boids_alignment")


def test_spec_runs_through_engine():
    """The `alignment.yaml` spec loads and drives the operator through the engine.

    Reuses the generic runner (plexus.schema.load + plexus.engine.run) -- no
    bespoke harness -- to prove the registered operator is spec-drivable end to
    end. Asserts the run is finite, the right shape, the flock actually moves,
    and the Vicsek order parameter (heading coherence) *rises* as it aligns.
    """
    from plexus.schema import load
    from plexus.engine import build, run

    spec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alignment.yaml")
    sim = load(spec_path)

    # Vicsek order parameter phi = |mean(v/|v|)|: ~0 for random headings, ->1 aligned.
    def order(v):
        u = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return u.mean(0).norm().item()

    phi_start = order(build(sim).level("particle").get("vel"))   # seeded initial headings
    H, out = run(sim)                                            # build hierarchy + iterate schedule

    pos = out["sets"]["particle"]["pos"]                # [n_frames+1, N, 2]
    assert pos.shape == (sim.n_frames + 1, 600, 2), pos.shape
    assert torch.isfinite(torch.as_tensor(pos)).all()
    moved = torch.as_tensor(pos[-1] - pos[0]).abs().sum().item()
    assert moved > 0.0, "flock never moved"

    # contact-gating builds *local* alignment domains -> global phi rises clearly
    # above the random start but plateaus below 1; just assert the increase.
    phi_end = order(H.level("particle").get("vel"))
    print(f"ok  spec_runs_through_engine   (phi {phi_start:.3f} -> {phi_end:.3f}, moved={moved:.2f})")
    assert phi_end > phi_start + 0.05, f"alignment did not raise heading coherence ({phi_start:.3f} -> {phi_end:.3f})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall alignment tests passed.")

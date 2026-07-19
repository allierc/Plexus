"""PR 1 smoke tests: State as a first-class primitive.

Covers the three things the refactor adds, WITHOUT any signaling operators or
neural entities (those are later PRs):

  1. the compatibility shim -- a legacy {block:(c0,c1)} dict normalizes to a
     StateSchema that is still dict-indexable and integrates like pos/vel;
  2. schema-driven integration of a NON-spatial, first-order, free-boundary block
     (a neuron voltage), with no world clamp and no rate block;
  3. byte-identical integration of the spatial pos/vel path (velocity and
     acceleration EMIT), so existing specs are unchanged.

The 153-spec byte-identical regression is proved separately by
`scripts/state_baseline.py compare`; this file unit-tests the new code paths.
"""
import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.state import (
    StateSchema, Block, spatial_schema, schema_from_spec,
    FIRST_ORDER, SECOND_ORDER_COORDINATE, SECOND_ORDER_RATE, BOUNDARY_WORLD,
)
from plexus.engine import _integrate


# --------------------------------------------------------------------------- #
#  1. the compatibility shim
# --------------------------------------------------------------------------- #
def test_normalize_legacy_dict():
    s = StateSchema.normalize({"pos": (0, 2), "vel": (2, 4)})
    assert s.dim == 4
    assert s["pos"] == (0, 2) and s["vel"] == (2, 4)      # still dict-indexable
    assert "pos" in s and "vel" in s
    assert s.coordinate.name == "pos"
    assert s.coordinate.integration == SECOND_ORDER_COORDINATE
    assert s.coordinate.boundary == BOUNDARY_WORLD
    assert s.rate.name == "vel" and s.rate.integration == SECOND_ORDER_RATE


def test_normalize_is_idempotent():
    s = spatial_schema(3)
    assert StateSchema.normalize(s) is s
    assert s.dim == 6


def test_schema_from_spec_neuron():
    s = schema_from_spec({"voltage": 1, "calcium": 1})
    assert s.dim == 2
    assert s.coordinate.name == "voltage" and s.coordinate.integration == FIRST_ORDER
    assert s.rate is None                                  # first-order: no rate block
    assert s.coordinate.boundary == "free"                # no world box
    assert [b.name for b in s.recorded] == ["voltage", "calcium"]


# --------------------------------------------------------------------------- #
#  helpers: a minimal one-set Hierarchy
# --------------------------------------------------------------------------- #
def _one_level(schema: StateSchema, n=4, emit="velocity", **hattrs) -> Hierarchy:
    H = Hierarchy()
    H.dim = 2
    state = torch.zeros(n, schema.dim)
    H.add_level(Level("s", depth=0, state=state, state_schema=schema))
    H.world_size = torch.tensor([1.0, 1.0])
    H.boundary = hattrs.get("boundary", "wall")
    H.emit_order = {"s": emit}
    return H


# --------------------------------------------------------------------------- #
#  2. non-spatial (voltage) integration: first-order, free boundary
# --------------------------------------------------------------------------- #
def test_voltage_integrates_first_order_no_clamp():
    schema = schema_from_spec({"voltage": 1})
    H = _one_level(schema, n=3, emit="velocity", boundary="wall")
    lvl = H.level("s")
    # start voltage at 0.5; inject a constant derivative of +10 (would blow past the
    # world box [0,1] -- proving a free block is NOT clamped like a spatial coordinate)
    lvl.state[:, 0] = 0.5
    dt = 0.1
    H._delta = {"s": torch.full((3, 1), 10.0)}
    _integrate(H, dt)
    # first-order: v_next = v + dt*delta = 0.5 + 0.1*10 = 1.5  (exceeds the box, not clamped)
    assert torch.allclose(lvl.state[:, 0], torch.full((3,), 1.5))


# --------------------------------------------------------------------------- #
#  3. spatial pos/vel integration matches the hand-written formula exactly
# --------------------------------------------------------------------------- #
def test_spatial_velocity_emit_matches_formula():
    H = _one_level(spatial_schema(2), n=5, emit="velocity", boundary="free")
    lvl = H.level("s")
    lvl.state[:, 0:2] = 0.3                                 # pos
    delta = torch.randn(5, 2, generator=torch.Generator().manual_seed(0))
    H._delta = {"s": delta.clone()}
    dt = 0.05
    _integrate(H, dt)
    # EMIT=velocity: vel := delta; pos += dt*vel
    assert torch.allclose(lvl.state[:, 2:4], delta)                    # vel set to the delta
    assert torch.allclose(lvl.state[:, 0:2], 0.3 + dt * delta)         # pos advanced by dt*vel


def test_spatial_acceleration_emit_matches_formula():
    H = _one_level(spatial_schema(2), n=5, emit="acceleration", boundary="free")
    lvl = H.level("s")
    lvl.state[:, 0:2] = 0.3                                 # pos
    lvl.state[:, 2:4] = 0.1                                 # vel
    acc = torch.randn(5, 2, generator=torch.Generator().manual_seed(1))
    H._delta = {"s": acc.clone()}
    dt = 0.05
    _integrate(H, dt)
    v_expected = 0.1 + dt * acc
    assert torch.allclose(lvl.state[:, 2:4], v_expected)               # vel += dt*acc
    assert torch.allclose(lvl.state[:, 0:2], 0.3 + dt * v_expected)    # pos += dt*vel


def test_delta_dim_is_coordinate_width():
    # a neuron delta is voltage-width (1), a spatial delta is pos-width (== H.dim)
    Hn = _one_level(schema_from_spec({"voltage": 1}), n=3)
    Hn.zero_delta()
    assert Hn.delta("s").shape == (3, 1)
    Hs = _one_level(spatial_schema(2), n=3)
    Hs.zero_delta()
    assert Hs.delta("s").shape == (3, 2)

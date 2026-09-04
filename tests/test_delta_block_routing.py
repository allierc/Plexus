"""One operator, two independently-integrated blocks of one set -- R1(a) of the apico-basal promotion.

WHAT THIS PINS. `engine._run`'s tick loop read ONE class attribute, `INTEGRAND`, and applied it to
every delta an operator returned, so an operator could advance a set's coordinate block OR one other
block, never both. The apico-basal `cell_mechanics` has a single energy whose gradient falls on two
blocks of the vertex set -- the mid-surface `pos` and the apico-basal separation `sep` -- and
splitting it into two operators would mean differentiating one energy twice and calling the halves
separate mechanisms. A returned key may now be `(set, block)`.

THE SECOND TEST IS THE ONE THAT MATTERS FOR THE PROMOTION. `test_bare_key_is_unchanged` is the
byte-identity claim in miniature: an operator that returns bare set keys must reach exactly the
accumulators it reached before, because every one of the 2,456 specs in the corpus does that and the
twin gate hashes their trajectories. A routing change that moved a bare-key delta by one ulp would
be invisible here and fatal there.

THE THIRD IS THE TRAP THE `wall`-SET DESIGN FELL INTO. `_integrate` walks `H.emit_order`, built from
each operator's DECLARED set, so a delta returned for a set nothing is declared `at:` never
integrates -- it sits in `H._delta` and `zero_delta` wipes it at the next tick. Putting a second
surface on its own set would have left it FROZEN while the gate rows measured it, and the run would
have looked healthy throughout. `test_second_block_is_not_a_second_set` records that a block on the
declared set does integrate, which is the property the `(set, block)` key buys.
"""
from __future__ import annotations

import torch

from plexus.models.base import Hierarchy, Level
from plexus.models.state import Block, StateSchema, FIRST_ORDER, SECOND_ORDER_COORDINATE, \
    SECOND_ORDER_RATE, BOUNDARY_WORLD


def _hierarchy(n=4, dim=3):
    """The vertex set as production builds it -- `pos`/`vel` inertial and world-clamped -- PLUS one
    extra first-order block standing in for the apico-basal separation.

    BUILT FROM THE SAME `Block`s THE SPEC LOADER EMITS, not from a hand-rolled stand-in. The
    `renumber_set` defect of 23 August is the precedent: its unit test built `H.levels = {...}` as a
    plain dict, which has `.get`, while production drove an `nn.ModuleDict`, which does not -- so the
    fixture exercised the live branch and production the dead one, and the test passed for months
    over a method that never acted. A fixture that cannot reproduce production is not a test.
    """
    schema = StateSchema([
        Block("pos", dim, role="coordinate", integration=SECOND_ORDER_COORDINATE,
              boundary=BOUNDARY_WORLD),
        Block("vel", dim, role="rate", integration=SECOND_ORDER_RATE, boundary="free"),
        Block("sep", dim, integration=FIRST_ORDER, boundary="free"),
    ])
    H = Hierarchy()
    H.dim = dim
    H.add_level(Level("vertex", depth=0, state=torch.zeros(n, schema.dim), state_schema=schema))
    H.world_size = torch.ones(dim)
    H.boundary = "free"
    H.emit_order = {"vertex": "velocity"}
    H.zero_delta()
    return H, schema


def test_tuple_key_routes_to_the_named_block():
    """`(set, block)` puts the delta in that block's accumulator, not the coordinate one."""
    H, _ = _hierarchy()
    H.add_delta("vertex", torch.ones(4, 3), "sep")
    assert "sep" in H._delta_blocks["vertex"], "the named block got no accumulator"
    assert torch.count_nonzero(H._delta["vertex"]) == 0, (
        "a non-coordinate delta leaked into the coordinate accumulator -- this is the failure that "
        "would move `pos` when only the apico-basal separation was meant to change")


def test_bare_key_is_unchanged():
    """A delta with block=None goes to the coordinate accumulator, exactly as before.

    THE BYTE-IDENTITY CLAIM. Every spec in the corpus returns bare set keys; if this moved, the twin
    gate would report a DIFFER on 2,456 specs and the promotion would be indistinguishable from a
    regression.
    """
    H, _ = _hierarchy()
    H.add_delta("vertex", torch.full((4, 3), 2.0), None)
    assert torch.equal(H._delta["vertex"], torch.full((4, 3), 2.0))
    assert H._delta_blocks == {}, "a coordinate delta created a block accumulator"


def test_coordinate_block_by_name_is_the_coordinate_accumulator():
    """Naming the coordinate block explicitly must be the same thing as naming nothing.

    `add_delta` guards on `block == coord_name`, and without that guard an operator that spelled its
    own integrand `pos` would get a SECOND accumulator for the coordinate -- integrated by the
    block loop as well as by the coordinate loop, i.e. twice.
    """
    H, _ = _hierarchy()
    H.add_delta("vertex", torch.ones(4, 3), "pos")
    assert torch.equal(H._delta["vertex"], torch.ones(4, 3))
    assert H._delta_blocks == {}


def test_two_blocks_of_one_set_accumulate_apart():
    """The whole point: one operator, one call, two blocks that do not contaminate each other."""
    H, _ = _hierarchy()
    for key, d in {("vertex", None): torch.ones(4, 3),
                   ("vertex", "sep"): torch.full((4, 3), 5.0)}.items():
        H.add_delta(key[0], d, key[1])
    assert torch.equal(H._delta["vertex"], torch.ones(4, 3))
    assert torch.equal(H._delta_blocks["vertex"]["sep"], torch.full((4, 3), 5.0))


def test_repeated_adds_sum_within_a_block():
    """Two operators writing the same block sum into it, as the coordinate accumulator does."""
    H, _ = _hierarchy()
    H.add_delta("vertex", torch.ones(4, 3), "sep")
    H.add_delta("vertex", torch.full((4, 3), 3.0), "sep")
    assert torch.equal(H._delta_blocks["vertex"]["sep"], torch.full((4, 3), 4.0))


# --------------------------------------------------------------------------------------------- #
#  THE END-TO-END CASE. Everything above tests `Hierarchy.add_delta`, WHICH ALREADY TOOK A `block`
#  BEFORE THIS CHANGE -- so on its own it is a suite of green rows on a spec that cannot exercise
#  the thing it is named for. The change is in `engine._run`'s tick loop, which USED to apply one
#  class-level `INTEGRAND` to every returned delta; only a run through that loop can see it.
# --------------------------------------------------------------------------------------------- #
def _probe_spec(tmpdir, dt=1.0, n_frames=3):
    """A two-particle set with a `sep` block, driven by a probe operator that writes BOTH blocks.

    Written out and read back through `schema.load`, as `test_operator_dt` does, so the probe goes
    through the same parser a real spec does rather than a hand-assembled Spec the loader would
    have rejected.

    `pos` IS DECLARED EXPLICITLY AND THAT IS LOAD-BEARING. `schema_from_spec` makes the first
    declared block the coordinate unless it recognises `pos`, so a set declaring only `sep` gets
    `sep` AS ITS COORDINATE -- and then `add_delta(..., "sep")` hits the `block == coord_name` guard
    and lands in the coordinate accumulator, exactly where the untagged delta went. Both deltas sum,
    the block appears to integrate, and the test passes whether or not the tick loop routes anything.
    Written the wrong way first, this probe read 9.0 (= (1+2) x 3 ticks) instead of 6.0.
    """
    import os
    import yaml
    raw = {
        "general": {"name": "delta_block_probe", "seed": 0, "n_frames": n_frames, "dt": dt,
                    "dim": 3, "world": [10.0, 10.0, 10.0], "boundary": "free"},
        "sets": {"p": {"n": 2, "state": {"pos": {"width": 3},
                                         "sep": {"width": 3, "integration": "first_order"}}}},
        "fields": {},
        "operators": [{"op": "_two_block_probe", "at": "p"}],
        "schedule": ["_two_block_probe"],
    }
    path = os.path.join(tmpdir, "delta_block_probe.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return path


def test_tick_loop_routes_a_tuple_key(tmp_path):
    """One operator, one call, two blocks -- through `engine.run`, not through `add_delta`.

    The probe returns `{"p": +1 on pos, ("p", "sep"): +2 on sep}`. Before the change both keys would
    have taken the operator's single `INTEGRAND` (None), so BOTH would have landed on `pos`: `sep`
    would have stayed at zero and `pos` would have moved by 3 per frame instead of 1. Those are the
    two numbers this asserts.
    """
    import torch as _t
    from plexus.engine import run
    from plexus.models.base import Lateral
    from plexus.models.registry import register_operator
    from plexus.schema import load as load_spec

    @register_operator("_two_block_probe", set="p", kind="lateral", family="mechanics")
    class _TwoBlockProbe(Lateral):
        """Writes a constant +1 to the coordinate block and +2 to `sep`, in ONE forward()."""
        EMIT = "velocity"
        SUPPORTED_DIMS = (3,)

        def forward(self, H, mask=None):
            n = H.level("p").n
            dev = H.level("p").state.device
            return {"p": _t.ones(n, 3, device=dev),
                    ("p", "sep"): _t.full((n, 3), 2.0, device=dev)}

    sim = load_spec(_probe_spec(str(tmp_path), dt=1.0, n_frames=3))
    H, _traj = run(sim, out_path=None, device="cpu")
    lvl = H.level("p")
    c0, c1 = lvl.state_schema.slice("sep")
    sep = lvl.state[:, c0:c1]

    assert _t.allclose(sep, _t.full_like(sep, 6.0)), (
        f"`sep` should have integrated 3 ticks x dt 1.0 x delta 2.0 = 6.0, got {sep[0].tolist()}. "
        "Zero here means the tuple key was dropped and both deltas went to the coordinate block -- "
        "the pre-change behaviour, and the one that would have frozen a second surface silently.")
    assert not _t.allclose(sep, _t.zeros_like(sep)), "the second block never moved"


def test_second_block_is_not_a_second_set():
    """`_integrate` advances a set's extra blocks WITHOUT consulting `emit_order`.

    This is why `sep` is a block on the vertex set and not a `wall` set of its own: the block loop
    runs for every set in `_delta_blocks` unconditionally, while the coordinate loop is gated by the
    emit order that `_resolve_emit` builds from declared operator sets.
    """
    from plexus.engine import _integrate
    H, schema = _hierarchy()
    cx0, cx1 = schema.slice("sep")
    H.add_delta("vertex", torch.full((4, 3), 2.0), "sep")
    _integrate(H, dt=0.5)
    assert torch.allclose(H.levels["vertex"].state[:, cx0:cx1], torch.full((4, 3), 1.0)), (
        "the extra block did not advance: 4 rows x dt 0.5 x delta 2.0 should be 1.0")

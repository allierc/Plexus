"""The two engine hooks a learnable substitution needs, before any learnable exists.

Both are useful on their own and neither changes a forward run:

    H.operators / H.parameters()   an operator holding a fitted tensor can be found from the
                                   Hierarchy, so a caller who differentiates a loss has something
                                   to hand an optimiser
    H.measured_maps()              WHICH RELATIONS each operator actually walked, measured

The second is the one with teeth. A learnable may replace an operator only if it satisfies the
same contract, and matching the declared blocks is NOT sufficient: an MLP on each neuron's own
state satisfies every field of `neuron_signal` -- same sets, same blocks, same EMIT -- while
silently dropping the connectome. It would train, score well and mean nothing. What separates
them is the relation traversed, and that is what these tests pin down.

Measured, not declared, because a declared `MAPS` was tried and removed: 23 of ~150 operators
filled it in, so an empty one meant either "checked, walks nothing" or "nobody looked".
"""
from __future__ import annotations

import pytest
import torch

import plexus.operators                                    # noqa: F401  self-registers
from plexus import engine
from plexus.schema import load

SPEC = "config/neural/ctrnn_eyeG_rig.yaml"


def _run(n_frames=3, record=False):
    sim = load(SPEC)
    sim.n_frames = n_frames
    if not record:
        return engine.run(sim, device="cpu", progress=False)[0]
    orig = engine.build

    def build(s, device):
        H = orig(s, device)
        H.record_maps(True)
        return H

    engine.build = build
    try:
        return engine.run(sim, device="cpu", progress=False)[0]
    finally:
        engine.build = orig


def test_operators_are_children_of_the_hierarchy_in_spec_order():
    H = _run()
    assert len(H.operators) == len(H.operator_names) == 6
    assert H.operator_names == ["project", "neuron_update", "neuron_signal", "readout",
                                "muscle_pose_map", "organ_mechanics"], \
        "the order is the SPEC's and is part of the contract -- callers index by it"
    assert isinstance(H.operators, torch.nn.ModuleList)


def test_parameters_is_empty_when_nothing_is_fitted():
    """The whole promoted library holds buffers, not parameters. If this ever returns a leaf,
    some operator has acquired a fitted quantity without anyone deciding it should."""
    assert sum(p.numel() for p in _run().parameters()) == 0


def test_a_parameter_on_an_operator_is_found_and_gets_a_gradient():
    """The property the hook exists for, asserted end to end rather than by inspection."""
    H = _run()
    op = H.operators[H.operator_names.index("neuron_update")]
    op.probe = torch.nn.Parameter(torch.zeros(1))
    found = [p for p in H.parameters() if p is op.probe]
    assert len(found) == 1, "a parameter on an operator must be reachable from H.parameters()"
    (op.probe * 3.0).sum().backward()
    assert op.probe.grad is not None and float(op.probe.grad) == pytest.approx(3.0)


def test_measured_maps_separates_relational_operators_from_intrinsic_ones():
    """THE SUBSTITUTABILITY TEST. Three operators walk an edge set, three walk nothing or a
    containment map, and nothing about that was declared anywhere."""
    H = _run(record=True)
    m = H.measured_maps()

    # relational: gathered along `pre`, scattered along `post`, of their OWN edge set
    assert m["project"] == [("afferent", "post"), ("afferent", "pre")]
    assert m["neuron_signal"] == [("recurrent", "post"), ("recurrent", "pre")]
    assert m["readout"] == [("junction", "post"), ("junction", "pre")]

    # containment is a relation too, and an Aggregate traverses it through `lift_index`
    assert m["muscle_pose_map"] == [("muscle->eye", "parent")]

    # intrinsic: each entity's own state, no relation at all
    for op in ("neuron_update", "organ_mechanics"):
        assert op not in m or m[op] == [], f"{op} traverses no relation and must measure as none"


def test_recording_is_off_by_default_and_costs_a_run_nothing():
    H = _run(record=False)
    assert H.measured_maps() == {}, "recording must be opt-in"


def test_the_hooks_do_not_change_what_a_run_produces():
    """A forward run is byte-identical with the recorder on and off -- otherwise the measurement
    changes the thing it measures."""
    a = _run(n_frames=5, record=False).level("eye").get("pose").detach().clone()
    b = _run(n_frames=5, record=True).level("eye").get("pose").detach().clone()
    assert torch.equal(a, b), "recording perturbed the trajectory"

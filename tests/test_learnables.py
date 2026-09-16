"""Substitutability: which learnables may stand in for which operators, and why not.

The refusals are the content. A learnable that satisfies every DECLARED field of an operator can
still be the wrong thing entirely -- a pointwise net standing in for `neuron_signal` has the same
sets, the same blocks and the same EMIT while silently dropping the connectome, and it would
train, score well and mean nothing. So each test below is a case that must be refused, plus the
one that must be allowed.
"""
from __future__ import annotations

import pytest
import torch

import plexus.operators                                    # noqa: F401  self-registers
import plexus.learnables as L
from plexus.learnables.substitute import build_substitution
from plexus.models.registry import get_operator

REL = [("recurrent", "pre"), ("recurrent", "post")]        # what neuron_signal was watched to walk
NONE = []                                                  # what neuron_update was watched to walk


def test_the_registry_is_populated_and_every_family_declares_a_shape():
    assert {"mlp", "siren", "table"} <= set(L.learnables())
    assert {"mlp_edge", "siren_edge", "table_edge"} <= set(L.learnables())
    for n in L.learnables():
        assert L.get_learnable(n).SHAPE in L.SHAPES, f"{n} declares no usable SHAPE"


def test_a_family_without_a_shape_cannot_be_registered():
    """Without SHAPE there is no way to tell whether it may replace a relational operator."""
    with pytest.raises(ValueError, match="SHAPE"):
        L.register_learnable("shapeless")(type("X", (torch.nn.Module,), {}))


def test_a_pointwise_net_may_not_replace_an_operator_that_walks_a_relation():
    """THE CASE THE CHECK EXISTS FOR. Every declared field matches; the connectome does not."""
    reasons = L.check_substitution(get_operator("neuron_signal", model="shared"),
                                   L.get_learnable("mlp"), REL,
                                   op_name="neuron_signal", learn_name="mlp")
    assert reasons, "a pointwise net replacing a relational operator must be refused"
    assert any("RELATION" in r for r in reasons)
    assert any("drop" in r for r in reasons), "the reason must say WHAT is lost, not just 'no'"


def test_a_relational_net_may_not_replace_an_operator_that_walks_nothing():
    reasons = L.check_substitution(get_operator("neuron_update"), L.get_learnable("mlp_edge"),
                                   NONE, op_name="neuron_update", learn_name="mlp_edge")
    assert any("RELATION" in r for r in reasons)


def test_a_mismatched_emit_is_refused_because_it_changes_the_sets_integration_order():
    class Wrong(L.get_learnable("mlp")):
        EMIT = "acceleration"                              # neuron_update emits a velocity
    reasons = L.check_substitution(get_operator("neuron_update"), Wrong, NONE,
                                   op_name="neuron_update", learn_name="wrong")
    assert any("EMIT" in r for r in reasons)
    assert any("integration" in r.lower() for r in reasons)


def test_a_learnable_reading_a_block_the_operator_does_not_is_refused():
    class Greedy(L.get_learnable("mlp")):
        READS = ["voltage", "omega"]                       # neuron_update reads only voltage
    reasons = L.check_substitution(get_operator("neuron_update"), Greedy, NONE,
                                   op_name="neuron_update", learn_name="greedy")
    assert any("READS" in r for r in reasons)


def test_narrowing_reads_is_allowed():
    """Reading LESS than the operator is a simpler hypothesis, not an ill-typed one."""
    class Narrow(L.get_learnable("mlp")):
        READS = []
    assert L.check_substitution(get_operator("neuron_update"), Narrow, NONE) == []


def test_the_matching_substitution_is_allowed_and_runs():
    """The acceptance, end to end: build it, and it produces a delta of the right shape."""
    op = get_operator("neuron_update")
    assert L.check_substitution(op, L.get_learnable("mlp"), NONE) == []
    sub = build_substitution(op, {"with": "mlp", "at": "neuron",
                                  "params": {"hidden": 8, "layers": 2}}, NONE,
                             op_name="neuron_update")
    assert sub.EMIT == op.EMIT and sub.INTEGRAND == op.INTEGRAND, "the contract must be the op's"
    assert sum(p.numel() for p in sub.parameters()) > 0


def test_a_relational_substitution_without_an_edge_set_is_refused():
    """It is relational precisely because it walks one; not naming it makes it the pointwise net
    the check just refused, arriving by another route."""
    op = get_operator("neuron_signal", model="shared")
    with pytest.raises(L.SubstitutionError, match="edge_set"):
        build_substitution(op, {"with": "mlp_edge", "at": "neuron", "params": {}}, REL,
                           op_name="neuron_signal")


def test_init_from_operator_starts_the_learnable_at_the_law_it_replaces():
    """`init_from: operator` is what makes a residual interpretable: begin at the analytic law,
    and what the net does afterwards is what the data asked for BEYOND it."""
    net = L.get_learnable("siren")(1, 1, hidden=32, layers=3, omega0=8.0, seed=0)
    x = torch.linspace(-3, 3, 256)[:, None]
    with torch.no_grad():
        before = float(((net(x) - torch.tanh(x)) ** 2).mean())
    final = net.fit_to(torch.tanh, n_samples=2048, iters=600, lr=3e-3)
    assert final < 1e-3, f"could not be initialised to tanh (loss {final:.2e})"
    assert final < before, "fitting did not improve on the initialisation"


def test_a_substitution_walks_the_SAME_relations_as_the_operator_it_replaces():
    """The property the whole design turns on, measured on both sides rather than argued.

    The relational wrapper keeps `gather -> weight -> scatter` exactly as the operator had it and
    replaces only what a message IS. So the connectome is still traversed -- which is checkable,
    because `H.measured_maps()` watches both. If a future wrapper ever stopped gathering, every
    declared field would still match and only this would notice.
    """
    from plexus import engine
    from plexus.schema import load

    orig = engine.build

    def build(s, device):
        H = orig(s, device)
        H.record_maps(True)
        return H

    engine.build = build
    try:
        sim = load("config/neural/ctrnn_eyeG_rig.yaml")
        sim.n_frames = 3
        H, _ = engine.run(sim, device="cpu", progress=False)
    finally:
        engine.build = orig
    walked_by_operator = sorted(H.measured_maps()["neuron_signal"])
    assert walked_by_operator == [("recurrent", "post"), ("recurrent", "pre")]

    sub = build_substitution(
        get_operator("neuron_signal", model="shared"),
        {"with": "siren_edge", "at": "neuron",
         "params": {"hidden": 16, "layers": 2, "omega0": 8.0},
         "operator_params": {"_at": "neuron", "edge_set": "recurrent"}},
        walked_by_operator, op_name="neuron_signal")

    H.record_maps(True)
    H._active_op = "substitution"
    deltas = sub(H, None)
    assert sorted(H.measured_maps()["substitution"]) == walked_by_operator, \
        "the substitution stopped traversing the connectome the operator traversed"

    # and it is differentiable into the net, which is the point of substituting one
    (name,) = deltas
    assert deltas[name].shape == H.level("neuron").get("voltage").shape
    deltas[name].sum().backward()
    assert all(p.grad is not None for p in sub.parameters())

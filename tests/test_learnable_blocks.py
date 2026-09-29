"""`learnable: {block: w, of: recurrent}` -- fitting a stated mechanism's constants.

The one form of learnable anything fits, and the one connectome work actually needs: the mechanism
is stated and its numbers are not. A synaptic weight, a time constant, a conductance. An operator
substitution -- the LAW unknown -- is fitted by nothing yet, so it is refused rather than allowed to
run as the original operator, which is what it silently did while the schema accepted it.

What must hold, and each is asserted rather than assumed:

    the block becomes a real tensor leaf, reachable from `H.parameters()`
    it STARTS at whatever the spec put there -- a connectome's measured synapse areas, not noise
    a gradient reaches it through the whole forward chain
    the SAME leaf is reused across rollouts, or an optimiser would move objects no later
      rollout reads and the loss would wander with nothing to say why
"""
from __future__ import annotations

import pytest
import torch

import plexus.operators                                    # noqa: F401  self-registers
from plexus import engine
from plexus.schema import load

SPEC = "config/neural/ctrnn_eyeG_rig.yaml"


def _sim(learnable, n_frames=4):
    sim = load(SPEC)
    sim.n_frames = n_frames
    sim.learnable = list(learnable)
    return sim


def test_a_declared_block_becomes_a_parameter_reachable_from_the_hierarchy():
    sim = _sim([{"block": "w", "of": "recurrent"}])
    H, _ = engine.run(sim, device="cpu", progress=False, grad=True)
    ps = [p for p in H.parameters()]
    assert len(ps) == 1 and ps[0].numel() == 4032, "the recurrent edge set has 4032 weights"
    assert ps[0].requires_grad


def test_it_starts_at_the_value_the_spec_declared_not_at_noise():
    """A connectome's `w` is a MEASUREMENT. A fit starting anywhere else discards the only thing
    that makes the result readable: what training did is the DEPARTURE from the measurement."""
    plain, _ = engine.run(_sim([]), device="cpu", progress=False)
    w_spec = plain.level("junction").get("w").detach().clone()
    sim = _sim([{"block": "w", "of": "junction"}])
    H, _ = engine.run(sim, device="cpu", progress=False, grad=True)
    assert torch.allclose(H.level("junction")._fitted["w"].detach(), w_spec), \
        "the fitted block did not start at the spec's own value"


def test_a_gradient_reaches_it_through_the_whole_chain():
    """circuit -> readout -> muscles -> the eye's own mechanics, and back."""
    sim = _sim([{"block": "w", "of": "afferent"}, {"block": "w", "of": "recurrent"},
                {"block": "w", "of": "junction"}], n_frames=6)
    H, _ = engine.run(sim, device="cpu", progress=False, grad=True)
    (H.level("eye").get("pose") ** 2).sum().backward()
    for s in ("afferent", "recurrent", "junction"):
        g = H.level(s)._fitted["w"].grad
        assert g is not None, f"{s}.w received no gradient"
        assert torch.isfinite(g).all(), f"{s}.w gradient is not finite"
    assert float(H.level("junction")._fitted["w"].grad.abs().max()) > 0, \
        "the readout's gradient is identically zero, so nothing would train"


def test_the_same_leaf_is_reused_across_rollouts():
    """THE PROPERTY TRAINING DEPENDS ON. `engine.run` rebuilds the Hierarchy every call, so a
    parameter made inside one rollout would not survive into the next; an optimiser holding the
    first rollout's tensors would move objects no later rollout reads."""
    sim = _sim([{"block": "w", "of": "junction"}])
    H1, _ = engine.run(sim, device="cpu", progress=False, grad=True)
    p1 = H1.level("junction")._fitted["w"]
    with torch.no_grad():
        p1 += 1.0                                          # as an optimiser step would
    H2, _ = engine.run(sim, device="cpu", progress=False, grad=True)
    p2 = H2.level("junction")._fitted["w"]
    assert p2 is p1, "a second rollout made a NEW parameter; training could not accumulate"
    assert torch.allclose(H2.level("junction").get("w").detach(), p1.detach()), \
        "the updated value was not written into the state the operators read"


def test_a_forward_spec_declares_nothing_learnable():
    assert load(SPEC).learnable == [], "training must not leak into the forward description"


def test_only_the_block_form_is_accepted_and_it_must_name_real_things():
    import yaml
    raw = yaml.safe_load(open(SPEC))

    def _load(entry):
        from plexus.schema import _parse_learnable
        return _parse_learnable([entry], raw)

    with pytest.raises(ValueError, match="law substitution"):
        _load({"replaces": "neuron_signal", "with": "mlp"})
    with pytest.raises(ValueError, match="law substitution"):
        _load({"block": "w", "of": "recurrent", "replaces": "neuron_signal"})
    with pytest.raises(ValueError, match="needs `block:`"):
        _load({"with": "mlp"})
    with pytest.raises(ValueError, match="nothing reads"):
        _load({"block": "w", "of": "recurrent", "with": "siren"})
    with pytest.raises(ValueError, match="not a declared set"):
        _load({"block": "w", "of": "nonexistent"})
    assert _load({"block": "w", "of": "recurrent"})[0]["of"] == "recurrent"


def test_the_engine_refuses_an_entry_it_cannot_fit_rather_than_skipping_it():
    """THE SILENT NO-OP THIS REPLACES. A `{replaces:, with:}` entry set on `sim.learnable` directly
    -- past the schema -- used to be skipped, and the run went ahead with the original operator."""
    with pytest.raises(ValueError, match="no `block:`"):
        engine.run(_sim([{"replaces": "neuron_signal", "with": "mlp"}]), device="cpu",
                   progress=False, grad=True)

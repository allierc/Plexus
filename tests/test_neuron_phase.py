"""neuron_signal[model: phase_rotated] -- exp18's broadcast angle on the cell-type-pair phase.

Identity: at varphi = 0 and alpha = 0 the variant IS `shared`. Planted: a known pair phase and angle
give cos(varphi - alpha); a one-hot context selects its own angle; half a turn of alpha negates the
message (the J(alpha + pi) = -J(alpha) of GNN_Transformer.tex Part IV).
"""
import math

import pytest
import torch

import plexus.operators                                    # noqa: F401  self-registers `neural`
from plexus.models.registry import get_contract, get_operator

from test_neural import _circuit, _set_v, _types

ROW = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def _two_type_circuit():
    """neuron 0 (type 0) -> neuron 2 (type 1) weight 2.0; neuron 1 (type 1) -> neuron 2 weight 0.5."""
    _sim, H = _circuit([[0, 2], [1, 2]], [2.0, 0.5])
    _types(H, [0, 1, 1], [ROW, ROW])
    _set_v(H, [1.0, 1.0, 0.0])
    return H


def _phase(params):
    return get_operator("neuron_signal", model="phase_rotated")(
        {"_at": "neuron", "edge_set": "synapse", "activation": "identity", "n_types": 2, **params}, "cpu")


def test_default_of_neuron_signal_is_still_shared():
    """T-55: a variant registered beside its base must not become the name's default."""
    assert get_contract("neuron_signal").default != "phase_rotated"


def test_zero_phase_zero_angle_is_shared_exactly():
    H = _two_type_circuit()
    shared = get_operator("neuron_signal", model="shared")(
        {"_at": "neuron", "edge_set": "synapse", "activation": "identity"}, "cpu").forward(H)["neuron"]
    rot = _phase({}).forward(H)["neuron"]
    assert torch.equal(rot, shared)


def test_planted_pair_phase_gives_cos_of_phase_minus_angle():
    H = _two_type_circuit()
    op = _phase({"alpha": [0.3]})
    op.phi = torch.tensor([[0.0, 1.1], [0.0, -0.4]])      # pair (0 -> 1) at 1.1 rad, (1 -> 1) at -0.4
    got = op.forward(H)["neuron"][2].item()
    want = 2.0 * math.cos(1.1 - 0.3) + 0.5 * math.cos(-0.4 - 0.3)
    assert got == pytest.approx(want, rel=1e-6)


def test_half_a_turn_negates_the_message():
    H = _two_type_circuit()
    a, b = _phase({"alpha": [0.7]}), _phase({"alpha": [0.7 + math.pi]})
    for op in (a, b):
        op.phi = torch.tensor([[0.0, 1.1], [0.0, -0.4]])
    assert b.forward(H)["neuron"][2].item() == pytest.approx(-a.forward(H)["neuron"][2].item(), abs=1e-6)


def test_one_hot_context_selects_its_own_angle():
    sensor = {"sensor": {"parent": "brain", "per_parent": 3,
                         "state": {"signal": {"width": 1, "integration": "none", "boundary": "free"}}}}
    _sim, H = _circuit([[0, 2], [1, 2]], [2.0, 0.5], sets_extra=sensor)
    _types(H, [0, 1, 1], [ROW, ROW])
    _set_v(H, [1.0, 1.0, 0.0])
    lvl = H.level("sensor")
    a0, _ = lvl.state_schema["signal"]
    lvl.state[:, a0] = torch.tensor([5.0, 0.0, 1.0])       # line 0 = stimulus (ignored); line 2 = context 2
    op = _phase({"context_set": "sensor", "context_channels": [1, 3], "alpha": [0.0, 1.0]})
    op.phi = torch.tensor([[0.0, 0.4], [0.0, 0.4]])
    got = op.forward(H)["neuron"][2].item()
    assert got == pytest.approx(2.5 * math.cos(0.4 - 1.0), rel=1e-6)


def test_alpha_length_must_match_the_context_lines():
    with pytest.raises(ValueError, match="angles"):
        _phase({"context_set": "sensor", "context_channels": [1, 4], "alpha": [0.0, 1.0]})


def test_torus_at_zero_is_shared_exactly():
    H = _two_type_circuit()
    shared = get_operator("neuron_signal", model="shared")(
        {"_at": "neuron", "edge_set": "synapse", "activation": "identity"}, "cpu").forward(H)["neuron"]
    rot = _phase({"n_axes": 2, "alpha": [[0.0, 0.0]]}).forward(H)["neuron"]
    assert torch.equal(rot, shared)


def test_torus_factor_is_the_mean_over_axes():
    H = _two_type_circuit()
    op = _phase({"n_axes": 2, "alpha": [[0.3, -1.2]]})
    op.phi = torch.tensor([[[0.0, 1.1], [0.0, -0.4]], [[0.0, 0.5], [0.0, 2.0]]])   # [L, T, T]
    got = op.forward(H)["neuron"][2].item()
    want = (2.0 * 0.5 * (math.cos(1.1 - 0.3) + math.cos(0.5 + 1.2))
            + 0.5 * 0.5 * (math.cos(-0.4 - 0.3) + math.cos(2.0 + 1.2)))
    assert got == pytest.approx(want, rel=1e-6)


def test_torus_alpha_shape_is_checked():
    with pytest.raises(ValueError, match="axes"):
        _phase({"n_axes": 2, "alpha": [0.0]})


def test_zero_jitter_and_spread_change_nothing():
    H = _two_type_circuit()
    a = _phase({"alpha": [0.4]}).forward(H)["neuron"]
    b = _phase({"alpha": [0.4], "alpha_jitter": 0.0, "alpha_spread": 0.0}).forward(H)["neuron"]
    assert torch.equal(a, b)


def test_jitter_is_held_for_the_trial_and_moves_the_angle():
    H = _two_type_circuit()
    torch.manual_seed(0)
    op = _phase({"alpha": [0.4], "alpha_jitter": 0.3})
    op.phi = torch.tensor([[0.0, 1.1], [0.0, -0.4]])
    first = op.forward(H)["neuron"][2].item()
    assert op.forward(H)["neuron"][2].item() == first                     # same offset within the trial
    torch.manual_seed(0)
    xi = float(torch.randn(1) * 0.3)
    assert first == pytest.approx(2.0 * math.cos(1.1 - 0.4 - xi) + 0.5 * math.cos(-0.4 - 0.4 - xi), rel=1e-5)


def test_spread_offsets_the_angle_per_receiver():
    H = _two_type_circuit()
    torch.manual_seed(1)
    op = _phase({"alpha": [0.4], "alpha_spread": 0.5})
    op.phi = torch.tensor([[0.0, 1.1], [0.0, -0.4]])
    got = op.forward(H)["neuron"][2].item()
    xi2 = float(op._spr[2])                                               # neuron 2 receives both edges
    assert got == pytest.approx(2.0 * math.cos(1.1 - 0.4 - xi2) + 0.5 * math.cos(-0.4 - 0.4 - xi2), rel=1e-5)


def test_nonneg_at_zero_is_shared_and_never_reverses():
    H = _two_type_circuit()
    shared = get_operator("neuron_signal", model="shared")(
        {"_at": "neuron", "edge_set": "synapse", "activation": "identity"}, "cpu").forward(H)["neuron"]
    assert torch.equal(_phase({"nonneg": True}).forward(H)["neuron"], shared)
    op = _phase({"nonneg": True, "alpha": [0.4 + math.pi]})               # half a turn: reversal under cos
    op.phi = torch.tensor([[0.0, 0.4], [0.0, 0.4]])
    assert op.forward(H)["neuron"][2].item() == pytest.approx(0.0, abs=1e-6)   # (1 + cos pi) / 2 = 0: silenced

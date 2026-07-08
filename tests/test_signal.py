"""PR 3 smoke test: the passive `signal` operator (connectome signalling).

Builds a tiny 2-neuron / 1-synapse circuit and checks the operator computes the
first-order voltage ODE correctly and drives the postsynaptic neuron through the
engine's schema-driven (boundary-free) integration:

    tau * dv_i/dt = -v_i + b_i + sum_{e: post(e)=i} W_e * phi(v_{pre(e)}) .
"""
import torch

from plexus.schema import Spec, OpSpec, Selector
from plexus.engine import build, _integrate, _resolve_emit
from plexus.models.registry import get_operator


def _circuit():
    sets = {
        "network": {"n": 1},
        "neuron": {"parent": "network", "per_parent": 2, "state": {"voltage": 1}},
        "synapse": {"parent": "network", "edge_set": True, "pre": "neuron", "post": "neuron",
                    "edges": [[0, 1]], "weights": [2.0],
                    "state": {"w": {"width": 1, "integration": "none", "record": False}}},
    }
    ops = [OpSpec(op="signal", on=Selector("neuron"),
                  params={"edge_set": "synapse", "tau": 1.0, "activation": "relu", "bias": 0.0})]
    sim = Spec(name="sig", seed=0, n_frames=1, dt=1.0, sets=sets, fields={},
               operators=ops, schedule=["signal"])
    H = build(sim, device="cpu")
    H.emit_order = _resolve_emit(sim)
    return sim, H


def test_signal_weight_loaded_into_synapse():
    _, H = _circuit()
    assert H.level("synapse").get("w").squeeze(-1).tolist() == [2.0]


def test_signal_computes_first_order_voltage_ode():
    sim, H = _circuit()
    neuron = H.level("neuron")
    neuron.state[0, 0] = 1.0                       # v_0 = 1, v_1 = 0
    op = get_operator("signal")(
        {"edge_set": "synapse", "tau": 1.0, "activation": "relu", "bias": 0.0, "_at": "neuron"}, "cpu")
    H.zero_delta()
    dv = op(H, None)["neuron"].squeeze(-1)
    # neuron 1 receives W*phi(v_0) = 2*relu(1) = 2 -> dv_1 = (-0 + 2)/1 = 2
    # neuron 0 has no input -> dv_0 = (-1)/1 = -1
    assert torch.allclose(dv, torch.tensor([-1.0, 2.0]))


def test_signal_integrates_through_engine():
    sim, H = _circuit()
    neuron = H.level("neuron")
    neuron.state[0, 0] = 1.0
    op = get_operator("signal")(
        {"edge_set": "synapse", "tau": 1.0, "activation": "relu", "bias": 0.0, "_at": "neuron"}, "cpu")
    H.zero_delta()
    H.add_delta("neuron", op(H, None)["neuron"])
    _integrate(H, sim.dt)                          # dt = 1.0
    # v_0 = 1 + 1*(-1) = 0 ; v_1 = 0 + 1*2 = 2 ; free boundary -> no clamp
    assert torch.allclose(neuron.get("voltage").squeeze(-1), torch.tensor([0.0, 2.0]))

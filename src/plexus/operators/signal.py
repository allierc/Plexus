"""signal -- passive connectome signalling on a neuron set.

A fixed-connectome Lateral operator: each neuron integrates a first-order voltage ODE
driven by weighted, activated input from its presynaptic neighbours,

    tau * dv_i/dt = -v_i + b_i + sum_{e: post(e)=i} W_e * phi(v_{pre(e)}) .

"Passive" means the synaptic weight `W_e` is a fixed edge parameter -- no synapse state
(a stateful `synapse_ode` is a later PR). The connectome is read as a `synapse` EDGE-SET
(PR2): gather phi(v) along the `pre` incidence map, weight by the synapse `w` block, and
scatter onto the `post` neuron via `H.scatter_along` (an Aggregate along an incidence
map). First-order in voltage, so EMIT=velocity; the engine integrates the neuron's
`voltage` coordinate (PR1's schema-driven, boundary-free integration).

This is the paper's degenerate case: drop the synapse state and the edge-set collapses
to a plain weighted connectome, one Lateral operator on the neuron set.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

_ACT = {
    "relu": torch.relu,
    "tanh": torch.tanh,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}


@register_operator("signal", family="interaction", level="neuron", kind="lateral")
class Signal(Lateral):
    EMIT = "velocity"                     # first-order voltage ODE (dv/dt); engine integrates the `voltage` coordinate
    SUPPORTED_DIMS = [2, 3]               # voltage is scalar -- the operator ignores spatial dimension
    REQUIRES_PARAMS = ["tau", "edge_set"]
    MECHANISM_TAGS = ["signal_propagation", "connectome", "recurrent"]
    PARAM_ROLES = {
        "tau": "membrane_time_constant",
        "edge_set": "connectome_synapse_set",
        "activation": "presynaptic_nonlinearity",
        "bias": "resting_drive",
        "weight": "synapse_weight_block",
    }

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.tau = float(params["tau"])
        self.edge_set = params["edge_set"]
        self.act = _ACT[params.get("activation", "relu")]
        self.bias = float(params.get("bias", 0.0))
        self.weight_block = params.get("weight", "w")     # synapse state block holding W_e
        self.block = params.get("block", "voltage")       # the neuron state block to evolve
        self.at = params.get("_at", "neuron")

    def forward(self, H, mask=None):
        neuron = H.level(self.at)
        v = neuron.get(self.block)                                 # [N, 1]  membrane voltage
        es = H.level(self.edge_set)
        v_pre = H.gather(self.edge_set, "pre", self.block)         # [E, 1]  presynaptic voltage per edge (lift along `pre`)
        w = es.get(self.weight_block)                              # [E, 1]  fixed synaptic weight W_e
        edge_msg = w * self.act(v_pre)                             # [E, 1]  W_e * phi(v_pre)
        current = H.scatter_along(self.edge_set, "post", edge_msg) # [N, 1]  synaptic current onto post neuron (Aggregate along `post`)
        dv = (-v + self.bias + current) / self.tau                 # [N, 1]  first-order voltage derivative
        dv = dv * neuron.occ[:, None]                              # dormant neurons do not move
        if mask is not None:
            dv = dv * mask[:, None].float()
        return {self.at: dv}

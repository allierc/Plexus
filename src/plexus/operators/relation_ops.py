"""Relations between two kinds, and what travels along them.

    project   lateral   one set's state across a relation -- a RATE into the receiver's derivative
    readout   lateral   the same three steps, but the message IS the receiver's block

A relation is already a first-class Plexus object: a set with `edge_set: true` and two incidence
maps. What this module adds is the thing that TRAVELS along one -- gather the sender's state,
weight it by the edge, aggregate onto the receiver:

    y_i = act( gain * sum_{e : post(e) = i} w_e * send(x_pre(e)) + bias )

`neuron_signal` is these same three steps with a neuron's biology welded on, and is rightly filed
with the neurons. The steps themselves are not about neurons: they are what happens whenever one
population drives another through a weighted relation -- a sensor into a circuit (W_in), a circuit
into an effector (W_out), one circuit into another, any set into any set across any edge set.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.operators.field_ops import _ACT


class _Project(Lateral):
    """Carry one set's state across a relation onto another set's block.

    (pre, edge_set) -> post: gathers `block` along `pre`, weights each edge by the edge set's
    own weight, aggregates along `post`.

        y_i = act( gain * sum_{e : post(e) = i} w_e * send(x_pre(e)) + bias )

    x is the sender's `block`, in whatever units that block holds; w_e is the edge's weight,
    carrying the units of the map; gain is a scalar multiplier on the aggregated sum and bias a
    scalar offset, both after the sum and before `act`. Both nonlinearities are named from the
    shared activation table and both default to identity.

    TWO NONLINEARITIES, AND THEY ARE NOT INTERCHANGEABLE. `send:` is applied to the sender's
    state before the weights, once per sender; `act:` after the sum, once per receiver. Collapsing
    them would move a nonlinearity across a linear map, and sum(w tanh(x)) is not tanh(sum(w x)).

    TWO CONTRACTS, NOT ONE WITH A SWITCH, because what an operator emits IS its contract with the
    clock: `project` emits a RATE summed into the receiver's derivative (W_in), `readout` writes
    the receiver's `into:` block in place (W_out). Everything else is shared, which is why they
    are one class with two registrations.

    The two endpoints may be different SETS: each leg resolves through the edge set's own
    incidence map, where `neuron_signal` indexes one type table with both legs.

    Reference: Plexus (this work). The three-step gather/weight/scatter is the general form of
    which `neuron_signal` is the neural specialisation.
    """

    EMIT = "velocity"                  # `readout` overrides to None; see the docstring
    INPUTS = ["neuron"]                # the real endpoints come from the edge set, not from here
    OUTPUTS = ["neuron"]
    READS = ["w"]
    WRITES = []
    SUPPORTED_DIMS = [2, 3]
    DIFFERENTIABLE = True
    MAY_MUTATE_INTEGRATED_STATE = True     # `emit: none` writes the receiver's block in place
    REQUIRES_PARAMS = ["edge_set", "block"]
    WRITES_BLOCK = False               # `readout` sets True: the message IS the block
    MECHANISM_TAGS = ["projection", "relation", "input_map", "output_map", "linear_map"]
    PARAM_ROLES = {
        "edge_set": "the_relation_traversed", "block": "sender_state_block",
        "into": "receiver_state_block", "weight": "edge_weight_block",
        "gain": "scalar_on_the_aggregated_sum",
        "bias": "offset_before_activation_scalar_or_a_receiver_state_block",
        "send": "nonlinearity_the_sender_applies_before_the_weights",
        "activation": "nonlinearity_the_receiver_applies_after_the_sum",
    }
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.edge_set = params["edge_set"]
        self.block = params["block"]
        # `into:` and not `to:`: `to:` is reserved for a destination FIELD.
        self.to_block = params.get("into", self.block)
        self.weight_block = params.get("weight", "w")
        self.gain = float(params.get("gain", 1.0))
        # `bias:` is a number (a property of the MAP) or a receiver state block (a property of
        # each receiving element, as `torch.nn.Linear`'s bias vector is).
        _b = params.get("bias", 0.0)
        self.bias_block = _b if isinstance(_b, str) else None
        self.bias = 0.0 if self.bias_block else float(_b)
        self.send = _ACT[params.get("send", "identity")]          # sender-side, before the weights
        self.act = _ACT[params.get("activation", "identity")]     # receiver-side, after the sum
        if self.WRITES_BLOCK and "into" not in params:
            raise ValueError(
                "readout writes a block and must be told which: give `into:`. (`project` needs "
                "no `into:` -- its delta is integrated into the receiver's own coordinate.)")

    def forward(self, H, mask=None):
        es = H.level(self.edge_set)
        post = H.level(es.post_name)
        x_pre = self.send(H.gather(self.edge_set, "pre", self.block))   # [.., E, w]; send() first
        w_e = es.get(self.weight_block)                              # [.., E, 1]
        msg = H.scatter_along(self.edge_set, "post", w_e * x_pre)    # [.., N_post, w] along post
        b = self.bias
        if self.bias_block is not None:
            if self.bias_block not in post.state_schema:
                raise ValueError(
                    f"{self.__class__.__name__}: `bias: {self.bias_block!r}` names a state block "
                    f"the receiving set {es.post_name!r} does not have (it has: "
                    f"{', '.join(bl.name for bl in post.state_schema.blocks)}). Declare it under "
                    f"`sets.{es.post_name}.state:` -- one column, integration none -- or give "
                    f"`bias:` as a number.")
            b = post.get(self.bias_block)                            # [.., N_post, 1]
        y = self.act(self.gain * msg + b) * post.occ[:, None]
        if mask is not None:
            y = y * mask[:, None].to(y.dtype)
        if self.WRITES_BLOCK:
            # The message IS the block: written in place, cloned only under autograd.
            b0, b1 = post.state_schema[self.to_block]
            # The width is the LAST axis: a batched run carries y as [B, N, w].
            if b1 - b0 != y.shape[-1]:
                raise ValueError(
                    f"readout: the message is {y.shape[-1]} wide but {es.post_name}."
                    f"{self.to_block!r} is {b1 - b0}. The width comes from the SENDER's "
                    f"{self.block!r} block, so these must agree.")
            # A masked `readout` writes only the elements it acts on, so two readouts over
            # disjoint receivers compose instead of the second erasing the first.
            keep = y if mask is None else torch.where(
                mask[:, None].to(torch.bool), y, post.get(self.to_block))
            if torch.is_grad_enabled():
                st = post.state.clone()
                st[..., b0:b1] = keep
                post.state = st
            else:
                post.state[..., b0:b1] = keep
            return {}
        return {es.post_name: y}


@register_operator("project", family="relation", set="neuron", kind="lateral",
                   equation=r"""$$\frac{dx_i}{dt}\mathrel{+}=\mathrm{act}\Big(\text{gain}\!\!\sum_{e\,:\,\mathrm{post}(e)=i}\!\! w_e\,\mathrm{send}\big(x_{\mathrm{pre}(e)}\big)+\text{bias}\Big)$$""")
class Project(_Project):
    """A rate across a relation: the message is summed into the receiver's derivative.

    (pre, edge_set) -> post: gathers `block` along `pre`, weights by the edge, aggregates
    along `post`, emits a rate.

        dx_i/dt += act( gain * sum_{e : post(e) = i} w_e * send(x_pre(e)) + bias )

    W_in (a sensor population into a circuit) and circuit -> circuit are both this. The receiver
    INTEGRATES what arrives, so the message carries the units of the receiver's coordinate per
    unit time, and `into:` is not read -- the delta goes to the receiver's own integrated block.

    Reference: Plexus (this work); the general form of which `neuron_signal` is the neural case.
    """

    EMIT = "velocity"
    WRITES_BLOCK = False


@register_operator("readout", family="relation", set="neuron", kind="lateral",
                   equation=r"""$$x_i=\mathrm{act}\Big(\text{gain}\!\!\sum_{e\,:\,\mathrm{post}(e)=i}\!\! w_e\,\mathrm{send}\big(x_{\mathrm{pre}(e)}\big)+\text{bias}\Big)$$""")
class Readout(_Project):
    """A value across a relation: the message IS the receiver's `into:` block, written each frame.

    (pre, edge_set) -> post: the same gather, weight and aggregate, written in place.

        x_i = act( gain * sum_{e : post(e) = i} w_e * send(x_pre(e)) + bias )

    W_out (a circuit into an effector population) is this. The receiver does NOT integrate what
    arrives -- it holds it -- so the message carries the units of `into:` itself, and `into:` is
    required. Every block written this way should be `integration: none` in its own schema; a
    block that is both integrated and overwritten each frame has two laws and the last one wins.

    Reference: Plexus (this work); the general form of which `neuron_signal` is the neural case.
    """

    EMIT = None
    WRITES_BLOCK = True

"""Relations between two kinds, and what travels along them.

    project   lateral   one set's state across a relation -- a RATE into the receiver's derivative
    readout   lateral   the same three steps, but the message IS the receiver's block

A relation is already a first-class Plexus object: a set with `edge_set: true` and two incidence
maps, which is the paper's own move for structure that is not a function -- give the relation the
status of entities and both of its legs become ordinary maps. What this module adds is the thing
that TRAVELS along one: gather the sender's state, weight it by the edge, aggregate onto the
receiver.

WHY THIS IS NOT IN `neuron_ops`. `neuron_signal` is the same three steps with a neuron's biology
welded on -- a per-type transfer function, a Dale sign, a coupling gain read off the receiving
cell's row -- and it is rightly filed with the neurons. But the three steps themselves are not
about neurons at all. They are what happens whenever one population drives another through a
weighted relation, and the cases that need them are not one biology:

    a sensor population into a circuit          (the input map, W_in)
    a circuit into an effector population       (the output map, W_out)
    one circuit into another circuit            (two connectomes, composed)
    any set into any set, across any edge set

Naming it `neuron_project` and filing it with the neurons would make the general case reachable
only through a word that misdescribes it, which is how `structural` came to mean "may write in
place" and then absorbed twenty operators that were nothing of the kind.
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

    TWO NONLINEARITIES, AND THEY ARE NOT INTERCHANGEABLE. `send:` is what the SENDER emits --
    the map is applied to the sender's state before the weights, once per sender. `act:` is what
    the RECEIVER does with what arrives -- applied after the sum, once per receiver. A rate-coded
    population is the ordinary case for `send:` (a membrane voltage is not what a downstream set
    receives; a firing rate is, and `send: tanh` is that saturation), while rectification belongs
    on the receiving side (`act: softplus` on a muscle, because a muscle pulls or does nothing).
    Collapsing them into one would silently move a nonlinearity across a linear map, which is a
    different model: sum(w * tanh(x)) is not tanh(sum(w * x)).

    TWO CONTRACTS, NOT ONE WITH A SWITCH. What can happen to an arriving message is one of two
    ordinary things, and they are registered separately because what an operator emits IS its
    contract with the clock -- the engine resolves a whole SET's integration order from it, so
    hiding the choice in a parameter would let a spec change how a set moves in time without
    that showing anywhere the engine looks:

        project   the message is a RATE, summed into the receiver's derivative like any other
                  operator's delta. An input current into a circuit: what arrives changes how
                  fast the receiver's state moves. This is W_in, and circuit -> circuit.
        readout   the message IS the receiver's `into:` block, written in place each frame. An
                  effector reading its drive: a muscle's contraction is not something it
                  integrates, it is what it is being told right now. This is W_out.

    Everything else -- the gather, the weight, the sum, the gain, the bias, the activation --
    is shared, which is why they are one class with two registrations rather than two operators.

    THE TWO ENDPOINTS MAY BE DIFFERENT KINDS, and that is the whole point. `H.gather` and
    `H.scatter_along` resolve each leg through the edge set's own `incidence_name`, so nothing
    here assumes the sender and the receiver are the same set -- `neuron_signal` assumes it only
    because it indexes ONE type table with both `es.pre` and `es.post`.

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
        "gain": "scalar_on_the_aggregated_sum", "bias": "scalar_offset_before_activation",
        "send": "nonlinearity_the_sender_applies_before_the_weights",
        "activation": "nonlinearity_the_receiver_applies_after_the_sum",
    }
    REFERENCE = "Plexus (this work)."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.edge_set = params["edge_set"]
        self.block = params["block"]
        # `into:` AND NOT `to:`. `to:` is a reserved spec key naming a destination FIELD, and
        # the schema rejects it when it names a state block -- which is the right refusal, the
        # same one `cell_ops` records for `from:`. An Exchange writes a field, a readout writes a
        # BLOCK, and spelling both `to` would make the two indistinguishable in a spec.
        self.to_block = params.get("into", self.block)
        self.weight_block = params.get("weight", "w")
        self.gain = float(params.get("gain", 1.0))
        self.bias = float(params.get("bias", 0.0))
        self.send = _ACT[params.get("send", "identity")]          # sender-side, before the weights
        self.act = _ACT[params.get("activation", "identity")]     # receiver-side, after the sum
        if self.WRITES_BLOCK and "into" not in params:
            raise ValueError(
                "readout writes a block and must be told which: give `into:`. (`project` needs "
                "no `into:` -- its delta is integrated into the receiver's own coordinate.)")

    def forward(self, H, mask=None):
        es = H.level(self.edge_set)
        post = H.level(es.post_name)
        x_pre = self.send(H.gather(self.edge_set, "pre", self.block))   # [E, w] lift, then send()
        w_e = es.get(self.weight_block)                              # [E, 1]
        msg = H.scatter_along(self.edge_set, "post", w_e * x_pre)    # [N_post, w] sum along post
        y = self.act(self.gain * msg + self.bias) * post.occ[:, None]
        if mask is not None:
            y = y * mask[:, None].to(y.dtype)
        if self.WRITES_BLOCK:
            # The message IS the block. Written in place, clone-and-reassign under autograd so
            # the tape keeps the previous state alive (see `aggregate_centroid` for why the
            # forward path must not clone).
            b0, b1 = post.state_schema[self.to_block]
            if b1 - b0 != y.shape[1]:
                raise ValueError(
                    f"readout: the message is {y.shape[1]} wide but {es.post_name}."
                    f"{self.to_block!r} is {b1 - b0}. The width comes from the SENDER's "
                    f"{self.block!r} block, so these must agree.")
            if torch.is_grad_enabled():
                st = post.state.clone()
                st[:, b0:b1] = y
                post.state = st
            else:
                post.state[:, b0:b1] = y
            return {}
        return {es.post_name: y}


@register_operator("project", family="relation", set="neuron", kind="lateral")
class Project(_Project):
    """A rate across a relation: the message is summed into the receiver's derivative.

    W_in (a sensor population into a circuit) and circuit -> circuit are both this. The receiver
    integrates what arrives, so the message has the units of the receiver's coordinate per unit
    time, and `into:` is not read -- the delta goes to the receiver's own integrated block.
    """

    EMIT = "velocity"
    WRITES_BLOCK = False


@register_operator("readout", family="relation", set="neuron", kind="lateral")
class Readout(_Project):
    """A value across a relation: the message IS the receiver's `into:` block, written each frame.

    W_out (a circuit into an effector population) is this. The receiver does not integrate what
    arrives -- it holds it -- so the message has the units of `into:` itself, and `into:` is
    required. Every block written this way should be `integration: none` in its own schema; a
    block that is both integrated and overwritten each frame has two laws and the last one wins.
    """

    EMIT = None
    WRITES_BLOCK = True

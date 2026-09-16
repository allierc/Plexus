"""The wrapper that stands in for an operator, and the two shapes it comes in.

A `learnable:` entry names an operator and a family; this builds the thing the engine actually
runs in its place. The wrapper carries the REPLACED OPERATOR'S CONTRACT -- its `EMIT`,
`INTEGRAND`, `READS`, `WRITES` -- because that is what the engine dispatches on, and a
substitution that changed any of them would be a different model rather than a fitted one.

    LearnableIntrinsic     reads its own set's blocks, emits a delta of the same shape.
                           Stands in for `neuron_update`, `organ_mechanics`.
    LearnableRelational    gathers x_pre along `pre`, pairs it with x_post and the edge weight,
                           applies the net PER EDGE, aggregates along `post`.
                           Stands in for `neuron_signal`, `project`, `readout`.

THE RELATIONAL WRAPPER KEEPS THE TRAVERSAL AND LEARNS ONLY THE MESSAGE, and that is the whole
design. The connectome is not a thing to be learnt -- it was measured -- so the substitution
leaves `gather -> weight -> scatter` exactly as the operator had it and replaces only what a
message IS. What is fitted is the synaptic transfer function; what is given is the wiring. A
learnable free to ignore the edge set would satisfy every declared field of the operator and
quietly discard the structure, which is the failure `check_substitution` exists to refuse, and it
would be perverse for the wrapper to then reintroduce it.

The consequence is that the relations the wrapper walks are the ones the operator walked, by
construction: it calls the same `H.gather` and `H.scatter_along`, so `H.measured_maps()` reports
the same legs for the substitution as for the original. That is checkable, and it is checked.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from plexus.learnables import INTRINSIC, RELATIONAL, SubstitutionError, check_substitution, \
    get_learnable


class _Wrapper(nn.Module):
    """Shared: carry the operator's contract, hold the net, keep the params it was given."""

    def __init__(self, op_cls, net, params, replaces):
        super().__init__()
        self.net = net
        self.params = dict(params)
        self.replaces = replaces
        self.at = params.get("_at", getattr(op_cls, "SET", None))
        # THE CONTRACT IS THE OPERATOR'S, copied rather than re-declared. The engine reads these
        # off the instance to resolve integration order and routing, so they must be the values
        # it would have read from the operator -- a wrapper inventing its own would be a
        # different model under the same schedule token.
        for attr in ("EMIT", "INTEGRAND", "KIND", "SET", "INPUTS", "OUTPUTS", "READS", "WRITES",
                     "SUPPORTED_DIMS", "MAY_MUTATE_INTEGRATED_STATE"):
            setattr(self, attr, getattr(op_cls, attr, None))
        self.DIFFERENTIABLE = True


class LearnableIntrinsic(_Wrapper):
    """A pointwise map over the set's own blocks: delta_i = net(blocks of element i).

    Stands in for an operator that traverses no relation. The blocks are concatenated in the
    order the operator declares `READS`, so which column is which is fixed by the operator's own
    contract rather than by this file.
    """

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        cols = [lvl.get(b) for b in (self.READS or []) if b in lvl.state_schema]
        if not cols:
            raise SubstitutionError(
                f"learnable for {self.replaces!r}: none of its READS {self.READS} are blocks of "
                f"{self.at!r}, so there is nothing to read.")
        x = torch.cat(cols, dim=-1)
        y = self.net(x) * lvl.occ[:, None]
        if mask is not None:
            y = y * mask[:, None].to(y.dtype)
        return {self.at: y}


class LearnableRelational(_Wrapper):
    """A learnt message over the operator's OWN edge set, aggregated exactly as it aggregated.

        m_e = net([x_pre(e), x_post(e), w_e])          per edge
        d_i = sum_{e : post(e) = i} m_e                along `post`

    The net sees three things and the choice is deliberate. `x_pre` because a message depends on
    what the sender is doing; `x_post` because a synapse's effect can depend on the receiver's
    state, which is what `neuron_signal[type_pairwise]` claims and `[shared]` denies -- leaving it
    out would make the learnable unable to express a hypothesis the analytic library already
    holds. And `w_e` because the connectome is a GIVEN: the net weights its message by the
    measured synapse rather than being asked to rediscover which cells are connected.
    """

    def __init__(self, op_cls, net, params, replaces):
        super().__init__(op_cls, net, params, replaces)
        self.edge_set = params.get("edge_set")
        self.block = params.get("block", (self.READS or ["voltage"])[0])
        self.weight_block = params.get("weight", "w")
        if not self.edge_set:
            raise SubstitutionError(
                f"learnable for {replaces!r} is relational and needs `edge_set:` -- the same one "
                f"the operator gathers along. Without it there is no relation to walk, and the "
                f"substitution would be the pointwise net the check refuses.")

    def forward(self, H, mask=None):
        es = H.level(self.edge_set)
        post = H.level(es.post_name)
        x_pre = H.gather(self.edge_set, "pre", self.block)            # [E, w]
        x_post = H.gather(self.edge_set, "post", self.block)          # [E, w]
        w_e = es.get(self.weight_block)                               # [E, 1]
        m = self.net(torch.cat([x_pre, x_post, w_e], dim=-1))         # [E, out]
        d = H.scatter_along(self.edge_set, "post", m) * post.occ[:, None]
        if mask is not None:
            d = d * mask[:, None].to(d.dtype)
        return {es.post_name: d}


def build_substitution(op_cls, spec, measured_maps, *, op_name="?", device="cpu"):
    """`learnable:` entry -> the instance the engine runs instead of the operator.

    Refuses before constructing anything. `measured_maps` is what the operator was WATCHED to
    traverse (`H.measured_maps()[op_name]`), so the relation check is against behaviour rather
    than against a declaration -- the declaration having been tried, found unmaintained on 23 of
    ~150 operators, and removed.
    """
    fam = spec["with"]
    learn_cls = get_learnable(fam)
    reasons = check_substitution(op_cls, learn_cls, measured_maps,
                                 op_name=op_name, learn_name=fam)
    if reasons:
        raise SubstitutionError(
            f"{fam!r} cannot stand in for {op_name!r}:\n  - " + "\n  - ".join(reasons))

    p = dict(spec.get("params") or {})
    shape = learn_cls.SHAPE
    n_in = int(p.pop("n_in", 3 if shape == RELATIONAL else 1))
    n_out = int(p.pop("n_out", 1))
    net = learn_cls(n_in, n_out, **p).to(device)
    op_params = dict(spec.get("operator_params") or {})
    op_params.setdefault("_at", spec.get("at"))
    cls = LearnableRelational if shape == RELATIONAL else LearnableIntrinsic
    return cls(op_cls, net, op_params, op_name)

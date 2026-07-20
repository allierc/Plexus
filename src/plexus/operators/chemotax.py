"""chemotax -- set <- field. Nodes follow (or flee) a field's gradient.

The Keller-Segel coupling: each selected node gets `gain * grad(field)` sampled at its
position (gain<0 to flee). ONE operator for both integration routings, chosen by `emit:`:

* `emit: velocity` (default) -- a velocity the ENGINE integrates; it simply sums with any
  other velocity the node carries (e.g. attraction_repulsion). This is the old `chemotaxis`.
* `emit: mpm_acceleration` -- the SAME `gain*grad` returned as a body acceleration routed to
  the MPM substep (p2g reads `H.delta` as `a_ext`), or added into a 2nd-order set's delta.
  This is the old `chemo_force`.

The returned vector is identical either way (`gain*grad`); only the engine's interpretation
differs -- which is exactly why the two former operators are one operator plus an `emit:`
switch, and why `forward` never branches on `emit`. The gradient itself is `Field.grad_at`
(Axis B: spatial field math lives on the grid, not copied into every operator).

`channel: null` (default) sums all channels (follow any trail); `by_material` flips the sign
per phase (solids climb `+|gain|`, liquids flee `-|gain|`); `noise` adds isotropic
exploration. The field may be deposited (slime), reaction-diffused, or prescribed (a video).
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("chemotax", family="fields", set="particle", kind="exchange")
class Chemotax(Exchange):
    EMIT = "velocity"                           # default routing; override in the spec with `emit: mpm_acceleration`
    # typed signature (Plexus2 sec. 2.1): field -> set (Exchange). Reads the `from:`
    # field gradient at each node's position, writes a velocity/accel on the node.
    INPUTS = ["particle"]
    OUTPUTS = ["particle"]
    READS = ["pos"]
    WRITES = ["pos"]                            # gain*grad(field) as a velocity (or mpm_acceleration)
    MAPS = ["field"]                            # Exchange: a gather map from the `from:` field
    SUPPORTED_DIMS = [2]                         # Field.grad_at is 2D for now (N-D is a follow-up)
    REQUIRES_PARAMS = ["from"]
    MECHANISM_TAGS = ["gradient_following", "field_templated_aggregation", "field_templated_flow"]
    PARAM_ROLES = {"gain": "field_sensitivity", "noise": "exploration_noise"}
    REFERENCE = "Keller, E. F. & Segel, L. A. (1971). Model for chemotaxis. J. Theor. Biol. 30:225-234."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.field_name = params.get("from")
        self.gain = float(params.get("gain", 1.0))
        ch = params.get("channel", None)                    # None -> sum all channels (any trail)
        self.channel = None if ch is None else int(ch)
        self.by_material = bool(params.get("by_material", False))  # solids climb (+|gain|), liquids flee (-|gain|)
        self.noise = float(params.get("noise", 0.0))        # isotropic exploration noise (off by default)
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        fld = H.fields[self.field_name]
        grad = fld.grad_at(pos, self.channel, periodic=getattr(H, "periodic", False))   # [N, D]
        if self.by_material and getattr(lvl, "is_liquid", None) is not None:
            # same field, opposite pull per phase: solid climbs the filaments (+|gain|),
            # liquid is pushed into the voids (-|gain|).
            sign = torch.where(lvl.is_liquid, -1.0, 1.0).to(grad.dtype)[:, None]
            d = abs(self.gain) * sign * grad
        else:
            d = self.gain * grad
        d = d * lvl.occ[:, None]
        if self.noise > 0.0:                                # exploratory noise on the chemotactic delta
            d = d + self.noise * torch.randn(d.shape[0], d.shape[-1],
                                             generator=getattr(H, "rng", None),
                                             device=d.device) * lvl.occ[:, None]
        if mask is not None:
            d = d * mask[:, None].float()
        return {self.at: d}

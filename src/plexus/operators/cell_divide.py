"""cell_divide (agent set, structural): proliferation on a fixed buffer via occupancy.

Embryonic tissue stays confluent because cells DIVIDE to fill space (proliferation + exclusion),
not because they attract. This is the structural operator that grows an active-matter set: each
live cell divides stochastically at a per-type rate; a daughter WAKES a dormant slot (occ 0 -> 1)
right next to the mother, inheriting all of her per-node state (type, heading, speed, remodel
rate, internal state ...). Runs on a fixed buffer -- declare `buffer:` > `n` on the set to
reserve dormant slots; when the buffer is full, division stops (contact-inhibited by capacity).

Per-cell division probability this tick is `1 - exp(-rate*dt)`, with `rate` taken from a per-type
`div_rate` buffer if present (so types proliferate at different rates), else the `rate` param.
`max_occ` caps the live fraction of the buffer (homeostatic ceiling). `kind=structural`,
`EMIT=None`; mutates occ/state in place and returns {}.
"""
from __future__ import annotations

import torch

from plexus.models.base import Structural
from plexus.models.registry import register_operator


@register_operator("cell_divide", family="growth", level="cell", kind="structural")
class CellDivide(Structural):
    EMIT = None                                       # structural: wakes dormant slots, mutates occ+state in place; returns {} — no integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                              # no required params — `rate` falls back to per-type div_rate else 0
    MECHANISM_TAGS = ["proliferation", "mitosis", "growth"]
    PARAM_ROLES = {"rate": "division_rate", "max_occ": "homeostatic_ceiling"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.rate = float(params.get("rate", 0.0))        # fallback rate if no per-type div_rate
        self.offset = float(params.get("offset", 0.006))  # daughter placement jitter (world units)
        self.max_occ = float(params.get("max_occ", 0.98)) # stop when this fraction of the buffer is live

    def forward(self, H, mask=None):
        lvl = H.level(self.at); dev = lvl.state.device
        dt = float(getattr(H.config, "dt", 1.0))
        occ = lvl.occ
        live = occ > 0
        nlive = int(live.sum())
        buf = occ.shape[0]
        free = (~live).nonzero(as_tuple=True)[0]
        if nlive == 0 or free.numel() == 0 or nlive >= int(self.max_occ * buf):
            return {}

        rate = getattr(lvl, "div_rate", None)
        rate = rate if rate is not None else torch.full((buf,), self.rate, device=dev)
        p = (1.0 - torch.exp(-rate.clamp(min=0.0) * dt)) * live.float()
        draw = torch.rand(buf, generator=getattr(H, "rng", None), device=dev)
        movers = (draw < p).nonzero(as_tuple=True)[0]
        if movers.numel() == 0:
            return {}
        cap = min(movers.numel(), free.numel(), int(self.max_occ * buf) - nlive)
        if cap <= 0:
            return {}
        parents = movers[:cap]; slots = free[:cap]

        # inherit EVERY per-node buffer (state, heading, node_type, speeds, div_rate, s, ...)
        D = lvl.state.shape[1]
        lvl.state[slots] = lvl.state[parents].clone()
        for name, b in list(lvl.named_buffers()):
            if b is None or b.dim() == 0 or b.shape[0] != buf:
                continue
            b[slots] = b[parents].clone()
        # place the daughter beside the mother
        px0, px1 = lvl.state_schema["pos"]
        jitter = (torch.rand(cap, px1 - px0, generator=getattr(H, "rng", None), device=dev) - 0.5) * (2 * self.offset)
        lvl.state[slots, px0:px1] = lvl.state[parents, px0:px1] + jitter
        lvl.occ[slots] = 1.0
        if hasattr(lvl, "birth"):
            lvl.birth[slots] = 1.0
        return {}

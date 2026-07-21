"""polarity_align (was heading_align) (agent set -> agent heading): FIRST-ORDER Vicsek polar alignment.

The orientational half of the Vicsek model, realised as a HEADING-STEER (like
`flow_align`/`chemotax`) so it composes with the first-derivative embryo cell set
(`glide` + `repel` + `mpm_to_agent` confine) WITHOUT the integration-order conflict
that blocks the 2nd-derivative `alignment`/`cruise` operators (the engine forces one
integration order per set; `mpm_to_agent` confine is `velocity`-locked).

Each cell relaxes its polarity (unit heading n_i) toward the mean heading of its
radius-graph neighbours j (the edges left by `radius_graph`), so neighbouring cells
swim TOGETHER -- the polar-order term whose balance against jostling noise drives the
flocking transition (flock <-> disorder, with bands/swirls between):

    n_bar_i = mean_{j~i} n_j
    n_i  <-  renorm( n_i + gain*dt * (n_hat - (n_hat . n_i) n_i) ),   n_hat = n_bar/|n_bar|

Cells with no neighbours (or n_bar ~ 0, i.e. neighbours cancel) keep their heading.
`kind=exchange`, mutates `heading` in place and returns {} (a heading-steering op, like
`flow_align`). Schedule it alongside `flow_align` (both steer heading before the next
`glide`). `gain` is the alignment rate = the Vicsek order lever.

Refs: Vicsek, Czirok, Ben-Jacob, Cohen & Shochet, PRL 75 (1995); the heading-kinematic
(first-order) sibling of the acceleration-based `alignment` operator.
"""
from __future__ import annotations

import torch

from plexus.models.base import Exchange
from plexus.models.registry import register_operator


@register_operator("polarity_align", "heading_align", family="polarity", set="cell", kind="exchange")
class PolarityAlign(Exchange):                   # (alias `heading_align`, one migration cycle)
    EMIT = None                                 # writes `heading` in place (Vicsek steering); returns {} — not an integrable delta
    SUPPORTED_DIMS = [2, 3]
    REQUIRES_PARAMS = []                        # no required params — gain/noise optional (defaults in __init__)
    MECHANISM_TAGS = ["vicsek", "polar_alignment", "collective_motion", "flocking"]
    PARAM_ROLES = {"gain": "alignment_rate", "noise": "orientation_noise"}
    REFERENCE = "Vicsek, T. et al. (1995). Phys. Rev. Lett. 75:1226-1229."

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "agent")
        self.gain = float(params.get("gain", 1.0))
        self.noise = float(params.get("noise", 0.0))

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        h = lvl.heading; occ = lvl.occ; dev = h.device
        N, D = h.shape
        dt = float(getattr(H.config, "dt", 1.0))
        ei = getattr(lvl, "edge_index", None)
        if ei is None or ei.numel() == 0:
            return {}
        i, j = ei[0], ei[1]                                       # row0 receiver i, row1 neighbour j
        w = occ[j].to(h.dtype)                                    # mask dormant neighbours
        hbar = torch.zeros(N, D, device=dev, dtype=h.dtype).index_add_(0, i, h[j] * w[:, None])
        deg = torch.zeros(N, device=dev, dtype=h.dtype).index_add_(0, i, w)
        hbar = hbar / deg.clamp(min=1.0)[:, None]                 # mean neighbour heading [N,D]
        hmag = hbar.norm(dim=-1, keepdim=True)
        hhat = hbar / hmag.clamp(min=1e-9)
        perp = hhat - (hhat * h).sum(-1, keepdim=True) * h        # component of n_hat perp to n
        new_h = h + (self.gain * dt) * perp
        if self.noise > 0.0:                                      # Vicsek angular noise: order vs disorder
            new_h = new_h + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        new_h = new_h / new_h.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        keep = (occ > 0) & (deg > 0) & (hmag[:, 0] > 1e-7)        # keep heading where no coherent neighbour signal
        if mask is not None:
            keep = keep & (mask > 0)
        lvl.heading = torch.where(keep[:, None], new_h, h)
        return {}

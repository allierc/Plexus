"""Analytic attraction-repulsion: a smooth, per-type pairwise interaction law.

A Lateral operator over a neighbour graph. For a receiver particle i of type t,
summed over its neighbours j (the edges left by a `rewire` operator such as
`radius_graph`):

    f(r) = p1 * exp(-r^(2 p2) / 2σ²)  -  p3 * exp(-r^(2 p4) / 2σ²)      (a length)
    dpos_i = Σ_j  f(r_ij) * (pos_j - pos_i)                            (a velocity)

with p = [p1,p2,p3,p4] the per-type parameters and σ a global width. The first
Gaussian is the long-range pull, the second the short-range push; their balance
gives type-specific phases (clusters, networks, lattices). First-derivative law:
returns a velocity (set `prediction: first_derivative`).

This is message passing on `Level.edge_index` (O(E), scales to 1e4-1e5 nodes), not
a dense O(N^2) matrix. Per-type parameters come from the spec's `types:` block
(each type carries `p`), assembled by the engine into `Level.type_params`.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("attraction_repulsion", level="particle", kind="lateral")
class AttractionRepulsion(Lateral):
    PREDICTION = "first_derivative"             # emits a velocity (overdamped law)
    REQUIRES_PARAMS = ["sigma"]                 # the cutoff lives on the radius_graph rewire op
    REQUIRES_TYPE_PROPS = ["p"]                 # per-type force-law params [p1,p2,p3,p4]
    # mechanism-search metadata: the long-range Gaussian (p1,p2) is the pull, the
    # short-range Gaussian (p3,p4) the push; their balance sets the phase.
    MECHANISM_TAGS = ["long_range_attraction", "short_range_repulsion", "coarsening", "lattice_forming"]
    MORPHOLOGY_PRIOR = ["single_cluster", "multi_cluster", "lattice", "filaments"]
    PARAM_ROLES = {"sigma": "interaction_length",
                   "p": "[pull_strength, pull_range, push_strength, push_range] per type"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.sigma = float(params["sigma"])
        self.aggr = params.get("aggr", "mean")               # mean (default, matches the reference) or sum
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        N = pos.shape[0]
        ei = lvl.edge_index                                  # [2, E]: row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, 2, device=pos.device)}
        i, j = ei[0], ei[1]

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_width", 1.0))    # j - i  [E, 2]
        r2 = (d * d).sum(-1)                                  # [E]
        p = lvl.type_params[lvl.node_type[i]]                # receiver-type params [E, 4]
        s2 = 2.0 * self.sigma ** 2
        f = (p[:, 0] * torch.exp(-(r2 ** p[:, 1]) / s2)
             - p[:, 2] * torch.exp(-(r2 ** p[:, 3]) / s2))   # [E]
        f = f * occ[j]                                       # ignore dormant neighbours

        dpos = torch.zeros(N, 2, device=pos.device)
        dpos.index_add_(0, i, f[:, None] * d)                # aggregate at the receiver
        if self.aggr == "mean":                              # average over neighbours (density-independent)
            deg = torch.zeros(N, device=pos.device).index_add_(0, i, occ[j])
            dpos = dpos / deg.clamp(min=1.0)[:, None]
        dpos = dpos * occ[:, None]
        if mask is not None:
            dpos = dpos * mask[:, None].float()
        return {self.at: dpos}

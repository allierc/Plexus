"""Relation-building (rewire) operators: construct a Level's `edge_index`.

A `rewire` operator rebuilds the within-set relation E each tick so it tracks the
live configuration, then emits no delta -- lateral operators (interaction,
springs, ...) read the edges it leaves on the Level. Separating "who interacts"
(rewire) from "how they interact" (lateral) is what lets a dense pairwise law
become O(E) message passing.
"""
from __future__ import annotations

from plexus.models.base import Rewire
from plexus.models.registry import register_operator
from plexus.geometry import edges_radius_blockwise


@register_operator("radius_graph", level="particle", kind="rewire")
class RadiusGraph(Rewire):
    """Set `Level.edge_index` to all live pairs within `radius` (optionally beyond
    `min_radius`). Blockwise build -> scales to 1e4-1e5 nodes; minimum-image under
    periodic BC. Run before a pairwise lateral operator in the schedule."""
    SUPPORTED_DIMS = [2, 3]                      # pairwise distances are dimension-generic
    REQUIRES_PARAMS = ["radius"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r_max = float(params["radius"])
        self.r_min = float(params.get("min_radius", 0.0))
        self.block = int(params.get("block", 2048))
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        lvl.edge_index = edges_radius_blockwise(
            lvl.get("pos"), lvl.occ, self.r_min, self.r_max,
            periodic=getattr(H, "periodic", False),
            world_width=getattr(H, "world_size", getattr(H, "world_width", 1.0)), block=self.block,
        )
        return {}

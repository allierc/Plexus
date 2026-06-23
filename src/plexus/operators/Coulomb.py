"""coulomb -- the electrostatic inverse-square law between charged particles.

A Lateral, second-derivative operator over a neighbour graph (the Plexus port of
ParticleGraph's `PDE_E`). For a receiver particle i of charge q_i, summed over its
neighbours j (the edges left by a `rewire` op such as `radius_graph`):

    a_i = Σ_j  -q_i q_j (pos_j - pos_i) / |d_ij|^3                  (an acceleration)

i.e. Coulomb's law a = q_i q_j r / |r|^3 with the sign convention that LIKE charges
(q_i q_j > 0) repel and OPPOSITE charges attract. Per-type `charge` comes from the
spec's `types:` block; aggregation is `add` (the physical superposition of forces,
NOT a mean). Second-derivative law: returns an acceleration the engine integrates
(v += dt·a; x += dt·v). Dimension-generic (SUPPORTED_DIMS [2,3]) -- the law reads
D = pos.shape[-1], so the same operator runs in 2D and 3D.

Message passing on `Level.edge_index` (O(E)); the `radius_graph`'s `min_radius`
keeps |d| off zero so the 1/|d|^3 stays finite (optionally bounded by `clamp`).
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("Coulomb", level="particle", kind="lateral")
class Coulomb(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration (charges have inertia)
    SUPPORTED_DIMS = [2, 3]                      # dimension-generic (reads D = pos.shape[-1])
    REQUIRES_TYPE_PROPS = ["charge"]            # per-type charge q (signed scalar)
    MECHANISM_TAGS = ["electrostatics", "coulomb", "inverse_square", "long_range"]
    MORPHOLOGY_PRIOR = ["plasma", "ionic_lattice", "charge_neutral_clusters"]
    PARAM_ROLES = {"clamp": "max |acceleration| (0 = unbounded; the radius graph's "
                            "min_radius already bounds 1/r^3)"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "particle")
        self.k = float(params.get("k", 1.0))             # Coulomb constant (overall strength)
        self.clamp = float(params.get("clamp", 0.0))     # optional cap on |a| (0 = off)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        occ = lvl.occ
        N, D = pos.shape[0], pos.shape[-1]
        ei = lvl.edge_index                              # [2, E]: row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=pos.device)}
        i, j = ei[0], ei[1]
        q = lvl.charge                                   # [N] signed per-particle charge

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_size", getattr(H, "world_width", 1.0)))   # j - i  [E, D]
        r = d.norm(dim=-1).clamp(min=1e-6)               # |d| (off zero via the graph's min_radius)
        # a_i from j: -q_i q_j d / r^3  (like charges repel, opposite attract); add-aggregated
        coef = -self.k * q[i] * q[j] / (r ** 3) * occ[j]
        acc = torch.zeros(N, D, device=pos.device).index_add_(0, i, coef[:, None] * d)
        if self.clamp > 0:                               # optional stability cap on |a|
            mag = acc.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            acc = acc * (mag.clamp(max=self.clamp) / mag)
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

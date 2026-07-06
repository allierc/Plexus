"""embryo_cell_sorting_ops -- Plexus operator for **differential-adhesion cell sorting**.

A strict-Plexus reproduction of cell sorting by the **Differential Adhesion Hypothesis**
(M. Steinberg, Science 1963), in the particle form of **Zhang, Thomas, Newman et al.,
"Computer Simulations of Cell Sorting Due to Differential Adhesion" (PLoS ONE 2011)** and the
Vicsek-like adhesion ABM vendored at `papers/cell-sorting/` (Wauford, Patel, Tordoff et al.,
"Synthetic symmetry breaking and programmable multicellular structure formation"). Cells are
rigid, motile spheres of a few types; each cell-type PAIR has its own adhesion strength. When
like-like adhesion exceeds unlike adhesion, an initially mixed aggregate SORTS: cells of a type
cluster together, and the most cohesive type is engulfed by the less cohesive one -- the
mechanism that positions germ layers in early development.

In Plexus the cells are a `cell` set with types; the contact graph is `radius_graph`; and this
one **lateral** operator computes the overdamped velocity from steric repulsion (overlap) +
type-pair adhesion (attraction within a cutoff) + a weak central confinement, keyed on both
endpoints' `node_type` via an adhesion matrix `A[type_i, type_j]`. `EMIT=velocity` (overdamped).
"""
from __future__ import annotations

import math

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator


@register_operator("differential_adhesion", level="cell", kind="lateral")
class DifferentialAdhesion(Lateral):
    """Overdamped cell motion from: steric repulsion (r<σ), type-pair adhesion (σ<r<r_adh,
    strength A[type_i,type_j]), a weak central confinement (holds the aggregate together), and
    optional Brownian noise (lets cells escape local jams so sorting can proceed)."""
    SUPPORTED_DIMS = [2, 3]
    EMIT = "velocity"                                       # overdamped: v = μ · F_net
    REQUIRES_TYPE_PROPS = []
    MECHANISM_TAGS = ["differential_adhesion", "cell_sorting", "steric_repulsion",
                      "steinberg_hypothesis", "morphogenesis", "self_organisation"]
    PARAM_ROLES = {"adhesion": "type_pair_adhesion_matrix", "sigma": "cell_diameter",
                   "k_rep": "steric_stiffness"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.at = params.get("_at", "cell")
        self.sigma = float(params.get("sigma", 0.03))       # cell diameter
        self.r_adh = float(params.get("r_adh", 0.048))      # adhesion cutoff (> sigma)
        self.k_rep = float(params.get("k_rep", 60.0))       # steric stiffness
        self.mu = float(params.get("mu", 0.0006))           # mobility (v = mu * force)
        self.confine = float(params.get("confine", 0.004))  # weak pull to centre (cohesion of the whole)
        self.noise = float(params.get("noise", 0.0008))     # Brownian velocity amplitude
        adh = params.get("adhesion", [1.0])
        T = int(round(math.sqrt(len(adh))))
        self.Amat = torch.tensor(adh, dtype=torch.float32, device=device).view(T, T)

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")                                # [N, D]
        N, D = pos.shape
        dev = pos.device
        ei = getattr(lvl, "edge_index", None)
        v = torch.zeros(N, D, device=dev)
        if ei is not None and ei.numel() > 0:
            i, j = ei[0], ei[1]
            d = pos[j] - pos[i]
            r = d.norm(dim=-1).clamp(min=1e-6)
            u = d / r[:, None]
            A = self.Amat[lvl.node_type[i], lvl.node_type[j]]          # [E] type-pair adhesion
            overlap = (self.sigma - r).clamp(min=0.0)                  # >0 when cells overlap
            in_range = ((r > self.sigma) & (r < self.r_adh)).float()
            f_adh = A * ((self.r_adh - r) / (self.r_adh - self.sigma)).clamp(0, 1) * in_range
            f_mag = (f_adh - self.k_rep * overlap) * lvl.occ[j]        # + toward neighbour, - apart
            force = torch.zeros(N, D, device=dev).index_add_(0, i, f_mag[:, None] * u)
            v = self.mu * force
        # weak central confinement -> the aggregate stays one blob (so types sort, not disperse)
        c = 0.5 * H.world_size[:D]
        v = v - self.confine * (pos - c)
        if self.noise > 0:
            v = v + self.noise * torch.randn(N, D, generator=getattr(H, "rng", None), device=dev)
        return {self.at: v * lvl.occ[:, None]}

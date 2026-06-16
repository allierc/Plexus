"""Attraction / repulsion -- the simplest Lateral operators (one set, pairwise).

Two composable forces on a single set of points, the canonical starting point
(ParticleGraph's interacting particles). Each is a `Lateral` operator: it reads
the set named by its `at:` selector, computes a pairwise force over the live
(`occ > 0`) nodes within a cutoff `radius`, and returns `{set: accel}`. The engine
sums the two into the set's accel accumulator and the `integrate` builtin steps it.

  repulse : soft short-range push  (incompressibility -> points keep a spacing)
  attract : pull toward neighbours within a larger radius (cohesion -> a cluster)

Balanced, the two settle the points at a characteristic spacing -- a crystal /
blob. O(N^2) dense pairwise (fine at the small N we start with).

The operator learns which set it acts on from `_at` (the selector's set name,
injected by the engine), so it is not hard-wired to a level name.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator

EPS = 1e-6


@register_operator("repulse", level="particle", kind="lateral")
class Repulse(Lateral):
    """Soft short-range repulsion: within `radius`, a linear push that is `k` at
    contact and fades to 0 at the cutoff. Pushes overlapping points apart."""
    REQUIRES_PARAMS = ["radius", "k"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r = float(params["radius"])
        self.k = float(params["k"])
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.state[:, :2]
        occ = lvl.occ
        diff = pos[:, None, :] - pos[None, :, :]                  # i <- j  (point away from j)
        d = torch.sqrt((diff * diff).sum(2) + 1e-9)
        within = (d < self.r) & (d > EPS)
        mag = (self.k * (1.0 - d / self.r)).clamp_min(0.0) * within.float() * occ[None, :]
        f = (mag[:, :, None] * diff / d[:, :, None]).sum(1)
        if mask is not None:
            f = f * mask[:, None].float()
        return {self.at: f * occ[:, None]}


@register_operator("attract", level="particle", kind="lateral")
class Attract(Lateral):
    """Cohesion: pull each point toward the mean direction of its neighbours
    within `radius`, strength `k`. Normalised by neighbour count so it does not
    blow up with density -- a steady pull to the local centroid."""
    REQUIRES_PARAMS = ["radius", "k"]

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.r = float(params["radius"])
        self.k = float(params["k"])
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.state[:, :2]
        occ = lvl.occ
        diff = pos[None, :, :] - pos[:, None, :]                  # i -> j  (toward neighbour)
        d = torch.sqrt((diff * diff).sum(2) + 1e-9)
        within = ((d < self.r) & (d > EPS)).float() * occ[None, :]
        cnt = within.sum(1).clamp(min=1.0)[:, None]
        f = self.k * (within[:, :, None] * diff).sum(1) / cnt
        if mask is not None:
            f = f * mask[:, None].float()
        return {self.at: f * occ[:, None]}

"""alignment -- Vicsek-style neighbour velocity alignment (the NOMINAL model);
Reynolds' boids alignment rule is recovered as a SPECIAL CASE.

Each receiver i steers toward the (weighted) mean velocity of its graph neighbours
j -- the edges left by a `rewire` op such as `radius_graph`. This is the alignment
interaction at the heart of the Vicsek model of active matter and of boids:

    acc_i = a * (sum_j w_ij (vel_j - vel_i)) / (sum_j w_ij)

NOMINAL -- Vicsek (`gate: none`, `per_type: false`): every radius neighbour counts
equally (w_ij = 1). Paired with `cruise` over a `radius_graph` this is the
metric Vicsek model, whose polar order undergoes the noise-driven flocking
transition (flock <-> disorder, with bands/swirls in between).

SPECIAL CASE -- boids (`per_type: true`): fold in the set's per-type alignment
strength (the `alignment` type property), reproducing Reynolds' boids alignment
term exactly; compose with `cohesion` + `separation` for the full boids law.

OPTION -- `gate: contact`: a smoothstep contact gate weights w_ij by how close i
and j are (1 in contact, 0 at radius r), for dense/contact active matter with no
discontinuity as neighbours drift in and out of range.

Second-derivative law (an acceleration): the engine integrates vel += dt*acc,
pos += dt*vel. No 1/|d|^2 term, so it needs no `min_radius` and composes cleanly
with `cruise`, `cohesion`, `separation`, `drag`.

Refs: Vicsek, Czirok, Ben-Jacob, Cohen & Shochet, PRL 75 (1995) (active matter);
Reynolds, SIGGRAPH 1987 (boids); the active-matter Vicsek generator in The Well
(Ohana et al., NeurIPS 2024, github.com/PolymathicAI/the_well), ported in
prototype/well/well_ops.py.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("alignment", level="particle", kind="lateral")
class Alignment(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration
    SUPPORTED_DIMS = [2, 3]                     # velocity neighbour-mean is dimension-generic
    OPTIONAL_TYPE_PROPS = ["alignment"]        # read per-receiver only when `per_type: true` (boids)
    MECHANISM_TAGS = ["velocity_alignment", "collective_motion", "vicsek"]
    MORPHOLOGY_PRIOR = ["flock", "bands", "swirl", "streams"]
    PARAM_ROLES = {"a": "alignment_strength", "gate": "neighbour_weighting", "r": "contact_radius"}

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("a", params.get("scale", 5e-4)))    # alignment scale (`scale`: legacy alias)
        self.gate = str(params.get("gate", "none"))                   # "none" (Vicsek/boids) | "contact"
        self.r = float(params.get("r", 0.05))                         # contact radius (gate="contact")
        self.softness = float(params.get("softness", 0.5))            # falloff band [0,1]; 0 = hard cutoff
        self.per_type = bool(params.get("per_type", False))           # boids special case: per-type weight
        self.weight_prop = str(params.get("weight", "alignment"))     # which type property holds the weight
        self.at = params.get("_at", "particle")
        if self.gate not in ("none", "contact"):
            raise ValueError(f"alignment: gate must be 'none' or 'contact', got {self.gate!r}")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos, vel, occ = lvl.get("pos"), lvl.get("vel"), lvl.occ
        N, D = vel.shape[0], vel.shape[-1]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, D, device=vel.device)}
        i, j = ei[0], ei[1]

        if self.gate == "contact":                          # smoothstep contact gate: 1 -> 0 over [r_in, r]
            d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                              getattr(H, "world_size", getattr(H, "world_width", 1.0)))
            dist = d.norm(dim=-1)
            r_in = (1.0 - self.softness) * self.r
            t = ((self.r - dist) / max(self.r - r_in, 1e-12)).clamp(0.0, 1.0)
            gate = t * t * (3.0 - 2.0 * t)
        else:                                               # gate="none": every neighbour equal (Vicsek/boids)
            gate = torch.ones(ei.shape[1], device=vel.device)
        w = gate * occ[j]                                   # gate + mask dormant neighbours

        msg = self.a * (vel[j] - vel[i]) * w[:, None]
        if self.per_type:                                   # boids: per-receiver alignment weight
            pw = getattr(lvl, self.weight_prop)[i]
            msg = msg * pw[:, None]
        acc = torch.zeros(N, D, device=vel.device).index_add_(0, i, msg)
        deg = torch.zeros(N, device=vel.device).index_add_(0, i, w)
        acc = (acc / deg.clamp(min=1.0)[:, None]) * occ[:, None]       # (weighted) mean over neighbours
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}

"""Contact neighbour alignment -- a second-derivative Lateral operator (gated Vicsek / boids rule 2).

This is the alignment term of the classic boids law, isolated as its own
one-concern operator and *gated by contact*: a receiver particle i only matches
heading with the neighbours j it is (almost) touching. The contact gate is a
smooth (smoothstep) falloff so the force has no discontinuity as particles drift
in and out of contact:

    w_ij  = smoothstep over |pos_j - pos_i| from 1 (full contact) to 0 at radius r
    acc_i = a * (sum_j w_ij (vel_j - vel_i)) / (sum_j w_ij)

with alignment scale `a` (default 5e-4, the canonical PDE_B alignment constant),
contact radius `r` (default 0.05, world units; the gate reaches exactly 0 here),
and `softness` in [0, 1] (default 0.5) giving the falloff band width: the weight
is 1 for |d| <= (1 - softness)*r and smoothstep-interpolated to 0 at |d| = r.
softness = 0 recovers a hard contact cutoff. All overridable via the spec
`params:` block.

The graph edges left by a `rewire` operator such as `radius_graph` set the
*candidate* neighbourhood; the contact gate then weights the near-touching
subset, so set the graph radius >= r. The mean is taken over contacting
neighbours (weighted by the gate) -> a particle with no one in contact gets zero
alignment.

It is a second-derivative law -> it returns an acceleration (set
`prediction: second_derivative`); the engine then integrates vel += dt*acc,
pos += dt*vel. Because it has no 1/|d|^2 term it needs no `min_radius` on the
graph, and it composes cleanly with `attraction_repulsion`, `drag`, etc.
"""
from __future__ import annotations

import torch

from plexus.models.base import Lateral
from plexus.models.registry import register_operator
from plexus.geometry import minimum_image


@register_operator("alignment", level="particle", kind="lateral")
class Alignment(Lateral):
    PREDICTION = "second_derivative"            # emits an acceleration

    def __init__(self, params, device="cpu"):
        super().__init__(params, device)
        self.a = float(params.get("a", 5e-4))         # alignment scale
        self.r = float(params.get("r", 0.05))         # contact radius (world units)
        self.softness = float(params.get("softness", 0.5))  # falloff band, [0,1]; 0 = hard cutoff
        self.at = params.get("_at", "particle")

    def forward(self, H, mask=None):
        lvl = H.level(self.at)
        pos = lvl.get("pos")
        vel = lvl.get("vel")
        occ = lvl.occ
        N = vel.shape[0]
        ei = lvl.edge_index                                 # row0 = receiver i, row1 = neighbour j
        if ei.numel() == 0:
            return {self.at: torch.zeros(N, 2, device=vel.device)}
        i, j = ei[0], ei[1]

        d = minimum_image(pos[j] - pos[i], getattr(H, "periodic", False),
                          getattr(H, "world_width", 1.0))   # j - i  [E, 2]
        dist = d.norm(dim=-1)                                # |d|  [E]

        # smoothstep contact gate: 1 for dist <= r_in, 0 at dist >= r, Hermite in between.
        r_in = (1.0 - self.softness) * self.r
        denom = max(self.r - r_in, 1e-12)                    # softness=0 -> denom~0 -> hard step
        t = ((self.r - dist) / denom).clamp(0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)                          # smoothstep weight  [E]
        contact = w * occ[j]                                 # gate & mask empty neighbour slots

        dv = vel[j] - vel[i]                                 # match neighbour heading  [E, 2]
        msg = self.a * dv * contact[:, None]

        acc = torch.zeros(N, 2, device=vel.device)
        acc.index_add_(0, i, msg)
        deg = torch.zeros(N, device=vel.device).index_add_(0, i, contact)
        acc = acc / deg.clamp(min=1.0)[:, None]              # weighted mean over contacting neighbours
        acc = acc * occ[:, None]
        if mask is not None:
            acc = acc * mask[:, None].float()
        return {self.at: acc}
